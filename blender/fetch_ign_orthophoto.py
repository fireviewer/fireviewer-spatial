"""Plan and fetch an IGN BD ORTHO mosaic for a Lambert-93 Blender terrain.

The command is deliberately explicit: either ``--dry-run`` estimates the
request without network traffic, or ``--execute`` downloads and validates all
WMS tiles.  The produced GeoTIFF keeps the requested Lambert-93 bounds exactly
and a sibling ``.source.json`` records provenance and SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window, bounds as window_bounds

IGN_WMS_URL = "https://data.geopf.fr/wms-r/wms"
IGN_WMS_CAPABILITIES_URL = (
    "https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities"
)
IGN_METADATA_URL = (
    "https://data.geopf.fr/csw?REQUEST=GetRecordById&SERVICE=CSW&VERSION=2.0.2"
    "&OUTPUTSCHEMA=http://standards.iso.org/iso/19115/-3/mdb/2.0"
    "&elementSetName=full&ID=IGNF_BD-ORTHO"
)
IGN_ACQUISITION_DATES_URL = (
    "https://data.geopf.fr/annexes/ressources/fiches/"
    "photographies-aeriennes-RVB/"
    "geoportail_dates_des_prises_de_vues_aeriennes-RVB.pdf"
)
IGN_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"
OUTPUT_CRS = "EPSG:2154"
GENERIC_EXAMPLE_BOUNDS = (700000.0, 6600000.0, 711000.0, 6613000.0)
GLOBAL_JPEG_BRIGHTNESS = 0.78
GLOBAL_JPEG_CONTRAST = 1.08
GLOBAL_JPEG_SATURATION = 1.12
GLOBAL_JPEG_PRESET = "fireviewer_global_terrain_v1"


@dataclass(frozen=True)
class JpegDisplayTransform:
    """Explicit sRGB display correction applied only to the derived JPEG.

    Multipliers are neutral at ``1.0`` for backwards compatibility.  The
    transformation order is brightness, Rec.709-luminance saturation,
    contrast around 0.5, then clipping and uint8 quantisation.
    """

    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0

    def validate(self) -> None:
        for name, value in (
            ("brightness", self.brightness),
            ("contrast", self.contrast),
            ("saturation", self.saturation),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 4.0:
                raise ValueError(f"jpeg {name} must be finite and between 0 and 4")

    @property
    def is_neutral(self) -> bool:
        return self.brightness == self.contrast == self.saturation == 1.0

    def metadata(self) -> dict[str, object]:
        self.validate()
        if (
            self.brightness == GLOBAL_JPEG_BRIGHTNESS
            and self.contrast == GLOBAL_JPEG_CONTRAST
            and self.saturation == GLOBAL_JPEG_SATURATION
        ):
            preset = GLOBAL_JPEG_PRESET
        elif self.is_neutral:
            preset = "neutral"
        else:
            preset = "custom"
        return {
            "scope": "derived_blender_rgb_jpeg_only",
            "preset": preset,
            "raw_geotiff_modified": False,
            "input_encoding": "sRGB_uint8",
            "output_encoding": "sRGB_uint8",
            "brightness_multiplier": self.brightness,
            "contrast_multiplier": self.contrast,
            "contrast_pivot": 0.5,
            "saturation_multiplier": self.saturation,
            "saturation_luminance_coefficients_rec709": [0.2126, 0.7152, 0.0722],
            "operation_order": [
                "brightness_multiply",
                "rec709_luminance_saturation",
                "contrast_around_0_5",
                "clip_0_1",
                "round_uint8",
            ],
            "neutral": self.is_neutral,
        }


@dataclass(frozen=True)
class OrthophotoTile:
    """One WMS request and its destination window in the final mosaic."""

    row: int
    column: int
    window: Window
    bounds_l93: tuple[float, float, float, float]
    url: str

    @property
    def width(self) -> int:
        return int(self.window.width)

    @property
    def height(self) -> int:
        return int(self.window.height)


@dataclass(frozen=True)
class OrthophotoPlan:
    """Fully resolved output grid and WMS tile requests."""

    bounds_l93: tuple[float, float, float, float]
    nominal_resolution_m: float
    width: int
    height: int
    transform: Affine
    tile_pixels: int
    tiles: tuple[OrthophotoTile, ...]

    @property
    def effective_resolution_m(self) -> tuple[float, float]:
        return (float(self.transform.a), float(abs(self.transform.e)))

    @property
    def estimated_network_bytes(self) -> int:
        # IGN image/geotiff responses are RGB uint8 scanline TIFFs.  The small
        # fixed overhead makes this a conservative estimate for planning.
        return self.width * self.height * 3 + len(self.tiles) * 4096

    @property
    def uncompressed_rgb_bytes(self) -> int:
        return self.width * self.height * 3


def _finite_bounds(bounds: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in bounds)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("bounds must contain four finite Lambert-93 coordinates")
    min_x, min_y, max_x, max_y = values
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("bounds must satisfy min_x < max_x and min_y < max_y")
    return min_x, min_y, max_x, max_y


def _wms_url(
    bounds_l93: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    service_url: str = IGN_WMS_URL,
    layer: str = IGN_LAYER,
) -> str:
    parameters = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": OUTPUT_CRS,
        "BBOX": ",".join(f"{value:.9f}" for value in bounds_l93),
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/geotiff",
    }
    return service_url + "?" + urlencode(parameters)


def build_plan(
    bounds_l93: Iterable[float],
    resolution_m: float = 2.0,
    tile_pixels: int = 4000,
    *,
    service_url: str = IGN_WMS_URL,
    layer: str = IGN_LAYER,
) -> OrthophotoPlan:
    """Build a pixel-aligned plan without touching the network."""

    bounds = _finite_bounds(bounds_l93)
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be a positive finite number")
    if tile_pixels < 256 or tile_pixels > 5010:
        raise ValueError("tile_pixels must be between 256 and 5010")

    min_x, min_y, max_x, max_y = bounds
    width = math.ceil((max_x - min_x) / resolution_m)
    height = math.ceil((max_y - min_y) / resolution_m)
    transform = rasterio.transform.from_bounds(*bounds, width, height)
    tiles: list[OrthophotoTile] = []
    row_count = math.ceil(height / tile_pixels)
    column_count = math.ceil(width / tile_pixels)
    for row in range(row_count):
        row_off = row * tile_pixels
        tile_height = min(tile_pixels, height - row_off)
        for column in range(column_count):
            column_off = column * tile_pixels
            tile_width = min(tile_pixels, width - column_off)
            window = Window(column_off, row_off, tile_width, tile_height)
            tile_bounds = tuple(
                float(value) for value in window_bounds(window, transform)
            )
            tiles.append(
                OrthophotoTile(
                    row=row,
                    column=column,
                    window=window,
                    bounds_l93=tile_bounds,
                    url=_wms_url(
                        tile_bounds,
                        tile_width,
                        tile_height,
                        service_url=service_url,
                        layer=layer,
                    ),
                )
            )
    return OrthophotoPlan(
        bounds_l93=bounds,
        nominal_resolution_m=float(resolution_m),
        width=width,
        height=height,
        transform=transform,
        tile_pixels=tile_pixels,
        tiles=tuple(tiles),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_lambert93(crs: CRS | None) -> bool:
    """Recognise EPSG:2154, including the WMS server's incomplete datum WKT.

    The current IGN GeoTIFF response labels the projection ``EPSG:2154`` but
    omits the RGF93 datum authority and substitutes WGS84 in its generated WKT.
    Rasterio therefore cannot always recover the EPSG code.  The seven Lambert
    parameters still identify Lambert-93 unambiguously for this controlled AOI.
    """

    if crs is None:
        return False
    if crs.to_epsg() == 2154:
        return True
    parameters = crs.to_dict()
    expected = {
        "lat_0": 46.5,
        "lon_0": 3.0,
        "lat_1": 49.0,
        "lat_2": 44.0,
        "x_0": 700000.0,
        "y_0": 6600000.0,
    }
    return (
        parameters.get("proj") == "lcc"
        and parameters.get("units") == "m"
        and all(
            math.isclose(float(parameters.get(name, math.nan)), value, abs_tol=1e-9)
            for name, value in expected.items()
        )
    )


def _download(url: str, destination: Path, timeout_s: float, retries: int = 3) -> None:
    request = Request(url, headers={"User-Agent": "FireViewer-orthophoto/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with (
                urlopen(request, timeout=timeout_s) as response,
                destination.open("wb") as out,
            ):
                content_type = response.headers.get_content_type()
                if content_type not in {
                    "image/geotiff",
                    "image/tiff",
                    "application/octet-stream",
                }:
                    excerpt = response.read(512).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"unexpected WMS content type {content_type!r}: {excerpt!r}"
                    )
                shutil.copyfileobj(response, out, length=1024 * 1024)
            return
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"IGN WMS download failed after {retries} attempts: {url}"
    ) from last_error


def validate_wms_tile(path: Path, tile: OrthophotoTile) -> dict[str, object]:
    """Validate dimensions, RGB type, CRS, bounds and usable data."""

    with rasterio.open(path) as dataset:
        if dataset.width != tile.width or dataset.height != tile.height:
            raise RuntimeError(
                "IGN WMS tile dimensions do not match the request: "
                f"got {dataset.width}x{dataset.height}, expected {tile.width}x{tile.height}"
            )
        if dataset.count < 3 or any(dtype != "uint8" for dtype in dataset.dtypes[:3]):
            raise RuntimeError("IGN WMS tile is not an RGB uint8 raster")
        if not _is_lambert93(dataset.crs):
            raise RuntimeError(
                f"IGN WMS tile CRS is {dataset.crs!s}, expected EPSG:2154"
            )
        actual_bounds = tuple(float(value) for value in dataset.bounds)
        tolerance = max(dataset.res) * 0.02 + 1e-6
        if any(
            abs(actual - expected) > tolerance
            for actual, expected in zip(actual_bounds, tile.bounds_l93, strict=True)
        ):
            raise RuntimeError(
                f"IGN WMS tile bounds {actual_bounds!r} do not match {tile.bounds_l93!r}"
            )
        valid_mask = dataset.dataset_mask()
        valid_pixels = int(np.count_nonzero(valid_mask))
        if valid_pixels == 0:
            raise RuntimeError("IGN WMS tile contains no valid pixel")
        return {
            "driver": dataset.driver,
            "crs": OUTPUT_CRS,
            "source_crs_wkt": dataset.crs.to_wkt() if dataset.crs else None,
            "crs_validation": "Lambert-93 projection parameters",
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "dtype": dataset.dtypes[0],
            "bounds_l93_m": list(actual_bounds),
            "valid_pixel_count": valid_pixels,
            "valid_pixel_ratio": valid_pixels / (dataset.width * dataset.height),
        }


def _world_file_lines(
    transform: Affine,
) -> tuple[float, float, float, float, float, float]:
    """Return the six values of a world file using pixel-centre coordinates."""

    return (
        float(transform.a),
        float(transform.d),
        float(transform.b),
        float(transform.e),
        float(transform.c + transform.a / 2 + transform.b / 2),
        float(transform.f + transform.d / 2 + transform.e / 2),
    )


def write_world_file(path: Path, transform: Affine) -> None:
    values = _world_file_lines(transform)
    path.write_text(
        "\n".join(f"{value:.12f}" for value in values) + "\n", encoding="ascii"
    )


def _validate_output(path: Path, plan: OrthophotoPlan) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        if (dataset.width, dataset.height, dataset.count) != (
            plan.width,
            plan.height,
            3,
        ):
            raise RuntimeError(
                "assembled orthophoto dimensions or RGB band count are invalid"
            )
        if dataset.crs is None or dataset.crs.to_epsg() != 2154:
            raise RuntimeError("assembled orthophoto is not in EPSG:2154")
        actual_bounds = tuple(float(value) for value in dataset.bounds)
        if any(
            abs(actual - expected) > 1e-5
            for actual, expected in zip(actual_bounds, plan.bounds_l93, strict=True)
        ):
            raise RuntimeError(
                "assembled orthophoto does not preserve the requested bounds"
            )
        return {
            "driver": dataset.driver,
            "crs": OUTPUT_CRS,
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "dtype": dataset.dtypes[0],
            "bounds_l93_m": list(actual_bounds),
            "pixel_size_m": [float(dataset.res[0]), float(dataset.res[1])],
            "compression": dataset.compression.name if dataset.compression else None,
            "overview_factors": dataset.overviews(1),
        }


def _output_profile(plan: OrthophotoPlan, jpeg_quality: int) -> dict[str, object]:
    return {
        "driver": "GTiff",
        "width": plan.width,
        "height": plan.height,
        "count": 3,
        "dtype": "uint8",
        "crs": OUTPUT_CRS,
        "transform": plan.transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "JPEG",
        "photometric": "YCBCR",
        "jpeg_quality": jpeg_quality,
        "BIGTIFF": "IF_SAFER",
    }


def apply_jpeg_display_transform(
    rgb: np.ndarray,
    transform: JpegDisplayTransform | None = None,
) -> np.ndarray:
    """Return display-adjusted RGB uint8 without mutating the source array."""

    active = transform or JpegDisplayTransform()
    active.validate()
    values = np.asarray(rgb)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(
            "JPEG display input must have shape (3, height, width) and uint8 dtype"
        )
    if active.is_neutral:
        return values.copy()

    display = values.astype("float32") / 255.0
    display *= active.brightness
    luminance = display[0:1] * 0.2126 + display[1:2] * 0.7152 + display[2:3] * 0.0722
    display = luminance + active.saturation * (display - luminance)
    display = 0.5 + active.contrast * (display - 0.5)
    return np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype("uint8")


def _write_display_jpeg(
    source_geotiff: Path,
    destination: Path,
    *,
    jpeg_quality: int,
    display_transform: JpegDisplayTransform,
) -> None:
    """Stream a derived display JPEG while leaving the GeoTIFF untouched."""

    display_transform.validate()
    with rasterio.open(source_geotiff) as source:
        profile = {
            "driver": "JPEG",
            "width": source.width,
            "height": source.height,
            "count": 3,
            "dtype": "uint8",
            "quality": str(jpeg_quality),
        }
        # The explicit .jgw written below carries georeferencing.  Disabling
        # GDAL PAM prevents an untracked .aux.xml sidecar for the plain JPEG.
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                with rasterio.open(destination, "w", **profile) as target:
                    target.colorinterp = (
                        ColorInterp.red,
                        ColorInterp.green,
                        ColorInterp.blue,
                    )
                    for _, window in source.block_windows(1):
                        target.write(
                            apply_jpeg_display_transform(
                                source.read((1, 2, 3), window=window),
                                display_transform,
                            ),
                            window=window,
                        )


def execute_plan(
    plan: OrthophotoPlan,
    output: Path,
    *,
    jpeg_output: Path | None = None,
    jpeg_quality: int = 90,
    jpeg_brightness: float = GLOBAL_JPEG_BRIGHTNESS,
    jpeg_contrast: float = GLOBAL_JPEG_CONTRAST,
    jpeg_saturation: float = GLOBAL_JPEG_SATURATION,
    timeout_s: float = 180.0,
    overwrite: bool = False,
    fetcher: Callable[[str, Path, float], None] = _download,
) -> dict[str, object]:
    """Download, validate and assemble all planned WMS tiles."""

    if not 50 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 50 and 100")
    display_transform = JpegDisplayTransform(
        brightness=jpeg_brightness,
        contrast=jpeg_contrast,
        saturation=jpeg_saturation,
    )
    display_transform.validate()
    source_record = output.with_suffix(".source.json")
    generated_paths = [output, source_record]
    if jpeg_output is not None:
        generated_paths.extend([jpeg_output, jpeg_output.with_suffix(".jgw")])
    existing = [path for path in generated_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if jpeg_output is not None:
        jpeg_output.parent.mkdir(parents=True, exist_ok=True)

    tile_records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="fireviewer-ign-ortho-") as temporary:
        temporary_dir = Path(temporary)
        with rasterio.open(
            output, "w", **_output_profile(plan, jpeg_quality)
        ) as mosaic:
            mosaic.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            for index, tile in enumerate(plan.tiles):
                tile_path = temporary_dir / f"tile-{tile.row:02d}-{tile.column:02d}.tif"
                fetcher(tile.url, tile_path, timeout_s)
                validation = validate_wms_tile(tile_path, tile)
                with rasterio.open(tile_path) as source:
                    mosaic.write(source.read((1, 2, 3)), window=tile.window)
                tile_records.append(
                    {
                        "index": index,
                        "row": tile.row,
                        "column": tile.column,
                        "window": [
                            int(tile.window.col_off),
                            int(tile.window.row_off),
                            tile.width,
                            tile.height,
                        ],
                        "bounds_l93_m": list(tile.bounds_l93),
                        "request_url": tile.url,
                        "downloaded_bytes": tile_path.stat().st_size,
                        "sha256": sha256_file(tile_path),
                        "validation": validation,
                    }
                )
            factors = [
                factor
                for factor in (2, 4, 8, 16, 32)
                if min(plan.width, plan.height) // factor >= 128
            ]
            if factors:
                mosaic.build_overviews(factors, Resampling.average)
                mosaic.update_tags(ns="rio_overview", resampling="average")

    output_validation = _validate_output(output, plan)
    outputs: list[dict[str, object]] = [
        {
            "role": "lambert93_geotiff_blender_texture",
            "file_name": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "display_transform": "none_source_mosaic_pixels_unchanged",
            "validation": output_validation,
        }
    ]
    if jpeg_output is not None:
        _write_display_jpeg(
            output,
            jpeg_output,
            jpeg_quality=jpeg_quality,
            display_transform=display_transform,
        )
        world_file = jpeg_output.with_suffix(".jgw")
        write_world_file(world_file, plan.transform)
        outputs.append(
            {
                "role": "blender_rgb_jpeg",
                "file_name": jpeg_output.name,
                "bytes": jpeg_output.stat().st_size,
                "sha256": sha256_file(jpeg_output),
                "world_file_name": world_file.name,
                "world_file_sha256": sha256_file(world_file),
                "display_transform": display_transform.metadata(),
            }
        )

    record: dict[str, object] = {
        "schema": "fireviewer.ign-orthophoto-source.v1",
        "status": "downloaded_and_validated",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "IGN / Géoplateforme",
        "product": "BD ORTHO",
        "layer": IGN_LAYER,
        "service": {
            "protocol": "WMS 1.3.0",
            "getmap_url": IGN_WMS_URL,
            "capabilities_url": IGN_WMS_CAPABILITIES_URL,
            "metadata_url": IGN_METADATA_URL,
            "acquisition_dates_url": IGN_ACQUISITION_DATES_URL,
        },
        "license": {
            "name": "Licence Ouverte / Open License",
            "reference": "https://www.etalab.gouv.fr/licence-ouverte-open-licence/",
        },
        "aoi_source_note": {
            "department": "26 Drôme",
            "acquisition_year": 2023,
            "native_resolution_m": 0.2,
            "reference_checked_date": "2026-05-19",
            "reference": IGN_ACQUISITION_DATES_URL,
        },
        "request": plan_to_dict(plan),
        "jpeg_display_transform": display_transform.metadata(),
        "tiles": tile_records,
        "outputs": outputs,
    }
    source_record.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def plan_to_dict(plan: OrthophotoPlan) -> dict[str, object]:
    return {
        "bounds_l93_m": list(plan.bounds_l93),
        "crs": OUTPUT_CRS,
        "nominal_resolution_m": plan.nominal_resolution_m,
        "effective_pixel_size_m": list(plan.effective_resolution_m),
        "width": plan.width,
        "height": plan.height,
        "pixel_count": plan.width * plan.height,
        "tile_pixels": plan.tile_pixels,
        "tile_count": len(plan.tiles),
        "estimated_network_bytes": plan.estimated_network_bytes,
        "estimated_network_mib": plan.estimated_network_bytes / (1024 * 1024),
        "uncompressed_rgb_bytes": plan.uncompressed_rgb_bytes,
        "uncompressed_rgb_mib": plan.uncompressed_rgb_bytes / (1024 * 1024),
        "tiles": [
            {
                "row": tile.row,
                "column": tile.column,
                "width": tile.width,
                "height": tile.height,
                "bounds_l93_m": list(tile.bounds_l93),
                "request_url": tile.url,
            }
            for tile in plan.tiles
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
        default=GENERIC_EXAMPLE_BOUNDS,
        help="Lambert-93 bounds; defaults to a synthetic example AOI",
    )
    parser.add_argument("--resolution-m", type=float, default=2.0)
    parser.add_argument("--tile-pixels", type=int, default=4000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jpeg-output", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--jpeg-brightness",
        type=float,
        default=GLOBAL_JPEG_BRIGHTNESS,
        help=(
            "derived JPEG brightness multiplier; global production default is "
            f"{GLOBAL_JPEG_BRIGHTNESS}, 1.0 is explicit neutral"
        ),
    )
    parser.add_argument(
        "--jpeg-contrast",
        type=float,
        default=GLOBAL_JPEG_CONTRAST,
        help=(
            "derived JPEG contrast multiplier around 0.5; global production "
            f"default is {GLOBAL_JPEG_CONTRAST}, 1.0 is explicit neutral"
        ),
    )
    parser.add_argument(
        "--jpeg-saturation",
        type=float,
        default=GLOBAL_JPEG_SATURATION,
        help=(
            "derived JPEG Rec.709 saturation multiplier; global production "
            f"default is {GLOBAL_JPEG_SATURATION}, 1.0 is explicit neutral"
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-estimated-download-mib", type=float, default=256.0)
    parser.add_argument("--allow-large-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_plan(args.bounds, args.resolution_m, args.tile_pixels)
    display_transform = JpegDisplayTransform(
        brightness=args.jpeg_brightness,
        contrast=args.jpeg_contrast,
        saturation=args.jpeg_saturation,
    )
    display_transform.validate()
    plan_record = {
        "schema": "fireviewer.ign-orthophoto-plan.v1",
        "provider": "IGN / Géoplateforme",
        "product": "BD ORTHO",
        "layer": IGN_LAYER,
        "mode": "dry-run" if args.dry_run else "execute",
        "plan": plan_to_dict(plan),
        "jpeg_display_transform": display_transform.metadata(),
    }
    print(json.dumps(plan_record, ensure_ascii=False, indent=2))
    estimated_mib = plan.estimated_network_bytes / (1024 * 1024)
    if (
        estimated_mib > args.max_estimated_download_mib
        and not args.allow_large_download
    ):
        raise RuntimeError(
            f"estimated WMS download is {estimated_mib:.1f} MiB, above the "
            f"{args.max_estimated_download_mib:.1f} MiB safety limit; use a coarser "
            "resolution or --allow-large-download after reviewing the plan"
        )
    if args.dry_run:
        return 0
    if args.output is None:
        raise ValueError("--output is required with --execute")
    execute_plan(
        plan,
        args.output.resolve(),
        jpeg_output=args.jpeg_output.resolve() if args.jpeg_output else None,
        jpeg_quality=args.jpeg_quality,
        jpeg_brightness=args.jpeg_brightness,
        jpeg_contrast=args.jpeg_contrast,
        jpeg_saturation=args.jpeg_saturation,
        timeout_s=args.timeout_s,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
