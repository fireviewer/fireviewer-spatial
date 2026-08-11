"""Acquire one deterministic 2 m MNT/MNS source pair for adaptive terrain.

The module deliberately stops at acquisition and raster validation.  It does
not build terrain meshes, request orthophotos, or expose a variable-resolution
mode.  A caller supplies one square Lambert-93 core and receives two 2 m
GeoTIFFs covering that core plus the mandatory 10 m processing halo.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Literal, Mapping, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from affine import Affine


CRS = "EPSG:2154"
RESOLUTION_M = 2.0
HALO_M = 10.0
RESAMPLING = "BILINEAR"
SCHEMA = "fireviewer.terrain-source-pair.v1"
CANONICAL_SCHEMA = "fireviewer.terrain-source-canonical-grid.v1"
TILE_SIZE_M = 500.0
CORE_VERTEX_COUNT = 251
NORMAL_HALO_VERTEX_COUNT = 253
USER_AGENT = "FireViewer-terrain-source/1.0"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class Response(Protocol):
    """Small subset shared by urllib responses and the synthetic test double."""

    headers: Mapping[str, str]
    status: int | None

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> Response: ...

    def __exit__(self, *args: object) -> None: ...


Opener = Callable[..., Response]
SourceRole = Literal["mnt", "mns"]


@dataclass(frozen=True)
class SourceGrid:
    """The globally aligned raster grid shared by the MNT and MNS requests."""

    core_bounds_l93_m: tuple[float, float, float, float]
    request_bounds_l93_m: tuple[float, float, float, float]
    width: int
    height: int
    transform: Affine
    resolution_m: float = RESOLUTION_M
    halo_m: float = HALO_M
    crs: str = CRS


@dataclass(frozen=True)
class RasterRequest:
    """One fully resolved and immutable WMS GetMap request."""

    role: SourceRole
    layer: str
    url: str
    grid: SourceGrid
    resampling: str = RESAMPLING


@dataclass(frozen=True)
class SourcePairPlan:
    """Co-registered MNT/MNS requests for one source band."""

    grid: SourceGrid
    mnt: RasterRequest
    mns: RasterRequest


@dataclass(frozen=True)
class DownloadReceipt:
    """Content identity recorded after a complete or resumed transfer."""

    byte_count: int
    sha256: str
    resumed_from_byte: int


@dataclass(frozen=True)
class CanonicalRaster:
    """Millimetre-quantized source vertices retained after raw-raster cleanup.

    ``normal_halo_mm`` is ordered south-to-north, west-to-east and covers
    ``[-2 m, 502 m]`` around the exact 500 m tile.  Its inner ``[1:-1]`` view
    is therefore the canonical 251 x 251 terrain grid.  The one-vertex halo is
    sufficient for deterministic centred gradients at every core vertex.
    """

    core_mm: np.ndarray
    normal_halo_mm: np.ndarray
    working_grid_sha256: str
    core_sha256: str
    normal_halo_sha256: str


def _finite_bounds(
    bounds: tuple[float, float, float, float] | list[float],
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in bounds)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("bounds must contain four finite Lambert-93 coordinates")
    min_x, min_y, max_x, max_y = values
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("bounds must satisfy min_x < max_x and min_y < max_y")
    return min_x, min_y, max_x, max_y


def _is_grid_aligned(value: float) -> bool:
    quotient = value / RESOLUTION_M
    return math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=1e-9)


def build_grid(
    core_bounds_l93_m: tuple[float, float, float, float] | list[float],
) -> SourceGrid:
    """Resolve the only supported grid: square, 2 m aligned, with 10 m halo."""

    bounds = _finite_bounds(core_bounds_l93_m)
    min_x, min_y, max_x, max_y = bounds
    side_x = max_x - min_x
    side_y = max_y - min_y
    if not math.isclose(side_x, side_y, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("terrain source core BBOX must be square")
    if not all(_is_grid_aligned(value) for value in bounds):
        raise ValueError("terrain source core BBOX must align to the global 2 m grid")
    if not _is_grid_aligned(side_x):
        raise ValueError("terrain source core side must be an exact multiple of 2 m")
    if not math.isclose(side_x, TILE_SIZE_M, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("terrain source core BBOX must describe one exact 500 m tile")

    request_bounds = (
        min_x - HALO_M,
        min_y - HALO_M,
        max_x + HALO_M,
        max_y + HALO_M,
    )
    request_side = side_x + 2.0 * HALO_M
    pixel_count = request_side / RESOLUTION_M
    if not math.isclose(pixel_count, round(pixel_count), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("halo-expanded BBOX does not resolve to whole 2 m pixels")
    width = height = int(round(pixel_count))
    transform = Affine(
        RESOLUTION_M,
        0.0,
        request_bounds[0],
        0.0,
        -RESOLUTION_M,
        request_bounds[3],
    )
    return SourceGrid(
        core_bounds_l93_m=bounds,
        request_bounds_l93_m=request_bounds,
        width=width,
        height=height,
        transform=transform,
    )


def _validate_service_url(service_url: str) -> None:
    parts = urlsplit(service_url)
    if parts.scheme not in {"https", "file"}:
        raise ValueError("terrain sources must use an HTTPS or file URL")
    if parts.scheme == "https" and not parts.netloc:
        raise ValueError("HTTPS terrain source URL must include a host")


def build_wms_request(
    *,
    role: SourceRole,
    service_url: str,
    layer: str,
    grid: SourceGrid,
) -> RasterRequest:
    """Build a locked WMS 1.3 GetMap request for the supplied source grid."""

    if role not in {"mnt", "mns"}:
        raise ValueError("source role must be mnt or mns")
    _validate_service_url(service_url)
    if not layer.strip():
        raise ValueError("source layer cannot be empty")
    if "ortho" in layer.casefold() or "ortho" in service_url.casefold():
        raise ValueError("orthophoto sources are forbidden in terrain acquisition")
    if grid.resolution_m != RESOLUTION_M or grid.halo_m != HALO_M or grid.crs != CRS:
        raise ValueError("terrain source grid must use EPSG:2154, 2 m and a 10 m halo")

    parameters = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": CRS,
        "BBOX": ",".join(f"{value:.3f}" for value in grid.request_bounds_l93_m),
        "WIDTH": str(grid.width),
        "HEIGHT": str(grid.height),
        "FORMAT": "image/geotiff",
        "RESAMPLING": RESAMPLING,
    }
    separator = "&" if "?" in service_url else "?"
    url = service_url + separator + urlencode(parameters)
    return RasterRequest(role=role, layer=layer, url=url, grid=grid)


def build_source_pair_plan(
    core_bounds_l93_m: tuple[float, float, float, float] | list[float],
    *,
    mnt_service_url: str,
    mnt_layer: str,
    mns_service_url: str,
    mns_layer: str,
) -> SourcePairPlan:
    """Create the two requests from one grid so they cannot drift apart."""

    grid = build_grid(core_bounds_l93_m)
    return SourcePairPlan(
        grid=grid,
        mnt=build_wms_request(
            role="mnt",
            service_url=mnt_service_url,
            layer=mnt_layer,
            grid=grid,
        ),
        mns=build_wms_request(
            role="mns",
            service_url=mns_service_url,
            layer=mns_layer,
            grid=grid,
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantize_mm(values_m: np.ndarray) -> np.ndarray:
    values = np.asarray(values_m, dtype="float64")
    if not np.isfinite(values).all():
        raise RuntimeError("terrain source contains non-finite elevation samples")
    scaled = values * 1_000.0
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    if rounded.min() < np.iinfo("int32").min or rounded.max() > np.iinfo("int32").max:
        raise RuntimeError("terrain source millimetres exceed signed int32")
    return rounded.astype("<i4")


def _grid_sha256(values: np.ndarray, domain: bytes) -> str:
    canonical = np.asarray(values, dtype="<i4")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        int(canonical.shape[0]).to_bytes(4, "little", signed=False)
        + int(canonical.shape[1]).to_bytes(4, "little", signed=False)
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def canonical_normal_halo_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i4")
    if canonical.shape != (NORMAL_HALO_VERTEX_COUNT, NORMAL_HALO_VERTEX_COUNT):
        raise ValueError("canonical normal halo hash requires shape (253, 253)")
    digest = hashlib.sha256()
    digest.update(b"FVTQ-NORMAL-HALO-2M-MM-V1\0")
    digest.update(
        struct.pack(
            "<III",
            NORMAL_HALO_VERTEX_COUNT,
            NORMAL_HALO_VERTEX_COUNT,
            int(RESOLUTION_M * 1_000),
        )
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def canonicalize_source_raster(path: Path, request: RasterRequest) -> CanonicalRaster:
    """Convert the 260 WMS pixels into exact shared 2 m terrain vertices.

    A 520 m WMS image at 2 m contains cell centres at odd metre offsets
    ``-9, -7, ..., 509`` relative to the tile.  Every desired even-coordinate
    vertex is therefore the exact mean of four surrounding pixel centres.
    Quantization happens before this interpolation and division is rounded half
    away from zero, making the result independent of floating-point workers.
    """

    validate_raster(path, request)
    if (request.grid.width, request.grid.height) != (260, 260):
        raise RuntimeError("canonical terrain conversion requires a 260x260 WMS grid")
    with rasterio.open(path) as dataset:
        # Raster rows are north-to-south.  Every canonical terrain array is the
        # opposite, as required by FVTQ and compact HAG.
        working_mm = _quantize_mm(np.flipud(dataset.read(1)).copy())

    # Indices 3..256 have centres at -3..503 m.  Averaging each neighbouring
    # 2x2 group yields 253 vertices at -2..502 m.  The inner 251 vertices are
    # exactly 0..500 m and adjacent tile conversions use the same source centres.
    samples = np.asarray(working_mm[3:257, 3:257], dtype="int64")
    sums = samples[:-1, :-1] + samples[1:, :-1] + samples[:-1, 1:] + samples[1:, 1:]
    halo_values = np.where(sums >= 0, (sums + 2) // 4, -((-sums + 2) // 4))
    if halo_values.shape != (NORMAL_HALO_VERTEX_COUNT, NORMAL_HALO_VERTEX_COUNT):
        raise AssertionError("canonical normal halo has an unexpected shape")
    if (
        halo_values.min() < np.iinfo("int32").min
        or halo_values.max() > np.iinfo("int32").max
    ):
        raise RuntimeError("canonical terrain vertices exceed signed int32")
    normal_halo_mm = np.asarray(halo_values, dtype="<i4")
    core_mm = np.asarray(normal_halo_mm[1:-1, 1:-1], dtype="<i4").copy()
    if core_mm.shape != (CORE_VERTEX_COUNT, CORE_VERTEX_COUNT):
        raise AssertionError("canonical terrain core has an unexpected shape")
    return CanonicalRaster(
        core_mm=core_mm,
        normal_halo_mm=normal_halo_mm,
        working_grid_sha256=_grid_sha256(
            working_mm, b"FIREVIEWER-WMS-WORKING-GRID-MM-V1\0"
        ),
        core_sha256=_grid_sha256(core_mm, b"FIREVIEWER-TERRAIN-CORE-MM-V1\0"),
        normal_halo_sha256=canonical_normal_halo_sha256(normal_halo_mm),
    )


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    staging = path.with_name(path.name + ".tmp")
    with staging.open("wb") as output:
        np.save(output, np.asarray(values, dtype="<i4"), allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(staging, path)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is not None:
        return str(value)
    lowered = name.casefold()
    for key, candidate in headers.items():
        if str(key).casefold() == lowered:
            return str(candidate)
    return None


def _response_status(response: Response) -> int | None:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()  # type: ignore[attr-defined]
    return None if status is None else int(status)


def _copy_response(response: Response, destination: BinaryIO) -> int:
    received = 0
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            return received
        destination.write(chunk)
        received += len(chunk)


def receive_to_part(
    request_url: str,
    destination: Path,
    *,
    opener: Opener = urlopen,
    timeout_s: float = 180.0,
) -> DownloadReceipt:
    """Download into ``<destination>.part`` with strict HTTP Range resume.

    The function intentionally leaves the completed payload in ``.part``.
    Raster validation must succeed before the caller atomically replaces the
    final destination.
    """

    _validate_service_url(request_url)
    if timeout_s <= 0 or not math.isfinite(timeout_s):
        raise ValueError("timeout_s must be a positive finite number")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    http_request = Request(request_url, headers=headers)

    with opener(http_request, timeout=timeout_s) as response:
        status = _response_status(response)
        content_length_header = _header(response.headers, "Content-Length")
        content_length = (
            int(content_length_header) if content_length_header is not None else None
        )
        expected_total: int | None = None

        if offset:
            if status != 206:
                raise RuntimeError(
                    f"strict Range resume expected HTTP 206, received {status!r}"
                )
            content_range = _header(response.headers, "Content-Range")
            match = _CONTENT_RANGE.fullmatch(content_range or "")
            if match is None:
                raise RuntimeError("strict Range resume requires a valid Content-Range")
            start, end, expected_total = (int(value) for value in match.groups())
            if start != offset or end < start or end + 1 != expected_total:
                raise RuntimeError(
                    "Content-Range must start at the local byte count and end at total - 1"
                )
            expected_response_bytes = end - start + 1
            if content_length is not None and content_length != expected_response_bytes:
                raise RuntimeError("Content-Length and Content-Range disagree")
            mode = "ab"
        else:
            scheme = urlsplit(request_url).scheme
            if scheme == "https" and status != 200:
                raise RuntimeError(
                    f"fresh HTTPS download expected HTTP 200, received {status!r}"
                )
            if scheme == "file" and status not in {None, 200}:
                raise RuntimeError(
                    f"fresh file download returned unexpected status {status!r}"
                )
            mode = "wb"

        with part.open(mode) as output:
            received = _copy_response(response, output)

    if content_length is not None and received != content_length:
        raise RuntimeError(
            f"response ended after {received} bytes, expected {content_length}"
        )
    byte_count = part.stat().st_size
    if expected_total is not None and byte_count != expected_total:
        raise RuntimeError(
            f"resumed payload contains {byte_count} bytes, expected {expected_total}"
        )
    return DownloadReceipt(
        byte_count=byte_count,
        sha256=sha256_file(part),
        resumed_from_byte=offset,
    )


def _same_transform(left: Affine, right: Affine) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) for a, b in zip(left, right)
    )


def validate_raster(path: Path, request: RasterRequest) -> dict[str, object]:
    """Validate one staged source against every locked grid field."""

    with rasterio.open(path) as dataset:
        if dataset.crs is None or dataset.crs.to_epsg() != 2154:
            raise RuntimeError(f"{request.role.upper()} CRS must be exactly EPSG:2154")
        if dataset.width != request.grid.width or dataset.height != request.grid.height:
            raise RuntimeError(
                f"{request.role.upper()} dimensions are {dataset.width}x{dataset.height}; "
                f"expected {request.grid.width}x{request.grid.height}"
            )
        if not _same_transform(dataset.transform, request.grid.transform):
            raise RuntimeError(
                f"{request.role.upper()} transform does not match the 2 m grid"
            )
        if dataset.count != 1 or not np.issubdtype(
            np.dtype(dataset.dtypes[0]), np.number
        ):
            raise RuntimeError(f"{request.role.upper()} must contain one numeric band")
        if dataset.nodata is None or not math.isfinite(float(dataset.nodata)):
            raise RuntimeError(
                f"{request.role.upper()} must declare a finite nodata value"
            )
        values = dataset.read(1, masked=True)
        nodata_pixels = int(np.count_nonzero(np.ma.getmaskarray(values)))
        finite_pixels = int(np.count_nonzero(np.isfinite(values.compressed())))
        if nodata_pixels:
            raise RuntimeError(
                f"{request.role.upper()} contains {nodata_pixels} nodata pixels"
            )
        expected_pixels = dataset.width * dataset.height
        if finite_pixels != expected_pixels:
            raise RuntimeError(
                f"{request.role.upper()} contains non-finite elevation samples"
            )
        return {
            "crs": CRS,
            "width": dataset.width,
            "height": dataset.height,
            "transform": [float(value) for value in dataset.transform[:6]],
            "bounds_l93_m": [float(value) for value in dataset.bounds],
            "dtype": dataset.dtypes[0],
            "nodata": float(dataset.nodata),
            "nodata_pixel_count": nodata_pixels,
            "finite_pixel_count": finite_pixels,
        }


def validate_coregistration(mnt_path: Path, mns_path: Path) -> None:
    """Fail closed unless MNT and MNS share the exact same raster grid."""

    with rasterio.open(mnt_path) as mnt, rasterio.open(mns_path) as mns:
        if mnt.crs != mns.crs:
            raise RuntimeError("MNT/MNS CRS mismatch")
        if (mnt.width, mnt.height) != (mns.width, mns.height):
            raise RuntimeError("MNT/MNS dimensions mismatch")
        if not _same_transform(mnt.transform, mns.transform):
            raise RuntimeError("MNT/MNS transform mismatch")
        if tuple(mnt.bounds) != tuple(mns.bounds):
            raise RuntimeError("MNT/MNS bounds mismatch")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    staging = path.with_name(path.name + ".tmp")
    with staging.open("w", encoding="utf-8", newline="\n") as output:
        output.write(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        output.flush()
        os.fsync(output.fileno())
    os.replace(staging, path)


def _validated_local_download(path: Path, request: RasterRequest) -> DownloadReceipt:
    validate_raster(path, request)
    return DownloadReceipt(
        byte_count=path.stat().st_size,
        sha256=sha256_file(path),
        resumed_from_byte=path.stat().st_size,
    )


def _resume_or_restart_download(
    request: RasterRequest,
    destination: Path,
    *,
    opener: Opener,
    timeout_s: float,
) -> DownloadReceipt:
    part = destination.with_name(destination.name + ".part")
    try:
        return receive_to_part(
            request.url, destination, opener=opener, timeout_s=timeout_s
        )
    except (HTTPError, RuntimeError) as error:
        message = str(error)
        recoverable_eof = isinstance(error, HTTPError) and error.code == 416
        malformed_resume = isinstance(error, RuntimeError) and any(
            marker in message
            for marker in ("Range resume", "Content-Range", "Content-Length")
        )
        if not part.exists() or not (recoverable_eof or malformed_resume):
            raise
        # A complete but invalid staged response typically receives HTTP 416 or
        # an empty EOF range.  It cannot be appended safely; restart this one
        # explicit source while preserving the transactional pair directory.
        part.unlink()
        return receive_to_part(
            request.url, destination, opener=opener, timeout_s=timeout_s
        )


def _load_accepted_pair(
    output_directory: Path, plan: SourcePairPlan
) -> dict[str, object]:
    resolved_output = output_directory.resolve(strict=True)

    def bounded_child(file_name: str) -> Path:
        child = (output_directory / file_name).resolve(strict=False)
        if child.parent != resolved_output:
            raise RuntimeError("accepted source-pair file escapes its output directory")
        return child

    receipt_path = output_directory / "source-pair.done.json"
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("accepted source-pair receipt is unreadable") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SCHEMA
        or payload.get("status") != "downloaded_validated_coregistered_canonicalized"
    ):
        raise RuntimeError("accepted source-pair receipt has an invalid contract")
    grid_record = payload.get("grid")
    if not isinstance(grid_record, dict) or (
        grid_record.get("core_bounds_l93_m") != list(plan.grid.core_bounds_l93_m)
        or grid_record.get("request_bounds_l93_m")
        != list(plan.grid.request_bounds_l93_m)
        or grid_record.get("width") != plan.grid.width
        or grid_record.get("height") != plan.grid.height
    ):
        raise RuntimeError("accepted source-pair grid differs from the requested plan")
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("accepted source-pair receipt has no sources")
    for request in (plan.mnt, plan.mns):
        record = sources.get(request.role)
        if not isinstance(record, dict):
            raise RuntimeError(f"accepted source-pair receipt misses {request.role}")
        expected_raw_name = f"{request.role}-2m.tif"
        if record.get("file_name") != expected_raw_name:
            raise RuntimeError(
                f"accepted {request.role} raw source file name is not canonical"
            )
        if (
            record.get("layer") != request.layer
            or record.get("request_url") != request.url
        ):
            raise RuntimeError(
                f"accepted {request.role} provenance differs from the plan"
            )
        canonical = record.get("canonical")
        if not isinstance(canonical, dict):
            raise RuntimeError(
                f"accepted {request.role} canonical provenance is absent"
            )
        expected_canonical_name = f"{request.role}-canonical-normal-halo-2m-mm.npy"
        if canonical.get("file_name") != expected_canonical_name:
            raise RuntimeError(
                f"accepted {request.role} canonical file name is not canonical"
            )
        canonical_path = bounded_child(expected_canonical_name)
        if not canonical_path.is_file() or sha256_file(canonical_path) != canonical.get(
            "file_sha256"
        ):
            raise RuntimeError(f"accepted {request.role} canonical grid hash mismatch")
        try:
            with canonical_path.open("rb") as source:
                halo_mm = np.load(source, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"accepted {request.role} canonical grid is unreadable"
            ) from error
        if (
            halo_mm.shape != (NORMAL_HALO_VERTEX_COUNT, NORMAL_HALO_VERTEX_COUNT)
            or halo_mm.dtype != np.dtype("int32")
            or canonical_normal_halo_sha256(halo_mm)
            != canonical.get("normal_halo_sha256")
            or _grid_sha256(halo_mm[1:-1, 1:-1], b"FIREVIEWER-TERRAIN-CORE-MM-V1\0")
            != canonical.get("core_sha256")
        ):
            raise RuntimeError(
                f"accepted {request.role} canonical payload hash mismatch"
            )
        raw_path = bounded_child(expected_raw_name)
        # Raw GeoTIFFs may legitimately have been deleted after all dependent
        # packages were accepted.  If present, they remain strictly validated.
        if raw_path.exists():
            if sha256_file(raw_path) != record.get("sha256"):
                raise RuntimeError(f"accepted {request.role} raw source hash mismatch")
            validate_raster(raw_path, request)
    return payload


def acquire_source_pair(
    plan: SourcePairPlan,
    output_directory: Path,
    *,
    opener: Opener = urlopen,
    timeout_s: float = 180.0,
) -> dict[str, object]:
    """Acquire and atomically publish one resumable canonical MNT/MNS pair.

    Every mutable transfer lives in a sibling staging directory.  MNT, MNS,
    canonical halo grids and their receipt become visible together through one
    final directory rename, so a crash cannot publish a partial accepted pair.
    """

    output_directory = Path(output_directory)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = output_directory.with_name(f".{output_directory.name}.staging")
    if output_directory.exists():
        receipt_path = output_directory / "source-pair.done.json"
        if receipt_path.is_file():
            return _load_accepted_pair(output_directory, plan)
        if staging_directory.exists():
            raise RuntimeError(
                "both incomplete source output and staging directory exist"
            )
        # Recover directories authored by the pre-transactional implementation.
        os.replace(output_directory, staging_directory)
    staging_directory.mkdir(parents=True, exist_ok=True)
    destinations = {
        "mnt": staging_directory / "mnt-2m.tif",
        "mns": staging_directory / "mns-2m.tif",
    }
    receipt_path = staging_directory / "source-pair.done.json"

    download_receipts: dict[str, DownloadReceipt] = {}
    validations: dict[str, dict[str, object]] = {}
    canonical_grids: dict[str, CanonicalRaster] = {}
    canonical_paths: dict[str, Path] = {}
    for request in (plan.mnt, plan.mns):
        destination = destinations[request.role]
        part = destination.with_name(destination.name + ".part")
        if destination.is_file():
            staged_source = destination
            download_receipts[request.role] = _validated_local_download(
                staged_source, request
            )
        else:
            staged_source = part
            if part.is_file():
                try:
                    download_receipts[request.role] = _validated_local_download(
                        part, request
                    )
                except (OSError, RuntimeError, rasterio.errors.RasterioError):
                    download_receipts[request.role] = _resume_or_restart_download(
                        request,
                        destination,
                        opener=opener,
                        timeout_s=timeout_s,
                    )
            else:
                download_receipts[request.role] = receive_to_part(
                    request.url,
                    destination,
                    opener=opener,
                    timeout_s=timeout_s,
                )
        validations[request.role] = validate_raster(staged_source, request)
        canonical = canonicalize_source_raster(staged_source, request)
        canonical_path = staging_directory / (
            f"{request.role}-canonical-normal-halo-2m-mm.npy"
        )
        _atomic_npy(canonical_path, canonical.normal_halo_mm)
        canonical_grids[request.role] = canonical
        canonical_paths[request.role] = canonical_path

    staged_mnt = (
        destinations["mnt"]
        if destinations["mnt"].is_file()
        else destinations["mnt"].with_name(destinations["mnt"].name + ".part")
    )
    staged_mns = (
        destinations["mns"]
        if destinations["mns"].is_file()
        else destinations["mns"].with_name(destinations["mns"].name + ".part")
    )
    validate_coregistration(staged_mnt, staged_mns)
    if staged_mnt != destinations["mnt"]:
        os.replace(staged_mnt, destinations["mnt"])
    if staged_mns != destinations["mns"]:
        os.replace(staged_mns, destinations["mns"])

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "downloaded_validated_coregistered_canonicalized",
        "grid": {
            "crs": CRS,
            "resolution_m": RESOLUTION_M,
            "halo_m": HALO_M,
            "core_bounds_l93_m": list(plan.grid.core_bounds_l93_m),
            "request_bounds_l93_m": list(plan.grid.request_bounds_l93_m),
            "width": plan.grid.width,
            "height": plan.grid.height,
            "resampling": RESAMPLING,
            "canonical_layout": {
                "schema": CANONICAL_SCHEMA,
                "row_order": "south_to_north",
                "height_unit": "millimetre",
                "normal_halo_shape": [253, 253],
                "normal_halo_bounds_relative_m": [-2.0, -2.0, 502.0, 502.0],
                "core_shape": [251, 251],
                "core_slice": [1, 252, 1, 252],
            },
        },
        "sources": {},
        "excluded": ["orthophoto", "uniform_0.5m_source"],
    }
    sources = payload["sources"]
    assert isinstance(sources, dict)
    for request in (plan.mnt, plan.mns):
        receipt = download_receipts[request.role]
        sources[request.role] = {
            "layer": request.layer,
            "request_url": request.url,
            "file_name": destinations[request.role].name,
            "byte_count": receipt.byte_count,
            "sha256": receipt.sha256,
            "resumed_from_byte": receipt.resumed_from_byte,
            "validation": validations[request.role],
            "canonical": {
                "schema": CANONICAL_SCHEMA,
                "file_name": canonical_paths[request.role].name,
                "file_sha256": sha256_file(canonical_paths[request.role]),
                "working_grid_sha256": canonical_grids[
                    request.role
                ].working_grid_sha256,
                "core_sha256": canonical_grids[request.role].core_sha256,
                "normal_halo_sha256": canonical_grids[request.role].normal_halo_sha256,
            },
        }
    _atomic_json(receipt_path, payload)
    os.replace(staging_directory, output_directory)
    return payload


def cleanup_partial_files(root: Path, destinations: list[Path]) -> list[Path]:
    """Remove only explicitly named ``.part`` siblings contained by ``root``."""

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("partial cleanup root must be a directory")
    removed: list[Path] = []
    seen: set[Path] = set()
    for destination in destinations:
        resolved_destination = destination.resolve(strict=False)
        if not resolved_destination.is_relative_to(resolved_root):
            raise ValueError("partial cleanup destination escapes its bounded root")
        part = resolved_destination.with_name(resolved_destination.name + ".part")
        if part in seen:
            continue
        seen.add(part)
        if part.is_file():
            part.unlink()
            removed.append(part)
    return removed
