"""Build reproducible 5 m FAR MNT/MNS COGs from the validated 0.5 m tile sources."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.shutil import copy as raster_copy


SCHEMA = "fireviewer.far-raster-mosaic.v1"
PRODUCTION_MANIFEST_SCHEMA = "fireviewer.global-05m-production-manifest.v1"
NODATA = -9999.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _aoi_geometries(path: Path) -> tuple[list[dict[str, Any]], list[float]]:
    document = read_json(path)
    if document.get("type") == "FeatureCollection":
        features = document.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("AOI FeatureCollection contains no feature")
        geometries = [
            feature.get("geometry") for feature in features if isinstance(feature, dict)
        ]
    elif document.get("type") == "Feature":
        geometries = [document.get("geometry")]
    else:
        geometries = [document]
    valid = [geometry for geometry in geometries if isinstance(geometry, dict)]
    if not valid:
        raise ValueError("AOI contains no geometry")

    coordinates: list[tuple[float, float]] = []

    def collect(node: object) -> None:
        if (
            isinstance(node, list)
            and len(node) >= 2
            and isinstance(node[0], (int, float))
            and isinstance(node[1], (int, float))
        ):
            coordinates.append((float(node[0]), float(node[1])))
            return
        if isinstance(node, list):
            for child in node:
                collect(child)

    for geometry in valid:
        collect(geometry.get("coordinates"))
    if not coordinates:
        raise ValueError("AOI geometry contains no coordinate")
    xs, ys = zip(*coordinates, strict=True)
    return valid, [min(xs), min(ys), max(xs), max(ys)]


def _snap_bounds(
    bounds: Sequence[float], resolution_m: float
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = map(float, bounds)
    if resolution_m <= 0 or xmax <= xmin or ymax <= ymin:
        raise ValueError("invalid FAR raster bounds or resolution")
    return (
        math.floor(xmin / resolution_m) * resolution_m,
        math.floor(ymin / resolution_m) * resolution_m,
        math.ceil(xmax / resolution_m) * resolution_m,
        math.ceil(ymax / resolution_m) * resolution_m,
    )


def _source_paths(
    manifest: dict[str, Any], manifest_directory: Path, product: str
) -> list[Path]:
    paths: list[Path] = []
    for source in manifest.get("source_tiles", []):
        if not isinstance(source, dict):
            raise ValueError("production manifest contains an invalid source tile")
        assets = source.get("assets")
        record = assets.get(product) if isinstance(assets, dict) else None
        if not isinstance(record, dict):
            raise ValueError(f"source tile {source.get('id')} lacks {product}")
        path = manifest_directory / Path(str(record.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"source {product} is absent: {path}")
        if path.stat().st_size != int(record.get("byte_count", -1)):
            raise ValueError(f"source {product} size differs: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"source {product} checksum differs: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"production manifest contains no {product} source")
    return paths


def _write_cog(
    paths: Sequence[Path],
    output: Path,
    *,
    geometries: Sequence[dict[str, Any]],
    bounds: Sequence[float],
    resolution_m: float,
    product: str,
) -> dict[str, Any]:
    staging = output.with_name(f".{output.name}.{os.getpid()}.staging.tif")
    cog_staging = output.with_name(f".{output.name}.{os.getpid()}.cog.tif")
    if output.exists():
        raise FileExistsError(f"immutable FAR raster already exists: {output}")
    try:
        with ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(path)) for path in paths]
            for dataset in datasets:
                if dataset.crs is None or dataset.crs.to_epsg() != 2154:
                    raise ValueError(f"{dataset.name} is not EPSG:2154")
            mosaic, transform = merge(
                datasets,
                bounds=tuple(bounds),
                res=resolution_m,
                nodata=NODATA,
                dtype="float32",
                resampling=Resampling.average,
                method="first",
            )
        inside = geometry_mask(
            geometries,
            out_shape=mosaic.shape[1:],
            transform=transform,
            invert=True,
            all_touched=False,
        )
        mosaic[0, ~inside] = NODATA
        valid = mosaic[0][np.isfinite(mosaic[0]) & (mosaic[0] != NODATA)]
        if valid.size == 0:
            raise ValueError(f"FAR {product} mosaic contains no valid elevation")
        profile = {
            "driver": "GTiff",
            "width": mosaic.shape[2],
            "height": mosaic.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:2154",
            "transform": transform,
            "nodata": NODATA,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "DEFLATE",
            "predictor": 3,
        }
        with rasterio.open(staging, "w", **profile) as target:
            target.write(mosaic)
            target.update_tags(
                source_role=product.upper(),
                processing=f"0.5 m source mosaic averaged to {resolution_m:g} m and AOI masked",
            )
        raster_copy(
            staging,
            cog_staging,
            driver="COG",
            compress="DEFLATE",
            blocksize=512,
            overview_resampling="AVERAGE",
            overview_count=4,
            bigtiff="IF_SAFER",
        )
        os.replace(cog_staging, output)
    finally:
        staging.unlink(missing_ok=True)
        cog_staging.unlink(missing_ok=True)

    with rasterio.open(output) as dataset:
        if dataset.crs is None or dataset.crs.to_epsg() != 2154:
            raise ValueError(f"published FAR {product} is not EPSG:2154")
        return {
            "path": output.name,
            "sha256": sha256_file(output),
            "byte_count": output.stat().st_size,
            "resolution_m": resolution_m,
            "width": dataset.width,
            "height": dataset.height,
            "bounds_l93_m": list(dataset.bounds),
            "elevation_min_m": float(valid.min()),
            "elevation_max_m": float(valid.max()),
            "valid_sample_count": int(valid.size),
        }


def build_far_rasters(
    production_manifest_path: Path,
    aoi_path: Path,
    output_directory: Path,
    *,
    resolution_m: float = 5.0,
    allow_source_only: bool = False,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "far-raster-manifest.json"
    production = read_json(production_manifest_path)
    if production.get("schema") != PRODUCTION_MANIFEST_SCHEMA:
        raise ValueError("unsupported 0.5 m production manifest")
    production_ready = production.get("status") == "ready"
    sources_ready = all(
        isinstance(source, dict) and source.get("status", {}).get("state") == "ready"
        for source in production.get("source_tiles", [])
    )
    if not production_ready and not (allow_source_only and sources_ready):
        raise ValueError(
            "0.5 m production manifest is not ready; global-only export requires every LiDAR source to be ready"
        )
    geometries, geometry_bounds = _aoi_geometries(aoi_path)
    bounds = _snap_bounds(geometry_bounds, resolution_m)
    production_sha256 = sha256_file(production_manifest_path)
    aoi_sha256 = sha256_file(aoi_path)
    if result_path.exists():
        result = read_json(result_path)
        if result.get("schema") != SCHEMA:
            raise ValueError("existing FAR raster manifest uses another schema")
        if float(result.get("resolution_m", -1)) != float(resolution_m):
            raise ValueError("existing FAR raster resolution differs")
        if result.get("aoi", {}).get("sha256") != aoi_sha256:
            raise ValueError("existing FAR raster AOI differs")
        if result.get("production_manifest", {}).get("sha256") != production_sha256:
            raise ValueError("existing FAR raster source manifest differs")
        for record in result.get("outputs", {}).values():
            path = output_directory / str(record["path"])
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError("existing FAR raster output differs from its manifest")
        return result

    outputs: dict[str, Any] = {}
    for product in ("mnt", "mns"):
        paths = _source_paths(production, production_manifest_path.parent, product)
        outputs[product] = _write_cog(
            paths,
            output_directory / f"{product}-global.cog.tif",
            geometries=geometries,
            bounds=bounds,
            resolution_m=resolution_m,
            product=product,
        )
    result = {
        "schema": SCHEMA,
        "crs": "EPSG:2154",
        "resolution_m": resolution_m,
        "aoi": {
            "path": aoi_path.name,
            "sha256": aoi_sha256,
            "bounds_l93_m": geometry_bounds,
        },
        "production_manifest": {
            "path": production_manifest_path.name,
            "sha256": production_sha256,
        },
        "source_only": bool(allow_source_only and not production_ready),
        "outputs": outputs,
    }
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_bytes(json_bytes(result))
    os.replace(temporary, result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution-m", type=float, default=5.0)
    parser.add_argument(
        "--allow-source-only",
        action="store_true",
        help="Build the global terrain from checked LiDAR sources without MID/NEAR tile production.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_far_rasters(
        args.production_manifest.resolve(),
        args.aoi.resolve(),
        args.output_dir.resolve(),
        resolution_m=args.resolution_m,
        allow_source_only=args.allow_source_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
