"""Fetch a compact, resampled IGN LiDAR-HD MNT mosaic for global-only maps.

This deliberately does not acquire MNS data and does not pretend that a 5 m
WMS response is a native 0.5 m LiDAR cache.  It is the bounded terrain source
for the ``global_mnt_only`` production contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy

from fetch_ign_orthophoto import (
    IGN_WMS_URL,
    OUTPUT_CRS,
    OrthophotoPlan,
    OrthophotoTile,
    _download,
    _is_lambert93,
    build_plan,
    sha256_file,
)


MNT_LAYER = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
SCHEMA = "fireviewer.ign-global-mnt-source.v1"
NODATA = -9999.0


def _validate_tile(path: Path, tile: OrthophotoTile) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        if dataset.width != tile.width or dataset.height != tile.height:
            raise RuntimeError(
                f"MNT tile dimensions {dataset.width}x{dataset.height} do not match "
                f"{tile.width}x{tile.height}"
            )
        if dataset.count != 1 or not np.issubdtype(np.dtype(dataset.dtypes[0]), np.number):
            raise RuntimeError("IGN MNT WMS tile is not a single numeric band")
        if not _is_lambert93(dataset.crs):
            raise RuntimeError(f"IGN MNT tile CRS is {dataset.crs!s}, expected EPSG:2154")
        actual = tuple(float(value) for value in dataset.bounds)
        if not all(math.isclose(left, right, abs_tol=0.02) for left, right in zip(actual, tile.bounds_l93)):
            raise RuntimeError(f"IGN MNT tile bounds {actual!r} do not match {tile.bounds_l93!r}")
        values = dataset.read(1, masked=True)
        usable = int(np.count_nonzero(~np.ma.getmaskarray(values) & np.isfinite(values.data)))
        return {
            "width": dataset.width,
            "height": dataset.height,
            "dtype": dataset.dtypes[0],
            "crs": OUTPUT_CRS,
            "bounds_l93_m": list(actual),
            "usable_pixel_count": usable,
            "coverage_state": "measured" if usable else "no_data",
        }


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
        "estimated_float32_network_bytes": plan.width * plan.height * 4 + len(plan.tiles) * 4096,
        "estimated_float32_network_mib": (plan.width * plan.height * 4 + len(plan.tiles) * 4096) / (1024 * 1024),
    }


def execute_plan(
    plan: OrthophotoPlan,
    output: Path,
    *,
    timeout_s: float = 180.0,
    overwrite: bool = False,
    fetcher: Callable[[str, Path, float], None] = _download,
) -> dict[str, object]:
    """Download and validate each WMS chunk, retaining only the final MNT COG."""

    source_record = output.with_suffix(".source.json")
    existing = [path for path in (output, source_record) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite existing output(s): " + ", ".join(map(str, existing)))
    output.parent.mkdir(parents=True, exist_ok=True)
    tile_records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="fireviewer-ign-mnt-", dir=output.parent) as temporary:
        temporary_dir = Path(temporary)
        staging = temporary_dir / "mnt-global.staging.tif"
        profile = {
            "driver": "GTiff",
            "width": plan.width,
            "height": plan.height,
            "count": 1,
            "dtype": "float32",
            "crs": OUTPUT_CRS,
            "transform": plan.transform,
            "nodata": NODATA,
            "compress": "DEFLATE",
            "predictor": 3,
            "bigtiff": "IF_SAFER",
        }
        if min(plan.width, plan.height) >= 16:
            block_size = max(16, min(512, (min(plan.width, plan.height) // 16) * 16))
            profile.update({"tiled": True, "blockxsize": block_size, "blockysize": block_size})
        with rasterio.open(staging, "w", **profile) as mosaic:
            for index, tile in enumerate(plan.tiles):
                tile_path = temporary_dir / f"tile-{tile.row:03d}-{tile.column:03d}.tif"
                fetcher(tile.url, tile_path, timeout_s)
                validation = _validate_tile(tile_path, tile)
                with rasterio.open(tile_path) as source:
                    values = source.read(1, masked=True).astype("float32")
                    dense = np.ma.filled(values, NODATA)
                    dense[~np.isfinite(dense)] = NODATA
                    mosaic.write(dense, 1, window=tile.window)
                tile_records.append({
                    "index": index,
                    "row": tile.row,
                    "column": tile.column,
                    "window": [int(tile.window.col_off), int(tile.window.row_off), tile.width, tile.height],
                    "bounds_l93_m": list(tile.bounds_l93),
                    "request_url": tile.url,
                    "downloaded_bytes": tile_path.stat().st_size,
                    "sha256": sha256_file(tile_path),
                    "validation": validation,
                })
            factors = [factor for factor in (2, 4, 8, 16, 32) if min(plan.width, plan.height) // factor >= 128]
            if factors:
                mosaic.build_overviews(factors, Resampling.average)
                mosaic.update_tags(ns="rio_overview", resampling="average")
        cog = temporary_dir / "mnt-global.cog.tif"
        rio_copy(staging, cog, driver="COG", compress="DEFLATE", blocksize=512, overview_resampling="AVERAGE", bigtiff="IF_SAFER")
        os.replace(cog, output)
    with rasterio.open(output) as dataset:
        output_validation = {
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": OUTPUT_CRS if _is_lambert93(dataset.crs) else str(dataset.crs),
            "bounds_l93_m": [float(value) for value in dataset.bounds],
            "nodata": dataset.nodata,
            "overview_count": len(dataset.overviews(1)),
        }
    if output_validation["crs"] != OUTPUT_CRS or output_validation["count"] != 1:
        raise RuntimeError("final MNT COG validation failed")
    record: dict[str, object] = {
        "schema": SCHEMA,
        "status": "downloaded_and_validated",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "IGN / Géoplateforme",
        "product": "LiDAR HD MNT elevation grid",
        "terrain_surface": "mnt_wms_resampled",
        "excluded_surfaces": ["mns", "vegetation_assets", "building_assets", "detail_lod"],
        "layer": MNT_LAYER,
        "service": {"protocol": "WMS 1.3.0", "getmap_url": IGN_WMS_URL},
        "request": plan_to_dict(plan),
        "tiles": tile_records,
        "coverage": {
            "measured_tile_count": sum(
                record["validation"]["coverage_state"] == "measured"
                for record in tile_records
            ),
            "no_data_tile_count": sum(
                record["validation"]["coverage_state"] == "no_data"
                for record in tile_records
            ),
            "no_data_policy": "preserved_as_masked_cells_in_far_terrain; no elevation is invented",
        },
        "output": {"file_name": output.name, "bytes": output.stat().st_size, "sha256": sha256_file(output), "validation": output_validation},
    }
    source_record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def write_production_manifest(
    path: Path,
    *,
    aoi: Path,
    origin: list[float],
    plan: OrthophotoPlan,
    source_record: dict[str, object],
) -> None:
    """Write the minimal global-surface manifest consumed by the catalog exporter."""

    output = source_record["output"]
    manifest = {
        "schema": "fireviewer.global-05m-production-manifest.v1",
        "status": "ready",
        "plan_id": f"global-mnt-5m-{sha256_file(aoi)[:12]}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_l93_m": origin,
        "aoi": {"sha256": sha256_file(aoi), "bounds_l93_m": list(plan.bounds_l93)},
        "tiling": {
            "mode": "global_mnt_only",
            "terrain_surface": "mnt_wms_resampled_5m",
            "output_tile_size_m": None,
            "halo_m": 0.0,
        },
        "source_tiles": [],
        "tiles": [],
        "global_mnt": {
            "source_record_sha256": sha256_file(path.parent.parent / "terrain" / f"{Path(str(output['file_name'])).stem}.source.json"),
            "output": output,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", nargs=4, type=float, required=True, metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"))
    parser.add_argument("--resolution-m", type=float, default=5.0)
    parser.add_argument("--tile-pixels", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aoi", type=Path)
    parser.add_argument("--origin", nargs=3, type=float)
    parser.add_argument("--production-manifest", type=Path)
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
    if args.resolution_m != 5.0:
        raise ValueError("global_mnt_only is fixed at 5 m")
    plan = build_plan(args.bounds, args.resolution_m, args.tile_pixels, layer=MNT_LAYER)
    record = {"schema": "fireviewer.ign-global-mnt-plan.v1", "mode": "dry-run" if args.dry_run else "execute", "provider": "IGN / Géoplateforme", "product": "LiDAR HD MNT elevation grid", "terrain_surface": "mnt_wms_resampled", "layer": MNT_LAYER, "plan": plan_to_dict(plan)}
    print(json.dumps(record, ensure_ascii=False, indent=2))
    estimated_mib = float(record["plan"]["estimated_float32_network_mib"])
    if estimated_mib > args.max_estimated_download_mib and not args.allow_large_download:
        raise RuntimeError(f"estimated MNT WMS download is {estimated_mib:.1f} MiB; use --allow-large-download after reviewing the plan")
    if args.dry_run:
        return 0
    if args.output is None:
        raise ValueError("--output is required with --execute")
    source_record = execute_plan(plan, args.output.resolve(), timeout_s=args.timeout_s, overwrite=args.overwrite)
    manifest_options = (args.aoi, args.origin, args.production_manifest)
    if any(value is not None for value in manifest_options):
        if not all(value is not None for value in manifest_options):
            raise ValueError("--aoi, --origin and --production-manifest must be supplied together")
        write_production_manifest(
            args.production_manifest.resolve(),
            aoi=args.aoi.resolve(),
            origin=[float(value) for value in args.origin],
            plan=plan,
            source_record=source_record,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
