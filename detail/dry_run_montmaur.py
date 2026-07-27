"""Validate Montmaur detail inputs without querying or decompressing COPC points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import laspy
from shapely.geometry import box

from prepare_montmaur_detail import (
    BUILDING_CLASS,
    VEGETATION_CLASSES,
    _is_copc,
    _source_asset,
    _validate_point_header,
    _write_json,
    load_raster_pair,
    load_vector_features,
    sha256_file,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _zone(contract: dict[str, Any], zone_id: str) -> dict[str, Any]:
    matches = [zone for zone in contract.get("zones", []) if zone.get("id") == zone_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one zone {zone_id!r}, found {len(matches)}")
    return matches[0]


def dry_run(
    *,
    zone_contract_path: Path,
    source_root: Path,
    buildings_path: Path,
    hedges_path: Path,
    output_path: Path,
    zone_id: str = "montmaur",
) -> dict[str, Any]:
    contract = _read(zone_contract_path)
    zone = _zone(contract, zone_id)
    bounds = tuple(float(value) for value in zone["bounds_l93_metres"])
    if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Invalid detail zone bounds")
    aoi = box(*bounds)
    source_manifest_path = source_root / "source-manifest.json"
    source_manifest = _read(source_manifest_path)
    if source_manifest.get("zone", {}).get("id") != zone_id:
        raise ValueError(
            "Source manifest zone does not match the requested contract zone"
        )
    if source_manifest.get("source_contract", {}).get("sha256") != sha256_file(
        zone_contract_path
    ):
        raise ValueError(
            "Source manifest does not reference the current detail zone contract hash"
        )

    source_groups = source_manifest.get("sources", {})
    copc_records = source_groups.get("copc", [])
    mnt_records = source_groups.get("mnt_0_5_m", [])
    mns_records = source_groups.get("mns_0_5_m", [])
    copc_paths = [source_root / record["path"] for record in copc_records]
    mnt_paths = [source_root / record["path"] for record in mnt_records]
    mns_paths = [source_root / record["path"] for record in mns_records]
    all_records_and_paths = [
        *zip(copc_records, copc_paths),
        *zip(mnt_records, mnt_paths),
        *zip(mns_records, mns_paths),
    ]
    for record, path in all_records_and_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record.get("byte_count"):
            raise ValueError(f"Source size differs from the validated manifest: {path}")

    rasters = load_raster_pair(mnt_paths, mns_paths, aoi, required_resolution_m=0.5)
    buildings, building_stats = load_vector_features(
        buildings_path, aoi, crs="auto", kind="building"
    )
    hedges, hedge_stats = load_vector_features(
        hedges_path, aoi, crs="auto", kind="hedge"
    )

    selected_classes = {*VEGETATION_CLASSES, BUILDING_CLASS}
    header_records: list[dict[str, Any]] = []
    estimated_all_points = 0.0
    estimated_selected_points = 0.0
    for manifest_record, path in zip(copc_records, copc_paths):
        if not _is_copc(path):
            raise ValueError(f"Not a COPC source: {path}")
        with laspy.CopcReader.open(path) as reader:
            header = _validate_point_header(reader.header, path, aoi)
            tile = box(*header["bounds_l93"])
            overlap_area = tile.intersection(aoi).area
            tile_area = tile.area
            ratio = overlap_area / tile_area if tile_area else 0.0
            point_count = int(reader.header.point_count)
        class_counts = {
            int(item["id"]): int(item["count"])
            for item in manifest_record.get("observed_classifications", [])
        }
        selected_count = sum(class_counts.get(value, 0) for value in selected_classes)
        estimated_all_points += point_count * ratio
        estimated_selected_points += selected_count * ratio
        header_records.append(
            {
                "file_name": path.name,
                "byte_count": path.stat().st_size,
                "copc_vlr_observed": True,
                "header_point_count": point_count,
                "bounds_l93": header["bounds_l93"],
                "aoi_overlap_area_m2": overlap_area,
                "aoi_overlap_ratio": ratio,
                "selected_classes_full_tile_count_from_source_manifest": selected_count,
                "aoi_selected_point_count_area_scaled_estimate": selected_count * ratio,
                "count_status": "INFÉRÉ_area_scaled_not_observed_in_aoi",
            }
        )

    estimated_selected = int(round(estimated_selected_points))
    # Current in-memory implementation keeps coordinates, class, ground,
    # normalized heights and segmentation work arrays. 64 B/point is a
    # conservative planning estimate, not a measured peak.
    estimated_peak_bytes = int(math.ceil(estimated_selected_points * 64.0))
    report = {
        "schema_version": "1.0",
        "status": "dry_run_validated",
        "zone": zone,
        "network_access": "none",
        "copc_query_executed": False,
        "OBSERVÉ": {
            "source_manifest": _source_asset(source_manifest_path),
            "zone_contract": _source_asset(zone_contract_path),
            "copc_tile_count": len(copc_paths),
            "copc_headers": header_records,
            "raster_mosaic": rasters.source_metadata["mosaic"],
            "mnt_tile_count": len(mnt_paths),
            "mns_tile_count": len(mns_paths),
            "building_footprint_count_in_aoi": len(buildings),
            "hedge_axis_count_in_aoi": len(hedges),
            "building_selection": building_stats,
            "hedge_selection": hedge_stats,
        },
        "INFÉRÉ": {
            "all_class_point_count_in_aoi_area_scaled": int(
                round(estimated_all_points)
            ),
            "selected_class_3_4_5_6_point_count_in_aoi_area_scaled": estimated_selected,
            "current_pipeline_peak_memory_planning_bytes_at_64_per_selected_point": estimated_peak_bytes,
            "warning": "Point density and classes are not spatially uniform; these are planning estimates only.",
        },
        "NON_VÉRIFIÉ": [
            "Exact class counts inside the 1000 m square (no COPC query was executed).",
            "Individual tree, crown, hedge and building measurements.",
            "Peak memory and duration of exhaustive segmentation.",
            "Completeness against a field tree inventory.",
        ],
        "decision": {
            "exhaustive_run_started": False,
            "recommended_execution": "partition AOI into buffered sectors before exhaustive full-resolution segmentation",
            "reason": "avoid a monolithic multi-gigabyte point working set and preserve crowns at sector seams",
        },
    }
    _write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone-contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--buildings", type=Path, required=True)
    parser.add_argument("--hedges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zone-id", default="montmaur")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = dry_run(
        zone_contract_path=args.zone_contract,
        source_root=args.source_root,
        buildings_path=args.buildings,
        hedges_path=args.hedges,
        output_path=args.output,
        zone_id=args.zone_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
