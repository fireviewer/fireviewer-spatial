#!/usr/bin/env python3
"""Rebuild FireViewer building payloads after merging stacked source components."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import build_fireviewer_dataset_usd as builder


REPRESENTATION = "merged_source_footprint_oriented_box_gabled_roof_windows_and_doors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rebuild(source: Path, package: Path) -> dict[str, Any]:
    catalog = json.loads((source / "catalog.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source / "package-manifest.json").read_text(encoding="utf-8"))
    manifest_path = package / "manifest.json"
    acceptance_path = package / "qa" / "acceptance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if manifest.get("package_id") != source_manifest.get("package_id"):
        raise ValueError("Source and dataset package identifiers do not match")

    anchor = tuple(float(value) for value in catalog["spatial_contract"]["common_anchor_l93_metres"])
    _, boxes, _ = builder.collect_derived_geometry(source, catalog, anchor, include_source_trees=False)
    original_count = int(
        manifest["site"].get("building_component_merge", {}).get(
            "source_components", manifest["site"]["building_instances"]
        )
    )
    merged_count = len(boxes)
    visual_module_count = len(builder.visual_building_modules(boxes))
    if not (0 < merged_count < original_count):
        raise RuntimeError(f"Building merge did not reduce the component count: {original_count} -> {merged_count}")

    buildings_path = package / "site" / "payloads" / "buildings.payload.usda"
    occlusion_path = package / "site" / "payloads" / "occlusion.payload.usda"
    buildings_temporary = buildings_path.with_suffix(buildings_path.suffix + ".partial")
    occlusion_temporary = occlusion_path.with_suffix(occlusion_path.suffix + ".partial")
    builder.write_simple_buildings_payload(buildings_temporary, boxes)
    builder.write_occlusion_payload(occlusion_temporary, boxes)

    buildings_text = buildings_temporary.read_text(encoding="utf-8")
    occlusion_text = occlusion_temporary.read_text(encoding="utf-8")
    required_building_tokens = (
        f'custom int fireviewer:source_building_count = {merged_count}',
        f'custom int fireviewer:visual_building_module_count = {visual_module_count}',
        f'custom int fireviewer:render_instance_count = {visual_module_count * 7}',
        f'custom string fireviewer:representation = "{REPRESENTATION}"',
    )
    if any(token not in buildings_text for token in required_building_tokens):
        raise RuntimeError("Rebuilt building payload failed its count or representation contract")
    if f"custom int fireviewer:source_building_count = {merged_count}" in occlusion_text:
        raise RuntimeError("Unexpected building metadata leaked into the occlusion payload")

    os.replace(buildings_temporary, buildings_path)
    os.replace(occlusion_temporary, occlusion_path)

    manifest["site"]["building_instances"] = merged_count
    manifest["site"]["visual_building_modules"] = visual_module_count
    manifest["site"]["occlusion_proxies"] = merged_count
    manifest["site"]["building_representation"] = REPRESENTATION
    manifest["site"]["building_component_merge"] = {
        "source_components": original_count,
        "merged_buildings": merged_count,
        "visual_building_modules": visual_module_count,
        "collapsed_stacked_components": original_count - merged_count,
        "method": "adjacent_components_matching_oriented_xy_footprint",
    }
    acceptance["automated"]["expected_building_references"] = merged_count
    acceptance["automated"]["expected_occlusion_proxies"] = merged_count
    atomic_json(manifest_path, manifest)
    atomic_json(acceptance_path, acceptance)

    report = {
        "schema": "fireviewer.building-component-merge.v1",
        "package_id": manifest["package_id"],
        "source_components": original_count,
        "merged_buildings": merged_count,
        "visual_building_modules": visual_module_count,
        "collapsed_stacked_components": original_count - merged_count,
        "render_instances": visual_module_count * 7,
        "representation": REPRESENTATION,
        "buildings_payload_sha256": sha256_file(buildings_path),
        "occlusion_payload_sha256": sha256_file(occlusion_path),
    }
    atomic_json(package / "qa" / "building-component-merge.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--dataset-package", type=Path, required=True)
    args = parser.parse_args()
    report = rebuild(args.source_package.resolve(), args.dataset_package.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
