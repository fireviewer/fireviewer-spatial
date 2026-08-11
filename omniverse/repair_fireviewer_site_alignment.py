"""Ground visual buildings on the MNT and normalize PointInstancer orientation types."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_fireviewer_dataset_usd import (
    HeightSampler,
    canonical_json,
    collect_derived_geometry,
    visual_building_modules,
    write_simple_buildings_payload,
)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


def normalize_instancer_orientation_type(path: Path) -> bool:
    temporary = path.with_suffix(path.suffix + ".orientation-partial")
    changed = False
    with path.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            updated = line.replace("quatf[] orientations", "quath[] orientations")
            changed = changed or updated != line
            target.write(updated)
    if changed:
        temporary.replace(path)
    else:
        temporary.unlink()
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--site-source", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    source = args.site_source.resolve()
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = json.loads((source / "catalog.json").read_text(encoding="utf-8"))
    anchor = tuple(float(value) for value in catalog["spatial_contract"]["common_anchor_l93_metres"])
    sampler = HeightSampler(source, catalog)
    try:
        _, boxes, _ = collect_derived_geometry(source, catalog, anchor, include_source_trees=False)
        visual_boxes = visual_building_modules(boxes)

        def ground_at_local(local_x: float, local_y: float) -> float:
            return sampler.at(local_x + anchor[0], local_y + anchor[1])

        source_base_offsets = [float(box.minimum[2]) - ground_at_local(*box.visual_center_xy) for box in visual_boxes]
        write_simple_buildings_payload(
            package / "site/payloads/buildings.payload.usda",
            boxes,
            ground_at_local=ground_at_local,
        )
    finally:
        sampler.close()

    normalized_paths = []
    orientation_targets = [package / "site/payloads/vegetation.payload.usda"]
    orientation_targets.extend(sorted((package / "scenarios/states").glob("state_*.usda")))
    for path in orientation_targets:
        if normalize_instancer_orientation_type(path):
            normalized_paths.append(str(path.relative_to(package)).replace("\\", "/"))

    manifest["site"]["building_representation"] = "merged_source_footprint_mnt_grounded_oriented_box_gabled_roof_windows_and_doors"
    manifest["site"]["building_vertical_alignment"] = "each_visual_module_sampled_at_its_center_on_mnt"
    manifest["site_alignment_repaired_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(manifest_path, manifest)

    qa_path = package / "qa/building-component-merge.json"
    if qa_path.is_file():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa["vertical_alignment"] = "each_visual_module_sampled_at_its_center_on_mnt"
        qa["source_base_offset_before_grounding_m"] = {
            "minimum": min(source_base_offsets),
            "maximum": max(source_base_offsets),
            "mean": sum(source_base_offsets) / len(source_base_offsets),
        }
        write_json_atomic(qa_path, qa)

    print(
        json.dumps(
            {
                "status": "site_alignment_repaired",
                "package": str(package),
                "source_building_count": len(boxes),
                "visual_building_module_count": len(visual_boxes),
                "source_base_offset_before_grounding_m": {
                    "minimum": min(source_base_offsets),
                    "maximum": max(source_base_offsets),
                    "mean": sum(source_base_offsets) / len(source_base_offsets),
                },
                "orientation_type_normalized_file_count": len(normalized_paths),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
