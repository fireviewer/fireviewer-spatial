"""Regenerate only the refined FireViewer fire-state layers in an existing package."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_fireviewer_dataset_usd import (
    HeightSampler,
    build_capture_schedule,
    canonical_json,
    load_mnt_mns_trees,
    look_at_quaternion,
    sha256_file,
    simulate_fire_spread,
    write_dataset_stage,
    write_flow_layer,
    write_propagation_sidecars,
    write_scenario_layer,
    write_fire_state,
)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


def close_enough(left: Any, right: Any, tolerance: float = 1e-3) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--site-source", type=Path, required=True)
    parser.add_argument("--vegetation-index", type=Path, required=True)
    parser.add_argument("--allow-model-update", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    source = args.site_source.resolve()
    vegetation_index = args.vegetation_index.resolve()
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    propagation = json.loads((package / "scenarios/propagation.json").read_text(encoding="utf-8"))
    catalog = json.loads((source / "catalog.json").read_text(encoding="utf-8"))
    anchor = tuple(float(value) for value in catalog["spatial_contract"]["common_anchor_l93_metres"])
    bounds = tuple(float(value) for value in catalog["bounds_l93_metres"])
    sampler = HeightSampler(source, catalog)
    try:
        trees, _ = load_mnt_mns_trees(vegetation_index, package_id=manifest["package_id"], anchor=anchor)
        result = simulate_fire_spread(
            package_id=manifest["package_id"],
            trees=trees,
            anchor_l93_m=anchor,
            source_bounds_l93_m=bounds,
            height_at=sampler.at,
            state_count=180,
        )
        ignition_ground_z = sampler.at(*result.ignition_l93_m)
    finally:
        sampler.close()
    if result.simulation_id != propagation["simulation_id"] and not args.allow_model_update:
        raise RuntimeError(f"Simulation identity changed: {result.simulation_id} != {propagation['simulation_id']}")
    if len(result.states) != 180:
        raise RuntimeError(f"Expected 180 states, received {len(result.states)}")

    expected_records = manifest["scenario"]["states"]
    final_burned_ids = result.states[-1].burned_tree_ids
    burned_tree_by_id = {tree.tree_id: tree for tree in trees if tree.tree_id in final_burned_ids}
    if len(burned_tree_by_id) != len(final_burned_ids):
        raise RuntimeError("The final burned-tree truth set does not resolve against MNT/MNS vegetation")

    temporary_root = package / "scenarios/states-refined-partial"
    if temporary_root.exists():
        raise FileExistsError(f"Refusing to reuse partial fire-state directory: {temporary_root}")
    temporary_root.mkdir(parents=True)
    refined_records: list[dict[str, Any]] = []
    surface_sampler = HeightSampler(source, catalog)
    try:
        for state, expected in zip(result.states, expected_records):
            state_id = f"state_{state.state_index:03d}"
            if expected["state_id"] != state_id:
                raise RuntimeError(f"State identity mismatch at {state_id}")
            record = write_fire_state(
                temporary_root / f"{state_id}.usda",
                result=result,
                state=state,
                state_count=180,
                anchor=anchor,
                trees=trees,
                tree_by_id=burned_tree_by_id,
                terrain_height_at=surface_sampler.at,
            )
            if not args.allow_model_update:
                for field in ("elapsed_s", "burned_area_m2", "active_front_length_m", "mean_front_spread_rate_m_s"):
                    if not close_enough(record[field], expected[field]):
                        raise RuntimeError(f"Propagation metric changed for {state_id}.{field}: {record[field]} != {expected[field]}")
                if int(record["burned_tree_count"]) != int(expected["burned_tree_count"]):
                    raise RuntimeError(f"Burned-tree truth changed for {state_id}")
            refined_records.append(record)
    finally:
        surface_sampler.close()

    if any(
        float(current["burned_area_m2"]) <= float(previous["burned_area_m2"])
        for previous, current in zip(refined_records, refined_records[1:])
    ):
        raise RuntimeError("The refined propagation does not visibly expand at every one of the 180 states")

    for record in refined_records:
        state_id = record["state_id"]
        generated = temporary_root / f"{state_id}.usda"
        target = package / "scenarios/states" / f"{state_id}.usda"
        generated.replace(target)
    temporary_root.rmdir()

    fire_target_local = [
        float(result.ignition_l93_m[0] - anchor[0]),
        float(result.ignition_l93_m[1] - anchor[1]),
        float(ignition_ground_z + 2.0),
    ]
    for camera in manifest["cameras"]["plan"]:
        position = [float(value) for value in camera["position_local_m"]]
        yaw_offset = math.radians(float(camera["fire_bearing_offset_degrees"]))
        fire_dx = fire_target_local[0] - position[0]
        fire_dy = fire_target_local[1] - position[1]
        aim = [
            position[0] + fire_dx * math.cos(yaw_offset) - fire_dy * math.sin(yaw_offset),
            position[1] + fire_dx * math.sin(yaw_offset) + fire_dy * math.cos(yaw_offset),
            fire_target_local[2],
        ]
        camera["fire_target_local_m"] = list(fire_target_local)
        camera["target_local_m"] = aim
        camera["distance_to_target_m"] = math.hypot(fire_dx, fire_dy)
        camera["orientation_quat_wxyz"] = look_at_quaternion(position, aim)

    propagation_reference = write_propagation_sidecars(package, result)
    write_scenario_layer(package / "scenarios/scenario.usda", refined_records, propagation_reference)
    write_flow_layer(package / "scenarios/flow.usda", result=result, anchor=anchor)
    write_dataset_stage(package / "dataset.usda", package_id=manifest["package_id"], anchor=anchor)
    capture_schedule = build_capture_schedule(manifest["package_id"], refined_records, manifest["cameras"]["plan"])
    schedule_path = package / "runtime/capture-schedule.json"
    write_json_atomic(schedule_path, capture_schedule)
    runtime_path = package / "runtime/runtime-contract.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["capture"]["states"] = [record["state_id"] for record in refined_records]
    runtime["capture"]["schedule_sha256"] = sha256_file(schedule_path)
    runtime["capture"]["zooms_per_view"] = int(capture_schedule["zooms_per_view"])
    runtime["capture"]["captures_per_state"] = int(capture_schedule["captures_per_state"])
    runtime["capture"]["expected_viewpoint_plans"] = int(capture_schedule["expected_viewpoint_plans"])
    runtime["capture"]["expected_capture_cases"] = int(capture_schedule["expected_capture_cases"])
    runtime["flow_contract"]["beauty_view"] = "omni_flowusd_only_truth_front_and_smoke_points_hidden_in_visual_validation_session"
    runtime["flow_contract"]["animation"] = "single_persistent_time_sampled_flow_volume_256_patch_mesh_front_plus_48_hotspots_plus_aligned_smoke_mesh"
    runtime["flow_contract"]["combustion"] = "meter_scaled_omni_flowusd_combustion_plus_aligned_buoyant_smoke_without_direct_burn"
    write_json_atomic(runtime_path, runtime)

    manifest["scenario"]["states"] = refined_records
    manifest["scenario"]["state_count"] = len(refined_records)
    manifest["scenario"]["propagation"] = propagation_reference
    manifest["dataset"]["capture_schedule"]["sha256"] = sha256_file(schedule_path)
    manifest["dataset"]["zooms_per_view"] = int(capture_schedule["zooms_per_view"])
    manifest["dataset"]["captures_per_state"] = int(capture_schedule["captures_per_state"])
    manifest["dataset"]["expected_viewpoint_plans"] = int(capture_schedule["expected_viewpoint_plans"])
    manifest["dataset"]["expected_capture_cases"] = int(capture_schedule["expected_capture_cases"])
    manifest["dataset"]["expected_positive_cases"] = int(capture_schedule["expected_positive_cases"])
    manifest["dataset"]["expected_negative_cases"] = int(capture_schedule["expected_negative_cases"])
    manifest["appearance"] = [
        "continuous_terrain_following_mesh_front_flow_combustion",
        "meter_scaled_buoyant_flow_smoke_columns",
        "forty_eight_hotspot_tongues_without_direct_smoke_or_burn",
        "forty_eight_smoke_plume_bases_xy_aligned_above_hotspots",
        "25m_distinct_state_propagation_grid",
        "terrain_conforming_burned_ground",
        "source_tree_destruction_with_burned_replacement",
    ]
    manifest["fire_visual_repaired_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(manifest_path, manifest)
    qa_path = package / "qa/acceptance.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["propagation"] = propagation_reference
    qa["automated"]["expected_viewpoint_plans"] = int(capture_schedule["expected_viewpoint_plans"])
    qa["automated"]["expected_capture_cases"] = int(capture_schedule["expected_capture_cases"])
    qa["automated"]["zooms_per_view"] = int(capture_schedule["zooms_per_view"])
    if "five_zoom_same_pose_capture_contract" not in qa["automated"]["checks"]:
        qa["automated"]["checks"].append("five_zoom_same_pose_capture_contract")
    write_json_atomic(qa_path, qa)
    print(
        json.dumps(
            {
                "status": "fire_states_refined",
                "package": str(package),
                "simulation_id": result.simulation_id,
                "state_count": len(refined_records),
                "initial_burned_area_m2": refined_records[0]["burned_area_m2"],
                "initial_burned_tree_count": refined_records[0]["burned_tree_count"],
                "final_burned_area_m2": refined_records[-1]["burned_area_m2"],
                "final_burned_tree_count": len(final_burned_ids),
                "maximum_flow_truth_sources": max(record["flow_emitter_count"] for record in refined_records),
                "native_flow_emitter_prims": 50,
                "smoke_plume_bases": 48,
                "continuous_flow_front_patches": 256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
