"""Rebuild only the persistent Flow visual from validated FireViewer state fronts."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_fireviewer_dataset_usd import build_capture_schedule, canonical_json, sha256_file, write_flow_layer
from fire_spread_model import FireFrontSegment, FireSpreadDrivers, FireSpreadResult, FireSpreadState


POINT_RE = re.compile(
    r"\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)"
)


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


def mesh_points(layer_text: str, mesh_name: str) -> list[tuple[float, float, float]]:
    mesh_marker = f'def Mesh "{mesh_name}"'
    mesh_start = layer_text.find(mesh_marker)
    if mesh_start < 0:
        raise ValueError(f"USD layer has no {mesh_name} mesh")
    points_marker = "point3f[] points = ["
    points_start = layer_text.find(points_marker, mesh_start)
    if points_start < 0:
        raise ValueError(f"USD {mesh_name} mesh has no points")
    array_start = layer_text.find("[", points_start)
    array_end = layer_text.find("\n        ]", array_start)
    if array_end < 0:
        raise ValueError(f"USD {mesh_name} point array is not terminated")
    return [tuple(float(value) for value in match) for match in POINT_RE.findall(layer_text[array_start:array_end])]


def load_validated_result(package: Path, manifest: dict[str, object], propagation: dict[str, object]) -> FireSpreadResult:
    manifest_states = list(manifest["scenario"]["states"])
    propagation_states = list(propagation["states"])
    if len(manifest_states) != 180 or len(propagation_states) != 180:
        raise ValueError("Flow-only repair requires the validated 180-state incident")
    states: list[FireSpreadState] = []
    for index, (manifest_state, propagation_state) in enumerate(
        zip(manifest_states, propagation_states), start=1
    ):
        state_id = f"state_{index:03d}"
        if manifest_state["state_id"] != state_id or propagation_state["state_id"] != state_id:
            raise ValueError(f"Fire state identity mismatch at {state_id}")
        state_path = package / "scenarios" / str(manifest_state["path"])
        points = mesh_points(state_path.read_text(encoding="utf-8"), "VisibleFireFront")
        if not points or len(points) % 4:
            raise ValueError(f"VisibleFireFront topology is invalid for {state_id}")
        spread_rate = float(manifest_state["mean_front_spread_rate_m_s"])
        segments = [
            FireFrontSegment(start=points[offset], end=points[offset + 1], spread_rate_m_s=spread_rate)
            for offset in range(0, len(points), 4)
        ]
        parsed_length = sum(
            math.hypot(segment.end[0] - segment.start[0], segment.end[1] - segment.start[1])
            for segment in segments
        )
        expected_length = float(manifest_state["active_front_length_m"])
        if abs(parsed_length - expected_length) > max(0.1, expected_length * 1e-5):
            raise ValueError(
                f"VisibleFireFront length mismatch for {state_id}: {parsed_length} != {expected_length}"
            )
        states.append(
            FireSpreadState(
                state_index=index,
                elapsed_s=float(manifest_state["elapsed_s"]),
                burned_mask=np.empty((0, 0), dtype=np.bool_),
                front_segments=segments,
                burned_tree_ids=set(),
                burned_area_m2=float(manifest_state["burned_area_m2"]),
                active_front_length_m=expected_length,
                mean_front_spread_rate_m_s=spread_rate,
            )
        )
    drivers = FireSpreadDrivers(**{key: float(value) for key, value in propagation["drivers"].items()})
    return FireSpreadResult(
        simulation_id=str(propagation["simulation_id"]),
        model_metadata=dict(propagation),
        domain_bounds_l93_m=tuple(float(value) for value in propagation["domain_bounds_l93_m"]),
        ignition_l93_m=tuple(float(value) for value in propagation["ignition_l93_m"]),
        drivers=drivers,
        elevation_m=np.empty((0, 0), dtype=np.float64),
        fuel_load=np.empty((0, 0), dtype=np.float64),
        burnable_mask=np.empty((0, 0), dtype=np.bool_),
        arrival_time_s=np.empty((0, 0), dtype=np.float64),
        spread_rate_m_s=np.empty((0, 0), dtype=np.float64),
        cell_tree_ids=[],
        states=states,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest_path = package / "manifest.json"
    runtime_path = package / "runtime/runtime-contract.json"
    propagation_path = package / "scenarios/propagation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    propagation = json.loads(propagation_path.read_text(encoding="utf-8"))
    if manifest["scenario"]["propagation"]["simulation_id"] != propagation["simulation_id"]:
        raise ValueError("Manifest and propagation identities do not match")
    result = load_validated_result(package, manifest, propagation)
    temporary = package / "scenarios/flow.usda.partial"
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse partial Flow layer: {temporary}")
    write_flow_layer(temporary, result=result, anchor=(0.0, 0.0))
    temporary.replace(package / "scenarios/flow.usda")

    capture_schedule = build_capture_schedule(
        str(manifest["package_id"]),
        list(manifest["scenario"]["states"]),
        list(manifest["cameras"]["plan"]),
    )
    capture_schedule_path = package / "runtime/capture-schedule.json"
    write_json_atomic(capture_schedule_path, capture_schedule)

    runtime["flow_contract"]["beauty_view"] = (
        "omni_flowusd_only_truth_front_and_smoke_points_hidden_in_visual_validation_session"
    )
    runtime["flow_contract"]["animation"] = (
        "single_persistent_time_sampled_flow_volume_256_patch_mesh_front_plus_48_hotspots_plus_aligned_smoke_mesh"
    )
    runtime["flow_contract"]["combustion"] = (
        "meter_scaled_omni_flowusd_combustion_plus_aligned_buoyant_smoke_without_direct_burn"
    )
    runtime["capture"]["zooms_per_view"] = int(capture_schedule["zooms_per_view"])
    runtime["capture"]["captures_per_state"] = int(capture_schedule["captures_per_state"])
    runtime["capture"]["expected_viewpoint_plans"] = int(capture_schedule["expected_viewpoint_plans"])
    runtime["capture"]["expected_capture_cases"] = int(capture_schedule["expected_capture_cases"])
    runtime["capture"]["schedule_sha256"] = sha256_file(capture_schedule_path)
    runner_source = Path(__file__).resolve().parent / "run_fireviewer_replicator_dataset.py"
    runner_target = package / "runtime/run_fireviewer_replicator_dataset.py"
    writer_source = Path(__file__).resolve().parent / "fireviewer_replicator_writer.py"
    writer_target = package / "runtime/fireviewer_replicator_writer.py"
    shutil.copy2(runner_source, runner_target)
    shutil.copy2(writer_source, writer_target)
    runtime["runner_module"]["sha256"] = sha256_file(runner_target)
    runtime["writer_module"]["sha256"] = sha256_file(writer_target)
    write_json_atomic(runtime_path, runtime)
    manifest["dataset"]["zooms_per_view"] = int(capture_schedule["zooms_per_view"])
    manifest["dataset"]["captures_per_state"] = int(capture_schedule["captures_per_state"])
    manifest["dataset"]["expected_viewpoint_plans"] = int(capture_schedule["expected_viewpoint_plans"])
    manifest["dataset"]["expected_capture_cases"] = int(capture_schedule["expected_capture_cases"])
    manifest["dataset"]["expected_positive_cases"] = int(capture_schedule["expected_positive_cases"])
    manifest["dataset"]["expected_negative_cases"] = int(capture_schedule["expected_negative_cases"])
    manifest["dataset"]["capture_schedule"]["sha256"] = sha256_file(capture_schedule_path)
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
    qa["automated"]["expected_viewpoint_plans"] = int(capture_schedule["expected_viewpoint_plans"])
    qa["automated"]["expected_capture_cases"] = int(capture_schedule["expected_capture_cases"])
    qa["automated"]["zooms_per_view"] = int(capture_schedule["zooms_per_view"])
    if "five_zoom_same_pose_capture_contract" not in qa["automated"]["checks"]:
        qa["automated"]["checks"].append("five_zoom_same_pose_capture_contract")
    write_json_atomic(qa_path, qa)
    print(
        json.dumps(
            {
                "status": "flow_visual_rebuilt_from_validated_state_fronts",
                "package": str(package),
                "simulation_id": result.simulation_id,
                "states": len(result.states),
                "front_patches": 256,
                "hotspot_emitters": 48,
                "native_flow_emitter_prims": 50,
                "smoke_plume_bases": 48,
                "smoke_source_lift_m": 1.6,
                "direct_smoke": 0.32,
                "direct_burn": 0.0,
                "viewpoint_plans": 3600,
                "zooms_per_view": 5,
                "capture_cases": 18000,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
