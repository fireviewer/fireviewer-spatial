"""Run a FireViewer modular USD dataset package in Isaac Sim / Replicator.

This runner is intentionally fail-closed: rendering requires a real Isaac Sim
runtime with both Replicator and NVIDIA Flow available.  It never saves edits
to the source stage.  The generated dataset is synthetic ground truth on a
real uploaded map and includes a per-frame abstention decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from fireviewer_capture_storage import storage_profile_contract


INCIDENT_DAYS = 18
STATES_PER_DAY = 10
EXPECTED_STATES = INCIDENT_DAYS * STATES_PER_DAY
DEFAULT_SECONDS_PER_DAY = 60.0
DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

FLOW_CAPTURE_PLUME_PROFILE_ID = "wildfire_convective_plume_mid_distance_v2"
FLOW_CAPTURE_PLUME_PROFILE = {
    "schema": "fireviewer.flow-capture-profile.v1",
    "profile_id": FLOW_CAPTURE_PLUME_PROFILE_ID,
    "session_layer_only": True,
    "source_stage_modified": False,
    "intent": (
        "sustained buoyant wildfire smoke plume visible from human and aerial "
        "dataset cameras without replacing NVIDIA Flow volumetrics"
    ),
    "physics_contract": {
        "source": "continuous_aligned_smoke_mesh",
        "rise": "direct_vertical_momentum_plus_smoke_buoyancy",
        "wind": "accepted_front_wind_direction_scaled_for_plume_advection",
        "expansion": "source_divergence_plus_smoke_weighted_vorticity",
        "persistence": "reduced_smoke_damping_and_fade_with_neighbor_allocation",
        "truth_alignment": (
            "session_reprojection_of_48_hotspots_and_smoke_bases_to_active_state_truth"
        ),
    },
    "static_overrides": {
        "/World/FireScenario/FlowVisual/Simulate": {
            "blockMinLifetime": 24,
            "velocitySubSteps": 3,
        },
        "/World/FireScenario/FlowVisual/Simulate/advection": {
            "buoyancyMaxSmoke": 1.4,
            "buoyancyPerSmoke": 7.0,
        },
        "/World/FireScenario/FlowVisual/Simulate/advection/smoke": {
            "damping": 0.01,
            "fade": 0.006,
        },
        "/World/FireScenario/FlowVisual/Simulate/advection/velocity": {
            "damping": 0.004,
            "fade": 0.02,
        },
        "/World/FireScenario/FlowVisual/Simulate/advection/temperature": {
            "damping": 0.02,
            "fade": 0.18,
        },
        "/World/FireScenario/FlowVisual/Simulate/vorticity": {
            "smokeMask": 0.8,
        },
        "/World/FireScenario/FlowVisual/Simulate/summaryAllocate": {
            "smokeThreshold": 0.003,
            "speedThreshold": 0.2,
        },
        "/World/FireScenario/FlowVisual/SmokePlumeEmitter": {
            "minDistance": -0.6,
            "maxDistance": 1.6,
            "temperature": 0.45,
            "coupleRateTemperature": 2.2,
            "smoke": 0.95,
            "coupleRateSmoke": 4.0,
            "coupleRateVelocity": 4.2,
            "divergence": 0.25,
            "coupleRateDivergence": 1.5,
        },
        "/World/FireScenario/FlowVisual/Offscreen/colormap": {
            "colorScale": 3.3,
            "rgbaPoints": [
                [0.025, 0.03, 0.035, 0.04],
                [0.24, 0.26, 0.28, 0.32],
                [0.55, 0.54, 0.52, 0.58],
                [1.2, 0.10, 0.008, 0.78],
                [12.0, 2.2, 0.08, 0.92],
                [48.0, 17.0, 2.4, 0.84],
            ],
        },
        "/World/FireScenario/FlowVisual/Render/rayMarch": {
            "attenuation": 8.5,
            "colorScale": 1.25,
        },
    },
    "dynamic_smoke_velocity": {
        "horizontal_front_wind_scale": 0.65,
        "vertical_m_s": 22.0,
    },
    "state_regimes": {
        "incipient": {
            "smoke": 0.34,
            "coupleRateSmoke": 2.4,
            "minDistance": -0.25,
            "maxDistance": 0.75,
            "temperature": 0.28,
            "coupleRateTemperature": 1.5,
            "vertical_m_s": 14.0,
            "divergence": 0.08,
            "coupleRateDivergence": 0.8,
            "buoyancyPerSmoke": 4.5,
        },
        "stalled": {
            "smoke": 0.45,
            "coupleRateSmoke": 2.8,
            "minDistance": -0.35,
            "maxDistance": 1.05,
            "temperature": 0.32,
            "coupleRateTemperature": 1.7,
            "vertical_m_s": 16.0,
            "divergence": 0.12,
            "coupleRateDivergence": 1.0,
            "buoyancyPerSmoke": 5.0,
        },
        "slowed": {
            "smoke": 0.65,
            "coupleRateSmoke": 3.2,
            "minDistance": -0.45,
            "maxDistance": 1.35,
            "temperature": 0.38,
            "coupleRateTemperature": 1.9,
            "vertical_m_s": 18.0,
            "divergence": 0.18,
            "coupleRateDivergence": 1.2,
            "buoyancyPerSmoke": 6.0,
        },
        "established": {
            "smoke": 0.95,
            "coupleRateSmoke": 4.0,
            "minDistance": -0.6,
            "maxDistance": 1.6,
            "temperature": 0.45,
            "coupleRateTemperature": 2.2,
            "vertical_m_s": 22.0,
            "divergence": 0.25,
            "coupleRateDivergence": 1.5,
            "buoyancyPerSmoke": 7.0,
        },
    },
}

SKY_CAPTURE_PROFILE_ID = "clear_daylight_smoke_contrast_v2"
SKY_CAPTURE_PROFILE = {
    "schema": "fireviewer.sky-capture-profile.v1",
    "profile_id": SKY_CAPTURE_PROFILE_ID,
    "session_layer_only": True,
    "source_stage_modified": False,
    "dome_prim": "/World/SkyFill",
    "background_source": "accepted_farm_field_puresky_hdr",
    "effective_background": "session_uniform_clear_day_dome",
    "contrast_contract": (
        "suppress_cloud_texture_in_session_and_use_uniform_mid_blue_daylight_to_"
        "separate_neutral_gray_smoke"
    ),
    "overrides": {
        "inputs:texture:file": "",
        "inputs:color": [0.58, 0.72, 0.95],
        "inputs:intensity": 180.0,
        "inputs:exposure": 0.0,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_capture_schedule(package: Path, manifest: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    capture = runtime["capture"]
    incident_days = int(capture["incident_days"])
    states_per_day = int(capture["states_per_day"])
    expected_states = incident_days * states_per_day
    views_per_state = int(capture["views_per_state"])
    zooms_per_view = int(capture["zooms_per_view"])
    captures_per_state = int(capture["captures_per_state"])
    expected_viewpoint_plans = int(capture["expected_viewpoint_plans"])
    expected_capture_cases = int(capture["expected_capture_cases"])
    camera_pool_count = len(capture.get("camera_pool", [])) or int(
        manifest["cameras"]["fixed_count"]
    )
    schedule_path = package / "runtime" / str(capture["schedule_path"])
    schedule = read_json(schedule_path)
    actual_sha256 = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    manifest_reference = manifest["dataset"]["capture_schedule"]
    if actual_sha256 != capture["schedule_sha256"] or actual_sha256 != manifest_reference["sha256"]:
        raise RuntimeError("capture schedule checksum mismatch")
    if (
        schedule.get("schema") != "fireviewer.random-camera-zoom-schedule.v2"
        or schedule.get("incident_days") != incident_days
        or schedule.get("states_per_day") != states_per_day
        or schedule.get("views_per_state") != views_per_state
        or schedule.get("zooms_per_view") != zooms_per_view
        or schedule.get("captures_per_state") != captures_per_state
        or schedule.get("camera_pool_count") != camera_pool_count
        or schedule.get("expected_viewpoint_plans") != expected_viewpoint_plans
        or schedule.get("expected_capture_cases") != expected_capture_cases
        or len(schedule.get("states", [])) != expected_states
    ):
        raise RuntimeError("invalid FireViewer capture schedule contract")
    for index, state in enumerate(schedule["states"], start=1):
        expected_id = f"state_{index:03d}"
        camera_ids = state.get("camera_ids", [])
        if (
            state.get("state_id") != expected_id
            or len(camera_ids) != views_per_state
            or len(set(camera_ids)) != views_per_state
        ):
            raise RuntimeError(f"invalid scheduled views for {expected_id}")
        if (
            state.get("zoom_count_per_view") != zooms_per_view
            or state.get("capture_count") != captures_per_state
        ):
            raise RuntimeError(f"invalid scheduled zoom count for {expected_id}")
        if any(
            len(view.get("zoom_variants", [])) != zooms_per_view
            for view in state.get("views", [])
        ):
            raise RuntimeError(f"incomplete zoom variants for {expected_id}")
        positive_count = int(state.get("positive_view_count", -1))
        negative_count = int(state.get("negative_view_count", -1))
        if positive_count < 0 or negative_count < 0 or positive_count + negative_count != views_per_state:
            raise RuntimeError(f"invalid positive/negative quota for {expected_id}")
    if (
        sum(int(state["positive_view_count"]) for state in schedule["states"]) * zooms_per_view
        != int(schedule.get("expected_positive_cases", -1))
        or sum(int(state["negative_view_count"]) for state in schedule["states"]) * zooms_per_view
        != int(schedule.get("expected_negative_cases", -1))
    ):
        raise RuntimeError("capture schedule positive/negative totals do not match its state quotas")
    return schedule


def parse_state_indices(
    value: str | None,
    *,
    state_count: int,
    option_name: str = "--pilot-state-indices",
) -> list[int]:
    if value is None:
        return list(range(1, state_count + 1))
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must be a comma-separated list of integers"
        ) from exc
    if not indices:
        raise ValueError(f"{option_name} cannot be empty")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{option_name} cannot contain duplicates")
    if any(index < 1 or index > state_count for index in indices):
        raise ValueError(f"{option_name} must stay between 1 and {state_count}")
    return indices


def selected_schedule_states(
    schedule: dict[str, Any],
    state_indices: str | None,
    *,
    option_name: str = "--pilot-state-indices",
) -> list[dict[str, Any]]:
    states = list(schedule["states"])
    indices = parse_state_indices(
        state_indices,
        state_count=len(states),
        option_name=option_name,
    )
    return [states[index - 1] for index in indices]


def selected_capture_counts(
    states: list[dict[str, Any]], *, zooms_per_view: int, frames_per_state: int = 1
) -> dict[str, int]:
    multiplier = zooms_per_view * frames_per_state
    return {
        "viewpoint_plans": sum(int(state["view_count"]) for state in states),
        "capture_cases": sum(int(state["view_count"]) for state in states) * multiplier,
        "positive_cases": sum(int(state["positive_view_count"]) for state in states) * multiplier,
        "negative_cases": sum(int(state["negative_view_count"]) for state in states) * multiplier,
    }


def selected_visual_calibration_counts(
    states: list[dict[str, Any]],
    *,
    camera_ids: list[str],
    zooms_per_view: int,
    frames_per_state: int = 1,
) -> dict[str, int]:
    """Count an explicitly non-dataset visual calibration subset."""

    multiplier = zooms_per_view * frames_per_state
    positive_views = 0
    negative_views = 0
    for state in states:
        views_by_camera = {
            str(view["camera_id"]): view for view in state.get("views", [])
        }
        missing = [camera_id for camera_id in camera_ids if camera_id not in views_by_camera]
        if missing:
            raise ValueError(
                "--visual-calibration-camera-ids contains cameras outside "
                f"{state['state_id']}: {', '.join(missing)}"
            )
        for camera_id in camera_ids:
            if bool(views_by_camera[camera_id].get("expected_fire_visible")):
                positive_views += 1
            else:
                negative_views += 1
    viewpoint_plans = len(states) * len(camera_ids)
    return {
        "viewpoint_plans": viewpoint_plans,
        "capture_cases": viewpoint_plans * multiplier,
        "positive_cases": positive_views * multiplier,
        "negative_cases": negative_views * multiplier,
    }


def validate_dataset_id(value: str | None, *, fallback: str) -> str:
    dataset_id = fallback if value is None else value
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError(
            "--dataset-id must use 1-128 ASCII letters, digits, dot, underscore or hyphen"
        )
    return dataset_id


def validate_production_chunk_id(value: str | None) -> str:
    if value is None:
        raise ValueError(
            "--production-chunk-id is required with --production-state-indices"
        )
    if not DATASET_ID_RE.fullmatch(value):
        raise ValueError(
            "--production-chunk-id must use 1-128 ASCII letters, digits, dot, "
            "underscore or hyphen"
        )
    return value


def flow_capture_plume_profile_contract() -> dict[str, Any]:
    """Return a detached, hash-addressed copy of the runtime Flow profile."""

    profile = json.loads(json.dumps(FLOW_CAPTURE_PLUME_PROFILE))
    encoded = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    profile["profile_sha256"] = hashlib.sha256(encoded).hexdigest()
    return profile


def build_flow_truth_alignment_geometry(
    truth_points: list[list[float]],
    widths_m: list[float],
    *,
    hotspot_lift_m: float = 0.25,
    smoke_base_lift_above_hotspot_m: float = 1.6,
) -> dict[str, list[list[float]]]:
    """Build deterministic hotspot positions and smoke quads from active truth."""

    if len(truth_points) != 48 or len(widths_m) != len(truth_points):
        raise ValueError("Flow truth alignment requires exactly 48 points and widths")
    if hotspot_lift_m <= 0.0 or smoke_base_lift_above_hotspot_m <= 0.0:
        raise ValueError("Flow truth alignment lifts must be positive")
    hotspot_positions: list[list[float]] = []
    smoke_mesh_positions: list[list[float]] = []
    for point, width_m in zip(truth_points, widths_m):
        if len(point) != 3 or not all(math.isfinite(float(value)) for value in point):
            raise ValueError("Flow truth alignment received a non-finite 3D point")
        width = float(width_m)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("Flow truth alignment received an invalid smoke width")
        x, y, terrain_z = (float(value) for value in point)
        hotspot_z = terrain_z + float(hotspot_lift_m)
        hotspot_positions.append([x, y, hotspot_z])
        smoke_z = hotspot_z + float(smoke_base_lift_above_hotspot_m)
        half_extent = max(0.35, 0.5 * width)
        smoke_mesh_positions.extend(
            [
                [x - half_extent, y - half_extent, smoke_z],
                [x + half_extent, y - half_extent, smoke_z],
                [x + half_extent, y + half_extent, smoke_z],
                [x - half_extent, y + half_extent, smoke_z],
            ]
        )
    return {
        "hotspot_positions": hotspot_positions,
        "smoke_mesh_positions": smoke_mesh_positions,
    }


def sky_capture_profile_contract() -> dict[str, Any]:
    profile = json.loads(json.dumps(SKY_CAPTURE_PROFILE))
    encoded = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    profile["profile_sha256"] = hashlib.sha256(encoded).hexdigest()
    return profile


def load_pilot_acceptance_receipt(
    report_path: Path,
    *,
    source_package_id: str,
    source_stage_sha256: str,
    resolution_px: tuple[int, int],
    rt_subframes: int,
    flow_warmup_updates: int,
    playback_seconds_per_day: float,
    flow_profile_sha256: str,
    sky_profile_sha256: str,
    capture_storage_profile_sha256: str,
) -> dict[str, Any]:
    """Validate and reduce a passed pilot audit into an immutable receipt."""

    report_path = report_path.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = read_json(report_path)
    failures: list[str] = []
    if report.get("schema") != "fireviewer.capture-metadata-audit.v2":
        failures.append("invalid_audit_schema")
    if report.get("status") != "passed":
        failures.append("audit_not_passed")
    if int(report.get("failed_capture_count", -1)) != 0:
        failures.append("pilot_capture_failures_present")
    if int(report.get("abstention_warning_count", -1)) != 0:
        failures.append("pilot_abstention_warnings_present")
    captures = int(report.get("captures", -1))
    expected_captures = int(report.get("expected_captures", -2))
    if captures <= 0 or captures != expected_captures:
        failures.append("pilot_capture_count_mismatch")
    viewpoint_plans = int(report.get("viewpoint_plans", -1))
    expected_viewpoint_plans = int(report.get("expected_viewpoint_plans", -2))
    if viewpoint_plans <= 0 or viewpoint_plans != expected_viewpoint_plans:
        failures.append("pilot_viewpoint_plan_count_mismatch")
    if str(report.get("source_package_id")) != source_package_id:
        failures.append("pilot_source_package_mismatch")

    contract_value = report.get("run_contract")
    if not isinstance(contract_value, str) or not contract_value:
        failures.append("missing_pilot_run_contract")
        contract_path = report_path.parent / "missing-run-contract.json"
        contract: dict[str, Any] = {}
    else:
        contract_path = Path(contract_value)
        if not contract_path.is_absolute():
            contract_path = report_path.parent / contract_path
        contract_path = contract_path.resolve()
        if not contract_path.is_file():
            failures.append("pilot_run_contract_not_found")
            contract = {}
        else:
            contract = read_json(contract_path)

    if contract:
        if contract.get("schema") != "fireviewer.kit-dataset-production-run.v1":
            failures.append("invalid_pilot_run_contract_schema")
        if contract.get("run_kind") != "pilot":
            failures.append("acceptance_source_is_not_a_pilot")
        if contract.get("dataset_admissible") is not True:
            failures.append("pilot_was_not_dataset_admissible")
        if str(contract.get("source_package_id")) != source_package_id:
            failures.append("pilot_contract_source_package_mismatch")
        if str(contract.get("source_stage_sha256")) != source_stage_sha256:
            failures.append("pilot_source_stage_hash_mismatch")
        if int(contract.get("expected_capture_cases", -1)) != captures:
            failures.append("pilot_contract_capture_count_mismatch")
        if int(contract.get("expected_viewpoint_plans", -1)) != viewpoint_plans:
            failures.append("pilot_contract_viewpoint_count_mismatch")
        if int(contract.get("selected_state_count", 0)) < 1:
            failures.append("pilot_contract_has_no_selected_state")
        if list(contract.get("resolution_px", [])) != [
            int(resolution_px[0]),
            int(resolution_px[1]),
        ]:
            failures.append("pilot_resolution_mismatch")
        if int(contract.get("rt_subframes", -1)) != int(rt_subframes):
            failures.append("pilot_rt_subframes_mismatch")
        if int(contract.get("flow_warmup_updates", -1)) != int(flow_warmup_updates):
            failures.append("pilot_flow_warmup_mismatch")
        if not math.isclose(
            float(contract.get("playback_seconds_per_day", -1.0)),
            float(playback_seconds_per_day),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            failures.append("pilot_playback_cadence_mismatch")
        if str((contract.get("flow_capture_profile") or {}).get("profile_sha256")) != flow_profile_sha256:
            failures.append("pilot_flow_profile_mismatch")
        if str((contract.get("sky_capture_profile") or {}).get("profile_sha256")) != sky_profile_sha256:
            failures.append("pilot_sky_profile_mismatch")
        if str((contract.get("capture_storage_profile") or {}).get("profile_sha256")) != capture_storage_profile_sha256:
            failures.append("pilot_capture_storage_profile_mismatch")

    if failures:
        raise ValueError("invalid pilot acceptance report: " + ", ".join(failures))

    return {
        "schema": "fireviewer.accepted-pilot-receipt.v1",
        "audit_report": str(report_path),
        "audit_report_sha256": sha256_file(report_path),
        "run_contract": str(contract_path),
        "run_contract_sha256": sha256_file(contract_path),
        "pilot_dataset_id": str(report.get("dataset_id")),
        "pilot_selected_state_indices": [
            int(value) for value in contract.get("selected_state_indices", [])
        ],
        "captures": captures,
        "viewpoint_plans": viewpoint_plans,
        "flow_profile_sha256": flow_profile_sha256,
        "sky_profile_sha256": sky_profile_sha256,
        "capture_storage_profile_sha256": capture_storage_profile_sha256,
        "source_stage_sha256": source_stage_sha256,
    }


def classify_fire_visibility_regime(
    *, fire_elapsed_s: float, burned_area_m2: float, mean_front_spread_rate_m_s: float
) -> dict[str, Any]:
    """Classify legitimate low-signal fire states without hiding QA failures."""

    elapsed = float(fire_elapsed_s)
    burned_area = float(burned_area_m2)
    spread_rate = float(mean_front_spread_rate_m_s)
    if elapsed <= 60.0 or burned_area <= 1_000.0:
        regime = "incipient"
        reason = "incident_start_small_burned_area"
    elif spread_rate <= 0.01:
        regime = "stalled"
        reason = "front_progression_plateau"
    elif spread_rate < 0.08:
        regime = "slowed"
        reason = "front_progression_slowed"
    else:
        regime = "established"
        reason = "established_active_front"
    return {
        "regime": regime,
        "reason": reason,
        "low_signal_state_allowed": regime != "established",
        "fire_elapsed_s": elapsed,
        "burned_area_m2": burned_area,
        "mean_front_spread_rate_m_s": spread_rate,
    }


def classify_capture_visibility_acceptance(view: dict[str, Any]) -> dict[str, Any]:
    """Describe whether a small target is expected and valid for training."""

    expected_fire = bool(view.get("expected_fire_visible"))
    regime = str(view.get("fire_visibility_regime") or "established")
    distance_m = float(view.get("nearest_flame_distance_m") or view.get("distance_to_target_m") or 0.0)
    zoom_multiplier = float(view.get("zoom_multiplier") or 1.0)
    reasons: list[str] = []
    if expected_fire and regime != "established":
        reasons.append(f"fire_state_{regime}")
    if expected_fire and distance_m >= 1_500.0:
        reasons.append("long_range_small_angular_target")
    if expected_fire and zoom_multiplier < 1.0:
        reasons.append("wide_frame_small_target")
    if not expected_fire:
        acceptance_class = "negative_context"
    elif reasons:
        acceptance_class = "valid_low_or_partial_signal_positive"
    else:
        acceptance_class = "standard_positive"
    return {
        "schema": "fireviewer.capture-visibility-acceptance.v1",
        "acceptance_class": acceptance_class,
        "low_signal_allowed": bool(expected_fire and reasons),
        "low_signal_reasons": reasons,
        "fire_visibility_regime": regime,
        "distance_to_nearest_flame_m": distance_m,
        "zoom_multiplier": zoom_multiplier,
        "minimum_positive_projected_pixels": 1 if expected_fire else 0,
        "qa_contract": (
            "retain_legitimate_small_targets_but_reject_missing_or_inconsistent_truth"
        ),
        "training_role": (
            "hard_positive" if expected_fire and reasons else (
                "positive" if expected_fire else "negative_context"
            )
        ),
    }


def parse_camera_priority(
    value: str | None, *, available_camera_ids: set[str]
) -> list[str]:
    if value is None:
        return []
    camera_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not camera_ids:
        raise ValueError("--pilot-camera-priority cannot be empty")
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("--pilot-camera-priority cannot contain duplicates")
    missing = sorted(set(camera_ids) - available_camera_ids)
    if missing:
        raise ValueError(
            "--pilot-camera-priority contains cameras outside the selected pilot: "
            + ", ".join(missing)
        )
    return camera_ids


def parse_visual_calibration_camera_ids(
    value: str | None, *, available_camera_ids: set[str]
) -> list[str]:
    if value is None:
        return []
    camera_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not camera_ids:
        raise ValueError("--visual-calibration-camera-ids cannot be empty")
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("--visual-calibration-camera-ids cannot contain duplicates")
    missing = sorted(set(camera_ids) - available_camera_ids)
    if missing:
        raise ValueError(
            "--visual-calibration-camera-ids contains cameras outside the selected pilot: "
            + ", ".join(missing)
        )
    return camera_ids


def prioritize_camera_ids(camera_ids: list[str], priority: list[str]) -> list[str]:
    """Reorder a pilot for fast visual QA without changing its capture content."""

    ordered_priority = [camera_id for camera_id in priority if camera_id in camera_ids]
    priority_set = set(ordered_priority)
    return ordered_priority + [
        camera_id for camera_id in camera_ids if camera_id not in priority_set
    ]


def resolve_state_selection(
    *,
    pilot_state_indices: str | None,
    production_state_indices: str | None,
) -> tuple[str | None, str, str]:
    if pilot_state_indices is not None and production_state_indices is not None:
        raise ValueError(
            "--pilot-state-indices and --production-state-indices are mutually exclusive"
        )
    if production_state_indices is not None:
        return (
            production_state_indices,
            "--production-state-indices",
            "production_chunk",
        )
    if pilot_state_indices is not None:
        return pilot_state_indices, "--pilot-state-indices", "pilot"
    return None, "--pilot-state-indices", "full"


def resolve_production_chunk_id(
    *,
    production_state_indices: str | None,
    production_chunk_id: str | None,
    pilot_acceptance_report: Path | None,
) -> str | None:
    if production_state_indices is None:
        if production_chunk_id is not None:
            raise ValueError(
                "--production-chunk-id requires --production-state-indices"
            )
        if pilot_acceptance_report is not None:
            raise ValueError(
                "--pilot-acceptance-report requires --production-state-indices"
            )
        return None
    chunk_id = validate_production_chunk_id(production_chunk_id)
    if pilot_acceptance_report is None:
        raise ValueError(
            "--pilot-acceptance-report is required with --production-state-indices"
        )
    return chunk_id


def dry_run(
    stage_path: Path,
    *,
    dataset_id: str | None = None,
    pilot_state_indices: str | None = None,
    production_state_indices: str | None = None,
    production_chunk_id: str | None = None,
    pilot_acceptance_report: Path | None = None,
    visual_calibration_camera_ids: str | None = None,
    resolution: str = "1280x720",
    rt_subframes: int = 8,
    seconds_per_day: float = DEFAULT_SECONDS_PER_DAY,
    flow_warmup_updates: int = 180,
    render_product_batch_size: int = 1,
) -> int:
    if render_product_batch_size < 1:
        raise ValueError("--render-product-batch-size must be positive")
    if rt_subframes < 1 or seconds_per_day <= 0 or flow_warmup_updates < 1:
        raise ValueError(
            "--rt-subframes, --seconds-per-day and --flow-warmup-updates must be positive"
        )
    resolution_px = parse_resolution(resolution)
    package = stage_path.parent
    manifest = read_json(package / "manifest.json")
    runtime = read_json(package / "runtime/runtime-contract.json")
    expected = int(runtime["capture"]["expected_capture_cases"])
    schedule = load_capture_schedule(package, manifest, runtime)
    production_dataset_id = validate_dataset_id(dataset_id, fallback=str(manifest["package_id"]))
    state_selector, state_selector_option, selection_run_kind = resolve_state_selection(
        pilot_state_indices=pilot_state_indices,
        production_state_indices=production_state_indices,
    )
    resolved_production_chunk_id = resolve_production_chunk_id(
        production_state_indices=production_state_indices,
        production_chunk_id=production_chunk_id,
        pilot_acceptance_report=pilot_acceptance_report,
    )
    selected_states = selected_schedule_states(
        schedule,
        state_selector,
        option_name=state_selector_option,
    )
    all_scheduled_camera_ids = sorted(
        {
            str(camera_id)
            for state in selected_states
            for camera_id in state["camera_ids"]
        }
    )
    if visual_calibration_camera_ids is not None and pilot_state_indices is None:
        raise ValueError(
            "--visual-calibration-camera-ids is allowed only with --pilot-state-indices"
        )
    calibration_camera_ids = parse_visual_calibration_camera_ids(
        visual_calibration_camera_ids,
        available_camera_ids=set(all_scheduled_camera_ids),
    )
    selected_counts = (
        selected_visual_calibration_counts(
            selected_states,
            camera_ids=calibration_camera_ids,
            zooms_per_view=int(schedule["zooms_per_view"]),
        )
        if calibration_camera_ids
        else selected_capture_counts(
            selected_states,
            zooms_per_view=int(schedule["zooms_per_view"]),
        )
    )
    run_kind = (
        "visual_calibration"
        if calibration_camera_ids
        else selection_run_kind
    )
    if calibration_camera_ids and production_state_indices is not None:
        raise ValueError(
            "--visual-calibration-camera-ids cannot be combined with "
            "--production-state-indices"
        )
    flow_profile = flow_capture_plume_profile_contract()
    sky_profile = sky_capture_profile_contract()
    capture_storage_profile = storage_profile_contract()
    source_stage_sha256 = sha256_file(stage_path)
    pilot_acceptance = (
        load_pilot_acceptance_receipt(
            pilot_acceptance_report,
            source_package_id=str(manifest["package_id"]),
            source_stage_sha256=source_stage_sha256,
            resolution_px=resolution_px,
            rt_subframes=rt_subframes,
            flow_warmup_updates=flow_warmup_updates,
            playback_seconds_per_day=seconds_per_day,
            flow_profile_sha256=str(flow_profile["profile_sha256"]),
            sky_profile_sha256=str(sky_profile["profile_sha256"]),
            capture_storage_profile_sha256=str(
                capture_storage_profile["profile_sha256"]
            ),
        )
        if pilot_acceptance_report is not None
        else None
    )
    expected_states = int(runtime["capture"]["incident_days"]) * int(
        runtime["capture"]["states_per_day"]
    )
    expected_camera_count = len(runtime["capture"].get("camera_pool", [])) or int(
        manifest["cameras"]["fixed_count"]
    )
    if (
        manifest["scenario"]["state_count"] != expected_states
        or manifest["cameras"]["fixed_count"] != expected_camera_count
        or expected != int(schedule["expected_capture_cases"])
    ):
        raise SystemExit("invalid FireViewer dataset contract")
    print(json.dumps({"status": "dry_run_passed", "stage": str(stage_path), "dataset_id": production_dataset_id, "source_package_id": manifest["package_id"], "source_stage_sha256": source_stage_sha256, "run_kind": run_kind, "production_chunk_id": resolved_production_chunk_id, "pilot_acceptance": pilot_acceptance, "capture_storage_profile": capture_storage_profile, "required_extensions": runtime["required_extensions"], "incident_days": schedule["incident_days"], "states": len(schedule["states"]), "selected_state_indices": [int(state["global_state_index"]) for state in selected_states], "selected_states": len(selected_states), "camera_pool": schedule["camera_pool_count"], "visual_calibration_camera_ids": calibration_camera_ids, "render_product_batch_size": int(render_product_batch_size), "kit_session_contract": "single_persistent_session_single_stage_load", "views_per_state": schedule["views_per_state"], "zooms_per_view": schedule["zooms_per_view"], "captures_per_state": schedule["captures_per_state"], "expected_viewpoint_plans": selected_counts["viewpoint_plans"], "expected_capture_cases": selected_counts["capture_cases"], "expected_positive_cases": selected_counts["positive_cases"], "expected_negative_cases": selected_counts["negative_cases"], "full_expected_capture_cases": expected}, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", type=Path, required=True, help="dataset.usda in a generated package")
    result.add_argument("--output-root", type=Path, help="Dataset output directory; required unless --dry-run")
    result.add_argument("--dataset-id", help="Stable production dataset identifier used in output and metadata")
    result.add_argument(
        "--pilot-state-indices",
        help="Comma-separated 1-based schedule indices; when set, capture only this controlled pilot subset",
    )
    result.add_argument(
        "--production-state-indices",
        help=(
            "Comma-separated 1-based schedule indices for one dataset-admissible "
            "production chunk. Requires an accepted pilot audit receipt."
        ),
    )
    result.add_argument(
        "--production-chunk-id",
        help="Stable, unique identifier for a --production-state-indices chunk",
    )
    result.add_argument(
        "--pilot-acceptance-report",
        type=Path,
        help=(
            "Passed fireviewer.capture-metadata-audit.v2 report authorizing a "
            "production chunk with identical stage and render profiles"
        ),
    )
    result.add_argument(
        "--pilot-camera-priority",
        help=(
            "Optional comma-separated camera order used only to present selected pilot "
            "views earlier; it never filters or changes the capture quota."
        ),
    )
    result.add_argument(
        "--visual-calibration-camera-ids",
        help=(
            "Comma-separated camera subset for a non-dataset visual calibration run. "
            "Requires --pilot-state-indices and preserves all five zooms per selected camera."
        ),
    )
    result.add_argument("--resolution", default="1280x720")
    result.add_argument("--frames-per-state", type=int, default=1)
    result.add_argument("--rt-subframes", type=int, default=8)
    result.add_argument(
        "--render-product-batch-size",
        type=int,
        default=1,
        help=(
            "Maximum number of RTX render products kept active at once. "
            "The same Kit session and loaded stage are reused across all batches."
        ),
    )
    result.add_argument(
        "--seconds-per-day",
        type=float,
        default=DEFAULT_SECONDS_PER_DAY,
        help="Live playback cadence. The FireViewer contract uses 60 real seconds per simulated day.",
    )
    result.add_argument(
        "--flow-warmup-updates",
        type=int,
        default=180,
        help=(
            "Deterministic fixed-state Flow update count after clearing each independently "
            "captured state. The timeline remains frozen at the exact state time."
        ),
    )
    result.add_argument(
        "--simulation-only",
        action="store_true",
        help="Play and validate all 180 fire states without creating render products or writing captures.",
    )
    result.add_argument(
        "--progress-path",
        type=Path,
        help="Optional JSON progress file for the simulation-only or capture run.",
    )
    result.add_argument(
        "--kit-cache-root",
        type=Path,
        help="Persistent Kit/UJITSO/OptiX cache shared by visible production runs.",
    )
    result.add_argument("--headless", action="store_true")
    result.add_argument("--visible-camera", default="CAM_11", help="Camera shown in the native Kit viewport when not headless")
    result.add_argument(
        "--visible-view-updates",
        type=int,
        default=3,
        help="Viewport update count after every camera/zoom switch in visible production mode.",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("--resolution must use WIDTHxHEIGHT, for example 1280x720") from exc
    if width < 16 or height < 16:
        raise ValueError("--resolution dimensions must each be at least 16")
    return width, height


def split_camera_batches(camera_ids: list[str], batch_size: int) -> list[list[str]]:
    """Split scheduled cameras without changing order or duplicating a view."""

    if batch_size < 1:
        raise ValueError("--render-product-batch-size must be positive")
    return [
        camera_ids[offset : offset + batch_size]
        for offset in range(0, len(camera_ids), batch_size)
    ]


def require_flow_extension(app: Any) -> None:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    enabled = bool(manager.is_extension_enabled("omni.flowusd"))
    print(json.dumps({"status": "flow_extension_probe", "enabled": enabled}, sort_keys=True), flush=True)
    if not enabled:
        manager.set_extension_enabled_immediate("omni.flowusd", True)
        enabled = bool(manager.is_extension_enabled("omni.flowusd"))
        print(json.dumps({"status": "flow_extension_enable_requested", "enabled": enabled}, sort_keys=True), flush=True)
    if not enabled:
        raise RuntimeError(
            "omni.flowusd is not enabled. Refusing capture because FlowClose and SmokeMidDistance would be absent. "
            "Run this package with an Isaac/Kit experience that includes and enables omni.flowusd."
        )
    app.update()


def configure_daylight_renderer() -> None:
    """Use the authored HDR dome as the visible daylight background."""
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/rtx/domeLight/enabled", True)
    settings.set("/rtx/background/source/type", 0)
    settings.set("/rtx/post/backgroundZeroAlpha/enabled", False)
    print(
        json.dumps(
            {
                "status": "hdr_daylight_renderer_ready",
                "background_source": "authored_dome_light",
                "dome_light_enabled": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def configure_flow_renderer() -> None:
    """Enable native Flow integration explicitly for the active RTX renderer."""
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/rtx/flow/enabled", True)
    settings.set("/rtx/flow/maxBlocks", 16384)
    settings.set("/rtx/flow/pathTracingEnabled", True)
    settings.set("/rtx/flow/rayTracedReflectionsEnabled", True)
    settings.set("/rtx/flow/rayTracedShadowsEnabled", True)
    settings.set("/rtx/flow/rayTracedTranslucencyEnabled", True)
    print(json.dumps({"status": "flow_renderer_ready", "rtx_flow_enabled": True, "maximum_blocks": 16384}, sort_keys=True), flush=True)


def apply_burned_tree_destruction(stage: Any) -> int:
    """Hide source vegetation IDs consumed by the selected fire state."""
    from pxr import Sdf, UsdGeom, Vt

    source_path = "/World/FireViewerSite/Vegetation/Trees"
    burned_path = "/World/FireScenario/Truth3D/BurnedVegetation/Trees"
    source_prim = stage.GetPrimAtPath(source_path)
    burned_prim = stage.GetPrimAtPath(burned_path)
    if not source_prim or not source_prim.IsA(UsdGeom.PointInstancer):
        raise RuntimeError(f"source vegetation point instancer is missing: {source_path}")
    burned_ids_value = (
        burned_prim.GetAttribute("fireviewer:burned_tree_ids").Get()
        if burned_prim and burned_prim.HasAttribute("fireviewer:burned_tree_ids")
        else burned_prim.GetAttribute("ids").Get()
        if burned_prim
        else None
    )
    burned_ids = [int(value) for value in burned_ids_value] if burned_ids_value is not None else []
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        with Sdf.ChangeBlock():
            UsdGeom.PointInstancer(source_prim).CreateInvisibleIdsAttr().Set(Vt.Int64Array(burned_ids))
    finally:
        stage.SetEditTarget(previous_edit_target)
    print(
        json.dumps(
            {
                "status": "tree_destruction_applied",
                "source_instancer": source_path,
                "burned_instancer": burned_path,
                "hidden_source_tree_count": len(burned_ids),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return len(burned_ids)


def hide_truth_proxies_for_beauty(stage: Any) -> list[str]:
    """Keep truth geometry queryable while excluding all non-photoreal proxies from beauty."""
    from pxr import Sdf, UsdGeom

    paths = (
        "/World/FireScenario/Truth3D/BurnedPerimeter",
        "/World/FireScenario/Truth3D/VisibleFireFront",
        "/World/FireScenario/Truth3D/SmokeSources",
        "/World/FireScenario/Truth3D/BurnedVegetation/Trees",
    )
    hidden: list[str] = []
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        with Sdf.ChangeBlock():
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsA(UsdGeom.Imageable):
                    raise RuntimeError(f"beauty truth proxy is missing: {path}")
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                hidden.append(path)
    finally:
        stage.SetEditTarget(previous_edit_target)
    print(json.dumps({"status": "beauty_truth_proxies_hidden", "paths": hidden}, sort_keys=True), flush=True)
    return hidden


def _flow_value_to_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    asset_path = getattr(value, "path", None)
    if asset_path is not None:
        return str(asset_path)
    try:
        return [_flow_value_to_json(item) for item in value]
    except TypeError:
        return str(value)


def apply_flow_capture_plume_profile(stage: Any) -> dict[str, Any]:
    """Author a traceable convective-plume profile in the anonymous session layer.

    The accepted USD and its Flow layer stay immutable.  The overrides only
    affect the current capture process and are reapplied deterministically for
    every pilot or production run.
    """

    from pxr import Gf, Sdf, Vt

    contract = flow_capture_plume_profile_contract()
    profile_overrides = json.loads(
        json.dumps(contract["static_overrides"], ensure_ascii=False)
    )
    front = stage.GetPrimAtPath(
        "/World/FireScenario/FlowVisual/FrontRibbonEmitter"
    )
    smoke = stage.GetPrimAtPath(
        "/World/FireScenario/FlowVisual/SmokePlumeEmitter"
    )
    if (
        not front
        or front.GetTypeName() != "FlowEmitterMesh"
        or not smoke
        or smoke.GetTypeName() != "FlowEmitterMesh"
    ):
        raise RuntimeError("cannot apply plume profile without both Flow mesh emitters")
    front_velocity = front.GetAttribute("velocity").Get()
    if front_velocity is None or len(front_velocity) != 3:
        raise RuntimeError("front Flow emitter has no accepted wind velocity")
    velocity_contract = contract["dynamic_smoke_velocity"]
    smoke_velocity = Gf.Vec3f(
        float(front_velocity[0])
        * float(velocity_contract["horizontal_front_wind_scale"]),
        float(front_velocity[1])
        * float(velocity_contract["horizontal_front_wind_scale"]),
        float(velocity_contract["vertical_m_s"]),
    )
    profile_overrides[
        "/World/FireScenario/FlowVisual/SmokePlumeEmitter"
    ]["velocity"] = smoke_velocity

    source_values: dict[str, dict[str, Any]] = {}
    effective_values: dict[str, dict[str, Any]] = {}
    previous_edit_target = stage.GetEditTarget()
    session_layer = stage.GetSessionLayer()
    stage.SetEditTarget(session_layer)
    try:
        with Sdf.ChangeBlock():
            for prim_path, attributes in profile_overrides.items():
                prim = stage.GetPrimAtPath(prim_path)
                if not prim:
                    raise RuntimeError(
                        f"Flow plume profile prim is missing: {prim_path}"
                    )
                source_values[prim_path] = {}
                effective_values[prim_path] = {}
                for attribute_name, requested_value in attributes.items():
                    attribute = prim.GetAttribute(attribute_name)
                    if not attribute:
                        raise RuntimeError(
                            "Flow plume profile attribute is missing: "
                            f"{prim_path}.{attribute_name}"
                        )
                    source_values[prim_path][attribute_name] = _flow_value_to_json(
                        attribute.Get()
                    )
                    authored_value: Any = requested_value
                    if attribute_name == "rgbaPoints":
                        authored_value = Vt.Vec4fArray(
                            [Gf.Vec4f(*point) for point in requested_value]
                        )
                    if not attribute.Set(authored_value):
                        raise RuntimeError(
                            "Flow plume profile could not author: "
                            f"{prim_path}.{attribute_name}"
                        )
                    effective_values[prim_path][attribute_name] = _flow_value_to_json(
                        attribute.Get()
                    )
    finally:
        stage.SetEditTarget(previous_edit_target)

    smoke_path = "/World/FireScenario/FlowVisual/SmokePlumeEmitter"
    advection_path = "/World/FireScenario/FlowVisual/Simulate/advection"
    smoke_channel_path = advection_path + "/smoke"
    vorticity_path = "/World/FireScenario/FlowVisual/Simulate/vorticity"
    smoke_values = effective_values[smoke_path]
    velocity_values = [float(value) for value in smoke_values["velocity"]]
    applied_parameter_count = sum(
        len(attributes) for attributes in effective_values.values()
    )
    receipt = {
        "schema": "fireviewer.flow-capture-profile-receipt.v1",
        "profile_id": contract["profile_id"],
        "profile_sha256": contract["profile_sha256"],
        "session_layer_only": True,
        "source_stage_modified": False,
        "session_layer_identifier": str(session_layer.identifier),
        "applied_parameter_count": applied_parameter_count,
        "smoke_source_value": float(smoke_values["smoke"]),
        "smoke_couple_rate": float(smoke_values["coupleRateSmoke"]),
        "smoke_source_extent_m": float(smoke_values["maxDistance"])
        - float(smoke_values["minDistance"]),
        "smoke_velocity_local_m_s": velocity_values,
        "plume_vertical_velocity_m_s": velocity_values[2],
        "smoke_buoyancy_per_unit": float(
            effective_values[advection_path]["buoyancyPerSmoke"]
        ),
        "smoke_damping": float(effective_values[smoke_channel_path]["damping"]),
        "smoke_fade_per_s": float(effective_values[smoke_channel_path]["fade"]),
        "smoke_vorticity_mask": float(
            effective_values[vorticity_path]["smokeMask"]
        ),
    }
    if (
        receipt["smoke_source_value"] < 0.8
        or receipt["smoke_source_extent_m"] < 2.0
        or receipt["plume_vertical_velocity_m_s"] < 18.0
        or receipt["smoke_buoyancy_per_unit"] < 6.0
        or receipt["smoke_fade_per_s"] > 0.01
    ):
        raise RuntimeError(f"Flow plume profile failed its physical gate: {receipt}")
    print(
        json.dumps(
            {
                "status": "flow_capture_plume_profile_applied",
                **receipt,
                "source_values": source_values,
                "effective_values": effective_values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def configure_flow_capture_plume_for_state(
    stage: Any,
    *,
    state_metrics: dict[str, Any],
    base_profile_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Scale plume production to the incident phase while preserving physics."""

    from pxr import Gf, Sdf

    classification = classify_fire_visibility_regime(
        fire_elapsed_s=float(state_metrics["fire_elapsed_s"]),
        burned_area_m2=float(state_metrics["burned_area_m2"]),
        mean_front_spread_rate_m_s=float(
            state_metrics["mean_front_spread_rate_m_s"]
        ),
    )
    contract = flow_capture_plume_profile_contract()
    regime_values = contract["state_regimes"][classification["regime"]]
    smoke_path = "/World/FireScenario/FlowVisual/SmokePlumeEmitter"
    advection_path = "/World/FireScenario/FlowVisual/Simulate/advection"
    smoke = stage.GetPrimAtPath(smoke_path)
    advection = stage.GetPrimAtPath(advection_path)
    if not smoke or not advection:
        raise RuntimeError("cannot scale Flow plume for state: required prim is missing")
    base_velocity = list(base_profile_receipt["smoke_velocity_local_m_s"])
    state_velocity = Gf.Vec3f(
        float(base_velocity[0]),
        float(base_velocity[1]),
        float(regime_values["vertical_m_s"]),
    )
    emitter_values = {
        "smoke": float(regime_values["smoke"]),
        "coupleRateSmoke": float(regime_values["coupleRateSmoke"]),
        "minDistance": float(regime_values["minDistance"]),
        "maxDistance": float(regime_values["maxDistance"]),
        "temperature": float(regime_values["temperature"]),
        "coupleRateTemperature": float(regime_values["coupleRateTemperature"]),
        "velocity": state_velocity,
        "divergence": float(regime_values["divergence"]),
        "coupleRateDivergence": float(regime_values["coupleRateDivergence"]),
    }
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        with Sdf.ChangeBlock():
            for attribute_name, value in emitter_values.items():
                attribute = smoke.GetAttribute(attribute_name)
                if not attribute or not attribute.Set(value):
                    raise RuntimeError(
                        f"cannot apply state plume value: {smoke_path}.{attribute_name}"
                    )
            buoyancy = advection.GetAttribute("buoyancyPerSmoke")
            if not buoyancy or not buoyancy.Set(
                float(regime_values["buoyancyPerSmoke"])
            ):
                raise RuntimeError("cannot apply state smoke buoyancy")
    finally:
        stage.SetEditTarget(previous_edit_target)
    receipt = {
        **base_profile_receipt,
        "state_visibility_regime": classification,
        "smoke_source_value": float(smoke.GetAttribute("smoke").Get()),
        "smoke_couple_rate": float(smoke.GetAttribute("coupleRateSmoke").Get()),
        "smoke_source_extent_m": float(smoke.GetAttribute("maxDistance").Get())
        - float(smoke.GetAttribute("minDistance").Get()),
        "smoke_velocity_local_m_s": [
            float(value) for value in smoke.GetAttribute("velocity").Get()
        ],
        "plume_vertical_velocity_m_s": float(
            smoke.GetAttribute("velocity").Get()[2]
        ),
        "smoke_buoyancy_per_unit": float(
            advection.GetAttribute("buoyancyPerSmoke").Get()
        ),
        "state_scaling_contract": "incident_phase_controls_plume_mass_extent_rise_and_expansion_v1",
    }
    print(
        json.dumps(
            {"status": "flow_capture_plume_state_scaled", **receipt},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def apply_sky_capture_visibility_profile(stage: Any) -> dict[str, Any]:
    """Use a clear capture sky without editing the accepted HDR or source USD."""

    from pxr import Gf, Sdf

    contract = sky_capture_profile_contract()
    dome = stage.GetPrimAtPath(str(contract["dome_prim"]))
    if not dome or dome.GetTypeName() != "DomeLight":
        raise RuntimeError("accepted daylight DomeLight is missing")
    source_values: dict[str, Any] = {}
    effective_values: dict[str, Any] = {}
    previous_edit_target = stage.GetEditTarget()
    session_layer = stage.GetSessionLayer()
    stage.SetEditTarget(session_layer)
    try:
        with Sdf.ChangeBlock():
            for name, requested in contract["overrides"].items():
                attribute = dome.GetAttribute(name)
                if not attribute:
                    raise RuntimeError(f"DomeLight attribute is missing: {name}")
                source_values[name] = _flow_value_to_json(attribute.Get())
                value: Any = requested
                if name == "inputs:color":
                    value = Gf.Vec3f(*requested)
                elif name == "inputs:texture:file":
                    value = Sdf.AssetPath(str(requested))
                if not attribute.Set(value):
                    raise RuntimeError(f"cannot apply DomeLight override: {name}")
                effective_values[name] = _flow_value_to_json(attribute.Get())
    finally:
        stage.SetEditTarget(previous_edit_target)
    receipt = {
        "schema": "fireviewer.sky-capture-profile-receipt.v1",
        "profile_id": contract["profile_id"],
        "profile_sha256": contract["profile_sha256"],
        "session_layer_only": True,
        "source_stage_modified": False,
        "background_source": contract["background_source"],
        "effective_background": contract["effective_background"],
        "contrast_contract": contract["contrast_contract"],
        "dome_prim": contract["dome_prim"],
        "source_texture_asset": source_values["inputs:texture:file"],
        "effective_texture_asset": effective_values["inputs:texture:file"],
        "cloud_texture_suppressed": effective_values["inputs:texture:file"] == "",
        "effective_color_rgb": effective_values["inputs:color"],
        "effective_intensity": float(effective_values["inputs:intensity"]),
        "effective_exposure": float(effective_values["inputs:exposure"]),
    }
    print(
        json.dumps(
            {
                "status": "sky_capture_visibility_profile_applied",
                **receipt,
                "source_values": source_values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def _ordered_flow_sphere_emitters(root: Any) -> list[Any]:
    emitters = [
        prim for prim in root.GetChildren() if prim.GetTypeName() == "FlowEmitterSphere"
    ]
    try:
        return sorted(
            emitters,
            key=lambda prim: int(str(prim.GetName()).rsplit("_", 1)[1]),
        )
    except (IndexError, ValueError) as exc:
        raise RuntimeError("Flow hotspot emitters do not use the accepted numeric order") from exc


def align_flow_emitters_to_active_truth(
    stage: Any, *, target_time_s: float
) -> dict[str, Any]:
    """Repair per-state Flow/truth drift in the anonymous session layer only."""

    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    root = stage.GetPrimAtPath("/World/FireScenario/FlowVisual")
    smoke_mesh = stage.GetPrimAtPath(
        "/World/FireScenario/FlowVisual/SmokePlumeEmitter"
    )
    truth_prim = stage.GetPrimAtPath(
        "/World/FireScenario/Truth3D/SmokeSources"
    )
    if not root or not smoke_mesh or not truth_prim or not truth_prim.IsA(UsdGeom.Points):
        raise RuntimeError("Flow truth alignment inputs are missing")
    sphere_emitters = _ordered_flow_sphere_emitters(root)
    truth_schema = UsdGeom.Points(truth_prim)
    truth_value = truth_schema.GetPointsAttr().Get()
    widths_value = truth_schema.GetWidthsAttr().Get()
    if truth_value is None or widths_value is None:
        raise RuntimeError("active smoke truth points or widths are missing")
    truth_points = [
        [float(point[0]), float(point[1]), float(point[2])]
        for point in truth_value
    ]
    widths_m = [float(width) for width in widths_value]
    geometry = build_flow_truth_alignment_geometry(truth_points, widths_m)
    if len(sphere_emitters) != len(geometry["hotspot_positions"]):
        raise RuntimeError("Flow hotspot count does not match active truth alignment geometry")
    time_code = Usd.TimeCode(
        float(target_time_s) * float(stage.GetTimeCodesPerSecond())
    )
    source_index_xy_errors: list[float] = []
    for emitter, truth_point in zip(sphere_emitters, truth_points):
        source_position = emitter.GetAttribute("position").Get(time_code)
        if source_position is None:
            raise RuntimeError(f"Flow hotspot has no position: {emitter.GetPath()}")
        source_index_xy_errors.append(
            math.hypot(
                float(source_position[0]) - truth_point[0],
                float(source_position[1]) - truth_point[1],
            )
        )
    previous_edit_target = stage.GetEditTarget()
    session_layer = stage.GetSessionLayer()
    stage.SetEditTarget(session_layer)
    try:
        with Sdf.ChangeBlock():
            for emitter, position in zip(
                sphere_emitters, geometry["hotspot_positions"]
            ):
                if not emitter.GetAttribute("position").Set(
                    Gf.Vec3f(*position), time_code
                ):
                    raise RuntimeError(
                        f"cannot align Flow hotspot in session: {emitter.GetPath()}"
                    )
            smoke_positions = Vt.Vec3fArray(
                [Gf.Vec3f(*position) for position in geometry["smoke_mesh_positions"]]
            )
            if not smoke_mesh.GetAttribute("meshPositions").Set(
                smoke_positions, time_code
            ):
                raise RuntimeError("cannot align Flow smoke mesh in session")
    finally:
        stage.SetEditTarget(previous_edit_target)
    receipt = {
        "schema": "fireviewer.flow-truth-session-alignment.v1",
        "alignment_contract": (
            "48_hotspots_at_active_truth_plus_0_25m_and_smoke_bases_plus_1_60m"
        ),
        "session_layer_only": True,
        "source_stage_modified": False,
        "session_layer_identifier": str(session_layer.identifier),
        "time_code": float(time_code.GetValue()),
        "truth_point_count": len(truth_points),
        "hotspot_override_count": len(geometry["hotspot_positions"]),
        "smoke_quad_override_count": len(geometry["smoke_mesh_positions"]) // 4,
        "source_index_alignment_max_xy_error_m": round(
            max(source_index_xy_errors), 6
        ),
        "effective_hotspot_lift_m": 0.25,
        "effective_smoke_base_lift_above_hotspot_m": 1.6,
    }
    print(
        json.dumps(
            {"status": "flow_emitters_aligned_to_active_truth", **receipt},
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def probe_flow_state(stage: Any) -> dict[str, float | int | str]:
    """Fail closed unless the persistent Flow fire uses real combustion inputs."""
    import omni.timeline
    from pxr import Usd

    root = stage.GetPrimAtPath("/World/FireScenario/FlowVisual")
    simulate = stage.GetPrimAtPath("/World/FireScenario/FlowVisual/Simulate")
    if not root or not simulate or simulate.GetTypeName() != "FlowSimulate":
        raise RuntimeError("composed native Flow simulator is missing")
    sphere_emitters = _ordered_flow_sphere_emitters(root)
    front_mesh = stage.GetPrimAtPath("/World/FireScenario/FlowVisual/FrontRibbonEmitter")
    smoke_mesh = stage.GetPrimAtPath("/World/FireScenario/FlowVisual/SmokePlumeEmitter")
    mesh_emitters = [front_mesh, smoke_mesh]
    if (
        len(sphere_emitters) != 48
        or any(not prim or prim.GetTypeName() != "FlowEmitterMesh" for prim in mesh_emitters)
    ):
        raise RuntimeError("wildfire Flow source mismatch: expected front mesh, aligned smoke mesh, and 48 hotspots")
    timeline_seconds = float(omni.timeline.get_timeline_interface().get_current_time())
    time_code = Usd.TimeCode(timeline_seconds * float(stage.GetTimeCodesPerSecond()))
    front_positions = front_mesh.GetAttribute("meshPositions").Get(time_code)
    smoke_positions = smoke_mesh.GetAttribute("meshPositions").Get(time_code)
    if front_positions is None or len(front_positions) != 256 * 4:
        raise RuntimeError("native Flow mesh front does not expose the 256 terrain-following patches")
    if smoke_positions is None or len(smoke_positions) != 48 * 4:
        raise RuntimeError("native Flow smoke mesh does not expose 48 aligned plume bases")
    flame_emitters = [front_mesh, *sphere_emitters]
    emitters = [*flame_emitters, smoke_mesh]
    flame_fuel_values = []
    flame_temperature_values = []
    smoke_values = []
    burn_values = []
    for prim in emitters:
        fuel = prim.GetAttribute("fuel").Get(time_code)
        temperature = prim.GetAttribute("temperature").Get(time_code)
        smoke = prim.GetAttribute("smoke").Get(time_code)
        smoke_values.append(float(smoke) if smoke is not None else 0.0)
        burn = prim.GetAttribute("burn").Get(time_code)
        burn_values.append(float(burn) if burn is not None else 0.0)
        if prim != smoke_mesh:
            flame_fuel_values.append(float(fuel) if fuel is not None else 0.0)
            flame_temperature_values.append(float(temperature) if temperature is not None else 0.0)
    if min(flame_fuel_values) <= 0.0 or min(flame_temperature_values) <= 0.0:
        raise RuntimeError("native Flow emitters do not inject hot fuel for combustion")
    flame_smoke_values = [
        float(prim.GetAttribute("smoke").Get(time_code) or 0.0)
        for prim in flame_emitters
    ]
    if max(abs(value) for value in flame_smoke_values) > 1e-6:
        raise RuntimeError("native Flow flame emitters must preserve the accepted combustion-only profile")
    if max(abs(value) for value in burn_values) > 1e-6:
        raise RuntimeError("native Flow emitters bypass combustion with direct burn injection")
    smoke_source_value = float(smoke_mesh.GetAttribute("smoke").Get(time_code) or 0.0)
    smoke_source_fuel = float(smoke_mesh.GetAttribute("fuel").Get(time_code) or 0.0)
    if smoke_source_value <= 0.0 or abs(smoke_source_fuel) > 1e-6:
        raise RuntimeError("aligned smoke emitter must inject smoke without adding flame fuel")
    alignment_xy_errors = []
    alignment_lifts = []
    truth_smoke_points = _point_cloud(
        stage, "/World/FireScenario/Truth3D/SmokeSources"
    )
    if len(truth_smoke_points) != len(sphere_emitters):
        raise RuntimeError(
            "native Flow hotspot count does not match the active state's smoke truth points"
        )
    truth_alignment_xy_errors = []
    truth_alignment_lifts = []
    matched_truth_indices: set[int] = set()
    for emitter_index, sphere in enumerate(sphere_emitters):
        sphere_position = sphere.GetAttribute("position").Get(time_code)
        quad = smoke_positions[emitter_index * 4 : emitter_index * 4 + 4]
        center = tuple(sum(float(vertex[axis]) for vertex in quad) / 4.0 for axis in range(3))
        alignment_xy_errors.append(
            math.hypot(center[0] - float(sphere_position[0]), center[1] - float(sphere_position[1]))
        )
        alignment_lifts.append(center[2] - float(sphere_position[2]))
        truth_distances = [
            math.hypot(
                float(sphere_position[0]) - float(candidate[0]),
                float(sphere_position[1]) - float(candidate[1]),
            )
            for candidate in truth_smoke_points
        ]
        truth_index = min(range(len(truth_distances)), key=truth_distances.__getitem__)
        matched_truth_indices.add(truth_index)
        truth_point = truth_smoke_points[truth_index]
        truth_alignment_xy_errors.append(
            float(truth_distances[truth_index])
        )
        truth_alignment_lifts.append(
            float(sphere_position[2]) - float(truth_point[2])
        )
    if max(alignment_xy_errors) > 0.01 or min(alignment_lifts) < 1.55 or max(alignment_lifts) > 1.65:
        raise RuntimeError("native Flow smoke plume bases are not aligned above the active flame hotspots")
    if (
        len(matched_truth_indices) != len(truth_smoke_points)
        or
        max(truth_alignment_xy_errors) > 1.0
        or min(truth_alignment_lifts) < 0.20
        or max(truth_alignment_lifts) > 0.30
    ):
        raise RuntimeError(
            "native Flow hotspot emitters are not bijectively aligned with the active state's truth points: "
            f"matches={len(matched_truth_indices)}/{len(truth_smoke_points)}, "
            f"max_xy_error_m={max(truth_alignment_xy_errors):.6f}, "
            f"lift_range_m=[{min(truth_alignment_lifts):.6f},{max(truth_alignment_lifts):.6f}]"
        )
    metrics = {
        "flow_animation": "persistent_time_sampled_mesh_front_hotspots_and_aligned_smoke",
        "flow_emitter_count": len(emitters),
        "flow_mesh_emitter_count": len(mesh_emitters),
        "flow_hotspot_emitter_count": len(sphere_emitters),
        "flow_front_patch_count": len(front_positions) // 4,
        "flow_smoke_plume_count": len(smoke_positions) // 4,
        "smoke_source_alignment_max_xy_error_m": round(max(alignment_xy_errors), 6),
        "smoke_source_lift_min_m": round(min(alignment_lifts), 3),
        "smoke_source_lift_max_m": round(max(alignment_lifts), 3),
        "flow_truth_alignment_max_xy_error_m": round(
            max(truth_alignment_xy_errors), 6
        ),
        "flow_truth_hotspot_lift_min_m": round(min(truth_alignment_lifts), 3),
        "flow_truth_hotspot_lift_max_m": round(max(truth_alignment_lifts), 3),
        "flow_probe_time_seconds": round(timeline_seconds, 3),
        "minimum_fuel": round(min(flame_fuel_values), 3),
        "maximum_fuel": round(max(flame_fuel_values), 3),
        "minimum_temperature": round(min(flame_temperature_values), 3),
        "maximum_temperature": round(max(flame_temperature_values), 3),
        "maximum_direct_smoke": round(max(smoke_values), 3),
        "maximum_direct_burn": round(max(burn_values), 3),
    }
    print(json.dumps({"status": "flow_state_probe_passed", **metrics}, sort_keys=True), flush=True)
    return metrics


def prepare_flow_state_for_capture(
    *,
    app: Any,
    stage: Any,
    timeline: Any,
    target_time_s: float,
    warmup_updates: int,
    flow_capture_profile: dict[str, Any],
) -> dict[str, Any]:
    """Reset and warm Flow at one exact state time without timeline drift.

    Captures are intentionally independent: the Flow volume is cleared, the
    emitter animation is sampled at the exact state time, and Flow advances at
    that frozen time.  This prevents density emitted during camera/metadata
    setup from a later state from contaminating the current capture.
    """
    from pxr import Sdf

    if warmup_updates < 1:
        raise ValueError("--flow-warmup-updates must be positive")
    timeline.pause()
    timeline.set_current_time(float(target_time_s))
    app.update()
    alignment_receipt = align_flow_emitters_to_active_truth(
        stage, target_time_s=float(target_time_s)
    )
    app.update()
    simulate = stage.GetPrimAtPath("/World/FireScenario/FlowVisual/Simulate")
    if not simulate or simulate.GetTypeName() != "FlowSimulate":
        raise RuntimeError("composed native Flow simulator is missing during capture preparation")
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        with Sdf.ChangeBlock():
            simulate.GetAttribute("simulateWhenPaused").Set(True)
            simulate.GetAttribute("forceSimulate").Set(True)
            simulate.GetAttribute("forceClear").Set(True)
        app.update()
        with Sdf.ChangeBlock():
            simulate.GetAttribute("forceClear").Set(False)
        for _ in range(int(warmup_updates)):
            app.update()
        with Sdf.ChangeBlock():
            simulate.GetAttribute("simulateWhenPaused").Set(False)
            simulate.GetAttribute("forceSimulate").Set(False)
    finally:
        stage.SetEditTarget(previous_edit_target)
    app.update()
    if timeline.is_playing():
        raise RuntimeError("capture timeline advanced during fixed-time Flow warmup")
    actual_time_s = float(timeline.get_current_time())
    if abs(actual_time_s - float(target_time_s)) > 1e-6:
        raise RuntimeError(
            f"capture timeline drifted during Flow warmup: {actual_time_s} != {target_time_s}"
        )
    result = {
        "flow_capture_preparation_contract": "independent_volume_clear_and_fixed_state_time_warmup_v1",
        "flow_capture_time_s": actual_time_s,
        "flow_warmup_updates": int(warmup_updates),
        "flow_timeline_frozen_during_warmup": True,
        "flow_truth_alignment": alignment_receipt,
        "flow_capture_profile": dict(flow_capture_profile),
        **probe_flow_state(stage),
    }
    print(json.dumps({"status": "flow_capture_state_ready", **result}, sort_keys=True), flush=True)
    return result


def wait_for_stage(app: Any, context: Any, *, attempts: int = 600) -> Any:
    for _ in range(attempts):
        app.update()
        stage = context.get_stage()
        if stage is not None:
            return stage
    raise RuntimeError("Timed out while opening the FireViewer USD stage")


def configure_visible_viewport(
    app: Any,
    camera_prims: list[Any],
    *,
    preferred_camera_id: str = "CAM_11",
    settle_updates: int = 12,
) -> str:
    """Bind the visible Kit viewport to a real dataset camera."""
    import omni.kit.viewport.utility as viewport_utils

    selected = next(
        (
            camera
            for camera in camera_prims
            if str(camera.GetAttribute("fireviewer:camera_id").Get()) == preferred_camera_id
        ),
        camera_prims[0],
    )
    viewport = None
    for _ in range(180):
        app.update()
        viewport = viewport_utils.get_active_viewport()
        if viewport is not None:
            break
    if viewport is None:
        raise RuntimeError("Visible mode requested but Kit has no active viewport")

    camera_path = selected.GetPath()
    viewport.camera_path = camera_path
    for _ in range(settle_updates):
        app.update()
    active_camera_path = str(viewport.camera_path)
    if active_camera_path != str(camera_path):
        raise RuntimeError(
            f"Visible viewport rejected camera {camera_path}; active camera is {active_camera_path}"
        )
    camera_id = str(selected.GetAttribute("fireviewer:camera_id").Get())
    print(
        json.dumps(
            {
                "status": "visible_viewport_camera_ready",
                "camera_id": camera_id,
                "camera_path": active_camera_path,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return camera_id


def write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def fire_state_metrics(scenario: Any) -> dict[str, Any]:
    def required(name: str) -> Any:
        value = scenario.GetAttribute(f"fireviewer:{name}").Get()
        if value is None:
            raise RuntimeError(f"selected fire state is missing fireviewer:{name}")
        return value

    ignition = required("ignition_l93_m")
    burned_scope = scenario.GetChild("Truth3D")
    burned_tree_count = burned_scope.GetAttribute("fireviewer:burned_tree_count").Get() if burned_scope else None
    if burned_tree_count is None:
        raise RuntimeError("selected fire state is missing burned-tree truth metadata")
    active_in_scene = scenario.GetAttribute("fireviewer:active_in_scene").Get()
    return {
        "state_index": int(required("state_index")),
        "state_count": int(required("state_count")),
        "day_index": int(required("incident_day_index")),
        "state_in_day": int(required("state_in_day")),
        "observation_elapsed_s": float(required("observation_elapsed_s")),
        "fire_elapsed_s": float(required("elapsed_s")),
        "burned_area_m2": float(required("burned_area_m2")),
        "active_front_length_m": float(required("active_front_length_m")),
        "burned_tree_count": int(burned_tree_count),
        "ignition_l93_m": [float(ignition[0]), float(ignition[1])],
        "active_in_scene": True if active_in_scene is None else bool(active_in_scene),
    }


def validate_fire_state_progression(
    *,
    metrics: dict[str, Any],
    expected_index: int,
    previous: dict[str, Any] | None,
    fixed_ignition: list[float] | None,
    states_per_day: int = STATES_PER_DAY,
    expected_states: int = EXPECTED_STATES,
) -> list[float]:
    if states_per_day <= 0 or expected_states <= 0:
        raise RuntimeError("invalid runtime incident cadence")
    expected_day = (expected_index - 1) // states_per_day + 1
    expected_state_in_day = (expected_index - 1) % states_per_day + 1
    if (
        metrics["state_index"] != expected_index
        or metrics["state_count"] != expected_states
        or metrics["day_index"] != expected_day
        or metrics["state_in_day"] != expected_state_in_day
    ):
        raise RuntimeError(f"invalid selected-state identity at state_{expected_index:03d}: {metrics}")
    ignition = metrics["ignition_l93_m"]
    if fixed_ignition is not None and any(abs(a - b) > 1e-6 for a, b in zip(ignition, fixed_ignition)):
        raise RuntimeError(f"ignition moved at state_{expected_index:03d}: {ignition} != {fixed_ignition}")
    if previous is not None:
        for name in ("observation_elapsed_s", "fire_elapsed_s", "burned_area_m2", "burned_tree_count"):
            if metrics[name] < previous[name]:
                raise RuntimeError(f"non-monotone {name} at state_{expected_index:03d}")
    return ignition


def _normalize(vector: Any) -> Any:
    import numpy as np

    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise RuntimeError("cannot orient a camera toward a coincident target")
    return vector / length


def _quaternion_from_rotation(matrix: Any) -> tuple[float, float, float, float]:
    import numpy as np

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale)
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)


def _camera_basis(position: Any, target: Any) -> tuple[Any, Any, Any, tuple[float, float, float, float]]:
    import numpy as np

    forward = _normalize(target - position)
    up_hint = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.98:
        up_hint = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    right = _normalize(np.cross(forward, up_hint))
    up = _normalize(np.cross(right, forward))
    local_to_world = np.column_stack((right, up, -forward))
    return forward, right, up, _quaternion_from_rotation(local_to_world)


def _project_point(
    *,
    point: Any,
    position: Any,
    forward: Any,
    right: Any,
    up: Any,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    vertical_aperture_mm: float,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    import numpy as np

    relative = point - position
    depth = float(np.dot(relative, forward))
    if depth <= 1e-6:
        return {"pixel_xy": None, "normalized_xy": None, "in_frame": False, "camera_depth_m": depth}
    normalized_x = float(np.dot(relative, right)) / (depth * horizontal_aperture_mm / (2.0 * focal_length_mm))
    normalized_y = float(np.dot(relative, up)) / (depth * vertical_aperture_mm / (2.0 * focal_length_mm))
    pixel_x = (normalized_x + 1.0) * 0.5 * resolution[0]
    pixel_y = (1.0 - normalized_y) * 0.5 * resolution[1]
    return {
        "pixel_xy": [pixel_x, pixel_y],
        "normalized_xy": [normalized_x, normalized_y],
        "in_frame": bool(-1.0 <= normalized_x <= 1.0 and -1.0 <= normalized_y <= 1.0),
        "camera_depth_m": depth,
    }


def _point_cloud(stage: Any, path: str) -> Any:
    import numpy as np
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"required dynamic target prim is missing: {path}")
    points = UsdGeom.PointBased(prim).GetPointsAttr().Get()
    if points is None or len(points) == 0:
        raise RuntimeError(f"required dynamic target prim has no points: {path}")
    return np.asarray([[float(value[0]), float(value[1]), float(value[2])] for value in points], dtype=np.float64)


def _mesh_projection_geometry(stage: Any, path: str) -> dict[str, Any]:
    """Load one authored truth mesh into compact NumPy topology arrays."""
    import numpy as np
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"required truth mesh is missing: {path}")
    mesh = UsdGeom.Mesh(prim)
    points_value = mesh.GetPointsAttr().Get()
    counts_value = mesh.GetFaceVertexCountsAttr().Get()
    indices_value = mesh.GetFaceVertexIndicesAttr().Get()
    if points_value is None or counts_value is None or indices_value is None:
        raise RuntimeError(f"required truth mesh topology is incomplete: {path}")
    points = np.asarray(points_value, dtype=np.float64)
    counts = np.asarray(counts_value, dtype=np.int32)
    indices = np.asarray(indices_value, dtype=np.int32)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or counts.ndim != 1
        or indices.ndim != 1
        or points.size == 0
        or counts.size == 0
        or int(np.sum(counts)) != int(indices.size)
        or int(np.min(counts)) < 3
        or int(np.min(indices)) < 0
        or int(np.max(indices)) >= int(len(points))
    ):
        raise RuntimeError(f"invalid truth mesh topology: {path}")
    return {
        "path": path,
        "points": points,
        "face_vertex_counts": counts,
        "face_vertex_indices": indices,
    }


def load_truth_projection_geometry(stage: Any) -> dict[str, Any]:
    """Load the active state's image-label geometry once, before its captures."""
    import numpy as np
    from pxr import UsdGeom

    smoke_path = "/World/FireScenario/Truth3D/SmokeSources"
    smoke_prim = stage.GetPrimAtPath(smoke_path)
    if not smoke_prim or not smoke_prim.IsValid() or not smoke_prim.IsA(UsdGeom.Points):
        raise RuntimeError(f"required smoke truth points are missing: {smoke_path}")
    smoke_schema = UsdGeom.Points(smoke_prim)
    smoke_points_value = smoke_schema.GetPointsAttr().Get()
    smoke_widths_value = smoke_schema.GetWidthsAttr().Get()
    if smoke_points_value is None or len(smoke_points_value) == 0:
        raise RuntimeError(f"required smoke truth points are empty: {smoke_path}")
    smoke_points = np.asarray(smoke_points_value, dtype=np.float64)
    if smoke_widths_value is None or len(smoke_widths_value) == 0:
        smoke_widths = np.full(len(smoke_points), 1.5, dtype=np.float64)
    else:
        smoke_widths = np.asarray(smoke_widths_value, dtype=np.float64)
        if smoke_widths.size == 1:
            smoke_widths = np.full(len(smoke_points), float(smoke_widths[0]), dtype=np.float64)
    if smoke_widths.shape != (len(smoke_points),) or not np.all(np.isfinite(smoke_widths)):
        raise RuntimeError(f"invalid smoke truth widths: {smoke_path}")

    smoke_velocity = np.asarray((-0.097617, 0.219251, 8.5), dtype=np.float64)
    for prim in stage.Traverse():
        if prim.GetName() != "SmokePlumeEmitter":
            continue
        value = prim.GetAttribute("velocity").Get()
        if value is not None and len(value) == 3:
            candidate = np.asarray(value, dtype=np.float64)
            if np.all(np.isfinite(candidate)) and float(candidate[2]) > 0.0:
                smoke_velocity = candidate
                break

    geometry = {
        "schema": "fireviewer.projected-truth-geometry.v1",
        "fire_front": _mesh_projection_geometry(
            stage, "/World/FireScenario/Truth3D/VisibleFireFront"
        ),
        "fire_perimeter": _mesh_projection_geometry(
            stage, "/World/FireScenario/Truth3D/BurnedPerimeter"
        ),
        "burned_area": _mesh_projection_geometry(
            stage, "/World/FireScenario/Truth3D/BurnedSurface"
        ),
        "smoke_source": {
            "path": smoke_path,
            "points": smoke_points,
            "widths_m": smoke_widths,
        },
        "smoke_velocity_local_m_s": smoke_velocity,
    }
    print(
        json.dumps(
            {
                "status": "truth_projection_geometry_loaded",
                "fire_front_vertices": int(len(geometry["fire_front"]["points"])),
                "fire_front_faces": int(len(geometry["fire_front"]["face_vertex_counts"])),
                "fire_perimeter_vertices": int(len(geometry["fire_perimeter"]["points"])),
                "burned_area_vertices": int(len(geometry["burned_area"]["points"])),
                "smoke_source_points": int(len(smoke_points)),
                "smoke_velocity_local_m_s": smoke_velocity.tolist(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return geometry


def _project_points_array(
    *,
    points: Any,
    position: Any,
    forward: Any,
    right: Any,
    up: Any,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    vertical_aperture_mm: float,
    resolution: tuple[int, int],
) -> tuple[Any, Any, Any]:
    """Vectorized world-to-pixel projection matching :func:`_project_point`."""
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("projection points must have shape (N, 3)")
    relative = values - np.asarray(position, dtype=np.float64)
    depths = relative @ np.asarray(forward, dtype=np.float64)
    valid = np.isfinite(depths) & (depths > 1e-6)
    safe_depths = np.where(valid, depths, 1.0)
    normalized_x = (relative @ np.asarray(right, dtype=np.float64)) / (
        safe_depths * float(horizontal_aperture_mm) / (2.0 * float(focal_length_mm))
    )
    normalized_y = (relative @ np.asarray(up, dtype=np.float64)) / (
        safe_depths * float(vertical_aperture_mm) / (2.0 * float(focal_length_mm))
    )
    pixels = np.column_stack(
        (
            (normalized_x + 1.0) * 0.5 * int(resolution[0]),
            (1.0 - normalized_y) * 0.5 * int(resolution[1]),
        )
    )
    pixels[~valid] = np.nan
    return pixels, depths, valid


def _rasterize_projected_mesh(
    *,
    geometry: dict[str, Any],
    position: Any,
    forward: Any,
    right: Any,
    up: Any,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    vertical_aperture_mm: float,
    resolution: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    """Rasterize visible, in-front truth faces into one deterministic mask."""
    import cv2
    import numpy as np

    width, height = int(resolution[0]), int(resolution[1])
    pixels, depths, point_valid = _project_points_array(
        points=geometry["points"],
        position=position,
        forward=forward,
        right=right,
        up=up,
        focal_length_mm=focal_length_mm,
        horizontal_aperture_mm=horizontal_aperture_mm,
        vertical_aperture_mm=vertical_aperture_mm,
        resolution=resolution,
    )
    counts = np.asarray(geometry["face_vertex_counts"], dtype=np.int32)
    indices = np.asarray(geometry["face_vertex_indices"], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons: list[Any] | Any
    if counts.size and bool(np.all(counts == counts[0])):
        vertices_per_face = int(counts[0])
        face_indices = indices.reshape(-1, vertices_per_face)
        face_valid = np.all(point_valid[face_indices], axis=1)
        face_pixels = pixels[face_indices]
        face_valid &= np.max(face_pixels[..., 0], axis=1) >= 0.0
        face_valid &= np.min(face_pixels[..., 0], axis=1) < width
        face_valid &= np.max(face_pixels[..., 1], axis=1) >= 0.0
        face_valid &= np.min(face_pixels[..., 1], axis=1) < height
        polygons = np.rint(
            np.clip(face_pixels[face_valid], (-4 * width, -4 * height), (5 * width, 5 * height))
        ).astype(np.int32)
        visible_face_count = int(np.count_nonzero(face_valid))
    else:
        polygons = []
        cursor = 0
        visible_face_count = 0
        for count in counts:
            face_indices = indices[cursor : cursor + int(count)]
            cursor += int(count)
            if not bool(np.all(point_valid[face_indices])):
                continue
            polygon = pixels[face_indices]
            if (
                float(np.max(polygon[:, 0])) < 0.0
                or float(np.min(polygon[:, 0])) >= width
                or float(np.max(polygon[:, 1])) < 0.0
                or float(np.min(polygon[:, 1])) >= height
            ):
                continue
            polygons.append(
                np.rint(
                    np.clip(polygon, (-4 * width, -4 * height), (5 * width, 5 * height))
                ).astype(np.int32)
            )
            visible_face_count += 1
    if visible_face_count:
        cv2.fillPoly(mask, polygons, 255, lineType=cv2.LINE_8)
    finite_depths = depths[point_valid]
    return mask, {
        "source_prim": str(geometry["path"]),
        "source_vertex_count": int(len(geometry["points"])),
        "source_face_count": int(len(counts)),
        "projected_face_count": visible_face_count,
        "nearest_projected_depth_m": (
            float(np.min(finite_depths)) if finite_depths.size else None
        ),
    }


def _dilate_physical_envelope(
    mask: Any,
    *,
    physical_radius_m: float,
    representative_depth_m: float | None,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    resolution: tuple[int, int],
    minimum_radius_px: int,
    maximum_radius_px: int,
) -> tuple[Any, int]:
    import cv2
    import numpy as np

    depth = max(float(representative_depth_m or 1.0), 1.0)
    focal_px = float(focal_length_mm) / float(horizontal_aperture_mm) * int(resolution[0])
    radius_px = int(
        np.clip(
            math.ceil(float(physical_radius_m) * focal_px / depth),
            int(minimum_radius_px),
            int(maximum_radius_px),
        )
    )
    if radius_px <= 0 or not bool(np.any(mask)):
        return np.asarray(mask, dtype=np.uint8), radius_px
    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel), radius_px


def project_truth_masks(
    *,
    view: dict[str, Any],
    truth_geometry: dict[str, Any],
    resolution: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project active-state truth into camera-aligned dense training masks.

    NVIDIA Flow supplies the accepted volumetric beauty.  Flow volumes do not
    participate in Replicator semantic IDs, so this projection is the explicit,
    traceable label source for fire, perimeter, smoke and burned ground.
    """
    import cv2
    import numpy as np

    width, height = int(resolution[0]), int(resolution[1])
    empty = lambda: np.zeros((height, width), dtype=np.uint8)
    if not bool(view.get("expected_fire_visible")):
        arrays = {
            "fire_front": empty(),
            "fire_perimeter": empty(),
            "smoke_source": empty(),
            "smoke": empty(),
            "burned_area": empty(),
        }
        metadata = {
            "schema": "fireviewer.projected-dense-targets.v1",
            "source_contract": "schedule_negative_context_forced_empty_after_camera_rotated_away",
            "expected_fire_visible": False,
            "resolution_px": [width, height],
            "pixel_counts": {name: 0 for name in arrays},
            "source_geometry_schema": truth_geometry.get("schema"),
        }
        return arrays, metadata

    position = np.asarray(view["camera_position_local_m"], dtype=np.float64)
    aim = np.asarray(view["camera_aim_local_m"], dtype=np.float64)
    forward, right, up, _ = _camera_basis(position, aim)
    focal_length = float(view["focal_length_mm"])
    horizontal_aperture = float(view["horizontal_aperture_mm"])
    vertical_aperture = float(view["vertical_aperture_mm"])
    projection_kwargs = {
        "position": position,
        "forward": forward,
        "right": right,
        "up": up,
        "focal_length_mm": focal_length,
        "horizontal_aperture_mm": horizontal_aperture,
        "vertical_aperture_mm": vertical_aperture,
        "resolution": resolution,
    }
    front_mask, front_meta = _rasterize_projected_mesh(
        geometry=truth_geometry["fire_front"], **projection_kwargs
    )
    perimeter_mask, perimeter_meta = _rasterize_projected_mesh(
        geometry=truth_geometry["fire_perimeter"], **projection_kwargs
    )
    burned_mask, burned_meta = _rasterize_projected_mesh(
        geometry=truth_geometry["burned_area"], **projection_kwargs
    )
    front_mask, front_radius_px = _dilate_physical_envelope(
        front_mask,
        physical_radius_m=2.5,
        representative_depth_m=front_meta["nearest_projected_depth_m"],
        focal_length_mm=focal_length,
        horizontal_aperture_mm=horizontal_aperture,
        resolution=resolution,
        minimum_radius_px=2,
        maximum_radius_px=16,
    )
    perimeter_mask, perimeter_radius_px = _dilate_physical_envelope(
        perimeter_mask,
        physical_radius_m=0.75,
        representative_depth_m=perimeter_meta["nearest_projected_depth_m"],
        focal_length_mm=focal_length,
        horizontal_aperture_mm=horizontal_aperture,
        resolution=resolution,
        minimum_radius_px=1,
        maximum_radius_px=8,
    )

    smoke_geometry = truth_geometry["smoke_source"]
    smoke_points = np.asarray(smoke_geometry["points"], dtype=np.float64)
    smoke_widths = np.asarray(smoke_geometry["widths_m"], dtype=np.float64)
    smoke_pixels, smoke_depths, smoke_valid = _project_points_array(
        points=smoke_points, **projection_kwargs
    )
    smoke_source_mask = empty()
    smoke_mask = empty()
    focal_px = focal_length / horizontal_aperture * width
    visible_source_count = 0
    for index in np.flatnonzero(smoke_valid):
        pixel_x, pixel_y = smoke_pixels[index]
        if pixel_x < 0.0 or pixel_x >= width or pixel_y < 0.0 or pixel_y >= height:
            continue
        radius_px = int(
            np.clip(
                math.ceil(max(float(smoke_widths[index]) * 0.5, 0.75) * focal_px / float(smoke_depths[index])),
                2,
                12,
            )
        )
        center = (int(round(pixel_x)), int(round(pixel_y)))
        cv2.circle(smoke_source_mask, center, radius_px, 255, thickness=-1, lineType=cv2.LINE_8)
        visible_source_count += 1

    smoke_velocity = np.asarray(
        truth_geometry.get("smoke_velocity_local_m_s", (0.0, 0.0, 8.5)),
        dtype=np.float64,
    )
    vertical_velocity = max(float(smoke_velocity[2]), 1e-3)
    height_levels_m = np.asarray((0.0, 8.0, 18.0, 32.0, 50.0, 70.0), dtype=np.float64)
    horizontal_per_height = smoke_velocity[:2] / vertical_velocity
    for source_index, source in enumerate(smoke_points):
        plume_points = np.column_stack(
            (
                source[0] + horizontal_per_height[0] * height_levels_m,
                source[1] + horizontal_per_height[1] * height_levels_m,
                source[2] + 1.6 + height_levels_m,
            )
        )
        plume_pixels, plume_depths, plume_valid = _project_points_array(
            points=plume_points, **projection_kwargs
        )
        for level in range(len(height_levels_m)):
            if not bool(plume_valid[level]):
                continue
            pixel_x, pixel_y = plume_pixels[level]
            if (
                pixel_x < -64.0
                or pixel_x >= width + 64.0
                or pixel_y < -64.0
                or pixel_y >= height + 64.0
            ):
                continue
            physical_radius = max(float(smoke_widths[source_index]) * 0.6, 1.0) + 0.10 * float(height_levels_m[level])
            radius_px = int(
                np.clip(
                    math.ceil(physical_radius * focal_px / max(float(plume_depths[level]), 1.0)),
                    3,
                    24,
                )
            )
            cv2.circle(
                smoke_mask,
                (int(round(pixel_x)), int(round(pixel_y))),
                radius_px,
                255,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
        valid_indices = np.flatnonzero(plume_valid)
        for start, end in zip(valid_indices[:-1], valid_indices[1:]):
            first = plume_pixels[start]
            second = plume_pixels[end]
            if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
                continue
            mean_height = 0.5 * (height_levels_m[start] + height_levels_m[end])
            physical_diameter = 2.0 * (max(float(smoke_widths[source_index]) * 0.6, 1.0) + 0.10 * float(mean_height))
            mean_depth = max(0.5 * (float(plume_depths[start]) + float(plume_depths[end])), 1.0)
            thickness_px = int(np.clip(math.ceil(physical_diameter * focal_px / mean_depth), 5, 48))
            cv2.line(
                smoke_mask,
                (int(round(first[0])), int(round(first[1]))),
                (int(round(second[0])), int(round(second[1]))),
                255,
                thickness=thickness_px,
                lineType=cv2.LINE_8,
            )
    smoke_mask = cv2.bitwise_or(smoke_mask, smoke_source_mask)
    arrays = {
        "fire_front": np.asarray(front_mask, dtype=np.uint8),
        "fire_perimeter": np.asarray(perimeter_mask, dtype=np.uint8),
        "smoke_source": np.asarray(smoke_source_mask, dtype=np.uint8),
        "smoke": np.asarray(smoke_mask, dtype=np.uint8),
        "burned_area": np.asarray(burned_mask, dtype=np.uint8),
    }
    metadata = {
        "schema": "fireviewer.projected-dense-targets.v1",
        "source_contract": "active_usd_truth_geometry_projected_through_exact_capture_camera",
        "beauty_fire_contract": "nvidia_flow_volumetric_fire_and_smoke_not_semantic_id_geometry",
        "expected_fire_visible": True,
        "resolution_px": [width, height],
        "camera_id": view.get("camera_id"),
        "state_id": view.get("state_id"),
        "zoom_index": view.get("zoom_index"),
        "fire_front": {
            **front_meta,
            "combustion_envelope_radius_m": 2.5,
            "raster_dilation_radius_px": front_radius_px,
        },
        "fire_perimeter": {
            **perimeter_meta,
            "visibility_envelope_radius_m": 0.75,
            "raster_dilation_radius_px": perimeter_radius_px,
        },
        "burned_area": burned_meta,
        "smoke_source": {
            "source_prim": str(smoke_geometry["path"]),
            "source_point_count": int(len(smoke_points)),
            "projected_source_count": visible_source_count,
            "source_widths_m": "authored_UsdGeomPoints_widths",
        },
        "smoke_plume": {
            "model": "flow_source_velocity_advected_vertical_envelope_v1",
            "velocity_local_m_s": smoke_velocity.tolist(),
            "source_lift_m": 1.6,
            "height_levels_m": height_levels_m.tolist(),
            "radius_growth_m_per_height_m": 0.10,
            "not_a_radiometric_or_optical_density_measurement": True,
        },
        "pixel_counts": {
            name: int(np.count_nonzero(mask)) for name, mask in arrays.items()
        },
        "source_geometry_schema": truth_geometry.get("schema"),
    }
    return arrays, metadata


def terrain_elevation_at(stage: Any, *, x_local_m: float, y_local_m: float) -> tuple[float, float]:
    """Return the nearest accepted MNT mesh elevation and horizontal sample distance."""
    import numpy as np
    from pxr import Usd, UsdGeom

    terrain = stage.GetPrimAtPath("/World/FireViewerSite/Terrain")
    if not terrain or not terrain.IsValid():
        raise RuntimeError("accepted terrain root is missing")
    best_squared_distance = math.inf
    best_elevation: float | None = None
    for prim in Usd.PrimRange(terrain):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points is None or len(points) == 0:
            continue
        values = np.asarray(points, dtype=np.float64)
        squared_distances = (
            (values[:, 0] - float(x_local_m)) ** 2
            + (values[:, 1] - float(y_local_m)) ** 2
        )
        index = int(np.argmin(squared_distances))
        distance = float(squared_distances[index])
        if distance < best_squared_distance:
            best_squared_distance = distance
            best_elevation = float(values[index, 2])
    if best_elevation is None:
        raise RuntimeError("accepted terrain meshes contain no MNT sample")
    horizontal_distance = math.sqrt(best_squared_distance)
    if horizontal_distance > 25.0:
        raise RuntimeError(
            f"ignition is {horizontal_distance:.3f} m from the nearest accepted MNT sample"
        )
    return best_elevation, horizontal_distance


def retarget_cameras_for_state(
    *,
    stage: Any,
    camera_prims: list[Any],
    anchor: tuple[float, float],
    resolution: tuple[int, int],
    view_plan_by_camera: dict[str, dict[str, Any]] | None = None,
    truth_geometry: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom

    flame_points = (
        np.asarray(truth_geometry["fire_front"]["points"], dtype=np.float64)
        if truth_geometry is not None
        else _point_cloud(stage, "/World/FireScenario/Truth3D/VisibleFireFront")
    )
    print(json.dumps({"status": "flame_targets_loaded", "points": int(len(flame_points))}, sort_keys=True), flush=True)
    smoke_points = (
        np.asarray(truth_geometry["smoke_source"]["points"], dtype=np.float64)
        if truth_geometry is not None
        else _point_cloud(stage, "/World/FireScenario/Truth3D/SmokeSources")
    )
    print(json.dumps({"status": "smoke_targets_loaded", "points": int(len(smoke_points))}, sort_keys=True), flush=True)
    result: dict[str, dict[str, Any]] = {}
    camera_updates: list[tuple[Any, Any, Any, Any]] = []
    for camera in camera_prims:
        camera_schema = UsdGeom.Camera(camera)
        if not camera_schema:
            raise RuntimeError(f"invalid UsdGeom.Camera schema: {camera.GetPath()}")
        camera_id = str(camera.GetAttribute("fireviewer:camera_id").Get())
        position_value = camera.GetAttribute("xformOp:translate").Get()
        if position_value is None:
            raise RuntimeError(f"camera has no position: {camera.GetPath()}")
        position = np.asarray([float(position_value[0]), float(position_value[1]), float(position_value[2])], dtype=np.float64)
        flame_index = int(np.argmin(np.linalg.norm(flame_points - position, axis=1)))
        smoke_index = int(np.argmin(np.linalg.norm(smoke_points - position, axis=1)))
        nearest_flame = flame_points[flame_index]
        nearest_smoke = smoke_points[smoke_index]
        plan = (view_plan_by_camera or {}).get(camera_id, {})
        expected_fire = bool(plan.get("expected_fire_visible", camera.GetAttribute("fireviewer:expected_fire_in_frame").Get()))
        if expected_fire:
            aim = nearest_flame.copy()
            targeting_mode = "nearest_fire_front_point_centered"
        else:
            delta = nearest_flame - position
            horizontal_distance = max(float(np.linalg.norm(delta[:2])), 1.0)
            sign = -1.0 if float(plan.get("fire_bearing_offset_degrees", camera.GetAttribute("fireviewer:fire_bearing_offset_degrees").Get() or 95.0)) < 0 else 1.0
            angle = math.radians(sign * 95.0)
            rotated_x = delta[0] * math.cos(angle) - delta[1] * math.sin(angle)
            rotated_y = delta[0] * math.sin(angle) + delta[1] * math.cos(angle)
            aim = np.asarray((position[0] + rotated_x, position[1] + rotated_y, nearest_flame[2]), dtype=np.float64)
            if float(np.linalg.norm(aim[:2] - position[:2])) < horizontal_distance * 0.9:
                raise RuntimeError(f"negative targeting rotation collapsed for {camera_id}")
            targeting_mode = "negative_context_rotated_away_from_nearest_fire_front_point"
        forward, right, up, quaternion = _camera_basis(position, aim)
        camera_updates.append(
            (
                camera,
                Gf.Quatf(float(quaternion[0]), Gf.Vec3f(float(quaternion[1]), float(quaternion[2]), float(quaternion[3]))),
                Gf.Vec3d(float(aim[0]), float(aim[1]), float(aim[2])),
                Gf.Vec3d(float(nearest_flame[0]), float(nearest_flame[1]), float(nearest_flame[2])),
            )
        )
        focal_length = float(camera_schema.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera_schema.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera_schema.GetVerticalApertureAttr().Get())
        flame_projection = _project_point(point=nearest_flame, position=position, forward=forward, right=right, up=up, focal_length_mm=focal_length, horizontal_aperture_mm=horizontal_aperture, vertical_aperture_mm=vertical_aperture, resolution=resolution)
        smoke_projection = _project_point(point=nearest_smoke, position=position, forward=forward, right=right, up=up, focal_length_mm=focal_length, horizontal_aperture_mm=horizontal_aperture, vertical_aperture_mm=vertical_aperture, resolution=resolution)
        yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
        pitch = math.degrees(math.atan2(float(forward[2]), math.hypot(float(forward[0]), float(forward[1]))))
        result[camera_id] = {
            "dynamic_targeting_contract": "nearest_fire_and_smoke_points_per_state_v1",
            "dynamic_targeting_mode": targeting_mode,
            "camera_position_local_m": position.tolist(),
            "camera_position_l93_ngf_ign69_m": [anchor[0] + float(position[0]), anchor[1] + float(position[1]), float(position[2])],
            "camera_aim_local_m": aim.tolist(),
            "camera_orientation_quat_wxyz": [float(value) for value in quaternion],
            "camera_orientation_yaw_pitch_roll_degrees": [yaw, pitch, 0.0],
            "camera_forward_local": forward.tolist(),
            "image_resolution_px": [int(resolution[0]), int(resolution[1])],
            "focal_length_mm": focal_length,
            "horizontal_aperture_mm": horizontal_aperture,
            "vertical_aperture_mm": vertical_aperture,
            "horizontal_fov_degrees": math.degrees(2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))),
            "vertical_fov_degrees": math.degrees(2.0 * math.atan(vertical_aperture / (2.0 * focal_length))),
            "nearest_flame_point_index": flame_index,
            "nearest_flame_point_local_m": nearest_flame.tolist(),
            "nearest_flame_point_l93_ngf_ign69_m": [anchor[0] + float(nearest_flame[0]), anchor[1] + float(nearest_flame[1]), float(nearest_flame[2])],
            "nearest_flame_distance_m": float(np.linalg.norm(nearest_flame - position)),
            "nearest_flame_projection": flame_projection,
            "nearest_smoke_point_index": smoke_index,
            "nearest_smoke_point_local_m": nearest_smoke.tolist(),
            "nearest_smoke_point_l93_ngf_ign69_m": [anchor[0] + float(nearest_smoke[0]), anchor[1] + float(nearest_smoke[1]), float(nearest_smoke[2])],
            "nearest_smoke_distance_m": float(np.linalg.norm(nearest_smoke - position)),
            "nearest_smoke_projection": smoke_projection,
            "fire_front_point_count": int(len(flame_points)),
            "smoke_source_point_count": int(len(smoke_points)),
            "visible_flame_points_local_m": flame_points.tolist(),
            "smoke_source_points_local_m": smoke_points.tolist(),
            "active_front_local_m": flame_points.tolist(),
        }
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        with Sdf.ChangeBlock():
            for camera, orientation, aim, nearest_flame in camera_updates:
                camera.GetAttribute("xformOp:orient").Set(orientation)
                camera.GetAttribute("fireviewer:look_at_local_m").Set(aim)
                camera.GetAttribute("fireviewer:fire_target_local_m").Set(nearest_flame)
    finally:
        stage.SetEditTarget(previous_edit_target)
    print(json.dumps({"status": "camera_pose_batch_authored", "cameras": len(camera_updates)}, sort_keys=True), flush=True)
    return result


def author_camera_focal_lengths(stage: Any, cameras_by_id: dict[str, Any], focal_lengths: dict[str, float]) -> None:
    """Author a focal-length batch without changing camera position or orientation."""
    from pxr import Sdf, UsdGeom

    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        with Sdf.ChangeBlock():
            for camera_id, focal_length in focal_lengths.items():
                if camera_id not in cameras_by_id or float(focal_length) <= 0.0:
                    raise RuntimeError(f"invalid zoom focal length for {camera_id}: {focal_length}")
                UsdGeom.Camera(cameras_by_id[camera_id]).GetFocalLengthAttr().Set(float(focal_length))
    finally:
        stage.SetEditTarget(previous_edit_target)


def apply_zoom_metadata(
    *,
    stage: Any,
    cameras_by_id: dict[str, Any],
    view_plan_by_camera: dict[str, dict[str, Any]],
    resolution: tuple[int, int],
    truth_geometry: dict[str, Any] | None = None,
) -> None:
    """Apply one zoom level and recompute projections while keeping the pose fixed."""
    import numpy as np

    focal_lengths = {
        camera_id: float(view["focal_length_mm"])
        for camera_id, view in view_plan_by_camera.items()
    }
    author_camera_focal_lengths(stage, cameras_by_id, focal_lengths)
    for camera_id, view in view_plan_by_camera.items():
        position = np.asarray(view["camera_position_local_m"], dtype=np.float64)
        aim = np.asarray(view["camera_aim_local_m"], dtype=np.float64)
        nearest_flame = np.asarray(view["nearest_flame_point_local_m"], dtype=np.float64)
        nearest_smoke = np.asarray(view["nearest_smoke_point_local_m"], dtype=np.float64)
        forward, right, up, _ = _camera_basis(position, aim)
        focal_length = float(view["focal_length_mm"])
        horizontal_aperture = float(view["horizontal_aperture_mm"])
        vertical_aperture = float(view["vertical_aperture_mm"])
        view["horizontal_fov_degrees"] = math.degrees(
            2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
        )
        view["vertical_fov_degrees"] = math.degrees(
            2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
        )
        view["nearest_flame_projection"] = _project_point(
            point=nearest_flame,
            position=position,
            forward=forward,
            right=right,
            up=up,
            focal_length_mm=focal_length,
            horizontal_aperture_mm=horizontal_aperture,
            vertical_aperture_mm=vertical_aperture,
            resolution=resolution,
        )
        view["nearest_smoke_projection"] = _project_point(
            point=nearest_smoke,
            position=position,
            forward=forward,
            right=right,
            up=up,
            focal_length_mm=focal_length,
            horizontal_aperture_mm=horizontal_aperture,
            vertical_aperture_mm=vertical_aperture,
            resolution=resolution,
        )
        view["zoom_runtime_contract"] = "focal_length_only_pose_and_simulation_time_frozen"
        if truth_geometry is not None:
            dense_target_arrays, dense_target_metadata = project_truth_masks(
                view=view,
                truth_geometry=truth_geometry,
                resolution=resolution,
            )
            view["_dense_target_arrays"] = dense_target_arrays
            view["dense_target_projection"] = dense_target_metadata
            visibility_acceptance = classify_capture_visibility_acceptance(view)
            projected_union = np.logical_or(
                dense_target_arrays["fire_front"] != 0,
                dense_target_arrays["smoke"] != 0,
            )
            projected_pixels = int(np.count_nonzero(projected_union))
            if not bool(view.get("expected_fire_visible")):
                projected_tier = "none_expected"
            elif projected_pixels <= 32:
                projected_tier = "trace"
            elif projected_pixels <= 256:
                projected_tier = "small"
            elif projected_pixels <= 2_048:
                projected_tier = "moderate"
            else:
                projected_tier = "clear"
            visibility_acceptance.update(
                {
                    "projected_fire_or_smoke_pixels": projected_pixels,
                    "projected_frame_fraction": projected_pixels
                    / float(int(resolution[0]) * int(resolution[1])),
                    "projected_visibility_tier": projected_tier,
                    "visibility_validation_status": (
                        "accepted"
                        if (
                            bool(view.get("expected_fire_visible"))
                            and projected_pixels
                            >= int(
                                visibility_acceptance[
                                    "minimum_positive_projected_pixels"
                                ]
                            )
                        )
                        or (
                            not bool(view.get("expected_fire_visible"))
                            and projected_pixels == 0
                        )
                        else "rejected"
                    ),
                }
            )
            view["visibility_acceptance"] = visibility_acceptance


def build_zoom_view_plan(
    base_view_plan: dict[str, dict[str, Any]],
    *,
    zoom_offset: int,
    zooms_per_view: int,
) -> dict[str, dict[str, Any]]:
    """Materialize one zoom plan for every selected camera."""
    result: dict[str, dict[str, Any]] = {}
    for camera_id, base_view in base_view_plan.items():
        variants_for_view = list(base_view["zoom_variants"])
        if len(variants_for_view) != zooms_per_view:
            raise RuntimeError(
                f"incomplete zoom set for {base_view.get('state_id')} {camera_id}"
            )
        zoom_view = dict(base_view)
        zoom_view.pop("zoom_variants", None)
        zoom_view["base_focal_length_mm"] = float(base_view["focal_length_mm"])
        zoom_view["base_focal_length_35mm_equivalent_mm"] = float(
            base_view["focal_length_35mm_equivalent_mm"]
        )
        zoom_view.update(dict(variants_for_view[zoom_offset]))
        zoom_view["zoom_set_size"] = zooms_per_view
        result[camera_id] = zoom_view
    return result



def play_simulation_without_capture(
    *,
    app: Any,
    variants: Any,
    scenario: Any,
    stage: Any,
    camera_prims: list[Any],
    anchor: tuple[float, float],
    resolution: tuple[int, int],
    state_ids: list[str],
    seconds_per_day: float,
    progress_path: Path | None,
    incident_days: int,
    states_per_day: int,
) -> int:
    import omni.timeline

    seconds_per_state = seconds_per_day / states_per_day
    started = time.monotonic()
    previous: dict[str, Any] | None = None
    fixed_ignition: list[float] | None = None
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0.0)
    timeline.set_end_time(incident_days * seconds_per_day + seconds_per_state)
    timeline.set_current_time(0.0)
    timeline.set_looping(False)
    timeline.play()
    app.update()
    print(json.dumps({"status": "flow_timeline_playing", "seconds_per_day": seconds_per_day, "seconds_per_state": seconds_per_state}, sort_keys=True), flush=True)
    try:
        for index, state_id in enumerate(state_ids, start=1):
            print(json.dumps({"status": "state_selection_started", "state_id": state_id, "state_index": index}, sort_keys=True), flush=True)
            variants.SetVariantSelection(state_id)
            print(json.dumps({"status": "state_variant_selected", "state_id": state_id, "state_index": index}, sort_keys=True), flush=True)
            app.update()
            hidden_tree_count = apply_burned_tree_destruction(stage)
            hidden_truth_proxies = hide_truth_proxies_for_beauty(stage)
            flow_probe = probe_flow_state(stage)
            print(json.dumps({"status": "state_composition_updated", "state_id": state_id, "state_index": index}, sort_keys=True), flush=True)
            metrics = fire_state_metrics(scenario)
            dynamic_targets = (
                retarget_cameras_for_state(
                    stage=stage,
                    camera_prims=camera_prims,
                    anchor=anchor,
                    resolution=resolution,
                )
                if metrics["active_in_scene"]
                else {}
            )
            print(json.dumps({"status": "state_cameras_retargeted", "state_id": state_id, "state_index": index, "cameras": len(dynamic_targets)}, sort_keys=True), flush=True)
            app.update()
            if hidden_tree_count != int(metrics["burned_tree_count"]):
                raise RuntimeError(f"tree destruction count mismatch for {state_id}: {hidden_tree_count} != {metrics['burned_tree_count']}")
            fixed_ignition = validate_fire_state_progression(
                metrics=metrics,
                expected_index=index,
                previous=previous,
                fixed_ignition=fixed_ignition,
                states_per_day=states_per_day,
                expected_states=len(state_ids),
            )
            previous = metrics
            state_deadline = started + index * seconds_per_state
            timeline.set_current_time(min(time.monotonic() - started, incident_days * seconds_per_day))
            app.update()
            progress = {
                "status": "simulation_without_capture_running",
                "state_id": state_id,
                "completed_states": index,
                "expected_states": len(state_ids),
                "day_index": metrics["day_index"],
                "state_in_day": metrics["state_in_day"],
                "seconds_per_day": seconds_per_day,
                "seconds_per_state": seconds_per_state,
                "target_total_duration_s": incident_days * seconds_per_day,
                "wall_elapsed_s": round(time.monotonic() - started, 3),
                "flow_timeline_seconds": round(float(timeline.get_current_time()), 3),
                "hidden_source_tree_count": hidden_tree_count,
                "hidden_beauty_truth_proxy_count": len(hidden_truth_proxies),
                **flow_probe,
                "dynamically_retargeted_cameras": len(dynamic_targets),
                "positive_targeting_contract": "nearest_fire_front_point_centered",
                "negative_targeting_contract": "rotated_away_from_nearest_fire_front_point",
                **metrics,
            }
            write_progress(progress_path, progress)
            print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
            while time.monotonic() < state_deadline:
                timeline.set_current_time(min(time.monotonic() - started, incident_days * seconds_per_day))
                app.update()
                time.sleep(0.001)
    finally:
        timeline.stop()
    completed = {
        "status": "simulation_without_capture_complete",
        "states": len(state_ids),
        "incident_days": incident_days,
        "seconds_per_day": seconds_per_day,
        "seconds_per_state": seconds_per_state,
        "target_total_duration_s": incident_days * seconds_per_day,
        "wall_elapsed_s": round(time.monotonic() - started, 3),
        "ignition_l93_m": fixed_ignition,
    }
    write_progress(progress_path, completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)
    return 0


def apply_fireviewer_semantics(stage: Any) -> None:
    from isaacsim.core.experimental.utils import semantics as semantics_utils

    for prim in stage.Traverse():
        label = prim.GetAttribute("fireviewer:semantic_class").Get()
        if isinstance(label, str) and label:
            semantics_utils.add_labels(prim, labels=[label], taxonomy="class")


def camera_geolocation(camera: Any, *, anchor: tuple[float, float]) -> dict[str, Any]:
    from pxr import UsdGeom

    translate = camera.GetAttribute("xformOp:translate").Get()
    if translate is None:
        raise RuntimeError(f"camera has no explicit translate op: {camera.GetPath()}")
    local = [float(translate[0]), float(translate[1]), float(translate[2])]
    target = camera.GetAttribute("fireviewer:look_at_local_m").Get()
    target_values = [float(item) for item in target] if target is not None else None
    absolute = camera.GetAttribute("fireviewer:position_l93_ngf_ign69_m").Get()
    fire_target = camera.GetAttribute("fireviewer:fire_target_local_m").Get()
    result = {
        "camera_id": str(camera.GetAttribute("fireviewer:camera_id").Get()),
        "camera_path": str(camera.GetPath()),
        "position_local_m": local,
        "look_at_local_m": target_values,
        "fire_target_local_m": [float(item) for item in fire_target] if fire_target is not None else None,
        "position_epsg2154_ngf_ign69_m": [float(item) for item in absolute] if absolute is not None else [anchor[0] + local[0], anchor[1] + local[1], local[2]],
        "focal_length_mm": float(UsdGeom.Camera(camera).GetFocalLengthAttr().Get()),
    }
    for name in (
        "camera_role", "placement_type", "placement_contract", "access_surface", "access_tile",
        "host_building", "target_kind", "sample_capability", "expected_fire_in_frame", "thermal_capture",
        "capture_device_profile", "framing_style", "focal_length_35mm_equivalent_mm",
        "line_of_sight_verified", "eye_height_m", "height_above_ground_m", "ground_elevation_m",
        "tree_clearance_m", "building_clearance_m", "terrain_los_clearance_m", "foreground_clearance_m",
        "distance_to_target_m", "fire_bearing_offset_degrees",
    ):
        value = camera.GetAttribute(f"fireviewer:{name}").Get()
        result[name] = value
    return result


def run(args: argparse.Namespace) -> int:
    stage_path = args.stage.resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    if args.dry_run:
        return dry_run(
            stage_path,
            dataset_id=args.dataset_id,
            pilot_state_indices=args.pilot_state_indices,
            production_state_indices=args.production_state_indices,
            production_chunk_id=args.production_chunk_id,
            pilot_acceptance_report=args.pilot_acceptance_report,
            visual_calibration_camera_ids=args.visual_calibration_camera_ids,
            resolution=args.resolution,
            rt_subframes=args.rt_subframes,
            seconds_per_day=args.seconds_per_day,
            flow_warmup_updates=args.flow_warmup_updates,
            render_product_batch_size=args.render_product_batch_size,
        )
    if args.output_root is None and not args.simulation_only:
        raise ValueError("--output-root is required unless --dry-run or --simulation-only")
    if args.frames_per_state != 1:
        raise ValueError("--frames-per-state must be exactly 1 so every capture_id remains unique")
    if args.rt_subframes < 1 or args.seconds_per_day <= 0 or args.flow_warmup_updates < 1:
        raise ValueError("--rt-subframes, --seconds-per-day and --flow-warmup-updates must be positive")
    if args.render_product_batch_size < 1:
        raise ValueError("--render-product-batch-size must be positive")
    if args.visible_view_updates < 1:
        raise ValueError("--visible-view-updates must be positive")
    if args.simulation_only and args.production_state_indices is not None:
        raise ValueError(
            "--production-state-indices cannot be combined with --simulation-only"
        )
    resolution = parse_resolution(args.resolution)
    package = stage_path.parent
    manifest = read_json(package / "manifest.json")
    runtime = read_json(package / "runtime/runtime-contract.json")
    propagation = read_json(package / "scenarios/propagation.json")
    incident_days = int(runtime["capture"]["incident_days"])
    states_per_day = int(runtime["capture"]["states_per_day"])
    expected_state_count = incident_days * states_per_day
    if propagation.get("state_count") != expected_state_count or propagation.get("simulation_id") != manifest["scenario"]["propagation"]["simulation_id"]:
        raise RuntimeError("propagation sidecar does not match the dataset manifest")
    schedule = load_capture_schedule(package, manifest, runtime)
    production_dataset_id = validate_dataset_id(
        args.dataset_id,
        fallback=str(manifest["package_id"]),
    )
    state_selector, state_selector_option, selection_run_kind = resolve_state_selection(
        pilot_state_indices=args.pilot_state_indices,
        production_state_indices=args.production_state_indices,
    )
    production_chunk_id = resolve_production_chunk_id(
        production_state_indices=args.production_state_indices,
        production_chunk_id=args.production_chunk_id,
        pilot_acceptance_report=args.pilot_acceptance_report,
    )
    selected_states = selected_schedule_states(
        schedule,
        state_selector,
        option_name=state_selector_option,
    )
    all_scheduled_camera_ids = sorted(
        {
            str(camera_id)
            for state in selected_states
            for camera_id in state["camera_ids"]
        }
    )
    if args.visual_calibration_camera_ids is not None and args.pilot_state_indices is None:
        raise ValueError(
            "--visual-calibration-camera-ids is allowed only with --pilot-state-indices"
        )
    if (
        args.visual_calibration_camera_ids is not None
        and args.pilot_camera_priority is not None
    ):
        raise ValueError(
            "--visual-calibration-camera-ids cannot be combined with --pilot-camera-priority"
        )
    visual_calibration_camera_ids = parse_visual_calibration_camera_ids(
        args.visual_calibration_camera_ids,
        available_camera_ids=set(all_scheduled_camera_ids),
    )
    selected_counts = (
        selected_visual_calibration_counts(
            selected_states,
            camera_ids=visual_calibration_camera_ids,
            zooms_per_view=int(schedule["zooms_per_view"]),
            frames_per_state=int(args.frames_per_state),
        )
        if visual_calibration_camera_ids
        else selected_capture_counts(
            selected_states,
            zooms_per_view=int(schedule["zooms_per_view"]),
            frames_per_state=int(args.frames_per_state),
        )
    )
    scheduled_camera_ids = (
        visual_calibration_camera_ids
        if visual_calibration_camera_ids
        else all_scheduled_camera_ids
    )
    run_kind = (
        "visual_calibration"
        if visual_calibration_camera_ids
        else selection_run_kind
    )
    if args.pilot_camera_priority is not None and args.pilot_state_indices is None:
        raise ValueError("--pilot-camera-priority is allowed only with --pilot-state-indices")
    pilot_camera_priority = parse_camera_priority(
        args.pilot_camera_priority,
        available_camera_ids=set(scheduled_camera_ids),
    )
    flow_profile_contract = flow_capture_plume_profile_contract()
    sky_profile_contract = sky_capture_profile_contract()
    capture_storage_profile_contract = storage_profile_contract()
    source_package_id = str(manifest["package_id"])
    source_stage_sha256 = sha256_file(stage_path)
    pilot_acceptance = (
        load_pilot_acceptance_receipt(
            args.pilot_acceptance_report,
            source_package_id=source_package_id,
            source_stage_sha256=source_stage_sha256,
            resolution_px=resolution,
            rt_subframes=args.rt_subframes,
            flow_warmup_updates=args.flow_warmup_updates,
            playback_seconds_per_day=args.seconds_per_day,
            flow_profile_sha256=str(flow_profile_contract["profile_sha256"]),
            sky_profile_sha256=str(sky_profile_contract["profile_sha256"]),
            capture_storage_profile_sha256=str(
                capture_storage_profile_contract["profile_sha256"]
            ),
        )
        if args.pilot_acceptance_report is not None
        else None
    )
    case_id = str(manifest.get("case_id") or source_package_id)
    map_package_id = str(
        (manifest.get("base_scene") or {}).get("package_id") or source_package_id
    )
    scenario_id = str(propagation["simulation_id"])
    propagation_states = {
        str(state["state_id"]): state for state in propagation.get("states", [])
    }
    environment_drivers = propagation.get("drivers") or {
        "simulation_adaptation": propagation.get("simulation_adaptation"),
        "visual_intensity": propagation.get("visual_intensity"),
        "terrain": propagation.get("terrain"),
    }
    weather_state_id = "weather-" + hashlib.sha256(
        json.dumps(environment_drivers, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    dependency_paths = {
        "stage": stage_path,
        "manifest": package / "manifest.json",
        "runtime_contract": package / "runtime" / "runtime-contract.json",
        "capture_schedule": package / "runtime" / "capture-schedule.json",
        "propagation": package / "scenarios" / "propagation.json",
        "scenario_layer": package / "scenarios" / "scenario.usda",
        "flow_layer": package / "scenarios" / "flow.usda",
        "camera_rig": package / "cameras" / "fixed_cameras.usda",
        "executing_runner": Path(__file__).resolve(),
        "executing_writer": Path(__file__).resolve().parent / "fireviewer_replicator_writer.py",
        "metadata_auditor": Path(__file__).resolve().parent / "audit_fireviewer_capture_metadata.py",
        "capture_storage": Path(__file__).resolve().parent / "fireviewer_capture_storage.py",
    }
    dependency_sha256 = {
        name: sha256_file(path) for name, path in dependency_paths.items() if path.is_file()
    }
    synthetic_epoch = datetime(2026, 7, 3, tzinfo=timezone.utc)
    output_root = args.output_root.resolve() if args.output_root is not None else None
    capture_progress_path = args.progress_path
    if capture_progress_path is None and output_root is not None:
        capture_progress_path = output_root / "capture-progress.json"
    if output_root is not None:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(f"Refusing to merge captures into a non-empty output: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        write_progress(
            output_root / "run-contract.json",
            {
                "schema": "fireviewer.kit-dataset-production-run.v1",
                "run_kind": run_kind,
                "production_chunk_id": production_chunk_id,
                "dataset_id": production_dataset_id,
                "source_package_id": source_package_id,
                "source_stage": str(stage_path),
                "source_stage_sha256": source_stage_sha256,
                "source_manifest_sha256": sha256_file(package / "manifest.json"),
                "capture_schedule_sha256": sha256_file(package / "runtime" / "capture-schedule.json"),
                "selected_state_indices": [int(state["global_state_index"]) for state in selected_states],
                "selected_state_ids": [str(state["state_id"]) for state in selected_states],
                "selected_state_count": len(selected_states),
                "resolution_px": [int(resolution[0]), int(resolution[1])],
                "frames_per_state": int(args.frames_per_state),
                "rt_subframes": int(args.rt_subframes),
                "expected_viewpoint_plans": selected_counts["viewpoint_plans"],
                "expected_capture_cases": selected_counts["capture_cases"],
                "expected_positive_cases": selected_counts["positive_cases"],
                "expected_negative_cases": selected_counts["negative_cases"],
                "scheduled_camera_ids": scheduled_camera_ids,
                "scheduled_camera_count": len(scheduled_camera_ids),
                "render_product_batch_size": int(args.render_product_batch_size),
                "max_active_render_product_count": min(
                    int(args.render_product_batch_size), len(scheduled_camera_ids)
                ),
                "kit_session_contract": "single_persistent_session_single_stage_load",
                "render_product_contract": "persistent_pool_rebound_to_sequential_camera_batches",
                "full_dataset_expected_capture_cases": int(schedule["expected_capture_cases"]),
                "full_dataset_capture_authorized": run_kind == "production_chunk",
                "training_admission_authorized": False,
                "kit_window_visible": not bool(args.headless),
                "viewport_capture_presentation": (
                    "sequential_camera_then_five_zooms_after_each_view_change"
                    if not args.headless
                    else "headless_sequential_render_product_batches"
                ),
                "simulation_playing_during_visible_capture": not bool(args.headless),
                "playback_seconds_per_day": float(args.seconds_per_day),
                "flow_capture_preparation_contract": "independent_volume_clear_and_fixed_state_time_warmup_v1",
                "flow_warmup_updates": int(args.flow_warmup_updates),
                "flow_capture_profile": flow_profile_contract,
                "sky_capture_profile": sky_profile_contract,
                "capture_storage_profile": capture_storage_profile_contract,
                "pilot_camera_priority": pilot_camera_priority,
                "pilot_camera_priority_contract": (
                    "presentation_order_only_full_state_quota_preserved"
                    if pilot_camera_priority
                    else None
                ),
                "visual_calibration_camera_ids": visual_calibration_camera_ids,
                "visual_calibration_contract": (
                    "non_dataset_visual_gate_all_five_zooms_preserved"
                    if visual_calibration_camera_ids
                    else None
                ),
                "production_chunk_contract": (
                    "accepted_pilot_gated_disjoint_state_subset_with_global_capture_ids"
                    if run_kind == "production_chunk"
                    else None
                ),
                "pilot_acceptance": pilot_acceptance,
                "dataset_admissible": not bool(visual_calibration_camera_ids),
                "kit_cache_root": str(
                    args.kit_cache_root.resolve()
                    if args.kit_cache_root is not None
                    else (
                        (
                            args.progress_path.resolve().parent
                            if args.progress_path is not None
                            else package / "runtime"
                        )
                        / "kit-cache"
                    )
                ),
            },
        )
    startup_progress = {
        "status": "runtime_initializing",
        "stage": str(stage_path),
        "dataset_id": production_dataset_id,
        "source_package_id": source_package_id,
        "run_kind": run_kind,
        "production_chunk_id": production_chunk_id,
        "pilot_acceptance_report_sha256": (
            pilot_acceptance["audit_report_sha256"] if pilot_acceptance else None
        ),
        "selected_state_indices": [int(state["global_state_index"]) for state in selected_states],
        "expected_capture_cases": selected_counts["capture_cases"],
        "simulation_only": bool(args.simulation_only),
        "headless": bool(args.headless),
        "render_product_batch_size": int(args.render_product_batch_size),
        "seconds_per_day": float(args.seconds_per_day),
        "flow_warmup_updates": int(args.flow_warmup_updates),
        "flow_capture_profile_id": flow_profile_contract["profile_id"],
        "flow_capture_profile_sha256": flow_profile_contract["profile_sha256"],
        "sky_capture_profile_id": sky_profile_contract["profile_id"],
        "sky_capture_profile_sha256": sky_profile_contract["profile_sha256"],
        "capture_storage_profile_id": capture_storage_profile_contract["profile_id"],
        "capture_storage_profile_sha256": capture_storage_profile_contract[
            "profile_sha256"
        ],
        "pilot_camera_priority": pilot_camera_priority,
        "visual_calibration_camera_ids": visual_calibration_camera_ids,
    }
    write_progress(capture_progress_path, startup_progress)
    print(json.dumps(startup_progress, ensure_ascii=False, sort_keys=True), flush=True)

    cache_root = (
        args.kit_cache_root.resolve()
        if args.kit_cache_root is not None
        else (
            (
                args.progress_path.resolve().parent
                if args.progress_path is not None
                else package / "runtime"
            )
            / "kit-cache"
        )
    )
    optix_cache = cache_root / "optix"
    ujitso_cache = cache_root / "derived-data"
    warp_cache = cache_root / "warp"
    portable_root = cache_root / "portable"
    optix_cache.mkdir(parents=True, exist_ok=True)
    ujitso_cache.mkdir(parents=True, exist_ok=True)
    warp_cache.mkdir(parents=True, exist_ok=True)
    portable_root.mkdir(parents=True, exist_ok=True)
    os.environ["OPTIX_CACHE_PATH"] = str(optix_cache)
    os.environ["WARP_CACHE_PATH"] = str(warp_cache)
    from isaacsim import SimulationApp

    isaac_root = Path(sys.executable).resolve().parents[2]
    experience_name = "isaacsim.exp.base.python.kit"
    experience_path = isaac_root / "apps" / experience_name
    if not experience_path.is_file():
        raise RuntimeError(f"required Isaac Sim Flow experience is missing: {experience_path}")
    if "--portable-root" not in sys.argv:
        sys.argv.extend(["--portable-root", str(portable_root)])
    launch_config = {
        "headless": bool(args.headless),
        "hide_ui": bool(args.headless),
        "width": 1600,
        "height": 900,
        "renderer": "MinimalRendering" if args.headless and args.simulation_only else "RaytracedLighting",
        "disable_viewport_updates": bool(args.headless and args.simulation_only),
        "extra_args": [
            "--enable",
            "omni.flowusd",
            "--/UJITSO/datastore/allowHubDataStore=false",
            f"--/UJITSO/datastore/localCachePath={ujitso_cache}",
        ],
    }
    print(
        json.dumps(
            {
                "status": "kit_cache_ready",
                "optix_cache": str(optix_cache),
                "ujitso_cache": str(ujitso_cache),
                "warp_cache": str(warp_cache),
                "portable_root": str(portable_root),
                "experience": str(experience_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    app = SimulationApp(launch_config, experience=str(experience_path))
    try:
        write_progress(
            capture_progress_path,
            {**startup_progress, "status": "simulation_app_ready"},
        )
        print(json.dumps({"status": "simulation_app_ready"}, sort_keys=True), flush=True)
        import carb.settings
        print(json.dumps({"status": "carb_ready"}, sort_keys=True), flush=True)
        import omni.replicator.core as rep
        print(json.dumps({"status": "replicator_ready"}, sort_keys=True), flush=True)
        import omni.usd
        from pxr import UsdGeom
        print(json.dumps({"status": "usd_ready"}, sort_keys=True), flush=True)

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fireviewer_replicator_writer import FireViewerWriter, register
        print(json.dumps({"status": "writer_module_ready"}, sort_keys=True), flush=True)

        require_flow_extension(app)
        print(json.dumps({"status": "flow_extension_ready"}, sort_keys=True), flush=True)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
        configure_flow_renderer()
        configure_daylight_renderer()
        context = omni.usd.get_context()
        context.open_stage(str(stage_path))
        stage = wait_for_stage(app, context)
        stage.Load()
        app.update()
        write_progress(
            capture_progress_path,
            {**startup_progress, "status": "stage_loaded", "stage": str(stage_path)},
        )
        print(json.dumps({"status": "stage_loaded", "stage": str(stage_path)}, sort_keys=True), flush=True)
        world = stage.GetPrimAtPath("/World")
        world_anchor = world.GetAttribute("fireviewer:common_anchor_l93_m").Get()
        if world_anchor is None or len(world_anchor) != 2:
            raise RuntimeError("dataset stage has no EPSG:2154 anchor metadata")
        anchor = tuple(float(value) for value in world_anchor)
        scenario = stage.GetPrimAtPath("/World/FireScenario")
        variants = scenario.GetVariantSets().GetVariantSet("fire_state")
        state_ids = list(variants.GetVariantNames())
        expected_states = [
            f"state_{index:03d}" for index in range(1, expected_state_count + 1)
        ]
        if state_ids != expected_states:
            raise RuntimeError(f"Unexpected fire state variants: {state_ids}")
        print(json.dumps({"status": "fire_variants_ready", "states": len(state_ids)}, sort_keys=True), flush=True)
        camera_scope = stage.GetPrimAtPath("/World/Cameras")
        camera_prims = [child for child in camera_scope.GetChildren() if child.IsA(UsdGeom.Camera)]
        camera_prims.sort(key=lambda prim: str(prim.GetAttribute("fireviewer:camera_id").Get()))
        expected_camera_count = int(manifest["cameras"]["fixed_count"])
        if len(camera_prims) != expected_camera_count:
            raise RuntimeError(
                f"Expected a {expected_camera_count}-camera pool, received {len(camera_prims)}"
            )
        cameras_by_id = {
            str(camera.GetAttribute("fireviewer:camera_id").Get()): camera
            for camera in camera_prims
        }
        missing_selected_cameras = sorted(set(scheduled_camera_ids) - set(cameras_by_id))
        if missing_selected_cameras:
            raise RuntimeError(
                f"scheduled cameras are absent from the USD stage: {missing_selected_cameras}"
            )
        base_focal_lengths = {
            camera_id: float(UsdGeom.Camera(camera).GetFocalLengthAttr().Get())
            for camera_id, camera in cameras_by_id.items()
        }
        print(json.dumps({"status": "camera_pool_ready", "cameras": len(camera_prims)}, sort_keys=True), flush=True)
        if not args.headless:
            configure_visible_viewport(
                app,
                camera_prims,
                preferred_camera_id=str(args.visible_camera),
                settle_updates=12,
            )
        if args.simulation_only:
            return play_simulation_without_capture(
                app=app,
                variants=variants,
                scenario=scenario,
                stage=stage,
                camera_prims=camera_prims,
                anchor=anchor,
                resolution=resolution,
                state_ids=expected_states,
                seconds_per_day=float(args.seconds_per_day),
                progress_path=args.progress_path,
                incident_days=incident_days,
                states_per_day=states_per_day,
            )
        sky_capture_state = apply_sky_capture_visibility_profile(stage)
        base_flow_capture_profile = apply_flow_capture_plume_profile(stage)
        apply_fireviewer_semantics(stage)
        raw_ignition_l93 = list(propagation.get("ignition_l93_m") or [anchor[0], anchor[1]])
        if len(raw_ignition_l93) < 2:
            raise RuntimeError("propagation sidecar has no valid EPSG:2154 ignition point")
        ignition_x_l93 = float(raw_ignition_l93[0])
        ignition_y_l93 = float(raw_ignition_l93[1])
        ignition_x_local = ignition_x_l93 - anchor[0]
        ignition_y_local = ignition_y_l93 - anchor[1]
        if len(raw_ignition_l93) >= 3:
            ignition_elevation = float(raw_ignition_l93[2])
            ignition_elevation_source = "propagation_sidecar"
            ignition_mnt_sample_distance_m = 0.0
        else:
            ignition_elevation, ignition_mnt_sample_distance_m = terrain_elevation_at(
                stage,
                x_local_m=ignition_x_local,
                y_local_m=ignition_y_local,
            )
            ignition_elevation_source = "accepted_mnt_nearest_vertex"
        ignition_l93 = [ignition_x_l93, ignition_y_l93, ignition_elevation]
        ignition_local = [ignition_x_local, ignition_y_local, ignition_elevation]
        print(
            json.dumps(
                {
                    "status": "ignition_geolocation_ready",
                    "ignition_l93_ngf_ign69_m": ignition_l93,
                    "ignition_local_m": ignition_local,
                    "elevation_source": ignition_elevation_source,
                    "mnt_sample_distance_m": ignition_mnt_sample_distance_m,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        import omni.timeline

        register()
        writer = rep.WriterRegistry.get("FireViewerWriter")
        writer.initialize(output_dir=str(output_root))
        render_product_pool: list[Any] = []
        initial_camera_ids = scheduled_camera_ids[
            : min(int(args.render_product_batch_size), len(scheduled_camera_ids))
        ]
        for camera_id in initial_camera_ids:
            camera = cameras_by_id[camera_id]
            render_product_pool.append(
                rep.create.render_product(str(camera.GetPath()), resolution)
            )
        if not render_product_pool:
            raise RuntimeError("capture schedule selected no camera render products")
        write_progress(
            capture_progress_path,
            {
                **startup_progress,
                "status": "render_product_pool_ready",
                "render_product_pool_size": len(render_product_pool),
                "scheduled_camera_count": len(scheduled_camera_ids),
            },
        )
        rep.orchestrator.set_capture_on_play(False)
        capture_timeline = omni.timeline.get_timeline_interface()
        capture_timeline.set_start_time(0.0)
        capture_timeline.set_end_time(
            incident_days * float(args.seconds_per_day)
            + float(args.seconds_per_day / states_per_day)
        )
        capture_timeline.set_current_time(0.0)
        capture_timeline.set_looping(False)
        capture_timeline.pause()
        captures = 0
        capture_started = time.monotonic()
        for selected_sequence, scheduled_state in enumerate(selected_states, start=1):
            state_id = str(scheduled_state["state_id"])
            playback_elapsed_s = (
                (int(scheduled_state["global_state_index"]) - 1)
                * args.seconds_per_day
                / states_per_day
            )
            capture_timeline.pause()
            capture_timeline.set_current_time(float(playback_elapsed_s))
            variants.SetVariantSelection(state_id)
            app.update()
            scheduled_state_metrics = scheduled_state["views"][0]
            flow_capture_profile = configure_flow_capture_plume_for_state(
                stage,
                state_metrics={
                    "fire_elapsed_s": float(
                        scheduled_state_metrics["fire_state_elapsed_s"]
                    ),
                    "burned_area_m2": float(
                        scheduled_state_metrics["burned_area_m2"]
                    ),
                    "mean_front_spread_rate_m_s": float(
                        scheduled_state_metrics["mean_front_spread_rate_m_s"]
                    ),
                },
                base_profile_receipt=base_flow_capture_profile,
            )
            flow_capture_state = prepare_flow_state_for_capture(
                app=app,
                stage=stage,
                timeline=capture_timeline,
                target_time_s=float(playback_elapsed_s),
                warmup_updates=int(args.flow_warmup_updates),
                flow_capture_profile=flow_capture_profile,
            )
            hidden_tree_count = apply_burned_tree_destruction(stage)
            scheduled_burned_tree_count = int(scheduled_state["views"][0]["burned_tree_count"])
            if hidden_tree_count != scheduled_burned_tree_count:
                raise RuntimeError(f"tree destruction count mismatch for {state_id}: {hidden_tree_count} != {scheduled_burned_tree_count}")
            truth_geometry = load_truth_projection_geometry(stage)
            base_view_plan = {
                str(view["camera_id"]): dict(view) for view in scheduled_state["views"]
            }
            author_camera_focal_lengths(stage, cameras_by_id, base_focal_lengths)
            dynamic_targets = retarget_cameras_for_state(
                stage=stage,
                camera_prims=camera_prims,
                anchor=anchor,
                resolution=resolution,
                view_plan_by_camera=base_view_plan,
                truth_geometry=truth_geometry,
            )
            propagation_state = propagation_states.get(state_id, {})
            observation_elapsed_s = int(
                scheduled_state["views"][0].get("observation_elapsed_s", 0)
            )
            valid_at = propagation_state.get("valid_at")
            if not valid_at:
                valid_at = (
                    synthetic_epoch + timedelta(seconds=observation_elapsed_s)
                ).isoformat().replace("+00:00", "Z")
            for camera_id, view in base_view_plan.items():
                view.update(dynamic_targets[camera_id])
                view["playback_seconds_per_day"] = float(args.seconds_per_day)
                view["playback_seconds_per_state"] = float(
                    args.seconds_per_day / states_per_day
                )
                view["playback_elapsed_s"] = float(playback_elapsed_s)
                view_observation_elapsed_s = int(view["observation_elapsed_s"])
                time_of_day_s = view_observation_elapsed_s % 86400
                hours, remainder = divmod(time_of_day_s, 3600)
                minutes, seconds = divmod(remainder, 60)
                view["simulation_time_of_day_s"] = time_of_day_s
                view["simulation_time_of_day_hhmmss"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                view["simulation_clock_contract"] = (
                    "source_retrospective_utc_datetime"
                    if propagation_state.get("valid_at")
                    else "relative_synthetic_incident_clock_no_claim_of_real_incident_datetime"
                )
                view["dataset_id"] = production_dataset_id
                view["run_kind"] = run_kind
                view["production_chunk_id"] = production_chunk_id
                view["pilot_acceptance_report_sha256"] = (
                    pilot_acceptance["audit_report_sha256"]
                    if pilot_acceptance
                    else None
                )
                view["source_package_id"] = source_package_id
                view["case_id"] = case_id
                view["map_package_id"] = map_package_id
                view["scenario_id"] = scenario_id
                view["simulation_id"] = scenario_id
                view["valid_at"] = str(valid_at)
                view["timezone"] = "UTC"
                view["weather_state_id"] = weather_state_id
                view["source_stage_sha256"] = dependency_sha256["stage"]
                view["dependency_sha256"] = dependency_sha256
                view["capture_storage_profile"] = capture_storage_profile_contract
                view["source_provenance_ids"] = list(
                    propagation_state.get("source_revision_ids") or []
                )
                view["propagation_model"] = propagation.get("model") or "daily_retrospective_perimeter_replay"
                view["propagation_solver"] = propagation.get("solver") or "source_constrained_daily_state_replay"
                view["environment_drivers"] = environment_drivers
                view["flow_capture_state"] = flow_capture_state
                view["sky_capture_state"] = sky_capture_state
                view["fire_visibility_regime"] = flow_capture_profile[
                    "state_visibility_regime"
                ]["regime"]
                view["ignition_l93_m"] = ignition_l93
                view["ignition_point_local_m"] = ignition_local
                view["ignition_elevation_source"] = ignition_elevation_source
                view["domain_bounds_l93_m"] = propagation["domain_bounds_l93_m"]
                view["fuel_input"] = propagation.get("fuel_input") or propagation.get("vegetation")
                view["truth_scope"] = manifest["scenario"]["truth_scope"]
                view["line_of_sight_receipt"] = {
                    "verified": bool(view.get("line_of_sight_verified")),
                    "terrain_clearance_m": view.get("terrain_los_clearance_m"),
                    "foreground_clearance_m": view.get("foreground_clearance_m"),
                    "distance_to_target_m": view.get("distance_to_target_m"),
                }
                modalities = list(view.get("expected_modalities", []))
                for modality in ("instance_segmentation", "depth_distance_to_camera_m", "depth_metadata", "normals", "normals_metadata", "flame_mask", "smoke_mask", "burned_area_mask", "smoke_source_mask", "nearest_flame_keypoint_3d_2d", "nearest_smoke_keypoint_3d_2d", "dynamic_camera_pose"):
                    if modality not in modalities:
                        modalities.append(modality)
                view["expected_modalities"] = modalities
            state_camera_ids = [
                str(camera_id) for camera_id in scheduled_state["camera_ids"]
            ]
            if visual_calibration_camera_ids:
                state_camera_id_set = set(state_camera_ids)
                state_camera_ids = [
                    camera_id
                    for camera_id in visual_calibration_camera_ids
                    if camera_id in state_camera_id_set
                ]
                if len(state_camera_ids) != len(visual_calibration_camera_ids):
                    raise RuntimeError(
                        f"visual calibration camera selection drifted for {state_id}"
                    )
            else:
                state_camera_ids = prioritize_camera_ids(
                    state_camera_ids, pilot_camera_priority
                )
            camera_batches = split_camera_batches(
                state_camera_ids, int(args.render_product_batch_size)
            )
            zooms_per_view = int(schedule["zooms_per_view"])
            capture_sequence_cursor = 0
            for batch_index, camera_batch in enumerate(camera_batches, start=1):
                active_products = render_product_pool[: len(camera_batch)]
                for render_product, camera_id in zip(active_products, camera_batch):
                    render_product_path = str(getattr(render_product, "path", ""))
                    if not render_product_path.startswith("/Render/"):
                        raise RuntimeError(
                            f"invalid render product path: {render_product_path!r}"
                        )
                    render_product.hydra_texture.set_camera_path(
                        str(cameras_by_id[camera_id].GetPath())
                    )
                    writer.register_render_product(
                        render_product_path,
                        camera_id,
                        camera_geolocation(cameras_by_id[camera_id], anchor=anchor),
                    )
                app.update()
                write_progress(
                    capture_progress_path,
                    {
                        "status": "capture_batch_ready",
                        "dataset_id": production_dataset_id,
                        "run_kind": run_kind,
                        "production_chunk_id": production_chunk_id,
                        "state_id": state_id,
                        "selected_state_sequence": selected_sequence,
                        "batch_index": batch_index,
                        "batch_count": len(camera_batches),
                        "camera_ids": camera_batch,
                        "captures": captures,
                        "expected_captures": selected_counts["capture_cases"],
                        "render_product_pool_size": len(render_product_pool),
                        "kit_session_reused": True,
                    },
                )
                camera_base_view_plan = {
                    camera_id: base_view_plan[camera_id] for camera_id in camera_batch
                }
                for zoom_offset in range(zooms_per_view):
                    zoom_view_plan = build_zoom_view_plan(
                        camera_base_view_plan,
                        zoom_offset=zoom_offset,
                        zooms_per_view=zooms_per_view,
                    )
                    apply_zoom_metadata(
                        stage=stage,
                        cameras_by_id=cameras_by_id,
                        view_plan_by_camera=zoom_view_plan,
                        resolution=resolution,
                        truth_geometry=truth_geometry,
                    )
                    for camera_offset, camera_id in enumerate(camera_batch, start=1):
                        zoom_view_plan[camera_id]["capture_timeline_time_s"] = float(
                            playback_elapsed_s
                        )
                        zoom_view_plan[camera_id]["capture_sequence_in_state"] = int(
                            capture_sequence_cursor + camera_offset
                        )
                        zoom_view_plan[camera_id]["render_product_batch_index"] = int(
                            batch_index
                        )
                        zoom_view_plan[camera_id]["render_product_batch_size"] = int(
                            len(camera_batch)
                        )
                        zoom_view_plan[camera_id]["timeline_playing_during_transition"] = False
                        zoom_view_plan[camera_id]["capture_trigger_contract"] = (
                            "state_variant_played_once_then_persistent_render_product_pool_rebound_and_captured"
                        )
                    writer.set_capture_context(
                        package_id=production_dataset_id,
                        source_package_id=source_package_id,
                        incident_id=schedule["incident_id"],
                        state_id=state_id,
                        day_index=int(scheduled_state["day_index"]),
                        state_in_day=int(scheduled_state["state_in_day"]),
                        view_plan_by_camera=zoom_view_plan,
                    )
                    if args.headless:
                        writer.attach(active_products)
                        rep.orchestrator.step(
                            rt_subframes=args.rt_subframes,
                            delta_time=1.0 / 30.0,
                            pause_timeline=True,
                        )
                        writer.detach()
                        captures += len(active_products)
                    else:
                        for product_offset, camera_id in enumerate(camera_batch):
                            configure_visible_viewport(
                                app,
                                [cameras_by_id[camera_id]],
                                preferred_camera_id=camera_id,
                                settle_updates=int(args.visible_view_updates),
                            )
                            writer.attach([active_products[product_offset]])
                            rep.orchestrator.step(
                                rt_subframes=args.rt_subframes,
                                delta_time=1.0 / 30.0,
                                pause_timeline=True,
                            )
                            writer.detach()
                            captures += 1
                            print(
                                json.dumps(
                                    {
                                        "status": "visible_capture_presented",
                                        "dataset_id": production_dataset_id,
                                        "state_id": state_id,
                                        "camera_id": camera_id,
                                        "zoom_index": int(zoom_offset + 1),
                                        "capture_id": zoom_view_plan[camera_id]["capture_id"],
                                        "captures": captures,
                                        "expected_captures": selected_counts["capture_cases"],
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                    capture_sequence_cursor += len(camera_batch)
            completed_global_state_index = int(scheduled_state["global_state_index"])
            write_progress(
                capture_progress_path,
                {
                    "status": "capture_running",
                    "state_id": state_id,
                    "dataset_id": production_dataset_id,
                    "run_kind": run_kind,
                    "production_chunk_id": production_chunk_id,
                    "completed_selected_states": selected_sequence,
                    "expected_selected_states": len(selected_states),
                    "last_global_state_index": completed_global_state_index,
                    "captures": captures,
                    "expected_captures": selected_counts["capture_cases"],
                    "day_index": int(scheduled_state["day_index"]),
                    "state_in_day": int(scheduled_state["state_in_day"]),
                    "playback_seconds_per_day": float(args.seconds_per_day),
                    "playback_elapsed_s": round(
                        (completed_global_state_index - 1) * args.seconds_per_day / states_per_day,
                        3,
                    ),
                    "wall_elapsed_s": round(time.monotonic() - capture_started, 3),
                    "output_root": str(output_root),
                },
            )
        rep.orchestrator.wait_until_complete()
        capture_timeline.stop()
        for render_product in render_product_pool:
            render_product.destroy()
        if captures != selected_counts["capture_cases"]:
            raise RuntimeError(
                f"capture count mismatch: {captures} != {selected_counts['capture_cases']}"
            )
        completed = {"status": "render_complete", "dataset_id": production_dataset_id, "source_package_id": source_package_id, "run_kind": run_kind, "production_chunk_id": production_chunk_id, "incident_days": incident_days, "full_state_count": len(expected_states), "selected_state_count": len(selected_states), "selected_state_indices": [int(state["global_state_index"]) for state in selected_states], "camera_pool": len(camera_prims), "scheduled_camera_count": len(scheduled_camera_ids), "render_product_pool_size": len(render_product_pool), "render_product_batch_size": int(args.render_product_batch_size), "kit_session_reused": True, "views_per_state": int(schedule["views_per_state"]), "zooms_per_view": int(schedule["zooms_per_view"]), "captures_per_state": int(schedule["captures_per_state"]), "captures": captures, "expected_positive_cases": selected_counts["positive_cases"], "expected_negative_cases": selected_counts["negative_cases"], "playback_seconds_per_day": float(args.seconds_per_day), "wall_elapsed_s": round(time.monotonic() - capture_started, 3), "output_root": str(output_root)}
        write_progress(capture_progress_path, completed)
        print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        failure = {
            "status": "runtime_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_progress(capture_progress_path, failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
        raise
    finally:
        app.close()


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args)
    except BaseException as exc:
        print(f"FireViewer dataset run failed [{type(exc).__name__}]: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
