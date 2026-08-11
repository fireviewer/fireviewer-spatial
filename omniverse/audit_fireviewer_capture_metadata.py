"""Fail-closed audit of FireViewer per-capture files and agentic training metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fireviewer_capture_storage import (
    load_array,
    load_named_arrays_npz,
    storage_profile_contract,
    validate_pointcloud_storage,
)


REQUIRED_FILES = (
    "rgb.png",
    "semantic_ids.npz",
    "semantic_info.json",
    "instance_ids.npz",
    "instance_info.json",
    "front_visible_mask.npz",
    "flame_mask.npz",
    "perimeter_mask.npz",
    "smoke_source_mask.npz",
    "smoke_mask.npz",
    "burned_area_mask.npz",
    "dense_target_projection.json",
    "depth_distance_to_camera_m.npz",
    "depth_preview.png",
    "depth_metadata.json",
    "normals_replicator.npz",
    "normals_preview.png",
    "normals_metadata.json",
    "pointcloud.npz",
    "pointcloud_attributes.npz",
    "pointcloud_info.json",
    "camera_params.json",
    "geolocation.json",
    "capture-plan.json",
    "training-targets.json",
    "abstention.json",
)
THERMAL_FILES = ("thermal_kelvin.npz", "thermal_16bit.png", "thermal_metadata.json")
HASHED_BASE_FILES = (
    "rgb.png",
    "semantic_ids.npz",
    "semantic_info.json",
    "instance_ids.npz",
    "instance_info.json",
    "front_visible_mask.npz",
    "flame_mask.npz",
    "perimeter_mask.npz",
    "smoke_source_mask.npz",
    "smoke_mask.npz",
    "burned_area_mask.npz",
    "dense_target_projection.json",
    "depth_distance_to_camera_m.npz",
    "depth_preview.png",
    "depth_metadata.json",
    "normals_replicator.npz",
    "normals_preview.png",
    "normals_metadata.json",
    "pointcloud.npz",
    "pointcloud_attributes.npz",
    "pointcloud_info.json",
    "camera_params.json",
    "geolocation.json",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def nested(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for name in path.split("."):
        if not isinstance(current, dict) or name not in current:
            raise KeyError(path)
        current = current[name]
    return current


def finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)


def png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"invalid PNG header: {path}")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_capture(directory: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for name in REQUIRED_FILES:
        if not (directory / name).is_file():
            errors.append(f"missing_file:{name}")
    if errors:
        return {}, errors, warnings

    target = read_json(directory / "training-targets.json")
    plan = read_json(directory / "capture-plan.json")
    abstention = read_json(directory / "abstention.json")
    depth_metadata = read_json(directory / "depth_metadata.json")
    normals_metadata = read_json(directory / "normals_metadata.json")
    pointcloud_metadata = read_json(directory / "pointcloud_info.json")
    projection_metadata = read_json(directory / "dense_target_projection.json")
    required_paths = (
        "dataset_id",
        "capture_storage_profile.profile_id",
        "capture_storage_profile.profile_sha256",
        "package_id",
        "source_package_id",
        "source_stage_sha256",
        "observation_id",
        "case_id",
        "map_package_id",
        "incident_id",
        "scenario_id",
        "simulation_id",
        "state_id",
        "capture_id",
        "camera_id",
        "sample_kind",
        "captured_at_utc",
        "valid_at",
        "timezone",
        "source_provenance_ids",
        "simulation_time.day_index",
        "simulation_time.state_in_day",
        "simulation_time.observation_elapsed_s",
        "simulation_time.fire_state_elapsed_s",
        "simulation_time.playback_elapsed_s",
        "simulation_time.playback_seconds_per_day",
        "simulation_time.playback_seconds_per_state",
        "simulation_time.time_of_day_s",
        "simulation_time.time_of_day_hhmmss",
        "simulation_time.clock_contract",
        "simulation_time.capture_timeline_time_s",
        "simulation_time.capture_sequence_in_state",
        "simulation_time.timeline_playing_during_transition",
        "simulation_time.capture_trigger_contract",
        "fire_state.burned_area_m2",
        "fire_state.burned_tree_count",
        "fire_state.active_front_length_m",
        "fire_state.mean_front_spread_rate_m_s",
        "fire_state.ignition_l93_m",
        "fire_state.ignition_point_local_m",
        "fire_state.truth_scope",
        "fire_state.propagation_model",
        "fire_state.propagation_solver",
        "fire_state.fuel_input",
        "environment.drivers",
        "environment.domain_bounds_l93_m",
        "environment.weather_state_id",
        "environment.flow_capture_state.flow_capture_preparation_contract",
        "environment.flow_capture_state.flow_capture_time_s",
        "environment.flow_capture_state.flow_warmup_updates",
        "environment.flow_capture_state.flow_timeline_frozen_during_warmup",
        "environment.flow_capture_state.flow_truth_alignment.alignment_contract",
        "environment.flow_capture_state.flow_truth_alignment.session_layer_only",
        "environment.flow_capture_state.flow_truth_alignment.source_stage_modified",
        "environment.flow_capture_state.flow_truth_alignment.truth_point_count",
        "environment.flow_capture_state.flow_truth_alignment.hotspot_override_count",
        "environment.flow_capture_state.flow_truth_alignment.smoke_quad_override_count",
        "environment.flow_capture_state.flow_truth_alignment.effective_hotspot_lift_m",
        "environment.flow_capture_state.flow_truth_alignment.effective_smoke_base_lift_above_hotspot_m",
        "environment.flow_capture_state.flow_capture_profile.profile_id",
        "environment.flow_capture_state.flow_capture_profile.profile_sha256",
        "environment.flow_capture_state.flow_capture_profile.session_layer_only",
        "environment.flow_capture_state.flow_capture_profile.source_stage_modified",
        "environment.flow_capture_state.flow_capture_profile.applied_parameter_count",
        "environment.flow_capture_state.flow_capture_profile.smoke_source_value",
        "environment.flow_capture_state.flow_capture_profile.smoke_source_extent_m",
        "environment.flow_capture_state.flow_capture_profile.smoke_velocity_local_m_s",
        "environment.flow_capture_state.flow_capture_profile.plume_vertical_velocity_m_s",
        "environment.flow_capture_state.flow_capture_profile.smoke_buoyancy_per_unit",
        "environment.flow_capture_state.flow_capture_profile.smoke_fade_per_s",
        "environment.flow_capture_state.flow_capture_profile.state_visibility_regime.regime",
        "environment.flow_capture_state.flow_capture_profile.state_visibility_regime.reason",
        "environment.flow_capture_state.flow_capture_profile.state_visibility_regime.low_signal_state_allowed",
        "environment.sky_capture_state.profile_id",
        "environment.sky_capture_state.profile_sha256",
        "environment.sky_capture_state.session_layer_only",
        "environment.sky_capture_state.source_stage_modified",
        "environment.sky_capture_state.source_texture_asset",
        "environment.sky_capture_state.effective_texture_asset",
        "environment.sky_capture_state.cloud_texture_suppressed",
        "environment.sky_capture_state.effective_color_rgb",
        "environment.sky_capture_state.effective_intensity",
        "environment.sky_capture_state.effective_exposure",
        "camera.position_local_m",
        "camera.position_l93_ngf_ign69_m",
        "camera.aim_local_m",
        "camera.orientation_quat_wxyz",
        "camera.orientation_yaw_pitch_roll_degrees",
        "camera.forward_local",
        "camera.image_resolution_px",
        "camera.focal_length_mm",
        "camera.focal_length_35mm_equivalent_mm",
        "camera.horizontal_aperture_mm",
        "camera.vertical_aperture_mm",
        "camera.horizontal_fov_degrees",
        "camera.vertical_fov_degrees",
        "camera.capture_device_profile",
        "camera.framing_style",
        "camera.camera_role",
        "camera.intrinsics",
        "zoom.zoom_set_id",
        "zoom.zoom_set_size",
        "zoom.zoom_index",
        "zoom.zoom_label",
        "zoom.zoom_multiplier",
        "nearest_flame.point_index",
        "nearest_flame.point_local_m",
        "nearest_flame.point_l93_ngf_ign69_m",
        "nearest_flame.distance_m",
        "nearest_flame.projection",
        "nearest_flame.source_point_count",
        "nearest_smoke.point_index",
        "nearest_smoke.point_local_m",
        "nearest_smoke.point_l93_ngf_ign69_m",
        "nearest_smoke.distance_m",
        "nearest_smoke.projection",
        "nearest_smoke.source_point_count",
        "visible_flame_points_local_m",
        "smoke_source_points_local_m",
        "active_front_local_m",
        "dense_targets.semantic_ids",
        "dense_targets.instance_ids",
        "dense_targets.fire_front_mask",
        "dense_targets.flame_mask",
        "dense_targets.fire_perimeter_mask",
        "dense_targets.smoke_source_mask",
        "dense_targets.smoke_mask",
        "dense_targets.burned_area_mask",
        "dense_targets.projection_metadata",
        "dense_targets.depth_m",
        "dense_targets.depth_metadata",
        "dense_targets.normals",
        "dense_targets.normals_metadata",
        "dense_targets.pointcloud",
        "dense_targets.pointcloud_attributes",
        "visibility.expected_fire_visible",
        "visibility.fire_front_visible",
        "visibility.fire_perimeter_visible",
        "visibility.smoke_source_visible",
        "visibility.smoke_visible",
        "visibility.acceptance.acceptance_class",
        "visibility.acceptance.low_signal_allowed",
        "visibility.acceptance.low_signal_reasons",
        "visibility.acceptance.fire_visibility_regime",
        "visibility.acceptance.projected_fire_or_smoke_pixels",
        "visibility.acceptance.projected_frame_fraction",
        "visibility.acceptance.projected_visibility_tier",
        "visibility.acceptance.visibility_validation_status",
        "visibility.acceptance.training_role",
        "dense_target_source",
        "expected_fire_in_frame",
        "line_of_sight_receipt",
        "expected_modalities",
        "weather_state_id",
        "modality_sha256",
        "dependency_sha256",
        "split_group.base_map_package_id",
        "split_group.incident_id",
        "split_group.scenario_id",
        "targeting_contract",
        "targeting_mode",
    )
    for path in required_paths:
        try:
            value = nested(target, path)
            if value is None:
                errors.append(f"null_field:{path}")
        except KeyError:
            errors.append(f"missing_field:{path}")

    if errors:
        return target, errors, warnings
    if target["capture_storage_profile"] != storage_profile_contract():
        errors.append("capture_storage_profile_mismatch")
    try:
        datetime.fromisoformat(str(target["captured_at_utc"]).replace("Z", "+00:00"))
    except ValueError:
        errors.append("invalid_captured_at_utc")
    try:
        datetime.fromisoformat(str(target["valid_at"]).replace("Z", "+00:00"))
    except ValueError:
        errors.append("invalid_valid_at")
    if target["timezone"] != "UTC":
        errors.append("unsupported_timezone")
    if not isinstance(target["source_provenance_ids"], list):
        errors.append("invalid_source_provenance_ids")
    flow_capture_state = target["environment"]["flow_capture_state"]
    if flow_capture_state["flow_capture_preparation_contract"] != "independent_volume_clear_and_fixed_state_time_warmup_v1":
        errors.append("invalid_flow_capture_preparation_contract")
    if not bool(flow_capture_state["flow_timeline_frozen_during_warmup"]):
        errors.append("flow_timeline_not_frozen_during_warmup")
    if int(flow_capture_state["flow_warmup_updates"]) < 1:
        errors.append("invalid_flow_warmup_updates")
    if abs(float(flow_capture_state["flow_capture_time_s"]) - float(target["simulation_time"]["playback_elapsed_s"])) > 1e-6:
        errors.append("flow_capture_time_mismatch")
    plume_profile = flow_capture_state["flow_capture_profile"]
    if plume_profile["profile_id"] != "wildfire_convective_plume_mid_distance_v2":
        errors.append("invalid_flow_capture_profile")
    if not re.fullmatch(r"[0-9a-f]{64}", str(plume_profile["profile_sha256"])):
        errors.append("invalid_flow_capture_profile_sha256")
    if not bool(plume_profile["session_layer_only"]):
        errors.append("flow_capture_profile_not_session_only")
    if bool(plume_profile["source_stage_modified"]):
        errors.append("flow_capture_profile_modified_source_stage")
    if int(plume_profile["applied_parameter_count"]) < 20:
        errors.append("incomplete_flow_capture_profile")
    state_regime = plume_profile["state_visibility_regime"]
    regime_name = str(state_regime["regime"])
    regime_minimums = {
        "incipient": {"smoke": 0.3, "extent": 0.9, "rise": 12.0, "buoyancy": 4.0},
        "stalled": {"smoke": 0.4, "extent": 1.3, "rise": 14.0, "buoyancy": 4.5},
        "slowed": {"smoke": 0.6, "extent": 1.7, "rise": 17.0, "buoyancy": 5.5},
        "established": {"smoke": 0.8, "extent": 2.0, "rise": 18.0, "buoyancy": 6.0},
    }
    minimums = regime_minimums.get(regime_name)
    if minimums is None:
        errors.append("invalid_fire_visibility_regime")
        minimums = regime_minimums["established"]
    if bool(state_regime["low_signal_state_allowed"]) != (regime_name != "established"):
        errors.append("invalid_low_signal_state_allowance")
    if float(plume_profile["smoke_source_value"]) < minimums["smoke"]:
        errors.append("insufficient_flow_smoke_source_for_regime")
    if float(plume_profile["smoke_source_extent_m"]) < minimums["extent"]:
        errors.append("insufficient_flow_smoke_source_extent_for_regime")
    if not finite_vector(plume_profile["smoke_velocity_local_m_s"], 3):
        errors.append("invalid_flow_smoke_velocity")
    if float(plume_profile["plume_vertical_velocity_m_s"]) < minimums["rise"]:
        errors.append("insufficient_flow_plume_rise_for_regime")
    if float(plume_profile["smoke_buoyancy_per_unit"]) < minimums["buoyancy"]:
        errors.append("insufficient_flow_smoke_buoyancy_for_regime")
    if float(plume_profile["smoke_fade_per_s"]) > 0.01:
        errors.append("excessive_flow_smoke_fade")
    alignment = flow_capture_state["flow_truth_alignment"]
    if not bool(alignment["session_layer_only"]):
        errors.append("flow_truth_alignment_not_session_only")
    if bool(alignment["source_stage_modified"]):
        errors.append("flow_truth_alignment_modified_source_stage")
    for count_name in (
        "truth_point_count",
        "hotspot_override_count",
        "smoke_quad_override_count",
    ):
        if int(alignment[count_name]) != 48:
            errors.append(f"invalid_flow_truth_alignment_count:{count_name}")
    if abs(float(alignment["effective_hotspot_lift_m"]) - 0.25) > 1e-6:
        errors.append("invalid_flow_truth_hotspot_lift")
    if (
        abs(float(alignment["effective_smoke_base_lift_above_hotspot_m"]) - 1.6)
        > 1e-6
    ):
        errors.append("invalid_flow_truth_smoke_base_lift")
    sky_capture_state = target["environment"]["sky_capture_state"]
    if sky_capture_state["profile_id"] != "clear_daylight_smoke_contrast_v2":
        errors.append("invalid_sky_capture_profile")
    if not re.fullmatch(r"[0-9a-f]{64}", str(sky_capture_state["profile_sha256"])):
        errors.append("invalid_sky_capture_profile_sha256")
    if not bool(sky_capture_state["session_layer_only"]):
        errors.append("sky_capture_profile_not_session_only")
    if bool(sky_capture_state["source_stage_modified"]):
        errors.append("sky_capture_profile_modified_source_stage")
    if not str(sky_capture_state["source_texture_asset"]).endswith(
        "farm_field_puresky_4k.hdr"
    ):
        errors.append("invalid_sky_source_texture")
    if str(sky_capture_state["effective_texture_asset"]):
        errors.append("sky_cloud_texture_still_active")
    if not bool(sky_capture_state["cloud_texture_suppressed"]):
        errors.append("sky_cloud_texture_not_suppressed")
    if not finite_vector(sky_capture_state["effective_color_rgb"], 3):
        errors.append("invalid_sky_capture_color")
    if float(sky_capture_state["effective_intensity"]) <= 0.0:
        errors.append("invalid_sky_capture_intensity")
    for name in ("ignition_l93_m", "ignition_point_local_m"):
        if not finite_vector(target["fire_state"][name], 3):
            errors.append(f"invalid_fire_state_vector:{name}")

    camera = target["camera"]
    for name in ("position_local_m", "position_l93_ngf_ign69_m", "aim_local_m", "orientation_yaw_pitch_roll_degrees", "forward_local"):
        if not finite_vector(camera[name], 3):
            errors.append(f"invalid_camera_vector:{name}")
    quaternion = camera["orientation_quat_wxyz"]
    if not finite_vector(quaternion, 4) or abs(math.sqrt(sum(float(item) ** 2 for item in quaternion)) - 1.0) > 1e-3:
        errors.append("invalid_camera_quaternion")
    forward = camera["forward_local"]
    if finite_vector(forward, 3) and abs(math.sqrt(sum(float(item) ** 2 for item in forward)) - 1.0) > 1e-4:
        errors.append("invalid_camera_forward")
    resolution = camera["image_resolution_px"]
    if not (isinstance(resolution, list) and len(resolution) == 2 and all(isinstance(item, int) and item > 0 for item in resolution)):
        errors.append("invalid_image_resolution")
        resolution = [0, 0]

    sample_kind = str(target["sample_kind"])
    flame_projection = target["nearest_flame"]["projection"]
    visibility_acceptance = target["visibility"]["acceptance"]
    if visibility_acceptance["fire_visibility_regime"] != regime_name:
        errors.append("capture_visibility_regime_mismatch")
    if visibility_acceptance["visibility_validation_status"] != "accepted":
        errors.append("capture_visibility_not_accepted")
    if not isinstance(visibility_acceptance["low_signal_reasons"], list):
        errors.append("invalid_low_signal_reasons")
    if sample_kind == "positive_fire":
        if not bool(target["expected_fire_in_frame"]):
            errors.append("positive_expected_fire_flag_false")
        low_signal_allowed = bool(visibility_acceptance["low_signal_allowed"])
        expected_acceptance_class = (
            "valid_low_or_partial_signal_positive"
            if low_signal_allowed
            else "standard_positive"
        )
        if visibility_acceptance["acceptance_class"] != expected_acceptance_class:
            errors.append("invalid_positive_visibility_acceptance_class")
        if visibility_acceptance["training_role"] != (
            "hard_positive" if low_signal_allowed else "positive"
        ):
            errors.append("invalid_positive_visibility_training_role")
        normalized = flame_projection.get("normalized_xy") if isinstance(flame_projection, dict) else None
        if not flame_projection.get("in_frame") or not finite_vector(normalized, 2) or max(abs(float(item)) for item in normalized) > 1e-4:
            errors.append("positive_nearest_flame_not_centered")
    elif sample_kind == "negative_context":
        if bool(target["expected_fire_in_frame"]):
            errors.append("negative_expected_fire_flag_true")
        if (
            visibility_acceptance["acceptance_class"] != "negative_context"
            or visibility_acceptance["training_role"] != "negative_context"
            or bool(visibility_acceptance["low_signal_allowed"])
        ):
            errors.append("invalid_negative_visibility_acceptance")
        if flame_projection.get("in_frame"):
            errors.append("negative_nearest_flame_in_frame")
    else:
        errors.append("invalid_sample_kind")

    try:
        semantic = load_array(directory / "semantic_ids.npz")
        instance = load_array(directory / "instance_ids.npz")
        front = load_array(directory / "front_visible_mask.npz")
        flame = load_array(directory / "flame_mask.npz")
        perimeter = load_array(directory / "perimeter_mask.npz")
        smoke = load_array(directory / "smoke_source_mask.npz")
        smoke_alias = load_array(directory / "smoke_mask.npz")
        burned = load_array(directory / "burned_area_mask.npz")
        depth = load_array(directory / "depth_distance_to_camera_m.npz")
        normals = load_array(directory / "normals_replicator.npz")
        pointcloud = load_array(directory / "pointcloud.npz")
        pointcloud_attributes = load_named_arrays_npz(
            directory / "pointcloud_attributes.npz"
        )
        expected_shape = (int(resolution[1]), int(resolution[0]))
        for name, array in (("semantic", semantic), ("instance", instance), ("front", front), ("flame", flame), ("perimeter", perimeter), ("smoke", smoke), ("smoke_alias", smoke_alias), ("burned", burned), ("depth", depth), ("normals", normals)):
            if tuple(array.shape[:2]) != expected_shape:
                errors.append(f"shape_mismatch:{name}:{tuple(array.shape)}:{expected_shape}")
        if not np.array_equal(front, flame):
            errors.append("flame_mask_alias_mismatch")
        if np.any((smoke != 0) & (smoke_alias == 0)):
            errors.append("smoke_mask_does_not_include_all_smoke_sources")
        errors.extend(
            validate_pointcloud_storage(
                points=pointcloud,
                attributes=pointcloud_attributes,
                metadata=pointcloud_metadata,
            )
        )
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid_depth):
            errors.append("depth_has_no_finite_positive_pixel")
        if depth_metadata.get("schema") != "fireviewer.depth-pass.v1":
            errors.append("invalid_depth_metadata_schema")
        if depth_metadata.get("annotator") != "distance_to_camera":
            errors.append("invalid_depth_annotator")
        if depth_metadata.get("raw_file") != "depth_distance_to_camera_m.npz":
            errors.append("invalid_depth_raw_file")
        if depth_metadata.get("shape") != list(depth.shape):
            errors.append("depth_metadata_shape_mismatch")
        if depth_metadata.get("valid_pixel_count") != int(np.count_nonzero(valid_depth)):
            errors.append("depth_metadata_valid_pixel_count_mismatch")
        if normals.ndim < 3 or normals.shape[-1] < 3:
            errors.append(f"invalid_normals_shape:{tuple(normals.shape)}")
        else:
            normal_xyz = np.asarray(normals[..., :3])
            if not np.all(np.isfinite(normal_xyz)):
                errors.append("normals_have_non_finite_values")
            if np.any(np.abs(normal_xyz) > 1.001):
                errors.append("normals_outside_minus_one_to_one")
            normal_lengths = np.linalg.norm(normal_xyz, axis=-1)
            surface_lengths = normal_lengths[normal_lengths > 1e-4]
            if surface_lengths.size == 0:
                errors.append("normals_have_no_surface_pixel")
            elif np.any(np.abs(surface_lengths - 1.0) > 0.15):
                errors.append("normals_are_not_unit_length")
        if normals_metadata.get("schema") != "fireviewer.normal-pass.v1":
            errors.append("invalid_normals_metadata_schema")
        if normals_metadata.get("annotator") != "normals":
            errors.append("invalid_normals_annotator")
        if normals_metadata.get("raw_file") != "normals_replicator.npz":
            errors.append("invalid_normals_raw_file")
        if normals_metadata.get("shape") != list(normals.shape):
            errors.append("normals_metadata_shape_mismatch")
        expected_dense_targets = {
            "semantic_ids": "semantic_ids.npz",
            "instance_ids": "instance_ids.npz",
            "fire_front_mask": "front_visible_mask.npz",
            "flame_mask": "flame_mask.npz",
            "fire_perimeter_mask": "perimeter_mask.npz",
            "smoke_source_mask": "smoke_source_mask.npz",
            "smoke_mask": "smoke_mask.npz",
            "burned_area_mask": "burned_area_mask.npz",
            "projection_metadata": "dense_target_projection.json",
            "depth_m": "depth_distance_to_camera_m.npz",
            "depth_metadata": "depth_metadata.json",
            "normals": "normals_replicator.npz",
            "normals_metadata": "normals_metadata.json",
            "pointcloud": "pointcloud.npz",
            "pointcloud_attributes": "pointcloud_attributes.npz",
        }
        if target["dense_targets"] != expected_dense_targets:
            errors.append("dense_target_file_contract_mismatch")
        actual_visibility = {
            "fire_front_visible": bool(np.any(front)),
            "fire_perimeter_visible": bool(np.any(perimeter)),
            "smoke_source_visible": bool(np.any(smoke)),
            "smoke_visible": bool(np.any(smoke_alias)),
        }
        for name, actual in actual_visibility.items():
            if bool(target["visibility"][name]) != actual:
                errors.append(f"visibility_mask_mismatch:{name}")
        if target["dense_target_source"] != "active_usd_truth_geometry_camera_projection":
            errors.append(f"unsupported_dense_target_source:{target['dense_target_source']}")
        if projection_metadata.get("schema") != "fireviewer.projected-dense-targets.v1":
            errors.append("invalid_dense_target_projection_schema")
        if projection_metadata.get("resolution_px") != resolution:
            errors.append("dense_target_projection_resolution_mismatch")
        if bool(projection_metadata.get("expected_fire_visible")) != bool(target["expected_fire_in_frame"]):
            errors.append("dense_target_projection_expected_fire_mismatch")
        expected_pixel_counts = {
            "fire_front": int(np.count_nonzero(front)),
            "fire_perimeter": int(np.count_nonzero(perimeter)),
            "smoke_source": int(np.count_nonzero(smoke)),
            "smoke": int(np.count_nonzero(smoke_alias)),
            "burned_area": int(np.count_nonzero(burned)),
        }
        if projection_metadata.get("pixel_counts") != expected_pixel_counts:
            errors.append("dense_target_projection_pixel_count_mismatch")
        projected_union_pixels = int(
            np.count_nonzero((front != 0) | (smoke_alias != 0))
        )
        if sample_kind == "negative_context":
            expected_visibility_tier = "none_expected"
        elif projected_union_pixels <= 32:
            expected_visibility_tier = "trace"
        elif projected_union_pixels <= 256:
            expected_visibility_tier = "small"
        elif projected_union_pixels <= 2_048:
            expected_visibility_tier = "moderate"
        else:
            expected_visibility_tier = "clear"
        if int(visibility_acceptance["projected_fire_or_smoke_pixels"]) != projected_union_pixels:
            errors.append("projected_visibility_pixel_count_mismatch")
        expected_fraction = projected_union_pixels / float(
            int(resolution[0]) * int(resolution[1])
        )
        if abs(float(visibility_acceptance["projected_frame_fraction"]) - expected_fraction) > 1e-12:
            errors.append("projected_visibility_fraction_mismatch")
        if visibility_acceptance["projected_visibility_tier"] != expected_visibility_tier:
            errors.append("projected_visibility_tier_mismatch")
        if sample_kind == "positive_fire":
            if not expected_pixel_counts["fire_front"]:
                errors.append("positive_fire_front_mask_empty")
            if not expected_pixel_counts["fire_perimeter"]:
                errors.append("positive_fire_perimeter_mask_empty")
            if not expected_pixel_counts["smoke_source"]:
                errors.append("positive_smoke_source_mask_empty")
            if not expected_pixel_counts["smoke"]:
                errors.append("positive_smoke_mask_empty")
        elif any(expected_pixel_counts.values()):
            errors.append("negative_context_has_nonempty_fire_truth_mask")
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"invalid_numpy_artifact:{type(exc).__name__}:{exc}")

    try:
        if tuple(png_dimensions(directory / "rgb.png")) != tuple(resolution):
            errors.append("rgb_resolution_mismatch")
        depth_preview_resolution = depth_metadata.get("preview_resolution_px")
        normals_preview_resolution = normals_metadata.get("preview_resolution_px")
        if not finite_vector(depth_preview_resolution, 2):
            errors.append("invalid_depth_preview_resolution_metadata")
            depth_preview_resolution = [-1, -1]
        if not finite_vector(normals_preview_resolution, 2):
            errors.append("invalid_normals_preview_resolution_metadata")
            normals_preview_resolution = [-1, -1]
        if tuple(png_dimensions(directory / "depth_preview.png")) != tuple(
            int(value) for value in depth_preview_resolution
        ):
            errors.append("depth_preview_resolution_mismatch")
        if tuple(png_dimensions(directory / "normals_preview.png")) != tuple(
            int(value) for value in normals_preview_resolution
        ):
            errors.append("normals_preview_resolution_mismatch")
        if depth_metadata.get("preview_downsample_factor") != 4:
            errors.append("invalid_depth_preview_downsample_factor")
        if normals_metadata.get("preview_downsample_factor") != 4:
            errors.append("invalid_normals_preview_downsample_factor")
    except ValueError as exc:
        errors.append(str(exc))

    thermal_expected = bool(plan.get("thermal_expected"))
    for name in THERMAL_FILES:
        exists = (directory / name).is_file()
        if thermal_expected and not exists:
            errors.append(f"missing_thermal_file:{name}")
        if not thermal_expected and exists:
            errors.append(f"unexpected_thermal_file:{name}")
    if thermal_expected and (directory / "thermal_16bit.png").is_file():
        if tuple(png_dimensions(directory / "thermal_16bit.png")) != tuple(resolution):
            errors.append("thermal_resolution_mismatch")
        try:
            thermal_kelvin = load_array(directory / "thermal_kelvin.npz")
            thermal_metadata = read_json(directory / "thermal_metadata.json")
            expected_shape = (int(resolution[1]), int(resolution[0]))
            if tuple(thermal_kelvin.shape) != expected_shape:
                errors.append("thermal_kelvin_shape_mismatch")
            if thermal_kelvin.dtype != np.dtype(np.float32):
                errors.append("thermal_kelvin_dtype_mismatch")
            if not np.all(np.isfinite(thermal_kelvin)):
                errors.append("thermal_kelvin_has_non_finite_values")
            if thermal_metadata.get("schema") != "fireviewer.synthetic-thermal.v1":
                errors.append("invalid_thermal_metadata_schema")
            if thermal_metadata.get("raw_file") != "thermal_kelvin.npz":
                errors.append("invalid_thermal_raw_file")
        except (OSError, ValueError, IndexError) as exc:
            errors.append(f"invalid_thermal_artifact:{type(exc).__name__}:{exc}")

    modality_sha256 = target["modality_sha256"]
    expected_hashed_files = set(HASHED_BASE_FILES)
    if thermal_expected:
        expected_hashed_files.update(THERMAL_FILES)
    if not isinstance(modality_sha256, dict):
        errors.append("invalid_modality_sha256")
    else:
        for name in sorted(expected_hashed_files):
            expected_digest = modality_sha256.get(name)
            artifact = directory / name
            if not isinstance(expected_digest, str) or not SHA256_RE.fullmatch(expected_digest):
                errors.append(f"missing_or_invalid_modality_sha256:{name}")
            elif artifact.is_file() and sha256_file(artifact) != expected_digest:
                errors.append(f"modality_sha256_mismatch:{name}")
        unexpected_hashes = sorted(set(modality_sha256) - expected_hashed_files)
        if unexpected_hashes:
            errors.append(f"unexpected_modality_sha256:{','.join(unexpected_hashes)}")

    dependency_sha256 = target["dependency_sha256"]
    if not isinstance(dependency_sha256, dict) or not dependency_sha256:
        errors.append("invalid_dependency_sha256")
    else:
        for name, digest in sorted(dependency_sha256.items()):
            if not isinstance(name, str) or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                errors.append(f"invalid_dependency_sha256_entry:{name}")

    if target["capture_id"] != plan.get("capture_id") or target["capture_id"] != abstention.get("capture_id"):
        errors.append("capture_identity_mismatch")
    if bool(abstention.get("abstain")):
        warnings.append("capture_abstained")
    return target, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--expected-captures", type=int)
    parser.add_argument("--run-contract", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.capture_root.resolve()
    contract_path = args.run_contract.resolve() if args.run_contract is not None else None
    if contract_path is None:
        for candidate in (root / "run-contract.json", root.parent / "run-contract.json"):
            if candidate.is_file():
                contract_path = candidate
                break
    run_contract: dict[str, Any] = {}
    contract_failures: list[str] = []
    if contract_path is not None and contract_path.is_file():
        run_contract = read_json(contract_path)
        if run_contract.get("schema") != "fireviewer.kit-dataset-production-run.v1":
            contract_failures.append("invalid_run_contract_schema")
    elif args.expected_captures is None:
        contract_failures.append("missing_run_contract_and_expected_capture_count")

    contract_expected_captures = run_contract.get("expected_capture_cases")
    if args.expected_captures is not None:
        expected_captures = int(args.expected_captures)
        if contract_expected_captures is not None and expected_captures != int(contract_expected_captures):
            contract_failures.append("expected_capture_override_mismatches_run_contract")
    elif contract_expected_captures is not None:
        expected_captures = int(contract_expected_captures)
    else:
        expected_captures = -1
    expected_positive = run_contract.get("expected_positive_cases")
    expected_negative = run_contract.get("expected_negative_cases")
    expected_viewpoint_plans = run_contract.get("expected_viewpoint_plans")
    expected_dataset_id = run_contract.get("dataset_id")
    expected_source_package_id = run_contract.get("source_package_id")
    expected_state_ids = {str(value) for value in run_contract.get("selected_state_ids", [])}
    directories = sorted(path.parent for path in root.rglob("training-targets.json"))
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    warnings: Counter[str] = Counter()
    capture_ids: set[str] = set()
    state_camera_zoom_keys: set[tuple[str, str, int]] = set()
    zooms_by_viewpoint: dict[tuple[str, str], set[int]] = {}
    datasets: set[str] = set()
    source_packages: set[str] = set()
    observed_state_ids: set[str] = set()
    for directory in directories:
        try:
            target, capture_errors, capture_warnings = audit_capture(directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            target, capture_errors, capture_warnings = {}, [f"audit_exception:{type(exc).__name__}:{exc}"], []
        counts[str(target.get("sample_kind", "unknown"))] += 1
        capture_id = str(target.get("capture_id", ""))
        state_id = str(target.get("state_id", ""))
        camera_id = str(target.get("camera_id", ""))
        raw_zoom_index = (target.get("zoom") or {}).get("zoom_index") if isinstance(target.get("zoom"), dict) else None
        zoom_index = int(raw_zoom_index) if isinstance(raw_zoom_index, int) else -1
        key = (state_id, camera_id, zoom_index)
        if not capture_id or capture_id in capture_ids:
            capture_errors.append("missing_or_duplicate_capture_id")
        capture_ids.add(capture_id)
        if key in state_camera_zoom_keys:
            capture_errors.append("duplicate_state_camera_zoom")
        state_camera_zoom_keys.add(key)
        zooms_by_viewpoint.setdefault((state_id, camera_id), set()).add(zoom_index)
        datasets.add(str(target.get("dataset_id", "")))
        source_packages.add(str(target.get("source_package_id", "")))
        observed_state_ids.add(state_id)
        warnings.update(capture_warnings)
        if capture_errors:
            failures.append({"directory": str(directory), "errors": capture_errors})

    aggregate_errors = list(contract_failures)
    if expected_captures >= 0 and len(directories) != expected_captures:
        aggregate_errors.append(f"capture_count:{len(directories)}:{expected_captures}")
    if expected_positive is not None and counts["positive_fire"] != int(expected_positive):
        aggregate_errors.append(f"positive_class_quota:{counts['positive_fire']}:{int(expected_positive)}")
    if expected_negative is not None and counts["negative_context"] != int(expected_negative):
        aggregate_errors.append(f"negative_class_quota:{counts['negative_context']}:{int(expected_negative)}")
    if expected_viewpoint_plans is not None and len(zooms_by_viewpoint) != int(expected_viewpoint_plans):
        aggregate_errors.append(f"viewpoint_count:{len(zooms_by_viewpoint)}:{int(expected_viewpoint_plans)}")
    for (state_id, camera_id), zoom_indices in sorted(zooms_by_viewpoint.items()):
        if zoom_indices != {1, 2, 3, 4, 5}:
            aggregate_errors.append(f"incomplete_zoom_set:{state_id}:{camera_id}:{sorted(zoom_indices)}")
    if expected_dataset_id is not None and datasets != {str(expected_dataset_id)}:
        aggregate_errors.append(f"dataset_identity:{sorted(datasets)}:{expected_dataset_id}")
    if expected_source_package_id is not None and source_packages != {str(expected_source_package_id)}:
        aggregate_errors.append(f"source_package_identity:{sorted(source_packages)}:{expected_source_package_id}")
    if expected_state_ids and observed_state_ids != expected_state_ids:
        aggregate_errors.append(f"state_identity:{sorted(observed_state_ids)}:{sorted(expected_state_ids)}")
    if aggregate_errors:
        failures.append({"directory": str(root), "errors": aggregate_errors})

    report = {
        "schema": "fireviewer.capture-metadata-audit.v2",
        "status": "passed" if not failures else "failed",
        "capture_root": str(root),
        "run_contract": str(contract_path) if contract_path is not None else None,
        "dataset_id": expected_dataset_id,
        "source_package_id": expected_source_package_id,
        "captures": len(directories),
        "expected_captures": expected_captures,
        "viewpoint_plans": len(zooms_by_viewpoint),
        "expected_viewpoint_plans": expected_viewpoint_plans,
        "sample_counts": dict(counts),
        "abstention_warning_count": warnings["capture_abstained"],
        "failed_capture_count": len(failures),
        "failures": failures[:200],
        "failures_truncated": len(failures) > 200,
    }
    if args.report is not None:
        output = args.report.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
