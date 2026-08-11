"""Fail-closed structural validation for FireViewer modular USD dataset packages.

The validator deliberately works without Isaac Sim.  It verifies the complete
composition contract, all local USD references, state/camera counts, semantic
contracts and standalone dependency paths.  RTX/Flow pixels remain a separate
runtime and human-acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


REFERENCE_RE = re.compile(r"@([^@]+)@")
ABSOLUTE_OR_REMOTE_RE = re.compile(r"(?:[A-Za-z]:[\\/]|https?://|omniverse://)", re.IGNORECASE)
CAMERA_RE = re.compile(r'^\s*def Camera "(CAM_\d{2})"', re.MULTILINE)
STATE_RE = re.compile(r'^\s*"(state_\d{3})"\s*(?:\([^)]*\)\s*)?\{', re.MULTILINE)
EXPECTED_MODALITIES = {
    "rgb",
    "aerial_synthetic_thermal_16bit",
    "semantic_masks",
    "pointcloud",
    "fire_front_visible",
    "fire_perimeter",
    "depth",
    "geolocation",
    "abstention",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_usd_references(package: Path) -> int:
    count = 0
    for path in package.rglob("*.usd*"):
        # USDZ and USDC are binary containers.  Their composition is opened by
        # the native Kit gate; the text-only dependency scanner must not try to
        # decode them as UTF-8.
        if path.suffix.lower() not in {".usd", ".usda"}:
            continue
        text = path.read_text(encoding="utf-8")
        if ABSOLUTE_OR_REMOTE_RE.search(text):
            raise ValueError(f"non-standalone USD dependency found in {path}")
        for reference in REFERENCE_RE.findall(text):
            count += 1
            target = Path(reference)
            require(not target.is_absolute(), f"absolute USD reference in {path}: {reference}")
            resolved = (path.parent / target).resolve()
            require(resolved.is_file(), f"missing USD dependency from {path}: {reference}")
    return count


def validate_package(package: Path) -> dict[str, Any]:
    manifest = read_json(package / "manifest.json")
    package_id = str(manifest["package_id"])
    require((package / manifest["entry_stage"]).is_file(), f"missing entry stage for {package_id}")
    require(manifest["site"]["terrain_payloads"] > 0, f"no terrain payloads for {package_id}")
    ground_texture = manifest["site"].get("ground_texture", {})
    require(ground_texture.get("role") == "real_georeferenced_ground_texture", f"ground is not a real georeferenced orthophoto for {package_id}")
    require(ground_texture.get("provider") == "IGN / Géoplateforme" and ground_texture.get("product") == "BD ORTHO", f"unexpected ground imagery provenance for {package_id}")
    require(ground_texture.get("bounds_l93_metres") == [876000.0, 6403000.0, 892000.0, 6413000.0], f"orthophoto does not cover the exact site bounds for {package_id}")
    require(abs(float(ground_texture.get("nominal_resolution_m", 0.0)) - 2.0) < 0.001, f"unexpected ground orthophoto resolution for {package_id}")
    ground_path = package / "source-usd" / str(ground_texture.get("texture", ""))
    ground_world_file = package / "source-usd" / str(ground_texture.get("world_file", ""))
    ground_source_record_path = package / "source-usd" / str(ground_texture.get("source_record", ""))
    require(ground_path.is_file() and ground_world_file.is_file() and ground_source_record_path.is_file(), f"standalone orthophoto dependencies are incomplete for {package_id}")
    require(hashlib.sha256(ground_path.read_bytes()).hexdigest() == ground_texture.get("sha256"), f"orthophoto checksum mismatch for {package_id}")
    with rasterio.open(ground_path) as ground_image:
        require((ground_image.width, ground_image.height) == (8000, 5000) and ground_image.count >= 3, f"invalid orthophoto raster dimensions for {package_id}")
    ground_source_record = read_json(ground_source_record_path)
    require(ground_source_record.get("schema") == "fireviewer.ign-orthophoto-source.v1" and ground_source_record.get("status") == "downloaded_and_validated", f"invalid orthophoto source evidence for {package_id}")
    require(ground_source_record["coverage"]["no_data_tile_count"] == 0, f"orthophoto has no-data tiles for {package_id}")
    require(manifest["site"]["building_payload_references"] >= 0, f"invalid building payload count for {package_id}")
    require(manifest["site"]["vegetation_point_instances"] > 0, f"no vegetation instances for {package_id}")
    require(manifest["site"]["vegetation_prototype_assets"] == 6, f"expected exactly six tree assets for {package_id}")
    require(manifest["site"]["vegetation_lod_policy"] == "none_all_detected_instances_resident", f"vegetation LOD is not disabled for {package_id}")
    require(manifest["site"]["building_representation"] == "merged_source_footprint_mnt_grounded_oriented_box_gabled_roof_windows_and_doors", f"buildings do not use the MNT-grounded merged architectural representation for {package_id}")
    require(manifest["site"].get("building_vertical_alignment") == "each_visual_module_sampled_at_its_center_on_mnt", f"buildings are not vertically aligned to the MNT for {package_id}")
    require(manifest["site"]["route_representation"] == "source_flat_surface_meshes_draped_on_mnt", f"routes are not flat source surfaces for {package_id}")
    require(manifest["site"]["occlusion_proxies"] >= 0, f"invalid occlusion proxy count for {package_id}")
    require(manifest["scenario"]["state_count"] == 180 and len(manifest["scenario"]["states"]) == 180, f"expected exactly 180 fire states for {package_id}")
    require(manifest["scenario"]["incident_days"] == 18 and manifest["scenario"]["states_per_day"] == 10, f"invalid 18-day incident cadence for {package_id}")
    require(manifest["scenario"]["truth_scope"] == "synthetic_physically_driven_fire_spread_on_real_uploaded_map_not_incident_reconstruction", f"invalid fire truth scope for {package_id}")
    camera_plan = manifest["cameras"]["plan"]
    require(manifest["cameras"]["fixed_count"] == 62 and len(camera_plan) == 62, f"expected a sixty-two-camera pool for {package_id}")
    human_cameras = [camera for camera in camera_plan if camera.get("role") != "aerial"]
    aerial_cameras = [camera for camera in camera_plan if camera.get("role") == "aerial"]
    negative_cameras = [camera for camera in camera_plan if camera.get("sample_capability") == "negative_context"]
    thermal_cameras = [camera for camera in camera_plan if camera.get("thermal_capture") is True]
    require(len(human_cameras) == manifest["cameras"]["human_count"] == 55, f"expected exactly fifty-five human cameras for {package_id}")
    require(len(aerial_cameras) == manifest["cameras"]["aerial_count"] == 7, f"expected exactly seven aerial cameras for {package_id}")
    require(len(negative_cameras) == manifest["cameras"]["negative_context_count"] == 11, f"expected exactly eleven negative context cameras for {package_id}")
    require(len(thermal_cameras) == manifest["cameras"]["thermal_count"] == 7, f"thermal contract must cover exactly seven cameras for {package_id}")
    phone_cameras = [camera for camera in human_cameras if camera.get("capture_device_profile") == "smartphone_main_26mm_equivalent"]
    professional_cameras = [camera for camera in human_cameras if camera.get("capture_device_profile") == "professional_full_frame_50mm"]
    require(len(phone_cameras) == 18 and len(professional_cameras) == 37, f"human phone/professional profile pool is invalid for {package_id}")
    require(manifest["cameras"]["profile_counts"].get("smartphone_main_26mm_equivalent") == 18, f"phone profile count mismatch for {package_id}")
    require(manifest["cameras"]["profile_counts"].get("professional_full_frame_50mm") == 37, f"professional profile count mismatch for {package_id}")
    rotated_positive_human_count = 0
    for camera in camera_plan:
        camera_id = str(camera.get("camera_id"))
        position = camera.get("position_l93_ngf_ign69_m", [])
        require(len(position) == 3, f"{camera_id} has no georeferenced position for {package_id}")
        height_above_ground = float(camera.get("height_above_ground_m", 0.0))
        require(abs(float(position[2]) - float(camera.get("ground_elevation_m", 0.0)) - height_above_ground) < 0.05, f"{camera_id} height metadata is inconsistent for {package_id}")
        require(camera.get("line_of_sight_verified") is True, f"{camera_id} has no verified physical fire corridor for {package_id}")
        require(float(camera.get("terrain_los_clearance_m", -1.0)) >= 1.0, f"terrain blocks {camera_id} physical fire corridor for {package_id}")
        require(len(camera.get("target_local_m", [])) == 3 and len(camera.get("fire_target_local_m", [])) == 3, f"{camera_id} target metadata is incomplete for {package_id}")
        if camera.get("role") == "aerial":
            require(camera.get("placement_type") == "aerial" and camera.get("placement_contract") == "aerial_overview_with_clear_fire_target", f"{camera_id} is not a valid aerial camera for {package_id}")
            require(camera.get("access_surface") == "airborne_platform", f"{camera_id} has invalid aerial provenance for {package_id}")
            require(height_above_ground >= 175.0, f"{camera_id} is below the aerial altitude contract for {package_id}")
            require(camera.get("sample_capability") == "positive_fire" and camera.get("expected_fire_in_frame") is True, f"{camera_id} is not an aerial positive view for {package_id}")
            require(camera.get("thermal_capture") is True, f"{camera_id} is missing thermal capture for {package_id}")
            require(camera.get("target_kind") == "fire_ignition_and_visible_front", f"{camera_id} does not target the fire for {package_id}")
            require(abs(float(camera.get("fire_bearing_offset_degrees", 0.0))) < 0.001, f"{camera_id} aerial bearing is not centred on the fire for {package_id}")
            require(float(camera.get("focal_length_mm", 100.0)) <= 35.0, f"{camera_id} has an invalid aerial focal length for {package_id}")
            require(float(camera.get("distance_to_target_m", 10000.0)) <= 580.1, f"{camera_id} is too far from the fire target for {package_id}")
            continue
        require(camera.get("placement_contract") == "human_photo_site_with_clear_axis_to_fire", f"{camera_id} is not placed as a human photographer for {package_id}")
        require(camera.get("placement_type") in {"roadside", "bridge", "upper_floor", "garden"}, f"{camera_id} has an invalid human photo-site type for {package_id}")
        require(bool(camera.get("access_surface")), f"{camera_id} has no access provenance for {package_id}")
        profile = camera.get("capture_device_profile")
        require(profile in {"smartphone_main_26mm_equivalent", "professional_full_frame_50mm"}, f"{camera_id} has an invalid capture-device profile for {package_id}")
        expected_eye_height = 1.75 if profile == "smartphone_main_26mm_equivalent" else 2.15
        expected_focal = 26.0 if profile == "smartphone_main_26mm_equivalent" else 50.0
        expected_frame = "phone_context_frame" if profile == "smartphone_main_26mm_equivalent" else "professional_standard_frame"
        require(abs(float(camera.get("eye_height_m", 0.0)) - expected_eye_height) < 0.001, f"{camera_id} has inconsistent mount height for {package_id}")
        require(abs(float(camera.get("focal_length_mm", 0.0)) - expected_focal) < 0.001, f"{camera_id} has inconsistent focal length for {package_id}")
        require(camera.get("framing_style") == expected_frame, f"{camera_id} has inconsistent framing metadata for {package_id}")
        require(abs(float(camera.get("horizontal_aperture_mm", 0.0)) - 36.0) < 0.001 and abs(float(camera.get("vertical_aperture_mm", 0.0)) - 20.25) < 0.001, f"{camera_id} has invalid full-frame aperture metadata for {package_id}")
        require(camera.get("thermal_capture") is False, f"{camera_id} incorrectly enables aerial-only thermal capture for {package_id}")
        if camera.get("placement_type") == "upper_floor":
            require(height_above_ground >= 3.2 and bool(camera.get("host_building")), f"{camera_id} is not on a valid upper floor for {package_id}")
        else:
            require(height_above_ground >= 1.699, f"{camera_id} is below human eye level for {package_id}")
        require(float(camera.get("foreground_clearance_m", -1.0)) >= 100.0, f"foreground objects block {camera_id} fire view for {package_id}")
        bearing_offset = abs(float(camera.get("fire_bearing_offset_degrees", 0.0)))
        if camera.get("sample_capability") == "negative_context":
            require(camera.get("expected_fire_in_frame") is False and camera.get("target_kind") == "negative_context_away_from_fire", f"{camera_id} is not an explicit negative view for {package_id}")
            require(90.0 <= bearing_offset <= 100.0, f"{camera_id} does not rotate far enough away from the fire for {package_id}")
        else:
            require(camera.get("sample_capability") == "positive_fire" and camera.get("expected_fire_in_frame") is True, f"{camera_id} has an invalid positive-view contract for {package_id}")
            require(camera.get("target_kind") == "fire_ignition_and_visible_front", f"{camera_id} does not target the fire for {package_id}")
            require(bearing_offset <= 8.0, f"{camera_id} rotation pushes the fire out of frame for {package_id}")
            rotated_positive_human_count += int(bearing_offset > 0.0)
    require(rotated_positive_human_count >= 8, f"too few intentionally rotated positive human cameras for {package_id}")
    require(manifest["dataset"]["expected_viewpoint_plans"] == 3600, f"expected 3,600 scheduled viewpoint plans for {package_id}")
    require(manifest["dataset"]["zooms_per_view"] == 5 and manifest["dataset"]["captures_per_state"] == 100, f"invalid five-zoom contract for {package_id}")
    require(manifest["dataset"]["expected_capture_cases"] == 18000, f"expected 18,000 scheduled captures for {package_id}")
    require(manifest["dataset"]["expected_positive_cases"] == 14400 and manifest["dataset"]["expected_negative_cases"] == 3600, f"invalid positive/negative totals for {package_id}")
    require(set(manifest["dataset"]["modalities"]) == EXPECTED_MODALITIES, f"unexpected modality contract for {package_id}")

    site = (package / "site/site.usda").read_text(encoding="utf-8")
    for token in ("Terrain", "Buildings", "Vegetation", "OcclusionProxies", "CameraCandidates"):
        require(f'"{token}"' in site, f"site missing {token} payload for {package_id}")
    require('"SourceVegetationMesh"' not in site, f"legacy vegetation is still composed for {package_id}")
    terrain_sources = sorted(
        path for path in (package / "source-usd/terrain").glob("*.usda") if path.name != "index.usda"
    )
    require(len(terrain_sources) == manifest["site"]["terrain_payloads"], f"terrain source count mismatch for {package_id}")
    source_usd_manifest = read_json(package / "source-usd/manifest.json")
    require(source_usd_manifest.get("orthophoto", {}).get("sha256") == ground_texture.get("sha256"), f"source USD orthophoto provenance mismatch for {package_id}")
    terrain_records = {str(record["terrain_tile_id"]).replace("-", "_"): record for record in source_usd_manifest["terrain"]}
    for terrain_source in terrain_sources:
        terrain_text = terrain_source.read_text(encoding="utf-8")
        require(
            'uniform token orientation = "leftHanded"' in terrain_text,
            f"terrain winding is not declared for top-side orthophoto rendering: {terrain_source}",
        )
        require("UsdUVTexture" in terrain_text and "primvars:st" in terrain_text, f"terrain orthophoto projection is incomplete: {terrain_source}")
        require('fireviewer:ground_texture = "ign_bd_ortho_georeferenced"' in terrain_text, f"terrain does not declare the real ground texture: {terrain_source}")
        require("@../textures/ign-bd-ortho-real-ground.jpg@" in terrain_text and "_colour.png" not in terrain_text, f"terrain still references a stylized colour layer: {terrain_source}")
        require('token inputs:sourceColorSpace = "sRGB"' in terrain_text, f"terrain texture color space is not explicit: {terrain_source}")
        record = terrain_records[terrain_source.stem]
        west, south, east, north = (float(value) for value in record["bounds_l93_metres"])
        expected_northwest_uv = ((west - 876000.0) / 16000.0, (north - 6403000.0) / 10000.0)
        expected_token = f"({expected_northwest_uv[0]:.4f}, {expected_northwest_uv[1]:.4f})"
        require(expected_token in terrain_text, f"terrain UVs are not georeferenced to site coordinates: {terrain_source}")
    vegetation = (package / "site/payloads/vegetation.payload.usda").read_text(encoding="utf-8")
    require('def PointInstancer "Trees"' in vegetation, f"site vegetation is not point-instanced for {package_id}")
    require(f'custom int fireviewer:source_tree_count = {manifest["site"]["vegetation_point_instances"]}' in vegetation, f"tree source count mismatch for {package_id}")
    placement = read_json(package / str(manifest["site"]["vegetation_placement"]))
    require(placement["counts"]["tree_instances"] == manifest["site"]["vegetation_point_instances"], f"MNT/MNS placement count mismatch for {package_id}")
    require(placement["counts"]["prototype_assets"] == 6 and len(placement["assets"]) == 6, f"tree asset manifest is incomplete for {package_id}")
    require(placement["source_rebuild"]["source_hashes_verified"] is True, f"MNT/MNS hashes are not verified for {package_id}")
    require(placement["method"]["segmentation"] == "all_accepted_0m50_crown_apices_without_post_detection_thinning", f"vegetation was thinned for {package_id}")
    require(len(placement["tiles"]) == 128 and all(int(tile["tree_instances"]) > 0 for tile in placement["tiles"]), f"vegetation does not cover all 128 source tiles for {package_id}")
    buildings = (package / "site/payloads/buildings.payload.usda").read_text(encoding="utf-8")
    require(
        'def PointInstancer "SimpleBuildings"' in buildings
        and buildings.count('def Mesh "BodyStyle') == 4
        and buildings.count('def Mesh "RoofStyle') == 4
        and 'def Mesh "WindowBand"' in buildings
        and 'def Mesh "Entrance"' in buildings
        and 'quath[] orientations' in buildings,
        f"improved oriented building bodies, gabled roofs, windows or entrances are missing for {package_id}",
    )
    require(
        'custom string fireviewer:component_merge = "adjacent_lower_upper_source_components_with_matching_oriented_xy_footprint"' in buildings,
        f"stacked source building components were not merged for {package_id}",
    )
    occlusion = (package / "site/payloads/occlusion.payload.usda").read_text(encoding="utf-8")
    require('fireviewer:occlusion_proxy = true' in occlusion or manifest["site"]["occlusion_proxies"] == 0, f"missing occlusion proxy flags for {package_id}")

    scenario = (package / "scenarios/scenario.usda").read_text(encoding="utf-8")
    state_ids = STATE_RE.findall(scenario)
    expected_state_ids = [f"state_{index:03d}" for index in range(1, 181)]
    require(state_ids == expected_state_ids, f"invalid ordered 180-state variant set for {package_id}")
    propagation_reference = manifest["scenario"].get("propagation", {})
    propagation_path = package / str(propagation_reference.get("path", ""))
    require(propagation_path.is_file(), f"missing continuous propagation sidecar for {package_id}")
    require(hashlib.sha256(propagation_path.read_bytes()).hexdigest() == propagation_reference.get("sha256"), f"propagation sidecar hash mismatch for {package_id}")
    propagation = read_json(propagation_path)
    require(propagation["schema"] == "fireviewer.synthetic-fire-spread.v2", f"unexpected propagation schema for {package_id}")
    require(propagation["model"] == "least_arrival_time_25m_grid_with_fuel_wind_slope_and_distinct_state_fronts", f"unexpected propagation model for {package_id}")
    require(abs(float(propagation["drivers"]["cell_size_m"]) - 25.0) < 0.001, f"fire grid is not refined to 25 m for {package_id}")
    require(propagation["terrain_input"] == "uploaded_elevation_cog_tiles" and propagation["fuel_input"] == "uploaded_tree_instances_density", f"propagation inputs are not tied to the uploaded map for {package_id}")
    require(propagation["weather_input"] == "explicit_synthetic_driver_not_observed_incident_weather", f"weather provenance is not explicit for {package_id}")
    require(propagation["simulation_id"] == propagation_reference.get("simulation_id"), f"propagation identity mismatch for {package_id}")
    field_path = propagation_path.parent / str(propagation["field_file"])
    require(field_path.is_file(), f"missing propagation grid field for {package_id}")
    with np.load(field_path) as field:
        require(set(field.files) == {"elevation_m", "fuel_load", "burnable_mask", "arrival_time_s", "spread_rate_m_s"}, f"invalid propagation grid fields for {package_id}")
        shape = tuple(int(value) for value in propagation["grid_shape"])
        require(tuple(field["arrival_time_s"].shape) == shape, f"propagation grid shape mismatch for {package_id}")
        require(np.isfinite(field["arrival_time_s"]).any(), f"propagation grid has no reached cells for {package_id}")
    propagation_states = propagation["states"]
    require(len(propagation_states) == 180, f"propagation sidecar does not have 180 snapshots for {package_id}")
    elapsed = [float(record["elapsed_s"]) for record in propagation_states]
    burned_area = [float(record["burned_area_m2"]) for record in propagation_states]
    require(all(current > previous for previous, current in zip(elapsed, elapsed[1:])), f"non-continuous state timing for {package_id}")
    require(all(current > previous for previous, current in zip(burned_area, burned_area[1:])), f"every fire state must visibly expand for {package_id}")
    require(len(set(burned_area)) == 180, f"fire progression does not have 180 distinct states for {package_id}")
    require(burned_area[0] <= 625.0, f"initial ignition is already too large for {package_id}")
    for state_index, (state, propagation_state) in enumerate(zip(manifest["scenario"]["states"], propagation_states), start=1):
        expected_day = (state_index - 1) // 10 + 1
        expected_state_in_day = (state_index - 1) % 10 + 1
        expected_observation_elapsed = (state_index - 1) * 8640
        require(state["state_id"] == f"state_{state_index:03d}", f"state identity mismatch at position {state_index} for {package_id}")
        require(state["incident_day_index"] == expected_day and state["state_in_day"] == expected_state_in_day, f"state day cadence mismatch for {state['state_id']}")
        require(state["observation_elapsed_s"] == expected_observation_elapsed, f"observation clock mismatch for {state['state_id']}")
        require(propagation_state["state_id"] == state["state_id"], f"propagation state identity mismatch for {state['state_id']}")
        require(propagation_state["incident_day_index"] == expected_day and propagation_state["state_in_day"] == expected_state_in_day, f"propagation cadence mismatch for {state['state_id']}")
        require(propagation_state["observation_elapsed_s"] == expected_observation_elapsed, f"propagation observation clock mismatch for {state['state_id']}")
        state_path = package / "scenarios" / state["path"]
        text = state_path.read_text(encoding="utf-8")
        for token in ("BurnedPerimeter", "VisibleFireFront", "SmokeSources", "synthetic_physically_driven_fire_spread_on_real_uploaded_map_not_incident_reconstruction"):
            require(token in text, f"{state_path} missing {token}")
        require(
            "FlowVisual" not in text and "FlowEmitterSphere" not in text and "FlowEmitterMesh" not in text,
            f"{state_path} incorrectly restarts the persistent Flow volume",
        )
        require(f'fireviewer:simulation_id = "{propagation["simulation_id"]}"' in text, f"{state_path} does not belong to propagation simulation")
        require(int(state["flow_emitter_count"]) == 48, f"{state_path} does not expose the 48-source fire-front truth contract")
        require(abs(float(state["elapsed_s"]) - float(propagation_state["elapsed_s"])) < 0.01, f"elapsed state mismatch for {state_path}")
        require(abs(float(state["burned_area_m2"]) - float(propagation_state["burned_area_m2"])) < 0.01, f"burned area mismatch for {state_path}")

    flow_path = package / "scenarios/flow.usda"
    require(flow_path.is_file(), f"missing persistent Flow layer for {package_id}")
    flow = flow_path.read_text(encoding="utf-8")
    for token in (
        'fireviewer:mode = "persistent_mesh_front_wildland_combustion"',
        'custom int fireviewer:emitter_count = 50',
        'custom int fireviewer:hotspot_emitter_count = 48',
        'custom int fireviewer:front_patch_count = 256',
        'custom int fireviewer:smoke_plume_count = 48',
        'custom float fireviewer:seconds_per_state = 6.000000',
        'def FlowRenderSettingsParams "RenderSettings"',
        'custom int maxBlocks = 16384',
        'custom float densityCellSize = 0.45',
        'float buoyancyPerTemp = 2.8',
        'custom float buoyancyPerSmoke = 4.5',
        'custom float damping = 0.025000',
        'float fade = 0.025000',
        'float3 gravity = (0, 0, -9.81)',
        'custom float attenuation = 4.5',
        'custom float attenuation = 6',
        'custom float smokeMask = 0.28',
        '(0.22, 0.24, 0.26, 0.16), (0.52, 0.50, 0.47, 0.38)',
        'custom float coupleRateBurn = 0',
        'custom float coupleRateSmoke = 0',
        'custom float burn = 0',
        'custom float smoke = 0',
        'def FlowEmitterMesh "FrontRibbonEmitter"',
        'def FlowEmitterMesh "SmokePlumeEmitter"',
        'custom string fireviewer:placement_contract = "same_xy_as_hotspot_front_lifted_1.600000m"',
        'float smoke = 0.320000',
        'float coupleRateSmoke = 1.6',
        'float coupleRateVelocity = 3.2',
        'float3[] meshPositions.timeSamples',
        'int[] meshFaceVertexCounts',
        'custom float3 position.timeSamples',
        'custom float radius.timeSamples',
        'custom float fuel.timeSamples',
        'float temperature.timeSamples',
        'custom float[] xPoints = [0, 0.035, 0.12, 0.32, 0.62, 1]',
    ):
        require(token in flow, f"persistent Flow layer missing {token} for {package_id}")
    require(flow.count('def FlowEmitterSphere "Emitter_') == 48, f"persistent Flow layer does not have 48 emitters for {package_id}")
    require(flow.count('def FlowEmitterMesh "FrontRibbonEmitter"') == 1, f"persistent Flow layer does not have one continuous front emitter for {package_id}")
    require(flow.count('def FlowEmitterMesh "SmokePlumeEmitter"') == 1, f"persistent Flow layer does not have one aligned smoke emitter for {package_id}")
    require(flow.count(".timeSamples = {") == 48 * 5 + 2, f"persistent Flow layer has an incomplete animation schedule for {package_id}")

    appearance = (package / "appearance/appearance.usda").read_text(encoding="utf-8")
    for token in ("FlowClose", "SmokeMidDistance", "DistantFire", "DecorDegradation", "omni.flowusd"):
        require(token in appearance, f"appearance missing {token} for {package_id}")
    dataset_stage = (package / "dataset.usda").read_text(encoding="utf-8")
    require('int "rtx:flow:maxBlocks" = 16384' in dataset_stage, f"dataset Flow block budget is not authored for {package_id}")
    require('@scenarios/flow.usda@' in dataset_stage, f"dataset does not compose the persistent Flow layer for {package_id}")
    require('timeCodesPerSecond = 1' in dataset_stage and 'endTimeCode = 1080' in dataset_stage, f"dataset Flow playback clock is not one real second per time code for {package_id}")
    cameras = (package / "cameras/fixed_cameras.usda").read_text(encoding="utf-8")
    camera_ids = CAMERA_RE.findall(cameras)
    require(camera_ids == [f"CAM_{index:02d}" for index in range(1, 63)], f"invalid fixed camera pool for {package_id}: {camera_ids}")
    require(cameras.count('fireviewer:placement_contract = "human_photo_site_with_clear_axis_to_fire"') == 55, f"camera placement contract is incomplete for {package_id}")
    require(cameras.count('fireviewer:placement_contract = "aerial_overview_with_clear_fire_target"') == 7, f"aerial placement contract is incomplete for {package_id}")
    require(cameras.count('fireviewer:capture_device_profile = "smartphone_main_26mm_equivalent"') == 18, f"phone camera metadata is incomplete for {package_id}")
    require(cameras.count('fireviewer:capture_device_profile = "professional_full_frame_50mm"') == 37, f"professional camera metadata is incomplete for {package_id}")
    require(cameras.count('fireviewer:capture_device_profile = "aerial_rgb_thermal_mapping_camera"') == 7, f"aerial camera profile metadata is incomplete for {package_id}")
    require(cameras.count("fireviewer:eye_height_m = 1.750000") == 18 and cameras.count("fireviewer:eye_height_m = 2.150000") == 37, f"camera mount-height metadata is incomplete for {package_id}")
    require(cameras.count("fireviewer:line_of_sight_verified = true") == 62, f"camera fire-corridor metadata is incomplete for {package_id}")
    require(cameras.count("fireviewer:thermal_capture = true") == 7, f"aerial thermal metadata is incomplete for {package_id}")
    require(cameras.count('fireviewer:sample_capability = "negative_context"') == 11, f"negative camera metadata is incomplete for {package_id}")
    camera_candidates = (package / "site/payloads/camera_candidates.payload.usda").read_text(encoding="utf-8")
    require('fireviewer:role = "selected_human_and_aerial_camera_positions"' in camera_candidates, f"camera candidate payload does not cover the complete pool for {package_id}")
    buildings_payload = (package / "site/payloads/buildings.payload.usda").read_text(encoding="utf-8")
    require('fireviewer:vertical_alignment = "module_center_sampled_from_mnt_ground"' in buildings_payload, f"building payload is not grounded on the MNT for {package_id}")
    require("quatf[] orientations" not in buildings_payload and "quath[] orientations" in buildings_payload, f"building PointInstancer orientation type is invalid for {package_id}")

    runtime = read_json(package / "runtime/runtime-contract.json")
    require(runtime["required_extensions"] == ["omni.replicator.core", "omni.flowusd"], f"invalid runtime extension contract for {package_id}")
    for module_name in ("writer_module", "runner_module"):
        module_record = runtime[module_name]
        module_path = package / "runtime" / str(module_record["path"])
        require(module_path.is_file(), f"missing standalone runtime {module_name} for {package_id}")
        require(hashlib.sha256(module_path.read_bytes()).hexdigest() == module_record["sha256"], f"runtime {module_name} checksum mismatch for {package_id}")
    capture = runtime["capture"]
    require(capture["incident_days"] == 18 and capture["states_per_day"] == 10 and capture["views_per_state"] == 20, f"invalid runtime incident cadence for {package_id}")
    require(capture["human_cameras"] == 55 and capture["aerial_cameras"] == 7 and len(capture["camera_pool"]) == 62, f"invalid runtime camera pool for {package_id}")
    require(capture["positive_views_per_state"] == 16 and capture["negative_views_per_state"] == 4, f"invalid runtime positive/negative quota for {package_id}")
    require(capture["profile_mix_per_state"] == {"professional": 10, "phone": 8, "aerial": 2}, f"invalid runtime camera-profile mix for {package_id}")
    require(capture["zooms_per_view"] == 5 and capture["captures_per_state"] == 100, f"invalid runtime zoom contract for {package_id}")
    require(capture["expected_viewpoint_plans"] == 3600 and capture["expected_capture_cases"] == 18000, f"invalid runtime capture total for {package_id}")
    schedule_path = package / "runtime" / str(capture["schedule_path"])
    require(schedule_path.is_file(), f"missing capture schedule for {package_id}")
    schedule_sha256 = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    require(schedule_sha256 == capture["schedule_sha256"], f"runtime capture schedule hash mismatch for {package_id}")
    schedule_reference = manifest["dataset"]["capture_schedule"]
    require((package / str(schedule_reference["path"])).resolve() == schedule_path.resolve(), f"manifest schedule path mismatch for {package_id}")
    require(schedule_sha256 == schedule_reference["sha256"], f"manifest capture schedule hash mismatch for {package_id}")
    schedule = read_json(schedule_path)
    require(schedule["schema"] == "fireviewer.random-camera-zoom-schedule.v2", f"unexpected capture schedule schema for {package_id}")
    require(schedule["incident_days"] == 18 and schedule["states_per_day"] == 10 and schedule["views_per_state"] == 20, f"invalid capture schedule cadence for {package_id}")
    require(schedule["zooms_per_view"] == 5 and schedule["captures_per_state"] == 100, f"invalid capture schedule zoom cadence for {package_id}")
    require(schedule["camera_pool_count"] == 62 and schedule["human_camera_count"] == 55 and schedule["aerial_camera_count"] == 7, f"invalid capture schedule camera pool for {package_id}")
    require(schedule["expected_viewpoint_plans"] == 3600 and schedule["expected_capture_cases"] == 18000, f"invalid capture schedule totals for {package_id}")
    require(schedule["expected_positive_cases"] == 14400 and schedule["expected_negative_cases"] == 3600, f"invalid capture schedule class totals for {package_id}")
    require(len(schedule["states"]) == 180, f"capture schedule does not contain 180 states for {package_id}")
    camera_by_id = {str(camera["camera_id"]): camera for camera in camera_plan}
    scheduled_positive = 0
    scheduled_negative = 0
    scheduled_thermal = 0
    required_view_fields = {
        "capture_id", "plan_id", "zoom_count", "zoom_variants", "camera_pose_contract", "incident_id", "simulation_id", "day_index", "state_in_day", "global_state_index", "state_id",
        "fire_state_elapsed_s", "burned_area_m2", "active_front_length_m", "mean_front_spread_rate_m_s", "burned_tree_count",
        "camera_id", "camera_path", "camera_role", "placement_type", "placement_contract", "capture_device_profile", "framing_style", "access_surface", "access_tile",
        "host_building", "sample_kind", "negative_reason", "expected_fire_visible", "thermal_expected", "thermal_contract",
        "expected_modalities", "line_of_sight_verified", "position_local_m", "position_l93_ngf_ign69_m", "look_at_local_m",
        "fire_target_local_m", "height_above_ground_m", "ground_elevation_m", "terrain_los_clearance_m",
        "foreground_clearance_m", "distance_to_target_m", "fire_bearing_offset_degrees", "focal_length_mm", "focal_length_35mm_equivalent_mm", "horizontal_aperture_mm", "vertical_aperture_mm",
        "observation_elapsed_s", "selection_token_sha256",
    }
    for state_index, schedule_state in enumerate(schedule["states"], start=1):
        state_id = f"state_{state_index:03d}"
        expected_day = (state_index - 1) // 10 + 1
        expected_slot = (state_index - 1) % 10 + 1
        require(schedule_state["state_id"] == state_id and schedule_state["global_state_index"] == state_index, f"schedule state identity mismatch for {state_id}")
        require(schedule_state["day_index"] == expected_day and schedule_state["state_in_day"] == expected_slot, f"schedule cadence mismatch for {state_id}")
        selected_ids = schedule_state["camera_ids"]
        views = schedule_state["views"]
        require(schedule_state["view_count"] == len(selected_ids) == len(views) == 20, f"schedule must select twenty views for {state_id}")
        require(schedule_state["zoom_count_per_view"] == 5 and schedule_state["capture_count"] == 100, f"schedule must produce five zooms per view for {state_id}")
        require(len(set(selected_ids)) == 20 and all(camera_id in camera_by_id for camera_id in selected_ids), f"schedule has invalid or repeated cameras for {state_id}")
        require(schedule_state["positive_view_count"] == 16 and schedule_state["negative_view_count"] == 4, f"invalid sample quota for {state_id}")
        require(schedule_state["professional_view_count"] == 10 and schedule_state["phone_view_count"] == 8 and schedule_state["aerial_view_count"] == 2, f"invalid camera-profile mix for {state_id}")
        require(hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest() == schedule_state["selection_sha256"], f"camera selection hash mismatch for {state_id}")
        state_positive = 0
        state_negative = 0
        for view_index, (camera_id, view) in enumerate(zip(selected_ids, views), start=1):
            require(required_view_fields <= set(view), f"capture metadata is incomplete for {state_id} view {view_index}")
            require(view["camera_id"] == camera_id and view["state_id"] == state_id, f"capture identity mismatch for {state_id} view {view_index}")
            require(view["capture_id"] == view["plan_id"] and view["zoom_count"] == 5, f"zoom-set identity mismatch for {state_id} view {view_index}")
            require(view["camera_pose_contract"] == "same_position_orientation_and_target_across_zoom_set", f"zoom set can alter pose for {view['capture_id']}")
            require(view["day_index"] == expected_day and view["state_in_day"] == expected_slot and view["global_state_index"] == state_index, f"capture cadence mismatch for {view['capture_id']}")
            require(view["observation_elapsed_s"] == (state_index - 1) * 8640, f"capture observation clock mismatch for {view['capture_id']}")
            camera = camera_by_id[camera_id]
            require(view["camera_role"] == camera["role"] and view["placement_type"] == camera["placement_type"], f"capture camera metadata mismatch for {view['capture_id']}")
            require(view["capture_device_profile"] == camera["capture_device_profile"] and view["framing_style"] == camera["framing_style"], f"capture framing profile mismatch for {view['capture_id']}")
            require(view["camera_path"] == f"/World/Cameras/{camera_id}" and view["placement_contract"] == camera["placement_contract"], f"capture camera identity/contract mismatch for {view['capture_id']}")
            require(view["access_surface"] == camera["access_surface"] and view["access_tile"] == camera["access_tile"] and view["host_building"] == camera["host_building"], f"capture access provenance mismatch for {view['capture_id']}")
            require(view["sample_kind"] == camera["sample_capability"], f"capture sample kind mismatch for {view['capture_id']}")
            require(view["expected_fire_visible"] is camera["expected_fire_in_frame"], f"capture visibility contract mismatch for {view['capture_id']}")
            require(view["thermal_expected"] is camera["thermal_capture"], f"capture thermal contract mismatch for {view['capture_id']}")
            require(bool(view["thermal_expected"]) == (camera["role"] == "aerial"), f"thermal output is not aerial-only for {view['capture_id']}")
            if view["thermal_expected"]:
                require(view["thermal_contract"] == "synthetic_16bit_kelvin_non_radiometric" and {"synthetic_thermal_kelvin", "synthetic_thermal_16bit", "thermal_metadata"} <= set(view["expected_modalities"]), f"aerial thermal metadata is incomplete for {view['capture_id']}")
            else:
                require(view["thermal_contract"] is None and "synthetic_thermal_16bit" not in view["expected_modalities"], f"ground view incorrectly requests thermal output for {view['capture_id']}")
            require(view["negative_reason"] == ("camera_rotated_away_from_fire_for_context_control" if view["sample_kind"] == "negative_context" else None), f"negative sample rationale mismatch for {view['capture_id']}")
            require(view["line_of_sight_verified"] is True, f"capture has no verified physical fire corridor for {view['capture_id']}")
            require(view["position_local_m"] == list(camera["position_local_m"]) and view["position_l93_ngf_ign69_m"] == list(camera["position_l93_ngf_ign69_m"]), f"capture camera coordinates mismatch for {view['capture_id']}")
            require(view["look_at_local_m"] == list(camera["target_local_m"]) and view["fire_target_local_m"] == list(camera["fire_target_local_m"]), f"capture target coordinates mismatch for {view['capture_id']}")
            require(view["simulation_id"] == propagation["simulation_id"] and view["incident_id"] == schedule["incident_id"], f"capture incident/simulation identity mismatch for {view['capture_id']}")
            require(abs(float(view["fire_bearing_offset_degrees"]) - float(camera["fire_bearing_offset_degrees"])) < 0.001, f"capture bearing metadata mismatch for {view['capture_id']}")
            require(abs(float(view["focal_length_mm"]) - float(camera["focal_length_mm"])) < 0.001, f"capture focal metadata mismatch for {view['capture_id']}")
            require(abs(float(view["focal_length_35mm_equivalent_mm"]) - float(camera["focal_length_35mm_equivalent_mm"])) < 0.001, f"capture equivalent focal metadata mismatch for {view['capture_id']}")
            require(abs(float(view["horizontal_aperture_mm"]) - float(camera["horizontal_aperture_mm"])) < 0.001 and abs(float(view["vertical_aperture_mm"]) - float(camera["vertical_aperture_mm"])) < 0.001, f"capture aperture metadata mismatch for {view['capture_id']}")
            expected_token = hashlib.sha256(f"{package_id}|{view['capture_id']}".encode("utf-8")).hexdigest()
            require(view["selection_token_sha256"] == expected_token, f"capture token mismatch for {view['capture_id']}")
            variants = view["zoom_variants"]
            require(len(variants) == 5, f"capture plan does not expose five zooms for {view['capture_id']}")
            expected_multipliers = [0.75, 1.0, 1.25, 1.5, 2.0]
            require([float(variant["zoom_multiplier"]) for variant in variants] == expected_multipliers, f"invalid zoom progression for {view['capture_id']}")
            require(len({str(variant["capture_id"]) for variant in variants}) == 5, f"zoom capture identifiers are not unique for {view['capture_id']}")
            for zoom_index, variant in enumerate(variants, start=1):
                require(variant["zoom_set_id"] == view["plan_id"] and int(variant["zoom_index"]) == zoom_index, f"zoom identity mismatch for {view['capture_id']}")
                require(variant["camera_pose_contract"] == view["camera_pose_contract"], f"zoom pose contract mismatch for {variant['capture_id']}")
                require(abs(float(variant["focal_length_mm"]) - float(camera["focal_length_mm"]) * expected_multipliers[zoom_index - 1]) < 0.001, f"zoom focal mismatch for {variant['capture_id']}")
                require(abs(float(variant["focal_length_35mm_equivalent_mm"]) - float(camera["focal_length_35mm_equivalent_mm"]) * expected_multipliers[zoom_index - 1]) < 0.001, f"zoom equivalent focal mismatch for {variant['capture_id']}")
                require(float(variant["horizontal_aperture_mm"]) == float(camera["horizontal_aperture_mm"]) and float(variant["vertical_aperture_mm"]) == float(camera["vertical_aperture_mm"]), f"zoom changes sensor aperture for {variant['capture_id']}")
                expected_zoom_token = hashlib.sha256(f"{package_id}|{variant['capture_id']}".encode("utf-8")).hexdigest()
                require(variant["selection_token_sha256"] == expected_zoom_token, f"zoom capture token mismatch for {variant['capture_id']}")
            state_positive += int(view["sample_kind"] == "positive_fire")
            state_negative += int(view["sample_kind"] == "negative_context")
            scheduled_thermal += int(bool(view["thermal_expected"]))
        require(state_positive == 16 and state_negative == 4, f"actual sample quota mismatch for {state_id}")
        scheduled_positive += state_positive
        scheduled_negative += state_negative
    require(scheduled_positive == 2880 and scheduled_negative == 720, f"aggregate viewpoint positive/negative schedule mismatch for {package_id}")
    require(scheduled_positive * 5 == 14400 and scheduled_negative * 5 == 3600, f"aggregate zoomed capture class totals mismatch for {package_id}")
    require(scheduled_thermal > 0, f"schedule never selects an aerial thermal view for {package_id}")
    qa = read_json(package / "qa/acceptance.json")
    require(qa["automated"]["expected_viewpoint_plans"] == 3600 and qa["automated"]["expected_capture_cases"] == 18000, f"invalid QA capture contract for {package_id}")
    require(qa["automated"]["zooms_per_view"] == 5, f"invalid QA zoom contract for {package_id}")
    require(qa["automated"]["expected_fire_states"] == 180 and qa["automated"]["expected_fixed_cameras"] == 62, f"invalid QA state/camera contract for {package_id}")
    require(qa["automated"]["states_per_day"] == 10 and qa["automated"]["random_views_per_state"] == 20, f"invalid QA cadence for {package_id}")
    require((package / "qa/HUMAN_REVIEW.md").is_file(), f"missing human review gate for {package_id}")
    reference_count = check_usd_references(package)
    return {"package_id": package_id, "usd_references": reference_count, "states": len(state_ids), "cameras": len(camera_ids), "tree_instances": manifest["site"]["vegetation_point_instances"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    index = read_json(root / "index.json")
    records = index.get("packages", [])
    require(records, "dataset index has no packages")
    details = []
    for record in records:
        package = root / str(record["package_id"])
        details.append(validate_package(package))
    print(json.dumps({"schema": "fireviewer.omniverse-dataset-validation.v1", "validated_packages": len(details), "details": details}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
