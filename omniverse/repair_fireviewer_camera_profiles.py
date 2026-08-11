"""Apply the approved mixed phone/pro camera framing to an existing package."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_fireviewer_dataset_usd import (
    HeightSampler,
    build_capture_schedule,
    canonical_json,
    look_at_quaternion,
    sha256_file,
    write_camera_candidate_payload,
    write_dataset_stage,
    write_fixed_cameras,
)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


def repair_camera(camera: dict[str, Any]) -> dict[str, Any]:
    camera = dict(camera)
    camera_index = int(str(camera["camera_id"]).split("_")[-1])
    position_local = [float(value) for value in camera["position_local_m"]]
    position_global = [float(value) for value in camera["position_l93_ngf_ign69_m"]]
    if camera["role"] == "aerial":
        camera["capture_device_profile"] = "aerial_rgb_thermal_mapping_camera"
        camera["framing_style"] = "aerial_overhead_frame" if camera["camera_id"] == "CAM_62" else "aerial_context_frame"
        camera["focal_length_35mm_equivalent_mm"] = float(camera["focal_length_mm"])
    else:
        is_phone = camera_index % 3 == 0
        eye_height = 1.75 if is_phone else 2.15
        current_height = float(camera["height_above_ground_m"])
        position_raise = 0.0 if camera["placement_type"] == "upper_floor" else max(0.0, eye_height - current_height)
        position_local[2] += position_raise
        position_global[2] += position_raise
        camera["height_above_ground_m"] = current_height + position_raise
        camera["eye_height_m"] = eye_height
        camera["placement_contract"] = "human_photo_site_with_clear_axis_to_fire"
        camera["capture_device_profile"] = "smartphone_main_26mm_equivalent" if is_phone else "professional_full_frame_50mm"
        camera["framing_style"] = "phone_context_frame" if is_phone else "professional_standard_frame"
        camera["focal_length_mm"] = 26.0 if is_phone else 50.0
        camera["focal_length_35mm_equivalent_mm"] = float(camera["focal_length_mm"])
    camera["horizontal_aperture_mm"] = 36.0
    camera["vertical_aperture_mm"] = 20.25
    camera["position_local_m"] = position_local
    camera["position_l93_ngf_ign69_m"] = position_global
    camera["orientation_quat_wxyz"] = look_at_quaternion(position_local, camera["target_local_m"])
    return camera


def repair_aerial_cameras(
    cameras: list[dict[str, Any]],
    *,
    sampler: HeightSampler,
    anchor: tuple[float, float],
) -> None:
    radii = (420.0, 500.0, 580.0)
    altitudes = (180.0, 210.0, 240.0)

    def terrain_clearance(position: list[float], target: list[float]) -> float:
        clearance = float("inf")
        for sample_index in range(1, 20):
            fraction = sample_index / 20.0
            local_x = position[0] + (target[0] - position[0]) * fraction
            local_y = position[1] + (target[1] - position[1]) * fraction
            ray_z = position[2] + (target[2] - position[2]) * fraction
            clearance = min(clearance, ray_z - sampler.at(local_x + anchor[0], local_y + anchor[1]))
        return clearance

    for camera in cameras:
        if camera["role"] != "aerial":
            continue
        target = [float(value) for value in camera["fire_target_local_m"]]
        camera_index = int(str(camera["camera_id"]).split("_")[-1])
        if camera["camera_id"] == "CAM_62":
            local_x, local_y = target[0], target[1]
            ground = sampler.at(local_x + anchor[0], local_y + anchor[1])
            position = [local_x, local_y, ground + 350.0]
            focal_length = 32.0
            camera["framing_style"] = "aerial_overhead_frame"
        else:
            previous = [float(value) for value in camera["position_local_m"]]
            delta_x = previous[0] - target[0]
            delta_y = previous[1] - target[1]
            direction_length = max((delta_x * delta_x + delta_y * delta_y) ** 0.5, 1e-6)
            slot = (camera_index - 56) % 3
            local_x = target[0] + delta_x / direction_length * radii[slot]
            local_y = target[1] + delta_y / direction_length * radii[slot]
            ground = sampler.at(local_x + anchor[0], local_y + anchor[1])
            position = [local_x, local_y, ground + altitudes[slot]]
            for fraction in tuple(index / 20.0 for index in range(1, 20)):
                sample_x = local_x + (target[0] - local_x) * fraction
                sample_y = local_y + (target[1] - local_y) * fraction
                terrain_z = sampler.at(sample_x + anchor[0], sample_y + anchor[1])
                required_z = (terrain_z + 8.0 - target[2] * fraction) / (1.0 - fraction)
                position[2] = max(position[2], required_z)
            focal_length = 35.0
            camera["framing_style"] = "aerial_context_frame"
        camera["position_local_m"] = position
        camera["position_l93_ngf_ign69_m"] = [position[0] + anchor[0], position[1] + anchor[1], position[2]]
        camera["ground_elevation_m"] = ground
        camera["height_above_ground_m"] = position[2] - ground
        camera["distance_to_target_m"] = ((position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2) ** 0.5
        camera["terrain_los_clearance_m"] = terrain_clearance(position, target)
        camera["focal_length_mm"] = focal_length
        camera["focal_length_35mm_equivalent_mm"] = focal_length
        camera["orientation_quat_wxyz"] = look_at_quaternion(position, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--site-source", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest_path = package / "manifest.json"
    runtime_path = package / "runtime/runtime-contract.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    cameras = [repair_camera(camera) for camera in manifest["cameras"]["plan"]]
    site_source = args.site_source.resolve()
    catalog = json.loads((site_source / "catalog.json").read_text(encoding="utf-8"))
    anchor = tuple(float(value) for value in catalog["spatial_contract"]["common_anchor_l93_metres"])
    sampler = HeightSampler(site_source, catalog)
    try:
        repair_aerial_cameras(cameras, sampler=sampler, anchor=anchor)
    finally:
        sampler.close()
    if len(cameras) != 62:
        raise RuntimeError(f"Expected 62 cameras, received {len(cameras)}")

    write_fixed_cameras(package / "cameras/fixed_cameras.usda", cameras)
    write_camera_candidate_payload(
        package / "site/payloads/camera_candidates.payload.usda",
        [tuple(camera["position_local_m"]) for camera in cameras],
    )
    capture_schedule = build_capture_schedule(manifest["package_id"], manifest["scenario"]["states"], cameras)
    schedule_path = package / "runtime/capture-schedule.json"
    write_json_atomic(schedule_path, capture_schedule)

    source_root = Path(__file__).resolve().parent
    writer_path = package / "runtime/fireviewer_replicator_writer.py"
    runner_path = package / "runtime/run_fireviewer_replicator_dataset.py"
    shutil.copy2(source_root / "fireviewer_replicator_writer.py", writer_path)
    shutil.copy2(source_root / "run_fireviewer_replicator_dataset.py", runner_path)
    runtime["writer_module"]["sha256"] = sha256_file(writer_path)
    runtime["runner_module"]["sha256"] = sha256_file(runner_path)
    runtime["capture"]["schedule_sha256"] = sha256_file(schedule_path)
    runtime["capture"]["camera_pool"] = [camera["camera_id"] for camera in cameras]
    runtime["capture"]["profile_mix_per_state"] = capture_schedule["profile_mix_per_state"]
    write_json_atomic(runtime_path, runtime)

    profile_counts = {
        profile: sum(camera["capture_device_profile"] == profile for camera in cameras)
        for profile in sorted({camera["capture_device_profile"] for camera in cameras})
    }
    manifest["cameras"]["plan"] = cameras
    manifest["cameras"]["profile_counts"] = profile_counts
    manifest["dataset"]["capture_schedule"]["sha256"] = sha256_file(schedule_path)
    manifest["dataset"]["camera_profile_mix_per_state"] = capture_schedule["profile_mix_per_state"]
    provenance_path = package / "assets/environments/farm_field_puresky_4k.provenance.json"
    environment_path = package / "assets/environments/farm_field_puresky_4k.hdr"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest["environment"] = {
        "lighting": "cc0_latlong_hdri_plus_soft_distant_sun",
        "hdri_path": "assets/environments/farm_field_puresky_4k.hdr",
        "hdri_sha256": sha256_file(environment_path),
        "provenance_path": "assets/environments/farm_field_puresky_4k.provenance.json",
        "source_url": provenance["source_page"],
        "license": provenance["license"],
    }
    manifest["camera_profile_repaired_at"] = datetime.now(timezone.utc).isoformat()
    write_dataset_stage(package / "dataset.usda", package_id=manifest["package_id"], anchor=anchor)
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "camera_profiles_repaired",
                "package": str(package),
                "camera_count": len(cameras),
                "profile_counts": profile_counts,
                "profile_mix_per_state": capture_schedule["profile_mix_per_state"],
                "schedule_sha256": sha256_file(schedule_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
