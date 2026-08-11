"""FireViewer custom Replicator writer for multimodal wildfire SDG captures.

This module is loaded only inside an Isaac Sim / Omniverse Replicator Python
runtime.  It writes one coherent record per scheduled camera and fire state: RGB,
semantic data, class-specific front/perimeter masks, depth, point cloud, camera
data, EPSG:2154 geolocation and a fail-closed abstention record.  Scheduled
aerial views additionally receive an aligned synthetic 16-bit thermal product
that is explicitly non-radiometric.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import omni.replicator.core as rep
from omni.replicator.core import Writer
from omni.replicator.core.functional import write_image

from fireviewer_capture_storage import (
    compact_pointcloud_attributes,
    storage_profile_contract,
    write_array_npz,
    write_named_arrays_npz,
)


PREVIEW_DOWNSAMPLE_FACTOR = 4


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(*, data: Any, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, dict) and "data" in value:
        raw_info = value.get("info")
        if isinstance(raw_info, dict):
            info = dict(raw_info)
        elif raw_info is None:
            info = {}
        else:
            info = {"raw_info": raw_info}
        # Replicator 6 renderProduct payloads flatten annotator ``info`` keys
        # (for example ``idToLabels``) next to ``data``.  Preserve both that
        # shape and the legacy nested shape so semantic masks remain portable.
        for key, item in value.items():
            if key not in {"data", "info"}:
                info[str(key)] = item
        return value["data"], info
    return value, {}


def _labels(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        result: set[str] = set()
        for key, item in value.items():
            if str(key) in {"class", "label", "labels", "semanticLabel", "semanticLabels"}:
                result.update(_labels(item))
            elif isinstance(item, (dict, list, tuple, set)):
                result.update(_labels(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            result.update(_labels(item))
        return result
    return set()


def _class_ids(info: dict[str, Any], wanted: str) -> list[int]:
    mapping = info.get("idToLabels") or info.get("idToSemantics") or info.get("id_to_labels") or {}
    result: list[int] = []
    if not isinstance(mapping, dict):
        return result
    for raw_identifier, annotation in mapping.items():
        try:
            identifier = int(raw_identifier)
        except (TypeError, ValueError):
            continue
        if wanted in _labels(annotation):
            result.append(identifier)
    return result


def _synthetic_thermal(
    rgb: Any,
    semantic: Any,
    front_mask: np.ndarray,
    perimeter_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create a deterministic aligned temperature proxy, not sensor radiometry."""

    semantic_array = np.asarray(semantic)
    while semantic_array.ndim > 2 and semantic_array.shape[-1] == 1:
        semantic_array = semantic_array[..., 0]
    if rgb is not None:
        rgb_array = np.asarray(rgb)[..., :3].astype(np.float32, copy=False)
        if np.issubdtype(np.asarray(rgb).dtype, np.integer):
            rgb_array /= float(np.iinfo(np.asarray(rgb).dtype).max)
        luminance = rgb_array[..., 0] * 0.2126 + rgb_array[..., 1] * 0.7152 + rgb_array[..., 2] * 0.0722
        kelvin = 289.15 + np.clip(luminance, 0.0, 1.0) * 13.0
    else:
        kelvin = np.full(semantic_array.shape[:2], 293.15, dtype=np.float32)
    front = np.asarray(front_mask).squeeze().astype(bool)
    perimeter = np.asarray(perimeter_mask).squeeze().astype(bool)
    kelvin = np.asarray(kelvin, dtype=np.float32)
    kelvin[perimeter] = np.maximum(kelvin[perimeter], 650.0)
    kelvin[front] = np.maximum(kelvin[front], 1100.0)
    scale_kelvin_per_dn = 0.02
    encoded = np.clip(np.rint(kelvin / scale_kelvin_per_dn), 0, 65535).astype(np.uint16)
    metadata = {
        "schema": "fireviewer.synthetic-thermal.v1",
        "product_type": "synthetic_aligned_temperature_proxy_not_radiometric_sensor_output",
        "units": "kelvin",
        "encoding": "uint16_png",
        "scale_kelvin_per_dn": scale_kelvin_per_dn,
        "offset_kelvin": 0.0,
        "ambient_temperature_range_kelvin": [289.15, 302.15],
        "fire_perimeter_temperature_kelvin": 650.0,
        "fire_front_temperature_kelvin": 1100.0,
        "emissivity_assumption": 0.95,
        "alignment": "pixel_aligned_with_rgb_and_semantic_ids",
        "source_modalities": ["rgb", "projected_fire_truth_masks"],
        "limitations": [
            "not_calibrated_to_a_physical_thermal_camera",
            "not_valid_for_radiometric_temperature_measurement",
            "camera_projected_fire_truth_geometry_drives_hot_regions",
        ],
    }
    return kelvin, encoded, metadata


class FireViewerWriter(Writer):
    """Write the FireViewer dataset contract from Replicator annotator output."""

    def __init__(
        self,
        *,
        output_dir: str,
        backend: Any | None = None,
        rgb: bool = True,
        semantic_segmentation: bool = True,
        instance_segmentation: bool = True,
        depth: bool = True,
        normals: bool = True,
        pointcloud: bool = True,
        camera_params: bool = True,
        **_: Any,
    ) -> None:
        self.data_structure = "renderProduct"
        self._backend = backend
        self._output_dir = Path(output_dir).resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_id = 0
        self._context: dict[str, Any] = {
            "state_id": "unassigned",
            "package_id": "unassigned",
            "source_package_id": "unassigned",
            "incident_id": "unassigned",
            "day_index": 0,
            "state_in_day": 0,
            "view_plan_by_camera": {},
        }
        self._render_product_camera_ids: dict[str, str] = {}
        self._camera_geolocation: dict[str, dict[str, Any]] = {}
        self.annotators = []
        for enabled, name in (
            (rgb, "rgb"),
            (semantic_segmentation, "semantic_segmentation"),
            (instance_segmentation, "instance_segmentation"),
            (depth, "distance_to_camera"),
            (normals, "normals"),
            (pointcloud, "pointcloud"),
            (camera_params, "camera_params"),
        ):
            if enabled:
                self.annotators.append(rep.annotators.get(name))

    def set_capture_context(
        self,
        *,
        package_id: str,
        source_package_id: str,
        incident_id: str,
        state_id: str,
        day_index: int,
        state_in_day: int,
        view_plan_by_camera: dict[str, dict[str, Any]],
    ) -> None:
        if not 1 <= len(view_plan_by_camera) <= 20:
            raise RuntimeError(
                f"{state_id} capture context must contain between one and twenty scheduled views"
            )
        self._context = {
            "package_id": package_id,
            "source_package_id": source_package_id,
            "incident_id": incident_id,
            "state_id": state_id,
            "day_index": int(day_index),
            "state_in_day": int(state_in_day),
            "view_plan_by_camera": view_plan_by_camera,
        }

    def register_render_product(self, render_product_path: str, camera_id: str, geolocation: dict[str, Any]) -> None:
        self._render_product_camera_ids[str(render_product_path)] = camera_id
        self._camera_geolocation[camera_id] = geolocation

    def _write_or_schedule(self, callback: Any, **kwargs: Any) -> None:
        if self._backend is not None:
            self._backend.schedule(callback, **kwargs)
        else:
            callback(**kwargs)

    def _camera_id(self, render_product: str) -> str:
        if render_product in self._render_product_camera_ids:
            return self._render_product_camera_ids[render_product]
        for key, value in self._render_product_camera_ids.items():
            if render_product.endswith(key) or key.endswith(render_product):
                return value
        return Path(render_product).name.replace("RenderProduct_", "") or "unknown_camera"

    def _frame_directory(self, camera_id: str) -> Path:
        return self._output_dir / str(self._context["package_id"]) / str(self._context["state_id"]) / camera_id / f"frame_{self._frame_id:06d}"

    def _write_render_product(self, render_product: str, payloads: dict[str, Any]) -> None:
        camera_id = self._camera_id(render_product)
        directory = self._frame_directory(camera_id)
        directory.mkdir(parents=True, exist_ok=True)
        rgb, _ = _payload(payloads.get("rgb"))
        semantic, semantic_info = _payload(payloads.get("semantic_segmentation"))
        instance, instance_info = _payload(payloads.get("instance_segmentation"))
        depth, _ = _payload(payloads.get("distance_to_camera"))
        normals, _ = _payload(payloads.get("normals"))
        points, points_info = _payload(payloads.get("pointcloud"))
        camera_params, camera_info = _payload(payloads.get("camera_params"))
        view_plan = dict(self._context.get("view_plan_by_camera", {}).get(camera_id, {}))
        if not view_plan:
            raise RuntimeError(f"Replicator returned unscheduled render product for {camera_id}")
        dense_target_arrays = view_plan.pop("_dense_target_arrays", None)
        dense_target_projection = view_plan.get("dense_target_projection")
        captured_at_utc = datetime.now(timezone.utc).isoformat()
        view_plan["captured_at_utc"] = captured_at_utc

        if rgb is not None:
            self._write_or_schedule(write_image, data=rgb, path=str(directory / "rgb.png"))
        if semantic is not None:
            semantic_array = np.asarray(semantic)
            self._write_or_schedule(write_array_npz, data=semantic_array, path=str(directory / "semantic_ids.npz"))
            self._write_or_schedule(_write_json, data=semantic_info, path=str(directory / "semantic_info.json"))
            front_ids = _class_ids(semantic_info, "fire_front")
            perimeter_ids = _class_ids(semantic_info, "fire_perimeter")
            smoke_ids = _class_ids(semantic_info, "smoke_source")
            burned_ids = _class_ids(semantic_info, "burned_ground")
            semantic_front_mask = np.isin(semantic_array, front_ids).astype(np.uint8) * 255
            semantic_perimeter_mask = np.isin(semantic_array, perimeter_ids).astype(np.uint8) * 255
            semantic_smoke_mask = np.isin(semantic_array, smoke_ids).astype(np.uint8) * 255
            semantic_burned_mask = np.isin(semantic_array, burned_ids).astype(np.uint8) * 255
            mapping_available = bool(semantic_info)
        else:
            semantic_array = None
            semantic_front_mask = None
            semantic_perimeter_mask = None
            semantic_smoke_mask = None
            semantic_burned_mask = None
            mapping_available = False

        if dense_target_arrays is not None:
            if not isinstance(dense_target_arrays, dict):
                raise RuntimeError(f"invalid projected dense target payload for {camera_id}")
            expected_shape = tuple(np.asarray(semantic_array).shape[:2]) if semantic_array is not None else None
            required_dense_keys = {
                "fire_front",
                "fire_perimeter",
                "smoke_source",
                "smoke",
                "burned_area",
            }
            if set(dense_target_arrays) != required_dense_keys:
                raise RuntimeError(
                    f"incomplete projected dense targets for {camera_id}: {sorted(dense_target_arrays)}"
                )
            projected_masks = {
                name: (np.asarray(value) != 0).astype(np.uint8) * 255
                for name, value in dense_target_arrays.items()
            }
            if expected_shape is not None and any(
                tuple(mask.shape) != expected_shape for mask in projected_masks.values()
            ):
                raise RuntimeError(
                    f"projected dense target shape mismatch for {camera_id}: "
                    f"{sorted({tuple(mask.shape) for mask in projected_masks.values()})} != {expected_shape}"
                )
            front_mask = projected_masks["fire_front"]
            perimeter_mask = projected_masks["fire_perimeter"]
            smoke_source_mask = projected_masks["smoke_source"]
            smoke_mask = projected_masks["smoke"]
            burned_mask = projected_masks["burned_area"]
            dense_target_source = "active_usd_truth_geometry_camera_projection"
            projection_metadata = dense_target_projection or {
                "schema": "fireviewer.projected-dense-targets.v1",
                "source_contract": dense_target_source,
            }
        elif semantic_array is not None:
            front_mask = semantic_front_mask
            perimeter_mask = semantic_perimeter_mask
            smoke_source_mask = semantic_smoke_mask
            smoke_mask = semantic_smoke_mask
            burned_mask = semantic_burned_mask
            dense_target_source = "replicator_semantic_ids_fallback"
            projection_metadata = {
                "schema": "fireviewer.semantic-derived-dense-targets.v1",
                "source_contract": dense_target_source,
                "semantic_class_ids": {
                    "fire_front": front_ids,
                    "fire_perimeter": perimeter_ids,
                    "smoke_source": smoke_ids,
                    "burned_ground": burned_ids,
                },
            }
        else:
            raise RuntimeError(f"no dense target source was available for {camera_id}")

        self._write_or_schedule(write_array_npz, data=front_mask, path=str(directory / "front_visible_mask.npz"))
        self._write_or_schedule(write_array_npz, data=front_mask, path=str(directory / "flame_mask.npz"))
        self._write_or_schedule(write_array_npz, data=perimeter_mask, path=str(directory / "perimeter_mask.npz"))
        self._write_or_schedule(write_array_npz, data=smoke_source_mask, path=str(directory / "smoke_source_mask.npz"))
        self._write_or_schedule(write_array_npz, data=smoke_mask, path=str(directory / "smoke_mask.npz"))
        self._write_or_schedule(write_array_npz, data=burned_mask, path=str(directory / "burned_area_mask.npz"))
        self._write_or_schedule(
            _write_json,
            data=projection_metadata,
            path=str(directory / "dense_target_projection.json"),
        )
        front_visible = bool(np.any(front_mask))
        perimeter_visible = bool(np.any(perimeter_mask))
        smoke_source_visible = bool(np.any(smoke_source_mask))
        smoke_visible = bool(np.any(smoke_mask))
        if instance is not None:
            self._write_or_schedule(write_array_npz, data=instance, path=str(directory / "instance_ids.npz"))
            self._write_or_schedule(
                _write_json,
                data=instance_info,
                path=str(directory / "instance_info.json"),
            )
        if depth is not None:
            depth_array = np.asarray(depth, dtype=np.float32)
            while depth_array.ndim > 2 and depth_array.shape[-1] == 1:
                depth_array = depth_array[..., 0]
            self._write_or_schedule(write_array_npz, data=depth_array, path=str(directory / "depth_distance_to_camera_m.npz"))
            valid_depth = np.isfinite(depth_array) & (depth_array > 0.0)
            valid_values = depth_array[valid_depth]
            if valid_values.size:
                preview_near = float(np.percentile(valid_values, 1.0))
                preview_far = float(np.percentile(valid_values, 99.0))
                if preview_far <= preview_near:
                    preview_far = preview_near + 1.0
                normalized_depth = np.zeros(depth_array.shape[:2], dtype=np.float32)
                normalized_depth[valid_depth] = 1.0 - np.clip(
                    (depth_array[valid_depth] - preview_near)
                    / (preview_far - preview_near),
                    0.0,
                    1.0,
                )
                depth_preview = np.rint(normalized_depth * 255.0).astype(np.uint8)
            else:
                preview_near = None
                preview_far = None
                depth_preview = np.zeros(depth_array.shape[:2], dtype=np.uint8)
            depth_preview = depth_preview[
                ::PREVIEW_DOWNSAMPLE_FACTOR,
                ::PREVIEW_DOWNSAMPLE_FACTOR,
            ]
            self._write_or_schedule(write_image, data=depth_preview, path=str(directory / "depth_preview.png"))
            self._write_or_schedule(
                _write_json,
                data={
                    "schema": "fireviewer.depth-pass.v1",
                    "annotator": "distance_to_camera",
                    "raw_file": "depth_distance_to_camera_m.npz",
                    "units": "metres",
                    "distance_definition": "euclidean_distance_from_camera",
                    "dtype": str(depth_array.dtype),
                    "shape": list(depth_array.shape),
                    "valid_pixel_count": int(valid_values.size),
                    "preview_file": "depth_preview.png",
                    "preview_resolution_px": [
                        int(depth_preview.shape[1]),
                        int(depth_preview.shape[0]),
                    ],
                    "preview_downsample_factor": PREVIEW_DOWNSAMPLE_FACTOR,
                    "preview_sampling": "stride_nearest_no_interpolation",
                    "preview_percentile_range_m": [preview_near, preview_far],
                    "preview_is_training_target": False,
                },
                path=str(directory / "depth_metadata.json"),
            )
        if normals is not None:
            normals_array = np.asarray(normals, dtype=np.float32)
            self._write_or_schedule(write_array_npz, data=normals_array, path=str(directory / "normals_replicator.npz"))
            normal_rgb = np.clip(
                (normals_array[..., :3] * 0.5 + 0.5) * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8)
            normal_preview = normal_rgb[
                ::PREVIEW_DOWNSAMPLE_FACTOR,
                ::PREVIEW_DOWNSAMPLE_FACTOR,
            ]
            self._write_or_schedule(write_image, data=normal_preview, path=str(directory / "normals_preview.png"))
            self._write_or_schedule(
                _write_json,
                data={
                    "schema": "fireviewer.normal-pass.v1",
                    "annotator": "normals",
                    "raw_file": "normals_replicator.npz",
                    "component_contract": "first_three_channels_are_xyz_in_minus_one_to_one",
                    "coordinate_space": "native_replicator_normals_buffer",
                    "dtype": str(normals_array.dtype),
                    "shape": list(normals_array.shape),
                    "preview_file": "normals_preview.png",
                    "preview_resolution_px": [
                        int(normal_preview.shape[1]),
                        int(normal_preview.shape[0]),
                    ],
                    "preview_downsample_factor": PREVIEW_DOWNSAMPLE_FACTOR,
                    "preview_sampling": "stride_nearest_no_interpolation",
                    "preview_mapping": "uint8_clamp((xyz*0.5+0.5)*255)",
                    "preview_is_training_target": False,
                },
                path=str(directory / "normals_metadata.json"),
            )
        if points is not None:
            points_array = np.asarray(points)
            if points_array.ndim != 2 or points_array.shape[1] < 3:
                raise RuntimeError(f"invalid pointcloud shape for {camera_id}: {points_array.shape}")
            point_attributes, pointcloud_metadata = compact_pointcloud_attributes(
                points_info,
                point_count=int(points_array.shape[0]),
            )
            self._write_or_schedule(
                write_array_npz,
                data=points_array,
                path=str(directory / "pointcloud.npz"),
            )
            self._write_or_schedule(
                write_named_arrays_npz,
                data=point_attributes,
                path=str(directory / "pointcloud_attributes.npz"),
            )
            self._write_or_schedule(
                _write_json,
                data=pointcloud_metadata,
                path=str(directory / "pointcloud_info.json"),
            )
        if camera_params is not None:
            self._write_or_schedule(_write_json, data={"camera_params": camera_params, "info": camera_info}, path=str(directory / "camera_params.json"))

        expected_fire_visible = bool(view_plan.get("expected_fire_visible"))
        sample_kind = str(view_plan.get("sample_kind"))
        abstention_reasons = []
        if not mapping_available:
            abstention_reasons.append("semantic_mapping_unavailable")
        if expected_fire_visible:
            if not front_visible:
                abstention_reasons.append("expected_fire_front_not_visible")
            if not perimeter_visible:
                abstention_reasons.append("expected_fire_perimeter_not_visible")
            label_consistent = front_visible and perimeter_visible
        else:
            if front_visible or perimeter_visible:
                abstention_reasons.append("negative_example_contains_fire")
            label_consistent = not front_visible and not perimeter_visible
        geolocation = self._camera_geolocation.get(camera_id, {"camera_id": camera_id, "status": "unresolved"})
        thermal_expected = bool(view_plan.get("thermal_expected"))
        if thermal_expected != bool(geolocation.get("thermal_capture", False)):
            abstention_reasons.append("thermal_schedule_camera_contract_mismatch")
        if thermal_expected:
            if semantic is None:
                abstention_reasons.append("thermal_semantic_source_unavailable")
            else:
                kelvin, encoded, thermal_metadata = _synthetic_thermal(rgb, semantic, front_mask, perimeter_mask)
                thermal_metadata.update(
                    {
                        "raw_file": "thermal_kelvin.npz",
                        "package_id": self._context["package_id"],
                        "source_package_id": self._context["source_package_id"],
                        "incident_id": self._context["incident_id"],
                        "state_id": self._context["state_id"],
                        "camera_id": camera_id,
                        "capture_id": view_plan["capture_id"],
                    }
                )
                self._write_or_schedule(write_array_npz, data=kelvin, path=str(directory / "thermal_kelvin.npz"))
                self._write_or_schedule(write_image, data=encoded, path=str(directory / "thermal_16bit.png"))
                self._write_or_schedule(_write_json, data=thermal_metadata, path=str(directory / "thermal_metadata.json"))
        self._write_or_schedule(_write_json, data=geolocation, path=str(directory / "geolocation.json"))
        self._write_or_schedule(_write_json, data=view_plan, path=str(directory / "capture-plan.json"))
        modality_files = (
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
            "thermal_kelvin.npz",
            "thermal_16bit.png",
            "thermal_metadata.json",
        )
        modality_sha256 = {
            name: _sha256_file(directory / name)
            for name in modality_files
            if (directory / name).is_file()
        }
        camera_intrinsics = {
            "fx_px": float(view_plan.get("focal_length_mm"))
            / float(view_plan.get("horizontal_aperture_mm"))
            * float(view_plan.get("image_resolution_px", [0, 0])[0]),
            "fy_px": float(view_plan.get("focal_length_mm"))
            / float(view_plan.get("vertical_aperture_mm"))
            * float(view_plan.get("image_resolution_px", [0, 0])[1]),
            "cx_px": float(view_plan.get("image_resolution_px", [0, 0])[0]) / 2.0,
            "cy_px": float(view_plan.get("image_resolution_px", [0, 0])[1]) / 2.0,
        }
        self._write_or_schedule(
            _write_json,
            data={
                "schema": "fireviewer.agentic-training-targets.v2",
                "capture_storage_profile": storage_profile_contract(),
                "dataset_id": self._context["package_id"],
                "package_id": self._context["package_id"],
                "source_package_id": self._context["source_package_id"],
                "source_stage_sha256": view_plan.get("source_stage_sha256"),
                "observation_id": view_plan["capture_id"],
                "case_id": view_plan.get("case_id"),
                "map_package_id": view_plan.get("map_package_id"),
                "incident_id": self._context["incident_id"],
                "scenario_id": view_plan.get("scenario_id"),
                "simulation_id": view_plan.get("simulation_id"),
                "state_id": self._context["state_id"],
                "capture_id": view_plan["capture_id"],
                "plan_id": view_plan.get("plan_id"),
                "camera_id": camera_id,
                "sample_kind": sample_kind,
                "captured_at_utc": captured_at_utc,
                "valid_at": view_plan.get("valid_at"),
                "timezone": view_plan.get("timezone"),
                "source_provenance_ids": view_plan.get("source_provenance_ids"),
                "simulation_time": {
                    "day_index": self._context["day_index"],
                    "state_in_day": self._context["state_in_day"],
                    "observation_elapsed_s": view_plan.get("observation_elapsed_s"),
                    "fire_state_elapsed_s": view_plan.get("fire_state_elapsed_s"),
                    "playback_elapsed_s": view_plan.get("playback_elapsed_s"),
                    "playback_seconds_per_day": view_plan.get("playback_seconds_per_day"),
                    "playback_seconds_per_state": view_plan.get("playback_seconds_per_state"),
                    "time_of_day_s": view_plan.get("simulation_time_of_day_s"),
                    "time_of_day_hhmmss": view_plan.get("simulation_time_of_day_hhmmss"),
                    "clock_contract": view_plan.get("simulation_clock_contract"),
                    "capture_timeline_time_s": view_plan.get("capture_timeline_time_s"),
                    "capture_sequence_in_state": view_plan.get("capture_sequence_in_state"),
                    "timeline_playing_during_transition": view_plan.get("timeline_playing_during_transition"),
                    "capture_trigger_contract": view_plan.get("capture_trigger_contract"),
                },
                "fire_state": {
                    "burned_area_m2": view_plan.get("burned_area_m2"),
                    "burned_tree_count": view_plan.get("burned_tree_count"),
                    "active_front_length_m": view_plan.get("active_front_length_m"),
                    "mean_front_spread_rate_m_s": view_plan.get("mean_front_spread_rate_m_s"),
                    "ignition_l93_m": view_plan.get("ignition_l93_m"),
                    "ignition_point_local_m": view_plan.get("ignition_point_local_m"),
                    "truth_scope": view_plan.get("truth_scope"),
                    "propagation_model": view_plan.get("propagation_model"),
                    "propagation_solver": view_plan.get("propagation_solver"),
                    "fuel_input": view_plan.get("fuel_input"),
                },
                "environment": {
                    "drivers": view_plan.get("environment_drivers"),
                    "domain_bounds_l93_m": view_plan.get("domain_bounds_l93_m"),
                    "weather_state_id": view_plan.get("weather_state_id"),
                    "flow_capture_state": view_plan.get("flow_capture_state"),
                    "sky_capture_state": view_plan.get("sky_capture_state"),
                },
                "zoom": {
                    "zoom_set_id": view_plan.get("zoom_set_id"),
                    "zoom_set_size": view_plan.get("zoom_set_size"),
                    "zoom_index": view_plan.get("zoom_index"),
                    "zoom_label": view_plan.get("zoom_label"),
                    "zoom_multiplier": view_plan.get("zoom_multiplier"),
                    "base_focal_length_mm": view_plan.get("base_focal_length_mm"),
                    "base_focal_length_35mm_equivalent_mm": view_plan.get("base_focal_length_35mm_equivalent_mm"),
                    "camera_pose_contract": view_plan.get("camera_pose_contract"),
                    "runtime_contract": view_plan.get("zoom_runtime_contract"),
                },
                "camera": {
                    "position_local_m": view_plan.get("camera_position_local_m"),
                    "position_l93_ngf_ign69_m": view_plan.get("camera_position_l93_ngf_ign69_m"),
                    "aim_local_m": view_plan.get("camera_aim_local_m"),
                    "orientation_quat_wxyz": view_plan.get("camera_orientation_quat_wxyz"),
                    "orientation_yaw_pitch_roll_degrees": view_plan.get("camera_orientation_yaw_pitch_roll_degrees"),
                    "forward_local": view_plan.get("camera_forward_local"),
                    "image_resolution_px": view_plan.get("image_resolution_px"),
                    "focal_length_mm": view_plan.get("focal_length_mm"),
                    "focal_length_35mm_equivalent_mm": view_plan.get("focal_length_35mm_equivalent_mm"),
                    "horizontal_aperture_mm": view_plan.get("horizontal_aperture_mm"),
                    "vertical_aperture_mm": view_plan.get("vertical_aperture_mm"),
                    "horizontal_fov_degrees": view_plan.get("horizontal_fov_degrees"),
                    "vertical_fov_degrees": view_plan.get("vertical_fov_degrees"),
                    "capture_device_profile": view_plan.get("capture_device_profile"),
                    "framing_style": view_plan.get("framing_style"),
                    "camera_role": view_plan.get("camera_role"),
                    "intrinsics": camera_intrinsics,
                    "placement_type": view_plan.get("placement_type"),
                    "access_surface": view_plan.get("access_surface"),
                    "host_building": view_plan.get("host_building"),
                },
                "nearest_flame": {
                    "point_index": view_plan.get("nearest_flame_point_index"),
                    "point_local_m": view_plan.get("nearest_flame_point_local_m"),
                    "point_l93_ngf_ign69_m": view_plan.get("nearest_flame_point_l93_ngf_ign69_m"),
                    "distance_m": view_plan.get("nearest_flame_distance_m"),
                    "projection": view_plan.get("nearest_flame_projection"),
                    "source_point_count": view_plan.get("fire_front_point_count"),
                },
                "nearest_smoke": {
                    "point_index": view_plan.get("nearest_smoke_point_index"),
                    "point_local_m": view_plan.get("nearest_smoke_point_local_m"),
                    "point_l93_ngf_ign69_m": view_plan.get("nearest_smoke_point_l93_ngf_ign69_m"),
                    "distance_m": view_plan.get("nearest_smoke_distance_m"),
                    "projection": view_plan.get("nearest_smoke_projection"),
                    "source_point_count": view_plan.get("smoke_source_point_count"),
                },
                "visible_flame_points_local_m": view_plan.get("visible_flame_points_local_m"),
                "smoke_source_points_local_m": view_plan.get("smoke_source_points_local_m"),
                "active_front_local_m": view_plan.get("active_front_local_m"),
                "dense_targets": {
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
                },
                "visibility": {
                    "expected_fire_visible": expected_fire_visible,
                    "fire_front_visible": front_visible,
                    "fire_perimeter_visible": perimeter_visible,
                    "smoke_source_visible": smoke_source_visible,
                    "smoke_visible": smoke_visible,
                    "acceptance": view_plan.get("visibility_acceptance"),
                },
                "dense_target_source": dense_target_source,
                "expected_modalities": view_plan.get("expected_modalities"),
                "expected_fire_in_frame": expected_fire_visible,
                "line_of_sight_receipt": view_plan.get("line_of_sight_receipt"),
                "negative_reason": view_plan.get("negative_reason"),
                "weather_state_id": view_plan.get("weather_state_id"),
                "modality_sha256": modality_sha256,
                "dependency_sha256": view_plan.get("dependency_sha256"),
                "split_group": {
                    "base_map_package_id": view_plan.get("map_package_id"),
                    "incident_id": self._context["incident_id"],
                    "scenario_id": view_plan.get("scenario_id"),
                },
                "targeting_contract": view_plan.get("dynamic_targeting_contract"),
                "targeting_mode": view_plan.get("dynamic_targeting_mode"),
            },
            path=str(directory / "training-targets.json"),
        )
        self._write_or_schedule(
            _write_json,
            data={
                "dataset_id": self._context["package_id"],
                "package_id": self._context["package_id"],
                "source_package_id": self._context["source_package_id"],
                "incident_id": self._context["incident_id"],
                "state_id": self._context["state_id"],
                "day_index": self._context["day_index"],
                "state_in_day": self._context["state_in_day"],
                "camera_id": camera_id,
                "capture_id": view_plan["capture_id"],
                "plan_id": view_plan.get("plan_id"),
                "zoom_set_id": view_plan.get("zoom_set_id"),
                "zoom_index": view_plan.get("zoom_index"),
                "zoom_multiplier": view_plan.get("zoom_multiplier"),
                "frame_id": self._frame_id,
                "abstain": bool(abstention_reasons),
                "reasons": abstention_reasons,
                "sample_kind": sample_kind,
                "expected_fire_visible": expected_fire_visible,
                "front_visible": front_visible,
                "perimeter_visible": perimeter_visible,
                "label_consistent": label_consistent,
                "thermal_expected": thermal_expected,
                "smoke_source_visible": smoke_source_visible,
                "smoke_visible": smoke_visible,
                "dense_target_source": dense_target_source,
                "captured_at_utc": captured_at_utc,
                "dynamic_targeting_contract": view_plan.get("dynamic_targeting_contract"),
                "visibility_acceptance": view_plan.get("visibility_acceptance"),
            },
            path=str(directory / "abstention.json"),
        )

    def write(self, data: dict[str, Any]) -> None:
        render_products = data.get("renderProducts", data)
        if not isinstance(render_products, dict):
            raise RuntimeError("Replicator writer did not provide render-product structured data")
        for render_product, payloads in render_products.items():
            if isinstance(payloads, dict):
                self._write_render_product(str(render_product), payloads)
        self._frame_id += 1


def register() -> None:
    """Register the writer once in the active Replicator registry."""

    if "FireViewerWriter" not in rep.WriterRegistry.get_writers():
        rep.WriterRegistry.register(FireViewerWriter)
