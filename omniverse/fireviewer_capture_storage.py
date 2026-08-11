"""Lossless compact storage helpers for FireViewer capture modalities."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


CAPTURE_STORAGE_PROFILE_ID = "lossless_npz_lzma6_pointcloud_attributes_v2"
CAPTURE_STORAGE_PROFILE = {
    "schema": "fireviewer.capture-storage-profile.v1",
    "profile_id": CAPTURE_STORAGE_PROFILE_ID,
    "array_container": "numpy_npz",
    "array_compression": "zip_lzma",
    "array_compression_level": 6,
    "array_key": "data",
    "loss_contract": "lossless_for_authored_numpy_arrays",
    "pointcloud_attribute_contract": (
        "replicator_point_attributes_preserved_in_named_npz_arrays_with_explicit_dtypes"
    ),
    "pointcloud_attribute_dtypes": {
        "pointInstance": "uint32",
        "pointNormals": "float32",
        "pointRgb": "uint8",
        "pointSemantic": "uint32",
    },
    "pointcloud_info_contract": "small_schema_metadata_only_no_per_point_json_arrays",
    "preview_contract": "non_training_stride4_preview_raw_arrays_remain_full_resolution",
}


def storage_profile_contract() -> dict[str, Any]:
    profile = json.loads(json.dumps(CAPTURE_STORAGE_PROFILE))
    encoded = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    profile["profile_sha256"] = hashlib.sha256(encoded).hexdigest()
    return profile


def _write_npz_arrays(*, arrays: dict[str, Any], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_LZMA,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name, value in arrays.items():
                buffer = BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.asarray(value),
                    allow_pickle=False,
                )
                archive.writestr(f"{name}.npy", buffer.getvalue())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_array_npz(*, data: Any, path: str) -> None:
    _write_npz_arrays(arrays={"data": data}, path=path)


def load_array(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if loaded.files != ["data"]:
                raise ValueError(
                    f"single-array NPZ must contain only 'data': {path}"
                )
            return np.asarray(loaded["data"])
        finally:
            loaded.close()
    return np.asarray(loaded)


def _checked_integer_array(value: Any, *, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return array.astype(dtype, copy=False)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError(f"{name} must contain finite integer values")
    minimum = int(np.min(array))
    maximum = int(np.max(array))
    bounds = np.iinfo(dtype)
    if minimum < int(bounds.min) or maximum > int(bounds.max):
        raise ValueError(f"{name} values exceed {dtype} range")
    return array.astype(dtype, copy=False)


def compact_pointcloud_attributes(
    points_info: dict[str, Any],
    *,
    point_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {"pointInstance", "pointNormals", "pointRgb", "pointSemantic"}
    missing = sorted(required - set(points_info))
    if missing:
        raise ValueError("missing pointcloud attributes: " + ", ".join(missing))

    point_instance = _checked_integer_array(
        points_info["pointInstance"],
        name="pointInstance",
        dtype=np.dtype(np.uint32),
    )
    point_semantic = _checked_integer_array(
        points_info["pointSemantic"],
        name="pointSemantic",
        dtype=np.dtype(np.uint32),
    )
    point_rgb = _checked_integer_array(
        points_info["pointRgb"],
        name="pointRgb",
        dtype=np.dtype(np.uint8),
    )
    source_normals = np.asarray(points_info["pointNormals"])
    if not np.issubdtype(source_normals.dtype, np.number):
        raise ValueError("pointNormals must be numeric")
    point_normals = source_normals.astype(np.float32, copy=False)
    if not np.all(np.isfinite(point_normals)):
        raise ValueError("pointNormals must contain finite values")
    if not np.array_equal(
        point_normals.astype(source_normals.dtype, copy=False),
        source_normals,
    ):
        raise ValueError("pointNormals cannot be represented losslessly as float32")
    arrays = {
        "pointInstance": point_instance,
        "pointNormals": point_normals,
        "pointRgb": point_rgb,
        "pointSemantic": point_semantic,
    }
    for name, array in arrays.items():
        if array.ndim < 1 or int(array.shape[0]) != int(point_count):
            raise ValueError(
                f"{name} point count mismatch: {array.shape} != {point_count}"
            )
    if point_normals.ndim != 2 or point_normals.shape[1] < 3:
        raise ValueError(f"invalid pointNormals shape: {point_normals.shape}")
    if point_rgb.ndim != 2 or point_rgb.shape[1] < 3:
        raise ValueError(f"invalid pointRgb shape: {point_rgb.shape}")

    extra_info: dict[str, Any] = {}
    for name, value in points_info.items():
        if name in required:
            continue
        if isinstance(value, np.ndarray) or (
            isinstance(value, (list, tuple)) and len(value) > 64
        ):
            raise ValueError(
                f"unexpected bulk pointcloud attribute must not enter JSON: {name}"
            )
        if isinstance(value, np.generic):
            value = value.item()
        extra_info[str(name)] = value

    metadata = {
        "schema": "fireviewer.pointcloud-pass.v2",
        "annotator": "pointcloud",
        "points_file": "pointcloud.npz",
        "attributes_file": "pointcloud_attributes.npz",
        "point_count": int(point_count),
        "compression": "numpy_npz_zip_lzma_level_6",
        "attribute_arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "minimum": (
                    float(np.min(array)) if array.size and np.issubdtype(array.dtype, np.floating)
                    else int(np.min(array)) if array.size else None
                ),
                "maximum": (
                    float(np.max(array)) if array.size and np.issubdtype(array.dtype, np.floating)
                    else int(np.max(array)) if array.size else None
                ),
            }
            for name, array in arrays.items()
        },
        "extra_info": extra_info,
        "storage_profile": storage_profile_contract(),
    }
    return arrays, metadata


def write_named_arrays_npz(*, data: dict[str, Any], path: str) -> None:
    _write_npz_arrays(arrays=data, path=path)


def load_named_arrays_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def validate_pointcloud_storage(
    *,
    points: np.ndarray,
    attributes: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] < 1:
        errors.append("invalid_pointcloud")
        return errors
    if metadata.get("schema") != "fireviewer.pointcloud-pass.v2":
        errors.append("invalid_pointcloud_metadata_schema")
    if metadata.get("points_file") != "pointcloud.npz":
        errors.append("invalid_pointcloud_points_file")
    if metadata.get("attributes_file") != "pointcloud_attributes.npz":
        errors.append("invalid_pointcloud_attributes_file")
    if int(metadata.get("point_count", -1)) != int(points.shape[0]):
        errors.append("pointcloud_metadata_count_mismatch")
    expected_dtypes = CAPTURE_STORAGE_PROFILE["pointcloud_attribute_dtypes"]
    if set(attributes) != set(expected_dtypes):
        errors.append("invalid_pointcloud_attribute_keys")
        return errors
    described = metadata.get("attribute_arrays")
    if not isinstance(described, dict):
        errors.append("missing_pointcloud_attribute_metadata")
        described = {}
    for name, dtype in expected_dtypes.items():
        array = attributes[name]
        if array.ndim < 1 or int(array.shape[0]) != int(points.shape[0]):
            errors.append(f"pointcloud_attribute_count_mismatch:{name}")
        if str(array.dtype) != dtype:
            errors.append(f"pointcloud_attribute_dtype_mismatch:{name}")
        description = described.get(name) if isinstance(described, dict) else None
        if not isinstance(description, dict):
            errors.append(f"missing_pointcloud_attribute_description:{name}")
        elif description.get("shape") != list(array.shape) or description.get("dtype") != str(array.dtype):
            errors.append(f"pointcloud_attribute_description_mismatch:{name}")
    profile = metadata.get("storage_profile") or {}
    expected_profile = storage_profile_contract()
    if profile.get("profile_id") != CAPTURE_STORAGE_PROFILE_ID:
        errors.append("invalid_capture_storage_profile")
    if profile.get("profile_sha256") != expected_profile["profile_sha256"]:
        errors.append("capture_storage_profile_hash_mismatch")
    return errors
