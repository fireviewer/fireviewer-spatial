from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fireviewer_capture_storage import (
    compact_pointcloud_attributes,
    load_array,
    load_named_arrays_npz,
    storage_profile_contract,
    validate_pointcloud_storage,
    write_array_npz,
    write_named_arrays_npz,
)


def point_info(point_count: int = 4) -> dict[str, object]:
    return {
        "pointInstance": np.arange(point_count, dtype=np.uint32),
        "pointNormals": np.tile(
            np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            (point_count, 1),
        ),
        "pointRgb": np.tile(
            np.asarray([[10, 20, 30, 255]], dtype=np.uint8),
            (point_count, 1),
        ),
        "pointSemantic": np.arange(point_count, dtype=np.uint32) + 3,
    }


def test_single_array_npz_roundtrip_is_exact(tmp_path: Path) -> None:
    source = np.asarray([[0.25, -1.5], [np.inf, np.nan]], dtype=np.float32)
    output = tmp_path / "array.npz"

    write_array_npz(data=source, path=str(output))
    restored = load_array(output)

    assert restored.dtype == source.dtype
    assert restored.shape == source.shape
    assert np.array_equal(restored, source, equal_nan=True)


def test_pointcloud_attributes_roundtrip_with_typed_lossless_contract(
    tmp_path: Path,
) -> None:
    attributes, metadata = compact_pointcloud_attributes(
        point_info(),
        point_count=4,
    )
    output = tmp_path / "pointcloud_attributes.npz"

    write_named_arrays_npz(data=attributes, path=str(output))
    restored = load_named_arrays_npz(output)

    assert set(restored) == {
        "pointInstance",
        "pointNormals",
        "pointRgb",
        "pointSemantic",
    }
    assert restored["pointInstance"].dtype == np.dtype(np.uint32)
    assert restored["pointNormals"].dtype == np.dtype(np.float32)
    assert restored["pointRgb"].dtype == np.dtype(np.uint8)
    assert restored["pointSemantic"].dtype == np.dtype(np.uint32)
    assert all(np.array_equal(restored[name], attributes[name]) for name in restored)
    assert metadata["point_count"] == 4
    assert metadata["storage_profile"] == storage_profile_contract()


def test_bulk_unknown_pointcloud_attribute_is_rejected() -> None:
    info = point_info()
    info["unexpectedBulk"] = np.zeros((4, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="unexpected bulk pointcloud attribute"):
        compact_pointcloud_attributes(info, point_count=4)


def test_lossy_point_normal_conversion_is_rejected() -> None:
    info = point_info()
    info["pointNormals"] = np.asarray(
        [[0.123456789012345, 0.0, 1.0, 0.0]] * 4,
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="losslessly as float32"):
        compact_pointcloud_attributes(info, point_count=4)


def test_pointcloud_storage_validation_detects_count_and_dtype_drift() -> None:
    attributes, metadata = compact_pointcloud_attributes(
        point_info(),
        point_count=4,
    )
    points = np.zeros((4, 3), dtype=np.float32)

    assert validate_pointcloud_storage(
        points=points,
        attributes=attributes,
        metadata=metadata,
    ) == []

    attributes["pointSemantic"] = attributes["pointSemantic"].astype(np.uint16)
    errors = validate_pointcloud_storage(
        points=points,
        attributes=attributes,
        metadata=metadata,
    )
    assert "pointcloud_attribute_dtype_mismatch:pointSemantic" in errors
