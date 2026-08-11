from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compact_hag import (
    NODATA,
    quantize_hag_max_cm,
    quantize_hag_max_cm_from_canonical_mm,
    read_hag_max_2m,
    write_hag_max_2m,
)


def test_quantizes_vertex_hag_to_exact_2m_cell_maxima() -> None:
    mnt = np.full((251, 251), 100.0)
    mns = mnt.copy()
    mns[10, 20] += 3.456
    mns[50, 50] -= 2.0

    result = quantize_hag_max_cm(mnt, mns)

    assert result.shape == (250, 250)
    assert result.dtype == np.uint16
    assert result[9:11, 19:21].tolist() == [[346, 346], [346, 346]]
    assert result[49, 49] == 0


def test_quantizes_retained_canonical_halos_without_float_roundtrip() -> None:
    mnt = np.full((253, 253), 100_000, dtype="int32")
    mns = mnt.copy()
    mns[11, 21] += 3_455

    result = quantize_hag_max_cm_from_canonical_mm(mnt, mns)

    assert result.shape == (250, 250)
    assert result.dtype == np.uint16
    assert result[9:11, 19:21].tolist() == [[346, 346], [346, 346]]


def test_geotiff_is_reproducible_aligned_and_roundtrips(
    tmp_path: Path,
) -> None:
    values = np.zeros((250, 250), dtype="uint16")
    values[20:80, 30:90] = 1_250
    first = tmp_path / "first" / "hag-max-2m.tif"
    second = tmp_path / "second" / "hag-max-2m.tif"

    first_hash = write_hag_max_2m(
        first, values, tile_origin_l93_m=(700_000.0, 6_300_000.0)
    )
    second_hash = write_hag_max_2m(
        second, values, tile_origin_l93_m=(700_000.0, 6_300_000.0)
    )
    restored, metadata = read_hag_max_2m(first)

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    assert np.array_equal(restored, values)
    assert metadata["bounds_l93_m"] == [
        700_000.0,
        6_300_000.0,
        700_500.0,
        6_300_500.0,
    ]
    assert metadata["nodata"] == NODATA


def test_rejects_nodata_collision_and_misaligned_origin(tmp_path: Path) -> None:
    values = np.zeros((250, 250), dtype="uint16")
    values[0, 0] = NODATA
    with pytest.raises(ValueError, match="nodata"):
        write_hag_max_2m(
            tmp_path / "bad.tif",
            values,
            tile_origin_l93_m=(700_000.0, 6_300_000.0),
        )
    with pytest.raises(ValueError, match="align"):
        write_hag_max_2m(
            tmp_path / "bad-origin.tif",
            np.zeros((250, 250), dtype="uint16"),
            tile_origin_l93_m=(700_001.0, 6_300_000.0),
        )
