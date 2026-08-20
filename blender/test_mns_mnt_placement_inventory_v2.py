from __future__ import annotations

import numpy as np

from mns_mnt_placement_inventory_v2 import build_placement_inventory_v2


ORIGIN = (700_000, 6_600_000)
SHAPE_1M = (520, 520)
SHAPE_05M = (1040, 1040)


def _canonical_tree_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mnt = np.full(SHAPE_1M, 100.0, dtype="float64")
    mns = mnt.copy()
    vegetation = np.zeros(SHAPE_1M, dtype=bool)
    center_row = 260
    center_column = 260
    for row in range(center_row - 2, center_row + 3):
        for column in range(center_column - 2, center_column + 3):
            distance = max(abs(row - center_row), abs(column - center_column))
            mns[row, column] += 10.0 - distance * 2.0
            vegetation[row, column] = True
    return mnt, mns, vegetation


def test_misaligned_optional_native_pair_keeps_canonical_tree_candidates() -> None:
    mnt, mns, vegetation = _canonical_tree_pair()
    baseline = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-NATIVE-FALLBACK",
        context_masks={"vegetation": vegetation},
    )
    native_mnt = np.full(SHAPE_05M, 100.0, dtype="float64")
    native_mns = np.full(SHAPE_05M, 98.0, dtype="float64")
    rejected = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-NATIVE-FALLBACK",
        context_masks={"vegetation": vegetation},
        native_mnt_05m=native_mnt,
        native_mns_05m=native_mns,
    )

    baseline_trees = baseline.inventory["trees"]
    rejected_trees = rejected.inventory["trees"]
    assert rejected_trees["candidates"] == baseline_trees["candidates"]
    assert rejected_trees["native_05m_refinement_count"] == 0
    assert rejected_trees["native_05m_refinement_resolution_m"] is None
    assert rejected_trees["native_05m_source_resolution_m"] == 0.5
    assert (
        rejected_trees["native_05m_refinement_status"]
        == "rejected_misaligned_below_mnt"
    )
    assert rejected.inventory["build_id"] != baseline.inventory["build_id"]
