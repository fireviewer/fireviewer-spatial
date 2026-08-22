from __future__ import annotations

import numpy as np
import pytest

import mns_mnt_placement_inventory as v1
from mns_mnt_placement_inventory_v2 import (
    NativeGridMisalignmentError,
    TREE_ASSET_SELECTION_POLICY,
    build_placement_inventory_v2,
)


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


def test_systemically_misaligned_native_pair_fails_closed() -> None:
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
    assert baseline.inventory["trees"]["valid_count"] == 1
    with pytest.raises(NativeGridMisalignmentError, match="systemic"):
        build_placement_inventory_v2(
            mnt,
            mns,
            tile_origin_l93_m=ORIGIN,
            zone_id="GPS-NATIVE-FALLBACK",
            context_masks={"vegetation": vegetation},
            native_mnt_05m=native_mnt,
            native_mns_05m=native_mns,
        )


def test_36_bounded_negative_cells_do_not_erase_a_measured_tile() -> None:
    mnt, mns, vegetation = _canonical_tree_pair()
    for index in range(36):
        mns[index // 6, index % 6] = mnt[index // 6, index % 6] - 0.75

    with pytest.raises(v1.PlacementInventoryError, match="too many samples"):
        v1.build_placement_inventory(
            mnt,
            mns,
            tile_origin_l93_m=ORIGIN,
            zone_id="GPS-LEGACY-CUTOFF",
            context_masks={"vegetation": vegetation},
        )
    factual = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-V2-CLAMP",
        context_masks={"vegetation": vegetation},
    )

    assert factual.inventory["trees"]["valid_count"] == 1
    assert factual.inventory["hag"]["negative_outlier_below_tolerance_count"] == 36
    assert factual.inventory["hag"]["negative_outlier_policy"].startswith("clamp_all")


def test_sparse_severe_canonical_negatives_are_clamped_but_systemic_offset_fails() -> None:
    mnt, mns, vegetation = _canonical_tree_pair()
    for index in range(52):
        mns[index // 13, index % 13] = mnt[index // 13, index % 13] - 2.0

    factual = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-V2-SPARSE-SEVERE-CLAMP",
        context_masks={"vegetation": vegetation},
    )

    assert factual.inventory["trees"]["valid_count"] == 1
    assert factual.inventory["hag"]["minimum_source_delta_mm"] == -2_000
    assert factual.inventory["hag"]["severe_negative_below_100cm_count"] == 52

    systemic_mns = np.full(SHAPE_1M, 98.0, dtype="float64")
    with pytest.raises(v1.PlacementInventoryError, match="systemic negative offset"):
        build_placement_inventory_v2(
            mnt,
            systemic_mns,
            tile_origin_l93_m=ORIGIN,
            zone_id="GPS-V2-SYSTEMIC-CANONICAL-OFFSET",
            context_masks={"vegetation": vegetation},
        )


def test_native_support_uses_highest_ground_inside_original_peak_cell() -> None:
    mnt, mns, vegetation = _canonical_tree_pair()
    baseline = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-NATIVE-SUPPORT",
        context_masks={"vegetation": vegetation},
    )
    peak_x, peak_y = baseline.inventory["trees"]["candidates"][0]["peak_cell_l93"]
    grid_west = ORIGIN[0] - v1.HALO_M
    grid_north = ORIGIN[1] + v1.TILE_SIZE_M + v1.HALO_M
    column0 = round((peak_x - grid_west) / 0.5)
    row0 = round((grid_north - (peak_y + 1)) / 0.5)
    native_mnt = np.full(SHAPE_05M, 100.0, dtype="float64")
    native_mns = native_mnt.copy()
    native_mns[row0, column0] = 110.0
    native_mnt[row0 + 1, column0 + 1] = 101.25
    native_mns[row0 + 1, column0 + 1] = 101.25

    refined = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-NATIVE-SUPPORT",
        context_masks={"vegetation": vegetation},
        native_mnt_05m=native_mnt,
        native_mns_05m=native_mns,
    )
    candidate = refined.inventory["trees"]["candidates"][0]
    assert candidate["ground_elevation_mm"] == 100_000
    assert candidate["support_elevation_mm"] == 100_150
    assert candidate["height_cm"] == 1_000
    assert candidate["asset_selection_policy"] == TREE_ASSET_SELECTION_POLICY


def test_tree_semantics_prefer_current_composition_then_forest_inventory() -> None:
    mnt, mns, vegetation = _canonical_tree_pair()
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [ORIGIN[0], ORIGIN[1]],
            [ORIGIN[0] + 500, ORIGIN[1]],
            [ORIGIN[0] + 500, ORIGIN[1] + 500],
            [ORIGIN[0], ORIGIN[1] + 500],
            [ORIGIN[0], ORIGIN[1]],
        ]],
    }
    features = {
        "vegetation": [{
            "source_id": "bdtopo-conifer",
            "geometry": polygon,
            "properties": {"nature": "Forêt fermée de conifères"},
        }],
        "forest_composition": [{
            "source_id": "bdforet-oak",
            "geometry": polygon,
            "properties": {"libelle2": "FUTAIE DE CHENES DECIDUS PURS"},
        }],
    }
    current = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-TREE-SEMANTICS",
        context_masks={"vegetation": vegetation},
        context_features=features,
    )
    candidate = current.inventory["trees"]["candidates"][0]
    assert candidate["source_properties"]["nature"] == "Forêt fermée de conifères"
    assert "forest_libelle2" not in candidate["source_properties"]

    features["vegetation"][0]["properties"]["nature"] = "Bois"
    historical = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-TREE-SEMANTICS",
        context_masks={"vegetation": vegetation},
        context_features=features,
    )
    candidate = historical.inventory["trees"]["candidates"][0]
    assert candidate["source_properties"]["forest_libelle2"] == (
        "FUTAIE DE CHENES DECIDUS PURS"
    )
    assert historical.inventory["trees"]["valid_count"] == (
        current.inventory["trees"]["valid_count"]
    )


def test_valid_building_footprint_excludes_tree_candidates_inside_it() -> None:
    mnt, mns, vegetation = _canonical_tree_pair()
    footprint = {
        "source_id": "building-over-tree-peak",
        "properties": {"nature": "Bâtiment indifférencié"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [ORIGIN[0] + 240, ORIGIN[1] + 240],
                [ORIGIN[0] + 260, ORIGIN[1] + 240],
                [ORIGIN[0] + 260, ORIGIN[1] + 260],
                [ORIGIN[0] + 240, ORIGIN[1] + 260],
                [ORIGIN[0] + 240, ORIGIN[1] + 240],
            ]],
        },
    }

    result = build_placement_inventory_v2(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="GPS-BUILDING-TREE-EXCLUSION",
        building_footprints=[footprint],
        context_masks={"vegetation": vegetation},
    )

    assert result.inventory["buildings"]["valid_count"] == 1
    assert result.inventory["trees"]["valid_count"] == 0
