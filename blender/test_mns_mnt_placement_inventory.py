from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, box, mapping

import mns_mnt_placement_inventory as inventory
from mns_mnt_placement_inventory import (
    CONTRACT_SCHEMA,
    GeoPackageUnavailableError,
    PlacementInventoryError,
    assert_d_storage_path,
    build_placement_inventory,
    canonical_json_bytes,
    gpkg_supported,
    merge_tree_inventories,
    main,
    read_hag_1m,
    validate_inventory,
    write_hag_1m,
    write_inventory_gpkg,
    write_inventory_json,
)


ORIGIN = (700_000, 6_600_000)
SHAPE = (520, 520)


def _base_pair() -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices(SHAPE)
    mnt = 100.0 + columns * 0.001 + rows * 0.0005
    return mnt, mnt.copy()


def _index(origin: tuple[int, int], global_x: int, global_y: int) -> tuple[int, int]:
    west, south = origin
    column = global_x - (west - 10)
    row = south + 510 - global_y - 1
    return row, column


def _set_height_cell(
    mnt: np.ndarray,
    mns: np.ndarray,
    origin: tuple[int, int],
    global_x: int,
    global_y: int,
    height_m: float,
) -> None:
    row, column = _index(origin, global_x, global_y)
    mns[row, column] = mnt[row, column] + height_m


def _set_crown(
    mnt: np.ndarray,
    mns: np.ndarray,
    origin: tuple[int, int],
    peak_x: int,
    peak_y: int,
    radius: int = 2,
) -> None:
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            distance = max(abs(delta_x), abs(delta_y))
            height = 10.0 - distance * 2.0
            _set_height_cell(
                mnt,
                mns,
                origin,
                peak_x + delta_x,
                peak_y + delta_y,
                height,
            )


def _vegetation_mask(
    origin: tuple[int, int], cells: list[tuple[int, int]]
) -> np.ndarray:
    result = np.zeros(SHAPE, dtype=bool)
    for global_x, global_y in cells:
        result[_index(origin, global_x, global_y)] = True
    return result


def _crown_cells(peak_x: int, peak_y: int, radius: int = 2) -> list[tuple[int, int]]:
    return [
        (peak_x + delta_x, peak_y + delta_y)
        for delta_y in range(-radius, radius + 1)
        for delta_x in range(-radius, radius + 1)
    ]


def test_build_is_bit_deterministic_and_hag_is_exact_uint16() -> None:
    mnt, mns = _base_pair()
    _set_crown(mnt, mns, ORIGIN, 700_100, 6_600_100)
    vegetation = _vegetation_mask(ORIGIN, _crown_cells(700_100, 6_600_100))
    footprints = [
        {
            "source_id": "BAT-001",
            "geometry": mapping(box(700_020.0, 6_600_020.0, 700_025.0, 6_600_025.0)),
            "properties": {"usage_1": "Commercial", "nature": "Supermarché"},
        }
    ]
    for global_y in range(6_600_020, 6_600_025):
        for global_x in range(700_020, 700_025):
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 7.345)

    first = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        building_footprints=footprints,
        context_masks={"vegetation": vegetation},
    )
    second = build_placement_inventory(
        mnt.copy(),
        mns.copy(),
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        building_footprints=copy.deepcopy(footprints),
        context_masks={"vegetation": vegetation.copy()},
    )

    assert first.hag_core_cm.shape == (500, 500)
    assert first.hag_core_cm.dtype == np.uint16
    assert int(first.hag_core_cm.max()) == 1_000
    building = next(
        candidate
        for candidate in first.inventory["buildings"]["candidates"]
        if candidate["confirmed_source_id"] == "BAT-001"
    )
    assert building["height_cm"] == 735
    assert building["status"] == "valid"
    assert building["source_properties"] == {
        "nature": "Supermarché",
        "usage_1": "Commercial",
    }
    assert building["footprint_geojson"]["type"] == "Polygon"
    assert np.array_equal(first.hag_core_cm, second.hag_core_cm)
    assert canonical_json_bytes(first.inventory) == canonical_json_bytes(
        second.inventory
    )


def test_stable_sig_features_create_grounded_context_asset_candidates() -> None:
    mnt, mns = _base_pair()
    road = {
        "source_id": "ROAD-001",
        "geometry": mapping(LineString([(700_020, 6_600_100), (700_220, 6_600_100)])),
        "properties": {"nature": "Route à une chaussée"},
    }
    rail = {
        "source_id": "RAIL-001",
        "geometry": mapping(LineString([(700_030, 6_600_200), (700_230, 6_600_200)])),
        "properties": {"nature": "Voie ferrée principale"},
    }
    water = {
        "source_id": "WATER-001",
        "geometry": mapping(LineString([(700_120, 6_600_050), (700_120, 6_600_250)])),
        "properties": {"nature": "Ruisseau"},
    }
    features = {
        "roads": [road],
        "rail": [rail],
        "hydro_lines": [water],
    }

    first = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-context",
        context_features=features,
    )
    second = build_placement_inventory(
        mnt.copy(),
        mns.copy(),
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-context",
        context_features=copy.deepcopy(features),
    )

    candidates = first.inventory["context_assets"]["candidates"]
    assert first.inventory["context_assets"]["source_count"] == 4
    assert {candidate["asset_category"] for candidate in candidates} == {
        "road_equipment",
        "rail_equipment",
        "hydro_equipment",
        "drainage_equipment",
    }
    assert all(candidate["status"] == "valid" for candidate in candidates)
    assert all(
        isinstance(candidate["ground_elevation_mm"], int) for candidate in candidates
    )
    assert canonical_json_bytes(first.inventory) == canonical_json_bytes(
        second.inventory
    )


def test_fixed_asset_is_grounded_once_and_keeps_the_exact_catalog_id() -> None:
    mnt, mns = _base_pair()
    fixed_assets = [
        {
            "schema": "fireviewer.projected-fixed-asset-placement.v1",
            "placement_id": "church-main",
            "asset_id": "church_village_01",
            "asset_category": "building",
            "source_wgs84": [43.9, 4.5],
            "position_l93_m": [700_100.25, 6_600_120.75],
            "owner_tile_origin_l93_m": list(ORIGIN),
            "yaw_rad": 0.5,
        }
    ]
    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-fixed",
        fixed_asset_placements=fixed_assets,
    )
    fixed_candidates = [
        candidate
        for candidate in result.inventory["context_assets"]["candidates"]
        if candidate.get("fixed_placement_id") == "church-main"
    ]
    assert len(fixed_candidates) == 1
    candidate = fixed_candidates[0]
    assert candidate["fixed_asset_id"] == "church_village_01"
    assert candidate["asset_category"] == "building"
    assert candidate["position_l93_m"] == [700_100.25, 6_600_120.75]
    assert candidate["yaw_rad"] == 0.5
    assert candidate["ground_elevation_mm"] == 100_305
    assert result.inventory["context"]["fixed_asset_placement_count"] == 1

    without_fixed = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-fixed",
    )
    assert result.inventory["build_id"] != without_fixed.inventory["build_id"]


def test_confirmed_roof_is_measured_when_connected_to_a_large_canopy() -> None:
    mnt, mns = _base_pair()
    roof_cells: list[tuple[int, int]] = []
    for global_y in range(6_600_040, 6_600_050):
        for global_x in range(700_040, 700_060):
            roof_cells.append((global_x, global_y))
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 7.0)
    canopy_cells: list[tuple[int, int]] = []
    for global_y in range(6_600_050, 6_600_090):
        for global_x in range(700_040, 700_090):
            canopy_cells.append((global_x, global_y))
            _set_height_cell(
                mnt,
                mns,
                ORIGIN,
                global_x,
                global_y,
                4.0 + ((global_x + global_y) % 7),
            )
    footprint = {
        "source_id": "BAT-connected-roof",
        "geometry": mapping(box(700_040, 6_600_040, 700_060, 6_600_050)),
    }
    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        building_footprints=[footprint],
        context_masks={
            "vegetation": _vegetation_mask(ORIGIN, canopy_cells),
        },
    )
    buildings = result.inventory["buildings"]
    confirmed = next(
        candidate
        for candidate in buildings["candidates"]
        if candidate["confirmed_source_id"] == "BAT-connected-roof"
    )
    assert confirmed["status"] == "valid"
    assert confirmed["footprint_area_m2"] == len(roof_cells)
    assert confirmed["height_cm"] == 700
    assert confirmed["source_id"].startswith("hag-component-")
    assert confirmed["confirmation_matches"][0]["footprint_overlap_ratio"] == 1.0
    assert buildings["unmatched_confirmation_count"] == 0
    assert result.inventory["trees"]["excluded_pixel_count"] >= len(roof_cells)


def test_adjacent_confirmations_do_not_share_boundary_pixels() -> None:
    mnt, mns = _base_pair()
    for global_y in range(6_600_040, 6_600_050):
        for global_x in range(700_040, 700_060):
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 7.0)
    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        building_footprints=[
            {
                "source_id": "BAT-left",
                "geometry": mapping(box(700_040, 6_600_040, 700_050, 6_600_050)),
            },
            {
                "source_id": "BAT-right",
                "geometry": mapping(box(700_050, 6_600_040, 700_060, 6_600_050)),
            },
        ],
    )
    confirmed = [
        candidate
        for candidate in result.inventory["buildings"]["candidates"]
        if candidate["confirmed_source_id"] is not None
    ]
    assert len(confirmed) == 2
    assert {candidate["confirmed_source_id"] for candidate in confirmed} == {
        "BAT-left",
        "BAT-right",
    }
    assert all(candidate["status"] == "valid" for candidate in confirmed)
    assert result.inventory["buildings"]["non_univocal_confirmation_count"] == 0


def test_one_semantic_footprint_keeps_distinct_measured_roof_components() -> None:
    mnt, mns = _base_pair()
    component_origins = [
        (700_040, 6_600_040),
        (700_050, 6_600_040),
        (700_060, 6_600_040),
    ]
    for component_west, component_south in component_origins:
        for global_y in range(component_south, component_south + 4):
            for global_x in range(component_west, component_west + 4):
                _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 7.0)

    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-multi-roof",
        building_footprints=[
            {
                "source_id": "BAT-campus",
                "geometry": mapping(box(700_039, 6_600_039, 700_064, 6_600_045)),
            }
        ],
    )

    buildings = result.inventory["buildings"]
    confirmed = [
        candidate
        for candidate in buildings["candidates"]
        if candidate["confirmed_source_id"] == "BAT-campus"
    ]
    assert len(confirmed) == 3
    assert buildings["confirmed_hag_component_count"] == 3
    assert buildings["multi_component_confirmation_count"] == 1
    assert buildings["non_univocal_confirmation_count"] == 0
    assert len({candidate["candidate_id"] for candidate in confirmed}) == 3
    assert all(candidate["status"] == "valid" for candidate in confirmed)
    assert all(candidate["footprint_area_m2"] == 16 for candidate in confirmed)


def test_isolated_negative_hag_outliers_are_clamped_and_recorded() -> None:
    mnt, mns = _base_pair()
    for row, column in ((20, 30), (200, 250), (510, 519)):
        mns[row, column] = mnt[row, column] - 0.58

    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        context_masks={"vegetation": np.zeros(SHAPE, dtype=bool)},
    )

    assert result.inventory["hag"]["minimum_source_delta_mm"] == -580
    assert result.inventory["hag"]["negative_source_sample_count_clamped"] == 3
    assert result.inventory["hag"]["negative_outlier_below_tolerance_count"] == 3
    assert result.inventory["hag"]["negative_outlier_fraction"] == round(
        3 / (520 * 520), 12
    )
    assert int(result.hag_core_cm.min()) == 0


def test_sparse_realistic_negative_hag_cluster_is_clamped_and_recorded() -> None:
    mnt, mns = _base_pair()
    samples = tuple(
        (row, column) for row in range(303, 307) for column in range(133, 136)
    )
    for row, column in samples:
        mns[row, column] = mnt[row, column] - 0.632

    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-sparse-negative-cluster",
        context_masks={"vegetation": np.zeros(SHAPE, dtype=bool)},
    )

    assert result.inventory["hag"]["minimum_source_delta_mm"] == -633
    assert result.inventory["hag"]["negative_source_sample_count_clamped"] == 12
    assert result.inventory["hag"]["negative_outlier_below_tolerance_count"] == 12
    assert result.inventory["hag"]["negative_outlier_fraction"] == round(
        12 / (520 * 520), 12
    )
    assert int(result.hag_core_cm.min()) == 0


def test_negative_hag_outlier_population_is_fail_closed() -> None:
    mnt, mns = _base_pair()
    mns.flat[:33] = mnt.flat[:33] - 0.51

    with pytest.raises(PlacementInventoryError, match="too many samples"):
        build_placement_inventory(
            mnt,
            mns,
            tile_origin_l93_m=ORIGIN,
            zone_id="FR-30-00001",
        )


def test_negative_hag_hard_outlier_is_fail_closed() -> None:
    mnt, mns = _base_pair()
    mns[100, 100] = mnt[100, 100] - 1.001

    with pytest.raises(PlacementInventoryError, match="more than 100 cm"):
        build_placement_inventory(
            mnt,
            mns,
            tile_origin_l93_m=ORIGIN,
            zone_id="FR-30-00001",
        )


def test_negative_hag_diagnostics_are_validated() -> None:
    mnt, mns = _base_pair()
    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        context_masks={"vegetation": np.zeros(SHAPE, dtype=bool)},
    )
    tampered = copy.deepcopy(result.inventory)
    tampered["hag"]["negative_outlier_below_tolerance_count"] = 33
    tampered["inventory_sha256"] = "0" * 64

    with pytest.raises(PlacementInventoryError, match="diagnostics violate"):
        validate_inventory(tampered)


def test_building_and_tree_sources_are_fully_reconciled_without_quota() -> None:
    mnt, mns = _base_pair()
    valid_building = mapping(box(700_020.0, 6_600_020.0, 700_025.0, 6_600_025.0))
    ambiguous_building = mapping(box(700_040.0, 6_600_040.0, 700_045.0, 6_600_045.0))
    for global_y in range(6_600_020, 6_600_025):
        for global_x in range(700_020, 700_025):
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 8.0)
    for global_y in range(6_600_040, 6_600_045):
        for global_x in range(700_040, 700_045):
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 1.0)

    _set_crown(mnt, mns, ORIGIN, 700_100, 6_600_100)
    _set_height_cell(mnt, mns, ORIGIN, 700_130, 6_600_130, 5.0)
    _set_crown(mnt, mns, ORIGIN, 700_160, 6_600_160)
    vegetation_cells = (
        _crown_cells(700_100, 6_600_100)
        + [(700_130, 6_600_130)]
        + _crown_cells(700_160, 6_600_160)
    )
    vegetation = _vegetation_mask(ORIGIN, vegetation_cells)
    roads = _vegetation_mask(ORIGIN, _crown_cells(700_160, 6_600_160))

    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        building_footprints=[
            {"source_id": "BAT-valid", "geometry": valid_building},
            {"source_id": "BAT-ambiguous", "geometry": ambiguous_building},
        ],
        context_masks={"vegetation": vegetation, "roads": roads},
    )
    buildings = result.inventory["buildings"]
    trees = result.inventory["trees"]

    # Source count is the reconciled HAG-component inventory, not the number
    # of semantic confirmations.  The two crown-shaped components remain
    # explicit rejected building candidates instead of disappearing.
    assert buildings["source_count"] == 3
    assert buildings["valid_count"] == 1
    assert buildings["ambiguous_count"] == 0
    assert buildings["rejected_count"] == 2
    assert buildings["confirmation_source_count"] == 2
    assert buildings["unmatched_confirmation_count"] == 1
    assert buildings["source_count"] == sum(
        buildings[f"{status}_count"] for status in ("valid", "ambiguous", "rejected")
    )
    assert buildings["source_count"] == (
        buildings["placement_ready_count"] + buildings["placement_blocked_count"]
    )
    assert buildings["instantiated_asset_count"] == 0
    assert trees["source_count"] == 2
    assert trees["valid_count"] == 2
    assert trees["rejected_count"] == 0
    assert trees["candidate_count_without_quota_or_thinning"] == 2
    assert trees["source_count"] == sum(
        trees[f"{status}_count"] for status in ("valid", "ambiguous", "rejected")
    )
    assert trees["source_count"] == (
        trees["placement_ready_count"] + trees["placement_blocked_count"]
    )
    assert trees["instantiated_asset_count"] == 0
    # 25 pixels are the confirmed HAG roof and 25 are the road-overlapping
    # crown.  Both are measured exclusions, not vegetation thinning.
    assert trees["excluded_pixel_count"] == 50
    retained_small_crown = next(
        candidate
        for candidate in trees["candidates"]
        if candidate["observed_crown_area_m2"] == 1
    )
    assert retained_small_crown["status"] == "valid"
    assert retained_small_crown["height_cm"] == 500
    assert retained_small_crown["minimum_crown_area_m2_applied"] == 1


def test_vegetation_context_classifies_but_never_authorizes_tree_placement() -> None:
    mnt, mns = _base_pair()
    first_peak = (700_100, 6_600_100)
    second_peak = (700_220, 6_600_220)
    _set_crown(mnt, mns, ORIGIN, *first_peak)
    _set_crown(mnt, mns, ORIGIN, *second_peak)
    no_prior = np.zeros(SHAPE, dtype=bool)
    partial_prior = _vegetation_mask(ORIGIN, _crown_cells(*first_peak))

    hag_only = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        context_masks={"vegetation": no_prior},
    )
    classified = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        context_masks={"vegetation": partial_prior},
    )

    def structural_candidates(inventory: dict[str, object]) -> list[tuple[object, ...]]:
        trees = inventory["trees"]
        assert isinstance(trees, dict)
        return [
            (
                candidate["candidate_id"],
                candidate["status"],
                candidate["peak_cell_l93"],
                candidate["position_l93_m"],
                candidate["height_cm"],
                candidate["observed_crown_area_m2"],
            )
            for candidate in trees["candidates"]
        ]

    assert structural_candidates(hag_only.inventory) == structural_candidates(
        classified.inventory
    )
    assert hag_only.inventory["trees"]["valid_count"] == 2
    assert classified.inventory["trees"]["valid_count"] == 2
    by_peak = {
        tuple(candidate["peak_cell_l93"]): candidate
        for candidate in classified.inventory["trees"]["candidates"]
    }
    assert by_peak[first_peak]["context_classification"] == "vegetation_prior"
    assert by_peak[second_peak]["context_classification"] == "hag_only"


def test_woody_context_retains_measured_one_to_two_metre_vegetation() -> None:
    mnt, mns = _base_pair()
    cells = [
        (global_x, global_y)
        for global_y in range(6_600_300, 6_600_303)
        for global_x in range(700_300, 700_303)
    ]
    for global_x, global_y in cells:
        _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 1.5)
    no_prior = np.zeros(SHAPE, dtype=bool)
    woody_prior = _vegetation_mask(ORIGIN, cells)

    default_floor = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        context_masks={"vegetation": no_prior},
    )
    wooded = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        context_masks={"vegetation": woody_prior},
    )

    assert default_floor.inventory["trees"]["source_count"] == 0
    assert wooded.inventory["trees"]["valid_count"] == 1
    candidate = wooded.inventory["trees"]["candidates"][0]
    assert candidate["height_cm"] == 150
    assert candidate["observed_crown_area_m2"] == 9
    assert candidate["context_classification"] == "low_1_to_3m_vegetation_prior"
    assert wooded.inventory["trees"]["woody_context_minimum_height_cm"] == 100
    assert wooded.inventory["trees"]["default_minimum_height_cm"] == 200


def test_woody_context_uses_measured_height_dependent_crown_area_without_quota() -> (
    None
):
    mnt, mns = _base_pair()
    tall_single = (700_100, 6_600_100)
    low_single = (700_200, 6_600_200)
    low_pair = [(700_300, 6_600_300), (700_301, 6_600_300)]
    outside_small = [(700_400, 6_600_400), (700_401, 6_600_400)]
    _set_height_cell(mnt, mns, ORIGIN, *tall_single, 4.0)
    _set_height_cell(mnt, mns, ORIGIN, *low_single, 1.5)
    _set_height_cell(mnt, mns, ORIGIN, *low_pair[0], 1.5)
    _set_height_cell(mnt, mns, ORIGIN, *low_pair[1], 1.4)
    _set_height_cell(mnt, mns, ORIGIN, *outside_small[0], 4.0)
    _set_height_cell(mnt, mns, ORIGIN, *outside_small[1], 3.9)
    woody_cells = [tall_single, low_single, *low_pair]

    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        context_masks={"vegetation": _vegetation_mask(ORIGIN, woody_cells)},
    )
    by_peak = {
        tuple(candidate["peak_cell_l93"]): candidate
        for candidate in result.inventory["trees"]["candidates"]
    }

    assert by_peak[tall_single]["status"] == "valid"
    assert by_peak[tall_single]["minimum_crown_area_m2_applied"] == 1
    assert by_peak[low_single]["status"] == "rejected"
    assert by_peak[low_single]["minimum_crown_area_m2_applied"] == 2
    assert by_peak[low_pair[0]]["status"] == "valid"
    assert by_peak[low_pair[0]]["minimum_crown_area_m2_applied"] == 2
    assert by_peak[outside_small[0]]["status"] == "rejected"
    assert by_peak[outside_small[0]]["minimum_crown_area_m2_applied"] == 4
    trees = result.inventory["trees"]
    assert trees["candidate_count_without_quota_or_thinning"] == trees["source_count"]


def test_watershed_assigns_the_complete_measured_crown_to_its_peak() -> None:
    mnt = np.full(SHAPE, 100.0, dtype="float64")
    rows, columns = np.indices(SHAPE, dtype="float64")
    radius = np.sqrt((rows - 260.0) ** 2 + (columns - 270.0) ** 2)
    hag = np.maximum(0.0, 9.0 * (1.0 - radius / 7.0))
    origin = (820_000, 6_312_500)
    woody_polygon = mapping(box(origin[0], origin[1], origin[0] + 500, origin[1] + 500))

    result = build_placement_inventory(
        mnt,
        mnt + hag,
        tile_origin_l93_m=origin,
        zone_id="FR-30-00001",
        context_geometries={"vegetation": [woody_polygon]},
    )

    trees = result.inventory["trees"]
    assert trees["source_count"] == 1
    assert trees["valid_count"] == 1
    assert trees["eligible_pixel_count"] == 121
    assert trees["candidates"][0]["observed_crown_area_m2"] == 121


def test_flat_woody_canopy_plateau_gets_global_measured_markers() -> None:
    mnt, mns = _base_pair()
    plateau_cells: list[tuple[int, int]] = []
    for global_y in range(6_600_100, 6_600_160):
        for global_x in range(700_100, 700_160):
            plateau_cells.append((global_x, global_y))
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 8.0)
    vegetation = _vegetation_mask(ORIGIN, plateau_cells)

    first = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-flat-forest",
        context_masks={"vegetation": vegetation},
    )
    second = build_placement_inventory(
        mnt.copy(),
        mns.copy(),
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-flat-forest",
        context_masks={"vegetation": vegetation.copy()},
    )

    trees = first.inventory["trees"]
    assert trees["eligible_pixel_count"] == 3_600
    assert trees["source_count"] == 225
    assert trees["valid_count"] == 225
    assert trees["split_flat_woody_plateau_count_processing_window"] == 1
    assert trees["flat_woody_plateau_extra_marker_count_processing_window"] == 224
    assert (
        sum(candidate["observed_crown_area_m2"] for candidate in trees["candidates"])
        == 3_600
    )
    assert (
        max(candidate["observed_crown_area_m2"] for candidate in trees["candidates"])
        <= 36
    )
    assert canonical_json_bytes(first.inventory) == canonical_json_bytes(
        second.inventory
    )


def test_small_flat_woody_crown_is_not_split() -> None:
    mnt, mns = _base_pair()
    crown_cells: list[tuple[int, int]] = []
    for global_y in range(6_600_100, 6_600_105):
        for global_x in range(700_100, 700_105):
            crown_cells.append((global_x, global_y))
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 8.0)

    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-flat-crown",
        context_masks={"vegetation": _vegetation_mask(ORIGIN, crown_cells)},
    )

    trees = result.inventory["trees"]
    assert trees["source_count"] == 1
    assert trees["valid_count"] == 1
    assert trees["split_flat_woody_plateau_count_processing_window"] == 0


def test_hag_autodetection_separates_flat_roof_from_tree_crown() -> None:
    mnt, mns = _base_pair()
    for global_y in range(6_600_020, 6_600_028):
        for global_x in range(700_020, 700_030):
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, 7.0)
    _set_crown(mnt, mns, ORIGIN, 700_100, 6_600_100, radius=3)

    first = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
    )
    second = build_placement_inventory(
        mnt.copy(),
        mns.copy(),
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
    )
    buildings = first.inventory["buildings"]

    assert buildings["detection_mode"] == "mns_mnt_hag_with_semantic_confirmation"
    assert buildings["source_count"] == 2
    assert buildings["valid_count"] == 0
    assert buildings["ambiguous_count"] == 1
    assert buildings["rejected_count"] == 1
    assert buildings["source_count"] == (
        buildings["placement_ready_count"] + buildings["placement_blocked_count"]
    )
    roof = next(
        candidate
        for candidate in buildings["candidates"]
        if candidate["status"] == "ambiguous"
    )
    rejected_tree_shape = next(
        candidate
        for candidate in buildings["candidates"]
        if candidate["status"] == "rejected"
    )
    assert roof["footprint_area_m2"] == 80
    assert roof["height_cm"] == 700
    assert roof["metrics"]["height_dispersion_p90_p10_cm"] == 0
    assert roof["metrics"]["rectangularity"] == 1.0
    assert roof["footprint_geojson"]["type"] == "Polygon"
    assert roof["reason_codes"] == ["morphology_requires_semantic_confirmation"]
    assert "height_dispersion_tree_like" in rejected_tree_shape["reason_codes"]
    assert canonical_json_bytes(first.inventory) == canonical_json_bytes(
        second.inventory
    )

    building_confirmation = _vegetation_mask(
        ORIGIN,
        [
            (global_x, global_y)
            for global_y in range(6_600_020, 6_600_028)
            for global_x in range(700_020, 700_030)
        ],
    )
    confirmed = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="FR-30-00001",
        context_masks={"buildings": building_confirmation},
    )
    confirmed_buildings = confirmed.inventory["buildings"]
    confirmed_trees = confirmed.inventory["trees"]
    assert confirmed_buildings["valid_count"] == 1
    assert confirmed_buildings["ambiguous_count"] == 0
    assert confirmed_buildings["rejected_count"] == 1
    assert confirmed_trees["source_count"] == 1
    assert confirmed_trees["valid_count"] == 1
    assert confirmed_trees["candidates"][0]["peak_cell_l93"] == [
        700_100,
        6_600_100,
    ]


def test_rectangular_flat_canopy_never_becomes_valid_without_confirmation() -> None:
    mnt, mns = _base_pair()
    canopy_cells: list[tuple[int, int]] = []
    for global_y in range(6_600_200, 6_600_212):
        for global_x in range(700_200, 700_212):
            canopy_cells.append((global_x, global_y))
            height = 6.0 + 0.4 * ((global_x + global_y) % 2)
            _set_height_cell(mnt, mns, ORIGIN, global_x, global_y, height)

    morphology_only = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="test",
    )
    assert morphology_only.inventory["buildings"]["valid_count"] == 0
    assert morphology_only.inventory["buildings"]["source_count"] == 1
    assert morphology_only.inventory["buildings"]["ambiguous_count"] == 1

    vegetation_confirmed = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="test",
        context_masks={"vegetation": _vegetation_mask(ORIGIN, canopy_cells)},
    )
    candidate = vegetation_confirmed.inventory["buildings"]["candidates"][0]
    assert candidate["status"] == "rejected"
    assert "vegetation_prior_dominant" in candidate["reason_codes"]
    assert vegetation_confirmed.inventory["trees"]["source_count"] > 0


def test_adjacent_tiles_share_peak_id_and_half_open_ownership_then_merge() -> None:
    left_origin = (700_000, 6_600_000)
    right_origin = (700_500, 6_600_000)
    peak_x = 700_499
    peak_y = 6_600_250

    def build(origin: tuple[int, int]):
        mnt, mns = _base_pair()
        # _base_pair uses local row/column slopes only; MNS-MNT and detection
        # remain globally identical, which is what this seam test exercises.
        _set_crown(mnt, mns, origin, peak_x, peak_y, radius=3)
        vegetation = _vegetation_mask(origin, _crown_cells(peak_x, peak_y, radius=3))
        return build_placement_inventory(
            mnt,
            mns,
            tile_origin_l93_m=origin,
            zone_id="FR-30-00001",
            context_masks={"vegetation": vegetation},
        )

    left = build(left_origin)
    right = build(right_origin)
    left_candidates = left.inventory["trees"]["candidates"]
    right_candidates = right.inventory["trees"]["candidates"]

    assert len(left_candidates) == 1
    assert right_candidates == []
    left_fragment = next(
        fragment
        for fragment in left.inventory["trees"]["fragments"]
        if fragment["candidate_id"] == left_candidates[0]["candidate_id"]
    )
    right_fragment = next(
        fragment
        for fragment in right.inventory["trees"]["fragments"]
        if fragment["candidate_id"] == left_candidates[0]["candidate_id"]
    )
    assert left_fragment["owned"] is True
    assert right_fragment["owned"] is False
    assert set(left_fragment["seam_keys"]) & set(right_fragment["seam_keys"])

    merged = merge_tree_inventories([left.inventory, right.inventory])
    assert merged["candidate_group_count"] == 1
    assert merged["shared_seam_key_count"] > 0
    group = next(group for group in merged["groups"] if group["owner_tiles"])
    assert group["owner_tiles"] == [left.inventory["tile_id"]]
    assert group["member_candidate_ids"] == [left_candidates[0]["candidate_id"]]
    assert group["total_owned_core_area_m2"] == 49


def test_flat_woody_plateau_uses_the_same_global_markers_across_tiles() -> None:
    left_origin = (700_000, 6_600_000)
    right_origin = (700_500, 6_600_000)
    plateau_cells = [
        (global_x, global_y)
        for global_y in range(6_600_200, 6_600_240)
        for global_x in range(700_480, 700_520)
    ]

    def build(origin: tuple[int, int]):
        mnt, mns = _base_pair()
        visible_cells = [
            (global_x, global_y)
            for global_x, global_y in plateau_cells
            if origin[0] - 10 <= global_x < origin[0] + 510
        ]
        for global_x, global_y in visible_cells:
            _set_height_cell(mnt, mns, origin, global_x, global_y, 8.0)
        return build_placement_inventory(
            mnt,
            mns,
            tile_origin_l93_m=origin,
            zone_id="FR-flat-forest-seam",
            context_masks={
                "vegetation": _vegetation_mask(origin, visible_cells),
            },
        )

    left = build(left_origin)
    right = build(right_origin)
    left_owned = {
        tuple(candidate["peak_cell_l93"])
        for candidate in left.inventory["trees"]["candidates"]
    }
    right_owned = {
        tuple(candidate["peak_cell_l93"])
        for candidate in right.inventory["trees"]["candidates"]
    }
    left_fragments = {
        fragment["candidate_id"] for fragment in left.inventory["trees"]["fragments"]
    }
    right_fragments = {
        fragment["candidate_id"] for fragment in right.inventory["trees"]["fragments"]
    }

    assert len(left_owned) == 50
    assert len(right_owned) == 50
    assert left_owned.isdisjoint(right_owned)
    assert left_fragments & right_fragments
    merged = merge_tree_inventories([left.inventory, right.inventory])
    assert merged["candidate_group_count"] == 100
    assert merged["shared_seam_key_count"] > 0


def test_rejects_corrupt_grids_masks_and_duplicate_source_ids() -> None:
    mnt, mns = _base_pair()
    bad_nan = mns.copy()
    bad_nan[0, 0] = np.nan
    with pytest.raises(PlacementInventoryError, match="nodata or NaN"):
        build_placement_inventory(
            mnt,
            bad_nan,
            tile_origin_l93_m=ORIGIN,
            zone_id="test",
        )

    with pytest.raises(PlacementInventoryError, match="corrupt or misaligned"):
        build_placement_inventory(
            mnt,
            mnt - 1.0,
            tile_origin_l93_m=ORIGIN,
            zone_id="test",
        )

    with pytest.raises(PlacementInventoryError, match="must be boolean"):
        build_placement_inventory(
            mnt,
            mns,
            tile_origin_l93_m=ORIGIN,
            zone_id="test",
            context_masks={"roads": np.zeros(SHAPE, dtype="uint8")},
        )

    footprint = mapping(box(700_020, 6_600_020, 700_025, 6_600_025))
    with pytest.raises(PlacementInventoryError, match="duplicate building source_id"):
        build_placement_inventory(
            mnt,
            mns,
            tile_origin_l93_m=ORIGIN,
            zone_id="test",
            building_footprints=[
                {"source_id": "duplicate", "geometry": footprint},
                {"source_id": "duplicate", "geometry": footprint},
            ],
        )


def test_sparse_positive_uint16_outlier_is_repaired_to_local_ground() -> None:
    mnt, mns = _base_pair()
    mns[12, 34] = mnt[12, 34] + 700.0

    mnt_mm, mns_mm, hag_cm, diagnostics = inventory._source_grid(mnt, mns)

    assert mns_mm[12, 34] == mnt_mm[12, 34]
    assert hag_cm[12, 34] == 0
    assert diagnostics["maximum_source_delta_mm_before_repair"] == 700_000
    assert diagnostics["positive_uint16_outlier_count_repaired_to_ground"] == 1
    assert diagnostics["positive_uint16_outlier_fraction"] == round(1 / (520 * 520), 12)


def test_dense_positive_uint16_outliers_are_rejected_as_corrupt() -> None:
    mnt, mns = _base_pair()
    mns[0, :33] = mnt[0, :33] + 700.0

    with pytest.raises(PlacementInventoryError, match="corrupt or misaligned"):
        inventory._source_grid(mnt, mns)


def test_inventory_hash_detects_tampering() -> None:
    mnt, mns = _base_pair()
    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="test",
        context_masks={"vegetation": np.zeros(SHAPE, dtype=bool)},
    )
    corrupted = copy.deepcopy(result.inventory)
    corrupted["trees"]["source_count"] = 1
    with pytest.raises(PlacementInventoryError, match="reconciliation"):
        validate_inventory(corrupted)

    corrupted = copy.deepcopy(result.inventory)
    corrupted["zone_id"] = "tampered"
    with pytest.raises(PlacementInventoryError, match="hash mismatch"):
        validate_inventory(corrupted)


def test_d_only_atomic_outputs_round_trip_and_are_reproducible(tmp_path: Path) -> None:
    mnt, mns = _base_pair()
    result = build_placement_inventory(
        mnt,
        mns,
        tile_origin_l93_m=ORIGIN,
        zone_id="test",
        context_masks={"vegetation": np.zeros(SHAPE, dtype=bool)},
    )
    first_hag = tmp_path / "first" / "placement-hag-1m.tif"
    second_hag = tmp_path / "second" / "placement-hag-1m.tif"
    first_json = tmp_path / "first" / "placement-inventory.json"
    second_json = tmp_path / "second" / "placement-inventory.json"

    first_hag_hash = write_hag_1m(
        first_hag, result.hag_core_cm, tile_origin_l93_m=ORIGIN
    )
    second_hag_hash = write_hag_1m(
        second_hag, result.hag_core_cm, tile_origin_l93_m=ORIGIN
    )
    first_json_hash = write_inventory_json(first_json, result.inventory)
    second_json_hash = write_inventory_json(second_json, result.inventory)
    restored, metadata = read_hag_1m(first_hag)

    assert first_hag_hash == second_hag_hash
    assert first_hag.read_bytes() == second_hag.read_bytes()
    assert first_json_hash == second_json_hash
    assert first_json.read_bytes() == second_json.read_bytes()
    assert np.array_equal(restored, result.hag_core_cm)
    assert metadata["bounds_l93_m"] == [
        700_000.0,
        6_600_000.0,
        700_500.0,
        6_600_500.0,
    ]

    with pytest.raises(PlacementInventoryError, match="forbidden on C"):
        assert_d_storage_path(Path("C:/fireviewer/placement-inventory.json"))
    if not gpkg_supported():
        with pytest.raises(GeoPackageUnavailableError, match="Fiona"):
            write_inventory_gpkg(
                tmp_path / "placement-candidates.gpkg", result.inventory
            )


def test_contract_is_locked_and_matches_module() -> None:
    contract = json.loads(
        Path(__file__)
        .with_name("mns_mnt_placement_contract.v1.json")
        .read_text(encoding="utf-8")
    )
    assert contract["schema"] == CONTRACT_SCHEMA
    assert contract["status"] == "locked"
    assert contract["trees"]["quota"] == "forbidden"
    assert contract["trees"]["thinning"] == "forbidden"
    assert contract["retention"]["storage_drive"] == "D"


def test_cli_and_output_api_produce_only_compact_consumable_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mnt, mns = _base_pair()
    _set_crown(mnt, mns, ORIGIN, 700_100, 6_600_100)
    vegetation = _vegetation_mask(ORIGIN, _crown_cells(700_100, 6_600_100))
    source = tmp_path / "source.npz"
    np.savez(source, mnt=mnt, mns=mns, mask_vegetation=vegetation)
    output = tmp_path / "outputs"

    assert (
        main(
            [
                "--source-npz",
                str(source),
                "--zone",
                "FR-30-00001",
                "--tile-origin",
                str(ORIGIN[0]),
                str(ORIGIN[1]),
                "--output-dir",
                str(output),
                "--gpkg",
                "off",
                "--execute",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "execute"
    assert summary["trees"]["source_count"] == 1
    assert sorted(path.name for path in output.iterdir()) == [
        "placement-hag-1m.tif",
        "placement-inventory.json",
    ]
    inventory = json.loads(
        (output / "placement-inventory.json").read_text(encoding="utf-8")
    )
    validate_inventory(inventory)
    assert inventory["trees"]["candidates"][0]["ground_elevation_mm"] is not None
