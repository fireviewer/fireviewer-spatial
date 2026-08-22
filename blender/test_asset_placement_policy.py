from __future__ import annotations

from shapely.geometry import box

import asset_placement_policy as policy
import mns_mnt_placement_inventory_v2 as placement


def test_building_form_uses_strict_measured_thresholds() -> None:
    assert policy.building_form(height_m=7.99, footprint_area_m2=179.99) == (
        "low_rise_house"
    )
    assert policy.building_form(height_m=8.0, footprint_area_m2=80.0) == (
        "mid_rise_residential"
    )
    assert policy.building_form(height_m=13.0, footprint_area_m2=80.0) == (
        "multi_storey_residential"
    )


def test_fuel_station_requires_explicit_source_terms_and_is_zone_bounded() -> None:
    assert (
        policy.special_building_role(
            semantic_tags=["commercial"], reference_terms=["commerce"]
        )
        == "commercial"
    )
    assert (
        policy.special_building_role(
            semantic_tags=["commercial"], reference_terms=["station", "essence"]
        )
        == "fuel_station"
    )
    assert policy.special_role_limit("fuel_station", total_buildings=100_000) == 1


def test_procedural_selection_is_repeatable_and_family_specific() -> None:
    building = policy.procedural_asset_id(
        family="buildings", building_form_name="low_rise_house"
    )
    tree = policy.procedural_asset_id(
        family="trees", semantic_tags=["conifer"]
    )
    assert building == "procedural-building-low-rise-house"
    assert tree == "procedural-tree-conifer"
    first = policy.procedural_selection_seed(
        zone_id="zone", candidate_id="candidate", asset_id=building
    )
    second = policy.procedural_selection_seed(
        zone_id="zone", candidate_id="candidate", asset_id=building
    )
    assert first == second


def test_asset_shape_compatibility_rejects_stretching_but_allows_uniform_scale() -> None:
    assert policy.asset_shape_is_compatible(
        native_dimensions_m=(10.0, 5.0, 8.0),
        measured_dimensions_m=(20.0, 10.0, 16.0),
    )
    assert not policy.asset_shape_is_compatible(
        native_dimensions_m=(10.0, 5.0, 8.0),
        measured_dimensions_m=(40.0, 5.0, 8.0),
    )


def test_asset_repeat_limit_bounds_one_asset_without_overreacting_to_small_zones() -> None:
    assert policy.asset_repeat_limit(candidate_count=12, compatible_asset_count=4) == 8
    assert policy.asset_repeat_limit(candidate_count=1_000, compatible_asset_count=10) == 200


def _building(candidate_id: str, source_id: str, height_cm: int, area: float):
    return {
        "candidate_id": candidate_id,
        "source_id": source_id,
        "status": "valid",
        "reason_codes": [],
        "height_cm": height_cm,
        "footprint_area_m2": area,
    }


def test_building_contact_is_allowed_but_positive_area_overlap_is_removed() -> None:
    footprints = [
        ("a", box(0, 0, 10, 10), "a" * 64, {}),
        ("touch", box(10, 0, 20, 10), "b" * 64, {}),
        ("overlap", box(5, 0, 15, 10), "c" * 64, {}),
    ]
    buildings = {
        "candidates": [
            _building("building-a", "a", 900, 100.0),
            _building("building-touch", "touch", 800, 100.0),
            _building("building-overlap", "overlap", 700, 100.0),
        ]
    }
    removed = placement._remove_overlapping_buildings(footprints, buildings)
    indexed = {item["candidate_id"]: item for item in buildings["candidates"]}
    assert removed == 1
    assert indexed["building-a"]["status"] == "valid"
    assert indexed["building-touch"]["status"] == "valid"
    assert indexed["building-overlap"]["status"] == "rejected"
    assert indexed["building-overlap"]["reason_codes"] == [
        "building_positive_area_overlap_removed"
    ]
    assert buildings["valid_count"] == 2
    assert buildings["rejected_count"] == 1


def test_tree_covered_by_valid_building_is_removed_with_reason() -> None:
    footprints = [("a", box(0, 0, 10, 10), "a" * 64, {})]
    buildings = {"candidates": [_building("building-a", "a", 900, 100.0)]}
    trees = {
        "candidates": [
            {
                "candidate_id": "tree-inside",
                "status": "valid",
                "reason_codes": [],
                "position_l93_m": [5.0, 5.0],
            },
            {
                "candidate_id": "tree-outside",
                "status": "valid",
                "reason_codes": [],
                "position_l93_m": [20.0, 20.0],
            },
        ]
    }
    removed = placement._remove_trees_inside_buildings(trees, footprints, buildings)
    indexed = {item["candidate_id"]: item for item in trees["candidates"]}
    assert removed == 1
    assert indexed["tree-inside"]["status"] == "rejected"
    assert indexed["tree-inside"]["reason_codes"] == [
        "tree_inside_building_footprint_removed"
    ]
    assert indexed["tree-outside"]["status"] == "valid"
