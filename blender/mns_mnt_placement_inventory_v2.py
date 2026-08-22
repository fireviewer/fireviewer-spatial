"""Experimental factual-placement profile for the RunPod v2 map builder.

This module deliberately leaves the production v1 inventory untouched.  It
keeps the v1 outer inventory schema so the existing OpenUSD scene assembler can
consume the result, but records a distinct v2 contract and algorithm identity.

The profile keeps quantity measured while tightening factual presentation:

* BD TOPO footprints author building XY geometry; MNT/MNS author ground/height.
* Tree count/status stays identical to the measured 1 m crown detector, while
  the position/ground/height of each candidate is refined inside its exact
  original 1 m peak cell from the native 0.5 m elevation pair when available.
* Tree crown and height measurements select and uniformly resize a compatible
  prototype; the visual base uses bounded local MNT support, and IGN forest
  composition is retained for deterministic selection.
* Road/rail/hydro features no longer create generic equipment instances.  Only
  validated fixed-coordinate context assets are instantiated by this profile.

No quota or thinning is introduced. The legacy Lightning image does not import
this module and remains an independent comparison reference.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Point

import mns_mnt_placement_inventory as v1
from fixed_asset_placement import (
    FixedAssetPlacementError,
    validate_projected_placements,
)
from fixed_asset_placement import canonical_json_bytes as canonical_fixed_asset_bytes

CONTRACT_SCHEMA = "fireviewer.mns-mnt-placement-contract.v2"
ALGORITHM = "fireviewer.mns-mnt-placement-algorithm.v10"
PLACEMENT_PROFILE = "fireviewer.factual-placement-profile.v2"
NATIVE_RESOLUTION_M = 0.5
NATIVE_SOURCE_SIZE = 1040
NATIVE_SEVERE_NEGATIVE_LIMIT_MM = -1_000
NATIVE_SEVERE_NEGATIVE_MAX_FRACTION = 0.01
CANONICAL_SEVERE_NEGATIVE_LIMIT_MM = -1_000
CANONICAL_SEVERE_NEGATIVE_MAX_FRACTION = 0.01
TREE_BASE_MAX_CLEARANCE_MM = 150
TREE_ASSET_SELECTION_POLICY = (
    "current_bdtopo_composition_else_bdforet_v1_then_conifer_or_oak_only"
)
BUILDING_INTERIOR_OVERLAP_EPSILON_M2 = 0.01


class NativeGridMisalignmentError(v1.PlacementInventoryError):
    """The optional native pair cannot safely refine canonical 1 m candidates."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contract_sha256() -> str:
    path = Path(__file__).with_name("mns_mnt_placement_contract.v2.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("algorithm") != ALGORITHM:
        raise v1.PlacementInventoryError("v2 placement contract identity is invalid")
    return _sha256(_canonical_bytes(payload))


def _native_pair_mm(
    mnt_05m: Any | None,
    mns_05m: Any | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if mnt_05m is None or mns_05m is None:
        return None
    mnt = np.asarray(mnt_05m, dtype="float64")
    mns = np.asarray(mns_05m, dtype="float64")
    expected = (NATIVE_SOURCE_SIZE, NATIVE_SOURCE_SIZE)
    if mnt.shape != expected or mns.shape != expected:
        raise v1.PlacementInventoryError(
            f"native 0.5 m MNT/MNS must have shape {expected}"
        )
    if not np.isfinite(mnt).all() or not np.isfinite(mns).all():
        raise v1.PlacementInventoryError("native 0.5 m MNT/MNS must be finite")
    return v1._quantize_mm(mnt), v1._quantize_mm(mns)


def _source_grid_v2(
    mnt_m: Any, mns_m: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """Build HAG without discarding a tile for a few harmless low MNS cells.

    Negative HAG cannot create a false object: it is clamped to ground. The v1
    count threshold rejected real mountain tiles for only four extra samples.
    Isolated larger negative samples are also harmless after that clamp, while
    a systemic negative offset still proves that the grids are misaligned.
    """

    mnt = np.asarray(mnt_m, dtype="float64")
    mns = np.asarray(mns_m, dtype="float64")
    expected = (v1.PROCESSING_SIZE, v1.PROCESSING_SIZE)
    if mnt.shape != expected or mns.shape != expected:
        raise v1.PlacementInventoryError(
            f"MNT and MNS must be co-registered arrays with shape {expected}"
        )
    if not np.isfinite(mnt).all() or not np.isfinite(mns).all():
        raise v1.PlacementInventoryError("MNT and MNS must not contain nodata or NaN")
    mnt_mm = v1._quantize_mm(mnt)
    mns_mm = v1._quantize_mm(mns)
    delta_mm = mns_mm.astype("int64") - mnt_mm.astype("int64")
    minimum_delta_mm = int(delta_mm.min())
    severe_negative_count = int(
        np.count_nonzero(delta_mm < CANONICAL_SEVERE_NEGATIVE_LIMIT_MM)
    )
    severe_negative_fraction = severe_negative_count / int(delta_mm.size)
    if severe_negative_fraction > CANONICAL_SEVERE_NEGATIVE_MAX_FRACTION:
        raise v1.PlacementInventoryError(
            "factual v2 canonical MNS/MNT has a systemic negative offset"
        )
    negative_outlier_count = int(
        np.count_nonzero(delta_mm < -(v1.NEGATIVE_HAG_TOLERANCE_CM * 10))
    )
    negative_outlier_fraction = negative_outlier_count / int(delta_mm.size)
    negative_sample_count = int(np.count_nonzero(delta_mm < 0))
    delta_cm = (np.maximum(delta_mm, 0) + 5) // 10
    maximum_delta_mm = int(delta_mm.max())
    positive_outliers = delta_cm > v1.MAX_HAG_CM
    positive_outlier_count = int(np.count_nonzero(positive_outliers))
    positive_outlier_fraction = positive_outlier_count / int(delta_mm.size)
    if (
        positive_outlier_count > v1.POSITIVE_HAG_MAX_OUTLIER_COUNT
        or positive_outlier_fraction > v1.POSITIVE_HAG_MAX_OUTLIER_FRACTION
    ):
        raise v1.PlacementInventoryError(
            "factual v2 MNS-MNT has too many samples above the uint16 contract"
        )
    if positive_outlier_count:
        mns_mm = mns_mm.copy()
        delta_mm = delta_mm.copy()
        delta_cm = delta_cm.copy()
        mns_mm[positive_outliers] = mnt_mm[positive_outliers]
        delta_mm[positive_outliers] = 0
        delta_cm[positive_outliers] = 0
    diagnostics: dict[str, int | float | str] = {
        "minimum_source_delta_mm": minimum_delta_mm,
        "maximum_source_delta_mm_before_repair": maximum_delta_mm,
        "negative_source_sample_count_clamped": negative_sample_count,
        "negative_outlier_below_tolerance_count": negative_outlier_count,
        "negative_outlier_fraction": round(negative_outlier_fraction, 12),
        "severe_negative_below_100cm_count": severe_negative_count,
        "severe_negative_below_100cm_fraction": round(
            severe_negative_fraction, 12
        ),
        "negative_outlier_policy": (
            "clamp_all_when_below_minus_100cm_fraction_is_at_most_1pct"
        ),
        "positive_uint16_outlier_count_repaired_to_ground": positive_outlier_count,
        "positive_uint16_outlier_fraction": round(positive_outlier_fraction, 12),
    }
    return mnt_mm, mns_mm, delta_cm.astype("uint16"), diagnostics


def _refine_tree_candidates_native_05m(
    trees: dict[str, Any],
    *,
    mnt_mm_05m: np.ndarray,
    mns_mm_05m: np.ndarray,
    west: int,
    south: int,
) -> tuple[int, dict[str, int | float | str]]:
    """Refine each existing tree inside its original 1 m peak cell only.

    The operation cannot create, delete, merge or move a candidate into another
    1 m crown cell.  It therefore improves sub-metre placement without changing
    the v1 quantity authority used by the validation run.
    """

    delta_mm = mns_mm_05m.astype("int64") - mnt_mm_05m.astype("int64")
    severe_negative_count = int(
        np.count_nonzero(delta_mm < NATIVE_SEVERE_NEGATIVE_LIMIT_MM)
    )
    severe_negative_fraction = severe_negative_count / int(delta_mm.size)
    if severe_negative_fraction > NATIVE_SEVERE_NEGATIVE_MAX_FRACTION:
        raise NativeGridMisalignmentError(
            "factual v2 native MNS/MNT has a systemic negative offset"
        )
    hag_mm = np.maximum(delta_mm, 0)
    grid_west = west - v1.HALO_M
    grid_north = south + v1.TILE_SIZE_M + v1.HALO_M
    refined = 0

    candidates = trees.get("candidates")
    if not isinstance(candidates, list):
        raise v1.PlacementInventoryError("tree candidate array is invalid")
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("status") not in {
            "valid",
            "ambiguous",
        }:
            continue
        peak = candidate.get("peak_cell_l93")
        if (
            not isinstance(peak, list)
            or len(peak) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in peak)
        ):
            raise v1.PlacementInventoryError("tree peak cell is invalid")
        peak_x = int(peak[0])
        peak_y = int(peak[1])
        if float(peak[0]) != peak_x or float(peak[1]) != peak_y:
            raise v1.PlacementInventoryError("tree peak cell must use integer metres")

        column0 = int(round((peak_x - grid_west) / NATIVE_RESOLUTION_M))
        row0 = int(round((grid_north - (peak_y + 1)) / NATIVE_RESOLUTION_M))
        if not (
            0 <= row0 < NATIVE_SOURCE_SIZE - 1
            and 0 <= column0 < NATIVE_SOURCE_SIZE - 1
        ):
            raise v1.PlacementInventoryError("tree native refinement escaped source grid")

        choices: list[tuple[int, float, float, int, int, int]] = []
        for row in (row0, row0 + 1):
            for column in (column0, column0 + 1):
                x_m = grid_west + (column + 0.5) * NATIVE_RESOLUTION_M
                y_m = grid_north - (row + 0.5) * NATIVE_RESOLUTION_M
                choices.append(
                    (
                        int(hag_mm[row, column]),
                        x_m,
                        y_m,
                        row,
                        column,
                        int(mnt_mm_05m[row, column]),
                    )
                )
        height_mm, x_m, y_m, row, column, ground_mm = min(
            choices,
            key=lambda item: (-item[0], item[1], item[2], item[3], item[4]),
        )
        support_elevation_mm = min(
            max(item[5] for item in choices),
            ground_mm + TREE_BASE_MAX_CLEARANCE_MM,
        )
        height_cm = max(1, int((height_mm + 5) // 10))
        previous = candidate.get("position_l93_m")
        candidate["position_l93_m"] = [round(x_m, 3), round(y_m, 3)]
        candidate["ground_elevation_mm"] = ground_mm
        candidate["support_elevation_mm"] = support_elevation_mm
        candidate["height_cm"] = height_cm
        candidate["top_elevation_mm"] = ground_mm + height_cm * 10
        candidate["native_05m_refinement"] = {
            "schema": "fireviewer.tree-native-refinement.v1",
            "resolution_m": NATIVE_RESOLUTION_M,
            "policy": "highest_hag_sample_inside_original_1m_peak_cell",
            "original_position_l93_m": previous,
            "native_row": row,
            "native_column": column,
            "native_hag_mm": height_mm,
            "support_policy": (
                "highest_native_mnt_inside_peak_cell_bounded_to_15cm_clearance"
            ),
            "support_elevation_mm": support_elevation_mm,
        }
        refined += 1
    return refined, {
        "minimum_native_delta_mm": int(delta_mm.min()),
        "negative_native_sample_count": int(np.count_nonzero(delta_mm < 0)),
        "severe_negative_native_sample_count": severe_negative_count,
        "severe_negative_native_fraction": round(severe_negative_fraction, 12),
        "integrity_policy": (
            "reject_when_more_than_1pct_is_below_mnt_by_more_than_100cm"
        ),
    }


def _feature_matches_point(
    features: Sequence[Mapping[str, Any]], point: Point
) -> list[Mapping[str, Any]]:
    matches = [
        feature
        for feature in features
        if feature["geometry"].bounds[0] <= point.x <= feature["geometry"].bounds[2]
        and feature["geometry"].bounds[1] <= point.y <= feature["geometry"].bounds[3]
        and feature["geometry"].covers(point)
    ]
    return sorted(
        matches,
        key=lambda item: (float(item["geometry"].area), str(item["source_id"])),
    )


def _has_current_forest_composition(properties: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(value).casefold()
        for value in properties.values()
        if isinstance(value, str)
    )
    return any(
        term in text
        for term in ("conif", "feuill", "mixte", "mélang", "melang")
    )


def _enrich_tree_semantics(
    trees: Mapping[str, Any],
    *,
    vegetation_features: Sequence[Mapping[str, Any]],
    forest_composition_features: Sequence[Mapping[str, Any]],
) -> int:
    candidates = trees.get("candidates")
    if not isinstance(candidates, list):
        raise v1.PlacementInventoryError("tree candidate array is invalid")
    enriched = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise v1.PlacementInventoryError("tree candidate is invalid")
        position = candidate.get("position_l93_m")
        if not isinstance(position, list) or len(position) != 2:
            raise v1.PlacementInventoryError("tree position is invalid")
        point = Point(float(position[0]), float(position[1]))
        current = _feature_matches_point(vegetation_features, point)
        historical = _feature_matches_point(forest_composition_features, point)
        current_primary = current[0] if current else None
        historical_primary = historical[0] if historical else None
        selection_properties: dict[str, Any] = {}
        if current_primary is not None:
            selection_properties.update(dict(current_primary["source_properties"]))
        # BD Foret v1 is older. It refines only generic current BD TOPO zones;
        # an explicit current conifer/broadleaf/mixed class always wins.
        if historical_primary is not None and not _has_current_forest_composition(
            selection_properties
        ):
            selection_properties.update(
                {
                    f"forest_{key}": value
                    for key, value in historical_primary["source_properties"].items()
                }
            )
        candidate["source_properties"] = selection_properties
        candidate["vegetation_context_source_ids"] = [
            str(feature["source_id"]) for feature in current
        ]
        candidate["forest_composition_source_ids"] = [
            str(feature["source_id"]) for feature in historical
        ]
        candidate["semantic_context_policy"] = (
            "current_bdtopo_composition_else_bdforet_v1_then_generic"
        )
        if selection_properties:
            enriched += 1
    return enriched


def _valid_building_exclusion_mask(
    footprints: list[tuple[str, Any, str, dict[str, Any]]],
    buildings: Mapping[str, Any],
    *,
    transform: Any,
) -> np.ndarray:
    candidates = buildings.get("candidates")
    if not isinstance(candidates, list):
        raise v1.PlacementInventoryError("building candidate array is invalid")
    valid_ids = {
        str(candidate.get("source_id"))
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("status") == "valid"
    }
    geometries = [geometry for source_id, geometry, _digest, _props in footprints if source_id in valid_ids]
    return v1._rasterize_geometries(geometries, transform=transform)


def _reconcile_candidate_statuses(family: dict[str, Any]) -> None:
    candidates = family.get("candidates")
    if not isinstance(candidates, list):
        raise v1.PlacementInventoryError("placement candidate array is invalid")
    counts = {
        status: sum(
            isinstance(candidate, Mapping) and candidate.get("status") == status
            for candidate in candidates
        )
        for status in ("valid", "ambiguous", "rejected")
    }
    family["source_count"] = len(candidates)
    family["valid_count"] = counts["valid"]
    family["ambiguous_count"] = counts["ambiguous"]
    family["rejected_count"] = counts["rejected"]
    family["placement_ready_count"] = counts["valid"]
    family["placement_blocked_count"] = counts["ambiguous"] + counts["rejected"]


def _remove_overlapping_buildings(
    footprints: list[tuple[str, Any, str, dict[str, Any]]],
    buildings: dict[str, Any],
) -> int:
    """Allow boundary contact, but never retain positive-area intersections."""

    geometry_by_source = {source_id: geometry for source_id, geometry, _digest, _props in footprints}
    candidates = buildings.get("candidates")
    if not isinstance(candidates, list):
        raise v1.PlacementInventoryError("building candidate array is invalid")
    valid = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("status") == "valid"
    ]
    valid.sort(
        key=lambda candidate: (
            -int(candidate.get("height_cm") or 0),
            -float(candidate.get("footprint_area_m2") or 0.0),
            str(candidate.get("source_id")),
        )
    )
    accepted: list[tuple[Mapping[str, Any], Any]] = []
    removed = 0
    for candidate in valid:
        geometry = geometry_by_source.get(str(candidate.get("source_id")))
        if geometry is None:
            candidate["status"] = "rejected"
            candidate["reason_codes"] = [
                *candidate.get("reason_codes", []),
                "building_footprint_geometry_missing_removed",
            ]
            removed += 1
            continue
        conflict = next(
            (
                kept
                for kept, kept_geometry in accepted
                if float(geometry.intersection(kept_geometry).area)
                > BUILDING_INTERIOR_OVERLAP_EPSILON_M2
            ),
            None,
        )
        if conflict is not None:
            candidate["status"] = "rejected"
            candidate["reason_codes"] = [
                *candidate.get("reason_codes", []),
                "building_positive_area_overlap_removed",
            ]
            candidate["conflicting_candidate_id"] = str(conflict["candidate_id"])
            removed += 1
            continue
        accepted.append((candidate, geometry))
    _reconcile_candidate_statuses(buildings)
    return removed


def _remove_trees_inside_buildings(
    trees: dict[str, Any],
    footprints: list[tuple[str, Any, str, dict[str, Any]]],
    buildings: Mapping[str, Any],
) -> int:
    valid_sources = {
        str(candidate.get("source_id"))
        for candidate in buildings.get("candidates", [])
        if isinstance(candidate, Mapping) and candidate.get("status") == "valid"
    }
    geometries = [
        geometry
        for source_id, geometry, _digest, _props in footprints
        if source_id in valid_sources
    ]
    candidates = trees.get("candidates")
    if not isinstance(candidates, list):
        raise v1.PlacementInventoryError("tree candidate array is invalid")
    removed = 0
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("status") != "valid":
            continue
        position = candidate.get("position_l93_m")
        if not isinstance(position, list) or len(position) != 2:
            candidate["status"] = "rejected"
            candidate["reason_codes"] = [
                *candidate.get("reason_codes", []),
                "tree_position_invalid_removed",
            ]
            removed += 1
            continue
        point = Point(float(position[0]), float(position[1]))
        if any(geometry.covers(point) for geometry in geometries):
            candidate["status"] = "rejected"
            candidate["reason_codes"] = [
                *candidate.get("reason_codes", []),
                "tree_inside_building_footprint_removed",
            ]
            removed += 1
    _reconcile_candidate_statuses(trees)
    return removed


def _fixed_only_context_assets(
    *,
    mnt_mm: np.ndarray,
    west: int,
    south: int,
    fixed_asset_placements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    empty = {role: [] for role in v1._CONTEXT_FEATURE_ROLES}
    payload = v1._context_asset_inventory(
        empty,
        mnt_mm=mnt_mm,
        west=west,
        south=south,
        fixed_asset_placements=fixed_asset_placements,
    )
    payload["automatic_feature_assets_disabled"] = True
    payload["automatic_feature_asset_policy"] = (
        "road_rail_hydro_geometry_never_implies_equipment"
    )
    return payload


def build_placement_inventory_v2(
    mnt_m: Any,
    mns_m: Any,
    *,
    tile_origin_l93_m: Sequence[float],
    zone_id: str,
    building_footprints: Iterable[Mapping[str, Any]] = (),
    context_masks: Mapping[str, Any] | None = None,
    context_geometries: Mapping[str, Iterable[Any]] | None = None,
    context_features: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    fixed_asset_placements: Sequence[Mapping[str, Any]] = (),
    native_mnt_05m: Any | None = None,
    native_mns_05m: Any | None = None,
) -> v1.PlacementResult:
    """Build the side-by-side factual placement profile for one 500 m tile."""

    if not isinstance(zone_id, str) or not zone_id.strip():
        raise v1.PlacementInventoryError("zone_id must be a non-empty string")
    zone_id = zone_id.strip()
    west, south = v1._origin(tile_origin_l93_m)
    try:
        normalized_fixed_assets = validate_projected_placements(
            fixed_asset_placements,
            tile_origin_l93_m=(west, south),
        )
    except FixedAssetPlacementError as error:
        raise v1.PlacementInventoryError(
            f"fixed asset placements are invalid: {error}"
        ) from error

    mnt_mm, mns_mm, hag_cm, source_diagnostics = _source_grid_v2(mnt_m, mns_m)
    masks_input = dict(context_masks or {})
    geometries_input = dict(context_geometries or {})
    unknown_masks = sorted(set(masks_input) - set(v1._CONTEXT_KEYS))
    unknown_geometries = sorted(set(geometries_input) - set(v1._CONTEXT_KEYS[1:]))
    if unknown_masks or unknown_geometries:
        raise v1.PlacementInventoryError(
            f"unknown context keys: {unknown_masks + unknown_geometries}"
        )

    transform = from_origin(
        west - v1.HALO_M,
        south + v1.TILE_SIZE_M + v1.HALO_M,
        v1.RESOLUTION_M,
        v1.RESOLUTION_M,
    )
    masks: dict[str, np.ndarray] = {}
    geometry_hashes: dict[str, str] = {}
    geometry_masks: dict[str, np.ndarray] = {}
    for key in v1._CONTEXT_KEYS:
        masks[key] = v1._mask(
            masks_input.get(key), name=key, default=(key == "vegetation")
        )
    for key in v1._CONTEXT_KEYS[1:]:
        geometries, geometry_hash = v1._geometry_context(
            geometries_input.get(key, ()), name=key
        )
        geometry_hashes[key] = geometry_hash
        geometry_masks[key] = v1._rasterize_geometries(geometries, transform=transform)

    vegetation_context_supplied = "vegetation" in masks_input or bool(
        geometries_input.get("vegetation")
    )
    if geometry_masks["vegetation"].any():
        if "vegetation" in masks_input:
            masks["vegetation"] |= geometry_masks["vegetation"]
        else:
            masks["vegetation"] = geometry_masks["vegetation"]
    for key in ("roads", "rail", "water"):
        masks[key] |= geometry_masks[key]

    footprints = v1._normalise_footprints(building_footprints)
    features, context_features_hash = v1._normalise_context_features(context_features)
    buildings, _all_footprints_mask, footprint_context_hash = v1._building_inventory(
        footprints,
        mnt_mm=mnt_mm,
        hag_cm=hag_cm,
        transform=transform,
        west=west,
        south=south,
    )
    buildings["detection_mode"] = "bdtopo_footprint_geometry_with_mns_mnt_height_v2"
    buildings["xy_geometry_authority"] = "bdtopo_footprint"
    buildings["morphology_only_instantiation"] = "forbidden"
    buildings["interior_overlap_policy"] = (
        "allow_boundary_contact_remove_lower_priority_positive_area_overlap"
    )
    buildings["interior_overlap_epsilon_m2"] = BUILDING_INTERIOR_OVERLAP_EPSILON_M2
    buildings["overlap_removed_count"] = _remove_overlapping_buildings(
        footprints, buildings
    )
    buildings["form_classification_policy"] = (
        "strict_measured_height_and_footprint_area_v1"
    )
    buildings["asset_failure_policy"] = (
        "measured_procedural_fallback_else_remove_with_reason"
    )

    vegetation_prior = (
        masks["vegetation"]
        if vegetation_context_supplied
        else np.zeros_like(masks["vegetation"])
    )
    valid_building_mask = _valid_building_exclusion_mask(
        footprints, buildings, transform=transform
    )
    exclusion = valid_building_mask | masks["roads"] | masks["rail"] | masks["water"]
    trees = v1._tree_inventory(
        mnt_mm=mnt_mm,
        hag_cm=hag_cm,
        vegetation_mask=vegetation_prior,
        exclusion_mask=exclusion,
        west=west,
        south=south,
    )
    trees["inside_building_removed_count"] = _remove_trees_inside_buildings(
        trees, footprints, buildings
    )
    trees["building_exclusion_policy"] = (
        "raster_exclusion_plus_exact_footprint_cover_removal"
    )
    trees["asset_failure_policy"] = (
        "measured_procedural_fallback_else_remove_with_reason"
    )
    trees["count_semantics"] = "estimated_individual_crowns_not_certified_tree_stems"
    trees["native_refinement_may_change_candidate_count"] = False
    trees["geometry_scale_policy"] = (
        "uniform_fit_inside_measured_crown_and_height_bounds"
    )
    trees["base_elevation_policy"] = (
        "highest_native_mnt_inside_peak_cell_bounded_to_15cm_clearance"
    )
    trees["asset_selection_policy"] = TREE_ASSET_SELECTION_POLICY
    for candidate in trees["candidates"]:
        candidate["support_elevation_mm"] = candidate["ground_elevation_mm"]
        candidate["geometry_scale_policy"] = (
            "uniform_fit_inside_measured_crown_and_height_bounds"
        )
        candidate["asset_selection_policy"] = TREE_ASSET_SELECTION_POLICY
    native = _native_pair_mm(native_mnt_05m, native_mns_05m)
    native_refined = 0
    native_status = "not_available"
    native_hashes: dict[str, str] = {}
    native_diagnostics: dict[str, int | float | str] = {}
    if native is not None:
        native_mnt_mm, native_mns_mm = native
        native_hashes = {
            "native_mnt_05m_mm_sha256": _sha256(
                np.asarray(native_mnt_mm, dtype="<i4").tobytes(order="C")
            ),
            "native_mns_05m_mm_sha256": _sha256(
                np.asarray(native_mns_mm, dtype="<i4").tobytes(order="C")
            ),
        }
        native_refined, native_diagnostics = _refine_tree_candidates_native_05m(
            trees,
            mnt_mm_05m=native_mnt_mm,
            mns_mm_05m=native_mns_mm,
            west=west,
            south=south,
        )
        native_status = "applied"
    trees["native_05m_refinement_count"] = native_refined
    trees["native_05m_refinement_resolution_m"] = (
        NATIVE_RESOLUTION_M if native_status == "applied" else None
    )
    trees["native_05m_source_resolution_m"] = (
        NATIVE_RESOLUTION_M if native is not None else None
    )
    trees["native_05m_refinement_status"] = native_status
    trees["native_05m_integrity"] = native_diagnostics
    trees["semantic_context_enrichment_count"] = _enrich_tree_semantics(
        trees,
        vegetation_features=features["vegetation"],
        forest_composition_features=features["forest_composition"],
    )

    context_assets = _fixed_only_context_assets(
        mnt_mm=mnt_mm,
        west=west,
        south=south,
        fixed_asset_placements=normalized_fixed_assets,
    )

    context_record = {
        "building_footprints_sha256": footprint_context_hash,
        "masks_sha256": {key: v1._mask_sha256(masks[key]) for key in v1._CONTEXT_KEYS},
        "geometries_sha256": geometry_hashes,
        "features_sha256": context_features_hash,
        "fixed_asset_placement_count": len(normalized_fixed_assets),
        "fixed_asset_placements_sha256": _sha256(
            canonical_fixed_asset_bytes(normalized_fixed_assets)
        ),
        "vegetation_context_supplied": vegetation_context_supplied,
        "automatic_context_feature_assets": "disabled",
    }
    contract_sha256 = _contract_sha256()
    sources = {
        "mnt_mm_sha256": _sha256(np.asarray(mnt_mm, dtype="<i4").tobytes(order="C")),
        "mns_mm_sha256": _sha256(np.asarray(mns_mm, dtype="<i4").tobytes(order="C")),
        "context_sha256": _sha256(_canonical_bytes(context_record)),
        "contract_sha256": contract_sha256,
        "algorithm_sha256": _sha256(ALGORITHM.encode("ascii")),
        **native_hashes,
    }
    build_id = _sha256(
        _canonical_bytes(
            {
                "zone_id": zone_id,
                "tile_origin_l93_m": [west, south],
                "sources": sources,
                "placement_profile": PLACEMENT_PROFILE,
            }
        )
    )
    hag_core = np.asarray(
        hag_cm[v1.CORE_START : v1.CORE_STOP, v1.CORE_START : v1.CORE_STOP],
        dtype="uint16",
    ).copy()
    inventory: dict[str, Any] = {
        # Kept intentionally for the existing scene assembler.  Contract and
        # algorithm below distinguish the experimental v2 semantics.
        "schema": v1.SCHEMA,
        "contract_schema": CONTRACT_SCHEMA,
        "algorithm": ALGORITHM,
        "placement_profile": PLACEMENT_PROFILE,
        "compatibility_inventory_schema": v1.SCHEMA,
        "build_id": build_id,
        "zone_id": zone_id,
        "tile_id": f"E{west}-N{south}",
        "crs": v1.CRS,
        "grid": {
            "resolution_m": v1.RESOLUTION_M,
            "native_refinement_resolution_m": (
                NATIVE_RESOLUTION_M if native is not None else None
            ),
            "processing_halo_m": v1.HALO_M,
            "processing_shape": [v1.PROCESSING_SIZE, v1.PROCESSING_SIZE],
            "core_shape": [v1.TILE_SIZE_M, v1.TILE_SIZE_M],
            "core_bounds_l93_m": [
                west,
                south,
                west + v1.TILE_SIZE_M,
                south + v1.TILE_SIZE_M,
            ],
            "row_order": "north_to_south",
            "ownership": "half_open",
        },
        "sources": sources,
        "context": context_record,
        "hag": {
            "schema": v1.HAG_SCHEMA,
            "dtype": "uint16",
            "unit": "centimetre",
            "nodata": v1.NODATA_UINT16,
            "minimum_cm": int(hag_core.min()),
            "maximum_cm": int(hag_core.max()),
            "raw_sha256": _sha256(
                np.asarray(hag_core, dtype="<u2").tobytes(order="C")
            ),
            **source_diagnostics,
        },
        "buildings": buildings,
        "trees": trees,
        "context_assets": context_assets,
    }
    inventory["inventory_sha256"] = _sha256(_canonical_bytes(inventory))
    validate_inventory_v2(inventory)
    return v1.PlacementResult(hag_core_cm=hag_core, inventory=inventory)


def validate_inventory_v2(inventory: Mapping[str, Any]) -> None:
    v1.validate_inventory(inventory)
    if (
        inventory.get("contract_schema") != CONTRACT_SCHEMA
        or inventory.get("algorithm") != ALGORITHM
        or inventory.get("placement_profile") != PLACEMENT_PROFILE
    ):
        raise v1.PlacementInventoryError("v2 placement identity is invalid")
    buildings = inventory.get("buildings")
    trees = inventory.get("trees")
    context_assets = inventory.get("context_assets")
    if not all(isinstance(value, Mapping) for value in (buildings, trees, context_assets)):
        raise v1.PlacementInventoryError("v2 placement families are incomplete")
    assert isinstance(buildings, Mapping)
    assert isinstance(trees, Mapping)
    assert isinstance(context_assets, Mapping)
    if (
        buildings.get("xy_geometry_authority") != "bdtopo_footprint"
        or buildings.get("morphology_only_instantiation") != "forbidden"
        or buildings.get("interior_overlap_policy")
        != "allow_boundary_contact_remove_lower_priority_positive_area_overlap"
        or buildings.get("interior_overlap_epsilon_m2")
        != BUILDING_INTERIOR_OVERLAP_EPSILON_M2
        or buildings.get("form_classification_policy")
        != "strict_measured_height_and_footprint_area_v1"
        or buildings.get("asset_failure_policy")
        != "measured_procedural_fallback_else_remove_with_reason"
        or trees.get("native_refinement_may_change_candidate_count") is not False
        or trees.get("count_semantics")
        != "estimated_individual_crowns_not_certified_tree_stems"
        or trees.get("geometry_scale_policy")
        != "uniform_fit_inside_measured_crown_and_height_bounds"
        or trees.get("base_elevation_policy")
        != "highest_native_mnt_inside_peak_cell_bounded_to_15cm_clearance"
        or trees.get("asset_selection_policy") != TREE_ASSET_SELECTION_POLICY
        or trees.get("building_exclusion_policy")
        != "raster_exclusion_plus_exact_footprint_cover_removal"
        or trees.get("asset_failure_policy")
        != "measured_procedural_fallback_else_remove_with_reason"
        or context_assets.get("automatic_feature_assets_disabled") is not True
    ):
        raise v1.PlacementInventoryError("v2 factual placement policy changed")
    for family, key in (
        (buildings, "overlap_removed_count"),
        (trees, "inside_building_removed_count"),
    ):
        value = family.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise v1.PlacementInventoryError(f"v2 {key} is invalid")
    for candidate in context_assets.get("candidates", []):
        if not isinstance(candidate, Mapping) or candidate.get("fixed_placement_id") is None:
            raise v1.PlacementInventoryError(
                "v2 context assets must come from an explicit fixed placement"
            )
    if not isinstance(trees.get("native_05m_refinement_count"), int):
        raise v1.PlacementInventoryError("v2 native tree refinement count is invalid")
    native_status = trees.get("native_05m_refinement_status")
    if native_status not in {"not_available", "applied"}:
        raise v1.PlacementInventoryError("v2 native tree refinement status is invalid")
    if (
        native_status != "applied"
        and trees.get("native_05m_refinement_count") != 0
    ):
        raise v1.PlacementInventoryError(
            "v2 rejected native refinement changed canonical candidates"
        )
    candidates = trees.get("candidates")
    if not isinstance(candidates, list):
        raise v1.PlacementInventoryError("v2 tree candidates are invalid")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise v1.PlacementInventoryError("v2 tree candidate is invalid")
        ground = candidate.get("ground_elevation_mm")
        support = candidate.get("support_elevation_mm")
        if (
            isinstance(ground, bool)
            or not isinstance(ground, int)
            or isinstance(support, bool)
            or not isinstance(support, int)
            or support < ground
            or support - ground > TREE_BASE_MAX_CLEARANCE_MM
        ):
            raise v1.PlacementInventoryError("v2 tree support elevation is invalid")
        if not isinstance(candidate.get("source_properties"), Mapping):
            raise v1.PlacementInventoryError("v2 tree semantic properties are invalid")
        if candidate.get("asset_selection_policy") != TREE_ASSET_SELECTION_POLICY:
            raise v1.PlacementInventoryError("v2 tree asset selection policy is invalid")


__all__ = [
    "ALGORITHM",
    "CONTRACT_SCHEMA",
    "NativeGridMisalignmentError",
    "PLACEMENT_PROFILE",
    "TREE_ASSET_SELECTION_POLICY",
    "build_placement_inventory_v2",
    "validate_inventory_v2",
]
