"""Experimental factual-placement profile for the RunPod v2 map builder.

This module deliberately leaves the production v1 inventory untouched.  It
keeps the v1 outer inventory schema so the existing OpenUSD scene assembler can
consume the result, but records a distinct v2 contract and algorithm identity.

The profile changes three things only:

* BD TOPO footprints author building XY geometry; MNT/MNS author ground/height.
* Tree count/status stays identical to the measured 1 m crown detector, while
  the position/ground/height of each candidate is refined inside its exact
  original 1 m peak cell from the native 0.5 m elevation pair when available.
* Road/rail/hydro features no longer create generic equipment instances.  Only
  validated fixed-coordinate context assets are instantiated by this profile.

No quota or thinning is introduced.  The legacy Lightning image does not import
this module and remains an independent fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import from_origin

import mns_mnt_placement_inventory as v1
from fixed_asset_placement import (
    FixedAssetPlacementError,
    validate_projected_placements,
)
from fixed_asset_placement import canonical_json_bytes as canonical_fixed_asset_bytes

CONTRACT_SCHEMA = "fireviewer.mns-mnt-placement-contract.v2"
ALGORITHM = "fireviewer.mns-mnt-placement-algorithm.v5"
PLACEMENT_PROFILE = "fireviewer.factual-placement-profile.v2"
NATIVE_RESOLUTION_M = 0.5
NATIVE_SOURCE_SIZE = 1040


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


def _refine_tree_candidates_native_05m(
    trees: dict[str, Any],
    *,
    mnt_mm_05m: np.ndarray,
    mns_mm_05m: np.ndarray,
    west: int,
    south: int,
) -> int:
    """Refine each existing tree inside its original 1 m peak cell only.

    The operation cannot create, delete, merge or move a candidate into another
    1 m crown cell.  It therefore improves sub-metre placement without changing
    the v1 quantity authority used by the validation run.
    """

    delta_mm = mns_mm_05m.astype("int64") - mnt_mm_05m.astype("int64")
    if int(delta_mm.min()) < -(v1.NEGATIVE_HAG_HARD_LIMIT_CM * 10):
        raise NativeGridMisalignmentError(
            "native MNS lies more than 100 cm below MNT; grids are misaligned"
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

        choices: list[tuple[int, float, float, int, int]] = []
        for row in (row0, row0 + 1):
            for column in (column0, column0 + 1):
                x_m = grid_west + (column + 0.5) * NATIVE_RESOLUTION_M
                y_m = grid_north - (row + 0.5) * NATIVE_RESOLUTION_M
                choices.append((int(hag_mm[row, column]), x_m, y_m, row, column))
        height_mm, x_m, y_m, row, column = min(
            choices,
            key=lambda item: (-item[0], item[1], item[2], item[3], item[4]),
        )
        ground_mm = int(mnt_mm_05m[row, column])
        height_cm = max(1, int((height_mm + 5) // 10))
        previous = candidate.get("position_l93_m")
        candidate["position_l93_m"] = [round(x_m, 3), round(y_m, 3)]
        candidate["ground_elevation_mm"] = ground_mm
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
        }
        refined += 1
    return refined


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

    mnt_mm, mns_mm, hag_cm, source_diagnostics = v1._source_grid(mnt_m, mns_m)
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
    _features, context_features_hash = v1._normalise_context_features(context_features)
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
    trees["count_semantics"] = "estimated_individual_crowns_not_certified_tree_stems"
    trees["native_refinement_may_change_candidate_count"] = False
    native = _native_pair_mm(native_mnt_05m, native_mns_05m)
    native_refined = 0
    native_status = "not_available"
    native_hashes: dict[str, str] = {}
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
        try:
            native_refined = _refine_tree_candidates_native_05m(
                trees,
                mnt_mm_05m=native_mnt_mm,
                mns_mm_05m=native_mns_mm,
                west=west,
                south=south,
            )
        except NativeGridMisalignmentError:
            # The canonical 1 m MNS/MNT pair has already passed the strict v1
            # integrity checks and authored the candidate set above.  A broken
            # optional 0.5 m pair must not invalidate that measured placement;
            # reject only the sub-metre refinement and record the decision.
            native_status = "rejected_misaligned_below_mnt"
        else:
            native_status = "applied"
    trees["native_05m_refinement_count"] = native_refined
    trees["native_05m_refinement_resolution_m"] = (
        NATIVE_RESOLUTION_M if native_status == "applied" else None
    )
    trees["native_05m_source_resolution_m"] = (
        NATIVE_RESOLUTION_M if native is not None else None
    )
    trees["native_05m_refinement_status"] = native_status

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
        or trees.get("native_refinement_may_change_candidate_count") is not False
        or trees.get("count_semantics")
        != "estimated_individual_crowns_not_certified_tree_stems"
        or context_assets.get("automatic_feature_assets_disabled") is not True
    ):
        raise v1.PlacementInventoryError("v2 factual placement policy changed")
    for candidate in context_assets.get("candidates", []):
        if not isinstance(candidate, Mapping) or candidate.get("fixed_placement_id") is None:
            raise v1.PlacementInventoryError(
                "v2 context assets must come from an explicit fixed placement"
            )
    if not isinstance(trees.get("native_05m_refinement_count"), int):
        raise v1.PlacementInventoryError("v2 native tree refinement count is invalid")
    native_status = trees.get("native_05m_refinement_status")
    if native_status not in {
        "not_available",
        "applied",
        "rejected_misaligned_below_mnt",
    }:
        raise v1.PlacementInventoryError("v2 native tree refinement status is invalid")
    if (
        native_status != "applied"
        and trees.get("native_05m_refinement_count") != 0
    ):
        raise v1.PlacementInventoryError(
            "v2 rejected native refinement changed canonical candidates"
        )


__all__ = [
    "ALGORITHM",
    "CONTRACT_SCHEMA",
    "NativeGridMisalignmentError",
    "PLACEMENT_PROFILE",
    "build_placement_inventory_v2",
    "validate_inventory_v2",
]
