from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest


OMNIVERSE_ROOT = Path(__file__).resolve().parent
CONTRACT_ROOT = OMNIVERSE_ROOT / "contracts" / "v2"
sys.path.insert(0, str(OMNIVERSE_ROOT))
sys.path.insert(0, str(CONTRACT_ROOT))

from scene_composition import (  # noqa: E402
    ALGORITHM_ID,
    CompositionContractError,
    build_portfolio_plan,
    canonical_json,
    validate_asset_catalog,
    validate_composition_contract,
)
from validate_contracts import load_json, validate_all  # noqa: E402


def _ready_catalog() -> dict[str, object]:
    assets = [
        {
            "stable_id": f"asset-{index:03d}",
            "usd_path": f"assets/asset-{index:03d}.usdc",
            "sha256": hashlib.sha256(f"asset-{index:03d}".encode()).hexdigest(),
            "status": "accepted",
            "kit_open": "passed",
            "roles": ["environment"],
            "contexts": ["universal"],
            "placeholder": False,
        }
        for index in range(1, 296)
    ]
    return {
        "schema": "fireviewer.omniverse-asset-catalog.v2",
        "catalog_id": "fireviewer-asset4sim-usd-r5",
        "status": "ready",
        "expected_asset_count": 295,
        "received_asset_count": 295,
        "assets": assets,
        "release_gates": {
            "all_usd_received": "passed",
            "sha256_inventory": "passed",
            "isolated_kit_open": "passed",
            "license_and_provenance": "passed",
        },
    }


def _ready_contract(catalog: dict[str, object]) -> dict[str, object]:
    contract = load_json(CONTRACT_ROOT / "examples" / "scene-composition.pending.json")
    contract["status"] = "ready"
    contract["asset_catalog"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        canonical_json(catalog).encode("utf-8")
    ).hexdigest()
    contract["release_gates"] = {
        "usd_catalog": "passed",
        "repeatability": "passed",
        "isolated_kit_open": "pending",
        "visual_acceptance": "pending",
        "download_publication": "blocked",
    }
    return contract


def test_pending_contracts_are_valid_but_planning_is_blocked() -> None:
    assert validate_all() == []
    catalog = load_json(CONTRACT_ROOT / "examples" / "asset-catalog.pending.json")
    contract = load_json(CONTRACT_ROOT / "examples" / "scene-composition.pending.json")

    with pytest.raises(CompositionContractError, match="blocked until all 295"):
        build_portfolio_plan(catalog, contract)


def test_ready_catalog_produces_the_same_complete_twenty_scene_plan() -> None:
    catalog = _ready_catalog()
    contract = _ready_contract(catalog)

    first = build_portfolio_plan(catalog, contract)
    second = build_portfolio_plan(copy.deepcopy(catalog), copy.deepcopy(contract))

    assert first == second
    assert first["algorithm"] == ALGORITHM_ID
    assert first["scene_count"] == 20
    assert first["asset_count"] == 295
    assert first["placeholder_substitution"] is False
    assert first["simplified_geometry_fallback"] is False
    assert [scene["scene_id"] for scene in first["scenes"]] == [
        f"SIM-{index:02d}" for index in range(1, 21)
    ]
    required = [
        stable_id
        for scene in first["scenes"]
        for stable_id in scene["required_asset_ids"]
    ]
    assert sorted(required) == [f"asset-{index:03d}" for index in range(1, 296)]
    assert len(required) == len(set(required)) == 295


def test_catalog_rejects_missing_unsorted_or_unsafe_assets() -> None:
    catalog = _ready_catalog()
    catalog["assets"] = catalog["assets"][:-1]  # type: ignore[index]
    catalog["received_asset_count"] = 294
    assert "a ready asset catalog must contain exactly 295 assets" in validate_asset_catalog(catalog)

    catalog = _ready_catalog()
    catalog["assets"][0]["usd_path"] = "../placeholder.usda"  # type: ignore[index]
    catalog["assets"][0]["placeholder"] = True  # type: ignore[index]
    errors = validate_asset_catalog(catalog)
    assert any("portable USD path" in error for error in errors)
    assert any("placeholder must be false" in error for error in errors)


def test_contract_rejects_any_simplification_or_count_reduction() -> None:
    contract = load_json(CONTRACT_ROOT / "examples" / "scene-composition.pending.json")
    contract["scene_count"] = 4
    contract["simplified_geometry_fallback"] = "allowed"
    contract["placeholder_policy"] = "allowed"

    errors = validate_composition_contract(contract)

    assert any("scene_count" in error for error in errors)
    assert any("simplified_geometry_fallback" in error for error in errors)
    assert any("placeholder_policy" in error for error in errors)


def test_catalog_hash_is_part_of_the_reproducibility_boundary() -> None:
    catalog = _ready_catalog()
    contract = _ready_contract(catalog)
    contract["asset_catalog"]["sha256"] = "0" * 64  # type: ignore[index]

    with pytest.raises(CompositionContractError, match="does not match"):
        build_portfolio_plan(catalog, contract)
