from __future__ import annotations

import json
from pathlib import Path

import pytest

from ground_context_binding import (
    ATLAS_SCHEMA,
    classify_feature,
    load_context_contract,
    load_runtime_contract,
    select_profile,
    validate_profile_bindings,
)


ROOT = Path(__file__).parent


def test_contract_binds_all_72_runtime_profiles() -> None:
    context = load_context_contract(ROOT / "ground_context_contract.v1.json")
    runtime = load_runtime_contract(ROOT / "ground_surface_runtime_contract.v3.json")
    bindings = validate_profile_bindings(context, runtime)
    assert len(bindings) == 72
    assert bindings["natural_ground.mediterranean_limestone"] == [
        "geology:limestone",
        "landcover:bare_soil",
    ]
    assert bindings["road_surface.asphalt_fine"] == [
        "transport:road",
        "road:asphalt",
        "road:fine",
    ]
    assert "tile_id" in runtime["determinism"]["forbidden_seed_inputs"]
    assert "fire_id" in runtime["determinism"]["forbidden_seed_inputs"]
    assert "global_feature_id" in runtime["determinism"]["seed_inputs"]


def test_official_attributes_are_classified_without_image_inference() -> None:
    assert classify_feature(
        "landcover", {"code_cs": "CS2.1.1.2", "code_us": "US1.2"}
    ) >= {"landcover:conifer_forest", "landcover:forest"}
    assert "geology:schist" in classify_feature(
        "geology", {"DESCR": "Micaschistes et schistes sombres"}
    )
    assert classify_feature(
        "roads", {"nature": "Route empierrée", "itineraire_vert": "DFCI"}
    ) >= {"transport:road", "road:unpaved", "road:gravel"}


def test_profile_selection_is_deterministic_and_fails_closed(tmp_path: Path) -> None:
    context = load_context_contract(ROOT / "ground_context_contract.v1.json")
    atlas = {
        "schema": ATLAS_SCHEMA,
        "profiles": [
            {
                "id": "natural_ground.mediterranean_limestone",
                "family": "natural_ground",
                "variant": "mediterranean_limestone",
            },
            {
                "id": "natural_ground.pine_duff",
                "family": "natural_ground",
                "variant": "pine_duff",
            },
        ],
    }
    selected = select_profile(
        context,
        atlas,
        family="natural_ground",
        semantic_tags={"geology:limestone", "landcover:bare_soil"},
        seed="FR-TEST:tile-1",
    )
    assert selected["variant"] == "mediterranean_limestone"
    with pytest.raises(ValueError, match="matches approved semantic context"):
        select_profile(
            context,
            atlas,
            family="natural_ground",
            semantic_tags={"unsupported:value"},
            seed="FR-TEST:tile-1",
        )


def test_missing_variant_binding_is_rejected(tmp_path: Path) -> None:
    context = load_context_contract(ROOT / "ground_context_contract.v1.json")
    runtime = load_runtime_contract(ROOT / "ground_surface_runtime_contract.v3.json")
    broken = json.loads(json.dumps(context))
    del broken["profile_bindings"]["natural_ground"]["variant_tags"][
        "mediterranean_limestone"
    ]
    with pytest.raises(ValueError, match="variant tags are incomplete"):
        validate_profile_bindings(broken, runtime)
