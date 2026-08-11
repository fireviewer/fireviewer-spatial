from __future__ import annotations

import pytest

from ground_surface_composition import (
    CATALOG_SCHEMA,
    compose_linear_schedule,
    compose_watercourse_schedule,
    select_cliff_profile,
)


def _catalog() -> dict:
    profiles = []
    for index in range(6):
        profiles.append(
            {
                "id": f"road_surface.road_{index}",
                "family": "road_surface",
                "variant": f"road_{index}",
                "application_mode": "linear_overlay",
            }
        )
    for index in range(5):
        profiles.append(
            {
                "id": f"watercourse.water_{index}",
                "family": "watercourse",
                "variant": f"water_{index}",
                "application_mode": "watercourse_overlay",
            }
        )
    for variant in ("limestone_strata", "karst_limestone", "layered_schist"):
        profiles.append(
            {
                "id": f"cliff_surface.{variant}",
                "family": "cliff_surface",
                "variant": variant,
                "application_mode": "slope_cliff_overlay",
            }
        )
    return {
        "schema": CATALOG_SCHEMA,
        "catalog_sha256": "test-catalog",
        "profiles": profiles,
    }


def test_linear_schedule_is_deterministic_continuous_and_non_repeating() -> None:
    catalog = _catalog()
    first = compose_linear_schedule(
        catalog,
        fire_id="FR-TEST",
        feature_id="road-42",
        length_m=2_000.0,
        family="road_surface",
    )
    second = compose_linear_schedule(
        catalog,
        fire_id="FR-TEST",
        feature_id="road-42",
        length_m=2_000.0,
        family="road_surface",
    )
    assert first == second
    assert len(first) == 8
    for index, segment in enumerate(first):
        assert segment["uv_origin_m"] == segment["start_m"]
        if index:
            assert segment["start_m"] == first[index - 1]["end_m"]
            assert segment["profile_id"] != first[index - 1]["profile_id"]
        if index >= 2:
            assert segment["profile_id"] != first[index - 2]["profile_id"]


def test_watercourse_schedule_keeps_downstream_uv_continuity() -> None:
    schedule = compose_watercourse_schedule(
        _catalog(),
        fire_id="FR-TEST",
        feature_id="river-7",
        length_m=620.0,
    )
    assert [segment["start_m"] for segment in schedule] == [0.0, 250.0, 500.0]
    assert schedule[-1]["end_m"] == 620.0
    assert all(segment["uv_direction"] == "downstream_tangent" for segment in schedule)


def test_cliff_profile_requires_supported_geology_without_fallback() -> None:
    profile = select_cliff_profile(
        _catalog(),
        geology="calcaire stratifié",
        fire_id="FR-TEST",
        tile_id="tile-1",
    )
    assert profile["variant"] == "limestone_strata"
    with pytest.raises(ValueError, match="Unsupported cliff geology"):
        select_cliff_profile(
            _catalog(),
            geology="unknown formation",
            fire_id="FR-TEST",
            tile_id="tile-1",
        )
