from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, box, mapping

from compile_tile_composition import (
    OVERLAY_SCHEMA,
    SCHEMA,
    compile_tile_composition,
    serialized_outputs,
    write_tile_composition,
)


def _profile(family: str, variant: str, application_mode: str) -> dict:
    return {
        "id": f"{family}.{variant}",
        "family": family,
        "variant": variant,
        "application_mode": application_mode,
    }


def _contracts() -> tuple[dict, dict]:
    variants = {
        "natural_ground": {
            "grass": ["landcover:herbaceous"],
            "forest": ["landcover:deciduous_forest"],
        },
        "agriculture_field": {
            "vineyard_a": ["agriculture:vineyard"],
            "vineyard_b": ["agriculture:vineyard"],
            "vineyard_c": ["agriculture:vineyard"],
        },
        "cliff_surface": {
            "granite_a": ["geology:granite", "terrain:cliff"],
        },
        "path_surface": {
            "path_a": ["transport:path"],
            "path_b": ["transport:path"],
            "path_c": ["transport:path"],
        },
        "road_surface": {
            "road_a": ["transport:road", "road:asphalt"],
            "road_b": ["transport:road", "road:asphalt"],
            "road_c": ["transport:road", "road:asphalt"],
        },
        "railway_bed": {
            "rail_a": ["transport:rail", "rail:active"],
            "rail_b": ["transport:rail", "rail:active"],
            "rail_c": ["transport:rail", "rail:active"],
        },
        "watercourse": {
            "water_a": ["hydro:persistent"],
            "water_b": ["hydro:persistent"],
            "water_c": ["hydro:persistent"],
        },
    }
    modes = {
        "natural_ground": "ground_blend",
        "agriculture_field": "directional_area",
        "cliff_surface": "slope_cliff_overlay",
        "path_surface": "linear_overlay",
        "road_surface": "linear_overlay",
        "railway_bed": "linear_overlay",
        "watercourse": "watercourse_overlay",
    }
    profiles = [
        _profile(family, variant, modes[family])
        for family, family_variants in variants.items()
        for variant in family_variants
    ]
    context = {
        "schema": "fireviewer.ground-context-contract.v1",
        "profile_bindings": {
            family: {"variant_tags": family_variants}
            for family, family_variants in variants.items()
        },
    }
    catalog = {
        "schema": "fireviewer.ground-surface-atlas-library.v3",
        "catalog_sha256": "catalog-sha256-test",
        "profiles": profiles,
    }
    return context, catalog


def _feature(
    feature_id: str,
    layer_id: str,
    geometry,
    properties: dict,
    **extra,
) -> dict:
    return {
        "feature_id": feature_id,
        "layer_id": layer_id,
        "geometry": mapping(geometry),
        "properties": properties,
        **extra,
    }


def _two_tile_features() -> list[dict]:
    coverage = box(0, 0, 1_000, 500)
    return [
        _feature(
            "landcover:global-grass",
            "landcover",
            coverage,
            {"code_cs": "CS2.2.1", "code_us": ""},
        ),
        _feature(
            "geology:global-granite",
            "geology",
            coverage,
            {"formation": "granite"},
        ),
        _feature(
            "parcel:vineyard-42",
            "agricultural_parcels",
            box(400, 100, 600, 250),
            {"cat_cult_p": "vigne"},
        ),
        _feature(
            "road:network-7",
            "roads",
            LineString([(0, 350), (1_000, 350)]),
            {"nature": "route revêtue", "width_m": 7.0},
        ),
        _feature(
            "river:downstream-9",
            "hydro_lines",
            LineString([(0, 425), (1_000, 425)]),
            {
                "persistance": "permanent",
                "width_m": 5.5,
                "flow_direction": "forward",
            },
        ),
        _feature(
            "land-parcel:vineyard-42",
            "land_parcels",
            box(400, 100, 600, 250),
            {"idu": "vineyard-42"},
        ),
    ]


def _overlays_with_role(composition, role: str) -> list[dict]:
    return [
        overlay
        for overlay in composition.overlays["features"]
        if overlay["role"] == role
    ]


def test_compilation_is_deterministic_and_weights_sum_to_255() -> None:
    context, catalog = _contracts()
    arguments = {
        "bounds_l93_m": [0, 0, 500, 500],
        "features": _two_tile_features(),
        "context_contract": context,
        "atlas_catalog": catalog,
        "contract_sha256": "composition-contract-sha256-test",
        "context_sha256": "global-context-sha256-test",
    }
    first = compile_tile_composition(**arguments)
    second = compile_tile_composition(**arguments)

    assert first.manifest == second.manifest
    assert first.overlays == second.overlays
    assert np.array_equal(first.profile_ids, second.profile_ids)
    assert np.array_equal(first.profile_weights, second.profile_weights)
    assert serialized_outputs(first) == serialized_outputs(second)
    assert first.manifest["schema"] == SCHEMA
    assert first.overlays["schema"] == OVERLAY_SCHEMA
    assert first.profile_ids.shape == (100, 100, 4)
    assert first.profile_ids.dtype == np.uint8
    assert first.profile_weights.dtype == np.uint8
    assert np.all(first.profile_weights.sum(axis=2, dtype=np.uint16) == 255)
    assert "tile_id" in first.manifest["forbidden_seed_inputs"]


def test_atlas_hash_is_an_explicit_seed_dependency() -> None:
    context, catalog = _contracts()
    common = {
        "bounds_l93_m": [0, 0, 500, 500],
        "features": _two_tile_features(),
        "context_contract": context,
        "contract_sha256": "composition-contract-sha256-test",
        "context_sha256": "global-context-sha256-test",
    }
    first = compile_tile_composition(atlas_catalog=catalog, **common)
    changed_catalog = copy.deepcopy(catalog)
    changed_catalog["catalog_sha256"] = "different-catalog-sha256-test"
    second = compile_tile_composition(atlas_catalog=changed_catalog, **common)
    changed_context = compile_tile_composition(
        atlas_catalog=catalog,
        **{**common, "context_sha256": "different-global-context-sha256-test"},
    )

    assert "atlas_catalog_sha256" in first.manifest["seed_inputs"]
    assert (
        first.manifest["seed_namespace_sha256"]
        != second.manifest["seed_namespace_sha256"]
    )
    assert first.manifest["context_sha256"] == second.manifest["context_sha256"]
    assert (
        _overlays_with_role(first, "road")[0]["uv_seed"]
        != _overlays_with_role(second, "road")[0]["uv_seed"]
    )
    assert (
        first.manifest["seed_namespace_sha256"]
        != changed_context.manifest["seed_namespace_sha256"]
    )
    assert (
        _overlays_with_role(first, "road")[0]["uv_seed"]
        != _overlays_with_role(changed_context, "road")[0]["uv_seed"]
    )
    assert all(
        forbidden in first.manifest["forbidden_seed_inputs"]
        for forbidden in ("tile_id", "path", "clock", "worker_order")
    )


def test_global_abscissa_orientation_and_uv_are_continuous_across_tiles() -> None:
    context, catalog = _contracts()
    common = {
        "context_contract": context,
        "atlas_catalog": catalog,
        "contract_sha256": "composition-contract-sha256-test",
        "context_sha256": "global-context-sha256-test",
    }
    west_features = _two_tile_features() + [
        _feature(
            "land-parcel:west-only-context",
            "land_parcels",
            box(10, 10, 20, 20),
            {"idu": "west-only"},
        )
    ]
    west = compile_tile_composition(
        bounds_l93_m=[0, 0, 500, 500], features=west_features, **common
    )
    east = compile_tile_composition(
        bounds_l93_m=[500, 0, 1_000, 500],
        features=_two_tile_features(),
        **common,
    )
    assert (
        west.manifest["context_feature_set_sha256"]
        != east.manifest["context_feature_set_sha256"]
    )
    assert (
        west.manifest["seed_namespace_sha256"] == east.manifest["seed_namespace_sha256"]
    )

    for role in ("road", "hydro"):
        west_segments = _overlays_with_role(west, role)
        east_segments = _overlays_with_role(east, role)
        assert max(segment["abscissa_m"][1] for segment in west_segments) == 500.0
        assert min(segment["abscissa_m"][0] for segment in east_segments) == 500.0
        west_edge = next(
            segment for segment in west_segments if segment["abscissa_m"][1] == 500.0
        )
        east_edge = next(
            segment for segment in east_segments if segment["abscissa_m"][0] == 500.0
        )
        assert west_edge["profile_id"] == east_edge["profile_id"]
        assert west_edge["uv_seed"] == east_edge["uv_seed"]
        assert (
            west_edge["uv_origin_l93_m"]
            == east_edge["uv_origin_l93_m"]
            == [0.0, west_edge["uv_origin_l93_m"][1]]
        )
        assert west_edge["orientation_deg"] == east_edge["orientation_deg"] == 0.0

    west_parcel = _overlays_with_role(west, "agriculture")[0]
    east_parcel = _overlays_with_role(east, "agriculture")[0]
    assert west_parcel["land_parcel_feature_ids"] == ["land-parcel:vineyard-42"]
    assert east_parcel["land_parcel_feature_ids"] == ["land-parcel:vineyard-42"]
    for key in (
        "feature_id",
        "profile_id",
        "orientation_deg",
        "uv_seed",
        "uv_origin_l93_m",
    ):
        assert west_parcel[key] == east_parcel[key]
    west_x = [
        coordinate[0]
        for ring in west_parcel["geometry_l93_m"]["coordinates"]
        for coordinate in ring
    ]
    east_x = [
        coordinate[0]
        for ring in east_parcel["geometry_l93_m"]["coordinates"]
        for coordinate in ring
    ]
    assert max(west_x) == min(east_x) == 500.0


def test_crossings_fail_closed_then_use_explicit_highest_priority_override() -> None:
    context, catalog = _contracts()
    features = _two_tile_features()
    features.append(
        _feature(
            "road:crossing-vertical",
            "roads",
            LineString([(250, 0), (250, 500)]),
            {"nature": "route revêtue", "width_m": 6.0},
        )
    )
    arguments = {
        "bounds_l93_m": [0, 0, 500, 500],
        "features": features,
        "context_contract": context,
        "atlas_catalog": catalog,
        "contract_sha256": "composition-contract-sha256-test",
        "context_sha256": "global-context-sha256-test",
    }
    with pytest.raises(ValueError, match="Missing crossing override"):
        compile_tile_composition(**arguments)

    composition = compile_tile_composition(
        **arguments,
        crossing_overrides=[
            {
                "transport_feature_id": "road:crossing-vertical",
                "hydro_feature_id": "river:downstream-9",
                "kind": "bridge",
            }
        ],
    )
    crossing = _overlays_with_role(composition, "crossing_override")[0]
    assert crossing["crossing_kind"] == "bridge"
    assert crossing["priority"] == max(
        overlay["priority"] for overlay in composition.overlays["features"]
    )
    assert crossing["geometry_l93_m"] == {
        "type": "Point",
        "coordinates": [250.0, 425.0],
    }


def test_clipped_linear_features_require_and_preserve_global_measurement() -> None:
    context, catalog = _contracts()
    common = {
        "context_contract": context,
        "atlas_catalog": catalog,
        "contract_sha256": "composition-contract-sha256-test",
        "context_sha256": "global-context-sha256-test",
    }

    def clipped_features(*, west: float, east: float, abscissa: float) -> list[dict]:
        coverage = box(west, 0, east, 500)
        return [
            _feature(
                "landcover:global-grass",
                "landcover",
                coverage,
                {"code_cs": "CS2.2.1", "code_us": ""},
            ),
            _feature(
                "geology:global-granite",
                "geology",
                coverage,
                {"formation": "granite"},
            ),
            _feature(
                "road:network-7",
                "roads",
                LineString([(west, 350), (east, 350)]),
                {
                    "nature": "route revêtue",
                    "width_m": 7.0,
                    "geometry_scope": "clipped_feature",
                    "network_chain_id": "road-chain:n7",
                    "global_abscissa_start_m": abscissa,
                    "global_uv_origin_l93_m": [0.0, 350.0],
                    "chain_direction": "forward",
                },
            ),
        ]

    west = compile_tile_composition(
        bounds_l93_m=[0, 0, 500, 500],
        features=clipped_features(west=0, east=500, abscissa=0),
        **common,
    )
    east = compile_tile_composition(
        bounds_l93_m=[500, 0, 1_000, 500],
        features=clipped_features(west=500, east=1_000, abscissa=500),
        **common,
    )
    west_segments = _overlays_with_role(west, "road")
    east_segments = _overlays_with_role(east, "road")
    west_edge = next(
        segment for segment in west_segments if segment["abscissa_m"][1] == 500.0
    )
    east_edge = next(
        segment for segment in east_segments if segment["abscissa_m"][0] == 500.0
    )
    assert west_edge["network_chain_id"] == east_edge["network_chain_id"]
    assert west_edge["global_segment_index"] == east_edge["global_segment_index"]
    assert west_edge["profile_id"] == east_edge["profile_id"]
    assert west_edge["uv_seed"] == east_edge["uv_seed"]
    assert west_edge["uv_origin_l93_m"] == east_edge["uv_origin_l93_m"] == [0.0, 350.0]

    missing_measurement = clipped_features(west=0, east=500, abscissa=0)
    del missing_measurement[2]["properties"]["global_abscissa_start_m"]
    with pytest.raises(ValueError, match="global_abscissa_start_m"):
        compile_tile_composition(
            bounds_l93_m=[0, 0, 500, 500],
            features=missing_measurement,
            **common,
        )


def test_crossing_override_hash_is_order_independent_and_kind_sensitive() -> None:
    context, catalog = _contracts()
    features = _two_tile_features() + [
        _feature(
            "road:crossing-a",
            "roads",
            LineString([(200, 0), (200, 500)]),
            {"nature": "route revêtue", "width_m": 6.0},
        ),
        _feature(
            "road:crossing-b",
            "roads",
            LineString([(300, 0), (300, 500)]),
            {"nature": "route revêtue", "width_m": 6.0},
        ),
    ]
    common = {
        "bounds_l93_m": [0, 0, 500, 500],
        "features": features,
        "context_contract": context,
        "atlas_catalog": catalog,
        "contract_sha256": "contract",
        "context_sha256": "context",
    }
    overrides = [
        {
            "transport_feature_id": "road:crossing-a",
            "hydro_feature_id": "river:downstream-9",
            "kind": "bridge",
        },
        {
            "transport_feature_id": "road:crossing-b",
            "hydro_feature_id": "river:downstream-9",
            "kind": "ford",
        },
    ]
    first = compile_tile_composition(**common, crossing_overrides=overrides)
    reordered = compile_tile_composition(
        **common, crossing_overrides=list(reversed(overrides))
    )
    changed = copy.deepcopy(overrides)
    changed[1]["kind"] = "culvert"
    different = compile_tile_composition(**common, crossing_overrides=changed)

    assert (
        first.manifest["crossing_overrides_sha256"]
        == reordered.manifest["crossing_overrides_sha256"]
    )
    assert serialized_outputs(first) == serialized_outputs(reordered)
    assert (
        first.manifest["crossing_overrides_sha256"]
        != different.manifest["crossing_overrides_sha256"]
    )


def test_hydro_surfaces_stay_vectorial_and_require_crossing_override() -> None:
    context, catalog = _contracts()
    features = _two_tile_features() + [
        _feature(
            "water-surface:pond-3",
            "hydro_surfaces",
            box(100, 50, 200, 100),
            {"persistance": "permanent"},
        ),
        _feature(
            "road:pond-crossing",
            "roads",
            LineString([(150, 0), (150, 200)]),
            {"nature": "route revêtue", "width_m": 4.0},
        ),
    ]
    arguments = {
        "bounds_l93_m": [0, 0, 500, 500],
        "features": features,
        "context_contract": context,
        "atlas_catalog": catalog,
        "contract_sha256": "composition-contract-sha256-test",
        "context_sha256": "global-context-sha256-test",
    }
    with pytest.raises(ValueError, match="Missing crossing override"):
        compile_tile_composition(**arguments)

    composition = compile_tile_composition(
        **arguments,
        crossing_overrides=[
            {
                "transport_feature_id": "road:pond-crossing",
                "hydro_feature_id": "water-surface:pond-3",
                "kind": "culvert",
            }
        ],
    )
    pond = next(
        overlay
        for overlay in _overlays_with_role(composition, "hydro")
        if overlay["feature_id"] == "water-surface:pond-3"
    )
    assert pond["geometry_l93_m"]["type"] == "Polygon"
    assert pond["orientation_deg"] is None
    assert pond["abscissa_m"] is None
    crossing = next(
        overlay
        for overlay in _overlays_with_role(composition, "crossing_override")
        if overlay["hydro_feature_id"] == "water-surface:pond-3"
    )
    assert crossing["crossing_kind"] == "culvert"
    assert crossing["abscissa_m"][1] is None
    assert crossing["geometry_l93_m"] == {
        "type": "Point",
        "coordinates": [150.0, 75.0],
    }


def test_writer_uses_reproducible_gzip_mtime_zero(tmp_path: Path) -> None:
    context, catalog = _contracts()
    composition = compile_tile_composition(
        bounds_l93_m=[0, 0, 500, 500],
        features=_two_tile_features(),
        context_contract=context,
        atlas_catalog=catalog,
        contract_sha256="composition-contract-sha256-test",
        context_sha256="global-context-sha256-test",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt_first = write_tile_composition(composition, first)
    receipt_second = write_tile_composition(composition, second)
    assert receipt_first == receipt_second
    assert write_tile_composition(composition, first) == receipt_first
    for name in (
        "ground-profile-ids.png",
        "ground-profile-weights.png",
        "surface-overlays.json.gz",
        "tile-composition.json.gz",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for name in ("surface-overlays.json.gz", "tile-composition.json.gz"):
        payload = (first / name).read_bytes()
        assert payload[4:8] == b"\0\0\0\0"
        assert json.loads(gzip.decompress(payload))["schema"] in {
            SCHEMA,
            OVERLAY_SCHEMA,
        }

    original_manifest = (first / "tile-composition.json.gz").read_bytes()
    (first / "tile-composition.json.gz").write_bytes(b"different")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_tile_composition(composition, first)
    assert (first / "tile-composition.json.gz").read_bytes() == b"different"
    (first / "tile-composition.json.gz").write_bytes(original_manifest)


def test_missing_width_orientation_flow_or_semantic_match_has_no_fallback() -> None:
    context, catalog = _contracts()
    base = _two_tile_features()

    no_land_parcel = [
        feature for feature in base if feature["layer_id"] != "land_parcels"
    ]
    with pytest.raises(ValueError, match="no approved land parcel link"):
        compile_tile_composition(
            bounds_l93_m=[0, 0, 500, 500],
            features=no_land_parcel,
            context_contract=context,
            atlas_catalog=catalog,
            contract_sha256="contract",
            context_sha256="global-context",
        )

    no_width = [dict(feature) for feature in base]
    no_width[3] = {**no_width[3], "properties": {"nature": "route revêtue"}}
    with pytest.raises(ValueError, match="width is missing"):
        compile_tile_composition(
            bounds_l93_m=[0, 0, 500, 500],
            features=no_width,
            context_contract=context,
            atlas_catalog=catalog,
            contract_sha256="contract",
            context_sha256="global-context",
        )

    no_flow = [dict(feature) for feature in base]
    no_flow[4] = {
        **no_flow[4],
        "properties": {"persistance": "permanent", "width_m": 5.5},
    }
    with pytest.raises(ValueError, match="flow direction"):
        compile_tile_composition(
            bounds_l93_m=[0, 0, 500, 500],
            features=no_flow,
            context_contract=context,
            atlas_catalog=catalog,
            contract_sha256="contract",
            context_sha256="global-context",
        )

    ambiguous_field = base + [
        _feature(
            "parcel:square",
            "agricultural_parcels",
            box(20, 20, 80, 80),
            {"cat_cult_p": "vigne"},
        )
    ]
    with pytest.raises(ValueError, match="orientation is ambiguous"):
        compile_tile_composition(
            bounds_l93_m=[0, 0, 500, 500],
            features=ambiguous_field,
            context_contract=context,
            atlas_catalog=catalog,
            contract_sha256="contract",
            context_sha256="global-context",
        )

    unsupported = [dict(feature) for feature in base]
    unsupported[1] = {
        **unsupported[1],
        "properties": {"formation": "formation inconnue"},
    }
    with pytest.raises(
        ValueError, match="Geology classification has no approved semantic match"
    ):
        compile_tile_composition(
            bounds_l93_m=[0, 0, 500, 500],
            features=unsupported,
            context_contract=context,
            atlas_catalog=catalog,
            contract_sha256="contract",
            context_sha256="global-context",
        )
