from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
import pytest

from adaptive_terrain_quadtree import read_fvtq
from build_adaptive_terrain_fixture import build_fixture
from omniverse.adaptive_terrain_usd import author_lod_usda
from validate_adaptive_terrain_usd import (
    BlenderTerrainQaError,
    TRIPLANAR_ALGORITHM,
    TRIPLANAR_AXIS_BASES,
    TRIPLANAR_BLEND_EXPONENT,
    _fvtq_vertex_normals,
    compose_reference_pbr_maps,
    inspect_package,
    rasterize_fvtq_surface,
    validate_atlas_visual_receipt,
    validate_primary_terrain_aovs,
    validate_surface_library_visual_receipt,
)


D_TEST_ROOT = Path("D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="module")
def source_fixture():
    output = D_TEST_ROOT / f"blender-inspect-{uuid.uuid4().hex}"
    try:
        yield build_fixture(output)
    finally:
        if output.is_dir() and output.resolve().is_relative_to(D_TEST_ROOT.resolve()):
            shutil.rmtree(output)


@pytest.fixture
def copied_fixture(source_fixture):
    output = D_TEST_ROOT / f"blender-inspect-copy-{uuid.uuid4().hex}"
    shutil.copytree(source_fixture.output_root, output)
    try:
        yield output
    finally:
        if output.is_dir() and output.resolve().is_relative_to(D_TEST_ROOT.resolve()):
            shutil.rmtree(output)


def _tile(zone: Path) -> Path:
    return sorted((zone / "tiles").iterdir())[0]


def _asymmetric_pbr_atlases(
    *, normal_rgb: tuple[float, float, float] = (0.5, 0.5, 1.0)
) -> dict[str, np.ndarray]:
    axis = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    basecolor = np.zeros((64, 64, 4), dtype=np.float32)
    basecolor[:, :, 0] = axis[None, :]
    basecolor[:, :, 1] = axis[:, None]
    basecolor[:, :, 2] = 0.25 * axis[None, :] + 0.65 * axis[:, None]
    basecolor[:, :, 3] = 1.0
    normal = np.ones_like(basecolor)
    normal[:, :, :3] = normal_rgb
    height = np.zeros_like(basecolor)
    height[:, :, 0] = 0.7 * axis[None, :] + 0.3 * axis[:, None]
    height[:, :, 3] = 1.0
    orm = np.ones_like(basecolor)
    orm[:, :, 0] = 1.0
    orm[:, :, 1] = 0.2 + 0.6 * axis[:, None]
    orm[:, :, 2] = 0.1 * axis[None, :]
    return {
        "basecolor": basecolor,
        "normal": normal,
        "height": height,
        "orm": orm,
    }


def _single_profile_maps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    profile_ids = np.zeros((500, 500, 4), dtype=np.uint8)
    profile_weights = np.zeros_like(profile_ids)
    profile_weights[:, :, 0] = 255
    confidence = np.full((500, 500), 255, dtype=np.uint8)
    orientation = np.zeros((500, 500), dtype=np.uint8)
    return profile_ids, profile_weights, confidence, orientation


def _profile(*, projection: str | None) -> list[dict[str, object]]:
    result: dict[str, object] = {
        "index": 0,
        "id": "cliff_surface.asymmetric_fixture",
        "atlas_uv": {"offset": [0.0, 0.0], "scale": [1.0, 1.0]},
        "physical_scale_m": 1_000.0,
        "variant_selection": "baked_profile_id",
        "runtime_modulation": "none",
    }
    if projection is not None:
        result["projection"] = projection
    return [result]


def _terrain_surface(
    normal: tuple[float, float, float],
    *,
    output_size: int,
    elevation_m: float | np.ndarray,
    tile_origin: tuple[float, float] = (0.0, 0.0),
) -> dict[str, object]:
    vector = np.asarray(normal, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    normals = np.broadcast_to(vector, (output_size, output_size, 3)).copy()
    elevation = np.asarray(elevation_m, dtype=np.float64)
    if elevation.ndim == 0:
        elevation = np.full((output_size, output_size), float(elevation))
    return {
        "schema": "fireviewer.fvtq-surface-raster.v1",
        "tile_origin_l93_m": [float(tile_origin[0]), float(tile_origin[1])],
        "grid_size_px": [output_size, output_size],
        "stitch_mask": 0,
        "normal_formula": "normalize(-gradient_x_mm_per_4m,-gradient_y_mm_per_4m,4000)",
        "world_z_m": elevation,
        "world_normal": normals,
        "world_z_sha256": "1" * 64,
        "world_normal_sha256": "2" * 64,
        "covered_pixel_count": output_size * output_size,
    }


def _compose_single_triplanar(
    normal: tuple[float, float, float],
    *,
    elevation_m: float | np.ndarray,
    output_size: int = 16,
    orientation_value: int = 0,
    tile_origin: tuple[float, float] = (0.0, 0.0),
    normal_rgb: tuple[float, float, float] = (0.5, 0.5, 1.0),
) -> dict[str, object]:
    ids, weights, confidence, orientation = _single_profile_maps()
    orientation.fill(orientation_value)
    return compose_reference_pbr_maps(
        ids,
        weights,
        _asymmetric_pbr_atlases(normal_rgb=normal_rgb),
        _profile(projection="world_triplanar"),
        tile_origin_l93_m=tile_origin,
        output_size=output_size,
        ground_confidence=confidence,
        ground_orientation=orientation,
        terrain_surface=_terrain_surface(
            normal,
            output_size=output_size,
            elevation_m=elevation_m,
            tile_origin=tile_origin,
        ),
    )


def test_inspect_package_locks_default_lod_hashes_material_and_maps(
    source_fixture,
) -> None:
    inspected = inspect_package(source_fixture.tile_roots[0])
    assert inspected["selected_lod"] == 0
    assert inspected["metrics"]["vertex_count"] > 0
    assert inspected["ground_material_contract"]["profile_count"] == 72
    assert set(inspected["composition_paths"]) == {
        "ground_profile_ids",
        "ground_profile_weights",
        "ground_confidence",
        "ground_orientation",
    }


def test_inspect_package_rejects_corruption(copied_fixture: Path) -> None:
    tile = _tile(copied_fixture)
    (tile / "terrain-lod0.usda").write_text("corrupt", encoding="utf-8")
    with pytest.raises(BlenderTerrainQaError, match="hash mismatch"):
        inspect_package(tile)


def test_inspect_package_rejects_rehashed_non_lod0_default_tampering(
    copied_fixture: Path,
) -> None:
    tile = _tile(copied_fixture)
    stage = tile / "terrain-tile.usda"
    stage.write_text(
        stage.read_text(encoding="utf-8").replace(
            'string terrainLod = "lod0"', 'string terrainLod = "lod2"'
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = tile / "terrain-usd-package.v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = stage.read_bytes()
    manifest["outputs"]["terrain-tile.usda"] = {
        "bytes": len(content),
        "sha256": _sha256(content),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(BlenderTerrainQaError, match="root stage does not reproduce"):
        inspect_package(tile)


def test_atlas_visual_receipt_is_bound_to_exact_shared_catalog(
    source_fixture,
) -> None:
    inspected = inspect_package(source_fixture.tile_roots[0])
    # The canonical fixture now uses the clean-PBR generation.  Exercise the
    # retained legacy v2 receipt reader explicitly instead of mislabelling the
    # clean fixture as an accepted historic atlas.
    legacy_inspected = {
        **inspected,
        "manifest": {
            **inspected["manifest"],
            "ground_material": {
                **inspected["manifest"]["ground_material"],
                "source_library_schema": "fireviewer.ground-surface-atlas-library.v3",
            },
        },
    }
    receipt_path = source_fixture.output_root / "atlas-visual.accepted.v1.json"
    receipt = {
        "schema": "fireviewer.ground-atlas-visual-acceptance.v1",
        "status": "accepted_blender_visual",
        "source_library_identity_sha256": inspected["manifest"]["ground_material"][
            "source_library_identity_sha256"
        ],
        "profile_count": 72,
        "texture_count": 4,
        "scale_bands": ["micro", "meso", "macro"],
        "invalid_profile_count": 0,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    validated = validate_atlas_visual_receipt(receipt_path, legacy_inspected)
    assert validated["status"] == "accepted_blender_visual"
    assert (
        validated["source_library_identity_sha256"]
        == receipt["source_library_identity_sha256"]
    )
    assert len(validated["sha256"]) == 64

    receipt["source_library_identity_sha256"] = "0" * 64
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(BlenderTerrainQaError, match="another atlas"):
        validate_atlas_visual_receipt(receipt_path, legacy_inspected)


def test_clean_pbr_visual_receipt_is_bound_to_content_and_texture_contract(
    source_fixture,
) -> None:
    inspected = inspect_package(source_fixture.tile_roots[0])
    clean_inspected = json.loads(json.dumps(inspected, default=str))
    material = clean_inspected["manifest"]["ground_material"]
    material.update(
        {
            "source_library_schema": "fireviewer.clean-pbr-texture-library.v1",
            "source_library_identity_sha256": "1" * 64,
            "source_library_content_sha256": "2" * 64,
            "texture_contract_sha256": "3" * 64,
        }
    )
    receipt_path = source_fixture.output_root / "clean-pbr-visual.accepted.v1.json"
    receipt = {
        "schema": "fireviewer.clean-pbr-texture-visual-acceptance.v1",
        "status": "accepted_human_visual",
        "texture_contract_sha256": "3" * 64,
        "library_content_sha256": "2" * 64,
        "profile_count": 72,
        "atlas_roles": ["basecolor", "normal", "height", "orm"],
        "invalid_profile_count": 0,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    validated = validate_surface_library_visual_receipt(receipt_path, clean_inspected)
    assert validated["schema"] == receipt["schema"]
    assert validated["source_library_identity_sha256"] == "1" * 64
    assert validated["library_content_sha256"] == "2" * 64
    assert validated["texture_contract_sha256"] == "3" * 64

    receipt["library_content_sha256"] = "0" * 64
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(BlenderTerrainQaError, match="another surface library"):
        validate_surface_library_visual_receipt(receipt_path, clean_inspected)


@pytest.mark.parametrize(
    ("lod_values", "coverage_values", "message"),
    [
        ([0.0, 0.0], [0.0, 0.0], "coverage AOV"),
        ([0.0, float("nan")], [1.0, 1.0], "forbidden terrain LOD"),
        ([0.0, 1.0], [1.0, 1.0], "forbidden terrain LOD"),
        ([0.0, 0.0], [1.0, float("nan")], "coverage AOV"),
    ],
)
def test_primary_aov_gate_rejects_missing_nan_or_lower_lod(
    lod_values, coverage_values, message
) -> None:
    with pytest.raises(BlenderTerrainQaError, match=message):
        validate_primary_terrain_aovs(lod_values, coverage_values)


def test_primary_aov_gate_accepts_only_complete_lod0() -> None:
    result = validate_primary_terrain_aovs(
        np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)
    )
    assert result == {
        "terrain_pixel_count": 32,
        "invalid_lod_pixel_count": 0,
        "invalid_coverage_pixel_count": 0,
        "maximum_lod_absolute_error": 0.0,
    }


def test_primary_aov_gate_rejects_lod1_even_with_complete_coverage() -> None:
    with pytest.raises(BlenderTerrainQaError, match="forbidden terrain LOD"):
        validate_primary_terrain_aovs(
            np.ones(32, dtype=np.float32),
            np.ones(32, dtype=np.float32),
        )


def test_reference_material_samples_all_pbr_maps_in_lambert93_world_phase() -> None:
    profile_ids = np.zeros((100, 100, 4), dtype=np.uint8)
    profile_weights = np.zeros_like(profile_ids)
    profile_weights[:, :, 0] = 255
    x_ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    basecolor = np.zeros((64, 64, 4), dtype=np.float32)
    basecolor[:, :, 0] = x_ramp[None, :]
    basecolor[:, :, 1] = 0.25
    basecolor[:, :, 2] = 0.5
    basecolor[:, :, 3] = 1.0
    normal = np.zeros_like(basecolor)
    normal[:, :, :3] = (0.5, 0.5, 1.0)
    normal[:, :, 3] = 1.0
    height = np.zeros_like(basecolor)
    height[:, :, 0] = x_ramp[None, :]
    height[:, :, 3] = 1.0
    orm = np.ones_like(basecolor)
    orm[:, :, 1] = 0.7
    orm[:, :, 2] = 0.0
    atlases = {
        "basecolor": basecolor,
        "normal": normal,
        "height": height,
        "orm": orm,
    }
    profiles = [
        {
            "index": 0,
            "atlas_uv": {"offset": [0.0, 0.0], "scale": [1.0, 1.0]},
            "physical_scale_m": 1_000.0,
        }
    ]

    west = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlases,
        profiles,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        output_size=100,
    )
    east = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlases,
        profiles,
        tile_origin_l93_m=(700_500.0, 6_300_000.0),
        output_size=100,
    )

    assert west["world_projection"] == "EPSG:2154 metric XY"
    assert west["used_profile_indices"] == [0]
    assert set(west["derived_sha256"]) == {"basecolor", "normal", "height", "orm"}
    assert west["derived_sha256"] != east["derived_sha256"]
    assert float(west["maps"]["basecolor"][50, -1, 0]) < float(
        east["maps"]["basecolor"][50, 0, 0]
    )
    assert np.allclose(west["maps"]["normal"][:, :, :3], (0.5, 0.5, 1.0))
    assert np.allclose(west["maps"]["orm"][:, :, 1], 0.7)


def test_reference_material_rasterizes_narrow_vector_overlays_above_5m_grid() -> None:
    profile_ids = np.zeros((100, 100, 4), dtype=np.uint8)
    profile_weights = np.zeros_like(profile_ids)
    profile_weights[:, :, 0] = 255
    basecolor = np.zeros((8, 8, 4), dtype=np.float32)
    basecolor[:, :4] = (0.1, 0.7, 0.2, 1.0)
    basecolor[:, 4:] = (0.1, 0.2, 0.9, 1.0)
    normal = np.zeros_like(basecolor)
    normal[:, :, :3] = (0.5, 0.5, 1.0)
    normal[:, :, 3] = 1.0
    height = np.zeros_like(basecolor)
    height[:, :, 3] = 1.0
    orm = np.ones_like(basecolor)
    orm[:, :, 2] = 0.0
    profiles = [
        {
            "index": 0,
            "id": "natural_ground.grass",
            "atlas_uv": {"offset": [0.0, 0.0], "scale": [0.5, 1.0]},
            "physical_scale_m": 1_000.0,
        },
        {
            "index": 1,
            "id": "watercourse.stream",
            "atlas_uv": {"offset": [0.5, 0.0], "scale": [0.5, 1.0]},
            "physical_scale_m": 1_000.0,
        },
    ]
    overlays = {
        "schema": "fireviewer.surface-overlays.v1",
        "crs": "EPSG:2154",
        "feature_count": 1,
        "features": [
            {
                "feature_id": "hydro:test",
                "priority": 6,
                "profile_id": "watercourse.stream",
                "role": "hydro",
                "width_m": 10.0,
                "geometry_l93_m": {
                    "type": "LineString",
                    "coordinates": [
                        [700_000.0, 6_300_250.0],
                        [700_500.0, 6_300_250.0],
                    ],
                },
            }
        ],
    }

    result = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        {
            "basecolor": basecolor,
            "normal": normal,
            "height": height,
            "orm": orm,
        },
        profiles,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        output_size=100,
        surface_overlays=overlays,
    )

    assert result["used_profile_indices"] == [0, 1]
    assert result["surface_overlays"] == {
        "feature_count": 1,
        "applied_feature_count": 1,
        "covered_pixel_count": 200,
        "role_pixel_counts": {"hydro": 200},
    }
    assert float(result["maps"]["basecolor"][50, 50, 2]) > 0.8
    assert float(result["maps"]["basecolor"][10, 50, 1]) > 0.6


def test_v3_orientation_is_causal_and_world_projection_is_seam_continuous() -> None:
    profile_ids = np.zeros((500, 500, 4), dtype=np.uint8)
    profile_weights = np.zeros_like(profile_ids)
    profile_weights[:, :, 0] = 255
    confidence = np.full((500, 500), 211, dtype=np.uint8)
    orientation_x = np.zeros((500, 500), dtype=np.uint8)
    orientation_y = np.full((500, 500), 128, dtype=np.uint8)
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    basecolor = np.zeros((256, 256, 4), dtype=np.float32)
    basecolor[:, :, 0] = ramp[None, :]
    basecolor[:, :, 1] = ramp[:, None]
    basecolor[:, :, 3] = 1.0
    normal = np.zeros_like(basecolor)
    normal[:, :, :3] = (0.5, 0.5, 1.0)
    normal[:, :, 3] = 1.0
    height = basecolor.copy()
    orm = np.ones_like(basecolor)
    orm[:, :, 1] = 0.7
    orm[:, :, 2] = 0.0
    atlases = {
        "basecolor": basecolor,
        "normal": normal,
        "height": height,
        "orm": orm,
    }
    profiles = [
        {
            "index": 0,
            "id": "fixture.profile",
            "atlas_uv": {"offset": [0.0, 0.0], "scale": [1.0, 1.0]},
            "physical_scale_m": 1_000.0,
            "variant_selection": "baked_profile_id",
            "runtime_modulation": "none",
        }
    ]

    west = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlases,
        profiles,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        output_size=500,
        ground_confidence=confidence,
        ground_orientation=orientation_x,
    )
    north = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlases,
        profiles,
        tile_origin_l93_m=(700_000.0, 6_300_500.0),
        output_size=500,
        ground_confidence=confidence,
        ground_orientation=orientation_x,
    )
    east = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlases,
        profiles,
        tile_origin_l93_m=(700_500.0, 6_300_000.0),
        output_size=500,
        ground_confidence=confidence,
        ground_orientation=orientation_x,
    )
    rotated = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlases,
        profiles,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        output_size=64,
        ground_confidence=confidence,
        ground_orientation=orientation_y,
    )
    endpoint = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlases,
        profiles,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        output_size=64,
        ground_confidence=confidence,
        ground_orientation=np.full((500, 500), 255, dtype=np.uint8),
    )
    unrotated_64 = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlases,
        profiles,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        output_size=64,
        ground_confidence=confidence,
        ground_orientation=orientation_x,
    )

    assert west["mapping_schema"] == "fireviewer.ground-surface-mapping.v3"
    assert west["confidence"] == {
        "minimum": 211,
        "maximum": 211,
        "mean": 211.0,
        "shader_input": False,
    }
    assert np.max(
        np.abs(
            west["maps"]["basecolor"][:, -1, :2] - east["maps"]["basecolor"][:, 0, :2]
        )
    ) <= (2.0 / 255.0)
    assert np.max(
        np.abs(
            west["maps"]["basecolor"][0, :, :2] - north["maps"]["basecolor"][-1, :, :2]
        )
    ) <= (2.0 / 255.0)
    assert rotated["derived_sha256"] != unrotated_64["derived_sha256"]
    assert endpoint["derived_sha256"] == unrotated_64["derived_sha256"]


def test_world_xy_reference_path_is_bit_identical_with_explicit_projection() -> None:
    ids, weights, confidence, orientation = _single_profile_maps()
    atlases = _asymmetric_pbr_atlases()
    arguments = {
        "tile_origin_l93_m": (700_000.0, 6_300_000.0),
        "output_size": 32,
        "ground_confidence": confidence,
        "ground_orientation": orientation,
    }
    implicit = compose_reference_pbr_maps(
        ids,
        weights,
        atlases,
        _profile(projection=None),
        **arguments,
    )
    explicit = compose_reference_pbr_maps(
        ids,
        weights,
        atlases,
        _profile(projection="world_xy"),
        **arguments,
    )

    assert implicit["normal_space"] == explicit["normal_space"] == "TANGENT"
    assert implicit["derived_sha256"] == explicit["derived_sha256"]
    for role in ("basecolor", "normal", "height", "orm"):
        assert np.array_equal(implicit["maps"][role], explicit["maps"][role])


@pytest.mark.parametrize(
    ("surface_normal", "expected_normal_rgb", "z_is_causal"),
    [
        ((1.0, 0.0, 0.0), (1.0, 0.5, 0.5), True),
        ((-1.0, 0.0, 0.0), (0.0, 0.5, 0.5), True),
        ((0.0, 1.0, 0.0), (0.5, 1.0, 0.5), True),
        ((0.0, -1.0, 0.0), (0.5, 0.0, 0.5), True),
        ((0.0, 0.0, 1.0), (0.5, 0.5, 1.0), False),
        ((0.0, 0.0, -1.0), (0.5, 0.5, 0.0), False),
    ],
)
def test_triplanar_axes_are_causal_and_use_signed_right_handed_bases(
    surface_normal, expected_normal_rgb, z_is_causal
) -> None:
    low = _compose_single_triplanar(surface_normal, elevation_m=100.0)
    high = _compose_single_triplanar(surface_normal, elevation_m=350.0)

    assert low["normal_space"] == "WORLD"
    assert low["triplanar"]["axis_bases"] == TRIPLANAR_AXIS_BASES
    assert low["triplanar"]["pixel_count"] == 16 * 16
    assert np.allclose(
        low["maps"]["normal"][:, :, :3], expected_normal_rgb, atol=1.0e-6
    )
    assert (
        low["derived_sha256"]["basecolor"] != high["derived_sha256"]["basecolor"]
    ) is z_is_causal


def test_triplanar_open_gl_tangent_normal_is_transformed_to_world() -> None:
    # On a +X-facing projection, tangent +X follows signed basis U=+Y.
    result = _compose_single_triplanar(
        (1.0, 0.0, 0.0),
        elevation_m=125.0,
        normal_rgb=(1.0, 0.5, 0.5),
    )
    assert np.allclose(result["maps"]["normal"][:, :, :3], (0.5, 1.0, 0.5), atol=1.0e-6)


def test_triplanar_asymmetric_slope_orientation_and_hash_are_deterministic() -> None:
    size = 24
    axis = (np.arange(size, dtype=np.float64) + 0.5) * (500.0 / size)
    world_x, world_y = np.meshgrid(axis, axis[::-1], indexing="xy")
    elevation = 80.0 + 0.3 * world_x - 0.15 * world_y
    normal = (-0.3, 0.15, 1.0)
    first = _compose_single_triplanar(
        normal, elevation_m=elevation, output_size=size, orientation_value=0
    )
    repeated = _compose_single_triplanar(
        normal, elevation_m=elevation, output_size=size, orientation_value=0
    )
    rotated = _compose_single_triplanar(
        normal, elevation_m=elevation, output_size=size, orientation_value=128
    )

    assert first["derived_sha256"] == repeated["derived_sha256"]
    assert first["derived_sha256"] != rotated["derived_sha256"]
    assert first["triplanar"]["algorithm"] == TRIPLANAR_ALGORITHM
    assert first["triplanar"]["blend_exponent"] == TRIPLANAR_BLEND_EXPONENT
    assert first["triplanar"]["used_profile_indices"] == [0]
    assert first["triplanar"]["pixel_count"] == size * size
    assert first["triplanar"]["surface"]["covered_pixel_count"] == size * size
    unit_normal = np.asarray(normal, dtype=np.float64)
    unit_normal /= np.linalg.norm(unit_normal)
    neutral_world = (
        np.sign(unit_normal) * np.abs(unit_normal) ** TRIPLANAR_BLEND_EXPONENT
    )
    neutral_world /= np.linalg.norm(neutral_world)
    assert np.allclose(
        first["maps"]["normal"][0, 0, :3],
        neutral_world * 0.5 + 0.5,
        atol=1.0e-6,
    )


def test_triplanar_fails_closed_without_validated_fvtq_surface() -> None:
    ids, weights, confidence, orientation = _single_profile_maps()
    with pytest.raises(BlenderTerrainQaError, match="validated LOD0 FVTQ surface"):
        compose_reference_pbr_maps(
            ids,
            weights,
            _asymmetric_pbr_atlases(),
            _profile(projection="world_triplanar"),
            tile_origin_l93_m=(0.0, 0.0),
            output_size=16,
            ground_confidence=confidence,
            ground_orientation=orientation,
        )


def test_fvtq_surface_raster_is_deterministic_and_matches_usd_normal_formula(
    source_fixture,
) -> None:
    mesh = read_fvtq(source_fixture.tile_roots[0] / "terrain-lod0.fvtq")
    gradients = np.asarray(mesh.vertex_gradients_mm_per_4m, dtype=np.float64)
    expected = np.column_stack(
        (-gradients[:, 0], -gradients[:, 1], np.full(len(gradients), 4_000.0))
    )
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.array_equal(_fvtq_vertex_normals(mesh), expected)
    normal_block = (
        author_lod_usda(mesh)
        .decode("utf-8")
        .split("normal3f[] normals = ", 1)[1]
        .split(" (\n        interpolation", 1)[0]
    )
    authored_normals = np.asarray(
        [
            tuple(float(value) for value in match)
            for match in re.findall(
                r"\((-?[0-9.eE+]+), (-?[0-9.eE+]+), (-?[0-9.eE+]+)\)",
                normal_block,
            )
        ],
        dtype=np.float64,
    )
    assert authored_normals.shape == expected.shape
    assert np.allclose(authored_normals, expected, atol=1.0e-9, rtol=0.0)

    first = rasterize_fvtq_surface(mesh, stitch_mask=0, output_size=48)
    repeated = rasterize_fvtq_surface(mesh, stitch_mask=0, output_size=48)
    stitched = rasterize_fvtq_surface(mesh, stitch_mask=15, output_size=48)
    assert first["world_z_sha256"] == repeated["world_z_sha256"]
    assert first["world_normal_sha256"] == repeated["world_normal_sha256"]
    assert first["covered_pixel_count"] == 48 * 48
    assert stitched["covered_pixel_count"] == 48 * 48
    assert np.allclose(np.linalg.norm(first["world_normal"], axis=2), 1.0, atol=1.0e-12)


def test_neighbor_fvtq_edges_and_triplanar_world_phase_are_continuous(
    source_fixture,
) -> None:
    meshes = [
        read_fvtq(path / "terrain-lod0.fvtq") for path in source_fixture.tile_roots
    ]
    east_pair = None
    for west in meshes:
        for east in meshes:
            if (
                east.tile_origin_mm[0] - west.tile_origin_mm[0] == 500_000
                and east.tile_origin_mm[1] == west.tile_origin_mm[1]
            ):
                east_pair = (west, east)
                break
        if east_pair is not None:
            break
    assert east_pair is not None
    west_mesh, east_mesh = east_pair
    west_edge = west_mesh.edge_vertex_indices[1]
    east_edge = east_mesh.edge_vertex_indices[0]
    west_values = [
        (
            west_mesh.vertices[index][1],
            west_mesh.z_origin_mm + west_mesh.vertices[index][2],
            west_mesh.vertex_gradients_mm_per_4m[index],
        )
        for index in west_edge
    ]
    east_values = [
        (
            east_mesh.vertices[index][1],
            east_mesh.z_origin_mm + east_mesh.vertices[index][2],
            east_mesh.vertex_gradients_mm_per_4m[index],
        )
        for index in east_edge
    ]
    assert west_values == east_values

    normal = tuple(_fvtq_vertex_normals(west_mesh)[west_edge[len(west_edge) // 2]])
    west_origin = tuple(value / 1_000.0 for value in west_mesh.tile_origin_mm)
    east_origin = tuple(value / 1_000.0 for value in east_mesh.tile_origin_mm)
    west = _compose_single_triplanar(
        normal,
        elevation_m=100.0,
        output_size=500,
        tile_origin=west_origin,
    )
    east = _compose_single_triplanar(
        normal,
        elevation_m=100.0,
        output_size=500,
        tile_origin=east_origin,
    )
    assert np.max(
        np.abs(
            west["maps"]["basecolor"][:, -1, :3] - east["maps"]["basecolor"][:, 0, :3]
        )
    ) <= (2.0 / 63.0)
