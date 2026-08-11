from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
BLENDER_ROOT = REPOSITORY_ROOT / "blender"
if str(BLENDER_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_ROOT))

from adaptive_terrain_usd import (  # noqa: E402
    LOD_FILE_NAMES,
    ROOT_FILE_NAME,
    TerrainUsdError,
    export_tile_usd,
    validate_tile_usd_package,
)
from adaptive_terrain_quadtree import read_fvtq, write_fvtq  # noqa: E402
from build_adaptive_terrain_fixture import build_fixture  # noqa: E402


D_TEST_ROOT = Path("D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest")


@pytest.fixture(scope="module")
def source_fixture():
    root = D_TEST_ROOT / f"usd-source-{uuid.uuid4().hex}"
    try:
        yield build_fixture(root)
    finally:
        if root.is_dir() and root.resolve().is_relative_to(D_TEST_ROOT.resolve()):
            shutil.rmtree(root)


@pytest.fixture
def copied_zone(source_fixture):
    root = D_TEST_ROOT / f"usd-copy-{uuid.uuid4().hex}"
    shutil.copytree(source_fixture.output_root, root)
    try:
        yield root
    finally:
        if root.is_dir() and root.resolve().is_relative_to(D_TEST_ROOT.resolve()):
            shutil.rmtree(root)


def _tile(zone_root: Path) -> Path:
    return sorted((zone_root / "tiles").iterdir())[0]


def _export(zone_root: Path):
    tile_root = _tile(zone_root)
    return export_tile_usd(
        tuple(tile_root / f"terrain-lod{lod}.fvtq" for lod in range(3)),
        tile_root,
        tile_id=tile_root.name,
        zone_origin_l93_m=(700_000.0, 6_300_000.0),
        ground_material_contract=(
            zone_root
            / "shared"
            / "ground-material"
            / "ground-material-contract.v2.json"
        ),
        zone_package_root=zone_root,
    )


def test_exports_portable_deterministic_nested_payloads(source_fixture) -> None:
    roots = [D_TEST_ROOT / f"usd-portable-{uuid.uuid4().hex}" for _ in range(2)]
    try:
        for root in roots:
            shutil.copytree(source_fixture.output_root, root)
            _export(root)
        first_tile, second_tile = (_tile(root) for root in roots)
        names = (*LOD_FILE_NAMES, ROOT_FILE_NAME, "terrain-usd-package.v1.json")
        for name in names:
            assert (first_tile / name).read_bytes() == (second_tile / name).read_bytes()
        root_text = (first_tile / ROOT_FILE_NAME).read_text(encoding="utf-8")
        assert str(roots[0]) not in root_text
        assert 'terrainLod = "lod0"' in root_text
        assert "rel material:binding = </TerrainTile/GroundMaterial>" in root_text
        assert "../../shared/ground-material/ground-material.usda" in root_text
        assert "orthophoto" not in root_text.casefold()
        for lod, payload_path in enumerate(
            first_tile / name for name in LOD_FILE_NAMES
        ):
            payload = payload_path.read_text(encoding="utf-8")
            assert f"fireviewer:terrain_lod = {lod}" in payload
            assert (
                'fireviewer:stitch_delta_encoding = "fvtq-base-remove-add.v1"'
                in payload
            )
            assert "fireviewer:stitch_removed_triangle_offsets" in payload
            assert "fireviewer:stitch_removed_triangle_indices" in payload
            assert "fireviewer:stitch_replacement_triangle_offsets" in payload
            assert "fireviewer:stitch_replacement_face_vertex_indices" in payload
            assert 'subdivisionScheme = "none"' in payload
        manifest = validate_tile_usd_package(first_tile, zone_package_root=roots[0])
        assert manifest["primary_camera_allowed_lods"] == [0]
        tile_package = json.loads(
            (first_tile / "tile-package.v3.json").read_text(encoding="utf-8")
        )
        assert manifest["recipe_id"] == tile_package["recipe_id"]
        assert manifest["recipe_build_id"] == tile_package["recipe_build_id"]
        assert manifest["available_stitch_masks"] == list(range(16))
        assert manifest["tile_package_schema"] == "fireviewer.tile-package.v3"
        assert set(manifest["composition_assets"]) == {
            "ground_profile_ids",
            "ground_profile_weights",
            "ground_confidence",
            "ground_orientation",
        }
        assert "surface_overlays" not in root_text
        assert "ground-confidence.png" in root_text
        assert "ground-orientation.png" in root_text
        assert "procedural" not in root_text.casefold()
        assert (
            'fireviewer:runtime_shader_status = "pending_dedicated_mdl_validation"'
            in root_text
        )
        assert "fireviewer:production_textured_runtime_qualified = false" in root_text
        assert all(
            [
                record["mask"]
                for record in manifest["lod_metrics"][f"lod{lod}"]["stitch_variants"]
            ]
            == list(range(16))
            for lod in range(3)
        )
        assert (
            manifest["ground_material"]["visual_acceptance"] == "pending_human_review"
        )
        assert manifest["runtime_textured_operational"] is False
        assert manifest["preview_surface_policy"] == "diagnostic_untextured_only"
        material_contract = json.loads(
            source_fixture.material_contract.read_text(encoding="utf-8")
        )
        assert (
            material_contract["runtime_shader"]["production_textured_runtime_qualified"]
            is False
        )
        material_layer = (
            source_fixture.output_root
            / "shared"
            / "ground-material"
            / "ground-material.usda"
        ).read_text(encoding="utf-8")
        assert 'def Shader "GroundSurface"' not in material_layer
        assert "color3f inputs:diffuseColor = (1, 0, 1)" in material_layer
        assert "color3f inputs:diffuseColor = (0.18, 0.32, 0.12)" not in material_layer
        assert (
            manifest["ground_material"]["source_library_manifest_sha256"]
            == (material_contract["source_library"]["manifest_sha256"])
        )
    finally:
        for root in roots:
            if root.is_dir() and root.resolve().is_relative_to(D_TEST_ROOT.resolve()):
                shutil.rmtree(root)


def test_validation_rejects_corrupted_payload(copied_zone: Path) -> None:
    package = _export(copied_zone)
    package.lod_payloads[1].write_text("corrupt", encoding="utf-8")
    with pytest.raises(TerrainUsdError, match="hash mismatch"):
        validate_tile_usd_package(package.output_root, zone_package_root=copied_zone)


def test_export_rejects_mixed_tile_origins(copied_zone: Path) -> None:
    tile_root = _tile(copied_zone)
    lod2_path = tile_root / "terrain-lod2.fvtq"
    altered = replace(
        read_fvtq(lod2_path),
        tile_origin_mm=(700_500_000, 6_300_000_000),
    )
    write_fvtq(altered, lod2_path)
    with pytest.raises(TerrainUsdError, match="tile origin"):
        _export(copied_zone)


def test_manifest_declares_hashes_for_every_output(copied_zone: Path) -> None:
    package = _export(copied_zone)
    manifest = json.loads(package.manifest.read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == {ROOT_FILE_NAME, *LOD_FILE_NAMES}
    assert all(len(record["sha256"]) == 64 for record in manifest["outputs"].values())
    assert set(manifest["fvtq_inputs"]) == {
        "terrain-lod0.fvtq",
        "terrain-lod1.fvtq",
        "terrain-lod2.fvtq",
    }


@pytest.mark.parametrize("role", ["basecolor", "normal", "height", "orm"])
def test_validation_rejects_each_corrupted_shared_atlas(
    copied_zone: Path, role: str
) -> None:
    package = _export(copied_zone)
    atlas = copied_zone / "shared" / "ground-material" / "runtime-atlas" / f"{role}.png"
    payload = bytearray(atlas.read_bytes())
    payload[-1] ^= 1
    atlas.write_bytes(payload)
    with pytest.raises(TerrainUsdError, match="hash mismatch"):
        validate_tile_usd_package(package.output_root, zone_package_root=copied_zone)


@pytest.mark.parametrize(
    "name",
    [
        "ground-profile-ids.png",
        "ground-profile-weights.png",
        "ground-confidence.png",
        "ground-orientation.png",
    ],
)
def test_validation_rejects_corrupted_composition_map(
    copied_zone: Path, name: str
) -> None:
    package = _export(copied_zone)
    path = package.output_root / name
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(TerrainUsdError, match="Composition asset hash mismatch"):
        validate_tile_usd_package(package.output_root, zone_package_root=copied_zone)


def test_zone_contains_one_shared_atlas_and_no_tile_duplicates(
    source_fixture,
) -> None:
    atlas_files = list(source_fixture.output_root.rglob("runtime-atlas/*.png"))
    assert len(atlas_files) == 4
    assert {path.name for path in atlas_files} == {
        "basecolor.png",
        "normal.png",
        "height.png",
        "orm.png",
    }
    assert not any(
        path
        for tile in source_fixture.tile_roots
        for path in tile.rglob("runtime-atlas/*.png")
    )


def test_validation_rejects_orthophoto_disguised_as_composition_asset(
    copied_zone: Path,
) -> None:
    package = _export(copied_zone)
    manifest = json.loads(package.manifest.read_text(encoding="utf-8"))
    source = package.output_root / "ground-confidence.png"
    forbidden = package.output_root / "orthophoto-runtime.png"
    source.rename(forbidden)
    record = manifest["composition_assets"]["ground_confidence"]
    record["path"] = forbidden.name
    record["bytes"] = forbidden.stat().st_size
    record["sha256"] = hashlib.sha256(forbidden.read_bytes()).hexdigest()
    package.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(TerrainUsdError, match="package-relative"):
        validate_tile_usd_package(package.output_root, zone_package_root=copied_zone)
