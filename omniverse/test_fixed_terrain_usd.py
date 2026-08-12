from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from affine import Affine
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BLENDER_ROOT = REPOSITORY_ROOT / "blender"
for import_root in (REPOSITORY_ROOT, BLENDER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fixed_terrain_grid import compile_fixed_terrain, write_fixed_terrain  # noqa: E402
from orthophoto_ground_texture import (  # noqa: E402
    compile_aligned_window,
    serialize_tile_outputs,
    slice_tile,
)
from fixed_terrain_usd import (  # noqa: E402
    LOD_FILE_NAMES,
    MANIFEST_FILE_NAME,
    ROOT_FILE_NAME,
    FixedTerrainUsdError,
    export_fixed_terrain_usd,
    main,
    validate_fixed_terrain_usd_package,
)


ORIGIN = (700_000, 6_300_000)
BOUNDS = (*ORIGIN, ORIGIN[0] + 500, ORIGIN[1] + 500)
TEST_ROOT = (
    Path(
        os.environ.get(
            "FIREVIEWER_TEST_ROOT",
            "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest",
        )
    )
    / "fixed-terrain-usd"
)
FORBIDDEN = (
    "atlas",
    "orthophoto",
    "normalmap",
    "heightmap",
    "orm.png",
    "inputs:orm",
    '"orm"',
    "pbr",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_package(root: Path) -> Path:
    root.mkdir(parents=True)
    rows, columns = np.indices((261, 261), dtype=np.float64)
    halo = (
        140.0
        + columns * 0.013
        + rows * 0.009
        + ((columns - 130.0) ** 2 + (rows - 130.0) ** 2) * 0.00002
    )
    terrain = compile_fixed_terrain(
        halo[5:256, 5:256],
        source_halo_heights_m=halo,
        tile_origin_l93_m=ORIGIN,
    )
    write_fixed_terrain(terrain, root / "terrain.fvtg")

    height = width = 520
    image_rows, image_columns = np.indices((height, width), dtype=np.int32)
    rgb = np.stack(
        (
            (40 + image_columns // 3 + image_rows // 7) % 256,
            (80 + image_columns // 5 + image_rows // 2) % 256,
            (25 + image_columns // 11 + image_rows // 4) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    # The source is north-up: its first row is the north side of the tile.
    rgb[:10, :, :] = (210, 35, 25)
    rgb[-10:, :, :] = (25, 45, 210)
    window = compile_aligned_window(
        rgb,
        transform=Affine(1, 0, ORIGIN[0] - 10, 0, -1, ORIGIN[1] + 510),
        crs="EPSG:2154",
        core_bounds_l93_m=BOUNDS,
        orthophoto_source_manifest_sha256=_digest("source-manifest-r1"),
        orthophoto_revision="source-r1",
    )
    for name, payload in serialize_tile_outputs(slice_tile(window, BOUNDS)).items():
        (root / name).write_bytes(payload)
    return root


@pytest.fixture
def package_root() -> Path:
    root = TEST_ROOT / uuid4().hex
    try:
        yield _build_package(root)
    finally:
        if root.is_dir() and root.resolve().is_relative_to(TEST_ROOT.resolve()):
            shutil.rmtree(root)


def _export(root: Path):
    return export_fixed_terrain_usd(
        root / "terrain.fvtg",
        root / "ground-color.png",
        root / "ground-color.json",
        root,
        tile_id="FR-30-00001_700000_6300000",
        zone_origin_l93_m=(697_500, 6_297_500),
    )


def test_exports_one_texture_three_lods_with_exact_world_uv(package_root: Path) -> None:
    package = _export(package_root)
    manifest = validate_fixed_terrain_usd_package(package_root)

    assert package.root_stage.name == ROOT_FILE_NAME
    assert [path.name for path in package.lod_payloads] == list(LOD_FILE_NAMES)
    assert manifest["default_lod"] == 0
    assert manifest["material"] == {
        "texture": "ground-color.png",
        "texture_count": 1,
        "color_space": "sRGB",
        "surface_shader": "UsdPreviewSurface",
        "texture_shader": "UsdUVTexture",
        "roughness": 0.9,
        "metallic": 0,
    }

    root_text = package.root_stage.read_text(encoding="utf-8")
    assert 'terrainLod = "lod0"' in root_text
    assert 'prepend variantSets = "terrainLod"' in root_text
    assert 'fireviewer:texture_coordinates = "EPSG2154_u_east_v_north"' in root_text
    assert 'def Material "GroundMaterial"' not in root_text

    for lod, payload_path in enumerate(package.lod_payloads):
        payload = payload_path.read_text(encoding="utf-8")
        assert 'upAxis = "Z"' in payload
        assert "metersPerUnit = 1" in payload
        assert f"fireviewer:terrain_lod = {lod}" in payload
        assert 'def Mesh "Core"' in payload
        assert 'def Mesh "Skirt"' in payload
        assert 'visibility = "invisible"' in payload
        assert 'fireviewer:main_camera_visibility = "forbidden"' in payload
        assert "texCoord2f[] primvars:st" in payload
        assert "double2[] primvars:" not in payload
        assert "(0, 0)," in payload
        assert "(1, 1)," in payload
        assert payload.count("@ground-color.png@") == 1
        assert 'info:id = "UsdPreviewSurface"' in payload
        assert 'info:id = "UsdUVTexture"' in payload
        assert 'inputs:sourceColorSpace = "sRGB"' in payload
        assert "float inputs:roughness = 0.9" in payload
        assert "float inputs:metallic = 0" in payload
        assert "rel material:binding = </TerrainPayload/GroundMaterial>" in payload
        assert "</TerrainTile/" not in payload


def test_rebuild_and_relocation_are_byte_reproducible(package_root: Path) -> None:
    first_package = _export(package_root)
    first = {
        name: (package_root / name).read_bytes()
        for name in (*LOD_FILE_NAMES, ROOT_FILE_NAME, MANIFEST_FILE_NAME)
    }
    _export(package_root)
    assert first == {
        name: (package_root / name).read_bytes()
        for name in (*LOD_FILE_NAMES, ROOT_FILE_NAME, MANIFEST_FILE_NAME)
    }

    moved = TEST_ROOT / uuid4().hex
    try:
        shutil.copytree(first_package.output_root, moved)
        assert (
            validate_fixed_terrain_usd_package(moved)["build_id"]
            == json.loads(first[MANIFEST_FILE_NAME])["build_id"]
        )
        assert all(
            str(package_root) not in payload.decode("utf-8")
            for payload in first.values()
        )
    finally:
        if moved.is_dir() and moved.resolve().is_relative_to(TEST_ROOT.resolve()):
            shutil.rmtree(moved)


@pytest.mark.parametrize(
    "name",
    [
        "terrain.fvtg",
        "ground-color.png",
        "ground-color.json",
        ROOT_FILE_NAME,
        LOD_FILE_NAMES[1],
    ],
)
def test_validation_rejects_every_corrupted_input_or_output(
    package_root: Path, name: str
) -> None:
    _export(package_root)
    path = package_root / name
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)
    with pytest.raises(FixedTerrainUsdError):
        validate_fixed_terrain_usd_package(package_root)


def test_rejects_external_input_even_when_contents_match(package_root: Path) -> None:
    external_root = TEST_ROOT / uuid4().hex
    external_root.mkdir(parents=True)
    external = external_root / "terrain.fvtg"
    shutil.copy2(package_root / "terrain.fvtg", external)
    try:
        with pytest.raises(FixedTerrainUsdError, match="escapes"):
            export_fixed_terrain_usd(
                external,
                package_root / "ground-color.png",
                package_root / "ground-color.json",
                package_root,
                tile_id="FR-30-00001_700000_6300000",
                zone_origin_l93_m=(697_500, 6_297_500),
            )
    finally:
        if external_root.resolve().is_relative_to(TEST_ROOT.resolve()):
            shutil.rmtree(external_root)


def test_runtime_outputs_have_no_legacy_surface_system_tokens(
    package_root: Path,
) -> None:
    _export(package_root)
    runtime_files = (*LOD_FILE_NAMES, ROOT_FILE_NAME, MANIFEST_FILE_NAME)
    combined = "\n".join(
        (package_root / name).read_text(encoding="utf-8").casefold()
        for name in runtime_files
    )
    assert not [token for token in FORBIDDEN if token in combined]
    assert "ground-color.png" in combined
    assert "basecolor" not in combined
    assert "custom shader" not in combined


def test_rejects_texture_metadata_not_aligned_with_fvtg(package_root: Path) -> None:
    manifest_path = package_root / "ground-color.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bounds_l93_m"][0] += 500
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FixedTerrainUsdError, match="differs from its tile"):
        _export(package_root)


def test_nested_ground_assets_and_cli_stay_portable(
    package_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ground_root = package_root / "ground"
    ground_root.mkdir()
    for name in ("ground-color.png", "ground-color.json"):
        (package_root / name).replace(ground_root / name)

    assert (
        main(
            [
                "export",
                "--package-root",
                str(package_root),
                "--terrain",
                str(package_root / "terrain.fvtg"),
                "--ground-color",
                str(ground_root / "ground-color.png"),
                "--ground-manifest",
                str(ground_root / "ground-color.json"),
                "--tile-id",
                "FR-30-00001_700000_6300000",
                "--zone-origin-x",
                "697500",
                "--zone-origin-y",
                "6297500",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "verified"
    manifest = validate_fixed_terrain_usd_package(package_root)
    assert manifest["inputs"]["ground_color"]["path"] == "ground/ground-color.png"
    assert (
        manifest["inputs"]["ground_color_manifest"]["path"]
        == "ground/ground-color.json"
    )
    root_text = (package_root / ROOT_FILE_NAME).read_text(encoding="utf-8")
    lod_text = (package_root / LOD_FILE_NAMES[0]).read_text(encoding="utf-8")
    assert "@ground/ground-color.png@" in lod_text
    assert str(package_root) not in root_text + lod_text
