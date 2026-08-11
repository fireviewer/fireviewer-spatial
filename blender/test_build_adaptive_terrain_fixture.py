from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image
import pytest

from build_adaptive_terrain_fixture import (
    GRID_SHAPE,
    RECEIPT_SCHEMA,
    build_fixture,
)
from compact_hag import (
    quantize_hag_max_cm_from_canonical_mm,
    read_hag_max_2m,
)
from frustum_streaming import TerrainTileCatalog
from ground_material_contract import (
    GroundMaterialContractError,
    build_ground_material_bundle,
    validate_ground_material_contract,
)
from tile_package import validate_tile_done
from validate_adaptive_terrain_zone import (
    ZoneVisualQaError,
    _validate_tile_done_without_optional_dependencies,
)

try:
    from omniverse.adaptive_terrain_usd import validate_tile_usd_package
except ModuleNotFoundError:  # pragma: no cover - direct module test path
    from adaptive_terrain_usd import validate_tile_usd_package


D_TEST_ROOT = Path("D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest")
CONTRACT_ROOT = Path(__file__).parents[1] / "contracts" / "terrain" / "v1"


@pytest.fixture(scope="module")
def built_fixture():
    output = D_TEST_ROOT / f"adaptive-fixture-{uuid.uuid4().hex}"
    try:
        yield build_fixture(output)
    finally:
        if output.is_dir() and output.resolve().is_relative_to(D_TEST_ROOT.resolve()):
            shutil.rmtree(output)


def test_builds_four_contiguous_tiles_and_all_qualification_receipts(
    built_fixture,
) -> None:
    receipt = json.loads(built_fixture.receipt.read_text(encoding="utf-8"))

    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["status"] == "accepted_synthetic"
    assert receipt["source"]["tile_count"] == 4
    assert receipt["source"]["resolution_m"] == 2.0
    assert receipt["source"]["grid_shape"] == list(GRID_SHAPE)
    assert len(receipt["source"]["source_seam_checks"]) == 4
    assert receipt["geometry"]["fvtq_payload_count"] == 12
    assert receipt["geometry"]["stitch_masks_per_lod"] == list(range(16))
    assert receipt["geometry"]["stitch_variant_count"] == 192
    assert len(receipt["geometry"]["shared_edge_checks"]) == 12
    assert all(
        all(record["fvtq_lods_match"])
        and all(record["usd_lods_match"])
        and record["usd_root_match"]
        for record in receipt["geometry"]["bitwise_rebuild_checks"]
    )
    assert receipt["usd"]["validated_package_count"] == 4
    assert receipt["surface_composition"]["package_count"] == 4
    assert receipt["surface_composition"]["weights_sum"] == 255
    assert receipt["canonical_packages"]["tile_package_count"] == 4
    assert receipt["canonical_packages"]["tile_done_count"] == 4
    assert (
        receipt["canonical_packages"]["recipe_build_id"]
        != receipt["canonical_packages"]["build_id"]
    )
    assert receipt["canonical_packages"]["build_id"] == receipt["build_id"]
    catalog = json.loads(built_fixture.catalog.read_text(encoding="utf-8"))
    assert {tile["build_id"] for tile in catalog["tiles"]} == {receipt["build_id"]}
    material_contract = validate_ground_material_contract(
        built_fixture.material_contract
    )
    assert material_contract["schema"] == "fireviewer.ground-material-contract.v2"
    assert material_contract["material_model"] == "FireViewerGroundSurface_v2"
    assert material_contract["composition"]["grid_size_px"] == [500, 500]
    assert material_contract["composition"]["runtime_procedural_material"] == (
        "forbidden"
    )
    assert material_contract["composition"]["runtime_orthophoto"] == "forbidden"
    assert all(
        profile["surface_basis"] == "atlas_pbr"
        and profile["atlas_uv"] is not None
        and profile["physical_scale_m"] > 0
        and profile["variant_selection"] == "baked_profile_id"
        and profile["runtime_modulation"] == "none"
        for profile in material_contract["profile_table"]
    )
    assert len(material_contract["source_library"]["manifest_sha256"]) == 64
    assert len(material_contract["source_library"]["identity_sha256"]) == 64
    assert receipt["prohibited_dependencies"] == {
        "network_requests": 0,
        "orthophoto": False,
        "blender_runtime": False,
    }

    for tile_root in built_fixture.tile_roots:
        source = np.load(tile_root / "source" / "mnt-2m-mm.npy", allow_pickle=False)
        normal_halo = np.load(
            tile_root / "source" / "mnt-normal-halo-2m-mm.npy",
            allow_pickle=False,
        )
        mns = np.load(tile_root / "source" / "mns-2m-mm.npy", allow_pickle=False)
        mns_normal_halo = np.load(
            tile_root / "source" / "mns-normal-halo-2m-mm.npy",
            allow_pickle=False,
        )
        assert source.shape == GRID_SHAPE
        assert normal_halo.shape == (GRID_SHAPE[0] + 2, GRID_SHAPE[1] + 2)
        assert np.array_equal(normal_halo[1:-1, 1:-1], source)
        assert mns.shape == source.shape
        assert mns_normal_halo.shape == normal_halo.shape
        assert np.array_equal(mns_normal_halo[1:-1, 1:-1], mns)
        assert np.all(mns >= source)
        assert source.dtype == np.dtype("int32")
        assert mns.dtype == source.dtype
        assert all((tile_root / f"terrain-lod{lod}.fvtq").is_file() for lod in range(3))
        validate_tile_usd_package(tile_root)
        identifiers = np.asarray(Image.open(tile_root / "ground-profile-ids.png"))
        weights = np.asarray(Image.open(tile_root / "ground-profile-weights.png"))
        confidence = Image.open(tile_root / "ground-confidence.png")
        orientation = Image.open(tile_root / "ground-orientation.png")
        assert identifiers.shape == weights.shape == (500, 500, 4)
        assert np.all(weights.sum(axis=2, dtype=np.uint16) == 255)
        assert confidence.mode == orientation.mode == "L"
        assert confidence.size == orientation.size == (500, 500)
        assert not (tile_root / "surface-overlays.json.gz").exists()
        assert not (tile_root / "tile-composition.json.gz").exists()
        done = validate_tile_done(tile_root)
        assert done["schema"] == "fireviewer.tile.done.v3"
        assert done["surface_mapping"]["grid_size_px"] == [500, 500]
        assert done["surface_mapping"]["runtime_orthophoto"] == "forbidden"
        assert done["surface_mapping"]["runtime_procedural_material"] == "forbidden"
        assert len(done["normal_halo_sha256"]) == 64
        assert done["stitch_variants"]["available_masks"] == list(range(16))
        assert set(done["stitch_variants"]["lods"]) == {"lod0", "lod1", "lod2"}
        assert all(
            len(done["stitch_variants"]["lods"][f"lod{lod}"]) == 16 for lod in range(3)
        )
        assert done["ground_material"]["zone_path"].startswith(
            "shared/ground-material/"
        )
        assert (
            done["ground_material"]["source_library_manifest_sha256"]
            == (material_contract["source_library"]["manifest_sha256"])
        )
        assert done["inputs"]["surface_correspondence"]["path"] == (
            "surface-correspondence.json"
        )

    assert len(list(built_fixture.output_root.rglob("runtime-atlas/*.png"))) == 4
    assert not any(
        path
        for tile_root in built_fixture.tile_roots
        for path in tile_root.rglob("runtime-atlas/*.png")
    )


def test_v3_fixture_payloads_match_public_json_schemas(built_fixture) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")

    registry = referencing.Registry()
    schemas: dict[str, dict[str, object]] = {}
    for path in sorted(CONTRACT_ROOT.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        schemas[path.name] = schema
        registry = registry.with_resource(
            schema["$id"], referencing.Resource.from_contents(schema)
        )

    material_validator = jsonschema.Draft202012Validator(
        schemas["ground-material-contract.v2.schema.json"], registry=registry
    )
    package_validator = jsonschema.Draft202012Validator(
        schemas["tile-package.v3.schema.json"], registry=registry
    )
    done_validator = jsonschema.Draft202012Validator(
        schemas["tile-done.v3.schema.json"], registry=registry
    )

    material_validator.validate(
        material_payload := json.loads(
            built_fixture.material_contract.read_text(encoding="utf-8")
        )
    )
    assert material_payload["source_library"]["schema"] == (
        "fireviewer.clean-pbr-texture-library.v1"
    )
    assert (
        material_payload["runtime_shader"]["production_textured_runtime_qualified"]
        is False
    )

    missing_projection = copy.deepcopy(material_payload)
    missing_projection["profile_table"][0].pop("projection")
    with pytest.raises(jsonschema.ValidationError):
        material_validator.validate(missing_projection)

    missing_textures = copy.deepcopy(material_payload)
    missing_textures["profile_table"][0].pop("textures")
    with pytest.raises(jsonschema.ValidationError):
        material_validator.validate(missing_textures)

    legacy_source = copy.deepcopy(material_payload)
    legacy_source["source_library"]["schema"] = (
        "fireviewer.ground-surface-atlas-library.v3"
    )
    legacy_source["source_library"]["texture_contract_sha256"] = None
    with pytest.raises(jsonschema.ValidationError):
        material_validator.validate(legacy_source)

    for tile_root in built_fixture.tile_roots:
        package_payload = json.loads(
            (tile_root / "tile-package.v3.json").read_text(encoding="utf-8")
        )
        package_validator.validate(package_payload)
        done_validator.validate(
            json.loads((tile_root / "tile.done.v3.json").read_text(encoding="utf-8"))
        )

    legacy_package = copy.deepcopy(package_payload)
    legacy_package["ground_material"]["source_library_schema"] = (
        "fireviewer.ground-surface-atlas-library.v3"
    )
    legacy_package["ground_material"]["texture_contract_sha256"] = None
    with pytest.raises(jsonschema.ValidationError):
        package_validator.validate(legacy_package)

    for map_name, field_name, invalid_value in (
        ("profile_ids", "file", "ids.png"),
        ("profile_weights", "mode", "L8"),
        ("profile_weights", "encoding", "weights_not_normalized"),
        ("confidence", "file", "ground-confidence.exr"),
        ("orientation", "encoding", "directed_angle_0_to_2pi"),
    ):
        invalid_mapping = copy.deepcopy(package_payload)
        invalid_mapping["surface_mapping"][map_name][field_name] = invalid_value
        with pytest.raises(jsonschema.ValidationError):
            package_validator.validate(invalid_mapping)


def test_compact_hag_is_self_describing_and_small(built_fixture) -> None:
    for tile_root in built_fixture.tile_roots:
        path = tile_root / "hag-max-2m.tif"
        values, metadata = read_hag_max_2m(path)
        mnt_halo = np.load(
            tile_root / "source" / "mnt-normal-halo-2m-mm.npy",
            allow_pickle=False,
        )
        mns_halo = np.load(
            tile_root / "source" / "mns-normal-halo-2m-mm.npy",
            allow_pickle=False,
        )
        assert values.shape == (250, 250)
        assert np.array_equal(
            values,
            quantize_hag_max_cm_from_canonical_mm(mnt_halo, mns_halo),
        )
        assert metadata["resolution_m"] == 2.0
        assert metadata["unit"] == "centimetre"
        assert int(values.max()) in {0, 875, 1_250}
        assert path.stat().st_size < 20_000


def test_catalog_is_directly_consumable_by_frustum_streaming(built_fixture) -> None:
    manifest = json.loads(built_fixture.catalog.read_text(encoding="utf-8"))
    catalog = TerrainTileCatalog.from_manifest(manifest)

    assert len(catalog.tiles) == 4
    assert all(
        tile.cost(0).triangles > tile.cost(1).triangles for tile in catalog.tiles
    )
    assert all(
        tile.cost(1).triangles >= tile.cost(2).triangles for tile in catalog.tiles
    )
    assert any(
        tile.cost(1).triangles > tile.cost(2).triangles for tile in catalog.tiles
    )
    assert all(
        tile.bounds.maximum[2] >= tile.bounds.minimum[2] for tile in catalog.tiles
    )
    assert all(tile.stitch_masks == tuple(range(16)) for tile in catalog.tiles)
    assert all(
        len(tile.stitch_triangle_counts[lod]) == 16
        for tile in catalog.tiles
        for lod in range(3)
    )


def test_tile_done_detects_canonical_payload_corruption(
    built_fixture, tmp_path: Path
) -> None:
    corrupted = tmp_path / "corrupted-tile"
    shutil.copytree(built_fixture.tile_roots[0], corrupted)
    lod2 = corrupted / "terrain-lod2.fvtq"
    payload = bytearray(lod2.read_bytes())
    payload[-1] ^= 1
    lod2.write_bytes(payload)

    with pytest.raises(ValueError, match="digest mismatch|hash mismatch"):
        validate_tile_done(corrupted)


def test_dependency_free_tile_done_validator_matches_canonical_validator(
    built_fixture,
) -> None:
    for tile_root in built_fixture.tile_roots:
        canonical = validate_tile_done(tile_root)
        lightweight_done, lightweight_package = (
            _validate_tile_done_without_optional_dependencies(tile_root)
        )
        assert lightweight_done == canonical
        assert lightweight_package == json.loads(
            (tile_root / "tile-package.v3.json").read_text(encoding="utf-8")
        )


def test_dependency_free_tile_done_validator_rejects_hash_and_recipe_corruption(
    built_fixture, tmp_path: Path
) -> None:
    source = built_fixture.tile_roots[0]
    corrupted_output = tmp_path / "corrupted-output"
    shutil.copytree(source, corrupted_output)
    with (corrupted_output / "terrain-lod0.fvtq").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ZoneVisualQaError, match="output hash mismatch"):
        _validate_tile_done_without_optional_dependencies(corrupted_output)

    corrupted_recipe = tmp_path / "corrupted-recipe"
    shutil.copytree(source, corrupted_recipe)
    done_path = corrupted_recipe / "tile.done.v3.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["recipe_build_id"] = "0" * 64
    done_path.write_text(
        json.dumps(done, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ZoneVisualQaError, match="recipe_build_id differs"):
        _validate_tile_done_without_optional_dependencies(corrupted_recipe)


def test_material_v2_rejects_legacy_atlas_production_source(
    built_fixture, tmp_path: Path
) -> None:
    assert not (
        built_fixture.output_root
        / "shared"
        / "ground-material"
        / "ground-surface-atlas-catalog.json"
    ).exists()
    legacy_path = tmp_path / "ground-surface-atlas-catalog.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schema": "fireviewer.ground-surface-atlas-library.v3",
                "status": "synthetic_fixture_technical_only",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GroundMaterialContractError, match="legacy atlas libraries"):
        build_ground_material_bundle(legacy_path, tmp_path / "rejected-legacy")


def test_tile_v3_rejects_non_l8_confidence(built_fixture, tmp_path: Path) -> None:
    corrupted = tmp_path / "bad-confidence"
    shutil.copytree(built_fixture.tile_roots[0], corrupted)
    Image.fromarray(np.zeros((500, 500, 4), dtype=np.uint8), mode="RGBA").save(
        corrupted / "ground-confidence.png"
    )
    with pytest.raises(ValueError, match="500x500 L8"):
        validate_tile_done(corrupted)


def test_rejects_non_d_output_on_windows(tmp_path: Path) -> None:
    if tmp_path.drive.upper() == "D:":
        pytest.skip("pytest temporary root is already on D:")
    with pytest.raises(ValueError, match="must be on D"):
        build_fixture(tmp_path / "forbidden")
