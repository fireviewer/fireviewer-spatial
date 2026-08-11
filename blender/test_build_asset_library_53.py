from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

import build_asset_library_53 as builder


CONTRACT_PATH = Path(__file__).with_name("asset_library_contract.v1.json")
SCHEMA_PATH = (
    Path(__file__).parents[1]
    / "contracts"
    / "terrain"
    / "v1"
    / "asset-library.v1.schema.json"
)
REPOSITORIES_ROOT = Path(__file__).parents[2]
REAL_GENERATED_ROOT = (
    REPOSITORIES_ROOT / "fireviewer-sdg" / "asset4sim" / "generated_hunyuan3d_v2"
)
REAL_REFERENCE_MANIFEST = REAL_GENERATED_ROOT / "reference-manifest.json"
REAL_BATCH_ROOT = REAL_GENERATED_ROOT / "review_batch_53"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture_assets() -> list[tuple[str, str]]:
    categories = (
        ("building", 24, "02_batiments/fixture"),
        ("tree", 18, "01_arbres/fixture"),
        (
            "road_equipment",
            8,
            "01_Lot_3D_T1_bordures_routieres_securite/fixture",
        ),
        ("vehicle", 2, "04_vehicules/fixture"),
        (
            "pasture_equipment",
            1,
            "05_Lot_3D_T5_parcelles_bocage_paturages/fixture",
        ),
    )
    result: list[tuple[str, str]] = []
    index = 0
    for category, count, prefix in categories:
        for category_index in range(count):
            index += 1
            asset_id = f"{index:012x}_{category}_{category_index:02d}"
            result.append((asset_id, f"{prefix}/{asset_id}.png"))
    return result


def _write_fixture(root: Path) -> tuple[Path, Path, dict[str, dict[str, object]]]:
    reference_root = root / "reference"
    generated_root = root / "generated"
    batch_root = generated_root / "review_batch_53"
    reference_records: list[dict[str, object]] = []
    conversion_results: list[dict[str, object]] = []
    glb_records: list[dict[str, object]] = []
    metadata: dict[str, dict[str, object]] = {}
    for index, (asset_id, relative) in enumerate(_fixture_assets()):
        source = reference_root.joinpath(*relative.split("/"))
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source:{asset_id}".encode())
        reference_records.append(
            {
                "asset_id": asset_id,
                "route": "hunyuan3d",
                "source": str(source),
                "source_relative": relative,
                "source_bytes": source.stat().st_size,
                "source_sha256": _sha256(source),
            }
        )

        glb = batch_root / "glb" / f"{asset_id}.glb"
        usd = batch_root / "usd" / f"{asset_id}.usd"
        texture = batch_root / "usd" / "textures" / f"{asset_id}.png"
        for path, prefix in ((glb, b"glb"), (usd, b"usd"), (texture, b"png")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(prefix + asset_id.encode())
        conversion = {
            "asset": asset_id,
            "source_glb": str(glb),
            "source_glb_sha256": _sha256(glb),
            "usd": str(usd),
            "usd_sha256": _sha256(usd),
            "structural_validation": {
                "texture": str(texture),
                "texture_sha256": _sha256(texture),
            },
            "passed": True,
        }
        conversion_results.append(conversion)
        _write_json(batch_root / "reports" / "usd" / f"{asset_id}-usd.json", conversion)
        glb_records.append(
            {
                "asset_id": "glb",
                "path": str(glb),
                "bytes": glb.stat().st_size,
                "sha256": _sha256(glb),
                "bounds": [[-1.0, -0.5, -0.25], [1.0, 0.5, 0.25]],
                "bounds_diagonal": 2.29128784747792,
                "passed": True,
            }
        )
        metadata[asset_id] = {
            "status": "inspected",
            "up_axis": "Y",
            "meters_per_unit": 0.001,
            "default_prim": f"/Fixture_{index:02d}",
        }

    reference_manifest = generated_root / "reference-manifest.json"
    _write_json(
        reference_manifest,
        {
            "schema_version": 1,
            "reference_root": str(reference_root),
            "asset_count": len(reference_records),
            "assets": reference_records,
        },
    )
    _write_json(
        batch_root / "reports" / "usd" / "usd-conversion-manifest.json",
        {
            "schema_version": 1,
            "asset_count": 53,
            "passed_count": 53,
            "failed_count": 0,
            "passed": True,
            "results": conversion_results,
        },
    )
    _write_json(
        batch_root / "reports" / "final-glb-validation-local.json",
        {
            "schema_version": 1,
            "expected_count": 53,
            "asset_count": 53,
            "passed_count": 53,
            "failed_count": 0,
            "passed": True,
            "assets": glb_records,
        },
    )
    return reference_manifest, batch_root, metadata


def test_builds_exact_53_catalogue_and_rejects_glb_report_identity(
    tmp_path: Path,
) -> None:
    assert tmp_path.drive.casefold() == "d:"
    reference_manifest, batch_root, metadata = _write_fixture(tmp_path)

    first = builder.build_asset_library(
        reference_manifest,
        batch_root,
        usd_metadata=metadata,
    )
    second = builder.build_asset_library(
        reference_manifest,
        batch_root,
        usd_metadata=metadata,
    )

    assert first == second
    assert first["asset_count"] == 53
    assert first["category_counts"] == {
        "building": 24,
        "pasture_equipment": 1,
        "road_equipment": 8,
        "tree": 18,
        "vehicle": 2,
    }
    assert {
        category: len(ids) for category, ids in first["selection_pools"].items()
    } == {
        "building": 24,
        "tree": 18,
    }
    assert all(asset["asset_id"] != "glb" for asset in first["assets"])
    assert all(
        asset["identity"]["rejected_reported_asset_id"] == "glb"
        for asset in first["assets"]
    )
    assert all(asset["usd_stage"]["up_axis"] == "Y" for asset in first["assets"])
    assert all(
        asset["qualification"]["dimensions"]["value_m"] is None
        and asset["qualification"]["ground_anchor"]["offset_m"] is None
        and asset["qualification"]["forward_axis"]["value"] is None
        and asset["qualification"]["scale"]["uniform_scale"] is None
        and asset["qualification"]["visual"]["accepted"] is False
        for asset in first["assets"]
    )
    assert all(
        artifact["path"]
        and ":" not in artifact["path"]
        and "\\" not in artifact["path"]
        for asset in first["assets"]
        for artifact in (
            asset["source"],
            asset["usd"],
            asset["texture"],
            asset["receipt"],
        )
    )
    jsonschema.Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    ).validate(first)
    assert builder.validate_asset_library(first)["asset_count"] == 53


def test_selection_is_stable_and_final_usage_is_fail_closed(tmp_path: Path) -> None:
    assert tmp_path.drive.casefold() == "d:"
    reference_manifest, batch_root, metadata = _write_fixture(tmp_path)
    library = builder.build_asset_library(
        reference_manifest,
        batch_root,
        usd_metadata=metadata,
    )

    first = builder.select_asset_for_candidate(
        library,
        category="building",
        zone="FR-30-00001",
        candidate="building-000123",
        rule_version="fireviewer.ledenon-building-rule.v1",
        usage="technical_pilot_non_final",
    )
    second = builder.select_asset_for_candidate(
        library,
        category="building",
        zone="FR-30-00001",
        candidate="building-000123",
        rule_version="fireviewer.ledenon-building-rule.v1",
        usage="technical_pilot_non_final",
    )
    changed_candidate = builder.select_asset_for_candidate(
        library,
        category="building",
        zone="FR-30-00001",
        candidate="building-000124",
        rule_version="fireviewer.ledenon-building-rule.v1",
        usage="technical_pilot_non_final",
    )

    assert first == second
    assert first["asset_id"] in library["selection_pools"]["building"]
    assert first["usage_status"] == "technical_pilot_non_final"
    assert first["visual_accepted"] is False
    assert first["selection_seed"] != changed_candidate["selection_seed"]
    with pytest.raises(builder.AssetLibraryBuildError, match="Final scene selection"):
        builder.select_asset_for_candidate(
            library,
            category="tree",
            zone="FR-30-00001",
            candidate="tree-000001",
            rule_version="fireviewer.ledenon-tree-rule.v1",
            usage="final_scene",
        )


def test_identity_mismatch_tamper_and_overwrite_fail_closed(tmp_path: Path) -> None:
    assert tmp_path.drive.casefold() == "d:"
    reference_manifest, batch_root, metadata = _write_fixture(tmp_path)
    library = builder.build_asset_library(
        reference_manifest,
        batch_root,
        usd_metadata=metadata,
    )

    output = tmp_path / "output" / "asset-library.v1.json"
    builder.write_asset_library(library, output)
    assert output.is_file()
    with pytest.raises(builder.AssetLibraryBuildError, match="overwrite"):
        builder.write_asset_library(library, output)

    tampered = copy.deepcopy(library)
    tampered["selection_pools"]["building"][0] = tampered["selection_pools"]["tree"][0]
    tampered["catalog_revision"] = builder.canonical_sha256(
        builder._catalog_revision_payload(tampered)
    )
    with pytest.raises(builder.AssetLibraryBuildError, match="pool membership"):
        builder.validate_asset_library(tampered)

    validation_path = batch_root / "reports" / "final-glb-validation-local.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["assets"][0]["asset_id"] = "wrong_identity"
    _write_json(validation_path, validation)
    with pytest.raises(
        builder.AssetLibraryBuildError, match="neither valid nor explicitly"
    ):
        builder.build_asset_library(
            reference_manifest,
            batch_root,
            usd_metadata=metadata,
        )


@pytest.mark.skipif(
    not REAL_REFERENCE_MANIFEST.is_file() or not REAL_BATCH_ROOT.is_dir(),
    reason="The sibling reviewed 53-asset production batch is not available",
)
def test_real_reviewed_batch_is_catalogued_53_of_53() -> None:
    library = builder.build_asset_library(REAL_REFERENCE_MANIFEST, REAL_BATCH_ROOT)

    assert library["asset_count"] == 53
    assert library["category_counts"]["building"] == 24
    assert library["category_counts"]["tree"] == 18
    assert (
        sum(asset["usd_stage"]["status"] == "inspected" for asset in library["assets"])
        == 53
    )
    assert {
        asset["identity"]["rejected_reported_asset_id"] for asset in library["assets"]
    } == {"glb"}
    jsonschema.Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    ).validate(library)


def test_rejects_c_drive_inputs() -> None:
    with pytest.raises(builder.AssetLibraryBuildError, match="stored on D"):
        builder.build_asset_library(
            Path("C:/fireviewer-forbidden/reference-manifest.json"),
            Path("D:/unused"),
        )
