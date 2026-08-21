from __future__ import annotations

import json
from pathlib import Path

import pytest

from map_validation_folder import (
    MapValidationFolderError,
    build_validation_folder,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _inventory(tile_id: str) -> dict[str, object]:
    empty_family = {
        "valid_count": 0,
        "ambiguous_count": 0,
        "rejected_count": 0,
        "candidates": [],
    }
    return {
        "tile_id": tile_id,
        "algorithm": "test",
        "inventory_sha256": "0" * 64,
        "sources": {
            "mnt_mm_sha256": "1" * 64,
            "mns_mm_sha256": "2" * 64,
            "context_sha256": "3" * 64,
        },
        "buildings": dict(empty_family),
        "trees": dict(empty_family),
        "context_assets": dict(empty_family),
    }


def _job_root(tmp_path: Path) -> Path:
    root = tmp_path / "job"
    tiles = [
        {
            "tile_id": f"x{x}_y{y}",
            "origin_l93_m": [x * 500, y * 500],
        }
        for y in range(3)
        for x in range(3)
    ]
    _write_json(
        root / "zone-plan.json",
        {
            "tiles": tiles,
            "production_bounds_l93_m": [0, 0, 1500, 1500],
            "source_revisions": {
                "elevation": "test",
                "orthophoto": "test",
                "context": "test",
            },
        },
    )
    _write_json(
        root / "zone.done.json",
        {
            "zone_id": "ZONE",
            "build_id": "a" * 64,
            "tile_count": 9,
            "building_count": 0,
            "tree_count": 0,
            "context_asset_count": 0,
            "placeholder_instance_count": 0,
            "degraded_mns_tile_count": 0,
        },
    )
    for tile in tiles:
        tile_id = str(tile["tile_id"])
        package = root / "packages" / tile_id
        _write_json(
            package / "placement" / "placement-inventory.json",
            _inventory(tile_id),
        )
        _write_json(
            package / "simple-measured-tile-receipt.v1.json",
            {"tile_id": tile_id},
        )
    return root


def test_folder_native_evidence_contains_no_zip(tmp_path: Path) -> None:
    root = _job_root(tmp_path)
    folder, summary = build_validation_folder(
        root,
        provider="runpod",
        stage="placement",
        require_nine_tiles=True,
    )
    assert folder.is_dir()
    assert summary["artifact_role"] == "validation_evidence_only_not_a_map"
    assert summary["tile_count"] == 9
    assert not any(path.suffix.casefold() == ".zip" for path in folder.rglob("*"))
    assert (folder / "validation-summary.json").is_file()
    assert len(list((folder / "tiles").iterdir())) == 9


def test_viewer_evidence_requires_complete_tiled_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _job_root(tmp_path)
    _write_json(
        root / "viewer-tiled" / "viewer-tiled-scene.v1.json",
        {"status": "complete", "source": {"kind": "sealed"}},
    )
    _write_json(root / "viewer-tiled" / "catalog.json", {"schema": "test"})
    import build_tiled_viewer_package

    monkeypatch.setattr(
        build_tiled_viewer_package,
        "validate_tiled_viewer_package",
        lambda _root: (
            {"status": "complete", "source": {"kind": "sealed"}},
            {
                "catalog_path": "viewer-tiled/catalog.json",
                "completeness": {"mesh_coverage": "complete"},
            },
        ),
    )
    _write_json(
        root / "viewer-scene.v1.json",
        {
            "status": "complete",
            "viewer": {"sha256": "f" * 64, "byte_count": 123},
            "completeness": {
                "policy": "fail_closed_exact_visual_scene",
                "mesh_coverage": "complete",
                "family_instance_counts": {
                    "buildings": 0,
                    "trees": 0,
                    "context_assets": 0,
                },
            },
        },
    )
    folder, summary = build_validation_folder(
        root,
        provider="runpod",
        stage="viewer",
        require_nine_tiles=True,
    )
    assert summary["viewer_role"] == "complete_non_simplified_map_representation"
    assert (
        summary["viewer_receipt"]["representation"]
        == "complete_tiled_non_simplified_map"
    )
    assert summary["monolithic_viewer_oracle_receipt"] is not None
    assert (folder / "viewer-tiled" / "catalog.json").is_file()
    assert (folder / "viewer-scene.v1.json").is_file()


def test_placeholder_instances_fail_closed(tmp_path: Path) -> None:
    root = _job_root(tmp_path)
    receipt = json.loads((root / "zone.done.json").read_text(encoding="utf-8"))
    receipt["placeholder_instance_count"] = 1
    _write_json(root / "zone.done.json", receipt)
    with pytest.raises(MapValidationFolderError, match="placeholder"):
        build_validation_folder(
            root,
            provider="runpod",
            stage="placement",
            require_nine_tiles=True,
        )
