from __future__ import annotations

import json
from pathlib import Path

import pytest

from compare_validation_folders import (
    ValidationComparisonError,
    compare_validation_folders,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evidence(root: Path, *, provider: str, tree_x: float) -> Path:
    tile_id = "x0_y0"
    _write_json(
        root / "validation-summary.json",
        {
            "artifact_role": "validation_evidence_only_not_a_map",
            "provider": provider,
            "zone_id": "ZONE",
            "build_id": provider,
            "tile_count": 1,
            "source_revisions": {"elevation": "same"},
            "zone_counts": {
                "buildings": 1,
                "trees": 1,
                "context_assets": 0,
            },
            "tiles": [
                {
                    "tile_id": tile_id,
                    "origin_l93_m": [0, 0],
                }
            ],
            "viewer_receipt": {
                "representation": "complete_non_simplified_map",
                "viewer": {"byte_count": 100},
                "completeness": {
                    "policy": "fail_closed_exact_visual_scene",
                    "mesh_coverage": "complete",
                    "family_instance_counts": {
                        "buildings": 1,
                        "trees": 1,
                        "context_assets": 0,
                    },
                },
            },
        },
    )
    _write_json(
        root / "tiles" / tile_id / "placement-inventory.json",
        {
            "sources": {
                "mnt_mm_sha256": "1" * 64,
                "mns_mm_sha256": "2" * 64,
                "context_sha256": "3" * 64,
            },
            "trees": {
                "candidates": [
                    {
                        "candidate_id": "tree-1",
                        "status": "valid",
                        "position_l93_m": [tree_x, 0.5],
                        "height_cm": 1000,
                    }
                ]
            },
            "buildings": {
                "candidates": [
                    {
                        "candidate_id": "building-1",
                        "source_id": "building-source-1",
                        "status": "valid",
                        "anchor_l93_m": [10.0, 10.0],
                        "height_cm": 500,
                    }
                ]
            },
            "context_assets": {"candidates": []},
        },
    )
    return root


def test_compare_folder_native_evidence(tmp_path: Path) -> None:
    left = _evidence(tmp_path / "left", provider="lightning", tree_x=0.5)
    right = _evidence(tmp_path / "right", provider="runpod", tree_x=0.75)
    result = compare_validation_folders(left, right)
    assert result["same_tile_set"] is True
    assert result["all_canonical_1m_source_hashes_equal"] is True
    assert result["tiles"][0]["trees"]["xy_delta_m"]["p50"] == 0.25
    assert result["left"]["viewer"]["representation"] == "complete_non_simplified_map"


def test_zip_inside_validation_folder_is_rejected(tmp_path: Path) -> None:
    left = _evidence(tmp_path / "left", provider="lightning", tree_x=0.5)
    right = _evidence(tmp_path / "right", provider="runpod", tree_x=0.75)
    (left / "forbidden.zip").write_bytes(b"not-an-archive")
    with pytest.raises(ValidationComparisonError, match="ZIP"):
        compare_validation_folders(left, right)
