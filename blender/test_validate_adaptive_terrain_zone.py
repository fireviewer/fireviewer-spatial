from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

import validate_adaptive_terrain_zone as zone_qa


BUILD_ID = "a" * 64
RECIPE_ID = "b" * 64
RECIPE_BUILD_ID = "c" * 64


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _png_rgba(path: Path, width: int, height: int, pixels: bytes) -> None:
    assert len(pixels) == width * height * 4

    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
        )

    scanlines = b"".join(
        b"\0" + pixels[row * width * 4 : (row + 1) * width * 4] for row in range(height)
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _tile_record(tile_id: str, grid_x: int, grid_y: int, index: int) -> dict:
    west = grid_x * 500.0
    south = grid_y * 500.0
    costs = {}
    for lod, triangles in enumerate((10 + index, 4, 1)):
        costs[f"lod{lod}"] = {
            "cpu_bytes": 100 + lod,
            "gpu_bytes": 200 + lod,
            "triangles": triangles,
            "sha256": hashlib.sha256(f"{tile_id}/lod{lod}".encode()).hexdigest(),
            "stitch_triangle_counts": [triangles] * 16,
        }
    return {
        "id": tile_id,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "bounds_l93_ngf_m": [
            west,
            south,
            100.0 + index,
            west + 500.0,
            south + 500.0,
            120.0 + index * 2,
        ],
        "build_id": BUILD_ID,
        "stitch_masks": list(range(16)),
        "resource_costs": costs,
    }


def _fixture(tmp_path: Path, *, bad_seam: bool = False) -> tuple[Path, Path, Path]:
    zone_root = tmp_path / "zone"
    records = [
        _tile_record("x700000_y6300000", 1400, 12600, 0),
        _tile_record("x700500_y6300000", 1401, 12600, 1),
        _tile_record("x700000_y6300500", 1400, 12601, 2),
        _tile_record("x700500_y6300500", 1401, 12601, 3),
    ]
    catalog = {
        "schema": zone_qa.CATALOG_SCHEMA,
        "crs": "EPSG:2154",
        "tiles": records,
    }
    catalog_path = zone_root / "terrain-tile-catalog.v1.json"
    _json(catalog_path, catalog)
    for index, record in enumerate(records):
        tile_root = zone_root / "tiles" / record["id"]
        tile_root.mkdir(parents=True)
        ids = bytes([index, 0, 0, 0]) * 10_000
        weights = bytes([255, 0, 0, 0]) * 10_000
        _png_rgba(tile_root / "ground-profile-ids.png", 100, 100, ids)
        _png_rgba(tile_root / "ground-profile-weights.png", 100, 100, weights)
        overlay = {
            "schema": "fireviewer.surface-overlays.v1",
            "features": [
                {
                    "role": "road",
                    "geometry_l93_m": {
                        "type": "LineString",
                        "coordinates": [[0.0, 0.0], [100.0 + index, 0.0]],
                    },
                }
            ],
        }
        (tile_root / "surface-overlays.json.gz").write_bytes(
            gzip.compress(json.dumps(overlay).encode())
        )

    seams = [
        "x700000_y6300000--x700500_y6300000",
        "x700000_y6300000--x700000_y6300500",
        "x700500_y6300000--x700500_y6300500",
        "x700000_y6300500--x700500_y6300500",
    ]
    if bad_seam:
        seams[0] = "x700000_y6300000--x700000_y6300000"
    qa_path = tmp_path / "qa.v1.json"
    _json(
        qa_path,
        {
            "status": "passed",
            "recipe_id": RECIPE_ID,
            "recipe_build_id": RECIPE_BUILD_ID,
            "build_id": BUILD_ID,
            "validated_tile_ids": [record["id"] for record in records],
            "seam_metrics": [
                {
                    "id": seam_id,
                    "maximum_height_gap_mm": rank,
                    "normal_mismatch_count": 0,
                    "stitch_signature_mismatch_count": 0,
                    "composition_failure_count": 0,
                }
                for rank, seam_id in enumerate(seams)
            ],
            "metrics": {"seam_count": 4},
        },
    )
    job = zone_qa.build_zone_visual_job(
        zone_id="FR-TEST",
        revision="r1",
        recipe_id=RECIPE_ID,
        recipe_build_id=RECIPE_BUILD_ID,
        build_id=BUILD_ID,
        zone_root=zone_root,
        catalog_path=catalog_path,
        qa_metrics_path=qa_path,
        output_root=tmp_path / "visual",
        resolution=512,
        maximum_seams=4,
        cpu_budget_bytes=10_000,
        gpu_budget_bytes=10_000,
        triangle_budget=10_000,
    )
    job_path = tmp_path / "zone-visual-job.v1.json"
    _json(job_path, job)
    return job_path, catalog_path, qa_path


def test_job_rejects_catalog_tampering(tmp_path: Path) -> None:
    job_path, catalog_path, _ = _fixture(tmp_path)
    catalog_path.write_bytes(catalog_path.read_bytes() + b" ")
    with pytest.raises(zone_qa.ZoneVisualQaError, match="catalog was modified"):
        zone_qa.inspect_zone_job(job_path, require_d=False, validate_packages=False)


def test_job_separates_recipe_build_from_final_build(tmp_path: Path) -> None:
    job_path, _, _ = _fixture(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["recipe_id"] == RECIPE_ID
    assert job["recipe_build_id"] == RECIPE_BUILD_ID
    assert job["build_id"] == BUILD_ID

    inspection = zone_qa.inspect_zone_job(
        job_path, require_d=False, validate_packages=False
    )
    plan = zone_qa.build_capture_plan(inspection)
    assert plan["recipe_build_id"] == RECIPE_BUILD_ID
    assert plan["build_id"] == BUILD_ID


def test_job_rejects_tile_done_recipe_build_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_path, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        zone_qa,
        "_validate_tile_done_without_optional_dependencies",
        lambda _root: (
            {
                "recipe_id": RECIPE_ID,
                "recipe_build_id": "d" * 64,
            },
            {
                "recipe_id": RECIPE_ID,
                "recipe_build_id": RECIPE_BUILD_ID,
            },
        ),
    )
    monkeypatch.setattr(
        zone_qa,
        "inspect_package",
        lambda root: {
            "manifest": {
                "tile_id": Path(root).name,
                "recipe_id": RECIPE_ID,
                "recipe_build_id": RECIPE_BUILD_ID,
                "primary_camera_allowed_lods": [0],
            },
            "selected_lod": 0,
        },
    )
    with pytest.raises(zone_qa.ZoneVisualQaError, match="identity/LOD contract"):
        zone_qa.inspect_zone_job(job_path, require_d=False, validate_packages=True)


def test_job_rejects_absent_tile_package(tmp_path: Path) -> None:
    job_path, _, _ = _fixture(tmp_path)
    missing = tmp_path / "zone" / "tiles" / "x700500_y6300500"
    for child in missing.iterdir():
        child.unlink()
    missing.rmdir()
    with pytest.raises(zone_qa.ZoneVisualQaError, match="Tile package is absent"):
        zone_qa.inspect_zone_job(job_path, require_d=False, validate_packages=False)


def test_job_rejects_mono_tile_seam(tmp_path: Path) -> None:
    job_path, _, _ = _fixture(tmp_path, bad_seam=True)
    with pytest.raises(zone_qa.ZoneVisualQaError, match="mono-tile seam"):
        zone_qa.inspect_zone_job(job_path, require_d=False, validate_packages=False)


def test_capture_plan_contains_full_grid_seams_and_obliques(tmp_path: Path) -> None:
    job_path, _, _ = _fixture(tmp_path)
    inspection = zone_qa.inspect_zone_job(
        job_path, require_d=False, validate_packages=False
    )
    plan = zone_qa.build_capture_plan(inspection)
    categories = [capture["category"] for capture in plan["captures"]]
    assert categories.count("orthographic_full_square") == 1
    assert categories.count("orthographic_grid_3x3") == 9
    assert categories.count("orthographic_worst_seam") == 4
    assert categories.count("oblique_relief_extreme") == 3
    assert categories.count("oblique_network_richness") == 1
    assert categories.count("oblique_dominant_surface") == 3
    seams = [
        capture
        for capture in plan["captures"]
        if capture["category"] == "orthographic_worst_seam"
    ]
    assert all(len(capture["tile_ids"]) == 2 for capture in seams)
    assert all(
        capture["selection_basis"]["render_contract"]
        == "exactly_two_adjacent_lod0_tiles"
        for capture in seams
    )
    assert plan["selection_contract"]["available_seams"] == 4
    assert plan["selection_contract"]["selected_worst_seams"] == 4
    assert plan["plan_sha256"] == zone_qa._canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


def test_technical_receipt_rejects_truncated_capture_plan(tmp_path: Path) -> None:
    capture = {
        "capture_id": "overview-full-square",
        "category": "orthographic_full_square",
        "projection": "orthographic",
        "tile_ids": ["tile-a"],
    }
    plan_basis = {
        "schema": zone_qa.CAPTURE_PLAN_SCHEMA,
        "zone_id": "FR-TEST",
        "revision": "r1",
        "recipe_id": RECIPE_ID,
        "recipe_build_id": RECIPE_BUILD_ID,
        "build_id": BUILD_ID,
        "catalog_sha256": "d" * 64,
        "qa_metrics_sha256": "e" * 64,
        "selection_contract": {
            "maximum_worst_seams": 0,
            "available_seams": 0,
            "selected_worst_seams": 0,
        },
        "captures": [capture],
    }
    plan = {**plan_basis, "plan_sha256": zone_qa._canonical_sha256(plan_basis)}
    plan_path = tmp_path / "zone-visual-capture-plan.v1.json"
    _json(plan_path, plan)
    technical_path = tmp_path / "zone-visual-technical-receipt.v1.json"
    _json(
        technical_path,
        {
            "schema": zone_qa.TECHNICAL_RECEIPT_SCHEMA,
            "status": zone_qa.TECHNICAL_STATUS,
            "production_visual_gate_passed": False,
            "human_visual_acceptance": "pending_exhaustive_review",
            "zone_id": "FR-TEST",
            "revision": "r1",
            "recipe_id": RECIPE_ID,
            "recipe_build_id": RECIPE_BUILD_ID,
            "build_id": BUILD_ID,
            "job_sha256": "f" * 64,
            "catalog_sha256": "d" * 64,
            "qa_metrics_sha256": "e" * 64,
            "capture_plan": zone_qa._artifact(plan_path, relative_to=tmp_path),
            "capture_plan_sha256": plan["plan_sha256"],
            "capture_count": 1,
            "captures": [],
        },
    )
    with pytest.raises(zone_qa.ZoneVisualQaError, match="exhaustive required set"):
        zone_qa.validate_technical_receipt(technical_path)


def test_technical_receipt_rehashes_capture_plan(tmp_path: Path) -> None:
    job_path, _, _ = _fixture(tmp_path)
    inspection = zone_qa.inspect_zone_job(
        job_path, require_d=False, validate_packages=False
    )
    plan = zone_qa.build_capture_plan(inspection)
    visual_root = tmp_path / "visual"
    plan_path = visual_root / "zone-visual-capture-plan.v1.json"
    _json(plan_path, plan)
    technical_path = visual_root / "zone-visual-technical-receipt.v1.json"
    _json(
        technical_path,
        {
            "schema": zone_qa.TECHNICAL_RECEIPT_SCHEMA,
            "status": zone_qa.TECHNICAL_STATUS,
            "production_visual_gate_passed": False,
            "human_visual_acceptance": "pending_exhaustive_review",
            "zone_id": "FR-TEST",
            "revision": "r1",
            "recipe_id": RECIPE_ID,
            "recipe_build_id": RECIPE_BUILD_ID,
            "build_id": BUILD_ID,
            "job_sha256": "f" * 64,
            "catalog_sha256": inspection.catalog_sha256,
            "qa_metrics_sha256": inspection.qa_metrics_sha256,
            "capture_plan": zone_qa._artifact(plan_path, relative_to=visual_root),
            "capture_plan_sha256": plan["plan_sha256"],
            "capture_count": len(plan["captures"]),
            "captures": [],
        },
    )
    plan_path.write_bytes(plan_path.read_bytes() + b" ")
    with pytest.raises(zone_qa.ZoneVisualQaError, match="artifact hash mismatch"):
        zone_qa.validate_technical_receipt(technical_path)


def test_budget_fails_closed_before_full_zone_import(tmp_path: Path) -> None:
    job_path, _, _ = _fixture(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["budget"]["triangles"] = 10
    _json(job_path, job)
    with pytest.raises(zone_qa.ZoneVisualQaError, match="exceeds reserved budget"):
        zone_qa.inspect_zone_job(job_path, require_d=False, validate_packages=False)


def test_primary_capture_aov_rejects_lod1_and_missing_coverage() -> None:
    assert (
        zone_qa.validate_zone_capture_aovs(
            np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
        )["invalid_lod_pixel_count"]
        == 0
    )
    with pytest.raises(zone_qa.ZoneVisualQaError, match="forbidden terrain LOD"):
        zone_qa.validate_zone_capture_aovs(
            np.array([0.0, 1.0]), np.ones(2, dtype=np.float32)
        )
    with pytest.raises(zone_qa.ZoneVisualQaError, match="coverage AOV"):
        zone_qa.validate_zone_capture_aovs(
            np.zeros(2, dtype=np.float32), np.array([1.0, 0.0])
        )


def test_root_stage_artifact_is_confined_present_and_rehashed(tmp_path: Path) -> None:
    zone_root = tmp_path / "zone"
    tile_root = zone_root / "tiles" / "tile-a"
    tile_root.mkdir(parents=True)
    stage_path = tile_root / "terrain-tile.usda"
    stage_path.write_text("#usda 1.0\n", encoding="utf-8")
    stage_artifact = zone_qa._artifact(stage_path, relative_to=tile_root)
    package = {
        "root_stage": stage_artifact["path"],
        "outputs": {
            stage_artifact["path"]: {
                "bytes": stage_artifact["bytes"],
                "sha256": stage_artifact["sha256"],
            }
        },
    }
    tile = zone_qa.ZoneTile(
        tile_id="tile-a",
        grid_x=0,
        grid_y=0,
        bounds=(0.0, 0.0, 0.0, 500.0, 500.0, 1.0),
        tile_root=tile_root,
        package=package,
        done={},
        lod0_cpu_bytes=1,
        lod0_gpu_bytes=1,
        lod0_triangles=1,
        lod0_sha256="0" * 64,
    )
    assert zone_qa._validated_tile_root_stage(tile) == stage_path.resolve()

    package["root_stage"] = "../outside.usda"
    with pytest.raises(zone_qa.ZoneVisualQaError, match="escapes"):
        zone_qa._validated_tile_root_stage(tile)

    sibling_stage = zone_root / "tiles" / "tile-b" / "terrain-tile.usda"
    sibling_stage.parent.mkdir(parents=True)
    sibling_stage.write_text("#usda 1.0\n", encoding="utf-8")
    package["root_stage"] = "../tile-b/terrain-tile.usda"
    with pytest.raises(zone_qa.ZoneVisualQaError, match="escapes"):
        zone_qa._validated_tile_root_stage(tile)

    package["root_stage"] = stage_artifact["path"]
    package["outputs"] = {}
    with pytest.raises(zone_qa.ZoneVisualQaError, match="output record is missing"):
        zone_qa._validated_tile_root_stage(tile)

    package["outputs"] = {
        stage_artifact["path"]: {
            "bytes": stage_artifact["bytes"],
            "sha256": stage_artifact["sha256"],
        }
    }
    package["root_stage"] = stage_artifact["path"]
    stage_path.unlink()
    with pytest.raises(zone_qa.ZoneVisualQaError, match="absent"):
        zone_qa._validated_tile_root_stage(tile)

    stage_path.write_text("#usda 1.0\n", encoding="utf-8")
    stage_artifact = zone_qa._artifact(stage_path, relative_to=tile_root)
    package["root_stage"] = stage_artifact["path"]
    package["outputs"] = {
        stage_artifact["path"]: {
            "bytes": stage_artifact["bytes"],
            "sha256": stage_artifact["sha256"],
        }
    }
    stage_path.write_text("#usda 1.0\n# tampered\n", encoding="utf-8")
    with pytest.raises(zone_qa.ZoneVisualQaError, match="hash mismatch"):
        zone_qa._validated_tile_root_stage(tile)


def _capture_receipt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for file_name in zone_qa.CAPTURE_ARTIFACT_NAMES.values():
        (root / file_name).write_bytes(file_name.encode())
    receipt = {
        "schema": zone_qa.CAPTURE_RECEIPT_SCHEMA,
        "status": "rendered_technical",
        "capture_id": "seam-01-test",
        "category": "orthographic_worst_seam",
        "capture_spec_sha256": "c" * 64,
        "capture_plan_sha256": "d" * 64,
        "job_sha256": "e" * 64,
        "tile_ids": ["a", "b"],
        "artifacts": {
            key: zone_qa._artifact(root / file_name, relative_to=root)
            for key, file_name in zone_qa.CAPTURE_ARTIFACT_NAMES.items()
        },
        "aov": {
            "terrain_lod_name": zone_qa.TERRAIN_AOV,
            "terrain_coverage_name": zone_qa.COVERAGE_AOV,
            "expected_lod": 0,
            "invalid_lod_pixel_count": 0,
            "invalid_coverage_pixel_count": 0,
            "terrain_pixel_count": 10,
        },
        "imported_lod0": {
            "lod": 0,
            "tile_ids": ["a", "b"],
            "mesh_count": 2,
            "forbidden_lod_mesh_count": 0,
        },
        "reference_material": {
            "model": "FireViewerGroundSurface_v1 Blender reference",
            "connected_channels": ["basecolor", "normal", "height_bump", "orm"],
        },
    }
    path = root / "capture.done.v1.json"
    _json(path, receipt)
    return path


def test_capture_receipt_rejects_tampered_or_absent_artifact(tmp_path: Path) -> None:
    receipt_path = _capture_receipt(tmp_path / "capture")
    zone_qa.validate_capture_receipt(receipt_path)
    beauty = receipt_path.parent / "beauty.png"
    beauty.write_bytes(b"tampered")
    with pytest.raises(zone_qa.ZoneVisualQaError, match="hash mismatch"):
        zone_qa.validate_capture_receipt(receipt_path)
    beauty.unlink()
    with pytest.raises(zone_qa.ZoneVisualQaError, match="absent"):
        zone_qa.validate_capture_receipt(receipt_path)


def test_capture_receipt_rejects_mono_tile_seam(tmp_path: Path) -> None:
    receipt_path = _capture_receipt(tmp_path / "capture")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["tile_ids"] = ["a"]
    receipt["imported_lod0"]["tile_ids"] = ["a"]
    receipt["imported_lod0"]["mesh_count"] = 1
    _json(receipt_path, receipt)
    capture = {
        "capture_id": "seam-01-test",
        "category": "orthographic_worst_seam",
        "tile_ids": ["a"],
    }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["capture_spec_sha256"] = zone_qa._canonical_sha256(capture)
    _json(receipt_path, receipt)
    with pytest.raises(zone_qa.ZoneVisualQaError, match="exactly two tiles"):
        zone_qa.validate_capture_receipt(receipt_path, expected_capture=capture)


def test_pending_review_cannot_emit_acceptance(tmp_path: Path) -> None:
    technical_path = tmp_path / "technical.json"
    review_path = tmp_path / "review.json"
    technical = {
        "capture_set_sha256": "f" * 64,
        "captures": [],
    }
    _json(technical_path, technical)
    _json(
        review_path,
        {
            "schema": zone_qa.HUMAN_REVIEW_SCHEMA,
            "decision": "pending",
            "reviewer": {"kind": "human", "id": "reviewer"},
            "decision_recorded_at_utc": "2026-08-09T12:00:00Z",
            "technical_receipt_sha256": zone_qa._sha256(technical_path),
            "capture_set_sha256": "f" * 64,
            "capture_reviews": [],
        },
    )
    with pytest.raises(zone_qa.ZoneVisualQaError, match="not explicitly accepted"):
        zone_qa._validate_human_review(review_path, technical_path, technical)
