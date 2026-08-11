from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import render_ground_atlas_acceptance as gate


@pytest.fixture(autouse=True)
def _allow_isolated_test_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve_test_path(
        path: Path,
        label: str,
        *,
        must_exist: bool = False,
    ) -> Path:
        del label
        resolved = path.resolve()
        if must_exist and not resolved.exists():
            raise gate.AtlasAcceptanceError(f"absent: {resolved}")
        return resolved

    monkeypatch.setattr(gate, "_require_d_path", resolve_test_path)


def _write_catalog(root: Path) -> tuple[Path, dict]:
    assets: dict[str, dict] = {}
    for role in gate.REQUIRED_TEXTURE_ROLES:
        path = root / "runtime-atlas" / f"{role}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((f"runtime-atlas-{role}-" * 32).encode())
        assets[role] = {
            "path": f"runtime-atlas/{role}.png",
            "sha256": gate.sha256_file(path),
            "byte_count": path.stat().st_size,
            "width": 4096,
            "height": 4096,
        }
    families = (
        "natural_ground",
        "burned_ground",
        *gate.DISTANT_FAMILIES,
    )
    micro_source = {
        "id": "test-source",
        "physical_scale_m": 3.0,
        "atlas_uv": {"offset": [0.01, 0.01], "scale": [0.1, 0.1]},
    }
    profiles = [
        {
            "id": f"{families[index % len(families)]}.profile_{index:02d}",
            "family": families[index % len(families)],
            "application_mode": "ground_blend",
            "surface_basis": "atlas_pbr",
            "micro_source_id": "test-source",
            "parameters": {},
        }
        for index in range(72)
    ]
    catalog = {
        "schema": gate.CATALOG_SCHEMA,
        "status": "generated_pending_blender_visual_acceptance",
        "runtime_texture_count": 4,
        "runtime_atlas": {"assets": assets},
        "profile_count": 72,
        "profiles": profiles,
        "micro_sources": [micro_source],
    }
    catalog["catalog_sha256"] = gate.canonical_sha256(catalog)
    path = root / "ground-surface-atlas-catalog.json"
    path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, catalog


def _pending_payload(catalog_path: Path, output: Path) -> tuple[Path, dict]:
    catalog, lock = gate.load_catalog(catalog_path)
    matrix = gate.build_render_matrix(catalog)
    renders = {}
    for key in gate.expected_render_keys():
        artifact = output / f"{key}.png"
        artifact.write_bytes((f"render-{key}-" * 16).encode())
        renders[key] = {
            "path": artifact.name,
            "sha256": gate.sha256_file(artifact),
            "byte_count": artifact.stat().st_size,
            "width_px": 512,
            "height_px": 512,
            "metrics": {
                "sample_count": 64,
                "minimum_luminance": 0.01,
                "maximum_luminance": 0.8,
                "mean_luminance": 0.3,
                "luminance_standard_deviation": 0.1,
            },
        }
        if key.startswith("runtime_"):
            renders[key].update(
                {
                    "diagnostic": "runtime_atlas_channel",
                    "role": key.removeprefix("runtime_"),
                }
            )
        elif key.startswith("profiles_"):
            band = key.removeprefix("profiles_")
            cell_ids = [cell["cell_id"] for cell in matrix if cell["band"] == band]
            measured = [
                {
                    "id": cell_id,
                    "mean_luminance": 0.3,
                    "maximum_luminance": 0.8,
                    "dynamic_range": 0.6,
                    "non_dark_fraction": 1.0,
                    "bright_fraction": 0.2,
                }
                for cell_id in cell_ids
            ]
            renders[key].update(
                {
                    "diagnostic": "physical_profile_contact_sheet",
                    "band": band,
                    "cell_count": 72,
                    "cell_ids_sha256": gate.canonical_sha256(cell_ids),
                    "cell_metrics": measured,
                    "label_metrics": copy.deepcopy(measured),
                    "cell_validation": {"invalid_cell_count": 0},
                    "label_validation": {"invalid_label_count": 0},
                }
            )
        else:
            renders[key].update(
                {
                    "diagnostic": "distant_representative_surface",
                    "family": key.removeprefix("distant_"),
                    "surface_validation": {"invalid_surface_count": 0},
                }
            )
    payload = {
        "schema": gate.PENDING_SCHEMA,
        "status": "rendered_pending_visual_review",
        "production_visual_gate_passed": False,
        "blender": {
            "version": "4.5.3 LTS",
            "version_tuple": [4, 5, 3],
            "binary_filename": "blender.exe",
            "binary_sha256": "a" * 64,
            "render_engine": "BLENDER_EEVEE_NEXT",
        },
        "d_only_environment": {name: "D:" for name in gate.REQUIRED_D_ENVIRONMENT},
        "catalog": {
            "file_name": lock["file_name"],
            "file_sha256": lock["file_sha256"],
            "declared_sha256": lock["declared_sha256"],
            "profile_count": 72,
            "texture_count": 4,
        },
        "runtime_atlas": lock["runtime_atlas"],
        "scale_bands": [
            {
                "id": band,
                "minimum_span_m": spans[0],
                "maximum_span_m": spans[-1],
                "sampled_spans_m": list(spans),
            }
            for band, spans in gate.SCALE_BANDS.items()
        ],
        "matrix": matrix,
        "matrix_cell_count": 216,
        "matrix_sha256": gate.canonical_sha256(matrix),
        "renders": renders,
        "required_human_review": {
            "schema": gate.HUMAN_REVIEW_SCHEMA,
            "checks": list(gate.REQUIRED_REVIEW_CHECKS),
            "cell_verdict_count": 216,
            "render_verdict_keys": list(gate.expected_render_keys()),
            "automatic_acceptance": "forbidden",
        },
    }
    pending = gate._sealed_payload(payload, "receipt_content_sha256")
    pending_path = output / "atlas-render.pending-visual-review.v1.json"
    gate._atomic_json(pending_path, pending)
    return pending_path, pending


def _review_payload(pending_path: Path, pending: dict) -> dict:
    payload = {
        "schema": gate.HUMAN_REVIEW_SCHEMA,
        "status": "review_complete",
        "verdict": "accepted",
        "reviewer": {"kind": "human", "id": "qa-reviewer"},
        "reviewed_at_utc": "2026-08-09T12:00:00Z",
        "render_receipt_sha256": gate.sha256_file(pending_path),
        "catalog_file_sha256": pending["catalog"]["file_sha256"],
        "matrix_sha256": pending["matrix_sha256"],
        "checks": [
            {"id": check, "verdict": "accepted"}
            for check in gate.REQUIRED_REVIEW_CHECKS
        ],
        "cells": [
            {"cell_id": cell["cell_id"], "verdict": "accepted"}
            for cell in pending["matrix"]
        ],
        "renders": [
            {
                "key": key,
                "sha256": pending["renders"][key]["sha256"],
                "verdict": "accepted",
            }
            for key in gate.expected_render_keys()
        ],
        "rejections": [],
    }
    return gate._sealed_payload(payload, "review_content_sha256")


def _write_review(output: Path, review: dict) -> Path:
    path = output / "human-visual-review.v1.json"
    gate._atomic_json(path, review)
    return path


def _proofs(tmp_path: Path) -> tuple[Path, Path, dict, Path]:
    catalog_path, _catalog = _write_catalog(tmp_path)
    output = tmp_path / "proofs"
    output.mkdir()
    pending_path, pending = _pending_payload(catalog_path, output)
    review_path = _write_review(output, _review_payload(pending_path, pending))
    return catalog_path, pending_path, pending, review_path


def test_catalog_and_matrix_lock_four_atlases_and_exactly_216_cells(
    tmp_path: Path,
) -> None:
    catalog_path, _catalog = _write_catalog(tmp_path)
    loaded, lock = gate.load_catalog(catalog_path)
    matrix = gate.build_render_matrix(loaded)

    assert len(matrix) == 216
    assert len({cell["cell_id"] for cell in matrix}) == 216
    assert {cell["band"] for cell in matrix} == {"micro", "meso", "macro"}
    assert set(lock["runtime_atlas"]) == set(gate.REQUIRED_TEXTURE_ROLES)

    (tmp_path / "runtime-atlas" / "normal.png").write_bytes(b"tampered")
    with pytest.raises(gate.AtlasAcceptanceError, match="changed or is absent: normal"):
        gate.load_catalog(catalog_path)


def test_render_cli_never_accepts_and_uses_blender_separator() -> None:
    arguments = gate._parse_arguments(
        ["--phase", "render", "--catalog", "D:/catalog.json", "--output", "D:/qa"]
    )
    assert arguments.phase == "render"
    with pytest.raises(SystemExit):
        gate._parse_arguments(
            [
                "--phase",
                "render",
                "--catalog",
                "D:/catalog.json",
                "--output",
                "D:/qa",
                "--acceptance-receipt",
                "D:/qa/accepted.json",
            ]
        )


def test_acceptance_requires_exhaustive_hash_bound_human_review(
    tmp_path: Path,
) -> None:
    catalog_path, pending_path, _pending, review_path = _proofs(tmp_path)
    acceptance_path = pending_path.parent / "atlas-visual.accepted.v1.json"

    path, acceptance = gate.accept_review(
        catalog_path,
        pending_path,
        review_path,
        acceptance_path,
    )

    assert path == acceptance_path
    assert acceptance["schema"] == gate.ACCEPTANCE_SCHEMA
    assert acceptance["status"] == "accepted_blender_visual"
    assert acceptance["profile_count"] == 72
    assert acceptance["reviewed_cell_count"] == 216
    assert acceptance["invalid_profile_count"] == 0
    assert acceptance["scale_bands"] == ["micro", "meso", "macro"]
    assert acceptance["atlas_catalog_sha256"] == gate.sha256_file(catalog_path)
    assert acceptance["human_review"]["sha256"] == gate.sha256_file(review_path)
    assert gate.validate_acceptance(catalog_path, acceptance_path) == acceptance


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda review: review["cells"].pop(), "cell coverage is incomplete"),
        (
            lambda review: review["checks"].__setitem__(
                0,
                {"id": review["checks"][0]["id"], "verdict": "rejected"},
            ),
            "rejected check",
        ),
        (
            lambda review: review.__setitem__(
                "reviewer", {"kind": "agent", "id": "not-human"}
            ),
            "identify a human reviewer",
        ),
        (
            lambda review: review.__setitem__(
                "rejections", [{"cell_id": "micro:00:any"}]
            ),
            "contains one or more rejections",
        ),
        (
            lambda review: review["renders"][0].__setitem__("sha256", "b" * 64),
            "render hash mismatch",
        ),
    ],
)
def test_human_review_rejects_incomplete_or_negative_evidence(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    catalog_path, pending_path, pending, review_path = _proofs(tmp_path)
    review = _review_payload(pending_path, pending)
    review.pop("review_content_sha256")
    mutation(review)
    review = gate._sealed_payload(review, "review_content_sha256")
    gate._atomic_json(review_path, review)

    with pytest.raises(gate.AtlasAcceptanceError, match=message):
        gate.accept_review(
            catalog_path,
            pending_path,
            review_path,
            pending_path.parent / "accepted.json",
        )


def test_tampering_with_review_or_render_is_fail_closed(tmp_path: Path) -> None:
    catalog_path, pending_path, pending, review_path = _proofs(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewer"]["id"] = "tampered-without-resealing"
    gate._atomic_json(review_path, review)
    with pytest.raises(
        gate.AtlasAcceptanceError, match="human review content hash mismatch"
    ):
        gate.accept_review(
            catalog_path,
            pending_path,
            review_path,
            pending_path.parent / "accepted.json",
        )

    gate._atomic_json(review_path, _review_payload(pending_path, pending))
    first_key = gate.expected_render_keys()[0]
    (pending_path.parent / pending["renders"][first_key]["path"]).write_bytes(
        b"changed"
    )
    with pytest.raises(gate.AtlasAcceptanceError, match="Render artifact changed"):
        gate.accept_review(
            catalog_path,
            pending_path,
            review_path,
            pending_path.parent / "accepted.json",
        )


def test_resealed_matrix_or_traversal_tampering_is_fail_closed(tmp_path: Path) -> None:
    catalog_path, pending_path, pending, _review_path = _proofs(tmp_path)
    changed = copy.deepcopy(pending)
    changed.pop("receipt_content_sha256")
    changed["matrix"][0]["physical_span_m"] = 999.0
    changed = gate._sealed_payload(changed, "receipt_content_sha256")
    gate._atomic_json(pending_path, changed)
    with pytest.raises(gate.AtlasAcceptanceError, match="216-cell matrix mismatch"):
        gate.validate_pending_receipt(catalog_path, pending_path)

    changed = copy.deepcopy(pending)
    changed.pop("receipt_content_sha256")
    first_key = gate.expected_render_keys()[0]
    changed["renders"][first_key]["path"] = "../escaped.png"
    changed = gate._sealed_payload(changed, "receipt_content_sha256")
    gate._atomic_json(pending_path, changed)
    with pytest.raises(gate.AtlasAcceptanceError, match="bounded relative path"):
        gate.validate_pending_receipt(catalog_path, pending_path)


def test_pending_receipt_rejects_black_profile_sheet_metrics(tmp_path: Path) -> None:
    catalog_path, pending_path, pending, _review_path = _proofs(tmp_path)
    changed = copy.deepcopy(pending)
    changed.pop("receipt_content_sha256")
    changed["renders"]["profiles_micro"]["cell_validation"]["invalid_cell_count"] = 72
    changed = gate._sealed_payload(changed, "receipt_content_sha256")
    gate._atomic_json(pending_path, changed)

    with pytest.raises(gate.AtlasAcceptanceError, match="occupancy proof is invalid"):
        gate.validate_pending_receipt(catalog_path, pending_path)


def test_failed_render_keeps_only_a_light_sealed_receipt(tmp_path: Path) -> None:
    output = tmp_path / "atlas-qa"
    staging = tmp_path / ".atlas-qa.rendering"
    staging.mkdir()
    (staging / "large-intermediate.png").write_bytes(b"large")
    path = gate._publish_render_failure(
        output=output,
        staging=staging,
        error=gate.AtlasAcceptanceError("24 dark or flat cells"),
        blender_lock={"version": "4.5.3 LTS", "binary_sha256": "a" * 64},
        catalog_lock={
            "file_name": "catalog.json",
            "file_sha256": "b" * 64,
            "declared_sha256": "c" * 64,
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert not staging.exists()
    assert [item.name for item in output.iterdir()] == [path.name]
    assert payload["schema"] == gate.FAILURE_SCHEMA
    assert payload["status"] == "failed_technical_render"
    assert payload["pending_visual_review_receipt_emitted"] is False
    assert payload["visual_acceptance_receipt_emitted"] is False
    gate._validate_seal(payload, "failure_content_sha256", "failure")


def test_read_only_acceptance_validation_rehashes_images_and_proof_files(
    tmp_path: Path,
) -> None:
    catalog_path, pending_path, pending, review_path = _proofs(tmp_path)
    acceptance_path = pending_path.parent / "atlas-visual.accepted.v1.json"
    _path, acceptance = gate.accept_review(
        catalog_path,
        pending_path,
        review_path,
        acceptance_path,
    )

    first_key = gate.expected_render_keys()[0]
    image = pending_path.parent / pending["renders"][first_key]["path"]
    original = image.read_bytes()
    image.write_bytes(b"corrupted")
    with pytest.raises(gate.AtlasAcceptanceError, match="Render artifact changed"):
        gate.validate_acceptance(catalog_path, acceptance_path)
    image.write_bytes(original)

    moved = review_path.with_name("moved-review.json")
    review_path.rename(moved)
    with pytest.raises(gate.AtlasAcceptanceError, match="absent or moved"):
        gate.validate_acceptance(catalog_path, acceptance_path)
    moved.rename(review_path)
    assert gate.validate_acceptance(catalog_path, acceptance_path) == acceptance


def test_read_only_acceptance_rejects_resealed_path_traversal(tmp_path: Path) -> None:
    catalog_path, pending_path, _pending, review_path = _proofs(tmp_path)
    acceptance_path = pending_path.parent / "atlas-visual.accepted.v1.json"
    _path, acceptance = gate.accept_review(
        catalog_path,
        pending_path,
        review_path,
        acceptance_path,
    )
    acceptance.pop("acceptance_content_sha256")
    acceptance["human_review"]["path"] = "../escaped-review.json"
    acceptance = gate._sealed_payload(acceptance, "acceptance_content_sha256")
    gate._atomic_json(acceptance_path, acceptance)

    with pytest.raises(gate.AtlasAcceptanceError, match="bounded relative path"):
        gate.validate_acceptance(catalog_path, acceptance_path)
