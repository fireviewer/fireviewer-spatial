from __future__ import annotations

import json
from pathlib import Path

import pytest

from assemble_hf_simulation_dataset import assemble


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_capture(
    root: Path,
    *,
    zoom_index: int,
    multiplier: float,
    day_index: int = 1,
    state_index: int = 1,
    state_in_day: int = 1,
) -> None:
    capture_id = (
        f"day_{day_index:02d}_state_{state_index:03d}_view_01_"
        f"CAM_25_zoom_{zoom_index:02d}"
    )
    frame = (
        root
        / "simulationDS1_diev1"
        / f"state_{state_index:03d}"
        / "CAM_25"
        / f"frame_{zoom_index:06d}"
    )
    frame.mkdir(parents=True)
    (frame / "rgb.png").write_bytes(f"rgb-{zoom_index}".encode())
    write_json(
        frame / "training-targets.json",
        {
            "capture_id": capture_id,
            "dataset_id": "simulationDS1_diev1",
            "source_package_id": "source-r1",
            "sample_kind": "negative_context",
            "simulation_time": {
                "day_index": day_index,
                "state_in_day": state_in_day,
            },
            "zoom": {"zoom_index": zoom_index, "zoom_multiplier": multiplier},
            "modality_sha256": {},
        },
    )


def clean_audit(path: Path) -> None:
    write_json(
        path,
        {
            "schema": "fireviewer.capture-metadata-audit.v2",
            "status": "passed",
            "captures": 5,
            "expected_captures": 5,
            "failed_capture_count": 0,
            "abstention_warning_count": 0,
        },
    )


def test_assembly_uses_exact_day_case_point_contract_and_original_folder(tmp_path: Path) -> None:
    source = tmp_path / "captures"
    for zoom_index, multiplier in enumerate((0.75, 1.0, 1.25, 1.5, 2.0), start=1):
        make_capture(source, zoom_index=zoom_index, multiplier=multiplier)
    audit = tmp_path / "audit.json"
    clean_audit(audit)

    manifest = assemble(
        capture_root=source,
        audit_report=audit,
        destination_root=tmp_path / "staging",
        kind="sim",
        link_mode="copy",
    )

    point = tmp_path / "staging" / "massif-of-justin" / "sim" / "raw_files" / "day01" / "case01" / "point01"
    assert (point / "original" / "rgb.png").read_bytes() == b"rgb-2"
    assert (point / "zoom01_0p75x" / "capture-package.json").is_file()
    assert (point / "zoom05_2p00x" / "training-targets.json").is_file()
    assert manifest["captures"] == 5
    assert manifest["points"] == 1
    assert manifest["publication_authorized"] is False


def test_assembly_rejects_failed_or_abstained_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    clean_audit(audit)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["abstention_warning_count"] = 1
    write_json(audit, payload)

    with pytest.raises(RuntimeError, match="publication-staging clean"):
        assemble(
            capture_root=tmp_path / "captures",
            audit_report=audit,
            destination_root=tmp_path / "staging",
            kind="sim",
        )


def test_assembly_can_append_disjoint_passed_state_batches(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    for state_index, state_in_day in ((1, 1), (2, 2)):
        source = tmp_path / f"captures-{state_index}"
        for zoom_index, multiplier in enumerate((0.75, 1.0, 1.25, 1.5, 2.0), start=1):
            make_capture(
                source,
                zoom_index=zoom_index,
                multiplier=multiplier,
                state_index=state_index,
                state_in_day=state_in_day,
            )
        audit = tmp_path / f"audit-{state_index}.json"
        clean_audit(audit)
        manifest = assemble(
            capture_root=source,
            audit_report=audit,
            destination_root=staging,
            kind="sim",
            link_mode="copy",
            append=state_index > 1,
        )

    assert manifest["captures"] == 10
    assert manifest["points"] == 2
    assert len(manifest["source_batches"]) == 2
    assert (
        staging
        / "massif-of-justin"
        / "sim"
        / "raw_files"
        / "day01"
        / "case02"
        / "point01"
        / "original"
        / "rgb.png"
    ).is_file()
