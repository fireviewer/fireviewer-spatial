from __future__ import annotations

import json
from pathlib import Path

import pytest

from fireviewer_capture_storage import storage_profile_contract

from run_fireviewer_dataset_chunks import (
    build_runner_command,
    chunk_id,
    parse_state_spec,
    passed_chunk_receipt,
    storage_capacity_receipt,
)


def test_state_spec_supports_ordered_ranges() -> None:
    assert parse_state_spec("1-3,7,10-12") == [1, 2, 3, 7, 10, 11, 12]


@pytest.mark.parametrize("value", ["", "1,,2", "3-1", "0", "181", "1,1"])
def test_invalid_state_specs_fail_closed(value: str) -> None:
    with pytest.raises(ValueError):
        parse_state_spec(value)


def test_runner_command_names_one_globally_stable_production_chunk(tmp_path: Path) -> None:
    command = build_runner_command(
        launcher=tmp_path / "python.bat",
        runner=tmp_path / "runner.py",
        stage=tmp_path / "dataset.usda",
        chunk_root=tmp_path / "chunks" / "state_004",
        dataset_id="simulationDS1_diev1",
        state_index=4,
        pilot_acceptance_report=tmp_path / "pilot-audit.json",
        kit_cache_root=tmp_path / "cache",
        resolution="1280x720",
        rt_subframes=8,
        render_product_batch_size=1,
        seconds_per_day=60.0,
        flow_warmup_updates=180,
        headless=True,
    )

    assert command[0] == str(tmp_path / "python.bat")
    assert command[command.index("--production-state-indices") + 1] == "4"
    assert command[command.index("--production-chunk-id") + 1] == (
        "simulationDS1_diev1-state004"
    )
    assert command[command.index("--render-product-batch-size") + 1] == "1"
    assert command[-1] == "--headless"


def test_only_a_passed_matching_chunk_is_resumable(tmp_path: Path) -> None:
    dataset_id = "simulationDS1_diev1"
    state_index = 1
    stage_sha256 = "a" * 64
    chunk_root = tmp_path / "chunks" / "state_001"
    chunk_root.mkdir(parents=True)
    contract_path = chunk_root / "run-contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema": "fireviewer.kit-dataset-production-run.v1",
                "run_kind": "production_chunk",
                "dataset_admissible": True,
                "full_dataset_capture_authorized": True,
                "dataset_id": dataset_id,
                "production_chunk_id": chunk_id(dataset_id, state_index),
                "selected_state_indices": [state_index],
                "source_stage_sha256": stage_sha256,
                "capture_storage_profile": storage_profile_contract(),
                "expected_capture_cases": 100,
            }
        ),
        encoding="utf-8",
    )
    audit_path = chunk_root / "audit-report.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": "fireviewer.capture-metadata-audit.v2",
                "status": "passed",
                "failed_capture_count": 0,
                "abstention_warning_count": 0,
                "captures": 100,
                "expected_captures": 100,
                "viewpoint_plans": 20,
                "sample_counts": {
                    "positive_fire": 80,
                    "negative_context": 20,
                },
            }
        ),
        encoding="utf-8",
    )

    receipt = passed_chunk_receipt(
        chunk_root,
        dataset_id=dataset_id,
        state_index=state_index,
        source_stage_sha256=stage_sha256,
    )

    assert receipt is not None
    assert receipt["production_chunk_id"] == "simulationDS1_diev1-state001"
    assert receipt["captures"] == 100
    assert len(receipt["audit_report_sha256"]) == 64

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["selected_state_indices"] = [2]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    assert (
        passed_chunk_receipt(
            chunk_root,
            dataset_id=dataset_id,
            state_index=state_index,
            source_stage_sha256=stage_sha256,
        )
        is None
    )


def test_storage_capacity_receipt_uses_pilot_size_and_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    (pilot_root / "capture.bin").write_bytes(b"x" * 1000)
    report = tmp_path / "audit-report.json"
    report.write_text(
        json.dumps({"captures": 100, "capture_root": str(pilot_root)}),
        encoding="utf-8",
    )
    output_root = tmp_path / "production"
    monkeypatch.setattr(
        "run_fireviewer_dataset_chunks.shutil.disk_usage",
        lambda _: type("Usage", (), {"free": 2000})(),
    )

    receipt = storage_capacity_receipt(
        output_root=output_root,
        pilot_report=report,
        expected_chunk_captures=100,
        minimum_free_gib=0.0,
        safety_factor=1.25,
    )

    assert receipt["pilot_size_bytes"] == 1000
    assert receipt["estimated_chunk_bytes"] == 1250
    assert receipt["required_before_chunk_bytes"] == 1250
    assert receipt["admissible"] is True
