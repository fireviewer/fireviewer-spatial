from __future__ import annotations

import json
from pathlib import Path

import pytest

import accept_simple_measured_tile_review as acceptance
import run_simple_measured_tile_qa as tile_qa
from test_run_simple_measured_tile_qa import FakeBlender, build_synthetic_package


@pytest.fixture
def synthetic_package(tmp_path: Path) -> tuple[Path, Path]:
    return build_synthetic_package(tmp_path)


def test_explicit_human_review_is_sealed_and_replayed(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    tile_qa.execute_qa(package_root, blender, command_runner=FakeBlender())
    path, receipt = acceptance.accept_human_review(
        package_root,
        blender,
        review_statement="c valider enchaine",
    )
    first_bytes = path.read_bytes()
    assert receipt["status"] == acceptance.ACCEPTANCE_STATUS
    assert receipt["accepted_human"] is True
    assert receipt["automatic_acceptance"] is False
    assert receipt["review"] == {
        "kind": "human",
        "verdict": "accepted",
        "statement": "c valider enchaine",
    }
    assert set(receipt["captures"]) == {"topdown", "oblique"}
    assert acceptance.verify_human_acceptance(package_root, blender) == receipt
    second_path, _second = acceptance.accept_human_review(
        package_root,
        blender,
        review_statement="c valider enchaine",
    )
    assert second_path.read_bytes() == first_bytes


def test_acceptance_rejects_missing_statement_and_changed_verdict(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    tile_qa.execute_qa(package_root, blender, command_runner=FakeBlender())
    with pytest.raises(
        acceptance.SimpleMeasuredTileHumanAcceptanceError,
        match="explicit human review statement",
    ):
        acceptance.accept_human_review(
            package_root,
            blender,
            review_statement="",
        )
    path, _receipt = acceptance.accept_human_review(
        package_root,
        blender,
        review_statement="accepted",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["review"]["verdict"] = "rejected"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        acceptance.SimpleMeasuredTileHumanAcceptanceError,
        match="content hash is invalid",
    ):
        acceptance.verify_human_acceptance(package_root, blender)


def test_capture_tamper_invalidates_human_acceptance(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    tile_qa.execute_qa(package_root, blender, command_runner=FakeBlender())
    acceptance.accept_human_review(
        package_root,
        blender,
        review_statement="accepted",
    )
    capture = package_root / "qa" / "renders" / "oblique.png"
    capture.write_bytes(capture.read_bytes() + b"tampered")
    with pytest.raises(
        tile_qa.SimpleMeasuredTileQaRunnerError,
        match="Technical QA receipt is invalid",
    ):
        acceptance.verify_human_acceptance(package_root, blender)


def test_contract_forbids_automatic_acceptance() -> None:
    contract = json.loads(
        Path(acceptance.__file__)
        .with_name("simple_measured_tile_human_acceptance_contract.v1.json")
        .read_text(encoding="utf-8")
    )
    assert contract["automatic_acceptance"] is False
    assert contract["inputs"]["reviewer_kind"] == "human"
    assert contract["inputs"]["required_captures"] == ["topdown", "oblique"]
