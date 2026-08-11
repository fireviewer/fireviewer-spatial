from __future__ import annotations

import copy
import sys
from pathlib import Path


CONTRACT_ROOT = Path(__file__).resolve().parent / "contracts" / "v1"
sys.path.insert(0, str(CONTRACT_ROOT))

from validate_contracts import (  # noqa: E402
    BUNDLE_EXAMPLE_PATH,
    BUNDLE_SCHEMA_PATH,
    CASE_EXAMPLE_PATH,
    CASE_SCHEMA_PATH,
    MAP_EXAMPLE_PATH,
    MAP_SCHEMA_PATH,
    PERIMETER_EXAMPLE_PATH,
    PERIMETER_SCHEMA_PATH,
    load_json,
    schema_errors,
    validate_all,
    validate_bundle_semantics,
    validate_case_semantics,
    validate_map_semantics,
    validate_perimeter_semantics,
)


def test_contract_bundle_is_valid() -> None:
    assert validate_all() == []


def test_map_schema_rejects_active_simulation() -> None:
    record = load_json(MAP_EXAMPLE_PATH)
    record["simulation"]["enabled"] = True
    assert schema_errors(record, load_json(MAP_SCHEMA_PATH))


def test_candidate_map_cannot_allow_upload() -> None:
    record = load_json(MAP_EXAMPLE_PATH)
    record["release"]["upload_allowed"] = True
    assert "a candidate map cannot allow upload" in validate_map_semantics(record)


def test_active_map_requires_visual_receipt_and_all_gates() -> None:
    record = load_json(MAP_EXAMPLE_PATH)
    record["contract_status"] = "active"
    errors = validate_map_semantics(record)
    assert any("acceptance hashes" in error for error in errors)
    assert any("quality gates" in error for error in errors)
    assert any("human visual decision" in error for error in errors)


def test_case_count_arithmetic_includes_out_of_scene_negatives() -> None:
    map_record = load_json(MAP_EXAMPLE_PATH)
    perimeter_record = load_json(PERIMETER_EXAMPLE_PATH)
    case_record = load_json(CASE_EXAMPLE_PATH)
    errors = validate_case_semantics(
        case_record,
        map_record,
        MAP_EXAMPLE_PATH,
        perimeter_record,
        PERIMETER_EXAMPLE_PATH,
    )
    assert errors == []
    assert case_record["dataset"]["expected_positive_cases"] == 1600
    assert case_record["dataset"]["expected_negative_cases"] == 500


def test_case_rejects_incorrect_capture_count() -> None:
    map_record = load_json(MAP_EXAMPLE_PATH)
    perimeter_record = load_json(PERIMETER_EXAMPLE_PATH)
    case_record = load_json(CASE_EXAMPLE_PATH)
    case_record["dataset"]["expected_capture_cases"] += 1
    errors = validate_case_semantics(
        case_record,
        map_record,
        MAP_EXAMPLE_PATH,
        perimeter_record,
        PERIMETER_EXAMPLE_PATH,
    )
    assert "dataset expected_capture_cases must equal 2100" in errors


def test_case_schema_locks_camera_retargeting_and_five_zooms() -> None:
    case_record = load_json(CASE_EXAMPLE_PATH)
    schema = load_json(CASE_SCHEMA_PATH)
    invalid_target = copy.deepcopy(case_record)
    invalid_target["view_plan"]["retargeting"]["positive_target"] = "ignition_point"
    assert schema_errors(invalid_target, schema)
    invalid_zooms = copy.deepcopy(case_record)
    invalid_zooms["view_plan"]["capture"]["zooms_per_view"] = 4
    assert schema_errors(invalid_zooms, schema)


def test_candidate_case_cannot_capture_or_enter_training() -> None:
    map_record = load_json(MAP_EXAMPLE_PATH)
    perimeter_record = load_json(PERIMETER_EXAMPLE_PATH)
    case_record = load_json(CASE_EXAMPLE_PATH)
    case_record["release"]["pilot_capture_allowed"] = True
    errors = validate_case_semantics(
        case_record,
        map_record,
        MAP_EXAMPLE_PATH,
        perimeter_record,
        PERIMETER_EXAMPLE_PATH,
    )
    assert any("cannot enable simulation, capture" in error for error in errors)


def test_map_schema_rejects_embedded_camera_rig() -> None:
    record = load_json(MAP_EXAMPLE_PATH)
    record["simulation"]["camera_rig_embedded"] = True
    assert schema_errors(record, load_json(MAP_SCHEMA_PATH))


def test_perimeter_states_are_ordered_and_separate() -> None:
    map_record = load_json(MAP_EXAMPLE_PATH)
    record = load_json(PERIMETER_EXAMPLE_PATH)
    assert schema_errors(record, load_json(PERIMETER_SCHEMA_PATH)) == []
    assert validate_perimeter_semantics(record, map_record, MAP_EXAMPLE_PATH) == []
    record["progression"]["state_records"][1]["append_order"] = 1
    assert any(
        "append_order values must be contiguous" in error
        for error in validate_perimeter_semantics(record, map_record, MAP_EXAMPLE_PATH)
    )


def test_candidate_perimeter_cannot_attach() -> None:
    map_record = load_json(MAP_EXAMPLE_PATH)
    record = load_json(PERIMETER_EXAMPLE_PATH)
    record["release"]["layer_attachment_allowed"] = True
    errors = validate_perimeter_semantics(record, map_record, MAP_EXAMPLE_PATH)
    assert "a candidate perimeter package cannot allow layer attachment" in errors


def test_download_bundle_is_reproduction_only_and_candidate_locked() -> None:
    record = load_json(BUNDLE_EXAMPLE_PATH)
    case_record = load_json(CASE_EXAMPLE_PATH)
    assert schema_errors(record, load_json(BUNDLE_SCHEMA_PATH)) == []
    assert validate_bundle_semantics(record, case_record, CASE_EXAMPLE_PATH) == []
    record["release"]["download_allowed"] = True
    assert "a candidate download bundle cannot allow download" in validate_bundle_semantics(
        record, case_record, CASE_EXAMPLE_PATH
    )
