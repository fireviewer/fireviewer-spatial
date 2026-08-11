from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parent
MAP_SCHEMA_PATH = ROOT / "map-upload-contract.schema.json"
PERIMETER_SCHEMA_PATH = ROOT / "perimeter-layer-contract.schema.json"
CASE_SCHEMA_PATH = ROOT / "simulated-case-production-contract.schema.json"
BUNDLE_SCHEMA_PATH = ROOT / "reproducible-download-bundle-contract.schema.json"
PROFILES_PATH = ROOT / "production-profiles.v1.json"
MAP_EXAMPLE_PATH = ROOT / "examples" / "die-map-upload.candidate.json"
PERIMETER_EXAMPLE_PATH = ROOT / "examples" / "die-progressive-perimeters.candidate.json"
CASE_EXAMPLE_PATH = ROOT / "examples" / "die-retrospective-case.candidate.json"
BUNDLE_EXAMPLE_PATH = ROOT / "examples" / "die-reproduction-download.candidate.json"

ACTIVE_STATUS = "active"
CANDIDATE_STATUS = "candidate_pending_die_visual_acceptance"

REQUIRED_MAP_SOURCE_KINDS = {
    "mnt",
    "mns",
    "orthophoto",
    "vegetation",
    "buildings",
    "roads",
}
FORBIDDEN_MAP_SOURCE_KINDS = {"camera_plan", "perimeter", "simulation"}
REQUIRED_MODALITIES = {
    "rgb",
    "aerial_thermal_16bit",
    "depth_distance_to_camera",
    "semantic_segmentation",
    "instance_segmentation",
    "flame_mask",
    "smoke_mask",
    "burned_area_mask",
    "active_front_geometry",
    "visible_flame_points",
    "smoke_source_points",
    "geolocation",
    "abstention",
}
REQUIRED_METADATA_FIELDS = {
    "observation_id",
    "case_id",
    "map_package_id",
    "incident_id",
    "scenario_id",
    "simulation_id",
    "state_id",
    "incident_day_index",
    "state_in_day",
    "valid_at",
    "timezone",
    "camera_id",
    "camera_role",
    "capture_device_profile",
    "framing_style",
    "zoom_index",
    "focal_length_mm",
    "resolution_px",
    "camera_position_local_m",
    "camera_position_l93_ngf_ign69_m",
    "camera_orientation_quat_wxyz",
    "camera_intrinsics",
    "nearest_visible_flame_point_local_m",
    "nearest_visible_flame_distance_m",
    "visible_flame_points_local_m",
    "smoke_source_points_local_m",
    "active_front_local_m",
    "ignition_point_local_m",
    "burned_area_m2",
    "burned_tree_count",
    "expected_fire_in_frame",
    "line_of_sight_receipt",
    "negative_reason",
    "weather_state_id",
    "modality_sha256",
    "dependency_sha256",
}
REQUIRED_SPLIT_KEYS = {"base_map_package_id", "incident_id", "scenario_id"}
REQUIRED_BUNDLE_ROLES = {
    "map_contract",
    "map_manifest",
    "perimeter_contract",
    "perimeter_manifest",
    "camera_rig",
    "scenario",
    "flow",
    "runtime_contract",
    "source_truth",
    "asset_inventory",
}
CASE_ACTIVATION_GATES = {
    "base_map_active",
    "perimeter_layer_active",
    "source_integrity",
    "timeline_validation",
    "usd_validation",
    "tree_destruction",
    "flow_runtime_no_capture",
    "camera_retargeting",
    "human_visual_review",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_errors(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.absolute_path))
    ]


def _all_gates_passed(gates: dict[str, str]) -> bool:
    return bool(gates) and all(value == "passed" for value in gates.values())


def _selected_gates_passed(gates: dict[str, str], required: set[str]) -> bool:
    return all(gates.get(name) == "passed" for name in required)


def validate_map_semantics(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_kinds = {item["kind"] for item in record["source_locks"]}
    missing_sources = sorted(REQUIRED_MAP_SOURCE_KINDS - source_kinds)
    if missing_sources:
        errors.append(f"missing required map source kinds: {missing_sources}")
    forbidden_sources = sorted(FORBIDDEN_MAP_SOURCE_KINDS & source_kinds)
    if forbidden_sources:
        errors.append(f"map source locks contain forbidden production roles: {forbidden_sources}")

    bounds = record["spatial_reference"]["bounds_l93_m"]
    if not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
        errors.append("spatial bounds must be ordered xmin, ymin, xmax, ymax")

    simulation = record["simulation"]
    if any(simulation.values()):
        errors.append("map upload must contain no timeline, perimeter, camera or simulation content")

    status = record["contract_status"]
    release = record["release"]
    if status == ACTIVE_STATUS:
        required_hashes = (
            record["package"]["entry_stage_sha256"],
            record["package"]["manifest_sha256"],
            release["acceptance_receipt_sha256"],
        )
        if any(value is None for value in required_hashes):
            errors.append("an active map requires final stage, manifest and acceptance hashes")
        if not _all_gates_passed(record["quality_gates"]):
            errors.append("all map quality gates must pass before activation")
        if release["human_visual_decision"] != "accepted":
            errors.append("an active map requires an accepted human visual decision")
        if not release["upload_allowed"]:
            errors.append("an active map must explicitly allow upload")
    elif status == CANDIDATE_STATUS:
        if release["upload_allowed"]:
            errors.append("a candidate map cannot allow upload")
        if release["human_visual_decision"] == "accepted":
            errors.append("a candidate map cannot claim accepted visual review")
    return errors


def validate_perimeter_semantics(
    record: dict[str, Any],
    map_record: dict[str, Any],
    map_record_path: Path,
) -> list[str]:
    errors: list[str] = []
    base_map = record["base_map"]
    if base_map["contract_record_sha256"] != sha256_file(map_record_path):
        errors.append("base map contract record SHA-256 does not match")
    if base_map["package_id"] != map_record["package"]["package_id"]:
        errors.append("base map package_id does not match the referenced record")
    if base_map["revision"] != map_record["package"]["revision"]:
        errors.append("base map revision does not match the referenced record")

    progression = record["progression"]
    states = progression["state_records"]
    if progression["state_count"] != len(states):
        errors.append("perimeter state_count must equal the number of state_records")
    expected_orders = list(range(1, len(states) + 1))
    if [item["append_order"] for item in states] != expected_orders:
        errors.append("perimeter append_order values must be contiguous and ordered")
    expected_ids = [f"perimeter_{index:03d}" for index in expected_orders]
    if [item["state_id"] for item in states] != expected_ids:
        errors.append("perimeter state_id values must match append order")
    dates = [date.fromisoformat(item["local_date"]) for item in states]
    if dates:
        expected_dates = [(dates[0].toordinal() + index) for index in range(len(dates))]
        if [item.toordinal() for item in dates] != expected_dates:
            errors.append("perimeter local dates must be contiguous and ordered")
        if states[0]["local_date"] != progression["first_local_date"]:
            errors.append("first perimeter state date must match first_local_date")
        if states[-1]["local_date"] != progression["last_local_date"]:
            errors.append("last perimeter state date must match last_local_date")
    if len({item["layer_revision_id"] for item in states}) != len(states):
        errors.append("perimeter layer_revision_id values must be unique")
    if len({item["layer_path"] for item in states}) != len(states):
        errors.append("perimeter layer paths must be unique")

    status = record["contract_status"]
    release = record["release"]
    if status == ACTIVE_STATUS:
        if map_record["contract_status"] != ACTIVE_STATUS:
            errors.append("an active perimeter package requires an active base map")
        required_hashes = (
            record["layer_package"]["entry_layer_sha256"],
            record["layer_package"]["manifest_sha256"],
            base_map["acceptance_receipt_sha256"],
            release["acceptance_receipt_sha256"],
        )
        required_hashes += tuple(item["layer_sha256"] for item in states)
        if any(value is None for value in required_hashes):
            errors.append("an active perimeter package requires all artifact and acceptance hashes")
        if not _all_gates_passed(record["quality_gates"]):
            errors.append("all perimeter quality gates must pass before activation")
        if release["human_visual_decision"] != "accepted":
            errors.append("an active perimeter package requires accepted human visual review")
        if not release["layer_attachment_allowed"]:
            errors.append("an active perimeter package must allow explicit layer attachment")
    elif status == CANDIDATE_STATUS:
        if release["layer_attachment_allowed"]:
            errors.append("a candidate perimeter package cannot allow layer attachment")
        if release["human_visual_decision"] == "accepted":
            errors.append("a candidate perimeter package cannot claim accepted visual review")
    return errors


def validate_case_semantics(
    record: dict[str, Any],
    map_record: dict[str, Any],
    map_record_path: Path,
    perimeter_record: dict[str, Any],
    perimeter_record_path: Path,
) -> list[str]:
    errors: list[str] = []
    base_map = record["base_map"]
    if base_map["contract_record_sha256"] != sha256_file(map_record_path):
        errors.append("base map contract record SHA-256 does not match")
    if base_map["contract_schema"] != map_record["schema"]:
        errors.append("base map contract schema does not match the referenced record")
    if base_map["package_id"] != map_record["package"]["package_id"]:
        errors.append("base map package_id does not match the referenced record")
    if base_map["revision"] != map_record["package"]["revision"]:
        errors.append("base map revision does not match the referenced record")

    perimeter = record["perimeter_layers"]
    if perimeter["contract_record_sha256"] != sha256_file(perimeter_record_path):
        errors.append("perimeter contract record SHA-256 does not match")
    if perimeter["contract_schema"] != perimeter_record["schema"]:
        errors.append("perimeter contract schema does not match the referenced record")
    if perimeter["layer_package_id"] != perimeter_record["layer_package"]["layer_package_id"]:
        errors.append("perimeter layer_package_id does not match the referenced record")
    if perimeter["revision"] != perimeter_record["layer_package"]["revision"]:
        errors.append("perimeter revision does not match the referenced record")

    timeline = record["timeline"]
    if timeline["out_of_scene_state_count"] > timeline["capture_state_count"]:
        errors.append("out_of_scene_state_count cannot exceed capture_state_count")
    first_day = date.fromisoformat(timeline["first_local_date"])
    last_day = date.fromisoformat(timeline["last_local_date"])
    if (last_day - first_day).days + 1 != timeline["incident_days"]:
        errors.append("timeline dates must cover incident_days inclusively")
    if timeline["source_key_state_count"] != perimeter_record["progression"]["state_count"]:
        errors.append("case source_key_state_count must match the perimeter package")

    profile = record["case"]["production_profile"]
    if profile == "retrospective_daily_replay_v1":
        if record["truth_scope"]["kind"] != "retrospective_replay":
            errors.append("retrospective profile requires retrospective_replay truth scope")
        if timeline["states_per_day"] != 1:
            errors.append("retrospective profile requires one source-driven state per day")
        if timeline["source_key_state_count"] != timeline["incident_days"]:
            errors.append("retrospective source_key_state_count must equal incident_days")
        if timeline["capture_state_count"] != timeline["source_key_state_count"]:
            errors.append("retrospective capture_state_count must equal source_key_state_count")
    elif profile == "synthetic_training_18d_v1":
        expected = {
            "incident_days": 18,
            "states_per_day": 10,
            "capture_state_count": 180,
            "out_of_scene_state_count": 0,
        }
        for field, value in expected.items():
            if timeline[field] != value:
                errors.append(f"synthetic_training_18d_v1 requires {field}={value}")
        if record["truth_scope"]["kind"] != "deterministic_synthetic":
            errors.append("synthetic profile requires deterministic_synthetic truth scope")

    cameras = record["view_plan"]["camera_pool"]
    if cameras["total_count"] != cameras["human_count"] + cameras["aerial_count"]:
        errors.append("case camera total must equal human_count + aerial_count")
    if cameras["thermal_aerial_count"] != cameras["aerial_count"]:
        errors.append("every aerial case camera must provide thermal capture")

    capture = record["view_plan"]["capture"]
    in_scene_states = timeline["capture_state_count"] - timeline["out_of_scene_state_count"]
    expected_plans = timeline["capture_state_count"] * capture["views_per_state"]
    expected_positive = in_scene_states * capture["positive_views_per_state"] * capture["zooms_per_view"]
    expected_negative = (
        in_scene_states * capture["negative_views_per_state"]
        + timeline["out_of_scene_state_count"] * capture["views_per_state"]
    ) * capture["zooms_per_view"]
    expected_cases = expected_plans * capture["zooms_per_view"]
    expected_counts = {
        "expected_viewpoint_plans": expected_plans,
        "expected_capture_cases": expected_cases,
        "expected_positive_cases": expected_positive,
        "expected_negative_cases": expected_negative,
    }
    for field, value in expected_counts.items():
        if record["dataset"][field] != value:
            errors.append(f"dataset {field} must equal {value}")
    if expected_positive + expected_negative != expected_cases:
        errors.append("positive and negative capture arithmetic must cover all cases")

    modalities = set(record["observation_contract"]["modalities"])
    missing_modalities = sorted(REQUIRED_MODALITIES - modalities)
    if missing_modalities:
        errors.append(f"missing required observation modalities: {missing_modalities}")
    metadata_fields = set(record["observation_contract"]["metadata_fields"])
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - metadata_fields)
    if missing_metadata:
        errors.append(f"missing required observation metadata: {missing_metadata}")
    split_keys = set(record["dataset"]["split_group_keys"])
    missing_split_keys = sorted(REQUIRED_SPLIT_KEYS - split_keys)
    if missing_split_keys:
        errors.append(f"missing required split group keys: {missing_split_keys}")

    status = record["contract_status"]
    release = record["release"]
    if status == ACTIVE_STATUS:
        if map_record["contract_status"] != ACTIVE_STATUS:
            errors.append("an active simulated case requires an active base map record")
        if perimeter_record["contract_status"] != ACTIVE_STATUS:
            errors.append("an active simulated case requires an active perimeter record")
        if base_map["artifact_manifest_sha256"] is None:
            errors.append("an active simulated case requires the base map manifest hash")
        if base_map["acceptance_receipt_sha256"] is None:
            errors.append("an active simulated case requires the base map acceptance receipt")
        if perimeter["acceptance_receipt_sha256"] is None:
            errors.append("an active simulated case requires the perimeter acceptance receipt")
        if record["case"]["entry_stage_sha256"] is None:
            errors.append("an active simulated case requires the final stage SHA-256")
        if not _selected_gates_passed(record["quality_gates"], CASE_ACTIVATION_GATES):
            errors.append("all simulated case activation gates must pass before activation")
        if release["human_visual_decision"] != "accepted":
            errors.append("an active simulated case requires accepted human visual review")
        if release["acceptance_receipt_sha256"] is None:
            errors.append("an active simulated case requires its visual acceptance receipt")
        if not release["simulation_use_allowed"]:
            errors.append("an active simulated case must allow reproducible simulation use")
        if release["full_dataset_capture_allowed"] and record["quality_gates"]["capture_validation"] != "passed":
            errors.append("full dataset capture requires a passed pilot capture validation")
        if release["dataset_release_allowed"] and not release["full_dataset_capture_allowed"]:
            errors.append("dataset release requires full dataset capture authorization")
        if release["training_use_allowed"] and record["quality_gates"]["training_readiness"] != "passed":
            errors.append("training use requires a passed training_readiness gate")
    elif status == CANDIDATE_STATUS:
        release_flags = (
            "simulation_use_allowed",
            "pilot_capture_allowed",
            "full_dataset_capture_allowed",
            "dataset_release_allowed",
            "training_use_allowed",
        )
        if any(release[field] for field in release_flags):
            errors.append("a candidate simulated case cannot enable simulation, capture, release or training")
        if release["human_visual_decision"] == "accepted":
            errors.append("a candidate simulated case cannot claim accepted visual review")
    return errors


def validate_bundle_semantics(
    record: dict[str, Any],
    case_record: dict[str, Any],
    case_record_path: Path,
) -> list[str]:
    errors: list[str] = []
    reference = record["case_reference"]
    if reference["contract_record_sha256"] != sha256_file(case_record_path):
        errors.append("case contract record SHA-256 does not match")
    if reference["case_id"] != case_record["case"]["case_id"]:
        errors.append("bundle case_id does not match the referenced case")
    if reference["package_id"] != case_record["case"]["package_id"]:
        errors.append("bundle package_id does not match the referenced case")
    roles = {item["role"] for item in record["dependency_locks"]}
    missing_roles = sorted(REQUIRED_BUNDLE_ROLES - roles)
    if missing_roles:
        errors.append(f"bundle is missing required dependency roles: {missing_roles}")

    status = record["contract_status"]
    release = record["release"]
    if status == ACTIVE_STATUS:
        if case_record["contract_status"] != ACTIVE_STATUS:
            errors.append("an active download bundle requires an active simulated case")
        required_hashes = (
            record["bundle"]["entry_stage_sha256"],
            record["bundle"]["manifest_sha256"],
            record["bundle"]["archive_sha256"],
            record["portability"]["dependency_inventory_sha256"],
            release["acceptance_receipt_sha256"],
        )
        required_hashes += tuple(item["sha256"] for item in record["dependency_locks"])
        if any(value is None for value in required_hashes):
            errors.append("an active download bundle requires all final dependency and archive hashes")
        if not _all_gates_passed(record["quality_gates"]):
            errors.append("all download bundle quality gates must pass before activation")
        if record["portability"]["isolated_reopen"] != "passed":
            errors.append("an active download bundle requires an isolated reopen proof")
        if release["human_visual_decision"] != "accepted":
            errors.append("an active download bundle requires accepted human visual review")
        if not release["download_allowed"]:
            errors.append("an active download bundle must explicitly allow download")
    elif status == CANDIDATE_STATUS:
        if release["download_allowed"]:
            errors.append("a candidate download bundle cannot allow download")
        if release["human_visual_decision"] == "accepted":
            errors.append("a candidate download bundle cannot claim accepted visual review")
    return errors


def validate_profiles(profiles: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if profiles.get("schema") != "fireviewer.omniverse-production-profiles.v1":
        errors.append("invalid production profile schema")
    if profiles.get("contract_status") != CANDIDATE_STATUS:
        errors.append("profiles must remain candidate until Die visual acceptance")
    activation = profiles.get("activation_gate", {})
    if activation.get("automatic_activation") is not False:
        errors.append("contract activation must never be automatic")
    map_profile = profiles.get("map_upload", {})
    if map_profile.get("perimeter_embedded") is not False:
        errors.append("the map upload profile must exclude perimeter layers")
    if map_profile.get("camera_rig_embedded") is not False:
        errors.append("the map upload profile must exclude camera rigs")
    perimeter_profile = profiles.get("progressive_perimeter_layer", {})
    if perimeter_profile.get("separate_from_map_upload") is not True:
        errors.append("the perimeter profile must remain separate from map upload")
    shared = profiles.get("simulated_case", {})
    expected_shared = {
        "seconds_per_day": 60.0,
        "views_per_state": 20,
        "positive_views_per_in_scene_state": 16,
        "negative_views_per_in_scene_state": 4,
        "zooms_per_view": 5,
    }
    for field, value in expected_shared.items():
        if shared.get(field) != value:
            errors.append(f"production profiles require {field}={value}")
    synthetic = shared.get("profiles", {}).get("synthetic_training_18d_v1", {})
    expected_synthetic = {
        "incident_days": 18,
        "states_per_day": 10,
        "capture_state_count": 180,
        "expected_viewpoint_plans": 3600,
        "expected_capture_cases": 18000,
        "expected_positive_cases": 14400,
        "expected_negative_cases": 3600,
    }
    for field, value in expected_synthetic.items():
        if synthetic.get(field) != value:
            errors.append(f"synthetic production profile requires {field}={value}")
    download = profiles.get("reproducible_download", {})
    if download.get("map_upload_included") is not False:
        errors.append("the download profile must not apply to map uploads")
    if download.get("backend_implementation_in_scope") is not False:
        errors.append("backend access implementation must remain outside this contract scope")
    return errors


def validate_all() -> list[str]:
    schemas = {
        "map": load_json(MAP_SCHEMA_PATH),
        "perimeter": load_json(PERIMETER_SCHEMA_PATH),
        "case": load_json(CASE_SCHEMA_PATH),
        "bundle": load_json(BUNDLE_SCHEMA_PATH),
    }
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)

    map_record = load_json(MAP_EXAMPLE_PATH)
    perimeter_record = load_json(PERIMETER_EXAMPLE_PATH)
    case_record = load_json(CASE_EXAMPLE_PATH)
    bundle_record = load_json(BUNDLE_EXAMPLE_PATH)
    profiles = load_json(PROFILES_PATH)

    errors = [f"map schema: {error}" for error in schema_errors(map_record, schemas["map"])]
    errors.extend(f"map semantics: {error}" for error in validate_map_semantics(map_record))
    errors.extend(
        f"perimeter schema: {error}"
        for error in schema_errors(perimeter_record, schemas["perimeter"])
    )
    errors.extend(
        f"perimeter semantics: {error}"
        for error in validate_perimeter_semantics(perimeter_record, map_record, MAP_EXAMPLE_PATH)
    )
    errors.extend(f"case schema: {error}" for error in schema_errors(case_record, schemas["case"]))
    errors.extend(
        f"case semantics: {error}"
        for error in validate_case_semantics(
            case_record,
            map_record,
            MAP_EXAMPLE_PATH,
            perimeter_record,
            PERIMETER_EXAMPLE_PATH,
        )
    )
    errors.extend(
        f"bundle schema: {error}"
        for error in schema_errors(bundle_record, schemas["bundle"])
    )
    errors.extend(
        f"bundle semantics: {error}"
        for error in validate_bundle_semantics(bundle_record, case_record, CASE_EXAMPLE_PATH)
    )
    errors.extend(f"profiles: {error}" for error in validate_profiles(profiles))
    return errors


def main() -> int:
    errors = validate_all()
    if errors:
        print("CONTRACT_VALIDATION_FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("CONTRACT_VALIDATION_PASSED")
    print(f"map_contract={MAP_EXAMPLE_PATH.relative_to(ROOT).as_posix()}")
    print(f"perimeter_contract={PERIMETER_EXAMPLE_PATH.relative_to(ROOT).as_posix()}")
    print(f"case_contract={CASE_EXAMPLE_PATH.relative_to(ROOT).as_posix()}")
    print(f"download_contract={BUNDLE_EXAMPLE_PATH.relative_to(ROOT).as_posix()}")
    print("status=candidate_pending_die_visual_acceptance")
    print("publication_triggered=false")
    print("dataset_capture_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
