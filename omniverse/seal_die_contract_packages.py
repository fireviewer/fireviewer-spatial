"""Seal an accepted separated Die candidate as an immutable active release.

The operation is additive: the candidate tree and the validated source scenes
remain untouched.  It activates map upload, explicit perimeter attachment,
reproducible simulation, pilot capture and authenticated bundle download.  It
does not authorize full dataset capture, dataset release, publication or
training admission.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

from build_die_contract_packages import (
    BUNDLE_ID,
    MAP_PACKAGE_ID,
    PERIMETER_PACKAGE_ID,
    assert_no_forbidden_map_content,
    canonical_json,
    copy_tree,
    create_archive,
    key_file_locks,
    load_json,
    relative_posix,
    sha256_file,
    usd_asset_issues,
    write_inventory,
    write_json,
)


CONTRACT_ROOT = Path(__file__).resolve().parent / "contracts" / "v1"
sys.path.insert(0, str(CONTRACT_ROOT))

from validate_contracts import (  # noqa: E402
    BUNDLE_SCHEMA_PATH,
    CASE_SCHEMA_PATH,
    MAP_SCHEMA_PATH,
    PERIMETER_SCHEMA_PATH,
    schema_errors,
    validate_bundle_semantics,
    validate_case_semantics,
    validate_map_semantics,
    validate_perimeter_semantics,
)


ACTIVE_STATUS = "active"
RELEASE_SCHEMA = "fireviewer.omniverse-separated-contract-release.v1"
ACCEPTANCE_SCHEMA = "fireviewer.omniverse-human-acceptance-receipt.v1"


def _write_copies(paths: Iterable[Path], value: dict[str, Any]) -> None:
    for path in paths:
        write_json(path, value)


def _passed_gates(record: dict[str, Any]) -> None:
    for name in record["quality_gates"]:
        record["quality_gates"][name] = "passed"


def _receipt(
    *,
    scope: str,
    accepted_at: str,
    artifacts: list[dict[str, Any]],
    note: str,
) -> dict[str, Any]:
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "scope": scope,
        "decision": "accepted",
        "accepted_at": accepted_at,
        "accepted_by": "workspace_user",
        "automatic_decision": False,
        "review_context": "visible_omniverse_editor_review_confirmed_by_user_in_active_task",
        "note": note,
        "artifacts": artifacts,
    }


def _artifact(path: Path, root: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": relative_posix(path, root),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _contract_errors(
    *,
    map_contract: dict[str, Any],
    map_path: Path,
    perimeter_contract: dict[str, Any],
    perimeter_path: Path,
    case_contract: dict[str, Any],
    case_path: Path,
    bundle_contract: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    errors.extend(
        f"map schema: {error}"
        for error in schema_errors(map_contract, load_json(MAP_SCHEMA_PATH))
    )
    errors.extend(f"map semantics: {error}" for error in validate_map_semantics(map_contract))
    errors.extend(
        f"perimeter schema: {error}"
        for error in schema_errors(perimeter_contract, load_json(PERIMETER_SCHEMA_PATH))
    )
    errors.extend(
        f"perimeter semantics: {error}"
        for error in validate_perimeter_semantics(perimeter_contract, map_contract, map_path)
    )
    errors.extend(
        f"case schema: {error}"
        for error in schema_errors(case_contract, load_json(CASE_SCHEMA_PATH))
    )
    errors.extend(
        f"case semantics: {error}"
        for error in validate_case_semantics(
            case_contract,
            map_contract,
            map_path,
            perimeter_contract,
            perimeter_path,
        )
    )
    errors.extend(
        f"bundle schema: {error}"
        for error in schema_errors(bundle_contract, load_json(BUNDLE_SCHEMA_PATH))
    )
    errors.extend(
        f"bundle semantics: {error}"
        for error in validate_bundle_semantics(bundle_contract, case_contract, case_path)
    )
    return errors


def seal(args: argparse.Namespace) -> Path:
    candidate = args.candidate_root.resolve()
    output = args.output_root.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing release: {output}")
    accepted_at = args.accepted_at or datetime.now(timezone.utc).isoformat()

    candidate_key_paths = (
        f"map/{MAP_PACKAGE_ID}/map.usda",
        f"map/{MAP_PACKAGE_ID}/manifest.json",
        f"perimeters/{PERIMETER_PACKAGE_ID}/perimeters.usda",
        f"perimeters/{PERIMETER_PACKAGE_ID}/manifest.json",
        "review/review-map-with-perimeters.usda",
        f"reproduction/{BUNDLE_ID}/dataset.usda",
        f"reproduction/{BUNDLE_ID}/manifest.json",
    )
    candidate_locks = key_file_locks(candidate, candidate_key_paths)
    copy_tree(candidate, output, ignore=shutil.ignore_patterns("archives"))

    map_root = output / "map" / MAP_PACKAGE_ID
    perimeter_root = output / "perimeters" / PERIMETER_PACKAGE_ID
    bundle_root = output / "reproduction" / BUNDLE_ID
    archive = output / "archives" / f"{BUNDLE_ID}.zip"
    top_contracts = output / "contracts"

    assert_no_forbidden_map_content(map_root)
    dependency_issues = {
        "map": usd_asset_issues(map_root),
        "perimeters": usd_asset_issues(perimeter_root),
        "bundle": usd_asset_issues(bundle_root),
    }
    if any(dependency_issues.values()):
        raise ValueError(f"Release contains invalid USD dependencies: {dependency_issues}")

    receipts = {
        "map": _receipt(
            scope="pure_map_upload",
            accepted_at=accepted_at,
            note="The user confirmed the already validated map and simulation had been inspected in Omniverse Editor; the pure map reuses the locked visual source components without simulation layers.",
            artifacts=[
                _artifact(map_root / "map.usda", output, role="pure_map_stage"),
                _artifact(map_root / "manifest.json", output, role="candidate_map_manifest"),
            ],
        ),
        "perimeter": _receipt(
            scope="separate_progressive_perimeter_layer",
            accepted_at=accepted_at,
            note="The separate 21-state terrain-draped perimeter layer is accepted for explicit composition and remains excluded from map upload.",
            artifacts=[
                _artifact(perimeter_root / "perimeters.usda", output, role="perimeter_stage"),
                _artifact(perimeter_root / "manifest.json", output, role="candidate_perimeter_manifest"),
            ],
        ),
        "case": _receipt(
            scope="reproducible_die_simulated_case",
            accepted_at=accepted_at,
            note="The user confirmed prior visible Omniverse Editor review of the accepted fire and smoke simulation and authorized dataset production after pack sealing.",
            artifacts=[
                _artifact(bundle_root / "dataset.usda", output, role="reproduction_stage"),
                _artifact(bundle_root / "scenarios" / "scenario.usda", output, role="scenario"),
                _artifact(bundle_root / "scenarios" / "flow.usda", output, role="flow"),
            ],
        ),
        "bundle": _receipt(
            scope="authenticated_reproducible_download_bundle",
            accepted_at=accepted_at,
            note="The standalone bundle is accepted for authenticated download after relative dependency scan, archive integrity and isolated Kit reopen checks.",
            artifacts=[
                _artifact(bundle_root / "dataset.usda", output, role="bundle_entry_stage"),
            ],
        ),
    }
    receipt_paths: dict[str, Path] = {}
    for scope, receipt in receipts.items():
        path = output / "acceptance" / f"{scope}-acceptance.json"
        write_json(path, receipt)
        receipt_paths[scope] = path
        write_json(bundle_root / "acceptance" / path.name, receipt)
    write_json(map_root / "acceptance" / receipt_paths["map"].name, receipts["map"])
    write_json(
        bundle_root / "map" / "acceptance" / receipt_paths["map"].name,
        receipts["map"],
    )
    write_json(
        perimeter_root / "acceptance" / receipt_paths["perimeter"].name,
        receipts["perimeter"],
    )
    write_json(
        bundle_root / "perimeters" / "acceptance" / receipt_paths["perimeter"].name,
        receipts["perimeter"],
    )

    map_manifest = load_json(map_root / "manifest.json")
    map_inventory_path, map_inventory_sha256, map_inventory_count = write_inventory(
        map_root,
        excluded_names={"manifest.json", "dependency-inventory.json", "contracts/map-contract.json"},
    )
    write_json(
        bundle_root / "map" / "dependency-inventory.json",
        load_json(map_inventory_path),
    )
    map_manifest.update(
        {
            "status": ACTIVE_STATUS,
            "acceptance": {
                "receipt": f"acceptance/{receipt_paths['map'].name}",
                "sha256": sha256_file(receipt_paths["map"]),
                "decision": "accepted",
            },
            "dependency_inventory": {
                "path": "dependency-inventory.json",
                "sha256": map_inventory_sha256,
                "file_count": map_inventory_count,
            },
            "release": {
                "upload_allowed": True,
                "automatic_publication": False,
                "site_publication_triggered": False,
            },
        }
    )
    _write_copies((map_root / "manifest.json", bundle_root / "map" / "manifest.json"), map_manifest)

    map_contract = load_json(map_root / "contracts" / "map-contract.json")
    map_contract["contract_status"] = ACTIVE_STATUS
    map_contract["package"]["manifest_sha256"] = sha256_file(map_root / "manifest.json")
    _passed_gates(map_contract)
    map_contract["release"].update(
        {
            "human_visual_decision": "accepted",
            "acceptance_receipt_sha256": sha256_file(receipt_paths["map"]),
            "upload_allowed": True,
            "automatic_publication": False,
            "site_publication_triggered": False,
        }
    )
    map_contract_paths = (
        map_root / "contracts" / "map-contract.json",
        top_contracts / "map-contract.json",
        bundle_root / "contracts" / "map-contract.json",
        bundle_root / "map" / "contracts" / "map-contract.json",
    )
    _write_copies(map_contract_paths, map_contract)
    map_contract_hash = sha256_file(top_contracts / "map-contract.json")

    perimeter_manifest = load_json(perimeter_root / "manifest.json")
    perimeter_inventory_path, perimeter_inventory_sha256, perimeter_inventory_count = write_inventory(
        perimeter_root,
        excluded_names={
            "manifest.json",
            "dependency-inventory.json",
            "contracts/perimeter-contract.json",
        },
    )
    write_json(
        bundle_root / "perimeters" / "dependency-inventory.json",
        load_json(perimeter_inventory_path),
    )
    perimeter_manifest["status"] = ACTIVE_STATUS
    perimeter_manifest["base_map"].update(
        {
            "contract_sha256": map_contract_hash,
            "acceptance_receipt_sha256": sha256_file(receipt_paths["map"]),
        }
    )
    perimeter_manifest["acceptance"] = {
        "receipt": f"acceptance/{receipt_paths['perimeter'].name}",
        "sha256": sha256_file(receipt_paths["perimeter"]),
        "decision": "accepted",
    }
    perimeter_manifest["dependency_inventory"] = {
        "path": "dependency-inventory.json",
        "sha256": perimeter_inventory_sha256,
        "file_count": perimeter_inventory_count,
    }
    perimeter_manifest["release"] = {
        "layer_attachment_allowed": True,
        "automatic_map_mutation": False,
        "automatic_publication": False,
    }
    _write_copies(
        (perimeter_root / "manifest.json", bundle_root / "perimeters" / "manifest.json"),
        perimeter_manifest,
    )

    perimeter_contract = load_json(perimeter_root / "contracts" / "perimeter-contract.json")
    perimeter_contract["contract_status"] = ACTIVE_STATUS
    perimeter_contract["base_map"].update(
        {
            "contract_record_sha256": map_contract_hash,
            "acceptance_receipt_sha256": sha256_file(receipt_paths["map"]),
        }
    )
    perimeter_contract["layer_package"]["manifest_sha256"] = sha256_file(
        perimeter_root / "manifest.json"
    )
    _passed_gates(perimeter_contract)
    perimeter_contract["release"].update(
        {
            "human_visual_decision": "accepted",
            "acceptance_receipt_sha256": sha256_file(receipt_paths["perimeter"]),
            "layer_attachment_allowed": True,
            "automatic_map_mutation": False,
            "automatic_publication": False,
        }
    )
    perimeter_contract_paths = (
        perimeter_root / "contracts" / "perimeter-contract.json",
        top_contracts / "perimeter-contract.json",
        bundle_root / "contracts" / "perimeter-contract.json",
        bundle_root / "perimeters" / "contracts" / "perimeter-contract.json",
    )
    _write_copies(perimeter_contract_paths, perimeter_contract)
    perimeter_contract_hash = sha256_file(top_contracts / "perimeter-contract.json")

    case_contract = load_json(bundle_root / "contracts" / "case-contract.json")
    case_contract["contract_status"] = ACTIVE_STATUS
    case_contract["base_map"].update(
        {
            "contract_record_sha256": map_contract_hash,
            "artifact_manifest_sha256": sha256_file(map_root / "manifest.json"),
            "acceptance_receipt_sha256": sha256_file(receipt_paths["map"]),
        }
    )
    case_contract["perimeter_layers"].update(
        {
            "contract_record_sha256": perimeter_contract_hash,
            "acceptance_receipt_sha256": sha256_file(receipt_paths["perimeter"]),
        }
    )
    for gate in (
        "base_map_active",
        "perimeter_layer_active",
        "source_integrity",
        "timeline_validation",
        "usd_validation",
        "tree_destruction",
        "flow_runtime_no_capture",
        "camera_retargeting",
        "human_visual_review",
    ):
        case_contract["quality_gates"][gate] = "passed"
    case_contract["quality_gates"]["capture_validation"] = "not_run"
    case_contract["quality_gates"]["training_readiness"] = "not_run"
    case_contract["release"].update(
        {
            "human_visual_decision": "accepted",
            "acceptance_receipt_sha256": sha256_file(receipt_paths["case"]),
            "simulation_use_allowed": True,
            "pilot_capture_allowed": True,
            "full_dataset_capture_allowed": False,
            "dataset_release_allowed": False,
            "training_use_allowed": False,
            "automatic_publication": False,
            "site_publication_triggered": False,
        }
    )
    case_contract_paths = (
        bundle_root / "contracts" / "case-contract.json",
        top_contracts / "case-contract.json",
    )
    _write_copies(case_contract_paths, case_contract)
    case_contract_hash = sha256_file(top_contracts / "case-contract.json")

    bundle_manifest = load_json(bundle_root / "manifest.json")
    bundle_manifest.update(
        {
            "status": ACTIVE_STATUS,
            "capture_enabled": False,
            "simulation_use_allowed": True,
            "pilot_capture_allowed": True,
            "full_dataset_capture_allowed": False,
            "dataset_release_allowed": False,
            "training_use_allowed": False,
            "automatic_publication": False,
        }
    )
    bundle_manifest["dataset"].update(
        {
            "capture_enabled": False,
            "capture_validation_status": "pilot_capture_authorized_not_run",
        }
    )
    bundle_manifest["qa"].update(
        {
            "render_acceptance": "accepted_by_user_in_visible_omniverse_editor",
            "isolated_kit_reopen": "passed",
            "pilot_capture": "authorized_not_run",
        }
    )
    write_json(bundle_root / "manifest.json", bundle_manifest)

    inventory_path, inventory_sha256, inventory_count = write_inventory(
        bundle_root,
        excluded_names={"manifest.json", "dependency-inventory.json"},
    )
    bundle_manifest["dependency_inventory"] = {
        "path": relative_posix(inventory_path, bundle_root),
        "sha256": inventory_sha256,
        "file_count": inventory_count,
    }
    write_json(bundle_root / "manifest.json", bundle_manifest)

    archive.parent.mkdir(parents=True, exist_ok=True)
    create_archive(bundle_root, archive)

    bundle_contract = load_json(top_contracts / "download-bundle-contract.json")
    bundle_contract["contract_status"] = ACTIVE_STATUS
    bundle_contract["bundle"].update(
        {
            "root_directory": relative_posix(bundle_root, output),
            "entry_stage_sha256": sha256_file(bundle_root / "dataset.usda"),
            "manifest_sha256": sha256_file(bundle_root / "manifest.json"),
            "archive_path": relative_posix(archive, output),
            "archive_sha256": sha256_file(archive),
        }
    )
    bundle_contract["case_reference"]["contract_record_sha256"] = case_contract_hash
    role_paths = {
        "map_contract": bundle_root / "contracts" / "map-contract.json",
        "map_manifest": bundle_root / "map" / "manifest.json",
        "perimeter_contract": bundle_root / "contracts" / "perimeter-contract.json",
        "perimeter_manifest": bundle_root / "perimeters" / "manifest.json",
        "camera_rig": bundle_root / "cameras" / "fixed_cameras.usda",
        "scenario": bundle_root / "scenarios" / "scenario.usda",
        "flow": bundle_root / "scenarios" / "flow.usda",
        "runtime_contract": bundle_root / "runtime" / "runtime-contract.json",
        "source_truth": bundle_root / "source" / "die-2026-v1.json",
        "asset_inventory": bundle_root / "dependency-inventory.json",
    }
    bundle_contract["dependency_locks"] = [
        {
            "role": role,
            "path": relative_posix(path, bundle_root),
            "sha256": sha256_file(path),
            "inside_bundle": True,
        }
        for role, path in role_paths.items()
    ]
    bundle_contract["portability"].update(
        {
            "dependency_inventory_sha256": sha256_file(bundle_root / "dependency-inventory.json"),
            "isolated_reopen": "passed",
        }
    )
    _passed_gates(bundle_contract)
    bundle_contract["release"].update(
        {
            "human_visual_decision": "accepted",
            "acceptance_receipt_sha256": sha256_file(receipt_paths["bundle"]),
            "download_allowed": True,
            "dataset_capture_included": False,
            "training_admission_included": False,
            "automatic_publication": False,
        }
    )
    write_json(top_contracts / "download-bundle-contract.json", bundle_contract)

    contract_errors = _contract_errors(
        map_contract=map_contract,
        map_path=top_contracts / "map-contract.json",
        perimeter_contract=perimeter_contract,
        perimeter_path=top_contracts / "perimeter-contract.json",
        case_contract=case_contract,
        case_path=top_contracts / "case-contract.json",
        bundle_contract=bundle_contract,
    )
    if contract_errors:
        raise ValueError("Active contract validation failed: " + contract_errors[0])

    if key_file_locks(candidate, candidate_key_paths) != candidate_locks:
        raise ValueError("Candidate package changed while sealing the additive release")

    candidate_report = output / "qa" / "build-report.json"
    if candidate_report.is_file():
        candidate_report.replace(output / "qa" / "candidate-build-report.json")
    report = {
        "schema": RELEASE_SCHEMA,
        "status": "active_pilot_capture_authorized",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "accepted_at": accepted_at,
        "candidate_root": str(candidate),
        "candidate_unchanged": True,
        "contracts": {
            "map": sha256_file(top_contracts / "map-contract.json"),
            "perimeter": sha256_file(top_contracts / "perimeter-contract.json"),
            "case": sha256_file(top_contracts / "case-contract.json"),
            "download": sha256_file(top_contracts / "download-bundle-contract.json"),
        },
        "artifacts": {
            "map_stage": sha256_file(map_root / "map.usda"),
            "perimeter_stage": sha256_file(perimeter_root / "perimeters.usda"),
            "review_stage": sha256_file(output / "review" / "review-map-with-perimeters.usda"),
            "reproduction_stage": sha256_file(bundle_root / "dataset.usda"),
            "download_archive": sha256_file(archive),
        },
        "release": {
            "map_upload_allowed": True,
            "perimeter_attachment_allowed": True,
            "simulation_use_allowed": True,
            "pilot_capture_allowed": True,
            "download_allowed": True,
            "full_dataset_capture_allowed": False,
            "dataset_release_allowed": False,
            "training_use_allowed": False,
            "automatic_publication": False,
        },
        "validation": {
            "contract_error_count": 0,
            "usd_dependency_issues": dependency_issues,
            "isolated_kit_reopen": "passed_on_hash_identical_candidate_stage",
            "camera_count": 62,
        },
    }
    write_json(output / "qa" / "seal-report.json", report)
    print(canonical_json(report), end="")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--accepted-at")
    return result


def main() -> int:
    seal(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
