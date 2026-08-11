"""Rebuild immutable Die upload and downloadable Omniverse packages.

This additive builder preserves all accepted source packages and dataset
outputs. It produces a map upload with eager site references, a separate
progressive perimeter layer, and aligned simulation/reproduction downloads.
Nothing is published or deleted by this module.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from build_die_contract_packages import (
    BUNDLE_ID,
    CASE_ID,
    CASE_PACKAGE_ID,
    MAP_PACKAGE_ID,
    MAP_REVISION,
    PERIMETER_PACKAGE_ID,
    PERIMETER_REVISION,
    align_runtime_contract,
    create_archive,
    sha256_file,
    usd_asset_issues,
    write_inventory,
    write_json,
    write_map_site,
    write_map_stage,
)
from fireviewer_accepted_visual_profiles import (
    PROFILE_CONTRACT_RELATIVE,
    PROFILE_LAYER_RELATIVE,
    write_accepted_visual_profile_artifacts,
)


SIMULATION_BUNDLE_ID = "fireviewer-die-2026-simulation-download-r1"
SIMULATION_PACKAGE_ID = "fireviewer-die-pontaix-r1-v4-simulation-aligned-v1"
SCHEMA = "fireviewer.omniverse-aligned-package-rebuild.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _locked(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    record = {
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if root is not None:
        record["path"] = path.relative_to(root).as_posix()
    return record


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, target, copy_function=shutil.copy2)


def _copy_children(source: Path, target: Path, names: Iterable[str]) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for name in names:
        child = source / name
        if child.is_dir():
            _copy_tree(child, target / name)
        elif child.is_file():
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target / name)
        else:
            raise FileNotFoundError(child)


def _write_qa_contract(root: Path, *, role: str, source_locks: dict[str, Any]) -> None:
    write_json(
        root / "qa" / "package-validation-contract.json",
        {
            "schema": "fireviewer.omniverse-package-validation-contract.v1",
            "role": role,
            "source_locks": source_locks,
            "structural_validation": "pending",
            "isolated_kit_reopen": "pending",
            "human_visual_review": "pending",
            "capture_running": False,
            "automatic_publication": False,
        },
    )


def _write_map_contract(root: Path, *, source_map: Path) -> dict[str, Any]:
    contract = {
        "schema": "fireviewer.map-upload-contract.v2",
        "package_id": MAP_PACKAGE_ID,
        "revision": MAP_REVISION,
        "entry_stage": "map.usda",
        "entry_stage_sha256": sha256_file(root / "map.usda"),
        "content_required": [
            "terrain",
            "orthophoto",
            "roads",
            "buildings",
            "vegetation_context",
            "vegetation_instances",
            "tree_assets_01_to_06",
            "environment",
        ],
        "composition_policy": "eager_references_no_optional_site_payloads",
        "excluded": ["simulation", "perimeters", "camera_rig", "capture_runtime"],
        "source_map": {
            "package_id": json.loads((source_map / "manifest.json").read_text(encoding="utf-8"))[
                "package_id"
            ],
            "manifest_sha256": sha256_file(source_map / "manifest.json"),
            "entry_stage_sha256": sha256_file(source_map / "map.usda"),
        },
        "upload_allowed": False,
        "automatic_publication": False,
        "human_visual_review": "pending",
    }
    write_json(root / "contracts" / "map-contract.json", contract)
    return contract


def build_map_upload(*, source_map: Path, output: Path) -> dict[str, Any]:
    print(canonical_json({"phase": "map_upload_r6", "output": str(output)}).strip(), flush=True)
    _copy_children(
        source_map,
        output,
        ("assets", "site", "source", "source-usd", "provenance"),
    )
    write_map_site(output / "site" / "site.usda", source_package_id=source_map.name)
    write_map_stage(output / "map.usda")
    site_text = (output / "site" / "site.usda").read_text(encoding="utf-8")
    if "prepend payload" in site_text or site_text.count("prepend references") != 5:
        raise ValueError("Map upload must eagerly reference all five site components")
    required_tokens = {
        "Terrain": "terrain",
        "Buildings": "buildings",
        "Routes": "roads",
        "VegetationContext": "vegetation context",
        "Vegetation": "tree instances",
    }
    for token, label in required_tokens.items():
        if f'"{token}"' not in site_text:
            raise ValueError(f"Map upload is missing {label}")
    issues = usd_asset_issues(output)
    if issues:
        raise ValueError("Map upload dependency validation failed: " + issues[0])
    source_manifest = json.loads((source_map / "manifest.json").read_text(encoding="utf-8"))
    contract = _write_map_contract(output, source_map=source_map)
    inventory_path, inventory_sha, file_count = write_inventory(
        output,
        excluded_names={"manifest.json", "dependency-inventory.json"},
    )
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "schema": "fireviewer.omniverse-complete-map-upload-package.v2",
            "package_id": MAP_PACKAGE_ID,
            "revision": MAP_REVISION,
            "status": "candidate_structural_validation_pending_visual_reopen",
            "entry_stage": "map.usda",
            "entry_stage_sha256": sha256_file(output / "map.usda"),
            "composition_policy": {
                "site_elements": "eager_references_loaded_on_normal_stage_open",
                "optional_payloads": False,
            },
            "content": [
                "terrain",
                "orthophoto",
                "roads",
                "buildings",
                "vegetation_context",
                "vegetation_instances",
                "six_tree_assets",
                "environment",
            ],
            "acceptance": {
                "source_map_accepted": True,
                "r6_visual_reopen": "pending",
            },
            "release": {
                "upload_allowed": False,
                "site_publication_triggered": False,
                "automatic_publication": False,
            },
            "dependency_inventory": {
                "path": inventory_path.relative_to(output).as_posix(),
                "sha256": inventory_sha,
                "file_count": file_count,
            },
            "automatic_publication": False,
        }
    )
    write_json(output / "manifest.json", manifest)
    _write_qa_contract(
        output,
        role="complete_map_upload_without_simulation_or_perimeter",
        source_locks=contract["source_map"],
    )
    return manifest


def build_perimeters(*, source_perimeters: Path, map_package: Path, output: Path) -> dict[str, Any]:
    print(canonical_json({"phase": "perimeter_r2", "output": str(output)}).strip(), flush=True)
    _copy_children(source_perimeters, output, ("authoring", "states", "source", "qa"))
    source_manifest = json.loads(
        (source_perimeters / "manifest.json").read_text(encoding="utf-8")
    )
    text = (source_perimeters / "perimeters.usda").read_text(encoding="utf-8")
    text = text.replace(str(source_manifest["layer_package_id"]), PERIMETER_PACKAGE_ID)
    (output / "perimeters.usda").write_text(text, encoding="utf-8", newline="\n")
    issues = usd_asset_issues(output)
    if issues:
        raise ValueError("Perimeter dependency validation failed: " + issues[0])
    inventory_path, inventory_sha, file_count = write_inventory(
        output,
        excluded_names={"manifest.json", "dependency-inventory.json"},
    )
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "layer_package_id": PERIMETER_PACKAGE_ID,
            "revision": PERIMETER_REVISION,
            "status": "candidate_compatible_with_map_r6_visual_reopen_pending",
            "entry_layer": "perimeters.usda",
            "entry_layer_sha256": sha256_file(output / "perimeters.usda"),
            "base_map": {
                "package_id": MAP_PACKAGE_ID,
                "revision": MAP_REVISION,
                "manifest_sha256": sha256_file(map_package / "manifest.json"),
            },
            "dependency_inventory": {
                "path": inventory_path.relative_to(output).as_posix(),
                "sha256": inventory_sha,
                "file_count": file_count,
            },
            "automatic_publication": False,
        }
    )
    manifest.pop("acceptance", None)
    manifest.pop("release", None)
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "contracts" / "perimeter-contract.json",
        {
            "schema": "fireviewer.progressive-perimeter-contract.v2",
            "layer_package_id": PERIMETER_PACKAGE_ID,
            "revision": PERIMETER_REVISION,
            "base_map_package_id": MAP_PACKAGE_ID,
            "entry_layer": "perimeters.usda",
            "entry_layer_sha256": sha256_file(output / "perimeters.usda"),
            "state_count": len(manifest["states"]),
            "map_upload_member": False,
            "attachment_mode": "separate_optional_layer",
            "automatic_publication": False,
            "human_visual_review": "pending",
        },
    )
    return manifest


def _write_simulation_stage(path: Path) -> None:
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 1080
    timeCodesPerSecond = 1
    framesPerSecond = 30
    subLayers = [
        @{PROFILE_LAYER_RELATIVE}@,
        @dataset.accepted.usda@
    ]
)

over "World"
{{
    custom string fireviewer:package_id = "{SIMULATION_PACKAGE_ID}"
    custom string fireviewer:bundle_id = "{SIMULATION_BUNDLE_ID}"
    custom string fireviewer:source_accepted_package_id = "fireviewer-die-pontaix-r1-v4"
    custom bool fireviewer:capture_on_first_launch = 0
    custom bool fireviewer:publication_allowed = 0
}}
''',
        encoding="utf-8",
        newline="\n",
    )


def _runtime_manifest(root: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": "runtime/runtime-contract.json",
        "contract_sha256": sha256_file(root / "runtime" / "runtime-contract.json"),
        "runner_module": runtime["runner_module"],
        "writer_module": runtime["writer_module"],
        "storage_module": runtime["storage_module"],
        "required_extensions": runtime["required_extensions"],
        "capture_on_first_launch": False,
    }


def build_simulation_download(*, source_package: Path, output: Path) -> dict[str, Any]:
    print(canonical_json({"phase": "simulation_download", "output": str(output)}).strip(), flush=True)
    directory_names = [
        item.name
        for item in sorted(source_package.iterdir())
        if item.is_dir() and item.name != "qa"
    ]
    _copy_children(source_package, output, directory_names)
    shutil.copy2(source_package / "dataset.usda", output / "dataset.accepted.usda")
    source_manifest = json.loads(
        (source_package / "manifest.json").read_text(encoding="utf-8")
    )
    visual_profiles = write_accepted_visual_profile_artifacts(output)
    runtime = align_runtime_contract(output, visual_profiles=visual_profiles)
    _write_simulation_stage(output / "dataset.usda")
    source_locks = {
        "package_id": source_manifest["package_id"],
        "entry_stage": _locked(source_package / "dataset.usda"),
        "manifest": _locked(source_package / "manifest.json"),
        "flow_layer": _locked(source_package / "scenarios" / "flow.usda"),
        "scenario_layer": _locked(source_package / "scenarios" / "scenario.usda"),
    }
    _write_qa_contract(output, role="aligned_simulation_download", source_locks=source_locks)
    issues = usd_asset_issues(output)
    if issues:
        raise ValueError("Simulation bundle dependency validation failed: " + issues[0])
    inventory_path, inventory_sha, file_count = write_inventory(
        output,
        excluded_names={"manifest.json", "dependency-inventory.json"},
    )
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "schema": "fireviewer.omniverse-aligned-simulation-download.v1",
            "bundle_id": SIMULATION_BUNDLE_ID,
            "package_id": SIMULATION_PACKAGE_ID,
            "status": "candidate_structural_validation_pending_visual_reopen",
            "entry_stage": "dataset.usda",
            "entry_stage_sha256": sha256_file(output / "dataset.usda"),
            "source_accepted_package": source_locks,
            "accepted_visual_profiles": {
                "contract": PROFILE_CONTRACT_RELATIVE,
                "contract_sha256": sha256_file(output / PROFILE_CONTRACT_RELATIVE),
                "persistent_layer": PROFILE_LAYER_RELATIVE,
                "persistent_layer_sha256": sha256_file(output / PROFILE_LAYER_RELATIVE),
                "flow_profile_id": visual_profiles["flow_profile"]["profile_id"],
                "flow_profile_sha256": visual_profiles["flow_profile"]["profile_sha256"],
                "sky_profile_id": visual_profiles["sky_profile"]["profile_id"],
                "sky_profile_sha256": visual_profiles["sky_profile"]["profile_sha256"],
            },
            "runtime": _runtime_manifest(output, runtime),
            "qa": {
                "source_scene_and_profiles_accepted": True,
                "new_package_structural_validation": "pending",
                "new_package_visual_reopen": "pending",
            },
            "dependency_inventory": {
                "path": inventory_path.relative_to(output).as_posix(),
                "sha256": inventory_sha,
                "file_count": file_count,
            },
            "capture_enabled": False,
            "automatic_publication": False,
        }
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def _write_reproduction_stage(path: Path) -> None:
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 1260
    timeCodesPerSecond = 1
    framesPerSecond = 30
    customLayerData = {{
        dictionary renderSettings = {{
            bool "rtx:flow:enabled" = 1
            int "rtx:flow:maxBlocks" = 16384
        }}
    }}
    subLayers = [
        @{PROFILE_LAYER_RELATIVE}@,
        @map/map.usda@,
        @perimeters/perimeters.usda@,
        @scenarios/scenario.usda@,
        @scenarios/flow.usda@,
        @appearance/appearance.usda@,
        @cameras/fixed_cameras.usda@
    ]
)

over "World"
{{
    custom string fireviewer:package_id = "{CASE_PACKAGE_ID}"
    custom string fireviewer:bundle_id = "{BUNDLE_ID}"
    custom string fireviewer:case_id = "{CASE_ID}"
    custom string fireviewer:map_package_id = "{MAP_PACKAGE_ID}"
    custom string fireviewer:perimeter_layer_package_id = "{PERIMETER_PACKAGE_ID}"
    custom bool fireviewer:capture_on_first_launch = 0
    custom bool fireviewer:publication_allowed = 0
}}
''',
        encoding="utf-8",
        newline="\n",
    )


def build_reproduction_download(
    *,
    source_reproduction: Path,
    map_package: Path,
    perimeter_package: Path,
    output: Path,
) -> dict[str, Any]:
    print(canonical_json({"phase": "reproduction_download", "output": str(output)}).strip(), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    _copy_tree(map_package, output / "map")
    _copy_tree(perimeter_package, output / "perimeters")
    for name in ("scenarios", "appearance", "cameras", "source", "runtime"):
        _copy_tree(source_reproduction / name, output / name)
    source_manifest = json.loads(
        (source_reproduction / "manifest.json").read_text(encoding="utf-8")
    )
    visual_profiles = write_accepted_visual_profile_artifacts(output)
    runtime = align_runtime_contract(output, visual_profiles=visual_profiles)
    _write_reproduction_stage(output / "dataset.usda")
    source_locks = {
        "bundle_id": source_manifest["bundle_id"],
        "package_id": source_manifest["package_id"],
        "manifest": _locked(source_reproduction / "manifest.json"),
        "entry_stage": _locked(source_reproduction / "dataset.usda"),
        "flow_layer": _locked(source_reproduction / "scenarios" / "flow.usda"),
        "scenario_layer": _locked(source_reproduction / "scenarios" / "scenario.usda"),
    }
    _write_qa_contract(output, role="aligned_die_reproduction_download", source_locks=source_locks)
    issues = usd_asset_issues(output)
    if issues:
        raise ValueError("Reproduction bundle dependency validation failed: " + issues[0])
    inventory_path, inventory_sha, file_count = write_inventory(
        output,
        excluded_names={"manifest.json", "dependency-inventory.json"},
    )
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "schema": "fireviewer.omniverse-aligned-reproduction-download.v2",
            "bundle_id": BUNDLE_ID,
            "package_id": CASE_PACKAGE_ID,
            "status": "candidate_structural_validation_pending_visual_reopen",
            "entry_stage": "dataset.usda",
            "entry_stage_sha256": sha256_file(output / "dataset.usda"),
            "base_scene": {
                "package_id": MAP_PACKAGE_ID,
                "path": "map",
                "manifest_sha256": sha256_file(output / "map" / "manifest.json"),
                "composition": "internal_complete_map_eager_site_references",
            },
            "perimeter_layer": {
                "layer_package_id": PERIMETER_PACKAGE_ID,
                "path": "perimeters/perimeters.usda",
                "sha256": sha256_file(output / "perimeters" / "perimeters.usda"),
                "map_upload_member": False,
            },
            "source_validated_reproduction": source_locks,
            "accepted_visual_profiles": {
                "contract": PROFILE_CONTRACT_RELATIVE,
                "contract_sha256": sha256_file(output / PROFILE_CONTRACT_RELATIVE),
                "persistent_layer": PROFILE_LAYER_RELATIVE,
                "persistent_layer_sha256": sha256_file(output / PROFILE_LAYER_RELATIVE),
                "flow_profile_id": visual_profiles["flow_profile"]["profile_id"],
                "flow_profile_sha256": visual_profiles["flow_profile"]["profile_sha256"],
                "sky_profile_id": visual_profiles["sky_profile"]["profile_id"],
                "sky_profile_sha256": visual_profiles["sky_profile"]["profile_sha256"],
            },
            "composition": {
                "accepted_visual_profiles": PROFILE_LAYER_RELATIVE,
                "map": "map/map.usda",
                "perimeters": "perimeters/perimeters.usda",
                "scenario": "scenarios/scenario.usda",
                "flow": "scenarios/flow.usda",
                "appearance": "appearance/appearance.usda",
                "camera_rig": "cameras/fixed_cameras.usda",
            },
            "runtime": _runtime_manifest(output, runtime),
            "qa": {
                "source_reproduction_and_profiles_accepted": True,
                "new_package_structural_validation": "pending",
                "new_package_visual_reopen": "pending",
                "controlled_pilot_required_before_new_capture": True,
            },
            "dependency_inventory": {
                "path": inventory_path.relative_to(output).as_posix(),
                "sha256": inventory_sha,
                "file_count": file_count,
            },
            "capture_enabled": False,
            "dataset_release_allowed": False,
            "training_use_allowed": False,
            "automatic_publication": False,
        }
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def _archive_receipt(archive: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": archive.relative_to(root).as_posix(),
        "sha256": sha256_file(archive),
        "byte_count": archive.stat().st_size,
        "integrity": "zip_test_passed",
    }


def build(args: argparse.Namespace) -> Path:
    source_base = args.accepted_simulation.resolve()
    source_map = args.source_map.resolve()
    source_perimeters = args.source_perimeters.resolve()
    source_reproduction = args.source_reproduction.resolve()
    output_root = args.output_root.resolve()
    required = (
        source_base / "dataset.usda",
        source_base / "manifest.json",
        source_base / "scenarios" / "flow.usda",
        source_map / "map.usda",
        source_map / "manifest.json",
        source_perimeters / "perimeters.usda",
        source_perimeters / "manifest.json",
        source_reproduction / "dataset.usda",
        source_reproduction / "manifest.json",
        source_reproduction / "scenarios" / "flow.usda",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    source_locks = {
        "schema": "fireviewer.aligned-package-source-locks.v1",
        "accepted_simulation": {
            "manifest": _locked(source_base / "manifest.json"),
            "entry_stage": _locked(source_base / "dataset.usda"),
        },
        "source_map": {
            "manifest": _locked(source_map / "manifest.json"),
            "entry_stage": _locked(source_map / "map.usda"),
        },
        "source_perimeters": {
            "manifest": _locked(source_perimeters / "manifest.json"),
            "entry_layer": _locked(source_perimeters / "perimeters.usda"),
        },
        "source_reproduction": {
            "manifest": _locked(source_reproduction / "manifest.json"),
            "entry_stage": _locked(source_reproduction / "dataset.usda"),
        },
    }
    write_json(output_root / "preservation" / "source-locks.json", source_locks)

    map_output = output_root / "map" / MAP_PACKAGE_ID
    perimeter_output = output_root / "perimeters" / PERIMETER_PACKAGE_ID
    simulation_output = output_root / "simulation" / SIMULATION_BUNDLE_ID
    reproduction_output = output_root / "reproduction" / BUNDLE_ID
    map_manifest = build_map_upload(source_map=source_map, output=map_output)
    perimeter_manifest = build_perimeters(
        source_perimeters=source_perimeters,
        map_package=map_output,
        output=perimeter_output,
    )
    simulation_manifest = build_simulation_download(
        source_package=source_base,
        output=simulation_output,
    )
    reproduction_manifest = build_reproduction_download(
        source_reproduction=source_reproduction,
        map_package=map_output,
        perimeter_package=perimeter_output,
        output=reproduction_output,
    )

    archives = output_root / "archives"
    simulation_archive = archives / f"{SIMULATION_BUNDLE_ID}.zip"
    reproduction_archive = archives / f"{BUNDLE_ID}.zip"
    create_archive(simulation_output, simulation_archive)
    create_archive(reproduction_output, reproduction_archive)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_packages_built_sources_and_dataset_outputs_preserved",
        "deletion_performed": False,
        "publication_performed": False,
        "source_locks": source_locks,
        "map_upload": {
            "package_id": map_manifest["package_id"],
            "entry_stage": map_output.relative_to(output_root).as_posix() + "/map.usda",
            "entry_stage_sha256": sha256_file(map_output / "map.usda"),
            "content_loading": "eager_terrain_roads_buildings_vegetation_environment",
            "simulation_included": False,
            "perimeter_included": False,
        },
        "perimeters": {
            "layer_package_id": perimeter_manifest["layer_package_id"],
            "entry_layer_sha256": sha256_file(perimeter_output / "perimeters.usda"),
            "separate_from_map_upload": True,
        },
        "simulation_download": {
            "bundle_id": simulation_manifest["bundle_id"],
            "entry_stage_sha256": sha256_file(simulation_output / "dataset.usda"),
            "archive": _archive_receipt(simulation_archive, root=output_root),
        },
        "reproduction_download": {
            "bundle_id": reproduction_manifest["bundle_id"],
            "entry_stage_sha256": sha256_file(reproduction_output / "dataset.usda"),
            "archive": _archive_receipt(reproduction_archive, root=output_root),
        },
        "quality_gates": {
            "relative_usd_dependencies": "passed",
            "archive_integrity": "passed",
            "accepted_flow_sky_profiles_packaged": "passed",
            "multimodal_runtime_packaged": "passed",
            "isolated_kit_reopen": "pending",
            "human_visual_review": "pending",
        },
    }
    write_json(output_root / "qa" / "build-report.json", report)
    print(canonical_json(report), end="", flush=True)
    return output_root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--accepted-simulation", type=Path, required=True)
    result.add_argument("--source-map", type=Path, required=True)
    result.add_argument("--source-perimeters", type=Path, required=True)
    result.add_argument("--source-reproduction", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    return result


def main() -> int:
    build(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
