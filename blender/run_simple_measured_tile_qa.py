"""Prepare, execute and verify tile-local Blender QA for one simple package."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


BLENDER_ROOT = Path(__file__).resolve().parent
OMNIVERSE_ROOT = BLENDER_ROOT.parent / "omniverse"
for _module_root in (BLENDER_ROOT, OMNIVERSE_ROOT):
    if str(_module_root) not in sys.path:
        sys.path.insert(0, str(_module_root))

import validate_measured_scene as measured_qa  # noqa: E402
from build_measured_scene_usd import (  # noqa: E402
    validate_measured_scene_package,
)


CONTRACT_SCHEMA = "fireviewer.simple-measured-tile-qa-runner-contract.v1"
LINK_SCHEMA = "fireviewer.simple-measured-tile-qa-link.v1"
SIMPLE_RECEIPT_SCHEMA = "fireviewer.simple-measured-tile-production.v1"
SCENE_RECEIPT_SCHEMA = "fireviewer.measured-scene-receipt.v1"
JOB_NAME = "measured-scene-render-job.v1.json"
SIMPLE_RECEIPT_NAME = "simple-measured-tile-receipt.v1.json"
TECHNICAL_RECEIPT_NAME = measured_qa.RECEIPT_FILE_NAME
LINK_RECEIPT_NAME = "simple-measured-tile-qa-link.v1.json"
CAPTURE_DIRECTORY = Path("qa") / "renders"
RUNTIME_DIRECTORY = Path("qa") / "runtime"
BLENDER_VERSION = re.compile(r"(?m)^Blender\s+(4\.5(?:\.\d+)?)\b")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ENVIRONMENT_PATHS = {
    "TEMP": Path("temp"),
    "TMP": Path("tmp"),
    "PYTHONPYCACHEPREFIX": Path("python-cache"),
    "BLENDER_USER_CONFIG": Path("blender") / "config",
    "BLENDER_USER_SCRIPTS": Path("blender") / "scripts",
    "BLENDER_USER_DATAFILES": Path("blender") / "datafiles",
    "BLENDER_USER_EXTENSIONS": Path("blender") / "extensions",
}


class SimpleMeasuredTileQaRunnerError(RuntimeError):
    """A package, Blender invocation or QA linkage is invalid."""


@dataclass(frozen=True, slots=True)
class PackageState:
    root: Path
    simple_receipt_path: Path
    simple_receipt: Mapping[str, Any]
    scene_path: Path
    scene_receipt_path: Path
    scene_receipt: Mapping[str, Any]
    ground_texture_path: Path
    expected_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[Sequence[str], Mapping[str, str]], ProcessResult]


def _canonical_bytes(value: Any) -> bytes:
    return measured_qa.canonical_json_bytes(value)


def _compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimpleMeasuredTileQaRunnerError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise SimpleMeasuredTileQaRunnerError(f"{label} must be a JSON object")
    return value


def _contract_path() -> Path:
    return Path(__file__).with_name("simple_measured_tile_qa_runner_contract.v1.json")


def _load_contract() -> dict[str, Any]:
    contract = _read_json(_contract_path(), "simple measured tile QA contract")
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != "locked":
        raise SimpleMeasuredTileQaRunnerError(
            "Unsupported or unlocked QA runner contract"
        )
    if contract.get("scope") != {
        "tile_count": 1,
        "network": "forbidden",
        "pbr": "forbidden",
        "automatic_human_acceptance": False,
    }:
        raise SimpleMeasuredTileQaRunnerError("QA runner scope changed")
    blender = contract.get("blender")
    if not isinstance(blender, Mapping) or blender.get(
        "temporary_environment_variables"
    ) != list(ENVIRONMENT_PATHS):
        raise SimpleMeasuredTileQaRunnerError("QA runner Blender environment changed")
    if contract.get("phases") != ["prepare", "execute", "verify"]:
        raise SimpleMeasuredTileQaRunnerError("QA runner phases changed")
    return contract


def _require_d_path(
    value: Path | str,
    label: str,
    *,
    kind: str | None = None,
) -> Path:
    lexical_drive = PureWindowsPath(str(value)).drive.upper()
    if lexical_drive and lexical_drive != "D:":
        raise SimpleMeasuredTileQaRunnerError(f"{label} must remain on D:; got {value}")
    try:
        path = Path(value).resolve(strict=kind is not None)
    except OSError as error:
        raise SimpleMeasuredTileQaRunnerError(f"Missing {label}: {value}") from error
    if os.name == "nt" and path.drive.upper() != "D:":
        raise SimpleMeasuredTileQaRunnerError(f"{label} must remain on D:; got {path}")
    if kind == "file" and not path.is_file():
        raise SimpleMeasuredTileQaRunnerError(f"{label} is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise SimpleMeasuredTileQaRunnerError(f"{label} is not a directory: {path}")
    return path


def _relative_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SimpleMeasuredTileQaRunnerError(
            f"{label} must be a portable relative path"
        )
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SimpleMeasuredTileQaRunnerError(f"{label} escapes the package")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SimpleMeasuredTileQaRunnerError(f"{label} escapes the package") from error
    return path


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise SimpleMeasuredTileQaRunnerError(
            "QA artifact escapes its tile package"
        ) from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SimpleMeasuredTileQaRunnerError(
            f"Missing or empty QA artifact: {resolved}"
        )
    return {
        "path": relative,
        "byte_count": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _validate_artifact(record: Any, root: Path, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "byte_count",
        "sha256",
    }:
        raise SimpleMeasuredTileQaRunnerError(f"{label} artifact record is invalid")
    path = _relative_file(root, record.get("path"), f"{label}.path")
    if not path.is_file():
        raise SimpleMeasuredTileQaRunnerError(f"{label} is missing: {path}")
    if record.get("byte_count") != path.stat().st_size:
        raise SimpleMeasuredTileQaRunnerError(f"{label} byte count changed")
    digest = record.get("sha256")
    if not isinstance(digest, str) or HASH_PATTERN.fullmatch(digest) is None:
        raise SimpleMeasuredTileQaRunnerError(f"{label} SHA-256 is invalid")
    if _sha256_file(path) != digest:
        raise SimpleMeasuredTileQaRunnerError(f"{label} hash changed")
    return path


def inspect_simple_package(package_root: Path | str) -> PackageState:
    """Rehash the source-free package and derive counts from scene.done.json."""

    root = _require_d_path(package_root, "simple tile package", kind="directory")
    simple_path = root / SIMPLE_RECEIPT_NAME
    simple_receipt = _read_json(simple_path, "simple tile receipt")
    if (
        simple_receipt.get("schema") != SIMPLE_RECEIPT_SCHEMA
        or simple_receipt.get("status") != "technical_pilot_non_final"
        or simple_receipt.get("accepted_final") is not False
    ):
        raise SimpleMeasuredTileQaRunnerError("Simple tile receipt status is invalid")
    declared = simple_receipt.get("receipt_sha256")
    without_hash = dict(simple_receipt)
    without_hash.pop("receipt_sha256", None)
    if declared != _sha256_bytes(_compact_bytes(without_hash)):
        raise SimpleMeasuredTileQaRunnerError("Simple tile receipt hash is invalid")
    request = simple_receipt.get("request")
    if not isinstance(request, Mapping) or simple_receipt.get(
        "build_id"
    ) != _sha256_bytes(_compact_bytes(request)):
        raise SimpleMeasuredTileQaRunnerError("Simple tile build identity is invalid")
    outputs = simple_receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise SimpleMeasuredTileQaRunnerError("Simple tile output records are missing")
    for relative_name, record in outputs.items():
        if not isinstance(relative_name, str) or not isinstance(record, Mapping):
            raise SimpleMeasuredTileQaRunnerError(
                "Simple tile output record is malformed"
            )
        if record.get("path") != relative_name:
            raise SimpleMeasuredTileQaRunnerError("Simple tile output path differs")
        _validate_artifact(record, root, f"package output {relative_name}")
    required = {
        "scene/scene.usda",
        "scene/scene.done.json",
        "ground/ground-color.png",
    }
    if not required.issubset(outputs):
        raise SimpleMeasuredTileQaRunnerError("Simple tile lacks QA source artifacts")

    scene_receipt_path = root / "scene" / "scene.done.json"
    try:
        scene_receipt = validate_measured_scene_package(root / "scene")
    except Exception as error:
        raise SimpleMeasuredTileQaRunnerError(
            f"Measured scene package is invalid: {error}"
        ) from error
    if (
        scene_receipt.get("schema") != SCENE_RECEIPT_SCHEMA
        or scene_receipt.get("accepted_final") is not False
    ):
        raise SimpleMeasuredTileQaRunnerError(
            "Measured scene receipt status is invalid"
        )
    reconciliation = scene_receipt.get("reconciliation")
    if not isinstance(reconciliation, Mapping):
        raise SimpleMeasuredTileQaRunnerError(
            "Measured scene reconciliation is missing"
        )
    counts: dict[str, int] = {"terrain_objects": 1}
    for family, target in (
        ("buildings", "building_instances"),
        ("trees", "tree_instances"),
        ("context_assets", "context_asset_instances"),
    ):
        record = reconciliation.get(family)
        count = (
            record.get("instance_count")
            if isinstance(record, Mapping)
            else 0
            if family == "context_assets"
            else None
        )
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SimpleMeasuredTileQaRunnerError(
                f"Measured scene {family} instance count is invalid"
            )
        counts[target] = count
    placement = simple_receipt.get("placement")
    if isinstance(placement, Mapping) and (
        placement.get("building_valid_count") != counts["building_instances"]
        or placement.get("tree_valid_count") != counts["tree_instances"]
        or placement.get("context_asset_valid_count", 0)
        != counts["context_asset_instances"]
    ):
        raise SimpleMeasuredTileQaRunnerError(
            "Simple tile placement and scene counts differ"
        )
    return PackageState(
        root=root,
        simple_receipt_path=simple_path,
        simple_receipt=simple_receipt,
        scene_path=root / "scene" / "scene.usda",
        scene_receipt_path=scene_receipt_path,
        scene_receipt=scene_receipt,
        ground_texture_path=root / "ground" / "ground-color.png",
        expected_counts=counts,
    )


def _write_same_or_reject(path: Path, payload: Mapping[str, Any]) -> None:
    content = _canonical_bytes(payload)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise SimpleMeasuredTileQaRunnerError(
                f"Refusing to replace different locked QA artifact: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_job(package_root: Path | str, blender_path: Path | str) -> Path:
    """Create and inspect one tile-local public validator job."""

    _load_contract()
    state = inspect_simple_package(package_root)
    _require_d_path(blender_path, "Blender executable", kind="file")
    job_path = state.root / JOB_NAME
    job = measured_qa.build_job(
        job_root=state.root,
        scene_path=state.scene_path,
        ground_texture_path=state.ground_texture_path,
        output_directory=state.root / CAPTURE_DIRECTORY,
        expected_counts=state.expected_counts,
    )
    _write_same_or_reject(job_path, job)
    inspected = measured_qa.inspect_job(job_path, require_d=True)
    if inspected.expected_counts != state.expected_counts:
        raise SimpleMeasuredTileQaRunnerError(
            "Locked job counts differ from scene receipt"
        )
    return job_path


def _runtime_environment(root: Path) -> tuple[dict[str, str], Path]:
    runtime = (root / RUNTIME_DIRECTORY).resolve()
    try:
        runtime.relative_to(root)
    except ValueError as error:
        raise SimpleMeasuredTileQaRunnerError("Unsafe QA runtime directory") from error
    environment = dict(os.environ)
    for name, relative in ENVIRONMENT_PATHS.items():
        target = _require_d_path(runtime / relative, name)
        target.mkdir(parents=True, exist_ok=True)
        environment[name] = str(target)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment, runtime


def _default_runner(
    command: Sequence[str], environment: Mapping[str, str]
) -> ProcessResult:
    completed = subprocess.run(
        list(command),
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    return ProcessResult(completed.returncode, completed.stdout, completed.stderr)


def _blender_version(
    blender: Path,
    *,
    environment: Mapping[str, str],
    command_runner: CommandRunner,
) -> str:
    result = command_runner((str(blender), "--version"), environment)
    if result.returncode != 0:
        raise SimpleMeasuredTileQaRunnerError("Blender --version failed")
    match = BLENDER_VERSION.search(f"{result.stdout}\n{result.stderr}")
    if match is None:
        raise SimpleMeasuredTileQaRunnerError("Blender must be version 4.5")
    return match.group(1)


def _technical_receipt(state: PackageState) -> Mapping[str, Any]:
    receipt_path = state.root / TECHNICAL_RECEIPT_NAME
    try:
        receipt = measured_qa.verify_receipt(receipt_path, require_d=True)
    except Exception as error:
        raise SimpleMeasuredTileQaRunnerError(
            f"Technical QA receipt is invalid: {error}"
        ) from error
    if (
        receipt.get("status") != measured_qa.TECHNICAL_STATUS
        or receipt.get("human_review_required") is not True
        or receipt.get("counts") != state.expected_counts
    ):
        raise SimpleMeasuredTileQaRunnerError(
            "Technical QA receipt status or counts differ"
        )
    return receipt


def _link_payload(
    state: PackageState,
    *,
    blender: Path,
    blender_version: str,
    job_path: Path,
    technical_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    package_artifact = _artifact(state.simple_receipt_path, state.root)
    scene_artifact = _artifact(state.scene_receipt_path, state.root)
    job_artifact = _artifact(job_path, state.root)
    qa_artifact = _artifact(state.root / TECHNICAL_RECEIPT_NAME, state.root)
    link_set = {
        "package": package_artifact,
        "scene": scene_artifact,
        "job": job_artifact,
        "technical_qa": qa_artifact,
        "qa_evidence_set_sha256": technical_receipt["evidence_set_sha256"],
    }
    payload: dict[str, Any] = {
        "schema": LINK_SCHEMA,
        "status": measured_qa.TECHNICAL_STATUS,
        "human_review_required": True,
        "accepted_human": False,
        "automatic_acceptance": False,
        "package_build_id": state.simple_receipt["build_id"],
        "scene_build_id": state.scene_receipt.get("build_id"),
        "expected_counts": dict(state.expected_counts),
        "package": package_artifact,
        "scene": scene_artifact,
        "job": job_artifact,
        "technical_qa": qa_artifact,
        "blender": {
            "file_name": blender.name,
            "byte_count": blender.stat().st_size,
            "sha256": _sha256_file(blender),
            "version": blender_version,
        },
        "runner": {
            "contract_sha256": _sha256_file(_contract_path()),
            "algorithm_sha256": _sha256_file(Path(__file__)),
        },
        "link_set_sha256": measured_qa.canonical_sha256(link_set),
    }
    payload["link_content_sha256"] = measured_qa.canonical_sha256(payload)
    return payload


def _publish_link(
    state: PackageState,
    *,
    blender: Path,
    blender_version: str,
    job_path: Path,
    technical_receipt: Mapping[str, Any],
) -> Path:
    path = state.root / LINK_RECEIPT_NAME
    _write_same_or_reject(
        path,
        _link_payload(
            state,
            blender=blender,
            blender_version=blender_version,
            job_path=job_path,
            technical_receipt=technical_receipt,
        ),
    )
    return path


def execute_qa(
    package_root: Path | str,
    blender_path: Path | str,
    *,
    command_runner: CommandRunner | None = None,
) -> Path:
    """Run Blender once, verify public evidence, and publish a non-accepting link."""

    job_path = prepare_job(package_root, blender_path)
    state = inspect_simple_package(package_root)
    blender = _require_d_path(blender_path, "Blender executable", kind="file")
    runner = command_runner or _default_runner
    environment, runtime = _runtime_environment(state.root)
    try:
        version = _blender_version(
            blender, environment=environment, command_runner=runner
        )
        link_path = state.root / LINK_RECEIPT_NAME
        if link_path.is_file():
            verify_link(state.root, blender)
            return link_path
        technical_path = state.root / TECHNICAL_RECEIPT_NAME
        if not technical_path.is_file():
            command = (
                str(blender),
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python",
                str((BLENDER_ROOT / "validate_measured_scene.py").resolve(strict=True)),
                "--",
                "--job",
                str(job_path),
            )
            result = runner(command, environment)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[-2000:]
                raise SimpleMeasuredTileQaRunnerError(
                    f"Blender measured-scene QA failed: {detail}"
                )
        technical = _technical_receipt(state)
        return _publish_link(
            state,
            blender=blender,
            blender_version=version,
            job_path=job_path,
            technical_receipt=technical,
        )
    finally:
        if runtime.exists():
            resolved = runtime.resolve()
            if resolved == (
                state.root / RUNTIME_DIRECTORY
            ).resolve() and resolved.is_relative_to(state.root):
                shutil.rmtree(resolved)


def verify_link(
    package_root: Path | str,
    blender_path: Path | str,
) -> dict[str, Any]:
    """Rehash the package, job, public QA receipt and linkage without Blender."""

    _load_contract()
    state = inspect_simple_package(package_root)
    blender = _require_d_path(blender_path, "Blender executable", kind="file")
    path = state.root / LINK_RECEIPT_NAME
    payload = _read_json(path, "simple tile QA link receipt")
    if payload.get("schema") != LINK_SCHEMA:
        raise SimpleMeasuredTileQaRunnerError("QA link receipt schema is invalid")
    declared = payload.get("link_content_sha256")
    without_hash = dict(payload)
    without_hash.pop("link_content_sha256", None)
    if declared != measured_qa.canonical_sha256(without_hash):
        raise SimpleMeasuredTileQaRunnerError("QA link receipt hash is invalid")
    if (
        payload.get("status") != measured_qa.TECHNICAL_STATUS
        or payload.get("human_review_required") is not True
        or payload.get("accepted_human") is not False
        or payload.get("automatic_acceptance") is not False
    ):
        raise SimpleMeasuredTileQaRunnerError("QA link cannot grant human acceptance")
    if payload.get("package_build_id") != state.simple_receipt.get("build_id"):
        raise SimpleMeasuredTileQaRunnerError("QA link package build changed")
    if payload.get("scene_build_id") != state.scene_receipt.get("build_id"):
        raise SimpleMeasuredTileQaRunnerError("QA link scene build changed")
    if payload.get("expected_counts") != state.expected_counts:
        raise SimpleMeasuredTileQaRunnerError("QA link expected counts changed")
    expected_artifacts = {
        "package": _artifact(state.simple_receipt_path, state.root),
        "scene": _artifact(state.scene_receipt_path, state.root),
        "job": _artifact(state.root / JOB_NAME, state.root),
        "technical_qa": _artifact(state.root / TECHNICAL_RECEIPT_NAME, state.root),
    }
    for label, expected in expected_artifacts.items():
        if payload.get(label) != expected:
            raise SimpleMeasuredTileQaRunnerError(f"QA link {label} artifact changed")
        _validate_artifact(payload[label], state.root, label)
    inspected = measured_qa.inspect_job(state.root / JOB_NAME, require_d=True)
    if inspected.expected_counts != state.expected_counts:
        raise SimpleMeasuredTileQaRunnerError("QA job counts changed")
    technical = _technical_receipt(state)
    link_set = {
        **expected_artifacts,
        "qa_evidence_set_sha256": technical["evidence_set_sha256"],
    }
    if payload.get("link_set_sha256") != measured_qa.canonical_sha256(link_set):
        raise SimpleMeasuredTileQaRunnerError("QA link evidence set changed")
    expected_blender = {
        "file_name": blender.name,
        "byte_count": blender.stat().st_size,
        "sha256": _sha256_file(blender),
        "version": payload.get("blender", {}).get("version"),
    }
    if (
        not isinstance(expected_blender["version"], str)
        or BLENDER_VERSION.fullmatch(f"Blender {expected_blender['version']}") is None
        or payload.get("blender") != expected_blender
    ):
        raise SimpleMeasuredTileQaRunnerError("QA link Blender identity changed")
    if payload.get("runner") != {
        "contract_sha256": _sha256_file(_contract_path()),
        "algorithm_sha256": _sha256_file(Path(__file__)),
    }:
        raise SimpleMeasuredTileQaRunnerError("QA runner identity changed")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "execute", "verify"), required=True
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.phase == "prepare":
        artifact = prepare_job(options.package_root, options.blender)
        status = "job_prepared"
    elif options.phase == "execute":
        artifact = execute_qa(options.package_root, options.blender)
        status = measured_qa.TECHNICAL_STATUS
    else:
        verify_link(options.package_root, options.blender)
        artifact = Path(options.package_root).resolve() / LINK_RECEIPT_NAME
        status = measured_qa.TECHNICAL_STATUS
    print(
        json.dumps(
            {
                "schema": LINK_SCHEMA,
                "phase": options.phase,
                "status": status,
                "artifact": str(artifact),
                "human_review_required": options.phase != "prepare",
                "accepted_human": False,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CONTRACT_SCHEMA",
    "LINK_SCHEMA",
    "LINK_RECEIPT_NAME",
    "ProcessResult",
    "SimpleMeasuredTileQaRunnerError",
    "execute_qa",
    "inspect_simple_package",
    "main",
    "prepare_job",
    "verify_link",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
