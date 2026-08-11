"""Seal and replay explicit human acceptance of one measured-tile QA proof."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Mapping, Sequence

import run_simple_measured_tile_qa as tile_qa
import validate_measured_scene as measured_qa


CONTRACT_SCHEMA = "fireviewer.simple-measured-tile-human-acceptance-contract.v1"
ACCEPTANCE_SCHEMA = "fireviewer.simple-measured-tile-human-acceptance.v1"
ACCEPTANCE_STATUS = "accepted_human_visual"
ACCEPTANCE_NAME = "simple-measured-tile-human-acceptance.v1.json"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SimpleMeasuredTileHumanAcceptanceError(RuntimeError):
    """The explicit review or its hash-bound technical evidence is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return measured_qa.canonical_json_bytes(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimpleMeasuredTileHumanAcceptanceError(
            f"Invalid {label}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SimpleMeasuredTileHumanAcceptanceError(f"{label} must be a JSON object")
    return payload


def _contract_path() -> Path:
    return Path(__file__).with_name(
        "simple_measured_tile_human_acceptance_contract.v1.json"
    )


def _load_contract() -> dict[str, Any]:
    contract = _read_json(_contract_path(), "human acceptance contract")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "locked"
        or contract.get("automatic_acceptance") is not False
        or contract.get("accepted_status") != ACCEPTANCE_STATUS
    ):
        raise SimpleMeasuredTileHumanAcceptanceError(
            "Unsupported or unlocked human acceptance contract"
        )
    return contract


def _require_d_package(package_root: Path | str) -> Path:
    lexical_drive = PureWindowsPath(str(package_root)).drive.upper()
    if lexical_drive and lexical_drive != "D:":
        raise SimpleMeasuredTileHumanAcceptanceError(
            f"Package must remain on D:; got {package_root}"
        )
    try:
        root = Path(package_root).resolve(strict=True)
    except OSError as error:
        raise SimpleMeasuredTileHumanAcceptanceError(
            f"Missing package: {package_root}"
        ) from error
    if not root.is_dir() or (root.drive and root.drive.upper() != "D:"):
        raise SimpleMeasuredTileHumanAcceptanceError(
            f"Package must be a D: directory; got {root}"
        )
    return root


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise SimpleMeasuredTileHumanAcceptanceError(
            f"Acceptance artifact escapes or is missing: {path}"
        ) from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SimpleMeasuredTileHumanAcceptanceError(
            f"Acceptance artifact is empty: {resolved}"
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
        raise SimpleMeasuredTileHumanAcceptanceError(
            f"{label} artifact record is invalid"
        )
    relative = record.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise SimpleMeasuredTileHumanAcceptanceError(f"{label} path is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SimpleMeasuredTileHumanAcceptanceError(
            f"{label} escapes the package"
        ) from error
    if not path.is_file() or record.get("byte_count") != path.stat().st_size:
        raise SimpleMeasuredTileHumanAcceptanceError(f"{label} size changed")
    digest = record.get("sha256")
    if (
        not isinstance(digest, str)
        or HASH_PATTERN.fullmatch(digest) is None
        or _sha256_file(path) != digest
    ):
        raise SimpleMeasuredTileHumanAcceptanceError(f"{label} hash changed")
    return path


def _review(statement: str) -> dict[str, str]:
    if (
        not isinstance(statement, str)
        or statement != statement.strip()
        or not statement
        or len(statement) > 1000
        or any(ord(character) < 32 for character in statement)
    ):
        raise SimpleMeasuredTileHumanAcceptanceError(
            "A non-empty explicit human review statement is required"
        )
    return {
        "kind": "human",
        "verdict": "accepted",
        "statement": statement,
    }


def _expected_payload(
    root: Path,
    blender_path: Path | str,
    *,
    review_statement: str,
) -> dict[str, Any]:
    link = tile_qa.verify_link(root, blender_path)
    link_path = root / tile_qa.LINK_RECEIPT_NAME
    technical_path = root / tile_qa.TECHNICAL_RECEIPT_NAME
    technical = _read_json(technical_path, "technical QA receipt")
    captures = technical.get("captures")
    if not isinstance(captures, Mapping) or set(captures) != {"topdown", "oblique"}:
        raise SimpleMeasuredTileHumanAcceptanceError(
            "Technical QA must expose topdown and oblique captures"
        )
    capture_records: dict[str, Mapping[str, Any]] = {}
    for name in ("topdown", "oblique"):
        record = captures.get(name)
        _validate_artifact(record, root, f"{name} capture")
        capture_records[name] = dict(record)

    evidence = {
        "qa_link": _artifact(link_path, root),
        "technical_qa": _artifact(technical_path, root),
        "captures": capture_records,
        "package_build_id": link["package_build_id"],
        "scene_build_id": link["scene_build_id"],
        "expected_counts": link["expected_counts"],
    }
    payload: dict[str, Any] = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": ACCEPTANCE_STATUS,
        "accepted_human": True,
        "automatic_acceptance": False,
        "review": _review(review_statement),
        "package_build_id": link["package_build_id"],
        "scene_build_id": link["scene_build_id"],
        "expected_counts": link["expected_counts"],
        "qa_link": evidence["qa_link"],
        "technical_qa": evidence["technical_qa"],
        "captures": capture_records,
        "evidence_set_sha256": _canonical_sha256(evidence),
        "acceptor": {
            "contract_sha256": _sha256_file(_contract_path()),
            "algorithm_sha256": _sha256_file(Path(__file__)),
        },
    }
    payload["acceptance_content_sha256"] = _canonical_sha256(payload)
    return payload


def _write_same_or_reject(path: Path, payload: Mapping[str, Any]) -> None:
    content = _canonical_bytes(payload)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise SimpleMeasuredTileHumanAcceptanceError(
                f"Refusing to replace a different human acceptance: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def accept_human_review(
    package_root: Path | str,
    blender_path: Path | str,
    *,
    review_statement: str,
) -> tuple[Path, dict[str, Any]]:
    """Seal an explicit human verdict over the unchanged technical evidence."""

    _load_contract()
    root = _require_d_package(package_root)
    payload = _expected_payload(
        root,
        blender_path,
        review_statement=review_statement,
    )
    path = root / ACCEPTANCE_NAME
    _write_same_or_reject(path, payload)
    return path, payload


def verify_human_acceptance(
    package_root: Path | str,
    blender_path: Path | str,
) -> dict[str, Any]:
    """Replay the package, QA link, captures and explicit human verdict read-only."""

    _load_contract()
    root = _require_d_package(package_root)
    path = root / ACCEPTANCE_NAME
    payload = _read_json(path, "human acceptance receipt")
    if (
        payload.get("schema") != ACCEPTANCE_SCHEMA
        or payload.get("status") != ACCEPTANCE_STATUS
        or payload.get("accepted_human") is not True
        or payload.get("automatic_acceptance") is not False
    ):
        raise SimpleMeasuredTileHumanAcceptanceError(
            "Human acceptance status is invalid"
        )
    declared = payload.get("acceptance_content_sha256")
    without_hash = dict(payload)
    without_hash.pop("acceptance_content_sha256", None)
    if declared != _canonical_sha256(without_hash):
        raise SimpleMeasuredTileHumanAcceptanceError(
            "Human acceptance content hash is invalid"
        )
    review = payload.get("review")
    if not isinstance(review, Mapping) or set(review) != {
        "kind",
        "verdict",
        "statement",
    }:
        raise SimpleMeasuredTileHumanAcceptanceError("Human review record is invalid")
    expected = _expected_payload(
        root,
        blender_path,
        review_statement=review.get("statement"),
    )
    if payload != expected:
        raise SimpleMeasuredTileHumanAcceptanceError(
            "Human acceptance no longer matches its evidence"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("accept", "verify"), required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--review-statement")
    options = parser.parse_args(argv)
    if options.phase == "accept":
        if options.review_statement is None:
            parser.error("accept requires --review-statement")
        path, payload = accept_human_review(
            options.package_root,
            options.blender,
            review_statement=options.review_statement,
        )
    else:
        payload = verify_human_acceptance(options.package_root, options.blender)
        path = Path(options.package_root).resolve() / ACCEPTANCE_NAME
    print(
        json.dumps(
            {
                "schema": ACCEPTANCE_SCHEMA,
                "phase": options.phase,
                "status": payload["status"],
                "accepted_human": payload["accepted_human"],
                "artifact": str(path),
                "package_build_id": payload["package_build_id"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ACCEPTANCE_NAME",
    "ACCEPTANCE_SCHEMA",
    "ACCEPTANCE_STATUS",
    "SimpleMeasuredTileHumanAcceptanceError",
    "accept_human_review",
    "main",
    "verify_human_acceptance",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
