"""Validate and authorize an audited Hugging Face simulation staging tree.

The finalizer removes workstation-local path provenance from public metadata,
validates the five-zoom point contract, writes a deterministic file inventory,
and records the explicit publication boundary. It does not contact or mutate
the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


WINDOWS_PATH_RE = re.compile(r"(?i)(?:^|[\"'\s])(?:[a-z]:[\\/])")
TEXT_SUFFIXES = {".json", ".md", ".txt", ".csv", ".usda"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(canonical_json(value), encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sanitize_capture_receipt(path: Path) -> None:
    receipt = read_json(path)
    source_frame = receipt.pop("source_frame", None)
    if source_frame is not None and not receipt.get("source_frame_relative"):
        source_parts = Path(str(source_frame)).parts
        try:
            start = source_parts.index("simulationDS1_diev1")
        except ValueError:
            receipt["source_frame_relative"] = str(receipt["capture_id"])
        else:
            receipt["source_frame_relative"] = Path(*source_parts[start:]).as_posix()
    write_json(path, receipt)


def _source_audit_locks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    batches = manifest.get("source_batches")
    if not isinstance(batches, list) or not batches:
        raise RuntimeError("staging manifest has no source batches")
    locks: list[dict[str, Any]] = []
    for batch in batches:
        audit_value = batch.get("audit_report") or batch.get("audit_file")
        capture_value = batch.get("capture_root") or batch.get("batch_id")
        if not audit_value or not capture_value:
            raise RuntimeError("source batch identity is incomplete")
        audit_path = Path(str(audit_value))
        if not audit_path.is_absolute():
            raise RuntimeError(
                "source audit paths must remain locally resolvable until publication finalization"
            )
        audit = read_json(audit_path)
        digest = sha256_file(audit_path)
        if (
            audit.get("schema") != "fireviewer.capture-metadata-audit.v2"
            or audit.get("status") != "passed"
            or int(audit.get("failed_capture_count", -1)) != 0
            or int(audit.get("abstention_warning_count", -1)) != 0
            or digest != batch.get("audit_sha256")
        ):
            raise RuntimeError(f"source audit is no longer publication-clean: {audit_path}")
        locks.append(
            {
                "batch_id": Path(str(capture_value)).name,
                "audit_file": audit_path.name,
                "audit_sha256": digest,
                "captures": int(batch["captures"]),
                "points": int(batch["points"]),
            }
        )
    return locks


def _validate_points(kind_root: Path) -> tuple[int, int]:
    point_manifests = sorted(kind_root.rglob("point-manifest.json"))
    capture_receipts = sorted(kind_root.rglob("capture-package.json"))
    for point_path in point_manifests:
        point = read_json(point_path)
        captures = point.get("captures")
        if not isinstance(captures, list) or len(captures) != 5:
            raise RuntimeError(f"point does not contain five captures: {point_path}")
        zooms = {int(item["zoom_index"]) for item in captures}
        originals = [item for item in captures if bool(item["is_original_framing"])]
        if zooms != {1, 2, 3, 4, 5} or len(originals) != 1 or originals[0]["zoom_index"] != 2:
            raise RuntimeError(f"point zoom contract failed: {point_path}")
    for receipt_path in capture_receipts:
        _sanitize_capture_receipt(receipt_path)
        receipt = read_json(receipt_path)
        files = receipt.get("source_file_sha256")
        if not isinstance(files, dict) or not files:
            raise RuntimeError(f"capture has no source-file hashes: {receipt_path}")
        missing = [name for name in files if not (receipt_path.parent / name).is_file()]
        if missing:
            raise RuntimeError(f"capture payload is incomplete: {receipt_path}: {missing[0]}")
    return len(point_manifests), len(capture_receipts)


def _write_inventory(staging_root: Path) -> tuple[Path, dict[str, Any]]:
    excluded = {"publication-inventory.json", "publication-receipt.json"}
    records = []
    for path in sorted(item for item in staging_root.rglob("*") if item.is_file()):
        relative = path.relative_to(staging_root).as_posix()
        if relative in excluded:
            continue
        records.append({"path": relative, "byte_count": path.stat().st_size})
    records_sha256 = hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()
    payload = {
        "schema": "fireviewer.hf-publication-file-inventory.v1",
        "root": ".",
        "file_count": len(records),
        "total_byte_count": sum(int(item["byte_count"]) for item in records),
        "records_sha256": records_sha256,
        "files": records,
    }
    path = staging_root / "publication-inventory.json"
    write_json(path, payload)
    return path, payload


def _absolute_path_files(staging_root: Path) -> list[str]:
    issues = []
    for path in sorted(item for item in staging_root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if WINDOWS_PATH_RE.search(text):
            issues.append(path.relative_to(staging_root).as_posix())
    return issues


def finalize(
    *,
    staging_root: Path,
    kind: str,
    repository_id: str,
    expected_captures: int,
    expected_points: int,
    authorized_at: str | None = None,
) -> dict[str, Any]:
    staging_root = staging_root.resolve()
    kind_root = staging_root / "massif-of-justin" / kind
    manifest_path = kind_root / "dataset-manifest.json"
    manifest = read_json(manifest_path)
    source_locks = _source_audit_locks(manifest)
    points, captures = _validate_points(kind_root)
    if points != expected_points or captures != expected_captures:
        raise RuntimeError(
            f"staging counts differ from the authorized scope: {points}/{captures}"
        )
    if int(manifest.get("points", -1)) != points or int(manifest.get("captures", -1)) != captures:
        raise RuntimeError("staging manifest counts do not match the assembled tree")

    manifest.update(
        {
            "publication_authorized": True,
            "publication_authorized_at": authorized_at
            or datetime.now(timezone.utc).isoformat(),
            "repository_id": repository_id,
            "publication_scope": "passed_simulation_states_001_through_022_only",
            "source_capture_root": None,
            "source_audit_report": None,
            "source_audit_sha256": None,
            "source_batches": source_locks,
            "publication_receipt": "../../publication-receipt.json",
        }
    )
    write_json(manifest_path, manifest)
    absolute_issues = _absolute_path_files(staging_root)
    if absolute_issues:
        raise RuntimeError(f"public staging contains a Windows path: {absolute_issues[0]}")

    inventory_path, inventory = _write_inventory(staging_root)
    receipt = {
        "schema": "fireviewer.hf-dataset-publication-authorization.v1",
        "authorized": True,
        "authorized_at": manifest["publication_authorized_at"],
        "authorization_basis": "explicit_user_instruction_in_active_task",
        "repository_id": repository_id,
        "repository_type": "dataset",
        "visibility": "public",
        "included": {
            "kind": kind,
            "dataset_id": "simulationDS1_diev1",
            "state_range": [1, 22],
            "captures": captures,
            "points": points,
            "source_audit_locks": source_locks,
        },
        "excluded": [
            "state_023_interrupted_before_first_capture",
            "historical_reproduction_captures",
            "omniverse_download_packages",
            "training_admission",
        ],
        "inventory": {
            "path": inventory_path.name,
            "sha256": sha256_file(inventory_path),
            "file_count_excluding_inventory_and_receipt": inventory["file_count"],
            "total_byte_count_excluding_inventory_and_receipt": inventory[
                "total_byte_count"
            ],
            "records_sha256": inventory["records_sha256"],
        },
        "local_cleanup_authorized_after_remote_verification": True,
    }
    write_json(staging_root / "publication-receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("sim", "repro"), required=True)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--expected-captures", type=int, required=True)
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("--authorized-at")
    args = parser.parse_args()
    receipt = finalize(
        staging_root=args.staging_root,
        kind=args.kind,
        repository_id=args.repository_id,
        expected_captures=args.expected_captures,
        expected_points=args.expected_points,
        authorized_at=args.authorized_at,
    )
    print(canonical_json(receipt), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
