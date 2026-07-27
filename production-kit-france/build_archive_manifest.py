"""Create or verify a deterministic SHA-256 inventory for a map archive tree."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA = "fireviewer.map-reproduction-archive-manifest.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_manifest(
    root: Path,
    output: Path,
    *,
    archive_id: str,
    description: str,
) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise ValueError(f"archive root is absent: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.resolve() == output:
            continue
        relative = path.relative_to(root).as_posix()
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"unsafe archive path: {relative}")
        files.append(
            {
                "path": relative,
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("archive tree contains no file")
    manifest = {
        "schema": SCHEMA,
        "archive_id": archive_id,
        "description": description,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "inventory": {
            "file_count": len(files),
            "byte_count": sum(record["byte_count"] for record in files),
            "sha256_algorithm": "SHA-256",
        },
        "files": files,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(json_bytes(manifest))
    os.replace(temporary, output)
    return manifest


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported archive manifest")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("archive manifest contains no file")
    expected: set[str] = set()
    for record in records:
        relative = str(record.get("path", ""))
        parsed = PurePosixPath(relative)
        if not relative or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"unsafe archive path: {relative!r}")
        if relative in expected:
            raise ValueError(f"duplicate archive path: {relative}")
        expected.add(relative)
        path = root.joinpath(*parsed.parts)
        if not path.is_file():
            raise FileNotFoundError(f"archived file is absent: {relative}")
        if path.stat().st_size != int(record.get("byte_count", -1)):
            raise ValueError(f"archived file size differs: {relative}")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"archived file checksum differs: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path
    }
    if actual != expected:
        raise ValueError(
            f"archive tree inventory differs: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    inventory = manifest.get("inventory", {})
    if int(inventory.get("file_count", -1)) != len(records):
        raise ValueError("archive file count is inconsistent")
    if int(inventory.get("byte_count", -1)) != sum(
        int(record["byte_count"]) for record in records
    ):
        raise ValueError("archive byte count is inconsistent")
    return {
        "status": "valid",
        "archive_id": manifest.get("archive_id"),
        "file_count": len(records),
        "byte_count": int(inventory["byte_count"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--archive-id")
    parser.add_argument("--description", default="FireViewer map reproduction archive")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.create:
        if not args.archive_id:
            raise ValueError("--archive-id is required with --create")
        result = build_manifest(
            args.root,
            args.manifest,
            archive_id=args.archive_id,
            description=args.description,
        )
        output = {
            "status": "created",
            "archive_id": result["archive_id"],
            **result["inventory"],
        }
    else:
        output = verify_manifest(args.root, args.manifest)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
