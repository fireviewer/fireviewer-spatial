"""Verify a published FireViewer Hugging Face dataset against its local inventory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


CAPTURE_PATH_RE = re.compile(
    r"^massif-of-justin/sim/raw_files/day\d{2}/case\d{2}/point\d{2}/"
    r"(?:original|zoom01_0p75x|zoom03_1p25x|zoom04_1p50x|zoom05_2p00x)/"
    r"capture-package\.json$"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify(
    *,
    repository_id: str,
    staging_root: Path,
    output: Path,
) -> dict[str, Any]:
    staging_root = staging_root.resolve()
    inventory_path = staging_root / "publication-inventory.json"
    receipt_path = staging_root / "publication-receipt.json"
    inventory = read_json(inventory_path)
    local_files = {
        str(record["path"]): int(record["byte_count"])
        for record in inventory["files"]
    }
    local_files[inventory_path.name] = inventory_path.stat().st_size
    local_files[receipt_path.name] = receipt_path.stat().st_size

    api = HfApi()
    info = api.dataset_info(repository_id)
    if info.private:
        raise RuntimeError("dataset repository is not public")
    revision = str(info.sha)
    remote_files: dict[str, int] = {}
    for item in api.list_repo_tree(
        repository_id,
        recursive=True,
        expand=False,
        revision=revision,
        repo_type="dataset",
    ):
        if isinstance(item, RepoFile):
            remote_files[str(item.path)] = int(item.size)

    missing = sorted(set(local_files) - set(remote_files))
    unexpected = sorted(set(remote_files) - set(local_files) - {".gitattributes"})
    size_mismatches = sorted(
        {
            path: {"local": size, "remote": remote_files.get(path)}
            for path, size in local_files.items()
            if remote_files.get(path) != size
        }.items()
    )
    capture_receipts = sorted(path for path in remote_files if CAPTURE_PATH_RE.fullmatch(path))
    verification_cache = output.resolve().parent / ".hf-verification-cache"

    downloaded_inventory = Path(
        hf_hub_download(
            repository_id,
            inventory_path.name,
            repo_type="dataset",
            revision=revision,
            cache_dir=verification_cache,
        )
    )
    downloaded_receipt = Path(
        hf_hub_download(
            repository_id,
            receipt_path.name,
            repo_type="dataset",
            revision=revision,
            cache_dir=verification_cache,
        )
    )
    remote_receipt = read_json(downloaded_receipt)
    remote_inventory_sha = sha256_file(downloaded_inventory)
    remote_receipt_sha = sha256_file(downloaded_receipt)
    expected_inventory_sha = sha256_file(inventory_path)
    expected_receipt_sha = sha256_file(receipt_path)

    failures = []
    if missing:
        failures.append(f"missing remote file: {missing[0]}")
    if unexpected:
        failures.append(f"unexpected remote file: {unexpected[0]}")
    if size_mismatches:
        failures.append(f"remote size mismatch: {size_mismatches[0][0]}")
    if len(capture_receipts) != 2200:
        failures.append(f"capture receipt count is {len(capture_receipts)}, expected 2200")
    if remote_inventory_sha != expected_inventory_sha:
        failures.append("remote publication inventory hash differs from local")
    if remote_receipt_sha != expected_receipt_sha:
        failures.append("remote publication receipt hash differs from local")
    if remote_receipt.get("local_cleanup_authorized_after_remote_verification") is not False:
        failures.append("remote receipt does not preserve local outputs")

    report = {
        "schema": "fireviewer.hf-publication-verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "repository_id": repository_id,
        "repository_url": f"https://huggingface.co/datasets/{repository_id}",
        "revision": revision,
        "public": not bool(info.private),
        "expected_local_file_count": len(local_files),
        "remote_file_count": len(remote_files),
        "remote_extra_repository_files": sorted(set(remote_files) - set(local_files)),
        "expected_total_byte_count": sum(local_files.values()),
        "remote_dataset_payload_byte_count": sum(
            remote_files[path] for path in local_files if path in remote_files
        ),
        "capture_receipt_count": len(capture_receipts),
        "missing_file_count": len(missing),
        "unexpected_file_count": len(unexpected),
        "size_mismatch_count": len(size_mismatches),
        "publication_inventory_sha256": expected_inventory_sha,
        "publication_receipt_sha256": expected_receipt_sha,
        "local_outputs_retained": True,
        "failures": failures,
    }
    write_json(output.resolve(), report)
    if failures:
        raise RuntimeError(failures[0])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(
        repository_id=args.repository_id,
        staging_root=args.staging_root,
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
