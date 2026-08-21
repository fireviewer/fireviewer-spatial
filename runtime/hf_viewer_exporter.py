#!/usr/bin/env python3
"""Publish one validated tiled viewer to the canonical Hugging Face dataset.

The exporter deliberately knows nothing about AWS.  Its execution adapter
provides a local input directory, a dataset destination and an output receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from build_tiled_viewer_package import validate_tiled_viewer_package

PUBLICATION_SCHEMA = "fireviewer.hf-viewer-publication.v1"
ZONE_RECEIPT_SCHEMA = "fireviewer.simple-measured-zone-production.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_ZONE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")
_JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_DATASET_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)


class HfViewerExportError(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HfViewerExportError(f"invalid {label}") from error
    if not isinstance(payload, dict):
        raise HfViewerExportError(f"invalid {label}")
    return dict(payload)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _safe_remote_root(value: str, *, zone_id: str, build_id: str) -> str:
    expected = f"maps/{zone_id}/{build_id}/runtime"
    path = PurePosixPath(value)
    if (
        value != expected
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise HfViewerExportError("invalid remote dataset root")
    return value


def _verify_remote_files(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    paths: tuple[str, ...],
) -> None:
    missing: list[str] = []
    for path in paths:
        available = False
        for attempt in range(5):
            if api.file_exists(
                repo_id=repo_id,
                filename=path,
                repo_type="dataset",
                revision=revision,
            ):
                available = True
                break
            if attempt < 4:
                time.sleep(2.0)
        if not available:
            missing.append(path)
    if missing:
        raise HfViewerExportError(
            "files missing after Hugging Face commit: " + ", ".join(missing)
        )


def publish_viewer(
    input_root: Path | str,
    *,
    repo_id: str,
    remote_root: str,
    build_id: str,
    job_id: str,
    exporter_image_digest: str,
    output_receipt: Path | str,
    token: str | None = None,
) -> dict[str, Any]:
    root = Path(input_root).resolve(strict=True)
    if _DATASET_RE.fullmatch(repo_id) is None:
        raise HfViewerExportError("invalid Hugging Face dataset id")
    if _JOB_RE.fullmatch(job_id) is None:
        raise HfViewerExportError("invalid map job id")
    if (
        not exporter_image_digest.startswith("sha256:")
        or _SHA256_RE.fullmatch(exporter_image_digest.removeprefix("sha256:")) is None
    ):
        raise HfViewerExportError("invalid exporter image digest")

    zone = _load_json(root / "zone.done.json", "zone receipt")
    zone_id = zone.get("zone_id")
    scientific_build_id = zone.get("build_id")
    tile_count = zone.get("tile_count")
    if (
        zone.get("schema") != ZONE_RECEIPT_SCHEMA
        or zone.get("status") != "technical_scene_produced"
        or not isinstance(zone_id, str)
        or _ZONE_RE.fullmatch(zone_id) is None
        or not isinstance(scientific_build_id, str)
        or _SHA256_RE.fullmatch(scientific_build_id) is None
        or isinstance(tile_count, bool)
        or not isinstance(tile_count, int)
        or tile_count <= 0
        or zone.get("placeholder_instance_count") != 0
    ):
        raise HfViewerExportError("invalid sealed zone receipt")
    if _SHA256_RE.fullmatch(build_id) is None:
        raise HfViewerExportError("invalid Map Job build id")
    runtime_root = _safe_remote_root(
        remote_root,
        zone_id=zone_id,
        build_id=build_id,
    )

    try:
        _receipt, viewer = validate_tiled_viewer_package(
            root,
            require_sealed_source_assets=False,
        )
    except Exception as error:
        raise HfViewerExportError("invalid tiled viewer package") from error

    hf_token = (token or os.environ.get("HF_TOKEN") or "").strip()
    if len(hf_token) < 16:
        raise HfViewerExportError("HF_TOKEN is missing")
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if getattr(info, "private", None) is not False:
        raise HfViewerExportError("target Hugging Face dataset must be public")

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(root / "viewer-tiled"),
        path_in_repo=f"{runtime_root}/viewer-tiled",
        commit_message=f"Publish FireViewer runtime {zone_id} ({job_id})",
    )
    revision = getattr(commit, "oid", None)
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise HfViewerExportError("Hugging Face commit revision is missing")

    bootstrap = viewer.get("bootstrap_asset")
    completeness = viewer.get("completeness")
    if not isinstance(bootstrap, Mapping) or not isinstance(completeness, Mapping):
        raise HfViewerExportError("invalid tiled viewer descriptor")
    catalog_path = f"{runtime_root}/{viewer['catalog_path']}"
    viewer_receipt_path = f"{runtime_root}/{viewer['receipt_path']}"
    bootstrap_path = f"{runtime_root}/{bootstrap['path']}"
    _verify_remote_files(
        api,
        repo_id=repo_id,
        revision=revision,
        paths=(catalog_path, viewer_receipt_path, bootstrap_path),
    )

    publication = {
        "schema": PUBLICATION_SCHEMA,
        "status": "published",
        "job_id": job_id,
        "zone_id": zone_id,
        "build_id": build_id,
        "scientific_build_id": scientific_build_id,
        "tile_count": tile_count,
        "degraded_mns_tile_count": zone.get("degraded_mns_tile_count", 0),
        "dataset": {
            "repo_id": repo_id,
            "revision": revision,
            "root": runtime_root,
            "visibility": "public",
        },
        "viewer": {
            "catalog_path": catalog_path,
            "receipt_path": viewer_receipt_path,
            "catalog_sha256": viewer["catalog_sha256"],
            "catalog_byte_count": viewer["catalog_byte_count"],
            "payload_file_count": viewer["payload_file_count"],
            "payload_byte_count": viewer["payload_byte_count"],
            "bootstrap_asset": {
                **dict(bootstrap),
                "path": bootstrap_path,
            },
            "representation": "complete_tiled_non_simplified_map",
            "completeness": dict(completeness),
        },
        "exporter": {
            "image_digest": exporter_image_digest,
            "batch_job_id": os.environ.get("AWS_BATCH_JOB_ID"),
        },
        "published_at": datetime.now(UTC).isoformat(),
    }
    encoded = _canonical_bytes(publication) + b"\n"
    destination = Path(output_receipt).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    return publication


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output-receipt", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = publish_viewer(
        args.input,
        repo_id=args.repo_id,
        remote_root=args.remote_root,
        build_id=args.build_id,
        job_id=args.job_id,
        exporter_image_digest=args.image_digest,
        output_receipt=args.output_receipt,
    )
    print(
        json.dumps(
            {
                "published": True,
                "job_id": receipt["job_id"],
                "revision": receipt["dataset"]["revision"],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
