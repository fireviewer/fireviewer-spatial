"""Create and optionally publish a compact map-validation artifact.

The pack is intentionally independent from the large scientific folder and the
browser GLB.  It contains only the files needed to compare two 9-tile builds:
plan, zone receipt, per-tile placement inventories/receipts, compact source
receipts and (when available) the viewer receipt.  A failed large HF upload can
therefore no longer erase the evidence required for the A/B validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "fireviewer.map-validation-pack.v1"
SUMMARY_NAME = "validation-summary.json"
PACK_ROOT = "fireviewer-map-validation"
_ALLOWED_STAGES = {"placement", "viewer"}
_TILE_FILES = (
    "placement/placement-inventory.json",
    "simple-measured-tile-receipt.v1.json",
)
_PROVENANCE_FILES = (
    "elevation-source-05m.json",
    "orthophoto-source.json",
)
_ROOT_FILES = (
    "zone-plan.json",
    "zone.done.json",
)
_VIEWER_RECEIPT = "viewer-scene.v1.json"


class MapValidationPackError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MapValidationPackError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise MapValidationPackError(f"{label} must be a JSON object")
    return value


def _safe_provider(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "-")
    if not cleaned or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-." for character in cleaned):
        raise MapValidationPackError("validation provider label is invalid")
    return cleaned


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _candidate_counts(inventory: Mapping[str, Any], family: str) -> dict[str, int]:
    payload = inventory.get(family)
    if not isinstance(payload, Mapping):
        raise MapValidationPackError(f"placement inventory lacks {family}")
    result: dict[str, int] = {}
    for status in ("valid", "ambiguous", "rejected"):
        value = payload.get(f"{status}_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MapValidationPackError(f"invalid {family} {status} count")
        result[status] = value
    result["source"] = sum(result.values())
    return result


def build_validation_pack(
    job_root: Path | str,
    *,
    provider: str,
    stage: str,
    output: Path | str | None = None,
    require_nine_tiles: bool = False,
) -> tuple[Path, dict[str, Any]]:
    root = Path(job_root).resolve(strict=True)
    provider_name = _safe_provider(provider)
    if stage not in _ALLOWED_STAGES:
        raise MapValidationPackError(f"unsupported validation stage: {stage}")
    plan = _load_json(root / "zone-plan.json", "zone plan")
    receipt = _load_json(root / "zone.done.json", "zone receipt")
    tiles = plan.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise MapValidationPackError("zone plan has no tiles")
    if require_nine_tiles and len(tiles) != 9:
        raise MapValidationPackError(
            f"validation run must contain exactly 9 tiles; got {len(tiles)}"
        )
    if receipt.get("tile_count") != len(tiles):
        raise MapValidationPackError("zone receipt tile count differs from plan")

    selected: list[Path] = []
    for name in _ROOT_FILES:
        path = root / name
        if not path.is_file():
            raise MapValidationPackError(f"missing required validation file: {name}")
        selected.append(path)

    tile_summaries: list[dict[str, Any]] = []
    for raw_tile in tiles:
        if not isinstance(raw_tile, Mapping):
            raise MapValidationPackError("zone plan tile is invalid")
        tile_id = raw_tile.get("tile_id")
        origin = raw_tile.get("origin_l93_m")
        if not isinstance(tile_id, str) or not tile_id or not isinstance(origin, list) or len(origin) != 2:
            raise MapValidationPackError("zone plan tile identity is invalid")
        package_root = root / "packages" / tile_id
        records: dict[str, Any] = {}
        for relative in _TILE_FILES:
            path = package_root / relative
            if not path.is_file():
                raise MapValidationPackError(
                    f"missing validation file for {tile_id}: {relative}"
                )
            selected.append(path)
            records[relative] = _artifact(path, root)
        provenance_records: dict[str, Any] = {}
        provenance_root = root / "provenance" / tile_id
        for name in _PROVENANCE_FILES:
            path = provenance_root / name
            if path.is_file():
                selected.append(path)
                provenance_records[name] = _artifact(path, root)
        inventory = _load_json(
            package_root / "placement" / "placement-inventory.json",
            f"placement inventory {tile_id}",
        )
        tile_summaries.append(
            {
                "tile_id": tile_id,
                "origin_l93_m": origin,
                "placement_profile": inventory.get("placement_profile", "legacy-v1"),
                "placement_algorithm": inventory.get("algorithm"),
                "placement_inventory_sha256": inventory.get("inventory_sha256"),
                "buildings": _candidate_counts(inventory, "buildings"),
                "trees": _candidate_counts(inventory, "trees"),
                "context_assets": _candidate_counts(inventory, "context_assets"),
                "native_tree_refinement_count": inventory.get("trees", {}).get(
                    "native_05m_refinement_count", 0
                )
                if isinstance(inventory.get("trees"), Mapping)
                else 0,
                "files": records,
                "provenance": provenance_records,
            }
        )

    viewer_record: dict[str, Any] | None = None
    viewer_receipt_path = root / _VIEWER_RECEIPT
    if viewer_receipt_path.is_file():
        selected.append(viewer_receipt_path)
        viewer_receipt = _load_json(viewer_receipt_path, "viewer receipt")
        viewer_record = {
            **_artifact(viewer_receipt_path, root),
            "status": viewer_receipt.get("status"),
            "viewer": viewer_receipt.get("viewer"),
            "completeness": viewer_receipt.get("completeness"),
        }
    elif stage == "viewer":
        raise MapValidationPackError("viewer-stage pack requires viewer-scene.v1.json")

    source_revisions = plan.get("source_revisions")
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": stage,
        "provider": provider_name,
        "zone_id": receipt.get("zone_id"),
        "build_id": receipt.get("build_id"),
        "tile_count": len(tiles),
        "production_bounds_l93_m": plan.get("production_bounds_l93_m"),
        "source_revisions": source_revisions if isinstance(source_revisions, Mapping) else {},
        "zone_counts": {
            "buildings": receipt.get("building_count"),
            "trees": receipt.get("tree_count"),
            "context_assets": receipt.get("context_asset_count", 0),
            "degraded_mns_tiles": receipt.get("degraded_mns_tile_count", 0),
        },
        "tiles": tile_summaries,
        "viewer_receipt": viewer_record,
    }
    summary["summary_sha256"] = hashlib.sha256(_canonical_bytes(summary)).hexdigest()

    destination = (
        Path(output)
        if output is not None
        else root
        / "validation"
        / f"fireviewer-validation-{provider_name}-{stage}.zip"
    )
    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.unlink(missing_ok=True)
    selected = sorted(set(selected), key=lambda path: path.relative_to(root).as_posix())
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=False,
        ) as archive:
            summary_info = zipfile.ZipInfo(f"{PACK_ROOT}/{SUMMARY_NAME}")
            summary_info.date_time = (1980, 1, 1, 0, 0, 0)
            summary_info.compress_type = zipfile.ZIP_DEFLATED
            summary_info.external_attr = 0o100644 << 16
            archive.writestr(summary_info, _canonical_bytes(summary) + b"\n")
            for path in selected:
                relative = PurePosixPath(PACK_ROOT) / path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(relative.as_posix())
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise MapValidationPackError("validation pack was not written")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, summary


def publish_validation_pack_hf(
    pack: Path | str,
    summary: Mapping[str, Any],
    *,
    dataset_id: str,
    token: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    path = Path(pack).resolve(strict=True)
    zone_id = summary.get("zone_id")
    build_id = summary.get("build_id")
    provider = summary.get("provider")
    stage = summary.get("stage")
    if not all(isinstance(value, str) and value for value in (zone_id, build_id, provider, stage)):
        raise MapValidationPackError("validation summary identity is incomplete")
    remote_root = f"validation/{zone_id}/{build_id}/{provider}"
    remote_path = f"{remote_root}/{path.name}"
    api = HfApi(token=token)
    info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
    if getattr(info, "private", None) is not False:
        raise MapValidationPackError("validation dataset must be public")
    commit = api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote_path,
        repo_id=dataset_id,
        repo_type="dataset",
        commit_message=f"Publish FireViewer {provider} {stage} validation pack",
    )
    oid = getattr(commit, "oid", None)
    if not isinstance(oid, str) or not oid:
        raise MapValidationPackError("validation pack HF revision is missing")
    if not api.file_exists(
        repo_id=dataset_id,
        filename=remote_path,
        repo_type="dataset",
        revision=oid,
    ):
        raise MapValidationPackError("validation pack is missing after HF commit")
    return {
        "schema": "fireviewer.map-validation-publication.v1",
        "status": "published_public",
        "dataset_id": dataset_id,
        "revision": oid,
        "path": remote_path,
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def create_and_publish_validation_pack(
    job_root: Path | str,
    *,
    provider: str,
    stage: str,
    require_nine_tiles: bool = False,
) -> dict[str, Any]:
    pack, summary = build_validation_pack(
        job_root,
        provider=provider,
        stage=stage,
        require_nine_tiles=require_nine_tiles,
    )
    result: dict[str, Any] = {
        "schema": "fireviewer.map-validation-result.v1",
        "pack": {
            "path": str(pack),
            "sha256": _sha256_file(pack),
            "byte_count": pack.stat().st_size,
        },
        "summary": dict(summary),
        "publication": None,
    }
    dataset_id = os.environ.get("FIREVIEWER_HF_DATASET_ID", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if dataset_id and token:
        result["publication"] = publish_validation_pack_hf(
            pack,
            summary,
            dataset_id=dataset_id,
            token=token,
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--stage", choices=sorted(_ALLOWED_STAGES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-nine-tiles", action="store_true")
    parser.add_argument("--publish-hf", action="store_true")
    options = parser.parse_args(argv)
    pack, summary = build_validation_pack(
        options.job_root,
        provider=options.provider,
        stage=options.stage,
        output=options.output,
        require_nine_tiles=options.require_nine_tiles,
    )
    result: dict[str, Any] = {
        "pack": str(pack),
        "sha256": _sha256_file(pack),
        "byte_count": pack.stat().st_size,
        "summary": summary,
    }
    if options.publish_hf:
        dataset_id = os.environ.get("FIREVIEWER_HF_DATASET_ID", "").strip()
        token = os.environ.get("HF_TOKEN", "").strip()
        if not dataset_id or not token:
            raise MapValidationPackError(
                "FIREVIEWER_HF_DATASET_ID and HF_TOKEN are required for publication"
            )
        result["publication"] = publish_validation_pack_hf(
            pack,
            summary,
            dataset_id=dataset_id,
            token=token,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MapValidationPackError",
    "build_validation_pack",
    "create_and_publish_validation_pack",
    "publish_validation_pack_hf",
]
