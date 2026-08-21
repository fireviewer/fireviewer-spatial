"""Build and publish folder-native evidence for one FireViewer validation map.

Validation evidence is deliberately not a map package and is never archived.
It is a small directory containing the plan, receipts, placement inventories and
the tiled-viewer catalogue receipt needed to compare production runs.  An
optional monolithic oracle receipt may also be included for small maps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "fireviewer.map-validation-folder.v2"
PUBLICATION_SCHEMA = "fireviewer.map-validation-publication.v2"
SUMMARY_NAME = "validation-summary.json"
_ALLOWED_STAGES = {"placement", "viewer"}
_TILE_FILES = (
    ("placement/placement-inventory.json", "placement-inventory.json"),
    ("simple-measured-tile-receipt.v1.json", "tile-receipt.json"),
)
_PROVENANCE_FILES = (
    "elevation-source-05m.json",
    "orthophoto-source.json",
)
_VIEWER_RECEIPT = "viewer-scene.v1.json"
_TILED_VIEWER_ROOT = "viewer-tiled"
_TILED_VIEWER_RECEIPT = "viewer-tiled-scene.v1.json"
_TILED_VIEWER_CATALOG = "catalog.json"


class MapValidationFolderError(RuntimeError):
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MapValidationFolderError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise MapValidationFolderError(f"{label} must be a JSON object")
    return value


def _safe_provider(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "-")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-."
    if not cleaned or any(character not in allowed for character in cleaned):
        raise MapValidationFolderError("validation provider label is invalid")
    return cleaned


def _candidate_counts(inventory: Mapping[str, Any], family: str) -> dict[str, int]:
    payload = inventory.get(family)
    if not isinstance(payload, Mapping):
        raise MapValidationFolderError(f"placement inventory lacks {family}")
    result: dict[str, int] = {}
    for status in ("valid", "ambiguous", "rejected"):
        value = payload.get(f"{status}_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MapValidationFolderError(f"invalid {family} {status} count")
        result[status] = value
    result["source"] = sum(result.values())
    return result


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise MapValidationFolderError(f"missing validation file: {source}")
    if source.suffix.casefold() == ".zip":
        raise MapValidationFolderError("ZIP validation artifacts are forbidden")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _viewer_summary(
    receipt: Mapping[str, Any],
    zone_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    viewer = receipt.get("viewer")
    completeness = receipt.get("completeness")
    if not isinstance(viewer, Mapping) or not isinstance(completeness, Mapping):
        raise MapValidationFolderError("viewer receipt is incomplete")
    if (
        completeness.get("policy") != "fail_closed_exact_visual_scene"
        or completeness.get("mesh_coverage") != "complete"
    ):
        raise MapValidationFolderError(
            "viewer must be the complete non-simplified visual map"
        )
    counts = completeness.get("family_instance_counts")
    expected = {
        "buildings": zone_receipt.get("building_count"),
        "trees": zone_receipt.get("tree_count"),
        "context_assets": zone_receipt.get("context_asset_count", 0),
    }
    if not isinstance(counts, Mapping) or any(counts.get(key) != value for key, value in expected.items()):
        raise MapValidationFolderError(
            "viewer instance counts differ from the canonical map counts"
        )
    return {
        "representation": "complete_non_simplified_map",
        "status": receipt.get("status"),
        "viewer": dict(viewer),
        "completeness": dict(completeness),
    }


def build_validation_folder(
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
        raise MapValidationFolderError(f"unsupported validation stage: {stage}")

    plan = _load_json(root / "zone-plan.json", "zone plan")
    zone_receipt = _load_json(root / "zone.done.json", "zone receipt")
    tiles = plan.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise MapValidationFolderError("zone plan has no tiles")
    if require_nine_tiles and len(tiles) != 9:
        raise MapValidationFolderError(
            f"validation run must contain exactly 9 tiles; got {len(tiles)}"
        )
    if zone_receipt.get("tile_count") != len(tiles):
        raise MapValidationFolderError("zone receipt tile count differs from plan")
    placeholder_count = zone_receipt.get("placeholder_instance_count", 0)
    if placeholder_count != 0:
        raise MapValidationFolderError(
            f"real-assets-only contract violated: {placeholder_count} placeholder instances"
        )

    destination = (
        Path(output).resolve(strict=False)
        if output is not None
        else (root / "validation" / provider_name).resolve(strict=False)
    )
    staging = destination.with_name(f".{destination.name}.{stage}.part")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        _copy_file(root / "zone-plan.json", staging / "zone-plan.json")
        _copy_file(root / "zone.done.json", staging / "zone.done.json")

        tile_summaries: list[dict[str, Any]] = []
        for raw_tile in tiles:
            if not isinstance(raw_tile, Mapping):
                raise MapValidationFolderError("zone plan tile is invalid")
            tile_id = raw_tile.get("tile_id")
            origin = raw_tile.get("origin_l93_m")
            if (
                not isinstance(tile_id, str)
                or not tile_id
                or not isinstance(origin, list)
                or len(origin) != 2
            ):
                raise MapValidationFolderError("zone plan tile identity is invalid")

            package_root = root / "packages" / tile_id
            tile_root = staging / "tiles" / tile_id
            for source_relative, output_name in _TILE_FILES:
                _copy_file(package_root / source_relative, tile_root / output_name)

            provenance_root = root / "provenance" / tile_id
            provenance_names: list[str] = []
            for name in _PROVENANCE_FILES:
                source = provenance_root / name
                if source.is_file():
                    _copy_file(source, tile_root / "provenance" / name)
                    provenance_names.append(name)

            inventory = _load_json(
                package_root / "placement" / "placement-inventory.json",
                f"placement inventory {tile_id}",
            )
            tile_summaries.append(
                {
                    "tile_id": tile_id,
                    "origin_l93_m": origin,
                    "placement_profile": inventory.get(
                        "placement_profile", "legacy-v1"
                    ),
                    "placement_algorithm": inventory.get("algorithm"),
                    "placement_inventory_sha256": inventory.get(
                        "inventory_sha256"
                    ),
                    "buildings": _candidate_counts(inventory, "buildings"),
                    "trees": _candidate_counts(inventory, "trees"),
                    "context_assets": _candidate_counts(
                        inventory, "context_assets"
                    ),
                    "native_tree_refinement_count": (
                        inventory.get("trees", {}).get(
                            "native_05m_refinement_count", 0
                        )
                        if isinstance(inventory.get("trees"), Mapping)
                        else 0
                    ),
                    "provenance_files": provenance_names,
                }
            )

        viewer_record: dict[str, Any] | None = None
        monolithic_viewer_record: dict[str, Any] | None = None
        tiled_receipt_path = (
            root / _TILED_VIEWER_ROOT / _TILED_VIEWER_RECEIPT
        )
        tiled_catalog_path = root / _TILED_VIEWER_ROOT / _TILED_VIEWER_CATALOG
        if tiled_receipt_path.is_file() and tiled_catalog_path.is_file():
            from build_tiled_viewer_package import validate_tiled_viewer_package

            try:
                tiled_receipt, tiled_viewer = validate_tiled_viewer_package(root)
            except Exception as error:
                raise MapValidationFolderError(
                    "tiled viewer package is invalid"
                ) from error
            viewer_record = {
                "representation": "complete_tiled_non_simplified_map",
                "status": tiled_receipt.get("status"),
                "viewer": tiled_viewer,
                "source": tiled_receipt.get("source"),
            }
            _copy_file(
                tiled_receipt_path,
                staging / _TILED_VIEWER_ROOT / _TILED_VIEWER_RECEIPT,
            )
            _copy_file(
                tiled_catalog_path,
                staging / _TILED_VIEWER_ROOT / _TILED_VIEWER_CATALOG,
            )
        elif stage == "viewer":
            raise MapValidationFolderError(
                "viewer-stage evidence requires a valid tiled viewer package"
            )

        viewer_receipt_path = root / _VIEWER_RECEIPT
        if viewer_receipt_path.is_file():
            viewer_receipt = _load_json(viewer_receipt_path, "viewer receipt")
            monolithic_viewer_record = _viewer_summary(
                viewer_receipt, zone_receipt
            )
            _copy_file(viewer_receipt_path, staging / _VIEWER_RECEIPT)

        source_revisions = plan.get("source_revisions")
        summary: dict[str, Any] = {
            "schema": SCHEMA,
            "artifact_role": "validation_evidence_only_not_a_map",
            "stage": stage,
            "provider": provider_name,
            "zone_id": zone_receipt.get("zone_id"),
            "build_id": zone_receipt.get("build_id"),
            "tile_count": len(tiles),
            "production_bounds_l93_m": plan.get("production_bounds_l93_m"),
            "source_revisions": (
                dict(source_revisions)
                if isinstance(source_revisions, Mapping)
                else {}
            ),
            "zone_counts": {
                "buildings": zone_receipt.get("building_count"),
                "trees": zone_receipt.get("tree_count"),
                "context_assets": zone_receipt.get("context_asset_count", 0),
                "placeholder_instances": placeholder_count,
                "degraded_mns_tiles": zone_receipt.get(
                    "degraded_mns_tile_count", 0
                ),
            },
            "viewer_role": "complete_non_simplified_map_representation",
            "tiles": tile_summaries,
            "viewer_receipt": viewer_record,
            "monolithic_viewer_oracle_receipt": monolithic_viewer_record,
        }
        summary["summary_sha256"] = hashlib.sha256(
            _canonical_bytes(summary)
        ).hexdigest()
        _write_json(staging / SUMMARY_NAME, summary)

        if any(path.suffix.casefold() == ".zip" for path in staging.rglob("*")):
            raise MapValidationFolderError(
                "validation evidence must remain folder-native; ZIP found"
            )
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise MapValidationFolderError(
                    "validation destination is not a regular directory"
                )
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        return destination, summary
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _relative_files(folder: Path) -> tuple[str, ...]:
    files = tuple(
        path.relative_to(folder).as_posix()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    )
    if not files:
        raise MapValidationFolderError("validation evidence folder is empty")
    if any(name.casefold().endswith(".zip") for name in files):
        raise MapValidationFolderError("ZIP validation artifacts are forbidden")
    return files


def publish_validation_folder_hf(
    folder: Path | str,
    summary: Mapping[str, Any],
    *,
    dataset_id: str,
    token: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    local_root = Path(folder).resolve(strict=True)
    zone_id = summary.get("zone_id")
    build_id = summary.get("build_id")
    provider = summary.get("provider")
    stage = summary.get("stage")
    if not all(
        isinstance(value, str) and value
        for value in (zone_id, build_id, provider, stage)
    ):
        raise MapValidationFolderError("validation summary identity is incomplete")

    files = _relative_files(local_root)
    remote_root = f"validation/{zone_id}/{build_id}/{provider}"
    api = HfApi(token=token)
    info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
    if getattr(info, "private", None) is not False:
        raise MapValidationFolderError("validation dataset must be public")
    commit = api.upload_folder(
        repo_id=dataset_id,
        repo_type="dataset",
        folder_path=str(local_root),
        path_in_repo=remote_root,
        allow_patterns=list(files),
        commit_message=f"Publish FireViewer {provider} {stage} validation evidence",
    )
    oid = getattr(commit, "oid", None)
    if not isinstance(oid, str) or not oid:
        raise MapValidationFolderError("validation publication revision is missing")

    missing = [
        name
        for name in files
        if not api.file_exists(
            repo_id=dataset_id,
            filename=f"{remote_root}/{name}",
            repo_type="dataset",
            revision=oid,
        )
    ]
    if missing:
        raise MapValidationFolderError(
            "validation files missing after HF commit: " + ", ".join(missing)
        )
    return {
        "schema": PUBLICATION_SCHEMA,
        "status": "published_public_folder",
        "dataset_id": dataset_id,
        "revision": oid,
        "root": remote_root,
        "file_count": len(files),
        "summary_path": f"{remote_root}/{SUMMARY_NAME}",
    }


def create_and_publish_validation_folder(
    job_root: Path | str,
    *,
    provider: str,
    stage: str,
    require_nine_tiles: bool = False,
) -> dict[str, Any]:
    folder, summary = build_validation_folder(
        job_root,
        provider=provider,
        stage=stage,
        require_nine_tiles=require_nine_tiles,
    )
    result: dict[str, Any] = {
        "schema": "fireviewer.map-validation-result.v2",
        "folder": {
            "path": str(folder),
            "file_count": len(_relative_files(folder)),
        },
        "summary": dict(summary),
        "publication": None,
    }
    dataset_id = os.environ.get("FIREVIEWER_HF_DATASET_ID", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if dataset_id and token:
        result["publication"] = publish_validation_folder_hf(
            folder,
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-nine-tiles", action="store_true")
    parser.add_argument("--publish-hf", action="store_true")
    options = parser.parse_args(argv)
    folder, summary = build_validation_folder(
        options.job_root,
        provider=options.provider,
        stage=options.stage,
        output=options.output_dir,
        require_nine_tiles=options.require_nine_tiles,
    )
    result: dict[str, Any] = {
        "folder": str(folder),
        "file_count": len(_relative_files(folder)),
        "summary": summary,
    }
    if options.publish_hf:
        dataset_id = os.environ.get("FIREVIEWER_HF_DATASET_ID", "").strip()
        token = os.environ.get("HF_TOKEN", "").strip()
        if not dataset_id or not token:
            raise MapValidationFolderError(
                "FIREVIEWER_HF_DATASET_ID and HF_TOKEN are required for publication"
            )
        result["publication"] = publish_validation_folder_hf(
            folder,
            summary,
            dataset_id=dataset_id,
            token=token,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MapValidationFolderError",
    "build_validation_folder",
    "create_and_publish_validation_folder",
    "publish_validation_folder_hf",
]
