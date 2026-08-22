"""Provider-neutral one-shot FireViewer 9-tile validation job.

This entrypoint is separate from the stable Lightning and legacy RunPod images.
It runs either the current placement profile or factual v2, publishes compact
validation evidence as ordinary Hugging Face folders, exports the complete
non-simplified viewer map, and publishes that viewer under the canonical
``maps/<zone>/<build>/runtime`` path.

No validation archive and no final zone archive are created.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import simple_production_engine as production_engine
from fixed_asset_placement import (
    EMPTY_REQUEST,
    normalize_request as normalize_fixed_assets,
)
from build_tiled_viewer_package import (
    OUTPUT_DIRECTORY as TILED_OUTPUT_DIRECTORY,
    build_tiled_viewer_package,
    validate_tiled_viewer_package,
)
from map_validation_folder import create_and_publish_validation_folder
from portable_scene_package import validate_map_upload_package
from runpod_map_production import normalize_map_request, request_sha256
from simple_production_engine import (
    ProductionConfig,
    ProductionEngine,
    TileCheckpointShardReady,
    plan_zone,
)

SCHEMA = "fireviewer.map-validation-job.v2"
RESULT_NAME = "validation-result.json"
VIEWER_RECEIPT_NAME = "viewer-scene.v1.json"
VIEWER_GLB_NAME = "viewer.glb"
PROFILES = {"legacy-v1", "factual-v2"}


class MapValidationJobError(RuntimeError):
    pass


class _SealedFolderReady(RuntimeError):
    def __init__(self, root: Path) -> None:
        super().__init__(str(root))
        self.root = root


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() not in {"0", "false", "no", "off"}


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MapValidationJobError(f"missing required environment variable: {name}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_request() -> dict[str, Any]:
    raw = os.environ.get("FIREVIEWER_VALIDATION_REQUEST_JSON", "").strip()
    path = os.environ.get("FIREVIEWER_VALIDATION_REQUEST_FILE", "").strip()
    if bool(raw) == bool(path):
        raise MapValidationJobError(
            "set exactly one of FIREVIEWER_VALIDATION_REQUEST_JSON or "
            "FIREVIEWER_VALIDATION_REQUEST_FILE"
        )
    try:
        payload = (
            json.loads(raw)
            if raw
            else json.loads(Path(path).read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as error:
        raise MapValidationJobError("validation request JSON is invalid") from error
    return normalize_map_request(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _skip_archive_budget(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "schema": "fireviewer.map-folder-publication.v1",
        "status": "archive_disabled",
    }


@contextmanager
def _sealed_folder_mode() -> Iterator[None]:
    original_budget = production_engine._write_archive_budget_receipt
    original_seal = production_engine.seal_map_upload_package

    def seal_and_stop(root: Path | str) -> None:
        if not _bool_env("FIREVIEWER_FAST_FINALIZE", False):
            original_seal(root)
        raise _SealedFolderReady(Path(root).resolve(strict=True))

    production_engine._write_archive_budget_receipt = _skip_archive_budget
    production_engine.seal_map_upload_package = seal_and_stop
    try:
        yield
    finally:
        production_engine._write_archive_budget_receipt = original_budget
        production_engine.seal_map_upload_package = original_seal


def _validate_no_placeholders(zone_receipt: Mapping[str, Any]) -> None:
    placeholder_count = zone_receipt.get("placeholder_instance_count", 0)
    if placeholder_count != 0:
        raise MapValidationJobError(
            "real-assets-only contract violated: "
            f"{placeholder_count} placeholder instances"
        )


def _validate_complete_viewer(
    job_root: Path,
    zone_receipt: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    glb_path = job_root / VIEWER_GLB_NAME
    viewer = receipt.get("viewer")
    completeness = receipt.get("completeness")
    if not glb_path.is_file() or not isinstance(viewer, Mapping):
        raise MapValidationJobError("complete viewer GLB is missing")
    if not isinstance(completeness, Mapping):
        raise MapValidationJobError("viewer completeness receipt is missing")
    if (
        completeness.get("policy") != "fail_closed_exact_visual_scene"
        or completeness.get("mesh_coverage") != "complete"
    ):
        raise MapValidationJobError(
            "viewer is not declared as the complete non-simplified map"
        )
    expected_counts = {
        "buildings": zone_receipt.get("building_count"),
        "trees": zone_receipt.get("tree_count"),
        "context_assets": zone_receipt.get("context_asset_count", 0),
    }
    observed_counts = completeness.get("family_instance_counts")
    if (
        not isinstance(observed_counts, Mapping)
        or any(
            observed_counts.get(family) != count
            for family, count in expected_counts.items()
        )
    ):
        raise MapValidationJobError(
            "viewer instance counts differ from the canonical map"
        )
    if viewer.get("sha256") != _sha256_file(glb_path):
        raise MapValidationJobError("viewer SHA-256 differs from the exported GLB")
    if viewer.get("byte_count") != glb_path.stat().st_size:
        raise MapValidationJobError("viewer byte count differs from the exported GLB")
    return {
        "path": VIEWER_GLB_NAME,
        "receipt_path": VIEWER_RECEIPT_NAME,
        "sha256": viewer["sha256"],
        "byte_count": viewer["byte_count"],
        "representation": "complete_non_simplified_map",
        "completeness": dict(completeness),
    }


def _run_viewer_export(
    config: ProductionConfig,
    job_root: Path,
    zone_receipt: Mapping[str, Any],
    viewer_phase_times: dict[str, float] | None = None,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    tiled_timeout = int(
        os.environ.get(
            "FIREVIEWER_VALIDATION_TILED_PROTOTYPE_TIMEOUT_SECONDS", "1800"
        )
    )
    tiled_build_started = time.perf_counter()
    build_tiled_viewer_package(
        job_root,
        blender=config.blender,
        timeout_seconds=tiled_timeout,
    )
    tiled_build_finished = time.perf_counter()
    try:
        tiled_receipt, tiled_viewer = validate_tiled_viewer_package(job_root)
    except Exception as error:
        raise MapValidationJobError("tiled viewer package is invalid") from error
    tiled_validation_finished = time.perf_counter()
    if viewer_phase_times is not None:
        viewer_phase_times.update(
            {
                "tiled_build": round(
                    tiled_build_finished - tiled_build_started, 6
                ),
                "tiled_validation": round(
                    tiled_validation_finished - tiled_build_finished, 6
                ),
            }
        )

    policy = os.environ.get(
        "FIREVIEWER_VALIDATION_MONOLITHIC_VIEWER", "off"
    ).strip().lower()
    if policy not in {"off", "auto", "required"}:
        raise MapValidationJobError(
            "FIREVIEWER_VALIDATION_MONOLITHIC_VIEWER must be off, auto, or required"
        )
    tile_count = zone_receipt.get("tile_count")
    maximum_tiles = int(
        os.environ.get("FIREVIEWER_VALIDATION_MONOLITHIC_MAX_TILES", "9")
    )
    should_export = policy == "required" or (
        policy == "auto"
        and isinstance(tile_count, int)
        and not isinstance(tile_count, bool)
        and tile_count <= maximum_tiles
    )
    if not should_export:
        return (
            None,
            {
                "status": "skipped",
                "policy": policy,
                "reason": (
                    "disabled"
                    if policy == "off"
                    else f"tile_count_exceeds_{maximum_tiles}"
                ),
            },
            tiled_receipt,
            tiled_viewer,
        )

    script_name = os.environ.get(
        "FIREVIEWER_VALIDATION_VIEWER_SCRIPT",
        "export_complete_viewer_glb.py",
    ).strip()
    if not script_name or "/" in script_name or "\\" in script_name:
        raise MapValidationJobError("viewer script name is invalid")
    script = Path(__file__).with_name(script_name).resolve(strict=True)
    command = (
        str(config.blender),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        "--job-root",
        str(job_root),
    )
    timeout = int(
        os.environ.get("FIREVIEWER_VALIDATION_VIEWER_TIMEOUT_SECONDS", "1800")
    )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        if policy == "required":
            raise MapValidationJobError("monolithic viewer oracle timed out") from error
        return (
            None,
            {
                "status": "failed_non_blocking",
                "policy": policy,
                "reason": "timeout",
                "timeout_seconds": timeout,
            },
            tiled_receipt,
            tiled_viewer,
        )
    if result.returncode != 0:
        details = (result.stdout + "\n" + result.stderr)[-5000:]
        if policy == "required":
            raise MapValidationJobError(
                "monolithic viewer oracle failed:\n" + details
            )
        return (
            None,
            {
                "status": "failed_non_blocking",
                "policy": policy,
                "reason": "export_failed",
                "returncode": result.returncode,
                "log_tail": details,
            },
            tiled_receipt,
            tiled_viewer,
        )
    receipt_path = job_root / VIEWER_RECEIPT_NAME
    if not receipt_path.is_file():
        if policy == "required":
            raise MapValidationJobError(
                "monolithic viewer oracle completed without its receipt"
            )
        return (
            None,
            {
                "status": "failed_non_blocking",
                "policy": policy,
                "reason": "receipt_missing",
            },
            tiled_receipt,
            tiled_viewer,
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        if policy == "required":
            raise MapValidationJobError(
                "monolithic viewer oracle receipt is invalid"
            ) from error
        return (
            None,
            {
                "status": "failed_non_blocking",
                "policy": policy,
                "reason": "receipt_invalid",
            },
            tiled_receipt,
            tiled_viewer,
        )
    if not isinstance(receipt, dict):
        if policy == "required":
            raise MapValidationJobError(
                "monolithic viewer oracle receipt is invalid"
            )
        return (
            None,
            {
                "status": "failed_non_blocking",
                "policy": policy,
                "reason": "receipt_invalid",
            },
            tiled_receipt,
            tiled_viewer,
        )
    try:
        viewer = _validate_complete_viewer(job_root, zone_receipt, receipt)
    except Exception as error:
        if policy == "required":
            raise MapValidationJobError(
                "monolithic viewer oracle is incomplete"
            ) from error
        return (
            receipt,
            {
                "status": "failed_non_blocking",
                "policy": policy,
                "reason": "validation_failed",
            },
            tiled_receipt,
            tiled_viewer,
        )
    return (
        receipt,
        {"status": "complete", "policy": policy, "viewer": viewer},
        tiled_receipt,
        tiled_viewer,
    )


def _publish_complete_viewer_public(
    job_root: Path,
    *,
    dataset_id: str,
    zone_id: str,
    build_id: str,
    viewer: Mapping[str, Any],
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = _required_env("HF_TOKEN")
    api = HfApi(token=token)
    info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
    if getattr(info, "private", None) is not False:
        raise MapValidationJobError("map dataset must be public")
    remote_root = f"maps/{zone_id}/{build_id}/runtime"
    names = [f"{TILED_OUTPUT_DIRECTORY}/**"]
    if (job_root / VIEWER_RECEIPT_NAME).is_file():
        names.append(VIEWER_RECEIPT_NAME)
    commit = api.upload_folder(
        repo_id=dataset_id,
        repo_type="dataset",
        folder_path=str(job_root),
        path_in_repo=remote_root,
        allow_patterns=names,
        commit_message=f"Publish complete FireViewer runtime {zone_id}",
    )
    oid = getattr(commit, "oid", None)
    if not isinstance(oid, str) or not oid:
        raise MapValidationJobError("viewer publication revision is missing")
    required = (
        str(viewer["catalog_path"]),
        str(viewer["receipt_path"]),
        str(viewer["bootstrap_asset"]["path"]),
    )
    missing = [
        name
        for name in required
        if not api.file_exists(
            repo_id=dataset_id,
            filename=f"{remote_root}/{name}",
            repo_type="dataset",
            revision=oid,
        )
    ]
    if missing:
        raise MapValidationJobError(
            "public viewer files missing after HF commit: " + ", ".join(missing)
        )
    return {
        "status": "published_public",
        "dataset": {
            "repo_id": dataset_id,
            "revision": oid,
            "root": remote_root,
            "visibility": "public",
        },
        "viewer": {
            "catalog_path": f"{remote_root}/{viewer['catalog_path']}",
            "receipt_path": f"{remote_root}/{viewer['receipt_path']}",
            "catalog_byte_count": viewer["catalog_byte_count"],
            "payload_file_count": viewer["payload_file_count"],
            "payload_byte_count": viewer["payload_byte_count"],
            "bootstrap_asset": {
                **dict(viewer["bootstrap_asset"]),
                "path": f"{remote_root}/{viewer['bootstrap_asset']['path']}",
            },
            "representation": "complete_tiled_non_simplified_map",
            "completeness": dict(viewer["completeness"]),
        },
    }


def _producer_for_profile(profile: str) -> Any:
    """Load v2 only when requested so legacy-v1 cannot be monkey-patched."""

    if profile == "legacy-v1":
        return production_engine.produce_simple_measured_tile
    if profile == "factual-v2":
        from produce_simple_measured_tile_v2 import (
            produce_simple_measured_tile_v2,
        )

        return produce_simple_measured_tile_v2
    raise MapValidationJobError(f"unsupported validation profile: {profile}")


def run() -> dict[str, Any]:
    started = time.perf_counter()
    request = _load_request()
    provider = _required_env("FIREVIEWER_VALIDATION_PROVIDER").strip().lower()
    profile = os.environ.get(
        "FIREVIEWER_VALIDATION_PROFILE",
        "legacy-v1",
    ).strip()
    if profile not in PROFILES:
        raise MapValidationJobError(f"unsupported validation profile: {profile}")
    config = ProductionConfig.from_environment()
    plan = plan_zone(
        request["latitude"],
        request["longitude"],
        request["side_km"],
        max_side_m=config.max_side_m,
        max_tiles=config.max_tiles,
    )
    tile_count = len(plan.tiles)
    require_nine_tiles = _bool_env(
        "FIREVIEWER_VALIDATION_REQUIRE_NINE_TILES",
        True,
    )
    create_evidence = _bool_env(
        "FIREVIEWER_VALIDATION_CREATE_EVIDENCE",
        True,
    ) and not _bool_env("FIREVIEWER_FAST_FINALIZE", False)
    if require_nine_tiles and tile_count != 9:
        raise MapValidationJobError(
            "hard stop: validation request must resolve to exactly 9 tiles, "
            f"got {tile_count}"
        )

    shard_index_raw = os.environ.get("FIREVIEWER_TILE_SHARD_INDEX", "").strip()
    shard_count_raw = os.environ.get("FIREVIEWER_TILE_SHARD_COUNT", "").strip()
    if bool(shard_index_raw) != bool(shard_count_raw):
        raise MapValidationJobError(
            "FIREVIEWER_TILE_SHARD_INDEX and FIREVIEWER_TILE_SHARD_COUNT "
            "must be set together"
        )
    try:
        shard_index = int(shard_index_raw) if shard_index_raw else None
        shard_count = int(shard_count_raw) if shard_count_raw else None
    except ValueError as error:
        raise MapValidationJobError("tile shard coordinates must be integers") from error

    selected_producer = _producer_for_profile(profile)
    engine_config = replace(
        config,
        dataset_id=None,
        dataset_publication_required=False,
    )
    engine = ProductionEngine(
        engine_config,
        produce_tile_fn=selected_producer,
    )
    fixed = normalize_fixed_assets(
        request["fixed_asset_placements"] or dict(EMPTY_REQUEST),
        engine.asset_library_payload,
    )

    job_root: Path | None = None
    progress_events: list[dict[str, Any]] = []

    def progress(fraction: float, message: str) -> None:
        event = {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "progress": round(float(fraction), 6),
            "message": str(message),
        }
        progress_events.append(event)
        print(
            "FIREVIEWER_VALIDATION_PROGRESS "
            + json.dumps(event, ensure_ascii=False),
            flush=True,
        )

    try:
        with _sealed_folder_mode():
            for _message, archive, _gallery in engine.run(
                request["latitude"],
                request["longitude"],
                request["side_km"],
                progress_callback=progress,
                archive_ready_callback=None,
                fixed_asset_placements=fixed,
                tile_shard_index=shard_index,
                tile_shard_count=shard_count,
            ):
                if archive is not None:
                    raise MapValidationJobError(
                        "folder-native validation unexpectedly produced an archive"
                    )
    except _SealedFolderReady as ready:
        job_root = ready.root
    except TileCheckpointShardReady as ready:
        if shard_index is None or shard_count is None:
            raise MapValidationJobError(
                "engine returned a tile shard outside shard mode"
            ) from ready
        job_root = ready.root
        completed = time.perf_counter()
        result = {
            "schema": SCHEMA,
            "status": "tile_shard_completed",
            "provider": provider,
            "profile": profile,
            "request_sha256": request_sha256(request),
            "request": request,
            "zone_id": ready.receipt["zone_id"],
            "tile_count": tile_count,
            "shard": ready.receipt,
            "timings_seconds": {"total": round(completed - started, 3)},
            "progress_events": progress_events,
        }
        _write_json(job_root / RESULT_NAME, result)
        print(
            "FIREVIEWER_VALIDATION_SHARD_RESULT "
            + json.dumps(result, ensure_ascii=False),
            flush=True,
        )
        return result

    if job_root is None:
        raise MapValidationJobError("engine did not return a sealed map folder")
    if not _bool_env("FIREVIEWER_FAST_FINALIZE", False):
        validate_map_upload_package(job_root)
    zone_receipt = json.loads(
        (job_root / "zone.done.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(zone_receipt, dict)
        or zone_receipt.get("tile_count") != tile_count
    ):
        raise MapValidationJobError(
            "sealed zone receipt tile count differs from the production plan"
        )
    _validate_no_placeholders(zone_receipt)
    zone_id = str(zone_receipt["zone_id"])
    build_id = str(zone_receipt["build_id"])

    placement_finished = time.perf_counter()
    placement_evidence: dict[str, Any] | None = None
    if create_evidence:
        placement_evidence = create_and_publish_validation_folder(
            job_root,
            provider=provider,
            stage="placement",
            require_nine_tiles=require_nine_tiles,
        )
        if (
            _bool_env("FIREVIEWER_VALIDATION_REQUIRE_EVIDENCE_UPLOAD", True)
            and not placement_evidence.get("publication")
        ):
            raise MapValidationJobError(
                "placement evidence was created but not published; "
                "refusing expensive viewer export"
            )
    placement_evidence_finished = time.perf_counter()

    viewer_phase_times: dict[str, float] = {}
    viewer_receipt, monolithic_viewer, tiled_viewer_receipt, viewer = _run_viewer_export(
        config,
        job_root,
        zone_receipt,
        viewer_phase_times,
    )
    viewer_finished = time.perf_counter()

    viewer_evidence: dict[str, Any] | None = None
    if create_evidence:
        viewer_evidence = create_and_publish_validation_folder(
            job_root,
            provider=provider,
            stage="viewer",
            require_nine_tiles=require_nine_tiles,
        )
        if (
            _bool_env("FIREVIEWER_VALIDATION_REQUIRE_EVIDENCE_UPLOAD", True)
            and not viewer_evidence.get("publication")
        ):
            raise MapValidationJobError(
                "viewer validation evidence was not published"
            )
    viewer_evidence_finished = time.perf_counter()

    viewer_publication: dict[str, Any] | None = None
    viewer_publication_error: str | None = None
    if _bool_env("FIREVIEWER_VALIDATION_PUBLISH_VIEWER", True):
        dataset_id = _required_env("FIREVIEWER_HF_DATASET_ID")
        try:
            viewer_publication = _publish_complete_viewer_public(
                job_root,
                dataset_id=dataset_id,
                zone_id=zone_id,
                build_id=build_id,
                viewer=viewer,
            )
        except Exception as error:
            viewer_publication_error = f"{type(error).__name__}: {error}"
            if _bool_env("FIREVIEWER_VALIDATION_REQUIRE_VIEWER_PUBLICATION", True):
                raise MapValidationJobError(
                    "complete viewer could not be published publicly"
                ) from error
    elif _bool_env("FIREVIEWER_VALIDATION_REQUIRE_VIEWER_PUBLICATION", True):
        raise MapValidationJobError(
            "viewer publication is required but FIREVIEWER_VALIDATION_PUBLISH_VIEWER is disabled"
        )

    completed = time.perf_counter()
    dataset = (
        dict(viewer_publication["dataset"])
        if isinstance(viewer_publication, Mapping)
        else None
    )
    public_viewer = (
        dict(viewer_publication["viewer"])
        if isinstance(viewer_publication, Mapping)
        else None
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed",
        "provider": provider,
        "profile": profile,
        "request_sha256": request_sha256(request),
        "request": request,
        "zone_id": zone_id,
        "build_id": build_id,
        "tile_count": tile_count,
        "tile_origins_l93_m": [
            list(tile.origin_l93_m) for tile in plan.tiles
        ],
        "counts": {
            "buildings": zone_receipt.get("building_count"),
            "trees": zone_receipt.get("tree_count"),
            "context_assets": zone_receipt.get("context_asset_count", 0),
            "placeholder_instances": zone_receipt.get(
                "placeholder_instance_count", 0
            ),
            "degraded_mns_tiles": zone_receipt.get(
                "degraded_mns_tile_count", 0
            ),
        },
        "validation_artifact_role": "evidence_only_not_a_map",
        "placement_validation_folder": placement_evidence,
        "viewer_validation_folder": viewer_evidence,
        "viewer_receipt": viewer_receipt,
        "tiled_viewer_receipt": tiled_viewer_receipt,
        "monolithic_viewer_build_oracle": monolithic_viewer,
        "viewer_representation": "complete_tiled_non_simplified_map",
        "viewer_publication": viewer_publication,
        "viewer_publication_error": viewer_publication_error,
        "dataset": dataset,
        "viewer": public_viewer,
        "viewer_phase_times_seconds": viewer_phase_times,
        "timings_seconds": {
            "sealed_map": round(placement_finished - started, 3),
            "placement_evidence": round(
                placement_evidence_finished - placement_finished,
                3,
            ),
            "viewer_export": round(
                viewer_finished - placement_evidence_finished,
                3,
            ),
            "viewer_evidence": round(
                viewer_evidence_finished - viewer_finished,
                3,
            ),
            "viewer_publication": round(
                completed - viewer_evidence_finished,
                3,
            ),
            "total": round(completed - started, 3),
        },
        "progress_events": progress_events,
    }
    _write_json(job_root / RESULT_NAME, result)
    print(
        "FIREVIEWER_VALIDATION_RESULT "
        + json.dumps(result, ensure_ascii=False),
        flush=True,
    )
    return result


def main() -> None:
    print(
        json.dumps(run(), ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


if __name__ == "__main__":
    main()
