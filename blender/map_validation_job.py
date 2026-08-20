"""Provider-neutral one-shot FireViewer 9-tile validation job.

This is deliberately separate from the stable Lightning production entrypoint.
It can run the current placement profile (for the comparison Lightning image)
or the factual v2 profile (for the RunPod image), creates a compact validation
pack before any large viewer upload, then exports the exact-count GLB.

The job never publishes the full scientific folder and never creates a final
zone ZIP.  Its only purpose is the low-cost 9-tile A/B validation.
"""

from __future__ import annotations

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
from fixed_asset_placement import EMPTY_REQUEST, normalize_request as normalize_fixed_assets
from map_validation_pack import create_and_publish_validation_pack
from portable_scene_package import validate_map_upload_package
from produce_simple_measured_tile_v2 import produce_simple_measured_tile_v2
from runpod_map_production import normalize_map_request, request_sha256
from simple_production_engine import ProductionConfig, ProductionEngine, plan_zone

SCHEMA = "fireviewer.map-validation-job.v1"
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


def _load_request() -> dict[str, Any]:
    raw = os.environ.get("FIREVIEWER_VALIDATION_REQUEST_JSON", "").strip()
    path = os.environ.get("FIREVIEWER_VALIDATION_REQUEST_FILE", "").strip()
    if bool(raw) == bool(path):
        raise MapValidationJobError(
            "set exactly one of FIREVIEWER_VALIDATION_REQUEST_JSON or "
            "FIREVIEWER_VALIDATION_REQUEST_FILE"
        )
    try:
        payload = json.loads(raw) if raw else json.loads(Path(path).read_text(encoding="utf-8"))
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
    return {"schema": "fireviewer.map-folder-publication.v1", "status": "zip_disabled"}


@contextmanager
def _sealed_folder_mode() -> Iterator[None]:
    original_budget = production_engine._write_archive_budget_receipt
    original_seal = production_engine.seal_map_upload_package

    def seal_and_stop(root: Path | str) -> None:
        original_seal(root)
        raise _SealedFolderReady(Path(root).resolve(strict=True))

    production_engine._write_archive_budget_receipt = _skip_archive_budget
    production_engine.seal_map_upload_package = seal_and_stop
    try:
        yield
    finally:
        production_engine._write_archive_budget_receipt = original_budget
        production_engine.seal_map_upload_package = original_seal


def _run_viewer_export(config: ProductionConfig, job_root: Path) -> dict[str, Any]:
    script_name = os.environ.get(
        "FIREVIEWER_VALIDATION_VIEWER_SCRIPT", "export_complete_viewer_glb.py"
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
    timeout = int(os.environ.get("FIREVIEWER_VALIDATION_VIEWER_TIMEOUT_SECONDS", "1800"))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise MapValidationJobError(
            "viewer export failed:\n" + (result.stdout + "\n" + result.stderr)[-5000:]
        )
    receipt_path = job_root / VIEWER_RECEIPT_NAME
    glb_path = job_root / VIEWER_GLB_NAME
    if not receipt_path.is_file() or not glb_path.is_file():
        raise MapValidationJobError("viewer export completed without GLB/receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise MapValidationJobError("viewer receipt is invalid")
    return receipt


def _publish_viewer_best_effort(
    job_root: Path,
    *,
    dataset_id: str,
    provider: str,
    zone_id: str,
    build_id: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = _required_env("HF_TOKEN")
    api = HfApi(token=token)
    info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
    if getattr(info, "private", None) is not False:
        raise MapValidationJobError("validation dataset must be public")
    remote_root = f"validation-viewers/{zone_id}/{build_id}/{provider}"
    commit = api.upload_folder(
        repo_id=dataset_id,
        repo_type="dataset",
        folder_path=str(job_root),
        path_in_repo=remote_root,
        allow_patterns=[VIEWER_GLB_NAME, VIEWER_RECEIPT_NAME],
        commit_message=f"Publish FireViewer validation viewer {provider}",
    )
    oid = getattr(commit, "oid", None)
    if not isinstance(oid, str) or not oid:
        raise MapValidationJobError("viewer publication revision is missing")
    return {
        "status": "published_public",
        "dataset_id": dataset_id,
        "revision": oid,
        "root": remote_root,
        "viewer": f"{remote_root}/{VIEWER_GLB_NAME}",
        "receipt": f"{remote_root}/{VIEWER_RECEIPT_NAME}",
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    request = _load_request()
    provider = _required_env("FIREVIEWER_VALIDATION_PROVIDER").strip().lower()
    profile = os.environ.get("FIREVIEWER_VALIDATION_PROFILE", "legacy-v1").strip()
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
    if len(plan.tiles) != 9:
        raise MapValidationJobError(
            f"hard stop: validation request must resolve to exactly 9 tiles, got {len(plan.tiles)}"
        )

    engine_config = replace(
        config,
        dataset_id=None,
        dataset_publication_required=False,
    )
    engine = ProductionEngine(
        engine_config,
        produce_tile_fn=(
            produce_simple_measured_tile_v2
            if profile == "factual-v2"
            else production_engine.produce_simple_measured_tile
        ),
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
        print("FIREVIEWER_VALIDATION_PROGRESS " + json.dumps(event, ensure_ascii=False), flush=True)

    try:
        with _sealed_folder_mode():
            for _message, archive, _gallery in engine.run(
                request["latitude"],
                request["longitude"],
                request["side_km"],
                progress_callback=progress,
                archive_ready_callback=None,
                fixed_asset_placements=fixed,
            ):
                if archive is not None:
                    raise MapValidationJobError(
                        "folder-native validation unexpectedly produced a final ZIP"
                    )
    except _SealedFolderReady as ready:
        job_root = ready.root

    if job_root is None:
        raise MapValidationJobError("engine did not return a sealed map folder")
    validate_map_upload_package(job_root)
    zone_receipt = json.loads((job_root / "zone.done.json").read_text(encoding="utf-8"))
    if not isinstance(zone_receipt, dict) or zone_receipt.get("tile_count") != 9:
        raise MapValidationJobError("sealed zone receipt does not contain 9 tiles")
    zone_id = str(zone_receipt["zone_id"])
    build_id = str(zone_receipt["build_id"])

    placement_finished = time.perf_counter()
    placement_pack = create_and_publish_validation_pack(
        job_root,
        provider=provider,
        stage="placement",
        require_nine_tiles=True,
    )
    if _bool_env("FIREVIEWER_VALIDATION_REQUIRE_PACK_UPLOAD", True) and not placement_pack.get(
        "publication"
    ):
        raise MapValidationJobError(
            "placement validation pack was created but not published; refusing expensive viewer export"
        )
    placement_pack_finished = time.perf_counter()

    viewer_receipt = _run_viewer_export(config, job_root)
    viewer_finished = time.perf_counter()
    viewer_pack = create_and_publish_validation_pack(
        job_root,
        provider=provider,
        stage="viewer",
        require_nine_tiles=True,
    )
    if _bool_env("FIREVIEWER_VALIDATION_REQUIRE_PACK_UPLOAD", True) and not viewer_pack.get(
        "publication"
    ):
        raise MapValidationJobError("viewer validation pack was not published")
    viewer_pack_finished = time.perf_counter()

    viewer_publication: dict[str, Any] | None = None
    viewer_publication_error: str | None = None
    if _bool_env("FIREVIEWER_VALIDATION_PUBLISH_VIEWER", True):
        dataset_id = _required_env("FIREVIEWER_HF_DATASET_ID")
        try:
            viewer_publication = _publish_viewer_best_effort(
                job_root,
                dataset_id=dataset_id,
                provider=provider,
                zone_id=zone_id,
                build_id=build_id,
            )
        except Exception as error:
            # The small validation packs are already durable.  A large GLB
            # transfer failure is evidence to record, not a reason to lose the
            # comparison result again.
            viewer_publication_error = f"{type(error).__name__}: {error}"
    completed = time.perf_counter()

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed",
        "provider": provider,
        "profile": profile,
        "request_sha256": request_sha256(request),
        "request": request,
        "zone_id": zone_id,
        "build_id": build_id,
        "tile_count": 9,
        "tile_origins_l93_m": [list(tile.origin_l93_m) for tile in plan.tiles],
        "counts": {
            "buildings": zone_receipt.get("building_count"),
            "trees": zone_receipt.get("tree_count"),
            "context_assets": zone_receipt.get("context_asset_count", 0),
            "degraded_mns_tiles": zone_receipt.get("degraded_mns_tile_count", 0),
        },
        "placement_validation_pack": placement_pack,
        "viewer_validation_pack": viewer_pack,
        "viewer_receipt": viewer_receipt,
        "viewer_publication": viewer_publication,
        "viewer_publication_error": viewer_publication_error,
        "timings_seconds": {
            "sealed_map": round(placement_finished - started, 3),
            "placement_pack": round(placement_pack_finished - placement_finished, 3),
            "viewer_export": round(viewer_finished - placement_pack_finished, 3),
            "viewer_pack": round(viewer_pack_finished - viewer_finished, 3),
            "viewer_publication": round(completed - viewer_pack_finished, 3),
            "total": round(completed - started, 3),
        },
        "progress_events": progress_events,
    }
    _write_json(job_root / RESULT_NAME, result)
    print("FIREVIEWER_VALIDATION_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
