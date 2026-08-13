"""Headless HTTP API for FireViewer map and perimeter production.

The API contains no UI and no database.  It exposes the existing deterministic
production engine through hash-stable jobs below ``FIREVIEWER_WORK_ROOT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fixed_asset_placement import (
    EMPTY_REQUEST as EMPTY_FIXED_ASSET_REQUEST,
)
from fixed_asset_placement import (
    FixedAssetPlacementError,
)
from fixed_asset_placement import (
    normalize_request as normalize_fixed_asset_request,
)
from fixed_asset_placement import (
    project_request as project_fixed_asset_request,
)
from fixed_asset_placement import (
    request_sha256 as fixed_asset_request_sha256,
)
from geographic_perimeter_layer import (
    MAX_INPUT_BYTES,
    produce_perimeter_layer,
)
from geographic_perimeter_viewer import (
    MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    build_perimeter_timeline_viewer,
)
from portable_scene_package import (
    materialize_perimeter_upload_package,
    read_map_reference_from_archive,
)
from pydantic import BaseModel, ConfigDict, Field
from simple_production_engine import (
    CAPTURE_COUNT,
    ProductionConfig,
    ProductionEngine,
    SimpleProductionError,
    plan_zone,
    validate_embedded_assets,
    validate_embedded_runtime,
)

API_SCHEMA = "fireviewer.simple-production-api.v1"
CONTRACT_SCHEMA = "fireviewer.simple-production-api-contract.v1"
JOB_SCHEMA = "fireviewer.simple-production-job.v1"
_ACTIVE_STATES = {"queued", "running"}


class ApiContractError(RuntimeError):
    """The API contract or a persisted job artifact is invalid."""


class ZoneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float
    longitude: float
    side_km: float = Field(gt=0)


class ProductionRequest(ZoneRequest):
    fixed_asset_placements: dict[str, Any] | None = None


class FixedAssetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: dict[str, Any]
    latitude: float | None = None
    longitude: float | None = None
    side_km: float | None = None


@dataclass(slots=True)
class JobRecord:
    job_id: str
    kind: str
    request_sha256: str
    state: str = "queued"
    phase: str = "queued"
    progress: float = 0.0
    message: str = "En attente"
    created_at: str = field(default_factory=lambda: _now())
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    archive: Path | None = None
    captures: list[tuple[Path, str]] = field(default_factory=list)
    capture_media_type: str = "image/png"


class ApiState:
    def __init__(
        self,
        config: ProductionConfig,
        engine: ProductionEngine | Any | None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.jobs: dict[str, JobRecord] = {}
        self.lock = threading.Lock()

    def production_engine(self) -> ProductionEngine | Any:
        with self.lock:
            if self.engine is None:
                self.engine = ProductionEngine(self.config)
            return self.engine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract() -> dict[str, Any]:
    path = Path(__file__).with_name("simple_production_api_contract.v1.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ApiContractError(f"contrat API illisible: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CONTRACT_SCHEMA
        or payload.get("status") != "locked"
        or payload.get("transport", {}).get("framework") != "fastapi"
        or payload.get("transport", {}).get("gradio") != "forbidden"
        or payload.get("transport", {}).get("database") != "forbidden"
        or payload.get("map_production", {}).get("tile_size_m") != 500
        or payload.get("map_production", {}).get("capture_count") != CAPTURE_COUNT
        or payload.get("acceptance", {}).get("automatic_human_acceptance") is not False
    ):
        raise ApiContractError("contrat API absent, modifié ou non verrouillé")
    return payload


def _confined(root: Path, path: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ApiContractError(f"{label} sort du volume de travail") from error
    return path


def _plan_payload(plan: Any) -> dict[str, Any]:
    west, south, east, north = plan.production_bounds_l93_m
    return {
        "schema": API_SCHEMA,
        "zone_id": plan.zone_id,
        "center_wgs84": [plan.latitude, plan.longitude],
        "center_l93_m": list(plan.center_l93_m),
        "requested_side_m": plan.side_m,
        "requested_area_km2": (plan.side_m * plan.side_m) / 1_000_000,
        "production_bounds_l93_m": [west, south, east, north],
        "production_area_km2": ((east - west) * (north - south)) / 1_000_000,
        "tile_size_m": 500,
        "tile_count": len(plan.tiles),
        "tiles": [
            {"tile_id": tile.tile_id, "origin_l93_m": list(tile.origin_l93_m)}
            for tile in plan.tiles
        ],
    }


def _job_payload(record: JobRecord) -> dict[str, Any]:
    captures = [
        {
            "index": index,
            "caption": caption,
            "url": f"/v1/jobs/{record.job_id}/captures/{index}",
            "media_type": record.capture_media_type,
        }
        for index, (_path, caption) in enumerate(record.captures)
    ]
    return {
        "schema": JOB_SCHEMA,
        "job_id": record.job_id,
        "kind": record.kind,
        "request_sha256": record.request_sha256,
        "state": record.state,
        "phase": record.phase,
        "progress": round(record.progress, 6),
        "message": record.message,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error": record.error,
        "archive_url": (
            f"/v1/jobs/{record.job_id}/archive" if record.archive is not None else None
        ),
        "captures": captures,
    }


def _sync_disk_status(state: ApiState, record: JobRecord) -> None:
    path = state.config.work_root / "jobs" / record.job_id / "job-status.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    phase = payload.get("phase")
    if isinstance(phase, str) and phase:
        record.phase = phase


def _run_map_job(
    state: ApiState,
    record: JobRecord,
    request: ProductionRequest,
    fixed_assets: Mapping[str, Any],
) -> None:
    record.state = "running"
    record.phase = "starting"
    record.started_at = _now()

    def progress(fraction: float, message: str) -> None:
        with state.lock:
            record.progress = max(0.0, min(1.0, float(fraction)))
            record.message = str(message)
            _sync_disk_status(state, record)

    try:
        for message, archive, gallery in state.production_engine().run(
            request.latitude,
            request.longitude,
            request.side_km,
            progress_callback=progress,
            fixed_asset_placements=fixed_assets,
        ):
            with state.lock:
                record.message = message
                if archive is not None:
                    record.archive = _confined(
                        state.config.work_root, Path(archive), "archive de carte"
                    )
                if gallery:
                    record.captures = [
                        (
                            _confined(
                                state.config.work_root,
                                Path(path),
                                "capture de carte",
                            ),
                            str(caption),
                        )
                        for path, caption in gallery
                    ]
        with state.lock:
            if record.archive is None or len(record.captures) != CAPTURE_COUNT:
                raise ApiContractError("production terminée sans ZIP autonome")
            record.state = "completed"
            record.phase = "completed"
            record.progress = 1.0
            record.finished_at = _now()
    except Exception as error:  # noqa: BLE001 - persist any worker failure
        with state.lock:
            record.state = "failed"
            record.phase = "failed"
            record.error = str(error)
            record.message = f"Échec : {error}"
            record.finished_at = _now()


def _run_perimeter_job(
    state: ApiState,
    record: JobRecord,
    source: Path,
    map_archive: Path | None,
) -> None:
    record.state = "running"
    record.phase = "perimeter_compile"
    record.started_at = _now()
    try:
        product = produce_perimeter_layer(source, state.config.work_root)
        archive = product.archive
        map_reference = None
        viewer = None
        if map_archive is not None:
            map_reference = read_map_reference_from_archive(map_archive)
            record.phase = "timeline_viewer"
            viewer = build_perimeter_timeline_viewer(
                map_archive, product.package_root, state.config.work_root
            )
            _upload_root, archive, _upload_manifest = (
                materialize_perimeter_upload_package(
                    product.package_root,
                    map_reference,
                    state.config.work_root,
                    viewer_root=viewer.root,
                )
            )
        with state.lock:
            record.progress = 0.55
            record.message = (
                "Calques USD fixes liés à la carte"
                if map_reference is not None
                else "Calques USD fixes produits"
            )
            record.archive = _confined(
                state.config.work_root, archive, "archive des calques"
            )
        if viewer is not None:
            with state.lock:
                record.captures = [
                    (
                        _confined(
                            state.config.work_root,
                            frame.model,
                            "frame de timeline",
                        ),
                        frame.caption,
                    )
                    for frame in viewer.frames
                ]
                record.capture_media_type = "model/gltf-binary"
                record.message = f"{len(record.captures)} états de timeline produits"
        with state.lock:
            record.state = "completed"
            record.phase = "completed"
            record.progress = 1.0
            record.finished_at = _now()
    except Exception as error:  # noqa: BLE001 - persist any worker failure
        with state.lock:
            record.state = "failed"
            record.phase = "failed"
            record.error = str(error)
            record.message = f"Échec : {error}"
            record.finished_at = _now()
    finally:
        shutil.rmtree(source.parent, ignore_errors=True)


async def _save_upload(upload: UploadFile, path: Path, maximum: int) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    size = 0
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as stream:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum:
                    raise HTTPException(
                        status_code=413, detail="fichier trop volumineux"
                    )
                digest.update(chunk)
                stream.write(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail="fichier vide")
        os.replace(temporary, path)
    finally:
        await upload.close()
        if temporary.exists():
            temporary.unlink()
    return size, digest.hexdigest()


def create_app(
    *,
    config: ProductionConfig | None = None,
    engine: ProductionEngine | Any | None = None,
) -> FastAPI:
    """Create the API with optional dependency injection for deterministic tests."""

    contract = _load_contract()
    selected_config = config or ProductionConfig.from_environment()
    api_state = ApiState(selected_config, engine)
    application = FastAPI(title="FireViewer production API", version="1")
    application.state.fireviewer = api_state
    application.state.contract = contract

    origins = [
        value.strip()
        for value in os.environ.get(
            "FIREVIEWER_API_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if value.strip()
    ]
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    def authorize(request: Request) -> None:
        expected = os.environ.get("FIREVIEWER_API_TOKEN", "")
        if not expected:
            return
        supplied = request.headers.get("authorization", "")
        if supplied != f"Bearer {expected}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="jeton API invalide",
            )

    def current_state(request: Request) -> ApiState:
        return request.app.state.fireviewer

    @application.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", "schema": API_SCHEMA}

    @application.get("/v1/config", dependencies=[Depends(authorize)])
    def get_config(
        state: ApiState = Depends(current_state),  # noqa: B008
    ) -> dict[str, Any]:
        selected = state.production_engine()
        return {
            "schema": API_SCHEMA,
            "limits": {
                "minimum_side_km": 0.5,
                "maximum_side_km": state.config.max_side_m / 1000,
                "maximum_tiles": state.config.max_tiles,
                "tile_size_m": 500,
            },
            "assets": {
                "count": selected.asset_summary["asset_count"],
                "catalog_revision": selected.asset_summary["catalog_revision"],
                "choices": [
                    {"label": label, "asset_id": asset_id}
                    for label, asset_id in selected.fixed_asset_choices
                ],
            },
            "fixed_asset_template": dict(EMPTY_FIXED_ASSET_REQUEST),
            "capabilities": {
                "map_production": True,
                "fixed_asset_placement": True,
                "perimeter_layers": True,
                "timeline_viewer": True,
                "human_auto_acceptance": False,
            },
        }

    @application.post("/v1/plan", dependencies=[Depends(authorize)])
    def create_plan(
        body: ZoneRequest,
        state: ApiState = Depends(current_state),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            plan = plan_zone(
                body.latitude,
                body.longitude,
                body.side_km,
                max_side_m=state.config.max_side_m,
                max_tiles=state.config.max_tiles,
            )
            return _plan_payload(plan)
        except (SimpleProductionError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/v1/fixed-assets/validate", dependencies=[Depends(authorize)])
    def validate_fixed_assets(
        body: FixedAssetRequest,
        state: ApiState = Depends(current_state),  # noqa: B008
    ) -> dict[str, Any]:
        selected = state.production_engine()
        try:
            normalized = normalize_fixed_asset_request(
                body.request, selected.asset_library_payload
            )
            projected: Sequence[Mapping[str, Any]] = ()
            if None not in (body.latitude, body.longitude, body.side_km):
                plan = plan_zone(
                    float(body.latitude),
                    float(body.longitude),
                    float(body.side_km),
                    max_side_m=state.config.max_side_m,
                    max_tiles=state.config.max_tiles,
                )
                projected = project_fixed_asset_request(
                    normalized,
                    selected.asset_library_payload,
                    requested_bounds_l93_m=plan.requested_bounds_l93_m,
                )
            return {
                "schema": API_SCHEMA,
                "request": normalized,
                "request_sha256": fixed_asset_request_sha256(normalized),
                "placement_count": len(normalized["placements"]),
                "projected": list(projected),
            }
        except (FixedAssetPlacementError, SimpleProductionError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post(
        "/v1/jobs",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authorize)],
    )
    def create_job(
        body: ProductionRequest,
        state: ApiState = Depends(current_state),  # noqa: B008
    ) -> dict[str, Any]:
        selected = state.production_engine()
        try:
            plan = plan_zone(
                body.latitude,
                body.longitude,
                body.side_km,
                max_side_m=state.config.max_side_m,
                max_tiles=state.config.max_tiles,
            )
            fixed = normalize_fixed_asset_request(
                body.fixed_asset_placements or EMPTY_FIXED_ASSET_REQUEST,
                selected.asset_library_payload,
            )
            if fixed["placements"]:
                project_fixed_asset_request(
                    fixed,
                    selected.asset_library_payload,
                    requested_bounds_l93_m=plan.requested_bounds_l93_m,
                )
                plan = replace(
                    plan,
                    zone_id=(
                        f"{plan.zone_id}-fixed-{fixed_asset_request_sha256(fixed)[:12]}"
                    ),
                )
        except (FixedAssetPlacementError, SimpleProductionError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        request_payload = {
            "latitude": body.latitude,
            "longitude": body.longitude,
            "side_km": body.side_km,
            "fixed_asset_placements": fixed,
        }
        request_hash = hashlib.sha256(_canonical_bytes(request_payload)).hexdigest()
        with state.lock:
            existing = state.jobs.get(plan.zone_id)
            if existing is not None:
                if existing.request_sha256 != request_hash:
                    raise HTTPException(
                        status_code=409, detail="identifiant de job occupé"
                    )
                return _job_payload(existing)
            if any(job.state in _ACTIVE_STATES for job in state.jobs.values()):
                raise HTTPException(
                    status_code=409, detail="une production est déjà active"
                )
            record = JobRecord(plan.zone_id, "map", request_hash)
            state.jobs[record.job_id] = record
        threading.Thread(
            target=_run_map_job,
            args=(state, record, body, fixed),
            name=f"fireviewer-map-{record.job_id}",
            daemon=True,
        ).start()
        return _job_payload(record)

    @application.post(
        "/v1/perimeter-jobs",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authorize)],
    )
    async def create_perimeter_job(
        source: UploadFile = File(...),  # noqa: B008
        map_archive: UploadFile | None = File(None),  # noqa: B008
        state: ApiState = Depends(current_state),  # noqa: B008
    ) -> dict[str, Any]:
        suffix = Path(source.filename or "").suffix.casefold()
        if suffix not in {".json", ".geojson"}:
            raise HTTPException(
                status_code=422, detail="source JSON ou GeoJSON requise"
            )
        upload_root = state.config.work_root / "api-uploads"
        temporary_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        source_path = upload_root / temporary_id / f"source{suffix}"
        try:
            _size, source_hash = await _save_upload(
                source, source_path, MAX_INPUT_BYTES
            )
            map_path: Path | None = None
            map_hash: str | None = None
            if map_archive is not None and map_archive.filename:
                map_path = source_path.parent / "map.zip"
                _map_size, map_hash = await _save_upload(
                    map_archive, map_path, MAX_ARCHIVE_UNCOMPRESSED_BYTES
                )
        except Exception:
            shutil.rmtree(source_path.parent, ignore_errors=True)
            raise
        request_hash = hashlib.sha256(
            _canonical_bytes({"source_sha256": source_hash, "map_sha256": map_hash})
        ).hexdigest()
        job_id = f"perimeter-{request_hash[:20]}"
        with state.lock:
            existing = state.jobs.get(job_id)
            if existing is not None:
                shutil.rmtree(source_path.parent, ignore_errors=True)
                return _job_payload(existing)
            if any(job.state in _ACTIVE_STATES for job in state.jobs.values()):
                shutil.rmtree(source_path.parent, ignore_errors=True)
                raise HTTPException(
                    status_code=409, detail="une production est déjà active"
                )
            record = JobRecord(job_id, "perimeter", request_hash)
            state.jobs[job_id] = record
        threading.Thread(
            target=_run_perimeter_job,
            args=(state, record, source_path, map_path),
            name=f"fireviewer-perimeter-{job_id}",
            daemon=True,
        ).start()
        return _job_payload(record)

    @application.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def get_job(
        job_id: str,
        state: ApiState = Depends(current_state),  # noqa: B008
    ) -> dict[str, Any]:
        with state.lock:
            record = state.jobs.get(job_id)
            if record is None:
                raise HTTPException(status_code=404, detail="job inconnu")
            return _job_payload(record)

    @application.get("/v1/jobs/{job_id}/archive", dependencies=[Depends(authorize)])
    def get_archive(
        job_id: str,
        state: ApiState = Depends(current_state),  # noqa: B008
    ) -> FileResponse:
        with state.lock:
            record = state.jobs.get(job_id)
            if record is None or record.archive is None:
                raise HTTPException(status_code=404, detail="archive indisponible")
            path = _confined(state.config.work_root, record.archive, "archive")
        return FileResponse(
            path,
            filename=path.name,
            media_type="application/zip",
            headers={"X-FireViewer-SHA256": _sha256_file(path)},
        )

    @application.get(
        "/v1/jobs/{job_id}/captures/{index}", dependencies=[Depends(authorize)]
    )
    def get_capture(
        job_id: str,
        index: int,
        state: ApiState = Depends(current_state),  # noqa: B008
    ) -> FileResponse:
        with state.lock:
            record = state.jobs.get(job_id)
            if record is None or not 0 <= index < len(record.captures):
                raise HTTPException(status_code=404, detail="capture indisponible")
            path, _caption = record.captures[index]
            path = _confined(state.config.work_root, path, "capture")
            media_type = record.capture_media_type
        return FileResponse(
            path,
            filename=path.name,
            media_type=media_type,
            headers={"X-FireViewer-SHA256": _sha256_file(path)},
        )

    return application


app = create_app()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verify-assets-only", action="store_true")
    parser.add_argument("--verify-runtime-only", action="store_true")
    options = parser.parse_args(argv)
    config = ProductionConfig.from_environment()
    if options.verify_assets_only:
        print(json.dumps(validate_embedded_assets(config), sort_keys=True))
        return 0
    if options.verify_runtime_only:
        print(json.dumps(validate_embedded_runtime(config), sort_keys=True))
        return 0
    import uvicorn

    uvicorn.run(app, host=options.host, port=options.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "API_SCHEMA",
    "ApiContractError",
    "FixedAssetRequest",
    "ProductionRequest",
    "ZoneRequest",
    "app",
    "create_app",
    "main",
]
