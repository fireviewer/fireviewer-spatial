"""Modal CPU Serverless deployment for asynchronous FireViewer map production."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import modal

APP_NAME = "fireviewer-map-production"
JOB_SCHEMA = "fireviewer.modal-map-job.v2"
PUBLIC_SCHEMA = "fireviewer.simple-production-job.v1"
IMAGE_REFERENCE = os.environ.get(
    "FIREVIEWER_MODAL_IMAGE_REF",
    "charlibillabert/fireviewer-simple-production-ui:pilot-v1-20260813-r14-runpod",
)
VOLUME_NAME = "fireviewer-map-production-work-v1"
JOB_STORE_NAME = "fireviewer-map-production-jobs-v1"
SECRET_NAME = "fireviewer-map-production-secrets"
JOB_ID_RE = re.compile(r"^map_[0-9a-f]{32}$")


class ModalMapContractError(ValueError):
    """The submitted job or stored status violates the public contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def normalize_map_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ModalMapContractError("La requête de carte doit être un objet JSON")
    allowed = {"latitude", "longitude", "side_km", "fixed_asset_placements"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ModalMapContractError(
            "Champs de requête inattendus: " + ", ".join(unexpected)
        )
    try:
        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])
        side_km = float(payload["side_km"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModalMapContractError(
            "latitude, longitude et side_km sont obligatoires"
        ) from error
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise ModalMapContractError("Coordonnées WGS84 hors limites")
    if not (0.5 <= side_km <= 15):
        raise ModalMapContractError("side_km doit rester entre 0,5 et 15 km")
    fixed = payload.get("fixed_asset_placements")
    if fixed is not None and not isinstance(fixed, dict):
        raise ModalMapContractError("fixed_asset_placements doit être un objet JSON")
    return {
        "latitude": latitude,
        "longitude": longitude,
        "side_km": side_km,
        "fixed_asset_placements": dict(fixed) if fixed is not None else None,
    }


def request_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(normalize_map_request(payload))).hexdigest()


def job_id_for_request(payload: Any) -> str:
    return f"map_{request_sha256(payload)[:32]}"


def initial_job_status(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if not JOB_ID_RE.fullmatch(job_id):
        raise ModalMapContractError("job_id Modal invalide")
    return {
        "schema": JOB_SCHEMA,
        "job_id": job_id,
        "request": request,
        "request_sha256": request_sha256(request),
        "state": "queued",
        "phase": "queued",
        "progress": 0.0,
        "message": "En attente d'une ressource Modal CPU Serverless",
        "current_tile": None,
        "tile_count": None,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "dataset": None,
        "archive": None,
        "captures": [],
    }


def public_job_status(status: Any) -> dict[str, Any]:
    if not isinstance(status, dict) or status.get("schema") != JOB_SCHEMA:
        raise ModalMapContractError("Statut Modal absent ou invalide")
    job_id = status.get("job_id")
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise ModalMapContractError("Statut Modal avec job_id invalide")
    if status.get("captures") != []:
        raise ModalMapContractError("Inventaire des captures Modal invalide")
    return {
        "schema": PUBLIC_SCHEMA,
        "job_id": job_id,
        "kind": "map",
        "request_sha256": status["request_sha256"],
        "state": status["state"],
        "phase": status["phase"],
        "progress": status["progress"],
        "message": status["message"],
        "current_tile": status["current_tile"],
        "tile_count": status["tile_count"],
        "created_at": status["created_at"],
        "started_at": status["started_at"],
        "finished_at": status["finished_at"],
        "error": status["error"],
        "archive_url": (
            f"/v1/map-jobs/{job_id}/download-link"
            if status["state"] == "completed"
            else None
        ),
        "captures": [],
    }


app = modal.App(APP_NAME)
work_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
job_store = modal.Dict.from_name(JOB_STORE_NAME, create_if_missing=True)
production_secret = modal.Secret.from_name(SECRET_NAME)

runtime_image = (
    modal.Image.from_registry(IMAGE_REFERENCE)
    .add_local_dir(
        Path(__file__).resolve().parent,
        "/opt/fireviewer/fireviewer-spatial/blender",
        copy=True,
        ignore=["**/__pycache__/**", "test_*.py"],
    )
    .env(
        {
            "PYTHONPATH": (
                "/opt/fireviewer/fireviewer-spatial/blender:"
                "/opt/fireviewer/fireviewer-spatial/omniverse"
            ),
            "FIREVIEWER_WORK_ROOT": "/work",
            "FIREVIEWER_TILE_WORKERS": "8",
            "FIREVIEWER_IMAGE_REFERENCE": "modal-cpu-serverless-no-captures-v1",
        }
    )
)

api_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.128.8", "requests==2.32.5"
)


@app.function(
    image=runtime_image,
    secrets=[production_secret],
    volumes={"/work": work_volume},
    cpu=8.0,
    memory=32768,
    timeout=86_400,
    min_containers=0,
    max_containers=1,
    scaledown_window=30,
)
def produce_map(job_id: str, request: dict[str, Any]) -> None:
    from simple_production_engine import ProductionConfig, ProductionEngine

    status = initial_job_status(job_id, request)
    status.update(state="running", phase="starting", started_at=_now())
    job_store[job_id] = status
    try:
        engine = ProductionEngine(ProductionConfig.from_environment())

        def progress(fraction: float, message: str) -> None:
            current = dict(job_store[job_id])
            match = re.search(r"Tuile\s+(\d+)/(\d+)", message, re.IGNORECASE)
            current.update(
                state="running",
                phase="in_progress",
                progress=max(0.0, min(0.999, float(fraction))),
                message=str(message),
                current_tile=int(match.group(1)) if match else current["current_tile"],
                tile_count=int(match.group(2)) if match else current["tile_count"],
            )
            job_store[job_id] = current

        archive_path: Path | None = None
        final_message = "Production terminée"
        for message, archive, captures in engine.run(
            request["latitude"],
            request["longitude"],
            request["side_km"],
            fixed_asset_placements=request.get("fixed_asset_placements"),
            progress_callback=progress,
        ):
            if captures:
                raise ModalMapContractError("Le moteur a produit une galerie obsolète")
            final_message = message
            if archive:
                archive_path = Path(archive)
        if archive_path is None or not archive_path.is_file():
            raise ModalMapContractError("Le moteur n'a pas publié le ZIP autonome")
        root = archive_path.parent
        publication = json.loads(
            (root / "dataset-publication.json").read_text(encoding="utf-8")
        )
        receipt = json.loads((root / "zone.done.json").read_text(encoding="utf-8"))
        if publication.get("captures") != []:
            raise ModalMapContractError("La publication contient une galerie obsolète")
        remote_root = str(publication["path_in_repo"])
        status = dict(job_store[job_id])
        status.update(
            state="completed",
            phase="completed",
            progress=1.0,
            message=final_message,
            current_tile=receipt["tile_count"],
            tile_count=receipt["tile_count"],
            finished_at=_now(),
            dataset={
                "repo_id": publication["dataset_id"],
                "revision": publication["commit_oid"],
                "root": remote_root,
            },
            archive={
                "path": f"{remote_root}/fireviewer-zone.zip",
                "sha256": publication["archive_sha256"],
            },
            captures=[],
        )
        job_store[job_id] = status
        work_volume.commit()
    except Exception as error:
        failed = dict(job_store.get(job_id, status))
        failed.update(
            state="failed",
            phase="failed",
            message="Échec de production",
            error=str(error),
            finished_at=_now(),
        )
        job_store[job_id] = failed
        raise


def _authorize(authorization: str | None) -> None:
    expected = os.environ.get("FIREVIEWER_API_TOKEN", "")
    if not expected or authorization != f"Bearer {expected}":
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Authentification requise")


def _signed_download(status: dict[str, Any]) -> str:
    import requests

    dataset = status.get("dataset")
    archive = status.get("archive")
    token = os.environ.get("HF_TOKEN", "")
    if not isinstance(dataset, dict) or not isinstance(archive, dict) or not token:
        raise ModalMapContractError("Archive privée indisponible")
    url = (
        "https://huggingface.co/datasets/"
        f"{quote(str(dataset['repo_id']), safe='/')}/resolve/"
        f"{quote(str(dataset['revision']), safe='')}/"
        f"{quote(str(archive['path']), safe='/')}?download=true"
    )
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        allow_redirects=False,
        timeout=30,
    )
    location = response.headers.get("location")
    if response.status_code not in {301, 302, 303, 307, 308} or not location:
        raise ModalMapContractError("Lien privé Hugging Face indisponible")
    return location


@app.function(
    image=api_image,
    secrets=[production_secret],
    cpu=0.25,
    memory=512,
    timeout=300,
    min_containers=0,
    max_containers=2,
    scaledown_window=15,
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, Header, HTTPException

    web = FastAPI(
        title="FireViewer Modal map production", docs_url=None, redoc_url=None
    )

    @web.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", "compute": "cpu_serverless", "capture_count": 0}

    @web.post("/v1/map-jobs", status_code=202)
    def submit(
        request: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _authorize(authorization)
        normalized = normalize_map_request(request)
        job_id = job_id_for_request(normalized)
        initial = initial_job_status(job_id, normalized)
        inserted = job_store.put(job_id, initial, skip_if_exists=True)
        if inserted:
            produce_map.spawn(job_id, normalized)
        return public_job_status(job_store[job_id])

    @web.get("/v1/map-jobs/{job_id}")
    def status(
        job_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _authorize(authorization)
        stored = job_store.get(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Job introuvable")
        return public_job_status(stored)

    @web.get("/v1/map-jobs/{job_id}/captures")
    def captures(
        job_id: str, authorization: str | None = Header(default=None)
    ) -> list[dict[str, Any]]:
        _authorize(authorization)
        if job_store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Job introuvable")
        return []

    @web.get("/v1/map-jobs/{job_id}/download-link")
    def download(
        job_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, str]:
        _authorize(authorization)
        stored = job_store.get(job_id)
        if not isinstance(stored, dict):
            raise HTTPException(status_code=404, detail="Job introuvable")
        if stored.get("state") != "completed":
            raise HTTPException(status_code=409, detail="Production incomplète")
        return {"url": _signed_download(stored)}

    return web


__all__ = [
    "JOB_SCHEMA",
    "ModalMapContractError",
    "initial_job_status",
    "job_id_for_request",
    "normalize_map_request",
    "public_job_status",
    "request_sha256",
]
