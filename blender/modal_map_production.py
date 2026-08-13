"""Modal CPU Serverless deployment for asynchronous FireViewer map production."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
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
ACTIVE_STATES = frozenset({"queued", "running"})
PUBLIC_STATES = frozenset({"queued", "running", "completed", "failed"})
RESUMABLE_PHASES = frozenset(
    {"canceled", "stale", "worker_timeout", "worker_terminated"}
)
CALL_REGISTRATION_GRACE_SECONDS = 30.0
CALL_KEY_PREFIX = "_call:"
RESUME_KEY_PREFIX = "_resume:"


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
    now = _now()
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
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "heartbeat_at": now,
        "attempt": 1,
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
    if status.get("state") not in PUBLIC_STATES:
        raise ModalMapContractError("Statut Modal avec état invalide")
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


def _attempt(status: dict[str, Any]) -> int:
    raw = status.get("attempt", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ModalMapContractError("Statut Modal avec tentative invalide")
    return raw


def _call_key(job_id: str, attempt: int) -> str:
    return f"{CALL_KEY_PREFIX}{job_id}:{attempt}"


def _resume_key(job_id: str, attempt: int) -> str:
    return f"{RESUME_KEY_PREFIX}{job_id}:{attempt}"


def _timestamp_age_seconds(status: dict[str, Any]) -> float:
    raw = (
        status.get("heartbeat_at")
        or status.get("started_at")
        or status.get("created_at")
    )
    if not isinstance(raw, str):
        return float("inf")
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if stamp.tzinfo is None:
        return float("inf")
    return max(0.0, (datetime.now(UTC) - stamp.astimezone(UTC)).total_seconds())


def _terminal_transition(
    status: dict[str, Any],
    *,
    phase: str,
    message: str,
    error: str,
) -> dict[str, Any]:
    terminal = dict(status)
    terminal.update(
        state="failed",
        phase=phase,
        message=message,
        error=error,
        finished_at=_now(),
        heartbeat_at=_now(),
    )
    return terminal


def _poll_function_call(function_call_id: str) -> None:
    modal.FunctionCall.from_id(function_call_id).get(timeout=0)


def _looks_canceled(error: BaseException) -> bool:
    message = str(error).strip().lower()
    return isinstance(error, modal.exception.RemoteError) and (
        not message or "cancel" in message or "terminat" in message
    )


def _reconcile_active_status(status: dict[str, Any]) -> dict[str, Any]:
    """Turn a terminal Modal call into a terminal public status on the next poll."""

    if status.get("state") not in ACTIVE_STATES:
        return status
    job_id = str(status.get("job_id", ""))
    attempt = _attempt(status)
    call = job_store.get(_call_key(job_id, attempt))
    if not isinstance(call, dict) or not isinstance(call.get("function_call_id"), str):
        if _timestamp_age_seconds(status) < CALL_REGISTRATION_GRACE_SECONDS:
            return status
        stale = _terminal_transition(
            status,
            phase="stale",
            message="Production interrompue — reprise disponible",
            error="L'appel Modal actif est absent; les tuiles validées restent réutilisables.",
        )
        job_store[job_id] = stale
        return stale

    try:
        _poll_function_call(str(call["function_call_id"]))
    except modal.exception.OutputExpiredError:
        phase = "stale"
        error_message = (
            "Le résultat de l'appel Modal a expiré; les tuiles validées restent "
            "réutilisables."
        )
    except modal.exception.FunctionTimeoutError as error:
        phase = "worker_timeout"
        error_message = str(error) or "Le worker Modal a dépassé sa durée maximale."
    except modal.exception.TimeoutError:
        return dict(job_store.get(job_id, status))
    except Exception as error:  # Modal reports external cancellation as RemoteError.
        latest = dict(job_store.get(job_id, status))
        if latest.get("state") not in ACTIVE_STATES:
            return latest
        phase = "canceled" if _looks_canceled(error) else "worker_terminated"
        error_message = str(error) or (
            "L'appel Modal a été annulé."
            if phase == "canceled"
            else "Le worker Modal s'est arrêté sans statut final."
        )
    else:
        latest = dict(job_store.get(job_id, status))
        if latest.get("state") not in ACTIVE_STATES:
            return latest
        phase = "stale"
        error_message = "Le worker Modal s'est terminé sans publier de statut final."

    latest = dict(job_store.get(job_id, status))
    if latest.get("state") not in ACTIVE_STATES or _attempt(latest) != attempt:
        return latest
    terminal = _terminal_transition(
        latest,
        phase=phase,
        message=(
            "Production annulée — reprise disponible"
            if phase == "canceled"
            else "Production interrompue — reprise disponible"
        ),
        error=error_message,
    )
    job_store[job_id] = terminal
    return terminal


def _dispatch(job_id: str, request: dict[str, Any], attempt: int) -> None:
    try:
        call = produce_map.spawn(job_id, request, attempt)
    except Exception as error:
        current = dict(job_store[job_id])
        if _attempt(current) == attempt and current.get("state") in ACTIVE_STATES:
            job_store[job_id] = _terminal_transition(
                current,
                phase="dispatch_failed",
                message="Échec du lancement Modal",
                error=str(error),
            )
        return
    job_store[_call_key(job_id, attempt)] = {
        "function_call_id": call.object_id,
        "created_at": _now(),
    }


def _resume_if_interrupted(
    status: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    if status.get("state") != "failed" or status.get("phase") not in RESUMABLE_PHASES:
        return status
    next_attempt = _attempt(status) + 1
    lease = job_store.put(
        _resume_key(str(status["job_id"]), next_attempt),
        {"created_at": _now()},
        skip_if_exists=True,
    )
    if not lease:
        return dict(job_store.get(str(status["job_id"]), status))
    current = dict(job_store.get(str(status["job_id"]), status))
    if current.get("state") != "failed" or current.get("phase") not in RESUMABLE_PHASES:
        return current
    resumed = dict(current)
    resumed.update(
        request=request,
        state="queued",
        phase="resume_queued",
        message="Reprise demandée — revalidation des tuiles déjà produites",
        attempt=next_attempt,
        started_at=None,
        finished_at=None,
        heartbeat_at=_now(),
        error=None,
    )
    job_store[str(status["job_id"])] = resumed
    _dispatch(str(status["job_id"]), request, next_attempt)
    return dict(job_store[str(status["job_id"])])


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
            "FIREVIEWER_TILE_WORKERS": "4",
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
def produce_map(job_id: str, request: dict[str, Any], attempt: int) -> None:
    from simple_production_engine import ProductionConfig, ProductionEngine

    work_volume.reload()
    stored = job_store.get(job_id)
    if not isinstance(stored, dict) or _attempt(stored) != attempt:
        return
    status = dict(stored)
    status.update(
        state="running",
        phase="starting",
        started_at=_now(),
        heartbeat_at=_now(),
        finished_at=None,
        error=None,
    )
    job_store[job_id] = status
    volume_commit_lock = threading.Lock()
    try:
        engine = ProductionEngine(ProductionConfig.from_environment())

        def progress(fraction: float, message: str) -> None:
            # The engine only emits this message after an atomic tile package has
            # been revalidated and its raw sources have been removed. Persist that
            # checkpoint before advertising it, so a killed worker can reuse it.
            if "package validé, sources brutes absentes" in str(message):
                with volume_commit_lock:
                    work_volume.commit()
            current = dict(job_store[job_id])
            if _attempt(current) != attempt:
                raise ModalMapContractError("La tentative Modal a été remplacée")
            match = re.search(r"Tuile\s+(\d+)/(\d+)", message, re.IGNORECASE)
            current.update(
                state="running",
                phase="in_progress",
                progress=max(
                    float(current.get("progress", 0.0)),
                    max(0.0, min(0.999, float(fraction))),
                ),
                message=str(message),
                current_tile=int(match.group(1)) if match else current["current_tile"],
                tile_count=int(match.group(2)) if match else current["tile_count"],
                heartbeat_at=_now(),
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
            heartbeat_at=_now(),
        )
        job_store[job_id] = status
        work_volume.commit()
    except modal.exception.InputCancellation as error:
        canceled = dict(job_store.get(job_id, status))
        if _attempt(canceled) == attempt and canceled.get("state") in ACTIVE_STATES:
            canceled = _terminal_transition(
                canceled,
                phase="canceled",
                message="Production annulée — reprise disponible",
                error=str(error) or "L'appel Modal a été annulé.",
            )
            job_store[job_id] = canceled
        # Preserve every atomic tile that reached disk before cancellation.
        work_volume.commit()
        raise
    except Exception as error:
        failed = dict(job_store.get(job_id, status))
        if _attempt(failed) == attempt and failed.get("state") in ACTIVE_STATES:
            job_store[job_id] = _terminal_transition(
                failed,
                phase="failed",
                message="Échec de production",
                error=str(error),
            )
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
            _dispatch(job_id, normalized, 1)
        stored = dict(job_store[job_id])
        if stored.get("request_sha256") != request_sha256(normalized):
            raise HTTPException(status_code=409, detail="Identité de job incohérente")
        stored = _reconcile_active_status(stored)
        stored = _resume_if_interrupted(stored, normalized)
        return public_job_status(stored)

    @web.get("/v1/map-jobs/{job_id}")
    def status(
        job_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _authorize(authorization)
        stored = job_store.get(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Job introuvable")
        return public_job_status(_reconcile_active_status(dict(stored)))

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
