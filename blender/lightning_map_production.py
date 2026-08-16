"""Lightning Batch Job entrypoint for FireViewer map production.

The job uses the embedded immutable assets, a mounted data connection for tile
checkpoints and local scratch for assembly. Only progress metadata is sent back
to the admin backend. The standalone ZIP is uploaded once to private admin
storage before the secondary Hugging Face publication is attempted.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from fixed_asset_placement import (
    EMPTY_REQUEST,
    normalize_request as normalize_fixed_assets,
    request_sha256 as fixed_assets_request_sha256,
)
from runpod_map_production import normalize_map_request, request_sha256
from simple_production_engine import ProductionConfig, ProductionEngine, plan_zone

REQUEST_SCHEMA = "fireviewer.map-production-request.v1"
PROGRESS_SCHEMA = "fireviewer.map-production-progress.v1"
RESULT_SCHEMA = "fireviewer.map-production-result.v2"
ARCHIVE_UPLOAD_SCHEMA = "fireviewer.map-archive-upload.v1"
PROGRESS_MIN_INTERVAL_SECONDS = 10.0


class LightningMapContractError(ValueError):
    """The Lightning job environment or callback payload is invalid."""


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise LightningMapContractError(f"Variable obligatoire absente: {name}")
    return value


class CallbackClient:
    def __init__(self) -> None:
        self.job_id = _required_environment("FIREVIEWER_MAP_JOB_ID")
        self.request_url = _required_environment("FIREVIEWER_MAP_REQUEST_URL")
        self.progress_url = _required_environment("FIREVIEWER_MAP_PROGRESS_URL")
        self.result_url = _required_environment("FIREVIEWER_MAP_RESULT_URL")
        self.archive_token_url = _required_environment(
            "FIREVIEWER_MAP_ARCHIVE_TOKEN_URL"
        )
        self.token = _required_environment("FIREVIEWER_MAP_CALLBACK_TOKEN")
        self.sequence = 0
        self._last_sent_at = 0.0
        self._last_progress = -1.0
        self._progress_lock = threading.Lock()
        self.archive_delivery: dict[str, Any] | None = None
        self._client = httpx.Client(
            headers={"X-FireViewer-Map-Token": self.token},
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
            follow_redirects=False,
            trust_env=False,
        )

    def _request(
        self, method: str, url: str, *, payload: dict[str, Any] | None = None
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client.request(method, url, json=payload)
                response.raise_for_status()
                return response.json() if response.content else None
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2**attempt)
        raise LightningMapContractError(
            "Callback FireViewer inaccessible"
        ) from last_error

    def fetch_request(self) -> dict[str, Any]:
        envelope = self._request("GET", self.request_url)
        if (
            not isinstance(envelope, Mapping)
            or envelope.get("schema") != REQUEST_SCHEMA
            or envelope.get("job_id") != self.job_id
            or not isinstance(envelope.get("request"), Mapping)
        ):
            raise LightningMapContractError("Requête Lightning invalide")
        request = normalize_map_request(envelope["request"])
        if envelope.get("request_sha256") != request_sha256(request):
            raise LightningMapContractError("Hash de requête Lightning divergent")
        return request

    def progress(
        self,
        fraction: float,
        message: str,
        *,
        phase: str,
        current_tile: int | None,
        tile_count: int | None,
        state: str = "running",
        error: str | None = None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        bounded = max(0.0, min(1.0, float(fraction)))
        with self._progress_lock:
            if not force and now - self._last_sent_at < PROGRESS_MIN_INTERVAL_SECONDS:
                return
            payload = {
                "schema": PROGRESS_SCHEMA,
                "job_id": self.job_id,
                "sequence": self.sequence,
                "state": state,
                "phase": phase,
                "progress": bounded,
                "message": str(message),
                "current_tile": current_tile,
                "tile_count": tile_count,
                "error": error,
            }
            self.sequence += 1
            self._last_sent_at = now
            self._last_progress = max(self._last_progress, bounded)
        self._request("POST", self.progress_url, payload=payload)

    def result(self, payload: dict[str, Any]) -> None:
        self._request("POST", self.result_url, payload=payload)

    def upload_archive(self, archive: Path, byte_count: int, sha256: str) -> None:
        if archive.stat().st_size != byte_count:
            raise LightningMapContractError(
                "La taille du ZIP a changé avant son upload"
            )
        last_error: BaseException | None = None
        for attempt in range(3):
            try:
                grant = self._request(
                    "POST",
                    self.archive_token_url,
                    payload={"byte_count": byte_count, "sha256": sha256},
                )
                if (
                    not isinstance(grant, Mapping)
                    or grant.get("schema") != ARCHIVE_UPLOAD_SCHEMA
                    or not isinstance(grant.get("pathname"), str)
                    or not isinstance(grant.get("upload_required"), bool)
                ):
                    raise LightningMapContractError("Autorisation de ZIP invalide")
                pathname = str(grant["pathname"])
                if grant["upload_required"]:
                    from vercel.blob import BlobClient

                    client_token = grant.get("client_token")
                    if not isinstance(client_token, str) or not client_token:
                        raise LightningMapContractError("Jeton d'upload ZIP absent")
                    result = BlobClient(token=client_token).upload_file(
                        archive,
                        pathname,
                        access="private",
                        content_type="application/zip",
                        add_random_suffix=False,
                        overwrite=False,
                        cache_control_max_age=31_536_000,
                        multipart=True,
                    )
                    if result.pathname != pathname:
                        raise LightningMapContractError(
                            "Vercel Blob a retourné un chemin ZIP inattendu"
                        )
                self.archive_delivery = {
                    "provider": "vercel_blob_private",
                    "pathname": pathname,
                    "byte_count": byte_count,
                    "sha256": sha256,
                }
                return
            except BaseException as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2**attempt)
        raise LightningMapContractError(
            "Upload du ZIP privé vers l'administration échoué"
        ) from last_error


def _status_snapshot(
    config: ProductionConfig,
    request: dict[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    work = config.work_root.resolve()
    base_plan = plan_zone(
        request["latitude"],
        request["longitude"],
        request["side_km"],
        max_side_m=config.max_side_m,
        max_tiles=config.max_tiles,
    )
    zone_id = base_plan.zone_id
    if fixed.get("placements"):
        zone_id = f"{zone_id}-fixed-{fixed_assets_request_sha256(fixed)[:12]}"
    path = work / "jobs" / zone_id / "job-status.json"
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _result_payload(
    *,
    callback: CallbackClient,
    request: dict[str, Any],
    archive_path: Path,
    final_message: str,
) -> dict[str, Any]:
    job_root = archive_path.parent
    publication_path = job_root / "dataset-publication.json"
    receipt_path = job_root / "zone.done.json"
    if not publication_path.is_file() or not receipt_path.is_file():
        raise LightningMapContractError("Les reçus finaux du pack sont absents")
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if publication.get("captures") != []:
        raise LightningMapContractError("La publication contient une galerie obsolète")
    delivery = callback.archive_delivery
    if not isinstance(delivery, dict):
        raise LightningMapContractError("Le ZIP privé n'a pas été remis au backend")
    publication_status = publication.get("status")
    if publication_status not in {"published_private", "failed_pending_retry"}:
        raise LightningMapContractError("Le reçu Hugging Face est invalide")
    dataset: dict[str, Any] = {
        "repo_id": publication["dataset_id"],
        "status": publication_status,
    }
    if publication_status == "published_private":
        dataset.update(
            {
                "revision": publication["commit_oid"],
                "root": publication["path_in_repo"],
            }
        )
    return {
        "schema": RESULT_SCHEMA,
        "job_id": callback.job_id,
        "status": "technical_scene_produced",
        "request_sha256": request_sha256(request),
        "zone_id": receipt["zone_id"],
        "build_id": receipt["build_id"],
        "message": final_message,
        "tile_count": receipt["tile_count"],
        "degraded_mns_tile_count": receipt.get("degraded_mns_tile_count", 0),
        "dataset": dataset,
        "archive": dict(delivery),
        "captures": [],
    }


def _cleanup_local_result(config: ProductionConfig, archive_path: Path) -> None:
    if config.scratch_root is None:
        return
    scratch = config.scratch_root.resolve(strict=True)
    result_root = archive_path.parent.resolve(strict=True)
    try:
        relative = result_root.relative_to(scratch)
    except ValueError:
        return
    if len(relative.parts) != 2 or relative.parts[0] != "jobs":
        raise LightningMapContractError("Le résultat local sort du staging de job")
    shutil.rmtree(result_root)


def run() -> dict[str, Any]:
    callback = CallbackClient()
    request: dict[str, Any] | None = None
    last_fraction = 0.0
    last_message = "Initialisation Lightning"
    try:
        request = callback.fetch_request()
        config = ProductionConfig.from_environment()
        engine = ProductionEngine(config)
        fixed = normalize_fixed_assets(
            request["fixed_asset_placements"] or dict(EMPTY_REQUEST),
            engine.asset_library_payload,
        )

        def report(fraction: float, message: str) -> None:
            nonlocal last_fraction, last_message
            last_fraction = fraction
            last_message = message
            snapshot = _status_snapshot(config, request or {}, fixed)
            callback.progress(
                fraction,
                message,
                phase=str(snapshot.get("phase") or "production"),
                current_tile=(
                    int(snapshot["completed_tiles"])
                    if isinstance(snapshot.get("completed_tiles"), int)
                    else None
                ),
                tile_count=(
                    int(snapshot["total_tiles"])
                    if isinstance(snapshot.get("total_tiles"), int)
                    else None
                ),
            )

        callback.progress(
            0.0,
            "Job Lightning démarré",
            phase="starting",
            current_tile=0,
            tile_count=None,
            force=True,
        )
        archive_path: Path | None = None
        gallery: list[tuple[str, str]] = []
        final_message = "Production terminée"
        for message, archive, items in engine.run(
            request["latitude"],
            request["longitude"],
            request["side_km"],
            progress_callback=report,
            archive_ready_callback=callback.upload_archive,
            fixed_asset_placements=fixed,
        ):
            final_message = message
            if archive:
                archive_path = Path(archive)
            if items:
                gallery = items
        if archive_path is None or gallery:
            raise LightningMapContractError("Le moteur n'a pas publié le pack complet")
        result = _result_payload(
            callback=callback,
            request=request,
            archive_path=archive_path,
            final_message=final_message,
        )
        callback.result(result)
        callback.progress(
            1.0,
            final_message,
            phase="completed",
            current_tile=int(result["tile_count"]),
            tile_count=int(result["tile_count"]),
            state="completed",
            force=True,
        )
        _cleanup_local_result(config, archive_path)
        return result
    except Exception as exc:
        callback.progress(
            last_fraction,
            last_message if last_message != "Initialisation Lightning" else str(exc),
            phase="failed",
            current_tile=None,
            tile_count=None,
            state="failed",
            error=str(exc),
            force=True,
        )
        raise


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["CallbackClient", "LightningMapContractError", "main", "run"]
