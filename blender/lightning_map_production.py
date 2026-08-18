"""Lightning Batch Job entrypoint for complete FireViewer map production.

The complete reproduction ZIP is published only to the private Hugging Face
dataset. A separately validated complete viewer GLB is then added to the same
immutable dataset root. The final callback uses the existing result-v1 contract,
so no multi-gigabyte ZIP is copied through Vercel Blob.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from export_complete_viewer_glb import GLB_NAME, RECEIPT_NAME, SCHEMA as VIEWER_SCHEMA, STATUS as VIEWER_STATUS
from fixed_asset_placement import EMPTY_REQUEST, normalize_request as normalize_fixed_assets, request_sha256 as fixed_assets_request_sha256
from runpod_map_production import normalize_map_request, request_sha256
from simple_production_engine import ProductionConfig, ProductionEngine, plan_zone

REQUEST_SCHEMA = "fireviewer.map-production-request.v1"
PROGRESS_SCHEMA = "fireviewer.map-production-progress.v1"
RESULT_SCHEMA = "fireviewer.map-production-result.v1"
PROGRESS_MIN_INTERVAL_SECONDS = 10.0


class LightningMapContractError(ValueError):
    pass


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise LightningMapContractError(f"Variable obligatoire absente: {name}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CallbackClient:
    def __init__(self) -> None:
        self.job_id = _required_environment("FIREVIEWER_MAP_JOB_ID")
        self.request_url = _required_environment("FIREVIEWER_MAP_REQUEST_URL")
        self.progress_url = _required_environment("FIREVIEWER_MAP_PROGRESS_URL")
        self.result_url = _required_environment("FIREVIEWER_MAP_RESULT_URL")
        self.token = _required_environment("FIREVIEWER_MAP_CALLBACK_TOKEN")
        self.sequence = 0
        self._last_sent_at = 0.0
        self._progress_lock = threading.Lock()
        self._client = httpx.Client(headers={"X-FireViewer-Map-Token": self.token}, timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0), follow_redirects=False, trust_env=False)

    def _request(self, method: str, url: str, *, payload: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client.request(method, url, json=payload)
                response.raise_for_status()
                return response.json() if response.content else None
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt < 3:
                    time.sleep(2**attempt)
        raise LightningMapContractError("Callback FireViewer inaccessible") from last_error

    def fetch_request(self) -> dict[str, Any]:
        envelope = self._request("GET", self.request_url)
        if not isinstance(envelope, Mapping) or envelope.get("schema") != REQUEST_SCHEMA or envelope.get("job_id") != self.job_id or not isinstance(envelope.get("request"), Mapping):
            raise LightningMapContractError("Requête Lightning invalide")
        request = normalize_map_request(envelope["request"])
        if envelope.get("request_sha256") != request_sha256(request):
            raise LightningMapContractError("Hash de requête Lightning divergent")
        return request

    def progress(self, fraction: float, message: str, *, phase: str, current_tile: int | None, tile_count: int | None, state: str = "running", error: str | None = None, force: bool = False) -> None:
        now = time.monotonic()
        bounded = max(0.0, min(1.0, float(fraction)))
        with self._progress_lock:
            if not force and now - self._last_sent_at < PROGRESS_MIN_INTERVAL_SECONDS:
                return
            payload = {"schema": PROGRESS_SCHEMA, "job_id": self.job_id, "sequence": self.sequence, "state": state, "phase": phase, "progress": bounded, "message": str(message), "current_tile": current_tile, "tile_count": tile_count, "error": error}
            self.sequence += 1
            self._last_sent_at = now
        self._request("POST", self.progress_url, payload=payload)

    def result(self, payload: dict[str, Any]) -> None:
        self._request("POST", self.result_url, payload=payload)


def _status_snapshot(config: ProductionConfig, request: dict[str, Any], fixed: Mapping[str, Any]) -> dict[str, Any]:
    base = plan_zone(request["latitude"], request["longitude"], request["side_km"], max_side_m=config.max_side_m, max_tiles=config.max_tiles)
    zone_id = base.zone_id
    if fixed.get("placements"):
        zone_id = f"{zone_id}-fixed-{fixed_assets_request_sha256(fixed)[:12]}"
    path = config.work_root.resolve() / "jobs" / zone_id / "job-status.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _run_blender_script(config: ProductionConfig, job_root: Path, script_name: str) -> None:
    script = Path(__file__).with_name(script_name).resolve()
    command = (str(config.blender), "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1", "--python", str(script), "--", "--job-root", str(job_root))
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30 * 60, check=False)
    if result.returncode != 0:
        raise LightningMapContractError(f"{script_name} a échoué:\n" + (result.stdout + "\n" + result.stderr)[-4000:])


def _export_viewer(config: ProductionConfig, job_root: Path) -> dict[str, Any]:
    _run_blender_script(config, job_root, "export_complete_viewer_glb.py")
    _run_blender_script(config, job_root, "validate_complete_viewer_meshes.py")
    receipt_path = job_root / RECEIPT_NAME
    glb_path = job_root / GLB_NAME
    if not receipt_path.is_file() or not glb_path.is_file():
        raise LightningMapContractError("Artefacts viewer complets absents")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    completeness = receipt.get("completeness") if isinstance(receipt, dict) else None
    if not isinstance(receipt, dict) or receipt.get("schema") != VIEWER_SCHEMA or receipt.get("status") != VIEWER_STATUS or receipt.get("viewer", {}).get("sha256") != _sha256_file(glb_path) or not isinstance(completeness, Mapping) or completeness.get("mesh_coverage") != "complete":
        raise LightningMapContractError("Reçu de complétude viewer invalide")
    return receipt


def _publish_viewer(job_root: Path, publication: Mapping[str, Any], viewer: Mapping[str, Any]) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi
    dataset_id = publication.get("dataset_id")
    remote_root = publication.get("path_in_repo")
    if not isinstance(dataset_id, str) or not dataset_id or not isinstance(remote_root, str) or not remote_root:
        raise LightningMapContractError("Publication HF de base invalide")
    token = _required_environment("HF_TOKEN")
    operations = [
        CommitOperationAdd(path_in_repo=f"{remote_root}/{GLB_NAME}", path_or_fileobj=str(job_root / GLB_NAME)),
        CommitOperationAdd(path_in_repo=f"{remote_root}/{RECEIPT_NAME}", path_or_fileobj=str(job_root / RECEIPT_NAME)),
    ]
    commit = HfApi(token=token).create_commit(repo_id=dataset_id, repo_type="dataset", operations=operations, commit_message=f"Add complete viewer scene for {viewer['zone_id']}")
    oid = getattr(commit, "oid", None)
    if not isinstance(oid, str) or not oid:
        raise LightningMapContractError("Révision HF viewer absente")
    return oid


def _cleanup_local_result(config: ProductionConfig, job_root: Path) -> None:
    if config.scratch_root is None:
        return
    scratch = config.scratch_root.resolve(strict=True)
    root = job_root.resolve(strict=True)
    try:
        relative = root.relative_to(scratch)
    except ValueError:
        return
    if len(relative.parts) != 2 or relative.parts[0] != "jobs":
        raise LightningMapContractError("Le résultat local sort du staging de job")
    shutil.rmtree(root)


def run() -> dict[str, Any]:
    callback = CallbackClient()
    request: dict[str, Any] | None = None
    last_fraction = 0.0
    last_message = "Initialisation Lightning"
    try:
        request = callback.fetch_request()
        config = ProductionConfig.from_environment()
        engine = ProductionEngine(config)
        fixed = normalize_fixed_assets(request["fixed_asset_placements"] or dict(EMPTY_REQUEST), engine.asset_library_payload)

        def report(fraction: float, message: str) -> None:
            nonlocal last_fraction, last_message
            last_fraction, last_message = fraction, message
            snapshot = _status_snapshot(config, request or {}, fixed)
            callback.progress(fraction, message, phase=str(snapshot.get("phase") or "production"), current_tile=int(snapshot["completed_tiles"]) if isinstance(snapshot.get("completed_tiles"), int) else None, tile_count=int(snapshot["total_tiles"]) if isinstance(snapshot.get("total_tiles"), int) else None)

        callback.progress(0.0, "Job Lightning démarré", phase="starting", current_tile=0, tile_count=None, force=True)
        archive_path: Path | None = None
        final_message = "Production terminée"
        gallery: list[tuple[str, str]] = []
        for message, archive, items in engine.run(request["latitude"], request["longitude"], request["side_km"], progress_callback=report, archive_ready_callback=None, fixed_asset_placements=fixed):
            final_message = message
            if archive:
                archive_path = Path(archive)
            if items:
                gallery = items
        if archive_path is None or gallery:
            raise LightningMapContractError("Le moteur n'a pas publié le pack complet")
        job_root = archive_path.parent
        publication_path = job_root / "dataset-publication.json"
        zone_receipt_path = job_root / "zone.done.json"
        if not publication_path.is_file() or not zone_receipt_path.is_file():
            raise LightningMapContractError("Reçus finaux du pack absents")
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        zone_receipt = json.loads(zone_receipt_path.read_text(encoding="utf-8"))
        if not isinstance(publication, dict) or publication.get("status") != "published_private":
            raise LightningMapContractError("Le ZIP complet doit être publié sur Hugging Face avant le viewer")
        callback.progress(0.997, "Export de la map 3D complète pour le viewer", phase="viewer_export", current_tile=int(zone_receipt["tile_count"]), tile_count=int(zone_receipt["tile_count"]), force=True)
        viewer = _export_viewer(config, job_root)
        callback.progress(0.999, "Publication de la map viewer complète sur Hugging Face", phase="viewer_publication", current_tile=int(zone_receipt["tile_count"]), tile_count=int(zone_receipt["tile_count"]), force=True)
        revision = _publish_viewer(job_root, publication, viewer)
        remote_root = str(publication["path_in_repo"])
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": callback.job_id,
            "status": "technical_scene_produced",
            "request_sha256": request_sha256(request),
            "zone_id": zone_receipt["zone_id"],
            "build_id": zone_receipt["build_id"],
            "message": final_message + " Viewer 3D complet validé.",
            "tile_count": zone_receipt["tile_count"],
            "degraded_mns_tile_count": zone_receipt.get("degraded_mns_tile_count", 0),
            "dataset": {"repo_id": publication["dataset_id"], "revision": revision, "root": remote_root},
            "archive": {"path": f"{remote_root}/fireviewer-zone.zip", "sha256": _sha256_file(archive_path)},
            "viewer": {"path": f"{remote_root}/{GLB_NAME}", "receipt_path": f"{remote_root}/{RECEIPT_NAME}", "sha256": viewer["viewer"]["sha256"], "byte_count": viewer["viewer"]["byte_count"], "completeness": viewer["completeness"]},
            "captures": [],
        }
        callback.result(result)
        callback.progress(1.0, str(result["message"]), phase="completed", current_tile=int(result["tile_count"]), tile_count=int(result["tile_count"]), state="completed", force=True)
        _cleanup_local_result(config, job_root)
        return result
    except Exception as error:
        callback.progress(last_fraction, last_message if last_message != "Initialisation Lightning" else str(error), phase="failed", current_tile=None, tile_count=None, state="failed", error=str(error), force=True)
        raise


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
