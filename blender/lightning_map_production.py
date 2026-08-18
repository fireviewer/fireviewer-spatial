"""Lightning Batch Job entrypoint for complete FireViewer map production.

The complete reproduction ZIP and the separately validated complete viewer GLB
are published to the private Hugging Face dataset with the resumable Xet-backed
folder uploader. The final callback uses the result-v1 contract only after both
remote publications have been verified.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from export_complete_viewer_glb import GLB_NAME, RECEIPT_NAME, SCHEMA as VIEWER_SCHEMA, STATUS as VIEWER_STATUS
from fixed_asset_placement import EMPTY_REQUEST, normalize_request as normalize_fixed_assets, request_sha256 as fixed_assets_request_sha256
from runpod_map_production import normalize_map_request, request_sha256
from simple_production_engine import (
    DATASET_ENTRY_NAME,
    DATASET_PUBLICATION_NAME,
    ZIP_NAME,
    ZONE_RECEIPT_NAME,
    ProductionConfig,
    ProductionEngine,
    plan_zone,
)

REQUEST_SCHEMA = "fireviewer.map-production-request.v1"
PROGRESS_SCHEMA = "fireviewer.map-production-progress.v1"
RESULT_SCHEMA = "fireviewer.map-production-result.v1"
DATASET_PUBLICATION_SCHEMA = "fireviewer.simple-measured-scene-dataset-publication.v1"
PROGRESS_MIN_INTERVAL_SECONDS = 10.0
HF_UPLOAD_MAX_ATTEMPTS = 3
HF_UPLOAD_BACKOFF_SECONDS = (5.0, 20.0)


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LightningMapContractError(f"{label} JSON invalide") from error
    if not isinstance(value, dict):
        raise LightningMapContractError(f"{label} invalide")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_authentication_error(error: BaseException) -> bool:
    message = str(error).casefold()
    return any(
        fragment in message
        for fragment in (
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "invalid token",
            "authentication",
            "permission",
        )
    )


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
        self._client = httpx.Client(
            headers={"X-FireViewer-Map-Token": self.token},
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
            follow_redirects=False,
            trust_env=False,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any:
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
        self._request("POST", self.progress_url, payload=payload)

    def result(self, payload: dict[str, Any]) -> None:
        self._request("POST", self.result_url, payload=payload)


def _status_snapshot(
    config: ProductionConfig,
    request: dict[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    base = plan_zone(
        request["latitude"],
        request["longitude"],
        request["side_km"],
        max_side_m=config.max_side_m,
        max_tiles=config.max_tiles,
    )
    zone_id = base.zone_id
    if fixed.get("placements"):
        zone_id = f"{zone_id}-fixed-{fixed_assets_request_sha256(fixed)[:12]}"
    path = config.work_root.resolve() / "jobs" / zone_id / "job-status.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _run_blender_script(
    config: ProductionConfig,
    job_root: Path,
    script_name: str,
) -> None:
    script = Path(__file__).with_name(script_name).resolve()
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
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30 * 60,
        check=False,
    )
    if result.returncode != 0:
        raise LightningMapContractError(
            f"{script_name} a échoué:\n"
            + (result.stdout + "\n" + result.stderr)[-4000:]
        )


def _export_viewer(
    config: ProductionConfig,
    job_root: Path,
) -> dict[str, Any]:
    _run_blender_script(config, job_root, "export_complete_viewer_glb.py")
    _run_blender_script(config, job_root, "validate_complete_viewer_meshes.py")
    receipt_path = job_root / RECEIPT_NAME
    glb_path = job_root / GLB_NAME
    if not receipt_path.is_file() or not glb_path.is_file():
        raise LightningMapContractError("Artefacts viewer complets absents")
    receipt = _load_json(receipt_path, "Reçu viewer")
    completeness = receipt.get("completeness")
    viewer = receipt.get("viewer")
    if (
        receipt.get("schema") != VIEWER_SCHEMA
        or receipt.get("status") != VIEWER_STATUS
        or not isinstance(viewer, Mapping)
        or viewer.get("sha256") != _sha256_file(glb_path)
        or not isinstance(completeness, Mapping)
        or completeness.get("mesh_coverage") != "complete"
    ):
        raise LightningMapContractError("Reçu de complétude viewer invalide")
    return receipt


def _verify_remote_files(
    api: Any,
    *,
    dataset_id: str,
    revision: str,
    remote_root: str,
    file_names: Sequence[str],
) -> None:
    missing: list[str] = []
    for file_name in file_names:
        remote_path = f"{remote_root}/{file_name}"
        available = False
        for attempt in range(5):
            if api.file_exists(
                repo_id=dataset_id,
                filename=remote_path,
                repo_type="dataset",
                revision=revision,
            ):
                available = True
                break
            if attempt < 4:
                time.sleep(2.0)
        if not available:
            missing.append(remote_path)
    if missing:
        raise LightningMapContractError(
            "Fichiers Hugging Face absents après commit: " + ", ".join(missing)
        )


def _upload_hf_folder(
    job_root: Path,
    *,
    dataset_id: str,
    remote_root: str,
    file_names: Sequence[str],
    commit_message: str,
) -> str:
    from huggingface_hub import HfApi

    token = _required_environment("HF_TOKEN")
    last_error: Exception | None = None
    for attempt in range(1, HF_UPLOAD_MAX_ATTEMPTS + 1):
        try:
            api = HfApi(token=token)
            info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
            if info.private is not True:
                raise LightningMapContractError(
                    "La dataset cible doit être privée avant toute publication"
                )
            commit = api.upload_folder(
                repo_id=dataset_id,
                repo_type="dataset",
                folder_path=str(job_root),
                path_in_repo=remote_root,
                allow_patterns=list(file_names),
                commit_message=commit_message,
            )
            oid = getattr(commit, "oid", None)
            if not isinstance(oid, str) or not oid:
                raise LightningMapContractError("Révision Hugging Face absente")
            _verify_remote_files(
                api,
                dataset_id=dataset_id,
                revision=oid,
                remote_root=remote_root,
                file_names=file_names,
            )
            return oid
        except Exception as error:
            last_error = error
            if (
                attempt >= HF_UPLOAD_MAX_ATTEMPTS
                or _is_authentication_error(error)
            ):
                break
            time.sleep(HF_UPLOAD_BACKOFF_SECONDS[attempt - 1])
    raise LightningMapContractError(
        f"Publication Hugging Face/Xet échouée après {HF_UPLOAD_MAX_ATTEMPTS} tentatives"
    ) from last_error


def _publish_archive(
    config: ProductionConfig,
    *,
    job_root: Path,
    archive_path: Path,
    zone_receipt: Mapping[str, Any],
    dataset_entry: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_id = config.dataset_id
    zone_id = zone_receipt.get("zone_id")
    build_id = zone_receipt.get("build_id")
    entry_sha256 = dataset_entry.get("entry_sha256")
    archive_sha256 = _sha256_file(archive_path)
    archive_record = dataset_entry.get("archive")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_entry.get("dataset_id") != dataset_id
        or not isinstance(zone_id, str)
        or not zone_id
        or dataset_entry.get("zone_id") != zone_id
        or not _is_sha256(build_id)
        or dataset_entry.get("build_id") != build_id
        or not _is_sha256(entry_sha256)
        or not isinstance(archive_record, Mapping)
        or archive_record.get("file") != ZIP_NAME
        or archive_record.get("sha256") != archive_sha256
        or archive_record.get("byte_count") != archive_path.stat().st_size
    ):
        raise LightningMapContractError(
            "Métadonnées du pack incohérentes avant publication Hugging Face"
        )

    remote_root = f"zones/{zone_id}/{build_id}"
    revision = _upload_hf_folder(
        job_root,
        dataset_id=dataset_id,
        remote_root=remote_root,
        file_names=(ZIP_NAME, ZONE_RECEIPT_NAME, DATASET_ENTRY_NAME),
        commit_message=f"Add measured FireViewer scene {zone_id}",
    )
    publication = {
        "schema": DATASET_PUBLICATION_SCHEMA,
        "status": "published_private",
        "dataset_id": dataset_id,
        "zone_id": zone_id,
        "build_id": build_id,
        "archive_sha256": archive_sha256,
        "entry_sha256": entry_sha256,
        "commit_oid": revision,
        "path_in_repo": remote_root,
        "captures": [],
        "transport": "huggingface_hub.upload_folder+xet",
    }
    _write_json(job_root / DATASET_PUBLICATION_NAME, publication)
    return publication


def _publish_viewer(
    job_root: Path,
    publication: Mapping[str, Any],
    viewer: Mapping[str, Any],
) -> str:
    dataset_id = publication.get("dataset_id")
    remote_root = publication.get("path_in_repo")
    zone_id = viewer.get("zone_id")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(remote_root, str)
        or not remote_root
        or not isinstance(zone_id, str)
        or not zone_id
    ):
        raise LightningMapContractError("Publication HF de base invalide")
    return _upload_hf_folder(
        job_root,
        dataset_id=dataset_id,
        remote_root=remote_root,
        file_names=(GLB_NAME, RECEIPT_NAME),
        commit_message=f"Add complete viewer scene for {zone_id}",
    )


def _cleanup_local_result(
    config: ProductionConfig,
    job_root: Path,
) -> None:
    if config.scratch_root is None:
        return
    scratch = config.scratch_root.resolve(strict=True)
    root = job_root.resolve(strict=True)
    try:
        relative = root.relative_to(scratch)
    except ValueError:
        return
    if len(relative.parts) != 2 or relative.parts[0] != "jobs":
        raise LightningMapContractError(
            "Le résultat local sort du staging de job"
        )
    shutil.rmtree(root)


def run() -> dict[str, Any]:
    callback = CallbackClient()
    request: dict[str, Any] | None = None
    last_fraction = 0.0
    last_message = "Initialisation Lightning"
    try:
        request = callback.fetch_request()
        config = ProductionConfig.from_environment()
        if config.dataset_id is None:
            raise LightningMapContractError(
                "FIREVIEWER_HF_DATASET_ID est obligatoire"
            )
        hf_token = _required_environment("HF_TOKEN")
        engine_config = replace(
            config,
            dataset_publication_required=False,
        )
        engine = ProductionEngine(engine_config)
        fixed = normalize_fixed_assets(
            request["fixed_asset_placements"] or dict(EMPTY_REQUEST),
            engine.asset_library_payload,
        )

        def report(fraction: float, message: str) -> None:
            nonlocal last_fraction, last_message
            last_fraction, last_message = fraction, message
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
        final_message = "Production terminée"
        gallery: list[tuple[str, str]] = []

        # The engine still creates the canonical dataset entry and retry receipt,
        # but its legacy one-shot create_commit upload is deliberately bypassed.
        # The wrapper below performs the real resumable Xet upload.
        os.environ.pop("HF_TOKEN", None)
        try:
            for message, archive, items in engine.run(
                request["latitude"],
                request["longitude"],
                request["side_km"],
                progress_callback=report,
                archive_ready_callback=None,
                fixed_asset_placements=fixed,
            ):
                final_message = message
                if archive:
                    archive_path = Path(archive)
                if items:
                    gallery = items
        finally:
            os.environ["HF_TOKEN"] = hf_token

        if archive_path is None or gallery:
            raise LightningMapContractError(
                "Le moteur n'a pas publié le pack complet"
            )
        job_root = archive_path.parent
        publication_path = job_root / DATASET_PUBLICATION_NAME
        entry_path = job_root / DATASET_ENTRY_NAME
        zone_receipt_path = job_root / ZONE_RECEIPT_NAME
        if (
            not publication_path.is_file()
            or not entry_path.is_file()
            or not zone_receipt_path.is_file()
        ):
            raise LightningMapContractError("Reçus finaux du pack absents")

        zone_receipt = _load_json(zone_receipt_path, "Reçu de zone")
        dataset_entry = _load_json(entry_path, "Entrée dataset")
        tile_count = zone_receipt.get("tile_count")
        if isinstance(tile_count, bool) or not isinstance(tile_count, int) or tile_count <= 0:
            raise LightningMapContractError("Nombre de tuiles final invalide")

        callback.progress(
            0.995,
            "Publication résumable du ZIP complet sur Hugging Face/Xet",
            phase="dataset_publication_xet",
            current_tile=tile_count,
            tile_count=tile_count,
            force=True,
        )
        publication = _publish_archive(
            config,
            job_root=job_root,
            archive_path=archive_path,
            zone_receipt=zone_receipt,
            dataset_entry=dataset_entry,
        )
        final_message = final_message.replace(
            " Publication Hugging Face différée; téléchargement admin disponible.",
            f" Publié dans {config.dataset_id}.",
        )

        callback.progress(
            0.997,
            "Export de la map 3D complète pour le viewer",
            phase="viewer_export",
            current_tile=tile_count,
            tile_count=tile_count,
            force=True,
        )
        viewer = _export_viewer(config, job_root)

        callback.progress(
            0.999,
            "Publication de la map viewer complète sur Hugging Face/Xet",
            phase="viewer_publication_xet",
            current_tile=tile_count,
            tile_count=tile_count,
            force=True,
        )
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
            "tile_count": tile_count,
            "degraded_mns_tile_count": zone_receipt.get(
                "degraded_mns_tile_count",
                0,
            ),
            "dataset": {
                "repo_id": publication["dataset_id"],
                "revision": revision,
                "root": remote_root,
            },
            "archive": {
                "path": f"{remote_root}/{ZIP_NAME}",
                "sha256": _sha256_file(archive_path),
            },
            "viewer": {
                "path": f"{remote_root}/{GLB_NAME}",
                "receipt_path": f"{remote_root}/{RECEIPT_NAME}",
                "sha256": viewer["viewer"]["sha256"],
                "byte_count": viewer["viewer"]["byte_count"],
                "completeness": viewer["completeness"],
            },
            "captures": [],
        }
        callback.result(result)
        callback.progress(
            1.0,
            str(result["message"]),
            phase="completed",
            current_tile=tile_count,
            tile_count=tile_count,
            state="completed",
            force=True,
        )
        _cleanup_local_result(config, job_root)
        return result
    except Exception as error:
        callback.progress(
            last_fraction,
            (
                last_message
                if last_message != "Initialisation Lightning"
                else str(error)
            ),
            phase="failed",
            current_tile=None,
            tile_count=None,
            state="failed",
            error=str(error),
            force=True,
        )
        raise


def main() -> None:
    print(
        json.dumps(
            run(),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
