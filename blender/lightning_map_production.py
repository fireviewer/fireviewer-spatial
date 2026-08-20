"""Lightning Batch Job entrypoint for FireViewer map production.

New map jobs never create a final ZIP. The portable scene is sealed as a normal
folder, the complete browser viewer is published first to the public Hugging
Face dataset, and a viewer-ready callback immediately makes the map eligible
for incident publication. The sealed scientific folder is then uploaded through
Hugging Face/Xet as an independent, resumable publication. A source-folder
upload failure never invalidates an already published viewer.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import simple_production_engine as production_engine

from export_complete_viewer_glb import (
    GLB_NAME,
    RECEIPT_NAME,
    SCHEMA as VIEWER_SCHEMA,
    STATUS as VIEWER_STATUS,
)
from build_tiled_viewer_package import (
    OUTPUT_DIRECTORY as TILED_OUTPUT_DIRECTORY,
    build_tiled_viewer_package,
    validate_tiled_viewer_package,
)
from fixed_asset_placement import (
    EMPTY_REQUEST,
    normalize_request as normalize_fixed_assets,
    request_sha256 as fixed_assets_request_sha256,
)
from portable_scene_package import (
    INVENTORY_NAME,
    MANIFEST_NAME,
    MAP_CONTRACT_PATH,
    validate_map_upload_package,
)
from runpod_map_production import normalize_map_request, request_sha256
from simple_production_engine import (
    ZONE_RECEIPT_NAME,
    ProductionConfig,
    ProductionEngine,
    plan_zone,
)

REQUEST_SCHEMA = "fireviewer.map-production-request.v1"
PROGRESS_SCHEMA = "fireviewer.map-production-progress.v1"
RESULT_SCHEMA = "fireviewer.map-production-result.v3"
VIEWER_READY_SCHEMA = "fireviewer.map-viewer-ready.v1"
SOURCE_PUBLICATION_SCHEMA = "fireviewer.map-source-publication.v1"
PROGRESS_MIN_INTERVAL_SECONDS = 10.0
BLENDER_HEARTBEAT_SECONDS = 15.0
BLENDER_SCRIPT_TIMEOUT_SECONDS = 30 * 60
HF_UPLOAD_MAX_ATTEMPTS = 4
HF_UPLOAD_BACKOFF_SECONDS = (5.0, 15.0, 30.0)
_SOURCE_METADATA_FILES = (MANIFEST_NAME, INVENTORY_NAME, MAP_CONTRACT_PATH)


class LightningMapContractError(ValueError):
    pass


class _SealedFolderReady(RuntimeError):
    def __init__(self, root: Path) -> None:
        super().__init__(str(root))
        self.root = root


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
        self.viewer_ready_url = _required_environment("FIREVIEWER_MAP_VIEWER_READY_URL")
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

    def viewer_ready(self, payload: dict[str, Any]) -> None:
        self._request("POST", self.viewer_ready_url, payload=payload)

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
    *,
    heartbeat: Callable[[float], None] | None = None,
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
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    started_at = time.monotonic()
    stdout = ""
    stderr = ""
    while True:
        elapsed = time.monotonic() - started_at
        remaining = BLENDER_SCRIPT_TIMEOUT_SECONDS - elapsed
        if remaining <= 0:
            process.kill()
            stdout, stderr = process.communicate()
            raise LightningMapContractError(
                f"{script_name} a dépassé {BLENDER_SCRIPT_TIMEOUT_SECONDS // 60} minutes:\n"
                + (stdout + "\n" + stderr)[-4000:]
            )
        try:
            stdout, stderr = process.communicate(
                timeout=min(BLENDER_HEARTBEAT_SECONDS, remaining)
            )
            break
        except subprocess.TimeoutExpired:
            if heartbeat is not None:
                heartbeat(time.monotonic() - started_at)
    if process.returncode != 0:
        raise LightningMapContractError(
            f"{script_name} a échoué:\n" + (stdout + "\n" + stderr)[-4000:]
        )


def _export_viewer(
    config: ProductionConfig,
    job_root: Path,
    *,
    callback: CallbackClient,
    tile_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def heartbeat(elapsed: float) -> None:
        callback.progress(
            0.967,
            f"Export viewer Blender en cours · {int(elapsed)} s",
            phase="viewer_export_blender",
            current_tile=tile_count,
            tile_count=tile_count,
            force=True,
        )

    _run_blender_script(
        config,
        job_root,
        "export_complete_viewer_glb.py",
        heartbeat=heartbeat,
    )
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
    try:
        build_tiled_viewer_package(job_root)
        tiled_receipt, tiled_viewer = validate_tiled_viewer_package(job_root)
    except Exception as error:
        raise LightningMapContractError(
            "Paquet viewer canonique tuilé invalide"
        ) from error
    return tiled_receipt, tiled_viewer


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
        remote_path = f"{remote_root.rstrip('/')}/{file_name}"
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
    folder: Path,
    *,
    dataset_id: str,
    remote_root: str,
    file_names: Sequence[str],
    verify_names: Sequence[str],
    commit_message: str,
) -> str:
    from huggingface_hub import HfApi

    token = _required_environment("HF_TOKEN")
    last_error: Exception | None = None
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    for attempt in range(1, HF_UPLOAD_MAX_ATTEMPTS + 1):
        try:
            api = HfApi(token=token)
            info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
            if getattr(info, "private", None) is not False:
                raise LightningMapContractError(
                    "La dataset cible FireViewer doit être publique"
                )
            commit = api.upload_folder(
                repo_id=dataset_id,
                repo_type="dataset",
                folder_path=str(folder),
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
                file_names=verify_names,
            )
            return oid
        except LightningMapContractError:
            raise
        except Exception as error:
            last_error = error
            if attempt >= HF_UPLOAD_MAX_ATTEMPTS or _is_authentication_error(error):
                break
            time.sleep(HF_UPLOAD_BACKOFF_SECONDS[attempt - 1])
    raise LightningMapContractError(
        f"Publication Hugging Face/Xet échouée après {HF_UPLOAD_MAX_ATTEMPTS} tentatives"
    ) from last_error


def _skip_archive_budget(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "schema": "fireviewer.map-folder-publication.v1",
        "status": "zip_disabled",
    }


@contextmanager
def _sealed_folder_mode() -> Iterator[None]:
    """Stop the legacy engine immediately after it seals the portable folder.

    The core engine remains backward compatible for local/legacy callers that
    still request a ZIP. Lightning explicitly uses the folder-native contract:
    no archive budget and no ZIP writer are reached.
    """

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


def _safe_inventory_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise LightningMapContractError("Chemin d'inventaire portable invalide")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise LightningMapContractError("Chemin d'inventaire portable non confiné")
    return path.as_posix()


def _source_file_names(job_root: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    inventory = _load_json(job_root / INVENTORY_NAME, "Inventaire portable")
    files = inventory.get("files")
    if (
        inventory.get("schema") != "fireviewer.portable-package-inventory.v1"
        or inventory.get("status") != "sealed"
        or inventory.get("package_role") != "map"
        or not isinstance(files, list)
        or inventory.get("file_count") != len(files)
        or not files
    ):
        raise LightningMapContractError("Inventaire portable incomplet")
    paths: list[str] = []
    total_bytes = 0
    for record in files:
        if not isinstance(record, Mapping):
            raise LightningMapContractError("Entrée d'inventaire portable invalide")
        relative = _safe_inventory_path(record.get("path"))
        byte_count = record.get("byte_count")
        sha256 = record.get("sha256")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not _is_sha256(sha256)
        ):
            raise LightningMapContractError("Entrée d'inventaire portable invalide")
        path = job_root / relative
        if not path.is_file() or path.stat().st_size != byte_count:
            raise LightningMapContractError(
                f"Fichier scellé absent avant publication: {relative}"
            )
        paths.append(relative)
        total_bytes += byte_count
    for relative in _SOURCE_METADATA_FILES:
        path = job_root / relative
        if not path.is_file():
            raise LightningMapContractError(f"Métadonnée scellée absente: {relative}")
        paths.append(relative)
        total_bytes += path.stat().st_size
    names = tuple(sorted(set(paths)))
    return names, {
        "schema": SOURCE_PUBLICATION_SCHEMA,
        "file_count": len(names),
        "byte_count": total_bytes,
        "inventory_sha256": _sha256_file(job_root / INVENTORY_NAME),
        "manifest_sha256": _sha256_file(job_root / MANIFEST_NAME),
        "contract_sha256": _sha256_file(job_root / MAP_CONTRACT_PATH),
    }


def _materialize_sealed_symlinks(job_root: Path, file_names: Sequence[str]) -> None:
    """Turn portable file symlinks into regular files before remote upload."""

    for relative in file_names:
        path = job_root / relative
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        if not target.is_file():
            raise LightningMapContractError(
                f"Lien scellé non régulier avant publication: {relative}"
            )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.materialize")
        try:
            shutil.copyfile(target, temporary)
            path.unlink()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


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


def _viewer_payload(
    *,
    callback: CallbackClient,
    request: Mapping[str, Any],
    zone_receipt: Mapping[str, Any],
    viewer: Mapping[str, Any],
    dataset_id: str,
    runtime_root: str,
    revision: str,
) -> dict[str, Any]:
    completeness = viewer.get("completeness")
    bootstrap_asset = viewer.get("bootstrap_asset")
    if not isinstance(completeness, Mapping) or not isinstance(bootstrap_asset, Mapping):
        raise LightningMapContractError("Reçu viewer incomplet")
    return {
        "schema": VIEWER_READY_SCHEMA,
        "job_id": callback.job_id,
        "status": "viewer_ready",
        "request_sha256": request_sha256(dict(request)),
        "zone_id": zone_receipt["zone_id"],
        "build_id": zone_receipt["build_id"],
        "tile_count": zone_receipt["tile_count"],
        "degraded_mns_tile_count": zone_receipt.get("degraded_mns_tile_count", 0),
        "dataset": {
            "repo_id": dataset_id,
            "revision": revision,
            "root": runtime_root,
            "visibility": "public",
        },
        "viewer": {
            "catalog_path": f"{runtime_root}/{viewer['catalog_path']}",
            "receipt_path": f"{runtime_root}/{viewer['receipt_path']}",
            "catalog_sha256": viewer["catalog_sha256"],
            "catalog_byte_count": viewer["catalog_byte_count"],
            "payload_file_count": viewer["payload_file_count"],
            "payload_byte_count": viewer["payload_byte_count"],
            "bootstrap_asset": {
                **dict(bootstrap_asset),
                "path": f"{runtime_root}/{bootstrap_asset['path']}",
            },
            "representation": "complete_tiled_non_simplified_map",
            "completeness": dict(completeness),
        },
        "captures": [],
    }


def run() -> dict[str, Any]:
    callback = CallbackClient()
    request: dict[str, Any] | None = None
    last_fraction = 0.0
    last_message = "Initialisation Lightning"
    viewer_ready_sent = False
    try:
        request = callback.fetch_request()
        config = ProductionConfig.from_environment()
        if config.dataset_id is None:
            raise LightningMapContractError("FIREVIEWER_HF_DATASET_ID est obligatoire")
        _required_environment("HF_TOKEN")

        # The spatial engine remains the authority for measured-scene creation
        # and sealing. Dataset/ZIP delivery is owned by this Lightning wrapper.
        engine_config = replace(
            config,
            dataset_id=None,
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

        job_root: Path | None = None
        try:
            with _sealed_folder_mode():
                for _message, archive, _gallery in engine.run(
                    request["latitude"],
                    request["longitude"],
                    request["side_km"],
                    progress_callback=report,
                    archive_ready_callback=None,
                    fixed_asset_placements=fixed,
                ):
                    if archive is not None:
                        raise LightningMapContractError(
                            "Le mode Lightning folder-only a produit une archive inattendue"
                        )
        except _SealedFolderReady as ready:
            job_root = ready.root

        if job_root is None:
            raise LightningMapContractError(
                "Le moteur n'a pas livré le dossier scellé de la map"
            )

        validate_map_upload_package(job_root)
        zone_receipt = _load_json(job_root / ZONE_RECEIPT_NAME, "Reçu de zone")
        tile_count = zone_receipt.get("tile_count")
        zone_id = zone_receipt.get("zone_id")
        build_id = zone_receipt.get("build_id")
        if (
            isinstance(tile_count, bool)
            or not isinstance(tile_count, int)
            or tile_count <= 0
            or not isinstance(zone_id, str)
            or not zone_id
            or not _is_sha256(build_id)
        ):
            raise LightningMapContractError("Identité finale de la map invalide")

        last_fraction, last_message = 0.965, "Export de la map 3D complète pour le viewer"
        callback.progress(
            last_fraction,
            last_message,
            phase="viewer_export",
            current_tile=tile_count,
            tile_count=tile_count,
            force=True,
        )
        _tiled_viewer_receipt, tiled_viewer = _export_viewer(
            config,
            job_root,
            callback=callback,
            tile_count=tile_count,
        )

        base_root = f"maps/{zone_id}/{build_id}"
        runtime_root = f"{base_root}/runtime"
        source_root = f"{base_root}/source"

        last_fraction, last_message = 0.975, "Publication du viewer 3D sur Hugging Face/Xet"
        callback.progress(
            last_fraction,
            last_message,
            phase="viewer_publication_xet",
            current_tile=tile_count,
            tile_count=tile_count,
            force=True,
        )
        viewer_revision = _upload_hf_folder(
            job_root,
            dataset_id=config.dataset_id,
            remote_root=runtime_root,
            file_names=(f"{TILED_OUTPUT_DIRECTORY}/**",),
            verify_names=(
                str(tiled_viewer["catalog_path"]),
                str(tiled_viewer["receipt_path"]),
                str(tiled_viewer["bootstrap_asset"]["path"]),
            ),
            commit_message=f"Publish FireViewer runtime {zone_id}",
        )
        viewer_ready = _viewer_payload(
            callback=callback,
            request=request,
            zone_receipt=zone_receipt,
            viewer=tiled_viewer,
            dataset_id=config.dataset_id,
            runtime_root=runtime_root,
            revision=viewer_revision,
        )
        callback.viewer_ready(viewer_ready)
        viewer_ready_sent = True

        last_fraction = 0.985
        last_message = "Viewer prêt à publier — envoi du dossier scientifique sur Hugging Face/Xet"
        callback.progress(
            last_fraction,
            last_message,
            phase="source_publication_xet",
            current_tile=tile_count,
            tile_count=tile_count,
            force=True,
        )

        source_names, source_record = _source_file_names(job_root)
        _materialize_sealed_symlinks(job_root, source_names)
        validate_map_upload_package(job_root)
        source_record.update(
            {
                "status": "failed_pending_retry",
                "root": source_root,
                "visibility": "public",
            }
        )
        try:
            source_revision = _upload_hf_folder(
                job_root,
                dataset_id=config.dataset_id,
                remote_root=source_root,
                file_names=source_names,
                verify_names=(
                    MANIFEST_NAME,
                    INVENTORY_NAME,
                    MAP_CONTRACT_PATH,
                    "zone.usda",
                    ZONE_RECEIPT_NAME,
                ),
                commit_message=f"Publish measured FireViewer scene {zone_id}",
            )
            source_record.update(
                {
                    "status": "published_public",
                    "revision": source_revision,
                }
            )
        except Exception as source_error:
            source_record.update(
                {
                    "error_type": type(source_error).__name__,
                    "error": "Publication du dossier source à reprendre",
                }
            )

        source_published = source_record["status"] == "published_public"
        message = (
            "Production terminée — viewer 3D et dossier scientifique publiés sur Hugging Face."
            if source_published
            else "Production terminée — viewer 3D publié et utilisable; dossier scientifique HF à reprendre."
        )
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": callback.job_id,
            "status": "technical_scene_produced",
            "request_sha256": request_sha256(request),
            "zone_id": zone_id,
            "build_id": build_id,
            "message": message,
            "tile_count": tile_count,
            "degraded_mns_tile_count": zone_receipt.get(
                "degraded_mns_tile_count",
                0,
            ),
            "dataset": dict(viewer_ready["dataset"]),
            "viewer": dict(viewer_ready["viewer"]),
            "source": source_record,
            "captures": [],
        }
        callback.result(result)
        callback.progress(
            1.0,
            message,
            phase=("completed" if source_published else "completed_source_pending"),
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
                "Viewer publié mais finalisation du job interrompue"
                if viewer_ready_sent
                else last_message
                if last_message != "Initialisation Lightning"
                else str(error)
            ),
            phase="failed_after_viewer" if viewer_ready_sent else "failed",
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
