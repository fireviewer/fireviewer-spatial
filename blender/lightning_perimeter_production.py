"""Lightning Batch Job entrypoint for observed perimeter timeline production."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from geographic_perimeter_layer import produce_perimeter_layer
from geographic_perimeter_viewer import build_perimeter_timeline_viewer
from portable_scene_package import (
    materialize_perimeter_upload_package,
    read_map_reference_from_archive,
    validate_perimeter_upload_package,
)

REQUEST_SCHEMA = "fireviewer.perimeter-production-request.v1"
PROGRESS_SCHEMA = "fireviewer.perimeter-production-progress.v1"
RESULT_SCHEMA = "fireviewer.perimeter-production-result.v1"
_JOB_ID_RE = re.compile(r"^perimeter-[0-9a-f]{32}$")


class LightningPerimeterContractError(ValueError):
    """The perimeter job environment or callback payload is invalid."""


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise LightningPerimeterContractError(f"Variable obligatoire absente: {name}")
    return value


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


class CallbackClient:
    def __init__(self) -> None:
        self.job_id = _required_environment("FIREVIEWER_PERIMETER_JOB_ID")
        if _JOB_ID_RE.fullmatch(self.job_id) is None:
            raise LightningPerimeterContractError(
                "Identifiant de job périmètre invalide"
            )
        self.request_url = _required_environment("FIREVIEWER_PERIMETER_REQUEST_URL")
        self.progress_url = _required_environment("FIREVIEWER_PERIMETER_PROGRESS_URL")
        self.result_url = _required_environment("FIREVIEWER_PERIMETER_RESULT_URL")
        self.token = _required_environment("FIREVIEWER_PERIMETER_CALLBACK_TOKEN")
        self.sequence = 0
        self._client = httpx.Client(
            headers={"X-FireViewer-Perimeter-Token": self.token},
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
        raise LightningPerimeterContractError(
            "Callback FireViewer inaccessible"
        ) from last_error

    def fetch_request(self) -> tuple[dict[str, Any], str]:
        envelope = self._request("GET", self.request_url)
        if (
            not isinstance(envelope, Mapping)
            or envelope.get("schema") != REQUEST_SCHEMA
            or envelope.get("job_id") != self.job_id
            or not isinstance(envelope.get("request"), Mapping)
        ):
            raise LightningPerimeterContractError("Requête de périmètre invalide")
        request = dict(envelope["request"])
        request_hash = hashlib.sha256(_canonical_bytes(request)).hexdigest()
        if envelope.get("request_sha256") != request_hash:
            raise LightningPerimeterContractError("Hash de requête périmètre divergent")
        if not isinstance(request.get("source"), Mapping) or not isinstance(
            request.get("base_map"), Mapping
        ):
            raise LightningPerimeterContractError("Source ou carte de base absente")
        return request, request_hash

    def progress(
        self,
        fraction: float,
        message: str,
        *,
        phase: str,
        state: str = "running",
        error: str | None = None,
    ) -> None:
        payload = {
            "schema": PROGRESS_SCHEMA,
            "job_id": self.job_id,
            "sequence": self.sequence,
            "state": state,
            "phase": phase,
            "progress": max(0.0, min(1.0, float(fraction))),
            "message": message,
            "current_tile": None,
            "tile_count": None,
            "error": error,
        }
        self.sequence += 1
        self._request("POST", self.progress_url, payload=payload)

    def result(self, payload: dict[str, Any]) -> None:
        self._request("POST", self.result_url, payload=payload)


def _download_map(base_map: Mapping[str, Any], root: Path, token: str) -> Path:
    dataset = base_map.get("dataset")
    archive = base_map.get("archive")
    if not isinstance(dataset, Mapping) or not isinstance(archive, Mapping):
        raise LightningPerimeterContractError("Référence de carte incomplète")
    from huggingface_hub import hf_hub_download

    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=str(dataset["repo_id"]),
                repo_type="dataset",
                revision=str(dataset["revision"]),
                filename=str(archive["path"]),
                token=token,
                local_dir=root / "map-download",
            )
        ).resolve(strict=True)
    except Exception as exc:
        raise LightningPerimeterContractError(
            "Téléchargement de la carte privée impossible"
        ) from exc
    if _sha256_file(downloaded) != archive.get("sha256"):
        raise LightningPerimeterContractError("Le ZIP de carte téléchargé a changé")
    return downloaded


def _publish(
    *,
    dataset_id: str,
    token: str,
    archive: Path,
    viewer: Any,
    layer_build_id: str,
    map_build_id: str,
) -> tuple[str, str, list[dict[str, Any]]]:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    try:
        info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
        if info.private is not True:
            raise LightningPerimeterContractError(
                "La dataset cible des périmètres doit rester privée"
            )
        remote_root = f"perimeters/{layer_build_id}/{map_build_id}"
        captures: list[dict[str, Any]] = []
        operations = [
            CommitOperationAdd(
                path_in_repo=f"{remote_root}/fireviewer-perimeter-layer.zip",
                path_or_fileobj=archive,
            )
        ]
        for frame in viewer.frames:
            remote_path = f"{remote_root}/viewer/{frame.model.name}"
            operations.append(
                CommitOperationAdd(
                    path_in_repo=remote_path, path_or_fileobj=frame.model
                )
            )
            record = viewer.manifest["frames"][frame.index]
            captures.append(
                {
                    "index": frame.index,
                    "observed_at": frame.observed_at,
                    "caption": frame.caption,
                    "path": remote_path,
                    "sha256": record["sha256"],
                    "byte_count": record["byte_count"],
                }
            )
        commit = api.create_commit(
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=f"Add observed perimeter timeline {layer_build_id[:12]}",
            operations=operations,
        )
    except LightningPerimeterContractError:
        raise
    except Exception as exc:
        raise LightningPerimeterContractError(
            "Publication privée du périmètre impossible"
        ) from exc
    return str(commit.oid), remote_root, captures


def run() -> dict[str, Any]:
    callback = CallbackClient()
    last_fraction = 0.0
    last_message = "Initialisation du périmètre"
    job_root: Path | None = None
    try:
        request, request_hash = callback.fetch_request()
        work_root = Path(_required_environment("FIREVIEWER_WORK_ROOT")).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        token = _required_environment("HF_TOKEN")
        dataset_id = _required_environment("FIREVIEWER_HF_DATASET_ID")
        jobs_root = (work_root / "jobs").resolve()
        jobs_root.mkdir(parents=True, exist_ok=True)
        job_root = (jobs_root / callback.job_id).resolve()
        if job_root.parent != jobs_root:
            raise LightningPerimeterContractError("Chemin de job périmètre invalide")
        if job_root.exists():
            shutil.rmtree(job_root)
        job_root.mkdir(parents=True)
        source = job_root / "perimeters.json"
        source.write_bytes(_canonical_bytes(request["source"]) + b"\n")

        callback.progress(0.05, "Source temporelle validée", phase="source_validation")
        last_fraction, last_message = 0.05, "Source temporelle validée"
        product = produce_perimeter_layer(source, job_root)
        layer_build_id = str(product.manifest["build_id"])

        callback.progress(0.25, "Calques OpenUSD produits", phase="perimeter_compile")
        last_fraction, last_message = 0.25, "Calques OpenUSD produits"
        map_archive = _download_map(request["base_map"], job_root, token)
        map_reference = read_map_reference_from_archive(map_archive)
        expected_map = request["base_map"]
        if map_reference.zone_id != expected_map.get(
            "zone_id"
        ) or map_reference.map_build_id != expected_map.get("build_id"):
            raise LightningPerimeterContractError(
                "La carte téléchargée ne correspond pas à la requête"
            )

        callback.progress(0.45, "Carte de base contrôlée", phase="map_validation")
        last_fraction, last_message = 0.45, "Carte de base contrôlée"
        viewer = build_perimeter_timeline_viewer(
            map_archive, product.package_root, job_root
        )
        _upload_root, archive, upload_manifest = materialize_perimeter_upload_package(
            product.package_root,
            map_reference,
            job_root,
            viewer_root=viewer.root,
        )
        validate_perimeter_upload_package(_upload_root)

        callback.progress(
            0.75, "Timeline 3D et ZIP validés", phase="package_validation"
        )
        last_fraction, last_message = 0.75, "Timeline 3D et ZIP validés"
        commit_oid, remote_root, captures = _publish(
            dataset_id=dataset_id,
            token=token,
            archive=archive,
            viewer=viewer,
            layer_build_id=layer_build_id,
            map_build_id=map_reference.map_build_id,
        )
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": callback.job_id,
            "status": "technical_perimeter_produced",
            "request_sha256": request_hash,
            "layer_build_id": layer_build_id,
            "state_count": upload_manifest["timeline"]["state_count"],
            "message": (
                f"Terminé — {len(captures)} états observés liés à la carte, "
                "prêts pour validation humaine."
            ),
            "base_map": {
                "job_id": request["base_map"]["job_id"],
                "zone_id": map_reference.zone_id,
                "build_id": map_reference.map_build_id,
                "archive_sha256": request["base_map"]["archive"]["sha256"],
            },
            "dataset": {
                "repo_id": dataset_id,
                "revision": commit_oid,
                "root": remote_root,
            },
            "archive": {
                "path": f"{remote_root}/fireviewer-perimeter-layer.zip",
                "sha256": _sha256_file(archive),
            },
            "captures": captures,
        }
        callback.result(result)
        callback.progress(1.0, result["message"], phase="completed", state="completed")
        return result
    except Exception as exc:
        callback.progress(
            last_fraction,
            last_message,
            phase="failed",
            state="failed",
            error=str(exc),
        )
        raise
    finally:
        if job_root is not None and job_root.exists():
            try:
                shutil.rmtree(job_root)
            except OSError:
                # Le résultat publié reste valide ; un nettoyage Lightning tardif
                # ne doit pas transformer un job terminé en échec.
                pass


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["CallbackClient", "LightningPerimeterContractError", "main", "run"]
