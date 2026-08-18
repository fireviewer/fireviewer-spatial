from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import lightning_map_production as worker
from simple_production_engine import ProductionConfig


REQUEST = {
    "latitude": 43.9,
    "longitude": 4.5,
    "side_km": 0.5,
    "fixed_asset_placements": None,
}


def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "FIREVIEWER_MAP_JOB_ID": "map-" + "a" * 32,
        "FIREVIEWER_MAP_REQUEST_URL": "https://api.example/request",
        "FIREVIEWER_MAP_PROGRESS_URL": "https://api.example/progress",
        "FIREVIEWER_MAP_RESULT_URL": "https://api.example/result",
        "FIREVIEWER_MAP_CALLBACK_TOKEN": "b" * 64,
        "HF_TOKEN": "hf-test-token",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_callback_fetches_hash_locked_request_and_posts_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch)
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        assert request.headers["X-FireViewer-Map-Token"] == "b" * 64
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "schema": worker.REQUEST_SCHEMA,
                    "job_id": "map-" + "a" * 32,
                    "request_sha256": worker.request_sha256(REQUEST),
                    "request": REQUEST,
                },
            )
        return httpx.Response(204)

    original_client = httpx.Client
    monkeypatch.setattr(
        worker.httpx,
        "Client",
        lambda **kwargs: original_client(
            headers=kwargs.get("headers"),
            transport=httpx.MockTransport(handler),
        ),
    )
    callback = worker.CallbackClient()
    assert callback.fetch_request() == REQUEST
    callback.progress(
        0.5,
        "Tuile 1/1",
        phase="terrain",
        current_tile=1,
        tile_count=1,
        force=True,
    )
    progress = json.loads(observed[-1].content)
    assert progress["schema"] == worker.PROGRESS_SCHEMA
    assert progress["sequence"] == 0
    assert progress["current_tile"] == 1


def test_callback_rejects_request_hash_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch)
    handler = lambda _request: httpx.Response(  # noqa: E731
        200,
        json={
            "schema": worker.REQUEST_SCHEMA,
            "job_id": "map-" + "a" * 32,
            "request_sha256": "0" * 64,
            "request": REQUEST,
        },
    )
    original_client = httpx.Client
    monkeypatch.setattr(
        worker.httpx,
        "Client",
        lambda **kwargs: original_client(
            headers=kwargs.get("headers"),
            transport=httpx.MockTransport(handler),
        ),
    )
    with pytest.raises(worker.LightningMapContractError, match="Hash"):
        worker.CallbackClient().fetch_request()


def test_callback_coalesces_concurrent_progress_without_reusing_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch)
    observed: list[dict[str, Any]] = []
    observed_lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            with observed_lock:
                observed.append(json.loads(request.content))
        return httpx.Response(204)

    original_client = httpx.Client
    monkeypatch.setattr(
        worker.httpx,
        "Client",
        lambda **kwargs: original_client(
            headers=kwargs.get("headers"),
            transport=httpx.MockTransport(handler),
        ),
    )
    monotonic_values = iter([100.0] + [101.0] * 8 + [111.0])
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(monotonic_values))
    callback = worker.CallbackClient()

    callback.progress(
        0.1,
        "first",
        phase="sources",
        current_tile=0,
        tile_count=625,
    )
    threads = [
        threading.Thread(
            target=callback.progress,
            args=(0.2 + index / 1000, f"coalesced-{index}"),
            kwargs={
                "phase": "sources",
                "current_tile": index,
                "tile_count": 625,
            },
        )
        for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    callback.progress(
        0.3,
        "later",
        phase="terrain",
        current_tile=1,
        tile_count=625,
    )

    assert [payload["sequence"] for payload in observed] == [0, 1]
    assert [payload["message"] for payload in observed] == ["first", "later"]


def test_hf_publication_uses_resumable_folder_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    for name in ("fireviewer-zone.zip", "zone.done.json", "dataset-entry.json"):
        (tmp_path / name).write_bytes(b"payload")
    calls: dict[str, Any] = {}

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            calls["token"] = token

        def repo_info(self, *, repo_id: str, repo_type: str) -> Any:
            calls["repo_info"] = (repo_id, repo_type)
            return SimpleNamespace(private=True)

        def upload_folder(self, **kwargs: Any) -> Any:
            calls["upload_folder"] = kwargs
            return SimpleNamespace(oid="a" * 40)

        def file_exists(self, **kwargs: Any) -> bool:
            calls.setdefault("file_exists", []).append(kwargs)
            return True

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi),
    )
    revision = worker._upload_hf_folder(
        tmp_path,
        dataset_id="fireviewer/simple-measured-scenes-v1",
        remote_root="zones/GPS-TEST/" + "b" * 64,
        file_names=(
            "fireviewer-zone.zip",
            "zone.done.json",
            "dataset-entry.json",
        ),
        commit_message="test upload",
    )

    assert revision == "a" * 40
    upload = calls["upload_folder"]
    assert upload["folder_path"] == str(tmp_path)
    assert upload["allow_patterns"] == [
        "fireviewer-zone.zip",
        "zone.done.json",
        "dataset-entry.json",
    ]
    assert len(calls["file_exists"]) == 3


def test_archive_publication_writes_verified_private_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "fireviewer-zone.zip"
    archive.write_bytes(b"complete map")
    archive_sha256 = worker._sha256_file(archive)
    entry = {
        "dataset_id": "fireviewer/simple-measured-scenes-v1",
        "zone_id": "GPS-TEST",
        "build_id": "b" * 64,
        "entry_sha256": "c" * 64,
        "archive": {
            "file": "fireviewer-zone.zip",
            "byte_count": archive.stat().st_size,
            "sha256": archive_sha256,
        },
    }
    receipt = {"zone_id": "GPS-TEST", "build_id": "b" * 64}
    config = SimpleNamespace(dataset_id="fireviewer/simple-measured-scenes-v1")
    monkeypatch.setattr(
        worker,
        "_upload_hf_folder",
        lambda *_args, **_kwargs: "d" * 40,
    )

    publication = worker._publish_archive(
        config,
        job_root=tmp_path,
        archive_path=archive,
        zone_receipt=receipt,
        dataset_entry=entry,
    )

    assert publication["status"] == "published_private"
    assert publication["transport"] == "huggingface_hub.upload_folder+xet"
    assert publication["commit_oid"] == "d" * 40
    stored = worker._load_json(tmp_path / "dataset-publication.json", "receipt")
    assert stored == publication


def test_job_publishes_hf_result_and_cleans_local_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _environment(monkeypatch)
    work = tmp_path / "work"
    scratch = tmp_path / "scratch"
    assets = tmp_path / "assets.json"
    review = tmp_path / "assets"
    blender = tmp_path / "blender"
    work.mkdir()
    scratch.mkdir()
    review.mkdir()
    assets.write_text("{}", encoding="utf-8")
    blender.write_bytes(b"blender")
    config = ProductionConfig(
        work_root=work,
        portable_root=tmp_path,
        asset_library=assets,
        review_batch=review,
        elevation_revision="elevation-v1",
        orthophoto_revision="ortho-v1",
        context_revision="context-v1",
        blender=blender,
        dataset_id="fireviewer/simple-measured-scenes-v1",
        scratch_root=scratch,
    )

    class FakeCallback:
        job_id = "map-" + "a" * 32

        def __init__(self) -> None:
            self.progress_records: list[dict[str, Any]] = []
            self.result_record: dict[str, Any] | None = None

        def fetch_request(self) -> dict[str, Any]:
            return dict(REQUEST)

        def progress(self, fraction: float, message: str, **details: Any) -> None:
            self.progress_records.append(
                {"fraction": fraction, "message": message, **details}
            )

        def result(self, payload: dict[str, Any]) -> None:
            self.result_record = payload

    callback = FakeCallback()

    class FakeEngine:
        asset_library_payload: dict[str, Any] = {}

        def __init__(self, received: ProductionConfig) -> None:
            assert received.dataset_publication_required is False
            assert received.dataset_id == config.dataset_id

        def run(
            self,
            *_args: Any,
            progress_callback: Any,
            archive_ready_callback: Any,
            **_kwargs: Any,
        ) -> Any:
            assert archive_ready_callback is None
            progress_callback(0.5, "Tuile 1/1 — terrain")
            root = scratch / "jobs" / "GPS-TEST"
            root.mkdir(parents=True)
            archive = root / "fireviewer-zone.zip"
            archive.write_bytes(b"zip")
            archive_sha256 = hashlib.sha256(b"zip").hexdigest()
            (root / "dataset-publication.json").write_text(
                json.dumps({"status": "failed_pending_retry"}),
                encoding="utf-8",
            )
            (root / "dataset-entry.json").write_text(
                json.dumps(
                    {
                        "dataset_id": config.dataset_id,
                        "zone_id": "GPS-TEST",
                        "build_id": "c" * 64,
                        "entry_sha256": "f" * 64,
                        "archive": {
                            "file": "fireviewer-zone.zip",
                            "byte_count": 3,
                            "sha256": archive_sha256,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "zone.done.json").write_text(
                json.dumps(
                    {
                        "zone_id": "GPS-TEST",
                        "build_id": "c" * 64,
                        "tile_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            yield "Terminé — Publication Hugging Face différée; téléchargement admin disponible.", str(archive), []

    monkeypatch.setattr(worker, "CallbackClient", lambda: callback)
    monkeypatch.setattr(worker.ProductionConfig, "from_environment", lambda: config)
    monkeypatch.setattr(worker, "ProductionEngine", FakeEngine)
    monkeypatch.setattr(
        worker,
        "normalize_fixed_assets",
        lambda request, _catalog: request,
    )
    monkeypatch.setattr(
        worker,
        "_publish_archive",
        lambda *_args, **_kwargs: {
            "dataset_id": config.dataset_id,
            "path_in_repo": "zones/GPS-TEST/" + "c" * 64,
        },
    )
    monkeypatch.setattr(
        worker,
        "_export_viewer",
        lambda *_args, **_kwargs: {
            "zone_id": "GPS-TEST",
            "viewer": {"sha256": "d" * 64, "byte_count": 123},
            "completeness": {"mesh_coverage": "complete"},
        },
    )
    monkeypatch.setattr(
        worker,
        "_publish_viewer",
        lambda *_args, **_kwargs: "e" * 40,
    )

    result = worker.run()

    assert result["schema"] == worker.RESULT_SCHEMA
    assert result["captures"] == []
    assert result["dataset"]["revision"] == "e" * 40
    assert result["viewer"]["completeness"]["mesh_coverage"] == "complete"
    assert callback.result_record == result
    assert callback.progress_records[-1]["state"] == "completed"
    assert not (scratch / "jobs" / "GPS-TEST").exists()
