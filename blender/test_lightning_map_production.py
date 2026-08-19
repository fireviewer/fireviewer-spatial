from __future__ import annotations

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
        "FIREVIEWER_MAP_VIEWER_READY_URL": "https://api.example/viewer-ready",
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


def test_hf_publication_accepts_public_dataset_and_uses_xet_folder_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    (tmp_path / "viewer.glb").write_bytes(b"viewer")
    (tmp_path / "viewer-scene.v1.json").write_bytes(b"{}")
    calls: dict[str, Any] = {}

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            calls["token"] = token

        def repo_info(self, *, repo_id: str, repo_type: str) -> Any:
            calls["repo_info"] = (repo_id, repo_type)
            return SimpleNamespace(private=False)

        def upload_folder(self, **kwargs: Any) -> Any:
            calls["upload_folder"] = kwargs
            return SimpleNamespace(oid="a" * 40)

        def file_exists(self, **kwargs: Any) -> bool:
            calls.setdefault("file_exists", []).append(kwargs)
            return True

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    revision = worker._upload_hf_folder(
        tmp_path,
        dataset_id="fireviewer/simple-measured-scenes-v1",
        remote_root="maps/GPS-TEST/" + "b" * 64 + "/runtime",
        file_names=("viewer.glb", "viewer-scene.v1.json"),
        verify_names=("viewer.glb", "viewer-scene.v1.json"),
        commit_message="test upload",
    )

    assert revision == "a" * 40
    assert calls["token"] == "hf-test-token"
    assert calls["upload_folder"]["folder_path"] == str(tmp_path)
    assert calls["upload_folder"]["allow_patterns"] == [
        "viewer.glb",
        "viewer-scene.v1.json",
    ]
    assert len(calls["file_exists"]) == 2


def test_hf_publication_rejects_private_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    (tmp_path / "viewer.glb").write_bytes(b"viewer")

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            del token

        def repo_info(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(private=True)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)
    with pytest.raises(worker.LightningMapContractError, match="doit être publique"):
        worker._upload_hf_folder(
            tmp_path,
            dataset_id="fireviewer/simple-measured-scenes-v1",
            remote_root="runtime",
            file_names=("viewer.glb",),
            verify_names=("viewer.glb",),
            commit_message="test",
        )


def _config(tmp_path: Path) -> ProductionConfig:
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
    return ProductionConfig(
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
        self.viewer_ready_record: dict[str, Any] | None = None
        self.result_record: dict[str, Any] | None = None
        self.events: list[str] = []

    def fetch_request(self) -> dict[str, Any]:
        return dict(REQUEST)

    def progress(self, fraction: float, message: str, **details: Any) -> None:
        self.progress_records.append({"fraction": fraction, "message": message, **details})

    def viewer_ready(self, payload: dict[str, Any]) -> None:
        self.events.append("viewer_ready")
        self.viewer_ready_record = payload

    def result(self, payload: dict[str, Any]) -> None:
        self.events.append("result")
        self.result_record = payload


def _install_fake_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    source_upload_fails: bool,
) -> tuple[FakeCallback, ProductionConfig]:
    _environment(monkeypatch)
    config = _config(tmp_path)
    callback = FakeCallback()

    class FakeEngine:
        asset_library_payload: dict[str, Any] = {}

        def __init__(self, received: ProductionConfig) -> None:
            assert received.dataset_publication_required is False
            assert received.dataset_id is None

        def run(self, *_args: Any, progress_callback: Any, **_kwargs: Any) -> Any:
            progress_callback(0.5, "Tuile 1/1 — terrain")
            root = config.scratch_root / "jobs" / "GPS-TEST-run"
            root.mkdir(parents=True)
            (root / "zone.done.json").write_text(
                json.dumps(
                    {
                        "zone_id": "GPS-TEST",
                        "build_id": "c" * 64,
                        "tile_count": 1,
                        "degraded_mns_tile_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            raise worker._SealedFolderReady(root)
            yield  # pragma: no cover

    monkeypatch.setattr(worker, "CallbackClient", lambda: callback)
    monkeypatch.setattr(worker.ProductionConfig, "from_environment", lambda: config)
    monkeypatch.setattr(worker, "ProductionEngine", FakeEngine)
    monkeypatch.setattr(worker, "normalize_fixed_assets", lambda request, _catalog: request)
    monkeypatch.setattr(worker, "validate_map_upload_package", lambda _root: None)
    monkeypatch.setattr(
        worker,
        "_export_viewer",
        lambda *_args, **_kwargs: {
            "zone_id": "GPS-TEST",
            "viewer": {"sha256": "d" * 64, "byte_count": 123},
            "completeness": {
                "mesh_coverage": "complete",
                "policy": "fail_closed_exact_visual_scene",
                "family_instance_counts": {
                    "buildings": 1,
                    "trees": 1,
                    "context_assets": 0,
                },
            },
        },
    )
    monkeypatch.setattr(
        worker,
        "_source_file_names",
        lambda _root: (
            ("manifest.json",),
            {
                "schema": worker.SOURCE_PUBLICATION_SCHEMA,
                "file_count": 1,
                "byte_count": 10,
                "inventory_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "contract_sha256": "3" * 64,
            },
        ),
    )
    monkeypatch.setattr(worker, "_materialize_sealed_symlinks", lambda *_args: None)

    calls = 0

    def upload(*_args: Any, **_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "e" * 40
        if source_upload_fails:
            raise worker.LightningMapContractError("source failed")
        return "f" * 40

    monkeypatch.setattr(worker, "_upload_hf_folder", upload)
    return callback, config


def test_job_publishes_viewer_before_source_and_never_creates_zip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback, config = _install_fake_job(
        monkeypatch,
        tmp_path,
        source_upload_fails=False,
    )

    result = worker.run()

    assert result["schema"] == worker.RESULT_SCHEMA
    assert result["dataset"]["visibility"] == "public"
    assert result["source"]["status"] == "published_public"
    assert callback.events == ["viewer_ready", "result"]
    assert callback.viewer_ready_record is not None
    assert callback.viewer_ready_record["schema"] == worker.VIEWER_READY_SCHEMA
    assert callback.result_record == result
    assert not list(config.scratch_root.rglob("*.zip"))
    assert not (config.scratch_root / "jobs" / "GPS-TEST-run").exists()


def test_source_upload_failure_keeps_viewer_publishable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback, _config_value = _install_fake_job(
        monkeypatch,
        tmp_path,
        source_upload_fails=True,
    )

    result = worker.run()

    assert callback.events == ["viewer_ready", "result"]
    assert result["source"]["status"] == "failed_pending_retry"
    assert result["viewer"]["sha256"] == "d" * 64
    assert callback.progress_records[-1]["state"] == "completed"
    assert callback.progress_records[-1]["phase"] == "completed_source_pending"
