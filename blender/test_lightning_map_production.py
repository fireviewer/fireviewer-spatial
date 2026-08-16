from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
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
        "FIREVIEWER_MAP_ARCHIVE_TOKEN_URL": "https://api.example/archive-upload-token",
        "FIREVIEWER_MAP_ARCHIVE_READY_URL": "https://api.example/archive-ready",
        "FIREVIEWER_MAP_CALLBACK_TOKEN": "b" * 64,
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


def test_callback_marks_private_archive_ready_immediately_after_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _environment(monkeypatch)
    archive = tmp_path / "fireviewer-zone.zip"
    archive.write_bytes(b"standalone-scene")
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    pathname = (
        "firewarning/map-production/jobs/map-"
        + "a" * 32
        + f"/archive/{sha256}/fireviewer-zone.zip"
    )
    observed: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        observed.append((str(request.url), payload))
        if request.url.path.endswith("/archive-upload-token"):
            return httpx.Response(
                200,
                json={
                    "schema": worker.ARCHIVE_UPLOAD_SCHEMA,
                    "pathname": pathname,
                    "upload_required": False,
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
    callback.upload_archive(archive, archive.stat().st_size, sha256)

    assert [url.rsplit("/", 1)[-1] for url, _payload in observed] == [
        "archive-upload-token",
        "archive-ready",
    ]
    assert observed[-1][1] == {
        "provider": "vercel_blob_private",
        "pathname": pathname,
        "byte_count": archive.stat().st_size,
        "sha256": sha256,
    }
    assert callback.archive_delivery == observed[-1][1]


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


def test_job_publishes_result_without_captures_and_cleans_local_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
            self.archive_delivery: dict[str, Any] | None = None

        def fetch_request(self) -> dict[str, Any]:
            return dict(REQUEST)

        def progress(self, fraction: float, message: str, **details: Any) -> None:
            self.progress_records.append(
                {"fraction": fraction, "message": message, **details}
            )

        def result(self, payload: dict[str, Any]) -> None:
            self.result_record = payload

        def upload_archive(self, archive: Path, byte_count: int, sha256: str) -> None:
            assert archive.read_bytes() == b"zip"
            self.archive_delivery = {
                "provider": "vercel_blob_private",
                "pathname": (
                    f"firewarning/map-production/jobs/{self.job_id}/archive/"
                    f"{sha256}/fireviewer-zone.zip"
                ),
                "byte_count": byte_count,
                "sha256": sha256,
            }

    callback = FakeCallback()

    class FakeEngine:
        asset_library_payload: dict[str, Any] = {}

        def __init__(self, received: ProductionConfig) -> None:
            assert received is config

        def run(
            self,
            *_args: Any,
            progress_callback: Any,
            archive_ready_callback: Any,
            **_kwargs: Any,
        ) -> Any:
            progress_callback(0.5, "Tuile 1/1 — terrain")
            root = scratch / "jobs" / "GPS-TEST"
            root.mkdir(parents=True)
            archive = root / "fireviewer-zone.zip"
            archive.write_bytes(b"zip")
            archive_ready_callback(archive, 3, hashlib.sha256(b"zip").hexdigest())
            (root / "dataset-publication.json").write_text(
                json.dumps(
                    {
                        "captures": [],
                        "status": "published_private",
                        "path_in_repo": "zones/GPS-TEST/build",
                        "dataset_id": "fireviewer/simple-measured-scenes-v1",
                        "commit_oid": "d" * 40,
                        "archive_sha256": "e" * 64,
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
            yield "Terminé", str(archive), []

    monkeypatch.setattr(worker, "CallbackClient", lambda: callback)
    monkeypatch.setattr(worker.ProductionConfig, "from_environment", lambda: config)
    monkeypatch.setattr(worker, "ProductionEngine", FakeEngine)
    monkeypatch.setattr(
        worker, "normalize_fixed_assets", lambda request, _catalog: request
    )
    result = worker.run()
    assert result["schema"] == worker.RESULT_SCHEMA
    assert result["captures"] == []
    assert result["archive"]["provider"] == "vercel_blob_private"
    assert result["archive"]["byte_count"] == 3
    assert callback.result_record == result
    assert callback.progress_records[-1]["state"] == "completed"
    assert not (scratch / "jobs" / "GPS-TEST").exists()
