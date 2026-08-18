from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import lightning_perimeter_production as worker


JOB_ID = "perimeter-" + "a" * 32
REQUEST = {
    "source": {
        "schema": "fireviewer.geographic-perimeter-source.v1",
        "dataset_id": "observed-test",
        "frames": [],
    },
    "base_map": {
        "job_id": "map-" + "b" * 32,
        "zone_id": "GPS-TEST",
        "build_id": "c" * 64,
        "dataset": {
            "repo_id": "fireviewer/simple-measured-scenes-v1",
            "revision": "d" * 40,
            "root": "zones/GPS-TEST/build",
        },
        "archive": {
            "path": "zones/GPS-TEST/build/fireviewer-zone.zip",
            "sha256": "e" * 64,
        },
    },
}


def _environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "FIREVIEWER_PERIMETER_JOB_ID": JOB_ID,
        "FIREVIEWER_PERIMETER_REQUEST_URL": "https://api.example/request",
        "FIREVIEWER_PERIMETER_PROGRESS_URL": "https://api.example/progress",
        "FIREVIEWER_PERIMETER_RESULT_URL": "https://api.example/result",
        "FIREVIEWER_PERIMETER_CALLBACK_TOKEN": "f" * 64,
        "FIREVIEWER_WORK_ROOT": str(tmp_path / "work"),
        "FIREVIEWER_HF_DATASET_ID": "fireviewer/simple-measured-scenes-v1",
        "HF_TOKEN": "hf-test-token-not-a-secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_callback_rejects_unsafe_job_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(monkeypatch, tmp_path)
    monkeypatch.setenv("FIREVIEWER_PERIMETER_JOB_ID", "../outside")
    with pytest.raises(worker.LightningPerimeterContractError, match="Identifiant"):
        worker.CallbackClient()


def test_worker_publishes_timeline_and_cleans_scratch_without_blocking_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(monkeypatch, tmp_path)

    class FakeCallback:
        job_id = JOB_ID

        def __init__(self) -> None:
            self.progress_records: list[dict[str, Any]] = []
            self.result_record: dict[str, Any] | None = None

        def fetch_request(self) -> tuple[dict[str, Any], str]:
            return REQUEST, hashlib.sha256(worker._canonical_bytes(REQUEST)).hexdigest()

        def progress(self, fraction: float, message: str, **details: Any) -> None:
            self.progress_records.append(
                {"fraction": fraction, "message": message, **details}
            )

        def result(self, payload: dict[str, Any]) -> None:
            self.result_record = payload

    callback = FakeCallback()
    package_root = tmp_path / "layer"
    package_root.mkdir()
    archive = tmp_path / "perimeter.zip"
    archive.write_bytes(b"perimeter-zip")
    map_archive = tmp_path / "map.zip"
    map_archive.write_bytes(b"map-zip")
    map_reference = SimpleNamespace(zone_id="GPS-TEST", map_build_id="c" * 64)
    viewer = SimpleNamespace(root=tmp_path / "viewer", frames=(), manifest={})

    monkeypatch.setattr(worker, "CallbackClient", lambda: callback)
    monkeypatch.setattr(
        worker,
        "produce_perimeter_layer",
        lambda _source, _root: SimpleNamespace(
            package_root=package_root,
            manifest={"build_id": "1" * 64},
        ),
    )
    monkeypatch.setattr(worker, "_download_map", lambda *_args: map_archive)
    monkeypatch.setattr(
        worker, "read_map_reference_from_archive", lambda _archive: map_reference
    )
    monkeypatch.setattr(
        worker, "build_perimeter_timeline_viewer", lambda *_args: viewer
    )
    monkeypatch.setattr(
        worker,
        "materialize_perimeter_upload_package",
        lambda *_args, **_kwargs: (
            tmp_path / "upload",
            archive,
            {"timeline": {"state_count": 1}},
        ),
    )
    monkeypatch.setattr(worker, "validate_perimeter_upload_package", lambda _root: None)
    monkeypatch.setattr(
        worker,
        "_publish",
        lambda **_kwargs: (
            "2" * 40,
            f"perimeters/{'1' * 64}/{'c' * 64}",
            [
                {
                    "index": 0,
                    "observed_at": "2026-08-15T12:00:00Z",
                    "caption": "État observé",
                    "path": "viewer/frame-0000.glb",
                    "sha256": "3" * 64,
                    "byte_count": 128,
                }
            ],
        ),
    )

    result = worker.run()

    assert result["schema"] == worker.RESULT_SCHEMA
    assert result["state_count"] == 1
    assert callback.result_record == result
    assert callback.progress_records[-1]["state"] == "completed"
    assert not (tmp_path / "work" / "jobs" / JOB_ID).exists()


def test_cleanup_failure_does_not_invalidate_published_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(monkeypatch, tmp_path)
    cleanup_calls = 0

    class FakeCallback:
        job_id = JOB_ID

        def fetch_request(self) -> tuple[dict[str, Any], str]:
            return REQUEST, "4" * 64

        def progress(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def result(self, _payload: dict[str, Any]) -> None:
            return None

    archive = tmp_path / "perimeter.zip"
    archive.write_bytes(b"perimeter-zip")
    map_archive = tmp_path / "map.zip"
    map_archive.write_bytes(b"map-zip")
    layer = tmp_path / "layer"
    layer.mkdir()
    monkeypatch.setattr(worker, "CallbackClient", FakeCallback)
    monkeypatch.setattr(
        worker,
        "produce_perimeter_layer",
        lambda *_args: SimpleNamespace(
            package_root=layer, manifest={"build_id": "1" * 64}
        ),
    )
    monkeypatch.setattr(worker, "_download_map", lambda *_args: map_archive)
    monkeypatch.setattr(
        worker,
        "read_map_reference_from_archive",
        lambda _archive: SimpleNamespace(zone_id="GPS-TEST", map_build_id="c" * 64),
    )
    monkeypatch.setattr(
        worker,
        "build_perimeter_timeline_viewer",
        lambda *_args: SimpleNamespace(
            root=tmp_path / "viewer", frames=(), manifest={}
        ),
    )
    monkeypatch.setattr(
        worker,
        "materialize_perimeter_upload_package",
        lambda *_args, **_kwargs: (
            tmp_path / "upload",
            archive,
            {"timeline": {"state_count": 0}},
        ),
    )
    monkeypatch.setattr(worker, "validate_perimeter_upload_package", lambda _root: None)
    monkeypatch.setattr(
        worker,
        "_publish",
        lambda **_kwargs: ("2" * 40, "perimeters/test", []),
    )

    original_rmtree = worker.shutil.rmtree

    def fail_final_cleanup(path: Path) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if Path(path).name == JOB_ID:
            raise OSError("late cleanup failure")
        original_rmtree(path)

    monkeypatch.setattr(worker.shutil, "rmtree", fail_final_cleanup)
    assert worker.run()["status"] == "technical_perimeter_produced"
    assert cleanup_calls == 1
