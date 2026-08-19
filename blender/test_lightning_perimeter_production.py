from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import lightning_perimeter_production as worker


JOB_ID = "perimeter-" + "a" * 32
MAP_BUILD_ID = "c" * 64
MAP_SOURCE_ROOT = f"maps/GPS-TEST/{MAP_BUILD_ID}/source"
REQUEST = {
    "source": {
        "schema": "fireviewer.geographic-perimeter-source.v1",
        "dataset_id": "observed-test",
        "frames": [],
    },
    "base_map": {
        "job_id": "map-" + "b" * 32,
        "zone_id": "GPS-TEST",
        "build_id": MAP_BUILD_ID,
        "dataset": {
            "repo_id": "fireviewer/simple-measured-scenes-v1",
            "revision": "d" * 40,
            "root": MAP_SOURCE_ROOT,
            "visibility": "public",
        },
        "source": {
            "schema": "fireviewer.map-source-publication.v1",
            "status": "published_public",
            "root": MAP_SOURCE_ROOT,
            "visibility": "public",
            "manifest_sha256": "e" * 64,
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


def test_download_map_materializes_public_hf_folder_without_zip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "downloaded"
    source_root = local_root / MAP_SOURCE_ROOT
    source_root.mkdir(parents=True)
    reference = SimpleNamespace(
        zone_id="GPS-TEST",
        map_build_id=MAP_BUILD_ID,
        manifest_sha256="e" * 64,
    )
    observed: dict[str, Any] = {}

    def snapshot_download(**kwargs: Any) -> str:
        observed.update(kwargs)
        return str(local_root)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setattr(worker, "validate_map_upload_package", lambda root: reference)

    downloaded, returned, identity = worker._download_map(
        REQUEST["base_map"],
        tmp_path,
        "hf-token",
    )

    assert downloaded == source_root.resolve()
    assert returned is reference
    assert identity == {"source_manifest_sha256": "e" * 64}
    assert observed["repo_id"] == "fireviewer/simple-measured-scenes-v1"
    assert observed["revision"] == "d" * 40
    assert observed["allow_patterns"] == [f"{MAP_SOURCE_ROOT}/**"]
    assert not list(tmp_path.rglob("*.zip"))


def test_worker_publishes_timeline_and_cleans_scratch_without_map_zip(
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
    perimeter_archive = tmp_path / "perimeter.zip"
    perimeter_archive.write_bytes(b"perimeter-zip")
    map_folder = tmp_path / "map-folder"
    map_folder.mkdir()
    map_reference = SimpleNamespace(zone_id="GPS-TEST", map_build_id=MAP_BUILD_ID)
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
    monkeypatch.setattr(
        worker,
        "_download_map",
        lambda *_args: (
            map_folder,
            map_reference,
            {"source_manifest_sha256": "e" * 64},
        ),
    )
    monkeypatch.setattr(
        worker,
        "build_perimeter_timeline_viewer_for_map",
        lambda *_args: viewer,
    )
    monkeypatch.setattr(
        worker,
        "materialize_perimeter_upload_package",
        lambda *_args, **_kwargs: (
            tmp_path / "upload",
            perimeter_archive,
            {"timeline": {"state_count": 1}},
        ),
    )
    monkeypatch.setattr(worker, "validate_perimeter_upload_package", lambda _root: None)
    monkeypatch.setattr(
        worker,
        "_publish",
        lambda **_kwargs: (
            "2" * 40,
            f"perimeters/{'1' * 64}/{MAP_BUILD_ID}",
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
    assert result["base_map"]["source_manifest_sha256"] == "e" * 64
    assert "archive_sha256" not in result["base_map"]
    assert result["dataset"]["visibility"] == "public"
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

    perimeter_archive = tmp_path / "perimeter.zip"
    perimeter_archive.write_bytes(b"perimeter-zip")
    map_folder = tmp_path / "map-folder"
    map_folder.mkdir()
    layer = tmp_path / "layer"
    layer.mkdir()
    map_reference = SimpleNamespace(zone_id="GPS-TEST", map_build_id=MAP_BUILD_ID)

    monkeypatch.setattr(worker, "CallbackClient", FakeCallback)
    monkeypatch.setattr(
        worker,
        "produce_perimeter_layer",
        lambda *_args: SimpleNamespace(
            package_root=layer,
            manifest={"build_id": "1" * 64},
        ),
    )
    monkeypatch.setattr(
        worker,
        "_download_map",
        lambda *_args: (
            map_folder,
            map_reference,
            {"source_manifest_sha256": "e" * 64},
        ),
    )
    monkeypatch.setattr(
        worker,
        "build_perimeter_timeline_viewer_for_map",
        lambda *_args: SimpleNamespace(
            root=tmp_path / "viewer",
            frames=(),
            manifest={},
        ),
    )
    monkeypatch.setattr(
        worker,
        "materialize_perimeter_upload_package",
        lambda *_args, **_kwargs: (
            tmp_path / "upload",
            perimeter_archive,
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
