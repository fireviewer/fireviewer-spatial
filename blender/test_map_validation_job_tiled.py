from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import map_validation_job


def test_fast_finalize_stops_before_portable_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def expensive_seal(_root: Path) -> None:
        nonlocal called
        called = True

    monkeypatch.setenv("FIREVIEWER_FAST_FINALIZE", "1")
    monkeypatch.setattr(
        map_validation_job.production_engine,
        "seal_map_upload_package",
        expensive_seal,
    )
    with pytest.raises(map_validation_job._SealedFolderReady):
        with map_validation_job._sealed_folder_mode():
            map_validation_job.production_engine.seal_map_upload_package(tmp_path)
    assert called is False


@pytest.mark.parametrize(
    ("policy", "tile_count", "reason"),
    (("off", 9, "disabled"), ("auto", 625, "tile_count_exceeds_9")),
)
def test_tiled_viewer_is_primary_and_monolithic_can_be_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    tile_count: int,
    reason: str,
) -> None:
    calls: list[str] = []

    def build(*_args: object, **_kwargs: object) -> None:
        calls.append("build_tiled")

    def validate(*_args: object, **_kwargs: object):
        calls.append("validate_tiled")
        return {"status": "complete"}, {"catalog_path": "viewer-tiled/catalog.json"}

    monkeypatch.setattr(map_validation_job, "build_tiled_viewer_package", build)
    monkeypatch.setattr(map_validation_job, "validate_tiled_viewer_package", validate)
    monkeypatch.setattr(
        map_validation_job.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("monolithic exporter was launched"),
    )
    monkeypatch.setenv("FIREVIEWER_VALIDATION_MONOLITHIC_VIEWER", policy)

    timings: dict[str, float] = {}
    receipt, oracle, tiled_receipt, viewer = map_validation_job._run_viewer_export(
        SimpleNamespace(blender=tmp_path / "blender"),
        tmp_path,
        {"tile_count": tile_count},
        timings,
    )

    assert calls == ["build_tiled", "validate_tiled"]
    assert receipt is None
    assert oracle == {"status": "skipped", "policy": policy, "reason": reason}
    assert tiled_receipt == {"status": "complete"}
    assert viewer["catalog_path"] == "viewer-tiled/catalog.json"
    assert set(timings) == {"tiled_build", "tiled_validation"}
    assert all(value >= 0.0 for value in timings.values())
