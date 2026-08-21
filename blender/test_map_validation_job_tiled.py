from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import map_validation_job


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

    receipt, oracle, tiled_receipt, viewer = map_validation_job._run_viewer_export(
        SimpleNamespace(blender=tmp_path / "blender"),
        tmp_path,
        {"tile_count": tile_count},
    )

    assert calls == ["build_tiled", "validate_tiled"]
    assert receipt is None
    assert oracle == {"status": "skipped", "policy": policy, "reason": reason}
    assert tiled_receipt == {"status": "complete"}
    assert viewer["catalog_path"] == "viewer-tiled/catalog.json"
