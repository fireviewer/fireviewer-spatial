from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import runpod_map_production as worker


class FakeEngine:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_side_m=15_000, max_tiles=900, tile_workers=4)
        self.asset_summary = {
            "asset_count": 307,
            "catalog_revision": "a" * 64,
        }
        self.fixed_asset_choices = (("Église", "church"),)
        self.asset_library_payload = {
            "asset_count": 1,
            "assets": [{"asset_id": "church", "category": "building"}],
        }


def test_normalize_request_is_bounded_and_stable() -> None:
    request = worker.normalize_map_request(
        {"latitude": 44.74412, "longitude": 5.35025, "side_km": 10}
    )
    assert request == {
        "latitude": 44.74412,
        "longitude": 5.35025,
        "side_km": 10.0,
        "fixed_asset_placements": None,
    }
    assert worker.request_sha256(request) == worker.request_sha256(dict(request))
    with pytest.raises(worker.RunPodMapContractError, match="0,5 et 15"):
        worker.normalize_map_request(
            {"latitude": 44.0, "longitude": 5.0, "side_km": 20}
        )


def test_handler_exposes_runpod_config_without_modal_capability(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_engine", FakeEngine)
    payload = worker.handler({"id": "local_test", "input": {"operation": "config"}})
    assert payload["limits"]["parallel_tile_workers"] == 4
    assert payload["assets"]["count"] == 307
    assert payload["capabilities"]["provider_runpod_serverless"] is True
    assert "provider_modal" not in payload["capabilities"]


def test_handler_rejects_unknown_operations(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_engine", FakeEngine)
    with pytest.raises(worker.RunPodMapContractError, match="inconnue"):
        worker.handler({"id": "local_test", "input": {"operation": "delete_all"}})


def test_production_result_is_capture_free(tmp_path, monkeypatch) -> None:
    job_root = tmp_path / "job"
    archive = job_root / "fireviewer-zone.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"portable-zone")
    (job_root / "dataset-publication.json").write_text(
        json.dumps(
            {
                "dataset_id": "fireviewer/simple-measured-scenes-v1",
                "commit_oid": "d" * 40,
                "path_in_repo": "zones/test/build",
                "archive_sha256": "e" * 64,
                "captures": [],
            }
        ),
        encoding="utf-8",
    )
    (job_root / "zone.done.json").write_text(
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

    class ProducingEngine(FakeEngine):
        def run(self, *_args, **_kwargs):
            yield "Terminé — scène autonome sans captures.", str(archive), []

    monkeypatch.setattr(worker, "_engine", ProducingEngine)
    result = worker.handler(
        {
            "id": "local_test",
            "input": {
                "operation": "produce_map",
                "request": {
                    "latitude": 43.9,
                    "longitude": 4.5,
                    "side_km": 0.5,
                },
            },
        }
    )

    assert result["schema"] == worker.RESULT_SCHEMA
    assert result["captures"] == []
    assert result["archive"]["path"].endswith("/fireviewer-zone.zip")
