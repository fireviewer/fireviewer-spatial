from __future__ import annotations

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
        self.asset_library_payload = {"assets": []}


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
