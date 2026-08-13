from __future__ import annotations

from typing import Any

import modal
import modal_map_production as production
import pytest
from fastapi.testclient import TestClient


class FakeStore:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value


class FakeProducer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], int]] = []

    def spawn(self, job_id: str, request: dict[str, Any], attempt: int) -> Any:
        self.calls.append((job_id, request, attempt))
        return type("FakeCall", (), {"object_id": f"fc-{attempt}"})()


def test_request_identity_and_public_status_have_no_captures() -> None:
    request = {
        "latitude": 43.90349754,
        "longitude": 4.49681631,
        "side_km": 0.5,
        "fixed_asset_placements": None,
    }
    job_id = production.job_id_for_request(request)
    assert job_id == production.job_id_for_request(
        dict(reversed(list(request.items())))
    )
    assert job_id.startswith("map_") and len(job_id) == 36
    status = production.initial_job_status(job_id, request)
    public = production.public_job_status(status)
    assert public["captures"] == []
    assert public["archive_url"] is None


def test_completed_status_exposes_only_the_zip() -> None:
    request = {"latitude": 44, "longitude": 5, "side_km": 1}
    job_id = production.job_id_for_request(request)
    status = production.initial_job_status(
        job_id, production.normalize_map_request(request)
    )
    status.update(state="completed", phase="completed", progress=1.0)
    public = production.public_job_status(status)
    assert public["archive_url"] == f"/v1/map-jobs/{job_id}/download-link"
    assert public["captures"] == []


def test_missing_legacy_call_is_exposed_as_stale_failed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    monkeypatch.setattr(production, "job_store", store)
    request = production.normalize_map_request(
        {"latitude": 44.73, "longitude": 5.33, "side_km": 2}
    )
    job_id = production.job_id_for_request(request)
    status = production.initial_job_status(job_id, request)
    status.update(
        state="running",
        phase="in_progress",
        created_at="2020-01-01T00:00:00Z",
        heartbeat_at="2020-01-01T00:00:00Z",
    )
    store[job_id] = status

    reconciled = production._reconcile_active_status(status)

    assert reconciled["state"] == "failed"
    assert reconciled["phase"] == "stale"
    assert "reprise disponible" in reconciled["message"]
    assert production.public_job_status(reconciled)["state"] == "failed"


def test_external_modal_cancellation_becomes_terminal_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    monkeypatch.setattr(production, "job_store", store)
    request = production.normalize_map_request(
        {"latitude": 44.73, "longitude": 5.33, "side_km": 2}
    )
    job_id = production.job_id_for_request(request)
    status = production.initial_job_status(job_id, request)
    status.update(state="running", phase="in_progress")
    store[job_id] = status
    store[production._call_key(job_id, 1)] = {"function_call_id": "fc-canceled"}

    def canceled(_function_call_id: str) -> None:
        raise modal.exception.RemoteError("")

    monkeypatch.setattr(production, "_poll_function_call", canceled)
    reconciled = production._reconcile_active_status(status)

    assert reconciled["state"] == "failed"
    assert reconciled["phase"] == "canceled"
    assert reconciled["finished_at"] is not None


@pytest.mark.parametrize("interrupted_phase", sorted(production.RESUMABLE_PHASES))
def test_repeat_submit_resumes_once_without_changing_public_contract(
    monkeypatch: pytest.MonkeyPatch, interrupted_phase: str
) -> None:
    store = FakeStore()
    producer = FakeProducer()
    monkeypatch.setattr(production, "job_store", store)
    monkeypatch.setattr(production, "produce_map", producer)
    monkeypatch.setenv("FIREVIEWER_API_TOKEN", "test-token")

    def still_active(_function_call_id: str) -> None:
        raise modal.exception.TimeoutError()

    monkeypatch.setattr(production, "_poll_function_call", still_active)
    client = TestClient(production.api.local())
    request = {"latitude": 44.73, "longitude": 5.33, "side_km": 2}
    headers = {"Authorization": "Bearer test-token"}

    first = client.post("/v1/map-jobs", json=request, headers=headers)
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    interrupted = dict(store[job_id])
    interrupted.update(
        state="failed",
        phase=interrupted_phase,
        progress=0.35,
        finished_at="2026-08-13T20:30:00Z",
    )
    store[job_id] = interrupted

    resumed = client.post("/v1/map-jobs", json=request, headers=headers)
    duplicate = client.post("/v1/map-jobs", json=request, headers=headers)

    assert resumed.status_code == 202
    assert resumed.json()["job_id"] == job_id
    assert resumed.json()["state"] == "queued"
    assert resumed.json()["phase"] == "resume_queued"
    assert resumed.json()["progress"] == 0.35
    assert duplicate.status_code == 202
    assert duplicate.json()["state"] == "queued"
    assert [attempt for _job, _request, attempt in producer.calls] == [1, 2]
    assert set(resumed.json()) == {
        "schema",
        "job_id",
        "kind",
        "request_sha256",
        "state",
        "phase",
        "progress",
        "message",
        "current_tile",
        "tile_count",
        "created_at",
        "started_at",
        "finished_at",
        "error",
        "archive_url",
        "captures",
    }


def test_asgi_submit_accepts_the_request_as_the_json_body() -> None:
    web = production.api.local()
    operation = web.openapi()["paths"]["/v1/map-jobs"]["post"]
    assert operation["requestBody"]["required"] is True
    assert [parameter["name"] for parameter in operation.get("parameters", [])] == [
        "authorization"
    ]
    response = TestClient(web).post(
        "/v1/map-jobs",
        json={"latitude": 44.73, "longitude": 5.33, "side_km": 2},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {"latitude": 91, "longitude": 0, "side_km": 1},
        {"latitude": 44, "longitude": 5, "side_km": 0.1},
        {"latitude": 44, "longitude": 5, "side_km": 1, "extra": True},
    ],
)
def test_request_validation_is_fail_closed(payload: dict[str, object]) -> None:
    with pytest.raises(production.ModalMapContractError):
        production.normalize_map_request(payload)
