from __future__ import annotations

import modal_map_production as production
import pytest


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
