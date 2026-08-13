from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import simple_production_api as api
from fastapi.testclient import TestClient
from pyproj import Transformer
from simple_production_engine import ProductionConfig


class _FakeEngine:
    def __init__(self, config: ProductionConfig) -> None:
        self.config = config
        self.asset_summary = {
            "asset_count": 1,
            "catalog_revision": "a" * 64,
        }
        self.asset_library_payload = {
            "asset_count": 1,
            "assets": [{"asset_id": "asset-tree", "category": "tree"}],
        }
        self.fixed_asset_choices = (("tree — Chêne [asset-tree]", "asset-tree"),)

    def run(
        self,
        latitude: float,
        longitude: float,
        side_km: float,
        *,
        progress_callback,
        fixed_asset_placements,
    ):
        plan = api.plan_zone(latitude, longitude, side_km)
        if fixed_asset_placements["placements"]:
            plan_id = (
                f"{plan.zone_id}-fixed-"
                f"{api.fixed_asset_request_sha256(fixed_asset_placements)[:12]}"
            )
        else:
            plan_id = plan.zone_id
        root = self.config.work_root / "jobs" / plan_id
        root.mkdir(parents=True, exist_ok=True)
        progress_callback(0.25, "Sources validées")
        archive = root / "fireviewer-zone.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("zone.usda", "#usda 1.0\n")
        progress_callback(1.0, "Terminé")
        yield "Terminé — scène autonome sans captures.", str(archive), []


def _fixture(tmp_path: Path) -> tuple[TestClient, tuple[float, float]]:
    longitude, latitude = Transformer.from_crs(2154, 4326, always_xy=True).transform(
        820750, 6313250
    )
    config = ProductionConfig(
        work_root=tmp_path,
        portable_root=tmp_path,
        asset_library=tmp_path / "assets.json",
        review_batch=tmp_path / "assets",
        elevation_revision="elevation-v1",
        orthophoto_revision="ortho-v1",
        context_revision="context-v1",
    )
    return TestClient(api.create_app(config=config, engine=_FakeEngine(config))), (
        latitude,
        longitude,
    )


def test_api_is_headless_and_plans_exact_three_by_three(tmp_path: Path) -> None:
    client, (latitude, longitude) = _fixture(tmp_path)

    assert "gradio" not in sys.modules
    assert client.get("/healthz").json() == {
        "status": "ok",
        "schema": api.API_SCHEMA,
    }
    config = client.get("/v1/config").json()
    assert config["assets"]["count"] == 1
    assert config["assets"]["choices"] == [
        {"label": "tree — Chêne [asset-tree]", "asset_id": "asset-tree"}
    ]
    assert config["capabilities"]["human_auto_acceptance"] is False

    response = client.post(
        "/v1/plan",
        json={"latitude": latitude, "longitude": longitude, "side_km": 1.5},
    )
    assert response.status_code == 200
    plan = response.json()
    assert plan["tile_count"] == 9
    assert plan["production_area_km2"] == 2.25
    assert plan["production_bounds_l93_m"] == [820000, 6312500, 821500, 6314000]


def test_fixed_assets_are_validated_before_async_production(tmp_path: Path) -> None:
    client, (latitude, longitude) = _fixture(tmp_path)
    fixed = {
        "schema": "fireviewer.fixed-asset-placement-request.v1",
        "crs": "EPSG:4326",
        "placements": [
            {
                "placement_id": "church-1",
                "asset_id": "asset-tree",
                "latitude": latitude,
                "longitude": longitude,
                "yaw_deg": 12,
            }
        ],
    }
    checked = client.post(
        "/v1/fixed-assets/validate",
        json={
            "request": fixed,
            "latitude": latitude,
            "longitude": longitude,
            "side_km": 1.5,
        },
    )
    assert checked.status_code == 200
    assert checked.json()["placement_count"] == 1
    assert len(checked.json()["projected"]) == 1

    created = client.post(
        "/v1/jobs",
        json={
            "latitude": latitude,
            "longitude": longitude,
            "side_km": 1.5,
            "fixed_asset_placements": fixed,
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    for _attempt in range(100):
        payload = client.get(f"/v1/jobs/{job_id}").json()
        if payload["state"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert payload["state"] == "completed", payload
    assert payload["progress"] == 1.0
    assert payload["captures"] == []
    assert payload["archive_url"] == f"/v1/jobs/{job_id}/archive"
    assert client.get(payload["archive_url"]).status_code == 200


def test_api_rejects_invalid_inputs_auth_and_perimeter_type(
    tmp_path: Path, monkeypatch
) -> None:
    client, (latitude, longitude) = _fixture(tmp_path)
    assert (
        client.post(
            "/v1/plan", json={"latitude": 92, "longitude": 3, "side_km": 1}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/fixed-assets/validate",
            json={
                "request": {
                    "schema": "fireviewer.fixed-asset-placement-request.v1",
                    "crs": "EPSG:4326",
                    "placements": [
                        {
                            "placement_id": "outside",
                            "asset_id": "asset-tree",
                            "latitude": latitude + 1,
                            "longitude": longitude,
                            "yaw_deg": 0,
                        }
                    ],
                },
                "latitude": latitude,
                "longitude": longitude,
                "side_km": 1.5,
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/perimeter-jobs",
            files={"source": ("perimeter.txt", b"{}", "text/plain")},
        ).status_code
        == 422
    )

    monkeypatch.setenv("FIREVIEWER_API_TOKEN", "secret-test-token")
    protected, _coordinates = _fixture(tmp_path / "protected")
    assert protected.get("/healthz").status_code == 200
    assert protected.get("/v1/config").status_code == 401
    assert (
        protected.get(
            "/v1/config", headers={"Authorization": "Bearer secret-test-token"}
        ).status_code
        == 200
    )


def test_perimeter_upload_is_removed_after_the_worker_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    client, _coordinates = _fixture(tmp_path)

    def fake_produce(source: Path, work_root: Path) -> SimpleNamespace:
        assert source.is_file()
        package_root = work_root / "perimeters" / "fixture"
        package_root.mkdir(parents=True)
        archive = package_root / "perimeter.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("perimeter.usda", "#usda 1.0\n")
        return SimpleNamespace(archive=archive, package_root=package_root)

    monkeypatch.setattr(api, "produce_perimeter_layer", fake_produce)
    created = client.post(
        "/v1/perimeter-jobs",
        files={
            "source": (
                "perimeter.geojson",
                b'{"type":"FeatureCollection","features":[]}',
                "application/geo+json",
            )
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    for _attempt in range(100):
        payload = client.get(f"/v1/jobs/{job_id}").json()
        if payload["state"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert payload["state"] == "completed", payload
    assert not (tmp_path / "api-uploads").exists() or not any(
        (tmp_path / "api-uploads").iterdir()
    )


def test_contract_is_locked_and_contains_no_gradio_surface() -> None:
    contract = json.loads(
        Path(api.__file__)
        .with_name("simple_production_api_contract.v1.json")
        .read_text(encoding="utf-8")
    )
    assert contract["transport"] == {
        "framework": "fastapi",
        "database": "forbidden",
        "gradio": "forbidden",
        "job_state": "memory_plus_hash_locked_files_below_work_root",
    }
    assert contract["map_production"]["capture_count"] == 0
    assert contract["acceptance"]["automatic_human_acceptance"] is False
