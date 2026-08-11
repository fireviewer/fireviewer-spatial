from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import rasterio

from prepare_global_05m import (
    MID_PACKAGE_TERRAIN_CONTRACT,
    NEAR_ORTHOPHOTO_MAX_TILES_PER_RUN,
    NEAR_ORTHOPHOTO_RESOLUTION_M,
    RECEIPT_SCHEMA,
    SCHEMA,
    Global05mConfig,
    _asset_lock,
    _atomic_write_json,
    _lock_owner_is_alive,
    _selected_tiles,
    build_plan,
    completion_receipt,
    ensure_elevation_source,
    ensure_near_orthophoto_tile,
    ensure_orthophoto_tile,
    execute_manifest,
    execute_near_orthophoto_manifest,
    ign_tile_bounds,
    ign_tile_id,
    main,
    refresh_manifest,
)


def _write_aoi(path: Path, bounds: tuple[float, float, float, float]) -> None:
    min_x, min_y, max_x, max_y = bounds
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [min_x, min_y],
                                    [max_x, min_y],
                                    [max_x, max_y],
                                    [min_x, max_y],
                                    [min_x, min_y],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_ign_tile_id_uses_north_edge_index() -> None:
    assert ign_tile_id(886, 6400) == "0886_6401"
    assert ign_tile_bounds("0886_6401") == [
        886_000.0,
        6_400_000.0,
        887_000.0,
        6_401_000.0,
    ]


def test_plan_builds_500m_cores_with_10m_halo_and_relative_assets(
    tmp_path: Path,
) -> None:
    aoi = tmp_path / "aoi.geojson"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 887_100.0, 6_401_100.0))

    plan = build_plan(
        aoi,
        tmp_path / "production",
        config=Global05mConfig(output_tile_size_m=500, halo_m=10.0),
        expected_source_tile_count=4,
    )

    assert plan["schema"] == SCHEMA
    assert plan["summary"] == {
        "source_tile_count": 4,
        "output_tile_count": 9,
        "output_states": {"pending": 9},
        "elevation_source_request_count": 8,
        "orthophoto_request_count": 9,
        "network_access_performed": False,
    }
    tile = next(item for item in plan["tiles"] if item["id"] == "x886500_y6400500_s500")
    assert tile["bounds_l93_m"] == [886_500.0, 6_400_500.0, 887_000.0, 6_401_000.0]
    assert tile["processing_bounds_l93_m"] == [
        886_490.0,
        6_400_490.0,
        887_010.0,
        6_401_010.0,
    ]
    assert tile["source_tile_ids"] == [
        "0886_6401",
        "0886_6402",
        "0887_6401",
        "0887_6402",
    ]
    assert tile["visibility"]["default_visible"] is False
    assert tile["assets"]["mid_package"]["path"].startswith("tiles/")
    assert not Path(tile["assets"]["mid_package"]["path"]).is_absolute()
    assert tile["assets"]["near_orthophoto_source"]["required"] is False
    assert tile["assets"]["near_orthophoto_source"]["path"].endswith(
        "orthophoto-0m20.source.json"
    )
    request = parse_qs(urlparse(tile["orthophoto_request"]["url"]).query)
    assert [float(value) for value in request["BBOX"][0].split(",")] == pytest.approx(
        tile["bounds_l93_m"]
    )
    assert plan["worker_contract"]["orthophoto_display"] == {
        "source_geotiff_transform": "none",
        "jpeg_quality": 92,
        "jpeg_brightness": 0.78,
        "jpeg_contrast": 1.08,
        "jpeg_saturation": 1.12,
    }
    near_request = tile["near_orthophoto_request"]
    assert near_request["resolution_m"] == NEAR_ORTHOPHOTO_RESOLUTION_M
    assert near_request["pixel_size"] == [2500, 2500]
    near_query = parse_qs(urlparse(near_request["url"]).query)
    assert near_query["WIDTH"] == ["2500"]
    assert near_query["HEIGHT"] == ["2500"]
    assert plan["worker_contract"]["orthophoto_lod"] == {
        "far_global_resolution_m": 2.0,
        "mid_tile_resolution_m": 0.5,
        "near_tile_resolution_m": 0.2,
        "near_loading": "optional_explicit_selected_tiles_only",
        "near_fallback": "mid_tile_orthophoto",
    }
    assert plan["tiling"]["terrain_edge_contract"] == MID_PACKAGE_TERRAIN_CONTRACT
    assert (
        plan["worker_contract"]["mid_package_terrain_contract"]
        == MID_PACKAGE_TERRAIN_CONTRACT
    )


def test_completion_receipt_and_resume_require_hashes_for_every_required_asset(
    tmp_path: Path,
) -> None:
    aoi = tmp_path / "aoi.geojson"
    root = tmp_path / "production"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))
    plan = build_plan(aoi, root, expected_source_tile_count=1)
    tile = plan["tiles"][0]
    for name in (
        "mid_package",
        "orthophoto_source",
        "orthophoto_image",
        "orthophoto_geotiff",
    ):
        path = root / tile["assets"][name]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"validated-{name}".encode())
    receipt = completion_receipt(tile, root, producer="offline-test")
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["mid_package_terrain_contract"] == MID_PACKAGE_TERRAIN_CONTRACT
    receipt_path = root / tile["assets"]["completion_receipt"]["path"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    refresh_manifest(plan, root)
    assert plan["status"] == "ready"
    assert plan["tiles"][0]["status"]["state"] == "ready"

    (root / tile["assets"]["orthophoto_image"]["path"]).write_bytes(b"corrupt")
    refresh_manifest(plan, root)
    assert plan["status"] == "in_progress"
    assert plan["tiles"][0]["status"]["state"] == "incomplete"
    assert "hash mismatch" in plan["tiles"][0]["status"]["last_error"]


def test_cli_dry_run_prints_complete_plan_without_creating_output(
    tmp_path: Path, capsys: object
) -> None:
    aoi = tmp_path / "aoi.geojson"
    root = tmp_path / "must-not-exist"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))

    assert (
        main(
            [
                "--aoi",
                str(aoi),
                "--output-root",
                str(root),
                "--expected-source-tile-count",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload["schema"] == SCHEMA
    assert payload["summary"]["network_access_performed"] is False
    assert not root.exists()


def test_shared_elevation_cache_is_validated_and_downloaded_once(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    source = {
        "id": "0886_6401",
        "assets": {
            "mnt": {
                "path": "sources/mnt/LHD_FXX_0886_6401_MNT.tif",
                "url": "https://example.invalid/mnt",
            }
        },
    }

    def fetch(url: str, destination: Path, _timeout_s: float) -> None:
        calls.append(url)
        assert destination.name.endswith(".part")
        destination.write_bytes(b"validated-source")

    def validate(path: Path, root: Path, product: str, tile_id: str) -> dict[str, str]:
        assert path.read_bytes() == b"validated-source"
        assert root == tmp_path / "sources"
        assert (product, tile_id) == ("mnt", "0886_6401")
        return {"status": "valid"}

    first = ensure_elevation_source(
        source, "mnt", tmp_path, fetcher=fetch, validator=validate
    )
    second = ensure_elevation_source(
        source, "mnt", tmp_path, fetcher=fetch, validator=validate
    )
    assert first["cache"] == "downloaded"
    assert second["cache"] == "hit"
    assert calls == ["https://example.invalid/mnt"]
    assert not list((tmp_path / "sources").rglob("*.part"))


def test_global_only_source_phase_downloads_sources_without_detail_tiles(
    tmp_path: Path,
) -> None:
    aoi = tmp_path / "aoi.geojson"
    root = tmp_path / "production"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))
    manifest = build_plan(aoi, root, expected_source_tile_count=1, source_only=True)
    manifest_path = root / "production-manifest.json"
    _atomic_write_json(manifest_path, manifest, overwrite=False)
    fetched: list[str] = []

    def fetch(url: str, destination: Path, _timeout_s: float) -> None:
        fetched.append(url)
        destination.write_bytes(b"synthetic-lidar")

    def validate(
        _path: Path, _root: Path, product: str, _tile_id: str
    ) -> dict[str, str]:
        return {"product": product, "status": "valid"}

    result = execute_manifest(
        manifest_path,
        aoi,
        phase="sources",
        minimum_free_gib=0.0,
        source_fetcher=fetch,
        source_validator=validate,
    )

    assert manifest["tiles"] == []
    assert manifest["tiling"]["detail_grid_planned"] is False
    assert result["selected_tile_count"] == 0
    assert result["elevation_downloaded_or_validated"] == 2
    assert len(fetched) == 2
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["source_tiles"][0]["status"]["state"] == "ready"


def test_terrain_only_execution_fails_closed_without_accepted_ground_gate(
    tmp_path: Path,
) -> None:
    aoi = tmp_path / "aoi.geojson"
    root = tmp_path / "production"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))
    manifest = build_plan(
        aoi,
        root,
        expected_source_tile_count=1,
        terrain_only=True,
    )
    manifest_path = root / "production-manifest.json"
    _atomic_write_json(manifest_path, manifest, overwrite=False)

    with pytest.raises(RuntimeError, match="ground surface atlas"):
        execute_manifest(
            manifest_path,
            aoi,
            phase="sources",
            minimum_free_gib=0.0,
        )


def test_asset_lock_reclaims_a_dead_process_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.tif"
    lock = tmp_path / ".source.tif.lock"
    lock.write_text("pid=99999999\n", encoding="ascii")
    monkeypatch.setattr("prepare_global_05m._lock_owner_is_alive", lambda _path: False)

    with _asset_lock(target, timeout_s=0.1):
        assert lock.read_text(encoding="ascii") == f"pid={os.getpid()}\n"

    assert not lock.exists()


def test_lock_owner_probe_is_conservative_for_malformed_content(
    tmp_path: Path,
) -> None:
    lock = tmp_path / ".source.tif.lock"
    lock.write_text("not-a-pid\n", encoding="ascii")

    assert _lock_owner_is_alive(lock) is None


def test_orthophoto_cache_is_atomic_resumable_and_uses_visual_contract(
    tmp_path: Path,
) -> None:
    aoi = tmp_path / "aoi.geojson"
    root = tmp_path / "production"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))
    tile = build_plan(aoi, root, expected_source_tile_count=1)["tiles"][0]
    calls: list[str] = []

    def fetch(url: str, destination: Path, _timeout_s: float) -> None:
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        width = int(query["WIDTH"][0])
        height = int(query["HEIGHT"][0])
        bounds = tuple(float(value) for value in query["BBOX"][0].split(","))
        data = np.empty((3, height, width), dtype=np.uint8)
        data[0].fill(80)
        data[1].fill(120)
        data[2].fill(60)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs="EPSG:2154",
            transform=rasterio.transform.from_bounds(*bounds, width, height),
        ) as dataset:
            dataset.write(data)

    first = ensure_orthophoto_tile(tile, root, fetcher=fetch)
    second = ensure_orthophoto_tile(tile, root, fetcher=fetch)
    assert first["cache"] == "downloaded"
    assert second["cache"] == "hit"
    assert len(calls) == 1
    display = first["source"]["jpeg_display_transform"]
    assert display["brightness_multiplier"] == 0.78
    assert display["contrast_multiplier"] == 1.08
    assert display["saturation_multiplier"] == 1.12
    assert first["source"]["outputs"][0]["display_transform"] == (
        "none_source_mosaic_pixels_unchanged"
    )
    assert not list(root.rglob(".orthophoto-work-*"))


def test_native_near_orthophoto_is_optional_atomic_and_resumable(
    tmp_path: Path,
) -> None:
    aoi = tmp_path / "aoi.geojson"
    root = tmp_path / "production"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))
    tile = build_plan(aoi, root, expected_source_tile_count=1)["tiles"][0]
    for key in (
        "near_orthophoto_source",
        "near_orthophoto_image",
        "near_orthophoto_geotiff",
    ):
        del tile["assets"][key]
    del tile["near_orthophoto_request"]
    # Keep the synthetic raster small while exercising the same 20 cm profile.
    tile["bounds_l93_m"] = [886_000.0, 6_400_000.0, 886_020.0, 6_400_020.0]
    calls: list[str] = []

    def fetch(url: str, destination: Path, _timeout_s: float) -> None:
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        width = int(query["WIDTH"][0])
        height = int(query["HEIGHT"][0])
        bounds = tuple(float(value) for value in query["BBOX"][0].split(","))
        data = np.full((3, height, width), 96, dtype=np.uint8)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs="EPSG:2154",
            transform=rasterio.transform.from_bounds(*bounds, width, height),
        ) as dataset:
            dataset.write(data)

    first = ensure_near_orthophoto_tile(tile, root, fetcher=fetch)
    second = ensure_near_orthophoto_tile(tile, root, fetcher=fetch)

    assert first["cache"] == "downloaded"
    assert second["cache"] == "hit"
    assert len(calls) == 1
    request = parse_qs(urlparse(calls[0]).query)
    assert request["WIDTH"] == ["100"]
    assert request["HEIGHT"] == ["100"]
    assert first["source"]["request"]["nominal_resolution_m"] == 0.2
    assert tile["assets"]["near_orthophoto_source"]["required"] is False
    assert not list(root.rglob(".near-orthophoto-work-*"))


def test_near_batch_requires_explicit_all_tiles_opt_in(tmp_path: Path) -> None:
    identifiers = [
        f"tile_{index:02d}" for index in range(NEAR_ORTHOPHOTO_MAX_TILES_PER_RUN + 1)
    ]
    with pytest.raises(ValueError, match="16-tile safety limit"):
        execute_near_orthophoto_manifest(
            tmp_path / "missing.json",
            tile_ids=identifiers,
            minimum_free_gib=0.0,
        )


def test_near_batch_all_tiles_is_idempotent_and_updates_optional_assets(
    tmp_path: Path,
) -> None:
    aoi = tmp_path / "aoi.geojson"
    root = tmp_path / "production"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))
    manifest = build_plan(aoi, root, expected_source_tile_count=1)
    tile = manifest["tiles"][0]
    tile["status"]["state"] = "ready"
    tile["bounds_l93_m"] = [886_000.0, 6_400_000.0, 886_020.0, 6_400_020.0]
    for key in (
        "near_orthophoto_source",
        "near_orthophoto_image",
        "near_orthophoto_geotiff",
    ):
        del tile["assets"][key]
    del tile["near_orthophoto_request"]
    manifest_path = root / "production-manifest.json"
    _atomic_write_json(manifest_path, manifest, overwrite=False)
    calls: list[str] = []

    def fetch(url: str, destination: Path, _timeout_s: float) -> None:
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        width = int(query["WIDTH"][0])
        height = int(query["HEIGHT"][0])
        bounds = tuple(float(value) for value in query["BBOX"][0].split(","))
        data = np.full((3, height, width), 128, dtype=np.uint8)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs="EPSG:2154",
            transform=rasterio.transform.from_bounds(*bounds, width, height),
        ) as dataset:
            dataset.write(data)

    first = execute_near_orthophoto_manifest(
        manifest_path,
        all_tiles=True,
        minimum_free_gib=0.0,
        fetcher=fetch,
    )
    second = execute_near_orthophoto_manifest(
        manifest_path,
        all_tiles=True,
        minimum_free_gib=0.0,
        fetcher=fetch,
    )

    assert first["completed_or_validated"] == 1
    assert first["cache_hits"] == 0
    assert second["completed_or_validated"] == 1
    assert second["cache_hits"] == 1
    assert len(calls) == 1
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_tile = stored["tiles"][0]
    assert stored_tile["near_orthophoto_status"]["state"] == "ready"
    assert stored_tile["assets"]["near_orthophoto_source"]["exists"] is True
    assert stored["near_orthophoto_profile"]["ready_tile_count"] == 1


def test_failed_tiles_are_only_selected_with_retry_flag(tmp_path: Path) -> None:
    aoi = tmp_path / "aoi.geojson"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))
    manifest = build_plan(aoi, tmp_path / "production", expected_source_tile_count=1)
    manifest["tiles"][0]["status"]["state"] = "failed"
    assert _selected_tiles(manifest, (), max_tiles=None, retry_failed=False) == []
    assert (
        _selected_tiles(manifest, (), max_tiles=None, retry_failed=True)
        == manifest["tiles"]
    )


def test_resume_recovers_an_interrupted_running_tile(tmp_path: Path) -> None:
    aoi = tmp_path / "aoi.geojson"
    root = tmp_path / "production"
    _write_aoi(aoi, (886_100.0, 6_400_100.0, 886_400.0, 6_400_400.0))
    manifest = build_plan(aoi, root, expected_source_tile_count=1)
    tile = manifest["tiles"][0]
    tile["status"].update({"state": "running", "last_error": None})

    refresh_manifest(manifest, root)

    assert tile["status"]["state"] == "pending"
    assert tile["status"]["last_error"] == (
        "interrupted before completion receipt; eligible for resume"
    )
    assert _selected_tiles(manifest, (), max_tiles=None, retry_failed=False) == [tile]


def test_explicit_mid_rebuild_includes_ready_tiles() -> None:
    manifest = {
        "tiles": [
            {"id": "ready", "status": {"state": "ready"}},
            {"id": "pending", "status": {"state": "pending"}},
        ]
    }
    assert (
        _selected_tiles(
            manifest,
            (),
            max_tiles=None,
            retry_failed=False,
            include_ready=True,
        )
        == manifest["tiles"]
    )
    assert _selected_tiles(
        manifest,
        ("ready",),
        max_tiles=None,
        retry_failed=False,
        include_ready=True,
    ) == [manifest["tiles"][0]]


def test_mid_rebuild_is_restricted_to_offline_tile_phase(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires phase='tiles'"):
        execute_manifest(
            tmp_path / "missing-manifest.json",
            tmp_path / "missing-aoi.geojson",
            rebuild_mid_packages=True,
        )


def test_atomic_manifest_refuses_overwrite_without_explicit_permission(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest.json"
    _atomic_write_json(destination, {"first": True}, overwrite=False)
    try:
        _atomic_write_json(destination, {"second": True}, overwrite=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing production manifest was overwritten")
    assert json.loads(destination.read_text(encoding="utf-8")) == {"first": True}
