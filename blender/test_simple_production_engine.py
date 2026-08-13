from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import render_simple_zone_gallery as zone_gallery
import simple_production_engine as production
from fixed_terrain_grid import (
    compile_fixed_terrain_from_canonical_mm,
    write_fixed_terrain,
)
from pyproj import Transformer


def _pilot_gps() -> tuple[float, float]:
    longitude, latitude = Transformer.from_crs(2154, 4326, always_xy=True).transform(
        820250, 6312750
    )
    return latitude, longitude


def test_pilot_center_and_1_5_km_produce_the_exact_validated_3x3() -> None:
    latitude, longitude = _pilot_gps()
    plan = production.plan_zone(latitude, longitude, 1.5)
    assert plan.production_bounds_l93_m == (819500, 6312000, 821000, 6313500)
    assert len(plan.tiles) == 9
    assert [tile.origin_l93_m for tile in plan.tiles] == [
        (819500, 6313000),
        (820000, 6313000),
        (820500, 6313000),
        (819500, 6312500),
        (820000, 6312500),
        (820500, 6312500),
        (819500, 6312000),
        (820000, 6312000),
        (820500, 6312000),
    ]


def test_zone_stage_is_one_portable_entry_over_all_tiles(tmp_path: Path) -> None:
    latitude, longitude = _pilot_gps()
    plan = production.plan_zone(latitude, longitude, 1.5)
    stage = production._write_zone_stage(tmp_path, plan)
    text = stage.read_text(encoding="utf-8")
    assert 'defaultPrim = "FireViewerZone"' in text
    assert text.count("prepend references") == 9
    assert "packages/x819500_y6312000/scene/scene.usda" in text
    assert "double3 xformOp:translate = (0, 0, 0)" in text
    assert "double3 xformOp:translate = (1000, 1000, 0)" in text
    assert ":\\" not in text


def _fake_sources(root: Path) -> SimpleNamespace:
    root.mkdir(parents=True)
    files = {
        "mnt-05m.tif": b"mnt",
        "mns-05m.tif": b"mns",
        "orthophoto-1m.png": b"ortho",
        "elevation-source-05m.json": b"{}\n",
        "orthophoto-source.json": b"{}\n",
        "building-source.json": b"{}\n",
        "bdtopo-buildings.geojson": b'{"type":"FeatureCollection","features":[]}\n',
        "placement-context.json": b"{}\n",
        "simple-measured-tile-sources.v1.json": b"{}\n",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    return SimpleNamespace(
        root=root,
        mnt=root / "mnt-05m.tif",
        mns=root / "mns-05m.tif",
        orthophoto=root / "orthophoto-1m.png",
        elevation_receipt=root / "elevation-source-05m.json",
        orthophoto_receipt=root / "orthophoto-source.json",
        placement_context=root / "placement-context.json",
        reused=False,
    )


def test_engine_returns_full_pack_plus_unified_scene_and_removes_rasters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = production.ProductionConfig(
        work_root=tmp_path / "work",
        portable_root=tmp_path,
        asset_library=tmp_path / "assets" / "asset-library.v1.json",
        review_batch=tmp_path / "assets" / "review_batch_53",
        elevation_revision="elevation-test",
        orthophoto_revision="orthophoto-test",
        context_revision="context-test",
        tile_workers=3,
    )
    config.work_root.mkdir()
    config.review_batch.mkdir(parents=True)
    config.asset_library.write_text(
        json.dumps(
            {
                "asset_count": 1,
                "assets": [
                    {
                        "asset_id": "church_village_01",
                        "category": "building",
                        "reference": {"path": "buildings/church_village_01.png"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    prepared_fixed_assets: list[list[dict[str, object]]] = []

    def fake_context(**kwargs: object) -> SimpleNamespace:
        path = Path(kwargs["output_path"])
        path.write_text('{"context":"ok"}\n', encoding="utf-8")
        return SimpleNamespace(path=path, content_sha256="a" * 64, feature_counts={})

    def fake_prepare(**kwargs: object) -> SimpleNamespace:
        prepared_fixed_assets.append(list(kwargs["fixed_asset_placements"]))
        callback = kwargs["progress_callback"]
        callback("download_mnt_started", {"tile_id": kwargs["tile_id"]})
        callback(
            "download_mnt_completed",
            {"tile_id": kwargs["tile_id"], "byte_count": 1_048_576},
        )
        callback(
            "sources_published",
            {"tile_id": kwargs["tile_id"], "byte_count": 3_145_728},
        )
        return _fake_sources(Path(kwargs["output_root"]))

    def fake_produce(**kwargs: object) -> SimpleNamespace:
        root = Path(kwargs["output_root"])
        (root / "scene").mkdir(parents=True)
        (root / "scene" / "scene.usda").write_text(
            '#usda 1.0\ndef Xform "MeasuredScene" {}\n', encoding="utf-8"
        )
        (root / "scene" / "scene.done.json").write_text(
            json.dumps(
                {
                    "reconciliation": {
                        "buildings": {"instance_count": 2},
                        "trees": {"instance_count": 7},
                    }
                }
            ),
            encoding="utf-8",
        )
        origin = tuple(kwargs["tile_origin_l93_m"])
        normal_halo_mm = np.full((253, 253), 125_000, dtype="<i4")
        terrain = compile_fixed_terrain_from_canonical_mm(
            normal_halo_mm,
            tile_origin_l93_m=origin,
        )
        write_fixed_terrain(terrain, root / "terrain.fvtg")
        portable = Path(kwargs["portable_root"]).resolve()
        bundle = Path(kwargs["asset_bundle_root"]).resolve()
        request = {
            "schema": "fireviewer.simple-measured-tile-request.v1",
            "algorithm": "fireviewer.simple-measured-tile-algorithm.v1",
            "zone_id": kwargs["zone_id"],
            "tile_id": kwargs["tile_id"],
            "crs": "EPSG:2154",
            "tile_origin_l93_m": list(origin),
            "core_bounds_l93_m": [
                origin[0],
                origin[1],
                origin[0] + 500,
                origin[1] + 500,
            ],
            "sources": {"test": "source"},
            "asset_library_sha256": production._sha256_file(
                Path(kwargs["asset_library"])
            ),
            "asset_root_names": sorted(kwargs["asset_roots"]),
            "usage": "technical_pilot_non_final",
            "mns_fallback_policy": "ground_only_on_hag_validation_failure",
            "pipeline_files": {"test": "pipeline"},
            "prototype_bundle": {
                "scope": "explicit_shared",
                "portable_path": bundle.relative_to(portable).as_posix(),
            },
        }
        (root / "simple-measured-tile-receipt.v1.json").write_text(
            json.dumps({"build_id": "b" * 64, "request": request}),
            encoding="utf-8",
        )
        callback = kwargs["progress_callback"]
        callback(
            "placement_measured",
            {"tile_id": kwargs["tile_id"], "building_count": 2, "tree_count": 7},
        )
        callback("tile_published", {"tile_id": kwargs["tile_id"]})
        return SimpleNamespace(output_root=root, reused=False)

    validated_requests: list[dict[str, object]] = []

    def fake_validate(*args: object, **kwargs: object) -> dict[str, object]:
        validated_requests.append(dict(kwargs["expected_request"]))
        return {}

    def fake_gallery(
        job_root: Path, render: bool, callback: object
    ) -> list[tuple[str, str]]:
        assert render is True
        (job_root / production.BLEND_NAME).write_bytes(b"blend")
        gallery_root = job_root / "qa" / "gallery"
        gallery_root.mkdir(parents=True)
        receipt = job_root / production.GALLERY_RECEIPT_PATH
        receipt.parent.mkdir(parents=True, exist_ok=True)
        items: list[tuple[str, str]] = []
        records: list[dict[str, object]] = []
        for index, capture in enumerate(
            zone_gallery.build_capture_plan(500, 500), start=1
        ):
            image = gallery_root / f"{capture['capture_id']}.png"
            image.write_bytes(b"png")
            records.append(
                {**capture, "artifact": zone_gallery._artifact(image, job_root)}
            )
            items.append((str(image), f"capture {index}"))
            if callback is not None:
                callback(index, image.stem)
        gallery_receipt: dict[str, object] = {
            "schema": zone_gallery.SCHEMA,
            "status": zone_gallery.STATUS,
            "human_review_required": True,
            "accepted_human": False,
            "resolution": [zone_gallery.RESOLUTION, zone_gallery.RESOLUTION],
            "capture_count": zone_gallery.CAPTURE_COUNT,
            "zone_stage": zone_gallery._artifact(job_root / "zone.usda", job_root),
            "zone_plan": zone_gallery._artifact(job_root / "zone-plan.json", job_root),
            "zone_receipt": zone_gallery._artifact(
                job_root / "zone.done.json", job_root
            ),
            "standalone_blend": zone_gallery._artifact(
                job_root / production.BLEND_NAME, job_root
            ),
            "scene_bounds_m": {"minimum": [0, 0, 0], "maximum": [500, 500, 1]},
            "instance_counts": {"buildings": 2, "trees": 7},
            "render_policy": dict(zone_gallery.RENDER_POLICY),
            "captures": records,
        }
        gallery_receipt["capture_set_sha256"] = zone_gallery.hashlib.sha256(
            zone_gallery._canonical_bytes(records)
        ).hexdigest()
        gallery_receipt["receipt_content_sha256"] = zone_gallery.hashlib.sha256(
            zone_gallery._canonical_bytes(gallery_receipt)
        ).hexdigest()
        zone_gallery._write_json(receipt, gallery_receipt)
        return items

    monkeypatch.setattr(
        production, "validate_simple_measured_tile_package", fake_validate
    )
    engine = production.ProductionEngine(
        config,
        prepare_context_fn=fake_context,
        prepare_sources_fn=fake_prepare,
        produce_tile_fn=fake_produce,
        render_gallery_fn=fake_gallery,
        validate_assets=False,
    )
    latitude, longitude = _pilot_gps()
    progress_events: list[tuple[float, str]] = []
    results = list(
        engine.run(
            latitude,
            longitude,
            1.0,
            fixed_asset_placements={
                "schema": "fireviewer.fixed-asset-placement-request.v1",
                "crs": "EPSG:4326",
                "placements": [
                    {
                        "placement_id": "church-main",
                        "asset_id": "church_village_01",
                        "latitude": latitude,
                        "longitude": longitude,
                        "yaw_deg": 0,
                    }
                ],
            },
            progress_callback=lambda fraction, message: progress_events.append(
                (fraction, message)
            ),
        )
    )
    assert all(len(result) == 3 for result in results)
    archive = Path(results[-1][1])
    assert len(results[-1][2]) == 20
    assert archive.is_file()
    assert sum(len(items) for items in prepared_fixed_assets) == 1
    fixed = next(items[0] for items in prepared_fixed_assets if items)
    assert fixed["asset_id"] == "church_village_01"
    assert fixed["owner_tile_origin_l93_m"] == [
        820_000,
        6_312_500,
    ]
    job_root = archive.parent
    assert not (job_root / "sources").exists() or not any(
        (job_root / "sources").iterdir()
    )
    receipt = json.loads(
        (job_root / production.ZONE_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert receipt["tile_count"] > 1
    assert len(prepared_fixed_assets) == receipt["tile_count"]
    assert receipt["building_count"] == 2 * receipt["tile_count"]
    assert receipt["tree_count"] == 7 * receipt["tile_count"]
    assert receipt["accepted_human"] is False
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    prefix = f"fireviewer-{receipt['zone_id']}/"
    assert prefix + "zone.usda" in names
    assert prefix + "zone.done.json" in names
    assert prefix + "zone.blend" in names
    assert prefix + production.FIXED_ASSET_REQUEST_NAME in names
    assert (
        len([name for name in names if name.startswith(prefix + "qa/gallery/")]) == 20
    )
    assert any(name.startswith(prefix + "packages/") for name in names)
    assert any(name.startswith(prefix + "provenance/") for name in names)
    assert not any("orthophoto-1m.png" in name for name in names)
    assert not any("mnt-05m.tif" in name or "mns-05m.tif" in name for name in names)
    assert validated_requests
    expected_tile_ids = {tile["tile_id"] for tile in receipt["tiles"]}
    assert {request["tile_id"] for request in validated_requests} == expected_tile_ids
    messages = [message for _fraction, message in progress_events]
    assert any("Moteur embarqué validé" in message for message in messages)
    assert any("3 tuiles simultanées" in message for message in messages)
    assert any("MNT 0,5 m reçu" in message for message in messages)
    assert any("Placement MNS−MNT mesuré" in message for message in messages)
    assert any("Compression du pack autonome" in message for message in messages)
    assert any("capture 20/20" in message for message in messages)


def test_expected_request_rejects_a_package_from_another_tile(tmp_path: Path) -> None:
    latitude, longitude = _pilot_gps()
    plan = production.plan_zone(latitude, longitude, 0.5)
    tile = plan.tiles[0]
    library = tmp_path / "asset-library.json"
    library.write_text("{}", encoding="utf-8")
    review_batch = tmp_path / "review-batch"
    review_batch.mkdir()
    bundle = tmp_path / "shared" / "prototypes"
    bundle.mkdir(parents=True)
    package = tmp_path / "package"
    package.mkdir()
    request = {
        "schema": "fireviewer.simple-measured-tile-request.v1",
        "algorithm": "fireviewer.simple-measured-tile-algorithm.v1",
        "zone_id": plan.zone_id,
        "tile_id": "x0_y0",
        "crs": "EPSG:2154",
        "tile_origin_l93_m": [0, 0],
        "core_bounds_l93_m": [0, 0, 500, 500],
        "sources": {},
        "asset_library_sha256": production._sha256_file(library),
        "asset_root_names": ["review_batch"],
        "usage": "technical_pilot_non_final",
        "pipeline_files": {},
        "prototype_bundle": {
            "scope": "explicit_shared",
            "portable_path": bundle.relative_to(tmp_path).as_posix(),
        },
    }
    (package / "simple-measured-tile-receipt.v1.json").write_text(
        json.dumps({"request": request}), encoding="utf-8"
    )
    with pytest.raises(
        production.SimpleProductionError, match="diffère du job courant"
    ):
        production._expected_request_from_receipt(
            package,
            plan=plan,
            tile=tile,
            asset_library=library,
            asset_roots={"review_batch": review_batch},
            portable_root=tmp_path,
            asset_bundle_root=bundle,
        )


def test_private_dataset_publication_is_atomic_idempotent_and_hides_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latitude, longitude = _pilot_gps()
    plan = production.plan_zone(latitude, longitude, 0.5)
    config = production.ProductionConfig(
        work_root=tmp_path,
        portable_root=tmp_path,
        asset_library=tmp_path / "asset-library.json",
        review_batch=tmp_path / "review-batch",
        elevation_revision="e",
        orthophoto_revision="o",
        context_revision="c",
        dataset_id="fireviewer/simple-measured-scenes-v1",
    )
    receipt = {
        "build_id": "a" * 64,
        "tile_count": 1,
        "building_count": 2,
        "tree_count": 7,
    }
    production._write_json(tmp_path / production.ZONE_RECEIPT_NAME, receipt)
    archive = tmp_path / production.ZIP_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fireviewer-zone/zone.usda", "#usda 1.0\n")

    commits: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret-not-for-files"

        def repo_info(self, *, repo_id: str, repo_type: str) -> SimpleNamespace:
            assert repo_id == config.dataset_id
            assert repo_type == "dataset"
            return SimpleNamespace(private=True)

        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            commits.append(dict(kwargs))
            return SimpleNamespace(oid="c" * 40)

    class FakeOperation:
        def __init__(self, **kwargs: object) -> None:
            self.values = dict(kwargs)

    monkeypatch.setenv("HF_TOKEN", "secret-not-for-files")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=FakeOperation, HfApi=FakeApi),
    )
    capture_records = []
    for index in range(20):
        relative = f"qa/captures/capture-{index:02d}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"capture-{index}".encode())
        capture_records.append(
            {
                "capture_id": f"capture-{index:02d}",
                "category": "overview" if index < 4 else "detail",
                "artifact": {"path": relative},
            }
        )
    monkeypatch.setattr(
        production,
        "verify_gallery",
        lambda _root: {"captures": capture_records},
    )
    first = production._publish_dataset_entry(
        config,
        job_root=tmp_path,
        plan=plan,
        receipt=receipt,
        archive=archive,
    )
    second = production._publish_dataset_entry(
        config,
        job_root=tmp_path,
        plan=plan,
        receipt=receipt,
        archive=archive,
    )
    assert first == second
    assert len(commits) == 1
    operations = commits[0]["operations"]
    assert len(operations) == 23
    assert all(
        operation.values["path_in_repo"].startswith(
            f"zones/{plan.zone_id}/{receipt['build_id']}/"
        )
        for operation in operations
    )
    for path in tmp_path.iterdir():
        if path.is_file():
            assert b"secret-not-for-files" not in path.read_bytes()


def test_gallery_items_preserve_overview_then_detail_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captures = [
        {
            "capture_id": f"{index:02d}-overview",
            "category": "overview",
            "artifact": {"path": f"qa/{index:02d}.png"},
        }
        for index in range(4)
    ] + [
        {
            "capture_id": f"{index:02d}-detail",
            "category": "detail_coverage",
            "artifact": {"path": f"qa/{index:02d}.png"},
        }
        for index in range(4, 20)
    ]
    monkeypatch.setattr(
        production, "verify_gallery", lambda _root: {"captures": captures}
    )
    items = production._gallery_items(tmp_path)
    assert len(items) == 20
    assert all("overview" in caption for _path, caption in items[:4])
    assert all("detail_coverage" in caption for _path, caption in items[4:])
