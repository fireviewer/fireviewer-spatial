from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from pyproj import Transformer
import pytest

import simple_production_gradio as ui
import render_simple_zone_gallery as zone_gallery
from fixed_terrain_grid import (
    compile_fixed_terrain_from_canonical_mm,
    write_fixed_terrain,
)


def _pilot_gps() -> tuple[float, float]:
    longitude, latitude = Transformer.from_crs(2154, 4326, always_xy=True).transform(
        820250, 6312750
    )
    return latitude, longitude


def test_pilot_center_and_1_5_km_produce_the_exact_validated_3x3() -> None:
    latitude, longitude = _pilot_gps()
    plan = ui.plan_zone(latitude, longitude, 1.5)
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
    plan = ui.plan_zone(latitude, longitude, 1.5)
    stage = ui._write_zone_stage(tmp_path, plan)
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
    config = ui.ProductionConfig(
        work_root=tmp_path / "work",
        portable_root=tmp_path,
        asset_library=tmp_path / "assets" / "asset-library.v1.json",
        review_batch=tmp_path / "assets" / "review_batch_53",
        elevation_revision="elevation-test",
        orthophoto_revision="orthophoto-test",
        context_revision="context-test",
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
            "asset_library_sha256": ui._sha256_file(Path(kwargs["asset_library"])),
            "asset_root_names": sorted(kwargs["asset_roots"]),
            "usage": "technical_pilot_non_final",
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
        (job_root / ui.BLEND_NAME).write_bytes(b"blend")
        gallery_root = job_root / "qa" / "gallery"
        gallery_root.mkdir(parents=True)
        receipt = job_root / ui.GALLERY_RECEIPT_PATH
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
                job_root / ui.BLEND_NAME, job_root
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

    monkeypatch.setattr(ui, "validate_simple_measured_tile_package", fake_validate)
    engine = ui.ProductionEngine(
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
            0.5,
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
    assert len(prepared_fixed_assets) == 1
    assert prepared_fixed_assets[0][0]["asset_id"] == "church_village_01"
    assert prepared_fixed_assets[0][0]["owner_tile_origin_l93_m"] == [
        820_000,
        6_312_500,
    ]
    job_root = archive.parent
    assert not (job_root / "sources").exists() or not any(
        (job_root / "sources").iterdir()
    )
    receipt = json.loads((job_root / ui.ZONE_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["tile_count"] == 1
    assert receipt["building_count"] == 2
    assert receipt["tree_count"] == 7
    assert receipt["accepted_human"] is False
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    prefix = f"fireviewer-{receipt['zone_id']}/"
    assert prefix + "zone.usda" in names
    assert prefix + "zone.done.json" in names
    assert prefix + "zone.blend" in names
    assert prefix + ui.FIXED_ASSET_REQUEST_NAME in names
    assert (
        len([name for name in names if name.startswith(prefix + "qa/gallery/")]) == 20
    )
    assert any(name.startswith(prefix + "packages/") for name in names)
    assert any(name.startswith(prefix + "provenance/") for name in names)
    assert not any("orthophoto-1m.png" in name for name in names)
    assert not any("mnt-05m.tif" in name or "mns-05m.tif" in name for name in names)
    assert validated_requests
    assert all(
        request["tile_id"] == receipt["tiles"][0]["tile_id"]
        for request in validated_requests
    )
    messages = [message for _fraction, message in progress_events]
    assert any("Moteur embarqué validé" in message for message in messages)
    assert any("MNT 0,5 m reçu" in message for message in messages)
    assert any("Placement MNS−MNT mesuré" in message for message in messages)
    assert any("Compression du pack autonome" in message for message in messages)
    assert any("capture 20/20" in message for message in messages)


def test_expected_request_rejects_a_package_from_another_tile(tmp_path: Path) -> None:
    latitude, longitude = _pilot_gps()
    plan = ui.plan_zone(latitude, longitude, 0.5)
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
        "asset_library_sha256": ui._sha256_file(library),
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
    with pytest.raises(ui.SimpleProductionUiError, match="diffère du job courant"):
        ui._expected_request_from_receipt(
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
    plan = ui.plan_zone(latitude, longitude, 0.5)
    config = ui.ProductionConfig(
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
    ui._write_json(tmp_path / ui.ZONE_RECEIPT_NAME, receipt)
    archive = tmp_path / ui.ZIP_NAME
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
    first = ui._publish_dataset_entry(
        config,
        job_root=tmp_path,
        plan=plan,
        receipt=receipt,
        archive=archive,
    )
    second = ui._publish_dataset_entry(
        config,
        job_root=tmp_path,
        plan=plan,
        receipt=receipt,
        archive=archive,
    )
    assert first == second
    assert len(commits) == 1
    operations = commits[0]["operations"]
    assert len(operations) == 3
    assert all(
        operation.values["path_in_repo"].startswith(
            f"zones/{plan.zone_id}/{receipt['build_id']}/"
        )
        for operation in operations
    )
    for path in tmp_path.iterdir():
        if path.is_file():
            assert b"secret-not-for-files" not in path.read_bytes()


def test_gradio_app_has_one_screen_without_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ui.ProductionConfig(
        work_root=tmp_path / "work",
        portable_root=tmp_path,
        asset_library=tmp_path / "asset-library.json",
        review_batch=tmp_path / "review-batch",
        elevation_revision="e",
        orthophoto_revision="o",
        context_revision="c",
    )
    engine = ui.ProductionEngine(config, validate_assets=False)
    engine.asset_library_payload = {
        "asset_count": 1,
        "assets": [
            {
                "asset_id": "church_village_01",
                "category": "building",
                "reference": {"path": "buildings/church_village_01.png"},
            }
        ],
    }
    engine.fixed_asset_choices = tuple(ui.asset_choices(engine.asset_library_payload))
    perimeter_archive = tmp_path / "perimeters.zip"
    perimeter_archive.write_bytes(b"zip")
    perimeter_package = tmp_path / "perimeter-package"
    perimeter_package.mkdir()
    map_archive = tmp_path / "map.zip"
    map_archive.write_bytes(b"map")
    frame_models = [tmp_path / "frame-0000.glb", tmp_path / "frame-0001.glb"]
    for model in frame_models:
        model.write_bytes(b"glb")

    def fake_perimeter_producer(source: Path, work_root: Path) -> SimpleNamespace:
        assert source == tmp_path / "perimeters.json"
        assert work_root == config.work_root
        return SimpleNamespace(
            archive=perimeter_archive,
            package_root=perimeter_package,
            manifest={"frame_count": 2, "fixed_layer_count": 3},
        )

    def fake_perimeter_viewer(
        source_map: Path, package_root: Path, work_root: Path
    ) -> SimpleNamespace:
        assert source_map == map_archive
        assert package_root == perimeter_package
        assert work_root == config.work_root
        return SimpleNamespace(
            frames=(
                SimpleNamespace(model=frame_models[0], caption="Jour 1"),
                SimpleNamespace(model=frame_models[1], caption="Jour 2"),
            )
        )

    app = ui.build_app(
        engine,
        perimeter_producer=fake_perimeter_producer,
        perimeter_viewer=fake_perimeter_viewer,
    )
    components = app.get_config_file()["components"]
    types = [component["type"] for component in components]
    assert types.count("number") == 5
    assert types.count("button") == 5
    assert types.count("file") == 5
    assert types.count("gallery") == 1
    assert types.count("dropdown") == 1
    assert types.count("dataframe") == 1
    assert types.count("json") == 1
    assert types.count("downloadbutton") == 1
    assert types.count("timer") == 1
    assert types.count("model3d") == 1
    assert types.count("slider") == 1
    assert "tabs" not in types
    assert "navbar" not in types

    numbers = [component for component in components if component["type"] == "number"]
    assert [component["props"]["minimum"] for component in numbers] == [
        -90,
        -180,
        0.5,
        -90,
        -180,
    ]
    assert [component["props"]["maximum"] for component in numbers] == [
        90,
        180,
        15.0,
        90,
        180,
    ]
    assert numbers[0]["props"]["placeholder"] == "43.90349754"
    assert numbers[1]["props"]["placeholder"] == "4.49681631"
    fixed_dropdown = next(
        component for component in components if component["type"] == "dropdown"
    )
    assert fixed_dropdown["props"]["choices"][0][1] == "church_village_01"
    fixed_upload = next(
        component
        for component in components
        if component["type"] == "file"
        and component["props"]["label"] == "Charger la liste JSON contractuelle"
    )
    assert fixed_upload["props"]["file_types"] == [".json"]

    download = next(
        component
        for component in components
        if component["type"] == "file"
        and component["props"]["label"] == "Scène autonome et pack complet"
    )
    assert download["props"]["label"] == "Scène autonome et pack complet"
    perimeter_source = next(
        component
        for component in components
        if component["type"] == "file"
        and component["props"]["label"].startswith("Observations de périmètres")
    )
    assert perimeter_source["props"]["file_types"] == [".json", ".geojson"]
    perimeter_map = next(
        component
        for component in components
        if component["type"] == "file"
        and component["props"]["label"].startswith("Carte FireViewer autonome")
    )
    assert perimeter_map["props"]["file_types"] == [".zip"]
    perimeter_download = next(
        component
        for component in components
        if component["type"] == "file"
        and component["props"]["label"] == "Calques fixes USD et timeline de simulation"
    )
    assert perimeter_download["props"]["interactive"] is False
    gallery = next(
        component for component in components if component["type"] == "gallery"
    )
    assert gallery["props"]["columns"] == 4
    assert gallery["props"]["rows"] == 5
    assert gallery["props"]["allow_preview"] is True
    assert gallery["props"]["buttons"] == ["fullscreen", "download"]

    dependencies = app.get_config_file()["dependencies"]
    public = [
        dependency
        for dependency in dependencies
        if dependency["api_visibility"] == "public"
    ]
    assert {dependency["api_name"] for dependency in public} == {
        "launch",
        "launch_with_fixed_assets",
        "generate_perimeter_layer",
    }
    launch_dependency = next(
        dependency for dependency in public if dependency["api_name"] == "launch"
    )
    assert len(launch_dependency["inputs"]) == 3
    assert len(launch_dependency["outputs"]) == 3
    fixed_launch_dependency = next(
        dependency
        for dependency in public
        if dependency["api_name"] == "launch_with_fixed_assets"
    )
    assert len(fixed_launch_dependency["inputs"]) == 4
    assert len(fixed_launch_dependency["outputs"]) == 3
    perimeter_dependency = next(
        dependency
        for dependency in public
        if dependency["api_name"] == "generate_perimeter_layer"
    )
    assert len(perimeter_dependency["inputs"]) == 2
    assert len(perimeter_dependency["outputs"]) == 2
    assert all(
        dependency["api_visibility"] == "private"
        for dependency in dependencies
        if dependency not in public
    )

    begin_ui = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "begin_ui"
    )
    button_start, timer_start, started_at, elapsed = begin_ui()
    assert button_start["interactive"] is False
    assert button_start["value"] == "Production en cours…"
    assert timer_start["active"] is True
    assert started_at > 0
    assert elapsed == "Temps écoulé : 00:00"

    finish_ui = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "finish_ui"
    )
    button_end, timer_end = finish_ui()
    assert button_end["interactive"] is True
    assert button_end["value"] == "Lancer la production"
    assert timer_end["active"] is False

    def fail_run(*_args: object, **_kwargs: object):
        raise RuntimeError("panne simulée")
        yield

    monkeypatch.setattr(engine, "run", fail_run)
    launch = next(
        function.fn for function in app.fns.values() if function.fn.__name__ == "launch"
    )
    assert list(launch(43.9, 4.5, 1.5)) == [("Échec : panne simulée", None, [])]
    launch_with_fixed_assets = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "launch_with_fixed_assets"
    )
    assert list(
        launch_with_fixed_assets(43.9, 4.5, 1.5, ui.EMPTY_FIXED_ASSET_REQUEST)
    ) == [("Échec : panne simulée", None, [])]

    add_fixed_asset = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "add_fixed_asset"
    )
    fixed_state, fixed_rows, fixed_message = add_fixed_asset(
        43.9,
        4.5,
        "church_village_01",
        ui.EMPTY_FIXED_ASSET_REQUEST,
    )
    assert len(fixed_state["placements"]) == 1
    assert fixed_rows[0][1] == "church_village_01"
    assert fixed_message.startswith("✅ 1 placement")

    imported_path = tmp_path / "fixed-assets.json"
    imported_path.write_text(json.dumps(fixed_state), encoding="utf-8")
    import_fixed_assets = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "import_fixed_assets"
    )
    imported_state, imported_rows, imported_message = import_fixed_assets(
        str(imported_path), ui.EMPTY_FIXED_ASSET_REQUEST
    )
    assert imported_state == fixed_state
    assert imported_rows == fixed_rows
    assert imported_message.startswith("✅ JSON validé")

    clear_fixed_assets = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "clear_fixed_assets"
    )
    empty_state, empty_rows, empty_message = clear_fixed_assets()
    assert empty_state == ui.EMPTY_FIXED_ASSET_REQUEST
    assert empty_rows == []
    assert empty_message.startswith("ℹ️")

    perimeter_source_path = tmp_path / "perimeters.json"
    perimeter_source_path.write_text("{}", encoding="utf-8")
    generate_perimeter_layer = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "generate_perimeter_layer"
    )
    perimeter_status, perimeter_file = generate_perimeter_layer(
        str(perimeter_source_path), str(map_archive)
    )
    assert perimeter_status.startswith("Terminé — 2 observations, 3 calques fixes")
    assert perimeter_file == str(perimeter_archive)
    assert generate_perimeter_layer(None, None) == (
        "Échec : importez un fichier JSON ou GeoJSON",
        None,
    )

    prepare_perimeter_viewer = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "prepare_perimeter_viewer"
    )
    frames, slider_update, model_update, caption = prepare_perimeter_viewer(
        str(perimeter_source_path), str(map_archive)
    )
    assert frames == [
        (str(frame_models[0]), "Jour 1"),
        (str(frame_models[1]), "Jour 2"),
    ]
    assert slider_update["maximum"] == 1
    assert slider_update["visible"] is True
    assert model_update["value"] == str(frame_models[0])
    assert model_update["visible"] is True
    assert caption == "Observation 1/2 — Jour 1"

    select_perimeter_frame = next(
        function.fn
        for function in app.fns.values()
        if function.fn.__name__ == "select_perimeter_frame"
    )
    selected_model, selected_caption = select_perimeter_frame(1, frames)
    assert selected_model["value"] == str(frame_models[1])
    assert selected_caption == "Observation 2/2 — Jour 2"

    no_frames, hidden_slider, hidden_model, no_map_caption = prepare_perimeter_viewer(
        str(perimeter_source_path), None
    )
    assert no_frames == []
    assert hidden_slider["visible"] is False
    assert hidden_model["visible"] is False
    assert "Importez le ZIP autonome" in no_map_caption


def test_zone_preview_uses_the_exact_production_grid() -> None:
    aligned = ui._zone_preview(
        43.903497538,
        4.49681631,
        1.5,
        max_side_m=15_000,
        max_tiles=900,
    )
    assert "2.25 km²" in aligned
    assert "**9 tuiles**" in aligned
    assert "emprise produite 2.25 km²" in aligned
    assert "GPS-75A895C259FA2E84" in aligned

    non_aligned = ui._zone_preview(
        43.9,
        4.5,
        1.5,
        max_side_m=15_000,
        max_tiles=900,
    )
    assert "**16 tuiles**" in non_aligned
    assert "emprise produite 4 km²" in non_aligned


def test_zone_preview_reports_input_and_projection_errors() -> None:
    missing = ui._zone_preview(
        None,
        None,
        1.5,
        max_side_m=15_000,
        max_tiles=900,
    )
    assert missing.startswith("ℹ️")
    invalid_gps = ui._zone_preview(
        91,
        4,
        1.5,
        max_side_m=15_000,
        max_tiles=900,
    )
    assert invalid_gps == "⚠️ Coordonnées GPS invalides"
    outside_l93 = ui._zone_preview(
        0,
        0,
        1.5,
        max_side_m=15_000,
        max_tiles=900,
    )
    assert outside_l93 == "⚠️ Le centre doit être couvert par la projection Lambert-93"


def test_elapsed_label_is_deterministic() -> None:
    assert ui._elapsed_label(None, now=100) == "Temps écoulé : 00:00"
    assert ui._elapsed_label(100, now=100) == "Temps écoulé : 00:00"
    assert ui._elapsed_label(100, now=161) == "Temps écoulé : 01:01"
    assert ui._elapsed_label(100, now=3_801) == "Temps écoulé : 01:01:41"


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
    monkeypatch.setattr(ui, "verify_gallery", lambda _root: {"captures": captures})
    items = ui._gallery_items(tmp_path)
    assert len(items) == 20
    assert all("overview" in caption for _path, caption in items[:4])
    assert all("detail_coverage" in caption for _path, caption in items[4:])


def test_current_embedded_asset_inputs_are_complete(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    config = ui.ProductionConfig(
        work_root=tmp_path / "work",
        portable_root=repository,
        asset_library=repository
        / "fireviewer-work"
        / "production"
        / "fr-30-00001-pilot-v1"
        / "shared"
        / "assets"
        / "asset-library.v1.json",
        review_batch=repository
        / "fireviewer-sdg"
        / "asset4sim"
        / "generated_hunyuan3d_v2"
        / "review_batch_53",
        elevation_revision="e",
        orthophoto_revision="o",
        context_revision="c",
    )
    summary = ui.validate_embedded_assets(config)
    assert summary["asset_count"] == 53
    assert summary["checked_artifacts"] == 106
    assert summary["building_assets"] == 24
    assert summary["tree_assets"] == 18
