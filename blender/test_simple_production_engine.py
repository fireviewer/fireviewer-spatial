from __future__ import annotations

import json
from io import StringIO
import os
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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


def test_zone_stage_delegates_all_tiles_to_the_compact_author(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latitude, longitude = _pilot_gps()
    plan = production.plan_zone(latitude, longitude, 1.5)
    observed: dict[str, object] = {}

    def build(root: Path, **kwargs: object) -> Path:
        observed.update(kwargs)
        stage = root / production.ENTRY_STAGE
        stage.write_text("#usda 1.0\n", encoding="utf-8")
        return stage

    monkeypatch.setattr(production, "build_compact_zone_stage", build)
    stage = production._write_zone_stage(tmp_path, plan)
    assert stage == tmp_path / production.ENTRY_STAGE
    assert observed["zone_id"] == plan.zone_id
    assert observed["production_bounds_l93_m"] == plan.production_bounds_l93_m
    compact_tiles = observed["tiles"]
    assert isinstance(compact_tiles, tuple)
    assert [(tile.tile_id, tile.origin_l93_m) for tile in compact_tiles] == [
        (tile.tile_id, tile.origin_l93_m) for tile in plan.tiles
    ]


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


def test_config_defaults_to_six_workers_and_keeps_eight_as_measured_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FIREVIEWER_TILE_WORKERS", raising=False)
    monkeypatch.delenv("FIREVIEWER_SOURCE_WORKERS", raising=False)
    assert production.ProductionConfig.from_environment().tile_workers == 6
    assert production.ProductionConfig.from_environment().source_workers == 12
    assert (
        production.ProductionConfig.from_environment().max_archive_bytes == 12 * 1024**3
    )
    config = production.ProductionConfig(
        work_root=tmp_path,
        portable_root=tmp_path,
        asset_library=tmp_path / "asset-library.json",
        review_batch=tmp_path / "assets",
        elevation_revision="elevation-test",
        orthophoto_revision="orthophoto-test",
        context_revision="context-test",
        tile_workers=9,
    )
    with pytest.raises(production.SimpleProductionError, match="Limites du pod"):
        production.validate_production_config(config)

    accepted = production.ProductionConfig(
        work_root=tmp_path,
        portable_root=tmp_path,
        asset_library=tmp_path / "asset-library.json",
        review_batch=tmp_path / "assets",
        elevation_revision="elevation-test",
        orthophoto_revision="orthophoto-test",
        context_revision="context-test",
        tile_workers=8,
        source_workers=16,
    )
    monkeypatch.setattr(production, "validate_embedded_assets", lambda _config: {})
    monkeypatch.setattr(production, "validate_embedded_runtime", lambda _config: {})
    production.validate_production_config(accepted)

    config = production.ProductionConfig(
        work_root=tmp_path,
        portable_root=tmp_path,
        asset_library=tmp_path / "asset-library.json",
        review_batch=tmp_path / "assets",
        elevation_revision="elevation-test",
        orthophoto_revision="orthophoto-test",
        context_revision="context-test",
        tile_workers=4,
        source_workers=3,
    )
    with pytest.raises(production.SimpleProductionError, match="Limites du pod"):
        production.validate_production_config(config)


def test_interrupted_staging_cleanup_is_bounded_to_owned_patterns(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    source_staging = job / "sources" / ".x1_y1.simple-sources.part"
    package_staging = job / "packages" / ".x1_y1.simple-measured-tile.part"
    prototype_staging = job / "shared" / "prototypes" / ".oak.part"
    for staging in (source_staging, package_staging, prototype_staging):
        staging.mkdir(parents=True)
        (staging / "partial.bin").write_bytes(b"partial")
    unrelated = job / "packages" / "keep.part"
    unrelated.write_bytes(b"keep")
    checkpoint_staging = (
        job
        / production.TILE_CHECKPOINT_COLLECTION_NAME
        / production.TILE_CHECKPOINT_VERSION
        / ".x1_y1.zip.part"
    )
    checkpoint_staging.parent.mkdir(parents=True)
    checkpoint_staging.write_bytes(b"partial")

    removed = production._remove_interrupted_staging(job)

    assert removed == [
        "sources/.x1_y1.simple-sources.part",
        "packages/.x1_y1.simple-measured-tile.part",
        "tile-checkpoints/v1/.x1_y1.zip.part",
    ]
    assert not source_staging.exists()
    assert not package_staging.exists()
    assert prototype_staging.is_dir()
    assert (prototype_staging / "partial.bin").read_bytes() == b"partial"
    assert unrelated.read_bytes() == b"keep"


def test_overlapping_retries_get_isolated_scratch_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        production,
        "_absolute",
        lambda path, _label, *, exists: Path(path).resolve(),
    )
    config = production.ProductionConfig(
        work_root=tmp_path / "work",
        scratch_root=tmp_path / "scratch",
        portable_root=tmp_path,
        asset_library=tmp_path / "asset-library.json",
        review_batch=tmp_path / "assets",
        elevation_revision="elevation-test",
        orthophoto_revision="orthophoto-test",
        context_revision="context-test",
    )

    first = production._prepare_scratch_job(config, "GPS-SAME-ZONE")
    assert first is not None
    sentinel = first / "active.part"
    sentinel.write_bytes(b"active")
    second = production._prepare_scratch_job(config, "GPS-SAME-ZONE")

    assert second is not None
    assert first != second
    assert first.parent == second.parent == config.scratch_root / "jobs"
    assert first.name.startswith("GPS-SAME-ZONE-")
    assert second.name.startswith("GPS-SAME-ZONE-")
    assert sentinel.read_bytes() == b"active"


def test_checkpoint_is_deterministic_streamed_and_rehashed_before_restore(
    tmp_path: Path,
) -> None:
    package = tmp_path / "local" / "packages" / "x1_y2"
    (package / "scene").mkdir(parents=True)
    (package / "scene" / "scene.usda").write_bytes(b"#usda 1.0\n")
    (package / "terrain.fvtg").write_bytes(b"terrain" * 100)
    (package / "simple-measured-tile-receipt.v1.json").write_bytes(
        b'{"build_id":"fixture"}\n'
    )
    first_job = tmp_path / "network-a" / "jobs" / "GPS-CHECKPOINT"
    second_job = tmp_path / "network-b" / "jobs" / "GPS-CHECKPOINT"
    first = production._write_tile_checkpoint(package, first_job, tile_id="x1_y2")
    second = production._write_tile_checkpoint(package, second_job, tile_id="x1_y2")
    first_archive, _first_receipt = production._tile_checkpoint_paths(
        first_job, "x1_y2"
    )
    second_archive, _second_receipt = production._tile_checkpoint_paths(
        second_job, "x1_y2"
    )
    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert first_archive.read_bytes() == second_archive.read_bytes()

    restored = tmp_path / "scratch" / "packages" / "x1_y2"
    restored_record = production._restore_tile_checkpoint(
        first_job, restored, tile_id="x1_y2"
    )
    assert restored_record == first
    assert (restored / "terrain.fvtg").read_bytes() == b"terrain" * 100

    first_archive.write_bytes(first_archive.read_bytes() + b"tamper")
    with pytest.raises(production.SimpleProductionError, match="checkpoint|Checkpoint"):
        production._restore_tile_checkpoint(
            first_job,
            tmp_path / "second-restore" / "x1_y2",
            tile_id="x1_y2",
        )


def test_checkpoint_is_built_on_local_staging_before_persistent_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "scratch" / "packages" / "x1_y2"
    package.mkdir(parents=True)
    (package / "terrain.fvtg").write_bytes(b"terrain" * 100)
    (package / "simple-measured-tile-receipt.v1.json").write_bytes(
        b'{"build_id":"fixture"}\n'
    )
    job = tmp_path / "persistent" / "jobs" / "GPS-CHECKPOINT"
    local_staging = tmp_path / "scratch" / "checkpoint-staging"
    copied_sources: list[Path] = []

    def copy_local(source: Path, destination: Path, *, timeout_seconds: float) -> None:
        assert timeout_seconds == production.CHECKPOINT_PUBLISH_TIMEOUT_SECONDS
        copied_sources.append(source)
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(production, "_copy_checkpoint_file_with_timeout", copy_local)
    production._write_tile_checkpoint(
        package,
        job,
        tile_id="x1_y2",
        local_staging_root=local_staging,
        publish_lock=threading.Lock(),
    )

    assert len(copied_sources) == 2
    assert all(source.parent == local_staging for source in copied_sources)
    assert not list(local_staging.glob("*.part"))
    archive, receipt = production._tile_checkpoint_paths(job, "x1_y2")
    assert archive.is_file()
    assert receipt.is_file()


def test_checkpoint_copy_timeout_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "local.zip"
    destination = tmp_path / "persistent.part"
    source.write_bytes(b"checkpoint")
    monkeypatch.setattr(production, "CHECKPOINT_COPY_USES_SUBPROCESS", True)

    def expire(*_args: object, **_kwargs: object) -> None:
        destination.write_bytes(b"partial")
        raise production.subprocess.TimeoutExpired("cp", 0.01)

    monkeypatch.setattr(production.subprocess, "run", expire)
    with pytest.raises(production.SimpleProductionError, match="expirée"):
        production._copy_checkpoint_file_with_timeout(
            source, destination, timeout_seconds=0.01
        )
    assert not destination.exists()


def test_checkpoint_publications_are_serialized_across_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages: list[Path] = []
    for tile_id in ("x1_y2", "x2_y2"):
        package = tmp_path / "scratch" / "packages" / tile_id
        package.mkdir(parents=True)
        (package / "terrain.fvtg").write_bytes(tile_id.encode() * 100)
        (package / "simple-measured-tile-receipt.v1.json").write_bytes(
            f'{{"build_id":"{tile_id}"}}\n'.encode()
        )
        packages.append(package)
    job = tmp_path / "persistent" / "jobs" / "GPS-SERIAL"
    local_staging = tmp_path / "scratch" / "checkpoint-staging"
    publish_lock = threading.Lock()
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def slow_copy(source: Path, destination: Path, *, timeout_seconds: float) -> None:
        nonlocal active, maximum_active
        assert timeout_seconds == production.CHECKPOINT_PUBLISH_TIMEOUT_SECONDS
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            threading.Event().wait(0.01)
            destination.write_bytes(source.read_bytes())
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(production, "_copy_checkpoint_file_with_timeout", slow_copy)
    errors: list[BaseException] = []

    def checkpoint(package: Path) -> None:
        try:
            production._write_tile_checkpoint(
                package,
                job,
                tile_id=package.name,
                local_staging_root=local_staging,
                publish_lock=publish_lock,
            )
        except BaseException as error:  # pragma: no cover - surfaced below
            errors.append(error)

    threads = [
        threading.Thread(target=checkpoint, args=(package,)) for package in packages
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert maximum_active == 1
    assert len(list((job / "tile-checkpoints" / "v1").glob("*.zip"))) == 2


def test_25_by_25_plan_is_interleaved_across_49_four_by_four_metatiles() -> None:
    latitude, longitude = _pilot_gps()
    plan = production.plan_zone(latitude, longitude, 12.0)
    pending = list(enumerate(plan.tiles))
    ordered = production._interleave_tiles_by_metatile(pending)

    def block(item: tuple[int, production.TilePlan]) -> tuple[int, int]:
        x, y = item[1].origin_l93_m
        size = production.SOURCE_METATILE_SIZE_M
        return (x // size * size, y // size * size)

    assert len(plan.tiles) == 625
    assert len({block(item) for item in pending}) == 49
    assert len({block(item) for item in ordered[:12]}) == 12
    assert sorted(index for index, _tile in ordered) == list(range(625))


def test_versioned_prototype_bundle_preserves_legacy_and_resumes_stably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "jobs" / "GPS-MIGRATION"
    legacy_wrapper = job / "shared" / "prototypes" / "legacy-tree" / "prototype.usda"
    legacy_wrapper.parent.mkdir(parents=True)
    legacy_wrapper.write_bytes(b"r17-wrapper-must-remain")
    monkeypatch.setattr(
        production, "_prototype_bundle_builder_sha256", lambda: "b" * 64
    )
    summary = {"catalog_sha256": "a" * 64}

    first = production._prototype_bundle_root(job, summary)
    first.mkdir(parents=True)
    interrupted = first / ".tree.part"
    interrupted.mkdir()
    (interrupted / "partial.bin").write_bytes(b"partial")

    removed = production._remove_interrupted_staging(job)
    resumed = production._prototype_bundle_root(job, summary)

    assert resumed == first
    assert first.parent.name == production.PROTOTYPE_BUNDLE_COLLECTION_NAME
    assert first.name.startswith("v1-")
    assert removed == []
    assert interrupted.is_dir()
    assert legacy_wrapper.read_bytes() == b"r17-wrapper-must-remain"

    active_wrapper = first / "tree" / "prototype.usda"
    active_wrapper.parent.mkdir(parents=True)
    active_wrapper.write_bytes(b"r18-wrapper")
    scratch = tmp_path / "scratch" / "jobs" / "GPS-MIGRATION"
    scratch.mkdir(parents=True)
    assembly = production._prepare_local_assembly(
        job, scratch, active_prototype_bundle=first
    )
    assert (assembly / active_wrapper.relative_to(job)).read_bytes() == b"r18-wrapper"
    assert not (assembly / legacy_wrapper.relative_to(job)).exists()
    assert legacy_wrapper.read_bytes() == b"r17-wrapper-must-remain"

    monkeypatch.setattr(
        production, "_prototype_bundle_builder_sha256", lambda: "c" * 64
    )
    upgraded = production._prototype_bundle_root(job, summary)
    assert upgraded != first
    assert legacy_wrapper.read_bytes() == b"r17-wrapper-must-remain"


def test_prototype_checkpoint_sync_does_not_rehash_unchanged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "scratch" / "bundle"
    destination = tmp_path / "network" / "bundle"
    (source / "tree").mkdir(parents=True)
    wrapper = source / "tree" / "prototype.usda"
    wrapper.write_bytes(b"wrapper-one")
    validated: dict[
        str,
        tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ],
    ] = {}
    original_sha256_file = production._sha256_file
    hashed: list[Path] = []

    def counted_sha256(path: Path) -> str:
        hashed.append(path)
        return original_sha256_file(path)

    monkeypatch.setattr(production, "_sha256_file", counted_sha256)
    production._sync_prototype_bundle(source, destination, validated_files=validated)
    production._sync_prototype_bundle(source, destination, validated_files=validated)
    assert hashed == []

    destination_wrapper = destination / "tree" / "prototype.usda"
    previous_mtime = destination_wrapper.stat().st_mtime_ns
    destination_wrapper.write_bytes(b"wrapper-two")
    os.utime(
        destination_wrapper,
        ns=(previous_mtime + 1_000_000_000, previous_mtime + 1_000_000_000),
    )
    with pytest.raises(
        production.SimpleProductionError, match="prototype immuable divergent"
    ):
        production._sync_prototype_bundle(
            source, destination, validated_files=validated
        )
    assert len(hashed) == 2


def test_prototype_bundle_namespace_rejects_an_unsealed_catalog_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(production.SimpleProductionError, match="identité du catalogue"):
        production._prototype_bundle_root(
            tmp_path / "job", {"catalog_sha256": "not-a-sha256"}
        )


def test_parallel_scheduler_never_starts_more_than_four_and_stops_queue_on_error() -> (
    None
):
    active = 0
    maximum_active = 0
    started: list[int] = []
    release = threading.Event()
    lock = threading.Lock()
    stop = threading.Event()
    tiles = [
        (index, production.TilePlan(f"tile-{index}", (index * 500, 0)))
        for index in range(12)
    ]

    def work(index: int, _tile: production.TilePlan) -> str:
        nonlocal active, maximum_active
        with lock:
            started.append(index)
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if index == 0:
                raise RuntimeError("first tile failed")
            release.wait(timeout=0.2)
            return str(index)
        finally:
            with lock:
                active -= 1

    results = production._bounded_parallel_tile_results(
        tiles,
        worker_count=4,
        process_tile=work,
        stop_event=stop,
        heartbeat_seconds=0.01,
    )
    with pytest.raises(RuntimeError, match="first tile failed"):
        next(results)
    release.set()
    results.close()

    assert maximum_active <= 4
    assert set(started) <= {0, 1, 2, 3}
    assert stop.is_set()


def test_parallel_scheduler_fails_instead_of_heartbeating_forever() -> None:
    stop = threading.Event()
    release = threading.Event()
    tile = production.TilePlan("tile-stalled", (0, 0))

    def work(_index: int, _tile: production.TilePlan) -> str:
        release.wait(timeout=0.05)
        return "done"

    results = production._bounded_parallel_tile_results(
        [(0, tile), (1, production.TilePlan("tile-second", (500, 0)))],
        worker_count=2,
        process_tile=work,
        stop_event=stop,
        heartbeat_seconds=0.001,
        last_activity=lambda _index, _tile: 0.0,
        stall_timeout_seconds=5.0,
        monotonic=lambda: 10.0,
    )
    with pytest.raises(production.SimpleProductionError, match="inactive.*reprise"):
        next(results)
    release.set()
    results.close()
    assert stop.is_set()


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
        scratch_root=tmp_path / "scratch",
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
    prepared_source_roots: list[Path] = []

    def fake_context(**kwargs: object) -> SimpleNamespace:
        path = Path(kwargs["output_path"])
        path.write_text('{"context":"ok"}\n', encoding="utf-8")
        return SimpleNamespace(path=path, content_sha256="a" * 64, feature_counts={})

    def fake_prepare(**kwargs: object) -> SimpleNamespace:
        prepared_fixed_assets.append(list(kwargs["fixed_asset_placements"]))
        prepared_source_roots.append(Path(kwargs["output_root"]))
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
        bundle = Path(kwargs["asset_bundle_identity_root"]).resolve()
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
        assert callback is None
        return []

    compact_layout = {
        "payload_count": 1,
        "prototype_count": 1,
        "prototype_policy": "one_zone_definition_per_family_asset",
        "instance_policy": "tile_point_instancers_preserved",
    }

    def fake_compact_stage(root: Path, **_kwargs: object) -> Path:
        stage = root / production.ENTRY_STAGE
        stage.write_text('#usda 1.0\ndef Xform "FireViewerZone" {}\n', encoding="utf-8")
        (root / production.ZONE_STAGE_LAYOUT_NAME).write_text(
            json.dumps(compact_layout), encoding="utf-8"
        )
        return stage

    monkeypatch.setattr(
        production, "validate_simple_measured_tile_package", fake_validate
    )
    monkeypatch.setattr(production, "build_compact_zone_stage", fake_compact_stage)
    monkeypatch.setattr(
        production,
        "validate_compact_zone_stage",
        lambda *_args, **_kwargs: compact_layout,
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
    fixed_request = {
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
    }
    results = list(
        engine.run(
            latitude,
            longitude,
            1.0,
            fixed_asset_placements=fixed_request,
            progress_callback=lambda fraction, message: progress_events.append(
                (fraction, message)
            ),
        )
    )
    assert all(len(result) == 3 for result in results)
    archive = Path(results[-1][1])
    assert results[-1][2] == []
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
    network_job_root = config.work_root / "jobs" / receipt["zone_id"]
    assert network_job_root != job_root
    assert not (network_job_root / "packages").exists()
    checkpoint_root = (
        network_job_root
        / production.TILE_CHECKPOINT_COLLECTION_NAME
        / production.TILE_CHECKPOINT_VERSION
    )
    assert len(list(checkpoint_root.glob("*.zip"))) == receipt["tile_count"]
    assert len(list(checkpoint_root.glob("*.json"))) == receipt["tile_count"]
    assert not (network_job_root / production.ZONE_CONTEXT_NAME).exists()
    assert (network_job_root / production.ZONE_RECEIPT_NAME).is_file()
    assert not (network_job_root / production.ZIP_NAME).exists()
    assert not (network_job_root / production.BLEND_NAME).exists()
    assert receipt["tile_count"] > 1
    assert len(prepared_fixed_assets) == receipt["tile_count"]
    scratch_jobs = (config.scratch_root / "jobs").resolve()
    assert all(root.is_relative_to(scratch_jobs) for root in prepared_source_roots)
    scratch_job_names = {
        root.relative_to(scratch_jobs).parts[0] for root in prepared_source_roots
    }
    assert len(scratch_job_names) == 1
    assert next(iter(scratch_job_names)).startswith(f"{receipt['zone_id']}-")
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
    assert not any(name.startswith(prefix + "qa/gallery/") for name in names)
    assert any(name.startswith(prefix + "packages/") for name in names)
    assert any(name.startswith(prefix + "provenance/") for name in names)
    assert not any("orthophoto-1m.png" in name for name in names)
    assert not any("mnt-05m.tif" in name or "mns-05m.tif" in name for name in names)
    # The producer has already fully validated every freshly sealed package.
    # The engine only revalidates packages restored from a checkpoint.
    assert validated_requests == []
    messages = [message for _fraction, message in progress_events]
    assert any("Moteur embarqué validé" in message for message in messages)
    assert any("acquisitions et 3 compilations" in message for message in messages)
    assert any("MNT 0,5 m reçu" in message for message in messages)
    assert any("Placement MNS−MNT mesuré" in message for message in messages)
    assert any("Compression du pack autonome" in message for message in messages)
    assert any("scène autonome sans captures" in message for message in messages)

    prepared_count = len(prepared_source_roots)
    validated_requests.clear()
    resumed = list(
        engine.run(
            latitude,
            longitude,
            1.0,
            fixed_asset_placements=fixed_request,
        )
    )
    assert Path(resumed[-1][1]).is_file()
    assert len(prepared_source_roots) == prepared_count
    expected_tile_ids = {tile["tile_id"] for tile in receipt["tiles"]}
    assert {request["tile_id"] for request in validated_requests} == expected_tile_ids


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
    monkeypatch.setenv("FIREVIEWER_IMAGE_REFERENCE", "worker:r17")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=FakeOperation, HfApi=FakeApi),
    )
    first = production._publish_dataset_entry(
        config,
        job_root=tmp_path,
        plan=plan,
        receipt=receipt,
        archive=archive,
    )
    monkeypatch.setenv("FIREVIEWER_IMAGE_REFERENCE", "worker:r18")
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


def test_optional_dataset_publication_attempts_large_archive_only_once(
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
        dataset_publication_required=False,
    )
    receipt = {
        "build_id": "b" * 64,
        "tile_count": 625,
        "building_count": 16_000,
        "tree_count": 100_000,
    }
    production._write_json(tmp_path / production.ZONE_RECEIPT_NAME, receipt)
    archive = tmp_path / production.ZIP_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fireviewer-zone/zone.usda", "#usda 1.0\n")

    attempts: list[list[object]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret-not-for-files"

        def repo_info(self, *, repo_id: str, repo_type: str) -> SimpleNamespace:
            assert repo_id == config.dataset_id
            assert repo_type == "dataset"
            return SimpleNamespace(private=True)

        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            operations = list(kwargs["operations"])
            attempts.append(operations)
            raise TimeoutError(
                "Timeout: Request error: error decoding response body, domain: no-url"
            )

    class FakeOperation:
        def __init__(self, **kwargs: object) -> None:
            self.values = dict(kwargs)

    sleeps: list[float] = []
    progress: list[tuple[str, str]] = []
    monkeypatch.setenv("HF_TOKEN", "secret-not-for-files")
    monkeypatch.setattr(production.time, "sleep", sleeps.append)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=FakeOperation, HfApi=FakeApi),
    )

    publication = production._publish_dataset_entry(
        config,
        job_root=tmp_path,
        plan=plan,
        receipt=receipt,
        archive=archive,
        publication_progress=lambda phase, message: progress.append((phase, message)),
    )

    assert publication is not None
    assert publication["status"] == "failed_pending_retry"
    assert len(attempts) == 1
    assert sleeps == []
    assert [phase for phase, _message in progress] == [
        "dataset_publication_attempt",
    ]


def test_optional_dataset_publication_records_retry_without_blocking_archive(
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
        dataset_publication_required=False,
    )
    receipt = {
        "build_id": "f" * 64,
        "tile_count": 625,
        "building_count": 16_000,
        "tree_count": 100_000,
    }
    production._write_json(tmp_path / production.ZONE_RECEIPT_NAME, receipt)
    archive = tmp_path / production.ZIP_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fireviewer-zone/zone.usda", "#usda 1.0\n")

    class FailingApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret-not-for-files"

        def repo_info(self, *, repo_id: str, repo_type: str) -> SimpleNamespace:
            assert repo_id == config.dataset_id
            assert repo_type == "dataset"
            return SimpleNamespace(private=True)

        def create_commit(self, **_kwargs: object) -> SimpleNamespace:
            raise TimeoutError(
                "Timeout: Request error: error decoding response body, domain: no-url"
            )

    class FakeOperation:
        def __init__(self, **kwargs: object) -> None:
            self.values = dict(kwargs)

    monkeypatch.setenv("HF_TOKEN", "secret-not-for-files")
    monkeypatch.setattr(production.time, "sleep", lambda _delay: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=FakeOperation, HfApi=FailingApi),
    )

    publication = production._publish_dataset_entry(
        config,
        job_root=tmp_path,
        plan=plan,
        receipt=receipt,
        archive=archive,
    )

    assert publication is not None
    assert publication["status"] == "failed_pending_retry"
    assert publication["archive_sha256"] == production._sha256_file(archive)
    assert publication["captures"] == []
    assert archive.is_file()


def test_private_dataset_publication_does_not_retry_auth_failure(
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
        "build_id": "c" * 64,
        "tile_count": 1,
        "building_count": 0,
        "tree_count": 1,
    }
    production._write_json(tmp_path / production.ZONE_RECEIPT_NAME, receipt)
    archive = tmp_path / production.ZIP_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fireviewer-zone/zone.usda", "#usda 1.0\n")

    attempts = 0

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret-not-for-files"

        def repo_info(self, *, repo_id: str, repo_type: str) -> SimpleNamespace:
            return SimpleNamespace(private=True)

        def create_commit(self, **_kwargs: object) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("401 Unauthorized")

    class FakeOperation:
        def __init__(self, **kwargs: object) -> None:
            self.values = dict(kwargs)

    sleeps: list[float] = []
    monkeypatch.setenv("HF_TOKEN", "secret-not-for-files")
    monkeypatch.setattr(production.time, "sleep", sleeps.append)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=FakeOperation, HfApi=FakeApi),
    )

    with pytest.raises(production.SimpleProductionError, match="401 Unauthorized"):
        production._publish_dataset_entry(
            config,
            job_root=tmp_path,
            plan=plan,
            receipt=receipt,
            archive=archive,
        )
    assert attempts == 1
    assert sleeps == []


def test_final_zip_streams_files_and_skips_recompression_for_binary_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assembly = tmp_path / "assembly"
    (assembly / "shared").mkdir(parents=True)
    binary = assembly / "shared" / "prototype.usdc"
    binary.write_bytes(b"binary-usdc" * 250_000)
    text_stage = assembly / "zone.usda"
    text_stage.write_text("#usda 1.0\n", encoding="utf-8")

    def forbidden_read_bytes(_path: Path) -> bytes:
        raise AssertionError(
            "the final ZIP must stream files instead of buffering them"
        )

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    progress: list[tuple[int, int, int, int, str]] = []
    archive_path = production._write_zip(
        assembly,
        "GPS-STREAM",
        progress_callback=lambda *values: progress.append(values),
    )

    with zipfile.ZipFile(archive_path) as archive:
        binary_info = archive.getinfo("fireviewer-GPS-STREAM/shared/prototype.usdc")
        text_info = archive.getinfo("fireviewer-GPS-STREAM/zone.usda")
        assert binary_info.compress_type == zipfile.ZIP_STORED
        assert text_info.compress_type == zipfile.ZIP_DEFLATED
        with archive.open(binary_info) as stream:
            assert stream.read(11) == b"binary-usdc"
    assert progress
    assert progress[-1][0] == progress[-1][1]
    assert progress[-1][2] == progress[-1][3] == 2
    assert progress[-1][4] == "zone.usda"


def test_archive_budget_blocks_oversized_pack_before_zip_or_upload(
    tmp_path: Path,
) -> None:
    assembly = tmp_path / "assembly"
    assembly.mkdir()
    (assembly / "zone.blend").write_bytes(b"x" * 32)

    with pytest.raises(production.SimpleProductionError, match="hors budget"):
        production._write_archive_budget_receipt(assembly, 16)
    assert not (assembly / production.ARCHIVE_BUDGET_NAME).exists()
    assert not (assembly / production.ZIP_NAME).exists()


def test_archive_budget_receipt_records_blend_prototypes_tiles_and_payloads(
    tmp_path: Path,
) -> None:
    assembly = tmp_path / "assembly"
    (assembly / "packages" / "tile").mkdir(parents=True)
    (assembly / "shared" / "prototype").mkdir(parents=True)
    (assembly / production.ZONE_PAYLOAD_DIRECTORY).mkdir(parents=True)
    (assembly / "zone.blend").write_bytes(b"b" * 11)
    (assembly / "packages" / "tile" / "terrain.fvtg").write_bytes(b"t" * 13)
    (assembly / "shared" / "prototype" / "source.usdc").write_bytes(b"p" * 17)
    (assembly / production.ZONE_PAYLOAD_DIRECTORY / "meta.usda").write_bytes(b"u" * 19)
    (assembly / production.ENTRY_STAGE).write_bytes(b"s" * 23)

    receipt = production._write_archive_budget_receipt(assembly, 1024)

    assert receipt["input_byte_count"] == 83
    assert receipt["breakdown_bytes"] == {
        "standalone_blend": 11,
        "tile_packages": 13,
        "shared_prototypes": 17,
        "compact_openusd": 42,
        "other": 0,
    }
    assert receipt["status"] == "within_limit"


def test_final_archive_size_is_rechecked_before_publication(tmp_path: Path) -> None:
    assembly = tmp_path / "assembly"
    assembly.mkdir()
    (assembly / "empty.txt").write_bytes(b"")

    with pytest.raises(production.SimpleProductionError, match="Archive.*hors budget"):
        production._write_zip(assembly, "GPS-BUDGET", maximum_bytes=1)
    assert not (assembly / production.ZIP_NAME).exists()
    assert not (assembly / f".{production.ZIP_NAME}.part").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="worker links are Linux-only")
def test_local_assembly_and_zip_materialize_linked_files_once(tmp_path: Path) -> None:
    network = tmp_path / "network" / "jobs" / "GPS-LINKED"
    scratch = tmp_path / "scratch" / "jobs" / "GPS-LINKED"
    embedded = tmp_path / "image" / "tree.usd"
    embedded.parent.mkdir(parents=True)
    embedded.write_bytes(b"immutable-usd")
    (network / "packages" / "tile" / "scene").mkdir(parents=True)
    (network / "packages" / "tile" / "scene" / "scene.usda").write_bytes(b"#usda 1.0\n")
    prototype = network / "shared" / "prototypes" / "tree" / "source.usd"
    prototype.parent.mkdir(parents=True)
    prototype.symlink_to(embedded)
    (network / production.PLAN_NAME).write_text("{}", encoding="utf-8")
    scratch.mkdir(parents=True)

    assembly = production._prepare_local_assembly(network, scratch)
    assert (assembly / "shared/prototypes/tree/source.usd").is_symlink()
    archive = production._write_zip(assembly, "GPS-LINKED")

    with zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo("fireviewer-GPS-LINKED/shared/prototypes/tree/source.usd")
        assert info.external_attr >> 16 == 0o100644
        assert bundle.read(info) == b"immutable-usd"
        assert (
            bundle.namelist().count(
                "fireviewer-GPS-LINKED/shared/prototypes/tree/source.usd"
            )
            == 1
        )


def test_copy_result_sidecars_preserves_same_physical_receipt_and_copies_new_files(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network" / "jobs" / "GPS-LINKED"
    scratch = tmp_path / "scratch" / "jobs" / "GPS-LINKED"
    network.mkdir(parents=True)
    scratch.mkdir(parents=True)
    receipt = network / production.ZONE_RECEIPT_NAME
    receipt.write_bytes(b"network-receipt")
    (scratch / production.ZONE_RECEIPT_NAME).hardlink_to(receipt)
    (scratch / production.DATASET_ENTRY_NAME).write_bytes(b"dataset-entry")
    (scratch / production.DATASET_PUBLICATION_NAME).write_bytes(b"publication")

    production._copy_result_sidecars(scratch, network)

    assert receipt.read_bytes() == b"network-receipt"
    assert (network / production.DATASET_ENTRY_NAME).read_bytes() == b"dataset-entry"
    assert (
        network / production.DATASET_PUBLICATION_NAME
    ).read_bytes() == b"publication"


def test_copy_result_sidecars_accepts_receipt_copied_to_local_assembly(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network" / "jobs" / "GPS-SYMLINKED"
    scratch = tmp_path / "scratch" / "jobs" / "GPS-SYMLINKED"
    network.mkdir(parents=True)
    scratch.mkdir(parents=True)
    receipt = network / production.ZONE_RECEIPT_NAME
    receipt.write_bytes(b"network-receipt")
    assembly = production._prepare_local_assembly(network, scratch)
    local_receipt = assembly / production.ZONE_RECEIPT_NAME
    assert not local_receipt.is_symlink()
    assert local_receipt.read_bytes() == b"network-receipt"
    receipt.write_bytes(b"network-changed-after-local-copy")
    assert local_receipt.read_bytes() == b"network-receipt"
    (assembly / production.DATASET_ENTRY_NAME).write_bytes(b"dataset-entry")
    (assembly / production.DATASET_PUBLICATION_NAME).write_bytes(b"publication")

    production._copy_result_sidecars(assembly, network)

    assert receipt.read_bytes() == b"network-receipt"
    assert (network / production.DATASET_ENTRY_NAME).read_bytes() == b"dataset-entry"
    assert (
        network / production.DATASET_PUBLICATION_NAME
    ).read_bytes() == b"publication"


def test_gallery_items_are_empty_and_require_the_standalone_scene(
    tmp_path: Path,
) -> None:
    with pytest.raises(production.SimpleProductionError, match="zone.blend"):
        production._gallery_items(tmp_path)
    (tmp_path / production.BLEND_NAME).write_bytes(b"blend")
    assert production._gallery_items(tmp_path) == []


def test_blender_pack_surfaces_python_output_when_blend_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = production.ProductionConfig(
        work_root=tmp_path / "work",
        portable_root=tmp_path,
        asset_library=tmp_path / "asset-library.json",
        review_batch=tmp_path / "assets",
        elevation_revision="elevation-test",
        orthophoto_revision="orthophoto-test",
        context_revision="context-test",
        blender=Path("/opt/blender/blender"),
    )
    job_root = tmp_path / "job"
    job_root.mkdir()
    commands: list[tuple[str, ...]] = []

    class FakeProcess:
        stdout = StringIO("Python traceback from Blender\nprecise failure\n")

        @staticmethod
        def kill() -> None:
            return None

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout: float) -> int:
            del timeout
            return 0

    def fake_popen(command: tuple[str, ...], **_kwargs: object) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(production.subprocess, "Popen", fake_popen)

    with pytest.raises(production.SimpleProductionError, match="precise failure"):
        production._render_zone_gallery(config, job_root, True, None)

    assert commands
    assert commands[0][commands[0].index("--python-exit-code") + 1] == "1"
