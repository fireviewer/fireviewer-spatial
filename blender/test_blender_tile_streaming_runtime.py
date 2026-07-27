from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest

import blender_tile_streaming_runtime as runtime


class _FakeObject:
    def __init__(self, name: str, data: Any = None) -> None:
        self.name = name
        self.type = "EMPTY" if data is None else "MESH"
        self.location = [0.0, 0.0, 0.0]
        self.empty_display_type = "PLAIN_AXES"
        self.empty_display_size = 1.0


class _FakeObjects(dict[str, _FakeObject]):
    def new(self, name: str, data: Any) -> _FakeObject:
        result = _FakeObject(name, data)
        self[name] = result
        return result


class _ObjectLinks:
    def __init__(self) -> None:
        self.linked: list[_FakeObject] = []

    def link(self, value: _FakeObject) -> None:
        self.linked.append(value)


class _FakeScene(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__()
        self.collection = SimpleNamespace(objects=_ObjectLinks())
        self.camera = None
        self.pointer = 0xF1E0

    def as_pointer(self) -> int:
        return self.pointer


class _FakeTimers:
    def __init__(self) -> None:
        self.registrations: dict[Callable[[], float], dict[str, Any]] = {}

    def register(self, callback: Callable[[], float], **kwargs: Any) -> None:
        self.registrations[callback] = kwargs

    def unregister(self, callback: Callable[[], float]) -> None:
        self.registrations.pop(callback)

    def is_registered(self, callback: Callable[[], float]) -> bool:
        return callback in self.registrations


class _FakeHandlers:
    def __init__(self) -> None:
        self.load_post: list[Callable[[Any], None]] = []

    @staticmethod
    def persistent(callback: Callable[[Any], None]) -> Callable[[Any], None]:
        return callback


class _FakeBpy:
    def __init__(
        self, blend_path: Path, scene: _FakeScene, view_distance: float
    ) -> None:
        self.data = SimpleNamespace(
            filepath=str(blend_path),
            objects=_FakeObjects(),
            collections=[],
        )
        region_3d = SimpleNamespace(view_distance=view_distance)
        space = SimpleNamespace(region_3d=region_3d)
        area = SimpleNamespace(type="VIEW_3D", spaces=SimpleNamespace(active=space))
        self.context = SimpleNamespace(
            scene=scene,
            screen=SimpleNamespace(areas=[area]),
        )
        self.app = SimpleNamespace(
            timers=_FakeTimers(),
            handlers=_FakeHandlers(),
        )


def _tile(identifier: str, west: float, south: float) -> dict[str, Any]:
    return {
        "id": identifier,
        "bounds_l93_m": [west, south, west + 500.0, south + 500.0],
        "status": {"state": "ready"},
    }


def _configure_scene(
    scene: _FakeScene,
    *,
    manifest_relative_path: str,
    detail_radius_m: float = 500.0,
    debounce_ticks: int = 1,
    maximum_load_failure_count: int = 3,
    load_retry_backoff_ticks: int = 2,
) -> None:
    scene["origin_l93_x_m"] = 1_000.0
    scene["origin_l93_y_m"] = 2_000.0
    scene[runtime.CONFIG_PROPERTY] = json.dumps(
        {
            "schema": runtime.CONFIG_SCHEMA,
            "manifest_path": manifest_relative_path,
            "global_package_path": "global-package.json.gz",
            "detail_radius_m": detail_radius_m,
            "detail_view_distance_max_m": 500.0,
            "detail_view_footprint_factor": 1.2,
            "maximum_resident_tile_count": 16,
            "debounce_ticks": debounce_ticks,
            "maximum_load_failure_count": maximum_load_failure_count,
            "load_retry_backoff_ticks": load_retry_backoff_ticks,
            "timer_interval_s": 0.1,
            "focus_object_name": runtime.FOCUS_OBJECT_NAME,
        }
    )


def _install_runtime_modules(
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, Any],
    event_log: list[tuple[Any, ...]],
    *,
    materialize_error: BaseException | None = None,
) -> None:
    tiled_scene = ModuleType("tiled_scene")
    tiled_scene.load_global_05m_manifest = lambda _path: manifest
    build_control = ModuleType("build_control_scene")

    def activate_global_fallback(_bpy: Any) -> None:
        event_log.append(("fallback",))

    def evict_global_tiles(_bpy: Any, tile_ids: tuple[str, ...]) -> None:
        event_log.append(("evict", tuple(tile_ids)))

    def materialize_global_tile(_bpy: Any, tile_id: str) -> None:
        event_log.append(("load", tile_id))
        if materialize_error is not None:
            raise materialize_error

    def publish_detail_tiles(_bpy: Any, tile_ids: tuple[str, ...]) -> None:
        event_log.append(("publish", tuple(tile_ids)))

    build_control.activate_global_fallback = activate_global_fallback
    build_control.evict_global_tiles = evict_global_tiles
    build_control.materialize_global_tile = materialize_global_tile
    build_control.publish_detail_tiles = publish_detail_tiles
    monkeypatch.setitem(__import__("sys").modules, "tiled_scene", tiled_scene)
    monkeypatch.setitem(__import__("sys").modules, "build_control_scene", build_control)


def _make_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, Any],
    event_log: list[tuple[Any, ...]],
    *,
    view_distance: float = 100.0,
    materialize_error: BaseException | None = None,
) -> tuple[_FakeBpy, _FakeScene, Callable[[], float]]:
    blend = tmp_path / "blender" / "scene.blend"
    manifest_file = tmp_path / "global-05m" / "production-manifest.json"
    global_package = blend.parent / "global-package.json.gz"
    blend.parent.mkdir(parents=True)
    manifest_file.parent.mkdir(parents=True)
    blend.touch()
    global_package.touch()
    manifest_file.write_text("{}", encoding="utf-8")
    scene = _FakeScene()
    _configure_scene(
        scene,
        manifest_relative_path="../global-05m/production-manifest.json",
    )
    fake_bpy = _FakeBpy(blend, scene, view_distance)
    _install_runtime_modules(
        monkeypatch,
        manifest,
        event_log,
        materialize_error=materialize_error,
    )
    runtime.register(fake_bpy)
    callback = next(iter(fake_bpy.app.timers.registrations))
    return fake_bpy, scene, callback


@pytest.fixture(autouse=True)
def _isolate_adapter_globals() -> Any:
    yield
    if runtime._runtime_bpy is not None:
        runtime.unregister(runtime._runtime_bpy)


def test_register_streams_atomically_from_local_focus_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    fake_bpy, scene, callback = _make_runtime(tmp_path, monkeypatch, manifest, events)

    runtime.register(fake_bpy)
    assert len(fake_bpy.app.handlers.load_post) == 1
    assert len(fake_bpy.app.timers.registrations) == 1
    assert fake_bpy.app.timers.registrations[callback]["persistent"] is True

    event_counts = [len(events)]
    for _ in range(4):
        assert callback() == pytest.approx(0.1)
        event_counts.append(len(events))

    assert all(
        after - before <= 1 for before, after in zip(event_counts, event_counts[1:])
    )
    assert events == [
        ("fallback",),
        ("fallback",),
        ("load", "detail"),
        ("publish", ("detail",)),
    ]
    focus = fake_bpy.data.objects[runtime.FOCUS_OBJECT_NAME]
    assert focus.type == "EMPTY"
    assert scene.collection.objects.linked == [focus]
    telemetry = json.loads(scene[runtime.TELEMETRY_PROPERTY])
    assert telemetry["status"] == "ready"
    assert telemetry["focus_l93_m"] == [1_000.0, 2_000.0]
    assert telemetry["planner_state"]["published_tile_ids"] == ["detail"]
    assert Path(telemetry["manifest_path"]).is_absolute()
    assert Path(telemetry["global_package_path"]).is_absolute()
    assert scene["fireviewer_runtime_tile_manifest_path"] == str(
        (tmp_path / "global-05m" / "production-manifest.json").resolve()
    )
    assert scene["fireviewer_runtime_global_package_path"] == str(
        (tmp_path / "blender" / "global-package.json.gz").resolve()
    )
    assert scene["fireviewer_runtime_view_distance_m"] == pytest.approx(100.0)
    assert telemetry["detail_view_footprint_factor"] == pytest.approx(1.2)
    assert telemetry["detail_required_coverage_radius_m"] == pytest.approx(120.0)
    assert runtime._runtime_state is not None
    assert runtime._runtime_state.scene_identity == scene.pointer

    runtime.register(fake_bpy)
    callback()
    assert events[-1] == ("publish", ("detail",))
    runtime.unregister(fake_bpy)
    runtime.unregister(fake_bpy)
    runtime.unregister()
    assert fake_bpy.app.handlers.load_post == []
    assert fake_bpy.app.timers.registrations == {}


def test_far_view_restores_fallback_then_evicts_without_partial_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    fake_bpy, scene, callback = _make_runtime(tmp_path, monkeypatch, manifest, events)
    for _ in range(4):
        callback()
    assert events[-1] == ("publish", ("detail",))

    view = fake_bpy.context.screen.areas[0].spaces.active.region_3d
    view.view_distance = 501.0
    before = len(events)
    callback()
    assert events[before:] == [("fallback",)]
    callback()
    assert events[before:] == [("fallback",), ("evict", ("detail",))]
    callback()

    telemetry = json.loads(scene[runtime.TELEMETRY_PROPERTY])
    assert telemetry["detail_requested"] is False
    assert telemetry["planner_state"]["resident_tile_ids"] == []
    assert telemetry["planner_state"]["published_tile_ids"] == []
    assert telemetry["planner_state"]["global_fallback_visible"] is True


def test_detail_requires_both_near_band_and_complete_view_footprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    fake_bpy, scene, callback = _make_runtime(
        tmp_path,
        monkeypatch,
        manifest,
        events,
        view_distance=450.0,
    )

    callback()
    callback()
    telemetry = json.loads(scene[runtime.TELEMETRY_PROPERTY])
    assert telemetry["detail_requested"] is False
    assert telemetry["detail_required_coverage_radius_m"] == pytest.approx(540.0)
    assert events == [("fallback",)]

    view = fake_bpy.context.screen.areas[0].spaces.active.region_3d
    view.view_distance = 500.0
    callback()
    assert json.loads(scene[runtime.TELEMETRY_PROPERTY])["detail_requested"] is False
    assert events == [("fallback",)]

    view.view_distance = 400.0
    callback()
    callback()
    callback()
    assert events[-3:] == [
        ("fallback",),
        ("load", "detail"),
        ("publish", ("detail",)),
    ]


def test_file_residents_are_evicted_during_fail_safe_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    fake_bpy, scene, callback = _make_runtime(
        tmp_path,
        monkeypatch,
        manifest,
        events,
        view_distance=501.0,
    )
    fake_bpy.data.collections.append(
        {
            "fireviewer_tile_id": "detail",
            "fireviewer_tile_loaded": True,
        }
    )

    callback()
    assert events == [("fallback",)]
    callback()
    assert events == [("fallback",), ("evict", ("detail",))]
    callback()

    telemetry = json.loads(scene[runtime.TELEMETRY_PROPERTY])
    assert telemetry["bootstrap_phase"] == "ready"
    assert telemetry["bootstrap_resident_tile_ids"] == []
    assert telemetry["planner_state"]["resident_tile_ids"] == []
    assert telemetry["detail_requested"] is False


def test_failed_load_is_reported_retried_and_never_adds_a_second_tick_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    _fake_bpy, scene, callback = _make_runtime(
        tmp_path,
        monkeypatch,
        manifest,
        events,
        materialize_error=RuntimeError("corrupt tile"),
    )
    callback()
    callback()
    before = len(events)
    callback()

    assert events[before:] == [("load", "detail")]
    telemetry = json.loads(scene[runtime.TELEMETRY_PROPERTY])
    assert telemetry["status"] == "error"
    assert telemetry["last_error"] == "RuntimeError: corrupt tile"
    assert telemetry["planner_state"]["global_fallback_visible"] is True
    assert telemetry["planner_state"]["published_tile_ids"] == []
    assert telemetry["planner_telemetry"]["load_failure_count"] == 1

    callback()
    callback()
    assert events[before:] == [("load", "detail")]
    telemetry = json.loads(scene[runtime.TELEMETRY_PROPERTY])
    assert telemetry["planner_state"]["load_retry_backoff_remaining_ticks"] == 0

    callback()
    assert events[before:] == [("load", "detail"), ("load", "detail")]


def test_repeated_failed_load_is_quarantined_after_three_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    _fake_bpy, scene, callback = _make_runtime(
        tmp_path,
        monkeypatch,
        manifest,
        events,
        materialize_error=RuntimeError("corrupt tile"),
    )

    callback()
    callback()
    for _ in range(3):
        callback()
        callback()
        callback()
    event_count_after_quarantine = len(events)
    for _ in range(8):
        callback()

    assert events == [
        ("fallback",),
        ("fallback",),
        ("load", "detail"),
        ("load", "detail"),
        ("load", "detail"),
    ]
    assert len(events) == event_count_after_quarantine
    telemetry = json.loads(scene[runtime.TELEMETRY_PROPERTY])
    assert telemetry["status"] == "error"
    assert telemetry["planner_state"]["quarantined_tile_ids"] == ["detail"]
    assert telemetry["planner_state"]["published_tile_ids"] == []
    assert telemetry["planner_telemetry"]["load_attempt_count"] == 3
    assert telemetry["planner_telemetry"]["load_failure_count"] == 3
    assert telemetry["planner_telemetry"]["quarantined_tile_count"] == 1


def test_invalid_relative_config_stops_without_mutating_the_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    _fake_bpy, scene, callback = _make_runtime(tmp_path, monkeypatch, manifest, events)
    payload = json.loads(scene[runtime.CONFIG_PROPERTY])
    payload["manifest_path"] = str((tmp_path / "absolute.json").resolve())
    scene[runtime.CONFIG_PROPERTY] = json.dumps(payload)

    baseline = dict(scene)
    assert callback() is None

    assert events == []
    assert dict(scene) == baseline
    assert runtime._runtime_state is None


def test_missing_contract_registers_only_the_load_handler_and_never_mutates(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "plain.blend"
    blend.touch()
    scene = _FakeScene()
    fake_bpy = _FakeBpy(blend, scene, 100.0)
    baseline = dict(scene)

    runtime.register(fake_bpy)

    assert len(fake_bpy.app.handlers.load_post) == 1
    assert fake_bpy.app.timers.registrations == {}
    assert runtime._timer_tick() is None
    assert dict(scene) == baseline


def test_missing_runtime_resource_stops_without_fallback_or_scene_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    fake_bpy, scene, callback = _make_runtime(tmp_path, monkeypatch, manifest, events)
    manifest_path = (
        Path(fake_bpy.data.filepath).parent / "../global-05m/production-manifest.json"
    ).resolve()
    manifest_path.unlink()
    baseline = dict(scene)

    assert callback() is None

    assert events == []
    assert dict(scene) == baseline
    assert runtime._runtime_state is None


def test_load_post_forgets_file_state_and_restores_a_missing_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    manifest = {"tiles": [_tile("detail", 1_000.0, 2_000.0)]}
    fake_bpy, _scene, callback = _make_runtime(tmp_path, monkeypatch, manifest, events)
    callback()
    assert runtime._runtime_state is not None
    fake_bpy.app.timers.unregister(callback)

    fake_bpy.app.handlers.load_post[0](None)

    assert runtime._runtime_state is None
    assert fake_bpy.app.timers.is_registered(callback)
    registration = fake_bpy.app.timers.registrations[callback]
    assert registration["persistent"] is True

    scene = fake_bpy.context.scene
    scene.pop(runtime.CONFIG_PROPERTY)
    fake_bpy.app.handlers.load_post[0](None)
    assert runtime._runtime_state is None
    assert not fake_bpy.app.timers.is_registered(callback)


def test_import_contract_never_binds_a_bpy_module_name() -> None:
    assert "bpy" not in runtime.__dict__
