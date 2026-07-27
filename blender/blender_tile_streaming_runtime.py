"""Blender adapter for the pure :mod:`tile_streaming` state machine.

This module stays importable in ordinary CPython: :mod:`bpy` is resolved only
when :func:`register`, :func:`unregister`, or the registered timer executes.
It never starts a thread; every Blender mutation runs on Blender's main thread
through ``bpy.app.timers``.

Scene contract
--------------
``scene["fireviewer_tile_streaming_config_json"]`` must contain a JSON object::

    {
      "schema": "fireviewer.blender-tile-streaming.v1",
      "manifest_path": "../global-05m/production-manifest.json",
      "global_package_path": "synthetic-zone-global-control.json.gz",
      "detail_radius_m": 750.0,
      "detail_view_distance_max_m": 1500.0,
      "detail_view_footprint_factor": 1.2,
      "maximum_resident_tile_count": 16,
      "debounce_ticks": 2,
      "maximum_load_failure_count": 3,
      "load_retry_backoff_ticks": 2,
      "timer_interval_s": 0.25,
      "focus_object_name": "FireViewerFocus"
    }

Resource paths are deliberately relative to the saved ``.blend`` file.
Lambert-93 origin coordinates come from ``scene["origin_l93_x_m"]`` and
``scene["origin_l93_y_m"]``.  The focus Empty stores local metric coordinates,
so its L93 position is ``origin + (location.x, location.y)``.  If it does not
exist, the adapter creates the Empty named ``FireViewerFocus``.

The adapter expects these public functions in :mod:`build_control_scene`::

    activate_global_fallback(bpy)
    evict_global_tiles(bpy, tile_ids)
    materialize_global_tile(bpy, tile_id)
    publish_detail_tiles(bpy, tile_ids)

Each timer tick executes at most one of those callbacks.  A transition exposes
the global context before any eviction or load, and publication occurs only
after the complete target is resident.  Failures are committed to the planner,
kept on the global fallback, and recorded as JSON in
``scene["fireviewer_tile_streaming_telemetry_json"]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tile_streaming import (
    ACTIVATE_GLOBAL_FALLBACK,
    DEFAULT_LOAD_RETRY_BACKOFF_TICKS,
    DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT,
    EVICT_TILES,
    LOAD_TILE,
    NOOP,
    PUBLISH_DETAIL,
    StreamingAction,
    TileStreamingPlanner,
)


CONFIG_PROPERTY = "fireviewer_tile_streaming_config_json"
TELEMETRY_PROPERTY = "fireviewer_tile_streaming_telemetry_json"
CONFIG_SCHEMA = "fireviewer.blender-tile-streaming.v1"
TELEMETRY_SCHEMA = "fireviewer.blender-tile-streaming-telemetry.v1"
DEFAULT_TIMER_INTERVAL_S = 0.25
DEFAULT_DETAIL_VIEW_DISTANCE_MAX_M = 1_500.0
DEFAULT_DETAIL_VIEW_FOOTPRINT_FACTOR = 1.2
FOCUS_OBJECT_NAME = "FireViewerFocus"


def _finite_number(value: Any, field_name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number >= {minimum}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a finite number >= {minimum}"
        ) from error
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field_name} must be a finite number >= {minimum}")
    return result


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _strictly_positive_number(value: Any, field_name: str) -> float:
    result = _finite_number(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be a finite number > 0")
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated scene-owned runtime settings."""

    manifest_path: Path
    global_package_path: Path
    detail_radius_m: float
    detail_view_distance_max_m: float
    detail_view_footprint_factor: float
    maximum_resident_tile_count: int
    debounce_ticks: int
    maximum_load_failure_count: int
    load_retry_backoff_ticks: int
    timer_interval_s: float
    focus_object_name: str


@dataclass
class _RuntimeState:
    scene_identity: int
    config: RuntimeConfig
    manifest: Mapping[str, Any]
    planner: TileStreamingPlanner
    focus_object: Any
    inactive_focus_l93_m: tuple[float, float]
    build_control_scene: Any
    bootstrap_phase: str
    bootstrap_resident_tile_ids: tuple[str, ...]
    last_action: StreamingAction | None = None
    last_error: str | None = None


_runtime_bpy: Any | None = None
_runtime_state: _RuntimeState | None = None
_persistent_load_handler: Any | None = None


def _resolve_bpy(bpy_module: Any | None = None) -> Any:
    return bpy_module if bpy_module is not None else importlib.import_module("bpy")


def _scene_value(scene: Any, key: str, default: Any = None) -> Any:
    getter = getattr(scene, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return scene[key]
    except (KeyError, TypeError):
        return default


def _scene_has_fireviewer_contract(scene: Any) -> bool:
    """Return whether a scene explicitly opts into this persistent runtime."""

    encoded = _scene_value(scene, CONFIG_PROPERTY)
    if not isinstance(encoded, str) or not encoded.strip():
        return False
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, Mapping) and payload.get("schema") == CONFIG_SCHEMA


def _scene_identity(scene: Any) -> int:
    """Use Blender's stable RNA pointer, with a CPython-test fallback."""

    as_pointer = getattr(scene, "as_pointer", None)
    if callable(as_pointer):
        return int(as_pointer())
    return id(scene)


def _parse_scene_config(bpy_module: Any, scene: Any) -> RuntimeConfig:
    encoded = _scene_value(scene, CONFIG_PROPERTY)
    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError(f"scene[{CONFIG_PROPERTY!r}] must be a JSON object string")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"scene[{CONFIG_PROPERTY!r}] is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"scene[{CONFIG_PROPERTY!r}] must encode an object")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"scene[{CONFIG_PROPERTY!r}].schema must be {CONFIG_SCHEMA!r}")

    blend_path = Path(str(getattr(bpy_module.data, "filepath", "")))
    if not str(blend_path) or str(blend_path) == ".":
        raise ValueError("Tile streaming requires a saved .blend file")

    def relative_resource(field_name: str) -> Path:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty relative path")
        portable = Path(value)
        if portable.is_absolute():
            raise ValueError(f"{field_name} must be relative to the .blend file")
        resolved = (blend_path.resolve().parent / portable).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    manifest_path = relative_resource("manifest_path")
    global_package_path = relative_resource("global_package_path")

    focus_name = payload.get("focus_object_name", FOCUS_OBJECT_NAME)
    if not isinstance(focus_name, str) or not focus_name.strip():
        raise ValueError("focus_object_name must be a non-empty string")
    return RuntimeConfig(
        manifest_path=manifest_path,
        global_package_path=global_package_path,
        detail_radius_m=_finite_number(
            payload.get("detail_radius_m"), "detail_radius_m"
        ),
        detail_view_distance_max_m=_finite_number(
            payload.get(
                "detail_view_distance_max_m",
                _scene_value(
                    scene,
                    "scene_distance_lod_near_max_m",
                    DEFAULT_DETAIL_VIEW_DISTANCE_MAX_M,
                ),
            ),
            "detail_view_distance_max_m",
        ),
        detail_view_footprint_factor=_strictly_positive_number(
            payload.get(
                "detail_view_footprint_factor",
                DEFAULT_DETAIL_VIEW_FOOTPRINT_FACTOR,
            ),
            "detail_view_footprint_factor",
        ),
        maximum_resident_tile_count=_positive_integer(
            payload.get("maximum_resident_tile_count", 16),
            "maximum_resident_tile_count",
        ),
        debounce_ticks=_positive_integer(
            payload.get("debounce_ticks", 2), "debounce_ticks"
        ),
        maximum_load_failure_count=_positive_integer(
            payload.get(
                "maximum_load_failure_count",
                DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT,
            ),
            "maximum_load_failure_count",
        ),
        load_retry_backoff_ticks=_positive_integer(
            payload.get(
                "load_retry_backoff_ticks",
                DEFAULT_LOAD_RETRY_BACKOFF_TICKS,
            ),
            "load_retry_backoff_ticks",
        ),
        timer_interval_s=_finite_number(
            payload.get("timer_interval_s", DEFAULT_TIMER_INTERVAL_S),
            "timer_interval_s",
            minimum=0.01,
        ),
        focus_object_name=focus_name.strip(),
    )


def _scene_origin_l93_m(scene: Any) -> tuple[float, float]:
    return (
        _finite_number(
            _scene_value(scene, "origin_l93_x_m"),
            "scene.origin_l93_x_m",
            minimum=-math.inf,
        ),
        _finite_number(
            _scene_value(scene, "origin_l93_y_m"),
            "scene.origin_l93_y_m",
            minimum=-math.inf,
        ),
    )


def _ensure_focus_empty(bpy_module: Any, scene: Any, name: str) -> Any:
    focus = bpy_module.data.objects.get(name)
    if focus is None:
        focus = bpy_module.data.objects.new(name, None)
        scene.collection.objects.link(focus)
        focus.empty_display_type = "SPHERE"
        focus.empty_display_size = 20.0
    if getattr(focus, "type", None) != "EMPTY":
        raise TypeError(f"{name!r} must be a Blender Empty")
    return focus


def _resident_tile_ids(bpy_module: Any) -> tuple[str, ...]:
    residents: set[str] = set()
    for collection in bpy_module.data.collections:
        tile_id = _scene_value(collection, "fireviewer_tile_id")
        if tile_id and bool(_scene_value(collection, "fireviewer_tile_loaded", False)):
            residents.add(str(tile_id))
    return tuple(sorted(residents))


def _inactive_focus(
    manifest: Mapping[str, Any], detail_radius_m: float
) -> tuple[float, float]:
    ready_bounds = [
        tile["bounds_l93_m"]
        for tile in manifest.get("tiles", [])
        if tile.get("status", {}).get("state") == "ready"
    ]
    if not ready_bounds:
        return (detail_radius_m + 1.0, detail_radius_m + 1.0)
    return (
        max(float(bounds[2]) for bounds in ready_bounds) + detail_radius_m + 1.0,
        max(float(bounds[3]) for bounds in ready_bounds) + detail_radius_m + 1.0,
    )


def _load_runtime_state(bpy_module: Any, scene: Any) -> _RuntimeState:
    config = _parse_scene_config(bpy_module, scene)
    tiled_scene = importlib.import_module("tiled_scene")
    manifest = tiled_scene.load_global_05m_manifest(config.manifest_path)
    build_control_scene = importlib.import_module("build_control_scene")
    residents = _resident_tile_ids(bpy_module)
    planner = TileStreamingPlanner(
        manifest,
        maximum_resident_tile_count=config.maximum_resident_tile_count,
        debounce_ticks=config.debounce_ticks,
        maximum_load_failure_count=config.maximum_load_failure_count,
        load_retry_backoff_ticks=config.load_retry_backoff_ticks,
        # Bootstrap first returns every loaded file to a known global-only
        # state, so the pure planner starts from the matching empty residency.
        resident_tile_ids=(),
        published_tile_ids=(),
        global_fallback_visible=True,
    )
    focus = _ensure_focus_empty(bpy_module, scene, config.focus_object_name)
    # Public build_control_scene callbacks deliberately read scene properties,
    # not adapter globals, so a saved file remains inspectable and portable.
    # These mutations happen only after every external resource and module has
    # been validated above.
    scene["fireviewer_runtime_tile_manifest_path"] = str(config.manifest_path)
    scene["fireviewer_runtime_global_package_path"] = str(config.global_package_path)
    return _RuntimeState(
        scene_identity=_scene_identity(scene),
        config=config,
        manifest=manifest,
        planner=planner,
        focus_object=focus,
        inactive_focus_l93_m=_inactive_focus(manifest, config.detail_radius_m),
        build_control_scene=build_control_scene,
        bootstrap_phase="fallback",
        bootstrap_resident_tile_ids=residents,
    )


def _focus_l93_m(scene: Any, focus_object: Any) -> tuple[float, float]:
    origin_x, origin_y = _scene_origin_l93_m(scene)
    location = focus_object.location
    local_x = _finite_number(
        location[0], "FireViewerFocus.location.x", minimum=-math.inf
    )
    local_y = _finite_number(
        location[1], "FireViewerFocus.location.y", minimum=-math.inf
    )
    return origin_x + local_x, origin_y + local_y


def _view_distance_m(bpy_module: Any, scene: Any, focus_object: Any) -> float:
    screen = getattr(getattr(bpy_module, "context", None), "screen", None)
    for area in getattr(screen, "areas", ()) if screen is not None else ():
        if getattr(area, "type", None) != "VIEW_3D":
            continue
        region = getattr(getattr(area, "spaces", None), "active", None)
        region_3d = getattr(region, "region_3d", None)
        value = getattr(region_3d, "view_distance", None)
        if value is not None:
            return _finite_number(value, "VIEW_3D.view_distance")

    camera = getattr(scene, "camera", None)
    if camera is not None:
        delta = [
            float(camera.location[index]) - float(focus_object.location[index])
            for index in range(3)
        ]
        return _finite_number(
            math.sqrt(sum(value * value for value in delta)), "camera_distance"
        )
    return _finite_number(
        _scene_value(scene, "scene_distance_lod_view_distance_m", 0.0),
        "scene.scene_distance_lod_view_distance_m",
    )


def _detail_requested(config: RuntimeConfig, view_distance_m: float) -> bool:
    """Fail closed unless the configured detail radius covers the view."""

    return bool(
        view_distance_m < config.detail_view_distance_max_m
        and view_distance_m * config.detail_view_footprint_factor
        <= config.detail_radius_m
    )


def _execute_action(
    state: _RuntimeState, bpy_module: Any, action: StreamingAction
) -> None:
    callbacks = state.build_control_scene
    if action.kind == ACTIVATE_GLOBAL_FALLBACK:
        callbacks.activate_global_fallback(bpy_module)
    elif action.kind == EVICT_TILES:
        callbacks.evict_global_tiles(bpy_module, action.tile_ids)
    elif action.kind == LOAD_TILE:
        callbacks.materialize_global_tile(bpy_module, action.tile_ids[0])
    elif action.kind == PUBLISH_DETAIL:
        callbacks.publish_detail_tiles(bpy_module, action.tile_ids)
    elif action.kind != NOOP:
        raise RuntimeError(f"Unsupported streaming action: {action.kind}")


def _telemetry_payload(
    state: _RuntimeState | None,
    *,
    focus_l93_m: Sequence[float] | None = None,
    view_distance_m: float | None = None,
    detail_requested: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": TELEMETRY_SCHEMA,
        "status": "error" if error else "ready",
        "last_error": error,
    }
    if state is None:
        return payload
    payload.update(
        {
            "manifest_path": state.config.manifest_path.as_posix(),
            "global_package_path": state.config.global_package_path.as_posix(),
            "focus_l93_m": list(focus_l93_m) if focus_l93_m is not None else None,
            "view_distance_m": view_distance_m,
            "detail_requested": detail_requested,
            "detail_view_footprint_factor": (state.config.detail_view_footprint_factor),
            "detail_required_coverage_radius_m": (
                None
                if view_distance_m is None
                else view_distance_m * state.config.detail_view_footprint_factor
            ),
            "last_action": (
                asdict(state.last_action) if state.last_action is not None else None
            ),
            "bootstrap_phase": state.bootstrap_phase,
            "bootstrap_resident_tile_ids": list(state.bootstrap_resident_tile_ids),
            "planner_state": asdict(state.planner.state),
            "planner_telemetry": asdict(state.planner.telemetry),
        }
    )
    return payload


def _write_telemetry(scene: Any, payload: Mapping[str, Any]) -> None:
    scene[TELEMETRY_PROPERTY] = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def _fail_safe_before_action(
    bpy_module: Any, scene: Any, error: BaseException
) -> float:
    """Use the tick's only callback to restore global context after setup failure."""

    global _runtime_state
    previous_state = _runtime_state
    message = f"{type(error).__name__}: {error}"
    fallback_error: str | None = None
    try:
        callbacks = importlib.import_module("build_control_scene")
        callbacks.activate_global_fallback(bpy_module)
    except Exception as fallback_exception:  # noqa: BLE001 - timer must stay alive
        fallback_error = (
            f"; fallback failed: {type(fallback_exception).__name__}: "
            f"{fallback_exception}"
        )
    _write_telemetry(
        scene,
        _telemetry_payload(
            previous_state,
            error=message + (fallback_error or ""),
        ),
    )
    # The callback changed Blender visibility outside the planner handshake.
    # Re-bootstrap next tick so planner state can never claim detail is still
    # published while the real scene is showing the global fallback.
    _runtime_state = None
    return (
        previous_state.config.timer_interval_s
        if previous_state is not None
        else DEFAULT_TIMER_INTERVAL_S
    )


def _timer_tick() -> float | None:
    """Run one planner observation and at most one Blender mutation."""

    global _runtime_state
    bpy_module = _resolve_bpy(_runtime_bpy)
    scene = bpy_module.context.scene
    if not _scene_has_fireviewer_contract(scene):
        # This callback is persistent by design, but it must never mutate an
        # unrelated .blend file. Returning None unregisters a Blender timer;
        # the persistent load_post handler can reactivate it for a later scene
        # that explicitly carries the FireViewer contract.
        _runtime_state = None
        return None
    action_attempted = False
    try:
        if _runtime_state is None or _runtime_state.scene_identity != _scene_identity(
            scene
        ):
            # Never report planner data from the previously loaded scene when
            # parsing the new scene's relative configuration fails.
            _runtime_state = None
            try:
                _runtime_state = _load_runtime_state(bpy_module, scene)
            except Exception:  # noqa: BLE001 - irrecoverable scene contract
                # A missing/invalid local resource cannot become healthy by
                # polling every 250 ms. Stop without changing scene visibility
                # or properties; load_post/register can retry after a new file
                # or an explicit operator action.
                _runtime_state = None
                return None
            # Initialization has no planner action yet.  Its one mutation makes
            # a loaded file fail-safe before streaming starts on the next tick.
            initial_focus = _focus_l93_m(scene, _runtime_state.focus_object)
            initial_view_distance = _view_distance_m(
                bpy_module, scene, _runtime_state.focus_object
            )
            scene["fireviewer_runtime_view_distance_m"] = initial_view_distance
            action_attempted = True
            _runtime_state.build_control_scene.activate_global_fallback(bpy_module)
            _runtime_state.bootstrap_phase = (
                "eviction" if _runtime_state.bootstrap_resident_tile_ids else "ready"
            )
            _write_telemetry(
                scene,
                _telemetry_payload(
                    _runtime_state,
                    focus_l93_m=initial_focus,
                    view_distance_m=initial_view_distance,
                    detail_requested=_detail_requested(
                        _runtime_state.config, initial_view_distance
                    ),
                ),
            )
            return _runtime_state.config.timer_interval_s

        state = _runtime_state
        focus = _focus_l93_m(scene, state.focus_object)
        view_distance = _view_distance_m(bpy_module, scene, state.focus_object)
        scene["fireviewer_runtime_view_distance_m"] = view_distance
        detail_requested = _detail_requested(state.config, view_distance)
        if state.bootstrap_phase == "eviction":
            action_attempted = True
            try:
                state.build_control_scene.evict_global_tiles(
                    bpy_module, state.bootstrap_resident_tile_ids
                )
            except Exception as error:  # noqa: BLE001 - retry next timer tick
                state.last_error = f"{type(error).__name__}: {error}"
            else:
                state.last_error = None
                state.bootstrap_resident_tile_ids = ()
                state.bootstrap_phase = "ready"
            _write_telemetry(
                scene,
                _telemetry_payload(
                    state,
                    focus_l93_m=focus,
                    view_distance_m=view_distance,
                    detail_requested=detail_requested,
                    error=state.last_error,
                ),
            )
            return state.config.timer_interval_s
        observation = focus if detail_requested else state.inactive_focus_l93_m
        action = state.planner.tick(observation, state.config.detail_radius_m)
        state.last_action = action
        if action.kind != NOOP:
            action_attempted = True
            try:
                _execute_action(state, bpy_module, action)
            except Exception as error:  # noqa: BLE001 - commit and retry safely
                message = f"{type(error).__name__}: {error}"
                state.last_error = message
                state.planner.commit(action, succeeded=False, error=message)
            else:
                state.last_error = None
                state.planner.commit(action, succeeded=True)
        _write_telemetry(
            scene,
            _telemetry_payload(
                state,
                focus_l93_m=focus,
                view_distance_m=view_distance,
                detail_requested=detail_requested,
                error=state.last_error,
            ),
        )
        return state.config.timer_interval_s
    except Exception as error:  # noqa: BLE001 - Blender timers must not unregister
        if action_attempted:
            # The planner always exposes global context before eviction/load.
            # Do not execute a second callback in the same timer tick.
            message = f"{type(error).__name__}: {error}"
            _write_telemetry(scene, _telemetry_payload(_runtime_state, error=message))
            interval = (
                _runtime_state.config.timer_interval_s
                if _runtime_state is not None
                else DEFAULT_TIMER_INTERVAL_S
            )
            # A callback was attempted but its planner handshake did not
            # complete.  Re-bootstrap on the next tick rather than trusting a
            # possibly divergent in-memory state.
            _runtime_state = None
            return interval
        return _fail_safe_before_action(bpy_module, scene, error)


def _load_post(_unused: Any) -> None:
    """Forget file-specific planner state after Blender loads another file."""

    global _runtime_state
    _runtime_state = None
    if _runtime_bpy is None:
        return
    timers = _runtime_bpy.app.timers
    scene = _runtime_bpy.context.scene
    if timers.is_registered(_timer_tick):
        timers.unregister(_timer_tick)
    if _scene_has_fireviewer_contract(scene):
        timers.register(
            _timer_tick,
            first_interval=DEFAULT_TIMER_INTERVAL_S,
            persistent=True,
        )


def register(bpy_module: Any | None = None) -> None:
    """Register one persistent load handler and one persistent main-thread timer."""

    global _persistent_load_handler, _runtime_bpy, _runtime_state
    bpy_runtime = _resolve_bpy(bpy_module)
    same_runtime = _runtime_bpy is bpy_runtime
    if _runtime_bpy is not None and _runtime_bpy is not bpy_runtime:
        unregister(_runtime_bpy)
    _runtime_bpy = bpy_runtime
    if not same_runtime:
        _runtime_state = None
    if _persistent_load_handler is None:
        _persistent_load_handler = bpy_runtime.app.handlers.persistent(_load_post)
    handlers = bpy_runtime.app.handlers.load_post
    if _persistent_load_handler not in handlers:
        handlers.append(_persistent_load_handler)
    timers = bpy_runtime.app.timers
    if not _scene_has_fireviewer_contract(bpy_runtime.context.scene):
        _runtime_state = None
        if timers.is_registered(_timer_tick):
            timers.unregister(_timer_tick)
    elif not timers.is_registered(_timer_tick):
        timers.register(
            _timer_tick,
            first_interval=DEFAULT_TIMER_INTERVAL_S,
            persistent=True,
        )


def unregister(bpy_module: Any | None = None) -> None:
    """Remove the adapter registrations; repeated calls are harmless."""

    global _persistent_load_handler, _runtime_bpy, _runtime_state
    if bpy_module is None and _runtime_bpy is None:
        _persistent_load_handler = None
        _runtime_state = None
        return
    bpy_runtime = _resolve_bpy(bpy_module or _runtime_bpy)
    if bpy_runtime.app.timers.is_registered(_timer_tick):
        bpy_runtime.app.timers.unregister(_timer_tick)
    handlers = bpy_runtime.app.handlers.load_post
    if _persistent_load_handler in handlers:
        handlers.remove(_persistent_load_handler)
    _persistent_load_handler = None
    _runtime_state = None
    _runtime_bpy = None


__all__ = [
    "CONFIG_PROPERTY",
    "CONFIG_SCHEMA",
    "FOCUS_OBJECT_NAME",
    "TELEMETRY_PROPERTY",
    "TELEMETRY_SCHEMA",
    "RuntimeConfig",
    "register",
    "unregister",
]
