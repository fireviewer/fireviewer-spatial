"""Pure-Python planner for atomic Blender detail-tile streaming.

The module deliberately does not import :mod:`bpy`.  It turns camera focus
observations into explicit actions that a later Blender adapter can execute
and acknowledge.  Detail is never published while only part of a target tile
set is resident: every transition first exposes the global fallback, evicts
obsolete residents, loads at most one tile per planner tick, then publishes
the complete target in one action.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from tiled_scene import ready_tiles, tile_distance_to_point_m


HARD_MAXIMUM_RESIDENT_TILE_COUNT = 16
DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT = 3
DEFAULT_LOAD_RETRY_BACKOFF_TICKS = 2

ACTIVATE_GLOBAL_FALLBACK = "activate_global_fallback"
EVICT_TILES = "evict_tiles"
LOAD_TILE = "load_tile"
PUBLISH_DETAIL = "publish_detail"
NOOP = "noop"

_PHASE_IDLE = "idle"
_PHASE_GLOBAL_FALLBACK = "global_fallback"
_PHASE_EVICTION = "eviction"
_PHASE_LOADING = "loading"
_PHASE_PUBLICATION = "publication"


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _focus(value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("focus_l93_m must contain Lambert-93 x and y")
    result: list[float] = []
    for index, component in enumerate(value):
        if isinstance(component, bool):
            raise ValueError(f"focus_l93_m[{index}] must be a finite number")
        try:
            number = float(component)
        except (TypeError, ValueError) as error:
            raise ValueError(f"focus_l93_m[{index}] must be a finite number") from error
        if not math.isfinite(number):
            raise ValueError(f"focus_l93_m[{index}] must be a finite number")
        result.append(number)
    return result[0], result[1]


def _radius(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("radius_m must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("radius_m must be a finite non-negative number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("radius_m must be a finite non-negative number")
    return result


def _maximum_tile_count(value: int) -> int:
    result = _positive_integer(value, "maximum_tile_count")
    if result > HARD_MAXIMUM_RESIDENT_TILE_COUNT:
        raise ValueError(
            "maximum_tile_count cannot exceed the hard resident-tile limit "
            f"of {HARD_MAXIMUM_RESIDENT_TILE_COUNT}"
        )
    return result


@dataclass(frozen=True)
class DesiredTileSelection:
    """Deterministic, budgeted result of one focus/radius query."""

    tile_ids: tuple[str, ...]
    matching_tile_count: int
    truncated_tile_count: int


def select_desired_tiles(
    manifest: Mapping[str, Any],
    focus_l93_m: Sequence[float],
    radius_m: float,
    *,
    maximum_tile_count: int = HARD_MAXIMUM_RESIDENT_TILE_COUNT,
) -> DesiredTileSelection:
    """Select ready tiles intersecting a metric focus radius.

    Candidates are ordered by distance to their core bounds, then by stable
    tile identifier. If the complete matching set exceeds the hard cap, the
    returned tile set is empty: publishing an arbitrary nearest subset would
    expose a visibly partial near LOD.
    """

    focus = _focus(focus_l93_m)
    radius = _radius(radius_m)
    maximum = _maximum_tile_count(maximum_tile_count)
    candidates: list[tuple[float, str]] = []
    for tile in ready_tiles(manifest):
        distance = tile_distance_to_point_m(tile["bounds_l93_m"], focus)
        if distance <= radius:
            candidates.append((distance, tile["id"]))
    candidates.sort(key=lambda item: (item[0], item[1]))
    truncated = max(0, len(candidates) - maximum)
    # A partial near set is visually worse than the complete lightweight
    # fallback: vegetation and tile-local vectors would stop on an arbitrary
    # budget boundary.  Keep the ordered candidates for diagnostics through
    # the counts below, but fail closed instead of returning a publishable
    # subset.
    selected = () if truncated else tuple(identifier for _, identifier in candidates)
    return DesiredTileSelection(
        tile_ids=selected,
        matching_tile_count=len(candidates),
        truncated_tile_count=truncated,
    )


def desired_tile_ids(
    manifest: Mapping[str, Any],
    focus_l93_m: Sequence[float],
    radius_m: float,
    *,
    maximum_tile_count: int = HARD_MAXIMUM_RESIDENT_TILE_COUNT,
) -> tuple[str, ...]:
    """Return only the ordered identifiers from :func:`select_desired_tiles`."""

    return select_desired_tiles(
        manifest,
        focus_l93_m,
        radius_m,
        maximum_tile_count=maximum_tile_count,
    ).tile_ids


@dataclass(frozen=True)
class StreamingAction:
    """One executor action emitted for a single planner tick."""

    sequence: int
    generation: int
    kind: str
    tile_ids: tuple[str, ...]
    reason: str

    @property
    def requires_commit(self) -> bool:
        return self.kind != NOOP


@dataclass(frozen=True)
class StreamingState:
    """Externally observable state of the atomic transition."""

    phase: str
    generation: int
    target_tile_ids: tuple[str, ...]
    candidate_tile_ids: tuple[str, ...]
    candidate_observation_count: int
    debounce_ticks: int
    resident_tile_ids: tuple[str, ...]
    published_tile_ids: tuple[str, ...]
    global_fallback_visible: bool
    load_retry_backoff_remaining_ticks: int
    quarantined_tile_ids: tuple[str, ...]
    pending_action: StreamingAction | None


@dataclass(frozen=True)
class StreamingTelemetry:
    """Counters and gauges suitable for Blender or Unity diagnostics."""

    tick_count: int
    transition_count: int
    fallback_activation_count: int
    eviction_action_count: int
    evicted_tile_count: int
    load_attempt_count: int
    load_success_count: int
    load_failure_count: int
    publication_count: int
    action_failure_count: int
    matching_tile_count: int
    truncated_tile_count: int
    resident_tile_count: int
    published_tile_count: int
    load_retry_backoff_remaining_ticks: int
    quarantined_tile_count: int
    phase: str
    generation: int
    last_error: str | None


class TileStreamingPlanner:
    """Plan and acknowledge atomic tile-set transitions without using Blender.

    ``tick`` observes the current focus and returns at most one executable
    action.  Every non-noop action must be passed to ``commit`` before the
    next tick.  This handshake keeps failures observable and prevents the
    planner from assuming that a Blender mutation succeeded.
    """

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        maximum_resident_tile_count: int = HARD_MAXIMUM_RESIDENT_TILE_COUNT,
        debounce_ticks: int = 2,
        maximum_load_failure_count: int = DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT,
        load_retry_backoff_ticks: int = DEFAULT_LOAD_RETRY_BACKOFF_TICKS,
        resident_tile_ids: Iterable[str] = (),
        published_tile_ids: Iterable[str] = (),
        global_fallback_visible: bool = True,
    ) -> None:
        self._manifest = manifest
        self._maximum = _maximum_tile_count(maximum_resident_tile_count)
        self._debounce_ticks = _positive_integer(debounce_ticks, "debounce_ticks")
        self._maximum_load_failure_count = _positive_integer(
            maximum_load_failure_count, "maximum_load_failure_count"
        )
        self._load_retry_backoff_ticks = _positive_integer(
            load_retry_backoff_ticks, "load_retry_backoff_ticks"
        )
        ready_identifiers = {tile["id"] for tile in ready_tiles(manifest)}
        resident = set(resident_tile_ids)
        published = set(published_tile_ids)
        unknown = sorted(resident - ready_identifiers)
        if unknown:
            raise ValueError(
                f"Unknown or non-ready resident tile ids: {', '.join(unknown)}"
            )
        if len(resident) > self._maximum:
            raise ValueError(
                f"Initial resident set has {len(resident)} tiles, maximum is "
                f"{self._maximum}"
            )
        if not published <= resident:
            raise ValueError("Published detail tiles must be resident")
        if published and global_fallback_visible:
            raise ValueError(
                "Published detail and the global fallback cannot be visible together"
            )
        if not published and not global_fallback_visible:
            raise ValueError(
                "The global fallback must be visible when no detail set is published"
            )

        self._resident = resident
        self._published = published
        self._global_fallback_visible = global_fallback_visible
        self._target = tuple(sorted(published))
        self._candidate = self._target
        self._candidate_observations = 0
        self._phase = _PHASE_IDLE
        self._generation = 0
        self._sequence = 0
        self._pending: StreamingAction | None = None
        self._load_failure_counts: dict[str, int] = {}
        self._load_retry_backoff_remaining_ticks = 0
        self._quarantined_tile_ids: set[str] = set()

        self._tick_count = 0
        self._transition_count = 0
        self._fallback_activation_count = 0
        self._eviction_action_count = 0
        self._evicted_tile_count = 0
        self._load_attempt_count = 0
        self._load_success_count = 0
        self._load_failure_count = 0
        self._publication_count = 0
        self._action_failure_count = 0
        self._matching_tile_count = 0
        self._truncated_tile_count = 0
        self._last_error: str | None = None

    @property
    def state(self) -> StreamingState:
        return StreamingState(
            phase=self._phase,
            generation=self._generation,
            target_tile_ids=self._target,
            candidate_tile_ids=self._candidate,
            candidate_observation_count=self._candidate_observations,
            debounce_ticks=self._debounce_ticks,
            resident_tile_ids=tuple(sorted(self._resident)),
            published_tile_ids=tuple(sorted(self._published)),
            global_fallback_visible=self._global_fallback_visible,
            load_retry_backoff_remaining_ticks=(
                self._load_retry_backoff_remaining_ticks
            ),
            quarantined_tile_ids=tuple(sorted(self._quarantined_tile_ids)),
            pending_action=self._pending,
        )

    @property
    def telemetry(self) -> StreamingTelemetry:
        return StreamingTelemetry(
            tick_count=self._tick_count,
            transition_count=self._transition_count,
            fallback_activation_count=self._fallback_activation_count,
            eviction_action_count=self._eviction_action_count,
            evicted_tile_count=self._evicted_tile_count,
            load_attempt_count=self._load_attempt_count,
            load_success_count=self._load_success_count,
            load_failure_count=self._load_failure_count,
            publication_count=self._publication_count,
            action_failure_count=self._action_failure_count,
            matching_tile_count=self._matching_tile_count,
            truncated_tile_count=self._truncated_tile_count,
            resident_tile_count=len(self._resident),
            published_tile_count=len(self._published),
            load_retry_backoff_remaining_ticks=(
                self._load_retry_backoff_remaining_ticks
            ),
            quarantined_tile_count=len(self._quarantined_tile_ids),
            phase=self._phase,
            generation=self._generation,
            last_error=self._last_error,
        )

    def _action(
        self, kind: str, tile_ids: Iterable[str], reason: str
    ) -> StreamingAction:
        self._sequence += 1
        action = StreamingAction(
            sequence=self._sequence,
            generation=self._generation,
            kind=kind,
            tile_ids=tuple(tile_ids),
            reason=reason,
        )
        if action.requires_commit:
            self._pending = action
        return action

    def _noop(self, reason: str) -> StreamingAction:
        return self._action(NOOP, (), reason)

    def _observe(self, selection: DesiredTileSelection) -> None:
        desired = selection.tile_ids
        self._matching_tile_count = selection.matching_tile_count
        self._truncated_tile_count = selection.truncated_tile_count
        # Distance affects load priority, but publication identity is the tile
        # set.  Moving inside the same covered set must not restart a complete
        # fallback transition merely because two distances swapped order.
        if frozenset(desired) == frozenset(self._candidate):
            self._candidate_observations += 1
        else:
            self._candidate = desired
            self._candidate_observations = 1
        if self._candidate_observations >= self._debounce_ticks and frozenset(
            self._candidate
        ) != frozenset(self._target):
            self._target = self._candidate
            self._generation += 1
            self._transition_count += 1
            # Quarantine is bounded to one stable target generation. Moving
            # away and returning later permits a repaired source to be tried
            # again without allowing an endless retry loop for the current
            # target.
            self._load_failure_counts.clear()
            self._load_retry_backoff_remaining_ticks = 0
            self._quarantined_tile_ids.clear()
            self._phase = _PHASE_GLOBAL_FALLBACK

    def tick(self, focus_l93_m: Sequence[float], radius_m: float) -> StreamingAction:
        """Observe the camera and emit at most one ordered transition action."""

        if self._pending is not None:
            raise RuntimeError(
                "The pending streaming action must be committed before the next tick"
            )
        self._tick_count += 1
        selection = select_desired_tiles(
            self._manifest,
            focus_l93_m,
            radius_m,
            maximum_tile_count=self._maximum,
        )
        self._observe(selection)

        if self._phase == _PHASE_GLOBAL_FALLBACK:
            return self._action(
                ACTIVATE_GLOBAL_FALLBACK,
                (),
                "start_atomic_target_transition",
            )

        if self._phase == _PHASE_EVICTION:
            obsolete = tuple(sorted(self._resident - set(self._target)))
            if obsolete:
                self._eviction_action_count += 1
                return self._action(
                    EVICT_TILES,
                    obsolete,
                    "remove_non_target_residents",
                )
            self._phase = _PHASE_LOADING

        if self._phase == _PHASE_LOADING:
            missing = tuple(
                identifier
                for identifier in self._target
                if identifier not in self._resident
            )
            if missing:
                quarantined = tuple(
                    identifier
                    for identifier in missing
                    if identifier in self._quarantined_tile_ids
                )
                if quarantined:
                    return self._noop("target_contains_quarantined_tile")
                if self._load_retry_backoff_remaining_ticks > 0:
                    self._load_retry_backoff_remaining_ticks -= 1
                    return self._noop("load_retry_backoff")
                self._load_attempt_count += 1
                return self._action(LOAD_TILE, missing[:1], "load_target_one_per_tick")
            if not self._target:
                self._phase = _PHASE_IDLE
                return self._noop("empty_target_complete_on_global_fallback")
            self._phase = _PHASE_PUBLICATION

        if self._phase == _PHASE_PUBLICATION:
            if set(self._target) != self._resident:
                raise RuntimeError(
                    "Cannot publish until the complete target is the resident set"
                )
            return self._action(
                PUBLISH_DETAIL,
                self._target,
                "publish_complete_target_atomically",
            )

        if self._candidate_observations < self._debounce_ticks:
            return self._noop("debounce_pending")
        return self._noop("stable_target")

    def commit(
        self,
        action: StreamingAction,
        *,
        succeeded: bool,
        error: str | None = None,
    ) -> None:
        """Acknowledge the exact pending executor action.

        Failed actions leave their phase unchanged so the next tick retries
        safely.  In particular, a failed load can never expose a partial set.
        """

        if self._pending is None:
            raise RuntimeError("There is no pending streaming action to commit")
        if action != self._pending:
            raise ValueError("Committed action does not match the pending action")
        self._pending = None
        if not succeeded:
            self._action_failure_count += 1
            if action.kind == LOAD_TILE:
                self._load_failure_count += 1
                tile_id = action.tile_ids[0]
                failure_count = self._load_failure_counts.get(tile_id, 0) + 1
                self._load_failure_counts[tile_id] = failure_count
                if failure_count >= self._maximum_load_failure_count:
                    self._quarantined_tile_ids.add(tile_id)
                    self._load_retry_backoff_remaining_ticks = 0
                else:
                    self._load_retry_backoff_remaining_ticks = (
                        self._load_retry_backoff_ticks
                    )
            self._last_error = error or f"{action.kind} failed"
            return

        self._last_error = None
        if action.kind == ACTIVATE_GLOBAL_FALLBACK:
            self._global_fallback_visible = True
            self._published.clear()
            self._fallback_activation_count += 1
            self._phase = _PHASE_EVICTION
        elif action.kind == EVICT_TILES:
            self._resident.difference_update(action.tile_ids)
            self._evicted_tile_count += len(action.tile_ids)
            self._phase = _PHASE_LOADING
        elif action.kind == LOAD_TILE:
            if len(action.tile_ids) != 1:
                raise RuntimeError("A load action must contain exactly one tile")
            self._resident.add(action.tile_ids[0])
            self._load_failure_counts.pop(action.tile_ids[0], None)
            self._quarantined_tile_ids.discard(action.tile_ids[0])
            self._load_retry_backoff_remaining_ticks = 0
            if len(self._resident) > self._maximum:
                raise RuntimeError("Resident detail tile budget exceeded")
            self._load_success_count += 1
            self._phase = _PHASE_LOADING
        elif action.kind == PUBLISH_DETAIL:
            if tuple(action.tile_ids) != self._target:
                raise RuntimeError("Publication does not match the stable target")
            if set(action.tile_ids) != self._resident:
                raise RuntimeError("Publication requires the complete resident target")
            self._published = set(action.tile_ids)
            self._global_fallback_visible = False
            self._publication_count += 1
            self._phase = _PHASE_IDLE
        else:
            raise RuntimeError(f"Unsupported streaming action: {action.kind}")


__all__ = [
    "ACTIVATE_GLOBAL_FALLBACK",
    "DEFAULT_LOAD_RETRY_BACKOFF_TICKS",
    "DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT",
    "DesiredTileSelection",
    "EVICT_TILES",
    "HARD_MAXIMUM_RESIDENT_TILE_COUNT",
    "LOAD_TILE",
    "NOOP",
    "PUBLISH_DETAIL",
    "StreamingAction",
    "StreamingState",
    "StreamingTelemetry",
    "TileStreamingPlanner",
    "desired_tile_ids",
    "select_desired_tiles",
]
