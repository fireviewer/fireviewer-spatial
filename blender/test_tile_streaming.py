from __future__ import annotations

from typing import Any

import pytest

from tile_streaming import (
    ACTIVATE_GLOBAL_FALLBACK,
    EVICT_TILES,
    LOAD_TILE,
    NOOP,
    PUBLISH_DETAIL,
    StreamingAction,
    TileStreamingPlanner,
    desired_tile_ids,
    select_desired_tiles,
)


def _tile(
    identifier: str,
    west: float,
    south: float = 6_400_000.0,
    *,
    state: str = "ready",
) -> dict[str, Any]:
    return {
        "id": identifier,
        "bounds_l93_m": [west, south, west + 500.0, south + 500.0],
        "status": {"state": state},
    }


def _manifest(*tiles: dict[str, Any]) -> dict[str, Any]:
    return {"tiles": list(tiles)}


def _commit_success(planner: TileStreamingPlanner, action: StreamingAction) -> None:
    assert action.requires_commit
    planner.commit(action, succeeded=True)


def test_desired_ids_fail_closed_when_the_hard_cap_would_truncate() -> None:
    tiles = [
        _tile(f"tile_{index:02d}", 880_000.0 + index * 500.0) for index in range(18)
    ]
    tiles.reverse()
    tiles.append(_tile("not_ready", 880_000.0, state="planned"))
    manifest = _manifest(*tiles)

    selection = select_desired_tiles(
        manifest,
        (880_250.0, 6_400_250.0),
        20_000.0,
    )

    assert selection.tile_ids == ()
    assert selection.matching_tile_count == 18
    assert selection.truncated_tile_count == 2
    assert (
        desired_tile_ids(
            manifest,
            (880_250.0, 6_400_250.0),
            20_000.0,
        )
        == ()
    )
    assert desired_tile_ids(
        manifest,
        (880_250.0, 6_400_250.0),
        0.0,
    ) == ("tile_00",)
    with pytest.raises(ValueError, match="hard resident-tile limit of 16"):
        desired_tile_ids(
            manifest,
            (880_250.0, 6_400_250.0),
            20_000.0,
            maximum_tile_count=17,
        )


def test_atomic_transition_fallback_evict_load_one_per_tick_then_publish() -> None:
    manifest = _manifest(
        _tile("old", 879_000.0),
        _tile("new_a", 881_000.0),
        _tile("new_b", 881_500.0),
    )
    planner = TileStreamingPlanner(
        manifest,
        debounce_ticks=2,
        resident_tile_ids=("old",),
        published_tile_ids=("old",),
        global_fallback_visible=False,
    )
    focus = (881_500.0, 6_400_250.0)

    assert planner.tick(focus, 500.0).kind == NOOP
    fallback = planner.tick(focus, 500.0)
    assert fallback.kind == ACTIVATE_GLOBAL_FALLBACK
    assert planner.state.published_tile_ids == ("old",)
    _commit_success(planner, fallback)
    assert planner.state.global_fallback_visible
    assert planner.state.published_tile_ids == ()

    eviction = planner.tick(focus, 500.0)
    assert eviction.kind == EVICT_TILES
    assert eviction.tile_ids == ("old",)
    _commit_success(planner, eviction)

    first_load = planner.tick(focus, 500.0)
    assert first_load.kind == LOAD_TILE
    assert first_load.tile_ids == ("new_a",)
    assert planner.state.published_tile_ids == ()
    _commit_success(planner, first_load)

    second_load = planner.tick(focus, 500.0)
    assert second_load.kind == LOAD_TILE
    assert second_load.tile_ids == ("new_b",)
    assert planner.state.resident_tile_ids == ("new_a",)
    _commit_success(planner, second_load)

    publication = planner.tick(focus, 500.0)
    assert publication.kind == PUBLISH_DETAIL
    assert publication.tile_ids == ("new_a", "new_b")
    assert planner.state.global_fallback_visible
    assert planner.state.published_tile_ids == ()
    _commit_success(planner, publication)

    assert planner.state.published_tile_ids == ("new_a", "new_b")
    assert not planner.state.global_fallback_visible
    assert planner.telemetry.fallback_activation_count == 1
    assert planner.telemetry.evicted_tile_count == 1
    assert planner.telemetry.load_attempt_count == 2
    assert planner.telemetry.load_success_count == 2
    assert planner.telemetry.publication_count == 1


def test_planner_never_publishes_a_budget_truncated_target() -> None:
    manifest = _manifest(
        *(_tile(f"tile_{index:02d}", 880_000.0 + index * 500.0) for index in range(17))
    )
    planner = TileStreamingPlanner(manifest, debounce_ticks=1)

    for _ in range(5):
        action = planner.tick((880_250.0, 6_400_250.0), 20_000.0)
        assert action.kind == NOOP

    assert planner.state.target_tile_ids == ()
    assert planner.state.resident_tile_ids == ()
    assert planner.state.published_tile_ids == ()
    assert planner.state.global_fallback_visible
    assert planner.telemetry.matching_tile_count == 17
    assert planner.telemetry.truncated_tile_count == 1


def test_debounce_ignores_camera_jitter_until_one_target_is_stable() -> None:
    manifest = _manifest(
        _tile("west", 880_000.0),
        _tile("east", 881_000.0),
    )
    planner = TileStreamingPlanner(manifest, debounce_ticks=3)

    assert planner.tick((880_250.0, 6_400_250.0), 0.0).kind == NOOP
    assert planner.tick((881_250.0, 6_400_250.0), 0.0).kind == NOOP
    assert planner.tick((880_250.0, 6_400_250.0), 0.0).kind == NOOP
    assert planner.state.generation == 0
    assert planner.state.target_tile_ids == ()

    assert planner.tick((880_250.0, 6_400_250.0), 0.0).kind == NOOP
    action = planner.tick((880_250.0, 6_400_250.0), 0.0)
    assert action.kind == ACTIVATE_GLOBAL_FALLBACK
    assert planner.state.target_tile_ids == ("west",)
    assert planner.state.generation == 1


def test_distance_order_change_does_not_republish_the_same_tile_set() -> None:
    manifest = _manifest(
        _tile("a_west", 880_000.0),
        _tile("b_east", 880_500.0),
    )
    planner = TileStreamingPlanner(
        manifest,
        debounce_ticks=1,
        resident_tile_ids=("a_west", "b_east"),
        published_tile_ids=("a_west", "b_east"),
        global_fallback_visible=False,
    )

    action = planner.tick((880_999.0, 6_400_250.0), 1_000.0)

    assert desired_tile_ids(manifest, (880_999.0, 6_400_250.0), 1_000.0) == (
        "b_east",
        "a_west",
    )
    assert action.kind == NOOP
    assert planner.state.generation == 0
    assert planner.state.published_tile_ids == ("a_west", "b_east")


def test_failed_load_retries_without_partial_publication_and_is_reported() -> None:
    manifest = _manifest(_tile("detail", 880_000.0))
    planner = TileStreamingPlanner(manifest, debounce_ticks=1)
    focus = (880_250.0, 6_400_250.0)

    _commit_success(planner, planner.tick(focus, 0.0))
    failed_load = planner.tick(focus, 0.0)
    assert failed_load.kind == LOAD_TILE
    planner.commit(failed_load, succeeded=False, error="disk read failed")

    assert planner.state.resident_tile_ids == ()
    assert planner.state.published_tile_ids == ()
    assert planner.state.global_fallback_visible
    assert planner.telemetry.load_failure_count == 1
    assert planner.telemetry.last_error == "disk read failed"

    first_backoff = planner.tick(focus, 0.0)
    second_backoff = planner.tick(focus, 0.0)
    assert first_backoff.kind == NOOP
    assert first_backoff.reason == "load_retry_backoff"
    assert second_backoff.kind == NOOP
    assert second_backoff.reason == "load_retry_backoff"

    retry = planner.tick(focus, 0.0)
    assert retry.kind == LOAD_TILE
    assert retry.tile_ids == ("detail",)
    _commit_success(planner, retry)
    publication = planner.tick(focus, 0.0)
    assert publication.kind == PUBLISH_DETAIL


def test_failed_load_is_quarantined_after_a_bounded_number_of_attempts() -> None:
    manifest = _manifest(_tile("detail", 880_000.0))
    planner = TileStreamingPlanner(
        manifest,
        debounce_ticks=1,
        maximum_load_failure_count=3,
        load_retry_backoff_ticks=1,
    )
    focus = (880_250.0, 6_400_250.0)

    _commit_success(planner, planner.tick(focus, 0.0))
    for attempt in range(3):
        failed_load = planner.tick(focus, 0.0)
        assert failed_load.kind == LOAD_TILE
        planner.commit(failed_load, succeeded=False, error="corrupt tile")
        if attempt < 2:
            backoff = planner.tick(focus, 0.0)
            assert backoff.kind == NOOP
            assert backoff.reason == "load_retry_backoff"

    for _ in range(5):
        quarantined = planner.tick(focus, 0.0)
        assert quarantined.kind == NOOP
        assert quarantined.reason == "target_contains_quarantined_tile"

    assert planner.state.quarantined_tile_ids == ("detail",)
    assert planner.state.resident_tile_ids == ()
    assert planner.state.published_tile_ids == ()
    assert planner.state.global_fallback_visible
    assert planner.telemetry.load_attempt_count == 3
    assert planner.telemetry.load_failure_count == 3
    assert planner.telemetry.quarantined_tile_count == 1

    # A genuinely different stable target starts a new generation and clears
    # the bounded quarantine, so repaired resources can be tried later.
    far_away = (900_000.0, 6_500_000.0)
    transition = planner.tick(far_away, 0.0)
    assert transition.kind == ACTIVATE_GLOBAL_FALLBACK
    assert planner.state.quarantined_tile_ids == ()


def test_empty_target_returns_to_global_and_evicts_detail() -> None:
    manifest = _manifest(_tile("detail", 880_000.0))
    planner = TileStreamingPlanner(
        manifest,
        debounce_ticks=1,
        resident_tile_ids=("detail",),
        published_tile_ids=("detail",),
        global_fallback_visible=False,
    )
    far_away = (900_000.0, 6_500_000.0)

    fallback = planner.tick(far_away, 0.0)
    assert fallback.kind == ACTIVATE_GLOBAL_FALLBACK
    _commit_success(planner, fallback)
    eviction = planner.tick(far_away, 0.0)
    assert eviction.kind == EVICT_TILES
    _commit_success(planner, eviction)
    complete = planner.tick(far_away, 0.0)

    assert complete.kind == NOOP
    assert complete.reason == "empty_target_complete_on_global_fallback"
    assert planner.state.resident_tile_ids == ()
    assert planner.state.published_tile_ids == ()
    assert planner.state.global_fallback_visible


def test_pending_action_requires_exact_acknowledgement() -> None:
    manifest = _manifest(_tile("detail", 880_000.0))
    planner = TileStreamingPlanner(manifest, debounce_ticks=1)
    action = planner.tick((880_250.0, 6_400_250.0), 0.0)

    with pytest.raises(RuntimeError, match="must be committed"):
        planner.tick((880_250.0, 6_400_250.0), 0.0)
    wrong = StreamingAction(
        sequence=action.sequence + 1,
        generation=action.generation,
        kind=action.kind,
        tile_ids=action.tile_ids,
        reason=action.reason,
    )
    with pytest.raises(ValueError, match="does not match"):
        planner.commit(wrong, succeeded=True)

    planner.commit(action, succeeded=True)


def test_initial_state_rejects_invalid_visibility_and_residency() -> None:
    manifest = _manifest(_tile("detail", 880_000.0))
    with pytest.raises(ValueError, match="must be resident"):
        TileStreamingPlanner(manifest, published_tile_ids=("detail",))
    with pytest.raises(ValueError, match="cannot be visible together"):
        TileStreamingPlanner(
            manifest,
            resident_tile_ids=("detail",),
            published_tile_ids=("detail",),
        )
    with pytest.raises(ValueError, match="Unknown or non-ready"):
        TileStreamingPlanner(manifest, resident_tile_ids=("unknown",))
