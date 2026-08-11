from __future__ import annotations

import json
import hashlib
import math

import pytest

from frustum_streaming import (
    Aabb3D,
    CAMERA_RESIDENCY_PLAN_SCHEMA,
    EVICT_PAYLOADS,
    Frustum,
    FrustumStreamingAction,
    FrustumStreamingPlanner,
    LOAD_PAYLOAD,
    LOD0,
    LOD1,
    LOD2,
    NOOP,
    PUBLISH_CAMERA,
    PayloadRef,
    ResourceCost,
    StreamingBudget,
    StreamingBudgetExceeded,
    STREAMING_STATE_SCHEMA,
    TerrainTile,
    TerrainTileCatalog,
    CameraEnvelope,
    CameraView,
    load_streaming_contract,
    motion_requires_publication_hold,
    compute_stitch_masks,
    plan_camera_sequence,
    plan_residency,
    predict_interactive_envelope,
    select_tiles_for_envelope,
    stitch_mask_for_neighbors,
    validate_camera_residency_plan,
)


def _costs(
    *, lod0: int = 1_000, lod1: int = 250, lod2: int = 50
) -> tuple[ResourceCost, ResourceCost, ResourceCost]:
    return (
        ResourceCost(cpu_bytes=lod0, gpu_bytes=lod0, triangles=lod0 // 2),
        ResourceCost(cpu_bytes=lod1, gpu_bytes=lod1, triangles=lod1 // 2),
        ResourceCost(cpu_bytes=lod2, gpu_bytes=lod2, triangles=lod2 // 2),
    )


def _tile(
    grid_x: int,
    grid_y: int,
    *,
    identifier: str | None = None,
    minimum_z: float = 0.0,
    maximum_z: float = 10.0,
    costs: tuple[ResourceCost, ResourceCost, ResourceCost] | None = None,
) -> TerrainTile:
    west = grid_x * 500.0
    south = grid_y * 500.0
    tile_id = identifier or f"tile_{grid_x:+03d}_{grid_y:+03d}"
    build_id = hashlib.sha256(b"fireviewer-test-build").hexdigest()
    return TerrainTile(
        tile_id=tile_id,
        grid_x=grid_x,
        grid_y=grid_y,
        bounds=Aabb3D(
            minimum=(west, south, minimum_z),
            maximum=(west + 500.0, south + 500.0, maximum_z),
        ),
        costs=costs or _costs(),
        build_id=build_id,
        payload_sha256=tuple(
            hashlib.sha256(f"{build_id}:{tile_id}:lod{lod}".encode()).hexdigest()
            for lod in range(3)
        ),
        stitch_masks=tuple(range(16)),
        stitch_triangle_counts=tuple(
            tuple(cost.triangles for _mask in range(16)) for cost in (costs or _costs())
        ),
    )


def _grid(radius: int) -> TerrainTileCatalog:
    return TerrainTileCatalog(
        _tile(grid_x, grid_y)
        for grid_x in range(-radius, radius + 1)
        for grid_y in range(-radius, radius + 1)
    )


def _view(
    view_id: str,
    *,
    position: tuple[float, float, float],
    forward: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    vertical_fov_deg: float = 10.0,
    aspect_ratio: float = 1.0,
    near_clip_m: float = 1.0,
    far_clip_m: float = 2_000.0,
) -> CameraView:
    return CameraView(
        view_id=view_id,
        position_l93_ngf_m=position,
        forward=forward,
        up=up,
        vertical_fov_deg=vertical_fov_deg,
        aspect_ratio=aspect_ratio,
        near_clip_m=near_clip_m,
        far_clip_m=far_clip_m,
    )


def _envelope(
    camera_id: str, *views: CameraView, mode: str = "planned"
) -> CameraEnvelope:
    return CameraEnvelope(camera_id=camera_id, views=tuple(views), mode=mode)


def _large_budget() -> StreamingBudget:
    return StreamingBudget(
        cpu_bytes=1_000_000_000,
        gpu_bytes=1_000_000_000,
        maximum_triangles=1_000_000_000,
    )


def _commit_success(
    planner: FrustumStreamingPlanner, action: FrustumStreamingAction
) -> None:
    assert action.requires_commit
    planner.commit(
        action,
        succeeded=True,
        observed_build_id=action.expected_build_id,
        observed_sha256=action.expected_sha256,
    )


def test_exact_frustum_aabb_handles_front_oblique_tangent_and_outside() -> None:
    frontal = _view(
        "front",
        position=(0.0, 0.0, 0.0),
        forward=(1.0, 0.0, 0.0),
        vertical_fov_deg=90.0,
        far_clip_m=1_000.0,
    )
    frustum = Frustum.from_camera_view(frontal)

    assert frustum.intersects_aabb(_tile(1, -1, minimum_z=-500.0).bounds)
    assert not frustum.intersects_aabb(_tile(-1, -1, minimum_z=-500.0).bounds)
    # y=x at the far plane: this tile touches the frustum at x=1000,y=1000.
    assert frustum.intersects_aabb(_tile(1, 2, minimum_z=-500.0).bounds)
    assert not frustum.intersects_aabb(_tile(1, 3, minimum_z=-500.0).bounds)

    oblique = Frustum.from_camera_view(
        _view(
            "oblique",
            position=(0.0, 0.0, 0.0),
            forward=(1.0, 1.0, 0.0),
            vertical_fov_deg=20.0,
            far_clip_m=2_000.0,
        )
    )
    assert oblique.intersects_aabb(_tile(1, 1, minimum_z=-500.0).bounds)
    assert not oblique.intersects_aabb(_tile(1, -2, minimum_z=-500.0).bounds)


def test_residency_is_frustum_guard_two_lod1_rings_then_lod2_without_cap() -> None:
    catalog = _grid(4)
    active = _envelope(
        "CAM_CENTER",
        _view(
            "center",
            position=(250.0, 250.0, 100.0),
            forward=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0),
            far_clip_m=200.0,
        ),
    )

    plan = plan_residency(catalog, active, _large_budget())

    assert plan.sets.visible_lod0 == ("tile_+00_+00",)
    assert len(plan.sets.guard_lod0) == 8
    assert len(plan.sets.resident_lod1) == 40
    assert len(plan.sets.resident_lod2) == 32
    assert (
        len(plan.sets.all_lod0)
        + len(plan.sets.resident_lod1)
        + len(plan.sets.resident_lod2)
        == 81
    )

    wide = _envelope(
        "CAM_WIDE",
        _view(
            "wide",
            position=(250.0, 250.0, 5_000.0),
            forward=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0),
            vertical_fov_deg=90.0,
            far_clip_m=6_000.0,
        ),
    )
    wide_plan = plan_residency(catalog, wide, _large_budget())
    assert len(wide_plan.sets.visible_lod0) > 16
    assert wide_plan.budget_report.within_budget


def test_camera_envelope_uses_widest_zoom_and_preloads_next_camera() -> None:
    catalog = _grid(5)
    narrow = _view(
        "zoom-50mm",
        position=(250.0, 250.0, 1_000.0),
        forward=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vertical_fov_deg=20.0,
        far_clip_m=1_500.0,
    )
    widest = _view(
        "zoom-26mm",
        position=(250.0, 250.0, 1_000.0),
        forward=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        vertical_fov_deg=60.0,
        far_clip_m=1_500.0,
    )
    camera_a = _envelope("CAM_A", narrow, widest)
    camera_b = _envelope(
        "CAM_B",
        _view(
            "next",
            position=(2_250.0, 2_250.0, 500.0),
            forward=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0),
            vertical_fov_deg=15.0,
            far_clip_m=700.0,
        ),
    )

    visible_union = select_tiles_for_envelope(catalog, camera_a)
    visible_widest = select_tiles_for_envelope(
        catalog, _envelope("WIDEST_ONLY", widest)
    )
    assert visible_union == visible_widest

    sequence = plan_camera_sequence(catalog, (camera_a, camera_b), _large_budget())
    first = sequence.entries[0]
    assert first.staging_camera_id == "CAM_B"
    assert first.sets.staging_lod0 == select_tiles_for_envelope(catalog, camera_b)
    payload = sequence.to_dict()
    assert payload["schema"] == CAMERA_RESIDENCY_PLAN_SCHEMA
    assert json.loads(json.dumps(payload)) == payload
    assert sequence.entries[1].staging_camera_id is None


def test_interactive_prediction_samples_two_seconds_and_covers_180_rotation() -> None:
    catalog = TerrainTileCatalog(
        (
            _tile(1, -1, identifier="front", minimum_z=-250.0),
            _tile(-2, -1, identifier="back", minimum_z=-250.0),
            _tile(-1, 1, identifier="side", minimum_z=-250.0),
        )
    )
    current = _view(
        "interactive",
        position=(0.0, 0.0, 0.0),
        forward=(1.0, 0.0, 0.0),
        vertical_fov_deg=20.0,
        far_clip_m=1_000.0,
    )

    predicted = predict_interactive_envelope(
        current,
        camera_id="CAM_PREDICTED",
        angular_velocity_axis=(0.0, 0.0, 1.0),
        angular_velocity_deg_s=90.0,
    )

    assert len(predicted.views) == 9
    assert predicted.views[0].forward == pytest.approx((1.0, 0.0, 0.0))
    assert predicted.views[-1].forward == pytest.approx((-1.0, 0.0, 0.0))
    selected = set(select_tiles_for_envelope(catalog, predicted))
    assert {"front", "back"} <= selected

    rolled = _view(
        "rolled",
        position=current.position_l93_ngf_m,
        forward=current.forward,
        up=(0.0, 1.0, 0.0),
        vertical_fov_deg=current.vertical_fov_deg,
        aspect_ratio=16.0 / 9.0,
        far_clip_m=current.far_clip_m,
    )
    assert motion_requires_publication_hold(
        current,
        rolled,
        elapsed_seconds=0.1,
        maximum_linear_speed_mps=200.0,
        maximum_angular_speed_deg_s=90.0,
    )


def test_teleport_uses_double_buffer_and_three_second_hysteresis() -> None:
    catalog = TerrainTileCatalog(
        (
            _tile(1, 0, identifier="a"),
            _tile(10, 0, identifier="b"),
        )
    )
    camera_a = _envelope(
        "CAM_A",
        _view(
            "a",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    camera_b = _envelope(
        "CAM_B",
        _view(
            "b",
            position=(6_000.0, 250.0, 5.0),
            forward=(-1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    assert motion_requires_publication_hold(
        camera_a.views[0],
        camera_b.views[0],
        elapsed_seconds=0.1,
        maximum_linear_speed_mps=200.0,
        maximum_angular_speed_deg_s=90.0,
    )

    planner = FrustumStreamingPlanner(catalog, _large_budget())
    planner.stage_camera(camera_a, now_s=0.0)
    first_load = planner.tick(now_s=0.0)
    assert first_load.kind == LOAD_PAYLOAD
    _commit_success(planner, first_load)
    first_publish = planner.tick(now_s=0.1)
    assert first_publish.kind == PUBLISH_CAMERA
    _commit_success(planner, first_publish)
    assert planner.state.active_camera_id == "CAM_A"
    assert planner.state.published_visible_lod0_tile_ids == ("a",)

    planner.stage_camera(camera_b, now_s=1.0)
    second_load = planner.tick(now_s=1.0)
    assert second_load.kind == LOAD_PAYLOAD
    assert second_load.payloads == (PayloadRef("b", LOD0),)
    assert planner.state.active_camera_id == "CAM_A"
    _commit_success(planner, second_load)
    second_publish = planner.tick(now_s=1.1)
    assert second_publish.kind == PUBLISH_CAMERA
    assert planner.state.active_camera_id == "CAM_A"
    _commit_success(planner, second_publish)
    assert planner.state.active_camera_id == "CAM_B"
    assert planner.state.published_visible_lod0_tile_ids == ("b",)
    assert PayloadRef("a", LOD0) in planner.state.resident_payloads

    replacement = planner.tick(now_s=2.0)
    assert replacement.kind == LOAD_PAYLOAD
    assert replacement.payloads == (PayloadRef("a", LOD2),)
    _commit_success(planner, replacement)
    retained = planner.tick(now_s=4.0)
    assert retained.kind == NOOP
    assert retained.reason == "hysteresis_retains_previous_lod0"
    eviction = planner.tick(now_s=4.2)
    assert eviction.kind == EVICT_PAYLOADS
    assert eviction.payloads == (PayloadRef("a", LOD0),)
    _commit_success(planner, eviction)


def test_failed_staging_lod0_retries_three_times_then_quarantines() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    camera = _envelope(
        "CAM_FAIL",
        _view(
            "fail",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(
        catalog,
        _large_budget(),
        maximum_load_failure_count=3,
        load_retry_backoff_ticks=0,
    )
    planner.stage_camera(camera, now_s=0.0)

    for attempt in range(3):
        action = planner.tick(now_s=float(attempt))
        assert action.kind == LOAD_PAYLOAD
        planner.commit(action, succeeded=False, error="corrupt payload")

    blocked = planner.tick(now_s=3.0)
    assert blocked.kind == NOOP
    assert blocked.reason == "staging_contains_quarantined_lod0"
    assert planner.state.active_camera_id is None
    assert planner.state.published_visible_lod0_tile_ids == ()
    assert planner.state.quarantined_payloads == (PayloadRef("detail", LOD0),)
    assert planner.telemetry.load_attempt_count == 3
    assert planner.telemetry.load_failure_count == 3
    assert planner.telemetry.publication_count == 0


def test_budget_counts_every_role_once_adds_25_percent_and_fails_closed() -> None:
    catalog = _grid(4)
    active = _envelope(
        "CAM_BUDGET",
        _view(
            "budget",
            position=(250.0, 250.0, 100.0),
            forward=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0),
            far_clip_m=200.0,
        ),
    )
    report = plan_residency(catalog, active, _large_budget()).budget_report

    assert set(report.categories) == {
        "active_lod0",
        "staging_lod0",
        "guard_lod0",
        "lod1",
        "lod2",
    }
    assert report.required_with_reserve.cpu_bytes == math.ceil(
        report.unreserved.cpu_bytes * 1.25
    )
    assert report.required_with_reserve.gpu_bytes == math.ceil(
        report.unreserved.gpu_bytes * 1.25
    )

    insufficient = StreamingBudget(
        cpu_bytes=report.required_with_reserve.cpu_bytes - 1,
        gpu_bytes=report.required_with_reserve.gpu_bytes,
        maximum_triangles=report.unreserved.triangles,
    )
    with pytest.raises(StreamingBudgetExceeded, match="including reserve"):
        plan_residency(catalog, active, insufficient)
    with pytest.raises(ValueError, match="reserve_fraction"):
        StreamingBudget(cpu_bytes=10_000, gpu_bytes=10_000, reserve_fraction=0.249)


def test_budget_includes_unexpired_lod0_from_a_previous_transition() -> None:
    catalog = TerrainTileCatalog(
        (
            _tile(1, 0, identifier="a"),
            _tile(10, 0, identifier="b"),
            _tile(20, 0, identifier="c"),
        )
    )

    def camera(identifier: str, x: float, forward_x: float) -> CameraEnvelope:
        return _envelope(
            f"CAM_{identifier.upper()}",
            _view(
                identifier,
                position=(x, 250.0, 5.0),
                forward=(forward_x, 0.0, 0.0),
                far_clip_m=1_500.0,
            ),
        )

    camera_a = camera("a", 0.0, 1.0)
    camera_b = camera("b", 6_000.0, -1.0)
    camera_c = camera("c", 11_000.0, -1.0)
    budget = StreamingBudget(cpu_bytes=3_000, gpu_bytes=3_000)
    planner = FrustumStreamingPlanner(catalog, budget)

    planner.stage_camera(camera_a, now_s=0.0)
    _commit_success(planner, planner.tick(now_s=0.0))
    _commit_success(planner, planner.tick(now_s=0.1))
    planner.stage_camera(camera_b, now_s=1.0)
    _commit_success(planner, planner.tick(now_s=1.0))
    _commit_success(planner, planner.tick(now_s=1.1))

    with pytest.raises(
        StreamingBudgetExceeded, match="double-buffer transition to CAM_C"
    ):
        planner.stage_camera(camera_c, now_s=2.0)
    assert planner.state.active_camera_id == "CAM_B"
    assert planner.state.staging_camera_id is None


def test_contract_and_camera_envelope_round_trip_are_locked() -> None:
    contract = load_streaming_contract()
    assert contract["selection"]["radial_selection_permitted"] is False
    assert contract["selection"]["fixed_tile_count_cap_permitted"] is False
    assert contract["transition"]["visible_lower_lod_fallback"] == "forbidden"
    assert contract["transition"]["payload_integrity_gate"] == (
        "catalog_build_id_and_sha256_must_match_observed_load"
    )
    assert contract["residency"]["guard_publication_gate"] == (
        "complete_before_camera_publication"
    )
    assert contract["resume"]["lower_lod_on_visible_tile_permitted"] is False
    assert contract["budget"]["included_roles"] == [
        "active_lod0",
        "staging_lod0",
        "guard_lod0",
        "lod1",
        "lod2",
    ]

    camera = _envelope(
        "CAM_ROUND_TRIP",
        _view(
            "main",
            position=(700_000.0, 6_400_000.0, 250.0),
            forward=(1.0, 0.0, -0.1),
            vertical_fov_deg=45.0,
            aspect_ratio=16.0 / 9.0,
            near_clip_m=0.1,
            far_clip_m=5_400.0,
        ),
    )
    restored = CameraEnvelope.from_mapping(camera.to_dict())
    assert restored == camera


def test_pending_action_requires_exact_commit() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    camera = _envelope(
        "CAM",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, _large_budget())
    planner.stage_camera(camera, now_s=0.0)
    action = planner.tick(now_s=0.0)
    with pytest.raises(RuntimeError, match="must be committed"):
        planner.tick(now_s=0.1)
    wrong = FrustumStreamingAction(
        sequence=action.sequence + 1,
        generation=action.generation,
        kind=action.kind,
        payloads=action.payloads,
        camera_id=action.camera_id,
        visible_lod0_tile_ids=action.visible_lod0_tile_ids,
        reason=action.reason,
        issued_at_s=action.issued_at_s,
        expected_build_id=action.expected_build_id,
        expected_sha256=action.expected_sha256,
    )
    with pytest.raises(ValueError, match="does not match"):
        planner.commit(wrong, succeeded=True)
    _commit_success(planner, action)


def test_streaming_state_v2_is_canonical_json_serializable() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    camera = _envelope(
        "CAM_STATE",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, _large_budget())
    planner.stage_camera(camera, now_s=4.0)
    payload = planner.state.to_dict()

    assert payload["schema"] == STREAMING_STATE_SCHEMA
    assert payload["staging_camera_id"] == "CAM_STATE"
    assert payload["active_camera_id"] is None
    assert payload["pending_action"] is None
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_tile_manifest_requires_explicit_3d_bounds_and_all_lod_costs() -> None:
    payload = {
        "tiles": [
            {
                "id": "tile",
                "grid_x": 1400,
                "grid_y": 12800,
                "bounds_l93_ngf_m": [
                    700_000.0,
                    6_400_000.0,
                    10.0,
                    700_500.0,
                    6_400_500.0,
                    310.0,
                ],
                "resource_costs": {
                    "lod0": {
                        "cpu_bytes": 10,
                        "gpu_bytes": 20,
                        "triangles": 30,
                        "sha256": "a" * 64,
                        "stitch_triangle_counts": [30] * 16,
                    },
                    "lod1": {
                        "cpu_bytes": 4,
                        "gpu_bytes": 5,
                        "triangles": 6,
                        "sha256": "b" * 64,
                        "stitch_triangle_counts": [6] * 16,
                    },
                    "lod2": {
                        "cpu_bytes": 1,
                        "gpu_bytes": 2,
                        "triangles": 3,
                        "sha256": "c" * 64,
                        "stitch_triangle_counts": [3] * 16,
                    },
                },
                "stitch_masks": list(range(16)),
                "build_id": "d" * 64,
            }
        ]
    }

    catalog = TerrainTileCatalog.from_manifest(payload)

    assert catalog.tile_ids == ("tile",)
    assert catalog.tile("tile").bounds.maximum == (700_500.0, 6_400_500.0, 310.0)
    assert catalog.tile("tile").cost(LOD0).gpu_bytes == 20

    misaligned = json.loads(json.dumps(payload))
    misaligned["tiles"][0]["bounds_l93_ngf_m"][0] += 1.0
    misaligned["tiles"][0]["bounds_l93_ngf_m"][3] += 1.0
    with pytest.raises(ValueError, match="global Lambert-93 500 m grid"):
        TerrainTileCatalog.from_manifest(misaligned)


def test_guard_lod0_is_loaded_before_first_camera_publication() -> None:
    catalog = _grid(1)
    camera = _envelope(
        "CAM_GUARD",
        _view(
            "main",
            position=(250.0, 250.0, 100.0),
            forward=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0),
            far_clip_m=200.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, _large_budget())
    plan = planner.stage_camera(camera, now_s=0.0)
    assert len(plan.sets.visible_lod0) == 1
    assert len(plan.sets.guard_lod0) == 8

    loaded = []
    for tick in range(20):
        action = planner.tick(now_s=tick / 10.0)
        if action.kind == PUBLISH_CAMERA:
            assert len(loaded) == 9
            assert {
                payload.tile_id
                for payload in planner.state.resident_payloads
                if payload.lod == LOD0
            } == set(plan.sets.all_lod0)
            _commit_success(planner, action)
            break
        assert action.kind == LOAD_PAYLOAD
        loaded.extend(action.payloads)
        _commit_success(planner, action)
    else:
        raise AssertionError("Camera was not published after complete guard loading")


def test_hash_mismatch_quarantines_then_explicit_retry_can_publish() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    camera = _envelope(
        "CAM_HASH",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, _large_budget())
    planner.stage_camera(camera, now_s=0.0)
    corrupt = planner.tick(now_s=0.0)
    planner.commit(
        corrupt,
        succeeded=True,
        observed_build_id=corrupt.expected_build_id,
        observed_sha256="0" * 64,
    )
    assert planner.state.quarantined_payloads == corrupt.payloads
    assert planner.tick(now_s=0.1).reason == "staging_contains_quarantined_lod0"

    generation = planner.state.generation
    assert planner.retry_quarantined() == corrupt.payloads
    assert planner.state.generation == generation + 1
    repaired = planner.tick(now_s=0.2)
    _commit_success(planner, repaired)
    publish = planner.tick(now_s=0.3)
    assert publish.kind == PUBLISH_CAMERA
    _commit_success(planner, publish)


def test_successful_load_commit_requires_observed_build_and_hash() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    camera = _envelope(
        "CAM_OBSERVED",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, _large_budget())
    planner.stage_camera(camera, now_s=0.0)
    action = planner.tick(now_s=0.0)
    planner.commit(action, succeeded=True)

    assert planner.state.quarantined_payloads == action.payloads
    assert "observed_build_id" in (planner.state.last_error or "")
    assert planner.state.published_visible_lod0_tile_ids == ()


def test_restored_state_rejects_lower_visible_and_evicts_obsolete_before_load() -> None:
    catalog = TerrainTileCatalog(
        (_tile(1, 0, identifier="visible"), _tile(10, 0, identifier="obsolete"))
    )
    camera = _envelope(
        "CAM_RESTORE",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    with pytest.raises(ValueError, match="must not retain LOD1 or LOD2"):
        FrustumStreamingPlanner(
            catalog,
            _large_budget(),
            active_camera=camera,
            resident_payloads=(
                PayloadRef("visible", LOD0),
                PayloadRef("visible", 1),
            ),
            published_visible_lod0_tile_ids=("visible",),
        )
    with pytest.raises(StreamingBudgetExceeded, match="restored resident payloads"):
        FrustumStreamingPlanner(
            catalog,
            StreamingBudget(cpu_bytes=1_500, gpu_bytes=1_500),
            active_camera=camera,
            resident_payloads=(
                PayloadRef("visible", LOD0),
                PayloadRef("obsolete", LOD0),
            ),
            published_visible_lod0_tile_ids=("visible",),
        )

    planner = FrustumStreamingPlanner(
        catalog,
        _large_budget(),
        active_camera=camera,
        resident_payloads=(
            PayloadRef("visible", LOD0),
            PayloadRef("obsolete", LOD0),
        ),
        published_visible_lod0_tile_ids=("visible",),
    )
    first = planner.tick(now_s=0.0)
    assert first.kind == EVICT_PAYLOADS
    assert first.payloads == (PayloadRef("obsolete", LOD0),)


def test_stitch_masks_follow_fvtq_edge_bits_and_reject_lod_delta_two() -> None:
    catalog = TerrainTileCatalog(
        (
            _tile(0, 0, identifier="west"),
            _tile(1, 0, identifier="middle"),
            _tile(2, 0, identifier="east"),
        )
    )

    assert (
        stitch_mask_for_neighbors(
            LOD0,
            {"west": None, "east": LOD1, "south": None, "north": None},
        )
        == 2
    )
    assert compute_stitch_masks(
        catalog, {"west": LOD0, "middle": LOD1, "east": LOD2}
    ) == {"east": 0, "middle": 2, "west": 2}
    with pytest.raises(ValueError, match="delta exceeds 1"):
        compute_stitch_masks(catalog, {"west": LOD0, "middle": LOD2, "east": LOD2})


def test_stitch_variant_cost_uses_exact_catalog_triangle_count() -> None:
    counts = list(range(10, 26))
    tile = TerrainTile(
        tile_id="variant",
        grid_x=0,
        grid_y=0,
        bounds=Aabb3D((0.0, 0.0, 0.0), (500.0, 500.0, 10.0)),
        costs=(
            ResourceCost(1_000, 2_000, 10),
            ResourceCost(500, 1_000, 10),
            ResourceCost(250, 500, 10),
        ),
        build_id="d" * 64,
        payload_sha256=("a" * 64, "b" * 64, "c" * 64),
        stitch_masks=tuple(range(16)),
        stitch_triangle_counts=(tuple(counts), tuple(counts), tuple(counts)),
    )

    assert tile.cost(LOD0, 15) == ResourceCost(1_000, 2_180, 25)


def test_public_plan_validator_recomputes_sets_masks_and_budget() -> None:
    catalog = _grid(4)
    camera = _envelope(
        "CAM_VALIDATE",
        _view(
            "main",
            position=(250.0, 250.0, 100.0),
            forward=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0),
            far_clip_m=200.0,
        ),
    )
    serialized = plan_camera_sequence(catalog, (camera,), _large_budget()).to_dict()

    assert validate_camera_residency_plan(serialized, catalog).to_dict() == serialized
    binding = serialized["camera_envelope_bindings"][0]
    assert binding["camera_id"] == camera.camera_id
    assert binding["sha256"] == camera.sha256
    assert any(
        payload["stitch_mask"] != 0 for payload in serialized["entries"][0]["payloads"]
    )

    wrong_envelope_hash = json.loads(json.dumps(serialized))
    wrong_envelope_hash["camera_envelope_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding SHA-256"):
        validate_camera_residency_plan(wrong_envelope_hash, catalog)

    wrong_frustum = json.loads(json.dumps(serialized))
    changed_binding = wrong_frustum["camera_envelope_bindings"][0]
    changed_binding["envelope"]["views"][0]["position_l93_ngf_m"][0] += 1_000.0
    changed_binding["sha256"] = CameraEnvelope.from_mapping(
        changed_binding["envelope"]
    ).sha256
    with pytest.raises(ValueError, match="bound camera-envelope frusta"):
        validate_camera_residency_plan(wrong_frustum, catalog)

    wrong_mask = json.loads(json.dumps(serialized))
    variant = next(
        item
        for item in wrong_mask["entries"][0]["payloads"]
        if item["stitch_mask"] != 0
    )
    variant["stitch_mask"] ^= 1
    with pytest.raises(ValueError, match="stitch masks"):
        validate_camera_residency_plan(wrong_mask, catalog)

    wrong_budget = json.loads(json.dumps(serialized))
    wrong_budget["entries"][0]["budget"]["unreserved"]["triangles"] += 1
    with pytest.raises(ValueError, match="budget"):
        validate_camera_residency_plan(wrong_budget, catalog)

    incomplete = json.loads(json.dumps(serialized))
    removed = incomplete["entries"][0]["sets"]["resident_lod2"].pop()
    incomplete["entries"][0]["payloads"] = [
        item
        for item in incomplete["entries"][0]["payloads"]
        if item["tile_id"] != removed
    ]
    with pytest.raises(ValueError, match="exhaust"):
        validate_camera_residency_plan(incomplete, catalog)


def test_from_state_restores_pending_load_and_completes_publication() -> None:
    catalog = _grid(1)
    budget = _large_budget()
    camera = _envelope(
        "CAM_RESTART",
        _view(
            "main",
            position=(250.0, 250.0, 100.0),
            forward=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0),
            far_clip_m=200.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, budget)
    planner.stage_camera(camera, now_s=0.0)
    planner.tick(now_s=0.0)
    checkpoint = planner.state.to_dict()

    restored = FrustumStreamingPlanner.from_state(catalog, budget, checkpoint)
    assert restored.state.to_dict() == checkpoint
    _commit_success(restored, restored.state.pending_action)  # type: ignore[arg-type]
    for tick in range(1, 20):
        action = restored.tick(now_s=tick / 10.0)
        _commit_success(restored, action)
        if action.kind == PUBLISH_CAMERA:
            break
    else:
        pytest.fail("restored planner never published the complete camera")
    assert restored.state.active_camera_id == "CAM_RESTART"
    assert all(ref.stitch_mask in range(16) for ref in restored.state.resident_payloads)


def test_from_state_restores_quarantine_retry_and_rejects_tampering() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    budget = _large_budget()
    camera = _envelope(
        "CAM_QUARANTINE_RESTART",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(
        catalog, budget, maximum_load_failure_count=1, load_retry_backoff_ticks=2
    )
    planner.stage_camera(camera, now_s=0.0)
    failed = planner.tick(now_s=0.0)
    pending_checkpoint = planner.state.to_dict()
    planner.commit(failed, succeeded=False, error="bad payload")
    checkpoint = planner.state.to_dict()

    restored = FrustumStreamingPlanner.from_state(catalog, budget, checkpoint)
    assert restored.state.failure_counts == {failed.payloads[0]: 1}
    assert restored.tick(now_s=1.0).reason == "staging_contains_quarantined_lod0"
    restored.retry_quarantined()
    repaired = restored.tick(now_s=2.0)
    _commit_success(restored, repaired)

    wrong_build = json.loads(json.dumps(checkpoint))
    wrong_build["terrain_build_id"] = "0" * 64
    with pytest.raises(ValueError, match="build_id"):
        FrustumStreamingPlanner.from_state(catalog, budget, wrong_build)

    wrong_budget = json.loads(json.dumps(checkpoint))
    wrong_budget["budget"]["cpu_bytes"] += 1
    with pytest.raises(ValueError, match="budget"):
        FrustumStreamingPlanner.from_state(catalog, budget, wrong_budget)

    wrong_pending = json.loads(json.dumps(pending_checkpoint))
    wrong_pending["last_action_sequence"] += 1
    with pytest.raises(ValueError, match="generation/sequence"):
        FrustumStreamingPlanner.from_state(catalog, budget, wrong_pending)


def test_from_state_restores_masked_publication_handshake() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    budget = _large_budget()
    camera = _envelope(
        "CAM_PENDING_PUBLISH",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, budget)
    planner.stage_camera(camera, now_s=0.0)
    _commit_success(planner, planner.tick(now_s=0.0))
    publish = planner.tick(now_s=0.1)
    assert publish.kind == PUBLISH_CAMERA
    assert publish.payloads == (PayloadRef("detail", LOD0, 0),)
    checkpoint = planner.state.to_dict()

    restored = FrustumStreamingPlanner.from_state(catalog, budget, checkpoint)
    _commit_success(restored, restored.state.pending_action)  # type: ignore[arg-type]
    assert restored.state.active_camera_id == camera.camera_id

    wrong_mask = json.loads(json.dumps(checkpoint))
    wrong_mask["pending_action"]["payloads"][0]["stitch_mask"] = 1
    with pytest.raises(ValueError, match="publication"):
        FrustumStreamingPlanner.from_state(catalog, budget, wrong_mask)


def test_pending_publication_restore_rejects_lower_lod_on_staged_visible_tile() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    budget = _large_budget()
    camera = _envelope(
        "CAM_FORGED_PUBLISH",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, budget)
    planner.stage_camera(camera, now_s=0.0)
    _commit_success(planner, planner.tick(now_s=0.0))
    publish = planner.tick(now_s=0.1)
    assert publish.kind == PUBLISH_CAMERA

    forged = planner.state.to_dict()
    forged["resident_payloads"].append(PayloadRef("detail", LOD2, 0).to_dict())
    with pytest.raises(ValueError, match="visible staging tiles"):
        FrustumStreamingPlanner.from_state(catalog, budget, forged)


def test_publication_commit_keeps_pending_action_when_preconditions_fail() -> None:
    catalog = TerrainTileCatalog((_tile(1, 0, identifier="detail"),))
    budget = _large_budget()
    camera = _envelope(
        "CAM_TRANSACTIONAL_PUBLISH",
        _view(
            "main",
            position=(0.0, 250.0, 5.0),
            forward=(1.0, 0.0, 0.0),
            far_clip_m=1_500.0,
        ),
    )
    planner = FrustumStreamingPlanner(catalog, budget)
    planner.stage_camera(camera, now_s=0.0)
    _commit_success(planner, planner.tick(now_s=0.0))
    publish = planner.tick(now_s=0.1)
    assert publish.kind == PUBLISH_CAMERA

    # Simulate corruption between action issuance and acknowledgement.
    planner._resident.add(PayloadRef("detail", LOD2, 0))
    before = planner.state.to_dict()
    with pytest.raises(RuntimeError, match="forbids LOD1 or LOD2"):
        planner.commit(publish, succeeded=True)

    assert planner.state.to_dict() == before
    assert planner.state.pending_action == publish
