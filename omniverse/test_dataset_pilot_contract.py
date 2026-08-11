from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from run_fireviewer_replicator_dataset import (
    build_flow_truth_alignment_geometry,
    classify_capture_visibility_acceptance,
    classify_fire_visibility_regime,
    flow_capture_plume_profile_contract,
    load_pilot_acceptance_receipt,
    parse_camera_priority,
    parse_state_indices,
    parse_visual_calibration_camera_ids,
    prioritize_camera_ids,
    project_truth_masks,
    resolve_production_chunk_id,
    resolve_state_selection,
    selected_capture_counts,
    selected_schedule_states,
    selected_visual_calibration_counts,
    sky_capture_profile_contract,
    split_camera_batches,
    validate_dataset_id,
    validate_production_chunk_id,
)


def _quad(path: str, points: list[list[float]]) -> dict:
    return {
        "path": path,
        "points": np.asarray(points, dtype=np.float64),
        "face_vertex_counts": np.asarray([4], dtype=np.int32),
        "face_vertex_indices": np.asarray([0, 1, 2, 3], dtype=np.int32),
    }


def _truth_geometry() -> dict:
    return {
        "schema": "fireviewer.projected-truth-geometry.v1",
        "fire_front": _quad(
            "/Truth/VisibleFireFront",
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 4.0], [-2.0, 0.0, 4.0]],
        ),
        "fire_perimeter": _quad(
            "/Truth/BurnedPerimeter",
            [[-3.0, -1.0, 0.1], [3.0, -1.0, 0.1], [3.0, 1.0, 0.1], [-3.0, 1.0, 0.1]],
        ),
        "burned_area": _quad(
            "/Truth/BurnedSurface",
            [[-4.0, -3.0, 0.0], [4.0, -3.0, 0.0], [4.0, 3.0, 0.0], [-4.0, 3.0, 0.0]],
        ),
        "smoke_source": {
            "path": "/Truth/SmokeSources",
            "points": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            "widths_m": np.asarray([2.0], dtype=np.float64),
        },
        "smoke_velocity_local_m_s": np.asarray([0.0, 0.0, 8.5], dtype=np.float64),
    }


def _view(*, expected_fire_visible: bool) -> dict:
    return {
        "camera_id": "CAM_TEST",
        "state_id": "state_001",
        "zoom_index": 2,
        "expected_fire_visible": expected_fire_visible,
        "camera_position_local_m": [0.0, -30.0, 5.0],
        "camera_aim_local_m": [0.0, 0.0, 2.0],
        "focal_length_mm": 50.0,
        "horizontal_aperture_mm": 36.0,
        "vertical_aperture_mm": 20.25,
    }


def sample_schedule() -> dict:
    return {
        "zooms_per_view": 5,
        "states": [
            {
                "state_id": "state_001",
                "global_state_index": 1,
                "view_count": 20,
                "positive_view_count": 16,
                "negative_view_count": 4,
            },
            {
                "state_id": "state_002",
                "global_state_index": 2,
                "view_count": 20,
                "positive_view_count": 0,
                "negative_view_count": 20,
            },
            {
                "state_id": "state_003",
                "global_state_index": 3,
                "view_count": 20,
                "positive_view_count": 16,
                "negative_view_count": 4,
            },
        ],
    }


def test_pilot_selection_preserves_requested_order_and_exact_quota() -> None:
    schedule = sample_schedule()
    states = selected_schedule_states(schedule, "3,2")

    assert [state["state_id"] for state in states] == ["state_003", "state_002"]
    assert selected_capture_counts(states, zooms_per_view=5) == {
        "viewpoint_plans": 40,
        "capture_cases": 200,
        "positive_cases": 80,
        "negative_cases": 120,
    }


@pytest.mark.parametrize("value", ["", "1,1", "0", "4", "one"])
def test_invalid_pilot_state_indices_fail_closed(value: str) -> None:
    with pytest.raises(ValueError):
        parse_state_indices(value, state_count=3)


def test_production_selection_is_distinct_from_pilot_selection() -> None:
    selector, option_name, run_kind = resolve_state_selection(
        pilot_state_indices=None,
        production_state_indices="1,2,3",
    )

    assert selector == "1,2,3"
    assert option_name == "--production-state-indices"
    assert run_kind == "production_chunk"


def test_pilot_and_production_state_selectors_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_state_selection(
            pilot_state_indices="1",
            production_state_indices="2",
        )


def test_production_chunk_requires_id_and_pilot_acceptance_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="production-chunk-id"):
        resolve_production_chunk_id(
            production_state_indices="1",
            production_chunk_id=None,
            pilot_acceptance_report=tmp_path / "audit.json",
        )
    with pytest.raises(ValueError, match="pilot-acceptance-report"):
        resolve_production_chunk_id(
            production_state_indices="1",
            production_chunk_id="simulationDS1_diev1-state001",
            pilot_acceptance_report=None,
        )


@pytest.mark.parametrize("value", [None, "", "bad id", "/absolute", "échec"])
def test_invalid_production_chunk_identity_is_rejected(value: str | None) -> None:
    with pytest.raises(ValueError):
        validate_production_chunk_id(value)


def _write_accepted_pilot_receipt_fixture(tmp_path: Path) -> Path:
    contract_path = tmp_path / "run-contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema": "fireviewer.kit-dataset-production-run.v1",
                "run_kind": "pilot",
                "dataset_admissible": True,
                "dataset_id": "simulationDS1_diev1_state003_fullpilot",
                "source_package_id": "fireviewer-die-pontaix-r1-v4",
                "source_stage_sha256": "a" * 64,
                "expected_capture_cases": 100,
                "expected_viewpoint_plans": 20,
                "selected_state_count": 1,
                "selected_state_indices": [3],
                "resolution_px": [1280, 720],
                "rt_subframes": 8,
                "flow_warmup_updates": 180,
                "playback_seconds_per_day": 60.0,
                "flow_capture_profile": {"profile_sha256": "b" * 64},
                "sky_capture_profile": {"profile_sha256": "c" * 64},
                "capture_storage_profile": {"profile_sha256": "e" * 64},
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "audit-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "fireviewer.capture-metadata-audit.v2",
                "status": "passed",
                "failed_capture_count": 0,
                "abstention_warning_count": 0,
                "captures": 100,
                "expected_captures": 100,
                "viewpoint_plans": 20,
                "expected_viewpoint_plans": 20,
                "dataset_id": "simulationDS1_diev1_state003_fullpilot",
                "source_package_id": "fireviewer-die-pontaix-r1-v4",
                "run_contract": str(contract_path),
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_production_receipt_locks_stage_profiles_and_render_parameters(
    tmp_path: Path,
) -> None:
    report_path = _write_accepted_pilot_receipt_fixture(tmp_path)

    receipt = load_pilot_acceptance_receipt(
        report_path,
        source_package_id="fireviewer-die-pontaix-r1-v4",
        source_stage_sha256="a" * 64,
        resolution_px=(1280, 720),
        rt_subframes=8,
        flow_warmup_updates=180,
        playback_seconds_per_day=60.0,
        flow_profile_sha256="b" * 64,
        sky_profile_sha256="c" * 64,
        capture_storage_profile_sha256="e" * 64,
    )

    assert receipt["schema"] == "fireviewer.accepted-pilot-receipt.v1"
    assert receipt["pilot_selected_state_indices"] == [3]
    assert receipt["captures"] == 100
    assert len(receipt["audit_report_sha256"]) == 64
    assert len(receipt["run_contract_sha256"]) == 64


def test_production_receipt_rejects_render_profile_drift(tmp_path: Path) -> None:
    report_path = _write_accepted_pilot_receipt_fixture(tmp_path)

    with pytest.raises(ValueError, match="pilot_flow_profile_mismatch"):
        load_pilot_acceptance_receipt(
            report_path,
            source_package_id="fireviewer-die-pontaix-r1-v4",
            source_stage_sha256="a" * 64,
            resolution_px=(1280, 720),
            rt_subframes=8,
            flow_warmup_updates=180,
            playback_seconds_per_day=60.0,
            flow_profile_sha256="d" * 64,
            sky_profile_sha256="c" * 64,
            capture_storage_profile_sha256="e" * 64,
        )


def test_production_receipt_rejects_capture_storage_profile_drift(
    tmp_path: Path,
) -> None:
    report_path = _write_accepted_pilot_receipt_fixture(tmp_path)

    with pytest.raises(ValueError, match="pilot_capture_storage_profile_mismatch"):
        load_pilot_acceptance_receipt(
            report_path,
            source_package_id="fireviewer-die-pontaix-r1-v4",
            source_stage_sha256="a" * 64,
            resolution_px=(1280, 720),
            rt_subframes=8,
            flow_warmup_updates=180,
            playback_seconds_per_day=60.0,
            flow_profile_sha256="b" * 64,
            sky_profile_sha256="c" * 64,
            capture_storage_profile_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("simulationDS1_diev1", "simulationDS1_diev1"),
        ("reproDS1_diev1", "reproDS1_diev1"),
        (None, "source-package-r1"),
    ],
)
def test_dataset_identity_contract(value: str | None, expected: str) -> None:
    assert validate_dataset_id(value, fallback="source-package-r1") == expected


@pytest.mark.parametrize("value", ["bad id", "/absolute", "", "échec"])
def test_invalid_dataset_identity_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        validate_dataset_id(value, fallback="fallback")


def test_camera_micro_batches_preserve_order_without_duplication() -> None:
    camera_ids = [f"CAM_{index:02d}" for index in range(1, 21)]

    batches = split_camera_batches(camera_ids, batch_size=2)

    assert len(batches) == 10
    assert max(map(len, batches)) == 2
    assert [camera_id for batch in batches for camera_id in batch] == camera_ids


def test_camera_micro_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        split_camera_batches(["CAM_01"], batch_size=0)


def test_pilot_camera_priority_only_reorders_and_preserves_full_content() -> None:
    camera_ids = ["CAM_25", "CAM_58", "CAM_38", "CAM_03"]
    priority = parse_camera_priority(
        "CAM_58,CAM_38", available_camera_ids=set(camera_ids)
    )

    reordered = prioritize_camera_ids(camera_ids, priority)

    assert reordered == ["CAM_58", "CAM_38", "CAM_25", "CAM_03"]
    assert len(reordered) == len(camera_ids)
    assert set(reordered) == set(camera_ids)


@pytest.mark.parametrize("value", ["", "CAM_58,CAM_58", "CAM_99"])
def test_invalid_pilot_camera_priority_fails_closed(value: str) -> None:
    with pytest.raises(ValueError):
        parse_camera_priority(value, available_camera_ids={"CAM_58", "CAM_38"})


def test_visual_calibration_subset_keeps_five_zooms_but_is_counted_separately() -> None:
    states = [
        {
            "state_id": "state_001",
            "views": [
                {"camera_id": "CAM_38", "expected_fire_visible": True},
                {"camera_id": "CAM_58", "expected_fire_visible": False},
            ],
        }
    ]
    cameras = parse_visual_calibration_camera_ids(
        "CAM_38", available_camera_ids={"CAM_38", "CAM_58"}
    )

    assert selected_visual_calibration_counts(
        states, camera_ids=cameras, zooms_per_view=5
    ) == {
        "viewpoint_plans": 1,
        "capture_cases": 5,
        "positive_cases": 5,
        "negative_cases": 0,
    }


@pytest.mark.parametrize("value", ["", "CAM_38,CAM_38", "CAM_99"])
def test_invalid_visual_calibration_camera_subset_fails_closed(value: str) -> None:
    with pytest.raises(ValueError):
        parse_visual_calibration_camera_ids(
            value, available_camera_ids={"CAM_38", "CAM_58"}
        )


def test_visual_calibration_camera_must_exist_in_each_selected_state() -> None:
    states = [
        {
            "state_id": "state_001",
            "views": [{"camera_id": "CAM_38", "expected_fire_visible": True}],
        }
    ]
    with pytest.raises(ValueError):
        selected_visual_calibration_counts(
            states, camera_ids=["CAM_58"], zooms_per_view=5
        )


def test_convective_plume_profile_is_session_only_and_physically_visible() -> None:
    profile = flow_capture_plume_profile_contract()
    smoke = profile["static_overrides"][
        "/World/FireScenario/FlowVisual/SmokePlumeEmitter"
    ]
    advection = profile["static_overrides"][
        "/World/FireScenario/FlowVisual/Simulate/advection"
    ]
    smoke_channel = profile["static_overrides"][
        "/World/FireScenario/FlowVisual/Simulate/advection/smoke"
    ]

    assert profile["profile_id"] == "wildfire_convective_plume_mid_distance_v2"
    assert profile["session_layer_only"] is True
    assert profile["source_stage_modified"] is False
    assert len(profile["profile_sha256"]) == 64
    assert smoke["smoke"] >= 0.8
    assert smoke["maxDistance"] - smoke["minDistance"] >= 2.0
    assert profile["dynamic_smoke_velocity"]["vertical_m_s"] >= 18.0
    assert advection["buoyancyPerSmoke"] >= 6.0
    assert smoke_channel["fade"] <= 0.01


def test_flow_truth_alignment_geometry_reprojects_all_emitters_and_smoke_bases() -> None:
    truth = [[float(index), float(index * 2), 700.0 + index] for index in range(48)]
    widths = [2.0] * 48

    geometry = build_flow_truth_alignment_geometry(truth, widths)

    assert len(geometry["hotspot_positions"]) == 48
    assert len(geometry["smoke_mesh_positions"]) == 48 * 4
    assert geometry["hotspot_positions"][0] == [0.0, 0.0, 700.25]
    first_quad = geometry["smoke_mesh_positions"][:4]
    assert first_quad == [
        [-1.0, -1.0, 701.85],
        [1.0, -1.0, 701.85],
        [1.0, 1.0, 701.85],
        [-1.0, 1.0, 701.85],
    ]


@pytest.mark.parametrize(
    ("truth", "widths"),
    [
        ([[0.0, 0.0, 0.0]], [1.0]),
        ([[0.0, 0.0, 0.0]] * 48, [0.0] * 48),
        ([[0.0, 0.0, float("nan")]] * 48, [1.0] * 48),
    ],
)
def test_invalid_flow_truth_alignment_geometry_fails_closed(
    truth: list[list[float]], widths: list[float]
) -> None:
    with pytest.raises(ValueError):
        build_flow_truth_alignment_geometry(truth, widths)


@pytest.mark.parametrize(
    ("elapsed_s", "area_m2", "spread_rate", "expected_regime"),
    [
        (1.0, 625.0, 0.0, "incipient"),
        (600.0, 5_000.0, 0.0, "stalled"),
        (900.0, 10_000.0, 0.06, "slowed"),
        (900.0, 10_000.0, 0.16, "established"),
    ],
)
def test_fire_visibility_regimes_preserve_legitimate_low_signal_states(
    elapsed_s: float,
    area_m2: float,
    spread_rate: float,
    expected_regime: str,
) -> None:
    result = classify_fire_visibility_regime(
        fire_elapsed_s=elapsed_s,
        burned_area_m2=area_m2,
        mean_front_spread_rate_m_s=spread_rate,
    )

    assert result["regime"] == expected_regime
    assert result["low_signal_state_allowed"] is (
        expected_regime != "established"
    )


def test_long_range_incipient_capture_is_retained_as_hard_positive() -> None:
    result = classify_capture_visibility_acceptance(
        {
            "expected_fire_visible": True,
            "fire_visibility_regime": "incipient",
            "nearest_flame_distance_m": 2_471.0,
            "zoom_multiplier": 0.75,
        }
    )

    assert result["acceptance_class"] == "valid_low_or_partial_signal_positive"
    assert result["low_signal_allowed"] is True
    assert result["training_role"] == "hard_positive"
    assert set(result["low_signal_reasons"]) == {
        "fire_state_incipient",
        "long_range_small_angular_target",
        "wide_frame_small_target",
    }


def test_sky_contrast_profile_is_session_only_and_traceable() -> None:
    profile = sky_capture_profile_contract()

    assert profile["profile_id"] == "clear_daylight_smoke_contrast_v2"
    assert profile["session_layer_only"] is True
    assert profile["source_stage_modified"] is False
    assert profile["overrides"]["inputs:texture:file"] == ""
    assert profile["overrides"]["inputs:color"][2] > profile["overrides"]["inputs:color"][0]
    assert len(profile["profile_sha256"]) == 64


def test_projected_truth_masks_are_dense_aligned_and_traceable() -> None:
    resolution = (320, 180)

    arrays, metadata = project_truth_masks(
        view=_view(expected_fire_visible=True),
        truth_geometry=_truth_geometry(),
        resolution=resolution,
    )

    assert set(arrays) == {
        "fire_front",
        "fire_perimeter",
        "smoke_source",
        "smoke",
        "burned_area",
    }
    assert all(array.shape == (180, 320) for array in arrays.values())
    assert all(array.dtype == np.uint8 for array in arrays.values())
    assert all(np.any(array) for array in arrays.values())
    assert np.all((arrays["smoke_source"] != 0) <= (arrays["smoke"] != 0))
    assert np.count_nonzero(arrays["smoke"]) > np.count_nonzero(arrays["smoke_source"])
    assert metadata["schema"] == "fireviewer.projected-dense-targets.v1"
    assert metadata["source_contract"] == "active_usd_truth_geometry_projected_through_exact_capture_camera"
    assert metadata["pixel_counts"] == {
        name: int(np.count_nonzero(array)) for name, array in arrays.items()
    }


def test_negative_context_truth_masks_are_forced_empty_by_schedule_contract() -> None:
    arrays, metadata = project_truth_masks(
        view=_view(expected_fire_visible=False),
        truth_geometry=_truth_geometry(),
        resolution=(320, 180),
    )

    assert all(not np.any(array) for array in arrays.values())
    assert metadata["expected_fire_visible"] is False
    assert metadata["source_contract"] == "schedule_negative_context_forced_empty_after_camera_rotated_away"
