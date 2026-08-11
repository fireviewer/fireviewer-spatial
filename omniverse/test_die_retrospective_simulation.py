from __future__ import annotations

import unittest

from build_fireviewer_dataset_usd import build_capture_schedule, flow_state_time_samples


def camera(camera_id: str, *, capability: str, profile: str, role: str = "near") -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "sample_capability": capability,
        "role": role,
        "capture_device_profile": profile,
        "horizontal_aperture_mm": 36.0,
        "vertical_aperture_mm": 20.25,
        "focal_length_mm": 26.0 if profile == "smartphone_main_26mm_equivalent" else 50.0,
        "focal_length_35mm_equivalent_mm": 26.0 if profile == "smartphone_main_26mm_equivalent" else 50.0,
        "placement_type": "roadside" if role != "aerial" else "aerial",
        "placement_contract": "test",
        "framing_style": "test",
        "access_surface": "mapped_road",
        "access_tile": "tile",
        "host_building": "",
        "expected_fire_in_frame": capability == "positive_fire",
        "thermal_capture": role == "aerial",
        "line_of_sight_verified": True,
        "position_local_m": [0.0, 0.0, 10.0],
        "position_l93_ngf_ign69_m": [884000.0, 6408000.0, 10.0],
        "target_local_m": [10.0, 0.0, 10.0],
        "fire_target_local_m": [10.0, 0.0, 10.0],
        "height_above_ground_m": 2.0,
        "ground_elevation_m": 8.0,
        "terrain_los_clearance_m": 5.0,
        "foreground_clearance_m": 10.0,
        "distance_to_target_m": 10.0,
        "fire_bearing_offset_degrees": 0.0,
    }


class DieRetrospectiveSimulationTest(unittest.TestCase):
    def test_stepped_flow_states_hold_until_each_daily_boundary(self) -> None:
        samples = flow_state_time_samples(
            ["day_1", "day_2", "day_3"],
            seconds_per_state=60.0,
            stepped_transitions=True,
        )

        self.assertEqual(
            samples,
            [
                (0.0, "day_1"),
                (59.999, "day_1"),
                (60.0, "day_2"),
                (119.999, "day_2"),
                (120.0, "day_3"),
            ],
        )

    def test_default_flow_states_keep_the_existing_continuous_contract(self) -> None:
        samples = flow_state_time_samples(
            ["state_1", "state_2"],
            seconds_per_state=6.0,
            stepped_transitions=False,
        )

        self.assertEqual(samples, [(0.0, "state_1"), (6.0, "state_2")])

    def test_daily_schedule_marks_out_of_scene_fire_as_negative_without_changing_camera_pool(self) -> None:
        cameras = []
        for _ in range(7):
            cameras.append(camera(f"CAM_{len(cameras) + 1:02d}", capability="positive_fire", profile="aerial", role="aerial"))
        for _ in range(20):
            cameras.append(camera(f"CAM_{len(cameras) + 1:02d}", capability="positive_fire", profile="smartphone_main_26mm_equivalent"))
        for _ in range(20):
            cameras.append(camera(f"CAM_{len(cameras) + 1:02d}", capability="positive_fire", profile="professional_full_frame_50mm"))
        for _ in range(7):
            cameras.append(camera(f"CAM_{len(cameras) + 1:02d}", capability="negative_context", profile="smartphone_main_26mm_equivalent"))
        for _ in range(8):
            cameras.append(camera(f"CAM_{len(cameras) + 1:02d}", capability="negative_context", profile="professional_full_frame_50mm"))
        states = [
            {
                "state_id": f"state_{index:03d}",
                "simulation_id": "simulation",
                "elapsed_s": float((index - 1) * 86400),
                "burned_area_m2": float(index),
                "active_front_length_m": 100.0,
                "mean_front_spread_rate_m_s": 0.1,
                "burned_tree_count": index,
                "active_in_scene": index != 15,
            }
            for index in range(1, 22)
        ]

        schedule = build_capture_schedule(
            "package",
            states,
            cameras,
            incident_days=21,
            states_per_day=1,
            incident_kind="retrospective-replay",
            observation_seconds_per_state=86400,
        )

        self.assertEqual(schedule["expected_viewpoint_plans"], 420)
        self.assertEqual(schedule["expected_capture_cases"], 2100)
        self.assertEqual(schedule["expected_positive_cases"], 1600)
        self.assertEqual(schedule["expected_negative_cases"], 500)
        outside = schedule["states"][14]
        self.assertEqual(outside["positive_view_count"], 0)
        self.assertEqual(outside["negative_view_count"], 20)
        self.assertTrue(all(not view["expected_fire_visible"] for view in outside["views"]))
        self.assertTrue(
            all(view["negative_reason"] == "daily_active_zone_outside_validated_scene" for view in outside["views"])
        )


if __name__ == "__main__":
    unittest.main()
