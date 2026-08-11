from __future__ import annotations

import math
import unittest

import numpy as np

from build_fireviewer_dataset_usd import (
    CAPTURE_ZOOM_PROFILES,
    FLOW_FRONT_PATCH_COUNT,
    FLOW_SMOKE_PLUME_LIFT_M,
    capture_zoom_variants,
    select_flow_front_patches,
    smoke_plume_mesh_vertices,
)
from fire_spread_model import FireFrontSegment


class FlowWildfireProfileTest(unittest.TestCase):
    def test_front_patches_have_fixed_finite_terrain_following_topology(self) -> None:
        segments = [
            FireFrontSegment(start=(1000.0, 2000.0, 500.0), end=(1100.0, 2000.0, 510.0), spread_rate_m_s=0.24),
            FireFrontSegment(start=(1100.0, 2000.0, 510.0), end=(1100.0, 2100.0, 515.0), spread_rate_m_s=0.16),
        ]

        patches = select_flow_front_patches(segments, anchor=(900.0, 1900.0))

        self.assertEqual(len(patches), FLOW_FRONT_PATCH_COUNT)
        for patch in patches:
            vertices = np.asarray(patch["vertices"], dtype=np.float64)
            center = np.asarray(patch["center"], dtype=np.float64)
            self.assertEqual(vertices.shape, (4, 3))
            self.assertTrue(bool(np.all(np.isfinite(vertices))))
            self.assertTrue(bool(np.allclose(vertices.mean(axis=0), center, atol=1e-8)))
            edge_lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
            self.assertTrue(all(math.isfinite(float(value)) and float(value) > 0.0 for value in edge_lengths))

    def test_front_patches_require_a_positive_count(self) -> None:
        segment = FireFrontSegment(start=(0.0, 0.0, 0.0), end=(25.0, 0.0, 0.0), spread_rate_m_s=0.1)
        with self.assertRaisesRegex(ValueError, "patch count"):
            select_flow_front_patches([segment], anchor=(0.0, 0.0), count=0)

    def test_smoke_plume_quads_stay_centered_above_flame_emitters(self) -> None:
        emitters = [
            {"position": (10.0, 20.0, 3.0), "radius_m": 0.8},
            {"position": (-5.0, 7.0, 11.0), "radius_m": 2.0},
        ]

        vertices = np.asarray(smoke_plume_mesh_vertices(emitters), dtype=np.float64).reshape((-1, 4, 3))

        self.assertEqual(vertices.shape, (2, 4, 3))
        for emitter, quad in zip(emitters, vertices):
            center = quad.mean(axis=0)
            position = np.asarray(emitter["position"], dtype=np.float64)
            self.assertTrue(bool(np.allclose(center[:2], position[:2], atol=1e-8)))
            self.assertAlmostEqual(float(center[2] - position[2]), FLOW_SMOKE_PLUME_LIFT_M)
            self.assertTrue(bool(np.allclose(quad[:, 2], center[2], atol=1e-8)))
            side_lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            self.assertTrue(bool(np.all(side_lengths <= 1.24 + 1e-8)))

    def test_smoke_plume_lift_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "plume lift"):
            smoke_plume_mesh_vertices([{"position": (0.0, 0.0, 0.0), "radius_m": 1.0}], lift_m=0.0)

    def test_five_zoom_variants_preserve_sensor_and_pose_contract(self) -> None:
        camera = {
            "focal_length_mm": 50.0,
            "focal_length_35mm_equivalent_mm": 50.0,
            "horizontal_aperture_mm": 36.0,
            "vertical_aperture_mm": 20.25,
        }

        variants = capture_zoom_variants("package", "plan", camera)

        self.assertEqual(len(variants), 5)
        self.assertEqual([variant["zoom_multiplier"] for variant in variants], [profile[1] for profile in CAPTURE_ZOOM_PROFILES])
        self.assertEqual([variant["focal_length_mm"] for variant in variants], [37.5, 50.0, 62.5, 75.0, 100.0])
        self.assertEqual(len({variant["capture_id"] for variant in variants}), 5)
        self.assertTrue(all(variant["horizontal_aperture_mm"] == 36.0 for variant in variants))
        self.assertTrue(all(variant["camera_pose_contract"] == "same_position_orientation_and_target_across_zoom_set" for variant in variants))


if __name__ == "__main__":
    unittest.main()
