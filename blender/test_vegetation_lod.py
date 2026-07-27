from __future__ import annotations

import json
import math
import unittest

import numpy as np
from affine import Affine

from vegetation_lod import (
    LOD_SCHEMA,
    TreeProxy,
    VegetationLodConfig,
    build_tree_proxy_mesh,
    detect_tree_proxies,
    generate_mid_distance_vegetation_lod,
)


GRID_5M = Affine.translation(1000.0, 2000.0) * Affine.scale(5.0, -5.0)


class VegetationLodDetectionTests(unittest.TestCase):
    def test_equal_height_plateau_has_one_deterministic_representative(self) -> None:
        mnt = np.fromfunction(
            lambda row, column: 100.0 + row * 0.2 + column * 0.1, (6, 7)
        )
        heights = np.zeros((6, 7), dtype="float64")
        heights[1, 1] = 10.0
        heights[1, 2] = 10.0
        heights[4, 5] = 8.0
        mns = mnt + heights
        vegetation = heights >= 3.0
        exclusion = np.zeros_like(vegetation)
        config = VegetationLodConfig(
            min_tree_height_m=3.0,
            local_max_radius_m=5.1,
            min_spacing_m=10.0,
            crown_search_radius_m=5.1,
        )

        first, first_statistics = detect_tree_proxies(
            mnt, mns, GRID_5M, vegetation, exclusion, config
        )
        second, second_statistics = detect_tree_proxies(
            mnt.copy(), mns.copy(), GRID_5M, vegetation.copy(), exclusion.copy(), config
        )

        self.assertEqual(first, second)
        self.assertEqual(first_statistics, second_statistics)
        self.assertEqual([(tree.row, tree.column) for tree in first], [(1, 1), (4, 5)])
        self.assertEqual(first_statistics["locally_maximal_plateau_pixel_count"], 3)
        self.assertEqual(first_statistics["local_maximum_candidate_count"], 2)
        self.assertEqual(
            first_statistics["semantics"],
            "deterministic_lod_representatives_not_exact_tree_inventory",
        )

    def test_exclusion_and_nodata_are_removed_before_peak_selection(self) -> None:
        mnt = np.full((5, 5), 100.0)
        heights = np.zeros((5, 5), dtype="float64")
        heights[1, 1] = 20.0
        heights[3, 3] = 7.0
        heights[4, 4] = 12.0
        mns = mnt + heights
        mns[4, 4] = np.nan
        vegetation = heights > 0
        exclusion = np.zeros((5, 5), dtype=bool)
        exclusion[1, 1] = True

        proxies, statistics = detect_tree_proxies(
            mnt,
            mns,
            GRID_5M,
            vegetation,
            exclusion,
            VegetationLodConfig(crown_search_radius_m=5.0),
        )

        self.assertEqual([(tree.row, tree.column) for tree in proxies], [(3, 3)])
        self.assertEqual(statistics["excluded_vegetation_pixel_count"], 1)
        self.assertEqual(statistics["invalid_vegetation_pixel_count"], 1)
        self.assertEqual(statistics["selected_proxy_count"], 1)

    def test_metric_spacing_keeps_the_higher_peak(self) -> None:
        mnt = np.zeros((6, 7), dtype="float64")
        heights = np.zeros_like(mnt)
        heights[1, 1] = 12.0
        heights[1, 3] = 11.0  # 10 m from the first peak.
        heights[4, 6] = 9.0
        vegetation = heights > 0
        config = VegetationLodConfig(
            local_max_radius_m=4.9,
            min_spacing_m=10.1,
            crown_search_radius_m=4.9,
        )

        proxies, statistics = detect_tree_proxies(
            mnt,
            mnt + heights,
            GRID_5M,
            vegetation,
            np.zeros_like(vegetation),
            config,
        )

        self.assertEqual(
            [(tree.row, tree.column) for tree in proxies], [(1, 1), (4, 6)]
        )
        self.assertEqual(statistics["spacing_rejected_candidate_count"], 1)

    def test_height_and_equivalent_area_diameter_are_measured_from_rasters(
        self,
    ) -> None:
        mnt = np.fromfunction(lambda row, column: 100.0 + row + 2.0 * column, (5, 5))
        heights = np.zeros((5, 5), dtype="float64")
        heights[1:4, 1:4] = 7.0
        heights[2, 2] = 12.0
        vegetation = heights > 0
        config = VegetationLodConfig(
            min_tree_height_m=3.0,
            local_max_radius_m=7.2,
            min_spacing_m=20.0,
            crown_search_radius_m=8.0,
            crown_support_height_ratio=0.5,
        )

        proxies, statistics = detect_tree_proxies(
            mnt,
            mnt + heights,
            GRID_5M,
            vegetation,
            np.zeros_like(vegetation),
            config,
        )

        self.assertEqual(len(proxies), 1)
        proxy = proxies[0]
        self.assertEqual((proxy.row, proxy.column), (2, 2))
        self.assertEqual(proxy.ground_elevation_m, 106.0)
        self.assertEqual(proxy.top_elevation_m, 118.0)
        self.assertEqual(proxy.height_m, 12.0)
        self.assertEqual(proxy.crown_support_pixel_count, 9)
        self.assertEqual(proxy.crown_area_m2, 225.0)
        self.assertAlmostEqual(proxy.crown_diameter_m, 2.0 * math.sqrt(225.0 / math.pi))
        self.assertTrue(proxy.crown_search_limited)
        self.assertEqual(statistics["height_measurement"], "co_located_mns_minus_mnt")
        self.assertEqual(statistics["minimum_measured_height_m"], 12.0)

    def test_optional_proxy_budget_is_deterministic_and_reported(self) -> None:
        mnt = np.zeros((7, 7), dtype="float64")
        heights = np.zeros_like(mnt)
        for row, column, height in (
            (1, 1, 6.0),
            (1, 5, 9.0),
            (5, 1, 8.0),
            (5, 5, 7.0),
        ):
            heights[row, column] = height
        vegetation = heights > 0
        config = VegetationLodConfig(
            local_max_radius_m=4.9,
            min_spacing_m=5.0,
            crown_search_radius_m=4.9,
            max_proxy_count=2,
        )

        proxies, statistics = detect_tree_proxies(
            mnt,
            mnt + heights,
            GRID_5M,
            vegetation,
            np.zeros_like(vegetation),
            config,
        )

        self.assertEqual([tree.height_m for tree in proxies], [9.0, 8.0])
        self.assertEqual(statistics["selected_proxy_count_before_budget"], 4)
        self.assertEqual(statistics["proxy_budget_rejected_candidate_count"], 2)
        self.assertEqual(statistics["selected_proxy_count"], 2)

    def test_masks_must_use_the_full_raster_sample_grid(self) -> None:
        values = np.ones((4, 5), dtype="float64")
        with self.assertRaisesRegex(ValueError, "vegetation_mask shape mismatch"):
            detect_tree_proxies(
                values,
                values + 5.0,
                GRID_5M,
                np.ones((3, 4), dtype=bool),
                np.zeros((4, 5), dtype=bool),
            )


class VegetationLodMeshTests(unittest.TestCase):
    def test_mesh_has_a_volumetric_crown_and_grounded_trunk(self) -> None:
        proxy = TreeProxy(
            row=3,
            column=4,
            x_m=105.0,
            y_m=205.0,
            ground_elevation_m=50.0,
            top_elevation_m=62.0,
            height_m=12.0,
            crown_support_pixel_count=4,
            crown_area_m2=math.pi * 25.0,
            crown_diameter_m=10.0,
            crown_search_limited=False,
        )
        config = VegetationLodConfig(crown_radial_segments=6)

        mesh, statistics = build_tree_proxy_mesh([proxy], (100.0, 200.0, 40.0), config)

        self.assertEqual(len(mesh["vertices"]), 16)
        self.assertEqual(len(mesh["faces"]), 16)
        self.assertEqual(sum(len(face) == 3 for face in mesh["faces"]), 12)
        self.assertEqual(sum(len(face) == 4 for face in mesh["faces"]), 4)
        self.assertEqual(mesh["vertices"][0], [5.0, 5.0, 22.0])
        self.assertEqual(min(vertex[2] for vertex in mesh["vertices"]), 10.0)
        self.assertEqual(max(vertex[2] for vertex in mesh["vertices"]), 22.0)
        ring_xy = {(vertex[0], vertex[1]) for vertex in mesh["vertices"][1:7]}
        self.assertEqual(len(ring_xy), 6)
        self.assertEqual(
            statistics["primitive"], "low_poly_biconic_crown_and_square_trunk"
        )
        self.assertEqual(statistics["vertices_per_proxy"], 16)
        self.assertEqual(statistics["faces_per_proxy"], 16)
        self.assertEqual(statistics["estimated_triangle_count_after_triangulation"], 20)

    def test_integrated_result_is_json_serializable_and_package_ready(self) -> None:
        mnt = np.full((3, 3), 100.0)
        heights = np.zeros((3, 3), dtype="float64")
        heights[1, 1] = 9.0
        vegetation = heights > 0
        result = generate_mid_distance_vegetation_lod(
            mnt,
            mnt + heights,
            GRID_5M,
            vegetation,
            np.zeros_like(vegetation),
            (1000.0, 2000.0, 100.0),
            VegetationLodConfig(
                local_max_radius_m=5.1,
                min_spacing_m=10.0,
                crown_search_radius_m=5.1,
            ),
        )

        json.dumps(result, allow_nan=False)
        self.assertEqual(result["schema"], LOD_SCHEMA)
        self.assertEqual(result["statistics"]["detection"]["selected_proxy_count"], 1)
        self.assertGreater(len(result["mesh"]["vertices"]), 0)
        self.assertGreater(len(result["mesh"]["faces"]), 0)

    def test_configuration_rejects_non_volumetric_or_unbounded_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "crown_radial_segments"):
            VegetationLodConfig(crown_radial_segments=3)
        with self.assertRaisesRegex(ValueError, "max_proxy_count"):
            VegetationLodConfig(max_proxy_count=0)
        with self.assertRaisesRegex(ValueError, "crown_widest_height_ratio"):
            VegetationLodConfig(
                crown_base_height_ratio=0.7, crown_widest_height_ratio=0.6
            )


if __name__ == "__main__":
    unittest.main()
