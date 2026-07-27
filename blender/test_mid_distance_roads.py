from __future__ import annotations

import json
import unittest

import numpy as np
from affine import Affine
from shapely.geometry import LineString

from mid_distance_roads import (
    MidDistanceRoadConfig,
    build_mid_distance_road_geometry,
    resolve_road_width_m,
    resolve_shoulder_width_m,
    road_importance_rank,
)
from spatial_data import LineFeature


def _terrain_plane(size: int = 32) -> np.ndarray:
    """At cell centre (x, y), elevation is (x - .5) + 10 * (y - .5)."""

    rows, columns = np.indices((size, size), dtype=float)
    return columns + rows * 10.0


class MidDistanceRoadGeometryTests(unittest.TestCase):
    def test_width_importance_and_shoulder_rules_match_bd_topo_properties(self) -> None:
        self.assertEqual(
            resolve_road_width_m({"largeur_de_chaussee": "6,5 m"}),
            (6.5, "source_width"),
        )
        self.assertEqual(resolve_road_width_m({"importance": "2"}), (8.0, "importance"))
        self.assertEqual(resolve_road_width_m({}), (4.0, "fallback_unclassified"))
        self.assertEqual(road_importance_rank({"road_class": 3}), 3)
        self.assertIsNone(road_importance_rank({"importance": 9}))

        config = MidDistanceRoadConfig()
        self.assertEqual(
            resolve_shoulder_width_m({"importance": 1}, config),
            (1.75, "major_importance_1_2"),
        )
        self.assertEqual(
            resolve_shoulder_width_m({"importance": 4}, config),
            (1.2, "secondary_importance_3_4"),
        )
        self.assertEqual(
            resolve_shoulder_width_m({"importance": 7}, config),
            (0.7, "minor_importance_5_7"),
        )

    def test_builds_four_surface_meshes_and_drapes_every_lateral_edge(self) -> None:
        feature = LineFeature(
            "axis:0",
            LineString([(2.5, 10.5), (22.5, 10.5)]),
            {"largeur_de_chaussee": 2.0, "importance": 2},
        )
        meshes, statistics = build_mid_distance_road_geometry(
            [feature],
            _terrain_plane(),
            Affine.identity(),
            (0.0, 0.0, 0.0),
        )

        self.assertEqual(
            set(meshes),
            {"carriageway", "left_shoulders", "right_shoulders", "center_markings"},
        )
        self.assertEqual(meshes["carriageway"]["faces"], [[0, 1, 3, 2]])
        self.assertEqual(meshes["left_shoulders"]["faces"], [[0, 1, 3, 2]])
        self.assertEqual(meshes["right_shoulders"]["faces"], [[0, 1, 3, 2]])
        self.assertEqual(len(meshes["center_markings"]["faces"]), 2)

        # Straight eastbound road: left/right edges are at y +/- 1 m.  Their
        # unequal Z values prove that the two sides were sampled independently.
        self.assertEqual(meshes["carriageway"]["vertices"][0], [2.5, 11.5, 112.35])
        self.assertEqual(meshes["carriageway"]["vertices"][1], [2.5, 9.5, 92.35])
        self.assertEqual(meshes["left_shoulders"]["vertices"][0], [2.5, 13.25, 129.79])
        self.assertEqual(meshes["left_shoulders"]["vertices"][1], [2.5, 11.5, 112.335])
        self.assertEqual(meshes["right_shoulders"]["vertices"][0], [2.5, 9.5, 92.335])
        self.assertEqual(meshes["right_shoulders"]["vertices"][1], [2.5, 7.75, 74.79])

        marking_vertices = meshes["center_markings"]["vertices"]
        self.assertEqual(marking_vertices[0], [2.5, 10.59, 103.268])
        self.assertEqual(marking_vertices[1], [2.5, 10.41, 101.468])
        self.assertEqual(
            statistics["altitude_method"], "bilinear_mnt_independent_lateral_rails"
        )
        self.assertEqual(statistics["marking_eligible_line_count"], 1)
        self.assertEqual(statistics["center_marking_dash_count"], 2)
        self.assertEqual(statistics["center_marking_length_m"], 6.0)
        self.assertEqual(statistics["terrain_fallback_sample_count"], 0)
        self.assertEqual(
            set(statistics["draped_rail_sample_counts"]),
            {
                "carriageway_left_edge",
                "carriageway_right_edge",
                "center_marking_left_edge",
                "center_marking_right_edge",
                "left_shoulder_inner_edge",
                "left_shoulder_outer_edge",
                "right_shoulder_inner_edge",
                "right_shoulder_outer_edge",
            },
        )
        json.dumps({"meshes": meshes, "statistics": statistics}, allow_nan=False)

    def test_minor_roads_keep_surface_shoulders_but_do_not_receive_markings(
        self,
    ) -> None:
        feature = LineFeature(
            "track:0",
            LineString([(2.5, 8.5), (12.5, 8.5)]),
            {"importance": 6},
        )
        meshes, statistics = build_mid_distance_road_geometry(
            [feature], _terrain_plane(), Affine.identity(), (0.0, 0.0, 0.0)
        )
        self.assertEqual(len(meshes["carriageway"]["faces"]), 1)
        self.assertEqual(len(meshes["left_shoulders"]["faces"]), 1)
        self.assertEqual(len(meshes["right_shoulders"]["faces"]), 1)
        self.assertEqual(meshes["center_markings"], {"vertices": [], "faces": []})
        self.assertEqual(statistics["marking_eligible_line_count"], 0)
        self.assertEqual(statistics["maximum_shoulder_width_m"], 0.7)

    def test_north_up_affine_and_lambert_origin_are_applied_before_local_coordinates(
        self,
    ) -> None:
        rows, columns = np.indices((10, 10), dtype=float)
        terrain = columns * 10.0 + rows * 100.0
        transform = Affine.translation(1000.0, 2000.0) * Affine.scale(5.0, -5.0)
        feature = LineFeature(
            "lambert:0",
            LineString([(1012.5, 1982.5), (1022.5, 1982.5)]),
            {"largeur_de_chaussee": 2.0, "importance": 6},
        )
        meshes, statistics = build_mid_distance_road_geometry(
            [feature], terrain, transform, (1000.0, 1900.0, 250.0)
        )
        self.assertEqual(
            meshes["carriageway"]["vertices"][:2],
            [[12.5, 83.5, 50.35], [12.5, 81.5, 90.35]],
        )
        self.assertEqual(statistics["terrain_clamped_sample_count"], 0)
        self.assertEqual(
            statistics["terrain_sample_count"],
            sum(statistics["draped_rail_sample_counts"].values()),
        )

    def test_densification_is_effective_but_bounded_per_source_segment(self) -> None:
        config = MidDistanceRoadConfig(
            max_drape_segment_length_m=10.0,
            max_subdivisions_per_source_segment=4,
        )
        feature = LineFeature(
            "long:0",
            LineString([(1.5, 5.5), (101.5, 5.5)]),
            {},
        )
        meshes, statistics = build_mid_distance_road_geometry(
            [feature],
            _terrain_plane(128),
            Affine.identity(),
            (0.0, 0.0, 0.0),
            config=config,
        )
        self.assertEqual(len(meshes["carriageway"]["faces"]), 4)
        self.assertEqual(statistics["source_segment_count"], 1)
        self.assertEqual(statistics["draped_segment_count"], 4)
        self.assertEqual(statistics["draped_station_count"], 5)
        self.assertEqual(statistics["added_drape_station_count"], 3)
        self.assertEqual(statistics["bounded_segment_expansion_ratio"], 4.0)
        self.assertEqual(statistics["maximum_subdivisions_used"], 4)
        self.assertEqual(statistics["maximum_realized_segment_length_m"], 25.0)
        self.assertEqual(statistics["subdivision_cap_hit_count"], 1)

    def test_nodata_uses_explicit_origin_fallback_and_reports_every_sample(
        self,
    ) -> None:
        feature = LineFeature(
            "nodata:0",
            LineString([(1.5, 5.5), (6.5, 5.5)]),
            {"importance": 6},
        )
        meshes, statistics = build_mid_distance_road_geometry(
            [feature],
            np.full((10, 10), np.nan),
            Affine.identity(),
            (0.0, 0.0, 100.0),
        )
        self.assertGreater(statistics["terrain_sample_count"], 0)
        self.assertEqual(
            statistics["terrain_fallback_sample_count"],
            statistics["terrain_sample_count"],
        )
        for vertex in meshes["carriageway"]["vertices"]:
            self.assertEqual(vertex[2], 0.35)

    def test_invalid_budget_configuration_is_rejected_before_geometry(self) -> None:
        feature = LineFeature("road:0", LineString([(0.0, 0.0), (1.0, 0.0)]), {})
        with self.assertRaisesRegex(ValueError, "max_subdivisions"):
            build_mid_distance_road_geometry(
                [feature],
                np.ones((2, 2)),
                Affine.identity(),
                (0.0, 0.0, 0.0),
                config=MidDistanceRoadConfig(max_subdivisions_per_source_segment=0),
            )


if __name__ == "__main__":
    unittest.main()
