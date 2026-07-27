from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import LineString, Polygon

from prepare_control_package import (
    _assert_aligned_grids,
    _canopy_geometry,
    _prisms,
    _road_geometry,
    _road_width_m,
    _water_segment_width_m,
    _water_surface_geometry,
    _vegetation_exclusion_mask,
    parse_arguments,
    prepare_package,
)
from spatial_data import (
    TARGET_CRS,
    LineFeature,
    PolygonFeature,
    choose_local_origin,
    find_absolute_local_paths,
    infer_geojson_crs,
    lambert93_validation,
    numeric_property,
    positive_numeric_property,
    read_polygon_features,
)


class SpatialDataTests(unittest.TestCase):
    def test_portability_guard_finds_windows_and_posix_absolute_paths(self) -> None:
        payload = {
            "portable": {"file_name": "mnt-global.cog.tif"},
            "windows": r"E:\fixtures\fireviewer\mnt.tif",
            "posix": "/tmp/mnt.tif",
        }
        self.assertEqual(
            find_absolute_local_paths(payload),
            ["root.windows", "root.posix"],
        )

    def test_portability_guard_accepts_file_names_and_relative_paths(self) -> None:
        payload = {
            "terrain": {"file_name": "mnt-global.cog.tif"},
            "vector": {"relative_path": "vectors/buildings.l93.geojson"},
        }
        self.assertEqual(find_absolute_local_paths(payload), [])

    def test_cli_height_defaults_are_opt_in(self) -> None:
        args = parse_arguments(
            [
                "prepare",
                "--mnt",
                "mnt.tif",
                "--perimeter",
                "fire.geojson",
                "--validate-only",
            ]
        )
        self.assertIsNone(args.default_building_height_m)
        self.assertEqual(args.terrain_step, 4)
        self.assertEqual(args.building_simplify_m, 0.05)
        self.assertEqual(args.minimum_visible_building_wall_m, 2.70)

    def test_cli_accepts_repeatable_road_and_water_sources(self) -> None:
        args = parse_arguments(
            [
                "prepare",
                "--mnt",
                "mnt.tif",
                "--perimeter",
                "fire.geojson",
                "--roads",
                "roads-1.geojson",
                "--roads",
                "roads-2.geojson",
                "--water-courses",
                "courses.geojson",
                "--water-segments",
                "segments.geojson",
                "--water-surfaces",
                "surfaces.geojson",
                "--validate-only",
            ]
        )
        self.assertEqual(
            [path.name for path in args.roads], ["roads-1.geojson", "roads-2.geojson"]
        )
        self.assertEqual(args.water_courses[0].name, "courses.geojson")
        self.assertEqual(args.water_segments[0].name, "segments.geojson")
        self.assertEqual(args.water_surfaces[0].name, "surfaces.geojson")

    def test_cli_rejects_non_positive_opt_in_height(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            parse_arguments(
                [
                    "prepare",
                    "--mnt",
                    "mnt.tif",
                    "--perimeter",
                    "fire.geojson",
                    "--default-building-height-m",
                    "0",
                    "--validate-only",
                ]
            )

    def test_vegetation_requires_mns_before_any_file_is_read(self) -> None:
        args = parse_arguments(
            [
                "prepare",
                "--mnt",
                "missing-mnt.tif",
                "--perimeter",
                "missing-fire.geojson",
                "--vegetation",
                "missing-vegetation.geojson",
                "--validate-only",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--mns is required"):
            prepare_package(args)

    def test_prepare_package_keeps_buildings_and_adds_mid_distance_lods(self) -> None:
        import rasterio

        transform = Affine.translation(700_000.0, 6_600_030.0) * Affine.scale(5.0, -5.0)
        terrain = np.full((6, 6), 100.0, dtype="float32")
        surface = terrain.copy()
        surface[2, 4] = 111.0
        surface[4, 4] = 108.0

        def feature_collection(features: list[dict[str, object]]) -> str:
            return json.dumps({"type": "FeatureCollection", "features": features})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mnt_path = root / "mnt.tif"
            mns_path = root / "mns.tif"
            for path, values in ((mnt_path, terrain), (mns_path, surface)):
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=values.shape[1],
                    height=values.shape[0],
                    count=1,
                    dtype=values.dtype,
                    crs="EPSG:2154",
                    transform=transform,
                ) as dataset:
                    dataset.write(values, 1)

            perimeter_path = root / "perimeter.geojson"
            perimeter_path.write_text(
                feature_collection(
                    [
                        {
                            "type": "Feature",
                            "id": "fire",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [700_000.0, 6_600_000.0],
                                        [700_030.0, 6_600_000.0],
                                        [700_030.0, 6_600_030.0],
                                        [700_000.0, 6_600_030.0],
                                        [700_000.0, 6_600_000.0],
                                    ]
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            buildings_path = root / "buildings.geojson"
            buildings_path.write_text(
                feature_collection(
                    [
                        {
                            "type": "Feature",
                            "id": "building",
                            "properties": {"hauteur": 7.0},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [700_001.0, 6_600_021.0],
                                        [700_009.0, 6_600_021.0],
                                        [700_009.0, 6_600_029.0],
                                        [700_001.0, 6_600_029.0],
                                        [700_001.0, 6_600_021.0],
                                    ]
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            vegetation_path = root / "vegetation.geojson"
            vegetation_path.write_text(
                feature_collection(
                    [
                        {
                            "type": "Feature",
                            "id": "forest",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [700_000.0, 6_600_000.0],
                                        [700_030.0, 6_600_000.0],
                                        [700_030.0, 6_600_030.0],
                                        [700_000.0, 6_600_030.0],
                                        [700_000.0, 6_600_000.0],
                                    ]
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            roads_path = root / "roads.geojson"
            roads_path.write_text(
                feature_collection(
                    [
                        {
                            "type": "Feature",
                            "id": "road",
                            "properties": {"importance": 2, "largeur_de_chaussee": 6.0},
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [
                                    [700_001.0, 6_600_002.0],
                                    [700_029.0, 6_600_002.0],
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            args = parse_arguments(
                [
                    "prepare",
                    "--mnt",
                    str(mnt_path),
                    "--mns",
                    str(mns_path),
                    "--perimeter",
                    str(perimeter_path),
                    "--perimeter-crs",
                    "EPSG:2154",
                    "--buildings",
                    str(buildings_path),
                    "--buildings-crs",
                    "EPSG:2154",
                    "--vegetation",
                    str(vegetation_path),
                    "--vegetation-crs",
                    "EPSG:2154",
                    "--roads",
                    str(roads_path),
                    "--roads-crs",
                    "EPSG:2154",
                    "--buffer-m",
                    "0",
                    "--terrain-step",
                    "1",
                    "--vegetation-building-clearance-m",
                    "0",
                    "--vegetation-road-clearance-m",
                    "0",
                    "--mid-tree-spacing-m",
                    "5",
                    "--mid-tree-local-max-radius-m",
                    "5",
                    "--validate-only",
                ]
            )
            package = prepare_package(args)

        self.assertEqual(len(package["buildings"]["prisms"]), 1)
        self.assertGreater(
            package["vegetation"]["statistics"]["building_exclusion_cell_count"], 0
        )
        self.assertIn("mid_distance_lod", package["vegetation"])
        self.assertGreater(
            package["vegetation"]["mid_distance_lod"]["statistics"]["mesh"][
                "proxy_count"
            ],
            0,
        )
        self.assertGreater(len(package["roads"]["meshes"]["carriageway"]["faces"]), 0)
        self.assertGreater(
            len(package["roads"]["meshes"]["left_shoulders"]["faces"]), 0
        )
        self.assertGreater(
            len(package["roads"]["meshes"]["center_markings"]["faces"]), 0
        )

    def test_accepts_ign_lambert93_projection_parameters_without_authority(
        self,
    ) -> None:
        class IgnCrs:
            @staticmethod
            def to_epsg():
                return None

            @staticmethod
            def to_dict():
                return {
                    "proj": "lcc",
                    "units": "m",
                    "lat_0": 46.5,
                    "lon_0": 3,
                    "lat_1": 49,
                    "lat_2": 44,
                    "x_0": 700000,
                    "y_0": 6600000,
                }

        self.assertEqual(
            lambert93_validation(IgnCrs()), "lambert93_projection_parameters"
        )

    def test_infers_wgs84_rfc7946_coordinates(self) -> None:
        payload = {
            "type": "Polygon",
            "coordinates": [[[5.3, 44.7], [5.4, 44.7], [5.4, 44.8], [5.3, 44.7]]],
        }
        self.assertEqual(infer_geojson_crs(payload), "EPSG:4326")

    def test_prefers_declared_lambert93_crs(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::2154"},
            },
            "features": [],
        }
        self.assertEqual(infer_geojson_crs(payload), TARGET_CRS)

    def test_reprojects_geojson_polygon_to_lambert93(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "fire-1",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[5.30, 44.70], [5.31, 44.70], [5.31, 44.71], [5.30, 44.70]]
                        ],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "perimeter.geojson"
            source.write_text(json.dumps(payload), encoding="utf-8")
            features, source_crs = read_polygon_features(source)
        self.assertEqual(source_crs, "EPSG:4326")
        self.assertEqual(len(features), 1)
        min_x, min_y, max_x, max_y = features[0].geometry.bounds
        self.assertTrue(870_000 < min_x < 900_000)
        self.assertTrue(6_380_000 < min_y < 6_430_000)
        self.assertGreater(max_x, min_x)
        self.assertGreater(max_y, min_y)

    def test_explicit_axis_swap_normalizes_effis_style_coordinates(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": "SYNTH-001"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[44.70, 5.30], [44.70, 5.31], [44.71, 5.31], [44.70, 5.30]]
                        ],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "effis.geojson"
            source.write_text(json.dumps(payload), encoding="utf-8")
            features, _ = read_polygon_features(source, "EPSG:4326", swap_xy=True)
        self.assertEqual(features[0].feature_id, "SYNTH-001:0")
        min_x, min_y, _, _ = features[0].geometry.bounds
        self.assertTrue(870_000 < min_x < 900_000)
        self.assertTrue(6_380_000 < min_y < 6_430_000)

    def test_numeric_property_accepts_french_decimal_text(self) -> None:
        properties = {"HAUTEUR": "12,5 m"}
        self.assertEqual(numeric_property(properties, ("hauteur",)), 12.5)

    def test_positive_property_uses_priority_and_skips_null_or_zero(self) -> None:
        properties = {"block_height_m": None, "hauteur": 0, "height": "7,25 m"}
        self.assertEqual(
            positive_numeric_property(
                properties, ("block_height_m", "hauteur", "height")
            ),
            7.25,
        )

    def test_positive_property_prefers_computed_block_height(self) -> None:
        properties = {"block_height_m": 11.5, "hauteur": 8.0}
        self.assertEqual(
            positive_numeric_property(properties, ("block_height_m", "hauteur")),
            11.5,
        )

    def test_origin_is_metre_aligned(self) -> None:
        origin = choose_local_origin((700000.0, 6600000.0, 711000.0, 6613000.0), 317.8)
        self.assertEqual(origin, (705500, 6606500, 317))

    def test_prisms_prioritize_computed_values_and_report_missing_height(self) -> None:
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        features = [
            PolygonFeature(
                "computed:0",
                polygon,
                {
                    "base_elevation_m": 123.0,
                    "altitude_minimale_sol": 80.0,
                    "block_height_m": 10.0,
                    "hauteur": 3.0,
                },
            ),
            PolygonFeature(
                "missing:0",
                polygon,
                {"base_elevation_m": 121.0, "block_height_m": None, "hauteur": None},
            ),
            PolygonFeature(
                "missing:1",
                polygon,
                {"base_elevation_m": 121.0, "block_height_m": None, "hauteur": None},
            ),
        ]
        prisms, statistics = _prisms(
            features,
            np.full((2, 2), 100.0),
            Affine.translation(0, 2) * Affine.scale(1, -1),
            (0.0, 0.0, 100.0),
            None,
            0.0,
            ("block_height_m", "hauteur"),
            ("base_elevation_m", "altitude_minimale_sol"),
        )
        self.assertEqual(len(prisms), 1)
        self.assertEqual(prisms[0]["base_z"], 23.0)
        self.assertEqual(prisms[0]["height"], 10.0)
        self.assertEqual(prisms[0]["ground_z_rings"], [[0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(prisms[0]["roof_z"], 33.0)
        self.assertEqual(statistics["draped_foundation_prism_count"], 1)
        self.assertEqual(
            statistics["roof_reference"],
            "max_source_roof_and_highest_mnt_footprint_plus_minimum_wall",
        )
        self.assertEqual(statistics["not_extruded_no_positive_height_count"], 1)
        self.assertEqual(
            statistics["not_extruded_no_positive_height_feature_ids"], ["missing"]
        )
        self.assertEqual(statistics["input_polygon_count"], 3)
        self.assertEqual(statistics["input_entity_count"], 2)
        self.assertEqual(statistics["extruded_entity_count"], 1)
        self.assertEqual(statistics["default_height_used_count"], 0)

    def test_prisms_use_default_only_when_explicitly_provided(self) -> None:
        feature = PolygonFeature(
            "fallback:0",
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            {"base_elevation_m": 100.0, "block_height_m": None},
        )
        prisms, statistics = _prisms(
            [feature],
            np.full((2, 2), 100.0),
            Affine.translation(0, 2) * Affine.scale(1, -1),
            (0.0, 0.0, 100.0),
            6.5,
            0.0,
            ("block_height_m",),
            ("base_elevation_m",),
        )
        self.assertEqual(prisms[0]["height"], 6.5)
        self.assertEqual(statistics["not_extruded_no_positive_height_count"], 0)
        self.assertEqual(statistics["default_height_used_count"], 1)

    def test_prisms_keep_foundations_on_mnt_and_raise_roof_above_terrain(
        self,
    ) -> None:
        feature = PolygonFeature(
            "slope:0",
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            {"base_elevation_m": 100.0, "block_height_m": 5.0},
        )
        prisms, statistics = _prisms(
            [feature],
            np.array([[100.0, 107.0], [100.0, 100.0]]),
            Affine.identity(),
            (0.0, 0.0, 100.0),
            None,
            0.0,
            ("block_height_m",),
            ("base_elevation_m",),
        )
        self.assertEqual(prisms[0]["roof_z"], 9.7)
        self.assertIn(7.0, prisms[0]["ground_z_rings"][0])
        self.assertEqual(prisms[0]["roof_raise_m"], 4.7)
        self.assertEqual(statistics["raised_roof_prism_count"], 1)
        self.assertEqual(statistics["source_clearance_shortfall_vertex_count"], 1)
        self.assertEqual(statistics["maximum_ground_above_roof_m"], 2.0)
        self.assertAlmostEqual(statistics["maximum_roof_raise_m"], 4.7)
        self.assertEqual(statistics["minimum_visible_wall_height_m"], 2.70)

    def test_canopy_top_and_grounded_skirts_follow_sloped_mns_and_mnt(self) -> None:
        mnt = np.array(
            [
                [100.0, 102.0, 104.0],
                [110.0, 112.0, 114.0],
                [120.0, 122.0, 124.0],
            ]
        )
        mns = mnt + 5.0
        cell_mask = np.array([[True, False], [False, False]])
        transform = Affine.translation(0.0, 15.0) * Affine.scale(5.0, -5.0)

        mesh, statistics = _canopy_geometry(
            mnt,
            mns,
            transform,
            cell_mask,
            (0.0, 0.0, 100.0),
        )

        by_xy: dict[tuple[float, float], list[float]] = {}
        for x, y, z in mesh["vertices"]:
            by_xy.setdefault((x, y), []).append(z)
        expected_ground = {
            (2.5, 12.5): 0.0,
            (2.5, 7.5): 10.0,
            (7.5, 7.5): 12.0,
            (7.5, 12.5): 2.0,
        }
        self.assertEqual(set(by_xy), set(expected_ground))
        for coordinate, ground_z in expected_ground.items():
            self.assertEqual(sorted(by_xy[coordinate]), [ground_z, ground_z + 5.0])
        self.assertEqual(statistics["top_face_count"], 1)
        self.assertEqual(statistics["boundary_skirt_face_count"], 4)
        self.assertEqual(statistics["valid_canopy_cell_count"], 1)
        self.assertEqual(statistics["minimum_canopy_height_m"], 5.0)
        self.assertEqual(statistics["maximum_canopy_height_m"], 5.0)

    def test_mns_mnt_alignment_accepts_identical_grid(self) -> None:
        transform = Affine.translation(879000.0, 6412000.0) * Affine.scale(5.0, -5.0)
        _assert_aligned_grids((100, 120), transform, (100, 120), transform)

    def test_mns_mnt_alignment_rejects_shifted_grid(self) -> None:
        mnt_transform = Affine.translation(879000.0, 6412000.0) * Affine.scale(
            5.0, -5.0
        )
        shifted_mns = Affine.translation(879002.5, 6412000.0) * Affine.scale(5.0, -5.0)
        with self.assertRaisesRegex(ValueError, "transform mismatch"):
            _assert_aligned_grids((100, 120), mnt_transform, (100, 120), shifted_mns)

    def test_mns_mnt_alignment_rejects_different_shape(self) -> None:
        transform = Affine.translation(879000.0, 6412000.0) * Affine.scale(5.0, -5.0)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            _assert_aligned_grids((100, 120), transform, (99, 120), transform)

    def test_road_width_prefers_bdtopo_width_then_importance(self) -> None:
        self.assertEqual(
            _road_width_m({"largeur_de_chaussee": 5, "importance": "1"}),
            (5.0, "source_width"),
        )
        self.assertEqual(
            _road_width_m({"largeur_de_chaussee": None, "importance": "2"}),
            (8.0, "importance"),
        )
        self.assertEqual(
            _road_width_m({"largeur_de_chaussee": None, "importance": "6"}),
            (2.5, "importance"),
        )

    def test_vegetation_exclusion_keeps_buildings_but_removes_their_canopy_cells(
        self,
    ) -> None:
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
        building = PolygonFeature(
            "building:0",
            Polygon([(1.1, 2.1), (1.9, 2.1), (1.9, 2.9), (1.1, 2.9)]),
            {},
        )
        road = LineFeature(
            "road:0",
            LineString([(0.0, 0.5), (4.0, 0.5)]),
            {"largeur_de_chaussee": 1.5},
        )
        water = PolygonFeature(
            "water:0",
            Polygon([(2.1, 3.1), (2.9, 3.1), (2.9, 3.9), (2.1, 3.9)]),
            {},
        )
        mask, statistics = _vegetation_exclusion_mask(
            [building],
            [road],
            [],
            [],
            [water],
            (4, 4),
            transform,
            0.0,
            0.0,
            0.0,
        )
        self.assertTrue(mask[1, 1])
        self.assertTrue(mask[3, 1])
        self.assertTrue(mask[0, 2])
        self.assertGreater(statistics["combined_exclusion_cell_count"], 0)
        self.assertEqual(statistics["building_exclusion_cell_count"], 1)

    def test_road_ribbon_edges_are_individually_draped_on_slope(self) -> None:
        terrain = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
            ]
        )
        feature = LineFeature(
            "road:0",
            __import__("shapely.geometry", fromlist=["LineString"]).LineString(
                [(1.5, 1.5), (2.5, 1.5)]
            ),
            {"largeur_de_chaussee": 2.0},
        )
        mesh, statistics = _road_geometry(
            [feature], terrain, Affine.identity(), (0.0, 0.0, 0.0), 0.25
        )
        self.assertEqual(mesh["faces"], [[0, 1, 3, 2]])
        self.assertEqual(
            mesh["vertices"],
            [
                [1.5, 2.5, 21.25],
                [1.5, 0.5, 1.25],
                [2.5, 2.5, 22.25],
                [2.5, 0.5, 2.25],
            ],
        )
        self.assertEqual(statistics["width_method_counts"], {"source_width": 1})

    def test_water_width_class_and_surface_vertices_follow_mnt(self) -> None:
        self.assertEqual(
            _water_segment_width_m({"classe_de_largeur": "Entre 5 et 15 m"}),
            (10.0, "hydro_width_class"),
        )
        terrain = np.array(
            [
                [100.0, 101.0, 102.0, 103.0],
                [110.0, 111.0, 112.0, 113.0],
                [120.0, 121.0, 122.0, 123.0],
                [130.0, 131.0, 132.0, 133.0],
            ]
        )
        feature = PolygonFeature(
            "water:0",
            Polygon([(0.2, 0.2), (2.2, 0.2), (2.2, 2.2), (0.2, 2.2)]),
            {},
        )
        mesh, statistics = _water_surface_geometry(
            [feature], terrain, Affine.identity(), (0.0, 0.0, 100.0), 0.1
        )
        self.assertEqual(statistics["triangle_count"], 2)
        self.assertEqual(statistics["altitude_method"], "mnt_draped")
        for x, y, z in mesh["vertices"]:
            expected = terrain[int(y), int(x)] - 100.0 + 0.1
            self.assertAlmostEqual(z, expected, places=3)


if __name__ == "__main__":
    unittest.main()
