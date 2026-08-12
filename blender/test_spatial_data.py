from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spatial_data import (
    TARGET_CRS,
    choose_local_origin,
    find_absolute_local_paths,
    infer_geojson_crs,
    lambert93_validation,
    numeric_property,
    positive_numeric_property,
    read_polygon_features,
)


class SpatialDataTests(unittest.TestCase):
    def test_portability_guard_finds_absolute_paths(self) -> None:
        payload = {
            "portable": {"file_name": "mnt.tif"},
            "windows": r"E:\fixtures\fireviewer\mnt.tif",
            "posix": "/tmp/mnt.tif",
        }
        self.assertEqual(
            find_absolute_local_paths(payload),
            ["root.windows", "root.posix"],
        )

    def test_portability_guard_accepts_relative_paths(self) -> None:
        payload = {
            "terrain": {"file_name": "mnt.tif"},
            "vector": {"relative_path": "vectors/buildings.geojson"},
        }
        self.assertEqual(find_absolute_local_paths(payload), [])

    def test_accepts_ign_lambert93_parameters_without_authority(self) -> None:
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
            lambert93_validation(IgnCrs()),
            "lambert93_projection_parameters",
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
                            [
                                [5.30, 44.70],
                                [5.31, 44.70],
                                [5.31, 44.71],
                                [5.30, 44.70],
                            ]
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

    def test_explicit_axis_swap_normalizes_effis_coordinates(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": "SYNTH-001"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [44.70, 5.30],
                                [44.70, 5.31],
                                [44.71, 5.31],
                                [44.70, 5.30],
                            ]
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
        self.assertEqual(numeric_property({"HAUTEUR": "12,5 m"}, ("hauteur",)), 12.5)

    def test_positive_property_uses_priority(self) -> None:
        properties = {"block_height_m": None, "hauteur": 0, "height": "7,25 m"}
        self.assertEqual(
            positive_numeric_property(
                properties,
                ("block_height_m", "hauteur", "height"),
            ),
            7.25,
        )

    def test_origin_is_metre_aligned(self) -> None:
        origin = choose_local_origin(
            (700000.0, 6600000.0, 711000.0, 6613000.0),
            317.8,
        )
        self.assertEqual(origin, (705500, 6606500, 317))


if __name__ == "__main__":
    unittest.main()
