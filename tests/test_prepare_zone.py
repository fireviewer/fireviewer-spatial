from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry


TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from prepare_zone import (  # noqa: E402
    PreparationError,
    build_package,
    inspect_source_rasters,
    load_effis_feature,
)
from verify_package import VerificationError, verify_package  # noqa: E402


def swap_geojson_xy(value):
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(
            isinstance(item, (float, int)) for item in value[:2]
        ):
            return [value[1], value[0], *value[2:]]
        return [swap_geojson_xy(item) for item in value]
    return value


def write_raster(
    path: Path, *, transform, width: int, height: int, value: float
) -> None:
    data = np.full((height, width), value, dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:2154",
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data, 1)


class HybridZonePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.effis = cls.root / "effis.geojson"
        cls.mnt = cls.root / "mnt.tif"
        cls.mns = cls.root / "mns.tif"
        cls.package = cls.root / "package"

        target_wgs84 = box(5.30, 44.70, 5.32, 44.72)
        decoy_wgs84 = box(5.45, 44.80, 5.46, 44.81)
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": "decoy", "AREA_HA": "1"},
                    "geometry": {
                        **mapping(decoy_wgs84),
                        "coordinates": swap_geojson_xy(
                            mapping(decoy_wgs84)["coordinates"]
                        ),
                    },
                },
                {
                    "type": "Feature",
                    "properties": {
                        "id": "SYNTH-001",
                        "FIREDATE": "2026-07-03 00:39:00",
                        "FINALDATE": "2026-07-11 12:55:00",
                        "LASTUPDATE": "2026-07-07 07:05:31",
                        "COMMUNE": "Solaure en Diois",
                        "AREA_HA": "4192",
                        "CLASS": "30DAYS",
                    },
                    "geometry": {
                        **mapping(target_wgs84),
                        "coordinates": swap_geojson_xy(
                            mapping(target_wgs84)["coordinates"]
                        ),
                    },
                },
            ],
        }
        cls.effis.write_text(json.dumps(payload), encoding="utf-8")

        transformer = Transformer.from_crs(4326, 2154, always_xy=True)
        target_l93 = transform_geometry(transformer.transform, target_wgs84)
        west, south, east, north = target_l93.bounds
        resolution = 10.0
        raster_west = west - 2500
        raster_north = north + 2500
        width = int(np.ceil((east - west + 5000) / resolution))
        height = int(np.ceil((north - south + 5000) / resolution))
        transform = from_origin(raster_west, raster_north, resolution, resolution)
        write_raster(
            cls.mnt, transform=transform, width=width, height=height, value=500.0
        )
        write_raster(
            cls.mns, transform=transform, width=width, height=height, value=515.0
        )

        build_package(
            mnt_path=cls.mnt,
            mns_path=cls.mns,
            effis_path=cls.effis,
            output_dir=cls.package,
            package_id="synthetic-synthetic-zone-v1",
            fire_id="SYNTH-001",
            buffer_metres=1500.0,
            axis_order="lat-lon",
            generated_at="2026-07-15T00:00:00Z",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_selects_axis_normalizes_buffers_and_verifies(self) -> None:
        _, normalized = load_effis_feature(
            self.effis, "SYNTH-001", axis_order="lat-lon"
        )
        self.assertEqual(normalized.bounds, (5.3, 44.7, 5.32, 44.72))

        result = verify_package(self.package)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["selected_effis_feature_id"], "SYNTH-001")
        self.assertEqual(result["buffer_metres"], 1500.0)
        self.assertEqual(result["cog"]["layout"], "COG")
        self.assertEqual(result["cog"]["overviews"], [2, 4, 8, 16])

        aoi = json.loads(
            (self.package / "vectors" / "area-of-interest.l93.geojson").read_text(
                encoding="utf-8"
            )
        )
        perimeter = json.loads(
            (self.package / "vectors" / "fire-perimeter.l93.geojson").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            shape(aoi["features"][0]["geometry"]).covers(
                shape(perimeter["features"][0]["geometry"])
            )
        )

        catalog = json.loads(
            (self.package / "catalog.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            catalog["deferred_layers"]["mns"]["status"],
            "validated_source_only_not_published",
        )
        self.assertGreater(catalog["layers"]["terrain_mnt"]["masked_pixel_count"], 0)

    def test_missing_fire_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(PreparationError, "exactement une entité"):
            load_effis_feature(self.effis, "absent", axis_order="lat-lon")

    def test_misaligned_mns_is_rejected(self) -> None:
        misaligned = self.root / "mns-misaligned.tif"
        with rasterio.open(self.mnt) as mnt:
            transform = from_origin(
                mnt.bounds.left + 10,
                mnt.bounds.top,
                abs(mnt.transform.a),
                abs(mnt.transform.e),
            )
            write_raster(
                misaligned,
                transform=transform,
                width=mnt.width,
                height=mnt.height,
                value=515.0,
            )
        with self.assertRaisesRegex(PreparationError, "ne sont pas alignées"):
            inspect_source_rasters(self.mnt, misaligned)

    def test_hash_tampering_is_detected(self) -> None:
        path = self.package / "vectors" / "fire-perimeter.geojson"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(
                VerificationError, "Taille incorrecte|SHA-256 incorrect"
            ):
                verify_package(self.package)
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
