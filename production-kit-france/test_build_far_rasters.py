from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from build_far_rasters import SCHEMA, build_far_rasters  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geojson(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [700000.0, 6600000.0],
                        [700064.0, 6600000.0],
                        [700064.0, 6600064.0],
                        [700000.0, 6600064.0],
                        [700000.0, 6600000.0],
                    ]
                ],
            }
        ),
        encoding="utf-8",
    )


def _raster(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((64, 64), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="float32",
        crs="EPSG:2154",
        transform=from_origin(700000.0, 6600064.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as target:
        target.write(data, 1)


def test_far_rasters_are_cog_mosaics_and_resume_is_input_locked(tmp_path: Path) -> None:
    source_root = tmp_path / "global-05m"
    mnt = source_root / "sources/mnt/tile.tif"
    mns = source_root / "sources/mns/tile.tif"
    _raster(mnt, 125.0)
    _raster(mns, 137.0)
    manifest = source_root / "production-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "fireviewer.global-05m-production-manifest.v1",
                "status": "ready",
                "source_tiles": [
                    {
                        "id": "tile",
                        "assets": {
                            "mnt": {
                                "path": str(mnt.relative_to(source_root)),
                                "byte_count": mnt.stat().st_size,
                                "sha256": _sha256(mnt),
                            },
                            "mns": {
                                "path": str(mns.relative_to(source_root)),
                                "byte_count": mns.stat().st_size,
                                "sha256": _sha256(mns),
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    aoi = tmp_path / "aoi.geojson"
    _geojson(aoi)

    output = tmp_path / "far"
    first = build_far_rasters(manifest, aoi, output, resolution_m=2.0)
    second = build_far_rasters(manifest, aoi, output, resolution_m=2.0)

    assert first == second
    assert first["schema"] == SCHEMA
    for product, expected in (("mnt", 125.0), ("mns", 137.0)):
        path = output / first["outputs"][product]["path"]
        with rasterio.open(path) as dataset:
            assert dataset.driver == "GTiff"
            assert dataset.crs.to_epsg() == 2154
            assert dataset.res == pytest.approx((2.0, 2.0))
            values = dataset.read(1, masked=True)
            assert float(values.mean()) == pytest.approx(expected)

    aoi.write_text(aoi.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AOI differs"):
        build_far_rasters(manifest, aoi, output, resolution_m=2.0)
