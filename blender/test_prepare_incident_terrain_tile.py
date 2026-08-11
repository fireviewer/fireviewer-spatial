from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from prepare_incident_terrain_tile import (
    build_terrain_outputs,
    validate_terrain_evidence,
    validate_terrain_package,
    write_ground_material_map,
    write_terrain_package,
)


def _write_raster(path: Path, values: np.ndarray, transform: Affine) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:2154",
        transform=transform,
        nodata=np.nan,
    ) as dataset:
        dataset.write(values.astype("float32"), 1)


def test_builds_bare_terrain_and_lightweight_2d_ground_map(tmp_path: Path) -> None:
    transform = Affine.translation(-1.0, 6.0) * Affine.scale(0.5, -0.5)
    rows, columns = np.indices((14, 14))
    mnt = 100.0 + rows * 0.2 + columns * 0.1
    mns = mnt + np.where((rows > 5) & (columns > 5), 6.0, 0.0)
    mnt_path = tmp_path / "mnt" / "mnt.tif"
    mns_path = tmp_path / "mns" / "mns.tif"
    mnt_path.parent.mkdir()
    mns_path.parent.mkdir()
    _write_raster(mnt_path, mnt, transform)
    _write_raster(mns_path, mns, transform)
    args = argparse.Namespace(
        mnt=[mnt_path],
        mns=[mns_path],
        bounds=[0.0, 0.0, 5.0, 5.0],
        processing_bounds=[-0.5, -0.5, 5.5, 5.5],
        terrain_step_pixels=2,
        ground_material_map_size=32,
        origin_x=0.0,
        origin_y=0.0,
        origin_z=100.0,
    )

    package, ground_map = build_terrain_outputs(args)

    assert package["metadata"]["authored_layers"] == ["bare_terrain_3d"]
    assert package["metadata"]["ground_2d"]["orthophoto_dependency"] == "forbidden"
    assert package["statistics"]["vegetation_instance_count"] == 0
    assert ground_map.shape == (32, 32, 4)
    assert ground_map.dtype == np.uint8
    assert int(ground_map[:, :, 3].max()) > 0

    output = tmp_path / "terrain-0m50.json.gz"
    ground_output = tmp_path / "ground-material-map.png"
    write_terrain_package(package, output)
    write_ground_material_map(ground_map, ground_output)
    tile = {
        "bounds_l93_m": [0.0, 0.0, 5.0, 5.0],
        "processing_bounds_l93_m": [-0.5, -0.5, 5.5, 5.5],
        "origin_l93_m": [0.0, 0.0, 100.0],
    }
    assert validate_terrain_package(output, tile)["schema"].endswith(".v1")
    evidence = validate_terrain_evidence(output, tile, tmp_path)
    assert evidence["source_hashes_match"] is True
    assert evidence["mesh_reproducible_from_mnt"] is True
    assert evidence["terrain_context_map_reproducible_from_mnt_mns"] is True
    assert evidence["contextual_surface_mapping_status"] == "not_authored"

    from PIL import Image
    with Image.open(ground_output) as image:
        altered = np.asarray(image.convert("RGBA"), dtype="uint8").copy()
    altered[0, 0, 0] ^= 1
    Image.fromarray(altered, mode="RGBA").save(ground_output)
    with pytest.raises(ValueError, match="not reproducible"):
        validate_terrain_evidence(output, tile, tmp_path)


def test_rejects_source_raster_in_wrong_crs(tmp_path: Path) -> None:
    transform = Affine.translation(-1.0, 6.0) * Affine.scale(0.5, -0.5)
    values = np.ones((14, 14), dtype="float32")
    mnt_path = tmp_path / "mnt.tif"
    mns_path = tmp_path / "mns.tif"
    _write_raster(mnt_path, values, transform)
    with rasterio.open(
        mns_path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=np.nan,
    ) as dataset:
        dataset.write(values, 1)
    args = argparse.Namespace(
        mnt=[mnt_path],
        mns=[mns_path],
        bounds=[0.0, 0.0, 5.0, 5.0],
        processing_bounds=[-0.5, -0.5, 5.5, 5.5],
        terrain_step_pixels=2,
        ground_material_map_size=32,
        origin_x=0.0,
        origin_y=0.0,
        origin_z=0.0,
    )

    with pytest.raises(ValueError, match="EPSG:2154"):
        build_terrain_outputs(args)
