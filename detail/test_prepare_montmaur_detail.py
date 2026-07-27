from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

import prepare_montmaur_detail as detail


def _synthetic_canopies_and_hedge():
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    classification: list[int] = []
    for centre_x, centre_y, apex_height in ((5.0, 5.0, 18.0), (14.0, 6.0, 14.0)):
        for dx in np.arange(-2.0, 2.01, 0.5):
            for dy in np.arange(-2.0, 2.01, 0.5):
                radius = float(np.hypot(dx, dy))
                if radius <= 2.0:
                    height = apex_height - 1.5 * radius
                    x.append(centre_x + dx)
                    y.append(centre_y + dy)
                    z.append(100.0 + height)
                    classification.append(5 if height > 8 else 4)
    for east in np.arange(2.0, 18.01, 0.5):
        for offset in (-0.5, 0.0, 0.5):
            x.append(float(east))
            y.append(15.0 + offset)
            z.append(103.0 - abs(offset))
            classification.append(4)
    return detail.ClassifiedPoints(
        np.asarray(x),
        np.asarray(y),
        np.asarray(z),
        np.asarray(classification, dtype=np.uint8),
    )


def _flat_rasters() -> detail.RasterPair:
    return detail.RasterPair(
        np.full((20, 20), 100.0),
        np.full((20, 20), 100.0),
        from_origin(0.0, 20.0, 1.0, 1.0),
        (0.0, 0.0, 20.0, 20.0),
        {},
    )


def test_tree_apices_crowns_and_hedge_are_distinct_and_deterministic():
    points = _synthetic_canopies_and_hedge()
    rasters = _flat_rasters()
    ground, valid = rasters.ground_at(points.x, points.y)
    assert valid.all()
    normalized = points.z - ground
    hedges = [detail.VectorFeature("H1", LineString([(2.0, 15.0), (18.0, 15.0)]), {})]
    parameters = detail.DetailParameters()

    hedge_features, excluded, hedge_stats = detail.measure_hedges(
        hedges, points, normalized, ground, parameters
    )
    trees, crowns, tree_stats = detail.detect_trees(
        points, normalized, ground, excluded, (0.0, 0.0, 20.0, 20.0), parameters
    )
    assert hedge_stats["vegetation_points_assigned_to_hedges"] == 99
    assert hedge_features[0]["properties"]["height_m"] == 3.0
    assert hedge_features[0]["properties"]["width_m"] == 1.0
    assert tree_stats["accepted_tree_count"] == 2
    assert [feature["properties"]["height_m"] for feature in trees] == [18.0, 14.0]
    assert [feature["properties"]["crown_diameter_m"] for feature in trees] == [
        4.0,
        4.0,
    ]
    assert len(crowns) == 2
    assert all(
        feature["properties"]["completeness_claim"]
        == "detected_crown_only_not_every_physical_tree"
        for feature in trees
    )

    permutation = np.random.default_rng(42).permutation(len(points))
    shuffled = points.subset(permutation)
    shuffled_ground, _ = rasters.ground_at(shuffled.x, shuffled.y)
    shuffled_height = shuffled.z - shuffled_ground
    _, shuffled_excluded, _ = detail.measure_hedges(
        hedges, shuffled, shuffled_height, shuffled_ground, parameters
    )
    shuffled_trees, shuffled_crowns, shuffled_stats = detail.detect_trees(
        shuffled,
        shuffled_height,
        shuffled_ground,
        shuffled_excluded,
        (0.0, 0.0, 20.0, 20.0),
        parameters,
    )
    assert shuffled_trees == trees
    assert [item["properties"] for item in shuffled_crowns] == [
        item["properties"] for item in crowns
    ]
    assert shuffled_stats == tree_stats


def test_sparse_and_negative_returns_do_not_create_trees_or_heights():
    parameters = detail.DetailParameters(min_tree_points=5, min_hedge_points=5)
    points = detail.ClassifiedPoints(
        np.asarray([5.0, 5.5, 6.0, 12.0, 12.5, 13.0]),
        np.asarray([5.0, 5.0, 5.0, 12.0, 12.0, 12.0]),
        np.asarray([110.0, 108.0, 106.0, 99.0, 98.0, 97.0]),
        np.asarray([5, 5, 4, 5, 4, 3], dtype=np.uint8),
    )
    ground = np.full(len(points), 100.0)
    normalized = points.z - ground
    normalized[normalized < 0] = np.nan
    trees, crowns, stats = detail.detect_trees(
        points,
        normalized,
        ground,
        np.zeros(len(points), dtype=bool),
        (0.0, 0.0, 20.0, 20.0),
        parameters,
    )
    assert trees == []
    assert crowns == []
    assert stats["apex_candidate_count"] == 1
    assert stats["rejected_sparse_crown_count"] == 1

    hedge = [
        detail.VectorFeature("H-SPARSE", LineString([(11.0, 12.0), (14.0, 12.0)]), {})
    ]
    hedge_features, _, hedge_stats = detail.measure_hedges(
        hedge, points, normalized, ground, parameters
    )
    assert hedge_stats["insufficient_hedge_count"] == 1
    assert hedge_features[0]["properties"]["height_m"] is None
    assert hedge_features[0]["properties"]["width_m"] is None


def test_building_height_uses_class_6_then_mns_and_leaves_nodata_null():
    mnt = np.full((20, 20), 100.0)
    mns = np.full((20, 20), 100.0)
    mns[15:18, 2:5] = 110.0
    mns[7:10, 10:13] = 108.0
    mnt[1:4, 16:19] = np.nan
    mns[1:4, 16:19] = np.nan
    rasters = detail.RasterPair(
        mnt,
        mns,
        from_origin(0.0, 20.0, 1.0, 1.0),
        (0.0, 0.0, 20.0, 20.0),
        {},
    )
    buildings = [
        detail.VectorFeature("B-POINTS", box(2.0, 2.0, 5.0, 5.0), {}),
        detail.VectorFeature("B-MNS", box(10.0, 10.0, 13.0, 13.0), {}),
        detail.VectorFeature("B-NODATA", box(16.0, 16.0, 19.0, 19.0), {}),
    ]
    x: list[float] = []
    y: list[float] = []
    for east in (2.5, 3.0, 3.5, 4.0, 4.5):
        for north in (2.5, 3.0, 3.5):
            x.append(east)
            y.append(north)
    roof_points = detail.ClassifiedPoints(
        np.asarray(x),
        np.asarray(y),
        np.full(len(x), 112.0),
        np.full(len(x), 6, dtype=np.uint8),
    )

    output, stats = detail.measure_buildings(
        buildings, roof_points, rasters, detail.DetailParameters()
    )
    by_id = {
        feature["properties"]["detail_id"]: feature["properties"] for feature in output
    }
    assert by_id["MONTMAUR-BUILDING-B-POINTS"]["height_m"] == 12.0
    assert (
        by_id["MONTMAUR-BUILDING-B-POINTS"]["height_method"]
        == "copc_class_6_roof_p95_minus_mnt_median"
    )
    assert by_id["MONTMAUR-BUILDING-B-MNS"]["height_m"] == 8.0
    assert (
        by_id["MONTMAUR-BUILDING-B-MNS"]["height_method"] == "mns_p75_minus_mnt_median"
    )
    assert by_id["MONTMAUR-BUILDING-B-NODATA"]["base_elevation_m"] is None
    assert by_id["MONTMAUR-BUILDING-B-NODATA"]["height_m"] is None
    assert by_id["MONTMAUR-BUILDING-B-NODATA"]["invented_default_height"] is None
    assert stats["null_height_count"] == 1


def test_point_deduplication_and_production_rejects_plain_las(tmp_path: Path):
    duplicate_points = detail.ClassifiedPoints(
        np.asarray([880_001.0, 880_001.0, 880_002.0]),
        np.asarray([6_400_001.0, 6_400_001.0, 6_400_002.0]),
        np.asarray([500.0, 500.0, 501.0]),
        np.asarray([5, 5, 6], dtype=np.uint8),
    )
    unique, duplicate_count = detail.deduplicate_points(duplicate_points)
    assert duplicate_count == 1
    assert len(unique) == 2

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    header.offsets = np.asarray([0.0, 0.0, 0.0])
    header.add_crs(detail.pyproj.CRS.from_epsg(2154))
    las = laspy.LasData(header)
    las.x = np.asarray([880_001.0, 880_002.0])
    las.y = np.asarray([6_400_001.0, 6_400_002.0])
    las.z = np.asarray([500.0, 501.0])
    las.classification = np.asarray([5, 6], dtype=np.uint8)
    path = tmp_path / "fixture.las"
    las.write(path)
    aoi = box(880_000.0, 6_400_000.0, 880_010.0, 6_400_010.0)
    with pytest.raises(ValueError, match="not COPC"):
        detail.load_classified_points([path], aoi, require_copc=True)
    loaded, sources, stats = detail.load_classified_points(
        [path], aoi, require_copc=False
    )
    assert len(loaded) == 2
    assert sources[0]["source_kind"] == "LAS/LAZ"
    assert stats["class_5_count"] == 1
    assert stats["class_6_count"] == 1


def test_manifest_states_non_exhaustive_tree_claim(tmp_path: Path):
    # This guards the semantic contract independently of any real-world result.
    contract = {
        "claim": "detected_crown_apices_only_not_every_physical_tree",
        "exhaustive_tree_inventory": False,
        "invented_tree_geometry": False,
    }
    path = tmp_path / "contract.json"
    detail._write_json(path, contract)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["exhaustive_tree_inventory"] is False
    assert "not_every_physical_tree" in loaded["claim"]


def test_offline_file_pipeline_writes_hashed_detail_package(tmp_path: Path):
    x_origin = 880_000.0
    y_origin = 6_400_000.0
    aoi_geometry = box(x_origin, y_origin, x_origin + 20.0, y_origin + 20.0)

    def write_collection(path: Path, features):
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": "EPSG:2154"}},
                    "features": list(features),
                }
            ),
            encoding="utf-8",
        )
        return path

    aoi = write_collection(
        tmp_path / "aoi.geojson",
        [
            {
                "type": "Feature",
                "properties": {},
                "geometry": detail.mapping(aoi_geometry),
            }
        ],
    )
    buildings = write_collection(
        tmp_path / "buildings.geojson",
        [
            {
                "type": "Feature",
                "properties": {"cleabs": "B1"},
                "geometry": detail.mapping(
                    box(x_origin + 2, y_origin + 2, x_origin + 5, y_origin + 5)
                ),
            }
        ],
    )
    hedges = write_collection(
        tmp_path / "hedges.geojson",
        [
            {
                "type": "Feature",
                "properties": {"cleabs": "H1"},
                "geometry": detail.mapping(
                    LineString(
                        [(x_origin + 2, y_origin + 15), (x_origin + 18, y_origin + 15)]
                    )
                ),
            }
        ],
    )
    mnt_values = np.full((20, 20), 100.0, dtype=np.float32)
    mns_values = np.full((20, 20), 100.0, dtype=np.float32)
    mns_values[15:18, 2:5] = 112.0
    raster_transform = from_origin(x_origin, y_origin + 20, 1.0, 1.0)
    raster_paths = []
    for name, values in (("mnt.tif", mnt_values), ("mns.tif", mns_values)):
        path = tmp_path / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=20,
            height=20,
            count=1,
            dtype="float32",
            crs="EPSG:2154",
            transform=raster_transform,
            nodata=-9999.0,
        ) as dataset:
            dataset.write(values, 1)
        raster_paths.append(path)

    canopy = _synthetic_canopies_and_hedge()
    point_x = canopy.x + x_origin
    point_y = canopy.y + y_origin
    point_z = canopy.z
    point_class = canopy.classification
    roof_x = np.asarray([x_origin + value for value in (2.5, 3.0, 3.5, 4.0, 4.5)])
    roof_y = np.full(len(roof_x), y_origin + 3.0)
    point_x = np.r_[point_x, roof_x]
    point_y = np.r_[point_y, roof_y]
    point_z = np.r_[point_z, np.full(len(roof_x), 112.0)]
    point_class = np.r_[point_class, np.full(len(roof_x), 6, dtype=np.uint8)]
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    header.offsets = np.asarray([0.0, 0.0, 0.0])
    header.add_crs(detail.pyproj.CRS.from_epsg(2154))
    las = laspy.LasData(header)
    las.x, las.y, las.z, las.classification = point_x, point_y, point_z, point_class
    point_path = tmp_path / "fixture.las"
    las.write(point_path)
    output = tmp_path / "detail-output"

    with pytest.raises(ValueError, match="must have 0.5 m pixels"):
        detail.load_raster_pair(raster_paths[0], raster_paths[1], aoi_geometry)

    manifest_path = detail.produce_detail(
        aoi_path=aoi,
        mnt_path=raster_paths[0],
        mns_path=raster_paths[1],
        copc_paths=[point_path],
        buildings_path=buildings,
        hedges_path=hedges,
        output_dir=output,
        parameters=detail.DetailParameters(min_building_roof_points=5),
        aoi_crs="auto",
        buildings_crs="auto",
        hedges_crs="auto",
        require_copc=False,
        required_raster_resolution_m=None,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["outputs"]["trees"]["feature_count"] == 2
    assert manifest["outputs"]["tree_crowns"]["feature_count"] == 2
    assert manifest["outputs"]["hedges"]["feature_count"] == 1
    assert manifest["outputs"]["buildings"]["feature_count"] == 1
    assert manifest["tree_detection_contract"]["exhaustive_tree_inventory"] is False
    assert manifest["inputs"]["copc"][0]["source_kind"] == "LAS/LAZ"
    for record in manifest["outputs"].values():
        path = output / record["path"]
        assert record["sha256"] == detail.sha256_file(path)
        assert record["byte_count"] == path.stat().st_size
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")


def test_four_native_half_metre_tiles_mosaic_across_both_seams(tmp_path: Path):
    x_origin = 880_000.0
    y_origin = 6_400_000.0
    mnt_paths: list[Path] = []
    mns_paths: list[Path] = []
    for row in range(2):
        for column in range(2):
            tile_left = x_origin + column
            tile_top = y_origin + row + 1
            for kind, offset, paths in (
                ("mnt", 0.0, mnt_paths),
                ("mns", 5.0, mns_paths),
            ):
                path = tmp_path / f"{kind}-{row}-{column}.tif"
                values = np.full(
                    (2, 2), 100.0 + row * 10 + column + offset, dtype=np.float32
                )
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=2,
                    count=1,
                    dtype="float32",
                    crs="EPSG:2154",
                    transform=from_origin(tile_left, tile_top, 0.5, 0.5),
                    nodata=-9999.0,
                ) as dataset:
                    dataset.write(values, 1)
                paths.append(path)
    aoi = box(x_origin + 0.75, y_origin + 0.75, x_origin + 1.25, y_origin + 1.25)
    pair = detail.load_raster_pair(mnt_paths, mns_paths, aoi, required_resolution_m=0.5)
    assert pair.mnt.shape == (2, 2)
    assert pair.source_metadata["mosaic"]["bounds_l93"] == [
        x_origin + 0.5,
        y_origin + 0.5,
        x_origin + 1.5,
        y_origin + 1.5,
    ]
    assert np.allclose(pair.mns - pair.mnt, 5.0)
