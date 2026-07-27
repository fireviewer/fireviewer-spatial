from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, box, mapping, shape
from shapely.ops import transform as transform_geometry

import prepare_vector_blocks as blocks
import verify_package as package_verifier


L93_TO_WGS84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)


def _wgs84(geometry):
    return mapping(transform_geometry(L93_TO_WGS84.transform, geometry))


def _feature(cleabs: str, geometry, **properties):
    return {
        "type": "Feature",
        "properties": {"cleabs": cleabs, **properties},
        "geometry": _wgs84(geometry),
    }


def _write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _collection(path: Path, features) -> Path:
    return _write_json(path, {"type": "FeatureCollection", "features": list(features)})


def _raster(
    path: Path, values: np.ndarray, *, x: float, top: float, nodata=-9999.0, crs=None
) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(x, top, 1.0, 1.0),
        nodata=nodata,
    ) as dataset:
        dataset.write(values.astype("float32"), 1)
    return path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_clips_polygon_and_multipolygon_and_deduplicates_cleabs(
    tmp_path: Path, capsys
):
    x = 880_000.0
    bottom = 6_400_000.0
    top = bottom + 10.0
    aoi_l93 = box(x + 1, bottom + 1, x + 9, bottom + 9)
    aoi = _collection(
        tmp_path / "aoi.geojson",
        [{"type": "Feature", "properties": {}, "geometry": _wgs84(aoi_l93)}],
    )
    mnt = _raster(tmp_path / "mnt.tif", np.full((10, 10), 100.0), x=x, top=top)
    mns = _raster(tmp_path / "mns.tif", np.full((10, 10), 106.0), x=x, top=top)

    first_building = box(x + 2, bottom + 2, x + 3, bottom + 3)
    multi_building = MultiPolygon(
        [
            box(x + 4, bottom + 4, x + 5, bottom + 5),
            box(x + 6, bottom + 6, x + 7, bottom + 7),
        ]
    )
    crossing_building = box(x + 8, bottom + 8, x + 11, bottom + 11)
    page_one = _collection(
        tmp_path / "buildings-1.geojson",
        [
            _feature(
                "b1",
                first_building,
                altitude_minimale_sol=101.0,
                hauteur=8.0,
                methode_d_acquisition_altimetrique="Photogrammétrie",
                precision_altimetrique=1.5,
            ),
            _feature(
                "b2",
                multi_building,
                altitude_maximale_toit=115.0,
                methode_d_acquisition_altimetrique="Photogrammétrie",
                precision_altimetrique=1.5,
            ),
        ],
    )
    page_two = _collection(
        tmp_path / "buildings-2.geojson",
        [
            _feature("b1", first_building, altitude_minimale_sol=101.0, hauteur=99.0),
            _feature(
                "b3",
                crossing_building,
                altitude_minimale_sol=532.1,
                methode_d_acquisition_altimetrique="Pas de Z",
                precision_altimetrique=9999,
            ),
            _feature("outside", box(x + 20, bottom + 20, x + 21, bottom + 21)),
        ],
    )
    vegetation = _collection(
        tmp_path / "vegetation.geojson",
        [
            _feature(
                "v1", box(x + 2, bottom + 2, x + 5, bottom + 5), nature="Forêt ouverte"
            ),
            _feature(
                "v2",
                MultiPolygon(
                    [
                        box(x + 5, bottom + 2, x + 7, bottom + 4),
                        box(x + 5, bottom + 6, x + 7, bottom + 8),
                    ]
                ),
                nature="Bois",
            ),
        ],
    )
    output = tmp_path / "output"

    exit_code = blocks.main(
        [
            "--aoi",
            str(aoi),
            "--mnt",
            str(mnt),
            "--mns",
            str(mns),
            "--building-pages",
            str(page_one),
            "--building-pages",
            str(page_two),
            "--vegetation",
            str(vegetation),
            "--output-dir",
            str(output),
        ]
    )
    assert exit_code == 0
    assert '"status": "ok"' in capsys.readouterr().out

    buildings = _read(output / "buildings.l93.geojson")
    by_id = {
        feature["properties"]["cleabs"]: feature for feature in buildings["features"]
    }
    assert set(by_id) == {"b1", "b2", "b3"}
    assert by_id["b1"]["properties"]["base_elevation_m"] == 100.0
    assert (
        by_id["b1"]["properties"]["base_method"] == "mnt_representative_point_aligned"
    )
    assert by_id["b1"]["properties"]["source_base_elevation_m"] == 101.0
    assert by_id["b1"]["properties"]["source_base_delta_to_mnt_m"] == 1.0
    assert by_id["b1"]["properties"]["block_height_m"] == 8.0
    assert by_id["b1"]["properties"]["height_method"] == "bdtopo_hauteur"
    assert by_id["b2"]["geometry"]["type"] == "MultiPolygon"
    assert by_id["b2"]["properties"]["base_elevation_m"] == 100.0
    assert by_id["b2"]["properties"]["block_height_m"] == 15.0
    assert (
        by_id["b2"]["properties"]["height_method"]
        == "bdtopo_altitude_maximale_toit_minus_base"
    )
    assert by_id["b3"]["properties"]["block_height_m"] == 6.0
    assert by_id["b3"]["properties"]["base_elevation_m"] == 100.0
    assert (
        by_id["b3"]["properties"]["base_method"] == "mnt_representative_point_aligned"
    )
    assert by_id["b3"]["properties"]["bdtopo_z_quality"] == "rejected_method_without_z"
    assert shape(by_id["b3"]["geometry"]).within(aoi_l93.buffer(1e-5))

    vegetation_result = _read(output / "vegetation.l93.geojson")
    assert {
        feature["geometry"]["type"] for feature in vegetation_result["features"]
    } == {
        "Polygon",
        "MultiPolygon",
    }
    for feature in vegetation_result["features"]:
        properties = feature["properties"]
        assert properties["base_elevation_m"] == 100.0
        assert properties["block_height_m"] == 6.0
        assert properties["height_quality"] == "good"

    manifest = _read(output / "vector-manifest.json")
    assert manifest["crs"] == "EPSG:2154"
    assert manifest["inputs"]["mnt"]["observed_crs"] is None
    assert manifest["inputs"]["mnt"]["assigned_crs"] == "EPSG:2154"
    assert manifest["statistics"]["buildings"]["duplicate_cleabs_count"] == 1
    assert manifest["statistics"]["buildings"]["outside_aoi_count"] == 1
    assert manifest["outputs"]["buildings"]["sha256"] == blocks.sha256_file(
        output / "buildings.l93.geojson"
    )
    assert len(manifest["outputs"]["vegetation"]["sha256"]) == 64


def test_nodata_stays_null_and_negative_surface_difference_is_clamped(tmp_path: Path):
    x = 880_000.0
    bottom = 6_400_000.0
    top = bottom + 6.0
    aoi_l93 = box(x, bottom, x + 6, top)
    aoi = _collection(
        tmp_path / "aoi.geojson",
        [{"type": "Feature", "properties": {}, "geometry": _wgs84(aoi_l93)}],
    )
    ground = np.full((6, 6), 100.0)
    surface = np.full((6, 6), 95.0)
    ground[:, :3] = -9999.0
    surface[:, :3] = -9999.0
    mnt = _raster(tmp_path / "mnt.tif", ground, x=x, top=top)
    mns = _raster(tmp_path / "mns.tif", surface, x=x, top=top)
    buildings = _collection(
        tmp_path / "buildings.geojson",
        [
            _feature("b-nodata", box(x, bottom, x + 2, top)),
            _feature("b-negative", box(x + 3, bottom, x + 6, top)),
        ],
    )
    vegetation = _collection(
        tmp_path / "vegetation.geojson",
        [
            _feature("v-nodata", box(x, bottom, x + 2, top)),
            _feature("v-negative", box(x + 3, bottom, x + 6, top)),
        ],
    )
    output = tmp_path / "output"

    blocks.prepare_vector_blocks(
        aoi_path=aoi,
        mnt_path=mnt,
        mns_path=mns,
        building_pages=[buildings],
        vegetation_path=vegetation,
        output_dir=output,
    )

    building_features = {
        feature["properties"]["cleabs"]: feature["properties"]
        for feature in _read(output / "buildings.l93.geojson")["features"]
    }
    assert building_features["b-nodata"]["base_elevation_m"] is None
    assert building_features["b-nodata"]["block_height_m"] is None
    assert building_features["b-negative"]["base_elevation_m"] == 100.0
    assert building_features["b-negative"]["block_height_m"] is None
    assert building_features["b-negative"]["height_quality"] == "non_positive"

    vegetation_features = {
        feature["properties"]["cleabs"]: feature["properties"]
        for feature in _read(output / "vegetation.l93.geojson")["features"]
    }
    assert vegetation_features["v-nodata"]["base_elevation_m"] is None
    assert vegetation_features["v-nodata"]["block_height_m"] is None
    assert vegetation_features["v-nodata"]["height_quality"] == "insufficient"
    assert vegetation_features["v-negative"]["base_elevation_m"] == 100.0
    assert vegetation_features["v-negative"]["block_height_m"] == 0.0
    assert (
        vegetation_features["v-negative"]["height_method"]
        == "raster_p75_clamped_mns_minus_mnt"
    )


def test_rejects_a_raster_that_is_not_a_plausible_lambert93_grid(tmp_path: Path):
    aoi_l93 = box(880_000, 6_400_000, 880_002, 6_400_002)
    wrong = _raster(tmp_path / "wrong.tif", np.ones((2, 2)), x=5.0, top=45.0)
    with rasterio.open(wrong) as dataset:
        with pytest.raises(ValueError, match="not plausible Lambert-93"):
            blocks.validate_l93_raster(dataset, aoi_l93)


def test_package_attachment_updates_catalog_and_its_manifest_hash(tmp_path: Path):
    x = 880_000.0
    bottom = 6_400_000.0
    aoi_l93 = box(x, bottom, x + 4, bottom + 4)
    package = tmp_path / "package"
    vectors = package / "vectors"
    vectors.mkdir(parents=True)
    aoi = _collection(
        vectors / "area-of-interest.geojson",
        [{"type": "Feature", "properties": {}, "geometry": _wgs84(aoi_l93)}],
    )
    mnt = _raster(tmp_path / "mnt.tif", np.full((4, 4), 100.0), x=x, top=bottom + 4)
    mns = _raster(tmp_path / "mns.tif", np.full((4, 4), 104.0), x=x, top=bottom + 4)
    buildings = _collection(
        tmp_path / "buildings.geojson", [_feature("b1", aoi_l93, hauteur=4.0)]
    )
    vegetation = _collection(tmp_path / "vegetation.geojson", [_feature("v1", aoi_l93)])
    _write_json(
        package / "catalog.json",
        {
            "package_id": "synthetic",
            "layers": {},
            "deferred_layers": {
                "buildings": {"status": "not_processed"},
                "vegetation_blocks": {"status": "not_processed"},
            },
            "limitations": [
                "No BD TOPO building, vegetation or hedge data is included in this revision."
            ],
        },
    )
    _write_json(
        package / "package-manifest.json",
        {
            "package_id": "synthetic",
            "catalog": {},
            "processing": {},
        },
    )

    manifest_path = blocks.prepare_vector_blocks(
        aoi_path=aoi,
        mnt_path=mnt,
        mns_path=mns,
        building_pages=[buildings],
        vegetation_path=vegetation,
        output_dir=vectors,
        package_root=package,
    )

    vector_manifest = _read(manifest_path)
    assert "path" not in vector_manifest["inputs"]["mnt"]
    assert vector_manifest["inputs"]["mnt"]["file_name"] == "mnt.tif"
    catalog = _read(package / "catalog.json")
    assert catalog["deferred_layers"]["buildings"]["status"] == "produced"
    assert catalog["deferred_layers"]["vegetation_blocks"]["status"] == "produced"
    assert catalog["layers"]["buildings_l93"]["path"] == "vectors/buildings.l93.geojson"
    assert (
        catalog["layers"]["vector_model_manifest"]["path"]
        == "vectors/vector-manifest.json"
    )
    package_manifest = _read(package / "package-manifest.json")
    assert (
        package_manifest["catalog"]["byte_count"]
        == (package / "catalog.json").stat().st_size
    )
    assert package_manifest["catalog"]["sha256"] == blocks.sha256_file(
        package / "catalog.json"
    )
    extension = package_verifier._verify_vector_extension(
        package,
        catalog,
        {
            layer_id: package / record["path"]
            for layer_id, record in catalog["layers"].items()
        },
        aoi_l93,
    )
    assert extension["buildings"]["feature_count"] == 1
    assert extension["vegetation"]["feature_count"] == 1
