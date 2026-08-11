from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from affine import Affine
import numpy as np
from PIL import Image
import pytest
from rasterio.io import MemoryFile

import prepare_simple_measured_tile_sources as sources
import prepare_simple_measured_zone_context as zone_context


ORIGIN = (819500, 6312500)
BOUNDS = (819490, 6312490, 820010, 6313010)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _tiff_bytes(value: float) -> bytes:
    values = np.full(
        (sources.ELEVATION_SIZE, sources.ELEVATION_SIZE), value, dtype="float32"
    )
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=sources.ELEVATION_SIZE,
            height=sources.ELEVATION_SIZE,
            count=1,
            dtype="float32",
            crs=sources.CRS,
            transform=Affine(0.5, 0, BOUNDS[0], 0, -0.5, BOUNDS[3]),
        ) as dataset:
            dataset.write(values, 1)
        return memory.read()


def _png_bytes() -> bytes:
    image = Image.new("RGB", (sources.ORTHOPHOTO_SIZE, sources.ORTHOPHOTO_SIZE))
    pixels = np.asarray(image).copy()
    pixels[:, :, 0] = np.arange(sources.ORTHOPHOTO_SIZE, dtype="uint16") % 256
    pixels[:, :, 1] = np.arange(sources.ORTHOPHOTO_SIZE, dtype="uint16")[:, None] % 256
    image = Image.fromarray(pixels, mode="RGB")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    gpkg = tmp_path / "ground-context.gpkg"
    manifest = tmp_path / "ground-context-manifest.json"
    snapshot = tmp_path / "buildings.geojson"
    snapshot_receipt = tmp_path / "buildings-receipt.json"
    gpkg.write_bytes(b"synthetic-gpkg")
    manifest.write_bytes(b"{}\n")
    snapshot.write_bytes(
        _canonical_bytes({"type": "FeatureCollection", "features": []})
    )
    snapshot_receipt.write_bytes(
        _canonical_bytes(
            {
                "schema": "fireviewer.building-confirmation-source.v1",
                "role": "semantic_confirmation_only",
                "placement_measurement": "MNS-MNT",
                "response": {
                    "byte_count": snapshot.stat().st_size,
                    "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                },
            }
        )
    )
    return {
        "ground_context_gpkg": gpkg,
        "ground_context_manifest": manifest,
        "building_snapshot": snapshot,
        "building_snapshot_receipt": snapshot_receipt,
    }


def test_wms_request_is_exact_and_lambert93() -> None:
    url = sources.wms_url(sources.MNT_LAYER, BOUNDS, 1040, "image/tiff")
    parsed = urlparse(url)
    values = parse_qs(parsed.query, keep_blank_values=True)
    assert parsed.scheme == "https"
    assert values == {
        "SERVICE": ["WMS"],
        "VERSION": ["1.3.0"],
        "REQUEST": ["GetMap"],
        "LAYERS": [sources.MNT_LAYER],
        "STYLES": [""],
        "CRS": ["EPSG:2154"],
        "BBOX": ["819490,6312490,820010,6313010"],
        "WIDTH": ["1040"],
        "HEIGHT": ["1040"],
        "FORMAT": ["image/tiff"],
        "TRANSPARENT": ["FALSE"],
    }


def test_building_snapshot_is_filtered_to_processing_window() -> None:
    snapshot = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "inside",
                "properties": {"cleabs": "BATIMENT-inside"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [819600, 6312600],
                            [819610, 6312600],
                            [819610, 6312610],
                            [819600, 6312600],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "id": "outside",
                "properties": {"cleabs": "BATIMENT-outside"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [821000, 6314000],
                            [821010, 6314000],
                            [821010, 6314010],
                            [821000, 6314000],
                        ]
                    ],
                },
            },
        ],
    }
    features, footprints = sources._building_footprints(snapshot, BOUNDS)
    assert [feature["id"] for feature in features] == ["inside"]
    assert [footprint["source_id"] for footprint in footprints] == ["BATIMENT-inside"]
    assert footprints[0]["properties"]["cleabs"] == "BATIMENT-inside"


def test_source_bundle_publishes_all_files_at_once_and_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "x819500_y6312500"
    mnt = _tiff_bytes(130.0)
    mns = _tiff_bytes(142.0)
    ortho = _png_bytes()
    calls: list[str] = []
    fixed_assets = [
        {
            "schema": "fireviewer.projected-fixed-asset-placement.v1",
            "placement_id": "church-main",
            "asset_id": "church_village_01",
            "asset_category": "building",
            "source_wgs84": [43.9, 4.5],
            "position_l93_m": [819_600.0, 6_312_600.0],
            "owner_tile_origin_l93_m": list(ORIGIN),
            "yaw_rad": 0.0,
        }
    ]

    def fake_get(url: str) -> bytes:
        calls.append(url)
        if sources.MNT_LAYER in url:
            return mnt
        if sources.MNS_LAYER in url:
            return mns
        return ortho

    monkeypatch.setattr(
        sources,
        "_placement_context",
        lambda **kwargs: {
            "schema": "fireviewer.placement-context-input.v1",
            "crs": sources.CRS,
            "tile_origin_l93_m": list(ORIGIN),
            "processing_bounds_l93_m": list(BOUNDS),
            "building_footprints": kwargs["footprints"],
            "context_geometries": {
                "vegetation": [],
                "roads": [],
                "rail": [],
                "water": [],
            },
        },
    )
    prepared = sources.prepare_sources(
        output_root=output,
        zone_id="FR-30-00001",
        tile_id="x819500_y6312500",
        tile_origin_l93_m=ORIGIN,
        elevation_revision="elevation-r1",
        orthophoto_revision="orthophoto-r1",
        fixed_asset_placements=fixed_assets,
        http_get=fake_get,
        **inputs,
    )
    assert prepared.reused is False
    assert len(calls) == 3
    assert not output.with_name(f".{output.name}.simple-sources.part").exists()
    bundle = json.loads(
        (output / "simple-measured-tile-sources.v1.json").read_text(encoding="utf-8")
    )
    assert set(bundle["files"]) == {
        "mnt-05m.tif",
        "mns-05m.tif",
        "orthophoto-1m.png",
        "elevation-source-05m.json",
        "orthophoto-source.json",
        "bdtopo-buildings.geojson",
        "building-source.json",
        "placement-context.json",
    }
    for name, record in bundle["files"].items():
        path = output / name
        assert path.stat().st_size == record["byte_count"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    context = json.loads(
        (output / "placement-context.json").read_text(encoding="utf-8")
    )
    assert context["fixed_asset_placements"] == fixed_assets
    assert context["provenance"]["fixed_asset_placement_count"] == 1

    reused = sources.prepare_sources(
        output_root=output,
        zone_id="FR-30-00001",
        tile_id="x819500_y6312500",
        tile_origin_l93_m=ORIGIN,
        elevation_revision="elevation-r1",
        orthophoto_revision="orthophoto-r1",
        fixed_asset_placements=fixed_assets,
        http_get=lambda _url: pytest.fail("reused bundle must not download"),
        **inputs,
    )
    assert reused.reused is True
    with pytest.raises(
        sources.SimpleMeasuredTileSourceError,
        match="fixed asset placements differ",
    ):
        sources.prepare_sources(
            output_root=output,
            zone_id="FR-30-00001",
            tile_id="x819500_y6312500",
            tile_origin_l93_m=ORIGIN,
            elevation_revision="elevation-r1",
            orthophoto_revision="orthophoto-r1",
            http_get=lambda _url: pytest.fail("mismatch must fail before download"),
            **inputs,
        )


def test_source_bundle_accepts_one_hash_locked_zone_context(tmp_path: Path) -> None:
    output = tmp_path / "zone-source"
    zone_path = tmp_path / "zone-context.json"

    def zone_get(url: str) -> bytes:
        values = parse_qs(urlparse(url).query)
        layer = values["TYPENAMES"][0]
        role = next(
            role for role, typename in zone_context.LAYERS.items() if typename == layer
        )
        if role == "buildings":
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [819600, 6312600],
                        [819610, 6312600],
                        [819610, 6312610],
                        [819600, 6312600],
                    ]
                ],
            }
        elif role == "vegetation":
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [819550, 6312550],
                        [819700, 6312550],
                        [819700, 6312700],
                        [819550, 6312550],
                    ]
                ],
            }
        else:
            geometry = {
                "type": "LineString",
                "coordinates": [[819550, 6312550], [819700, 6312700]],
            }
        feature = {
            "type": "Feature",
            "id": f"{role}-1",
            "properties": {
                "cleabs": f"{role}-1",
                "nature": "Route empierrée",
                "importance": "5",
                "largeur_de_chaussee": None,
            },
            "geometry": geometry,
        }
        return json.dumps(
            {
                "type": "FeatureCollection",
                "numberMatched": 1,
                "numberReturned": 1,
                "features": [feature],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    zone_context.prepare_zone_context(
        output_path=zone_path,
        zone_id="gps-test-zone",
        bounds_l93_m=(819490, 6312490, 820010, 6313010),
        source_revision="bdtopo-v3-test",
        http_get=zone_get,
    )
    mnt = _tiff_bytes(130.0)
    mns = _tiff_bytes(142.0)
    ortho = _png_bytes()

    def fake_get(url: str) -> bytes:
        if sources.MNT_LAYER in url:
            return mnt
        if sources.MNS_LAYER in url:
            return mns
        return ortho

    prepared = sources.prepare_sources(
        output_root=output,
        zone_id="gps-test-zone",
        tile_id="x819500_y6312500",
        tile_origin_l93_m=ORIGIN,
        elevation_revision="elevation-r1",
        orthophoto_revision="orthophoto-r1",
        zone_context=zone_path,
        http_get=fake_get,
    )
    placement = json.loads(prepared.placement_context.read_text(encoding="utf-8"))
    assert placement["provenance"]["feature_counts"] == {
        "buildings": 1,
        "vegetation": 1,
        "roads": 1,
        "rail": 1,
        "water": 2,
    }
    assert placement["provenance"]["zone_context_content_sha256"]
    assert {
        role: len(records) for role, records in placement["context_features"].items()
    } == {
        "vegetation": 1,
        "roads": 1,
        "rail": 1,
        "hydro_lines": 1,
        "hydro_surfaces": 1,
    }
    assert placement["context_features"]["roads"][0]["source_id"] == "roads-1"
    assert placement["building_footprints"][0]["properties"]["nature"] == (
        "Route empierrée"
    )
    building_source = json.loads(
        (output / "building-source.json").read_text(encoding="utf-8")
    )
    assert building_source["source_zone_context_sha256"]
