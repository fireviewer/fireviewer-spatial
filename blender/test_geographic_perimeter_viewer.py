from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
from PIL import Image
from pyproj import Transformer
import pytest

from fixed_terrain_grid import (
    compile_fixed_terrain_from_canonical_mm,
    encode_fixed_terrain,
)
import geographic_perimeter_layer as layer
import geographic_perimeter_viewer as viewer


TILE_ORIGIN = (820000, 6312500)


def _wgs_polygon(
    west: float, south: float, east: float, north: float
) -> dict[str, object]:
    transformer = Transformer.from_crs(2154, 4326, always_xy=True)
    points = [
        transformer.transform(west, south),
        transformer.transform(east, south),
        transformer.transform(east, north),
        transformer.transform(west, north),
        transformer.transform(west, south),
    ]
    return {"type": "Polygon", "coordinates": [[list(point) for point in points]]}


def _dataset(*, outside: bool = False) -> dict[str, object]:
    offset = 1000 if outside else 0
    affected = _wgs_polygon(
        TILE_ORIGIN[0] + 100 + offset,
        TILE_ORIGIN[1] + 100,
        TILE_ORIGIN[0] + 300 + offset,
        TILE_ORIGIN[1] + 300,
    )
    active = _wgs_polygon(
        TILE_ORIGIN[0] + 220 + offset,
        TILE_ORIGIN[1] + 180,
        TILE_ORIGIN[0] + 340 + offset,
        TILE_ORIGIN[1] + 340,
    )
    return {
        "id": "viewer-fire",
        "name": "Viewer fire",
        "timeline": [
            {
                "timestamp": "2026-08-10T00:00:00Z",
                "time_window": {
                    "start": "2026-08-09T18:00:00Z",
                    "end": "2026-08-10T00:00:00Z",
                },
                "affected": affected,
                "active": active,
            },
            {
                "timestamp": "2026-08-11T00:00:00Z",
                "affected": affected,
            },
        ],
    }


def _png() -> bytes:
    image = Image.new("RGB", (250, 250), (96, 121, 72))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _map_zip(path: Path, *, unsafe: bool = False) -> Path:
    terrain = compile_fixed_terrain_from_canonical_mm(
        np.full((253, 253), 100_000, dtype="int32"),
        tile_origin_l93_m=TILE_ORIGIN,
    )
    plan = {
        "schema": viewer.PLAN_SCHEMA,
        "zone_id": "viewer-zone",
        "crs": "EPSG:2154",
        "tile_size_m": 500,
        "tile_count": 1,
        "production_bounds_l93_m": [
            TILE_ORIGIN[0],
            TILE_ORIGIN[1],
            TILE_ORIGIN[0] + 500,
            TILE_ORIGIN[1] + 500,
        ],
        "tiles": [
            {
                "tile_id": "x820000_y6312500",
                "origin_l93_m": list(TILE_ORIGIN),
            }
        ],
    }
    prefix = "fireviewer-viewer-zone"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{prefix}/zone-plan.json", json.dumps(plan))
        archive.writestr(
            f"{prefix}/packages/x820000_y6312500/terrain.fvtg",
            encode_fixed_terrain(terrain),
        )
        archive.writestr(
            f"{prefix}/packages/x820000_y6312500/ground/ground-color.png",
            _png(),
        )
        if unsafe:
            archive.writestr("../escape.txt", "forbidden")
    return path


def _layer_package(tmp_path: Path, *, outside: bool = False) -> Path:
    compiled = layer.compile_perimeter_layer(_dataset(outside=outside))
    root = tmp_path / ("outside-layer" if outside else "layer")
    layer.write_perimeter_layer_package(compiled, root)
    return root


def _glb_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total == len(payload)
    json_length, chunk_type = struct.unpack_from("<I4s", payload, 12)
    assert chunk_type == b"JSON"
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


def test_builds_one_interactive_glb_per_observed_frame(tmp_path: Path) -> None:
    map_zip = _map_zip(tmp_path / "map.zip")
    package = _layer_package(tmp_path)

    first = viewer.build_perimeter_timeline_viewer(map_zip, package, tmp_path / "work")
    second = viewer.build_perimeter_timeline_viewer(map_zip, package, tmp_path / "work")

    assert first == second
    assert first.manifest["authoritative"] is False
    assert first.manifest["frame_count"] == 2
    assert first.manifest["between_observations"] == "undefined"
    assert [frame.observed_at for frame in first.frames] == [
        "2026-08-10T00:00:00Z",
        "2026-08-11T00:00:00Z",
    ]
    assert "touché" in first.frames[0].caption
    assert "actif" in first.frames[0].caption
    assert "2026-08-09T18:00:00Z → 2026-08-10T00:00:00Z" in (first.frames[0].caption)
    assert first.manifest["frames"][0]["time_range"]["kind"] == ("explicit_interval")

    frame0 = _glb_json(first.frames[0].model)
    frame1 = _glb_json(first.frames[1].model)
    assert [node["name"] for node in frame0["nodes"]] == [
        "FireViewerTerrain",
        "AffectedObserved",
        "ActiveObserved",
    ]
    assert [node["name"] for node in frame1["nodes"]] == [
        "FireViewerTerrain",
        "AffectedObserved",
    ]
    assert frame0["extras"]["previewOnly"] is True
    assert frame0["extras"]["authoritativeSource"] == layer.STAGE_NAME


def test_viewer_rejects_tampered_frame(tmp_path: Path) -> None:
    product = viewer.build_perimeter_timeline_viewer(
        _map_zip(tmp_path / "map.zip"),
        _layer_package(tmp_path),
        tmp_path / "work",
    )
    product.frames[0].model.write_bytes(b"tampered")

    with pytest.raises(viewer.GeographicPerimeterViewerError, match="altérée"):
        viewer.build_perimeter_timeline_viewer(
            tmp_path / "map.zip", tmp_path / "layer", tmp_path / "work"
        )


def test_viewer_rejects_perimeters_outside_uploaded_map(tmp_path: Path) -> None:
    with pytest.raises(viewer.GeographicPerimeterViewerError, match="hors de la carte"):
        viewer.build_perimeter_timeline_viewer(
            _map_zip(tmp_path / "map.zip"),
            _layer_package(tmp_path, outside=True),
            tmp_path / "work",
        )


def test_viewer_rejects_zip_traversal(tmp_path: Path) -> None:
    with pytest.raises(viewer.GeographicPerimeterViewerError, match="non confiné"):
        viewer.build_perimeter_timeline_viewer(
            _map_zip(tmp_path / "map.zip", unsafe=True),
            _layer_package(tmp_path),
            tmp_path / "work",
        )
