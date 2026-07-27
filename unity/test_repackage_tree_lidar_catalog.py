from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from fwtile import build_container, read_container  # noqa: E402
from repackage_tree_lidar_catalog import repackage  # noqa: E402


def _asset(
    root: Path, relative: str, content: bytes, *, resolution: object | None = None
) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    result: dict[str, object] = {
        "path": relative,
        "url": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }
    if resolution is not None:
        result["resolution_m"] = resolution
    return result


def test_repackage_keeps_only_terrain_and_trees(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = build_container(
        kind="detail_tile",
        tile_id="x0_y0_s500",
        bounds_l93_m=[0.0, 0.0, 500.0, 500.0],
        origin_l93_m=[0.0, 0.0, 0.0],
        sections=[
            ("terrain", b"terrain", {"encoding": "test"}),
            ("trees", b"trees", {"count": 2, "encoding": "test"}),
            ("buildings", b"buildings", {"encoding": "test"}),
            ("roads", b"roads", {"encoding": "test"}),
            ("water", b"water", {"encoding": "test"}),
        ],
    )
    catalog = {
        "schema": "fireviewer.remote-tile-catalog.v1",
        "catalog_version": 1,
        "crs": "EPSG:2154",
        "linear_unit": "metre",
        "origin_l93_m": [0.0, 0.0, 0.0],
        "source": {"terrain_surface": "mns_lidar"},
        "lod_policy": {
            "far": {
                "bounds_l93_m": [0.0, 0.0, 500.0, 500.0],
                "terrain": _asset(
                    source, "assets/far/global.fwterrain", b"far", resolution=[5.0, 5.0]
                ),
                "imagery": _asset(
                    source, "assets/far/global.jpg", b"image", resolution=2.0
                ),
            }
        },
        "tiles": [
            {
                "id": "x0_y0_s500",
                "bounds_l93_m": [0.0, 0.0, 500.0, 500.0],
                "payload": _asset(source, "assets/detail/tile.fwtile", payload),
                "imagery": _asset(
                    source, "assets/imagery/tile.jpg", b"tile-image", resolution=0.5
                ),
                "counts": {"terrain_vertices": 4},
            }
        ],
    }
    (source / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    receipt = repackage(source_root=source, output_root=tmp_path / "output")
    output = json.loads((tmp_path / "output/catalog.json").read_text(encoding="utf-8"))
    detail = output["tiles"][0]
    payload_path = tmp_path / "output" / detail["payload"]["url"]

    assert receipt["gpu_used"] is False
    assert receipt["fixed_maximum_bounding_area_km2"] == 250.0
    assert detail["sections"] == ["terrain", "trees"]
    assert detail["counts"]["buildings"] == 0
    assert set(read_container(payload_path.read_bytes())["sections"]) == {
        "terrain",
        "trees",
    }
