from __future__ import annotations

from pathlib import Path
import sys

import pytest


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from export_remote_catalog import FWTileError, validate_catalog  # noqa: E402


def _reference(url: str, *, resolution: float | list[float]) -> dict[str, object]:
    return {
        "url": url,
        "sha256": "a" * 64,
        "byte_count": 1,
        "resolution_m": resolution,
    }


def _global_only_catalog() -> dict[str, object]:
    return {
        "schema": "fireviewer.remote-tile-catalog.v1",
        "exported_detail_tile_count": 0,
        "lod_policy": {
            "far": {
                "terrain": _reference("far/global.fwterrain", resolution=[5.0, 5.0]),
                "imagery": _reference("far/global.jpg", resolution=2.0),
            },
            "detail": {
                "enabled": False,
                "mode": "global_only",
                "publish_distance_m": 600.0,
                "preload_radius_m": 750.0,
                "maximum_resident_tile_count": 0,
                "near_disabled": True,
                "transition": "global_surface_only",
                "eviction": "not_applicable",
            },
        },
        "tiles": [],
    }


def test_global_only_catalog_is_valid_without_detail_tiles() -> None:
    validate_catalog(_global_only_catalog())


def test_global_only_catalog_rejects_a_detail_tile() -> None:
    catalog = _global_only_catalog()
    catalog["tiles"] = [{"id": "x0_y0_s500"}]

    with pytest.raises(FWTileError, match="global-only"):
        validate_catalog(catalog)


def test_tree_lidar_catalog_accepts_only_terrain_and_trees() -> None:
    catalog = _global_only_catalog()
    catalog["exported_detail_tile_count"] = 1
    catalog["lod_policy"]["detail"] = {
        "enabled": True,
        "mode": "streamed_detail",
        "content_mode": "trees_3d_lidar_infrastructure",
        "publish_distance_m": 600.0,
        "preload_radius_m": 750.0,
        "maximum_resident_tile_count": 16,
        "near_disabled": True,
    }
    catalog["tiles"] = [
        {
            "id": "x0_y0_s500",
            "bounds_l93_m": [0.0, 0.0, 500.0, 500.0],
            "payload": _reference("detail/tile.fwtile", resolution=0.5),
            "imagery": _reference("imagery/tile.jpg", resolution=0.5),
            "sections": ["terrain", "trees"],
        }
    ]

    validate_catalog(catalog)
