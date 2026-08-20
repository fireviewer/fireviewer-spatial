"""Create one GPS request guaranteed to resolve to a 3 x 3 Lambert-93 tile box."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pyproj import Transformer

from simple_production_engine import TILE_SIZE_M, plan_zone

SIDE_KM = 1.5
EXPECTED_TILE_COUNT = 9


class NineTilePlanError(RuntimeError):
    pass


def build_nine_tile_request(latitude: float, longitude: float) -> dict[str, Any]:
    latitude = float(latitude)
    longitude = float(longitude)
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise NineTilePlanError("latitude and longitude must be finite")
    forward = Transformer.from_crs(4326, 2154, always_xy=True)
    reverse = Transformer.from_crs(2154, 4326, always_xy=True)
    target_x, target_y = forward.transform(longitude, latitude)
    center_tile_west = math.floor(target_x / TILE_SIZE_M) * TILE_SIZE_M
    center_tile_south = math.floor(target_y / TILE_SIZE_M) * TILE_SIZE_M
    center_x = center_tile_west + TILE_SIZE_M / 2.0
    center_y = center_tile_south + TILE_SIZE_M / 2.0
    centered_longitude, centered_latitude = reverse.transform(center_x, center_y)

    request = {
        "latitude": round(float(centered_latitude), 10),
        "longitude": round(float(centered_longitude), 10),
        "side_km": SIDE_KM,
        "fixed_asset_placements": None,
    }
    plan = plan_zone(
        request["latitude"],
        request["longitude"],
        request["side_km"],
        max_tiles=EXPECTED_TILE_COUNT,
    )
    expected_origins = {
        (center_tile_west + dx * TILE_SIZE_M, center_tile_south + dy * TILE_SIZE_M)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }
    observed_origins = {tile.origin_l93_m for tile in plan.tiles}
    if len(plan.tiles) != EXPECTED_TILE_COUNT or observed_origins != expected_origins:
        raise NineTilePlanError(
            "round-trip GPS planning did not preserve the exact 3 x 3 Lambert-93 grid"
        )
    return {
        "schema": "fireviewer.map-validation-request-plan.v1",
        "target_input": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "request": request,
        "zone_id": plan.zone_id,
        "tile_count": len(plan.tiles),
        "production_bounds_l93_m": list(plan.production_bounds_l93_m),
        "center_l93_m": [center_x, center_y],
        "tiles": [
            {
                "tile_id": tile.tile_id,
                "origin_l93_m": list(tile.origin_l93_m),
            }
            for tile in plan.tiles
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(argv)
    result = build_nine_tile_request(options.latitude, options.longitude)
    content = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["NineTilePlanError", "build_nine_tile_request"]
