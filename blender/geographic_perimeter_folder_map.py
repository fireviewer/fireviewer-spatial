"""Adapt the perimeter preview compiler to sealed FireViewer map folders.

New Lightning map jobs publish their canonical map as a normal Hugging Face
folder. The historical perimeter viewer still exposes a ZIP-oriented internal
reader; this module supplies the equivalent fail-closed folder reader without
creating any temporary archive.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

import geographic_perimeter_viewer as viewer
from fixed_terrain_grid import decode_fixed_terrain
from portable_scene_package import validate_map_upload_package


class FolderMapError(RuntimeError):
    pass


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FolderMapError(f"{label} JSON invalide: {error}") from error
    if not isinstance(value, dict):
        raise FolderMapError(f"{label}: objet JSON attendu")
    return value


def read_map_folder(path: Path | str) -> Any:
    """Validate a sealed map folder and expose terrain data to the viewer."""

    root = Path(path).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise FolderMapError("la source de carte HF doit être un dossier régulier")

    reference = validate_map_upload_package(root)
    plan = _json(root / viewer.MAP_PLAN_NAME, "zone-plan")
    if (
        plan.get("schema") != viewer.PLAN_SCHEMA
        or plan.get("crs") != "EPSG:2154"
        or plan.get("tile_size_m") != viewer.TILE_SIZE_M
        or plan.get("zone_id") != reference.zone_id
    ):
        raise FolderMapError("zone-plan du dossier HF incompatible")

    bounds = plan.get("production_bounds_l93_m")
    records = plan.get("tiles")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in bounds)
        or not isinstance(records, list)
        or not records
        or len(records) > viewer.MAX_MAP_TILES
        or plan.get("tile_count") != len(records)
    ):
        raise FolderMapError("grille du dossier HF invalide")
    west, south, east, north = bounds
    if east <= west or north <= south:
        raise FolderMapError("emprise du dossier HF invalide")

    tiles: list[Any] = []
    seen_origins: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise FolderMapError(f"tuile {index}: objet attendu")
        tile_id = record.get("tile_id")
        origin = record.get("origin_l93_m")
        if (
            not isinstance(tile_id, str)
            or not isinstance(origin, list)
            or len(origin) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in origin)
        ):
            raise FolderMapError(f"tuile {index}: identité invalide")
        origin_tuple = (origin[0], origin[1])
        if origin_tuple in seen_origins:
            raise FolderMapError("origine de tuile dupliquée")
        seen_origins.add(origin_tuple)

        terrain_path = root / "packages" / tile_id / "terrain.fvtg"
        ground_path = root / "packages" / tile_id / "ground" / "ground-color.png"
        if not terrain_path.is_file() or not ground_path.is_file():
            raise FolderMapError(
                f"tuile {tile_id}: terrain.fvtg ou ground-color.png absent"
            )
        try:
            terrain = decode_fixed_terrain(terrain_path.read_bytes())
        except Exception as error:
            raise FolderMapError(f"tuile {tile_id}: FVTG invalide: {error}") from error
        expected_origin_mm = (origin[0] * 1000, origin[1] * 1000)
        if terrain.tile_origin_mm != expected_origin_mm:
            raise FolderMapError(f"tuile {tile_id}: origine FVTG incohérente")
        ground_png = ground_path.read_bytes()
        try:
            with Image.open(BytesIO(ground_png)) as image:
                image.load()
                if image.mode != "RGB" or image.size != (
                    viewer.GROUND_SIZE,
                    viewer.GROUND_SIZE,
                ):
                    raise FolderMapError(
                        f"tuile {tile_id}: ground-color doit être RGB "
                        f"{viewer.GROUND_SIZE}x{viewer.GROUND_SIZE}"
                    )
        except (OSError, ValueError) as error:
            raise FolderMapError(f"tuile {tile_id}: texture sol invalide") from error
        tiles.append(viewer._MapTile(tile_id, origin_tuple, terrain, ground_png))

    expected_origins = {
        (x, y)
        for y in range(south, north, viewer.TILE_SIZE_M)
        for x in range(west, east, viewer.TILE_SIZE_M)
    }
    if seen_origins != expected_origins:
        raise FolderMapError("la grille du dossier HF n'est pas exhaustive")
    return viewer._MapData(
        # The manifest hash is the immutable identity of the sealed folder and
        # replaces the historical hash of the monolithic ZIP container.
        map_sha256=reference.manifest_sha256,
        zone_id=reference.zone_id,
        bounds=(west, south, east, north),
        tiles=tuple(sorted(tiles, key=lambda tile: (tile.origin[1], tile.origin[0]))),
    )


def build_perimeter_timeline_viewer_for_map(
    map_source: Path | str,
    layer_package_root: Path | str,
    work_root: Path | str,
) -> Any:
    """Build the perimeter viewer from either a new folder or a legacy ZIP."""

    source = Path(map_source).resolve(strict=True)
    if source.is_file():
        return viewer.build_perimeter_timeline_viewer(
            source,
            layer_package_root,
            work_root,
        )
    if not source.is_dir():
        raise FolderMapError("source de carte absente")

    # The upstream viewer's only ZIP-specific dependency is its private map
    # reader. Replace it for this single synchronous call and restore it even on
    # failure. A Lightning perimeter process compiles one job at a time.
    original = viewer._read_map_archive
    viewer._read_map_archive = read_map_folder
    try:
        return viewer.build_perimeter_timeline_viewer(
            source,
            layer_package_root,
            work_root,
        )
    finally:
        viewer._read_map_archive = original


__all__ = [
    "FolderMapError",
    "build_perimeter_timeline_viewer_for_map",
    "read_map_folder",
]
