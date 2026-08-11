"""Build a lightweight interactive timeline preview from authoritative USD layers.

The viewer consumes a portable FireViewer map ZIP and a validated perimeter
package.  It rebuilds only FVTG LOD2 terrain plus the baked ground mosaic, then
creates one GLB preview per observed frame.  GLB files are derived visual aids;
the USD fixed layers and JSON timeline remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import struct
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely import constrained_delaunay_triangles
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import transform

from fixed_terrain_grid import (
    FixedTerrainTile,
    decode_fixed_terrain,
    lod_absolute_heights_mm,
    source_grid_mm,
)
from geographic_perimeter_layer import (
    MANIFEST_NAME,
    SOURCE_NAME,
    TIMELINE_NAME,
    validate_perimeter_layer_package,
)


VIEWER_SCHEMA = "fireviewer.geographic-perimeter-timeline-viewer.v1"
PLAN_SCHEMA = "fireviewer.simple-measured-zone-plan.v1"
VIEWER_MANIFEST_NAME = "perimeter-viewer.manifest.json"
MAP_PLAN_NAME = "zone-plan.json"
TILE_SIZE_M = 500
GROUND_SIZE = 250
MAX_MAP_TILES = 900
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 50 * 1024 * 1024 * 1024
MAX_MOSAIC_SIDE = 2048
_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


class GeographicPerimeterViewerError(RuntimeError):
    """The map or fixed layers cannot be visualized safely."""


@dataclass(frozen=True, slots=True)
class ViewerFrame:
    index: int
    observed_at: str
    caption: str
    model: Path


@dataclass(frozen=True, slots=True)
class PerimeterViewerProduct:
    root: Path
    manifest: dict[str, Any]
    frames: tuple[ViewerFrame, ...]


@dataclass(frozen=True, slots=True)
class _MapTile:
    tile_id: str
    origin: tuple[int, int]
    terrain: FixedTerrainTile
    ground_png: bytes


@dataclass(frozen=True, slots=True)
class _MapData:
    map_sha256: str
    zone_id: str
    bounds: tuple[int, int, int, int]
    tiles: tuple[_MapTile, ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeographicPerimeterViewerError(
            f"{label} JSON invalide: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise GeographicPerimeterViewerError(f"{label}: objet JSON attendu")
    return payload


def _require_d_output(path: Path) -> Path:
    lexical = PureWindowsPath(str(path))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise GeographicPerimeterViewerError(
            "les sorties du viewer doivent rester sur D:"
        )
    resolved = path.resolve(strict=False)
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise GeographicPerimeterViewerError(
            "les sorties du viewer doivent rester sur D:"
        )
    return resolved


def _safe_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        name = info.filename
        if "\\" in name:
            raise GeographicPerimeterViewerError("chemin ZIP Windows interdit")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise GeographicPerimeterViewerError("chemin ZIP non confiné")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise GeographicPerimeterViewerError("lien symbolique ZIP interdit")
        if info.is_dir():
            continue
        if name in members:
            raise GeographicPerimeterViewerError("entrée ZIP dupliquée")
        total += info.file_size
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise GeographicPerimeterViewerError("archive de carte trop volumineuse")
        members[name] = info
    return members


def _member_name(prefix: PurePosixPath, relative: str) -> str:
    return (prefix / PurePosixPath(relative)).as_posix()


def _read_map_archive(path: Path) -> _MapData:
    if path.suffix.lower() != ".zip":
        raise GeographicPerimeterViewerError(
            "la carte doit être le ZIP autonome FireViewer"
        )
    map_sha256 = _sha256_file(path)
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise GeographicPerimeterViewerError(
            f"ZIP de carte invalide: {error}"
        ) from error
    with archive:
        members = _safe_zip_members(archive)
        plan_names = [
            name for name in members if PurePosixPath(name).name == MAP_PLAN_NAME
        ]
        if len(plan_names) != 1:
            raise GeographicPerimeterViewerError(
                "le ZIP doit contenir un unique zone-plan.json"
            )
        plan_name = plan_names[0]
        prefix = PurePosixPath(plan_name).parent
        plan = _load_json_bytes(archive.read(plan_name), "zone-plan")
        if (
            plan.get("schema") != PLAN_SCHEMA
            or plan.get("crs") != "EPSG:2154"
            or plan.get("tile_size_m") != TILE_SIZE_M
        ):
            raise GeographicPerimeterViewerError("zone-plan incompatible")
        bounds = plan.get("production_bounds_l93_m")
        records = plan.get("tiles")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 4
            or not all(isinstance(value, int) for value in bounds)
            or not isinstance(records, list)
            or not records
            or len(records) > MAX_MAP_TILES
            or plan.get("tile_count") != len(records)
        ):
            raise GeographicPerimeterViewerError("grille de carte invalide")
        west, south, east, north = bounds
        if east <= west or north <= south:
            raise GeographicPerimeterViewerError("emprise de carte invalide")
        tiles: list[_MapTile] = []
        seen_origins: set[tuple[int, int]] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise GeographicPerimeterViewerError(f"tuile {index}: objet attendu")
            tile_id = record.get("tile_id")
            origin = record.get("origin_l93_m")
            if (
                not isinstance(tile_id, str)
                or not isinstance(origin, list)
                or len(origin) != 2
                or not all(isinstance(value, int) for value in origin)
            ):
                raise GeographicPerimeterViewerError(
                    f"tuile {index}: identité invalide"
                )
            origin_tuple = (origin[0], origin[1])
            if origin_tuple in seen_origins:
                raise GeographicPerimeterViewerError("origine de tuile dupliquée")
            seen_origins.add(origin_tuple)
            terrain_name = _member_name(prefix, f"packages/{tile_id}/terrain.fvtg")
            ground_name = _member_name(
                prefix, f"packages/{tile_id}/ground/ground-color.png"
            )
            if terrain_name not in members or ground_name not in members:
                raise GeographicPerimeterViewerError(
                    f"tuile {tile_id}: terrain.fvtg ou ground-color.png absent"
                )
            try:
                terrain = decode_fixed_terrain(archive.read(terrain_name))
            except Exception as error:
                raise GeographicPerimeterViewerError(
                    f"tuile {tile_id}: FVTG invalide: {error}"
                ) from error
            expected_origin_mm = (origin[0] * 1000, origin[1] * 1000)
            if terrain.tile_origin_mm != expected_origin_mm:
                raise GeographicPerimeterViewerError(
                    f"tuile {tile_id}: origine FVTG incohérente"
                )
            ground_png = archive.read(ground_name)
            try:
                with Image.open(BytesIO(ground_png)) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (GROUND_SIZE, GROUND_SIZE):
                        raise GeographicPerimeterViewerError(
                            f"tuile {tile_id}: ground-color doit être RGB {GROUND_SIZE}x{GROUND_SIZE}"
                        )
            except (OSError, ValueError) as error:
                raise GeographicPerimeterViewerError(
                    f"tuile {tile_id}: texture sol invalide"
                ) from error
            tiles.append(_MapTile(tile_id, origin_tuple, terrain, ground_png))
    expected_origins = {
        (x, y)
        for y in range(south, north, TILE_SIZE_M)
        for x in range(west, east, TILE_SIZE_M)
    }
    if seen_origins != expected_origins:
        raise GeographicPerimeterViewerError("la grille de carte n’est pas exhaustive")
    return _MapData(
        map_sha256=map_sha256,
        zone_id=str(plan.get("zone_id", "unknown-zone")),
        bounds=(west, south, east, north),
        tiles=tuple(sorted(tiles, key=lambda tile: (tile.origin[1], tile.origin[0]))),
    )


def _ground_mosaic(map_data: _MapData) -> bytes:
    west, south, east, north = map_data.bounds
    columns = (east - west) // TILE_SIZE_M
    rows = (north - south) // TILE_SIZE_M
    canvas = Image.new("RGB", (columns * GROUND_SIZE, rows * GROUND_SIZE))
    for tile in map_data.tiles:
        column = (tile.origin[0] - west) // TILE_SIZE_M
        row = (north - tile.origin[1] - TILE_SIZE_M) // TILE_SIZE_M
        with Image.open(BytesIO(tile.ground_png)) as image:
            canvas.paste(image, (column * GROUND_SIZE, row * GROUND_SIZE))
    if max(canvas.size) > MAX_MOSAIC_SIDE:
        ratio = MAX_MOSAIC_SIDE / max(canvas.size)
        canvas = canvas.resize(
            (max(1, round(canvas.width * ratio)), max(1, round(canvas.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=True, compress_level=9)
    return buffer.getvalue()


def _terrain_arrays(
    map_data: _MapData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    west, south, east, north = map_data.bounds
    width = east - west
    height = north - south
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    indices: list[int] = []
    for tile in map_data.tiles:
        mesh = tile.terrain.lods[2]
        heights = lod_absolute_heights_mm(tile.terrain, 2) / 1000.0
        vertex_offset = len(positions)
        spacing = TILE_SIZE_M / (mesh.grid_size - 1)
        for row in range(mesh.grid_size):
            y_l93 = tile.origin[1] + row * spacing
            for column in range(mesh.grid_size):
                x_l93 = tile.origin[0] + column * spacing
                elevation = float(heights[row, column])
                positions.append((x_l93 - west, elevation, -(y_l93 - south)))
                nx, ny, nz = mesh.normals_snorm16[row * mesh.grid_size + column]
                normals.append((nx / 32767.0, nz / 32767.0, -ny / 32767.0))
                texcoords.append(((x_l93 - west) / width, (y_l93 - south) / height))
        indices.extend(
            vertex_offset + vertex
            for triangle in mesh.core_triangles
            for vertex in triangle
        )
    return (
        np.asarray(positions, dtype="<f4"),
        np.asarray(normals, dtype="<f4"),
        np.asarray(texcoords, dtype="<f4"),
        np.asarray(indices, dtype="<u4"),
    )


class _HeightSampler:
    def __init__(self, map_data: _MapData) -> None:
        self.bounds = map_data.bounds
        self.grids = {
            tile.origin: source_grid_mm(tile.terrain).astype("float64") / 1000.0
            for tile in map_data.tiles
        }

    def sample(self, x_l93: float, y_l93: float) -> float:
        west, south, east, north = self.bounds
        if not west <= x_l93 <= east or not south <= y_l93 <= north:
            raise GeographicPerimeterViewerError("périmètre hors de la carte chargée")
        tile_x = min(math.floor(x_l93 / TILE_SIZE_M) * TILE_SIZE_M, east - TILE_SIZE_M)
        tile_y = min(math.floor(y_l93 / TILE_SIZE_M) * TILE_SIZE_M, north - TILE_SIZE_M)
        tile_x = max(tile_x, west)
        tile_y = max(tile_y, south)
        grid = self.grids.get((tile_x, tile_y))
        if grid is None:
            raise GeographicPerimeterViewerError("périmètre sur une tuile absente")
        column = min(250.0, max(0.0, (x_l93 - tile_x) / 2.0))
        row = min(250.0, max(0.0, (y_l93 - tile_y) / 2.0))
        x0 = int(math.floor(column))
        y0 = int(math.floor(row))
        x1 = min(250, x0 + 1)
        y1 = min(250, y0 + 1)
        tx = column - x0
        ty = row - y0
        return float(
            (1 - tx) * (1 - ty) * grid[y0, x0]
            + tx * (1 - ty) * grid[y0, x1]
            + (1 - tx) * ty * grid[y1, x0]
            + tx * ty * grid[y1, x1]
        )


def _projected_geometry(value: Any) -> BaseGeometry | None:
    if value is None:
        return None
    geometry = shape(value)
    return transform(_TO_L93.transform, geometry).normalize()


def _layer_arrays(
    geometry: BaseGeometry | None,
    *,
    sampler: _HeightSampler,
    origin: tuple[int, int],
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if geometry is None:
        return None
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    indices: list[int] = []
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        triangles = sorted(
            (
                orient(item, sign=1.0)
                for item in constrained_delaunay_triangles(polygon).geoms
            ),
            key=lambda item: (
                round(item.bounds[0], 6),
                round(item.bounds[1], 6),
                round(item.centroid.x, 6),
                round(item.centroid.y, 6),
            ),
        )
        for triangle in triangles:
            for x_l93, y_l93 in list(triangle.exterior.coords)[:3]:
                elevation = sampler.sample(x_l93, y_l93) + offset_m
                positions.append((x_l93 - origin[0], elevation, -(y_l93 - origin[1])))
                normals.append((0.0, 1.0, 0.0))
                indices.append(len(indices))
    if not positions:
        return None
    return (
        np.asarray(positions, dtype="<f4"),
        np.asarray(normals, dtype="<f4"),
        np.asarray(indices, dtype="<u4"),
    )


class _GlbBuilder:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []

    def _blob(self, value: bytes, *, target: int | None = None) -> int:
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        self.binary.extend(value)
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(value),
        }
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def _accessor(
        self,
        array: np.ndarray,
        *,
        kind: str,
        component_type: int,
        target: int,
        include_bounds: bool = False,
    ) -> int:
        view = self._blob(array.tobytes(order="C"), target=target)
        accessor: dict[str, Any] = {
            "bufferView": view,
            "componentType": component_type,
            "count": int(array.shape[0]),
            "type": kind,
        }
        if include_bounds:
            reshaped = array.reshape(array.shape[0], -1)
            accessor["min"] = [float(value) for value in reshaped.min(axis=0)]
            accessor["max"] = [float(value) for value in reshaped.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_mesh(
        self,
        name: str,
        positions: np.ndarray,
        normals: np.ndarray,
        indices: np.ndarray,
        material: int,
        texcoords: np.ndarray | None = None,
    ) -> None:
        attributes = {
            "POSITION": self._accessor(
                positions,
                kind="VEC3",
                component_type=5126,
                target=34962,
                include_bounds=True,
            ),
            "NORMAL": self._accessor(
                normals, kind="VEC3", component_type=5126, target=34962
            ),
        }
        if texcoords is not None:
            attributes["TEXCOORD_0"] = self._accessor(
                texcoords, kind="VEC2", component_type=5126, target=34962
            )
        index_accessor = self._accessor(
            indices, kind="SCALAR", component_type=5125, target=34963
        )
        self.meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": index_accessor,
                        "material": material,
                        "mode": 4,
                    }
                ],
            }
        )
        self.nodes.append({"name": name, "mesh": len(self.meshes) - 1})

    def build(self, *, mosaic_png: bytes, frame: Mapping[str, Any]) -> bytes:
        image_view = self._blob(mosaic_png)
        gltf: dict[str, Any] = {
            "asset": {
                "version": "2.0",
                "generator": "FireViewer USD timeline viewer v1",
            },
            "extensionsUsed": ["KHR_materials_unlit"],
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "buffers": [{"byteLength": len(self.binary)}],
            "bufferViews": self.buffer_views,
            "accessors": self.accessors,
            "images": [{"bufferView": image_view, "mimeType": "image/png"}],
            "samplers": [
                {"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}
            ],
            "textures": [{"sampler": 0, "source": 0}],
            "materials": [
                {
                    "name": "BakedGround",
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": 0},
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0,
                    },
                },
                {
                    "name": "AffectedObserved",
                    "extensions": {"KHR_materials_unlit": {}},
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.918, 0.345, 0.047, 0.62],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.6,
                    },
                    "alphaMode": "BLEND",
                    "doubleSided": True,
                },
                {
                    "name": "ActiveObserved",
                    "extensions": {"KHR_materials_unlit": {}},
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.961, 0.62, 0.043, 0.86],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.45,
                    },
                    "alphaMode": "BLEND",
                    "doubleSided": True,
                },
            ],
            "extras": {
                "authoritativeSource": "geographic-perimeters.usda",
                "observedAt": frame["observed_at"],
                "elapsedSeconds": frame["elapsed_seconds"],
                "previewOnly": True,
            },
        }
        json_chunk = _canonical_bytes(gltf)
        json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
        binary_chunk = bytes(self.binary)
        binary_chunk += b"\0" * ((4 - len(binary_chunk) % 4) % 4)
        total = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
        return b"".join(
            [
                struct.pack("<4sII", b"glTF", 2, total),
                struct.pack("<I4s", len(json_chunk), b"JSON"),
                json_chunk,
                struct.pack("<I4s", len(binary_chunk), b"BIN\0"),
                binary_chunk,
            ]
        )


def _frame_glb(
    *,
    map_data: _MapData,
    terrain: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    mosaic_png: bytes,
    sampler: _HeightSampler,
    normalized_frame: Mapping[str, Any],
    timeline_frame: Mapping[str, Any],
) -> bytes:
    west, south, east, north = map_data.bounds
    coverage = box(west, south, east, north)
    affected = _projected_geometry(normalized_frame.get("affected"))
    active = _projected_geometry(normalized_frame.get("active"))
    for label, geometry in (("affected", affected), ("active", active)):
        if geometry is not None and not coverage.covers(geometry):
            raise GeographicPerimeterViewerError(
                f"observation {timeline_frame['index']} {label}: périmètre hors de la carte"
            )
    builder = _GlbBuilder()
    positions, normals, texcoords, indices = terrain
    builder.add_mesh("FireViewerTerrain", positions, normals, indices, 0, texcoords)
    affected_arrays = _layer_arrays(
        affected, sampler=sampler, origin=(west, south), offset_m=2.0
    )
    if affected_arrays is not None:
        builder.add_mesh("AffectedObserved", *affected_arrays, material=1)
    active_arrays = _layer_arrays(
        active, sampler=sampler, origin=(west, south), offset_m=3.0
    )
    if active_arrays is not None:
        builder.add_mesh("ActiveObserved", *active_arrays, material=2)
    return builder.build(mosaic_png=mosaic_png, frame=timeline_frame)


def _validate_existing_viewer(
    root: Path, expected_build_id: str
) -> PerimeterViewerProduct:
    manifest_path = root / VIEWER_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeographicPerimeterViewerError(
            f"manifest viewer invalide: {error}"
        ) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != VIEWER_SCHEMA
        or manifest.get("build_id") != expected_build_id
        or manifest.get("authoritative") is not False
    ):
        raise GeographicPerimeterViewerError("identité du viewer incohérente")
    frames: list[ViewerFrame] = []
    expected_files = {VIEWER_MANIFEST_NAME}
    for record in manifest.get("frames", []):
        if not isinstance(record, Mapping):
            raise GeographicPerimeterViewerError("frame viewer invalide")
        relative = record.get("path")
        if not isinstance(relative, str) or PurePosixPath(relative).name != relative:
            raise GeographicPerimeterViewerError("chemin de frame viewer invalide")
        path = root / relative
        if not path.is_file() or _sha256_file(path) != record.get("sha256"):
            raise GeographicPerimeterViewerError("frame GLB absente ou altérée")
        expected_files.add(relative)
        frames.append(
            ViewerFrame(
                int(record["index"]),
                str(record["observed_at"]),
                str(record["caption"]),
                path,
            )
        )
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_dir() for path in root.iterdir()):
        raise GeographicPerimeterViewerError("contenu du viewer inattendu")
    if len(frames) != manifest.get("frame_count") or not frames:
        raise GeographicPerimeterViewerError("timeline du viewer incomplète")
    return PerimeterViewerProduct(root, manifest, tuple(frames))


def build_perimeter_timeline_viewer(
    map_archive: Path | str,
    layer_package_root: Path | str,
    work_root: Path | str,
) -> PerimeterViewerProduct:
    """Build or revalidate one GLB timeline viewer for a map/layer pair."""

    map_path = Path(map_archive).resolve(strict=True)
    package_root = Path(layer_package_root).resolve(strict=True)
    layer_manifest = validate_perimeter_layer_package(package_root)
    map_data = _read_map_archive(map_path)
    viewer_compiler_sha256 = _sha256_file(Path(__file__))
    build_id = _sha256_bytes(
        _canonical_bytes(
            {
                "schema": VIEWER_SCHEMA,
                "map_sha256": map_data.map_sha256,
                "layer_build_id": layer_manifest["build_id"],
                "viewer_compiler_sha256": viewer_compiler_sha256,
            }
        )
    )
    output_root = _require_d_output(Path(work_root)) / "perimeter-viewers" / build_id
    if output_root.exists():
        return _validate_existing_viewer(output_root, build_id)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    normalized = _load_json_bytes(
        (package_root / SOURCE_NAME).read_bytes(), "source normalisée"
    )
    timeline = _load_json_bytes((package_root / TIMELINE_NAME).read_bytes(), "timeline")
    normalized_frames = normalized.get("frames")
    timeline_frames = timeline.get("frames")
    if (
        not isinstance(normalized_frames, list)
        or not isinstance(timeline_frames, list)
        or len(normalized_frames) != len(timeline_frames)
        or not normalized_frames
    ):
        raise GeographicPerimeterViewerError("frames USD/timeline incohérentes")
    terrain = _terrain_arrays(map_data)
    mosaic_png = _ground_mosaic(map_data)
    sampler = _HeightSampler(map_data)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}-staging-", dir=str(output_root.parent))
    )
    records: list[dict[str, Any]] = []
    try:
        for index, (normalized_frame, timeline_frame) in enumerate(
            zip(normalized_frames, timeline_frames, strict=True)
        ):
            glb = _frame_glb(
                map_data=map_data,
                terrain=terrain,
                mosaic_png=mosaic_png,
                sampler=sampler,
                normalized_frame=normalized_frame,
                timeline_frame=timeline_frame,
            )
            name = f"frame-{index:04d}.glb"
            (staging / name).write_bytes(glb)
            affected_area = timeline_frame["affected"]["area_ha"]
            active_area = timeline_frame["active"]["area_ha"]
            time_range = timeline_frame["time_range"]
            temporal_label = timeline_frame["observed_at"]
            if time_range["start"] != time_range["end"]:
                temporal_label = f"{time_range['start']} → {time_range['end']}"
            caption = (
                f"{temporal_label} · touché {affected_area:g} ha · "
                f"actif {active_area:g} ha"
            )
            records.append(
                {
                    "index": index,
                    "observed_at": timeline_frame["observed_at"],
                    "elapsed_seconds": timeline_frame["elapsed_seconds"],
                    "time_range": time_range,
                    "caption": caption,
                    "path": name,
                    "sha256": _sha256_bytes(glb),
                    "byte_count": len(glb),
                }
            )
        manifest: dict[str, Any] = {
            "schema": VIEWER_SCHEMA,
            "status": "derived_visual_timeline",
            "authoritative": False,
            "build_id": build_id,
            "map_sha256": map_data.map_sha256,
            "map_zone_id": map_data.zone_id,
            "layer_build_id": layer_manifest["build_id"],
            "layer_manifest_sha256": _sha256_file(package_root / MANIFEST_NAME),
            "viewer_compiler_sha256": viewer_compiler_sha256,
            "frame_count": len(records),
            "selection": "one_observed_frame_at_a_time",
            "between_observations": "undefined",
            "frames": records,
        }
        manifest["manifest_sha256"] = _sha256_bytes(_canonical_bytes(manifest))
        (staging / VIEWER_MANIFEST_NAME).write_bytes(_canonical_bytes(manifest) + b"\n")
        os.replace(staging, output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _validate_existing_viewer(output_root, build_id)


__all__ = [
    "GeographicPerimeterViewerError",
    "PerimeterViewerProduct",
    "VIEWER_SCHEMA",
    "ViewerFrame",
    "build_perimeter_timeline_viewer",
]
