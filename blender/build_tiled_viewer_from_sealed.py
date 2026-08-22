"""Build browser viewer tiles from sealed scientific zone artefacts."""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from build_tiled_viewer_package import (
    CATALOG_NAME,
    FAMILY_ROOTS,
    OUTPUT_DIRECTORY,
    PROTOTYPE_EXPORT_NAME,
    PROTOTYPE_EXPORT_SCHEMA,
    PROTOTYPE_PLAN_NAME,
    PROTOTYPE_PLAN_SCHEMA,
    RECEIPT_NAME,
    RECEIPT_SCHEMA,
    SCHEMA,
    TILE_SIZE_M,
    Asset,
    Prototype,
    TiledViewerPackageError,
    _FarGlb,
    _asset,
    _expected_counts,
    _load_json,
    _read_glb,
    _tile_records,
    _write_instances,
    _write_json,
)
from fixed_terrain_grid import FixedTerrainTile, read_fixed_terrain

LAYOUT_SCHEMA = "fireviewer.compact-zone-stage-layout.v1"
SEALED_SOURCE_KIND = "sealed_zone_and_tile_packages"
BUILD_METRICS_NAME = "viewer-build-metrics.v1.json"
BUILD_METRICS_SCHEMA = "fireviewer.tiled-viewer-build-metrics.v1"
PROTOTYPE_CACHE_SCHEMA = "fireviewer.tiled-prototype-cache.v1"
TILE_CACHE_SCHEMA = "fireviewer.tiled-viewer-tile-cache.v1"
_USD_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_USD_IDENTIFIER = re.compile(r"^Asset_[A-Za-z0-9_]+$")
_POINT_INSTANCER = re.compile(r'def\s+PointInstancer\s+"([A-Za-z0-9_]+)"\s*\{')
_FAMILY_BY_PRIM = {
    "Buildings": "buildings",
    "Trees": "trees",
    "ContextAssets": "context_assets",
}


@dataclass(frozen=True, slots=True)
class SealedPrototype:
    prototype_id: str
    family: str
    asset_id: str
    identifier: str


@dataclass(frozen=True, slots=True)
class CompiledTile:
    tile_id: str
    tile_origin: tuple[int, int]
    terrain_asset: Asset
    instance_asset: Asset
    family_counts: dict[str, int]
    prototype_instance_counts: dict[str, int]
    prototype_ids: tuple[str, ...]
    far_positions: Any
    far_normals: Any
    far_texcoords: Any
    far_indices: Any
    far_image: bytes
    node_chain: tuple[dict[str, Any], ...]
    timings_seconds: dict[str, float]
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class FailedTile:
    tile_id: str
    error: str


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root).as_posix()
    except ValueError as error:
        raise TiledViewerPackageError(f"Artefact scellé hors du job: {path}") from error


def _first_file(root: Path, candidates: Sequence[str], label: str) -> Path:
    for relative in candidates:
        path = root.joinpath(*relative.split("/"))
        if path.is_file():
            return path
    raise TiledViewerPackageError(f"{label} absent")


def _validate_asset_reference(
    root: Path, directory: Path, raw: object, label: str
) -> Path:
    if not isinstance(raw, Mapping):
        raise TiledViewerPackageError(f"Référence {label} invalide")
    relative = raw.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise TiledViewerPackageError(f"Chemin {label} invalide")
    path = directory.joinpath(*relative.split("/"))
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise TiledViewerPackageError(f"{label} absent ou hors du job") from error
    byte_count = raw.get("byte_count", raw.get("bytes"))
    if (
        not path.is_file()
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count != path.stat().st_size
    ):
        raise TiledViewerPackageError(f"{label} diverge de son reçu scellé")
    return path


def _load_sealed_sources(
    root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, Path]:
    zone_path = _first_file(root, ("zone.done.json",), "reçu de zone")
    layout_path = _first_file(
        root,
        ("zone-stage-layout.v1.json", "manifests/zone-stage-layout.v1.json"),
        "layout de zone",
    )
    blend_path = _first_file(
        root, ("zone.blend", "scientific/zone.blend"), "zone.blend scellé"
    )
    packages = root / "packages"
    if not packages.is_dir():
        raise TiledViewerPackageError("Paquets de tuiles scellés absents")
    zone = _load_json(zone_path, "reçu de zone")
    layout = _load_json(layout_path, "layout de zone")
    stage_reference = zone.get("stage_layout")
    if (
        layout.get("schema") != LAYOUT_SCHEMA
        or layout.get("status") != "sealed"
        or layout.get("zone_id") != zone.get("zone_id")
        or not isinstance(stage_reference, Mapping)
        or stage_reference.get("byte_count") != layout_path.stat().st_size
        or stage_reference.get("prototype_policy")
        != "one_zone_definition_per_family_asset"
        or stage_reference.get("instance_policy") != "tile_point_instancers_preserved"
    ):
        raise TiledViewerPackageError("Layout de zone non scellé ou divergent")
    records = _tile_records(zone)
    if layout.get("tile_count") != len(records):
        raise TiledViewerPackageError("Nombre de tuiles divergent dans le layout")
    return zone, layout_path, layout, blend_path, packages


def _sealed_prototypes(layout: Mapping[str, Any]) -> list[SealedPrototype]:
    raw_rows = layout.get("prototypes")
    if not isinstance(raw_rows, list):
        raise TiledViewerPackageError("Prototypes absents du layout scellé")
    counters = {family: 0 for family in FAMILY_ROOTS}
    identifiers: set[tuple[str, str]] = set()
    result: list[SealedPrototype] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise TiledViewerPackageError("Prototype du layout invalide")
        family = raw.get("family")
        asset_id = raw.get("asset_id")
        identifier = raw.get("identifier")
        if (
            family not in counters
            or not isinstance(asset_id, str)
            or not asset_id
            or not isinstance(identifier, str)
            or _USD_IDENTIFIER.fullmatch(identifier) is None
            or (family, identifier) in identifiers
        ):
            raise TiledViewerPackageError("Identité de prototype scellé invalide")
        prototype_id = f"{family}-{counters[family]:04d}"
        counters[family] += 1
        identifiers.add((family, identifier))
        result.append(SealedPrototype(prototype_id, family, asset_id, identifier))
    if layout.get("prototype_count") != len(result):
        raise TiledViewerPackageError("Nombre de prototypes divergent dans le layout")
    return result


def _block(text: str, start: int) -> str:
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise TiledViewerPackageError("Bloc PointInstancer USDA non terminé")


def _array(block: str, declaration: str) -> str:
    match = re.search(re.escape(declaration) + r"\s*=\s*\[(.*?)\]", block, re.DOTALL)
    if match is None:
        raise TiledViewerPackageError(f"Tableau USDA absent: {declaration}")
    return match.group(1)


def _numbers(raw: str, *, integer: bool = False) -> list[float] | list[int]:
    values = _USD_NUMBER.findall(raw)
    if integer:
        return [int(value) for value in values]
    return [float(value) for value in values]


def _vectors(raw: str, size: int, label: str) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for item in re.findall(r"\(([^()]*)\)", raw):
        values = tuple(float(value) for value in _USD_NUMBER.findall(item))
        if len(values) != size or any(not math.isfinite(value) for value in values):
            raise TiledViewerPackageError(f"Vecteur USDA invalide: {label}")
        rows.append(values)
    return rows


def _multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        sum(left[row * 4 + inner] * right[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    ]


def _quaternion_matrix(
    position: tuple[float, float, float],
    scale: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> list[float]:
    qw, qx, qy, qz = orientation
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise TiledViewerPackageError("Quaternion USDA nul")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    sx, sy, sz = scale
    px, py, pz = position
    return [
        (1 - 2 * (qy * qy + qz * qz)) * sx,
        (2 * (qx * qy - qz * qw)) * sy,
        (2 * (qx * qz + qy * qw)) * sz,
        px,
        (2 * (qx * qy + qz * qw)) * sx,
        (1 - 2 * (qx * qx + qz * qz)) * sy,
        (2 * (qy * qz - qx * qw)) * sz,
        py,
        (2 * (qx * qz - qy * qw)) * sx,
        (2 * (qy * qz + qx * qw)) * sy,
        (1 - 2 * (qx * qx + qy * qy)) * sz,
        pz,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _rotation_quaternion(rotation: Sequence[Sequence[float]]) -> tuple[float, ...]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0:
        root = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * root
        qx = (rotation[2][1] - rotation[1][2]) / root
        qy = (rotation[0][2] - rotation[2][0]) / root
        qz = (rotation[1][0] - rotation[0][1]) / root
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        root = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2
        qw = (rotation[2][1] - rotation[1][2]) / root
        qx = 0.25 * root
        qy = (rotation[0][1] + rotation[1][0]) / root
        qz = (rotation[0][2] + rotation[2][0]) / root
    elif rotation[1][1] > rotation[2][2]:
        root = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2
        qw = (rotation[0][2] - rotation[2][0]) / root
        qx = (rotation[0][1] + rotation[1][0]) / root
        qy = 0.25 * root
        qz = (rotation[1][2] + rotation[2][1]) / root
    else:
        root = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2
        qw = (rotation[1][0] - rotation[0][1]) / root
        qx = (rotation[0][2] + rotation[2][0]) / root
        qy = (rotation[1][2] + rotation[2][1]) / root
        qz = 0.25 * root
    return qw, qx, qy, qz


def _gltf_instance_record(matrix: Sequence[float]) -> tuple[float, ...]:
    columns = [
        [matrix[row * 4 + column] for row in range(3)] for column in range(3)
    ]
    scales = [math.sqrt(sum(value * value for value in column)) for column in columns]
    if any(value <= 1e-12 or not math.isfinite(value) for value in scales):
        raise TiledViewerPackageError("Échelle composée dégénérée")
    determinant = (
        columns[0][0]
        * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
        - columns[1][0]
        * (columns[0][1] * columns[2][2] - columns[0][2] * columns[2][1])
        + columns[2][0]
        * (columns[0][1] * columns[1][2] - columns[0][2] * columns[1][1])
    )
    if determinant < 0:
        scales[0] = -scales[0]
    rotation_columns = [
        [value / scales[index] for value in column]
        for index, column in enumerate(columns)
    ]
    rotation = [
        [rotation_columns[column][row] for column in range(3)]
        for row in range(3)
    ]
    qw, qx, qy, qz = _rotation_quaternion(rotation)
    return (
        matrix[3],
        matrix[11],
        -matrix[7],
        qx,
        qz,
        -qy,
        qw,
        scales[0],
        scales[2],
        scales[1],
    )


def _scene_instances(
    scene_path: Path,
    *,
    tile_id: str,
    tile_origin: tuple[int, int],
    zone_origin: tuple[int, int, int],
    prototypes: Sequence[SealedPrototype],
    prototype_transforms: Mapping[str, Sequence[float]],
) -> tuple[dict[str, list[tuple[float, ...]]], dict[str, int]]:
    try:
        text = scene_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TiledViewerPackageError(f"scene.usda illisible pour {tile_id}") from error
    by_identifier = {
        (prototype.family, prototype.identifier): prototype
        for prototype in prototypes
    }
    groups = {prototype.prototype_id: [] for prototype in prototypes}
    counts = {family: 0 for family in FAMILY_ROOTS}
    observed_families: set[str] = set()
    for match in _POINT_INSTANCER.finditer(text):
        prim_name = match.group(1)
        family = _FAMILY_BY_PRIM.get(prim_name)
        if family is None:
            continue
        if family in observed_families:
            raise TiledViewerPackageError(f"PointInstancer dupliqué: {family}")
        observed_families.add(family)
        body = _block(text, match.end() - 1)
        targets = re.findall(r"<[^>]+/(Asset_[A-Za-z0-9_]+)>", _array(body, "rel prototypes"))
        identifiers = [value for value in targets]
        if len(identifiers) != len(set(identifiers)):
            raise TiledViewerPackageError("Cibles de prototypes USDA dupliquées")
        family_prototypes: list[SealedPrototype] = []
        for identifier in identifiers:
            prototype = by_identifier.get((family, identifier))
            if prototype is None:
                raise TiledViewerPackageError(
                    f"Prototype USDA absent du layout: {identifier}"
                )
            family_prototypes.append(prototype)
        ids = _numbers(_array(body, "int64[] ids"), integer=True)
        positions = _vectors(_array(body, "point3f[] positions"), 3, "positions")
        scales = _vectors(_array(body, "float3[] scales"), 3, "scales")
        orientations = _vectors(
            _array(body, "quath[] orientations"), 4, "orientations"
        )
        indices = _numbers(_array(body, "int[] protoIndices"), integer=True)
        size = len(ids)
        if not len(set(ids)) == size or not all(
            len(values) == size for values in (positions, scales, orientations, indices)
        ):
            raise TiledViewerPackageError(f"Tableaux PointInstancer incohérents: {family}")
        for position, scale, orientation, raw_index in zip(
            positions, scales, orientations, indices, strict=True
        ):
            if not 0 <= raw_index < len(family_prototypes):
                raise TiledViewerPackageError("protoIndex USDA hors limites")
            prototype = family_prototypes[raw_index]
            source_transform = prototype_transforms.get(prototype.prototype_id)
            if source_transform is None:
                raise TiledViewerPackageError(
                    f"Transform de prototype absent: {prototype.prototype_id}"
                )
            composed = _multiply(
                _quaternion_matrix(
                    (
                        tile_origin[0] - zone_origin[0] + position[0],
                        tile_origin[1] - zone_origin[1] + position[1],
                        position[2],
                    ),
                    scale,
                    orientation,
                ),
                source_transform,
            )
            record = _gltf_instance_record(composed)
            groups[prototype.prototype_id].append(record)
            counts[family] += 1
    if observed_families != set(FAMILY_ROOTS):
        raise TiledViewerPackageError(
            f"Familles PointInstancer incomplètes pour {tile_id}"
        )
    return groups, counts


@lru_cache(maxsize=8)
def _terrain_template(
    side: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache topology, source indices, XZ positions and UVs for fixed grids."""

    selected = np.concatenate(
        (np.arange(0, side - 1, stride, dtype=np.int64), np.array([side - 1]))
    )
    rows, columns = np.meshgrid(selected, selected, indexing="ij")
    source_indices = (rows * side + columns).reshape(-1)
    cell = TILE_SIZE_M / (side - 1)
    static_positions = np.column_stack(
        (
            columns.reshape(-1) * cell,
            -rows.reshape(-1) * cell,
        )
    ).astype("<f4", copy=False)
    texcoords = np.column_stack(
        (
            columns.reshape(-1) / (side - 1),
            1.0 - rows.reshape(-1) / (side - 1),
        )
    ).astype("<f4", copy=False)
    width = len(selected)
    row, column = np.meshgrid(
        np.arange(width - 1, dtype=np.uint32),
        np.arange(width - 1, dtype=np.uint32),
        indexing="ij",
    )
    northwest = (row * width + column).reshape(-1)
    northeast = northwest + 1
    southwest = northwest + width
    southeast = southwest + 1
    index_dtype = np.uint16 if width * width <= 65_535 else np.uint32
    indices = np.column_stack(
        (northwest, northeast, southeast, northwest, southeast, southwest)
    ).astype(index_dtype, copy=False).reshape(-1)
    for value in (source_indices, static_positions, texcoords, indices):
        value.setflags(write=False)
    return source_indices, static_positions, texcoords, indices


def _terrain_geometry(
    tile: FixedTerrainTile, *, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize the exact legacy terrain conversion without changing geometry."""

    mesh = tile.lods[0]
    source_indices, static_positions, texcoords, indices = _terrain_template(
        mesh.grid_size, stride
    )
    heights = np.asarray(mesh.relative_heights_mm, dtype=np.float64)[source_indices]
    source_normals = np.asarray(mesh.normals_snorm16, dtype=np.float64)[source_indices]
    lengths = np.linalg.norm(source_normals, axis=1)
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise TiledViewerPackageError("Normale FVTG dégénérée")
    positions = np.empty((len(source_indices), 3), dtype="<f4")
    positions[:, 0] = static_positions[:, 0]
    positions[:, 1] = heights / 1_000.0
    positions[:, 2] = static_positions[:, 1]
    normalized = source_normals / lengths[:, None]
    normals = np.column_stack(
        (normalized[:, 0], normalized[:, 2], -normalized[:, 1])
    ).astype("<f4", copy=False)
    return positions, normals, texcoords, indices


def _jpeg_thumbnail(path: Path) -> bytes:
    try:
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGB")
            image.thumbnail((96, 96), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=76, optimize=True)
            return output.getvalue()
    except OSError as error:
        raise TiledViewerPackageError(f"Orthophoto indécodable: {path}") from error


def _node_chain(
    tile_id: str,
    tile_origin: tuple[int, int],
    zone_origin: tuple[int, int, int],
    z_origin_mm: int,
) -> list[dict[str, Any]]:
    return [
        {
            "name": tile_id,
            "translation": [
                tile_origin[0] - zone_origin[0],
                z_origin_mm / 1_000.0,
                -(tile_origin[1] - zone_origin[1]),
            ],
        },
        {"name": f"{tile_id}_Terrain"},
    ]


def _validate_identity_prototype(path: Path) -> int:
    gltf, _binary = _read_glb(path)
    nodes = gltf.get("nodes")
    meshes = gltf.get("meshes")
    if not isinstance(nodes, list) or not isinstance(meshes, list) or not meshes:
        raise TiledViewerPackageError(f"Prototype GLB sans mesh: {path.name}")
    identity_matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    mesh_nodes = 0
    for raw in nodes:
        if not isinstance(raw, Mapping):
            raise TiledViewerPackageError("Nœud de prototype GLB invalide")
        if "mesh" in raw:
            mesh_nodes += 1
            if (
                raw.get("translation", [0, 0, 0]) != [0, 0, 0]
                or raw.get("rotation", [0, 0, 0, 1]) != [0, 0, 0, 1]
                or raw.get("scale", [1, 1, 1]) != [1, 1, 1]
                or raw.get("matrix", identity_matrix) != identity_matrix
            ):
                raise TiledViewerPackageError(
                    f"Transform interne non figé dans {path.name}"
                )
    if mesh_nodes == 0:
        raise TiledViewerPackageError(f"Prototype GLB sans nœud mesh: {path.name}")
    return mesh_nodes


def _prototype_cache_key(prototype: SealedPrototype, exporter_script: Path) -> str:
    safe_asset = re.sub(r"[^A-Za-z0-9_.-]+", "_", prototype.asset_id)
    safe_identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", prototype.identifier)
    return f"{prototype.family}-{safe_asset}-{safe_identifier}"


def _prototype_cache_root() -> Path | None:
    raw = os.environ.get("FIREVIEWER_TILED_PROTOTYPE_CACHE", "").strip()
    return Path(raw).resolve() if raw else None


def _tile_cache_root() -> Path | None:
    raw = os.environ.get("FIREVIEWER_TILED_TILE_CACHE", "").strip()
    return Path(raw).resolve() if raw else None


def _materialize_cached_file(source: Path, destination: Path) -> None:
    """Prefer a zero-copy hardlink and remain portable across filesystems."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _load_cached_prototype(
    cache_root: Path,
    prototype: SealedPrototype,
    exporter_script: Path,
    destination: Path,
) -> tuple[Asset, tuple[float, ...]] | None:
    cache_key = _prototype_cache_key(prototype, exporter_script)
    entry = cache_root / cache_key
    source = entry / "prototype.glb"
    receipt_path = entry / "prototype-cache.v1.json"
    if not source.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = _load_json(receipt_path, "reçu de cache prototype")
        raw_transform = receipt.get("prototype_transform_z_up")
        if (
            receipt.get("schema") != PROTOTYPE_CACHE_SCHEMA
            or receipt.get("cache_key") != cache_key
            or receipt.get("family") != prototype.family
            or receipt.get("asset_id") != prototype.asset_id
            or receipt.get("identifier") != prototype.identifier
            or receipt.get("byte_count") != source.stat().st_size
            or not isinstance(raw_transform, list)
            or len(raw_transform) != 16
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in raw_transform
            )
        ):
            return None
        _validate_identity_prototype(source)
        _materialize_cached_file(source, destination)
        return (
            _asset(destination, destination.parents[1], "model/gltf-binary"),
            tuple(float(value) for value in raw_transform),
        )
    except (OSError, TiledViewerPackageError):
        return None


def _store_cached_prototype(
    cache_root: Path,
    prototype: SealedPrototype,
    exporter_script: Path,
    source: Path,
    transform: Sequence[float],
) -> None:
    """Publish an immutable cache entry; cache failures never fail a map."""

    cache_key = _prototype_cache_key(prototype, exporter_script)
    entry = cache_root / cache_key
    try:
        entry.mkdir(parents=True, exist_ok=True)
        target = entry / "prototype.glb"
        temporary = entry / ".prototype.glb.part"
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        _write_json(
            entry / "prototype-cache.v1.json",
            {
                "schema": PROTOTYPE_CACHE_SCHEMA,
                "cache_key": cache_key,
                "family": prototype.family,
                "asset_id": prototype.asset_id,
                "identifier": prototype.identifier,
                "prototype_transform_z_up": [float(value) for value in transform],
                "byte_count": target.stat().st_size,
            },
        )
    except OSError:
        # A prebuilt image cache can intentionally be mounted read-only.  A
        # miss still falls back to Blender and remains a valid production run.
        return


def _export_prototypes(
    root: Path,
    staging: Path,
    blend_path: Path,
    prototypes: Sequence[SealedPrototype],
    blender: Path | str,
    timeout_seconds: int,
    metrics: dict[str, Any] | None = None,
) -> tuple[list[Asset], dict[str, tuple[float, ...]]]:
    started = time.perf_counter()
    plan_path = staging / PROTOTYPE_PLAN_NAME
    blend_reference = {
        "path": _relative_to_root(root, blend_path),
        "byte_count": blend_path.stat().st_size,
    }
    script = Path(__file__).with_name("export_tiled_viewer_prototypes.py")
    cache_root = _prototype_cache_root()
    assets_by_id: dict[str, Asset] = {}
    transforms: dict[str, tuple[float, ...]] = {}
    misses: list[SealedPrototype] = []
    for prototype in prototypes:
        destination = staging / "prototypes" / f"{prototype.prototype_id}.glb"
        cached = (
            _load_cached_prototype(
                cache_root, prototype, script, destination
            )
            if cache_root is not None
            else None
        )
        if cached is None:
            misses.append(prototype)
            continue
        assets_by_id[prototype.prototype_id] = cached[0]
        transforms[prototype.prototype_id] = cached[1]

    plan = {
        "schema": PROTOTYPE_PLAN_SCHEMA,
        "zone_blend": blend_reference,
        "prototypes": [
            {
                "id": prototype.prototype_id,
                "family": prototype.family,
                "asset_id": prototype.asset_id,
                "identifier": prototype.identifier,
                "output": (
                    staging.relative_to(root)
                    / "prototypes"
                    / f"{prototype.prototype_id}.glb"
                ).as_posix(),
            }
            for prototype in misses
        ],
    }
    _write_json(plan_path, plan)
    completed: Any | None = None
    if misses:
        blender_path = Path(blender)
        if not blender_path.is_file():
            raise TiledViewerPackageError(f"Blender absent: {blender_path}")
        command = [
            str(blender_path),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            "--job-root",
            str(root),
            "--plan",
            str(plan_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TiledViewerPackageError(
                f"Export des prototypes impossible: {error}"
            ) from error
        if completed.returncode != 0:
            details = "\n".join(
                (completed.stdout + completed.stderr).splitlines()[-40:]
            )
            raise TiledViewerPackageError(
                f"Export des prototypes en échec ({completed.returncode}):\n{details}"
            )
    else:
        _write_json(
            staging / PROTOTYPE_EXPORT_NAME,
            {
                "schema": PROTOTYPE_EXPORT_SCHEMA,
                "status": "complete",
                "zone_blend": blend_reference,
                "prototype_count": 0,
                "prototypes": [],
            },
        )
    receipt_path = staging / PROTOTYPE_EXPORT_NAME
    if not receipt_path.is_file():
        details = (
            "\n".join((completed.stdout + completed.stderr).splitlines()[-40:])
            if completed is not None
            else ""
        )
        raise TiledViewerPackageError(
            "Blender a terminé sans produire le reçu prototypes"
            + (f":\n{details}" if details else "")
        )
    receipt = _load_json(receipt_path, "reçu prototypes")
    rows = receipt.get("prototypes")
    if (
        receipt.get("schema") != PROTOTYPE_EXPORT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("zone_blend") != blend_reference
        or receipt.get("prototype_count") != len(misses)
        or not isinstance(rows, list)
        or len(rows) != len(misses)
    ):
        raise TiledViewerPackageError("Reçu d'export des prototypes invalide")
    by_id = {
        row.get("id"): row for row in rows if isinstance(row, Mapping)
    }
    for prototype in misses:
        row = by_id.get(prototype.prototype_id)
        path = staging / "prototypes" / f"{prototype.prototype_id}.glb"
        raw_transform = (
            row.get("prototype_transform_z_up")
            if isinstance(row, Mapping)
            else None
        )
        if (
            not isinstance(row, Mapping)
            or row.get("byte_count") != path.stat().st_size
            or not isinstance(raw_transform, list)
            or len(raw_transform) != 16
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in raw_transform
            )
            or any(
                abs(float(raw_transform[index]) - expected) > 1e-8
                for index, expected in ((12, 0), (13, 0), (14, 0), (15, 1))
            )
        ):
            raise TiledViewerPackageError(
                f"Prototype exporté divergent: {prototype.prototype_id}"
            )
        _validate_identity_prototype(path)
        asset = _asset(path, staging, "model/gltf-binary")
        transform = tuple(
            float(value) for value in raw_transform
        )
        assets_by_id[prototype.prototype_id] = asset
        transforms[prototype.prototype_id] = transform
        if cache_root is not None:
            _store_cached_prototype(
                cache_root, prototype, script, path, transform
            )
    if metrics is not None:
        fallbacks = [
            {
                "prototype_id": str(row.get("id")),
                "family": str(row.get("family")),
                "asset_id": str(row.get("asset_id")),
                "fallback": dict(row["fallback"]),
            }
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("fallback"), Mapping)
        ]
        metrics.update(
            {
                "cache_enabled": cache_root is not None,
                "cache_hits": len(prototypes) - len(misses),
                "cache_misses": len(misses),
                "procedural_fallback_count": len(fallbacks),
                "procedural_fallbacks": fallbacks,
                "duration_seconds": round(time.perf_counter() - started, 6),
            }
        )
    return [assets_by_id[item.prototype_id] for item in prototypes], transforms


def _validate_tile_sources(
    root: Path, package: Path, tile_id: str, record: Mapping[str, Any]
) -> tuple[Path, Path, Path, dict[str, Any]]:
    terrain_receipt_path = package / "fixed-terrain-usd.v1.json"
    scene_receipt_path = package / "scene" / "scene.done.json"
    terrain_receipt = _load_json(terrain_receipt_path, f"reçu terrain {tile_id}")
    scene_receipt = _load_json(scene_receipt_path, f"reçu scène {tile_id}")
    terrain_inputs = terrain_receipt.get("inputs")
    if not isinstance(terrain_inputs, Mapping):
        raise TiledViewerPackageError(
            f"Entrées du terrain scellé invalides pour {tile_id}"
        )
    terrain = _validate_asset_reference(
        root, package, terrain_inputs.get("terrain"), f"FVTG {tile_id}"
    )
    image = _validate_asset_reference(
        root,
        package,
        terrain_inputs.get("ground_color"),
        f"orthophoto {tile_id}",
    )
    scene = _validate_asset_reference(
        root, package / "scene", scene_receipt.get("scene"), f"scène {tile_id}"
    )
    reconciliation = scene_receipt.get("reconciliation")
    sealed_counts = {
        "buildings": record.get("building_count"),
        "trees": record.get("tree_count"),
        "context_assets": record.get("context_asset_count", 0),
    }
    if (
        terrain_receipt.get("schema") != "fireviewer.fixed-terrain-usd-package.v1"
        or terrain_receipt.get("status") != "compiled"
        or terrain_receipt.get("tile_id") != tile_id
        or not isinstance(reconciliation, Mapping)
        or any(
            not isinstance(reconciliation.get(family), Mapping)
            or reconciliation[family].get("instance_count") != count
            for family, count in sealed_counts.items()
        )
    ):
        raise TiledViewerPackageError(f"Sources scellées divergentes pour {tile_id}")
    source_identity = {
        "tile_id": tile_id,
        "origin_l93_m": list(record["origin_l93_m"]),
        "family_counts": sealed_counts,
        "terrain_bytes": terrain.stat().st_size,
        "terrain_mtime_ns": terrain.stat().st_mtime_ns,
        "image_bytes": image.stat().st_size,
        "image_mtime_ns": image.stat().st_mtime_ns,
        "scene_bytes": scene.stat().st_size,
        "scene_mtime_ns": scene.stat().st_mtime_ns,
    }
    return terrain, image, scene, source_identity


def _tile_cache_key(
    *,
    tile_id: str,
    source_identity: Mapping[str, Any],
    origin: tuple[int, int, int],
    prototypes: Sequence[SealedPrototype],
    prototype_transforms: Mapping[str, Sequence[float]],
) -> str:
    return (
        f"{tile_id}-{source_identity['scene_mtime_ns']}-"
        f"{source_identity['terrain_bytes']}-{source_identity['image_bytes']}"
    )


def _cached_asset(
    source: Path,
    destination: Path,
    staging: Path,
    media_type: str,
) -> Asset:
    _materialize_cached_file(source, destination)
    return _asset(destination, staging, media_type)


def _load_cached_tile(
    cache_root: Path,
    *,
    cache_key: str,
    tile_id: str,
    source_identity: Mapping[str, Any],
    staging: Path,
) -> CompiledTile | None:
    entry = cache_root / cache_key
    receipt_path = entry / "tile-cache.v1.json"
    terrain_source = entry / "terrain.glb"
    instances_source = entry / "instances.fvi"
    far_source = entry / "far.npz"
    image_source = entry / "far.jpg"
    if not all(
        path.is_file()
        for path in (
            receipt_path,
            terrain_source,
            instances_source,
            far_source,
            image_source,
        )
    ):
        return None
    restore_started = time.perf_counter()
    try:
        receipt = _load_json(receipt_path, "reçu cache tuile")
        if (
            receipt.get("schema") != TILE_CACHE_SCHEMA
            or receipt.get("cache_key") != cache_key
            or receipt.get("tile_id") != tile_id
            or receipt.get("source_identity") != dict(source_identity)
        ):
            return None
        terrain_asset = _cached_asset(
            terrain_source,
            staging / "tiles" / tile_id / "terrain.glb",
            staging,
            "model/gltf-binary",
        )
        instance_asset = _cached_asset(
            instances_source,
            staging / "tiles" / tile_id / "instances.fvi",
            staging,
            "application/vnd.fireviewer.instances",
        )
        with np.load(far_source, allow_pickle=False) as far:
            positions = np.asarray(far["positions"])
            normals = np.asarray(far["normals"])
            texcoords = np.asarray(far["texcoords"])
            indices = np.asarray(far["indices"])
        family_counts = receipt.get("family_counts")
        prototype_counts = receipt.get("prototype_instance_counts")
        prototype_ids = receipt.get("prototype_ids")
        node_chain = receipt.get("node_chain")
        tile_origin = receipt.get("tile_origin")
        if (
            not isinstance(family_counts, dict)
            or not isinstance(prototype_counts, dict)
            or not isinstance(prototype_ids, list)
            or not isinstance(node_chain, list)
            or not isinstance(tile_origin, list)
            or len(tile_origin) != 2
        ):
            return None
        elapsed = time.perf_counter() - restore_started
        return CompiledTile(
            tile_id=tile_id,
            tile_origin=(int(tile_origin[0]), int(tile_origin[1])),
            terrain_asset=terrain_asset,
            instance_asset=instance_asset,
            family_counts={str(key): int(value) for key, value in family_counts.items()},
            prototype_instance_counts={
                str(key): int(value) for key, value in prototype_counts.items()
            },
            prototype_ids=tuple(str(value) for value in prototype_ids),
            far_positions=positions,
            far_normals=normals,
            far_texcoords=texcoords,
            far_indices=indices,
            far_image=image_source.read_bytes(),
            node_chain=tuple(dict(value) for value in node_chain),
            timings_seconds={
                "source_validation": 0.0,
                "cache_restore": round(elapsed, 6),
                "terrain_glb": 0.0,
                "far_fragment": 0.0,
                "instances_fvi": 0.0,
                "output_hashing": 0.0,
                "total": round(elapsed, 6),
            },
            cache_hit=True,
        )
    except (KeyError, OSError, TypeError, ValueError, TiledViewerPackageError):
        return None


def _store_cached_tile(
    cache_root: Path,
    *,
    cache_key: str,
    source_identity: Mapping[str, Any],
    staging: Path,
    compiled: CompiledTile,
) -> None:
    """Write immutable tile artefacts and publish the receipt last."""

    entry = cache_root / cache_key
    try:
        entry.mkdir(parents=True, exist_ok=True)
        sources = {
            "terrain": staging.joinpath(*compiled.terrain_asset.path.split("/")),
            "instances": staging.joinpath(*compiled.instance_asset.path.split("/")),
        }
        targets = {
            "terrain": entry / "terrain.glb",
            "instances": entry / "instances.fvi",
        }
        for name, source in sources.items():
            temporary = entry / f".{targets[name].name}.part"
            if temporary.exists():
                temporary.unlink()
            _materialize_cached_file(source, temporary)
            os.replace(temporary, targets[name])
        far_target = entry / "far.npz"
        far_temporary = entry / ".far.npz.part"
        with far_temporary.open("wb") as stream:
            np.savez(
                stream,
                positions=np.asarray(compiled.far_positions),
                normals=np.asarray(compiled.far_normals),
                texcoords=np.asarray(compiled.far_texcoords),
                indices=np.asarray(compiled.far_indices),
            )
        os.replace(far_temporary, far_target)
        image_target = entry / "far.jpg"
        image_temporary = entry / ".far.jpg.part"
        image_temporary.write_bytes(compiled.far_image)
        os.replace(image_temporary, image_target)
        _write_json(
            entry / "tile-cache.v1.json",
            {
                "schema": TILE_CACHE_SCHEMA,
                "cache_key": cache_key,
                "tile_id": compiled.tile_id,
                "source_identity": dict(source_identity),
                "tile_origin": list(compiled.tile_origin),
                "family_counts": compiled.family_counts,
                "prototype_instance_counts": compiled.prototype_instance_counts,
                "prototype_ids": list(compiled.prototype_ids),
                "node_chain": list(compiled.node_chain),
            },
        )
    except (OSError, TypeError, ValueError):
        return


def _viewer_tile_workers(tile_count: int) -> int:
    raw = os.environ.get("FIREVIEWER_VIEWER_TILE_WORKERS", "").strip()
    try:
        configured = int(raw) if raw else min(8, os.cpu_count() or 1)
    except ValueError as error:
        raise TiledViewerPackageError(
            "FIREVIEWER_VIEWER_TILE_WORKERS doit être un entier"
        ) from error
    if configured < 1 or configured > 8:
        raise TiledViewerPackageError(
            "FIREVIEWER_VIEWER_TILE_WORKERS doit être compris entre 1 et 8"
        )
    return min(configured, max(1, tile_count))


def _compile_sealed_tile(
    arguments: tuple[
        Path,
        Path,
        Path,
        str,
        dict[str, Any],
        tuple[int, int, int],
        tuple[SealedPrototype, ...],
        dict[str, tuple[float, ...]],
        Path | None,
    ]
) -> CompiledTile:
    (
        root,
        staging,
        packages,
        tile_id,
        record,
        origin,
        prototypes,
        prototype_transforms,
        cache_root,
    ) = arguments
    total_started = time.perf_counter()
    phase_started = total_started
    tile_origin = tuple(int(value) for value in record["origin_l93_m"])
    package = packages / tile_id
    terrain_path, image_path, scene_path, source_identity = _validate_tile_sources(
        root, package, tile_id, record
    )
    source_validation_seconds = time.perf_counter() - phase_started
    cache_key = _tile_cache_key(
        tile_id=tile_id,
        source_identity=source_identity,
        origin=origin,
        prototypes=prototypes,
        prototype_transforms=prototype_transforms,
    )
    if cache_root is not None:
        cached = _load_cached_tile(
            cache_root,
            cache_key=cache_key,
            tile_id=tile_id,
            source_identity=source_identity,
            staging=staging,
        )
        if cached is not None:
            cached.timings_seconds["source_validation"] = round(
                source_validation_seconds, 6
            )
            cached.timings_seconds["total"] = round(
                time.perf_counter() - total_started, 6
            )
            return cached

    phase_started = time.perf_counter()
    fixed = read_fixed_terrain(terrain_path)
    if fixed.tile_origin_mm != (
        tile_origin[0] * 1_000,
        tile_origin[1] * 1_000,
    ):
        raise TiledViewerPackageError(f"Origine FVTG divergente: {tile_id}")
    terrain_geometry = _terrain_geometry(fixed)
    chain = tuple(_node_chain(tile_id, tile_origin, origin, fixed.z_origin_mm))
    terrain_output = staging / "tiles" / tile_id / "terrain.glb"
    terrain_builder = _FarGlb(generator="FireViewer sealed terrain tile")
    terrain_builder.add_tile(
        tile_id=tile_id,
        node_chain=chain,
        positions=terrain_geometry[0],
        normals=terrain_geometry[1],
        texcoords=terrain_geometry[2],
        indices=terrain_geometry[3],
        image=image_path.read_bytes(),
        image_mime="image/png",
    )
    terrain_builder.write(terrain_output)
    terrain_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    far_geometry = _terrain_geometry(fixed, stride=8)
    far_image = _jpeg_thumbnail(image_path)
    far_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    groups, counts = _scene_instances(
        scene_path,
        tile_id=tile_id,
        tile_origin=tile_origin,
        zone_origin=origin,
        prototypes=prototypes,
        prototype_transforms=prototype_transforms,
    )
    sealed_counts = {
        "buildings": record.get("building_count"),
        "trees": record.get("tree_count"),
        "context_assets": record.get("context_asset_count", 0),
    }
    if counts != sealed_counts:
        raise TiledViewerPackageError(
            f"Comptages d'instances divergents pour {tile_id}: "
            f"attendu={sealed_counts}, obtenu={counts}"
        )
    instance_groups: list[tuple[Prototype, Sequence[tuple[float, ...]]]] = []
    prototype_ids: list[str] = []
    prototype_instance_counts: dict[str, int] = {}
    for prototype in prototypes:
        records = groups[prototype.prototype_id]
        if not records:
            continue
        instance_groups.append(
            (
                Prototype(
                    prototype.prototype_id,
                    prototype.family,
                    -1,
                    (),
                ),
                records,
            )
        )
        prototype_ids.append(prototype.prototype_id)
        prototype_instance_counts[prototype.prototype_id] = len(records)
    instance_output = staging / "tiles" / tile_id / "instances.fvi"
    _write_instances(instance_output, tile_id=tile_id, groups=instance_groups)
    instances_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    terrain_asset = _asset(terrain_output, staging, "model/gltf-binary")
    instance_asset = _asset(
        instance_output,
        staging,
        "application/vnd.fireviewer.instances",
    )
    metadata_seconds = time.perf_counter() - phase_started
    compiled = CompiledTile(
        tile_id=tile_id,
        tile_origin=tile_origin,
        terrain_asset=terrain_asset,
        instance_asset=instance_asset,
        family_counts=counts,
        prototype_instance_counts=prototype_instance_counts,
        prototype_ids=tuple(prototype_ids),
        far_positions=far_geometry[0],
        far_normals=far_geometry[1],
        far_texcoords=far_geometry[2],
        far_indices=far_geometry[3],
        far_image=far_image,
        node_chain=chain,
        timings_seconds={
            "source_validation": round(source_validation_seconds, 6),
            "cache_restore": 0.0,
            "terrain_glb": round(terrain_seconds, 6),
            "far_fragment": round(far_seconds, 6),
            "instances_fvi": round(instances_seconds, 6),
            "output_metadata": round(metadata_seconds, 6),
            "total": round(time.perf_counter() - total_started, 6),
        },
        cache_hit=False,
    )
    if cache_root is not None:
        _store_cached_tile(
            cache_root,
            cache_key=cache_key,
            source_identity=source_identity,
            staging=staging,
            compiled=compiled,
        )
    return compiled


def _compile_sealed_tile_safe(
    arguments: tuple[
        Path,
        Path,
        Path,
        str,
        dict[str, Any],
        tuple[int, int, int],
        tuple[SealedPrototype, ...],
        dict[str, tuple[float, ...]],
        Path | None,
    ]
) -> CompiledTile | FailedTile:
    try:
        return _compile_sealed_tile(arguments)
    except Exception as error:
        return FailedTile(arguments[3], f"{type(error).__name__}: {error}")


def _summarize_tile_timings(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    result: dict[str, dict[str, float]] = {}
    for phase in rows[0]:
        values = sorted(float(row[phase]) for row in rows)

        def percentile(fraction: float) -> float:
            index = min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))
            return values[index]

        result[phase] = {
            "sum": round(sum(values), 6),
            "mean": round(sum(values) / len(values), 6),
            "p50": round(percentile(0.50), 6),
            "p95": round(percentile(0.95), 6),
            "max": round(values[-1], 6),
        }
    return result


def build_tiled_viewer_from_sealed(
    job_root: Path | str,
    *,
    blender: Path | str,
    timeout_seconds: int = 1_800,
) -> Path:
    """Build the complete tiled viewer without creating a monolithic viewer."""

    root = Path(job_root).resolve(strict=True)
    zone, layout_path, layout, blend_path, packages = _load_sealed_sources(root)
    expected = _expected_counts(zone)
    zone_tiles = _tile_records(zone)
    prototypes = _sealed_prototypes(layout)
    west = min(int(record["origin_l93_m"][0]) for record in zone_tiles.values())
    south = min(int(record["origin_l93_m"][1]) for record in zone_tiles.values())
    east = max(
        int(record["origin_l93_m"][0]) + TILE_SIZE_M
        for record in zone_tiles.values()
    )
    north = max(
        int(record["origin_l93_m"][1]) + TILE_SIZE_M
        for record in zone_tiles.values()
    )
    origin = (west, south, 0)
    staging = root / f".{OUTPUT_DIRECTORY}.part"
    output = root / OUTPUT_DIRECTORY
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    build_started = time.perf_counter()
    print(
        "FIREVIEWER_VIEWER_PACK_PROGRESS "
        + json.dumps(
            {"phase": "start", "tile_count": len(zone_tiles)},
            separators=(",", ":"),
        ),
        flush=True,
    )
    try:
        prototype_metrics: dict[str, Any] = {}
        prototype_assets, prototype_transforms = _export_prototypes(
            root,
            staging,
            blend_path,
            prototypes,
            blender,
            timeout_seconds,
            prototype_metrics,
        )
        print(
            "FIREVIEWER_VIEWER_PACK_PROGRESS "
            + json.dumps(
                {
                    "phase": "prototypes_complete",
                    "prototype_count": len(prototypes),
                    "elapsed_seconds": round(time.perf_counter() - build_started, 3),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        (staging / PROTOTYPE_PLAN_NAME).unlink()
        (staging / PROTOTYPE_EXPORT_NAME).unlink()
        prototype_counts = {prototype.prototype_id: 0 for prototype in prototypes}
        total_counts = {family: 0 for family in FAMILY_ROOTS}
        tile_assets: list[Asset] = []
        tile_rows: list[dict[str, Any]] = []
        tile_timings: list[dict[str, float]] = []
        compiled_tile_count = 0
        tile_cache_hits = 0
        far_builder = _FarGlb(
            generator="FireViewer sealed FAR LOD",
            binary_spool=staging / ".far-binary.part",
        )
        sorted_tiles = sorted(
            zone_tiles.items(),
            key=lambda item: (item[1]["origin_l93_m"][1], item[1]["origin_l93_m"][0]),
        )
        tile_workers = _viewer_tile_workers(len(sorted_tiles))
        tile_cache = _tile_cache_root()
        tile_arguments = [
            (
                root,
                staging,
                packages,
                tile_id,
                dict(record),
                origin,
                tuple(prototypes),
                dict(prototype_transforms),
                tile_cache,
            )
            for tile_id, record in sorted_tiles
        ]

        def consume_compiled_tile(compiled: CompiledTile) -> None:
            nonlocal compiled_tile_count, tile_cache_hits
            far_builder.add_tile(
                tile_id=compiled.tile_id,
                node_chain=compiled.node_chain,
                positions=compiled.far_positions,
                normals=compiled.far_normals,
                texcoords=compiled.far_texcoords,
                indices=compiled.far_indices,
                image=compiled.far_image,
            )
            tile_origin = compiled.tile_origin
            sealed_counts = {
                "buildings": compiled.family_counts["buildings"],
                "trees": compiled.family_counts["trees"],
                "context_assets": compiled.family_counts["context_assets"],
            }
            for prototype in prototypes:
                instance_count = compiled.prototype_instance_counts.get(
                    prototype.prototype_id, 0
                )
                if instance_count == 0:
                    continue
                prototype_counts[prototype.prototype_id] += instance_count
                total_counts[prototype.family] += instance_count
            tile_assets.extend((compiled.terrain_asset, compiled.instance_asset))
            tile_timings.append(compiled.timings_seconds)
            compiled_tile_count += 1
            tile_cache_hits += int(compiled.cache_hit)
            tile_rows.append(
                {
                    "id": compiled.tile_id,
                    "bounds_l93_m": [
                        tile_origin[0],
                        tile_origin[1],
                        tile_origin[0] + TILE_SIZE_M,
                        tile_origin[1] + TILE_SIZE_M,
                    ],
                    "terrain": compiled.terrain_asset.payload(),
                    "instances": compiled.instance_asset.payload(),
                    "family_instance_counts": compiled.family_counts,
                    "source_family_instance_counts": sealed_counts,
                    "prototype_ids": list(compiled.prototype_ids),
                }
            )
            if compiled_tile_count % 8 == 0 or compiled_tile_count == len(sorted_tiles):
                print(
                    "FIREVIEWER_VIEWER_PACK_PROGRESS "
                    + json.dumps(
                        {
                            "phase": "tiles",
                            "completed": compiled_tile_count,
                            "total": len(sorted_tiles),
                            "cache_hits": tile_cache_hits,
                            "workers": tile_workers,
                            "elapsed_seconds": round(
                                time.perf_counter() - build_started, 3
                            ),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

        tile_compile_started = time.perf_counter()
        if tile_workers == 1:
            for arguments in tile_arguments:
                consume_compiled_tile(_compile_sealed_tile(arguments))
        else:
            with ProcessPoolExecutor(max_workers=tile_workers) as executor:
                for arguments, result in zip(
                    tile_arguments,
                    executor.map(_compile_sealed_tile_safe, tile_arguments),
                    strict=True,
                ):
                    if isinstance(result, FailedTile):
                        print(
                            "FIREVIEWER_VIEWER_PACK_PROGRESS "
                            + json.dumps(
                                {
                                    "phase": "tile_retry",
                                    "tile_id": result.tile_id,
                                    "reason": result.error,
                                },
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                        result = _compile_sealed_tile(arguments)
                    consume_compiled_tile(result)
        tile_compile_wall_seconds = time.perf_counter() - tile_compile_started
        if total_counts != expected:
            raise TiledViewerPackageError(
                f"Comptages tuilés incomplets: attendu={expected}, obtenu={total_counts}"
            )
        far_write_started = time.perf_counter()
        far_path = staging / "far.glb"
        far_builder.write(far_path)
        far_asset = _asset(far_path, staging, "model/gltf-binary")
        far_write_seconds = time.perf_counter() - far_write_started
        prototype_rows = [
            {
                "id": prototype.prototype_id,
                "family": prototype.family,
                "asset_id": prototype.asset_id,
                "instance_count": prototype_counts[prototype.prototype_id],
                "asset": asset.payload(),
            }
            for prototype, asset in zip(prototypes, prototype_assets, strict=True)
        ]
        all_assets = [far_asset, *prototype_assets, *tile_assets]
        source = {
            "kind": SEALED_SOURCE_KIND,
            "zone_receipt": {
                "path": "zone.done.json",
                "byte_count": (root / "zone.done.json").stat().st_size,
            },
            "stage_layout": {
                "path": _relative_to_root(root, layout_path),
                "byte_count": layout_path.stat().st_size,
            },
            "zone_blend": {
                "path": _relative_to_root(root, blend_path),
                "byte_count": blend_path.stat().st_size,
            },
            "tile_package_count": len(zone_tiles),
        }
        catalog = {
            "schema": SCHEMA,
            "catalog_version": 1,
            "crs": "EPSG:2154",
            "linear_unit": "metre",
            "coordinate_frame": "gltf-y-up-local",
            "origin_l93_m": list(origin),
            "bounds_l93_m": [west, south, east, north],
            "tile_size_m": TILE_SIZE_M,
            "loading": {
                "detail_publish_distance_m": 2_200,
                "detail_preload_radius_m": 900,
                "maximum_resident_tile_count": 64,
                "maximum_concurrent_requests": 4,
                "terrain_before_instances": True,
            },
            "far": {
                "role": "navigation_lod_not_counted_as_canonical_detail",
                "asset": far_asset.payload(),
            },
            "canonical": {
                "representation": "complete_non_simplified_map",
                "policy": "fail_closed_exact_visual_scene",
                "source": source,
                "family_instance_counts": expected,
            },
            "prototype_count": len(prototypes),
            "tile_count": len(tile_rows),
            "payload_file_count": len(all_assets),
            "payload_byte_count": sum(asset.byte_count for asset in all_assets),
            "prototypes": prototype_rows,
            "tiles": tile_rows,
        }
        catalog_path = staging / CATALOG_NAME
        _write_json(catalog_path, catalog)
        catalog_asset = _asset(catalog_path, staging, "application/json")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "complete",
            "zone_id": zone.get("zone_id"),
            "build_id": zone.get("build_id"),
            "representation": "complete_tiled_non_simplified_map",
            "catalog": catalog_asset.payload(),
            "payload_file_count": len(all_assets),
            "payload_byte_count": sum(asset.byte_count for asset in all_assets),
            "family_instance_counts": expected,
            "source": source,
        }
        _write_json(staging / RECEIPT_NAME, receipt)
        _write_json(
            staging / BUILD_METRICS_NAME,
            {
                "schema": BUILD_METRICS_SCHEMA,
                "tile_count": compiled_tile_count,
                "tile_workers": tile_workers,
                "streaming_result_consumption": True,
                "tile_cache": {
                    "enabled": tile_cache is not None,
                    "hits": tile_cache_hits,
                    "misses": compiled_tile_count - tile_cache_hits,
                },
                "prototype_export": prototype_metrics,
                "phase_times_seconds": {
                    "tile_compile_wall": round(tile_compile_wall_seconds, 6),
                    "far_write": round(far_write_seconds, 6),
                    "total_before_publish": round(
                        time.perf_counter() - build_started, 6
                    ),
                },
                "tile_phase_statistics_seconds": _summarize_tile_timings(
                    tile_timings
                ),
            },
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / RECEIPT_NAME


__all__ = ["SEALED_SOURCE_KIND", "build_tiled_viewer_from_sealed"]
