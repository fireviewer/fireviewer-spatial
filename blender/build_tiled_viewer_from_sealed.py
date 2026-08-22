"""Build browser viewer tiles from sealed scientific zone artefacts."""

from __future__ import annotations

import io
import hashlib
import math
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    _canonical_bytes,
    _expected_counts,
    _load_json,
    _read_glb,
    _sha256_file,
    _tile_records,
    _write_instances,
    _write_json,
)
from fixed_terrain_grid import FixedTerrainTile, read_fixed_terrain

LAYOUT_SCHEMA = "fireviewer.compact-zone-stage-layout.v1"
SEALED_SOURCE_KIND = "sealed_zone_and_tile_packages"
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
    identity_sha256: str


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
        or raw.get("sha256") != _sha256_file(path)
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
        or stage_reference.get("sha256") != _sha256_file(layout_path)
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
        identity_sha256 = raw.get("identity_sha256")
        if (
            family not in counters
            or not isinstance(asset_id, str)
            or not asset_id
            or not isinstance(identifier, str)
            or _USD_IDENTIFIER.fullmatch(identifier) is None
            or (family, identifier) in identifiers
            or not isinstance(identity_sha256, str)
            or len(identity_sha256) != 64
        ):
            raise TiledViewerPackageError("Identité de prototype scellé invalide")
        prototype_id = f"{family}-{counters[family]:04d}"
        counters[family] += 1
        identifiers.add((family, identifier))
        result.append(
            SealedPrototype(
                prototype_id, family, asset_id, identifier, identity_sha256
            )
        )
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


def _terrain_geometry(
    tile: FixedTerrainTile, *, stride: int = 1
) -> tuple[
    list[tuple[float, ...]],
    list[tuple[float, ...]],
    list[tuple[float, ...]],
    list[int],
]:
    mesh = tile.lods[0]
    side = mesh.grid_size
    selected = list(range(0, side - 1, stride)) + [side - 1]
    positions: list[tuple[float, ...]] = []
    normals: list[tuple[float, ...]] = []
    texcoords: list[tuple[float, ...]] = []
    cell = TILE_SIZE_M / (side - 1)
    for row in selected:
        for column in selected:
            source_index = row * side + column
            nx, ny, nz = mesh.normals_snorm16[source_index]
            normal_length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if normal_length <= 0:
                raise TiledViewerPackageError("Normale FVTG dégénérée")
            positions.append(
                (
                    column * cell,
                    mesh.relative_heights_mm[source_index] / 1_000.0,
                    -row * cell,
                )
            )
            normals.append(
                (nx / normal_length, nz / normal_length, -ny / normal_length)
            )
            texcoords.append((column / (side - 1), 1.0 - row / (side - 1)))
    width = len(selected)
    indices: list[int] = []
    for row in range(width - 1):
        for column in range(width - 1):
            northwest = row * width + column
            northeast = northwest + 1
            southwest = northwest + width
            southeast = southwest + 1
            indices.extend(
                (northwest, northeast, southeast, northwest, southeast, southwest)
            )
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


def _export_prototypes(
    root: Path,
    staging: Path,
    blend_path: Path,
    prototypes: Sequence[SealedPrototype],
    blender: Path | str,
    timeout_seconds: int,
) -> tuple[list[Asset], dict[str, tuple[float, ...]]]:
    plan_path = staging / PROTOTYPE_PLAN_NAME
    blend_reference = {
        "path": _relative_to_root(root, blend_path),
        "sha256": _sha256_file(blend_path),
        "byte_count": blend_path.stat().st_size,
    }
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
            for prototype in prototypes
        ],
    }
    _write_json(plan_path, plan)
    blender_path = Path(blender)
    if not blender_path.is_file():
        raise TiledViewerPackageError(f"Blender absent: {blender_path}")
    script = Path(__file__).with_name("export_tiled_viewer_prototypes.py")
    command = [
        str(blender_path),
        "--background",
        "--factory-startup",
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
        raise TiledViewerPackageError(f"Export des prototypes impossible: {error}") from error
    if completed.returncode != 0:
        details = "\n".join((completed.stdout + completed.stderr).splitlines()[-40:])
        raise TiledViewerPackageError(
            f"Export des prototypes en échec ({completed.returncode}):\n{details}"
        )
    receipt = _load_json(staging / PROTOTYPE_EXPORT_NAME, "reçu prototypes")
    rows = receipt.get("prototypes")
    if (
        receipt.get("schema") != PROTOTYPE_EXPORT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("zone_blend") != blend_reference
        or receipt.get("prototype_count") != len(prototypes)
        or not isinstance(rows, list)
        or len(rows) != len(prototypes)
    ):
        raise TiledViewerPackageError("Reçu d'export des prototypes invalide")
    by_id = {
        row.get("id"): row for row in rows if isinstance(row, Mapping)
    }
    assets: list[Asset] = []
    transforms: dict[str, tuple[float, ...]] = {}
    for prototype in prototypes:
        row = by_id.get(prototype.prototype_id)
        path = staging / "prototypes" / f"{prototype.prototype_id}.glb"
        raw_transform = (
            row.get("prototype_transform_z_up")
            if isinstance(row, Mapping)
            else None
        )
        if (
            not isinstance(row, Mapping)
            or row.get("sha256") != _sha256_file(path)
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
        assets.append(_asset(path, staging, "model/gltf-binary"))
        transforms[prototype.prototype_id] = tuple(
            float(value) for value in raw_transform
        )
    return assets, transforms


def _validate_tile_sources(
    root: Path, package: Path, tile_id: str, record: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    terrain_receipt = _load_json(
        package / "fixed-terrain-usd.v1.json", f"reçu terrain {tile_id}"
    )
    scene_receipt = _load_json(
        package / "scene" / "scene.done.json", f"reçu scène {tile_id}"
    )
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
    return terrain, image, scene


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
    try:
        prototype_assets, prototype_transforms = _export_prototypes(
            root, staging, blend_path, prototypes, blender, timeout_seconds
        )
        (staging / PROTOTYPE_PLAN_NAME).unlink()
        (staging / PROTOTYPE_EXPORT_NAME).unlink()
        prototype_counts = {prototype.prototype_id: 0 for prototype in prototypes}
        total_counts = {family: 0 for family in FAMILY_ROOTS}
        tile_assets: list[Asset] = []
        tile_rows: list[dict[str, Any]] = []
        far_builder = _FarGlb(generator="FireViewer sealed FAR LOD")
        for tile_id, record in sorted(
            zone_tiles.items(),
            key=lambda item: (item[1]["origin_l93_m"][1], item[1]["origin_l93_m"][0]),
        ):
            tile_origin = tuple(int(value) for value in record["origin_l93_m"])
            package = packages / tile_id
            terrain_path, image_path, scene_path = _validate_tile_sources(
                root, package, tile_id, record
            )
            fixed = read_fixed_terrain(terrain_path)
            if fixed.tile_origin_mm != (
                tile_origin[0] * 1_000,
                tile_origin[1] * 1_000,
            ):
                raise TiledViewerPackageError(f"Origine FVTG divergente: {tile_id}")
            terrain_geometry = _terrain_geometry(fixed)
            chain = _node_chain(
                tile_id, tile_origin, origin, fixed.z_origin_mm
            )
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
            far_geometry = _terrain_geometry(fixed, stride=8)
            far_builder.add_tile(
                tile_id=tile_id,
                node_chain=chain,
                positions=far_geometry[0],
                normals=far_geometry[1],
                texcoords=far_geometry[2],
                indices=far_geometry[3],
                image=_jpeg_thumbnail(image_path),
            )
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
            instance_groups: list[
                tuple[Prototype, Sequence[tuple[float, ...]]]
            ] = []
            prototype_ids: list[str] = []
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
                prototype_counts[prototype.prototype_id] += len(records)
                total_counts[prototype.family] += len(records)
            instance_output = staging / "tiles" / tile_id / "instances.fvi"
            _write_instances(
                instance_output, tile_id=tile_id, groups=instance_groups
            )
            terrain_asset = _asset(
                terrain_output, staging, "model/gltf-binary"
            )
            instance_asset = _asset(
                instance_output,
                staging,
                "application/vnd.fireviewer.instances",
            )
            tile_assets.extend((terrain_asset, instance_asset))
            tile_rows.append(
                {
                    "id": tile_id,
                    "bounds_l93_m": [
                        tile_origin[0],
                        tile_origin[1],
                        tile_origin[0] + TILE_SIZE_M,
                        tile_origin[1] + TILE_SIZE_M,
                    ],
                    "terrain": terrain_asset.payload(),
                    "instances": instance_asset.payload(),
                    "family_instance_counts": counts,
                    "source_family_instance_counts": sealed_counts,
                    "prototype_ids": prototype_ids,
                }
            )
        if total_counts != expected:
            raise TiledViewerPackageError(
                f"Comptages tuilés incomplets: attendu={expected}, obtenu={total_counts}"
            )
        far_path = staging / "far.glb"
        far_builder.write(far_path)
        far_asset = _asset(far_path, staging, "model/gltf-binary")
        prototype_rows = [
            {
                "id": prototype.prototype_id,
                "family": prototype.family,
                "asset_id": prototype.asset_id,
                "identity_sha256": prototype.identity_sha256,
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
                "sha256": _sha256_file(root / "zone.done.json"),
                "byte_count": (root / "zone.done.json").stat().st_size,
            },
            "stage_layout": {
                "path": _relative_to_root(root, layout_path),
                "sha256": _sha256_file(layout_path),
                "byte_count": layout_path.stat().st_size,
            },
            "zone_blend": {
                "path": _relative_to_root(root, blend_path),
                "sha256": _sha256_file(blend_path),
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
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_bytes(receipt)
        ).hexdigest()
        _write_json(staging / RECEIPT_NAME, receipt)
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / RECEIPT_NAME


__all__ = ["SEALED_SOURCE_KIND", "build_tiled_viewer_from_sealed"]
