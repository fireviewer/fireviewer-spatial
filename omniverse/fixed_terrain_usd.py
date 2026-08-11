"""Portable OpenUSD export for one fixed FireViewer terrain tile.

The exporter deliberately keeps the runtime representation small: one FVTG
height source, one RGB ground texture, one PreviewSurface material and three
regular LOD payloads.  It does not depend on an atlas, a custom shader, or a
source imagery file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

try:
    from fixed_terrain_grid import FixedLodMesh, FixedTerrainTile, read_fixed_terrain
except ModuleNotFoundError:  # pragma: no cover - package-style import
    _BLENDER_ROOT = Path(__file__).resolve().parents[1] / "blender"
    if str(_BLENDER_ROOT) not in sys.path:
        sys.path.insert(0, str(_BLENDER_ROOT))
    from fixed_terrain_grid import FixedLodMesh, FixedTerrainTile, read_fixed_terrain


CONTRACT_SCHEMA = "fireviewer.fixed-terrain-usd-contract.v1"
PACKAGE_SCHEMA = "fireviewer.fixed-terrain-usd-package.v1"
GROUND_COLOR_SCHEMA = "fireviewer.orthophoto-ground-texture-tile.v1"
CRS = "EPSG:2154"
TILE_SIZE_M = 500
GROUND_SIZE_PX = 500
TERRAIN_FILE_NAME = "terrain.fvtg"
GROUND_COLOR_FILE_NAME = "ground-color.png"
GROUND_COLOR_MANIFEST_NAME = "ground-color.json"
LOD_FILE_NAMES = tuple(f"terrain-lod{lod}.usda" for lod in range(3))
ROOT_FILE_NAME = "terrain-tile.usda"
MANIFEST_FILE_NAME = "fixed-terrain-usd.v1.json"
CONTRACT_FILE_NAME = "fixed_terrain_usd_contract.v1.json"
OUTPUT_FILE_NAMES = (*LOD_FILE_NAMES, ROOT_FILE_NAME)
HEX_DIGITS = frozenset("0123456789abcdef")
PORTABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
FORBIDDEN_OUTPUT_TOKENS = (
    "atlas",
    "orthophoto",
    "normalmap",
    "heightmap",
    "orm.png",
    "inputs:orm",
    '"orm"',
    "customshader",
    "pbr",
)


class FixedTerrainUsdError(ValueError):
    """A fixed terrain package cannot be exported or validated safely."""


@dataclass(frozen=True)
class FixedTerrainUsdPackage:
    """The five portable files produced beside the canonical tile inputs."""

    output_root: Path
    root_stage: Path
    lod_payloads: tuple[Path, Path, Path]
    manifest: Path


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(character not in HEX_DIGITS for character in digest):
        raise FixedTerrainUsdError(f"{label} must be a lowercase SHA-256")
    return digest


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise FixedTerrainUsdError("USD numeric values must be finite")
    if value == 0.0:
        return "0"
    rendered = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _wrapped(values: Iterable[str], *, indent: str = "        ") -> str:
    sequence = tuple(values)
    if not sequence:
        return "[]"
    return "[\n" + "\n".join(f"{indent}{value}," for value in sequence) + "\n    ]"


def _load_contract(path: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    resolved = (
        Path(__file__).with_name(CONTRACT_FILE_NAME) if path is None else Path(path)
    ).resolve()
    try:
        contract = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixedTerrainUsdError(f"Invalid fixed USD contract: {error}") from error
    expected_stage = {
        "meters_per_unit": 1,
        "up_axis": "Z",
        "default_lod": 0,
        "lod_variant_set": "terrainLod",
    }
    expected_inputs = {
        "terrain": TERRAIN_FILE_NAME,
        "ground_color": GROUND_COLOR_FILE_NAME,
        "ground_color_manifest": GROUND_COLOR_MANIFEST_NAME,
    }
    expected_outputs = {
        "root_stage": ROOT_FILE_NAME,
        "lod_payloads": list(LOD_FILE_NAMES),
        "manifest": MANIFEST_FILE_NAME,
    }
    expected_material = {
        "surface_shader": "UsdPreviewSurface",
        "texture_shader": "UsdUVTexture",
        "texture_count": 1,
        "texture_color_space": "sRGB",
        "roughness": 0.9,
        "metallic": 0,
        "wrap_s": "clamp",
        "wrap_t": "clamp",
        "custom_shader": "forbidden",
    }
    expected_uv = {
        "source_crs": CRS,
        "u_axis": "east",
        "v_axis": "north",
        "tile_normalization_m": TILE_SIZE_M,
        "southwest": [0, 0],
        "northeast": [1, 1],
    }
    expected_skirts = {
        "separate_mesh": True,
        "default_visibility": "invisible",
        "main_camera_visibility": "forbidden",
        "collision": "forbidden",
        "shadow": "forbidden",
    }
    expected_integrity = {
        "algorithm": "sha256",
        "input_paths": "package_local_relative_paths_with_exact_basenames",
        "output_paths": "package_local_exact_names",
        "absolute_paths_in_output": "forbidden",
        "validation": "rehash_and_reauthor_all_payloads",
    }
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "locked"
        or contract.get("crs") != CRS
        or contract.get("stage") != expected_stage
        or contract.get("inputs") != expected_inputs
        or contract.get("outputs") != expected_outputs
        or contract.get("material") != expected_material
        or contract.get("uv") != expected_uv
        or contract.get("skirts") != expected_skirts
        or contract.get("integrity") != expected_integrity
    ):
        raise FixedTerrainUsdError("Fixed USD contract differs from locked v1")
    return contract, resolved, _sha256_file(resolved)


def _require_d_root(output_root: Path) -> Path:
    root = Path(output_root).resolve(strict=False)
    if os.name == "nt" and root.drive.upper() != "D:":
        raise FixedTerrainUsdError("Fixed terrain USD outputs must stay on D:")
    return root


def _package_file(root: Path, supplied: Path, expected_name: str) -> Path:
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise FixedTerrainUsdError(f"Missing package input: {expected_name}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise FixedTerrainUsdError(
            f"Package input escapes its root: {expected_name}"
        ) from error
    if resolved.name != expected_name or not resolved.is_file():
        raise FixedTerrainUsdError(
            f"Package input must use the exact basename {expected_name}"
        )
    return resolved


def _relative_asset(root: Path, path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise FixedTerrainUsdError("USD asset path escapes the package root") from error
    if not relative or relative.startswith("/") or "\\" in relative or "@" in relative:
        raise FixedTerrainUsdError("USD asset path is not portable")
    return relative


def _load_ground_color(
    texture_path: Path,
    manifest_path: Path,
    terrain: FixedTerrainTile,
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixedTerrainUsdError(f"Invalid ground-color manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != GROUND_COLOR_SCHEMA:
        raise FixedTerrainUsdError("Unsupported ground-color manifest")
    origin_x = terrain.tile_origin_mm[0] // 1_000
    origin_y = terrain.tile_origin_mm[1] // 1_000
    expected_bounds = [
        origin_x,
        origin_y,
        origin_x + TILE_SIZE_M,
        origin_y + TILE_SIZE_M,
    ]
    expected_grid = {
        "resolution_m": 1,
        "width": GROUND_SIZE_PX,
        "height": GROUND_SIZE_PX,
        "affine": [1, 0, origin_x, 0, -1, origin_y + TILE_SIZE_M],
        "pixel_interpretation": "area",
    }
    runtime = manifest.get("runtime")
    artifact = manifest.get("artifact")
    if (
        manifest.get("status") != "compiled_ground_color_no_source_payload"
        or manifest.get("crs") != CRS
        or manifest.get("bounds_l93_m") != expected_bounds
        or manifest.get("grid") != expected_grid
        or not isinstance(runtime, dict)
        or runtime.get("texture_file") != GROUND_COLOR_FILE_NAME
        or runtime.get("orthophoto_source_file_dependency") != "forbidden"
        or runtime.get("orthophoto_source_path_present") is not False
        or not isinstance(artifact, dict)
        or artifact.get("file") != GROUND_COLOR_FILE_NAME
        or artifact.get("mode") != "RGB8"
    ):
        raise FixedTerrainUsdError("Ground-color metadata differs from its tile")
    try:
        with Image.open(texture_path) as image:
            image.load()
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size
                != (
                    GROUND_SIZE_PX,
                    GROUND_SIZE_PX,
                )
            ):
                raise FixedTerrainUsdError("Ground color must be one 500 x 500 RGB PNG")
    except FixedTerrainUsdError:
        raise
    except Exception as error:
        raise FixedTerrainUsdError(f"Invalid ground-color PNG: {error}") from error
    texture_hash = _sha256_file(texture_path)
    if (
        _require_sha256(artifact.get("sha256"), label="ground-color artifact")
        != texture_hash
        or artifact.get("byte_count") != texture_path.stat().st_size
    ):
        raise FixedTerrainUsdError("Ground-color PNG hash or byte count mismatch")
    return manifest


def _core_arrays(
    tile: FixedTerrainTile, mesh: FixedLodMesh
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    size = mesh.grid_size
    denominator = size - 1
    points: list[str] = []
    normals: list[str] = []
    texture_coordinates: list[str] = []
    for index, (height_mm, normal) in enumerate(
        zip(mesh.relative_heights_mm, mesh.normals_snorm16, strict=True)
    ):
        row, column = divmod(index, size)
        x = column * TILE_SIZE_M / denominator
        y = row * TILE_SIZE_M / denominator
        points.append(f"({_number(x)}, {_number(y)}, {_number(height_mm / 1_000.0)})")
        normals.append(
            "(" + ", ".join(_number(component / 32_767.0) for component in normal) + ")"
        )
        texture_coordinates.append(
            f"({_number(column / denominator)}, {_number(row / denominator)})"
        )
    return (
        tuple(points),
        tuple(normals),
        tuple(texture_coordinates),
    )


def _skirt_arrays(
    mesh: FixedLodMesh,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, int], ...]]:
    size = mesh.grid_size
    denominator = size - 1
    perimeter = mesh.skirt_core_vertex_indices
    if len(perimeter) != len(mesh.skirt_relative_heights_mm):
        raise FixedTerrainUsdError("FVTG skirt perimeter and heights do not align")
    points: list[str] = []
    for core_index in perimeter:
        row, column = divmod(core_index, size)
        points.append(
            f"({_number(column * TILE_SIZE_M / denominator)}, "
            f"{_number(row * TILE_SIZE_M / denominator)}, "
            f"{_number(mesh.relative_heights_mm[core_index] / 1_000.0)})"
        )
    for core_index, height_mm in zip(
        perimeter, mesh.skirt_relative_heights_mm, strict=True
    ):
        row, column = divmod(core_index, size)
        points.append(
            f"({_number(column * TILE_SIZE_M / denominator)}, "
            f"{_number(row * TILE_SIZE_M / denominator)}, "
            f"{_number(height_mm / 1_000.0)})"
        )
    count = len(perimeter)
    triangles: list[tuple[int, int, int]] = []
    for index in range(count):
        following = (index + 1) % count
        triangles.append((index, count + index, following))
        triangles.append((following, count + index, count + following))
    if len(triangles) != mesh.skirt_triangle_count:
        raise FixedTerrainUsdError("FVTG skirt triangle count changed")
    return tuple(points), tuple(triangles)


def _validated_ground_color_asset(value: str) -> str:
    asset_path = Path(value)
    if (
        not value
        or asset_path.is_absolute()
        or ".." in asset_path.parts
        or "\\" in value
        or "@" in value
        or asset_path.name != GROUND_COLOR_FILE_NAME
    ):
        raise FixedTerrainUsdError(
            "ground_color_asset must be a confined portable path"
        )
    return asset_path.as_posix()


def author_fixed_lod_usda(
    tile: FixedTerrainTile,
    lod: int,
    *,
    ground_color_asset: str = GROUND_COLOR_FILE_NAME,
) -> bytes:
    """Author one regular core mesh and a separate invisible skirt mesh."""

    if isinstance(lod, bool) or not isinstance(lod, int) or lod not in (0, 1, 2):
        raise FixedTerrainUsdError("LOD must be 0, 1, or 2")
    mesh = tile.lods[lod]
    if mesh.lod != lod:
        raise FixedTerrainUsdError("FVTG LODs are not ordered")
    texture_asset = _validated_ground_color_asset(ground_color_asset)
    points, normals, texture_coordinates = _core_arrays(tile, mesh)
    skirt_points, skirt_triangles = _skirt_arrays(mesh)
    core_counts = _wrapped("3" for _ in mesh.core_triangles)
    core_indices = _wrapped(
        str(index) for triangle in mesh.core_triangles for index in triangle
    )
    skirt_counts = _wrapped("3" for _ in skirt_triangles)
    skirt_indices = _wrapped(
        str(index) for triangle in skirt_triangles for index in triangle
    )
    text = f'''#usda 1.0
(
    defaultPrim = "TerrainPayload"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "TerrainPayload"
{{
    custom string fireviewer:crs = "{CRS}"
    custom int fireviewer:terrain_lod = {lod}
    custom string fireviewer:source_grid_sha256 = "{tile.source_grid_sha256.hex()}"
    custom string fireviewer:terrain_contract_sha256 = "{tile.contract_sha256.hex()}"

    def Material "GroundMaterial"
    {{
        token outputs:surface.connect = </TerrainPayload/GroundMaterial/Surface.outputs:surface>

        def Shader "Surface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </TerrainPayload/GroundMaterial/Color.outputs:rgb>
            float inputs:metallic = 0
            float inputs:roughness = 0.9
            token outputs:surface
        }}

        def Shader "Color"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{texture_asset}@
            token inputs:sourceColorSpace = "sRGB"
            float2 inputs:st.connect = </TerrainPayload/GroundMaterial/TexCoord.outputs:result>
            token inputs:wrapS = "clamp"
            token inputs:wrapT = "clamp"
            float3 outputs:rgb
        }}

        def Shader "TexCoord"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }}
    }}

    def Mesh "Core" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        custom int fireviewer:terrain_lod = {lod}
        custom string fireviewer:texture_coordinates = "EPSG2154_u_east_v_north"
        uniform token subdivisionScheme = "none"
        point3f[] points = {_wrapped(points)}
        normal3f[] normals = {_wrapped(normals)} (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = {_wrapped(texture_coordinates)} (
            interpolation = "vertex"
        )
        int[] faceVertexCounts = {core_counts}
        int[] faceVertexIndices = {core_indices}
        rel material:binding = </TerrainPayload/GroundMaterial>
    }}

    def Mesh "Skirt"
    {{
        custom string fireviewer:main_camera_visibility = "forbidden"
        custom string fireviewer:collision = "forbidden"
        custom string fireviewer:shadow = "forbidden"
        uniform token purpose = "guide"
        token visibility = "invisible"
        uniform token subdivisionScheme = "none"
        point3f[] points = {_wrapped(skirt_points)}
        int[] faceVertexCounts = {skirt_counts}
        int[] faceVertexIndices = {skirt_indices}
    }}
}}
'''
    return text.encode("utf-8")


def author_fixed_root_usda(
    tile: FixedTerrainTile,
    *,
    tile_id: str,
    zone_origin_l93_m: Sequence[float],
) -> bytes:
    """Author the portable root stage with LOD0 selected by default."""

    if PORTABLE_ID.fullmatch(tile_id) is None:
        raise FixedTerrainUsdError("tile_id must be a portable non-path identifier")
    if len(zone_origin_l93_m) != 2:
        raise FixedTerrainUsdError("zone_origin_l93_m must contain two coordinates")
    zone_x, zone_y = (float(value) for value in zone_origin_l93_m)
    if not math.isfinite(zone_x) or not math.isfinite(zone_y):
        raise FixedTerrainUsdError("zone_origin_l93_m must contain finite values")
    tile_x = tile.tile_origin_mm[0] / 1_000.0
    tile_y = tile.tile_origin_mm[1] / 1_000.0
    z_origin = tile.z_origin_mm / 1_000.0
    variants = "\n".join(
        f"""        "lod{lod}" {{
            def Xform "Terrain" (
                prepend payload = @{file_name}@</TerrainPayload>
            )
            {{
            }}
        }}"""
        for lod, file_name in enumerate(LOD_FILE_NAMES)
    )
    text = f'''#usda 1.0
(
    defaultPrim = "TerrainTile"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "TerrainTile" (
    kind = "component"
    variants = {{
        string terrainLod = "lod0"
    }}
    prepend variantSets = "terrainLod"
)
{{
    custom string fireviewer:crs = "{CRS}"
    custom string fireviewer:tile_id = {_quote(tile_id)}
    custom double2 fireviewer:tile_origin_l93_m = ({_number(tile_x)}, {_number(tile_y)})
    custom double2 fireviewer:zone_origin_l93_m = ({_number(zone_x)}, {_number(zone_y)})
    custom string fireviewer:texture_coordinates = "EPSG2154_u_east_v_north"
    double3 xformOp:translate = ({_number(tile_x - zone_x)}, {_number(tile_y - zone_y)}, {_number(z_origin)})
    uniform token[] xformOpOrder = ["xformOp:translate"]

    variantSet "terrainLod" = {{
{variants}
    }}
}}
'''
    return text.encode("utf-8")


def _artifact(path: Path, *, relative_name: str) -> dict[str, object]:
    return {
        "path": relative_name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _manifest_payload(
    *,
    tile: FixedTerrainTile,
    tile_id: str,
    zone_origin_l93_m: Sequence[float],
    contract_hash: str,
    package_root: Path,
    input_paths: Mapping[str, Path],
    outputs: Mapping[str, bytes],
) -> dict[str, object]:
    inputs = {
        role: _artifact(path, relative_name=_relative_asset(package_root, path))
        for role, path in sorted(input_paths.items())
    }
    output_records = {
        name: {
            "path": name,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for name, payload in sorted(outputs.items())
    }
    build_identity = {
        "schema": PACKAGE_SCHEMA,
        "contract_sha256": contract_hash,
        "tile_id": tile_id,
        "zone_origin_l93_m": [float(value) for value in zone_origin_l93_m],
        "inputs": {role: record["sha256"] for role, record in inputs.items()},
        "outputs": {name: record["sha256"] for name, record in output_records.items()},
    }
    return {
        "schema": PACKAGE_SCHEMA,
        "status": "compiled",
        "crs": CRS,
        "tile_id": tile_id,
        "tile_origin_l93_m": [
            tile.tile_origin_mm[0] / 1_000.0,
            tile.tile_origin_mm[1] / 1_000.0,
        ],
        "zone_origin_l93_m": [float(value) for value in zone_origin_l93_m],
        "build_id": _sha256_bytes(_canonical_json(build_identity)),
        "contract_sha256": contract_hash,
        "inputs": inputs,
        "outputs": output_records,
        "lods": {
            f"lod{mesh.lod}": {
                "grid_size": mesh.grid_size,
                "core_vertices": mesh.core_vertex_count,
                "core_triangles": mesh.core_triangle_count,
                "skirt_vertices": mesh.skirt_vertex_count * 2,
                "skirt_triangles": mesh.skirt_triangle_count,
            }
            for mesh in tile.lods
        },
        "material": {
            "texture": GROUND_COLOR_FILE_NAME,
            "texture_count": 1,
            "color_space": "sRGB",
            "surface_shader": "UsdPreviewSurface",
            "texture_shader": "UsdUVTexture",
            "roughness": 0.9,
            "metallic": 0,
        },
        "uv": {
            "source_crs": CRS,
            "u_axis": "east",
            "v_axis": "north",
            "southwest": [0, 0],
            "northeast": [1, 1],
        },
        "skirts": {
            "separate_mesh": True,
            "default_visibility": "invisible",
            "main_camera_visibility": "forbidden",
        },
        "default_lod": 0,
    }


def export_fixed_terrain_usd(
    terrain_path: Path,
    ground_color_path: Path,
    ground_color_manifest_path: Path,
    output_root: Path,
    *,
    tile_id: str,
    zone_origin_l93_m: Sequence[float],
    contract_path: Path | None = None,
) -> FixedTerrainUsdPackage:
    """Export and atomically publish all portable fixed-terrain USD files."""

    root = _require_d_root(output_root)
    if not root.is_dir():
        raise FixedTerrainUsdError("Fixed terrain package root must already exist")
    terrain_file = _package_file(root, terrain_path, TERRAIN_FILE_NAME)
    ground_color_file = _package_file(root, ground_color_path, GROUND_COLOR_FILE_NAME)
    ground_manifest_file = _package_file(
        root, ground_color_manifest_path, GROUND_COLOR_MANIFEST_NAME
    )
    _contract, _resolved_contract, contract_hash = _load_contract(contract_path)
    try:
        terrain = read_fixed_terrain(terrain_file)
    except Exception as error:
        raise FixedTerrainUsdError(f"Invalid canonical FVTG: {error}") from error
    _load_ground_color(ground_color_file, ground_manifest_file, terrain)
    if PORTABLE_ID.fullmatch(tile_id) is None:
        raise FixedTerrainUsdError("tile_id must be a portable non-path identifier")
    zone_origin = tuple(float(value) for value in zone_origin_l93_m)
    if len(zone_origin) != 2 or not all(math.isfinite(value) for value in zone_origin):
        raise FixedTerrainUsdError("zone_origin_l93_m must contain two finite values")

    texture_asset = _relative_asset(root, ground_color_file)
    outputs = {
        file_name: author_fixed_lod_usda(terrain, lod, ground_color_asset=texture_asset)
        for lod, file_name in enumerate(LOD_FILE_NAMES)
    }
    outputs[ROOT_FILE_NAME] = author_fixed_root_usda(
        terrain,
        tile_id=tile_id,
        zone_origin_l93_m=zone_origin,
    )
    input_paths = {
        "ground_color": ground_color_file,
        "ground_color_manifest": ground_manifest_file,
        "terrain": terrain_file,
    }
    manifest = _manifest_payload(
        tile=terrain,
        tile_id=tile_id,
        zone_origin_l93_m=zone_origin,
        contract_hash=contract_hash,
        package_root=root,
        input_paths=input_paths,
        outputs=outputs,
    )
    manifest_bytes = _canonical_json(manifest)
    for name, payload in (
        *sorted(outputs.items()),
        (MANIFEST_FILE_NAME, manifest_bytes),
    ):
        _atomic_write(root / name, payload)
    package = FixedTerrainUsdPackage(
        output_root=root,
        root_stage=root / ROOT_FILE_NAME,
        lod_payloads=tuple(root / name for name in LOD_FILE_NAMES),  # type: ignore[arg-type]
        manifest=root / MANIFEST_FILE_NAME,
    )
    validate_fixed_terrain_usd_package(root, contract_path=contract_path)
    return package


def validate_fixed_terrain_usd_package(
    package_root: Path,
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Rehash inputs and outputs, then re-author every USD file byte for byte."""

    root = _require_d_root(package_root)
    if not root.is_dir():
        raise FixedTerrainUsdError("Fixed terrain package root is missing")
    _contract, _resolved_contract, contract_hash = _load_contract(contract_path)
    try:
        manifest = json.loads((root / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixedTerrainUsdError(f"Invalid fixed USD manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != PACKAGE_SCHEMA:
        raise FixedTerrainUsdError("Unsupported fixed USD package manifest")
    if (
        manifest.get("status") != "compiled"
        or manifest.get("contract_sha256") != contract_hash
    ):
        raise FixedTerrainUsdError("Fixed USD manifest contract or status mismatch")
    input_records = manifest.get("inputs")
    if not isinstance(input_records, dict) or set(input_records) != {
        "ground_color",
        "ground_color_manifest",
        "terrain",
    }:
        raise FixedTerrainUsdError("Fixed USD manifest input set changed")

    def input_path(role: str, expected_name: str) -> Path:
        record = input_records.get(role)
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise FixedTerrainUsdError(f"Fixed USD input path is invalid: {role}")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise FixedTerrainUsdError(f"Fixed USD input path escapes package: {role}")
        return _package_file(root, root / candidate, expected_name)

    terrain_file = input_path("terrain", TERRAIN_FILE_NAME)
    ground_color_file = input_path("ground_color", GROUND_COLOR_FILE_NAME)
    ground_manifest_file = input_path(
        "ground_color_manifest", GROUND_COLOR_MANIFEST_NAME
    )
    tile_id = manifest.get("tile_id")
    if not isinstance(tile_id, str) or PORTABLE_ID.fullmatch(tile_id) is None:
        raise FixedTerrainUsdError("Fixed USD manifest has an invalid tile_id")
    zone_origin = manifest.get("zone_origin_l93_m")
    if not isinstance(zone_origin, list) or len(zone_origin) != 2:
        raise FixedTerrainUsdError("Fixed USD manifest has no zone origin")
    try:
        zone_origin_tuple = tuple(float(value) for value in zone_origin)
    except (TypeError, ValueError) as error:
        raise FixedTerrainUsdError(
            "Fixed USD manifest has an invalid zone origin"
        ) from error
    if not all(math.isfinite(value) for value in zone_origin_tuple):
        raise FixedTerrainUsdError("Fixed USD manifest has a non-finite zone origin")
    try:
        terrain = read_fixed_terrain(terrain_file)
    except Exception as error:
        raise FixedTerrainUsdError(f"Invalid canonical FVTG: {error}") from error
    _load_ground_color(ground_color_file, ground_manifest_file, terrain)

    expected_inputs = {
        "ground_color": ground_color_file,
        "ground_color_manifest": ground_manifest_file,
        "terrain": terrain_file,
    }
    for role, path in expected_inputs.items():
        record = input_records.get(role)
        if not isinstance(record, dict) or record != _artifact(
            path, relative_name=_relative_asset(root, path)
        ):
            raise FixedTerrainUsdError(f"Fixed USD input hash mismatch: {role}")

    texture_asset = _relative_asset(root, ground_color_file)
    expected_outputs = {
        file_name: author_fixed_lod_usda(terrain, lod, ground_color_asset=texture_asset)
        for lod, file_name in enumerate(LOD_FILE_NAMES)
    }
    expected_outputs[ROOT_FILE_NAME] = author_fixed_root_usda(
        terrain,
        tile_id=tile_id,
        zone_origin_l93_m=zone_origin_tuple,
    )
    output_records = manifest.get("outputs")
    if not isinstance(output_records, dict) or set(output_records) != set(
        expected_outputs
    ):
        raise FixedTerrainUsdError("Fixed USD manifest output set changed")
    for name, expected_payload in expected_outputs.items():
        output_path = root / name
        if not output_path.is_file():
            raise FixedTerrainUsdError(f"Fixed USD output is missing: {name}")
        observed_payload = output_path.read_bytes()
        expected_record = {
            "path": name,
            "bytes": len(expected_payload),
            "sha256": _sha256_bytes(expected_payload),
        }
        if (
            observed_payload != expected_payload
            or output_records.get(name) != expected_record
        ):
            raise FixedTerrainUsdError(f"Fixed USD output hash mismatch: {name}")
        lower_text = observed_payload.decode("utf-8").casefold()
        if name in LOD_FILE_NAMES and "</terraintile/" in lower_text:
            raise FixedTerrainUsdError(
                f"Fixed USD payload targets a prim outside its payload scope: {name}"
            )
        forbidden = [token for token in FORBIDDEN_OUTPUT_TOKENS if token in lower_text]
        if forbidden:
            raise FixedTerrainUsdError(
                f"Fixed USD output contains forbidden runtime tokens: {forbidden}"
            )

    expected_manifest = _manifest_payload(
        tile=terrain,
        tile_id=tile_id,
        zone_origin_l93_m=zone_origin_tuple,
        contract_hash=contract_hash,
        package_root=root,
        input_paths=expected_inputs,
        outputs=expected_outputs,
    )
    if manifest != expected_manifest:
        raise FixedTerrainUsdError("Fixed USD manifest is not canonical")
    manifest_text = _canonical_json(manifest).decode("utf-8").casefold()
    forbidden = [token for token in FORBIDDEN_OUTPUT_TOKENS if token in manifest_text]
    if forbidden:
        raise FixedTerrainUsdError(
            f"Fixed USD manifest contains forbidden runtime tokens: {forbidden}"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Export or verify one package without requiring Kit or pxr."""

    parser = argparse.ArgumentParser(
        description="Export one fixed FVTG terrain with one ground-color texture"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="author and verify USD payloads")
    export_parser.add_argument("--package-root", type=Path, required=True)
    export_parser.add_argument("--terrain", type=Path, required=True)
    export_parser.add_argument("--ground-color", type=Path, required=True)
    export_parser.add_argument("--ground-manifest", type=Path, required=True)
    export_parser.add_argument("--tile-id", required=True)
    export_parser.add_argument("--zone-origin-x", type=float, required=True)
    export_parser.add_argument("--zone-origin-y", type=float, required=True)
    export_parser.add_argument("--contract", type=Path)
    verify_parser = commands.add_parser("verify", help="rehash and re-author a package")
    verify_parser.add_argument("--package-root", type=Path, required=True)
    verify_parser.add_argument("--contract", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        package = export_fixed_terrain_usd(
            args.terrain,
            args.ground_color,
            args.ground_manifest,
            args.package_root,
            tile_id=args.tile_id,
            zone_origin_l93_m=(args.zone_origin_x, args.zone_origin_y),
            contract_path=args.contract,
        )
        manifest = json.loads(package.manifest.read_text(encoding="utf-8"))
    else:
        manifest = validate_fixed_terrain_usd_package(
            args.package_root, contract_path=args.contract
        )
    summary = {
        "schema": PACKAGE_SCHEMA,
        "status": "verified",
        "package_root": str(Path(args.package_root).resolve()),
        "tile_id": manifest["tile_id"],
        "build_id": manifest["build_id"],
        "root_stage": ROOT_FILE_NAME,
        "lod_payloads": list(LOD_FILE_NAMES),
    }
    print(_canonical_json(summary).decode("utf-8"), end="")
    return 0


__all__ = [
    "CONTRACT_SCHEMA",
    "FixedTerrainUsdError",
    "FixedTerrainUsdPackage",
    "GROUND_COLOR_FILE_NAME",
    "GROUND_COLOR_MANIFEST_NAME",
    "LOD_FILE_NAMES",
    "MANIFEST_FILE_NAME",
    "PACKAGE_SCHEMA",
    "ROOT_FILE_NAME",
    "TERRAIN_FILE_NAME",
    "author_fixed_lod_usda",
    "author_fixed_root_usda",
    "export_fixed_terrain_usd",
    "main",
    "validate_fixed_terrain_usd_package",
]


if __name__ == "__main__":
    raise SystemExit(main())
