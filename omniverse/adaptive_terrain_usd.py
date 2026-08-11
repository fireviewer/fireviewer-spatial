"""Deterministic OpenUSD packaging for canonical FVTQ terrain tiles.

The exporter deliberately writes portable USDA text with relative payload
arcs.  It does not need Kit or ``pxr`` and never reads a source raster.  The
accepted FVTQ package is the sole geometry input, so USD can be regenerated
without contacting the source provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping


try:
    from adaptive_terrain_quadtree import (
        FvtqMesh,
        GRID_UNITS,
        materialize_stitch_triangles,
        read_fvtq,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from blender.adaptive_terrain_quadtree import (
        FvtqMesh,
        GRID_UNITS,
        materialize_stitch_triangles,
        read_fvtq,
    )

try:
    from ground_material_contract import (
        CONTRACT_SCHEMA as GROUND_MATERIAL_SCHEMA,
        LEGACY_CONTRACT_SCHEMA as LEGACY_GROUND_MATERIAL_SCHEMA,
        RUNTIME_SHADER_PENDING_STATUS,
        RUNTIME_SHADER_SCHEMA,
        RUNTIME_TEXTURE_ROLES,
        material_identity,
        resolve_zone_asset,
        sha256_file,
        validate_ground_material_contract,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from blender.ground_material_contract import (
        CONTRACT_SCHEMA as GROUND_MATERIAL_SCHEMA,
        LEGACY_CONTRACT_SCHEMA as LEGACY_GROUND_MATERIAL_SCHEMA,
        RUNTIME_SHADER_PENDING_STATUS,
        RUNTIME_SHADER_SCHEMA,
        RUNTIME_TEXTURE_ROLES,
        material_identity,
        resolve_zone_asset,
        sha256_file,
        validate_ground_material_contract,
    )

PACKAGE_SCHEMA = "fireviewer.terrain-usd-package.v1"
LEGACY_TILE_PACKAGE_SCHEMA = "fireviewer.tile-package.v2"
TILE_PACKAGE_SCHEMA = "fireviewer.tile-package.v3"
LEGACY_TILE_PACKAGE_FILE_NAME = "tile-package.v2.json"
TILE_PACKAGE_FILE_NAME = "tile-package.v3.json"
CRS = "EPSG:2154"
LOD_FILE_NAMES = tuple(f"terrain-lod{lod}.usda" for lod in range(3))
ROOT_FILE_NAME = "terrain-tile.usda"
MANIFEST_FILE_NAME = "terrain-usd-package.v1.json"


class TerrainUsdError(ValueError):
    """The canonical terrain package cannot be represented safely in USD."""


def _tile_recipe_identity(package_root: Path) -> dict[str, Any]:
    """Read the dependency-free recipe identity required by USD and Blender.

    The full canonical package validator depends on raster libraries that are
    intentionally absent from Blender's embedded Python.  Geometry, material,
    composition and output hashes are independently reopened below; this small
    reader only binds the USD package to the already-produced tile recipe.
    """

    root = Path(package_root)
    path = root / TILE_PACKAGE_FILE_NAME
    if not path.is_file():
        path = root / LEGACY_TILE_PACKAGE_FILE_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TerrainUsdError(f"Invalid canonical tile package: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") not in {
        LEGACY_TILE_PACKAGE_SCHEMA,
        TILE_PACKAGE_SCHEMA,
    }:
        raise TerrainUsdError("Unsupported canonical tile package")
    identity: dict[str, Any] = {"package_schema": payload["schema"]}
    for field in ("tile_id", "recipe_id", "recipe_build_id"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise TerrainUsdError(f"Canonical tile package has no {field}")
        if field != "tile_id" and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise TerrainUsdError(f"Canonical tile package has an invalid {field}")
        identity[field] = value
    identity["surface_mapping"] = payload.get("surface_mapping")
    return identity


@dataclass(frozen=True)
class TerrainUsdPackage:
    output_root: Path
    root_stage: Path
    lod_payloads: tuple[Path, Path, Path]
    manifest: Path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise TerrainUsdError("USD coordinates must be finite")
    if value == 0.0:
        return "0"
    rendered = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _wrapped_values(values: Iterable[str], *, indent: str = "        ") -> str:
    sequence = tuple(values)
    if not sequence:
        return "[]"
    return "[\n" + "\n".join(f"{indent}{value}," for value in sequence) + "\n    ]"


def _stitch_variant_records(mesh: FvtqMesh) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for mask, variant in enumerate(mesh.stitch_variants):
        if variant.mask != mask:
            raise TerrainUsdError("FVTQ stitch variants are not ordered masks 0..15")
        triangles = materialize_stitch_triangles(mesh, mask)
        encoded = b"".join(struct.pack("<III", *triangle) for triangle in triangles)
        records.append(
            {
                "mask": mask,
                "triangle_count": len(triangles),
                "triangle_indices_sha256": _sha256(encoded),
                "maximum_error_mm": variant.maximum_error_mm,
                "effective_edge_signatures": [
                    signature.hex() for signature in variant.effective_edge_signatures
                ],
            }
        )
    if len(records) != 16:
        raise TerrainUsdError("FVTQ must expose exactly 16 stitch variants")
    return tuple(records)


def _validate_mesh_set(meshes: tuple[FvtqMesh, FvtqMesh, FvtqMesh]) -> None:
    if tuple(mesh.lod for mesh in meshes) != (0, 1, 2):
        raise TerrainUsdError("FVTQ inputs must be ordered LOD0, LOD1, LOD2")
    origins = {mesh.tile_origin_mm for mesh in meshes}
    source_hashes = {mesh.source_grid_sha256 for mesh in meshes}
    contract_hashes = {mesh.contract_sha256 for mesh in meshes}
    normal_halo_hashes = {mesh.normal_halo_sha256 for mesh in meshes}
    z_origins = {mesh.z_origin_mm for mesh in meshes}
    if len(origins) != 1:
        raise TerrainUsdError("FVTQ LODs do not share one tile origin")
    if len(source_hashes) != 1 or len(contract_hashes) != 1:
        raise TerrainUsdError("FVTQ LODs do not share their canonical inputs")
    if len(normal_halo_hashes) != 1:
        raise TerrainUsdError("FVTQ LODs do not share one canonical normal halo")
    if len(z_origins) != 1:
        raise TerrainUsdError("FVTQ LODs do not share one vertical origin")
    fine_vertices = set(meshes[0].vertices)
    middle_vertices = set(meshes[1].vertices)
    coarse_vertices = set(meshes[2].vertices)
    if not coarse_vertices.issubset(middle_vertices) or not middle_vertices.issubset(
        fine_vertices
    ):
        raise TerrainUsdError("FVTQ LOD vertex sets are not nested")


def author_lod_usda(mesh: FvtqMesh, *, stitch_mask: int = 0) -> bytes:
    """Author one mesh payload plus all compact deterministic stitch deltas.

    Vanilla USD consumers see the explicitly selected topology.  The terrain
    streamer can switch masks without reopening the FVTQ by applying the
    complete base-remove-add tables embedded alongside that topology.
    """

    if isinstance(stitch_mask, bool) or not isinstance(stitch_mask, int):
        raise TerrainUsdError("stitch_mask must be an integer from 0 to 15")
    if not 0 <= stitch_mask < 16:
        raise TerrainUsdError("stitch_mask must be an integer from 0 to 15")
    if mesh.lod == 2 and stitch_mask != 0:
        raise TerrainUsdError("LOD2 cannot select a coarser-neighbor stitch mask")
    stitch_records = _stitch_variant_records(mesh)
    selected_triangles = materialize_stitch_triangles(mesh, stitch_mask)

    prim_name = f"TerrainLod{mesh.lod}"
    points = _wrapped_values(
        (
            "("
            f"{_number(x * 500.0 / GRID_UNITS)}, "
            f"{_number(y * 500.0 / GRID_UNITS)}, "
            f"{_number(relative_z_mm / 1000.0)}"
            ")"
            for x, y, relative_z_mm in mesh.vertices
        )
    )
    face_counts = _wrapped_values(("3" for _ in selected_triangles))
    face_indices = _wrapped_values(
        (str(index) for triangle in selected_triangles for index in triangle)
    )
    if len(mesh.vertex_gradients_mm_per_4m) != len(mesh.vertices):
        raise TerrainUsdError("FVTQ vertex gradients do not align with vertices")
    normal_values: list[str] = []
    for dz_dx, dz_dy in mesh.vertex_gradients_mm_per_4m:
        nx, ny, nz = -float(dz_dx), -float(dz_dy), 4_000.0
        magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
        normal_values.append(
            f"({_number(nx / magnitude)}, {_number(ny / magnitude)}, {_number(nz / magnitude)})"
        )
    normals = _wrapped_values(normal_values)
    source_hash = mesh.source_grid_sha256.hex()
    contract_hash = mesh.contract_sha256.hex()
    stitch_counts = _wrapped_values(
        (str(record["triangle_count"]) for record in stitch_records)
    )
    stitch_errors = _wrapped_values(
        (str(record["maximum_error_mm"]) for record in stitch_records)
    )
    stitch_hashes = _wrapped_values(
        (_quote(str(record["triangle_indices_sha256"])) for record in stitch_records)
    )
    stitch_edge_signatures = _wrapped_values(
        (
            _quote(
                f"{record['mask']}:"
                + ":".join(str(value) for value in record["effective_edge_signatures"])
            )
            for record in stitch_records
        )
    )
    removed_offsets = [0]
    removed_indices: list[int] = []
    replacement_offsets = [0]
    replacement_indices: list[int] = []
    for variant in mesh.stitch_variants:
        removed_indices.extend(variant.removed_triangle_indices)
        removed_offsets.append(len(removed_indices))
        replacement_indices.extend(
            index for triangle in variant.replacement_triangles for index in triangle
        )
        replacement_offsets.append(len(replacement_indices) // 3)
    if len(removed_offsets) != 17 or len(replacement_offsets) != 17:
        raise TerrainUsdError("USD stitch delta offsets must cover masks 0..15")
    text = f'''#usda 1.0
(
    defaultPrim = "{prim_name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Mesh "{prim_name}"
{{
    custom string fireviewer:crs = "{CRS}"
    custom int fireviewer:terrain_lod = {mesh.lod}
    custom int fireviewer:stitch_mask = {stitch_mask}
    custom int[] fireviewer:available_stitch_masks = {_wrapped_values(str(mask) for mask in range(16))}
    custom int[] fireviewer:stitch_triangle_counts = {stitch_counts}
    custom int[] fireviewer:stitch_maximum_error_mm = {stitch_errors}
    custom string[] fireviewer:stitch_triangle_indices_sha256 = {stitch_hashes}
    custom string[] fireviewer:stitch_effective_edge_signatures = {stitch_edge_signatures}
    custom string fireviewer:stitch_delta_encoding = "fvtq-base-remove-add.v1"
    custom int[] fireviewer:stitch_removed_triangle_offsets = {_wrapped_values(str(value) for value in removed_offsets)}
    custom int[] fireviewer:stitch_removed_triangle_indices = {_wrapped_values(str(value) for value in removed_indices)}
    custom int[] fireviewer:stitch_replacement_triangle_offsets = {_wrapped_values(str(value) for value in replacement_offsets)}
    custom int[] fireviewer:stitch_replacement_face_vertex_indices = {_wrapped_values(str(value) for value in replacement_indices)}
    custom int fireviewer:maximum_final_error_mm = {mesh.maximum_final_error_mm}
    custom string fireviewer:source_grid_sha256 = "{source_hash}"
    custom string fireviewer:terrain_contract_sha256 = "{contract_hash}"
    custom string fireviewer:normal_halo_sha256 = "{mesh.normal_halo_sha256.hex()}"
    custom int fireviewer:z_origin_mm = {mesh.z_origin_mm}
    uniform token subdivisionScheme = "none"
    point3f[] points = {points}
    normal3f[] normals = {normals} (
        interpolation = "vertex"
    )
    int[] faceVertexCounts = {face_counts}
    int[] faceVertexIndices = {face_indices}
}}
'''
    return text.encode("utf-8")


def _normalize_stitch_masks(value: Mapping[int, int] | None) -> dict[int, int]:
    if value is None:
        return {0: 0, 1: 0, 2: 0}
    if set(value) != {0, 1, 2}:
        raise TerrainUsdError("stitch masks must declare exactly LOD0, LOD1 and LOD2")
    normalized: dict[int, int] = {}
    for lod in range(3):
        mask = value[lod]
        if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask < 16:
            raise TerrainUsdError(
                f"LOD{lod} stitch mask must be an integer from 0 to 15"
            )
        normalized[lod] = mask
    if normalized[2] != 0:
        raise TerrainUsdError("LOD2 stitch mask must be zero")
    return normalized


def author_root_usda(
    meshes: tuple[FvtqMesh, FvtqMesh, FvtqMesh],
    *,
    tile_id: str,
    zone_origin_l93_m: tuple[float, float],
    composition_assets: Mapping[str, str],
    material_layer_asset: str,
    material_contract_asset: str,
    ground_material_schema: str = GROUND_MATERIAL_SCHEMA,
    surface_mapping_schema: str = "fireviewer.ground-surface-mapping.v3",
    stitch_masks: Mapping[int, int] | None = None,
) -> bytes:
    """Author the portable variant stage that selects exactly one terrain LOD."""

    if not tile_id or any(character in tile_id for character in ("/", "\\", "\0")):
        raise TerrainUsdError("tile_id must be a non-empty portable identifier")
    _validate_mesh_set(meshes)
    selected_stitch_masks = _normalize_stitch_masks(stitch_masks)
    zone_x, zone_y = zone_origin_l93_m
    if not math.isfinite(zone_x) or not math.isfinite(zone_y):
        raise TerrainUsdError("zone_origin_l93_m must contain finite values")
    tile_x = meshes[0].tile_origin_mm[0] / 1000.0
    tile_y = meshes[0].tile_origin_mm[1] / 1000.0
    z_origin = meshes[0].z_origin_mm / 1000.0
    for label, asset in (
        ("material_layer_asset", material_layer_asset),
        ("material_contract_asset", material_contract_asset),
    ):
        if not asset or "@" in asset or "\\" in asset:
            raise TerrainUsdError(f"{label} must be a portable USD asset path")
    asset_lines: list[str] = []
    for name, relative_path in sorted(composition_assets.items()):
        if not name or not relative_path:
            raise TerrainUsdError("Composition asset names and paths cannot be empty")
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise TerrainUsdError("Composition asset paths must be package-relative")
        property_name = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in name
        )
        asset_lines.append(
            f"    custom asset fireviewer:{property_name} = @{candidate.as_posix()}@"
        )
    variants = []
    for lod, file_name in enumerate(LOD_FILE_NAMES):
        variants.append(
            f"""        "lod{lod}" {{
            def Mesh "TerrainPayload" (
                prepend payload = @{file_name}@</TerrainLod{lod}>
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {{
                rel material:binding = </TerrainTile/GroundMaterial>
            }}
        }}"""
        )
    joined_assets = "\n".join(asset_lines)
    joined_variants = "\n".join(variants)
    legacy_v2 = ground_material_schema == LEGACY_GROUND_MATERIAL_SCHEMA
    material_custom_assets = (
        """        custom asset fireviewer:ground_profile_ids = @ground-profile-ids.png@
        custom asset fireviewer:ground_profile_weights = @ground-profile-weights.png@
        custom asset fireviewer:surface_overlays = @surface-overlays.json.gz@"""
        if legacy_v2
        else f"""        custom asset fireviewer:ground_profile_ids = @ground-profile-ids.png@
        custom asset fireviewer:ground_profile_weights = @ground-profile-weights.png@
        custom asset fireviewer:ground_confidence = @ground-confidence.png@
        custom asset fireviewer:ground_orientation = @ground-orientation.png@
        custom string fireviewer:surface_mapping_schema = \"{surface_mapping_schema}\""""
    )
    material_input_assets = (
        """        asset inputs:groundProfileIds = @ground-profile-ids.png@
        asset inputs:groundProfileWeights = @ground-profile-weights.png@
        asset inputs:surfaceOverlays = @surface-overlays.json.gz@"""
        if legacy_v2
        else """        asset inputs:groundProfileIds = @ground-profile-ids.png@
        asset inputs:groundProfileWeights = @ground-profile-weights.png@
        asset inputs:groundConfidence = @ground-confidence.png@
        asset inputs:groundOrientation = @ground-orientation.png@"""
    )
    composition_grid_size = 100 if legacy_v2 else 500
    composition_cell_size_m = 5 if legacy_v2 else 1
    runtime_shader_metadata = (
        ""
        if legacy_v2
        else f'''    custom string fireviewer:runtime_shader_schema = "{RUNTIME_SHADER_SCHEMA}"
    custom string fireviewer:runtime_shader_status = "{RUNTIME_SHADER_PENDING_STATUS}"
    custom bool fireviewer:production_textured_runtime_qualified = false
    custom string fireviewer:preview_surface_policy = "diagnostic_untextured_only"
'''
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
    custom string fireviewer:ground_material_schema = "{ground_material_schema}"
{runtime_shader_metadata}    custom int[] fireviewer:stitch_masks_lod0_lod1_lod2 = {_wrapped_values(str(selected_stitch_masks[lod]) for lod in range(3))}
{joined_assets}
    double3 xformOp:translate = ({_number(tile_x - zone_x)}, {_number(tile_y - zone_y)}, {_number(z_origin)})
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Material "GroundMaterial" (
        prepend references = @{material_layer_asset}@</FireViewerMaterials/GroundMaterial>
    )
    {{
        custom asset fireviewer:ground_material_contract = @{material_contract_asset}@
{material_custom_assets}
        custom double2 fireviewer:tile_origin_l93_m = ({_number(tile_x)}, {_number(tile_y)})
{material_input_assets}
        double2 inputs:tileOriginL93M = ({_number(tile_x)}, {_number(tile_y)})
        int inputs:compositionGridSize = {composition_grid_size}
        double inputs:compositionCellSizeM = {composition_cell_size_m}
    }}

    variantSet "terrainLod" = {{
{joined_variants}
    }}
}}
'''
    return text.encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def export_tile_usd(
    fvtq_paths: Iterable[Path],
    output_root: Path,
    *,
    tile_id: str,
    zone_origin_l93_m: tuple[float, float],
    composition_assets: Mapping[str, str] | None = None,
    ground_material_contract: Path,
    zone_package_root: Path,
    stitch_masks: Mapping[int, int] | None = None,
) -> TerrainUsdPackage:
    """Regenerate the three USD payloads and their root from canonical FVTQ."""

    output_root = Path(output_root).resolve()
    zone_root = Path(zone_package_root).resolve()
    if output_root != zone_root and zone_root not in output_root.parents:
        raise TerrainUsdError("USD output root must remain inside its zone package")
    paths = tuple(Path(path).resolve() for path in fvtq_paths)
    if len(paths) != 3:
        raise TerrainUsdError("Exactly three FVTQ paths are required")
    expected_fvtq = tuple(output_root / f"terrain-lod{lod}.fvtq" for lod in range(3))
    if paths != expected_fvtq:
        raise TerrainUsdError(
            "Canonical FVTQ inputs must be the three package-local terrain LOD files"
        )
    contract_path = Path(ground_material_contract).resolve()
    try:
        contract_path.relative_to(zone_root)
    except ValueError as error:
        raise TerrainUsdError(
            "Ground material contract is outside the zone package"
        ) from error
    material_contract = validate_ground_material_contract(contract_path)
    material_id = material_identity(contract_path, zone_root)
    material_root = contract_path.parent
    material_layer_path = material_root / material_contract["material_layer"]["path"]

    def zone_artifact(path: Path) -> dict[str, object]:
        try:
            relative = path.resolve().relative_to(zone_root).as_posix()
        except ValueError as error:
            raise TerrainUsdError(
                f"Shared material asset escapes zone: {path}"
            ) from error
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    material_record: dict[str, object] = {
        **material_id,
        "zone_root_relative": Path(os.path.relpath(zone_root, output_root)).as_posix(),
        "contract": zone_artifact(contract_path),
        "material_layer": zone_artifact(material_layer_path),
        "runtime_atlas": {
            role: zone_artifact(
                material_root
                / material_contract["runtime_atlas"]["assets"][role]["path"]
            )
            for role in RUNTIME_TEXTURE_ROLES
        },
    }
    if material_contract["schema"] == LEGACY_GROUND_MATERIAL_SCHEMA:
        material_record["atlas_catalog"] = zone_artifact(
            material_root / material_contract["atlas_catalog"]["path"]
        )
    material_layer_asset = Path(
        os.path.relpath(material_layer_path, output_root)
    ).as_posix()
    material_contract_asset = Path(
        os.path.relpath(contract_path, output_root)
    ).as_posix()
    meshes = tuple(read_fvtq(path) for path in paths)
    _validate_mesh_set(meshes)  # type: ignore[arg-type]
    typed_meshes = meshes  # keep the tuple shape explicit for type checkers
    canonical_tile_package = _tile_recipe_identity(output_root)
    if canonical_tile_package["tile_id"] != tile_id:
        raise TerrainUsdError("USD tile identifier differs from its tile package")
    legacy_v2 = canonical_tile_package["package_schema"] == LEGACY_TILE_PACKAGE_SCHEMA
    if legacy_v2 != (material_contract["schema"] == LEGACY_GROUND_MATERIAL_SCHEMA):
        raise TerrainUsdError(
            "Tile package and ground material contract generations differ"
        )
    selected_stitch_masks = _normalize_stitch_masks(stitch_masks)
    assets = composition_assets or (
        {
            "ground_profile_ids": "ground-profile-ids.png",
            "ground_profile_weights": "ground-profile-weights.png",
            "ground_overlays": "surface-overlays.json.gz",
            "tile_composition": "tile-composition.json.gz",
        }
        if legacy_v2
        else {
            "ground_profile_ids": "ground-profile-ids.png",
            "ground_profile_weights": "ground-profile-weights.png",
            "ground_confidence": "ground-confidence.png",
            "ground_orientation": "ground-orientation.png",
        }
    )
    expected_assets = (
        {
            "ground_profile_ids",
            "ground_profile_weights",
            "ground_overlays",
            "tile_composition",
        }
        if legacy_v2
        else {
            "ground_profile_ids",
            "ground_profile_weights",
            "ground_confidence",
            "ground_orientation",
        }
    )
    if set(assets) != expected_assets:
        raise TerrainUsdError("Composition assets differ from the package generation")
    composition_records: dict[str, dict[str, object]] = {}
    for name, relative_path in sorted(assets.items()):
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise TerrainUsdError("Composition asset paths must be package-relative")
        asset_path = output_root / candidate
        if not asset_path.is_file():
            raise TerrainUsdError(f"Composition asset is missing: {relative_path}")
        content = asset_path.read_bytes()
        composition_records[name] = {
            "path": candidate.as_posix(),
            "bytes": len(content),
            "sha256": _sha256(content),
        }
    payloads: dict[str, bytes] = {
        file_name: author_lod_usda(mesh, stitch_mask=selected_stitch_masks[mesh.lod])
        for file_name, mesh in zip(LOD_FILE_NAMES, typed_meshes, strict=True)
    }
    payloads[ROOT_FILE_NAME] = author_root_usda(
        typed_meshes,  # type: ignore[arg-type]
        tile_id=tile_id,
        zone_origin_l93_m=zone_origin_l93_m,
        composition_assets=assets,
        material_layer_asset=material_layer_asset,
        material_contract_asset=material_contract_asset,
        ground_material_schema=material_contract["schema"],
        stitch_masks=selected_stitch_masks,
    )
    manifest: dict[str, object] = {
        "schema": PACKAGE_SCHEMA,
        "tile_id": tile_id,
        "recipe_id": canonical_tile_package["recipe_id"],
        "recipe_build_id": canonical_tile_package["recipe_build_id"],
        "crs": CRS,
        "tile_origin_l93_m": [
            typed_meshes[0].tile_origin_mm[0] / 1000.0,
            typed_meshes[0].tile_origin_mm[1] / 1000.0,
        ],
        "zone_origin_l93_m": list(zone_origin_l93_m),
        "source_grid_sha256": typed_meshes[0].source_grid_sha256.hex(),
        "terrain_contract_sha256": typed_meshes[0].contract_sha256.hex(),
        "normal_halo_sha256": typed_meshes[0].normal_halo_sha256.hex(),
        "root_stage": ROOT_FILE_NAME,
        "default_lod": 0,
        "primary_camera_allowed_lods": [0],
        "selected_stitch_masks": {
            f"lod{lod}": selected_stitch_masks[lod] for lod in range(3)
        },
        "available_stitch_masks": list(range(16)),
        "terrain_lod_aov": "fireviewer:terrain_lod",
        "orthophoto_dependency": "forbidden",
        "tile_package_schema": canonical_tile_package["package_schema"],
        "surface_mapping": canonical_tile_package.get("surface_mapping"),
        "ground_material": material_record,
        "runtime_textured_operational": material_contract.get("runtime_shader", {}).get(
            "production_textured_runtime_qualified"
        )
        is True,
        "preview_surface_policy": (
            "legacy_unqualified"
            if material_contract["schema"] == LEGACY_GROUND_MATERIAL_SCHEMA
            else material_contract["runtime_shader"]["preview_surface_policy"]
        ),
        "composition_assets": composition_records,
        "lod_metrics": {
            f"lod{mesh.lod}": {
                "vertex_count": len(mesh.vertices),
                "triangle_count": len(
                    materialize_stitch_triangles(mesh, selected_stitch_masks[mesh.lod])
                ),
                "base_triangle_count": len(mesh.triangles),
                "maximum_final_error_mm": mesh.maximum_final_error_mm,
                "stitch_variants": list(_stitch_variant_records(mesh)),
                "minimum_height_m": (mesh.z_origin_mm + mesh.minimum_relative_height_mm)
                / 1000.0,
                "maximum_height_m": (mesh.z_origin_mm + mesh.maximum_relative_height_mm)
                / 1000.0,
            }
            for mesh in typed_meshes
        },
        "outputs": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in sorted(payloads.items())
        },
        "fvtq_inputs": {
            f"terrain-lod{mesh.lod}.fvtq": {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path.read_bytes()),
            }
            for path, mesh in zip(paths, typed_meshes, strict=True)
        },
    }
    manifest_bytes = _canonical_json(manifest)
    for name, content in payloads.items():
        _atomic_write(output_root / name, content)
    manifest_path = output_root / MANIFEST_FILE_NAME
    _atomic_write(manifest_path, manifest_bytes)
    return TerrainUsdPackage(
        output_root=output_root,
        root_stage=output_root / ROOT_FILE_NAME,
        lod_payloads=tuple(output_root / name for name in LOD_FILE_NAMES),
        manifest=manifest_path,
    )


def validate_tile_usd_package(
    package_root: Path, *, zone_package_root: Path | None = None
) -> dict[str, object]:
    """Rehash every declared output and reject absolute or escaping assets."""

    package_root = Path(package_root).resolve()
    manifest_path = package_root / MANIFEST_FILE_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TerrainUsdError(f"Invalid USD package manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != PACKAGE_SCHEMA:
        raise TerrainUsdError("Unsupported USD package manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        ROOT_FILE_NAME,
        *LOD_FILE_NAMES,
    }:
        raise TerrainUsdError("USD package output set is incomplete")
    for name, record in outputs.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise TerrainUsdError("USD output paths must remain package-relative")
        path = package_root / name
        if not path.is_file() or not isinstance(record, dict):
            raise TerrainUsdError(f"USD package output is missing: {name}")
        content = path.read_bytes()
        if record.get("bytes") != len(content) or record.get("sha256") != _sha256(
            content
        ):
            raise TerrainUsdError(f"USD package output hash mismatch: {name}")
    composition_assets = manifest.get("composition_assets")
    if not isinstance(composition_assets, dict) or not composition_assets:
        raise TerrainUsdError("USD package composition assets are missing")
    tile_package_schema = manifest.get("tile_package_schema")
    if (
        tile_package_schema is None
        and (package_root / LEGACY_TILE_PACKAGE_FILE_NAME).is_file()
        and not (package_root / TILE_PACKAGE_FILE_NAME).is_file()
    ):
        # v2 manifests predate the explicit generation field.  This is a
        # read-only compatibility path; every newly exported package is v3.
        tile_package_schema = LEGACY_TILE_PACKAGE_SCHEMA
    expected_composition_assets = (
        {
            "ground_profile_ids",
            "ground_profile_weights",
            "ground_overlays",
            "tile_composition",
        }
        if tile_package_schema == LEGACY_TILE_PACKAGE_SCHEMA
        else {
            "ground_profile_ids",
            "ground_profile_weights",
            "ground_confidence",
            "ground_orientation",
        }
    )
    if tile_package_schema not in {LEGACY_TILE_PACKAGE_SCHEMA, TILE_PACKAGE_SCHEMA}:
        raise TerrainUsdError("USD package does not identify its tile package schema")
    if set(composition_assets) != expected_composition_assets:
        raise TerrainUsdError("USD composition assets differ from the package schema")
    for name, record in composition_assets.items():
        if not isinstance(record, dict):
            raise TerrainUsdError(f"Invalid composition asset record: {name}")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise TerrainUsdError(f"Invalid composition asset path: {name}")
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or "ortho" in relative.casefold()
        ):
            raise TerrainUsdError(
                "Composition asset paths must remain package-relative"
            )
        path = package_root / candidate
        if not path.is_file():
            raise TerrainUsdError(f"Composition asset is missing: {relative}")
        content = path.read_bytes()
        if record.get("bytes") != len(content) or record.get("sha256") != _sha256(
            content
        ):
            raise TerrainUsdError(f"Composition asset hash mismatch: {relative}")
    fvtq_inputs = manifest.get("fvtq_inputs")
    expected_fvtq = {f"terrain-lod{lod}.fvtq" for lod in range(3)}
    if not isinstance(fvtq_inputs, dict) or set(fvtq_inputs) != expected_fvtq:
        raise TerrainUsdError("Canonical FVTQ input set is incomplete")
    for name, record in fvtq_inputs.items():
        path = package_root / name
        if not path.is_file() or not isinstance(record, dict):
            raise TerrainUsdError(f"Canonical FVTQ input is missing: {name}")
        content = path.read_bytes()
        if record.get("bytes") != len(content) or record.get("sha256") != _sha256(
            content
        ):
            raise TerrainUsdError(f"Canonical FVTQ input hash mismatch: {name}")
    try:
        meshes = tuple(
            read_fvtq(package_root / f"terrain-lod{lod}.fvtq") for lod in range(3)
        )
        _validate_mesh_set(meshes)  # type: ignore[arg-type]
    except ValueError as error:
        raise TerrainUsdError(f"Canonical FVTQ input is invalid: {error}") from error
    canonical_tile_package = _tile_recipe_identity(package_root)
    if any(
        (
            manifest.get("tile_id") != canonical_tile_package.get("tile_id"),
            manifest.get("recipe_id") != canonical_tile_package.get("recipe_id"),
            manifest.get("recipe_build_id")
            != canonical_tile_package.get("recipe_build_id"),
        )
    ):
        raise TerrainUsdError("USD package recipe identity differs from tile package")
    declared_tile_schema = manifest.get("tile_package_schema")
    if (
        declared_tile_schema is None
        and canonical_tile_package.get("package_schema") == LEGACY_TILE_PACKAGE_SCHEMA
    ):
        declared_tile_schema = LEGACY_TILE_PACKAGE_SCHEMA
    if declared_tile_schema != canonical_tile_package.get("package_schema"):
        raise TerrainUsdError("USD package generation differs from tile package")
    if manifest.get("surface_mapping") != canonical_tile_package.get("surface_mapping"):
        raise TerrainUsdError("USD surface mapping differs from tile package")
    selected_raw = manifest.get("selected_stitch_masks")
    if not isinstance(selected_raw, dict) or set(selected_raw) != {
        "lod0",
        "lod1",
        "lod2",
    }:
        raise TerrainUsdError("USD package selected stitch masks are missing")
    selected_stitch_masks = _normalize_stitch_masks(
        {lod: selected_raw[f"lod{lod}"] for lod in range(3)}
    )
    if manifest.get("available_stitch_masks") != list(range(16)):
        raise TerrainUsdError("USD package must expose all 16 stitch masks")
    lod_metrics = manifest.get("lod_metrics")
    if not isinstance(lod_metrics, dict) or set(lod_metrics) != {
        "lod0",
        "lod1",
        "lod2",
    }:
        raise TerrainUsdError("USD package LOD metrics are incomplete")
    for mesh, payload_name in zip(meshes, LOD_FILE_NAMES, strict=True):
        metrics = lod_metrics[f"lod{mesh.lod}"]
        records = list(_stitch_variant_records(mesh))
        selected_triangles = materialize_stitch_triangles(
            mesh, selected_stitch_masks[mesh.lod]
        )
        if not isinstance(metrics, dict) or any(
            (
                metrics.get("vertex_count") != len(mesh.vertices),
                metrics.get("triangle_count") != len(selected_triangles),
                metrics.get("base_triangle_count") != len(mesh.triangles),
                metrics.get("maximum_final_error_mm") != mesh.maximum_final_error_mm,
                metrics.get("stitch_variants") != records,
            )
        ):
            raise TerrainUsdError(f"USD package LOD{mesh.lod} metrics differ from FVTQ")
        if (package_root / payload_name).read_bytes() != author_lod_usda(
            mesh, stitch_mask=selected_stitch_masks[mesh.lod]
        ):
            raise TerrainUsdError(
                f"USD payload does not reproduce canonical FVTQ LOD{mesh.lod}"
            )

    ground_material = manifest.get("ground_material")
    expected_material_schema = (
        LEGACY_GROUND_MATERIAL_SCHEMA
        if canonical_tile_package["package_schema"] == LEGACY_TILE_PACKAGE_SCHEMA
        else GROUND_MATERIAL_SCHEMA
    )
    if (
        not isinstance(ground_material, dict)
        or ground_material.get("schema") != expected_material_schema
    ):
        raise TerrainUsdError("USD package ground material identity is missing")
    relative_zone = ground_material.get("zone_root_relative")
    if (
        not isinstance(relative_zone, str)
        or not relative_zone
        or "\\" in relative_zone
        or Path(relative_zone).is_absolute()
        or (Path(relative_zone).parts and Path(relative_zone).parts[0].endswith(":"))
    ):
        raise TerrainUsdError("USD package has no portable zone root")
    discovered_zone = (package_root / Path(relative_zone)).resolve()
    if discovered_zone not in package_root.parents:
        raise TerrainUsdError("USD package zone root must be an ancestor of the tile")
    if (
        zone_package_root is not None
        and discovered_zone != Path(zone_package_root).resolve()
    ):
        raise TerrainUsdError("USD package zone root differs from the declared package")

    def validate_zone_record(value: Any, label: str) -> Path:
        if not isinstance(value, dict):
            raise TerrainUsdError(f"Invalid shared ground material record: {label}")
        relative = value.get("path")
        if not isinstance(relative, str):
            raise TerrainUsdError(f"Invalid shared ground material path: {label}")
        try:
            path = resolve_zone_asset(discovered_zone, relative, label)
        except ValueError as error:
            raise TerrainUsdError(str(error)) from error
        if not path.is_file():
            raise TerrainUsdError(f"Shared ground material asset is missing: {label}")
        content = path.read_bytes()
        if value.get("bytes") != len(content) or value.get("sha256") != _sha256(
            content
        ):
            raise TerrainUsdError(f"Shared ground material hash mismatch: {label}")
        return path

    contract_path = validate_zone_record(ground_material.get("contract"), "contract")
    material_layer_path = validate_zone_record(
        ground_material.get("material_layer"), "material_layer"
    )
    if expected_material_schema == LEGACY_GROUND_MATERIAL_SCHEMA:
        validate_zone_record(ground_material.get("atlas_catalog"), "atlas_catalog")
    elif "atlas_catalog" in ground_material:
        raise TerrainUsdError("Material v2 must not retain a legacy atlas catalog")
    runtime_atlas = ground_material.get("runtime_atlas")
    if not isinstance(runtime_atlas, dict) or set(runtime_atlas) != set(
        RUNTIME_TEXTURE_ROLES
    ):
        raise TerrainUsdError("Shared runtime atlas set is incomplete")
    for role in RUNTIME_TEXTURE_ROLES:
        validate_zone_record(runtime_atlas[role], f"runtime_atlas.{role}")
    try:
        validated_contract = validate_ground_material_contract(contract_path)
        actual_identity = material_identity(contract_path, discovered_zone)
    except ValueError as error:
        raise TerrainUsdError(str(error)) from error
    for key, expected in actual_identity.items():
        if ground_material.get(key) != expected:
            raise TerrainUsdError(f"Ground material identity mismatch: {key}")
    expected_runtime_qualified = (
        actual_identity.get("runtime_shader", {}).get(
            "production_textured_runtime_qualified"
        )
        is True
    )
    expected_preview_policy = (
        actual_identity.get("runtime_shader", {}).get("preview_surface_policy")
        if expected_material_schema == GROUND_MATERIAL_SCHEMA
        else "legacy_unqualified"
    )
    if expected_material_schema == LEGACY_GROUND_MATERIAL_SCHEMA:
        if (
            "runtime_textured_operational" in manifest
            or "preview_surface_policy" in manifest
        ) and (
            manifest.get("runtime_textured_operational")
            is not expected_runtime_qualified
            or manifest.get("preview_surface_policy") != expected_preview_policy
        ):
            raise TerrainUsdError("Legacy USD runtime qualification is inconsistent")
    elif (
        manifest.get("runtime_textured_operational") is not expected_runtime_qualified
        or manifest.get("preview_surface_policy") != expected_preview_policy
    ):
        raise TerrainUsdError("USD runtime material qualification is inconsistent")
    if expected_material_schema == GROUND_MATERIAL_SCHEMA and (
        expected_runtime_qualified
        or actual_identity["runtime_shader"].get("schema") != RUNTIME_SHADER_SCHEMA
        or actual_identity["runtime_shader"].get("status")
        != RUNTIME_SHADER_PENDING_STATUS
    ):
        raise TerrainUsdError(
            "Ground material v2 must remain fail-closed until dedicated MDL validation"
        )
    if validated_contract.get("visual_acceptance") != ground_material.get(
        "visual_acceptance"
    ):
        raise TerrainUsdError("Ground material visual status mismatch")
    try:
        zone_origin_raw = manifest["zone_origin_l93_m"]
        if not isinstance(zone_origin_raw, list) or len(zone_origin_raw) != 2:
            raise ValueError("zone origin must contain two values")
        zone_origin = (float(zone_origin_raw[0]), float(zone_origin_raw[1]))
        expected_root = author_root_usda(
            meshes,  # type: ignore[arg-type]
            tile_id=str(manifest["tile_id"]),
            zone_origin_l93_m=zone_origin,
            composition_assets={
                name: str(record["path"]) for name, record in composition_assets.items()
            },
            material_layer_asset=Path(
                os.path.relpath(material_layer_path, package_root)
            ).as_posix(),
            material_contract_asset=Path(
                os.path.relpath(contract_path, package_root)
            ).as_posix(),
            ground_material_schema=validated_contract["schema"],
            stitch_masks=selected_stitch_masks,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TerrainUsdError(
            f"USD root reconstruction inputs are invalid: {error}"
        ) from error
    if (package_root / ROOT_FILE_NAME).read_bytes() != expected_root:
        raise TerrainUsdError("USD root stage does not reproduce its canonical inputs")
    root_text = (package_root / ROOT_FILE_NAME).read_text(encoding="utf-8")
    if "fireviewer:terrain_lod" not in "".join(
        (package_root / name).read_text(encoding="utf-8") for name in LOD_FILE_NAMES
    ):
        raise TerrainUsdError("USD payloads do not expose the terrain LOD AOV value")
    if "orthophoto" in root_text.casefold():
        raise TerrainUsdError(
            "USD root stage contains a forbidden orthophoto dependency"
        )
    if (
        "rel material:binding = </TerrainTile/GroundMaterial>" not in root_text
        or "fireviewer:ground_material_contract" not in root_text
    ):
        raise TerrainUsdError("USD root stage has no shared ground material binding")
    if expected_material_schema == GROUND_MATERIAL_SCHEMA and any(
        token not in root_text
        for token in (
            f'fireviewer:runtime_shader_status = "{RUNTIME_SHADER_PENDING_STATUS}"',
            "fireviewer:production_textured_runtime_qualified = false",
            'fireviewer:preview_surface_policy = "diagnostic_untextured_only"',
        )
    ):
        raise TerrainUsdError(
            "USD root stage does not expose its fail-closed shader gate"
        )
    return manifest
