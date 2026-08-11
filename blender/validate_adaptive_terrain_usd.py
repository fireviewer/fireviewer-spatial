"""Headless Blender acceptance probe for one adaptive terrain USD package.

Run this file with Blender 4.5 LTS, ``--background --factory-startup`` and
``--disable-autoexec``.  It imports the default terrain variant, verifies that
only LOD0 can reach the primary camera, renders a value AOV named
``fireviewer:terrain_lod`` and emits a deterministic JSON receipt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BLENDER_MODULE_ROOT = Path(__file__).resolve().parent
for module_root in (REPOSITORY_ROOT, BLENDER_MODULE_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

try:
    from ground_material_contract import (
        RUNTIME_TEXTURE_ROLES,
        resolve_zone_asset,
        sha256_file,
        validate_ground_material_contract,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from blender.ground_material_contract import (
        RUNTIME_TEXTURE_ROLES,
        resolve_zone_asset,
        sha256_file,
        validate_ground_material_contract,
    )

try:
    from omniverse.adaptive_terrain_usd import (
        TerrainUsdError,
        validate_tile_usd_package,
    )
except ModuleNotFoundError:  # pragma: no cover - direct module path
    from adaptive_terrain_usd import TerrainUsdError, validate_tile_usd_package

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


RECEIPT_SCHEMA = "fireviewer.blender-adaptive-terrain-qa.v2"
PACKAGE_SCHEMA = "fireviewer.terrain-usd-package.v1"
ATLAS_VISUAL_RECEIPT_SCHEMA = "fireviewer.ground-atlas-visual-acceptance.v1"
CLEAN_PBR_LIBRARY_SCHEMA = "fireviewer.clean-pbr-texture-library.v1"
CLEAN_PBR_VISUAL_RECEIPT_SCHEMA = "fireviewer.clean-pbr-texture-visual-acceptance.v1"
TERRAIN_AOV = "fireviewer:terrain_lod"
COVERAGE_AOV = "fireviewer:terrain_coverage"
LOD_PATTERN = re.compile(r"fireviewer:terrain_lod\s*=\s*([012])")
DEFAULT_VARIANT_PATTERN = re.compile(r'terrainLod\s*=\s*"lod([012])"')
TRIPLANAR_BLEND_EXPONENT = 4.0
TRIPLANAR_ALGORITHM = "fireviewer.cpu-reference-triplanar.v1"
TRIPLANAR_AXIS_BASES = {
    "X_YZ": "U=(0,sign(Nx),0);V=(0,0,1);Q=(sign(Nx),0,0)",
    "Y_XZ": "U=(-sign(Ny),0,0);V=(0,0,1);Q=(0,sign(Ny),0)",
    "Z_XY": "U=(sign(Nz),0,0);V=(0,1,0);Q=(0,0,sign(Nz))",
}
_SURFACE_MAPPING_RECORDS = {
    "profile_ids": {
        "file": "ground-profile-ids.png",
        "mode": "RGBA8",
        "encoding": "four_zero_based_stable_profile_indices",
    },
    "profile_weights": {
        "file": "ground-profile-weights.png",
        "mode": "RGBA8",
        "encoding": "four_profile_weights_sum_exactly_255_per_pixel",
    },
    "confidence": {
        "file": "ground-confidence.png",
        "mode": "L8",
        "encoding": "best_vs_next_semantic_class_margin_0_to_255",
    },
    "orientation": {
        "file": "ground-orientation.png",
        "mode": "L8",
        "encoding": "undirected_angle_0_to_pi_mapped_to_uint8",
    },
}


class BlenderTerrainQaError(RuntimeError):
    """A package failed structural, import or rendered-AOV acceptance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_d_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise BlenderTerrainQaError(f"{label} must remain on D:, got {resolved}")
    return resolved


def _relative_package_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BlenderTerrainQaError(f"{label} must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BlenderTerrainQaError(f"{label} escapes its package")
    return root / relative


def inspect_package(package_root: Path) -> dict[str, Any]:
    """Validate declared hashes and the selected terrain LOD without Blender."""

    root = package_root.resolve()
    try:
        validate_tile_usd_package(root)
    except TerrainUsdError as error:
        raise BlenderTerrainQaError(str(error)) from error
    manifest_path = root / "terrain-usd-package.v1.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlenderTerrainQaError(f"Invalid terrain USD manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != PACKAGE_SCHEMA:
        raise BlenderTerrainQaError("Unsupported terrain USD manifest")
    if manifest.get("orthophoto_dependency") != "forbidden":
        raise BlenderTerrainQaError("Orthophoto dependency is not forbidden")
    if manifest.get("primary_camera_allowed_lods") != [0]:
        raise BlenderTerrainQaError("Primary camera contract must allow only LOD0")
    if manifest.get("terrain_lod_aov") != TERRAIN_AOV:
        raise BlenderTerrainQaError("Terrain AOV contract is missing")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise BlenderTerrainQaError("Terrain USD output hashes are missing")
    for name, record in outputs.items():
        path = _relative_package_path(root, name, f"outputs.{name}")
        if not path.is_file() or not isinstance(record, Mapping):
            raise BlenderTerrainQaError(f"Missing terrain USD output: {name}")
        if record.get("bytes") != path.stat().st_size or record.get(
            "sha256"
        ) != _sha256(path):
            raise BlenderTerrainQaError(f"Terrain USD hash mismatch: {name}")
    stage_path = _relative_package_path(root, manifest.get("root_stage"), "root_stage")
    root_text = stage_path.read_text(encoding="utf-8")
    match = DEFAULT_VARIANT_PATTERN.search(root_text)
    if match is None:
        raise BlenderTerrainQaError("Terrain root stage has no default LOD variant")
    selected_lod = int(match.group(1))
    payload_path = root / f"terrain-lod{selected_lod}.usda"
    payload_match = LOD_PATTERN.search(payload_path.read_text(encoding="utf-8"))
    if payload_match is None or int(payload_match.group(1)) != selected_lod:
        raise BlenderTerrainQaError("Selected USD payload has incoherent LOD metadata")
    metrics = manifest.get("lod_metrics", {}).get(f"lod{selected_lod}")
    if not isinstance(metrics, Mapping):
        raise BlenderTerrainQaError("Selected USD payload metrics are missing")
    composition_assets = manifest.get("composition_assets")
    legacy_v2 = manifest.get("tile_package_schema") == "fireviewer.tile-package.v2"
    required_composition = (
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
    if (
        not isinstance(composition_assets, Mapping)
        or set(composition_assets) != required_composition
    ):
        raise BlenderTerrainQaError("Terrain USD composition asset set is incomplete")
    composition_paths: dict[str, Path] = {}
    for name, record in composition_assets.items():
        if not isinstance(record, Mapping):
            raise BlenderTerrainQaError(f"Invalid composition record: {name}")
        path = _relative_package_path(root, record.get("path"), f"composition.{name}")
        if (
            not path.is_file()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            raise BlenderTerrainQaError(f"Terrain composition hash mismatch: {name}")
        composition_paths[name] = path

    material = manifest.get("ground_material")
    if not isinstance(material, Mapping):
        raise BlenderTerrainQaError("Shared ground material identity is missing")
    zone_relative = material.get("zone_root_relative")
    if not isinstance(zone_relative, str) or not zone_relative:
        raise BlenderTerrainQaError("Shared ground material has no zone root")
    zone_root = (root / Path(zone_relative)).resolve()
    contract_record = material.get("contract")
    if not isinstance(contract_record, Mapping) or not isinstance(
        contract_record.get("path"), str
    ):
        raise BlenderTerrainQaError("Shared ground material contract record is invalid")
    try:
        contract_path = resolve_zone_asset(
            zone_root, contract_record["path"], "ground_material.contract"
        )
        ground_contract = validate_ground_material_contract(contract_path)
    except ValueError as error:
        raise BlenderTerrainQaError(str(error)) from error
    if contract_record.get(
        "bytes"
    ) != contract_path.stat().st_size or contract_record.get("sha256") != sha256_file(
        contract_path
    ):
        raise BlenderTerrainQaError("Shared ground material contract hash mismatch")
    for role in RUNTIME_TEXTURE_ROLES:
        record = material.get("runtime_atlas", {}).get(role)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise BlenderTerrainQaError(
                f"Shared runtime atlas record is missing: {role}"
            )
        try:
            atlas_path = resolve_zone_asset(zone_root, record["path"], f"atlas.{role}")
        except ValueError as error:
            raise BlenderTerrainQaError(str(error)) from error
        if record.get("bytes") != atlas_path.stat().st_size or record.get(
            "sha256"
        ) != sha256_file(atlas_path):
            raise BlenderTerrainQaError(f"Shared runtime atlas hash mismatch: {role}")
    tile_composition: dict[str, Any] | None = None
    if legacy_v2:
        try:
            tile_composition = json.loads(
                gzip.decompress(composition_paths["tile_composition"].read_bytes())
            )
        except (OSError, json.JSONDecodeError) as error:
            raise BlenderTerrainQaError(f"Invalid tile composition: {error}") from error
        expected_profiles = [item["id"] for item in ground_contract["profile_table"]]
        if tile_composition.get("profile_table") != expected_profiles:
            raise BlenderTerrainQaError(
                "Tile profile indices differ from the shared material contract"
            )
    else:
        mapping = manifest.get("surface_mapping")
        if (
            not isinstance(mapping, Mapping)
            or mapping.get("schema") != "fireviewer.ground-surface-mapping.v3"
            or mapping.get("crs") != "EPSG:2154"
            or mapping.get("grid_size_px") != [500, 500]
            or mapping.get("cell_size_m") != 1
            or mapping.get("profile_count") != 72
            or mapping.get("profile_ids") != _SURFACE_MAPPING_RECORDS["profile_ids"]
            or mapping.get("profile_weights")
            != _SURFACE_MAPPING_RECORDS["profile_weights"]
            or mapping.get("confidence") != _SURFACE_MAPPING_RECORDS["confidence"]
            or mapping.get("orientation") != _SURFACE_MAPPING_RECORDS["orientation"]
            or mapping.get("runtime_procedural_material") != "forbidden"
            or mapping.get("runtime_orthophoto") != "forbidden"
        ):
            raise BlenderTerrainQaError("Terrain USD surface mapping v3 is invalid")
    selected_stitch_mask = manifest.get("selected_stitch_masks", {}).get("lod0")
    if (
        isinstance(selected_stitch_mask, bool)
        or not isinstance(selected_stitch_mask, int)
        or not 0 <= selected_stitch_mask < 16
    ):
        raise BlenderTerrainQaError("Terrain USD LOD0 stitch mask is invalid")
    lod0_fvtq_path = root / "terrain-lod0.fvtq"
    if not lod0_fvtq_path.is_file():
        raise BlenderTerrainQaError("Canonical LOD0 FVTQ input is missing")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "stage_path": stage_path,
        "selected_lod": selected_lod,
        "metrics": dict(metrics),
        "zone_root": zone_root,
        "ground_material_contract_path": contract_path,
        "ground_material_contract": ground_contract,
        "composition_paths": composition_paths,
        "tile_composition": tile_composition,
        "lod0_fvtq_path": lod0_fvtq_path,
        "lod0_stitch_mask": selected_stitch_mask,
    }


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--render-exr", required=True, type=Path)
    parser.add_argument("--coverage-exr", required=True, type=Path)
    parser.add_argument("--beauty-topdown", required=True, type=Path)
    parser.add_argument("--beauty-oblique", required=True, type=Path)
    parser.add_argument(
        "--surface-library-acceptance-receipt",
        "--atlas-acceptance-receipt",
        dest="surface_library_acceptance_receipt",
        type=Path,
    )
    parser.add_argument("--resolution", type=int, default=512)
    parsed = parser.parse_args(arguments)
    if parsed.resolution < 512 or parsed.resolution > 2048:
        parser.error("--resolution must be between 512 and 2048")
    return parsed


def validate_surface_library_visual_receipt(
    receipt_path: Path,
    inspected: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind visual acceptance to the exact shared surface-library content.

    Clean PBR v3 production uses the library's own acceptance receipt.  The
    historic atlas receipt remains accepted only for legacy/synthetic material
    bundles so existing v2 packages stay readable.
    """

    path = _require_d_path(receipt_path, "surface_library_acceptance_receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlenderTerrainQaError(
            f"Invalid atlas visual acceptance receipt: {error}"
        ) from error
    material_record = inspected["manifest"]["ground_material"]
    if material_record.get("source_library_schema") == CLEAN_PBR_LIBRARY_SCHEMA:
        expected_fields = {
            "schema",
            "status",
            "texture_contract_sha256",
            "library_content_sha256",
            "profile_count",
            "atlas_roles",
            "invalid_profile_count",
        }
        if (
            not isinstance(receipt, dict)
            or set(receipt) != expected_fields
            or receipt.get("schema") != CLEAN_PBR_VISUAL_RECEIPT_SCHEMA
            or receipt.get("status") != "accepted_human_visual"
            or receipt.get("texture_contract_sha256")
            != material_record.get("texture_contract_sha256")
            or receipt.get("library_content_sha256")
            != material_record.get("source_library_content_sha256")
            or receipt.get("profile_count") != 72
            or receipt.get("atlas_roles") != ["basecolor", "normal", "height", "orm"]
            or receipt.get("invalid_profile_count") != 0
        ):
            raise BlenderTerrainQaError(
                "Clean PBR visual acceptance receipt is incomplete or bound to "
                "another surface library"
            )
        return {
            "schema": CLEAN_PBR_VISUAL_RECEIPT_SCHEMA,
            "status": "accepted_human_visual",
            "path": path.name,
            "sha256": _sha256(path),
            "source_library_schema": CLEAN_PBR_LIBRARY_SCHEMA,
            "source_library_identity_sha256": material_record[
                "source_library_identity_sha256"
            ],
            "library_content_sha256": receipt["library_content_sha256"],
            "texture_contract_sha256": receipt["texture_contract_sha256"],
        }

    identity_field = (
        "source_library_identity_sha256"
        if "source_library_identity_sha256" in material_record
        else "source_atlas_catalog_sha256"
    )
    receipt_identity_field = (
        "source_library_identity_sha256"
        if identity_field == "source_library_identity_sha256"
        else "atlas_catalog_sha256"
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != ATLAS_VISUAL_RECEIPT_SCHEMA
        or receipt.get("status") != "accepted_blender_visual"
        or receipt.get(receipt_identity_field) != material_record.get(identity_field)
        or receipt.get("profile_count") != 72
        or receipt.get("texture_count") != 4
        or receipt.get("scale_bands") != ["micro", "meso", "macro"]
        or receipt.get("invalid_profile_count") != 0
    ):
        raise BlenderTerrainQaError(
            "Atlas visual acceptance receipt is incomplete or bound to another atlas"
        )
    return {
        "schema": ATLAS_VISUAL_RECEIPT_SCHEMA,
        "status": "accepted_blender_visual",
        "path": path.name,
        "sha256": _sha256(path),
        receipt_identity_field: receipt[receipt_identity_field],
    }


def validate_atlas_visual_receipt(
    receipt_path: Path,
    inspected: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility alias for callers reading legacy v2 material packages."""

    return validate_surface_library_visual_receipt(receipt_path, inspected)


def _operator_arguments(operator: Any, desired: Mapping[str, Any]) -> dict[str, Any]:
    supported = set(operator.get_rna_type().properties.keys())
    return {key: value for key, value in desired.items() if key in supported}


def _terrain_objects(bpy: Any) -> list[Any]:
    return sorted(
        (item for item in bpy.context.scene.objects if item.type == "MESH"),
        key=lambda item: item.name,
    )


def _load_image_pixels(bpy: Any, path: Path, *, non_color: bool) -> Any:
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        if non_color:
            image.colorspace_settings.name = "Non-Color"
        width, height = (int(value) for value in image.size)
        values = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(values)
        # Blender exposes pixels from the lower-left; FireViewer raster
        # contracts are north-to-south.  Normalize once at the boundary.
        return values.reshape((height, width, 4))[::-1].copy()
    finally:
        bpy.data.images.remove(image)


def _create_float_image(bpy: Any, name: str, values: Any) -> Any:
    import numpy as np

    height, width, channels = values.shape
    if channels != 4:
        raise BlenderTerrainQaError(f"Reference material image {name} is not RGBA")
    image = bpy.data.images.new(
        name,
        width=width,
        height=height,
        alpha=True,
        float_buffer=True,
    )
    image.colorspace_settings.name = "Non-Color"
    image.pixels.foreach_set(np.asarray(values, dtype=np.float32)[::-1].ravel())
    image.update()
    return image


def _line_mask(world_x: Any, world_y: Any, coordinates: Any, width_m: Any) -> Any:
    import numpy as np

    points = np.asarray(coordinates, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
    ):
        raise BlenderTerrainQaError("Overlay line coordinates are invalid")
    width = float(width_m)
    if not math.isfinite(width) or width <= 0.0:
        raise BlenderTerrainQaError("Overlay line width must be positive")
    maximum_distance_squared = (width * 0.5) ** 2
    mask = np.zeros(world_x.shape, dtype=bool)
    for start, end in zip(points[:-1], points[1:], strict=True):
        delta_x = float(end[0] - start[0])
        delta_y = float(end[1] - start[1])
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 0.0:
            continue
        parameter = np.clip(
            (
                (world_x - float(start[0])) * delta_x
                + (world_y - float(start[1])) * delta_y
            )
            / length_squared,
            0.0,
            1.0,
        )
        closest_x = float(start[0]) + parameter * delta_x
        closest_y = float(start[1]) + parameter * delta_y
        mask |= (world_x - closest_x) ** 2 + (
            world_y - closest_y
        ) ** 2 <= maximum_distance_squared
    return mask


def _ring_mask(world_x: Any, world_y: Any, coordinates: Any) -> Any:
    import numpy as np

    points = np.asarray(coordinates, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] < 4
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or not np.array_equal(points[0], points[-1])
    ):
        raise BlenderTerrainQaError("Overlay polygon ring is invalid")
    inside = np.zeros(world_x.shape, dtype=bool)
    for start, end in zip(points[:-1], points[1:], strict=True):
        y_crossing = (float(start[1]) > world_y) != (float(end[1]) > world_y)
        denominator = float(end[1] - start[1])
        if denominator == 0.0:
            continue
        edge_x = (world_y - float(start[1])) * float(
            end[0] - start[0]
        ) / denominator + float(start[0])
        inside ^= y_crossing & (world_x < edge_x)
    return inside


def _polygon_mask(world_x: Any, world_y: Any, coordinates: Any) -> Any:
    import numpy as np

    if not isinstance(coordinates, list) or not coordinates:
        raise BlenderTerrainQaError("Overlay polygon coordinates are invalid")
    mask = _ring_mask(world_x, world_y, coordinates[0])
    for hole in coordinates[1:]:
        mask &= ~_ring_mask(world_x, world_y, hole)
    return np.asarray(mask, dtype=bool)


def _overlay_geometry_mask(
    world_x: Any,
    world_y: Any,
    geometry: Any,
    *,
    width_m: Any,
) -> Any:
    import numpy as np

    if not isinstance(geometry, Mapping):
        raise BlenderTerrainQaError("Overlay geometry is missing")
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "LineString":
        return _line_mask(world_x, world_y, coordinates, width_m)
    if geometry_type == "MultiLineString":
        if not isinstance(coordinates, list):
            raise BlenderTerrainQaError("Overlay multiline coordinates are invalid")
        mask = np.zeros(world_x.shape, dtype=bool)
        for line in coordinates:
            mask |= _line_mask(world_x, world_y, line, width_m)
        return mask
    if geometry_type == "Polygon":
        return _polygon_mask(world_x, world_y, coordinates)
    if geometry_type == "MultiPolygon":
        if not isinstance(coordinates, list):
            raise BlenderTerrainQaError("Overlay multipolygon coordinates are invalid")
        mask = np.zeros(world_x.shape, dtype=bool)
        for polygon in coordinates:
            mask |= _polygon_mask(world_x, world_y, polygon)
        return mask
    if geometry_type == "Point":
        point = np.asarray(coordinates, dtype=np.float64)
        width = float(width_m)
        if (
            point.shape != (2,)
            or not np.isfinite(point).all()
            or not math.isfinite(width)
            or width <= 0.0
        ):
            raise BlenderTerrainQaError("Overlay point geometry is invalid")
        return (world_x - float(point[0])) ** 2 + (world_y - float(point[1])) ** 2 <= (
            width * 0.5
        ) ** 2
    raise BlenderTerrainQaError(f"Unsupported overlay geometry type: {geometry_type}")


def _apply_surface_overlays(
    sampled_ids: Any,
    sampled_weights: Any,
    world_x: Any,
    world_y: Any,
    surface_overlays: Mapping[str, Any] | None,
    profile_table: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    if surface_overlays is None:
        return {
            "feature_count": 0,
            "applied_feature_count": 0,
            "covered_pixel_count": 0,
            "role_pixel_counts": {},
        }
    if (
        surface_overlays.get("schema") != "fireviewer.surface-overlays.v1"
        or surface_overlays.get("crs") != "EPSG:2154"
        or not isinstance(surface_overlays.get("features"), list)
        or surface_overlays.get("feature_count") != len(surface_overlays["features"])
    ):
        raise BlenderTerrainQaError("Surface overlay contract is invalid")
    profile_indices = {
        str(profile["id"]): int(profile["index"]) for profile in profile_table
    }
    winning_roles = np.full(world_x.shape, "", dtype="<U32")
    applied = 0
    previous_priority = -1
    for feature in surface_overlays["features"]:
        if not isinstance(feature, Mapping):
            raise BlenderTerrainQaError("Surface overlay feature is invalid")
        priority = feature.get("priority")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise BlenderTerrainQaError("Surface overlay priority is invalid")
        if priority < previous_priority:
            raise BlenderTerrainQaError("Surface overlays are not priority ordered")
        previous_priority = priority
        profile_id = feature.get("profile_id")
        if profile_id not in profile_indices:
            raise BlenderTerrainQaError(
                f"Surface overlay references an unknown profile: {profile_id}"
            )
        mask = _overlay_geometry_mask(
            world_x,
            world_y,
            feature.get("geometry_l93_m"),
            width_m=feature.get("width_m"),
        )
        if not np.any(mask):
            continue
        applied += 1
        sampled_ids[mask] = 0
        sampled_weights[mask] = 0.0
        sampled_ids[mask, 0] = profile_indices[str(profile_id)]
        sampled_weights[mask, 0] = 1.0
        winning_roles[mask] = str(feature.get("role", "unknown"))
    role_pixel_counts = {
        str(role): int(np.count_nonzero(winning_roles == role))
        for role in sorted(set(winning_roles.ravel()) - {""})
    }
    return {
        "feature_count": len(surface_overlays["features"]),
        "applied_feature_count": applied,
        "covered_pixel_count": int(np.count_nonzero(winning_roles != "")),
        "role_pixel_counts": role_pixel_counts,
    }


def _fvtq_vertex_normals(mesh: FvtqMesh) -> Any:
    """Return the exact vertex normals authored by ``author_lod_usda``."""

    import numpy as np

    gradients = np.asarray(mesh.vertex_gradients_mm_per_4m, dtype=np.float64)
    if gradients.shape != (len(mesh.vertices), 2) or not np.isfinite(gradients).all():
        raise BlenderTerrainQaError("LOD0 FVTQ vertex gradients are invalid")
    normals = np.column_stack(
        (-gradients[:, 0], -gradients[:, 1], np.full(len(gradients), 4_000.0))
    )
    magnitude = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(magnitude <= 1.0e-12):  # pragma: no cover - fixed positive Z
        raise BlenderTerrainQaError("LOD0 FVTQ contains a zero terrain normal")
    return normals / magnitude


def rasterize_fvtq_surface(
    mesh: FvtqMesh, *, stitch_mask: int, output_size: int
) -> dict[str, Any]:
    """Rasterize the exact selected LOD0 surface at QA texture pixel centres.

    FVTQ retains the halo-derived gradient at every adaptive vertex.  Linear
    interpolation over the selected stitch topology therefore reproduces the
    positions and normals consumed by the USD renderer without retaining the
    source raster or its 253x253 normal halo.
    """

    import numpy as np

    if mesh.lod != 0:
        raise BlenderTerrainQaError("Triplanar QA requires canonical FVTQ LOD0")
    if isinstance(stitch_mask, bool) or not isinstance(stitch_mask, int):
        raise BlenderTerrainQaError("LOD0 stitch mask must be an integer")
    if not 0 <= stitch_mask < 16:
        raise BlenderTerrainQaError("LOD0 stitch mask must be between 0 and 15")
    if (
        isinstance(output_size, bool)
        or not isinstance(output_size, int)
        or output_size <= 0
    ):
        raise BlenderTerrainQaError("FVTQ surface raster size must be positive")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape != (len(mesh.vertices), 3) or len(vertices) < 3:
        raise BlenderTerrainQaError("LOD0 FVTQ vertices are invalid")
    local_xyz = np.empty_like(vertices)
    local_xyz[:, 0] = vertices[:, 0] * 500.0 / float(GRID_UNITS)
    local_xyz[:, 1] = vertices[:, 1] * 500.0 / float(GRID_UNITS)
    local_xyz[:, 2] = (float(mesh.z_origin_mm) + vertices[:, 2]) / 1_000.0
    vertex_normals = _fvtq_vertex_normals(mesh)
    try:
        triangles = materialize_stitch_triangles(mesh, stitch_mask)
    except ValueError as error:
        raise BlenderTerrainQaError(
            f"Invalid LOD0 FVTQ stitch topology: {error}"
        ) from error
    if not triangles:
        raise BlenderTerrainQaError("LOD0 FVTQ stitch topology is empty")

    step = 500.0 / float(output_size)
    x_centres = (np.arange(output_size, dtype=np.float64) + 0.5) * step
    y_centres = 500.0 - (np.arange(output_size, dtype=np.float64) + 0.5) * step
    world_z = np.full((output_size, output_size), np.nan, dtype=np.float64)
    normals = np.zeros((output_size, output_size, 3), dtype=np.float64)
    covered = np.zeros((output_size, output_size), dtype=bool)
    tolerance = 1.0e-10

    for raw_triangle in triangles:
        indices = np.asarray(raw_triangle, dtype=np.int64)
        points = local_xyz[indices]
        minimum_x, maximum_x = float(points[:, 0].min()), float(points[:, 0].max())
        minimum_y, maximum_y = float(points[:, 1].min()), float(points[:, 1].max())
        column0 = max(0, int(math.ceil(minimum_x / step - 0.5 - tolerance)))
        column1 = min(
            output_size - 1,
            int(math.floor(maximum_x / step - 0.5 + tolerance)),
        )
        row0 = max(
            0,
            int(math.ceil((500.0 - maximum_y) / step - 0.5 - tolerance)),
        )
        row1 = min(
            output_size - 1,
            int(math.floor((500.0 - minimum_y) / step - 0.5 + tolerance)),
        )
        if column1 < column0 or row1 < row0:
            continue
        sample_x, sample_y = np.meshgrid(
            x_centres[column0 : column1 + 1],
            y_centres[row0 : row1 + 1],
            indexing="xy",
        )
        x0, y0 = points[0, :2]
        x1, y1 = points[1, :2]
        x2, y2 = points[2, :2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) <= 1.0e-18:
            raise BlenderTerrainQaError("LOD0 FVTQ contains a degenerate triangle")
        weight0 = (
            (y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)
        ) / denominator
        weight1 = (
            (y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)
        ) / denominator
        weight2 = 1.0 - weight0 - weight1
        target_covered = covered[row0 : row1 + 1, column0 : column1 + 1]
        inside = (
            (weight0 >= -tolerance)
            & (weight1 >= -tolerance)
            & (weight2 >= -tolerance)
            & ~target_covered
        )
        if not np.any(inside):
            continue
        target_z = world_z[row0 : row1 + 1, column0 : column1 + 1]
        interpolated_z = (
            weight0 * points[0, 2] + weight1 * points[1, 2] + weight2 * points[2, 2]
        )
        target_z[inside] = interpolated_z[inside]
        interpolated_normal = (
            weight0[:, :, None] * vertex_normals[indices[0]]
            + weight1[:, :, None] * vertex_normals[indices[1]]
            + weight2[:, :, None] * vertex_normals[indices[2]]
        )
        target_normal = normals[row0 : row1 + 1, column0 : column1 + 1]
        target_normal[inside] = interpolated_normal[inside]
        target_covered[inside] = True

    if not np.all(covered) or not np.isfinite(world_z).all():
        missing = int(np.count_nonzero(~covered))
        raise BlenderTerrainQaError(
            f"LOD0 FVTQ does not cover the QA surface raster: {missing} pixels"
        )
    normal_magnitude = np.linalg.norm(normals, axis=2, keepdims=True)
    if np.any(normal_magnitude <= 1.0e-12):
        raise BlenderTerrainQaError("LOD0 FVTQ surface normal raster is incomplete")
    normals /= normal_magnitude
    origin = [
        mesh.tile_origin_mm[0] / 1_000.0,
        mesh.tile_origin_mm[1] / 1_000.0,
    ]
    return {
        "schema": "fireviewer.fvtq-surface-raster.v1",
        "tile_origin_l93_m": origin,
        "grid_size_px": [output_size, output_size],
        "stitch_mask": stitch_mask,
        "normal_formula": "normalize(-gradient_x_mm_per_4m,-gradient_y_mm_per_4m,4000)",
        "world_z_m": world_z,
        "world_normal": normals,
        "world_z_sha256": hashlib.sha256(
            np.asarray(world_z, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "world_normal_sha256": hashlib.sha256(
            np.asarray(normals, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "covered_pixel_count": int(np.count_nonzero(covered)),
    }


def _validate_triplanar_surface(
    surface: Mapping[str, Any] | None,
    *,
    output_size: int,
    tile_origin_l93_m: Sequence[float],
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    if not isinstance(surface, Mapping):
        raise BlenderTerrainQaError(
            "Weighted world_triplanar profiles require the validated LOD0 FVTQ surface"
        )
    if surface.get("tile_origin_l93_m") != [
        float(value) for value in tile_origin_l93_m
    ]:
        raise BlenderTerrainQaError("Triplanar surface belongs to another terrain tile")
    if surface.get("grid_size_px") != [output_size, output_size]:
        raise BlenderTerrainQaError("Triplanar surface raster dimensions differ")
    world_z = np.asarray(surface.get("world_z_m"), dtype=np.float64)
    world_normal = np.asarray(surface.get("world_normal"), dtype=np.float64)
    if (
        world_z.shape != (output_size, output_size)
        or world_normal.shape != (output_size, output_size, 3)
        or not np.isfinite(world_z).all()
        or not np.isfinite(world_normal).all()
    ):
        raise BlenderTerrainQaError("Triplanar FVTQ surface arrays are invalid")
    magnitude = np.linalg.norm(world_normal, axis=2, keepdims=True)
    if np.any(magnitude <= 1.0e-12):
        raise BlenderTerrainQaError("Triplanar FVTQ surface contains a zero normal")
    normalized = world_normal / magnitude
    metadata = {
        key: value
        for key, value in surface.items()
        if key not in {"world_z_m", "world_normal"}
    }
    return world_z, normalized, metadata


def compose_reference_pbr_maps(
    profile_ids: Any,
    profile_weights: Any,
    atlas_maps: Mapping[str, Any],
    profile_table: Sequence[Mapping[str, Any]],
    *,
    tile_origin_l93_m: Sequence[float],
    output_size: int,
    ground_confidence: Any | None = None,
    ground_orientation: Any | None = None,
    surface_overlays: Mapping[str, Any] | None = None,
    terrain_surface: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure four-profile atlas compositor using global Lambert-93 phase."""

    import numpy as np

    raw_ids = np.asarray(profile_ids)
    raw_weights = np.asarray(profile_weights)
    if (
        raw_ids.ndim != 3
        or raw_ids.shape[2] != 4
        or raw_ids.shape[0] != raw_ids.shape[1]
        or raw_ids.shape[0] not in {100, 500}
        or raw_weights.shape != raw_ids.shape
    ):
        raise BlenderTerrainQaError(
            "Ground profile maps must be 100x100 legacy or 500x500 v3 RGBA"
        )
    if (
        not np.issubdtype(raw_ids.dtype, np.number)
        or not np.issubdtype(raw_weights.dtype, np.number)
        or not np.isfinite(raw_ids).all()
        or not np.isfinite(raw_weights).all()
        or np.any(raw_ids != np.floor(raw_ids))
        or np.any(raw_weights != np.floor(raw_weights))
        or np.any(raw_ids < 0)
        or np.any(raw_ids > 255)
        or np.any(raw_weights < 0)
        or np.any(raw_weights > 255)
    ):
        raise BlenderTerrainQaError(
            "Ground profile maps must contain exact RGBA8 values"
        )
    ids_u8 = raw_ids.astype(np.uint8)
    weights_u8 = raw_weights.astype(np.uint8)
    if not np.all(weights_u8.sum(axis=2, dtype=np.uint16) == 255):
        raise BlenderTerrainQaError("Ground profile weights do not sum exactly to 255")
    grid_size = int(ids_u8.shape[0])
    is_v3 = grid_size == 500
    if is_v3:
        confidence = np.asarray(ground_confidence)
        orientation = np.asarray(ground_orientation)
        if (
            confidence.shape != (500, 500)
            or orientation.shape != confidence.shape
            or not np.issubdtype(confidence.dtype, np.number)
            or not np.issubdtype(orientation.dtype, np.number)
            or not np.isfinite(confidence).all()
            or not np.isfinite(orientation).all()
            or np.any(confidence != np.floor(confidence))
            or np.any(orientation != np.floor(orientation))
            or np.any(confidence < 0)
            or np.any(confidence > 255)
            or np.any(orientation < 0)
            or np.any(orientation > 255)
        ):
            raise BlenderTerrainQaError(
                "Ground confidence and orientation must be 500x500 L8 values"
            )
        confidence_u8 = confidence.astype(np.uint8)
        orientation_u8 = orientation.astype(np.uint8)
        if surface_overlays is not None:
            raise BlenderTerrainQaError(
                "Surface overlays are not a runtime dependency in mapping v3"
            )
    else:
        if ground_confidence is not None or ground_orientation is not None:
            raise BlenderTerrainQaError(
                "Legacy composition cannot declare v3 confidence/orientation"
            )
        confidence_u8 = None
        orientation_u8 = np.zeros((100, 100), dtype=np.uint8)
    if output_size <= 0:
        raise BlenderTerrainQaError("Reference material output size must be positive")
    if len(tile_origin_l93_m) != 2 or not all(
        math.isfinite(float(value)) for value in tile_origin_l93_m
    ):
        raise BlenderTerrainQaError("Tile origin must contain two finite L93 values")
    if [item.get("index") for item in profile_table] != list(range(len(profile_table))):
        raise BlenderTerrainQaError("Ground material profile table is not zero-based")

    scale_lookup = np.full(len(profile_table), np.nan, dtype=np.float64)
    offset_lookup = np.full((len(profile_table), 2), np.nan, dtype=np.float64)
    atlas_scale_lookup = np.full((len(profile_table), 2), np.nan, dtype=np.float64)
    triplanar_lookup = np.zeros(len(profile_table), dtype=bool)
    for item in profile_table:
        index = int(item["index"])
        atlas_uv = item.get("atlas_uv")
        if atlas_uv is None:
            continue
        try:
            scale_lookup[index] = float(item["physical_scale_m"])
            offset_lookup[index] = [float(value) for value in atlas_uv["offset"]]
            atlas_scale_lookup[index] = [float(value) for value in atlas_uv["scale"]]
        except (KeyError, TypeError, ValueError) as error:
            raise BlenderTerrainQaError(
                f"Ground profile {index} has an invalid atlas sampling contract"
            ) from error
        projection = item.get("projection", "world_xy")
        if projection not in {"world_xy", "world_triplanar"}:
            raise BlenderTerrainQaError(
                f"Ground profile {index} has an invalid projection"
            )
        triplanar_lookup[index] = projection == "world_triplanar"
    used_ids = ids_u8[weights_u8 > 0]
    if used_ids.size == 0 or int(used_ids.max()) >= len(profile_table):
        raise BlenderTerrainQaError("Ground profile map references an invalid profile")
    atlases: dict[str, Any] = {}
    atlas_shape: tuple[int, int] | None = None
    for role in RUNTIME_TEXTURE_ROLES:
        if role not in atlas_maps:
            raise BlenderTerrainQaError(f"Reference atlas is missing: {role}")
        atlas = np.asarray(atlas_maps[role], dtype=np.float32)
        if atlas.ndim != 3 or atlas.shape[2] != 4 or min(atlas.shape[:2]) <= 0:
            raise BlenderTerrainQaError(f"Reference atlas is not RGBA: {role}")
        if not np.isfinite(atlas).all():
            raise BlenderTerrainQaError(f"Reference atlas is non-finite: {role}")
        if atlas_shape is None:
            atlas_shape = (int(atlas.shape[0]), int(atlas.shape[1]))
        elif atlas.shape[:2] != atlas_shape:
            raise BlenderTerrainQaError("Runtime atlas dimensions differ by role")
        atlases[role] = atlas
    if atlas_shape is None:  # pragma: no cover
        raise AssertionError("Runtime texture roles are empty")
    atlas_height, atlas_width = atlas_shape

    axis = (np.arange(output_size, dtype=np.float64) + 0.5) / output_size
    u, row_from_north = np.meshgrid(axis, axis, indexing="xy")
    composition_x = np.minimum((u * grid_size).astype(np.int32), grid_size - 1)
    composition_y = np.minimum(
        (row_from_north * grid_size).astype(np.int32), grid_size - 1
    )
    sampled_ids = ids_u8[composition_y, composition_x]
    sampled_weights = (
        weights_u8[composition_y, composition_x].astype(np.float32) / 255.0
    )
    tile_x, tile_y = (float(value) for value in tile_origin_l93_m)
    world_x = tile_x + u * 500.0
    world_y = tile_y + (1.0 - row_from_north) * 500.0
    overlay_report = _apply_surface_overlays(
        sampled_ids,
        sampled_weights,
        world_x,
        world_y,
        surface_overlays,
        profile_table,
    )
    final_used_ids = sampled_ids[sampled_weights > 0.0]
    used_triplanar_profiles = sorted(
        {
            int(profile_index)
            for profile_index in final_used_ids.tolist()
            if triplanar_lookup[int(profile_index)]
        }
    )
    if not np.isfinite(scale_lookup[final_used_ids]).all() or np.any(
        scale_lookup[final_used_ids] <= 0
    ):
        raise BlenderTerrainQaError(
            "A weighted ground profile has no explicit atlas sampling contract"
        )
    channel_is_triplanar = triplanar_lookup[sampled_ids] & (sampled_weights > 0.0)
    triplanar_pixel_mask = np.any(channel_is_triplanar, axis=2)
    triplanar_pixel_count = int(np.count_nonzero(triplanar_pixel_mask))
    if used_triplanar_profiles:
        world_z, surface_normal, surface_metadata = _validate_triplanar_surface(
            terrain_surface,
            output_size=output_size,
            tile_origin_l93_m=(tile_x, tile_y),
        )
    else:
        world_z = None
        surface_normal = None
        surface_metadata = None
    sample_x = np.empty((4, output_size, output_size), dtype=np.int32)
    sample_y = np.empty((4, output_size, output_size), dtype=np.int32)
    sampled_orientation = orientation_u8[composition_y, composition_x]
    # 0 and 255 encode the same undirected axis endpoint.  Modulo 255 makes
    # that equivalence exact instead of relying on the atlas being symmetric.
    angle = np.mod(sampled_orientation.astype(np.float64), 255.0) * (math.pi / 255.0)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    projected_x = cosine * world_x + sine * world_y
    projected_y = -sine * world_x + cosine * world_y
    for channel in range(4):
        indices = sampled_ids[:, :, channel]
        physical_scale = scale_lookup[indices]
        atlas_u = (
            offset_lookup[indices, 0]
            + np.mod(projected_x / physical_scale, 1.0) * atlas_scale_lookup[indices, 0]
        )
        atlas_v = (
            offset_lookup[indices, 1]
            + np.mod(projected_y / physical_scale, 1.0) * atlas_scale_lookup[indices, 1]
        )
        sample_x[channel] = np.clip(
            np.floor(atlas_u * atlas_width).astype(np.int32), 0, atlas_width - 1
        )
        sample_y[channel] = np.clip(
            atlas_height - 1 - np.floor(atlas_v * atlas_height).astype(np.int32),
            0,
            atlas_height - 1,
        )

    composed: dict[str, Any] = {}
    normal_space = "TANGENT"
    if not used_triplanar_profiles:
        # This is the historic world-XY path.  Keep its operation and addition
        # order unchanged so existing reference hashes remain bit-for-bit stable.
        for role in RUNTIME_TEXTURE_ROLES:
            result = np.zeros((output_size, output_size, 4), dtype=np.float32)
            atlas = atlases[role]
            for channel in range(4):
                result += (
                    atlas[sample_y[channel], sample_x[channel]]
                    * sampled_weights[:, :, channel, None]
                )
            result[:, :, 3] = 1.0
            if role == "normal":
                vectors = result[:, :, :3] * 2.0 - 1.0
                magnitude = np.linalg.norm(vectors, axis=2, keepdims=True)
                vectors /= np.maximum(magnitude, 1.0e-8)
                result[:, :, :3] = vectors * 0.5 + 0.5
            composed[role] = result
    else:
        if world_z is None or surface_normal is None:  # pragma: no cover - gate above
            raise AssertionError("Triplanar surface validation did not return arrays")
        normal_space = "WORLD"
        non_normal_roles = tuple(
            role for role in RUNTIME_TEXTURE_ROLES if role != "normal"
        )
        composed.update(
            {
                role: np.zeros((output_size, output_size, 4), dtype=np.float32)
                for role in non_normal_roles
            }
        )
        accumulated_world_normal = np.zeros(
            (output_size, output_size, 3), dtype=np.float64
        )
        normal_x = surface_normal[:, :, 0]
        normal_y = surface_normal[:, :, 1]
        normal_z = surface_normal[:, :, 2]
        sign_x = np.where(normal_x < 0.0, -1.0, 1.0)
        sign_y = np.where(normal_y < 0.0, -1.0, 1.0)
        sign_z = np.where(normal_z < 0.0, -1.0, 1.0)
        plane_weights = np.abs(surface_normal) ** TRIPLANAR_BLEND_EXPONENT
        plane_weight_sum = plane_weights.sum(axis=2, keepdims=True)
        if np.any(plane_weight_sum <= 1.0e-12):  # pragma: no cover - unit normals
            raise BlenderTerrainQaError("Triplanar plane weights are degenerate")
        plane_weights /= plane_weight_sum

        # World-XY tangent basis used by Blender for the tile UV map.  The
        # fallback is only reachable for a mathematically vertical X-facing
        # surface, but keeps the reference deterministic for synthetic tests.
        tangent = np.empty_like(surface_normal)
        tangent[:, :, 0] = 1.0 - normal_x * normal_x
        tangent[:, :, 1] = -normal_x * normal_y
        tangent[:, :, 2] = -normal_x * normal_z
        tangent_magnitude = np.linalg.norm(tangent, axis=2, keepdims=True)
        fallback = tangent_magnitude[:, :, 0] <= 1.0e-12
        if np.any(fallback):
            fallback_tangent = np.empty_like(surface_normal)
            fallback_tangent[:, :, 0] = -normal_y * normal_x
            fallback_tangent[:, :, 1] = 1.0 - normal_y * normal_y
            fallback_tangent[:, :, 2] = -normal_y * normal_z
            tangent[fallback] = fallback_tangent[fallback]
            tangent_magnitude = np.linalg.norm(tangent, axis=2, keepdims=True)
        if np.any(tangent_magnitude <= 1.0e-12):
            raise BlenderTerrainQaError("Terrain tangent basis is degenerate")
        tangent /= tangent_magnitude
        bitangent = np.cross(surface_normal, tangent)

        def plane_coordinates(axis_index: int) -> tuple[Any, Any]:
            if axis_index == 0:  # X projection samples signed Y/Z.
                raw_u, raw_v = sign_x * world_y, world_z
            elif axis_index == 1:  # Y projection samples signed -X/Z.
                raw_u, raw_v = -sign_y * world_x, world_z
            elif axis_index == 2:  # Z projection samples signed X/Y.
                raw_u, raw_v = sign_z * world_x, world_y
            else:  # pragma: no cover - fixed three-axis loop
                raise AssertionError("Unknown triplanar axis")
            return (
                cosine * raw_u + sine * raw_v,
                -sine * raw_u + cosine * raw_v,
            )

        def atlas_indices(
            projected_u: Any, projected_v: Any, indices: Any
        ) -> tuple[Any, Any]:
            physical_scale = scale_lookup[indices]
            atlas_u = (
                offset_lookup[indices, 0]
                + np.mod(projected_u / physical_scale, 1.0)
                * atlas_scale_lookup[indices, 0]
            )
            atlas_v = (
                offset_lookup[indices, 1]
                + np.mod(projected_v / physical_scale, 1.0)
                * atlas_scale_lookup[indices, 1]
            )
            return (
                np.clip(
                    np.floor(atlas_u * atlas_width).astype(np.int32),
                    0,
                    atlas_width - 1,
                ),
                np.clip(
                    atlas_height
                    - 1
                    - np.floor(atlas_v * atlas_height).astype(np.int32),
                    0,
                    atlas_height - 1,
                ),
            )

        for channel in range(4):
            indices = sampled_ids[:, :, channel]
            profile_weight = sampled_weights[:, :, channel]
            is_triplanar = channel_is_triplanar[:, :, channel]
            xy_x, xy_y = sample_x[channel], sample_y[channel]
            plane_indices = [
                atlas_indices(*plane_coordinates(axis_index), indices)
                for axis_index in range(3)
            ]
            for role in non_normal_roles:
                atlas = atlases[role]
                profile_sample = atlas[xy_y, xy_x].copy()
                if np.any(is_triplanar):
                    triplanar_sample = np.zeros_like(profile_sample)
                    for axis_index, (axis_x, axis_y) in enumerate(plane_indices):
                        triplanar_sample += (
                            atlas[axis_y, axis_x]
                            * plane_weights[:, :, axis_index, None]
                        )
                    profile_sample[is_triplanar] = triplanar_sample[is_triplanar]
                composed[role] += profile_sample * profile_weight[:, :, None]

            normal_atlas = atlases["normal"]
            xy_tangent_normal = normal_atlas[xy_y, xy_x, :3] * 2.0 - 1.0
            profile_world_normal = (
                xy_tangent_normal[:, :, 0, None] * tangent
                + xy_tangent_normal[:, :, 1, None] * bitangent
                + xy_tangent_normal[:, :, 2, None] * surface_normal
            )
            if np.any(is_triplanar):
                triplanar_world_normal = np.zeros_like(profile_world_normal)
                for axis_index, (axis_x, axis_y) in enumerate(plane_indices):
                    sampled_tangent = normal_atlas[axis_y, axis_x, :3] * 2.0 - 1.0
                    if axis_index == 0:
                        sampled_world = np.stack(
                            (
                                sampled_tangent[:, :, 2] * sign_x,
                                sampled_tangent[:, :, 0] * sign_x,
                                sampled_tangent[:, :, 1],
                            ),
                            axis=2,
                        )
                    elif axis_index == 1:
                        sampled_world = np.stack(
                            (
                                -sampled_tangent[:, :, 0] * sign_y,
                                sampled_tangent[:, :, 2] * sign_y,
                                sampled_tangent[:, :, 1],
                            ),
                            axis=2,
                        )
                    else:
                        sampled_world = np.stack(
                            (
                                sampled_tangent[:, :, 0] * sign_z,
                                sampled_tangent[:, :, 1],
                                sampled_tangent[:, :, 2] * sign_z,
                            ),
                            axis=2,
                        )
                    triplanar_world_normal += (
                        sampled_world * plane_weights[:, :, axis_index, None]
                    )
                tri_magnitude = np.linalg.norm(
                    triplanar_world_normal, axis=2, keepdims=True
                )
                triplanar_world_normal /= np.maximum(tri_magnitude, 1.0e-12)
                profile_world_normal[is_triplanar] = triplanar_world_normal[
                    is_triplanar
                ]
            accumulated_world_normal += (
                profile_world_normal * profile_weight[:, :, None]
            )

        for role in non_normal_roles:
            composed[role][:, :, 3] = 1.0
        accumulated_magnitude = np.linalg.norm(
            accumulated_world_normal, axis=2, keepdims=True
        )
        if np.any(accumulated_magnitude <= 1.0e-12):
            raise BlenderTerrainQaError("Composed triplanar normal is degenerate")
        accumulated_world_normal /= accumulated_magnitude
        composed["normal"] = np.ones((output_size, output_size, 4), dtype=np.float32)
        composed["normal"][:, :, :3] = (accumulated_world_normal * 0.5 + 0.5).astype(
            np.float32
        )
    composed["basecolor"][:, :, :3] *= composed["orm"][:, :, 0:1]
    return {
        "maps": composed,
        "derived_sha256": {
            role: hashlib.sha256(
                np.asarray(composed[role], dtype="<f4").tobytes(order="C")
            ).hexdigest()
            for role in RUNTIME_TEXTURE_ROLES
        },
        "used_profile_indices": sorted(
            {int(value) for value in final_used_ids.tolist()}
        ),
        "world_projection": (
            "EPSG:2154 metric XY/YZ/XZ"
            if used_triplanar_profiles
            else "EPSG:2154 metric XY"
        ),
        "normal_space": normal_space,
        "triplanar": {
            "algorithm": TRIPLANAR_ALGORITHM,
            "blend_exponent": TRIPLANAR_BLEND_EXPONENT,
            "axis_order": ["X_YZ", "Y_XZ", "Z_XY"],
            "axis_bases": dict(TRIPLANAR_AXIS_BASES),
            "orientation_rotation": "[cos sin; -sin cos], L8 modulo 255 on every plane",
            "normal_map_conversion": "OpenGL tangent to signed right-handed world basis",
            "used_profile_indices": used_triplanar_profiles,
            "pixel_count": triplanar_pixel_count,
            "surface": surface_metadata,
        },
        "tile_origin_l93_m": [tile_x, tile_y],
        "surface_overlays": overlay_report,
        "mapping_schema": (
            "fireviewer.ground-surface-mapping.v3"
            if is_v3
            else "fireviewer.ground-surface-mapping.v2-legacy"
        ),
        "orientation_encoding": "L8 undirected axis 0..pi",
        "confidence": (
            {
                "minimum": int(confidence_u8.min()),
                "maximum": int(confidence_u8.max()),
                "mean": float(confidence_u8.mean()),
                "shader_input": False,
            }
            if confidence_u8 is not None
            else None
        ),
    }


def _reference_pbr_maps(
    bpy: Any, inspected: Mapping[str, Any], size: int
) -> dict[str, Any]:
    """Read hash-locked assets then call the pure material compositor."""

    import numpy as np

    contract = inspected["ground_material_contract"]
    composition_paths = inspected["composition_paths"]
    ids_pixels = _load_image_pixels(
        bpy, composition_paths["ground_profile_ids"], non_color=True
    )
    weight_pixels = _load_image_pixels(
        bpy, composition_paths["ground_profile_weights"], non_color=True
    )
    profile_ids = np.floor(ids_pixels * 255.0 + 0.5).astype(np.uint8)
    profile_weights = np.floor(weight_pixels * 255.0 + 0.5).astype(np.uint8)
    if "ground_confidence" in composition_paths:
        confidence_pixels = _load_image_pixels(
            bpy, composition_paths["ground_confidence"], non_color=True
        )
        orientation_pixels = _load_image_pixels(
            bpy, composition_paths["ground_orientation"], non_color=True
        )
        ground_confidence = np.floor(confidence_pixels[:, :, 0] * 255.0 + 0.5).astype(
            np.uint8
        )
        ground_orientation = np.floor(orientation_pixels[:, :, 0] * 255.0 + 0.5).astype(
            np.uint8
        )
        surface_overlays = None
    else:
        ground_confidence = None
        ground_orientation = None
        try:
            surface_overlays = json.loads(
                gzip.decompress(composition_paths["ground_overlays"].read_bytes())
            )
        except (OSError, json.JSONDecodeError) as error:
            raise BlenderTerrainQaError(f"Invalid surface overlays: {error}") from error

    material_record = inspected["manifest"]["ground_material"]
    zone_root = inspected["zone_root"]
    atlas_maps: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for role in RUNTIME_TEXTURE_ROLES:
        record = material_record["runtime_atlas"][role]
        atlas_path = resolve_zone_asset(zone_root, record["path"], f"atlas.{role}")
        atlas_maps[role] = _load_image_pixels(
            bpy, atlas_path, non_color=(role != "basecolor")
        )
        source_hashes[role] = record["sha256"]

    used_indices = {int(value) for value in profile_ids[profile_weights > 0].tolist()}
    requires_triplanar = any(
        contract["profile_table"][index].get("projection") == "world_triplanar"
        for index in used_indices
    )
    terrain_surface = None
    if requires_triplanar:
        try:
            mesh = read_fvtq(inspected["lod0_fvtq_path"])
        except (OSError, ValueError) as error:
            raise BlenderTerrainQaError(
                f"Canonical LOD0 FVTQ cannot drive triplanar QA: {error}"
            ) from error
        terrain_surface = rasterize_fvtq_surface(
            mesh,
            stitch_mask=inspected["lod0_stitch_mask"],
            output_size=size,
        )

    result = compose_reference_pbr_maps(
        profile_ids,
        profile_weights,
        atlas_maps,
        contract["profile_table"],
        tile_origin_l93_m=inspected["manifest"]["tile_origin_l93_m"],
        output_size=size,
        ground_confidence=ground_confidence,
        ground_orientation=ground_orientation,
        surface_overlays=surface_overlays,
        terrain_surface=terrain_surface,
    )
    result.update(
        {
            "atlas_sha256": source_hashes,
            "ground_profile_ids_sha256": _sha256(
                composition_paths["ground_profile_ids"]
            ),
            "ground_profile_weights_sha256": _sha256(
                composition_paths["ground_profile_weights"]
            ),
            **(
                {
                    "ground_confidence_sha256": _sha256(
                        composition_paths["ground_confidence"]
                    ),
                    "ground_orientation_sha256": _sha256(
                        composition_paths["ground_orientation"]
                    ),
                }
                if "ground_confidence" in composition_paths
                else {}
            ),
        }
    )
    return result


def _install_reference_material(
    bpy: Any,
    terrain_objects: Sequence[Any],
    lod: int,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    material = bpy.data.materials.new("FireViewerGroundSurfaceReference")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in tuple(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    images = {
        role: _create_float_image(
            bpy, f"FireViewerReference_{role}", reference["maps"][role]
        )
        for role in RUNTIME_TEXTURE_ROLES
    }
    textures = {}
    for role, image in images.items():
        texture = nodes.new("ShaderNodeTexImage")
        texture.name = f"FireViewer_{role}"
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "EXTEND"
        textures[role] = texture
    orm = nodes.new("ShaderNodeSeparateColor")
    normal = nodes.new("ShaderNodeNormalMap")
    normal_space = reference.get("normal_space")
    if normal_space not in {"TANGENT", "WORLD"}:
        raise BlenderTerrainQaError("Reference normal-space contract is invalid")
    normal.space = normal_space
    height = nodes.new("ShaderNodeSeparateColor")
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.15
    material.node_tree.links.new(
        textures["basecolor"].outputs["Color"], shader.inputs["Base Color"]
    )
    material.node_tree.links.new(textures["orm"].outputs["Color"], orm.inputs["Color"])
    material.node_tree.links.new(orm.outputs["Green"], shader.inputs["Roughness"])
    material.node_tree.links.new(orm.outputs["Blue"], shader.inputs["Metallic"])
    material.node_tree.links.new(
        textures["normal"].outputs["Color"], normal.inputs["Color"]
    )
    material.node_tree.links.new(
        textures["height"].outputs["Color"], height.inputs["Color"]
    )
    material.node_tree.links.new(height.outputs["Red"], bump.inputs["Height"])
    material.node_tree.links.new(normal.outputs["Normal"], bump.inputs["Normal"])
    material.node_tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    lod_aov = nodes.new("ShaderNodeOutputAOV")
    lod_aov.name = "FireViewerTerrainLodAov"
    lod_aov.aov_name = TERRAIN_AOV
    lod_aov.inputs["Value"].default_value = float(lod)
    coverage_aov = nodes.new("ShaderNodeOutputAOV")
    coverage_aov.name = "FireViewerTerrainCoverageAov"
    coverage_aov.aov_name = COVERAGE_AOV
    coverage_aov.inputs["Value"].default_value = 1.0
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    for terrain in terrain_objects:
        uv_layer = terrain.data.uv_layers.get(
            "FireViewerTileUv"
        ) or terrain.data.uv_layers.new(name="FireViewerTileUv")
        xs = [vertex.co.x for vertex in terrain.data.vertices]
        ys = [vertex.co.y for vertex in terrain.data.vertices]
        minimum_x, maximum_x = min(xs), max(xs)
        minimum_y, maximum_y = min(ys), max(ys)
        if maximum_x <= minimum_x or maximum_y <= minimum_y:
            raise BlenderTerrainQaError("Imported terrain has invalid XY bounds")
        for polygon in terrain.data.polygons:
            for loop_index in polygon.loop_indices:
                vertex = terrain.data.vertices[
                    terrain.data.loops[loop_index].vertex_index
                ]
                uv_layer.data[loop_index].uv = (
                    (vertex.co.x - minimum_x) / (maximum_x - minimum_x),
                    (vertex.co.y - minimum_y) / (maximum_y - minimum_y),
                )
        terrain.data.materials.clear()
        terrain.data.materials.append(material)
    return {
        "material_name": material.name,
        "model": "FireViewerGroundSurface_v2 Blender reference",
        "connected_channels": [
            "basecolor",
            "normal",
            "height_bump",
            "orm",
            "orientation_projection",
        ],
        "normal_space": normal_space,
        "aovs": [TERRAIN_AOV, COVERAGE_AOV],
    }


def _frame_camera(bpy: Any, terrain_objects: Sequence[Any]) -> Any:
    from mathutils import Vector

    corners = []
    for terrain in terrain_objects:
        corners.extend(
            terrain.matrix_world @ Vector(corner) for corner in terrain.bound_box
        )
    minimum_x = min(point.x for point in corners)
    maximum_x = max(point.x for point in corners)
    minimum_y = min(point.y for point in corners)
    maximum_y = max(point.y for point in corners)
    maximum_z = max(point.z for point in corners)
    camera_data = bpy.data.cameras.new("FireViewerPrimaryCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(maximum_x - minimum_x, maximum_y - minimum_y) * 1.05
    camera_data.lens = 50.0
    camera_data.clip_start = 0.1
    camera_data.clip_end = max(10_000.0, camera_data.ortho_scale * 4.0)
    camera = bpy.data.objects.new("FireViewerPrimaryCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (
        (minimum_x + maximum_x) * 0.5,
        (minimum_y + maximum_y) * 0.5,
        maximum_z + max(1000.0, camera_data.ortho_scale),
    )
    # Blender cameras look along local -Z; identity rotation is top-down.
    camera.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = camera
    return camera


def _frame_oblique_camera(bpy: Any, terrain_objects: Sequence[Any]) -> Any:
    from mathutils import Vector

    corners = [
        terrain.matrix_world @ Vector(corner)
        for terrain in terrain_objects
        for corner in terrain.bound_box
    ]
    minimum = Vector(
        (
            min(point.x for point in corners),
            min(point.y for point in corners),
            min(point.z for point in corners),
        )
    )
    maximum = Vector(
        (
            max(point.x for point in corners),
            max(point.y for point in corners),
            max(point.z for point in corners),
        )
    )
    centre = (minimum + maximum) * 0.5
    span = max(maximum.x - minimum.x, maximum.y - minimum.y, 500.0)
    camera_data = bpy.data.cameras.new("FireViewerObliqueProofCamera")
    camera_data.type = "PERSP"
    camera_data.lens = 46.0
    camera_data.clip_start = 1.0
    camera_data.clip_end = 20_000.0
    camera = bpy.data.objects.new("FireViewerObliqueProofCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = centre + Vector((-1.20 * span, -1.20 * span, 1.00 * span))
    camera.rotation_euler = (
        (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    bpy.context.scene.camera = camera
    return camera


def validate_primary_terrain_aovs(
    lod_values: Any,
    coverage_values: Any,
    *,
    expected_lod: int = 0,
) -> dict[str, Any]:
    """Pure fail-closed AOV gate used by Blender and unit tests."""

    import numpy as np

    lod_array = np.asarray(lod_values, dtype=np.float64).ravel()
    coverage_array = np.asarray(coverage_values, dtype=np.float64).ravel()
    if lod_array.size == 0 or lod_array.shape != coverage_array.shape:
        raise BlenderTerrainQaError("Terrain AOV samples are empty or misaligned")
    invalid_coverage = int(
        np.count_nonzero(
            ~np.isfinite(coverage_array)
            | (coverage_array <= 0.5)
            | (coverage_array > 1.0 + 1.0e-6)
        )
    )
    if invalid_coverage:
        raise BlenderTerrainQaError(
            f"Terrain coverage AOV is absent or incomplete: {invalid_coverage} pixels"
        )
    lod_error = np.abs(lod_array - float(expected_lod))
    invalid_lod = int(np.count_nonzero(~np.isfinite(lod_array) | (lod_error > 1.0e-6)))
    if expected_lod != 0 or invalid_lod:
        raise BlenderTerrainQaError(
            f"Primary camera AOV contains forbidden terrain LOD pixels: {invalid_lod}"
        )
    return {
        "terrain_pixel_count": int(lod_array.size),
        "invalid_lod_pixel_count": invalid_lod,
        "invalid_coverage_pixel_count": invalid_coverage,
        "maximum_lod_absolute_error": float(np.max(lod_error)),
    }


def _render_and_validate_aov(
    bpy: Any,
    terrain_objects: Sequence[Any],
    *,
    lod: int,
    resolution: int,
    render_exr: Path,
    coverage_exr: Path,
    beauty_topdown: Path,
    beauty_oblique: Path,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    view_layer = scene.view_layers[0]
    for name in (TERRAIN_AOV, COVERAGE_AOV):
        aov = view_layer.aovs.add()
        aov.name = name
        aov.type = "VALUE"
    material_report = _install_reference_material(bpy, terrain_objects, lod, reference)
    topdown_camera = _frame_camera(bpy, terrain_objects)
    render_exr = _require_d_path(render_exr, "render_exr")
    coverage_exr = _require_d_path(coverage_exr, "coverage_exr")
    beauty_topdown = _require_d_path(beauty_topdown, "beauty_topdown")
    beauty_oblique = _require_d_path(beauty_oblique, "beauty_oblique")
    oblique_render_exr = _require_d_path(
        render_exr.with_name(f"{render_exr.stem}-oblique{render_exr.suffix}"),
        "oblique_render_exr",
    )
    oblique_coverage_exr = _require_d_path(
        coverage_exr.with_name(f"{coverage_exr.stem}-oblique{coverage_exr.suffix}"),
        "oblique_coverage_exr",
    )
    for path in (
        render_exr,
        coverage_exr,
        beauty_topdown,
        beauty_oblique,
        oblique_render_exr,
        oblique_coverage_exr,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    world = scene.world or bpy.data.worlds.new("FireViewerQaWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise BlenderTerrainQaError("Blender world has no Background node")
    background.inputs["Color"].default_value = (0.025, 0.035, 0.05, 1.0)
    background.inputs["Strength"].default_value = 0.35
    sun_data = bpy.data.lights.new("FireViewerQaSun", type="SUN")
    sun_data.energy = 2.5
    sun_data.angle = math.radians(4.0)
    sun = bpy.data.objects.new("FireViewerQaSun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(28.0), math.radians(-18.0), math.radians(32.0))

    def saved_pixels(path: Path, *, file_format: str) -> tuple[Any, int, int]:
        render_result = bpy.data.images.get("Render Result")
        if render_result is None:
            raise BlenderTerrainQaError("Blender did not produce a render result")
        scene.render.image_settings.file_format = file_format
        scene.render.image_settings.color_mode = "RGBA"
        if file_format == "PNG":
            scene.render.image_settings.color_depth = "8"
        elif file_format == "OPEN_EXR":
            scene.render.image_settings.color_depth = "32"
        render_result.save_render(filepath=str(path), scene=scene)
        image = bpy.data.images.load(str(path), check_existing=False)
        try:
            width, height = (int(value) for value in image.size)
            pixels = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            if pixels.size != width * height * 4:
                raise BlenderTerrainQaError(
                    f"Unexpected pixel layout for {path.name}: {pixels.size} values"
                )
            return pixels.reshape((height, width, 4)), width, height
        finally:
            bpy.data.images.remove(image)

    def visual_metrics(
        pixels: Any, terrain_mask: Any, *, label: str, minimum_coverage: float
    ) -> dict[str, Any]:
        terrain_rgb = np.asarray(pixels[:, :, :3][terrain_mask], dtype=np.float64)
        if terrain_rgb.size == 0 or not np.isfinite(terrain_rgb).all():
            raise BlenderTerrainQaError(f"{label} beauty has no finite terrain pixels")
        variance = float(np.var(terrain_rgb))
        channel_ranges = np.ptp(terrain_rgb, axis=0)
        quantized = np.clip(np.floor(terrain_rgb * 255.0 + 0.5), 0, 255).astype(
            np.uint8
        )
        distinct_colours = int(np.unique(quantized, axis=0).shape[0])
        coverage_ratio = float(np.count_nonzero(terrain_mask) / terrain_mask.size)
        if (
            variance <= 1.0e-6
            or float(np.max(channel_ranges)) <= 0.01
            or distinct_colours < 16
        ):
            raise BlenderTerrainQaError(
                f"{label} textured beauty is uniform or has insufficient colour evidence"
            )
        if coverage_ratio < minimum_coverage:
            raise BlenderTerrainQaError(
                f"{label} terrain framing is too small: {coverage_ratio:.4f}"
            )
        return {
            "terrain_pixel_count": int(np.count_nonzero(terrain_mask)),
            "frame_coverage_ratio": coverage_ratio,
            "rgb_variance": variance,
            "rgb_mean": [float(value) for value in np.mean(terrain_rgb, axis=0)],
            "rgb_channel_range": [float(value) for value in channel_ranges],
            "distinct_rgb8_count": distinct_colours,
        }

    scene.use_nodes = True
    node_tree = scene.node_tree
    if node_tree is None:
        raise BlenderTerrainQaError("Blender scene has no compositor node tree")

    def render_aov(name: str, path: Path, camera: Any) -> Any:
        scene.use_nodes = True
        scene.camera = camera
        node_tree.nodes.clear()
        render_layers = node_tree.nodes.new("CompositorNodeRLayers")
        composite = node_tree.nodes.new("CompositorNodeComposite")
        aov_output = render_layers.outputs.get(name)
        if aov_output is None:
            raise BlenderTerrainQaError(
                f"Blender compositor did not expose the {name!r} AOV"
            )
        node_tree.links.new(aov_output, composite.inputs["Image"])
        bpy.ops.render.render(write_still=False)
        pixels, width, height = saved_pixels(path, file_format="OPEN_EXR")
        if (width, height) != (resolution, resolution):
            raise BlenderTerrainQaError(f"{name} AOV proof has unexpected dimensions")
        return pixels

    scene.use_nodes = False
    scene.camera = topdown_camera
    bpy.ops.render.render(write_still=False)
    beauty_pixels, beauty_width, beauty_height = saved_pixels(
        beauty_topdown, file_format="PNG"
    )
    if (beauty_width, beauty_height) != (resolution, resolution):
        raise BlenderTerrainQaError("Top-down beauty proof has unexpected dimensions")

    coverage_pixels = render_aov(COVERAGE_AOV, coverage_exr, topdown_camera)
    terrain_mask = coverage_pixels[:, :, 0] > 0.5
    terrain_indices = np.flatnonzero(terrain_mask.ravel())
    if terrain_indices.size == 0:
        raise BlenderTerrainQaError("Primary camera rendered no terrain pixels")
    coverage_values = coverage_pixels[:, :, 0].ravel()[terrain_indices]
    aov_pixels = render_aov(TERRAIN_AOV, render_exr, topdown_camera)
    lod_values = aov_pixels[:, :, 0].ravel()[terrain_indices]
    aov_metrics = validate_primary_terrain_aovs(
        lod_values, coverage_values, expected_lod=lod
    )
    topdown_metrics = visual_metrics(
        beauty_pixels, terrain_mask, label="Top-down", minimum_coverage=0.70
    )

    oblique_camera = _frame_oblique_camera(bpy, terrain_objects)
    scene.use_nodes = False
    scene.camera = oblique_camera
    bpy.ops.render.render(write_still=False)
    oblique_pixels, oblique_width, oblique_height = saved_pixels(
        beauty_oblique, file_format="PNG"
    )
    if (oblique_width, oblique_height) != (resolution, resolution):
        raise BlenderTerrainQaError("Oblique beauty proof has unexpected dimensions")
    oblique_coverage = render_aov(COVERAGE_AOV, oblique_coverage_exr, oblique_camera)
    oblique_mask = oblique_coverage[:, :, 0] > 0.5
    oblique_indices = np.flatnonzero(oblique_mask.ravel())
    if oblique_indices.size == 0:
        raise BlenderTerrainQaError("Oblique primary camera rendered no terrain pixels")
    oblique_lod = render_aov(TERRAIN_AOV, oblique_render_exr, oblique_camera)
    oblique_aov_metrics = validate_primary_terrain_aovs(
        oblique_lod[:, :, 0].ravel()[oblique_indices],
        oblique_coverage[:, :, 0].ravel()[oblique_indices],
        expected_lod=lod,
    )
    oblique_metrics = visual_metrics(
        oblique_pixels, oblique_mask, label="Oblique", minimum_coverage=0.18
    )
    return {
        "material": material_report,
        "beauty": {
            "topdown": {
                "path": beauty_topdown.name,
                "sha256": _sha256(beauty_topdown),
                "camera": "orthographic_full_tile",
                "background": "opaque_neutral",
                **topdown_metrics,
            },
            "oblique": {
                "path": beauty_oblique.name,
                "sha256": _sha256(beauty_oblique),
                "camera": "distant_oblique_full_tile",
                "background": "opaque_neutral",
                **oblique_metrics,
            },
        },
        "aov": {
            "validated_primary_views": ["topdown", "oblique"],
            "lod": {
                "name": TERRAIN_AOV,
                "path": render_exr.name,
                "sha256": _sha256(render_exr),
                "expected_value": 0,
                "terrain_pixel_count": int(terrain_indices.size),
                "invalid_pixel_count": aov_metrics["invalid_lod_pixel_count"],
                "maximum_absolute_error": aov_metrics["maximum_lod_absolute_error"],
            },
            "coverage": {
                "name": COVERAGE_AOV,
                "path": coverage_exr.name,
                "sha256": _sha256(coverage_exr),
                "expected_value": 1,
                "invalid_pixel_count": aov_metrics["invalid_coverage_pixel_count"],
            },
            "oblique_lod": {
                "name": TERRAIN_AOV,
                "path": oblique_render_exr.name,
                "sha256": _sha256(oblique_render_exr),
                "expected_value": 0,
                "terrain_pixel_count": int(oblique_indices.size),
                "invalid_pixel_count": oblique_aov_metrics["invalid_lod_pixel_count"],
                "maximum_absolute_error": oblique_aov_metrics[
                    "maximum_lod_absolute_error"
                ],
            },
            "oblique_coverage": {
                "name": COVERAGE_AOV,
                "path": oblique_coverage_exr.name,
                "sha256": _sha256(oblique_coverage_exr),
                "expected_value": 1,
                "invalid_pixel_count": oblique_aov_metrics[
                    "invalid_coverage_pixel_count"
                ],
            },
        },
    }


def run_blender_qa(arguments: argparse.Namespace) -> dict[str, Any]:
    try:
        import bpy
    except ModuleNotFoundError as error:  # pragma: no cover - Blender only
        raise BlenderTerrainQaError("This command must run inside Blender") from error

    inspected = inspect_package(arguments.package)
    if inspected["selected_lod"] != 0:
        raise BlenderTerrainQaError("Primary camera stage does not select LOD0")
    surface_library_visual_receipt = (
        validate_surface_library_visual_receipt(
            arguments.surface_library_acceptance_receipt, inspected
        )
        if arguments.surface_library_acceptance_receipt is not None
        else None
    )
    if tuple(bpy.app.version[:2]) != (4, 5):
        raise BlenderTerrainQaError(
            f"Blender 4.5 LTS is required, got {bpy.app.version_string}"
        )
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_arguments = _operator_arguments(
        bpy.ops.wm.usd_import,
        {
            "filepath": str(inspected["stage_path"]),
            "import_cameras": False,
            "import_lights": False,
            "import_materials": True,
            "import_textures_mode": "IMPORT_NONE",
            "read_mesh_attributes": True,
            "attr_import_mode": "ALL",
            "validate_meshes": True,
            "relative_path": True,
        },
    )
    result = bpy.ops.wm.usd_import(**import_arguments)
    if "FINISHED" not in result:
        raise BlenderTerrainQaError(f"Blender USD import failed: {result}")
    terrain_objects = _terrain_objects(bpy)
    if len(terrain_objects) != 1:
        raise BlenderTerrainQaError(
            f"Expected one selected terrain mesh, imported {len(terrain_objects)}"
        )
    terrain = terrain_objects[0]
    expected = inspected["metrics"]
    vertex_count = len(terrain.data.vertices)
    triangle_count = len(terrain.data.polygons)
    if vertex_count != expected.get("vertex_count"):
        raise BlenderTerrainQaError("Blender vertex count differs from FVTQ manifest")
    if triangle_count != expected.get("triangle_count"):
        raise BlenderTerrainQaError("Blender triangle count differs from FVTQ manifest")
    reference = _reference_pbr_maps(
        bpy,
        inspected,
        max(256, int(arguments.resolution)),
    )
    render_report = _render_and_validate_aov(
        bpy,
        terrain_objects,
        lod=0,
        resolution=arguments.resolution,
        render_exr=arguments.render_exr,
        coverage_exr=arguments.coverage_exr,
        beauty_topdown=arguments.beauty_topdown,
        beauty_oblique=arguments.beauty_oblique,
        reference=reference,
    )
    manifest = inspected["manifest"]
    material_contract = inspected["ground_material_contract"]
    runtime_shader = material_contract.get("runtime_shader")
    runtime_textured_qualified = material_contract.get(
        "schema"
    ) == "fireviewer.ground-material-contract.v1" or (
        isinstance(runtime_shader, Mapping)
        and runtime_shader.get("production_textured_runtime_qualified") is True
    )
    surface_library_visual_accepted = surface_library_visual_receipt is not None
    production_visual_gate_passed = (
        surface_library_visual_accepted and runtime_textured_qualified
    )
    binary_path = Path(bpy.app.binary_path).resolve()
    return {
        "schema": RECEIPT_SCHEMA,
        "status": (
            "accepted_blender_textured_visual"
            if production_visual_gate_passed
            else (
                "textured_reference_probe_runtime_shader_pending"
                if surface_library_visual_accepted
                else "textured_technical_probe_pending_surface_library"
            )
        ),
        "geometry_lod_status": "accepted_blender_geometry_lod",
        "production_visual_gate_passed": production_visual_gate_passed,
        "runtime_textured_operational": runtime_textured_qualified,
        "runtime_shader": runtime_shader,
        "reference_material_only": not runtime_textured_qualified,
        "human_visual_acceptance": material_contract["visual_acceptance"],
        "source_library_status": (
            material_contract.get("source_library", {}).get("status")
            or material_contract.get("atlas_catalog_status")
        ),
        "surface_library_visual_acceptance_receipt": (surface_library_visual_receipt),
        "surface_library_acceptance_receipt_sha256": (
            surface_library_visual_receipt["sha256"]
            if surface_library_visual_receipt is not None
            else None
        ),
        "blender_version": ".".join(str(value) for value in bpy.app.version),
        "blender_binary_sha256": _sha256(binary_path),
        "package_manifest_sha256": _sha256(inspected["manifest_path"]),
        "root_stage_sha256": _sha256(inspected["stage_path"]),
        "ground_material_contract_sha256": _sha256(
            inspected["ground_material_contract_path"]
        ),
        "tile_id": manifest["tile_id"],
        "selected_lod": 0,
        "imported_mesh_count": 1,
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
        "render_resolution": [arguments.resolution, arguments.resolution],
        "reference_material": {
            key: value for key, value in reference.items() if key != "maps"
        },
        **render_report,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    raw = (
        list(sys.argv[sys.argv.index("--") + 1 :])
        if arguments is None and "--" in sys.argv
        else arguments
    )
    parsed = _parse_arguments(raw or ())
    parsed.package = _require_d_path(parsed.package, "package")
    parsed.report = _require_d_path(parsed.report, "report")
    parsed.render_exr = _require_d_path(parsed.render_exr, "render_exr")
    parsed.coverage_exr = _require_d_path(parsed.coverage_exr, "coverage_exr")
    parsed.beauty_topdown = _require_d_path(parsed.beauty_topdown, "beauty_topdown")
    parsed.beauty_oblique = _require_d_path(parsed.beauty_oblique, "beauty_oblique")
    if parsed.surface_library_acceptance_receipt is not None:
        parsed.surface_library_acceptance_receipt = _require_d_path(
            parsed.surface_library_acceptance_receipt,
            "surface_library_acceptance_receipt",
        )
    receipt = run_blender_qa(parsed)
    parsed.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = parsed.report.with_name(f".{parsed.report.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(parsed.report)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
