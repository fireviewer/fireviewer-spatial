"""Build a Blender control scene from a prepared FireViewer package.

Runtime dependencies: Blender's ``bpy`` and ``mathutils`` plus Python's
standard library. No rasterio, Shapely, pyproj or external numpy is imported.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_SCHEMA = "fireviewer.blender-preview-package.v2"
LEGACY_PACKAGE_SCHEMA = "fireviewer.blender-preview-package.v1"
MID_VEGETATION_SCHEMA = "fireviewer.vegetation-mid-distance-0m50.v1"
ORTHOPHOTO_SOURCE_SCHEMA = "fireviewer.ign-orthophoto-source.v1"
COLOR_MANAGEMENT_DISPLAY = "sRGB"
COLOR_MANAGEMENT_VIEW_TRANSFORM = "AgX"
COLOR_MANAGEMENT_LOOK = "None"
COLOR_MANAGEMENT_EXPOSURE = 0.0
COLOR_MANAGEMENT_GAMMA = 1.0
LIGHTING_RIG_SCHEMA = "fireviewer.blender-lighting-rig.v1"
LIGHTING_WORLD_NAME = "FireViewerWorld"
LIGHTING_BACKGROUND_NODE_NAME = "FireViewerAmbient"
LIGHTING_OUTPUT_NODE_NAME = "FireViewerWorldOutput"
LIGHTING_BACKGROUND_COLOR = (0.18, 0.22, 0.28, 1.0)
LIGHTING_BACKGROUND_STRENGTH = 0.38
LIGHTING_SUN_OBJECT_NAME = "FireViewerSun"
LIGHTING_SUN_DATA_NAME = "FireViewerSunData"
LIGHTING_SUN_ENERGY = 1.6
LIGHTING_SUN_ANGLE_DEGREES = 18.0
LIGHTING_SUN_ROTATION_DEGREES = (38.0, -24.0, -32.0)
TILED_COMPOSITING_SCHEMA = "fireviewer.tiled-compositing-strategy.v3"
GLOBAL_BASE_VISIBILITY_STRATEGY = (
    "global_monolith_or_partitioned_context_plus_complete_detail"
)
GLOBAL_TERRAIN_CONTEXT_COLLECTION_NAME = "GlobalTerrainContext"
GLOBAL_TERRAIN_CONTEXT_OBJECT_PREFIX = "GlobalTerrainContext_"
GLOBAL_TERRAIN_CONTEXT_SCHEMA = "fireviewer.global-terrain-context.v1"
GLOBAL_VECTOR_CONTEXT_COLLECTION_NAME = "GlobalVectorContext"
GLOBAL_VECTOR_CONTEXT_COLLECTION_PREFIX = "GlobalVectorContext_"
GLOBAL_VECTOR_CONTEXT_SCHEMA = "fireviewer.global-vector-context.v1"
DETAIL_TERRAIN_SOURCE_Z_OFFSET_M = 0.0
DETAIL_TERRAIN_RENDER_Z_OFFSET_M = 0.0
SCENE_DISTANCE_LOD_SCHEMA = "fireviewer.scene-distance-lod.v3"
SCENE_DISTANCE_LOD_NEAR_MAX_M = 1_500.0
SCENE_DISTANCE_LOD_FAR_MIN_M = 6_000.0
SCENE_DISTANCE_LOD_DETAIL_RADIUS_M = 750.0
SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES = 16
SCENE_DISTANCE_LOD_VIEW_FOOTPRINT_FACTOR = 1.2
SCENE_DISTANCE_LOD_SIMPLE_COLLECTIONS = (
    "FirePerimeter",
    "Buildings",
    "Roads",
    "Water",
)


def scene_distance_lod_strategy(
    view_distance_m: float,
    *,
    near_max_m: float = SCENE_DISTANCE_LOD_NEAR_MAX_M,
    far_min_m: float = SCENE_DISTANCE_LOD_FAR_MIN_M,
) -> dict[str, Any]:
    """Return the deterministic three-band visibility contract.

    The near boundary is exclusive and the far boundary inclusive:
    ``distance < near`` enables 0.5 m detail, ``near <= distance < far``
    keeps the global simple models, and ``distance >= far`` shows only the
    continuous MNT terrain.
    """

    distance = float(view_distance_m)
    near = float(near_max_m)
    far = float(far_min_m)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("view_distance_m must be finite and non-negative")
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError("near_max_m must be finite and strictly positive")
    if not math.isfinite(far) or far <= near:
        raise ValueError("far_min_m must be finite and greater than near_max_m")

    if distance < near:
        band = "near"
    elif distance < far:
        band = "medium"
    else:
        band = "far"
    return {
        "schema": SCENE_DISTANCE_LOD_SCHEMA,
        "band": band,
        "view_distance_m": distance,
        "near_max_m": near,
        "far_min_m": far,
        "terrain_visible": True,
        "terrain_lod_requested": "detail_0m50" if band == "near" else "global_2m",
        "simple_models_visible": band in {"near", "medium"},
        # This is only the distance-band request. The applied visibility also
        # requires complete resident coverage of the requested view radius.
        "detail_tiles_visible": band == "near",
        "vegetation_mode": "none",
        "far_content": "mnt_only",
        "medium_content": "mnt_plus_simple_models_without_tree_geometry",
        "near_content": "complete_0m50_tiles_plus_detail_vectors_when_covered",
    }


def tiled_compositing_strategy(
    visible_tile_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the deterministic global/detail compositing contract.

    The monolithic global terrain is the safe initial state. Global tree
    geometry is always disabled. The distance-LOD applicator switches to
    detail only after complete resident coverage of the requested view
    footprint is proven; it then replaces only the matching lightweight
    global chunks, while every other global terrain chunk remains visible.
    """

    visible = sorted({str(identifier) for identifier in visible_tile_ids})
    return {
        "schema": TILED_COMPOSITING_SCHEMA,
        "global_base_visibility_strategy": GLOBAL_BASE_VISIBILITY_STRATEGY,
        "global_terrain_visible": True,
        "global_context_visible": False,
        "global_vegetation_visible": False,
        "visible_detail_tile_ids": visible,
        "detail_terrain_source_z_offset_m": DETAIL_TERRAIN_SOURCE_Z_OFFSET_M,
        "detail_terrain_render_z_offset_m": DETAIL_TERRAIN_RENDER_Z_OFFSET_M,
    }


def _ensure_sibling_modules_available() -> None:
    source_path = globals().get("__file__")
    if source_path is None:
        return
    directory = str(Path(source_path).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)


def _read_json_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_package(path: Path) -> dict[str, Any]:
    package = _read_json_file(path)
    if package.get("schema") not in {PACKAGE_SCHEMA, LEGACY_PACKAGE_SCHEMA}:
        raise ValueError(
            f"Unsupported preview package schema: {package.get('schema')!r}"
        )
    for key in (
        "metadata",
        "terrain",
        "fire_perimeter",
        "analysis_buffer",
        "buildings",
        "vegetation",
    ):
        if key not in package:
            raise ValueError(f"Preview package is missing {key!r}")
    if package.get("schema") == PACKAGE_SCHEMA:
        for key in ("roads", "water"):
            if key not in package:
                raise ValueError(f"Preview package is missing {key!r}")
    origin = package["metadata"].get("origin_l93_m")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("Preview package has no valid Lambert-93 origin")
    return package


def load_mid_vegetation_package(path: Path) -> dict[str, Any]:
    package = _read_json_file(path)
    if package.get("schema") != MID_VEGETATION_SCHEMA:
        raise ValueError(
            f"Unsupported mid-distance vegetation schema: {package.get('schema')!r}"
        )
    for key in ("metadata", "tree_instances", "terrain", "statistics"):
        if key not in package:
            raise ValueError(f"Mid-distance package is missing {key!r}")
    metadata = package["metadata"]
    if metadata.get("crs") != "EPSG:2154":
        raise ValueError("Mid-distance package must use EPSG:2154")
    origin = metadata.get("origin_l93_m")
    bounds = metadata.get("bounds_l93_m")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("Mid-distance package has no valid Lambert-93 origin")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError("Mid-distance package has no valid Lambert-93 bounds")
    from tree_instances import decode_instance_attributes

    decode_instance_attributes(package["tree_instances"])
    return package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_orthophoto_source(path: Path) -> tuple[Path, list[float], dict[str, Any]]:
    source_file = path.expanduser().resolve()
    source = _read_json_file(source_file)
    if source.get("schema") != ORTHOPHOTO_SOURCE_SCHEMA:
        raise ValueError(
            f"Unsupported orthophoto source schema: {source.get('schema')!r}"
        )
    request = source.get("request", {})
    bounds = request.get("bounds_l93_m")
    if (
        request.get("crs") != "EPSG:2154"
        or not isinstance(bounds, list)
        or len(bounds) != 4
    ):
        raise ValueError("Orthophoto source must provide EPSG:2154 bounds")
    output = next(
        (
            item
            for item in source.get("outputs", [])
            if item.get("role") == "blender_rgb_jpeg"
        ),
        None,
    )
    if output is None:
        raise ValueError("Orthophoto source has no Blender JPEG output")
    image = source_file.parent / output["file_name"]
    if not image.is_file():
        raise FileNotFoundError(f"Orthophoto JPEG does not exist: {image}")
    observed_sha256 = _sha256(image)
    if observed_sha256 != output.get("sha256"):
        raise ValueError(
            "Orthophoto JPEG SHA-256 mismatch: "
            f"expected {output.get('sha256')}, got {observed_sha256}"
        )
    return image, [float(value) for value in bounds], source


def _package_identity(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return path.name, digest.hexdigest()


def _srgb_to_linear_channel(value: float) -> float:
    """Convert a display-referred sRGB channel to Blender scene-linear RGB."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"sRGB channel must be between 0 and 1, got {value!r}")
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _srgb_to_linear_color(
    color: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    red, green, blue, alpha = color
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha channel must be between 0 and 1, got {alpha!r}")
    return (
        _srgb_to_linear_channel(red),
        _srgb_to_linear_channel(green),
        _srgb_to_linear_channel(blue),
        alpha,
    )


def _configure_color_management(scene: Any) -> dict[str, Any]:
    """Apply one traceable Blender 5.2 display transform to every scene."""

    scene.display_settings.display_device = COLOR_MANAGEMENT_DISPLAY
    scene.view_settings.view_transform = COLOR_MANAGEMENT_VIEW_TRANSFORM
    scene.view_settings.look = COLOR_MANAGEMENT_LOOK
    scene.view_settings.exposure = COLOR_MANAGEMENT_EXPOSURE
    scene.view_settings.gamma = COLOR_MANAGEMENT_GAMMA
    settings = {
        "display_device": COLOR_MANAGEMENT_DISPLAY,
        "view_transform": COLOR_MANAGEMENT_VIEW_TRANSFORM,
        "look": COLOR_MANAGEMENT_LOOK,
        "exposure_stops": COLOR_MANAGEMENT_EXPOSURE,
        "gamma": COLOR_MANAGEMENT_GAMMA,
        "purpose": "common_contrasted_control_scene_display",
    }
    scene["fireviewer_color_management_json"] = json.dumps(settings, sort_keys=True)
    return settings


def _configure_lighting_rig(bpy: Any, scene: Any) -> dict[str, Any]:
    """Create the deterministic FireViewer world and sun without duplicates."""

    world = bpy.data.worlds.get(LIGHTING_WORLD_NAME)
    if world is None:
        world = bpy.data.worlds.new(LIGHTING_WORLD_NAME)
    scene.world = world
    world.color = LIGHTING_BACKGROUND_COLOR[:3]

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    background = nodes.new("ShaderNodeBackground")
    background.name = LIGHTING_BACKGROUND_NODE_NAME
    background.label = "FireViewer cool ambient"
    background.inputs["Color"].default_value = LIGHTING_BACKGROUND_COLOR
    background.inputs["Strength"].default_value = LIGHTING_BACKGROUND_STRENGTH

    output = nodes.new("ShaderNodeOutputWorld")
    output.name = LIGHTING_OUTPUT_NODE_NAME
    output.label = "FireViewer world output"
    links.new(background.outputs["Background"], output.inputs["Surface"])

    sun = bpy.data.objects.get(LIGHTING_SUN_OBJECT_NAME)
    if sun is None:
        sun_data = bpy.data.lights.get(LIGHTING_SUN_DATA_NAME)
        if sun_data is None:
            sun_data = bpy.data.lights.new(LIGHTING_SUN_DATA_NAME, type="SUN")
        sun = bpy.data.objects.new(LIGHTING_SUN_OBJECT_NAME, sun_data)
        scene.collection.objects.link(sun)
    elif getattr(sun, "type", None) != "LIGHT":
        raise ValueError(
            f"{LIGHTING_SUN_OBJECT_NAME!r} exists but is not a light object"
        )
    elif sun.name not in scene.collection.objects:
        scene.collection.objects.link(sun)

    sun.data.type = "SUN"
    sun.data.energy = LIGHTING_SUN_ENERGY
    sun.data.angle = math.radians(LIGHTING_SUN_ANGLE_DEGREES)
    sun.rotation_mode = "XYZ"
    sun.rotation_euler = tuple(
        math.radians(value) for value in LIGHTING_SUN_ROTATION_DEGREES
    )

    settings = {
        "schema": LIGHTING_RIG_SCHEMA,
        "world": {
            "name": world.name,
            "background_node": LIGHTING_BACKGROUND_NODE_NAME,
            "output_node": LIGHTING_OUTPUT_NODE_NAME,
            "color_linear_rgba": list(LIGHTING_BACKGROUND_COLOR),
            "strength": LIGHTING_BACKGROUND_STRENGTH,
        },
        "sun": {
            "object_name": LIGHTING_SUN_OBJECT_NAME,
            "data_name": sun.data.name,
            "type": "SUN",
            "energy": LIGHTING_SUN_ENERGY,
            "angle_degrees": LIGHTING_SUN_ANGLE_DEGREES,
            "rotation_mode": "XYZ",
            "rotation_euler_degrees": list(LIGHTING_SUN_ROTATION_DEGREES),
        },
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    scene["fireviewer_lighting_rig_json"] = encoded
    world["fireviewer_lighting_rig_json"] = encoded
    sun["fireviewer_lighting_rig_json"] = encoded
    return settings


def _terrain_orthophoto_material_config(
    material_name: str,
    *,
    boundary_tolerance_m: float,
    pack_image_in_blend: bool,
) -> Any:
    """Return the one non-double-graded terrain material contract.

    Values are explicit here so TerrainPreview, TerrainMidMontmaur and each
    0.5 m detail tile cannot silently diverge if library defaults evolve.
    """

    from terrain_texture import OrthophotoMaterialConfig

    return OrthophotoMaterialConfig(
        material_name=material_name,
        shader_mode="blender_balanced",
        texture_value=1.0,
        texture_saturation=1.0,
        principled_mix_fraction=0.45,
        emission_mix_fraction=0.55,
        emission_strength=1.0,
        boundary_tolerance_m=boundary_tolerance_m,
        pack_image_in_blend=pack_image_in_blend,
    )


def _material(
    bpy: Any, name: str, color: tuple[float, float, float, float], roughness: float
) -> Any:
    linear_color = _srgb_to_linear_color(color)
    material = bpy.data.materials.new(name)
    material.diffuse_color = linear_color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = linear_color
        principled.inputs["Roughness"].default_value = roughness
    return material


def _vegetation_material(bpy: Any) -> Any:
    """Create a dark, metrically scaled canopy shader for mid-distance reading."""
    material = _material(bpy, "MAT_Vegetation", (0.12, 0.31, 0.08, 1.0), 1.0)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    if principled is None:
        return material

    coordinates = nodes.new("ShaderNodeTexCoord")
    coordinates.name = "CanopyCoordinatesMetres"

    color_noise = nodes.new("ShaderNodeTexNoise")
    color_noise.name = "CanopyColorVariation"
    color_noise.noise_dimensions = "3D"
    color_noise.inputs["Scale"].default_value = 0.055
    color_noise.inputs["Detail"].default_value = 4.0
    color_noise.inputs["Roughness"].default_value = 0.65

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.name = "CanopyForestPalette"
    color_ramp.color_ramp.interpolation = "EASE"
    color_ramp.color_ramp.elements[0].position = 0.22
    color_ramp.color_ramp.elements[0].color = _srgb_to_linear_color(
        (0.055, 0.17, 0.035, 1.0)
    )
    color_ramp.color_ramp.elements[1].position = 0.78
    color_ramp.color_ramp.elements[1].color = _srgb_to_linear_color(
        (0.16, 0.36, 0.10, 1.0)
    )

    detail_noise = nodes.new("ShaderNodeTexNoise")
    detail_noise.name = "CanopyFineRelief"
    detail_noise.noise_dimensions = "3D"
    detail_noise.inputs["Scale"].default_value = 0.22
    detail_noise.inputs["Detail"].default_value = 2.5
    detail_noise.inputs["Roughness"].default_value = 0.55

    bump = nodes.new("ShaderNodeBump")
    bump.name = "CanopyFineBump"
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 1.0

    links.new(coordinates.outputs["Object"], color_noise.inputs["Vector"])
    links.new(coordinates.outputs["Object"], detail_noise.inputs["Vector"])
    links.new(color_noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(detail_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return material


def _noise_surface_material(
    bpy: Any,
    name: str,
    base_srgb: tuple[float, float, float, float],
    dark_srgb: tuple[float, float, float, float],
    light_srgb: tuple[float, float, float, float],
    roughness: float,
    color_scale_per_m: float,
    detail_scale_per_m: float,
    bump_strength: float,
    bump_distance_m: float,
) -> Any:
    material = _material(bpy, name, base_srgb, roughness)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    if principled is None:
        return material

    coordinates = nodes.new("ShaderNodeTexCoord")
    color_noise = nodes.new("ShaderNodeTexNoise")
    color_noise.noise_dimensions = "3D"
    color_noise.inputs["Scale"].default_value = color_scale_per_m
    color_noise.inputs["Detail"].default_value = 4.0
    color_noise.inputs["Roughness"].default_value = 0.62
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.interpolation = "EASE"
    color_ramp.color_ramp.elements[0].position = 0.25
    color_ramp.color_ramp.elements[0].color = _srgb_to_linear_color(dark_srgb)
    color_ramp.color_ramp.elements[1].position = 0.75
    color_ramp.color_ramp.elements[1].color = _srgb_to_linear_color(light_srgb)
    detail_noise = nodes.new("ShaderNodeTexNoise")
    detail_noise.noise_dimensions = "3D"
    detail_noise.inputs["Scale"].default_value = detail_scale_per_m
    detail_noise.inputs["Detail"].default_value = 2.5
    detail_noise.inputs["Roughness"].default_value = 0.55
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = bump_distance_m
    links.new(coordinates.outputs["Object"], color_noise.inputs["Vector"])
    links.new(coordinates.outputs["Object"], detail_noise.inputs["Vector"])
    links.new(color_noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(detail_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return material


def _reset_scene(bpy: Any) -> dict[str, Any]:
    for item in list(bpy.data.objects):
        bpy.data.objects.remove(item, do_unlink=True)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.node_groups,
        bpy.data.images,
    ):
        for datablock in list(datablocks):
            try:
                datablocks.remove(datablock, do_unlink=True)
            except TypeError:
                datablocks.remove(datablock)
    collections: dict[str, Any] = {}
    for name in (
        "Terrain",
        GLOBAL_TERRAIN_CONTEXT_COLLECTION_NAME,
        GLOBAL_VECTOR_CONTEXT_COLLECTION_NAME,
        "FirePerimeter",
        "Buildings",
        "Vegetation",
        "Roads",
        "Water",
        "GlobalTiles",
    ):
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        collections[name] = collection
    return collections


def _mesh_object(
    bpy: Any,
    name: str,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    material: Any | None,
    smooth: bool = False,
) -> Any:
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    if material is not None:
        mesh.materials.append(material)
    mesh.update()
    if smooth:
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    result = bpy.data.objects.new(name, mesh)
    result["vertex_count"] = len(vertices)
    result["face_count"] = len(faces)
    return result


def _global_context_object_name(chunk_id: str) -> str:
    return f"{GLOBAL_TERRAIN_CONTEXT_OBJECT_PREFIX}{chunk_id}"


def _assign_explicit_loop_uvs(
    mesh: Any,
    loop_uvs: Sequence[Sequence[float]],
    *,
    layer_name: str = "UVMap",
) -> None:
    """Assign already-interpolated UVs in Blender face-loop order."""

    if len(loop_uvs) != len(mesh.loops):
        raise ValueError(
            f"UV loop count mismatch: {len(loop_uvs)} != {len(mesh.loops)}"
        )
    layer = mesh.uv_layers.get(layer_name)
    if layer is None:
        layer = mesh.uv_layers.new(name=layer_name)
    for loop, uv in zip(mesh.loops, loop_uvs, strict=True):
        layer.data[loop.index].uv = (float(uv[0]), float(uv[1]))
    mesh.uv_layers.active = layer
    if hasattr(layer, "active_render"):
        layer.active_render = True


def build_global_terrain_context(
    bpy: Any,
    terrain_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    origin_l93_m: Sequence[float],
    terrain_material: Any,
    *,
    texture_bounds_l93_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Create one lightweight global terrain object per 500 m detail cell.

    The objects share the global orthophoto material.  They are hidden during
    far and medium views and become the continuous context around a complete
    near-detail working set.  Only a context object with the exact same tile
    identifier as a published HD tile is hidden.
    """

    from global_terrain_context import (
        GLOBAL_CONTEXT_CHUNK_ID,
        partition_global_terrain,
    )

    parent = bpy.data.collections.get(GLOBAL_TERRAIN_CONTEXT_COLLECTION_NAME)
    if parent is None:
        parent = bpy.data.collections.new(GLOBAL_TERRAIN_CONTEXT_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(parent)
    partition = partition_global_terrain(
        terrain_spec,
        manifest,
        origin_l93_m,
        texture_bounds_l93_m=texture_bounds_l93_m,
    )
    chunks = list(partition.tile_chunks.values())
    if not partition.context_chunk.is_empty:
        chunks.append(partition.context_chunk)
    for chunk in chunks:
        if chunk.is_empty:
            continue
        terrain_object = _mesh_object(
            bpy,
            _global_context_object_name(chunk.chunk_id),
            chunk.vertices,
            chunk.faces,
            terrain_material,
            smooth=True,
        )
        if chunk.loop_uvs is not None:
            _assign_explicit_loop_uvs(terrain_object.data, chunk.loop_uvs)
        terrain_object["fireviewer_global_context_schema"] = (
            GLOBAL_TERRAIN_CONTEXT_SCHEMA
        )
        terrain_object["fireviewer_global_context_chunk_id"] = chunk.chunk_id
        terrain_object["fireviewer_tile_id"] = chunk.tile_id or ""
        terrain_object["render_surface_triangulation"] = "fixed_nw_se_diagonal"
        terrain_object["detail_replacement_exclusive"] = chunk.tile_id is not None
        if chunk.bounds_l93_m is not None:
            terrain_object["fireviewer_core_bounds_l93_json"] = json.dumps(
                chunk.bounds_l93_m, separators=(",", ":")
            )
        parent.objects.link(terrain_object)
    parent["fireviewer_global_context_schema"] = GLOBAL_TERRAIN_CONTEXT_SCHEMA
    parent["fireviewer_manifest_tile_count"] = len(partition.manifest_tile_ids)
    parent["fireviewer_populated_tile_count"] = len(partition.populated_tile_ids)
    parent["fireviewer_context_chunk_id"] = GLOBAL_CONTEXT_CHUNK_ID
    parent["fireviewer_source_face_count"] = partition.validation.source_face_count
    parent["fireviewer_output_triangle_count"] = (
        partition.validation.output_triangle_count
    )
    parent["fireviewer_total_source_area_m2"] = (
        partition.validation.total_source_area_m2
    )
    parent["fireviewer_total_partitioned_area_m2"] = (
        partition.validation.total_partitioned_area_m2
    )
    parent["fireviewer_maximum_face_area_error_m2"] = (
        partition.validation.maximum_source_face_area_error_m2
    )
    _set_collection_visibility(parent, False)
    return {
        "partition": partition,
        "collection": parent,
        "object_count": len(chunks),
    }


def _set_global_terrain_context_visibility(
    bpy: Any,
    *,
    visible: bool,
    replaced_tile_ids: Sequence[str] = (),
) -> list[str]:
    """Show the complete global context except exact published HD cells."""

    parent = bpy.data.collections.get(GLOBAL_TERRAIN_CONTEXT_COLLECTION_NAME)
    if parent is None:
        if visible:
            raise ValueError("The global terrain context collection is missing")
        return []
    replaced = {str(identifier) for identifier in replaced_tile_ids}
    known = {
        str(item.get("fireviewer_tile_id"))
        for item in parent.objects
        if item.get("fireviewer_tile_id")
    }
    unknown = sorted(replaced - known)
    if unknown:
        raise ValueError(f"Unknown global terrain context tile ids: {unknown}")
    _set_collection_visibility(parent, visible)
    visible_ids: list[str] = []
    for item in parent.objects:
        tile_id = str(item.get("fireviewer_tile_id", ""))
        item_visible = bool(visible and tile_id not in replaced)
        item.hide_viewport = not item_visible
        item.hide_render = not item_visible
        if item_visible and tile_id:
            visible_ids.append(tile_id)
    return sorted(visible_ids)


def _global_vector_context_collection_name(chunk_id: str) -> str:
    return f"{GLOBAL_VECTOR_CONTEXT_COLLECTION_PREFIX}{chunk_id}"


def _ensure_global_vector_chunk_collections(
    bpy: Any, parent: Any, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    from global_terrain_context import GLOBAL_CONTEXT_CHUNK_ID

    result: dict[str, Any] = {}
    for tile in manifest.get("tiles", []):
        tile_id = str(tile["id"])
        collection = _new_child_collection(
            bpy, parent, _global_vector_context_collection_name(tile_id)
        )
        collection["fireviewer_global_vector_context_schema"] = (
            GLOBAL_VECTOR_CONTEXT_SCHEMA
        )
        collection["fireviewer_global_vector_context_chunk_id"] = tile_id
        collection["fireviewer_tile_id"] = tile_id
        collection["fireviewer_has_content"] = False
        result[tile_id] = collection
    context = _new_child_collection(
        bpy, parent, _global_vector_context_collection_name(GLOBAL_CONTEXT_CHUNK_ID)
    )
    context["fireviewer_global_vector_context_schema"] = GLOBAL_VECTOR_CONTEXT_SCHEMA
    context["fireviewer_global_vector_context_chunk_id"] = GLOBAL_CONTEXT_CHUNK_ID
    context["fireviewer_tile_id"] = ""
    context["fireviewer_has_content"] = False
    result[GLOBAL_CONTEXT_CHUNK_ID] = context
    return result


def build_global_vector_context(
    bpy: Any,
    package: Mapping[str, Any],
    manifest: Mapping[str, Any],
    origin_l93_m: Sequence[float],
    materials: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one exclusive simple-vector owner for every spatial LOD cell."""

    from global_terrain_context import GLOBAL_CONTEXT_CHUNK_ID
    from global_vector_context import (
        partition_building_prisms,
        partition_vector_surface,
    )

    parent = bpy.data.collections.get(GLOBAL_VECTOR_CONTEXT_COLLECTION_NAME)
    if parent is None:
        parent = bpy.data.collections.new(GLOBAL_VECTOR_CONTEXT_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(parent)
    if parent.children or parent.objects:
        raise ValueError("Global vector context collection must be empty before build")
    chunks = _ensure_global_vector_chunk_collections(bpy, parent, manifest)
    required_materials = {
        "buildings",
        "road_carriageway",
        "road_shoulders",
        "road_markings",
        "water_segments",
        "water_surfaces",
    }
    missing_materials = sorted(required_materials.difference(materials))
    if missing_materials:
        raise ValueError(
            "Global vector context materials are incomplete: "
            + ", ".join(missing_materials)
        )

    object_count = 0
    populated_chunk_ids: set[str] = set()
    building_partition = partition_building_prisms(
        package.get("buildings", {}).get("prisms", []),
        manifest,
        origin_l93_m,
    )
    building_groups: dict[str, Sequence[Mapping[str, Any]]] = {
        **building_partition.prisms_by_tile,
        GLOBAL_CONTEXT_CHUNK_ID: building_partition.context_prisms,
    }
    for chunk_id, prisms in building_groups.items():
        if not prisms:
            continue
        building_object = _prism_object(
            bpy,
            f"GlobalBuildingsSimple_{chunk_id}",
            prisms,
            materials["buildings"],
        )
        building_object["fireviewer_global_vector_context_schema"] = (
            GLOBAL_VECTOR_CONTEXT_SCHEMA
        )
        building_object["fireviewer_global_vector_layer"] = "buildings"
        building_object["fireviewer_tile_id"] = (
            "" if chunk_id == GLOBAL_CONTEXT_CHUNK_ID else chunk_id
        )
        chunks[chunk_id].objects.link(building_object)
        chunks[chunk_id]["fireviewer_has_content"] = True
        populated_chunk_ids.add(chunk_id)
        object_count += 1

    road_meshes = package.get("roads", {}).get("meshes", {})
    surface_layers = (
        (
            "road_carriageway",
            road_meshes.get("carriageway"),
            materials["road_carriageway"],
            False,
        ),
        (
            "road_left_shoulders",
            road_meshes.get("left_shoulders"),
            materials["road_shoulders"],
            False,
        ),
        (
            "road_right_shoulders",
            road_meshes.get("right_shoulders"),
            materials["road_shoulders"],
            False,
        ),
        (
            "road_center_markings",
            road_meshes.get("center_markings"),
            materials["road_markings"],
            False,
        ),
        (
            "water_segments",
            package.get("water", {}).get("segments", {}).get("mesh"),
            materials["water_segments"],
            True,
        ),
        (
            "water_surfaces",
            package.get("water", {}).get("surfaces", {}).get("mesh"),
            materials["water_surfaces"],
            False,
        ),
    )
    layer_statistics: dict[str, Any] = {}
    for layer_name, mesh, material, smooth in surface_layers:
        if not isinstance(mesh, Mapping) or not mesh.get("faces"):
            layer_statistics[layer_name] = {
                "source_face_count": 0,
                "output_triangle_count": 0,
                "populated_chunk_count": 0,
            }
            continue
        partition = partition_vector_surface(mesh, manifest, origin_l93_m)
        partition_chunks = list(partition.tile_chunks.values())
        if not partition.context_chunk.is_empty:
            partition_chunks.append(partition.context_chunk)
        layer_populated: set[str] = set()
        for chunk in partition_chunks:
            if chunk.is_empty:
                continue
            vector_object = _mesh_object(
                bpy,
                f"GlobalVectorSimple_{layer_name}_{chunk.chunk_id}",
                chunk.vertices,
                chunk.faces,
                material,
                smooth=smooth,
            )
            vector_object["fireviewer_global_vector_context_schema"] = (
                GLOBAL_VECTOR_CONTEXT_SCHEMA
            )
            vector_object["fireviewer_global_vector_layer"] = layer_name
            vector_object["fireviewer_tile_id"] = chunk.tile_id or ""
            chunks[chunk.chunk_id].objects.link(vector_object)
            chunks[chunk.chunk_id]["fireviewer_has_content"] = True
            populated_chunk_ids.add(chunk.chunk_id)
            layer_populated.add(chunk.chunk_id)
            object_count += 1
        validation = partition.validation
        layer_statistics[layer_name] = {
            "source_face_count": validation.source_face_count,
            "source_triangle_count": validation.source_triangle_count,
            "output_triangle_count": validation.output_triangle_count,
            "context_triangle_count": validation.context_triangle_count,
            "populated_chunk_count": len(layer_populated),
            "source_area_m2": validation.total_source_area_m2,
            "partitioned_area_m2": validation.total_partitioned_area_m2,
            "maximum_face_area_error_m2": (
                validation.maximum_source_face_area_error_m2
            ),
        }

    parent["fireviewer_global_vector_context_schema"] = GLOBAL_VECTOR_CONTEXT_SCHEMA
    parent["fireviewer_manifest_tile_count"] = len(manifest.get("tiles", []))
    parent["fireviewer_chunk_collection_count"] = len(chunks)
    parent["fireviewer_populated_chunk_count"] = len(populated_chunk_ids)
    parent["fireviewer_object_count"] = object_count
    parent["fireviewer_source_building_prism_count"] = (
        building_partition.source_prism_count
    )
    parent["fireviewer_context_building_prism_count"] = len(
        building_partition.context_prisms
    )
    parent["fireviewer_layer_statistics_json"] = json.dumps(
        layer_statistics, sort_keys=True, separators=(",", ":")
    )
    _set_collection_visibility(parent, False)
    for chunk in chunks.values():
        _set_collection_visibility(chunk, False)
    return {
        "collection": parent,
        "chunk_collections": chunks,
        "object_count": object_count,
        "populated_chunk_ids": sorted(populated_chunk_ids),
        "building_partition": building_partition,
        "layer_statistics": layer_statistics,
    }


def _set_global_vector_context_visibility(
    bpy: Any,
    *,
    visible: bool,
    replaced_tile_ids: Sequence[str] = (),
) -> list[str]:
    """Show simple vectors everywhere except exact published HD cells."""

    from global_terrain_context import GLOBAL_CONTEXT_CHUNK_ID
    from global_vector_context import vector_context_visibility

    parent = bpy.data.collections.get(GLOBAL_VECTOR_CONTEXT_COLLECTION_NAME)
    if parent is None:
        if visible:
            raise ValueError("The global vector context collection is missing")
        return []
    children_by_chunk = {
        str(child.get("fireviewer_global_vector_context_chunk_id")): child
        for child in parent.children
        if child.get("fireviewer_global_vector_context_chunk_id")
    }
    known_tile_ids = sorted(
        str(child.get("fireviewer_tile_id"))
        for child in parent.children
        if child.get("fireviewer_tile_id")
    )
    if visible:
        visibility = vector_context_visibility(known_tile_ids, replaced_tile_ids)
    else:
        visibility = {chunk_id: False for chunk_id in children_by_chunk}
    _set_collection_visibility(parent, visible)
    visible_ids: list[str] = []
    for chunk_id, child in children_by_chunk.items():
        child_visible = bool(visible and visibility.get(chunk_id, False))
        _set_collection_visibility(child, child_visible)
        tile_id = str(child.get("fireviewer_tile_id", ""))
        if (
            child_visible
            and tile_id
            and bool(child.get("fireviewer_has_content", False))
        ):
            visible_ids.append(tile_id)
    if visible and GLOBAL_CONTEXT_CHUNK_ID not in children_by_chunk:
        raise ValueError("The residual global vector context chunk is missing")
    return sorted(visible_ids)


def _triangulated_terrain_faces(
    faces: Sequence[Sequence[int]],
) -> list[tuple[int, ...]]:
    """Split every terrain quad along its deterministic NW-SE diagonal.

    Terrain packages order each regular-grid quad as NW, SW, SE, NE. Leaving
    the diagonal implicit lets Blender choose a render triangulation that can
    differ by metres from a bilinear CPU height sample on abrupt relief. The
    vector drape sampler uses this same explicit ``NW-SE`` split.
    """

    triangulated: list[tuple[int, ...]] = []
    for face in faces:
        if len(face) == 4:
            northwest, southwest, southeast, northeast = (int(index) for index in face)
            triangulated.append((northwest, southwest, southeast))
            triangulated.append((northwest, southeast, northeast))
        else:
            triangulated.append(tuple(int(index) for index in face))
    return triangulated


def _boundary_object(
    bpy: Any,
    name: str,
    rings: Sequence[Sequence[Sequence[float]]],
    material: Any,
    radius_m: float,
) -> Any:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = radius_m
    curve.bevel_resolution = 2
    for ring in rings:
        if len(ring) < 3:
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(len(ring) - 1)
        for point, coordinate in zip(spline.points, ring):
            point.co = (
                float(coordinate[0]),
                float(coordinate[1]),
                float(coordinate[2]),
                1.0,
            )
        spline.use_cyclic_u = True
    curve.materials.append(material)
    result = bpy.data.objects.new(name, curve)
    result["ring_count"] = len(rings)
    return result


def _resolve_tessellated_triangle(
    triangle: Sequence[Any],
    flattened_top_indices: Sequence[int],
    top_lookup: dict[tuple[float, float, float], int],
) -> list[int]:
    """Resolve Blender 5.2 integer tessellation or legacy Vector output."""
    if triangle and isinstance(triangle[0], int):
        return [flattened_top_indices[int(point)] for point in triangle]
    return [
        top_lookup[(round(point.x, 7), round(point.y, 7), round(point.z, 7))]
        for point in triangle
    ]


def _append_prism(
    prism: dict[str, Any],
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
) -> None:
    from mathutils import Vector
    from mathutils.geometry import tessellate_polygon

    base_z = float(prism["base_z"])
    top_z = float(prism.get("roof_z", base_z + float(prism["height"])))
    ground_z_rings = prism.get("ground_z_rings")
    top_lookup: dict[tuple[float, float, float], int] = {}
    tessellation_rings: list[list[Any]] = []
    tessellation_top_indices: list[int] = []

    for ring_index, ring in enumerate(prism["rings"]):
        if len(ring) < 3:
            continue
        ground_z_ring = (
            ground_z_rings[ring_index]
            if isinstance(ground_z_rings, list) and ring_index < len(ground_z_rings)
            else None
        )
        bottom_indices: list[int] = []
        top_indices: list[int] = []
        top_vectors: list[Any] = []
        for coordinate_index, (x, y) in enumerate(ring):
            local_x, local_y = float(x), float(y)
            bottom_z = (
                float(ground_z_ring[coordinate_index])
                if isinstance(ground_z_ring, list)
                and coordinate_index < len(ground_z_ring)
                else base_z
            )
            bottom_indices.append(len(vertices))
            vertices.append((local_x, local_y, bottom_z))
            top_index = len(vertices)
            top_indices.append(top_index)
            vertices.append((local_x, local_y, top_z))
            vector = Vector((local_x, local_y, top_z))
            top_vectors.append(vector)
            tessellation_top_indices.append(top_index)
            top_lookup[(round(local_x, 7), round(local_y, 7), round(top_z, 7))] = (
                top_index
            )
        for index in range(len(ring)):
            following = (index + 1) % len(ring)
            faces.append(
                (
                    bottom_indices[index],
                    bottom_indices[following],
                    top_indices[following],
                    top_indices[index],
                )
            )
        tessellation_rings.append(top_vectors)

    for triangle in tessellate_polygon(tessellation_rings):
        # Blender 5.2 returns integer offsets into the flattened input rings,
        # while older releases returned Vector objects. Support both APIs so
        # the same control package stays reproducible across Blender versions.
        indices = _resolve_tessellated_triangle(
            triangle, tessellation_top_indices, top_lookup
        )
        a, b, c = (vertices[index] for index in indices)
        signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        faces.append(tuple(indices if signed_area >= 0 else reversed(indices)))


def _prism_object(
    bpy: Any, name: str, prisms: Sequence[dict[str, Any]], material: Any
) -> Any:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for prism in prisms:
        _append_prism(prism, vertices, faces)
    result = _mesh_object(bpy, name, vertices, faces, material)
    result["prism_count"] = len(prisms)
    return result


def _same_origin(
    left: Sequence[float], right: Sequence[float], *, tolerance_m: float = 0.001
) -> bool:
    return len(left) == len(right) == 3 and all(
        math.isfinite(float(left_value))
        and math.isfinite(float(right_value))
        and abs(float(left_value) - float(right_value)) <= tolerance_m
        for left_value, right_value in zip(left, right, strict=True)
    )


def _new_child_collection(bpy: Any, parent: Any, name: str) -> Any:
    collection = bpy.data.collections.new(name)
    parent.children.link(collection)
    return collection


def _set_collection_visibility(collection: Any, visible: bool) -> None:
    collection.hide_viewport = not visible
    collection.hide_render = not visible


def _tile_collection_name(tile_id: str) -> str:
    return f"GlobalTile_{tile_id}"


_TILE_DATABLOCK_GROUPS = (
    "objects",
    "meshes",
    "curves",
    "materials",
    "images",
    "node_groups",
    "collections",
)


def _datablock_identity(item: Any) -> int:
    as_pointer = getattr(item, "as_pointer", None)
    if callable(as_pointer):
        pointer = int(as_pointer())
        if pointer:
            return pointer
    return id(item)


def _snapshot_tile_datablocks(bpy: Any) -> dict[str, set[int]]:
    return {
        group: {_datablock_identity(item) for item in getattr(bpy.data, group, ())}
        for group in _TILE_DATABLOCK_GROUPS
    }


def _new_tile_datablocks(
    bpy: Any, snapshot: Mapping[str, set[int]]
) -> dict[str, list[Any]]:
    return {
        group: [
            item
            for item in getattr(bpy.data, group, ())
            if _datablock_identity(item) not in snapshot.get(group, set())
        ]
        for group in _TILE_DATABLOCK_GROUPS
    }


def _tag_new_tile_datablocks(
    bpy: Any,
    tile_id: str,
    snapshot: Mapping[str, set[int]],
) -> list[str]:
    tagged: list[str] = []
    for group, items in _new_tile_datablocks(bpy, snapshot).items():
        for item in items:
            item["fireviewer_streaming_owner_tile_id"] = str(tile_id)
            tagged.append(f"{group}:{item.name}")
    return sorted(tagged)


def _rollback_new_tile_datablocks(
    bpy: Any,
    tile_id: str,
    snapshot: Mapping[str, set[int]],
) -> list[str]:
    """Remove every datablock created by one failed tile transaction."""

    created = _new_tile_datablocks(bpy, snapshot)
    removed: list[str] = []
    # Unlink users before their data.  Only identities absent from the baseline
    # are removed, so cached/shared materials and images remain untouched.
    for group in (
        "objects",
        "collections",
        "meshes",
        "curves",
        "materials",
        "node_groups",
        "images",
    ):
        datablocks = getattr(bpy.data, group, ())
        for item in list(created.get(group, ())):
            try:
                item["fireviewer_streaming_owner_tile_id"] = str(tile_id)
            except (AttributeError, TypeError):
                pass
            name = str(getattr(item, "name", "<unnamed>"))
            try:
                datablocks.remove(item, do_unlink=True)
            except TypeError:
                datablocks.remove(item)
            removed.append(f"{group}:{name}")
    remaining = {
        group: [str(getattr(item, "name", "<unnamed>")) for item in items]
        for group, items in _new_tile_datablocks(bpy, snapshot).items()
        if items
    }
    if remaining:
        raise RuntimeError(
            f"Tile {tile_id!r} rollback retained new datablocks: {remaining}"
        )
    return sorted(removed)


def _resident_tile_ids_within_budget(
    loaded_tile_ids: Sequence[str],
    requested_tile_ids: Sequence[str],
    maximum_resident_tile_count: int | None,
) -> list[str]:
    resident = sorted(
        {
            *(str(identifier) for identifier in loaded_tile_ids),
            *(str(identifier) for identifier in requested_tile_ids),
        }
    )
    if maximum_resident_tile_count is None:
        return resident
    if (
        isinstance(maximum_resident_tile_count, bool)
        or not isinstance(maximum_resident_tile_count, int)
        or maximum_resident_tile_count <= 0
    ):
        raise ValueError("maximum_resident_tile_count must be a positive integer")
    if len(resident) > maximum_resident_tile_count:
        raise ValueError(
            "Detail tile resident budget exceeded: "
            f"requested {len(resident)}, maximum {maximum_resident_tile_count}"
        )
    return resident


def _create_tile_collection_tree(
    bpy: Any, parent: Any, tile: dict[str, Any]
) -> tuple[Any, Any, Any]:
    root_name = _tile_collection_name(str(tile["id"]))
    tile_root = bpy.data.collections.get(root_name)
    if tile_root is None:
        tile_root = _new_child_collection(bpy, parent, root_name)
    elif tile_root.get("fireviewer_tile_id") != str(tile["id"]):
        raise ValueError(f"Collection name collision for tile {tile['id']!r}")
    terrain_name = f"Terrain_{tile['id']}"
    vegetation_name = f"Vegetation_{tile['id']}"
    terrain = bpy.data.collections.get(terrain_name)
    if terrain is None:
        terrain = _new_child_collection(bpy, tile_root, terrain_name)
    vegetation = bpy.data.collections.get(vegetation_name)
    if vegetation is None:
        vegetation = _new_child_collection(bpy, tile_root, vegetation_name)
    tile_root["fireviewer_tile_id"] = str(tile["id"])
    tile_root["fireviewer_tile_schema"] = "fireviewer.global-05m-tile.v1"
    tile_root["fireviewer_core_bounds_l93_json"] = json.dumps(
        tile["bounds_l93_m"], separators=(",", ":")
    )
    tile_root["fireviewer_processing_bounds_l93_json"] = json.dumps(
        tile["processing_bounds_l93_m"], separators=(",", ":")
    )
    if "fireviewer_tile_loaded" not in tile_root:
        tile_root["fireviewer_tile_loaded"] = False
    tile_root["fireviewer_instances_realized"] = False
    return tile_root, terrain, vegetation


def _iter_collection_objects(collection: Any) -> Any:
    all_objects = getattr(collection, "all_objects", None)
    if all_objects is not None:
        yield from all_objects
        return
    yield from collection.objects
    for child in collection.children:
        yield from _iter_collection_objects(child)


def _assert_collection_has_no_realize_instances(collection: Any) -> None:
    for item in _iter_collection_objects(collection):
        if bool(item.get("instances_realized", False)):
            raise ValueError(
                f"Tile collection {collection.name!r} contains realized instances"
            )
        for modifier in getattr(item, "modifiers", ()):
            node_group = getattr(modifier, "node_group", None)
            if node_group is None:
                continue
            if any(
                getattr(node, "bl_idname", "") == "GeometryNodeRealizeInstances"
                for node in node_group.nodes
            ):
                raise ValueError(
                    f"Tile collection {collection.name!r} contains Realize Instances"
                )


def _load_linked_tile_library(
    bpy: Any,
    tile: dict[str, Any],
    tile_root: Any,
    library_path: Path,
    requested_collection_name: str | None,
) -> None:
    with bpy.data.libraries.load(str(library_path), link=True) as (
        data_from,
        data_to,
    ):
        available = list(data_from.collections)
        candidates = [
            requested_collection_name,
            str(tile["id"]),
            _tile_collection_name(str(tile["id"])),
            "FireViewerTile",
        ]
        selected = next(
            (candidate for candidate in candidates if candidate in available), None
        )
        if selected is None and len(available) == 1:
            selected = available[0]
        if selected is None:
            raise ValueError(
                f"Tile library {library_path.name!r} has no unambiguous root "
                f"collection; available: {available}"
            )
        data_to.collections = [selected]
    linked = data_to.collections[0]
    if linked is None:
        raise ValueError(f"Failed to link tile library {library_path}")
    tile_root.children.link(linked)
    _assert_collection_has_no_realize_instances(linked)
    tile_root["fireviewer_asset_kind"] = "linked_blender_library"
    tile_root["fireviewer_asset_file_name"] = library_path.name


def _validate_detail_terrain_core_contract(
    tile: dict[str, Any], terrain_spec: dict[str, Any]
) -> None:
    """Reject detail terrain that cannot satisfy the crack-free tile contract."""

    geometric_bounds = terrain_spec.get("geometric_bounds_l93_m")
    if not isinstance(geometric_bounds, list) or len(geometric_bounds) != 4:
        raise ValueError(
            f"Tile {tile['id']!r} terrain has no exact geometric core bounds"
        )
    if any(
        abs(float(left) - float(right)) > 1e-6
        for left, right in zip(geometric_bounds, tile["bounds_l93_m"], strict=True)
    ):
        raise ValueError(
            f"Tile {tile['id']!r} terrain geometry does not reach its 500 m core"
        )
    if terrain_spec.get("boundary_sampling") != (
        "bilinear_processing_halo_at_exact_lambert93_core_coordinates"
    ):
        raise ValueError(
            f"Tile {tile['id']!r} terrain has no deterministic edge sampling"
        )
    if terrain_spec.get("adjacent_edge_contract") != (
        "coincident_xy_and_identical_sample_coordinates"
    ):
        raise ValueError(f"Tile {tile['id']!r} terrain has no adjacent-edge contract")


def _build_tile_from_source_packages(
    bpy: Any,
    tile: dict[str, Any],
    tile_root: Any,
    terrain_collection: Any,
    vegetation_collection: Any,
    package_path: Path,
    orthophoto_source_path: Path,
    global_origin: Sequence[float],
    *,
    global_vector_package: dict[str, Any] | None = None,
    detail_vector_materials: Mapping[str, Any] | None = None,
) -> None:
    package = load_mid_vegetation_package(package_path)
    tile_origin = package["metadata"]["origin_l93_m"]
    if not _same_origin(global_origin, tile_origin):
        raise ValueError(
            f"Tile {tile['id']!r} and the global package must share one origin"
        )
    package_bounds = package["metadata"]["bounds_l93_m"]
    if any(
        abs(float(left) - float(right)) > 0.01
        for left, right in zip(package_bounds, tile["bounds_l93_m"], strict=True)
    ):
        raise ValueError(
            f"Tile {tile['id']!r} package bounds do not match its 500 m core"
        )
    orthophoto = load_orthophoto_source(orthophoto_source_path)
    orthophoto_resolution_m = float(
        orthophoto[2].get("request", {}).get("nominal_resolution_m", 0.5)
    )
    if not math.isfinite(orthophoto_resolution_m) or orthophoto_resolution_m <= 0.0:
        raise ValueError("Tile orthophoto has no valid nominal resolution")
    orthophoto_resolution_token = f"{orthophoto_resolution_m:.2f}".replace(".", "m")
    terrain_spec = package["terrain"]
    _validate_detail_terrain_core_contract(tile, terrain_spec)
    terrain_object = _mesh_object(
        bpy,
        f"TerrainTile_{tile['id']}",
        terrain_spec["vertices"],
        _triangulated_terrain_faces(terrain_spec["faces"]),
        None,
        smooth=True,
    )
    terrain_object["fireviewer_tile_id"] = str(tile["id"])
    terrain_object["detail_role"] = "mnt_hd_0m50_exclusive_near_lod"
    terrain_object["source_pixel_size_m"] = json.dumps(
        terrain_spec.get("source_pixel_size_m")
    )
    terrain_object["sample_spacing_m"] = json.dumps(
        terrain_spec.get("sample_spacing_m")
    )
    terrain_object["geometric_bounds_l93_json"] = json.dumps(
        terrain_spec["geometric_bounds_l93_m"], separators=(",", ":")
    )
    terrain_object["boundary_sampling"] = terrain_spec["boundary_sampling"]
    terrain_object["adjacent_edge_contract"] = terrain_spec["adjacent_edge_contract"]
    terrain_object["render_surface_triangulation"] = "fixed_nw_se_diagonal"
    # Preserve source MNT elevations and render them without a display offset.
    # The global terrain is hidden whenever the complete detail footprint is
    # enabled, so no coplanar overlap needs to be disguised with a Z bias.
    terrain_object.location.z = DETAIL_TERRAIN_RENDER_Z_OFFSET_M
    terrain_object["source_vertex_z_offset_m"] = DETAIL_TERRAIN_SOURCE_Z_OFFSET_M
    terrain_object["render_transform_z_offset_m"] = DETAIL_TERRAIN_RENDER_Z_OFFSET_M
    terrain_object["altitude_measurement_reference"] = "source_mnt_vertex_z"
    tile_root["detail_terrain_source_z_offset_m"] = DETAIL_TERRAIN_SOURCE_Z_OFFSET_M
    tile_root["detail_terrain_render_z_offset_m"] = DETAIL_TERRAIN_RENDER_Z_OFFSET_M
    terrain_collection.objects.link(terrain_object)

    from terrain_texture import apply_georeferenced_orthophoto

    applied = apply_georeferenced_orthophoto(
        bpy,
        terrain_object,
        orthophoto[0],
        global_origin,
        orthophoto[1],
        config=_terrain_orthophoto_material_config(
            f"MAT_TerrainTile_{tile['id']}_IGN{orthophoto_resolution_token}",
            boundary_tolerance_m=0.01,
            # Keep tile images external. Packing hundreds of orthophotos would
            # recreate the monolithic .blend this index is designed to avoid.
            pack_image_in_blend=False,
        ),
    )
    terrain_object["texture_role"] = f"ign_bd_ortho_tiled_{orthophoto_resolution_token}"
    terrain_object["texture_nominal_resolution_m"] = orthophoto_resolution_m
    terrain_object["texture_image_file_name"] = orthophoto[0].name
    terrain_object["texture_uv_min_u"] = applied.statistics.minimum_u
    terrain_object["texture_uv_min_v"] = applied.statistics.minimum_v
    terrain_object["texture_uv_max_u"] = applied.statistics.maximum_u
    terrain_object["texture_uv_max_v"] = applied.statistics.maximum_v

    from tree_instances import build_blender_tree_system

    tree_system = build_blender_tree_system(
        bpy,
        package["tree_instances"],
        vegetation_collection,
        name=f"VegetationTile_{tile['id']}_0m50",
    )
    tree_system["fireviewer_tile_id"] = str(tile["id"])
    tree_system["detection_grid_m"] = 0.5
    tree_system["instances_realized"] = False
    statistics = package.get("statistics", {})
    if "post_detection_spacing_rejected_count" in statistics:
        tree_system["post_detection_spacing_rejected_count"] = int(
            statistics["post_detection_spacing_rejected_count"]
        )
    if "completeness_claim" in statistics:
        tree_system["completeness_claim"] = statistics["completeness_claim"]
    _assert_collection_has_no_realize_instances(vegetation_collection)

    if global_vector_package is not None:
        from detail_vector_lod import build_detail_tile_vectors

        materials = dict(detail_vector_materials or {})
        required_materials = {
            "buildings",
            "road_carriageway",
            "road_shoulders",
            "road_markings",
            "water_segments",
            "water_surfaces",
        }
        missing_materials = sorted(required_materials.difference(materials))
        if missing_materials:
            raise ValueError(
                "Detail vector LOD materials are incomplete: "
                + ", ".join(missing_materials)
            )
        detail_vectors = build_detail_tile_vectors(
            global_vector_package,
            package,
            tile["bounds_l93_m"],
            global_origin,
        )
        vector_collection = _new_child_collection(
            bpy, tile_root, f"Vectors_{tile['id']}"
        )
        building_collection = _new_child_collection(
            bpy, vector_collection, f"BuildingsDetail_{tile['id']}"
        )
        road_collection = _new_child_collection(
            bpy, vector_collection, f"RoadsDetail_{tile['id']}"
        )
        water_collection = _new_child_collection(
            bpy, vector_collection, f"WaterDetail_{tile['id']}"
        )

        building_prisms = detail_vectors["buildings"]["prisms"]
        if building_prisms:
            building_object = _prism_object(
                bpy,
                f"BuildingsDetail_{tile['id']}",
                building_prisms,
                materials["buildings"],
            )
            building_object["fireviewer_tile_id"] = str(tile["id"])
            building_object["detail_role"] = "buildings_draped_on_tile_mnt"
            building_collection.objects.link(building_object)

        road_material_by_layer = {
            "carriageway": materials["road_carriageway"],
            "left_shoulders": materials["road_shoulders"],
            "right_shoulders": materials["road_shoulders"],
            "center_markings": materials["road_markings"],
        }
        for layer, road_mesh in detail_vectors["roads"]["meshes"].items():
            if not road_mesh["faces"]:
                continue
            road_object = _mesh_object(
                bpy,
                f"RoadDetail_{layer}_{tile['id']}",
                road_mesh["vertices"],
                road_mesh["faces"],
                road_material_by_layer[layer],
                smooth=False,
            )
            road_object["fireviewer_tile_id"] = str(tile["id"])
            road_object["detail_role"] = "road_draped_on_tile_mnt"
            road_object["road_layer"] = layer
            road_collection.objects.link(road_object)

        for layer, material in (
            ("segments", materials["water_segments"]),
            ("surfaces", materials["water_surfaces"]),
        ):
            water_mesh = detail_vectors["water"][layer]["mesh"]
            if not water_mesh["faces"]:
                continue
            water_object = _mesh_object(
                bpy,
                f"WaterDetail_{layer}_{tile['id']}",
                water_mesh["vertices"],
                water_mesh["faces"],
                material,
                smooth=layer == "segments",
            )
            water_object["fireviewer_tile_id"] = str(tile["id"])
            water_object["detail_role"] = "water_draped_on_tile_mnt"
            water_collection.objects.link(water_object)

        vector_collection["fireviewer_tile_id"] = str(tile["id"])
        vector_collection["detail_vector_schema"] = detail_vectors["schema"]
        vector_collection["water_linear_render_source"] = detail_vectors["water"][
            "rendered_linear_source"
        ]
        vector_collection["water_courses_rendered"] = bool(
            detail_vectors["water"]["courses_rendered"]
        )
        tile_root["detail_vector_statistics_json"] = json.dumps(
            {
                "buildings": detail_vectors["buildings"]["statistics"],
                "roads": detail_vectors["roads"]["statistics"],
                "water_segments": detail_vectors["water"]["segments"]["statistics"],
                "water_surfaces": detail_vectors["water"]["surfaces"]["statistics"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        tile_root["detail_vector_lod_complete"] = True

    tile_root["fireviewer_asset_kind"] = "mid_package_plus_orthophoto"
    tile_root["fireviewer_asset_file_name"] = package_path.name
    tile_root["fireviewer_orthophoto_source_file_name"] = orthophoto_source_path.name
    tile_root["fireviewer_orthophoto_resolution_m"] = orthophoto_resolution_m


def load_global_tiles_into_scene(
    bpy: Any,
    manifest_path: str | Path,
    parent_collection: Any,
    global_origin: Sequence[float],
    *,
    selected_tile_ids: Sequence[str] | None = None,
    load_mode: str = "visible",
    focus_l93_m: Sequence[float] | None = None,
    visible_radius_m: float | None = None,
    maximum_resident_tile_count: int | None = (SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES),
    global_vector_package: dict[str, Any] | None = None,
    detail_vector_materials: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create all ready tile collections and load only the requested subset.

    ``visible`` is the safe default: a tile is loaded only when selected,
    declared visible by the manifest, or inside the focus radius. The resident
    budget is checked before the first package is materialized, preventing an
    accidental 475-tile scene or a partially loaded over-budget result.
    """

    from tiled_scene import (
        file_sha256,
        load_global_05m_manifest,
        ready_tiles,
        select_tile_asset,
        tile_is_visible,
    )

    if load_mode not in {"visible", "all_ready"}:
        raise ValueError("tile load_mode must be 'visible' or 'all_ready'")
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = load_global_05m_manifest(manifest_file)
    if not _same_origin(global_origin, manifest["origin_l93_m"]):
        raise ValueError("Global package and tile manifest must share one origin")
    all_ready = ready_tiles(manifest)
    manifest_sha256 = file_sha256(manifest_file)
    explicit_ids = set(selected_tile_ids or ())
    if explicit_ids:
        ready_tiles(manifest, explicit_ids)

    tile_states: list[tuple[dict[str, Any], Any, Any, Any, bool, bool]] = []
    already_loaded_ids: list[str] = []
    requested_load_ids: list[str] = []
    for tile in all_ready:
        tile_id = str(tile["id"])
        tile_root, terrain_collection, vegetation_collection = (
            _create_tile_collection_tree(bpy, parent_collection, tile)
        )
        visible = tile_is_visible(
            tile,
            focus_l93_m=focus_l93_m,
            radius_m=visible_radius_m,
            explicitly_selected=tile_id in explicit_ids,
        )
        should_load = load_mode == "all_ready" or visible
        already_loaded = bool(tile_root.get("fireviewer_tile_loaded", False))
        if already_loaded:
            already_loaded_ids.append(tile_id)
        if should_load:
            requested_load_ids.append(tile_id)
        tile_states.append(
            (
                tile,
                tile_root,
                terrain_collection,
                vegetation_collection,
                visible,
                should_load,
            )
        )

    resident_ids = _resident_tile_ids_within_budget(
        already_loaded_ids,
        requested_load_ids,
        maximum_resident_tile_count,
    )
    loaded_ids: list[str] = []
    visible_ids: list[str] = []
    placeholder_ids: list[str] = []
    for (
        tile,
        tile_root,
        terrain_collection,
        vegetation_collection,
        visible,
        should_load,
    ) in tile_states:
        tile_id = str(tile["id"])
        already_loaded = bool(tile_root.get("fireviewer_tile_loaded", False))
        if should_load and not already_loaded:
            datablock_snapshot = _snapshot_tile_datablocks(bpy)
            try:
                selection = select_tile_asset(manifest_file, tile)
                if selection.kind == "blender_library":
                    _load_linked_tile_library(
                        bpy,
                        tile,
                        tile_root,
                        selection.primary_path,
                        selection.library_collection_name,
                    )
                else:
                    if selection.orthophoto_source_path is None:
                        raise ValueError(f"Tile {tile_id!r} has no orthophoto source")
                    _build_tile_from_source_packages(
                        bpy,
                        tile,
                        tile_root,
                        terrain_collection,
                        vegetation_collection,
                        selection.primary_path,
                        selection.orthophoto_source_path,
                        global_origin,
                        global_vector_package=global_vector_package,
                        detail_vector_materials=detail_vector_materials,
                    )
                tagged_datablocks = _tag_new_tile_datablocks(
                    bpy, tile_id, datablock_snapshot
                )
            except Exception as build_error:
                try:
                    removed = _rollback_new_tile_datablocks(
                        bpy, tile_id, datablock_snapshot
                    )
                except Exception as rollback_error:
                    raise RuntimeError(
                        f"Tile {tile_id!r} materialization failed and rollback "
                        f"also failed: {rollback_error}"
                    ) from build_error
                for key in list(tile_root.keys()):
                    if key.startswith("detail_") or key.startswith("fireviewer_asset_"):
                        del tile_root[key]
                    elif key in {
                        "fireviewer_orthophoto_source_file_name",
                        "fireviewer_orthophoto_resolution_m",
                        "fireviewer_source_manifest_sha256",
                        "fireviewer_streaming_datablock_count",
                        "fireviewer_streaming_datablocks_json",
                    }:
                        del tile_root[key]
                tile_root["fireviewer_tile_loaded"] = False
                tile_root["fireviewer_requested_visible"] = False
                tile_root["fireviewer_last_rollback_datablocks_json"] = json.dumps(
                    removed, separators=(",", ":")
                )
                _set_collection_visibility(tile_root, False)
                raise
            tile_root["fireviewer_tile_loaded"] = True
            tile_root["fireviewer_streaming_datablock_count"] = len(tagged_datablocks)
            tile_root["fireviewer_streaming_datablocks_json"] = json.dumps(
                tagged_datablocks, separators=(",", ":")
            )
        if bool(tile_root.get("fireviewer_tile_loaded", False)):
            tile_root["fireviewer_source_manifest_sha256"] = manifest_sha256
            loaded_ids.append(tile_id)
        if not should_load and not already_loaded:
            placeholder_ids.append(tile_id)
        visible_and_loaded = visible and bool(
            tile_root.get("fireviewer_tile_loaded", False)
        )
        _set_collection_visibility(tile_root, visible_and_loaded)
        tile_root["fireviewer_requested_visible"] = visible
        if visible_and_loaded:
            visible_ids.append(tile_id)
    return {
        "manifest": manifest,
        "manifest_file": manifest_file,
        "ready_tile_count": len(all_ready),
        "loaded_tile_ids": loaded_ids,
        "visible_tile_ids": visible_ids,
        "placeholder_tile_ids": placeholder_ids,
        "resident_tile_ids": resident_ids,
        "maximum_resident_tile_count": maximum_resident_tile_count,
    }


def _remove_collection_contents_recursive(bpy: Any, collection: Any) -> None:
    for child in list(collection.children):
        _remove_collection_contents_recursive(bpy, child)
        bpy.data.collections.remove(child)
    for item in list(collection.objects):
        bpy.data.objects.remove(item, do_unlink=True)


def _remove_owned_tile_datablocks(bpy: Any, tile_id: str) -> list[str]:
    removed: list[str] = []
    for group in _TILE_DATABLOCK_GROUPS:
        datablocks = getattr(bpy.data, group, ())
        for item in list(datablocks):
            if str(item.get("fireviewer_streaming_owner_tile_id", "")) != tile_id:
                continue
            name = str(item.name)
            try:
                datablocks.remove(item, do_unlink=True)
            except TypeError:
                datablocks.remove(item)
            removed.append(f"{group}:{name}")
    remaining = [
        f"{group}:{item.name}"
        for group in _TILE_DATABLOCK_GROUPS
        for item in getattr(bpy.data, group, ())
        if str(item.get("fireviewer_streaming_owner_tile_id", "")) == tile_id
    ]
    if remaining:
        raise RuntimeError(
            f"Tile {tile_id!r} retained owned Blender datablocks: {remaining}"
        )
    return sorted(removed)


def _resident_tile_ids_from_scene(bpy: Any) -> list[str]:
    return sorted(
        str(collection.get("fireviewer_tile_id"))
        for collection in bpy.data.collections
        if collection.get("fireviewer_tile_id")
        and bool(collection.get("fireviewer_tile_loaded", False))
    )


def _record_resident_tile_ids(bpy: Any) -> list[str]:
    resident = _resident_tile_ids_from_scene(bpy)
    scene = bpy.context.scene
    scene["global_05m_loaded_tile_count"] = len(resident)
    scene["global_05m_loaded_tile_ids_json"] = json.dumps(
        resident, separators=(",", ":")
    )
    return resident


def activate_global_fallback(bpy: Any) -> dict[str, Any]:
    """Expose one complete, non-partial global scene during any transition."""

    scene = bpy.context.scene
    view_distance_m = float(
        scene.get(
            "fireviewer_runtime_view_distance_m",
            scene.get("scene_distance_lod_view_distance_m", 0.0),
        )
    )
    near_max_m = float(
        scene.get("scene_distance_lod_near_max_m", SCENE_DISTANCE_LOD_NEAR_MAX_M)
    )
    far_min_m = float(
        scene.get("scene_distance_lod_far_min_m", SCENE_DISTANCE_LOD_FAR_MIN_M)
    )
    strategy = scene_distance_lod_strategy(
        view_distance_m, near_max_m=near_max_m, far_min_m=far_min_m
    )
    terrain = bpy.data.collections.get("Terrain")
    if terrain is None:
        raise ValueError("The global Terrain collection is missing")
    _set_collection_visibility(terrain, True)
    _set_global_terrain_context_visibility(bpy, visible=False)
    detail_parent = bpy.data.collections.get("GlobalTiles")
    if detail_parent is not None:
        _set_collection_visibility(detail_parent, False)
    for collection in bpy.data.collections:
        if collection.get("fireviewer_tile_id"):
            _set_collection_visibility(collection, False)
    vegetation = bpy.data.collections.get("Vegetation")
    if vegetation is not None:
        _set_collection_visibility(vegetation, False)
    for name in SCENE_DISTANCE_LOD_SIMPLE_COLLECTIONS:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            _set_collection_visibility(
                collection, bool(strategy["simple_models_visible"])
            )
    vector_context = bpy.data.collections.get(GLOBAL_VECTOR_CONTEXT_COLLECTION_NAME)
    if vector_context is not None:
        _set_global_vector_context_visibility(
            bpy,
            visible=bool(strategy["simple_models_visible"]),
            replaced_tile_ids=(),
        )
        # The partitioned context is the only simple-vector representation in
        # a tiled scene; keep legacy monolithic collections disabled.
        for name in ("Buildings", "Roads", "Water"):
            collection = bpy.data.collections.get(name)
            if collection is not None:
                _set_collection_visibility(collection, False)
    scene["fireviewer_streaming_state"] = "global_fallback"
    scene["fireviewer_streaming_published_tile_ids_json"] = "[]"
    scene["scene_distance_lod_global_terrain_visible"] = True
    scene["scene_distance_lod_global_context_visible"] = False
    scene["scene_distance_lod_detail_terrain_visible"] = False
    scene["scene_distance_lod_terrain_overlap_active"] = False
    scene["scene_distance_lod_global_vector_context_visible"] = bool(
        vector_context is not None and strategy["simple_models_visible"]
    )
    return strategy


def evict_global_tiles(bpy: Any, tile_ids: Sequence[str]) -> list[str]:
    """Return resident HD tiles to lightweight indexed placeholders."""

    activate_global_fallback(bpy)
    evicted: list[str] = []
    for tile_id in sorted({str(identifier) for identifier in tile_ids}):
        tile_root = bpy.data.collections.get(_tile_collection_name(tile_id))
        if tile_root is None:
            raise ValueError(f"Unknown global detail tile {tile_id!r}")
        if not bool(tile_root.get("fireviewer_tile_loaded", False)):
            continue
        _remove_collection_contents_recursive(bpy, tile_root)
        _remove_owned_tile_datablocks(bpy, tile_id)
        _new_child_collection(bpy, tile_root, f"Terrain_{tile_id}")
        _new_child_collection(bpy, tile_root, f"Vegetation_{tile_id}")
        for key in list(tile_root.keys()):
            if key.startswith("detail_") or key.startswith("fireviewer_asset_"):
                del tile_root[key]
            elif key in {
                "fireviewer_orthophoto_source_file_name",
                "fireviewer_orthophoto_resolution_m",
                "fireviewer_source_manifest_sha256",
                "fireviewer_streaming_datablock_count",
                "fireviewer_streaming_datablocks_json",
            }:
                del tile_root[key]
        tile_root["fireviewer_tile_loaded"] = False
        tile_root["fireviewer_requested_visible"] = False
        _set_collection_visibility(tile_root, False)
        evicted.append(tile_id)
    _record_resident_tile_ids(bpy)
    scene = bpy.context.scene
    scene["fireviewer_streaming_last_evicted_tile_ids_json"] = json.dumps(
        evicted, separators=(",", ":")
    )
    return evicted


_RUNTIME_GLOBAL_PACKAGE_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def _runtime_resource_path(bpy: Any, scene_key: str) -> Path:
    scene = bpy.context.scene
    configured = scene.get(scene_key)
    if not isinstance(configured, str) or not configured:
        raise ValueError(f"Scene property {scene_key!r} is missing")
    candidate = Path(configured)
    if not candidate.is_absolute():
        blend_path = Path(str(getattr(bpy.data, "filepath", "")))
        if not blend_path.is_file():
            raise ValueError(
                f"Cannot resolve relative runtime resource {configured!r} "
                "before the Blender file is saved"
            )
        candidate = blend_path.parent / candidate
    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _runtime_global_package(path: Path) -> dict[str, Any]:
    key = (str(path), path.stat().st_mtime_ns)
    cached = _RUNTIME_GLOBAL_PACKAGE_CACHE.get(key)
    if cached is not None:
        return cached
    _RUNTIME_GLOBAL_PACKAGE_CACHE.clear()
    loaded = load_package(path)
    _RUNTIME_GLOBAL_PACKAGE_CACHE[key] = loaded
    return loaded


def materialize_global_tile(bpy: Any, tile_id: str) -> dict[str, Any]:
    """Materialize one indexed HD tile while the global fallback stays visible."""

    identifier = str(tile_id)
    manifest_path = _runtime_resource_path(bpy, "fireviewer_runtime_tile_manifest_path")
    package_path = _runtime_resource_path(bpy, "fireviewer_runtime_global_package_path")
    scene = bpy.context.scene
    origin = (
        float(scene["origin_l93_x_m"]),
        float(scene["origin_l93_y_m"]),
        float(scene["origin_l93_z_m"]),
    )
    parent = bpy.data.collections.get("GlobalTiles")
    if parent is None:
        raise ValueError("The GlobalTiles collection is missing")
    material_names = {
        "buildings": "MAT_Buildings",
        "road_carriageway": "MAT_Roads",
        "road_shoulders": "MAT_RoadShoulders",
        "road_markings": "MAT_RoadMarkings",
        "water_segments": "MAT_WaterSegments",
        "water_surfaces": "MAT_WaterSurfaces",
    }
    materials = {
        key: bpy.data.materials.get(name) for key, name in material_names.items()
    }
    missing = sorted(key for key, value in materials.items() if value is None)
    if missing:
        raise ValueError(f"Runtime detail materials are missing: {missing}")
    result = load_global_tiles_into_scene(
        bpy,
        manifest_path,
        parent,
        origin,
        selected_tile_ids=[identifier],
        load_mode="visible",
        maximum_resident_tile_count=int(
            scene.get(
                "global_05m_maximum_resident_tile_count",
                SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES,
            )
        ),
        global_vector_package=_runtime_global_package(package_path),
        detail_vector_materials=materials,
    )
    tile_root = bpy.data.collections.get(_tile_collection_name(identifier))
    if tile_root is None or not bool(tile_root.get("fireviewer_tile_loaded", False)):
        raise RuntimeError(f"Tile {identifier!r} was not materialized")
    _set_collection_visibility(tile_root, False)
    _set_collection_visibility(parent, False)
    _record_resident_tile_ids(bpy)
    scene["fireviewer_streaming_last_loaded_tile_id"] = identifier
    return result


def publish_detail_tiles(bpy: Any, tile_ids: Sequence[str]) -> list[str]:
    """Atomically replace exact global chunks with one complete HD tile set."""

    published = sorted({str(identifier) for identifier in tile_ids})
    if not published:
        raise ValueError("A detailed publication requires at least one tile")
    if len(published) > SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES:
        raise ValueError("Detailed publication exceeds the 16-tile hard ceiling")
    roots: list[Any] = []
    for tile_id in published:
        root = bpy.data.collections.get(_tile_collection_name(tile_id))
        if root is None or not bool(root.get("fireviewer_tile_loaded", False)):
            raise ValueError(f"Detailed tile {tile_id!r} is not resident")
        if not bool(root.get("detail_vector_lod_complete", False)):
            raise ValueError(f"Detailed tile {tile_id!r} has incomplete vectors")
        roots.append(root)
    resident = _resident_tile_ids_from_scene(bpy)
    if resident != published:
        raise ValueError(
            "Detailed publication requires the resident set to equal the target set"
        )

    terrain = bpy.data.collections.get("Terrain")
    detail_parent = bpy.data.collections.get("GlobalTiles")
    if terrain is None or detail_parent is None:
        raise ValueError("Terrain or GlobalTiles collection is missing")
    published_set = set(published)
    terrain_context = bpy.data.collections.get(GLOBAL_TERRAIN_CONTEXT_COLLECTION_NAME)
    vector_context = bpy.data.collections.get(GLOBAL_VECTOR_CONTEXT_COLLECTION_NAME)
    if terrain_context is None or vector_context is None:
        raise ValueError("Global terrain or vector context is missing")
    terrain_context_ids = {
        str(item.get("fireviewer_tile_id"))
        for item in terrain_context.objects
        if item.get("fireviewer_tile_id")
    }
    vector_context_ids = {
        str(item.get("fireviewer_tile_id"))
        for item in vector_context.children
        if item.get("fireviewer_tile_id")
    }
    if not published_set.issubset(terrain_context_ids):
        raise ValueError("Published tiles are missing from global terrain context")
    if not published_set.issubset(vector_context_ids):
        raise ValueError("Published tiles are missing from global vector context")

    try:
        # Make the complete replacement visible first.  The monolithic global
        # terrain remains visible until the very last mutation, so an error can
        # never expose an empty/white scene.
        _set_global_terrain_context_visibility(
            bpy, visible=True, replaced_tile_ids=published
        )
        visible_simple_vector_ids = _set_global_vector_context_visibility(
            bpy, visible=True, replaced_tile_ids=published
        )
        _set_collection_visibility(detail_parent, True)
        for collection in bpy.data.collections:
            tile_id = str(collection.get("fireviewer_tile_id", ""))
            if tile_id and collection.name.startswith("GlobalTile_"):
                _set_collection_visibility(collection, tile_id in published_set)
        for name in ("Buildings", "Roads", "Water"):
            collection = bpy.data.collections.get(name)
            if collection is not None:
                _set_collection_visibility(collection, False)
        vegetation = bpy.data.collections.get("Vegetation")
        if vegetation is not None:
            _set_collection_visibility(vegetation, False)
        _set_collection_visibility(terrain, False)
    except Exception:
        activate_global_fallback(bpy)
        raise
    scene = bpy.context.scene
    scene["fireviewer_streaming_state"] = "detail_published"
    scene["fireviewer_streaming_published_tile_ids_json"] = json.dumps(
        published, separators=(",", ":")
    )
    scene["global_05m_visible_tile_count"] = len(published)
    scene["global_05m_visible_tile_ids_json"] = json.dumps(
        published, separators=(",", ":")
    )
    scene["scene_distance_lod_global_terrain_visible"] = False
    scene["scene_distance_lod_global_context_visible"] = True
    scene["scene_distance_lod_detail_terrain_visible"] = True
    scene["scene_distance_lod_terrain_mode"] = "detail_0m50_tiles_with_global_context"
    scene["scene_distance_lod_terrain_overlap_active"] = False
    scene["global_context_continuous_around_detail_tiles"] = True
    scene["scene_distance_lod_global_vector_context_visible"] = True
    scene["scene_distance_lod_visible_global_vector_context_tile_ids_json"] = (
        json.dumps(visible_simple_vector_ids, separators=(",", ":"))
    )
    scene["scene_distance_lod_global_vectors_replaced_tile_ids_json"] = json.dumps(
        published, separators=(",", ":")
    )
    return published


def apply_tiled_collection_visibility(
    bpy: Any,
    focus_l93_m: Sequence[float],
    radius_m: float,
    *,
    hide_global_base: bool = False,
) -> list[str]:
    """Refresh pre-LOD tile culling while retaining the global terrain.

    This helper never enables global vegetation. The final exclusive terrain
    decision is made by :func:`apply_scene_distance_lod`, which can prove that
    the complete view footprint is resident before hiding the global soil.
    """

    from tiled_scene import tile_distance_to_point_m

    radius = float(radius_m)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_m must be finite and non-negative")
    if hide_global_base:
        raise ValueError(
            "hide_global_base=True is unavailable during pre-LOD culling; "
            "apply_scene_distance_lod must first prove complete detail coverage"
        )
    visible: list[str] = []
    for collection in bpy.data.collections:
        tile_id = collection.get("fireviewer_tile_id")
        if not tile_id:
            continue
        bounds = json.loads(collection["fireviewer_core_bounds_l93_json"])
        requested = tile_distance_to_point_m(bounds, focus_l93_m) <= radius
        loaded = bool(collection.get("fireviewer_tile_loaded", False))
        _set_collection_visibility(collection, requested and loaded)
        collection["fireviewer_requested_visible"] = requested
        if requested and loaded:
            visible.append(str(tile_id))
    strategy = tiled_compositing_strategy(visible)
    for name, strategy_key in (
        ("Terrain", "global_terrain_visible"),
        ("Vegetation", "global_vegetation_visible"),
    ):
        collection = bpy.data.collections.get(name)
        if collection is not None:
            _set_collection_visibility(collection, bool(strategy[strategy_key]))
    _set_global_terrain_context_visibility(bpy, visible=False)
    scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is not None:
        scene["global_05m_visible_tile_count"] = len(visible)
        scene["global_05m_visible_tile_ids_json"] = json.dumps(
            sorted(visible), separators=(",", ":")
        )
        scene["global_base_hidden_for_detail_tiles"] = False
        scene["global_base_visibility_strategy"] = strategy[
            "global_base_visibility_strategy"
        ]
        scene["global_base_continuous_under_detail_tiles"] = False
        scene["detail_terrain_source_z_offset_m"] = strategy[
            "detail_terrain_source_z_offset_m"
        ]
        scene["detail_terrain_render_z_offset_m"] = strategy[
            "detail_terrain_render_z_offset_m"
        ]
        scene["tiled_compositing_schema"] = strategy["schema"]
    return sorted(visible)


def apply_scene_distance_lod(
    bpy: Any,
    view_distance_m: float,
    *,
    focus_l93_m: Sequence[float] | None = None,
    detail_radius_m: float = SCENE_DISTANCE_LOD_DETAIL_RADIUS_M,
    near_max_m: float = SCENE_DISTANCE_LOD_NEAR_MAX_M,
    far_min_m: float = SCENE_DISTANCE_LOD_FAR_MIN_M,
    maximum_detail_tile_count: int = SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES,
) -> dict[str, Any]:
    """Apply a binary far/medium/near collection visibility contract.

    Global vegetation proxies are always hidden. Detailed 0.5 m tile roots are
    revealed only when the near band is active, the configured detail radius
    covers a conservative view footprint, every tile intersecting that radius
    is resident with a complete tile-local vector LOD, and the complete set
    stays within the hard 16-tile ceiling. In that proven state the global
    terrain and global vector models are hidden; otherwise every detail tile
    stays hidden and the global terrain/vector set remains visible. The
    all-or-nothing rule prevents sparse tree islands, overlapping soils and
    duplicate road, water or building geometry.
    This function never materializes missing assets; callers use
    :func:`load_global_tiles_into_scene` first.
    """

    from tiled_scene import tile_distance_to_point_m

    strategy = scene_distance_lod_strategy(
        view_distance_m,
        near_max_m=near_max_m,
        far_min_m=far_min_m,
    )
    radius = float(detail_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("detail_radius_m must be finite and strictly positive")
    if (
        isinstance(maximum_detail_tile_count, bool)
        or not isinstance(maximum_detail_tile_count, int)
        or maximum_detail_tile_count <= 0
    ):
        raise ValueError("maximum_detail_tile_count must be a positive integer")
    if maximum_detail_tile_count > SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES:
        raise ValueError(
            "maximum_detail_tile_count cannot exceed the hard ceiling of "
            f"{SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES}"
        )
    if focus_l93_m is not None:
        if len(focus_l93_m) < 2 or not all(
            math.isfinite(float(value)) for value in focus_l93_m[:2]
        ):
            raise ValueError("focus_l93_m must contain finite Lambert-93 x and y")
    if strategy["detail_tiles_visible"] and focus_l93_m is None:
        raise ValueError("Near LOD requires a Lambert-93 focus for detail tiles")

    global_vegetation = bpy.data.collections.get("Vegetation")
    if global_vegetation is not None:
        _set_collection_visibility(global_vegetation, False)
    for name in SCENE_DISTANCE_LOD_SIMPLE_COLLECTIONS:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            _set_collection_visibility(
                collection, bool(strategy["simple_models_visible"])
            )

    detail_requested_by_band = bool(strategy["detail_tiles_visible"])
    requested_detail_ids: list[str] = []
    loaded_requested_detail_ids: list[str] = []
    vector_complete_requested_detail_ids: list[str] = []
    tile_collections: list[tuple[Any, str]] = []
    detail_vector_contract_declared = False
    for collection in bpy.data.collections:
        tile_id = collection.get("fireviewer_tile_id")
        bounds_json = collection.get("fireviewer_core_bounds_l93_json")
        if not tile_id or bounds_json is None:
            continue
        tile_identifier = str(tile_id)
        tile_collections.append((collection, tile_identifier))
        if collection.get("detail_vector_lod_complete") is not None:
            detail_vector_contract_declared = True
        requested = False
        if detail_requested_by_band and focus_l93_m is not None:
            bounds = json.loads(bounds_json)
            requested = tile_distance_to_point_m(bounds, focus_l93_m) <= radius
        loaded = bool(collection.get("fireviewer_tile_loaded", False))
        collection["fireviewer_requested_visible"] = requested
        if requested:
            requested_detail_ids.append(tile_identifier)
            if loaded:
                loaded_requested_detail_ids.append(tile_identifier)
                if bool(collection.get("detail_vector_lod_complete", False)):
                    vector_complete_requested_detail_ids.append(tile_identifier)

    requested_detail_ids = sorted(set(requested_detail_ids))
    loaded_requested_detail_ids = sorted(set(loaded_requested_detail_ids))
    missing_detail_ids = sorted(
        set(requested_detail_ids) - set(loaded_requested_detail_ids)
    )
    required_coverage_radius_m = (
        strategy["view_distance_m"] * SCENE_DISTANCE_LOD_VIEW_FOOTPRINT_FACTOR
    )
    view_radius_covered = bool(
        detail_requested_by_band and required_coverage_radius_m <= radius
    )
    tile_budget_satisfied = len(requested_detail_ids) <= maximum_detail_tile_count
    resident_coverage_complete = bool(requested_detail_ids) and not missing_detail_ids
    detail_vector_coverage_complete = bool(
        requested_detail_ids
        and set(requested_detail_ids).issubset(
            set(vector_complete_requested_detail_ids)
        )
    )
    global_context = bpy.data.collections.get(GLOBAL_TERRAIN_CONTEXT_COLLECTION_NAME)
    global_context_tile_ids = {
        str(item.get("fireviewer_tile_id"))
        for item in (() if global_context is None else global_context.objects)
        if item.get("fireviewer_tile_id")
    }
    global_context_coverage_complete = bool(
        requested_detail_ids
        and set(requested_detail_ids).issubset(global_context_tile_ids)
    )
    global_vector_context = bpy.data.collections.get(
        GLOBAL_VECTOR_CONTEXT_COLLECTION_NAME
    )
    global_vector_context_tile_ids = {
        str(item.get("fireviewer_tile_id"))
        for item in (
            () if global_vector_context is None else global_vector_context.children
        )
        if item.get("fireviewer_tile_id")
    }
    global_vector_context_coverage_complete = bool(
        requested_detail_ids
        and set(requested_detail_ids).issubset(global_vector_context_tile_ids)
    )
    detail_coverage_complete = bool(
        detail_requested_by_band
        and view_radius_covered
        and tile_budget_satisfied
        and resident_coverage_complete
        and detail_vector_coverage_complete
        and global_context_coverage_complete
        and global_vector_context_coverage_complete
    )
    if not detail_requested_by_band:
        coverage_reason = "distance_band_without_detail"
    elif not view_radius_covered:
        coverage_reason = "view_exceeds_loaded_radius"
    elif not requested_detail_ids:
        coverage_reason = "no_intersecting_tiles"
    elif not tile_budget_satisfied:
        coverage_reason = "detail_tile_budget_exceeded"
    elif missing_detail_ids:
        coverage_reason = "resident_tile_coverage_incomplete"
    elif not detail_vector_coverage_complete:
        coverage_reason = "detail_vector_coverage_incomplete"
    elif not global_context_coverage_complete:
        coverage_reason = "global_terrain_context_incomplete"
    elif not global_vector_context_coverage_complete:
        coverage_reason = "global_vector_context_incomplete"
    else:
        coverage_reason = "complete"

    detail_parent = bpy.data.collections.get("GlobalTiles")
    if detail_parent is not None:
        _set_collection_visibility(detail_parent, detail_coverage_complete)

    terrain = bpy.data.collections.get("Terrain")
    global_terrain_visible = not detail_coverage_complete
    if terrain is not None:
        _set_collection_visibility(terrain, global_terrain_visible)
    global_context_visible = detail_coverage_complete
    visible_global_context_ids = _set_global_terrain_context_visibility(
        bpy,
        visible=global_context_visible,
        replaced_tile_ids=(requested_detail_ids if detail_coverage_complete else ()),
    )

    global_vector_models_visible = bool(strategy["simple_models_visible"])
    visible_global_vector_context_ids: list[str] = []
    global_vector_context_visible = bool(
        global_vector_context is not None and global_vector_models_visible
    )
    if global_vector_context is not None:
        visible_global_vector_context_ids = _set_global_vector_context_visibility(
            bpy,
            visible=global_vector_context_visible,
            replaced_tile_ids=(
                requested_detail_ids if detail_coverage_complete else ()
            ),
        )
    legacy_global_vector_models_visible = bool(
        global_vector_context is None
        and global_vector_models_visible
        and not detail_coverage_complete
    )
    for name in ("Buildings", "Roads", "Water"):
        collection = bpy.data.collections.get(name)
        if collection is not None:
            _set_collection_visibility(collection, legacy_global_vector_models_visible)

    visible_detail_ids: list[str] = []
    requested_id_set = set(requested_detail_ids)
    for collection, tile_identifier in tile_collections:
        visible = detail_coverage_complete and tile_identifier in requested_id_set
        _set_collection_visibility(collection, visible)
        if visible:
            visible_detail_ids.append(tile_identifier)
    visible_detail_ids = sorted(set(visible_detail_ids))

    result = dict(strategy)
    result["detail_tiles_requested_by_band"] = detail_requested_by_band
    result["detail_tiles_visible"] = detail_coverage_complete
    result["detail_radius_m"] = radius
    result["detail_view_footprint_factor"] = SCENE_DISTANCE_LOD_VIEW_FOOTPRINT_FACTOR
    result["detail_required_coverage_radius_m"] = required_coverage_radius_m
    result["detail_view_radius_covered"] = view_radius_covered
    result["detail_tile_budget_satisfied"] = tile_budget_satisfied
    result["detail_resident_coverage_complete"] = resident_coverage_complete
    result["detail_vector_contract_declared"] = detail_vector_contract_declared
    result["detail_vector_coverage_complete"] = detail_vector_coverage_complete
    result["global_context_coverage_complete"] = global_context_coverage_complete
    result["global_vector_context_coverage_complete"] = (
        global_vector_context_coverage_complete
    )
    result["detail_coverage_complete"] = detail_coverage_complete
    result["detail_coverage_reason"] = coverage_reason
    result["maximum_detail_tile_count"] = maximum_detail_tile_count
    result["requested_detail_tile_ids"] = requested_detail_ids
    result["loaded_requested_detail_tile_ids"] = loaded_requested_detail_ids
    result["missing_detail_tile_ids"] = missing_detail_ids
    result["visible_detail_tile_ids"] = visible_detail_ids
    result["visible_detail_tile_count"] = len(visible_detail_ids)
    result["vegetation_mode"] = (
        "detailed_0m50_tiles" if detail_coverage_complete else "none"
    )
    result["terrain_mode"] = (
        "detail_0m50_tiles_with_global_context"
        if detail_coverage_complete
        else "global_2m"
    )
    result["global_terrain_visible"] = global_terrain_visible
    result["global_context_visible"] = global_context_visible
    result["visible_global_context_tile_ids"] = visible_global_context_ids
    result["visible_global_context_tile_count"] = len(visible_global_context_ids)
    result["global_context_continuous_around_detail_tiles"] = detail_coverage_complete
    result["detail_terrain_visible"] = detail_coverage_complete
    result["terrain_overlap_active"] = False
    result["global_vector_models_visible"] = global_vector_models_visible
    result["legacy_global_vector_models_visible"] = legacy_global_vector_models_visible
    result["global_vector_context_visible"] = global_vector_context_visible
    result["visible_global_vector_context_tile_ids"] = visible_global_vector_context_ids
    result["visible_global_vector_context_tile_count"] = len(
        visible_global_vector_context_ids
    )
    result["global_vector_replaced_tile_ids"] = (
        requested_detail_ids if detail_coverage_complete else []
    )
    result["detail_vector_models_visible"] = detail_coverage_complete
    result["vector_model_overlap_active"] = False

    scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is not None:
        scene["scene_distance_lod_schema"] = strategy["schema"]
        scene["scene_distance_lod_band"] = strategy["band"]
        scene["scene_distance_lod_view_distance_m"] = strategy["view_distance_m"]
        scene["scene_distance_lod_near_max_m"] = strategy["near_max_m"]
        scene["scene_distance_lod_far_min_m"] = strategy["far_min_m"]
        scene["scene_distance_lod_detail_radius_m"] = radius
        scene["scene_distance_lod_detail_view_footprint_factor"] = (
            SCENE_DISTANCE_LOD_VIEW_FOOTPRINT_FACTOR
        )
        scene["scene_distance_lod_detail_required_coverage_radius_m"] = (
            required_coverage_radius_m
        )
        scene["scene_distance_lod_far_content"] = strategy["far_content"]
        scene["scene_distance_lod_medium_content"] = strategy["medium_content"]
        scene["scene_distance_lod_near_content"] = strategy["near_content"]
        scene["scene_distance_lod_simple_models_visible"] = bool(
            strategy["simple_models_visible"]
        )
        scene["scene_distance_lod_detail_tiles_requested_by_band"] = (
            detail_requested_by_band
        )
        scene["scene_distance_lod_detail_tiles_visible"] = detail_coverage_complete
        scene["scene_distance_lod_detail_view_radius_covered"] = view_radius_covered
        scene["scene_distance_lod_detail_tile_budget_satisfied"] = tile_budget_satisfied
        scene["scene_distance_lod_detail_resident_coverage_complete"] = (
            resident_coverage_complete
        )
        scene["scene_distance_lod_detail_coverage_complete"] = detail_coverage_complete
        scene["scene_distance_lod_detail_coverage_reason"] = coverage_reason
        scene["scene_distance_lod_detail_vector_contract_declared"] = (
            detail_vector_contract_declared
        )
        scene["scene_distance_lod_detail_vector_coverage_complete"] = (
            detail_vector_coverage_complete
        )
        scene["scene_distance_lod_global_context_coverage_complete"] = (
            global_context_coverage_complete
        )
        scene["scene_distance_lod_global_vector_context_coverage_complete"] = (
            global_vector_context_coverage_complete
        )
        scene["scene_distance_lod_global_vector_models_visible"] = (
            global_vector_models_visible
        )
        scene["scene_distance_lod_legacy_global_vector_models_visible"] = (
            legacy_global_vector_models_visible
        )
        scene["scene_distance_lod_global_vector_context_visible"] = (
            global_vector_context_visible
        )
        scene["scene_distance_lod_visible_global_vector_context_tile_ids_json"] = (
            json.dumps(visible_global_vector_context_ids, separators=(",", ":"))
        )
        scene["scene_distance_lod_global_vectors_replaced_tile_ids_json"] = json.dumps(
            requested_detail_ids if detail_coverage_complete else [],
            separators=(",", ":"),
        )
        scene["scene_distance_lod_detail_vector_models_visible"] = (
            detail_coverage_complete
        )
        scene["scene_distance_lod_vector_model_overlap_active"] = False
        scene["scene_distance_lod_maximum_detail_tile_count"] = (
            maximum_detail_tile_count
        )
        scene["scene_distance_lod_requested_detail_tile_count"] = len(
            requested_detail_ids
        )
        scene["scene_distance_lod_loaded_requested_detail_tile_count"] = len(
            loaded_requested_detail_ids
        )
        scene["scene_distance_lod_missing_detail_tile_ids_json"] = json.dumps(
            missing_detail_ids, separators=(",", ":")
        )
        scene["scene_distance_lod_global_vegetation_visible"] = False
        scene["scene_distance_lod_vegetation_mode"] = result["vegetation_mode"]
        scene["scene_distance_lod_terrain_mode"] = result["terrain_mode"]
        scene["scene_distance_lod_global_terrain_visible"] = global_terrain_visible
        scene["scene_distance_lod_global_context_visible"] = global_context_visible
        scene["scene_distance_lod_visible_global_context_tile_count"] = len(
            visible_global_context_ids
        )
        scene["scene_distance_lod_detail_terrain_visible"] = detail_coverage_complete
        scene["scene_distance_lod_terrain_overlap_active"] = False
        scene["global_base_hidden_for_detail_tiles"] = detail_coverage_complete
        scene["global_base_visibility_strategy"] = GLOBAL_BASE_VISIBILITY_STRATEGY
        scene["global_base_continuous_under_detail_tiles"] = False
        scene["global_context_continuous_around_detail_tiles"] = (
            detail_coverage_complete
        )
        scene["global_05m_visible_tile_count"] = len(visible_detail_ids)
        scene["global_05m_visible_tile_ids_json"] = json.dumps(
            visible_detail_ids, separators=(",", ":")
        )
        if focus_l93_m is not None:
            scene["scene_distance_lod_focus_l93_json"] = json.dumps(
                [float(focus_l93_m[0]), float(focus_l93_m[1])],
                separators=(",", ":"),
            )
    return result


def build_from_package(
    package_path: str | Path,
    output_path: str | Path | None = None,
    *,
    orthophoto_source_path: str | Path | None = None,
    mid_vegetation_path: str | Path | None = None,
    mid_orthophoto_source_path: str | Path | None = None,
    tile_index_path: str | Path | None = None,
    tile_ids: Sequence[str] | None = None,
    tile_load_mode: str = "visible",
    tile_focus_l93_m: Sequence[float] | None = None,
    tile_visible_radius_m: float | None = None,
    scene_lod_view_distance_m: float | None = None,
    scene_lod_near_max_m: float = SCENE_DISTANCE_LOD_NEAR_MAX_M,
    scene_lod_far_min_m: float = SCENE_DISTANCE_LOD_FAR_MIN_M,
    scene_lod_detail_radius_m: float = SCENE_DISTANCE_LOD_DETAIL_RADIUS_M,
    maximum_resident_tile_count: int | None = (SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES),
) -> Any:
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "This function must run inside Blender because bpy is unavailable"
        ) from exc

    _ensure_sibling_modules_available()
    package_file = Path(package_path).expanduser().resolve()
    package = load_package(package_file)
    tile_manifest: dict[str, Any] | None = None
    global_orthophoto = (
        load_orthophoto_source(Path(orthophoto_source_path))
        if orthophoto_source_path is not None
        else None
    )
    mid_package_file = (
        Path(mid_vegetation_path).expanduser().resolve()
        if mid_vegetation_path is not None
        else None
    )
    mid_package = (
        load_mid_vegetation_package(mid_package_file)
        if mid_package_file is not None
        else None
    )
    if mid_orthophoto_source_path is not None and mid_package is None:
        raise ValueError("A mid orthophoto requires a mid-distance vegetation package")
    mid_orthophoto = (
        load_orthophoto_source(Path(mid_orthophoto_source_path))
        if mid_orthophoto_source_path is not None
        else None
    )
    if (tile_focus_l93_m is None) != (tile_visible_radius_m is None):
        raise ValueError("A tile focus and visible radius must be provided together")
    if tile_focus_l93_m is not None and len(tile_focus_l93_m) != 2:
        raise ValueError("Tile focus must contain Lambert-93 x and y")
    package_file_name, package_sha256 = _package_identity(package_file)
    collections = _reset_scene(bpy)
    scene = bpy.context.scene
    _configure_color_management(scene)
    metadata = package["metadata"]
    origin = metadata["origin_l93_m"]
    if mid_package is not None:
        mid_origin = mid_package["metadata"]["origin_l93_m"]
        if any(
            abs(float(left) - float(right)) > 0.001
            for left, right in zip(origin, mid_origin, strict=True)
        ):
            raise ValueError(
                "Global and mid-distance packages must share the same Lambert-93 origin"
            )
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    scene["fireviewer_schema"] = metadata["schema"]
    scene["preview_package_schema"] = package["schema"]
    if "preview_package_path" in scene:
        del scene["preview_package_path"]
    scene["preview_package_file_name"] = package_file_name
    scene["preview_package_sha256"] = package_sha256
    scene["source_crs"] = metadata["target_crs"]
    scene["origin_l93_x_m"] = float(origin[0])
    scene["origin_l93_y_m"] = float(origin[1])
    scene["origin_l93_z_m"] = float(origin[2])
    scene["axis_convention"] = metadata["axis_convention"]
    scene["linear_unit"] = "metre"
    scene["source_manifest_json"] = json.dumps(metadata, sort_keys=True)
    if global_orthophoto is not None:
        scene["terrain_texture_source_json"] = json.dumps(
            global_orthophoto[2], sort_keys=True
        )
    if mid_package_file is not None:
        scene["mid_vegetation_package_file_name"] = mid_package_file.name
        scene["mid_vegetation_package_sha256"] = _sha256(mid_package_file)
    if mid_orthophoto is not None:
        scene["mid_terrain_texture_source_json"] = json.dumps(
            mid_orthophoto[2], sort_keys=True
        )
    if tile_index_path is not None:
        from tiled_scene import load_global_05m_manifest

        tile_index_file = Path(tile_index_path).expanduser().resolve()
        tile_manifest = load_global_05m_manifest(tile_index_file)
        if not _same_origin(origin, tile_manifest["origin_l93_m"]):
            raise ValueError("Global package and tile manifest must share one origin")
        scene["global_05m_manifest_file_name"] = tile_index_file.name
        scene["global_05m_manifest_sha256"] = _sha256(tile_index_file)
        scene["global_05m_tile_load_mode"] = tile_load_mode

    terrain_material = (
        None
        if global_orthophoto is not None
        else _noise_surface_material(
            bpy,
            "MAT_TerrainFallback",
            (0.22, 0.34, 0.16, 1.0),
            (0.12, 0.18, 0.09, 1.0),
            (0.34, 0.39, 0.19, 1.0),
            0.9,
            0.008,
            0.12,
            0.12,
            0.7,
        )
    )
    fire_material = _material(bpy, "MAT_FirePerimeter", (0.85, 0.025, 0.01, 1.0), 0.35)
    buffer_material = _material(
        bpy, "MAT_AnalysisBuffer", (0.95, 0.35, 0.025, 1.0), 0.5
    )
    building_material = _noise_surface_material(
        bpy,
        "MAT_Buildings",
        (0.44, 0.40, 0.36, 1.0),
        (0.25, 0.23, 0.21, 1.0),
        (0.58, 0.53, 0.46, 1.0),
        0.82,
        0.035,
        0.42,
        0.09,
        0.12,
    )
    vegetation_material = _vegetation_material(bpy)
    road_material = _noise_surface_material(
        bpy,
        "MAT_Roads",
        (0.18, 0.17, 0.16, 1.0),
        (0.055, 0.06, 0.06, 1.0),
        (0.20, 0.19, 0.17, 1.0),
        0.92,
        0.6,
        2.2,
        0.22,
        0.035,
    )
    road_shoulder_material = _noise_surface_material(
        bpy,
        "MAT_RoadShoulders",
        (0.28, 0.23, 0.16, 1.0),
        (0.13, 0.11, 0.075, 1.0),
        (0.42, 0.34, 0.22, 1.0),
        1.0,
        0.16,
        1.3,
        0.25,
        0.04,
    )
    road_marking_material = _material(
        bpy, "MAT_RoadMarkings", (0.82, 0.78, 0.63, 1.0), 0.58
    )
    water_course_material = _material(
        bpy, "MAT_WaterCourses", (0.02, 0.22, 0.52, 1.0), 0.3
    )
    water_segment_material = _material(
        bpy, "MAT_WaterSegments", (0.03, 0.32, 0.66, 1.0), 0.28
    )
    water_surface_material = _material(
        bpy, "MAT_WaterSurfaces", (0.02, 0.38, 0.72, 1.0), 0.22
    )

    terrain = _mesh_object(
        bpy,
        "TerrainPreview",
        package["terrain"]["vertices"],
        _triangulated_terrain_faces(package["terrain"]["faces"]),
        terrain_material,
        smooth=True,
    )
    terrain["preview_step_pixels"] = int(package["terrain"]["step_pixels"])
    terrain["render_surface_triangulation"] = "fixed_nw_se_diagonal"
    collections["Terrain"].objects.link(terrain)
    if global_orthophoto is not None:
        from terrain_texture import apply_georeferenced_orthophoto

        global_texture = apply_georeferenced_orthophoto(
            bpy,
            terrain,
            global_orthophoto[0],
            origin,
            global_orthophoto[1],
            config=_terrain_orthophoto_material_config(
                "MAT_TerrainOrthophotoIGN2m",
                boundary_tolerance_m=15.0,
                pack_image_in_blend=True,
            ),
        )
        terrain["texture_role"] = "ign_bd_ortho_global"
        terrain["texture_uv_min_u"] = global_texture.statistics.minimum_u
        terrain["texture_uv_min_v"] = global_texture.statistics.minimum_v
        terrain["texture_uv_max_u"] = global_texture.statistics.maximum_u
        terrain["texture_uv_max_v"] = global_texture.statistics.maximum_v

    if tile_manifest is not None:
        if not terrain.data.materials:
            raise ValueError("Global terrain context requires a terrain material")
        global_context = build_global_terrain_context(
            bpy,
            package["terrain"],
            tile_manifest,
            origin,
            terrain.data.materials[0],
            texture_bounds_l93_m=(
                None if global_orthophoto is None else global_orthophoto[1]
            ),
        )
        partition = global_context["partition"]
        scene["global_terrain_context_schema"] = GLOBAL_TERRAIN_CONTEXT_SCHEMA
        scene["global_terrain_context_object_count"] = global_context["object_count"]
        scene["global_terrain_context_populated_tile_count"] = len(
            partition.populated_tile_ids
        )
        scene["global_terrain_context_output_triangle_count"] = (
            partition.validation.output_triangle_count
        )
        scene["global_terrain_context_area_m2"] = (
            partition.validation.total_partitioned_area_m2
        )
        scene["global_terrain_context_area_error_m2"] = abs(
            partition.validation.total_partitioned_area_m2
            - partition.validation.total_source_area_m2
        )
        vector_context = build_global_vector_context(
            bpy,
            package,
            tile_manifest,
            origin,
            {
                "buildings": building_material,
                "road_carriageway": road_material,
                "road_shoulders": road_shoulder_material,
                "road_markings": road_marking_material,
                "water_segments": water_segment_material,
                "water_surfaces": water_surface_material,
            },
        )
        scene["global_vector_context_schema"] = GLOBAL_VECTOR_CONTEXT_SCHEMA
        scene["global_vector_context_object_count"] = vector_context["object_count"]
        scene["global_vector_context_populated_chunk_count"] = len(
            vector_context["populated_chunk_ids"]
        )
        scene["global_vector_context_manifest_tile_count"] = len(
            tile_manifest.get("tiles", [])
        )

    if mid_package is not None:
        local_terrain_spec = mid_package["terrain"]
        local_terrain = _mesh_object(
            bpy,
            "TerrainMidMontmaur",
            local_terrain_spec["vertices"],
            _triangulated_terrain_faces(local_terrain_spec["faces"]),
            terrain_material,
            smooth=True,
        )
        local_terrain["detail_role"] = "mnt_hd_local_overlay"
        local_terrain["render_surface_triangulation"] = "fixed_nw_se_diagonal"
        local_terrain["source_pixel_size_m"] = json.dumps(
            local_terrain_spec["source_pixel_size_m"]
        )
        local_terrain["sample_spacing_m"] = json.dumps(
            local_terrain_spec["sample_spacing_m"]
        )
        collections["Terrain"].objects.link(local_terrain)
        active_mid_orthophoto = mid_orthophoto or global_orthophoto
        if active_mid_orthophoto is not None:
            from terrain_texture import apply_georeferenced_orthophoto

            local_texture = apply_georeferenced_orthophoto(
                bpy,
                local_terrain,
                active_mid_orthophoto[0],
                origin,
                active_mid_orthophoto[1],
                config=_terrain_orthophoto_material_config(
                    "MAT_TerrainMontmaurOrthophotoIGN0m50",
                    boundary_tolerance_m=0.01,
                    pack_image_in_blend=True,
                ),
            )
            local_terrain["texture_role"] = "ign_bd_ortho_mid_distance"
            local_terrain["texture_uv_min_u"] = local_texture.statistics.minimum_u
            local_terrain["texture_uv_min_v"] = local_texture.statistics.minimum_v
            local_terrain["texture_uv_max_u"] = local_texture.statistics.maximum_u
            local_terrain["texture_uv_max_v"] = local_texture.statistics.maximum_v

    fire = _boundary_object(
        bpy,
        "FirePerimeter",
        package["fire_perimeter"]["rings"],
        fire_material,
        radius_m=4.0,
    )
    analysis_buffer = _boundary_object(
        bpy,
        "AnalysisBuffer",
        package["analysis_buffer"]["rings"],
        buffer_material,
        radius_m=2.0,
    )
    collections["FirePerimeter"].objects.link(fire)
    collections["FirePerimeter"].objects.link(analysis_buffer)

    if tile_manifest is None:
        buildings = _prism_object(
            bpy, "Buildings", package["buildings"]["prisms"], building_material
        )
        collections["Buildings"].objects.link(buildings)
    use_legacy_vegetation = mid_package is None
    scene["legacy_vegetation_geometry_enabled"] = use_legacy_vegetation
    vegetation_mid_lod = (
        package["vegetation"].get("mid_distance_lod") if use_legacy_vegetation else None
    )
    if (
        use_legacy_vegetation
        and vegetation_mid_lod is None
        and "mesh" in package["vegetation"]
    ):
        vegetation_mesh = package["vegetation"]["mesh"]
        vegetation = _mesh_object(
            bpy,
            "VegetationCanopy",
            vegetation_mesh["vertices"],
            vegetation_mesh["faces"],
            vegetation_material,
            smooth=True,
        )
        vegetation["grid_step_pixels"] = int(vegetation_mesh["grid_step_pixels"])
        for key, value in package["vegetation"].get("statistics", {}).items():
            if isinstance(value, (bool, int, float, str)) and value is not None:
                vegetation[key] = value
        collections["Vegetation"].objects.link(vegetation)
    elif use_legacy_vegetation and vegetation_mid_lod is None:
        vegetation = _prism_object(
            bpy,
            "VegetationBlocks",
            package["vegetation"]["prisms"],
            vegetation_material,
        )
        collections["Vegetation"].objects.link(vegetation)

    if vegetation_mid_lod is not None:
        mid_tree_mesh = vegetation_mid_lod["mesh"]
        mid_trees = _mesh_object(
            bpy,
            "VegetationMidLOD",
            mid_tree_mesh["vertices"],
            mid_tree_mesh["faces"],
            vegetation_material,
            smooth=True,
        )
        mid_trees["lod_schema"] = vegetation_mid_lod["schema"]
        proxy_count = (
            vegetation_mid_lod.get("statistics", {}).get("mesh", {}).get("proxy_count")
        )
        if proxy_count is not None:
            mid_trees["proxy_count"] = int(proxy_count)
        collections["Vegetation"].objects.link(mid_trees)

    if mid_package is not None:
        from tree_instances import build_blender_tree_system

        tree_system = build_blender_tree_system(
            bpy,
            mid_package["tree_instances"],
            collections["Vegetation"],
            name="VegetationMontmaur0m50",
        )
        tree_system["detection_grid_m"] = 0.5
        tree_system["post_detection_spacing_rejected_count"] = int(
            mid_package["statistics"]["post_detection_spacing_rejected_count"]
        )
        tree_system["completeness_claim"] = mid_package["statistics"][
            "completeness_claim"
        ]

    if tile_manifest is None and "roads" in package:
        if "meshes" in package["roads"]:
            road_layers = (
                ("carriageway", "RoadCarriageway", road_material),
                ("left_shoulders", "RoadShoulderLeft", road_shoulder_material),
                ("right_shoulders", "RoadShoulderRight", road_shoulder_material),
                ("center_markings", "RoadCenterMarkings", road_marking_material),
            )
            for key, name, material in road_layers:
                road_mesh = package["roads"]["meshes"][key]
                road_object = _mesh_object(
                    bpy,
                    name,
                    road_mesh["vertices"],
                    road_mesh["faces"],
                    material,
                    smooth=False,
                )
                road_object["road_layer"] = key
                collections["Roads"].objects.link(road_object)
        else:
            road_mesh = package["roads"]["mesh"]
            roads = _mesh_object(
                bpy,
                "Roads",
                road_mesh["vertices"],
                road_mesh["faces"],
                road_material,
                smooth=False,
            )
            collections["Roads"].objects.link(roads)

    if tile_manifest is None and "water" in package:
        segment_geometry_present = bool(
            package["water"].get("segments", {}).get("mesh", {}).get("faces")
        )
        for key, name, material in (
            ("courses", "WaterCourses", water_course_material),
            ("segments", "WaterSegments", water_segment_material),
            ("surfaces", "WaterSurfaces", water_surface_material),
        ):
            if key == "courses" and segment_geometry_present:
                # Named courses are retained in package metadata for labels,
                # but BD TOPO segments already carry their full geometry.
                continue
            water_mesh = package["water"][key]["mesh"]
            water_object = _mesh_object(
                bpy,
                name,
                water_mesh["vertices"],
                water_mesh["faces"],
                material,
                smooth=key != "surfaces",
            )
            collections["Water"].objects.link(water_object)
        if segment_geometry_present:
            scene["water_linear_render_source"] = "segments_only"
            scene["water_courses_geometry_suppressed"] = True

    if tile_index_path is not None:
        tiled = load_global_tiles_into_scene(
            bpy,
            tile_index_path,
            collections["GlobalTiles"],
            origin,
            selected_tile_ids=tile_ids,
            load_mode=tile_load_mode,
            focus_l93_m=tile_focus_l93_m,
            visible_radius_m=tile_visible_radius_m,
            maximum_resident_tile_count=maximum_resident_tile_count,
            global_vector_package=package,
            detail_vector_materials={
                "buildings": building_material,
                "road_carriageway": road_material,
                "road_shoulders": road_shoulder_material,
                "road_markings": road_marking_material,
                "water_segments": water_segment_material,
                "water_surfaces": water_surface_material,
            },
        )
        scene["global_05m_manifest_schema"] = tiled["manifest"]["schema"]
        scene["global_05m_planned_tile_count"] = len(tiled["manifest"]["tiles"])
        scene["global_05m_ready_tile_count"] = tiled["ready_tile_count"]
        scene["global_05m_loaded_tile_count"] = len(tiled["loaded_tile_ids"])
        scene["global_05m_maximum_resident_tile_count"] = (
            -1
            if maximum_resident_tile_count is None
            else int(maximum_resident_tile_count)
        )
        scene["global_05m_visible_tile_count"] = len(tiled["visible_tile_ids"])
        scene["global_05m_loaded_tile_ids_json"] = json.dumps(
            tiled["loaded_tile_ids"], separators=(",", ":")
        )
        scene["global_05m_visible_tile_ids_json"] = json.dumps(
            tiled["visible_tile_ids"], separators=(",", ":")
        )
        if tile_focus_l93_m is not None:
            scene["global_05m_focus_l93_json"] = json.dumps(
                [float(value) for value in tile_focus_l93_m],
                separators=(",", ":"),
            )
            scene["global_05m_visible_radius_m"] = float(tile_visible_radius_m)
        strategy = tiled_compositing_strategy(tiled["visible_tile_ids"])
        _set_collection_visibility(
            collections["Terrain"], bool(strategy["global_terrain_visible"])
        )
        _set_collection_visibility(
            collections["Vegetation"], bool(strategy["global_vegetation_visible"])
        )
        _set_global_terrain_context_visibility(bpy, visible=False)
        scene["global_base_hidden_for_detail_tiles"] = False
        scene["global_base_visibility_strategy"] = strategy[
            "global_base_visibility_strategy"
        ]
        scene["global_base_continuous_under_detail_tiles"] = False
        scene["detail_terrain_source_z_offset_m"] = strategy[
            "detail_terrain_source_z_offset_m"
        ]
        scene["detail_terrain_render_z_offset_m"] = strategy[
            "detail_terrain_render_z_offset_m"
        ]
        scene["tiled_compositing_schema"] = strategy["schema"]

    if scene_lod_view_distance_m is not None:
        apply_scene_distance_lod(
            bpy,
            scene_lod_view_distance_m,
            focus_l93_m=tile_focus_l93_m,
            detail_radius_m=scene_lod_detail_radius_m,
            near_max_m=scene_lod_near_max_m,
            far_min_m=scene_lod_far_min_m,
        )

    _configure_lighting_rig(bpy, scene)

    if output_path:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(destination), check_existing=False)
    return scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--orthophoto-source", type=Path)
    parser.add_argument("--mid-vegetation", type=Path)
    parser.add_argument("--mid-orthophoto-source", type=Path)
    parser.add_argument("--tile-index", type=Path)
    parser.add_argument("--tile-id", action="append", default=None)
    parser.add_argument(
        "--tile-load-mode", choices=("visible", "all_ready"), default="visible"
    )
    parser.add_argument("--tile-focus-l93", type=float, nargs=2)
    parser.add_argument("--tile-visible-radius-m", type=float)
    parser.add_argument("--scene-lod-view-distance-m", type=float)
    parser.add_argument(
        "--scene-lod-near-max-m",
        type=float,
        default=SCENE_DISTANCE_LOD_NEAR_MAX_M,
    )
    parser.add_argument(
        "--scene-lod-far-min-m",
        type=float,
        default=SCENE_DISTANCE_LOD_FAR_MIN_M,
    )
    parser.add_argument(
        "--scene-lod-detail-radius-m",
        type=float,
        default=SCENE_DISTANCE_LOD_DETAIL_RADIUS_M,
    )
    parser.add_argument(
        "--maximum-resident-tile-count",
        type=int,
        default=SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        values = list(sys.argv if argv is None else argv)
        if "--" in values:
            values = values[values.index("--") + 1 :]
        else:
            values = values[1:]
        args = _parser().parse_args(values)
        build_from_package(
            args.package,
            args.output,
            orthophoto_source_path=args.orthophoto_source,
            mid_vegetation_path=args.mid_vegetation,
            mid_orthophoto_source_path=args.mid_orthophoto_source,
            tile_index_path=args.tile_index,
            tile_ids=args.tile_id,
            tile_load_mode=args.tile_load_mode,
            tile_focus_l93_m=args.tile_focus_l93,
            tile_visible_radius_m=args.tile_visible_radius_m,
            scene_lod_view_distance_m=args.scene_lod_view_distance_m,
            scene_lod_near_max_m=args.scene_lod_near_max_m,
            scene_lod_far_min_m=args.scene_lod_far_min_m,
            scene_lod_detail_radius_m=args.scene_lod_detail_radius_m,
            maximum_resident_tile_count=args.maximum_resident_tile_count,
        )
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


def _console_or_cli_entry() -> int | None:
    package_path = globals().get("FIREVIEWER_PACKAGE")
    output_path = globals().get("FIREVIEWER_OUTPUT")
    orthophoto_source_path = globals().get("FIREVIEWER_ORTHOPHOTO_SOURCE")
    mid_vegetation_path = globals().get("FIREVIEWER_MID_VEGETATION")
    mid_orthophoto_source_path = globals().get("FIREVIEWER_MID_ORTHOPHOTO_SOURCE")
    tile_index_path = globals().get("FIREVIEWER_TILE_INDEX")
    tile_ids = globals().get("FIREVIEWER_TILE_IDS")
    tile_load_mode = globals().get("FIREVIEWER_TILE_LOAD_MODE", "visible")
    tile_focus_l93_m = globals().get("FIREVIEWER_TILE_FOCUS_L93")
    tile_visible_radius_m = globals().get("FIREVIEWER_TILE_VISIBLE_RADIUS_M")
    scene_lod_view_distance_m = globals().get("FIREVIEWER_SCENE_LOD_VIEW_DISTANCE_M")
    scene_lod_near_max_m = globals().get(
        "FIREVIEWER_SCENE_LOD_NEAR_MAX_M", SCENE_DISTANCE_LOD_NEAR_MAX_M
    )
    scene_lod_far_min_m = globals().get(
        "FIREVIEWER_SCENE_LOD_FAR_MIN_M", SCENE_DISTANCE_LOD_FAR_MIN_M
    )
    scene_lod_detail_radius_m = globals().get(
        "FIREVIEWER_SCENE_LOD_DETAIL_RADIUS_M",
        SCENE_DISTANCE_LOD_DETAIL_RADIUS_M,
    )
    maximum_resident_tile_count = globals().get(
        "FIREVIEWER_MAXIMUM_RESIDENT_TILE_COUNT",
        SCENE_DISTANCE_LOD_MAX_RESIDENT_TILES,
    )
    if package_path:
        build_from_package(
            package_path,
            output_path,
            orthophoto_source_path=orthophoto_source_path,
            mid_vegetation_path=mid_vegetation_path,
            mid_orthophoto_source_path=mid_orthophoto_source_path,
            tile_index_path=tile_index_path,
            tile_ids=tile_ids,
            tile_load_mode=tile_load_mode,
            tile_focus_l93_m=tile_focus_l93_m,
            tile_visible_radius_m=tile_visible_radius_m,
            scene_lod_view_distance_m=scene_lod_view_distance_m,
            scene_lod_near_max_m=scene_lod_near_max_m,
            scene_lod_far_min_m=scene_lod_far_min_m,
            scene_lod_detail_radius_m=scene_lod_detail_radius_m,
            maximum_resident_tile_count=maximum_resident_tile_count,
        )
        return 0
    if "--" in sys.argv:
        return main()
    print(
        "FireViewer builder loaded. Set FIREVIEWER_PACKAGE (and optionally "
        "FIREVIEWER_OUTPUT) before exec(), or call build_from_package(path, output)."
    )
    return None


if __name__ == "__main__":
    result = _console_or_cli_entry()
    if result not in (None, 0):
        raise SystemExit(result)
