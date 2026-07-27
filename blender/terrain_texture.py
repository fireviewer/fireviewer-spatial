"""Georeferenced terrain texture helpers for the Blender control scene.

The geospatial part of this module is deliberately independent from ``bpy``.
Terrain vertices are stored in a local metre frame while the orthophoto bounds
are expressed in Lambert-93.  UVs are therefore recovered with the following
round-trip for every Blender mesh loop::

    east_l93 = local_x + origin_l93_x
    north_l93 = local_y + origin_l93_y
    u = (east_l93 - west) / (east - west)
    v = (north_l93 - south) / (north - south)

Blender UV origin is the lower-left corner, so a conventional north-up raster
uses ``south -> v=0`` and ``north -> v=1``.  No vertical flip is required.
Bounds are outer pixel bounds (for example ``rasterio.DatasetReader.bounds``),
not first/last pixel-centre coordinates.  Consequently a terrain vertex at a
raster cell centre lands at the centre of the corresponding texture pixel.

The default Blender graph keeps ``Value=1`` and ``Saturation=1`` because the
derived production JPEG already carries the display correction.  That same
texture drives a 45% Principled / 55% Emission mix.  A separate
``gltf_principled`` mode keeps the core glTF graph available; the balanced
Blender graph must be baked before glTF export.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


GLTF_CORE_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
SOLID_VIEW_FALLBACK_RGBA = (0.12, 0.18, 0.10, 1.0)


@dataclass(frozen=True)
class Lambert93Bounds:
    """Axis-aligned, outer image bounds in EPSG:2154 metres."""

    west_m: float
    south_m: float
    east_m: float
    north_m: float

    def __post_init__(self) -> None:
        values = (self.west_m, self.south_m, self.east_m, self.north_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Lambert-93 image bounds must contain only finite values")
        if self.east_m <= self.west_m:
            raise ValueError(
                "Lambert-93 image east bound must be greater than west bound"
            )
        if self.north_m <= self.south_m:
            raise ValueError(
                "Lambert-93 image north bound must be greater than south bound"
            )

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> Lambert93Bounds:
        if len(values) != 4:
            raise ValueError(
                "Lambert-93 image bounds must contain west, south, east and north"
            )
        try:
            return cls(*(float(value) for value in values))
        except (TypeError, ValueError) as error:
            raise ValueError("Lambert-93 image bounds must be numeric") from error

    @property
    def width_m(self) -> float:
        return self.east_m - self.west_m

    @property
    def height_m(self) -> float:
        return self.north_m - self.south_m

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.west_m, self.south_m, self.east_m, self.north_m


@dataclass(frozen=True)
class OrthophotoMaterialConfig:
    """Stable Blender and opt-in glTF material settings for an orthophoto."""

    material_name: str = "MAT_TerrainOrthophoto"
    uv_layer_name: str = "UVMap"
    interpolation: str = "Linear"
    extension: str = "EXTEND"
    roughness: float = 0.95
    metallic: float = 0.0
    shader_mode: str = "blender_balanced"
    texture_value: float = 1.0
    texture_saturation: float = 1.0
    principled_mix_fraction: float = 0.45
    emission_mix_fraction: float = 0.55
    emission_strength: float = 1.0
    gltf_emission_strength: float = 0.2
    boundary_tolerance_m: float = 0.01
    require_gltf_core_image: bool = True
    pack_image_in_blend: bool = False
    replace_existing_materials: bool = True
    solid_view_fallback_rgba: tuple[float, float, float, float] = (
        SOLID_VIEW_FALLBACK_RGBA
    )

    def validate(self) -> None:
        if not self.material_name.strip():
            raise ValueError("material_name cannot be empty")
        if not self.uv_layer_name.strip():
            raise ValueError("uv_layer_name cannot be empty")
        if self.interpolation != "Linear":
            raise ValueError(
                "interpolation must be 'Linear' so Blender and glTF use filtered sampling"
            )
        if self.extension != "EXTEND":
            raise ValueError(
                "extension must be 'EXTEND' so texture edges clamp instead of repeating"
            )
        if not math.isfinite(self.roughness) or not 0.0 <= self.roughness <= 1.0:
            raise ValueError("roughness must be finite and between 0 and 1")
        if not math.isfinite(self.metallic) or not 0.0 <= self.metallic <= 1.0:
            raise ValueError("metallic must be finite and between 0 and 1")
        if self.shader_mode not in {"blender_balanced", "gltf_principled"}:
            raise ValueError(
                "shader_mode must be 'blender_balanced' or 'gltf_principled'"
            )
        for name, value in (
            ("texture_value", self.texture_value),
            ("texture_saturation", self.texture_saturation),
            ("emission_strength", self.emission_strength),
            ("gltf_emission_strength", self.gltf_emission_strength),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 4.0:
                raise ValueError(f"{name} must be finite and between 0 and 4")
        for name, value in (
            ("principled_mix_fraction", self.principled_mix_fraction),
            ("emission_mix_fraction", self.emission_mix_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if not math.isclose(
            self.principled_mix_fraction + self.emission_mix_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "principled_mix_fraction and emission_mix_fraction must sum to 1"
            )
        if (
            not math.isfinite(self.boundary_tolerance_m)
            or self.boundary_tolerance_m < 0.0
        ):
            raise ValueError("boundary_tolerance_m must be finite and non-negative")
        if len(self.solid_view_fallback_rgba) != 4 or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.solid_view_fallback_rgba
        ):
            raise ValueError(
                "solid_view_fallback_rgba must contain four finite values between 0 and 1"
            )


@dataclass(frozen=True)
class UvAssignmentStatistics:
    """Observable result of assigning the georeferenced Blender UV layer."""

    uv_layer_name: str
    loop_count: int
    minimum_u: float
    minimum_v: float
    maximum_u: float
    maximum_v: float


@dataclass(frozen=True)
class AppliedTerrainOrthophoto:
    """Objects and measurements returned by the Blender integration helper."""

    material: Any
    image: Any
    statistics: UvAssignmentStatistics


def _origin_xy(origin_l93_m: Sequence[float]) -> tuple[float, float]:
    if len(origin_l93_m) != 3:
        raise ValueError("Lambert-93 origin must contain exactly x, y and z")
    try:
        x_m, y_m, z_m = (float(value) for value in origin_l93_m)
    except (TypeError, ValueError) as error:
        raise ValueError("Lambert-93 origin must contain numeric values") from error
    if not all(math.isfinite(value) for value in (x_m, y_m, z_m)):
        raise ValueError("Lambert-93 origin must contain only finite values")
    return x_m, y_m


def _local_xy(local_vertex: Sequence[float]) -> tuple[float, float]:
    if len(local_vertex) < 2:
        raise ValueError("Terrain vertex must contain at least local x and y")
    try:
        x_m = float(local_vertex[0])
        y_m = float(local_vertex[1])
    except (TypeError, ValueError) as error:
        raise ValueError("Terrain vertex local x and y must be numeric") from error
    if not math.isfinite(x_m) or not math.isfinite(y_m):
        raise ValueError("Terrain vertex local x and y must be finite")
    return x_m, y_m


def _coerce_bounds(
    bounds_l93_m: Lambert93Bounds | Sequence[float],
) -> Lambert93Bounds:
    if isinstance(bounds_l93_m, Lambert93Bounds):
        return bounds_l93_m
    return Lambert93Bounds.from_sequence(bounds_l93_m)


def _unit_coordinate(
    coordinate_m: float,
    minimum_m: float,
    maximum_m: float,
    *,
    axis_name: str,
    tolerance_m: float,
) -> float:
    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("boundary_tolerance_m must be finite and non-negative")
    if coordinate_m < minimum_m - tolerance_m or coordinate_m > maximum_m + tolerance_m:
        raise ValueError(
            f"Terrain vertex {axis_name}={coordinate_m:.6f} m lies outside orthophoto "
            f"bounds [{minimum_m:.6f}, {maximum_m:.6f}] m"
        )
    # Local package coordinates are rounded to millimetres.  Clamp only the
    # explicitly tolerated numerical overshoot at an outer image edge.
    bounded_coordinate = min(max(coordinate_m, minimum_m), maximum_m)
    return (bounded_coordinate - minimum_m) / (maximum_m - minimum_m)


def lambert93_uv_from_local_vertex(
    local_vertex: Sequence[float],
    origin_l93_m: Sequence[float],
    image_bounds_l93_m: Lambert93Bounds | Sequence[float],
    *,
    boundary_tolerance_m: float = 0.01,
) -> tuple[float, float]:
    """Map one local terrain vertex to an exact north-up orthophoto UV.

    ``origin_l93_m`` is the package origin stored in scene metadata.  Vertex Z
    and origin Z intentionally do not affect UV coordinates.
    """

    local_x_m, local_y_m = _local_xy(local_vertex)
    origin_x_m, origin_y_m = _origin_xy(origin_l93_m)
    bounds = _coerce_bounds(image_bounds_l93_m)
    east_m = origin_x_m + local_x_m
    north_m = origin_y_m + local_y_m
    return (
        _unit_coordinate(
            east_m,
            bounds.west_m,
            bounds.east_m,
            axis_name="east",
            tolerance_m=boundary_tolerance_m,
        ),
        _unit_coordinate(
            north_m,
            bounds.south_m,
            bounds.north_m,
            axis_name="north",
            tolerance_m=boundary_tolerance_m,
        ),
    )


def loop_uvs_from_indexed_faces(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    origin_l93_m: Sequence[float],
    image_bounds_l93_m: Lambert93Bounds | Sequence[float],
    *,
    boundary_tolerance_m: float = 0.01,
) -> list[tuple[float, float]]:
    """Return UVs in the face-loop order used by ``Mesh.from_pydata``."""

    result: list[tuple[float, float]] = []
    for face_number, face in enumerate(faces):
        if len(face) < 3:
            raise ValueError(
                f"Terrain face {face_number} must contain at least 3 vertices"
            )
        for index in face:
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(
                    f"Terrain face {face_number} contains a non-integer index"
                )
            if index < 0 or index >= len(vertices):
                raise ValueError(
                    f"Terrain face {face_number} references vertex {index}, "
                    f"but the mesh has {len(vertices)} vertices"
                )
            result.append(
                lambert93_uv_from_local_vertex(
                    vertices[index],
                    origin_l93_m,
                    image_bounds_l93_m,
                    boundary_tolerance_m=boundary_tolerance_m,
                )
            )
    return result


def validate_gltf_core_image_path(image_path: str | Path) -> Path:
    """Resolve and validate an image encodable by core glTF 2.0.

    Core glTF supports PNG and JPEG.  GeoTIFF/COG inputs must first be rendered
    to a north-up PNG or JPEG while retaining their Lambert-93 outer bounds.
    """

    path = Path(image_path).expanduser().resolve()
    if path.suffix.casefold() not in GLTF_CORE_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(GLTF_CORE_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Orthophoto image {path.name!r} is not a core glTF image; expected {supported}"
        )
    return path


def connect_orthophoto_color_to_principled(
    image_texture: Any,
    principled: Any,
    links: Any,
    *,
    emission_strength: float,
) -> dict[str, Any]:
    """Connect one orthophoto to albedo and a restrained emissive component.

    Blender 5.2 names the input ``Emission Color``.  The legacy ``Emission``
    fallback keeps the material helper usable with older Blender files while
    failing explicitly if neither socket exists.
    """

    if not math.isfinite(emission_strength) or not 0.0 <= emission_strength <= 1.0:
        raise ValueError("emission_strength must be finite and between 0 and 1")
    base_color = principled.inputs.get("Base Color")
    emission_color = principled.inputs.get("Emission Color")
    if emission_color is None:
        emission_color = principled.inputs.get("Emission")
    emission_power = principled.inputs.get("Emission Strength")
    color_output = image_texture.outputs.get("Color")
    if base_color is None or emission_color is None or emission_power is None:
        raise RuntimeError("Principled BSDF has no compatible color/emission sockets")
    if color_output is None:
        raise RuntimeError("Orthophoto image node has no Color output")
    links.new(color_output, base_color)
    links.new(color_output, emission_color)
    emission_power.default_value = emission_strength
    return {
        "base_color_source": "orthophoto_srgb",
        "emission_color_source": "same_orthophoto_srgb",
        "emission_strength": emission_strength,
    }


def connect_balanced_orthophoto_shader(
    image_texture: Any,
    color_adjustment: Any,
    principled: Any,
    emission: Any,
    mix_shader: Any,
    material_output: Any,
    links: Any,
    *,
    config: OrthophotoMaterialConfig,
) -> dict[str, Any]:
    """Wire the exact Blender v5 terrain shader contract."""

    config.validate()
    if config.shader_mode != "blender_balanced":
        raise ValueError("balanced shader wiring requires blender_balanced mode")
    image_color = image_texture.outputs.get("Color")
    adjusted_color_input = color_adjustment.inputs.get("Color")
    adjusted_color = color_adjustment.outputs.get("Color")
    base_color = principled.inputs.get("Base Color")
    emission_color = emission.inputs.get("Color")
    emission_power = emission.inputs.get("Strength")
    principled_shader = principled.outputs.get("BSDF")
    emission_shader = emission.outputs.get("Emission")
    mixed_shader = mix_shader.outputs.get("Shader")
    surface = material_output.inputs.get("Surface")
    required = (
        image_color,
        adjusted_color_input,
        adjusted_color,
        base_color,
        emission_color,
        emission_power,
        principled_shader,
        emission_shader,
        mixed_shader,
        surface,
    )
    if any(socket is None for socket in required):
        raise RuntimeError("Balanced orthophoto shader has incompatible node sockets")
    color_adjustment.inputs["Hue"].default_value = 0.5
    color_adjustment.inputs["Saturation"].default_value = config.texture_saturation
    color_adjustment.inputs["Value"].default_value = config.texture_value
    color_adjustment.inputs["Fac"].default_value = 1.0
    emission_power.default_value = config.emission_strength
    mix_shader.inputs[0].default_value = config.emission_mix_fraction
    links.new(image_color, adjusted_color_input)
    links.new(adjusted_color, base_color)
    links.new(adjusted_color, emission_color)
    links.new(principled_shader, mix_shader.inputs[1])
    links.new(emission_shader, mix_shader.inputs[2])
    links.new(mixed_shader, surface)
    return {
        "shader_mode": config.shader_mode,
        "texture_value": config.texture_value,
        "texture_saturation": config.texture_saturation,
        "principled_mix_fraction": config.principled_mix_fraction,
        "emission_mix_fraction": config.emission_mix_fraction,
        "emission_strength": config.emission_strength,
        "texture_contract": "same_adjusted_texture_for_principled_and_emission",
        "gltf_export": "bake_balanced_graph_or_use_gltf_principled_mode",
    }


def assign_lambert93_uv_layer(
    mesh: Any,
    origin_l93_m: Sequence[float],
    image_bounds_l93_m: Lambert93Bounds | Sequence[float],
    *,
    uv_layer_name: str = "UVMap",
    boundary_tolerance_m: float = 0.01,
) -> UvAssignmentStatistics:
    """Assign exact UVs to a Blender mesh without importing ``bpy``.

    The ``mesh`` contract is the regular Blender ``Mesh`` API: ``vertices``,
    ``loops`` and ``uv_layers``.  Keeping this function free of a direct
    ``bpy`` import makes its geospatial behaviour unit-testable in normal
    Python.
    """

    if not uv_layer_name.strip():
        raise ValueError("uv_layer_name cannot be empty")
    loops = mesh.loops
    if len(loops) == 0:
        raise ValueError("Terrain mesh has no loops to texture")
    uv_layer = mesh.uv_layers.get(uv_layer_name)
    if uv_layer is None:
        uv_layer = mesh.uv_layers.new(name=uv_layer_name)
    values: list[tuple[float, float]] = []
    for loop in loops:
        vertex_index = int(loop.vertex_index)
        if vertex_index < 0 or vertex_index >= len(mesh.vertices):
            raise ValueError(
                f"Terrain loop {loop.index} references invalid vertex {vertex_index}"
            )
        coordinate = mesh.vertices[vertex_index].co
        uv = lambert93_uv_from_local_vertex(
            coordinate,
            origin_l93_m,
            image_bounds_l93_m,
            boundary_tolerance_m=boundary_tolerance_m,
        )
        uv_layer.data[loop.index].uv = uv
        values.append(uv)
    mesh.uv_layers.active = uv_layer
    if hasattr(uv_layer, "active_render"):
        uv_layer.active_render = True
    return UvAssignmentStatistics(
        uv_layer_name=uv_layer_name,
        loop_count=len(values),
        minimum_u=min(value[0] for value in values),
        minimum_v=min(value[1] for value in values),
        maximum_u=max(value[0] for value in values),
        maximum_v=max(value[1] for value in values),
    )


def create_orthophoto_material(
    bpy: Any,
    image_path: str | Path,
    *,
    config: OrthophotoMaterialConfig | None = None,
) -> tuple[Any, Any]:
    """Create the balanced Blender shader or the opt-in glTF core shader.

    ``Linear`` texture-node interpolation gives filtered viewport sampling.
    Blender uses mipmaps for image textures when available; older APIs expose
    this as ``Image.use_mipmap`` and it is enabled explicitly when present.
    glTF runtimes generate/use mipmaps through their normal sampler path.  The
    default Mix Shader graph is Blender-specific and must be baked for glTF;
    ``shader_mode='gltf_principled'`` avoids that graph.
    """

    active_config = config or OrthophotoMaterialConfig()
    active_config.validate()
    path = (
        validate_gltf_core_image_path(image_path)
        if active_config.require_gltf_core_image
        else Path(image_path).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"Orthophoto image does not exist: {path}")

    image = bpy.data.images.load(str(path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    if hasattr(image, "use_mipmap"):
        image.use_mipmap = True
    if active_config.pack_image_in_blend and not image.packed_file:
        image.pack()

    material = bpy.data.materials.get(active_config.material_name)
    if material is None:
        material = bpy.data.materials.new(active_config.material_name)
    material.use_nodes = True
    # Blender's Solid viewport ignores the node graph and displays this value.
    # Keep a restrained terrain colour so opening the control scene outside
    # Material Preview never makes the complete textured ground look white.
    material.diffuse_color = active_config.solid_view_fallback_rgba
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = "Orthophoto Output"
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Orthophoto Principled"
    image_texture = nodes.new("ShaderNodeTexImage")
    image_texture.name = "Orthophoto Image"
    image_texture.image = image
    image_texture.interpolation = active_config.interpolation
    image_texture.projection = "FLAT"
    image_texture.extension = active_config.extension
    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.name = "Orthophoto UV"
    uv_map.uv_map = active_config.uv_layer_name

    principled.inputs["Metallic"].default_value = active_config.metallic
    principled.inputs["Roughness"].default_value = active_config.roughness
    links.new(uv_map.outputs["UV"], image_texture.inputs["Vector"])
    if active_config.shader_mode == "blender_balanced":
        color_adjustment = nodes.new("ShaderNodeHueSaturation")
        color_adjustment.name = "Orthophoto Value Saturation"
        emission = nodes.new("ShaderNodeEmission")
        emission.name = "Orthophoto Emission"
        mix_shader = nodes.new("ShaderNodeMixShader")
        mix_shader.name = "Orthophoto Principled Emission Mix"
        emission_contract = connect_balanced_orthophoto_shader(
            image_texture,
            color_adjustment,
            principled,
            emission,
            mix_shader,
            output,
            links,
            config=active_config,
        )
    else:
        emission_contract = connect_orthophoto_color_to_principled(
            image_texture,
            principled,
            links,
            emission_strength=active_config.gltf_emission_strength,
        )
        emission_contract.update(
            {
                "shader_mode": active_config.shader_mode,
                "gltf_export": "core_principled_image_texture_graph",
            }
        )
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    actual_emission_strength = (
        active_config.emission_strength
        if active_config.shader_mode == "blender_balanced"
        else active_config.gltf_emission_strength
    )
    material["fireviewer_orthophoto_shader_mode"] = active_config.shader_mode
    material["fireviewer_orthophoto_emission_strength"] = actual_emission_strength
    material["fireviewer_orthophoto_emission_mix_fraction"] = (
        active_config.emission_mix_fraction
    )
    material["fireviewer_orthophoto_texture_value"] = active_config.texture_value
    material["fireviewer_orthophoto_texture_saturation"] = (
        active_config.texture_saturation
    )
    material["fireviewer_solid_view_fallback_rgba_json"] = json.dumps(
        list(active_config.solid_view_fallback_rgba), separators=(",", ":")
    )
    material["fireviewer_orthophoto_shader_json"] = json.dumps(
        emission_contract, sort_keys=True
    )
    return material, image


def apply_georeferenced_orthophoto(
    bpy: Any,
    terrain_object: Any,
    image_path: str | Path,
    origin_l93_m: Sequence[float],
    image_bounds_l93_m: Lambert93Bounds | Sequence[float],
    *,
    config: OrthophotoMaterialConfig | None = None,
) -> AppliedTerrainOrthophoto:
    """UV-map a terrain object and replace its material with an orthophoto.

    The terrain object's mesh coordinates must still use the package-local
    X=east/Y=north frame.  Object transforms are intentionally ignored because
    ``origin_l93_m`` reconstructs Lambert-93 from those source coordinates.
    """

    active_config = config or OrthophotoMaterialConfig()
    active_config.validate()
    if getattr(terrain_object, "type", None) != "MESH":
        raise ValueError("Orthophoto can only be applied to a Blender mesh object")
    mesh = terrain_object.data
    statistics = assign_lambert93_uv_layer(
        mesh,
        origin_l93_m,
        image_bounds_l93_m,
        uv_layer_name=active_config.uv_layer_name,
        boundary_tolerance_m=active_config.boundary_tolerance_m,
    )
    material, image = create_orthophoto_material(
        bpy,
        image_path,
        config=active_config,
    )
    if active_config.replace_existing_materials:
        mesh.materials.clear()
    material_index = mesh.materials.find(material.name)
    if material_index < 0:
        mesh.materials.append(material)
        material_index = len(mesh.materials) - 1
    for polygon in mesh.polygons:
        polygon.material_index = material_index
    terrain_object["texture_shader_mode"] = active_config.shader_mode
    terrain_object["texture_value"] = active_config.texture_value
    terrain_object["texture_saturation"] = active_config.texture_saturation
    terrain_object["texture_principled_mix_fraction"] = (
        active_config.principled_mix_fraction
    )
    terrain_object["texture_emission_mix_fraction"] = (
        active_config.emission_mix_fraction
    )
    terrain_object["texture_emission_strength"] = (
        active_config.emission_strength
        if active_config.shader_mode == "blender_balanced"
        else active_config.gltf_emission_strength
    )
    terrain_object["texture_emission_source"] = (
        "same_neutral_texture_balanced_mix"
        if active_config.shader_mode == "blender_balanced"
        else "same_orthophoto_principled_emission"
    )
    return AppliedTerrainOrthophoto(
        material=material,
        image=image,
        statistics=statistics,
    )


__all__ = [
    "AppliedTerrainOrthophoto",
    "GLTF_CORE_IMAGE_EXTENSIONS",
    "Lambert93Bounds",
    "OrthophotoMaterialConfig",
    "SOLID_VIEW_FALLBACK_RGBA",
    "UvAssignmentStatistics",
    "apply_georeferenced_orthophoto",
    "assign_lambert93_uv_layer",
    "connect_balanced_orthophoto_shader",
    "create_orthophoto_material",
    "connect_orthophoto_color_to_principled",
    "lambert93_uv_from_local_vertex",
    "loop_uvs_from_indexed_faces",
    "validate_gltf_core_image_path",
]
