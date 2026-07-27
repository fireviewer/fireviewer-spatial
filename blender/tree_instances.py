"""Compact, deterministic and non-realized Blender tree instances.

This module replaces one-mesh-per-tree and biconic visual proxies with a small
shared prototype library.  It has two deliberately separate halves:

* :func:`build_tree_instance_set` is pure Python and package-preparation safe.
  It accepts every measured tree/crown candidate and emits compact base64
  buffers plus reusable prototype meshes.  It performs no thinning and has no
  implicit instance budget.
* :func:`build_blender_tree_system` is imported only by Blender.  It creates a
  single point mesh, one Geometry Nodes modifier, and an unlinked collection of
  shared prototypes.  The node graph leaves the instances unrealized.

Canonical input record fields are ``x_m``, ``y_m``,
``ground_elevation_m``, ``height_m`` and ``crown_diameter_m``.  Records may be
mappings or objects (including ``vegetation_lod.TreeProxy``).  Optional
``vegetation_type`` values can explicitly select ``broadleaf`` or ``conifer``;
``source_id`` can provide a stable identifier.  When type is absent, both
silhouette families are used as deterministic *visual forms only*.  They must
not be interpreted as a remotely sensed species classification.

Integration in the package-preparation process::

    instance_set = build_tree_instance_set(
        detected_proxies,  # every candidate; do not apply 15 m thinning here
        origin_l93_m,
        TreeInstanceConfig(profile="global_mid"),
    )
    package["vegetation"]["tree_instances"] = instance_set

Integration in ``build_control_scene.py`` after creating the Vegetation
collection::

    from tree_instances import build_blender_tree_system
    build_blender_tree_system(
        bpy,
        package["vegetation"]["tree_instances"],
        collections["Vegetation"],
        name="VegetationTrees",
    )

The source coordinates are metres in the package CRS, Z is up, and the local
position is relative to the supplied package origin.  Prototype height is one
metre and crown diameter is one metre before instance scaling.  Z scale is the
measured height; XY scale preserves the area represented by the measured
equivalent crown diameter while adding a small deterministic anisotropy.

The Geometry Nodes result is appropriate for a controlled Blender scene.  A
plain GLB export generally realizes evaluated instances and can therefore be
very large.  Shipping to the browser should retain this instance buffer (or
spatially tile it) rather than exporting hundreds of thousands of realized
prototype copies.
"""

from __future__ import annotations

from array import array
import base64
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import sys
from typing import Any
import unicodedata


TREE_INSTANCE_SCHEMA = "fireviewer.blender-tree-instances.v1"
BUFFER_SCHEMA = "fireviewer.numeric-buffer.v1"

_MATERIAL_NAMES = (
    "trunk",
    "foliage_dark",
    "foliage_mid",
    "foliage_light",
)


@dataclass(frozen=True)
class TreeInstanceConfig:
    """Controls reusable silhouettes, never tree detection or thinning."""

    profile: str = "global_mid"
    seed: int = 0x5EED_2026
    broadleaf_variant_count: int = 6
    conifer_variant_count: int = 4
    unknown_visual_conifer_fraction: float = 0.38
    maximum_crown_anisotropy: float = 1.12

    def __post_init__(self) -> None:
        if self.profile not in {"global_mid", "close_up"}:
            raise ValueError("profile must be 'global_mid' or 'close_up'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name, value in (
            ("broadleaf_variant_count", self.broadleaf_variant_count),
            ("conifer_variant_count", self.conifer_variant_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a strictly positive integer")
        if self.broadleaf_variant_count + self.conifer_variant_count > 256:
            raise ValueError("the total prototype count must not exceed 256")
        if not math.isfinite(self.unknown_visual_conifer_fraction) or not (
            0.0 <= self.unknown_visual_conifer_fraction <= 1.0
        ):
            raise ValueError("unknown_visual_conifer_fraction must be in [0, 1]")
        if not math.isfinite(self.maximum_crown_anisotropy) or not (
            1.0 <= self.maximum_crown_anisotropy <= 1.5
        ):
            raise ValueError("maximum_crown_anisotropy must be in [1, 1.5]")


def _stable_digest(seed: int, source_key: str) -> bytes:
    payload = f"{seed}:{source_key}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _unit_interval(digest: bytes, offset: int) -> float:
    return int.from_bytes(digest[offset : offset + 4], "little") / 2**32


def _variant_rng(seed: int, family: str, variant_index: int) -> Any:
    """Small stable generator independent of Python's randomized hash seed."""

    state = int.from_bytes(
        hashlib.sha256(f"{seed}:{family}:{variant_index}".encode()).digest()[:8],
        "little",
    )

    def next_value() -> float:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 31
        return (value & 0xFFFFFFFFFFFFFFFF) / 2**64

    return next_value


def _append_cylinder(
    vertices: list[list[float]],
    faces: list[list[int]],
    material_indices: list[int],
    smooth_faces: list[bool],
    *,
    radius: float,
    top_z: float,
    radial_segments: int,
    phase: float,
) -> None:
    bottom: list[int] = []
    top: list[int] = []
    for z, indices in ((0.0, bottom), (top_z, top)):
        for segment in range(radial_segments):
            angle = phase + segment * 2.0 * math.pi / radial_segments
            indices.append(len(vertices))
            vertices.append([radius * math.cos(angle), radius * math.sin(angle), z])
    for segment in range(radial_segments):
        following = (segment + 1) % radial_segments
        faces.append([bottom[segment], bottom[following], top[following], top[segment]])
        material_indices.append(0)
        smooth_faces.append(True)
    faces.append(list(reversed(bottom)))
    material_indices.append(0)
    smooth_faces.append(False)
    faces.append(top)
    material_indices.append(0)
    smooth_faces.append(False)


_ICO_VERTICES = (
    (-1, 1.61803398875, 0),
    (1, 1.61803398875, 0),
    (-1, -1.61803398875, 0),
    (1, -1.61803398875, 0),
    (0, -1, 1.61803398875),
    (0, 1, 1.61803398875),
    (0, -1, -1.61803398875),
    (0, 1, -1.61803398875),
    (1.61803398875, 0, -1),
    (1.61803398875, 0, 1),
    (-1.61803398875, 0, -1),
    (-1.61803398875, 0, 1),
)
_ICO_FACES = (
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
)
_ICO_NORMALIZED = tuple(
    tuple(component / math.sqrt(x * x + y * y + z * z) for component in (x, y, z))
    for x, y, z in _ICO_VERTICES
)


def _append_ico_lobe(
    vertices: list[list[float]],
    faces: list[list[int]],
    material_indices: list[int],
    smooth_faces: list[bool],
    *,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    material_index: int,
    rotation: float,
) -> list[int]:
    base = len(vertices)
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    indices: list[int] = []
    for source_x, source_y, source_z in _ICO_NORMALIZED:
        rotated_x = source_x * cosine - source_y * sine
        rotated_y = source_x * sine + source_y * cosine
        indices.append(len(vertices))
        vertices.append(
            [
                center[0] + rotated_x * radii[0],
                center[1] + rotated_y * radii[1],
                center[2] + source_z * radii[2],
            ]
        )
    for face in _ICO_FACES:
        faces.append([base + index for index in face])
        material_indices.append(material_index)
        smooth_faces.append(True)
    return indices


def _normalize_crown_vertices(
    vertices: list[list[float]], crown_indices: Sequence[int], crown_base_z: float
) -> None:
    maximum_radius = max(
        math.hypot(vertices[index][0], vertices[index][1]) for index in crown_indices
    )
    if maximum_radius <= 0:
        raise RuntimeError("generated crown has no horizontal extent")
    horizontal_scale = 0.5 / maximum_radius
    minimum_z = min(vertices[index][2] for index in crown_indices)
    maximum_z = max(vertices[index][2] for index in crown_indices)
    if maximum_z <= minimum_z:
        raise RuntimeError("generated crown has no vertical extent")
    for index in crown_indices:
        vertex = vertices[index]
        vertex[0] *= horizontal_scale
        vertex[1] *= horizontal_scale
        vertex[2] = crown_base_z + (vertex[2] - minimum_z) * (
            (1.0 - crown_base_z) / (maximum_z - minimum_z)
        )


def _broadleaf_prototype(
    config: TreeInstanceConfig, variant_index: int
) -> dict[str, Any]:
    random_value = _variant_rng(config.seed, "broadleaf", variant_index)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    material_indices: list[int] = []
    smooth_faces: list[bool] = []
    crown_indices: list[int] = []
    lobe_count = (4 if config.profile == "global_mid" else 6) + variant_index % 2
    crown_base_z = 0.27 + 0.05 * random_value()
    palette_index = 1 + variant_index % 3

    # One high central mass guarantees a readable crown and the measured apex.
    crown_indices.extend(
        _append_ico_lobe(
            vertices,
            faces,
            material_indices,
            smooth_faces,
            center=(0.0, 0.0, 0.73),
            radii=(0.31, 0.29, 0.29),
            material_index=palette_index,
            rotation=random_value() * math.pi,
        )
    )
    for lobe_index in range(lobe_count - 1):
        angle = (
            2.0 * math.pi * lobe_index / max(1, lobe_count - 1)
            + (random_value() - 0.5) * 0.7
        )
        radial_offset = 0.14 + 0.07 * random_value()
        radius_x = 0.20 + 0.07 * random_value()
        radius_y = 0.20 + 0.07 * random_value()
        radius_z = 0.19 + 0.07 * random_value()
        crown_indices.extend(
            _append_ico_lobe(
                vertices,
                faces,
                material_indices,
                smooth_faces,
                center=(
                    math.cos(angle) * radial_offset,
                    math.sin(angle) * radial_offset,
                    0.60 + 0.17 * random_value(),
                ),
                radii=(radius_x, radius_y, radius_z),
                material_index=palette_index,
                rotation=random_value() * math.pi,
            )
        )
    _normalize_crown_vertices(vertices, crown_indices, crown_base_z)

    trunk_segments = 7 if config.profile == "global_mid" else 10
    _append_cylinder(
        vertices,
        faces,
        material_indices,
        smooth_faces,
        radius=0.027 + 0.012 * random_value(),
        top_z=crown_base_z + 0.12,
        radial_segments=trunk_segments,
        phase=random_value() * math.pi,
    )
    return _prototype_mapping(
        f"Broadleaf_{variant_index:02d}",
        "broadleaf",
        vertices,
        faces,
        material_indices,
        smooth_faces,
        crown_volume_count=lobe_count,
    )


def _append_cone_layer(
    vertices: list[list[float]],
    faces: list[list[int]],
    material_indices: list[int],
    smooth_faces: list[bool],
    *,
    base_z: float,
    apex_z: float,
    radius_x: float,
    radius_y: float,
    radial_segments: int,
    phase: float,
    material_index: int,
) -> None:
    ring: list[int] = []
    for segment in range(radial_segments):
        angle = phase + segment * 2.0 * math.pi / radial_segments
        ring.append(len(vertices))
        vertices.append(
            [radius_x * math.cos(angle), radius_y * math.sin(angle), base_z]
        )
    apex = len(vertices)
    vertices.append([0.0, 0.0, apex_z])
    for segment in range(radial_segments):
        faces.append([ring[segment], ring[(segment + 1) % radial_segments], apex])
        material_indices.append(material_index)
        smooth_faces.append(True)
    faces.append(list(reversed(ring)))
    material_indices.append(material_index)
    smooth_faces.append(False)


def _conifer_prototype(
    config: TreeInstanceConfig, variant_index: int
) -> dict[str, Any]:
    random_value = _variant_rng(config.seed, "conifer", variant_index)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    material_indices: list[int] = []
    smooth_faces: list[bool] = []
    radial_segments = 8 if config.profile == "global_mid" else 12
    layer_count = (4 if config.profile == "global_mid" else 6) + variant_index % 2
    palette_index = 1 + variant_index % 3
    lower_base = 0.21 + 0.05 * random_value()
    for layer_index in range(layer_count):
        progress = layer_index / max(1, layer_count - 1)
        base_z = lower_base + progress * 0.56
        apex_z = min(1.0, base_z + 0.43 - progress * 0.18)
        radius = 0.5 * (1.0 - 0.60 * progress)
        ellipticity = 0.92 + 0.16 * random_value()
        _append_cone_layer(
            vertices,
            faces,
            material_indices,
            smooth_faces,
            base_z=base_z,
            apex_z=1.0 if layer_index == layer_count - 1 else apex_z,
            radius_x=radius * math.sqrt(ellipticity),
            radius_y=radius / math.sqrt(ellipticity),
            radial_segments=radial_segments,
            phase=random_value() * 2.0 * math.pi / radial_segments,
            material_index=palette_index,
        )
    maximum_radius = max(math.hypot(vertex[0], vertex[1]) for vertex in vertices)
    for vertex in vertices:
        vertex[0] *= 0.5 / maximum_radius
        vertex[1] *= 0.5 / maximum_radius
    _append_cylinder(
        vertices,
        faces,
        material_indices,
        smooth_faces,
        radius=0.022 + 0.010 * random_value(),
        top_z=0.68 + 0.08 * random_value(),
        radial_segments=radial_segments,
        phase=random_value() * math.pi,
    )
    return _prototype_mapping(
        f"Conifer_{variant_index:02d}",
        "conifer",
        vertices,
        faces,
        material_indices,
        smooth_faces,
        crown_volume_count=layer_count,
    )


def _prototype_mapping(
    name: str,
    visual_form: str,
    vertices: list[list[float]],
    faces: list[list[int]],
    material_indices: list[int],
    smooth_faces: list[bool],
    *,
    crown_volume_count: int,
) -> dict[str, Any]:
    rounded_vertices = [
        [round(component, 6) for component in vertex] for vertex in vertices
    ]
    triangle_count = sum(max(1, len(face) - 2) for face in faces)
    return {
        "name": name,
        "visual_form": visual_form,
        "normalized_height_m": 1.0,
        "normalized_crown_diameter_m": 1.0,
        "crown_volume_count": crown_volume_count,
        "mesh": {
            "vertices": rounded_vertices,
            "faces": faces,
            "material_indices": material_indices,
            "smooth_faces": smooth_faces,
            "estimated_triangle_count": triangle_count,
        },
    }


def build_tree_prototype_library(
    config: TreeInstanceConfig | None = None,
) -> list[dict[str, Any]]:
    """Return normalized, recognizable and reusable tree silhouettes."""

    active_config = config or TreeInstanceConfig()
    result = [
        _broadleaf_prototype(active_config, index)
        for index in range(active_config.broadleaf_variant_count)
    ]
    result.extend(
        _conifer_prototype(active_config, index)
        for index in range(active_config.conifer_variant_count)
    )
    return result


def _little_endian_bytes(values: array) -> bytes:
    if values.itemsize > 1 and sys.byteorder != "little":
        copied = array(values.typecode, values)
        copied.byteswap()
        return copied.tobytes()
    return values.tobytes()


def _encode_buffer(
    values: array, component_type: str, components: int
) -> dict[str, Any]:
    raw = _little_endian_bytes(values)
    if len(values) % components:
        raise ValueError(
            "numeric buffer length is not divisible by its component count"
        )
    return {
        "schema": BUFFER_SCHEMA,
        "component_type": component_type,
        "components": components,
        "count": len(values) // components,
        "byte_order": "little",
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


_COMPONENT_INFO = {
    "FLOAT32": ("f", 4),
    "UINT8": ("B", 1),
}


def decode_numeric_buffer(buffer: Mapping[str, Any]) -> array:
    """Decode and integrity-check one compact instance attribute buffer."""

    if buffer.get("schema") != BUFFER_SCHEMA:
        raise ValueError(f"unsupported numeric buffer schema: {buffer.get('schema')!r}")
    component_type = buffer.get("component_type")
    if component_type not in _COMPONENT_INFO:
        raise ValueError(f"unsupported numeric component type: {component_type!r}")
    if buffer.get("byte_order") != "little":
        raise ValueError("numeric buffers must use little-endian byte order")
    components = buffer.get("components")
    count = buffer.get("count")
    if (
        isinstance(components, bool)
        or not isinstance(components, int)
        or components <= 0
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
    ):
        raise ValueError("numeric buffer components/count are invalid")
    try:
        raw = base64.b64decode(buffer["data_base64"], validate=True)
    except (KeyError, ValueError) as error:
        raise ValueError("numeric buffer base64 payload is invalid") from error
    typecode, byte_width = _COMPONENT_INFO[component_type]
    expected_length = components * count * byte_width
    if len(raw) != expected_length:
        raise ValueError(
            "numeric buffer byte length mismatch: "
            f"expected {expected_length}, got {len(raw)}"
        )
    if hashlib.sha256(raw).hexdigest() != buffer.get("sha256"):
        raise ValueError("numeric buffer SHA-256 mismatch")
    values = array(typecode)
    values.frombytes(raw)
    if byte_width > 1 and sys.byteorder != "little":
        values.byteswap()
    return values


def _record_value(record: Any, key: str, *, optional: bool = False) -> Any:
    if isinstance(record, Mapping):
        if optional:
            return record.get(key)
        if key not in record:
            raise ValueError(f"tree record is missing {key!r}")
        return record[key]
    if optional:
        return getattr(record, key, None)
    try:
        return getattr(record, key)
    except AttributeError as error:
        raise ValueError(f"tree record is missing {key!r}") from error


def _finite_float(record: Any, key: str, *, positive: bool = False) -> float:
    try:
        value = float(_record_value(record, key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"tree record {key!r} must be numeric") from error
    if not math.isfinite(value) or (positive and value <= 0):
        qualifier = "finite and strictly positive" if positive else "finite"
        raise ValueError(f"tree record {key!r} must be {qualifier}")
    return value


def _normalized_tree_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", str(value).strip().lower())
        if not unicodedata.combining(character)
    )
    broadleaf = {"broadleaf", "broadleaved", "deciduous", "feuillu", "feuillus"}
    conifer = {"conifer", "coniferous", "conifere", "coniferes", "resineux"}
    if normalized in broadleaf:
        return "broadleaf"
    if normalized in conifer:
        return "conifer"
    raise ValueError(
        "vegetation_type must identify broadleaf/feuillu or conifer/conifere"
    )


def _source_key(record: Any, x_m: float, y_m: float) -> str:
    explicit = _record_value(record, "source_id", optional=True)
    if explicit is not None:
        return str(explicit)
    row = _record_value(record, "row", optional=True)
    column = _record_value(record, "column", optional=True)
    if row is not None and column is not None:
        return f"r{row}:c{column}:x{x_m:.3f}:y{y_m:.3f}"
    return f"x{x_m:.3f}:y{y_m:.3f}"


def _validate_origin(origin: Sequence[float]) -> tuple[float, float, float]:
    if len(origin) != 3:
        raise ValueError("origin must contain exactly x, y and z")
    try:
        result = tuple(float(value) for value in origin)
    except (TypeError, ValueError) as error:
        raise ValueError("origin must contain numeric values") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError("origin must contain finite values")
    return result  # type: ignore[return-value]


def build_tree_instance_set(
    records: Iterable[Any],
    origin: Sequence[float],
    config: TreeInstanceConfig | None = None,
) -> dict[str, Any]:
    """Encode every input record as one deterministic shared-mesh instance.

    The function intentionally has no spacing, thinning or maximum-count
    option.  Invalid records fail the whole build instead of disappearing.
    """

    active_config = config or TreeInstanceConfig()
    origin_x, origin_y, origin_z = _validate_origin(origin)
    prototypes = build_tree_prototype_library(active_config)
    positions = array("f")
    scales = array("f")
    yaw_values = array("f")
    prototype_indices = array("B")
    explicit_broadleaf_count = 0
    explicit_conifer_count = 0
    unknown_visual_broadleaf_count = 0
    unknown_visual_conifer_count = 0
    minimum_height = math.inf
    maximum_height = -math.inf
    minimum_diameter = math.inf
    maximum_diameter = -math.inf

    for record in records:
        x_m = _finite_float(record, "x_m")
        y_m = _finite_float(record, "y_m")
        ground_elevation_m = _finite_float(record, "ground_elevation_m")
        height_m = _finite_float(record, "height_m", positive=True)
        crown_diameter_m = _finite_float(record, "crown_diameter_m", positive=True)
        source_key = _source_key(record, x_m, y_m)
        # Shape choice changes reproducibly when the measured morphology changes,
        # while the semantic warning below prevents interpreting it as species.
        digest = _stable_digest(
            active_config.seed,
            f"{source_key}:h{height_m:.3f}:d{crown_diameter_m:.3f}",
        )
        explicit_type = _normalized_tree_type(
            _record_value(record, "vegetation_type", optional=True)
        )
        if explicit_type is None:
            slenderness = height_m / crown_diameter_m
            morphology_bias = max(-0.16, min(0.22, (slenderness - 1.8) * 0.14))
            visual_conifer_threshold = max(
                0.0,
                min(
                    1.0,
                    active_config.unknown_visual_conifer_fraction + morphology_bias,
                ),
            )
            visual_form = (
                "conifer"
                if _unit_interval(digest, 0) < visual_conifer_threshold
                else "broadleaf"
            )
            if visual_form == "conifer":
                unknown_visual_conifer_count += 1
            else:
                unknown_visual_broadleaf_count += 1
        else:
            visual_form = explicit_type
            if visual_form == "conifer":
                explicit_conifer_count += 1
            else:
                explicit_broadleaf_count += 1

        if visual_form == "broadleaf":
            variant = int(
                _unit_interval(digest, 4) * active_config.broadleaf_variant_count
            )
            prototype_index = min(variant, active_config.broadleaf_variant_count - 1)
        else:
            variant = int(
                _unit_interval(digest, 4) * active_config.conifer_variant_count
            )
            prototype_index = active_config.broadleaf_variant_count + min(
                variant, active_config.conifer_variant_count - 1
            )

        maximum_anisotropy = active_config.maximum_crown_anisotropy
        exponent = (_unit_interval(digest, 8) * 2.0 - 1.0) * math.log(
            maximum_anisotropy
        )
        anisotropy = math.exp(exponent)
        x_scale = crown_diameter_m * math.sqrt(anisotropy)
        y_scale = crown_diameter_m / math.sqrt(anisotropy)
        yaw = _unit_interval(digest, 12) * 2.0 * math.pi

        positions.extend(
            (x_m - origin_x, y_m - origin_y, ground_elevation_m - origin_z)
        )
        scales.extend((x_scale, y_scale, height_m))
        yaw_values.append(yaw)
        prototype_indices.append(prototype_index)
        minimum_height = min(minimum_height, height_m)
        maximum_height = max(maximum_height, height_m)
        minimum_diameter = min(minimum_diameter, crown_diameter_m)
        maximum_diameter = max(maximum_diameter, crown_diameter_m)

    instance_count = len(prototype_indices)
    attributes = {
        "position_xyz_m": _encode_buffer(positions, "FLOAT32", 3),
        "scale_xyz": _encode_buffer(scales, "FLOAT32", 3),
        "yaw_radians": _encode_buffer(yaw_values, "FLOAT32", 1),
        "prototype_index": _encode_buffer(prototype_indices, "UINT8", 1),
    }
    prototype_triangle_count = sum(
        prototype["mesh"]["estimated_triangle_count"] for prototype in prototypes
    )
    statistics = {
        "input_record_count": instance_count,
        "instance_count": instance_count,
        "dropped_record_count": 0,
        "thinning": "none_one_input_record_per_instance",
        "point_mesh_vertex_count": instance_count,
        "prototype_count": len(prototypes),
        "shared_prototype_triangle_count": prototype_triangle_count,
        "realized_instance_geometry": False,
        "explicit_broadleaf_count": explicit_broadleaf_count,
        "explicit_conifer_count": explicit_conifer_count,
        "unknown_visual_broadleaf_count": unknown_visual_broadleaf_count,
        "unknown_visual_conifer_count": unknown_visual_conifer_count,
        "minimum_measured_height_m": minimum_height if instance_count else None,
        "maximum_measured_height_m": maximum_height if instance_count else None,
        "minimum_measured_crown_diameter_m": (
            minimum_diameter if instance_count else None
        ),
        "maximum_measured_crown_diameter_m": (
            maximum_diameter if instance_count else None
        ),
    }
    result = {
        "schema": TREE_INSTANCE_SCHEMA,
        "semantics": {
            "instances": "visual_tree_or_crown_candidates_not_certified_inventory",
            "unknown_type_assignment": (
                "deterministic_morphology_and_hash_visual_form_not_species"
            ),
            "grounding": "instance_origin_at_measured_ground_elevation",
            "height": "prototype_top_scaled_to_measured_height",
            "crown_diameter": (
                "area_preserving_xy_anisotropy_from_measured_equivalent_diameter"
            ),
        },
        "coordinate_system": {
            "axis_convention": "local_x_east_y_north_z_up",
            "linear_unit": "metre",
            "origin": [origin_x, origin_y, origin_z],
        },
        "config": {
            "profile": active_config.profile,
            "seed": active_config.seed,
            "broadleaf_variant_count": active_config.broadleaf_variant_count,
            "conifer_variant_count": active_config.conifer_variant_count,
            "unknown_visual_conifer_fraction": (
                active_config.unknown_visual_conifer_fraction
            ),
            "maximum_crown_anisotropy": active_config.maximum_crown_anisotropy,
        },
        "material_slots": list(_MATERIAL_NAMES),
        "prototypes": prototypes,
        "attributes": attributes,
        "statistics": statistics,
    }
    json.dumps(result, allow_nan=False)
    return result


def decode_instance_attributes(instance_set: Mapping[str, Any]) -> dict[str, array]:
    """Validate the instance contract and decode all point attributes."""

    if instance_set.get("schema") != TREE_INSTANCE_SCHEMA:
        raise ValueError(
            f"unsupported tree instance schema: {instance_set.get('schema')!r}"
        )
    attributes = instance_set.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("tree instance set has no attribute mapping")
    expected_components = {
        "position_xyz_m": 3,
        "scale_xyz": 3,
        "yaw_radians": 1,
        "prototype_index": 1,
    }
    decoded: dict[str, array] = {}
    expected_count: int | None = None
    for name, components in expected_components.items():
        try:
            encoded = attributes[name]
        except KeyError as error:
            raise ValueError(
                f"tree instance set is missing attribute {name!r}"
            ) from error
        if encoded.get("components") != components:
            raise ValueError(f"tree instance attribute {name!r} has invalid components")
        if expected_count is None:
            expected_count = int(encoded["count"])
        elif encoded.get("count") != expected_count:
            raise ValueError("tree instance attribute counts do not match")
        decoded[name] = decode_numeric_buffer(encoded)
    expected_count = expected_count or 0
    declared_count = instance_set.get("statistics", {}).get("instance_count")
    if declared_count != expected_count:
        raise ValueError(
            "tree instance count mismatch: "
            f"attributes={expected_count}, statistics={declared_count}"
        )
    prototype_count = len(instance_set.get("prototypes", ()))
    if any(index >= prototype_count for index in decoded["prototype_index"]):
        raise ValueError("tree instance references an unknown prototype")
    return decoded


def _srgb_channel_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _linear_color(color: Sequence[float]) -> tuple[float, float, float, float]:
    return (
        _srgb_channel_to_linear(float(color[0])),
        _srgb_channel_to_linear(float(color[1])),
        _srgb_channel_to_linear(float(color[2])),
        float(color[3]),
    )


def _blender_material(
    bpy: Any,
    name: str,
    dark_srgb: tuple[float, float, float, float],
    light_srgb: tuple[float, float, float, float],
    *,
    noise_scale: float,
) -> Any:
    material = bpy.data.materials.new(name)
    material.diffuse_color = _linear_color(dark_srgb)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    if principled is None:
        return material
    principled.inputs["Roughness"].default_value = 0.92
    coordinates = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.68
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.24
    ramp.color_ramp.elements[0].color = _linear_color(dark_srgb)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = _linear_color(light_srgb)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.20
    bump.inputs["Distance"].default_value = 0.08
    links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return material


def _create_blender_materials(bpy: Any, prefix: str) -> dict[str, Any]:
    return {
        "trunk": _blender_material(
            bpy,
            f"{prefix}_Trunk",
            (0.11, 0.055, 0.022, 1.0),
            (0.30, 0.15, 0.055, 1.0),
            noise_scale=7.0,
        ),
        "foliage_dark": _blender_material(
            bpy,
            f"{prefix}_FoliageDark",
            (0.025, 0.10, 0.018, 1.0),
            (0.10, 0.28, 0.045, 1.0),
            noise_scale=3.2,
        ),
        "foliage_mid": _blender_material(
            bpy,
            f"{prefix}_FoliageMid",
            (0.035, 0.14, 0.025, 1.0),
            (0.16, 0.38, 0.065, 1.0),
            noise_scale=3.8,
        ),
        "foliage_light": _blender_material(
            bpy,
            f"{prefix}_FoliageLight",
            (0.055, 0.17, 0.030, 1.0),
            (0.23, 0.43, 0.085, 1.0),
            noise_scale=4.4,
        ),
    }


def _new_geometry_socket(node_group: Any, name: str, in_out: str) -> None:
    node_group.interface.new_socket(
        name=name,
        in_out=in_out,
        socket_type="NodeSocketGeometry",
    )


def build_blender_tree_system(
    bpy: Any,
    instance_set: Mapping[str, Any],
    target_collection: Any,
    *,
    name: str = "VegetationTrees",
    materials: Mapping[str, Any] | None = None,
) -> Any:
    """Create one Blender 5.2 point instancer with shared tree prototypes.

    No ``Realize Instances`` node is created.  The returned object is the only
    scene-linked object added by this function.  Prototype objects live in an
    unlinked data collection referenced by the Geometry Nodes graph.
    """

    decoded = decode_instance_attributes(instance_set)
    positions = decoded["position_xyz_m"]
    scales = decoded["scale_xyz"]
    yaws = decoded["yaw_radians"]
    prototype_indices = decoded["prototype_index"]
    instance_count = len(prototype_indices)
    material_names = instance_set.get("material_slots")
    if material_names != list(_MATERIAL_NAMES):
        raise ValueError("tree material slot contract is invalid")
    active_materials = dict(materials or _create_blender_materials(bpy, f"MAT_{name}"))
    if any(material_name not in active_materials for material_name in _MATERIAL_NAMES):
        raise ValueError(
            "materials mapping does not satisfy the tree material contract"
        )

    prototype_collection = bpy.data.collections.new(f"_{name}_Prototypes")
    prototype_objects: list[Any] = []
    for prototype_index, prototype in enumerate(instance_set["prototypes"]):
        mesh_spec = prototype["mesh"]
        mesh = bpy.data.meshes.new(f"{name}Prototype{prototype_index:03d}Mesh")
        mesh.from_pydata(mesh_spec["vertices"], [], mesh_spec["faces"])
        for material_name in _MATERIAL_NAMES:
            mesh.materials.append(active_materials[material_name])
        for polygon, material_index, smooth in zip(
            mesh.polygons,
            mesh_spec["material_indices"],
            mesh_spec["smooth_faces"],
            strict=True,
        ):
            polygon.material_index = int(material_index)
            polygon.use_smooth = bool(smooth)
        mesh.update()
        prototype_object = bpy.data.objects.new(
            f"{name}Prototype{prototype_index:03d}_{prototype['name']}", mesh
        )
        prototype_object["visual_form"] = prototype["visual_form"]
        prototype_collection.objects.link(prototype_object)
        prototype_objects.append(prototype_object)

    point_mesh = bpy.data.meshes.new(f"{name}PointMesh")
    point_mesh.vertices.add(instance_count)
    if instance_count:
        point_mesh.vertices.foreach_set("co", positions)
    scale_attribute = point_mesh.attributes.new(
        name="tree_scale", type="FLOAT_VECTOR", domain="POINT"
    )
    if instance_count:
        scale_attribute.data.foreach_set("vector", scales)
    rotations = array("f", (0.0 for _ in range(instance_count * 3)))
    for index, yaw in enumerate(yaws):
        rotations[index * 3 + 2] = yaw
    rotation_attribute = point_mesh.attributes.new(
        name="tree_rotation", type="FLOAT_VECTOR", domain="POINT"
    )
    if instance_count:
        rotation_attribute.data.foreach_set("vector", rotations)
    prototype_attribute = point_mesh.attributes.new(
        name="tree_prototype_index", type="INT", domain="POINT"
    )
    if instance_count:
        prototype_attribute.data.foreach_set("value", array("i", prototype_indices))
    point_mesh.update()

    point_object = bpy.data.objects.new(name, point_mesh)
    target_collection.objects.link(point_object)
    node_group = bpy.data.node_groups.new(f"{name}GeometryNodes", "GeometryNodeTree")
    _new_geometry_socket(node_group, "Geometry", "INPUT")
    _new_geometry_socket(node_group, "Geometry", "OUTPUT")
    nodes = node_group.nodes
    links = node_group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    collection_info = nodes.new("GeometryNodeCollectionInfo")
    collection_info.inputs["Collection"].default_value = prototype_collection
    collection_info.inputs["Separate Children"].default_value = True
    collection_info.inputs["Reset Children"].default_value = False
    prototype_index_node = nodes.new("GeometryNodeInputNamedAttribute")
    prototype_index_node.data_type = "INT"
    prototype_index_node.inputs["Name"].default_value = "tree_prototype_index"
    rotation_node = nodes.new("GeometryNodeInputNamedAttribute")
    rotation_node.data_type = "FLOAT_VECTOR"
    rotation_node.inputs["Name"].default_value = "tree_rotation"
    scale_node = nodes.new("GeometryNodeInputNamedAttribute")
    scale_node.data_type = "FLOAT_VECTOR"
    scale_node.inputs["Name"].default_value = "tree_scale"
    instance_node = nodes.new("GeometryNodeInstanceOnPoints")
    instance_node.inputs["Selection"].default_value = True
    instance_node.inputs["Pick Instance"].default_value = True
    links.new(group_input.outputs["Geometry"], instance_node.inputs["Points"])
    links.new(collection_info.outputs["Instances"], instance_node.inputs["Instance"])
    links.new(
        prototype_index_node.outputs["Attribute"],
        instance_node.inputs["Instance Index"],
    )
    links.new(rotation_node.outputs["Attribute"], instance_node.inputs["Rotation"])
    links.new(scale_node.outputs["Attribute"], instance_node.inputs["Scale"])
    links.new(instance_node.outputs["Instances"], group_output.inputs["Geometry"])
    modifier = point_object.modifiers.new(name="SharedTreeInstances", type="NODES")
    modifier.node_group = node_group

    point_object["tree_instance_schema"] = TREE_INSTANCE_SCHEMA
    point_object["tree_instance_count"] = instance_count
    point_object["tree_prototype_count"] = len(prototype_objects)
    point_object["instances_realized"] = False
    point_object["source_semantics"] = instance_set["semantics"]["instances"]
    return point_object
