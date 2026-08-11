"""Render and gate the FireViewer ground atlas without auto-acceptance.

This module intentionally keeps rendering and acceptance as two independent
operations:

* ``render`` must run in a locked Blender binary.  It produces physical PBR
  renders and a receipt whose status is *pending* human visual review.
* ``accept`` runs only after a separate, hash-bound human review covers every
  rendered profile/scale cell and every diagnostic image without a rejection.

The tool never promotes a render by itself.  All persisted artifact paths are
portable, bounded paths relative to the render directory; all production
inputs, outputs, Blender state, caches and temporary directories must be on
``D:``.
"""

from __future__ import annotations

import argparse
from array import array
from collections.abc import Mapping, Sequence
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys
from typing import Any


CATALOG_SCHEMA = "fireviewer.ground-surface-atlas-library.v3"
PENDING_SCHEMA = "fireviewer.ground-atlas-blender-render.v1"
HUMAN_REVIEW_SCHEMA = "fireviewer.ground-atlas-human-visual-review.v1"
ACCEPTANCE_SCHEMA = "fireviewer.ground-atlas-visual-acceptance.v1"
FAILURE_SCHEMA = "fireviewer.ground-atlas-blender-render-failure.v1"

REQUIRED_TEXTURE_ROLES = ("basecolor", "normal", "height", "orm")
SCALE_BANDS: dict[str, tuple[float, ...]] = {
    "micro": (1.5, 3.0, 4.5, 6.0),
    "meso": (16.0, 32.0, 48.0, 64.0),
    "macro": (128.0, 256.0, 384.0, 512.0),
}
DISTANT_FAMILIES = (
    "agriculture_field",
    "road_surface",
    "path_surface",
    "watercourse",
    "railway_bed",
    "cliff_surface",
)
REQUIRED_REVIEW_CHECKS = (
    "runtime_atlas_channels_readable",
    "all_216_profile_scale_cells_reviewed",
    "atlas_cells_and_gutters_seam_free",
    "basecolor_normal_height_orm_response_coherent",
    "micro_meso_macro_scale_behavior_coherent",
    "no_obvious_tiling_or_close_source_image_artifact",
    "surface_families_semantically_coherent",
    "distant_fields_roads_paths_water_rail_cliffs_coherent",
)
REQUIRED_D_ENVIRONMENT = (
    "TEMP",
    "TMP",
    "PYTHONPYCACHEPREFIX",
    "BLENDER_USER_CONFIG",
    "BLENDER_USER_SCRIPTS",
    "BLENDER_USER_EXTENSIONS",
)


class AtlasAcceptanceError(RuntimeError):
    """Raised when an atlas proof is absent, changed, or incomplete."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AtlasAcceptanceError(f"{label} is not a lowercase SHA-256")
    return value


def _require_d_path(path: Path, label: str, *, must_exist: bool = False) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise AtlasAcceptanceError(f"{label} must stay on D: ({resolved})")
    if must_exist and not resolved.exists():
        raise AtlasAcceptanceError(f"{label} is absent: {resolved}")
    return resolved


def _require_d_environment() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in REQUIRED_D_ENVIRONMENT:
        raw = os.environ.get(name)
        if not raw:
            raise AtlasAcceptanceError(f"{name} must be explicitly redirected to D:")
        _require_d_path(Path(raw), name)
        result[name] = "D:"
    return result


def _bounded_relative_path(root: Path, raw: Any, label: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw:
        raise AtlasAcceptanceError(f"{label} path is absent")
    if "\\" in raw:
        raise AtlasAcceptanceError(f"{label} path must use portable '/' separators")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise AtlasAcceptanceError(f"{label} path is not a bounded relative path")
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise AtlasAcceptanceError(f"{label} path escapes its artifact root")
    return relative.as_posix(), target


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sealed_payload(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = copy.deepcopy(dict(payload))
    sealed[field] = canonical_sha256(sealed)
    return sealed


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    expected = _require_sha256(payload.get(field), f"{label}.{field}")
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop(field, None)
    if canonical_sha256(unsigned) != expected:
        raise AtlasAcceptanceError(f"{label} content hash mismatch")


def load_catalog(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog_path = _require_d_path(path, "atlas catalog", must_exist=True)
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AtlasAcceptanceError(f"Invalid atlas catalog: {error}") from error
    if not isinstance(catalog, dict) or catalog.get("schema") != CATALOG_SCHEMA:
        raise AtlasAcceptanceError("Unsupported ground atlas catalog")

    unsigned = copy.deepcopy(catalog)
    declared_hash = unsigned.pop("catalog_sha256", None)
    _require_sha256(declared_hash, "catalog.catalog_sha256")
    if canonical_sha256(unsigned) != declared_hash:
        raise AtlasAcceptanceError("Atlas catalog canonical hash mismatch")

    profiles = catalog.get("profiles")
    if (
        catalog.get("profile_count") != 72
        or not isinstance(profiles, list)
        or len(profiles) != 72
    ):
        raise AtlasAcceptanceError("Atlas catalog must contain exactly 72 profiles")
    profile_ids = [
        profile.get("id") for profile in profiles if isinstance(profile, dict)
    ]
    if (
        len(profile_ids) != 72
        or any(
            not isinstance(profile_id, str) or not profile_id
            for profile_id in profile_ids
        )
        or len(set(profile_ids)) != 72
    ):
        raise AtlasAcceptanceError("Atlas profile identifiers are absent or duplicated")

    if catalog.get("runtime_texture_count") != 4:
        raise AtlasAcceptanceError(
            "Atlas catalog must contain exactly four runtime atlases"
        )
    runtime_atlas = catalog.get("runtime_atlas")
    assets = runtime_atlas.get("assets") if isinstance(runtime_atlas, dict) else None
    if not isinstance(assets, dict) or set(assets) != set(REQUIRED_TEXTURE_ROLES):
        raise AtlasAcceptanceError(
            "Runtime atlas roles must be basecolor, normal, height and ORM"
        )

    root = catalog_path.parent.resolve()
    asset_locks: dict[str, dict[str, Any]] = {}
    for role in REQUIRED_TEXTURE_ROLES:
        record = assets[role]
        if not isinstance(record, dict):
            raise AtlasAcceptanceError(f"Runtime atlas record is invalid: {role}")
        relative, target = _bounded_relative_path(
            root, record.get("path"), f"atlas {role}"
        )
        _require_d_path(target, f"atlas {role}", must_exist=True)
        expected_hash = _require_sha256(record.get("sha256"), f"atlas {role}.sha256")
        if not target.is_file() or sha256_file(target) != expected_hash:
            raise AtlasAcceptanceError(f"Runtime atlas changed or is absent: {role}")
        if record.get("byte_count") != target.stat().st_size:
            raise AtlasAcceptanceError(f"Runtime atlas byte count mismatch: {role}")
        width = record.get("width")
        height = record.get("height")
        if width != 4096 or height != 4096:
            raise AtlasAcceptanceError(f"Runtime atlas dimensions are invalid: {role}")
        asset_locks[role] = {
            "path": relative,
            "sha256": expected_hash,
            "byte_count": target.stat().st_size,
            "width": width,
            "height": height,
        }

    sources = catalog.get("micro_sources")
    if not isinstance(sources, list):
        raise AtlasAcceptanceError("Atlas micro source table is absent")
    sources_by_id = {
        source.get("id"): source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    }
    for profile in profiles:
        if not isinstance(profile, dict):
            raise AtlasAcceptanceError("Atlas profile record is invalid")
        if profile.get("family") not in {
            "natural_ground",
            "burned_ground",
            *DISTANT_FAMILIES,
        }:
            raise AtlasAcceptanceError(
                f"Unknown atlas profile family: {profile.get('id')}"
            )
        basis = profile.get("surface_basis")
        source_id = profile.get("micro_source_id")
        if basis == "procedural_only":
            if source_id is not None:
                raise AtlasAcceptanceError(
                    f"Procedural profile has a micro source: {profile.get('id')}"
                )
        elif source_id not in sources_by_id:
            raise AtlasAcceptanceError(
                f"Atlas profile source is absent: {profile.get('id')}"
            )

    evidence = {
        "file_sha256": sha256_file(catalog_path),
        "declared_sha256": declared_hash,
        "file_name": catalog_path.name,
        "runtime_atlas": asset_locks,
    }
    return catalog, evidence


def build_render_matrix(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    profiles = catalog["profiles"]
    matrix: list[dict[str, Any]] = []
    for band, spans in SCALE_BANDS.items():
        for profile_index, profile in enumerate(profiles):
            row, column = divmod(profile_index, 9)
            span = spans[profile_index % len(spans)]
            matrix.append(
                {
                    "cell_id": f"{band}:{profile_index:02d}:{profile['id']}",
                    "band": band,
                    "profile_index": profile_index,
                    "profile_id": profile["id"],
                    "family": profile["family"],
                    "application_mode": profile["application_mode"],
                    "row": row,
                    "column": column,
                    "physical_span_m": span,
                    "render_key": f"profiles_{band}",
                }
            )
    if len(matrix) != 216 or len({item["cell_id"] for item in matrix}) != 216:
        raise AtlasAcceptanceError(
            "Profile/scale matrix is not exactly 216 unique cells"
        )
    return matrix


def expected_render_keys() -> tuple[str, ...]:
    return (
        *(f"runtime_{role}" for role in REQUIRED_TEXTURE_ROLES),
        *(f"profiles_{band}" for band in SCALE_BANDS),
        *(f"distant_{family}" for family in DISTANT_FAMILIES),
    )


def _require_blender() -> Any:
    try:
        import bpy  # type: ignore
    except ImportError as error:
        raise AtlasAcceptanceError(
            "The render phase must run inside Blender"
        ) from error
    return bpy


def _blender_lock(bpy: Any) -> dict[str, Any]:
    executable = _require_d_path(
        Path(bpy.app.binary_path), "Blender executable", must_exist=True
    )
    version = str(bpy.app.version_string).strip()
    if not version or tuple(bpy.app.version) < (4, 5, 0):
        raise AtlasAcceptanceError("Blender 4.5 LTS or newer is required")
    return {
        "version": version,
        "version_tuple": list(bpy.app.version),
        "binary_filename": executable.name,
        "binary_sha256": sha256_file(executable),
        "render_engine": "BLENDER_EEVEE_NEXT",
    }


def _clear_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in tuple(bpy.data.materials):
        bpy.data.materials.remove(block)
    for block in tuple(bpy.data.curves):
        if block.users == 0:
            bpy.data.curves.remove(block)


def _configure_scene(bpy: Any, width: int, height: int) -> Any:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.image_settings.compression = 40
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 1.0
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.055, 0.065, 0.08, 1.0)
    background.inputs["Strength"].default_value = 0.6
    return scene


def _load_runtime_images(
    bpy: Any,
    catalog: Mapping[str, Any],
    catalog_root: Path,
) -> dict[str, Any]:
    images: dict[str, Any] = {}
    for role in REQUIRED_TEXTURE_ROLES:
        relative, target = _bounded_relative_path(
            catalog_root,
            catalog["runtime_atlas"]["assets"][role]["path"],
            f"atlas {role}",
        )
        del relative
        image = bpy.data.images.get(str(target)) or bpy.data.images.load(
            str(target), check_existing=True
        )
        image.colorspace_settings.name = "sRGB" if role == "basecolor" else "Non-Color"
        images[role] = image
    return images


def _add_camera_ortho(bpy: Any, *, scale: float) -> Any:
    bpy.ops.object.camera_add(location=(0.0, 0.0, 40.0))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = scale
    camera.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = camera
    return camera


def _point_camera(camera: Any, target: tuple[float, float, float]) -> None:
    from mathutils import Vector  # type: ignore

    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_lighting(bpy: Any, *, span: float) -> None:
    bpy.ops.object.light_add(type="AREA", location=(-span * 0.3, -span * 0.35, span))
    key = bpy.context.object
    key.data.energy = 5200.0
    key.data.shape = "DISK"
    key.data.size = span * 0.65
    bpy.ops.object.light_add(
        type="AREA", location=(span * 0.4, span * 0.25, span * 0.55)
    )
    fill = bpy.context.object
    fill.data.energy = 2100.0
    fill.data.size = span * 0.6
    bpy.ops.object.light_add(
        type="SUN",
        location=(0.0, 0.0, span),
        rotation=(math.radians(24.0), math.radians(-18.0), math.radians(-32.0)),
    )
    sun = bpy.context.object
    sun.data.energy = 2.2
    sun.data.angle = math.radians(18.0)


def _text_material(bpy: Any) -> Any:
    material = bpy.data.materials.get("AcceptanceText")
    if material is not None:
        return material
    material = bpy.data.materials.new("AcceptanceText")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.92, 0.95, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 2.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _add_text(
    bpy: Any,
    body: str,
    *,
    location: tuple[float, float, float],
    size: float,
) -> None:
    curve = bpy.data.curves.new(
        f"label-{hashlib.sha1(body.encode()).hexdigest()[:10]}", "FONT"
    )
    curve.body = body
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = size
    curve.extrude = 0.003
    obj = bpy.data.objects.new(curve.name, curve)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(_text_material(bpy))


def _source_map(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {source["id"]: source for source in catalog["micro_sources"]}


def _family_color(family: str) -> tuple[float, float, float, float]:
    colors = {
        "road_surface": (0.08, 0.085, 0.09, 1.0),
        "watercourse": (0.035, 0.12, 0.18, 1.0),
        "railway_bed": (0.16, 0.14, 0.12, 1.0),
        "path_surface": (0.22, 0.16, 0.09, 1.0),
        "cliff_surface": (0.31, 0.27, 0.22, 1.0),
        "agriculture_field": (0.23, 0.18, 0.08, 1.0),
    }
    return colors.get(family, (0.12, 0.11, 0.08, 1.0))


def _profile_material(
    bpy: Any,
    *,
    profile: Mapping[str, Any],
    source: Mapping[str, Any] | None,
    images: Mapping[str, Any],
    physical_span_m: float,
) -> Any:
    name_hash = hashlib.sha1(
        f"{profile['id']}:{physical_span_m:.3f}".encode()
    ).hexdigest()[:12]
    material = bpy.data.materials.new(f"AtlasPhysical-{name_hash}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Metallic"].default_value = 0.0
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    macro_noise = nodes.new("ShaderNodeTexNoise")
    macro_noise.inputs["Scale"].default_value = max(
        0.35,
        physical_span_m
        / max(float(profile.get("parameters", {}).get("macro_scale_m", 256.0)), 1.0),
    )
    macro_noise.inputs["Detail"].default_value = 3.0
    macro_noise.inputs["Roughness"].default_value = 0.62
    links.new(texcoord.outputs["Generated"], macro_noise.inputs["Vector"])

    if source is None:
        base = nodes.new("ShaderNodeRGB")
        base.outputs[0].default_value = _family_color(str(profile["family"]))
        variation = nodes.new("ShaderNodeMixRGB")
        variation.blend_type = "MULTIPLY"
        variation.inputs[0].default_value = 0.22
        links.new(base.outputs[0], variation.inputs[1])
        links.new(macro_noise.outputs["Color"], variation.inputs[2])
        links.new(variation.outputs[0], principled.inputs["Base Color"])
        principled.inputs["Roughness"].default_value = 0.82
        return material

    offset_u, offset_v = source["atlas_uv"]["offset"]
    scale_u, scale_v = source["atlas_uv"]["scale"]
    repeat = max(1.0, physical_span_m / float(source["physical_scale_m"]))
    repeat_node = nodes.new("ShaderNodeVectorMath")
    repeat_node.operation = "MULTIPLY"
    repeat_node.inputs[1].default_value = (repeat, repeat, 1.0)
    fraction = nodes.new("ShaderNodeVectorMath")
    fraction.operation = "FRACTION"
    cell_scale = nodes.new("ShaderNodeVectorMath")
    cell_scale.operation = "MULTIPLY"
    cell_scale.inputs[1].default_value = (scale_u, scale_v, 1.0)
    cell_offset = nodes.new("ShaderNodeVectorMath")
    cell_offset.operation = "ADD"
    cell_offset.inputs[1].default_value = (offset_u, offset_v, 0.0)
    links.new(texcoord.outputs["Generated"], repeat_node.inputs[0])
    links.new(repeat_node.outputs["Vector"], fraction.inputs[0])
    links.new(fraction.outputs["Vector"], cell_scale.inputs[0])
    links.new(cell_scale.outputs["Vector"], cell_offset.inputs[0])

    textures: dict[str, Any] = {}
    for role in REQUIRED_TEXTURE_ROLES:
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = images[role]
        texture.extension = "CLIP"
        texture.interpolation = "Linear"
        links.new(cell_offset.outputs["Vector"], texture.inputs["Vector"])
        textures[role] = texture

    gain = nodes.new("ShaderNodeVectorMath")
    gain.operation = "SCALE"
    gain.inputs[3].default_value = float(
        profile.get("parameters", {}).get("basecolor_gain", 1.0)
    )
    links.new(textures["basecolor"].outputs["Color"], gain.inputs[0])
    color_variation = nodes.new("ShaderNodeMixRGB")
    color_variation.blend_type = "MULTIPLY"
    color_variation.inputs[0].default_value = 0.12
    links.new(gain.outputs["Vector"], color_variation.inputs[1])
    links.new(macro_noise.outputs["Color"], color_variation.inputs[2])
    links.new(color_variation.outputs[0], principled.inputs["Base Color"])

    orm = nodes.new("ShaderNodeSeparateColor")
    links.new(textures["orm"].outputs["Color"], orm.inputs["Color"])
    roughness = nodes.new("ShaderNodeMath")
    roughness.operation = "ADD"
    roughness.use_clamp = True
    roughness.inputs[1].default_value = float(
        profile.get("parameters", {}).get("roughness_delta", 0.0)
    )
    links.new(orm.outputs["Green"], roughness.inputs[0])
    links.new(roughness.outputs[0], principled.inputs["Roughness"])

    normal = nodes.new("ShaderNodeNormalMap")
    normal.space = "TANGENT"
    normal.inputs["Strength"].default_value = float(
        profile.get("parameters", {}).get("normal_multiplier", 1.0)
    )
    links.new(textures["normal"].outputs["Color"], normal.inputs["Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.07
    links.new(textures["height"].outputs["Color"], bump.inputs["Height"])
    links.new(normal.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return material


def _render_file(bpy: Any, scene: Any, output: Path) -> dict[str, Any]:
    if output.exists():
        raise AtlasAcceptanceError(
            f"Refusing to overwrite an existing render: {output}"
        )
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    if not output.is_file() or output.stat().st_size < 1024:
        raise AtlasAcceptanceError(f"Blender produced an empty render: {output}")
    rendered_image = bpy.data.images.load(str(output), check_existing=False)
    try:
        width, height = rendered_image.size
        pixels = array("f", [0.0]) * len(rendered_image.pixels)
        rendered_image.pixels.foreach_get(pixels)
        total = width * height
        sample_count = min(4096, total)
        pixel_step = max(1, total // sample_count)
        luminances: list[float] = []
        for pixel_index in range(0, total, pixel_step):
            offset = pixel_index * 4
            red, green, blue = pixels[offset], pixels[offset + 1], pixels[offset + 2]
            luminances.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)
            if len(luminances) == sample_count:
                break
    finally:
        bpy.data.images.remove(rendered_image)
    metrics = {
        "sample_count": len(luminances),
        "minimum_luminance": round(min(luminances), 8),
        "maximum_luminance": round(max(luminances), 8),
        "mean_luminance": round(statistics.fmean(luminances), 8),
        "luminance_standard_deviation": round(statistics.pstdev(luminances), 8),
    }
    if metrics["maximum_luminance"] - metrics["minimum_luminance"] < 0.01:
        raise AtlasAcceptanceError(f"Render has insufficient luminance range: {output}")
    return {
        "path": output.name,
        "sha256": sha256_file(output),
        "byte_count": output.stat().st_size,
        "width_px": scene.render.resolution_x,
        "height_px": scene.render.resolution_y,
        "metrics": metrics,
    }


def _sample_image_regions(
    bpy: Any,
    output: Path,
    regions: Sequence[Mapping[str, Any]],
    *,
    samples_x: int,
    samples_y: int,
) -> list[dict[str, Any]]:
    image = bpy.data.images.load(str(output), check_existing=False)
    try:
        width, height = image.size
        pixels = array("f", [0.0]) * len(image.pixels)
        image.pixels.foreach_get(pixels)
        measured: list[dict[str, Any]] = []
        for region in regions:
            u0, v0, u1, v1 = (float(value) for value in region["bounds"])
            if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
                raise AtlasAcceptanceError(
                    f"Invalid image QA region: {region.get('id')}"
                )
            luminances: list[float] = []
            for sample_y in range(samples_y):
                v = v0 + (v1 - v0) * (sample_y + 0.5) / samples_y
                pixel_y = min(height - 1, max(0, int(v * height)))
                for sample_x in range(samples_x):
                    u = u0 + (u1 - u0) * (sample_x + 0.5) / samples_x
                    pixel_x = min(width - 1, max(0, int(u * width)))
                    offset = (pixel_y * width + pixel_x) * 4
                    red, green, blue = (
                        pixels[offset],
                        pixels[offset + 1],
                        pixels[offset + 2],
                    )
                    luminances.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)
            minimum = min(luminances)
            maximum = max(luminances)
            measured.append(
                {
                    "id": region["id"],
                    "sample_count": len(luminances),
                    "minimum_luminance": round(minimum, 8),
                    "maximum_luminance": round(maximum, 8),
                    "mean_luminance": round(statistics.fmean(luminances), 8),
                    "luminance_standard_deviation": round(
                        statistics.pstdev(luminances), 8
                    ),
                    "dynamic_range": round(maximum - minimum, 8),
                    "non_dark_fraction": round(
                        sum(value >= 0.02 for value in luminances) / len(luminances),
                        8,
                    ),
                    "bright_fraction": round(
                        sum(value >= 0.2 for value in luminances) / len(luminances),
                        8,
                    ),
                }
            )
        return measured
    finally:
        bpy.data.images.remove(image)


def _world_region(
    *,
    center_x: float,
    center_y: float,
    size_x: float,
    size_y: float,
    ortho_scale: float,
    width_px: int,
    height_px: int,
) -> tuple[float, float, float, float]:
    half_y = ortho_scale / 2.0
    half_x = half_y * width_px / height_px

    def horizontal(value: float) -> float:
        return (value + half_x) / (2.0 * half_x)

    def vertical(value: float) -> float:
        return (value + half_y) / (2.0 * half_y)

    return (
        horizontal(center_x - size_x / 2.0),
        vertical(center_y - size_y / 2.0),
        horizontal(center_x + size_x / 2.0),
        vertical(center_y + size_y / 2.0),
    )


def _render_runtime_channel(
    bpy: Any,
    *,
    role: str,
    image: Any,
    output: Path,
) -> dict[str, Any]:
    _clear_scene(bpy)
    scene = _configure_scene(bpy, 1280, 1280)
    material = bpy.data.materials.new(f"RuntimeChannel-{role}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    shader = nodes.new("ShaderNodeEmission")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    output_node = nodes.new("ShaderNodeOutputMaterial")
    links.new(texture.outputs["Color"], shader.inputs["Color"])
    links.new(shader.outputs["Emission"], output_node.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=10.0)
    bpy.context.object.data.materials.append(material)
    _add_text(bpy, role.upper(), location=(0.0, -5.35, 0.03), size=0.38)
    _add_camera_ortho(bpy, scale=11.5)
    record = _render_file(bpy, scene, output)
    record["diagnostic"] = "runtime_atlas_channel"
    record["role"] = role
    return record


def _render_profile_sheet(
    bpy: Any,
    *,
    catalog: Mapping[str, Any],
    images: Mapping[str, Any],
    matrix: Sequence[Mapping[str, Any]],
    band: str,
    output: Path,
) -> dict[str, Any]:
    _clear_scene(bpy)
    scene = _configure_scene(bpy, 3072, 2944)
    sources = _source_map(catalog)
    cells = [item for item in matrix if item["band"] == band]
    spacing_x = 3.15
    spacing_y = 3.2
    plane_size = 2.5
    for cell in cells:
        profile = catalog["profiles"][cell["profile_index"]]
        source = sources.get(profile.get("micro_source_id"))
        x = (cell["column"] - 4.0) * spacing_x
        y = (3.5 - cell["row"]) * spacing_y
        material = _profile_material(
            bpy,
            profile=profile,
            source=source,
            images=images,
            physical_span_m=float(cell["physical_span_m"]),
        )
        bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(x, y + 0.2, 0.0))
        plane = bpy.context.object
        plane.name = f"Cell-{cell['band']}-{cell['profile_index']:02d}"
        plane.data.materials.append(material)
        profile_label = str(cell["profile_id"])
        if len(profile_label) > 29:
            profile_label = profile_label[:28] + "…"
        _add_text(
            bpy,
            f"{cell['profile_index']:02d} {profile_label}\n{cell['physical_span_m']:g} m",
            location=(x, y - 1.18, 0.025),
            size=0.145,
        )
    _add_text(
        bpy,
        f"FIREVIEWER — 72 PHYSICAL PROFILES — {band.upper()}",
        location=(0.0, 13.45, 0.03),
        size=0.42,
    )
    _add_lighting(bpy, span=30.0)
    ortho_scale = 27.7
    _add_camera_ortho(bpy, scale=ortho_scale)
    record = _render_file(bpy, scene, output)
    surface_regions = []
    label_regions = []
    for cell in cells:
        x = (cell["column"] - 4.0) * spacing_x
        y = (3.5 - cell["row"]) * spacing_y
        surface_regions.append(
            {
                "id": cell["cell_id"],
                "bounds": _world_region(
                    center_x=x,
                    center_y=y + 0.2,
                    size_x=plane_size * 0.82,
                    size_y=plane_size * 0.82,
                    ortho_scale=ortho_scale,
                    width_px=scene.render.resolution_x,
                    height_px=scene.render.resolution_y,
                ),
            }
        )
        label_regions.append(
            {
                "id": cell["cell_id"],
                "bounds": _world_region(
                    center_x=x,
                    center_y=y - 1.18,
                    size_x=plane_size,
                    size_y=0.52,
                    ortho_scale=ortho_scale,
                    width_px=scene.render.resolution_x,
                    height_px=scene.render.resolution_y,
                ),
            }
        )
    cell_metrics = _sample_image_regions(
        bpy,
        output,
        surface_regions,
        samples_x=14,
        samples_y=14,
    )
    label_metrics = _sample_image_regions(
        bpy,
        output,
        label_regions,
        samples_x=64,
        samples_y=16,
    )
    invalid_cells = [
        metric["id"]
        for metric in cell_metrics
        if metric["mean_luminance"] < 0.025
        or metric["maximum_luminance"] < 0.06
        or metric["dynamic_range"] < 0.012
        or metric["non_dark_fraction"] < 0.70
    ]
    invalid_labels = [
        metric["id"]
        for metric in label_metrics
        if metric["maximum_luminance"] < 0.35 or metric["bright_fraction"] < 0.001
    ]
    if invalid_cells:
        raise AtlasAcceptanceError(
            f"{band} contact sheet contains {len(invalid_cells)} dark or flat cells: "
            + ", ".join(invalid_cells)
        )
    if invalid_labels:
        raise AtlasAcceptanceError(
            f"{band} contact sheet contains {len(invalid_labels)} illegible labels"
        )
    record.update(
        {
            "diagnostic": "physical_profile_contact_sheet",
            "band": band,
            "cell_count": len(cells),
            "cell_ids_sha256": canonical_sha256([cell["cell_id"] for cell in cells]),
            "physical_span_min_m": min(
                float(cell["physical_span_m"]) for cell in cells
            ),
            "physical_span_max_m": max(
                float(cell["physical_span_m"]) for cell in cells
            ),
            "cell_metrics": cell_metrics,
            "cell_validation": {
                "minimum_mean_luminance": 0.025,
                "minimum_maximum_luminance": 0.06,
                "minimum_dynamic_range": 0.012,
                "minimum_non_dark_fraction": 0.70,
                "invalid_cell_count": 0,
            },
            "label_metrics": label_metrics,
            "label_validation": {
                "minimum_maximum_luminance": 0.35,
                "minimum_bright_fraction": 0.001,
                "invalid_label_count": 0,
            },
        }
    )
    return record


def _render_distant_family(
    bpy: Any,
    *,
    catalog: Mapping[str, Any],
    images: Mapping[str, Any],
    family: str,
    output: Path,
) -> dict[str, Any]:
    profiles = [
        profile for profile in catalog["profiles"] if profile["family"] == family
    ]
    if not profiles:
        raise AtlasAcceptanceError(f"No representative profile for {family}")
    profile = profiles[0]
    source = _source_map(catalog).get(profile.get("micro_source_id"))
    _clear_scene(bpy)
    scene = _configure_scene(bpy, 1600, 900)
    material = _profile_material(
        bpy,
        profile=profile,
        source=source,
        images=images,
        physical_span_m=512.0,
    )
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=65, y_subdivisions=65, size=20.0)
    surface = bpy.context.object
    surface.name = f"Distant-{family}"
    surface.scale.y = 0.58
    family_phase = DISTANT_FAMILIES.index(family) * 0.37
    for vertex in surface.data.vertices:
        x, y = vertex.co.x, vertex.co.y
        vertex.co.z = 0.16 * math.sin(x * 0.35 + family_phase) + 0.09 * math.cos(
            y * 0.58 - family_phase
        )
    surface.data.materials.append(material)
    _add_lighting(bpy, span=22.0)
    bpy.ops.object.camera_add(location=(0.0, -13.5, 8.5))
    camera = bpy.context.object
    camera.data.type = "PERSP"
    camera.data.lens = 42.0
    _point_camera(camera, (0.0, 1.0, 0.0))
    scene.camera = camera
    record = _render_file(bpy, scene, output)
    surface_metric = _sample_image_regions(
        bpy,
        output,
        [{"id": family, "bounds": (0.12, 0.12, 0.88, 0.78)}],
        samples_x=64,
        samples_y=36,
    )[0]
    if (
        surface_metric["mean_luminance"] < 0.025
        or surface_metric["maximum_luminance"] < 0.06
        or surface_metric["dynamic_range"] < 0.012
        or surface_metric["non_dark_fraction"] < 0.45
    ):
        raise AtlasAcceptanceError(f"Distant {family} render is dark or empty")
    record.update(
        {
            "diagnostic": "distant_representative_surface",
            "family": family,
            "profile_id": profile["id"],
            "physical_span_m": 512.0,
            "camera": {"type": "perspective", "lens_mm": 42.0},
            "surface_metrics": surface_metric,
            "surface_validation": {
                "minimum_mean_luminance": 0.025,
                "minimum_maximum_luminance": 0.06,
                "minimum_dynamic_range": 0.012,
                "minimum_non_dark_fraction": 0.45,
                "invalid_surface_count": 0,
            },
        }
    )
    return record


def _render_all_proofs(
    bpy: Any,
    *,
    catalog: Mapping[str, Any],
    catalog_lock: Mapping[str, Any],
    blender_lock: Mapping[str, Any],
    environment: Mapping[str, str],
    matrix: Sequence[Mapping[str, Any]],
    images: Mapping[str, Any],
    staging: Path,
) -> dict[str, Any]:
    renders: dict[str, dict[str, Any]] = {}
    for role in REQUIRED_TEXTURE_ROLES:
        key = f"runtime_{role}"
        renders[key] = _render_runtime_channel(
            bpy,
            role=role,
            image=images[role],
            output=staging / f"{key}.png",
        )
    for band in SCALE_BANDS:
        key = f"profiles_{band}"
        renders[key] = _render_profile_sheet(
            bpy,
            catalog=catalog,
            images=images,
            matrix=matrix,
            band=band,
            output=staging / f"{key}.png",
        )
    for family in DISTANT_FAMILIES:
        key = f"distant_{family}"
        renders[key] = _render_distant_family(
            bpy,
            catalog=catalog,
            images=images,
            family=family,
            output=staging / f"{key}.png",
        )
    if tuple(renders) != expected_render_keys():
        raise AtlasAcceptanceError("Atlas render set is incomplete")

    payload = {
        "schema": PENDING_SCHEMA,
        "status": "rendered_pending_visual_review",
        "production_visual_gate_passed": False,
        "blender": blender_lock,
        "d_only_environment": environment,
        "catalog": {
            "file_name": catalog_lock["file_name"],
            "file_sha256": catalog_lock["file_sha256"],
            "declared_sha256": catalog_lock["declared_sha256"],
            "profile_count": 72,
            "texture_count": 4,
        },
        "runtime_atlas": catalog_lock["runtime_atlas"],
        "scale_bands": [
            {
                "id": band,
                "minimum_span_m": spans[0],
                "maximum_span_m": spans[-1],
                "sampled_spans_m": list(spans),
            }
            for band, spans in SCALE_BANDS.items()
        ],
        "matrix": matrix,
        "matrix_cell_count": 216,
        "matrix_sha256": canonical_sha256(matrix),
        "renders": renders,
        "required_human_review": {
            "schema": HUMAN_REVIEW_SCHEMA,
            "checks": list(REQUIRED_REVIEW_CHECKS),
            "cell_verdict_count": 216,
            "render_verdict_keys": list(expected_render_keys()),
            "automatic_acceptance": "forbidden",
        },
    }
    return _sealed_payload(payload, "receipt_content_sha256")


def _publish_render_failure(
    *,
    output: Path,
    staging: Path,
    error: Exception,
    blender_lock: Mapping[str, Any],
    catalog_lock: Mapping[str, Any],
) -> Path:
    expected_staging = output.with_name(f".{output.name}.rendering")
    if staging.resolve() != expected_staging.resolve():
        raise AtlasAcceptanceError(
            "Refusing to clean an unexpected render staging path"
        )
    if staging.exists():
        shutil.rmtree(staging)
    if output.exists():
        if any(output.iterdir()):
            raise AtlasAcceptanceError("Refusing to replace a non-empty QA output")
        output.rmdir()
    output.mkdir(parents=True, exist_ok=False)
    payload = _sealed_payload(
        {
            "schema": FAILURE_SCHEMA,
            "status": "failed_technical_render",
            "production_visual_gate_passed": False,
            "pending_visual_review_receipt_emitted": False,
            "visual_acceptance_receipt_emitted": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "blender": dict(blender_lock),
            "catalog": {
                "file_name": catalog_lock["file_name"],
                "file_sha256": catalog_lock["file_sha256"],
                "declared_sha256": catalog_lock["declared_sha256"],
            },
            "artifacts_retained": [],
            "staging_cleaned": True,
        },
        "failure_content_sha256",
    )
    path = output / "atlas-render.failed-technical.v1.json"
    _atomic_json(path, payload)
    return path


def render_pending(
    catalog_path: Path, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    bpy = _require_blender()
    output = _require_d_path(output_root, "atlas render output")
    if output.exists() and any(output.iterdir()):
        raise AtlasAcceptanceError(f"Atlas render output must be empty: {output}")
    staging = output.with_name(f".{output.name}.rendering")
    _require_d_path(staging, "atlas render staging")
    if staging.exists():
        raise AtlasAcceptanceError(f"Atlas render staging already exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    environment = _require_d_environment()
    catalog, catalog_lock = load_catalog(catalog_path)
    blender_lock = _blender_lock(bpy)
    matrix = build_render_matrix(catalog)
    images = _load_runtime_images(bpy, catalog, Path(catalog_path).resolve().parent)
    try:
        receipt = _render_all_proofs(
            bpy,
            catalog=catalog,
            catalog_lock=catalog_lock,
            blender_lock=blender_lock,
            environment=environment,
            matrix=matrix,
            images=images,
            staging=staging,
        )
    except Exception as error:
        failure_path = _publish_render_failure(
            output=output,
            staging=staging,
            error=error,
            blender_lock=blender_lock,
            catalog_lock=catalog_lock,
        )
        raise AtlasAcceptanceError(
            f"{error} (failure receipt: {failure_path.name})"
        ) from error
    receipt_path = staging / "atlas-render.pending-visual-review.v1.json"
    _atomic_json(receipt_path, receipt)
    if output.exists():
        output.rmdir()
    staging.replace(output)
    return output / receipt_path.name, receipt


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AtlasAcceptanceError(f"Invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise AtlasAcceptanceError(f"{label} must be a JSON object")
    return payload


def validate_pending_receipt(
    catalog_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog, catalog_lock = load_catalog(catalog_path)
    path = _require_d_path(receipt_path, "pending render receipt", must_exist=True)
    if not path.is_file():
        raise AtlasAcceptanceError("Pending render receipt is not a file")
    receipt = _load_json_object(path, "pending render receipt")
    if (
        receipt.get("schema") != PENDING_SCHEMA
        or receipt.get("status") != "rendered_pending_visual_review"
        or receipt.get("production_visual_gate_passed") is not False
    ):
        raise AtlasAcceptanceError("Render receipt is not pending human visual review")
    _validate_seal(receipt, "receipt_content_sha256", "render receipt")
    expected_catalog = {
        "file_name": catalog_lock["file_name"],
        "file_sha256": catalog_lock["file_sha256"],
        "declared_sha256": catalog_lock["declared_sha256"],
        "profile_count": 72,
        "texture_count": 4,
    }
    if receipt.get("catalog") != expected_catalog:
        raise AtlasAcceptanceError("Render receipt targets another atlas catalog")
    if receipt.get("runtime_atlas") != catalog_lock["runtime_atlas"]:
        raise AtlasAcceptanceError("Render receipt runtime atlas lock changed")

    matrix = build_render_matrix(catalog)
    if (
        receipt.get("matrix") != matrix
        or receipt.get("matrix_cell_count") != 216
        or receipt.get("matrix_sha256") != canonical_sha256(matrix)
    ):
        raise AtlasAcceptanceError("Render receipt 216-cell matrix mismatch")
    expected_bands = [
        {
            "id": band,
            "minimum_span_m": spans[0],
            "maximum_span_m": spans[-1],
            "sampled_spans_m": list(spans),
        }
        for band, spans in SCALE_BANDS.items()
    ]
    if receipt.get("scale_bands") != expected_bands:
        raise AtlasAcceptanceError("Render receipt scale bands changed")

    expected_review = {
        "schema": HUMAN_REVIEW_SCHEMA,
        "checks": list(REQUIRED_REVIEW_CHECKS),
        "cell_verdict_count": 216,
        "render_verdict_keys": list(expected_render_keys()),
        "automatic_acceptance": "forbidden",
    }
    if receipt.get("required_human_review") != expected_review:
        raise AtlasAcceptanceError("Render receipt human review contract changed")
    blender = receipt.get("blender")
    if (
        not isinstance(blender, dict)
        or not isinstance(blender.get("version"), str)
        or not blender["version"].strip()
        or blender.get("render_engine") != "BLENDER_EEVEE_NEXT"
    ):
        raise AtlasAcceptanceError("Render receipt does not lock Blender")
    _require_sha256(blender.get("binary_sha256"), "render receipt Blender binary")
    if receipt.get("d_only_environment") != {
        name: "D:" for name in REQUIRED_D_ENVIRONMENT
    }:
        raise AtlasAcceptanceError("Render receipt does not prove a D-only environment")

    renders = receipt.get("renders")
    if (
        not isinstance(renders, dict)
        or set(renders) != set(expected_render_keys())
        or len(renders) != len(expected_render_keys())
    ):
        raise AtlasAcceptanceError(
            "Render receipt proof set is incomplete or reordered"
        )
    artifact_root = path.parent.resolve()
    for key, record in renders.items():
        if not isinstance(record, dict):
            raise AtlasAcceptanceError(f"Render record is invalid: {key}")
        relative, target = _bounded_relative_path(
            artifact_root, record.get("path"), f"render {key}"
        )
        if relative != f"{key}.png" or not target.is_file():
            raise AtlasAcceptanceError(f"Render artifact is absent: {key}")
        if sha256_file(target) != record.get(
            "sha256"
        ) or target.stat().st_size != record.get("byte_count"):
            raise AtlasAcceptanceError(f"Render artifact changed: {key}")
        _require_sha256(record.get("sha256"), f"render {key}.sha256")
        if (
            not isinstance(record.get("width_px"), int)
            or record["width_px"] < 512
            or not isinstance(record.get("height_px"), int)
            or record["height_px"] < 512
            or not isinstance(record.get("metrics"), dict)
            or record["metrics"].get("sample_count", 0) < 64
        ):
            raise AtlasAcceptanceError(f"Render metrics are incomplete: {key}")
        if key.startswith("runtime_"):
            role = key.removeprefix("runtime_")
            if (
                record.get("diagnostic") != "runtime_atlas_channel"
                or record.get("role") != role
            ):
                raise AtlasAcceptanceError(f"Runtime channel proof is invalid: {key}")
        elif key.startswith("profiles_"):
            band = key.removeprefix("profiles_")
            expected_cells = [
                cell["cell_id"] for cell in matrix if cell["band"] == band
            ]
            cell_metrics = record.get("cell_metrics")
            label_metrics = record.get("label_metrics")
            if (
                record.get("diagnostic") != "physical_profile_contact_sheet"
                or record.get("band") != band
                or record.get("cell_count") != 72
                or record.get("cell_ids_sha256") != canonical_sha256(expected_cells)
                or not isinstance(cell_metrics, list)
                or [metric.get("id") for metric in cell_metrics] != expected_cells
                or not isinstance(label_metrics, list)
                or [metric.get("id") for metric in label_metrics] != expected_cells
                or record.get("cell_validation", {}).get("invalid_cell_count") != 0
                or record.get("label_validation", {}).get("invalid_label_count") != 0
            ):
                raise AtlasAcceptanceError(
                    f"Profile sheet luminance/occupancy proof is invalid: {key}"
                )
        elif key.startswith("distant_"):
            family = key.removeprefix("distant_")
            if (
                record.get("diagnostic") != "distant_representative_surface"
                or record.get("family") != family
                or record.get("surface_validation", {}).get("invalid_surface_count")
                != 0
            ):
                raise AtlasAcceptanceError(f"Distant surface proof is invalid: {key}")
    return receipt, catalog_lock


def _parse_reviewed_at(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AtlasAcceptanceError(
            "Human review reviewed_at_utc must be an ISO UTC time"
        )
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise AtlasAcceptanceError("Human review reviewed_at_utc is invalid") from error


def _review_verdict_map(
    raw: Any,
    *,
    key_field: str,
    expected: Sequence[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw, list):
        raise AtlasAcceptanceError(f"Human review {label} verdicts are absent")
    records: dict[str, Mapping[str, Any]] = {}
    for record in raw:
        if not isinstance(record, dict):
            raise AtlasAcceptanceError(f"Human review {label} verdict is invalid")
        key = record.get(key_field)
        if not isinstance(key, str) or key in records:
            raise AtlasAcceptanceError(f"Human review {label} verdict is duplicated")
        if record.get("verdict") != "accepted":
            raise AtlasAcceptanceError(f"Human review rejected {label}: {key}")
        records[key] = record
    if set(records) != set(expected) or len(records) != len(expected):
        raise AtlasAcceptanceError(f"Human review {label} coverage is incomplete")
    return records


def validate_human_review(
    review_path: Path,
    pending_path: Path,
    pending: Mapping[str, Any],
) -> dict[str, Any]:
    review_file = _require_d_path(review_path, "human review", must_exist=True)
    if review_file.parent.resolve() != pending_path.parent.resolve():
        raise AtlasAcceptanceError(
            "Human review must be stored beside its render receipt"
        )
    review = _load_json_object(review_file, "human visual review")
    if (
        review.get("schema") != HUMAN_REVIEW_SCHEMA
        or review.get("status") != "review_complete"
        or review.get("verdict") != "accepted"
    ):
        raise AtlasAcceptanceError("Human visual review is not explicitly accepted")
    _validate_seal(review, "review_content_sha256", "human review")
    if review.get("render_receipt_sha256") != sha256_file(pending_path):
        raise AtlasAcceptanceError("Human review targets another render receipt")
    if review.get("catalog_file_sha256") != pending["catalog"]["file_sha256"]:
        raise AtlasAcceptanceError("Human review targets another atlas catalog")
    if review.get("matrix_sha256") != pending["matrix_sha256"]:
        raise AtlasAcceptanceError("Human review targets another 216-cell matrix")
    reviewer = review.get("reviewer")
    if (
        not isinstance(reviewer, dict)
        or reviewer.get("kind") != "human"
        or not isinstance(reviewer.get("id"), str)
        or not reviewer["id"].strip()
    ):
        raise AtlasAcceptanceError("Human review must identify a human reviewer")
    _parse_reviewed_at(review.get("reviewed_at_utc"))
    if review.get("rejections") != []:
        raise AtlasAcceptanceError("Human review contains one or more rejections")

    checks = _review_verdict_map(
        review.get("checks"),
        key_field="id",
        expected=REQUIRED_REVIEW_CHECKS,
        label="check",
    )
    cells = _review_verdict_map(
        review.get("cells"),
        key_field="cell_id",
        expected=[cell["cell_id"] for cell in pending["matrix"]],
        label="cell",
    )
    renders = _review_verdict_map(
        review.get("renders"),
        key_field="key",
        expected=expected_render_keys(),
        label="render",
    )
    for key, record in renders.items():
        if record.get("sha256") != pending["renders"][key]["sha256"]:
            raise AtlasAcceptanceError(f"Human review render hash mismatch: {key}")
    if len(checks) != len(REQUIRED_REVIEW_CHECKS) or len(cells) != 216:
        raise AtlasAcceptanceError("Human visual review is not exhaustive")
    return review


def _acceptance_payload(
    *,
    pending_file: Path,
    pending: Mapping[str, Any],
    review_file: Path,
    review: Mapping[str, Any],
    catalog_lock: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "accepted_blender_visual",
        "atlas_catalog_sha256": catalog_lock["file_sha256"],
        "atlas_catalog_declared_sha256": catalog_lock["declared_sha256"],
        "runtime_atlas_sha256": {
            role: catalog_lock["runtime_atlas"][role]["sha256"]
            for role in REQUIRED_TEXTURE_ROLES
        },
        "profile_count": 72,
        "texture_count": 4,
        "scale_bands": list(SCALE_BANDS),
        "invalid_profile_count": 0,
        "reviewed_cell_count": 216,
        "reviewed_render_count": len(expected_render_keys()),
        "rejection_count": 0,
        "matrix_sha256": pending["matrix_sha256"],
        "render_receipt": {
            "path": pending_file.name,
            "sha256": sha256_file(pending_file),
            "content_sha256": pending["receipt_content_sha256"],
        },
        "human_review": {
            "path": review_file.name,
            "sha256": sha256_file(review_file),
            "content_sha256": review["review_content_sha256"],
            "reviewer": review["reviewer"],
            "reviewed_at_utc": review["reviewed_at_utc"],
        },
        "blender": pending["blender"],
        "renders": {
            key: {
                "path": record["path"],
                "sha256": record["sha256"],
            }
            for key, record in pending["renders"].items()
        },
    }


def accept_review(
    catalog_path: Path,
    pending_path: Path,
    human_review_path: Path,
    acceptance_path: Path,
) -> tuple[Path, dict[str, Any]]:
    pending_file = _require_d_path(
        pending_path, "pending render receipt", must_exist=True
    )
    target = _require_d_path(acceptance_path, "atlas acceptance receipt")
    if target.parent.resolve() != pending_file.parent.resolve():
        raise AtlasAcceptanceError(
            "Acceptance receipt must stay beside its render proofs"
        )
    if target.exists():
        raise AtlasAcceptanceError(f"Refusing to overwrite atlas acceptance: {target}")

    pending, catalog_lock = validate_pending_receipt(catalog_path, pending_file)
    review = validate_human_review(
        human_review_path,
        pending_file,
        pending,
    )
    payload = _acceptance_payload(
        pending_file=pending_file,
        pending=pending,
        review_file=Path(human_review_path),
        review=review,
        catalog_lock=catalog_lock,
    )
    acceptance = _sealed_payload(payload, "acceptance_content_sha256")
    _atomic_json(target, acceptance)
    return target, acceptance


def validate_acceptance(
    catalog_path: Path,
    acceptance_path: Path,
) -> dict[str, Any]:
    """Read-only validation of the complete accepted-atlas proof chain."""

    target = _require_d_path(
        acceptance_path,
        "atlas acceptance receipt",
        must_exist=True,
    )
    if not target.is_file():
        raise AtlasAcceptanceError("Atlas acceptance receipt is not a file")
    acceptance = _load_json_object(target, "atlas acceptance receipt")
    if (
        acceptance.get("schema") != ACCEPTANCE_SCHEMA
        or acceptance.get("status") != "accepted_blender_visual"
    ):
        raise AtlasAcceptanceError("Atlas acceptance receipt is not accepted")
    _validate_seal(acceptance, "acceptance_content_sha256", "atlas acceptance")

    root = target.parent.resolve()
    render_record = acceptance.get("render_receipt")
    review_record = acceptance.get("human_review")
    if not isinstance(render_record, dict) or not isinstance(review_record, dict):
        raise AtlasAcceptanceError("Atlas acceptance proof references are absent")
    _, pending_file = _bounded_relative_path(
        root,
        render_record.get("path"),
        "accepted render receipt",
    )
    _, review_file = _bounded_relative_path(
        root,
        review_record.get("path"),
        "accepted human review",
    )
    if not pending_file.is_file() or not review_file.is_file():
        raise AtlasAcceptanceError("An accepted atlas proof file is absent or moved")
    if sha256_file(pending_file) != render_record.get("sha256"):
        raise AtlasAcceptanceError("Accepted render receipt hash mismatch")
    if sha256_file(review_file) != review_record.get("sha256"):
        raise AtlasAcceptanceError("Accepted human review hash mismatch")

    pending, catalog_lock = validate_pending_receipt(catalog_path, pending_file)
    review = validate_human_review(review_file, pending_file, pending)
    expected = _sealed_payload(
        _acceptance_payload(
            pending_file=pending_file,
            pending=pending,
            review_file=review_file,
            review=review,
            catalog_lock=catalog_lock,
        ),
        "acceptance_content_sha256",
    )
    if acceptance != expected:
        raise AtlasAcceptanceError(
            "Atlas acceptance fields do not match the proof chain"
        )
    return acceptance


def _script_argv(arguments: Sequence[str] | None) -> list[str]:
    if arguments is not None:
        return list(arguments)
    values = list(sys.argv[1:])
    return values[values.index("--") + 1 :] if "--" in values else values


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render or accept the FireViewer 72-profile ground atlas",
    )
    parser.add_argument("--phase", choices=("render", "accept"), required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render-receipt", type=Path)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--acceptance-receipt", type=Path)
    parsed = parser.parse_args(_script_argv(arguments))
    if parsed.phase == "render":
        if parsed.output is None:
            parser.error("render requires --output")
        if any(
            value is not None
            for value in (
                parsed.render_receipt,
                parsed.human_review,
                parsed.acceptance_receipt,
            )
        ):
            parser.error("render accepts only --catalog and --output")
    else:
        if parsed.output is not None:
            parser.error("accept does not use --output")
        if any(
            value is None
            for value in (
                parsed.render_receipt,
                parsed.human_review,
                parsed.acceptance_receipt,
            )
        ):
            parser.error(
                "accept requires --render-receipt, --human-review and --acceptance-receipt"
            )
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parse_arguments(arguments)
    if parsed.phase == "render":
        path, payload = render_pending(parsed.catalog, parsed.output)
    else:
        path, payload = accept_review(
            parsed.catalog,
            parsed.render_receipt,
            parsed.human_review,
            parsed.acceptance_receipt,
        )
    print(
        json.dumps(
            {
                "path": str(path),
                "schema": payload["schema"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AtlasAcceptanceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
