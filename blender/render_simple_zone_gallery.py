"""Render and revalidate the 20-image visual gallery of one measured zone.

The module is standard-library only until ``render_gallery`` imports Blender.
It also saves a packed ``zone.blend`` so the downloaded result can be opened
locally without resolving any external USD or texture dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

SCHEMA = "fireviewer.simple-zone-gallery-receipt.v1"
STATUS = "rendered_pending_human_review"
CAPTURE_COUNT = 20
RESOLUTION = 512
GALLERY_DIRECTORY = Path("qa") / "gallery"
RECEIPT_PATH = Path("qa") / "zone-gallery-receipt.v1.json"
BLEND_NAME = "zone.blend"
RENDER_EXPOSURE = 0.0
SUN_ENERGY = 2.0
BUILDING_FOCUS_CAPTURE_ID = "03-overview-oblique"
BUILDING_FOCUS_DISTANCE_M = 55.0
BUILDING_FOCUS_ELEVATION_M = 28.0
RENDER_POLICY = {
    "exposure": RENDER_EXPOSURE,
    "sun_energy": SUN_ENERGY,
    "view_look": "Medium High Contrast",
    "ground_texture_resolution_m": 1,
    "building_focus_required_when_present": True,
}


class SimpleZoneGalleryError(RuntimeError):
    """The unified scene or its visual evidence is incomplete or corrupted."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimpleZoneGalleryError(f"Invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise SimpleZoneGalleryError(f"{label} must be a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _require_root(value: Path | str, *, exists: bool = True) -> Path:
    lexical = PureWindowsPath(str(value))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise SimpleZoneGalleryError("Gallery output must remain on D: under Windows")
    root = Path(value).resolve(strict=exists)
    if os.name == "nt" and root.drive.upper() != "D:":
        raise SimpleZoneGalleryError("Gallery output must remain on D: under Windows")
    return root


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise SimpleZoneGalleryError(
            "Gallery artifact escapes the zone package"
        ) from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SimpleZoneGalleryError(f"Missing gallery artifact: {resolved}")
    return {
        "path": relative,
        "byte_count": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _validate_artifact(record: Any, root: Path, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "byte_count",
        "sha256",
    }:
        raise SimpleZoneGalleryError(f"Invalid {label} artifact record")
    relative = record.get("path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise SimpleZoneGalleryError(f"Invalid {label} path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SimpleZoneGalleryError(f"{label} escapes the zone package") from error
    if (
        not path.is_file()
        or path.stat().st_size != record.get("byte_count")
        or _sha256_file(path) != record.get("sha256")
    ):
        raise SimpleZoneGalleryError(f"{label} is missing or corrupted")
    return path


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if (
        len(header) != 24
        or not header.startswith(b"\x89PNG\r\n\x1a\n")
        or header[12:16] != b"IHDR"
    ):
        raise SimpleZoneGalleryError(f"Capture is not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def build_capture_plan(width_m: float, height_m: float) -> list[dict[str, Any]]:
    """Return four zone overviews and a complete deterministic 4x4 coverage."""

    if not math.isfinite(width_m) or not math.isfinite(height_m):
        raise SimpleZoneGalleryError("Zone dimensions must be finite")
    if width_m <= 0 or height_m <= 0:
        raise SimpleZoneGalleryError("Zone dimensions must be positive")
    center = [width_m / 2.0, height_m / 2.0]
    maximum = max(width_m, height_m)
    captures: list[dict[str, Any]] = [
        {
            "capture_id": "00-overview-topdown",
            "category": "overview",
            "projection": "orthographic",
            "center_xy_m": center,
            "frame_size_m": maximum * 1.08,
        }
    ]
    for index, azimuth in enumerate((225.0, 45.0, 135.0), start=1):
        captures.append(
            {
                "capture_id": f"{index:02d}-overview-oblique",
                "category": "overview",
                "projection": "perspective",
                "center_xy_m": center,
                "azimuth_degrees": azimuth,
                "lens_mm": 50.0,
            }
        )
    cell_width = width_m / 4.0
    cell_height = height_m / 4.0
    frame_size = max(cell_width, cell_height) * 1.04
    capture_index = 4
    for row in range(4):
        for column in range(4):
            captures.append(
                {
                    "capture_id": f"{capture_index:02d}-detail-r{row + 1}-c{column + 1}",
                    "category": "detail_coverage",
                    "projection": "orthographic",
                    "grid_cell": [row, column],
                    "center_xy_m": [
                        (column + 0.5) * cell_width,
                        height_m - (row + 0.5) * cell_height,
                    ],
                    "frame_size_m": frame_size,
                }
            )
            capture_index += 1
    if len(captures) != CAPTURE_COUNT:
        raise AssertionError("The gallery plan must contain exactly 20 captures")
    return captures


def _with_building_focus(
    capture_plan: Sequence[Mapping[str, Any]],
    samples: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    """Replace one redundant oblique overview with a measured-building proof."""

    plan = [dict(capture) for capture in capture_plan]
    if not samples:
        return plan
    normalized: list[tuple[float, float, float, float]] = []
    for sample in samples:
        if len(sample) != 4:
            raise SimpleZoneGalleryError("Building focus sample must be x/y/z/score")
        values = tuple(float(value) for value in sample)
        if any(not math.isfinite(value) for value in values) or values[3] < 0.0:
            raise SimpleZoneGalleryError("Building focus sample is invalid")
        normalized.append(values)
    selected = min(
        normalized,
        key=lambda value: (-value[3], value[0], value[1], value[2]),
    )
    for index, capture in enumerate(plan):
        if capture.get("capture_id") != BUILDING_FOCUS_CAPTURE_ID:
            continue
        plan[index] = {
            **capture,
            "category": "building_detail",
            "center_xy_m": [selected[0], selected[1]],
            "target_z_m": selected[2] + 6.0,
            "distance_m": BUILDING_FOCUS_DISTANCE_M,
            "elevation_m": BUILDING_FOCUS_ELEVATION_M,
            "lens_mm": 55.0,
        }
        return plan
    raise SimpleZoneGalleryError("Building focus capture is absent from the plan")


def _building_focus_samples(bpy: Any) -> list[tuple[float, float, float, float]]:
    """Read measured building points and choose size only as a focus priority."""

    from mathutils import Vector  # type: ignore

    samples: list[tuple[float, float, float, float]] = []
    for obj in bpy.context.scene.objects:
        if obj.type != "POINTCLOUD" or not obj.name.startswith("Buildings"):
            continue
        position_attribute = obj.data.attributes.get("position")
        scale_attribute = obj.data.attributes.get("scale")
        for index, point in enumerate(obj.data.points):
            if position_attribute is not None:
                raw_position = position_attribute.data[index].vector
            elif hasattr(point, "co"):
                raw_position = point.co
            else:
                raise SimpleZoneGalleryError(
                    "Building instances lack a measurable position"
                )
            world = obj.matrix_world @ Vector(tuple(float(v) for v in raw_position))
            if scale_attribute is None:
                score = 0.0
            else:
                scale = tuple(
                    abs(float(value)) for value in scale_attribute.data[index].vector
                )
                score = scale[0] * scale[1] * scale[2]
            samples.append((float(world[0]), float(world[1]), float(world[2]), score))
    return samples


def _look_at(camera: Any, target: Sequence[float]) -> None:
    from mathutils import Vector  # type: ignore

    direction = Vector(target) - camera.location
    if direction.length <= 1e-6:
        raise SimpleZoneGalleryError("Camera cannot be placed on its target")
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _scene_bounds(bpy: Any) -> tuple[list[float], list[float]]:
    from mathutils import Vector  # type: ignore

    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for obj in bpy.context.scene.objects:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"}:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], float(world[axis]))
                maximum[axis] = max(maximum[axis], float(world[axis]))
    if any(not math.isfinite(value) for value in (*minimum, *maximum)):
        raise SimpleZoneGalleryError("Imported zone contains no renderable bounds")
    return minimum, maximum


def _configure_scene(bpy: Any) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.view_settings.exposure = RENDER_EXPOSURE
    scene.view_settings.gamma = 1.0
    try:
        scene.view_settings.look = "Medium High Contrast"
    except TypeError:
        pass
    if scene.world is None:
        scene.world = bpy.data.worlds.new("FireViewer_Gallery_World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1.0)
        background.inputs["Strength"].default_value = 0.8


def _validate_measured_instances(bpy: Any) -> dict[str, int]:
    """Reject warped or tilted tree instances before any visual receipt."""

    counts = {"trees": 0, "buildings": 0}
    for obj in bpy.context.scene.objects:
        if obj.type != "POINTCLOUD":
            continue
        family = (
            "trees"
            if obj.name.startswith("Trees")
            else ("buildings" if obj.name.startswith("Buildings") else None)
        )
        if family is None:
            continue
        point_count = len(obj.data.points)
        counts[family] += point_count
        if family != "trees" or point_count == 0:
            continue
        scale_attribute = obj.data.attributes.get("scale")
        orientation_attribute = obj.data.attributes.get("orientation")
        if scale_attribute is None or orientation_attribute is None:
            raise SimpleZoneGalleryError("Tree instances lack scale or orientation")
        for index in range(point_count):
            scale = tuple(float(value) for value in scale_attribute.data[index].vector)
            if max(scale) - min(scale) > max(scale) * 1e-5:
                raise SimpleZoneGalleryError(
                    f"Tree instance {index} is deformed by non-uniform scaling"
                )
            quaternion = tuple(
                float(value) for value in orientation_attribute.data[index].value
            )
            if len(quaternion) != 4:
                raise SimpleZoneGalleryError("Tree orientation is not a quaternion")
            w, x, y, z = quaternion
            local_y_world_z = 2.0 * (y * z + w * x)
            if local_y_world_z < 0.999:
                raise SimpleZoneGalleryError(
                    f"Tree instance {index} is not upright after axis conversion"
                )
    return counts


def render_gallery(job_root: Path | str) -> Path:
    """Import the unified USD once, pack a Blend and render exactly 20 views."""

    root = _require_root(job_root)
    stage = root / "zone.usda"
    plan_path = root / "zone-plan.json"
    zone_receipt_path = root / "zone.done.json"
    if (
        not stage.is_file()
        or not plan_path.is_file()
        or not zone_receipt_path.is_file()
    ):
        raise SimpleZoneGalleryError("Unified zone stage, plan or receipt is missing")
    plan = _load_json(plan_path, "zone plan")
    zone_receipt = _load_json(zone_receipt_path, "zone receipt")
    bounds = plan.get("production_bounds_l93_m")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or not all(isinstance(value, (int, float)) for value in bounds)
    ):
        raise SimpleZoneGalleryError("Zone plan production bounds are invalid")
    width_m = float(bounds[2]) - float(bounds[0])
    height_m = float(bounds[3]) - float(bounds[1])
    capture_plan = build_capture_plan(width_m, height_m)

    try:
        import bpy  # type: ignore
    except ImportError as error:  # pragma: no cover - Blender runtime only
        raise SimpleZoneGalleryError(
            "Gallery render must run inside Blender"
        ) from error

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _configure_scene(bpy)
    result = bpy.ops.wm.usd_import(filepath=str(stage))
    if "FINISHED" not in result:
        raise SimpleZoneGalleryError("Blender failed to import the unified zone")
    instance_counts = _validate_measured_instances(bpy)
    if (
        zone_receipt.get("building_count") != instance_counts["buildings"]
        or zone_receipt.get("tree_count") != instance_counts["trees"]
    ):
        raise SimpleZoneGalleryError(
            "Imported Blender instances differ from the sealed production counts"
        )
    building_samples = _building_focus_samples(bpy)
    if len(building_samples) != instance_counts["buildings"]:
        raise SimpleZoneGalleryError(
            "Building focus samples differ from the sealed building count"
        )
    capture_plan = _with_building_focus(capture_plan, building_samples)
    minimum, maximum = _scene_bounds(bpy)
    center_z = (minimum[2] + maximum[2]) / 2.0
    radius = max(width_m, height_m, maximum[2] - minimum[2], 1.0)

    sun_data = bpy.data.lights.new("FireViewer_Gallery_Sun", type="SUN")
    sun_data.energy = SUN_ENERGY
    sun_data.angle = math.radians(10.0)
    sun = bpy.data.objects.new("FireViewer_Gallery_Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = (width_m / 2.0, height_m / 2.0, maximum[2] + radius)
    sun.rotation_euler = (math.radians(28.0), math.radians(-22.0), math.radians(32.0))

    camera_data = bpy.data.cameras.new("FireViewer_Gallery_Camera")
    camera = bpy.data.objects.new("FireViewer_Gallery_Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.clip_start = max(0.1, radius / 100_000.0)
    camera_data.clip_end = max(10_000.0, radius * 12.0)

    blend_path = root / BLEND_NAME
    for image in bpy.data.images:
        if image.source == "FILE" and image.filepath:
            try:
                image.pack()
            except RuntimeError as error:
                raise SimpleZoneGalleryError(
                    f"Cannot pack image {image.name}: {error}"
                ) from error
    gallery_root = root / GALLERY_DIRECTORY
    gallery_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, capture in enumerate(capture_plan, start=1):
        center_x, center_y = capture["center_xy_m"]
        target_z = float(capture.get("target_z_m", center_z))
        target = (center_x, center_y, target_z)
        if capture["projection"] == "orthographic":
            camera_data.type = "ORTHO"
            camera_data.ortho_scale = float(capture["frame_size_m"])
            camera.location = (
                center_x,
                center_y,
                maximum[2] + max(radius * 1.15, 50.0),
            )
        else:
            camera_data.type = "PERSP"
            camera_data.lens = float(capture["lens_mm"])
            azimuth = math.radians(float(capture["azimuth_degrees"]))
            distance_m = float(capture.get("distance_m", radius * 1.15))
            elevation_m = float(capture.get("elevation_m", radius * 0.90))
            camera.location = (
                center_x + math.cos(azimuth) * distance_m,
                center_y + math.sin(azimuth) * distance_m,
                (
                    target_z + elevation_m
                    if "target_z_m" in capture
                    else maximum[2] + elevation_m
                ),
            )
        _look_at(camera, target)
        image_path = gallery_root / f"{capture['capture_id']}.png"
        bpy.context.scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        if _png_dimensions(image_path) != (RESOLUTION, RESOLUTION):
            raise SimpleZoneGalleryError("Blender emitted an invalid gallery capture")
        records.append({**capture, "artifact": _artifact(image_path, root)})
        print(
            f"FIREVIEWER_CAPTURE {index}/{CAPTURE_COUNT} {capture['capture_id']}",
            flush=True,
        )

    # Keep the standalone Blend on a useful overview rather than the last
    # detail capture, and save only after the lighting/camera setup is final.
    overview = capture_plan[1]
    azimuth = math.radians(float(overview["azimuth_degrees"]))
    camera_data.type = "PERSP"
    camera_data.lens = float(overview["lens_mm"])
    camera.location = (
        width_m / 2.0 + math.cos(azimuth) * radius * 1.15,
        height_m / 2.0 + math.sin(azimuth) * radius * 1.15,
        maximum[2] + radius * 0.90,
    )
    _look_at(camera, (width_m / 2.0, height_m / 2.0, center_z))
    bpy.ops.wm.save_as_mainfile(
        filepath=str(blend_path), check_existing=False, compress=True
    )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "human_review_required": True,
        "accepted_human": False,
        "resolution": [RESOLUTION, RESOLUTION],
        "capture_count": CAPTURE_COUNT,
        "zone_stage": _artifact(stage, root),
        "zone_plan": _artifact(plan_path, root),
        "zone_receipt": _artifact(zone_receipt_path, root),
        "standalone_blend": _artifact(blend_path, root),
        "scene_bounds_m": {"minimum": minimum, "maximum": maximum},
        "instance_counts": instance_counts,
        "render_policy": dict(RENDER_POLICY),
        "captures": records,
    }
    receipt["capture_set_sha256"] = hashlib.sha256(
        _canonical_bytes(records)
    ).hexdigest()
    receipt["receipt_content_sha256"] = hashlib.sha256(
        _canonical_bytes(receipt)
    ).hexdigest()
    receipt_path = root / RECEIPT_PATH
    _write_json(receipt_path, receipt)
    verify_gallery(root)
    return receipt_path


def verify_gallery(job_root: Path | str) -> dict[str, Any]:
    """Rehash the packed Blend and every one of the 20 PNG captures."""

    root = _require_root(job_root)
    receipt_path = root / RECEIPT_PATH
    receipt = _load_json(receipt_path, "zone gallery receipt")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != STATUS
        or receipt.get("human_review_required") is not True
        or receipt.get("accepted_human") is not False
        or receipt.get("capture_count") != CAPTURE_COUNT
        or receipt.get("resolution") != [RESOLUTION, RESOLUTION]
        or receipt.get("render_policy") != RENDER_POLICY
    ):
        raise SimpleZoneGalleryError("Zone gallery receipt status is invalid")
    declared = receipt.get("receipt_content_sha256")
    content = dict(receipt)
    content.pop("receipt_content_sha256", None)
    if declared != hashlib.sha256(_canonical_bytes(content)).hexdigest():
        raise SimpleZoneGalleryError("Zone gallery receipt hash is invalid")
    for name in ("zone_stage", "zone_plan", "zone_receipt", "standalone_blend"):
        _validate_artifact(receipt.get(name), root, name)
    instance_counts = receipt.get("instance_counts")
    if (
        not isinstance(instance_counts, Mapping)
        or set(instance_counts) != {"trees", "buildings"}
        or any(
            not isinstance(value, int) or value < 0
            for value in instance_counts.values()
        )
    ):
        raise SimpleZoneGalleryError("Zone gallery instance counts are invalid")
    captures = receipt.get("captures")
    if not isinstance(captures, list) or len(captures) != CAPTURE_COUNT:
        raise SimpleZoneGalleryError("Zone gallery does not contain 20 captures")
    expected_ids = [
        item["capture_id"]
        for item in build_capture_plan(
            float(receipt["scene_bounds_m"]["maximum"][0])
            - float(receipt["scene_bounds_m"]["minimum"][0]),
            float(receipt["scene_bounds_m"]["maximum"][1])
            - float(receipt["scene_bounds_m"]["minimum"][1]),
        )
    ]
    actual_ids = [
        record.get("capture_id") for record in captures if isinstance(record, Mapping)
    ]
    if actual_ids != expected_ids:
        raise SimpleZoneGalleryError("Zone gallery capture identities are incomplete")
    building_proofs = [
        record
        for record in captures
        if isinstance(record, Mapping) and record.get("category") == "building_detail"
    ]
    if instance_counts["buildings"] > 0:
        if (
            len(building_proofs) != 1
            or building_proofs[0].get("capture_id") != BUILDING_FOCUS_CAPTURE_ID
        ):
            raise SimpleZoneGalleryError(
                "Zone gallery lacks its measured-building close-up"
            )
    elif building_proofs:
        raise SimpleZoneGalleryError(
            "Zone gallery declares a building close-up with no building"
        )
    for record in captures:
        path = _validate_artifact(
            record.get("artifact"), root, str(record.get("capture_id"))
        )
        if _png_dimensions(path) != (RESOLUTION, RESOLUTION):
            raise SimpleZoneGalleryError("Zone gallery capture resolution changed")
    if (
        receipt.get("capture_set_sha256")
        != hashlib.sha256(_canonical_bytes(captures)).hexdigest()
    ):
        raise SimpleZoneGalleryError("Zone gallery capture set hash is invalid")
    return receipt


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        values = values[values.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "verify"))
    parser.add_argument("--job-root", required=True, type=Path)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_arguments(argv)
    result = (
        render_gallery(options.job_root)
        if options.command == "render"
        else verify_gallery(options.job_root)
    )
    print(
        json.dumps(
            result if isinstance(result, dict) else {"receipt": str(result)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
