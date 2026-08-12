"""Minimal, reproducible Blender QA for a measured FireViewer scene.

The validator deliberately has no PBR or atlas dependency.  It checks one
hash-locked USD scene, one compact RGB ground texture and an exact inventory
of terrain, building and tree instances.  Blender produces two neutral 512 px
proofs.  A successful run is *technical evidence only* and always remains
``rendered_pending_human_review``.

Production usage (TEMP/TMP, Python cache and every Blender user directory must
already point to D:)::

    blender.exe --background --factory-startup --disable-autoexec \
      --python-exit-code 1 --python validate_measured_scene.py -- \
      --job D:\\...\\measured-scene-render-job.v1.json

The pure orchestration functions accept injected counters, pixel loaders and
renderers so their fail-closed behaviour can be tested without Blender.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

JOB_SCHEMA = "fireviewer.measured-scene-render-job.v1"
RECEIPT_SCHEMA = "fireviewer.measured-scene-technical-receipt.v1"
CONTRACT_SCHEMA = "fireviewer.measured-scene-validation-contract.v1"
TECHNICAL_STATUS = "rendered_pending_human_review"
RECEIPT_FILE_NAME = "measured-scene-technical-receipt.v1.json"
CONTRACT_PATH = Path(__file__).with_name("validate_measured_scene_contract.v1.json")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COUNT_KEYS = (
    "terrain_objects",
    "building_instances",
    "tree_instances",
    "context_asset_instances",
)
CAPTURE_NAMES = ("topdown", "oblique")


class MeasuredSceneQaError(RuntimeError):
    """The scene, its proofs or their provenance are incomplete or incoherent."""


@dataclass(frozen=True)
class InspectedJob:
    job_path: Path
    root: Path
    payload: dict[str, Any]
    scene_path: Path
    ground_texture_path: Path
    output_directory: Path
    expected_counts: dict[str, int]
    thresholds: dict[str, float]


PixelLoader = Callable[[Path], tuple[int, int, Sequence[tuple[int, int, int]]]]
SceneCounter = Callable[[Path], Mapping[str, int]]
RenderRunner = Callable[[Path, Path, Sequence[Mapping[str, Any]]], Mapping[str, Path]]


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeasuredSceneQaError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise MeasuredSceneQaError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise MeasuredSceneQaError(f"{label} must be a lowercase SHA-256")
    return value


def _require_d_path(path: Path, label: str, *, require_d: bool) -> Path:
    resolved = Path(path).resolve()
    if require_d and os.name == "nt" and resolved.drive.upper() != "D:":
        raise MeasuredSceneQaError(f"{label} must remain on D:, got {resolved}")
    return resolved


def _resolve_relative(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MeasuredSceneQaError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MeasuredSceneQaError(f"{label} must remain inside the job root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise MeasuredSceneQaError(f"{label} escapes the job root") from error
    return resolved


def _relative_posix(path: Path, root: Path, label: str) -> str:
    try:
        relative = Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError as error:
        raise MeasuredSceneQaError(f"{label} must remain inside {root}") from error
    return relative.as_posix()


def _artifact(path: Path, relative_to: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise MeasuredSceneQaError(f"Missing or empty evidence artifact: {resolved}")
    return {
        "path": _relative_posix(resolved, relative_to, "evidence artifact"),
        "byte_count": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_artifact(record: Any, root: Path, label: str, *, require_d: bool) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "byte_count",
        "sha256",
    }:
        raise MeasuredSceneQaError(
            f"{label} must contain exactly path, byte_count and sha256"
        )
    path = _require_d_path(
        _resolve_relative(root, record["path"], f"{label}.path"),
        label,
        require_d=require_d,
    )
    if not path.is_file():
        raise MeasuredSceneQaError(f"{label} does not exist: {path}")
    if record["byte_count"] != path.stat().st_size:
        raise MeasuredSceneQaError(f"{label} byte count changed")
    expected_hash = _require_hash(record["sha256"], f"{label}.sha256")
    if sha256_file(path) != expected_hash:
        raise MeasuredSceneQaError(f"{label} was modified after the job was locked")
    return path


def _contract() -> dict[str, Any]:
    contract = _read_json(CONTRACT_PATH, "measured scene validation contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise MeasuredSceneQaError("Unexpected measured scene contract schema")
    return contract


def _locked_job_content(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "job_content_sha256"}


def build_job(
    *,
    job_root: Path,
    scene_path: Path,
    ground_texture_path: Path,
    output_directory: Path,
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build, but do not write, a completely hash-locked render job."""

    root = Path(job_root).resolve()
    contract = _contract()
    normalized_counts = _normalize_counts(expected_counts, "expected_counts")
    scene_record = _artifact(scene_path, root)
    texture_record = _artifact(ground_texture_path, root)
    texture_record["media_type"] = "image/png"
    payload: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "validator": {
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "algorithm_sha256": sha256_file(Path(__file__)),
        },
        "scene": scene_record,
        "ground_texture": texture_record,
        "expected_counts": normalized_counts,
        "captures": contract["captures"],
        "ground_quality_thresholds": contract["ground_quality_thresholds"],
        "output_directory": _relative_posix(output_directory, root, "output_directory"),
    }
    payload["job_content_sha256"] = canonical_sha256(payload)
    return payload


def inspect_job(job_path: Path, *, require_d: bool = True) -> InspectedJob:
    job_path = _require_d_path(Path(job_path), "job", require_d=require_d)
    payload = _read_json(job_path, "measured scene render job")
    if payload.get("schema") != JOB_SCHEMA:
        raise MeasuredSceneQaError("Unexpected measured scene job schema")
    declared = _require_hash(payload.get("job_content_sha256"), "job_content_sha256")
    calculated = canonical_sha256(_locked_job_content(payload))
    if declared != calculated:
        raise MeasuredSceneQaError("Job content hash does not match its contents")

    validator = payload.get("validator")
    if not isinstance(validator, Mapping) or set(validator) != {
        "contract_sha256",
        "algorithm_sha256",
    }:
        raise MeasuredSceneQaError("validator must lock the contract and algorithm")
    if _require_hash(
        validator["contract_sha256"], "validator.contract_sha256"
    ) != sha256_file(CONTRACT_PATH):
        raise MeasuredSceneQaError("Validation contract does not match the locked job")
    if _require_hash(
        validator["algorithm_sha256"], "validator.algorithm_sha256"
    ) != sha256_file(Path(__file__)):
        raise MeasuredSceneQaError("Validator algorithm does not match the locked job")

    root = job_path.parent.resolve()
    scene_path = _validate_artifact(
        payload.get("scene"), root, "scene", require_d=require_d
    )
    if scene_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise MeasuredSceneQaError("scene must be an OpenUSD file")
    ground_record = payload.get("ground_texture")
    if not isinstance(ground_record, Mapping):
        raise MeasuredSceneQaError("ground_texture must be an artifact record")
    ground_artifact = {
        key: ground_record.get(key) for key in ("path", "byte_count", "sha256")
    }
    ground_texture = _validate_artifact(
        ground_artifact, root, "ground_texture", require_d=require_d
    )
    if (
        ground_record.get("media_type") != "image/png"
        or ground_texture.suffix.lower() != ".png"
    ):
        raise MeasuredSceneQaError("ground_texture must be one RGB8 PNG")

    contract = _contract()
    if payload.get("captures") != contract["captures"]:
        raise MeasuredSceneQaError("Capture settings differ from the locked contract")
    if (
        payload.get("ground_quality_thresholds")
        != contract["ground_quality_thresholds"]
    ):
        raise MeasuredSceneQaError("Ground thresholds differ from the locked contract")

    output = _require_d_path(
        _resolve_relative(root, payload.get("output_directory"), "output_directory"),
        "output_directory",
        require_d=require_d,
    )
    counts = _normalize_counts(payload.get("expected_counts"), "expected_counts")
    thresholds = _normalize_thresholds(payload.get("ground_quality_thresholds"))
    return InspectedJob(
        job_path=job_path,
        root=root,
        payload=payload,
        scene_path=scene_path,
        ground_texture_path=ground_texture,
        output_directory=output,
        expected_counts=counts,
        thresholds=thresholds,
    )


def _normalize_counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(COUNT_KEYS):
        raise MeasuredSceneQaError(
            f"{label} must contain exactly {', '.join(COUNT_KEYS)}"
        )
    result: dict[str, int] = {}
    for key in COUNT_KEYS:
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise MeasuredSceneQaError(f"{label}.{key} must be a non-negative integer")
        result[key] = count
    if result["terrain_objects"] < 1:
        raise MeasuredSceneQaError(f"{label}.terrain_objects must be at least one")
    return result


def _normalize_thresholds(value: Any) -> dict[str, float]:
    keys = {
        "maximum_mean_luminance",
        "maximum_near_white_fraction",
        "minimum_luminance_standard_deviation",
        "minimum_mean_saturation",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise MeasuredSceneQaError("ground_quality_thresholds has unexpected fields")
    result: dict[str, float] = {}
    for key in sorted(keys):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise MeasuredSceneQaError(f"{key} must be numeric")
        result[key] = float(number)
    return result


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (
        abs(estimate - left),
        abs(estimate - above),
        abs(estimate - upper_left),
    )
    return (left, above, upper_left)[distances.index(min(distances))]


def load_rgb8_png(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Read a non-interlaced RGB8/RGBA8 PNG using only the standard library."""

    raw = Path(path).read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise MeasuredSceneQaError("ground_texture is not a PNG")
    offset = 8
    header: tuple[int, int, int, int, int, int, int] | None = None
    compressed = bytearray()
    while offset + 12 <= len(raw):
        length = struct.unpack_from(">I", raw, offset)[0]
        chunk_name = raw[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(raw):
            raise MeasuredSceneQaError("Truncated ground PNG")
        payload = raw[start:end]
        expected_crc = struct.unpack_from(">I", raw, end)[0]
        if zlib.crc32(chunk_name + payload) & 0xFFFFFFFF != expected_crc:
            raise MeasuredSceneQaError("Ground PNG contains an invalid CRC")
        if chunk_name == b"IHDR":
            if length != 13:
                raise MeasuredSceneQaError("Invalid PNG IHDR")
            header = struct.unpack(">IIBBBBB", payload)
        elif chunk_name == b"IDAT":
            compressed.extend(payload)
        elif chunk_name == b"IEND":
            break
        offset = end + 4
    if header is None:
        raise MeasuredSceneQaError("Ground PNG has no IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = header
    if width <= 0 or height <= 0:
        raise MeasuredSceneQaError("Ground PNG dimensions must be positive")
    if (bit_depth, compression, filtering, interlace) != (8, 0, 0, 0):
        raise MeasuredSceneQaError("Ground PNG must be non-interlaced 8-bit PNG")
    channels = {2: 3, 6: 4}.get(color_type)
    if channels is None:
        raise MeasuredSceneQaError("Ground PNG must be RGB8 or RGBA8")
    try:
        scanlines = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise MeasuredSceneQaError(f"Cannot decompress ground PNG: {error}") from error
    stride = width * channels
    if len(scanlines) != height * (stride + 1):
        raise MeasuredSceneQaError("Ground PNG scanline size is invalid")

    rows: list[bytearray] = []
    cursor = 0
    prior = bytearray(stride)
    for _row_index in range(height):
        filter_type = scanlines[cursor]
        cursor += 1
        filtered = scanlines[cursor : cursor + stride]
        cursor += stride
        decoded = bytearray(stride)
        for index, encoded in enumerate(filtered):
            left = decoded[index - channels] if index >= channels else 0
            above = prior[index]
            upper_left = prior[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth(left, above, upper_left)
            else:
                raise MeasuredSceneQaError(f"Unsupported PNG filter {filter_type}")
            decoded[index] = (encoded + predictor) & 0xFF
        rows.append(decoded)
        prior = decoded
    pixels = [
        (row[index], row[index + 1], row[index + 2])
        for row in rows
        for index in range(0, stride, channels)
    ]
    return width, height, pixels


def analyze_ground_texture(
    path: Path, *, pixel_loader: PixelLoader | None = None
) -> dict[str, Any]:
    loader = pixel_loader or load_rgb8_png
    width, height, pixels = loader(Path(path))
    if width <= 0 or height <= 0 or len(pixels) != width * height:
        raise MeasuredSceneQaError(
            "Ground texture pixel loader returned invalid dimensions"
        )
    luminances: list[float] = []
    saturations: list[float] = []
    near_white = 0
    for pixel in pixels:
        if len(pixel) != 3:
            raise MeasuredSceneQaError("Ground texture must contain RGB triplets")
        red, green, blue = (max(0, min(255, int(channel))) / 255.0 for channel in pixel)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        saturation = 0.0 if maximum == 0.0 else (maximum - minimum) / maximum
        luminances.append(luminance)
        saturations.append(saturation)
        if luminance >= 0.90 and saturation <= 0.08:
            near_white += 1
    count = len(luminances)
    mean_luminance = math.fsum(luminances) / count
    variance = math.fsum((value - mean_luminance) ** 2 for value in luminances) / count
    return {
        "width": width,
        "height": height,
        "pixel_count": count,
        "mean_luminance": round(mean_luminance, 9),
        "luminance_standard_deviation": round(math.sqrt(variance), 9),
        "mean_saturation": round(math.fsum(saturations) / count, 9),
        "near_white_fraction": round(near_white / count, 9),
    }


def validate_ground_metrics(
    metrics: Mapping[str, Any], thresholds: Mapping[str, float]
) -> None:
    failures: list[str] = []
    checks = (
        (
            float(metrics["mean_luminance"]) <= thresholds["maximum_mean_luminance"],
            "mean luminance is too high (pale/white ground)",
        ),
        (
            float(metrics["near_white_fraction"])
            <= thresholds["maximum_near_white_fraction"],
            "too many near-white ground pixels",
        ),
        (
            float(metrics["luminance_standard_deviation"])
            >= thresholds["minimum_luminance_standard_deviation"],
            "ground texture is too flat",
        ),
        (
            float(metrics["mean_saturation"]) >= thresholds["minimum_mean_saturation"],
            "ground texture is too desaturated/pale",
        ),
    )
    for passed, reason in checks:
        if not passed:
            failures.append(reason)
    if failures:
        raise MeasuredSceneQaError("Ground texture QA failed: " + "; ".join(failures))


def _usd_attribute_value(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return None
    return attribute.Get()


def _point_instancer_count(prim: Any) -> int:
    """Derive an instancer count from its authored per-instance arrays."""

    arrays: dict[str, Sequence[Any]] = {}
    for name in ("ids", "positions", "scales", "orientations", "protoIndices"):
        value = _usd_attribute_value(prim, name)
        if value is None:
            raise MeasuredSceneQaError(
                f"PointInstancer {prim.GetPath()} has no authored {name} array"
            )
        arrays[name] = value
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise MeasuredSceneQaError(
            f"PointInstancer {prim.GetPath()} has inconsistent instance arrays: {lengths}"
        )
    ids = [int(value) for value in arrays["ids"]]
    if len(ids) != len(set(ids)):
        raise MeasuredSceneQaError(
            f"PointInstancer {prim.GetPath()} has duplicate deterministic ids"
        )
    return lengths["ids"]


def count_tagged_usd_scene(scene_path: Path) -> dict[str, int]:
    """Count explicit markers while deriving PointInstancer counts from USD arrays."""

    try:
        from pxr import Usd, UsdGeom, UsdShade  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on Blender runtime
        raise MeasuredSceneQaError(
            "OpenUSD Python bindings are required to count the scene"
        ) from error
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise MeasuredSceneQaError(f"Cannot open USD scene: {scene_path}")
    counts = {key: 0 for key in COUNT_KEYS}
    categories = {
        "terrain": "terrain_objects",
        "building": "building_instances",
        "tree": "tree_instances",
        "context_asset": "context_asset_instances",
    }
    tagged = 0
    prototype_meshes = []
    unbound_prototype_meshes = []
    instance_ground_z: list[tuple[str, float]] = []
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/MeasuredScene/Prototypes/") and prim.IsA(
            UsdGeom.Mesh
        ):
            prototype_meshes.append(str(prim.GetPath()))
            material, _binding = UsdShade.MaterialBindingAPI(
                prim
            ).ComputeBoundMaterial()
            if not material:
                unbound_prototype_meshes.append(str(prim.GetPath()))
        category = _usd_attribute_value(prim, "fireviewer:category")
        if category is None:
            continue
        key = categories.get(str(category).strip().lower())
        if key is None:
            continue
        authored_count = _usd_attribute_value(prim, "fireviewer:count")
        if isinstance(authored_count, bool) or not isinstance(authored_count, int):
            raise MeasuredSceneQaError(f"Invalid fireviewer:count on {prim.GetPath()}")
        if key == "terrain_objects":
            count = 1
        else:
            if prim.GetTypeName() != "PointInstancer":
                raise MeasuredSceneQaError(
                    f"Inventory marker {category!r} is not a PointInstancer on {prim.GetPath()}"
                )
            count = _point_instancer_count(prim)
        if authored_count != count:
            raise MeasuredSceneQaError(
                f"fireviewer:count does not match authored USD arrays on {prim.GetPath()}"
            )
        if key != "terrain_objects":
            for position in _usd_attribute_value(prim, "positions"):
                instance_ground_z.append((str(prim.GetPath()), float(position[2])))
        counts[key] += count
        tagged += 1
    if tagged == 0:
        raise MeasuredSceneQaError(
            "USD scene has no explicit fireviewer:category inventory markers"
        )
    if (
        counts["building_instances"]
        + counts["tree_instances"]
        + counts["context_asset_instances"]
        > 0
    ):
        if not prototype_meshes:
            raise MeasuredSceneQaError(
                "USD scene has instances but no composed prototype meshes"
            )
        if unbound_prototype_meshes:
            preview = ", ".join(unbound_prototype_meshes[:3])
            raise MeasuredSceneQaError(
                "USD prototype meshes have no resolved material binding: " + preview
            )
        terrain_prim = stage.GetPrimAtPath("/MeasuredScene/Terrain")
        if not terrain_prim:
            raise MeasuredSceneQaError("USD scene has no /MeasuredScene/Terrain prim")
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        terrain_range = bbox_cache.ComputeWorldBound(terrain_prim).ComputeAlignedRange()
        terrain_min_z = float(terrain_range.GetMin()[2])
        terrain_max_z = float(terrain_range.GetMax()[2])
        if not math.isfinite(terrain_min_z) or not math.isfinite(terrain_max_z):
            raise MeasuredSceneQaError("Terrain world bounds are not finite")
        tolerance_m = 1.0
        outside = [
            (path, z)
            for path, z in instance_ground_z
            if z < terrain_min_z - tolerance_m or z > terrain_max_z + tolerance_m
        ]
        if outside:
            path, z = outside[0]
            raise MeasuredSceneQaError(
                f"Instance ground Z {z:.3f} on {path} is outside terrain range "
                f"{terrain_min_z:.3f}..{terrain_max_z:.3f}"
            )
    return counts


def _normalize_actual_counts(value: Mapping[str, int]) -> dict[str, int]:
    return _normalize_counts(value, "actual_counts")


def _blender_module() -> Any:
    try:
        import bpy  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on Blender runtime
        raise MeasuredSceneQaError(
            "This render phase must run inside Blender"
        ) from error
    return bpy


def _scene_bounds(
    bpy: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    from mathutils import Vector  # type: ignore  # Blender-only lazy import

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
        raise MeasuredSceneQaError("Imported scene contains no renderable bounds")
    return tuple(minimum), tuple(maximum)  # type: ignore[return-value]


def _look_at(bpy: Any, camera: Any, target: Sequence[float]) -> None:
    from mathutils import Vector  # type: ignore  # Blender-only lazy import

    direction = Vector(target) - camera.location
    if direction.length <= 1e-6:
        raise MeasuredSceneQaError("Camera cannot be placed on its target")
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _configure_neutral_scene(bpy: Any) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:  # pragma: no cover - older supported fallback
        scene.view_settings.view_transform = "Standard"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("FireViewer_QA_World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.12, 0.12, 0.12, 1.0)
        background.inputs["Strength"].default_value = 0.35


def _new_sun(bpy: Any, center: Sequence[float]) -> None:
    data = bpy.data.lights.new("FireViewer_QA_Sun", type="SUN")
    data.energy = 2.0
    data.angle = math.radians(12.0)
    light = bpy.data.objects.new("FireViewer_QA_Sun", data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (center[0], center[1], center[2] + 1000.0)
    light.rotation_euler = (math.radians(28.0), math.radians(-22.0), math.radians(32.0))


def render_with_blender(
    scene_path: Path,
    output_directory: Path,
    captures: Sequence[Mapping[str, Any]],
) -> Mapping[str, Path]:
    """Import the USD scene once and render the two contract cameras."""

    bpy = _blender_module()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _configure_neutral_scene(bpy)
    result = bpy.ops.wm.usd_import(filepath=str(scene_path))
    if "FINISHED" not in result:
        raise MeasuredSceneQaError(f"Blender failed to import {scene_path}")
    minimum, maximum = _scene_bounds(bpy)
    center = tuple((minimum[index] + maximum[index]) / 2.0 for index in range(3))
    spans = tuple(maximum[index] - minimum[index] for index in range(3))
    horizontal = max(spans[0], spans[1], 1.0)
    radius = max(horizontal, spans[2], 1.0)
    _new_sun(bpy, center)

    camera_data = bpy.data.cameras.new("FireViewer_QA_Camera")
    camera = bpy.data.objects.new("FireViewer_QA_Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.clip_start = max(0.1, radius / 100_000.0)
    camera_data.clip_end = max(10_000.0, radius * 10.0)

    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    for capture in captures:
        name = capture.get("name")
        if name == "topdown":
            camera_data.type = "ORTHO"
            camera_data.ortho_scale = horizontal * 1.08
            camera.location = (
                center[0],
                center[1],
                maximum[2] + max(radius * 1.25, 10.0),
            )
        elif name == "oblique":
            camera_data.type = "PERSP"
            camera_data.lens = float(capture.get("lens_mm", 50.0))
            camera.location = (
                center[0] - radius * 1.15,
                center[1] - radius * 1.15,
                maximum[2] + radius * 0.90,
            )
        else:
            raise MeasuredSceneQaError(f"Unsupported capture {name!r}")
        _look_at(bpy, camera, center)
        target = output_directory / f"{name}.png"
        bpy.context.scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        artifacts[str(name)] = target
    return artifacts


def _png_dimensions(path: Path) -> tuple[int, int]:
    raw = Path(path).read_bytes()[:24]
    if (
        len(raw) != 24
        or not raw.startswith(b"\x89PNG\r\n\x1a\n")
        or raw[12:16] != b"IHDR"
    ):
        raise MeasuredSceneQaError(f"Capture is not a PNG: {path}")
    return struct.unpack(">II", raw[16:24])


def _validate_runtime_environment(*, require_d: bool) -> None:
    if not require_d or os.name != "nt":
        return
    required_paths = (
        "TEMP",
        "TMP",
        "PYTHONPYCACHEPREFIX",
        "BLENDER_USER_CONFIG",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_USER_DATAFILES",
        "BLENDER_USER_EXTENSIONS",
    )
    for name in required_paths:
        value = os.environ.get(name)
        if not value:
            raise MeasuredSceneQaError(
                f"{name} must be defined on D: before Blender runs"
            )
        _require_d_path(Path(value), name, require_d=True)


def run_validation(
    job_path: Path,
    *,
    require_d: bool = True,
    scene_counter: SceneCounter | None = None,
    pixel_loader: PixelLoader | None = None,
    render_runner: RenderRunner | None = None,
) -> Path:
    """Validate, render, hash proofs and emit a non-accepting receipt."""

    _validate_runtime_environment(require_d=require_d)
    inspected = inspect_job(job_path, require_d=require_d)
    metrics = analyze_ground_texture(
        inspected.ground_texture_path, pixel_loader=pixel_loader
    )
    validate_ground_metrics(metrics, inspected.thresholds)

    counter = scene_counter or count_tagged_usd_scene
    actual_counts = _normalize_actual_counts(counter(inspected.scene_path))
    if actual_counts != inspected.expected_counts:
        raise MeasuredSceneQaError(
            f"Scene inventory mismatch: expected {inspected.expected_counts}, "
            f"got {actual_counts}"
        )

    renderer = render_runner or render_with_blender
    capture_paths = renderer(
        inspected.scene_path,
        inspected.output_directory,
        inspected.payload["captures"],
    )
    if not isinstance(capture_paths, Mapping) or set(capture_paths) != set(
        CAPTURE_NAMES
    ):
        raise MeasuredSceneQaError("Renderer must return exactly topdown and oblique")
    capture_evidence: dict[str, dict[str, Any]] = {}
    for name in CAPTURE_NAMES:
        path = _require_d_path(
            Path(capture_paths[name]), f"{name} capture", require_d=require_d
        )
        if _png_dimensions(path) != (512, 512):
            raise MeasuredSceneQaError(f"{name} capture must be exactly 512x512")
        capture_evidence[name] = _artifact(path, inspected.root)

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": TECHNICAL_STATUS,
        "human_review_required": True,
        "job": _artifact(inspected.job_path, inspected.root),
        "scene": _artifact(inspected.scene_path, inspected.root),
        "ground_texture": _artifact(inspected.ground_texture_path, inspected.root),
        "validator": dict(inspected.payload["validator"]),
        "counts": actual_counts,
        "ground_metrics": metrics,
        "captures": capture_evidence,
    }
    receipt["evidence_set_sha256"] = canonical_sha256(
        {
            "job": receipt["job"],
            "scene": receipt["scene"],
            "ground_texture": receipt["ground_texture"],
            "counts": receipt["counts"],
            "ground_metrics": receipt["ground_metrics"],
            "captures": receipt["captures"],
        }
    )
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    receipt_path = inspected.root / RECEIPT_FILE_NAME
    _write_json_atomic(receipt_path, receipt)
    verify_receipt(receipt_path, require_d=require_d)
    return receipt_path


def verify_receipt(receipt_path: Path, *, require_d: bool = True) -> dict[str, Any]:
    receipt_path = _require_d_path(receipt_path, "receipt", require_d=require_d)
    receipt = _read_json(receipt_path, "measured scene receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise MeasuredSceneQaError("Unexpected receipt schema")
    if (
        receipt.get("status") != TECHNICAL_STATUS
        or receipt.get("human_review_required") is not True
    ):
        raise MeasuredSceneQaError("Receipt cannot claim automatic visual acceptance")
    declared = _require_hash(
        receipt.get("receipt_content_sha256"), "receipt_content_sha256"
    )
    content = {
        key: value for key, value in receipt.items() if key != "receipt_content_sha256"
    }
    if canonical_sha256(content) != declared:
        raise MeasuredSceneQaError("Receipt content hash is invalid")

    root = receipt_path.parent
    validated: dict[str, Any] = {}
    for label in ("job", "scene", "ground_texture"):
        validated[label] = receipt.get(label)
        _validate_artifact(receipt.get(label), root, label, require_d=require_d)
    captures = receipt.get("captures")
    if not isinstance(captures, Mapping) or set(captures) != set(CAPTURE_NAMES):
        raise MeasuredSceneQaError("Receipt must contain both captures")
    for name in CAPTURE_NAMES:
        _validate_artifact(captures[name], root, name, require_d=require_d)
    evidence = {
        "job": receipt["job"],
        "scene": receipt["scene"],
        "ground_texture": receipt["ground_texture"],
        "counts": receipt.get("counts"),
        "ground_metrics": receipt.get("ground_metrics"),
        "captures": receipt["captures"],
    }
    if _require_hash(
        receipt.get("evidence_set_sha256"), "evidence_set_sha256"
    ) != canonical_sha256(evidence):
        raise MeasuredSceneQaError("Receipt evidence set hash is invalid")
    return receipt


def _cli_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" in arguments:
        arguments = arguments[arguments.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    return parser.parse_args(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_arguments(argv)
    receipt_path = run_validation(args.job, require_d=True)
    print(f"MEASURED_SCENE_QA={receipt_path}")
    print(f"STATUS={TECHNICAL_STATUS}")
    return 0


if __name__ == "__main__":  # pragma: no cover - Blender entry point
    raise SystemExit(main())


__all__ = [
    "CONTRACT_SCHEMA",
    "JOB_SCHEMA",
    "RECEIPT_SCHEMA",
    "TECHNICAL_STATUS",
    "MeasuredSceneQaError",
    "analyze_ground_texture",
    "build_job",
    "canonical_sha256",
    "count_tagged_usd_scene",
    "inspect_job",
    "load_rgb8_png",
    "run_validation",
    "sha256_file",
    "validate_ground_metrics",
    "verify_receipt",
]
