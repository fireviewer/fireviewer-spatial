"""Compile temporary orthophotos into deterministic PBR surface selections.

The orthophoto is an offline classification input only.  No source pixel and no
source path is serialized.  Runtime outputs select complete, tileable PBR
profiles from a hash-locked 72-profile library.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from affine import Affine
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from shapely import force_2d, set_precision
from shapely.geometry import LineString, MultiLineString, mapping, shape
from shapely.geometry.base import BaseGeometry


CONTRACT_SCHEMA = "fireviewer.orthophoto-surface-correspondence-contract.v1"
LIBRARY_SCHEMA = "fireviewer.clean-pbr-texture-library.v1"
TEXTURE_CONTRACT_SCHEMA = "fireviewer.ground-surface-texture-contract.v4"
MODEL_SCHEMA = "fireviewer.orthophoto-surface-model.v1"
OUTPUT_SCHEMA = "fireviewer.surface-correspondence-tile.v1"
CRS = "EPSG:2154"
TILE_SIZE_M = 500
OUTPUT_NAMES = (
    "ground-profile-ids.png",
    "ground-profile-weights.png",
    "ground-confidence.png",
    "ground-orientation.png",
    "surface-correspondence.json",
)
REQUIRED_PBR_ASSETS = {"basecolor", "normal", "height", "orm"}
ALLOWED_PROJECTIONS = {"world_xy", "world_triplanar"}
HEX_DIGITS = frozenset("0123456789abcdef")
INFINITE_SCORE = np.iinfo(np.int64).max // 8


@dataclass(frozen=True)
class SurfaceCorrespondence:
    """One globally classified, tile-aligned core window."""

    core_bounds_l93_m: tuple[int, int, int, int]
    profile_ids: np.ndarray
    profile_weights: np.ndarray
    confidence: np.ndarray
    orientation: np.ndarray
    identity: Mapping[str, str]
    profile_table: tuple[Mapping[str, Any], ...]
    class_by_profile_index: tuple[str, ...]
    restriction_counts: Mapping[str, int]
    restriction_codes: np.ndarray
    restriction_labels: tuple[str, ...]
    tile_input_sha256_by_bounds: Mapping[str, str]


@dataclass(frozen=True)
class TileCorrespondence:
    """Exactly one 500 m by 500 m runtime correspondence tile."""

    bounds_l93_m: tuple[int, int, int, int]
    profile_ids: np.ndarray
    profile_weights: np.ndarray
    confidence: np.ndarray
    orientation: np.ndarray
    identity: Mapping[str, str]
    profile_table: tuple[Mapping[str, Any], ...]
    class_by_profile_index: tuple[str, ...]
    restriction_counts: Mapping[str, int]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
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
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _load_contract(path: Path | None) -> tuple[dict[str, Any], Path]:
    resolved = (
        Path(__file__).with_name("orthophoto_surface_correspondence_contract.v1.json")
        if path is None
        else Path(path)
    ).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("status") != "locked":
        raise ValueError("Unsupported or unlocked correspondence contract")
    if payload.get("crs") != CRS:
        raise ValueError("Correspondence contract must use EPSG:2154")
    classification = payload.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("Correspondence classification contract is absent")
    sizes = classification.get("feature_window_sizes_m")
    if sizes != [1, 5, 17]:
        raise ValueError("Correspondence feature windows must remain [1, 5, 17] m")
    if classification.get("maximum_profiles_per_pixel") != 4:
        raise ValueError("Correspondence output must use exactly four RGBA slots")
    if payload.get("runtime", {}).get("orthophoto_dependency") != "forbidden":
        raise ValueError("Runtime orthophoto dependency must remain forbidden")
    if payload.get("pbr_library", {}).get("procedural_materials") != "forbidden":
        raise ValueError("Procedural materials must remain forbidden")
    if payload.get("pbr_library", {}).get("schema") != LIBRARY_SCHEMA:
        raise ValueError("Correspondence contract does not bind the clean PBR library")
    return payload, resolved


def _asset_identity(value: Any, *, profile_id: str, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"PBR asset {profile_id}/{role} is absent")
    digest = _require_sha256(
        value.get("sha256"), label=f"PBR asset {profile_id}/{role}"
    )
    try:
        byte_count = int(value.get("byte_count"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"PBR artifact byte count is invalid: {profile_id}/{role}"
        ) from error
    path = str(value.get("path", "")).strip()
    if (
        byte_count <= 0
        or not path
        or Path(path).is_absolute()
        or bool(Path(path).drive)
        or ".." in Path(path).parts
    ):
        raise ValueError(f"PBR artifact identity is invalid: {profile_id}/{role}")
    # Paths belong to the shared library package.  Only content identities are
    # propagated to correspondence manifests, never source paths.
    return {"byte_count": byte_count, "sha256": digest}


def _validate_pbr_library(
    library: Mapping[str, Any], expected_sha256: str | None
) -> tuple[tuple[dict[str, Any], ...], str]:
    if library.get("schema") != LIBRARY_SCHEMA:
        raise ValueError("Unsupported PBR surface library")
    if library.get("status") != "accepted_clean_pbr_library":
        raise ValueError("PBR surface library has no accepted visual review")
    texture_contract_path = (
        Path(__file__).with_name("ground_surface_texture_contract.v4.json").resolve()
    )
    texture_contract = json.loads(texture_contract_path.read_text(encoding="utf-8"))
    if texture_contract.get("schema") != TEXTURE_CONTRACT_SCHEMA:
        raise ValueError("Ground texture contract v4 is absent")
    if library.get("texture_contract_sha256") != _sha256_file(texture_contract_path):
        raise ValueError("PBR library does not bind the ground texture contract v4")
    if library.get("orthophoto_dependency") != "forbidden":
        raise ValueError("PBR library must not retain an orthophoto dependency")
    lineage = library.get("lineage")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("source") != "fresh_clean_pbr_v4"
        or lineage.get("legacy_atlas_dependencies") != []
    ):
        raise ValueError("PBR library has a legacy atlas dependency")
    if "procedural_only" in _canonical_bytes(library).decode("utf-8").casefold():
        raise ValueError("PBR surface library contains a procedural-only profile")
    runtime_atlases = library.get("runtime_atlases")
    if (
        not isinstance(runtime_atlases, Mapping)
        or set(runtime_atlases) != REQUIRED_PBR_ASSETS
    ):
        raise ValueError("PBR surface library must expose four runtime atlases")
    for role in sorted(REQUIRED_PBR_ASSETS):
        _asset_identity(runtime_atlases[role], profile_id="runtime_atlas", role=role)
    visual = library.get("visual_acceptance")
    if (
        not isinstance(visual, Mapping)
        or visual.get("status") != "accepted_human_visual"
        or not isinstance(visual.get("receipt"), Mapping)
    ):
        raise ValueError("PBR surface library visual acceptance is absent")
    _asset_identity(visual["receipt"], profile_id="visual_acceptance", role="receipt")
    profiles = library.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 72:
        raise ValueError("PBR surface library must contain exactly 72 profiles")
    stable_ids = texture_contract.get("profile_contract", {}).get("stable_profile_ids")
    if not isinstance(stable_ids, list) or len(stable_ids) != 72:
        raise ValueError("Ground texture contract v4 has no stable 72-profile table")
    output: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for stable_index, (raw, expected_id) in enumerate(
        zip(profiles, stable_ids, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("Every PBR profile must be an object")
        profile_id = str(raw.get("id", "")).strip()
        if (
            raw.get("stable_index") != stable_index
            or profile_id != expected_id
            or profile_id in identifiers
            or any(separator in profile_id for separator in ("/", "\\", "\0"))
        ):
            raise ValueError("PBR stable profile identity differs from contract v4")
        identifiers.add(profile_id)
        if (
            raw.get("surface_basis") != "atlas_pbr"
            or raw.get("source_kind") != "clean_pbr_profile_texture"
        ):
            raise ValueError(
                f"PBR profile does not use clean atlas textures: {profile_id}"
            )
        if raw.get("atlas_slot") != stable_index:
            raise ValueError(f"PBR atlas slot differs from stable_index: {profile_id}")
        atlas = texture_contract["runtime_atlas"]
        column = stable_index % int(atlas["columns"])
        row = stable_index // int(atlas["columns"])
        cell = int(atlas["cell_size_px"])
        gutter = int(atlas["gutter_px"])
        expected_atlas_uv = {
            "offset": [
                (column * cell + gutter) / int(atlas["width_px"]),
                (row * cell + gutter) / int(atlas["height_px"]),
            ],
            "scale": [
                (cell - 2 * gutter) / int(atlas["width_px"]),
                (cell - 2 * gutter) / int(atlas["height_px"]),
            ],
        }
        atlas_uv = raw.get("atlas_uv")
        if not isinstance(atlas_uv, Mapping) or set(atlas_uv) != {
            "offset",
            "scale",
        }:
            raise ValueError(f"PBR atlas UV is invalid: {profile_id}")
        for field in ("offset", "scale"):
            values = atlas_uv[field]
            if (
                not isinstance(values, list)
                or len(values) != 2
                or any(
                    not math.isclose(float(value), expected, abs_tol=1.0e-12)
                    for value, expected in zip(
                        values, expected_atlas_uv[field], strict=True
                    )
                )
            ):
                raise ValueError(f"PBR atlas UV differs from contract v4: {profile_id}")
        projection = str(raw.get("projection", ""))
        if projection not in ALLOWED_PROJECTIONS:
            raise ValueError(f"PBR projection is invalid: {profile_id}")
        expected_projection = (
            "world_triplanar" if profile_id.startswith("cliff_surface.") else "world_xy"
        )
        if projection != expected_projection:
            raise ValueError(f"PBR projection differs from contract v4: {profile_id}")
        assets = raw.get("textures")
        if not isinstance(assets, Mapping) or set(assets) != REQUIRED_PBR_ASSETS:
            raise ValueError(f"PBR profile is incomplete: {profile_id}")
        try:
            physical_scale_m = float(raw.get("physical_scale_m"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"PBR physical scale is invalid: {profile_id}") from error
        if not math.isfinite(physical_scale_m) or not 1.5 <= physical_scale_m <= 8.0:
            raise ValueError(f"PBR physical scale is invalid: {profile_id}")
        output.append(
            {
                "stable_index": stable_index,
                "id": profile_id,
                "atlas_slot": stable_index,
                "atlas_uv": expected_atlas_uv,
                "projection": projection,
                "physical_scale_m": physical_scale_m,
                "textures": {
                    role: _asset_identity(
                        assets[role], profile_id=profile_id, role=role
                    )
                    for role in sorted(REQUIRED_PBR_ASSETS)
                },
            }
        )
    digest = _canonical_sha256(library)
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, label="PBR library"
    ):
        raise ValueError("PBR surface library hash mismatch")
    return tuple(output), digest


def _u8_sequence(value: Any, length: int, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {length} integers")
    result = tuple(int(item) for item in value)
    if len(result) != length or any(item < 0 or item > 255 for item in result):
        raise ValueError(f"{label} must contain {length} uint8 values")
    return result


def _validate_model(
    model: Mapping[str, Any],
    profile_table: Sequence[Mapping[str, Any]],
    expected_sha256: str | None,
) -> tuple[tuple[dict[str, Any], ...], str]:
    if model.get("schema") != MODEL_SCHEMA or model.get("status") != "locked":
        raise ValueError("Unsupported or unlocked orthophoto correspondence model")
    if model.get("resolution_m") != 1 or model.get("feature_window_sizes_m") != [
        1,
        5,
        17,
    ]:
        raise ValueError(
            "Correspondence model must target the locked 1 m multiscale grid"
        )
    raw_profiles = model.get("profiles")
    if not isinstance(raw_profiles, list) or len(raw_profiles) != 72:
        raise ValueError("Correspondence model must bind exactly 72 profiles")
    by_id = {str(profile["id"]): profile for profile in profile_table}
    normalized_by_id: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for raw in raw_profiles:
        if not isinstance(raw, Mapping):
            raise ValueError("Every correspondence profile must be an object")
        profile_id = str(raw.get("profile_id", "")).strip()
        if profile_id not in by_id or profile_id in seen:
            raise ValueError(f"Model profile binding is invalid: {profile_id}")
        seen.add(profile_id)
        class_id = str(raw.get("class_id", "")).strip()
        if (
            not class_id
            or ":" in class_id
            or any(separator in class_id for separator in ("/", "\\", "\0"))
        ):
            raise ValueError(f"Model semantic class is invalid: {profile_id}")
        raw_rgb = raw.get("rgb_u8_by_scale")
        if not isinstance(raw_rgb, list) or len(raw_rgb) != 3:
            raise ValueError(f"Model RGB prototypes are incomplete: {profile_id}")
        rgb = tuple(
            _u8_sequence(scale, 3, label=f"RGB prototype {profile_id}")
            for scale in raw_rgb
        )
        texture = _u8_sequence(
            raw.get("texture_u8_by_scale"), 3, label=f"texture prototype {profile_id}"
        )
        orientation_mode = str(raw.get("orientation_mode", ""))
        if orientation_mode not in {"fixed", "image_structure"}:
            raise ValueError(f"Orientation mode is invalid: {profile_id}")
        try:
            default_orientation = float(raw.get("default_orientation_deg", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Default orientation is invalid: {profile_id}") from error
        if not math.isfinite(default_orientation):
            raise ValueError(f"Default orientation is invalid: {profile_id}")
        normalized_by_id[profile_id] = {
            "profile_id": profile_id,
            "class_id": class_id,
            "rgb_u8_by_scale": rgb,
            "texture_u8_by_scale": texture,
            "orientation_mode": orientation_mode,
            "default_orientation_deg": default_orientation % 180.0,
        }
    if set(by_id) != seen:
        raise ValueError("Correspondence model does not bind the complete PBR library")
    digest = _canonical_sha256(model)
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, label="correspondence model"
    ):
        raise ValueError("Correspondence model hash mismatch")
    return tuple(
        normalized_by_id[str(profile["id"])] for profile in profile_table
    ), digest


def _integer_bounds(values: Sequence[Any], *, label: str) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    floats = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or not value.is_integer() for value in floats):
        raise ValueError(f"{label} must use integer metre EPSG:2154 coordinates")
    west, south, east, north = (int(value) for value in floats)
    if east <= west or north <= south:
        raise ValueError(f"{label} is empty")
    return west, south, east, north


def _validate_grid(
    rgb_u8: np.ndarray,
    transform: Affine | Sequence[float],
    crs: str,
    core_bounds: Sequence[Any],
    halo_m: int,
) -> tuple[Affine, tuple[int, int, int, int], tuple[int, int, int, int]]:
    if not isinstance(rgb_u8, np.ndarray) or rgb_u8.dtype != np.uint8:
        raise ValueError("Orthophoto window must be a uint8 NumPy array")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("Orthophoto window must have RGB shape H x W x 3")
    if crs != CRS:
        raise ValueError("Orthophoto window must use EPSG:2154")
    affine = transform if isinstance(transform, Affine) else Affine(*transform[:6])
    expected = (1.0, 0.0, 0.0, -1.0)
    if not all(
        math.isclose(actual, wanted, abs_tol=1e-12)
        for actual, wanted in zip((affine.a, affine.b, affine.d, affine.e), expected)
    ):
        raise ValueError("Orthophoto must be north-up on an exact 1 m grid")
    if not float(affine.c).is_integer() or not float(affine.f).is_integer():
        raise ValueError("Orthophoto transform must align to the global metre grid")
    height, width = rgb_u8.shape[:2]
    window = (
        int(affine.c),
        int(affine.f) - height,
        int(affine.c) + width,
        int(affine.f),
    )
    core = _integer_bounds(core_bounds, label="core bounds")
    if (
        core[0] % TILE_SIZE_M
        or core[1] % TILE_SIZE_M
        or core[2] % TILE_SIZE_M
        or core[3] % TILE_SIZE_M
    ):
        raise ValueError("Core bounds must align to the global 500 m tile grid")
    if (
        window[0] > core[0] - halo_m
        or window[1] > core[1] - halo_m
        or window[2] < core[2] + halo_m
        or window[3] < core[3] + halo_m
    ):
        raise ValueError(f"Orthophoto window must provide a {halo_m} m halo")
    return affine, window, core


def _core_slices(
    window_bounds: tuple[int, int, int, int],
    core_bounds: tuple[int, int, int, int],
) -> tuple[slice, slice]:
    west, _south, _east, north = window_bounds
    core_west, core_south, core_east, core_north = core_bounds
    row_start = north - core_north
    row_end = north - core_south
    column_start = core_west - west
    column_end = core_east - west
    return slice(row_start, row_end), slice(column_start, column_end)


def _multiscale_features(
    rgb: np.ndarray,
    row_slice: slice,
    column_slice: slice,
    window_sizes: Sequence[int],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    row_start, row_end = int(row_slice.start), int(row_slice.stop)
    column_start, column_end = int(column_slice.start), int(column_slice.stop)
    rows = np.arange(row_start, row_end, dtype=np.int64)
    columns = np.arange(column_start, column_end, dtype=np.int64)
    integral = np.pad(
        rgb.astype(np.uint64).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0), (0, 0)),
    )
    core = rgb[row_slice, column_slice].astype(np.int16)
    means: list[np.ndarray] = []
    textures: list[np.ndarray] = []
    for size in window_sizes:
        radius = (size - 1) // 2
        y0 = rows - radius
        y1 = rows + radius + 1
        x0 = columns - radius
        x1 = columns + radius + 1
        sums = (
            integral[y1[:, None], x1[None, :]]
            - integral[y0[:, None], x1[None, :]]
            - integral[y1[:, None], x0[None, :]]
            + integral[y0[:, None], x0[None, :]]
        )
        area = size * size
        mean = ((sums + area // 2) // area).astype(np.int16)
        texture = (np.abs(core - mean).sum(axis=2, dtype=np.int32) // 3).astype(
            np.int16
        )
        means.append(mean)
        textures.append(texture)
    return tuple(means), tuple(textures)


def _canonical_geometry(raw: Any, *, width_m: Any = None) -> BaseGeometry:
    geometry = force_2d(shape(raw))
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Semantic restriction geometry must be valid and non-empty")
    if isinstance(geometry, (LineString, MultiLineString)):
        try:
            width = float(width_m)
        except (TypeError, ValueError) as error:
            raise ValueError("Linear semantic restriction requires width_m") from error
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Linear semantic restriction width_m must be positive")
        geometry = geometry.buffer(width / 2.0, cap_style="flat", join_style="mitre")
    return set_precision(geometry, grid_size=0.001).normalize()


def _canonical_restrictions(
    entries: Iterable[Mapping[str, Any]],
    *,
    correction: bool,
    profile_ids: set[str],
    class_ids: set[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    id_key = "correction_id" if correction else "feature_id"
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("Every semantic restriction must be an object")
        identifier = str(raw.get(id_key, "")).strip()
        if (
            not identifier
            or identifier in identifiers
            or any(separator in identifier for separator in ("/", "\\", "\0"))
        ):
            raise ValueError(f"Semantic restriction {id_key} is invalid")
        identifiers.add(identifier)
        allowed_profiles = sorted(set(map(str, raw.get("allowed_profile_ids", []))))
        allowed_classes = sorted(set(map(str, raw.get("allowed_class_ids", []))))
        if bool(allowed_profiles) == bool(allowed_classes):
            raise ValueError(
                f"{identifier} must restrict exactly one of profiles or classes"
            )
        if (
            not set(allowed_profiles) <= profile_ids
            or not set(allowed_classes) <= class_ids
        ):
            raise ValueError(f"{identifier} references an unknown PBR profile or class")
        try:
            priority = int(raw.get("priority", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Semantic priority is invalid: {identifier}") from error
        if priority < 0 or priority > 1_000_000:
            raise ValueError(f"Semantic priority is invalid: {identifier}")
        geometry = _canonical_geometry(raw.get("geometry"), width_m=raw.get("width_m"))
        canonical: dict[str, Any] = {
            id_key: identifier,
            "priority": priority,
            "allowed_profile_ids": allowed_profiles,
            "allowed_class_ids": allowed_classes,
            "geometry": mapping(geometry),
        }
        if correction:
            if raw.get("approved") is not True:
                raise ValueError(f"Correction is not explicitly approved: {identifier}")
            canonical["approved"] = True
            canonical["approval_sha256"] = _require_sha256(
                raw.get("approval_sha256"), label=f"correction approval {identifier}"
            )
        output.append(canonical)
    return sorted(
        output,
        key=lambda entry: (
            int(entry["priority"]),
            str(entry[id_key]),
            _canonical_bytes(entry),
        ),
    )


def _restriction_raster(
    core_bounds: tuple[int, int, int, int],
    profile_table: Sequence[Mapping[str, Any]],
    context_priors: Sequence[Mapping[str, Any]],
    approved_corrections: Sequence[Mapping[str, Any]],
) -> tuple[
    np.ndarray,
    tuple[frozenset[int] | None, ...],
    tuple[str, ...],
    dict[str, int],
    str,
    str,
]:
    by_profile = {
        str(profile["id"]): index for index, profile in enumerate(profile_table)
    }
    by_class: dict[str, set[int]] = {}
    for index, profile in enumerate(profile_table):
        by_class.setdefault(str(profile["class_id"]), set()).add(index)
    priors = _canonical_restrictions(
        context_priors,
        correction=False,
        profile_ids=set(by_profile),
        class_ids=set(by_class),
    )
    corrections = _canonical_restrictions(
        approved_corrections,
        correction=True,
        profile_ids=set(by_profile),
        class_ids=set(by_class),
    )
    restrictions: list[frozenset[int] | None] = [None]
    shapes: list[tuple[dict[str, Any], int]] = []
    labels: list[str] = ["unrestricted"]
    # Priors are applied first; every approved correction is deliberately
    # applied afterwards and therefore overrides every contextual prior.
    for source, prefix in ((priors, "context"), (corrections, "correction")):
        for entry in source:
            allowed = {
                by_profile[profile_id] for profile_id in entry["allowed_profile_ids"]
            }
            for class_id in entry["allowed_class_ids"]:
                allowed.update(by_class[class_id])
            if not allowed:
                raise ValueError("A semantic restriction has no compatible PBR profile")
            restrictions.append(frozenset(allowed))
            code = len(restrictions) - 1
            shapes.append((entry["geometry"], code))
            labels.append(
                f"{prefix}:{entry['correction_id' if prefix == 'correction' else 'feature_id']}"
            )
    west, south, east, north = core_bounds
    height, width = north - south, east - west
    transform = Affine(1.0, 0.0, west, 0.0, -1.0, north)
    codes = (
        rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="uint32",
        )
        if shapes
        else np.zeros((height, width), dtype=np.uint32)
    )
    counts = {
        labels[code]: int(np.count_nonzero(codes == code))
        for code in range(len(labels))
        if np.any(codes == code)
    }
    return (
        codes,
        tuple(restrictions),
        tuple(labels),
        counts,
        _canonical_sha256(
            {"schema": "fireviewer.surface-context-priors.v1", "features": priors}
        ),
        _canonical_sha256(
            {
                "schema": "fireviewer.surface-approved-corrections.v1",
                "corrections": corrections,
            }
        ),
    )


def _tile_input_hashes(
    rgb: np.ndarray,
    window_bounds: tuple[int, int, int, int],
    core_bounds: tuple[int, int, int, int],
    halo_m: int,
) -> dict[str, str]:
    """Hash exact tile-plus-halo inputs independently of batch grouping."""

    window_west, _window_south, _window_east, window_north = window_bounds
    core_west, core_south, core_east, core_north = core_bounds
    output: dict[str, str] = {}
    for south in range(core_south, core_north, TILE_SIZE_M):
        north = south + TILE_SIZE_M
        for west in range(core_west, core_east, TILE_SIZE_M):
            east = west + TILE_SIZE_M
            source_bounds = (
                west - halo_m,
                south - halo_m,
                east + halo_m,
                north + halo_m,
            )
            row_start = window_north - source_bounds[3]
            row_end = window_north - source_bounds[1]
            column_start = source_bounds[0] - window_west
            column_end = source_bounds[2] - window_west
            pixels = rgb[row_start:row_end, column_start:column_end]
            identity = {
                "crs": CRS,
                "bounds_l93_m": list(source_bounds),
                "shape": list(pixels.shape),
                "pixels_sha256": _sha256_bytes(pixels.tobytes(order="C")),
            }
            output[f"{west},{south},{east},{north}"] = _canonical_sha256(identity)
    return output


def _scores(
    means: Sequence[np.ndarray],
    textures: Sequence[np.ndarray],
    model_profiles: Sequence[Mapping[str, Any]],
    restriction_codes: np.ndarray,
    restrictions: Sequence[frozenset[int] | None],
    rgb_weights: Sequence[int],
    texture_weights: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    height, width = restriction_codes.shape
    best_scores = np.full((height, width, 4), INFINITE_SCORE, dtype=np.int64)
    best_ids = np.full((height, width, 4), 255, dtype=np.uint8)
    category_best: dict[str, np.ndarray] = {}
    for profile_index, profile in enumerate(model_profiles):
        score = np.zeros((height, width), dtype=np.int64)
        for scale in range(3):
            prototype = np.asarray(profile["rgb_u8_by_scale"][scale], dtype=np.int16)
            rgb_distance = np.abs(means[scale] - prototype).sum(axis=2, dtype=np.int32)
            texture_distance = np.abs(
                textures[scale].astype(np.int32)
                - int(profile["texture_u8_by_scale"][scale])
            )
            score += rgb_distance.astype(np.int64) * int(rgb_weights[scale])
            score += texture_distance.astype(np.int64) * int(texture_weights[scale])
        for restriction_code, allowed in enumerate(restrictions):
            if allowed is not None and profile_index not in allowed:
                score[restriction_codes == restriction_code] = INFINITE_SCORE
        class_id = str(profile["class_id"])
        current_category = category_best.get(class_id)
        category_best[class_id] = (
            score.copy()
            if current_category is None
            else np.minimum(current_category, score)
        )
        candidate_scores = np.concatenate((best_scores, score[..., None]), axis=2)
        candidate_ids = np.concatenate(
            (
                best_ids,
                np.full((height, width, 1), profile_index, dtype=np.uint8),
            ),
            axis=2,
        )
        order = np.lexsort((candidate_ids, candidate_scores), axis=2)[..., :4]
        best_scores = np.take_along_axis(candidate_scores, order, axis=2)
        best_ids = np.take_along_axis(candidate_ids, order, axis=2)
    if np.any(best_scores[..., 0] >= INFINITE_SCORE):
        raise ValueError("Semantic restrictions leave pixels without a PBR profile")
    return best_scores, best_ids, category_best


def _quantize_weights(best_scores: np.ndarray, best_ids: np.ndarray) -> np.ndarray:
    valid = best_scores < INFINITE_SCORE
    deltas = np.where(valid, best_scores - best_scores[..., :1], 0)
    raw = np.where(valid, np.maximum(1, 65_536 // (1 + deltas)), 0).astype(np.int64)
    total = raw.sum(axis=2, keepdims=True)
    numerators = raw * 255
    weights = numerators // total
    remainders = numerators % total
    missing = 255 - weights.sum(axis=2)
    order = np.lexsort((best_ids, ~valid, -remainders), axis=2)
    for rank in range(4):
        channel = order[..., rank]
        increment = (missing > rank).astype(np.int64)
        np.put_along_axis(
            weights,
            channel[..., None],
            np.take_along_axis(weights, channel[..., None], axis=2)
            + increment[..., None],
            axis=2,
        )
    result = weights.astype(np.uint8)
    if not np.all(result.sum(axis=2, dtype=np.uint16) == 255):
        raise AssertionError("Profile weight quantization does not sum to 255")
    return result


def _confidence(category_best: Mapping[str, np.ndarray]) -> np.ndarray:
    if len(category_best) < 2:
        first = next(iter(category_best.values()))
        return np.full(first.shape, 255, dtype=np.uint8)
    stack = np.stack([category_best[key] for key in sorted(category_best)], axis=2)
    first_two = np.partition(stack, 1, axis=2)[..., :2]
    best = first_two.min(axis=2)
    second = first_two.max(axis=2)
    authoritative = second >= INFINITE_SCORE
    finite_second = np.where(authoritative, best, second)
    gap = np.maximum(0, finite_second - best)
    denominator = np.maximum(1, finite_second + 255)
    result = np.minimum(255, (gap * 255 + denominator // 2) // denominator)
    return np.where(authoritative, 255, result).astype(np.uint8)


def _fixed_orientation_u8(degrees: float) -> int:
    units = int(math.floor((degrees % 180.0) * 128.0 / 180.0 + 0.5)) % 128
    return (units * 255 + 63) // 127


def _image_orientation(
    rgb: np.ndarray, row_slice: slice, column_slice: slice
) -> tuple[np.ndarray, np.ndarray]:
    luminance = (
        rgb[..., 0].astype(np.int32) * 77
        + rgb[..., 1].astype(np.int32) * 150
        + rgb[..., 2].astype(np.int32) * 29
    )
    r0, r1 = int(row_slice.start), int(row_slice.stop)
    c0, c1 = int(column_slice.start), int(column_slice.stop)
    top = luminance[r0 - 1 : r1 - 1]
    middle = luminance[r0:r1]
    bottom = luminance[r0 + 1 : r1 + 1]
    gx = (
        top[:, c0 + 1 : c1 + 1]
        + 2 * middle[:, c0 + 1 : c1 + 1]
        + bottom[:, c0 + 1 : c1 + 1]
        - top[:, c0 - 1 : c1 - 1]
        - 2 * middle[:, c0 - 1 : c1 - 1]
        - bottom[:, c0 - 1 : c1 - 1]
    )
    gy = (
        bottom[:, c0 - 1 : c1 - 1]
        + 2 * bottom[:, c0:c1]
        + bottom[:, c0 + 1 : c1 + 1]
        - top[:, c0 - 1 : c1 - 1]
        - 2 * top[:, c0:c1]
        - top[:, c0 + 1 : c1 + 1]
    )
    ax, ay = np.abs(gx), np.abs(gy)
    nonzero = (ax + ay) > 0
    q = np.zeros_like(ax, dtype=np.int64)
    x_major = (ax >= ay) & nonzero
    y_major = (ay > ax) & nonzero
    q[x_major] = (ay[x_major] * 32 + ax[x_major] // 2) // ax[x_major]
    q[y_major] = 64 - (ax[y_major] * 32 + ay[y_major] // 2) // ay[y_major]
    gradient = np.where((gx < 0) != (gy < 0), 128 - q, q) % 128
    tangent = (gradient + 64) % 128
    encoded = ((tangent * 255 + 63) // 127).astype(np.uint8)
    return encoded, nonzero


def _orientation(
    rgb: np.ndarray,
    row_slice: slice,
    column_slice: slice,
    best_ids: np.ndarray,
    model_profiles: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    image_orientation, has_gradient = _image_orientation(rgb, row_slice, column_slice)
    fixed = np.asarray(
        [
            _fixed_orientation_u8(float(profile["default_orientation_deg"]))
            for profile in model_profiles
        ],
        dtype=np.uint8,
    )
    uses_image = np.asarray(
        [
            profile["orientation_mode"] == "image_structure"
            for profile in model_profiles
        ],
        dtype=bool,
    )
    primary = best_ids[..., 0]
    return np.where(
        uses_image[primary] & has_gradient, image_orientation, fixed[primary]
    ).astype(np.uint8)


def compile_aligned_window(
    rgb_u8: np.ndarray,
    *,
    transform: Affine | Sequence[float],
    crs: str,
    core_bounds_l93_m: Sequence[Any],
    orthophoto_sha256: str,
    pbr_library: Mapping[str, Any],
    pbr_library_sha256: str | None = None,
    correspondence_model: Mapping[str, Any],
    model_sha256: str | None = None,
    context_priors: Sequence[Mapping[str, Any]] = (),
    approved_corrections: Sequence[Mapping[str, Any]] = (),
    contract_path: Path | None = None,
) -> SurfaceCorrespondence:
    """Classify a globally aligned RGB window and return its halo-free core.

    The caller may compile one tile or a rectangular group of tiles.  Reusing
    the same source pixels with the required halo produces bit-identical tile
    outputs regardless of grouping or worker order.
    """

    contract, resolved_contract = _load_contract(contract_path)
    halo_m = int(contract["input"]["processing_halo_m"])
    _affine, window_bounds, core_bounds = _validate_grid(
        rgb_u8, transform, crs, core_bounds_l93_m, halo_m
    )
    raw_profile_table, library_digest = _validate_pbr_library(
        pbr_library, pbr_library_sha256
    )
    model_profiles, model_digest = _validate_model(
        correspondence_model, raw_profile_table, model_sha256
    )
    profile_table = tuple(
        {**profile, "class_id": str(model_profiles[index]["class_id"])}
        for index, profile in enumerate(raw_profile_table)
    )
    row_slice, column_slice = _core_slices(window_bounds, core_bounds)
    sizes = tuple(
        int(value) for value in contract["classification"]["feature_window_sizes_m"]
    )
    means, textures = _multiscale_features(rgb_u8, row_slice, column_slice, sizes)
    (
        restriction_codes,
        restrictions,
        restriction_labels,
        restriction_counts,
        priors_hash,
        corrections_hash,
    ) = _restriction_raster(
        core_bounds,
        profile_table,
        context_priors,
        approved_corrections,
    )
    rgb_weights = tuple(
        int(value) for value in contract["classification"]["rgb_l1_weights"]
    )
    texture_weights = tuple(
        int(value) for value in contract["classification"]["texture_l1_weights"]
    )
    best_scores, best_ids, category_best = _scores(
        means,
        textures,
        model_profiles,
        restriction_codes,
        restrictions,
        rgb_weights,
        texture_weights,
    )
    weights = _quantize_weights(best_scores, best_ids)
    best_ids = np.where(
        best_scores < INFINITE_SCORE, best_ids, best_ids[..., :1]
    ).astype(np.uint8)
    confidence = _confidence(category_best)
    orientation = _orientation(
        rgb_u8, row_slice, column_slice, best_ids, model_profiles
    )
    identity = {
        "orthophoto_source_sha256": _require_sha256(
            orthophoto_sha256, label="orthophoto source"
        ),
        "pbr_library_sha256": library_digest,
        "correspondence_model_sha256": model_digest,
        "algorithm_sha256": _sha256_file(Path(__file__).resolve()),
        "contract_sha256": _sha256_file(resolved_contract),
        "context_priors_sha256": priors_hash,
        "approved_corrections_sha256": corrections_hash,
    }
    return SurfaceCorrespondence(
        core_bounds_l93_m=core_bounds,
        profile_ids=best_ids,
        profile_weights=weights,
        confidence=confidence,
        orientation=orientation,
        identity=identity,
        profile_table=profile_table,
        class_by_profile_index=tuple(
            str(profile["class_id"]) for profile in profile_table
        ),
        restriction_counts=restriction_counts,
        restriction_codes=restriction_codes,
        restriction_labels=restriction_labels,
        tile_input_sha256_by_bounds=_tile_input_hashes(
            rgb_u8, window_bounds, core_bounds, halo_m
        ),
    )


def slice_tile(
    correspondence: SurfaceCorrespondence,
    tile_bounds_l93_m: Sequence[Any],
) -> TileCorrespondence:
    """Extract one exact 500 m tile from a globally classified core."""

    bounds = _integer_bounds(tile_bounds_l93_m, label="tile bounds")
    if bounds[2] - bounds[0] != TILE_SIZE_M or bounds[3] - bounds[1] != TILE_SIZE_M:
        raise ValueError("Correspondence tile must be exactly 500 m square")
    if any(value % TILE_SIZE_M for value in bounds):
        raise ValueError("Correspondence tile must align to the global 500 m grid")
    core = correspondence.core_bounds_l93_m
    if (
        bounds[0] < core[0]
        or bounds[1] < core[1]
        or bounds[2] > core[2]
        or bounds[3] > core[3]
    ):
        raise ValueError("Correspondence tile lies outside the classified core")
    row_start = core[3] - bounds[3]
    row_end = core[3] - bounds[1]
    column_start = bounds[0] - core[0]
    column_end = bounds[2] - core[0]
    rows, columns = slice(row_start, row_end), slice(column_start, column_end)
    tile_key = ",".join(str(value) for value in bounds)
    identity = dict(correspondence.identity)
    identity["orthophoto_tile_input_sha256"] = (
        correspondence.tile_input_sha256_by_bounds[tile_key]
    )
    tile_restriction_codes = correspondence.restriction_codes[rows, columns]
    restriction_counts = {
        correspondence.restriction_labels[code]: int(
            np.count_nonzero(tile_restriction_codes == code)
        )
        for code in range(len(correspondence.restriction_labels))
        if np.any(tile_restriction_codes == code)
    }
    return TileCorrespondence(
        bounds_l93_m=bounds,
        profile_ids=correspondence.profile_ids[rows, columns].copy(),
        profile_weights=correspondence.profile_weights[rows, columns].copy(),
        confidence=correspondence.confidence[rows, columns].copy(),
        orientation=correspondence.orientation[rows, columns].copy(),
        identity=identity,
        profile_table=correspondence.profile_table,
        class_by_profile_index=correspondence.class_by_profile_index,
        restriction_counts=restriction_counts,
    )


def _png_bytes(array: np.ndarray, mode: str) -> bytes:
    stream = BytesIO()
    Image.fromarray(array, mode=mode).save(
        stream, format="PNG", optimize=False, compress_level=9
    )
    return stream.getvalue()


def serialize_tile_outputs(tile: TileCorrespondence) -> dict[str, bytes]:
    """Serialize only runtime correspondence maps and their compact manifest."""

    if tile.profile_ids.shape != (500, 500, 4) or tile.profile_ids.dtype != np.uint8:
        raise ValueError("Profile ID output must be a 500 x 500 RGBA8 array")
    if (
        tile.profile_weights.shape != (500, 500, 4)
        or tile.profile_weights.dtype != np.uint8
    ):
        raise ValueError("Profile weight output must be a 500 x 500 RGBA8 array")
    if not np.all(tile.profile_weights.sum(axis=2, dtype=np.uint16) == 255):
        raise ValueError("Every profile weight pixel must sum exactly to 255")
    if tile.confidence.shape != (500, 500) or tile.confidence.dtype != np.uint8:
        raise ValueError("Confidence output must be a 500 x 500 L8 array")
    if tile.orientation.shape != (500, 500) or tile.orientation.dtype != np.uint8:
        raise ValueError("Orientation output must be a 500 x 500 L8 array")
    maps = {
        "ground-profile-ids.png": _png_bytes(tile.profile_ids, "RGBA"),
        "ground-profile-weights.png": _png_bytes(tile.profile_weights, "RGBA"),
        "ground-confidence.png": _png_bytes(tile.confidence, "L"),
        "ground-orientation.png": _png_bytes(tile.orientation, "L"),
    }
    selected_indices, pixel_counts = np.unique(
        tile.profile_ids[..., 0], return_counts=True
    )
    class_counts: dict[str, int] = {}
    for index, count in zip(
        selected_indices.tolist(), pixel_counts.tolist(), strict=True
    ):
        class_id = tile.class_by_profile_index[int(index)]
        class_counts[class_id] = class_counts.get(class_id, 0) + int(count)
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "status": "compiled_no_orthophoto_payload",
        "crs": CRS,
        "bounds_l93_m": list(tile.bounds_l93_m),
        "grid": {"resolution_m": 1, "width": 500, "height": 500},
        "identity": dict(sorted(tile.identity.items())),
        "runtime": {
            "orthophoto_dependency": "forbidden",
            "orthophoto_pixels_present": False,
            "procedural_materials": "forbidden",
            "projection": "profile_declared_world_xy_or_world_triplanar",
        },
        "profile_id_encoding": "RGBA8 zero-based profile_table index",
        "profile_weight_encoding": "RGBA8 sum exactly 255",
        "confidence_encoding": "L8 best-versus-next-semantic-class margin",
        "orientation_encoding": "L8 world texture orientation modulo 180 degrees",
        "profile_table": list(tile.profile_table),
        "primary_pixel_counts_by_class": dict(sorted(class_counts.items())),
        "restriction_pixel_counts": dict(sorted(tile.restriction_counts.items())),
        "artifacts": {
            name: {"byte_count": len(payload), "sha256": _sha256_bytes(payload)}
            for name, payload in sorted(maps.items())
        },
    }
    maps["surface-correspondence.json"] = _canonical_bytes(manifest) + b"\n"
    return maps


def write_tile_outputs(tile: TileCorrespondence, output_dir: Path) -> dict[str, str]:
    """Atomically publish one tile on D: without retaining its orthophoto."""

    destination = Path(output_dir).resolve(strict=False)
    if os.name == "nt" and destination.drive.upper() != "D:":
        raise ValueError("Surface correspondence outputs must stay on D:")
    if destination.exists():
        raise FileExistsError("Surface correspondence output directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    if staging.exists():
        raise FileExistsError("Surface correspondence staging directory already exists")
    staging.mkdir()
    try:
        outputs = serialize_tile_outputs(tile)
        if set(outputs) != set(OUTPUT_NAMES):
            raise AssertionError("Surface correspondence output set changed")
        for name, payload in outputs.items():
            (staging / name).write_bytes(payload)
        staging.replace(destination)
        return {
            name: _sha256_bytes(payload) for name, payload in sorted(outputs.items())
        }
    except Exception:
        for item in staging.iterdir() if staging.exists() else ():
            item.unlink(missing_ok=True)
        staging.rmdir() if staging.exists() else None
        raise


__all__ = [
    "CONTRACT_SCHEMA",
    "LIBRARY_SCHEMA",
    "MODEL_SCHEMA",
    "OUTPUT_SCHEMA",
    "SurfaceCorrespondence",
    "TileCorrespondence",
    "compile_aligned_window",
    "serialize_tile_outputs",
    "slice_tile",
    "write_tile_outputs",
]
