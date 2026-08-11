"""Validate the clean PBR ground-texture library and its tile control maps.

The validator is intentionally independent from Blender.  It accepts only a
fresh, hash-locked four-role PBR library for the 72 stable FireViewer ground
profiles.  Legacy atlases, packaged orthophotos and procedural material bases
are not valid inputs.  A hash-locked orthophoto may only be used temporarily
for build-time correspondence and must not enter this library.  Human visual
acceptance remains a separate, hash-bound gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


CONTRACT_SCHEMA = "fireviewer.ground-surface-texture-contract.v4"
CONTRACT_STATUS = "pending_clean_pbr_library"
LIBRARY_SCHEMA = "fireviewer.clean-pbr-texture-library.v1"
PENDING_LIBRARY_STATUS = "generated_pending_visual_review"
ACCEPTED_LIBRARY_STATUS = "accepted_clean_pbr_library"
VISUAL_RECEIPT_SCHEMA = "fireviewer.clean-pbr-texture-visual-acceptance.v1"
REQUIRED_TEXTURE_ROLES = ("basecolor", "normal", "height", "orm")
RUNTIME_MAP_ROLES = ("ids", "weights", "confidence", "orientation")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ORTHO_TOKEN = re.compile(r"(^|[^a-z0-9])ortho(?:photo)?s?([^a-z0-9]|$)")


class CleanPbrTextureLibraryError(ValueError):
    """A clean texture contract, library or runtime map is invalid."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise CleanPbrTextureLibraryError(
            f"{label} keys differ; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise CleanPbrTextureLibraryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CleanPbrTextureLibraryError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise CleanPbrTextureLibraryError(f"{label} must be a JSON object")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CleanPbrTextureLibraryError(f"{label} must be a positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CleanPbrTextureLibraryError(
            f"{label} must be a positive number"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise CleanPbrTextureLibraryError(f"{label} must be a positive number")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CleanPbrTextureLibraryError(f"{label} must be a positive integer")
    return value


def _contains_orthophoto_token(value: str) -> bool:
    return ORTHO_TOKEN.search(value.casefold().replace("\\", "/")) is not None


def _portable_artifact_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CleanPbrTextureLibraryError(
            f"{label}.path must be a non-empty portable relative path"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.parts[0].endswith(":"):
        raise CleanPbrTextureLibraryError(f"{label}.path escapes the library")
    if _contains_orthophoto_token(value):
        raise CleanPbrTextureLibraryError(
            f"{label}.path contains a forbidden orthophoto token"
        )
    candidate = root.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise CleanPbrTextureLibraryError(
            f"{label}.path resolves outside the library"
        ) from error
    return candidate


def _validate_artifact(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise CleanPbrTextureLibraryError(f"{label} must be an artifact object")
    _require_exact_keys(value, {"path", "byte_count", "sha256"}, label)
    path = _portable_artifact_path(root, value["path"], label)
    if not path.is_file():
        raise CleanPbrTextureLibraryError(f"{label} is missing: {path}")
    byte_count = value["byte_count"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise CleanPbrTextureLibraryError(
            f"{label}.byte_count must be a positive integer"
        )
    expected_hash = _require_sha256(value["sha256"], f"{label}.sha256")
    if byte_count != path.stat().st_size or expected_hash != sha256_file(path):
        raise CleanPbrTextureLibraryError(f"{label} size or SHA-256 mismatch")
    return path


def _png_header(path: Path, label: str) -> dict[str, int]:
    header = path.read_bytes()[:33]
    if len(header) < 33 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise CleanPbrTextureLibraryError(f"{label} is not a PNG")
    width, height, bit_depth, colour_type = struct.unpack(">IIBB", header[16:26])
    if width <= 0 or height <= 0:
        raise CleanPbrTextureLibraryError(f"{label} has invalid dimensions")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError) as error:
        raise CleanPbrTextureLibraryError(f"{label} is a corrupt PNG") from error
    return {
        "width_px": width,
        "height_px": height,
        "bit_depth": bit_depth,
        "colour_type": colour_type,
    }


def _expected_png_encoding(role: str) -> tuple[int, int]:
    if role == "height":
        return 16, 0
    if role in {"basecolor", "normal", "orm"}:
        return 8, 6
    raise AssertionError(f"Unknown PBR role: {role}")


def _validate_pbr_png(
    root: Path,
    record: Any,
    *,
    role: str,
    label: str,
    expected_size: tuple[int, int] | None = None,
) -> Path:
    path = _validate_artifact(root, record, label)
    if path.suffix.casefold() != ".png":
        raise CleanPbrTextureLibraryError(f"{label} must be a PNG")
    header = _png_header(path, label)
    expected_bit_depth, expected_colour_type = _expected_png_encoding(role)
    if (
        header["bit_depth"] != expected_bit_depth
        or header["colour_type"] != expected_colour_type
    ):
        raise CleanPbrTextureLibraryError(
            f"{label} does not match the {role} channel contract"
        )
    if (
        expected_size is not None
        and (header["width_px"], header["height_px"]) != expected_size
    ):
        raise CleanPbrTextureLibraryError(
            f"{label} dimensions differ from the atlas contract"
        )
    return path


def load_texture_contract(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the immutable profile and texture-layout contract."""

    contract_path = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("ground_surface_texture_contract.v4.json")
    ).resolve()
    contract = _read_json(contract_path, "ground texture contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise CleanPbrTextureLibraryError("Unsupported ground texture contract")
    if contract.get("status") != CONTRACT_STATUS:
        raise CleanPbrTextureLibraryError(
            f"Ground texture contract must remain {CONTRACT_STATUS!r}"
        )
    if contract.get("crs") != "EPSG:2154":
        raise CleanPbrTextureLibraryError("Ground texture CRS must be EPSG:2154")
    orthophoto = contract.get("orthophoto_policy")
    if (
        not isinstance(orthophoto, Mapping)
        or orthophoto.get("build_time_source")
        != "allowed_temporary_correspondence_only"
    ):
        raise CleanPbrTextureLibraryError(
            "Build-time orthophotos may only be temporary correspondence sources"
        )
    if orthophoto.get("build_time_requirements") != [
        "source_hash_locked",
        "excluded_from_clean_pbr_library",
        "deleted_after_package_sealing",
    ]:
        raise CleanPbrTextureLibraryError(
            "Temporary orthophoto correspondence requirements are incomplete"
        )
    if (
        orthophoto.get("runtime_dependency") != "forbidden"
        or orthophoto.get("package_artifact") != "forbidden"
    ):
        raise CleanPbrTextureLibraryError(
            "Orthophotos must be forbidden in runtime dependencies and packages"
        )
    freshness = contract.get("fresh_library_policy")
    if (
        not isinstance(freshness, Mapping)
        or freshness.get("accepted_legacy_atlas_hashes") != []
    ):
        raise CleanPbrTextureLibraryError("Legacy atlas hashes must not be accepted")
    for field in ("legacy_atlas_dependency", "rejected_atlas_v3_hash_reuse"):
        if freshness.get(field) != "forbidden":
            raise CleanPbrTextureLibraryError(f"{field} must be forbidden")

    atlas = contract.get("runtime_atlas")
    if not isinstance(atlas, Mapping):
        raise CleanPbrTextureLibraryError("Runtime atlas contract is missing")
    if tuple(atlas.get("roles", ())) != REQUIRED_TEXTURE_ROLES:
        raise CleanPbrTextureLibraryError("Exactly four PBR atlas roles are required")
    if atlas.get("texture_count") != 4:
        raise CleanPbrTextureLibraryError("Runtime atlas texture_count must be four")
    columns = _positive_integer(atlas.get("columns"), "runtime_atlas.columns")
    rows = _positive_integer(atlas.get("rows"), "runtime_atlas.rows")
    slots = _positive_integer(atlas.get("slot_count"), "runtime_atlas.slot_count")
    used = _positive_integer(
        atlas.get("used_slot_count"), "runtime_atlas.used_slot_count"
    )
    cell = _positive_integer(atlas.get("cell_size_px"), "runtime_atlas.cell_size_px")
    gutter = atlas.get("gutter_px")
    if isinstance(gutter, bool) or not isinstance(gutter, int) or gutter < 0:
        raise CleanPbrTextureLibraryError(
            "runtime_atlas.gutter_px must be non-negative"
        )
    if (
        slots != columns * rows
        or used != 72
        or atlas.get("width_px") != columns * cell
        or atlas.get("height_px") != rows * cell
        or cell <= 2 * gutter
    ):
        raise CleanPbrTextureLibraryError("Runtime atlas layout is inconsistent")

    profiles = contract.get("profile_contract")
    if not isinstance(profiles, Mapping):
        raise CleanPbrTextureLibraryError("Profile contract is missing")
    identifiers = profiles.get("stable_profile_ids")
    if (
        not isinstance(identifiers, list)
        or len(identifiers) != 72
        or profiles.get("profile_count") != 72
        or len(set(identifiers)) != 72
        or any(
            not isinstance(identifier, str) or "." not in identifier
            for identifier in identifiers
        )
    ):
        raise CleanPbrTextureLibraryError(
            "The contract must lock exactly 72 unique stable profile IDs"
        )
    if profiles.get("allowed_surface_basis") != "atlas_pbr":
        raise CleanPbrTextureLibraryError("Only atlas_pbr profiles are permitted")
    forbidden_bases = set(profiles.get("forbidden_surface_bases", ()))
    if forbidden_bases != {
        "procedural_only",
        "procedural_tint_with_atlas_relief",
        "procedural_water_over_contextual_bed",
    }:
        raise CleanPbrTextureLibraryError(
            "All procedural surface bases must be explicitly forbidden"
        )
    if tuple(profiles.get("required_texture_roles", ())) != REQUIRED_TEXTURE_ROLES:
        raise CleanPbrTextureLibraryError("Every profile must require four PBR roles")

    maps = contract.get("tile_runtime_maps")
    if (
        not isinstance(maps, Mapping)
        or maps.get("width_px") != 500
        or maps.get("height_px") != 500
    ):
        raise CleanPbrTextureLibraryError("Tile runtime maps must be exactly 500x500")
    if (
        maps.get("runtime_vector_dependency") != "forbidden"
        or maps.get("orthophoto_dependency") != "forbidden"
    ):
        raise CleanPbrTextureLibraryError(
            "Runtime vector and orthophoto dependencies must be forbidden"
        )
    map_contract = maps.get("maps")
    if not isinstance(map_contract, Mapping) or set(map_contract) != set(
        RUNTIME_MAP_ROLES
    ):
        raise CleanPbrTextureLibraryError(
            "Runtime IDs, weights, confidence and orientation maps are required"
        )
    expected_modes = {
        "ids": "RGBA8",
        "weights": "RGBA8",
        "confidence": "L8",
        "orientation": "L8",
    }
    for role, mode in expected_modes.items():
        if map_contract[role].get("mode") != mode:
            raise CleanPbrTextureLibraryError(f"Runtime map {role} must use {mode}")
    return contract


def _expected_atlas_uv(
    contract: Mapping[str, Any], slot: int
) -> dict[str, list[float]]:
    atlas = contract["runtime_atlas"]
    columns = int(atlas["columns"])
    cell = int(atlas["cell_size_px"])
    gutter = int(atlas["gutter_px"])
    width = int(atlas["width_px"])
    height = int(atlas["height_px"])
    column = slot % columns
    row = slot // columns
    return {
        "offset": [
            (column * cell + gutter) / width,
            (row * cell + gutter) / height,
        ],
        "scale": [
            (cell - 2 * gutter) / width,
            (cell - 2 * gutter) / height,
        ],
    }


def _validate_atlas_uv(
    value: Any, expected: Mapping[str, Sequence[float]], label: str
) -> None:
    if not isinstance(value, Mapping):
        raise CleanPbrTextureLibraryError(f"{label} must be an atlas UV object")
    _require_exact_keys(value, {"offset", "scale"}, label)
    for field in ("offset", "scale"):
        actual = value[field]
        target = expected[field]
        if (
            not isinstance(actual, list)
            or len(actual) != 2
            or any(
                not math.isclose(float(component), float(reference), abs_tol=1.0e-12)
                for component, reference in zip(actual, target, strict=True)
            )
        ):
            raise CleanPbrTextureLibraryError(
                f"{label}.{field} differs from the stable atlas slot"
            )


def _validate_visual_acceptance(
    root: Path,
    value: Any,
    *,
    contract_sha256: str,
    library_content_sha256: str,
) -> None:
    if not isinstance(value, Mapping):
        raise CleanPbrTextureLibraryError("visual_acceptance must be an object")
    _require_exact_keys(value, {"status", "receipt"}, "visual_acceptance")
    if value.get("status") != "accepted_human_visual":
        raise CleanPbrTextureLibraryError(
            "Clean PBR library lacks human visual acceptance"
        )
    receipt_path = _validate_artifact(
        root, value.get("receipt"), "visual_acceptance.receipt"
    )
    receipt = _read_json(receipt_path, "clean PBR visual acceptance receipt")
    expected_keys = {
        "schema",
        "status",
        "texture_contract_sha256",
        "library_content_sha256",
        "profile_count",
        "atlas_roles",
        "invalid_profile_count",
    }
    _require_exact_keys(receipt, expected_keys, "visual acceptance receipt")
    if (
        receipt.get("schema") != VISUAL_RECEIPT_SCHEMA
        or receipt.get("status") != "accepted_human_visual"
        or receipt.get("texture_contract_sha256") != contract_sha256
        or receipt.get("library_content_sha256") != library_content_sha256
        or receipt.get("profile_count") != 72
        or tuple(receipt.get("atlas_roles", ())) != REQUIRED_TEXTURE_ROLES
        or receipt.get("invalid_profile_count") != 0
    ):
        raise CleanPbrTextureLibraryError(
            "Visual acceptance receipt is incomplete or bound to another library"
        )


def validate_texture_library(
    library_path: Path | str,
    *,
    contract_path: Path | str | None = None,
    require_visual_acceptance: bool = False,
) -> dict[str, Any]:
    """Validate every clean PBR artifact and the exact 72-profile identity."""

    contract_file = (
        Path(contract_path)
        if contract_path is not None
        else Path(__file__).with_name("ground_surface_texture_contract.v4.json")
    ).resolve()
    contract = load_texture_contract(contract_file)
    contract_sha256 = sha256_file(contract_file)
    manifest_path = Path(library_path).resolve()
    library = _read_json(manifest_path, "clean PBR texture library")
    root = manifest_path.parent
    expected_keys = {
        "schema",
        "status",
        "texture_contract_sha256",
        "orthophoto_dependency",
        "lineage",
        "runtime_atlases",
        "profiles",
        "visual_acceptance",
    }
    _require_exact_keys(library, expected_keys, "clean PBR texture library")
    if library.get("schema") != LIBRARY_SCHEMA:
        raise CleanPbrTextureLibraryError("Unsupported clean PBR library schema")
    status = library.get("status")
    if status not in {PENDING_LIBRARY_STATUS, ACCEPTED_LIBRARY_STATUS}:
        raise CleanPbrTextureLibraryError("Unsupported clean PBR library status")
    if library.get("texture_contract_sha256") != contract_sha256:
        raise CleanPbrTextureLibraryError(
            "Clean PBR library is bound to another texture contract"
        )
    if library.get("orthophoto_dependency") != "forbidden":
        raise CleanPbrTextureLibraryError("Clean PBR library must forbid orthophotos")
    lineage = library.get("lineage")
    if not isinstance(lineage, Mapping):
        raise CleanPbrTextureLibraryError("Clean PBR lineage is missing")
    _require_exact_keys(
        lineage,
        {"source", "legacy_atlas_dependencies"},
        "clean PBR lineage",
    )
    if (
        lineage.get("source") != "fresh_clean_pbr_v4"
        or lineage.get("legacy_atlas_dependencies") != []
    ):
        raise CleanPbrTextureLibraryError(
            "Clean PBR library must not depend on the rejected legacy atlas"
        )

    atlases = library.get("runtime_atlases")
    if not isinstance(atlases, Mapping) or set(atlases) != set(REQUIRED_TEXTURE_ROLES):
        raise CleanPbrTextureLibraryError("Exactly four runtime atlases are required")
    atlas_size = (
        int(contract["runtime_atlas"]["width_px"]),
        int(contract["runtime_atlas"]["height_px"]),
    )
    atlas_paths = {
        role: _validate_pbr_png(
            root,
            atlases[role],
            role=role,
            label=f"runtime_atlases.{role}",
            expected_size=atlas_size,
        )
        for role in REQUIRED_TEXTURE_ROLES
    }
    if len(set(atlas_paths.values())) != 4:
        raise CleanPbrTextureLibraryError("Runtime atlas paths must be unique")

    profiles = library.get("profiles")
    stable_ids = contract["profile_contract"]["stable_profile_ids"]
    if not isinstance(profiles, list) or len(profiles) != len(stable_ids):
        raise CleanPbrTextureLibraryError("Clean PBR library must contain 72 profiles")
    minimum_scale = float(contract["profile_contract"]["physical_scale_m"]["minimum"])
    maximum_scale = float(contract["profile_contract"]["physical_scale_m"]["maximum"])
    source_paths: set[Path] = set()
    for index, (profile, expected_id) in enumerate(
        zip(profiles, stable_ids, strict=True)
    ):
        if not isinstance(profile, Mapping):
            raise CleanPbrTextureLibraryError(f"profiles[{index}] must be an object")
        _require_exact_keys(
            profile,
            {
                "stable_index",
                "id",
                "surface_basis",
                "atlas_slot",
                "atlas_uv",
                "physical_scale_m",
                "projection",
                "source_kind",
                "textures",
            },
            f"profiles[{index}]",
        )
        if profile.get("stable_index") != index or profile.get("id") != expected_id:
            raise CleanPbrTextureLibraryError(
                f"profiles[{index}] differs from the stable profile identity"
            )
        if profile.get("surface_basis") != "atlas_pbr":
            raise CleanPbrTextureLibraryError(
                f"profiles[{index}] must use atlas_pbr, never a procedural basis"
            )
        if profile.get("atlas_slot") != index:
            raise CleanPbrTextureLibraryError(
                f"profiles[{index}].atlas_slot must equal its stable index"
            )
        _validate_atlas_uv(
            profile.get("atlas_uv"),
            _expected_atlas_uv(contract, index),
            f"profiles[{index}].atlas_uv",
        )
        scale = _finite_positive(
            profile.get("physical_scale_m"),
            f"profiles[{index}].physical_scale_m",
        )
        if not minimum_scale <= scale <= maximum_scale:
            raise CleanPbrTextureLibraryError(
                f"profiles[{index}].physical_scale_m is outside the contract"
            )
        expected_projection = (
            "world_triplanar"
            if expected_id.startswith("cliff_surface.")
            else "world_xy"
        )
        if profile.get("projection") != expected_projection:
            raise CleanPbrTextureLibraryError(
                f"profiles[{index}].projection must be {expected_projection}"
            )
        if profile.get("source_kind") != "clean_pbr_profile_texture":
            raise CleanPbrTextureLibraryError(
                f"profiles[{index}] has a forbidden source kind"
            )
        textures = profile.get("textures")
        if not isinstance(textures, Mapping) or set(textures) != set(
            REQUIRED_TEXTURE_ROLES
        ):
            raise CleanPbrTextureLibraryError(
                f"profiles[{index}] must contain all four PBR texture roles"
            )
        for role in REQUIRED_TEXTURE_ROLES:
            source_path = _validate_pbr_png(
                root,
                textures[role],
                role=role,
                label=f"profiles[{index}].textures.{role}",
            )
            if source_path in source_paths:
                raise CleanPbrTextureLibraryError(
                    "Every profile texture role must use its own source artifact path"
                )
            source_paths.add(source_path)

    content_payload = dict(library)
    content_payload.pop("status")
    content_payload.pop("visual_acceptance")
    library_content_sha256 = _sha256_bytes(_canonical_bytes(content_payload))
    visual = library.get("visual_acceptance")
    if status == ACCEPTED_LIBRARY_STATUS or require_visual_acceptance:
        _validate_visual_acceptance(
            root,
            visual,
            contract_sha256=contract_sha256,
            library_content_sha256=library_content_sha256,
        )
    else:
        if visual != {"status": "pending_human_visual_review", "receipt": None}:
            raise CleanPbrTextureLibraryError(
                "A pending clean PBR library must not claim visual acceptance"
            )

    return {
        "schema": LIBRARY_SCHEMA,
        "status": status,
        "texture_contract_sha256": contract_sha256,
        "library_content_sha256": library_content_sha256,
        "profile_count": len(profiles),
        "runtime_atlas_count": len(atlases),
        "source_texture_count": len(source_paths),
        "visual_acceptance": visual["status"],
    }


def validate_runtime_maps(
    map_root: Path | str,
    *,
    contract_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one tile's exact 500x500 IDs/weights/confidence/orientation maps."""

    contract = load_texture_contract(contract_path)
    root = Path(map_root).resolve()
    map_contract = contract["tile_runtime_maps"]["maps"]
    size = (
        int(contract["tile_runtime_maps"]["width_px"]),
        int(contract["tile_runtime_maps"]["height_px"]),
    )
    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for role in RUNTIME_MAP_ROLES:
        file_name = map_contract[role]["file"]
        if _contains_orthophoto_token(file_name):
            raise CleanPbrTextureLibraryError(
                f"Runtime map {role} contains a forbidden orthophoto token"
            )
        path = (root / file_name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise CleanPbrTextureLibraryError(
                f"Runtime map {role} escapes its tile"
            ) from error
        if not path.is_file():
            raise CleanPbrTextureLibraryError(f"Runtime map is missing: {file_name}")
        try:
            with Image.open(path) as image:
                if image.size != size:
                    raise CleanPbrTextureLibraryError(
                        f"Runtime map {role} must be exactly 500x500"
                    )
                expected_mode = "RGBA" if role in {"ids", "weights"} else "L"
                if image.mode != expected_mode:
                    raise CleanPbrTextureLibraryError(
                        f"Runtime map {role} must use {map_contract[role]['mode']}"
                    )
                arrays[role] = np.asarray(image, dtype=np.uint8).copy()
        except OSError as error:
            raise CleanPbrTextureLibraryError(
                f"Runtime map {role} is a corrupt PNG"
            ) from error
        hashes[role] = sha256_file(path)

    weights = arrays["weights"]
    if not np.all(weights.sum(axis=2, dtype=np.uint16) == 255):
        raise CleanPbrTextureLibraryError(
            "Runtime profile weights must sum exactly to 255 per pixel"
        )
    weighted_ids = arrays["ids"][weights > 0]
    if weighted_ids.size == 0 or int(weighted_ids.max()) >= 72:
        raise CleanPbrTextureLibraryError(
            "Runtime IDs reference a profile outside the stable 0..71 range"
        )
    return {
        "width_px": size[0],
        "height_px": size[1],
        "profile_count": 72,
        "maps_sha256": hashes,
        "confidence_range": [
            int(arrays["confidence"].min()),
            int(arrays["confidence"].max()),
        ],
        "orientation_encoding": map_contract["orientation"]["encoding"],
    }


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a fresh 72-profile FireViewer PBR texture library."
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("ground_surface_texture_contract.v4.json"),
    )
    parser.add_argument("--runtime-maps", type=Path)
    parser.add_argument("--require-visual-acceptance", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)
    result: dict[str, Any] = {
        "library": validate_texture_library(
            options.library,
            contract_path=options.contract,
            require_visual_acceptance=options.require_visual_acceptance,
        )
    }
    if options.runtime_maps is not None:
        result["runtime_maps"] = validate_runtime_maps(
            options.runtime_maps,
            contract_path=options.contract,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTED_LIBRARY_STATUS",
    "CONTRACT_SCHEMA",
    "CONTRACT_STATUS",
    "CleanPbrTextureLibraryError",
    "LIBRARY_SCHEMA",
    "PENDING_LIBRARY_STATUS",
    "REQUIRED_TEXTURE_ROLES",
    "RUNTIME_MAP_ROLES",
    "load_texture_contract",
    "main",
    "sha256_file",
    "validate_runtime_maps",
    "validate_texture_library",
]
