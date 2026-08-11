"""Canonical package and completion receipt for one adaptive terrain tile."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path, PurePath
import struct
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

try:
    from adaptive_terrain_quadtree import (
        EDGE_ORDER,
        STITCH_MASK_BITS,
        materialize_stitch_triangles,
        read_fvtq,
    )
    from compact_hag import read_hag_max_2m
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from blender.adaptive_terrain_quadtree import (
        EDGE_ORDER,
        STITCH_MASK_BITS,
        materialize_stitch_triangles,
        read_fvtq,
    )
    from blender.compact_hag import read_hag_max_2m


LEGACY_PACKAGE_SCHEMA = "fireviewer.tile-package.v2"
LEGACY_DONE_SCHEMA = "fireviewer.tile.done.v2"
PACKAGE_SCHEMA = "fireviewer.tile-package.v3"
DONE_SCHEMA = "fireviewer.tile.done.v3"
CRS = "EPSG:2154"
LEGACY_GROUND_MATERIAL_SCHEMA = "fireviewer.ground-material-contract.v1"
GROUND_MATERIAL_SCHEMA = "fireviewer.ground-material-contract.v2"
RUNTIME_SHADER_SCHEMA = "fireviewer.ground-runtime-shader-binding.v1"
RUNTIME_SHADER_PENDING_STATUS = "pending_dedicated_mdl_validation"
LEGACY_PACKAGE_FILE_NAME = "tile-package.v2.json"
LEGACY_DONE_FILE_NAME = "tile.done.v2.json"
PACKAGE_FILE_NAME = "tile-package.v3.json"
DONE_FILE_NAME = "tile.done.v3.json"
SHA256_KEYS = ("path", "byte_count", "sha256")
LEGACY_OUTPUT_FILES = {
    "terrain_lod0": "terrain-lod0.fvtq",
    "terrain_lod1": "terrain-lod1.fvtq",
    "terrain_lod2": "terrain-lod2.fvtq",
    "hag_max_2m": "hag-max-2m.tif",
    "ground_profile_ids": "ground-profile-ids.png",
    "ground_profile_weights": "ground-profile-weights.png",
    "surface_overlays": "surface-overlays.json.gz",
    "tile_composition": "tile-composition.json.gz",
}
OUTPUT_FILES = {
    "terrain_lod0": "terrain-lod0.fvtq",
    "terrain_lod1": "terrain-lod1.fvtq",
    "terrain_lod2": "terrain-lod2.fvtq",
    "hag_max_2m": "hag-max-2m.tif",
    "ground_profile_ids": "ground-profile-ids.png",
    "ground_profile_weights": "ground-profile-weights.png",
    "ground_confidence": "ground-confidence.png",
    "ground_orientation": "ground-orientation.png",
}


class TilePackageError(ValueError):
    """A tile package is incomplete, incoherent, mutable or corrupted."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
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


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TilePackageError(f"{label} must be a lowercase SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise TilePackageError(f"{label} must be a lowercase SHA-256") from error
    if value != value.lower():
        raise TilePackageError(f"{label} must be a lowercase SHA-256")
    return value


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise TilePackageError(f"Artifact escapes tile package: {path}") from error
    content = path.read_bytes()
    return {"path": relative, "byte_count": len(content), "sha256": _sha256(content)}


def _validate_record(record: Mapping[str, Any], label: str) -> None:
    if set(record) != set(SHA256_KEYS):
        raise TilePackageError(f"{label} must contain path, byte_count and sha256")
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise TilePackageError(f"{label}.path must be a relative path")
    pure = PurePath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise TilePackageError(f"{label}.path escapes the tile package")
    byte_count = record.get("byte_count")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise TilePackageError(f"{label}.byte_count must be non-negative")
    _hash(record.get("sha256"), f"{label}.sha256")


def _validate_ground_material_identity(
    value: Any, *, legacy_v2: bool = False
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TilePackageError("ground_material identity is missing")
    required = {
        "schema",
        "zone_path",
        "contract_sha256",
        "runtime_atlas_sha256",
        "material_layer_sha256",
        "visual_acceptance",
    }
    required.update(
        {
            "source_atlas_catalog_sha256",
            "atlas_catalog_sha256",
        }
        if legacy_v2
        else {
            "source_library_schema",
            "source_library_manifest_sha256",
            "source_library_identity_sha256",
            "source_library_content_sha256",
            "texture_contract_sha256",
            "runtime_shader",
        }
    )
    expected_schema = (
        LEGACY_GROUND_MATERIAL_SCHEMA if legacy_v2 else GROUND_MATERIAL_SCHEMA
    )
    if set(value) != required or value.get("schema") != expected_schema:
        raise TilePackageError("ground_material identity is invalid")
    zone_path = value.get("zone_path")
    if not isinstance(zone_path, str) or not zone_path or "\\" in zone_path:
        raise TilePackageError("ground_material.zone_path must be portable")
    pure = PurePath(zone_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise TilePackageError("ground_material.zone_path escapes the zone")
    hash_fields = ["contract_sha256", "material_layer_sha256"]
    hash_fields.extend(
        ["source_atlas_catalog_sha256", "atlas_catalog_sha256"]
        if legacy_v2
        else [
            "source_library_manifest_sha256",
            "source_library_identity_sha256",
            "source_library_content_sha256",
        ]
    )
    for field in hash_fields:
        _hash(value.get(field), f"ground_material.{field}")
    if not legacy_v2:
        if value.get("source_library_schema") != (
            "fireviewer.clean-pbr-texture-library.v1"
        ):
            raise TilePackageError("ground_material source library schema is invalid")
        texture_contract = value.get("texture_contract_sha256")
        _hash(texture_contract, "ground_material.texture_contract_sha256")
        if value.get("runtime_shader") != {
            "schema": RUNTIME_SHADER_SCHEMA,
            "status": RUNTIME_SHADER_PENDING_STATUS,
            "implementation": None,
            "source_artifact": None,
            "production_textured_runtime_qualified": False,
            "preview_surface_policy": "diagnostic_untextured_only",
            "required_capabilities": [
                "four_profile_id_indirections",
                "rgba8_weighted_pbr_blend",
                "epsg2154_world_projection",
                "undirected_orientation_0_to_pi",
                "world_xy_and_world_triplanar",
            ],
        }:
            raise TilePackageError(
                "ground_material runtime shader must remain explicitly fail-closed"
            )
    atlas = value.get("runtime_atlas_sha256")
    if not isinstance(atlas, Mapping) or set(atlas) != {
        "basecolor",
        "normal",
        "height",
        "orm",
    }:
        raise TilePackageError("ground_material runtime atlas identity is incomplete")
    for role, digest in atlas.items():
        _hash(digest, f"ground_material.runtime_atlas_sha256.{role}")
    if value.get("visual_acceptance") not in {
        "pending_human_review",
        "accepted_human_visual",
    }:
        raise TilePackageError("ground_material visual acceptance status is invalid")
    return dict(value)


def _stitch_variant_catalog(meshes: Sequence[Any]) -> dict[str, Any]:
    lods: dict[str, list[dict[str, Any]]] = {}
    for mesh in meshes:
        if len(mesh.stitch_variants) != 16 or tuple(
            variant.mask for variant in mesh.stitch_variants
        ) != tuple(range(16)):
            raise TilePackageError(
                f"FVTQ LOD{mesh.lod} does not expose ordered stitch masks 0..15"
            )
        records: list[dict[str, Any]] = []
        for mask, variant in enumerate(mesh.stitch_variants):
            triangles = materialize_stitch_triangles(mesh, mask)
            triangle_bytes = b"".join(
                struct.pack("<III", *triangle) for triangle in triangles
            )
            if len(variant.effective_edge_signatures) != len(EDGE_ORDER):
                raise TilePackageError(
                    f"FVTQ LOD{mesh.lod} stitch mask {mask} has incomplete edges"
                )
            records.append(
                {
                    "mask": mask,
                    "triangle_count": len(triangles),
                    "triangle_indices_sha256": _sha256(triangle_bytes),
                    "maximum_error_mm": variant.maximum_error_mm,
                    "effective_edge_signatures": [
                        signature.hex()
                        for signature in variant.effective_edge_signatures
                    ],
                }
            )
        lods[f"lod{mesh.lod}"] = records
    return {
        "encoding": "fvtq-base-remove-add.v1",
        "edge_order": list(EDGE_ORDER),
        "edge_mask_bits": dict(STITCH_MASK_BITS),
        "available_masks": list(range(16)),
        "lods": lods,
    }


def _validate_geometry(
    root: Path, bounds: Sequence[float]
) -> tuple[str, dict[str, Any]]:
    meshes = tuple(
        read_fvtq(root / OUTPUT_FILES[f"terrain_lod{lod}"]) for lod in range(3)
    )
    if tuple(mesh.lod for mesh in meshes) != (0, 1, 2):
        raise TilePackageError("FVTQ LOD order is invalid")
    if len({mesh.tile_origin_mm for mesh in meshes}) != 1:
        raise TilePackageError("FVTQ tile origins differ")
    expected_origin = (round(float(bounds[0]) * 1000), round(float(bounds[1]) * 1000))
    if meshes[0].tile_origin_mm != expected_origin:
        raise TilePackageError("FVTQ origin differs from tile bounds")
    if len({mesh.source_grid_sha256 for mesh in meshes}) != 1:
        raise TilePackageError("FVTQ source grid hashes differ")
    if len({mesh.contract_sha256 for mesh in meshes}) != 1:
        raise TilePackageError("FVTQ terrain contract hashes differ")
    normal_halo_hashes = {mesh.normal_halo_sha256 for mesh in meshes}
    if len(normal_halo_hashes) != 1:
        raise TilePackageError("FVTQ normal halo hashes differ")
    if not set(meshes[2].vertices).issubset(meshes[1].vertices):
        raise TilePackageError("LOD2 vertices are not a subset of LOD1")
    if not set(meshes[1].vertices).issubset(meshes[0].vertices):
        raise TilePackageError("LOD1 vertices are not a subset of LOD0")
    return (
        next(iter(normal_halo_hashes)).hex(),
        _stitch_variant_catalog(meshes),
    )


def _validate_hag_and_surface(
    root: Path, bounds: Sequence[float], *, legacy_v2: bool
) -> dict[str, Any]:
    _values, hag = read_hag_max_2m(root / OUTPUT_FILES["hag_max_2m"])
    if hag["bounds_l93_m"] != [float(value) for value in bounds]:
        raise TilePackageError("HAG bounds differ from tile bounds")
    ids = np.asarray(Image.open(root / OUTPUT_FILES["ground_profile_ids"]))
    weights = np.asarray(Image.open(root / OUTPUT_FILES["ground_profile_weights"]))
    grid_size = 100 if legacy_v2 else 500
    if ids.shape != (grid_size, grid_size, 4) or weights.shape != ids.shape:
        raise TilePackageError(
            f"Ground profile maps must be {grid_size}x{grid_size} RGBA"
        )
    if ids.dtype != np.uint8 or weights.dtype != np.uint8:
        raise TilePackageError("Ground profile maps must be exact RGBA8 PNGs")
    if not np.all(weights.sum(axis=2, dtype=np.uint16) == 255):
        raise TilePackageError("Ground profile weights do not sum to 255")
    used = ids[weights > 0]
    if used.size == 0 or int(used.max()) >= 72:
        raise TilePackageError("Ground profile map references an invalid profile")
    if not legacy_v2:
        prohibited_residuals = {
            "surface-overlays.json.gz",
            "tile-composition.json.gz",
        }
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(root).as_posix()
            if candidate.name in prohibited_residuals or "ortho" in relative.casefold():
                raise TilePackageError(
                    f"Forbidden temporary/runtime ground artifact remains: {relative}"
                )
        confidence_image = Image.open(root / OUTPUT_FILES["ground_confidence"])
        orientation_image = Image.open(root / OUTPUT_FILES["ground_orientation"])
        confidence = np.asarray(confidence_image)
        orientation = np.asarray(orientation_image)
        if (
            confidence_image.mode != "L"
            or orientation_image.mode != "L"
            or confidence.shape != (500, 500)
            or orientation.shape != confidence.shape
            or confidence.dtype != np.uint8
            or orientation.dtype != np.uint8
        ):
            raise TilePackageError(
                "Ground confidence and orientation maps must be 500x500 L8 PNGs"
            )
        return {
            "schema": "fireviewer.ground-surface-mapping.v3",
            "crs": CRS,
            "bounds_l93_m": [float(value) for value in bounds],
            "grid_size_px": [500, 500],
            "cell_size_m": 1,
            "row_order": "north_to_south",
            "profile_count": 72,
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
            "world_projection": "EPSG:2154 metric XY with no tile-local phase reset",
            "variant_selection": "baked_profile_id",
            "runtime_procedural_material": "forbidden",
            "runtime_orthophoto": "forbidden",
            "surface_overlays": "not_packaged",
        }
    try:
        overlays = json.loads(
            gzip.decompress(
                (root / LEGACY_OUTPUT_FILES["surface_overlays"]).read_bytes()
            )
        )
        composition = json.loads(
            gzip.decompress(
                (root / LEGACY_OUTPUT_FILES["tile_composition"]).read_bytes()
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TilePackageError(f"Ground composition is invalid: {error}") from error
    if overlays.get("schema") != "fireviewer.surface-overlays.v1":
        raise TilePackageError("Surface overlay schema mismatch")
    if composition.get("schema") != "fireviewer.tile-composition.v1":
        raise TilePackageError("Tile composition schema mismatch")
    declared = composition.get("outputs")
    if not isinstance(declared, dict):
        raise TilePackageError("Tile composition output hashes are missing")
    for name in (
        "ground-profile-ids.png",
        "ground-profile-weights.png",
        "surface-overlays.json.gz",
    ):
        path = root / name
        record = declared.get(name)
        if not isinstance(record, dict):
            raise TilePackageError(f"Tile composition does not declare {name}")
        content = path.read_bytes()
        if record.get("bytes") != len(content) or record.get("sha256") != _sha256(
            content
        ):
            raise TilePackageError(f"Tile composition hash mismatch: {name}")
    return {
        "schema": "fireviewer.ground-surface-mapping.v2-legacy",
        "grid_size_px": [100, 100],
        "cell_size_m": 5,
    }


def _validate_surface_correspondence(
    root: Path, bounds: Sequence[float], record: Mapping[str, Any]
) -> dict[str, Any]:
    if record.get("path") != "surface-correspondence.json":
        raise TilePackageError(
            "inputs.surface_correspondence must point to surface-correspondence.json"
        )
    path = root / "surface-correspondence.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TilePackageError(
            f"Surface correspondence manifest is invalid: {error}"
        ) from error
    required = {
        "schema",
        "status",
        "crs",
        "bounds_l93_m",
        "grid",
        "identity",
        "runtime",
        "profile_id_encoding",
        "profile_weight_encoding",
        "confidence_encoding",
        "orientation_encoding",
        "profile_table",
        "primary_pixel_counts_by_class",
        "restriction_pixel_counts",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise TilePackageError("Surface correspondence fields are incomplete")
    if (
        payload.get("schema") != "fireviewer.surface-correspondence-tile.v1"
        or payload.get("status") != "compiled_no_orthophoto_payload"
        or payload.get("crs") != CRS
        or payload.get("bounds_l93_m") != [float(value) for value in bounds]
        or payload.get("grid") != {"resolution_m": 1, "width": 500, "height": 500}
        or payload.get("runtime")
        != {
            "orthophoto_dependency": "forbidden",
            "orthophoto_pixels_present": False,
            "procedural_materials": "forbidden",
            "projection": "profile_declared_world_xy_or_world_triplanar",
        }
        or payload.get("profile_id_encoding") != "RGBA8 zero-based profile_table index"
        or payload.get("profile_weight_encoding") != "RGBA8 sum exactly 255"
        or payload.get("confidence_encoding")
        != "L8 best-versus-next-semantic-class margin"
        or payload.get("orientation_encoding")
        != "L8 world texture orientation modulo 180 degrees"
    ):
        raise TilePackageError("Surface correspondence contract is invalid")
    identity = payload.get("identity")
    identity_fields = {
        "orthophoto_source_sha256",
        "orthophoto_tile_input_sha256",
        "pbr_library_sha256",
        "correspondence_model_sha256",
        "algorithm_sha256",
        "contract_sha256",
        "context_priors_sha256",
        "approved_corrections_sha256",
    }
    if not isinstance(identity, Mapping) or set(identity) != identity_fields:
        raise TilePackageError("Surface correspondence identity is incomplete")
    for name, digest in identity.items():
        _hash(digest, f"surface_correspondence.identity.{name}")
    profiles = payload.get("profile_table")
    if not isinstance(profiles, list) or len(profiles) != 72:
        raise TilePackageError("Surface correspondence profile table is incomplete")
    profile_ids: set[str] = set()
    for index, profile in enumerate(profiles):
        atlas_uv = profile.get("atlas_uv") if isinstance(profile, Mapping) else None
        if (
            not isinstance(profile, Mapping)
            or profile.get("stable_index") != index
            or not isinstance(profile.get("atlas_slot"), int)
            or profile["atlas_slot"] < 0
            or not isinstance(profile.get("id"), str)
            or not profile["id"]
            or profile["id"] in profile_ids
            or profile.get("projection") not in {"world_xy", "world_triplanar"}
            or not isinstance(profile.get("class_id"), str)
            or not profile["class_id"]
            or not isinstance(profile.get("physical_scale_m"), (int, float))
            or isinstance(profile.get("physical_scale_m"), bool)
            or float(profile["physical_scale_m"]) <= 0.0
            or not isinstance(atlas_uv, Mapping)
            or set(atlas_uv) != {"offset", "scale"}
        ):
            raise TilePackageError(
                f"Surface correspondence profile identity is invalid: {index}"
            )
        try:
            offset = [float(value) for value in atlas_uv["offset"]]
            scale = [float(value) for value in atlas_uv["scale"]]
        except (KeyError, TypeError, ValueError) as error:
            raise TilePackageError(
                f"Surface correspondence atlas UV is invalid: {index}"
            ) from error
        if (
            len(offset) != 2
            or len(scale) != 2
            or any(value < 0.0 or value > 1.0 for value in offset)
            or any(value <= 0.0 or value > 1.0 for value in scale)
            or any(offset[axis] + scale[axis] > 1.0 for axis in range(2))
        ):
            raise TilePackageError(
                f"Surface correspondence atlas UV is out of range: {index}"
            )
        profile_ids.add(profile["id"])
        textures = profile.get("textures")
        if not isinstance(textures, Mapping) or set(textures) != {
            "basecolor",
            "normal",
            "height",
            "orm",
        }:
            raise TilePackageError(
                f"Surface correspondence profile textures are incomplete: {index}"
            )
        for role, artifact in textures.items():
            if (
                not isinstance(artifact, Mapping)
                or set(artifact) != {"byte_count", "sha256"}
                or not isinstance(artifact.get("byte_count"), int)
                or artifact["byte_count"] <= 0
            ):
                raise TilePackageError(
                    f"Surface correspondence texture identity is invalid: {index}/{role}"
                )
            _hash(
                artifact.get("sha256"),
                f"surface_correspondence.profile_table.{index}.{role}",
            )
    artifacts = payload.get("artifacts")
    expected_artifacts = {
        "ground-profile-ids.png",
        "ground-profile-weights.png",
        "ground-confidence.png",
        "ground-orientation.png",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise TilePackageError(
            "Surface correspondence runtime artifacts are incomplete"
        )
    for name, artifact in artifacts.items():
        map_path = root / name
        content = map_path.read_bytes()
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"byte_count", "sha256"}
            or artifact.get("byte_count") != len(content)
            or artifact.get("sha256") != _sha256(content)
        ):
            raise TilePackageError(
                f"Surface correspondence runtime artifact hash mismatch: {name}"
            )
    return payload


def build_tile_package(
    tile_root: Path,
    *,
    tile_id: str,
    recipe_id: str,
    recipe_build_id: str,
    bounds_l93_m: Sequence[float],
    inputs: Mapping[str, Mapping[str, Any]],
    ground_material: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a v3 package.  Legacy v2 packages are validation-only."""

    root = Path(tile_root)
    if not tile_id:
        raise TilePackageError("tile_id must not be empty")
    _hash(recipe_id, "recipe_id")
    _hash(recipe_build_id, "recipe_build_id")
    if len(bounds_l93_m) != 4:
        raise TilePackageError("bounds_l93_m must contain four values")
    bounds = [float(value) for value in bounds_l93_m]
    if bounds[2] - bounds[0] != 500.0 or bounds[3] - bounds[1] != 500.0:
        raise TilePackageError("Tile bounds must be exactly 500 m square")
    if not inputs:
        raise TilePackageError("Tile package inputs must not be empty")
    normalized_inputs = {name: dict(record) for name, record in sorted(inputs.items())}
    if "surface_correspondence" not in normalized_inputs:
        raise TilePackageError(
            "A v3 tile package requires inputs.surface_correspondence"
        )
    for name, record in normalized_inputs.items():
        if (
            "ortho" in name.casefold()
            or "ortho" in str(record.get("path", "")).casefold()
        ):
            raise TilePackageError(
                "Temporary orthophoto matching inputs cannot enter a v3 tile package"
            )
        _validate_record(record, f"inputs.{name}")
        input_path = root / record["path"]
        try:
            input_path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise TilePackageError(f"inputs.{name} escapes the tile package") from error
        if not input_path.is_file():
            raise TilePackageError(
                f"inputs.{name} is missing while sealing the package"
            )
        content = input_path.read_bytes()
        if record["byte_count"] != len(content) or record["sha256"] != _sha256(content):
            raise TilePackageError(
                f"inputs.{name} hash mismatch while sealing the package"
            )
    normalized_ground_material = _validate_ground_material_identity(ground_material)
    correspondence = _validate_surface_correspondence(
        root, bounds, normalized_inputs["surface_correspondence"]
    )
    if (
        correspondence["identity"]["pbr_library_sha256"]
        != normalized_ground_material["source_library_identity_sha256"]
    ):
        raise TilePackageError(
            "Surface correspondence and ground material use different PBR libraries"
        )
    for file_name in OUTPUT_FILES.values():
        if not (root / file_name).is_file():
            raise TilePackageError(f"Canonical tile output is missing: {file_name}")
    normal_halo_sha256, stitch_variants = _validate_geometry(root, bounds)
    surface_mapping = _validate_hag_and_surface(root, bounds, legacy_v2=False)
    outputs = {
        name: _artifact(root / file_name, root)
        for name, file_name in sorted(OUTPUT_FILES.items())
    }
    payload = {
        "schema": PACKAGE_SCHEMA,
        "tile_id": tile_id,
        "recipe_id": recipe_id,
        "recipe_build_id": recipe_build_id,
        "crs": CRS,
        "bounds_l93_m": bounds,
        "normal_halo_sha256": normal_halo_sha256,
        "stitch_variants": stitch_variants,
        "inputs": normalized_inputs,
        "ground_material": normalized_ground_material,
        "surface_mapping": surface_mapping,
        "outputs": outputs,
    }
    destination = root / PACKAGE_FILE_NAME
    encoded = _canonical_json(payload)
    if destination.exists() and destination.read_bytes() != encoded:
        raise FileExistsError(
            f"Refusing to replace different tile package: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(destination)
    return payload


def write_tile_done(tile_root: Path, package: Mapping[str, Any]) -> dict[str, Any]:
    """Sign all canonical outputs plus `tile-package.v3.json`."""

    root = Path(tile_root)
    if package.get("schema") != PACKAGE_SCHEMA:
        raise TilePackageError("Cannot complete an unsupported tile package")
    validate_tile_package(root)
    outputs = dict(package["outputs"])
    outputs["tile_package"] = _artifact(root / PACKAGE_FILE_NAME, root)
    receipt = {
        "schema": DONE_SCHEMA,
        "tile_id": package["tile_id"],
        "recipe_id": package["recipe_id"],
        "recipe_build_id": package["recipe_build_id"],
        "normal_halo_sha256": package["normal_halo_sha256"],
        "stitch_variants": package["stitch_variants"],
        "inputs": package["inputs"],
        "ground_material": package["ground_material"],
        "surface_mapping": package["surface_mapping"],
        "outputs": outputs,
    }
    destination = root / DONE_FILE_NAME
    encoded = _canonical_json(receipt)
    if destination.exists() and destination.read_bytes() != encoded:
        raise FileExistsError(
            f"Refusing to replace different tile receipt: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(destination)
    return receipt


def _package_paths(root: Path) -> tuple[Path, Path, bool]:
    v3 = root / PACKAGE_FILE_NAME
    legacy = root / LEGACY_PACKAGE_FILE_NAME
    if v3.is_file():
        return v3, root / DONE_FILE_NAME, False
    if legacy.is_file():
        return legacy, root / LEGACY_DONE_FILE_NAME, True
    raise TilePackageError("Tile package manifest is missing")


def validate_tile_package(tile_root: Path) -> dict[str, Any]:
    root = Path(tile_root)
    package_path, _done_path, legacy_v2 = _package_paths(root)
    try:
        payload = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TilePackageError(f"Invalid tile package manifest: {error}") from error
    expected_schema = LEGACY_PACKAGE_SCHEMA if legacy_v2 else PACKAGE_SCHEMA
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise TilePackageError("Unsupported tile package manifest")
    _validate_ground_material_identity(
        payload.get("ground_material"), legacy_v2=legacy_v2
    )
    normal_halo_sha256, stitch_variants = _validate_geometry(
        root, payload["bounds_l93_m"]
    )
    if payload.get("normal_halo_sha256") != normal_halo_sha256:
        raise TilePackageError("Tile package normal halo identity differs from FVTQ")
    if payload.get("stitch_variants") != stitch_variants:
        raise TilePackageError("Tile package stitch variants differ from FVTQ")
    surface_mapping = _validate_hag_and_surface(
        root, payload["bounds_l93_m"], legacy_v2=legacy_v2
    )
    if not legacy_v2 and payload.get("surface_mapping") != surface_mapping:
        raise TilePackageError("Ground surface mapping contract changed")
    for collection in ("inputs", "outputs"):
        records = payload.get(collection)
        if not isinstance(records, dict) or not records:
            raise TilePackageError(f"Tile package {collection} are missing")
        for name, record in records.items():
            if not isinstance(record, dict):
                raise TilePackageError(f"Invalid {collection}.{name}")
            if not legacy_v2 and (
                "ortho" in str(name).casefold()
                or "ortho" in str(record.get("path", "")).casefold()
            ):
                raise TilePackageError(
                    "Temporary orthophoto matching inputs cannot enter a v3 tile package"
                )
            _validate_record(record, f"{collection}.{name}")
            path = root / record["path"]
            if not path.is_file():
                raise TilePackageError(f"Canonical {collection} is missing: {name}")
            content = path.read_bytes()
            if record["byte_count"] != len(content) or record["sha256"] != _sha256(
                content
            ):
                raise TilePackageError(f"Canonical {collection} hash mismatch: {name}")
    if not legacy_v2:
        inputs = payload["inputs"]
        if "surface_correspondence" not in inputs:
            raise TilePackageError(
                "A v3 tile package requires inputs.surface_correspondence"
            )
        correspondence = _validate_surface_correspondence(
            root, payload["bounds_l93_m"], inputs["surface_correspondence"]
        )
        if (
            correspondence["identity"]["pbr_library_sha256"]
            != payload["ground_material"]["source_library_identity_sha256"]
        ):
            raise TilePackageError(
                "Surface correspondence and ground material use different PBR libraries"
            )
    expected_outputs = set(LEGACY_OUTPUT_FILES if legacy_v2 else OUTPUT_FILES)
    if set(payload["outputs"]) != expected_outputs:
        raise TilePackageError("Canonical output set is incomplete")
    return payload


def validate_tile_done(tile_root: Path) -> dict[str, Any]:
    root = Path(tile_root)
    package = validate_tile_package(root)
    package_path, done_path, legacy_v2 = _package_paths(root)
    try:
        receipt = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TilePackageError(f"Invalid tile completion receipt: {error}") from error
    expected_done_schema = LEGACY_DONE_SCHEMA if legacy_v2 else DONE_SCHEMA
    if not isinstance(receipt, dict) or receipt.get("schema") != expected_done_schema:
        raise TilePackageError("Unsupported tile completion receipt")
    identity_fields = [
        "tile_id",
        "recipe_id",
        "recipe_build_id",
        "normal_halo_sha256",
        "stitch_variants",
        "inputs",
        "ground_material",
    ]
    if not legacy_v2:
        identity_fields.append("surface_mapping")
    if any(receipt.get(key) != package.get(key) for key in identity_fields):
        raise TilePackageError("Tile completion identity differs from its package")
    expected_outputs = dict(package["outputs"])
    expected_outputs["tile_package"] = _artifact(package_path, root)
    if receipt.get("outputs") != expected_outputs:
        raise TilePackageError("Tile completion output signatures differ")
    return receipt
