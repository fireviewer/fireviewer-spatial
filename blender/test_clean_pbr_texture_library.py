from __future__ import annotations

import binascii
import copy
import hashlib
import json
from pathlib import Path
import struct
import zlib

import jsonschema
import numpy as np
from PIL import Image
import pytest

import clean_pbr_texture_library as library


CONTRACT_PATH = Path(__file__).with_name("ground_surface_texture_contract.v4.json")
SCHEMA_PATH = (
    Path(__file__).parents[1]
    / "contracts"
    / "terrain"
    / "v1"
    / "clean-pbr-texture-library.schema.json"
)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
    )


def _write_png(
    path: Path,
    *,
    width: int,
    height: int,
    role: str,
    seed: int,
) -> None:
    """Write a small-on-disk valid PNG without allocating an atlas-sized array."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if role == "height":
        bit_depth = 16
        colour_type = 0
        pixel = struct.pack(">H", (seed * 257) & 0xFFFF)
    else:
        bit_depth = 8
        colour_type = 6
        pixel = bytes((seed & 0xFF, (seed + 37) & 0xFF, (seed + 83) & 0xFF, 255))
    row = b"\0" + pixel * width
    compressor = zlib.compressobj(level=6)
    compressed_parts: list[bytes] = []
    for _ in range(height):
        part = compressor.compress(row)
        if part:
            compressed_parts.append(part)
    compressed_parts.append(compressor.flush())
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, colour_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", b"".join(compressed_parts))
        + _chunk(b"IEND", b"")
    )


def _artifact(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "byte_count": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _atlas_uv(contract: dict, slot: int) -> dict[str, list[float]]:
    atlas = contract["runtime_atlas"]
    column = slot % atlas["columns"]
    row = slot // atlas["columns"]
    return {
        "offset": [
            (column * atlas["cell_size_px"] + atlas["gutter_px"]) / atlas["width_px"],
            (row * atlas["cell_size_px"] + atlas["gutter_px"]) / atlas["height_px"],
        ],
        "scale": [
            (atlas["cell_size_px"] - 2 * atlas["gutter_px"]) / atlas["width_px"],
            (atlas["cell_size_px"] - 2 * atlas["gutter_px"]) / atlas["height_px"],
        ],
    }


@pytest.fixture(scope="module")
def complete_library(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("clean-pbr-library")
    contract = library.load_texture_contract(CONTRACT_PATH)
    runtime_atlases: dict[str, dict[str, object]] = {}
    for role_index, role in enumerate(library.REQUIRED_TEXTURE_ROLES):
        relative = f"runtime-atlas/{role}.png"
        _write_png(
            root / relative,
            width=contract["runtime_atlas"]["width_px"],
            height=contract["runtime_atlas"]["height_px"],
            role=role,
            seed=17 + role_index,
        )
        runtime_atlases[role] = _artifact(root, relative)

    profiles = []
    for index, profile_id in enumerate(
        contract["profile_contract"]["stable_profile_ids"]
    ):
        textures: dict[str, dict[str, object]] = {}
        for role_index, role in enumerate(library.REQUIRED_TEXTURE_ROLES):
            relative = f"profile-sources/{index:02d}/{role}.png"
            _write_png(
                root / relative,
                width=4,
                height=4,
                role=role,
                seed=(index * 7 + role_index * 41) & 0xFF,
            )
            textures[role] = _artifact(root, relative)
        profiles.append(
            {
                "stable_index": index,
                "id": profile_id,
                "surface_basis": "atlas_pbr",
                "atlas_slot": index,
                "atlas_uv": _atlas_uv(contract, index),
                "physical_scale_m": 4.0,
                "projection": (
                    "world_triplanar"
                    if profile_id.startswith("cliff_surface.")
                    else "world_xy"
                ),
                "source_kind": "clean_pbr_profile_texture",
                "textures": textures,
            }
        )
    payload = {
        "schema": library.LIBRARY_SCHEMA,
        "status": library.PENDING_LIBRARY_STATUS,
        "texture_contract_sha256": library.sha256_file(CONTRACT_PATH),
        "orthophoto_dependency": "forbidden",
        "lineage": {
            "source": "fresh_clean_pbr_v4",
            "legacy_atlas_dependencies": [],
        },
        "runtime_atlases": runtime_atlases,
        "profiles": profiles,
        "visual_acceptance": {
            "status": "pending_human_visual_review",
            "receipt": None,
        },
    }
    manifest = root / "clean-pbr-texture-library.v1.json"
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, payload


def _variant_manifest(base_manifest: Path, payload: dict, name: str, mutate) -> Path:
    changed = copy.deepcopy(payload)
    mutate(changed)
    path = base_manifest.with_name(name)
    path.write_text(
        json.dumps(changed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_contract_locks_pending_status_stable_ids_maps_and_temporary_ortho() -> None:
    contract = library.load_texture_contract(CONTRACT_PATH)

    assert contract["status"] == "pending_clean_pbr_library"
    assert len(contract["profile_contract"]["stable_profile_ids"]) == 72
    assert contract["profile_contract"]["stable_profile_ids"][0] == (
        "natural_ground.mediterranean_limestone"
    )
    assert contract["profile_contract"]["stable_profile_ids"][-1] == (
        "cliff_surface.dark_basalt"
    )
    assert contract["profile_contract"]["allowed_surface_basis"] == "atlas_pbr"
    assert set(contract["profile_contract"]["forbidden_surface_bases"]) == {
        "procedural_only",
        "procedural_tint_with_atlas_relief",
        "procedural_water_over_contextual_bed",
    }
    assert contract["orthophoto_policy"] == {
        "build_time_source": "allowed_temporary_correspondence_only",
        "build_time_requirements": [
            "source_hash_locked",
            "excluded_from_clean_pbr_library",
            "deleted_after_package_sealing",
        ],
        "runtime_dependency": "forbidden",
        "package_artifact": "forbidden",
    }
    assert contract["tile_runtime_maps"]["width_px"] == 500
    assert contract["tile_runtime_maps"]["height_px"] == 500
    assert contract["tile_runtime_maps"]["maps"]["orientation"] == {
        "file": "ground-orientation.png",
        "mode": "L8",
        "encoding": "undirected_angle_0_to_pi_mapped_to_uint8",
    }


def test_complete_synthetic_library_validates_but_is_not_visually_promoted(
    complete_library: tuple[Path, dict],
) -> None:
    manifest, payload = complete_library

    result = library.validate_texture_library(
        manifest,
        contract_path=CONTRACT_PATH,
    )

    assert result["status"] == "generated_pending_visual_review"
    assert result["profile_count"] == 72
    assert result["runtime_atlas_count"] == 4
    assert result["source_texture_count"] == 72 * 4
    assert result["visual_acceptance"] == "pending_human_visual_review"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    with pytest.raises(
        library.CleanPbrTextureLibraryError,
        match="lacks human visual acceptance",
    ):
        library.validate_texture_library(
            manifest,
            contract_path=CONTRACT_PATH,
            require_visual_acceptance=True,
        )


def test_missing_profile_is_rejected(
    complete_library: tuple[Path, dict],
) -> None:
    manifest, payload = complete_library
    changed = _variant_manifest(
        manifest,
        payload,
        "missing-profile.json",
        lambda value: value["profiles"].pop(),
    )

    with pytest.raises(
        library.CleanPbrTextureLibraryError,
        match="must contain 72 profiles",
    ):
        library.validate_texture_library(changed, contract_path=CONTRACT_PATH)


def test_procedural_profile_is_rejected(
    complete_library: tuple[Path, dict],
) -> None:
    manifest, payload = complete_library
    changed = _variant_manifest(
        manifest,
        payload,
        "procedural-profile.json",
        lambda value: value["profiles"][30].update(
            {"surface_basis": "procedural_only"}
        ),
    )

    with pytest.raises(
        library.CleanPbrTextureLibraryError,
        match="must use atlas_pbr",
    ):
        library.validate_texture_library(changed, contract_path=CONTRACT_PATH)


def test_packaged_orthophoto_is_rejected(
    complete_library: tuple[Path, dict],
) -> None:
    manifest, payload = complete_library

    def inject_orthophoto(value: dict) -> None:
        value["profiles"][0]["textures"]["basecolor"]["path"] = (
            "orthophoto/profile-00-basecolor.png"
        )

    changed = _variant_manifest(
        manifest,
        payload,
        "orthophoto-source.json",
        inject_orthophoto,
    )
    with pytest.raises(
        library.CleanPbrTextureLibraryError,
        match="forbidden orthophoto token",
    ):
        library.validate_texture_library(changed, contract_path=CONTRACT_PATH)


def _write_runtime_maps(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ids = np.zeros((500, 500, 4), dtype=np.uint8)
    weights = np.zeros_like(ids)
    weights[:, :, 0] = 255
    confidence = np.full((500, 500), 255, dtype=np.uint8)
    orientation = np.broadcast_to(
        np.arange(500, dtype=np.uint16) % 256,
        (500, 500),
    ).astype(np.uint8)
    Image.fromarray(ids, mode="RGBA").save(root / "ground-profile-ids.png")
    Image.fromarray(weights, mode="RGBA").save(root / "ground-profile-weights.png")
    Image.fromarray(confidence, mode="L").save(root / "ground-confidence.png")
    Image.fromarray(orientation, mode="L").save(root / "ground-orientation.png")


def test_runtime_maps_are_exact_500px_and_orientation_is_l8(tmp_path: Path) -> None:
    _write_runtime_maps(tmp_path)

    result = library.validate_runtime_maps(tmp_path, contract_path=CONTRACT_PATH)

    assert result["width_px"] == 500
    assert result["height_px"] == 500
    assert result["profile_count"] == 72
    assert result["confidence_range"] == [255, 255]
    assert result["orientation_encoding"] == (
        "undirected_angle_0_to_pi_mapped_to_uint8"
    )
    assert set(result["maps_sha256"]) == set(library.RUNTIME_MAP_ROLES)


def test_runtime_maps_reject_invalid_weight_sum(tmp_path: Path) -> None:
    _write_runtime_maps(tmp_path)
    with Image.open(tmp_path / "ground-profile-weights.png") as image:
        weights = np.asarray(image, dtype=np.uint8).copy()
    weights[0, 0] = 0
    Image.fromarray(weights, mode="RGBA").save(tmp_path / "ground-profile-weights.png")

    with pytest.raises(
        library.CleanPbrTextureLibraryError,
        match="sum exactly to 255",
    ):
        library.validate_runtime_maps(tmp_path, contract_path=CONTRACT_PATH)
