"""Compile temporary 1 m orthophotos into compact deterministic ground color.

The orthophoto is a production input only.  Every 500 m tile retains one
500 x 500 RGB8 texture and a provenance manifest; no source path or source
payload is serialized for runtime use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from PIL import Image

CONTRACT_SCHEMA = "fireviewer.orthophoto-ground-texture-contract.v1"
OUTPUT_SCHEMA = "fireviewer.orthophoto-ground-texture-tile.v1"
CRS = "EPSG:2154"
SOURCE_RESOLUTION_M = 1
OUTPUT_RESOLUTION_M = 1
TILE_SIZE_M = 500
HALO_M = 10
OUTPUT_GRID_SIZE = TILE_SIZE_M // OUTPUT_RESOLUTION_M
OUTPUT_NAMES = ("ground-color.png", "ground-color.json")
HEX_DIGITS = frozenset("0123456789abcdef")
REVISION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class GroundTextureWindow:
    """One globally aligned, halo-free 1 m ground-color window."""

    core_bounds_l93_m: tuple[int, int, int, int]
    color_rgb_u8: np.ndarray
    identity: Mapping[str, str]
    tile_input_sha256_by_bounds: Mapping[str, str]


@dataclass(frozen=True)
class GroundTextureTile:
    """Exactly one 500 m tile with one 500 x 500 RGB8 texture."""

    bounds_l93_m: tuple[int, int, int, int]
    color_rgb_u8: np.ndarray
    identity: Mapping[str, str]


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


def _require_revision(value: Any) -> str:
    revision = str(value or "")
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("Orthophoto revision must be a stable non-path token")
    return revision


def _load_contract(path: Path | None) -> tuple[dict[str, Any], Path]:
    resolved = (
        Path(__file__).with_name("orthophoto_ground_texture_contract.v1.json")
        if path is None
        else Path(path)
    ).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("status") != "locked":
        raise ValueError("Unsupported or unlocked orthophoto ground contract")
    source = payload.get("input")
    output = payload.get("output")
    downsample = payload.get("downsample")
    runtime = payload.get("runtime")
    if payload.get("crs") != CRS or source != {
        "pixel_type": "RGB8",
        "resolution_m": SOURCE_RESOLUTION_M,
        "north_up": True,
        "global_grid_alignment_m": SOURCE_RESOLUTION_M,
        "processing_halo_m": HALO_M,
        "orthophoto_retention": "temporary_source_deleted_after_validation",
        "revision_format": "stable_non_path_token",
    }:
        raise ValueError("Orthophoto ground input contract differs from the locked v1")
    if output != {
        "tile_size_m": TILE_SIZE_M,
        "resolution_m": OUTPUT_RESOLUTION_M,
        "grid_size_px": [OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE],
        "texture": "ground-color.png_RGB8",
        "manifest": "ground-color.json",
    }:
        raise ValueError("Orthophoto ground output contract differs from the locked v1")
    if downsample != {
        "implementation": "none_exact_core_crop",
        "block_size_px": [1, 1],
        "channel_domain": "RGB8_source_values_preserved",
        "rounding": "not_applicable",
        "alignment": "global_EPSG2154_1m_cells",
    }:
        raise ValueError("Orthophoto ground downsample contract differs from locked v1")
    if runtime != {
        "orthophoto_source_file_dependency": "forbidden",
        "orthophoto_source_path_in_manifest": "forbidden",
        "texture_file": "ground-color.png",
    }:
        raise ValueError("Orthophoto ground runtime contract differs from locked v1")
    return payload, resolved


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
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
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
    if any(value % TILE_SIZE_M for value in core):
        raise ValueError("Core bounds must align to the global 500 m tile grid")
    if (
        window[0] > core[0] - HALO_M
        or window[1] > core[1] - HALO_M
        or window[2] < core[2] + HALO_M
        or window[3] < core[3] + HALO_M
    ):
        raise ValueError(f"Orthophoto window must provide a {HALO_M} m halo")
    return window, core


def _core_slices(
    window_bounds: tuple[int, int, int, int],
    core_bounds: tuple[int, int, int, int],
) -> tuple[slice, slice]:
    west, _south, _east, north = window_bounds
    core_west, core_south, core_east, core_north = core_bounds
    return (
        slice(north - core_north, north - core_south),
        slice(core_west - west, core_east - west),
    )


def _tile_input_hashes(
    rgb_u8: np.ndarray,
    window_bounds: tuple[int, int, int, int],
    core_bounds: tuple[int, int, int, int],
) -> dict[str, str]:
    """Hash exact 520 m tile inputs independently of batch grouping."""

    window_west, _window_south, _window_east, window_north = window_bounds
    core_west, core_south, core_east, core_north = core_bounds
    output: dict[str, str] = {}
    for south in range(core_south, core_north, TILE_SIZE_M):
        north = south + TILE_SIZE_M
        for west in range(core_west, core_east, TILE_SIZE_M):
            east = west + TILE_SIZE_M
            source_bounds = (
                west - HALO_M,
                south - HALO_M,
                east + HALO_M,
                north + HALO_M,
            )
            rows = slice(
                window_north - source_bounds[3], window_north - source_bounds[1]
            )
            columns = slice(
                source_bounds[0] - window_west, source_bounds[2] - window_west
            )
            pixels = rgb_u8[rows, columns]
            identity = {
                "crs": CRS,
                "bounds_l93_m": list(source_bounds),
                "shape": list(pixels.shape),
                "pixels_sha256": _sha256_bytes(pixels.tobytes(order="C")),
            }
            key = f"{west},{south},{east},{north}"
            output[key] = _sha256_bytes(_canonical_bytes(identity))
    return output


def compile_aligned_window(
    rgb_u8: np.ndarray,
    *,
    transform: Affine | Sequence[float],
    crs: str,
    core_bounds_l93_m: Sequence[Any],
    orthophoto_source_manifest_sha256: str,
    orthophoto_revision: str,
    contract_path: Path | None = None,
) -> GroundTextureWindow:
    """Crop one tile or rectangular batch on the exact global 1 m grid."""

    _contract, resolved_contract = _load_contract(contract_path)
    window_bounds, core_bounds = _validate_grid(
        rgb_u8, transform, crs, core_bounds_l93_m
    )
    rows, columns = _core_slices(window_bounds, core_bounds)
    color = rgb_u8[rows, columns].copy()
    expected_shape = (
        (core_bounds[3] - core_bounds[1]) // OUTPUT_RESOLUTION_M,
        (core_bounds[2] - core_bounds[0]) // OUTPUT_RESOLUTION_M,
        3,
    )
    if color.shape != expected_shape:
        raise AssertionError("Ground-color output differs from its global 1 m bounds")
    return GroundTextureWindow(
        core_bounds_l93_m=core_bounds,
        color_rgb_u8=color,
        identity={
            "orthophoto_source_manifest_sha256": _require_sha256(
                orthophoto_source_manifest_sha256,
                label="orthophoto source manifest",
            ),
            "orthophoto_revision": _require_revision(orthophoto_revision),
            "algorithm_sha256": _sha256_file(Path(__file__).resolve()),
            "contract_sha256": _sha256_file(resolved_contract),
        },
        tile_input_sha256_by_bounds=_tile_input_hashes(
            rgb_u8, window_bounds, core_bounds
        ),
    )


def slice_tile(
    window: GroundTextureWindow,
    tile_bounds_l93_m: Sequence[Any],
) -> GroundTextureTile:
    """Extract one exact 500 m tile from a compiled rectangular window."""

    bounds = _integer_bounds(tile_bounds_l93_m, label="tile bounds")
    if bounds[2] - bounds[0] != TILE_SIZE_M or bounds[3] - bounds[1] != TILE_SIZE_M:
        raise ValueError("Ground-color tile must be exactly 500 m square")
    if any(value % TILE_SIZE_M for value in bounds):
        raise ValueError("Ground-color tile must align to the global 500 m grid")
    core = window.core_bounds_l93_m
    if (
        bounds[0] < core[0]
        or bounds[1] < core[1]
        or bounds[2] > core[2]
        or bounds[3] > core[3]
    ):
        raise ValueError("Ground-color tile lies outside the compiled core")
    row_start = (core[3] - bounds[3]) // OUTPUT_RESOLUTION_M
    row_end = (core[3] - bounds[1]) // OUTPUT_RESOLUTION_M
    column_start = (bounds[0] - core[0]) // OUTPUT_RESOLUTION_M
    column_end = (bounds[2] - core[0]) // OUTPUT_RESOLUTION_M
    key = ",".join(str(value) for value in bounds)
    identity = dict(window.identity)
    identity["orthophoto_tile_input_sha256"] = window.tile_input_sha256_by_bounds[key]
    return GroundTextureTile(
        bounds_l93_m=bounds,
        color_rgb_u8=window.color_rgb_u8[
            row_start:row_end, column_start:column_end
        ].copy(),
        identity=identity,
    )


def _png_bytes(rgb_u8: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(rgb_u8, mode="RGB").save(
        stream, format="PNG", optimize=False, compress_level=9
    )
    return stream.getvalue()


def serialize_tile_outputs(tile: GroundTextureTile) -> dict[str, bytes]:
    """Serialize the sole runtime texture and its compact provenance manifest."""

    if tile.color_rgb_u8.shape != (OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE, 3):
        raise ValueError("Ground color must be a 500 x 500 RGB array")
    if tile.color_rgb_u8.dtype != np.uint8:
        raise ValueError("Ground color must use RGB8 pixels")
    color_png = _png_bytes(tile.color_rgb_u8)
    west, south, east, north = tile.bounds_l93_m
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "status": "compiled_ground_color_no_source_payload",
        "crs": CRS,
        "bounds_l93_m": [west, south, east, north],
        "source_window_bounds_l93_m": [
            west - HALO_M,
            south - HALO_M,
            east + HALO_M,
            north + HALO_M,
        ],
        "grid": {
            "resolution_m": OUTPUT_RESOLUTION_M,
            "width": OUTPUT_GRID_SIZE,
            "height": OUTPUT_GRID_SIZE,
            "affine": [OUTPUT_RESOLUTION_M, 0, west, 0, -OUTPUT_RESOLUTION_M, north],
            "pixel_interpretation": "area",
        },
        "downsample": {
            "source_resolution_m": SOURCE_RESOLUTION_M,
            "filter": "none_exact_core_crop",
            "rounding": "not_applicable",
        },
        "identity": dict(sorted(tile.identity.items())),
        "runtime": {
            "texture_file": "ground-color.png",
            "orthophoto_source_file_dependency": "forbidden",
            "orthophoto_source_path_present": False,
        },
        "artifact": {
            "file": "ground-color.png",
            "mode": "RGB8",
            "byte_count": len(color_png),
            "sha256": _sha256_bytes(color_png),
        },
    }
    return {
        "ground-color.png": color_png,
        "ground-color.json": _canonical_bytes(manifest) + b"\n",
    }


def write_tile_outputs(tile: GroundTextureTile, output_dir: Path) -> dict[str, str]:
    """Publish one tile atomically on D: without retaining source imagery."""

    destination = Path(output_dir).resolve(strict=False)
    if os.name == "nt" and destination.drive.upper() != "D:":
        raise ValueError("Ground texture outputs must stay on D:")
    if destination.exists():
        raise FileExistsError("Ground texture output directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    if staging.exists():
        raise FileExistsError("Ground texture staging directory already exists")
    staging.mkdir()
    try:
        outputs = serialize_tile_outputs(tile)
        if set(outputs) != set(OUTPUT_NAMES):
            raise AssertionError("Ground texture output set changed")
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
    "OUTPUT_SCHEMA",
    "GroundTextureTile",
    "GroundTextureWindow",
    "compile_aligned_window",
    "serialize_tile_outputs",
    "slice_tile",
    "write_tile_outputs",
]
