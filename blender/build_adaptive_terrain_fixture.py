"""Build the deterministic 2 x 2 qualification fixture for adaptive terrain.

The fixture is deliberately self-contained: it synthesizes four contiguous
canonical 2 m MNT grids, compiles the three nested FVTQ LODs, validates every
shared edge, regenerates portable OpenUSD payloads, and emits a streaming-cost
catalog.  It performs no network access and has no orthophoto dependency.

The builder accepts only an output on ``D:`` on Windows.  This keeps all
FireViewer qualification artifacts away from the system disk.
"""

from __future__ import annotations

import argparse
import binascii
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from typing import Iterable, Mapping, Sequence
import zlib

import numpy as np
from PIL import Image
from shapely.geometry import LineString, box, mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from adaptive_terrain_quadtree import (
        AdaptiveTerrainTile,
        Breakline,
        EDGE_ORDER,
        FvtqMesh,
        SOURCE_SAMPLE_COUNT,
        compile_adaptive_tile,
        encode_fvtq,
        materialize_stitch_triangles,
        quantize_heights_mm,
        quantize_normal_halo_mm,
        write_fvtq,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from blender.adaptive_terrain_quadtree import (
        AdaptiveTerrainTile,
        Breakline,
        EDGE_ORDER,
        FvtqMesh,
        SOURCE_SAMPLE_COUNT,
        compile_adaptive_tile,
        encode_fvtq,
        materialize_stitch_triangles,
        quantize_heights_mm,
        quantize_normal_halo_mm,
        write_fvtq,
    )

try:
    from frustum_streaming import TerrainTileCatalog
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from blender.frustum_streaming import TerrainTileCatalog

try:
    from compact_hag import (
        quantize_hag_max_cm_from_canonical_mm,
        read_hag_max_2m,
        write_hag_max_2m,
    )
    from compile_tile_composition import (
        compile_tile_composition,
        write_tile_composition,
    )
    from tile_package import (
        build_tile_package,
        validate_tile_done,
        write_tile_done,
    )
    from ground_material_contract import (
        build_ground_material_bundle,
        material_identity,
    )
    from clean_pbr_texture_library import (
        LIBRARY_SCHEMA as CLEAN_PBR_LIBRARY_SCHEMA,
        PENDING_LIBRARY_STATUS,
        REQUIRED_TEXTURE_ROLES,
        load_texture_contract,
        sha256_file as clean_pbr_sha256_file,
    )
    from ground_context_binding import (
        load_context_contract,
        load_runtime_contract,
        validate_profile_bindings,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from blender.compact_hag import (
        quantize_hag_max_cm_from_canonical_mm,
        read_hag_max_2m,
        write_hag_max_2m,
    )
    from blender.compile_tile_composition import (
        compile_tile_composition,
        write_tile_composition,
    )
    from blender.tile_package import (
        build_tile_package,
        validate_tile_done,
        write_tile_done,
    )
    from blender.ground_material_contract import (
        build_ground_material_bundle,
        material_identity,
    )
    from blender.clean_pbr_texture_library import (
        LIBRARY_SCHEMA as CLEAN_PBR_LIBRARY_SCHEMA,
        PENDING_LIBRARY_STATUS,
        REQUIRED_TEXTURE_ROLES,
        load_texture_contract,
        sha256_file as clean_pbr_sha256_file,
    )
    from blender.ground_context_binding import (
        load_context_contract,
        load_runtime_contract,
        validate_profile_bindings,
    )

try:
    from omniverse.adaptive_terrain_usd import (
        author_lod_usda,
        author_root_usda,
        export_tile_usd,
        validate_tile_usd_package,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script import
    from adaptive_terrain_usd import (  # type: ignore[no-redef]
        author_lod_usda,
        author_root_usda,
        export_tile_usd,
        validate_tile_usd_package,
    )


FIXTURE_SCHEMA = "fireviewer.adaptive-terrain-qualification-fixture.v1"
RECEIPT_SCHEMA = "fireviewer.adaptive-terrain-fixture-receipt.v1"
CATALOG_SCHEMA = "fireviewer.terrain-tile-catalog.v1"
ZONE_ORIGIN_L93_M = (700_000.0, 6_300_000.0)
TILE_SIZE_M = 500.0
SOURCE_RESOLUTION_M = 2.0
GRID_SHAPE = (SOURCE_SAMPLE_COUNT, SOURCE_SAMPLE_COUNT)
COMPOSITION_ASSETS = {
    "ground_profile_ids": "ground-profile-ids.png",
    "ground_profile_weights": "ground-profile-weights.png",
    "ground_confidence": "ground-confidence.png",
    "ground_orientation": "ground-orientation.png",
}

# The features are defined once in zone-global metric coordinates and clipped
# per tile.  Their identifiers therefore do not depend on a package path or a
# worker order.
GLOBAL_BREAKLINES: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = (
    # Explicit global seam locks give both compilers the exact same internal
    # boundary subdivision request before either tile is authored.
    ("fixture-seam-lock-x500", ((500.0, 0.0), (500.0, 1_000.0))),
    ("fixture-seam-lock-y500", ((0.0, 500.0), (1_000.0, 500.0))),
    ("fixture-ridge-main", ((531.25, 0.0), (531.25, 1_000.0))),
    ("fixture-ravine-main", ((0.0, 175.0), (1_000.0, 575.0))),
    ("fixture-cliff-main", ((787.5, 0.0), (587.5, 1_000.0))),
)


@dataclass(frozen=True)
class FixtureBuild:
    output_root: Path
    receipt: Path
    catalog: Path
    material_contract: Path
    tile_roots: tuple[Path, Path, Path, Path]


def _canonical_json(payload: Mapping[str, object]) -> bytes:
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _surface_contracts() -> tuple[dict[str, object], dict[str, object], str, str]:
    variants = {
        "natural_ground": {
            "dry_grass": ["landcover:herbaceous"],
        },
        "agriculture_field": {
            "vineyard_rows": ["agriculture:vineyard"],
        },
        "cliff_surface": {
            "blocky_granite": ["geology:granite", "terrain:cliff"],
        },
        "path_surface": {
            variant: ["transport:path"]
            for variant in ("mediterranean_earth", "rocky_forest", "fire_service")
        },
        "road_surface": {
            variant: ["transport:road", "road:asphalt"]
            for variant in ("asphalt_fine", "asphalt_weathered", "asphalt_patched")
        },
        "railway_bed": {
            variant: ["transport:rail", "rail:active"]
            for variant in ("clean_ballast", "aged_ballast", "concrete_sleepers")
        },
        "watercourse": {
            variant: ["hydro:persistent"]
            for variant in ("clear_gravel", "turquoise_limestone", "rocky_riffle")
        },
    }
    modes = {
        "natural_ground": "ground_blend",
        "agriculture_field": "directional_area",
        "cliff_surface": "slope_cliff_overlay",
        "path_surface": "linear_overlay",
        "road_surface": "linear_overlay",
        "railway_bed": "linear_overlay",
        "watercourse": "watercourse_overlay",
    }
    profiles = [
        {
            "id": f"{family}.{variant}",
            "family": family,
            "variant": variant,
            "application_mode": modes[family],
        }
        for family, family_variants in variants.items()
        for variant in family_variants
    ]
    context: dict[str, object] = {
        "schema": "fireviewer.ground-context-contract.v1",
        "profile_bindings": {
            family: {"variant_tags": family_variants}
            for family, family_variants in variants.items()
        },
    }
    context_sha256 = _sha256(_canonical_json(context))
    catalog: dict[str, object] = {
        "schema": "fireviewer.ground-surface-atlas-library.v3",
        "profiles": profiles,
    }
    catalog["catalog_sha256"] = _sha256(_canonical_json(catalog))
    contract_sha256 = _sha256(
        _canonical_json(
            {
                "schema": "fireviewer.tile-composition-contract.v1",
                "grid_cell_size_m": 5,
                "priority": [
                    "natural",
                    "agriculture",
                    "cliff",
                    "path",
                    "road",
                    "rail",
                    "hydro",
                    "crossing_override",
                ],
            }
        )
    )
    return context, catalog, context_sha256, contract_sha256


def _repository_surface_contracts() -> tuple[dict[str, object], str, str]:
    """Load the hash-locked 72-profile bindings for a real atlas probe."""

    context_path = Path(__file__).with_name("ground_context_contract.v1.json")
    runtime_path = Path(__file__).with_name("ground_surface_runtime_contract.v3.json")
    context = load_context_contract(context_path)
    runtime = load_runtime_contract(runtime_path)
    profile_tags = validate_profile_bindings(context, runtime)
    if len(profile_tags) != 72:
        raise ValueError("The repository context must bind exactly 72 profiles")
    return (
        context,
        _sha256(_canonical_json(context)),
        _sha256(_canonical_json(runtime)),
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
    )


def _write_uniform_pbr_png(
    path: Path, *, width: int, height: int, role: str, seed: int
) -> None:
    """Write a contract-sized valid PNG without allocating a full atlas."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if role == "height":
        bit_depth, colour_type = 16, 0
        pixel = struct.pack(">H", (seed * 257) & 0xFFFF)
    else:
        bit_depth, colour_type = 8, 6
        pixel = bytes((seed & 0xFF, (seed + 37) & 0xFF, (seed + 83) & 0xFF, 255))
    row = b"\0" + pixel * width
    compressor = zlib.compressobj(level=6)
    compressed: list[bytes] = []
    for _ in range(height):
        block = compressor.compress(row)
        if block:
            compressed.append(block)
    compressed.append(compressor.flush())
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, colour_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"".join(compressed))
        + _png_chunk(b"IEND", b"")
    )


def _fixture_artifact(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_synthetic_clean_pbr_library(source_root: Path) -> Path:
    """Create a complete clean-library contract used only by qualification."""

    texture_contract_path = Path(__file__).with_name(
        "ground_surface_texture_contract.v4.json"
    )
    contract = load_texture_contract(texture_contract_path)
    atlas = contract["runtime_atlas"]
    runtime_atlases: dict[str, dict[str, object]] = {}
    for role_index, role in enumerate(REQUIRED_TEXTURE_ROLES):
        relative = f"runtime-atlas/{role}.png"
        _write_uniform_pbr_png(
            source_root / relative,
            width=int(atlas["width_px"]),
            height=int(atlas["height_px"]),
            role=role,
            seed=17 + role_index,
        )
        runtime_atlases[role] = _fixture_artifact(source_root, relative)

    profiles: list[dict[str, object]] = []
    profile_ids = contract["profile_contract"]["stable_profile_ids"]
    for index, profile_id in enumerate(profile_ids):
        textures: dict[str, dict[str, object]] = {}
        for role_index, role in enumerate(REQUIRED_TEXTURE_ROLES):
            relative = f"profile-sources/{index:02d}/{role}.png"
            _write_uniform_pbr_png(
                source_root / relative,
                width=4,
                height=4,
                role=role,
                seed=(index * 7 + role_index * 41) & 0xFF,
            )
            textures[role] = _fixture_artifact(source_root, relative)
        column = index % int(atlas["columns"])
        row = index // int(atlas["columns"])
        profiles.append(
            {
                "stable_index": index,
                "id": profile_id,
                "surface_basis": "atlas_pbr",
                "atlas_slot": index,
                "atlas_uv": {
                    "offset": [
                        (column * int(atlas["cell_size_px"]) + int(atlas["gutter_px"]))
                        / int(atlas["width_px"]),
                        (row * int(atlas["cell_size_px"]) + int(atlas["gutter_px"]))
                        / int(atlas["height_px"]),
                    ],
                    "scale": [
                        (int(atlas["cell_size_px"]) - 2 * int(atlas["gutter_px"]))
                        / int(atlas["width_px"]),
                        (int(atlas["cell_size_px"]) - 2 * int(atlas["gutter_px"]))
                        / int(atlas["height_px"]),
                    ],
                },
                "physical_scale_m": 4.0,
                "projection": (
                    "world_triplanar"
                    if str(profile_id).startswith("cliff_surface.")
                    else "world_xy"
                ),
                "source_kind": "clean_pbr_profile_texture",
                "textures": textures,
            }
        )
    payload = {
        "schema": CLEAN_PBR_LIBRARY_SCHEMA,
        "status": PENDING_LIBRARY_STATUS,
        "texture_contract_sha256": clean_pbr_sha256_file(texture_contract_path),
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
    manifest = source_root / "clean-pbr-texture-library.v1.json"
    _atomic_write(manifest, _canonical_json(payload))
    return manifest


def _surface_features() -> list[dict[str, object]]:
    west, south = ZONE_ORIGIN_L93_M
    coverage = box(west, south, west + 1_000.0, south + 1_000.0)
    vineyard = box(
        west + 402.0,
        south + 402.0,
        west + 648.0,
        south + 598.0,
    )

    def feature(
        feature_id: str,
        layer_id: str,
        geometry: object,
        properties: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "feature_id": feature_id,
            "layer_id": layer_id,
            "geometry": mapping(geometry),
            "properties": dict(properties),
        }

    return [
        feature(
            "fixture:landcover:grass",
            "landcover",
            coverage,
            {"code_cs": "CS2.2.1", "code_us": ""},
        ),
        feature(
            "fixture:geology:granite",
            "geology",
            coverage,
            {"formation": "granite"},
        ),
        feature(
            "fixture:parcel:vineyard",
            "agricultural_parcels",
            vineyard,
            {"cat_cult_p": "vigne"},
        ),
        feature(
            "fixture:land-parcel:vineyard",
            "land_parcels",
            vineyard,
            {"idu": "fixture-vineyard"},
        ),
        feature(
            "fixture:cliff:granite",
            "cliffs",
            box(west + 742.0, south, west + 823.0, south + 1_000.0),
            {"geology": "granite", "aspect_deg": 90.0},
        ),
        feature(
            "fixture:path:west-east",
            "roads",
            LineString([(west, south + 200.0), (west + 1_000.0, south + 200.0)]),
            {"nature": "chemin empierré", "width_m": 3.0},
        ),
        feature(
            "fixture:rail:west-east",
            "railways",
            LineString([(west, south + 300.0), (west + 1_000.0, south + 300.0)]),
            {"etat": "en service", "width_m": 6.0},
        ),
        feature(
            "fixture:road:west-east",
            "roads",
            LineString([(west, south + 350.0), (west + 1_000.0, south + 350.0)]),
            {"nature": "route revêtue", "width_m": 7.0},
        ),
        feature(
            "fixture:river:west-east",
            "hydro_lines",
            LineString([(west, south + 425.0), (west + 1_000.0, south + 425.0)]),
            {
                "persistance": "permanent",
                "flow_direction": "forward",
                "width_m": 5.0,
            },
        ),
    ]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _require_d_output(output_root: Path) -> Path:
    resolved = output_root.resolve(strict=False)
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise ValueError("The adaptive terrain fixture output must be on D:")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError("The fixture output must be absent or empty")
    return resolved


def _synthetic_surface_m(
    tile_x: int, tile_y: int, local_axis: np.ndarray
) -> np.ndarray:
    """Sample the continuous fixture surface on an arbitrary tile-local axis."""

    local_northing, local_easting = np.meshgrid(local_axis, local_axis, indexing="ij")
    easting = local_easting + tile_x * TILE_SIZE_M
    northing = local_northing + tile_y * TILE_SIZE_M

    # A broad plane leaves a genuinely low-error sector in the south-west.
    surface = 96.0 + 0.004 * easting + 0.002 * northing
    # Narrow ridge, oblique ravine, and a sharp geological step exercise the
    # adaptive thresholds without introducing any non-terrain object.
    surface += 12.0 * np.exp(-(((easting - 531.25) / 35.0) ** 2))
    ravine_axis = 0.4 * easting + 175.0
    surface -= 8.0 * np.exp(-(((northing - ravine_axis) / 25.0) ** 2))
    cliff_axis = easting + 0.2 * northing - 787.5
    surface += 6.0 * (np.tanh(cliff_axis / 18.0) + 1.0)
    return surface


def _synthetic_height_mm(tile_x: int, tile_y: int) -> np.ndarray:
    """Return one exact 500 m core sampled from the continuous fixture surface."""

    local_axis = np.arange(SOURCE_SAMPLE_COUNT, dtype="float64") * SOURCE_RESOLUTION_M
    return quantize_heights_mm(_synthetic_surface_m(tile_x, tile_y, local_axis))


def _synthetic_normal_halo_m(tile_x: int, tile_y: int) -> np.ndarray:
    """Return the canonical one-sample 2 m halo used for terrain normals."""

    local_axis = (
        np.arange(-1, SOURCE_SAMPLE_COUNT + 1, dtype="float64") * SOURCE_RESOLUTION_M
    )
    return _synthetic_surface_m(tile_x, tile_y, local_axis)


def _clip_segment_to_box(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip one segment with deterministic Liang-Barsky parameters."""

    x0, y0 = start
    dx = end[0] - x0
    dy = end[1] - y0
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    lower = 0.0
    upper = 1.0
    for direction, distance in (
        (-dx, x0 - minimum_x),
        (dx, maximum_x - x0),
        (-dy, y0 - minimum_y),
        (dy, maximum_y - y0),
    ):
        if direction == 0.0:
            if distance < 0.0:
                return None
            continue
        ratio = distance / direction
        if direction < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    clipped_start = (x0 + lower * dx, y0 + lower * dy)
    clipped_end = (x0 + upper * dx, y0 + upper * dy)
    if clipped_start == clipped_end:
        return None
    return clipped_start, clipped_end


def _tile_breaklines(tile_x: int, tile_y: int) -> tuple[Breakline, ...]:
    west = tile_x * TILE_SIZE_M
    south = tile_y * TILE_SIZE_M
    bounds = (west, south, west + TILE_SIZE_M, south + TILE_SIZE_M)
    clipped: list[Breakline] = []
    for feature_id, points in GLOBAL_BREAKLINES:
        fragments: list[tuple[float, float]] = []
        for start, end in zip(points, points[1:], strict=False):
            segment = _clip_segment_to_box(start, end, bounds)
            if segment is None:
                continue
            for point in segment:
                local = (point[0] - west, point[1] - south)
                if not fragments or fragments[-1] != local:
                    fragments.append(local)
        if len(fragments) >= 2:
            clipped.append(Breakline.from_metres(feature_id, fragments))
    return tuple(sorted(clipped, key=lambda item: item.feature_id))


def _tile_id(origin_l93_m: tuple[float, float]) -> str:
    return f"x{int(origin_l93_m[0])}_y{int(origin_l93_m[1])}"


def _edge_samples(mesh: FvtqMesh, edge_name: str) -> tuple[tuple[int, int], ...]:
    edge_number = EDGE_ORDER.index(edge_name)
    axis = 1 if edge_name in {"west", "east"} else 0
    return tuple(
        (
            mesh.vertices[index][axis],
            mesh.vertices[index][2] + mesh.z_origin_mm,
        )
        for index in mesh.edge_vertex_indices[edge_number]
    )


def _validate_source_grid_seams(
    grids: Mapping[tuple[int, int], np.ndarray],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for tile_x, tile_y in sorted(grids):
        current = grids[(tile_x, tile_y)]
        for direction, neighbour_index, current_values, neighbour_edge in (
            ("east", (tile_x + 1, tile_y), current[:, -1], "west"),
            ("north", (tile_x, tile_y + 1), current[-1, :], "south"),
        ):
            neighbour = grids.get(neighbour_index)
            if neighbour is None:
                continue
            neighbour_values = (
                neighbour[:, 0] if neighbour_edge == "west" else neighbour[0, :]
            )
            if not np.array_equal(current_values, neighbour_values):
                raise ValueError(
                    f"Synthetic MNT discontinuity on {tile_x},{tile_y} {direction}"
                )
            records.append(
                {
                    "tile": [tile_x, tile_y],
                    "neighbour": list(neighbour_index),
                    "direction": direction,
                    "sample_count": int(current_values.size),
                    "edge_sha256": _sha256(
                        np.asarray(current_values, dtype="<i4").tobytes(order="C")
                    ),
                }
            )
    return records


def _validate_fvtq_seams(
    compiled: Mapping[tuple[int, int], AdaptiveTerrainTile],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for tile_x, tile_y in sorted(compiled):
        current = compiled[(tile_x, tile_y)]
        for direction, neighbour_index, current_edge, neighbour_edge in (
            ("east", (tile_x + 1, tile_y), "east", "west"),
            ("north", (tile_x, tile_y + 1), "north", "south"),
        ):
            neighbour = compiled.get(neighbour_index)
            if neighbour is None:
                continue
            for lod in range(3):
                left_mesh = current.lods[lod]
                right_mesh = neighbour.lods[lod]
                left_number = EDGE_ORDER.index(current_edge)
                right_number = EDGE_ORDER.index(neighbour_edge)
                left = _edge_samples(left_mesh, current_edge)
                right = _edge_samples(right_mesh, neighbour_edge)
                if left != right:
                    raise ValueError(
                        f"FVTQ LOD{lod} seam mismatch on {tile_x},{tile_y} {direction}"
                    )
                if (
                    left_mesh.edge_signatures[left_number]
                    != right_mesh.edge_signatures[right_number]
                ):
                    raise ValueError(
                        f"FVTQ LOD{lod} edge-signature mismatch on "
                        f"{tile_x},{tile_y} {direction}"
                    )
                records.append(
                    {
                        "tile": [tile_x, tile_y],
                        "neighbour": list(neighbour_index),
                        "direction": direction,
                        "lod": lod,
                        "vertex_count": len(left),
                        "edge_signature_sha256": left_mesh.edge_signatures[
                            left_number
                        ].hex(),
                    }
                )
    return records


def _synthetic_hag_m(tile_x: int, tile_y: int, local_axis: np.ndarray) -> np.ndarray:
    local_northing, local_easting = np.meshgrid(local_axis, local_axis, indexing="ij")
    easting = local_easting + tile_x * TILE_SIZE_M
    northing = local_northing + tile_y * TILE_SIZE_M
    first_patch = ((easting - 260.0) ** 2 + (northing - 760.0) ** 2) < 130.0**2
    second_patch = ((easting - 760.0) ** 2 + (northing - 720.0) ** 2) < 95.0**2
    return np.where(first_patch, 12.5, np.where(second_patch, 8.75, 0.0))


def _synthetic_mns_m(mnt_mm: np.ndarray, tile_x: int, tile_y: int) -> np.ndarray:
    """Add a compact deterministic height-above-ground reservation only."""

    local_axis = np.arange(SOURCE_SAMPLE_COUNT, dtype="float64") * SOURCE_RESOLUTION_M
    return mnt_mm.astype("float64") / 1_000.0 + _synthetic_hag_m(
        tile_x, tile_y, local_axis
    )


def _synthetic_mns_normal_halo_mm(
    mnt_normal_halo_m: np.ndarray, tile_x: int, tile_y: int
) -> np.ndarray:
    local_axis = (
        np.arange(-1, SOURCE_SAMPLE_COUNT + 1, dtype="float64") * SOURCE_RESOLUTION_M
    )
    values_m = mnt_normal_halo_m + _synthetic_hag_m(tile_x, tile_y, local_axis)
    return np.floor(values_m * 1_000.0 + 0.5).astype("<i4")


def _resource_cost(mesh: FvtqMesh, fvtq_path: Path) -> dict[str, object]:
    # CPU is the known canonical payload size.  GPU covers float3 positions and
    # uint3 triangle indices; material buffers are qualified separately.
    stitch_triangle_counts = [
        len(materialize_stitch_triangles(mesh, mask)) for mask in range(16)
    ]
    return {
        "cpu_bytes": fvtq_path.stat().st_size,
        "gpu_bytes": len(mesh.vertices) * 12 + stitch_triangle_counts[0] * 12,
        "triangles": stitch_triangle_counts[0],
        "stitch_triangle_counts": stitch_triangle_counts,
        "sha256": _sha256_file(fvtq_path),
    }


def _relative_outputs(root: Path, paths: Iterable[Path]) -> dict[str, object]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths)
    }


def _input_artifact(root: Path, path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_count": len(content),
        "sha256": _sha256(content),
    }


def build_fixture(
    output_root: Path, *, surface_library_path: Path | None = None
) -> FixtureBuild:
    """Build and fully self-check one deterministic qualification fixture."""

    root = _require_d_output(Path(output_root))
    root.mkdir(parents=True, exist_ok=True)
    tile_indices = tuple((x, y) for y in range(2) for x in range(2))
    grids = {index: _synthetic_height_mm(*index) for index in tile_indices}
    normal_halos = {index: _synthetic_normal_halo_m(*index) for index in tile_indices}
    source_seams = _validate_source_grid_seams(grids)
    ground_context, seed_catalog, context_sha256, composition_contract_sha256 = (
        _surface_contracts()
    )
    if surface_library_path is not None:
        (
            ground_context,
            context_sha256,
            composition_contract_sha256,
        ) = _repository_surface_contracts()
    shared_material_root = root / "shared" / "ground-material"
    generated_library_root = root / ".fixture-clean-pbr-source"
    source_library = (
        Path(surface_library_path)
        if surface_library_path is not None
        else _write_synthetic_clean_pbr_library(generated_library_root)
    )
    material_contract_path = build_ground_material_bundle(
        source_library, shared_material_root
    )
    if surface_library_path is None:
        shutil.rmtree(generated_library_root)
    composition_catalog = seed_catalog
    ground_material = material_identity(material_contract_path, root)
    ground_material_contract = json.loads(
        material_contract_path.read_text(encoding="utf-8")
    )
    stable_profile_indices = {
        str(profile["id"]): int(profile["index"])
        for profile in ground_material_contract["profile_table"]
    }
    surface_features = _surface_features()
    recipe_id = _sha256(
        _canonical_json(
            {
                "schema": "fireviewer.synthetic-terrain-recipe.v1",
                "quadtree_compiler_sha256": _sha256_file(
                    Path(compile_adaptive_tile.__code__.co_filename)
                ),
                "surface_compiler_sha256": _sha256_file(
                    Path(compile_tile_composition.__code__.co_filename)
                ),
                "hag_writer_sha256": _sha256_file(
                    Path(write_hag_max_2m.__code__.co_filename)
                ),
                "composition_contract_sha256": composition_contract_sha256,
                "context_sha256": context_sha256,
                "composition_seed_catalog_sha256": composition_catalog[
                    "catalog_sha256"
                ],
            }
        )
    )
    source_build_digest = hashlib.sha256()
    source_build_digest.update(b"FIREVIEWER-SYNTHETIC-SOURCE-BUILD-V1\0")
    source_build_digest.update(bytes.fromhex(recipe_id))
    for index, grid in sorted(grids.items()):
        source_build_digest.update(bytes(index))
        source_build_digest.update(np.asarray(grid, dtype="<i4").tobytes(order="C"))
        source_build_digest.update(
            quantize_normal_halo_mm(normal_halos[index])
            .astype("<i4")
            .tobytes(order="C")
        )
    recipe_build_id = source_build_digest.hexdigest()

    compiled: dict[tuple[int, int], AdaptiveTerrainTile] = {}
    tile_roots: list[Path] = []
    all_output_paths: list[Path] = sorted(
        path for path in shared_material_root.rglob("*") if path.is_file()
    )
    rebuild_records: list[dict[str, object]] = []
    catalog_tiles: list[dict[str, object]] = []

    for tile_x, tile_y in tile_indices:
        origin = (
            ZONE_ORIGIN_L93_M[0] + tile_x * TILE_SIZE_M,
            ZONE_ORIGIN_L93_M[1] + tile_y * TILE_SIZE_M,
        )
        identifier = _tile_id(origin)
        tile_root = root / "tiles" / identifier
        tile_roots.append(tile_root)
        source_path = tile_root / "source" / "mnt-2m-mm.npy"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(source_path, grids[(tile_x, tile_y)], allow_pickle=False)
        all_output_paths.append(source_path)
        normal_halo_path = tile_root / "source" / "mnt-normal-halo-2m-mm.npy"
        np.save(
            normal_halo_path,
            quantize_normal_halo_mm(normal_halos[(tile_x, tile_y)]),
            allow_pickle=False,
        )
        all_output_paths.append(normal_halo_path)

        breaklines = _tile_breaklines(tile_x, tile_y)
        tile = compile_adaptive_tile(
            grids[(tile_x, tile_y)].astype("float64") / 1_000.0,
            normal_halo_heights_m=normal_halos[(tile_x, tile_y)],
            tile_origin_l93_m=origin,
            breaklines=breaklines,
        )
        compiled[(tile_x, tile_y)] = tile

        fvtq_paths: list[Path] = []
        first_fvtq: list[bytes] = []
        for mesh in tile.lods:
            path = tile_root / f"terrain-lod{mesh.lod}.fvtq"
            write_fvtq(mesh, path)
            fvtq_paths.append(path)
            first_fvtq.append(path.read_bytes())
            all_output_paths.append(path)

        composition = compile_tile_composition(
            bounds_l93_m=[origin[0], origin[1], origin[0] + 500.0, origin[1] + 500.0],
            features=surface_features,
            context_contract=ground_context,
            atlas_catalog=composition_catalog,
            contract_sha256=composition_contract_sha256,
            context_sha256=context_sha256,
        )
        write_tile_composition(composition, tile_root)
        # The legacy semantic compiler is used only as a deterministic fixture
        # seed.  New packages store the matcher result at 1 m and never ship
        # the temporary vector-overlay representation.
        for name in ("ground-profile-ids.png", "ground-profile-weights.png"):
            path = tile_root / name
            values = np.asarray(Image.open(path), dtype=np.uint8)
            if name == "ground-profile-ids.png":
                source_profile_table = composition.manifest["profile_table"]
                if any(
                    profile_id not in stable_profile_indices
                    for profile_id in source_profile_table
                ):
                    raise ValueError(
                        "Synthetic composition references a profile outside the clean "
                        "72-profile contract"
                    )
                remap = np.zeros(256, dtype=np.uint8)
                for source_index, profile_id in enumerate(source_profile_table):
                    remap[source_index] = stable_profile_indices[profile_id]
                values = remap[values]
            Image.fromarray(
                np.repeat(np.repeat(values, 5, axis=0), 5, axis=1), mode="RGBA"
            ).save(path)
        confidence_path = tile_root / "ground-confidence.png"
        orientation_path = tile_root / "ground-orientation.png"
        Image.fromarray(np.full((500, 500), 240, dtype=np.uint8), mode="L").save(
            confidence_path
        )
        Image.fromarray(np.zeros((500, 500), dtype=np.uint8), mode="L").save(
            orientation_path
        )
        (tile_root / "surface-overlays.json.gz").unlink()
        (tile_root / "tile-composition.json.gz").unlink()
        composition_paths = [
            tile_root / name
            for name in (
                "ground-profile-ids.png",
                "ground-profile-weights.png",
                "ground-confidence.png",
                "ground-orientation.png",
            )
        ]
        correspondence_artifacts = {
            path.name: {
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in composition_paths
        }
        primary_ids = np.asarray(
            Image.open(tile_root / "ground-profile-ids.png"), dtype=np.uint8
        )[:, :, 0]
        unique_ids, unique_counts = np.unique(primary_ids, return_counts=True)
        correspondence_profile_table = [
            {
                "stable_index": profile["index"],
                "id": profile["id"],
                "atlas_slot": profile["atlas_slot"],
                "atlas_uv": profile["atlas_uv"],
                "projection": (
                    "world_triplanar"
                    if str(profile["id"]).startswith("cliff_surface.")
                    else "world_xy"
                ),
                "physical_scale_m": profile["physical_scale_m"],
                "class_id": str(profile["id"]).split(".", 1)[0],
                "textures": {
                    role: {
                        "byte_count": ground_material_contract["runtime_atlas"][
                            "assets"
                        ][role]["byte_count"],
                        "sha256": ground_material_contract["runtime_atlas"]["assets"][
                            role
                        ]["sha256"],
                    }
                    for role in ("basecolor", "normal", "height", "orm")
                },
            }
            for profile in ground_material_contract["profile_table"]
        ]
        class_counts: dict[str, int] = {}
        for profile_id, count in zip(
            unique_ids.tolist(), unique_counts.tolist(), strict=True
        ):
            class_id = correspondence_profile_table[int(profile_id)]["class_id"]
            class_counts[class_id] = class_counts.get(class_id, 0) + int(count)
        correspondence = {
            "schema": "fireviewer.surface-correspondence-tile.v1",
            "status": "compiled_no_orthophoto_payload",
            "crs": "EPSG:2154",
            "bounds_l93_m": [
                origin[0],
                origin[1],
                origin[0] + 500.0,
                origin[1] + 500.0,
            ],
            "grid": {"resolution_m": 1, "width": 500, "height": 500},
            "identity": {
                "orthophoto_source_sha256": _sha256(b"synthetic-offline-source"),
                "orthophoto_tile_input_sha256": _sha256(
                    f"synthetic-tile:{identifier}".encode("utf-8")
                ),
                "pbr_library_sha256": ground_material["source_library_identity_sha256"],
                "correspondence_model_sha256": _sha256(
                    b"synthetic-correspondence-model"
                ),
                "algorithm_sha256": _sha256_file(Path(__file__)),
                "contract_sha256": composition_contract_sha256,
                "context_priors_sha256": context_sha256,
                "approved_corrections_sha256": _sha256(b"no-approved-corrections"),
            },
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
            "profile_table": correspondence_profile_table,
            "primary_pixel_counts_by_class": dict(sorted(class_counts.items())),
            "restriction_pixel_counts": {},
            "artifacts": correspondence_artifacts,
        }
        correspondence_path = tile_root / "surface-correspondence.json"
        _atomic_write(correspondence_path, _canonical_json(correspondence))
        composition_paths.append(correspondence_path)
        all_output_paths.extend(composition_paths)

        mns_m = _synthetic_mns_m(grids[(tile_x, tile_y)], tile_x, tile_y)
        mns_mm = np.floor(mns_m * 1_000.0 + 0.5).astype("<i4")
        mns_normal_halo_mm = _synthetic_mns_normal_halo_mm(
            normal_halos[(tile_x, tile_y)], tile_x, tile_y
        )
        hag_values = quantize_hag_max_cm_from_canonical_mm(
            quantize_normal_halo_mm(normal_halos[(tile_x, tile_y)]),
            mns_normal_halo_mm,
        )
        hag_path = tile_root / "hag-max-2m.tif"
        write_hag_max_2m(hag_path, hag_values, tile_origin_l93_m=origin)
        restored_hag, hag_metadata = read_hag_max_2m(hag_path)
        if not np.array_equal(restored_hag, hag_values):
            raise ValueError(f"HAG round-trip mismatch for {identifier}")
        if hag_metadata["bounds_l93_m"] != [
            origin[0],
            origin[1],
            origin[0] + TILE_SIZE_M,
            origin[1] + TILE_SIZE_M,
        ]:
            raise ValueError(f"HAG bounds mismatch for {identifier}")
        all_output_paths.append(hag_path)

        mns_path = tile_root / "source" / "mns-2m-mm.npy"
        np.save(mns_path, mns_mm, allow_pickle=False)
        all_output_paths.append(mns_path)
        mns_normal_halo_path = tile_root / "source" / "mns-normal-halo-2m-mm.npy"
        np.save(mns_normal_halo_path, mns_normal_halo_mm, allow_pickle=False)
        all_output_paths.append(mns_normal_halo_path)
        tile_bounds = [
            origin[0],
            origin[1],
            origin[0] + TILE_SIZE_M,
            origin[1] + TILE_SIZE_M,
        ]
        package = build_tile_package(
            tile_root,
            tile_id=identifier,
            recipe_id=recipe_id,
            recipe_build_id=recipe_build_id,
            bounds_l93_m=tile_bounds,
            inputs={
                "mnt_2m": _input_artifact(tile_root, source_path),
                "mnt_normal_halo_2m": _input_artifact(tile_root, normal_halo_path),
                "mns_2m": _input_artifact(tile_root, mns_path),
                "mns_normal_halo_2m": _input_artifact(tile_root, mns_normal_halo_path),
                "surface_correspondence": _input_artifact(
                    tile_root, correspondence_path
                ),
            },
            ground_material=ground_material,
        )
        write_tile_done(tile_root, package)
        validate_tile_done(tile_root)
        all_output_paths.extend(
            [tile_root / "tile-package.v3.json", tile_root / "tile.done.v3.json"]
        )

        usd_package = export_tile_usd(
            fvtq_paths,
            tile_root,
            tile_id=identifier,
            zone_origin_l93_m=ZONE_ORIGIN_L93_M,
            composition_assets=COMPOSITION_ASSETS,
            ground_material_contract=material_contract_path,
            zone_package_root=root,
        )
        validate_tile_usd_package(usd_package.output_root)
        usd_paths = [
            *usd_package.lod_payloads,
            usd_package.root_stage,
            usd_package.manifest,
        ]
        all_output_paths.extend(usd_paths)

        # Re-open the persisted source and compile from scratch.  The second
        # build stays in memory so no disposable package is left on disk.
        reloaded_mm = np.load(source_path, allow_pickle=False)
        reloaded_normal_halo_mm = np.load(normal_halo_path, allow_pickle=False)
        rebuilt = compile_adaptive_tile(
            reloaded_mm.astype("float64") / 1_000.0,
            normal_halo_heights_m=(reloaded_normal_halo_mm.astype("float64") / 1_000.0),
            tile_origin_l93_m=origin,
            breaklines=tuple(reversed(breaklines)),
        )
        fvtq_matches = [
            encode_fvtq(mesh) == expected
            for mesh, expected in zip(rebuilt.lods, first_fvtq, strict=True)
        ]
        usd_lod_matches = [
            author_lod_usda(mesh) == path.read_bytes()
            for mesh, path in zip(rebuilt.lods, usd_package.lod_payloads, strict=True)
        ]
        root_matches = (
            author_root_usda(
                rebuilt.lods,
                tile_id=identifier,
                zone_origin_l93_m=ZONE_ORIGIN_L93_M,
                composition_assets=COMPOSITION_ASSETS,
                material_layer_asset=Path(
                    os.path.relpath(
                        material_contract_path.parent / "ground-material.usda",
                        tile_root,
                    )
                ).as_posix(),
                material_contract_asset=Path(
                    os.path.relpath(material_contract_path, tile_root)
                ).as_posix(),
            )
            == usd_package.root_stage.read_bytes()
        )
        if not all((*fvtq_matches, *usd_lod_matches, root_matches)):
            raise ValueError(f"Bitwise rebuild mismatch for {identifier}")
        rebuild_records.append(
            {
                "tile_id": identifier,
                "fvtq_lods_match": fvtq_matches,
                "usd_lods_match": usd_lod_matches,
                "usd_root_match": root_matches,
            }
        )

        lod2 = tile.lods[2]
        minimum_z = (lod2.z_origin_mm + lod2.minimum_relative_height_mm) / 1_000.0
        maximum_z = (lod2.z_origin_mm + lod2.maximum_relative_height_mm) / 1_000.0
        catalog_tiles.append(
            {
                "id": identifier,
                "grid_x": int(origin[0] / TILE_SIZE_M),
                "grid_y": int(origin[1] / TILE_SIZE_M),
                "bounds_l93_ngf_m": [
                    origin[0],
                    origin[1],
                    minimum_z,
                    origin[0] + TILE_SIZE_M,
                    origin[1] + TILE_SIZE_M,
                    maximum_z,
                ],
                "stitch_masks": list(range(16)),
                "resource_costs": {
                    f"lod{mesh.lod}": _resource_cost(mesh, fvtq_paths[mesh.lod])
                    for mesh in tile.lods
                },
            }
        )

    fvtq_seams = _validate_fvtq_seams(compiled)
    final_build_digest = hashlib.sha256()
    final_build_digest.update(b"FIREVIEWER-SYNTHETIC-FINAL-BUILD-V1\0")
    final_build_digest.update(bytes.fromhex(recipe_build_id))
    for tile_root in sorted(tile_roots):
        final_build_digest.update(tile_root.name.encode("utf-8"))
        final_build_digest.update(
            bytes.fromhex(_sha256_file(tile_root / "tile.done.v3.json"))
        )
    final_build_id = final_build_digest.hexdigest()
    for record in catalog_tiles:
        record["build_id"] = final_build_id
    catalog_payload: dict[str, object] = {
        "schema": CATALOG_SCHEMA,
        "crs": "EPSG:2154",
        "cost_model": {
            "cpu_bytes": "canonical_fvtq_payload",
            "gpu_bytes": "float3_positions_plus_uint3_triangle_indices",
            "triangles": "fvtq_triangle_count",
        },
        "tiles": catalog_tiles,
    }
    TerrainTileCatalog.from_manifest(catalog_payload)
    catalog_path = root / "terrain-tile-catalog.v1.json"
    _atomic_write(catalog_path, _canonical_json(catalog_payload))
    all_output_paths.append(catalog_path)

    algorithm_hashes = {
        "fixture_builder_sha256": _sha256_file(Path(__file__)),
        "quadtree_compiler_sha256": _sha256_file(
            Path(compile_adaptive_tile.__code__.co_filename)
        ),
        "usd_exporter_sha256": _sha256_file(Path(export_tile_usd.__code__.co_filename)),
        "surface_compiler_sha256": _sha256_file(
            Path(compile_tile_composition.__code__.co_filename)
        ),
        "hag_writer_sha256": _sha256_file(Path(write_hag_max_2m.__code__.co_filename)),
    }
    build_digest = hashlib.sha256()
    build_digest.update(FIXTURE_SCHEMA.encode("ascii"))
    for key, value in sorted(algorithm_hashes.items()):
        build_digest.update(key.encode("utf-8"))
        build_digest.update(bytes.fromhex(value))
    for path in sorted(all_output_paths):
        build_digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        build_digest.update(bytes.fromhex(_sha256_file(path)))

    receipt_payload: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "fixture_schema": FIXTURE_SCHEMA,
        "status": "accepted_synthetic",
        "build_id": final_build_id,
        "fixture_artifact_set_sha256": build_digest.hexdigest(),
        "crs": "EPSG:2154",
        "zone_origin_l93_m": list(ZONE_ORIGIN_L93_M),
        "extent_m": [1_000.0, 1_000.0],
        "source": {
            "kind": "synthetic_mnt",
            "tile_count": 4,
            "resolution_m": SOURCE_RESOLUTION_M,
            "grid_shape": list(GRID_SHAPE),
            "source_seam_checks": source_seams,
        },
        "geometry": {
            "fvtq_payload_count": 12,
            "stitch_masks_per_lod": list(range(16)),
            "stitch_variant_count": 12 * 16,
            "shared_edge_checks": fvtq_seams,
            "bitwise_rebuild_checks": rebuild_records,
        },
        "usd": {
            "package_count": 4,
            "validated_package_count": 4,
            "primary_camera_allowed_lods": [0],
            "terrain_lod_aov": "fireviewer:terrain_lod",
        },
        "compact_hag": {
            "format": "fireviewer.hag-max-2m.v1",
            "payload_count": 4,
            "unit": "centimetre",
            "nodata": 65535,
        },
        "surface_composition": {
            "package_count": 4,
            "grid_cell_size_m": 1,
            "grid_size_px": [500, 500],
            "weights_sum": 255,
            "confidence": "L8 matcher evidence",
            "orientation": "L8 undirected axis 0..pi",
            "runtime_overlays": False,
        },
        "canonical_packages": {
            "schema": "fireviewer.tile-package.v3",
            "tile_package_count": 4,
            "tile_done_count": 4,
            "recipe_id": recipe_id,
            "recipe_build_id": recipe_build_id,
            "build_id": final_build_id,
            "ground_material_contract_sha256": ground_material["contract_sha256"],
        },
        "streaming_catalog": catalog_path.relative_to(root).as_posix(),
        "algorithm_hashes": algorithm_hashes,
        "prohibited_dependencies": {
            "network_requests": 0,
            "orthophoto": False,
            "blender_runtime": False,
        },
        "outputs": _relative_outputs(root, all_output_paths),
    }
    receipt_path = root / "fixture.acceptance.v1.json"
    _atomic_write(receipt_path, _canonical_json(receipt_payload))
    return FixtureBuild(
        output_root=root,
        receipt=receipt_path,
        catalog=catalog_path,
        material_contract=material_contract_path,
        tile_roots=tuple(tile_roots),  # type: ignore[arg-type]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the offline adaptive-terrain 2x2 qualification fixture."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Fresh or empty output directory on D:.",
    )
    parser.add_argument(
        "--surface-library",
        type=Path,
        help=(
            "Optional clean PBR library v1. It is copied once into the zone's "
            "shared material bundle; tests synthesize the same production schema."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = build_fixture(
        arguments.output,
        surface_library_path=arguments.surface_library,
    )
    print(result.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
