"""Build one terrain-only 500 m tile from aligned native IGN MNT/MNS rasters.

This stage deliberately excludes buildings, roads, vegetation and simulation
content. The MNT authors the bare 3D terrain; the co-located MNS is retained
and validated now so later object layers can be grounded without changing the
terrain grid. The lightweight 2D ground material map is derived locally from
the same MNT/MNS pair; orthophotos and other aerial imagery are forbidden.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from shapely.geometry import box

from prepare_mid_vegetation_05m import (
    _crop_grid_to_bounds,
    _mosaic,
    build_local_terrain_mesh,
)


SCHEMA = "fireviewer.incident-terrain-tile-0m50.v1"
TERRAIN_CONTRACT = "bare-mnt-square-grid-with-colocated-mns.v1"


def build_terrain_outputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    core_bounds = tuple(float(value) for value in args.bounds)
    processing_bounds = tuple(
        float(value)
        for value in (getattr(args, "processing_bounds", None) or core_bounds)
    )
    if len(core_bounds) != 4 or len(processing_bounds) != 4:
        raise ValueError("bounds and processing_bounds must contain four values")
    if not box(*processing_bounds).covers(box(*core_bounds)):
        raise ValueError("processing_bounds must cover the complete core bounds")

    mnt, transform, mnt_sources = _mosaic(args.mnt, processing_bounds)
    mns, mns_transform, mns_sources = _mosaic(args.mns, processing_bounds)
    if mnt.shape != mns.shape or tuple(transform) != tuple(mns_transform):
        raise ValueError("MNT and MNS mosaics are not pixel-aligned")

    core_mnt, core_transform = _crop_grid_to_bounds(mnt, transform, core_bounds)
    core_mns, core_mns_transform = _crop_grid_to_bounds(
        mns, mns_transform, core_bounds
    )
    if core_mnt.shape != core_mns.shape or tuple(core_transform) != tuple(
        core_mns_transform
    ):
        raise ValueError("Core MNT and MNS grids are not pixel-aligned")

    finite_mnt = np.isfinite(core_mnt)
    finite_mns = np.isfinite(core_mns)
    if not finite_mnt.all() or not finite_mns.all():
        raise ValueError("A terrain tile must have complete finite MNT and MNS coverage")

    origin = (
        float(args.origin_x)
        if args.origin_x is not None
        else round((core_bounds[0] + core_bounds[2]) / 2),
        float(args.origin_y)
        if args.origin_y is not None
        else round((core_bounds[1] + core_bounds[3]) / 2),
        float(args.origin_z)
        if args.origin_z is not None
        else math.floor(float(core_mnt.min())),
    )
    valid_processing_mask = np.isfinite(mnt)
    terrain = build_local_terrain_mesh(
        mnt,
        transform,
        origin,
        valid_mask=valid_processing_mask,
        step_pixels=int(args.terrain_step_pixels),
        coordinate_precision=3,
        bounds=core_bounds,
    )
    canopy_delta = core_mns - core_mnt
    map_size = int(getattr(args, "ground_material_map_size", 256))
    if map_size < 32 or map_size > 1_024:
        raise ValueError("ground_material_map_size must be between 32 and 1024")
    rows = np.linspace(0, core_mnt.shape[0] - 1, map_size).round().astype(int)
    columns = np.linspace(0, core_mnt.shape[1] - 1, map_size).round().astype(int)
    sampled_mnt = core_mnt[np.ix_(rows, columns)]
    sampled_delta = np.maximum(core_mns[np.ix_(rows, columns)] - sampled_mnt, 0.0)
    spacing_x = (core_bounds[2] - core_bounds[0]) / max(map_size - 1, 1)
    spacing_y = (core_bounds[3] - core_bounds[1]) / max(map_size - 1, 1)
    gradient_y, gradient_x = np.gradient(sampled_mnt, spacing_y, spacing_x)
    slope = np.hypot(gradient_x, gradient_y)
    slope_weight = np.clip(slope / math.tan(math.radians(42.0)), 0.0, 1.0)
    roughness = np.hypot(*np.gradient(slope))
    roughness_weight = np.clip(roughness / 0.35, 0.0, 1.0)
    reservation_weight = np.clip(sampled_delta / 8.0, 0.0, 1.0)
    ground_material_map = np.stack(
        (
            np.clip(1.0 - slope_weight, 0.0, 1.0),
            slope_weight,
            roughness_weight,
            reservation_weight,
        ),
        axis=-1,
    )
    ground_material_map = np.rint(ground_material_map * 255.0).astype("uint8")
    package = {
        "schema": SCHEMA,
        "terrain_contract": TERRAIN_CONTRACT,
        "metadata": {
            "crs": "EPSG:2154",
            "axis_convention": "X=east, Y=north, Z=up",
            "linear_unit": "metre",
            "bounds_l93_m": list(core_bounds),
            "processing_bounds_l93_m": list(processing_bounds),
            "origin_l93_m": list(origin),
            "raster_transform": list(core_transform),
            "native_resolution_m": 0.5,
            "sources": {"mnt": mnt_sources, "mns": mns_sources},
            "authored_layers": ["bare_terrain_3d"],
            "deferred_layers": [
                "buildings",
                "roads",
                "small_specific_assets",
                "vegetation",
                "simulation",
            ],
            "ground_2d": {
                "file_name": "ground-material-map.png",
                "format": "rgba_png",
                "pixel_size": [map_size, map_size],
                "channels": {
                    "r": "flat_ground_weight_from_mnt",
                    "g": "steep_rock_weight_from_mnt",
                    "b": "rough_transition_weight_from_mnt",
                    "a": "future_object_reservation_from_mns_minus_mnt",
                },
                "orthophoto_dependency": "forbidden",
            },
        },
        "terrain": terrain,
        "mns_alignment": {
            "shape": list(core_mns.shape),
            "transform": list(core_mns_transform),
            "finite_sample_count": int(finite_mns.sum()),
            "mns_below_mnt_sample_count": int((canopy_delta < -0.01).sum()),
            "minimum_delta_m": float(canopy_delta.min()),
            "maximum_delta_m": float(canopy_delta.max()),
        },
        "statistics": {
            "mnt_sample_count": int(finite_mnt.sum()),
            "mns_sample_count": int(finite_mns.sum()),
            "minimum_mnt_m": float(core_mnt.min()),
            "maximum_mnt_m": float(core_mnt.max()),
            "terrain_vertex_count": len(terrain["vertices"]),
            "terrain_face_count": len(terrain["faces"]),
            "vegetation_instance_count": 0,
            "building_count": 0,
            "route_count": 0,
            "small_asset_count": 0,
        },
    }
    return package, ground_material_map


def build_terrain_package(args: argparse.Namespace) -> dict[str, Any]:
    package, _ground_material_map = build_terrain_outputs(args)
    return package


def write_terrain_package(package: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        package, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    temporary.replace(output)


def write_ground_material_map(values: np.ndarray, output: Path) -> None:
    from PIL import Image

    if values.ndim != 3 or values.shape[2] != 4 or values.dtype != np.uint8:
        raise ValueError("Ground material map must be an RGBA uint8 array")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    Image.fromarray(values, mode="RGBA").save(
        temporary, format="PNG", optimize=True, compress_level=9
    )
    temporary.replace(output)


def load_terrain_package(path: Path) -> dict[str, Any]:
    payload = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported incident terrain tile package")
    return payload


def validate_terrain_package(
    path: Path, tile: dict[str, Any]
) -> dict[str, Any]:
    package = load_terrain_package(path)
    metadata = package.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Terrain package metadata is absent")
    for key in ("bounds_l93_m", "processing_bounds_l93_m", "origin_l93_m"):
        expected = tile["origin_l93_m" if key == "origin_l93_m" else key]
        observed = metadata.get(key)
        if not isinstance(observed, list) or len(observed) != len(expected) or any(
            abs(float(left) - float(right)) > 0.001
            for left, right in zip(observed, expected, strict=True)
        ):
            raise ValueError(f"Terrain package {key} does not match its tile")
    if package.get("terrain_contract") != TERRAIN_CONTRACT:
        raise ValueError("Terrain package contract is invalid")
    if (
        metadata.get("crs") != "EPSG:2154"
        or metadata.get("linear_unit") != "metre"
        or not math.isclose(
            float(metadata.get("native_resolution_m", math.nan)),
            0.5,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Terrain package geospatial reference is invalid")
    bounds = [float(value) for value in metadata["bounds_l93_m"]]
    if not math.isclose(
        bounds[2] - bounds[0], bounds[3] - bounds[1], abs_tol=1e-6
    ):
        raise ValueError("Terrain package bounds are not square")
    transform = metadata.get("raster_transform")
    if (
        not isinstance(transform, list)
        or len(transform) != 9
        or not math.isclose(float(transform[0]), 0.5, abs_tol=1e-9)
        or not math.isclose(float(transform[4]), -0.5, abs_tol=1e-9)
        or not math.isclose(float(transform[1]), 0.0, abs_tol=1e-12)
        or not math.isclose(float(transform[3]), 0.0, abs_tol=1e-12)
    ):
        raise ValueError("Terrain package raster grid is invalid")

    ground_2d = metadata.get("ground_2d")
    if (
        not isinstance(ground_2d, dict)
        or ground_2d.get("orthophoto_dependency") != "forbidden"
    ):
        raise ValueError("Terrain package ground context contract is invalid")
    ground_map = path.parent / str(ground_2d.get("file_name", ""))
    if not ground_map.is_file():
        raise ValueError("Terrain package ground material map is absent")
    from PIL import Image

    with Image.open(ground_map) as image:
        expected_size = tuple(ground_2d["pixel_size"])
        if image.mode != "RGBA" or image.size != expected_size:
            raise ValueError("Terrain package ground material map is invalid")

    terrain = package.get("terrain")
    if not isinstance(terrain, dict):
        raise ValueError("Terrain mesh is absent")
    vertices = np.asarray(terrain.get("vertices"), dtype="float64")
    faces = np.asarray(terrain.get("faces"), dtype="int64")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Terrain mesh vertices are invalid")
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise ValueError("Terrain mesh faces are invalid")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("Terrain mesh face indices are invalid")
    if (
        terrain.get("vertex_count") != len(vertices)
        or terrain.get("face_count") != len(faces)
        or terrain.get("geometric_bounds_l93_m") != metadata["bounds_l93_m"]
    ):
        raise ValueError("Terrain mesh metadata is inconsistent")
    spacing = terrain.get("sample_spacing_m")
    if not isinstance(spacing, list) or len(spacing) != 2:
        raise ValueError("Terrain mesh spacing is absent")
    expected_columns = int(round((bounds[2] - bounds[0]) / float(spacing[0]))) + 1
    expected_rows = int(round((bounds[3] - bounds[1]) / float(spacing[1]))) + 1
    if len(vertices) != expected_rows * expected_columns or len(faces) != (
        expected_rows - 1
    ) * (expected_columns - 1):
        raise ValueError("Terrain mesh does not completely cover its square grid")
    origin = np.asarray(metadata["origin_l93_m"], dtype="float64")
    world = vertices + origin
    observed_bounds = [
        float(world[:, 0].min()),
        float(world[:, 1].min()),
        float(world[:, 0].max()),
        float(world[:, 1].max()),
    ]
    if any(
        not math.isclose(observed, expected, abs_tol=0.001)
        for observed, expected in zip(observed_bounds, bounds, strict=True)
    ):
        raise ValueError("Terrain mesh does not reach its exact Lambert-93 bounds")

    alignment = package.get("mns_alignment")
    if (
        not isinstance(alignment, dict)
        or alignment.get("transform") != transform
        or int(alignment.get("finite_sample_count", -1))
        != int(np.prod(alignment.get("shape", [])))
    ):
        raise ValueError("Terrain MNS alignment evidence is invalid")
    statistics = package.get("statistics")
    if not isinstance(statistics, dict) or any(
        statistics.get(key) != 0
        for key in (
            "vegetation_instance_count",
            "building_count",
            "route_count",
            "small_asset_count",
        )
    ):
        raise ValueError("Terrain-only package contains a deferred object layer")
    if (
        statistics.get("terrain_vertex_count") != len(vertices)
        or statistics.get("terrain_face_count") != len(faces)
        or statistics.get("mnt_sample_count") != alignment["finite_sample_count"]
        or statistics.get("mns_sample_count") != alignment["finite_sample_count"]
    ):
        raise ValueError("Terrain package statistics are inconsistent")
    return package


def validate_terrain_evidence(
    path: Path,
    tile: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    """Rebuild one tile from its referenced GeoTIFFs and compare every output.

    This is deliberately more expensive than the structural package validator.
    It proves source hashes, CRS, native grid, MNT/MNS co-registration, every
    rounded mesh elevation and every pixel of the lightweight terrain-context
    map. It does not claim that contextual soil, road or building masks exist.
    """

    package = validate_terrain_package(path, tile)
    metadata = package["metadata"]
    sources: dict[str, list[Path]] = {}
    for product in ("mnt", "mns"):
        entries = metadata.get("sources", {}).get(product)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Terrain package has no {product.upper()} sources")
        paths: list[Path] = []
        for entry in entries:
            file_name = str(entry.get("file_name", ""))
            if not file_name or Path(file_name).name != file_name:
                raise ValueError(f"Terrain package {product.upper()} source is unsafe")
            source = source_root / product / file_name
            if not source.is_file():
                raise ValueError(f"Terrain package source is absent: {source}")
            paths.append(source)
        sources[product] = paths

    terrain = package["terrain"]
    ground_2d = metadata["ground_2d"]
    arguments = argparse.Namespace(
        mnt=sources["mnt"],
        mns=sources["mns"],
        bounds=metadata["bounds_l93_m"],
        processing_bounds=metadata["processing_bounds_l93_m"],
        terrain_step_pixels=int(terrain["step_pixels"]),
        ground_material_map_size=int(ground_2d["pixel_size"][0]),
        origin_x=float(metadata["origin_l93_m"][0]),
        origin_y=float(metadata["origin_l93_m"][1]),
        origin_z=float(metadata["origin_l93_m"][2]),
    )
    rebuilt, rebuilt_ground_map = build_terrain_outputs(arguments)
    if rebuilt != package:
        raise ValueError("Terrain package is not reproducible from its MNT/MNS sources")

    from PIL import Image

    ground_map_path = path.parent / str(ground_2d["file_name"])
    with Image.open(ground_map_path) as image:
        observed_ground_map = np.asarray(image.convert("RGBA"), dtype="uint8")
    if not np.array_equal(observed_ground_map, rebuilt_ground_map):
        raise ValueError("Terrain context map is not reproducible from MNT/MNS")

    import rasterio

    def source_crs_is_canonical(source: Path) -> bool:
        with rasterio.open(source) as dataset:
            return dataset.crs is not None and dataset.crs.to_epsg() == 2154

    source_crs_mode = "canonical_epsg2154"
    if any(
        not source_crs_is_canonical(source)
        for product_paths in sources.values()
        for source in product_paths
    ):
        source_crs_mode = "lambert93_parameters_from_epsg2154_request"

    return {
        "schema": "fireviewer.incident-terrain-evidence.v1",
        "tile_id": tile.get("id"),
        "crs": metadata["crs"],
        "source_crs_mode": source_crs_mode,
        "bounds_l93_m": metadata["bounds_l93_m"],
        "native_resolution_m": metadata["native_resolution_m"],
        "mnt_source_count": len(sources["mnt"]),
        "mns_source_count": len(sources["mns"]),
        "source_hashes_match": True,
        "mns_mnt_grid_match": True,
        "mesh_reproducible_from_mnt": True,
        "terrain_context_map_reproducible_from_mnt_mns": True,
        "terrain_vertex_count": terrain["vertex_count"],
        "terrain_face_count": terrain["face_count"],
        "contextual_surface_mapping_status": "not_authored",
        "routes_buildings_status": "not_authored",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnt", type=Path, action="append", required=True)
    parser.add_argument("--mns", type=Path, action="append", required=True)
    parser.add_argument("--bounds", type=float, nargs=4, required=True)
    parser.add_argument("--processing-bounds", type=float, nargs=4)
    parser.add_argument("--terrain-step-pixels", type=int, default=2)
    parser.add_argument("--ground-material-map-size", type=int, default=256)
    parser.add_argument("--origin-x", type=float)
    parser.add_argument("--origin-y", type=float)
    parser.add_argument("--origin-z", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    package, ground_material_map = build_terrain_outputs(args)
    write_terrain_package(package, args.output)
    write_ground_material_map(
        ground_material_map, args.output.with_name("ground-material-map.png")
    )
    print(json.dumps(package["statistics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
