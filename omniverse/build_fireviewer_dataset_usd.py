"""Build standalone, modular FireViewer OpenUSD dataset packages.

The builder consumes completed published site-map packages and the portable USD
conversion produced by ``convert_published_maps_to_usd.py``.  It never changes
the source map.  For every complete map it writes a self-contained OpenUSD
assembly with independently loadable terrain, buildings, vegetation point
instancing, non-rendered occlusion proxies, camera candidates, ten synthetic
fire-ground-truth states, four Omniverse appearance contracts, fifty-five
human camera prims, and seven additional aerial cameras.

Fire propagation is deliberately synthetic and explicitly labelled as such.
It is a continuous least-arrival-time model driven by uploaded terrain and
vegetation, with explicit weather drivers.  It is not an operational incident
reconstruction or a calibrated wildfire forecast.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import rasterio
import trimesh
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from fire_spread_model import FireFrontSegment, FireSpreadResult, FireSpreadState, simulate_fire_spread


EXCLUDED_PARTS = {".git", "node_modules", ".pytest_cache", "__pycache__", "temp"}
SEMANTIC_API = 'prepend apiSchemas = ["SemanticsLabelsAPI:class"]'
FLOW_EMITTER_COUNT = 48
FLOW_FRONT_PATCH_COUNT = 256
FLOW_SECONDS_PER_STATE = 6.0
FLOW_SMOKE_PLUME_LIFT_M = 1.6
FLOW_SMOKE_SOURCE = 0.32
CAPTURE_ZOOM_PROFILES = (
    ("wide", 0.75),
    ("reference", 1.0),
    ("medium", 1.25),
    ("tight", 1.5),
    ("tele", 2.0),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identifier(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character == "_" else "_" for character in value)
    return "_" + cleaned if not cleaned or cleaned[0].isdigit() else cleaned


def usd_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def f(value: float) -> str:
    return f"{float(value):.6f}"


def v3(values: Iterable[float]) -> str:
    return "(" + ", ".join(f(value) for value in values) + ")"


def v3_array(values: Iterable[Iterable[float]]) -> str:
    return "[" + ", ".join(v3(value) for value in values) + "]"


def int_array(values: Iterable[int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in values) + "]"


def usd_header(default_prim: str | None = None) -> str:
    default = f'    defaultPrim = "{default_prim}"\n' if default_prim else ""
    return "#usda 1.0\n(\n" + default + '    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'


def write_wrapped_values(stream: Any, values: Iterable[str], *, width: int = 4) -> None:
    iterator = iter(values)
    try:
        value = next(iterator)
    except StopIteration:
        return
    index = 0
    while True:
        try:
            following = next(iterator)
            has_following = True
        except StopIteration:
            following = None
            has_following = False
        if index % width == 0:
            stream.write("        ")
        stream.write(value)
        if has_following:
            stream.write(", ")
        if index % width == width - 1 or not has_following:
            stream.write("\n")
        if not has_following:
            break
        value = following
        index += 1


def semantic_api_metadata(*, indent: str = "    ") -> str:
    """Return only metadata valid inside a USD prim metadata block."""

    return f"{indent}{SEMANTIC_API}\n"


def semantic_properties(label: str, *, indent: str = "    ") -> str:
    """Return semantic properties valid inside the USD prim body."""

    return (
        f'{indent}token[] semantics:labels:class = ["{usd_string(label)}"]\n'
        f'{indent}custom string fireviewer:semantic_class = "{usd_string(label)}"\n'
    )


def package_complete(package: Path) -> bool:
    try:
        catalog = json.loads((package / "catalog.json").read_text(encoding="utf-8"))
        if catalog.get("schema_version") != "1.1":
            return False
        for tile in catalog.get("terrain_tiles", []):
            for key in ("elevation", "colour"):
                if not (package / str(tile[key]["path"])).is_file():
                    return False
        for feature in catalog.get("feature_tiles", []):
            if not (package / str(feature["features"]["path"])).is_file():
                return False
        return bool(catalog.get("terrain_tiles")) and bool(catalog.get("feature_tiles"))
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False


def source_priority(path: Path) -> tuple[int, str]:
    text = path.as_posix().lower()
    if "/fireviewer-frontend/public/maps/" in text:
        return 0, text
    if "/fireviewer-frontend/dist/maps/" in text:
        return 1, text
    if "/public/maps/" in text:
        return 2, text
    if "/dist/maps/" in text:
        return 3, text
    if "/artifacts/" in text:
        return 4, text
    return 5, text


def discover_packages(roots: list[Path]) -> list[Path]:
    grouped: dict[str, list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("package-manifest.json"):
            if any(part in EXCLUDED_PARTS for part in manifest_path.parts):
                continue
            package = manifest_path.parent
            if not (package / "catalog.json").is_file() or not package_complete(package):
                continue
            try:
                package_id = str(json.loads(manifest_path.read_text(encoding="utf-8"))["package_id"])
            except (KeyError, OSError, TypeError, json.JSONDecodeError):
                continue
            grouped.setdefault(package_id, []).append(package)
    return [sorted(paths, key=source_priority)[0] for _, paths in sorted(grouped.items())]


@dataclass(frozen=True)
class TreeInstance:
    tree_id: int
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    yaw_rad: float
    source_tile: str


@dataclass(frozen=True)
class TreeAsset:
    asset_id: str
    source_path: Path
    package_path: str
    source_name: str
    sha256: str
    byte_count: int
    up_axis: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    default_prim: str


@dataclass(frozen=True)
class OcclusionBox:
    name: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    visual_center_xy: tuple[float, float]
    visual_width_m: float
    visual_depth_m: float
    visual_yaw_rad: float
    semantic: str


@dataclass(frozen=True)
class AccessSurfaceSample:
    position_local: tuple[float, float, float]
    source_type: str
    source_tile: str
    surface_area_m2: float
    triangle_count: int


class HeightSampler:
    def __init__(self, package: Path, catalog: dict[str, Any]) -> None:
        self._tiles = list(catalog["terrain_tiles"])
        self._package = package
        self._datasets: dict[str, Any] = {}
        self._sample_cache: dict[tuple[float, float], float] = {}

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.close()
        self._datasets.clear()
        self._sample_cache.clear()

    def at(self, east: float, north: float) -> float:
        cache_key = (round(float(east), 3), round(float(north), 3))
        cached = self._sample_cache.get(cache_key)
        if cached is not None:
            return cached
        chosen = None
        for tile in self._tiles:
            west, south, right, top = (float(value) for value in tile["bounds_l93_metres"])
            if west <= east <= right and south <= north <= top:
                chosen = tile
                break
        if chosen is None:
            chosen = min(
                self._tiles,
                key=lambda tile: (east - (float(tile["bounds_l93_metres"][0]) + float(tile["bounds_l93_metres"][2])) / 2.0) ** 2
                + (north - (float(tile["bounds_l93_metres"][1]) + float(tile["bounds_l93_metres"][3])) / 2.0) ** 2,
            )
        tile_id = str(chosen["terrain_tile_id"])
        if tile_id not in self._datasets:
            self._datasets[tile_id] = rasterio.open(self._package / str(chosen["elevation"]["path"]))
        value = next(self._datasets[tile_id].sample([(east, north)]))[0]
        resolved = float(value) if math.isfinite(float(value)) else 0.0
        self._sample_cache[cache_key] = resolved
        return resolved


def local_points(points: Any, transform: Any, origin: tuple[float, float, float], anchor: tuple[float, float]) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float64)
    homogeneous = np.column_stack([values, np.ones(len(values))])
    transformed = homogeneous @ np.asarray(transform, dtype=np.float64).T
    return np.column_stack((
        transformed[:, 0] + origin[0] - anchor[0],
        -transformed[:, 2] + origin[1] - anchor[1],
        transformed[:, 1] + origin[2],
    ))


def local_vertices(geometry: Any, transform: Any, origin: tuple[float, float, float], anchor: tuple[float, float]) -> np.ndarray:
    return local_points(geometry.vertices, transform, origin, anchor)


def tree_from_component(
    component: Any,
    transform: Any,
    origin: tuple[float, float, float],
    anchor: tuple[float, float],
    tree_id: int,
    source_tile: str,
) -> TreeInstance | None:
    points = local_vertices(component, transform, origin, anchor)
    if len(points) < 4:
        return None
    minimum_z = float(np.min(points[:, 2]))
    base = points[np.isclose(points[:, 2], minimum_z, atol=0.005)]
    if len(base) < 3:
        return None
    position = np.mean(base, axis=0)
    radial = np.linalg.norm(base[:, :2] - position[:2], axis=1)
    radius = float(np.max(radial))
    height = float(np.max(points[:, 2]) - minimum_z)
    if radius <= 0.02 or height <= 0.02:
        return None
    direction = base[0, :2] - position[:2]
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    return TreeInstance(
        tree_id=tree_id,
        position=(float(position[0]), float(position[1]), minimum_z),
        scale=(radius, radius, height),
        yaw_rad=yaw,
        source_tile=source_tile,
    )


def merge_stacked_building_components(boxes: list[OcclusionBox]) -> list[OcclusionBox]:
    """Merge the adjacent lower/upper mesh components of one source building.

    The published GLB stores nearly every building as two consecutive disconnected
    components with the same oriented XY footprint. Treating both components as
    complete buildings creates two overlapping bodies and two roofs.
    """

    def source_group(box: OcclusionBox) -> str:
        return box.name.rsplit("_", 1)[0]

    def axis_difference(left: float, right: float) -> float:
        delta = abs((left - right) % math.pi)
        return min(delta, math.pi - delta)

    def relative_match(left: float, right: float) -> bool:
        return min(left, right) / max(left, right) >= 0.65

    def matching_footprint(left: OcclusionBox, right: OcclusionBox) -> bool:
        if source_group(left) != source_group(right):
            return False
        center_distance = math.dist(left.visual_center_xy, right.visual_center_xy)
        if center_distance > 0.5:
            return False
        left_height = left.maximum[2] - left.minimum[2]
        right_height = right.maximum[2] - right.minimum[2]
        if min(left_height, right_height) <= 0.05:
            direct = relative_match(left.visual_width_m, right.visual_width_m) and relative_match(
                left.visual_depth_m, right.visual_depth_m
            )
            swapped = relative_match(left.visual_width_m, right.visual_depth_m) and relative_match(
                left.visual_depth_m, right.visual_width_m
            )
            if direct or swapped:
                return True
        aligned = axis_difference(left.visual_yaw_rad, right.visual_yaw_rad) <= math.radians(5.0)
        if aligned:
            return relative_match(left.visual_width_m, right.visual_width_m) and relative_match(
                left.visual_depth_m, right.visual_depth_m
            )
        perpendicular = abs(axis_difference(left.visual_yaw_rad, right.visual_yaw_rad) - math.pi * 0.5) <= math.radians(5.0)
        return perpendicular and relative_match(left.visual_width_m, right.visual_depth_m) and relative_match(
            left.visual_depth_m, right.visual_width_m
        )

    merged: list[OcclusionBox] = []
    index = 0
    while index < len(boxes):
        lower = boxes[index]
        if index + 1 >= len(boxes) or not matching_footprint(lower, boxes[index + 1]):
            merged.append(lower)
            index += 1
            continue
        upper = boxes[index + 1]
        footprint = max((lower, upper), key=lambda box: box.visual_width_m * box.visual_depth_m)
        merged.append(
            OcclusionBox(
                name=f"{lower.name}_merged",
                minimum=tuple(min(lower.minimum[axis], upper.minimum[axis]) for axis in range(3)),
                maximum=tuple(max(lower.maximum[axis], upper.maximum[axis]) for axis in range(3)),
                visual_center_xy=footprint.visual_center_xy,
                visual_width_m=max(lower.visual_width_m, upper.visual_width_m),
                visual_depth_m=max(lower.visual_depth_m, upper.visual_depth_m),
                visual_yaw_rad=footprint.visual_yaw_rad,
                semantic=lower.semantic,
            )
        )
        index += 2
    return merged


def visual_building_modules(boxes: list[OcclusionBox]) -> list[OcclusionBox]:
    """Split implausibly long rural footprints into detached house-sized modules."""

    modules: list[OcclusionBox] = []
    gap_m = 1.2
    for box in boxes:
        width_parts = math.ceil(box.visual_width_m / 22.0) if box.visual_width_m > 32.0 else 1
        depth_parts = math.ceil(box.visual_depth_m / 16.0) if box.visual_depth_m > 24.0 else 1
        if width_parts == 1 and depth_parts == 1:
            modules.append(box)
            continue
        module_width = (box.visual_width_m - gap_m * (width_parts - 1)) / width_parts
        module_depth = (box.visual_depth_m - gap_m * (depth_parts - 1)) / depth_parts
        unit_u = (math.cos(box.visual_yaw_rad), math.sin(box.visual_yaw_rad))
        unit_v = (-math.sin(box.visual_yaw_rad), math.cos(box.visual_yaw_rad))
        source_height = max(5.5, box.maximum[2] - box.minimum[2])
        for width_index in range(width_parts):
            offset_u = -box.visual_width_m * 0.5 + module_width * 0.5 + width_index * (module_width + gap_m)
            for depth_index in range(depth_parts):
                offset_v = -box.visual_depth_m * 0.5 + module_depth * 0.5 + depth_index * (module_depth + gap_m)
                module_name = f"{box.name}_module_{width_index + 1:02d}_{depth_index + 1:02d}"
                variation = int(hashlib.sha256(module_name.encode("utf-8")).hexdigest()[:4], 16) % 7
                module_height = min(16.0, max(5.5, source_height * (0.82 + variation * 0.035)))
                center_x = box.visual_center_xy[0] + unit_u[0] * offset_u + unit_v[0] * offset_v
                center_y = box.visual_center_xy[1] + unit_u[1] * offset_u + unit_v[1] * offset_v
                modules.append(
                    OcclusionBox(
                        name=module_name,
                        minimum=(box.minimum[0], box.minimum[1], box.minimum[2]),
                        maximum=(box.maximum[0], box.maximum[1], box.minimum[2] + module_height),
                        visual_center_xy=(center_x, center_y),
                        visual_width_m=module_width,
                        visual_depth_m=module_depth,
                        visual_yaw_rad=box.visual_yaw_rad,
                        semantic=box.semantic,
                    )
                )
    return modules


def collect_derived_geometry(
    package: Path,
    catalog: dict[str, Any],
    anchor: tuple[float, float],
    *,
    include_source_trees: bool = True,
) -> tuple[list[TreeInstance], list[OcclusionBox], list[AccessSurfaceSample]]:
    trees: list[TreeInstance] = []
    boxes: list[OcclusionBox] = []
    access_samples: list[AccessSurfaceSample] = []
    expected_tree_count = 0
    next_tree_id = 1
    for feature in catalog["feature_tiles"]:
        expected_tree_count += int(feature["features"].get("tree_count", 0))
        origin = tuple(float(value) for value in feature["gltf_local_origin_l93_ngf_ign69"])
        scene = trimesh.load(package / str(feature["features"]["path"]), force="scene", process=False)
        for name, geometry in scene.geometry.items():
            transform, _ = scene.graph.get(name)
            lowered = str(name).lower()
            if lowered == "trees_low_poly" and include_source_trees:
                for component in geometry.split(only_watertight=False):
                    tree = tree_from_component(component, transform, origin, anchor, next_tree_id, str(feature["tile_id"]))
                    if tree is not None:
                        trees.append(tree)
                        next_tree_id += 1
            elif lowered == "buildings":
                for component_index, component in enumerate(geometry.split(only_watertight=False), start=1):
                    points = local_vertices(component, transform, origin, anchor)
                    if not len(points):
                        continue
                    minimum = tuple(float(value) for value in np.min(points, axis=0))
                    maximum = tuple(float(value) for value in np.max(points, axis=0))
                    if maximum[0] - minimum[0] <= 1.0 or maximum[1] - minimum[1] <= 1.0:
                        continue
                    xy = points[:, :2]
                    mean_xy = np.mean(xy, axis=0)
                    covariance = np.cov(xy - mean_xy, rowvar=False)
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
                    perpendicular = np.asarray((-principal[1], principal[0]), dtype=np.float64)
                    projected_u = (xy - mean_xy) @ principal
                    projected_v = (xy - mean_xy) @ perpendicular
                    minimum_u, maximum_u = float(np.min(projected_u)), float(np.max(projected_u))
                    minimum_v, maximum_v = float(np.min(projected_v)), float(np.max(projected_v))
                    oriented_center = mean_xy + principal * ((minimum_u + maximum_u) * 0.5) + perpendicular * ((minimum_v + maximum_v) * 0.5)
                    boxes.append(OcclusionBox(
                        name=f"Building_{identifier(str(feature['tile_id']))}_{component_index:05d}",
                        minimum=minimum,
                        maximum=maximum,
                        visual_center_xy=(float(oriented_center[0]), float(oriented_center[1])),
                        visual_width_m=max(1.0, maximum_u - minimum_u),
                        visual_depth_m=max(1.0, maximum_v - minimum_v),
                        visual_yaw_rad=math.atan2(float(principal[1]), float(principal[0])),
                        semantic="building_occlusion_proxy",
                    ))
            elif lowered in {"road", "path", "itinerary"}:
                points = local_points(geometry.triangles_center, transform, origin, anchor)
                for point in points:
                    if not bool(np.isfinite(point).all()):
                        continue
                    access_samples.append(
                        AccessSurfaceSample(
                            position_local=(float(point[0]), float(point[1]), float(point[2])),
                            source_type=lowered,
                            source_tile=str(feature["tile_id"]),
                            surface_area_m2=float(geometry.area),
                            triangle_count=int(len(geometry.faces)),
                        )
                    )
    if include_source_trees and len(trees) != expected_tree_count:
        raise ValueError(f"Tree component count {len(trees)} does not match catalog count {expected_tree_count}")
    return trees, merge_stacked_building_components(boxes), access_samples


def load_mnt_mns_trees(
    index_path: Path,
    *,
    package_id: str,
    anchor: tuple[float, float],
) -> tuple[list[TreeInstance], dict[str, Any]]:
    index_path = index_path.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != "fireviewer.mnt-mns-vegetation-rebuild.v1":
        raise ValueError(f"Unsupported MNT/MNS vegetation index: {index_path}")
    if index.get("base_package_id") != package_id:
        raise ValueError(
            f"Vegetation index base package mismatch: {index.get('base_package_id')} != {package_id}"
        )
    if index.get("method", {}).get("segmentation") != (
        "all_accepted_0m50_crown_apices_without_post_detection_thinning"
    ):
        raise ValueError("Vegetation index does not preserve every accepted 0.5 m crown apex")
    if index.get("source_manifest", {}).get("all_source_hashes_verified") is not True:
        raise ValueError("Vegetation index does not prove MNT/MNS source hashes")
    trees: list[TreeInstance] = []
    next_tree_id = 1
    index_tiles = list(index.get("tiles", []))
    for tile_number, tile in enumerate(index_tiles, start=1):
        relative = str(tile["path"])
        candidate = (index_path.parent / relative).resolve()
        if index_path.parent not in candidate.parents:
            raise ValueError(f"Unsafe vegetation tile path: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing vegetation tile: {candidate}")
        if candidate.stat().st_size != int(tile["byte_count"]):
            raise ValueError(f"Vegetation tile byte count mismatch: {candidate}")
        if sha256_file(candidate) != str(tile["sha256"]):
            raise ValueError(f"Vegetation tile checksum mismatch: {candidate}")
        with np.load(candidate, allow_pickle=False) as values:
            positions = np.asarray(values["positions_l93_m"], dtype=np.float64)
            heights = np.asarray(values["heights_m"], dtype=np.float64)
            crowns = np.asarray(values["crown_diameters_m"], dtype=np.float64)
            variants = np.asarray(values["visual_variants"], dtype=np.int64)
            rotations = np.asarray(values["rotations_degrees"], dtype=np.float64)
        count = len(positions)
        if positions.shape != (count, 3) or any(
            len(array) != count for array in (heights, crowns, variants, rotations)
        ):
            raise ValueError(f"Vegetation tile arrays are inconsistent: {candidate}")
        if count != int(tile["accepted_crown_count"]):
            raise ValueError(f"Vegetation tile count mismatch: {candidate}")
        bounds = [float(value) for value in tile["bounds_l93_m"]]
        if count:
            finite = np.isfinite(positions).all(axis=1) & np.isfinite(heights) & np.isfinite(crowns) & np.isfinite(rotations)
            owned = (
                (positions[:, 0] >= bounds[0])
                & (positions[:, 0] < bounds[2])
                & (positions[:, 1] >= bounds[1])
                & (positions[:, 1] < bounds[3])
            )
            valid = finite & owned & (heights >= 2.0) & (crowns > 0.0) & (variants >= 0) & (variants < 6)
            if not bool(np.all(valid)):
                raise ValueError(f"Vegetation tile contains invalid or unowned instances: {candidate}")
        source_tile = str(tile["tile_id"])
        for position, height, crown, rotation in zip(positions, heights, crowns, rotations):
            trees.append(
                TreeInstance(
                    tree_id=next_tree_id,
                    position=(
                        float(position[0]) - anchor[0],
                        float(position[1]) - anchor[1],
                        float(position[2]),
                    ),
                    scale=(float(crown) / 2.0, float(crown) / 2.0, float(height)),
                    yaw_rad=math.radians(float(rotation) % 360.0),
                    source_tile=source_tile,
                )
            )
            next_tree_id += 1
        if tile_number % 16 == 0 or tile_number == len(index_tiles):
            print(
                json.dumps(
                    {
                        "phase": "load_mnt_mns_vegetation",
                        "complete_tiles": tile_number,
                        "total_tiles": len(index_tiles),
                        "loaded_instances": len(trees),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = int(index.get("counts", {}).get("tree_instances", -1))
    if len(trees) != expected:
        raise ValueError(f"Vegetation rebuild count mismatch: {len(trees)} != {expected}")
    return trees, index


def write_reference_payload(
    output: Path,
    *,
    root_name: str,
    semantic: str,
    references: list[tuple[str, str, str]],
    custom_properties: list[str] | None = None,
) -> None:
    """Write a payload that points at exact converted geometry by prim path."""

    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header(root_name))
        stream.write(f'def Xform "{root_name}" (\n    kind = "component"\n')
        stream.write(semantic_api_metadata(indent="    "))
        stream.write(")\n{\n")
        stream.write(semantic_properties(semantic))
        for property_line in custom_properties or []:
            stream.write(f"    {property_line}\n")
        for child, asset, target in references:
            stream.write(f'    def Mesh "{child}" (\n        prepend references = @{asset}@<{target}>\n')
            stream.write(semantic_api_metadata(indent="        "))
            stream.write("    )\n    {\n")
            stream.write(semantic_properties(semantic, indent="        "))
            stream.write("    }\n")
        stream.write("}\n")


def write_terrain_payload(output: Path, base_manifest: dict[str, Any]) -> None:
    references = []
    for record in base_manifest["terrain"]:
        tile_id = str(record["terrain_tile_id"])
        child = "Tile_" + identifier(tile_id)
        asset = "../../source-usd/terrain/" + identifier(tile_id) + ".usda"
        target = "/Terrain_" + identifier(tile_id)
        references.append((child, asset, target))
    write_reference_payload(output, root_name="TerrainPayload", semantic="terrain", references=references)


def geometry_references(base_manifest: dict[str, Any], names: set[str], *, root: str) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    for tile in base_manifest["features"]:
        tile_id = str(tile["tile_id"])
        for ordinal, geometry in enumerate(tile["geometry"], start=1):
            name = str(geometry["name"])
            if name not in names:
                continue
            asset = "../../source-usd/features/" + identifier(tile_id) + ".usda"
            target = "/Features_" + identifier(tile_id) + "/" + identifier(f"{ordinal}_{name}")
            child = identifier(f"{root}_{tile_id}_{ordinal}_{name}")
            records.append((child, asset, target))
    return records


def load_tree_assets_manifest(path: Path) -> list[TreeAsset]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "fireviewer.usdz-tree-asset-inspection.v1":
        raise ValueError(f"Unsupported tree asset manifest: {path}")
    records = manifest.get("assets")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("Exactly six inspected USDZ tree assets are required")
    assets: list[TreeAsset] = []
    seen_hashes: set[str] = set()
    for index, record in enumerate(records, start=1):
        source = Path(str(record["source_path"])).resolve()
        if source.suffix.lower() != ".usdz" or not source.is_file():
            raise FileNotFoundError(f"Tree asset is not a readable USDZ file: {source}")
        actual_sha256 = sha256_file(source)
        actual_size = source.stat().st_size
        if actual_sha256 != str(record["sha256"]) or actual_size != int(record["byte_count"]):
            raise ValueError(f"Tree asset changed after inspection: {source}")
        if actual_sha256 in seen_hashes:
            raise ValueError(f"Duplicate tree asset content is not allowed: {source}")
        seen_hashes.add(actual_sha256)
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if not members or not any(Path(member.filename).suffix.lower() in {".usd", ".usda", ".usdc"} for member in members):
                raise ValueError(f"USDZ has no USD root layer: {source}")
            for member in members:
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe USDZ member path in {source}: {member.filename}")
        minimum = tuple(float(value) for value in record["range_min"])
        maximum = tuple(float(value) for value in record["range_max"])
        if len(minimum) != 3 or len(maximum) != 3 or not all(math.isfinite(value) for value in (*minimum, *maximum)):
            raise ValueError(f"Invalid inspected bounds for tree asset: {source}")
        up_axis = str(record["up_axis"]).upper()
        if up_axis == "Y":
            horizontal_radius = max((maximum[0] - minimum[0]) / 2.0, (maximum[2] - minimum[2]) / 2.0)
            height = maximum[1] - minimum[1]
        elif up_axis == "Z":
            horizontal_radius = max((maximum[0] - minimum[0]) / 2.0, (maximum[1] - minimum[1]) / 2.0)
            height = maximum[2] - minimum[2]
        else:
            raise ValueError(f"Unsupported USDZ up axis {up_axis!r}: {source}")
        if horizontal_radius <= 0.01 or height <= 0.01:
            raise ValueError(f"Degenerate USDZ tree bounds: {source}")
        assets.append(TreeAsset(
            asset_id=f"tree_{index:02d}",
            source_path=source,
            package_path=f"assets/trees/tree_{index:02d}.usdz",
            source_name=source.name,
            sha256=actual_sha256,
            byte_count=actual_size,
            up_axis=up_axis,
            minimum=minimum,
            maximum=maximum,
            default_prim=str(record["default_prim"]),
        ))
    return assets


def copy_tree_assets(output: Path, assets: list[TreeAsset]) -> None:
    for asset in assets:
        target = output / asset.package_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset.source_path, target)
        if sha256_file(target) != asset.sha256:
            raise ValueError(f"Copied tree asset checksum mismatch: {target}")


def tree_asset_dimensions(asset: TreeAsset) -> tuple[tuple[float, float, float], float, float]:
    minimum, maximum = asset.minimum, asset.maximum
    if asset.up_axis == "Y":
        center = ((minimum[0] + maximum[0]) / 2.0, minimum[1], (minimum[2] + maximum[2]) / 2.0)
        radius = max((maximum[0] - minimum[0]) / 2.0, (maximum[2] - minimum[2]) / 2.0)
        height = maximum[1] - minimum[1]
    else:
        center = ((minimum[0] + maximum[0]) / 2.0, (minimum[1] + maximum[1]) / 2.0, minimum[2])
        radius = max((maximum[0] - minimum[0]) / 2.0, (maximum[1] - minimum[1]) / 2.0)
        height = maximum[2] - minimum[2]
    return center, radius, height


def write_tree_asset_prototype(stream: Any, asset: TreeAsset) -> None:
    center, radius, height = tree_asset_dimensions(asset)
    prim_name = identifier(asset.asset_id)
    relative_asset = "../../" + asset.package_path
    stream.write(f'        def Xform "{prim_name}"\n        {{\n')
    stream.write(f'            custom string fireviewer:asset_id = "{asset.asset_id}"\n')
    stream.write(f'            custom string fireviewer:asset_sha256 = "{asset.sha256}"\n')
    stream.write(f"            double3 xformOp:scale = ({f(1.0 / radius)}, {f(1.0 / radius)}, {f(1.0 / height)})\n")
    stream.write('            uniform token[] xformOpOrder = ["xformOp:scale"]\n')
    stream.write('            def Xform "Axis"\n            {\n')
    if asset.up_axis == "Y":
        stream.write("                double xformOp:rotateX = 90\n")
        stream.write('                uniform token[] xformOpOrder = ["xformOp:rotateX"]\n')
    stream.write('                def Xform "Offset"\n                {\n')
    stream.write(f"                    double3 xformOp:translate = ({f(-center[0])}, {f(-center[1])}, {f(-center[2])})\n")
    stream.write('                    uniform token[] xformOpOrder = ["xformOp:translate"]\n')
    stream.write(f'                    def Xform "Asset" (prepend references = @{relative_asset}@)\n                    {{\n')
    stream.write(f'                        custom string fireviewer:source_default_prim = "{usd_string(asset.default_prim)}"\n')
    stream.write("                    }\n                }\n            }\n        }\n")


def write_tree_mesh(stream: Any, *, path: str, colour: tuple[float, float, float]) -> None:
    stream.write(f'        def Mesh "{path}"\n        {{\n')
    stream.write("            uniform token subdivisionScheme = \"none\"\n")
    stream.write("            int[] faceVertexCounts = [3, 3, 3, 3, 3]\n")
    stream.write("            int[] faceVertexIndices = [0, 1, 5, 1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 0, 5]\n")
    points = [(math.cos(index * math.tau / 5), math.sin(index * math.tau / 5), 0.0) for index in range(5)] + [(0.0, 0.0, 1.0)]
    stream.write("            point3f[] points = [\n")
    write_wrapped_values(stream, [v3(point) for point in points], width=2)
    stream.write("            ]\n            color3f[] primvars:displayColor = [\n")
    write_wrapped_values(stream, [v3(colour)] * 6, width=3)
    stream.write('            ] ( interpolation = "vertex" )\n')
    stream.write("        }\n")


def write_instancer(stream: Any, *, path: str, prototype_paths: list[str], trees: list[TreeInstance], semantic: str) -> None:
    if not prototype_paths:
        raise ValueError("PointInstancer requires at least one prototype")
    stream.write(f'    def PointInstancer "{path}" (\n')
    stream.write(semantic_api_metadata(indent="        "))
    stream.write("    )\n    {\n")
    stream.write(semantic_properties(semantic, indent="        "))
    stream.write("        rel prototypes = [\n")
    write_wrapped_values(stream, [f"<{prototype_path}>" for prototype_path in prototype_paths], width=2)
    stream.write("        ]\n")
    stream.write("        int64[] ids = [\n")
    write_wrapped_values(stream, (str(tree.tree_id) for tree in trees), width=12)
    stream.write("        ]\n        point3f[] positions = [\n")
    write_wrapped_values(stream, (v3(tree.position) for tree in trees), width=2)
    stream.write("        ]\n        float3[] scales = [\n")
    write_wrapped_values(stream, (v3(tree.scale) for tree in trees), width=2)
    stream.write("        ]\n        quath[] orientations = [\n")
    orientations = (f"({f(math.cos(tree.yaw_rad / 2.0))}, 0, 0, {f(math.sin(tree.yaw_rad / 2.0))})" for tree in trees)
    write_wrapped_values(stream, orientations, width=2)
    stream.write("        ]\n        int[] protoIndices = [\n")
    write_wrapped_values(stream, (str((tree.tree_id - 1) % len(prototype_paths)) for tree in trees), width=20)
    stream.write("        ]\n")
    stream.write("    }\n")


def write_vegetation_payload(output: Path, trees: list[TreeInstance], tree_assets: list[TreeAsset]) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header("VegetationPayload"))
        stream.write('def Xform "VegetationPayload" (\n    kind = "component"\n')
        stream.write(semantic_api_metadata(indent="    "))
        stream.write(")\n{\n")
        stream.write(semantic_properties("vegetation"))
        stream.write('    def Scope "Prototypes"\n    {\n')
        for asset in tree_assets:
            write_tree_asset_prototype(stream, asset)
        stream.write("    }\n")
        write_instancer(
            stream,
            path="Trees",
            prototype_paths=[f"/VegetationPayload/Prototypes/{identifier(asset.asset_id)}" for asset in tree_assets],
            trees=trees,
            semantic="vegetation",
        )
        stream.write(f"    custom int fireviewer:source_tree_count = {len(trees)}\n")
        stream.write(f"    custom int fireviewer:prototype_asset_count = {len(tree_assets)}\n")
        stream.write('    custom string fireviewer:representation = "mnt_mns_tree_components_to_usdz_point_instancer"\n')
        stream.write('    custom string fireviewer:ground_source = "MNT"\n')
        stream.write('    custom string fireviewer:canopy_height_source = "MNS_minus_MNT"\n')
        stream.write("    custom bool fireviewer:post_detection_thinning = false\n")
        stream.write("}\n")


def simple_building_dimensions(box: OcclusionBox, *, ground_z: float | None = None) -> dict[str, tuple[float, float, float] | float]:
    minimum, maximum = box.minimum, box.maximum
    width = box.visual_width_m
    depth = box.visual_depth_m
    source_height = max(0.0, maximum[2] - minimum[2])
    total_height = min(16.0, max(5.5, source_height))
    roof_height = min(4.0, max(1.5, total_height * 0.24))
    body_height = max(4.0, total_height - roof_height)
    center_x, center_y = box.visual_center_xy
    minimum_z = float(minimum[2]) if ground_z is None else float(ground_z)
    return {
        "body_position": (center_x, center_y, minimum_z + body_height / 2.0),
        "body_scale": (width, depth, body_height),
        "roof_position": (center_x, center_y, minimum_z + body_height),
        "roof_scale": (width + 0.55, depth + 0.55, roof_height),
        "occlusion_position": (
            (minimum[0] + maximum[0]) / 2.0,
            (minimum[1] + maximum[1]) / 2.0,
            minimum_z + (body_height + roof_height) / 2.0,
        ),
        "occlusion_scale": (
            maximum[0] - minimum[0],
            maximum[1] - minimum[1],
            body_height + roof_height,
        ),
        "total_height": body_height + roof_height,
        "yaw_rad": box.visual_yaw_rad,
        "minimum_z": minimum_z,
    }


def write_unit_box_mesh(stream: Any, *, name: str, colour: tuple[float, float, float]) -> None:
    points = [
        (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0),
        (-0.5, -0.5, 1.0), (0.5, -0.5, 1.0), (0.5, 0.5, 1.0), (-0.5, 0.5, 1.0),
    ]
    stream.write(f'        def Mesh "{name}"\n        {{\n')
    stream.write('            uniform token subdivisionScheme = "none"\n')
    stream.write("            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n")
    stream.write("            int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]\n")
    stream.write("            point3f[] points = [\n")
    write_wrapped_values(stream, [v3(point) for point in points], width=2)
    stream.write("            ]\n")
    stream.write(f"            color3f[] primvars:displayColor = [{v3(colour)}] ( interpolation = \"constant\" )\n")
    stream.write("        }\n")


def write_unit_roof_mesh(stream: Any, *, name: str, colour: tuple[float, float, float]) -> None:
    """Write a gabled roof whose ridge follows the local X axis."""

    points = [
        (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0),
        (-0.5, 0.0, 1.0), (0.5, 0.0, 1.0),
    ]
    stream.write(f'        def Mesh "{name}"\n        {{\n')
    stream.write('            uniform token subdivisionScheme = "none"\n')
    stream.write("            int[] faceVertexCounts = [4, 4, 4, 3, 3]\n")
    stream.write("            int[] faceVertexIndices = [0, 3, 2, 1, 0, 1, 5, 4, 3, 4, 5, 2, 0, 4, 3, 1, 2, 5]\n")
    stream.write("            point3f[] points = [\n")
    write_wrapped_values(stream, [v3(point) for point in points], width=2)
    stream.write("            ]\n")
    stream.write(f"            color3f[] primvars:displayColor = [{v3(colour)}] ( interpolation = \"constant\" )\n")
    stream.write("        }\n")


def write_simple_buildings_payload(
    output: Path,
    boxes: list[OcclusionBox],
    *,
    ground_at_local: Any | None = None,
) -> None:
    ids: list[str] = []
    positions: list[str] = []
    scales: list[str] = []
    orientations: list[str] = []
    prototype_indices: list[str] = []
    wall_colours = ((0.72, 0.66, 0.56), (0.61, 0.58, 0.53), (0.79, 0.75, 0.66), (0.56, 0.54, 0.50))
    roof_colours = ((0.38, 0.16, 0.09), (0.46, 0.22, 0.12), (0.30, 0.28, 0.25), (0.52, 0.30, 0.17))

    def append_instance(position: tuple[float, float, float], scale: tuple[float, float, float], yaw_rad: float, prototype_index: int) -> None:
        ids.append(str(len(ids) + 1))
        positions.append(v3(position))
        scales.append(v3(scale))
        orientations.append(f"({f(math.cos(yaw_rad / 2.0))}, 0, 0, {f(math.sin(yaw_rad / 2.0))})")
        prototype_indices.append(str(prototype_index))

    visual_boxes = visual_building_modules(boxes)
    for box in visual_boxes:
        ground_z = ground_at_local(*box.visual_center_xy) if ground_at_local is not None else None
        dimensions = simple_building_dimensions(box, ground_z=ground_z)
        style = int(hashlib.sha256(box.name.encode("utf-8")).hexdigest()[:8], 16) % len(wall_colours)
        yaw = float(dimensions["yaw_rad"])
        body_position = dimensions["body_position"]
        body_scale = dimensions["body_scale"]
        width, depth, body_height = body_scale
        center_x, center_y, _ = body_position
        minimum_z = float(dimensions["minimum_z"])
        unit_u = (math.cos(yaw), math.sin(yaw))
        unit_v = (-math.sin(yaw), math.cos(yaw))
        append_instance(body_position, body_scale, yaw, style)
        append_instance(dimensions["roof_position"], dimensions["roof_scale"], yaw, 4 + style)

        window_z = minimum_z + min(body_height - 0.7, max(2.2, body_height * 0.58))
        front_back_scale = (max(1.0, width * 0.72), 0.14, 0.72)
        side_scale = (max(1.0, depth * 0.72), 0.14, 0.72)
        for sign in (-1.0, 1.0):
            append_instance(
                (center_x + unit_v[0] * sign * (depth * 0.5 + 0.08), center_y + unit_v[1] * sign * (depth * 0.5 + 0.08), window_z),
                front_back_scale,
                yaw,
                8,
            )
            append_instance(
                (center_x + unit_u[0] * sign * (width * 0.5 + 0.08), center_y + unit_u[1] * sign * (width * 0.5 + 0.08), window_z),
                side_scale,
                yaw + math.pi * 0.5,
                8,
            )
        door_height = min(2.4, max(1.9, body_height * 0.48))
        append_instance(
            (center_x - unit_v[0] * (depth * 0.5 + 0.10), center_y - unit_v[1] * (depth * 0.5 + 0.10), minimum_z + door_height * 0.5),
            (min(1.5, max(0.9, width * 0.14)), 0.18, door_height),
            yaw,
            9,
        )
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header("BuildingsPayload"))
        stream.write('def Xform "BuildingsPayload" (kind = "component")\n{\n')
        stream.write(semantic_properties("building"))
        stream.write('    custom string fireviewer:representation = "merged_source_footprint_oriented_box_gabled_roof_windows_and_doors"\n')
        stream.write('    custom string fireviewer:component_merge = "adjacent_lower_upper_source_components_with_matching_oriented_xy_footprint"\n')
        stream.write('    custom string fireviewer:oversized_footprint_policy = "split_above_32x24m_into_detached_house_modules_with_1m20_gaps"\n')
        stream.write('    custom string fireviewer:orientation_method = "principal_axis_from_source_building_component"\n')
        stream.write('    custom string fireviewer:vertical_alignment = "module_center_sampled_from_mnt_ground"\n')
        stream.write('    custom string fireviewer:architectural_detail = "four_wall_styles_four_roof_styles_gabled_eaves_window_bands_and_entrance"\n')
        stream.write(f"    custom int fireviewer:source_building_count = {len(boxes)}\n")
        stream.write(f"    custom int fireviewer:visual_building_module_count = {len(visual_boxes)}\n")
        stream.write(f"    custom int fireviewer:render_instance_count = {len(ids)}\n")
        stream.write('    def Scope "Prototypes"\n    {\n')
        for style, colour in enumerate(wall_colours):
            write_unit_box_mesh(stream, name=f"BodyStyle{style:02d}", colour=colour)
        for style, colour in enumerate(roof_colours):
            write_unit_roof_mesh(stream, name=f"RoofStyle{style:02d}", colour=colour)
        write_unit_box_mesh(stream, name="WindowBand", colour=(0.10, 0.18, 0.22))
        write_unit_box_mesh(stream, name="Entrance", colour=(0.16, 0.09, 0.055))
        stream.write("    }\n")
        stream.write('    def PointInstancer "SimpleBuildings"\n    {\n')
        prototype_paths = [f"</BuildingsPayload/Prototypes/BodyStyle{style:02d}>" for style in range(4)]
        prototype_paths += [f"</BuildingsPayload/Prototypes/RoofStyle{style:02d}>" for style in range(4)]
        prototype_paths += ["</BuildingsPayload/Prototypes/WindowBand>", "</BuildingsPayload/Prototypes/Entrance>"]
        stream.write("        rel prototypes = [\n")
        write_wrapped_values(stream, prototype_paths, width=2)
        stream.write("        ]\n")
        stream.write("        int64[] ids = [\n")
        write_wrapped_values(stream, ids, width=12)
        stream.write("        ]\n        point3f[] positions = [\n")
        write_wrapped_values(stream, positions, width=2)
        stream.write("        ]\n        float3[] scales = [\n")
        write_wrapped_values(stream, scales, width=2)
        stream.write("        ]\n        quath[] orientations = [\n")
        write_wrapped_values(stream, orientations, width=2)
        stream.write("        ]\n        int[] protoIndices = [\n")
        write_wrapped_values(stream, prototype_indices, width=20)
        stream.write("        ]\n")
        stream.write(semantic_properties("building", indent="        "))
        stream.write("    }\n}\n")


def write_occlusion_payload(output: Path, boxes: list[OcclusionBox]) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header("OcclusionPayload"))
        stream.write('def Xform "OcclusionPayload" (kind = "component")\n{\n')
        stream.write('    custom string fireviewer:purpose = "non_rendered_occlusion_and_camera_qa"\n')
        stream.write('    def Scope "Prototypes"\n    {\n')
        write_unit_box_mesh(stream, name="UnitBox", colour=(0.0, 0.0, 0.0))
        stream.write("    }\n")
        stream.write('    def PointInstancer "BuildingOcclusion"\n    {\n')
        stream.write('        token purpose = "guide"\n        token visibility = "invisible"\n')
        stream.write('        rel prototypes = </OcclusionPayload/Prototypes/UnitBox>\n')
        stream.write("        int64[] ids = [\n")
        write_wrapped_values(stream, [str(index) for index in range(1, len(boxes) + 1)], width=12)
        stream.write("        ]\n        point3f[] positions = [\n")
        write_wrapped_values(stream, [v3(simple_building_dimensions(box)["occlusion_position"]) for box in boxes], width=2)
        stream.write("        ]\n        float3[] scales = [\n")
        write_wrapped_values(stream, [v3(simple_building_dimensions(box)["occlusion_scale"]) for box in boxes], width=2)
        stream.write("        ]\n        int[] protoIndices = [\n")
        write_wrapped_values(stream, ["0"] * len(boxes), width=20)
        stream.write("        ]\n")
        stream.write(semantic_properties("building_occlusion_proxy", indent="        "))
        stream.write("        custom bool fireviewer:occlusion_proxy = true\n")
        stream.write("    }\n}\n")


def camera_candidates(center: tuple[float, float], bounds: tuple[float, float, float, float], sampler: HeightSampler) -> list[tuple[float, float, float]]:
    xmin, ymin, xmax, ymax = bounds
    span = min(xmax - xmin, ymax - ymin)
    candidates: list[tuple[float, float, float]] = []
    for ring, ratio in enumerate((0.14, 0.25, 0.37), start=1):
        for index in range(12):
            angle = math.tau * index / 12.0 + ring * 0.11
            east = min(xmax - 60.0, max(xmin + 60.0, center[0] + math.cos(angle) * span * ratio))
            north = min(ymax - 60.0, max(ymin + 60.0, center[1] + math.sin(angle) * span * ratio))
            candidates.append((east, north, sampler.at(east, north) + 5.0 + ring * 16.0))
    return candidates


def write_camera_candidate_payload(output: Path, candidates: list[tuple[float, float, float]]) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header("CameraCandidatesPayload"))
        stream.write('def Xform "CameraCandidatesPayload" (kind = "component")\n{\n')
        stream.write('    custom string fireviewer:role = "selected_human_and_aerial_camera_positions"\n')
        stream.write('    def Points "Candidates"\n    {\n        token purpose = "guide"\n        token visibility = "invisible"\n        point3f[] points = [\n')
        write_wrapped_values(stream, [v3(point) for point in candidates], width=2)
        stream.write("        ]\n        float[] widths = [\n")
        write_wrapped_values(stream, ["2.0"] * len(candidates), width=16)
        stream.write("        ]\n    }\n}\n")


def write_site_layer(output: Path, *, package_id: str, anchor: tuple[float, float], bounds: tuple[float, float, float, float]) -> None:
    components = (
        ("Terrain", "payloads/terrain.payload.usda", "TerrainPayload", "terrain"),
        ("Buildings", "payloads/buildings.payload.usda", "BuildingsPayload", "building"),
        ("Routes", "payloads/routes.payload.usda", "RoutesPayload", "route"),
        ("VegetationContext", "payloads/vegetation_context.payload.usda", "VegetationContextPayload", "vegetation"),
        ("Vegetation", "payloads/vegetation.payload.usda", "VegetationPayload", "vegetation"),
        ("OcclusionProxies", "payloads/occlusion.payload.usda", "OcclusionPayload", "building_occlusion_proxy"),
        ("CameraCandidates", "payloads/camera_candidates.payload.usda", "CameraCandidatesPayload", "camera_candidate"),
    )
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header())
        stream.write('over "World"\n{\n    def Xform "FireViewerSite" (kind = "assembly")\n    {\n')
        stream.write(f'        custom string fireviewer:package_id = "{usd_string(package_id)}"\n')
        stream.write('        custom string fireviewer:source_type = "published_site_map"\n')
        stream.write('        custom string fireviewer:crs = "EPSG:2154"\n')
        stream.write('        custom string fireviewer:vertical_datum = "NGF-IGN69"\n')
        stream.write(f"        custom double2 fireviewer:common_anchor_l93_m = ({f(anchor[0])}, {f(anchor[1])})\n")
        stream.write(f"        custom double4 fireviewer:bounds_l93_m = ({', '.join(f(value) for value in bounds)})\n")
        for name, asset, target, semantic in components:
            stream.write(f'        def Xform "{name}" (\n            kind = "component"\n            prepend payload = @{asset}@</{target}>\n')
            stream.write(semantic_api_metadata(indent="            "))
            stream.write("        )\n        {\n")
            stream.write(semantic_properties(semantic, indent="            "))
            if name in {"SourceVegetationMesh", "OcclusionProxies", "CameraCandidates"}:
                stream.write('            token visibility = "invisible"\n')
            stream.write("        }\n")
        stream.write("    }\n}\n")


def write_quad_mesh(
    stream: Any,
    *,
    name: str,
    quads: list[list[tuple[float, float, float]]],
    semantic: str,
    colour: tuple[float, float, float],
    quad_colours: list[tuple[float, float, float]] | None = None,
) -> None:
    if not quads:
        return
    points = [point for quad in quads for point in quad]
    stream.write(f'    def Mesh "{name}" (\n')
    stream.write(semantic_api_metadata(indent="        "))
    stream.write("    )\n    {\n")
    stream.write(semantic_properties(semantic, indent="        "))
    stream.write("        uniform token subdivisionScheme = \"none\"\n        int[] faceVertexCounts = [\n")
    write_wrapped_values(stream, ["4"] * len(quads), width=18)
    stream.write("        ]\n        int[] faceVertexIndices = [\n")
    write_wrapped_values(stream, [str(index) for index in range(len(points))], width=12)
    stream.write("        ]\n        point3f[] points = [\n")
    write_wrapped_values(stream, [v3(point) for point in points], width=2)
    stream.write("        ]\n        color3f[] primvars:displayColor = [\n")
    if quad_colours is not None:
        if len(quad_colours) != len(quads):
            raise ValueError(f"Quad colour count mismatch for {name}: {len(quad_colours)} != {len(quads)}")
        vertex_colours = [quad_colour for quad_colour in quad_colours for _ in range(4)]
    else:
        vertex_colours = [colour] * len(points)
    write_wrapped_values(stream, [v3(value) for value in vertex_colours], width=2)
    stream.write('        ] ( interpolation = "vertex" )\n    }\n')


def burned_surface_geometry(
    result: FireSpreadResult,
    state: FireSpreadState,
    anchor: tuple[float, float],
    *,
    terrain_height_at: Callable[[float, float], float] | None = None,
    subdivisions: int = 3,
) -> tuple[list[list[tuple[float, float, float]]], list[tuple[float, float, float]]]:
    if subdivisions <= 0:
        raise ValueError("Burned-surface subdivisions must be positive")
    west, south, _, _ = result.domain_bounds_l93_m
    size = result.drivers.cell_size_m
    refined_size = size / subdivisions
    quads: list[list[tuple[float, float, float]]] = []
    colours: list[tuple[float, float, float]] = []
    burned_indices = np.argwhere(state.burned_mask)
    if not len(burned_indices):
        return quads, colours
    minimum_y, minimum_x = burned_indices.min(axis=0)
    maximum_y, maximum_x = burned_indices.max(axis=0)
    refined_minimum_y = max(0, (int(minimum_y) - 1) * subdivisions)
    refined_minimum_x = max(0, (int(minimum_x) - 1) * subdivisions)
    refined_maximum_y = min(state.burned_mask.shape[0] * subdivisions, (int(maximum_y) + 2) * subdivisions)
    refined_maximum_x = min(state.burned_mask.shape[1] * subdivisions, (int(maximum_x) + 2) * subdivisions)

    def burned_occupancy(global_east: float, global_north: float) -> float:
        grid_x = (global_east - west) / size - 0.5
        grid_y = (global_north - south) / size - 0.5
        x0 = math.floor(grid_x)
        y0 = math.floor(grid_y)
        tx = grid_x - x0
        ty = grid_y - y0

        def value(sample_y: int, sample_x: int) -> float:
            if 0 <= sample_y < state.burned_mask.shape[0] and 0 <= sample_x < state.burned_mask.shape[1]:
                return 1.0 if bool(state.burned_mask[sample_y, sample_x]) else 0.0
            return 0.0

        bottom = value(y0, x0) * (1.0 - tx) + value(y0, x0 + 1) * tx
        top = value(y0 + 1, x0) * (1.0 - tx) + value(y0 + 1, x0 + 1) * tx
        return bottom * (1.0 - ty) + top * ty

    def height(global_east: float, global_north: float) -> float:
        if terrain_height_at is not None:
            return float(terrain_height_at(global_east, global_north)) + 0.035
        grid_x = min(result.elevation_m.shape[1] - 1, max(0, int((global_east - west) / size)))
        grid_y = min(result.elevation_m.shape[0] - 1, max(0, int((global_north - south) / size)))
        return float(result.elevation_m[grid_y, grid_x]) + 0.035

    for refined_y in range(refined_minimum_y, refined_maximum_y):
        global_y0 = south + refined_y * refined_size
        global_y1 = global_y0 + refined_size
        center_y = (global_y0 + global_y1) * 0.5
        for refined_x in range(refined_minimum_x, refined_maximum_x):
            global_x0 = west + refined_x * refined_size
            global_x1 = global_x0 + refined_size
            center_x = (global_x0 + global_x1) * 0.5
            if burned_occupancy(center_x, center_y) < 0.42:
                continue
            quads.append(
                [
                    (global_x0 - anchor[0], global_y0 - anchor[1], height(global_x0, global_y0)),
                    (global_x1 - anchor[0], global_y0 - anchor[1], height(global_x1, global_y0)),
                    (global_x1 - anchor[0], global_y1 - anchor[1], height(global_x1, global_y1)),
                    (global_x0 - anchor[0], global_y1 - anchor[1], height(global_x0, global_y1)),
                ]
            )
            noise = int(hashlib.sha256(f"{refined_x}:{refined_y}".encode("ascii")).hexdigest()[:4], 16) / 65535.0
            charcoal = 0.010 + noise * 0.014
            colours.append((charcoal * 1.18, charcoal, charcoal * 0.82))
    return quads, colours


def perimeter_quads(segments: list[FireFrontSegment], anchor: tuple[float, float]) -> list[list[tuple[float, float, float]]]:
    quads: list[list[tuple[float, float, float]]] = []
    for segment in segments:
        start = np.asarray(segment.start, dtype=np.float64)
        end = np.asarray(segment.end, dtype=np.float64)
        direction = end[:2] - start[:2]
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            continue
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64) / length * 0.55
        z = max(float(start[2]), float(end[2])) + 0.25
        p0 = (start[0] - anchor[0] - normal[0], start[1] - anchor[1] - normal[1], z)
        p1 = (end[0] - anchor[0] - normal[0], end[1] - anchor[1] - normal[1], z)
        p2 = (end[0] - anchor[0] + normal[0], end[1] - anchor[1] + normal[1], z)
        p3 = (start[0] - anchor[0] + normal[0], start[1] - anchor[1] + normal[1], z)
        quads.append([p0, p1, p2, p3])
    return quads


def front_quads(segments: list[FireFrontSegment], anchor: tuple[float, float]) -> list[list[tuple[float, float, float]]]:
    quads: list[list[tuple[float, float, float]]] = []
    for segment_index, segment in enumerate(segments):
        start = np.asarray(segment.start, dtype=np.float64)
        end = np.asarray(segment.end, dtype=np.float64)
        length = float(np.linalg.norm(end[:2] - start[:2]))
        subdivision_count = max(1, int(math.ceil(length / 4.0)))
        base_height = max(0.8, min(4.2, 0.8 + segment.spread_rate_m_s * 11.0))
        for subdivision_index in range(subdivision_count):
            start_fraction = subdivision_index / subdivision_count
            end_fraction = (subdivision_index + 1) / subdivision_count
            local_start = start + (end - start) * start_fraction
            local_end = start + (end - start) * end_fraction
            phase = segment_index * 0.754877666 + subdivision_index * 1.618033989
            left_height = base_height * (0.72 + 0.38 * (0.5 + 0.5 * math.sin(phase)))
            right_height = base_height * (0.72 + 0.38 * (0.5 + 0.5 * math.sin(phase + 1.9)))
            bottom_left = (local_start[0] - anchor[0], local_start[1] - anchor[1], local_start[2] + 0.18)
            bottom_right = (local_end[0] - anchor[0], local_end[1] - anchor[1], local_end[2] + 0.18)
            horizontal = local_end[:2] - local_start[:2]
            horizontal_length = max(float(np.linalg.norm(horizontal)), 1e-6)
            tangent = horizontal / horizontal_length
            taper = min(horizontal_length * 0.20, 0.55)
            top_right = (
                bottom_right[0] - tangent[0] * taper,
                bottom_right[1] - tangent[1] * taper,
                bottom_right[2] + right_height,
            )
            top_left = (
                bottom_left[0] + tangent[0] * taper,
                bottom_left[1] + tangent[1] * taper,
                bottom_left[2] + left_height,
            )
            quads.append([bottom_left, bottom_right, top_right, top_left])
    return quads


def select_flow_emitters(
    segments: list[FireFrontSegment],
    anchor: tuple[float, float],
    *,
    count: int = FLOW_EMITTER_COUNT,
) -> list[dict[str, float | tuple[float, float, float]]]:
    """Sample a stable number of discrete flame tongues along the active front.

    Flow is a sparse volumetric solver, not a perimeter renderer.  A small,
    stable set of hot fuel emitters gives the solver room to form individual
    flames and coherent plumes.  Injecting hundreds of direct smoke/burn
    spheres made the previous result look like bubbles and restarted the
    volume whenever a state variant changed.
    """

    active = [segment for segment in segments if segment.spread_rate_m_s > 0.0] or list(segments)
    if not active:
        return []
    if count <= 0:
        raise ValueError("Flow emitter count must be positive")

    lengths = np.asarray(
        [
            max(
                float(
                    np.linalg.norm(
                        np.asarray(segment.end[:2], dtype=np.float64)
                        - np.asarray(segment.start[:2], dtype=np.float64)
                    )
                ),
                1e-6,
            )
            for segment in active
        ],
        dtype=np.float64,
    )
    cumulative = np.cumsum(lengths)
    total_length = float(cumulative[-1])
    emitters: list[dict[str, float | tuple[float, float, float]]] = []
    for emitter_index in range(count):
        distance = (emitter_index + 0.5) * total_length / count
        segment_index = int(np.searchsorted(cumulative, distance, side="right"))
        segment_index = min(segment_index, len(active) - 1)
        previous_distance = float(cumulative[segment_index - 1]) if segment_index else 0.0
        segment = active[segment_index]
        start = np.asarray(segment.start, dtype=np.float64)
        end = np.asarray(segment.end, dtype=np.float64)
        horizontal = end[:2] - start[:2]
        length = float(lengths[segment_index])
        normal = np.asarray((-horizontal[1], horizontal[0]), dtype=np.float64) / length
        fraction = min(1.0, max(0.0, (distance - previous_distance) / length))
        center = start + (end - start) * fraction
        phase = segment_index * 1.618033989 + emitter_index * 0.754877666
        intensity = max(0.18, min(1.0, segment.spread_rate_m_s / 0.24))
        radius = 0.82 + 0.48 * intensity + 0.16 * math.sin(phase + 1.3)
        lateral_jitter = math.sin(phase) * min(0.42, length * 0.08)
        emitters.append(
            {
                "position": (
                    center[0] - anchor[0] + normal[0] * lateral_jitter,
                    center[1] - anchor[1] + normal[1] * lateral_jitter,
                    center[2] + max(0.35, radius * 0.34),
                ),
                "radius_m": max(0.68, radius),
                "fuel": 0.88 + 0.10 * intensity,
                "temperature": 2.70 + 0.65 * intensity + 0.14 * math.sin(phase + 0.7),
                "burn": 0.0,
                "smoke": 0.0,
                "vertical_velocity_m_s": 0.0,
                "spread_rate_m_s": segment.spread_rate_m_s,
            }
        )
    return emitters


def write_smoke_sources(stream: Any, emitters: list[dict[str, float | tuple[float, float, float]]]) -> None:
    stream.write('    def Points "SmokeSources" (\n')
    stream.write(semantic_api_metadata(indent="        "))
    stream.write("    )\n    {\n")
    stream.write(semantic_properties("smoke_source", indent="        "))
    stream.write("        point3f[] points = [\n")
    write_wrapped_values(stream, [v3(emitter["position"]) for emitter in emitters], width=2)
    stream.write("        ]\n        float[] widths = [\n")
    write_wrapped_values(stream, [f(float(emitter["radius_m"]) * 2.0) for emitter in emitters], width=12)
    stream.write("        ]\n        custom string fireviewer:flow_emitter_contract = \"truth_points_matching_hidden_native_flow_emitters\"\n    }\n")


def ordered_flow_emitter_states(
    result: FireSpreadResult,
    anchor: tuple[float, float],
) -> list[list[dict[str, float | tuple[float, float, float]]]]:
    """Keep emitter identities spatially stable across all 180 states."""

    ordered_states: list[list[dict[str, float | tuple[float, float, float]]]] = []
    previous: list[dict[str, float | tuple[float, float, float]]] | None = None
    for state in result.states:
        current = select_flow_emitters(state.front_segments, anchor)
        if len(current) != FLOW_EMITTER_COUNT:
            raise ValueError(
                f"state_{state.state_index:03d} yielded {len(current)} Flow emitters, expected {FLOW_EMITTER_COUNT}"
            )
        if previous is not None:
            previous_positions = np.asarray([emitter["position"] for emitter in previous], dtype=np.float64)
            current_positions = np.asarray([emitter["position"] for emitter in current], dtype=np.float64)
            distances = np.linalg.norm(previous_positions[:, None, :] - current_positions[None, :, :], axis=2)
            rows, columns = linear_sum_assignment(distances)
            matched: list[dict[str, float | tuple[float, float, float]] | None] = [None] * FLOW_EMITTER_COUNT
            for row, column in zip(rows, columns):
                matched[int(row)] = current[int(column)]
            if any(emitter is None for emitter in matched):
                raise RuntimeError(f"Flow emitter assignment is incomplete at state_{state.state_index:03d}")
            current = [emitter for emitter in matched if emitter is not None]
        ordered_states.append(current)
        previous = current
    return ordered_states


def select_flow_front_patches(
    segments: list[FireFrontSegment],
    anchor: tuple[float, float],
    *,
    count: int = FLOW_FRONT_PATCH_COUNT,
) -> list[dict[str, Any]]:
    """Build fixed-topology terrain-following patches for a continuous Flow front."""

    active = [segment for segment in segments if segment.spread_rate_m_s > 0.0] or list(segments)
    if not active:
        return []
    if count <= 0:
        raise ValueError("Flow front patch count must be positive")
    lengths = np.asarray(
        [
            max(
                float(
                    np.linalg.norm(
                        np.asarray(segment.end[:2], dtype=np.float64)
                        - np.asarray(segment.start[:2], dtype=np.float64)
                    )
                ),
                1e-6,
            )
            for segment in active
        ],
        dtype=np.float64,
    )
    cumulative = np.cumsum(lengths)
    total_length = float(cumulative[-1])
    patch_spacing = total_length / count
    patches: list[dict[str, Any]] = []
    for patch_index in range(count):
        distance = (patch_index + 0.5) * patch_spacing
        segment_index = min(int(np.searchsorted(cumulative, distance, side="right")), len(active) - 1)
        previous_distance = float(cumulative[segment_index - 1]) if segment_index else 0.0
        segment = active[segment_index]
        start = np.asarray(segment.start, dtype=np.float64)
        end = np.asarray(segment.end, dtype=np.float64)
        horizontal = end[:2] - start[:2]
        horizontal_length = float(lengths[segment_index])
        tangent_xy = horizontal / horizontal_length
        normal_xy = np.asarray((-tangent_xy[1], tangent_xy[0]), dtype=np.float64)
        fraction = min(1.0, max(0.0, (distance - previous_distance) / horizontal_length))
        center = start + (end - start) * fraction
        intensity = max(0.18, min(1.0, segment.spread_rate_m_s / 0.24))
        half_length = max(0.38, min(8.0, patch_spacing * 0.62))
        half_width = 0.70 + 0.95 * intensity
        slope_per_horizontal_m = float(end[2] - start[2]) / horizontal_length
        center_local = np.asarray(
            (center[0] - anchor[0], center[1] - anchor[1], center[2] + 0.32),
            dtype=np.float64,
        )
        tangent = np.asarray((tangent_xy[0], tangent_xy[1], slope_per_horizontal_m), dtype=np.float64)
        normal = np.asarray((normal_xy[0], normal_xy[1], 0.0), dtype=np.float64)
        vertices = (
            center_local - tangent * half_length - normal * half_width,
            center_local + tangent * half_length - normal * half_width,
            center_local + tangent * half_length + normal * half_width,
            center_local - tangent * half_length + normal * half_width,
        )
        patches.append(
            {
                "center": tuple(float(value) for value in center_local),
                "vertices": tuple(tuple(float(value) for value in vertex) for vertex in vertices),
                "spread_rate_m_s": float(segment.spread_rate_m_s),
            }
        )
    return patches


def ordered_flow_front_patch_states(
    result: FireSpreadResult,
    anchor: tuple[float, float],
) -> list[list[dict[str, Any]]]:
    """Keep independent front-patch identities spatially stable between states."""

    ordered_states: list[list[dict[str, Any]]] = []
    previous: list[dict[str, Any]] | None = None
    for state in result.states:
        current = select_flow_front_patches(state.front_segments, anchor)
        if len(current) != FLOW_FRONT_PATCH_COUNT:
            raise ValueError(
                f"state_{state.state_index:03d} yielded {len(current)} Flow front patches, "
                f"expected {FLOW_FRONT_PATCH_COUNT}"
            )
        if previous is not None:
            previous_centers = np.asarray([patch["center"] for patch in previous], dtype=np.float64)
            current_centers = np.asarray([patch["center"] for patch in current], dtype=np.float64)
            rows, columns = linear_sum_assignment(
                np.linalg.norm(previous_centers[:, None, :] - current_centers[None, :, :], axis=2)
            )
            matched: list[dict[str, Any] | None] = [None] * FLOW_FRONT_PATCH_COUNT
            for row, column in zip(rows, columns):
                matched[int(row)] = current[int(column)]
            if any(patch is None for patch in matched):
                raise RuntimeError(f"Flow front patch assignment is incomplete at state_{state.state_index:03d}")
            current = [patch for patch in matched if patch is not None]
        ordered_states.append(current)
        previous = current
    return ordered_states


def smoke_plume_mesh_vertices(
    emitters: list[dict[str, float | tuple[float, float, float]]],
    *,
    lift_m: float = FLOW_SMOKE_PLUME_LIFT_M,
) -> list[tuple[float, float, float]]:
    """Build invisible horizontal smoke-source quads aligned above flame tongues."""

    if lift_m <= 0.0:
        raise ValueError("Flow smoke plume lift must be positive")
    vertices: list[tuple[float, float, float]] = []
    for emitter in emitters:
        position = tuple(float(value) for value in emitter["position"])
        radius = float(emitter["radius_m"])
        half_extent = max(0.28, min(0.62, radius * 0.42))
        plume_z = position[2] + lift_m
        vertices.extend(
            (
                (position[0] - half_extent, position[1] - half_extent, plume_z),
                (position[0] + half_extent, position[1] - half_extent, plume_z),
                (position[0] + half_extent, position[1] + half_extent, plume_z),
                (position[0] - half_extent, position[1] + half_extent, plume_z),
            )
        )
    return vertices


def write_flow_time_samples(
    stream: Any,
    *,
    declaration: str,
    name: str,
    values: list[tuple[float, Any]],
    formatter: Callable[[Any], str],
    indent: str = "            ",
) -> None:
    stream.write(f"{indent}{declaration} {name}.timeSamples = {{\n")
    for sample_index, (sample_time, value) in enumerate(values):
        suffix = "," if sample_index < len(values) - 1 else ""
        stream.write(f"{indent}    {f(sample_time)}: {formatter(value)}{suffix}\n")
    stream.write(f"{indent}}}\n")


def flow_state_time_samples(
    values: list[Any],
    *,
    seconds_per_state: float,
    stepped_transitions: bool,
) -> list[tuple[float, Any]]:
    samples: list[tuple[float, Any]] = []
    for state_index, value in enumerate(values):
        start = state_index * seconds_per_state
        samples.append((start, value))
        if stepped_transitions and state_index < len(values) - 1:
            samples.append((start + seconds_per_state - 0.001, value))
    return samples


def write_flow_layer(
    output: Path,
    *,
    result: FireSpreadResult,
    anchor: tuple[float, float],
    seconds_per_state: float = FLOW_SECONDS_PER_STATE,
    stepped_state_transitions: bool = False,
) -> None:
    """Write one persistent, time-sampled Flow wildfire simulation."""

    if seconds_per_state <= 0.0:
        raise ValueError("Flow state duration must be positive")
    emitter_states = ordered_flow_emitter_states(result, anchor)
    front_patch_states = ordered_flow_front_patch_states(result, anchor)
    end_time = len(emitter_states) * seconds_per_state
    wind_angle = math.radians(result.drivers.wind_to_degrees_from_east)
    flame_velocity = (
        math.cos(wind_angle) * result.drivers.wind_speed_m_s * 0.35,
        math.sin(wind_angle) * result.drivers.wind_speed_m_s * 0.35,
        3.2,
    )
    smoke_velocity = (
        math.cos(wind_angle) * result.drivers.wind_speed_m_s * 0.04,
        math.sin(wind_angle) * result.drivers.wind_speed_m_s * 0.04,
        8.5,
    )
    mesh_face_counts = [4] * FLOW_FRONT_PATCH_COUNT
    mesh_face_indices = list(range(FLOW_FRONT_PATCH_COUNT * 4))
    mesh_position_samples = flow_state_time_samples(
        [[vertex for patch in state_patches for vertex in patch["vertices"]] for state_patches in front_patch_states],
        seconds_per_state=seconds_per_state,
        stepped_transitions=stepped_state_transitions,
    )
    smoke_mesh_face_counts = [4] * FLOW_EMITTER_COUNT
    smoke_mesh_face_indices = list(range(FLOW_EMITTER_COUNT * 4))
    smoke_mesh_position_samples = flow_state_time_samples(
        [smoke_plume_mesh_vertices(state_emitters) for state_emitters in emitter_states],
        seconds_per_state=seconds_per_state,
        stepped_transitions=stepped_state_transitions,
    )
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "#usda 1.0\n(\n"
            "    metersPerUnit = 1\n"
            "    upAxis = \"Z\"\n"
            "    startTimeCode = 0\n"
            f"    endTimeCode = {f(end_time)}\n"
            "    timeCodesPerSecond = 1\n"
            ")\n\n"
        )
        stream.write('over "World"\n{\n    over "FireScenario"\n    {\n        def Xform "FlowVisual"\n        {\n')
        stream.write('            custom string fireviewer:renderer = "omni.flowusd"\n')
        stream.write('            custom string fireviewer:mode = "persistent_mesh_front_wildland_combustion"\n')
        stream.write(f'            custom string fireviewer:simulation_id = "{result.simulation_id}"\n')
        stream.write(f"            custom int fireviewer:emitter_count = {FLOW_EMITTER_COUNT + 2}\n")
        stream.write(f"            custom int fireviewer:hotspot_emitter_count = {FLOW_EMITTER_COUNT}\n")
        stream.write(f"            custom int fireviewer:front_patch_count = {FLOW_FRONT_PATCH_COUNT}\n")
        stream.write(f"            custom int fireviewer:smoke_plume_count = {FLOW_EMITTER_COUNT}\n")
        stream.write(f"            custom float fireviewer:seconds_per_state = {f(seconds_per_state)}\n")
        stream.write(
            '            custom string fireviewer:state_transition_contract = '
            f'"{"stepped_daily_hold" if stepped_state_transitions else "continuous_linear"}"\n'
        )
        stream.write('            def FlowSimulate "Simulate"\n            {\n')
        stream.write('                int layer = 0\n                custom uint blockMinLifetime = 12\n                custom float densityCellSize = 0.45\n                custom int levelCount = 1\n                custom bool autoCellSize = 0\n                custom bool clearOnRescale = 0\n                custom bool enableLowPrecisionDensity = 0\n                custom bool enableLowPrecisionVelocity = 0\n                custom bool enableSmallBlocks = 0\n                custom bool enableVariableTimeStep = 0\n                custom bool forceClear = 0\n                custom bool forceDisableCoreSimulation = 0\n                custom bool forceDisableEmitters = 0\n                custom bool forceSimulate = 0\n                custom bool interpolateTimeSteps = 0\n                custom bool simulateWhenPaused = 0\n                custom uint maxStepsPerSimulate = 2\n                custom float stepsPerSecond = 60\n                custom float timeScale = 0.85\n                custom uint velocitySubSteps = 2\n                custom bool physicsCollisionEnabled = 0\n                custom bool physicsConvexCollision = 0\n')
        stream.write('                def FlowAdvectionCombustionParams "advection"\n                {\n                    custom bool enabled = 1\n                    custom bool combustionEnabled = 1\n                    custom bool downsampleEnabled = 1\n                    custom bool forceFadeEnabled = 0\n                    custom bool globalFetch = 1\n                    custom float buoyancyMaxSmoke = 1\n                    custom float buoyancyPerSmoke = 4.5\n                    float buoyancyPerTemp = 2.8\n                    custom float burnPerTemp = 4\n                    custom float coolingRate = 0.75\n                    custom float divergencePerBurn = 0.12\n                    custom float fuelPerBurn = 0.25\n                    float3 gravity = (0, 0, -9.81)\n                    custom float ignitionTemp = 0.05\n                    custom float smokePerBurn = 3.5\n                    custom float tempPerBurn = 5.5\n')
        channel_profiles = {
            "smoke": (0.025, 0.025, 0.9),
            "velocity": (0.01, 0.05, 0.5),
            "divergence": (0.01, 0.25, 0.5),
            "temperature": (0.03, 0.35, 0.9),
            "fuel": (0.02, 0.60, 0.9),
            "burn": (0.01, 0.45, 0.9),
        }
        for channel, (damping, fade, blend) in channel_profiles.items():
            stream.write(
                f'                    def FlowAdvectionChannelParams "{channel}"\n'
                '                    {\n'
                f'                        custom float damping = {f(damping)}\n'
                f'                        float fade = {f(fade)}\n'
                f'                        custom float secondOrderBlendFactor = {f(blend)}\n'
                '                        custom float secondOrderBlendThreshold = 0.001\n'
                '                    }\n'
            )
        stream.write('                }\n')
        stream.write('                def FlowVorticityParams "vorticity"\n                {\n                    custom bool enabled = 1\n                    float forceScale = 0.85\n                    custom float burnMask = 0.08\n                    custom float smokeMask = 0.28\n                    custom float temperatureMask = 0.12\n                    custom float velocityLogScale = 100\n                    custom float velocityMask = 1\n                }\n                def FlowPressureParams "pressure"\n                {\n                    custom bool enabled = 1\n                }\n                def FlowSummaryAllocateParams "summaryAllocate"\n                {\n                    custom bool enabled = 1\n                    custom bool enableNeighborAllocation = 1\n                    custom float smokeThreshold = 0.01\n                    custom float speedThreshold = 0.4\n                    custom float speedThresholdMinSmoke = 0\n                }\n            }\n')
        stream.write('            def FlowOffscreen "Offscreen"\n            {\n                int layer = 0\n                def FlowRayMarchColormapParams "colormap"\n                {\n                    custom float colorScale = 3.0\n                    custom float[] colorScalePoints = [1, 1, 1, 1, 1, 1]\n                    custom uint resolution = 64\n                    float4[] rgbaPoints = [(0.008, 0.010, 0.012, 0), (0.22, 0.24, 0.26, 0.16), (0.52, 0.50, 0.47, 0.38), (1.2, 0.10, 0.008, 0.78), (12, 2.2, 0.08, 0.92), (48, 17, 2.4, 0.84)]\n                    custom float[] xPoints = [0, 0.035, 0.12, 0.32, 0.62, 1]\n                }\n                def FlowShadowParams "shadow"\n                {\n                    custom bool enabled = 1\n                    custom float attenuation = 4.5\n                    custom bool coarsePropagate = 1\n                    custom float minIntensity = 0.08\n                    custom uint numSteps = 24\n                    custom float stepOffsetScale = 1\n                    custom float stepSizeScale = 0.65\n                }\n                def FlowDebugVolumeParams "debugVolume"\n                {\n                    custom bool enableSpeedAsTemperature = 0\n                    custom bool enableVelocityAsDensity = 0\n                    custom float3 velocityScale = (0.01, 0.01, 0.01)\n                }\n            }\n')
        stream.write('            def FlowRender "Render"\n            {\n                int layer = 0\n                def FlowRayMarchParams "rayMarch"\n                {\n                    custom float attenuation = 6\n                    custom float colorScale = 1.15\n                    custom bool enableBlockWireframe = 0\n                    custom bool enableRawMode = 0\n                    custom float shadowFactor = 1\n                    custom float stepSizeScale = 0.6\n                }\n                def FlowRenderSettingsParams "RenderSettings"\n                {\n                    custom bool compositeEnabled = 1\n                    custom bool enableAutoApply = 1\n                    custom bool flowEnabled = 1\n                    custom int maxBlocks = 16384\n                    custom bool pathTracingEnabled = 1\n                    custom bool pathTracingShadowsEnabled = 1\n                    custom bool rayTracedReflectionsEnabled = 1\n                    custom bool rayTracedShadowsEnabled = 1\n                    custom bool rayTracedTranslucencyEnabled = 1\n                }\n            }\n')
        stream.write('            def FlowEmitterMesh "FrontRibbonEmitter"\n            {\n                int layer = 0\n                custom bool enabled = 1\n                custom bool allocateMask = 1\n                custom bool applyPostPressure = 0\n                custom float minDistance = -0.35\n                custom float maxDistance = 1.30\n                custom uint numSubSteps = 2\n                custom bool orientationLeftHanded = 0\n')
        write_flow_time_samples(
            stream,
            declaration="float3[]",
            name="meshPositions",
            values=mesh_position_samples,
            formatter=v3_array,
            indent="                ",
        )
        stream.write(f"                int[] meshFaceVertexIndices = {int_array(mesh_face_indices)}\n")
        stream.write(f"                int[] meshFaceVertexCounts = {int_array(mesh_face_counts)}\n")
        stream.write('                float temperature = 3.2\n                float coupleRateTemperature = 10\n                float fuel = 0.96\n                float coupleRateFuel = 3.5\n                float burn = 0\n                float coupleRateBurn = 0\n                float smoke = 0\n                float coupleRateSmoke = 0\n')
        stream.write(f'                float3 velocity = {v3(flame_velocity)}\n                bool velocityIsWorldSpace = 1\n                float coupleRateVelocity = 1.4\n                float divergence = 0\n                float coupleRateDivergence = 0\n            }}\n')
        stream.write('            def FlowEmitterMesh "SmokePlumeEmitter"\n            {\n                int layer = 0\n                custom bool enabled = 1\n                custom bool allocateMask = 1\n                custom bool applyPostPressure = 0\n                custom float minDistance = -0.12\n                custom float maxDistance = 0.45\n                custom uint numSubSteps = 2\n                custom bool orientationLeftHanded = 0\n')
        write_flow_time_samples(
            stream,
            declaration="float3[]",
            name="meshPositions",
            values=smoke_mesh_position_samples,
            formatter=v3_array,
            indent="                ",
        )
        stream.write(f"                int[] meshFaceVertexIndices = {int_array(smoke_mesh_face_indices)}\n")
        stream.write(f"                int[] meshFaceVertexCounts = {int_array(smoke_mesh_face_counts)}\n")
        stream.write(f'                custom string fireviewer:placement_contract = "same_xy_as_hotspot_front_lifted_{f(FLOW_SMOKE_PLUME_LIFT_M)}m"\n')
        stream.write('                float temperature = 0.25\n                float coupleRateTemperature = 1.4\n                float fuel = 0\n                float coupleRateFuel = 0\n                float burn = 0\n                float coupleRateBurn = 0\n')
        stream.write(f'                float smoke = {f(FLOW_SMOKE_SOURCE)}\n                float coupleRateSmoke = 1.6\n')
        stream.write(f'                float3 velocity = {v3(smoke_velocity)}\n                bool velocityIsWorldSpace = 1\n                float coupleRateVelocity = 3.2\n                float divergence = 0\n                float coupleRateDivergence = 0\n            }}\n')
        for emitter_index in range(FLOW_EMITTER_COUNT):
            samples = flow_state_time_samples(
                [state_emitters[emitter_index] for state_emitters in emitter_states],
                seconds_per_state=seconds_per_state,
                stepped_transitions=stepped_state_transitions,
            )
            stream.write(f'            def FlowEmitterSphere "Emitter_{emitter_index + 1:02d}"\n            {{\n')
            stream.write('                int layer = 0\n                custom bool enabled = 1\n                custom float allocationScale = 1.35\n                custom bool applyPostPressure = 0\n                custom bool multisample = 1\n                custom uint numSubSteps = 2\n                custom bool radiusIsWorldSpace = 1\n')
            write_flow_time_samples(stream, declaration="custom float3", name="position", values=[(time, emitter["position"]) for time, emitter in samples], formatter=v3, indent="                ")
            write_flow_time_samples(stream, declaration="custom float", name="radius", values=[(time, emitter["radius_m"]) for time, emitter in samples], formatter=lambda value: f(float(value)), indent="                ")
            write_flow_time_samples(stream, declaration="custom float", name="fuel", values=[(time, emitter["fuel"]) for time, emitter in samples], formatter=lambda value: f(float(value)), indent="                ")
            write_flow_time_samples(stream, declaration="float", name="temperature", values=[(time, emitter["temperature"]) for time, emitter in samples], formatter=lambda value: f(float(value)), indent="                ")
            stream.write('                custom float burn = 0\n                custom float smoke = 0\n')
            stream.write(f'                float3 velocity = {v3(flame_velocity)}\n                custom bool velocityIsWorldSpace = 1\n                float coupleRateTemperature = 10\n                custom float coupleRateFuel = 3.5\n                custom float coupleRateBurn = 0\n                custom float coupleRateSmoke = 0\n                custom float coupleRateVelocity = 1.4\n                custom float divergence = 0\n                custom float coupleRateDivergence = 0\n')
            write_flow_time_samples(stream, declaration="custom float", name="fireviewer:front_spread_rate_m_s", values=[(time, emitter["spread_rate_m_s"]) for time, emitter in samples], formatter=lambda value: f(float(value)), indent="                ")
            stream.write('            }\n')
        stream.write('        }\n    }\n}\n')


def write_burned_instancer(stream: Any, trees: list[TreeInstance]) -> None:
    if not trees:
        return
    stream.write('    def Scope "BurnedVegetation"\n    {\n        def Scope "Prototypes"\n        {\n')
    write_tree_mesh(stream, path="BurnedTreeLowPoly", colour=(0.12, 0.065, 0.025))
    stream.write("        }\n")
    write_instancer(stream, path="Trees", prototype_paths=["/FireState/Truth3D/BurnedVegetation/Prototypes/BurnedTreeLowPoly"], trees=trees, semantic="burned_vegetation")
    stream.write("    }\n")


def write_burned_tree_ids(stream: Any, tree_ids: list[int]) -> None:
    """Author only the IDs needed to hide source vegetation at runtime.

    Retrospective perimeters may cover hundreds of thousands of source trees.
    Re-instancing every consumed tree as a second low-poly proxy would duplicate
    geometry and contradict the runtime contract that removes burned crowns.
    """

    stream.write('    def Scope "BurnedVegetation"\n    {\n')
    stream.write('        def Xform "Trees"\n        {\n')
    stream.write("            custom int[] fireviewer:burned_tree_ids = [\n")
    write_wrapped_values(stream, [str(tree_id) for tree_id in tree_ids], width=12)
    stream.write("            ]\n")
    stream.write('            custom string fireviewer:representation = "source_tree_ids_hidden_at_runtime"\n')
    stream.write("        }\n    }\n")


def write_fire_state(
    output: Path,
    *,
    result: FireSpreadResult,
    state: FireSpreadState,
    state_count: int,
    anchor: tuple[float, float],
    trees: list[TreeInstance],
    tree_by_id: dict[int, TreeInstance] | None = None,
    terrain_height_at: Callable[[float, float], float] | None = None,
    states_per_day: int = 10,
    observation_seconds_per_state: int = 8640,
    truth_scope: str = "synthetic_physically_driven_fire_spread_on_real_uploaded_map_not_incident_reconstruction",
    local_date: str | None = None,
    active_in_scene: bool = True,
    burned_tree_ids_override: Iterable[int] | None = None,
    render_burned_tree_proxies: bool = True,
    burned_surface_subdivisions: int = 3,
) -> dict[str, Any]:
    if states_per_day <= 0 or observation_seconds_per_state <= 0:
        raise ValueError("Fire-state day and observation cadence must be positive")
    emitters = select_flow_emitters(state.front_segments, anchor)
    burned_tree_ids = sorted(
        int(tree_id)
        for tree_id in (
            state.burned_tree_ids
            if burned_tree_ids_override is None
            else burned_tree_ids_override
        )
    )
    if render_burned_tree_proxies:
        if tree_by_id is None:
            tree_by_id = {tree.tree_id: tree for tree in trees}
        burned_trees = [tree_by_id[tree_id] for tree_id in burned_tree_ids if tree_id in tree_by_id]
    else:
        burned_trees = []
    day_index = (state.state_index - 1) // states_per_day + 1
    state_in_day = (state.state_index - 1) % states_per_day + 1
    observation_elapsed_s = (state.state_index - 1) * observation_seconds_per_state
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header("FireState"))
        stream.write('def Xform "FireState" (kind = "component")\n{\n')
        stream.write(f"    custom int fireviewer:state_index = {state.state_index}\n")
        stream.write(f"    custom int fireviewer:state_count = {state_count}\n")
        stream.write(f"    custom int fireviewer:incident_day_index = {day_index}\n")
        stream.write(f"    custom int fireviewer:state_in_day = {state_in_day}\n")
        stream.write(f"    custom double fireviewer:observation_elapsed_s = {f(observation_elapsed_s)}\n")
        stream.write(f'    custom string fireviewer:truth_scope = "{truth_scope}"\n')
        if local_date is not None:
            stream.write(f'    custom string fireviewer:source_local_date = "{local_date}"\n')
        stream.write(f"    custom bool fireviewer:active_in_scene = {1 if active_in_scene else 0}\n")
        stream.write(f'    custom string fireviewer:simulation_id = "{result.simulation_id}"\n')
        stream.write(f"    custom double2 fireviewer:ignition_l93_m = ({f(result.ignition_l93_m[0])}, {f(result.ignition_l93_m[1])})\n")
        stream.write(f"    custom double fireviewer:elapsed_s = {f(state.elapsed_s)}\n")
        stream.write(f"    custom double fireviewer:burned_area_m2 = {f(state.burned_area_m2)}\n")
        stream.write(f"    custom double fireviewer:active_front_length_m = {f(state.active_front_length_m)}\n")
        stream.write('    def Scope "Truth3D"\n    {\n')
        burned_quads, burned_colours = burned_surface_geometry(
            result,
            state,
            anchor,
            terrain_height_at=terrain_height_at,
            subdivisions=burned_surface_subdivisions,
        )
        write_quad_mesh(
            stream,
            name="BurnedSurface",
            quads=burned_quads,
            semantic="burned_ground",
            colour=(0.018, 0.015, 0.012),
            quad_colours=burned_colours,
        )
        write_quad_mesh(stream, name="BurnedPerimeter", quads=perimeter_quads(state.front_segments, anchor), semantic="fire_perimeter", colour=(0.12, 0.045, 0.012))
        write_quad_mesh(stream, name="VisibleFireFront", quads=front_quads(state.front_segments, anchor), semantic="fire_front", colour=(1.0, 0.22, 0.015))
        write_smoke_sources(stream, emitters)
        if render_burned_tree_proxies:
            write_burned_instancer(stream, burned_trees)
        else:
            write_burned_tree_ids(stream, burned_tree_ids)
        stream.write(f"        custom int fireviewer:burned_tree_count = {len(burned_tree_ids)}\n")
        stream.write("    }\n")
        stream.write("}\n")
    return {
        "state_id": f"state_{state.state_index:03d}",
        "path": f"states/state_{state.state_index:03d}.usda",
        "incident_day_index": day_index,
        "state_in_day": state_in_day,
        "observation_elapsed_s": observation_elapsed_s,
        "simulation_id": result.simulation_id,
        "elapsed_s": round(state.elapsed_s, 3),
        "burned_area_m2": round(state.burned_area_m2, 3),
        "active_front_length_m": round(state.active_front_length_m, 3),
        "mean_front_spread_rate_m_s": round(state.mean_front_spread_rate_m_s, 6),
        "smoke_source_count": len(emitters),
        "flow_emitter_count": len(emitters),
        "burned_tree_count": len(burned_tree_ids),
        "source_local_date": local_date,
        "active_in_scene": bool(active_in_scene),
    }


def write_propagation_sidecars(
    output: Path,
    result: FireSpreadResult,
    *,
    states_per_day: int = 10,
    observation_seconds_per_state: int = 8640,
) -> dict[str, Any]:
    if states_per_day <= 0 or observation_seconds_per_state <= 0:
        raise ValueError("Propagation day and observation cadence must be positive")
    propagation = {
        **result.model_metadata,
        "simulation_id": result.simulation_id,
        "domain_bounds_l93_m": [float(value) for value in result.domain_bounds_l93_m],
        "ignition_l93_m": [float(value) for value in result.ignition_l93_m],
        "drivers": result.drivers.as_dict(),
        "grid_shape": [int(value) for value in result.arrival_time_s.shape],
        "state_count": len(result.states),
        "states": [
            {
                "state_id": f"state_{state.state_index:03d}",
                "incident_day_index": (state.state_index - 1) // states_per_day + 1,
                "state_in_day": (state.state_index - 1) % states_per_day + 1,
                "observation_elapsed_s": (state.state_index - 1) * observation_seconds_per_state,
                "elapsed_s": round(state.elapsed_s, 3),
                "burned_area_m2": round(state.burned_area_m2, 3),
                "active_front_length_m": round(state.active_front_length_m, 3),
                "mean_front_spread_rate_m_s": round(state.mean_front_spread_rate_m_s, 6),
                "burned_tree_count": len(state.burned_tree_ids),
            }
            for state in result.states
        ],
        "field_file": "propagation-field.npz",
    }
    propagation_path = output / "scenarios/propagation.json"
    propagation_path.write_text(canonical_json(propagation), encoding="utf-8")
    np.savez_compressed(
        output / "scenarios/propagation-field.npz",
        elevation_m=result.elevation_m,
        fuel_load=result.fuel_load,
        burnable_mask=result.burnable_mask,
        arrival_time_s=result.arrival_time_s,
        spread_rate_m_s=result.spread_rate_m_s,
    )
    return {"path": "scenarios/propagation.json", "sha256": sha256_file(propagation_path), "simulation_id": result.simulation_id}


def write_scenario_layer(
    output: Path,
    states: list[dict[str, Any]],
    propagation: dict[str, Any],
    *,
    truth_scope: str = "synthetic_physically_driven_fire_spread_on_real_uploaded_map_not_incident_reconstruction",
    state_selection_contract: str = "incident_18_days_10_states_per_day_180_named_states",
) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header())
        stream.write('over "World"\n{\n    def Xform "FireScenario" (\n        kind = "component"\n        variants = { string fire_state = "state_001" }\n        prepend variantSets = "fire_state"\n    )\n    {\n')
        stream.write(f'        custom string fireviewer:truth_scope = "{truth_scope}"\n')
        stream.write(f'        custom string fireviewer:simulation_id = "{propagation["simulation_id"]}"\n')
        stream.write('        custom string fireviewer:propagation_sidecar = "propagation.json"\n')
        stream.write(f'        custom string fireviewer:state_selection_contract = "{state_selection_contract}"\n')
        stream.write('        variantSet "fire_state" = {\n')
        for record in states:
            stream.write(f'            "{record["state_id"]}" (\n                prepend references = @{record["path"]}@</FireState>\n            ) {{\n            }}\n')
        stream.write("        }\n    }\n}\n")


def write_appearance_layer(output: Path) -> None:
    entries = (
        ("FlowClose", "omni.flowusd", "/World/FireScenario/FlowVisual", "near_field_combustion_and_smoke", 0, 180),
        ("SmokeMidDistance", "omni.flowusd", "/World/FireScenario/FlowVisual", "mid_distance_smoke_volume", 180, 1800),
        ("DistantFire", "rtx_simplified", "/World/FireScenario/Truth3D/VisibleFireFront", "emissive_simplified_distant_fire", 1800, 1000000),
        ("DecorDegradation", "usd_material_override", "/World/FireScenario/Truth3D/BurnedSurface", "burned_ground_surface_only", 0, 1000000),
    )
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header())
        stream.write('over "World"\n{\n    def Xform "Appearance" (kind = "component")\n    {\n')
        stream.write('        custom string fireviewer:runtime = "Omniverse RTX / Isaac Sim"\n')
        stream.write('        custom string fireviewer:execution_contract = "runtime_must_enable_omni.flowusd_and_rtx_for_actual_combustion_and_smoke"\n')
        for name, runtime, source, mode, minimum, maximum in entries:
            stream.write(f'        def Scope "{name}"\n        {{\n')
            stream.write(f'            custom string fireviewer:renderer = "{runtime}"\n')
            stream.write(f'            custom string fireviewer:source_prim = "{source}"\n')
            stream.write(f'            custom string fireviewer:mode = "{mode}"\n')
            stream.write(f"            custom float2 fireviewer:distance_range_m = ({f(minimum)}, {f(maximum)})\n")
            stream.write("        }\n")
        stream.write("    }\n}\n")


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise ValueError("Cannot normalize zero vector")
    return vector / length


def quaternion_from_rotation(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale)
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)


def look_at_quaternion(position: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, float, float, float]:
    source = np.asarray(position, dtype=np.float64)
    focus = np.asarray(target, dtype=np.float64)
    forward = normalize(focus - source)
    up_hint = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.98:
        up_hint = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    local_to_world = np.column_stack((right, up, -forward))
    return quaternion_from_rotation(local_to_world)


def fixed_camera_plan(
    center: tuple[float, float],
    bounds: tuple[float, float, float, float],
    sampler: HeightSampler,
    anchor: tuple[float, float],
    access_samples: list[AccessSurfaceSample],
    trees: list[TreeInstance],
    boxes: list[OcclusionBox],
) -> list[dict[str, Any]]:
    """Choose wide human photo sites with a verified clear axis to the fire."""

    road_samples = [sample for sample in access_samples if sample.source_type == "road"]
    if not road_samples:
        raise ValueError("No mapped road surface is available for photographer cameras")

    xmin, ymin, xmax, ymax = bounds
    span = min(xmax - xmin, ymax - ymin)
    center_ground = sampler.at(center[0], center[1])
    target_global = (center[0], center[1], center_ground + 25.0)
    target_local = (target_global[0] - anchor[0], target_global[1] - anchor[1], target_global[2])
    eye_height = 1.70

    tree_xy = np.empty((len(trees), 2), dtype=np.float64)
    tree_radii = np.empty(len(trees), dtype=np.float64)
    tree_base = np.empty(len(trees), dtype=np.float64)
    tree_top = np.empty(len(trees), dtype=np.float64)
    for index, tree in enumerate(trees):
        tree_xy[index] = tree.position[:2]
        tree_radii[index] = tree.scale[0]
        tree_base[index] = tree.position[2]
        tree_top[index] = tree.position[2] + tree.scale[2]
    tree_index = cKDTree(tree_xy) if len(tree_xy) else None

    box_centers = np.asarray(
        [[(box.minimum[0] + box.maximum[0]) * 0.5, (box.minimum[1] + box.maximum[1]) * 0.5] for box in boxes],
        dtype=np.float64,
    )
    box_half_diagonal = np.asarray(
        [math.hypot((box.maximum[0] - box.minimum[0]) * 0.5, (box.maximum[1] - box.minimum[1]) * 0.5) for box in boxes],
        dtype=np.float64,
    )
    box_min_z = np.asarray([box.minimum[2] for box in boxes], dtype=np.float64)
    box_max_z = np.asarray([simple_building_dimensions(box)["occlusion_position"][2] + simple_building_dimensions(box)["occlusion_scale"][2] * 0.5 for box in boxes], dtype=np.float64)
    box_index = cKDTree(box_centers) if len(box_centers) else None

    access_xy = np.asarray([sample.position_local[:2] for sample in access_samples], dtype=np.float64)
    access_index = cKDTree(access_xy) if len(access_xy) else None

    def tree_clearance_at(local_x: float, local_y: float) -> float:
        if tree_index is None:
            return 100.0
        count = min(32, len(trees))
        distances, indices = tree_index.query((local_x, local_y), k=count, distance_upper_bound=60.0)
        clearance = 60.0
        for distance, tree_index_value in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            if int(tree_index_value) >= len(trees) or not math.isfinite(float(distance)):
                continue
            clearance = min(clearance, float(distance) - tree_radii[int(tree_index_value)])
        return clearance

    def building_clearance_at(local_x: float, local_y: float, ignored_box: int | None = None) -> float:
        if box_index is None:
            return 100.0
        clearance = 100.0
        for box_index_value in box_index.query_ball_point((local_x, local_y), 120.0):
            if ignored_box is not None and int(box_index_value) == ignored_box:
                continue
            box = boxes[int(box_index_value)]
            dx = max(box.minimum[0] - local_x, 0.0, local_x - box.maximum[0])
            dy = max(box.minimum[1] - local_y, 0.0, local_y - box.maximum[1])
            clearance = min(clearance, math.hypot(dx, dy))
        return clearance

    candidates: list[dict[str, Any]] = []
    deduplicated_roads: dict[tuple[int, int], AccessSurfaceSample] = {}
    for sample in road_samples:
        local_x, local_y, _ = sample.position_local
        key = (round(local_x / 10.0), round(local_y / 10.0))
        previous = deduplicated_roads.get(key)
        if previous is None or (sample.surface_area_m2, sample.triangle_count) > (previous.surface_area_m2, previous.triangle_count):
            deduplicated_roads[key] = sample

    for sample in deduplicated_roads.values():
        local_x, local_y, road_z = sample.position_local
        east = local_x + anchor[0]
        north = local_y + anchor[1]
        if not (xmin <= east <= xmax and ymin <= north <= ymax):
            continue
        ground = sampler.at(east, north)
        distance = math.hypot(east - center[0], north - center[1])
        if not math.isfinite(ground) or distance < 100.0:
            continue
        tree_clearance = tree_clearance_at(local_x, local_y)
        building_clearance = building_clearance_at(local_x, local_y)
        road_lift = max(0.0, road_z - ground)
        placement_type = "bridge" if road_lift >= 2.5 else "roadside"
        if tree_clearance < 8.0 or building_clearance < 12.0:
            continue
        position_z = ground + road_lift + eye_height
        candidates.append(
            {
                "placement_type": placement_type,
                "access_surface": "mapped_road",
                "access_tile": sample.source_tile,
                "host_building": "",
                "ignored_box": None,
                "local_x": local_x,
                "local_y": local_y,
                "east": east,
                "north": north,
                "position_z": position_z,
                "ground": ground,
                "height_above_ground_m": position_z - ground,
                "distance": distance,
                "angle": math.atan2(north - center[1], east - center[0]),
                "tree_clearance_m": tree_clearance,
                "building_clearance_m": building_clearance,
                "main_road_score": math.log1p(sample.surface_area_m2) + 0.25 * math.log1p(sample.triangle_count),
            }
        )

    for box_number, box in enumerate(boxes):
        dimensions = simple_building_dimensions(box)
        center_x = (box.minimum[0] + box.maximum[0]) * 0.5
        center_y = (box.minimum[1] + box.maximum[1]) * 0.5
        to_fire = np.asarray((target_local[0] - center_x, target_local[1] - center_y), dtype=np.float64)
        distance_to_fire = float(np.linalg.norm(to_fire))
        if distance_to_fire < 100.0:
            continue
        direction = to_fire / distance_to_fire
        half_x = max((box.maximum[0] - box.minimum[0]) * 0.5, 0.5)
        half_y = max((box.maximum[1] - box.minimum[1]) * 0.5, 0.5)
        facade_distance = min(
            half_x / max(abs(float(direction[0])), 1e-6),
            half_y / max(abs(float(direction[1])), 1e-6),
        )
        facade_x = center_x + float(direction[0]) * (facade_distance + 0.6)
        facade_y = center_y + float(direction[1]) * (facade_distance + 0.6)
        east = facade_x + anchor[0]
        north = facade_y + anchor[1]
        ground = sampler.at(east, north)
        upper_floor_z = box.minimum[2] + min(float(dimensions["total_height"]) - 1.2, max(4.5, float(dimensions["total_height"]) * 0.65))
        other_building_clearance = building_clearance_at(facade_x, facade_y, box_number)
        if upper_floor_z - ground >= 3.2 and other_building_clearance >= 5.0:
            candidates.append(
                {
                    "placement_type": "upper_floor",
                    "access_surface": "building_upper_floor_window",
                    "access_tile": box.name,
                    "host_building": box.name,
                    "ignored_box": box_number,
                    "local_x": facade_x,
                    "local_y": facade_y,
                    "east": east,
                    "north": north,
                    "position_z": upper_floor_z,
                    "ground": ground,
                    "height_above_ground_m": upper_floor_z - ground,
                    "distance": distance_to_fire,
                    "angle": math.atan2(north - center[1], east - center[0]),
                    "tree_clearance_m": tree_clearance_at(facade_x, facade_y),
                    "building_clearance_m": other_building_clearance,
                    "main_road_score": 0.0,
                }
            )

        garden_setback = facade_distance + 14.0
        garden_x = center_x + float(direction[0]) * garden_setback
        garden_y = center_y + float(direction[1]) * garden_setback
        garden_east = garden_x + anchor[0]
        garden_north = garden_y + anchor[1]
        if not (xmin <= garden_east <= xmax and ymin <= garden_north <= ymax):
            continue
        if access_index is not None:
            access_distance, _ = access_index.query((garden_x, garden_y), k=1)
            if float(access_distance) > 75.0:
                continue
        garden_ground = sampler.at(garden_east, garden_north)
        garden_tree_clearance = tree_clearance_at(garden_x, garden_y)
        garden_building_clearance = building_clearance_at(garden_x, garden_y, box_number)
        if garden_tree_clearance < 10.0 or garden_building_clearance < 6.0:
            continue
        candidates.append(
            {
                "placement_type": "garden",
                "access_surface": "inferred_open_garden_or_courtyard",
                "access_tile": box.name,
                "host_building": box.name,
                "ignored_box": box_number,
                "local_x": garden_x,
                "local_y": garden_y,
                "east": garden_east,
                "north": garden_north,
                "position_z": garden_ground + eye_height,
                "ground": garden_ground,
                "height_above_ground_m": eye_height,
                "distance": math.hypot(garden_east - center[0], garden_north - center[1]),
                "angle": math.atan2(garden_north - center[1], garden_east - center[0]),
                "tree_clearance_m": garden_tree_clearance,
                "building_clearance_m": garden_building_clearance,
                "main_road_score": 0.0,
            }
        )

    if len(candidates) < 55:
        raise ValueError(f"Only {len(candidates)} plausible human photo sites were found")

    def clear_fire_axis(candidate: dict[str, Any]) -> tuple[bool, float, float]:
        cached = candidate.get("line_of_sight")
        if cached is not None:
            return cached
        camera_xy = np.asarray((candidate["local_x"], candidate["local_y"]), dtype=np.float64)
        fire_xy = np.asarray(target_local[:2], dtype=np.float64)
        delta = fire_xy - camera_xy
        horizontal_distance = float(np.linalg.norm(delta))
        direction = delta / horizontal_distance

        foreground_clearance = min(horizontal_distance, 180.0)
        if tree_index is not None:
            nearby = np.asarray(tree_index.query_ball_point(camera_xy, 180.0), dtype=np.int64)
            if len(nearby):
                offsets = tree_xy[nearby] - camera_xy
                along = offsets @ direction
                lateral = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
                ray_z = candidate["position_z"] + (target_local[2] - candidate["position_z"]) * np.clip(along / horizontal_distance, 0.0, 1.0)
                blocked = (along > 3.0) & (along < 180.0) & (lateral < tree_radii[nearby] + 1.5) & (ray_z < tree_top[nearby] + 1.0) & (ray_z > tree_base[nearby] - 1.0)
                if bool(np.any(blocked)):
                    result = (False, 0.0, float(np.min(along[blocked])))
                    candidate["line_of_sight"] = result
                    return result

        if len(boxes):
            offsets = box_centers - camera_xy
            along = offsets @ direction
            lateral = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
            ray_z = candidate["position_z"] + (target_local[2] - candidate["position_z"]) * np.clip(along / horizontal_distance, 0.0, 1.0)
            blocked = (along > 2.0) & (along < horizontal_distance) & (lateral < box_half_diagonal + 1.0) & (ray_z >= box_min_z - 1.0) & (ray_z <= box_max_z + 1.0)
            ignored_box = candidate.get("ignored_box")
            if ignored_box is not None:
                blocked[int(ignored_box)] = False
            if bool(np.any(blocked)):
                result = (False, 0.0, float(np.min(along[blocked])))
                candidate["line_of_sight"] = result
                return result

        samples = max(16, min(96, int(horizontal_distance / 40.0)))
        minimum_terrain_clearance = math.inf
        for fraction in np.linspace(0.05, 0.95, samples):
            east = candidate["east"] + (target_global[0] - candidate["east"]) * float(fraction)
            north = candidate["north"] + (target_global[1] - candidate["north"]) * float(fraction)
            ray_z = candidate["position_z"] + (target_global[2] - candidate["position_z"]) * float(fraction)
            minimum_terrain_clearance = min(minimum_terrain_clearance, ray_z - sampler.at(east, north))
        result = (minimum_terrain_clearance >= 1.0, minimum_terrain_clearance, foreground_clearance)
        candidate["line_of_sight"] = result
        return result

    placement_cycle = (
        "roadside",
        "upper_floor",
        "roadside",
        "garden",
        "roadside",
        "bridge",
        "roadside",
        "upper_floor",
        "roadside",
        "garden",
        "roadside",
    )
    distance_ratios = (0.07, 0.11, 0.16, 0.22, 0.30)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    specifications = tuple(
        (
            placement_cycle[index % len(placement_cycle)],
            (index * golden_angle) % math.tau,
            distance_ratios[(index // len(placement_cycle)) % len(distance_ratios)],
        )
        for index in range(55)
    )
    fallbacks = {
        "roadside": {"roadside", "bridge"},
        "bridge": {"bridge", "roadside"},
        "upper_floor": {"upper_floor", "garden", "roadside"},
        "garden": {"garden", "upper_floor", "roadside"},
    }
    selected: list[dict[str, Any]] = []
    for desired_type, desired_angle, ratio in specifications:
        preferred_distance = max(300.0, span * ratio)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            if candidate in selected or candidate["placement_type"] not in fallbacks[desired_type]:
                continue
            if selected:
                nearest_selected = min(math.hypot(candidate["east"] - other["east"], candidate["north"] - other["north"]) for other in selected)
                if nearest_selected < 50.0:
                    continue
            else:
                nearest_selected = 1000.0
            angular_delta = abs(math.atan2(math.sin(candidate["angle"] - desired_angle), math.cos(candidate["angle"] - desired_angle)))
            distance_penalty = abs(math.log(max(candidate["distance"], 1.0) / preferred_distance))
            fallback_penalty = 0.0 if candidate["placement_type"] == desired_type else 1.25
            separation_penalty = max(0.0, (300.0 - nearest_selected) / 300.0)
            openness_bonus = min(candidate["tree_clearance_m"], 40.0) / 35.0 + min(candidate["building_clearance_m"], 100.0) / 300.0
            main_road_bonus = candidate["main_road_score"] * 0.08
            score = angular_delta * 1.7 + distance_penalty * 1.2 + fallback_penalty + separation_penalty - openness_bonus - main_road_bonus
            ranked.append((score, candidate))
        ranked.sort(key=lambda value: value[0])
        chosen = None
        for _, candidate in ranked:
            clear, terrain_clearance, foreground_clearance = clear_fire_axis(candidate)
            if clear:
                candidate["terrain_los_clearance_m"] = terrain_clearance
                candidate["foreground_clearance_m"] = foreground_clearance
                chosen = candidate
                break
        if chosen is None:
            raise ValueError(f"No clear wide fire view is available for requested camera type {desired_type}")
        selected.append(chosen)

    plan: list[dict[str, Any]] = []
    for camera_index, candidate in enumerate(selected, start=1):
        is_phone = camera_index % 3 == 0
        capture_device_profile = "smartphone_main_26mm_equivalent" if is_phone else "professional_full_frame_50mm"
        framing_style = "phone_context_frame" if is_phone else "professional_standard_frame"
        focal_length_mm = 26.0 if is_phone else 50.0
        horizontal_aperture_mm = 36.0
        vertical_aperture_mm = 20.25
        eye_height_m = 1.75 if is_phone else 2.15
        position_raise_m = 0.0 if candidate["placement_type"] == "upper_floor" else max(0.0, eye_height_m - float(candidate["height_above_ground_m"]))
        position_z = float(candidate["position_z"]) + position_raise_m
        height_above_ground_m = float(candidate["height_above_ground_m"]) + position_raise_m
        position_global = (candidate["east"], candidate["north"], position_z)
        position_local = (candidate["local_x"], candidate["local_y"], position_z)
        if camera_index % 5 == 0:
            yaw_offset_degrees = 95.0 if camera_index % 10 == 0 else -95.0
            sample_capability = "negative_context"
            expected_fire_in_frame = False
        elif camera_index % 6 == 0:
            yaw_offset_degrees = 8.0
            sample_capability = "positive_fire"
            expected_fire_in_frame = True
        elif camera_index % 9 == 0:
            yaw_offset_degrees = -8.0
            sample_capability = "positive_fire"
            expected_fire_in_frame = True
        else:
            yaw_offset_degrees = 0.0
            sample_capability = "positive_fire"
            expected_fire_in_frame = True
        yaw_offset = math.radians(yaw_offset_degrees)
        fire_dx = target_local[0] - position_local[0]
        fire_dy = target_local[1] - position_local[1]
        aim_local = (
            position_local[0] + fire_dx * math.cos(yaw_offset) - fire_dy * math.sin(yaw_offset),
            position_local[1] + fire_dx * math.sin(yaw_offset) + fire_dy * math.cos(yaw_offset),
            target_local[2],
        )
        distance_ratio = candidate["distance"] / span
        role = "near" if distance_ratio < 0.13 else "mid" if distance_ratio < 0.26 else "far"
        plan.append(
            {
                "camera_id": f"CAM_{camera_index:02d}",
                "role": role,
                "placement_type": candidate["placement_type"],
                "placement_contract": "human_photo_site_with_clear_axis_to_fire",
                "capture_device_profile": capture_device_profile,
                "framing_style": framing_style,
                "access_surface": candidate["access_surface"],
                "access_tile": candidate["access_tile"],
                "host_building": candidate["host_building"],
                "eye_height_m": eye_height_m,
                "height_above_ground_m": height_above_ground_m,
                "ground_elevation_m": candidate["ground"],
                "tree_clearance_m": candidate["tree_clearance_m"],
                "building_clearance_m": candidate["building_clearance_m"],
                "terrain_los_clearance_m": candidate["terrain_los_clearance_m"],
                "foreground_clearance_m": candidate["foreground_clearance_m"],
                "distance_to_target_m": candidate["distance"],
                "target_kind": "fire_ignition_and_visible_front" if expected_fire_in_frame else "negative_context_away_from_fire",
                "sample_capability": sample_capability,
                "expected_fire_in_frame": expected_fire_in_frame,
                "thermal_capture": False,
                "fire_bearing_offset_degrees": yaw_offset_degrees,
                "line_of_sight_verified": True,
                "position_local_m": position_local,
                "target_local_m": aim_local,
                "fire_target_local_m": target_local,
                "position_l93_ngf_ign69_m": position_global,
                "focal_length_mm": focal_length_mm,
                "focal_length_35mm_equivalent_mm": focal_length_mm,
                "horizontal_aperture_mm": horizontal_aperture_mm,
                "vertical_aperture_mm": vertical_aperture_mm,
                "orientation_quat_wxyz": look_at_quaternion(position_local, aim_local),
            }
        )

    aerial_specs = tuple(
        (
            math.tau * index / 6.0 + 0.18,
            (420.0, 500.0, 580.0)[index % 3],
            (180.0, 210.0, 240.0)[index % 3],
        )
        for index in range(6)
    )

    def aerial_terrain_clearance(east: float, north: float, position_z: float) -> float:
        clearance = math.inf
        for fraction in np.linspace(0.05, 0.95, 64):
            sample_east = east + (target_global[0] - east) * float(fraction)
            sample_north = north + (target_global[1] - north) * float(fraction)
            ray_z = position_z + (target_global[2] - position_z) * float(fraction)
            clearance = min(clearance, ray_z - sampler.at(sample_east, sample_north))
        return clearance

    def clear_aerial_position_z(east: float, north: float, initial_position_z: float) -> float:
        required_position_z = initial_position_z
        for fraction in np.linspace(0.05, 0.95, 64):
            sample_east = east + (target_global[0] - east) * float(fraction)
            sample_north = north + (target_global[1] - north) * float(fraction)
            terrain_z = sampler.at(sample_east, sample_north)
            required_for_sample = (terrain_z + 25.0 - target_global[2] * float(fraction)) / (1.0 - float(fraction))
            required_position_z = max(required_position_z, required_for_sample)
        return required_position_z

    for aerial_index, (angle, radius, altitude) in enumerate(aerial_specs, start=56):
        east = min(xmax - 80.0, max(xmin + 80.0, center[0] + math.cos(angle) * radius))
        north = min(ymax - 80.0, max(ymin + 80.0, center[1] + math.sin(angle) * radius))
        ground = sampler.at(east, north)
        position_z = clear_aerial_position_z(east, north, max(ground, center_ground) + altitude)
        position_local = (east - anchor[0], north - anchor[1], position_z)
        terrain_clearance = aerial_terrain_clearance(east, north, position_z)
        if terrain_clearance < 24.9:
            raise ValueError(f"Terrain blocks aerial camera CAM_{aerial_index:02d}")
        plan.append(
            {
                "camera_id": f"CAM_{aerial_index:02d}",
                "role": "aerial",
                "placement_type": "aerial",
                "placement_contract": "aerial_overview_with_clear_fire_target",
                "capture_device_profile": "aerial_rgb_thermal_mapping_camera",
                "framing_style": "aerial_context_frame",
                "access_surface": "airborne_platform",
                "access_tile": "aerial_overview",
                "host_building": "",
                "eye_height_m": 0.0,
                "height_above_ground_m": position_z - ground,
                "ground_elevation_m": ground,
                "tree_clearance_m": 100.0,
                "building_clearance_m": 100.0,
                "terrain_los_clearance_m": terrain_clearance,
                "foreground_clearance_m": 500.0,
                "distance_to_target_m": math.hypot(east - center[0], north - center[1]),
                "target_kind": "fire_ignition_and_visible_front",
                "sample_capability": "positive_fire",
                "expected_fire_in_frame": True,
                "thermal_capture": True,
                "fire_bearing_offset_degrees": 0.0,
                "line_of_sight_verified": True,
                "position_local_m": position_local,
                "target_local_m": target_local,
                "fire_target_local_m": target_local,
                "position_l93_ngf_ign69_m": (east, north, position_z),
                "focal_length_mm": 35.0,
                "focal_length_35mm_equivalent_mm": 35.0,
                "horizontal_aperture_mm": 36.0,
                "vertical_aperture_mm": 20.25,
                "orientation_quat_wxyz": look_at_quaternion(position_local, target_local),
            }
        )

    overhead_ground = sampler.at(center[0], center[1])
    overhead_position = (target_local[0], target_local[1], overhead_ground + 350.0)
    overhead_clearance = aerial_terrain_clearance(center[0], center[1], overhead_position[2])
    plan.append(
        {
            "camera_id": "CAM_62",
            "role": "aerial",
            "placement_type": "aerial",
            "placement_contract": "aerial_overview_with_clear_fire_target",
            "capture_device_profile": "aerial_rgb_thermal_mapping_camera",
            "framing_style": "aerial_overhead_frame",
            "access_surface": "airborne_platform",
            "access_tile": "aerial_overhead",
            "host_building": "",
            "eye_height_m": 0.0,
            "height_above_ground_m": 350.0,
            "ground_elevation_m": overhead_ground,
            "tree_clearance_m": 100.0,
            "building_clearance_m": 100.0,
            "terrain_los_clearance_m": overhead_clearance,
            "foreground_clearance_m": 500.0,
            "distance_to_target_m": 0.0,
            "target_kind": "fire_ignition_and_visible_front",
            "sample_capability": "positive_fire",
            "expected_fire_in_frame": True,
            "thermal_capture": True,
            "fire_bearing_offset_degrees": 0.0,
            "line_of_sight_verified": True,
            "position_local_m": overhead_position,
            "target_local_m": target_local,
            "fire_target_local_m": target_local,
            "position_l93_ngf_ign69_m": (center[0], center[1], overhead_ground + 350.0),
            "focal_length_mm": 32.0,
            "focal_length_35mm_equivalent_mm": 32.0,
            "horizontal_aperture_mm": 36.0,
            "vertical_aperture_mm": 20.25,
            "orientation_quat_wxyz": look_at_quaternion(overhead_position, target_local),
        }
    )
    return plan


def write_fixed_cameras(output: Path, plan: list[dict[str, Any]]) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(usd_header())
        stream.write('over "World"\n{\n    def Scope "Cameras"\n    {\n')
        for camera in plan:
            quat = camera["orientation_quat_wxyz"]
            stream.write(f'        def Camera "{camera["camera_id"]}"\n        {{\n')
            stream.write(f'            custom string fireviewer:camera_id = "{camera["camera_id"]}"\n')
            stream.write(f'            custom string fireviewer:camera_role = "{camera["role"]}"\n')
            stream.write(f'            custom string fireviewer:placement_type = "{camera["placement_type"]}"\n')
            stream.write(f'            custom string fireviewer:placement_contract = "{camera["placement_contract"]}"\n')
            stream.write(f'            custom string fireviewer:capture_device_profile = "{camera["capture_device_profile"]}"\n')
            stream.write(f'            custom string fireviewer:framing_style = "{camera["framing_style"]}"\n')
            stream.write(f'            custom string fireviewer:access_surface = "{camera["access_surface"]}"\n')
            stream.write(f'            custom string fireviewer:access_tile = "{usd_string(camera["access_tile"])}"\n')
            stream.write(f'            custom string fireviewer:host_building = "{usd_string(camera["host_building"])}"\n')
            stream.write(f'            custom string fireviewer:target_kind = "{camera["target_kind"]}"\n')
            stream.write(f'            custom string fireviewer:sample_capability = "{camera["sample_capability"]}"\n')
            stream.write(f"            custom bool fireviewer:expected_fire_in_frame = {str(bool(camera['expected_fire_in_frame'])).lower()}\n")
            stream.write(f"            custom bool fireviewer:thermal_capture = {str(bool(camera['thermal_capture'])).lower()}\n")
            stream.write(f"            custom bool fireviewer:line_of_sight_verified = {str(bool(camera['line_of_sight_verified'])).lower()}\n")
            stream.write(f"            custom float fireviewer:eye_height_m = {f(camera['eye_height_m'])}\n")
            stream.write(f"            custom float fireviewer:height_above_ground_m = {f(camera['height_above_ground_m'])}\n")
            stream.write(f"            custom float fireviewer:ground_elevation_m = {f(camera['ground_elevation_m'])}\n")
            stream.write(f"            custom float fireviewer:tree_clearance_m = {f(camera['tree_clearance_m'])}\n")
            stream.write(f"            custom float fireviewer:building_clearance_m = {f(camera['building_clearance_m'])}\n")
            stream.write(f"            custom float fireviewer:terrain_los_clearance_m = {f(camera['terrain_los_clearance_m'])}\n")
            stream.write(f"            custom float fireviewer:foreground_clearance_m = {f(camera['foreground_clearance_m'])}\n")
            stream.write(f"            custom float fireviewer:distance_to_target_m = {f(camera['distance_to_target_m'])}\n")
            stream.write(f"            custom float fireviewer:fire_bearing_offset_degrees = {f(camera['fire_bearing_offset_degrees'])}\n")
            stream.write(f"            custom float fireviewer:focal_length_35mm_equivalent_mm = {f(camera['focal_length_35mm_equivalent_mm'])}\n")
            stream.write(f"            custom double3 fireviewer:look_at_local_m = {v3(camera['target_local_m'])}\n")
            stream.write(f"            custom double3 fireviewer:fire_target_local_m = {v3(camera['fire_target_local_m'])}\n")
            stream.write(f"            custom double3 fireviewer:position_l93_ngf_ign69_m = {v3(camera['position_l93_ngf_ign69_m'])}\n")
            stream.write(f"            float focalLength = {f(camera['focal_length_mm'])}\n")
            stream.write(f"            float horizontalAperture = {f(camera['horizontal_aperture_mm'])}\n")
            stream.write(f"            float verticalAperture = {f(camera['vertical_aperture_mm'])}\n")
            stream.write(f"            double3 xformOp:translate = {v3(camera['position_local_m'])}\n")
            stream.write(f"            quatf xformOp:orient = ({f(quat[0])}, {f(quat[1])}, {f(quat[2])}, {f(quat[3])})\n")
            stream.write('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n')
            stream.write("        }\n")
        stream.write("    }\n}\n")


def write_dataset_stage(output: Path, *, package_id: str, anchor: tuple[float, float]) -> None:
    stage = (
        "#usda 1.0\n(\n"
        "    defaultPrim = \"World\"\n"
        "    metersPerUnit = 1\n"
        "    upAxis = \"Z\"\n"
        "    startTimeCode = 0\n"
        "    endTimeCode = 1080\n"
        "    timeCodesPerSecond = 1\n"
        "    framesPerSecond = 30\n"
        "    customLayerData = {\n        dictionary renderSettings = {\n            bool \"rtx:flow:enabled\" = 1\n            int \"rtx:flow:maxBlocks\" = 16384\n            bool \"rtx:flow:pathTracingEnabled\" = 1\n            bool \"rtx:flow:rayTracedReflectionsEnabled\" = 1\n            bool \"rtx:flow:rayTracedShadowsEnabled\" = 1\n            bool \"rtx:flow:rayTracedTranslucencyEnabled\" = 1\n        }\n    }\n"
        "    subLayers = [\n        @site/site.usda@,\n        @scenarios/scenario.usda@,\n        @scenarios/flow.usda@,\n        @appearance/appearance.usda@,\n        @cameras/fixed_cameras.usda@\n    ]\n"
        ")\n\n"
        + 'def Xform "World" (kind = "assembly")\n{\n'
        + f'    custom string fireviewer:package_id = "{usd_string(package_id)}"\n'
        + '    custom string fireviewer:dataset_contract = "synthetic_fire_sdg"\n'
        + f"    custom double2 fireviewer:common_anchor_l93_m = ({f(anchor[0])}, {f(anchor[1])})\n"
        + '    def DomeLight "SkyFill"\n    {\n        float intensity = 350\n        asset inputs:texture:file = @assets/environments/farm_field_puresky_4k.hdr@\n        token inputs:texture:format = "latlong"\n        float3 xformOp:rotateXYZ = (0, 0, 28)\n        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]\n    }\n'
        + '    def DistantLight "Sun"\n    {\n        float angle = 0.8\n        color3f color = (1.0, 0.91, 0.78)\n        float intensity = 950\n        float3 xformOp:rotateXYZ = (24, -18, 42)\n        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]\n    }\n'
        + "}\n"
    )
    output.write_text(stage, encoding="utf-8")


def capture_zoom_variants(package_id: str, plan_id: str, camera: dict[str, Any]) -> list[dict[str, Any]]:
    """Return five focal variants while preserving the camera pose and sensor aperture."""

    base_focal_length = float(camera["focal_length_mm"])
    base_equivalent = float(camera["focal_length_35mm_equivalent_mm"])
    if base_focal_length <= 0.0 or base_equivalent <= 0.0:
        raise ValueError("Capture zoom variants require positive focal lengths")
    return [
        {
            "capture_id": f"{plan_id}_zoom_{zoom_index:02d}",
            "zoom_set_id": plan_id,
            "zoom_index": zoom_index,
            "zoom_label": label,
            "zoom_multiplier": multiplier,
            "focal_length_mm": round(base_focal_length * multiplier, 6),
            "focal_length_35mm_equivalent_mm": round(base_equivalent * multiplier, 6),
            "horizontal_aperture_mm": float(camera["horizontal_aperture_mm"]),
            "vertical_aperture_mm": float(camera["vertical_aperture_mm"]),
            "camera_pose_contract": "same_position_orientation_and_target_across_zoom_set",
            "selection_token_sha256": hashlib.sha256(
                f"{package_id}|{plan_id}_zoom_{zoom_index:02d}".encode("utf-8")
            ).hexdigest(),
        }
        for zoom_index, (label, multiplier) in enumerate(CAPTURE_ZOOM_PROFILES, start=1)
    ]


def build_capture_schedule(
    package_id: str,
    states: list[dict[str, Any]],
    cameras: list[dict[str, Any]],
    *,
    incident_days: int = 18,
    states_per_day: int = 10,
    incident_kind: str = "synthetic-incident",
    observation_seconds_per_state: int = 8640,
) -> dict[str, Any]:
    camera_ids = [str(camera["camera_id"]) for camera in cameras]
    if incident_days <= 0 or states_per_day <= 0 or observation_seconds_per_state <= 0:
        raise ValueError("Capture schedule cadence must be positive")
    if len(states) != incident_days * states_per_day or len(camera_ids) != 62:
        raise ValueError(
            "Capture schedule state count must match the incident cadence and a sixty-two-camera pool"
        )
    incident_id = f"{package_id}-{incident_kind}-{incident_days}d"
    camera_by_id = {str(camera["camera_id"]): camera for camera in cameras}
    positive_aerial_ids = [camera_id for camera_id in camera_ids if camera_by_id[camera_id]["sample_capability"] == "positive_fire" and camera_by_id[camera_id]["role"] == "aerial"]
    positive_phone_ids = [camera_id for camera_id in camera_ids if camera_by_id[camera_id]["sample_capability"] == "positive_fire" and camera_by_id[camera_id]["capture_device_profile"] == "smartphone_main_26mm_equivalent"]
    positive_professional_ids = [camera_id for camera_id in camera_ids if camera_by_id[camera_id]["sample_capability"] == "positive_fire" and camera_by_id[camera_id]["capture_device_profile"] == "professional_full_frame_50mm"]
    negative_phone_ids = [camera_id for camera_id in camera_ids if camera_by_id[camera_id]["sample_capability"] == "negative_context" and camera_by_id[camera_id]["capture_device_profile"] == "smartphone_main_26mm_equivalent"]
    negative_professional_ids = [camera_id for camera_id in camera_ids if camera_by_id[camera_id]["sample_capability"] == "negative_context" and camera_by_id[camera_id]["capture_device_profile"] == "professional_full_frame_50mm"]
    quota_pools = (
        (positive_aerial_ids, 2, "positive_aerial"),
        (positive_phone_ids, 6, "positive_phone"),
        (positive_professional_ids, 8, "positive_professional"),
        (negative_phone_ids, 2, "negative_phone"),
        (negative_professional_ids, 2, "negative_professional"),
    )
    for pool, quota, label in quota_pools:
        if len(pool) < quota:
            raise ValueError(f"Camera pool cannot satisfy {label} quota: {len(pool)} < {quota}")
    records = []
    for global_state_index, state in enumerate(states, start=1):
        state_id = str(state["state_id"])
        active_in_scene = bool(state.get("active_in_scene", True))
        day_index = (global_state_index - 1) // states_per_day + 1
        state_in_day = (global_state_index - 1) % states_per_day + 1
        selected = []
        for pool, quota, label in quota_pools:
            ranked = sorted(
                pool,
                key=lambda camera_id, label=label: hashlib.sha256(f"{package_id}|day_{day_index:02d}|{state_id}|{label}|{camera_id}".encode("utf-8")).hexdigest(),
            )
            selected.extend(ranked[:quota])
        selected.sort(key=lambda camera_id: hashlib.sha256(f"{package_id}|{state_id}|order|{camera_id}".encode("utf-8")).hexdigest())
        views = []
        for view_index, camera_id in enumerate(selected, start=1):
            camera = camera_by_id[camera_id]
            base_sample_kind = str(camera["sample_capability"])
            sample_kind = base_sample_kind if active_in_scene else "negative_context_out_of_scene"
            plan_id = f"day_{day_index:02d}_{state_id}_view_{view_index:02d}_{camera_id}"
            expected_modalities = ["rgb", "semantic_ids", "fire_front_mask", "fire_perimeter_mask", "depth_m", "pointcloud", "camera_params", "geolocation", "abstention"]
            if camera["thermal_capture"]:
                expected_modalities.extend(("synthetic_thermal_kelvin", "synthetic_thermal_16bit", "thermal_metadata"))
            views.append(
                {
                    "capture_id": plan_id,
                    "plan_id": plan_id,
                    "zoom_count": len(CAPTURE_ZOOM_PROFILES),
                    "zoom_variants": capture_zoom_variants(package_id, plan_id, camera),
                    "camera_pose_contract": "same_position_orientation_and_target_across_zoom_set",
                    "incident_id": incident_id,
                    "simulation_id": state["simulation_id"],
                    "day_index": day_index,
                    "state_in_day": state_in_day,
                    "global_state_index": global_state_index,
                    "state_id": state_id,
                    "fire_state_elapsed_s": float(state["elapsed_s"]),
                    "burned_area_m2": float(state["burned_area_m2"]),
                    "active_front_length_m": float(state["active_front_length_m"]),
                    "mean_front_spread_rate_m_s": float(state["mean_front_spread_rate_m_s"]),
                    "burned_tree_count": int(state["burned_tree_count"]),
                    "camera_id": camera_id,
                    "camera_path": f"/World/Cameras/{camera_id}",
                    "camera_role": camera["role"],
                    "placement_type": camera["placement_type"],
                    "placement_contract": camera["placement_contract"],
                    "capture_device_profile": camera["capture_device_profile"],
                    "framing_style": camera["framing_style"],
                    "access_surface": camera["access_surface"],
                    "access_tile": camera["access_tile"],
                    "host_building": camera["host_building"],
                    "sample_kind": sample_kind,
                    "negative_reason": (
                        "daily_active_zone_outside_validated_scene"
                        if not active_in_scene
                        else "camera_rotated_away_from_fire_for_context_control"
                        if sample_kind == "negative_context"
                        else None
                    ),
                    "expected_fire_visible": bool(camera["expected_fire_in_frame"]) and active_in_scene,
                    "thermal_expected": bool(camera["thermal_capture"]),
                    "thermal_contract": "synthetic_16bit_kelvin_non_radiometric" if camera["thermal_capture"] else None,
                    "expected_modalities": expected_modalities,
                    "line_of_sight_verified": bool(camera["line_of_sight_verified"]),
                    "position_local_m": list(camera["position_local_m"]),
                    "position_l93_ngf_ign69_m": list(camera["position_l93_ngf_ign69_m"]),
                    "look_at_local_m": list(camera["target_local_m"]),
                    "fire_target_local_m": list(camera["fire_target_local_m"]),
                    "height_above_ground_m": float(camera["height_above_ground_m"]),
                    "ground_elevation_m": float(camera["ground_elevation_m"]),
                    "terrain_los_clearance_m": float(camera["terrain_los_clearance_m"]),
                    "foreground_clearance_m": float(camera["foreground_clearance_m"]),
                    "distance_to_target_m": float(camera["distance_to_target_m"]),
                    "fire_bearing_offset_degrees": float(camera["fire_bearing_offset_degrees"]),
                    "focal_length_mm": float(camera["focal_length_mm"]),
                    "focal_length_35mm_equivalent_mm": float(camera["focal_length_35mm_equivalent_mm"]),
                    "horizontal_aperture_mm": float(camera["horizontal_aperture_mm"]),
                    "vertical_aperture_mm": float(camera["vertical_aperture_mm"]),
                    "observation_elapsed_s": int(
                        (global_state_index - 1) * observation_seconds_per_state
                    ),
                    "selection_token_sha256": hashlib.sha256(f"{package_id}|{plan_id}".encode("utf-8")).hexdigest(),
                }
            )
        records.append(
            {
                "day_index": day_index,
                "state_in_day": state_in_day,
                "global_state_index": global_state_index,
                "state_id": state_id,
                "camera_ids": selected,
                "view_count": len(selected),
                "zoom_count_per_view": len(CAPTURE_ZOOM_PROFILES),
                "capture_count": len(selected) * len(CAPTURE_ZOOM_PROFILES),
                "positive_view_count": 16 if active_in_scene else 0,
                "negative_view_count": 4 if active_in_scene else 20,
                "active_in_scene": active_in_scene,
                "professional_view_count": 10,
                "phone_view_count": 8,
                "aerial_view_count": 2,
                "views": views,
                "selection_sha256": hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest(),
            }
        )
    return {
        "schema": "fireviewer.random-camera-zoom-schedule.v2",
        "incident_id": incident_id,
        "incident_days": incident_days,
        "states_per_day": states_per_day,
        "views_per_state": 20,
        "zooms_per_view": len(CAPTURE_ZOOM_PROFILES),
        "captures_per_state": 20 * len(CAPTURE_ZOOM_PROFILES),
        "positive_views_per_state": 16,
        "negative_views_per_state": 4,
        "camera_pool_count": len(camera_ids),
        "human_camera_count": 55,
        "aerial_camera_count": 7,
        "profile_mix_per_state": {"professional": 10, "phone": 8, "aerial": 2},
        "selection": "deterministic_sha256_rank_without_replacement_by_day_state_and_sample_kind",
        "expected_viewpoint_plans": len(states) * 20,
        "expected_capture_cases": len(states) * 20 * len(CAPTURE_ZOOM_PROFILES),
        "expected_positive_cases": sum(
            int(state["positive_view_count"]) * len(CAPTURE_ZOOM_PROFILES) for state in records
        ),
        "expected_negative_cases": sum(
            int(state["negative_view_count"]) * len(CAPTURE_ZOOM_PROFILES) for state in records
        ),
        "states": records,
    }


def write_qa_files(output: Path, *, package_id: str, states: list[dict[str, Any]], cameras: list[dict[str, Any]], capture_schedule: dict[str, Any], propagation: dict[str, Any], terrain_count: int, building_ref_count: int, tree_count: int, occlusion_count: int) -> None:
    qa = {
        "schema": "fireviewer.omniverse-dataset-qa.v1",
        "package_id": package_id,
        "automated": {
            "expected_terrain_payloads": terrain_count,
            "expected_building_references": building_ref_count,
            "expected_tree_instances": tree_count,
            "expected_occlusion_proxies": occlusion_count,
            "expected_fire_states": len(states),
            "expected_fixed_cameras": len(cameras),
            "expected_capture_cases": int(capture_schedule["expected_capture_cases"]),
            "expected_viewpoint_plans": int(capture_schedule["expected_viewpoint_plans"]),
            "zooms_per_view": int(capture_schedule["zooms_per_view"]),
            "states_per_day": int(capture_schedule["states_per_day"]),
            "random_views_per_state": int(capture_schedule["views_per_state"]),
            "checks": ["relative_usd_dependencies", "payload_targets", "tree_count_matches_catalog", "continuous_propagation_sidecar", "monotonic_burned_area", "monotonic_state_time", "flow_emitters", "state_schema", "camera_schema", "capture_schedule_schema", "five_zoom_same_pose_capture_contract", "positive_negative_quota", "aerial_thermal_contract", "writer_modalities", "abstention_records"],
        },
        "propagation": propagation,
        "human_review": {
            "status": "required_before_dataset_acceptance",
            "checks": ["real_terrain_and_feature_alignment", "camera_framing", "negative_view_fire_absence", "aerial_rgb_thermal_alignment", "near_field_flow", "mid_distance_smoke", "distant_fire_readability", "burned_decor_readability", "mask_alignment", "depth_and_pointcloud_plausibility", "abstention_precision"],
        },
    }
    (output / "qa" / "acceptance.json").write_text(canonical_json(qa), encoding="utf-8")
    (output / "qa" / "HUMAN_REVIEW.md").write_text(
        "# FireViewer Omniverse dataset — human acceptance\n\n"
        "This review is performed only after the automated validator and an Isaac/RTX render run pass.\n\n"
        "- [ ] Terrain, buildings, vegetation and routes align with the uploaded map.\n"
        "- [ ] Fifty-five human cameras use plausible photo sites, a controlled mix of professional and phone framing, and a verified clear fire axis.\n"
        "- [ ] Seven additional aerial cameras cover the fire globally without replacing the human views.\n"
        "- [ ] Every selected viewpoint produces five focal variants with identical position, orientation, target and simulation time.\n"
        "- [ ] Every aerial capture includes aligned synthetic 16-bit thermal output, explicitly marked non-radiometric.\n"
        "- [ ] Each state contains sixteen positive fire views and four intentional negative context views.\n"
        "- [ ] Flow combustion and smoke are driven by the active front; no static fire placeholders are visible.\n"
        "- [ ] Burned vegetation and ground degradation are readable and do not alter source geometry.\n"
        "- [ ] RGB, masks, depth, point cloud and geolocation agree on sampled frames.\n"
        "- [ ] Abstention records are emitted whenever a front or perimeter is not visible.\n",
        encoding="utf-8",
    )


def sanitize_source_manifest(source_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "fireviewer.omniverse-published-map.v1",
        "package_id": source_manifest["package_id"],
        "coordinate_convention": source_manifest["coordinate_convention"],
        "common_anchor_l93_metres": source_manifest["common_anchor_l93_metres"],
        "bounds_l93_metres": source_manifest["bounds_l93_metres"],
        "terrain_tile_count": source_manifest["terrain_tile_count"],
        "feature_tile_count": source_manifest["feature_tile_count"],
        "terrain": [
            {
                "terrain_tile_id": record["terrain_tile_id"],
                "mesh_grid": record["mesh_grid"],
                "vertex_count": record["vertex_count"],
                "quad_count": record["quad_count"],
                "bounds_l93_metres": record["bounds_l93_metres"],
                "source_elevation_sha256": record["source_elevation_sha256"],
            }
            for record in source_manifest["terrain"]
        ],
        "orthophoto": source_manifest["orthophoto"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "source_catalog_sha256": source_manifest["source_catalog_sha256"],
        "entry_stage": source_manifest["entry_stage"],
        "provenance_paths": "intentionally_removed_from_standalone_dataset_package",
    }


def tree_asset_records(assets: list[TreeAsset]) -> list[dict[str, Any]]:
    return [
        {
            "asset_id": asset.asset_id,
            "package_path": asset.package_path,
            "source_name": asset.source_name,
            "sha256": asset.sha256,
            "byte_count": asset.byte_count,
            "source_up_axis": asset.up_axis,
            "source_default_prim": asset.default_prim,
        }
        for asset in assets
    ]


def write_vegetation_placement_manifest(
    output: Path,
    *,
    trees: list[TreeInstance],
    assets: list[TreeAsset],
    source_index: dict[str, Any],
    source_index_sha256: str,
) -> dict[str, Any]:
    counts = Counter(tree.source_tile for tree in trees)
    tile_records = []
    for tile in source_index["tiles"]:
        tile_id = str(tile["tile_id"])
        bounds = [float(value) for value in tile["bounds_l93_m"]]
        area_km2 = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / 1_000_000.0
        count = counts.get(tile_id, 0)
        if count != int(tile["accepted_crown_count"]):
            raise ValueError(f"Loaded vegetation count differs from rebuild index for {tile_id}")
        tile_records.append({
            "tile_id": tile_id,
            "bounds_l93_m": bounds,
            "tree_instances": count,
            "instances_per_km2": count / area_km2 if area_km2 > 0 else None,
        })
    record = {
        "schema": "fireviewer.mnt-mns-vegetation-placement.v1",
        "source_rebuild": {
            "schema": source_index["schema"],
            "index_sha256": source_index_sha256,
            "source_manifest_sha256": source_index["source_manifest"]["sha256"],
            "source_hashes_verified": source_index["source_manifest"]["all_source_hashes_verified"],
        },
        "method": source_index["method"],
        "counts": {
            "tree_instances": len(trees),
            "prototype_assets": len(assets),
            "feature_tiles": len(tile_records),
            "feature_tiles_with_trees": sum(record["tree_instances"] > 0 for record in tile_records),
            "minimum_tree_height_m": min(tree.scale[2] for tree in trees),
            "maximum_tree_height_m": max(tree.scale[2] for tree in trees),
            "maximum_instances_on_one_km2_tile": max((record["tree_instances"] for record in tile_records), default=0),
        },
        "assets": tree_asset_records(assets),
        "tiles": tile_records,
    }
    output.write_text(canonical_json(record), encoding="utf-8")
    return record


def build_package(
    source: Path,
    base_root: Path,
    output_root: Path,
    tree_assets: list[TreeAsset],
    vegetation_index_path: Path,
) -> dict[str, Any]:
    source_manifest = json.loads((source / "package-manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((source / "catalog.json").read_text(encoding="utf-8"))
    package_id = str(source_manifest["package_id"])
    converted = base_root / package_id
    base_path = converted / "manifest.json"
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing converted USD package for {package_id}: {base_path}")
    base_manifest = json.loads(base_path.read_text(encoding="utf-8"))
    if base_manifest.get("package_id") != package_id:
        raise ValueError(f"Converted USD package identity mismatch for {package_id}")
    output = output_root / package_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(converted, output / "source-usd")
    copy_tree_assets(output, tree_assets)
    (output / "source-usd" / "manifest.json").write_text(canonical_json(sanitize_source_manifest(base_manifest)), encoding="utf-8")
    for directory in ("site/payloads", "scenarios/states", "appearance", "cameras", "qa", "runtime", "assets/environments"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    environment_source = Path(__file__).resolve().parent / "assets" / "environments"
    environment_asset = environment_source / "farm_field_puresky_4k.hdr"
    environment_provenance_path = environment_source / "farm_field_puresky_4k.provenance.json"
    if not environment_asset.is_file() or not environment_provenance_path.is_file():
        raise FileNotFoundError("The CC0 Farm Field (Pure Sky) HDRI and its provenance record are required")
    shutil.copy2(environment_asset, output / "assets/environments/farm_field_puresky_4k.hdr")
    shutil.copy2(environment_provenance_path, output / "assets/environments/farm_field_puresky_4k.provenance.json")
    environment_provenance = json.loads(environment_provenance_path.read_text(encoding="utf-8"))
    anchor = tuple(float(value) for value in catalog["spatial_contract"]["common_anchor_l93_metres"])
    bounds = tuple(float(value) for value in catalog["bounds_l93_metres"])
    sampler = HeightSampler(source, catalog)
    try:
        trees, vegetation_source_index = load_mnt_mns_trees(
            vegetation_index_path,
            package_id=package_id,
            anchor=anchor,
        )
        _, occlusion_boxes, access_samples = collect_derived_geometry(
            source,
            catalog,
            anchor,
            include_source_trees=False,
        )
        print(json.dumps({"phase": "derive_site_geometry", "building_instances": len(occlusion_boxes), "mapped_access_samples": len(access_samples)}, sort_keys=True), flush=True)
        write_terrain_payload(output / "site/payloads/terrain.payload.usda", base_manifest)
        write_simple_buildings_payload(
            output / "site/payloads/buildings.payload.usda",
            occlusion_boxes,
            ground_at_local=lambda local_x, local_y: sampler.at(local_x + anchor[0], local_y + anchor[1]),
        )
        route_refs = geometry_references(base_manifest, {"road", "path", "itinerary"}, root="Route")
        write_reference_payload(
            output / "site/payloads/routes.payload.usda",
            root_name="RoutesPayload",
            semantic="route",
            references=route_refs,
            custom_properties=['custom string fireviewer:representation = "source_flat_surface_meshes_draped_on_mnt"'],
        )
        vegetation_context_refs = geometry_references(base_manifest, {"vegetation_edges"}, root="VegetationContext")
        write_reference_payload(output / "site/payloads/vegetation_context.payload.usda", root_name="VegetationContextPayload", semantic="vegetation", references=vegetation_context_refs)
        write_vegetation_payload(output / "site/payloads/vegetation.payload.usda", trees, tree_assets)
        print(json.dumps({"phase": "write_vegetation_payload", "tree_instances": len(trees), "prototype_assets": len(tree_assets)}, sort_keys=True), flush=True)
        vegetation_placement = write_vegetation_placement_manifest(
            output / "site/vegetation-placement.json",
            trees=trees,
            assets=tree_assets,
            source_index=vegetation_source_index,
            source_index_sha256=sha256_file(vegetation_index_path),
        )
        write_occlusion_payload(output / "site/payloads/occlusion.payload.usda", occlusion_boxes)
        propagation_result = simulate_fire_spread(
            package_id=package_id,
            trees=trees,
            anchor_l93_m=anchor,
            source_bounds_l93_m=bounds,
            height_at=sampler.at,
            state_count=180,
        )
        print(json.dumps({"phase": "simulate_fire_spread", "states": len(propagation_result.states)}, sort_keys=True), flush=True)
        propagation = write_propagation_sidecars(output, propagation_result)
        center = propagation_result.ignition_l93_m
        write_site_layer(output / "site/site.usda", package_id=package_id, anchor=anchor, bounds=bounds)
        tree_by_id = {tree.tree_id: tree for tree in trees}
        states = [
            write_fire_state(output / "scenarios" / record_path, result=propagation_result, state=spread_state, state_count=len(propagation_result.states), anchor=anchor, trees=trees, tree_by_id=tree_by_id, terrain_height_at=sampler.at)
            for spread_state, record_path in ((spread_state, Path("states") / f"state_{spread_state.state_index:03d}.usda") for spread_state in propagation_result.states)
        ]
        print(json.dumps({"phase": "write_fire_states", "states": len(states)}, sort_keys=True), flush=True)
        write_scenario_layer(output / "scenarios/scenario.usda", states, propagation)
        write_flow_layer(output / "scenarios/flow.usda", result=propagation_result, anchor=anchor)
        write_appearance_layer(output / "appearance/appearance.usda")
        cameras = fixed_camera_plan(center, bounds, sampler, anchor, access_samples, trees, occlusion_boxes)
        candidates = [tuple(camera["position_local_m"]) for camera in cameras]
        write_camera_candidate_payload(output / "site/payloads/camera_candidates.payload.usda", candidates)
        write_fixed_cameras(output / "cameras/fixed_cameras.usda", cameras)
        write_dataset_stage(output / "dataset.usda", package_id=package_id, anchor=anchor)
        capture_schedule = build_capture_schedule(package_id, states, cameras)
        capture_schedule_path = output / "runtime/capture-schedule.json"
        capture_schedule_path.write_text(canonical_json(capture_schedule), encoding="utf-8")
        runtime_source = Path(__file__).resolve().parent
        writer_path = output / "runtime/fireviewer_replicator_writer.py"
        runner_path = output / "runtime/run_fireviewer_replicator_dataset.py"
        shutil.copy2(runtime_source / "fireviewer_replicator_writer.py", writer_path)
        shutil.copy2(runtime_source / "run_fireviewer_replicator_dataset.py", runner_path)
        write_qa_files(output, package_id=package_id, states=states, cameras=cameras, capture_schedule=capture_schedule, propagation=propagation, terrain_count=len(base_manifest["terrain"]), building_ref_count=len(occlusion_boxes), tree_count=len(trees), occlusion_count=len(occlusion_boxes))
        runtime = {
            "schema": "fireviewer.omniverse-runtime-contract.v1",
            "entry_stage": "dataset.usda",
            "required_extensions": ["omni.replicator.core", "omni.flowusd"],
            "writer_module": {"path": "fireviewer_replicator_writer.py", "sha256": sha256_file(writer_path)},
            "runner_module": {"path": "run_fireviewer_replicator_dataset.py", "sha256": sha256_file(runner_path)},
            "capture": {
                "incident_days": 18,
                "states_per_day": 10,
                "views_per_state": 20,
                "zooms_per_view": int(capture_schedule["zooms_per_view"]),
                "captures_per_state": int(capture_schedule["captures_per_state"]),
                "states": [record["state_id"] for record in states],
                "camera_pool": [record["camera_id"] for record in cameras],
                "human_cameras": 55,
                "aerial_cameras": 7,
                "positive_views_per_state": 16,
                "negative_views_per_state": 4,
                "profile_mix_per_state": {"professional": 10, "phone": 8, "aerial": 2},
                "schedule_path": "capture-schedule.json",
                "schedule_sha256": sha256_file(capture_schedule_path),
                "expected_capture_cases": int(capture_schedule["expected_capture_cases"]),
                "expected_viewpoint_plans": int(capture_schedule["expected_viewpoint_plans"]),
            },
            "modalities": ["rgb", "aerial_synthetic_thermal_16bit", "semantic_masks", "pointcloud", "fire_front_visible", "fire_perimeter", "depth", "geolocation", "abstention"],
            "flow_contract": {
                "close": "/World/Appearance/FlowClose",
                "mid_distance": "/World/Appearance/SmokeMidDistance",
                "source_points": "/World/FireScenario/Truth3D/SmokeSources",
                "flow_visual": "/World/FireScenario/FlowVisual",
                "runtime": "actual_omni.flowusd_combustion_and_smoke",
                "animation": "single_persistent_time_sampled_flow_volume_256_patch_mesh_front_plus_48_hotspots_plus_aligned_smoke_mesh",
                "combustion": "meter_scaled_omni_flowusd_combustion_plus_aligned_buoyant_smoke_without_direct_burn",
            },
        }
        (output / "runtime/runtime-contract.json").write_text(canonical_json(runtime), encoding="utf-8")
    finally:
        sampler.close()
    manifest = {
        "schema": "fireviewer.omniverse-dataset-package.v1",
        "package_id": package_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"package_manifest_sha256": sha256_file(source / "package-manifest.json"), "catalog_sha256": sha256_file(source / "catalog.json"), "source_type": "published_site_map"},
        "coordinate_convention": "EPSG:2154 local anchor, Z-up metres, NGF-IGN69 elevations",
        "entry_stage": "dataset.usda",
        "environment": {
            "lighting": "cc0_latlong_hdri_plus_soft_distant_sun",
            "hdri_path": "assets/environments/farm_field_puresky_4k.hdr",
            "hdri_sha256": sha256_file(output / "assets/environments/farm_field_puresky_4k.hdr"),
            "provenance_path": "assets/environments/farm_field_puresky_4k.provenance.json",
            "source_url": environment_provenance["source_page"],
            "license": environment_provenance["license"],
        },
        "site": {
            "terrain_payloads": len(base_manifest["terrain"]),
            "ground_texture": base_manifest["orthophoto"],
            "building_payload_references": 0,
            "building_instances": len(occlusion_boxes),
            "building_representation": "merged_source_footprint_mnt_grounded_oriented_box_gabled_roof_windows_and_doors",
            "building_vertical_alignment": "each_visual_module_sampled_at_its_center_on_mnt",
            "route_payload_references": len(route_refs),
            "route_representation": "source_flat_surface_meshes_draped_on_mnt",
            "vegetation_point_instances": len(trees),
            "vegetation_prototype_assets": len(tree_assets),
            "vegetation_lod_policy": "none_all_detected_instances_resident",
            "vegetation_placement": "site/vegetation-placement.json",
            "occlusion_proxies": len(occlusion_boxes),
            "camera_candidates": len(candidates),
        },
        "vegetation": vegetation_placement,
        "scenario": {
            "truth_scope": "synthetic_physically_driven_fire_spread_on_real_uploaded_map_not_incident_reconstruction",
            "incident_id": capture_schedule["incident_id"],
            "incident_days": 18,
            "states_per_day": 10,
            "propagation": propagation,
            "state_count": len(states),
            "states": states,
        },
        "appearance": ["actual_flow_combustion_close", "actual_flow_smoke_mid_distance", "simplified_distant_fire", "decor_degradation"],
        "cameras": {
            "fixed_count": len(cameras),
            "human_count": sum(camera["role"] != "aerial" for camera in cameras),
            "aerial_count": sum(camera["role"] == "aerial" for camera in cameras),
            "negative_context_count": sum(camera["sample_capability"] == "negative_context" for camera in cameras),
            "thermal_count": sum(bool(camera["thermal_capture"]) for camera in cameras),
            "placement_counts": {
                placement_type: sum(camera["placement_type"] == placement_type for camera in cameras)
                for placement_type in sorted({str(camera["placement_type"]) for camera in cameras})
            },
            "profile_counts": {
                profile: sum(camera["capture_device_profile"] == profile for camera in cameras)
                for profile in sorted({str(camera["capture_device_profile"]) for camera in cameras})
            },
            "plan": cameras,
        },
        "dataset": {
            "modalities": runtime["modalities"],
            "incident_days": 18,
            "states_per_day": 10,
            "views_per_state": 20,
            "zooms_per_view": int(capture_schedule["zooms_per_view"]),
            "captures_per_state": int(capture_schedule["captures_per_state"]),
            "positive_views_per_state": 16,
            "negative_views_per_state": 4,
            "expected_capture_cases": runtime["capture"]["expected_capture_cases"],
            "expected_viewpoint_plans": capture_schedule["expected_viewpoint_plans"],
            "expected_positive_cases": capture_schedule["expected_positive_cases"],
            "expected_negative_cases": capture_schedule["expected_negative_cases"],
            "capture_schedule": {
                "path": "runtime/capture-schedule.json",
                "sha256": runtime["capture"]["schedule_sha256"],
            },
        },
        "qa": {"automated": "qa/acceptance.json", "human_review": "qa/HUMAN_REVIEW.md", "render_acceptance": "pending_isaac_rtx_flow_run"},
    }
    (output / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", type=Path, action="append", required=True, help="A root containing published site map packages")
    parser.add_argument("--converted-root", type=Path, required=True, help="Output root from convert_published_maps_to_usd.py")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tree-assets-manifest", type=Path, required=True, help="Kit-inspected manifest for exactly six USDZ tree assets")
    parser.add_argument("--vegetation-index", type=Path, required=True, help="Checksum-verified MNT/MNS crown rebuild index")
    args = parser.parse_args()
    packages = discover_packages([root.resolve() for root in args.site_root])
    if not packages:
        raise SystemExit("No complete published map packages found")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tree_assets = load_tree_assets_manifest(args.tree_assets_manifest.resolve())
    results = [
        build_package(
            source,
            args.converted_root.resolve(),
            output_root,
            tree_assets,
            args.vegetation_index.resolve(),
        )
        for source in packages
    ]
    index = {
        "schema": "fireviewer.omniverse-dataset-index.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packages": [{"package_id": result["package_id"], "stage": f"{result['package_id']}/dataset.usda", "states": result["scenario"]["state_count"], "cameras": result["cameras"]["fixed_count"]} for result in results],
    }
    (output_root / "index.json").write_text(canonical_json(index), encoding="utf-8")
    print(canonical_json(index), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
