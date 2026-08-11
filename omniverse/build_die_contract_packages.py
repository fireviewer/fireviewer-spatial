"""Build separated FireViewer map, perimeter and reproducible Die packages.

The builder is additive. It never modifies the validated R4 base scene or the
validated V1 retrospective reproduction. The accepted fire scenario and camera
rig remain immutable, while the accepted Flow/sky capture corrections and the
current multimodal runtime are packaged into new revisions. The map upload and
progressive perimeter layers remain independent packages.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Iterator
import zipfile

import rasterio
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate

from build_die_retrospective_simulation import prepare_daily_geometries
from fireviewer_accepted_visual_profiles import (
    PROFILE_CONTRACT_RELATIVE,
    PROFILE_LAYER_RELATIVE,
    write_accepted_visual_profile_artifacts,
)


MAP_PACKAGE_ID = "fireviewer-die-pontaix-map-r6"
MAP_REVISION = 6
PERIMETER_PACKAGE_ID = "die-2026-progressive-perimeters-r2"
PERIMETER_REVISION = 2
CASE_PACKAGE_ID = "fireviewer-die-pontaix-map-r6-die-retrospective-v3"
BUNDLE_ID = "fireviewer-die-2026-reproduction-download-r2"
CASE_ID = "die-2026-retrospective-replay-v1"
SOURCE_SIMULATION_ID = "e77606f8b4982b6644a5047f"
PLAYBACK_SECONDS_PER_DAY = 60.0
PERIMETER_TERRAIN_OFFSET_M = 1.0
PERIMETER_TRACE_WIDTH_M = 8.0
PERIMETER_FILL_OPACITY = 0.24
USD_ASSET_RE = re.compile(r"@([^@]+)@")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
MNT_TILE_RE = re.compile(r"_(\d{4})_(\d{4})_MNT_")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8", newline="\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def convert_usda_to_usdc(source: Path, target: Path, *, usd_python: Path) -> None:
    """Convert an authored ASCII layer to a compact Crate layer with USD itself."""
    target.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "from pxr import Sdf; import sys; "
        "layer = Sdf.Layer.FindOrOpen(sys.argv[1]); "
        "assert layer is not None, sys.argv[1]; "
        "assert layer.Export(sys.argv[2], args={'format': 'usdc'}), sys.argv[2]"
    )
    completed = subprocess.run(
        [str(usd_python), "-c", command, str(source), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not target.is_file():
        detail = (completed.stderr or completed.stdout or "USD conversion produced no output").strip()
        raise RuntimeError(f"Could not convert {source} to {target}: {detail}")


def copy_tree(source: Path, target: Path, *, ignore: Any = None) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to merge into an existing directory: {target}")
    shutil.copytree(source, target, copy_function=shutil.copy2, ignore=ignore)


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def tree_inventory(root: Path, *, excluded_names: set[str] | None = None) -> list[dict[str, Any]]:
    excluded_names = excluded_names or set()
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = relative_posix(path, root)
        if relative in excluded_names:
            continue
        records.append(
            {
                "path": relative,
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def inventory_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def write_inventory(root: Path, *, excluded_names: set[str]) -> tuple[Path, str, int]:
    records = tree_inventory(root, excluded_names=excluded_names)
    payload = {
        "schema": "fireviewer.sha256-file-inventory.v1",
        "root": ".",
        "file_count": len(records),
        "total_byte_count": sum(int(item["byte_count"]) for item in records),
        "records_sha256": inventory_digest(records),
        "files": records,
    }
    path = root / "dependency-inventory.json"
    write_json(path, payload)
    return path, sha256_file(path), len(records)


def usd_asset_issues(root: Path) -> list[str]:
    issues: list[str] = []
    resolved_root = root.resolve()
    for layer in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in {".usd", ".usda", ".usdc"}
    ):
        if layer.suffix.lower() == ".usdc":
            continue
        text = layer.read_text(encoding="utf-8", errors="strict")
        for raw_asset in USD_ASSET_RE.findall(text):
            asset = raw_asset.strip()
            if not asset:
                continue
            if "://" in asset:
                issues.append(f"remote USD dependency in {relative_posix(layer, root)}: {asset}")
                continue
            if WINDOWS_ABSOLUTE_RE.match(asset) or asset.startswith(("/", "\\")):
                issues.append(f"absolute USD dependency in {relative_posix(layer, root)}: {asset}")
                continue
            resolved = (layer.parent / asset.replace("/", os.sep)).resolve()
            try:
                resolved.relative_to(resolved_root)
            except ValueError:
                issues.append(f"escaping USD dependency in {relative_posix(layer, root)}: {asset}")
                continue
            if not resolved.exists():
                issues.append(f"missing USD dependency in {relative_posix(layer, root)}: {asset}")
    return issues


def assert_no_forbidden_map_content(map_package: Path) -> None:
    forbidden = {
        "CameraCandidates": "camera candidate payload",
        "fixed_cameras.usda": "camera rig",
        "perimeters.usda": "perimeter layer",
        "scenarios/scenario.usda": "scenario layer",
        "scenarios/flow.usda": "Flow layer",
        "rtx:flow:enabled": "Flow render setting",
        "FireScenario": "fire scenario prim",
    }
    forbidden_directories = {"cameras", "perimeters", "scenarios", "appearance", "runtime"}
    for path in map_package.rglob("*"):
        if path.is_file() and forbidden_directories.intersection(path.relative_to(map_package).parts):
            raise ValueError(f"Pure map contains a forbidden production directory: {path}")
    text_layers = [
        item for item in map_package.rglob("*") if item.is_file() and item.suffix.lower() == ".usda"
    ]
    for path in text_layers:
        text = path.read_text(encoding="utf-8", errors="strict")
        for token, label in forbidden.items():
            if token in text:
                raise ValueError(f"Pure map boundary violation ({label}) in {path}")


def _usd_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_values(values: Iterable[str], *, indent: str, line_size: int = 8) -> str:
    items = list(values)
    if not items:
        return ""
    lines = []
    for index in range(0, len(items), line_size):
        lines.append(indent + ", ".join(items[index : index + line_size]))
    return ",\n".join(lines)


def _usd_points(points: Iterable[tuple[float, float, float]], *, indent: str = "            ") -> str:
    return _format_values(
        (f"({x:.3f}, {y:.3f}, {z:.3f})" for x, y, z in points),
        indent=indent,
        line_size=4,
    )


def _usd_ints(values: Iterable[int], *, indent: str = "            ") -> str:
    return _format_values((str(value) for value in values), indent=indent, line_size=16)


@dataclass
class MntCacheSampler:
    root: Path

    def __post_init__(self) -> None:
        self._paths: dict[tuple[int, int], Path] = {}
        self._datasets: dict[tuple[int, int], Any] = {}
        self._cache: dict[tuple[float, float], float] = {}
        for path in sorted(self.root.glob("*.tif")):
            match = MNT_TILE_RE.search(path.name)
            if match:
                self._paths[(int(match.group(1)), int(match.group(2)))] = path
        if not self._paths:
            raise FileNotFoundError(f"No IGN MNT tiles found in {self.root}")

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.close()
        self._datasets.clear()
        self._cache.clear()

    def __enter__(self) -> "MntCacheSampler":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _tile_key(self, east: float, north: float) -> tuple[int, int]:
        direct = (math.floor(east / 1000.0), math.floor(north / 1000.0))
        if direct in self._paths:
            return direct
        return min(
            self._paths,
            key=lambda key: (east - (key[0] * 1000.0 + 500.0)) ** 2
            + (north - (key[1] * 1000.0 + 500.0)) ** 2,
        )

    def at(self, east: float, north: float) -> float:
        cache_key = (round(float(east), 2), round(float(north), 2))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        tile_key = self._tile_key(east, north)
        dataset = self._datasets.get(tile_key)
        if dataset is None:
            dataset = rasterio.open(self._paths[tile_key])
            self._datasets[tile_key] = dataset
        value = float(next(dataset.sample([(east, north)]))[0])
        if not math.isfinite(value):
            raise ValueError(f"MNT sampling failed at ({east}, {north})")
        self._cache[cache_key] = value
        return value


def _polygon_parts(geometry: BaseGeometry) -> Iterator[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def _color(index: int, total: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    amount = 0.0 if total <= 1 else (index - 1) / (total - 1)
    fill_start = (1.0, 0.64, 0.03)
    fill_end = (0.34, 0.015, 0.008)
    trace_start = (1.0, 0.95, 0.20)
    trace_end = (1.0, 0.08, 0.015)
    fill = tuple(start + (end - start) * amount for start, end in zip(fill_start, fill_end))
    trace = tuple(start + (end - start) * amount for start, end in zip(trace_start, trace_end))
    return fill, trace


def perimeter_mesh_data(
    geometry: BaseGeometry,
    *,
    sampler: MntCacheSampler,
    anchor: tuple[float, float],
    terrain_offset_m: float,
) -> tuple[
    list[tuple[float, float, float]],
    list[int],
    list[int],
    list[tuple[float, float, float]],
    list[int],
]:
    mesh_points: list[tuple[float, float, float]] = []
    mesh_point_indices: dict[tuple[float, float, float], int] = {}
    face_counts: list[int] = []
    face_indices: list[int] = []
    trace_points: list[tuple[float, float, float]] = []
    curve_counts: list[int] = []
    simplified = geometry.simplify(1.0, preserve_topology=True)
    for polygon in _polygon_parts(simplified):
        if polygon.is_empty or polygon.area <= 1.0:
            continue
        for candidate in triangulate(polygon):
            if candidate.area <= 0.01 or not polygon.covers(candidate.representative_point()):
                continue
            coordinates = list(candidate.exterior.coords)[:3]
            for east, north in coordinates:
                point = (
                    float(east - anchor[0]),
                    float(north - anchor[1]),
                    float(sampler.at(east, north) + terrain_offset_m),
                )
                key = tuple(round(value, 6) for value in point)
                point_index = mesh_point_indices.get(key)
                if point_index is None:
                    point_index = len(mesh_points)
                    mesh_point_indices[key] = point_index
                    mesh_points.append(point)
                face_indices.append(point_index)
            face_counts.append(3)
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = list(ring.coords)
            if len(coordinates) < 2:
                continue
            curve_counts.append(len(coordinates))
            for east, north in coordinates:
                trace_points.append(
                    (
                        float(east - anchor[0]),
                        float(north - anchor[1]),
                        float(sampler.at(east, north) + terrain_offset_m + 0.08),
                    )
                )
    if not mesh_points or not trace_points:
        raise ValueError("Perimeter geometry produced no visible fill or trace")
    return mesh_points, face_counts, face_indices, trace_points, curve_counts


def _extent(points: list[tuple[float, float, float]], *, padding: float = 0.0) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not points:
        raise ValueError("Cannot compute an extent for an empty point array")
    minimum = tuple(min(point[axis] for point in points) - padding for axis in range(3))
    maximum = tuple(max(point[axis] for point in points) + padding for axis in range(3))
    return minimum, maximum


def _face_normals(
    points: list[tuple[float, float, float]], indices: list[int]
) -> list[tuple[float, float, float]]:
    normals: list[tuple[float, float, float]] = []
    for offset in range(0, len(indices), 3):
        p0 = points[indices[offset]]
        p1 = points[indices[offset + 1]]
        p2 = points[indices[offset + 2]]
        edge1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        edge2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        normal = (
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        )
        length = math.sqrt(sum(value * value for value in normal))
        if length <= 1e-12:
            normals.append((0.0, 0.0, 1.0))
        else:
            normalized = tuple(value / length for value in normal)
            normals.append(tuple(-value for value in normalized) if normalized[2] < 0 else normalized)
    return normals


def write_map_site(path: Path, *, source_package_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

over "World"
{
    def Xform "FireViewerSite" (kind = "assembly")
    {
        custom string fireviewer:package_id = "__PACKAGE_ID__"
        custom string fireviewer:derived_from_package_id = "__SOURCE_PACKAGE_ID__"
        custom string fireviewer:content_boundary = "terrain_vegetation_buildings_roads_only"
        custom string fireviewer:crs = "EPSG:2154"
        custom string fireviewer:vertical_datum = "NGF-IGN69"
        custom double2 fireviewer:common_anchor_l93_m = (884000.0, 6408000.0)
        custom double4 fireviewer:bounds_l93_m = (876000.0, 6403000.0, 892000.0, 6413000.0)
        def Xform "Terrain" (
            kind = "component"
            prepend references = @payloads/terrain.payload.usda@</TerrainPayload>
            prepend apiSchemas = ["SemanticsLabelsAPI:class"]
        )
        {
            token[] semantics:labels:class = ["terrain"]
        }
        def Xform "Buildings" (
            kind = "component"
            prepend references = @payloads/buildings.payload.usda@</BuildingsPayload>
            prepend apiSchemas = ["SemanticsLabelsAPI:class"]
        )
        {
            token[] semantics:labels:class = ["building"]
        }
        def Xform "Routes" (
            kind = "component"
            prepend references = @payloads/routes.payload.usda@</RoutesPayload>
            prepend apiSchemas = ["SemanticsLabelsAPI:class"]
        )
        {
            token[] semantics:labels:class = ["route"]
        }
        def Xform "VegetationContext" (
            kind = "component"
            prepend references = @payloads/vegetation_context.payload.usda@</VegetationContextPayload>
            prepend apiSchemas = ["SemanticsLabelsAPI:class"]
        )
        {
            token[] semantics:labels:class = ["vegetation"]
        }
        def Xform "Vegetation" (
            kind = "component"
            prepend references = @payloads/vegetation.payload.usda@</VegetationPayload>
            prepend apiSchemas = ["SemanticsLabelsAPI:class"]
        )
        {
            token[] semantics:labels:class = ["vegetation"]
        }
    }
}
""".replace("__PACKAGE_ID__", MAP_PACKAGE_ID).replace(
            "__SOURCE_PACKAGE_ID__", _usd_string(source_package_id)
        ),
        encoding="utf-8",
        newline="\n",
    )


def write_map_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        @site/site.usda@
    ]
)

def Xform "World" (kind = "assembly")
{
    custom string fireviewer:package_id = "__PACKAGE_ID__"
    custom int fireviewer:revision = __REVISION__
    custom string fireviewer:contract = "pure_map_upload_no_perimeter_no_cameras_no_simulation"
    custom double2 fireviewer:common_anchor_l93_m = (884000.0, 6408000.0)
    def DomeLight "SkyFill"
    {
        float intensity = 350
        asset inputs:texture:file = @assets/environments/farm_field_puresky_4k.hdr@
        token inputs:texture:format = "latlong"
        float3 xformOp:rotateXYZ = (0, 0, 28)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
    def DistantLight "Sun"
    {
        float angle = 0.8
        color3f color = (1.0, 0.91, 0.78)
        float intensity = 950
        float3 xformOp:rotateXYZ = (24, -18, 42)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
}
""".replace("__PACKAGE_ID__", MAP_PACKAGE_ID).replace("__REVISION__", str(MAP_REVISION)),
        encoding="utf-8",
        newline="\n",
    )


def write_perimeter_state(
    path: Path,
    *,
    state_index: int,
    state_count: int,
    record: dict[str, Any],
    geometry: BaseGeometry,
    sampler: MntCacheSampler,
    anchor: tuple[float, float],
) -> dict[str, Any]:
    mesh_points, face_counts, face_indices, trace_points, curve_counts = perimeter_mesh_data(
        geometry,
        sampler=sampler,
        anchor=anchor,
        terrain_offset_m=PERIMETER_TERRAIN_OFFSET_M,
    )
    mesh_extent = _extent(mesh_points)
    trace_extent = _extent(trace_points, padding=PERIMETER_TRACE_WIDTH_M * 0.5)
    normals = _face_normals(mesh_points, face_indices)
    fill, trace = _color(state_index, state_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    layer = f'''#usda 1.0
(
    defaultPrim = "PerimeterState"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "PerimeterState" (kind = "component")
{{
    custom string fireviewer:state_id = "perimeter_{state_index:03d}"
    custom string fireviewer:layer_revision_id = "{_usd_string(str(record['layer_revision_id']))}"
    custom string fireviewer:local_date = "{_usd_string(str(record['local_date']))}"
    custom string fireviewer:valid_at = "{_usd_string(str(record['valid_at']))}"
    custom string fireviewer:geometry_role = "cumulative_fire_perimeter"
    custom string fireviewer:geometry_origin = "post_incident_derived_not_operational_truth"
    custom string fireviewer:horizontal_crs = "EPSG:2154"
    custom float fireviewer:terrain_offset_m = {PERIMETER_TERRAIN_OFFSET_M:.3f}
    def Mesh "Fill" (
        prepend apiSchemas = ["SemanticsLabelsAPI:class"]
    )
    {{
        token[] semantics:labels:class = ["fire_perimeter_fill"]
        uniform bool doubleSided = true
        uniform token subdivisionScheme = "none"
        point3f[] points = [
{_usd_points(mesh_points)}
        ]
        float3[] extent = [
            ({mesh_extent[0][0]:.3f}, {mesh_extent[0][1]:.3f}, {mesh_extent[0][2]:.3f}),
            ({mesh_extent[1][0]:.3f}, {mesh_extent[1][1]:.3f}, {mesh_extent[1][2]:.3f})
        ]
        int[] faceVertexCounts = [
{_usd_ints(face_counts)}
        ]
        int[] faceVertexIndices = [
{_usd_ints(face_indices)}
        ]
        normal3f[] normals = [
{_usd_points(normals)}
        ] (
            interpolation = "uniform"
        )
        color3f[] primvars:displayColor = [({fill[0]:.6f}, {fill[1]:.6f}, {fill[2]:.6f})] (
            interpolation = "constant"
        )
        float[] primvars:displayOpacity = [{PERIMETER_FILL_OPACITY:.6f}] (
            interpolation = "constant"
        )
    }}
    def BasisCurves "Trace" (
        prepend apiSchemas = ["SemanticsLabelsAPI:class"]
    )
    {{
        token[] semantics:labels:class = ["fire_perimeter_trace"]
        uniform token type = "linear"
        uniform token wrap = "nonperiodic"
        int[] curveVertexCounts = [
{_usd_ints(curve_counts)}
        ]
        point3f[] points = [
{_usd_points(trace_points)}
        ]
        float3[] extent = [
            ({trace_extent[0][0]:.3f}, {trace_extent[0][1]:.3f}, {trace_extent[0][2]:.3f}),
            ({trace_extent[1][0]:.3f}, {trace_extent[1][1]:.3f}, {trace_extent[1][2]:.3f})
        ]
        float[] widths = [{PERIMETER_TRACE_WIDTH_M:.3f}] (
            interpolation = "constant"
        )
        color3f[] primvars:displayColor = [({trace[0]:.6f}, {trace[1]:.6f}, {trace[2]:.6f})] (
            interpolation = "constant"
        )
    }}
}}
'''
    path.write_text(layer, encoding="utf-8", newline="\n")
    return {
        "mesh_vertex_count": len(mesh_points),
        "triangle_count": len(face_counts),
        "trace_vertex_count": len(trace_points),
        "trace_ring_count": len(curve_counts),
        "fill_color": [round(value, 6) for value in fill],
        "trace_color": [round(value, 6) for value in trace],
    }


def write_perimeter_timeline(path: Path, *, states: list[dict[str, Any]]) -> None:
    children: list[str] = []
    for index, state in enumerate(states, start=1):
        start = float((index - 1) * PLAYBACK_SECONDS_PER_DAY)
        end = float(index * PLAYBACK_SECONDS_PER_DAY)
        samples: list[tuple[float, str]] = []
        if index > 1:
            samples.append((0.0, "invisible"))
        samples.append((start, "inherited"))
        if index < len(states):
            samples.append((end, "invisible"))
        sample_text = ",\n".join(
            f'                {time:.3f}: "{visibility}"' for time, visibility in samples
        )
        children.append(
            f'''        def Xform "State_{index:03d}" (
            prepend references = @states/perimeter_{index:03d}.usdc@</PerimeterState>
        )
        {{
            token visibility.timeSamples = {{
{sample_text}
            }}
        }}'''
        )
    end_time = float(len(states) * PLAYBACK_SECONDS_PER_DAY)
    layer = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = {end_time:.3f}
    timeCodesPerSecond = 1
    framesPerSecond = 30
)

def Xform "World" (kind = "assembly")
{{
    def Xform "IncidentPerimeters" (kind = "assembly")
    {{
        custom string fireviewer:layer_package_id = "{PERIMETER_PACKAGE_ID}"
        custom int fireviewer:revision = {PERIMETER_REVISION}
        custom string fireviewer:composition = "separate_progressive_visualization_layer"
        custom string fireviewer:timeline = "one_day_equals_60_seconds_single_visible_state"
        custom int fireviewer:state_count = {len(states)}
{chr(10).join(children)}
        def Scope "Legend"
        {{
            custom string fireviewer:palette = "chronological_yellow_orange_to_dark_red"
            custom string fireviewer:first_date = "{_usd_string(str(states[0]['local_date']))}"
            custom string fireviewer:last_date = "{_usd_string(str(states[-1]['local_date']))}"
            custom float fireviewer:trace_width_m = {PERIMETER_TRACE_WIDTH_M:.3f}
            custom float fireviewer:fill_opacity = {PERIMETER_FILL_OPACITY:.3f}
        }}
    }}
}}
'''
    path.write_text(layer, encoding="utf-8", newline="\n")


def write_review_stage(path: Path, *, map_relative: str, perimeter_relative: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 1260
    timeCodesPerSecond = 1
    framesPerSecond = 30
    subLayers = [
        @{map_relative}@,
        @{perimeter_relative}@
    ]
)

over "World"
{{
    custom string fireviewer:review_stage = "map_plus_separate_progressive_perimeter_layer"
    custom bool fireviewer:is_map_upload = 0
    custom bool fireviewer:simulation_enabled = 0
}}
''',
        encoding="utf-8",
        newline="\n",
    )


def write_reproduction_stage(path: Path) -> None:
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 1260
    timeCodesPerSecond = 1
    framesPerSecond = 30
    customLayerData = {{
        dictionary renderSettings = {{
            bool "rtx:flow:enabled" = 1
            int "rtx:flow:maxBlocks" = 16384
            bool "rtx:flow:pathTracingEnabled" = 1
            bool "rtx:flow:rayTracedReflectionsEnabled" = 1
            bool "rtx:flow:rayTracedShadowsEnabled" = 1
            bool "rtx:flow:rayTracedTranslucencyEnabled" = 1
        }}
    }}
    subLayers = [
        @appearance/accepted_capture_profiles.usda@,
        @map/map.usda@,
        @perimeters/perimeters.usda@,
        @scenarios/scenario.usda@,
        @scenarios/flow.usda@,
        @appearance/appearance.usda@,
        @cameras/fixed_cameras.usda@
    ]
)

over "World"
{{
    custom string fireviewer:package_id = "{CASE_PACKAGE_ID}"
    custom string fireviewer:case_id = "{CASE_ID}"
    custom string fireviewer:source_validated_reproduction = "fireviewer-die-pontaix-r1-v4-die-retrospective-v1"
    custom string fireviewer:simulation_id = "{SOURCE_SIMULATION_ID}"
    custom string fireviewer:map_package_id = "{MAP_PACKAGE_ID}"
    custom string fireviewer:perimeter_layer_package_id = "{PERIMETER_PACKAGE_ID}"
    custom bool fireviewer:publication_allowed = 0
}}
''',
        encoding="utf-8",
        newline="\n",
    )


def write_python_lock(path: Path) -> None:
    packages = {}
    for name in ("numpy", "rasterio", "shapely", "pyproj", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(
        path,
        {
            "schema": "fireviewer.python-runtime-lock.v1",
            "python": sys.version.split()[0],
            "packages": packages,
        },
    )


def source_lock(path: Path, *, kind: str, source_id: str, package_root: Path) -> dict[str, Any]:
    return {
        "kind": kind,
        "source_id": source_id,
        "path": relative_posix(path, package_root),
        "sha256": sha256_file(path),
        "verified": True,
    }


def build_map_package(
    *,
    base_package: Path,
    vegetation_package: Path,
    ign_source_manifest: Path,
    output: Path,
    map_template: dict[str, Any],
    original_locks: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    print(canonical_json({"phase": "copy_pure_map_dependencies", "output": str(output)}).strip(), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    copy_tree(base_package / "source-usd", output / "source-usd")
    copy_tree(base_package / "assets" / "trees", output / "assets" / "trees")
    copy_tree(base_package / "assets" / "environments", output / "assets" / "environments")
    selected_payloads = (
        "terrain.payload.usda",
        "buildings.payload.usda",
        "routes.payload.usda",
        "vegetation_context.payload.usda",
        "vegetation.payload.usda",
    )
    for name in selected_payloads:
        copy_file(base_package / "site" / "payloads" / name, output / "site" / "payloads" / name)
    copy_file(base_package / "site" / "vegetation-placement.json", output / "site" / "vegetation-placement.json")
    write_map_site(output / "site" / "site.usda", source_package_id=base_package.name)
    write_map_stage(output / "map.usda")

    copy_file(ign_source_manifest, output / "source" / "ign_sources.v1.json")
    vegetation_index = load_json(vegetation_package / "index.json")
    original_vegetation_hash = sha256_file(vegetation_package / "index.json")
    sanitized_index = copy.deepcopy(vegetation_index)
    sanitized_index["source_manifest"]["path"] = "../ign_sources.v1.json"
    write_json(output / "source" / "vegetation" / "index.json", sanitized_index)
    copy_tree(vegetation_package / "tiles", output / "source" / "vegetation" / "tiles")
    write_json(
        output / "provenance" / "source-artifact-locks.json",
        {
            "schema": "fireviewer.source-artifact-locks.v1",
            "base_scene": original_locks["base_scene"],
            "validated_reproduction": original_locks["validated_reproduction"],
            "original_vegetation_index_sha256": original_vegetation_hash,
            "ign_source_manifest_sha256": sha256_file(ign_source_manifest),
        },
    )
    assert_no_forbidden_map_content(output)
    issues = usd_asset_issues(output)
    if issues:
        raise ValueError("Pure map USD dependency validation failed: " + issues[0])

    inventory_path, inventory_sha256, file_count = write_inventory(
        output,
        excluded_names={"manifest.json", "dependency-inventory.json", "contracts/map-contract.json"},
    )
    base_manifest = load_json(base_package / "manifest.json")
    map_manifest = {
        "schema": "fireviewer.omniverse-pure-map-package.v1",
        "package_id": MAP_PACKAGE_ID,
        "revision": MAP_REVISION,
        "status": "candidate_pending_die_visual_acceptance",
        "entry_stage": "map.usda",
        "entry_stage_sha256": sha256_file(output / "map.usda"),
        "derived_from": {
            "package_id": str(base_manifest["package_id"]),
            "dataset_stage_sha256": original_locks["base_scene"]["dataset.usda"]["sha256"],
            "manifest_sha256": original_locks["base_scene"]["manifest.json"]["sha256"],
        },
        "content": ["terrain", "orthophoto", "vegetation", "buildings", "roads", "environment"],
        "composition_policy": {
            "site_elements": "eager_references_loaded_on_normal_stage_open",
            "optional_payloads": False,
        },
        "excluded": [
            "camera_candidates",
            "camera_rig",
            "perimeters",
            "timeline",
            "fire_truth",
            "fire_visuals",
            "flow",
            "tree_destruction",
            "capture",
        ],
        "spatial_reference": {
            "horizontal_crs": "EPSG:2154",
            "vertical_datum": "NGF-IGN69",
            "common_anchor_l93_m": [884000.0, 6408000.0],
            "bounds_l93_m": [876000.0, 6403000.0, 892000.0, 6413000.0],
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "scene_counts": {
            "tree_instances": int(base_manifest["vegetation"]["counts"]["tree_instances"]),
            "tree_prototypes": 6,
            "buildings": int(base_manifest["site"]["building_instances"]),
        },
        "dependency_inventory": {
            "path": relative_posix(inventory_path, output),
            "sha256": inventory_sha256,
            "file_count": file_count,
        },
        "automatic_publication": False,
    }
    write_json(output / "manifest.json", map_manifest)

    contract = copy.deepcopy(map_template)
    contract["package"].update(
        {
            "package_id": MAP_PACKAGE_ID,
            "revision": MAP_REVISION,
            "entry_stage": "map.usda",
            "entry_stage_sha256": sha256_file(output / "map.usda"),
            "manifest_sha256": sha256_file(output / "manifest.json"),
        }
    )
    contract["source_locks"] = [
        source_lock(output / "source" / "ign_sources.v1.json", kind="mnt", source_id="ign-lidar-hd-mnt-0m50-die", package_root=output),
        source_lock(output / "source" / "ign_sources.v1.json", kind="mns", source_id="ign-lidar-hd-mns-0m50-die", package_root=output),
        source_lock(output / "source-usd" / "textures" / "ign-bd-ortho-real-ground.jpg", kind="orthophoto", source_id="ign-bd-ortho-die-real-ground", package_root=output),
        source_lock(output / "source" / "vegetation" / "index.json", kind="vegetation", source_id="die-mnt-mns-vegetation-index", package_root=output),
        source_lock(output / "source-usd" / "source" / "catalog.json", kind="buildings", source_id="published-site-map-buildings-die", package_root=output),
        source_lock(output / "source-usd" / "source" / "package-manifest.json", kind="roads", source_id="published-site-map-roads-die", package_root=output),
        source_lock(output / "assets" / "environments" / "farm_field_puresky_4k.hdr", kind="environment", source_id="farm-field-puresky-4k", package_root=output),
    ]
    contract["quality_gates"].update(
        {
            "simulation_free_stage": "passed",
            "usd_validation": "pending",
            "upload_content_boundary": "passed",
            "kit_runtime_open": "pending",
            "human_visual_review": "pending",
        }
    )
    write_json(output / "contracts" / "map-contract.json", contract)
    return contract, output / "contracts" / "map-contract.json"


def build_perimeter_package(
    *,
    retrospective_path: Path,
    mnt_cache: Path,
    map_package: Path,
    map_contract: dict[str, Any],
    map_contract_path: Path,
    output: Path,
    perimeter_template: dict[str, Any],
    usd_python: Path,
) -> tuple[dict[str, Any], Path]:
    print(canonical_json({"phase": "build_progressive_perimeters", "output": str(output)}).strip(), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    copy_file(retrospective_path, output / "source" / "die-2026-v1.json")
    source = load_json(retrospective_path)
    source_manifest = load_json(map_package / "source-usd" / "manifest.json")
    anchor = tuple(float(value) for value in source_manifest["common_anchor_l93_metres"])
    bounds = tuple(float(value) for value in source_manifest["bounds_l93_metres"])
    daily = prepare_daily_geometries(source, site_bounds=bounds, scene_coverage=box(*bounds))
    state_records: list[dict[str, Any]] = []
    geometry_receipts: list[dict[str, Any]] = []
    last_global_index: int | None = None
    with MntCacheSampler(mnt_cache) as sampler:
        for index, (record, daily_geometry) in enumerate(zip(source["activity_zones"], daily), start=1):
            geometry_record = record["geometry_geojson"]
            if geometry_record.get("global_footprint_geojson") is not None:
                last_global_index = index - 1
            if last_global_index is None:
                raise ValueError(f"No cumulative perimeter source available for state {index}")
            clipped = daily_geometry.source_footprint_l93.intersection(box(*bounds))
            if clipped.is_empty:
                raise ValueError(f"Cumulative perimeter does not intersect the map on {record['local_date']}")
            authoring_path = output / "authoring" / f"perimeter_{index:03d}.usda"
            state_path = output / "states" / f"perimeter_{index:03d}.usdc"
            geometry_receipt = write_perimeter_state(
                authoring_path,
                state_index=index,
                state_count=len(daily),
                record=record,
                geometry=clipped,
                sampler=sampler,
                anchor=(anchor[0], anchor[1]),
            )
            convert_usda_to_usdc(authoring_path, state_path, usd_python=usd_python)
            source_selector = (
                f"#/activity_zones/{last_global_index}/geometry_geojson/global_footprint_geojson"
            )
            state_records.append(
                {
                    "append_order": index,
                    "state_id": f"perimeter_{index:03d}",
                    "layer_revision_id": str(record["layer_revision_id"]),
                    "local_date": str(record["local_date"]),
                    "valid_at": str(record["valid_at"]),
                    "source_selector": source_selector,
                    "geometry_type": "MultiPolygon",
                    "layer_path": f"states/perimeter_{index:03d}.usdc",
                    "layer_sha256": sha256_file(state_path),
                }
            )
            geometry_receipts.append(
                {
                    "state_id": f"perimeter_{index:03d}",
                    "local_date": str(record["local_date"]),
                    "source_selector": source_selector,
                    "authoring_layer_path": f"authoring/perimeter_{index:03d}.usda",
                    "authoring_layer_sha256": sha256_file(authoring_path),
                    "runtime_layer_path": f"states/perimeter_{index:03d}.usdc",
                    "runtime_layer_sha256": sha256_file(state_path),
                    "source_area_ha": round(float(daily_geometry.source_footprint_l93.area / 10000.0), 3),
                    "clipped_area_ha": round(float(clipped.area / 10000.0), 3),
                    **geometry_receipt,
                }
            )
            print(canonical_json({"phase": "perimeter_state", "complete": index, "total": len(daily)}).strip(), flush=True)
    write_perimeter_timeline(output / "perimeters.usda", states=state_records)
    issues = usd_asset_issues(output)
    if issues:
        raise ValueError("Perimeter USD dependency validation failed: " + issues[0])
    write_json(output / "qa" / "geometry-receipts.json", {"states": geometry_receipts})
    inventory_path, inventory_sha256, file_count = write_inventory(
        output,
        excluded_names={"manifest.json", "dependency-inventory.json", "contracts/perimeter-contract.json"},
    )
    manifest = {
        "schema": "fireviewer.omniverse-progressive-perimeter-package.v1",
        "layer_package_id": PERIMETER_PACKAGE_ID,
        "revision": PERIMETER_REVISION,
        "status": "candidate_pending_die_visual_acceptance",
        "entry_layer": "perimeters.usda",
        "entry_layer_sha256": sha256_file(output / "perimeters.usda"),
        "base_map": {
            "package_id": MAP_PACKAGE_ID,
            "revision": MAP_REVISION,
            "contract_sha256": sha256_file(map_contract_path),
        },
        "source": {
            "path": "source/die-2026-v1.json",
            "sha256": sha256_file(output / "source" / "die-2026-v1.json"),
            "truth_scope": "post_incident_derived_not_operational_truth",
        },
        "timeline": {
            "state_count": len(state_records),
            "first_local_date": state_records[0]["local_date"],
            "last_local_date": state_records[-1]["local_date"],
            "seconds_per_day": PLAYBACK_SECONDS_PER_DAY,
            "selection": "one_visible_state_per_daily_interval",
        },
        "rendering": {
            "trace_width_m": PERIMETER_TRACE_WIDTH_M,
            "fill_opacity": PERIMETER_FILL_OPACITY,
            "terrain_offset_m": PERIMETER_TERRAIN_OFFSET_M,
            "palette": "chronological_yellow_orange_to_dark_red",
        },
        "states": state_records,
        "dependency_inventory": {
            "path": relative_posix(inventory_path, output),
            "sha256": inventory_sha256,
            "file_count": file_count,
        },
        "automatic_map_mutation": False,
        "automatic_publication": False,
    }
    write_json(output / "manifest.json", manifest)

    contract = copy.deepcopy(perimeter_template)
    contract["layer_package"].update(
        {
            "layer_package_id": PERIMETER_PACKAGE_ID,
            "revision": PERIMETER_REVISION,
            "entry_layer": "perimeters.usda",
            "entry_layer_sha256": sha256_file(output / "perimeters.usda"),
            "manifest_sha256": sha256_file(output / "manifest.json"),
        }
    )
    contract["base_map"].update(
        {
            "contract_record": "map-contract.json",
            "contract_record_sha256": sha256_file(map_contract_path),
            "package_id": MAP_PACKAGE_ID,
            "revision": MAP_REVISION,
        }
    )
    contract["progression"]["source_record"] = "source/die-2026-v1.json"
    contract["progression"]["source_sha256"] = sha256_file(retrospective_path)
    contract["progression"]["state_records"] = state_records
    contract["progression"]["state_count"] = len(state_records)
    contract["quality_gates"].update(
        {
            "base_map_active": "pending",
            "source_integrity": "passed",
            "geometry_validation": "passed",
            "crs_projection": "passed",
            "progression_order": "passed",
            "usd_validation": "pending",
            "trace_and_color_render": "pending",
            "human_visual_review": "pending",
        }
    )
    write_json(output / "contracts" / "perimeter-contract.json", contract)
    return contract, output / "contracts" / "perimeter-contract.json"


def copy_validated_simulation_layers(validated_reproduction: Path, output: Path) -> None:
    copy_tree(validated_reproduction / "scenarios", output / "scenarios")
    copy_tree(
        validated_reproduction / "runtime",
        output / "runtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    copy_tree(validated_reproduction / "qa", output / "qa" / "validated-v1-source")
    runtime_source = Path(__file__).resolve().parent
    for name in (
        "run_fireviewer_replicator_dataset.py",
        "fireviewer_replicator_writer.py",
        "fireviewer_capture_storage.py",
    ):
        copy_file(runtime_source / name, output / "runtime" / name)


def align_runtime_contract(output: Path, *, visual_profiles: dict[str, Any]) -> dict[str, Any]:
    path = output / "runtime" / "runtime-contract.json"
    runtime = load_json(path)
    runtime_source = Path(__file__).resolve().parent
    runtime["runner_module"] = {
        "path": "run_fireviewer_replicator_dataset.py",
        "sha256": sha256_file(runtime_source / "run_fireviewer_replicator_dataset.py"),
    }
    runtime["writer_module"] = {
        "path": "fireviewer_replicator_writer.py",
        "sha256": sha256_file(runtime_source / "fireviewer_replicator_writer.py"),
    }
    runtime["storage_module"] = {
        "path": "fireviewer_capture_storage.py",
        "sha256": sha256_file(runtime_source / "fireviewer_capture_storage.py"),
    }
    runtime["accepted_visual_profiles"] = {
        "contract": "accepted-visual-profiles.json",
        "contract_sha256": sha256_file(output / PROFILE_CONTRACT_RELATIVE),
        "persistent_layer": PROFILE_LAYER_RELATIVE,
        "persistent_layer_sha256": visual_profiles["persistent_application"]["layer_sha256"],
        "flow_profile_id": visual_profiles["flow_profile"]["profile_id"],
        "flow_profile_sha256": visual_profiles["flow_profile"]["profile_sha256"],
        "sky_profile_id": visual_profiles["sky_profile"]["profile_id"],
        "sky_profile_sha256": visual_profiles["sky_profile"]["profile_sha256"],
    }
    runtime.setdefault("capture", {})["enabled"] = False
    runtime["capture"]["validation_status"] = (
        "disabled_in_download_package_until_controlled_pilot"
    )
    runtime.setdefault("simulation_playback", {})["capture_on_first_launch"] = False
    write_json(path, runtime)
    return runtime


def build_reproduction_bundle(
    *,
    base_package: Path,
    validated_reproduction: Path,
    retrospective_path: Path,
    map_package: Path,
    perimeter_package: Path,
    map_contract: dict[str, Any],
    map_contract_path: Path,
    perimeter_contract: dict[str, Any],
    perimeter_contract_path: Path,
    case_template: dict[str, Any],
    output: Path,
    original_locks: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    print(canonical_json({"phase": "assemble_reproduction_bundle", "output": str(output)}).strip(), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    copy_tree(map_package, output / "map")
    copy_tree(perimeter_package, output / "perimeters")
    copy_validated_simulation_layers(validated_reproduction, output)
    copy_file(base_package / "appearance" / "appearance.usda", output / "appearance" / "appearance.usda")
    copy_file(base_package / "cameras" / "fixed_cameras.usda", output / "cameras" / "fixed_cameras.usda")
    copy_file(retrospective_path, output / "source" / "die-2026-v1.json")
    visual_profiles = write_accepted_visual_profile_artifacts(output)
    runtime_contract = align_runtime_contract(output, visual_profiles=visual_profiles)
    write_python_lock(output / "runtime" / "python-lock.json")
    write_reproduction_stage(output / "dataset.usda")
    write_json(output / "provenance" / "original-artifact-locks.json", original_locks)

    copied_simulation_inventory = tree_inventory(output / "scenarios")
    original_simulation_inventory = tree_inventory(validated_reproduction / "scenarios")
    if copied_simulation_inventory != original_simulation_inventory:
        raise ValueError("The validated V1 scenario or Flow layers changed during the V2 copy")
    for relative in ("appearance/appearance.usda", "cameras/fixed_cameras.usda"):
        copied = output / relative
        source = base_package / relative
        if sha256_file(copied) != sha256_file(source):
            raise ValueError(f"Byte-preserving copy failed for {relative}")

    issues = usd_asset_issues(output)
    if issues:
        raise ValueError("Reproduction bundle USD dependency validation failed: " + issues[0])

    case_contract = copy.deepcopy(case_template)
    case_contract["case"].update(
        {
            "package_id": CASE_PACKAGE_ID,
            "entry_stage": "dataset.usda",
            "entry_stage_sha256": sha256_file(output / "dataset.usda"),
        }
    )
    case_contract["base_map"].update(
        {
            "contract_record": "map-contract.json",
            "contract_record_sha256": sha256_file(map_contract_path),
            "package_id": MAP_PACKAGE_ID,
            "revision": MAP_REVISION,
            "artifact_manifest_sha256": sha256_file(map_package / "manifest.json"),
        }
    )
    case_contract["perimeter_layers"].update(
        {
            "contract_record": "perimeter-contract.json",
            "contract_record_sha256": sha256_file(perimeter_contract_path),
            "layer_package_id": PERIMETER_PACKAGE_ID,
            "revision": PERIMETER_REVISION,
        }
    )
    case_contract["quality_gates"].update(
        {
            "base_map_active": "pending",
            "perimeter_layer_active": "pending",
            "source_integrity": "passed",
            "timeline_validation": "passed",
            "usd_validation": "pending",
            "tree_destruction": "passed",
            "flow_runtime_no_capture": "passed",
            "camera_retargeting": "passed",
            "metadata_completeness": "pending",
            "split_leakage": "pending",
            "human_visual_review": "pending",
            "capture_validation": "not_run",
            "training_readiness": "not_run",
        }
    )
    write_json(output / "contracts" / "map-contract.json", map_contract)
    write_json(output / "contracts" / "perimeter-contract.json", perimeter_contract)
    write_json(output / "contracts" / "case-contract.json", case_contract)

    inventory_path, inventory_sha256, file_count = write_inventory(
        output,
        excluded_names={"manifest.json", "dependency-inventory.json"},
    )
    source_manifest = load_json(validated_reproduction / "manifest.json")
    dataset_contract = copy.deepcopy(source_manifest["dataset"])
    dataset_contract.update(
        {
            "capture_enabled": False,
            "capture_validation_status": "blocked_until_pilot_capture_acceptance",
        }
    )
    environment_contract = copy.deepcopy(source_manifest["environment"])
    for path_field in ("hdri_path", "provenance_path"):
        if environment_contract.get(path_field):
            environment_contract[path_field] = f"map/{environment_contract[path_field]}"
    vegetation_contract = copy.deepcopy(source_manifest["vegetation"])
    vegetation_contract["source_rebuild_index"] = {
        "path": "map/source/vegetation/index.json",
        "sha256": sha256_file(output / "map" / "source" / "vegetation" / "index.json"),
        "inside_bundle": True,
    }
    manifest = {
        "schema": "fireviewer.omniverse-reproducible-reproduction-bundle.v1",
        "bundle_id": BUNDLE_ID,
        "package_id": CASE_PACKAGE_ID,
        "case_id": CASE_ID,
        "status": "candidate_pending_die_visual_acceptance",
        "entry_stage": "dataset.usda",
        "entry_stage_sha256": sha256_file(output / "dataset.usda"),
        "purpose": "downloadable_reproducible_omniverse_case_and_controlled_dataset_source",
        "truth_scope": source_manifest["truth_scope"],
        "common_anchor_l93_m": source_manifest["common_anchor_l93_m"],
        "coordinate_convention": source_manifest["coordinate_convention"],
        "scene_bounds_l93_m": source_manifest["scene_bounds_l93_m"],
        "source_retrospective": copy.deepcopy(source_manifest["source_retrospective"]),
        "base_scene": {
            "package_id": MAP_PACKAGE_ID,
            "path": "map",
            "composition": "internal_relative_map_appearance_camera_and_perimeter_layers",
            "locks": {
                "map_stage": {
                    "path": "map/map.usda",
                    "sha256": sha256_file(output / "map" / "map.usda"),
                },
                "map_manifest": {
                    "path": "map/manifest.json",
                    "sha256": sha256_file(output / "map" / "manifest.json"),
                },
                "perimeter_layer": {
                    "path": "perimeters/perimeters.usda",
                    "sha256": sha256_file(output / "perimeters" / "perimeters.usda"),
                },
                "appearance_layer": {
                    "path": "appearance/appearance.usda",
                    "sha256": sha256_file(output / "appearance" / "appearance.usda"),
                },
                "camera_layer": {
                    "path": "cameras/fixed_cameras.usda",
                    "sha256": sha256_file(output / "cameras" / "fixed_cameras.usda"),
                },
            },
        },
        "environment": environment_contract,
        "vegetation": vegetation_contract,
        "scenario": copy.deepcopy(source_manifest["scenario"]),
        "cameras": copy.deepcopy(source_manifest["cameras"]),
        "dataset": dataset_contract,
        "qa": {
            "automated_source": "qa/validated-v1-source/acceptance.json",
            "human_review_source": "qa/validated-v1-source/HUMAN_REVIEW.md",
            "source_render_acceptance": "accepted_profiles_reused_from_controlled_dataset_pilot",
            "package_visual_reopen": "pending",
            "pilot_capture": "required_before_full_dataset_capture",
        },
        "source_validated_reproduction": {
            "package_id": str(source_manifest["package_id"]),
            "manifest_sha256": original_locks["validated_reproduction"]["manifest.json"]["sha256"],
            "dataset_stage_sha256": original_locks["validated_reproduction"]["dataset.usda"]["sha256"],
            "scenario_tree_sha256": inventory_digest(original_simulation_inventory),
            "copy_mode": "byte_identical_scenario_flow_with_aligned_visual_profile_and_multimodal_runtime",
        },
        "composition": {
            "map": {"path": "map/map.usda", "package_id": MAP_PACKAGE_ID},
            "perimeters": {"path": "perimeters/perimeters.usda", "layer_package_id": PERIMETER_PACKAGE_ID},
            "scenario": "scenarios/scenario.usda",
            "flow": "scenarios/flow.usda",
            "appearance": "appearance/appearance.usda",
            "camera_rig": "cameras/fixed_cameras.usda",
            "accepted_visual_profiles": PROFILE_LAYER_RELATIVE,
        },
        "accepted_visual_profiles": {
            "contract": PROFILE_CONTRACT_RELATIVE,
            "contract_sha256": sha256_file(output / PROFILE_CONTRACT_RELATIVE),
            "persistent_layer": PROFILE_LAYER_RELATIVE,
            "persistent_layer_sha256": sha256_file(output / PROFILE_LAYER_RELATIVE),
            "flow_profile_id": visual_profiles["flow_profile"]["profile_id"],
            "flow_profile_sha256": visual_profiles["flow_profile"]["profile_sha256"],
            "sky_profile_id": visual_profiles["sky_profile"]["profile_id"],
            "sky_profile_sha256": visual_profiles["sky_profile"]["profile_sha256"],
        },
        "runtime": {
            "contract": "runtime/runtime-contract.json",
            "contract_sha256": sha256_file(output / "runtime" / "runtime-contract.json"),
            "python_lock": "runtime/python-lock.json",
            "required_extensions": copy.deepcopy(runtime_contract["required_extensions"]),
            "seed": 20260703,
            "seconds_per_day": PLAYBACK_SECONDS_PER_DAY,
        },
        "dependency_inventory": {
            "path": relative_posix(inventory_path, output),
            "sha256": inventory_sha256,
            "file_count": file_count,
        },
        "capture_enabled": False,
        "dataset_release_allowed": False,
        "training_use_allowed": False,
        "automatic_publication": False,
    }
    write_json(output / "manifest.json", manifest)
    return case_contract, output / "contracts" / "case-contract.json"


def build_download_contract(
    *,
    template: dict[str, Any],
    bundle: Path,
    archive: Path,
    production_root: Path,
    case_contract: dict[str, Any],
    case_contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    contract = copy.deepcopy(template)
    contract["bundle"].update(
        {
            "bundle_id": BUNDLE_ID,
            "root_directory": relative_posix(bundle, production_root),
            "entry_stage": "dataset.usda",
            "entry_stage_sha256": sha256_file(bundle / "dataset.usda"),
            "manifest_sha256": sha256_file(bundle / "manifest.json"),
            "archive_path": relative_posix(archive, production_root),
            "archive_sha256": sha256_file(archive),
        }
    )
    contract["case_reference"].update(
        {
            "contract_record": "case-contract.json",
            "contract_record_sha256": sha256_file(case_contract_path),
            "case_id": case_contract["case"]["case_id"],
            "package_id": case_contract["case"]["package_id"],
        }
    )
    role_paths = {
        "map_contract": bundle / "contracts" / "map-contract.json",
        "map_manifest": bundle / "map" / "manifest.json",
        "perimeter_contract": bundle / "contracts" / "perimeter-contract.json",
        "perimeter_manifest": bundle / "perimeters" / "manifest.json",
        "camera_rig": bundle / "cameras" / "fixed_cameras.usda",
        "scenario": bundle / "scenarios" / "scenario.usda",
        "flow": bundle / "scenarios" / "flow.usda",
        "runtime_contract": bundle / "runtime" / "runtime-contract.json",
        "source_truth": bundle / "source" / "die-2026-v1.json",
        "asset_inventory": bundle / "dependency-inventory.json",
    }
    contract["dependency_locks"] = [
        {
            "role": role,
            "path": relative_posix(path, bundle),
            "sha256": sha256_file(path),
            "inside_bundle": True,
        }
        for role, path in role_paths.items()
    ]
    contract["portability"]["dependency_inventory_sha256"] = sha256_file(
        bundle / "dependency-inventory.json"
    )
    contract["portability"]["isolated_reopen"] = "pending"
    contract["quality_gates"].update(
        {
            "case_active": "pending",
            "dependency_hashes": "passed",
            "relative_path_scan": "passed",
            "archive_integrity": "passed",
            "isolated_kit_reopen": "pending",
            "human_visual_review": "pending",
        }
    )
    write_json(output_path, contract)
    return contract


def create_archive(bundle: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    print(canonical_json({"phase": "archive_bundle", "archive": str(archive)}).strip(), flush=True)
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as handle:
        for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
            arcname = (Path(bundle.name) / path.relative_to(bundle)).as_posix()
            handle.write(path, arcname=arcname)
    with zipfile.ZipFile(archive, mode="r") as handle:
        corrupt = handle.testzip()
        if corrupt is not None:
            raise ValueError(f"Archive integrity failed at {corrupt}")


def key_file_locks(root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    result = {}
    for relative in relative_paths:
        path = root / relative
        result[relative] = {"byte_count": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def verify_original_locks(
    *,
    base_package: Path,
    validated_reproduction: Path,
    locks: dict[str, Any],
) -> None:
    current_base = key_file_locks(base_package, locks["base_scene"].keys())
    current_reproduction = key_file_locks(
        validated_reproduction, locks["validated_reproduction"].keys()
    )
    if current_base != locks["base_scene"]:
        raise ValueError("The original R4 base scene changed during the additive build")
    if current_reproduction != locks["validated_reproduction"]:
        raise ValueError("The original V1 reproduction changed during the additive build")


def build(args: argparse.Namespace) -> Path:
    base_package = args.base_package.resolve()
    validated_reproduction = args.validated_reproduction.resolve()
    retrospective_path = args.retrospective.resolve()
    vegetation_package = args.vegetation_package.resolve()
    mnt_cache = args.mnt_cache.resolve()
    ign_source_manifest = args.ign_source_manifest.resolve()
    contracts_root = args.contracts_root.resolve()
    output_root = args.output_root.resolve()
    required_files = (
        base_package / "dataset.usda",
        base_package / "manifest.json",
        base_package / "site" / "site.usda",
        base_package / "appearance" / "appearance.usda",
        base_package / "cameras" / "fixed_cameras.usda",
        validated_reproduction / "dataset.usda",
        validated_reproduction / "manifest.json",
        validated_reproduction / "scenarios" / "scenario.usda",
        validated_reproduction / "scenarios" / "flow.usda",
        retrospective_path,
        vegetation_package / "index.json",
        ign_source_manifest,
        contracts_root / "examples" / "die-map-upload.candidate.json",
        contracts_root / "examples" / "die-progressive-perimeters.candidate.json",
        contracts_root / "examples" / "die-retrospective-case.candidate.json",
        contracts_root / "examples" / "die-reproduction-download.candidate.json",
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not mnt_cache.is_dir():
        raise FileNotFoundError(mnt_cache)
    usd_python = args.usd_python.resolve()
    if not usd_python.is_file():
        raise FileNotFoundError(usd_python)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing production root: {output_root}")

    original_locks = {
        "schema": "fireviewer.preserved-original-artifacts.v1",
        "base_scene": key_file_locks(
            base_package,
            (
                "dataset.usda",
                "manifest.json",
                "site/site.usda",
                "appearance/appearance.usda",
                "cameras/fixed_cameras.usda",
            ),
        ),
        "validated_reproduction": key_file_locks(
            validated_reproduction,
            (
                "dataset.usda",
                "manifest.json",
                "scenarios/scenario.usda",
                "scenarios/flow.usda",
            ),
        ),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "preservation" / "original-artifact-locks.json", original_locks)

    map_package = output_root / "map" / MAP_PACKAGE_ID
    perimeter_package = output_root / "perimeters" / PERIMETER_PACKAGE_ID
    bundle = output_root / "reproduction" / BUNDLE_ID
    archive = output_root / "archives" / f"{BUNDLE_ID}.zip"

    map_template = load_json(contracts_root / "examples" / "die-map-upload.candidate.json")
    perimeter_template = load_json(
        contracts_root / "examples" / "die-progressive-perimeters.candidate.json"
    )
    case_template = load_json(contracts_root / "examples" / "die-retrospective-case.candidate.json")
    download_template = load_json(
        contracts_root / "examples" / "die-reproduction-download.candidate.json"
    )

    map_contract, map_contract_path = build_map_package(
        base_package=base_package,
        vegetation_package=vegetation_package,
        ign_source_manifest=ign_source_manifest,
        output=map_package,
        map_template=map_template,
        original_locks=original_locks,
    )
    perimeter_contract, perimeter_contract_path = build_perimeter_package(
        retrospective_path=retrospective_path,
        mnt_cache=mnt_cache,
        map_package=map_package,
        map_contract=map_contract,
        map_contract_path=map_contract_path,
        output=perimeter_package,
        perimeter_template=perimeter_template,
        usd_python=usd_python,
    )
    write_review_stage(
        output_root / "review" / "review-map-with-perimeters.usda",
        map_relative=f"../map/{MAP_PACKAGE_ID}/map.usda",
        perimeter_relative=f"../perimeters/{PERIMETER_PACKAGE_ID}/perimeters.usda",
    )
    case_contract, case_contract_path = build_reproduction_bundle(
        base_package=base_package,
        validated_reproduction=validated_reproduction,
        retrospective_path=retrospective_path,
        map_package=map_package,
        perimeter_package=perimeter_package,
        map_contract=map_contract,
        map_contract_path=map_contract_path,
        perimeter_contract=perimeter_contract,
        perimeter_contract_path=perimeter_contract_path,
        case_template=case_template,
        output=bundle,
        original_locks=original_locks,
    )
    if not args.skip_archive:
        create_archive(bundle, archive)
        archive_path: Path | None = archive
    else:
        archive_path = None

    top_contracts = output_root / "contracts"
    write_json(top_contracts / "map-contract.json", map_contract)
    write_json(top_contracts / "perimeter-contract.json", perimeter_contract)
    write_json(top_contracts / "case-contract.json", case_contract)
    if archive_path is not None:
        download_contract = build_download_contract(
            template=download_template,
            bundle=bundle,
            archive=archive_path,
            production_root=output_root,
            case_contract=case_contract,
            case_contract_path=case_contract_path,
            output_path=top_contracts / "download-bundle-contract.json",
        )
    else:
        download_contract = copy.deepcopy(download_template)
        write_json(top_contracts / "download-bundle-contract.json", download_contract)

    verify_original_locks(
        base_package=base_package,
        validated_reproduction=validated_reproduction,
        locks=original_locks,
    )
    report = {
        "schema": "fireviewer.omniverse-separated-production-build-report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_built_preserved_originals",
        "original_artifacts_unchanged": True,
        "map": {
            "stage": relative_posix(map_package / "map.usda", output_root),
            "sha256": sha256_file(map_package / "map.usda"),
        },
        "perimeters": {
            "stage": relative_posix(perimeter_package / "perimeters.usda", output_root),
            "sha256": sha256_file(perimeter_package / "perimeters.usda"),
            "state_count": 21,
        },
        "review": {
            "stage": "review/review-map-with-perimeters.usda",
            "sha256": sha256_file(output_root / "review" / "review-map-with-perimeters.usda"),
        },
        "reproduction": {
            "stage": relative_posix(bundle / "dataset.usda", output_root),
            "sha256": sha256_file(bundle / "dataset.usda"),
            "source_scenario_flow_byte_identical": True,
        },
        "download": (
            {
                "archive": relative_posix(archive, output_root),
                "sha256": sha256_file(archive),
                "byte_count": archive.stat().st_size,
            }
            if archive_path is not None
            else None
        ),
        "release": {
            "contract_status": "candidate_pending_die_visual_acceptance",
            "map_upload_allowed": False,
            "perimeter_attachment_allowed": False,
            "simulation_use_allowed": False,
            "pilot_capture_allowed": False,
            "full_dataset_capture_allowed": False,
            "training_use_allowed": False,
            "automatic_publication": False,
        },
    }
    write_json(output_root / "qa" / "build-report.json", report)
    print(canonical_json(report), end="", flush=True)
    return output_root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-package", type=Path, required=True)
    result.add_argument("--validated-reproduction", type=Path, required=True)
    result.add_argument("--retrospective", type=Path, required=True)
    result.add_argument("--vegetation-package", type=Path, required=True)
    result.add_argument("--mnt-cache", type=Path, required=True)
    result.add_argument("--ign-source-manifest", type=Path, required=True)
    result.add_argument("--contracts-root", type=Path, required=True)
    result.add_argument("--usd-python", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--skip-archive", action="store_true")
    return result


def main() -> int:
    build(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
