"""Build a local Die retrospective fire replay on the accepted FireViewer scene.

The generated package is an OpenUSD overlay.  It references the accepted site,
appearance, camera and environment layers in place and authors only a new
twenty-one-day fire scenario.  The retrospective geometries remain explicitly
labelled as post-incident derived data and are not an operational fire product.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
from pyproj import Transformer
import rasterio
from shapely import intersects_xy, make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

from build_fireviewer_dataset_usd import (
    HeightSampler,
    build_capture_schedule,
    canonical_json,
    sha256_file,
    usd_string,
    write_fire_state,
    write_flow_layer,
    write_scenario_layer,
)
from fire_spread_model import (
    FireFrontSegment,
    FireSpreadDrivers,
    FireSpreadResult,
    FireSpreadState,
)
from fireviewer_accepted_visual_profiles import (
    PROFILE_CONTRACT_RELATIVE,
    PROFILE_LAYER_RELATIVE,
    write_accepted_visual_profile_artifacts,
)


SCHEMA = "fireviewer.omniverse-die-retrospective-overlay.v1"
SOURCE_SCHEMA = "1.0"
SOURCE_DATASET_ID = "die-2026-retrospective-v1"
VEGETATION_SCHEMA = "fireviewer.mnt-mns-vegetation-rebuild.v1"
TRUTH_SCOPE = "retrospective_daily_perimeter_replay_on_validated_scene_not_operational_forecast"
STATE_SELECTION_CONTRACT = "incident_21_days_1_state_per_day_21_named_states"
INCIDENT_DAYS = 21
STATES_PER_DAY = 1
OBSERVATION_SECONDS_PER_STATE = 86400
PLAYBACK_SECONDS_PER_DAY = 60.0
GRID_CELL_SIZE_M = 25.0


@dataclass(frozen=True)
class DailyGeometry:
    local_date: str
    valid_at: str
    active_l93: BaseGeometry
    source_footprint_l93: BaseGeometry
    simulation_footprint_l93: BaseGeometry
    scene_footprint_l93: BaseGeometry
    flow_front_l93: BaseGeometry
    active_in_scene: bool
    activity_area_ha: float
    activity_method: str
    confidence: str | None
    uncertainty_m: float | None
    layer_revision_id: str
    source_revision_ids: tuple[str, ...]
    source_geometry_area_ha: float
    source_reported_area_ha: float | None
    source_area_quality: str
    source_footprint_ref_date: str | None


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    geometry = make_valid(geometry)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon))]
        return unary_union(polygons) if polygons else Polygon()
    return Polygon()


def _project_geojson(value: dict[str, Any], transformer: Transformer) -> BaseGeometry:
    source = shape({"type": value["type"], "coordinates": value["coordinates"]})
    return _polygonal(transform(transformer.transform, source))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_daily_dates(records: list[dict[str, Any]]) -> None:
    if len(records) != INCIDENT_DAYS:
        raise ValueError(f"Die retrospective must contain {INCIDENT_DAYS} daily activity zones")
    first = date.fromisoformat(str(records[0]["local_date"]))
    for index, record in enumerate(records):
        expected = first + timedelta(days=index)
        actual = date.fromisoformat(str(record["local_date"]))
        if actual != expected:
            raise ValueError(f"Non-contiguous retrospective date at index {index}: {actual} != {expected}")


def prepare_daily_geometries(
    retrospective: dict[str, Any],
    *,
    site_bounds: tuple[float, float, float, float],
    scene_coverage: BaseGeometry | None = None,
) -> list[DailyGeometry]:
    if str(retrospective.get("schema_version")) != SOURCE_SCHEMA:
        raise ValueError(f"Unsupported retrospective schema: {retrospective.get('schema_version')}")
    if retrospective.get("dataset_id") != SOURCE_DATASET_ID:
        raise ValueError(f"Unexpected retrospective dataset: {retrospective.get('dataset_id')}")
    records = list(retrospective.get("activity_zones", []))
    _validate_daily_dates(records)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    scene_box = _polygonal(scene_coverage) if scene_coverage is not None else box(*site_bounds)
    previous_source: BaseGeometry | None = None
    monotonic: BaseGeometry = Polygon()
    daily: list[DailyGeometry] = []
    for record in records:
        geometry_record = dict(record["geometry_geojson"])
        active = _project_geojson(geometry_record, transformer)
        source_value = geometry_record.get("global_footprint_geojson")
        if source_value is not None:
            previous_source = _project_geojson(dict(source_value), transformer)
        if previous_source is None:
            raise ValueError(f"No cumulative footprint is available for {record['local_date']}")
        source_footprint = previous_source
        monotonic = _polygonal(unary_union((monotonic, source_footprint, active)))
        scene_footprint = _polygonal(monotonic.intersection(scene_box))
        active_scene = _polygonal(active.intersection(scene_box))
        active_in_scene = not active_scene.is_empty and active_scene.area > 1.0
        flow_front = active_scene if active_in_scene else active
        if flow_front.is_empty:
            raise ValueError(f"Daily active zone is empty on {record['local_date']}")
        projected_source_area_ha = float(source_footprint.area / 10000.0)
        declared_source_area_ha = float(geometry_record["global_geometry_area_ha"])
        tolerance = max(0.5, declared_source_area_ha * 0.02)
        if abs(projected_source_area_ha - declared_source_area_ha) > tolerance:
            raise ValueError(
                f"Projected cumulative area mismatch on {record['local_date']}: "
                f"{projected_source_area_ha:.2f} ha != {declared_source_area_ha:.2f} ha"
            )
        daily.append(
            DailyGeometry(
                local_date=str(record["local_date"]),
                valid_at=str(record["valid_at"]),
                active_l93=active,
                source_footprint_l93=source_footprint,
                simulation_footprint_l93=monotonic,
                scene_footprint_l93=scene_footprint,
                flow_front_l93=flow_front,
                active_in_scene=active_in_scene,
                activity_area_ha=float(record["activity_area_ha"]),
                activity_method=str(record["activity_method"]),
                confidence=str(record["confidence"]) if record.get("confidence") is not None else None,
                uncertainty_m=float(record["uncertainty_m"]) if record.get("uncertainty_m") is not None else None,
                layer_revision_id=str(record["layer_revision_id"]),
                source_revision_ids=tuple(str(value) for value in record.get("source_revision_ids", [])),
                source_geometry_area_ha=declared_source_area_ha,
                source_reported_area_ha=(
                    float(geometry_record["global_reported_burned_area_ha"])
                    if geometry_record.get("global_reported_burned_area_ha") is not None
                    else None
                ),
                source_area_quality=str(geometry_record["global_area_quality"]),
                source_footprint_ref_date=(
                    str(geometry_record["global_footprint_ref_date"])
                    if geometry_record.get("global_footprint_ref_date") is not None
                    else None
                ),
            )
        )
    return daily


def _snap_domain(
    geometry: BaseGeometry,
    site_bounds: tuple[float, float, float, float],
    cell_size_m: float,
) -> tuple[float, float, float, float]:
    if geometry.is_empty:
        raise ValueError("The retrospective footprint does not intersect the accepted scene")
    xmin, ymin, xmax, ymax = geometry.bounds
    sxmin, symin, sxmax, symax = site_bounds
    west = max(sxmin, math.floor(xmin / cell_size_m) * cell_size_m)
    south = max(symin, math.floor(ymin / cell_size_m) * cell_size_m)
    east = min(sxmax, math.ceil(xmax / cell_size_m) * cell_size_m)
    north = min(symax, math.ceil(ymax / cell_size_m) * cell_size_m)
    if east <= west or north <= south:
        raise ValueError("Invalid retrospective simulation grid bounds")
    return west, south, east, north


def _daily_masks(
    daily: list[DailyGeometry],
    domain: tuple[float, float, float, float],
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    west, south, east, north = domain
    width = int(round((east - west) / cell_size_m))
    height = int(round((north - south) / cell_size_m))
    xs = west + (np.arange(width, dtype=np.float64) + 0.5) * cell_size_m
    ys = south + (np.arange(height, dtype=np.float64) + 0.5) * cell_size_m
    xx, yy = np.meshgrid(xs, ys)
    masks = [np.asarray(intersects_xy(item.scene_footprint_l93, xx, yy), dtype=np.bool_) for item in daily]
    for previous, current in zip(masks, masks[1:]):
        if np.any(previous & ~current):
            raise ValueError("Derived simulation footprint is not monotonic")
    return xs, ys, np.stack((xx, yy)), masks


def _sample_grid_elevation(
    source_map: Path,
    catalog: dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    required_mask: np.ndarray,
) -> np.ndarray:
    elevation = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for tile in catalog["terrain_tiles"]:
        west, south, east, north = (float(value) for value in tile["bounds_l93_metres"])
        x_indices = np.flatnonzero((xs >= west) & (xs <= east))
        y_indices = np.flatnonzero((ys >= south) & (ys <= north))
        if not len(x_indices) or not len(y_indices):
            continue
        xx, yy = np.meshgrid(xs[x_indices], ys[y_indices])
        coordinates = zip(xx.ravel().tolist(), yy.ravel().tolist())
        with rasterio.open(source_map / str(tile["elevation"]["path"])) as dataset:
            values = np.fromiter(
                (float(sample[0]) for sample in dataset.sample(coordinates)),
                dtype=np.float32,
                count=xx.size,
            ).reshape(xx.shape)
        elevation[np.ix_(y_indices, x_indices)] = values
    unresolved = ~np.isfinite(elevation)
    required_unresolved = unresolved & required_mask
    if bool(np.any(required_unresolved)):
        missing = int(np.count_nonzero(required_unresolved))
        raise ValueError(f"MNT grid sampling left {missing} cells unresolved")
    elevation[unresolved] = 0.0
    return elevation


def _front_rate(item: DailyGeometry, maximum_activity_ha: float) -> float:
    normalized = math.sqrt(max(item.activity_area_ha, 0.0) / max(maximum_activity_ha, 1e-6))
    return max(0.055, min(0.19, 0.045 + normalized * 0.145))


def _line_parts(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    if geometry.geom_type in {"LineString", "LinearRing"}:
        yield geometry
        return
    for child in getattr(geometry, "geoms", []):
        yield from _line_parts(child)


def _front_segments(
    geometry: BaseGeometry,
    *,
    height_sampler: HeightSampler,
    spread_rate_m_s: float,
) -> list[FireFrontSegment]:
    boundary = geometry.simplify(3.0, preserve_topology=True).boundary.segmentize(25.0)
    segments: list[FireFrontSegment] = []
    height_cache: dict[tuple[float, float], float] = {}

    def height(east: float, north: float) -> float:
        key = (round(east, 3), round(north, 3))
        if key not in height_cache:
            height_cache[key] = height_sampler.at(east, north)
        return height_cache[key]

    for line in _line_parts(boundary):
        coordinates = list(line.coords)
        for start, end in zip(coordinates, coordinates[1:]):
            if math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1])) <= 0.01:
                continue
            segments.append(
                FireFrontSegment(
                    start=(float(start[0]), float(start[1]), height(float(start[0]), float(start[1]))),
                    end=(float(end[0]), float(end[1]), height(float(end[0]), float(end[1]))),
                    spread_rate_m_s=spread_rate_m_s,
                )
            )
    if not segments:
        raise ValueError("A retrospective daily front produced no usable Flow segments")
    return segments


def classify_tree_burn_days(
    vegetation_index_path: Path,
    daily: list[DailyGeometry],
    *,
    expected_package_id: str,
    expected_index_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if sha256_file(vegetation_index_path) != expected_index_sha256:
        raise ValueError("MNT/MNS vegetation index checksum differs from the accepted scene lock")
    index = _load_json(vegetation_index_path)
    if index.get("schema") != VEGETATION_SCHEMA or index.get("base_package_id") != expected_package_id:
        raise ValueError("MNT/MNS vegetation index does not belong to the accepted base scene")
    total = int(index["counts"]["tree_instances"])
    burn_day = np.zeros(total, dtype=np.uint8)
    offset = 0
    for tile_number, tile in enumerate(index["tiles"], start=1):
        tile_path = vegetation_index_path.parent / str(tile["path"])
        if tile_path.stat().st_size != int(tile["byte_count"]) or sha256_file(tile_path) != str(tile["sha256"]):
            raise ValueError(f"MNT/MNS vegetation tile lock failed: {tile_path}")
        with np.load(tile_path, allow_pickle=False) as values:
            positions = np.asarray(values["positions_l93_m"], dtype=np.float64)
        count = len(positions)
        if count != int(tile["accepted_crown_count"]):
            raise ValueError(f"Vegetation count mismatch in {tile_path}")
        local_days = np.zeros(count, dtype=np.uint8)
        east = positions[:, 0]
        north = positions[:, 1]
        tile_bounds = box(*(float(value) for value in tile["bounds_l93_m"]))
        for day_index, item in enumerate(daily, start=1):
            if not item.scene_footprint_l93.intersects(tile_bounds):
                continue
            candidates = local_days == 0
            if not bool(np.any(candidates)):
                break
            inside = np.asarray(
                intersects_xy(item.scene_footprint_l93, east[candidates], north[candidates]),
                dtype=np.bool_,
            )
            candidate_indices = np.flatnonzero(candidates)
            local_days[candidate_indices[inside]] = day_index
        burn_day[offset : offset + count] = local_days
        offset += count
        if tile_number % 16 == 0 or tile_number == len(index["tiles"]):
            print(
                canonical_json(
                    {
                        "phase": "classify_mnt_mns_trees",
                        "complete_tiles": tile_number,
                        "total_tiles": len(index["tiles"]),
                        "processed_tree_count": offset,
                        "burned_tree_count": int(np.count_nonzero(burn_day[:offset])),
                    }
                ).strip(),
                flush=True,
            )
    if offset != total:
        raise ValueError(f"Vegetation ID coverage mismatch: {offset} != {total}")
    counts = [int(np.count_nonzero((burn_day > 0) & (burn_day <= day))) for day in range(1, INCIDENT_DAYS + 1)]
    if counts != sorted(counts):
        raise ValueError("Burned source-tree counts are not monotonic")
    return burn_day, index


def _relative_asset(from_directory: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_directory.resolve()).replace("\\", "/")


def write_overlay_stage(
    output: Path,
    *,
    package_id: str,
    anchor: tuple[float, float],
    base_package: Path,
) -> None:
    sublayers = (
        PROFILE_LAYER_RELATIVE,
        _relative_asset(output, base_package / "site/site.usda"),
        "scenarios/scenario.usda",
        "scenarios/flow.usda",
        _relative_asset(output, base_package / "appearance/appearance.usda"),
        _relative_asset(output, base_package / "cameras/fixed_cameras.usda"),
    )
    layer_lines = ",\n".join(f"        @{value}@" for value in sublayers)
    stage = (
        "#usda 1.0\n(\n"
        '    defaultPrim = "World"\n'
        "    metersPerUnit = 1\n"
        '    upAxis = "Z"\n'
        "    startTimeCode = 0\n"
        f"    endTimeCode = {INCIDENT_DAYS * PLAYBACK_SECONDS_PER_DAY:.6f}\n"
        "    timeCodesPerSecond = 1\n"
        "    framesPerSecond = 30\n"
        "    customLayerData = {\n"
        "        dictionary renderSettings = {\n"
        '            bool "rtx:flow:enabled" = 1\n'
        '            int "rtx:flow:maxBlocks" = 16384\n'
        '            bool "rtx:flow:pathTracingEnabled" = 1\n'
        '            bool "rtx:flow:rayTracedReflectionsEnabled" = 1\n'
        '            bool "rtx:flow:rayTracedShadowsEnabled" = 1\n'
        '            bool "rtx:flow:rayTracedTranslucencyEnabled" = 1\n'
        "        }\n"
        "    }\n"
        "    subLayers = [\n"
        f"{layer_lines}\n"
        "    ]\n"
        ")\n\n"
        'def Xform "World" (kind = "assembly")\n{\n'
        f'    custom string fireviewer:package_id = "{usd_string(package_id)}"\n'
        '    custom string fireviewer:dataset_contract = "local_retrospective_fire_replay"\n'
        f'    custom string fireviewer:base_scene_package_id = "{usd_string(base_package.name)}"\n'
        f"    custom double2 fireviewer:common_anchor_l93_m = ({anchor[0]:.6f}, {anchor[1]:.6f})\n"
        '    custom bool fireviewer:publication_allowed = 0\n'
        '    def DomeLight "SkyFill"\n    {\n'
        "        float intensity = 350\n"
        "        asset inputs:texture:file = @@\n"
        "        color3f inputs:color = (0.58, 0.72, 0.95)\n"
        "        float inputs:intensity = 180\n"
        "        float inputs:exposure = 0\n"
        '        token inputs:texture:format = "latlong"\n'
        "        float3 xformOp:rotateXYZ = (0, 0, 28)\n"
        '        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]\n'
        "    }\n"
        '    def DistantLight "Sun"\n    {\n'
        "        float angle = 0.8\n"
        "        color3f color = (1.0, 0.91, 0.78)\n"
        "        float intensity = 950\n"
        "        float3 xformOp:rotateXYZ = (24, -18, 42)\n"
        '        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]\n'
        "    }\n"
        "}\n"
    )
    (output / "dataset.usda").write_text(stage, encoding="utf-8", newline="\n")


def _locked_file(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "byte_count": path.stat().st_size}


def _write_runtime_contract(
    output: Path,
    *,
    capture_schedule: dict[str, Any],
    schedule_sha256: str,
    camera_ids: list[str],
    visual_profiles: dict[str, Any],
) -> dict[str, Any]:
    capture = {
        "enabled": False,
        "validation_status": "blocked_until_user_visual_acceptance",
        "incident_days": INCIDENT_DAYS,
        "states_per_day": STATES_PER_DAY,
        "states": [f"state_{index:03d}" for index in range(1, INCIDENT_DAYS + 1)],
        "camera_pool": camera_ids,
        "human_cameras": 55,
        "aerial_cameras": 7,
        "views_per_state": int(capture_schedule["views_per_state"]),
        "zooms_per_view": int(capture_schedule["zooms_per_view"]),
        "captures_per_state": int(capture_schedule["captures_per_state"]),
        "positive_views_per_state": 16,
        "negative_views_per_state": 4,
        "profile_mix_per_state": capture_schedule["profile_mix_per_state"],
        "expected_viewpoint_plans": int(capture_schedule["expected_viewpoint_plans"]),
        "expected_capture_cases": int(capture_schedule["expected_capture_cases"]),
        "schedule_path": "capture-schedule.json",
        "schedule_sha256": schedule_sha256,
    }
    runtime = {
        "schema": "fireviewer.omniverse-runtime.v1",
        "entry_stage": "dataset.usda",
        "runner_module": "run_fireviewer_replicator_dataset.py",
        "writer_module": "fireviewer_replicator_writer.py",
        "storage_module": "fireviewer_capture_storage.py",
        "accepted_visual_profiles": {
            "contract": "accepted-visual-profiles.json",
            "contract_sha256": sha256_file(output / PROFILE_CONTRACT_RELATIVE),
            "persistent_layer": PROFILE_LAYER_RELATIVE,
            "persistent_layer_sha256": visual_profiles["persistent_application"]["layer_sha256"],
            "flow_profile_id": visual_profiles["flow_profile"]["profile_id"],
            "flow_profile_sha256": visual_profiles["flow_profile"]["profile_sha256"],
            "sky_profile_id": visual_profiles["sky_profile"]["profile_id"],
            "sky_profile_sha256": visual_profiles["sky_profile"]["profile_sha256"],
        },
        "required_extensions": ["omni.replicator.core", "omni.flowusd"],
        "simulation_playback": {
            "seconds_per_day": PLAYBACK_SECONDS_PER_DAY,
            "state_transition": "stepped_daily_hold",
            "capture_on_first_launch": False,
        },
        "flow_contract": {
            "runtime": "actual_omni.flowusd_combustion_and_smoke",
            "combustion": "accepted_meter_scaled_flow_profile_reused_without_parameter_change",
            "animation": "twenty_one_daily_held_states_256_patch_front_48_hotspots_aligned_smoke_mesh",
            "flow_visual": "/World/FireScenario/FlowVisual",
            "source_points": "/World/FireScenario/Truth3D/SmokeSources",
            "close": "/World/Appearance/FlowClose",
            "mid_distance": "/World/Appearance/SmokeMidDistance",
            "beauty_view": "omni_flowusd_only_truth_front_and_smoke_points_hidden_in_visual_validation_session",
        },
        "modalities": [
            "rgb",
            "aerial_synthetic_thermal_16bit",
            "semantic_masks",
            "pointcloud",
            "fire_front_visible",
            "fire_perimeter",
            "depth",
            "geolocation",
            "abstention",
        ],
        "capture": capture,
    }
    path = output / "runtime/runtime-contract.json"
    path.write_text(canonical_json(runtime), encoding="utf-8")
    return runtime


def _write_qa(
    output: Path,
    *,
    source_sha256: str,
    base_locks: dict[str, Any],
    states: list[dict[str, Any]],
    daily_metadata: list[dict[str, Any]],
    tree_count: int,
) -> None:
    active_outside = [item["local_date"] for item in daily_metadata if not item["active_in_scene"]]
    qa = {
        "schema": "fireviewer.omniverse-retrospective-overlay-qa.v1",
        "status": "automated_validation_pending",
        "human_visual_acceptance": "required_in_visible_native_kit_window",
        "capture_status": "disabled_until_user_acceptance",
        "checks": {
            "same_base_scene_locked": True,
            "base_file_locks": base_locks,
            "source_retrospective_sha256": source_sha256,
            "daily_state_count": len(states),
            "daily_dates_contiguous": True,
            "burned_area_monotonic": True,
            "burned_tree_count_monotonic": True,
            "source_tree_population": tree_count,
            "flow_profile": "accepted_profile_reused",
            "active_outside_scene_dates": active_outside,
        },
        "proof_boundaries": {
            "usd_and_metadata_validation": "automated",
            "flow_runtime_validation": "requires_native_kit_execution",
            "visual_quality": "user_validation_required",
            "incident_truth": "post_incident_derived_replay_not_operational_truth",
        },
    }
    (output / "qa/acceptance.json").write_text(canonical_json(qa), encoding="utf-8")
    (output / "qa/HUMAN_REVIEW.md").write_text(
        "# Die retrospective replay - visual review\n\n"
        "This local package references the accepted FireViewer scene and assets in place.\n\n"
        "- [ ] The terrain, orthophoto, buildings, roads, vegetation, sky and cameras match the accepted scene.\n"
        "- [ ] Each minute selects the next daily state from 2026-07-03 through 2026-07-23.\n"
        "- [ ] Flames and smoke retain the accepted Flow appearance and remain attached to the daily active zone.\n"
        "- [ ] Burned trees disappear and the burned ground remains monotonic.\n"
        "- [ ] Dates whose active geometry is outside the fixed scene do not invent an in-scene fire.\n"
        "- [ ] Capture remains disabled until this review is accepted.\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> Path:
    source = args.source_retrospective.resolve()
    base_package = args.base_package.resolve()
    source_map = args.source_map.resolve()
    vegetation_index_path = args.vegetation_index.resolve()
    output = args.output.resolve()
    required = (
        source,
        base_package / "manifest.json",
        base_package / "dataset.usda",
        base_package / "site/site.usda",
        base_package / "appearance/appearance.usda",
        base_package / "cameras/fixed_cameras.usda",
        base_package / "assets/environments/farm_field_puresky_4k.hdr",
        source_map / "catalog.json",
        vegetation_index_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and not args.replace_generated_output:
            raise FileExistsError(f"Refusing to overwrite a completed output package: {output}")
        if manifest_path.exists():
            existing_manifest = _load_json(manifest_path)
            if (
                existing_manifest.get("schema") != SCHEMA
                or existing_manifest.get("publication_allowed") is not False
                or existing_manifest.get("purpose") != "local_omniverse_validation_only"
            ):
                raise FileExistsError(f"Refusing to replace an output not owned by this local builder: {output}")
        allowed_roots = {
            "dataset.usda",
            "manifest.json",
            "appearance",
            "scenarios",
            "runtime",
            "qa",
        }
        unexpected = [
            path
            for path in output.rglob("*")
            if path.is_file() and path.relative_to(output).parts[0] not in allowed_roots
        ]
        if unexpected:
            raise FileExistsError(f"Refusing to reuse an output containing unknown files: {unexpected[0]}")
    for relative in ("appearance", "scenarios/states", "runtime", "qa"):
        (output / relative).mkdir(parents=True, exist_ok=True)

    retrospective = _load_json(source)
    base_manifest = _load_json(base_package / "manifest.json")
    source_manifest = _load_json(base_package / "source-usd/manifest.json")
    catalog = _load_json(source_map / "catalog.json")
    base_package_id = str(base_manifest["package_id"])
    anchor = tuple(float(value) for value in source_manifest["common_anchor_l93_metres"])
    site_bounds = tuple(float(value) for value in source_manifest["bounds_l93_metres"])
    if len(anchor) != 2 or len(site_bounds) != 4:
        raise ValueError("Accepted base scene coordinate lock is incomplete")
    if base_package_id != source_manifest["package_id"]:
        raise ValueError("Accepted base scene manifests disagree on package identity")
    scene_coverage = _polygonal(
        unary_union(
            [
                box(*(float(value) for value in tile["bounds_l93_metres"]))
                for tile in catalog["terrain_tiles"]
            ]
        )
    )
    daily = prepare_daily_geometries(
        retrospective,
        site_bounds=site_bounds,
        scene_coverage=scene_coverage,
    )
    domain = _snap_domain(daily[-1].scene_footprint_l93, site_bounds, GRID_CELL_SIZE_M)
    xs, ys, grid_coordinates, masks = _daily_masks(daily, domain, GRID_CELL_SIZE_M)
    elevation = _sample_grid_elevation(source_map, catalog, xs, ys, required_mask=masks[-1])

    vegetation_lock = str(base_manifest["vegetation"]["source_rebuild"]["index_sha256"])
    burn_day, vegetation_index = classify_tree_burn_days(
        vegetation_index_path,
        daily,
        expected_package_id=base_package_id,
        expected_index_sha256=vegetation_lock,
    )
    source_sha256 = sha256_file(source)
    base_manifest_sha256 = sha256_file(base_package / "manifest.json")
    simulation_id = hashlib.sha256(
        f"{source_sha256}|{base_manifest_sha256}|{vegetation_lock}|{TRUTH_SCOPE}".encode("utf-8")
    ).hexdigest()[:24]
    base_propagation = _load_json(base_package / "scenarios/propagation.json")
    driver_values = dict(base_propagation["drivers"])
    driver_values["cell_size_m"] = GRID_CELL_SIZE_M
    drivers = FireSpreadDrivers(**driver_values)
    ignition_point = daily[0].active_l93.representative_point()
    ignition = (float(ignition_point.x), float(ignition_point.y))
    maximum_activity = max(item.activity_area_ha for item in daily)
    height_sampler = HeightSampler(source_map, catalog)
    try:
        states: list[FireSpreadState] = []
        state_metadata: list[dict[str, Any]] = []
        arrival_time_s = np.full(masks[0].shape, np.inf, dtype=np.float64)
        spread_rate_field = np.zeros(masks[0].shape, dtype=np.float32)
        for day_index, (item, mask) in enumerate(zip(daily, masks), start=1):
            rate = _front_rate(item, maximum_activity)
            newly_burned = mask & ~np.isfinite(arrival_time_s)
            arrival_time_s[newly_burned] = (day_index - 1) * OBSERVATION_SECONDS_PER_STATE
            spread_rate_field[newly_burned] = rate
            segments = _front_segments(
                item.flow_front_l93,
                height_sampler=height_sampler,
                spread_rate_m_s=rate,
            )
            burned_tree_count = int(np.count_nonzero((burn_day > 0) & (burn_day <= day_index)))
            state = FireSpreadState(
                state_index=day_index,
                elapsed_s=float((day_index - 1) * OBSERVATION_SECONDS_PER_STATE),
                burned_mask=mask,
                front_segments=segments,
                burned_tree_ids=set(),
                burned_area_m2=float(item.scene_footprint_l93.area),
                active_front_length_m=float(item.flow_front_l93.boundary.length),
                mean_front_spread_rate_m_s=rate,
            )
            states.append(state)
            state_metadata.append(
                {
                    "state_id": f"state_{day_index:03d}",
                    "local_date": item.local_date,
                    "valid_at": item.valid_at,
                    "active_in_scene": item.active_in_scene,
                    "activity_area_ha": round(item.activity_area_ha, 3),
                    "activity_method": item.activity_method,
                    "confidence": item.confidence,
                    "uncertainty_m": item.uncertainty_m,
                    "layer_revision_id": item.layer_revision_id,
                    "source_revision_ids": list(item.source_revision_ids),
                    "source_geometry_area_ha": round(item.source_geometry_area_ha, 3),
                    "source_reported_area_ha": item.source_reported_area_ha,
                    "source_area_quality": item.source_area_quality,
                    "source_footprint_ref_date": item.source_footprint_ref_date,
                    "source_active_area_projected_ha": round(item.active_l93.area / 10000.0, 3),
                    "simulation_monotonic_area_ha": round(item.simulation_footprint_l93.area / 10000.0, 3),
                    "simulation_area_in_scene_ha": round(item.scene_footprint_l93.area / 10000.0, 3),
                    "burned_tree_count": burned_tree_count,
                    "flow_front_segment_count": len(segments),
                    "visual_front_rate_m_s": round(rate, 6),
                }
            )
        result = FireSpreadResult(
            simulation_id=simulation_id,
            model_metadata={
                "model": "daily_retrospective_perimeter_replay",
                "truth_scope": TRUTH_SCOPE,
                "source_dataset_id": SOURCE_DATASET_ID,
                "source_sha256": source_sha256,
                "simulation_adaptation": "monotonic_union_of_source_daily_footprint_and_active_zone",
                "visual_intensity": "accepted_flow_profile_with_daily_activity_scaled_front_rate",
                "terrain": "accepted_IGN_MNT_NGF_IGN69",
                "vegetation": "accepted_MNT_MNS_detected_crowns_exact_source_ids",
            },
            domain_bounds_l93_m=domain,
            ignition_l93_m=ignition,
            drivers=drivers,
            elevation_m=elevation,
            fuel_load=masks[-1].astype(np.float32),
            burnable_mask=masks[-1].copy(),
            arrival_time_s=arrival_time_s,
            spread_rate_m_s=spread_rate_field,
            cell_tree_ids=[],
            states=states,
        )
        state_records: list[dict[str, Any]] = []
        for day_index, (state, item) in enumerate(zip(states, daily), start=1):
            burned_ids = np.flatnonzero((burn_day > 0) & (burn_day <= day_index)).astype(np.int64) + 1
            record = write_fire_state(
                output / f"scenarios/states/state_{day_index:03d}.usda",
                result=result,
                state=state,
                state_count=INCIDENT_DAYS,
                anchor=anchor,
                trees=[],
                terrain_height_at=height_sampler.at,
                states_per_day=STATES_PER_DAY,
                observation_seconds_per_state=OBSERVATION_SECONDS_PER_STATE,
                truth_scope=TRUTH_SCOPE,
                local_date=item.local_date,
                active_in_scene=item.active_in_scene,
                burned_tree_ids_override=burned_ids,
                render_burned_tree_proxies=False,
                burned_surface_subdivisions=1,
            )
            record.update(
                {
                    "valid_at": item.valid_at,
                    "activity_method": item.activity_method,
                    "layer_revision_id": item.layer_revision_id,
                    "source_revision_ids": list(item.source_revision_ids),
                }
            )
            state_records.append(record)
            print(
                canonical_json(
                    {
                        "phase": "write_daily_state",
                        "state": record["state_id"],
                        "local_date": item.local_date,
                        "active_in_scene": item.active_in_scene,
                        "burned_tree_count": record["burned_tree_count"],
                    }
                ).strip(),
                flush=True,
            )
        propagation = {
            **result.model_metadata,
            "simulation_id": simulation_id,
            "domain_bounds_l93_m": list(domain),
            "ignition_l93_m": list(ignition),
            "drivers": drivers.as_dict(),
            "grid_shape": list(arrival_time_s.shape),
            "state_count": len(states),
            "states_per_day": STATES_PER_DAY,
            "observation_seconds_per_state": OBSERVATION_SECONDS_PER_STATE,
            "playback_seconds_per_day": PLAYBACK_SECONDS_PER_DAY,
            "states": state_metadata,
            "field_file": "propagation-field.npz",
        }
        propagation_path = output / "scenarios/propagation.json"
        propagation_path.write_text(canonical_json(propagation), encoding="utf-8")
        np.savez_compressed(
            output / "scenarios/propagation-field.npz",
            elevation_m=elevation,
            fuel_load=result.fuel_load,
            burnable_mask=result.burnable_mask,
            arrival_time_s=arrival_time_s,
            spread_rate_m_s=spread_rate_field,
            grid_x_l93_m=grid_coordinates[0],
            grid_y_l93_m=grid_coordinates[1],
            tree_burn_day=burn_day,
        )
        propagation_reference = {
            "path": "scenarios/propagation.json",
            "sha256": sha256_file(propagation_path),
            "simulation_id": simulation_id,
        }
        write_scenario_layer(
            output / "scenarios/scenario.usda",
            state_records,
            propagation_reference,
            truth_scope=TRUTH_SCOPE,
            state_selection_contract=STATE_SELECTION_CONTRACT,
        )
        write_flow_layer(
            output / "scenarios/flow.usda",
            result=result,
            anchor=anchor,
            seconds_per_state=PLAYBACK_SECONDS_PER_DAY,
            stepped_state_transitions=True,
        )
    finally:
        height_sampler.close()

    visual_profiles = write_accepted_visual_profile_artifacts(output)
    package_id = f"{base_package_id}-die-retrospective-v2"
    write_overlay_stage(output, package_id=package_id, anchor=anchor, base_package=base_package)
    cameras = list(base_manifest["cameras"]["plan"])
    capture_schedule = build_capture_schedule(
        package_id,
        state_records,
        cameras,
        incident_days=INCIDENT_DAYS,
        states_per_day=STATES_PER_DAY,
        incident_kind="retrospective-replay",
        observation_seconds_per_state=OBSERVATION_SECONDS_PER_STATE,
    )
    schedule_path = output / "runtime/capture-schedule.json"
    schedule_path.write_text(canonical_json(capture_schedule), encoding="utf-8")
    schedule_sha256 = sha256_file(schedule_path)
    shutil.copy2(Path(__file__).with_name("run_fireviewer_replicator_dataset.py"), output / "runtime")
    shutil.copy2(Path(__file__).with_name("fireviewer_replicator_writer.py"), output / "runtime")
    shutil.copy2(Path(__file__).with_name("fireviewer_capture_storage.py"), output / "runtime")
    runtime = _write_runtime_contract(
        output,
        capture_schedule=capture_schedule,
        schedule_sha256=schedule_sha256,
        camera_ids=[str(camera["camera_id"]) for camera in cameras],
        visual_profiles=visual_profiles,
    )
    base_locks = {
        "manifest": _locked_file(base_package / "manifest.json"),
        "entry_stage": _locked_file(base_package / "dataset.usda"),
        "site_layer": _locked_file(base_package / "site/site.usda"),
        "appearance_layer": _locked_file(base_package / "appearance/appearance.usda"),
        "camera_layer": _locked_file(base_package / "cameras/fixed_cameras.usda"),
        "environment_hdri": _locked_file(base_package / "assets/environments/farm_field_puresky_4k.hdr"),
    }
    manifest = {
        "schema": SCHEMA,
        "package_id": package_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_stage": "dataset.usda",
        "purpose": "local_omniverse_validation_only",
        "publication_allowed": False,
        "truth_scope": TRUTH_SCOPE,
        "coordinate_convention": "EPSG:2154 local anchor, Z-up metres, NGF-IGN69 elevations",
        "common_anchor_l93_m": list(anchor),
        "scene_bounds_l93_m": list(site_bounds),
        "base_scene": {
            "package_id": base_package_id,
            "composition": "exact_external_sublayer_references_no_scene_or_asset_rebuild",
            "path": str(base_package),
            "locks": base_locks,
        },
        "source_retrospective": {
            "dataset_id": SOURCE_DATASET_ID,
            "path": str(source),
            "sha256": source_sha256,
            "notice": retrospective.get("notice"),
            "geometry_origin": "mixed_satellite_product_and_agent_derived_daily_activity",
        },
        "scenario": {
            "incident_id": (
                f"{retrospective['incident']['fire_id']}/{retrospective['incident']['episode_id']}"
            ),
            "simulation_id": simulation_id,
            "incident_days": INCIDENT_DAYS,
            "states_per_day": STATES_PER_DAY,
            "state_count": len(state_records),
            "first_local_date": daily[0].local_date,
            "last_local_date": daily[-1].local_date,
            "playback_seconds_per_day": PLAYBACK_SECONDS_PER_DAY,
            "state_transition": "stepped_daily_hold",
            "propagation": propagation_reference,
            "states": state_records,
            "active_outside_scene_dates": [item.local_date for item in daily if not item.active_in_scene],
        },
        "vegetation": {
            "tree_instances": int(vegetation_index["counts"]["tree_instances"]),
            "prototype_assets": 6,
            "lod_policy": "none_all_detected_instances_resident",
            "source_rebuild_index": _locked_file(vegetation_index_path),
            "tree_destruction": "exact_base_point_instancer_ids_hidden_per_monotonic_daily_perimeter",
        },
        "cameras": base_manifest["cameras"],
        "environment": base_manifest["environment"],
        "dataset": {
            "capture_enabled": False,
            "capture_validation_status": "blocked_until_user_visual_acceptance",
            "capture_schedule": {"path": "runtime/capture-schedule.json", "sha256": schedule_sha256},
            "expected_viewpoint_plans": int(capture_schedule["expected_viewpoint_plans"]),
            "expected_capture_cases": int(capture_schedule["expected_capture_cases"]),
            "expected_positive_cases": int(capture_schedule["expected_positive_cases"]),
            "expected_negative_cases": int(capture_schedule["expected_negative_cases"]),
            "views_per_state": int(capture_schedule["views_per_state"]),
            "zooms_per_view": int(capture_schedule["zooms_per_view"]),
        },
        "runtime": {
            "contract": "runtime/runtime-contract.json",
            "contract_sha256": sha256_file(output / "runtime/runtime-contract.json"),
            "required_extensions": runtime["required_extensions"],
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
        "qa": {
            "automated": "qa/acceptance.json",
            "human_review": "qa/HUMAN_REVIEW.md",
            "render_acceptance": "pending_visible_native_kit_user_validation",
        },
    }
    (output / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    _write_qa(
        output,
        source_sha256=source_sha256,
        base_locks=base_locks,
        states=state_records,
        daily_metadata=state_metadata,
        tree_count=int(vegetation_index["counts"]["tree_instances"]),
    )
    print(
        canonical_json(
            {
                "status": "built",
                "package": str(output),
                "stage": str(output / "dataset.usda"),
                "base_scene": str(base_package),
                "state_count": len(state_records),
                "capture_enabled": False,
            }
        ),
        end="",
        flush=True,
    )
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-retrospective", type=Path, required=True)
    result.add_argument("--base-package", type=Path, required=True)
    result.add_argument("--source-map", type=Path, required=True)
    result.add_argument("--vegetation-index", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--replace-generated-output",
        action="store_true",
        help="Replace only an existing local overlay previously produced by this builder.",
    )
    return result


def main() -> int:
    build(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
