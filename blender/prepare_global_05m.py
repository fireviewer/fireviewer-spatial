"""Plan, execute and resume the tiled 0.5 m secteur synthétique fire production.

Planning and resuming are strictly offline.  Network and heavy raster work are
only reachable through the explicit ``--execute`` mode.  Elevation kilometre
tiles are cached once, outputs are committed atomically, and every 500 m tile
is accepted only after a receipt hashes all required products.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Sequence
from uuid import uuid4

from shapely.geometry import box, shape
from shapely.ops import unary_union


SCHEMA = "fireviewer.global-05m-production-manifest.v1"
RECEIPT_SCHEMA = "fireviewer.global-05m-tile-receipt.v1"
MID_PACKAGE_TERRAIN_CONTRACT = "exact-native-grid-500m-edges.v1"
CRS = "EPSG:2154"
DEFAULT_ORIGIN = (885_173.0, 6_404_926.0, 320.0)
PACKAGE_WORKER_MEMORY_GIB = 0.75
NEAR_ORTHOPHOTO_RESOLUTION_M = 0.2
NEAR_ORTHOPHOTO_WMS_TILE_PIXELS = 4_000
NEAR_ORTHOPHOTO_MAX_TILES_PER_RUN = 16

REQUIRED_TILE_ASSETS = (
    "mid_package",
    "orthophoto_source",
    "orthophoto_image",
    "orthophoto_geotiff",
)


def _ensure_parent_tool_path() -> None:
    parent = str(Path(__file__).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)


@dataclass(frozen=True)
class Global05mConfig:
    source_tile_size_m: int = 1_000
    output_tile_size_m: int = 500
    halo_m: float = 10.0
    elevation_resolution_m: float = 0.5
    terrain_sample_spacing_m: float = 1.0
    orthophoto_resolution_m: float = 0.5

    def validate(self) -> None:
        if self.source_tile_size_m != 1_000:
            raise ValueError(
                "source_tile_size_m must stay aligned to the IGN 1 km grid"
            )
        if (
            isinstance(self.output_tile_size_m, bool)
            or not isinstance(self.output_tile_size_m, int)
            or self.output_tile_size_m <= 0
            or self.source_tile_size_m % self.output_tile_size_m
        ):
            raise ValueError("output_tile_size_m must be a positive divisor of 1000")
        if (
            not math.isfinite(self.halo_m)
            or not 0 < self.halo_m < self.output_tile_size_m / 2
        ):
            raise ValueError(
                "halo_m must be positive and smaller than half an output tile"
            )
        for name in (
            "elevation_resolution_m",
            "terrain_sample_spacing_m",
            "orthophoto_resolution_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_geojson_geometry(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = (
        payload.get("features", [])
        if payload.get("type") == "FeatureCollection"
        else [payload]
    )
    geometries = []
    for feature in features:
        geometry = (
            feature.get("geometry") if feature.get("type") == "Feature" else feature
        )
        if geometry:
            parsed = shape(geometry)
            if not parsed.is_empty:
                geometries.append(parsed)
    if not geometries:
        raise ValueError(f"AOI contains no usable geometry: {path}")
    result = unary_union(geometries)
    if result.is_empty or not result.is_valid:
        raise ValueError(f"AOI geometry is empty or invalid: {path}")
    return result


def _grid_indices(
    bounds: Sequence[float], tile_size_m: int
) -> Iterable[tuple[int, int]]:
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    epsilon = 1e-9
    minimum_x = math.floor(min_x / tile_size_m)
    maximum_x = math.floor((max_x - epsilon) / tile_size_m)
    minimum_y = math.floor(min_y / tile_size_m)
    maximum_y = math.floor((max_y - epsilon) / tile_size_m)
    for y_index in range(minimum_y, maximum_y + 1):
        for x_index in range(minimum_x, maximum_x + 1):
            yield x_index, y_index


def _cell_bounds(x_index: int, y_index: int, tile_size_m: int) -> list[float]:
    return [
        float(x_index * tile_size_m),
        float(y_index * tile_size_m),
        float((x_index + 1) * tile_size_m),
        float((y_index + 1) * tile_size_m),
    ]


def ign_tile_id(x_index: int, south_y_index: int) -> str:
    """Return IGN's west/north tile id for a south-indexed grid cell."""

    return f"{x_index:04d}_{south_y_index + 1:04d}"


def ign_tile_bounds(tile_id: str) -> list[float]:
    east_text, north_text = tile_id.split("_", maxsplit=1)
    east_index = int(east_text)
    north_index = int(north_text)
    return [
        float(east_index * 1_000),
        float((north_index - 1) * 1_000),
        float((east_index + 1) * 1_000),
        float(north_index * 1_000),
    ]


def output_tile_id(bounds: Sequence[float], tile_size_m: int) -> str:
    min_x, min_y = (int(round(float(value))) for value in bounds[:2])
    return f"x{min_x:06d}_y{min_y:07d}_s{tile_size_m}"


def _raster_url(product: str, tile_id: str) -> str:
    _ensure_parent_tool_path()
    from build_detail_source_manifest import raster_url

    return raster_url(product, tile_id)


def _orthophoto_url(
    bounds: Sequence[float], resolution_m: float, *, tile_pixels: int = 2_000
) -> str:
    from fetch_ign_orthophoto import build_plan as build_orthophoto_plan

    plan = build_orthophoto_plan(
        bounds, resolution_m=resolution_m, tile_pixels=tile_pixels
    )
    if len(plan.tiles) != 1:
        raise ValueError("A 500 m orthophoto output must resolve to one WMS request")
    return plan.tiles[0].url


def _asset(path: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "path": path.replace("\\", "/"),
        "required": required,
        "exists": False,
        "byte_count": None,
        "mtime_ns": None,
        "sha256": None,
    }


def _source_tiles(aoi: Any, config: Global05mConfig) -> list[dict[str, Any]]:
    records = []
    for x_index, y_index in _grid_indices(aoi.bounds, config.source_tile_size_m):
        bounds = _cell_bounds(x_index, y_index, config.source_tile_size_m)
        intersection_area = float(aoi.intersection(box(*bounds)).area)
        if intersection_area <= 0:
            continue
        tile_id = ign_tile_id(x_index, y_index)
        records.append(
            {
                "id": tile_id,
                "bounds_l93_m": bounds,
                "aoi_intersection_area_m2": intersection_area,
                "aoi_coverage_ratio": intersection_area
                / float(config.source_tile_size_m**2),
                "assets": {
                    product: {
                        **_asset(
                            f"sources/{product}/LHD_FXX_{tile_id}_{product.upper()}_O_0M50_LAMB93_IGN69.tif"
                        ),
                        "url": _raster_url(product, tile_id),
                        "resolution_m": config.elevation_resolution_m,
                    }
                    for product in ("mnt", "mns")
                },
                "status": {"state": "pending"},
            }
        )
    return records


def _intersecting_source_ids(
    processing_bounds: Sequence[float], source_tiles: Sequence[dict[str, Any]]
) -> list[str]:
    processing = box(*processing_bounds)
    return sorted(
        tile["id"]
        for tile in source_tiles
        if processing.intersection(box(*tile["bounds_l93_m"])).area > 0
    )


def _output_tiles(
    aoi: Any,
    source_tiles: Sequence[dict[str, Any]],
    origin: Sequence[float],
    config: Global05mConfig,
) -> list[dict[str, Any]]:
    records = []
    for x_index, y_index in _grid_indices(aoi.bounds, config.output_tile_size_m):
        bounds = _cell_bounds(x_index, y_index, config.output_tile_size_m)
        intersection_area = float(aoi.intersection(box(*bounds)).area)
        if intersection_area <= 0:
            continue
        tile_id = output_tile_id(bounds, config.output_tile_size_m)
        processing_bounds = [
            bounds[0] - config.halo_m,
            bounds[1] - config.halo_m,
            bounds[2] + config.halo_m,
            bounds[3] + config.halo_m,
        ]
        directory = f"tiles/{tile_id}"
        records.append(
            {
                "id": tile_id,
                "bounds_l93_m": bounds,
                "processing_bounds_l93_m": processing_bounds,
                "origin_l93_m": [float(value) for value in origin],
                "aoi_intersection_area_m2": intersection_area,
                "aoi_coverage_ratio": intersection_area
                / float(config.output_tile_size_m**2),
                "source_tile_ids": _intersecting_source_ids(
                    processing_bounds, source_tiles
                ),
                "status": {
                    "state": "pending",
                    "attempt_count": 0,
                    "last_error": None,
                    "updated_at_utc": None,
                },
                "assets": {
                    "mid_package": _asset(f"{directory}/mid-vegetation-0m50.json.gz"),
                    "orthophoto_source": _asset(
                        f"{directory}/orthophoto-0m50.source.json"
                    ),
                    "orthophoto_image": _asset(f"{directory}/orthophoto-0m50.jpg"),
                    "orthophoto_geotiff": _asset(f"{directory}/orthophoto-0m50.tif"),
                    # The native 20 cm image is deliberately optional. Existing
                    # 50 cm receipts stay valid and a close-view tile can be
                    # upgraded independently without republishing all 475 tiles.
                    "near_orthophoto_source": _asset(
                        f"{directory}/orthophoto-0m20.source.json", required=False
                    ),
                    "near_orthophoto_image": _asset(
                        f"{directory}/orthophoto-0m20.jpg", required=False
                    ),
                    "near_orthophoto_geotiff": _asset(
                        f"{directory}/orthophoto-0m20.tif", required=False
                    ),
                    "blender_library": _asset(
                        f"{directory}/scene.blend", required=False
                    ),
                    "completion_receipt": _asset(
                        f"{directory}/tile.done.json", required=False
                    ),
                },
                "orthophoto_request": {
                    "url": _orthophoto_url(bounds, config.orthophoto_resolution_m),
                    "resolution_m": config.orthophoto_resolution_m,
                    "wms_tile_pixels": 2_000,
                    "pixel_size": [
                        int(config.output_tile_size_m / config.orthophoto_resolution_m),
                        int(config.output_tile_size_m / config.orthophoto_resolution_m),
                    ],
                },
                "near_orthophoto_request": {
                    "url": _orthophoto_url(
                        bounds,
                        NEAR_ORTHOPHOTO_RESOLUTION_M,
                        tile_pixels=NEAR_ORTHOPHOTO_WMS_TILE_PIXELS,
                    ),
                    "resolution_m": NEAR_ORTHOPHOTO_RESOLUTION_M,
                    "wms_tile_pixels": NEAR_ORTHOPHOTO_WMS_TILE_PIXELS,
                    "pixel_size": [
                        int(config.output_tile_size_m / NEAR_ORTHOPHOTO_RESOLUTION_M),
                        int(config.output_tile_size_m / NEAR_ORTHOPHOTO_RESOLUTION_M),
                    ],
                    "loading": "explicit_selected_near_tiles_only",
                    "maximum_tiles_per_run": NEAR_ORTHOPHOTO_MAX_TILES_PER_RUN,
                },
                "visibility": {
                    "default_visible": False,
                    "activation": "camera_or_selected_attention_zone",
                    "lod": "mid_0m50",
                },
            }
        )
    return records


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    source_tiles = manifest["source_tiles"]
    output_tiles = manifest["tiles"]
    states: dict[str, int] = {}
    for tile in output_tiles:
        state = tile["status"]["state"]
        states[state] = states.get(state, 0) + 1
    return {
        "source_tile_count": len(source_tiles),
        "output_tile_count": len(output_tiles),
        "output_states": states,
        "elevation_source_request_count": len(source_tiles) * 2,
        "orthophoto_request_count": len(output_tiles),
        "network_access_performed": False,
    }


def build_plan(
    aoi_path: Path,
    output_root: Path,
    *,
    origin: Sequence[float] = DEFAULT_ORIGIN,
    config: Global05mConfig | None = None,
    expected_source_tile_count: int | None = None,
    source_only: bool = False,
) -> dict[str, Any]:
    active = config or Global05mConfig()
    active.validate()
    if len(origin) != 3 or not all(math.isfinite(float(value)) for value in origin):
        raise ValueError("origin must contain three finite coordinates")
    source_file = aoi_path.resolve()
    aoi = _read_geojson_geometry(source_file)
    sources = _source_tiles(aoi, active)
    if (
        expected_source_tile_count is not None
        and len(sources) != expected_source_tile_count
    ):
        raise ValueError(
            f"AOI intersects {len(sources)} source tiles; expected {expected_source_tile_count}"
        )
    # A global-only delivery consumes no 500 m detail packages. Avoiding this
    # unused grid keeps a large-area plan bounded by the 1 km source tiles.
    tiles = [] if source_only else _output_tiles(aoi, sources, origin, active)
    immutable_identity = {
        "aoi_sha256": sha256_file(source_file),
        "config": asdict(active),
        "origin_l93_m": [float(value) for value in origin],
        "source_tile_ids": [tile["id"] for tile in sources],
        "output_tile_ids": [tile["id"] for tile in tiles],
    }
    manifest = {
        "schema": SCHEMA,
        "plan_id": _canonical_sha256(immutable_identity),
        "generated_at_utc": utc_now(),
        "updated_at_utc": None,
        "status": "planned",
        "crs": CRS,
        "axis_convention": "X=east, Y=north, Z=up",
        "linear_unit": "metre",
        "origin_l93_m": [float(value) for value in origin],
        "aoi": {
            "file_name": source_file.name,
            "sha256": immutable_identity["aoi_sha256"],
            "bounds_l93_m": [float(value) for value in aoi.bounds],
            "area_m2": float(aoi.area),
        },
        "tiling": {
            **asdict(active),
            "source_grid": "IGN_LIDAR_HD_1KM_WEST_NORTH_ID",
            "ownership_rule": "apex_in_half_open_core_min_inclusive_max_exclusive",
            "terrain_rule": ("exact_500m_core_grid_sampled_on_native_ign_pixel_phase"),
            "terrain_edge_contract": MID_PACKAGE_TERRAIN_CONTRACT,
            "segmentation_rule": "process_core_plus_halo_then_keep_owned_apices",
            "aoi_rule": "mask_to_fire_plus_1m5km_polygon",
            "detail_grid_planned": not source_only,
        },
        "output_root_name": output_root.resolve().name,
        "path_policy": "relative_to_manifest_directory",
        "network_policy": {
            "planner_network_access": "none",
            "source_downloads_must_be_explicit": True,
            "partial_download_suffix": ".part",
        },
        "worker_contract": {
            "source_products": ["mnt", "mns", "orthophoto"],
            "vegetation_source": "MNS_minus_MNT_at_0m50",
            "post_detection_thinning": "forbidden",
            "required_tile_assets": list(REQUIRED_TILE_ASSETS),
            "completion_receipt_schema": RECEIPT_SCHEMA,
            "completion_rule": "receipt_hashes_all_required_assets",
            "mid_package_terrain_contract": MID_PACKAGE_TERRAIN_CONTRACT,
            "orthophoto_display": {
                "source_geotiff_transform": "none",
                "jpeg_quality": 92,
                "jpeg_brightness": 0.78,
                "jpeg_contrast": 1.08,
                "jpeg_saturation": 1.12,
            },
            "orthophoto_lod": {
                "far_global_resolution_m": 2.0,
                "mid_tile_resolution_m": active.orthophoto_resolution_m,
                "near_tile_resolution_m": NEAR_ORTHOPHOTO_RESOLUTION_M,
                "near_loading": "optional_explicit_selected_tiles_only",
                "near_fallback": "mid_tile_orthophoto",
            },
            "default_execution_limits": {
                "download_workers": 2,
                "package_workers": 1,
                "package_worker_memory_gib": PACKAGE_WORKER_MEMORY_GIB,
                "minimum_free_gib": 20.0,
            },
        },
        "source_tiles": sources,
        "tiles": tiles,
    }
    manifest["summary"] = _summary(manifest)
    return manifest


def _atomic_write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _refresh_asset(asset: dict[str, Any], root: Path) -> None:
    path = root / asset["path"]
    if not path.is_file():
        asset.update(
            {"exists": False, "byte_count": None, "mtime_ns": None, "sha256": None}
        )
        return
    stat = path.stat()
    unchanged = (
        asset.get("exists") is True
        and asset.get("byte_count") == stat.st_size
        and asset.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(asset.get("sha256"), str)
    )
    asset.update(
        {
            "exists": True,
            "byte_count": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": asset["sha256"] if unchanged else sha256_file(path),
        }
    )


def _validate_receipt(tile: dict[str, Any], root: Path) -> tuple[str, str | None]:
    receipt_asset = tile["assets"]["completion_receipt"]
    _refresh_asset(receipt_asset, root)
    if not receipt_asset["exists"]:
        return "pending", None
    receipt_path = root / receipt_asset["path"]
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "incomplete", f"invalid completion receipt: {exc}"
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("tile_id") != tile["id"]:
        return "incomplete", "completion receipt schema or tile id mismatch"
    recorded_outputs = receipt.get("outputs")
    if not isinstance(recorded_outputs, dict):
        return "incomplete", "completion receipt has no output hash mapping"
    for name in REQUIRED_TILE_ASSETS:
        asset = tile["assets"][name]
        _refresh_asset(asset, root)
        if not asset["exists"]:
            return "incomplete", f"required asset is absent: {name}"
        record = recorded_outputs.get(name)
        if not isinstance(record, dict) or record.get("sha256") != asset["sha256"]:
            return "incomplete", f"required asset hash mismatch: {name}"
    optional = tile["assets"]["blender_library"]
    _refresh_asset(optional, root)
    return "ready", None


def refresh_manifest(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(
            f"unsupported production manifest schema: {manifest.get('schema')!r}"
        )
    for source in manifest["source_tiles"]:
        for asset in source["assets"].values():
            _refresh_asset(asset, root)
        source["status"]["state"] = (
            "ready"
            if all(
                (not asset.get("required", True)) or asset["exists"]
                for asset in source["assets"].values()
            )
            else "pending"
        )
    for tile in manifest["tiles"]:
        state, error = _validate_receipt(tile, root)
        if state == "pending" and tile["status"].get("state") == "failed":
            state = "failed"
            error = tile["status"].get("last_error")
        elif state == "pending" and tile["status"].get("state") == "running":
            # A process can be interrupted after persisting its running state but
            # before it writes the receipt.  The output is then absent and the
            # tile must become selectable again on the next resume rather than
            # being stranded forever in a non-selectable running state.
            error = "interrupted before completion receipt; eligible for resume"
        tile["status"].update(
            {"state": state, "last_error": error, "updated_at_utc": utc_now()}
        )
    manifest["updated_at_utc"] = utc_now()
    manifest["summary"] = _summary(manifest)
    ready = manifest["summary"]["output_states"].get("ready", 0)
    manifest["status"] = "ready" if ready == len(manifest["tiles"]) else "in_progress"
    return manifest


def completion_receipt(
    tile: dict[str, Any], root: Path, *, producer: str
) -> dict[str, Any]:
    outputs = {}
    for name in REQUIRED_TILE_ASSETS:
        asset = tile["assets"][name]
        _refresh_asset(asset, root)
        if not asset["exists"]:
            raise FileNotFoundError(f"cannot complete {tile['id']}; missing {name}")
        outputs[name] = {
            "path": asset["path"],
            "byte_count": asset["byte_count"],
            "sha256": asset["sha256"],
        }
    return {
        "schema": RECEIPT_SCHEMA,
        "tile_id": tile["id"],
        "mid_package_terrain_contract": MID_PACKAGE_TERRAIN_CONTRACT,
        "completed_at_utc": utc_now(),
        "producer": producer,
        "bounds_l93_m": tile["bounds_l93_m"],
        "processing_bounds_l93_m": tile["processing_bounds_l93_m"],
        "outputs": outputs,
    }


@contextmanager
def _asset_lock(target: Path, *, timeout_s: float = 3_600.0) -> Iterable[None]:
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(f".{target.name}.lock")
    started = time.monotonic()
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock.stat().st_mtime > 6 * 60 * 60
            except FileNotFoundError:
                continue
            owner_alive = _lock_owner_is_alive(lock)
            if stale or owner_alive is False:
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() - started >= timeout_s:
                raise TimeoutError(f"timed out waiting for production lock: {lock}")
            time.sleep(0.25)
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def _lock_owner_is_alive(lock: Path) -> bool | None:
    """Return whether the PID recorded by a production lock still exists.

    ``None`` deliberately means "unknown" so malformed or unreadable locks
    retain the conservative six-hour expiry policy.  A hard-killed production
    process otherwise leaves every active source locked for six hours, which
    defeats the resumable download contract.
    """

    try:
        owner = lock.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if not owner.startswith("pid=") or not owner[4:].isdigit():
        return None
    pid = int(owner[4:])
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # CPython reports an absent Windows process as ERROR_INVALID_PARAMETER
        # rather than ProcessLookupError when probing with signal 0.
        if os.name == "nt" and getattr(exc, "winerror", None) == 87:
            return False
        return None
    return True


def ensure_elevation_source(
    source: dict[str, Any],
    product: str,
    root: Path,
    *,
    timeout_s: float = 180.0,
    fetcher: Any | None = None,
    validator: Any | None = None,
) -> dict[str, Any]:
    """Download one shared MNT/MNS source and commit it after validation."""

    from fetch_ign_orthophoto import _download

    _ensure_parent_tool_path()
    from build_detail_source_manifest import inspect_raster

    active_fetcher = fetcher or _download
    active_validator = validator or inspect_raster
    asset = source["assets"][product]
    destination = root / asset["path"]
    sources_root = root / "sources"
    with _asset_lock(destination, timeout_s=max(timeout_s * 4, 300.0)):
        if destination.is_file():
            try:
                record = active_validator(
                    destination, sources_root, product, source["id"]
                )
                return {"cache": "hit", "validation": record}
            except Exception:
                destination.unlink(missing_ok=True)
        partial = destination.with_name(f".{destination.name}.part")
        partial.unlink(missing_ok=True)
        try:
            active_fetcher(asset["url"], partial, timeout_s)
            record = active_validator(partial, sources_root, product, source["id"])
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    return {"cache": "downloaded", "validation": record}


def ensure_near_orthophoto_contract(tile: dict[str, Any]) -> dict[str, Any]:
    """Add the optional native-resolution asset contract to an existing tile.

    Production manifests created before the near profile must remain usable.
    This additive migration does not alter the required 50 cm assets or their
    completion receipt.
    """

    assets = tile.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("Production tile assets must be an object")
    mid = assets.get("mid_package")
    mid_path = mid.get("path") if isinstance(mid, dict) else None
    if not isinstance(mid_path, str) or not mid_path:
        raise ValueError("Production tile has no portable mid package path")
    directory = Path(mid_path).parent.as_posix()
    expected_assets = {
        "near_orthophoto_source": _asset(
            f"{directory}/orthophoto-0m20.source.json", required=False
        ),
        "near_orthophoto_image": _asset(
            f"{directory}/orthophoto-0m20.jpg", required=False
        ),
        "near_orthophoto_geotiff": _asset(
            f"{directory}/orthophoto-0m20.tif", required=False
        ),
    }
    for key, expected in expected_assets.items():
        current = assets.setdefault(key, expected)
        if not isinstance(current, dict) or not isinstance(current.get("path"), str):
            raise ValueError(f"Production tile {key} asset is invalid")

    request = tile.get("near_orthophoto_request")
    expected_pixels = int(
        round(
            (float(tile["bounds_l93_m"][2]) - float(tile["bounds_l93_m"][0]))
            / NEAR_ORTHOPHOTO_RESOLUTION_M
        )
    )
    if request is None:
        tile["near_orthophoto_request"] = {
            "url": _orthophoto_url(
                tile["bounds_l93_m"],
                NEAR_ORTHOPHOTO_RESOLUTION_M,
                tile_pixels=NEAR_ORTHOPHOTO_WMS_TILE_PIXELS,
            ),
            "resolution_m": NEAR_ORTHOPHOTO_RESOLUTION_M,
            "wms_tile_pixels": NEAR_ORTHOPHOTO_WMS_TILE_PIXELS,
            "pixel_size": [expected_pixels, expected_pixels],
            "loading": "explicit_selected_near_tiles_only",
            "maximum_tiles_per_run": NEAR_ORTHOPHOTO_MAX_TILES_PER_RUN,
        }
    elif not isinstance(request, dict) or not math.isclose(
        float(request.get("resolution_m", math.nan)),
        NEAR_ORTHOPHOTO_RESOLUTION_M,
        abs_tol=1e-12,
    ):
        raise ValueError("Near orthophoto request must use the native 0.20 m profile")
    return tile


def _orthophoto_profile(
    tile: dict[str, Any], *, near: bool
) -> tuple[str, str, str, dict[str, Any]]:
    if near:
        ensure_near_orthophoto_contract(tile)
        prefix = "near_orthophoto"
        request_key = "near_orthophoto_request"
    else:
        prefix = "orthophoto"
        request_key = "orthophoto_request"
    request = tile.get(request_key)
    if not isinstance(request, dict):
        raise ValueError(f"Production tile has no {request_key} contract")
    return (
        f"{prefix}_source",
        f"{prefix}_image",
        f"{prefix}_geotiff",
        request,
    )


def validate_orthophoto_tile(
    tile: dict[str, Any], root: Path, *, near: bool = False
) -> dict[str, Any]:
    from build_control_scene import load_orthophoto_source

    source_key, image_key, geotiff_key, request_contract = _orthophoto_profile(
        tile, near=near
    )
    source_path = root / tile["assets"][source_key]["path"]
    image_path, bounds, payload = load_orthophoto_source(source_path)
    if image_path != root / tile["assets"][image_key]["path"]:
        raise ValueError("Orthophoto source points to an unexpected JPEG")
    if any(
        abs(float(left) - float(right)) > 1e-6
        for left, right in zip(bounds, tile["bounds_l93_m"], strict=True)
    ):
        raise ValueError("Orthophoto bounds do not match the production tile")
    observed_resolution = payload.get("request", {}).get("nominal_resolution_m")
    expected_resolution = float(request_contract["resolution_m"])
    if not isinstance(observed_resolution, (int, float)) or not math.isclose(
        float(observed_resolution), expected_resolution, abs_tol=1e-12
    ):
        raise ValueError(
            "Orthophoto nominal resolution does not match the production profile"
        )
    geotiff = root / tile["assets"][geotiff_key]["path"]
    output = next(
        (
            item
            for item in payload.get("outputs", [])
            if item.get("role") == "lambert93_geotiff_blender_texture"
        ),
        None,
    )
    if output is None or output.get("file_name") != geotiff.name:
        raise ValueError("Orthophoto source has no expected GeoTIFF")
    if not geotiff.is_file() or sha256_file(geotiff) != output.get("sha256"):
        raise ValueError("Orthophoto GeoTIFF is absent or its hash does not match")
    return payload


def ensure_orthophoto_tile(
    tile: dict[str, Any],
    root: Path,
    *,
    timeout_s: float = 180.0,
    fetcher: Any | None = None,
) -> dict[str, Any]:
    """Build a validated orthophoto cache in a sibling work directory."""

    return _ensure_orthophoto_tile(
        tile,
        root,
        near=False,
        timeout_s=timeout_s,
        fetcher=fetcher,
    )


def ensure_near_orthophoto_tile(
    tile: dict[str, Any],
    root: Path,
    *,
    timeout_s: float = 180.0,
    fetcher: Any | None = None,
) -> dict[str, Any]:
    """Build one optional native 20 cm cache for an explicitly selected tile."""

    ensure_near_orthophoto_contract(tile)
    return _ensure_orthophoto_tile(
        tile,
        root,
        near=True,
        timeout_s=timeout_s,
        fetcher=fetcher,
    )


def _ensure_orthophoto_tile(
    tile: dict[str, Any],
    root: Path,
    *,
    near: bool,
    timeout_s: float,
    fetcher: Any | None,
) -> dict[str, Any]:
    """Shared atomic cache implementation for the 50 cm and 20 cm profiles."""

    from fetch_ign_orthophoto import build_plan as build_orthophoto_plan
    from fetch_ign_orthophoto import execute_plan as execute_orthophoto_plan

    source_key, image_key, geotiff_key, request_contract = _orthophoto_profile(
        tile, near=near
    )
    source_path = root / tile["assets"][source_key]["path"]
    try:
        return {
            "cache": "hit",
            "source": validate_orthophoto_tile(tile, root, near=near),
        }
    except (FileNotFoundError, OSError, ValueError):
        pass
    with _asset_lock(source_path, timeout_s=max(timeout_s * 4, 300.0)):
        try:
            return {
                "cache": "hit",
                "source": validate_orthophoto_tile(tile, root, near=near),
            }
        except (FileNotFoundError, OSError, ValueError):
            pass
        directory = source_path.parent
        work_prefix = ".near-orthophoto-work" if near else ".orthophoto-work"
        work = directory / f"{work_prefix}-{uuid4().hex}"
        work.mkdir(parents=True, exist_ok=False)
        geotiff_name = Path(tile["assets"][geotiff_key]["path"]).name
        jpeg_name = Path(tile["assets"][image_key]["path"]).name
        work_geotiff = work / geotiff_name
        work_jpeg = work / jpeg_name
        try:
            plan = build_orthophoto_plan(
                tile["bounds_l93_m"],
                resolution_m=float(request_contract["resolution_m"]),
                tile_pixels=int(request_contract.get("wms_tile_pixels", 2_000)),
            )
            arguments: dict[str, Any] = {
                "jpeg_output": work_jpeg,
                "jpeg_quality": 92,
                "jpeg_brightness": 0.78,
                "jpeg_contrast": 1.08,
                "jpeg_saturation": 1.12,
                "timeout_s": timeout_s,
            }
            if fetcher is not None:
                arguments["fetcher"] = fetcher
            execute_orthophoto_plan(plan, work_geotiff, **arguments)
            work_source = work_geotiff.with_suffix(".source.json")
            work_world = work_jpeg.with_suffix(".jgw")
            final_geotiff = root / tile["assets"][geotiff_key]["path"]
            final_jpeg = root / tile["assets"][image_key]["path"]
            final_world = final_jpeg.with_suffix(".jgw")
            directory.mkdir(parents=True, exist_ok=True)
            for temporary, final in (
                (work_geotiff, final_geotiff),
                (work_jpeg, final_jpeg),
                (work_world, final_world),
                (work_source, source_path),
            ):
                temporary.replace(final)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return {
        "cache": "downloaded",
        "source": validate_orthophoto_tile(tile, root, near=near),
    }


def validate_mid_package(tile: dict[str, Any], root: Path) -> dict[str, Any]:
    from build_control_scene import (
        _validate_detail_terrain_core_contract,
        load_mid_vegetation_package,
    )

    path = root / tile["assets"]["mid_package"]["path"]
    package = load_mid_vegetation_package(path)
    metadata = package["metadata"]
    for key in ("bounds_l93_m", "processing_bounds_l93_m", "origin_l93_m"):
        expected = tile["origin_l93_m" if key == "origin_l93_m" else key]
        observed = metadata.get(key)
        if not isinstance(observed, list) or any(
            abs(float(left) - float(right)) > 0.001
            for left, right in zip(observed, expected, strict=True)
        ):
            raise ValueError(f"Mid-distance package {key} does not match its tile")
    if package["statistics"].get("post_detection_spacing_rejected_count") != 0:
        raise ValueError("Mid-distance package applied forbidden tree thinning")
    _validate_detail_terrain_core_contract(tile, package["terrain"])
    return package


def _package_job(job: dict[str, Any]) -> dict[str, Any]:
    from prepare_mid_vegetation_05m import build_sector_package, write_sector_package

    tile = job["tile"]
    root = Path(job["root"])
    arguments = argparse.Namespace(
        mnt=[Path(path) for path in job["mnt"]],
        mns=[Path(path) for path in job["mns"]],
        bounds=tile["bounds_l93_m"],
        processing_bounds=tile["processing_bounds_l93_m"],
        include_polygons=[Path(job["aoi_path"])],
        exclude_polygons=[Path(path) for path in job["exclude_polygons"]],
        exclude_lines=[Path(path) for path in job["exclude_lines"]],
        polygon_clearance_m=2.0,
        line_half_width_m=4.0,
        min_tree_height_m=2.0,
        local_peak_radius_m=1.0,
        smoothing_sigma_m=0.5,
        terrain_step_pixels=2,
        origin_x=float(tile["origin_l93_m"][0]),
        origin_y=float(tile["origin_l93_m"][1]),
        origin_z=float(tile["origin_l93_m"][2]),
    )
    package = build_sector_package(arguments)
    output = root / tile["assets"]["mid_package"]["path"]
    write_sector_package(package, output)
    validated = validate_mid_package(tile, root)
    return {
        "tile_id": tile["id"],
        "statistics": validated["statistics"],
        "sha256": sha256_file(output),
    }


def _selected_tiles(
    manifest: dict[str, Any],
    tile_ids: Sequence[str],
    *,
    max_tiles: int | None,
    retry_failed: bool,
    include_ready: bool = False,
) -> list[dict[str, Any]]:
    requested = set(tile_ids)
    known = {tile["id"] for tile in manifest["tiles"]}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown production tile id(s): {', '.join(unknown)}")
    result = [
        tile
        for tile in manifest["tiles"]
        if (not requested or tile["id"] in requested)
        and (include_ready or tile["status"]["state"] != "ready")
        and (retry_failed or tile["status"]["state"] != "failed")
    ]
    return result[:max_tiles] if max_tiles is not None else result


def _write_runtime_manifest(
    manifest_path: Path, manifest: dict[str, Any], root: Path
) -> None:
    manifest["updated_at_utc"] = utc_now()
    manifest["summary"] = _summary(manifest)
    _atomic_write_json(manifest_path, manifest, overwrite=True)


def execute_manifest(
    manifest_path: Path,
    aoi_path: Path,
    *,
    exclude_polygons: Sequence[Path] = (),
    exclude_lines: Sequence[Path] = (),
    tile_ids: Sequence[str] = (),
    phase: str = "all",
    download_workers: int = 2,
    package_workers: int = 1,
    memory_budget_gib: float = 4.0,
    minimum_free_gib: float = 20.0,
    max_tiles: int | None = None,
    timeout_s: float = 180.0,
    retry_failed: bool = False,
    rebuild_mid_packages: bool = False,
    continue_on_error: bool = False,
    required_elevation_resolution_m: float | None = None,
    source_fetcher: Any | None = None,
    source_validator: Any | None = None,
    orthophoto_fetcher: Any | None = None,
    package_worker: Any = _package_job,
) -> dict[str, Any]:
    if phase not in {"all", "sources", "tiles"}:
        raise ValueError("phase must be one of: all, sources, tiles")
    if rebuild_mid_packages and phase != "tiles":
        raise ValueError("rebuild_mid_packages requires phase='tiles'")
    if not 1 <= download_workers <= 4:
        raise ValueError("download_workers must be between 1 and 4")
    if not 1 <= package_workers <= 4:
        raise ValueError("package_workers must be between 1 and 4")
    if package_workers * PACKAGE_WORKER_MEMORY_GIB > memory_budget_gib:
        raise ValueError(
            "package worker count exceeds the declared memory budget "
            f"({PACKAGE_WORKER_MEMORY_GIB:.2f} GiB planning allowance per worker)"
        )
    if max_tiles is not None and max_tiles <= 0:
        raise ValueError("max_tiles must be strictly positive")
    root = manifest_path.resolve().parent
    free_gib = shutil.disk_usage(root).free / (1024**3)
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB are free; {minimum_free_gib:.2f} GiB required"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if required_elevation_resolution_m is not None:
        actual_resolution = manifest.get("tiling", {}).get("elevation_resolution_m")
        if not isinstance(actual_resolution, (int, float)) or not math.isclose(
            float(actual_resolution),
            float(required_elevation_resolution_m),
            abs_tol=1e-12,
        ):
            raise ValueError(
                "production manifest elevation source resolution does not satisfy "
                f"the required native LiDAR resolution: expected "
                f"{float(required_elevation_resolution_m):g} m, got {actual_resolution!r}"
            )
    if rebuild_mid_packages:
        manifest.setdefault("worker_contract", {})["mid_package_terrain_contract"] = (
            MID_PACKAGE_TERRAIN_CONTRACT
        )
        manifest.setdefault("tiling", {})["terrain_edge_contract"] = (
            MID_PACKAGE_TERRAIN_CONTRACT
        )
    source_file = aoi_path.resolve()
    if sha256_file(source_file) != manifest["aoi"]["sha256"]:
        raise ValueError("AOI hash does not match the production manifest")
    refresh_manifest(manifest, root)
    tiles = _selected_tiles(
        manifest,
        tile_ids,
        max_tiles=max_tiles,
        retry_failed=retry_failed,
        include_ready=rebuild_mid_packages,
    )
    source_by_id = {source["id"]: source for source in manifest["source_tiles"]}
    required_source_ids = sorted(
        {source_id for tile in tiles for source_id in tile["source_tile_ids"]}
    )
    if phase == "sources" and not manifest.get("tiling", {}).get(
        "detail_grid_planned", True
    ):
        # A global-only manifest deliberately contains no 500 m detail tiles.
        # Its source phase must still materialize every LiDAR source tile.
        required_source_ids = sorted(source_by_id)
    errors: list[dict[str, str]] = []
    counters = {
        "selected_tile_count": len(tiles),
        "elevation_downloaded_or_validated": 0,
        "orthophoto_downloaded_or_validated": 0,
        "packages_completed": 0,
        "rebuild_mid_packages": rebuild_mid_packages,
    }

    if phase in {"all", "sources"}:
        tasks = [
            (source_by_id[source_id], product)
            for source_id in required_source_ids
            for product in ("mnt", "mns")
        ]
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = {
                executor.submit(
                    ensure_elevation_source,
                    source,
                    product,
                    root,
                    timeout_s=timeout_s,
                    fetcher=source_fetcher,
                    validator=source_validator,
                ): (source, product)
                for source, product in tasks
            }
            for future in as_completed(futures):
                source, product = futures[future]
                try:
                    future.result()
                    counters["elevation_downloaded_or_validated"] += 1
                except Exception as exc:
                    errors.append(
                        {
                            "scope": f"source:{source['id']}:{product}",
                            "error": str(exc),
                        }
                    )
                    if not continue_on_error:
                        for pending in futures:
                            pending.cancel()
                        break
        refresh_manifest(manifest, root)
        _write_runtime_manifest(manifest_path, manifest, root)
        if errors and not continue_on_error:
            raise RuntimeError(errors[0]["error"])

    if phase in {"all", "tiles"}:
        unavailable = [
            source_id
            for source_id in required_source_ids
            if source_by_id[source_id]["status"]["state"] != "ready"
        ]
        if unavailable and not continue_on_error:
            raise RuntimeError(
                "required elevation sources are absent: " + ", ".join(unavailable)
            )
        if unavailable:
            _write_runtime_manifest(manifest_path, manifest, root)
        successful_tiles: list[dict[str, Any]] = []
        if rebuild_mid_packages:
            # This maintenance mode is deliberately offline for imagery: it
            # validates the required 0.5 m texture but never invokes a fetcher,
            # and it never reads or mutates the optional 0.2 m near assets.
            for tile in tiles:
                try:
                    validate_orthophoto_tile(tile, root, near=False)
                    successful_tiles.append(tile)
                    counters["orthophoto_downloaded_or_validated"] += 1
                except Exception as exc:
                    tile["status"].update(
                        {
                            "state": "failed",
                            "last_error": f"orthophoto: {exc}",
                            "updated_at_utc": utc_now(),
                        }
                    )
                    errors.append(
                        {"scope": f"tile:{tile['id']}:orthophoto", "error": str(exc)}
                    )
                    _write_runtime_manifest(manifest_path, manifest, root)
                    if not continue_on_error:
                        break
        else:
            with ThreadPoolExecutor(max_workers=download_workers) as executor:
                futures = {
                    executor.submit(
                        ensure_orthophoto_tile,
                        tile,
                        root,
                        timeout_s=timeout_s,
                        fetcher=orthophoto_fetcher,
                    ): tile
                    for tile in tiles
                }
                for future in as_completed(futures):
                    tile = futures[future]
                    try:
                        future.result()
                        successful_tiles.append(tile)
                        counters["orthophoto_downloaded_or_validated"] += 1
                    except Exception as exc:
                        tile["status"].update(
                            {
                                "state": "failed",
                                "last_error": f"orthophoto: {exc}",
                                "updated_at_utc": utc_now(),
                            }
                        )
                        errors.append(
                            {
                                "scope": f"tile:{tile['id']}:orthophoto",
                                "error": str(exc),
                            }
                        )
                        _write_runtime_manifest(manifest_path, manifest, root)
                        if not continue_on_error:
                            for pending in futures:
                                pending.cancel()
                            break
        if errors and not continue_on_error:
            raise RuntimeError(errors[0]["error"])

        jobs = []
        for tile in successful_tiles:
            mnt_pairs: list[str] = []
            mns_pairs: list[str] = []
            missing_pairs = []
            for source_id in tile["source_tile_ids"]:
                source = source_by_id[source_id]
                mnt_path = root / source["assets"]["mnt"]["path"]
                mns_path = root / source["assets"]["mns"]["path"]
                if mnt_path.is_file() and mns_path.is_file():
                    mnt_pairs.append(str(mnt_path))
                    mns_pairs.append(str(mns_path))
                else:
                    missing_pairs.append(source_id)
            if missing_pairs and not mnt_pairs:
                tile["status"].update(
                    {
                        "state": "failed",
                        "last_error": (
                            "missing complete elevation source pair for tile and no "
                            f"ready fallback pairs available: {', '.join(missing_pairs)}"
                        ),
                        "updated_at_utc": utc_now(),
                    }
                )
                errors.append(
                    {
                        "scope": f"tile:{tile['id']}:package",
                        "error": tile["status"]["last_error"],
                    }
                )
                _write_runtime_manifest(manifest_path, manifest, root)
                if not continue_on_error:
                    break
                continue
            if missing_pairs and continue_on_error:
                tile["status"].update(
                    {
                        "last_warning": (
                            "elevation sources missing but ignored for missing pairs: "
                            + ", ".join(missing_pairs)
                        )
                    }
                )
            tile["status"].update(
                {
                    "state": "running",
                    "attempt_count": int(tile["status"].get("attempt_count", 0)) + 1,
                    "last_error": None,
                    "updated_at_utc": utc_now(),
                }
            )
            jobs.append(
                {
                    "tile": tile,
                    "root": str(root),
                    "aoi_path": str(source_file),
                    "exclude_polygons": [
                        str(path.resolve()) for path in exclude_polygons
                    ],
                    "exclude_lines": [str(path.resolve()) for path in exclude_lines],
                    "mnt": mnt_pairs,
                    "mns": mns_pairs,
                }
            )
        _write_runtime_manifest(manifest_path, manifest, root)

        def accept(job: dict[str, Any], result: dict[str, Any]) -> None:
            tile = job["tile"]
            receipt = completion_receipt(tile, root, producer="prepare_global_05m.py")
            receipt_path = root / tile["assets"]["completion_receipt"]["path"]
            _atomic_write_json(receipt_path, receipt, overwrite=True)
            state, error = _validate_receipt(tile, root)
            if state != "ready":
                raise RuntimeError(error or "completion receipt validation failed")
            tile["status"].update(
                {"state": "ready", "last_error": None, "updated_at_utc": utc_now()}
            )
            tile["production_statistics"] = result["statistics"]

        if package_workers == 1:
            for job in jobs:
                try:
                    accept(job, package_worker(job))
                    counters["packages_completed"] += 1
                except Exception as exc:
                    job["tile"]["status"].update(
                        {
                            "state": "failed",
                            "last_error": f"package: {exc}",
                            "updated_at_utc": utc_now(),
                        }
                    )
                    errors.append(
                        {
                            "scope": f"tile:{job['tile']['id']}:package",
                            "error": str(exc),
                        }
                    )
                    if not continue_on_error:
                        _write_runtime_manifest(manifest_path, manifest, root)
                        raise
                _write_runtime_manifest(manifest_path, manifest, root)
        else:
            with ProcessPoolExecutor(max_workers=package_workers) as executor:
                futures = {executor.submit(package_worker, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        accept(job, future.result())
                        counters["packages_completed"] += 1
                    except Exception as exc:
                        job["tile"]["status"].update(
                            {
                                "state": "failed",
                                "last_error": f"package: {exc}",
                                "updated_at_utc": utc_now(),
                            }
                        )
                        errors.append(
                            {
                                "scope": f"tile:{job['tile']['id']}:package",
                                "error": str(exc),
                            }
                        )
                        if not continue_on_error:
                            for pending in futures:
                                pending.cancel()
                    _write_runtime_manifest(manifest_path, manifest, root)
                    if errors and not continue_on_error:
                        break

    refresh_manifest(manifest, root)
    _write_runtime_manifest(manifest_path, manifest, root)
    return {**counters, "errors": errors, "summary": manifest["summary"]}


def execute_near_orthophoto_manifest(
    manifest_path: Path,
    *,
    tile_ids: Sequence[str] = (),
    all_tiles: bool = False,
    download_workers: int = 2,
    minimum_free_gib: float = 20.0,
    timeout_s: float = 180.0,
    continue_on_error: bool = False,
    fetcher: Any | None = None,
) -> dict[str, Any]:
    """Materialize only native 20 cm imagery, without rebuilding tile geometry.

    An AOI-wide execution must be opted into with ``all_tiles=True``. Explicit
    ad-hoc selections retain the resident-set ceiling used by Blender so a
    missing ``--all`` flag cannot accidentally start the 475-tile download.
    Existing valid files are cache hits, which makes interrupted runs safe to
    restart with the same command.
    """

    if not 1 <= download_workers <= 4:
        raise ValueError("download_workers must be between 1 and 4")
    if all_tiles and tile_ids:
        raise ValueError("all_tiles and explicit tile_ids are mutually exclusive")
    if not all_tiles and not tile_ids:
        raise ValueError("explicit tile_ids are required unless all_tiles is enabled")
    identifiers = list(dict.fromkeys(tile_ids))
    if not all_tiles and len(identifiers) > NEAR_ORTHOPHOTO_MAX_TILES_PER_RUN:
        raise ValueError(
            "explicit near orthophoto selection exceeds the 16-tile safety limit; "
            "use the dedicated all_tiles opt-in for the complete production"
        )

    manifest_file = manifest_path.resolve()
    root = manifest_file.parent
    free_gib = shutil.disk_usage(root).free / (1024**3)
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB are free; {minimum_free_gib:.2f} GiB required"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(
            f"unsupported production manifest schema: {manifest.get('schema')!r}"
        )
    by_id = {tile["id"]: tile for tile in manifest["tiles"]}
    if all_tiles:
        selected = sorted(
            (
                tile
                for tile in manifest["tiles"]
                if tile.get("status", {}).get("state") == "ready"
            ),
            key=lambda tile: tile["id"],
        )
    else:
        missing = sorted(set(identifiers) - set(by_id))
        if missing:
            raise ValueError("unknown tile ids: " + ", ".join(missing))
        not_ready = sorted(
            identifier
            for identifier in identifiers
            if by_id[identifier].get("status", {}).get("state") != "ready"
        )
        if not_ready:
            raise ValueError("tiles are not ready: " + ", ".join(not_ready))
        selected = sorted(
            (by_id[item] for item in identifiers), key=lambda tile: tile["id"]
        )

    for tile in selected:
        ensure_near_orthophoto_contract(tile)
        tile["near_orthophoto_status"] = {
            "state": "running",
            "last_error": None,
            "updated_at_utc": utc_now(),
        }
    manifest["near_orthophoto_profile"] = {
        "resolution_m": NEAR_ORTHOPHOTO_RESOLUTION_M,
        "scope": "ready_detail_tiles",
        "selection": "all_ready_tiles" if all_tiles else "explicit_tile_ids",
        "fallback_resolution_m": float(
            manifest.get("tiling", {}).get("orthophoto_resolution_m", 0.5)
        ),
        "global_far_resolution_m": 2.0,
        "restart_policy": "validate_cache_then_download_missing",
    }
    _atomic_write_json(manifest_file, manifest, overwrite=True)

    errors: list[dict[str, str]] = []
    completed = 0
    cache_hits = 0
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = {
            executor.submit(
                ensure_near_orthophoto_tile,
                tile,
                root,
                timeout_s=timeout_s,
                fetcher=fetcher,
            ): tile
            for tile in selected
        }
        for future in as_completed(futures):
            tile = futures[future]
            try:
                result = future.result()
                completed += 1
                cache_hits += int(result["cache"] == "hit")
                for key in (
                    "near_orthophoto_source",
                    "near_orthophoto_image",
                    "near_orthophoto_geotiff",
                ):
                    _refresh_asset(tile["assets"][key], root)
                tile["near_orthophoto_status"] = {
                    "state": "ready",
                    "cache": result["cache"],
                    "last_error": None,
                    "updated_at_utc": utc_now(),
                }
            except Exception as exc:
                error = str(exc)
                tile["near_orthophoto_status"] = {
                    "state": "failed",
                    "last_error": error,
                    "updated_at_utc": utc_now(),
                }
                errors.append(
                    {"scope": f"tile:{tile['id']}:near_orthophoto", "error": error}
                )
                if not continue_on_error:
                    for pending in futures:
                        pending.cancel()
            if completed % 25 == 0 or errors:
                manifest["updated_at_utc"] = utc_now()
                _atomic_write_json(manifest_file, manifest, overwrite=True)
            if errors and not continue_on_error:
                break

    manifest["updated_at_utc"] = utc_now()
    profile = manifest["near_orthophoto_profile"]
    profile["selected_tile_count"] = len(selected)
    profile["ready_tile_count"] = sum(
        tile.get("near_orthophoto_status", {}).get("state") == "ready"
        for tile in selected
    )
    profile["failed_tile_count"] = sum(
        tile.get("near_orthophoto_status", {}).get("state") == "failed"
        for tile in selected
    )
    _atomic_write_json(manifest_file, manifest, overwrite=True)
    return {
        "selected_tile_count": len(selected),
        "completed_or_validated": completed,
        "cache_hits": cache_hits,
        "errors": errors,
        "resolution_m": NEAR_ORTHOPHOTO_RESOLUTION_M,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--origin", type=float, nargs=3, default=DEFAULT_ORIGIN)
    parser.add_argument("--output-tile-size-m", type=int, default=500)
    parser.add_argument("--halo-m", type=float, default=10.0)
    parser.add_argument(
        "--expected-source-tile-count",
        type=int,
        help=(
            "Optional locked IGN 1 km source-tile count. Omit for a new AOI; "
            "set it after reviewing the written plan to detect later AOI drift."
        ),
    )
    parser.add_argument(
        "--source-only-plan",
        action="store_true",
        help=(
            "Plan only the IGN 1 km elevation sources; use this for a "
            "global-only delivery with no 500 m detail packages."
        ),
    )
    parser.add_argument("--manifest-name", default="production-manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exclude-polygons", type=Path, action="append", default=[])
    parser.add_argument("--exclude-lines", type=Path, action="append", default=[])
    parser.add_argument("--tile", action="append", default=[])
    parser.add_argument("--all-near-orthophoto-tiles", action="store_true")
    parser.add_argument("--phase", choices=("all", "sources", "tiles"), default="all")
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--package-workers", type=int, default=1)
    parser.add_argument("--memory-budget-gib", type=float, default=4.0)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--rebuild-mid-packages",
        action="store_true",
        help=(
            "Rebuild selected 0.5 m terrain/vegetation packages and receipts, "
            "including ready tiles, without downloading or modifying imagery"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--required-elevation-resolution-m",
        type=float,
        help="Fail closed unless the planned native elevation source has this exact resolution.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write-plan", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--execute-near-orthophoto", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.output_root.resolve()
    manifest_path = root / args.manifest_name
    if args.execute_near_orthophoto:
        result = execute_near_orthophoto_manifest(
            manifest_path,
            tile_ids=args.tile,
            all_tiles=args.all_near_orthophoto_tiles,
            download_workers=args.download_workers,
            minimum_free_gib=args.minimum_free_gib,
            timeout_s=args.timeout_s,
            continue_on_error=args.continue_on_error,
            required_elevation_resolution_m=args.required_elevation_resolution_m,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result["errors"] else 1
    if args.execute:
        if args.aoi is None:
            raise ValueError("--aoi is required with --execute")
        result = execute_manifest(
            manifest_path,
            args.aoi,
            exclude_polygons=args.exclude_polygons,
            exclude_lines=args.exclude_lines,
            tile_ids=args.tile,
            phase=args.phase,
            download_workers=args.download_workers,
            package_workers=args.package_workers,
            memory_budget_gib=args.memory_budget_gib,
            minimum_free_gib=args.minimum_free_gib,
            max_tiles=args.max_tiles,
            timeout_s=args.timeout_s,
            retry_failed=args.retry_failed,
            rebuild_mid_packages=args.rebuild_mid_packages,
            continue_on_error=args.continue_on_error,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result["errors"] else 1
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"production manifest does not exist: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        refresh_manifest(manifest, root)
        _atomic_write_json(manifest_path, manifest, overwrite=True)
        print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
        return 0
    if args.aoi is None:
        raise ValueError("--aoi is required with --dry-run or --write-plan")
    manifest = build_plan(
        args.aoi,
        root,
        origin=args.origin,
        config=Global05mConfig(
            output_tile_size_m=args.output_tile_size_m,
            halo_m=args.halo_m,
        ),
        expected_source_tile_count=args.expected_source_tile_count,
        source_only=args.source_only_plan,
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    _atomic_write_json(manifest_path, manifest, overwrite=args.overwrite)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
