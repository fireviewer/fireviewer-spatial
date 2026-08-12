"""Headless production engine for portable measured FireViewer zones.

The pod owns no database.  A job is a deterministic directory below ``/work``:
one JSON plan, one WFS context snapshot, sequential 500 m tile packages, one
shared prototype bundle, one unified ``zone.usda`` and one downloadable ZIP.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from build_asset_library_53 import validate_asset_library
from build_reference_usd_asset_library import (
    SCHEMA as REFERENCE_ASSET_LIBRARY_SCHEMA,
)
from build_reference_usd_asset_library import (
    validate_reference_asset_library,
)
from fixed_asset_placement import (
    EMPTY_REQUEST as EMPTY_FIXED_ASSET_REQUEST,
)
from fixed_asset_placement import (
    FixedAssetPlacementError,
    asset_choices,
)
from fixed_asset_placement import (
    normalize_request as normalize_fixed_asset_request,
)
from fixed_asset_placement import (
    project_request as project_fixed_asset_request,
)
from fixed_asset_placement import (
    request_sha256 as fixed_asset_request_sha256,
)
from fixed_asset_placement import (
    schema_path as fixed_asset_schema_path,
)
from fixed_asset_placement import (
    template_path as fixed_asset_template_path,
)
from portable_scene_package import (
    seal_map_upload_package,
    validate_map_upload_package,
)
from prepare_simple_measured_tile_sources import PreparedSources, prepare_sources
from prepare_simple_measured_zone_context import ZoneContext, prepare_zone_context
from produce_simple_measured_tile import (
    SimpleMeasuredTilePackage,
    produce_simple_measured_tile,
    validate_simple_measured_tile_package,
)
from pyproj import Transformer
from render_simple_zone_gallery import (
    BLEND_NAME,
    CAPTURE_COUNT,
    verify_gallery,
)
from render_simple_zone_gallery import (
    RECEIPT_PATH as GALLERY_RECEIPT_PATH,
)

SCHEMA = "fireviewer.simple-production-engine.v1"
PLAN_SCHEMA = "fireviewer.simple-measured-zone-plan.v1"
ZONE_RECEIPT_SCHEMA = "fireviewer.simple-measured-zone-production.v1"
TILE_SIZE_M = 500
HALO_M = 10
ENTRY_STAGE = "zone.usda"
ZIP_NAME = "fireviewer-zone.zip"
STATUS_NAME = "job-status.json"
PLAN_NAME = "zone-plan.json"
ZONE_CONTEXT_NAME = "zone-context.json"
ZONE_RECEIPT_NAME = "zone.done.json"
DATASET_ENTRY_NAME = "dataset-entry.json"
DATASET_PUBLICATION_NAME = "dataset-publication.json"
FIXED_ASSET_REQUEST_NAME = "fixed-asset-placements.v1.json"


class SimpleProductionError(RuntimeError):
    """The pod configuration or requested zone cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    work_root: Path
    portable_root: Path
    asset_library: Path
    review_batch: Path
    elevation_revision: str
    orthophoto_revision: str
    context_revision: str
    max_side_m: int = 15_000
    max_tiles: int = 900
    blender: Path = Path("/opt/blender/blender")
    dataset_id: str | None = None

    @classmethod
    def from_environment(cls) -> ProductionConfig:
        return cls(
            work_root=Path(os.environ.get("FIREVIEWER_WORK_ROOT", "/work")),
            portable_root=Path(os.environ.get("FIREVIEWER_PORTABLE_ROOT", "/")),
            asset_library=Path(
                os.environ.get(
                    "FIREVIEWER_ASSET_LIBRARY",
                    "/opt/fireviewer/assets/reference-asset-library.v1.json",
                )
            ),
            review_batch=Path(
                os.environ.get(
                    "FIREVIEWER_REVIEW_BATCH",
                    "/opt/fireviewer/assets/review_batch_53",
                )
            ),
            elevation_revision=os.environ.get(
                "FIREVIEWER_ELEVATION_REVISION", "geopf-lidar-hd-current"
            ),
            orthophoto_revision=os.environ.get(
                "FIREVIEWER_ORTHOPHOTO_REVISION", "geopf-orthophotos-current"
            ),
            context_revision=os.environ.get(
                "FIREVIEWER_CONTEXT_REVISION", "bdtopo-v3-current"
            ),
            max_side_m=int(os.environ.get("FIREVIEWER_MAX_SIDE_M", "15000")),
            max_tiles=int(os.environ.get("FIREVIEWER_MAX_TILES", "900")),
            blender=Path(os.environ.get("FIREVIEWER_BLENDER", "/opt/blender/blender")),
            dataset_id=(os.environ.get("FIREVIEWER_HF_DATASET_ID", "").strip() or None),
        )


@dataclass(frozen=True, slots=True)
class TilePlan:
    tile_id: str
    origin_l93_m: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ZonePlan:
    zone_id: str
    latitude: float
    longitude: float
    side_m: int
    center_l93_m: tuple[float, float]
    requested_bounds_l93_m: tuple[float, float, float, float]
    production_bounds_l93_m: tuple[int, int, int, int]
    context_bounds_l93_m: tuple[int, int, int, int]
    tiles: tuple[TilePlan, ...]


PrepareContext = Callable[..., ZoneContext]
PrepareSources = Callable[..., PreparedSources]
ProduceTile = Callable[..., SimpleMeasuredTilePackage]
ProgressCallback = Callable[[float, str], None]
GalleryProgress = Callable[[int, str], None]
GalleryItems = list[tuple[str, str]]
RenderGallery = Callable[[Path, bool, GalleryProgress | None], GalleryItems]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = _canonical_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimpleProductionError(f"{label} JSON invalide: {error}") from error
    if not isinstance(value, dict):
        raise SimpleProductionError(f"{label} doit être un objet JSON")
    return value


def _contract_path() -> Path:
    return Path(__file__).with_name("simple_production_engine_contract.v1.json")


def _load_contract() -> dict[str, Any]:
    payload = _load_json(_contract_path(), "contrat du moteur de production")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "locked"
        or payload.get("production", {}).get("tile_size_m") != TILE_SIZE_M
        or payload.get("output", {}).get("entry_stage") != ENTRY_STAGE
        or payload.get("output", {}).get("portable_zip") is not True
        or payload.get("perimeter_layer", {}).get("fixed_layers") is not True
        or payload.get("perimeter_layer", {}).get("timeline_data")
        != "observed_instants_and_explicit_ranges"
        or payload.get("perimeter_layer", {}).get("prediction") != "forbidden"
        or payload.get("perimeter_layer", {}).get("viewer")
        != "one_derived_glb_per_observed_instant_or_range"
        or payload.get("perimeter_layer", {}).get("viewer_authoritative") is not False
        or payload.get("fixed_asset_placement", {}).get("request_schema")
        != "fireviewer.fixed-asset-placement-request.v1"
        or payload.get("fixed_asset_placement", {}).get("z")
        != "sampled_from_tile_MNT_never_user_authored"
        or payload.get("fixed_asset_placement", {}).get("automatic_catalog_selection")
        != "bypassed_for_fixed_asset_id"
        or payload.get("acceptance", {}).get("automatic_human_acceptance") is not False
    ):
        raise SimpleProductionError("Contrat moteur absent, modifié ou non verrouillé")
    return payload


def _absolute(path: Path | str, label: str, *, exists: bool) -> Path:
    lexical = PureWindowsPath(str(path))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise SimpleProductionError(f"{label} doit rester sur D: sous Windows")
    try:
        resolved = Path(path).resolve(strict=exists)
    except OSError as error:
        raise SimpleProductionError(f"{label} introuvable: {path}") from error
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise SimpleProductionError(f"{label} doit rester sur D: sous Windows")
    if exists and not resolved.exists():
        raise SimpleProductionError(f"{label} introuvable: {resolved}")
    return resolved


def _inside(root: Path, child: Path, label: str) -> Path:
    try:
        child.relative_to(root)
    except ValueError as error:
        raise SimpleProductionError(f"{label} sort du volume portable") from error
    return child


def validate_embedded_assets(config: ProductionConfig) -> dict[str, Any]:
    """Rehash every resolved real USD/texture before serving the UI."""

    portable = _absolute(config.portable_root, "racine portable", exists=True)
    library_path = _inside(
        portable,
        _absolute(config.asset_library, "catalogue des assets", exists=True),
        "catalogue des assets",
    )
    review_batch = _inside(
        portable,
        _absolute(config.review_batch, "lot des assets", exists=True),
        "lot des assets",
    )
    library = _load_json(library_path, "catalogue des assets")
    if library.get("schema") == REFERENCE_ASSET_LIBRARY_SCHEMA:
        summary = validate_reference_asset_library(library)
        reference_count = (
            library.get("reference_manifest", {})
            .get("route_counts", {})
            .get("hunyuan3d")
        )
        if summary.get("asset_count") != reference_count:
            raise SimpleProductionError(
                "Le catalogue USD ne couvre pas toutes les références 3D"
            )
    else:
        summary = validate_asset_library(library)
        if (
            summary.get("asset_count") != 53
            or summary.get("category_counts", {}).get("building") != 24
            or summary.get("category_counts", {}).get("tree") != 18
        ):
            raise SimpleProductionError(
                "Le catalogue embarqué historique n'est pas le lot pilote 53/53"
            )
    checked = 0
    for asset in library["assets"]:
        for role in ("usd", "texture"):
            record = asset.get(role)
            if not isinstance(record, Mapping) or record.get("root") != "review_batch":
                raise SimpleProductionError(
                    f"Asset embarqué invalide: {asset.get('asset_id')}.{role}"
                )
            relative = record.get("path")
            if (
                not isinstance(relative, str)
                or not relative
                or "\\" in relative
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
            ):
                raise SimpleProductionError("Chemin d'asset embarqué invalide")
            target = review_batch.joinpath(*PurePosixPath(relative).parts).resolve()
            _inside(review_batch, target, "asset embarqué")
            if (
                not target.is_file()
                or target.stat().st_size != record.get("byte_count")
                or _sha256_file(target) != record.get("sha256")
            ):
                raise SimpleProductionError(
                    f"Asset embarqué absent ou altéré: {asset['asset_id']}.{role}"
                )
            checked += 1
    return {
        "asset_count": summary["asset_count"],
        "building_assets": summary["category_counts"].get("building", 0),
        "tree_assets": summary["category_counts"].get("tree", 0),
        "real_usd_assets": summary.get("availability_counts", {}).get(
            "real_usd", summary["asset_count"]
        ),
        "placeholder_usd_assets": summary.get("availability_counts", {}).get(
            "placeholder_usd", 0
        ),
        "fallback_usd_assets": summary.get("fallback_asset_count", 0),
        "unique_real_usd_assets": summary.get(
            "unique_real_usd_count", summary["asset_count"]
        ),
        "checked_artifacts": checked,
        "catalog_revision": library["catalog_revision"],
        "catalog_sha256": _sha256_file(library_path),
    }


def validate_embedded_runtime(config: ProductionConfig) -> dict[str, Any]:
    """Prove that Blender, OpenUSD and all production entrypoints are embedded."""

    blender = _absolute(config.blender, "Blender embarqué", exists=True)
    if not blender.is_file():
        raise SimpleProductionError("Blender embarqué est absent")
    required = (
        Path(__file__),
        Path(__file__).with_name("prepare_simple_measured_zone_context.py"),
        Path(__file__).with_name("prepare_simple_measured_tile_sources.py"),
        Path(__file__).with_name("produce_simple_measured_tile.py"),
        Path(__file__).with_name("run_simple_measured_tile_qa.py"),
        Path(__file__).with_name("render_simple_zone_gallery.py"),
        Path(__file__).with_name("build_reference_usd_asset_library.py"),
        Path(__file__).with_name("geographic_perimeter_layer.py"),
        Path(__file__).with_name("geographic_perimeter_layer_contract.v1.json"),
        Path(__file__).with_name("geographic_perimeter_viewer.py"),
        Path(__file__).with_name("portable_scene_package.py"),
        Path(__file__).with_name("fixed_asset_placement.py"),
        fixed_asset_schema_path(),
        fixed_asset_template_path(),
        Path(__file__).parent.parent / "omniverse" / "fixed_terrain_usd.py",
        Path(__file__).parent.parent / "omniverse" / "build_measured_scene_usd.py",
    )
    missing = [str(path) for path in required if not path.resolve().is_file()]
    if missing:
        raise SimpleProductionError(
            "Scripts de production embarqués absents: " + ", ".join(missing)
        )
    expression = (
        "import bpy; from pxr import Usd,UsdGeom,UsdShade; "
        "print('FIREVIEWER_BLENDER_USD=PASS')"
    )
    try:
        result = subprocess.run(
            (
                str(blender),
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-expr",
                expression,
            ),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SimpleProductionError(
            f"Blender/OpenUSD embarqué ne démarre pas: {error}"
        ) from error
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or "FIREVIEWER_BLENDER_USD=PASS" not in output:
        raise SimpleProductionError(
            "Le smoke embarqué Blender/OpenUSD a échoué: " + output[-1000:]
        )
    version_line = next(
        (line.strip() for line in output.splitlines() if line.startswith("Blender ")),
        "",
    )
    if not version_line.startswith("Blender 4.5"):
        raise SimpleProductionError(
            f"Blender embarqué doit être en version 4.5; reçu {version_line!r}"
        )
    return {
        "blender_path": str(blender),
        "blender_version": version_line.removeprefix("Blender "),
        "blender_sha256": _sha256_file(blender),
        "openusd": "bundled_with_blender_passed",
        "production_script_count": len(required),
    }


def validate_config(config: ProductionConfig) -> dict[str, Any]:
    """Validate the headless engine contract and shared production runtime."""

    _load_contract()
    return validate_production_config(config)


def validate_production_config(config: ProductionConfig) -> dict[str, Any]:
    """Validate only the headless production runtime shared by every frontend."""

    if config.max_side_m < TILE_SIZE_M or config.max_tiles < 1:
        raise SimpleProductionError("Limites du pod invalides")
    portable = _absolute(config.portable_root, "racine portable", exists=True)
    work = _absolute(config.work_root, "volume de travail", exists=False)
    _inside(portable, work, "volume de travail")
    work.mkdir(parents=True, exist_ok=True)
    for label, revision in (
        ("révision élévation", config.elevation_revision),
        ("révision orthophoto", config.orthophoto_revision),
        ("révision contexte", config.context_revision),
    ):
        if not revision or revision != revision.strip():
            raise SimpleProductionError(f"{label} invalide")
    return {
        **validate_embedded_assets(config),
        **validate_embedded_runtime(config),
    }


def plan_zone(
    latitude: float,
    longitude: float,
    side_km: float,
    *,
    max_side_m: int = 15_000,
    max_tiles: int = 900,
    transformer: Transformer | None = None,
) -> ZonePlan:
    """Transform the GPS centre and cover the requested square on the 500 m grid."""

    values = (float(latitude), float(longitude), float(side_km))
    if any(not math.isfinite(value) for value in values):
        raise SimpleProductionError("Latitude, longitude et taille doivent être finies")
    latitude, longitude, side_km = values
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise SimpleProductionError("Coordonnées GPS invalides")
    side_m_float = side_km * 1_000.0
    if side_m_float < TILE_SIZE_M or side_m_float > max_side_m:
        raise SimpleProductionError(
            f"La taille doit être comprise entre 0,5 et {max_side_m / 1000:g} km"
        )
    if not side_m_float.is_integer():
        raise SimpleProductionError(
            "La taille doit correspondre à un nombre entier de mètres"
        )
    side_m = int(side_m_float)
    convert = transformer or Transformer.from_crs(4326, 2154, always_xy=True)
    center_x, center_y = convert.transform(longitude, latitude)
    if (
        not math.isfinite(center_x)
        or not math.isfinite(center_y)
        or not -100_000 <= center_x <= 1_500_000
        or not 5_900_000 <= center_y <= 7_300_000
    ):
        raise SimpleProductionError(
            "Le centre doit être couvert par la projection Lambert-93"
        )
    half = side_m / 2.0
    requested = (
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
    )

    def grid_floor(value: float) -> int:
        nearest = round(value / TILE_SIZE_M) * TILE_SIZE_M
        if abs(value - nearest) <= 0.001:
            return int(nearest)
        return math.floor(value / TILE_SIZE_M) * TILE_SIZE_M

    def grid_ceil(value: float) -> int:
        nearest = round(value / TILE_SIZE_M) * TILE_SIZE_M
        if abs(value - nearest) <= 0.001:
            return int(nearest)
        return math.ceil(value / TILE_SIZE_M) * TILE_SIZE_M

    west = grid_floor(requested[0])
    south = grid_floor(requested[1])
    east = grid_ceil(requested[2])
    north = grid_ceil(requested[3])
    origins = tuple(
        (x, y)
        for y in range(north - TILE_SIZE_M, south - 1, -TILE_SIZE_M)
        for x in range(west, east, TILE_SIZE_M)
    )
    if not origins or len(origins) > max_tiles:
        raise SimpleProductionError(
            f"La zone demande {len(origins)} tuiles; limite du pod: {max_tiles}"
        )
    identity = {
        "latitude": round(latitude, 10),
        "longitude": round(longitude, 10),
        "side_m": side_m,
        "production_bounds_l93_m": [west, south, east, north],
    }
    zone_id = (
        "GPS-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:16].upper()
    )
    tiles = tuple(TilePlan(f"x{x}_y{y}", (x, y)) for x, y in origins)
    return ZonePlan(
        zone_id=zone_id,
        latitude=latitude,
        longitude=longitude,
        side_m=side_m,
        center_l93_m=(center_x, center_y),
        requested_bounds_l93_m=requested,
        production_bounds_l93_m=(west, south, east, north),
        context_bounds_l93_m=(
            west - HALO_M,
            south - HALO_M,
            east + HALO_M,
            north + HALO_M,
        ),
        tiles=tiles,
    )


def _plan_payload(
    plan: ZonePlan,
    config: ProductionConfig,
    *,
    fixed_asset_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "planned",
        "zone_id": plan.zone_id,
        "input": {
            "latitude": plan.latitude,
            "longitude": plan.longitude,
            "side_m": plan.side_m,
            "coordinate_role": "square_center",
        },
        "crs": "EPSG:2154",
        "center_l93_m": list(plan.center_l93_m),
        "requested_bounds_l93_m": list(plan.requested_bounds_l93_m),
        "production_bounds_l93_m": list(plan.production_bounds_l93_m),
        "context_bounds_l93_m": list(plan.context_bounds_l93_m),
        "tile_size_m": TILE_SIZE_M,
        "tile_count": len(plan.tiles),
        "tiles": [
            {"tile_id": tile.tile_id, "origin_l93_m": list(tile.origin_l93_m)}
            for tile in plan.tiles
        ],
        "source_revisions": {
            "elevation": config.elevation_revision,
            "orthophoto": config.orthophoto_revision,
            "context": config.context_revision,
        },
    }
    if fixed_asset_request is not None:
        payload["fixed_asset_placements"] = {
            "path": FIXED_ASSET_REQUEST_NAME,
            "schema": fixed_asset_request["schema"],
            "count": len(fixed_asset_request["placements"]),
            "sha256": hashlib.sha256(
                _canonical_bytes(fixed_asset_request) + b"\n"
            ).hexdigest(),
            "content_sha256": fixed_asset_request_sha256(fixed_asset_request),
        }
    payload["plan_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _status(
    root: Path,
    *,
    state: str,
    message: str,
    completed: int,
    total: int,
    error: str | None = None,
    phase: str | None = None,
    details: Mapping[str, Any] | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": "fireviewer.simple-production-job-status.v1",
        "state": state,
        "message": message,
        "completed_tiles": completed,
        "total_tiles": total,
    }
    if phase is not None:
        payload["phase"] = phase
    if details is not None:
        payload["details"] = dict(details)
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = round(elapsed_seconds, 3)
    if error is not None:
        payload["error"] = error
    _write_json(root / STATUS_NAME, payload)


def _copy_provenance(sources: PreparedSources, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    names = (
        "elevation-source-05m.json",
        "orthophoto-source.json",
        "building-source.json",
        "bdtopo-buildings.geojson",
        "placement-context.json",
        "simple-measured-tile-sources.v1.json",
    )
    for name in names:
        source = sources.root / name
        if not source.is_file():
            raise SimpleProductionError(f"Provenance de tuile absente: {name}")
        shutil.copyfile(source, target / name)


def _remove_sources(job_root: Path, source_root: Path) -> None:
    expected_root = (job_root / "sources").resolve()
    resolved = source_root.resolve(strict=True)
    try:
        resolved.relative_to(expected_root)
    except ValueError as error:
        raise SimpleProductionError("Nettoyage source hors du job refusé") from error
    if resolved == expected_root:
        raise SimpleProductionError("Nettoyage global des sources refusé")
    shutil.rmtree(resolved)


def _scene_counts(package_root: Path) -> tuple[int, int, int, int]:
    receipt = _load_json(package_root / "scene" / "scene.done.json", "reçu scène")
    reconciliation = receipt.get("reconciliation")
    if not isinstance(reconciliation, Mapping):
        raise SimpleProductionError("Reçu scène sans réconciliation")
    values: list[int] = []
    for family in ("buildings", "trees", "context_assets"):
        record = reconciliation.get(family)
        count = (
            record.get("instance_count")
            if isinstance(record, Mapping)
            else 0
            if family == "context_assets"
            else None
        )
        if not isinstance(count, int) or count < 0:
            raise SimpleProductionError(f"Comptage {family} invalide")
        values.append(count)
    placeholder_count = receipt.get("placeholder_instance_count", 0)
    if not isinstance(placeholder_count, int) or placeholder_count < 0:
        raise SimpleProductionError("Comptage des placeholders invalide")
    return values[0], values[1], values[2], placeholder_count


def _expected_request_from_receipt(
    package_root: Path,
    *,
    plan: ZonePlan,
    tile: TilePlan,
    asset_library: Path,
    asset_roots: Mapping[str, Path],
    portable_root: Path,
    asset_bundle_root: Path,
) -> dict[str, Any]:
    """Recover the signed request while binding it to the current zone request."""

    receipt = _load_json(
        package_root / "simple-measured-tile-receipt.v1.json", "reçu de tuile"
    )
    request = receipt.get("request")
    if not isinstance(request, Mapping):
        raise SimpleProductionError("Reçu de tuile sans requête scellée")
    expected_fixed = {
        "schema": "fireviewer.simple-measured-tile-request.v1",
        "algorithm": "fireviewer.simple-measured-tile-algorithm.v1",
        "zone_id": plan.zone_id,
        "tile_id": tile.tile_id,
        "crs": "EPSG:2154",
        "tile_origin_l93_m": list(tile.origin_l93_m),
        "core_bounds_l93_m": [
            tile.origin_l93_m[0],
            tile.origin_l93_m[1],
            tile.origin_l93_m[0] + TILE_SIZE_M,
            tile.origin_l93_m[1] + TILE_SIZE_M,
        ],
        "asset_library_sha256": _sha256_file(asset_library),
        "asset_root_names": sorted(asset_roots),
        "usage": "technical_pilot_non_final",
        "prototype_bundle": {
            "scope": "explicit_shared",
            "portable_path": asset_bundle_root.resolve()
            .relative_to(portable_root.resolve())
            .as_posix(),
        },
    }
    changed = [
        key for key, value in expected_fixed.items() if request.get(key) != value
    ]
    if changed:
        raise SimpleProductionError(
            "La requête scellée de la tuile diffère du job courant: "
            + ", ".join(changed)
        )
    if not isinstance(request.get("sources"), Mapping) or not isinstance(
        request.get("pipeline_files"), Mapping
    ):
        raise SimpleProductionError("Requête scellée de tuile incomplète")
    return dict(request)


def _write_zone_stage(job_root: Path, plan: ZonePlan) -> Path:
    west, south, _east, _north = plan.production_bounds_l93_m
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "FireViewerZone"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "FireViewerZone"',
        "{",
        f'    custom string fireviewer:zone_id = "{plan.zone_id}"',
        f"    custom int fireviewer:tile_count = {len(plan.tiles)}",
    ]
    for tile in plan.tiles:
        x, y = tile.origin_l93_m
        reference = f"packages/{tile.tile_id}/scene/scene.usda"
        lines.extend(
            [
                "",
                f'    def Xform "{tile.tile_id}" (',
                f"        prepend references = @{reference}@",
                "    )",
                "    {",
                f"        double3 xformOp:translate = ({x - west}, {y - south}, 0)",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
            ]
        )
    lines.extend(["}", ""])
    destination = job_root / ENTRY_STAGE
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return destination


def _write_zone_receipt(
    job_root: Path,
    plan: ZonePlan,
    asset_summary: Mapping[str, Any],
    portable_root: Path,
    shared_bundle: Path,
) -> dict[str, Any]:
    tile_records: list[dict[str, Any]] = []
    buildings = 0
    trees = 0
    context_assets = 0
    placeholders = 0
    for tile in plan.tiles:
        package_root = job_root / "packages" / tile.tile_id
        receipt_path = package_root / "simple-measured-tile-receipt.v1.json"
        receipt = _load_json(receipt_path, "reçu de tuile")
        validate_simple_measured_tile_package(
            package_root,
            expected_request=_expected_request_from_receipt(
                package_root,
                plan=plan,
                tile=tile,
                asset_library=Path(asset_summary["asset_library"]),
                asset_roots={"review_batch": Path(asset_summary["review_batch"])},
                portable_root=portable_root,
                asset_bundle_root=shared_bundle,
            ),
            asset_library=Path(asset_summary["asset_library"]),
            asset_roots={"review_batch": Path(asset_summary["review_batch"])},
        )
        (
            tile_buildings,
            tile_trees,
            tile_context_assets,
            tile_placeholders,
        ) = _scene_counts(package_root)
        buildings += tile_buildings
        trees += tile_trees
        context_assets += tile_context_assets
        placeholders += tile_placeholders
        tile_records.append(
            {
                "tile_id": tile.tile_id,
                "origin_l93_m": list(tile.origin_l93_m),
                "build_id": receipt["build_id"],
                "receipt_sha256": _sha256_file(receipt_path),
                "building_count": tile_buildings,
                "tree_count": tile_trees,
                "context_asset_count": tile_context_assets,
                "placeholder_instance_count": tile_placeholders,
            }
        )
    context_path = job_root / ZONE_CONTEXT_NAME
    plan_path = job_root / PLAN_NAME
    stage_path = job_root / ENTRY_STAGE
    payload: dict[str, Any] = {
        "schema": ZONE_RECEIPT_SCHEMA,
        "status": "technical_scene_produced",
        "accepted_human": False,
        "automatic_acceptance": False,
        "zone_id": plan.zone_id,
        "entry_stage": {
            "path": ENTRY_STAGE,
            "byte_count": stage_path.stat().st_size,
            "sha256": _sha256_file(stage_path),
        },
        "plan": {"path": PLAN_NAME, "sha256": _sha256_file(plan_path)},
        "context": {
            "path": ZONE_CONTEXT_NAME,
            "sha256": _sha256_file(context_path),
        },
        "asset_library": {
            "asset_count": asset_summary["asset_count"],
            "catalog_revision": asset_summary["catalog_revision"],
            "catalog_sha256": asset_summary["catalog_sha256"],
            "embedded_artifacts_checked": asset_summary["checked_artifacts"],
            "real_usd_assets": asset_summary.get(
                "real_usd_assets", asset_summary["asset_count"]
            ),
            "placeholder_usd_assets": asset_summary.get("placeholder_usd_assets", 0),
        },
        "tile_count": len(tile_records),
        "building_count": buildings,
        "tree_count": trees,
        "context_asset_count": context_assets,
        "placeholder_instance_count": placeholders,
        "tiles": tile_records,
    }
    fixed_request_path = job_root / FIXED_ASSET_REQUEST_NAME
    if fixed_request_path.is_file():
        fixed_request = _load_json(fixed_request_path, "placements fixes")
        payload["fixed_asset_placements"] = {
            "path": FIXED_ASSET_REQUEST_NAME,
            "count": len(fixed_request.get("placements", [])),
            "sha256": _sha256_file(fixed_request_path),
            "content_sha256": fixed_asset_request_sha256(fixed_request),
        }
    payload["build_id"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    _write_json(job_root / ZONE_RECEIPT_NAME, payload)
    return payload


def _write_zip(job_root: Path, zone_id: str) -> Path:
    destination = job_root / ZIP_NAME
    temporary = destination.with_name(f".{destination.name}.part")
    excluded_roots = {"sources", "download"}
    excluded_files = {STATUS_NAME, ZIP_NAME, temporary.name}
    files = [
        path
        for path in job_root.rglob("*")
        if path.is_file()
        and path.name not in excluded_files
        and path.relative_to(job_root).parts[0] not in excluded_roots
    ]
    prefix = PurePosixPath(f"fireviewer-{zone_id}")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for path in sorted(
            files, key=lambda value: value.relative_to(job_root).as_posix()
        ):
            relative = path.relative_to(job_root).as_posix()
            info = zipfile.ZipInfo((prefix / relative).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=1)
    os.replace(temporary, destination)
    return destination


def _gallery_items(job_root: Path) -> GalleryItems:
    receipt = verify_gallery(job_root)
    items: GalleryItems = []
    for record in receipt["captures"]:
        artifact = record["artifact"]
        items.append(
            (
                str(job_root / artifact["path"]),
                f"{record['capture_id']} — {record['category']}",
            )
        )
    if len(items) != CAPTURE_COUNT:
        raise SimpleProductionError("La galerie validée ne contient pas 20 images")
    return items


def _render_zone_gallery(
    config: ProductionConfig,
    job_root: Path,
    render: bool,
    progress_callback: GalleryProgress | None,
) -> GalleryItems:
    if not render:
        return _gallery_items(job_root)
    runtime_root = job_root / "qa" / "blender-runtime"
    runtime_paths = {
        "TEMP": runtime_root / "temp",
        "TMP": runtime_root / "temp",
        "PYTHONPYCACHEPREFIX": runtime_root / "pycache",
        "BLENDER_USER_CONFIG": runtime_root / "config",
        "BLENDER_USER_SCRIPTS": runtime_root / "scripts",
        "BLENDER_USER_DATAFILES": runtime_root / "datafiles",
        "BLENDER_USER_EXTENSIONS": runtime_root / "extensions",
        "XDG_CACHE_HOME": runtime_root / "xdg-cache",
        "XDG_CONFIG_HOME": runtime_root / "xdg-config",
    }
    for path in set(runtime_paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({name: str(path) for name, path in runtime_paths.items()})
    script = Path(__file__).with_name("render_simple_zone_gallery.py").resolve()
    command = (
        str(config.blender),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python",
        str(script),
        "--",
        "render",
        "--job-root",
        str(job_root),
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    if process.stdout is None:  # pragma: no cover - Popen invariant
        process.kill()
        raise SimpleProductionError("Blender ne fournit aucun journal de rendu")
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line.rstrip())
        messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    output_tail: list[str] = []
    deadline = time.monotonic() + 45 * 60
    stream_done = False
    while not stream_done:
        if time.monotonic() > deadline:
            process.kill()
            raise SimpleProductionError("Le rendu des 20 captures dépasse 45 minutes")
        try:
            line = messages.get(timeout=1.0)
        except queue.Empty:
            if process.poll() is not None and not reader.is_alive():
                break
            continue
        if line is None:
            stream_done = True
            continue
        output_tail.append(line)
        output_tail = output_tail[-120:]
        if line.startswith("FIREVIEWER_CAPTURE "):
            fields = line.split(maxsplit=3)
            if len(fields) >= 3 and "/" in fields[1]:
                index = int(fields[1].split("/", 1)[0])
                if progress_callback is not None:
                    progress_callback(index, fields[2])
    return_code = process.wait(timeout=30)
    reader.join(timeout=5)
    if return_code != 0:
        raise SimpleProductionError(
            "Le rendu Blender des 20 captures a échoué:\n"
            + "\n".join(output_tail[-30:])
        )
    return _gallery_items(job_root)


def _publish_dataset_entry(
    config: ProductionConfig,
    *,
    job_root: Path,
    plan: ZonePlan,
    receipt: Mapping[str, Any],
    archive: Path,
) -> dict[str, Any] | None:
    """Publish one validated private scene pack; never persist the HF token."""

    if config.dataset_id is None:
        return None
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SimpleProductionError(
            "HF_TOKEN absent: publication dans la dataset privée impossible"
        )
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise SimpleProductionError("Le ZIP est altéré avant publication dataset")
    build_id = receipt.get("build_id")
    if not isinstance(build_id, str) or len(build_id) != 64:
        raise SimpleProductionError("Build ID de zone invalide avant publication")
    archive_record = {
        "file": ZIP_NAME,
        "byte_count": archive.stat().st_size,
        "sha256": _sha256_file(archive),
    }
    entry: dict[str, Any] = {
        "schema": "fireviewer.simple-measured-scene-dataset-entry.v1",
        "status": "technical_scene_produced",
        "accepted_human": False,
        "dataset_id": config.dataset_id,
        "zone_id": plan.zone_id,
        "build_id": build_id,
        "request": {
            "latitude": plan.latitude,
            "longitude": plan.longitude,
            "side_m": plan.side_m,
            "production_bounds_l93_m": list(plan.production_bounds_l93_m),
        },
        "counts": {
            "tiles": receipt["tile_count"],
            "buildings": receipt["building_count"],
            "trees": receipt["tree_count"],
            "context_assets": receipt.get("context_asset_count", 0),
            "placeholder_instances": receipt.get("placeholder_instance_count", 0),
        },
        "archive": archive_record,
        "zone_receipt": {
            "file": ZONE_RECEIPT_NAME,
            "sha256": _sha256_file(job_root / ZONE_RECEIPT_NAME),
        },
        "container_image": os.environ.get("FIREVIEWER_IMAGE_REFERENCE", "unrecorded"),
    }
    entry["entry_sha256"] = hashlib.sha256(_canonical_bytes(entry)).hexdigest()
    entry_path = job_root / DATASET_ENTRY_NAME
    _write_json(entry_path, entry)
    publication_path = job_root / DATASET_PUBLICATION_NAME
    if publication_path.is_file():
        publication = _load_json(publication_path, "reçu de publication dataset")
        if (
            publication.get("dataset_id") != config.dataset_id
            or publication.get("zone_id") != plan.zone_id
            or publication.get("build_id") != build_id
            or publication.get("archive_sha256") != archive_record["sha256"]
            or publication.get("entry_sha256") != entry["entry_sha256"]
        ):
            raise SimpleProductionError(
                "Le reçu de publication dataset existant est incohérent"
            )
        return publication

    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi(token=token)
        info = api.repo_info(repo_id=config.dataset_id, repo_type="dataset")
        if info.private is not True:
            raise SimpleProductionError(
                "La dataset cible doit être privée avant toute publication"
            )
        remote_root = f"zones/{plan.zone_id}/{build_id}"
        commit = api.create_commit(
            repo_id=config.dataset_id,
            repo_type="dataset",
            commit_message=f"Add measured FireViewer scene {plan.zone_id}",
            operations=[
                CommitOperationAdd(
                    path_in_repo=f"{remote_root}/{ZIP_NAME}",
                    path_or_fileobj=archive,
                ),
                CommitOperationAdd(
                    path_in_repo=f"{remote_root}/{ZONE_RECEIPT_NAME}",
                    path_or_fileobj=job_root / ZONE_RECEIPT_NAME,
                ),
                CommitOperationAdd(
                    path_in_repo=f"{remote_root}/{DATASET_ENTRY_NAME}",
                    path_or_fileobj=entry_path,
                ),
            ],
        )
    except SimpleProductionError:
        raise
    except Exception as error:
        raise SimpleProductionError(
            f"Publication dataset privée échouée: {error}"
        ) from error
    publication = {
        "schema": "fireviewer.simple-measured-scene-dataset-publication.v1",
        "status": "published_private",
        "dataset_id": config.dataset_id,
        "zone_id": plan.zone_id,
        "build_id": build_id,
        "archive_sha256": archive_record["sha256"],
        "entry_sha256": entry["entry_sha256"],
        "commit_oid": commit.oid,
        "path_in_repo": remote_root,
    }
    _write_json(publication_path, publication)
    return publication


class ProductionEngine:
    """Sequential, one-job-at-a-time production engine used by the headless API."""

    def __init__(
        self,
        config: ProductionConfig,
        *,
        prepare_context_fn: PrepareContext = prepare_zone_context,
        prepare_sources_fn: PrepareSources = prepare_sources,
        produce_tile_fn: ProduceTile = produce_simple_measured_tile,
        render_gallery_fn: RenderGallery | None = None,
        validate_assets: bool = True,
    ) -> None:
        self.config = config
        self.prepare_context_fn = prepare_context_fn
        self.prepare_sources_fn = prepare_sources_fn
        self.produce_tile_fn = produce_tile_fn
        self.render_gallery_fn = render_gallery_fn or (
            lambda job_root, render, callback: _render_zone_gallery(
                config, job_root, render, callback
            )
        )
        self._lock = threading.Lock()
        if validate_assets:
            self.asset_summary = validate_production_config(config)
            self.asset_library_payload = _load_json(
                _absolute(config.asset_library, "catalogue des assets", exists=True),
                "catalogue des assets",
            )
            self.fixed_asset_choices = tuple(asset_choices(self.asset_library_payload))
        else:
            work = _absolute(config.work_root, "volume de travail", exists=False)
            work.mkdir(parents=True, exist_ok=True)
            self.asset_summary = {
                "asset_count": 53,
                "building_assets": 24,
                "tree_assets": 18,
                "checked_artifacts": 106,
                "catalog_revision": "0" * 64,
                "catalog_sha256": "0" * 64,
                "blender_path": str(config.blender),
                "blender_version": "test",
                "blender_sha256": "0" * 64,
                "openusd": "test",
                "production_script_count": 12,
            }
            try:
                payload = _load_json(
                    config.asset_library, "catalogue des assets de test"
                )
                choices = tuple(asset_choices(payload))
            except (OSError, SimpleProductionError, FixedAssetPlacementError):
                payload = {}
                choices = ()
            self.asset_library_payload = payload
            self.fixed_asset_choices = choices

    def run(
        self,
        latitude: float,
        longitude: float,
        side_km: float,
        *,
        progress_callback: ProgressCallback | None = None,
        fixed_asset_placements: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[str, str | None, GalleryItems]]:
        if not self._lock.acquire(blocking=False):
            raise SimpleProductionError("Une production est déjà en cours sur ce pod")
        try:
            yield from self._run(
                latitude,
                longitude,
                side_km,
                progress_callback=progress_callback,
                fixed_asset_placements=(
                    dict(EMPTY_FIXED_ASSET_REQUEST)
                    if fixed_asset_placements is None
                    else normalize_fixed_asset_request(
                        fixed_asset_placements, self.asset_library_payload
                    )
                ),
            )
        finally:
            self._lock.release()

    def _run(
        self,
        latitude: float,
        longitude: float,
        side_km: float,
        *,
        progress_callback: ProgressCallback | None,
        fixed_asset_placements: Mapping[str, Any],
    ) -> Iterator[tuple[str, str | None, GalleryItems]]:
        started = time.perf_counter()
        base_plan = plan_zone(
            latitude,
            longitude,
            side_km,
            max_side_m=self.config.max_side_m,
            max_tiles=self.config.max_tiles,
        )
        fixed_count = len(fixed_asset_placements["placements"])
        projected_fixed_assets = (
            project_fixed_asset_request(
                fixed_asset_placements,
                self.asset_library_payload,
                requested_bounds_l93_m=base_plan.requested_bounds_l93_m,
            )
            if fixed_count
            else ()
        )
        plan = (
            replace(
                base_plan,
                zone_id=(
                    f"{base_plan.zone_id}-fixed-"
                    f"{fixed_asset_request_sha256(fixed_asset_placements)[:12]}"
                ),
            )
            if fixed_count
            else base_plan
        )
        work = _absolute(self.config.work_root, "volume de travail", exists=True)
        job_root = work / "jobs" / plan.zone_id
        job_root.mkdir(parents=True, exist_ok=True)
        if fixed_count:
            fixed_request_path = job_root / FIXED_ASSET_REQUEST_NAME
            if fixed_request_path.exists():
                if _load_json(fixed_request_path, "placements fixes existants") != dict(
                    fixed_asset_placements
                ):
                    raise SimpleProductionError(
                        "Les placements fixes existants diffèrent de la requête"
                    )
            else:
                _write_json(fixed_request_path, fixed_asset_placements)

        def report(
            fraction: float,
            phase: str,
            message: str,
            *,
            completed_tiles: int = 0,
            details: Mapping[str, Any] | None = None,
        ) -> None:
            _status(
                job_root,
                state="producing",
                message=message,
                completed=completed_tiles,
                total=len(plan.tiles),
                phase=phase,
                details=details,
                elapsed_seconds=time.perf_counter() - started,
            )
            if progress_callback is not None:
                progress_callback(max(0.0, min(1.0, fraction)), message)

        report(
            0.01,
            "embedded_runtime_verified",
            "Moteur embarqué validé — Blender "
            f"{self.asset_summary['blender_version']}, OpenUSD, "
            f"{self.asset_summary['asset_count']} assets USD "
            f"({self.asset_summary.get('fallback_usd_assets', 0)} substitutions "
            "USD déterministes)",
        )
        plan_path = job_root / PLAN_NAME
        plan_payload = _plan_payload(
            plan,
            self.config,
            fixed_asset_request=fixed_asset_placements if fixed_count else None,
        )
        if plan_path.exists():
            if _load_json(plan_path, "plan existant") != plan_payload:
                raise SimpleProductionError(
                    "Un job différent utilise déjà cet identifiant"
                )
        else:
            _write_json(plan_path, plan_payload)

        existing_archive = job_root / ZIP_NAME
        existing_receipt = job_root / ZONE_RECEIPT_NAME
        existing_stage = job_root / ENTRY_STAGE
        existing_gallery = job_root / GALLERY_RECEIPT_PATH
        existing_blend = job_root / BLEND_NAME
        if (
            existing_archive.is_file()
            and existing_receipt.is_file()
            and existing_stage.is_file()
            and existing_gallery.is_file()
            and existing_blend.is_file()
        ):
            receipt = _load_json(existing_receipt, "reçu de zone existant")
            if (
                receipt.get("schema") != ZONE_RECEIPT_SCHEMA
                or receipt.get("zone_id") != plan.zone_id
                or receipt.get("tile_count") != len(plan.tiles)
                or receipt.get("entry_stage", {}).get("sha256")
                != _sha256_file(existing_stage)
            ):
                raise SimpleProductionError("La production existante est incohérente")
            validate_map_upload_package(job_root)
            with zipfile.ZipFile(existing_archive) as archive:
                expected = f"fireviewer-{plan.zone_id}/{ENTRY_STAGE}"
                if expected not in archive.namelist() or archive.testzip() is not None:
                    raise SimpleProductionError("Le ZIP existant est altéré")
            gallery = self.render_gallery_fn(job_root, False, None)
            if self.config.dataset_id is not None:
                report(
                    0.98,
                    "dataset_publication",
                    "Publication du pack validé dans la dataset FireViewer privée",
                    completed_tiles=len(plan.tiles),
                )
                _publish_dataset_entry(
                    self.config,
                    job_root=job_root,
                    plan=plan,
                    receipt=receipt,
                    archive=existing_archive,
                )
            completed_message = (
                f"Déjà produit — {receipt['tile_count']} terrains, "
                f"{receipt['building_count']} bâtiments, {receipt['tree_count']} arbres, "
                f"{receipt.get('context_asset_count', 0)} équipements contextuels, "
                "20 captures revalidées."
            )
            yield (
                completed_message,
                str(existing_archive),
                gallery,
            )
            return

        total = len(plan.tiles)
        report(
            0.03,
            "zone_context_download",
            "Contexte IGN — téléchargement BD TOPO/occupation du sol",
        )
        yield f"Préparation de {total} tuiles de 500 m…", None, []
        zone_context = self.prepare_context_fn(
            output_path=job_root / ZONE_CONTEXT_NAME,
            zone_id=plan.zone_id,
            bounds_l93_m=plan.context_bounds_l93_m,
            source_revision=self.config.context_revision,
        )
        report(
            0.08,
            "zone_context_validated",
            "Contexte IGN validé — "
            + ", ".join(
                f"{name}: {count}"
                for name, count in sorted(zone_context.feature_counts.items())
            ),
            details={"feature_counts": dict(zone_context.feature_counts)},
        )

        asset_roots = {"review_batch": self.config.review_batch}
        shared_bundle = job_root / "shared" / "prototypes"
        completed = 0
        source_phase = {
            "download_mnt_started": (0.04, "MNT 0,5 m — téléchargement"),
            "download_mnt_completed": (0.14, "MNT 0,5 m reçu"),
            "download_mns_started": (0.16, "MNS 0,5 m — téléchargement"),
            "download_mns_completed": (0.26, "MNS 0,5 m reçu"),
            "download_orthophoto_started": (0.28, "Orthophoto 1 m — téléchargement"),
            "download_orthophoto_completed": (0.38, "Orthophoto 1 m reçue"),
            "sources_decoded": (0.43, "Sources raster décodées et co-enregistrées"),
            "tile_context_prepared": (0.47, "Contexte de placement découpé"),
            "sources_published": (0.50, "Sources de tuile validées"),
            "sources_reused": (0.50, "Sources de tuile revalidées et réutilisées"),
        }
        production_phase = {
            "terrain_compiled": (0.60, "Terrain FVTG compilé"),
            "ground_texture_baked": (0.68, "Texture sol orthophoto cuite"),
            "placement_measured": (0.78, "Placement MNS−MNT mesuré"),
            "terrain_usd_exported": (0.84, "Terrain OpenUSD exporté"),
            "scene_usd_built": (0.92, "Scène OpenUSD assemblée"),
            "tile_staging_validated": (0.97, "Package de tuile rehashé"),
            "tile_published": (1.00, "Package de tuile publié"),
            "tile_reused": (1.00, "Package existant revalidé et réutilisé"),
        }
        for tile_index, tile in enumerate(plan.tiles):
            tile_base = 0.08 + (tile_index / total) * 0.82
            tile_span = 0.82 / total

            def tile_report(
                phase: str,
                details: Mapping[str, Any],
                *,
                table: Mapping[str, tuple[float, str]],
                current_tile_index: int = tile_index,
                current_tile_base: float = tile_base,
                current_tile_span: float = tile_span,
                completed_before_tile: int = completed,
            ) -> None:
                local, label = table.get(phase, (0.0, phase))
                suffixes: list[str] = []
                if isinstance(details.get("byte_count"), int):
                    suffixes.append(f"{details['byte_count'] / 1_048_576:.1f} Mio")
                for key, label_key in (
                    ("building_count", "bâtiments"),
                    ("tree_count", "arbres"),
                    ("context_asset_count", "équipements"),
                ):
                    if isinstance(details.get(key), int):
                        suffixes.append(f"{details[key]} {label_key}")
                message = f"Tuile {current_tile_index + 1}/{total} — {label}"
                if suffixes:
                    message += " — " + ", ".join(suffixes)
                report(
                    current_tile_base + local * current_tile_span,
                    phase,
                    message,
                    completed_tiles=completed_before_tile,
                    details=details,
                )

            _status(
                job_root,
                state="producing",
                message=f"Production {tile.tile_id}",
                completed=completed,
                total=total,
            )
            yield f"Tuile {completed + 1}/{total} — {tile.tile_id}", None, []
            package_root = job_root / "packages" / tile.tile_id
            if package_root.exists():
                validate_simple_measured_tile_package(
                    package_root,
                    expected_request=_expected_request_from_receipt(
                        package_root,
                        plan=plan,
                        tile=tile,
                        asset_library=self.config.asset_library,
                        asset_roots=asset_roots,
                        portable_root=self.config.portable_root,
                        asset_bundle_root=shared_bundle,
                    ),
                    asset_library=self.config.asset_library,
                    asset_roots=asset_roots,
                )
                tile_report(
                    "tile_reused",
                    {"tile_id": tile.tile_id},
                    table=production_phase,
                )
                completed += 1
                continue
            source_root = job_root / "sources" / tile.tile_id
            sources = self.prepare_sources_fn(
                output_root=source_root,
                zone_id=plan.zone_id,
                tile_id=tile.tile_id,
                tile_origin_l93_m=tile.origin_l93_m,
                elevation_revision=self.config.elevation_revision,
                orthophoto_revision=self.config.orthophoto_revision,
                zone_context=job_root / ZONE_CONTEXT_NAME,
                fixed_asset_placements=[
                    placement
                    for placement in projected_fixed_assets
                    if placement["owner_tile_origin_l93_m"] == list(tile.origin_l93_m)
                ],
                progress_callback=lambda phase, details: tile_report(
                    phase, details, table=source_phase
                ),
            )
            package = self.produce_tile_fn(
                mnt_05m=sources.mnt,
                mns_05m=sources.mns,
                orthophoto_1m=sources.orthophoto,
                elevation_source_receipt=sources.elevation_receipt,
                orthophoto_source_receipt=sources.orthophoto_receipt,
                placement_context=sources.placement_context,
                asset_library=self.config.asset_library,
                asset_roots=asset_roots,
                portable_root=self.config.portable_root,
                asset_bundle_root=shared_bundle,
                output_root=package_root,
                zone_id=plan.zone_id,
                tile_id=tile.tile_id,
                tile_origin_l93_m=tile.origin_l93_m,
                progress_callback=lambda phase, details: tile_report(
                    phase, details, table=production_phase
                ),
            )
            validate_simple_measured_tile_package(
                package.output_root,
                expected_request=_expected_request_from_receipt(
                    package.output_root,
                    plan=plan,
                    tile=tile,
                    asset_library=self.config.asset_library,
                    asset_roots=asset_roots,
                    portable_root=self.config.portable_root,
                    asset_bundle_root=shared_bundle,
                ),
                asset_library=self.config.asset_library,
                asset_roots=asset_roots,
            )
            _copy_provenance(sources, job_root / "provenance" / tile.tile_id)
            _remove_sources(job_root, sources.root)
            report(
                tile_base + tile_span,
                "raw_sources_removed",
                f"Tuile {tile_index + 1}/{total} — sources raster brutes supprimées",
                completed_tiles=completed + 1,
            )
            completed += 1

        _status(
            job_root,
            state="packaging",
            message="Assemblage de la scène unifiée",
            completed=completed,
            total=total,
        )
        yield "Assemblage de la scène unifiée et du ZIP…", None, []
        _write_zone_stage(job_root, plan)
        receipt_assets = {
            **self.asset_summary,
            "asset_library": str(self.config.asset_library),
            "review_batch": str(self.config.review_batch),
        }
        receipt = _write_zone_receipt(
            job_root,
            plan,
            receipt_assets,
            self.config.portable_root,
            shared_bundle,
        )
        report(
            0.94,
            "gallery_render_started",
            "Rendu Blender — préparation des 20 captures de contrôle",
            completed_tiles=completed,
        )

        def gallery_progress(index: int, capture_id: str) -> None:
            report(
                0.94 + 0.045 * (index / CAPTURE_COUNT),
                "gallery_render",
                f"Rendu Blender — capture {index}/{CAPTURE_COUNT}: {capture_id}",
                completed_tiles=completed,
                details={"capture_index": index, "capture_count": CAPTURE_COUNT},
            )

        gallery = self.render_gallery_fn(job_root, True, gallery_progress)
        seal_map_upload_package(job_root)
        report(
            0.99, "zip_write", "Compression du pack autonome", completed_tiles=completed
        )
        archive = _write_zip(job_root, plan.zone_id)
        publication = None
        if self.config.dataset_id is not None:
            report(
                0.995,
                "dataset_publication",
                "Publication du pack validé dans la dataset FireViewer privée",
                completed_tiles=completed,
            )
            publication = _publish_dataset_entry(
                self.config,
                job_root=job_root,
                plan=plan,
                receipt=receipt,
                archive=archive,
            )
        _status(
            job_root,
            state="completed",
            message="Scène produite",
            completed=completed,
            total=total,
        )
        dataset_suffix = (
            f" Publié dans {publication['dataset_id']}." if publication else ""
        )
        completed_message = (
            f"Terminé — {completed} terrains, {receipt['building_count']} bâtiments, "
            f"{receipt['tree_count']} arbres, "
            f"{receipt.get('context_asset_count', 0)} équipements contextuels, "
            f"20 captures.{dataset_suffix} "
            f"Ouvrir {BLEND_NAME} ou {ENTRY_STAGE} après extraction."
        )
        yield (
            completed_message,
            str(archive),
            gallery,
        )


__all__ = [
    "ENTRY_STAGE",
    "SCHEMA",
    "ProductionConfig",
    "ProductionEngine",
    "SimpleProductionError",
    "TilePlan",
    "ZonePlan",
    "plan_zone",
    "validate_config",
    "validate_embedded_assets",
    "validate_embedded_runtime",
    "validate_production_config",
]
