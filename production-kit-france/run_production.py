"""Plan, execute and resume a complete FireViewer map production in France."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping, Sequence


CONFIG_SCHEMA = "fireviewer.france-map-production.v1"
PROFILE_SCHEMA = "fireviewer.france-map-quality-profile.v1"
STATE_SCHEMA = "fireviewer.france-map-production-state.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
BLENDER_DIRECTORY = REPO_ROOT / "blender"
UNITY_DIRECTORY = REPO_ROOT / "unity"
KIT_DIRECTORY = Path(__file__).resolve().parent
STAGE_NAMES = (
    "global_mnt",
    "plan_05m",
    "produce_05m",
    "source_lidar",
    "near_imagery",
    "far_rasters",
    "far_imagery",
    "vector_package",
    "blender_scene",
    "unity_catalog",
    "validate_catalog",
    "site_upload",
)


class ProductionError(RuntimeError):
    """Raised when a production contract or stage is invalid."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionError(f"JSON illisible: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProductionError(f"la racine JSON doit etre un objet: {path}")
    return value


def canonical_hash(*values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_accepted_profile(profile: Mapping[str, Any]) -> None:
    delivery = profile.get("delivery", {})
    if delivery not in ({}, None) and not isinstance(delivery, Mapping):
        raise ProductionError("profile.delivery doit etre un objet")
    mode = delivery.get("mode", "standard") if isinstance(delivery, Mapping) else "standard"
    if mode == "global_mnt_only":
        terrain = profile.get("terrain", {})
        if profile.get("profile_id") != "unity-v1-global-mnt-5m":
            raise ProductionError("le profil global_mnt_only doit avoir son identifiant verrouille")
        expected_mnt = {
            "global_mnt_resolution_m": 5.0,
            "far_resolution_m": 5.0,
            "source": "IGN LiDAR HD MNT WMS resample a 5 m",
        }
        for field, required in expected_mnt.items():
            if terrain.get(field) != required:
                raise ProductionError(f"le profil global_mnt_only est invalide: terrain.{field}")
        if profile.get("imagery", {}).get("far_resolution_m") != 2.0:
            raise ProductionError("le profil global_mnt_only doit conserver l'orthophoto FAR a 2 m")
        return
    expected = {
        "profile_id": "unity-v1-accepted",
        "terrain.detail_resolution_m": 0.5,
        "terrain.tile_size_m": 500,
        "terrain.processing_halo_m": 10.0,
        "terrain.far_resolution_m": 5.0,
        "imagery.far_resolution_m": 2.0,
        "imagery.mid_resolution_m": 0.5,
        "imagery.near_resolution_m": 0.2,
        "imagery.jpeg_quality": 92,
        "imagery.brightness": 0.78,
        "imagery.contrast": 1.08,
        "imagery.saturation": 1.12,
        "runtime.publish_distance_m": 600.0,
        "runtime.preload_radius_m": 750.0,
        "runtime.maximum_resident_tile_count": 16,
    }
    for dotted, required in expected.items():
        value: object = profile
        for part in dotted.split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if value != required:
            raise ProductionError(
                f"le profil n'est plus identique a la V1 acceptee: {dotted}={value!r}"
            )
    if (
        isinstance(delivery, Mapping)
        and delivery.get("mode", "standard") == "global_only"
    ):
        resolution = profile.get("terrain", {}).get(
            "required_native_lidar_resolution_m"
        )
        if resolution not in {0.2, 0.5}:
            raise ProductionError(
                "un profil global_only doit exiger une source LiDAR native a 0,2 m ou 0,5 m"
            )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_path(config_directory: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_directory / path
    return path.resolve()


def executable_value(config_directory: Path, value: object, fallback: str) -> str:
    if value is None or not str(value).strip():
        return fallback
    raw = str(value)
    if any(separator in raw for separator in ("/", "\\")):
        return str(resolve_path(config_directory, raw))
    return raw


def _coordinates(node: object) -> Iterable[tuple[float, float]]:
    if (
        isinstance(node, list)
        and len(node) >= 2
        and isinstance(node[0], (int, float))
        and isinstance(node[1], (int, float))
    ):
        yield float(node[0]), float(node[1])
        return
    if isinstance(node, list):
        for child in node:
            yield from _coordinates(child)


def geojson_metadata(path: Path, *, require_geometry: bool) -> dict[str, Any]:
    document = read_json(path)
    root_type = document.get("type")
    if root_type == "FeatureCollection":
        features = document.get("features")
        if not isinstance(features, list):
            raise ProductionError(f"FeatureCollection invalide: {path}")
        geometries = [
            item.get("geometry")
            for item in features
            if isinstance(item, dict) and isinstance(item.get("geometry"), dict)
        ]
        feature_count = len(features)
    elif root_type == "Feature":
        geometries = [document.get("geometry")]
        feature_count = 1
    elif root_type in {
        "Polygon",
        "MultiPolygon",
        "LineString",
        "MultiLineString",
        "Point",
        "MultiPoint",
    }:
        geometries = [document]
        feature_count = 1
    else:
        raise ProductionError(f"type GeoJSON non supporte dans {path}: {root_type!r}")
    coordinates = [
        coordinate
        for geometry in geometries
        if isinstance(geometry, dict)
        for coordinate in _coordinates(geometry.get("coordinates"))
    ]
    if require_geometry and not coordinates:
        raise ProductionError(f"aucune geometrie exploitable: {path}")
    if coordinates:
        xs, ys = zip(*coordinates, strict=True)
        bounds: list[float] | None = [min(xs), min(ys), max(xs), max(ys)]
        if not all(-100_000.0 <= x <= 1_400_000.0 for x in xs) or not all(
            6_000_000.0 <= y <= 7_300_000.0 for y in ys
        ):
            raise ProductionError(
                f"coordonnees hors enveloppe Lambert-93 metropolitaine: {path}"
            )
    else:
        bounds = None
    return {
        "path": str(path),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        "feature_count": feature_count,
        "coordinate_count": len(coordinates),
        "bounds_l93_m": bounds,
    }


def intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProductionError(f"{label} doit etre un objet")
    return value


def _require_path_list(value: object, label: str, *, non_empty: bool) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ProductionError(f"{label} doit etre une liste de chemins")
    if non_empty and not value:
        raise ProductionError(f"{label} ne peut pas etre vide")
    return value


def load_contract(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ProductionError("schema de configuration non supporte")
    config_directory = config_path.parent
    profile_path = resolve_path(config_directory, config.get("quality_profile"))
    profile = read_json(profile_path)
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ProductionError("schema de profil qualite non supporte")
    if profile.get("crs") != "EPSG:2154":
        raise ProductionError("le profil doit utiliser EPSG:2154")
    validate_accepted_profile(profile)

    zone = _require_mapping(config.get("zone"), "zone")
    inputs = _require_mapping(config.get("inputs"), "inputs")
    execution = _require_mapping(config.get("execution"), "execution")
    delivery_mode = execution.get("delivery_mode", "standard")
    if delivery_mode not in {"standard", "global_only", "global_mnt_only"}:
        raise ProductionError(
            "execution.delivery_mode doit etre standard, global_only ou global_mnt_only"
        )
    global_only = delivery_mode in {"global_only", "global_mnt_only"}
    direct_global_mnt = delivery_mode == "global_mnt_only"
    profile_mode = profile.get("delivery", {}).get("mode", "standard")
    if profile_mode != delivery_mode and not (
        profile_mode == "standard" and delivery_mode == "standard"
    ):
        raise ProductionError("le mode d'execution doit correspondre au profil qualite")
    for field, pattern in (
        ("zone_id", r"^[A-Z0-9][A-Z0-9_-]{2,63}$"),
        ("package_id", r"^[a-z0-9][a-z0-9-]{2,95}$"),
        ("artifact_slug", r"^[a-z0-9][a-z0-9-]{2,63}$"),
    ):
        if not re.fullmatch(pattern, str(zone.get(field, ""))):
            raise ProductionError(f"zone.{field} est invalide")
    revision = zone.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ProductionError("zone.revision doit etre un entier positif")
    origin = zone.get("origin_l93_m")
    if (
        not isinstance(origin, list)
        or len(origin) != 3
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in origin
        )
    ):
        raise ProductionError("zone.origin_l93_m doit contenir trois nombres finis")

    single_input_names = (
        ("aoi_l93",)
        if global_only
        else (
            "aoi_l93",
            "production_envelope_l93",
            "buildings_l93",
            "vegetation_l93",
        )
    )
    resolved_inputs: dict[str, Any] = {}
    for name in single_input_names:
        if not isinstance(inputs.get(name), str) or not inputs[name]:
            raise ProductionError(f"inputs.{name} est obligatoire")
        resolved_inputs[name] = resolve_path(config_directory, inputs[name])
    for name in (
        ()
        if global_only
        else (
            "roads_l93",
            "water_courses_l93",
            "water_segments_l93",
            "water_surfaces_l93",
        )
    ):
        values = _require_path_list(
            inputs.get(name), name, non_empty=name == "roads_l93"
        )
        resolved_inputs[name] = [
            resolve_path(config_directory, item) for item in values
        ]
    if not global_only and not any(
        resolved_inputs[name]
        for name in ("water_courses_l93", "water_segments_l93", "water_surfaces_l93")
    ):
        raise ProductionError(
            "au moins un fichier hydrographique est requis, meme s'il contient zero entite"
        )

    all_paths = [
        *(resolved_inputs[name] for name in single_input_names),
        *(
            path
            for name in (
                ()
                if global_only
                else (
                    "roads_l93",
                    "water_courses_l93",
                    "water_segments_l93",
                    "water_surfaces_l93",
                )
            )
            for path in resolved_inputs[name]
        ),
    ]
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise ProductionError("sources locales absentes:\n- " + "\n- ".join(missing))

    aoi_metadata = geojson_metadata(resolved_inputs["aoi_l93"], require_geometry=True)
    envelope_metadata = (
        geojson_metadata(
            resolved_inputs["production_envelope_l93"], require_geometry=True
        )
        if not global_only
        else None
    )
    aoi_bounds = aoi_metadata["bounds_l93_m"]
    assert isinstance(aoi_bounds, list)
    width = aoi_bounds[2] - aoi_bounds[0]
    height = aoi_bounds[3] - aoi_bounds[1]
    limits = _require_mapping(profile.get("limits"), "profile.limits")
    maximum_area = float(limits.get("maximum_aoi_area_km2", 0.0))
    bounding_area_km2 = width * height / 1_000_000.0
    if width <= 0 or height <= 0 or bounding_area_km2 > maximum_area:
        raise ProductionError(
            f"emprise AOI invalide ou superieure a {maximum_area:g} km2"
        )
    if envelope_metadata is not None and not intersects(
        aoi_bounds, envelope_metadata["bounds_l93_m"]
    ):
        raise ProductionError("l'enveloppe de production n'intersecte pas l'AOI")
    if not (
        aoi_bounds[0] - 1500 <= float(origin[0]) <= aoi_bounds[2] + 1500
        and aoi_bounds[1] - 1500 <= float(origin[1]) <= aoi_bounds[3] + 1500
    ):
        raise ProductionError("l'origine XY est eloignee de l'AOI")

    source_metadata: dict[str, Any] = {
        "aoi_l93": aoi_metadata,
    }
    if envelope_metadata is not None:
        source_metadata["production_envelope_l93"] = envelope_metadata
    for name in () if global_only else ("buildings_l93", "vegetation_l93"):
        source_metadata[name] = geojson_metadata(
            resolved_inputs[name], require_geometry=False
        )
    for name in (
        ()
        if global_only
        else (
            "roads_l93",
            "water_courses_l93",
            "water_segments_l93",
            "water_surfaces_l93",
        )
    ):
        source_metadata[name] = [
            geojson_metadata(path, require_geometry=False)
            for path in resolved_inputs[name]
        ]

    attention = config.get("attention_zones")
    if not isinstance(attention, list) or not attention:
        raise ProductionError("attention_zones doit contenir au moins une zone")
    identifiers: set[str] = set()
    normalized_attention: list[dict[str, Any]] = []
    for item in attention:
        zone_item = _require_mapping(item, "attention_zones[]")
        identifier = str(zone_item.get("id", ""))
        bounds = zone_item.get("bounds_l93_m")
        if (
            not re.fullmatch(r"^[a-z0-9][a-z0-9_-]{1,63}$", identifier)
            or identifier in identifiers
            or not isinstance(bounds, list)
            or len(bounds) != 4
            or not all(isinstance(value, (int, float)) for value in bounds)
            or float(bounds[2]) <= float(bounds[0])
            or float(bounds[3]) <= float(bounds[1])
            or not intersects(aoi_bounds, [float(value) for value in bounds])
        ):
            raise ProductionError(f"zone d'attention invalide: {identifier!r}")
        identifiers.add(identifier)
        normalized_attention.append(
            {
                "id": identifier,
                "label": str(zone_item.get("label", identifier)),
                "bounds_l93_m": [float(value) for value in bounds],
            }
        )

    artifact_root = resolve_path(config_directory, execution.get("artifact_root"))
    expected_count = execution.get("expected_source_tile_count")
    if expected_count is not None and (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
    ):
        raise ProductionError("expected_source_tile_count doit etre null ou positif")
    blender_requested = bool(execution.get("build_blender_scene", True))
    near_lod_enabled = execution.get("near_lod_enabled", True)
    if not isinstance(near_lod_enabled, bool):
        raise ProductionError("execution.near_lod_enabled doit etre un booleen")
    if global_only and (near_lod_enabled or blender_requested):
        raise ProductionError(
            "global_only impose near_lod_enabled=false et build_blender_scene=false"
        )
    blender_executable = execution.get("blender_executable")
    if blender_requested and not blender_executable:
        raise ProductionError(
            "blender_executable est obligatoire quand build_blender_scene vaut true"
        )
    validation_paths: dict[str, Path] = {}
    for field in ("unity_validation_receipt", "unity_preview_png"):
        value = execution.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ProductionError(f"execution.{field} est obligatoire")
        validation_paths[field] = resolve_path(config_directory, value)

    artifact_config = deepcopy(config)
    artifact_config.get("execution", {}).pop("near_lod_enabled", None)
    artifact_config.get("execution", {}).pop("delivery_mode", None)
    delivery_policy = {"near_lod_enabled": near_lod_enabled}
    if global_only:
        delivery_policy.update(
            {
                "mode": delivery_mode,
                **(
                    {"required_native_lidar_resolution_m": profile.get("terrain", {}).get("required_native_lidar_resolution_m")}
                    if not direct_global_mnt
                    else {"terrain_surface": "mnt_wms_resampled", "resolution_m": 5.0}
                ),
            }
        )
    return {
        "config_path": config_path,
        "config": config,
        "profile_path": profile_path,
        "profile": profile,
        "config_hash": canonical_hash(
            artifact_config, profile, {"source_metadata": source_metadata}
        ),
        "delivery_policy": delivery_policy,
        "delivery_policy_hash": canonical_hash(delivery_policy),
        "zone": zone,
        "inputs": resolved_inputs,
        "attention_zones": normalized_attention,
        "execution": execution,
        "artifact_root": artifact_root,
        "aoi_bounds_l93_m": aoi_bounds,
        "aoi_bounding_area_km2": bounding_area_km2,
        "source_metadata": source_metadata,
        "python": executable_value(
            config_directory, execution.get("python_executable"), sys.executable
        ),
        "blender": (
            executable_value(config_directory, blender_executable, "blender")
            if blender_requested
            else None
        ),
        "build_blender_scene": blender_requested,
        "near_lod_enabled": near_lod_enabled,
        "delivery_mode": delivery_mode,
        "global_only": global_only,
        "direct_global_mnt": direct_global_mnt,
        "expected_source_tile_count": expected_count,
        **validation_paths,
    }


def detail_zone_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    zone = contract["zone"]
    return {
        "schema_version": "1.0",
        "parent_package_id": zone["package_id"],
        "horizontal_crs": "EPSG:2154",
        "vertical_reference": "NGF-IGN69",
        "detail_resolution_metres": 0.5,
        "zones": [
            {
                "id": item["id"],
                "label": item["label"],
                "bounds_l93_metres": item["bounds_l93_m"],
            }
            for item in contract["attention_zones"]
        ],
    }


def _append_paths(command: list[str], option: str, paths: Iterable[Path]) -> None:
    for path in paths:
        command.extend((option, str(path)))


def build_stages(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    profile = contract["profile"]
    limits = profile["limits"]
    terrain = profile["terrain"]
    imagery = profile["imagery"]
    vectors = profile["vectors"]
    inputs = contract["inputs"]
    root: Path = contract["artifact_root"]
    python = str(contract["python"])
    origin = [str(value) for value in contract["zone"]["origin_l93_m"]]
    global_root = root / "global-05m"
    production_manifest = global_root / "production-manifest.json"
    terrain_root = root / "terrain"
    blender_root = root / "blender"
    detail_contract = root / ".production/detail-zones.json"
    slug = contract["zone"]["artifact_slug"]
    orthophoto_tif = blender_root / f"{slug}-orthophoto-2m.tif"
    orthophoto_jpg = blender_root / f"{slug}-orthophoto-2m.jpg"
    vector_package = blender_root / f"{slug}-global-control.json.gz"
    blender_scene = blender_root / f"{slug}-control.blend"
    unity_root = root / "unity-remote-catalog-v1-complete"
    site_root = root / "site-upload" / contract["zone"]["package_id"]

    base_tile_command = [
        python,
        str(BLENDER_DIRECTORY / "prepare_global_05m.py"),
        "--aoi",
        str(inputs["aoi_l93"]),
        "--output-root",
        str(global_root),
        "--origin",
        *origin,
        "--output-tile-size-m",
        str(terrain.get("tile_size_m", 500)),
        "--halo-m",
        str(terrain.get("processing_halo_m", 10.0)),
    ]
    expected = contract["expected_source_tile_count"]
    if expected is not None:
        base_tile_command.extend(("--expected-source-tile-count", str(expected)))
    plan_command = [*base_tile_command, "--write-plan"]
    if contract["global_only"]:
        plan_command.append("--source-only-plan")
    produce_command = [
        *base_tile_command,
        "--download-workers",
        str(limits["download_workers"]),
        "--package-workers",
        str(limits["package_workers"]),
        "--memory-budget-gib",
        str(limits["package_memory_budget_gib"]),
        "--minimum-free-gib",
        str(limits["minimum_free_gib"]),
        "--execute",
    ]
    if not contract["global_only"]:
        _append_paths(produce_command, "--exclude-polygons", [inputs["buildings_l93"]])
        _append_paths(
            produce_command, "--exclude-polygons", inputs["water_surfaces_l93"]
        )
        _append_paths(produce_command, "--exclude-lines", inputs["roads_l93"])
        _append_paths(produce_command, "--exclude-lines", inputs["water_courses_l93"])
        _append_paths(produce_command, "--exclude-lines", inputs["water_segments_l93"])
    near_command = (
        [
            python,
            str(BLENDER_DIRECTORY / "prepare_global_05m.py"),
            "--output-root",
            str(global_root),
            "--download-workers",
            str(limits["download_workers"]),
            "--minimum-free-gib",
            str(limits["minimum_free_gib"]),
            "--all-near-orthophoto-tiles",
            "--execute-near-orthophoto",
        ]
        if contract["near_lod_enabled"]
        else None
    )
    far_command = [
        python,
        str(KIT_DIRECTORY / "build_far_rasters.py"),
        "--production-manifest",
        str(production_manifest),
        "--aoi",
        str(inputs["aoi_l93"]),
        "--output-dir",
        str(terrain_root),
        "--resolution-m",
        str(terrain["far_resolution_m"]),
    ]
    imagery_command = [
        python,
        str(BLENDER_DIRECTORY / "fetch_ign_orthophoto.py"),
        "--bounds",
        *[str(value) for value in contract["aoi_bounds_l93_m"]],
        "--resolution-m",
        str(imagery["far_resolution_m"]),
        "--output",
        str(orthophoto_tif),
        "--jpeg-output",
        str(orthophoto_jpg),
        "--jpeg-quality",
        str(imagery["jpeg_quality"]),
        "--jpeg-brightness",
        str(imagery["brightness"]),
        "--jpeg-contrast",
        str(imagery["contrast"]),
        "--jpeg-saturation",
        str(imagery["saturation"]),
        "--allow-large-download",
        "--execute",
    ]
    if contract["direct_global_mnt"]:
        global_mnt_command = [
            python,
            "-B",
            str(BLENDER_DIRECTORY / "fetch_ign_global_mnt.py"),
            "--bounds",
            *[str(value) for value in contract["aoi_bounds_l93_m"]],
            "--resolution-m",
            str(terrain["global_mnt_resolution_m"]),
            "--output",
            str(terrain_root / "mnt-global.cog.tif"),
            "--aoi",
            str(inputs["aoi_l93"]),
            "--origin",
            *origin,
            "--production-manifest",
            str(production_manifest),
            "--allow-large-download",
            "--execute",
        ]
        unity_command = [
            python,
            "-B",
            str(UNITY_DIRECTORY / "export_remote_catalog.py"),
            "--artifact-root",
            str(root),
            "--output-root",
            str(unity_root),
            "--production-manifest",
            str(production_manifest),
            "--far-terrain",
            str(terrain_root / "mnt-global.cog.tif"),
            "--far-imagery",
            str(orthophoto_jpg),
            "--far-imagery-resolution-m",
            str(imagery["far_resolution_m"]),
            "--global-only",
        ]
        validate_command = [
            python,
            "-B",
            str(UNITY_DIRECTORY / "validate_remote_catalog.py"),
            "--artifact-root",
            str(root),
            "--output-root",
            str(unity_root),
        ]
        upload_command = [
            python,
            "-B",
            str(UNITY_DIRECTORY / "export_site_upload_package.py"),
            "--source-root",
            str(unity_root),
            "--output-root",
            str(site_root),
            "--package-id",
            str(contract["zone"]["package_id"]),
            "--zone-id",
            str(contract["zone"]["zone_id"]),
            "--revision",
            str(contract["zone"]["revision"]),
            "--unity-validation-receipt",
            str(contract["unity_validation_receipt"]),
            "--unity-preview-png",
            str(contract["unity_preview_png"]),
        ]
        return [
            {"name": "global_mnt", "command": global_mnt_command, "requires": []},
            {"name": "far_imagery", "command": imagery_command, "requires": []},
            {"name": "unity_catalog", "command": unity_command, "requires": ["global_mnt", "far_imagery"]},
            {"name": "validate_catalog", "command": validate_command, "requires": ["unity_catalog"]},
            {"name": "site_upload", "command": upload_command, "requires": ["validate_catalog"]},
        ]
    if contract["global_only"]:
        source_command = [
            *base_tile_command,
            "--source-only-plan",
            "--download-workers",
            str(limits["download_workers"]),
            "--package-workers",
            "1",
            "--memory-budget-gib",
            str(limits["package_memory_budget_gib"]),
            "--minimum-free-gib",
            str(limits["minimum_free_gib"]),
            "--phase",
            "sources",
            "--required-elevation-resolution-m",
            str(terrain["required_native_lidar_resolution_m"]),
            "--execute",
        ]
        far_command.append("--allow-source-only")
        unity_command = [
            python,
            str(UNITY_DIRECTORY / "export_remote_catalog.py"),
            "--artifact-root",
            str(root),
            "--output-root",
            str(unity_root),
            "--production-manifest",
            str(production_manifest),
            "--far-terrain",
            str(terrain_root / "mnt-global.cog.tif"),
            "--far-imagery",
            str(orthophoto_jpg),
            "--far-imagery-resolution-m",
            str(imagery["far_resolution_m"]),
            "--global-only",
        ]
        validate_command = [
            python,
            str(UNITY_DIRECTORY / "validate_remote_catalog.py"),
            "--artifact-root",
            str(root),
            "--output-root",
            str(unity_root),
        ]
        upload_command = [
            python,
            str(UNITY_DIRECTORY / "export_site_upload_package.py"),
            "--source-root",
            str(unity_root),
            "--output-root",
            str(site_root),
            "--package-id",
            str(contract["zone"]["package_id"]),
            "--zone-id",
            str(contract["zone"]["zone_id"]),
            "--revision",
            str(contract["zone"]["revision"]),
            "--unity-validation-receipt",
            str(contract["unity_validation_receipt"]),
            "--unity-preview-png",
            str(contract["unity_preview_png"]),
        ]
        return [
            {"name": "plan_05m", "command": plan_command, "requires": []},
            {
                "name": "source_lidar",
                "command": source_command,
                "requires": ["plan_05m"],
            },
            {
                "name": "far_rasters",
                "command": far_command,
                "requires": ["source_lidar"],
            },
            {
                "name": "far_imagery",
                "command": imagery_command,
                "requires": ["plan_05m"],
            },
            {
                "name": "unity_catalog",
                "command": unity_command,
                "requires": ["far_rasters", "far_imagery"],
            },
            {
                "name": "validate_catalog",
                "command": validate_command,
                "requires": ["unity_catalog"],
            },
            {
                "name": "site_upload",
                "command": upload_command,
                "requires": ["validate_catalog"],
            },
        ]
    vector_command = [
        python,
        str(BLENDER_DIRECTORY / "prepare_control_package.py"),
        "--mnt",
        str(terrain_root / "mnt-global.cog.tif"),
        "--mns",
        str(terrain_root / "mns-global.cog.tif"),
        "--perimeter",
        str(inputs["production_envelope_l93"]),
        "--perimeter-crs",
        "EPSG:2154",
        "--buildings",
        str(inputs["buildings_l93"]),
        "--buildings-crs",
        "EPSG:2154",
        "--vegetation",
        str(inputs["vegetation_l93"]),
        "--vegetation-crs",
        "EPSG:2154",
        "--roads-crs",
        "EPSG:2154",
        "--water-crs",
        "EPSG:2154",
        "--terrain-step",
        str(vectors["terrain_step"]),
        "--buffer-m",
        str(vectors["buffer_m"]),
        "--building-simplify-m",
        str(vectors["building_simplify_m"]),
        "--minimum-visible-building-wall-m",
        str(vectors["minimum_visible_building_wall_m"]),
        "--road-offset-m",
        str(vectors["road_offset_m"]),
        "--water-course-offset-m",
        str(vectors["water_course_offset_m"]),
        "--water-segment-offset-m",
        str(vectors["water_segment_offset_m"]),
        "--water-surface-offset-m",
        str(vectors["water_surface_offset_m"]),
        "--vegetation-building-clearance-m",
        str(vectors["vegetation_building_clearance_m"]),
        "--vegetation-road-clearance-m",
        str(vectors["vegetation_road_clearance_m"]),
        "--vegetation-water-clearance-m",
        str(vectors["vegetation_water_clearance_m"]),
        "--mid-tree-min-height-m",
        str(vectors["mid_tree_min_height_m"]),
        "--mid-tree-spacing-m",
        str(vectors["mid_tree_spacing_m"]),
        "--mid-tree-local-max-radius-m",
        str(vectors["mid_tree_local_max_radius_m"]),
        "--mid-tree-max-count",
        str(vectors["mid_tree_max_count"]),
        "--origin-x",
        origin[0],
        "--origin-y",
        origin[1],
        "--origin-z",
        origin[2],
        "--output",
        str(vector_package),
    ]
    _append_paths(vector_command, "--roads", inputs["roads_l93"])
    _append_paths(vector_command, "--water-courses", inputs["water_courses_l93"])
    _append_paths(vector_command, "--water-segments", inputs["water_segments_l93"])
    _append_paths(vector_command, "--water-surfaces", inputs["water_surfaces_l93"])
    focus_bounds = contract["attention_zones"][0]["bounds_l93_m"]
    focus = [
        (focus_bounds[0] + focus_bounds[2]) / 2.0,
        (focus_bounds[1] + focus_bounds[3]) / 2.0,
    ]
    scene_command = (
        [
            str(contract["blender"]),
            "--background",
            "--python",
            str(BLENDER_DIRECTORY / "build_control_scene.py"),
            "--",
            "--package",
            str(vector_package),
            "--output",
            str(blender_scene),
            "--orthophoto-source",
            str(orthophoto_tif.with_suffix(".source.json")),
            "--tile-index",
            str(production_manifest),
            "--tile-load-mode",
            "visible",
            "--tile-focus-l93",
            str(focus[0]),
            str(focus[1]),
            "--tile-visible-radius-m",
            "2000",
            "--maximum-resident-tile-count",
            str(profile["runtime"]["maximum_resident_tile_count"]),
        ]
        if contract["build_blender_scene"]
        else None
    )
    unity_command = [
        python,
        str(UNITY_DIRECTORY / "run_export_batches.py"),
        "--artifact-root",
        str(root),
        "--output-root",
        str(unity_root),
        "--production-manifest",
        str(production_manifest),
        "--global-vector-package",
        str(vector_package),
        "--far-terrain",
        str(terrain_root / "mnt-global.cog.tif"),
        "--far-imagery",
        str(orthophoto_jpg),
        "--far-imagery-resolution-m",
        str(imagery["far_resolution_m"]),
        "--detail-zones",
        str(detail_contract),
        "--workers",
        str(limits["unity_workers"]),
        "--batch-size",
        str(limits["unity_batch_size"]),
    ]
    if not contract["near_lod_enabled"]:
        unity_command.append("--disable-near-lod")
    validate_command = [
        python,
        str(UNITY_DIRECTORY / "validate_remote_catalog.py"),
        "--artifact-root",
        str(root),
        "--output-root",
        str(unity_root),
    ]
    upload_command = [
        python,
        str(UNITY_DIRECTORY / "export_site_upload_package.py"),
        "--source-root",
        str(unity_root),
        "--output-root",
        str(site_root),
        "--package-id",
        str(contract["zone"]["package_id"]),
        "--zone-id",
        str(contract["zone"]["zone_id"]),
        "--revision",
        str(contract["zone"]["revision"]),
        "--unity-validation-receipt",
        str(contract["unity_validation_receipt"]),
        "--unity-preview-png",
        str(contract["unity_preview_png"]),
    ]
    return [
        {"name": "plan_05m", "command": plan_command, "requires": []},
        {"name": "produce_05m", "command": produce_command, "requires": ["plan_05m"]},
        {
            "name": "near_imagery",
            "command": near_command,
            "requires": ["produce_05m"],
            "optional": not contract["near_lod_enabled"],
        },
        {"name": "far_rasters", "command": far_command, "requires": ["produce_05m"]},
        {"name": "far_imagery", "command": imagery_command, "requires": ["plan_05m"]},
        {
            "name": "vector_package",
            "command": vector_command,
            "requires": ["far_rasters"],
        },
        {
            "name": "blender_scene",
            "command": scene_command,
            "requires": ["near_imagery", "far_imagery", "vector_package"],
            "optional": not contract["build_blender_scene"],
        },
        {
            "name": "unity_catalog",
            "command": unity_command,
            "requires": ["near_imagery", "far_imagery", "vector_package"],
        },
        {
            "name": "validate_catalog",
            "command": validate_command,
            "requires": ["unity_catalog"],
        },
        {
            "name": "site_upload",
            "command": upload_command,
            "requires": ["validate_catalog"],
        },
    ]


def _gzip_is_readable(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 1024:
            return False
        with gzip.open(path, "rb") as stream:
            return stream.read(1) == b"{"
    except OSError:
        return False


def stage_artifact_complete(name: str, contract: Mapping[str, Any]) -> bool:
    root: Path = contract["artifact_root"]
    slug = contract["zone"]["artifact_slug"]
    manifest_path = root / "global-05m/production-manifest.json"
    if name == "global_mnt":
        terrain_path = root / "terrain/mnt-global.cog.tif"
        source_path = terrain_path.with_suffix(".source.json")
        if not terrain_path.is_file() or not source_path.is_file() or not manifest_path.is_file():
            return False
        source = read_json(source_path)
        manifest = read_json(manifest_path)
        return (
            contract["direct_global_mnt"]
            and source.get("schema") == "fireviewer.ign-global-mnt-source.v1"
            and source.get("status") == "downloaded_and_validated"
            and source.get("terrain_surface") == "mnt_wms_resampled"
            and source.get("request", {}).get("nominal_resolution_m") == 5.0
            and source.get("output", {}).get("sha256") == sha256_file(terrain_path)
            and manifest.get("status") == "ready"
            and manifest.get("tiling", {}).get("mode") == "global_mnt_only"
            and manifest.get("aoi", {}).get("sha256") == sha256_file(contract["inputs"]["aoi_l93"])
        )
    if name == "plan_05m":
        if not manifest_path.is_file():
            return False
        manifest = read_json(manifest_path)
        terrain = contract["profile"]["terrain"]
        return (
            manifest.get("schema") == "fireviewer.global-05m-production-manifest.v1"
            and manifest.get("aoi", {}).get("sha256")
            == sha256_file(contract["inputs"]["aoi_l93"])
            and manifest.get("origin_l93_m")
            == [float(value) for value in contract["zone"]["origin_l93_m"]]
            and manifest.get("tiling", {}).get("output_tile_size_m")
            == terrain["tile_size_m"]
            and manifest.get("tiling", {}).get("halo_m") == terrain["processing_halo_m"]
        )
    if name == "produce_05m":
        return (
            stage_artifact_complete("plan_05m", contract)
            and read_json(manifest_path).get("status") == "ready"
        )
    if name == "source_lidar":
        return (
            contract["global_only"]
            and stage_artifact_complete("plan_05m", contract)
            and all(
                source.get("status", {}).get("state") == "ready"
                for source in read_json(manifest_path).get("source_tiles", [])
            )
        )
    if name == "near_imagery":
        if not stage_artifact_complete("produce_05m", contract):
            return False
        if not contract["near_lod_enabled"]:
            return True
        tiles = read_json(manifest_path).get("tiles", [])
        return bool(tiles) and all(
            tile.get("near_orthophoto_status", {}).get("state") == "ready"
            for tile in tiles
        )
    if name == "far_rasters":
        path = root / "terrain/far-raster-manifest.json"
        if not path.is_file() or not manifest_path.is_file():
            return False
        record = read_json(path)
        return (
            record.get("schema") == "fireviewer.far-raster-mosaic.v1"
            and record.get("aoi", {}).get("sha256")
            == sha256_file(contract["inputs"]["aoi_l93"])
            and record.get("production_manifest", {}).get("sha256")
            == sha256_file(manifest_path)
            and record.get("resolution_m")
            == contract["profile"]["terrain"]["far_resolution_m"]
            and bool(record.get("source_only", False)) is contract["global_only"]
        )
    if name == "far_imagery":
        image = root / "blender" / f"{slug}-orthophoto-2m.jpg"
        source = root / "blender" / f"{slug}-orthophoto-2m.source.json"
        if not image.is_file() or not source.is_file():
            return False
        record = read_json(source)
        request = record.get("request", {})
        transform = record.get("jpeg_display_transform", {})
        imagery = contract["profile"]["imagery"]
        jpeg_outputs = [
            output
            for output in record.get("outputs", [])
            if output.get("file_name") == image.name
        ]
        return (
            record.get("schema") == "fireviewer.ign-orthophoto-source.v1"
            and request.get("bounds_l93_m") == contract["aoi_bounds_l93_m"]
            and request.get("nominal_resolution_m") == imagery["far_resolution_m"]
            and transform.get("brightness_multiplier") == imagery["brightness"]
            and transform.get("contrast_multiplier") == imagery["contrast"]
            and transform.get("saturation_multiplier") == imagery["saturation"]
            and len(jpeg_outputs) == 1
            and jpeg_outputs[0].get("sha256") == sha256_file(image)
        )
    if name == "vector_package":
        path = root / "blender" / f"{slug}-global-control.json.gz"
        return _gzip_is_readable(path)
    if name == "blender_scene":
        if not contract["build_blender_scene"]:
            return True
        path = root / "blender" / f"{slug}-control.blend"
        return path.is_file() and path.stat().st_size > 1024
    if name == "unity_catalog":
        path = root / "unity-remote-catalog-v1-complete/catalog.json"
        if not path.is_file() or not manifest_path.is_file():
            return False
        catalog = read_json(path)
        manifest = read_json(manifest_path)
        detail = catalog.get("lod_policy", {}).get("detail", {})
        if contract["global_only"]:
            return (
                catalog.get("schema") == "fireviewer.remote-tile-catalog.v1"
                and detail.get("enabled") is False
                and detail.get("mode") == "global_only"
                and int(catalog.get("exported_detail_tile_count", -1)) == 0
                and catalog.get("tiles") == []
            )
        return (
            catalog.get("schema") == "fireviewer.remote-tile-catalog.v1"
            and int(catalog.get("exported_detail_tile_count", -1))
            == len(manifest.get("tiles", []))
            and bool(
                catalog.get("lod_policy", {})
                .get("detail", {})
                .get("near_disabled", False)
            )
            is (not contract["near_lod_enabled"])
        )
    if name == "site_upload":
        path = (
            root
            / "site-upload"
            / contract["zone"]["package_id"]
            / "package-manifest.json"
        )
        receipt_path: Path = contract["unity_validation_receipt"]
        preview_path: Path = contract["unity_preview_png"]
        if (
            not path.is_file()
            or not receipt_path.is_file()
            or not preview_path.is_file()
        ):
            return False
        receipt = read_json(receipt_path)
        manual_validation = read_json(path).get("manual_unity_validation")
        return bool(
            read_json(path).get("package_id") == contract["zone"]["package_id"]
            and receipt.get("decision") == "accepted"
            and receipt.get("catalog_sha256")
            == sha256_file(root / "unity-remote-catalog-v1-complete/catalog.json")
            and receipt.get("preview_sha256") == sha256_file(preview_path)
            and manual_validation == receipt
        )
    return False


def run_command(command: Sequence[str]) -> None:
    print(f"\n>>> {subprocess.list2cmdline(list(command))}", flush=True)
    completed = subprocess.run(list(command), cwd=REPO_ROOT, check=False)
    if completed.returncode:
        raise ProductionError(f"commande terminee avec le code {completed.returncode}")


def load_state(path: Path, config_hash: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": STATE_SCHEMA,
            "config_hash": config_hash,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "stages": {},
        }
    state = read_json(path)
    if state.get("schema") != STATE_SCHEMA:
        raise ProductionError("schema d'etat de production non supporte")
    if state.get("config_hash") != config_hash:
        raise ProductionError(
            "la configuration ou le profil ont change; utiliser un nouvel artifact_root"
        )
    return state


def plan_record(
    contract: Mapping[str, Any], stages: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": "fireviewer.france-map-production-plan.v1",
        "config": str(contract["config_path"]),
        "profile": str(contract["profile_path"]),
        "profile_id": contract["profile"]["profile_id"],
        "config_hash": contract["config_hash"],
        "delivery_policy": contract["delivery_policy"],
        "delivery_policy_hash": contract["delivery_policy_hash"],
        "artifact_root": str(contract["artifact_root"]),
        "aoi_bounds_l93_m": contract["aoi_bounds_l93_m"],
        "aoi_bounding_area_km2": contract["aoi_bounding_area_km2"],
        "source_metadata": contract["source_metadata"],
        "stage_count": len(stages),
        "stages": [
            {
                "name": stage["name"],
                "requires": stage.get("requires", []),
                "optional": bool(stage.get("optional", False)),
                "command": stage.get("command"),
                "already_complete": stage_artifact_complete(stage["name"], contract),
            }
            for stage in stages
        ],
        "final_upload_root": str(
            contract["artifact_root"] / "site-upload" / contract["zone"]["package_id"]
        ),
    }


def execute(
    contract: Mapping[str, Any], stages: Sequence[Mapping[str, Any]], selected: str
) -> None:
    root: Path = contract["artifact_root"]
    state_path = root / ".production/state.json"
    root.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path, str(contract["config_hash"]))
    previous_delivery_hash = state.get("delivery_policy_hash")
    if previous_delivery_hash not in (None, contract["delivery_policy_hash"]) and any(
        state["stages"].get(name, {}).get("status") == "complete"
        for name in ("near_imagery", "unity_catalog", "validate_catalog", "site_upload")
    ):
        raise ProductionError(
            "la politique LOD a change apres la production des artefacts de livraison"
        )
    state["delivery_policy_hash"] = contract["delivery_policy_hash"]
    atomic_json(root / ".production/detail-zones.json", detail_zone_contract(contract))
    atomic_json(root / ".production/preflight.json", plan_record(contract, stages))
    selected_stages = (
        list(stages)
        if selected == "all"
        else [stage for stage in stages if stage["name"] == selected]
    )
    if not selected_stages:
        raise ProductionError(f"etape inconnue: {selected}")
    for stage in selected_stages:
        name = str(stage["name"])
        if stage.get("optional") and stage.get("command") is None:
            state["stages"][name] = {"status": "skipped", "updated_at_utc": utc_now()}
            atomic_json(state_path, state)
            continue
        for dependency in stage.get("requires", []):
            dependency_state = state["stages"].get(dependency, {}).get("status")
            if dependency_state not in {
                "complete",
                "skipped",
            } and not stage_artifact_complete(dependency, contract):
                raise ProductionError(
                    f"l'etape {name} requiert d'abord l'etape {dependency}"
                )
        if stage_artifact_complete(name, contract) and name != "validate_catalog":
            state["stages"][name] = {
                "status": "complete",
                "source": "validated_existing_artifact",
                "updated_at_utc": utc_now(),
            }
            state["updated_at_utc"] = utc_now()
            atomic_json(state_path, state)
            print(f"[{name}] deja produit et valide", flush=True)
            continue
        state["stages"][name] = {"status": "running", "updated_at_utc": utc_now()}
        state["updated_at_utc"] = utc_now()
        atomic_json(state_path, state)
        try:
            run_command(stage["command"])
            if name != "validate_catalog" and not stage_artifact_complete(
                name, contract
            ):
                raise ProductionError(f"sortie attendue non valide apres {name}")
        except Exception as exc:
            state["stages"][name] = {
                "status": "failed",
                "error": str(exc),
                "updated_at_utc": utc_now(),
            }
            state["updated_at_utc"] = utc_now()
            atomic_json(state_path, state)
            raise
        state["stages"][name] = {"status": "complete", "updated_at_utc": utc_now()}
        state["updated_at_utc"] = utc_now()
        atomic_json(state_path, state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan", action="store_true", help="Preflight sans ecriture ni reseau"
    )
    mode.add_argument(
        "--execute", action="store_true", help="Execute et reprend les etapes"
    )
    parser.add_argument("--stage", choices=("all", *STAGE_NAMES), default="all")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_contract(args.config)
    stages = build_stages(contract)
    if args.plan:
        print(json.dumps(plan_record(contract, stages), ensure_ascii=False, indent=2))
        return 0
    execute(contract, stages, args.stage)
    print(
        json.dumps(
            {
                "status": "complete" if args.stage == "all" else "stage_complete",
                "stage": args.stage,
                "upload_root": str(
                    contract["artifact_root"]
                    / "site-upload"
                    / contract["zone"]["package_id"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProductionError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
