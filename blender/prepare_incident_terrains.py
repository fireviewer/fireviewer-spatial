"""Plan the six terrain-first FireViewer incident productions.

Every case is rebuilt from native IGN MNT/MNS. Lightweight 2D ground and
structured-overlay maps are composed from approved classified context layers,
without orthophoto or other aerial imagery. The square AOIs are conservative: they contain every
currently referenced public incident geometry plus a 5 km safety margin, then
snap outward to the 500 m production grid. No previous Die terrain, asset or
placement is reused.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from prepare_global_05m import Global05mConfig, build_plan, sha256_file


CATALOG_SCHEMA = "fireviewer.incident-terrain-catalog.v1"
SOURCE_CRS = "EPSG:2154"
TILE_SIZE_M = 500
SAFETY_MARGIN_M = 5_000
PUBLIC_INDEX_URL = "https://fireviewer-api.vercel.app/api/v1/incidents/recent"
GROUND_SURFACE_SCHEMA = "fireviewer.ground-surface-atlas-library.v3"
GROUND_RUNTIME_SCHEMA = "fireviewer.ground-surface-runtime-contract.v3"
GROUND_CONTEXT_CATALOG_SCHEMA = "fireviewer.ground-context-catalog.v1"
GROUND_MICRO_SOURCE_COUNT = 21
GROUND_PROFILE_COUNT = 72
GROUND_RUNTIME_TEXTURE_COUNT = 4
GROUND_APPLICATION_MODES = {
    "ground_blend",
    "directional_area",
    "linear_overlay",
    "watercourse_overlay",
    "slope_cliff_overlay",
}


@dataclass(frozen=True)
class IncidentTerrainCase:
    fire_id: str
    canonical_name: str
    incident_geometry_count: int
    source_bounds_epsg2154_m: tuple[float, float, float, float]
    square_bounds_epsg2154_m: tuple[int, int, int, int]

    @property
    def side_m(self) -> int:
        return self.square_bounds_epsg2154_m[2] - self.square_bounds_epsg2154_m[0]

    @property
    def origin_l93_m(self) -> tuple[float, float, float]:
        west, south, east, north = self.square_bounds_epsg2154_m
        return ((west + east) / 2, (south + north) / 2, 0.0)


CASES: tuple[IncidentTerrainCase, ...] = (
    IncidentTerrainCase(
        "FR-30-00001",
        "Incendie de Lédenon–Bezouce–Cabrières",
        1,
        (818_722.92, 6_310_771.91, 821_201.95, 6_314_958.74),
        (813_000, 6_305_500, 827_500, 6_320_000),
    ),
    IncidentTerrainCase(
        "FR-34-00001",
        "Incendie d’Oupia–Pouzols-Minervois",
        1,
        (680_680.04, 6_241_502.24, 686_069.80, 6_245_684.44),
        (675_500, 6_236_000, 691_500, 6_252_000),
    ),
    IncidentTerrainCase(
        "FR-83-00001",
        "Incendie de Taradeau–Les Arcs",
        2,
        (969_506.68, 6_267_229.87, 981_098.83, 6_279_369.24),
        (964_500, 6_262_000, 987_000, 6_284_500),
    ),
    IncidentTerrainCase(
        "FR-26-00001",
        "Incendie de Die - massif de Justin",
        21,
        (881_159.04, 6_400_806.10, 888_857.25, 6_410_500.00),
        (875_000, 6_395_500, 895_000, 6_415_500),
    ),
    IncidentTerrainCase(
        "FR-66-00001",
        "Incendie de Trévillach",
        5,
        (655_575.52, 6_162_436.94, 669_353.56, 6_179_241.65),
        (649_000, 6_157_000, 676_500, 6_184_500),
    ),
    IncidentTerrainCase(
        "FR-77-00001",
        "Forêt de Fontainebleau",
        4,
        (663_278.87, 6_806_041.42, 675_961.10, 6_815_276.96),
        (658_000, 6_799_500, 681_000, 6_822_500),
    ),
)


def validate_case(case: IncidentTerrainCase) -> None:
    west, south, east, north = case.square_bounds_epsg2154_m
    source_west, source_south, source_east, source_north = case.source_bounds_epsg2154_m
    if not case.fire_id.startswith("FR-") or not case.canonical_name.strip():
        raise ValueError("Incident identity is invalid")
    if case.incident_geometry_count <= 0:
        raise ValueError(f"{case.fire_id} has no referenced incident geometry")
    if east <= west or north <= south or east - west != north - south:
        raise ValueError(f"{case.fire_id} production bounds must form a square")
    if case.side_m % TILE_SIZE_M:
        raise ValueError(f"{case.fire_id} square is not aligned to the 500 m grid")
    if any(value % TILE_SIZE_M for value in (west, south, east, north)):
        raise ValueError(f"{case.fire_id} bounds are not on the 500 m grid")
    if not (
        west <= source_west - SAFETY_MARGIN_M
        and south <= source_south - SAFETY_MARGIN_M
        and east >= source_east + SAFETY_MARGIN_M
        and north >= source_north + SAFETY_MARGIN_M
    ):
        raise ValueError(f"{case.fire_id} square does not preserve its safety margin")


def square_aoi(case: IncidentTerrainCase) -> dict[str, Any]:
    validate_case(case)
    west, south, east, north = case.square_bounds_epsg2154_m
    return {
        "type": "FeatureCollection",
        "name": f"{case.fire_id}-terrain-square",
        "crs": {"type": "name", "properties": {"name": SOURCE_CRS}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "fire_id": case.fire_id,
                    "canonical_name": case.canonical_name,
                    "role": "conservative_square_terrain_aoi",
                    "safety_margin_m": SAFETY_MARGIN_M,
                    "legacy_terrain_reused": False,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [west, south],
                            [east, south],
                            [east, north],
                            [west, north],
                            [west, south],
                        ]
                    ],
                },
            }
        ],
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_ground_surface_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != GROUND_SURFACE_SCHEMA:
        raise ValueError("Unsupported ground surface atlas catalog schema")
    if payload.get("micro_source_count") != GROUND_MICRO_SOURCE_COUNT:
        raise ValueError(
            f"Ground surface atlas must pack {GROUND_MICRO_SOURCE_COUNT} micro sources"
        )
    if payload.get("profile_count") != GROUND_PROFILE_COUNT:
        raise ValueError(
            f"Ground surface catalog must contain {GROUND_PROFILE_COUNT} profiles"
        )
    if payload.get("runtime_texture_count") != GROUND_RUNTIME_TEXTURE_COUNT:
        raise ValueError(
            "Ground surface runtime must import exactly four atlas textures"
        )
    if payload.get("micro_source_runtime_import") != "forbidden":
        raise ValueError("Direct ImageGen micro source imports must be forbidden")
    if payload.get("orthophoto_dependency") != "forbidden":
        raise ValueError("Ground surface atlas must forbid orthophotos")
    unsigned = dict(payload)
    expected_hash = unsigned.pop("catalog_sha256", None)
    if expected_hash != _canonical_sha256(unsigned):
        raise ValueError("Ground surface atlas catalog SHA-256 is invalid")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("Ground surface catalog has no profile list")
    identifiers = [profile.get("id") for profile in profiles]
    if len(set(identifiers)) != GROUND_PROFILE_COUNT:
        raise ValueError("Ground surface profile identifiers must be unique")
    if payload.get("qa", {}).get("automated") != "passed":
        raise ValueError("Ground surface atlas must pass automated QA")
    application_modes = {profile.get("application_mode") for profile in profiles}
    if application_modes != GROUND_APPLICATION_MODES:
        raise ValueError("Ground surface catalog application modes are incomplete")
    return payload


def load_ground_runtime_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != GROUND_RUNTIME_SCHEMA:
        raise ValueError("Unsupported ground surface runtime contract")
    if payload.get("orthophoto_dependency") != "forbidden":
        raise ValueError("Ground surface runtime must forbid orthophotos")
    runtime = payload.get("runtime_atlas", {})
    if runtime.get("runtime_texture_count") != GROUND_RUNTIME_TEXTURE_COUNT:
        raise ValueError("Ground surface runtime must contain four atlases")
    scale = payload.get("scale_contract", {})
    if scale.get("direct_source_image_import") != "forbidden":
        raise ValueError("Ground surface runtime must pack offline sources")
    composition = payload.get("composition", {})
    ground_blend = composition.get("ground_blend", {})
    if ground_blend.get("grid_cell_size_m") != 5.0:
        raise ValueError("Ground blend composition cells must measure 5 m")
    if ground_blend.get("grid_size_px_per_500m_tile") != [100, 100]:
        raise ValueError("Ground blend maps must be 100x100 pixels per 500 m tile")
    if ground_blend.get("maximum_profiles_per_cell") != 4:
        raise ValueError("Ground blend must contain four or fewer profiles per cell")
    if ground_blend.get("profile_id_map") != "ground-profile-ids.png":
        raise ValueError("Ground blend profile id map name is invalid")
    if ground_blend.get("profile_weight_map") != "ground-profile-weights.png":
        raise ValueError("Ground blend profile weight map name is invalid")
    if ground_blend.get("profile_weight_encoding") != "rgba8_sum_exactly_255":
        raise ValueError("Ground blend weights must sum exactly to 255")
    if composition.get("rail_geometry", {}).get("steel_rails") != (
        "required_separate_future_3d_geometry"
    ):
        raise ValueError("Railway steel rails must remain separate 3D geometry")
    if payload.get("determinism", {}).get("silent_fallback") != "forbidden":
        raise ValueError("Ground surface runtime must fail closed")
    families = payload.get("profile_families", [])
    profile_count = sum(len(family.get("variant_ids", [])) for family in families)
    if profile_count != GROUND_PROFILE_COUNT:
        raise ValueError("Ground surface runtime profile families are incomplete")
    application_modes = {family.get("application_mode") for family in families}
    if application_modes != GROUND_APPLICATION_MODES:
        raise ValueError("Ground surface runtime application modes are incomplete")
    return payload


def load_ground_context_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != GROUND_CONTEXT_CATALOG_SCHEMA:
        raise ValueError("Unsupported ground context catalog")
    if payload.get("status") != "validated_six_case_context":
        raise ValueError("Ground context catalog is not validated")
    if payload.get("crs") != SOURCE_CRS:
        raise ValueError("Ground context catalog must use EPSG:2154")
    if payload.get("orthophoto_dependency") != "forbidden":
        raise ValueError("Ground context catalog must forbid orthophotos")
    if payload.get("profile_binding_count") != GROUND_PROFILE_COUNT:
        raise ValueError("Ground context catalog must bind all 72 profiles")
    unsigned = dict(payload)
    expected_hash = unsigned.pop("catalog_sha256", None)
    if expected_hash != _canonical_sha256(unsigned):
        raise ValueError("Ground context catalog SHA-256 is invalid")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("Ground context catalog must contain six cases")
    if {case.get("fire_id") for case in cases} != {case.fire_id for case in CASES}:
        raise ValueError("Ground context cases do not match terrain cases")
    root = path.parent
    for case in cases:
        package = root / case["package_path"]
        manifest_path = root / case["manifest_path"]
        if not package.is_file() or sha256_file(package) != case["package_sha256"]:
            raise ValueError(f"Ground context package mismatch: {case['fire_id']}")
        if (
            not manifest_path.is_file()
            or sha256_file(manifest_path) != case["manifest_sha256"]
        ):
            raise ValueError(f"Ground context manifest mismatch: {case['fire_id']}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fire_id") != case["fire_id"]:
            raise ValueError(
                f"Ground context manifest incident mismatch: {case['fire_id']}"
            )
        if manifest.get("package", {}).get("sha256") != case["package_sha256"]:
            raise ValueError(
                f"Ground context manifest package mismatch: {case['fire_id']}"
            )
        if manifest.get("profile_binding_count") != GROUND_PROFILE_COUNT:
            raise ValueError(
                f"Ground context profile binding mismatch: {case['fire_id']}"
            )
        if manifest.get("source_cleanup", {}).get("status") != (
            "completed_after_package_validation"
        ):
            raise ValueError(
                f"Ground context source cleanup is incomplete: {case['fire_id']}"
            )
    return payload


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_case_plan(
    case: IncidentTerrainCase,
    output_root: Path,
    ground_surface_catalog_path: Path,
    ground_runtime_contract_path: Path,
    ground_context_catalog_path: Path,
    ground_context_catalog: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    validate_case(case)
    case_root = output_root / case.fire_id.lower()
    aoi_path = case_root / "aoi.square.epsg2154.geojson"
    _write_json(aoi_path, square_aoi(case), overwrite=overwrite)
    manifest = build_plan(
        aoi_path,
        case_root,
        origin=case.origin_l93_m,
        config=Global05mConfig(
            output_tile_size_m=TILE_SIZE_M,
            elevation_resolution_m=0.5,
            terrain_sample_spacing_m=1.0,
        ),
        terrain_only=True,
    )
    manifest["incident"] = {
        "fire_id": case.fire_id,
        "canonical_name": case.canonical_name,
        "public_index_url": PUBLIC_INDEX_URL,
        "incident_geometry_count": case.incident_geometry_count,
        "source_bounds_epsg2154_m": list(case.source_bounds_epsg2154_m),
        "square_bounds_epsg2154_m": list(case.square_bounds_epsg2154_m),
        "safety_margin_m": SAFETY_MARGIN_M,
        "coverage_rule": "all_referenced_incident_geometries_plus_5km_margin",
    }
    manifest["legacy_reuse"] = {
        "terrain": "forbidden",
        "ground_texture": "forbidden",
        "assets": "forbidden",
        "placements": "forbidden",
        "die_retained_logic_only": ["counts", "composition_structure"],
    }
    ground_catalog = load_ground_surface_catalog(ground_surface_catalog_path)
    runtime_contract = load_ground_runtime_contract(ground_runtime_contract_path)
    context_case = next(
        entry
        for entry in ground_context_catalog["cases"]
        if entry["fire_id"] == case.fire_id
    )
    context_package_path = (
        ground_context_catalog_path.parent / context_case["package_path"]
    )
    context_manifest_path = (
        ground_context_catalog_path.parent / context_case["manifest_path"]
    )
    candidate_profile_ids_by_mode = {
        mode: sorted(
            profile["id"]
            for profile in ground_catalog["profiles"]
            if profile["application_mode"] == mode
            and case.fire_id in profile.get("compatible_incidents", [])
        )
        for mode in sorted(GROUND_APPLICATION_MODES)
    }
    if len(candidate_profile_ids_by_mode["ground_blend"]) < 8:
        raise ValueError(f"{case.fire_id} has insufficient compatible ground blends")
    if any(not identifiers for identifiers in candidate_profile_ids_by_mode.values()):
        raise ValueError(f"{case.fire_id} has an empty ground composition stream")
    catalog_relative = Path(
        os.path.relpath(ground_surface_catalog_path.resolve(), case_root.resolve())
    ).as_posix()
    runtime_relative = Path(
        os.path.relpath(ground_runtime_contract_path.resolve(), case_root.resolve())
    ).as_posix()
    manifest["ground_surface_gate"] = {
        "status": "blocked_pending_visual_acceptance",
        "catalog_path": catalog_relative,
        "catalog_sha256": ground_catalog["catalog_sha256"],
        "catalog_status": ground_catalog["status"],
        "runtime_contract_path": runtime_relative,
        "runtime_contract_sha256": sha256_file(ground_runtime_contract_path),
        "runtime_contract_status": runtime_contract["status"],
        "micro_source_count": GROUND_MICRO_SOURCE_COUNT,
        "runtime_texture_count": GROUND_RUNTIME_TEXTURE_COUNT,
        "candidate_profile_ids_by_application_mode": candidate_profile_ids_by_mode,
        "maximum_active_materials_per_tile": 4,
        "ground_context": {
            "catalog_path": Path(
                os.path.relpath(
                    ground_context_catalog_path.resolve(), case_root.resolve()
                )
            ).as_posix(),
            "catalog_sha256": ground_context_catalog["catalog_sha256"],
            "package_path": Path(
                os.path.relpath(context_package_path.resolve(), case_root.resolve())
            ).as_posix(),
            "package_sha256": context_case["package_sha256"],
            "manifest_path": Path(
                os.path.relpath(context_manifest_path.resolve(), case_root.resolve())
            ).as_posix(),
            "manifest_sha256": context_case["manifest_sha256"],
            "feature_count": context_case["feature_count"],
            "profile_binding_count": GROUND_PROFILE_COUNT,
            "status": "validated",
        },
        "structured_overlay_sources": [
            "approved_landcover_class",
            "approved_geological_parent_material",
            "approved_land_parcels",
            "approved_agricultural_parcels",
            "approved_transport_network",
            "approved_hydrography",
        ],
        "railway_steel_rails": "required_separate_future_3d_geometry",
        "silent_fallback": "forbidden",
    }
    manifest["status"] = "blocked_pending_ground_surface_visual_acceptance"
    manifest_path = case_root / "production-manifest.json"
    _write_json(manifest_path, manifest, overwrite=overwrite)
    return {
        "fire_id": case.fire_id,
        "canonical_name": case.canonical_name,
        "root": case_root.name,
        "aoi": aoi_path.name,
        "manifest": manifest_path.name,
        "plan_id": manifest["plan_id"],
        "square_bounds_epsg2154_m": list(case.square_bounds_epsg2154_m),
        "square_side_m": case.side_m,
        "source_tile_count": manifest["summary"]["source_tile_count"],
        "output_tile_count": manifest["summary"]["output_tile_count"],
        "candidate_ground_profile_count": sum(
            len(identifiers) for identifiers in candidate_profile_ids_by_mode.values()
        ),
        "candidate_ground_profile_count_by_application_mode": {
            mode: len(identifiers)
            for mode, identifiers in candidate_profile_ids_by_mode.items()
        },
    }


def write_catalog(
    output_root: Path,
    ground_surface_catalog_path: Path,
    ground_context_catalog_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if len(CASES) != 6 or len({case.fire_id for case in CASES}) != 6:
        raise ValueError("The terrain production catalog must contain six unique cases")
    runtime_source = Path(__file__).with_name("ground_surface_runtime_contract.v3.json")
    runtime_contract = load_ground_runtime_contract(runtime_source)
    ground_context_catalog = load_ground_context_catalog(ground_context_catalog_path)
    runtime_output = output_root / "ground-surface-runtime-contract.json"
    _write_json(runtime_output, runtime_contract, overwrite=overwrite)
    plans = [
        write_case_plan(
            case,
            output_root,
            ground_surface_catalog_path,
            runtime_output,
            ground_context_catalog_path,
            ground_context_catalog,
            overwrite=overwrite,
        )
        for case in CASES
    ]
    catalog = {
        "schema": CATALOG_SCHEMA,
        "status": "blocked_pending_ground_surface_visual_acceptance",
        "case_count": len(plans),
        "crs": SOURCE_CRS,
        "tile_size_m": TILE_SIZE_M,
        "native_elevation_resolution_m": 0.5,
        "ground_2d_asset": {
            "format": "rgba_png_ground_blend_plus_structured_overlay_masks",
            "pixel_size_per_500m_tile": [256, 256],
            "derivation": "approved_classified_context_aligned_to_mnt_mns",
            "orthophoto_dependency": "forbidden",
            "application_modes": sorted(GROUND_APPLICATION_MODES),
            "railway_steel_rails": "separate_future_3d_geometry",
        },
        "safety_margin_m": SAFETY_MARGIN_M,
        "production_order": [
            "terrain_3d_and_ground_2d",
            "buildings",
            "roads_and_small_specific_assets",
            "vegetation",
            "simulation",
        ],
        "legacy_die_policy": "counts_and_composition_logic_only",
        "ground_surface_library": {
            "catalog_path": Path(
                os.path.relpath(
                    ground_surface_catalog_path.resolve(), output_root.resolve()
                )
            ).as_posix(),
            "catalog_sha256": load_ground_surface_catalog(ground_surface_catalog_path)[
                "catalog_sha256"
            ],
            "micro_source_count": GROUND_MICRO_SOURCE_COUNT,
            "runtime_texture_count": GROUND_RUNTIME_TEXTURE_COUNT,
            "profile_count": GROUND_PROFILE_COUNT,
            "runtime_contract_path": runtime_output.name,
            "runtime_contract_sha256": sha256_file(runtime_output),
            "production_gate": "blocked",
        },
        "ground_context": {
            "catalog_path": Path(
                os.path.relpath(
                    ground_context_catalog_path.resolve(), output_root.resolve()
                )
            ).as_posix(),
            "catalog_sha256": ground_context_catalog["catalog_sha256"],
            "case_count": ground_context_catalog["case_count"],
            "profile_binding_count": ground_context_catalog["profile_binding_count"],
            "total_feature_count": ground_context_catalog["total_feature_count"],
            "production_gate": "passed",
        },
        "plans": plans,
    }
    catalog["catalog_sha256"] = _canonical_sha256(catalog)
    _write_json(output_root / "terrain-catalog.json", catalog, overwrite=overwrite)
    return catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--ground-surface-catalog",
        "--ground-material-catalog",
        dest="ground_surface_catalog",
        type=Path,
    )
    parser.add_argument("--ground-context-catalog", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-plans", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.write_plans:
        summary = [
            {
                **asdict(case),
                "side_m": case.side_m,
                "origin_l93_m": case.origin_l93_m,
            }
            for case in CASES
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.ground_surface_catalog is None:
        raise ValueError("--ground-surface-catalog is required with --write-plans")
    if args.ground_context_catalog is None:
        raise ValueError("--ground-context-catalog is required with --write-plans")
    catalog = write_catalog(
        args.output_root.resolve(),
        args.ground_surface_catalog.resolve(),
        args.ground_context_catalog.resolve(),
        overwrite=args.overwrite,
    )
    print(json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
