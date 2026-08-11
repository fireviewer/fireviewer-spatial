from __future__ import annotations

from collections import namedtuple
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from blender import validate_adaptive_terrain_zone as zone_qa

from terrainctl import (
    BackendUnavailableError,
    ContractError,
    PHASE_RECEIPT_NAMES,
    SAFETY_MARGIN_BYTES,
    SOURCE_LOCK_SCHEMA,
    StoragePolicyError,
    TILE_BLENDER_VISUAL_SCHEMA,
    TILE_RECEIPT_SCHEMA,
    TerrainController,
    ZONE_ACCEPTANCE_SCHEMA,
    ZONE_PLAN_SCHEMA,
    ZONE_VISUAL_JOB_RELATIVE_PATH,
    ZONE_VISUAL_RECEIPT_NAME,
    ZONE_VISUAL_REVIEW_RELATIVE_PATH,
    ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH,
    ZONE_VISUAL_TECHNICAL_RELATIVE_PATH,
    build_source_lock,
    build_final_source_identity,
    build_zone_plan,
    canonical_sha256,
    load_zone_spec,
    main,
    sha256_file,
)


DiskUsage = namedtuple("DiskUsage", "total used free")
TERRAINCTL_TEST_ROOT = Path(
    "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest/terrainctl"
)


@pytest.fixture
def d_workspace() -> Path:
    root = TERRAINCTL_TEST_ROOT / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.resolve().is_relative_to(TERRAINCTL_TEST_ROOT.resolve()):
            shutil.rmtree(root, ignore_errors=True)


def _hash(label: str) -> str:
    return canonical_sha256({"fixture": label})


def _write_zone(
    root: Path,
    *,
    zone_id: str = "FR-TEST-00001",
    revision: str = "r1",
    work_root: Path | None = None,
    export_root: Path | None = None,
    estimated_peak_bytes: int = 1024,
    source_hash_suffix: str = "a",
    regression_tile_id: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    work_root = work_root or root / "fireviewer-work"
    export_root = export_root or root / "fireviewer-exports"
    dependency_root = root / "dependencies"
    dependency_root.mkdir(parents=True, exist_ok=True)
    dependency_payloads = {
        "algorithm": b"fixture terrain algorithm v1\n",
        "clean_pbr_texture_library": json.dumps(
            {"schema": "fireviewer.clean-pbr-texture-library.v1"}, sort_keys=True
        ).encode(),
        "ground_texture_contract": json.dumps(
            {"schema": "fireviewer.ground-surface-texture-contract.v4"}, sort_keys=True
        ).encode(),
        "surface_correspondence_contract": json.dumps(
            {"schema": "fireviewer.orthophoto-surface-correspondence-contract.v1"},
            sort_keys=True,
        ).encode(),
        "surface_correspondence_model": json.dumps(
            {"schema": "fireviewer.orthophoto-surface-model.v1"}, sort_keys=True
        ).encode(),
        "surface_features": json.dumps(
            {"schema": "fireviewer.surface-features.v1", "features": []},
            sort_keys=True,
        ).encode(),
        "terrain_quadtree_contract": json.dumps(
            {"schema": "fireviewer.terrain-quadtree-contract.v1"}, sort_keys=True
        ).encode(),
        "toolchain": json.dumps(
            {"schema": "fireviewer.terrain-toolchain-lock.v1"}, sort_keys=True
        ).encode(),
    }
    dependencies = {}
    dependency_artifacts = {}
    for name, content in dependency_payloads.items():
        artifact_path = dependency_root / f"{name}.json"
        artifact_path.write_bytes(content)
        digest = sha256_file(artifact_path)
        dependencies[name] = digest
        dependency_artifacts[name] = {"path": str(artifact_path), "sha256": digest}
    source_requests = []
    for y_index in range(3):
        for x_index in range(3):
            west = 700_000 + x_index * 500
            south = 6_300_000 + y_index * 500
            core_bounds = [west, south, west + 500, south + 500]
            source_id = f"tile-{x_index}-{y_index}"
            for product in ("mnt", "mns"):
                source_requests.append(
                    {
                        "id": source_id,
                        "product": product,
                        "request": {
                            "service_url": f"https://example.invalid/{product}",
                            "layer": f"RGEALTI-{product.upper()}-2M",
                            "core_bounds_l93_m": core_bounds,
                            "resolution_m": 2,
                        },
                        "expected_sha256": _hash(
                            f"{product}-{source_id}-{source_hash_suffix}"
                        ),
                        "expected_byte_count": 4096,
                        "source_revision_id": "fixture-r1",
                        "license": "fixture-public-data",
                        "pilot_score": float(x_index + y_index),
                    }
                )
            source_requests.append(
                {
                    "id": source_id,
                    "product": "orthophoto",
                    "request": {
                        "service_kind": "wms",
                        "service_url": "https://example.invalid/orthophoto",
                        "layer": "ORTHO-RGB-1M",
                        "core_bounds_l93_m": core_bounds,
                        "resolution_m": 1,
                        "halo_m": 10,
                        "image_format": "image/png",
                        "maximum_download_bytes": 1_048_576,
                    },
                    "expected_sha256": _hash(
                        f"orthophoto-{source_id}-{source_hash_suffix}"
                    ),
                    "expected_byte_count": 8192,
                    "source_revision_id": "fixture-ortho-r1",
                    "license": "fixture-public-data",
                    "pilot_score": float(x_index + y_index),
                }
            )
    payload = {
        "schema": "fireviewer.zone-spec.v1",
        "zone_id": zone_id,
        "revision": revision,
        "crs": "EPSG:2154",
        "bounds_l93_m": [700_000, 6_300_000, 701_500, 6_301_500],
        "tile_size_m": 500,
        "halo_m": 10,
        "source_resolution_m": 2,
        "dependencies": dependencies,
        "dependency_artifacts": dependency_artifacts,
        "source_requests": source_requests,
        "workspace": {
            "work_root": str(work_root),
            "export_root": str(export_root),
        },
        "storage": {"estimated_peak_bytes": estimated_peak_bytes},
    }
    if regression_tile_id is not None:
        payload["pilot"] = {"regression_tile_id": regression_tile_id}
    path = root / f"{zone_id}-{revision}.zone-spec.v1.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _locked_sources(context) -> list[dict[str, object]]:
    return [
        {
            "id": request["id"],
            "product": request["product"],
            "request_sha256": canonical_sha256(request["request"]),
            "source_revision_id": request.get("source_revision_id"),
            "identity_status": "expected_identity_locked",
            "sha256": request["expected_sha256"],
            "byte_count": request["expected_byte_count"],
            "license": request["license"],
        }
        for request in context.plan["source_requests"]
    ]


def _final_identity_inputs(
    context,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    observed_sources = [
        {
            "id": request["id"],
            "product": request["product"],
            "request_sha256": canonical_sha256(request["request"]),
            "source_revision_id": request.get("source_revision_id"),
            "sha256": request["expected_sha256"],
            "byte_count": request["expected_byte_count"],
            "license": request["license"],
        }
        for request in context.plan["source_requests"]
    ]
    zone_root = context.export_root / context.zone.zone_id / context.zone.revision
    tile_receipts = [
        {
            "tile_id": tile["id"],
            "tile_done_sha256": sha256_file(
                zone_root / "tiles" / tile["id"] / "tile.done.v3.json"
            ),
        }
        for tile in context.plan["tiles"]
    ]
    return observed_sources, tile_receipts


def _final_identity(context) -> dict[str, object]:
    observed_sources, tile_receipts = _final_identity_inputs(context)
    return build_final_source_identity(
        context.plan,
        recipe_build_id=context.source_lock["recipe_build_id"],
        observed_sources=observed_sources,
        tile_receipts=tile_receipts,
    )


def _artifact(label: str) -> dict[str, object]:
    return {
        "path": f"artifacts/{label}.bin",
        "byte_count": len(label),
        "sha256": _hash(label),
    }


def _stitch_catalog(tile_id: str) -> dict[str, object]:
    return {
        "encoding": "fvtq-base-remove-add.v1",
        "edge_order": ["west", "east", "south", "north"],
        "edge_mask_bits": {"west": 1, "east": 2, "south": 4, "north": 8},
        "available_masks": list(range(16)),
        "lods": {
            f"lod{lod}": [
                {
                    "mask": mask,
                    "triangle_count": max(1, 128 >> lod),
                    "triangle_indices_sha256": _hash(
                        f"{tile_id}-lod{lod}-stitch-{mask}"
                    ),
                    "maximum_error_mm": (100, 500, 1_000)[lod],
                    "effective_edge_signatures": [
                        _hash(f"{tile_id}-lod{lod}-stitch-{mask}-{edge}")
                        for edge in ("west", "east", "south", "north")
                    ],
                }
                for mask in range(16)
            ]
            for lod in range(3)
        },
    }


def _dependency_proofs_from_plan(plan) -> dict[str, dict[str, object]]:
    return {
        name: {
            "file_name": f"{name}.json",
            "sha256": digest,
            "schema": (
                "fireviewer.clean-pbr-texture-library.v1"
                if name == "clean_pbr_texture_library"
                else None
            ),
            "status": (
                "accepted_clean_pbr_library"
                if name == "clean_pbr_texture_library"
                else None
            ),
        }
        for name, digest in plan["dependencies"].items()
    }


def _tile_receipt(context, tile_id: str) -> dict[str, object]:
    assert context.source_lock is not None
    tile = next(item for item in context.plan["tiles"] if item["id"] == tile_id)
    tile_root = (
        context.export_root
        / context.zone.zone_id
        / context.zone.revision
        / "tiles"
        / tile_id
    )
    tile_root.mkdir(parents=True, exist_ok=True)
    filenames = {
        "terrain_lod0": "terrain-lod0.fvtq",
        "terrain_lod1": "terrain-lod1.fvtq",
        "terrain_lod2": "terrain-lod2.fvtq",
        "hag_max_2m": "hag-max-2m.tif",
        "ground_profile_ids": "ground-profile-ids.png",
        "ground_profile_weights": "ground-profile-weights.png",
        "ground_confidence": "ground-confidence.png",
        "ground_orientation": "ground-orientation.png",
        "tile_package": "tile-package.v3.json",
    }
    outputs = {}
    for name, filename in filenames.items():
        content = f"{tile_id}:{name}:fixture\n".encode()
        path = tile_root / filename
        path.write_bytes(content)
        outputs[name] = {
            "path": filename,
            "byte_count": len(content),
            "sha256": sha256_file(path),
        }
    receipt = {
        "schema": TILE_RECEIPT_SCHEMA,
        "tile_id": tile_id,
        "recipe_id": context.plan["recipe_id"],
        "recipe_build_id": context.source_lock["recipe_build_id"],
        "normal_halo_sha256": _hash(f"normal-halo-{tile_id}"),
        "stitch_variants": _stitch_catalog(tile_id),
        "inputs": {
            "source_lock": _artifact(f"input-{tile_id}"),
            "surface_correspondence": {
                "path": "surface-correspondence.json",
                "byte_count": len(tile_id),
                "sha256": _hash(f"surface-correspondence-{tile_id}"),
            },
        },
        "ground_material": {
            "schema": "fireviewer.ground-material-contract.v2",
            "zone_path": "shared/ground-material/ground-material-contract.v2.json",
            "contract_sha256": _hash("ground-material-contract"),
            "source_library_schema": "fireviewer.clean-pbr-texture-library.v1",
            "source_library_manifest_sha256": _hash("ground-material-source-library"),
            "source_library_identity_sha256": _hash("ground-material-library-identity"),
            "source_library_content_sha256": _hash("ground-material-library-content"),
            "texture_contract_sha256": _hash("ground-texture-contract"),
            "runtime_shader": {
                "schema": "fireviewer.ground-runtime-shader-binding.v1",
                "status": "pending_dedicated_mdl_validation",
                "implementation": None,
                "source_artifact": None,
                "production_textured_runtime_qualified": False,
                "preview_surface_policy": "diagnostic_untextured_only",
                "required_capabilities": [
                    "four_profile_id_indirections",
                    "rgba8_weighted_pbr_blend",
                    "epsg2154_world_projection",
                    "undirected_orientation_0_to_pi",
                    "world_xy_and_world_triplanar",
                ],
            },
            "runtime_atlas_sha256": {
                role: _hash(f"ground-material-{role}")
                for role in ("basecolor", "normal", "height", "orm")
            },
            "material_layer_sha256": _hash("ground-material-layer"),
            "visual_acceptance": "accepted_human_visual",
        },
        "surface_mapping": {
            "schema": "fireviewer.ground-surface-mapping.v3",
            "crs": "EPSG:2154",
            "bounds_l93_m": tile["bounds_l93_m"],
            "grid_size_px": [500, 500],
            "cell_size_m": 1,
            "row_order": "north_to_south",
            "profile_count": 72,
            "profile_ids": {
                "file": "ground-profile-ids.png",
                "mode": "RGBA8",
                "encoding": "four_zero_based_stable_profile_indices",
            },
            "profile_weights": {
                "file": "ground-profile-weights.png",
                "mode": "RGBA8",
                "encoding": "four_profile_weights_sum_exactly_255_per_pixel",
            },
            "confidence": {
                "file": "ground-confidence.png",
                "mode": "L8",
                "encoding": "best_vs_next_semantic_class_margin_0_to_255",
            },
            "orientation": {
                "file": "ground-orientation.png",
                "mode": "L8",
                "encoding": "undirected_angle_0_to_pi_mapped_to_uint8",
            },
            "world_projection": "EPSG:2154 metric XY with no tile-local phase reset",
            "variant_selection": "baked_profile_id",
            "runtime_procedural_material": "forbidden",
            "runtime_orthophoto": "forbidden",
            "surface_overlays": "not_packaged",
        },
        "outputs": outputs,
    }
    (tile_root / "tile.done.v3.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _metrics(phase: str, tile_count: int, *, seam_count: int = 0) -> dict[str, object]:
    result: dict[str, object] = {
        "tile_count": tile_count,
        "source_pair_count": tile_count,
        "source_bytes": 8192,
        "package_bytes": tile_count * 1024,
        "lod0_triangles": tile_count * 128,
        "lod1_triangles": tile_count * 32,
        "lod2_triangles": tile_count * 8,
        "lod0_stitch_variant_count": tile_count * 16,
        "lod1_stitch_variant_count": tile_count * 16,
        "lod2_stitch_variant_count": tile_count * 16,
        "lod0_maximum_final_error_mm": 100,
        "lod1_maximum_final_error_mm": 500,
        "lod2_maximum_final_error_mm": 1_000,
        "lod0_maximum_stitch_triangles": 256,
        "lod1_maximum_stitch_triangles": 128,
        "lod2_maximum_stitch_triangles": 32,
        "bitwise_rebuild_count": tile_count,
        "elapsed_seconds": 0.01,
        "peak_python_bytes": 4096,
    }
    if phase == "pilot":
        result.update(
            {
                "projected_zone_package_bytes": 9 * 1024,
                "projected_peak_disk_bytes": 8192,
                "internal_seam_count": max(0, tile_count - 1),
                "blender_report_count": tile_count,
                "aov_invalid_pixel_count": 0,
            }
        )
    elif phase == "produce":
        result.update(
            {
                "built_tile_count": tile_count,
                "reused_tile_count": 0,
                "raw_source_count": 0,
                "part_file_count": 0,
            }
        )
    elif phase == "qa":
        result.update(
            {
                "seam_count": seam_count,
                "maximum_height_gap_mm": 0,
                "normal_mismatch_count": 0,
                "stitch_lod_pair_count": seam_count * 7,
                "stitch_signature_mismatch_count": 0,
                "nodata_pixel_count": 0,
                "fallback_material_count": 0,
                "composition_failure_count": 0,
                "blender_report_count": 1,
                "aov_invalid_pixel_count": 0,
                "deterministic_tile_count": tile_count,
            }
        )
    return result


def _write_visual_reports(context, tile_ids, *, label: str) -> list[dict[str, object]]:
    zone_root = context.export_root / context.zone.zone_id / context.zone.revision
    proof_root = zone_root / "qa" / label
    records = []
    for tile_id in tile_ids:
        report_path = proof_root / f"{tile_id}.accepted_blender_textured_visual.json"
        render_path = proof_root / f"{tile_id}.terrain-lod-aov.exr"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_path = proof_root / f"{tile_id}.terrain-coverage-aov.exr"
        oblique_lod_path = proof_root / f"{render_path.stem}-oblique.exr"
        oblique_coverage_path = proof_root / f"{coverage_path.stem}-oblique.exr"
        beauty_topdown_path = proof_root / f"{tile_id}.textured-topdown.png"
        beauty_oblique_path = proof_root / f"{tile_id}.textured-oblique.png"
        for path, content in (
            (render_path, f"fixture-lod-topdown:{tile_id}\n"),
            (coverage_path, f"fixture-coverage-topdown:{tile_id}\n"),
            (oblique_lod_path, f"fixture-lod-oblique:{tile_id}\n"),
            (oblique_coverage_path, f"fixture-coverage-oblique:{tile_id}\n"),
            (beauty_topdown_path, f"fixture-beauty-topdown:{tile_id}\n"),
            (beauty_oblique_path, f"fixture-beauty-oblique:{tile_id}\n"),
        ):
            path.write_bytes(content.encode())

        def internal_artifact(path: Path, **fields: object) -> dict[str, object]:
            return {"path": path.name, "sha256": sha256_file(path), **fields}

        report = {
            "schema": TILE_BLENDER_VISUAL_SCHEMA,
            "status": "accepted_blender_textured_visual",
            "geometry_lod_status": "accepted_blender_geometry_lod",
            "production_visual_gate_passed": True,
            "human_visual_acceptance": "accepted_human_visual",
            "source_library_status": "accepted_clean_pbr_library",
            "tile_id": tile_id,
            "selected_lod": 0,
            "render_resolution": [512, 512],
            "beauty": {
                "topdown": internal_artifact(
                    beauty_topdown_path,
                    terrain_pixel_count=896,
                    frame_coverage_ratio=0.875,
                    distinct_rgb8_count=32,
                ),
                "oblique": internal_artifact(
                    beauty_oblique_path,
                    terrain_pixel_count=256,
                    frame_coverage_ratio=0.25,
                    distinct_rgb8_count=32,
                ),
            },
            "aov": {
                "validated_primary_views": ["topdown", "oblique"],
                "lod": internal_artifact(
                    render_path,
                    name="fireviewer:terrain_lod",
                    expected_value=0,
                    terrain_pixel_count=896,
                    invalid_pixel_count=0,
                    maximum_absolute_error=0.0,
                ),
                "coverage": internal_artifact(
                    coverage_path,
                    name="fireviewer:terrain_coverage",
                    expected_value=1,
                    invalid_pixel_count=0,
                ),
                "oblique_lod": internal_artifact(
                    oblique_lod_path,
                    name="fireviewer:terrain_lod",
                    expected_value=0,
                    terrain_pixel_count=256,
                    invalid_pixel_count=0,
                    maximum_absolute_error=0.0,
                ),
                "oblique_coverage": internal_artifact(
                    oblique_coverage_path,
                    name="fireviewer:terrain_coverage",
                    expected_value=1,
                    invalid_pixel_count=0,
                ),
            },
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        def artifact(path: Path) -> dict[str, object]:
            return {
                "path": path.relative_to(zone_root).as_posix(),
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        records.append(
            {
                "tile_id": tile_id,
                "report": artifact(report_path),
                "render": artifact(render_path),
                "terrain_pixel_count": 1152,
                "invalid_pixel_count": 0,
                "maximum_absolute_error": 0.0,
            }
        )
    return records


def _write_zone_visual_technical(
    context, final_identity: dict[str, object]
) -> dict[str, object]:
    proof_root = context.zone.work_root / "proofs" / "zone-visual"
    proof_root.mkdir(parents=True, exist_ok=True)
    job_path = context.zone.work_root.joinpath(
        *Path(ZONE_VISUAL_JOB_RELATIVE_PATH).parts
    )
    job = {
        "schema": zone_qa.JOB_SCHEMA,
        "zone_id": context.zone.zone_id,
        "revision": context.zone.revision,
        "recipe_id": context.plan["recipe_id"],
        "recipe_build_id": final_identity["recipe_build_id"],
        "build_id": final_identity["build_id"],
    }
    job_path.write_text(
        json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    job_sha256 = sha256_file(job_path)

    tiles = [tile["id"] for tile in context.plan["tiles"]]
    captures: list[dict[str, object]] = [
        {
            "capture_id": "overview-full-square",
            "category": "orthographic_full_square",
            "tile_ids": tiles,
        }
    ]
    for index in range(9):
        captures.append(
            {
                "capture_id": f"grid-r{index // 3 + 1}-c{index % 3 + 1}",
                "category": "orthographic_grid_3x3",
                "tile_ids": [tiles[index % len(tiles)]],
            }
        )
    for index, seam in enumerate(context.plan["seams"][:20], start=1):
        captures.append(
            {
                "capture_id": f"seam-{index:02d}-fixture",
                "category": "orthographic_worst_seam",
                "tile_ids": [seam["a"], seam["b"]],
            }
        )
    for label in ("highest", "lowest", "range"):
        captures.append(
            {
                "capture_id": f"oblique-relief-{label}",
                "category": "oblique_relief_extreme",
                "tile_ids": [tiles[0]],
            }
        )
    captures.extend(
        (
            {
                "capture_id": "oblique-networks",
                "category": "oblique_network_richness",
                "tile_ids": [tiles[0]],
            },
            {
                "capture_id": "oblique-dominant-surface-01",
                "category": "oblique_dominant_surface",
                "tile_ids": [tiles[0]],
            },
        )
    )
    for capture in captures:
        capture["projection"] = (
            "orthographic"
            if str(capture["category"]).startswith("orthographic_")
            else "perspective_oblique"
        )
    plan_basis = {
        "schema": zone_qa.CAPTURE_PLAN_SCHEMA,
        "zone_id": context.zone.zone_id,
        "revision": context.zone.revision,
        "recipe_id": context.plan["recipe_id"],
        "recipe_build_id": final_identity["recipe_build_id"],
        "build_id": final_identity["build_id"],
        "catalog_sha256": _hash("zone-catalog"),
        "qa_metrics_sha256": _hash("zone-qa-metrics"),
        "zone_bounds_l93_ngf_m": [700000, 6300000, 0, 701500, 6301500, 100],
        "selection_contract": {
            "orthographic_full_square": 1,
            "orthographic_grid_3x3": 9,
            "maximum_worst_seams": min(20, len(context.plan["seams"])),
            "available_seams": len(context.plan["seams"]),
            "selected_worst_seams": min(20, len(context.plan["seams"])),
            "worst_seam_sort": [
                "maximum_height_gap_mm desc",
                "normal_mismatch_count desc",
                "stitch_signature_mismatch_count desc",
                "composition_failure_count desc",
                "id asc",
            ],
            "oblique": [
                "highest",
                "lowest",
                "maximum_local_relief",
                "network_richest",
                "up_to_three_distinct_dominant_profiles",
            ],
        },
        "captures": captures,
    }
    plan = {**plan_basis, "plan_sha256": zone_qa._canonical_sha256(plan_basis)}
    plan_path = proof_root / "zone-visual-capture-plan.v1.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    capture_records = []
    for capture in captures:
        capture_root = proof_root / "captures" / str(capture["capture_id"])
        capture_root.mkdir(parents=True, exist_ok=True)
        for key, file_name in zone_qa.CAPTURE_ARTIFACT_NAMES.items():
            (capture_root / file_name).write_bytes(
                f"{capture['capture_id']}:{key}\n".encode()
            )
        receipt = {
            "schema": zone_qa.CAPTURE_RECEIPT_SCHEMA,
            "status": "rendered_technical",
            "capture_id": capture["capture_id"],
            "category": capture["category"],
            "capture_spec_sha256": zone_qa._canonical_sha256(capture),
            "capture_plan_sha256": plan["plan_sha256"],
            "job_sha256": job_sha256,
            "tile_ids": capture["tile_ids"],
            "artifacts": {
                key: zone_qa._artifact(
                    capture_root / file_name, relative_to=capture_root
                )
                for key, file_name in zone_qa.CAPTURE_ARTIFACT_NAMES.items()
            },
            "aov": {
                "terrain_lod_name": zone_qa.TERRAIN_AOV,
                "terrain_coverage_name": zone_qa.COVERAGE_AOV,
                "expected_lod": 0,
                "invalid_lod_pixel_count": 0,
                "invalid_coverage_pixel_count": 0,
                "terrain_pixel_count": 64,
            },
            "imported_lod0": {
                "lod": 0,
                "tile_ids": capture["tile_ids"],
                "mesh_count": len(capture["tile_ids"]),
                "forbidden_lod_mesh_count": 0,
            },
            "reference_material": {
                "model": "FireViewerGroundSurface_v1 Blender reference",
                "connected_channels": [
                    "basecolor",
                    "normal",
                    "height_bump",
                    "orm",
                ],
            },
        }
        capture_path = capture_root / "capture.done.v1.json"
        capture_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        capture_records.append(receipt)

    technical = {
        "schema": zone_qa.TECHNICAL_RECEIPT_SCHEMA,
        "status": zone_qa.TECHNICAL_STATUS,
        "production_visual_gate_passed": False,
        "human_visual_acceptance": "pending_exhaustive_review",
        "zone_id": context.zone.zone_id,
        "revision": context.zone.revision,
        "recipe_id": context.plan["recipe_id"],
        "recipe_build_id": final_identity["recipe_build_id"],
        "build_id": final_identity["build_id"],
        "job_sha256": job_sha256,
        "catalog_sha256": _hash("zone-catalog"),
        "qa_metrics_sha256": _hash("zone-qa-metrics"),
        "capture_plan": zone_qa._artifact(plan_path, relative_to=proof_root),
        "capture_plan_sha256": plan["plan_sha256"],
        "capture_count": len(capture_records),
        "capture_set_sha256": zone_qa._technical_capture_set_sha256(capture_records),
        "captures": [
            {
                "capture_id": record["capture_id"],
                "category": record["category"],
                "tile_ids": record["tile_ids"],
                "capture_spec_sha256": record["capture_spec_sha256"],
                "receipt": zone_qa._artifact(
                    proof_root
                    / "captures"
                    / str(record["capture_id"])
                    / "capture.done.v1.json",
                    relative_to=proof_root,
                ),
                "artifacts": record["artifacts"],
            }
            for record in capture_records
        ],
        "aov": {
            "terrain_lod": zone_qa.TERRAIN_AOV,
            "terrain_coverage": zone_qa.COVERAGE_AOV,
            "expected_lod": 0,
            "invalid_lod_pixel_count": 0,
            "invalid_coverage_pixel_count": 0,
            "terrain_pixel_count": len(capture_records) * 64,
        },
        "acceptance_contract": {
            "automatic_acceptance": False,
            "required_review_schema": zone_qa.HUMAN_REVIEW_SCHEMA,
            "review_scope": "every_capture_and_every_beauty_lod_coverage_hash",
        },
    }
    technical_path = context.zone.work_root.joinpath(
        *Path(ZONE_VISUAL_TECHNICAL_RELATIVE_PATH).parts
    )
    technical_path.write_text(
        json.dumps(technical, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    template_path = context.zone.work_root.joinpath(
        *Path(ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH).parts
    )
    zone_qa.create_human_review_template(technical_path, template_path)
    return {
        "visual_job": ZONE_VISUAL_JOB_RELATIVE_PATH,
        "visual_job_sha256": sha256_file(job_path),
        "visual_technical_receipt": ZONE_VISUAL_TECHNICAL_RELATIVE_PATH,
        "visual_technical_receipt_sha256": sha256_file(technical_path),
        "visual_review_template": ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH,
        "visual_review_template_sha256": sha256_file(template_path),
    }


def _record_explicit_zone_visual_acceptance(zone) -> Path:
    template_path = zone.work_root.joinpath(
        *Path(ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH).parts
    )
    review = json.loads(template_path.read_text(encoding="utf-8"))
    review["decision"] = "accepted"
    review["reviewer"] = {"kind": "human", "id": "fixture-reviewer"}
    review["decision_recorded_at_utc"] = "2026-08-09T12:00:00Z"
    for item in review["capture_reviews"]:
        item["decision"] = "accepted"
    review_path = zone.work_root.joinpath(*Path(ZONE_VISUAL_REVIEW_RELATIVE_PATH).parts)
    review_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    technical_path = zone.work_root.joinpath(
        *Path(ZONE_VISUAL_TECHNICAL_RELATIVE_PATH).parts
    )
    acceptance_path = zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
    zone_qa.accept_zone_visual_review(technical_path, review_path, acceptance_path)
    return acceptance_path


class FixtureBackends:
    def __init__(self) -> None:
        self.fail_pilot_once = False
        self.calls: list[tuple[str, str, int, int]] = []

    def preflight(self, context):
        self.calls.append(
            (
                context.phase,
                context.mode,
                context.download_workers,
                context.package_workers,
            )
        )
        assert Path(context.environment["TEMP"]).drive.upper() == "D:"
        return {
            "sources": _locked_sources(context),
            "dependency_proofs": _dependency_proofs_from_plan(context.plan),
            "estimated_peak_bytes": 2048,
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    def pilot(self, context):
        self.calls.append(
            (
                context.phase,
                context.mode,
                context.download_workers,
                context.package_workers,
            )
        )
        if self.fail_pilot_once:
            self.fail_pilot_once = False
            raise RuntimeError("simulated interrupted pilot")
        selected = [tile["id"] for tile in context.plan["tiles"]][: context.max_tiles]
        regression_tile_id = context.plan["pilot"].get("regression_tile_id")
        if regression_tile_id is not None and regression_tile_id not in selected:
            selected.append(regression_tile_id)
        visual_reports = _write_visual_reports(context, selected, label="pilot")
        return {
            "status": "passed",
            "selection": {
                "strategy": "contiguous_relief_and_semantic_maximization",
                "tile_ids": selected,
            },
            "tile_receipts": [_tile_receipt(context, tile_id) for tile_id in selected],
            "visual_reports": visual_reports,
            "metrics": _metrics("pilot", len(selected)),
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    def produce(self, context):
        tile_ids = [tile["id"] for tile in context.plan["tiles"]]
        return {
            "status": "passed",
            "tile_receipts": [_tile_receipt(context, tile_id) for tile_id in tile_ids],
            "metrics": _metrics("produce", len(tile_ids)),
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    def qa(self, context):
        final_identity = _final_identity(context)
        visual_proofs = _write_zone_visual_technical(context, final_identity)
        return {
            "status": "passed",
            **final_identity,
            "validated_tile_ids": [tile["id"] for tile in context.plan["tiles"]],
            "validated_seam_ids": [seam["id"] for seam in context.plan["seams"]],
            **visual_proofs,
            "metrics": _metrics(
                "qa", len(context.plan["tiles"]), seam_count=len(context.plan["seams"])
            ),
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    def accept(self, context):
        qa_path = context.run_root / "receipts" / PHASE_RECEIPT_NAMES["qa"]
        final_identity = _final_identity(context)
        visual_path = context.zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
        return {
            "status": "accepted",
            "recipe_build_id": final_identity["recipe_build_id"],
            "build_id": final_identity["build_id"],
            "source_merkle_root_sha256": final_identity["source_merkle_root_sha256"],
            "tile_count": len(context.plan["tiles"]),
            "seam_count": len(context.plan["seams"]),
            "qa_receipt_sha256": sha256_file(qa_path),
            "visual_receipt": ZONE_VISUAL_RECEIPT_NAME,
            "visual_receipt_sha256": sha256_file(visual_path),
            "runtime_shader": {
                "status": "pending_dedicated_mdl_validation",
            },
            "usd_runtime_gate": False,
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    def cleanup(self, context):
        visual_path = context.zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
        return {
            "status": "cleaned",
            "raw_source_count": 0,
            "part_file_count": 0,
            "cache_entry_count": 0,
            "forbidden_c_artifact_count": 0,
            "cleanup_plan": [
                {
                    "path": "sources",
                    "kind": "raw_sources",
                    "existed": False,
                    "deleted_bytes": 0,
                },
                {
                    "path": "temp",
                    "kind": "temporary",
                    "existed": False,
                    "deleted_bytes": 0,
                },
                {
                    "path": "cache",
                    "kind": "cache",
                    "existed": False,
                    "deleted_bytes": 0,
                },
                {
                    "path": ".",
                    "kind": "parts",
                    "existed": False,
                    "deleted_bytes": 0,
                },
            ],
            "preserved": {
                "tile_count": len(context.plan["tiles"]),
                "index_sha256": _hash("canonical-index"),
                "visual_receipt_sha256": sha256_file(visual_path),
            },
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    def mapping(self):
        return {
            "preflight": self.preflight,
            "pilot": self.pilot,
            "produce": self.produce,
            "qa": self.qa,
            "accept": self.accept,
            "cleanup": self.cleanup,
        }


def _controller(root: Path, backends=None, *, free_bytes: int = 100 * 1024**3):
    return TerrainController(
        backends=backends,
        coordinator_root=root / "terrain-global-coordinator",
        disk_usage=lambda _path: DiskUsage(
            total=200 * 1024**3,
            used=200 * 1024**3 - free_bytes,
            free=free_bytes,
        ),
    )


def test_plan_is_morton_ordered_square_and_has_exact_seams(d_workspace: Path) -> None:
    path = _write_zone(d_workspace)
    zone = load_zone_spec(path)
    plan = build_zone_plan(zone)

    assert plan["schema"] == ZONE_PLAN_SCHEMA
    assert plan["grid"] == {
        "tile_size_m": 500,
        "halo_m": 10.0,
        "columns": 3,
        "rows": 3,
        "order": "morton",
    }
    assert [tile["grid"] for tile in plan["tiles"]] == [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 0],
        [2, 1],
        [0, 2],
        [1, 2],
        [2, 2],
    ]
    assert plan["summary"] == {
        "tile_count": 9,
        "seam_count": 12,
        "source_request_count": 27,
    }
    first = plan["tiles"][0]
    assert first["bounds_l93_m"] == [700000, 6300000, 700500, 6300500]
    assert first["processing_bounds_l93_m"] == [
        699990.0,
        6299990.0,
        700510.0,
        6300510.0,
    ]


def test_recipe_ignores_machine_paths_and_storage_but_plan_tracks_storage(
    d_workspace: Path,
) -> None:
    first = _write_zone(
        d_workspace / "first",
        work_root=d_workspace / "work-a",
        export_root=d_workspace / "exports-a",
        estimated_peak_bytes=1000,
    )
    second_root = d_workspace / "second"
    second_root.mkdir()
    second = _write_zone(
        second_root,
        work_root=d_workspace / "work-b",
        export_root=d_workspace / "exports-b",
        estimated_peak_bytes=2000,
    )

    first_plan = build_zone_plan(load_zone_spec(first))
    second_plan = build_zone_plan(load_zone_spec(second))

    assert first_plan["recipe_id"] == second_plan["recipe_id"]
    assert first_plan["plan_id"] != second_plan["plan_id"]


def test_source_content_changes_build_id_without_changing_run_state_identity(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    plan = build_zone_plan(load_zone_spec(zone_path))
    sources = []
    for request in plan["source_requests"]:
        sources.append(
            {
                "id": request["id"],
                "product": request["product"],
                "request_sha256": canonical_sha256(request["request"]),
                "source_revision_id": request["source_revision_id"],
                "identity_status": "expected_identity_locked",
                "sha256": request["expected_sha256"],
                "byte_count": request["expected_byte_count"],
                "license": request["license"],
            }
        )
    first = build_source_lock(
        plan,
        {
            "sources": sources,
            "dependency_proofs": _dependency_proofs_from_plan(plan),
            "estimated_peak_bytes": plan["storage"]["estimated_peak_bytes"],
        },
    )
    assert first["schema"] == SOURCE_LOCK_SCHEMA
    assert first["recipe_id"] == plan["recipe_id"]

    unlocked_path = _write_zone(
        d_workspace,
        revision="r2",
        source_hash_suffix="b",
    )
    changed_plan = build_zone_plan(load_zone_spec(unlocked_path))
    changed_sources = []
    for request in changed_plan["source_requests"]:
        changed_sources.append(
            {
                "id": request["id"],
                "product": request["product"],
                "request_sha256": canonical_sha256(request["request"]),
                "source_revision_id": request["source_revision_id"],
                "identity_status": "expected_identity_locked",
                "sha256": request["expected_sha256"],
                "byte_count": request["expected_byte_count"],
                "license": request["license"],
            }
        )
    second = build_source_lock(
        changed_plan,
        {
            "sources": changed_sources,
            "dependency_proofs": _dependency_proofs_from_plan(changed_plan),
            "estimated_peak_bytes": changed_plan["storage"]["estimated_peak_bytes"],
        },
    )
    assert first["recipe_build_id"] != second["recipe_build_id"]


def test_https_sources_bootstrap_from_revision_and_finalize_observed_identity(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    payload = json.loads(zone_path.read_text(encoding="utf-8"))
    for request in payload["source_requests"]:
        request.pop("expected_sha256")
        request.pop("expected_byte_count")
    zone_path.write_text(json.dumps(payload), encoding="utf-8")
    plan = build_zone_plan(load_zone_spec(zone_path))
    provisional_sources = [
        {
            "id": request["id"],
            "product": request["product"],
            "request_sha256": canonical_sha256(request["request"]),
            "source_revision_id": request["source_revision_id"],
            "identity_status": "revision_locked",
            "sha256": None,
            "byte_count": None,
            "license": request["license"],
        }
        for request in plan["source_requests"]
    ]
    source_lock = build_source_lock(
        plan,
        {
            "sources": provisional_sources,
            "dependency_proofs": _dependency_proofs_from_plan(plan),
            "estimated_peak_bytes": plan["storage"]["estimated_peak_bytes"],
        },
    )
    assert source_lock["identity_status"] == "provisional_source_revision"
    observed = [
        {
            "id": request["id"],
            "product": request["product"],
            "request_sha256": canonical_sha256(request["request"]),
            "source_revision_id": request["source_revision_id"],
            "sha256": _hash(f"observed-{request['id']}-{request['product']}"),
            "byte_count": 8192,
            "license": request["license"],
        }
        for request in plan["source_requests"]
    ]
    tiles = [
        {"tile_id": tile["id"], "tile_done_sha256": _hash(tile["id"])}
        for tile in plan["tiles"]
    ]
    first = build_final_source_identity(
        plan,
        recipe_build_id=source_lock["recipe_build_id"],
        observed_sources=observed,
        tile_receipts=tiles,
    )
    changed = [dict(record) for record in observed]
    changed[0]["sha256"] = _hash("provider-replacement")
    second = build_final_source_identity(
        plan,
        recipe_build_id=source_lock["recipe_build_id"],
        observed_sources=changed,
        tile_receipts=tiles,
    )
    assert first["recipe_build_id"] == second["recipe_build_id"]
    assert first["build_id"] != second["build_id"]


def test_file_source_requires_expected_hash_and_size(d_workspace: Path) -> None:
    zone_path = _write_zone(d_workspace)
    payload = json.loads(zone_path.read_text(encoding="utf-8"))
    request = payload["source_requests"][0]
    request["request"]["service_url"] = "file:///D:/fixture-mnt.tif"
    request.pop("expected_sha256")
    request.pop("expected_byte_count")
    zone_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="file sources require expected identity"):
        load_zone_spec(zone_path)


def test_pilot_adds_declared_regression_tile_outside_contiguous_base(
    d_workspace: Path,
) -> None:
    regression_tile_id = "x701000_y6301000_s500"
    zone_path = _write_zone(d_workspace, regression_tile_id=regression_tile_id)
    fixture = FixtureBackends()
    controller = _controller(d_workspace, fixture.mapping())
    controller.run(zone_path, phase="plan", mode="execute")
    controller.run(zone_path, phase="preflight", mode="execute")
    result = controller.run(zone_path, phase="pilot", mode="execute", max_tiles=4)
    receipt = json.loads(
        (
            load_zone_spec(zone_path).receipt_root / PHASE_RECEIPT_NAMES["pilot"]
        ).read_text(encoding="utf-8")
    )
    selected = receipt["result"]["selection"]["tile_ids"]
    assert result["status"] == "completed"
    assert len(selected) == 5
    assert regression_tile_id in selected


def test_dry_run_writes_nothing_and_reports_peak_plus_20_gib(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace, estimated_peak_bytes=12345)
    controller = _controller(d_workspace)

    result = controller.run(zone_path, phase="plan", mode="dry-run")

    assert result["eligible"] is True
    assert result["writes_performed"] is False
    assert result["network_access_performed"] is False
    assert result["disk"]["required_free_bytes"] == 12345 + SAFETY_MARGIN_BYTES
    assert not (d_workspace / "fireviewer-work").exists()


def test_full_phase_chain_is_gated_and_cleanup_releases_active_zone(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    fixture = FixtureBackends()
    controller = _controller(d_workspace, fixture.mapping())
    zone = load_zone_spec(zone_path)

    assert (
        controller.run(zone_path, phase="plan", mode="execute")["status"] == "completed"
    )
    assert (
        controller.run(zone_path, phase="preflight", mode="execute")["status"]
        == "completed"
    )
    assert (
        controller.run(zone_path, phase="pilot", mode="execute")["status"]
        == "completed"
    )
    assert (
        controller.run(zone_path, phase="produce", mode="execute")["status"]
        == "completed"
    )
    assert (
        controller.run(zone_path, phase="qa", mode="execute")["status"] == "completed"
    )
    _record_explicit_zone_visual_acceptance(zone)
    accepted = controller.run(zone_path, phase="accept", mode="execute")
    assert accepted["status"] == "completed"
    acceptance = json.loads(
        (zone.receipt_root / PHASE_RECEIPT_NAMES["accept"]).read_text(encoding="utf-8")
    )
    assert acceptance["schema"] == ZONE_ACCEPTANCE_SCHEMA
    assert acceptance["tile_count"] == 9
    assert acceptance["seam_count"] == 12
    assert acceptance["runtime_shader"] == {
        "status": "pending_dedicated_mdl_validation"
    }
    assert acceptance["usd_runtime_gate"] is False

    cleaned = controller.run(zone_path, phase="cleanup", mode="execute")
    assert cleaned["status"] == "completed"
    assert not controller.active_zone_path.exists()
    repeated = controller.run(zone_path, phase="cleanup", mode="execute")
    assert repeated["status"] == "completed"
    assert not controller.active_zone_path.exists()
    assert fixture.calls == [
        ("preflight", "execute", 2, 1),
        ("pilot", "execute", 2, 1),
    ]


def test_qa_requires_the_exact_tile_and_seam_sets(d_workspace: Path) -> None:
    zone_path = _write_zone(d_workspace)
    fixture = FixtureBackends()
    controller = _controller(d_workspace, fixture.mapping())
    controller.run(zone_path, phase="plan", mode="execute")
    controller.run(zone_path, phase="preflight", mode="execute")
    controller.run(zone_path, phase="pilot", mode="execute")
    controller.run(zone_path, phase="produce", mode="execute")

    def incomplete_qa(_context):
        return {
            "status": "passed",
            "validated_tile_ids": [],
            "validated_seam_ids": [],
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    bad_controller = _controller(d_workspace, {"qa": incomplete_qa})
    with pytest.raises(ContractError, match="every planned tile"):
        bad_controller.run(zone_path, phase="qa", mode="execute")


def test_qa_rejects_a_stitch_signature_mismatch(d_workspace: Path) -> None:
    zone_path = _write_zone(d_workspace)
    fixture = FixtureBackends()
    controller = _controller(d_workspace, fixture.mapping())
    for phase in ("plan", "preflight", "pilot", "produce"):
        controller.run(zone_path, phase=phase, mode="execute")

    def bad_qa(context):
        result = fixture.qa(context)
        result["metrics"]["stitch_signature_mismatch_count"] = 1
        return result

    bad_controller = _controller(d_workspace, {"qa": bad_qa})
    with pytest.raises(ContractError, match="blocking terrain"):
        bad_controller.run(zone_path, phase="qa", mode="execute")


def test_failed_phase_requires_resume_and_keeps_receipts_atomic(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    fixture = FixtureBackends()
    fixture.fail_pilot_once = True
    controller = _controller(d_workspace, fixture.mapping())
    zone = load_zone_spec(zone_path)

    controller.run(zone_path, phase="plan", mode="execute")
    controller.run(zone_path, phase="preflight", mode="execute")
    with pytest.raises(RuntimeError, match="simulated interrupted pilot"):
        controller.run(zone_path, phase="pilot", mode="execute")
    assert not (zone.receipt_root / PHASE_RECEIPT_NAMES["pilot"]).exists()
    state = json.loads(zone.state_path.read_text(encoding="utf-8"))
    assert state["phases"]["pilot"]["status"] == "failed"
    assert state["phases"]["pilot"]["attempts"] == 1

    with pytest.raises(Exception, match="use --resume"):
        controller.run(zone_path, phase="pilot", mode="execute")
    with pytest.raises(Exception, match="differs from the interrupted attempt"):
        controller.run(zone_path, phase="pilot", mode="resume", max_tiles=4)
    resumed = controller.run(zone_path, phase="pilot", mode="resume")
    assert resumed["status"] == "completed"
    state = json.loads(zone.state_path.read_text(encoding="utf-8"))
    assert state["phases"]["pilot"]["attempts"] == 2


def test_resume_reconciles_plan_receipt_after_state_write_crash(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    controller = _controller(d_workspace)
    zone = load_zone_spec(zone_path)
    controller.run(zone_path, phase="plan", mode="execute")
    zone.state_path.unlink()

    result = controller.run(zone_path, phase="plan", mode="resume")

    assert result["status"] == "completed"
    state = json.loads(zone.state_path.read_text(encoding="utf-8"))
    assert state["phases"]["plan"]["receipt_sha256"] == sha256_file(
        zone.receipt_root / PHASE_RECEIPT_NAMES["plan"]
    )


def test_resume_reconciles_source_lock_and_build_id_after_state_write_crash(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    fixture = FixtureBackends()
    controller = _controller(d_workspace, fixture.mapping())
    zone = load_zone_spec(zone_path)
    controller.run(zone_path, phase="plan", mode="execute")
    controller.run(zone_path, phase="preflight", mode="execute")
    source_lock = json.loads(
        (zone.receipt_root / PHASE_RECEIPT_NAMES["preflight"]).read_text(
            encoding="utf-8"
        )
    )
    state = json.loads(zone.state_path.read_text(encoding="utf-8"))
    state["recipe_build_id"] = None
    state["phases"]["preflight"]["status"] = "running"
    state["phases"]["preflight"]["receipt"] = None
    state["phases"]["preflight"]["receipt_sha256"] = None
    zone.state_path.write_text(json.dumps(state), encoding="utf-8")

    result = controller.run(zone_path, phase="preflight", mode="resume")

    assert result["recipe_build_id"] == source_lock["recipe_build_id"]
    assert result["build_id"] is None
    reconciled = json.loads(zone.state_path.read_text(encoding="utf-8"))
    assert reconciled["phases"]["preflight"]["status"] == "completed"
    assert reconciled["recipe_build_id"] == source_lock["recipe_build_id"]


def test_active_zone_blocks_a_second_zone_until_cleanup(d_workspace: Path) -> None:
    first = _write_zone(
        d_workspace / "first",
        work_root=d_workspace / "shared-work",
        export_root=d_workspace / "first-exports",
    )
    second_root = d_workspace / "second"
    second_root.mkdir()
    second = _write_zone(
        second_root,
        zone_id="FR-TEST-00002",
        work_root=d_workspace / "independent-work",
        export_root=d_workspace / "second-exports",
    )
    controller = _controller(d_workspace)

    controller.run(first, phase="plan", mode="execute")
    with pytest.raises(Exception, match="another zone or recipe is active"):
        controller.run(second, phase="plan", mode="execute")


def test_preflight_fails_before_backend_when_peak_plus_margin_is_unavailable(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace, estimated_peak_bytes=5 * 1024**3)
    fixture = FixtureBackends()
    controller = _controller(
        d_workspace,
        fixture.mapping(),
        free_bytes=SAFETY_MARGIN_BYTES + 5 * 1024**3 - 1,
    )
    controller.run(zone_path, phase="plan", mode="execute")

    with pytest.raises(StoragePolicyError, match="estimated peak plus 20 GiB"):
        controller.run(zone_path, phase="preflight", mode="execute")
    assert fixture.calls == []


def test_backend_c_artifact_audit_fails_closed(d_workspace: Path) -> None:
    zone_path = _write_zone(d_workspace)

    def bad_preflight(context):
        return {
            "sources": _locked_sources(context),
            "estimated_peak_bytes": 1024,
            "created_paths": [],
            "forbidden_c_artifacts": ["C:\\temp\\fireviewer.part"],
        }

    controller = _controller(d_workspace, {"preflight": bad_preflight})
    controller.run(zone_path, phase="plan", mode="execute")
    with pytest.raises(StoragePolicyError, match="forbidden C: artifacts"):
        controller.run(zone_path, phase="preflight", mode="execute")


def test_zone_contract_rejects_hidden_c_path_in_source_request(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    payload = json.loads(zone_path.read_text(encoding="utf-8"))
    payload["source_requests"][0]["request"]["cache_path"] = (
        "C:\\temp\\fireviewer\\mnt.tif"
    )
    zone_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StoragePolicyError, match="forbidden C: path"):
        load_zone_spec(zone_path)


def test_cli_refuses_all_multiple_zones_and_max_tiles_outside_pilot(
    d_workspace: Path,
) -> None:
    zone = _write_zone(d_workspace)
    with pytest.raises(SystemExit):
        main(["--zone", str(zone), "--phase", "plan", "--dry-run", "--all"])
    with pytest.raises(SystemExit):
        main(
            [
                "--zone",
                str(zone),
                "--zone",
                str(zone),
                "--phase",
                "plan",
                "--dry-run",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--zone",
                str(zone),
                "--phase",
                "produce",
                "--max-tiles",
                "9",
                "--dry-run",
            ]
        )


def test_injected_controller_without_backend_does_not_advance_preflight(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    controller = _controller(d_workspace)
    controller.run(zone_path, phase="plan", mode="execute")
    zone = load_zone_spec(zone_path)
    active_before = controller.active_zone_path.read_bytes()

    with pytest.raises(BackendUnavailableError, match="no injected backend"):
        controller.run(zone_path, phase="preflight", mode="execute")
    assert controller.active_zone_path.read_bytes() == active_before
    state = json.loads(zone.state_path.read_text(encoding="utf-8"))
    assert state["phases"]["preflight"]["status"] == "pending"


def test_failed_backend_statuses_and_missing_metrics_fail_closed(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    fixture = FixtureBackends()
    controller = _controller(d_workspace, fixture.mapping())
    controller.run(zone_path, phase="plan", mode="execute")
    controller.run(zone_path, phase="preflight", mode="execute")

    failed_pilot = _controller(
        d_workspace,
        {
            "pilot": lambda _context: {
                "status": "failed",
                "created_paths": [],
                "forbidden_c_artifacts": [],
            }
        },
    )
    with pytest.raises(ContractError, match="pilot status must be 'passed'"):
        failed_pilot.run(zone_path, phase="pilot", mode="execute")

    controller.run(zone_path, phase="pilot", mode="resume")

    def produce_without_metrics(context):
        return {
            "status": "passed",
            "tile_receipts": [
                _tile_receipt(context, tile["id"]) for tile in context.plan["tiles"]
            ],
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    missing_metrics = _controller(d_workspace, {"produce": produce_without_metrics})
    with pytest.raises(ContractError, match="strict metrics"):
        missing_metrics.run(zone_path, phase="produce", mode="execute")


def test_preflight_requires_exact_dependency_proofs_and_accepted_clean_library(
    d_workspace: Path,
) -> None:
    zone_path = _write_zone(d_workspace)
    plan = build_zone_plan(load_zone_spec(zone_path))
    sources = [
        {
            "id": request["id"],
            "product": request["product"],
            "request_sha256": canonical_sha256(request["request"]),
            "source_revision_id": request["source_revision_id"],
            "identity_status": "expected_identity_locked",
            "sha256": request["expected_sha256"],
            "byte_count": request["expected_byte_count"],
            "license": request["license"],
        }
        for request in plan["source_requests"]
    ]
    with pytest.raises(ContractError, match="dependency_proofs"):
        build_source_lock(
            plan,
            {
                "sources": sources,
                "estimated_peak_bytes": plan["storage"]["estimated_peak_bytes"],
            },
        )

    proofs = _dependency_proofs_from_plan(plan)
    proofs["clean_pbr_texture_library"] = {
        **proofs["clean_pbr_texture_library"],
        "status": "pending",
    }
    with pytest.raises(
        ContractError, match="clean PBR library dependency is not accepted"
    ):
        build_source_lock(
            plan,
            {
                "sources": sources,
                "dependency_proofs": proofs,
                "estimated_peak_bytes": plan["storage"]["estimated_peak_bytes"],
            },
        )


def test_zone_rejects_unpaired_sources_and_drive_relative_c_path(
    d_workspace: Path,
) -> None:
    unpaired = _write_zone(d_workspace / "unpaired")
    payload = json.loads(unpaired.read_text(encoding="utf-8"))
    payload["source_requests"] = [
        record
        for record in payload["source_requests"]
        if not (record["id"] == "tile-0-0" and record["product"] == "mns")
    ]
    unpaired.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="must contain one MNT, one MNS"):
        load_zone_spec(unpaired)

    c_relative = _write_zone(d_workspace / "c-relative")
    payload = json.loads(c_relative.read_text(encoding="utf-8"))
    payload["source_requests"][0]["request"]["service_url"] = "C:terrain.tif"
    c_relative.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StoragePolicyError, match="forbidden C: path"):
        load_zone_spec(c_relative)


def test_qa_rejects_changed_or_non_lod0_visual_proof(d_workspace: Path) -> None:
    zone_path = _write_zone(d_workspace)
    fixture = FixtureBackends()
    controller = _controller(d_workspace, fixture.mapping())
    for phase in ("plan", "preflight", "pilot", "produce"):
        controller.run(zone_path, phase=phase, mode="execute")

    def bad_visual_qa(context):
        result = dict(fixture.qa(context))
        visual_path = context.zone.work_root.joinpath(
            *Path(ZONE_VISUAL_TECHNICAL_RELATIVE_PATH).parts
        )
        visual = json.loads(visual_path.read_text(encoding="utf-8"))
        visual["aov"]["invalid_lod_pixel_count"] = 1
        visual_path.write_text(json.dumps(visual), encoding="utf-8")
        result["visual_technical_receipt_sha256"] = sha256_file(visual_path)
        return result

    bad_controller = _controller(d_workspace, {"qa": bad_visual_qa})
    with pytest.raises(ContractError, match="technical receipt"):
        bad_controller.run(zone_path, phase="qa", mode="execute")
