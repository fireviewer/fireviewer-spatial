from __future__ import annotations

from collections import namedtuple
import copy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from blender import validate_adaptive_terrain_zone as zone_qa

from terrain_production_backend import (
    BLENDER_REPORT_SCHEMA,
    ProductionTerrainBackend,
    SURFACE_FEATURE_SCHEMA,
    TOOLCHAIN_SCHEMA,
    _default_opener,
)
from blender.build_adaptive_terrain_fixture import (
    _surface_features,
)
from blender import clean_pbr_texture_library as clean_library
from blender.test_clean_pbr_texture_library import (
    _artifact as _library_artifact,
    _atlas_uv,
    _write_png,
)
from blender.test_orthophoto_surface_correspondence import _library_and_model
from blender.compact_hag import write_hag_max_2m
from terrainctl import (
    PHASE_RECEIPT_NAMES,
    StoragePolicyError,
    TerrainController,
    ZONE_VISUAL_RECEIPT_NAME,
    ZONE_VISUAL_REVIEW_RELATIVE_PATH,
    ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH,
    ZONE_VISUAL_SCHEMA,
    ZONE_VISUAL_TECHNICAL_RELATIVE_PATH,
    build_final_source_identity,
    canonical_sha256,
    load_zone_spec,
    sha256_file,
)


DiskUsage = namedtuple("DiskUsage", "total used free")
PRODUCTION_TEST_ROOT = Path(
    "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest/backend"
)


@pytest.fixture
def production_workspace() -> Path:
    root = PRODUCTION_TEST_ROOT / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.resolve().is_relative_to(PRODUCTION_TEST_ROOT.resolve()):
            shutil.rmtree(root, ignore_errors=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_source(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=260,
        height=260,
        count=1,
        dtype="float32",
        crs="EPSG:2154",
        transform=from_origin(699_990.0, 6_300_510.0, 2.0, 2.0),
        nodata=-9999.0,
        compress="deflate",
    ) as dataset:
        dataset.write(values.astype("float32"), 1)


class _MemoryResponse:
    def __init__(self, payload: bytes) -> None:
        self._stream = BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self.status = 200

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self._stream.close()


def _orthophoto_png() -> bytes:
    rgb = np.empty((520, 520, 3), dtype=np.uint8)
    rgb[:] = (172, 142, 72)
    stream = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


ORTHOPHOTO_FIXTURE_BYTES = _orthophoto_png()


def _fixture_opener(request, *, timeout: float):
    if str(request.full_url).startswith("https://fixture.invalid/"):
        return _MemoryResponse(ORTHOPHOTO_FIXTURE_BYTES)
    return _default_opener(request, timeout=timeout)


def _write_accepted_clean_library(root: Path) -> tuple[Path, Path]:
    contract_path = (
        Path(__file__).resolve().parent
        / "blender"
        / "ground_surface_texture_contract.v4.json"
    )
    contract = clean_library.load_texture_contract(contract_path)
    matcher_library, model = _library_and_model()
    payload = copy.deepcopy(matcher_library)
    payload["runtime_atlases"] = {}
    for role_index, role in enumerate(clean_library.REQUIRED_TEXTURE_ROLES):
        relative = f"runtime-atlas/{role}.png"
        _write_png(
            root / relative,
            width=contract["runtime_atlas"]["width_px"],
            height=contract["runtime_atlas"]["height_px"],
            role=role,
            seed=31 + role_index,
        )
        payload["runtime_atlases"][role] = _library_artifact(root, relative)
    for index, profile in enumerate(payload["profiles"]):
        profile["atlas_uv"] = _atlas_uv(contract, index)
        textures = {}
        for role_index, role in enumerate(clean_library.REQUIRED_TEXTURE_ROLES):
            relative = f"profile-sources/{index:02d}/{role}.png"
            _write_png(
                root / relative,
                width=4,
                height=4,
                role=role,
                seed=(index * 11 + role_index * 37) & 0xFF,
            )
            textures[role] = _library_artifact(root, relative)
        profile["textures"] = textures
    payload["status"] = clean_library.ACCEPTED_LIBRARY_STATUS
    payload["texture_contract_sha256"] = sha256_file(contract_path)
    content = copy.deepcopy(payload)
    content.pop("status")
    content.pop("visual_acceptance")
    content_sha256 = hashlib.sha256(
        clean_library._canonical_bytes(content)  # noqa: SLF001 - contract fixture
    ).hexdigest()
    acceptance = {
        "schema": "fireviewer.clean-pbr-texture-visual-acceptance.v1",
        "status": "accepted_human_visual",
        "texture_contract_sha256": sha256_file(contract_path),
        "library_content_sha256": content_sha256,
        "profile_count": 72,
        "atlas_roles": ["basecolor", "normal", "height", "orm"],
        "invalid_profile_count": 0,
    }
    acceptance_path = root / "qa" / "accepted-human-visual.json"
    _write_json(acceptance_path, acceptance)
    payload["visual_acceptance"] = {
        "status": "accepted_human_visual",
        "receipt": _library_artifact(root, "qa/accepted-human-visual.json"),
    }
    library_path = root / "clean-pbr-texture-library.v1.json"
    _write_json(library_path, payload)
    clean_library.validate_texture_library(
        library_path,
        contract_path=contract_path,
        require_visual_acceptance=True,
    )
    model_path = root / "orthophoto-surface-model.v1.json"
    _write_json(model_path, model)
    return library_path, model_path


def _write_production_zone(root: Path) -> Path:
    dependencies_root = root / "dependencies"
    source_root = root / "source-fixtures"
    work_root = root / "fireviewer-work"
    export_root = root / "fireviewer-exports"

    library_path, model_path = _write_accepted_clean_library(
        dependencies_root / "clean-pbr"
    )
    feature_path = dependencies_root / "surface-features.v1.json"
    _write_json(
        feature_path,
        {
            "schema": SURFACE_FEATURE_SCHEMA,
            "crs": "EPSG:2154",
            "bounds_l93_m": [700_000.0, 6_300_000.0, 700_500.0, 6_300_500.0],
            "features": _surface_features()[:2],
            "context_priors": [],
            "approved_corrections": [],
            "provenance": {
                name: canonical_sha256({"fixture": name})
                for name in (
                    "gpkg",
                    "manifest",
                    "ground_context_contract",
                    "ground_texture_contract",
                    "export_algorithm",
                )
            },
        },
    )

    fake_blender = dependencies_root / "blender-4.5-lts" / "blender.exe"
    fake_blender.parent.mkdir(parents=True, exist_ok=True)
    fake_blender.write_bytes(b"fixture blender 4.5 LTS\n")
    toolchain_path = dependencies_root / "terrain-toolchain.v1.json"
    _write_json(
        toolchain_path,
        {
            "schema": TOOLCHAIN_SCHEMA,
            "blender": {
                "version": "4.5.3 LTS fixture",
                "path": str(fake_blender),
                "sha256": sha256_file(fake_blender),
            },
        },
    )
    texture_contract = (
        Path(__file__).resolve().parent
        / "blender"
        / "ground_surface_texture_contract.v4.json"
    )
    correspondence_contract = (
        Path(__file__).resolve().parent
        / "blender"
        / "orthophoto_surface_correspondence_contract.v1.json"
    )
    quadtree_contract = (
        Path(__file__).resolve().parent
        / "blender"
        / "terrain_quadtree_contract.v1.json"
    )
    algorithm_path = (
        Path(__file__).resolve().parent / "blender" / "adaptive_terrain_quadtree.py"
    )

    rows, columns = np.mgrid[0:260, 0:260]
    mnt = 100.0 + columns * 0.01 + rows * 0.005
    mns = mnt + 2.5 + ((columns + rows) % 7) * 0.01
    mnt_path = source_root / "mnt-2m.tif"
    mns_path = source_root / "mns-2m.tif"
    _write_source(mnt_path, mnt)
    _write_source(mns_path, mns)

    dependency_paths = {
        "algorithm": algorithm_path,
        "clean_pbr_texture_library": library_path,
        "ground_texture_contract": texture_contract,
        "surface_correspondence_contract": correspondence_contract,
        "surface_correspondence_model": model_path,
        "surface_features": feature_path,
        "terrain_quadtree_contract": quadtree_contract,
        "toolchain": toolchain_path,
    }
    dependencies = {name: sha256_file(path) for name, path in dependency_paths.items()}
    source_requests = []
    for product, path in (("mnt", mnt_path), ("mns", mns_path)):
        source_requests.append(
            {
                "id": "tile-0-0",
                "product": product,
                "request": {
                    "service_url": path.resolve().as_uri(),
                    "layer": f"FIXTURE-{product.upper()}-2M",
                    "core_bounds_l93_m": [
                        700_000,
                        6_300_000,
                        700_500,
                        6_300_500,
                    ],
                    "resolution_m": 2,
                },
                "expected_sha256": sha256_file(path),
                "expected_byte_count": path.stat().st_size,
                "license": "synthetic-test-fixture",
                "pilot_score": 1.0,
            }
        )
    source_requests.append(
        {
            "id": "tile-0-0",
            "product": "orthophoto",
            "request": {
                "service_kind": "wms",
                "service_url": "https://fixture.invalid/orthophoto",
                "layer": "ORTHO-RGB-1M",
                "core_bounds_l93_m": [
                    700_000,
                    6_300_000,
                    700_500,
                    6_300_500,
                ],
                "resolution_m": 1,
                "halo_m": 10,
                "image_format": "image/png",
                "maximum_download_bytes": 4 * 1024**2,
            },
            "source_revision_id": "fixture-ortho-r1",
            "license": "synthetic-test-fixture",
            "pilot_score": 1.0,
        }
    )
    zone = {
        "schema": "fireviewer.zone-spec.v1",
        "zone_id": "FR-TEST-PRODUCTION",
        "revision": "r1",
        "crs": "EPSG:2154",
        "bounds_l93_m": [700_000, 6_300_000, 700_500, 6_300_500],
        "tile_size_m": 500,
        "halo_m": 10,
        "source_resolution_m": 2,
        "dependencies": dependencies,
        "dependency_artifacts": {
            name: {"path": str(path), "sha256": dependencies[name]}
            for name, path in dependency_paths.items()
        },
        "source_requests": source_requests,
        "workspace": {
            "work_root": str(work_root),
            "export_root": str(export_root),
        },
        "storage": {"estimated_peak_bytes": 64 * 1024**2},
    }
    zone_path = root / "FR-TEST-PRODUCTION-r1.zone-spec.v1.json"
    _write_json(zone_path, zone)
    return zone_path


def _fake_blender_runner(context, tile_root, report_path, render_path, blender_path):
    assert blender_path.drive.upper() == "D:"
    assert Path(context.environment["TEMP"]).drive.upper() == "D:"
    render_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path = render_path.with_name(f"{tile_root.name}.terrain-coverage-aov.exr")
    oblique_lod_path = render_path.with_name(
        f"{render_path.stem}-oblique{render_path.suffix}"
    )
    oblique_coverage_path = coverage_path.with_name(
        f"{coverage_path.stem}-oblique{coverage_path.suffix}"
    )
    beauty_topdown_path = render_path.with_name(
        f"{tile_root.name}.textured-topdown.png"
    )
    beauty_oblique_path = render_path.with_name(
        f"{tile_root.name}.textured-oblique.png"
    )
    for path, payload in (
        (render_path, b"fixture EXR terrain LOD topdown AOV\n"),
        (coverage_path, b"fixture EXR terrain coverage topdown AOV\n"),
        (oblique_lod_path, b"fixture EXR terrain LOD oblique AOV\n"),
        (oblique_coverage_path, b"fixture EXR terrain coverage oblique AOV\n"),
        (beauty_topdown_path, b"fixture PNG textured topdown\n"),
        (beauty_oblique_path, b"fixture PNG textured oblique\n"),
    ):
        path.write_bytes(payload)

    def proof(path: Path, **fields: object) -> dict[str, object]:
        return {"path": path.name, "sha256": sha256_file(path), **fields}

    acceptance_path, acceptance = ProductionTerrainBackend._clean_library_acceptance(
        context
    )

    report = {
        "schema": BLENDER_REPORT_SCHEMA,
        "status": "accepted_blender_textured_visual",
        "geometry_lod_status": "accepted_blender_geometry_lod",
        "production_visual_gate_passed": True,
        "human_visual_acceptance": "accepted_human_visual",
        "source_library_status": "accepted_clean_pbr_library",
        "surface_library_acceptance_receipt_sha256": sha256_file(acceptance_path),
        "surface_library_visual_acceptance_receipt": {
            "schema": "fireviewer.clean-pbr-texture-visual-acceptance.v1",
            "status": "accepted_human_visual",
            "path": acceptance_path.name,
            "sha256": sha256_file(acceptance_path),
            "source_library_schema": "fireviewer.clean-pbr-texture-library.v1",
            "source_library_identity_sha256": canonical_sha256(
                {"fixture": "library-identity"}
            ),
            "library_content_sha256": acceptance["library_content_sha256"],
            "texture_contract_sha256": acceptance["texture_contract_sha256"],
        },
        "tile_id": tile_root.name,
        "selected_lod": 0,
        "render_resolution": [512, 512],
        "beauty": {
            "topdown": proof(
                beauty_topdown_path,
                terrain_pixel_count=230_000,
                frame_coverage_ratio=0.88,
                distinct_rgb8_count=128,
            ),
            "oblique": proof(
                beauty_oblique_path,
                terrain_pixel_count=64_000,
                frame_coverage_ratio=0.24,
                distinct_rgb8_count=128,
            ),
        },
        "aov": {
            "validated_primary_views": ["topdown", "oblique"],
            "lod": proof(
                render_path,
                name="fireviewer:terrain_lod",
                expected_value=0,
                terrain_pixel_count=230_000,
                invalid_pixel_count=0,
                maximum_absolute_error=0.0,
            ),
            "coverage": proof(
                coverage_path,
                name="fireviewer:terrain_coverage",
                expected_value=1,
                invalid_pixel_count=0,
            ),
            "oblique_lod": proof(
                oblique_lod_path,
                name="fireviewer:terrain_lod",
                expected_value=0,
                terrain_pixel_count=64_000,
                invalid_pixel_count=0,
                maximum_absolute_error=0.0,
            ),
            "oblique_coverage": proof(
                oblique_coverage_path,
                name="fireviewer:terrain_coverage",
                expected_value=1,
                invalid_pixel_count=0,
            ),
        },
    }
    _write_json(report_path, report)
    return report


def _fake_zone_visual_runner(context, job_path, blender_path):
    assert blender_path.drive.upper() == "D:"
    assert Path(context.environment["TEMP"]).drive.upper() == "D:"
    inspection = zone_qa.inspect_zone_job(
        job_path, require_d=True, validate_packages=True
    )
    plan = zone_qa.build_capture_plan(inspection)
    plan_path = inspection.output_root / "zone-visual-capture-plan.v1.json"
    _write_json(plan_path, plan)
    job_sha256 = sha256_file(job_path)
    capture_records = []
    for capture in plan["captures"]:
        capture_root = inspection.output_root / "captures" / str(capture["capture_id"])
        capture_root.mkdir(parents=True, exist_ok=True)
        for name, file_name in zone_qa.CAPTURE_ARTIFACT_NAMES.items():
            (capture_root / file_name).write_bytes(
                f"fixture:{capture['capture_id']}:{name}\n".encode()
            )
        receipt = {
            "schema": zone_qa.CAPTURE_RECEIPT_SCHEMA,
            "status": "rendered_technical",
            "capture_id": capture["capture_id"],
            "category": capture["category"],
            "capture_spec_sha256": zone_qa._canonical_sha256(capture),
            "capture_plan_sha256": plan["plan_sha256"],
            "job_sha256": job_sha256,
            "tile_ids": list(capture["tile_ids"]),
            "artifacts": {
                name: zone_qa._artifact(
                    capture_root / file_name, relative_to=capture_root
                )
                for name, file_name in zone_qa.CAPTURE_ARTIFACT_NAMES.items()
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
                "tile_ids": list(capture["tile_ids"]),
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
        _write_json(capture_path, receipt)
        capture_records.append(receipt)
    technical = {
        "schema": zone_qa.TECHNICAL_RECEIPT_SCHEMA,
        "status": zone_qa.TECHNICAL_STATUS,
        "production_visual_gate_passed": False,
        "human_visual_acceptance": "pending_exhaustive_review",
        "zone_id": inspection.job["zone_id"],
        "revision": inspection.job["revision"],
        "recipe_id": inspection.job["recipe_id"],
        "recipe_build_id": inspection.job["recipe_build_id"],
        "build_id": inspection.job["build_id"],
        "job_sha256": job_sha256,
        "catalog_sha256": inspection.catalog_sha256,
        "qa_metrics_sha256": inspection.qa_metrics_sha256,
        "capture_plan": zone_qa._artifact(
            plan_path, relative_to=inspection.output_root
        ),
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
                    inspection.output_root
                    / "captures"
                    / str(record["capture_id"])
                    / "capture.done.v1.json",
                    relative_to=inspection.output_root,
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
    _write_json(technical_path, technical)
    return zone_qa.validate_technical_receipt(technical_path)


def _record_zone_visual_acceptance(zone) -> Path:
    template_path = zone.work_root.joinpath(
        *Path(ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH).parts
    )
    review = json.loads(template_path.read_text(encoding="utf-8"))
    review["decision"] = "accepted"
    review["reviewer"] = {"kind": "human", "id": "fixture-reviewer"}
    review["decision_recorded_at_utc"] = "2026-08-09T12:00:00Z"
    for capture in review["capture_reviews"]:
        capture["decision"] = "accepted"
    review_path = zone.work_root.joinpath(*Path(ZONE_VISUAL_REVIEW_RELATIVE_PATH).parts)
    _write_json(review_path, review)
    technical_path = zone.work_root.joinpath(
        *Path(ZONE_VISUAL_TECHNICAL_RELATIVE_PATH).parts
    )
    acceptance_path = zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
    zone_qa.accept_zone_visual_review(technical_path, review_path, acceptance_path)
    return acceptance_path


def _write_seam_surface(
    root: Path, *, edge_id: int = 0, confidence: int = 255, orientation: int = 0
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    identifiers = np.zeros((500, 500, 4), dtype=np.uint8)
    identifiers[:, :, 0] = edge_id
    weights = np.zeros_like(identifiers)
    weights[:, :, 0] = 255
    Image.fromarray(identifiers, mode="RGBA").save(root / "ground-profile-ids.png")
    Image.fromarray(weights, mode="RGBA").save(root / "ground-profile-weights.png")
    Image.fromarray(np.full((500, 500), confidence, dtype=np.uint8), mode="L").save(
        root / "ground-confidence.png"
    )
    Image.fromarray(np.full((500, 500), orientation, dtype=np.uint8), mode="L").save(
        root / "ground-orientation.png"
    )


def test_surface_seam_rejects_correspondence_discontinuities(
    production_workspace: Path,
) -> None:
    west = production_workspace / "west"
    east = production_workspace / "east"
    boundary = LineString([(500.0, 0.0), (500.0, 500.0)])
    _write_seam_surface(west)
    _write_seam_surface(east)
    assert (
        ProductionTerrainBackend._composition_seam_failures(
            west, east, edges=("east", "west"), boundary=boundary
        )
        == 0
    )

    _write_seam_surface(west, confidence=255)
    _write_seam_surface(east, confidence=127)
    assert (
        ProductionTerrainBackend._composition_seam_failures(
            west, east, edges=("east", "west"), boundary=boundary
        )
        > 0
    )

    _write_seam_surface(west, orientation=0)
    _write_seam_surface(east, orientation=64)
    assert (
        ProductionTerrainBackend._composition_seam_failures(
            west, east, edges=("east", "west"), boundary=boundary
        )
        > 0
    )

    _write_seam_surface(west, edge_id=0)
    _write_seam_surface(east, edge_id=1)
    assert (
        ProductionTerrainBackend._composition_seam_failures(
            west, east, edges=("east", "west"), boundary=boundary
        )
        > 0
    )


def test_seam_metric_replays_all_admissible_lod_pairs(
    production_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_root = production_workspace / "exports"
    roots = {
        name: export_root / "FR-TEST" / "r1" / "tiles" / name
        for name in ("west", "east")
    }
    for root in roots.values():
        _write_seam_surface(root)

    matching_signature = b"s" * 32

    def mesh(tile_id: str, lod: int):
        signatures = [matching_signature] * 4
        if tile_id == "east" and lod == 1:
            signatures = list(signatures)
            signatures[0] = b"x" * 32
        variant = SimpleNamespace(effective_edge_signatures=tuple(signatures))
        return SimpleNamespace(
            lod=lod,
            z_origin_mm=100_000,
            vertices=((0, 0, 0), (128, 0, 0), (128, 128, 0), (0, 128, 0)),
            vertex_gradients_mm_per_4m=((1, 2),) * 4,
            edge_vertex_indices=((0, 3), (1, 2), (0, 1), (3, 2)),
            stitch_variants=(variant,) * 16,
        )

    def fake_read_fvtq(path: Path):
        lod = int(path.stem[-1])
        return mesh(path.parent.name, lod)

    monkeypatch.setattr("terrain_production_backend.read_fvtq", fake_read_fvtq)
    context = SimpleNamespace(
        export_root=export_root,
        zone=SimpleNamespace(zone_id="FR-TEST", revision="r1"),
        plan={
            "tiles": [
                {
                    "id": "west",
                    "grid": [0, 0],
                    "bounds_l93_m": [700_000, 6_300_000, 700_500, 6_300_500],
                },
                {
                    "id": "east",
                    "grid": [1, 0],
                    "bounds_l93_m": [700_500, 6_300_000, 701_000, 6_300_500],
                },
            ]
        },
    )
    metric = ProductionTerrainBackend()._seam_metric(
        context, {"id": "west--east", "a": "west", "b": "east"}
    )
    assert metric["stitch_lod_pair_count"] == 7
    assert metric["stitch_signature_mismatch_count"] > 0


def test_backend_resume_revalidates_an_existing_source_pair(
    production_workspace: Path,
) -> None:
    rows, columns = np.mgrid[0:260, 0:260]
    mnt_path = production_workspace / "resume-input" / "mnt.tif"
    mns_path = production_workspace / "resume-input" / "mns.tif"
    _write_source(mnt_path, 80.0 + columns * 0.001 + rows * 0.001)
    _write_source(mns_path, 83.0 + columns * 0.001 + rows * 0.001)

    records = {}
    for role, path in (("mnt", mnt_path), ("mns", mns_path)):
        records[role] = {
            "id": "resume-pair",
            "product": role,
            "request": {
                "service_url": path.resolve().as_uri(),
                "layer": f"FIXTURE-{role.upper()}-2M",
                "core_bounds_l93_m": [
                    700_000,
                    6_300_000,
                    700_500,
                    6_300_500,
                ],
            },
            "expected_sha256": sha256_file(path),
            "expected_byte_count": path.stat().st_size,
            "license": "synthetic-test-fixture",
        }
    pair = SimpleNamespace(
        source_id="resume-pair",
        bounds=(700_000.0, 6_300_000.0, 700_500.0, 6_300_500.0),
        mnt=records["mnt"],
        mns=records["mns"],
    )
    context = SimpleNamespace(run_root=production_workspace / "resume-work")
    backend = ProductionTerrainBackend()
    raw_root, _receipt = backend._acquire_pair(context, pair)
    receipt_path = raw_root / "source-pair.done.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "forged-without-semantic-validation"
    _write_json(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="invalid contract"):
        backend._acquire_pair(context, pair)


def test_existing_tile_rejects_changed_mns_and_requires_a_new_build(
    production_workspace: Path,
) -> None:
    tile_id = "x700000_y6300000_s500"
    tile_root = production_workspace / "exports" / "FR-TEST" / "r1" / "tiles" / tile_id
    source_root = tile_root / "source"
    source_root.mkdir(parents=True)
    bounds = [700_000.0, 6_300_000.0, 700_500.0, 6_300_500.0]
    recipe_id = canonical_sha256({"zone": "FR-TEST", "revision": "r1"})
    recipe_build_id = canonical_sha256({"recipe": recipe_id, "revision": "r1"})
    request = {
        "service_url": "https://example.invalid/terrain",
        "layer": "FIXTURE-2M",
        "core_bounds_l93_m": bounds,
    }
    provenance = {
        role: {
            "schema": "fireviewer.tile-source-provenance.v1",
            "source_id": "tile-0-0",
            "role": role,
            "source_revision_id": "provider-r1",
            "license": "synthetic-test-fixture",
            "source_sha256": canonical_sha256({"role": role, "version": 1}),
            "source_byte_count": 10_000 + index,
            "request_sha256": canonical_sha256({**request, "layer": f"{role}-2M"}),
            "canonical": {
                "file_name": f"{role}-canonical-mm.npy",
                "file_sha256": canonical_sha256({"role": role, "canonical_version": 1}),
                "shape": [253, 253],
                "dtype": "<i4",
            },
        }
        for index, role in enumerate(("mnt", "mns"), start=1)
    }
    for role, payload in provenance.items():
        _write_json(source_root / f"{role}-provenance.v1.json", payload)

    original_hag = np.zeros((250, 250), dtype="uint16")
    original_hag[12, 34] = 250
    write_hag_max_2m(
        tile_root / "hag-max-2m.tif",
        original_hag,
        tile_origin_l93_m=bounds[:2],
    )
    surface_outputs = {
        "ground-profile-ids.png": b"profile-ids-r1",
        "ground-profile-weights.png": b"profile-weights-r1",
        "ground-confidence.png": b"confidence-r1",
        "ground-orientation.png": b"orientation-r1",
        "surface-correspondence.json": b"surface-correspondence-r1",
    }
    for name, content in surface_outputs.items():
        (tile_root / name).write_bytes(content)

    existing_receipt = {
        "recipe_id": recipe_id,
        "recipe_build_id": recipe_build_id,
    }
    backend = ProductionTerrainBackend()
    backend._assert_existing_tile_reusable(
        tile_root,
        tile_id=tile_id,
        existing_receipt=existing_receipt,
        recipe_id=recipe_id,
        recipe_build_id=recipe_build_id,
        source_provenance=provenance,
        hag_cm=original_hag,
        bounds_l93_m=bounds,
        surface_outputs=surface_outputs,
    )

    original_mns_provenance = (source_root / "mns-provenance.v1.json").read_bytes()
    original_hag_bytes = (tile_root / "hag-max-2m.tif").read_bytes()
    changed_provenance = {role: dict(payload) for role, payload in provenance.items()}
    changed_provenance["mns"]["source_sha256"] = canonical_sha256(
        {"role": "mns", "version": 2}
    )
    changed_provenance["mns"]["source_byte_count"] += 128
    changed_provenance["mns"]["canonical"] = {
        **changed_provenance["mns"]["canonical"],
        "file_sha256": canonical_sha256({"role": "mns", "canonical_version": 2}),
    }
    changed_hag = original_hag.copy()
    changed_hag[12, 34] = 375

    with pytest.raises(
        RuntimeError,
        match="existing package was preserved and a new zone revision/build is required",
    ):
        backend._assert_existing_tile_reusable(
            tile_root,
            tile_id=tile_id,
            existing_receipt=existing_receipt,
            recipe_id=recipe_id,
            recipe_build_id=recipe_build_id,
            source_provenance=changed_provenance,
            hag_cm=changed_hag,
            bounds_l93_m=bounds,
            surface_outputs=surface_outputs,
        )
    assert (
        source_root / "mns-provenance.v1.json"
    ).read_bytes() == original_mns_provenance
    assert (tile_root / "hag-max-2m.tif").read_bytes() == original_hag_bytes

    with pytest.raises(RuntimeError, match="hag_values"):
        backend._assert_existing_tile_reusable(
            tile_root,
            tile_id=tile_id,
            existing_receipt=existing_receipt,
            recipe_id=recipe_id,
            recipe_build_id=recipe_build_id,
            source_provenance=provenance,
            hag_cm=changed_hag,
            bounds_l93_m=bounds,
            surface_outputs=surface_outputs,
        )
    changed_surface = {
        **surface_outputs,
        "surface-correspondence.json": b"surface-correspondence-r2",
    }
    with pytest.raises(RuntimeError, match="surface:surface-correspondence.json"):
        backend._assert_existing_tile_reusable(
            tile_root,
            tile_id=tile_id,
            existing_receipt=existing_receipt,
            recipe_id=recipe_id,
            recipe_build_id=recipe_build_id,
            source_provenance=provenance,
            hag_cm=original_hag,
            bounds_l93_m=bounds,
            surface_outputs=changed_surface,
        )

    def final_identity(revision: str, mns_version: int) -> dict[str, object]:
        plan = {
            "source_requests": [
                {
                    "id": "tile-0-0",
                    "product": role,
                    "request": {**request, "layer": f"{role}-2M"},
                    "source_revision_id": revision,
                    "license": "synthetic-test-fixture",
                }
                for role in ("mnt", "mns")
            ],
            "tiles": [{"id": tile_id}],
        }
        observed = [
            {
                "id": item["id"],
                "product": item["product"],
                "request_sha256": canonical_sha256(item["request"]),
                "source_revision_id": item["source_revision_id"],
                "sha256": canonical_sha256(
                    {
                        "role": item["product"],
                        "version": mns_version if item["product"] == "mns" else 1,
                    }
                ),
                "byte_count": 10_128 if item["product"] == "mns" else 10_001,
                "license": item["license"],
            }
            for item in plan["source_requests"]
        ]
        return build_final_source_identity(
            plan,
            recipe_build_id=canonical_sha256(
                {"recipe": recipe_id, "source_revision": revision}
            ),
            observed_sources=observed,
            tile_receipts=[
                {
                    "tile_id": tile_id,
                    "tile_done_sha256": canonical_sha256(
                        {"revision": revision, "mns_version": mns_version}
                    ),
                }
            ],
        )

    first_build = final_identity("provider-r1", 1)
    second_build = final_identity("provider-r2", 2)
    assert first_build["build_id"] != second_build["build_id"]


def test_pilot_disk_gate_includes_retained_zone_packages(
    production_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = ProductionTerrainBackend(blender_runner=_fake_blender_runner)
    selected = [{"id": "pilot-0"}, {"id": "pilot-1"}]
    builds = [SimpleNamespace(receipt={"tile_id": tile["id"]}) for tile in selected]
    gib = 1024**3
    metrics = {
        "package_bytes": 10 * gib,
        "source_bytes": 2 * gib,
        "maximum_source_pair_bytes": 2 * gib,
    }
    monkeypatch.setattr(backend, "_select_pilot", lambda _context: selected)
    monkeypatch.setattr(
        backend,
        "_process_tiles",
        lambda _context, _selected: (builds, dict(metrics), [], []),
    )
    monkeypatch.setattr(backend, "_visual_reports", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(
        "terrain_production_backend.shutil.disk_usage",
        lambda _path: DiskUsage(total=256 * gib, used=156 * gib, free=100 * gib),
    )
    monkeypatch.setattr(backend, "_audit_c_since", lambda _started_ns: [])
    context = SimpleNamespace(
        max_tiles=2,
        plan={
            "tiles": [{"id": f"tile-{index}"} for index in range(50)],
            "seams": [],
        },
        export_root=production_workspace / "exports",
        zone=SimpleNamespace(
            zone_id="FR-TEST",
            revision="r1",
            work_root=production_workspace / "work",
        ),
    )

    with pytest.raises(StoragePolicyError, match="projected peak plus 20 GiB"):
        backend.pilot(context)


def test_concrete_backend_runs_one_zone_and_cleanup_is_idempotent(
    production_workspace: Path,
) -> None:
    zone_path = _write_production_zone(production_workspace)
    backend = ProductionTerrainBackend(
        opener=_fixture_opener,
        blender_runner=_fake_blender_runner,
        zone_visual_runner=_fake_zone_visual_runner,
    )
    controller = TerrainController(
        backends=backend.mapping(),
        coordinator_root=production_workspace / "global-coordinator",
        disk_usage=lambda _path: DiskUsage(
            total=256 * 1024**3,
            used=32 * 1024**3,
            free=224 * 1024**3,
        ),
    )

    for phase in ("plan", "preflight"):
        result = controller.run(zone_path, phase=phase, mode="execute")
        assert result["status"] == "completed"
    assert (
        controller.run(zone_path, phase="pilot", mode="execute", max_tiles=1)["status"]
        == "completed"
    )
    for phase in ("produce", "qa"):
        result = controller.run(zone_path, phase=phase, mode="execute")
        assert result["status"] == "completed"

    zone = load_zone_spec(zone_path)
    with pytest.raises(
        RuntimeError, match="explicit complete-zone visual review or acceptance"
    ):
        controller.run(zone_path, phase="accept", mode="execute")
    _record_zone_visual_acceptance(zone)
    assert (
        controller.run(zone_path, phase="accept", mode="resume")["status"]
        == "completed"
    )
    assert (
        controller.run(zone_path, phase="cleanup", mode="execute")["status"]
        == "completed"
    )

    package_root = zone.export_root / zone.zone_id / zone.revision
    tile_roots = list((package_root / "tiles").glob("*/tile.done.v3.json"))
    assert len(tile_roots) == 1
    tile_root = tile_roots[0].parent
    done = json.loads(tile_roots[0].read_text(encoding="utf-8"))
    package = json.loads(
        (tile_root / "tile-package.v3.json").read_text(encoding="utf-8")
    )
    assert done["schema"] == "fireviewer.tile.done.v3"
    assert package["schema"] == "fireviewer.tile-package.v3"
    assert done["inputs"]["surface_correspondence"]["path"] == (
        "surface-correspondence.json"
    )
    assert package["surface_mapping"] == done["surface_mapping"]
    assert package["ground_material"]["runtime_shader"]["status"] == (
        "pending_dedicated_mdl_validation"
    )
    assert (
        package["ground_material"]["runtime_shader"][
            "production_textured_runtime_qualified"
        ]
        is False
    )
    for name in (
        "ground-profile-ids.png",
        "ground-profile-weights.png",
        "ground-confidence.png",
        "ground-orientation.png",
        "surface-correspondence.json",
    ):
        assert (tile_root / name).is_file()
    assert not any(
        "ortho" in path.relative_to(tile_root).as_posix().casefold()
        for path in tile_root.rglob("*")
    )
    assert (package_root / "canonical-terrain-index.v1.json").is_file()
    visual_path = zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    assert visual["schema"] == ZONE_VISUAL_SCHEMA
    assert visual["status"] == "accepted_blender_visual"
    assert visual["automatic_acceptance"] is False
    assert visual["aov"]["invalid_lod_pixel_count"] == 0
    assert visual["aov"]["invalid_coverage_pixel_count"] == 0
    acceptance = json.loads(
        (zone.receipt_root / PHASE_RECEIPT_NAMES["accept"]).read_text(encoding="utf-8")
    )
    assert acceptance["runtime_shader"] == {
        "status": "pending_dedicated_mdl_validation"
    }
    assert acceptance["usd_runtime_gate"] is False
    assert not (zone.run_root / "sources").exists()
    assert not (zone.run_root / "temp").exists()
    assert not (zone.run_root / "cache").exists()
    assert not controller.active_zone_path.exists()

    repeated = controller.run(zone_path, phase="cleanup", mode="execute")
    assert repeated["status"] == "completed"
    assert not controller.active_zone_path.exists()
    assert (zone.receipt_root / PHASE_RECEIPT_NAMES["cleanup"]).is_file()
