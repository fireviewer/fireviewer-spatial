from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from blender.frustum_streaming import (  # noqa: E402
    Aabb3D,
    CameraEnvelope,
    CameraView,
    FrustumStreamingPlanner,
    ResourceCost,
    StreamingBudget,
    TerrainTile,
    TerrainTileCatalog,
    plan_camera_sequence,
)


CONTRACT_ROOT = Path(__file__).parent / "contracts" / "terrain" / "v1"


def _schema(name: str) -> dict:
    return json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))


def _camera() -> CameraEnvelope:
    return CameraEnvelope(
        camera_id="CAM_SCHEMA",
        views=(
            CameraView(
                view_id="wide",
                position_l93_ngf_m=(700_250.0, 6_300_250.0, 600.0),
                forward=(0.0, 0.0, -1.0),
                up=(0.0, 1.0, 0.0),
                vertical_fov_deg=50.0,
                aspect_ratio=16.0 / 9.0,
                near_clip_m=0.1,
                far_clip_m=2_000.0,
            ),
        ),
    )


def _catalog() -> TerrainTileCatalog:
    costs = (
        ResourceCost(10_000, 20_000, 5_000),
        ResourceCost(2_000, 4_000, 1_000),
        ResourceCost(500, 1_000, 200),
    )
    return TerrainTileCatalog(
        TerrainTile(
            tile_id=f"x{700_000 + x * 500}_y{6_300_000 + y * 500}",
            grid_x=1_400 + x,
            grid_y=12_600 + y,
            bounds=Aabb3D(
                minimum=(700_000.0 + x * 500, 6_300_000.0 + y * 500, 80.0),
                maximum=(700_500.0 + x * 500, 6_300_500.0 + y * 500, 180.0),
            ),
            costs=costs,
            build_id="d" * 64,
            payload_sha256=("a" * 64, "b" * 64, "c" * 64),
            stitch_masks=tuple(range(16)),
            stitch_triangle_counts=tuple(
                tuple(cost.triangles for _mask in range(16)) for cost in costs
            ),
        )
        for x in range(2)
        for y in range(2)
    )


def test_all_public_terrain_schemas_are_valid_draft_202012() -> None:
    for path in sorted(CONTRACT_ROOT.glob("*.schema.json")):
        jsonschema.Draft202012Validator.check_schema(
            json.loads(path.read_text(encoding="utf-8"))
        )


def test_zone_acceptance_keeps_usd_runtime_shader_gate_pending() -> None:
    acceptance = {
        "schema": "fireviewer.zone.acceptance.v1",
        "recipe_id": "a" * 64,
        "recipe_build_id": "b" * 64,
        "build_id": "c" * 64,
        "source_merkle_root_sha256": "d" * 64,
        "status": "accepted",
        "tile_count": 1,
        "seam_count": 0,
        "qa_receipt": "qa.receipt.v1.json",
        "qa_receipt_sha256": "e" * 64,
        "visual_receipt": "zone-visual.accepted_blender_visual.v2.json",
        "visual_receipt_sha256": "f" * 64,
        "runtime_shader": {"status": "pending_dedicated_mdl_validation"},
        "usd_runtime_gate": False,
    }
    schema = _schema("zone-acceptance.schema.json")
    jsonschema.validate(acceptance, schema)
    acceptance["usd_runtime_gate"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(acceptance, schema)


def test_camera_plan_and_streaming_state_match_their_public_schemas() -> None:
    camera = _camera()
    catalog = _catalog()
    budget = StreamingBudget(
        cpu_bytes=1_000_000,
        gpu_bytes=1_000_000,
        maximum_triangles=1_000_000,
    )
    sequence = plan_camera_sequence(catalog, (camera,), budget)
    planner = FrustumStreamingPlanner(catalog, budget)
    planner.stage_camera(camera, now_s=0.0)

    jsonschema.validate(camera.to_dict(), _schema("camera-envelope.schema.json"))
    jsonschema.validate(
        sequence.to_dict(), _schema("camera-residency-plan.schema.json")
    )
    jsonschema.validate(
        planner.state.to_dict(), _schema("terrain-streaming-state.schema.json")
    )


def test_camera_contracts_reject_under_reserved_or_unverified_states() -> None:
    camera = _camera()
    catalog = _catalog()
    budget = StreamingBudget(cpu_bytes=1_000_000, gpu_bytes=1_000_000)
    sequence = plan_camera_sequence(catalog, (camera,), budget).to_dict()
    plan_schema = _schema("camera-residency-plan.schema.json")

    under_reserved = json.loads(json.dumps(sequence))
    under_reserved["entries"][0]["budget"]["reserve_fraction"] = 0.249
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(under_reserved, plan_schema)

    failed_budget = json.loads(json.dumps(sequence))
    failed_budget["entries"][0]["budget"]["within_budget"] = False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(failed_budget, plan_schema)

    missing_plan_mask = json.loads(json.dumps(sequence))
    del missing_plan_mask["entries"][0]["payloads"][0]["stitch_mask"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_plan_mask, plan_schema)

    missing_envelope_hash = json.loads(json.dumps(sequence))
    del missing_envelope_hash["camera_envelope_bindings"][0]["sha256"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_envelope_hash, plan_schema)

    planner = FrustumStreamingPlanner(catalog, budget)
    planner.stage_camera(camera, now_s=0.0)
    planner.tick(now_s=0.0)
    state = planner.state.to_dict()
    state_schema = _schema("terrain-streaming-state.schema.json")
    jsonschema.validate(state, state_schema)
    state["pending_action"]["expected_sha256"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(state, state_schema)

    missing_state_mask = planner.state.to_dict()
    del missing_state_mask["pending_action"]["payloads"][0]["stitch_mask"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_state_mask, state_schema)


def test_canonical_tile_package_matches_tile_package_v2_schema() -> None:
    digest = "a" * 64
    artifact = {"path": "relative/artifact.bin", "byte_count": 10, "sha256": digest}
    payload = {
        "schema": "fireviewer.tile-package.v2",
        "tile_id": "x700000_y6300000",
        "recipe_id": "b" * 64,
        "recipe_build_id": "c" * 64,
        "crs": "EPSG:2154",
        "bounds_l93_m": [700_000, 6_300_000, 700_500, 6_300_500],
        "normal_halo_sha256": "2" * 64,
        "stitch_variants": {
            "encoding": "fvtq-base-remove-add.v1",
            "edge_order": ["west", "east", "south", "north"],
            "edge_mask_bits": {"west": 1, "east": 2, "south": 4, "north": 8},
            "available_masks": list(range(16)),
            "lods": {
                f"lod{lod}": [
                    {
                        "mask": mask,
                        "triangle_count": 128 >> lod,
                        "triangle_indices_sha256": f"{lod + 3:x}" * 64,
                        "maximum_error_mm": (500, 2_000, 8_000)[lod],
                        "effective_edge_signatures": [
                            f"{edge + 6:x}" * 64 for edge in range(4)
                        ],
                    }
                    for mask in range(16)
                ]
                for lod in range(3)
            },
        },
        "inputs": {"mnt": artifact, "mns": artifact},
        "ground_material": {
            "schema": "fireviewer.ground-material-contract.v1",
            "zone_path": "shared/ground-material/ground-material-contract.v1.json",
            "contract_sha256": "a" * 64,
            "source_atlas_catalog_sha256": "2" * 64,
            "atlas_catalog_sha256": "b" * 64,
            "runtime_atlas_sha256": {
                "basecolor": "c" * 64,
                "normal": "d" * 64,
                "height": "e" * 64,
                "orm": "f" * 64,
            },
            "material_layer_sha256": "1" * 64,
            "visual_acceptance": "pending_human_review",
        },
        "outputs": {
            name: artifact
            for name in (
                "terrain_lod0",
                "terrain_lod1",
                "terrain_lod2",
                "hag_max_2m",
                "ground_profile_ids",
                "ground_profile_weights",
                "surface_overlays",
                "tile_composition",
            )
        },
    }
    jsonschema.validate(payload, _schema("tile-package.schema.json"))
