from __future__ import annotations

import json
from pathlib import Path

import pytest

import map_builder
from compare_semantic_parity import compare

COMMIT = "766f157d00e15da72271ec197706c203f040fb7a"
DIGEST = "sha256:" + "7" * 64


def _job() -> dict:
    return {
        "schema": "fireviewer.map-job.v1",
        "build_id": "reference-v1",
        "zone_id": "GPS-0E12F428C04E6EEE",
        "center": {"lat": 44.7439034409, "lon": 5.3531898409},
        "side_m": 1500,
        "builder": {"git_commit": COMMIT, "image_digest": DIGEST},
        "profile": "factual-v2",
        "output": {"bucket": "logical-builds", "prefix": "maps/zone/build"},
    }


def test_normalize_job_translates_reference_contract() -> None:
    job = map_builder.normalize_job(_job())
    assert job["side_m"] == 1500
    assert job["center"] == {"lat": 44.7439034409, "lon": 5.3531898409}
    assert job["fixed_asset_placements"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("build_id", "../escape"), ("side_m", 1550)],
)
def test_normalize_job_rejects_unsafe_values(field: str, value: object) -> None:
    job = _job()
    job[field] = value
    with pytest.raises(map_builder.MapBuilderContractError):
        map_builder.normalize_job(job)


def test_publish_output_writes_zone_done_last(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for directory in ("viewer-tiled", "packages", "provenance"):
        (source / directory).mkdir(parents=True)
        (source / directory / "payload.bin").write_bytes(b"payload")
    for name in (
        "viewer.glb",
        "viewer-scene.v1.json",
        "zone.usda",
        "zone.blend",
        "zone.done.json",
        "manifest.json",
        "dependency-inventory.json",
        "zone-plan.json",
        "zone-context.json",
        "zone-stage-layout.v1.json",
        "validation-result.json",
    ):
        (source / name).write_text(json.dumps({"name": name}), encoding="utf-8")
    result = {
        "timings_seconds": {"sealed_map": 1.0, "viewer_export": 2.0, "total": 3.0},
        "counts": {"trees": 1, "buildings": 0},
        "tile_count": 1,
    }
    output = tmp_path / "output"
    map_builder.publish_output(source, output, _job(), result, {"ram_peak_gb": 1.0})
    assert (output / "zone.done.json").is_file()
    hashes = json.loads((output / "manifests" / "hashes.json").read_text())
    zone = next(item for item in hashes["artifacts"] if item["path"] == "zone.done.json")
    assert zone["publication_order"] == "last"
    assert not list(output.rglob("*.part"))


def test_semantic_comparator_rejects_count_drift(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "manifests").mkdir(parents=True)
    (output / "runtime").mkdir()
    baseline = {
        "zone_id": "GPS-TEST",
        "tile_count": 1,
        "counts": {"trees": 10},
        "spatial_reference": {"horizontal_crs": "EPSG:2154"},
        "viewer": {
            "representation": "complete_non_simplified_map",
            "mesh_coverage": "complete",
            "source_instance_count": 10,
            "external_dependencies": 0,
        },
    }
    (output / "manifests" / "validation-result.json").write_text(
        json.dumps(
            {
                "zone_id": "GPS-TEST",
                "tile_count": 1,
                "counts": {"trees": 9},
                "monolithic_viewer_build_oracle": {
                    "representation": "complete_non_simplified_map"
                },
            }
        ),
        encoding="utf-8",
    )
    (output / "manifests" / "manifest.json").write_text(
        json.dumps({"spatial_reference": {"horizontal_crs": "EPSG:2154"}}),
        encoding="utf-8",
    )
    (output / "manifests" / "hashes.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": "zone.done.json", "publication_order": "last"}
                ]
            }
        ),
        encoding="utf-8",
    )
    (output / "runtime" / "viewer-scene.v1.json").write_text(
        json.dumps(
            {
                "completeness": {
                    "mesh_coverage": "complete",
                    "source_instance_count": 10,
                },
                "viewer": {"external_dependencies": 0},
            }
        ),
        encoding="utf-8",
    )
    (output / "zone.done.json").write_text("{}", encoding="utf-8")
    assert compare(baseline, output)["status"] == "FAIL"
