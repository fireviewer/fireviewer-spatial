from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fixed_terrain_grid import (
    compile_fixed_terrain_from_canonical_mm,
    write_fixed_terrain,
)
import portable_scene_package as portable
import render_simple_zone_gallery as gallery


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map_root(root: Path) -> Path:
    root.mkdir(parents=True)
    build_id = "a" * 64
    _write_json(
        root / "zone-plan.json",
        {
            "schema": "fireviewer.simple-measured-zone-plan.v1",
            "status": "planned",
            "zone_id": "GPS-PORTABLE-TEST",
            "crs": "EPSG:2154",
            "production_bounds_l93_m": [820000, 6312500, 820500, 6313000],
        },
    )
    _write_json(
        root / "zone.done.json",
        {
            "schema": "fireviewer.simple-measured-zone-production.v1",
            "status": "technical_scene_produced",
            "accepted_human": False,
            "zone_id": "GPS-PORTABLE-TEST",
            "build_id": build_id,
        },
    )
    (root / "zone.usda").write_text(
        '#usda 1.0\ndef Xform "Zone" {}\n', encoding="utf-8"
    )
    (root / "zone.blend").write_bytes(b"BLENDER-fixture")
    terrain_root = root / "packages" / "x820000_y6312500"
    terrain_root.mkdir(parents=True)
    terrain = compile_fixed_terrain_from_canonical_mm(
        np.full((253, 253), 125_000, dtype="<i4"),
        tile_origin_l93_m=(820000, 6312500),
    )
    write_fixed_terrain(terrain, terrain_root / "terrain.fvtg")

    gallery_root = root / gallery.GALLERY_DIRECTORY
    gallery_root.mkdir(parents=True)
    captures: list[dict[str, object]] = []
    for capture in gallery.build_capture_plan(500, 500):
        image = gallery_root / f"{capture['capture_id']}.png"
        image.write_bytes(b"PNG" + str(capture["capture_id"]).encode("ascii"))
        captures.append({**capture, "artifact": gallery._artifact(image, root)})
    receipt: dict[str, object] = {
        "schema": gallery.SCHEMA,
        "status": gallery.STATUS,
        "human_review_required": True,
        "accepted_human": False,
        "resolution": [gallery.RESOLUTION, gallery.RESOLUTION],
        "capture_count": gallery.CAPTURE_COUNT,
        "zone_stage": gallery._artifact(root / "zone.usda", root),
        "zone_plan": gallery._artifact(root / "zone-plan.json", root),
        "zone_receipt": gallery._artifact(root / "zone.done.json", root),
        "standalone_blend": gallery._artifact(root / "zone.blend", root),
        "scene_bounds_m": {"minimum": [0, 0, 125], "maximum": [500, 500, 125]},
        "instance_counts": {"buildings": 0, "trees": 12},
        "render_policy": dict(gallery.RENDER_POLICY),
        "captures": captures,
    }
    receipt["capture_set_sha256"] = hashlib.sha256(
        gallery._canonical_bytes(captures)
    ).hexdigest()
    receipt["receipt_content_sha256"] = hashlib.sha256(
        gallery._canonical_bytes(receipt)
    ).hexdigest()
    _write_json(root / gallery.RECEIPT_PATH, receipt)
    return root


def _perimeter_sources(root: Path, *, map_zone_id: str) -> tuple[Path, Path]:
    raw = root / "raw"
    viewer = root / "viewer"
    raw.mkdir(parents=True)
    viewer.mkdir(parents=True)
    layer_build_id = "b" * 64
    frame = {
        "index": 0,
        "frame_id": "observed-0000",
        "observed_at": "2026-08-10T12:00:00Z",
        "time_range": {
            "start": "2026-08-10T12:00:00Z",
            "end": "2026-08-10T12:00:00Z",
        },
        "affected": {"area_ha": 12.5},
        "active": {"area_ha": 3.5},
    }
    _write_json(
        raw / portable.PERIMETER_SOURCE_MANIFEST_NAME,
        {
            "schema": "fireviewer.geographic-perimeter-layer-package.v1",
            "dataset_id": "observed-test",
            "build_id": layer_build_id,
        },
    )
    _write_json(
        raw / portable.PERIMETER_TIMELINE_NAME,
        {
            "schema": "fireviewer.fire-progression-timeline.v1",
            "build_id": layer_build_id,
            "between_observations": "undefined",
            "prediction": "none",
            "frames": [frame],
        },
    )
    _write_json(raw / "perimeters.normalized.json", {"frames": [frame]})
    (raw / portable.PERIMETER_STAGE_NAME).write_text(
        '#usda 1.0\ndef Xform "ObservedPerimeters" {}\n', encoding="utf-8"
    )
    model = viewer / "frame-0000.glb"
    model.write_bytes(b"glTF-derived-observed-frame")
    _write_json(
        viewer / portable.PERIMETER_VIEWER_MANIFEST_NAME,
        {
            "schema": "fireviewer.geographic-perimeter-timeline-viewer.v1",
            "status": "derived_visual_timeline",
            "authoritative": False,
            "map_zone_id": map_zone_id,
            "layer_build_id": layer_build_id,
            "frame_count": 1,
            "between_observations": "undefined",
            "frames": [
                {
                    "index": 0,
                    "observed_at": frame["observed_at"],
                    "caption": "Observation 1",
                    "path": model.name,
                    "sha256": _sha256(model),
                    "byte_count": model.stat().st_size,
                }
            ],
        },
    )
    return raw, viewer


def test_map_and_observed_timeline_are_sealed_for_the_same_site_contract(
    tmp_path: Path,
) -> None:
    map_root = _map_root(tmp_path / "map")
    manifest = portable.seal_map_upload_package(map_root)
    assert manifest["control_gallery"]["capture_count"] == 20
    assert [item["group"] for item in manifest["control_gallery"]["captures"][:4]] == [
        "general"
    ] * 4
    assert [item["group"] for item in manifest["control_gallery"]["captures"][4:]] == [
        "detail"
    ] * 16

    archive = portable.write_deterministic_package_archive(
        map_root, tmp_path / "map-upload.zip"
    )
    reference = portable.read_map_reference_from_archive(archive)
    raw, viewer = _perimeter_sources(
        tmp_path / "perimeter", map_zone_id=reference.zone_id
    )
    work_root = tmp_path / "work"
    work_root.mkdir()
    upload_root, upload_archive, perimeter_manifest = (
        portable.materialize_perimeter_upload_package(
            raw,
            reference,
            work_root,
            viewer_root=viewer,
        )
    )

    assert upload_archive.is_file()
    assert perimeter_manifest["base_map"]["map_build_id"] == reference.map_build_id
    assert perimeter_manifest["visual_timeline"]["frame_count"] == 1
    assert perimeter_manifest["visual_timeline"]["authoritative"] is False
    assert portable.validate_perimeter_upload_package(upload_root) == perimeter_manifest

    (upload_root / "preview" / "frame-0000.glb").write_bytes(b"changed")
    with pytest.raises(portable.PortableScenePackageError, match="dependency changed"):
        portable.validate_perimeter_upload_package(upload_root)
