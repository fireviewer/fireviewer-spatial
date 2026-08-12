from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

from PIL import Image
import pytest

import render_simple_zone_gallery as gallery


TEST_ROOT = (
    Path(
        os.environ.get(
            "FIREVIEWER_TEST_ROOT",
            "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest",
        )
    )
    / "simple-zone-gallery"
)


@pytest.fixture
def zone_root() -> Path:
    root = TEST_ROOT / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.resolve().is_relative_to(TEST_ROOT.resolve()):
            shutil.rmtree(root, ignore_errors=True)


def _png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (512, 512), (90, 120, 70)).save(stream, format="PNG")
    return stream.getvalue()


def _sealed_gallery(root: Path) -> dict:
    (root / "zone.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (root / "zone-plan.json").write_text(
        json.dumps({"production_bounds_l93_m": [0, 0, 500, 500]}),
        encoding="utf-8",
    )
    (root / "zone.done.json").write_text(
        json.dumps({"building_count": 2, "tree_count": 7}), encoding="utf-8"
    )
    (root / "zone.blend").write_bytes(b"packed-blend")
    records = []
    output = root / gallery.GALLERY_DIRECTORY
    output.mkdir(parents=True)
    plan = gallery._with_building_focus(
        gallery.build_capture_plan(500, 500),
        [(125.0, 375.0, 101.0, 12.0), (300.0, 200.0, 100.0, 3.0)],
    )
    for capture in plan:
        path = output / f"{capture['capture_id']}.png"
        path.write_bytes(_png())
        records.append({**capture, "artifact": gallery._artifact(path, root)})
    receipt = {
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
        "scene_bounds_m": {"minimum": [0, 0, 100], "maximum": [500, 500, 140]},
        "instance_counts": {"buildings": 2, "trees": 7},
        "render_policy": dict(gallery.RENDER_POLICY),
        "captures": records,
    }
    receipt["capture_set_sha256"] = gallery.hashlib.sha256(
        gallery._canonical_bytes(records)
    ).hexdigest()
    receipt["receipt_content_sha256"] = gallery.hashlib.sha256(
        gallery._canonical_bytes(receipt)
    ).hexdigest()
    gallery._write_json(root / gallery.RECEIPT_PATH, receipt)
    return receipt


def test_capture_plan_is_exactly_four_overviews_plus_complete_four_by_four() -> None:
    plan = gallery.build_capture_plan(1500, 1000)
    assert len(plan) == 20
    assert [item["category"] for item in plan].count("overview") == 4
    details = [item for item in plan if item["category"] == "detail_coverage"]
    assert {tuple(item["grid_cell"]) for item in details} == {
        (row, column) for row in range(4) for column in range(4)
    }


def test_building_focus_replaces_one_oblique_without_losing_grid_coverage() -> None:
    base = gallery.build_capture_plan(1500, 1000)
    focused = gallery._with_building_focus(
        base,
        [(900.0, 700.0, 118.0, 2.0), (125.0, 375.0, 121.0, 8.0)],
    )

    assert len(focused) == 20
    assert [item["capture_id"] for item in focused] == [
        item["capture_id"] for item in base
    ]
    proof = next(item for item in focused if item["category"] == "building_detail")
    assert proof["capture_id"] == gallery.BUILDING_FOCUS_CAPTURE_ID
    assert proof["center_xy_m"] == [125.0, 375.0]
    assert proof["target_z_m"] == 127.0
    assert [item["category"] for item in focused].count("overview") == 3
    assert {
        tuple(item["grid_cell"])
        for item in focused
        if item["category"] == "detail_coverage"
    } == {(row, column) for row in range(4) for column in range(4)}
    assert gallery._with_building_focus(base, []) == base


def test_verify_rehashes_blend_zone_receipt_and_all_twenty_images(
    zone_root: Path,
) -> None:
    expected = _sealed_gallery(zone_root)
    actual = gallery.verify_gallery(zone_root)
    assert actual == expected
    assert len(actual["captures"]) == 20

    first = zone_root / actual["captures"][0]["artifact"]["path"]
    first.write_bytes(first.read_bytes() + b"tampered")
    with pytest.raises(gallery.SimpleZoneGalleryError, match="missing or corrupted"):
        gallery.verify_gallery(zone_root)


def test_tree_instance_gate_rejects_non_uniform_or_non_upright_scales() -> None:
    def fake_point_cloud(
        scale: tuple[float, float, float], quaternion: tuple[float, ...]
    ):
        attributes = {
            "scale": SimpleNamespace(data=[SimpleNamespace(vector=scale)]),
            "orientation": SimpleNamespace(data=[SimpleNamespace(value=quaternion)]),
        }
        data = SimpleNamespace(
            points=[object()], attributes=SimpleNamespace(get=attributes.get)
        )
        return SimpleNamespace(type="POINTCLOUD", name="Trees", data=data)

    upright = fake_point_cloud((3.0, 3.0, 3.0), (2**-0.5, 2**-0.5, 0, 0))
    assert gallery._validate_measured_instances(
        SimpleNamespace(
            context=SimpleNamespace(scene=SimpleNamespace(objects=[upright]))
        )
    ) == {"trees": 1, "buildings": 0}

    warped = fake_point_cloud((3.0, 0.7, 4.5), (2**-0.5, 2**-0.5, 0, 0))
    with pytest.raises(gallery.SimpleZoneGalleryError, match="non-uniform"):
        gallery._validate_measured_instances(
            SimpleNamespace(
                context=SimpleNamespace(scene=SimpleNamespace(objects=[warped]))
            )
        )

    tilted = fake_point_cloud((3.0, 3.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(gallery.SimpleZoneGalleryError, match="not upright"):
        gallery._validate_measured_instances(
            SimpleNamespace(
                context=SimpleNamespace(scene=SimpleNamespace(objects=[tilted]))
            )
        )
