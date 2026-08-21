from __future__ import annotations

from pathlib import Path

import pytest

from build_tiled_viewer_from_sealed import (
    SealedPrototype,
    _gltf_instance_record,
    _multiply,
    _quaternion_matrix,
    _scene_instances,
)


IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def test_prototype_transform_is_composed_into_instance() -> None:
    point_transform = _quaternion_matrix(
        (10.0, 20.0, 30.0),
        (2.0, 3.0, 4.0),
        (1.0, 0.0, 0.0, 0.0),
    )
    prototype_transform = (
        5.0, 0.0, 0.0, 1.0,
        0.0, 5.0, 0.0, 2.0,
        0.0, 0.0, 5.0, 3.0,
        0.0, 0.0, 0.0, 1.0,
    )
    record = _gltf_instance_record(
        _multiply(point_transform, prototype_transform)
    )
    assert record[:3] == pytest.approx((12.0, 42.0, -26.0))
    assert record[7:] == pytest.approx((10.0, 20.0, 15.0))


def test_scene_instances_preserve_sealed_tile_membership(tmp_path: Path) -> None:
    scene = tmp_path / "scene.usda"
    scene.write_text(
        '''#usda 1.0
def PointInstancer "Buildings" {
  rel prototypes = []
  int64[] ids = []
  point3f[] positions = []
  float3[] scales = []
  quath[] orientations = []
  int[] protoIndices = []
}
def PointInstancer "Trees" {
  rel prototypes = [</MeasuredScene/Prototypes/Trees/Asset_tree>]
  int64[] ids = [42]
  point3f[] positions = [(1, 2, 3)]
  float3[] scales = [(2, 3, 4)]
  quath[] orientations = [(1, 0, 0, 0)]
  int[] protoIndices = [0]
}
def PointInstancer "ContextAssets" {
  rel prototypes = []
  int64[] ids = []
  point3f[] positions = []
  float3[] scales = []
  quath[] orientations = []
  int[] protoIndices = []
}
''',
        encoding="utf-8",
    )
    prototype = SealedPrototype(
        "trees-0000", "trees", "tree", "Asset_tree", "a" * 64
    )
    groups, counts = _scene_instances(
        scene,
        tile_id="x1500_y2500",
        tile_origin=(1500, 2500),
        zone_origin=(1000, 2000, 0),
        prototypes=[prototype],
        prototype_transforms={prototype.prototype_id: IDENTITY},
    )
    assert counts == {"buildings": 0, "trees": 1, "context_assets": 0}
    assert groups[prototype.prototype_id][0] == pytest.approx(
        (501.0, 3.0, -502.0, 0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 3.0)
    )
