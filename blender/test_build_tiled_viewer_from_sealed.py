from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import build_tiled_viewer_from_sealed as sealed_viewer

from build_tiled_viewer_from_sealed import (
    SealedPrototype,
    TiledViewerPackageError,
    _gltf_instance_record,
    _multiply,
    _quaternion_matrix,
    _scene_instances,
    _sealed_prototypes,
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


def test_shared_asset_identifier_is_scoped_by_family(tmp_path: Path) -> None:
    layout = {
        "prototype_count": 2,
        "prototypes": [
            {
                "family": "trees",
                "asset_id": "shared-asset",
                "identifier": "Asset_shared_asset",
                "identity_sha256": "a" * 64,
            },
            {
                "family": "context_assets",
                "asset_id": "shared-asset",
                "identifier": "Asset_shared_asset",
                "identity_sha256": "b" * 64,
            },
        ],
    }
    prototypes = _sealed_prototypes(layout)
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
  rel prototypes = [</MeasuredScene/Prototypes/Trees/Asset_shared_asset>]
  int64[] ids = [1]
  point3f[] positions = [(0, 0, 0)]
  float3[] scales = [(1, 1, 1)]
  quath[] orientations = [(1, 0, 0, 0)]
  int[] protoIndices = [0]
}
def PointInstancer "ContextAssets" {
  rel prototypes = [</MeasuredScene/Prototypes/ContextAssets/Asset_shared_asset>]
  int64[] ids = [2]
  point3f[] positions = [(1, 1, 1)]
  float3[] scales = [(1, 1, 1)]
  quath[] orientations = [(1, 0, 0, 0)]
  int[] protoIndices = [0]
}
''',
        encoding="utf-8",
    )
    transforms = {prototype.prototype_id: IDENTITY for prototype in prototypes}
    groups, counts = _scene_instances(
        scene,
        tile_id="x1000_y2000",
        tile_origin=(1000, 2000),
        zone_origin=(1000, 2000, 0),
        prototypes=prototypes,
        prototype_transforms=transforms,
    )
    assert counts == {"buildings": 0, "trees": 1, "context_assets": 1}
    assert len(groups["trees-0000"]) == 1
    assert len(groups["context_assets-0000"]) == 1


def test_duplicate_identifier_in_same_family_is_rejected() -> None:
    layout = {
        "prototype_count": 2,
        "prototypes": [
            {
                "family": "trees",
                "asset_id": "asset-a",
                "identifier": "Asset_collision",
                "identity_sha256": "a" * 64,
            },
            {
                "family": "trees",
                "asset_id": "asset-b",
                "identifier": "Asset_collision",
                "identity_sha256": "b" * 64,
            },
        ],
    }
    with pytest.raises(
        TiledViewerPackageError, match="Identité de prototype scellé invalide"
    ):
        _sealed_prototypes(layout)


def test_export_prototypes_propagates_blender_python_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blend = tmp_path / "zone.blend"
    blend.write_bytes(b"blend")
    blender = tmp_path / "blender"
    blender.write_bytes(b"binary")
    staging = tmp_path / ".viewer-tiled.part"
    staging.mkdir()
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        return SimpleNamespace(returncode=0, stdout="hidden traceback", stderr="")

    monkeypatch.setattr(sealed_viewer.subprocess, "run", fake_run)
    prototype = SealedPrototype(
        "trees-0000", "trees", "tree", "Asset_tree", "a" * 64
    )
    with pytest.raises(
        TiledViewerPackageError, match="sans produire le reçu prototypes"
    ) as error:
        sealed_viewer._export_prototypes(
            tmp_path, staging, blend, [prototype], blender, 30
        )
    assert "hidden traceback" in str(error.value)
    command = observed["command"]
    assert isinstance(command, list)
    assert command[command.index("--python-exit-code") + 1] == "1"
