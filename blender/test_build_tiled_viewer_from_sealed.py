from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import build_tiled_viewer_from_sealed as sealed_viewer
from build_tiled_viewer_package import _FarGlb

from build_tiled_viewer_from_sealed import (
    CompiledTile,
    SealedPrototype,
    TiledViewerPackageError,
    _gltf_instance_record,
    _multiply,
    _load_cached_prototype,
    _load_cached_tile,
    _store_cached_prototype,
    _store_cached_tile,
    _quaternion_matrix,
    _scene_instances,
    _sealed_prototypes,
    _terrain_geometry,
    _viewer_tile_workers,
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
    prototype = SealedPrototype("trees-0000", "trees", "tree", "Asset_tree")
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
    prototype = SealedPrototype("trees-0000", "trees", "tree", "Asset_tree")
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


def test_vectorized_terrain_geometry_preserves_legacy_values() -> None:
    mesh = SimpleNamespace(
        grid_size=3,
        relative_heights_mm=(0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000),
        normals_snorm16=((0, 0, 32767),) * 9,
    )
    positions, normals, texcoords, indices = _terrain_geometry(
        SimpleNamespace(lods=(mesh,)), stride=1
    )
    np.testing.assert_allclose(
        positions,
        [
            (0, 0, 0), (250, 1, 0), (500, 2, 0),
            (0, 3, -250), (250, 4, -250), (500, 5, -250),
            (0, 6, -500), (250, 7, -500), (500, 8, -500),
        ],
    )
    np.testing.assert_allclose(normals, [(0, 1, 0)] * 9)
    np.testing.assert_allclose(
        texcoords,
        [
            (0, 1), (0.5, 1), (1, 1),
            (0, 0.5), (0.5, 0.5), (1, 0.5),
            (0, 0), (0.5, 0), (1, 0),
        ],
    )
    assert indices.tolist() == [
        0, 1, 4, 0, 4, 3,
        1, 2, 5, 1, 5, 4,
        3, 4, 7, 3, 7, 6,
        4, 5, 8, 4, 8, 7,
    ]


def test_viewer_tile_worker_count_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIREVIEWER_VIEWER_TILE_WORKERS", "8")
    assert _viewer_tile_workers(3) == 3
    monkeypatch.setenv("FIREVIEWER_VIEWER_TILE_WORKERS", "0")
    with pytest.raises(TiledViewerPackageError, match="compris entre 1 et 8"):
        _viewer_tile_workers(3)


def test_viewer_defaults_to_eight_local_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIREVIEWER_VIEWER_TILE_WORKERS", raising=False)
    monkeypatch.setattr(sealed_viewer.os, "cpu_count", lambda: 20)
    assert _viewer_tile_workers(441) == 8


def test_viewer_asset_metadata_does_not_hash_payload(tmp_path: Path) -> None:
    payload = tmp_path / "terrain.glb"
    payload.write_bytes(b"glb")
    asset = sealed_viewer._asset(payload, tmp_path, "model/gltf-binary")
    assert asset.payload() == {
        "path": "terrain.glb",
        "byte_count": 3,
        "media_type": "model/gltf-binary",
    }


def test_prototype_cache_round_trip_is_asset_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    staging = tmp_path / "first" / "prototypes"
    staging.mkdir(parents=True)
    source = staging / "trees-0000.glb"
    source.write_bytes(b"deterministic-glb")
    script = Path(sealed_viewer.__file__).with_name(
        "export_tiled_viewer_prototypes.py"
    )
    prototype = SealedPrototype("trees-0000", "trees", "tree", "Asset_tree")
    monkeypatch.setattr(sealed_viewer, "_validate_identity_prototype", lambda _path: 1)
    _store_cached_prototype(cache, prototype, script, source, IDENTITY)

    destination = tmp_path / "second" / "prototypes" / "trees-0000.glb"
    cached = _load_cached_prototype(
        cache, prototype, script, destination
    )
    assert cached is not None
    assert cached[0].path == "prototypes/trees-0000.glb"
    assert cached[1] == IDENTITY
    assert destination.read_bytes() == source.read_bytes()


def test_tile_cache_round_trip_restores_complete_compiled_tile(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first_staging = tmp_path / "first" / ".viewer-tiled.part"
    terrain = first_staging / "tiles" / "x0_y0" / "terrain.glb"
    instances = first_staging / "tiles" / "x0_y0" / "instances.fvi"
    terrain.parent.mkdir(parents=True)
    terrain.write_bytes(b"terrain-glb")
    instances.write_bytes(b"instances-fvi")
    compiled = CompiledTile(
        tile_id="x0_y0",
        tile_origin=(0, 0),
        terrain_asset=sealed_viewer._asset(
            terrain, first_staging, "model/gltf-binary"
        ),
        instance_asset=sealed_viewer._asset(
            instances,
            first_staging,
            "application/vnd.fireviewer.instances",
        ),
        family_counts={"buildings": 1, "trees": 2, "context_assets": 0},
        prototype_instance_counts={"buildings-0000": 1, "trees-0000": 2},
        prototype_ids=("buildings-0000", "trees-0000"),
        far_positions=np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32),
        far_normals=np.asarray(((0.0, 1.0, 0.0),), dtype=np.float32),
        far_texcoords=np.asarray(((0.0, 0.0),), dtype=np.float32),
        far_indices=np.asarray((0,), dtype=np.uint32),
        far_image=b"jpeg-thumbnail",
        node_chain=({"name": "Tile_x0_y0"},),
        timings_seconds={"total": 1.0},
        cache_hit=False,
    )
    identity = {"scene_sha256": "a" * 64}
    _store_cached_tile(
        cache,
        cache_key="b" * 64,
        source_identity=identity,
        staging=first_staging,
        compiled=compiled,
    )

    second_staging = tmp_path / "second" / ".viewer-tiled.part"
    restored = _load_cached_tile(
        cache,
        cache_key="b" * 64,
        tile_id="x0_y0",
        source_identity=identity,
        staging=second_staging,
    )
    assert restored is not None
    assert restored.cache_hit is True
    assert restored.family_counts == compiled.family_counts
    np.testing.assert_array_equal(restored.far_positions, compiled.far_positions)
    assert (second_staging / restored.terrain_asset.path).read_bytes() == b"terrain-glb"
    assert (second_staging / restored.instance_asset.path).read_bytes() == b"instances-fvi"


def test_tile_cache_ignores_structurally_broken_far_fragment(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    staging = tmp_path / "first" / ".viewer-tiled.part"
    terrain = staging / "tiles" / "x0_y0" / "terrain.glb"
    instances = staging / "tiles" / "x0_y0" / "instances.fvi"
    terrain.parent.mkdir(parents=True)
    terrain.write_bytes(b"terrain")
    instances.write_bytes(b"instances")
    compiled = CompiledTile(
        "x0_y0", (0, 0),
        sealed_viewer._asset(terrain, staging, "model/gltf-binary"),
        sealed_viewer._asset(
            instances, staging, "application/vnd.fireviewer.instances"
        ),
        {"buildings": 0, "trees": 0, "context_assets": 0},
        {}, (),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1,), dtype=np.uint32),
        b"jpeg", ({"name": "Tile_x0_y0"},), {"total": 1.0}, False,
    )
    identity = {"scene_sha256": "c" * 64}
    _store_cached_tile(
        cache,
        cache_key="d" * 64,
        source_identity=identity,
        staging=staging,
        compiled=compiled,
    )
    (cache / ("d" * 64) / "far.npz").write_bytes(b"not-an-npz")
    assert _load_cached_tile(
        cache,
        cache_key="d" * 64,
        tile_id="x0_y0",
        source_identity=identity,
        staging=tmp_path / "second" / ".viewer-tiled.part",
    ) is None


def test_far_glb_packs_numpy_without_python_scalar_expansion() -> None:
    builder = _FarGlb(generator="test")
    accessor = builder.accessor(
        np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float32),
        component_type=5126,
        kind="VEC3",
        target=34962,
    )
    record = builder.gltf["accessors"][accessor]
    assert record["count"] == 2
    assert record["min"] == [1.0, 2.0, 3.0]
    assert record["max"] == [4.0, 5.0, 6.0]
    assert len(builder.binary) == 24


def test_far_glb_disk_spool_is_byte_identical(tmp_path: Path) -> None:
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    normals = np.asarray(((0.0, 1.0, 0.0),) * 3, dtype=np.float32)
    texcoords = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    indices = np.asarray((0, 1, 2), dtype=np.uint16)
    chain = ({"name": "Tile_x0_y0", "translation": [0, 0, 0]},)
    memory = _FarGlb(generator="parity")
    spooled = _FarGlb(generator="parity", binary_spool=tmp_path / "far.bin")
    for builder in (memory, spooled):
        builder.add_tile(
            tile_id="x0_y0",
            node_chain=chain,
            positions=positions,
            normals=normals,
            texcoords=texcoords,
            indices=indices,
            image=b"jpeg",
        )
    memory_path = tmp_path / "memory.glb"
    spool_path = tmp_path / "spooled.glb"
    memory.write(memory_path)
    spooled.write(spool_path)
    assert spool_path.read_bytes() == memory_path.read_bytes()
    assert not (tmp_path / "far.bin").exists()
