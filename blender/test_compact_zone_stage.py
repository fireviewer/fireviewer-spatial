from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

import compact_zone_stage as compact
import simple_production_engine as production


def _scene_bytes(tile_id: str, wrapper_reference: str) -> bytes:
    return f'''#usda 1.0
(
    defaultPrim = "MeasuredScene"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "MeasuredScene"
{{
    custom string fireviewer:zone_id = "GPS-COMPACT"
    custom string fireviewer:tile_id = "{tile_id}"

    def Xform "Terrain" (
        prepend references = @../terrain-tile.usda@
    )
    {{
    }}

    def Scope "Prototypes"
    {{
        def Scope "Trees"
        {{
            def Xform "Asset_tree_oak"
            {{
                def Xform "Source" (
                    prepend references = @{wrapper_reference}@
                )
                {{
                    double3 xformOp:translate = (0, 1.25, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                }}
            }}
        }}
    }}

    def PointInstancer "Buildings"
    {{
        custom string fireviewer:category = "building"
        custom int fireviewer:count = 0
        rel prototypes = []
        int64[] ids = []
        point3f[] positions = []
        float3[] scales = []
        quath[] orientations = []
        int[] protoIndices = []
    }}

    def PointInstancer "Trees"
    {{
        custom string fireviewer:category = "tree"
        custom int fireviewer:count = 2
        rel prototypes = [
            </MeasuredScene/Prototypes/Trees/Asset_tree_oak>,
        ]
        int64[] ids = [1, 2]
        point3f[] positions = [(10, 20, 130), (30, 40, 132)]
        float3[] scales = [(1, 1, 1), (2, 2, 2)]
        quath[] orientations = [(1, 0, 0, 0), (1, 0, 0, 0)]
        int[] protoIndices = [0, 0]
    }}

    def PointInstancer "ContextAssets"
    {{
        custom string fireviewer:category = "context_asset"
        custom int fireviewer:count = 0
        rel prototypes = []
        int64[] ids = []
        point3f[] positions = []
        float3[] scales = []
        quath[] orientations = []
        int[] protoIndices = []
    }}
}}
'''.encode()


def _write_tile(
    root: Path,
    tile: compact.CompactTile,
    *,
    bundle_root: Path,
) -> None:
    package = root / "packages" / tile.tile_id
    scene_root = package / "scene"
    scene_root.mkdir(parents=True)
    (package / "terrain-tile.usda").write_text(
        '#usda 1.0\n(defaultPrim = "TerrainTile")\ndef Xform "TerrainTile" {}\n',
        encoding="utf-8",
    )
    wrapper = bundle_root / "tree-oak" / "prototype.usda"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        '#usda 1.0\n(defaultPrim = "Prototype")\ndef Xform "Prototype" {}\n',
        encoding="utf-8",
    )
    source = bundle_root / "tree-oak" / "source.usdc"
    source.write_bytes(b"compact-source-usdc")
    wrapper_relative = os.path.relpath(wrapper, scene_root).replace("\\", "/")
    scene = _scene_bytes(tile.tile_id, wrapper_relative)
    (scene_root / "scene.usda").write_bytes(scene)
    receipt = {
        "schema": compact.MEASURED_SCENE_SCHEMA,
        "zone_id": "GPS-COMPACT",
        "scene": {
            "path": "scene.usda",
            "byte_count": len(scene),
            "sha256": hashlib.sha256(scene).hexdigest(),
        },
        "terrain": {"root_reference": "../terrain-tile.usda"},
        "prototype_count": 1,
        "prototype_bundle": {
            "scope": "explicit_shared",
            "root_reference": os.path.relpath(bundle_root, scene_root).replace(
                "\\", "/"
            ),
        },
        "prototypes": [
            {
                "family": "trees",
                "asset_id": "tree-oak",
                "availability": "real_usd",
                "native_min_y": -1.25,
                "source_up_axis": "Y",
                "source_usd": {
                    "path": "tree-oak/source.usdc",
                    "byte_count": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
                "texture": None,
                "wrapper": {
                    "path": "tree-oak/prototype.usda",
                    "byte_count": wrapper.stat().st_size,
                    "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                },
                "material": {"implementation": "source_package_pbr"},
                "fallback_resolution": None,
            }
        ],
    }
    (scene_root / "scene.done.json").write_text(json.dumps(receipt), encoding="utf-8")


def _fixture_5x5(root: Path) -> tuple[compact.CompactTile, ...]:
    bundle = root / "shared" / "prototype-bundles" / "v1-fixture"
    tiles = tuple(
        compact.CompactTile(f"x{x}_y{y}", (x, y))
        for y in range(2000, -1, -500)
        for x in range(0, 2500, 500)
    )
    for tile in tiles:
        _write_tile(root, tile, bundle_root=bundle)
    return tiles


def test_compact_zone_defines_one_prototype_and_four_metatile_payloads(
    tmp_path: Path,
) -> None:
    tiles = _fixture_5x5(tmp_path)

    stage = compact.build_compact_zone_stage(
        tmp_path,
        zone_id="GPS-COMPACT",
        production_bounds_l93_m=(0, 0, 2500, 2500),
        tiles=tiles,
    )
    layout = compact.validate_compact_zone_stage(
        tmp_path, expected_tile_ids=[tile.tile_id for tile in tiles]
    )

    root_text = stage.read_text(encoding="utf-8")
    assert layout["tile_count"] == 25
    assert layout["payload_count"] == 4
    assert layout["prototype_count"] == 1
    assert root_text.count("prepend payload") == 4
    assert root_text.count("prepend references") == 1
    assert root_text.count("rel prototypes") == 25
    assert "packages/" not in root_text
    assert "/scene/scene.usda" not in root_text
    payload_text = "\n".join(
        (tmp_path / record["path"]).read_text(encoding="utf-8")
        for record in layout["payloads"]
    )
    assert payload_text.count('def PointInstancer "Trees"') == 25
    assert "</FireViewerZone/Prototypes/Trees/Asset_tree_oak>" not in payload_text
    assert 'def Scope "Prototypes"' not in payload_text
    assert "MeasuredScene/Prototypes" not in payload_text
    assert "rel prototypes" not in payload_text
    assert max(record["tile_count"] for record in layout["payloads"]) <= 16


def test_compact_zone_zip_contains_shared_prototype_files_once(
    tmp_path: Path,
) -> None:
    tiles = _fixture_5x5(tmp_path)
    compact.build_compact_zone_stage(
        tmp_path,
        zone_id="GPS-COMPACT",
        production_bounds_l93_m=(0, 0, 2500, 2500),
        tiles=tiles,
    )
    (tmp_path / "zone.blend").write_bytes(b"compressed-blend-fixture")
    production._write_archive_budget_receipt(tmp_path, 64 * 1024 * 1024)

    archive = production._write_zip(
        tmp_path,
        "GPS-COMPACT",
        maximum_bytes=64 * 1024 * 1024,
    )

    with zipfile.ZipFile(archive) as payload:
        names = payload.namelist()
        prefix = "fireviewer-GPS-COMPACT/"
        assert (
            names.count(
                prefix + "shared/prototype-bundles/v1-fixture/tree-oak/prototype.usda"
            )
            == 1
        )
        assert (
            names.count(
                prefix + "shared/prototype-bundles/v1-fixture/tree-oak/source.usdc"
            )
            == 1
        )
        assert (
            len([name for name in names if name.startswith(prefix + "payloads/")]) == 4
        )
        assert prefix + "zone.usda" in names
        assert prefix + "zone-stage-layout.v1.json" in names
        assert prefix + "archive-budget.v1.json" in names


def test_compact_zone_composes_global_prototype_targets_with_openusd(
    tmp_path: Path,
) -> None:
    usd = pytest.importorskip("pxr.Usd")
    usd_geom = pytest.importorskip("pxr.UsdGeom")
    tiles = _fixture_5x5(tmp_path)
    stage_path = compact.build_compact_zone_stage(
        tmp_path,
        zone_id="GPS-COMPACT",
        production_bounds_l93_m=(0, 0, 2500, 2500),
        tiles=tiles,
    )

    stage = usd.Stage.Open(str(stage_path))
    assert stage is not None
    populated = []
    for prim in stage.Traverse():
        if not prim.IsA(usd_geom.PointInstancer):
            continue
        positions = prim.GetAttribute("positions").Get() or []
        if not positions:
            continue
        targets = [
            str(path) for path in prim.GetRelationship("prototypes").GetTargets()
        ]
        populated.append((len(positions), targets))

    assert populated == [(2, ["/FireViewerZone/Prototypes/Trees/Asset_tree_oak"])] * 25


def test_compact_zone_rejects_a_tampered_tile_scene(tmp_path: Path) -> None:
    tiles = _fixture_5x5(tmp_path)
    scene = tmp_path / "packages" / tiles[0].tile_id / "scene" / "scene.usda"
    scene.write_bytes(scene.read_bytes() + b"# tampered\n")

    with pytest.raises(compact.CompactZoneStageError, match="altérée"):
        compact.build_compact_zone_stage(
            tmp_path,
            zone_id="GPS-COMPACT",
            production_bounds_l93_m=(0, 0, 2500, 2500),
            tiles=tiles,
        )


def test_compact_zone_rejects_a_divergent_prototype_receipt(tmp_path: Path) -> None:
    tiles = _fixture_5x5(tmp_path)
    receipt_path = (
        tmp_path / "packages" / tiles[-1].tile_id / "scene" / "scene.done.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["prototypes"][0]["native_min_y"] = -9.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(compact.CompactZoneStageError, match="divergent"):
        compact.build_compact_zone_stage(
            tmp_path,
            zone_id="GPS-COMPACT",
            production_bounds_l93_m=(0, 0, 2500, 2500),
            tiles=tiles,
        )
