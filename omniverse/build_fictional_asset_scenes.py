"""Build fictional FireViewer worlds from reusable, high-fidelity USD assets.

The geographic terrain is retained as an anchor only.  Settlements, woodland
and water are intentionally fictional.  This builder has no primitive geometry
fallback: every visible non-terrain object is an authored NVIDIA USD asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from pxr import Gf, Usd, UsdGeom, UsdLux, Vt


FAMILIES = (
    ("forest_estate", 18, 65000, True),
    ("lake_settlement", 28, 85000, True),
    ("ridge_neighbourhood", 34, 74000, False),
    ("orchard_hamlet", 22, 96000, False),
    ("valley_community", 42, 115000, True),
)
BUILDINGS = (
    ("brownstone/Assets/Revit_Brownstone01/Revit_Brownstone01_Exterior.usd", "/World/Brownstone01"),
    ("brownstone/Assets/Revit_Brownstone02/Revit_Brownstone02_Exterior.usd", "/World/Brownstone02"),
    ("brownstone/Assets/Revit_Brownstone03/Revit_Brownstone03_Exterior.usd", "/World/Brownstone03"),
)
TREES = (
    ("city", "Assets/Vegetation/Black_Oak/Black_Oak.usd"),
    ("city", "Assets/Vegetation/Shumard_Oak/Shumard_Oak.usd"),
    ("city", "Assets/Vegetation/Common_Apple/Common_Apple.usd"),
    ("city", "Assets/Vegetation/Hawthorn/Hawthorn.usd"),
    ("world", "brownstone/Assets/Vegetation/Trees/American_Beech.usd"),
    ("world", "brownstone/Assets/Vegetation/Trees/Blue_Berry_Elder.usd"),
    ("world", "brownstone/Assets/Vegetation/Trees/Douglas_Fir.usd"),
    ("world", "brownstone/Assets/Vegetation/Trees/Honey_Locust.usd"),
    ("world", "brownstone/Assets/Vegetation/Trees/Largetooth_Aspen.usd"),
    ("world", "brownstone/Assets/Vegetation/Trees/Scarlet_Oak_fall.usd"),
    ("world", "brownstone/Assets/Vegetation/Trees/Service_Berry.usd"),
    ("world", "brownstone/Assets/Vegetation/Trees/White_Ash.usd"),
)
WATER = "sample-scenes/SubUSDs/Water_Mesh_v01.usd"
UNDERGROWTH = (
    "brownstone/Assets/Vegetation/Shrub/Grass_Short_A.usd",
    "brownstone/Assets/Vegetation/Shrub/Grass_Short_B.usd",
    "brownstone/Assets/Vegetation/Shrub/Meadowlark.usd",
)
STREETLIGHT = "brownstone/Props/StreetLight01/StreetLight01.usd"
BRIDGE = "sample-scenes/RT_bridge_rotating/RT_bridge_rotating.usd"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rng_for(package_id: str, ordinal: int) -> random.Random:
    raw = hashlib.sha256(f"fictional-high-fidelity:{package_id}:{ordinal}".encode()).digest()
    return random.Random(int.from_bytes(raw[:8], "little"))


def terrain_sampler(terrain_path: Path, manifest: dict[str, Any]):
    stage = Usd.Stage.Open(str(terrain_path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Terrain/GlobalMnt"))
    points = mesh.GetPointsAttr().Get()
    rows = int(manifest["terrain"]["mesh_grid"]["rows"])
    columns = int(manifest["terrain"]["mesh_grid"]["columns"])
    xmin, xmax = min(point[0] for point in points), max(point[0] for point in points)
    ymin, ymax = min(point[1] for point in points), max(point[1] for point in points)

    def at(x: float, y: float) -> float:
        col = max(0, min(columns - 1, round((x - xmin) / max(xmax - xmin, 1e-6) * (columns - 1))))
        row = max(0, min(rows - 1, round((ymax - y) / max(ymax - ymin, 1e-6) * (rows - 1))))
        return float(points[row * columns + col][2])

    return xmin, ymin, xmax, ymax, at


def reference_path(scene_path: Path, asset_path: Path) -> str:
    return os.path.relpath(asset_path, scene_path.parent).replace("\\", "/")


def add_reference(stage: Usd.Stage, path: str, asset: Path, scene_path: Path, position: tuple[float, float, float], scale: tuple[float, float, float], rotation_z: float = 0.0, payload: bool = False, rotation_x: float = 0.0, asset_prim: str | None = None) -> Usd.Prim:
    node = UsdGeom.Xform.Define(stage, path)
    xform = UsdGeom.Xformable(node)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateXOp().Set(rotation_x)
    xform.AddRotateZOp().Set(rotation_z)
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))
    asset_path = reference_path(scene_path, asset)
    if payload:
        if asset_prim:
            node.GetPrim().GetPayloads().AddPayload(asset_path, asset_prim)
        else:
            node.GetPrim().GetPayloads().AddPayload(asset_path)
    else:
        if asset_prim:
            node.GetPrim().GetReferences().AddReference(asset_path, asset_prim)
        else:
            node.GetPrim().GetReferences().AddReference(asset_path)
    return node.GetPrim()


def create_tree_instancer(stage: Usd.Stage, path: str, world_assets: Path, city_assets: Path, scene_path: Path, rng: random.Random, bounds: tuple[float, float, float, float], height_at, count: int) -> int:
    xmin, ymin, xmax, ymax = bounds
    prototypes = UsdGeom.Xform.Define(stage, path + "/Prototypes")
    targets = []
    for index, (source, relative) in enumerate(TREES):
        asset = (city_assets if source == "city" else world_assets) / relative
        if not asset.is_file():
            raise FileNotFoundError(asset)
        prototype = UsdGeom.Xform.Define(stage, f"{prototypes.GetPath()}/Tree{index + 1}")
        UsdGeom.Xformable(prototype).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
        prototype.GetPrim().GetReferences().AddReference(reference_path(scene_path, asset))
        targets.append(prototype.GetPath())
    instancer = UsdGeom.PointInstancer.Define(stage, path + "/Instances")
    instancer.CreatePrototypesRel().SetTargets(targets)
    positions, scales, indices, rotations = [], [], [], []
    # A few overlapping groves yield a continuous woodland and clearings instead
    # of a sparse, regular grid.  All prototypes are full authored tree assets.
    groves = [
        (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax), rng.uniform(0.055, 0.15))
        for _ in range(max(5, min(16, count // 450)))
    ]
    span = max(xmax - xmin, ymax - ymin)
    for _ in range(count):
        gx, gy, radius = rng.choice(groves)
        angle = rng.uniform(0, math.tau)
        distance = radius * span * math.sqrt(rng.random())
        x = max(xmin, min(xmax, gx + math.cos(angle) * distance))
        y = max(ymin, min(ymax, gy + math.sin(angle) * distance))
        scale = rng.uniform(0.75, 1.30)
        angle = math.radians(rng.uniform(0, 360))
        positions.append(Gf.Vec3f(x, y, height_at(x, y)))
        scales.append(Gf.Vec3f(scale, scale, scale))
        indices.append(rng.randrange(len(targets)))
        rotations.append(Gf.Quath(math.cos(angle / 2), Gf.Vec3h(0, 0, math.sin(angle / 2))))
    instancer.CreatePositionsAttr(Vt.Vec3fArray(positions))
    instancer.CreateScalesAttr(Vt.Vec3fArray(scales))
    instancer.CreateProtoIndicesAttr(Vt.IntArray(indices))
    instancer.CreateOrientationsAttr(Vt.QuathArray(rotations))
    return count


def create_undergrowth_instancer(stage: Usd.Stage, path: str, world_assets: Path, scene_path: Path, rng: random.Random, bounds: tuple[float, float, float, float], height_at, count: int) -> int:
    xmin, ymin, xmax, ymax = bounds
    prototypes = UsdGeom.Xform.Define(stage, path + "/Prototypes")
    targets = []
    for index, relative in enumerate(UNDERGROWTH):
        asset = world_assets / relative
        if not asset.is_file():
            raise FileNotFoundError(asset)
        prototype = UsdGeom.Xform.Define(stage, f"{prototypes.GetPath()}/GroundCover{index + 1}")
        UsdGeom.Xformable(prototype).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
        prototype.GetPrim().GetReferences().AddReference(reference_path(scene_path, asset))
        targets.append(prototype.GetPath())
    instancer = UsdGeom.PointInstancer.Define(stage, path + "/Instances")
    instancer.CreatePrototypesRel().SetTargets(targets)
    positions, scales, indices, rotations = [], [], [], []
    for _ in range(count):
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        scale = rng.uniform(0.65, 1.35)
        angle = math.radians(rng.uniform(0, 360))
        positions.append(Gf.Vec3f(x, y, height_at(x, y)))
        scales.append(Gf.Vec3f(scale, scale, scale))
        indices.append(rng.randrange(len(targets)))
        rotations.append(Gf.Quath(math.cos(angle / 2), Gf.Vec3h(0, 0, math.sin(angle / 2))))
    instancer.CreatePositionsAttr(Vt.Vec3fArray(positions))
    instancer.CreateScalesAttr(Vt.Vec3fArray(scales))
    instancer.CreateProtoIndicesAttr(Vt.IntArray(indices))
    instancer.CreateOrientationsAttr(Vt.QuathArray(rotations))
    return count


def scene_centres(family: str, bounds: tuple[float, float, float, float], rng: random.Random, count: int) -> list[tuple[float, float]]:
    xmin, ymin, xmax, ymax = bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    anchors = {
        "forest_estate": ((cx - (xmax - xmin) * 0.16, cy + (ymax - ymin) * 0.08),),
        "lake_settlement": ((cx - (xmax - xmin) * 0.16, cy - (ymax - ymin) * 0.18), (cx + (xmax - xmin) * 0.18, cy + (ymax - ymin) * 0.14)),
        "ridge_neighbourhood": ((cx, cy + (ymax - ymin) * 0.25),),
        "orchard_hamlet": ((cx - (xmax - xmin) * 0.22, cy),),
        "valley_community": ((cx - (xmax - xmin) * 0.20, cy - (ymax - ymin) * 0.12), (cx + (xmax - xmin) * 0.20, cy + (ymax - ymin) * 0.10)),
    }[family]
    result = []
    for index in range(count):
        ax, ay = anchors[index % len(anchors)]
        spread = 0.075 + 0.018 * (index % 3)
        result.append((ax + rng.uniform(-1, 1) * (xmax - xmin) * spread, ay + rng.uniform(-1, 1) * (ymax - ymin) * spread))
    return result


def composition_bounds(family: str, terrain_bounds: tuple[float, float, float, float], rng: random.Random) -> tuple[float, float, float, float]:
    """Select a dense, playable 2.4 km scene area inside the large terrain.

    The terrain remains available as a textured regional context.  Vegetation
    and architecture are intentionally authored at a human-readable density
    within the composition zone instead of being diluted over a 700–2,300 km²
    incident extent.
    """
    xmin, ymin, xmax, ymax = terrain_bounds
    span_x, span_y = xmax - xmin, ymax - ymin
    anchors = {
        "forest_estate": (0.24, 0.67),
        "lake_settlement": (0.64, 0.30),
        "ridge_neighbourhood": (0.50, 0.74),
        "orchard_hamlet": (0.27, 0.48),
        "valley_community": (0.70, 0.55),
    }
    fx, fy = anchors[family]
    cx = xmin + span_x * fx + rng.uniform(-0.035, 0.035) * span_x
    cy = ymin + span_y * fy + rng.uniform(-0.035, 0.035) * span_y
    half_x = min(1200.0, span_x * 0.14)
    half_y = min(1200.0, span_y * 0.14)
    cx = max(xmin + half_x, min(xmax - half_x, cx))
    cy = max(ymin + half_y, min(ymax - half_y, cy))
    return cx - half_x, cy - half_y, cx + half_x, cy + half_y


def build_scene(base_dir: Path, output: Path, ordinal: int, world_assets: Path, tree_assets: Path) -> dict[str, Any]:
    manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    package_id = str(manifest["package_id"])
    family, building_count, tree_count, water_enabled = FAMILIES[ordinal - 1]
    rng = rng_for(package_id, ordinal)
    xmin, ymin, xmax, ymax, height_at = terrain_sampler(base_dir / "terrain.usda", manifest)
    span_x, span_y = xmax - xmin, ymax - ymin
    bounds = composition_bounds(family, (xmin, ymin, xmax, ymax), rng)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths.append(str(Path("..").joinpath("..", "omniverse", base_dir.name, "terrain.usda")).replace("\\", "/"))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = UsdGeom.Xform.Define(stage, "/FictionalWorld")
    stage.SetDefaultPrim(root.GetPrim())
    for key, value in {
        "fireviewer:classification": "fictional_high_fidelity_scene",
        "fireviewer:base_terrain_package": package_id,
        "fireviewer:fiction_policy": "terrain_anchor_only; all settlement and landscape composition is fictional",
        "fireviewer:prohibited_fallback": "primitive_geometry",
        "fireviewer:family": family,
        "fireviewer:composition_extent_m": {"width": round(bounds[2] - bounds[0], 2), "height": round(bounds[3] - bounds[1], 2)},
    }.items():
        root.GetPrim().SetCustomDataByKey(key, value)
    near = UsdGeom.Xform.Define(stage, "/FictionalWorld/Near")
    mid = UsdGeom.Xform.Define(stage, "/FictionalWorld/Mid")
    far = UsdGeom.Xform.Define(stage, "/FictionalWorld/Far")
    building_assets = [(world_assets / relative, prim_path) for relative, prim_path in BUILDINGS]
    for asset, _ in building_assets:
        if not asset.is_file():
            raise FileNotFoundError(asset)
    for index, (x, y) in enumerate(scene_centres(family, bounds, rng, building_count)):
        z = height_at(x, y)
        asset, asset_prim = building_assets[index % len(building_assets)]
        rotation = rng.choice((0.0, 90.0, 180.0, 270.0)) + rng.uniform(-10, 10)
        add_reference(stage, f"{near.GetPath()}/Architecture/Residence{index + 1:02d}", asset, output, (x, y, z), (0.01, 0.01, 0.01), rotation, payload=True, asset_prim=asset_prim)
        if index % 2 == 0:
            add_reference(stage, f"{mid.GetPath()}/Architecture/Residence{index + 1:02d}", asset, output, (x, y, z), (0.01, 0.01, 0.01), rotation, payload=True, asset_prim=asset_prim)
        if index % 3 == 0:
            lamp = world_assets / STREETLIGHT
            add_reference(stage, f"{near.GetPath()}/StreetFurniture/Light{index + 1:02d}", lamp, output, (x + 12.0, y - 8.0, z), (0.01, 0.01, 0.01), rotation + 90.0, payload=True)
    near_trees = create_tree_instancer(stage, str(near.GetPath()) + "/Vegetation", world_assets, tree_assets, output, rng, bounds, height_at, tree_count)
    mid_trees = create_tree_instancer(stage, str(mid.GetPath()) + "/Vegetation", world_assets, tree_assets, output, rng, bounds, height_at, max(12000, int(tree_count * 0.70)))
    far_trees = create_tree_instancer(stage, str(far.GetPath()) + "/Vegetation", world_assets, tree_assets, output, rng, bounds, height_at, max(5000, int(tree_count * 0.25)))
    # Understory adds close-range richness without multiplying the stage payload
    # into the same order of magnitude as the tree canopy.
    near_ground_cover = create_undergrowth_instancer(stage, str(near.GetPath()) + "/GroundCover", world_assets, output, rng, bounds, height_at, max(2200, int(tree_count * 0.04)))
    mid_ground_cover = create_undergrowth_instancer(stage, str(mid.GetPath()) + "/GroundCover", world_assets, output, rng, bounds, height_at, max(900, int(tree_count * 0.012)))
    if water_enabled:
        asset = world_assets / WATER
        if not asset.is_file():
            raise FileNotFoundError(asset)
        wx = (bounds[0] + bounds[2]) / 2 + rng.uniform(-1, 1) * span_x * 0.10
        wy = (bounds[1] + bounds[3]) / 2 + rng.uniform(-1, 1) * span_y * 0.10
        water = add_reference(stage, f"{near.GetPath()}/Water/Feature", asset, output, (wx, wy, height_at(wx, wy) + 0.15), (0.16, 0.01, 0.07), rng.uniform(0, 180), payload=True, rotation_x=90.0)
        water.SetCustomDataByKey("fireviewer:fictional_water_feature", True)
        bridge_asset = world_assets / BRIDGE
        bridge = add_reference(stage, f"{near.GetPath()}/Water/Bridge", bridge_asset, output, (wx, wy, height_at(wx, wy) + 0.45), (0.38, 0.38, 0.38), rng.uniform(0, 180), payload=True, rotation_x=90.0)
        bridge.SetCustomDataByKey("fireviewer:fictional_bridge_asset", True)
    variants = root.GetPrim().GetVariantSets().AddVariantSet("lod")
    for selected, visible in (("far", far), ("mid", mid), ("near", near)):
        variants.AddVariant(selected)
        variants.SetVariantSelection(selected)
        with variants.GetVariantEditContext():
            for group in (far, mid, near):
                UsdGeom.Imageable(group).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited if group == visible else UsdGeom.Tokens.invisible)
    variants.SetVariantSelection("mid")
    root.GetPrim().SetCustomDataByKey("fireviewer:lod_ranges_m", {"near": [0, 140], "mid": [140, 1100], "far": [1100, 1000000]})
    sun = UsdLux.DistantLight.Define(stage, "/FictionalWorld/Sun")
    sun.CreateIntensityAttr(1900.0)
    sun.CreateAngleAttr(0.6)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(45.0, -30.0, 20.0 + ordinal * 9.0))
    sky = UsdLux.DomeLight.Define(stage, "/FictionalWorld/Sky")
    sky.CreateIntensityAttr(500.0)
    sky.CreateColorAttr(Gf.Vec3f(0.38, 0.52, 0.70))
    stage.GetRootLayer().Save()
    return {"schema": "fireviewer.fictional-world.v2", "scene_id": f"{package_id}-{ordinal:02d}", "base_package": package_id, "family": family, "stage": output.name, "classification": "fictional_high_fidelity_scene", "visible_assets": {"detailed_buildings": building_count, "tree_instances": {"far": far_trees, "mid": mid_trees, "near": near_trees}, "ground_cover_instances": {"mid": mid_ground_cover, "near": near_ground_cover}, "street_lights": (building_count + 2) // 3, "water_feature": int(water_enabled), "bridge": int(water_enabled)}, "lod": {"default": "mid", "variants": ["far", "mid", "near"]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--world-assets", type=Path, required=True)
    parser.add_argument("--tree-assets", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for base_dir in sorted(args.base_root.iterdir()):
        manifest_path = base_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "spatial_map":
            continue
        for ordinal in range(1, 6):
            output = args.output_root / str(manifest["package_id"]) / f"fiction-{ordinal:02d}.usda"
            records.append(build_scene(base_dir, output, ordinal, args.world_assets, args.tree_assets))
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "index.json", {"schema": "fireviewer.fictional-world-index.v1", "scene_count": len(records), "scenes": records})
    print(json.dumps({"scene_count": len(records), "classification": "fictional_high_fidelity_scene"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
