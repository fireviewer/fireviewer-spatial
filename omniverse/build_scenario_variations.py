"""Generate lightweight but materially credible synthetic OpenUSD scenarios.

Each scene sublayers an immutable FireViewer spatial-map wrapper.  The rural
settlement, vegetation and infrastructure authored here are *scenario
variations*, never a claim about the measured state of the incident area.
Near LOD uses reusable NVIDIA tree assets and detailed parametric buildings;
mid and far LODs deliberately reduce geometry while retaining composition.
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

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt


FAMILIES = (
    ("ridge_hamlet", 7, 145, 0.0, 0.0),
    ("riparian_grove", 4, 220, 0.55, 0.22),
    ("firebreak_estate", 10, 110, -0.25, 0.68),
    ("wetland_edge", 5, 185, 0.90, 0.42),
    ("dispersed_hamlet", 12, 170, -0.45, 0.12),
)
TREE_ASSETS = (
    "Assets/Vegetation/Black_Oak/Black_Oak.usd",
    "Assets/Vegetation/Shumard_Oak/Shumard_Oak.usd",
    "Assets/Vegetation/Common_Apple/Common_Apple.usd",
    "Assets/Vegetation/Hawthorn/Hawthorn.usd",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seeded(package_id: str, ordinal: int) -> random.Random:
    digest = hashlib.sha256(f"{package_id}:{ordinal}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def material(stage: Usd.Stage, path: str, color: tuple[float, float, float], roughness: float, metallic: float = 0.0) -> UsdShade.Material:
    value = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    value.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return value


def add_cube(stage: Usd.Stage, path: str, position: tuple[float, float, float], scale: tuple[float, float, float], binding: UsdShade.Material) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdShade.MaterialBindingAPI(cube).Bind(binding)
    return cube


def add_cylinder(stage: Usd.Stage, path: str, position: tuple[float, float, float], radius: float, height: float, binding: UsdShade.Material) -> UsdGeom.Cylinder:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdShade.MaterialBindingAPI(cylinder).Bind(binding)
    return cylinder


def add_gable_roof(stage: Usd.Stage, path: str, centre: tuple[float, float, float], width: float, depth: float, rise: float, binding: UsdShade.Material) -> UsdGeom.Mesh:
    """A true pitched roof instead of a scaled box proxy."""
    x, y, z = centre
    half_width, half_depth = width / 2.0, depth / 2.0
    points = (
        Gf.Vec3f(x - half_width, y - half_depth, z),
        Gf.Vec3f(x + half_width, y - half_depth, z),
        Gf.Vec3f(x + half_width, y + half_depth, z),
        Gf.Vec3f(x - half_width, y + half_depth, z),
        Gf.Vec3f(x, y - half_depth, z + rise),
        Gf.Vec3f(x, y + half_depth, z + rise),
    )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4, 4, 3, 3, 4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 4, 5, 3, 1, 2, 5, 4, 0, 1, 4, 3, 5, 2, 0, 3, 2, 1]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI(mesh).Bind(binding)
    return mesh


def set_visibility(prim: Usd.Prim, value: str) -> None:
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(value)


def terrain_sampler(terrain_path: Path, manifest: dict[str, Any]):
    stage = Usd.Stage.Open(str(terrain_path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Terrain/GlobalMnt"))
    points = mesh.GetPointsAttr().Get()
    rows = int(manifest["terrain"]["mesh_grid"]["rows"])
    columns = int(manifest["terrain"]["mesh_grid"]["columns"])
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def at(x: float, y: float) -> float:
        col = max(0, min(columns - 1, round((x - xmin) / max(1e-6, xmax - xmin) * (columns - 1))))
        row = max(0, min(rows - 1, round((ymax - y) / max(1e-6, ymax - ymin) * (rows - 1))))
        return float(points[row * columns + col][2])

    return xmin, ymin, xmax, ymax, at


def tree_reference_paths(asset_root: Path | None, stage_path: Path) -> list[str]:
    if asset_root is None:
        return []
    absolute = [asset_root / relative for relative in TREE_ASSETS]
    if not all(path.is_file() for path in absolute):
        return []
    return [os.path.relpath(path, stage_path.parent).replace("\\", "/") for path in absolute]


def create_tree_prototypes(stage: Usd.Stage, root: str, references: list[str], trunk: UsdShade.Material, foliage: UsdShade.Material) -> tuple[list[Sdf.Path], str]:
    prototypes = UsdGeom.Xform.Define(stage, root + "/Prototypes")
    paths: list[Sdf.Path] = []
    if references:
        for index, reference in enumerate(references):
            tree = UsdGeom.Xform.Define(stage, f"{prototypes.GetPath()}/Tree{index + 1}")
            tree.GetPrim().GetReferences().AddReference(reference)
            paths.append(tree.GetPath())
        return paths, "NVIDIA_AECO_realistic_tree_USD"
    for index, (trunk_height, crown_height, radius) in enumerate(((4.0, 10.0, 3.0), (4.8, 12.5, 4.0), (3.0, 7.5, 2.5))):
        tree = UsdGeom.Xform.Define(stage, f"{prototypes.GetPath()}/ProxyTree{index + 1}")
        add_cylinder(stage, f"{tree.GetPath()}/Trunk", (0.0, 0.0, trunk_height / 2.0), radius * 0.15, trunk_height, trunk)
        crown = UsdGeom.Cone.Define(stage, f"{tree.GetPath()}/Crown")
        crown.CreateHeightAttr(crown_height)
        crown.CreateRadiusAttr(radius)
        UsdGeom.Xformable(crown).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, trunk_height + crown_height / 2.0))
        UsdShade.MaterialBindingAPI(crown).Bind(foliage)
        paths.append(tree.GetPath())
    return paths, "proxy_tree_fallback"


def create_vegetation_instancer(stage: Usd.Stage, root: str, rng: random.Random, count: int, bounds: tuple[float, float, float, float], height_at, references: list[str], trunk: UsdShade.Material, foliage: UsdShade.Material) -> tuple[int, str]:
    xmin, ymin, xmax, ymax = bounds
    prototype_paths, asset_mode = create_tree_prototypes(stage, root, references, trunk, foliage)
    instancer = UsdGeom.PointInstancer.Define(stage, root + "/Forest")
    instancer.CreatePrototypesRel().SetTargets(prototype_paths)
    positions, scales, proto_indices, orientations = [], [], [], []
    for _ in range(count):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        positions.append(Gf.Vec3f(x, y, height_at(x, y)))
        factor = rng.uniform(0.72, 1.26)
        scales.append(Gf.Vec3f(factor, factor, factor))
        proto_indices.append(rng.randrange(len(prototype_paths)))
        angle = math.radians(rng.uniform(0, 360))
        orientations.append(Gf.Quath(math.cos(angle / 2), Gf.Vec3h(0, 0, math.sin(angle / 2))))
    instancer.CreatePositionsAttr(Vt.Vec3fArray(positions))
    instancer.CreateScalesAttr(Vt.Vec3fArray(scales))
    instancer.CreateProtoIndicesAttr(Vt.IntArray(proto_indices))
    instancer.CreateOrientationsAttr(Vt.QuathArray(orientations))
    return count, asset_mode


def building_positions(family: str, count: int, bounds: tuple[float, float, float, float], rng: random.Random, height_at) -> list[dict[str, Any]]:
    xmin, ymin, xmax, ymax = bounds
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    anchors = {
        "ridge_hamlet": ((cx, ymin + (ymax - ymin) * 0.68), (cx + (xmax - xmin) * 0.14, cy)),
        "riparian_grove": ((cx - (xmax - xmin) * 0.14, ymin + (ymax - ymin) * 0.32),),
        "firebreak_estate": ((cx + (xmax - xmin) * 0.10, cy + (ymax - ymin) * 0.10),),
        "wetland_edge": ((cx - (xmax - xmin) * 0.18, ymin + (ymax - ymin) * 0.48),),
        "dispersed_hamlet": (),
    }[family]
    styles = ("farmhouse", "masonry", "barn", "cabin")
    values: list[dict[str, Any]] = []
    for index in range(count):
        if anchors:
            ax, ay = anchors[index % len(anchors)]
            spread = 0.13 if family == "ridge_hamlet" else 0.20
            x = ax + rng.uniform(-1, 1) * (xmax - xmin) * spread
            y = ay + rng.uniform(-1, 1) * (ymax - ymin) * spread
        else:
            x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        x, y = max(xmin, min(xmax, x)), max(ymin, min(ymax, y))
        style = styles[(index + rng.randrange(len(styles))) % len(styles)]
        width = rng.uniform(10, 20) if style != "barn" else rng.uniform(18, 34)
        depth = rng.uniform(8, 16) if style != "barn" else rng.uniform(12, 22)
        wall_height = rng.uniform(3.8, 6.8) if style != "barn" else rng.uniform(5.2, 8.2)
        values.append({"x": x, "y": y, "width": width, "depth": depth, "wall_height": wall_height, "ground": height_at(x, y), "style": style})
    return values


def create_fence(stage: Usd.Stage, root: str, spec: dict[str, Any], material_value: UsdShade.Material) -> None:
    x, y, z = spec["x"], spec["y"], spec["ground"]
    half_width, half_depth = spec["width"] * 0.74, spec["depth"] * 0.80
    for index, (dx, dy) in enumerate(((-half_width, -half_depth), (half_width, -half_depth), (half_width, half_depth), (-half_width, half_depth))):
        add_cylinder(stage, f"{root}/Fence/Post{index}", (x + dx, y + dy, z + 0.75), 0.07, 1.5, material_value)
    for index, (dx, dy, sx, sy) in enumerate(((0, -half_depth, half_width * 2, 0.08), (0, half_depth, half_width * 2, 0.08), (-half_width, 0, 0.08, half_depth * 2), (half_width, 0, 0.08, half_depth * 2))):
        add_cube(stage, f"{root}/Fence/Rail{index}", (x + dx, y + dy, z + 1.05), (sx, sy, 0.07), material_value)


def create_building(stage: Usd.Stage, root: str, index: int, spec: dict[str, Any], lod: str, materials: dict[str, UsdShade.Material]) -> None:
    x, y, width, depth = spec["x"], spec["y"], spec["width"], spec["depth"]
    ground, wall_height, style = spec["ground"], spec["wall_height"], spec["style"]
    building = f"{root}/Buildings/{style.title()}{index:03d}"
    if lod == "far":
        add_cube(stage, building + "/Envelope", (x, y, ground + wall_height * 0.55), (width, depth, wall_height * 1.1), materials["masonry"])
        return
    wall_material = materials["timber"] if style == "cabin" else materials["masonry"] if style == "barn" else materials["plaster"]
    roof_material = materials["metal_roof"] if style == "barn" else materials["tile_roof"]
    add_cube(stage, building + "/Walls", (x, y, ground + wall_height / 2.0), (width, depth, wall_height), wall_material)
    roof_rise = max(1.45, wall_height * (0.28 if style == "barn" else 0.38))
    add_gable_roof(stage, building + "/PitchedRoof", (x, y, ground + wall_height), width * 1.12, depth * 1.12, roof_rise, roof_material)
    if lod == "mid":
        return
    add_cube(stage, building + "/Door", (x, y - depth * 0.507, ground + 1.15), (min(1.8, width * 0.16), 0.10, 2.3), materials["wood"])
    if style == "barn":
        add_cube(stage, building + "/SlidingDoor", (x + width * 0.24, y - depth * 0.509, ground + 1.75), (width * 0.32, 0.12, 3.5), materials["wood"])
        add_cylinder(stage, building + "/Silo", (x - width * 0.76, y + depth * 0.18, ground + 4.0), 2.1, 8.0, materials["metal"])
    else:
        for window_index, (dx, dy) in enumerate(((-0.28, -0.51), (0.28, -0.51), (-0.34, 0.51), (0.34, 0.51))):
            add_cube(stage, building + f"/Window{window_index}", (x + width * dx, y + depth * dy, ground + wall_height * 0.56), (width * 0.14, 0.08, wall_height * 0.20), materials["glass"])
            add_cube(stage, building + f"/ShutterL{window_index}", (x + width * (dx - 0.09), y + depth * (dy - math.copysign(0.012, dy)), ground + wall_height * 0.56), (width * 0.025, 0.045, wall_height * 0.23), materials["wood"])
            add_cube(stage, building + f"/ShutterR{window_index}", (x + width * (dx + 0.09), y + depth * (dy - math.copysign(0.012, dy)), ground + wall_height * 0.56), (width * 0.025, 0.045, wall_height * 0.23), materials["wood"])
        add_cylinder(stage, building + "/Chimney", (x + width * 0.23, y + depth * 0.10, ground + wall_height + roof_rise + 0.7), 0.32, 2.2, materials["masonry"])
        add_cube(stage, building + "/Porch", (x, y - depth * 0.63, ground + 0.13), (width * 0.34, depth * 0.22, 0.26), materials["wood"])
    create_fence(stage, building, spec, materials["fence"])


def create_ribbon(stage: Usd.Stage, path: str, coordinates: list[tuple[float, float]], width: float, height_at, elevation: float, binding: UsdShade.Material) -> None:
    points: list[Gf.Vec3f] = []
    for index, (x, y) in enumerate(coordinates):
        before = coordinates[max(0, index - 1)]
        after = coordinates[min(len(coordinates) - 1, index + 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = max(math.hypot(dx, dy), 0.001)
        nx, ny = -dy / length * width / 2.0, dx / length * width / 2.0
        points.extend((Gf.Vec3f(x + nx, y + ny, height_at(x + nx, y + ny) + elevation), Gf.Vec3f(x - nx, y - ny, height_at(x - nx, y - ny) + elevation)))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (len(coordinates) - 1)))
    indices: list[int] = []
    for index in range(len(coordinates) - 1):
        indices.extend((index * 2, index * 2 + 1, index * 2 + 3, index * 2 + 2))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI(mesh).Bind(binding)


def create_infrastructure(stage: Usd.Stage, root: str, family: str, bounds: tuple[float, float, float, float], height_at, materials: dict[str, UsdShade.Material]) -> None:
    xmin, ymin, xmax, ymax = bounds
    road, water = [], []
    for index in range(17):
        fraction = index / 16.0
        x = xmin + (xmax - xmin) * fraction
        road.append((x, (ymin + ymax) / 2.0 + math.sin(fraction * math.pi * 2.0) * (ymax - ymin) * 0.115))
        water.append((x, ymin + (ymax - ymin) * (0.24 + 0.16 * math.sin(fraction * math.pi * 1.6))))
    create_ribbon(stage, root + "/Road", road, 7.0 if family != "firebreak_estate" else 9.0, height_at, 0.14, materials["road"])
    create_ribbon(stage, root + "/RoadShoulder", road, 9.4 if family != "firebreak_estate" else 12.0, height_at, 0.09, materials["earth"])
    create_ribbon(stage, root + "/Watercourse", water, 14.0 if family == "wetland_edge" else 7.5, height_at, 0.16, materials["water"])
    bridge_x, bridge_y = water[len(water) // 2]
    bridge_z = height_at(bridge_x, bridge_y) + 0.36
    add_cube(stage, root + "/Bridge/Deck", (bridge_x, bridge_y, bridge_z), (10.0, 2.3, 0.28), materials["wood"])
    for name, sign in (("North", -1.0), ("South", 1.0)):
        add_cube(stage, root + f"/Bridge/Rail{name}", (bridge_x, bridge_y + sign * 2.05, bridge_z + 0.75), (10.0, 0.10, 0.10), materials["metal"])


def build_scene(base_dir: Path, output: Path, ordinal: int, asset_root: Path | None) -> dict[str, Any]:
    manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    package_id = str(manifest["package_id"])
    family, building_count, tree_count, _, _ = FAMILIES[ordinal - 1]
    rng = seeded(package_id, ordinal)
    xmin, ymin, xmax, ymax, height_at = terrain_sampler(base_dir / "terrain.usda", manifest)
    span_x, span_y = xmax - xmin, ymax - ymin
    local_bounds = (xmin + span_x * 0.16, ymin + span_y * 0.16, xmax - span_x * 0.16, ymax - span_y * 0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths.append(str(Path("..").joinpath("..", "omniverse", base_dir.name, "terrain.usda")).replace("\\", "/"))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    scenario = UsdGeom.Xform.Define(stage, "/Scenario")
    stage.SetDefaultPrim(scenario.GetPrim())
    scenario.GetPrim().SetCustomDataByKey("fireviewer:classification", "synthetic_scenario_variation")
    scenario.GetPrim().SetCustomDataByKey("fireviewer:base_package", package_id)
    scenario.GetPrim().SetCustomDataByKey("fireviewer:family", family)
    scenario.GetPrim().SetCustomDataByKey("fireviewer:seed", str(rng.getstate()[1][0]))
    scenario.GetPrim().SetCustomDataByKey("fireviewer:asset_provenance", "NVIDIA AECO CityMassing Demo Pack, supplied locally")
    materials_scope = UsdGeom.Scope.Define(stage, "/Scenario/Materials")
    mp = str(materials_scope.GetPath())
    materials = {
        "plaster": material(stage, mp + "/Plaster", (0.54, 0.43, 0.31), 0.86),
        "masonry": material(stage, mp + "/Masonry", (0.34, 0.18, 0.11), 0.90),
        "timber": material(stage, mp + "/Timber", (0.19, 0.075, 0.028), 0.84),
        "tile_roof": material(stage, mp + "/TileRoof", (0.34, 0.055, 0.028), 0.78),
        "metal_roof": material(stage, mp + "/MetalRoof", (0.16, 0.22, 0.20), 0.45, 0.55),
        "glass": material(stage, mp + "/Glass", (0.03, 0.18, 0.28), 0.13, 0.15),
        "wood": material(stage, mp + "/Wood", (0.25, 0.10, 0.035), 0.78),
        "fence": material(stage, mp + "/Fence", (0.18, 0.065, 0.020), 0.92),
        "metal": material(stage, mp + "/Metal", (0.26, 0.29, 0.30), 0.36, 0.70),
        "road": material(stage, mp + "/Road", (0.105, 0.095, 0.080), 0.91),
        "earth": material(stage, mp + "/Earth", (0.23, 0.15, 0.09), 0.98),
        "water": material(stage, mp + "/Water", (0.018, 0.15, 0.23), 0.10, 0.10),
        "trunk": material(stage, mp + "/ProxyTrunk", (0.15, 0.065, 0.020), 0.96),
        "foliage": material(stage, mp + "/ProxyFoliage", (0.04, 0.20, 0.065), 0.92),
    }
    composition = UsdGeom.Xform.Define(stage, "/Scenario/Composition")
    far_group = UsdGeom.Xform.Define(stage, str(composition.GetPath()) + "/Far")
    mid_group = UsdGeom.Xform.Define(stage, str(composition.GetPath()) + "/Mid")
    near_group = UsdGeom.Xform.Define(stage, str(composition.GetPath()) + "/Near")
    building_specs = building_positions(family, building_count, local_bounds, rng, height_at)
    for index, spec in enumerate(building_specs):
        create_building(stage, str(far_group.GetPath()), index, spec, "far", materials)
        create_building(stage, str(mid_group.GetPath()), index, spec, "mid", materials)
        create_building(stage, str(near_group.GetPath()), index, spec, "near", materials)
    references = tree_reference_paths(asset_root, output)
    far_vegetation, far_mode = create_vegetation_instancer(stage, str(far_group.GetPath()) + "/Vegetation", rng, max(28, tree_count // 5), local_bounds, height_at, [], materials["trunk"], materials["foliage"])
    mid_vegetation, mid_mode = create_vegetation_instancer(stage, str(mid_group.GetPath()) + "/Vegetation", rng, max(45, tree_count // 2), local_bounds, height_at, references, materials["trunk"], materials["foliage"])
    near_vegetation, near_mode = create_vegetation_instancer(stage, str(near_group.GetPath()) + "/Vegetation", rng, tree_count, local_bounds, height_at, references, materials["trunk"], materials["foliage"])
    create_infrastructure(stage, str(composition.GetPath()) + "/Infrastructure", family, local_bounds, height_at, materials)
    lod_set = scenario.GetPrim().GetVariantSets().AddVariantSet("lod")
    for selected, visible_group in (("far", far_group), ("mid", mid_group), ("near", near_group)):
        lod_set.AddVariant(selected)
        lod_set.SetVariantSelection(selected)
        with lod_set.GetVariantEditContext():
            for group in (far_group, mid_group, near_group):
                set_visibility(group.GetPrim(), UsdGeom.Tokens.inherited if group == visible_group else UsdGeom.Tokens.invisible)
    lod_set.SetVariantSelection("mid")
    scenario.GetPrim().SetCustomDataByKey("fireviewer:lod_ranges_m", {"near": [0, 110], "mid": [110, 900], "far": [900, 1000000]})
    sun = UsdLux.DistantLight.Define(stage, "/Scenario/Sun")
    sun.CreateIntensityAttr(2200.0)
    sun.CreateAngleAttr(1.2)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(42.0, -28.0, 18.0 + ordinal * 11.0))
    sky = UsdLux.DomeLight.Define(stage, "/Scenario/Sky")
    sky.CreateIntensityAttr(420.0)
    sky.CreateColorAttr(Gf.Vec3f(0.42, 0.56, 0.72))
    camera = UsdGeom.Camera.Define(stage, "/Scenario/OverviewCamera")
    camera.CreateFocalLengthAttr(28.0)
    centre_x, centre_y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    camera_z = height_at(centre_x, centre_y) + max(span_x, span_y) * 0.72
    UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(centre_x, ymin - span_y * 0.32, camera_z))
    stage.GetRootLayer().Save()
    return {
        "schema": "fireviewer.omniverse-scenario.v2",
        "scenario_id": f"{package_id}-{ordinal:02d}",
        "base_package": package_id,
        "family": family,
        "ordinal": ordinal,
        "stage": output.name,
        "classification": "synthetic_scenario_variation",
        "composition": {"buildings": building_count, "building_lod": {"far": "envelope", "mid": "walls_and_pitched_roof", "near": "windows_doors_shutters_chimney_fence"}, "vegetation_instances": {"far": far_vegetation, "mid": mid_vegetation, "near": near_vegetation}, "vegetation_asset_mode": {"far": far_mode, "mid": mid_mode, "near": near_mode}, "watercourse": 1, "road": 1, "bridge": 1, "sun": 1, "sky": 1, "camera": 1},
        "lod": {"default": "mid", "variants": ["far", "mid", "near"], "ranges_m": {"near": [0, 110], "mid": [110, 900], "far": [900, 1000000]}},
    }


def main() -> int:
    raise SystemExit(
        "REJECTED: this legacy generator authors primitive proxy geometry. "
        "Use build_fictional_asset_scenes.py; legacy scenes are not reviewable or publishable."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
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
            output = args.output_root / str(manifest["package_id"]) / f"scenario-{ordinal:02d}.usda"
            records.append(build_scene(base_dir, output, ordinal, args.asset_root))
    result = {"schema": "fireviewer.omniverse-scenario-index.v2", "scene_count": len(records), "scenes": records}
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "index.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
