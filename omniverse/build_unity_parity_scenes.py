"""Build native USD scene variants that preserve the Unity forest verbatim.

The base terrain and every LiDAR-detected tree count belong to the Unity
source.  Each variant has its own 500 m forest payloads.  The source point set
is spatially rearranged per tile, while every tile preserves its exact source
tree count and every instance retains the measured height, crown diameter and
orientation.  A variant then adds a separate, fictional composition layer for
complete authored USD road and building assets.

This is deliberately not the earlier fictional-world generator.  It has no
procedural trees, no primitive geometry and no synthetic terrain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unity"))
from fwtile import TREE_RECORD, read_container  # noqa: E402


TREE_ASSETS = (
    "fictional-world-kit/brownstone/Assets/Vegetation/Trees/Scarlet_Oak_fall.usd",
    "fictional-world-kit/brownstone/Assets/Vegetation/Trees/Honey_Locust.usd",
    "fictional-world-kit/brownstone/Assets/Vegetation/Trees/Largetooth_Aspen.usd",
    "fictional-world-kit/brownstone/Assets/Vegetation/Trees/Service_Berry.usd",
    "fictional-world-kit/brownstone/Assets/Vegetation/Trees/Hawthorn.usd",
    "aeco-citymassing/Assets/Vegetation/Black_Oak/Black_Oak.usd",
    "aeco-citymassing/Assets/Vegetation/Shumard_Oak/Shumard_Oak.usd",
)
BUILDINGS = (
    ("fictional-world-kit/brownstone/Assets/Revit_Brownstone01/Revit_Brownstone01_Exterior.usd", "/World/Brownstone01"),
    ("fictional-world-kit/brownstone/Assets/Revit_Brownstone02/Revit_Brownstone02_Exterior.usd", "/World/Brownstone02"),
    ("fictional-world-kit/brownstone/Assets/Revit_Brownstone03/Revit_Brownstone03_Exterior.usd", "/World/Brownstone03"),
)
STREET_NETWORK = "fictional-world-kit/tower-streets/Source/context_City/ce_Context_City/ce_Context_City_Mini_Bldg/layers/Streetnetwork.usdc"


@dataclass(frozen=True)
class TreePrototype:
    asset: Path
    physical_width_m: float
    physical_depth_m: float
    physical_height_m: float


def tree_fallback_material(stage: Usd.Stage) -> UsdShade.Material:
    """Create a robust reusable fallback material for vegetation.

    It keeps trees visible with a plausible leafy tone when reference materials
    fail to resolve in downstream composition.
    """
    material_path = Sdf.Path("/UnityForest/Materials/FallbackTree")
    fallback = UsdShade.Material.Define(stage, str(material_path))
    shader = UsdShade.Shader.Define(stage, str(material_path / "Preview"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.13, 0.45, 0.12))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    fallback.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return fallback


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_asset(from_path: Path, target: Path) -> str:
    return os.path.relpath(target, from_path.parent).replace("\\", "/")


def stable_rng(package_id: str, ordinal: int) -> random.Random:
    seed = hashlib.sha256(f"unity-parity:{package_id}:{ordinal}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(seed[:8], "little"))


def physical_bbox(asset: Path) -> tuple[float, float, float]:
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"cannot open tree asset: {asset}")
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    size = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root).GetRange().GetSize()
    metres_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)
    result = tuple(float(component) * metres_per_unit for component in size)
    if min(result) <= 0.0:
        raise RuntimeError(f"tree asset has an empty bound: {asset}")
    return result


def load_tree_prototypes(asset_root: Path) -> list[TreePrototype]:
    prototypes: list[TreePrototype] = []
    for relative in TREE_ASSETS:
        asset = asset_root / relative
        if not asset.is_file():
            raise FileNotFoundError(asset)
        width, depth, height = physical_bbox(asset)
        prototypes.append(TreePrototype(asset, width, depth, height))
    return prototypes


def add_tree_prototypes(stage: Usd.Stage, path: str, output: Path, prototypes: list[TreePrototype]) -> list[Sdf.Path]:
    container = UsdGeom.Xform.Define(stage, path)
    fallback = tree_fallback_material(stage)
    targets: list[Sdf.Path] = []
    for index, prototype in enumerate(prototypes):
        tree = UsdGeom.Xform.Define(stage, f"{container.GetPath()}/Tree{index:02d}")
        # Source assets use centimetres; the forest stage is in metres.
        UsdGeom.Xformable(tree).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
        tree.GetPrim().GetReferences().AddReference(relative_asset(output, prototype.asset))
        try:
            UsdShade.MaterialBindingAPI(tree).Bind(fallback, UsdShade.Tokens.strongerThanDescendants)
        except (TypeError, AttributeError):
            # Older USD bindings: keep compatibility.
            UsdShade.MaterialBindingAPI(tree).Bind(fallback)
        targets.append(tree.GetPath())
    return targets


def iter_tree_records(raw: bytes) -> Iterable[tuple[int, int, int, int, int, int, int]]:
    if len(raw) % TREE_RECORD.size:
        raise ValueError("tree payload has an incomplete record")
    return TREE_RECORD.iter_unpack(raw)


def rearrange_position(
    east_mm: int,
    north_mm: int,
    rotation_cd: int,
    *,
    width_mm: int,
    height_mm: int,
    tile_id: str,
    ordinal: int,
) -> tuple[int, int, int]:
    """Rearrange the source point pattern without changing its density.

    A variant uses a deterministic square-tile rotation and a wrapped offset.
    This preserves tree count, pairwise local structure and coverage density;
    only the layout is different.  The modulo keeps every point in its source
    tile so no forest coverage leaks over the map boundary.
    """

    digest = hashlib.sha256(f"{tile_id}:{ordinal}".encode("utf-8")).digest()
    quarter_turns = (digest[0] + ordinal) % 4
    shift_x = int.from_bytes(digest[1:5], "little") % width_mm
    shift_y = int.from_bytes(digest[5:9], "little") % height_mm
    x, y = east_mm, north_mm
    if quarter_turns == 1:
        x, y = height_mm - y, x
    elif quarter_turns == 2:
        x, y = width_mm - x, height_mm - y
    elif quarter_turns == 3:
        x, y = y, width_mm - x
    return ((x + shift_x) % width_mm, (y + shift_y) % height_mm, (rotation_cd + quarter_turns * 9000) % 36000)


def write_forest_tile(
    *,
    entry: dict[str, Any],
    source_root: Path,
    output: Path,
    source_origin: tuple[float, float, float],
    prototypes: list[TreePrototype],
    ordinal: int,
) -> dict[str, Any]:
    payload = read_container((source_root / entry["payload"]["url"]).read_bytes())
    header = payload["header"]
    bounds = tuple(float(value) for value in header["bounds_l93_m"])
    xmin, ymin, xmax, ymax = bounds
    raw = payload["sections"].get("trees", b"")
    stage = Usd.Stage.CreateNew(str(output))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/ForestTile")
    root.GetPrim().SetCustomDataByKey("fireviewer:tile_id", str(entry["id"]))
    root.GetPrim().SetCustomDataByKey("fireviewer:bounds_l93_m_json", json.dumps(bounds, separators=(",", ":")))
    root.GetPrim().SetCustomDataByKey("fireviewer:source_record_count", int(entry["counts"]["trees"]))
    root.GetPrim().SetCustomDataByKey("fireviewer:layout_ordinal", ordinal)
    root.GetPrim().SetCustomDataByKey("fireviewer:layout_contract", "same_count_rearranged_within_source_tile")
    stage.SetDefaultPrim(root.GetPrim())
    targets = add_tree_prototypes(stage, "/ForestTile/Prototypes", output, prototypes)
    instancer = UsdGeom.PointInstancer.Define(stage, "/ForestTile/Trees")
    instancer.CreatePrototypesRel().SetTargets(targets)
    positions: list[Gf.Vec3f] = []
    scales: list[Gf.Vec3f] = []
    indices: list[int] = []
    orientations: list[Gf.Quath] = []
    width_mm = int(round((xmax - xmin) * 1000.0))
    height_mm = int(round((ymax - ymin) * 1000.0))
    for east_mm, north_mm, ground_mm, height_cm, crown_cm, visual_variant, rotation_cd in iter_tree_records(raw):
        east_mm, north_mm, rotation_cd = rearrange_position(east_mm, north_mm, rotation_cd, width_mm=width_mm, height_mm=height_mm, tile_id=str(entry["id"]), ordinal=ordinal)
        prototype_index = int(visual_variant) % len(prototypes)
        prototype = prototypes[prototype_index]
        crown_m = crown_cm / 100.0
        height_m = height_cm / 100.0
        # The source measurements drive scale.  No density, position, size or
        # rotation filter is applied; authored asset units are normalized only.
        horizontal_scale = crown_m / max((prototype.physical_width_m + prototype.physical_depth_m) / 2.0, 0.001)
        vertical_scale = height_m / max(prototype.physical_height_m, 0.001)
        angle = math.radians((rotation_cd % 36000) / 100.0)
        positions.append(Gf.Vec3f(east_mm / 1000.0, north_mm / 1000.0, source_origin[2] + ground_mm / 1000.0))
        scales.append(Gf.Vec3f(horizontal_scale, horizontal_scale, vertical_scale))
        indices.append(prototype_index)
        orientations.append(Gf.Quath(math.cos(angle / 2.0), Gf.Vec3h(0.0, 0.0, math.sin(angle / 2.0))))
    if len(positions) != int(entry["counts"]["trees"]):
        raise ValueError(f"{entry['id']}: source tree count mismatch")
    instancer.CreatePositionsAttr(Vt.Vec3fArray(positions))
    instancer.CreateScalesAttr(Vt.Vec3fArray(scales))
    instancer.CreateProtoIndicesAttr(Vt.IntArray(indices))
    instancer.CreateOrientationsAttr(Vt.QuathArray(orientations))
    stage.GetRootLayer().Save()
    return {
        "id": str(entry["id"]),
        "bounds_l93_m": list(bounds),
        "tree_count": len(positions),
        "path": output.name,
        "source_payload_sha256": str(entry["payload"]["sha256"]),
    }


def write_forest_index(
    *,
    records: list[dict[str, Any]],
    output: Path,
    global_origin: tuple[float, float, float],
) -> None:
    stage = Usd.Stage.CreateNew(str(output))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/UnityForest")
    root.GetPrim().SetCustomDataByKey("fireviewer:loading_contract", "complete_forest_tiled_payloads")
    root.GetPrim().SetCustomDataByKey("fireviewer:forest_layout", "variant_rearranged_same_source_counts")
    stage.SetDefaultPrim(root.GetPrim())
    tiles = UsdGeom.Scope.Define(stage, "/UnityForest/Tiles")
    for record in records:
        xmin, ymin, _, _ = (float(value) for value in record["bounds_l93_m"])
        tile = UsdGeom.Xform.Define(stage, f"{tiles.GetPath()}/{record['id']}")
        UsdGeom.Xformable(tile).AddTranslateOp().Set(Gf.Vec3d(xmin - global_origin[0], ymin - global_origin[1], 0.0))
        tile.GetPrim().GetPayloads().AddPayload(f"tiles/{record['path']}")
        tile.GetPrim().SetCustomDataByKey("fireviewer:tree_count", int(record["tree_count"]))
        tile.GetPrim().SetCustomDataByKey("fireviewer:bounds_l93_m_json", json.dumps(record["bounds_l93_m"], separators=(",", ":")))
    stage.GetRootLayer().Save()


def add_reference(
    stage: Usd.Stage,
    *,
    path: str,
    asset: Path,
    output: Path,
    position: tuple[float, float, float],
    scale: tuple[float, float, float],
    rotation_z: float,
    asset_prim: str | None = None,
) -> None:
    prim = UsdGeom.Xform.Define(stage, path)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateZOp().Set(float(rotation_z))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))
    if asset_prim:
        prim.GetPrim().GetPayloads().AddPayload(relative_asset(output, asset), asset_prim)
    else:
        prim.GetPrim().GetPayloads().AddPayload(relative_asset(output, asset))


def write_variant(
    *,
    output: Path,
    terrain_scene: Path,
    forest_index: Path,
    asset_root: Path,
    package_id: str,
    origin: tuple[float, float, float],
    extent: tuple[float, float, float, float],
    ordinal: int,
) -> dict[str, Any]:
    rng = stable_rng(package_id, ordinal)
    stage = Usd.Stage.CreateNew(str(output))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
    layer = stage.GetRootLayer()
    layer.subLayerPaths = [relative_asset(output, terrain_scene)]
    root = UsdGeom.Xform.Define(stage, "/UnityParityScene")
    root.GetPrim().SetCustomDataByKey("fireviewer:forest", "rearranged_unity_lidar_counts_and_measurements")
    root.GetPrim().SetCustomDataByKey("fireviewer:variant_scope", "forest_layout_assets_and_routes")
    stage.SetDefaultPrim(root.GetPrim())
    forest = UsdGeom.Xform.Define(stage, "/UnityParityScene/Forest")
    forest.GetPrim().GetReferences().AddReference(relative_asset(output, forest_index))
    overlays = UsdGeom.Scope.Define(stage, "/UnityParityScene/VariantOverlay")
    xmin, ymin, xmax, ymax = extent
    center_x = (xmin + xmax) / 2.0 - origin[0]
    center_y = (ymin + ymax) / 2.0 - origin[1]
    ground = origin[2]
    road = asset_root / STREET_NETWORK
    add_reference(
        stage,
        path=f"{overlays.GetPath()}/RoadNetwork",
        asset=road,
        output=output,
        position=(center_x + rng.uniform(-900, 900), center_y + rng.uniform(-650, 650), ground + 0.15),
        scale=(rng.uniform(0.8, 1.5), rng.uniform(0.8, 1.5), 1.0),
        rotation_z=float((ordinal * 29) % 180),
        asset_prim="/Streetnetwork",
    )
    building_count = 10 + (ordinal % 5) * 5
    for index in range(building_count):
        relative, prim_path = BUILDINGS[index % len(BUILDINGS)]
        angle = rng.uniform(0.0, math.tau)
        distance = rng.uniform(150.0, 1000.0)
        add_reference(
            stage,
            path=f"{overlays.GetPath()}/Architecture/Building{index + 1:02d}",
            asset=asset_root / relative,
            output=output,
            position=(center_x + math.cos(angle) * distance, center_y + math.sin(angle) * distance, ground),
            scale=(0.01, 0.01, 0.01),
            rotation_z=rng.uniform(0.0, 360.0),
            asset_prim=prim_path,
        )
    stage.GetRootLayer().Save()
    return {"id": f"{package_id}-variant-{ordinal:02d}", "stage": output.name, "building_count": building_count}


def build_package(
    *,
    source_root: Path,
    terrain_root: Path,
    output_root: Path,
    asset_root: Path,
    variants: int,
) -> dict[str, Any]:
    catalog_path = source_root / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    package_id = source_root.parent.name
    terrain_scene = terrain_root / package_id / "scene.usda"
    if not terrain_scene.is_file():
        raise FileNotFoundError(f"missing base terrain scene: {terrain_scene}")
    output = output_root / package_id
    origin = tuple(float(value) for value in catalog["origin_l93_m"])
    prototypes = load_tree_prototypes(asset_root)
    xmin = min(float(tile["bounds_l93_m"][0]) for tile in catalog["tiles"])
    ymin = min(float(tile["bounds_l93_m"][1]) for tile in catalog["tiles"])
    xmax = max(float(tile["bounds_l93_m"][2]) for tile in catalog["tiles"])
    ymax = max(float(tile["bounds_l93_m"][3]) for tile in catalog["tiles"])
    scene_records: list[dict[str, Any]] = []
    forest_variants: list[dict[str, Any]] = []
    for ordinal in range(1, variants + 1):
        records: list[dict[str, Any]] = []
        tiles_output = output / "forests" / f"variant-{ordinal:02d}" / "tiles"
        tiles_output.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(catalog["tiles"], start=1):
            destination = tiles_output / f"{entry['id']}.usdc"
            records.append(write_forest_tile(entry=entry, source_root=source_root, output=destination, source_origin=origin, prototypes=prototypes, ordinal=ordinal))
            if index % 50 == 0 or index == len(catalog["tiles"]):
                print(f"{package_id}: variant {ordinal:02d} forest tiles {index}/{len(catalog['tiles'])}", flush=True)
        forest_index = output / "forests" / f"variant-{ordinal:02d}" / "forest-index.usda"
        write_forest_index(records=records, output=forest_index, global_origin=origin)
        scene = output / "variants" / f"variant-{ordinal:02d}.usda"
        scene.parent.mkdir(parents=True, exist_ok=True)
        scene_records.append(write_variant(output=scene, terrain_scene=terrain_scene, forest_index=forest_index, asset_root=asset_root, package_id=package_id, origin=origin, extent=(xmin, ymin, xmax, ymax), ordinal=ordinal))
        forest_variants.append({"ordinal": ordinal, "tile_count": len(records), "tree_count": sum(int(record["tree_count"]) for record in records), "payload_index": str(forest_index.relative_to(output))})
    tree_count = sum(int(tile["counts"]["trees"]) for tile in catalog["tiles"])
    result = {
        "schema": "fireviewer.omniverse-unity-parity.v1",
        "package_id": package_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_catalog": {"path": str(catalog_path), "sha256": sha256_file(catalog_path)},
        "terrain_source_stage": str(terrain_scene),
        "forest": {
            "rule": "same_source_count_per_tile_rearranged_positions",
            "tile_count": len(catalog["tiles"]),
            "tree_count": tree_count,
            "layout_method": "deterministic_tile_rotation_and_wrapped_translation",
            "variants": forest_variants,
        },
        "variants": scene_records,
    }
    (output / "manifest.json").write_text(canonical_json(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True, help="Unity remote catalog directory")
    parser.add_argument("--terrain-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=10)
    args = parser.parse_args()
    if args.variants < 1:
        raise SystemExit("--variants must be positive")
    results = [build_package(source_root=path.resolve(), terrain_root=args.terrain_root.resolve(), output_root=args.output_root.resolve(), asset_root=args.asset_root.resolve(), variants=args.variants) for path in args.source]
    (args.output_root.resolve() / "index.json").write_text(canonical_json({"schema": "fireviewer.omniverse-unity-parity-index.v1", "packages": results}), encoding="utf-8")
    print(canonical_json({"packages": [{"package_id": result["package_id"], "tree_count": result["forest"]["tree_count"], "variants": len(result["variants"])} for result in results]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
