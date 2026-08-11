"""Reject primitive-proxy scenes and validate fictional USD asset references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom


FORBIDDEN_TOKENS = ("def Cube", "def Cone", "def Cylinder", "def BasisCurves")
EXPECTED_TREE_PROTOTYPES = 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.root / "index.json").read_text(encoding="utf-8"))
    scenes = index.get("scenes", [])
    if index.get("scene_count") != 20 or len(scenes) != 20:
        raise SystemExit("expected 20 fictional scenes")
    details = []
    for record in scenes:
        path = args.root / record["base_package"] / record["stage"]
        content = path.read_text(encoding="utf-8")
        forbidden = [token for token in FORBIDDEN_TOKENS if token in content]
        if forbidden:
            raise SystemExit(f"forbidden primitive declaration in {path}: {forbidden}")
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
        root = stage.GetDefaultPrim()
        if not root.IsValid() or root.GetCustomDataByKey("fireviewer:classification") != "fictional_high_fidelity_scene":
            raise SystemExit(f"invalid fictional scene metadata: {path}")
        variants = root.GetVariantSets().GetVariantSet("lod")
        if set(variants.GetVariantNames()) != {"far", "mid", "near"}:
            raise SystemExit(f"invalid LOD variants: {path}")
        near = stage.GetPrimAtPath("/FictionalWorld/Near")
        building_payloads = sum(prim.HasPayload() for prim in Usd.PrimRange(near))
        tree_prototypes = stage.GetPrimAtPath("/FictionalWorld/Near/Vegetation/Prototypes").GetChildren()
        near_tree_count = len(stage.GetPrimAtPath("/FictionalWorld/Near/Vegetation/Instances").GetAttribute("positions").Get() or [])
        if building_payloads < int(record["visible_assets"]["detailed_buildings"]) or len(tree_prototypes) != EXPECTED_TREE_PROTOTYPES or near_tree_count != int(record["visible_assets"]["tree_instances"]["near"]):
            raise SystemExit(f"missing detailed asset references: {path}")
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        tree_size = bbox.ComputeWorldBound(stage.GetPrimAtPath("/FictionalWorld/Near/Vegetation/Prototypes/Tree1")).GetRange().GetSize()
        cover_size = bbox.ComputeWorldBound(stage.GetPrimAtPath("/FictionalWorld/Near/GroundCover/Prototypes/GroundCover1")).GetRange().GetSize()
        building_size = bbox.ComputeWorldBound(stage.GetPrimAtPath("/FictionalWorld/Near/Architecture/Residence01")).GetRange().GetSize()
        if not (1.0 <= max(tree_size) <= 35.0 and 0.05 <= max(cover_size) <= 5.0 and 5.0 <= max(building_size) <= 50.0):
            raise SystemExit(f"invalid authored asset scale in {path}: tree={tree_size}, ground_cover={cover_size}, building={building_size}")
        details.append({"scene": record["scene_id"], "building_payloads": building_payloads, "tree_prototypes": len(tree_prototypes), "near_trees": near_tree_count, "water": record["visible_assets"]["water_feature"]})
    print(json.dumps({"validated_fictional_high_fidelity_scenes": len(details), "details": details}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
