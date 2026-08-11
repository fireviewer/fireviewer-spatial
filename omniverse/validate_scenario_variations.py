"""Structural acceptance checks for generated FireViewer Omniverse scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.scenario_root / "index.json").read_text(encoding="utf-8"))
    if index.get("scene_count") != 20 or len(index.get("scenes", [])) != 20:
        raise SystemExit("expected exactly 20 indexed scenes")
    expected_assets = tuple(args.asset_root / "Assets" / "Vegetation" / name / f"{name}.usd" for name in ("Black_Oak", "Shumard_Oak", "Common_Apple", "Hawthorn"))
    missing = [str(path) for path in expected_assets if not path.is_file()]
    if missing:
        raise SystemExit("missing reusable tree asset(s): " + ", ".join(missing))
    fingerprints: set[tuple[object, ...]] = set()
    details = []
    for record in index["scenes"]:
        stage_path = args.scenario_root / record["base_package"] / record["stage"]
        stage = Usd.Stage.Open(str(stage_path))
        if stage is None:
            raise SystemExit(f"cannot open {stage_path}")
        scenario = stage.GetPrimAtPath("/Scenario")
        if not scenario.IsValid() or scenario.GetCustomDataByKey("fireviewer:classification") != "synthetic_scenario_variation":
            raise SystemExit(f"invalid scenario classification: {stage_path}")
        lod = scenario.GetVariantSets().GetVariantSet("lod")
        if set(lod.GetVariantNames()) != {"far", "mid", "near"}:
            raise SystemExit(f"invalid LOD variants: {stage_path}")
        near_root = stage.GetPrimAtPath("/Scenario/Composition/Near")
        mid_root = stage.GetPrimAtPath("/Scenario/Composition/Mid")
        far_root = stage.GetPrimAtPath("/Scenario/Composition/Far")
        if not near_root.IsValid() or not mid_root.IsValid() or not far_root.IsValid():
            raise SystemExit(f"missing LOD composition: {stage_path}")
        near_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(near_root)]
        roofs = sum(path.endswith("/PitchedRoof") for path in near_paths)
        windows = sum("/Window" in path for path in near_paths)
        fences = sum("/Fence/" in path for path in near_paths)
        if roofs < int(record["composition"]["buildings"]) or windows == 0 or fences == 0:
            raise SystemExit(f"near LOD lacks detailed built features: {stage_path}")
        tree_prototypes = stage.GetPrimAtPath("/Scenario/Composition/Near/Vegetation/Prototypes").GetChildren()
        if [prim.GetName() for prim in tree_prototypes] != ["Tree1", "Tree2", "Tree3", "Tree4"]:
            raise SystemExit(f"near LOD is not using all four tree asset prototypes: {stage_path}")
        fingerprint = (record["family"], record["composition"]["buildings"], record["composition"]["vegetation_instances"]["near"])
        fingerprints.add(fingerprint)
        details.append({"scene": record["scenario_id"], "roofs": roofs, "windows": windows, "fence_prims": fences, "near_tree_prototypes": len(tree_prototypes)})
    if len(fingerprints) != 5:
        raise SystemExit("expected five distinct composition families across the 20 scenes")
    print(json.dumps({"validated_complete_scenes": len(details), "composition_families": len(fingerprints), "details": details}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
