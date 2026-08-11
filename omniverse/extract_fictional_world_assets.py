"""Extract a compact, high-fidelity NVIDIA asset kit for fictional worlds.

Only complete dependency folders are extracted.  Scene builders reference
these assets directly and are forbidden from substituting primitive geometry.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


BROWNSTONE_PREFIX = "Demos/AEC/BrownstoneDemo/"
BROWNSTONE_FOLDERS = (
    "Assets/Revit_Brownstone01/",
    "Assets/Revit_Brownstone02/",
    "Assets/Revit_Brownstone03/",
    "Assets/Max_BrownstoneSite/",
    "Props/PlanterFence/",
)
BROWNSTONE_TREES_PREFIX = "Demos/AEC/BrownstoneDemo/Assets/Vegetation/Trees/"
WATER_PREFIX = "Samples/Flight/SubUSDs/"
TOWER_STREETS_PREFIX = "Demos/AEC/TowerDemo/TowerDemopack/Source/context_City/ce_Context_City/ce_Context_City_Mini_Bldg/"
UNDERGROWTH_ENTRIES = (
    "Demos/AEC/BrownstoneDemo/Assets/Vegetation/Shrub/Grass_Short_A.usd",
    "Demos/AEC/BrownstoneDemo/Assets/Vegetation/Shrub/Grass_Short_B.usd",
    "Demos/AEC/BrownstoneDemo/Assets/Vegetation/Shrub/Grass_Short_C.usd",
    "Demos/AEC/BrownstoneDemo/Assets/Vegetation/Shrub/Meadowlark.usd",
)
BRIDGE_PREFIX = "Samples/Marbles/assets/standalone/RT_bridge_rotating/"


def copy_prefixes(archive_path: Path, output_root: Path, prefixes: tuple[str, ...], strip_prefix: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith(prefixes):
                continue
            relative = Path(entry.filename).relative_to(strip_prefix)
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size != entry.file_size:
                with archive.open(entry) as source, destination.open("wb") as target:
                    target.write(source.read())
            records.append({"path": relative.as_posix(), "bytes": entry.file_size})
    return records


def copy_entries(archive_path: Path, output_root: Path, entries: tuple[str, ...], strip_prefix: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in entries:
            entry = archive.getinfo(name)
            relative = Path(entry.filename).relative_to(strip_prefix)
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size != entry.file_size:
                with archive.open(entry) as source, destination.open("wb") as target:
                    target.write(source.read())
            records.append({"path": relative.as_posix(), "bytes": entry.file_size})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aeco-demo", type=Path, required=True)
    parser.add_argument("--tower-demo", type=Path, required=True)
    parser.add_argument("--sample-scenes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for archive in (args.aeco_demo, args.tower_demo, args.sample_scenes):
        if not archive.is_file():
            raise SystemExit(f"missing archive: {archive}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = copy_prefixes(
        args.aeco_demo,
        args.output_root / "brownstone",
        tuple(BROWNSTONE_PREFIX + folder for folder in BROWNSTONE_FOLDERS),
        BROWNSTONE_PREFIX,
    )
    records += copy_entries(args.aeco_demo, args.output_root / "brownstone", UNDERGROWTH_ENTRIES, BROWNSTONE_PREFIX)
    records += copy_prefixes(args.aeco_demo, args.output_root / "brownstone", (BROWNSTONE_TREES_PREFIX,), BROWNSTONE_PREFIX)
    records += copy_prefixes(args.aeco_demo, args.output_root / "brownstone", (BROWNSTONE_PREFIX + "Props/StreetLamp/", BROWNSTONE_PREFIX + "Props/StreetLight01/"), BROWNSTONE_PREFIX)
    records += copy_prefixes(args.tower_demo, args.output_root / "tower-streets", (TOWER_STREETS_PREFIX,), "Demos/AEC/TowerDemo/TowerDemopack")
    records += copy_prefixes(
        args.sample_scenes,
        args.output_root / "sample-scenes",
        (WATER_PREFIX,),
        "Samples/Flight",
    )
    records += copy_prefixes(args.sample_scenes, args.output_root / "sample-scenes", (BRIDGE_PREFIX,), "Samples/Marbles/assets/standalone")
    manifest = {
        "schema": "fireviewer.fictional-world-asset-kit.v1",
        "classification": "licensed_local_nvidia_assets_for_fictional_scene_composition",
        "prohibited_fallback": "primitive_geometry",
        "assets": {
            "buildings": [
                "brownstone/Assets/Revit_Brownstone01/Revit_Brownstone01_Exterior.usd",
                "brownstone/Assets/Revit_Brownstone02/Revit_Brownstone02_Exterior.usd",
                "brownstone/Assets/Revit_Brownstone03/Revit_Brownstone03_Exterior.usd",
            ],
            "water": "sample-scenes/SubUSDs/Water_Mesh_v01.usd",
            "fences": "brownstone/Props/PlanterFence/PlanterFenceType1.usd",
            "undergrowth": ["brownstone/Assets/Vegetation/Shrub/Grass_Short_A.usd", "brownstone/Assets/Vegetation/Shrub/Grass_Short_B.usd", "brownstone/Assets/Vegetation/Shrub/Meadowlark.usd"],
            "trees": "brownstone/Assets/Vegetation/Trees/",
            "street_network": "tower-streets/Source/context_City/ce_Context_City/ce_Context_City_Mini_Bldg/layers/Streetnetwork.usdc",
            "bridge": "sample-scenes/RT_bridge_rotating/RT_bridge_rotating.usd",
        },
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }
    (args.output_root / "asset-kit-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": manifest["file_count"], "bytes": manifest["bytes"], "assets": manifest["assets"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
