"""Extract only the NVIDIA Base Materials needed by the Brownstone exteriors.

The source archive is intentionally not unpacked wholesale.  The selected
material MDLs and their sibling texture folders are placed where the authored
Brownstone USD relative references require them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


REQUIRED = {
    "Clear_Glass.mdl", "Frosted_Glass.mdl",
    "Brick_Wall_Red.mdl", "Concrete_Block.mdl", "Concrete_Formed.mdl", "Concrete_Smooth.mdl", "Stucco.mdl",
    "Aluminum_Polished.mdl", "Chrome.mdl", "Steel_Carbon.mdl", "Steel_Cast.mdl", "Steel_Stainless.mdl",
    "Paint_Satin.mdl", "Grass_Countryside.mdl", "Grass_Cut.mdl", "Plastic.mdl", "Rubber_Smooth.mdl",
    "Marble_Smooth.mdl", "Porcelain_Smooth.mdl", "Porcelain_Tile_6_Linen.mdl", "WhiteMode.mdl",
    "Gypsum.mdl", "Plaster.mdl", "Ash.mdl", "Bamboo_Planks.mdl", "Cherry.mdl", "Oak_Planks.mdl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    destination_root = args.asset_root.resolve() / "fictional-world-kit" / "brownstone"
    if not archive.is_file():
        raise SystemExit(f"missing archive: {archive}")
    with ZipFile(archive) as source:
        selected_roots = {
            str(PurePosixPath(info.filename).parent / PurePosixPath(info.filename).stem)
            for info in source.infolist()
            if info.filename.startswith("Materials/Base/")
            and PurePosixPath(info.filename).name in REQUIRED
        }
        if len(selected_roots) != len(REQUIRED):
            raise SystemExit(f"archive only contains {len(selected_roots)}/{len(REQUIRED)} required materials")
        selected = [
            info for info in source.infolist()
            if "/.thumbs/" not in info.filename
            and any(info.filename == root + ".mdl" or info.filename.startswith(root + "/") for root in selected_roots)
        ]
        written = 0
        byte_count = 0
        for info in selected:
            if info.is_dir():
                continue
            target = destination_root / PurePosixPath(info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with source.open(info) as input_stream, target.open("wb") as output_stream:
                output_stream.write(input_stream.read())
            written += 1
            byte_count += info.file_size
    report = {
        "schema": "fireviewer.omniverse-required-base-materials.v1",
        "archive": str(archive),
        "required_material_count": len(REQUIRED),
        "selected_file_count": len(selected),
        "written_file_count": written,
        "written_bytes": byte_count,
        "destination_root": str(destination_root),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
