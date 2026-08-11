"""Extract the small, reusable subset of the local NVIDIA AECO library.

The full CityMassing pack is deliberately not copied.  The selected vegetation
packages are retained intact so their USD material and texture dependencies
remain relative and portable inside FireViewer's Omniverse artifact library.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


CITYMASSING_ROOT = "Demos/AEC/TowerDemo/CityMassingDemopack/Assets/Vegetation"
ASSETS = ("Black_Oak", "Shumard_Oak", "Common_Apple", "Hawthorn")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"missing archive: {args.archive}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(f"{CITYMASSING_ROOT}/{asset}/" for asset in ASSETS)
    copied: list[dict[str, object]] = []
    with zipfile.ZipFile(args.archive) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith(prefixes):
                continue
            relative = Path(entry.filename).relative_to("Demos/AEC/TowerDemo/CityMassingDemopack")
            destination = args.output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, destination.open("wb") as target:
                target.write(source.read())
            copied.append({"path": relative.as_posix(), "bytes": entry.file_size})

    manifest = {
        "schema": "fireviewer.omniverse-library.v1",
        "source": str(args.archive),
        "license_origin": "NVIDIA AECO CityMassing Demo Pack supplied locally by user",
        "assets": list(ASSETS),
        "file_count": len(copied),
        "bytes": sum(int(item["bytes"]) for item in copied),
        "files": copied,
    }
    (args.output_root / "library-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("assets", "file_count", "bytes")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
