"""Inspect USDZ tree assets with the OpenUSD runtime bundled in Omniverse Kit.

The script is designed for ``kit.exe <experience>.kit --exec``. Input paths and
the JSON output path are supplied through environment variables so asset names
with spaces remain unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import omni.kit.app
from pxr import Usd, UsdGeom


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_asset(path: Path) -> dict[str, object]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"OpenUSD could not open {path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ValueError(f"USDZ has no valid default prim: {path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    aligned_range = cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
    minimum = aligned_range.GetMin()
    maximum = aligned_range.GetMax()
    if aligned_range.IsEmpty():
        raise ValueError(f"USDZ default prim has no geometric bounds: {path}")
    prims = list(stage.Traverse())
    mesh_count = sum(1 for prim in prims if prim.IsA(UsdGeom.Mesh))
    if mesh_count <= 0:
        raise ValueError(f"USDZ contains no mesh: {path}")
    return {
        "source_path": str(path),
        "source_name": path.name,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "default_prim": str(default_prim.GetPath()),
        "default_prim_type": default_prim.GetTypeName(),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)).upper(),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "range_min": [float(minimum[index]) for index in range(3)],
        "range_max": [float(maximum[index]) for index in range(3)],
        "prim_count": len(prims),
        "mesh_count": mesh_count,
    }


def main() -> int:
    raw_paths = json.loads(os.environ["FIREVIEWER_TREE_ASSET_PATHS_JSON"])
    output = Path(os.environ["FIREVIEWER_TREE_ASSET_INSPECTION_OUTPUT"]).resolve()
    if not isinstance(raw_paths, list) or len(raw_paths) != 6:
        raise ValueError("Exactly six USDZ paths are required")
    paths = [Path(str(raw_path)).resolve() for raw_path in raw_paths]
    if len({str(path).casefold() for path in paths}) != 6:
        raise ValueError("The six USDZ paths must be distinct")
    record = {
        "schema": "fireviewer.usdz-tree-asset-inspection.v1",
        "assets": [inspect_asset(path) for path in paths],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


try:
    exit_code = main()
except Exception as error:
    print(f"USDZ tree asset inspection failed: {error}")
    exit_code = 1

omni.kit.app.get_app().post_quit(exit_code)
