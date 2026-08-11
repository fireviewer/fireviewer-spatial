"""Inspect composed terrain material bindings with Omniverse OpenUSD.

Designed for ``kit.exe <experience>.kit --no-window --exec``.  The terrain
payload path is supplied through ``FIREVIEWER_TERRAIN_PAYLOAD``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import omni.kit.app
from pxr import Usd, UsdGeom


def asset_path_record(value: object) -> dict[str, str]:
    return {
        "authored": str(getattr(value, "path", value)),
        "resolved": str(getattr(value, "resolvedPath", "")),
    }


def main() -> int:
    payload = Path(os.environ["FIREVIEWER_TERRAIN_PAYLOAD"]).resolve()
    stage = Usd.Stage.Open(str(payload), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"OpenUSD could not open {payload}")

    records: list[dict[str, object]] = []
    traversed = list(stage.Traverse())
    for prim in traversed:
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName() != "Surface":
            continue
        binding = prim.GetRelationship("material:binding")
        targets = [str(path) for path in binding.GetTargets()]
        material = stage.GetPrimAtPath(targets[0]) if len(targets) == 1 else Usd.Prim()
        texture = material.GetChild("Texture") if material and material.IsValid() else Usd.Prim()
        file_attribute = texture.GetAttribute("inputs:file") if texture and texture.IsValid() else None
        file_value = file_attribute.Get() if file_attribute else None
        st_attribute = prim.GetAttribute("primvars:st")
        st_values = st_attribute.Get() or []
        records.append(
            {
                "mesh": str(prim.GetPath()),
                "active": prim.IsActive(),
                "loaded": prim.IsLoaded(),
                "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility()),
                "orientation": str(UsdGeom.Mesh(prim).GetOrientationAttr().Get()),
                "material_targets": targets,
                "material_valid": bool(material and material.IsValid()),
                "texture_shader_valid": bool(texture and texture.IsValid()),
                "texture_file": asset_path_record(file_value) if file_value is not None else None,
                "st_count": len(st_values),
                "st_interpolation": str(st_attribute.GetMetadata("interpolation")),
                "point_count": len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []),
            }
        )

    print(
        json.dumps(
            {
                "payload": str(payload),
                "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
                "traversed_prims": [
                    {"path": str(prim.GetPath()), "type": prim.GetTypeName(), "loaded": prim.IsLoaded()}
                    for prim in traversed
                ],
                "surface_count": len(records),
                "surfaces": records,
            },
            indent=2,
        )
    )
    if not records:
        raise ValueError("No composed terrain Surface meshes found")
    return 0


try:
    exit_code = main()
except Exception as error:
    print(f"Terrain material inspection failed: {error}")
    exit_code = 1

omni.kit.app.get_app().post_quit(exit_code)
