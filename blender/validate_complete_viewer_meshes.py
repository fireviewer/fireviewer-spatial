"""Prove that the complete viewer GLB retained every unique source mesh."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from export_complete_viewer_glb import (
    GLB_NAME,
    RECEIPT_NAME,
    SCHEMA,
    STATUS,
    _canonical_bytes,
    _read_glb_json,
    _sha256_file,
)
from render_simple_zone_gallery import BLEND_NAME


class CompleteViewerMeshError(RuntimeError):
    pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def require_mesh_coverage(source_unique_meshes: int, viewer_meshes: int) -> None:
    if source_unique_meshes < 1:
        raise CompleteViewerMeshError("La scène Blender ne contient aucun mesh source")
    if viewer_meshes < source_unique_meshes:
        raise CompleteViewerMeshError(
            "Meshes viewer incomplets: "
            f"{viewer_meshes} exportés < {source_unique_meshes} meshes source uniques"
        )


def _unique_source_mesh_count(bpy: Any) -> int:
    identities: set[int] = set()
    for obj in bpy.context.scene.objects:
        if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None:
            identities.add(int(obj.data.as_pointer()))
    for instance in bpy.context.evaluated_depsgraph_get().object_instances:
        evaluated = getattr(instance, "object", None)
        source = getattr(evaluated, "original", evaluated)
        if source is not None and getattr(source, "type", None) == "MESH" and getattr(source, "data", None) is not None:
            identities.add(int(source.data.as_pointer()))
    return len(identities)


def validate_complete_viewer_meshes(job_root: Path | str) -> Path:
    root = Path(job_root).resolve(strict=True)
    blend = root / BLEND_NAME
    glb = root / GLB_NAME
    receipt_path = root / RECEIPT_NAME
    if not blend.is_file() or not glb.is_file() or not receipt_path.is_file():
        raise CompleteViewerMeshError("zone.blend, viewer.glb ou reçu viewer absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA or receipt.get("status") != STATUS:
        raise CompleteViewerMeshError("Reçu viewer invalide avant contrôle des meshes")
    if receipt.get("viewer", {}).get("sha256") != _sha256_file(glb):
        raise CompleteViewerMeshError("Le GLB a changé avant contrôle des meshes")
    try:
        import bpy  # type: ignore
    except ImportError as error:  # pragma: no cover
        raise CompleteViewerMeshError("Le contrôle des meshes doit s'exécuter dans Blender") from error
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    if "FINISHED" not in result:
        raise CompleteViewerMeshError("Impossible d'ouvrir zone.blend pour le contrôle viewer")
    source_unique = _unique_source_mesh_count(bpy)
    gltf = _read_glb_json(glb)
    meshes = gltf.get("meshes")
    if not isinstance(meshes, list):
        raise CompleteViewerMeshError("Table de meshes GLB absente")
    require_mesh_coverage(source_unique, len(meshes))
    completeness = receipt.get("completeness")
    if not isinstance(completeness, dict):
        raise CompleteViewerMeshError("Reçu viewer sans bloc de complétude")
    completeness.update(
        {
            "source_unique_mesh_count": source_unique,
            "viewer_mesh_count": len(meshes),
            "mesh_coverage": "complete",
        }
    )
    receipt["completeness"] = completeness
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    _write_json(receipt_path, receipt)
    return receipt_path


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        values = values[values.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    result = validate_complete_viewer_meshes(_parse_arguments(argv).job_root)
    print(json.dumps({"viewer_mesh_receipt": str(result)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
