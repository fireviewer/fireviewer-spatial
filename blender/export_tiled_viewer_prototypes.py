"""Export the sealed shared prototypes without materializing zone instances.

The script opens the already sealed ``zone.blend`` and exports each normalized
mesh datablock with identity nodes.  Its object/collection transform is recorded
separately so the package builder can compose it into every sealed instance,
matching Blender's evaluated PointInstancer without materializing the scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PLAN_SCHEMA = "fireviewer.tiled-prototype-export-plan.v1"
RESULT_SCHEMA = "fireviewer.tiled-prototype-export.v1"
RESULT_NAME = "prototype-export.v1.json"


class TiledPrototypeExportError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TiledPrototypeExportError(f"{label} JSON invalide") from error
    if not isinstance(value, dict):
        raise TiledPrototypeExportError(f"{label} doit être un objet JSON")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def _inside(root: Path, candidate: Path, label: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise TiledPrototypeExportError(f"{label} sort du job") from error
    return resolved


def _relative_output(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TiledPrototypeExportError("Chemin de prototype invalide")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise TiledPrototypeExportError("Chemin de prototype non confiné")
    return _inside(root, root.joinpath(*relative.parts), "prototype")


def _collect_prototype_mesh(bpy: Any, root: Any) -> tuple[list[Any], list[float]]:
    from mathutils import Matrix

    sources: list[tuple[Any, Any]] = []

    def walk(obj: Any, parent_matrix: Any, collection_stack: tuple[int, ...]) -> None:
        matrix = parent_matrix @ obj.matrix_local
        if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None:
            sources.append((obj, matrix.copy()))

        collection = getattr(obj, "instance_collection", None)
        if getattr(obj, "instance_type", None) == "COLLECTION" and collection is not None:
            pointer = int(collection.as_pointer())
            if pointer in collection_stack:
                raise TiledPrototypeExportError("Cycle de collection dans un prototype")
            offset = Matrix.Translation(-collection.instance_offset)
            for child in collection.objects:
                if child.parent is None:
                    walk(child, matrix @ offset, (*collection_stack, pointer))

        for child in obj.children:
            walk(child, matrix, collection_stack)

    walk(root, Matrix.Identity(4), ())
    if len(sources) != 1:
        raise TiledPrototypeExportError(
            f"Prototype {root.name} doit résoudre exactement un mesh, obtenu={len(sources)}"
        )
    source, transform = sources[0]
    mesh = source.data.copy()
    mesh.update()
    clone = bpy.data.objects.new(f"FV_{source.name}", mesh)
    clone.matrix_world = Matrix.Identity(4)
    bpy.context.scene.collection.objects.link(clone)
    return [clone], [float(value) for row in transform for value in row]


def _export_selected_glb(bpy: Any, objects: Sequence[Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.part.glb")
    temporary.unlink(missing_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.hide_render = False
        obj.hide_viewport = False
        obj.hide_set(False)
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    operator = bpy.ops.export_scene.gltf
    supported = {prop.identifier for prop in operator.get_rna_type().properties}
    options: dict[str, Any] = {
        "filepath": str(temporary),
        "check_existing": False,
        "export_format": "GLB",
        "use_selection": True,
    }
    for name, value in {
        "export_yup": True,
        "export_cameras": False,
        "export_lights": False,
        "export_materials": "EXPORT",
        "export_texcoords": True,
        "export_normals": True,
        "export_apply": False,
        "export_attributes": True,
    }.items():
        if name in supported:
            options[name] = value
    result = operator(**options)
    if "FINISHED" not in result or not temporary.is_file() or temporary.stat().st_size <= 20:
        temporary.unlink(missing_ok=True)
        raise TiledPrototypeExportError("Export GLB de prototype incomplet")
    os.replace(temporary, destination)


def _remove_exported(bpy: Any, objects: Sequence[Any]) -> None:
    meshes = [obj.data for obj in objects if getattr(obj, "data", None) is not None]
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def export_prototypes(job_root: Path | str, plan_path: Path | str) -> Path:
    import bpy  # type: ignore

    root = Path(job_root).resolve(strict=True)
    plan_file = _inside(root, Path(plan_path), "plan prototypes")
    plan = _load_json(plan_file, "plan prototypes")
    source = plan.get("zone_blend")
    prototypes = plan.get("prototypes")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or not isinstance(source, Mapping)
        or not isinstance(prototypes, list)
    ):
        raise TiledPrototypeExportError("Plan de prototypes invalide")
    source_path = source.get("path")
    if not isinstance(source_path, str) or "\\" in source_path:
        raise TiledPrototypeExportError("Chemin zone.blend invalide")
    relative_source = PurePosixPath(source_path)
    if relative_source.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_source.parts
    ):
        raise TiledPrototypeExportError("Chemin zone.blend non confiné")
    blend_path = _inside(
        root, root.joinpath(*relative_source.parts), "zone.blend"
    )
    if (
        not blend_path.is_file()
        or source.get("byte_count") != blend_path.stat().st_size
        or source.get("sha256") != _sha256_file(blend_path)
    ):
        raise TiledPrototypeExportError("zone.blend diffère du plan scellé")

    result = bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
    if "FINISHED" not in result:
        raise TiledPrototypeExportError("Ouverture de zone.blend impossible")

    rows: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for raw in prototypes:
        if not isinstance(raw, Mapping):
            raise TiledPrototypeExportError("Prototype du plan invalide")
        prototype_id = raw.get("id")
        identifier = raw.get("identifier")
        if (
            not isinstance(prototype_id, str)
            or not prototype_id
            or prototype_id in observed_ids
            or not isinstance(identifier, str)
            or not identifier
        ):
            raise TiledPrototypeExportError("Identité de prototype invalide")
        observed_ids.add(prototype_id)
        source_root = bpy.data.objects.get(identifier)
        if source_root is None:
            raise TiledPrototypeExportError(f"Prototype absent de zone.blend: {identifier}")
        destination = _relative_output(root, raw.get("output"))
        exported, prototype_transform = _collect_prototype_mesh(bpy, source_root)
        try:
            _export_selected_glb(bpy, exported, destination)
        finally:
            _remove_exported(bpy, exported)
        rows.append(
            {
                "id": prototype_id,
                "family": raw.get("family"),
                "asset_id": raw.get("asset_id"),
                "identifier": identifier,
                "output": raw.get("output"),
                "mesh_count": len(exported),
                "prototype_transform_z_up": prototype_transform,
                "byte_count": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )

    receipt = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "zone_blend": dict(source),
        "prototype_count": len(rows),
        "prototypes": rows,
    }
    output = plan_file.parent / RESULT_NAME
    _write_json(output, receipt)
    return output


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        values = values[values.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_arguments(argv)
    receipt = export_prototypes(options.job_root, options.plan)
    print(json.dumps({"prototype_export_receipt": str(receipt)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - Blender entrypoint
    raise SystemExit(main())
