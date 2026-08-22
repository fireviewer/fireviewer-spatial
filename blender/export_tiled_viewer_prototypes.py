"""Export the sealed shared prototypes without materializing zone instances.

The script opens the already sealed ``zone.blend`` and exports each normalized
mesh datablock with identity nodes.  Its object/collection transform is recorded
separately so the package builder can compose it into every sealed instance,
matching Blender's evaluated PointInstancer without materializing the scene.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PLAN_SCHEMA = "fireviewer.tiled-prototype-export-plan.v1"
RESULT_SCHEMA = "fireviewer.tiled-prototype-export.v1"
RESULT_NAME = "prototype-export.v1.json"
FAMILY_SCOPE_NAMES = {
    "buildings": "Buildings",
    "trees": "Trees",
    "context_assets": "ContextAssets",
}


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


def _has_ancestor_named(obj: Any, name: str) -> bool:
    parent = getattr(obj, "parent", None)
    while parent is not None:
        if getattr(parent, "name", None) == name:
            return True
        parent = getattr(parent, "parent", None)
    return False


def _matches_imported_name(name: object, identifier: str) -> bool:
    if not isinstance(name, str):
        return False
    if name == identifier or identifier.startswith(name):
        return True
    base, separator, suffix = name.rpartition(".")
    return (
        separator == "."
        and len(suffix) == 3
        and suffix.isdigit()
        and identifier.startswith(base)
    )


def _root_for_asset_metadata(obj: Any, identifier: str) -> Any | None:
    current = obj
    while current is not None:
        if _matches_imported_name(getattr(current, "name", None), identifier):
            return current
        current = getattr(current, "parent", None)
    return None


def _resolve_prototype_root(
    bpy: Any, *, family: object, asset_id: object, identifier: str
) -> Any:
    scope_name = FAMILY_SCOPE_NAMES.get(family)
    if scope_name is None or not isinstance(asset_id, str) or not asset_id:
        raise TiledPrototypeExportError("Identité de prototype invalide")
    candidates_by_identity: dict[int, Any] = {}
    for obj in bpy.data.objects:
        if obj.get("fireviewer:asset_id") != asset_id:
            continue
        root = _root_for_asset_metadata(obj, identifier)
        if root is not None:
            candidates_by_identity[id(root)] = root
    candidates = list(candidates_by_identity.values())
    scoped = [root for root in candidates if _has_ancestor_named(root, scope_name)]
    if len(scoped) == 1:
        return scoped[0]
    if len(scoped) > 1 or not candidates:
        raise TiledPrototypeExportError(
            "Prototype absent ou ambigu dans zone.blend: "
            f"family={family}, asset_id={asset_id}, identifier={identifier}, "
            f"obtenu={len(scoped) if scoped else len(candidates)}"
        )
    return min(
        candidates,
        key=lambda root: (
            getattr(root, "name", None) != identifier,
            str(getattr(root, "name", "")),
        ),
    )


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
    if not sources:
        raise TiledPrototypeExportError(
            f"Prototype {root.name} ne contient aucun mesh exportable"
        )
    prototype_transform = root.matrix_local.copy()
    inverse_prototype = prototype_transform.inverted_safe()
    clones: list[Any] = []
    for index, (source, transform) in enumerate(sources):
        mesh = source.data.copy()
        mesh.transform(inverse_prototype @ transform)
        mesh.update()
        clone = bpy.data.objects.new(f"FV_{index:03d}_{source.name}", mesh)
        clone.matrix_world = Matrix.Identity(4)
        bpy.context.scene.collection.objects.link(clone)
        clones.append(clone)
    return clones, [
        float(value) for row in prototype_transform for value in row
    ]


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


def _procedural_fallback(
    bpy: Any, *, family: object, prototype_id: str
) -> list[Any]:
    """Create a small identity-transformed replacement for one broken asset."""

    created: list[Any] = []

    def freeze(obj: Any) -> None:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)
        created.append(obj)

    if family == "trees":
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.35, depth=5.0)
        trunk = bpy.context.active_object
        trunk.name = f"FV_Fallback_{prototype_id}_trunk"
        trunk.location.z = 2.5
        freeze(trunk)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=2.2)
        crown = bpy.context.active_object
        crown.name = f"FV_Fallback_{prototype_id}_crown"
        crown.location.z = 6.0
        freeze(crown)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        body = bpy.context.active_object
        body.name = f"FV_Fallback_{prototype_id}"
        body.dimensions = (8.0, 6.0, 5.0) if family == "buildings" else (2.0, 2.0, 2.0)
        body.location.z = body.dimensions.z / 2.0
        freeze(body)
    return created


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
        source_root = _resolve_prototype_root(
            bpy,
            family=raw.get("family"),
            asset_id=raw.get("asset_id"),
            identifier=identifier,
        )
        destination = _relative_output(root, raw.get("output"))
        exported: list[Any] = []
        fallback_reason: str | None = None
        try:
            try:
                exported, prototype_transform = _collect_prototype_mesh(
                    bpy, source_root
                )
                _export_selected_glb(bpy, exported, destination)
            except Exception as error:
                _remove_exported(bpy, exported)
                exported = _procedural_fallback(
                    bpy,
                    family=raw.get("family"),
                    prototype_id=prototype_id,
                )
                prototype_transform = [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ]
                fallback_reason = f"{type(error).__name__}: {error}"
                _export_selected_glb(bpy, exported, destination)
        finally:
            _remove_exported(bpy, exported)
        row = {
            "id": prototype_id,
            "family": raw.get("family"),
            "asset_id": raw.get("asset_id"),
            "identifier": identifier,
            "output": raw.get("output"),
            "mesh_count": len(exported),
            "prototype_transform_z_up": prototype_transform,
            "byte_count": destination.stat().st_size,
        }
        if fallback_reason is not None:
            row["fallback"] = {
                "kind": "procedural_family_proxy",
                "reason": fallback_reason,
            }
        rows.append(row)

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
