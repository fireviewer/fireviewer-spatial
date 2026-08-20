"""Export one complete, browser-ready GLB from the sealed FireViewer zone.

The exporter reuses the already packed ``zone.blend`` produced from the sealed
OpenUSD scene. PointInstancer content is materialized as lightweight Blender
objects sharing prototype mesh datablocks, then exported with
EXT_mesh_gpu_instancing. Publication is fail-closed: family instance counts,
renderable mesh coverage and embedded textures are checked against the sealed
zone before a viewer receipt is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

SCHEMA = "fireviewer.complete-viewer-scene.v1"
STATUS = "complete"
GLB_NAME = "viewer.glb"
RECEIPT_NAME = "viewer-scene.v1.json"
BLEND_NAME = "zone.blend"
FAMILY_MARKERS = {
    "buildings": "Buildings",
    "trees": "Trees",
    "context_assets": "ContextAssets",
}
FAMILY_ROOTS = {key: f"FireViewer_{value}" for key, value in FAMILY_MARKERS.items()}


class CompleteViewerExportError(RuntimeError):
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompleteViewerExportError(f"{label} JSON invalide: {error}") from error
    if not isinstance(value, dict):
        raise CompleteViewerExportError(f"{label} doit être un objet JSON")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def _require_root(value: Path | str) -> Path:
    lexical = PureWindowsPath(str(value))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise CompleteViewerExportError("Le viewer doit rester sur D: sous Windows")
    root = Path(value).resolve(strict=True)
    if os.name == "nt" and root.drive.upper() != "D:":
        raise CompleteViewerExportError("Le viewer doit rester sur D: sous Windows")
    return root


def _expected_counts(receipt: Mapping[str, Any]) -> dict[str, int]:
    raw = {
        "buildings": receipt.get("building_count"),
        "trees": receipt.get("tree_count"),
        "context_assets": receipt.get("context_asset_count", 0),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in raw.values()
    ):
        raise CompleteViewerExportError("Comptages scellés de zone invalides")
    return {key: int(value) for key, value in raw.items()}


def _family_from_instance(instance: Any) -> str | None:
    names: list[str] = []
    for value in (
        getattr(instance, "parent", None),
        getattr(instance, "object", None),
        getattr(getattr(instance, "object", None), "original", None),
    ):
        name = getattr(value, "name", None)
        if isinstance(name, str):
            names.append(name)
    joined = " ".join(names)
    for family, marker in FAMILY_MARKERS.items():
        if marker in joined:
            return family
    return None


def _snapshot_instances(bpy: Any) -> list[tuple[str, Any, Any]]:
    """Copy transient depsgraph instance data while each RNA handle is valid.

    ``Depsgraph.object_instances`` yields ephemeral ``DepsgraphObjectInstance``
    wrappers. Blender may invalidate a wrapper as soon as the iterator advances,
    so materializing the iterator with ``list(...)`` and reading it afterwards is
    unsafe. Only stable Blender ID datablocks plus a copied matrix leave this loop.
    """

    snapshots: list[tuple[str, Any, Any]] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for instance in depsgraph.object_instances:
        if not bool(getattr(instance, "is_instance", False)):
            continue
        family = _family_from_instance(instance)
        if family is None:
            continue
        evaluated = getattr(instance, "object", None)
        source = getattr(evaluated, "original", evaluated)
        if source is None or getattr(source, "type", None) != "MESH":
            continue
        matrix = instance.matrix_world.copy()
        snapshots.append((family, source, matrix))
    return snapshots


def _unique_source_mesh_count(bpy: Any) -> int:
    """Count unique source mesh datablocks before instance materialization."""

    identities: set[int] = set()
    for obj in bpy.context.scene.objects:
        if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None:
            identities.add(int(obj.data.as_pointer()))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for instance in depsgraph.object_instances:
        evaluated = getattr(instance, "object", None)
        source = getattr(evaluated, "original", evaluated)
        if (
            source is not None
            and getattr(source, "type", None) == "MESH"
            and getattr(source, "data", None) is not None
        ):
            identities.add(int(source.data.as_pointer()))
    if not identities:
        raise CompleteViewerExportError("La scène Blender ne contient aucun mesh source")
    return len(identities)


def _materialize_instances(bpy: Any, expected: Mapping[str, int]) -> dict[str, int]:
    collection = bpy.context.scene.collection
    roots: dict[str, Any] = {}
    for family, name in FAMILY_ROOTS.items():
        root = bpy.data.objects.new(name, None)
        root["fireviewer_family"] = family
        collection.objects.link(root)
        roots[family] = root
    counts = {family: 0 for family in FAMILY_MARKERS}

    # Never retain DepsgraphObjectInstance handles. Snapshot the family, stable
    # source Object and world matrix before the depsgraph iterator advances, then
    # mutate the Blender scene only after the snapshot is complete.
    snapshots = _snapshot_instances(bpy)
    for family, source, matrix_world in snapshots:
        clone = source.copy()
        clone.data = source.data
        clone.animation_data_clear()
        clone.parent = roots[family]
        clone.matrix_world = matrix_world
        clone.hide_render = False
        clone.hide_viewport = False
        clone.hide_set(False)
        clone.name = f"FV_{family}_{counts[family]:08d}_{source.name}"
        collection.objects.link(clone)
        counts[family] += 1

    for obj in list(bpy.context.scene.objects):
        if getattr(obj, "type", None) == "POINTCLOUD" and any(
            marker in obj.name for marker in FAMILY_MARKERS.values()
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
    if counts != dict(expected):
        raise CompleteViewerExportError(
            f"Instances viewer incomplètes: attendu={dict(expected)}, obtenu={counts}"
        )
    return counts


def _scene_metrics(bpy: Any) -> tuple[int, int]:
    images: set[int] = set()
    visited: set[int] = set()
    mesh_objects = 0
    unsupported: list[str] = []

    def visit_tree(tree: Any) -> None:
        if tree is None or id(tree) in visited:
            return
        visited.add(id(tree))
        for node in tree.nodes:
            image = getattr(node, "image", None)
            if image is not None:
                images.add(id(image))
            visit_tree(getattr(node, "node_tree", None))

    for obj in bpy.context.scene.objects:
        if bool(getattr(obj, "hide_render", False)):
            continue
        kind = getattr(obj, "type", None)
        if kind == "MESH":
            mesh_objects += 1
            for material in getattr(getattr(obj, "data", None), "materials", ()) or ():
                if material is not None:
                    visit_tree(getattr(material, "node_tree", None))
        elif kind in {"CURVE", "SURFACE", "META", "FONT", "VOLUME"}:
            unsupported.append(f"{obj.name}:{kind}")
    if unsupported:
        raise CompleteViewerExportError(
            "Contenu visible non exportable en glTF: "
            + ", ".join(sorted(unsupported)[:20])
        )
    if mesh_objects < 1:
        raise CompleteViewerExportError("Aucun mesh visible dans la scène viewer")
    return mesh_objects, len(images)


def _export_glb(bpy: Any, output: Path) -> None:
    temporary = output.with_name(f".{output.stem}.part.glb")
    temporary.unlink(missing_ok=True)
    operator = bpy.ops.export_scene.gltf
    supported = {prop.identifier for prop in operator.get_rna_type().properties}
    if "export_gpu_instances" not in supported:
        raise CompleteViewerExportError("Blender ne fournit pas EXT_mesh_gpu_instancing")
    options: dict[str, Any] = {
        "filepath": str(temporary),
        "check_existing": False,
        "export_format": "GLB",
    }
    for name, value in {
        "use_visible": True,
        "export_yup": True,
        "export_cameras": False,
        "export_lights": False,
        "export_materials": "EXPORT",
        "export_texcoords": True,
        "export_normals": True,
        "export_apply": False,
        "export_gpu_instances": True,
        "export_gn_mesh": True,
        "export_attributes": True,
    }.items():
        if name in supported:
            options[name] = value
    result = operator(**options)
    if (
        "FINISHED" not in result
        or not temporary.is_file()
        or temporary.stat().st_size <= 20
    ):
        temporary.unlink(missing_ok=True)
        raise CompleteViewerExportError("Blender n'a pas écrit le GLB viewer")
    os.replace(temporary, output)


def _read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20:
        raise CompleteViewerExportError("GLB viewer tronqué")
    magic, version, declared = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared != len(data):
        raise CompleteViewerExportError("En-tête GLB viewer invalide")
    cursor = 12
    payload: dict[str, Any] | None = None
    has_bin = False
    while cursor + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, cursor)
        cursor += 8
        end = cursor + length
        if end > len(data):
            raise CompleteViewerExportError("Chunk GLB viewer tronqué")
        chunk = data[cursor:end]
        cursor = end
        if kind == 0x4E4F534A:
            value = json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
            if not isinstance(value, dict):
                raise CompleteViewerExportError("JSON GLB viewer invalide")
            payload = value
        elif kind == 0x004E4942:
            has_bin = True
    if payload is None or not has_bin:
        raise CompleteViewerExportError("GLB viewer sans JSON ou buffer binaire")
    return payload


def _descendants(nodes: Sequence[Any], root: int) -> set[int]:
    pending = [root]
    found: set[int] = set()
    while pending:
        index = pending.pop()
        if index in found or not 0 <= index < len(nodes):
            continue
        found.add(index)
        node = nodes[index]
        if isinstance(node, Mapping) and isinstance(node.get("children"), list):
            pending.extend(
                item
                for item in node["children"]
                if isinstance(item, int) and not isinstance(item, bool)
            )
    return found


def _node_instance_count(node: Mapping[str, Any], accessors: Sequence[Any]) -> int:
    extensions = node.get("extensions")
    extension = (
        extensions.get("EXT_mesh_gpu_instancing")
        if isinstance(extensions, Mapping)
        else None
    )
    if isinstance(extension, Mapping):
        attributes = extension.get("attributes")
        if not isinstance(attributes, Mapping) or not attributes:
            raise CompleteViewerExportError("Extension GPU instancing invalide")
        counts: set[int] = set()
        for index in attributes.values():
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(accessors)
                or not isinstance(accessors[index], Mapping)
            ):
                raise CompleteViewerExportError("Accessor GPU instancing invalide")
            count = accessors[index].get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise CompleteViewerExportError("Comptage GPU instancing invalide")
            counts.add(count)
        if len(counts) != 1:
            raise CompleteViewerExportError(
                "Attributs GPU instancing de tailles différentes"
            )
        return next(iter(counts))
    return 1 if isinstance(node.get("mesh"), int) else 0


def _validate_gltf_payload(
    gltf: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
    source_images: int,
    source_meshes: int,
) -> dict[str, Any]:
    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])
    accessors = gltf.get("accessors", [])
    images = gltf.get("images", [])
    buffers = gltf.get("buffers", [])
    buffer_views = gltf.get("bufferViews", [])
    textures = gltf.get("textures", [])
    materials = gltf.get("materials", [])
    if (
        not all(
            isinstance(value, list)
            for value in (
                nodes,
                meshes,
                accessors,
                images,
                buffers,
                buffer_views,
                textures,
                materials,
            )
        )
        or not meshes
        or not buffers
        or isinstance(source_images, bool)
        or not isinstance(source_images, int)
        or source_images < 0
        or source_meshes < 1
    ):
        raise CompleteViewerExportError("Tables GLB viewer invalides")
    if any(
        not isinstance(item, Mapping) or item.get("uri") is not None
        for item in buffers
    ):
        raise CompleteViewerExportError("Le GLB viewer référence un buffer externe")
    if any(
        not isinstance(item, Mapping)
        or isinstance(item.get("buffer"), bool)
        or not isinstance(item.get("buffer"), int)
        or not 0 <= item["buffer"] < len(buffers)
        for item in buffer_views
    ):
        raise CompleteViewerExportError("Le GLB viewer contient un bufferView invalide")
    if any(
        not isinstance(item, Mapping)
        or item.get("uri") is not None
        or isinstance(item.get("bufferView"), bool)
        or not isinstance(item.get("bufferView"), int)
        or not 0 <= item["bufferView"] < len(buffer_views)
        for item in images
    ):
        raise CompleteViewerExportError("Le GLB viewer contient une texture externe")
    if source_images > 0 and not images:
        raise CompleteViewerExportError(
            "Le GLB viewer ne contient aucune image embarquée malgré les images source"
        )
    referenced_images: set[int] = set()
    for texture in textures:
        if not isinstance(texture, Mapping):
            raise CompleteViewerExportError("Table de textures GLB invalide")
        candidates: list[Any] = []
        if "source" in texture:
            candidates.append(texture.get("source"))
        extensions = texture.get("extensions")
        if isinstance(extensions, Mapping):
            for extension_name in (
                "KHR_texture_basisu",
                "EXT_texture_webp",
                "MSFT_texture_dds",
            ):
                extension = extensions.get(extension_name)
                if isinstance(extension, Mapping) and "source" in extension:
                    candidates.append(extension.get("source"))
        if not candidates:
            raise CompleteViewerExportError(
                "Une texture GLB ne référence aucune image embarquée"
            )
        for image_index in candidates:
            if (
                isinstance(image_index, bool)
                or not isinstance(image_index, int)
                or not 0 <= image_index < len(images)
            ):
                raise CompleteViewerExportError(
                    "Une texture GLB référence une image invalide"
                )
            referenced_images.add(image_index)
    if source_images > 0 and not textures:
        raise CompleteViewerExportError(
            "Le GLB viewer ne contient aucune texture malgré les images source"
        )

    material_texture_references = 0

    def validate_texture_info(value: Any, label: str) -> None:
        nonlocal material_texture_references
        if not isinstance(value, Mapping):
            raise CompleteViewerExportError(f"Référence de texture {label} invalide")
        texture_index = value.get("index")
        if (
            isinstance(texture_index, bool)
            or not isinstance(texture_index, int)
            or not 0 <= texture_index < len(textures)
        ):
            raise CompleteViewerExportError(f"Index de texture {label} invalide")
        material_texture_references += 1

    for material_index, material in enumerate(materials):
        if not isinstance(material, Mapping):
            raise CompleteViewerExportError("Table de matériaux GLB invalide")
        pbr = material.get("pbrMetallicRoughness")
        if pbr is not None:
            if not isinstance(pbr, Mapping):
                raise CompleteViewerExportError("Matériau PBR GLB invalide")
            for key in ("baseColorTexture", "metallicRoughnessTexture"):
                if key in pbr:
                    validate_texture_info(
                        pbr[key], f"materials[{material_index}].{key}"
                    )
        for key in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            if key in material:
                validate_texture_info(
                    material[key], f"materials[{material_index}].{key}"
                )
    names = {
        item.get("name"): index
        for index, item in enumerate(nodes)
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    actual: dict[str, int] = {}
    for family, expected in expected_counts.items():
        root = names.get(FAMILY_ROOTS[family])
        if expected == 0 and root is None:
            actual[family] = 0
            continue
        if not isinstance(root, int):
            raise CompleteViewerExportError(
                f"Racine {FAMILY_ROOTS[family]} absente du GLB"
            )
        count = sum(
            _node_instance_count(nodes[index], accessors)
            for index in _descendants(nodes, root)
            if index != root and isinstance(nodes[index], Mapping)
        )
        actual[family] = count
        if count != expected:
            raise CompleteViewerExportError(
                f"Viewer incomplet pour {family}: attendu={expected}, obtenu={count}"
            )
    return {
        "family_instance_counts": actual,
        "mesh_count": len(meshes),
        "node_count": len(nodes),
        "image_count": len(images),
        "texture_count": len(textures),
        "referenced_image_count": len(referenced_images),
        "material_texture_reference_count": material_texture_references,
        "source_image_datablock_count": source_images,
        "source_to_exported_image_delta": source_images - len(images),
        "material_count": len(materials),
        "extensions_used": (
            sorted(
                item
                for item in gltf.get("extensionsUsed", [])
                if isinstance(item, str)
            )
            if isinstance(gltf.get("extensionsUsed"), list)
            else []
        ),
    }


def export_complete_viewer(job_root: Path | str) -> Path:
    root = _require_root(job_root)
    stage = root / "zone.usda"
    blend = root / BLEND_NAME
    zone_receipt_path = root / "zone.done.json"
    if not stage.is_file() or not blend.is_file() or not zone_receipt_path.is_file():
        raise CompleteViewerExportError("zone.usda, zone.blend ou zone.done.json absent")
    zone_receipt = _load_json(zone_receipt_path, "reçu de zone")
    expected = _expected_counts(zone_receipt)
    try:
        import bpy  # type: ignore
    except ImportError as error:  # pragma: no cover
        raise CompleteViewerExportError(
            "L'export viewer doit s'exécuter dans Blender"
        ) from error

    print("FIREVIEWER_VIEWER_STAGE open_blend", flush=True)
    result = bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    if "FINISHED" not in result:
        raise CompleteViewerExportError("Ouverture de zone.blend impossible")

    print("FIREVIEWER_VIEWER_STAGE inspect_source_meshes", flush=True)
    source_unique_meshes = _unique_source_mesh_count(bpy)
    print("FIREVIEWER_VIEWER_STAGE materialize_instances", flush=True)
    materialized = _materialize_instances(bpy, expected)
    source_meshes, source_images = _scene_metrics(bpy)
    output = root / GLB_NAME
    print("FIREVIEWER_VIEWER_STAGE export_glb", flush=True)
    _export_glb(bpy, output)
    print("FIREVIEWER_VIEWER_STAGE validate_glb", flush=True)
    summary = _validate_gltf_payload(
        _read_glb_json(output),
        expected_counts=expected,
        source_images=source_images,
        source_meshes=source_meshes,
    )
    if summary["family_instance_counts"] != materialized:
        raise CompleteViewerExportError(
            "Le GLB diverge de la matérialisation Blender"
        )
    if int(summary["mesh_count"]) < source_unique_meshes:
        raise CompleteViewerExportError(
            "Meshes viewer incomplets: "
            f"{summary['mesh_count']} exportés < {source_unique_meshes} meshes source uniques"
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "zone_id": zone_receipt.get("zone_id"),
        "build_id": zone_receipt.get("build_id"),
        "source": {
            "entry_stage": "zone.usda",
            "entry_stage_sha256": _sha256_file(stage),
            "standalone_blend": BLEND_NAME,
            "standalone_blend_sha256": _sha256_file(blend),
            "zone_receipt": "zone.done.json",
            "zone_receipt_sha256": _sha256_file(zone_receipt_path),
        },
        "viewer": {
            "file": GLB_NAME,
            "byte_count": output.stat().st_size,
            "sha256": _sha256_file(output),
            "format": "glTF 2.0 binary",
            "external_dependencies": 0,
        },
        "completeness": {
            "policy": "fail_closed_exact_visual_scene",
            "terrain_and_non_instanced_meshes_required": True,
            "source_mesh_object_count": source_meshes,
            "source_unique_mesh_count": source_unique_meshes,
            "viewer_mesh_count": int(summary["mesh_count"]),
            "mesh_coverage": "complete",
            "source_image_count": source_images,
            "source_image_count_basis": "blender_material_node_image_datablocks",
            **summary,
        },
    }
    if not isinstance(payload["zone_id"], str) or not isinstance(
        payload["build_id"], str
    ):
        raise CompleteViewerExportError("Identité scellée de zone absente")
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()
    _write_json(root / RECEIPT_NAME, payload)
    return root / RECEIPT_NAME


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        values = values[values.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    receipt = export_complete_viewer(_parse_arguments(argv).job_root)
    print(
        json.dumps(
            {"viewer_receipt": str(receipt)},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
