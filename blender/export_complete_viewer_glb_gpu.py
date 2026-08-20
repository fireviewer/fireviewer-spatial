"""Fast exact-count FireViewer GLB exporter using Geometry Nodes GPU batches.

This exporter keeps every measured tree, building and context asset. It groups
logical instance transforms only by prototype and lets Blender serialize the
batches through EXT_mesh_gpu_instancing. Any count mismatch fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Blender's ``--python`` execution does not guarantee that the directory of the
# executed script is present on ``sys.path``.  The packaged image renames this
# module to ``export_complete_viewer_glb.py`` and keeps the previous exporter as
# a sibling named ``export_complete_viewer_glb_legacy.py``.  Resolve that sibling
# deterministically instead of depending on Blender's process-level PYTHONPATH.
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

try:  # In the Lightning image the known-good previous exporter is retained.
    import export_complete_viewer_glb_legacy as base
except ImportError:  # Local/CI import before image packaging.
    import export_complete_viewer_glb as base

SCHEMA = base.SCHEMA
STATUS = base.STATUS
GLB_NAME = base.GLB_NAME
RECEIPT_NAME = base.RECEIPT_NAME
BLEND_NAME = base.BLEND_NAME
FAMILY_MARKERS = base.FAMILY_MARKERS
FAMILY_ROOTS = base.FAMILY_ROOTS
CompleteViewerExportError = base.CompleteViewerExportError
ROTATION_ATTRIBUTE = "fv_instance_rotation"
SCALE_ATTRIBUTE = "fv_instance_scale"


def _source_key(source: Any) -> int:
    pointer = getattr(source, "as_pointer", None)
    if callable(pointer):
        value = int(pointer())
        if value:
            return value
    return id(source)


def _group_instance_snapshots(
    snapshots: Sequence[tuple[str, Any, Any]],
) -> list[tuple[str, Any, list[Any]]]:
    groups: dict[tuple[str, int], tuple[str, Any, list[Any]]] = {}
    order: list[tuple[str, int]] = []
    for family, source, matrix in snapshots:
        if family not in FAMILY_MARKERS:
            raise CompleteViewerExportError(f"Famille viewer inconnue: {family}")
        key = (family, _source_key(source))
        if key not in groups:
            groups[key] = (family, source, [])
            order.append(key)
        groups[key][2].append(matrix)
    return [groups[key] for key in order]


def _count_groups(groups: Sequence[tuple[str, Any, Sequence[Any]]]) -> dict[str, int]:
    counts = {family: 0 for family in FAMILY_MARKERS}
    for family, _source, matrices in groups:
        counts[family] += len(matrices)
    return counts


def _matrix_trs(matrix: Any) -> tuple[tuple[float, float, float], ...]:
    try:
        location, quaternion, scale = matrix.decompose()
        euler = quaternion.to_euler("XYZ")
    except Exception as error:
        raise CompleteViewerExportError("Transformation d'instance non décomposable") from error
    result = (
        tuple(float(value) for value in location),
        tuple(float(value) for value in euler),
        tuple(float(value) for value in scale),
    )
    if any(not math.isfinite(value) for vector in result for value in vector):
        raise CompleteViewerExportError("Transformation d'instance non finie")
    return result


def _set_vector_attribute(mesh: Any, name: str, values: Sequence[Sequence[float]]) -> None:
    attribute = mesh.attributes.new(name=name, type="FLOAT_VECTOR", domain="POINT")
    flat = [float(component) for vector in values for component in vector]
    setter = getattr(attribute.data, "foreach_set", None)
    if callable(setter):
        setter("vector", flat)
        return
    for item, value in zip(attribute.data, values, strict=True):  # pragma: no cover
        item.vector = value


def _geometry_sockets(group: Any) -> None:
    interface = getattr(group, "interface", None)
    if interface is not None and hasattr(interface, "new_socket"):
        interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
        return
    group.inputs.new("NodeSocketGeometry", "Geometry")  # pragma: no cover
    group.outputs.new("NodeSocketGeometry", "Geometry")  # pragma: no cover


def _named_vector(nodes: Any, name: str) -> Any:
    node = nodes.new("GeometryNodeInputNamedAttribute")
    node.data_type = "FLOAT_VECTOR"
    node.inputs["Name"].default_value = name
    return node


def _build_carrier(
    bpy: Any,
    *,
    root: Any,
    family: str,
    source: Any,
    matrices: Sequence[Any],
    index: int,
) -> None:
    transforms = [_matrix_trs(matrix) for matrix in matrices]
    translations = [value[0] for value in transforms]
    rotations = [value[1] for value in transforms]
    scales = [value[2] for value in transforms]
    collection = bpy.context.scene.collection

    proxy = bpy.data.objects.new(
        f"FV_Prototype_{family}_{index:04d}_{source.name}", source.data
    )
    proxy.hide_render = True
    proxy.hide_viewport = True
    collection.objects.link(proxy)

    mesh = bpy.data.meshes.new(f"FV_InstancePoints_{family}_{index:04d}")
    mesh.from_pydata(translations, [], [])
    mesh.update()
    _set_vector_attribute(mesh, ROTATION_ATTRIBUTE, rotations)
    _set_vector_attribute(mesh, SCALE_ATTRIBUTE, scales)
    carrier = bpy.data.objects.new(f"FV_{family}_GPU_{index:04d}_{source.name}", mesh)
    carrier.parent = root
    carrier["fireviewer_instance_count"] = len(matrices)
    collection.objects.link(carrier)

    group = bpy.data.node_groups.new(
        f"FV_GPU_InstanceGroup_{family}_{index:04d}", "GeometryNodeTree"
    )
    _geometry_sockets(group)
    nodes = group.nodes
    links = group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    instance_on_points = nodes.new("GeometryNodeInstanceOnPoints")
    object_info = nodes.new("GeometryNodeObjectInfo")
    object_info.inputs["Object"].default_value = proxy
    if "As Instance" in object_info.inputs:
        object_info.inputs["As Instance"].default_value = True
    rotation = _named_vector(nodes, ROTATION_ATTRIBUTE)
    scale = _named_vector(nodes, SCALE_ATTRIBUTE)
    links.new(group_input.outputs["Geometry"], instance_on_points.inputs["Points"])
    links.new(object_info.outputs["Geometry"], instance_on_points.inputs["Instance"])
    links.new(scale.outputs["Attribute"], instance_on_points.inputs["Scale"])
    try:
        euler_to_rotation = nodes.new("FunctionNodeEulerToRotation")
    except RuntimeError:  # pragma: no cover
        links.new(rotation.outputs["Attribute"], instance_on_points.inputs["Rotation"])
    else:
        links.new(rotation.outputs["Attribute"], euler_to_rotation.inputs["Euler"])
        links.new(euler_to_rotation.outputs["Rotation"], instance_on_points.inputs["Rotation"])
    links.new(instance_on_points.outputs["Instances"], group_output.inputs["Geometry"])
    modifier = carrier.modifiers.new(name="FireViewer GPU Instances", type="NODES")
    modifier.node_group = group


def _build_gpu_batches(
    bpy: Any,
    groups: Sequence[tuple[str, Any, Sequence[Any]]],
    expected: Mapping[str, int],
) -> None:
    counts = _count_groups(groups)
    if counts != dict(expected):
        raise CompleteViewerExportError(
            f"Instances viewer incomplètes avant export: attendu={dict(expected)}, obtenu={counts}"
        )
    collection = bpy.context.scene.collection
    roots: dict[str, Any] = {}
    for family, name in FAMILY_ROOTS.items():
        root = bpy.data.objects.new(name, None)
        root["fireviewer_family"] = family
        collection.objects.link(root)
        roots[family] = root
    indexes = {family: 0 for family in FAMILY_MARKERS}
    for family, source, matrices in groups:
        _build_carrier(
            bpy,
            root=roots[family],
            family=family,
            source=source,
            matrices=matrices,
            index=indexes[family],
        )
        indexes[family] += 1
    for obj in list(bpy.context.scene.objects):
        if getattr(obj, "type", None) == "POINTCLOUD" and any(
            marker in obj.name for marker in FAMILY_MARKERS.values()
        ):
            bpy.data.objects.remove(obj, do_unlink=True)


def _mesh_pointer(obj: Any) -> int | None:
    data = getattr(obj, "data", None)
    pointer = getattr(data, "as_pointer", None)
    if getattr(obj, "type", None) != "MESH" or not callable(pointer):
        return None
    return int(pointer())


def _source_metrics(bpy: Any, sources: Sequence[Any]) -> tuple[int, int, int]:
    visible = [
        obj
        for obj in bpy.context.scene.objects
        if getattr(obj, "type", None) == "MESH"
        and not bool(getattr(obj, "hide_render", False))
    ]
    pointers = {
        pointer
        for obj in [*visible, *sources]
        if (pointer := _mesh_pointer(obj)) is not None
    }
    if not visible or not pointers:
        raise CompleteViewerExportError("La scène viewer ne contient aucun mesh")
    images: set[int] = set()
    visited: set[int] = set()

    def visit(tree: Any) -> None:
        if tree is None or id(tree) in visited:
            return
        visited.add(id(tree))
        for node in tree.nodes:
            image = getattr(node, "image", None)
            if image is not None:
                images.add(id(image))
            visit(getattr(node, "node_tree", None))

    for obj in [*visible, *sources]:
        for material in getattr(getattr(obj, "data", None), "materials", ()) or ():
            if material is not None:
                visit(getattr(material, "node_tree", None))
    return len(visible), len(pointers), len(images)


def _gpu_count(node: Mapping[str, Any], accessors: Sequence[Any]) -> int:
    extensions = node.get("extensions")
    extension = (
        extensions.get("EXT_mesh_gpu_instancing")
        if isinstance(extensions, Mapping)
        else None
    )
    if not isinstance(extension, Mapping):
        # Blender serializes a Geometry Nodes batch containing exactly one
        # logical instance as a regular mesh node.  That node still represents
        # one complete source instance even though the GPU-instancing extension
        # is unnecessary.  Counting it as zero produces a false completeness
        # failure for every singleton prototype group.
        return 1 if isinstance(node.get("mesh"), int) else 0
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


def _validate_exact_gpu_counts(
    gltf: Mapping[str, Any], expected: Mapping[str, int]
) -> dict[str, int]:
    nodes = gltf.get("nodes", [])
    accessors = gltf.get("accessors", [])
    extensions = gltf.get("extensionsUsed", [])
    if not isinstance(nodes, list) or not isinstance(accessors, list):
        raise CompleteViewerExportError("Tables GLB viewer invalides")
    if sum(expected.values()) and (
        not isinstance(extensions, list)
        or "EXT_mesh_gpu_instancing" not in extensions
    ):
        raise CompleteViewerExportError("EXT_mesh_gpu_instancing absent du viewer")
    names = {
        node.get("name"): index
        for index, node in enumerate(nodes)
        if isinstance(node, Mapping) and isinstance(node.get("name"), str)
    }
    actual: dict[str, int] = {}
    for family, wanted in expected.items():
        root = names.get(FAMILY_ROOTS[family])
        if wanted == 0 and root is None:
            actual[family] = 0
            continue
        if not isinstance(root, int):
            raise CompleteViewerExportError(
                f"Racine {FAMILY_ROOTS[family]} absente du GLB"
            )
        count = sum(
            _gpu_count(nodes[index], accessors)
            for index in base._descendants(nodes, root)
            if index != root and isinstance(nodes[index], Mapping)
        )
        actual[family] = count
        if count != wanted:
            raise CompleteViewerExportError(
                f"Viewer incomplet pour {family}: attendu={wanted}, obtenu={count}"
            )
    return actual


def _export_glb(bpy: Any, output: Path) -> None:
    operator = bpy.ops.export_scene.gltf
    supported = {prop.identifier for prop in operator.get_rna_type().properties}
    if "export_gpu_instances" not in supported or "export_gn_mesh" not in supported:
        raise CompleteViewerExportError(
            "Blender ne fournit pas Geometry Nodes + EXT_mesh_gpu_instancing"
        )
    temporary = output.with_name(f".{output.stem}.part.glb")
    temporary.unlink(missing_ok=True)
    options = {
        "filepath": str(temporary),
        "check_existing": False,
        "export_format": "GLB",
        "export_gpu_instances": True,
        "export_gn_mesh": True,
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
        raise CompleteViewerExportError("Blender n'a pas écrit le GLB viewer")
    temporary.replace(output)


def export_complete_viewer(job_root: Path | str) -> Path:
    root = base._require_root(job_root)
    stage = root / "zone.usda"
    blend = root / BLEND_NAME
    zone_receipt_path = root / "zone.done.json"
    if not stage.is_file() or not blend.is_file() or not zone_receipt_path.is_file():
        raise CompleteViewerExportError(
            "zone.usda, zone.blend ou zone.done.json absent"
        )
    zone_receipt = base._load_json(zone_receipt_path, "reçu de zone")
    expected = base._expected_counts(zone_receipt)
    try:
        import bpy  # type: ignore
    except ImportError as error:  # pragma: no cover
        raise CompleteViewerExportError(
            "L'export viewer doit s'exécuter dans Blender"
        ) from error

    print("FIREVIEWER_VIEWER_STAGE open_blend", flush=True)
    if "FINISHED" not in bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False):
        raise CompleteViewerExportError("Ouverture de zone.blend impossible")
    print("FIREVIEWER_VIEWER_STAGE snapshot_instances", flush=True)
    groups = _group_instance_snapshots(base._snapshot_instances(bpy))
    counts = _count_groups(groups)
    if counts != expected:
        raise CompleteViewerExportError(
            f"Instances source incomplètes: attendu={expected}, obtenu={counts}"
        )
    sources = [source for _family, source, _matrices in groups]
    source_meshes, source_unique_meshes, source_images = _source_metrics(
        bpy, sources
    )
    print(
        "FIREVIEWER_VIEWER_STAGE build_gpu_batches "
        f"instances={sum(counts.values())} prototypes={len(groups)}",
        flush=True,
    )
    _build_gpu_batches(bpy, groups, expected)
    output = root / GLB_NAME
    print("FIREVIEWER_VIEWER_STAGE export_glb", flush=True)
    _export_glb(bpy, output)
    print("FIREVIEWER_VIEWER_STAGE validate_glb", flush=True)
    gltf = base._read_glb_json(output)
    actual = _validate_exact_gpu_counts(gltf, expected)
    if actual != expected:
        raise CompleteViewerExportError("Le GLB diverge des instances source")
    summary = base._validate_gltf_payload(
        gltf,
        expected_counts=expected,
        source_images=source_images,
        source_meshes=source_meshes,
    )
    if int(summary["mesh_count"]) < source_unique_meshes:
        raise CompleteViewerExportError("Meshes viewer incomplets")

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "zone_id": zone_receipt.get("zone_id"),
        "build_id": zone_receipt.get("build_id"),
        "source": {
            "entry_stage": "zone.usda",
            "entry_stage_sha256": base._sha256_file(stage),
            "standalone_blend": BLEND_NAME,
            "standalone_blend_sha256": base._sha256_file(blend),
            "zone_receipt": "zone.done.json",
            "zone_receipt_sha256": base._sha256_file(zone_receipt_path),
        },
        "viewer": {
            "file": GLB_NAME,
            "byte_count": output.stat().st_size,
            "sha256": base._sha256_file(output),
            "format": "glTF 2.0 binary",
            "external_dependencies": 0,
        },
        "completeness": {
            "policy": "fail_closed_exact_visual_scene",
            "terrain_and_non_instanced_meshes_required": True,
            "instance_strategy": "geometry_nodes_gpu_batches_by_prototype",
            "source_instance_count": sum(expected.values()),
            "source_prototype_group_count": len(groups),
            "source_mesh_object_count": source_meshes,
            "source_unique_mesh_count": source_unique_meshes,
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
        base._canonical_bytes(payload)
    ).hexdigest()
    base._write_json(root / RECEIPT_NAME, payload)
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
        json.dumps({"viewer_receipt": str(receipt)}, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
