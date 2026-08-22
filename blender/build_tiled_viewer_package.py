"""Build the canonical tiled viewer directly from the sealed zone package.

Terrain comes from each sealed ``terrain.fvtg`` and orthophoto, instance
transforms come from each tile ``scene.usda``, and shared prototypes are
extracted once from the sealed ``zone.blend``.  The monolithic ``viewer.glb``
is supported only by the explicit legacy helper used as a small-map test
oracle; it is not an input of the production tiled package.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "fireviewer.canonical-tile-catalog.v1"
RECEIPT_SCHEMA = "fireviewer.canonical-tiled-viewer-scene.v1"
INSTANCE_SCHEMA = "fireviewer.canonical-tile-instances.v1"
OUTPUT_DIRECTORY = "viewer-tiled"
CATALOG_NAME = "catalog.json"
RECEIPT_NAME = "viewer-tiled-scene.v1.json"
PROTOTYPE_PLAN_NAME = "prototype-export-plan.v1.json"
PROTOTYPE_EXPORT_NAME = "prototype-export.v1.json"
PROTOTYPE_PLAN_SCHEMA = "fireviewer.tiled-prototype-export-plan.v1"
PROTOTYPE_EXPORT_SCHEMA = "fireviewer.tiled-prototype-export.v1"
INSTANCE_MAGIC = b"FVINST1\0"
INSTANCE_VERSION = 1
INSTANCE_RECORD = struct.Struct("<10f")
TILE_SIZE_M = 500
FAMILY_ROOTS = {
    "buildings": "FireViewer_Buildings",
    "trees": "FireViewer_Trees",
    "context_assets": "FireViewer_ContextAssets",
}
_TILE_ID = re.compile(r"^x(-?\d+)_y(-?\d+)$")
_COMPONENT_FORMAT = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class TiledViewerPackageError(RuntimeError):
    """The canonical viewer cannot be partitioned without losing information."""


@dataclass(frozen=True, slots=True)
class Asset:
    path: str
    byte_count: int
    media_type: str

    def payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
        }


@dataclass(frozen=True, slots=True)
class Tile:
    tile_id: str
    west: int
    south: int
    node_index: int
    mesh_node_index: int
    chain: tuple[int, ...]

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.west, self.south, self.west + TILE_SIZE_M, self.south + TILE_SIZE_M)


@dataclass(frozen=True, slots=True)
class Prototype:
    prototype_id: str
    family: str
    mesh_index: int
    instances: tuple[tuple[float, ...], ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path, root: Path, media_type: str) -> Asset:
    relative = path.relative_to(root).as_posix()
    return Asset(relative, path.stat().st_size, media_type)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TiledViewerPackageError(f"{label} JSON invalide: {error}") from error
    if not isinstance(value, dict):
        raise TiledViewerPackageError(f"{label} doit être un objet JSON")
    return value


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise TiledViewerPackageError(f"GLB viewer illisible: {error}") from error
    if len(raw) < 28:
        raise TiledViewerPackageError("GLB viewer tronqué")
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total != len(raw):
        raise TiledViewerPackageError("En-tête GLB viewer invalide")
    json_length, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A or 20 + json_length + 8 > len(raw):
        raise TiledViewerPackageError("Chunk JSON GLB viewer invalide")
    try:
        gltf = json.loads(raw[20 : 20 + json_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TiledViewerPackageError("JSON GLB viewer invalide") from error
    binary_header = 20 + json_length
    binary_length, binary_type = struct.unpack_from("<II", raw, binary_header)
    binary_start = binary_header + 8
    if binary_type != 0x004E4942 or binary_start + binary_length > len(raw):
        raise TiledViewerPackageError("Chunk BIN GLB viewer invalide")
    if not isinstance(gltf, dict) or len(gltf.get("buffers", [])) != 1:
        raise TiledViewerPackageError("Le viewer doit contenir un buffer GLB unique")
    return gltf, raw[binary_start : binary_start + binary_length]


def _write_glb(path: Path, gltf: Mapping[str, Any], binary: bytes) -> None:
    json_raw = _canonical_bytes(gltf)
    json_padded = json_raw + b" " * ((-len(json_raw)) % 4)
    binary_padded = binary + b"\0" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_padded) + 8 + len(binary_padded)
    payload = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<II", len(json_padded), 0x4E4F534A),
            json_padded,
            struct.pack("<II", len(binary_padded), 0x004E4942),
            binary_padded,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_glb_from_spool(
    path: Path,
    gltf: Mapping[str, Any],
    binary_spool: Path,
    binary_length: int,
) -> None:
    """Finalize a GLB without materializing another full binary payload."""

    json_raw = _canonical_bytes(gltf)
    json_padded = json_raw + b" " * ((-len(json_raw)) % 4)
    binary_padding = (-binary_length) % 4
    total = 12 + 8 + len(json_padded) + 8 + binary_length + binary_padding
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("wb") as output, binary_spool.open("rb") as source:
        output.write(struct.pack("<4sII", b"glTF", 2, total))
        output.write(struct.pack("<II", len(json_padded), 0x4E4F534A))
        output.write(json_padded)
        output.write(struct.pack("<II", binary_length + binary_padding, 0x004E4942))
        shutil.copyfileobj(source, output, length=1024 * 1024)
        if binary_padding:
            output.write(b"\0" * binary_padding)
    os.replace(temporary, path)


def _table(gltf: Mapping[str, Any], name: str) -> list[Any]:
    value = gltf.get(name)
    if not isinstance(value, list):
        raise TiledViewerPackageError(f"Table GLB absente: {name}")
    return value


def _item(table: Sequence[Any], index: Any, label: str) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(table):
        raise TiledViewerPackageError(f"Index GLB invalide: {label}")
    value = table[index]
    if not isinstance(value, dict):
        raise TiledViewerPackageError(f"Entrée GLB invalide: {label}")
    return value


def _node_parents(nodes: Sequence[Any]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for parent_index, raw in enumerate(nodes):
        node = _item(nodes, parent_index, "node")
        for child in node.get("children", []):
            _item(nodes, child, "node child")
            if child in parents:
                raise TiledViewerPackageError("Un nœud GLB possède plusieurs parents")
            parents[child] = parent_index
    return parents


def _descendants(nodes: Sequence[Any], root: int) -> list[int]:
    result: list[int] = []
    pending = [root]
    seen: set[int] = set()
    while pending:
        index = pending.pop()
        if index in seen:
            raise TiledViewerPackageError("Cycle dans la hiérarchie GLB")
        seen.add(index)
        result.append(index)
        node = _item(nodes, index, "node descendant")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise TiledViewerPackageError("Enfants GLB invalides")
        pending.extend(reversed(children))
    return result


def _path_from_scene(nodes: Sequence[Any], parents: Mapping[int, int], target: int) -> tuple[int, ...]:
    chain = [target]
    while chain[-1] in parents:
        chain.append(parents[chain[-1]])
    chain.reverse()
    return tuple(chain)


def _node_index_by_name(nodes: Sequence[Any], wanted: str) -> int | None:
    matches: list[int] = []
    for index, raw in enumerate(nodes):
        node = _item(nodes, index, "node name")
        if node.get("name") == wanted:
            matches.append(index)
    if len(matches) > 1:
        raise TiledViewerPackageError(f"Nom de nœud GLB ambigu: {wanted}")
    return matches[0] if matches else None


def _node_translation(node: Mapping[str, Any]) -> tuple[float, float, float]:
    value = node.get("translation", [0.0, 0.0, 0.0])
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise TiledViewerPackageError("Translation GLB invalide")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise TiledViewerPackageError("Translation GLB non finie")
    return result


def _node_rotation(node: Mapping[str, Any]) -> tuple[float, float, float, float]:
    value = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise TiledViewerPackageError("Rotation GLB invalide")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise TiledViewerPackageError("Rotation GLB non finie")
    return result


def _node_scale(node: Mapping[str, Any]) -> tuple[float, float, float]:
    value = node.get("scale", [1.0, 1.0, 1.0])
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise TiledViewerPackageError("Échelle GLB invalide")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise TiledViewerPackageError("Échelle GLB non finie")
    return result


def _decode_accessor(
    gltf: Mapping[str, Any], binary: bytes, accessor_index: int
) -> tuple[tuple[float, ...], ...]:
    accessors = _table(gltf, "accessors")
    views = _table(gltf, "bufferViews")
    accessor = _item(accessors, accessor_index, "accessor")
    if "sparse" in accessor:
        raise TiledViewerPackageError("Accessor sparse non pris en charge")
    component_type = accessor.get("componentType")
    component_format = _COMPONENT_FORMAT.get(component_type)
    component_count = _TYPE_COMPONENTS.get(accessor.get("type"))
    count = accessor.get("count")
    if (
        component_format is None
        or component_count is None
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        raise TiledViewerPackageError("Accessor d'instances invalide")
    view = _item(views, accessor.get("bufferView"), "accessor bufferView")
    if view.get("buffer", 0) != 0:
        raise TiledViewerPackageError("Accessor hors du buffer GLB principal")
    component_size = struct.calcsize("<" + component_format)
    packed_size = component_size * component_count
    stride = view.get("byteStride", packed_size)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < packed_size:
        raise TiledViewerPackageError("Stride d'accessor invalide")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    view_end = int(view.get("byteOffset", 0)) + int(view.get("byteLength", 0))
    output: list[tuple[float, ...]] = []
    unpacker = struct.Struct("<" + component_format * component_count)
    for index in range(count):
        offset = start + index * stride
        if offset + packed_size > view_end or offset + packed_size > len(binary):
            raise TiledViewerPackageError("Accessor d'instances tronqué")
        output.append(tuple(float(item) for item in unpacker.unpack_from(binary, offset)))
    return tuple(output)


class _MeshSubset:
    def __init__(self, gltf: Mapping[str, Any], binary: bytes) -> None:
        self.source = gltf
        self.binary = binary
        self.output_binary = bytearray()
        self.output: dict[str, Any] = {
            "asset": copy.deepcopy(gltf.get("asset", {"version": "2.0"})),
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "materials": [],
            "textures": [],
            "images": [],
            "samplers": [],
            "buffers": [{"byteLength": 0}],
        }
        self.maps: dict[str, dict[int, int]] = {
            name: {}
            for name in (
                "bufferViews",
                "accessors",
                "materials",
                "textures",
                "images",
                "samplers",
            )
        }

    def _append(self, name: str, old_index: int, value: dict[str, Any]) -> int:
        mapping = self.maps[name]
        if old_index in mapping:
            return mapping[old_index]
        target = self.output[name]
        assert isinstance(target, list)
        new_index = len(target)
        mapping[old_index] = new_index
        target.append(value)
        return new_index

    def buffer_view(self, old_index: int) -> int:
        cached = self.maps["bufferViews"].get(old_index)
        if cached is not None:
            return cached
        source = copy.deepcopy(_item(_table(self.source, "bufferViews"), old_index, "bufferView"))
        if source.get("buffer", 0) != 0:
            raise TiledViewerPackageError("BufferView hors du GLB principal")
        offset = source.get("byteOffset", 0)
        length = source.get("byteLength")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or offset < 0
            or length <= 0
            or offset + length > len(self.binary)
        ):
            raise TiledViewerPackageError("BufferView source invalide")
        while len(self.output_binary) % 4:
            self.output_binary.append(0)
        new_offset = len(self.output_binary)
        self.output_binary.extend(self.binary[offset : offset + length])
        source["buffer"] = 0
        source["byteOffset"] = new_offset
        return self._append("bufferViews", old_index, source)

    def accessor(self, old_index: int) -> int:
        cached = self.maps["accessors"].get(old_index)
        if cached is not None:
            return cached
        source = copy.deepcopy(_item(_table(self.source, "accessors"), old_index, "accessor"))
        if "sparse" in source:
            raise TiledViewerPackageError("Mesh avec accessor sparse non pris en charge")
        source["bufferView"] = self.buffer_view(source.get("bufferView"))
        return self._append("accessors", old_index, source)

    def image(self, old_index: int) -> int:
        cached = self.maps["images"].get(old_index)
        if cached is not None:
            return cached
        source = copy.deepcopy(_item(_table(self.source, "images"), old_index, "image"))
        if "bufferView" not in source or "uri" in source:
            raise TiledViewerPackageError("Image viewer non embarquée")
        source["bufferView"] = self.buffer_view(source["bufferView"])
        return self._append("images", old_index, source)

    def sampler(self, old_index: int) -> int:
        cached = self.maps["samplers"].get(old_index)
        if cached is not None:
            return cached
        source = copy.deepcopy(_item(_table(self.source, "samplers"), old_index, "sampler"))
        return self._append("samplers", old_index, source)

    def texture(self, old_index: int) -> int:
        cached = self.maps["textures"].get(old_index)
        if cached is not None:
            return cached
        source = copy.deepcopy(_item(_table(self.source, "textures"), old_index, "texture"))
        source["source"] = self.image(source.get("source"))
        if "sampler" in source:
            source["sampler"] = self.sampler(source["sampler"])
        return self._append("textures", old_index, source)

    def _material_value(self, value: Any, parent_key: str = "") -> Any:
        if isinstance(value, list):
            return [self._material_value(item, parent_key) for item in value]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        result = {
            key: self._material_value(item, key)
            for key, item in value.items()
        }
        if parent_key.casefold().endswith("texture") and "index" in result:
            result["index"] = self.texture(result["index"])
        return result

    def material(self, old_index: int) -> int:
        cached = self.maps["materials"].get(old_index)
        if cached is not None:
            return cached
        source = _item(_table(self.source, "materials"), old_index, "material")
        return self._append("materials", old_index, self._material_value(source))

    def mesh(self, old_index: int) -> int:
        source = copy.deepcopy(_item(_table(self.source, "meshes"), old_index, "mesh"))
        primitives = source.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise TiledViewerPackageError("Prototype GLB sans primitive")
        for primitive in primitives:
            if not isinstance(primitive, dict) or "extensions" in primitive:
                raise TiledViewerPackageError("Primitive GLB compressée non prise en charge")
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict) or "POSITION" not in attributes:
                raise TiledViewerPackageError("Primitive GLB sans positions")
            primitive["attributes"] = {
                name: self.accessor(index) for name, index in attributes.items()
            }
            if "indices" in primitive:
                primitive["indices"] = self.accessor(primitive["indices"])
            if "targets" in primitive:
                primitive["targets"] = [
                    {name: self.accessor(index) for name, index in target.items()}
                    for target in primitive["targets"]
                ]
            if "material" in primitive:
                primitive["material"] = self.material(primitive["material"])
        target = self.output["meshes"]
        assert isinstance(target, list)
        target.append(source)
        return len(target) - 1

    def finish(self, *, mesh_index: int, node_chain: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], bytes]:
        self.mesh(mesh_index)
        nodes: list[dict[str, Any]] = []
        for source in node_chain:
            node = {
                key: copy.deepcopy(value)
                for key, value in source.items()
                if key in {"name", "translation", "rotation", "scale", "matrix", "extras"}
            }
            nodes.append(node)
        if not nodes:
            nodes.append({"name": "FireViewer prototype"})
        for index in range(len(nodes) - 1):
            nodes[index]["children"] = [index + 1]
        nodes[-1]["mesh"] = 0
        self.output["nodes"] = nodes
        self.output["scenes"] = [{"nodes": [0]}]
        self.output["buffers"] = [{"byteLength": len(self.output_binary)}]
        for name in ("samplers", "textures", "images", "materials"):
            if not self.output[name]:
                del self.output[name]
        used = self.source.get("extensionsUsed")
        if isinstance(used, list):
            retained = [item for item in used if item != "EXT_mesh_gpu_instancing"]
            if retained:
                self.output["extensionsUsed"] = retained
        return self.output, bytes(self.output_binary)


def _extract_mesh(
    gltf: Mapping[str, Any],
    binary: bytes,
    *,
    mesh_index: int,
    node_chain: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    subset = _MeshSubset(gltf, binary)
    payload, body = subset.finish(mesh_index=mesh_index, node_chain=node_chain)
    _write_glb(output, payload, body)


class _FarGlb:
    def __init__(
        self,
        *,
        generator: str = "FireViewer canonical FAR LOD",
        binary_spool: Path | None = None,
    ) -> None:
        self.binary = bytearray()
        self._binary_spool = binary_spool
        self._binary_stream = None
        self._binary_length = 0
        if binary_spool is not None:
            binary_spool.parent.mkdir(parents=True, exist_ok=True)
            self._binary_stream = binary_spool.open("wb")
        self.gltf: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": generator},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "materials": [],
            "textures": [],
            "images": [],
            "samplers": [
                {
                    "magFilter": 9729,
                    "minFilter": 9987,
                    "wrapS": 33071,
                    "wrapT": 33071,
                }
            ],
            "buffers": [{"byteLength": 0}],
        }

    def append_view(self, payload: bytes, *, target: int | None = None) -> int:
        if self._binary_stream is None:
            while len(self.binary) % 4:
                self.binary.append(0)
            offset = len(self.binary)
            self.binary.extend(payload)
            self._binary_length = len(self.binary)
        else:
            padding = (-self._binary_length) % 4
            if padding:
                self._binary_stream.write(b"\0" * padding)
                self._binary_length += padding
            offset = self._binary_length
            self._binary_stream.write(payload)
            self._binary_length += len(payload)
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
        }
        if target is not None:
            view["target"] = target
        views = self.gltf["bufferViews"]
        assert isinstance(views, list)
        views.append(view)
        return len(views) - 1

    def accessor(
        self,
        values: Sequence[Sequence[float]] | Sequence[int],
        *,
        component_type: int,
        kind: str,
        target: int,
    ) -> int:
        components = _TYPE_COMPONENTS[kind]
        code = _COMPONENT_FORMAT[component_type]
        # Terrain payloads are NumPy arrays in the optimized builder.  Packing
        # them through a Python list first duplicates tens of millions of
        # scalar objects on large maps.  Keep the generic sequence path for
        # callers that do not use NumPy, but stream contiguous numeric arrays
        # directly into the GLB buffer when available.
        array_values: Any | None = None
        try:
            import numpy as np

            if isinstance(values, np.ndarray):
                dtype = {
                    5120: np.dtype("<i1"),
                    5121: np.dtype("<u1"),
                    5122: np.dtype("<i2"),
                    5123: np.dtype("<u2"),
                    5125: np.dtype("<u4"),
                    5126: np.dtype("<f4"),
                }[component_type]
                array_values = np.ascontiguousarray(values, dtype=dtype)
                if components == 1:
                    array_values = array_values.reshape(-1)
                elif array_values.ndim != 2 or array_values.shape[1] != components:
                    raise TiledViewerPackageError(
                        f"Tableau GLB NumPy incompatible avec {kind}"
                    )
                payload = array_values.tobytes(order="C")
                count = int(array_values.size // components)
        except ImportError:  # pragma: no cover - production image includes NumPy
            array_values = None
        if array_values is None:
            if components == 1:
                flat = [int(item) for item in values]
            else:
                flat = [float(component) for row in values for component in row]
            payload = struct.pack("<" + code * len(flat), *flat)
            count = len(flat) // components
        view = self.append_view(payload, target=target)
        accessor: dict[str, Any] = {
            "bufferView": view,
            "componentType": component_type,
            "count": count,
            "type": kind,
        }
        if component_type == 5126 and count:
            if array_values is not None:
                rows = array_values.reshape(count, components)
                accessor["min"] = [float(value) for value in rows.min(axis=0)]
                accessor["max"] = [float(value) for value in rows.max(axis=0)]
            else:
                rows = [
                    flat[index * components : (index + 1) * components]
                    for index in range(count)
                ]
                accessor["min"] = [
                    min(row[index] for row in rows) for index in range(components)
                ]
                accessor["max"] = [
                    max(row[index] for row in rows) for index in range(components)
                ]
        accessors = self.gltf["accessors"]
        assert isinstance(accessors, list)
        accessors.append(accessor)
        return len(accessors) - 1

    def add_tile(
        self,
        *,
        tile_id: str,
        node_chain: Sequence[Mapping[str, Any]],
        positions: Sequence[Sequence[float]],
        normals: Sequence[Sequence[float]],
        texcoords: Sequence[Sequence[float]],
        indices: Sequence[int],
        image: bytes,
        image_mime: str = "image/jpeg",
    ) -> None:
        position_accessor = self.accessor(
            positions, component_type=5126, kind="VEC3", target=34962
        )
        normal_accessor = self.accessor(
            normals, component_type=5126, kind="VEC3", target=34962
        )
        texcoord_accessor = self.accessor(
            texcoords, component_type=5126, kind="VEC2", target=34962
        )
        index_accessor = self.accessor(
            indices,
            component_type=5123 if len(positions) <= 65_535 else 5125,
            kind="SCALAR",
            target=34963,
        )
        image_view = self.append_view(image)
        images = self.gltf["images"]
        textures = self.gltf["textures"]
        materials = self.gltf["materials"]
        meshes = self.gltf["meshes"]
        nodes = self.gltf["nodes"]
        scenes = self.gltf["scenes"]
        assert all(isinstance(value, list) for value in (images, textures, materials, meshes, nodes, scenes))
        images.append({"name": f"FAR {tile_id}", "mimeType": image_mime, "bufferView": image_view})
        image_index = len(images) - 1
        textures.append({"sampler": 0, "source": image_index})
        texture_index = len(textures) - 1
        materials.append(
            {
                "name": f"FAR terrain {tile_id}",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": texture_index},
                    "metallicFactor": 0,
                    "roughnessFactor": 0.9,
                },
            }
        )
        material_index = len(materials) - 1
        meshes.append(
            {
                "name": f"FAR terrain {tile_id}",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "TEXCOORD_0": texcoord_accessor,
                        },
                        "indices": index_accessor,
                        "material": material_index,
                    }
                ],
            }
        )
        first_node = len(nodes)
        for source in node_chain:
            node = {
                key: copy.deepcopy(value)
                for key, value in source.items()
                if key in {"name", "translation", "rotation", "scale", "matrix", "extras"}
            }
            nodes.append(node)
        for index in range(first_node, len(nodes) - 1):
            nodes[index]["children"] = [index + 1]
        nodes[-1]["mesh"] = len(meshes) - 1
        scenes[0]["nodes"].append(first_node)

    def write(self, path: Path) -> None:
        self.gltf["buffers"] = [{"byteLength": self._binary_length}]
        if self._binary_stream is None:
            _write_glb(path, self.gltf, bytes(self.binary))
            return
        self._binary_stream.flush()
        self._binary_stream.close()
        self._binary_stream = None
        assert self._binary_spool is not None
        try:
            _write_glb_from_spool(
                path,
                self.gltf,
                self._binary_spool,
                self._binary_length,
            )
        finally:
            self._binary_spool.unlink(missing_ok=True)


def _terrain_image(gltf: Mapping[str, Any], binary: bytes, mesh_index: int) -> bytes:
    from PIL import Image

    mesh = _item(_table(gltf, "meshes"), mesh_index, "terrain mesh")
    primitives = mesh.get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise TiledViewerPackageError("Terrain viewer avec primitives ambiguës")
    primitive = primitives[0]
    if not isinstance(primitive, dict):
        raise TiledViewerPackageError("Primitive terrain invalide")
    material = _item(_table(gltf, "materials"), primitive.get("material"), "terrain material")
    pbr = material.get("pbrMetallicRoughness")
    texture_info = pbr.get("baseColorTexture") if isinstance(pbr, dict) else None
    texture = _item(
        _table(gltf, "textures"),
        texture_info.get("index") if isinstance(texture_info, dict) else None,
        "terrain texture",
    )
    image = _item(_table(gltf, "images"), texture.get("source"), "terrain image")
    view = _item(_table(gltf, "bufferViews"), image.get("bufferView"), "terrain image view")
    offset = view.get("byteOffset", 0)
    length = view.get("byteLength")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or isinstance(length, bool)
        or not isinstance(length, int)
        or offset < 0
        or length <= 0
        or offset + length > len(binary)
    ):
        raise TiledViewerPackageError("Image terrain tronquée")
    try:
        with Image.open(io.BytesIO(binary[offset : offset + length])) as source:
            source.load()
            resized = source.convert("RGB")
            resized.thumbnail((96, 96), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            resized.save(output, format="JPEG", quality=76, optimize=True)
            return output.getvalue()
    except OSError as error:
        raise TiledViewerPackageError("Image terrain indécodable") from error


def _far_grid(
    gltf: Mapping[str, Any], binary: bytes, mesh_index: int, *, stride: int = 8
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]], list[tuple[float, ...]], list[int]]:
    mesh = _item(_table(gltf, "meshes"), mesh_index, "terrain mesh")
    primitives = mesh.get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1 or not isinstance(primitives[0], dict):
        raise TiledViewerPackageError("Terrain viewer non régulier")
    attributes = primitives[0].get("attributes")
    if not isinstance(attributes, dict) or set(attributes) < {"POSITION", "NORMAL", "TEXCOORD_0"}:
        raise TiledViewerPackageError("Attributs du terrain viewer incomplets")
    positions = _decode_accessor(gltf, binary, attributes["POSITION"])
    normals = _decode_accessor(gltf, binary, attributes["NORMAL"])
    texcoords = _decode_accessor(gltf, binary, attributes["TEXCOORD_0"])
    side = int(round(math.sqrt(len(positions))))
    if side * side != len(positions) or not len(normals) == len(texcoords) == len(positions):
        raise TiledViewerPackageError("Grille terrain viewer incohérente")
    selected = list(range(0, side - 1, stride)) + [side - 1]
    output_positions: list[tuple[float, ...]] = []
    output_normals: list[tuple[float, ...]] = []
    output_texcoords: list[tuple[float, ...]] = []
    for row in selected:
        for column in selected:
            source_index = row * side + column
            output_positions.append(positions[source_index])
            output_normals.append(normals[source_index])
            output_texcoords.append(texcoords[source_index])
    output_indices: list[int] = []
    width = len(selected)
    for row in range(width - 1):
        for column in range(width - 1):
            northwest = row * width + column
            northeast = northwest + 1
            southwest = northwest + width
            southeast = southwest + 1
            output_indices.extend((northwest, southwest, northeast, northeast, southwest, southeast))
    return output_positions, output_normals, output_texcoords, output_indices


def _build_far_glb(
    gltf: Mapping[str, Any], binary: bytes, tiles: Sequence[Tile], output: Path
) -> None:
    nodes = _table(gltf, "nodes")
    builder = _FarGlb()
    for tile in tiles:
        mesh_node = _item(nodes, tile.mesh_node_index, "terrain mesh node")
        mesh_index = int(mesh_node["mesh"])
        positions, normals, texcoords, indices = _far_grid(gltf, binary, mesh_index)
        builder.add_tile(
            tile_id=tile.tile_id,
            node_chain=[_item(nodes, index, "terrain chain") for index in tile.chain],
            positions=positions,
            normals=normals,
            texcoords=texcoords,
            indices=indices,
            image=_terrain_image(gltf, binary, mesh_index),
        )
    builder.write(output)


def _tile_records(zone_receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = zone_receipt.get("tiles")
    if not isinstance(rows, list) or not rows:
        raise TiledViewerPackageError("Reçu de zone sans tuiles")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise TiledViewerPackageError("Tuile du reçu de zone invalide")
        tile_id = raw.get("tile_id")
        origin = raw.get("origin_l93_m")
        if (
            not isinstance(tile_id, str)
            or _TILE_ID.fullmatch(tile_id) is None
            or not isinstance(origin, list)
            or len(origin) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in origin)
            or tile_id in result
        ):
            raise TiledViewerPackageError("Identité de tuile invalide")
        if tile_id != f"x{origin[0]}_y{origin[1]}":
            raise TiledViewerPackageError("ID et origine de tuile divergents")
        result[tile_id] = raw
    return result


def _find_tiles(
    gltf: Mapping[str, Any], zone_receipt: Mapping[str, Any]
) -> tuple[list[Tile], tuple[int, int, int]]:
    nodes = _table(gltf, "nodes")
    parents = _node_parents(nodes)
    records = _tile_records(zone_receipt)
    tiles: list[Tile] = []
    inferred_origins: set[tuple[int, int]] = set()
    for tile_id, record in records.items():
        node_index = _node_index_by_name(nodes, tile_id)
        if node_index is None:
            raise TiledViewerPackageError(f"Terrain viewer absent: {tile_id}")
        descendants = _descendants(nodes, node_index)
        mesh_nodes = [index for index in descendants if "mesh" in _item(nodes, index, "terrain node")]
        if len(mesh_nodes) != 1:
            raise TiledViewerPackageError(f"Terrain viewer ambigu: {tile_id}")
        west, south = record["origin_l93_m"]
        translation = _node_translation(_item(nodes, node_index, "tile node"))
        inferred_origins.add((round(west - translation[0]), round(south + translation[2])))
        mesh_node = mesh_nodes[0]
        tiles.append(
            Tile(
                tile_id=tile_id,
                west=west,
                south=south,
                node_index=node_index,
                mesh_node_index=mesh_node,
                chain=_path_from_scene(nodes, parents, mesh_node),
            )
        )
    if len(inferred_origins) != 1:
        raise TiledViewerPackageError("Origine locale GLB incohérente entre les terrains")
    origin_east, origin_north = next(iter(inferred_origins))
    return sorted(tiles, key=lambda item: (item.south, item.west)), (origin_east, origin_north, 0)


def _prototype_instances(
    gltf: Mapping[str, Any], binary: bytes, family: str, carrier_index: int
) -> tuple[int, tuple[tuple[float, ...], ...]]:
    nodes = _table(gltf, "nodes")
    carrier = _item(nodes, carrier_index, "prototype carrier")
    extension = carrier.get("extensions", {}).get("EXT_mesh_gpu_instancing")
    if isinstance(extension, dict):
        attributes = extension.get("attributes")
        if not isinstance(attributes, dict) or set(attributes) != {"TRANSLATION", "ROTATION", "SCALE"}:
            raise TiledViewerPackageError(f"Attributs GPU incomplets pour {family}")
        if "mesh" not in carrier:
            raise TiledViewerPackageError("Carrier GPU sans prototype mesh")
        translations = _decode_accessor(gltf, binary, attributes["TRANSLATION"])
        rotations = _decode_accessor(gltf, binary, attributes["ROTATION"])
        scales = _decode_accessor(gltf, binary, attributes["SCALE"])
        if not len(translations) == len(rotations) == len(scales):
            raise TiledViewerPackageError("Attributs GPU de tailles divergentes")
        return int(carrier["mesh"]), tuple(
            (*translation, *rotation, *scale)
            for translation, rotation, scale in zip(translations, rotations, scales, strict=True)
        )
    descendants = _descendants(nodes, carrier_index)
    mesh_nodes = [index for index in descendants if "mesh" in _item(nodes, index, "singleton node")]
    if len(mesh_nodes) != 1 or "matrix" in carrier:
        raise TiledViewerPackageError("Prototype singleton viewer ambigu")
    mesh_node = _item(nodes, mesh_nodes[0], "singleton mesh")
    instance = (
        *_node_translation(mesh_node),
        *_node_rotation(mesh_node),
        *_node_scale(mesh_node),
    )
    return int(mesh_node["mesh"]), (instance,)


def _find_prototypes(
    gltf: Mapping[str, Any], binary: bytes, expected: Mapping[str, int]
) -> list[Prototype]:
    nodes = _table(gltf, "nodes")
    prototypes: list[Prototype] = []
    actual = {family: 0 for family in FAMILY_ROOTS}
    for family, root_name in FAMILY_ROOTS.items():
        wanted = expected[family]
        root_index = _node_index_by_name(nodes, root_name)
        if root_index is None:
            if wanted == 0:
                continue
            raise TiledViewerPackageError(f"Racine viewer absente: {root_name}")
        root = _item(nodes, root_index, "family root")
        children = root.get("children", [])
        if not isinstance(children, list):
            raise TiledViewerPackageError("Liste de prototypes invalide")
        for family_index, carrier_index in enumerate(children):
            mesh_index, instances = _prototype_instances(gltf, binary, family, carrier_index)
            actual[family] += len(instances)
            prototypes.append(
                Prototype(
                    prototype_id=f"{family.replace('_', '-')}-{family_index:04d}",
                    family=family,
                    mesh_index=mesh_index,
                    instances=instances,
                )
            )
    if actual != dict(expected):
        raise TiledViewerPackageError(
            f"Comptages viewer divergents: attendu={dict(expected)}, obtenu={actual}"
        )
    return prototypes


def _owner_tile(
    record: Sequence[float], tiles: Sequence[Tile], origin: Sequence[int]
) -> Tile:
    east = origin[0] + record[0]
    north = origin[1] - record[2]
    # Blender stores GPU transforms as float32.  A point sealed on a 500 m tile
    # boundary can therefore round a few centimetres to either side.  Snap only
    # those near-boundary values before applying the deterministic half-open grid.
    for coordinate_name, coordinate in (("east", east), ("north", north)):
        snapped = round(coordinate / TILE_SIZE_M) * TILE_SIZE_M
        if abs(coordinate - snapped) <= 0.1:
            if coordinate_name == "east":
                east = float(snapped)
            else:
                north = float(snapped)
    candidates = [
        tile
        for tile in tiles
        if tile.west - 1e-3 <= east <= tile.west + TILE_SIZE_M + 1e-3
        and tile.south - 1e-3 <= north <= tile.south + TILE_SIZE_M + 1e-3
    ]
    if not candidates:
        raise TiledViewerPackageError(
            f"Instance hors de la zone tuilée: east={east:.3f}, north={north:.3f}"
        )
    if len(candidates) == 1:
        return candidates[0]
    # Boundary ownership follows the half-open grid, except on the outer edge.
    interior = [
        tile
        for tile in candidates
        if tile.west <= east < tile.west + TILE_SIZE_M
        and tile.south <= north < tile.south + TILE_SIZE_M
    ]
    if len(interior) == 1:
        return interior[0]
    return sorted(candidates, key=lambda tile: (tile.south, tile.west))[-1]


def _write_instances(
    path: Path,
    *,
    tile_id: str,
    groups: Sequence[tuple[Prototype, Sequence[tuple[float, ...]]]],
) -> None:
    records = bytearray()
    rows: list[dict[str, Any]] = []
    offset = 0
    family_counts = {family: 0 for family in FAMILY_ROOTS}
    for prototype, instances in groups:
        if not instances:
            continue
        rows.append(
            {
                "prototype_id": prototype.prototype_id,
                "family": prototype.family,
                "offset_records": offset,
                "count": len(instances),
            }
        )
        for record in instances:
            if len(record) != 10 or any(not math.isfinite(value) for value in record):
                raise TiledViewerPackageError("Transformation d'instance invalide")
            records.extend(INSTANCE_RECORD.pack(*record))
        offset += len(instances)
        family_counts[prototype.family] += len(instances)
    header = _canonical_bytes(
        {
            "schema": INSTANCE_SCHEMA,
            "tile_id": tile_id,
            "record_stride_bytes": INSTANCE_RECORD.size,
            "record_count": offset,
            "family_counts": family_counts,
            "groups": rows,
        }
    )
    payload = (
        INSTANCE_MAGIC
        + struct.pack("<HHI", INSTANCE_VERSION, 0, len(header))
        + header
        + records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _expected_counts(zone_receipt: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "buildings": zone_receipt.get("building_count"),
        "trees": zone_receipt.get("tree_count"),
        "context_assets": zone_receipt.get("context_asset_count", 0),
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise TiledViewerPackageError("Comptages scellés de zone invalides")
    return {name: int(value) for name, value in counts.items()}


def _validate_source_asset(
    root: Path,
    raw: object,
    label: str,
) -> None:
    if not isinstance(raw, Mapping):
        raise TiledViewerPackageError(f"Source scellée absente: {label}")
    relative = raw.get("path")
    if not isinstance(relative, str) or "\\" in relative:
        raise TiledViewerPackageError(f"Chemin de source invalide: {label}")
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or any(
        part in {"", ".", ".."} for part in posix_path.parts
    ):
        raise TiledViewerPackageError(f"Source non confinée: {label}")
    path = root.joinpath(*posix_path.parts)
    if (
        not path.is_file()
        or raw.get("byte_count") != path.stat().st_size
    ):
        raise TiledViewerPackageError(f"Source scellée divergente: {label}")


def _read_instance_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        prefix = stream.read(len(INSTANCE_MAGIC) + 8)
        if len(prefix) != len(INSTANCE_MAGIC) + 8 or prefix[:8] != INSTANCE_MAGIC:
            raise TiledViewerPackageError(f"En-tête FVI invalide: {path.name}")
        version, reserved, header_size = struct.unpack_from("<HHI", prefix, 8)
        if version != INSTANCE_VERSION or reserved != 0 or header_size <= 0:
            raise TiledViewerPackageError(f"Version FVI invalide: {path.name}")
        try:
            header = json.loads(stream.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TiledViewerPackageError(f"JSON FVI invalide: {path.name}") from error
        payload_size = path.stat().st_size - len(prefix) - header_size
    if not isinstance(header, dict):
        raise TiledViewerPackageError(f"En-tête FVI non objet: {path.name}")
    record_count = header.get("record_count")
    if (
        header.get("schema") != INSTANCE_SCHEMA
        or header.get("record_stride_bytes") != INSTANCE_RECORD.size
        or isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
        or payload_size != record_count * INSTANCE_RECORD.size
    ):
        raise TiledViewerPackageError(f"Payload FVI incohérent: {path.name}")
    return header, record_count


def _validate_self_contained_glb(path: Path) -> None:
    gltf, _binary = _read_glb(path)
    buffers = gltf.get("buffers")
    images = gltf.get("images", [])
    if (
        not isinstance(buffers, list)
        or any(
            not isinstance(buffer, Mapping) or buffer.get("uri") is not None
            for buffer in buffers
        )
        or not isinstance(images, list)
        or any(
            not isinstance(image, Mapping)
            or image.get("uri") is not None
            or not isinstance(image.get("bufferView"), int)
            for image in images
        )
    ):
        raise TiledViewerPackageError(f"GLB avec dépendance externe: {path.name}")


def validate_tiled_viewer_package(
    job_root: Path | str,
    *,
    require_sealed_source_assets: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every published payload and return the backend viewer descriptor."""

    root = Path(job_root).resolve(strict=True)
    tiled_root = root / OUTPUT_DIRECTORY
    zone_receipt = _load_json(root / "zone.done.json", "reçu de zone")
    receipt = _load_json(tiled_root / RECEIPT_NAME, "reçu viewer tuilé")
    catalog_path = tiled_root / CATALOG_NAME
    catalog = _load_json(catalog_path, "catalogue viewer tuilé")
    expected = _expected_counts(zone_receipt)
    catalog_reference = receipt.get("catalog")
    canonical = catalog.get("canonical")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("family_instance_counts") != expected
        or not isinstance(catalog_reference, Mapping)
        or catalog_reference.get("path") != CATALOG_NAME
        or catalog_reference.get("byte_count") != catalog_path.stat().st_size
        or catalog.get("schema") != SCHEMA
        or not isinstance(canonical, Mapping)
        or canonical.get("representation") != "complete_non_simplified_map"
        or canonical.get("policy") != "fail_closed_exact_visual_scene"
        or canonical.get("family_instance_counts") != expected
    ):
        raise TiledViewerPackageError("Le paquet viewer tuilé est incomplet")

    sealed_source = canonical.get("source")
    direct_source = isinstance(sealed_source, Mapping)
    if direct_source:
        if (
            sealed_source.get("kind") != "sealed_zone_and_tile_packages"
            or receipt.get("source") != sealed_source
            or sealed_source.get("tile_package_count")
            != len(_tile_records(zone_receipt))
        ):
            raise TiledViewerPackageError("Provenance scellée du viewer invalide")
        if require_sealed_source_assets:
            _validate_source_asset(root, sealed_source.get("zone_receipt"), "zone")
            _validate_source_asset(root, sealed_source.get("stage_layout"), "layout")
            _validate_source_asset(root, sealed_source.get("zone_blend"), "zone.blend")
    else:
        viewer_path = root / "viewer.glb"
        viewer_receipt = _load_json(root / "viewer-scene.v1.json", "reçu viewer")
        source_viewer = viewer_receipt.get("viewer")
        if (
            not viewer_path.is_file()
            or not isinstance(source_viewer, Mapping)
            or source_viewer.get("sha256") != _sha256_file(viewer_path)
            or source_viewer.get("byte_count") != viewer_path.stat().st_size
            or canonical.get("source_viewer_sha256") != source_viewer.get("sha256")
            or canonical.get("source_viewer_byte_count")
            != source_viewer.get("byte_count")
        ):
            raise TiledViewerPackageError(
                "Le paquet tuilé diverge du viewer monolithique oracle"
            )

    raw_assets: list[Mapping[str, Any]] = []
    far = catalog.get("far")
    if not isinstance(far, Mapping) or not isinstance(far.get("asset"), Mapping):
        raise TiledViewerPackageError("Le bootstrap FAR du viewer tuilé est absent")
    raw_assets.append(far["asset"])
    prototypes = catalog.get("prototypes")
    tiles = catalog.get("tiles")
    if not isinstance(prototypes, list) or not isinstance(tiles, list):
        raise TiledViewerPackageError("L'inventaire du viewer tuilé est invalide")
    prototype_ids: set[str] = set()
    prototype_family_counts = {family: 0 for family in FAMILY_ROOTS}
    for prototype in prototypes:
        if not isinstance(prototype, Mapping) or not isinstance(
            prototype.get("asset"), Mapping
        ):
            raise TiledViewerPackageError("Prototype du viewer tuilé invalide")
        prototype_id = prototype.get("id")
        family = prototype.get("family")
        instance_count = prototype.get("instance_count")
        if (
            not isinstance(prototype_id, str)
            or prototype_id in prototype_ids
            or family not in FAMILY_ROOTS
            or isinstance(instance_count, bool)
            or not isinstance(instance_count, int)
            or instance_count < 0
        ):
            raise TiledViewerPackageError("Inventaire des prototypes incohérent")
        prototype_ids.add(prototype_id)
        prototype_family_counts[str(family)] += instance_count
        raw_assets.append(prototype["asset"])
    if prototype_family_counts != expected:
        raise TiledViewerPackageError("Comptages par prototype incomplets")
    zone_tiles = _tile_records(zone_receipt)
    seen_tiles: set[str] = set()
    tiled_family_counts = {family: 0 for family in FAMILY_ROOTS}
    for tile in tiles:
        if not isinstance(tile, Mapping):
            raise TiledViewerPackageError("Tuile du viewer tuilé invalide")
        tile_id = tile.get("id")
        source_tile = zone_tiles.get(tile_id) if isinstance(tile_id, str) else None
        if source_tile is None or tile_id in seen_tiles:
            raise TiledViewerPackageError("Identité de tuile viewer invalide")
        seen_tiles.add(tile_id)
        west, south = source_tile["origin_l93_m"]
        source_counts = {
            "buildings": source_tile.get("building_count"),
            "trees": source_tile.get("tree_count"),
            "context_assets": source_tile.get("context_asset_count", 0),
        }
        actual_tile_counts = tile.get("family_instance_counts")
        if (
            tile.get("bounds_l93_m")
            != [west, south, west + TILE_SIZE_M, south + TILE_SIZE_M]
            or tile.get("source_family_instance_counts") != source_counts
            or not isinstance(actual_tile_counts, Mapping)
            or set(actual_tile_counts) != set(FAMILY_ROOTS)
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in actual_tile_counts.values()
            )
            or (direct_source and actual_tile_counts != source_counts)
            or not isinstance(tile.get("prototype_ids"), list)
            or not set(tile["prototype_ids"]).issubset(prototype_ids)
        ):
            raise TiledViewerPackageError(
                f"Tuile viewer divergente de la source scellée: {tile_id}"
            )
        for family, count in actual_tile_counts.items():
            tiled_family_counts[family] += int(count)
        for name in ("terrain", "instances"):
            if not isinstance(tile.get(name), Mapping):
                raise TiledViewerPackageError(
                    f"Payload {name} du viewer tuilé absent"
                )
            raw_assets.append(tile[name])
    if seen_tiles != set(zone_tiles) or tiled_family_counts != expected:
        raise TiledViewerPackageError("Couverture des tuiles viewer incomplète")

    seen: set[str] = set()
    payload_bytes = 0
    for asset in raw_assets:
        relative = asset.get("path")
        if not isinstance(relative, str) or "\\" in relative:
            raise TiledViewerPackageError("Chemin de payload tuilé invalide")
        posix_path = PurePosixPath(relative)
        if (
            posix_path.is_absolute()
            or any(part in {"", ".", ".."} for part in posix_path.parts)
            or relative in seen
        ):
            raise TiledViewerPackageError("Chemin de payload tuilé non confiné")
        seen.add(relative)
        path = tiled_root.joinpath(*posix_path.parts)
        byte_count = asset.get("byte_count")
        if (
            not path.is_file()
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or path.stat().st_size != byte_count
        ):
            raise TiledViewerPackageError(f"Payload tuilé divergent: {relative}")
        payload_bytes += byte_count
    for tile in tiles:
        instance_path = tiled_root.joinpath(
            *PurePosixPath(tile["instances"]["path"]).parts
        )
        header, record_count = _read_instance_header(instance_path)
        expected_tile_count = sum(tile["family_instance_counts"].values())
        groups = header.get("groups")
        if (
            header.get("tile_id") != tile.get("id")
            or header.get("family_counts") != tile.get("family_instance_counts")
            or record_count != expected_tile_count
            or not isinstance(groups, list)
            or any(
                not isinstance(group, Mapping)
                or group.get("prototype_id") not in prototype_ids
                for group in groups
            )
        ):
            raise TiledViewerPackageError(
                f"Instances FVI divergentes: {tile.get('id')}"
            )
    if (
        catalog.get("prototype_count") != len(prototypes)
        or catalog.get("tile_count") != len(tiles)
        or catalog.get("payload_file_count") != len(raw_assets)
        or catalog.get("payload_byte_count") != payload_bytes
        or receipt.get("payload_file_count") != len(raw_assets)
        or receipt.get("payload_byte_count") != payload_bytes
    ):
        raise TiledViewerPackageError("Totaux du paquet viewer tuilé invalides")
    bootstrap_asset = dict(far["asset"])
    bootstrap_asset["path"] = f"{OUTPUT_DIRECTORY}/{bootstrap_asset['path']}"
    return receipt, {
        "catalog_path": f"{OUTPUT_DIRECTORY}/{CATALOG_NAME}",
        "receipt_path": f"{OUTPUT_DIRECTORY}/{RECEIPT_NAME}",
        "catalog_byte_count": catalog_reference["byte_count"],
        "payload_file_count": len(raw_assets),
        "payload_byte_count": payload_bytes,
        "bootstrap_asset": bootstrap_asset,
        "representation": "complete_tiled_non_simplified_map",
        "completeness": {
            "policy": "fail_closed_exact_visual_scene",
            "mesh_coverage": "complete",
            "family_instance_counts": expected,
        },
    }


def build_tiled_viewer_package_from_monolithic(job_root: Path | str) -> Path:
    """Legacy small-map oracle: losslessly split an existing monolithic GLB."""
    root = Path(job_root).resolve(strict=True)
    viewer_path = root / "viewer.glb"
    viewer_receipt_path = root / "viewer-scene.v1.json"
    zone_receipt_path = root / "zone.done.json"
    if not viewer_path.is_file() or not viewer_receipt_path.is_file() or not zone_receipt_path.is_file():
        raise TiledViewerPackageError("Viewer ou reçus canoniques absents")
    viewer_receipt = _load_json(viewer_receipt_path, "reçu viewer")
    zone_receipt = _load_json(zone_receipt_path, "reçu de zone")
    expected = _expected_counts(zone_receipt)
    viewer = viewer_receipt.get("viewer")
    completeness = viewer_receipt.get("completeness")
    if (
        not isinstance(viewer, dict)
        or viewer.get("sha256") != _sha256_file(viewer_path)
        or viewer.get("byte_count") != viewer_path.stat().st_size
        or not isinstance(completeness, dict)
        or completeness.get("policy") != "fail_closed_exact_visual_scene"
        or completeness.get("family_instance_counts") != expected
    ):
        raise TiledViewerPackageError("Le viewer monolithique n'est pas un oracle complet valide")

    gltf, binary = _read_glb(viewer_path)
    tiles, origin = _find_tiles(gltf, zone_receipt)
    prototypes = _find_prototypes(gltf, binary, expected)
    nodes = _table(gltf, "nodes")
    meshes = _table(gltf, "meshes")
    staging = root / f".{OUTPUT_DIRECTORY}.part"
    output = root / OUTPUT_DIRECTORY
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        far_path = staging / "far.glb"
        _build_far_glb(gltf, binary, tiles, far_path)
        far_asset = _asset(far_path, staging, "model/gltf-binary")
        prototype_rows: list[dict[str, Any]] = []
        prototype_assets: list[Asset] = []
        for prototype in prototypes:
            mesh = _item(meshes, prototype.mesh_index, "prototype mesh")
            path = staging / "prototypes" / f"{prototype.prototype_id}.glb"
            _extract_mesh(
                gltf,
                binary,
                mesh_index=prototype.mesh_index,
                node_chain=({"name": mesh.get("name", prototype.prototype_id)},),
                output=path,
            )
            asset = _asset(path, staging, "model/gltf-binary")
            prototype_assets.append(asset)
            prototype_rows.append(
                {
                    "id": prototype.prototype_id,
                    "family": prototype.family,
                    "instance_count": len(prototype.instances),
                    "asset": asset.payload(),
                }
            )

        by_tile: dict[str, dict[str, list[tuple[float, ...]]]] = {
            tile.tile_id: {prototype.prototype_id: [] for prototype in prototypes}
            for tile in tiles
        }
        for prototype in prototypes:
            for record in prototype.instances:
                owner = _owner_tile(record, tiles, origin)
                by_tile[owner.tile_id][prototype.prototype_id].append(record)

        tile_rows: list[dict[str, Any]] = []
        tile_assets: list[Asset] = []
        total_counts = {family: 0 for family in FAMILY_ROOTS}
        zone_tiles = _tile_records(zone_receipt)
        for tile in tiles:
            mesh_node = _item(nodes, tile.mesh_node_index, "terrain mesh node")
            terrain_path = staging / "tiles" / tile.tile_id / "terrain.glb"
            chain = [_item(nodes, index, "terrain chain") for index in tile.chain]
            _extract_mesh(
                gltf,
                binary,
                mesh_index=int(mesh_node["mesh"]),
                node_chain=chain,
                output=terrain_path,
            )
            terrain_asset = _asset(terrain_path, staging, "model/gltf-binary")
            instance_path = staging / "tiles" / tile.tile_id / "instances.fvi"
            groups = [
                (prototype, by_tile[tile.tile_id][prototype.prototype_id])
                for prototype in prototypes
                if by_tile[tile.tile_id][prototype.prototype_id]
            ]
            _write_instances(instance_path, tile_id=tile.tile_id, groups=groups)
            instance_asset = _asset(
                instance_path, staging, "application/vnd.fireviewer.instances"
            )
            counts = {family: 0 for family in FAMILY_ROOTS}
            for prototype, records in groups:
                counts[prototype.family] += len(records)
                total_counts[prototype.family] += len(records)
            source_tile = zone_tiles[tile.tile_id]
            sealed_counts = {
                "buildings": source_tile.get("building_count"),
                "trees": source_tile.get("tree_count"),
                "context_assets": source_tile.get("context_asset_count", 0),
            }
            tile_assets.extend((terrain_asset, instance_asset))
            tile_rows.append(
                {
                    "id": tile.tile_id,
                    "bounds_l93_m": list(tile.bounds),
                    "terrain": terrain_asset.payload(),
                    "instances": instance_asset.payload(),
                    "family_instance_counts": counts,
                    "source_family_instance_counts": sealed_counts,
                    "prototype_ids": [prototype.prototype_id for prototype, _records in groups],
                }
            )
        if total_counts != expected:
            raise TiledViewerPackageError(
                f"Comptages tuilés incomplets: attendu={expected}, obtenu={total_counts}"
            )

        all_assets = [far_asset, *prototype_assets, *tile_assets]
        west = min(tile.west for tile in tiles)
        south = min(tile.south for tile in tiles)
        east = max(tile.west + TILE_SIZE_M for tile in tiles)
        north = max(tile.south + TILE_SIZE_M for tile in tiles)
        catalog = {
            "schema": SCHEMA,
            "catalog_version": 1,
            "crs": "EPSG:2154",
            "linear_unit": "metre",
            "coordinate_frame": "gltf-y-up-local",
            "origin_l93_m": list(origin),
            "bounds_l93_m": [west, south, east, north],
            "tile_size_m": TILE_SIZE_M,
            "loading": {
                "detail_publish_distance_m": 2_200,
                "detail_preload_radius_m": 900,
                "maximum_resident_tile_count": 64,
                "maximum_concurrent_requests": 4,
                "terrain_before_instances": True,
            },
            "far": {
                "role": "navigation_lod_not_counted_as_canonical_detail",
                "asset": far_asset.payload(),
            },
            "canonical": {
                "representation": "complete_non_simplified_map",
                "policy": "fail_closed_exact_visual_scene",
                "source_viewer_sha256": viewer["sha256"],
                "source_viewer_byte_count": viewer["byte_count"],
                "family_instance_counts": expected,
            },
            "prototype_count": len(prototypes),
            "tile_count": len(tiles),
            "payload_file_count": len(all_assets),
            "payload_byte_count": sum(item.byte_count for item in all_assets),
            "prototypes": prototype_rows,
            "tiles": tile_rows,
        }
        catalog_path = staging / CATALOG_NAME
        _write_json(catalog_path, catalog)
        catalog_asset = _asset(catalog_path, staging, "application/json")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "complete",
            "zone_id": zone_receipt.get("zone_id"),
            "build_id": zone_receipt.get("build_id"),
            "representation": "complete_tiled_non_simplified_map",
            "catalog": catalog_asset.payload(),
            "payload_file_count": len(all_assets),
            "payload_byte_count": sum(item.byte_count for item in all_assets),
            "family_instance_counts": expected,
            "source_viewer": {
                "path": "../viewer.glb",
                "sha256": viewer["sha256"],
                "byte_count": viewer["byte_count"],
            },
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
        _write_json(staging / RECEIPT_NAME, receipt)
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / RECEIPT_NAME


def build_tiled_viewer_package(
    job_root: Path | str,
    *,
    blender: Path | str,
    timeout_seconds: int = 1_800,
) -> Path:
    """Build the production viewer directly from sealed zone/tile artefacts."""

    from build_tiled_viewer_from_sealed import build_tiled_viewer_from_sealed

    return build_tiled_viewer_from_sealed(
        job_root, blender=blender, timeout_seconds=timeout_seconds
    )


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        values = values[values.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument(
        "--blender",
        type=Path,
        default=os.environ.get("FIREVIEWER_BLENDER"),
        help="Blender executable used only to export shared sealed prototypes",
    )
    parser.add_argument(
        "--prototype-timeout-seconds", type=int, default=1_800
    )
    parser.add_argument(
        "--from-monolithic-oracle",
        action="store_true",
        help="Use the legacy viewer.glb slicing path for small-map comparison only",
    )
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_arguments(argv)
    if options.from_monolithic_oracle:
        receipt = build_tiled_viewer_package_from_monolithic(options.job_root)
    else:
        if options.blender is None:
            raise TiledViewerPackageError(
                "--blender ou FIREVIEWER_BLENDER est requis pour le viewer tuilé direct"
            )
        receipt = build_tiled_viewer_package(
            options.job_root,
            blender=options.blender,
            timeout_seconds=options.prototype_timeout_seconds,
        )
    print(json.dumps({"tiled_viewer_receipt": str(receipt)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CATALOG_NAME",
    "INSTANCE_MAGIC",
    "INSTANCE_RECORD",
    "OUTPUT_DIRECTORY",
    "RECEIPT_NAME",
    "SCHEMA",
    "TiledViewerPackageError",
    "build_tiled_viewer_package",
    "build_tiled_viewer_package_from_monolithic",
    "validate_tiled_viewer_package",
]
