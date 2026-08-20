from __future__ import annotations

from types import SimpleNamespace

import pytest

import export_complete_viewer_glb as viewer


def _payload(*, buildings: int = 2, trees: int = 3, context: int = 1, images: int = 2):
    nodes = [
        {"name": viewer.FAMILY_ROOTS["buildings"], "children": list(range(3, 3 + buildings))},
        {"name": viewer.FAMILY_ROOTS["trees"], "children": list(range(3 + buildings, 3 + buildings + trees))},
        {"name": viewer.FAMILY_ROOTS["context_assets"], "children": list(range(3 + buildings + trees, 3 + buildings + trees + context))},
    ]
    nodes.extend({"mesh": 0} for _ in range(buildings + trees + context))
    return {
        "nodes": nodes,
        "meshes": [{}],
        "accessors": [],
        "buffers": [{"byteLength": 16}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": index, "byteLength": 1}
            for index in range(images)
        ],
        "images": [
            {"bufferView": index, "mimeType": "image/png"}
            for index in range(images)
        ],
        "textures": [{"source": index} for index in range(images)],
        "materials": (
            [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
            if images
            else [{}]
        ),
        "extensionsUsed": [],
    }


def test_complete_viewer_requires_exact_family_counts_and_embedded_images() -> None:
    summary = viewer._validate_gltf_payload(
        _payload(),
        expected_counts={"buildings": 2, "trees": 3, "context_assets": 1},
        source_images=2,
        source_meshes=4,
    )
    assert summary["family_instance_counts"] == {"buildings": 2, "trees": 3, "context_assets": 1}
    assert summary["image_count"] == 2
    assert summary["texture_count"] == 2
    assert summary["material_texture_reference_count"] == 1
    assert summary["source_to_exported_image_delta"] == 0


def test_complete_viewer_rejects_missing_context_assets() -> None:
    with pytest.raises(viewer.CompleteViewerExportError, match="context_assets"):
        viewer._validate_gltf_payload(
            _payload(context=0),
            expected_counts={"buildings": 2, "trees": 3, "context_assets": 1},
            source_images=2,
            source_meshes=4,
        )


def test_complete_viewer_allows_exporter_image_deduplication() -> None:
    payload = _payload(images=1)
    summary = viewer._validate_gltf_payload(
        payload,
        expected_counts={"buildings": 2, "trees": 3, "context_assets": 1},
        source_images=2,
        source_meshes=4,
    )
    assert summary["image_count"] == 1
    assert summary["source_to_exported_image_delta"] == 1


def test_complete_viewer_rejects_external_or_missing_textures() -> None:
    with pytest.raises(viewer.CompleteViewerExportError, match="aucune image embarquée"):
        viewer._validate_gltf_payload(
            _payload(images=0),
            expected_counts={"buildings": 2, "trees": 3, "context_assets": 1},
            source_images=2,
            source_meshes=4,
        )
    payload = _payload()
    payload["images"][0] = {"uri": "texture.png"}
    with pytest.raises(viewer.CompleteViewerExportError, match="texture externe"):
        viewer._validate_gltf_payload(
            payload,
            expected_counts={"buildings": 2, "trees": 3, "context_assets": 1},
            source_images=2,
            source_meshes=4,
        )


def test_complete_viewer_rejects_broken_texture_references() -> None:
    payload = _payload()
    payload["textures"][0]["source"] = 99
    with pytest.raises(viewer.CompleteViewerExportError, match="image invalide"):
        viewer._validate_gltf_payload(
            payload,
            expected_counts={"buildings": 2, "trees": 3, "context_assets": 1},
            source_images=2,
            source_meshes=4,
        )

    payload = _payload()
    payload["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"][
        "index"
    ] = 99
    with pytest.raises(viewer.CompleteViewerExportError, match="Index de texture"):
        viewer._validate_gltf_payload(
            payload,
            expected_counts={"buildings": 2, "trees": 3, "context_assets": 1},
            source_images=2,
            source_meshes=4,
        )


def test_snapshot_instances_does_not_retain_ephemeral_depsgraph_handles() -> None:
    class Matrix:
        def __init__(self, value: int) -> None:
            self.value = value

        def copy(self) -> tuple[str, int]:
            return ("matrix", self.value)

    class EphemeralInstance:
        def __init__(self, family: str, source: object, matrix: int) -> None:
            self.valid = True
            self._parent = SimpleNamespace(name=viewer.FAMILY_MARKERS[family])
            self._object = SimpleNamespace(
                name=f"{viewer.FAMILY_MARKERS[family]}Instance",
                original=source,
            )
            self._matrix = Matrix(matrix)

        def _require_valid(self) -> None:
            if not self.valid:
                raise ReferenceError("StructRNA of type DepsgraphObjectInstance has been removed")

        @property
        def is_instance(self) -> bool:
            self._require_valid()
            return True

        @property
        def parent(self):
            self._require_valid()
            return self._parent

        @property
        def object(self):
            self._require_valid()
            return self._object

        @property
        def matrix_world(self):
            self._require_valid()
            return self._matrix

    class EphemeralInstances:
        def __init__(self, items: list[EphemeralInstance]) -> None:
            self.items = items

        def __iter__(self):
            previous: EphemeralInstance | None = None
            for item in self.items:
                if previous is not None:
                    previous.valid = False
                previous = item
                yield item
            if previous is not None:
                previous.valid = False

    sources = [
        SimpleNamespace(name="Douglas", type="MESH"),
        SimpleNamespace(name="House", type="MESH"),
    ]

    stale = [
        EphemeralInstance("trees", sources[0], 10),
        EphemeralInstance("buildings", sources[1], 20),
    ]
    retained = list(EphemeralInstances(stale))
    with pytest.raises(ReferenceError, match="StructRNA"):
        _ = retained[0].is_instance

    fresh = [
        EphemeralInstance("trees", sources[0], 10),
        EphemeralInstance("buildings", sources[1], 20),
    ]
    bpy = SimpleNamespace(
        context=SimpleNamespace(
            evaluated_depsgraph_get=lambda: SimpleNamespace(
                object_instances=EphemeralInstances(fresh)
            )
        )
    )

    snapshots = viewer._snapshot_instances(bpy)

    assert [(family, source.name, matrix) for family, source, matrix in snapshots] == [
        ("trees", "Douglas", ("matrix", 10)),
        ("buildings", "House", ("matrix", 20)),
    ]
    assert all(not instance.valid for instance in fresh)
