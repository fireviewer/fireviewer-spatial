from __future__ import annotations

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
        "images": [{"bufferView": index} for index in range(images)],
        "materials": [{}],
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


def test_complete_viewer_rejects_missing_context_assets() -> None:
    with pytest.raises(viewer.CompleteViewerExportError, match="context_assets"):
        viewer._validate_gltf_payload(
            _payload(context=0),
            expected_counts={"buildings": 2, "trees": 3, "context_assets": 1},
            source_images=2,
            source_meshes=4,
        )


def test_complete_viewer_rejects_external_or_missing_textures() -> None:
    payload = _payload(images=1)
    with pytest.raises(viewer.CompleteViewerExportError, match="Textures viewer incomplètes"):
        viewer._validate_gltf_payload(
            payload,
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
