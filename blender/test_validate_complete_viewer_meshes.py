from __future__ import annotations

import pytest

from validate_complete_viewer_meshes import CompleteViewerMeshError, require_mesh_coverage


def test_mesh_coverage_accepts_all_unique_source_meshes() -> None:
    require_mesh_coverage(12, 12)
    require_mesh_coverage(12, 17)


def test_mesh_coverage_rejects_missing_source_meshes() -> None:
    with pytest.raises(CompleteViewerMeshError, match="incomplets"):
        require_mesh_coverage(12, 11)


def test_mesh_coverage_rejects_empty_source_scene() -> None:
    with pytest.raises(CompleteViewerMeshError, match="aucun mesh source"):
        require_mesh_coverage(0, 1)
