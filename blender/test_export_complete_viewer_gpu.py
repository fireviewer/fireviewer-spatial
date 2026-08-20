from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import export_complete_viewer_glb_gpu as viewer


def test_gpu_exporter_resolves_sibling_base_without_process_pythonpath(
    tmp_path: Path,
) -> None:
    script = Path(viewer.__file__).resolve()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy,sys; sys.path[:]=[p for p in sys.path if p != sys.argv[1]]; "
            "runpy.run_path(sys.argv[2], run_name='fireviewer_gpu_import_smoke')",
            str(script.parent),
            str(script),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _gpu_payload(*, buildings: int = 2, trees: int = 3, context: int = 1):
    nodes = [
        {"name": viewer.FAMILY_ROOTS["buildings"], "children": [3]},
        {"name": viewer.FAMILY_ROOTS["trees"], "children": [4]},
        {"name": viewer.FAMILY_ROOTS["context_assets"], "children": [5]},
    ]
    for index in range(3):
        nodes.append(
            {
                "mesh": 0,
                "extensions": {
                    "EXT_mesh_gpu_instancing": {
                        "attributes": {"TRANSLATION": index}
                    }
                },
            }
        )
    return {
        "nodes": nodes,
        "accessors": [
            {"count": buildings},
            {"count": trees},
            {"count": context},
        ],
        "extensionsUsed": ["EXT_mesh_gpu_instancing"],
    }


def test_gpu_counts_are_exact() -> None:
    assert viewer._validate_exact_gpu_counts(
        _gpu_payload(),
        {"buildings": 2, "trees": 3, "context_assets": 1},
    ) == {"buildings": 2, "trees": 3, "context_assets": 1}


def test_gpu_counts_include_singleton_batches_serialized_as_regular_meshes() -> None:
    payload = _gpu_payload(trees=3)
    tree_root = payload["nodes"][1]
    tree_root["children"] = [4, 6, 7]
    payload["nodes"].extend(
        [
            {"mesh": 1},
            {"mesh": 2},
        ]
    )
    assert viewer._validate_exact_gpu_counts(
        payload,
        {"buildings": 2, "trees": 5, "context_assets": 1},
    ) == {"buildings": 2, "trees": 5, "context_assets": 1}


def test_gpu_counts_reject_one_missing_tree() -> None:
    with pytest.raises(viewer.CompleteViewerExportError, match="trees"):
        viewer._validate_exact_gpu_counts(
            _gpu_payload(trees=2),
            {"buildings": 2, "trees": 3, "context_assets": 1},
        )


def test_gpu_export_rejects_non_instanced_payload() -> None:
    payload = _gpu_payload()
    payload["extensionsUsed"] = []
    with pytest.raises(viewer.CompleteViewerExportError, match="EXT_mesh_gpu_instancing"):
        viewer._validate_exact_gpu_counts(
            payload,
            {"buildings": 2, "trees": 3, "context_assets": 1},
        )


def test_grouping_preserves_all_logical_instances() -> None:
    tree_a = SimpleNamespace(name="Douglas", type="MESH")
    tree_b = SimpleNamespace(name="Pine", type="MESH")
    house = SimpleNamespace(name="House", type="MESH")
    context = SimpleNamespace(name="Lamp", type="MESH")
    snapshots = [
        *(("trees", tree_a, object()) for _ in range(3000)),
        *(("trees", tree_b, object()) for _ in range(2000)),
        *(("buildings", house, object()) for _ in range(900)),
        *(("context_assets", context, object()) for _ in range(40)),
    ]
    groups = viewer._group_instance_snapshots(snapshots)
    assert len(groups) == 4
    assert viewer._count_groups(groups) == {
        "buildings": 900,
        "trees": 5000,
        "context_assets": 40,
    }
    assert sum(len(matrices) for _family, _source, matrices in groups) == 5940
