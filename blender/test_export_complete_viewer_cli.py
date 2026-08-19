from __future__ import annotations

import inspect

import export_complete_viewer_glb as viewer


def test_viewer_cli_main_accepts_default_argv() -> None:
    assert inspect.signature(viewer.main).parameters["argv"].default is None
