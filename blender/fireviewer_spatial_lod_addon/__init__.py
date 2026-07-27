"""Persistent bootstrap add-on for FireViewer Blender tile streaming."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import bpy
from bpy.app.handlers import persistent


bl_info = {
    "name": "FireViewer Spatial LOD",
    "author": "FireViewer",
    "version": (1, 0, 0),
    "blender": (4, 3, 0),
    "location": "Scene > FireViewer",
    "description": "Restores atomic FireViewer detail-tile streaming after load",
    "category": "Scene",
}


CONFIG_PROPERTY = "fireviewer_tile_streaming_config_json"
MODULE_DIRECTORY_PROPERTY = "fireviewer_runtime_module_directory_relative"
STATUS_PROPERTY = "fireviewer_tile_streaming_addon_status"
ERROR_PROPERTY = "fireviewer_tile_streaming_addon_error"
_runtime = None


def _activate_for_current_file() -> None:
    global _runtime
    scene = getattr(bpy.context, "scene", None)
    if scene is None or not scene.get(CONFIG_PROPERTY):
        return
    blend_path = Path(str(bpy.data.filepath))
    relative_module_directory = scene.get(MODULE_DIRECTORY_PROPERTY)
    if not blend_path.is_file() or not isinstance(relative_module_directory, str):
        scene[STATUS_PROPERTY] = "error"
        scene[ERROR_PROPERTY] = (
            "Saved blend path or relative module directory is missing"
        )
        return
    module_directory = (blend_path.parent / relative_module_directory).resolve()
    if not module_directory.is_dir():
        scene[STATUS_PROPERTY] = "error"
        scene[ERROR_PROPERTY] = (
            f"Runtime module directory is missing: {module_directory}"
        )
        return
    if str(module_directory) not in sys.path:
        sys.path.insert(0, str(module_directory))
    try:
        _runtime = importlib.import_module("blender_tile_streaming_runtime")
        _runtime.register(bpy)
    except Exception as error:  # Blender load handlers must remain alive.
        scene[STATUS_PROPERTY] = "error"
        scene[ERROR_PROPERTY] = f"{type(error).__name__}: {error}"
        return
    scene[STATUS_PROPERTY] = "registered"
    scene[ERROR_PROPERTY] = ""


@persistent
def _fireviewer_load_post(_unused: object) -> None:
    _activate_for_current_file()


def register() -> None:
    handlers = bpy.app.handlers.load_post
    if _fireviewer_load_post not in handlers:
        handlers.append(_fireviewer_load_post)
    _activate_for_current_file()


def unregister() -> None:
    global _runtime
    handlers = bpy.app.handlers.load_post
    if _fireviewer_load_post in handlers:
        handlers.remove(_fireviewer_load_post)
    if _runtime is not None:
        _runtime.unregister(bpy)
    _runtime = None
