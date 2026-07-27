"""Blender-facing final-scene bootstrap helpers with testable boundaries."""

from __future__ import annotations

import importlib
from pathlib import Path
import shutil
from typing import Any, Sequence
import zipfile


def restore_material_viewport(
    bpy_module: Any,
    *,
    local_focus: Sequence[float],
    view_distance_m: float,
) -> int:
    """Save at least one framed 3D viewport, including console-launched builds.

    Running a build through ``Shift+F4`` replaces the active 3D area with a
    Python Console.  Blender persists that editor type in the ``.blend``.  The
    active Console/Text Editor is therefore restored first, then every 3D
    viewport receives the same deterministic material-preview framing.
    """

    if len(local_focus) != 3:
        raise ValueError("local_focus must contain exactly three coordinates")
    distance = float(view_distance_m)
    if distance <= 0.0:
        raise ValueError("view_distance_m must be strictly positive")

    context = getattr(bpy_module, "context", None)
    screen = getattr(context, "screen", None)
    if screen is None:
        return 0
    areas = list(getattr(screen, "areas", ()))
    context_area = getattr(context, "area", None)
    if context_area in areas and getattr(context_area, "type", None) in {
        "CONSOLE",
        "TEXT_EDITOR",
    }:
        context_area.type = "VIEW_3D"
    elif not any(getattr(area, "type", None) == "VIEW_3D" for area in areas):
        if context_area in areas:
            context_area.type = "VIEW_3D"
        elif areas:
            areas[0].type = "VIEW_3D"

    configured = 0
    for area in areas:
        if getattr(area, "type", None) != "VIEW_3D":
            continue
        space = area.spaces.active
        space.shading.type = "MATERIAL"
        space.shading.use_scene_world = True
        space.shading.use_scene_lights = True
        space.clip_start = 0.1
        space.clip_end = 100_000.0
        region_3d = space.region_3d
        region_3d.view_location = tuple(float(value) for value in local_focus)
        region_3d.view_distance = distance
        if hasattr(region_3d, "view_perspective"):
            region_3d.view_perspective = "PERSP"
        configured += 1
    return configured


def install_persistent_addon(
    bpy_module: Any,
    *,
    source_directory: str | Path,
    module_name: str,
) -> Path:
    """Install and persist one local add-on in Blender's user scripts folder.

    A Text datablock marked ``use_module`` is still governed by Blender's
    Auto-Run security preference.  A user-enabled add-on is the supported way
    to restore the streaming handler on later Blender launches without
    weakening that global security preference.
    """

    source = Path(source_directory).resolve()
    if not source.is_dir() or not (source / "__init__.py").is_file():
        raise FileNotFoundError(f"Invalid Blender add-on source: {source}")
    if source.name != module_name:
        raise ValueError("The add-on directory name must equal module_name")

    user_addons = bpy_module.utils.user_resource("SCRIPTS", path="addons", create=True)
    if not user_addons:
        raise RuntimeError("Blender did not expose a writable user add-ons path")
    destination = Path(user_addons).resolve() / module_name
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if "__pycache__" in relative.parts or item.suffix == ".pyc":
            continue
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
    if copied == 0 or not (destination / "__init__.py").is_file():
        raise RuntimeError("The persistent Blender add-on copy is incomplete")

    importlib.invalidate_caches()
    addon_utils = importlib.import_module("addon_utils")
    addon_utils.modules(refresh=True)
    enabled_module = addon_utils.enable(
        module_name,
        default_set=True,
        persistent=True,
    )
    if (
        enabled_module is None
        or module_name not in bpy_module.context.preferences.addons
    ):
        raise RuntimeError(f"Blender did not enable add-on {module_name!r}")
    save_result = bpy_module.ops.wm.save_userpref()
    if "FINISHED" not in save_result:
        raise RuntimeError("Blender did not persist the enabled add-on preference")
    return destination


def package_persistent_addon(
    *,
    source_directory: str | Path,
    destination_zip: str | Path,
) -> Path:
    """Create the one-click add-on archive without requiring Blender settings."""

    source = Path(source_directory).resolve()
    if not source.is_dir() or not (source / "__init__.py").is_file():
        raise FileNotFoundError(f"Invalid Blender add-on source: {source}")
    destination = Path(destination_zip).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for item in sorted(source.rglob("*")):
            relative = item.relative_to(source)
            if (
                not item.is_file()
                or "__pycache__" in relative.parts
                or item.suffix == ".pyc"
            ):
                continue
            archive.write(item, Path(source.name) / relative)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("The persistent Blender add-on archive is empty")
    return destination


__all__ = [
    "install_persistent_addon",
    "package_persistent_addon",
    "restore_material_viewport",
]
