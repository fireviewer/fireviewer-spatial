"""Open a generated FireViewer dataset stage in an already-started Kit session.

Designed for ``kit.exe <experience>.kit --exec``.  The stage path is supplied
through ``FIREVIEWER_STAGE_PATH`` so the stage remains outside this script and
no source USD is modified.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Sdf, UsdGeom


stage_path = Path(os.environ["FIREVIEWER_STAGE_PATH"]).resolve()
if not stage_path.is_file():
    raise FileNotFoundError(stage_path)

context = omni.usd.get_context()
context.open_stage(str(stage_path))
print(f"FireViewer dataset stage open requested: {stage_path}")


async def activate_default_camera() -> None:
    """Wait for stage composition, then frame the dataset with an authored camera."""

    camera_path = Sdf.Path(
        os.environ.get("FIREVIEWER_DEFAULT_CAMERA_PATH", "/World/Cameras/CAM_09")
    )
    app = omni.kit.app.get_app()

    for _ in range(600):
        await app.next_update_async()
        stage = context.get_stage()
        viewport = get_active_viewport()
        if stage is None or viewport is None:
            continue

        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            continue

        viewport.camera_path = camera_path
        print(f"FireViewer default viewport camera activated: {camera_path}")
        return

    print(f"FireViewer default viewport camera unavailable: {camera_path}")


asyncio.ensure_future(activate_default_camera())
