"""Open a complete Unity-parity forest scene in FireViewer USD Composer.

This script is run by Kit through ``--exec``.  It only authors the initial
review camera in Composer's session layer; the production USD and every forest
payload remain untouched.  The root stage is opened normally, so all payload
arcs in the selected variant remain available to Composer rather than a
near-only substitute.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import omni.kit.app
import omni.kit.viewport.utility
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


def _scene_bounds(stage: Usd.Stage) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """Return terrain bounds without requiring an invented map extent."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for candidate in ("/Terrain/GlobalMnt", "/Terrain"):
        prim = stage.GetPrimAtPath(candidate)
        if not prim.IsValid():
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not aligned.IsEmpty():
            return aligned.GetMin(), aligned.GetMax()
    # Safe fallback only if the terrain wrapper has unexpectedly changed.
    return Gf.Vec3d(-7500.0, -7500.0, 0.0), Gf.Vec3d(7500.0, 7500.0, 1200.0)


def _look_at(eye: Gf.Vec3d, target: Gf.Vec3d) -> Gf.Matrix4d:
    """Build the USD camera matrix (right, up, -forward, translation rows)."""
    forward = (target - eye).GetNormalized()
    world_up = Gf.Vec3d(0.0, 0.0, 1.0)
    right = Gf.Cross(forward, world_up).GetNormalized()
    if right.GetLength() < 1e-6:
        right = Gf.Vec3d(1.0, 0.0, 0.0)
    up = Gf.Cross(right, forward).GetNormalized()
    back = -forward
    return Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        back[0], back[1], back[2], 0.0,
        eye[0], eye[1], eye[2], 1.0,
    )


def _forest_tiles(forest_index_path: Path) -> list[tuple[str, Gf.Vec2d]]:
    """Read all delivered payload headers without composing tree payloads."""
    layer = Sdf.Layer.FindOrOpen(str(forest_index_path))
    if layer is None:
        raise RuntimeError(f"cannot open forest payload index: {forest_index_path}")
    tiles_root = layer.GetPrimAtPath("/UnityForest/Tiles")
    if tiles_root is None:
        raise RuntimeError("Unity-parity forest index is missing its tiled root")
    result: list[tuple[str, Gf.Vec2d]] = []
    for tile in tiles_root.nameChildren:
        bounds_json = tile.customData.get("fireviewer", {}).get("bounds_l93_m_json")
        if not bounds_json:
            continue
        xmin, ymin, xmax, ymax = (float(value) for value in json.loads(bounds_json))
        translation = (0.0, 0.0, 0.0)
        for property_spec in tile.properties:
            if property_spec.name == "xformOp:translate":
                translation = property_spec.default
                break
        scene_path = f"/UnityParityScene/Forest/Tiles/{tile.name}"
        result.append(
            (
                scene_path,
                Gf.Vec2d(
                    float(translation[0]) + (xmax - xmin) * 0.5,
                    float(translation[1]) + (ymax - ymin) * 0.5,
                ),
            )
        )
    if not result:
        raise RuntimeError("Unity-parity stage exposes no forest tile payloads")
    return result


def _camera_ground_focus(stage: Usd.Stage, camera_path: str, ground_z: float) -> tuple[Gf.Vec2d, float]:
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(camera_path))
    eye = matrix.ExtractTranslation()
    # USD camera row 2 is -forward (see `_look_at`).
    forward = -Gf.Vec3d(matrix[2][0], matrix[2][1], matrix[2][2]).GetNormalized()
    if forward[2] < -1e-4:
        distance = max(0.0, (ground_z - eye[2]) / forward[2])
    else:
        distance = 1000.0
    focus = eye + forward * distance
    return Gf.Vec2d(focus[0], focus[1]), max(0.0, eye[2] - ground_z)


async def _stream_visible_forest(
    stage: Usd.Stage,
    camera_path: str,
    tiles: list[tuple[str, Gf.Vec2d]],
    ground_z: float,
    initial_loaded: set[str],
) -> None:
    """Keep every forest tile delivered, while loading only the visible ring.

    Composer previously tried to build geometry processors for all 1.98 million
    detailed tree instances at once and crashed in ``rtx.hydra``.  This session
    controller keeps the complete forest in the stage manifest and loads the
    camera-visible tile ring (with a small guard band) as the user navigates.
    It never removes a tree record or swaps a tree for a primitive.
    """
    loaded = set(initial_loaded)
    centers = {path: center for path, center in tiles}
    max_loaded_tiles = 48
    while True:
        await omni.kit.app.get_app().next_update_async()
        focus, altitude = _camera_ground_focus(stage, camera_path, ground_z)
        radius = min(2100.0, max(1200.0, 750.0 + altitude * 0.72))
        ordered = sorted(
            ((center - focus).GetLength(), path)
            for path, center in tiles
        )
        desired = {path for distance, path in ordered if distance <= radius}
        if len(desired) > max_loaded_tiles:
            desired = {path for _, path in ordered[:max_loaded_tiles]}
        keep_radius = radius * 1.35
        for path in tuple(loaded):
            center = centers[path]
            if path not in desired and (center - focus).GetLength() > keep_radius:
                stage.Unload(path)
                loaded.remove(path)
        for path in desired - loaded:
            stage.Load(path)
            loaded.add(path)
        # Payload transitions and renderer preparation are intentionally given
        # room between reevaluations; this prevents repeated compose churn.
        for _ in range(29):
            await omni.kit.app.get_app().next_update_async()


async def _open_scene() -> None:
    raw_scene = os.getenv("FW_OMNIVERSE_SCENE", "").strip()
    raw_receipt = os.getenv("FW_OMNIVERSE_OPEN_RECEIPT", "").strip()
    if not raw_scene or not raw_receipt:
        raise RuntimeError("missing FW_OMNIVERSE_SCENE or FW_OMNIVERSE_OPEN_RECEIPT")

    scene_path = Path(raw_scene).resolve()
    receipt_path = Path(raw_receipt).resolve()
    if not scene_path.is_file():
        raise RuntimeError(f"Unity-parity scene is absent: {scene_path}")

    await omni.kit.app.get_app().next_update_async()
    # Keep the complete forest declared in USD but do not submit two million
    # detailed tree instances to the renderer in one composition transaction.
    result, error = await omni.usd.get_context().open_stage_async(
        str(scene_path), load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE
    )
    if not result:
        raise RuntimeError(f"Composer could not open Unity-parity scene: {error}")

    # Let composition establish the terrain before deriving the review view.
    for _ in range(8):
        await omni.kit.app.get_app().next_update_async()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Composer did not expose the opened USD stage")
    minimum, maximum = _scene_bounds(stage)
    # Fictional road/building overlays are few and always visible at this
    # review scale, so they are intentionally loaded in full.
    stage.Load("/UnityParityScene/VariantOverlay")
    center = (minimum + maximum) * 0.5
    span = max(maximum[0] - minimum[0], maximum[1] - minimum[1], 1.0)
    review_span = min(2200.0, max(1200.0, span * 0.16))
    target = Gf.Vec3d(center[0], center[1], minimum[2] + (maximum[2] - minimum[2]) * 0.24)
    eye = Gf.Vec3d(
        center[0] - review_span * 1.10,
        center[1] - review_span * 1.35,
        maximum[2] + review_span * 0.86,
    )
    camera_path = "/FireViewerSession/CompleteForestReviewCamera"
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, "/FireViewerSession")
        camera = UsdGeom.Camera.Define(stage, camera_path)
        xformable = UsdGeom.Xformable(camera)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(_look_at(eye, target))
        camera.CreateFocalLengthAttr().Set(22.0)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, float(max(span * 20.0, 100000.0))))

    viewport = omni.kit.viewport.utility.get_active_viewport()
    if viewport is None:
        raise RuntimeError("Composer has no active viewport")
    viewport.camera_path = camera_path
    await omni.kit.app.get_app().next_update_async()

    forest_index_path = scene_path.parent.parent / "forests" / scene_path.stem / "forest-index.usda"
    tiles = _forest_tiles(forest_index_path)
    # Populate the first complete visible ring before exposing the review
    # camera.  The remaining delivered tiles stream as the camera moves.
    initial_radius = 1600.0
    initial_candidates = sorted(
        ((center - Gf.Vec2d(target[0], target[1])).GetLength(), path)
        for path, center in tiles
    )
    initial_loaded = {
        path for distance, path in initial_candidates[:32] if distance <= initial_radius
    }
    for path in initial_loaded:
        stage.Load(path)
    for _ in range(8):
        await omni.kit.app.get_app().next_update_async()
    asyncio.ensure_future(
        _stream_visible_forest(stage, camera_path, tiles, minimum[2], initial_loaded)
    )

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "opened_at": datetime.now(UTC).isoformat(),
                "root_usd": str(scene_path),
                "camera_path": camera_path,
                "terrain_bounds": {"minimum": list(minimum), "maximum": list(maximum)},
                "forest_delivery": "complete_forest_tiled_payloads",
                "editor_load_policy": "visible_tile_streaming_without_tree_substitution",
                "initial_payload_tile_cap": 64,
                "initial_loaded_tile_count": len(initial_loaded),
                "delivered_forest_tile_count": len(tiles),
                "state": "opened_for_visual_review",
                "human_review": "pending",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


asyncio.ensure_future(_open_scene())
