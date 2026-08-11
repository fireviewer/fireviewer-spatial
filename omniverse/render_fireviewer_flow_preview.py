"""Render one RTX/Flow preview from a generated FireViewer USD dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--state", default="state_05")
    result.add_argument("--camera", default="CAM_01")
    result.add_argument("--resolution", default="960x540")
    result.add_argument("--warmup-frames", type=int, default=240)
    return result


def resolution(value: str) -> tuple[int, int]:
    width_text, height_text = value.lower().split("x", 1)
    width, height = int(width_text), int(height_text)
    if width < 16 or height < 16:
        raise ValueError("preview resolution must be at least 16x16")
    return width, height


def enable_extension(app, extension_name: str) -> None:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(extension_name):
        manager.set_extension_enabled_immediate(extension_name, True)
    for _ in range(60):
        app.update()
        if manager.is_extension_enabled(extension_name):
            return
    raise RuntimeError(f"Isaac Sim did not enable required extension: {extension_name}")


def run(args: argparse.Namespace) -> int:
    stage_path = args.stage.resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    if args.warmup_frames < 1:
        raise ValueError("--warmup-frames must be positive")
    width, height = resolution(args.resolution)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    render_product = None
    annotator = None
    try:
        enable_extension(app, "omni.flowusd")
        enable_extension(app, "omni.replicator.core")
        import carb.settings
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image

        settings = carb.settings.get_settings()
        settings.set("/persistent/app/usd/muteUsdDiagnostics", False)
        settings.set("/rtx/flow/enabled", True)
        settings.set("/rtx/flow/pathTracingEnabled", True)
        context = omni.usd.get_context()
        context.open_stage(str(stage_path))
        stage = None
        for _ in range(600):
            app.update()
            stage = context.get_stage()
            if stage is not None:
                break
        if stage is None:
            raise RuntimeError("Isaac Sim did not open the FireViewer dataset stage")
        stage.Load()
        for _ in range(1200):
            app.update()
            if context.get_stage_loading_status()[2] == 0:
                break
        else:
            raise RuntimeError("Isaac Sim did not finish loading the FireViewer dataset stage")
        scenario = stage.GetPrimAtPath("/World/FireScenario")
        if not scenario.IsValid():
            roots = [str(child.GetPath()) for child in stage.GetPseudoRoot().GetChildren()]
            world = stage.GetPrimAtPath("/World")
            world_children = [str(child.GetPath()) for child in world.GetChildren()] if world.IsValid() else []
            sublayers = list(stage.GetRootLayer().subLayerPaths)
            raise RuntimeError(f"dataset has no composed FireScenario prim; roots: {roots}; world_children: {world_children}; sublayers: {sublayers}")
        variants = scenario.GetVariantSets().GetVariantSet("fire_state")
        if args.state not in variants.GetVariantNames():
            raise ValueError(f"unknown fire state: {args.state}")
        variants.SetVariantSelection(args.state)
        camera_path = f"/World/Cameras/{args.camera}"
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            raise ValueError(f"unknown fixed camera: {args.camera}")
        for _ in range(45):
            app.update()
        render_product = rep.create.render_product(camera_path, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(args.warmup_frames):
            app.update()
        timeline.stop()
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=True)
        rep.orchestrator.wait_until_complete()
        for _ in range(24):
            app.update()
        image = np.asarray(annotator.get_data())
        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(f"RGB annotator returned invalid preview shape: {image.shape}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image[:, :, :3].astype(np.uint8), mode="RGB").save(args.output)
        flow_types: dict[str, int] = {}
        for prim in stage.Traverse():
            type_name = str(prim.GetTypeName())
            if type_name.startswith("Flow"):
                flow_types[type_name] = flow_types.get(type_name, 0) + 1
        metadata_path = args.output.with_suffix(".json")
        metadata_path.write_text(json.dumps({
            "stage": str(stage_path),
            "state": args.state,
            "camera": args.camera,
            "warmup_frames": args.warmup_frames,
            "resolution": [width, height],
            "flow_types": flow_types,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "flow_preview_rendered", "image": str(args.output), "metadata": str(metadata_path), "flow_types": flow_types}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".error.json").write_text(json.dumps({
            "status": "flow_preview_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if annotator is not None and render_product is not None:
            annotator.detach([render_product])
        if render_product is not None:
            render_product.destroy()
        app.close()


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".error.json").write_text(json.dumps({
            "status": "flow_preview_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"FireViewer Flow preview failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
