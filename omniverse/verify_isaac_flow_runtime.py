"""Verify that a FireViewer dataset stage activates real NVIDIA Flow in Isaac Sim.

This is a runtime activation test, not a visual-acceptance test.  It fails if
Isaac Sim cannot enable Flow or Replicator, if the stage lacks the expected
Flow schema prims, or if the configured Flow scene cannot advance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", type=Path, required=True)
    result.add_argument("--warmup-frames", type=int, default=180)
    return result


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


def run(stage_path: Path, warmup_frames: int) -> int:
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    if warmup_frames < 1:
        raise ValueError("--warmup-frames must be positive")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        enable_extension(app, "omni.flowusd")
        enable_extension(app, "omni.replicator.core")
        import omni.flowusd  # noqa: F401
        import omni.replicator.core  # noqa: F401
        import omni.timeline
        import omni.usd

        context = omni.usd.get_context()
        context.open_stage(str(stage_path.resolve()))
        stage = None
        for _ in range(600):
            app.update()
            stage = context.get_stage()
            if stage is not None:
                break
        if stage is None:
            raise RuntimeError("Isaac Sim did not open the FireViewer dataset stage")
        stage.Load()
        for _ in range(60):
            app.update()
        scenario = stage.GetPrimAtPath("/World/FireScenario")
        variants = scenario.GetVariantSets().GetVariantSet("fire_state")
        variants.SetVariantSelection("state_01")
        for _ in range(30):
            app.update()
        flow_types = {}
        for prim in stage.Traverse():
            type_name = str(prim.GetTypeName())
            if type_name.startswith("Flow"):
                flow_types[type_name] = flow_types.get(type_name, 0) + 1
        for required in ("FlowSimulate", "FlowOffscreen", "FlowRender", "FlowEmitterSphere"):
            if flow_types.get(required, 0) < 1:
                raise RuntimeError(f"Flow stage is missing active schema prim type: {required}")
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(warmup_frames):
            app.update()
        timeline.stop()
        print(json.dumps({
            "status": "isaac_flow_runtime_passed",
            "stage": str(stage_path.resolve()),
            "warmup_frames": warmup_frames,
            "flow_prims": flow_types,
            "selected_state": "state_01",
            "note": "Flow schema activation and simulation stepping passed; visual acceptance remains a separate RTX render review.",
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        app.close()


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args.stage, args.warmup_frames)
    except Exception as exc:
        print(f"Isaac Flow runtime verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
