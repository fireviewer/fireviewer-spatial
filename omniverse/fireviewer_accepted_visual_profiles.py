"""Package the accepted FireViewer Flow and sky capture profiles.

The production runner applies these values to an anonymous session layer.  A
downloadable package also needs a persistent layer so that a regular USD open
starts from the same accepted smoke and sky settings.  Per-state plume scaling
and truth alignment remain runtime operations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from run_fireviewer_replicator_dataset import (
    flow_capture_plume_profile_contract,
    sky_capture_profile_contract,
)


PROFILE_LAYER_RELATIVE = "appearance/accepted_capture_profiles.usda"
PROFILE_CONTRACT_RELATIVE = "runtime/accepted-visual-profiles.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _front_velocity(flow_layer: Path) -> tuple[float, float, float]:
    text = flow_layer.read_text(encoding="utf-8")
    front_marker = 'def FlowEmitterMesh "FrontRibbonEmitter"'
    smoke_marker = 'def FlowEmitterMesh "SmokePlumeEmitter"'
    front_start = text.find(front_marker)
    smoke_start = text.find(smoke_marker, front_start + len(front_marker))
    if front_start < 0 or smoke_start < 0:
        raise ValueError(f"Flow layer is missing the accepted mesh emitters: {flow_layer}")
    section = text[front_start:smoke_start]
    matches = re.findall(
        r"\bfloat3\s+velocity\s*=\s*\(\s*"
        r"([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)",
        section,
    )
    if not matches:
        raise ValueError(f"Front Flow emitter has no authored velocity: {flow_layer}")
    return tuple(float(value) for value in matches[-1])


def _usd_layer(*, smoke_velocity: tuple[float, float, float], flow_sha: str, sky_sha: str) -> str:
    vx, vy, vz = smoke_velocity
    return f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

over "World"
{{
    custom string fireviewer:flow_capture_profile_id = "wildfire_convective_plume_mid_distance_v2"
    custom string fireviewer:flow_capture_profile_sha256 = "{flow_sha}"
    custom string fireviewer:sky_capture_profile_id = "clear_daylight_smoke_contrast_v2"
    custom string fireviewer:sky_capture_profile_sha256 = "{sky_sha}"
    custom string fireviewer:profile_application = "persistent_static_baseline_plus_runtime_state_scaling"

    over "SkyFill"
    {{
        asset inputs:texture:file = @@
        color3f inputs:color = (0.58, 0.72, 0.95)
        float inputs:intensity = 180
        float inputs:exposure = 0
    }}

    over "FireScenario"
    {{
        over "FlowVisual"
        {{
            over "Simulate"
            {{
                custom uint blockMinLifetime = 24
                custom uint velocitySubSteps = 3
                over "advection"
                {{
                    custom float buoyancyMaxSmoke = 1.4
                    custom float buoyancyPerSmoke = 7
                    over "smoke"
                    {{
                        custom float damping = 0.01
                        float fade = 0.006
                    }}
                    over "velocity"
                    {{
                        custom float damping = 0.004
                        float fade = 0.02
                    }}
                    over "temperature"
                    {{
                        custom float damping = 0.02
                        float fade = 0.18
                    }}
                }}
                over "vorticity"
                {{
                    custom float smokeMask = 0.8
                }}
                over "summaryAllocate"
                {{
                    custom float smokeThreshold = 0.003
                    custom float speedThreshold = 0.2
                }}
            }}
            over "SmokePlumeEmitter"
            {{
                custom float minDistance = -0.6
                custom float maxDistance = 1.6
                float temperature = 0.45
                float coupleRateTemperature = 2.2
                custom float smoke = 0.95
                custom float coupleRateSmoke = 4
                float3 velocity = ({vx:.6f}, {vy:.6f}, {vz:.6f})
                custom float coupleRateVelocity = 4.2
                custom float divergence = 0.25
                custom float coupleRateDivergence = 1.5
            }}
            over "Offscreen"
            {{
                over "colormap"
                {{
                    custom float colorScale = 3.3
                    float4[] rgbaPoints = [
                        (0.025, 0.03, 0.035, 0.04),
                        (0.24, 0.26, 0.28, 0.32),
                        (0.55, 0.54, 0.52, 0.58),
                        (1.2, 0.10, 0.008, 0.78),
                        (12, 2.2, 0.08, 0.92),
                        (48, 17, 2.4, 0.84)
                    ]
                }}
            }}
            over "Render"
            {{
                over "rayMarch"
                {{
                    custom float attenuation = 8.5
                    custom float colorScale = 1.25
                }}
            }}
        }}
    }}
}}
'''


def write_accepted_visual_profile_artifacts(
    package_root: Path,
    *,
    flow_layer_relative: str = "scenarios/flow.usda",
) -> dict[str, Any]:
    """Write the persistent USD layer and its hash-addressed runtime contract."""

    package_root = package_root.resolve()
    flow_layer = package_root / flow_layer_relative
    if not flow_layer.is_file():
        raise FileNotFoundError(flow_layer)
    flow_profile = flow_capture_plume_profile_contract()
    sky_profile = sky_capture_profile_contract()
    front_velocity = _front_velocity(flow_layer)
    velocity_contract = flow_profile["dynamic_smoke_velocity"]
    smoke_velocity = (
        front_velocity[0] * float(velocity_contract["horizontal_front_wind_scale"]),
        front_velocity[1] * float(velocity_contract["horizontal_front_wind_scale"]),
        float(velocity_contract["vertical_m_s"]),
    )

    layer_path = package_root / PROFILE_LAYER_RELATIVE
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    layer_path.write_text(
        _usd_layer(
            smoke_velocity=smoke_velocity,
            flow_sha=str(flow_profile["profile_sha256"]),
            sky_sha=str(sky_profile["profile_sha256"]),
        ),
        encoding="utf-8",
        newline="\n",
    )
    contract = {
        "schema": "fireviewer.accepted-visual-profiles-package.v1",
        "source_session_profiles_unchanged": True,
        "persistent_application": {
            "layer": PROFILE_LAYER_RELATIVE,
            "layer_sha256": sha256_file(layer_path),
            "flow_layer": flow_layer_relative,
            "flow_layer_sha256": sha256_file(flow_layer),
            "mode": "static_baseline_in_usd_dynamic_state_scaling_and_truth_alignment_in_runtime",
            "smoke_velocity_local_m_s": [round(value, 6) for value in smoke_velocity],
        },
        "flow_profile": flow_profile,
        "sky_profile": sky_profile,
    }
    contract_path = package_root / PROFILE_CONTRACT_RELATIVE
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(canonical_json(contract), encoding="utf-8", newline="\n")
    return contract
