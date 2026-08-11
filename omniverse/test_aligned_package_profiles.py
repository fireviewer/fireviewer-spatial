from __future__ import annotations

import json
from pathlib import Path

from build_die_contract_packages import align_runtime_contract
from fireviewer_accepted_visual_profiles import (
    PROFILE_CONTRACT_RELATIVE,
    PROFILE_LAYER_RELATIVE,
    write_accepted_visual_profile_artifacts,
)
from rebuild_die_aligned_packages import _write_reproduction_stage, _write_simulation_stage


def _write_minimal_flow(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        '''#usda 1.0
def Xform "World"
{
    def Xform "FireScenario"
    {
        def Xform "FlowVisual"
        {
            def FlowEmitterMesh "FrontRibbonEmitter"
            {
                float3 velocity = (-2, 4, 3.2)
            }
            def FlowEmitterMesh "SmokePlumeEmitter"
            {
                float3 velocity = (0, 0, 3.2)
            }
        }
    }
}
''',
        encoding="utf-8",
    )


def test_accepted_profiles_are_persisted_with_dynamic_velocity_contract(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _write_minimal_flow(package / "scenarios" / "flow.usda")
    contract = write_accepted_visual_profile_artifacts(package)

    layer = (package / PROFILE_LAYER_RELATIVE).read_text(encoding="utf-8")
    assert 'fireviewer:flow_capture_profile_id = "wildfire_convective_plume_mid_distance_v2"' in layer
    assert 'fireviewer:sky_capture_profile_id = "clear_daylight_smoke_contrast_v2"' in layer
    assert "float3 velocity = (-1.300000, 2.600000, 22.000000)" in layer
    assert "custom float smokeThreshold = 0.003" in layer
    assert "asset inputs:texture:file = @@" in layer
    assert contract["persistent_application"]["smoke_velocity_local_m_s"] == [-1.3, 2.6, 22.0]
    assert contract["flow_profile"]["profile_sha256"] == (
        "c121afd86498a8e44889ee3e37afdbd294f8ae83881bf4f65981fd359d212559"
    )
    assert contract["sky_profile"]["profile_sha256"] == (
        "ac5c4523ad510f80a0cd9b27c23194465f5ff15d1f7a0205ac4875a1281d6dee"
    )


def test_runtime_alignment_packages_storage_and_disables_automatic_capture(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _write_minimal_flow(package / "scenarios" / "flow.usda")
    visual_profiles = write_accepted_visual_profile_artifacts(package)
    runtime_path = package / "runtime" / "runtime-contract.json"
    runtime_path.write_text(
        json.dumps(
            {
                "required_extensions": ["omni.replicator.core", "omni.flowusd"],
                "capture": {"enabled": True},
                "simulation_playback": {"capture_on_first_launch": True},
            }
        ),
        encoding="utf-8",
    )

    runtime = align_runtime_contract(package, visual_profiles=visual_profiles)

    assert runtime["capture"]["enabled"] is False
    assert runtime["simulation_playback"]["capture_on_first_launch"] is False
    assert runtime["storage_module"]["path"] == "fireviewer_capture_storage.py"
    assert runtime["accepted_visual_profiles"]["persistent_layer"] == PROFILE_LAYER_RELATIVE


def test_download_entry_stages_compose_visual_profile_as_strongest_sublayer(
    tmp_path: Path,
) -> None:
    simulation = tmp_path / "simulation.usda"
    reproduction = tmp_path / "reproduction.usda"
    _write_simulation_stage(simulation)
    _write_reproduction_stage(reproduction)

    simulation_text = simulation.read_text(encoding="utf-8")
    reproduction_text = reproduction.read_text(encoding="utf-8")
    assert simulation_text.index(f"@{PROFILE_LAYER_RELATIVE}@") < simulation_text.index(
        "@dataset.accepted.usda@"
    )
    assert reproduction_text.index(f"@{PROFILE_LAYER_RELATIVE}@") < reproduction_text.index(
        "@map/map.usda@"
    )
