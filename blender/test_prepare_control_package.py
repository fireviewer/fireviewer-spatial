from __future__ import annotations

from types import SimpleNamespace

from prepare_control_package import (
    NON_INCIDENT_PERIMETER_ROLE,
    should_render_fire_perimeter,
)


def _feature(role: str) -> SimpleNamespace:
    return SimpleNamespace(properties={"role": role})


def test_cems_activation_aoi_is_used_for_clipping_without_a_fire_ring() -> None:
    assert not should_render_fire_perimeter(
        [_feature(NON_INCIDENT_PERIMETER_ROLE)],
        explicitly_omitted=False,
    )


def test_incident_perimeter_keeps_legacy_ring_unless_explicitly_omitted() -> None:
    features = [_feature("observed-burn-perimeter")]

    assert should_render_fire_perimeter(features, explicitly_omitted=False)
    assert not should_render_fire_perimeter(features, explicitly_omitted=True)
