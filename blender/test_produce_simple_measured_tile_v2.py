from __future__ import annotations

from pathlib import Path

import pytest

import produce_simple_measured_tile_v2 as factual


def test_importing_factual_v2_does_not_replace_legacy_placement() -> None:
    assert factual.legacy.build_placement_inventory.__module__ == (
        "mns_mnt_placement_inventory"
    )


def test_factual_v2_forbids_degraded_mns_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factual, "_native_pair_from_inputs", lambda *_args: None)
    monkeypatch.setattr(
        factual.legacy,
        "produce_simple_measured_tile",
        lambda **_kwargs: pytest.fail("legacy fallback must not be invoked"),
    )

    with pytest.raises(
        factual.legacy.SimpleMeasuredTileError,
        match="publishing an empty degraded tile is forbidden",
    ):
        factual.produce_simple_measured_tile_v2(
            mnt_05m=Path("mnt.tif"),
            mns_05m=Path("mns.tif"),
            orthophoto_1m=Path("ortho.png"),
            elevation_source_receipt=Path("elevation.json"),
            orthophoto_source_receipt=Path("orthophoto.json"),
            placement_context=Path("context.json"),
            asset_library=Path("assets.json"),
            asset_roots={},
            portable_root=Path("."),
            output_root=Path("output"),
            zone_id="GPS-FAIL-CLOSED",
            tile_id="x700000_y6600000",
            tile_origin_l93_m=(700_000, 6_600_000),
        )
