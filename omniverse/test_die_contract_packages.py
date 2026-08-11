from __future__ import annotations

import sys
from pathlib import Path

from shapely.geometry import Polygon


OMNIVERSE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OMNIVERSE_ROOT))

from build_die_contract_packages import (  # noqa: E402
    assert_no_forbidden_map_content,
    perimeter_mesh_data,
    usd_asset_issues,
    write_map_site,
    write_map_stage,
    write_perimeter_timeline,
)


class ConstantSampler:
    def at(self, _east: float, _north: float) -> float:
        return 100.0


def test_perimeter_mesh_is_terrain_draped_and_traced() -> None:
    geometry = Polygon(
        [
            (884000.0, 6408000.0),
            (884100.0, 6408000.0),
            (884100.0, 6408100.0),
            (884000.0, 6408100.0),
            (884000.0, 6408000.0),
        ]
    )
    mesh_points, face_counts, face_indices, trace_points, curve_counts = perimeter_mesh_data(
        geometry,
        sampler=ConstantSampler(),  # type: ignore[arg-type]
        anchor=(884000.0, 6408000.0),
        terrain_offset_m=1.0,
    )
    assert mesh_points
    assert all(point[2] == 101.0 for point in mesh_points)
    assert len(face_indices) == len(face_counts) * 3
    assert trace_points
    assert all(point[2] == 101.08 for point in trace_points)
    assert curve_counts == [5]


def test_pure_map_stage_excludes_camera_perimeter_and_simulation(tmp_path: Path) -> None:
    package = tmp_path / "map"
    write_map_site(package / "site" / "site.usda", source_package_id="source-r4")
    write_map_stage(package / "map.usda")
    assert_no_forbidden_map_content(package)
    text = (package / "site" / "site.usda").read_text(encoding="utf-8")
    assert "CameraCandidates" not in text
    assert "OcclusionProxies" not in text
    assert "prepend payload" not in text
    assert text.count("prepend references = @payloads/") == 5
    for component in ("Terrain", "Buildings", "Routes", "VegetationContext", "Vegetation"):
        assert f'"{component}"' in text


def test_usd_asset_scan_rejects_escape_and_accepts_local_dependency(tmp_path: Path) -> None:
    package = tmp_path / "bundle"
    (package / "layers").mkdir(parents=True)
    (package / "layers" / "child.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (package / "root.usda").write_text(
        "#usda 1.0\n( subLayers = [@layers/child.usda@] )\n",
        encoding="utf-8",
    )
    assert usd_asset_issues(package) == []
    (package / "root.usda").write_text(
        "#usda 1.0\n( subLayers = [@../outside.usda@] )\n",
        encoding="utf-8",
    )
    assert any("escaping USD dependency" in error for error in usd_asset_issues(package))


def test_perimeter_timeline_selects_one_daily_layer(tmp_path: Path) -> None:
    states = [
        {"local_date": "2026-07-03"},
        {"local_date": "2026-07-04"},
        {"local_date": "2026-07-05"},
    ]
    path = tmp_path / "perimeters.usda"
    write_perimeter_timeline(path, states=states)
    text = path.read_text(encoding="utf-8")
    assert "endTimeCode = 180.000" in text
    assert '60.000: "inherited"' in text
    assert '120.000: "invisible"' in text
    assert text.count("prepend references = @states/perimeter_") == 3
