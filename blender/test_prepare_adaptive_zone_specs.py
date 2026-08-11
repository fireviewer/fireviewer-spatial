from __future__ import annotations

from collections import namedtuple
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

import prepare_adaptive_zone_specs as generator
import prepare_incident_terrains as legacy_incidents
from terrainctl import (
    TerrainController,
    build_zone_plan,
    load_zone_spec,
)


TEST_ROOT = Path(
    "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest/"
    "adaptive-zone-specs"
)
DiskUsage = namedtuple("DiskUsage", "total used free")


@pytest.fixture
def d_root() -> Path:
    root = TEST_ROOT / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.resolve().is_relative_to(TEST_ROOT.resolve()):
            shutil.rmtree(root, ignore_errors=True)


def _dependency_paths(root: Path) -> dict[str, Path]:
    dependency_root = root / "dependencies"
    dependency_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name in generator.DEPENDENCY_NAMES:
        path = dependency_root / f"{name}.json"
        payload = {"schema": f"fireviewer.fixture-{name}.v1", "status": "locked"}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result[name] = path
    return result


def _request(
    root: Path,
    *,
    dependencies: dict[str, Path] | None = None,
    bounds: tuple[int, int, int, int] = (700_000, 6_300_000, 701_500, 6_301_500),
    pilot_scores: dict[str, float] | None = None,
    regression_tile_id: str | None = None,
) -> generator.AdaptiveZoneSpecRequest:
    return generator.AdaptiveZoneSpecRequest(
        zone_id="FR-TEST-ADAPTIVE",
        revision="r1",
        bounds_l93_m=bounds,
        mnt=generator.SourceEndpoint(
            "https://data.example.invalid/wms-elevation", "RGEALTI-MNT-2M"
        ),
        mns=generator.SourceEndpoint(
            "https://data.example.invalid/wms-surface", "RGEALTI-MNS-2M"
        ),
        orthophoto=generator.OrthophotoEndpoint(
            "https://data.example.invalid/wms-orthophoto",
            "ORTHO-RGB-1M",
            "ign-ortho-2026-r1",
        ),
        source_revision_id="ign-elevation-2026-r1",
        dependency_artifacts=dependencies or _dependency_paths(root),
        work_root=root / "fireviewer-work",
        export_root=root / "fireviewer-exports",
        estimated_peak_bytes=1024**3,
        pilot_scores=pilot_scores,
        regression_tile_id=regression_tile_id,
    )


def test_locked_catalog_reuses_only_existing_bounds_in_production_order() -> None:
    catalog = generator.load_locked_zone_catalog()
    expected_order = [
        "FR-30-00001",
        "FR-34-00001",
        "FR-26-00001",
        "FR-83-00001",
        "FR-77-00001",
        "FR-66-00001",
    ]
    expected_tiles = [841, 1024, 1600, 2025, 2116, 3025]
    legacy_bounds = {
        case.fire_id: list(case.square_bounds_epsg2154_m)
        for case in legacy_incidents.CASES
    }

    assert catalog["production_order"] == expected_order
    assert [zone["zone_id"] for zone in catalog["zones"]] == expected_order
    assert [zone["tile_count"] for zone in catalog["zones"]] == expected_tiles
    assert all(
        zone["bounds_l93_m"] == legacy_bounds[zone["zone_id"]]
        for zone in catalog["zones"]
    )
    assert catalog["totals"] == {
        "zone_count": 6,
        "tile_count": 10_631,
        "source_pair_count": 10_631,
        "seam_count": 20_768,
    }
    assert catalog["bounds_provenance"] == {
        "source_module": "blender/prepare_incident_terrains.py",
        "source_symbol": "CASES.square_bounds_epsg2154_m",
        "reuse_scope": "incident_identity_and_square_bounds_only",
        "legacy_terrain_reused": False,
        "legacy_ground_reused": False,
        "legacy_asset_placements_reused": False,
    }


def test_catalog_grid_counts_are_derived_not_declared_only() -> None:
    catalog = generator.load_locked_zone_catalog()
    summaries = [
        generator.grid_summary(zone["bounds_l93_m"]) for zone in catalog["zones"]
    ]
    assert [summary["tile_count"] for summary in summaries] == [
        841,
        1024,
        1600,
        2025,
        2116,
        3025,
    ]
    assert sum(summary["tile_count"] for summary in summaries) == 10_631
    assert sum(summary["seam_count"] for summary in summaries) == 20_768


def test_spec_is_reproducible_and_derives_one_source_triplet_per_core(
    d_root: Path,
) -> None:
    dependencies = _dependency_paths(d_root)
    scores = {
        "x700500_y6300500_s500": 17.0,
        "x701000_y6301000_s500": 31.0,
    }
    first = generator.build_zone_spec(
        _request(
            d_root,
            dependencies=dependencies,
            pilot_scores=scores,
            regression_tile_id="x701000_y6301000_s500",
        )
    )
    second = generator.build_zone_spec(
        _request(
            d_root,
            dependencies=dict(reversed(list(dependencies.items()))),
            pilot_scores=dict(reversed(list(scores.items()))),
            regression_tile_id="x701000_y6301000_s500",
        )
    )

    assert first == second
    assert generator.canonical_sha256(first) == generator.canonical_sha256(second)
    assert first["source_resolution_m"] == 2
    assert len(first["source_requests"]) == 27
    assert [item["product"] for item in first["source_requests"]] == [
        *("mns" for _ in range(9)),
        *("mnt" for _ in range(9)),
        *("orthophoto" for _ in range(9)),
    ]
    pairs: dict[str, list[dict]] = {}
    for source in first["source_requests"]:
        pairs.setdefault(source["id"], []).append(source)
        assert "expected_sha256" not in source
        assert "expected_byte_count" not in source
        assert source["request"]["service_url"].startswith("https://")
        assert source["source_revision_id"] in {
            "ign-elevation-2026-r1",
            "ign-ortho-2026-r1",
        }
    assert len(pairs) == 9
    assert all(
        {source["product"] for source in pair} == {"mnt", "mns", "orthophoto"}
        and len({tuple(source["request"]["core_bounds_l93_m"]) for source in pair}) == 1
        for pair in pairs.values()
    )
    encoded = json.dumps(first, ensure_ascii=False).casefold()
    assert "0.5" not in encoded
    assert "orthophoto" in encoded
    orthophotos = [
        item for item in first["source_requests"] if item["product"] == "orthophoto"
    ]
    assert all(
        item["request"]["resolution_m"] == 1 and item["request"]["halo_m"] == 10
        for item in orthophotos
    )


def test_written_spec_loads_and_plan_dry_run_is_network_free(d_root: Path) -> None:
    output = d_root / "zones" / "FR-TEST-ADAPTIVE" / "zone-spec.v1.json"
    generator.write_zone_spec(_request(d_root), output)
    zone = load_zone_spec(output)
    plan = build_zone_plan(zone)
    controller = TerrainController(
        disk_usage=lambda _path: DiskUsage(100 * 1024**3, 0, 100 * 1024**3),
        coordinator_root=d_root / "coordinator",
    )
    dry_run = controller.run(output, phase="plan", mode="dry-run")

    assert dry_run["eligible"] is True
    assert dry_run["writes_performed"] is False
    assert dry_run["network_access_performed"] is False
    assert plan["summary"] == {
        "tile_count": 9,
        "seam_count": 12,
        "source_request_count": 27,
    }


def test_compact_mapping_resolves_relative_dependencies_on_d(d_root: Path) -> None:
    dependencies = _dependency_paths(d_root)
    config_root = d_root / "input"
    config_root.mkdir()
    mapping = {
        "schema": generator.INPUT_SCHEMA,
        "zone_id": "FR-TEST-ADAPTIVE",
        "revision": "r2",
        "bounds_l93_m": [700_000, 6_300_000, 701_000, 6_301_000],
        "sources": {
            "mnt": {
                "service_url": "https://data.example.invalid/mnt",
                "layer": "MNT-2M",
            },
            "mns": {
                "service_url": "https://data.example.invalid/mns",
                "layer": "MNS-2M",
            },
            "orthophoto": {
                "service_url": "https://data.example.invalid/orthophoto",
                "layer": "ORTHO-RGB-1M",
                "source_revision_id": "provider-ortho-r2",
                "service_kind": "wms",
                "image_format": "image/png",
                "maximum_download_bytes": 1048576,
            },
        },
        "source_revision_id": "provider-r2",
        "dependency_artifacts": {
            name: os.path.relpath(path, config_root)
            for name, path in dependencies.items()
        },
        "workspace": {
            "work_root": str(d_root / "work"),
            "export_root": str(d_root / "exports"),
        },
        "storage": {"estimated_peak_bytes": 4096},
    }
    request = generator.request_from_mapping(mapping, base=config_root)
    spec = generator.build_zone_spec(request)
    assert spec["bounds_l93_m"] == [700_000, 6_300_000, 701_000, 6_301_000]
    assert len(spec["source_requests"]) == 12
    assert all(
        Path(record["path"]).drive.upper() == "D:"
        for record in spec["dependency_artifacts"].values()
    )


@pytest.mark.parametrize(
    ("service_url", "layer", "message"),
    [
        ("http://data.example.invalid/mnt", "MNT-2M", "must be an HTTPS URL"),
        (
            "https://data.example.invalid/orthophoto",
            "MNT-2M",
            "orthophoto sources are forbidden",
        ),
        (
            "https://data.example.invalid/mnt",
            "ORTHOPHOTO-HD",
            "orthophoto sources are forbidden",
        ),
    ],
)
def test_non_https_and_orthophoto_sources_are_rejected(
    d_root: Path, service_url: str, layer: str, message: str
) -> None:
    request = _request(d_root)
    invalid = generator.AdaptiveZoneSpecRequest(
        **{
            **request.__dict__,
            "mnt": generator.SourceEndpoint(service_url, layer),
        }
    )
    with pytest.raises(generator.ZoneSpecGenerationError, match=message):
        generator.build_zone_spec(invalid)


def test_c_paths_and_tiles_outside_the_zone_are_rejected(d_root: Path) -> None:
    request = _request(d_root)
    with pytest.raises(generator.ZoneSpecGenerationError, match="must stay on D"):
        generator.build_zone_spec(
            generator.AdaptiveZoneSpecRequest(
                **{
                    **request.__dict__,
                    "work_root": Path("C:/fireviewer-work"),
                }
            )
        )
    with pytest.raises(generator.ZoneSpecGenerationError, match="outside the zone"):
        generator.build_zone_spec(
            generator.AdaptiveZoneSpecRequest(
                **{
                    **request.__dict__,
                    "pilot_scores": {"x999000_y6999000_s500": 1.0},
                }
            )
        )
    with pytest.raises(generator.ZoneSpecGenerationError, match="inside the zone"):
        generator.build_zone_spec(
            generator.AdaptiveZoneSpecRequest(
                **{
                    **request.__dict__,
                    "regression_tile_id": "x999000_y6999000_s500",
                }
            )
        )


def test_cli_has_no_batch_or_all_mode(
    d_root: Path, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(SystemExit) as caught:
        generator.main(
            [
                "--input",
                str(d_root / "input.json"),
                "--output",
                str(d_root / "zone-spec.json"),
                "--all",
            ]
        )
    assert caught.value.code == 2
    assert "unrecognized arguments: --all" in capsys.readouterr().err
