from __future__ import annotations

from pathlib import Path

import pyogrio
import pytest
from shapely import Point, to_wkb

from ground_context_binding import load_context_contract
from prepare_ground_context import (
    MANIFEST_SCHEMA,
    _safe_cleanup,
    build_wfs_url,
    plan_case,
    sha256_file,
    validate_package,
    write_layer,
)
from prepare_incident_terrains import CASES


ROOT = Path(__file__).parent


def test_plan_is_complete_and_does_not_write(tmp_path: Path) -> None:
    contract = load_context_contract(ROOT / "ground_context_contract.v1.json")
    case = next(case for case in CASES if case.fire_id == "FR-66-00001")
    plan = plan_case(case, contract)
    assert set(plan["output_layers"]) == {
        "land_parcels",
        "agricultural_parcels",
        "roads",
        "railways",
        "hydro_lines",
        "hydro_surfaces",
        "landcover",
        "geology",
    }
    assert plan["archives"]["ocsge"].endswith(
        "OCS-GE_2-0__GPKG_LAMB93_D066_2024-01-01.7z"
    )
    assert not list(tmp_path.iterdir())


def test_wfs_url_is_lambert93_paged_and_stably_sorted() -> None:
    contract = load_context_contract(ROOT / "ground_context_contract.v1.json")
    roads = next(source for source in contract["sources"] if source["id"] == "roads")
    url = build_wfs_url(roads, (1.0, 2.0, 3.0, 4.0), 20_000)
    assert "SRSNAME=EPSG%3A2154" in url
    assert "STARTINDEX=20000" in url
    assert "SORTBY=cleabs" in url
    assert "BBOX=1.0%2C2.0%2C3.0%2C4.0%2CEPSG%3A2154" in url


def test_compact_package_layer_and_hash_validation(tmp_path: Path) -> None:
    package = tmp_path / "context.gpkg"
    receipt = write_layer(
        package,
        layer_id="roads",
        fields=["cleabs", "largeur_de_chaussee", "nature"],
        geometries=[to_wkb(Point(700_000, 6_600_000))],
        records=[
            {
                "cleabs": "road-1",
                "largeur_de_chaussee": 5.5,
                "nature": "Route empierrée",
            }
        ],
    )
    assert receipt["feature_count"] == 1
    assert receipt["semantic_tag_counts"]["road:unpaved"] == 1
    assert pyogrio.read_info(package, layer="roads")["features"] == 1
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "package": {"sha256": sha256_file(package)},
        "layers": {"roads": receipt},
    }
    validate_package(package, manifest, {"roads"})
    package.write_bytes(package.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_package(package, manifest, {"roads"})


def test_cleanup_cannot_escape_declared_source_cache(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="outside source cache"):
        _safe_cleanup(root, [outside])
    assert outside.read_bytes() == b"preserve"
