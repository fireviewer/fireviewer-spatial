from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import prepare_simple_measured_zone_context as context


BOUNDS = (819490, 6311990, 821010, 6313510)


def _feature(layer: str) -> dict[str, object]:
    if (
        layer.endswith("batiment")
        or layer.endswith("zone_de_vegetation")
        or layer.endswith("resu_bdv1_shape")
    ):
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [819600, 6312200],
                    [819620, 6312200],
                    [819620, 6312220],
                    [819600, 6312200],
                ]
            ],
        }
    else:
        geometry = {
            "type": "LineString",
            "coordinates": [[819600, 6312200], [819700, 6312300]],
        }
    source_id = f"{layer.split(':')[-1]}-001"
    return {
        "type": "Feature",
        "id": source_id,
        "properties": {
            "cleabs": source_id,
            "nature": "Route empierrée",
            "libelle2": "FUTAIE DE PIN NOIR",
            "importance": "5",
            "largeur_de_chaussee": None,
        },
        "geometry": geometry,
    }


def _get(url: str) -> bytes:
    values = parse_qs(urlparse(url).query)
    layer = values["TYPENAMES"][0]
    start = int(values["STARTINDEX"][0])
    features = [_feature(layer)] if start == 0 else []
    return json.dumps(
        {
            "type": "FeatureCollection",
            "numberMatched": 1,
            "numberReturned": len(features),
            "features": features,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_wfs_request_is_lambert93_and_paged() -> None:
    url = context.wfs_url(context.LAYERS["buildings"], BOUNDS, start_index=5, count=25)
    values = parse_qs(urlparse(url).query)
    assert values == {
        "SERVICE": ["WFS"],
        "VERSION": ["2.0.0"],
        "REQUEST": ["GetFeature"],
        "TYPENAMES": ["BDTOPO_V3:batiment"],
        "SRSNAME": ["EPSG:2154"],
        "BBOX": ["819490,6311990,821010,6313510,EPSG:2154"],
        "OUTPUTFORMAT": ["application/json"],
        "SORTBY": ["cleabs"],
        "STARTINDEX": ["5"],
        "COUNT": ["25"],
    }

    forest_url = context.wfs_url(
        context.LAYERS["forest_composition"], BOUNDS, start_index=0
    )
    forest_values = parse_qs(urlparse(forest_url).query)
    assert forest_values["SORTBY"] == ["dep,tfifn,typn"]


def test_zone_context_is_atomic_hash_locked_and_reusable(tmp_path: Path) -> None:
    output = tmp_path / "zone-context.json"
    first = context.prepare_zone_context(
        output_path=output,
        zone_id="gps-test-zone",
        bounds_l93_m=BOUNDS,
        source_revision="bdtopo-v3-test",
        http_get=_get,
    )
    assert first.reused is False
    assert first.feature_counts == {role: 1 for role in context.LAYERS}
    assert not output.with_name(f".{output.name}.part").exists()
    loaded = context.load_zone_context(output)
    assert loaded["content_sha256"] == first.content_sha256

    reused = context.prepare_zone_context(
        output_path=output,
        zone_id="gps-test-zone",
        bounds_l93_m=BOUNDS,
        source_revision="bdtopo-v3-test",
        http_get=lambda _url: pytest.fail("a valid zone context must be reused"),
    )
    assert reused.reused is True


def test_zone_context_tamper_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "zone-context.json"
    context.prepare_zone_context(
        output_path=output,
        zone_id="gps-test-zone",
        bounds_l93_m=BOUNDS,
        source_revision="bdtopo-v3-test",
        http_get=_get,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["layers"]["buildings"]["features"][0]["id"] = "changed"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(context.SimpleMeasuredZoneContextError, match="canonical|hash"):
        context.load_zone_context(output)
