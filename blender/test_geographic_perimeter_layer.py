from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

import geographic_perimeter_layer as layer


def _polygon(
    west: float = 4.49,
    south: float = 43.90,
    east: float = 4.50,
    north: float = 43.91,
) -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


def _dataset() -> dict[str, object]:
    return {
        "id": "fire-real-2026",
        "name": "Incendie réel",
        "description": "Observations quotidiennes validées",
        "timeline": [
            {
                "id": "day-2",
                "timestamp": "2026-08-11T12:00:00+02:00",
                "affected": _polygon(4.488, 43.898, 4.502, 43.912),
            },
            {
                "id": "day-1",
                "timestamp": "2026-08-10T10:00:00Z",
                "category1": {"polygons": _polygon()["coordinates"]},
                "category2": {
                    "polygons": _polygon(4.495, 43.905, 4.498, 43.908)["coordinates"]
                },
            },
        ],
    }


def test_compiles_fixed_layers_and_observed_simulation_timeline() -> None:
    compiled = layer.compile_perimeter_layer(_dataset())

    assert compiled.dataset_id == "fire-real-2026"
    frames = compiled.timeline["frames"]
    assert [frame["frame_id"] for frame in frames] == ["day-1", "day-2"]
    assert [frame["elapsed_seconds"] for frame in frames] == [0, 86_400]
    assert compiled.timeline["between_observations"] == "undefined"
    assert compiled.timeline["prediction"] == "none"
    assert compiled.timeline["simulation_driver"] == (
        "explicit_observed_ranges_and_timestamps"
    )
    assert frames[0]["affected"]["present"] is True
    assert frames[0]["active"]["present"] is True
    assert frames[1]["active"]["present"] is False
    assert frames[1]["active"]["prim_path"] is None

    stage = compiled.stage_text
    assert 'def Scope "AffectedFixedLayers"' in stage
    assert 'def Scope "ActiveFixedLayers"' in stage
    assert 'def Xform "Frame_0000"' in stage
    assert 'def Xform "Frame_0001"' in stage
    assert "timeSamples" not in stage
    assert 'custom string fireviewer:between_observations = "undefined"' in stage
    assert "fireviewer:valid_from" in stage
    assert "fireviewer:valid_to" in stage
    assert "EPSG:2154" in stage


def test_explicit_time_window_is_preserved_without_inference() -> None:
    payload = {
        "id": "ranged-fire",
        "timeline": [
            {
                "id": "morning-range",
                "time_window": {
                    "start": "2026-08-10T06:00:00Z",
                    "end": "2026-08-10T12:00:00Z",
                },
                "affected": _polygon(),
            }
        ],
    }

    compiled = layer.compile_perimeter_layer(payload)
    normalized = compiled.normalized_source["frames"][0]
    frame = compiled.timeline["frames"][0]
    assert normalized["observed_at"] == "2026-08-10T12:00:00Z"
    assert normalized["time_range"] == {
        "start": "2026-08-10T06:00:00Z",
        "end": "2026-08-10T12:00:00Z",
        "kind": "explicit_interval",
    }
    assert compiled.timeline["time_origin"] == "2026-08-10T06:00:00Z"
    assert frame["elapsed_seconds"] == 21_600
    assert frame["time_range"]["elapsed_start_seconds"] == 0
    assert frame["time_range"]["elapsed_end_seconds"] == 21_600
    assert 'fireviewer:valid_from = "2026-08-10T06:00:00Z"' in compiled.stage_text
    assert 'fireviewer:valid_to = "2026-08-10T12:00:00Z"' in compiled.stage_text


def test_invalid_explicit_time_window_fails_closed() -> None:
    payload = {
        "timeline": [
            {
                "time_window": {
                    "start": "2026-08-10T12:00:00Z",
                    "end": "2026-08-10T06:00:00Z",
                },
                "affected": _polygon(),
            }
        ]
    }
    with pytest.raises(layer.GeographicPerimeterError, match="fin de plage"):
        layer.compile_perimeter_layer(payload)


def test_semantically_identical_input_order_is_bit_identical() -> None:
    first = layer.compile_perimeter_layer(_dataset())
    reordered = _dataset()
    reordered["timeline"] = list(reversed(reordered["timeline"]))
    second = layer.compile_perimeter_layer(reordered)

    assert first.build_id == second.build_id
    assert first.normalized_source == second.normalized_source
    assert first.timeline == second.timeline
    assert first.stage_text == second.stage_text


def test_geojson_groups_affected_and_active_features_by_timestamp() -> None:
    payload = {
        "type": "FeatureCollection",
        "id": "geojson-fire",
        "name": "GeoJSON réel",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "timestamp": "2026-08-10T00:00:00Z",
                    "category": "cumulative_affected",
                },
                "geometry": _polygon(),
            },
            {
                "type": "Feature",
                "properties": {
                    "timestamp": "2026-08-10T00:00:00Z",
                    "category": "active_front",
                },
                "geometry": _polygon(4.495, 43.905, 4.498, 43.908),
            },
        ],
    }

    compiled = layer.compile_perimeter_layer(payload)
    assert len(compiled.timeline["frames"]) == 1
    assert compiled.timeline["frames"][0]["affected"]["present"] is True
    assert compiled.timeline["frames"][0]["active"]["present"] is True


def test_package_is_immutable_replayable_and_tamper_evident(tmp_path: Path) -> None:
    compiled = layer.compile_perimeter_layer(_dataset())
    root = tmp_path / "package"
    first = layer.write_perimeter_layer_package(compiled, root)
    second = layer.write_perimeter_layer_package(compiled, root)

    assert first == second
    assert layer.validate_perimeter_layer_package(root) == first
    assert {path.name for path in root.iterdir()} == {
        layer.STAGE_NAME,
        layer.TIMELINE_NAME,
        layer.SOURCE_NAME,
        layer.MANIFEST_NAME,
    }

    stage = root / layer.STAGE_NAME
    stage.write_text(stage.read_text(encoding="utf-8") + "# tamper\n", encoding="utf-8")
    with pytest.raises(layer.GeographicPerimeterError, match="altérée"):
        layer.validate_perimeter_layer_package(root)


def test_uploaded_file_produces_deterministic_download_archive(tmp_path: Path) -> None:
    source = tmp_path / "observations.geojson"
    source.write_text(json.dumps(_dataset()), encoding="utf-8")

    first = layer.produce_perimeter_layer(source, tmp_path / "work")
    first_bytes = first.archive.read_bytes()
    second = layer.produce_perimeter_layer(source, tmp_path / "work")

    assert first.package_root == second.package_root
    assert first.archive == second.archive
    assert second.archive.read_bytes() == first_bytes
    with zipfile.ZipFile(first.archive) as archive:
        assert set(archive.namelist()) == {
            layer.STAGE_NAME,
            layer.TIMELINE_NAME,
            layer.SOURCE_NAME,
            layer.MANIFEST_NAME,
        }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["timeline"].append(dict(value["timeline"][0])),
            "même timestamp",
        ),
        (
            lambda value: value["timeline"][0].update(
                {"timestamp": "2026-08-11T12:00:00"}
            ),
            "fuseau horaire",
        ),
        (
            lambda value: value["timeline"][0].update(
                {"affected": _polygon(-120.0, 40.0, -119.9, 40.1)}
            ),
            "hors couverture Lambert-93",
        ),
    ],
)
def test_invalid_chronology_and_geography_fail_closed(
    mutation: object, message: str
) -> None:
    payload = _dataset()
    mutation(payload)
    with pytest.raises(layer.GeographicPerimeterError, match=message):
        layer.compile_perimeter_layer(payload)
