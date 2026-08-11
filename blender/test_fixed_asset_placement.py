from __future__ import annotations

import json

from jsonschema import Draft202012Validator
from pyproj import Transformer
import pytest

import fixed_asset_placement as fixed


def _library() -> dict:
    assets = [
        {
            "asset_id": "church_village_01",
            "category": "building",
            "reference": {"path": "buildings/church_village_01.png"},
        },
        {
            "asset_id": "hydrant_01",
            "category": "fire_equipment",
            "reference": {"path": "equipment/hydrant_01.png"},
        },
    ]
    return {"asset_count": len(assets), "assets": assets}


def _request(*placements: dict) -> dict:
    return {
        "schema": fixed.REQUEST_SCHEMA,
        "crs": fixed.SOURCE_CRS,
        "placements": list(placements),
    }


def _placement(**updates: object) -> dict:
    result = {
        "placement_id": "church-main",
        "asset_id": "church_village_01",
        "latitude": 43.90349754,
        "longitude": 4.49681631,
        "yaw_deg": 0,
    }
    result.update(updates)
    return result


def test_template_is_exactly_valid_against_the_public_schema() -> None:
    schema = json.loads(fixed.schema_path().read_text(encoding="utf-8"))
    template = json.loads(fixed.template_path().read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(template)
    assert fixed.normalize_request(template, _library()) == template


def test_request_is_sorted_and_rejects_unknown_or_missing_assets() -> None:
    normalized = fixed.normalize_request(
        _request(
            _placement(placement_id="z-last"),
            _placement(
                placement_id="a-first",
                asset_id="hydrant_01",
                yaw_deg=45,
            ),
        ),
        _library(),
    )
    assert [row["placement_id"] for row in normalized["placements"]] == [
        "a-first",
        "z-last",
    ]
    with pytest.raises(fixed.FixedAssetPlacementError, match="absent du catalogue"):
        fixed.normalize_request(_request(_placement(asset_id="missing")), _library())
    malformed = _placement()
    malformed["scale"] = 2
    with pytest.raises(fixed.FixedAssetPlacementError, match=r"inconnus=\['scale'\]"):
        fixed.normalize_request(_request(malformed), _library())


def test_manual_add_is_deterministic_and_idempotent() -> None:
    first = fixed.add_manual_placement(
        fixed.EMPTY_REQUEST,
        _library(),
        latitude=43.90349754,
        longitude=4.49681631,
        asset_id="church_village_01",
    )
    second = fixed.add_manual_placement(
        first,
        _library(),
        latitude=43.90349754,
        longitude=4.49681631,
        asset_id="church_village_01",
    )
    assert second == first
    assert first["placements"][0]["placement_id"].startswith("fixed-")
    assert fixed.rows_for_ui(first)[0][1:] == [
        "church_village_01",
        43.90349754,
        4.49681631,
        0.0,
    ]


def test_projection_assigns_one_owner_tile_and_rejects_outside_zone() -> None:
    convert = Transformer.from_crs(2154, 4326, always_xy=True)
    longitude, latitude = convert.transform(820_100, 6_312_600)
    request = _request(_placement(latitude=latitude, longitude=longitude, yaw_deg=90))
    projected = fixed.project_request(
        request,
        _library(),
        requested_bounds_l93_m=(819_500, 6_312_000, 821_000, 6_313_500),
    )
    assert len(projected) == 1
    assert projected[0]["position_l93_m"] == [820_100.0, 6_312_600.0]
    assert projected[0]["owner_tile_origin_l93_m"] == [820_000, 6_312_500]
    assert projected[0]["asset_category"] == "building"
    fixed.validate_projected_placements(
        projected, tile_origin_l93_m=(820_000, 6_312_500)
    )
    with pytest.raises(fixed.FixedAssetPlacementError, match="autre tuile"):
        fixed.validate_projected_placements(
            projected, tile_origin_l93_m=(820_500, 6_312_500)
        )
    with pytest.raises(fixed.FixedAssetPlacementError, match="hors du carré"):
        fixed.project_request(
            request,
            _library(),
            requested_bounds_l93_m=(819_500, 6_312_000, 820_000, 6_312_500),
        )


def test_file_import_is_bounded_and_exact(tmp_path) -> None:
    source = tmp_path / "placements.json"
    payload = _request(_placement())
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert fixed.load_request(source, _library()) == fixed.normalize_request(
        payload, _library()
    )
    source.write_bytes(b"{" + b" " * 1_048_576 + b"}")
    with pytest.raises(fixed.FixedAssetPlacementError, match="dépasse 1 Mio"):
        fixed.load_request(source, _library())


def test_asset_choices_keep_exact_ids_and_readable_labels() -> None:
    choices = fixed.asset_choices(_library())
    assert [value for _label, value in choices] == [
        "church_village_01",
        "hydrant_01",
    ]
    assert choices[0][0].startswith("building — church village 01")
