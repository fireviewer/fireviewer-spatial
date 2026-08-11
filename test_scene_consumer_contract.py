from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scene_consumer_contract import (
    SceneConsumerContractError,
    validate_scene_consumer_input,
)


jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "contracts" / "spatial" / "v1" / "scene-consumer-input.schema.json"
FIXTURE_PATH = (
    ROOT / "contracts" / "spatial" / "v1" / "fixtures" / "scene-consumer-input.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scene_consumer_input_accepts_bound_map_and_observed_timeline() -> None:
    schema = _load(SCHEMA_PATH)
    fixture = _load(FIXTURE_PATH)

    jsonschema.Draft202012Validator.check_schema(schema)
    assert validate_scene_consumer_input(fixture) == fixture


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("map", "entry_stage"), "map.usda"),
        (("perimeter_timeline", "between_observations"), "linear"),
        (("perimeter_timeline", "base_map_build_id"), "8" * 64),
        (("policy", "terrain_rebuild"), "allowed"),
        (("policy", "perimeter_rebuild"), "allowed"),
    ],
)
def test_scene_consumer_input_rejects_rebuild_or_unbound_timeline(
    path: tuple[str, str], value: object
) -> None:
    fixture = deepcopy(_load(FIXTURE_PATH))
    parent = fixture[path[0]]
    assert isinstance(parent, dict)
    parent[path[1]] = value

    with pytest.raises(SceneConsumerContractError):
        validate_scene_consumer_input(fixture)
