"""Validate immutable map/timeline references used by downstream consumers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "contracts" / "spatial" / "v1" / "scene-consumer-input.schema.json"


class SceneConsumerContractError(ValueError):
    """Raised when a consumer input is structurally or semantically invalid."""


def validate_scene_consumer_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    document = dict(payload)
    try:
        Draft202012Validator(schema).validate(document)
    except Exception as exc:
        raise SceneConsumerContractError(
            f"scene consumer input is invalid: {exc}"
        ) from exc

    map_reference = document["map"]
    timeline = document.get("perimeter_timeline")
    if (
        isinstance(timeline, Mapping)
        and timeline["base_map_build_id"] != map_reference["map_build_id"]
    ):
        raise SceneConsumerContractError(
            "perimeter timeline does not target the selected map build"
        )
    return document


__all__ = ["SceneConsumerContractError", "validate_scene_consumer_input"]
