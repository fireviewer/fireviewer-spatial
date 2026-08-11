from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
OMNIVERSE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(OMNIVERSE_ROOT))

from scene_composition import (  # noqa: E402
    validate_asset_catalog,
    validate_composition_contract,
)

CATALOG_SCHEMA = ROOT / "asset-catalog-contract.schema.json"
COMPOSITION_SCHEMA = ROOT / "scene-composition-contract.schema.json"
CATALOG_EXAMPLE = ROOT / "examples" / "asset-catalog.pending.json"
COMPOSITION_EXAMPLE = ROOT / "examples" / "scene-composition.pending.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_errors(instance: object, schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [error.message for error in sorted(validator.iter_errors(instance), key=str)]


def validate_all() -> list[str]:
    catalog = load_json(CATALOG_EXAMPLE)
    composition = load_json(COMPOSITION_EXAMPLE)
    return (
        schema_errors(catalog, load_json(CATALOG_SCHEMA))
        + schema_errors(composition, load_json(COMPOSITION_SCHEMA))
        + validate_asset_catalog(catalog)
        + validate_composition_contract(composition)
    )


def main() -> int:
    errors = validate_all()
    if errors:
        for error in errors:
            print(error)
        return 1
    print("CONTRACT_VALIDATION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
