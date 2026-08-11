"""Deterministic planning contract for the rebuilt 295-asset USD catalog.

This module plans composition only. It never authors or opens a scene. The
planner remains fail-closed while the replacement USD catalog is incomplete,
and it never substitutes primitives, placeholders or simplified geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ALGORITHM_ID = "fireviewer.omniverse.asset4sim-composition.v2"
EXPECTED_ASSET_COUNT = 295
BASE_SCENE_COUNT = 4
VARIANTS_PER_BASE = 5
PORTFOLIO_SCENE_COUNT = BASE_SCENE_COUNT * VARIANTS_PER_BASE
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


class CompositionContractError(ValueError):
    """Raised when composition would weaken or diverge from the contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _portable_usd_path(value: object) -> str | None:
    if not isinstance(value, str) or "\\" in value or "://" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in USD_SUFFIXES:
        return None
    return path.as_posix()


def _asset_errors(asset: object, *, index: int) -> list[str]:
    label = f"assets[{index}]"
    if not isinstance(asset, dict):
        return [f"{label} must be an object"]
    errors: list[str] = []
    stable_id = asset.get("stable_id")
    if not isinstance(stable_id, str) or ID_RE.fullmatch(stable_id) is None:
        errors.append(f"{label}.stable_id is invalid")
    if _portable_usd_path(asset.get("usd_path")) is None:
        errors.append(f"{label}.usd_path must be a portable USD path")
    if not isinstance(asset.get("sha256"), str) or SHA256_RE.fullmatch(asset["sha256"]) is None:
        errors.append(f"{label}.sha256 is invalid")
    if asset.get("status") != "accepted":
        errors.append(f"{label}.status must be accepted")
    if asset.get("kit_open") != "passed":
        errors.append(f"{label}.kit_open must be passed")
    roles = asset.get("roles")
    if not isinstance(roles, list) or not roles or any(not isinstance(item, str) or not item for item in roles):
        errors.append(f"{label}.roles must be a non-empty string array")
    contexts = asset.get("contexts")
    if not isinstance(contexts, list) or not contexts or any(not isinstance(item, str) or not item for item in contexts):
        errors.append(f"{label}.contexts must be a non-empty string array")
    if asset.get("placeholder") is not False:
        errors.append(f"{label}.placeholder must be false")
    return errors


def validate_asset_catalog(
    catalog: object,
    *,
    catalog_root: Path | None = None,
) -> list[str]:
    if not isinstance(catalog, dict):
        return ["asset catalog must be an object"]
    errors: list[str] = []
    if catalog.get("schema") != "fireviewer.omniverse-asset-catalog.v2":
        errors.append("unsupported asset catalog schema")
    if catalog.get("expected_asset_count") != EXPECTED_ASSET_COUNT:
        errors.append("asset catalog must expect exactly 295 assets")
    status = catalog.get("status")
    if status not in {"blocked_pending_usd_assets", "ready"}:
        errors.append("asset catalog status is invalid")
    assets = catalog.get("assets")
    if not isinstance(assets, list):
        return errors + ["asset catalog assets must be an array"]
    received = catalog.get("received_asset_count")
    if received != len(assets):
        errors.append("received_asset_count must equal the asset array length")
    if len(assets) > EXPECTED_ASSET_COUNT:
        errors.append("asset catalog exceeds the definitive 295-asset boundary")
    if status == "ready" and len(assets) != EXPECTED_ASSET_COUNT:
        errors.append("a ready asset catalog must contain exactly 295 assets")
    if status == "blocked_pending_usd_assets" and len(assets) == EXPECTED_ASSET_COUNT:
        errors.append("a complete catalog must not remain marked pending")
    for index, asset in enumerate(assets):
        errors.extend(_asset_errors(asset, index=index))
    ids = [asset.get("stable_id") for asset in assets if isinstance(asset, dict)]
    paths = [asset.get("usd_path") for asset in assets if isinstance(asset, dict)]
    if len(set(ids)) != len(ids):
        errors.append("asset stable IDs must be unique")
    if len(set(paths)) != len(paths):
        errors.append("asset USD paths must be unique")
    if ids != sorted(ids):
        errors.append("assets must be sorted by stable_id")
    if catalog_root is not None:
        root = catalog_root.resolve()
        for index, asset in enumerate(assets):
            if not isinstance(asset, dict):
                continue
            portable = _portable_usd_path(asset.get("usd_path"))
            if portable is None:
                continue
            path = (root / portable).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"assets[{index}].usd_path escapes the catalog root")
                continue
            if not path.is_file():
                errors.append(f"assets[{index}].usd_path is absent")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != asset.get("sha256"):
                errors.append(f"assets[{index}].sha256 does not match the USD file")
    return errors


def validate_composition_contract(contract: object) -> list[str]:
    if not isinstance(contract, dict):
        return ["composition contract must be an object"]
    errors: list[str] = []
    exact = {
        "schema": "fireviewer.omniverse-scene-composition.v2",
        "algorithm": ALGORITHM_ID,
        "expected_asset_count": EXPECTED_ASSET_COUNT,
        "base_scene_count": BASE_SCENE_COUNT,
        "variants_per_base": VARIANTS_PER_BASE,
        "scene_count": PORTFOLIO_SCENE_COUNT,
        "placeholder_policy": "forbidden",
        "simplified_geometry_fallback": "forbidden",
        "asset_coverage": "all_assets_at_least_once_across_portfolio",
        "spatial_anchor_policy": "preserve_authoritative_positions_and_stable_ids",
    }
    for field, expected in exact.items():
        if contract.get(field) != expected:
            errors.append(f"{field} must equal {expected!r}")
    if contract.get("status") not in {"blocked_pending_usd_assets", "ready"}:
        errors.append("composition contract status is invalid")
    if not isinstance(contract.get("master_seed"), int) or contract["master_seed"] < 0:
        errors.append("master_seed must be a non-negative integer")
    base_scenes = contract.get("base_scenes")
    if not isinstance(base_scenes, list) or len(base_scenes) != BASE_SCENE_COUNT:
        errors.append("base_scenes must contain exactly four records")
        return errors
    ids: list[object] = []
    for index, scene in enumerate(base_scenes):
        if not isinstance(scene, dict):
            errors.append(f"base_scenes[{index}] must be an object")
            continue
        scene_id = scene.get("id")
        ids.append(scene_id)
        if not isinstance(scene_id, str) or ID_RE.fullmatch(scene_id) is None:
            errors.append(f"base_scenes[{index}].id is invalid")
        contexts = scene.get("context_tags")
        if not isinstance(contexts, list) or not contexts or any(not isinstance(item, str) or not item for item in contexts):
            errors.append(f"base_scenes[{index}].context_tags must be non-empty")
    if len(set(ids)) != len(ids):
        errors.append("base scene IDs must be unique")
    if ids != sorted(ids):
        errors.append("base scenes must be sorted by id")
    catalog = contract.get("asset_catalog")
    if not isinstance(catalog, dict) or catalog.get("expected_asset_count") != EXPECTED_ASSET_COUNT:
        errors.append("asset_catalog must bind the definitive 295-asset catalog")
    elif contract.get("status") == "ready" and (
        not isinstance(catalog.get("sha256"), str)
        or SHA256_RE.fullmatch(catalog["sha256"]) is None
    ):
        errors.append("a ready contract must hash-lock its asset catalog")
    elif contract.get("status") == "blocked_pending_usd_assets" and catalog.get("sha256") is not None:
        errors.append("a pending contract must not claim a catalog SHA-256")
    return errors


def _scene_slots(contract: dict[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    sequence = 0
    for base in contract["base_scenes"]:
        for variant_index in range(1, VARIANTS_PER_BASE + 1):
            sequence += 1
            slots.append(
                {
                    "scene_id": f"SIM-{sequence:02d}",
                    "base_scene_id": base["id"],
                    "variant_index": variant_index,
                    "context_tags": list(base["context_tags"]),
                    "seed": int(
                        _digest(
                            f"{contract['master_seed']}\0{base['id']}\0{variant_index}"
                        )[:16],
                        16,
                    ),
                }
            )
    return slots


def _compatible(asset: dict[str, Any], slot: dict[str, Any]) -> bool:
    contexts = set(asset["contexts"])
    return "universal" in contexts or bool(contexts.intersection(slot["context_tags"]))


def build_portfolio_plan(
    catalog: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    catalog_errors = validate_asset_catalog(catalog)
    contract_errors = validate_composition_contract(contract)
    errors = catalog_errors + contract_errors
    if errors:
        raise CompositionContractError("; ".join(errors))
    if catalog["status"] != "ready" or contract["status"] != "ready":
        raise CompositionContractError(
            "composition is blocked until all 295 rebuilt USD assets are accepted"
        )
    catalog_sha = hashlib.sha256(canonical_json(catalog).encode("utf-8")).hexdigest()
    if contract["asset_catalog"]["sha256"] != catalog_sha:
        raise CompositionContractError("composition contract does not match the asset catalog")

    slots = _scene_slots(contract)
    required: dict[str, list[str]] = {slot["scene_id"]: [] for slot in slots}
    for asset in catalog["assets"]:
        candidates = [slot for slot in slots if _compatible(asset, slot)]
        if not candidates:
            raise CompositionContractError(
                f"{asset['stable_id']} is incompatible with every base scene"
            )
        selected = min(
            candidates,
            key=lambda slot: _digest(
                f"{contract['master_seed']}\0{asset['stable_id']}\0{slot['scene_id']}"
            ),
        )
        required[selected["scene_id"]].append(asset["stable_id"])

    scenes: list[dict[str, Any]] = []
    for slot in slots:
        available = [
            asset["stable_id"]
            for asset in catalog["assets"]
            if _compatible(asset, slot)
        ]
        available.sort(
            key=lambda stable_id: _digest(
                f"{slot['seed']}\0{stable_id}"
            )
        )
        scenes.append(
            {
                **slot,
                "required_asset_ids": sorted(required[slot["scene_id"]]),
                "available_asset_ids": available,
                "placement_status": "blocked_pending_scene_authoring",
            }
        )

    covered = sorted(
        stable_id
        for scene in scenes
        for stable_id in scene["required_asset_ids"]
    )
    expected = [asset["stable_id"] for asset in catalog["assets"]]
    if covered != expected:
        raise AssertionError("internal error: portfolio coverage is incomplete")
    plan: dict[str, Any] = {
        "schema": "fireviewer.omniverse-scene-plan.v2",
        "algorithm": ALGORITHM_ID,
        "asset_catalog_sha256": catalog_sha,
        "master_seed": contract["master_seed"],
        "scene_count": len(scenes),
        "asset_count": len(expected),
        "asset_coverage": "all_assets_exactly_once_as_required_minimum",
        "placeholder_substitution": False,
        "simplified_geometry_fallback": False,
        "scenes": scenes,
    }
    plan["plan_sha256"] = hashlib.sha256(
        canonical_json(plan).encode("utf-8")
    ).hexdigest()
    return plan


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CompositionContractError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8", newline="\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--catalog", type=Path, required=True)
    validate.add_argument("--contract", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--catalog", type=Path, required=True)
    plan.add_argument("--contract", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    catalog = _load(args.catalog)
    contract = _load(args.contract)
    errors = validate_asset_catalog(catalog) + validate_composition_contract(contract)
    if errors:
        raise CompositionContractError("; ".join(errors))
    if args.command == "validate":
        print("COMPOSITION_CONTRACT_VALID")
        return 0
    _write(args.output, build_portfolio_plan(catalog, contract))
    print("COMPOSITION_PLAN_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
