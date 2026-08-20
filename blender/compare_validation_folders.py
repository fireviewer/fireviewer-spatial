"""Compare two folder-native FireViewer validation evidence directories."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SUMMARY_NAME = "validation-summary.json"


class ValidationComparisonError(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationComparisonError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationComparisonError(f"{label} must be a JSON object")
    return value


def _folder_json(
    path: Path | str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(path).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValidationComparisonError(f"validation evidence is not a directory: {root}")
    if any(item.suffix.casefold() == ".zip" for item in root.rglob("*")):
        raise ValidationComparisonError("ZIP validation artifacts are forbidden")
    summary = _load_json(root / SUMMARY_NAME, f"validation summary {root.name}")
    if summary.get("artifact_role") != "validation_evidence_only_not_a_map":
        raise ValidationComparisonError("validation folder role is invalid")
    inventories: dict[str, dict[str, Any]] = {}
    tiles_root = root / "tiles"
    for tile_root in sorted(tiles_root.iterdir()) if tiles_root.is_dir() else ():
        if not tile_root.is_dir() or tile_root.is_symlink():
            continue
        inventory_path = tile_root / "placement-inventory.json"
        if not inventory_path.is_file():
            continue
        inventories[tile_root.name] = _load_json(
            inventory_path,
            f"placement inventory {tile_root.name}",
        )
    if len(inventories) != summary.get("tile_count"):
        raise ValidationComparisonError(
            f"inventory count differs from summary in {root.name}"
        )
    return summary, inventories


def _point(candidate: Mapping[str, Any], *, family: str) -> tuple[float, float] | None:
    key = "position_l93_m" if family == "trees" else "anchor_l93_m"
    value = candidate.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        return None
    return float(value[0]), float(value[1])


def _valid_candidates(
    inventory: Mapping[str, Any], family: str
) -> list[dict[str, Any]]:
    payload = inventory.get(family)
    candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(candidates, list):
        raise ValidationComparisonError(f"inventory lacks {family} candidates")
    return [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("status") == "valid"
    ]


def _tree_key(candidate: Mapping[str, Any]) -> str | None:
    value = candidate.get("candidate_id")
    return value if isinstance(value, str) and value else None


def _building_key(candidate: Mapping[str, Any]) -> str | None:
    for name in ("confirmed_source_id", "source_id"):
        value = candidate.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _indexed(
    candidates: Sequence[Mapping[str, Any]],
    *,
    family: str,
) -> dict[str, Mapping[str, Any]]:
    key_fn = _tree_key if family == "trees" else _building_key
    result: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        key = key_fn(candidate)
        if key is None or key in result:
            continue
        result[key] = candidate
    return result


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1),
    )
    return round(ordered[index], 4)


def _family_comparison(
    left_inventory: Mapping[str, Any],
    right_inventory: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    left_candidates = _valid_candidates(left_inventory, family)
    right_candidates = _valid_candidates(right_inventory, family)
    left_index = _indexed(left_candidates, family=family)
    right_index = _indexed(right_candidates, family=family)
    common = sorted(set(left_index) & set(right_index))
    distances: list[float] = []
    height_deltas: list[float] = []
    for key in common:
        left = left_index[key]
        right = right_index[key]
        left_point = _point(left, family=family)
        right_point = _point(right, family=family)
        if left_point is not None and right_point is not None:
            distances.append(math.dist(left_point, right_point))
        left_height = left.get("height_cm")
        right_height = right.get("height_cm")
        if (
            not isinstance(left_height, bool)
            and isinstance(left_height, (int, float))
            and not isinstance(right_height, bool)
            and isinstance(right_height, (int, float))
        ):
            height_deltas.append(
                abs(float(right_height) - float(left_height)) / 100.0
            )
    return {
        "left_valid_count": len(left_candidates),
        "right_valid_count": len(right_candidates),
        "matched_by_stable_identity": len(common),
        "only_left": len(set(left_index) - set(right_index)),
        "only_right": len(set(right_index) - set(left_index)),
        "xy_delta_m": {
            "p50": _percentile(distances, 0.50),
            "p95": _percentile(distances, 0.95),
            "maximum": round(max(distances), 4) if distances else None,
        },
        "absolute_height_delta_m": {
            "p50": _percentile(height_deltas, 0.50),
            "p95": _percentile(height_deltas, 0.95),
            "maximum": round(max(height_deltas), 4) if height_deltas else None,
        },
    }


def _source_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_sources = (
        left.get("sources") if isinstance(left.get("sources"), Mapping) else {}
    )
    right_sources = (
        right.get("sources") if isinstance(right.get("sources"), Mapping) else {}
    )
    keys = ("mnt_mm_sha256", "mns_mm_sha256", "context_sha256")
    return {
        key: {
            "left": left_sources.get(key),
            "right": right_sources.get(key),
            "equal": left_sources.get(key) == right_sources.get(key),
        }
        for key in keys
    }


def _viewer(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    receipt = summary.get("viewer_receipt")
    if not isinstance(receipt, Mapping):
        return None
    viewer = receipt.get("viewer")
    completeness = receipt.get("completeness")
    return {
        "representation": receipt.get("representation"),
        "byte_count": (
            viewer.get("byte_count") if isinstance(viewer, Mapping) else None
        ),
        "family_instance_counts": (
            completeness.get("family_instance_counts")
            if isinstance(completeness, Mapping)
            else None
        ),
        "mesh_coverage": (
            completeness.get("mesh_coverage")
            if isinstance(completeness, Mapping)
            else None
        ),
        "policy": (
            completeness.get("policy")
            if isinstance(completeness, Mapping)
            else None
        ),
    }


def compare_validation_folders(
    left_path: Path | str,
    right_path: Path | str,
) -> dict[str, Any]:
    left_summary, left_inventories = _folder_json(left_path)
    right_summary, right_inventories = _folder_json(right_path)
    left_tiles = {
        str(item.get("tile_id")): item.get("origin_l93_m")
        for item in left_summary.get("tiles", [])
        if isinstance(item, Mapping)
    }
    right_tiles = {
        str(item.get("tile_id")): item.get("origin_l93_m")
        for item in right_summary.get("tiles", [])
        if isinstance(item, Mapping)
    }
    if left_tiles != right_tiles or set(left_inventories) != set(right_inventories):
        raise ValidationComparisonError(
            "validation folders do not cover the same tile identities/origins"
        )

    tile_results: list[dict[str, Any]] = []
    tree_distances: list[float] = []
    building_distances: list[float] = []
    source_all_equal = True
    for tile_id in sorted(left_inventories):
        left_inventory = left_inventories[tile_id]
        right_inventory = right_inventories[tile_id]
        sources = _source_match(left_inventory, right_inventory)
        source_all_equal &= all(record["equal"] for record in sources.values())
        trees = _family_comparison(
            left_inventory,
            right_inventory,
            family="trees",
        )
        buildings = _family_comparison(
            left_inventory,
            right_inventory,
            family="buildings",
        )
        if trees["xy_delta_m"]["p50"] is not None:
            tree_distances.append(float(trees["xy_delta_m"]["p50"]))
        if buildings["xy_delta_m"]["p50"] is not None:
            building_distances.append(float(buildings["xy_delta_m"]["p50"]))
        tile_results.append(
            {
                "tile_id": tile_id,
                "origin_l93_m": left_tiles[tile_id],
                "sources": sources,
                "trees": trees,
                "buildings": buildings,
                "left_context_assets_valid": len(
                    _valid_candidates(left_inventory, "context_assets")
                ),
                "right_context_assets_valid": len(
                    _valid_candidates(right_inventory, "context_assets")
                ),
            }
        )

    return {
        "schema": "fireviewer.map-validation-comparison.v2",
        "left": {
            "provider": left_summary.get("provider"),
            "zone_id": left_summary.get("zone_id"),
            "build_id": left_summary.get("build_id"),
            "source_revisions": left_summary.get("source_revisions"),
            "zone_counts": left_summary.get("zone_counts"),
            "viewer": _viewer(left_summary),
        },
        "right": {
            "provider": right_summary.get("provider"),
            "zone_id": right_summary.get("zone_id"),
            "build_id": right_summary.get("build_id"),
            "source_revisions": right_summary.get("source_revisions"),
            "zone_counts": right_summary.get("zone_counts"),
            "viewer": _viewer(right_summary),
        },
        "same_tile_set": True,
        "all_canonical_1m_source_hashes_equal": source_all_equal,
        "source_revision_labels_equal": (
            left_summary.get("source_revisions")
            == right_summary.get("source_revisions")
        ),
        "tile_count": len(tile_results),
        "tiles": tile_results,
        "summary": {
            "median_tile_tree_xy_p50_m": (
                round(statistics.median(tree_distances), 4)
                if tree_distances
                else None
            ),
            "median_tile_building_xy_p50_m": (
                round(statistics.median(building_distances), 4)
                if building_distances
                else None
            ),
        },
    }


def _markdown(result: Mapping[str, Any]) -> str:
    left = result["left"]
    right = result["right"]
    lines = [
        "# FireViewer 9-tile validation comparison",
        "",
        f"- Left: `{left.get('provider')}` / `{left.get('build_id')}`",
        f"- Right: `{right.get('provider')}` / `{right.get('build_id')}`",
        f"- Same 9-tile set: `{result.get('same_tile_set')}`",
        "- Canonical 1 m source hashes equal: "
        f"`{result.get('all_canonical_1m_source_hashes_equal')}`",
        "- Viewer role: complete non-simplified map representation",
        "",
        "| Tile | Trees L | Trees R | Buildings L | Buildings R | Tree XY p50 m | Building XY p50 m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tile in result.get("tiles", []):
        trees = tile["trees"]
        buildings = tile["buildings"]
        lines.append(
            f"| {tile['tile_id']} | {trees['left_valid_count']} | "
            f"{trees['right_valid_count']} | {buildings['left_valid_count']} | "
            f"{buildings['right_valid_count']} | {trees['xy_delta_m']['p50']} | "
            f"{buildings['xy_delta_m']['p50']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    options = parser.parse_args(argv)
    result = compare_validation_folders(options.left, options.right)
    json_text = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    markdown_text = _markdown(result)
    if options.json_output is not None:
        options.json_output.parent.mkdir(parents=True, exist_ok=True)
        options.json_output.write_text(json_text, encoding="utf-8")
    if options.markdown_output is not None:
        options.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        options.markdown_output.write_text(markdown_text, encoding="utf-8")
    print(json_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ValidationComparisonError",
    "compare_validation_folders",
]
