"""Validate a complete FireViewer remote tile catalog and its deploy payload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Mapping, Sequence

from fwtile import FWTileError, read_container, sha256_file


CATALOG_SCHEMA = "fireviewer.remote-tile-catalog.v1"
RECEIPT_SCHEMA = "fireviewer.remote-tile-receipt.v1"
REQUIRED_DETAIL_SECTIONS = ("terrain", "trees", "buildings", "roads", "water")
SUPPORTED_TREE_ENCODINGS = {
    "tree-instance-mm.v1",
    "tree-instance-position-mm-dimension-cm.v2",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FWTileError(f"JSON root is not an object: {path}")
    return value


def _asset_path(output_root: Path, asset: Mapping[str, Any]) -> Path:
    relative = str(asset.get("url", ""))
    parsed = PurePosixPath(relative)
    if not relative or parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
        raise FWTileError(f"unsafe catalog URL: {relative!r}")
    path = output_root.joinpath(*parsed.parts)
    if not path.is_file():
        raise FWTileError(f"catalog asset is missing: {path}")
    byte_count = int(asset.get("byte_count", -1))
    if path.stat().st_size != byte_count:
        raise FWTileError(f"catalog byte count mismatch: {path}")
    if sha256_file(path) != asset.get("sha256"):
        raise FWTileError(f"catalog SHA-256 mismatch: {path}")
    return path


def _percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"minimum": 0, "median": 0, "p95": 0, "maximum": 0, "total": 0}
    return {
        "minimum": min(values),
        "median": int(median(values)),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
        "total": sum(values),
    }


def validate(artifact_root: Path, output_root: Path) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    output_root = output_root.resolve()
    manifest_path = artifact_root / "global-05m/production-manifest.json"
    catalog_path = output_root / "catalog.json"
    manifest = _load_json(manifest_path)
    catalog = _load_json(catalog_path)
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise FWTileError("remote catalog schema is unsupported")
    detail_policy = catalog.get("lod_policy", {}).get("detail", {})
    detail_enabled = detail_policy.get("enabled")
    if not isinstance(detail_enabled, bool):
        raise FWTileError("catalog does not declare detail streaming")
    source_ready = all(
        isinstance(source, dict) and source.get("status", {}).get("state") == "ready"
        for source in manifest.get("source_tiles", [])
    )
    if manifest.get("status") != "ready" and not (not detail_enabled and source_ready):
        raise FWTileError("production manifest is not ready")

    ready_ids = {
        str(tile["id"])
        for tile in manifest.get("tiles", [])
        if tile.get("status", {}).get("state") == "ready"
    }
    tiles = catalog.get("tiles", [])
    catalog_ids = [str(tile.get("id")) for tile in tiles]
    if len(catalog_ids) != len(set(catalog_ids)):
        raise FWTileError("catalog contains duplicate detail tile ids")
    if detail_enabled and set(catalog_ids) != ready_ids:
        missing = sorted(ready_ids - set(catalog_ids))
        unexpected = sorted(set(catalog_ids) - ready_ids)
        raise FWTileError(
            f"catalog/manifest mismatch: missing={missing}, unexpected={unexpected}"
        )
    if int(catalog.get("exported_detail_tile_count", -1)) != len(catalog_ids):
        raise FWTileError("catalog exported tile count is inconsistent")
    maximum_resident = int(detail_policy.get("maximum_resident_tile_count", -1))
    if detail_enabled and maximum_resident != 16:
        raise FWTileError("catalog does not enforce the global 16-tile budget")
    if float(detail_policy.get("publish_distance_m", -1)) != 600.0:
        raise FWTileError("catalog does not preserve the 600 m publish distance")
    if float(detail_policy.get("preload_radius_m", -1)) != 750.0:
        raise FWTileError("catalog does not preserve the 750 m preload radius")
    near_disabled = detail_policy.get("near_disabled")
    if not isinstance(near_disabled, bool):
        raise FWTileError("catalog does not declare the near LOD policy")
    if not detail_enabled and (
        catalog_ids
        or int(catalog.get("exported_detail_tile_count", -1)) != 0
        or maximum_resident != 0
        or detail_policy.get("mode") != "global_only"
        or near_disabled is not True
    ):
        raise FWTileError("global-only catalog contains a detail-streaming contract")

    receipt_paths = sorted((output_root / "receipts").glob("x*_s*.json"))
    receipt_ids: set[str] = set()
    for receipt_path in receipt_paths:
        receipt = _load_json(receipt_path)
        if receipt.get("schema") != RECEIPT_SCHEMA:
            raise FWTileError(f"invalid detail receipt schema: {receipt_path}")
        receipt_ids.add(str(receipt.get("tile_id")))
    if detail_enabled and receipt_ids != ready_ids:
        raise FWTileError("detail receipts do not exactly cover every ready tile")

    payload_bytes: list[int] = []
    imagery_bytes: list[int] = []
    combined_bytes: list[int] = []
    referenced_paths: set[Path] = {catalog_path}
    tree_encodings: dict[str, int] = {}
    tree_instances = 0
    for tile in tiles:
        payload_asset = tile.get("payload", {})
        imagery_asset = tile.get("imagery", {})
        imagery_resolution = float(imagery_asset.get("resolution_m", 0))
        if near_disabled and imagery_resolution < 0.5:
            raise FWTileError(
                f"near imagery remains published while near LOD is disabled: {tile.get('id')}"
            )
        payload_path = _asset_path(output_root, payload_asset)
        imagery_path = _asset_path(output_root, imagery_asset)
        referenced_paths.update((payload_path, imagery_path))
        payload = payload_path.read_bytes()
        parsed = read_container(payload, decode_sections=True)
        header = parsed["header"]
        if (
            header.get("tile_id") != tile.get("id")
            or header.get("kind") != "detail_tile"
        ):
            raise FWTileError(f"detail container identity mismatch: {payload_path}")
        section_headers = header.get("sections", [])
        section_names = tuple(section.get("name") for section in section_headers)
        if section_names != REQUIRED_DETAIL_SECTIONS:
            raise FWTileError(f"detail sections are incomplete: {payload_path}")
        trees = section_headers[1].get("metadata", {})
        encoding = str(trees.get("encoding"))
        if encoding not in SUPPORTED_TREE_ENCODINGS:
            raise FWTileError(f"unsupported tree encoding {encoding}: {payload_path}")
        tree_encodings[encoding] = tree_encodings.get(encoding, 0) + 1
        tree_instances += int(trees.get("count", -1))
        payload_size = int(payload_asset["byte_count"])
        imagery_size = int(imagery_asset["byte_count"])
        payload_bytes.append(payload_size)
        imagery_bytes.append(imagery_size)
        combined_bytes.append(payload_size + imagery_size)

    far = catalog.get("lod_policy", {}).get("far", {})
    terrain_resolution = far.get("terrain", {}).get("resolution_m")
    if (
        not isinstance(terrain_resolution, Sequence)
        or len(terrain_resolution) != 2
        or any(abs(float(value) - 5.0) > 0.01 for value in terrain_resolution)
    ):
        raise FWTileError("catalog FAR terrain resolution is not 5 m")
    if far.get("imagery", {}).get("resolution_m") != 2.0:
        raise FWTileError("catalog FAR imagery resolution is not 2 m")
    far_terrain = _asset_path(output_root, far.get("terrain", {}))
    far_imagery = _asset_path(output_root, far.get("imagery", {}))
    referenced_paths.update((far_terrain, far_imagery))
    parsed_far = read_container(far_terrain.read_bytes(), decode_sections=True)
    if parsed_far["header"].get("kind") != "global_far_terrain":
        raise FWTileError("far terrain container kind is invalid")

    part_files = sorted(output_root.rglob("*.part"))
    if part_files:
        raise FWTileError(f"temporary files remain in output: {part_files}")
    referenced_bytes = sum(path.stat().st_size for path in referenced_paths)
    physical_files = [path for path in output_root.rglob("*") if path.is_file()]
    physical_bytes = sum(path.stat().st_size for path in physical_files)
    return {
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "detail_tile_count": len(tiles),
        "detail_streaming_enabled": detail_enabled,
        "receipt_count": len(receipt_paths),
        "maximum_resident_tile_count": maximum_resident,
        "near_lod_disabled": near_disabled,
        "tree_instance_count": tree_instances,
        "tree_encoding_tile_counts": dict(sorted(tree_encodings.items())),
        "detail_payload_bytes": _distribution(payload_bytes),
        "detail_imagery_bytes": _distribution(imagery_bytes),
        "detail_combined_bytes": _distribution(combined_bytes),
        "far_bytes": far_terrain.stat().st_size + far_imagery.stat().st_size,
        "referenced_file_count": len(referenced_paths),
        "referenced_deploy_bytes": referenced_bytes,
        "physical_output_file_count": len(physical_files),
        "physical_output_bytes": physical_bytes,
        # Includes receipts and any immutable canary leftovers. These files
        # are useful production controls but are not referenced by catalog.json
        # and therefore are not part of the remote deploy payload.
        "non_catalog_output_bytes": physical_bytes - referenced_bytes,
        "status": "valid",
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    print(
        json.dumps(
            validate(arguments.artifact_root, arguments.output_root),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FWTileError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
