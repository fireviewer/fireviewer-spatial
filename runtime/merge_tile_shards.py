#!/usr/bin/env python3
"""Fail-closed merge of deterministic Map Builder tile checkpoint shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


class ShardMergeError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShardMergeError(f"invalid shard JSON: {path}") from error
    if not isinstance(value, dict):
        raise ShardMergeError(f"shard JSON must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_identical_or_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            destination.stat().st_size != source.stat().st_size
            or _sha256(destination) != _sha256(source)
        ):
            raise ShardMergeError(f"conflicting shard artifact: {destination}")
        return
    shutil.copy2(source, destination)


def merge_shards(
    source_root: Path,
    output_root: Path,
    *,
    expected_build_id: str,
    expected_zone_id: str,
    expected_shard_count: int,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ShardMergeError(f"merge output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    observed_indices: set[int] = set()
    observed_tile_ids: set[str] = set()
    plan_tile_count: int | None = None
    shard_metrics: list[dict[str, Any]] = []

    for shard_index in range(expected_shard_count):
        shard_root = source_root / str(shard_index)
        done = _json(shard_root / "shard.done.json")
        result = _json(shard_root / "tile-shard-result.json")
        if (
            done.get("schema") != "fireviewer.map-tile-shard-done.v1"
            or done.get("build_id") != expected_build_id
            or done.get("zone_id") != expected_zone_id
            or done.get("shard_index") != shard_index
            or done.get("shard_count") != expected_shard_count
            or result.get("schema") != "fireviewer.tile-checkpoint-shard.v1"
            or result.get("zone_id") != expected_zone_id
            or result.get("shard_index") != shard_index
            or result.get("shard_count") != expected_shard_count
            or done.get("tile_ids") != result.get("tile_ids")
        ):
            raise ShardMergeError(f"shard identity mismatch: {shard_index}")
        tile_ids = result.get("tile_ids")
        if (
            not isinstance(tile_ids, list)
            or not tile_ids
            or any(not isinstance(tile_id, str) or not tile_id for tile_id in tile_ids)
            or result.get("tile_count") != len(tile_ids)
        ):
            raise ShardMergeError(f"invalid tile inventory for shard {shard_index}")
        duplicate_tiles = observed_tile_ids.intersection(tile_ids)
        if duplicate_tiles:
            raise ShardMergeError(
                "tile assigned to multiple shards: " + ", ".join(sorted(duplicate_tiles))
            )
        observed_tile_ids.update(tile_ids)
        observed_indices.add(shard_index)
        current_plan_count = result.get("plan_tile_count")
        if not isinstance(current_plan_count, int) or current_plan_count < 1:
            raise ShardMergeError(f"invalid plan tile count for shard {shard_index}")
        if plan_tile_count is None:
            plan_tile_count = current_plan_count
        elif plan_tile_count != current_plan_count:
            raise ShardMergeError("shards disagree on the plan tile count")

        checkpoint_root = shard_root / "tile-checkpoints" / "v1"
        for tile_id in tile_ids:
            receipt_path = checkpoint_root / f"{tile_id}.json"
            archive_path = checkpoint_root / f"{tile_id}.zip"
            receipt = _json(receipt_path)
            archive = receipt.get("archive")
            if (
                receipt.get("schema")
                != "fireviewer.simple-measured-tile-checkpoint.v1"
                or receipt.get("tile_id") != tile_id
                or not isinstance(archive, Mapping)
                or archive.get("file") != archive_path.name
                or archive.get("byte_count") != archive_path.stat().st_size
                or archive.get("sha256") != _sha256(archive_path)
            ):
                raise ShardMergeError(f"invalid checkpoint for tile {tile_id}")

        for collection in ("tile-checkpoints", "prototype-bundles", "provenance"):
            collection_root = shard_root / collection
            if not collection_root.is_dir():
                continue
            for source in sorted(collection_root.rglob("*")):
                if source.is_file():
                    _copy_identical_or_new(
                        source,
                        output_root / collection / source.relative_to(collection_root),
                    )
        metrics_path = shard_root / "metrics" / "shard-metrics.json"
        if metrics_path.is_file():
            shard_metrics.append(_json(metrics_path))

    if observed_indices != set(range(expected_shard_count)):
        raise ShardMergeError("one or more shard indices are missing")
    if plan_tile_count != len(observed_tile_ids):
        raise ShardMergeError(
            f"merged tile inventory is incomplete: {len(observed_tile_ids)}/{plan_tile_count}"
        )
    receipt = {
        "schema": "fireviewer.map-tile-shard-merge.v1",
        "build_id": expected_build_id,
        "zone_id": expected_zone_id,
        "shard_count": expected_shard_count,
        "tile_count": len(observed_tile_ids),
        "tile_ids": sorted(observed_tile_ids),
        "shard_metrics": shard_metrics,
    }
    (output_root / "shard-merge.json").write_text(
        json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--shard-count", required=True, type=int)
    args = parser.parse_args()
    receipt = merge_shards(
        Path(args.source).resolve(strict=True),
        Path(args.output).resolve(),
        expected_build_id=args.build_id,
        expected_zone_id=args.zone_id,
        expected_shard_count=args.shard_count,
    )
    print(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
