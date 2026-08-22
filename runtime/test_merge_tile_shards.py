from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merge_tile_shards import ShardMergeError, merge_shards


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _shard(
    root: Path,
    *,
    index: int,
    count: int,
    tile_ids: list[str],
    build_id: str,
    zone_id: str,
    plan_tile_count: int,
) -> None:
    shard = root / str(index)
    prototype = (
        shard
        / "shared"
        / "prototype-bundles"
        / "v1-test"
        / f"prototype-{index}.usda"
    )
    prototype.parent.mkdir(parents=True, exist_ok=True)
    prototype.write_text(f"prototype-{index}", encoding="utf-8")
    checkpoint_root = shard / "tile-checkpoints" / "v1"
    for tile_id in tile_ids:
        archive = checkpoint_root / f"{tile_id}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(f"archive-{tile_id}".encode())
        _write_json(
            checkpoint_root / f"{tile_id}.json",
            {
                "schema": "fireviewer.simple-measured-tile-checkpoint.v1",
                "tile_id": tile_id,
                "archive": {
                    "file": archive.name,
                    "byte_count": archive.stat().st_size,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                },
            },
        )
    result = {
        "schema": "fireviewer.tile-checkpoint-shard.v1",
        "zone_id": zone_id,
        "plan_tile_count": plan_tile_count,
        "shard_index": index,
        "shard_count": count,
        "tile_count": len(tile_ids),
        "tile_ids": tile_ids,
    }
    _write_json(shard / "tile-shard-result.json", result)
    _write_json(
        shard / "shard.done.json",
        {
            "schema": "fireviewer.map-tile-shard-done.v1",
            "build_id": build_id,
            "zone_id": zone_id,
            "shard_index": index,
            "shard_count": count,
            "tile_count": len(tile_ids),
            "tile_ids": tile_ids,
        },
    )


def test_merge_shards_requires_complete_disjoint_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    build_id = "b" * 64
    zone_id = "GPS-GENERIC"
    _shard(
        source,
        index=0,
        count=2,
        tile_ids=["tile-0", "tile-2"],
        build_id=build_id,
        zone_id=zone_id,
        plan_tile_count=4,
    )
    _shard(
        source,
        index=1,
        count=2,
        tile_ids=["tile-1", "tile-3"],
        build_id=build_id,
        zone_id=zone_id,
        plan_tile_count=4,
    )

    receipt = merge_shards(
        source,
        tmp_path / "merged",
        expected_build_id=build_id,
        expected_zone_id=zone_id,
        expected_shard_count=2,
    )

    assert receipt["tile_count"] == 4
    assert len(list((tmp_path / "merged" / "tile-checkpoints" / "v1").glob("*.zip"))) == 4
    prototype_root = tmp_path / "merged" / "shared" / "prototype-bundles" / "v1-test"
    assert sorted(path.name for path in prototype_root.glob("*.usda")) == [
        "prototype-0.usda",
        "prototype-1.usda",
    ]


def test_merge_shards_rejects_duplicate_tile_assignment(tmp_path: Path) -> None:
    source = tmp_path / "source"
    build_id = "c" * 64
    zone_id = "GPS-GENERIC"
    for index in range(2):
        _shard(
            source,
            index=index,
            count=2,
            tile_ids=["same-tile"],
            build_id=build_id,
            zone_id=zone_id,
            plan_tile_count=2,
        )

    with pytest.raises(ShardMergeError, match="multiple shards"):
        merge_shards(
            source,
            tmp_path / "merged",
            expected_build_id=build_id,
            expected_zone_id=zone_id,
            expected_shard_count=2,
        )
