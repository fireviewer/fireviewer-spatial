from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from run_export_batches import _ready_tile_ids, _validate_limits  # noqa: E402


def test_runner_rejects_parallel_global_vector_workers() -> None:
    _validate_limits(1, 4)
    with pytest.raises(ValueError, match="exactly one"):
        _validate_limits(2, 4)


def test_runner_prioritizes_contract_zones_before_remaining_tiles(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    manifest_path = artifact_root / "global-05m" / "production-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "tiles": [
                    {
                        "id": "outside",
                        "bounds_l93_m": [100.0, 100.0, 110.0, 110.0],
                        "status": {"state": "ready"},
                    },
                    {
                        "id": "ausson",
                        "bounds_l93_m": [50.0, 50.0, 60.0, 60.0],
                        "status": {"state": "ready"},
                    },
                    {
                        "id": "montmaur",
                        "bounds_l93_m": [10.0, 10.0, 20.0, 20.0],
                        "status": {"state": "ready"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "detail_zones.v1.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "horizontal_crs": "EPSG:2154",
                "zones": [
                    {
                        "id": "montmaur",
                        "bounds_l93_metres": [10.5, 10.5, 20.5, 20.5],
                    },
                    {
                        "id": "ausson",
                        "bounds_l93_metres": [50.5, 50.5, 60.5, 60.5],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _ready_tile_ids(artifact_root, contract_path) == [
        "montmaur",
        "ausson",
        "outside",
    ]
