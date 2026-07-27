"""Run the remote tile exporter in short, isolated worker processes.

The geospatial stack used by the exporter includes native GDAL/GEOS modules.
Keeping a single Python interpreter alive for hundreds of tiles can retain
native allocations (and, on Windows, may terminate without a Python
traceback).  This production runner bounds that lifetime while preserving the
exporter's deterministic receipts and immutable assets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from export_remote_catalog import (
    DETAIL_ZONE_CONTRACT_PATH,
    _load_detail_zone_bounds,
    _prioritize_tiles,
)


DEFAULT_ARTIFACT_ROOT = Path(
    ".artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1"
)


@dataclass(frozen=True)
class BatchResult:
    tile_ids: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--global-vector-package", type=Path)
    parser.add_argument("--far-terrain", type=Path)
    parser.add_argument("--far-imagery", type=Path)
    parser.add_argument("--detail-zones", type=Path, default=DETAIL_ZONE_CONTRACT_PATH)
    parser.add_argument("--far-imagery-resolution-m", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--disable-near-lod", action="store_true")
    return parser.parse_args(argv)


def _ready_tile_ids(
    artifact_root: Path,
    detail_zone_contract: Path = DETAIL_ZONE_CONTRACT_PATH,
    production_manifest: Path | None = None,
) -> list[str]:
    manifest_path = (
        production_manifest or artifact_root / "global-05m/production-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        raise RuntimeError("production manifest is not ready")
    ready = [
        tile
        for tile in manifest["tiles"]
        if tile.get("status", {}).get("state") == "ready"
    ]
    return [
        str(tile["id"])
        for tile in _prioritize_tiles(
            ready, _load_detail_zone_bounds(detail_zone_contract)
        )
    ]


def _validate_limits(workers: int, batch_size: int) -> None:
    if workers != 1:
        raise ValueError(
            "workers must be exactly one: every exporter retains the complete "
            "global vector model and its native geospatial allocations"
        )
    if batch_size < 1:
        raise ValueError("batch size must be at least one")


def _chunks(values: Sequence[str], size: int) -> list[tuple[str, ...]]:
    return [
        tuple(values[index : index + size]) for index in range(0, len(values), size)
    ]


def _run_batch(
    exporter: Path,
    exporter_arguments: Sequence[str],
    output_root: Path,
    tile_ids: Sequence[str],
) -> BatchResult:
    command = [
        sys.executable,
        str(exporter),
        *exporter_arguments,
        "--output-root",
        str(output_root),
    ]
    for tile_id in tile_ids:
        command.extend(("--tile-id", tile_id))
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=exporter.parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return BatchResult(
        tile_ids=tuple(tile_ids),
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_seconds=time.monotonic() - started,
    )


def _print_result(result: BatchResult, complete: int, total: int) -> None:
    marker = "ok" if result.return_code == 0 else "failed"
    print(
        json.dumps(
            {
                "batch": marker,
                "complete_receipts": complete,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "return_code": result.return_code,
                "tile_count": len(result.tile_ids),
                "tile_first": result.tile_ids[0],
                "tile_last": result.tile_ids[-1],
                "total_ready": total,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if result.return_code:
        if result.stdout.strip():
            print(result.stdout.rstrip(), flush=True)
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr, flush=True)


def _receipt_count(output_root: Path) -> int:
    return len(list((output_root / "receipts").glob("x*_s*.json")))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    _validate_limits(arguments.workers, arguments.batch_size)

    artifact_root = arguments.artifact_root.resolve()
    output_root = arguments.output_root.resolve()
    exporter = Path(__file__).with_name("export_remote_catalog.py").resolve()
    production_manifest = (
        arguments.production_manifest.resolve()
        if arguments.production_manifest
        else artifact_root / "global-05m/production-manifest.json"
    )
    detail_zones = arguments.detail_zones.resolve()
    ready = _ready_tile_ids(artifact_root, detail_zones, production_manifest)
    if not ready:
        raise RuntimeError("production manifest contains no ready tile")
    exporter_arguments = [
        "--artifact-root",
        str(artifact_root),
        "--production-manifest",
        str(production_manifest),
        "--detail-zones",
        str(detail_zones),
        "--far-imagery-resolution-m",
        str(arguments.far_imagery_resolution_m),
    ]
    if arguments.disable_near_lod:
        exporter_arguments.append("--disable-near-lod")
    for option, value in (
        ("--global-vector-package", arguments.global_vector_package),
        ("--far-terrain", arguments.far_terrain),
        ("--far-imagery", arguments.far_imagery),
    ):
        if value is not None:
            exporter_arguments.extend((option, str(value.resolve())))
    pending = [
        tile_id
        for tile_id in ready
        if not (output_root / "receipts" / f"{tile_id}.json").is_file()
    ]
    print(
        json.dumps(
            {
                "batch_size": arguments.batch_size,
                "existing_receipts": len(ready) - len(pending),
                "pending": len(pending),
                "total_ready": len(ready),
                "workers": arguments.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    failures: list[BatchResult] = []
    batches = _chunks(pending, arguments.batch_size)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=arguments.workers
    ) as executor:
        futures = [
            executor.submit(
                _run_batch, exporter, exporter_arguments, output_root, batch
            )
            for batch in batches
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            _print_result(result, _receipt_count(output_root), len(ready))
            if result.return_code:
                failures.append(result)

    # A failed native batch has no trustworthy indication of which remaining
    # tile triggered it. Retry only its still-missing tiles, one fresh process
    # per tile, and retain every successful receipt from the first attempt.
    retry_ids = sorted(
        {
            tile_id
            for failure in failures
            for tile_id in failure.tile_ids
            if not (output_root / "receipts" / f"{tile_id}.json").is_file()
        }
    )
    retry_failures: list[BatchResult] = []
    if retry_ids:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=arguments.workers
        ) as executor:
            futures = [
                executor.submit(
                    _run_batch, exporter, exporter_arguments, output_root, (tile_id,)
                )
                for tile_id in retry_ids
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                _print_result(result, _receipt_count(output_root), len(ready))
                if result.return_code:
                    retry_failures.append(result)

    missing = [
        tile_id
        for tile_id in ready
        if not (output_root / "receipts" / f"{tile_id}.json").is_file()
    ]
    if missing or retry_failures:
        print(
            json.dumps(
                {
                    "missing_count": len(missing),
                    "missing_tile_ids": missing,
                    "status": "incomplete",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1

    # Reopen one existing tile in a final isolated process. The exporter
    # validates that receipt and atomically rebuilds catalog.json from every
    # receipt, without keeping the native geospatial stack alive for the full
    # production run.
    final = _run_batch(exporter, exporter_arguments, output_root, (ready[0],))
    _print_result(final, _receipt_count(output_root), len(ready))
    if final.return_code:
        return final.return_code
    print(
        json.dumps(
            {
                "catalog": str(output_root / "catalog.json"),
                "receipts": _receipt_count(output_root),
                "status": "complete",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
