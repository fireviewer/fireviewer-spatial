"""Side-by-side tile producer using the factual placement v2 profile.

The stable v1 producer is reused rather than copied. This module injects one
per-call dispatcher for the placement step and keeps the native 0.5 m
MNT/MNS arrays in thread-local storage so ProductionEngine can still process
multiple tiles concurrently.  The Lightning r46 image does not import this
module and remains a byte-for-byte independent comparison reference.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

import produce_simple_measured_tile as legacy
from mns_mnt_placement_inventory_v2 import build_placement_inventory_v2

_TLS = threading.local()
_ORIGINAL_PIPELINE_HASHES = legacy._pipeline_file_hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pipeline_file_hashes_v2() -> dict[str, str]:
    values = dict(_ORIGINAL_PIPELINE_HASHES())
    files = {
        "tile_profile_v2": Path(__file__),
        "placement_inventory_v2": Path(__file__).with_name(
            "mns_mnt_placement_inventory_v2.py"
        ),
        "placement_contract_v2": Path(__file__).with_name(
            "mns_mnt_placement_contract.v2.json"
        ),
    }
    values.update(
        {
            name: _sha256_file(path.resolve(strict=True))
            for name, path in sorted(files.items())
        }
    )
    return values


def _native_pair_from_inputs(
    mnt_path: Path | str,
    mns_path: Path | str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the source pair required by factual v2 placement."""

    try:
        with rasterio.open(Path(mnt_path)) as mnt_dataset:
            if (
                mnt_dataset.width != legacy.ELEVATION_SOURCE_SIZE
                or mnt_dataset.height != legacy.ELEVATION_SOURCE_SIZE
                or mnt_dataset.count < 1
            ):
                return None
            mnt = np.asarray(mnt_dataset.read(1), dtype="float64")
            mnt_signature = (
                mnt_dataset.width,
                mnt_dataset.height,
                mnt_dataset.crs.to_string() if mnt_dataset.crs else None,
                tuple(mnt_dataset.transform)[:6],
            )
        with rasterio.open(Path(mns_path)) as mns_dataset:
            if (
                mns_dataset.width != legacy.ELEVATION_SOURCE_SIZE
                or mns_dataset.height != legacy.ELEVATION_SOURCE_SIZE
                or mns_dataset.count < 1
            ):
                return None
            mns = np.asarray(mns_dataset.read(1), dtype="float64")
            mns_signature = (
                mns_dataset.width,
                mns_dataset.height,
                mns_dataset.crs.to_string() if mns_dataset.crs else None,
                tuple(mns_dataset.transform)[:6],
            )
    except (OSError, rasterio.errors.RasterioError):
        return None
    if mnt_signature != mns_signature:
        return None
    if not np.isfinite(mnt).all() or not np.isfinite(mns).all():
        return None
    return mnt, mns


def _placement_dispatcher(
    mnt_m: Any,
    mns_m: Any,
    **kwargs: Any,
) -> Any:
    native = getattr(_TLS, "native_pair", None)
    native_mnt: np.ndarray | None = None
    native_mns: np.ndarray | None = None
    if native is not None:
        native_mnt, native_mns = native
    return build_placement_inventory_v2(
        mnt_m,
        mns_m,
        native_mnt_05m=native_mnt,
        native_mns_05m=native_mns,
        **kwargs,
    )


def produce_simple_measured_tile_v2(
    *,
    mnt_05m: Path | str,
    mns_05m: Path | str,
    orthophoto_1m: Path | str,
    elevation_source_receipt: Path | str,
    orthophoto_source_receipt: Path | str,
    placement_context: Path | str,
    asset_library: Path | str,
    asset_roots: Mapping[str, Path | str],
    portable_root: Path | str,
    output_root: Path | str,
    zone_id: str,
    tile_id: str,
    tile_origin_l93_m: Sequence[float],
    asset_bundle_root: Path | str | None = None,
    asset_bundle_identity_root: Path | str | None = None,
    progress_callback: legacy.ProgressCallback | None = None,
) -> legacy.SimpleMeasuredTilePackage:
    """Produce one tile with v2 placement while retaining the v1 package shape."""

    if getattr(_TLS, "native_pair", None) is not None:
        raise legacy.SimpleMeasuredTileError(
            "nested v2 tile production on one worker thread is forbidden"
        )
    native_pair = _native_pair_from_inputs(mnt_05m, mns_05m)
    if native_pair is None:
        raise legacy.SimpleMeasuredTileError(
            "factual v2 requires a finite, co-registered native 0.5 m MNT/MNS pair; "
            "publishing an empty degraded tile is forbidden"
        )
    _TLS.native_pair = native_pair
    try:
        return legacy.produce_simple_measured_tile(
            mnt_05m=mnt_05m,
            mns_05m=mns_05m,
            orthophoto_1m=orthophoto_1m,
            elevation_source_receipt=elevation_source_receipt,
            orthophoto_source_receipt=orthophoto_source_receipt,
            placement_context=placement_context,
            asset_library=asset_library,
            asset_roots=asset_roots,
            portable_root=portable_root,
            output_root=output_root,
            zone_id=zone_id,
            tile_id=tile_id,
            tile_origin_l93_m=tile_origin_l93_m,
            asset_bundle_root=asset_bundle_root,
            asset_bundle_identity_root=asset_bundle_identity_root,
            progress_callback=progress_callback,
            _placement_builder=_placement_dispatcher,
            _pipeline_hash_provider=_pipeline_file_hashes_v2,
        )
    finally:
        _TLS.native_pair = None


# ProductionEngine expects a callable named by convention in several fixtures.
produce_simple_measured_tile = produce_simple_measured_tile_v2


__all__ = [
    "produce_simple_measured_tile",
    "produce_simple_measured_tile_v2",
]
