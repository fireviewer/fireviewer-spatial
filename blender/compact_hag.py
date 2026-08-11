"""Deterministic compact HAG reservation on the canonical 2 m tile grid."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin


SCHEMA = "fireviewer.hag-max-2m.v1"
CRS = "EPSG:2154"
TILE_SIZE_M = 500.0
RESOLUTION_M = 2.0
GRID_SIZE = 250
NODATA = 65_535
CANONICAL_HALO_SIZE = 253


def quantize_hag_max_cm(mnt_m: np.ndarray, mns_m: np.ndarray) -> np.ndarray:
    """Return one uint16 centimetre maximum for each 2 m terrain cell.

    MNT and MNS are vertex samples ordered south-to-north, west-to-east.  The
    output contains cell maxima, so 251 x 251 source vertices become the exact
    250 x 250 pixels that cover a 500 m tile.
    """

    mnt = np.asarray(mnt_m, dtype="float64")
    mns = np.asarray(mns_m, dtype="float64")
    if mnt.shape != (GRID_SIZE + 1, GRID_SIZE + 1) or mns.shape != mnt.shape:
        raise ValueError("MNT and MNS HAG inputs must have shape (251, 251)")
    if not np.isfinite(mnt).all() or not np.isfinite(mns).all():
        raise ValueError("MNT and MNS HAG inputs must not contain nodata")
    delta_cm = np.maximum(mns - mnt, 0.0) * 100.0
    rounded = np.floor(delta_cm + 0.5)
    if float(rounded.max()) >= NODATA:
        raise ValueError("HAG exceeds the uint16 centimetre contract")
    vertices = rounded.astype("uint16")
    return np.maximum.reduce(
        (
            vertices[:-1, :-1],
            vertices[1:, :-1],
            vertices[:-1, 1:],
            vertices[1:, 1:],
        )
    )


def quantize_hag_max_cm_from_canonical_mm(
    mnt_normal_halo_mm: np.ndarray, mns_normal_halo_mm: np.ndarray
) -> np.ndarray:
    """Build HAG directly from the retained canonical source halo grids."""

    mnt_halo = np.asarray(mnt_normal_halo_mm)
    mns_halo = np.asarray(mns_normal_halo_mm)
    expected = (CANONICAL_HALO_SIZE, CANONICAL_HALO_SIZE)
    if mnt_halo.shape != expected or mns_halo.shape != expected:
        raise ValueError("canonical MNT/MNS normal halos must have shape (253, 253)")
    if mnt_halo.dtype.kind not in {"i", "u"} or mns_halo.dtype.kind not in {
        "i",
        "u",
    }:
        raise ValueError("canonical MNT/MNS normal halos must contain millimetres")
    mnt = np.asarray(mnt_halo[1:-1, 1:-1], dtype="int64")
    mns = np.asarray(mns_halo[1:-1, 1:-1], dtype="int64")
    delta_mm = np.maximum(mns - mnt, 0)
    rounded_cm = (delta_mm + 5) // 10
    if int(rounded_cm.max()) >= NODATA:
        raise ValueError("HAG exceeds the uint16 centimetre contract")
    vertices = np.asarray(rounded_cm, dtype="uint16")
    return np.maximum.reduce(
        (
            vertices[:-1, :-1],
            vertices[1:, :-1],
            vertices[:-1, 1:],
            vertices[1:, 1:],
        )
    )


def _origin(value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("tile_origin_l93_m must contain easting and northing")
    easting, northing = (float(component) for component in value)
    if not math.isfinite(easting) or not math.isfinite(northing):
        raise ValueError("tile_origin_l93_m must contain finite values")
    if not math.isclose(easting / TILE_SIZE_M, round(easting / TILE_SIZE_M)):
        raise ValueError("HAG tile easting must align to the 500 m grid")
    if not math.isclose(northing / TILE_SIZE_M, round(northing / TILE_SIZE_M)):
        raise ValueError("HAG tile northing must align to the 500 m grid")
    return easting, northing


def _raw_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<u2").tobytes(order="C")
    ).hexdigest()


def write_hag_max_2m(
    path: Path,
    values_cm: np.ndarray,
    *,
    tile_origin_l93_m: Sequence[float],
) -> str:
    """Write an atomic, compressed and self-validating `hag-max-2m.tif`."""

    values = np.asarray(values_cm)
    if values.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError("HAG cell grid must have shape (250, 250)")
    if values.dtype.kind not in {"u", "i"}:
        raise ValueError("HAG centimetres must be integer values")
    if int(values.min()) < 0 or int(values.max()) >= NODATA:
        raise ValueError("HAG centimetres must fit below the nodata sentinel")
    canonical = np.asarray(values, dtype="uint16")
    west, south = _origin(tile_origin_l93_m)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.tif")
    temporary.unlink(missing_ok=True)
    try:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            width=GRID_SIZE,
            height=GRID_SIZE,
            count=1,
            dtype="uint16",
            crs=CRS,
            transform=from_origin(
                west,
                south + TILE_SIZE_M,
                RESOLUTION_M,
                RESOLUTION_M,
            ),
            nodata=NODATA,
            compress="DEFLATE",
            predictor=2,
            zlevel=9,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            BIGTIFF="NO",
            NUM_THREADS="1",
        ) as dataset:
            dataset.write(np.flipud(canonical), 1)
            dataset.update_tags(
                FIREVIEWER_SCHEMA=SCHEMA,
                FIREVIEWER_UNIT="centimetre",
                FIREVIEWER_ROW_ORDER="south_to_north",
                FIREVIEWER_RAW_SHA256=_raw_sha256(canonical),
                FIREVIEWER_SOURCE="MNS_minus_MNT_cell_max",
            )
        content = temporary.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if destination.exists():
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"Refusing to replace different HAG: {destination}"
                )
            temporary.unlink()
        else:
            temporary.replace(destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def read_hag_max_2m(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Read and validate a compact HAG, returning south-to-north cells."""

    with rasterio.open(path) as dataset:
        if (
            dataset.crs is None
            or dataset.crs.to_string() != CRS
            or dataset.width != GRID_SIZE
            or dataset.height != GRID_SIZE
            or dataset.count != 1
            or dataset.dtypes[0] != "uint16"
            or dataset.nodata != NODATA
        ):
            raise ValueError("HAG GeoTIFF raster contract mismatch")
        tags = dataset.tags()
        if tags.get("FIREVIEWER_SCHEMA") != SCHEMA:
            raise ValueError("HAG GeoTIFF schema tag mismatch")
        values = np.flipud(dataset.read(1)).copy()
        if tags.get("FIREVIEWER_RAW_SHA256") != _raw_sha256(values):
            raise ValueError("HAG GeoTIFF raw payload hash mismatch")
        bounds = dataset.bounds
        metadata = {
            "schema": SCHEMA,
            "crs": CRS,
            "bounds_l93_m": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "resolution_m": RESOLUTION_M,
            "unit": "centimetre",
            "nodata": NODATA,
            "raw_sha256": tags["FIREVIEWER_RAW_SHA256"],
        }
    return values, metadata
