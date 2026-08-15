"""Deterministic repair of sparse invalid elevation samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt


class ElevationNodataError(ValueError):
    """An elevation grid cannot be repaired without any measured sample."""


def repair_elevation_samples(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    nodata_values: Sequence[float | None] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill invalid cells from the nearest measured cell, deterministically.

    The returned array keeps the input dtype.  Euclidean nearest-neighbour
    indices from SciPy are deterministic for identical bytes and avoid
    inventing a slope across an unmeasured gap.
    """

    source = np.asarray(values)
    if source.ndim != 2 or source.size == 0:
        raise ElevationNodataError("Elevation grid must be one non-empty 2D array")
    valid = np.isfinite(source)
    if mask is not None:
        supplied_mask = np.asarray(mask)
        if supplied_mask.shape != source.shape:
            raise ElevationNodataError("Elevation validity mask shape differs")
        valid &= supplied_mask != 0
    for nodata in nodata_values:
        if nodata is None:
            continue
        numeric = float(nodata)
        if np.isfinite(numeric):
            valid &= source != numeric

    invalid = ~valid
    invalid_count = int(np.count_nonzero(invalid))
    valid_count = int(source.size - invalid_count)
    diagnostics: dict[str, Any] = {
        "applied": invalid_count > 0,
        "invalid_sample_count": invalid_count,
        "valid_sample_count": valid_count,
        "method": "nearest_valid_sample_euclidean",
        "maximum_fill_distance_pixels": 0.0,
    }
    if invalid_count == 0:
        return source.copy(), diagnostics
    if valid_count == 0:
        raise ElevationNodataError("Elevation grid contains no measured sample")

    distances, nearest = distance_transform_edt(
        invalid,
        return_distances=True,
        return_indices=True,
    )
    repaired = source.copy()
    repaired[invalid] = source[tuple(nearest[:, invalid])]
    diagnostics["maximum_fill_distance_pixels"] = round(
        float(distances[invalid].max()), 6
    )
    if not np.isfinite(repaired).all():
        raise ElevationNodataError("Elevation repair left non-finite samples")
    return repaired, diagnostics


__all__ = [
    "ElevationNodataError",
    "repair_elevation_samples",
]
