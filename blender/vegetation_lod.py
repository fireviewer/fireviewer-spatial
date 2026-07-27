"""Deterministic mid-distance vegetation proxies derived from MNS/MNT rasters.

The module is deliberately independent from :mod:`bpy`.  Detection can run in
the package-preparation Python process and the returned ``mesh`` mapping can be
passed directly to ``build_control_scene._mesh_object`` (``vertices`` and
``faces`` use the same JSON-compatible representation).

Integration contract
--------------------

``mnt`` and ``mns`` must be aligned, two-dimensional arrays.  Both masks use
the *raster sample* grid and therefore have exactly the same shape as the two
rasters.  This differs from the cell mask consumed by the existing canopy
shell.  A caller using ``rasterio.features.geometry_mask`` should use the MNT
shape and transform directly::

    vegetation_samples = geometry_mask(
        vegetation_geometries,
        out_shape=mnt.shape,
        transform=transform,
        invert=True,
    )
    exclusion_samples = geometry_mask(
        building_and_infrastructure_geometries,
        out_shape=mnt.shape,
        transform=transform,
        invert=True,
    )
    lod = generate_mid_distance_vegetation_lod(
        mnt,
        mns,
        transform,
        vegetation_samples,
        exclusion_samples,
        origin,
        VegetationLodConfig(min_spacing_m=15.0),
    )

The output is a visual LOD, not an individual-tree inventory.  Every proxy is
anchored to the co-located MNT sample and reaches the co-located MNS sample.
Its crown diameter is an equivalent-area estimate from connected canopy
support around the local maximum, bounded by a documented search radius.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence


LOD_SCHEMA = "fireviewer.vegetation-mid-distance-lod.v1"


@dataclass(frozen=True)
class VegetationLodConfig:
    """Controls deterministic peak selection and lightweight proxy geometry."""

    min_tree_height_m: float = 3.0
    local_max_radius_m: float = 7.5
    min_spacing_m: float = 15.0
    crown_search_radius_m: float | None = None
    crown_support_height_ratio: float = 0.45
    max_proxy_count: int | None = None
    crown_radial_segments: int = 6
    crown_base_height_ratio: float = 0.22
    crown_widest_height_ratio: float = 0.58
    trunk_radius_to_crown_ratio: float = 0.08
    minimum_trunk_radius_m: float = 0.12
    maximum_trunk_radius_m: float = 0.60
    coordinate_precision: int = 3

    def __post_init__(self) -> None:
        positive_values = (
            ("min_tree_height_m", self.min_tree_height_m),
            ("local_max_radius_m", self.local_max_radius_m),
            ("min_spacing_m", self.min_spacing_m),
        )
        for name, value in positive_values:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if self.crown_search_radius_m is not None and (
            not math.isfinite(self.crown_search_radius_m)
            or self.crown_search_radius_m <= 0
        ):
            raise ValueError(
                "crown_search_radius_m must be finite and strictly positive"
            )
        if not math.isfinite(self.crown_support_height_ratio) or not (
            0 < self.crown_support_height_ratio <= 1
        ):
            raise ValueError("crown_support_height_ratio must be in ]0, 1]")
        if self.max_proxy_count is not None and (
            isinstance(self.max_proxy_count, bool)
            or not isinstance(self.max_proxy_count, int)
            or self.max_proxy_count <= 0
        ):
            raise ValueError("max_proxy_count must be a strictly positive integer")
        if (
            isinstance(self.crown_radial_segments, bool)
            or not isinstance(self.crown_radial_segments, int)
            or not 4 <= self.crown_radial_segments <= 8
        ):
            raise ValueError("crown_radial_segments must be an integer between 4 and 8")
        if not math.isfinite(self.crown_base_height_ratio) or not (
            0 < self.crown_base_height_ratio < 1
        ):
            raise ValueError("crown_base_height_ratio must be in ]0, 1[")
        if not math.isfinite(self.crown_widest_height_ratio) or not (
            self.crown_base_height_ratio < self.crown_widest_height_ratio < 1
        ):
            raise ValueError(
                "crown_widest_height_ratio must be greater than crown_base_height_ratio "
                "and lower than 1"
            )
        for name, value in (
            ("trunk_radius_to_crown_ratio", self.trunk_radius_to_crown_ratio),
            ("minimum_trunk_radius_m", self.minimum_trunk_radius_m),
            ("maximum_trunk_radius_m", self.maximum_trunk_radius_m),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if self.minimum_trunk_radius_m > self.maximum_trunk_radius_m:
            raise ValueError(
                "minimum_trunk_radius_m must not exceed maximum_trunk_radius_m"
            )
        if (
            isinstance(self.coordinate_precision, bool)
            or not isinstance(self.coordinate_precision, int)
            or not 0 <= self.coordinate_precision <= 9
        ):
            raise ValueError("coordinate_precision must be an integer between 0 and 9")

    @property
    def resolved_crown_search_radius_m(self) -> float:
        """Use half the enforced spacing unless an explicit radius is supplied."""

        if self.crown_search_radius_m is not None:
            return self.crown_search_radius_m
        return self.min_spacing_m * 0.5


@dataclass(frozen=True)
class TreeProxy:
    """One measured LOD representative; it is not claimed to be an exact tree."""

    row: int
    column: int
    x_m: float
    y_m: float
    ground_elevation_m: float
    top_elevation_m: float
    height_m: float
    crown_support_pixel_count: int
    crown_area_m2: float
    crown_diameter_m: float
    crown_search_limited: bool


@dataclass(frozen=True)
class _AffineValues:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    determinant: float

    def sample_xy(self, row: int, column: int) -> tuple[float, float]:
        pixel_x = column + 0.5
        pixel_y = row + 0.5
        return (
            self.a * pixel_x + self.b * pixel_y + self.c,
            self.d * pixel_x + self.e * pixel_y + self.f,
        )

    def offset_xy(self, delta_row: int, delta_column: int) -> tuple[float, float]:
        return (
            self.a * delta_column + self.b * delta_row,
            self.d * delta_column + self.e * delta_row,
        )

    @property
    def pixel_area_m2(self) -> float:
        return abs(self.determinant)

    @property
    def pixel_circumradius_m(self) -> float:
        return 0.5 * max(
            math.hypot(
                column_sign * self.a + row_sign * self.b,
                column_sign * self.d + row_sign * self.e,
            )
            for column_sign in (-1, 1)
            for row_sign in (-1, 1)
        )


def _affine_values(transform: Any) -> _AffineValues:
    try:
        values = _AffineValues(
            a=float(transform.a),
            b=float(transform.b),
            c=float(transform.c),
            d=float(transform.d),
            e=float(transform.e),
            f=float(transform.f),
            determinant=float(transform.a) * float(transform.e)
            - float(transform.b) * float(transform.d),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "transform must expose finite affine coefficients a through f"
        ) from error
    coefficients = (values.a, values.b, values.c, values.d, values.e, values.f)
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("transform coefficients must be finite")
    if not math.isfinite(values.determinant) or abs(values.determinant) <= 1e-12:
        raise ValueError("transform must have a non-zero two-dimensional pixel area")
    return values


def _metric_offsets(affine: _AffineValues, radius_m: float) -> list[tuple[int, int]]:
    """Enumerate raster offsets whose sample centres lie inside a metric disk."""

    inverse_denominator = abs(affine.determinant)
    max_columns = math.ceil(
        radius_m * math.hypot(affine.e, affine.b) / inverse_denominator
    )
    max_rows = math.ceil(
        radius_m * math.hypot(affine.d, affine.a) / inverse_denominator
    )
    radius_squared = radius_m * radius_m
    tolerance = max(1e-9, radius_squared * 1e-12)
    offsets: list[tuple[int, int]] = []
    for delta_row in range(-max_rows, max_rows + 1):
        for delta_column in range(-max_columns, max_columns + 1):
            delta_x, delta_y = affine.offset_xy(delta_row, delta_column)
            if delta_x * delta_x + delta_y * delta_y <= radius_squared + tolerance:
                offsets.append((delta_row, delta_column))
    offsets.sort()
    return offsets


def _overlap_slices(
    height: int,
    width: int,
    delta_row: int,
    delta_column: int,
) -> tuple[slice, slice, slice, slice]:
    row_start = max(0, -delta_row)
    row_stop = min(height, height - delta_row)
    column_start = max(0, -delta_column)
    column_stop = min(width, width - delta_column)
    return (
        slice(row_start, row_stop),
        slice(column_start, column_stop),
        slice(row_start + delta_row, row_stop + delta_row),
        slice(column_start + delta_column, column_stop + delta_column),
    )


def _local_maxima(
    canopy_height: Any,
    eligible: Any,
    affine: _AffineValues,
    radius_m: float,
) -> tuple[Any, int]:
    """Return strict deterministic maxima and the pre-tie plateau pixel count."""

    import numpy as np

    height, width = canopy_height.shape
    offsets = _metric_offsets(affine, radius_m)
    maxima = np.asarray(eligible, dtype=bool).copy()

    # First retain every member of a locally maximal equal-height plateau.
    for delta_row, delta_column in offsets:
        if delta_row == 0 and delta_column == 0:
            continue
        centre_rows, centre_columns, neighbour_rows, neighbour_columns = (
            _overlap_slices(height, width, delta_row, delta_column)
        )
        neighbour_is_eligible = eligible[neighbour_rows, neighbour_columns]
        neighbour_is_higher = (
            canopy_height[neighbour_rows, neighbour_columns]
            > canopy_height[centre_rows, centre_columns]
        )
        maxima[centre_rows, centre_columns] &= ~(
            neighbour_is_eligible & neighbour_is_higher
        )

    plateau_pixel_count = int(np.count_nonzero(maxima))
    plateau_members = maxima.copy()

    # Then choose the lexicographically first equal member in the metric
    # neighbourhood.  This removes raster plateaus without random jitter.
    for delta_row, delta_column in offsets:
        if not (delta_row < 0 or (delta_row == 0 and delta_column < 0)):
            continue
        centre_rows, centre_columns, neighbour_rows, neighbour_columns = (
            _overlap_slices(height, width, delta_row, delta_column)
        )
        equal_predecessor = plateau_members[neighbour_rows, neighbour_columns] & (
            canopy_height[neighbour_rows, neighbour_columns]
            == canopy_height[centre_rows, centre_columns]
        )
        maxima[centre_rows, centre_columns] &= ~equal_predecessor
    return maxima, plateau_pixel_count


def _select_with_spacing(
    canopy_height: Any,
    local_maxima: Any,
    affine: _AffineValues,
    spacing_m: float,
) -> tuple[list[tuple[int, int, float, float]], int]:
    """Greedily retain highest maxima with deterministic metric spacing."""

    import numpy as np

    coordinates = np.argwhere(local_maxima)
    if coordinates.size == 0:
        return [], 0
    rows = coordinates[:, 0]
    columns = coordinates[:, 1]
    heights = canopy_height[rows, columns]
    order = np.lexsort((columns, rows, -heights))

    bucket_size = spacing_m
    spacing_squared = spacing_m * spacing_m
    tolerance = max(1e-9, spacing_squared * 1e-12)
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    selected: list[tuple[int, int, float, float]] = []
    rejected = 0

    for position in order:
        row = int(rows[position])
        column = int(columns[position])
        x_m, y_m = affine.sample_xy(row, column)
        bucket_x = math.floor(x_m / bucket_size)
        bucket_y = math.floor(y_m / bucket_size)
        too_close = False
        for neighbour_bucket_x in range(bucket_x - 1, bucket_x + 2):
            for neighbour_bucket_y in range(bucket_y - 1, bucket_y + 2):
                for other_x, other_y in buckets.get(
                    (neighbour_bucket_x, neighbour_bucket_y), ()
                ):
                    distance_squared = (x_m - other_x) ** 2 + (y_m - other_y) ** 2
                    if distance_squared < spacing_squared - tolerance:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            rejected += 1
            continue
        selected.append((row, column, x_m, y_m))
        buckets.setdefault((bucket_x, bucket_y), []).append((x_m, y_m))
    return selected, rejected


def _measure_crown(
    row: int,
    column: int,
    peak_height_m: float,
    canopy_height: Any,
    eligible: Any,
    affine: _AffineValues,
    config: VegetationLodConfig,
) -> tuple[int, float, float, bool]:
    """Measure connected relative-height support around one selected peak."""

    raster_height, raster_width = canopy_height.shape
    threshold_m = max(
        config.min_tree_height_m,
        peak_height_m * config.crown_support_height_ratio,
    )
    radius_m = config.resolved_crown_search_radius_m
    radius_squared = radius_m * radius_m
    tolerance = max(1e-9, radius_squared * 1e-12)
    queue: deque[tuple[int, int]] = deque(((row, column),))
    visited: set[tuple[int, int]] = set()
    support_pixel_count = 0
    maximum_centre_distance_m = 0.0

    while queue:
        current_row, current_column = queue.popleft()
        key = (current_row, current_column)
        if key in visited:
            continue
        visited.add(key)
        if not (
            0 <= current_row < raster_height and 0 <= current_column < raster_width
        ):
            continue
        delta_row = current_row - row
        delta_column = current_column - column
        delta_x, delta_y = affine.offset_xy(delta_row, delta_column)
        distance_squared = delta_x * delta_x + delta_y * delta_y
        if distance_squared > radius_squared + tolerance:
            continue
        if not eligible[current_row, current_column]:
            continue
        if canopy_height[current_row, current_column] < threshold_m:
            continue

        support_pixel_count += 1
        maximum_centre_distance_m = max(
            maximum_centre_distance_m, math.sqrt(max(0.0, distance_squared))
        )
        for neighbour_row in range(current_row - 1, current_row + 2):
            for neighbour_column in range(current_column - 1, current_column + 2):
                if neighbour_row == current_row and neighbour_column == current_column:
                    continue
                if (neighbour_row, neighbour_column) not in visited:
                    queue.append((neighbour_row, neighbour_column))

    # The peak always passes its own relative threshold, so this guard exposes
    # a programming/input-contract error rather than inventing a diameter.
    if support_pixel_count <= 0:
        raise RuntimeError("selected peak has no measurable connected crown support")
    crown_area_m2 = support_pixel_count * affine.pixel_area_m2
    crown_diameter_m = 2.0 * math.sqrt(crown_area_m2 / math.pi)
    search_limited = (
        maximum_centre_distance_m + affine.pixel_circumradius_m
        >= radius_m - max(1e-9, radius_m * 1e-12)
    )
    return support_pixel_count, crown_area_m2, crown_diameter_m, search_limited


def _distribution(values: Iterable[float], label: str) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            f"minimum_{label}": None,
            f"mean_{label}": None,
            f"median_{label}": None,
            f"maximum_{label}": None,
        }
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) * 0.5
    )
    return {
        f"minimum_{label}": ordered[0],
        f"mean_{label}": sum(ordered) / len(ordered),
        f"median_{label}": median,
        f"maximum_{label}": ordered[-1],
    }


def detect_tree_proxies(
    mnt: Any,
    mns: Any,
    transform: Any,
    vegetation_mask: Any,
    exclusion_mask: Any,
    config: VegetationLodConfig | None = None,
) -> tuple[list[TreeProxy], dict[str, Any]]:
    """Detect reproducible MNS maxima and return measured LOD representatives.

    The exclusion mask is intentionally mandatory: buildings, roads, water or
    known invalid surfaces must be removed by the integrating pipeline rather
    than guessed here from elevation alone.
    """

    import numpy as np

    active_config = config or VegetationLodConfig()
    affine = _affine_values(transform)
    ground = np.asarray(mnt, dtype="float64")
    surface = np.asarray(mns, dtype="float64")
    vegetation = np.asarray(vegetation_mask, dtype=bool)
    exclusion = np.asarray(exclusion_mask, dtype=bool)

    if ground.ndim != 2 or surface.ndim != 2:
        raise ValueError("mnt and mns must be two-dimensional arrays")
    if ground.shape != surface.shape:
        raise ValueError(
            f"MNS/MNT shape mismatch: MNT={ground.shape}, MNS={surface.shape}"
        )
    for name, mask in (("vegetation_mask", vegetation), ("exclusion_mask", exclusion)):
        if mask.ndim != 2 or mask.shape != ground.shape:
            raise ValueError(
                f"{name} shape mismatch: expected {ground.shape}, got {mask.shape}"
            )

    finite = np.isfinite(ground) & np.isfinite(surface)
    canopy_height = np.full(ground.shape, -np.inf, dtype="float64")
    canopy_height[finite] = surface[finite] - ground[finite]
    eligible = (
        vegetation
        & ~exclusion
        & finite
        & (canopy_height >= active_config.min_tree_height_m)
    )

    local_maxima, plateau_pixel_count = _local_maxima(
        canopy_height,
        eligible,
        affine,
        active_config.local_max_radius_m,
    )
    selected, spacing_rejected_count = _select_with_spacing(
        canopy_height,
        local_maxima,
        affine,
        active_config.min_spacing_m,
    )
    uncapped_selected_count = len(selected)
    if (
        active_config.max_proxy_count is not None
        and len(selected) > active_config.max_proxy_count
    ):
        # Selection is already ordered by descending measured height with a
        # row/column tie break, so an opt-in cap remains reproducible.
        selected = selected[: active_config.max_proxy_count]
    budget_rejected_count = uncapped_selected_count - len(selected)

    proxies: list[TreeProxy] = []
    for row, column, x_m, y_m in selected:
        height_m = float(canopy_height[row, column])
        support_count, crown_area_m2, crown_diameter_m, search_limited = _measure_crown(
            row,
            column,
            height_m,
            canopy_height,
            eligible,
            affine,
            active_config,
        )
        ground_elevation_m = float(ground[row, column])
        proxies.append(
            TreeProxy(
                row=row,
                column=column,
                x_m=x_m,
                y_m=y_m,
                ground_elevation_m=ground_elevation_m,
                top_elevation_m=float(surface[row, column]),
                height_m=height_m,
                crown_support_pixel_count=support_count,
                crown_area_m2=crown_area_m2,
                crown_diameter_m=crown_diameter_m,
                crown_search_limited=search_limited,
            )
        )

    valid_non_excluded_vegetation = vegetation & ~exclusion & finite
    statistics: dict[str, Any] = {
        "semantics": "deterministic_lod_representatives_not_exact_tree_inventory",
        "height_measurement": "co_located_mns_minus_mnt",
        "crown_diameter_measurement": (
            "connected_relative_height_support_equivalent_area_diameter"
        ),
        "grid_shape": [int(ground.shape[0]), int(ground.shape[1])],
        "pixel_area_m2": affine.pixel_area_m2,
        "nominal_pixel_size_m": math.sqrt(affine.pixel_area_m2),
        "minimum_tree_height_m": active_config.min_tree_height_m,
        "local_maximum_radius_m": active_config.local_max_radius_m,
        "minimum_spacing_m": active_config.min_spacing_m,
        "maximum_proxy_count": active_config.max_proxy_count,
        "crown_search_radius_m": active_config.resolved_crown_search_radius_m,
        "crown_support_height_ratio": active_config.crown_support_height_ratio,
        "vegetation_pixel_count": int(np.count_nonzero(vegetation)),
        "exclusion_pixel_count": int(np.count_nonzero(exclusion)),
        "excluded_vegetation_pixel_count": int(
            np.count_nonzero(vegetation & exclusion)
        ),
        "invalid_vegetation_pixel_count": int(np.count_nonzero(vegetation & ~finite)),
        "mns_below_mnt_vegetation_pixel_count": int(
            np.count_nonzero(vegetation & finite & (canopy_height < 0))
        ),
        "below_minimum_height_pixel_count": int(
            np.count_nonzero(
                valid_non_excluded_vegetation
                & (canopy_height < active_config.min_tree_height_m)
            )
        ),
        "eligible_pixel_count": int(np.count_nonzero(eligible)),
        "locally_maximal_plateau_pixel_count": plateau_pixel_count,
        "local_maximum_candidate_count": int(np.count_nonzero(local_maxima)),
        "spacing_rejected_candidate_count": spacing_rejected_count,
        "proxy_budget_rejected_candidate_count": budget_rejected_count,
        "selected_proxy_count_before_budget": uncapped_selected_count,
        "selected_proxy_count": len(proxies),
        "crown_search_limited_proxy_count": sum(
            proxy.crown_search_limited for proxy in proxies
        ),
    }
    statistics.update(
        _distribution((proxy.height_m for proxy in proxies), "measured_height_m")
    )
    statistics.update(
        _distribution(
            (proxy.crown_diameter_m for proxy in proxies),
            "measured_crown_diameter_m",
        )
    )
    return proxies, statistics


def _orientation_radians(proxy: TreeProxy, radial_segments: int) -> float:
    """Stable per-pixel phase that avoids a globally aligned polygon pattern."""

    value = (
        ((proxy.row + 1) * 0x9E3779B1) ^ ((proxy.column + 1) * 0x85EBCA77)
    ) & 0xFFFFFFFF
    return (value / 2**32) * (2.0 * math.pi / radial_segments)


def build_tree_proxy_mesh(
    proxies: Sequence[TreeProxy],
    origin: Sequence[float],
    config: VegetationLodConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one compact mesh of volumetric crowns and square trunks.

    Each crown is a low-poly bicone: the measured MNS point is its apex, the
    measured equivalent-area diameter defines its widest ring, and a lower
    crown point sits above the terrain.  An uncapped square trunk connects that
    lower point to the measured MNT.  The representation is intentionally
    coarse but has real volume from every camera direction.
    """

    active_config = config or VegetationLodConfig()
    if len(origin) != 3:
        raise ValueError("origin must contain exactly x, y and z")
    try:
        origin_x, origin_y, origin_z = (float(value) for value in origin)
    except (TypeError, ValueError) as error:
        raise ValueError("origin must contain finite numeric values") from error
    if not all(math.isfinite(value) for value in (origin_x, origin_y, origin_z)):
        raise ValueError("origin must contain finite numeric values")

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    precision = active_config.coordinate_precision
    for proxy in proxies:
        crown_radius_m = proxy.crown_diameter_m * 0.5
        crown_base_elevation_m = (
            proxy.ground_elevation_m
            + proxy.height_m * active_config.crown_base_height_ratio
        )
        widest_elevation_m = (
            proxy.ground_elevation_m
            + proxy.height_m * active_config.crown_widest_height_ratio
        )
        radial_segments = active_config.crown_radial_segments
        phase = _orientation_radians(proxy, radial_segments)

        crown_top_index = len(vertices)
        vertices.append(
            [
                round(proxy.x_m - origin_x, precision),
                round(proxy.y_m - origin_y, precision),
                round(proxy.top_elevation_m - origin_z, precision),
            ]
        )
        crown_ring_indices: list[int] = []
        for segment in range(radial_segments):
            angle = phase + segment * 2.0 * math.pi / radial_segments
            crown_ring_indices.append(len(vertices))
            vertices.append(
                [
                    round(
                        proxy.x_m + math.cos(angle) * crown_radius_m - origin_x,
                        precision,
                    ),
                    round(
                        proxy.y_m + math.sin(angle) * crown_radius_m - origin_y,
                        precision,
                    ),
                    round(widest_elevation_m - origin_z, precision),
                ]
            )
        crown_bottom_index = len(vertices)
        vertices.append(
            [
                round(proxy.x_m - origin_x, precision),
                round(proxy.y_m - origin_y, precision),
                round(crown_base_elevation_m - origin_z, precision),
            ]
        )
        for segment in range(radial_segments):
            current = crown_ring_indices[segment]
            following = crown_ring_indices[(segment + 1) % radial_segments]
            faces.append([crown_top_index, current, following])
            faces.append([crown_bottom_index, following, current])

        trunk_radius_m = min(
            max(
                crown_radius_m * active_config.trunk_radius_to_crown_ratio,
                active_config.minimum_trunk_radius_m,
            ),
            active_config.maximum_trunk_radius_m,
        )
        # A small overlap hides the open trunk top without adding cap faces.
        trunk_top_elevation_m = crown_base_elevation_m + 0.08 * (
            widest_elevation_m - crown_base_elevation_m
        )
        trunk_phase = phase + math.pi * 0.25
        trunk_bottom_indices: list[int] = []
        trunk_top_indices: list[int] = []
        for elevation_m, target_indices in (
            (proxy.ground_elevation_m, trunk_bottom_indices),
            (trunk_top_elevation_m, trunk_top_indices),
        ):
            for segment in range(4):
                angle = trunk_phase + segment * math.pi * 0.5
                target_indices.append(len(vertices))
                vertices.append(
                    [
                        round(
                            proxy.x_m + math.cos(angle) * trunk_radius_m - origin_x,
                            precision,
                        ),
                        round(
                            proxy.y_m + math.sin(angle) * trunk_radius_m - origin_y,
                            precision,
                        ),
                        round(elevation_m - origin_z, precision),
                    ]
                )
        for segment in range(4):
            following = (segment + 1) % 4
            faces.append(
                [
                    trunk_bottom_indices[segment],
                    trunk_bottom_indices[following],
                    trunk_top_indices[following],
                    trunk_top_indices[segment],
                ]
            )

    vertices_per_proxy = active_config.crown_radial_segments + 10
    faces_per_proxy = active_config.crown_radial_segments * 2 + 4
    triangles_after_triangulation_per_proxy = (
        active_config.crown_radial_segments * 2 + 8
    )
    statistics = {
        "proxy_count": len(proxies),
        "primitive": "low_poly_biconic_crown_and_square_trunk",
        "crown_radial_segments": active_config.crown_radial_segments,
        "crown_base_height_ratio": active_config.crown_base_height_ratio,
        "crown_widest_height_ratio": active_config.crown_widest_height_ratio,
        "trunk_radius_to_crown_ratio": active_config.trunk_radius_to_crown_ratio,
        "trunk_radius_range_m": [
            active_config.minimum_trunk_radius_m,
            active_config.maximum_trunk_radius_m,
        ],
        "vertices_per_proxy": vertices_per_proxy,
        "faces_per_proxy": faces_per_proxy,
        "triangles_after_triangulation_per_proxy": (
            triangles_after_triangulation_per_proxy
        ),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "estimated_triangle_count_after_triangulation": len(proxies)
        * triangles_after_triangulation_per_proxy,
        "grounding": "trunk_bottom_at_measured_mnt",
        "top_alignment": "crown_apex_at_measured_mns",
        "diameter_alignment": "widest_crown_ring_uses_measured_equivalent_area_diameter",
        "orientation": "deterministic_row_column_hash",
        "coordinate_precision_decimals": precision,
    }
    return {
        "vertices": vertices,
        "faces": faces,
        "primitive": statistics["primitive"],
    }, statistics


def generate_mid_distance_vegetation_lod(
    mnt: Any,
    mns: Any,
    transform: Any,
    vegetation_mask: Any,
    exclusion_mask: Any,
    origin: Sequence[float],
    config: VegetationLodConfig | None = None,
) -> dict[str, Any]:
    """Run detection and mesh generation, returning a package-ready mapping."""

    active_config = config or VegetationLodConfig()
    proxies, detection_statistics = detect_tree_proxies(
        mnt,
        mns,
        transform,
        vegetation_mask,
        exclusion_mask,
        active_config,
    )
    mesh, mesh_statistics = build_tree_proxy_mesh(proxies, origin, active_config)
    return {
        "schema": LOD_SCHEMA,
        "mesh": mesh,
        "statistics": {
            "detection": detection_statistics,
            "mesh": mesh_statistics,
        },
    }
