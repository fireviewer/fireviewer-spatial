"""Deterministic terrain, fuel, and weather-driven wildfire spread model.

The model is deliberately scoped as synthetic data generation.  It solves a
least-arrival-time propagation over a grid made from the uploaded elevation
tiles and the uploaded vegetation instances.  It is therefore continuous in
time and geographically tied to the source map, but it is *not* an incident
reconstruction or a fire-behaviour model calibrated for operational use.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class FireSpreadDrivers:
    """Explicit, inspectable synthetic drivers for one propagation scenario."""

    wind_to_degrees_from_east: float
    wind_speed_m_s: float = 6.0
    air_temperature_c: float = 28.0
    relative_humidity_percent: float = 32.0
    fine_fuel_moisture_percent: float = 8.0
    base_spread_rate_m_s: float = 0.10
    maximum_duration_s: float = 7200.0
    cell_size_m: float = 25.0
    domain_radius_m: float = 1800.0

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class FireFrontSegment:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    spread_rate_m_s: float


@dataclass
class FireSpreadState:
    state_index: int
    elapsed_s: float
    burned_mask: np.ndarray
    front_segments: list[FireFrontSegment]
    burned_tree_ids: set[int]
    burned_area_m2: float
    active_front_length_m: float
    mean_front_spread_rate_m_s: float


@dataclass
class FireSpreadResult:
    simulation_id: str
    model_metadata: dict[str, Any]
    domain_bounds_l93_m: tuple[float, float, float, float]
    ignition_l93_m: tuple[float, float]
    drivers: FireSpreadDrivers
    elevation_m: np.ndarray
    fuel_load: np.ndarray
    burnable_mask: np.ndarray
    arrival_time_s: np.ndarray
    spread_rate_m_s: np.ndarray
    cell_tree_ids: list[list[list[int]]]
    states: list[FireSpreadState]


def _smooth_stand_fuel(values: np.ndarray, radius: int = 6) -> np.ndarray:
    """Aggregate tree points into a local stand-fuel field (default 325 m)."""

    padded = np.pad(values, radius, mode="edge")
    result = np.zeros_like(values, dtype=np.float64)
    width = radius * 2 + 1
    for y_offset in range(width):
        for x_offset in range(width):
            result += padded[y_offset : y_offset + values.shape[0], x_offset : x_offset + values.shape[1]]
    return result / float(width * width)


def _stable_drivers(package_id: str) -> FireSpreadDrivers:
    digest = hashlib.sha256(package_id.encode("utf-8")).digest()
    wind_to = float(int.from_bytes(digest[:2], "little") % 360)
    wind_speed = 5.0 + float(digest[2] % 31) / 10.0
    return FireSpreadDrivers(wind_to_degrees_from_east=wind_to, wind_speed_m_s=wind_speed)


def _tree_position(tree: Any, anchor_l93_m: tuple[float, float]) -> tuple[float, float]:
    position = getattr(tree, "position")
    return float(position[0]) + anchor_l93_m[0], float(position[1]) + anchor_l93_m[1]


def _grid_bounds(
    tree_positions: np.ndarray,
    source_bounds_l93_m: tuple[float, float, float, float],
    radius_m: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
    if len(tree_positions) == 0:
        raise ValueError("The uploaded map has no vegetation instances for the fire-fuel model")
    center = np.mean(tree_positions, axis=0)
    xmin, ymin, xmax, ymax = source_bounds_l93_m
    radius = min(radius_m, (xmax - xmin) * 0.42, (ymax - ymin) * 0.42)
    ignition_east = min(xmax - radius, max(xmin + radius, float(center[0])))
    ignition_north = min(ymax - radius, max(ymin + radius, float(center[1])))
    return (
        (ignition_east - radius, ignition_north - radius, ignition_east + radius, ignition_north + radius),
        (ignition_east, ignition_north),
    )


def _sample_elevation(
    bounds: tuple[float, float, float, float],
    cell_size_m: float,
    height_at: Callable[[float, float], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = bounds
    x_centers = np.arange(xmin + cell_size_m / 2.0, xmax, cell_size_m, dtype=np.float64)
    y_centers = np.arange(ymin + cell_size_m / 2.0, ymax, cell_size_m, dtype=np.float64)
    elevation = np.empty((len(y_centers), len(x_centers)), dtype=np.float64)
    for y_index, north in enumerate(y_centers):
        for x_index, east in enumerate(x_centers):
            elevation[y_index, x_index] = height_at(float(east), float(north))
    return x_centers, y_centers, elevation


def _fuel_grid(
    trees: Iterable[Any],
    anchor_l93_m: tuple[float, float],
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[list[list[int]]]]:
    ymin, xmin = shape[0], shape[1]
    west, south, east, north = bounds
    density = np.zeros(shape, dtype=np.float64)
    cell_tree_ids: list[list[list[int]]] = [[[] for _ in range(xmin)] for _ in range(ymin)]
    cell_width = (east - west) / xmin
    cell_height = (north - south) / ymin
    for tree in trees:
        tree_east, tree_north = _tree_position(tree, anchor_l93_m)
        x_index = int((tree_east - west) / cell_width)
        y_index = int((tree_north - south) / cell_height)
        if 0 <= x_index < xmin and 0 <= y_index < ymin:
            density[y_index, x_index] += 1.0
            cell_tree_ids[y_index][x_index].append(int(getattr(tree, "tree_id")))
    positive = density[density > 0.0]
    if len(positive) == 0:
        raise ValueError("No uploaded vegetation instances fall within the fire-simulation domain")
    smoothed = _smooth_stand_fuel(density)
    normalization = max(1.0, float(np.percentile(smoothed[smoothed > 0.0], 80.0)))
    return np.clip(smoothed / normalization, 0.0, 1.0), cell_tree_ids


def _arrival_times(
    elevation: np.ndarray,
    fuel_load: np.ndarray,
    ignition_index: tuple[int, int],
    drivers: FireSpreadDrivers,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = elevation.shape
    density_threshold = max(0.012, float(np.percentile(fuel_load[fuel_load > 0.0], 7.0)))
    burnable = fuel_load >= density_threshold
    ignition_y, ignition_x = ignition_index
    if not burnable[ignition_y, ignition_x]:
        burnable[ignition_y, ignition_x] = True
    arrival = np.full((height, width), np.inf, dtype=np.float64)
    local_rate = np.zeros((height, width), dtype=np.float64)
    arrival[ignition_y, ignition_x] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, ignition_y, ignition_x)]
    wind_angle = math.radians(drivers.wind_to_degrees_from_east)
    wind = np.asarray((math.cos(wind_angle), math.sin(wind_angle)), dtype=np.float64)
    moisture_factor = max(0.32, min(1.0, 1.22 - drivers.fine_fuel_moisture_percent / 25.0 - drivers.relative_humidity_percent / 280.0))
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        elapsed, current_y, current_x = heapq.heappop(queue)
        if elapsed != arrival[current_y, current_x] or elapsed > drivers.maximum_duration_s:
            continue
        for y_offset, x_offset in directions:
            next_y, next_x = current_y + y_offset, current_x + x_offset
            if not (0 <= next_y < height and 0 <= next_x < width) or not burnable[next_y, next_x]:
                continue
            horizontal_distance = drivers.cell_size_m * math.hypot(x_offset, y_offset)
            direction = np.asarray((float(x_offset), float(y_offset)), dtype=np.float64)
            direction /= float(np.linalg.norm(direction))
            wind_alignment = float(np.dot(direction, wind))
            upslope = (elevation[next_y, next_x] - elevation[current_y, current_x]) / horizontal_distance
            fuel_factor = 0.22 + 0.78 * float(fuel_load[next_y, next_x])
            wind_factor = math.exp(0.09 * drivers.wind_speed_m_s * wind_alignment)
            slope_factor = math.exp(max(-0.65, min(0.65, 3.8 * upslope)))
            rate = max(0.015, min(0.42, drivers.base_spread_rate_m_s * fuel_factor * wind_factor * slope_factor * moisture_factor))
            candidate = elapsed + horizontal_distance / rate
            if candidate < arrival[next_y, next_x] and candidate <= drivers.maximum_duration_s:
                arrival[next_y, next_x] = candidate
                local_rate[next_y, next_x] = rate
                heapq.heappush(queue, (candidate, next_y, next_x))
    return arrival, local_rate, burnable


def _front_segments(
    burned: np.ndarray,
    arrival: np.ndarray,
    rates: np.ndarray,
    elevation: np.ndarray,
    bounds: tuple[float, float, float, float],
    cell_size_m: float,
) -> list[FireFrontSegment]:
    west, south, _, _ = bounds
    height, width = burned.shape
    segments: list[FireFrontSegment] = []
    sides = (
        (-1, 0, lambda x0, x1, y0, y1: ((x0, y0), (x1, y0))),
        (0, 1, lambda x0, x1, y0, y1: ((x1, y0), (x1, y1))),
        (1, 0, lambda x0, x1, y0, y1: ((x1, y1), (x0, y1))),
        (0, -1, lambda x0, x1, y0, y1: ((x0, y1), (x0, y0))),
    )
    for y_index, x_index in np.argwhere(burned):
        x0 = west + x_index * cell_size_m
        x1 = x0 + cell_size_m
        y0 = south + y_index * cell_size_m
        y1 = y0 + cell_size_m
        z = float(elevation[y_index, x_index])
        for y_offset, x_offset, edge in sides:
            neighbor_y, neighbor_x = y_index + y_offset, x_index + x_offset
            neighbor_burned = 0 <= neighbor_y < height and 0 <= neighbor_x < width and burned[neighbor_y, neighbor_x]
            if neighbor_burned:
                continue
            (start_x, start_y), (end_x, end_y) = edge(x0, x1, y0, y1)
            segments.append(FireFrontSegment(
                start=(float(start_x), float(start_y), z),
                end=(float(end_x), float(end_y), z),
                spread_rate_m_s=float(rates[y_index, x_index]),
            ))
    return segments


def _state_times(arrival: np.ndarray, state_count: int) -> list[float]:
    finite = np.unique(arrival[np.isfinite(arrival)])
    if len(finite) < state_count:
        raise ValueError("The terrain and vegetation inputs did not yield a viable continuous fire front")
    upper_index = max(state_count - 1, int(math.floor((len(finite) - 1) * 0.96)))
    indices = np.rint(np.linspace(0, upper_index, state_count)).astype(np.int64)
    if len(np.unique(indices)) != state_count:
        raise ValueError("The propagation field cannot provide one visibly distinct fire state per observation")
    values = [float(finite[index]) for index in indices]
    return [max(1.0, value) for value in values]


def simulate_fire_spread(
    *,
    package_id: str,
    trees: list[Any],
    anchor_l93_m: tuple[float, float],
    source_bounds_l93_m: tuple[float, float, float, float],
    height_at: Callable[[float, float], float],
    state_count: int = 10,
) -> FireSpreadResult:
    """Compute a single continuous spread field and derive timeline snapshots."""

    if state_count < 2:
        raise ValueError("A continuous fire timeline requires at least two states")
    drivers = _stable_drivers(package_id)
    tree_positions = np.asarray([_tree_position(tree, anchor_l93_m) for tree in trees], dtype=np.float64)
    bounds, weighted_center = _grid_bounds(tree_positions, source_bounds_l93_m, drivers.domain_radius_m)
    x_centers, y_centers, elevation = _sample_elevation(bounds, drivers.cell_size_m, height_at)
    fuel_load, cell_tree_ids = _fuel_grid(trees, anchor_l93_m, bounds, elevation.shape)
    center_x = int(np.clip(round((weighted_center[0] - bounds[0]) / drivers.cell_size_m - 0.5), 0, fuel_load.shape[1] - 1))
    center_y = int(np.clip(round((weighted_center[1] - bounds[1]) / drivers.cell_size_m - 0.5), 0, fuel_load.shape[0] - 1))
    grid_y, grid_x = np.indices(fuel_load.shape)
    distance_from_weighted_center = np.hypot(grid_x - center_x, grid_y - center_y) * drivers.cell_size_m
    ignition_score = fuel_load * np.exp(-0.5 * (distance_from_weighted_center / 700.0) ** 2)
    ignition_y, ignition_x = np.unravel_index(int(np.argmax(ignition_score)), fuel_load.shape)
    ignition = (float(x_centers[ignition_x]), float(y_centers[ignition_y]))
    arrival, rates, burnable = _arrival_times(elevation, fuel_load, (ignition_y, ignition_x), drivers)
    states: list[FireSpreadState] = []
    for state_index, elapsed_s in enumerate(_state_times(arrival, state_count), start=1):
        burned = np.isfinite(arrival) & (arrival <= elapsed_s)
        segments = _front_segments(burned, arrival, rates, elevation, bounds, drivers.cell_size_m)
        burned_tree_ids: set[int] = set()
        for y_index, x_index in np.argwhere(burned):
            burned_tree_ids.update(cell_tree_ids[y_index][x_index])
        active_rates = [segment.spread_rate_m_s for segment in segments if segment.spread_rate_m_s > 0.0]
        states.append(FireSpreadState(
            state_index=state_index,
            elapsed_s=elapsed_s,
            burned_mask=burned,
            front_segments=segments,
            burned_tree_ids=burned_tree_ids,
            burned_area_m2=float(np.count_nonzero(burned) * drivers.cell_size_m * drivers.cell_size_m),
            active_front_length_m=float(len(segments) * drivers.cell_size_m),
            mean_front_spread_rate_m_s=float(np.mean(active_rates)) if active_rates else 0.0,
        ))
    model_metadata = {
        "schema": "fireviewer.synthetic-fire-spread.v2",
        "model": "least_arrival_time_25m_grid_with_fuel_wind_slope_and_distinct_state_fronts",
        "truth_scope": "synthetic_physically_driven_fire_spread_on_real_uploaded_map_not_incident_reconstruction",
        "terrain_input": "uploaded_elevation_cog_tiles",
        "fuel_input": "uploaded_tree_instances_density",
        "weather_input": "explicit_synthetic_driver_not_observed_incident_weather",
        "solver": "dijkstra_least_arrival_time_on_8_neighbor_grid",
        "weighted_vegetation_center_l93_m": [float(weighted_center[0]), float(weighted_center[1])],
    }
    identity_input = {
        "package_id": package_id,
        "bounds": bounds,
        "ignition": ignition,
        "drivers": drivers.as_dict(),
        "tree_count": len(trees),
        "model": model_metadata["model"],
    }
    simulation_id = hashlib.sha256(repr(identity_input).encode("utf-8")).hexdigest()[:24]
    return FireSpreadResult(
        simulation_id=simulation_id,
        model_metadata=model_metadata,
        domain_bounds_l93_m=bounds,
        ignition_l93_m=ignition,
        drivers=drivers,
        elevation_m=elevation,
        fuel_load=fuel_load,
        burnable_mask=burnable,
        arrival_time_s=arrival,
        spread_rate_m_s=rates,
        cell_tree_ids=cell_tree_ids,
        states=states,
    )
