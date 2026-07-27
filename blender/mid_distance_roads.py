"""Bounded, terrain-draped road surfaces for the Blender mid-distance view.

The global control-scene pipeline already loads roads as :class:`LineFeature`
instances.  This module turns those centre lines into serialisable surface
meshes without importing ``bpy`` and without buffering the full road network
with Shapely.  Work remains bounded per source segment while the road edges,
shoulders and centre markings are sampled independently on the MNT.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from spatial_data import LineFeature, numeric_property, positive_numeric_property


Mesh = dict[str, list[list[float]] | list[list[int]]]
WidthResolver = Callable[[Mapping[str, Any]], tuple[float, str]]


@dataclass(frozen=True)
class MidDistanceRoadConfig:
    """Geometry budget and visual dimensions, all expressed in metres."""

    max_drape_segment_length_m: float = 20.0
    max_subdivisions_per_source_segment: int = 8
    pavement_offset_m: float = 0.35
    major_shoulder_width_m: float = 1.75
    secondary_shoulder_width_m: float = 1.20
    minor_shoulder_width_m: float = 0.70
    unclassified_shoulder_width_m: float = 0.90
    shoulder_inner_drop_m: float = 0.015
    shoulder_outer_drop_m: float = 0.060
    marking_width_m: float = 0.18
    marking_dash_length_m: float = 3.0
    marking_gap_length_m: float = 7.0
    marking_raise_m: float = 0.018
    marking_importance_max: int = 3
    miter_limit: float = 2.0
    minimum_segment_length_m: float = 0.01
    coordinate_precision: int = 3

    def validate(self) -> None:
        positive_values = {
            "max_drape_segment_length_m": self.max_drape_segment_length_m,
            "major_shoulder_width_m": self.major_shoulder_width_m,
            "secondary_shoulder_width_m": self.secondary_shoulder_width_m,
            "minor_shoulder_width_m": self.minor_shoulder_width_m,
            "unclassified_shoulder_width_m": self.unclassified_shoulder_width_m,
            "marking_width_m": self.marking_width_m,
            "marking_dash_length_m": self.marking_dash_length_m,
            "miter_limit": self.miter_limit,
            "minimum_segment_length_m": self.minimum_segment_length_m,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite, strictly positive value")
        non_negative_values = {
            "pavement_offset_m": self.pavement_offset_m,
            "shoulder_inner_drop_m": self.shoulder_inner_drop_m,
            "shoulder_outer_drop_m": self.shoulder_outer_drop_m,
            "marking_gap_length_m": self.marking_gap_length_m,
            "marking_raise_m": self.marking_raise_m,
        }
        for name, value in non_negative_values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative value")
        if (
            not isinstance(self.max_subdivisions_per_source_segment, int)
            or isinstance(self.max_subdivisions_per_source_segment, bool)
            or self.max_subdivisions_per_source_segment < 1
        ):
            raise ValueError("max_subdivisions_per_source_segment must be at least 1")
        if (
            not isinstance(self.marking_importance_max, int)
            or isinstance(self.marking_importance_max, bool)
            or not 1 <= self.marking_importance_max <= 7
        ):
            raise ValueError("marking_importance_max must be between 1 and 7")
        if (
            not isinstance(self.coordinate_precision, int)
            or isinstance(self.coordinate_precision, bool)
            or not 0 <= self.coordinate_precision <= 9
        ):
            raise ValueError("coordinate_precision must be between 0 and 9")
        if self.miter_limit < 1.0:
            raise ValueError("miter_limit must be at least 1")
        if self.shoulder_inner_drop_m > self.shoulder_outer_drop_m:
            raise ValueError(
                "shoulder_inner_drop_m cannot exceed shoulder_outer_drop_m"
            )
        if self.shoulder_outer_drop_m > self.pavement_offset_m:
            raise ValueError("shoulder_outer_drop_m cannot exceed pavement_offset_m")


def resolve_road_width_m(properties: Mapping[str, Any]) -> tuple[float, str]:
    """Resolve a carriageway width using the existing BD TOPO conventions."""

    source_width = positive_numeric_property(
        properties,
        ("largeur_de_chaussee", "largeur_chaussee", "largeur", "width"),
    )
    if source_width is not None:
        return min(max(source_width, 1.5), 30.0), "source_width"
    importance = road_importance_rank(properties)
    importance_widths = {1: 10.0, 2: 8.0, 3: 7.0, 4: 6.0, 5: 5.0, 6: 2.5, 7: 1.5}
    if importance in importance_widths:
        return importance_widths[importance], "importance"
    return 4.0, "fallback_unclassified"


def road_importance_rank(properties: Mapping[str, Any]) -> int | None:
    """Return a normalised BD TOPO importance rank (1 is most important)."""

    value = numeric_property(properties, ("importance", "classement", "road_class"))
    if value is None:
        return None
    rank = int(round(value))
    return rank if 1 <= rank <= 7 else None


def resolve_shoulder_width_m(
    properties: Mapping[str, Any],
    config: MidDistanceRoadConfig,
) -> tuple[float, str]:
    """Give primary roads wider readable shoulders without widening minor tracks."""

    rank = road_importance_rank(properties)
    if rank is None:
        return config.unclassified_shoulder_width_m, "unclassified"
    if rank <= 2:
        return config.major_shoulder_width_m, "major_importance_1_2"
    if rank <= 4:
        return config.secondary_shoulder_width_m, "secondary_importance_3_4"
    return config.minor_shoulder_width_m, "minor_importance_5_7"


class _TerrainSampler:
    """Bilinear sampling at raster cell centres with auditable fallbacks."""

    def __init__(self, mnt: Any, transform: Any, fallback: float) -> None:
        import numpy as np

        self.values = np.asarray(mnt)
        if self.values.ndim != 2 or not self.values.size:
            raise ValueError("mnt must be a non-empty two-dimensional array")
        if not math.isfinite(float(fallback)):
            raise ValueError("origin Z, used as MNT fallback, must be finite")
        try:
            self.inverse_transform = ~transform
        except Exception as exc:  # affine raises different exceptions across versions
            raise ValueError(
                "transform must be an invertible affine transform"
            ) from exc
        self.fallback = float(fallback)
        self.sample_count = 0
        self.clamped_sample_count = 0
        self.fallback_sample_count = 0

    def sample(self, x: float, y: float) -> float:
        import numpy as np

        self.sample_count += 1
        column_corner, row_corner = self.inverse_transform * (float(x), float(y))
        # Raster values represent cell centres, whereas Affine maps cell corners.
        column = float(column_corner) - 0.5
        row = float(row_corner) - 0.5
        max_column = self.values.shape[1] - 1
        max_row = self.values.shape[0] - 1
        clamped_column = min(max(column, 0.0), float(max_column))
        clamped_row = min(max(row, 0.0), float(max_row))
        if clamped_column != column or clamped_row != row:
            self.clamped_sample_count += 1
        column, row = clamped_column, clamped_row

        column_0 = int(math.floor(column))
        row_0 = int(math.floor(row))
        column_1 = min(column_0 + 1, max_column)
        row_1 = min(row_0 + 1, max_row)
        column_fraction = column - column_0
        row_fraction = row - row_0
        candidates = (
            (row_0, column_0, (1.0 - row_fraction) * (1.0 - column_fraction)),
            (row_0, column_1, (1.0 - row_fraction) * column_fraction),
            (row_1, column_0, row_fraction * (1.0 - column_fraction)),
            (row_1, column_1, row_fraction * column_fraction),
        )
        weighted_value = 0.0
        finite_weight = 0.0
        for row_index, column_index, weight in candidates:
            if weight <= 0:
                continue
            value = float(self.values[row_index, column_index])
            if np.isfinite(value):
                weighted_value += value * weight
                finite_weight += weight
        if finite_weight > 1e-12:
            return weighted_value / finite_weight
        self.fallback_sample_count += 1
        return self.fallback


def _clean_points(
    feature: LineFeature, minimum_length: float
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for coordinate in feature.geometry.coords:
        x, y = float(coordinate[0]), float(coordinate[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return []
        if (
            not points
            or math.hypot(x - points[-1][0], y - points[-1][1]) >= minimum_length
        ):
            points.append((x, y))
    return points


def _densify_points(
    points: Sequence[tuple[float, float]],
    config: MidDistanceRoadConfig,
) -> tuple[list[tuple[float, float]], dict[str, float | int]]:
    densified = [points[0]]
    cap_hit_count = 0
    source_segment_count = 0
    maximum_subdivisions = 1
    maximum_realized_length = 0.0
    length_m = 0.0
    for start, end in zip(points, points[1:]):
        delta_x, delta_y = end[0] - start[0], end[1] - start[1]
        segment_length = math.hypot(delta_x, delta_y)
        if segment_length < config.minimum_segment_length_m:
            continue
        source_segment_count += 1
        length_m += segment_length
        requested = max(
            1, int(math.ceil(segment_length / config.max_drape_segment_length_m))
        )
        subdivisions = min(requested, config.max_subdivisions_per_source_segment)
        if requested > subdivisions:
            cap_hit_count += 1
        maximum_subdivisions = max(maximum_subdivisions, subdivisions)
        maximum_realized_length = max(
            maximum_realized_length, segment_length / subdivisions
        )
        for subdivision in range(1, subdivisions + 1):
            factor = subdivision / subdivisions
            densified.append((start[0] + delta_x * factor, start[1] + delta_y * factor))
    return densified, {
        "source_segment_count": source_segment_count,
        "draped_segment_count": max(len(densified) - 1, 0),
        "subdivision_cap_hit_count": cap_hit_count,
        "maximum_subdivisions_used": maximum_subdivisions,
        "maximum_realized_segment_length_m": maximum_realized_length,
        "length_m": length_m,
    }


def _offset_points(
    points: Sequence[tuple[float, float]],
    offset_m: float,
    miter_limit: float,
) -> list[tuple[float, float]]:
    if abs(offset_m) <= 1e-12:
        return list(points)
    directions: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        delta_x, delta_y = end[0] - start[0], end[1] - start[1]
        length = math.hypot(delta_x, delta_y)
        directions.append((delta_x / length, delta_y / length))

    result: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if index == 0:
            direction = directions[0]
            shift_x, shift_y = -direction[1] * offset_m, direction[0] * offset_m
        elif index == len(points) - 1:
            direction = directions[-1]
            shift_x, shift_y = -direction[1] * offset_m, direction[0] * offset_m
        else:
            previous_normal = (-directions[index - 1][1], directions[index - 1][0])
            following_normal = (-directions[index][1], directions[index][0])
            miter_x = previous_normal[0] + following_normal[0]
            miter_y = previous_normal[1] + following_normal[1]
            miter_length = math.hypot(miter_x, miter_y)
            if miter_length <= 1e-9:
                shift_x = following_normal[0] * offset_m
                shift_y = following_normal[1] * offset_m
            else:
                miter_x, miter_y = miter_x / miter_length, miter_y / miter_length
                denominator = (
                    miter_x * following_normal[0] + miter_y * following_normal[1]
                )
                if abs(denominator) <= 1e-9:
                    scale = offset_m
                else:
                    scale = offset_m / denominator
                maximum_scale = abs(offset_m) * miter_limit
                scale = min(max(scale, -maximum_scale), maximum_scale)
                shift_x, shift_y = miter_x * scale, miter_y * scale
        result.append((point[0] + shift_x, point[1] + shift_y))
    return result


def _empty_mesh() -> Mesh:
    return {"vertices": [], "faces": []}


def _drape_rail(
    points: Sequence[tuple[float, float]],
    sampler: _TerrainSampler,
    origin: tuple[float, float, float],
    vertical_offset_m: float,
    precision: int,
    rail_name: str,
    rail_sample_counts: dict[str, int],
) -> list[list[float]]:
    origin_x, origin_y, origin_z = origin
    rail_sample_counts[rail_name] = rail_sample_counts.get(rail_name, 0) + len(points)
    return [
        [
            round(x - origin_x, precision),
            round(y - origin_y, precision),
            round(sampler.sample(x, y) + vertical_offset_m - origin_z, precision),
        ]
        for x, y in points
    ]


def _append_ribbon(
    mesh: Mesh, left: Sequence[list[float]], right: Sequence[list[float]]
) -> None:
    if len(left) != len(right) or len(left) < 2:
        return
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    start_index = len(vertices)
    for left_vertex, right_vertex in zip(left, right):
        vertices.append(left_vertex)
        vertices.append(right_vertex)
    for index in range(len(left) - 1):
        left_start = start_index + index * 2
        right_start = left_start + 1
        left_end = left_start + 2
        right_end = left_start + 3
        faces.append([left_start, right_start, right_end, left_end])


def _cumulative_lengths(points: Sequence[tuple[float, float]]) -> list[float]:
    values = [0.0]
    for start, end in zip(points, points[1:]):
        values.append(values[-1] + math.hypot(end[0] - start[0], end[1] - start[1]))
    return values


def _point_at_distance(
    points: Sequence[tuple[float, float]],
    cumulative: Sequence[float],
    distance: float,
) -> tuple[float, float]:
    distance = min(max(distance, 0.0), cumulative[-1])
    index = min(max(bisect_left(cumulative, distance) - 1, 0), len(points) - 2)
    span = cumulative[index + 1] - cumulative[index]
    factor = 0.0 if span <= 1e-12 else (distance - cumulative[index]) / span
    start, end = points[index], points[index + 1]
    return (
        start[0] + (end[0] - start[0]) * factor,
        start[1] + (end[1] - start[1]) * factor,
    )


def _slice_points(
    points: Sequence[tuple[float, float]],
    cumulative: Sequence[float],
    start_distance: float,
    end_distance: float,
) -> list[tuple[float, float]]:
    sliced = [_point_at_distance(points, cumulative, start_distance)]
    first_internal = max(bisect_right(cumulative, start_distance + 1e-9), 1)
    last_internal = min(bisect_left(cumulative, end_distance - 1e-9), len(points) - 1)
    sliced.extend(points[first_internal:last_internal])
    end_point = _point_at_distance(points, cumulative, end_distance)
    if math.hypot(end_point[0] - sliced[-1][0], end_point[1] - sliced[-1][1]) > 1e-9:
        sliced.append(end_point)
    return sliced


def _entity_id(feature_id: str) -> str:
    return feature_id.split(":", 1)[0]


def _mesh_statistics(mesh: Mesh) -> dict[str, int]:
    return {"vertex_count": len(mesh["vertices"]), "face_count": len(mesh["faces"])}


def build_mid_distance_road_geometry(
    features: Sequence[LineFeature],
    mnt: Any,
    transform: Any,
    origin: tuple[float, float, float],
    *,
    config: MidDistanceRoadConfig | None = None,
    width_resolver: WidthResolver = resolve_road_width_m,
) -> tuple[dict[str, Mesh], dict[str, Any]]:
    """Build bounded road, shoulder and marking meshes draped on the MNT.

    The returned dictionary can be written directly into the preview package;
    every member contains only JSON-compatible ``vertices`` and ``faces``.
    The four meshes are intentionally separate so Blender can assign distinct
    pavement, shoulder and road-marking materials without curve bevels.
    """

    active_config = config or MidDistanceRoadConfig()
    active_config.validate()
    if len(origin) != 3 or not all(math.isfinite(float(value)) for value in origin):
        raise ValueError("origin must contain three finite coordinates")
    sampler = _TerrainSampler(mnt, transform, float(origin[2]))
    meshes = {
        "carriageway": _empty_mesh(),
        "left_shoulders": _empty_mesh(),
        "right_shoulders": _empty_mesh(),
        "center_markings": _empty_mesh(),
    }

    rail_sample_counts: dict[str, int] = {}
    width_method_counts: dict[str, int] = {}
    shoulder_method_counts: dict[str, int] = {}
    importance_counts: dict[str, int] = {}
    rendered_entities: set[str] = set()
    widths: list[float] = []
    shoulder_widths: list[float] = []
    skipped_degenerate = 0
    rendered_line_count = 0
    source_vertex_count = 0
    rendered_source_vertex_count = 0
    draped_station_count = 0
    source_segment_count = 0
    draped_segment_count = 0
    subdivision_cap_hit_count = 0
    maximum_subdivisions_used = 1
    maximum_realized_segment_length = 0.0
    total_length_m = 0.0
    marking_eligible_line_count = 0
    marking_dash_count = 0
    marking_length_m = 0.0

    for feature in features:
        points = _clean_points(feature, active_config.minimum_segment_length_m)
        source_vertex_count += len(points)
        if len(points) < 2:
            skipped_degenerate += 1
            continue
        densified, densification = _densify_points(points, active_config)
        if len(densified) < 2:
            skipped_degenerate += 1
            continue
        rendered_source_vertex_count += len(points)

        width, width_method = width_resolver(feature.properties)
        width = float(width)
        if not math.isfinite(width) or width <= 0:
            raise ValueError(
                f"width resolver returned an invalid width for {feature.feature_id!r}"
            )
        shoulder_width, shoulder_method = resolve_shoulder_width_m(
            feature.properties, active_config
        )
        rank = road_importance_rank(feature.properties)
        importance_key = str(rank) if rank is not None else "unclassified"
        width_method_counts[width_method] = width_method_counts.get(width_method, 0) + 1
        shoulder_method_counts[shoulder_method] = (
            shoulder_method_counts.get(shoulder_method, 0) + 1
        )
        importance_counts[importance_key] = importance_counts.get(importance_key, 0) + 1
        widths.append(width)
        shoulder_widths.append(shoulder_width)

        source_segment_count += int(densification["source_segment_count"])
        draped_segment_count += int(densification["draped_segment_count"])
        subdivision_cap_hit_count += int(densification["subdivision_cap_hit_count"])
        maximum_subdivisions_used = max(
            maximum_subdivisions_used, int(densification["maximum_subdivisions_used"])
        )
        maximum_realized_segment_length = max(
            maximum_realized_segment_length,
            float(densification["maximum_realized_segment_length_m"]),
        )
        length_m = float(densification["length_m"])
        total_length_m += length_m
        draped_station_count += len(densified)

        half_width = width / 2.0
        carriage_left_xy = _offset_points(
            densified, half_width, active_config.miter_limit
        )
        carriage_right_xy = _offset_points(
            densified, -half_width, active_config.miter_limit
        )
        outer_left_xy = _offset_points(
            densified, half_width + shoulder_width, active_config.miter_limit
        )
        outer_right_xy = _offset_points(
            densified, -half_width - shoulder_width, active_config.miter_limit
        )

        carriage_left = _drape_rail(
            carriage_left_xy,
            sampler,
            origin,
            active_config.pavement_offset_m,
            active_config.coordinate_precision,
            "carriageway_left_edge",
            rail_sample_counts,
        )
        carriage_right = _drape_rail(
            carriage_right_xy,
            sampler,
            origin,
            active_config.pavement_offset_m,
            active_config.coordinate_precision,
            "carriageway_right_edge",
            rail_sample_counts,
        )
        _append_ribbon(meshes["carriageway"], carriage_left, carriage_right)

        left_inner = _drape_rail(
            carriage_left_xy,
            sampler,
            origin,
            active_config.pavement_offset_m - active_config.shoulder_inner_drop_m,
            active_config.coordinate_precision,
            "left_shoulder_inner_edge",
            rail_sample_counts,
        )
        left_outer = _drape_rail(
            outer_left_xy,
            sampler,
            origin,
            active_config.pavement_offset_m - active_config.shoulder_outer_drop_m,
            active_config.coordinate_precision,
            "left_shoulder_outer_edge",
            rail_sample_counts,
        )
        _append_ribbon(meshes["left_shoulders"], left_outer, left_inner)

        right_inner = _drape_rail(
            carriage_right_xy,
            sampler,
            origin,
            active_config.pavement_offset_m - active_config.shoulder_inner_drop_m,
            active_config.coordinate_precision,
            "right_shoulder_inner_edge",
            rail_sample_counts,
        )
        right_outer = _drape_rail(
            outer_right_xy,
            sampler,
            origin,
            active_config.pavement_offset_m - active_config.shoulder_outer_drop_m,
            active_config.coordinate_precision,
            "right_shoulder_outer_edge",
            rail_sample_counts,
        )
        _append_ribbon(meshes["right_shoulders"], right_inner, right_outer)

        if rank is not None and rank <= active_config.marking_importance_max:
            marking_eligible_line_count += 1
            cumulative = _cumulative_lengths(densified)
            period = (
                active_config.marking_dash_length_m + active_config.marking_gap_length_m
            )
            dash_start = 0.0
            while dash_start < cumulative[-1] - active_config.minimum_segment_length_m:
                dash_end = min(
                    dash_start + active_config.marking_dash_length_m, cumulative[-1]
                )
                if dash_end - dash_start >= active_config.minimum_segment_length_m:
                    dash_points = _slice_points(
                        densified, cumulative, dash_start, dash_end
                    )
                    if len(dash_points) >= 2:
                        marking_left_xy = _offset_points(
                            dash_points,
                            active_config.marking_width_m / 2.0,
                            active_config.miter_limit,
                        )
                        marking_right_xy = _offset_points(
                            dash_points,
                            -active_config.marking_width_m / 2.0,
                            active_config.miter_limit,
                        )
                        marking_offset = (
                            active_config.pavement_offset_m
                            + active_config.marking_raise_m
                        )
                        marking_left = _drape_rail(
                            marking_left_xy,
                            sampler,
                            origin,
                            marking_offset,
                            active_config.coordinate_precision,
                            "center_marking_left_edge",
                            rail_sample_counts,
                        )
                        marking_right = _drape_rail(
                            marking_right_xy,
                            sampler,
                            origin,
                            marking_offset,
                            active_config.coordinate_precision,
                            "center_marking_right_edge",
                            rail_sample_counts,
                        )
                        _append_ribbon(
                            meshes["center_markings"], marking_left, marking_right
                        )
                        marking_dash_count += 1
                        marking_length_m += dash_end - dash_start
                dash_start += period

        rendered_line_count += 1
        rendered_entities.add(_entity_id(feature.feature_id))

    mesh_statistics = {name: _mesh_statistics(mesh) for name, mesh in meshes.items()}
    added_station_count = max(draped_station_count - rendered_source_vertex_count, 0)
    statistics: dict[str, Any] = {
        "geometry_mode": "bounded_densified_surface_meshes",
        "altitude_method": "bilinear_mnt_independent_lateral_rails",
        "input_line_count": len(features),
        "input_entity_count": len(
            {_entity_id(feature.feature_id) for feature in features}
        ),
        "rendered_line_count": rendered_line_count,
        "rendered_entity_count": len(rendered_entities),
        "skipped_degenerate_line_count": skipped_degenerate,
        "source_vertex_count": source_vertex_count,
        "rendered_source_vertex_count": rendered_source_vertex_count,
        "source_segment_count": source_segment_count,
        "draped_station_count": draped_station_count,
        "added_drape_station_count": added_station_count,
        "draped_segment_count": draped_segment_count,
        "bounded_segment_expansion_ratio": round(
            draped_segment_count / source_segment_count, 3
        )
        if source_segment_count
        else None,
        "maximum_drape_segment_length_target_m": active_config.max_drape_segment_length_m,
        "maximum_subdivisions_per_source_segment": (
            active_config.max_subdivisions_per_source_segment
        ),
        "maximum_subdivisions_used": maximum_subdivisions_used
        if source_segment_count
        else 0,
        "maximum_realized_segment_length_m": round(maximum_realized_segment_length, 3)
        if source_segment_count
        else None,
        "subdivision_cap_hit_count": subdivision_cap_hit_count,
        "total_rendered_length_m": round(total_length_m, 3),
        "minimum_carriageway_width_m": min(widths) if widths else None,
        "maximum_carriageway_width_m": max(widths) if widths else None,
        "minimum_shoulder_width_m": min(shoulder_widths) if shoulder_widths else None,
        "maximum_shoulder_width_m": max(shoulder_widths) if shoulder_widths else None,
        "width_method_counts": dict(sorted(width_method_counts.items())),
        "shoulder_method_counts": dict(sorted(shoulder_method_counts.items())),
        "importance_rank_counts": dict(sorted(importance_counts.items())),
        "marking_importance_max": active_config.marking_importance_max,
        "marking_eligible_line_count": marking_eligible_line_count,
        "center_marking_dash_count": marking_dash_count,
        "center_marking_length_m": round(marking_length_m, 3),
        "terrain_sample_count": sampler.sample_count,
        "terrain_clamped_sample_count": sampler.clamped_sample_count,
        "terrain_fallback_sample_count": sampler.fallback_sample_count,
        "draped_rail_sample_counts": dict(sorted(rail_sample_counts.items())),
        "mesh_statistics": mesh_statistics,
    }
    return meshes, statistics


__all__ = [
    "MidDistanceRoadConfig",
    "build_mid_distance_road_geometry",
    "resolve_road_width_m",
    "resolve_shoulder_width_m",
    "road_importance_rank",
]
