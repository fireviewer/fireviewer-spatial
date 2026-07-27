from __future__ import annotations

import numpy as np
from affine import Affine

from prepare_mid_vegetation_05m import (
    SegmentationConfig,
    _bounds_aligned_to_native_grid,
    _crop_grid_to_bounds,
    _mosaic,
    _owned_instances,
    _tree_records,
    build_local_terrain_mesh,
    segment_vegetation_instances,
)
from tree_instances import build_tree_instance_set, decode_instance_attributes


def _synthetic_pair() -> tuple[np.ndarray, np.ndarray, Affine]:
    ground = np.full((31, 31), 100.0)
    rows, columns = np.indices(ground.shape)
    first = 12.0 * np.exp(-((rows - 10) ** 2 + (columns - 9) ** 2) / 10.0)
    second = 8.0 * np.exp(-((rows - 21) ** 2 + (columns - 22) ** 2) / 8.0)
    surface = ground + np.maximum(first, second)
    transform = Affine.translation(700_000.0, 6_600_015.5) * Affine.scale(0.5, -0.5)
    return ground, surface, transform


def test_detects_both_05m_crowns_without_spacing_rejection() -> None:
    ground, surface, transform = _synthetic_pair()
    instances, statistics = segment_vegetation_instances(
        ground,
        surface,
        transform,
        np.ones(ground.shape, dtype=bool),
        np.zeros(ground.shape, dtype=bool),
        (700_000.0, 6_600_000.0, 100.0),
        SegmentationConfig(
            min_tree_height_m=2.0,
            local_peak_radius_m=1.0,
            smoothing_sigma_m=0.5,
        ),
    )
    assert len(instances) == 2
    assert statistics["post_detection_spacing_rejected_count"] == 0
    assert sorted(instance[3] for instance in instances) == [8.0, 12.0]
    assert all(instance[4] >= 1.0 for instance in instances)


def test_exclusion_removes_only_the_covered_crown() -> None:
    ground, surface, transform = _synthetic_pair()
    exclusion = np.zeros(ground.shape, dtype=bool)
    exclusion[6:15, 5:14] = True
    instances, statistics = segment_vegetation_instances(
        ground,
        surface,
        transform,
        np.ones(ground.shape, dtype=bool),
        exclusion,
        (700_000.0, 6_600_000.0, 100.0),
    )
    assert len(instances) == 1
    assert instances[0][3] == 8.0
    assert statistics["excluded_pixel_count"] == int(np.count_nonzero(exclusion))


def test_plateau_resolution_is_deterministic() -> None:
    ground = np.full((9, 9), 50.0)
    surface = ground.copy()
    surface[3:5, 3:5] += 7.0
    transform = Affine.translation(0.0, 4.5) * Affine.scale(0.5, -0.5)
    arguments = (
        ground,
        surface,
        transform,
        np.ones(ground.shape, dtype=bool),
        np.zeros(ground.shape, dtype=bool),
        (0.0, 0.0, 50.0),
        SegmentationConfig(smoothing_sigma_m=0.01),
    )
    first, first_statistics = segment_vegetation_instances(*arguments)
    second, second_statistics = segment_vegetation_instances(*arguments)
    assert first == second
    assert first_statistics == second_statistics
    assert len(first) == 1


def test_local_terrain_uses_unaltered_mnt_samples_and_requested_spacing() -> None:
    ground = np.arange(25, dtype="float64").reshape(5, 5) + 100.0
    transform = Affine.translation(700_000.0, 6_600_002.5) * Affine.scale(0.5, -0.5)
    mesh = build_local_terrain_mesh(
        ground,
        transform,
        (700_000.0, 6_600_000.0, 100.0),
        step_pixels=2,
    )
    assert mesh["source_pixel_size_m"] == [0.5, 0.5]
    assert mesh["sample_spacing_m"] == [1.0, 1.0]
    assert mesh["vertex_count"] == 9
    assert mesh["face_count"] == 4
    assert mesh["vertices"][0] == [0.25, 2.25, 0.0]
    assert mesh["vertices"][-1] == [2.25, 0.25, 24.0]


def test_bounded_local_terrain_reaches_exact_core_edges_from_halo() -> None:
    transform = Affine.translation(-1.0, 6.0) * Affine.scale(0.5, -0.5)
    rows, columns = np.indices((14, 14))
    world_x = -1.0 + (columns + 0.5) * 0.5
    world_y = 6.0 - (rows + 0.5) * 0.5
    ground = 100.0 + 2.0 * world_x + 3.0 * world_y
    mesh = build_local_terrain_mesh(
        ground,
        transform,
        (0.0, 0.0, 100.0),
        step_pixels=2,
        bounds=(0.0, 0.0, 5.0, 5.0),
    )

    assert mesh["geometric_bounds_l93_m"] == [0.0, 0.0, 5.0, 5.0]
    assert mesh["vertex_count"] == 36
    assert mesh["face_count"] == 25
    assert min(vertex[0] for vertex in mesh["vertices"]) == 0.0
    assert max(vertex[0] for vertex in mesh["vertices"]) == 5.0
    assert min(vertex[1] for vertex in mesh["vertices"]) == 0.0
    assert max(vertex[1] for vertex in mesh["vertices"]) == 5.0
    assert mesh["vertices"][0] == [0.0, 5.0, 15.0]
    assert mesh["vertices"][-1] == [5.0, 0.0, 10.0]
    assert mesh["boundary_sampling"] == (
        "bilinear_processing_halo_at_exact_lambert93_core_coordinates"
    )


def test_adjacent_bounded_terrain_tiles_share_identical_edge_altitudes() -> None:
    transform = Affine.translation(-1.0, 6.0) * Affine.scale(0.5, -0.5)
    rows, columns = np.indices((14, 24))
    world_x = -1.0 + (columns + 0.5) * 0.5
    world_y = 6.0 - (rows + 0.5) * 0.5
    ground = 250.0 + 0.2 * world_x**2 + 0.3 * world_y**2
    origin = (0.0, 0.0, 250.0)

    left = build_local_terrain_mesh(
        ground,
        transform,
        origin,
        step_pixels=2,
        bounds=(0.0, 0.0, 5.0, 5.0),
    )
    right = build_local_terrain_mesh(
        ground,
        transform,
        origin,
        step_pixels=2,
        bounds=(5.0, 0.0, 10.0, 5.0),
    )

    left_edge = sorted(
        (vertex[1], vertex[2]) for vertex in left["vertices"] if vertex[0] == 5.0
    )
    right_edge = sorted(
        (vertex[1], vertex[2]) for vertex in right["vertices"] if vertex[0] == 5.0
    )
    assert left_edge == right_edge
    assert len(left_edge) == 6


def test_bounded_local_terrain_requires_processing_halo() -> None:
    transform = Affine.translation(0.0, 5.0) * Affine.scale(0.5, -0.5)
    with np.testing.assert_raises_regex(ValueError, "processing-raster halo"):
        build_local_terrain_mesh(
            np.full((10, 10), 100.0),
            transform,
            (0.0, 0.0, 100.0),
            step_pixels=2,
            bounds=(0.0, 0.0, 5.0, 5.0),
        )


def test_native_grid_alignment_expands_integer_bounds_to_ign_pixel_phase() -> None:
    transform = Affine.translation(887_999.75, 6_401_000.25) * Affine.scale(0.5, -0.5)
    assert _bounds_aligned_to_native_grid(
        (888_000.0, 6_400_000.0, 888_500.0, 6_400_500.0), transform
    ) == (887_999.75, 6_399_999.75, 888_500.25, 6_400_500.25)


def test_adjacent_mosaics_keep_one_native_phase_and_identical_mesh_edge(
    tmp_path,
) -> None:
    import rasterio

    transform = Affine.translation(-0.25, 5.25) * Affine.scale(0.5, -0.5)
    rows, columns = np.indices((12, 24))
    source_values = (100.0 + 0.1 * columns**2 + 0.2 * rows**2).astype("float32")
    source = tmp_path / "native-quarter-phase.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=source_values.shape[1],
        height=source_values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:2154",
        transform=transform,
    ) as dataset:
        dataset.write(source_values, 1)

    left_values, left_transform, _ = _mosaic([source], (0.0, 0.0, 5.0, 5.0))
    right_values, right_transform, _ = _mosaic([source], (5.0, 0.0, 10.0, 5.0))
    left = build_local_terrain_mesh(
        left_values,
        left_transform,
        (0.0, 0.0, 100.0),
        step_pixels=2,
        bounds=(0.0, 0.0, 5.0, 5.0),
    )
    right = build_local_terrain_mesh(
        right_values,
        right_transform,
        (0.0, 0.0, 100.0),
        step_pixels=2,
        bounds=(5.0, 0.0, 10.0, 5.0),
    )
    left_edge = sorted(
        (vertex[1], vertex[2]) for vertex in left["vertices"] if vertex[0] == 5.0
    )
    right_edge = sorted(
        (vertex[1], vertex[2]) for vertex in right["vertices"] if vertex[0] == 5.0
    )
    assert left_transform.c == -0.25
    assert right_transform.c == 4.75
    assert left_edge == right_edge


def test_core_ownership_is_half_open_and_discards_halo_apices() -> None:
    origin = (700_000.0, 6_600_000.0, 100.0)
    rows = [
        [-0.5, 10.0, 0.0, 5.0, 2.0, 0, 0.0],
        [0.0, 0.0, 0.0, 6.0, 2.0, 0, 0.0],
        [499.5, 499.5, 0.0, 7.0, 2.0, 0, 0.0],
        [500.0, 250.0, 0.0, 8.0, 2.0, 0, 0.0],
    ]
    owned = _owned_instances(
        rows,
        origin,
        (700_000.0, 6_600_000.0, 700_500.0, 6_600_500.0),
    )
    assert [row[3] for row in owned] == [6.0, 7.0]


def test_core_crop_keeps_exact_half_metre_alignment() -> None:
    values = np.arange(12 * 12).reshape(12, 12)
    transform = Affine.translation(99.0, 106.0) * Affine.scale(0.5, -0.5)
    cropped, cropped_transform = _crop_grid_to_bounds(
        values, transform, (100.0, 101.0, 104.0, 105.0)
    )
    assert cropped.shape == (8, 8)
    assert cropped[0, 0] == values[2, 2]
    assert tuple(cropped_transform) == tuple(
        Affine.translation(100.0, 105.0) * Affine.scale(0.5, -0.5)
    )


def test_local_terrain_mask_removes_faces_outside_aoi() -> None:
    ground = np.full((5, 5), 100.0)
    valid = np.ones((5, 5), dtype=bool)
    valid[:, -1] = False
    transform = Affine.translation(0.0, 2.5) * Affine.scale(0.5, -0.5)
    mesh = build_local_terrain_mesh(
        ground,
        transform,
        (0.0, 0.0, 100.0),
        valid_mask=valid,
        step_pixels=2,
    )
    assert mesh["vertex_count"] == 6
    assert mesh["face_count"] == 2


def test_segmented_rows_round_trip_to_one_shared_tree_instance_each() -> None:
    rows = [
        [10.0, 20.0, 30.0, 12.0, 7.0, 0, 0.0],
        [40.0, 50.0, 60.0, 8.0, 4.0, 1, 90.0],
    ]
    origin = (700_000.0, 6_600_000.0, 100.0)
    instance_set = build_tree_instance_set(_tree_records(rows, origin), origin)
    decoded = decode_instance_attributes(instance_set)
    assert instance_set["statistics"]["instance_count"] == len(rows)
    assert instance_set["statistics"]["dropped_record_count"] == 0
    assert list(decoded["position_xyz_m"]) == [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ]
