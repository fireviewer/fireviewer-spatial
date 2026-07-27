from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from fetch_ign_orthophoto import (
    GLOBAL_JPEG_BRIGHTNESS,
    GLOBAL_JPEG_CONTRAST,
    GLOBAL_JPEG_PRESET,
    GLOBAL_JPEG_SATURATION,
    JpegDisplayTransform,
    GENERIC_EXAMPLE_BOUNDS,
    _is_lambert93,
    _world_file_lines,
    apply_jpeg_display_transform,
    build_plan,
    execute_plan,
    main,
    sha256_file,
)


def test_jpeg_display_transform_is_neutral_by_default_and_does_not_mutate() -> None:
    rgb = np.array([[[0, 100]], [[50, 150]], [[200, 255]]], dtype=np.uint8)
    original = rgb.copy()

    transformed = apply_jpeg_display_transform(rgb)

    assert np.array_equal(transformed, original)
    assert np.array_equal(rgb, original)
    assert transformed is not rgb
    assert JpegDisplayTransform().metadata()["neutral"] is True
    assert JpegDisplayTransform().metadata()["preset"] == "neutral"


def test_jpeg_display_transform_has_explicit_brightness_saturation_contrast() -> None:
    rgb = np.array([[[100]], [[150]], [[200]]], dtype=np.uint8)
    transform = JpegDisplayTransform(
        brightness=0.8,
        contrast=1.1,
        saturation=1.2,
    )

    transformed = apply_jpeg_display_transform(rgb, transform)

    assert transformed[:, 0, 0].tolist() == [68, 120, 173]
    metadata = transform.metadata()
    assert metadata["scope"] == "derived_blender_rgb_jpeg_only"
    assert metadata["preset"] == "custom"
    assert metadata["raw_geotiff_modified"] is False
    assert metadata["operation_order"] == [
        "brightness_multiply",
        "rec709_luminance_saturation",
        "contrast_around_0_5",
        "clip_0_1",
        "round_uint8",
    ]


def test_jpeg_display_transform_rejects_invalid_values_or_shape() -> None:
    with pytest.raises(ValueError, match="brightness"):
        JpegDisplayTransform(brightness=-0.1).validate()
    with pytest.raises(ValueError, match="shape"):
        apply_jpeg_display_transform(np.zeros((2, 2, 3), dtype=np.uint8))


def test_synthetic_example_plan_estimates_four_two_metre_tiles() -> None:
    plan = build_plan(GENERIC_EXAMPLE_BOUNDS, resolution_m=2.0, tile_pixels=4000)

    assert (plan.width, plan.height) == (5500, 6500)
    assert len(plan.tiles) == 4
    assert plan.bounds_l93 == GENERIC_EXAMPLE_BOUNDS
    assert plan.effective_resolution_m == pytest.approx((2.0, 2.0))
    assert plan.estimated_network_bytes / (1024 * 1024) == pytest.approx(
        102.30, rel=0.01
    )
    assert {(tile.width, tile.height) for tile in plan.tiles} == {
        (4000, 4000),
        (1500, 4000),
        (4000, 2500),
        (1500, 2500),
    }
    assert all("LAYERS=ORTHOIMAGERY.ORTHOPHOTOS" in tile.url for tile in plan.tiles)
    assert all("CRS=EPSG%3A2154" in tile.url for tile in plan.tiles)


def test_plan_rejects_invalid_bounds_resolution_and_tile_size() -> None:
    with pytest.raises(ValueError, match="min_x"):
        build_plan((2, 0, 1, 1))
    with pytest.raises(ValueError, match="positive"):
        build_plan((0, 0, 1, 1), resolution_m=0)
    with pytest.raises(ValueError, match="between 256 and 5010"):
        build_plan((0, 0, 1, 1), tile_pixels=128)


def test_world_file_uses_upper_left_pixel_centre() -> None:
    transform = Affine(2, 0, 100, 0, -3, 200)
    assert _world_file_lines(transform) == (2.0, 0.0, 0.0, -3.0, 101.0, 198.5)


def test_lambert93_recognises_ign_wms_projection_without_datum_authority() -> None:
    ign_wms_crs = CRS.from_string(
        "+proj=lcc +lat_0=46.5 +lon_0=3 +lat_1=49 +lat_2=44 "
        "+x_0=700000 +y_0=6600000 +ellps=WGS84 +units=m +no_defs"
    )
    assert ign_wms_crs.to_epsg() is None
    assert _is_lambert93(ign_wms_crs)
    assert not _is_lambert93(CRS.from_epsg(3857))


def _synthetic_wms_fetcher() -> tuple[object, list[str]]:
    calls: list[str] = []

    def fetch(url: str, destination: Path, _timeout_s: float) -> None:
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        width = int(query["WIDTH"][0])
        height = int(query["HEIGHT"][0])
        bounds = tuple(float(value) for value in query["BBOX"][0].split(","))
        value = len(calls) * 20
        data = np.empty((3, height, width), dtype=np.uint8)
        data[0].fill(value)
        data[1].fill(value + 1)
        data[2].fill(value + 2)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs="EPSG:2154",
            transform=rasterio.transform.from_bounds(*bounds, width, height),
        ) as dataset:
            dataset.write(data)

    return fetch, calls


def test_execute_plan_assembles_validated_rgb_and_provenance(tmp_path: Path) -> None:
    plan = build_plan((879000, 6400000, 879512, 6400512), 1.0, 256)
    output = tmp_path / "synthetic-zone-ortho.tif"
    jpeg = tmp_path / "synthetic-zone-ortho.jpg"
    fetcher, calls = _synthetic_wms_fetcher()

    record = execute_plan(
        plan,
        output,
        jpeg_output=jpeg,
        fetcher=fetcher,  # type: ignore[arg-type]
    )

    assert len(calls) == 4
    assert output.exists()
    assert jpeg.exists()
    assert jpeg.with_suffix(".jgw").exists()
    assert record["status"] == "downloaded_and_validated"
    assert record["aoi_source_note"]["acquisition_year"] == 2023
    assert record["jpeg_display_transform"]["preset"] == GLOBAL_JPEG_PRESET
    assert record["jpeg_display_transform"]["brightness_multiplier"] == 0.78
    assert record["jpeg_display_transform"]["contrast_multiplier"] == 1.08
    assert record["jpeg_display_transform"]["saturation_multiplier"] == 1.12
    assert record["jpeg_display_transform"]["raw_geotiff_modified"] is False
    assert len(record["tiles"]) == 4
    with rasterio.open(output) as dataset:
        assert dataset.crs.to_epsg() == 2154
        assert (dataset.width, dataset.height, dataset.count) == (512, 512, 3)
        assert tuple(dataset.bounds) == pytest.approx(
            (879000, 6400000, 879512, 6400512)
        )
        # Lossy GeoTIFF compression is allowed a very small error.
        assert int(dataset.read(1, window=((10, 11), (10, 11)))[0, 0]) == pytest.approx(
            20, abs=2
        )
        assert int(
            dataset.read(1, window=((10, 11), (300, 301)))[0, 0]
        ) == pytest.approx(40, abs=2)
        assert dataset.overviews(1) == [2, 4]

    source_path = output.with_suffix(".source.json")
    stored = json.loads(source_path.read_text(encoding="utf-8"))
    assert stored["outputs"][0]["sha256"] == sha256_file(output)
    assert stored["outputs"][1]["sha256"] == sha256_file(jpeg)
    assert str(tmp_path) not in source_path.read_text(encoding="utf-8")


def test_jpeg_correction_changes_only_jpeg_and_is_traced(tmp_path: Path) -> None:
    plan = build_plan((879000, 6400000, 879256, 6400256), 1.0, 256)
    output = tmp_path / "ortho-raw.tif"
    jpeg = tmp_path / "ortho-display.jpg"
    fetcher, _ = _synthetic_wms_fetcher()

    record = execute_plan(
        plan,
        output,
        jpeg_output=jpeg,
        jpeg_brightness=0.5,
        jpeg_contrast=1.0,
        jpeg_saturation=0.0,
        fetcher=fetcher,  # type: ignore[arg-type]
    )

    with rasterio.open(output) as raw:
        raw_pixel = raw.read((1, 2, 3), window=((100, 101), (100, 101)))[:, 0, 0]
    with rasterio.open(jpeg) as display:
        display_pixel = display.read((1, 2, 3), window=((100, 101), (100, 101)))[
            :, 0, 0
        ]
    assert raw_pixel.tolist() == pytest.approx([20, 21, 22], abs=2)
    assert display_pixel.tolist() == pytest.approx([10, 10, 10], abs=2)
    assert record["outputs"][0]["display_transform"] == (
        "none_source_mosaic_pixels_unchanged"
    )
    jpeg_record = record["outputs"][1]
    assert jpeg_record["display_transform"]["brightness_multiplier"] == 0.5
    assert jpeg_record["display_transform"]["saturation_multiplier"] == 0.0
    assert record["jpeg_display_transform"] == jpeg_record["display_transform"]


def test_dry_run_never_requires_output_or_fetches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run"
    assert payload["plan"]["tile_count"] == 4
    transform = payload["jpeg_display_transform"]
    assert transform["brightness_multiplier"] == GLOBAL_JPEG_BRIGHTNESS
    assert transform["contrast_multiplier"] == GLOBAL_JPEG_CONTRAST
    assert transform["saturation_multiplier"] == GLOBAL_JPEG_SATURATION
    assert transform["raw_geotiff_modified"] is False
    assert transform["neutral"] is False
    assert transform["preset"] == GLOBAL_JPEG_PRESET


def test_dry_run_reports_requested_jpeg_display_transform(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--dry-run",
                "--jpeg-brightness",
                "0.86",
                "--jpeg-contrast",
                "1.18",
                "--jpeg-saturation",
                "1.12",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    transform = payload["jpeg_display_transform"]
    assert transform["brightness_multiplier"] == 0.86
    assert transform["contrast_multiplier"] == 1.18
    assert transform["saturation_multiplier"] == 1.12


def test_large_download_safety_gate_is_checked_in_dry_run() -> None:
    with pytest.raises(RuntimeError, match="safety limit"):
        main(["--dry-run", "--resolution-m", "0.5"])
