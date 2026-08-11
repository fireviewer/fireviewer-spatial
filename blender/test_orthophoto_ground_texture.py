from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from affine import Affine
import numpy as np
from PIL import Image
import pytest

from orthophoto_ground_texture import (
    OUTPUT_SCHEMA,
    compile_aligned_window,
    serialize_tile_outputs,
    slice_tile,
    write_tile_outputs,
)


CORE_BOUNDS = (700_000, 6_300_000, 700_500, 6_300_500)
BATCH_BOUNDS = (700_000, 6_300_000, 701_000, 6_300_500)
HALO = 10
TEST_ROOT = Path(
    "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest/orthophoto-ground"
)


def _window_bounds(core: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    west, south, east, north = core
    return west - HALO, south - HALO, east + HALO, north + HALO


def _rgb(bounds: tuple[int, int, int, int]) -> np.ndarray:
    west, south, east, north = bounds
    eastings = np.arange(west, east, dtype=np.int64)[None, :]
    northings = np.arange(north - 1, south - 1, -1, dtype=np.int64)[:, None]
    return np.stack(
        (
            np.broadcast_to((eastings + northings) % 256, (north - south, east - west)),
            np.broadcast_to(
                (2 * eastings + 3 * northings) % 256, (north - south, east - west)
            ),
            np.broadcast_to(
                (5 * eastings + 7 * northings) % 256, (north - south, east - west)
            ),
        ),
        axis=2,
    ).astype(np.uint8)


def _transform(bounds: tuple[int, int, int, int]) -> Affine:
    return Affine(1.0, 0.0, bounds[0], 0.0, -1.0, bounds[3])


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compile(core: tuple[int, int, int, int]):
    bounds = _window_bounds(core)
    return compile_aligned_window(
        _rgb(bounds),
        transform=_transform(bounds),
        crs="EPSG:2154",
        core_bounds_l93_m=core,
        orthophoto_source_manifest_sha256=_digest("ign-ortho-source-manifest-r2026.1"),
        orthophoto_revision="ign-ortho-r2026.1",
    )


@pytest.fixture
def d_output_root() -> Path:
    root = TEST_ROOT / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.resolve().is_relative_to(TEST_ROOT.resolve()):
            shutil.rmtree(root, ignore_errors=True)


def test_compiles_520_window_to_one_rgb8_500_texture_without_resampling() -> None:
    source_bounds = _window_bounds(CORE_BOUNDS)
    source = _rgb(source_bounds)
    window = compile_aligned_window(
        source,
        transform=_transform(source_bounds),
        crs="EPSG:2154",
        core_bounds_l93_m=CORE_BOUNDS,
        orthophoto_source_manifest_sha256=_digest("source-manifest"),
        orthophoto_revision="revision-2026.1",
    )
    tile = slice_tile(window, CORE_BOUNDS)
    outputs = serialize_tile_outputs(tile)

    assert set(outputs) == {"ground-color.png", "ground-color.json"}
    core = source[HALO : HALO + 500, HALO : HALO + 500]
    with Image.open(BytesIO(outputs["ground-color.png"])) as image:
        assert image.mode == "RGB"
        assert image.size == (500, 500)
        assert np.array_equal(np.asarray(image), core)

    manifest = json.loads(outputs["ground-color.json"])
    assert manifest["schema"] == OUTPUT_SCHEMA
    assert manifest["bounds_l93_m"] == list(CORE_BOUNDS)
    assert manifest["source_window_bounds_l93_m"] == list(source_bounds)
    assert manifest["grid"] == {
        "resolution_m": 1,
        "width": 500,
        "height": 500,
        "affine": [1, 0, CORE_BOUNDS[0], 0, -1, CORE_BOUNDS[3]],
        "pixel_interpretation": "area",
    }
    assert manifest["identity"]["orthophoto_revision"] == "revision-2026.1"
    assert manifest["runtime"] == {
        "texture_file": "ground-color.png",
        "orthophoto_source_file_dependency": "forbidden",
        "orthophoto_source_path_present": False,
    }
    assert b"orthophoto_path" not in outputs["ground-color.json"]
    assert (
        manifest["artifact"]["sha256"]
        == hashlib.sha256(outputs["ground-color.png"]).hexdigest()
    )


def test_batch_and_separate_tiles_are_bit_identical_and_exactly_contiguous() -> None:
    batch = _compile(BATCH_BOUNDS)
    left_bounds = CORE_BOUNDS
    right_bounds = (700_500, 6_300_000, 701_000, 6_300_500)

    batch_left = serialize_tile_outputs(slice_tile(batch, left_bounds))
    batch_right = serialize_tile_outputs(slice_tile(batch, right_bounds))
    separate_left = serialize_tile_outputs(
        slice_tile(_compile(left_bounds), left_bounds)
    )
    separate_right = serialize_tile_outputs(
        slice_tile(_compile(right_bounds), right_bounds)
    )

    assert batch_left == separate_left
    assert batch_right == separate_right
    assert np.array_equal(
        np.concatenate(
            (
                np.asarray(Image.open(BytesIO(batch_left["ground-color.png"]))),
                np.asarray(Image.open(BytesIO(batch_right["ground-color.png"]))),
            ),
            axis=1,
        ),
        batch.color_rgb_u8,
    )
    left_manifest = json.loads(batch_left["ground-color.json"])
    right_manifest = json.loads(batch_right["ground-color.json"])
    assert left_manifest["bounds_l93_m"][2] == right_manifest["bounds_l93_m"][0]
    assert (
        left_manifest["grid"]["affine"][2]
        + left_manifest["grid"]["width"] * left_manifest["grid"]["resolution_m"]
        == right_manifest["grid"]["affine"][2]
    )


def test_exact_crop_preserves_source_pixels_and_rejects_unaligned_inputs() -> None:
    bounds = _window_bounds(CORE_BOUNDS)
    source = np.zeros((520, 520, 3), dtype=np.uint8)
    source[HALO + 1, HALO + 1] = [2, 1, 6]
    window = compile_aligned_window(
        source,
        transform=_transform(bounds),
        crs="EPSG:2154",
        core_bounds_l93_m=CORE_BOUNDS,
        orthophoto_source_manifest_sha256=_digest("source"),
        orthophoto_revision="revision-1",
    )
    assert window.color_rgb_u8[1, 1].tolist() == [2, 1, 6]
    assert int(np.count_nonzero(window.color_rgb_u8)) == 3

    with pytest.raises(ValueError, match="10 m halo"):
        compile_aligned_window(
            _rgb(CORE_BOUNDS),
            transform=_transform(CORE_BOUNDS),
            crs="EPSG:2154",
            core_bounds_l93_m=CORE_BOUNDS,
            orthophoto_source_manifest_sha256=_digest("source"),
            orthophoto_revision="revision-1",
        )
    with pytest.raises(ValueError, match="global metre grid"):
        compile_aligned_window(
            source,
            transform=Affine(1.0, 0.0, bounds[0] + 0.5, 0.0, -1.0, bounds[3]),
            crs="EPSG:2154",
            core_bounds_l93_m=CORE_BOUNDS,
            orthophoto_source_manifest_sha256=_digest("source"),
            orthophoto_revision="revision-1",
        )
    with pytest.raises(ValueError, match="non-path token"):
        compile_aligned_window(
            source,
            transform=_transform(bounds),
            crs="EPSG:2154",
            core_bounds_l93_m=CORE_BOUNDS,
            orthophoto_source_manifest_sha256=_digest("source"),
            orthophoto_revision="../temporary-ortho.tif",
        )


def test_rebuild_is_bit_identical_and_publication_is_atomic_on_d(
    d_output_root: Path,
) -> None:
    first = serialize_tile_outputs(slice_tile(_compile(CORE_BOUNDS), CORE_BOUNDS))
    second_tile = slice_tile(_compile(CORE_BOUNDS), CORE_BOUNDS)
    second = serialize_tile_outputs(second_tile)
    assert first == second

    destination = d_output_root / "tile"
    hashes = write_tile_outputs(second_tile, destination)
    assert set(hashes) == {"ground-color.png", "ground-color.json"}
    assert sorted(item.name for item in destination.iterdir()) == sorted(hashes)
    assert not list(d_output_root.glob("*.part"))
    with pytest.raises(FileExistsError):
        write_tile_outputs(second_tile, destination)

    if os.name == "nt":
        with pytest.raises(ValueError, match="must stay on D"):
            write_tile_outputs(second_tile, Path("C:/tmp/fireviewer-ground-test"))
