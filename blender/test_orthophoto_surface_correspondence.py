from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from uuid import uuid4

from affine import Affine
import numpy as np
from PIL import Image
import pytest
from shapely.geometry import LineString, box, mapping

from orthophoto_surface_correspondence import (
    LIBRARY_SCHEMA,
    MODEL_SCHEMA,
    compile_aligned_window,
    serialize_tile_outputs,
    slice_tile,
    write_tile_outputs,
)


CORE_BOUNDS = (700_000, 6_300_000, 701_000, 6_300_500)
HALO = 10
WINDOW_BOUNDS = (
    CORE_BOUNDS[0] - HALO,
    CORE_BOUNDS[1] - HALO,
    CORE_BOUNDS[2] + HALO,
    CORE_BOUNDS[3] + HALO,
)
CLASSES = {
    "field": (181, 139, 71),
    "forest": (42, 91, 47),
    "water": (35, 111, 181),
    "rock": (145, 145, 142),
    "road": (75, 73, 70),
    "path": (128, 96, 61),
}
TEST_ROOT = Path(
    "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest/orthophoto-correspondence"
)
TEXTURE_CONTRACT_PATH = Path(__file__).with_name(
    "ground_surface_texture_contract.v4.json"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact(path: str) -> dict[str, object]:
    return {"path": path, "byte_count": 4096, "sha256": _digest(path)}


def _semantic_class(profile_id: str) -> str:
    family, variant = profile_id.split(".", 1)
    if family == "agriculture_field":
        return "field"
    if family in {"road_surface", "railway_bed"}:
        return "road"
    if family == "path_surface":
        return "path"
    if family == "watercourse":
        return "water"
    if family in {"cliff_surface", "burned_ground"}:
        return "rock"
    if variant in {"pine_duff", "oak_litter", "moss_humus", "heath_sand"}:
        return "forest"
    return (
        "field"
        if variant in {"dry_grass", "vineyard_stony", "harvested_stubble", "dark_loam"}
        else "rock"
    )


def _library_and_model() -> tuple[dict[str, object], dict[str, object]]:
    texture_contract = json.loads(TEXTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    profiles = []
    model_profiles = []
    class_variant_count: dict[str, int] = {}
    for stable_index, profile_id in enumerate(
        texture_contract["profile_contract"]["stable_profile_ids"]
    ):
        class_id = _semantic_class(profile_id)
        rgb = CLASSES[class_id]
        variant = class_variant_count.get(class_id, 0)
        class_variant_count[class_id] = variant + 1
        atlas = texture_contract["runtime_atlas"]
        column = stable_index % atlas["columns"]
        row = stable_index // atlas["columns"]
        cell = atlas["cell_size_px"]
        gutter = atlas["gutter_px"]
        profiles.append(
            {
                "stable_index": stable_index,
                "id": profile_id,
                "surface_basis": "atlas_pbr",
                "atlas_slot": stable_index,
                "atlas_uv": {
                    "offset": [
                        (column * cell + gutter) / atlas["width_px"],
                        (row * cell + gutter) / atlas["height_px"],
                    ],
                    "scale": [
                        (cell - 2 * gutter) / atlas["width_px"],
                        (cell - 2 * gutter) / atlas["height_px"],
                    ],
                },
                "physical_scale_m": 4.0,
                "projection": (
                    "world_triplanar"
                    if profile_id.startswith("cliff_surface.")
                    else "world_xy"
                ),
                "source_kind": "clean_pbr_profile_texture",
                "textures": {
                    role: _artifact(f"profile-sources/{stable_index:02d}/{role}.png")
                    for role in ("basecolor", "normal", "height", "orm")
                },
            }
        )
        # Small deterministic prototype offsets make four variants win with
        # non-uniform weights while preserving the semantic class.
        offset = variant // 4
        prototype = [min(255, channel + offset) for channel in rgb]
        model_profiles.append(
            {
                "profile_id": profile_id,
                "class_id": class_id,
                "rgb_u8_by_scale": [prototype, prototype, prototype],
                "texture_u8_by_scale": [0, 0, 0],
                "orientation_mode": (
                    "image_structure"
                    if class_id in {"field", "road", "path"}
                    else "fixed"
                ),
                "default_orientation_deg": {
                    "field": 15.0,
                    "forest": 0.0,
                    "water": 90.0,
                    "rock": 45.0,
                    "road": 0.0,
                    "path": 30.0,
                }[class_id],
            }
        )
    return (
        {
            "schema": LIBRARY_SCHEMA,
            "status": "accepted_clean_pbr_library",
            "texture_contract_sha256": hashlib.sha256(
                TEXTURE_CONTRACT_PATH.read_bytes()
            ).hexdigest(),
            "orthophoto_dependency": "forbidden",
            "lineage": {
                "source": "fresh_clean_pbr_v4",
                "legacy_atlas_dependencies": [],
            },
            "runtime_atlases": {
                role: _artifact(f"runtime-atlas/{role}.png")
                for role in ("basecolor", "normal", "height", "orm")
            },
            "profiles": profiles,
            "visual_acceptance": {
                "status": "accepted_human_visual",
                "receipt": _artifact("qa/accepted-human-visual.json"),
            },
        },
        {
            "schema": MODEL_SCHEMA,
            "status": "locked",
            "resolution_m": 1,
            "feature_window_sizes_m": [1, 5, 17],
            "profiles": model_profiles,
        },
    )


def _rgb(bounds: tuple[int, int, int, int] = WINDOW_BOUNDS) -> np.ndarray:
    west, south, east, north = bounds
    x = np.arange(west, east, dtype=np.int64) + 0.5
    y = np.arange(north - 1, south - 1, -1, dtype=np.int64) + 0.5
    eastings, northings = np.meshgrid(x, y)
    result = np.empty((north - south, east - west, 3), dtype=np.uint8)
    result[:] = CLASSES["field"]
    result[eastings >= 700_500] = CLASSES["forest"]
    result[(eastings >= 700_440) & (eastings < 700_480)] = CLASSES["water"]
    result[eastings >= 700_850] = CLASSES["rock"]
    result[(northings >= 6_300_235) & (northings < 6_300_265)] = CLASSES["road"]
    result[
        (northings - 6_300_050 >= (eastings - 700_000) // 4)
        & (northings - 6_300_050 < (eastings - 700_000) // 4 + 12)
    ] = CLASSES["path"]
    return result


def _transform(bounds: tuple[int, int, int, int]) -> Affine:
    return Affine(1.0, 0.0, bounds[0], 0.0, -1.0, bounds[3])


def _compile(
    rgb: np.ndarray,
    window_bounds: tuple[int, int, int, int],
    core_bounds: tuple[int, int, int, int],
    *,
    context_priors: list[dict[str, object]] | None = None,
    approved_corrections: list[dict[str, object]] | None = None,
):
    library, model = _library_and_model()
    return compile_aligned_window(
        rgb,
        transform=_transform(window_bounds),
        crs="EPSG:2154",
        core_bounds_l93_m=core_bounds,
        orthophoto_sha256=_digest("temporary-synthetic-orthophoto.tif"),
        pbr_library=library,
        correspondence_model=model,
        context_priors=context_priors or (),
        approved_corrections=approved_corrections or (),
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


def test_compiles_six_surface_classes_to_complete_pbr_profiles() -> None:
    correspondence = _compile(_rgb(), WINDOW_BOUNDS, CORE_BOUNDS)
    primary_classes = np.asarray(correspondence.class_by_profile_index, dtype=object)[
        correspondence.profile_ids[..., 0]
    ]
    assert set(np.unique(primary_classes)) == set(CLASSES)
    tile = slice_tile(correspondence, CORE_BOUNDS[:2] + (700_500, 6_300_500))
    outputs = serialize_tile_outputs(tile)

    assert set(outputs) == {
        "ground-profile-ids.png",
        "ground-profile-weights.png",
        "ground-confidence.png",
        "ground-orientation.png",
        "surface-correspondence.json",
    }
    assert np.all(tile.profile_weights.sum(axis=2, dtype=np.uint16) == 255)
    assert tile.profile_ids.shape == (500, 500, 4)
    assert tile.confidence.dtype == np.uint8
    assert tile.orientation.dtype == np.uint8
    manifest = json.loads(outputs["surface-correspondence.json"])
    assert len(manifest["profile_table"]) == 72
    texture_contract = json.loads(TEXTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert [profile["stable_index"] for profile in manifest["profile_table"]] == list(
        range(72)
    )
    assert [profile["id"] for profile in manifest["profile_table"]] == (
        texture_contract["profile_contract"]["stable_profile_ids"]
    )
    assert set(manifest["primary_pixel_counts_by_class"]) >= {
        "field",
        "water",
        "road",
        "path",
    }
    assert manifest["runtime"] == {
        "orthophoto_dependency": "forbidden",
        "orthophoto_pixels_present": False,
        "procedural_materials": "forbidden",
        "projection": "profile_declared_world_xy_or_world_triplanar",
    }
    assert (
        b"temporary-synthetic-orthophoto.tif"
        not in outputs["surface-correspondence.json"]
    )
    assert all(
        "path" not in texture
        for profile in manifest["profile_table"]
        for texture in profile["textures"].values()
    )
    for name, mode in {
        "ground-profile-ids.png": "RGBA",
        "ground-profile-weights.png": "RGBA",
        "ground-confidence.png": "L",
        "ground-orientation.png": "L",
    }.items():
        with Image.open(BytesIO(outputs[name])) as image:
            assert image.mode == mode
            assert image.size == (500, 500)


def test_grouping_and_adjacent_tile_compilation_are_bit_identical() -> None:
    global_result = _compile(_rgb(), WINDOW_BOUNDS, CORE_BOUNDS)
    global_left = serialize_tile_outputs(
        slice_tile(global_result, (700_000, 6_300_000, 700_500, 6_300_500))
    )
    global_right = serialize_tile_outputs(
        slice_tile(global_result, (700_500, 6_300_000, 701_000, 6_300_500))
    )

    left_window = (699_990, 6_299_990, 700_510, 6_300_510)
    right_window = (700_490, 6_299_990, 701_010, 6_300_510)
    left = _compile(
        _rgb(left_window),
        left_window,
        (700_000, 6_300_000, 700_500, 6_300_500),
    )
    right = _compile(
        _rgb(right_window),
        right_window,
        (700_500, 6_300_000, 701_000, 6_300_500),
    )
    separate_left = serialize_tile_outputs(
        slice_tile(left, (700_000, 6_300_000, 700_500, 6_300_500))
    )
    separate_right = serialize_tile_outputs(
        slice_tile(right, (700_500, 6_300_000, 701_000, 6_300_500))
    )

    assert global_left == separate_left
    assert global_right == separate_right


def test_context_prior_and_approved_correction_restrict_profiles() -> None:
    road_prior = {
        "feature_id": "road:global-1",
        "priority": 10,
        "geometry": mapping(LineString([(700_000, 6_300_250), (701_000, 6_300_250)])),
        "width_m": 30.0,
        "allowed_class_ids": ["road"],
    }
    water_correction = {
        "correction_id": "crossing:road-water-1",
        "priority": 0,
        "geometry": mapping(box(700_450, 6_300_245, 700_470, 6_300_255)),
        "allowed_class_ids": ["water"],
        "approved": True,
        "approval_sha256": _digest("approved crossing road-water-1 as ford"),
    }
    result = _compile(
        _rgb(),
        WINDOW_BOUNDS,
        CORE_BOUNDS,
        context_priors=[road_prior],
        approved_corrections=[water_correction],
    )
    primary_classes = np.asarray(result.class_by_profile_index, dtype=object)[
        result.profile_ids[..., 0]
    ]
    row = CORE_BOUNDS[3] - 6_300_250
    assert primary_classes[row, 100] == "road"
    assert primary_classes[row, 460] == "water"
    assert result.restriction_counts["context:road:global-1"] > 0
    assert result.restriction_counts["correction:crossing:road-water-1"] > 0


def test_unapproved_correction_and_incomplete_pbr_library_fail_closed() -> None:
    correction = {
        "correction_id": "not-approved",
        "geometry": mapping(box(700_000, 6_300_000, 700_010, 6_300_010)),
        "allowed_class_ids": ["rock"],
        "approved": False,
        "approval_sha256": _digest("not approved"),
    }
    with pytest.raises(ValueError, match="not explicitly approved"):
        _compile(
            _rgb(),
            WINDOW_BOUNDS,
            CORE_BOUNDS,
            approved_corrections=[correction],
        )

    library, model = _library_and_model()
    del library["profiles"][0]["textures"]["normal"]  # type: ignore[index]
    with pytest.raises(ValueError, match="incomplete"):
        compile_aligned_window(
            _rgb(),
            transform=_transform(WINDOW_BOUNDS),
            crs="EPSG:2154",
            core_bounds_l93_m=CORE_BOUNDS,
            orthophoto_sha256=_digest("source"),
            pbr_library=library,
            correspondence_model=model,
        )


def test_requires_global_grid_halo_and_publishes_atomically_on_d(
    d_output_root: Path,
) -> None:
    library, model = _library_and_model()
    with pytest.raises(ValueError, match="10 m halo"):
        compile_aligned_window(
            _rgb(CORE_BOUNDS),
            transform=_transform(CORE_BOUNDS),
            crs="EPSG:2154",
            core_bounds_l93_m=CORE_BOUNDS,
            orthophoto_sha256=_digest("source"),
            pbr_library=library,
            correspondence_model=model,
        )

    result = _compile(_rgb(), WINDOW_BOUNDS, CORE_BOUNDS)
    tile = slice_tile(result, (700_000, 6_300_000, 700_500, 6_300_500))
    destination = d_output_root / "tile"
    first_hashes = write_tile_outputs(tile, destination)
    assert set(first_hashes) == {
        "ground-profile-ids.png",
        "ground-profile-weights.png",
        "ground-confidence.png",
        "ground-orientation.png",
        "surface-correspondence.json",
    }
    assert sorted(item.name for item in destination.iterdir()) == sorted(first_hashes)
    assert not list(d_output_root.glob("*.part"))
    with pytest.raises(FileExistsError):
        write_tile_outputs(tile, destination)


def test_single_profile_restriction_has_no_invalid_rgba_index() -> None:
    library, _model = _library_and_model()
    profiles = library["profiles"]
    assert isinstance(profiles, list)
    selected = next(
        profile
        for profile in profiles
        if _semantic_class(str(profile["id"])) == "field"
    )
    result = _compile(
        _rgb(),
        WINDOW_BOUNDS,
        CORE_BOUNDS,
        context_priors=[
            {
                "feature_id": "approved-field-profile",
                "priority": 1,
                "geometry": mapping(box(*CORE_BOUNDS)),
                "allowed_profile_ids": [selected["id"]],
            }
        ],
    )
    stable_index = int(selected["stable_index"])
    assert np.all(result.profile_ids == stable_index)
    assert np.all(result.profile_weights[..., 0] == 255)
    assert np.all(result.profile_weights[..., 1:] == 0)
    assert np.all(result.confidence == 255)
