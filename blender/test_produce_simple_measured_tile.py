from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import sys
import uuid

from affine import Affine
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.crs import CRS


BLENDER_ROOT = Path(__file__).resolve().parent
if str(BLENDER_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_ROOT))

import produce_simple_measured_tile as simple  # noqa: E402


D_TEST_ROOT = Path(
    "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest-simple-measured-tile"
)
ORIGIN = (700_000, 6_600_000)
ZONE_ID = "FR-TEST-00001"
TILE_ID = "x700000_y6600000"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _artifact(root: str, path: str, content_hash: str, byte_count: int) -> dict:
    return {
        "root": root,
        "path": path,
        "byte_count": byte_count,
        "sha256": content_hash,
    }


def _catalogue(root: Path, prototype: Path) -> Path:
    contract_path = BLENDER_ROOT / "asset_library_contract.v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source_hash = _hash_file(prototype)
    source_size = prototype.stat().st_size
    texture_buffer = BytesIO()
    Image.new("RGB", (2, 2), (54, 112, 42)).save(texture_buffer, format="PNG")
    texture_bytes = texture_buffer.getvalue()
    texture_hash = _hash_bytes(texture_bytes)
    texture_size = len(texture_bytes)
    texture_root = prototype.parent.parent / "textures"
    texture_root.mkdir(parents=True)
    categories = [
        *("building" for _ in range(24)),
        "pasture_equipment",
        *("road_equipment" for _ in range(8)),
        *("tree" for _ in range(18)),
        *("vehicle" for _ in range(2)),
    ]
    assets = []
    for index, category in enumerate(categories):
        asset_id = f"{index:012x}_{category}_{index:02d}"
        assets.append(
            {
                "asset_id": asset_id,
                "category": category,
                "identity": {
                    "strategy": "artifact_stem+reference_manifest",
                    "rejected_reported_asset_id": None,
                },
                "source": _artifact(
                    "reference_assets",
                    f"sources/{asset_id}.png",
                    source_hash,
                    source_size,
                ),
                "usd": _artifact(
                    "review_batch", "prototypes/source.usda", source_hash, source_size
                ),
                "texture": _artifact(
                    "review_batch",
                    f"textures/{asset_id}.png",
                    texture_hash,
                    texture_size,
                ),
                "receipt": _artifact(
                    "review_batch",
                    f"receipts/{asset_id}.json",
                    source_hash,
                    source_size,
                ),
                "source_bounds": {
                    "status": "reported",
                    "coordinate_space": "source_glb_unscaled",
                    "minimum": [-1.0, 0.0, -1.0],
                    "maximum": [1.0, 4.0, 1.0],
                    "diagonal": math.sqrt(24.0),
                },
                "usd_stage": {
                    "status": "pending",
                    "up_axis": None,
                    "meters_per_unit": None,
                    "default_prim": None,
                },
                "qualification": contract["qualification_defaults"],
                "eligibility": contract["eligibility_defaults"],
            }
        )
        (texture_root / f"{asset_id}.png").write_bytes(texture_bytes)
    assets.sort(key=lambda value: value["asset_id"])
    payload = {
        "schema": "fireviewer.asset-library.v1",
        "status": "catalogued_pending_simready_qualification",
        "build_algorithm": "fireviewer.asset-library-53-builder.v1",
        "contract_sha256": _hash_file(contract_path),
        "asset_count": 53,
        "category_counts": contract["expected_category_counts"],
        "normalization_policy": contract["normalization_policy"],
        "selection_pools": {
            category: [
                asset["asset_id"] for asset in assets if asset["category"] == category
            ]
            for category in ("building", "tree")
        },
        "input_evidence": {
            "reference_manifest": _artifact(
                "reference_assets", "reference-manifest.json", source_hash, source_size
            ),
            "usd_conversion_manifest": _artifact(
                "review_batch", "conversion-manifest.json", source_hash, source_size
            ),
            "glb_validation_manifest": _artifact(
                "generated_assets", "validation-manifest.json", source_hash, source_size
            ),
        },
        "assets": assets,
    }
    payload["catalog_revision"] = _hash_bytes(simple.canonical_json_bytes(payload))
    path = root / "asset-library.v1.json"
    _write_json(path, payload)
    return path


def _sources(root: Path) -> dict[str, Path]:
    source_root = root / "sources"
    source_root.mkdir()
    elevation_rows, elevation_columns = np.indices((1040, 1040), dtype="float64")
    mnt = 100.0 + elevation_columns * 0.001 + (1039.0 - elevation_rows) * 0.0005
    radius = np.sqrt((elevation_rows - 520.0) ** 2 + (elevation_columns - 540.0) ** 2)
    hag = np.maximum(0.0, 9.0 * (1.0 - radius / 14.0))
    mns = mnt + hag
    transform = Affine(0.5, 0, ORIGIN[0] - 10, 0, -0.5, ORIGIN[1] + 510)
    paths: dict[str, Path] = {}
    wms_fallback_crs = CRS.from_dict(
        {
            "proj": "lcc",
            "lat_0": 46.5,
            "lon_0": 3,
            "lat_1": 49,
            "lat_2": 44,
            "x_0": 700000,
            "y_0": 6600000,
            "ellps": "WGS84",
            "units": "m",
            "no_defs": True,
        }
    )
    assert wms_fallback_crs.to_epsg() is None
    for name, values in (("mnt", mnt), ("mns", mns)):
        path = source_root / f"{name}-05m.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=1040,
            height=1040,
            count=1,
            dtype="float32",
            crs=wms_fallback_crs,
            transform=transform,
            nodata=-9999.0,
            compress="DEFLATE",
        ) as dataset:
            dataset.write(values.astype("float32"), 1)
        paths[name] = path
    rows, columns = np.indices((520, 520), dtype="uint16")
    rgb = np.empty((520, 520, 3), dtype="uint8")
    rgb[..., 0] = (columns * 3 % 256).astype("uint8")
    rgb[..., 1] = (rows * 5 % 256).astype("uint8")
    rgb[..., 2] = ((rows + columns) * 7 % 256).astype("uint8")
    orthophoto = source_root / "orthophoto-1m.png"
    Image.fromarray(rgb, mode="RGB").save(orthophoto, format="PNG", compress_level=9)
    paths["orthophoto"] = orthophoto

    bounds = [ORIGIN[0] - 10, ORIGIN[1] - 10, ORIGIN[0] + 510, ORIGIN[1] + 510]
    elevation_receipt = source_root / "elevation-source.json"
    elevation = {
        "schema": "fireviewer.mnt-mns-source-pair.v1",
        "status": "downloaded_coregistered_verified",
        "zone_id": ZONE_ID,
        "tile_id": TILE_ID,
        "crs": "EPSG:2154",
        "bounds_l93_m": bounds,
        "grid": {
            "resolution_m": 0.5,
            "width": 1040,
            "height": 1040,
            "halo_m": 10,
            "affine": [0.5, 0, ORIGIN[0] - 10, 0, -0.5, ORIGIN[1] + 510],
            "row_order": "north_to_south",
            "nodata": -9999.0,
        },
        "mnt": {
            "file": paths["mnt"].name,
            "byte_count": paths["mnt"].stat().st_size,
            "sha256": _hash_file(paths["mnt"]),
        },
        "mns": {
            "file": paths["mns"].name,
            "byte_count": paths["mns"].stat().st_size,
            "sha256": _hash_file(paths["mns"]),
        },
    }
    _write_json(elevation_receipt, elevation)
    paths["elevation_receipt"] = elevation_receipt
    orthophoto_receipt = source_root / "orthophoto-source.json"
    _write_json(
        orthophoto_receipt,
        {
            "schema": "fireviewer.orthophoto-source.v1",
            "status": "downloaded_verified",
            "zone_id": ZONE_ID,
            "tile_id": TILE_ID,
            "crs": "EPSG:2154",
            "bounds_l93_m": bounds,
            "grid": {
                "resolution_m": 1,
                "width": 520,
                "height": 520,
                "halo_m": 10,
                "row_order": "north_to_south",
            },
            "provider": {"revision": "synthetic-orthophoto-v1"},
            "source": {
                "file": orthophoto.name,
                "byte_count": orthophoto.stat().st_size,
                "sha256": _hash_file(orthophoto),
            },
        },
    )
    paths["orthophoto_receipt"] = orthophoto_receipt
    placement_context = source_root / "placement-context.json"
    _write_json(
        placement_context,
        {
            "schema": "fireviewer.placement-context-input.v1",
            "crs": "EPSG:2154",
            "tile_origin_l93_m": list(ORIGIN),
            "processing_bounds_l93_m": bounds,
            "building_footprints": [],
            "context_geometries": {
                "vegetation": [
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [ORIGIN[0], ORIGIN[1]],
                                [ORIGIN[0] + 500, ORIGIN[1]],
                                [ORIGIN[0] + 500, ORIGIN[1] + 500],
                                [ORIGIN[0], ORIGIN[1] + 500],
                                [ORIGIN[0], ORIGIN[1]],
                            ]
                        ],
                    }
                ],
                "roads": [],
                "rail": [],
                "water": [],
            },
        },
    )
    paths["placement_context"] = placement_context
    return paths


@pytest.fixture
def tile_fixture():
    root = D_TEST_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        sources = _sources(root)
        asset_root = root / "assets"
        prototype = asset_root / "prototypes" / "source.usda"
        prototype.parent.mkdir(parents=True)
        prototype.write_text(
            '#usda 1.0\n(\n    defaultPrim = "Asset"\n    metersPerUnit = 1\n    upAxis = "Y"\n)\ndef Xform "Asset" {}\n',
            encoding="utf-8",
            newline="\n",
        )
        library = _catalogue(root, prototype)
        yield root, sources, asset_root, library
    finally:
        if root.exists() and root.resolve().is_relative_to(D_TEST_ROOT.resolve()):
            shutil.rmtree(root)


def _produce(
    root: Path, sources: dict[str, Path], asset_root: Path, library: Path, name: str
):
    return simple.produce_simple_measured_tile(
        mnt_05m=sources["mnt"],
        mns_05m=sources["mns"],
        orthophoto_1m=sources["orthophoto"],
        elevation_source_receipt=sources["elevation_receipt"],
        orthophoto_source_receipt=sources["orthophoto_receipt"],
        placement_context=sources["placement_context"],
        asset_library=library,
        asset_roots={"review_batch": asset_root},
        portable_root=root,
        output_root=root / name,
        zone_id=ZONE_ID,
        tile_id=TILE_ID,
        tile_origin_l93_m=ORIGIN,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_canonical_mnt_sampler_quantizes_pixels_before_integer_mean() -> None:
    pixels_mm = np.arange(520 * 520, dtype="int64").reshape(520, 520) - 100_000
    source_m = pixels_mm.astype("float64") / 1000.0
    result = simple.canonical_mnt_normal_halo_mm(source_m)
    south = np.flipud(pixels_mm)
    summed = south[7, 7] + south[8, 7] + south[7, 8] + south[8, 8]
    expected = (summed + 2) // 4 if summed >= 0 else -((-summed + 2) // 4)
    assert result.shape == (253, 253)
    assert result.dtype == np.dtype("<i4")
    assert int(result[0, 0]) == int(expected)
    assert result.tobytes() == simple.canonical_mnt_normal_halo_mm(source_m).tobytes()


def test_native_half_metre_reduction_keeps_subpixel_crown_peak() -> None:
    mnt = np.full((1040, 1040), 100.0, dtype="float64")
    mns = mnt.copy()
    mns[200, 301] += 4.5

    mnt_1m, mns_1m, diagnostics = simple.canonical_elevation_pair_1m_from_05m(mnt, mns)

    assert mnt_1m.shape == (520, 520)
    assert mns_1m[100, 150] - mnt_1m[100, 150] == pytest.approx(4.5)
    assert diagnostics["native_resolution_m"] == 0.5
    assert diagnostics["hag_reducer"] == ("maximum_of_four_canonical_0.5m_deltas")


def test_rejects_near_lambert_crs_with_wrong_parameter() -> None:
    wrong = CRS.from_dict(
        {
            "proj": "lcc",
            "lat_0": 46.5,
            "lon_0": 4,
            "lat_1": 49,
            "lat_2": 44,
            "x_0": 700000,
            "y_0": 6600000,
            "ellps": "WGS84",
            "units": "m",
            "no_defs": True,
        }
    )
    assert not simple._is_lambert93_crs(wrong)


def test_produces_reuses_and_rebuilds_bit_identical_source_free_tile(
    tile_fixture,
) -> None:
    root, sources, asset_root, library = tile_fixture
    first = _produce(root, sources, asset_root, library, "tile-a")
    second = _produce(root, sources, asset_root, library, "tile-b")
    reused = _produce(root, sources, asset_root, library, "tile-a")

    assert not first.reused
    assert not second.reused
    assert reused.reused
    assert _tree_bytes(first.output_root) == _tree_bytes(second.output_root)
    receipt = json.loads(first.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "technical_pilot_non_final"
    assert receipt["accepted_final"] is False
    output_names = set(receipt["outputs"])
    assert simple.EXPECTED_OUTPUTS <= output_names
    bundled = output_names - simple.EXPECTED_OUTPUTS
    assert len(bundled) == 3
    assert all(simple._is_local_prototype_artifact(path) for path in bundled)
    assert simple._is_local_prototype_artifact("scene/prototypes/real_asset/source.usd")
    assert not any(
        "orthophoto-1m" in name or "mnt-05m" in name or "mns-05m" in name
        for name in receipt["outputs"]
    )
    with Image.open(first.ground_color) as compiled_ground:
        assert compiled_ground.mode == "RGB"
        assert compiled_ground.size == (500, 500)

    unexpected = first.output_root / "scene" / "prototypes" / "unexpected.txt"
    unexpected.write_text("not part of the sealed bundle", encoding="utf-8")
    with pytest.raises(simple.SimpleMeasuredTileError, match="output set mismatch"):
        simple.validate_simple_measured_tile_package(
            first.output_root,
            expected_request=receipt["request"],
            asset_library=library,
            asset_roots={"review_batch": asset_root},
        )


def test_two_tiles_reuse_one_explicit_shared_prototype_bundle(tile_fixture) -> None:
    root, sources, asset_root, library = tile_fixture
    shared = root / "shared" / "prototypes"

    def produce(name: str):
        return simple.produce_simple_measured_tile(
            mnt_05m=sources["mnt"],
            mns_05m=sources["mns"],
            orthophoto_1m=sources["orthophoto"],
            elevation_source_receipt=sources["elevation_receipt"],
            orthophoto_source_receipt=sources["orthophoto_receipt"],
            placement_context=sources["placement_context"],
            asset_library=library,
            asset_roots={"review_batch": asset_root},
            portable_root=root,
            asset_bundle_root=shared,
            output_root=root / name,
            zone_id=ZONE_ID,
            tile_id=TILE_ID,
            tile_origin_l93_m=ORIGIN,
        )

    first = produce("tile-shared-a")
    shared_bytes = _tree_bytes(shared)
    second = produce("tile-shared-b")

    assert _tree_bytes(shared) == shared_bytes
    for package in (first, second):
        receipt = json.loads(package.receipt.read_text(encoding="utf-8"))
        assert receipt["request"]["prototype_bundle"] == {
            "scope": "explicit_shared",
            "portable_path": "shared/prototypes",
        }
        assert not (package.output_root / "scene" / "prototypes").exists()
        scene = json.loads(
            (package.output_root / "scene" / "scene.done.json").read_text(
                encoding="utf-8"
            )
        )
        assert scene["prototype_bundle"]["scope"] == "explicit_shared"


def test_rejects_source_hash_drift_before_publishing(tile_fixture) -> None:
    root, sources, asset_root, library = tile_fixture
    sources["orthophoto"].write_bytes(sources["orthophoto"].read_bytes() + b"tamper")
    with pytest.raises(
        simple.SimpleMeasuredTileError, match="Orthophoto source hash mismatch"
    ):
        _produce(root, sources, asset_root, library, "rejected")
    assert not (root / "rejected").exists()


def test_context_hash_is_part_of_reuse_identity(tile_fixture) -> None:
    root, sources, asset_root, library = tile_fixture
    _produce(root, sources, asset_root, library, "tile-context")
    context = json.loads(sources["placement_context"].read_text(encoding="utf-8"))
    context["provenance"] = {"revision": "changed-after-build"}
    _write_json(sources["placement_context"], context)
    with pytest.raises(simple.SimpleMeasuredTileError, match="different request"):
        _produce(root, sources, asset_root, library, "tile-context")


def test_corrupt_mns_falls_back_to_explicit_ground_only_tile(tile_fixture) -> None:
    root, sources, asset_root, library = tile_fixture
    with rasterio.open(sources["mnt"]) as dataset:
        mnt = dataset.read(1)
        profile = dataset.profile
    with rasterio.open(sources["mns"], "w", **profile) as dataset:
        dataset.write((mnt - 0.75).astype("float32"), 1)
    elevation = json.loads(sources["elevation_receipt"].read_text(encoding="utf-8"))
    elevation["mns"]["byte_count"] = sources["mns"].stat().st_size
    elevation["mns"]["sha256"] = _hash_file(sources["mns"])
    _write_json(sources["elevation_receipt"], elevation)

    package = _produce(root, sources, asset_root, library, "tile-mns-fallback")
    receipt = json.loads(package.receipt.read_text(encoding="utf-8"))
    assert receipt["placement"]["source"]["mode"] == "degraded_mns_fallback"
    assert receipt["placement"]["source"]["degraded"] is True
    assert receipt["placement"]["building_valid_count"] == 0
    assert receipt["placement"]["tree_valid_count"] == 0
    assert package.ground_color.is_file()


def test_rejects_c_drive_output_before_reading_sources(tile_fixture) -> None:
    root, sources, asset_root, library = tile_fixture
    with pytest.raises(simple.SimpleMeasuredTileError, match="must stay on D"):
        simple.produce_simple_measured_tile(
            mnt_05m=sources["mnt"],
            mns_05m=sources["mns"],
            orthophoto_1m=sources["orthophoto"],
            elevation_source_receipt=sources["elevation_receipt"],
            orthophoto_source_receipt=sources["orthophoto_receipt"],
            placement_context=sources["placement_context"],
            asset_library=library,
            asset_roots={"review_batch": asset_root},
            portable_root=root,
            output_root=Path("C:/fireviewer-forbidden/tile"),
            zone_id=ZONE_ID,
            tile_id=TILE_ID,
            tile_origin_l93_m=ORIGIN,
        )
