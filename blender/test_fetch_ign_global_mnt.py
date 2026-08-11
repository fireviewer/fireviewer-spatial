from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from fetch_ign_global_mnt import MNT_LAYER, execute_plan, write_production_manifest  # noqa: E402
from fetch_ign_orthophoto import build_plan  # noqa: E402


def _fetcher(_url: str, destination: Path, _timeout: float) -> None:
    transform = from_bounds(700000.0, 6600000.0, 700010.0, 6600010.0, 2, 2)
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:2154",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.array([[10.0, 11.0], [12.0, 13.0]], dtype="float32"), 1)


def test_compact_mnt_keeps_only_final_cog_and_writes_honest_manifest(tmp_path: Path) -> None:
    aoi = tmp_path / "aoi.geojson"
    aoi.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    output = tmp_path / "terrain/mnt-global.cog.tif"
    plan = build_plan((700000.0, 6600000.0, 700010.0, 6600010.0), 5.0, 2000, layer=MNT_LAYER)

    source = execute_plan(plan, output, fetcher=_fetcher)
    manifest = tmp_path / "global-05m/production-manifest.json"
    write_production_manifest(
        manifest,
        aoi=aoi,
        origin=[700005.0, 6600005.0, 0.0],
        plan=plan,
        source_record=source,
    )

    assert source["terrain_surface"] == "mnt_wms_resampled"
    assert source["excluded_surfaces"] == ["mns", "vegetation_assets", "building_assets", "detail_lod"]
    assert output.is_file() and output.with_suffix(".source.json").is_file()
    assert manifest.is_file()
    assert '"mode": "global_mnt_only"' in manifest.read_text(encoding="utf-8")
