"""Rebuild exhaustive detected-crown vegetation from locked 0.5 m MNT/MNS.

The source map remains immutable.  Every source pair is checksum-verified,
segmentation is performed with a ten-metre neighbour halo, and only crown
apices owned by the current one-kilometre core are retained.  Existing map
building and route meshes are rasterised as exclusions before segmentation so
that elevated infrastructure is not misclassified as vegetation.

"Exhaustive" in this module means every accepted MNS-MNT crown apex under the
declared segmentation contract.  It is not a claim of field inventory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import urllib.request
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge as merge_rasters
from shapely.geometry import MultiPoint, mapping
import trimesh


HERE = Path(__file__).resolve().parent
BLENDER_DIR = HERE.parent / "blender"
if str(BLENDER_DIR) not in sys.path:
    sys.path.insert(0, str(BLENDER_DIR))

from prepare_mid_vegetation_05m import (  # noqa: E402
    SegmentationConfig,
    segment_vegetation_instances,
)


SCHEMA = "fireviewer.mnt-mns-vegetation-rebuild.v1"
RASTER_KINDS = ("mnt", "mns")
EXCLUDED_GEOMETRY = frozenset({"buildings", "road", "path", "itinerary"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raster_records(manifest: Mapping[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    value = manifest.get("rasters", {}).get(kind)
    if not isinstance(value, dict):
        raise ValueError(f"Source manifest has no {kind.upper()} record map")
    records: dict[str, dict[str, Any]] = {}
    for tile_id, raw in value.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Malformed {kind.upper()} record {tile_id}")
        record = dict(raw)
        record["tile_id"] = str(tile_id)
        records[str(tile_id)] = record
    return records


def source_target(cache_root: Path, kind: str, record: Mapping[str, Any]) -> Path:
    return cache_root / kind / str(record["name_download"])


def download_one(cache_root: Path, kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    target = source_target(cache_root, kind, record)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(record["byte_size"])
    expected_hash = str(record["sha256"]).lower()
    if target.is_file() and target.stat().st_size == expected_size:
        actual_hash = sha256_file(target)
        if actual_hash == expected_hash:
            return {"kind": kind, "tile_id": record["tile_id"], "state": "reused"}
    temporary = target.with_name(f".{target.name}.part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        str(record["url"]), headers={"User-Agent": "FireViewer-MNT-MNS-Rebuild/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size != expected_size:
            raise ValueError(
                f"Downloaded byte count mismatch for {kind}/{record['tile_id']}: "
                f"{temporary.stat().st_size} != {expected_size}"
            )
        actual_hash = sha256_file(temporary)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Downloaded SHA-256 mismatch for {kind}/{record['tile_id']}: "
                f"{actual_hash} != {expected_hash}"
            )
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"kind": kind, "tile_id": record["tile_id"], "state": "downloaded"}


def fetch_sources(
    cache_root: Path,
    records_by_kind: Mapping[str, Mapping[str, Mapping[str, Any]]],
    workers: int,
) -> dict[str, int]:
    tasks = [
        (kind, record)
        for kind in RASTER_KINDS
        for record in records_by_kind[kind].values()
    ]
    states = {"downloaded": 0, "reused": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, cache_root, kind, record): (kind, record["tile_id"])
            for kind, record in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            states[str(result["state"])] += 1
            if completed % 16 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {"phase": "fetch", "complete": completed, "total": len(tasks), **states},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return states


def verify_sources(
    cache_root: Path,
    records_by_kind: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, int]:
    verified = 0
    for kind in RASTER_KINDS:
        for record in records_by_kind[kind].values():
            target = source_target(cache_root, kind, record)
            if not target.is_file():
                raise FileNotFoundError(f"Missing cached source: {target}")
            if target.stat().st_size != int(record["byte_size"]):
                raise ValueError(f"Cached source byte count mismatch: {target}")
            if sha256_file(target) != str(record["sha256"]).lower():
                raise ValueError(f"Cached source SHA-256 mismatch: {target}")
            verified += 1
            if verified % 32 == 0:
                print(
                    json.dumps(
                        {"phase": "verify_sources", "complete": verified, "total": sum(len(records_by_kind[value]) for value in RASTER_KINDS)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return {"downloaded": 0, "reused": verified, "verified": verified}


def bounds_intersect(first: Sequence[float], second: Sequence[float]) -> bool:
    return not (
        float(first[2]) <= float(second[0])
        or float(second[2]) <= float(first[0])
        or float(first[3]) <= float(second[1])
        or float(second[3]) <= float(first[1])
    )


def global_vertices(
    geometry: Any,
    transform: Any,
    origin: Sequence[float],
) -> np.ndarray:
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if not len(vertices):
        return np.empty((0, 3), dtype=np.float64)
    homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
    transformed = homogeneous @ np.asarray(transform, dtype=np.float64).T
    return np.column_stack(
        (
            transformed[:, 0] + float(origin[0]),
            -transformed[:, 2] + float(origin[1]),
            transformed[:, 1] + float(origin[2]),
        )
    )


def exclusion_shapes(package: Path, feature: Mapping[str, Any]) -> list[dict[str, Any]]:
    origin = [float(value) for value in feature["gltf_local_origin_l93_ngf_ign69"]]
    scene = trimesh.load(
        package / str(feature["features"]["path"]), force="scene", process=False
    )
    shapes: list[dict[str, Any]] = []
    for name, geometry in scene.geometry.items():
        lowered = str(name).lower()
        if lowered not in EXCLUDED_GEOMETRY:
            continue
        transform, _ = scene.graph.get(name)
        if lowered == "buildings":
            for component in geometry.split(only_watertight=False):
                points = global_vertices(component, transform, origin)
                if len(points) < 3:
                    continue
                footprint = MultiPoint(points[:, :2]).convex_hull
                if not footprint.is_empty and footprint.area > 0.25:
                    shapes.append(mapping(footprint.buffer(1.0, join_style=2)))
            continue
        points = global_vertices(geometry, transform, origin)
        faces = np.asarray(geometry.faces, dtype=np.int64)
        for face in faces:
            ring = [(float(points[index, 0]), float(points[index, 1])) for index in face]
            if len(ring) >= 3:
                ring.append(ring[0])
                shapes.append({"type": "Polygon", "coordinates": [ring]})
    return shapes


def merge_pair(
    paths: Sequence[Path], processing_bounds: Sequence[float]
) -> tuple[np.ndarray, Any, np.ndarray]:
    datasets = [rasterio.open(path) for path in paths]
    try:
        merged, transform = merge_rasters(
            datasets,
            bounds=tuple(float(value) for value in processing_bounds),
            res=(0.5, 0.5),
            masked=True,
            method="first",
        )
    finally:
        for dataset in datasets:
            dataset.close()
    band = merged[0]
    values = np.asarray(band.filled(np.nan), dtype=np.float32)
    valid = ~np.ma.getmaskarray(band) & np.isfinite(values)
    return values, transform, valid


def save_tile(
    output: Path,
    instances: Sequence[Sequence[float | int]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(instances)
    values = np.asarray(instances, dtype=np.float64) if count else np.empty((0, 7))
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            # Absolute Lambert-93 northings are around 6.4 million metres.
            # Float32 would lose the 0.5 m LiDAR pixel phase at that magnitude.
            positions_l93_m=values[:, :3].astype(np.float64),
            heights_m=values[:, 3].astype(np.float32),
            crown_diameters_m=values[:, 4].astype(np.float32),
            visual_variants=values[:, 5].astype(np.uint8),
            rotations_degrees=values[:, 6].astype(np.float32),
        )
    temporary.replace(output)


def process_tiles(
    *,
    package: Path,
    catalog: Mapping[str, Any],
    cache_root: Path,
    output_root: Path,
    records_by_kind: Mapping[str, Mapping[str, Mapping[str, Any]]],
    halo_m: float,
) -> dict[str, Any]:
    tile_ids = sorted(records_by_kind["mnt"])
    features = list(catalog["feature_tiles"])
    config = SegmentationConfig()
    tile_records: list[dict[str, Any]] = []
    total = 0
    minimum_height = math.inf
    maximum_height = -math.inf
    for ordinal, tile_id in enumerate(tile_ids, start=1):
        record = records_by_kind["mnt"][tile_id]
        core = [float(value) for value in record["bounds_l93_metres"]]
        processing = [core[0] - halo_m, core[1] - halo_m, core[2] + halo_m, core[3] + halo_m]
        neighbours = [
            candidate_id
            for candidate_id, candidate in records_by_kind["mnt"].items()
            if bounds_intersect(processing, candidate["bounds_l93_metres"])
        ]
        mnt_paths = [source_target(cache_root, "mnt", records_by_kind["mnt"][value]) for value in neighbours]
        mns_paths = [source_target(cache_root, "mns", records_by_kind["mns"][value]) for value in neighbours]
        mnt, transform, valid_mnt = merge_pair(mnt_paths, processing)
        mns, mns_transform, valid_mns = merge_pair(mns_paths, processing)
        if tuple(transform) != tuple(mns_transform) or mnt.shape != mns.shape:
            raise ValueError(f"MNT/MNS alignment mismatch for {tile_id}")
        valid = valid_mnt & valid_mns
        shapes: list[dict[str, Any]] = []
        for feature in features:
            if bounds_intersect(processing, feature["bounds_l93_metres"]):
                shapes.extend(exclusion_shapes(package, feature))
        exclusions = (
            rasterize(
                ((shape, 1) for shape in shapes),
                out_shape=mnt.shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype="uint8",
            ).astype(bool)
            if shapes
            else np.zeros(mnt.shape, dtype=bool)
        )
        detected, statistics = segment_vegetation_instances(
            mnt,
            mns,
            transform,
            valid,
            exclusions,
            (0.0, 0.0, 0.0),
            config,
        )
        owned = [
            value
            for value in detected
            if core[0] <= float(value[0]) < core[2]
            and core[1] <= float(value[1]) < core[3]
        ]
        target = output_root / "tiles" / f"{tile_id}.npz"
        save_tile(target, owned)
        heights = [float(value[3]) for value in owned]
        if heights:
            minimum_height = min(minimum_height, min(heights))
            maximum_height = max(maximum_height, max(heights))
        total += len(owned)
        tile_records.append(
            {
                "tile_id": tile_id,
                "zone_id": record["zone_id"],
                "bounds_l93_m": core,
                "source_neighbour_count": len(neighbours),
                "accepted_crown_count": len(owned),
                "local_peak_candidate_count_with_halo": statistics["local_peak_candidate_count"],
                "rejected_small_crown_count_with_halo": statistics["rejected_small_crown_count"],
                "excluded_pixel_count_with_halo": statistics["excluded_pixel_count"],
                "post_detection_spacing_rejected_count": 0,
                "path": target.relative_to(output_root).as_posix(),
                "byte_count": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "segment",
                    "complete": ordinal,
                    "total": len(tile_ids),
                    "tile_id": tile_id,
                    "tile_crowns": len(owned),
                    "cumulative_crowns": total,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    counts = [record["accepted_crown_count"] for record in tile_records]
    return {
        "tree_instances": total,
        "tile_count": len(tile_records),
        "tiles_with_trees": sum(count > 0 for count in counts),
        "dense_tiles_ge_4000": sum(count >= 4000 for count in counts),
        "maximum_instances_on_one_km2_tile": max(counts, default=0),
        "minimum_tree_height_m": minimum_height if math.isfinite(minimum_height) else None,
        "maximum_tree_height_m": maximum_height if math.isfinite(maximum_height) else None,
        "tiles": tile_records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--map-package", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--halo-m", type=float, default=10.0)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    if args.download_workers < 1:
        raise SystemExit("--download-workers must be positive")
    if not math.isfinite(args.halo_m) or args.halo_m <= 0:
        raise SystemExit("--halo-m must be finite and positive")
    source_manifest = args.source_manifest.resolve()
    package = args.map_package.resolve()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    catalog = json.loads((package / "catalog.json").read_text(encoding="utf-8"))
    records_by_kind = {kind: raster_records(manifest, kind) for kind in RASTER_KINDS}
    if set(records_by_kind["mnt"]) != set(records_by_kind["mns"]):
        raise SystemExit("MNT and MNS source tile identifiers do not match")
    if len(records_by_kind["mnt"]) != len(catalog["feature_tiles"]):
        raise SystemExit(
            "Source pair count does not match the base map feature coverage: "
            f"{len(records_by_kind['mnt'])} != {len(catalog['feature_tiles'])}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    fetch = (
        verify_sources(cache_root, records_by_kind)
        if args.skip_download
        else fetch_sources(cache_root, records_by_kind, args.download_workers)
    )
    result = process_tiles(
        package=package,
        catalog=catalog,
        cache_root=cache_root,
        output_root=output_root,
        records_by_kind=records_by_kind,
        halo_m=args.halo_m,
    )
    index = {
        "schema": SCHEMA,
        "base_package_id": "fireviewer-die-pontaix-r1-v4",
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
            "mnt_pair_count": len(records_by_kind["mnt"]),
            "mns_pair_count": len(records_by_kind["mns"]),
            "all_source_hashes_verified": True,
        },
        "method": {
            "ground": "locked_IGN_LiDAR_HD_MNT_0m50",
            "surface": "locked_IGN_LiDAR_HD_MNS_0m50",
            "height": "co_located_MNS_minus_MNT",
            "segmentation": "all_accepted_0m50_crown_apices_without_post_detection_thinning",
            "halo_m": args.halo_m,
            "ownership": "core_min_inclusive_max_exclusive",
            "exclusions": sorted(EXCLUDED_GEOMETRY),
            "completeness_boundary": "detected_crowns_not_field_inventory",
            "config": vars(SegmentationConfig()),
        },
        "fetch": fetch,
        "counts": {key: value for key, value in result.items() if key != "tiles"},
        "tiles": result["tiles"],
    }
    index_path = output_root / "index.json"
    index_path.write_text(canonical_json(index), encoding="utf-8")
    print(canonical_json({"index": str(index_path), "counts": index["counts"]}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
