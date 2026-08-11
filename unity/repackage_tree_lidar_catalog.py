"""Repackage an approved remote catalog as LiDAR terrain plus 3D trees only.

The source package is immutable.  This CPU-only tool verifies every referenced
asset, rebuilds each detail container with only its terrain and detected-tree
sections, and writes a new catalog that must still undergo Unity review before
site-upload packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from fwtile import build_container, canonical_json_bytes, read_container, sha256_file


MODE = "trees_3d_lidar_infrastructure"
SCHEMA = "fireviewer.remote-tile-catalog.v1"
DETAIL_SECTIONS = ("terrain", "trees")
MAXIMUM_BOUNDING_AREA_KM2 = 250.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _source_asset(root: Path, reference: Mapping[str, Any]) -> Path:
    relative = str(reference.get("path") or reference.get("url") or "")
    parsed = PurePosixPath(relative)
    if not relative or parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
        raise ValueError(f"unsafe source asset path: {relative!r}")
    path = root.joinpath(*parsed.parts)
    if not path.is_file() or path.stat().st_size != int(
        reference.get("byte_count", -1)
    ):
        raise ValueError(f"source asset is absent or has a different size: {relative}")
    if sha256_file(path) != str(reference.get("sha256", "")):
        raise ValueError(f"source asset checksum differs: {relative}")
    return path


def _link_asset(
    source: Path, destination: Path, output_root: Path, reference: Mapping[str, Any]
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256_file(
            destination
        ) != str(reference["sha256"]):
            raise ValueError(f"immutable output collision: {destination}")
    else:
        os.link(source, destination)
    return {
        "url": destination.relative_to(output_root).as_posix(),
        "sha256": str(reference["sha256"]),
        "byte_count": int(reference["byte_count"]),
        **(
            {"resolution_m": reference["resolution_m"]}
            if "resolution_m" in reference
            else {}
        ),
    }


def _publish_payload(
    output_root: Path, tile: Mapping[str, Any], source: Path
) -> dict[str, Any]:
    parsed = read_container(source.read_bytes(), decode_sections=True)
    header = parsed["header"]
    metadata_by_name = {item["name"]: item["metadata"] for item in header["sections"]}
    if not all(
        name in parsed["sections"] and name in metadata_by_name
        for name in DETAIL_SECTIONS
    ):
        raise ValueError(f"tile {tile.get('id')} does not contain terrain and trees")
    payload = build_container(
        kind="detail_tile",
        tile_id=str(tile["id"]),
        bounds_l93_m=tile["bounds_l93_m"],
        origin_l93_m=header["origin_l93_m"],
        sections=[
            (name, parsed["sections"][name], metadata_by_name[name])
            for name in DETAIL_SECTIONS
        ],
        metadata={
            "content_mode": MODE,
            "infrastructure_representation": "lidar_surface_only",
            "source_payload_sha256": sha256_file(source),
        },
    )
    digest = hashlib.sha256(payload).hexdigest()
    relative = (
        PurePosixPath("detail") / str(tile["id"]) / f"{tile['id']}.{digest[:16]}.fwtile"
    )
    path = output_root.joinpath(*relative.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != digest:
            raise ValueError(f"immutable output collision: {path}")
    else:
        path.write_bytes(payload)
    return {"url": relative.as_posix(), "sha256": digest, "byte_count": len(payload)}


def repackage(*, source_root: Path, output_root: Path) -> dict[str, Any]:
    source_catalog_path = source_root / "catalog.json"
    source_catalog = _read_json(source_catalog_path)
    if source_catalog.get("schema") != SCHEMA:
        raise ValueError("source is not a FireViewer remote catalog")
    if source_catalog.get("source", {}).get("terrain_surface") != "mns_lidar":
        raise ValueError(
            "source catalog has no verified MNS LiDAR terrain surface; rebuild from the checked MNT+MNS sources before using the trees_3d_lidar_infrastructure mode"
        )
    tiles = source_catalog.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("source catalog contains no detail tiles")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    try:
        far_source = source_catalog["lod_policy"]["far"]
        bounds = [float(value) for value in far_source["bounds_l93_m"]]
        bounding_area_km2 = (
            (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / 1_000_000.0
        )
        if bounding_area_km2 > MAXIMUM_BOUNDING_AREA_KM2:
            raise ValueError(
                f"{MODE} is limited to {MAXIMUM_BOUNDING_AREA_KM2:g} km2; source bounds cover {bounding_area_km2:.3f} km2"
            )
        far: dict[str, Any] = {
            "role": "always_available_global_fallback",
            "bounds_l93_m": far_source["bounds_l93_m"],
        }
        for name in ("terrain", "imagery"):
            reference = far_source[name]
            source = _source_asset(source_root, reference)
            far[name] = _link_asset(
                source, staging / "far" / source.name, staging, reference
            )
        result_tiles: list[dict[str, Any]] = []
        for tile in tiles:
            if not isinstance(tile, dict):
                raise ValueError("source catalog tile is invalid")
            source_payload = _source_asset(source_root, tile["payload"])
            payload = _publish_payload(staging, tile, source_payload)
            source_imagery = _source_asset(source_root, tile["imagery"])
            imagery = _link_asset(
                source_imagery,
                staging / "imagery" / source_imagery.name,
                staging,
                tile["imagery"],
            )
            decoded = read_container(source_payload.read_bytes(), decode_sections=False)
            tree_header = next(
                item
                for item in decoded["header"]["sections"]
                if item["name"] == "trees"
            )
            result_tiles.append(
                {
                    "id": tile["id"],
                    "bounds_l93_m": tile["bounds_l93_m"],
                    "payload": payload,
                    "imagery": imagery,
                    "counts": {
                        "terrain_vertices": tile.get("counts", {}).get(
                            "terrain_vertices", 0
                        ),
                        "trees": tree_header["metadata"].get("count", 0),
                        "buildings": 0,
                        "road_triangles": 0,
                        "water_triangles": 0,
                    },
                    "sections": list(DETAIL_SECTIONS),
                }
            )
        catalog = {
            "schema": SCHEMA,
            "catalog_version": 1,
            "crs": source_catalog["crs"],
            "linear_unit": source_catalog["linear_unit"],
            "origin_l93_m": source_catalog["origin_l93_m"],
            "axes": source_catalog.get("axes", {}),
            "lod_policy": {
                "far": far,
                "detail": {
                    "enabled": True,
                    "mode": "streamed_detail",
                    "content_mode": MODE,
                    "publish_distance_m": 600.0,
                    "preload_radius_m": 750.0,
                    "maximum_resident_tile_count": 16,
                    "near_disabled": False,
                    "transition": "global_fallback_then_atomic_detail_footprint",
                    "eviction": "least_priority_outside_desired_footprint",
                },
            },
            "source": {
                "repackaged_from_catalog_sha256": sha256_file(source_catalog_path),
                "content_mode": MODE,
                "infrastructure_representation": "lidar_surface_only",
            },
            "exported_detail_tile_count": len(result_tiles),
            "tiles": result_tiles,
        }
        (staging / "catalog.json").write_bytes(canonical_json_bytes(catalog) + b"\n")
        receipt = {
            "schema": "fireviewer.tree-lidar-repackage.v1",
            "mode": MODE,
            "fixed_maximum_bounding_area_km2": MAXIMUM_BOUNDING_AREA_KM2,
            "source_bounds_l93_m": bounds,
            "source_bounding_area_km2": bounding_area_km2,
            "source_catalog_sha256": sha256_file(source_catalog_path),
            "output_catalog_sha256": sha256_file(staging / "catalog.json"),
            "detail_tile_count": len(result_tiles),
            "gpu_used": False,
            "requires_manual_unity_review": True,
        }
        (staging / "repackage-receipt.json").write_bytes(
            canonical_json_bytes(receipt) + b"\n"
        )
        os.replace(staging, output_root)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            repackage(
                source_root=args.source_root.resolve(),
                output_root=args.output_root.resolve(),
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
