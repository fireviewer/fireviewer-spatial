from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from threading import Thread


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from fwtile import (  # noqa: E402
    build_container,
    build_vector_sections,
    encode_detail_terrain,
    encode_far_grid,
    encode_tree_instances,
    sha256_bytes,
)
from test_fwtile import _detail_vectors, _package  # noqa: E402


def _detail_container_with_nonzero_origin() -> bytes:
    package = _package()
    package["metadata"]["origin_l93_m"] = [0.0, 0.0, 320.0]
    bounds = package["metadata"]["bounds_l93_m"]
    origin = package["metadata"]["origin_l93_m"]
    terrain = encode_detail_terrain(package["terrain"], bounds, origin)
    trees = encode_tree_instances(package, bounds, origin)
    vectors = build_vector_sections(_detail_vectors(), bounds, origin)
    return build_container(
        kind="detail_tile",
        tile_id="x0_y0_s2",
        bounds_l93_m=bounds,
        origin_l93_m=origin,
        sections=[
            ("terrain", *terrain),
            ("trees", *trees),
            ("buildings", *vectors["buildings"]),
            ("roads", *vectors["roads"]),
            ("water", *vectors["water"]),
        ],
    )


def test_dotnet_bridge_fetches_catalog_and_decodes_meshes_and_all_trees(
    tmp_path: Path,
) -> None:
    payload = _detail_container_with_nonzero_origin()
    digest = sha256_bytes(payload)
    payload_relative = f"detail/x0_y0_s2/x0_y0_s2.{digest}.fwtile"
    payload_path = tmp_path / Path(*payload_relative.split("/"))
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(payload)

    far_raw, far_metadata = encode_far_grid(
        [10.0, 11.0, 12.0, 13.0],
        [True, True, True, True],
        rows=2,
        columns=2,
        pixel_size_m=[1.0, 1.0],
        outer_bounds_l93_m=[0.0, 0.0, 2.0, 2.0],
    )
    far_payload = build_container(
        kind="global_far_terrain",
        tile_id="global-far",
        bounds_l93_m=[0.0, 0.0, 2.0, 2.0],
        origin_l93_m=[0.0, 0.0, 320.0],
        sections=[("terrain", far_raw, far_metadata)],
    )
    far_digest = sha256_bytes(far_payload)
    far_relative = f"far/global-mnt.{far_digest}.fwterrain"
    far_path = tmp_path / Path(*far_relative.split("/"))
    far_path.parent.mkdir(parents=True)
    far_path.write_bytes(far_payload)

    unused_hash = "a" * 64
    catalog = {
        "schema": "fireviewer.remote-tile-catalog.v1",
        "catalog_version": 1,
        "crs": "EPSG:2154",
        "linear_unit": "metre",
        "origin_l93_m": [0.0, 0.0, 320.0],
        "lod_policy": {
            "far": {
                "role": "always_available_global_fallback",
                "terrain": {
                    "url": far_relative,
                    "sha256": far_digest,
                    "byte_count": len(far_payload),
                    "resolution_m": [5.0, 5.0],
                },
                "imagery": {
                    "url": "far/unused.jpg",
                    "sha256": unused_hash,
                    "byte_count": 1,
                    "resolution_m": 2.0,
                },
                "bounds_l93_m": [0.0, 0.0, 2.0, 2.0],
            },
            "detail": {
                "publish_distance_m": 600.0,
                "preload_radius_m": 750.0,
                "maximum_resident_tile_count": 16,
                "near_disabled": True,
                "transition": "global_fallback_then_atomic_detail_footprint",
                "eviction": "least_priority_outside_desired_footprint",
            },
        },
        "exported_detail_tile_count": 1,
        "tiles": [
            {
                "id": "x0_y0_s2",
                "bounds_l93_m": [0.0, 0.0, 2.0, 2.0],
                "payload": {
                    "url": payload_relative,
                    "sha256": digest,
                    "byte_count": len(payload),
                },
                "imagery": {
                    "url": "imagery/unused.jpg",
                    "sha256": unused_hash,
                    "byte_count": 1,
                    "resolution_m": 0.2,
                },
                "sections": ["terrain", "trees", "buildings", "roads", "water"],
            }
        ],
    }
    (tmp_path / "catalog.json").write_text(
        json.dumps(catalog, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    project = MODULE_DIRECTORY / "dotnet-bridge-probe/FireViewer.TileProbe.csproj"
    subprocess.run(
        [
            "dotnet",
            "build",
            str(project),
            "--ignore-failed-sources",
            "-p:NuGetAudit=false",
            "--nologo",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        catalog_url = f"http://127.0.0.1:{server.server_port}/catalog.json"
        completed = subprocess.run(
            [
                "dotnet",
                "run",
                "--project",
                str(project),
                "--no-build",
                "--no-restore",
                "--",
                catalog_url,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    proof = json.loads(completed.stdout.strip().splitlines()[-1])
    assert proof == {
        "schema": "fireviewer.remote-tile-catalog.v1",
        "tile_id": "x0_y0_s2",
        "payload_url": f"http://127.0.0.1:{server.server_port}/{payload_relative}",
        "payload_sha256": digest,
        "terrain_vertices": 9,
        "terrain_triangles": 8,
        "terrain_first_y": 100.0,
        "trees": 2,
        "tree_first_y": 101.25,
        "building_meshes": 1,
        "road_meshes": 1,
        "water_meshes": 1,
        "far_terrain_vertices": 4,
        "far_terrain_triangles": 2,
        "far_terrain_first_y": 10.0,
        "near_750_tiles": 1,
        "mid_3000_tiles": 1,
        "far_over_3000_tiles": 0,
        "unordered_footprint_intersects": True,
        "margin_50_includes": True,
        "margin_49_9_excludes": True,
        "visible_beyond_radius_tiles": 1,
        "mid_visible_footprint_tiles": 1,
        "band_750": "near",
        "band_over_750": "mid",
        "band_3000": "mid",
        "band_over_3000": "far",
        "catalog_near_disabled": True,
        "band_300_near_disabled": "mid",
        "budget_clamped_tiles": 16,
        "budget_clamped_without_blocking": True,
        "partial_published": False,
        "complete_published": True,
        "far_during_partial": True,
        "far_after_publication": False,
        "far_after_failure": True,
    }
