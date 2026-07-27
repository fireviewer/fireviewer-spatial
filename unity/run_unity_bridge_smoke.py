from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
from threading import Thread

from fwtile import (
    build_container,
    build_vector_sections,
    encode_detail_terrain,
    encode_tree_instances,
    sha256_bytes,
)
from test_fwtile import _detail_vectors, _package


ROOT = Path(__file__).resolve().parent
UNITY = Path(r"C:\Program Files\Unity\Hub\Editor\6000.3.18f1\Editor\Unity.exe")
FIXTURE = ROOT / "unity-bridge-smoke-http"
PROJECT = ROOT / "unity-bridge-smoke"
LOG = ROOT / "unity-bridge-smoke.log"
SCENE_LOG = ROOT / "unity-bridge-scene-builder.log"
GENERATED_SCENE_DIRECTORY = PROJECT / "Assets/FireViewerSpatial"
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000200000002080600000072b60d24"
    "0000001549444154789c634ce8f1f8cfc0c0c0c0042240180021800237b55b861b"
    "0000000049454e44ae426082"
)


def _write_fixture() -> None:
    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)
    package = _package()
    package["metadata"]["origin_l93_m"] = [0.0, 0.0, 320.0]
    package["instances"]["values"] = [
        # Inside the building only (outside the road and water triangles).
        [1.1, 0.8, 101.25, 8.125, 0.2, 1, 30.25],
        # Outside all three masks: this is the single rendered tree.
        [1.5, 0.75, 105.4, 12.0, 0.2, 3, 359.99],
    ]
    bounds = package["metadata"]["bounds_l93_m"]
    origin = package["metadata"]["origin_l93_m"]
    terrain = encode_detail_terrain(package["terrain"], bounds, origin)
    trees = encode_tree_instances(package, bounds, origin)
    vectors = build_vector_sections(_detail_vectors(), bounds, origin)
    payload = build_container(
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
    payload_sha = sha256_bytes(payload)
    payload_url = f"detail/x0_y0_s2/x0_y0_s2.{payload_sha}.fwtile"
    payload_path = FIXTURE / Path(*payload_url.split("/"))
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(payload)
    image_sha = sha256_bytes(PNG_1X1)
    image_url = f"imagery/x0_y0_s2.{image_sha}.png"
    image_path = FIXTURE / Path(*image_url.split("/"))
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(PNG_1X1)
    unused = "a" * 64
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
                    "url": "far/unused.fwterrain",
                    "sha256": unused,
                    "byte_count": 1,
                    "resolution_m": [5.0, 5.0],
                },
                "imagery": {
                    "url": "far/unused.png",
                    "sha256": unused,
                    "byte_count": 1,
                    "resolution_m": 2.0,
                },
                "bounds_l93_m": [0.0, 0.0, 2.0, 2.0],
            },
            "detail": {
                "publish_distance_m": 600.0,
                "preload_radius_m": 750.0,
                "maximum_resident_tile_count": 16,
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
                    "url": payload_url,
                    "sha256": payload_sha,
                    "byte_count": len(payload),
                },
                "imagery": {
                    "url": image_url,
                    "sha256": image_sha,
                    "byte_count": len(PNG_1X1),
                    "resolution_m": 0.2,
                },
                "sections": ["terrain", "trees", "buildings", "roads", "water"],
            }
        ],
    }
    (FIXTURE / "catalog.json").write_text(
        json.dumps(catalog, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def main() -> int:
    if not UNITY.is_file():
        raise FileNotFoundError(UNITY)
    _write_fixture()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(FIXTURE))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        catalog_url = f"http://127.0.0.1:{server.server_port}/catalog.json"
        completed = subprocess.run(
            [
                str(UNITY),
                "-batchmode",
                "-nographics",
                "-quit",
                "-projectPath",
                str(PROJECT),
                "-executeMethod",
                "FireViewerBridgeSmoke.Run",
                "-catalogUrl",
                catalog_url,
                "-logFile",
                str(LOG),
            ],
            timeout=300,
            check=False,
        )
        scene_completed = subprocess.run(
            [
                str(UNITY),
                "-batchmode",
                "-nographics",
                "-quit",
                "-projectPath",
                str(PROJECT),
                "-executeMethod",
                "FireViewer.SpatialTiles.Editor.FwSpatialDemoSceneBuilder.CreateFromCommandLine",
                "-fireviewerCatalogUrl",
                catalog_url,
                "-logFile",
                str(SCENE_LOG),
            ],
            timeout=300,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    log = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    scene_log = (
        SCENE_LOG.read_text(encoding="utf-8", errors="replace")
        if SCENE_LOG.exists()
        else ""
    )
    if completed.returncode != 0 or "FIREVIEWER_UNITY_HTTP_BRIDGE_OK" not in log:
        print(log[-12000:])
        return completed.returncode or 1
    if (
        scene_completed.returncode != 0
        or "FIREVIEWER_SPATIAL_DEMO_CREATED" not in scene_log
        or not (GENERATED_SCENE_DIRECTORY / "FireViewerSpatialDemo.unity").is_file()
    ):
        print(scene_log[-12000:])
        return scene_completed.returncode or 1
    print(
        next(
            line
            for line in log.splitlines()
            if "FIREVIEWER_UNITY_HTTP_BRIDGE_OK" in line
        )
    )
    print(
        next(
            line
            for line in scene_log.splitlines()
            if "FIREVIEWER_SPATIAL_DEMO_CREATED" in line
        )
    )
    shutil.rmtree(GENERATED_SCENE_DIRECTORY)
    generated_meta = GENERATED_SCENE_DIRECTORY.with_suffix(".meta")
    if generated_meta.exists():
        generated_meta.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
