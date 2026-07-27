from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from export_site_upload_package import build_site_package, validate_site_package  # noqa: E402


_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _asset(root: Path, relative: str, payload: bytes) -> dict[str, object]:
    path = root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "url": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
    }


def _manual_validation(
    tmp_path: Path,
    *,
    source: Path,
    decision: str = "accepted",
    package_id: str = "pkg-synthetic-zone-r1-v1",
    zone_id: str = "SYNTHETIC-ZONE-01",
) -> tuple[Path, Path]:
    preview = tmp_path / "unity-preview.png"
    preview.write_bytes(_PNG)
    receipt = tmp_path / "unity-validation.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "fireviewer.unity-manual-validation.v1",
                "decision": decision,
                "approval_statement": (
                    "ACCEPTÉ POUR PUBLICATION"
                    if decision == "accepted"
                    else "REFUSÉ POUR PUBLICATION"
                ),
                "package_id": package_id,
                "zone_id": zone_id,
                "revision": 1,
                "reviewer": "Manual Unity reviewer",
                "reviewed_at_utc": "2026-07-17T12:00:00Z",
                "unity_version": "6000.3.18f1",
                "catalog_sha256": hashlib.sha256(
                    (source / "catalog.json").read_bytes()
                ).hexdigest(),
                "preview_sha256": hashlib.sha256(_PNG).hexdigest(),
                "checklist": {
                    "catalog_loaded": True,
                    "terrain_grounding": True,
                    "vegetation_exclusions": True,
                    "lod_streaming": True,
                    "detail_buildings": True,
                    "no_blocking_visual_artifacts": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return receipt, preview


def test_builds_exact_hardlinked_site_upload_inventory(tmp_path: Path) -> None:
    source = tmp_path / "remote"
    source.mkdir()
    catalog = {
        "schema": "fireviewer.remote-tile-catalog.v1",
        "crs": "EPSG:2154",
        "linear_unit": "metre",
        "origin_l93_m": [885_173.0, 6_404_926.0, 320.0],
        "exported_detail_tile_count": 1,
        "lod_policy": {
            "far": {
                "imagery": _asset(source, "far/global.jpg", b"far-image"),
                "terrain": _asset(source, "far/global.fwterrain", b"far-terrain"),
            }
        },
        "tiles": [
            {
                "id": "x0_y0_s500",
                "imagery": _asset(source, "imagery/tile.jpg", b"tile-image"),
                "payload": _asset(source, "detail/tile/tile.fwtile", b"tile-payload"),
            }
        ],
    }
    (source / "catalog.json").write_text(
        json.dumps(catalog, separators=(",", ":")), encoding="utf-8"
    )
    output = tmp_path / "upload" / "pkg-synthetic-zone-r1-v1"
    receipt, preview = _manual_validation(tmp_path, source=source)

    report = build_site_package(
        source_root=source,
        output_root=output,
        package_id="pkg-synthetic-zone-r1-v1",
        zone_id="SYNTHETIC-ZONE-01",
        revision=1,
        unity_validation_receipt=receipt,
        unity_preview_png=preview,
    )

    assert report["status"] == "valid"
    assert report["asset_count"] == 5
    assert report["file_count"] == 7
    assert validate_site_package(output)["status"] == "valid"
    packaged_catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert packaged_catalog["tiles"][0]["payload"]["url"].startswith("assets/detail/")
    assert packaged_catalog["tiles"][0]["payload"]["path"].startswith("assets/detail/")
    assert packaged_catalog["validation"]["unity_preview"]["path"] == (
        "assets/validation/unity-preview.png"
    )
    packaged_manifest = json.loads(
        (output / "package-manifest.json").read_text(encoding="utf-8")
    )
    assert packaged_manifest["manual_unity_validation"]["decision"] == "accepted"
    source_stat = os.stat(source / "detail/tile/tile.fwtile")
    target_stat = os.stat(output / "assets/detail/tile/tile.fwtile")
    assert source_stat.st_ino == target_stat.st_ino


def test_builds_global_only_site_upload_with_no_detail_assets(tmp_path: Path) -> None:
    source = tmp_path / "remote"
    source.mkdir()
    catalog = {
        "schema": "fireviewer.remote-tile-catalog.v1",
        "crs": "EPSG:2154",
        "linear_unit": "metre",
        "origin_l93_m": [885_173.0, 6_404_926.0, 320.0],
        "exported_detail_tile_count": 0,
        "lod_policy": {
            "far": {
                "imagery": _asset(source, "far/global.jpg", b"far-image"),
                "terrain": _asset(source, "far/global.fwterrain", b"far-terrain"),
            },
            "detail": {
                "enabled": False,
                "mode": "global_only",
                "maximum_resident_tile_count": 0,
                "near_disabled": True,
            },
        },
        "tiles": [],
    }
    (source / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    receipt, preview = _manual_validation(
        tmp_path,
        source=source,
        package_id="pkg-global-r1-v1",
        zone_id="CAP-FERRET-01",
    )

    report = build_site_package(
        source_root=source,
        output_root=tmp_path / "upload" / "pkg-global-r1-v1",
        package_id="pkg-global-r1-v1",
        zone_id="CAP-FERRET-01",
        revision=1,
        unity_validation_receipt=receipt,
        unity_preview_png=preview,
    )

    assert report["status"] == "valid"
    assert report["asset_count"] == 3


def test_refuses_site_upload_without_an_accepted_current_unity_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "remote"
    source.mkdir()
    catalog = {
        "schema": "fireviewer.remote-tile-catalog.v1",
        "exported_detail_tile_count": 1,
        "lod_policy": {
            "far": {
                "imagery": _asset(source, "far/global.jpg", b"far-image"),
                "terrain": _asset(source, "far/global.fwterrain", b"far-terrain"),
            }
        },
        "tiles": [
            {
                "imagery": _asset(source, "imagery/tile.jpg", b"tile-image"),
                "payload": _asset(source, "detail/tile/tile.fwtile", b"tile-payload"),
            }
        ],
    }
    (source / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    receipt, preview = _manual_validation(tmp_path, source=source, decision="rejected")

    with pytest.raises(ValueError, match="decision is not accepted"):
        build_site_package(
            source_root=source,
            output_root=tmp_path / "upload",
            package_id="pkg-synthetic-zone-r1-v1",
            zone_id="SYNTHETIC-ZONE-01",
            revision=1,
            unity_validation_receipt=receipt,
            unity_preview_png=preview,
        )

    receipt, preview = _manual_validation(tmp_path, source=source)
    preview.write_bytes(_PNG + b"stale")
    with pytest.raises(ValueError, match="preview_sha256 does not match"):
        build_site_package(
            source_root=source,
            output_root=tmp_path / "upload",
            package_id="pkg-synthetic-zone-r1-v1",
            zone_id="SYNTHETIC-ZONE-01",
            revision=1,
            unity_validation_receipt=receipt,
            unity_preview_png=preview,
        )
