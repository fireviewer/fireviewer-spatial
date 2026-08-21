#!/usr/bin/env python3
"""Compare a Map Builder output folder with a semantic golden baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def compare(baseline: dict[str, Any], output_root: Path) -> dict[str, Any]:
    validation = _load(output_root / "manifests" / "validation-result.json")
    manifest = _load(output_root / "manifests" / "manifest.json")
    tiled_receipt = _load(
        output_root / "runtime" / "viewer-tiled" / "viewer-tiled-scene.v1.json"
    )
    catalog = _load(output_root / "runtime" / "viewer-tiled" / "catalog.json")
    hashes = _load(output_root / "manifests" / "hashes.json")
    canonical = catalog.get("canonical", {})
    family_counts = canonical.get("family_instance_counts", {})
    source_instance_count = (
        sum(family_counts.values())
        if isinstance(family_counts, dict)
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in family_counts.values()
        )
        else None
    )
    checks = {
        "zone_id": validation.get("zone_id") == baseline.get("zone_id"),
        "tile_count": validation.get("tile_count") == baseline.get("tile_count"),
        "counts": validation.get("counts") == baseline.get("counts"),
        "spatial_reference": (
            manifest.get("spatial_reference") == baseline.get("spatial_reference")
        ),
        "viewer_representation": (
            canonical.get("representation")
            == baseline.get("viewer", {}).get("representation")
        ),
        "viewer_mesh_coverage": (
            (
                "complete"
                if tiled_receipt.get("representation")
                == "complete_tiled_non_simplified_map"
                else None
            )
            == baseline.get("viewer", {}).get("mesh_coverage")
        ),
        "viewer_source_instance_count": (
            source_instance_count
            == baseline.get("viewer", {}).get("source_instance_count")
        ),
        "viewer_external_dependencies": (
            0
            == baseline.get("viewer", {}).get("external_dependencies")
        ),
        "zone_done_last": any(
            artifact.get("path") == "zone.done.json"
            and artifact.get("publication_order") == "last"
            for artifact in hashes.get("artifacts", [])
        ),
        "zone_done_present": (output_root / "zone.done.json").is_file(),
    }
    return {
        "schema": "fireviewer.map-semantic-parity.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "binary_identity_required": False,
        "observed": {
            "build_id": validation.get("build_id"),
            "viewer_catalog_sha256": tiled_receipt.get("catalog", {}).get("sha256"),
            "viewer_catalog_byte_count": tiled_receipt.get("catalog", {}).get(
                "byte_count"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--write")
    args = parser.parse_args()
    result = compare(
        _load(Path(args.baseline).resolve(strict=True)),
        Path(args.output).resolve(strict=True),
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.write:
        destination = Path(args.write).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
