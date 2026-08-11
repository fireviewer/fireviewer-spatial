"""Build portable OpenUSD map wrappers for FireViewer spatial packages.

The exporter deliberately keeps the original FireViewer package untouched.  A
USD package contains a terrain mesh sampled from the authoritative FAR MNT,
the original orthophoto as a hard-linked texture, a stage wrapper and complete
provenance.  It does not manufacture buildings, trees or near-detail that are
absent from a global-only source package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unity"))

from fwtile import read_container  # noqa: E402


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def usd_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_mesh_layer(
    output: Path,
    *,
    elevations: array,
    valid: bytes,
    rows: int,
    columns: int,
    bounds: list[float],
    quantization: dict[str, Any],
    maximum_grid: int,
    texture_asset: str,
) -> dict[str, Any]:
    """Write a regular, decimated Z-up terrain mesh in local metric coordinates."""

    target_columns = min(columns, maximum_grid)
    target_rows = min(rows, maximum_grid)
    column_indices = [round(index * (columns - 1) / (target_columns - 1)) for index in range(target_columns)]
    row_indices = [round(index * (rows - 1) / (target_rows - 1)) for index in range(target_rows)]
    xmin, ymin, xmax, ymax = map(float, bounds)
    centre_x = (xmin + xmax) / 2.0
    centre_y = (ymin + ymax) / 2.0
    step = float(quantization["step_m"])
    minimum = float(quantization["minimum_m"])
    point_valid: list[bool] = []
    points: list[tuple[float, float, float]] = []
    st: list[tuple[float, float]] = []
    for target_row, source_row in enumerate(row_indices):
        north = ymax - (ymax - ymin) * source_row / max(1, rows - 1)
        for target_column, source_column in enumerate(column_indices):
            east = xmin + (xmax - xmin) * source_column / max(1, columns - 1)
            source_index = source_row * columns + source_column
            is_valid = bool(valid[source_index // 8] & (1 << (source_index % 8)))
            elevation = minimum + elevations[source_index] * step if is_valid else minimum
            point_valid.append(is_valid)
            points.append((east - centre_x, north - centre_y, elevation))
            st.append((target_column / max(1, target_columns - 1), 1.0 - target_row / max(1, target_rows - 1)))
    face_counts: list[int] = []
    indices: list[int] = []
    for row in range(target_rows - 1):
        for column in range(target_columns - 1):
            nw = row * target_columns + column
            quad = (nw, nw + 1, nw + target_columns + 1, nw + target_columns)
            if all(point_valid[index] for index in quad):
                face_counts.append(4)
                indices.extend(quad)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("#usda 1.0\n(\n    defaultPrim = \"Terrain\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n\n")
        stream.write('def Xform "Terrain" (\n    kind = "component"\n)\n{\n')
        stream.write('    custom string fireviewer:crs = "EPSG:2154"\n')
        stream.write('    custom double3 fireviewer:origin_l93_m = (%.6f, %.6f, 0)\n' % (centre_x, centre_y))
        stream.write('    def Mesh "GlobalMnt"\n    {\n')
        stream.write("        int[] faceVertexCounts = [" + ", ".join(map(str, face_counts)) + "]\n")
        stream.write("        int[] faceVertexIndices = [" + ", ".join(map(str, indices)) + "]\n")
        stream.write("        point3f[] points = [\n")
        for x, y, z in points:
            stream.write("            (%.3f, %.3f, %.3f),\n" % (x, y, z))
        stream.write("        ]\n")
        stream.write("        texCoord2f[] primvars:st = [\n")
        for u, v in st:
            stream.write("            (%.8f, %.8f),\n" % (u, v))
        stream.write("        ] (\n            interpolation = \"vertex\"\n        )\n")
        stream.write('        uniform token subdivisionScheme = "none"\n')
        stream.write('        rel material:binding = </Terrain/Materials/Orthophoto>\n    }\n')
        stream.write('    def Scope "Materials"\n    {\n')
        stream.write('        def Material "Orthophoto"\n        {\n')
        stream.write('            token outputs:surface.connect = </Terrain/Materials/Orthophoto/Preview.outputs:surface>\n')
        stream.write('            def Shader "Preview"\n            {\n                uniform token info:id = "UsdPreviewSurface"\n                color3f inputs:diffuseColor.connect = </Terrain/Materials/Orthophoto/Texture.outputs:rgb>\n                float inputs:roughness = 0.92\n                token outputs:surface\n            }\n')
        stream.write('            def Shader "Texture"\n            {\n                uniform token info:id = "UsdUVTexture"\n')
        stream.write('                asset inputs:file = @%s@\n' % usd_string(texture_asset))
        stream.write('                float2 inputs:st.connect = </Terrain/Materials/Orthophoto/StReader.outputs:result>\n                float3 outputs:rgb\n            }\n')
        stream.write('            def Shader "StReader"\n            {\n                uniform token info:id = "UsdPrimvarReader_float2"\n                token inputs:varname = "st"\n                float2 outputs:result\n            }\n')
        stream.write("        }\n    }\n}\n")
    return {
        "source_grid": {"rows": rows, "columns": columns},
        "mesh_grid": {"rows": target_rows, "columns": target_columns},
        "vertex_count": len(points),
        "quad_count": len(face_counts),
        "origin_l93_m": [centre_x, centre_y, 0.0],
    }


def build_spatial_package(site_package: Path, output_root: Path, maximum_grid: int) -> dict[str, Any]:
    manifest_path = site_package / "package-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    catalog = json.loads((site_package / "catalog.json").read_text(encoding="utf-8"))
    package_id = str(manifest.get("package_id") or site_package.parent.name)
    output = output_root / package_id
    output.mkdir(parents=True, exist_ok=True)
    far = catalog["lod_policy"]["far"]
    detail_policy = catalog["lod_policy"].get("detail", {})
    terrain_reference = far["terrain"].get("path") or far["terrain"].get("url")
    imagery_reference = far["imagery"].get("path") or far["imagery"].get("url")
    if not isinstance(terrain_reference, str) or not isinstance(imagery_reference, str):
        raise ValueError("catalog FAR terrain and imagery must each provide path or url")
    terrain_source = site_package / terrain_reference
    image_source = site_package / imagery_reference
    terrain_container = read_container(terrain_source.read_bytes())
    section = terrain_container["header"]["sections"][0]
    section_name = str(section["name"])
    raw = terrain_container["sections"][section_name]
    metadata = section["metadata"]
    elevation_bytes = int(metadata["elevation_bytes"])
    elevations = array("H")
    elevations.frombytes(raw[:elevation_bytes])
    if sys.byteorder != "little":
        elevations.byteswap()
    valid = raw[int(metadata["validity_mask_offset_bytes"]) :]
    link_mode = asset_link(image_source, output / "assets" / image_source.name)
    mesh = write_mesh_layer(
        output / "terrain.usda",
        elevations=elevations,
        valid=valid,
        rows=int(metadata["rows"]),
        columns=int(metadata["columns"]),
        bounds=[float(value) for value in metadata["outer_bounds_l93_m"]],
        quantization=metadata["elevation_quantization"],
        maximum_grid=maximum_grid,
        texture_asset="assets/" + image_source.name,
    )
    (output / "scene.usda").write_text(
        "#usda 1.0\n(\n    defaultPrim = \"FireViewerMap\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n    subLayers = [@terrain.usda@]\n)\n\ndef Xform \"FireViewerMap\"\n{\n"
        + '    custom string fireviewer:package_id = "' + usd_string(package_id) + '"\n'
        + '    custom string fireviewer:profile = "global_mnt_only"\n'
        + '    custom string fireviewer:crs = "EPSG:2154"\n'
        + "}\n",
        encoding="utf-8",
    )
    record = {
        "schema": "fireviewer.omniverse-map.v1",
        "kind": "spatial_map",
        "package_id": package_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_site_package": str(site_package),
        "source_catalog_sha256": sha256_file(site_package / "catalog.json"),
        "terrain": {"source": str(terrain_source), "sha256": sha256_file(terrain_source), **mesh},
        "orthophoto": {"source": str(image_source), "sha256": sha256_file(image_source), "delivery": link_mode},
        "lod": {
            "source_mode": detail_policy.get("mode", "unknown"),
            "source_detail_tile_count": int(catalog.get("exported_detail_tile_count", 0)),
            "generated_detail_tile_count": 0,
            "conversion": "global_surface_only",
        },
        "entry_stage": "scene.usda",
    }
    (output / "manifest.json").write_text(canonical_json(record), encoding="utf-8")
    return record


def build_evidence_package(source_archive: Path, output_root: Path) -> dict[str, Any]:
    """Preserve evidence archives as USD-readable incident context, not fake terrain."""

    package_id = source_archive.name.removesuffix(".source.zip").replace("_", "-")
    output = output_root / package_id
    output.mkdir(parents=True, exist_ok=True)
    archive_name = source_archive.name
    delivery = asset_link(source_archive, output / "sources" / archive_name)
    record = {
        "schema": "fireviewer.omniverse-map.v1",
        "kind": "incident_evidence_only",
        "package_id": package_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_archive": {"path": str(source_archive), "sha256": sha256_file(source_archive), "delivery": delivery},
        "spatial_content": "absent_from_source_archive",
        "entry_stage": "scene.usda",
    }
    (output / "manifest.json").write_text(canonical_json(record), encoding="utf-8")
    (output / "scene.usda").write_text(
        "#usda 1.0\n(\n    defaultPrim = \"IncidentEvidence\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n\ndef Xform \"IncidentEvidence\"\n{\n"
        + '    custom string fireviewer:package_id = "' + usd_string(package_id) + '"\n'
        + '    custom string fireviewer:source_archive = "sources/' + usd_string(archive_name) + '"\n'
        + '    custom string fireviewer:spatial_content = "absent_from_source_archive"\n'
        + "}\n",
        encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-package", type=Path, action="append", default=[])
    parser.add_argument("--source-archive", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-grid", type=int, default=1025)
    args = parser.parse_args()
    if args.maximum_grid < 2:
        raise SystemExit("--maximum-grid must be at least 2")
    if not args.site_package and not args.source_archive:
        raise SystemExit("at least one package or archive is required")
    output_root = args.output_root.resolve()
    records = []
    for package in args.site_package:
        records.append(build_spatial_package(package.resolve(), output_root, args.maximum_grid))
    for archive in args.source_archive:
        records.append(build_evidence_package(archive.resolve(), output_root))
    records = []
    for manifest_path in sorted(output_root.glob("*/manifest.json")):
        records.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    index = {"schema": "fireviewer.omniverse-index.v1", "generated_at": datetime.now(timezone.utc).isoformat(), "packages": records}
    (output_root / "index.json").write_text(canonical_json(index), encoding="utf-8")
    print(canonical_json(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
