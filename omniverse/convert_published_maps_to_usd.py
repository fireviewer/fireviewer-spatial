"""Convert every published FireViewer map package into a portable OpenUSD package.

The converter consumes the real static site package (COG elevation, colour
imagery and GLB feature tiles). It writes a self-contained USD stage made of
terrain and feature payload layers. Source packages are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
import trimesh
from rasterio.enums import Resampling


IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]" )
EXCLUDED_PARTS = {".git", "node_modules", ".pytest_cache", "__pycache__", "temp"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def usd_identifier(value: str) -> str:
    result = IDENTIFIER_RE.sub("_", value)
    if not result or result[0].isdigit():
        result = "_" + result
    return result


def usd_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def format_float(value: float) -> str:
    if not math.isfinite(value):
        value = 0.0
    return f"{value:.4f}"


def format_vec(values: Iterable[float]) -> str:
    return "(" + ", ".join(format_float(float(value)) for value in values) + ")"


def write_values(stream: Any, values: Iterable[str], *, width: int = 8) -> None:
    count = 0
    for index, value in enumerate(values):
        count += 1
        if index % width == 0:
            stream.write("        ")
        stream.write(value)
        # USDA list items remain comma-separated across physical line breaks.
        # Omitting the comma on every `width`th value makes the layer invalid
        # even though a text-only dependency scan still succeeds.
        stream.write(",")
        if index % width == width - 1:
            stream.write("\n")
    if count:
        if index % width != width - 1:
            stream.write("\n")


def terrain_source_path(package: Path, tile: dict[str, Any]) -> tuple[Path, Path]:
    elevation = package / str(tile["elevation"]["path"])
    colour = package / str(tile["colour"]["path"])
    if not elevation.is_file() or not colour.is_file():
        raise FileNotFoundError(f"Terrain assets missing for {tile['terrain_tile_id']}")
    return elevation, colour


def write_terrain_tile(
    *,
    package: Path,
    tile: dict[str, Any],
    output: Path,
    texture_path: str,
    anchor: tuple[float, float],
    grid: int,
    texture_bounds_l93_m: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    elevation_path, _ = terrain_source_path(package, tile)
    bounds = [float(value) for value in tile["bounds_l93_metres"]]
    with rasterio.open(elevation_path) as dataset:
        data = dataset.read(
            1,
            out_shape=(grid, grid),
            resampling=Resampling.bilinear,
        ).astype(np.float64, copy=False)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    west, south, east, north = bounds
    eastings = np.linspace(west, east, grid)
    northings = np.linspace(north, south, grid)
    points = [
        (eastings[column] - anchor[0], northings[row] - anchor[1], data[row, column])
        for row in range(grid)
        for column in range(grid)
    ]
    faces = [
        (row * grid + column, row * grid + column + 1, (row + 1) * grid + column + 1, (row + 1) * grid + column)
        for row in range(grid - 1)
        for column in range(grid - 1)
    ]
    if texture_bounds_l93_m is None:
        st = [
            (column / max(1, grid - 1), 1.0 - row / max(1, grid - 1))
            for row in range(grid)
            for column in range(grid)
        ]
    else:
        texture_west, texture_south, texture_east, texture_north = texture_bounds_l93_m
        st = [
            (
                (eastings[column] - texture_west) / (texture_east - texture_west),
                (northings[row] - texture_south) / (texture_north - texture_south),
            )
            for row in range(grid)
            for column in range(grid)
        ]
    prim = usd_identifier(str(tile["terrain_tile_id"]))
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            '#usda 1.0\n(\n'
            '    metersPerUnit = 1\n'
            '    upAxis = "Z"\n'
            ')\n\n'
        )
        stream.write(f'def Xform "Terrain_{prim}" (kind = "component")\n{{\n')
        stream.write(f'    custom string fireviewer:terrain_tile_id = "{usd_string(str(tile["terrain_tile_id"]))}"\n')
        stream.write('    custom string fireviewer:crs = "EPSG:2154"\n')
        stream.write('    custom string fireviewer:vertical_datum = "NGF-IGN69"\n')
        stream.write(f'    custom string fireviewer:ground_texture = "{"ign_bd_ortho_georeferenced" if texture_bounds_l93_m is not None else "published_colour_layer"}"\n')
        stream.write(f'    custom double2 fireviewer:origin_l93_m = ({anchor[0]:.4f}, {anchor[1]:.4f})\n')
        stream.write('    def Mesh "Surface"\n    {\n')
        stream.write('        uniform token subdivisionScheme = "none"\n')
        # Raster rows are authored north-to-south, so the generated quad
        # winding is left-handed in the local XY plane.  Declare that winding
        # explicitly; otherwise RTX back-face culling hides the terrain and
        # its orthophoto when viewed from above.
        stream.write('        uniform token orientation = "leftHanded"\n')
        stream.write("        int[] faceVertexCounts = [\n")
        write_values(stream, ["4"] * len(faces))
        stream.write("        ]\n        int[] faceVertexIndices = [\n")
        write_values(stream, [str(item) for face in faces for item in face])
        stream.write("        ]\n        point3f[] points = [\n")
        write_values(stream, [format_vec(point) for point in points], width=2)
        stream.write("        ]\n        texCoord2f[] primvars:st = [\n")
        write_values(stream, [format_vec(value) for value in st], width=4)
        stream.write('        ] ( interpolation = "vertex" )\n')
        stream.write(f'        rel material:binding = </Terrain_{prim}/Materials/Orthophoto>\n')
        stream.write("    }\n")
        stream.write('    def Scope "Materials"\n    {\n')
        stream.write('        def Material "Orthophoto"\n        {\n')
        stream.write('            token outputs:surface.connect = </Terrain_'
                     f'{prim}/Materials/Orthophoto/Preview.outputs:surface>\n')
        stream.write('            def Shader "Preview"\n            {\n')
        stream.write('                uniform token info:id = "UsdPreviewSurface"\n')
        stream.write('                color3f inputs:diffuseColor.connect = </Terrain_'
                     f'{prim}/Materials/Orthophoto/Texture.outputs:rgb>\n')
        stream.write('                float inputs:roughness = 0.92\n                token outputs:surface\n')
        stream.write("            }\n")
        stream.write('            def Shader "Texture"\n            {\n')
        stream.write('                uniform token info:id = "UsdUVTexture"\n')
        stream.write(f'                asset inputs:file = @{texture_path}@\n')
        stream.write('                token inputs:sourceColorSpace = "sRGB"\n')
        stream.write('                token inputs:wrapS = "clamp"\n')
        stream.write('                token inputs:wrapT = "clamp"\n')
        stream.write('                float2 inputs:st.connect = </Terrain_'
                     f'{prim}/Materials/Orthophoto/StReader.outputs:result>\n')
        stream.write('                float3 outputs:rgb\n            }\n')
        stream.write('            def Shader "StReader"\n            {\n')
        stream.write('                uniform token info:id = "UsdPrimvarReader_float2"\n')
        stream.write('                token inputs:varname = "st"\n                float2 outputs:result\n')
        stream.write("            }\n        }\n    }\n}\n")
    return {
        "terrain_tile_id": tile["terrain_tile_id"],
        "source_elevation": str(elevation_path),
        "source_elevation_sha256": sha256_file(elevation_path),
        "mesh_grid": [grid, grid],
        "vertex_count": len(points),
        "quad_count": len(faces),
        "bounds_l93_metres": bounds,
    }


def mesh_colour(geometry: Any, name: str) -> np.ndarray:
    visual = getattr(geometry, "visual", None)
    colours = getattr(visual, "vertex_colors", None)
    if colours is not None and len(colours) == len(geometry.vertices):
        return np.asarray(colours[:, :3], dtype=np.float64) / 255.0
    lowered = name.lower()
    colour = (0.19, 0.36, 0.20) if "tree" in lowered or "vegetation" in lowered else (0.30, 0.31, 0.30) if "road" in lowered or "path" in lowered else (0.43, 0.38, 0.30)
    return np.tile(np.asarray(colour, dtype=np.float64), (len(geometry.vertices), 1))


def write_feature_tile(
    *,
    package: Path,
    entry: dict[str, Any],
    output: Path,
    anchor: tuple[float, float],
) -> dict[str, Any]:
    source = package / str(entry["features"]["path"])
    if not source.is_file():
        raise FileNotFoundError(f"Feature asset missing for {entry['tile_id']}")
    origin = [float(value) for value in entry["gltf_local_origin_l93_ngf_ign69"]]
    scene = trimesh.load(source, force="scene", process=False)
    geometry_records: list[dict[str, Any]] = []
    prim = usd_identifier(str(entry["tile_id"]))
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write('#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n')
        stream.write(f'def Xform "Features_{prim}" (kind = "component")\n{{\n')
        stream.write(f'    custom string fireviewer:feature_tile_id = "{usd_string(str(entry["tile_id"]))}"\n')
        stream.write('    custom string fireviewer:crs = "EPSG:2154"\n')
        stream.write('    custom string fireviewer:gltf_axes = "local glTF (E, U, -N) metres"\n')
        stream.write(f'    custom double3 fireviewer:gltf_origin_l93_ngf_ign69 = {format_vec(origin)}\n')
        for ordinal, (name, geometry) in enumerate(scene.geometry.items(), start=1):
            transform, _ = scene.graph.get(name)
            vertices = np.asarray(geometry.vertices, dtype=np.float64)
            homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
            transformed = homogeneous @ np.asarray(transform, dtype=np.float64).T
            points = [
                (
                    vertex[0] + origin[0] - anchor[0],
                    -vertex[2] + origin[1] - anchor[1],
                    vertex[1] + origin[2],
                )
                for vertex in transformed
            ]
            faces = np.asarray(geometry.faces, dtype=np.int64)
            colours = mesh_colour(geometry, str(name))
            mesh_name = usd_identifier(f"{ordinal}_{name}")
            stream.write(f'    def Mesh "{mesh_name}"\n    {{\n')
            stream.write('        uniform token subdivisionScheme = "none"\n')
            stream.write("        int[] faceVertexCounts = [\n")
            write_values(stream, ["3"] * len(faces))
            stream.write("        ]\n        int[] faceVertexIndices = [\n")
            write_values(stream, [str(item) for face in faces for item in face])
            stream.write("        ]\n        point3f[] points = [\n")
            write_values(stream, [format_vec(point) for point in points], width=2)
            stream.write("        ]\n        color3f[] primvars:displayColor = [\n")
            write_values(stream, [format_vec(colour) for colour in colours], width=2)
            stream.write('        ] ( interpolation = "vertex" )\n')
            stream.write("    }\n")
            geometry_records.append({"name": str(name), "vertex_count": len(points), "triangle_count": len(faces)})
        stream.write("}\n")
    return {
        "tile_id": entry["tile_id"],
        "source_glb": str(source),
        "source_glb_sha256": sha256_file(source),
        "bounds_l93_metres": [float(value) for value in entry["bounds_l93_metres"]],
        "geometry": geometry_records,
    }


def write_index_layer(output: Path, directory: str, files: list[str]) -> None:
    index = output / directory / "index.usda"
    sublayers = ",\n".join(f"        @{Path(name).name}@" for name in files)
    index.write_text(
        '#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = "Z"\n    subLayers = [\n'
        + sublayers
        + "\n    ]\n)\n",
        encoding="utf-8",
    )


def package_priority(path: Path) -> tuple[int, str]:
    text = str(path).lower().replace("\\", "/")
    if "/fireviewer-frontend/public/maps/" in text:
        return (0, text)
    if "/fireviewer-frontend/dist/maps/" in text:
        return (1, text)
    if "/public/maps/" in text:
        return (2, text)
    if "/dist/maps/" in text:
        return (3, text)
    if "/artifacts/" in text:
        return (4, text)
    return (5, text)


def complete_published_package(package: Path) -> bool:
    try:
        catalog = json.loads((package / "catalog.json").read_text(encoding="utf-8"))
        if catalog.get("schema_version") != "1.1":
            return False
        for tile in catalog.get("terrain_tiles", []):
            for key in ("elevation", "colour"):
                if not (package / str(tile[key]["path"])).is_file():
                    return False
        for entry in catalog.get("feature_tiles", []):
            if not (package / str(entry["features"]["path"])).is_file():
                return False
        return bool(catalog.get("terrain_tiles")) and bool(catalog.get("feature_tiles"))
    except (KeyError, OSError, json.JSONDecodeError, TypeError):
        return False


def discover_packages(roots: list[Path]) -> list[Path]:
    by_id: dict[str, list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("package-manifest.json"):
            if any(part in EXCLUDED_PARTS for part in manifest_path.parts):
                continue
            package = manifest_path.parent
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            package_id = manifest.get("package_id")
            if isinstance(package_id, str) and (package / "catalog.json").is_file() and complete_published_package(package):
                by_id.setdefault(package_id, []).append(package)
    return [sorted(paths, key=package_priority)[0] for paths in sorted(by_id.values(), key=lambda group: str(group[0]))]


def load_orthophoto_source(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "fireviewer.ign-orthophoto-source.v1" or record.get("status") != "downloaded_and_validated":
        raise ValueError(f"Unsupported or incomplete IGN orthophoto source: {path}")
    bounds = tuple(float(value) for value in record["request"]["bounds_l93_m"])
    if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError(f"Invalid orthophoto bounds: {path}")
    jpeg = next((item for item in record.get("outputs", []) if item.get("role") == "blender_rgb_jpeg"), None)
    if not isinstance(jpeg, dict):
        raise ValueError(f"Orthophoto source has no RGB JPEG output: {path}")
    image = path.parent / str(jpeg["file_name"])
    if not image.is_file() or sha256_file(image) != str(jpeg["sha256"]):
        raise ValueError(f"Orthophoto JPEG is missing or changed: {image}")
    return {
        "source_record": path,
        "image": image,
        "bounds_l93_metres": bounds,
        "nominal_resolution_m": float(record["request"]["nominal_resolution_m"]),
        "provider": str(record["provider"]),
        "product": str(record["product"]),
        "license": record["license"],
        "sha256": str(jpeg["sha256"]),
    }


def convert_package(package: Path, output_root: Path, terrain_grid: int, orthophoto: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = json.loads((package / "package-manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((package / "catalog.json").read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "1.1" or not catalog.get("terrain_tiles") or not catalog.get("feature_tiles"):
        raise ValueError(f"Unsupported or incomplete published map catalog: {package}")
    package_id = str(manifest["package_id"])
    output = output_root / package_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already contains files: {output}")
    (output / "terrain").mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir(parents=True, exist_ok=True)
    (output / "source").mkdir(parents=True, exist_ok=True)
    shutil.copy2(package / "package-manifest.json", output / "source/package-manifest.json")
    shutil.copy2(package / "catalog.json", output / "source/catalog.json")
    anchor = tuple(float(value) for value in catalog["spatial_contract"]["common_anchor_l93_metres"])
    if orthophoto is not None:
        source_bounds = tuple(float(value) for value in catalog["bounds_l93_metres"])
        if any(abs(value - source_value) > 0.01 for value, source_value in zip(orthophoto["bounds_l93_metres"], source_bounds)):
            raise ValueError(
                f"IGN orthophoto bounds {orthophoto['bounds_l93_metres']} do not match package bounds {source_bounds}"
            )
        real_texture_name = "ign-bd-ortho-real-ground.jpg"
        shutil.copy2(orthophoto["image"], output / "textures" / real_texture_name)
        world_file = orthophoto["image"].with_suffix(".jgw")
        if not world_file.is_file():
            raise FileNotFoundError(f"Orthophoto world file is missing: {world_file}")
        shutil.copy2(world_file, output / "textures/ign-bd-ortho-real-ground.jgw")
        shutil.copy2(orthophoto["source_record"], output / "source/ign-orthophoto-source.json")
        if sha256_file(output / "textures" / real_texture_name) != orthophoto["sha256"]:
            raise ValueError("Copied IGN orthophoto checksum mismatch")
    terrain_records: list[dict[str, Any]] = []
    terrain_files: list[str] = []
    for tile in catalog["terrain_tiles"]:
        tile_id = usd_identifier(str(tile["terrain_tile_id"]))
        _, colour_source = terrain_source_path(package, tile)
        if orthophoto is None:
            shutil.copy2(colour_source, output / "textures" / f"{tile_id}_colour.png")
            texture_path = f"../textures/{tile_id}_colour.png"
            texture_bounds = None
        else:
            texture_path = f"../textures/{real_texture_name}"
            texture_bounds = orthophoto["bounds_l93_metres"]
        terrain_files.append(f"{tile_id}.usda")
        terrain_records.append(write_terrain_tile(
            package=package,
            tile=tile,
            output=output / "terrain" / f"{tile_id}.usda",
            texture_path=texture_path,
            anchor=anchor,
            grid=terrain_grid,
            texture_bounds_l93_m=texture_bounds,
        ))
    feature_records: list[dict[str, Any]] = []
    feature_files: list[str] = []
    for entry in catalog["feature_tiles"]:
        tile_id = usd_identifier(str(entry["tile_id"]))
        feature_files.append(f"{tile_id}.usda")
        feature_records.append(write_feature_tile(
            package=package,
            entry=entry,
            output=output / "features" / f"{tile_id}.usda",
            anchor=anchor,
        ))
    write_index_layer(output, "terrain", terrain_files)
    write_index_layer(output, "features", feature_files)
    stage = output / "scene.usda"
    stage.write_text(
        '#usda 1.0\n(\n'
        '    defaultPrim = "FireViewerSiteMap"\n'
        '    metersPerUnit = 1\n'
        '    upAxis = "Z"\n'
        '    subLayers = [@terrain/index.usda@, @features/index.usda@]\n'
        ')\n\n'
        'def Xform "FireViewerSiteMap" (kind = "assembly")\n{\n'
        f'    custom string fireviewer:package_id = "{usd_string(package_id)}"\n'
        '    custom string fireviewer:crs = "EPSG:2154"\n'
        '    custom string fireviewer:vertical_datum = "NGF-IGN69"\n'
        '    custom string fireviewer:source_type = "published_site_map"\n'
        f'    custom double2 fireviewer:common_anchor_l93_m = ({anchor[0]:.4f}, {anchor[1]:.4f})\n'
        '}\n',
        encoding="utf-8",
    )
    record = {
        "schema": "fireviewer.omniverse-published-map.v1",
        "package_id": package_id,
        "source_package": str(package),
        "source_manifest_sha256": sha256_file(package / "package-manifest.json"),
        "source_catalog_sha256": sha256_file(package / "catalog.json"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_convention": "usd_z_up_meters_epsg2154_local_anchor",
        "common_anchor_l93_metres": list(anchor),
        "bounds_l93_metres": catalog["bounds_l93_metres"],
        "terrain_tile_count": len(terrain_records),
        "feature_tile_count": len(feature_records),
        "terrain": terrain_records,
        "orthophoto": (
            {
                "role": "real_georeferenced_ground_texture",
                "provider": orthophoto["provider"],
                "product": orthophoto["product"],
                "license": orthophoto["license"],
                "bounds_l93_metres": list(orthophoto["bounds_l93_metres"]),
                "nominal_resolution_m": orthophoto["nominal_resolution_m"],
                "sha256": orthophoto["sha256"],
                "source_record": "source/ign-orthophoto-source.json",
                "texture": "textures/ign-bd-ortho-real-ground.jpg",
                "world_file": "textures/ign-bd-ortho-real-ground.jgw",
            }
            if orthophoto is not None
            else {"role": "published_colour_layer"}
        ),
        "features": feature_records,
        "entry_stage": "scene.usda",
    }
    (output / "manifest.json").write_text(canonical_json(record), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--terrain-grid", type=int, default=513)
    parser.add_argument("--orthophoto-source", type=Path, help="Validated fireviewer.ign-orthophoto-source.v1 sidecar covering the exact package bounds")
    args = parser.parse_args()
    if args.terrain_grid < 17 or args.terrain_grid > 4097:
        raise SystemExit("--terrain-grid must be between 17 and 4097")
    packages = discover_packages([path.resolve() for path in args.site_root])
    if not packages:
        raise SystemExit("No published map packages found")
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    orthophoto = load_orthophoto_source(args.orthophoto_source.resolve()) if args.orthophoto_source else None
    results = [convert_package(package, args.output_root.resolve(), args.terrain_grid, orthophoto) for package in packages]
    index = {
        "schema": "fireviewer.omniverse-published-map-index.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packages": [
            {"package_id": result["package_id"], "stage": f"{result['package_id']}/scene.usda", "terrain_tiles": result["terrain_tile_count"], "feature_tiles": result["feature_tile_count"]}
            for result in results
        ],
    }
    (args.output_root.resolve() / "index.json").write_text(canonical_json(index), encoding="utf-8")
    print(canonical_json(index), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
