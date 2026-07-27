"""Export immutable FireViewer spatial payloads and a small Unity catalog.

This command does not upload anything.  It creates a deployment-ready folder
whose catalog only contains relative, content-addressed URLs.  A CDN or object
store can publish that folder without rewriting the contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Callable, Mapping, Sequence

from fwtile import (
    FWTileError,
    build_container,
    build_vector_sections,
    canonical_json_bytes,
    encode_detail_terrain,
    encode_far_grid,
    encode_tree_instances,
    read_container,
    sha256_bytes,
    sha256_file,
)


DEFAULT_ARTIFACT_ROOT = Path(".artifacts/spatial-lidar-surface/synthetic-example-r1")
DETAIL_ZONE_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "detail_zones.v1.json"
CATALOG_SCHEMA = "fireviewer.remote-tile-catalog.v1"
RECEIPT_SCHEMA = "fireviewer.remote-tile-receipt.v1"
FAR_RECEIPT_SCHEMA = "fireviewer.remote-far-receipt.v1"
REQUIRED_DETAIL_SECTIONS = ("terrain", "trees", "buildings", "roads", "water")
DETAIL_CONTENT_MODES = {
    "complete_3d": REQUIRED_DETAIL_SECTIONS,
    "trees_3d_lidar_infrastructure": ("terrain", "trees"),
}
# The former revision rejected invalid roof rings before it could emit a tile.
# The current revision only adds a deterministic make-valid path for those
# previously unexportable buildings. Tiles which already have a receipt took
# the unchanged valid-ring path and remain byte-identical, so their fully
# checked receipts may be adopted without rebuilding the 3+ GiB vector model.
COMPATIBLE_EXPORTER_SHA256S = frozenset(
    {
        "c5871c045fd34fb009f672601a976be5277c319c15453563f5e594b07efe6960",
        "555cc75de809f6e4bc567e6f8ea6591442651b84e5f3f4e77e9b6ae073576b7a",
        "34348212cd6e425c367faeaad4b7e54384470057f0b1e0b2c4bbb1d1b83fe734",
        "be528038408d80b3a0d5d54be8130516633f0d0ac7d7055ee2456aee673604cc",
        "47e0d5a33a980a7192c91d61d11bd69975f6f736685220839af18b5360719543",
    }
)


def _exporter_fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    sources = [directory / "fwtile.py", Path(__file__).resolve()]
    content = b"\0".join(
        path.name.encode("utf-8") + b"\0" + path.read_bytes() for path in sources
    )
    return sha256_bytes(content)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FWTileError(f"cannot read JSON {path}") from exc
    if not isinstance(result, dict):
        raise FWTileError(f"JSON root in {path} is not an object")
    return result


def _load_json_gzip(path: Path) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise FWTileError(f"cannot read gzip JSON {path}") from exc
    if not isinstance(result, dict):
        raise FWTileError(f"gzip JSON root in {path} is not an object")
    return result


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_relative_url(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise FWTileError(f"catalog URL {value!r} is not a safe relative URL")
    return path.as_posix()


def _publish_immutable(
    output_root: Path, directory: str, stem: str, suffix: str, content: bytes
) -> dict[str, Any]:
    digest = sha256_bytes(content)
    relative = _validate_relative_url(
        (PurePosixPath(directory) / f"{stem}.{digest[:16]}{suffix}").as_posix()
    )
    destination = output_root / Path(*PurePosixPath(relative).parts)
    if destination.exists():
        if (
            destination.stat().st_size != len(content)
            or sha256_file(destination) != digest
        ):
            raise FWTileError(f"immutable output collision at {destination}")
    else:
        _atomic_write(destination, content)
    return {"url": relative, "sha256": digest, "byte_count": len(content)}


def _asset_path_and_hash(
    manifest_directory: Path,
    asset: Mapping[str, Any] | None,
    label: str,
) -> tuple[Path, str]:
    if not asset or not asset.get("path") or not asset.get("sha256"):
        raise FWTileError(f"{label} has no path/checksum in the production manifest")
    path = manifest_directory / Path(str(asset["path"]))
    if not path.is_file():
        raise FWTileError(f"{label} is missing: {path}")
    expected_size = asset.get("byte_count")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise FWTileError(f"{label} byte count does not match the manifest")
    digest = sha256_file(path)
    if digest != str(asset["sha256"]):
        raise FWTileError(f"{label} checksum does not match the manifest")
    return path, digest


def _choose_orthophoto(
    tile: Mapping[str, Any],
    manifest_directory: Path,
    *,
    near_lod_disabled: bool = False,
) -> tuple[Path, str, float]:
    assets = tile.get("assets", {})
    near_status = tile.get("near_orthophoto_status", {}).get("state")
    if near_status == "ready" and not near_lod_disabled:
        path, digest = _asset_path_and_hash(
            manifest_directory,
            assets.get("near_orthophoto_image"),
            f"tile {tile['id']} 0.20 m orthophoto",
        )
        return path, digest, 0.2
    path, digest = _asset_path_and_hash(
        manifest_directory,
        assets.get("orthophoto_image"),
        f"tile {tile['id']} 0.50 m orthophoto",
    )
    return path, digest, 0.5


def _same_numbers(
    left: Sequence[Any], right: Sequence[Any], tolerance: float = 1e-6
) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right, strict=True)
    )


def _terrain_has_complete_regular_grid(
    terrain: Mapping[str, Any], bounds: Sequence[float]
) -> bool:
    spacing = terrain.get("sample_spacing_m")
    vertices = terrain.get("vertices")
    if (
        not isinstance(spacing, Sequence)
        or len(spacing) != 2
        or not isinstance(vertices, list)
    ):
        return False
    spacing_x, spacing_y = map(float, spacing)
    columns = int(round((float(bounds[2]) - float(bounds[0])) / spacing_x)) + 1
    rows = int(round((float(bounds[3]) - float(bounds[1])) / spacing_y)) + 1
    return len(vertices) == rows * columns


def _terrain_is_regular_grid(terrain: Mapping[str, Any]) -> bool:
    vertices = terrain.get("vertices")
    faces = terrain.get("faces")
    if (
        not isinstance(vertices, list)
        or len(vertices) < 4
        or not isinstance(faces, list)
    ):
        return False
    first_y = float(vertices[0][1])
    columns = 0
    for vertex in vertices:
        if abs(float(vertex[1]) - first_y) > 1e-7:
            break
        columns += 1
    if columns < 2 or len(vertices) % columns:
        return False
    rows = len(vertices) // columns
    return rows >= 2 and len(faces) == (rows - 1) * (columns - 1)


def _crop_to_largest_measured_regular_grid(
    terrain: Mapping[str, Any],
) -> dict[str, Any]:
    """Crop a source-edge mesh to its largest fully measured rectangle.

    IGN source coverage can omit a small corner of the requested processing
    halo. The detailed terrain contract is an implicit rectangular grid, so
    it must never fill that corner with an invented elevation. This chooses
    the largest all-measured vertex rectangle and lets the global far terrain
    remain visible over the excluded edge strip.
    """

    source_vertices = terrain.get("vertices")
    if not isinstance(source_vertices, list) or len(source_vertices) < 4:
        raise FWTileError("measured terrain has too few vertices")
    lookup: dict[tuple[float, float], list[Any]] = {}
    for vertex in source_vertices:
        if not isinstance(vertex, list) or len(vertex) != 3:
            raise FWTileError("measured terrain contains an invalid vertex")
        key = (round(float(vertex[0]), 6), round(float(vertex[1]), 6))
        if key in lookup:
            raise FWTileError("measured terrain contains duplicate XY vertices")
        lookup[key] = vertex
    eastings = sorted({key[0] for key in lookup})
    northings = sorted({key[1] for key in lookup}, reverse=True)
    if len(eastings) < 2 or len(northings) < 2:
        raise FWTileError("measured terrain cannot form a rectangular grid")

    heights = [0] * len(eastings)
    best: tuple[int, int, int, int] | None = None
    best_key: tuple[int, int, int, int, int] | None = None
    for row, north in enumerate(northings):
        for column, east in enumerate(eastings):
            heights[column] = heights[column] + 1 if (east, north) in lookup else 0
        stack: list[int] = []
        for column in range(len(eastings) + 1):
            current = heights[column] if column < len(eastings) else 0
            while stack and heights[stack[-1]] > current:
                histogram_column = stack.pop()
                rectangle_height = heights[histogram_column]
                left = stack[-1] + 1 if stack else 0
                right = column - 1
                top = row - rectangle_height + 1
                bottom = row
                width = right - left + 1
                area = rectangle_height * width
                # Prefer maximum measured area; ties preserve east-west width,
                # then north-south height, then the north-west-most rectangle.
                candidate_key = (area, width, rectangle_height, -top, -left)
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best = (top, bottom, left, right)
            stack.append(column)
    if best is None:
        raise FWTileError("measured terrain has no rectangular grid")
    top, bottom, left, right = best
    rows = bottom - top + 1
    columns = right - left + 1
    if rows < 2 or columns < 2:
        raise FWTileError("largest measured terrain rectangle is not renderable")

    vertices = [
        lookup[(eastings[column], northings[row])]
        for row in range(top, bottom + 1)
        for column in range(left, right + 1)
    ]
    faces = [
        [
            row * columns + column,
            row * columns + column + 1,
            (row + 1) * columns + column + 1,
            (row + 1) * columns + column,
        ]
        for row in range(rows - 1)
        for column in range(columns - 1)
    ]
    result = dict(terrain)
    result.update(
        {
            "vertices": vertices,
            "faces": faces,
            "vertex_count": len(vertices),
            "face_count": len(faces),
        }
    )
    return result


def _load_detail_vector_builder() -> Callable[..., dict[str, Any]]:
    blender_directory = Path(__file__).resolve().parents[1] / "blender"
    path_text = str(blender_directory)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    from detail_vector_lod import build_detail_tile_vectors

    return build_detail_tile_vectors


@dataclass
class ExportContext:
    production_manifest_path: Path
    global_vector_package_path: Path | None
    output_root: Path
    manifest: dict[str, Any]
    production_manifest_sha256: str
    global_vector_sha256: str | None
    exporter_sha256: str
    near_lod_disabled: bool
    global_only: bool
    detail_content_mode: str
    vector_builder: Callable[..., dict[str, Any]]
    _global_vector_package: dict[str, Any] | None = None
    _validated_mnt_sources: dict[str, str] | None = None

    @property
    def manifest_directory(self) -> Path:
        return self.production_manifest_path.parent

    @property
    def global_vector_package(self) -> dict[str, Any]:
        if self.global_vector_package_path is None:
            raise FWTileError(
                "global-only catalog does not include a detail vector package"
            )
        if self._global_vector_package is None:
            self._global_vector_package = _load_json_gzip(
                self.global_vector_package_path
            )
            if (
                self._global_vector_package.get("schema")
                != "fireviewer.blender-preview-package.v2"
            ):
                raise FWTileError("unsupported global vector package schema")
        return self._global_vector_package

    @property
    def validated_mnt_sources(self) -> dict[str, str]:
        if self._validated_mnt_sources is None:
            self._validated_mnt_sources = {}
        return self._validated_mnt_sources


def _complete_real_terrain_grid(
    context: ExportContext,
    package: Mapping[str, Any],
    bounds: Sequence[float],
    origin: Sequence[float],
) -> dict[str, Any]:
    """Restore an AOI-masked edge grid from its checked real IGN MNT inputs."""

    source_records = package.get("metadata", {}).get("sources", {}).get("mnt")
    processing_bounds = package.get("metadata", {}).get("processing_bounds_l93_m")
    if not isinstance(source_records, list) or not source_records:
        raise FWTileError("partial terrain package has no MNT source records")
    if not isinstance(processing_bounds, list) or len(processing_bounds) != 4:
        raise FWTileError("partial terrain package has no processing bounds")
    source_paths: list[Path] = []
    for source in source_records:
        file_name = str(source.get("file_name", ""))
        expected_sha = str(source.get("sha256", ""))
        path = context.manifest_directory / "sources" / "mnt" / file_name
        if not path.is_file() or len(expected_sha) != 64:
            raise FWTileError(f"partial terrain MNT source is missing: {path}")
        cached = context.validated_mnt_sources.get(file_name)
        if cached is None:
            cached = sha256_file(path)
            context.validated_mnt_sources[file_name] = cached
        if cached != expected_sha:
            raise FWTileError(f"partial terrain MNT checksum changed: {path}")
        source_paths.append(path)

    blender_directory = Path(__file__).resolve().parents[1] / "blender"
    if str(blender_directory) not in sys.path:
        sys.path.insert(0, str(blender_directory))
    import numpy as np
    from prepare_mid_vegetation_05m import _mosaic, build_local_terrain_mesh

    values, transform, _source_metadata = _mosaic(source_paths, processing_bounds)
    complete = build_local_terrain_mesh(
        values,
        transform,
        origin,
        valid_mask=np.isfinite(values),
        step_pixels=2,
        coordinate_precision=3,
        bounds=bounds,
    )

    # Validate against the unpruned sample set first. IGN source boundaries
    # can contribute a measured vertex whose three neighbours are absent; it
    # carries no renderable triangle and is intentionally removed afterwards.
    restored_lookup = {
        (round(float(vertex[0]), 6), round(float(vertex[1]), 6)): vertex
        for vertex in complete["vertices"]
    }
    for index, vertex in enumerate(package["terrain"]["vertices"]):
        restored = restored_lookup.get(
            (round(float(vertex[0]), 6), round(float(vertex[1]), 6))
        )
        if restored is None or any(
            abs(float(left) - float(right)) > 0.0011
            for left, right in zip(vertex, restored, strict=True)
        ):
            raise FWTileError(
                f"restored real MNT differs from masked terrain vertex {index}"
            )

    if not _terrain_is_regular_grid(complete):
        complete = _crop_to_largest_measured_regular_grid(complete)
    if not _terrain_is_regular_grid(complete):
        raise FWTileError("real MNT inputs did not restore a regular tile grid")

    origin_x, origin_y = float(origin[0]), float(origin[1])
    vertices = complete["vertices"]
    first_y = float(vertices[0][1])
    columns = 0
    for vertex in vertices:
        if abs(float(vertex[1]) - first_y) > 1e-7:
            break
        columns += 1
    rows = len(vertices) // columns
    actual_bounds = [
        origin_x + float(vertices[0][0]),
        origin_y + float(vertices[(rows - 1) * columns][1]),
        origin_x + float(vertices[columns - 1][0]),
        origin_y + first_y,
    ]
    complete["geometric_bounds_l93_m"] = actual_bounds

    return complete


def create_context(
    production_manifest_path: Path,
    global_vector_package_path: Path | None,
    output_root: Path,
    *,
    near_lod_disabled: bool = False,
    global_only: bool = False,
    detail_content_mode: str = "complete_3d",
    vector_builder: Callable[..., dict[str, Any]] | None = None,
) -> ExportContext:
    if detail_content_mode not in DETAIL_CONTENT_MODES:
        raise FWTileError(f"unsupported detail content mode: {detail_content_mode}")
    manifest = _load_json(production_manifest_path)
    if manifest.get("schema") != "fireviewer.global-05m-production-manifest.v1":
        raise FWTileError("unsupported production manifest schema")
    source_ready = all(
        isinstance(source, Mapping) and source.get("status", {}).get("state") == "ready"
        for source in manifest.get("source_tiles", [])
    )
    if manifest.get("status") != "ready" and not (global_only and source_ready):
        raise FWTileError("production manifest is not globally ready")
    if (
        not global_only
        and detail_content_mode == "complete_3d"
        and (
            global_vector_package_path is None
            or not global_vector_package_path.is_file()
        )
    ):
        raise FWTileError(
            f"global vector package is missing: {global_vector_package_path}"
        )
    return ExportContext(
        production_manifest_path=production_manifest_path,
        global_vector_package_path=global_vector_package_path,
        output_root=output_root,
        manifest=manifest,
        production_manifest_sha256=sha256_file(production_manifest_path),
        global_vector_sha256=(
            sha256_file(global_vector_package_path)
            if global_vector_package_path is not None
            else None
        ),
        exporter_sha256=_exporter_fingerprint(),
        near_lod_disabled=near_lod_disabled,
        global_only=global_only,
        detail_content_mode=detail_content_mode,
        vector_builder=vector_builder or _load_detail_vector_builder(),
    )


def _receipt_path(context: ExportContext, tile_id: str) -> Path:
    return context.output_root / "receipts" / f"{tile_id}.json"


def _validate_output_reference(output_root: Path, reference: Mapping[str, Any]) -> Path:
    relative = _validate_relative_url(str(reference["url"]))
    path = output_root / Path(*PurePosixPath(relative).parts)
    if not path.is_file():
        raise FWTileError(f"published output is missing: {path}")
    if path.stat().st_size != int(reference["byte_count"]):
        raise FWTileError(f"published output byte count changed: {path}")
    if sha256_file(path) != str(reference["sha256"]):
        raise FWTileError(f"published output checksum changed: {path}")
    return path


def _try_resume_tile(
    context: ExportContext, tile_id: str, expected_inputs: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _receipt_path(context, tile_id)
    if not path.exists():
        return None
    try:
        receipt = _load_json(path)
        actual_inputs = receipt.get("inputs")
        compatible_inputs = (
            isinstance(actual_inputs, Mapping)
            and actual_inputs.get("exporter_sha256") in COMPATIBLE_EXPORTER_SHA256S
            and {
                key: value
                for key, value in actual_inputs.items()
                if key != "exporter_sha256"
            }
            == {
                key: value
                for key, value in expected_inputs.items()
                if key != "exporter_sha256"
            }
        )
        if (
            receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("tile_id") != tile_id
            or (actual_inputs != expected_inputs and not compatible_inputs)
        ):
            return None
        record = receipt["catalog_tile"]
        payload_path = _validate_output_reference(
            context.output_root, record["payload"]
        )
        _validate_output_reference(context.output_root, record["imagery"])
        parsed = read_container(payload_path.read_bytes(), decode_sections=False)
        if (
            tuple(item["name"] for item in parsed["header"]["sections"])
            != DETAIL_CONTENT_MODES[context.detail_content_mode]
        ):
            return None
        if compatible_inputs:
            receipt["inputs"] = dict(expected_inputs)
            _atomic_write(path, canonical_json_bytes(receipt) + b"\n")
        return record
    except (KeyError, TypeError, ValueError, OSError, FWTileError):
        return None


def export_tile(context: ExportContext, tile: Mapping[str, Any]) -> dict[str, Any]:
    tile_id = str(tile.get("id"))
    if tile.get("status", {}).get("state") != "ready":
        raise FWTileError(f"tile {tile_id} is not ready")
    bounds = tile.get("bounds_l93_m")
    origin = tile.get("origin_l93_m")
    if not isinstance(bounds, list) or not isinstance(origin, list):
        raise FWTileError(f"tile {tile_id} has no bounds/origin")
    mid_path, mid_sha = _asset_path_and_hash(
        context.manifest_directory,
        tile.get("assets", {}).get("mid_package"),
        f"tile {tile_id} package",
    )
    image_path, image_sha, image_resolution = _choose_orthophoto(
        tile,
        context.manifest_directory,
        near_lod_disabled=context.near_lod_disabled,
    )
    expected_inputs = {
        "exporter_sha256": context.exporter_sha256,
        "production_manifest_sha256": context.production_manifest_sha256,
        "global_vector_package_sha256": context.global_vector_sha256,
        "mid_package_sha256": mid_sha,
        "orthophoto_sha256": image_sha,
        "orthophoto_resolution_m": image_resolution,
    }
    resumed = _try_resume_tile(context, tile_id, expected_inputs)
    if resumed is not None:
        return resumed

    package = _load_json_gzip(mid_path)
    metadata = package.get("metadata", {})
    if not _same_numbers(metadata.get("bounds_l93_m", []), bounds):
        raise FWTileError(f"tile {tile_id} package bounds do not match the manifest")
    if not _same_numbers(metadata.get("origin_l93_m", []), origin):
        raise FWTileError(f"tile {tile_id} package origin does not match the manifest")
    source_count = int(package.get("statistics", {}).get("accepted_instance_count", -1))
    manifest_count = int(
        tile.get("production_statistics", {}).get("accepted_instance_count", -2)
    )
    if source_count < 0 or source_count != manifest_count:
        raise FWTileError(f"tile {tile_id} vegetation count is inconsistent")

    if not _terrain_has_complete_regular_grid(package["terrain"], bounds):
        completed_package = dict(package)
        completed_package["terrain"] = _complete_real_terrain_grid(
            context, package, bounds, origin
        )
        package = completed_package

    # Never clamp a vector onto an unmeasured terrain edge. Source-edge tiles
    # can legitimately expose a smaller geometric terrain rectangle; vectors
    # are clipped to that verified rectangle while the catalog tile keeps its
    # stable 500 m addressing bounds.
    terrain_bounds = package["terrain"].get("geometric_bounds_l93_m", bounds)
    terrain_raw, terrain_metadata = encode_detail_terrain(
        package["terrain"], bounds, origin
    )
    trees_raw, trees_metadata = encode_tree_instances(package, bounds, origin)
    sections = [
        ("terrain", terrain_raw, terrain_metadata),
        ("trees", trees_raw, trees_metadata),
    ]
    vector_sections: dict[str, tuple[bytes, dict[str, Any]]] = {}
    if context.detail_content_mode == "complete_3d":
        detail_vectors = context.vector_builder(
            context.global_vector_package, package, terrain_bounds, origin
        )
        vector_sections = build_vector_sections(detail_vectors, bounds, origin)
        sections.extend(
            [
                ("buildings", *vector_sections["buildings"]),
                ("roads", *vector_sections["roads"]),
                ("water", *vector_sections["water"]),
            ]
        )
    payload = build_container(
        kind="detail_tile",
        tile_id=tile_id,
        bounds_l93_m=bounds,
        origin_l93_m=origin,
        sections=sections,
        metadata={
            "source_mid_package_sha256": mid_sha,
            "source_global_vector_package_sha256": context.global_vector_sha256,
            "orthophoto_resolution_m": image_resolution,
            "terrain_geometry_lod": "actual_regular_grid",
            "vegetation_export": "all_detected_instances_no_export_thinning",
            "detail_vector_schema": (
                detail_vectors.get("schema")
                if context.detail_content_mode == "complete_3d"
                else None
            ),
            "content_mode": context.detail_content_mode,
        },
    )
    parsed = read_container(payload)
    section_by_name = {item["name"]: item for item in parsed["header"]["sections"]}
    payload_reference = _publish_immutable(
        context.output_root, f"detail/{tile_id}", tile_id, ".fwtile", payload
    )
    image_reference = _publish_immutable(
        context.output_root,
        "imagery",
        tile_id,
        image_path.suffix.lower(),
        image_path.read_bytes(),
    )
    image_reference["resolution_m"] = image_resolution
    record = {
        "id": tile_id,
        "bounds_l93_m": list(map(float, bounds)),
        "payload": payload_reference,
        "imagery": image_reference,
        "counts": {
            "terrain_vertices": terrain_metadata["vertex_count"],
            "trees": trees_metadata["count"],
            "buildings": section_by_name.get("buildings", {})
            .get("metadata", {})
            .get("mesh_count", 0),
            "road_triangles": section_by_name.get("roads", {})
            .get("metadata", {})
            .get("triangle_count", 0),
            "water_triangles": section_by_name.get("water", {})
            .get("metadata", {})
            .get("triangle_count", 0),
        },
        "sections": list(DETAIL_CONTENT_MODES[context.detail_content_mode]),
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "tile_id": tile_id,
        "inputs": expected_inputs,
        "catalog_tile": record,
    }
    _atomic_write(
        _receipt_path(context, tile_id), canonical_json_bytes(receipt) + b"\n"
    )
    return record


def export_far_assets(
    context: ExportContext,
    far_mnt_path: Path,
    far_imagery_path: Path,
    *,
    imagery_resolution_m: float = 2.0,
) -> dict[str, Any]:
    if not far_mnt_path.is_file() or not far_imagery_path.is_file():
        raise FWTileError("far MNT or far imagery source is missing")
    inputs = {
        "exporter_sha256": context.exporter_sha256,
        "far_mnt_sha256": sha256_file(far_mnt_path),
        "far_imagery_sha256": sha256_file(far_imagery_path),
        "production_manifest_sha256": context.production_manifest_sha256,
    }
    receipt_path = context.output_root / "receipts" / "global-far.json"
    if receipt_path.exists():
        try:
            receipt = _load_json(receipt_path)
            if (
                receipt.get("schema") == FAR_RECEIPT_SCHEMA
                and receipt.get("inputs") == inputs
            ):
                record = receipt["catalog_far"]
                _validate_output_reference(context.output_root, record["terrain"])
                _validate_output_reference(context.output_root, record["imagery"])
                return record
        except (KeyError, TypeError, ValueError, OSError, FWTileError):
            pass

    try:
        import numpy as np
        import rasterio
    except ImportError as exc:  # pragma: no cover - repository requirement
        raise FWTileError("Rasterio is required to export the global far MNT") from exc
    with rasterio.open(far_mnt_path) as dataset:
        values = dataset.read(1, masked=True)
        if dataset.crs is None or dataset.crs.to_string() != "EPSG:2154":
            raise FWTileError("far MNT is not EPSG:2154")
        valid = (~np.ma.getmaskarray(values)).reshape(-1).tolist()
        origin = context.manifest["origin_l93_m"]
        elevations = (
            values.filled(float(origin[2])).reshape(-1) - float(origin[2])
        ).tolist()
        bounds = [
            float(dataset.bounds.left),
            float(dataset.bounds.bottom),
            float(dataset.bounds.right),
            float(dataset.bounds.top),
        ]
        terrain_raw, terrain_metadata = encode_far_grid(
            elevations,
            valid,
            rows=dataset.height,
            columns=dataset.width,
            pixel_size_m=[abs(dataset.res[0]), abs(dataset.res[1])],
            outer_bounds_l93_m=bounds,
        )
    payload = build_container(
        kind="global_far_terrain",
        tile_id="global-far",
        bounds_l93_m=bounds,
        origin_l93_m=origin,
        sections=[("terrain", terrain_raw, terrain_metadata)],
        metadata={
            "source_mnt_sha256": inputs["far_mnt_sha256"],
            "actual_mnt_resolution_m": terrain_metadata["sample_spacing_m"],
        },
    )
    terrain_reference = _publish_immutable(
        context.output_root, "far", "global-mnt", ".fwterrain", payload
    )
    terrain_reference["resolution_m"] = terrain_metadata["sample_spacing_m"]
    imagery_reference = _publish_immutable(
        context.output_root,
        "far",
        "global-imagery",
        far_imagery_path.suffix.lower(),
        far_imagery_path.read_bytes(),
    )
    imagery_reference["resolution_m"] = float(imagery_resolution_m)
    record = {
        "terrain": terrain_reference,
        "imagery": imagery_reference,
        "bounds_l93_m": bounds,
    }
    receipt = {
        "schema": FAR_RECEIPT_SCHEMA,
        "inputs": inputs,
        "catalog_far": record,
    }
    _atomic_write(receipt_path, canonical_json_bytes(receipt) + b"\n")
    return record


def _catalog_tile_receipts(context: ExportContext) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    directory = context.output_root / "receipts"
    if not directory.exists():
        return result
    for path in sorted(directory.glob("x*_s*.json"), key=lambda item: item.name):
        receipt = _load_json(path)
        inputs = receipt.get("inputs", {})
        if (
            receipt.get("schema") != RECEIPT_SCHEMA
            or inputs.get("production_manifest_sha256")
            != context.production_manifest_sha256
            or inputs.get("global_vector_package_sha256")
            != context.global_vector_sha256
        ):
            continue
        record = receipt["catalog_tile"]
        _validate_output_reference(context.output_root, record["payload"])
        _validate_output_reference(context.output_root, record["imagery"])
        result.append(record)
    return sorted(result, key=lambda item: item["id"])


def build_catalog(context: ExportContext, far: Mapping[str, Any]) -> dict[str, Any]:
    tiles = [] if context.global_only else _catalog_tile_receipts(context)
    ready_tiles = [
        item
        for item in context.manifest["tiles"]
        if item.get("status", {}).get("state") == "ready" and not context.global_only
    ]
    catalog = {
        "schema": CATALOG_SCHEMA,
        "catalog_version": 1,
        "crs": "EPSG:2154",
        "linear_unit": "metre",
        "origin_l93_m": list(map(float, context.manifest["origin_l93_m"])),
        "axes": {
            "payload": "X=east,Y=north,Z=up",
            "unity_metre_mapping": "Unity(X,Y,Z)=(east,up,north)",
            "legacy_fireviewer_units_per_metre": 100,
        },
        "lod_policy": {
            "far": {
                "role": "always_available_global_fallback",
                "terrain": far["terrain"],
                "imagery": far["imagery"],
                "bounds_l93_m": far["bounds_l93_m"],
            },
            "detail": {
                "enabled": not context.global_only,
                "mode": "global_only" if context.global_only else "streamed_detail",
                "content_mode": None
                if context.global_only
                else context.detail_content_mode,
                "publish_distance_m": 600.0,
                "preload_radius_m": 750.0,
                "maximum_resident_tile_count": 0 if context.global_only else 16,
                "near_disabled": True
                if context.global_only
                else context.near_lod_disabled,
                "transition": "global_surface_only"
                if context.global_only
                else "global_fallback_then_atomic_detail_footprint",
                "eviction": "not_applicable"
                if context.global_only
                else "least_priority_outside_desired_footprint",
            },
        },
        "source": {
            "production_manifest_sha256": context.production_manifest_sha256,
            "production_plan_id": context.manifest.get("plan_id"),
            "global_vector_package_sha256": context.global_vector_sha256,
            "exporter_sha256": context.exporter_sha256,
            "ready_detail_tile_count": len(ready_tiles),
        },
        "exported_detail_tile_count": len(tiles),
        "tiles": tiles,
    }
    validate_catalog(catalog)
    return catalog


def validate_catalog(catalog: Mapping[str, Any]) -> None:
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise FWTileError("catalog schema is unsupported")
    policy = catalog.get("lod_policy", {}).get("detail", {})
    enabled = policy.get("enabled")
    if not isinstance(enabled, bool):
        raise FWTileError("catalog must declare whether detail streaming is enabled")
    if enabled and int(policy.get("maximum_resident_tile_count", 0)) != 16:
        raise FWTileError("catalog must enforce the 16 tile global resident budget")
    if float(policy.get("publish_distance_m", 0)) != 600.0:
        raise FWTileError("catalog must preserve the accepted 600 m publish distance")
    if float(policy.get("preload_radius_m", 0)) != 750.0:
        raise FWTileError("catalog must preserve the accepted 750 m preload radius")
    if not isinstance(policy.get("near_disabled"), bool):
        raise FWTileError("catalog must declare whether the near LOD is disabled")
    tiles_value = catalog.get("tiles")
    if not isinstance(tiles_value, list):
        raise FWTileError("catalog tiles must be an array")
    if not enabled:
        if (
            policy.get("mode") != "global_only"
            or policy.get("near_disabled") is not True
            or int(policy.get("maximum_resident_tile_count", -1)) != 0
            or int(catalog.get("exported_detail_tile_count", -1)) != 0
            or tiles_value
        ):
            raise FWTileError("global-only catalog must not publish detail tiles")
    urls: set[str] = set()
    far = catalog.get("lod_policy", {}).get("far", {})
    if far.get("terrain", {}).get("resolution_m") != [5.0, 5.0]:
        raise FWTileError("catalog must preserve the accepted 5 m FAR terrain")
    if far.get("imagery", {}).get("resolution_m") != 2.0:
        raise FWTileError("catalog must preserve the accepted 2 m FAR imagery")
    for reference in (far.get("terrain"), far.get("imagery")):
        if not isinstance(reference, Mapping):
            raise FWTileError("catalog far LOD is incomplete")
        urls.add(_validate_relative_url(str(reference["url"])))
    tile_ids: set[str] = set()
    for tile in tiles_value:
        tile_id = str(tile.get("id"))
        if tile_id in tile_ids:
            raise FWTileError(f"duplicate catalog tile {tile_id}")
        tile_ids.add(tile_id)
        content_mode = policy.get("content_mode", "complete_3d")
        expected_sections = DETAIL_CONTENT_MODES.get(content_mode)
        if (
            expected_sections is None
            or tuple(tile.get("sections", [])) != expected_sections
        ):
            raise FWTileError(f"catalog tile {tile_id} lacks required sections")
        imagery_resolution = float(tile.get("imagery", {}).get("resolution_m", 0))
        if policy["near_disabled"] and imagery_resolution < 0.5:
            raise FWTileError(
                f"catalog tile {tile_id} publishes near imagery while near LOD is disabled"
            )
        bounds = tile.get("bounds_l93_m", [])
        if (
            len(bounds) != 4
            or float(bounds[2]) <= float(bounds[0])
            or float(bounds[3]) <= float(bounds[1])
        ):
            raise FWTileError(f"catalog tile {tile_id} has invalid bounds")
        for reference in (tile.get("payload"), tile.get("imagery")):
            if not isinstance(reference, Mapping):
                raise FWTileError(f"catalog tile {tile_id} has an incomplete asset")
            url = _validate_relative_url(str(reference["url"]))
            if url in urls:
                raise FWTileError(f"catalog URL is reused: {url}")
            urls.add(url)
            if (
                int(reference.get("byte_count", 0)) <= 0
                or len(str(reference.get("sha256", ""))) != 64
            ):
                raise FWTileError(f"catalog tile {tile_id} asset metadata is invalid")


def write_catalog(context: ExportContext, far: Mapping[str, Any]) -> dict[str, Any]:
    catalog = build_catalog(context, far)
    _atomic_write(
        context.output_root / "catalog.json",
        canonical_json_bytes(catalog) + b"\n",
    )
    return catalog


def _load_detail_zone_bounds(
    path: Path = DETAIL_ZONE_CONTRACT_PATH,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    contract = _load_json(path)
    if contract.get("schema_version") != "1.0":
        raise FWTileError("unsupported detail-zone contract schema")
    if contract.get("horizontal_crs") != "EPSG:2154":
        raise FWTileError("detail-zone contract is not Lambert-93")
    zones = contract.get("zones")
    if not isinstance(zones, list) or not zones:
        raise FWTileError("detail-zone contract contains no zone")
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    identifiers: set[str] = set()
    for zone in zones:
        if not isinstance(zone, Mapping):
            raise FWTileError("detail-zone entry is not an object")
        identifier = str(zone.get("id", ""))
        bounds = zone.get("bounds_l93_metres")
        if (
            not identifier
            or identifier in identifiers
            or not isinstance(bounds, list)
            or len(bounds) != 4
        ):
            raise FWTileError("detail-zone entry has invalid identity or bounds")
        active = tuple(map(float, bounds))
        if active[2] <= active[0] or active[3] <= active[1]:
            raise FWTileError(f"detail zone {identifier} has empty bounds")
        identifiers.add(identifier)
        result.append((identifier, active))
    return result


def _tile_detail_zone_priority(
    tile: Mapping[str, Any],
    zones: Sequence[tuple[str, Sequence[float]]],
) -> int | None:
    bounds = tile.get("bounds_l93_m")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise FWTileError(f"tile {tile.get('id')} has invalid priority bounds")
    xmin, ymin, xmax, ymax = map(float, bounds)
    for index, (_identifier, zone_bounds) in enumerate(zones):
        zone_xmin, zone_ymin, zone_xmax, zone_ymax = map(float, zone_bounds)
        if (
            xmin < zone_xmax
            and xmax > zone_xmin
            and ymin < zone_ymax
            and ymax > zone_ymin
        ):
            return index
    return None


def _prioritize_tiles(
    tiles: Sequence[dict[str, Any]],
    zones: Sequence[tuple[str, Sequence[float]]],
) -> list[dict[str, Any]]:
    def key(tile: Mapping[str, Any]) -> tuple[int, int, str]:
        priority = _tile_detail_zone_priority(tile, zones)
        return (
            0 if priority is not None else 1,
            priority if priority is not None else len(zones),
            str(tile["id"]),
        )

    return sorted(tiles, key=key)


def _ready_tiles(
    context: ExportContext,
    *,
    detail_zones: Sequence[tuple[str, Sequence[float]]] | None = None,
) -> list[dict[str, Any]]:
    ready = [
        item
        for item in context.manifest["tiles"]
        if item.get("status", {}).get("state") == "ready"
    ]
    return _prioritize_tiles(
        ready,
        detail_zones if detail_zones is not None else _load_detail_zone_bounds(),
    )


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--global-vector-package", type=Path)
    parser.add_argument("--far-terrain", type=Path)
    parser.add_argument("--far-imagery", type=Path)
    parser.add_argument("--detail-zones", type=Path, default=DETAIL_ZONE_CONTRACT_PATH)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--tile-id", action="append")
    selection.add_argument("--all-ready", action="store_true")
    selection.add_argument("--global-only", action="store_true")
    parser.add_argument(
        "--detail-content-mode",
        choices=sorted(DETAIL_CONTENT_MODES),
        default="complete_3d",
    )
    parser.add_argument("--far-imagery-resolution-m", type=float, default=2.0)
    parser.add_argument("--disable-near-lod", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    artifact_root = arguments.artifact_root.resolve()
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root
        else artifact_root / "unity-remote-catalog"
    )
    context = create_context(
        (
            arguments.production_manifest.resolve()
            if arguments.production_manifest
            else artifact_root / "global-05m/production-manifest.json"
        ),
        (
            None
            if arguments.global_only or arguments.detail_content_mode != "complete_3d"
            else (
                arguments.global_vector_package.resolve()
                if arguments.global_vector_package
                else artifact_root
                / "blender/synthetic-zone-global-control-v5-vector-lod.json.gz"
            )
        ),
        output_root,
        near_lod_disabled=arguments.disable_near_lod,
        global_only=arguments.global_only,
        detail_content_mode=arguments.detail_content_mode,
    )
    far = export_far_assets(
        context,
        (
            arguments.far_terrain.resolve()
            if arguments.far_terrain
            else artifact_root / "terrain/mnt-global.cog.tif"
        ),
        (
            arguments.far_imagery.resolve()
            if arguments.far_imagery
            else artifact_root
            / "blender/synthetic-zone-ign-orthophoto-2m-display-v2.jpg"
        ),
        imagery_resolution_m=arguments.far_imagery_resolution_m,
    )
    if arguments.global_only:
        catalog = write_catalog(context, far)
        result = {
            "output_root": str(output_root),
            "selected_tile_count": 0,
            "catalog_tile_count": catalog["exported_detail_tile_count"],
            "catalog_sha256": sha256_file(output_root / "catalog.json"),
            "catalog_bytes": (output_root / "catalog.json").stat().st_size,
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    detail_zones = _load_detail_zone_bounds(arguments.detail_zones.resolve())
    ready = _ready_tiles(context, detail_zones=detail_zones)
    if arguments.all_ready:
        selected = ready
    else:
        requested = set(arguments.tile_id or [])
        by_id = {item["id"]: item for item in ready}
        missing = sorted(requested - by_id.keys())
        if missing:
            raise FWTileError(f"requested tile(s) are not ready: {', '.join(missing)}")
        selected = [by_id[tile_id] for tile_id in sorted(requested)]
    pending_priority_ids = {
        str(tile["id"])
        for tile in selected
        if _tile_detail_zone_priority(tile, detail_zones) is not None
    }
    records = []
    for tile in selected:
        records.append(export_tile(context, tile))
        pending_priority_ids.discard(str(tile["id"]))
        if not pending_priority_ids:
            # Publish a valid partial catalog as soon as the configured
            # attention zones are present. The same atomic path is replaced by
            # the final catalog when the deterministic order finishes.
            write_catalog(context, far)
            pending_priority_ids.add("__checkpoint_written__")
    catalog = write_catalog(context, far)
    result = {
        "output_root": str(output_root),
        "selected_tile_count": len(records),
        "catalog_tile_count": catalog["exported_detail_tile_count"],
        "catalog_sha256": sha256_file(output_root / "catalog.json"),
        "catalog_bytes": (output_root / "catalog.json").stat().st_size,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FWTileError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
