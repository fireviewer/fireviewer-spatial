"""Pure-Python contract helpers for the tiled Blender detail scene.

The production manifest is intentionally small.  It references one portable
asset set per 500 m core tile, so Blender can create collection placeholders
for the whole AOI while loading only the attention tiles required for a
working session.  No raster, point cloud or Blender dependency is imported by
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


GLOBAL_05M_MANIFEST_SCHEMA = "fireviewer.global-05m-production-manifest.v1"
READY_STATE = "ready"
NATIVE_NEAR_ORTHOPHOTO_RESOLUTION_M = 0.2
_TILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TileAssetSelection:
    """Resolved portable assets for one ready tile."""

    kind: str
    primary_path: Path
    orthophoto_source_path: Path | None = None
    expected_sha256: str | None = None
    orthophoto_expected_sha256: str | None = None
    orthophoto_resolution_m: float | None = None
    orthophoto_lod: str | None = None
    library_collection_name: str | None = None


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _origin(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must contain Lambert-93 x, y and z")
    return tuple(
        _finite_number(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )


def _bounds(value: Any, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field_name} must contain west, south, east and north")
    west, south, east, north = (
        _finite_number(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )
    if east <= west or north <= south:
        raise ValueError(f"{field_name} must have a strictly positive area")
    return west, south, east, north


def _portable_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or path.drive or value.startswith(("/", "\\")):
        raise ValueError(f"{field_name} must be relative to the manifest")
    return value


def _asset_path_if_present(
    asset: Mapping[str, Any], key: str, field_name: str
) -> str | None:
    value = asset.get(key)
    if value is None:
        return None
    return _portable_path(value, field_name)


def validate_global_05m_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the portable tile index without touching referenced assets."""

    if manifest.get("schema") != GLOBAL_05M_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported global 0.5 m manifest schema: {manifest.get('schema')!r}"
        )
    if manifest.get("crs") != "EPSG:2154":
        raise ValueError("Global 0.5 m manifest must use EPSG:2154")
    root_origin = _origin(manifest.get("origin_l93_m"), "origin_l93_m")
    aoi = manifest.get("aoi")
    if not isinstance(aoi, Mapping):
        raise ValueError("Global 0.5 m manifest is missing aoi")
    _bounds(aoi.get("bounds_l93_m"), "aoi.bounds_l93_m")
    if _finite_number(aoi.get("area_m2"), "aoi.area_m2") <= 0.0:
        raise ValueError("aoi.area_m2 must be strictly positive")
    aoi_sha256 = aoi.get("sha256")
    if aoi_sha256 is not None and (
        not isinstance(aoi_sha256, str) or not _SHA256_PATTERN.fullmatch(aoi_sha256)
    ):
        raise ValueError("aoi.sha256 must be a lowercase SHA-256 digest")

    tiling = manifest.get("tiling")
    if not isinstance(tiling, Mapping):
        raise ValueError("Global 0.5 m manifest is missing tiling")
    source_size = _finite_number(
        tiling.get("source_tile_size_m"), "tiling.source_tile_size_m"
    )
    output_size = _finite_number(
        tiling.get("output_tile_size_m"), "tiling.output_tile_size_m"
    )
    halo = _finite_number(tiling.get("halo_m"), "tiling.halo_m")
    if source_size <= 0.0 or output_size <= 0.0 or halo < 0.0:
        raise ValueError("Tile sizes must be positive and halo_m non-negative")
    if tiling.get("ownership_rule") not in {
        "apex_in_half_open_core",
        "apex_in_half_open_core_min_inclusive_max_exclusive",
    }:
        raise ValueError("Unsupported vegetation ownership rule")

    tiles = manifest.get("tiles")
    if not isinstance(tiles, list):
        raise ValueError("Global 0.5 m manifest tiles must be a list")
    identifiers: set[str] = set()
    for index, tile in enumerate(tiles):
        prefix = f"tiles[{index}]"
        if not isinstance(tile, Mapping):
            raise ValueError(f"{prefix} must be an object")
        identifier = tile.get("id")
        if not isinstance(identifier, str) or not _TILE_ID_PATTERN.fullmatch(
            identifier
        ):
            raise ValueError(f"{prefix}.id is not a portable tile identifier")
        if identifier in identifiers:
            raise ValueError(f"Duplicate tile id: {identifier}")
        identifiers.add(identifier)

        core = _bounds(tile.get("bounds_l93_m"), f"{prefix}.bounds_l93_m")
        processing = _bounds(
            tile.get("processing_bounds_l93_m"),
            f"{prefix}.processing_bounds_l93_m",
        )
        if not (
            processing[0] <= core[0]
            and processing[1] <= core[1]
            and processing[2] >= core[2]
            and processing[3] >= core[3]
        ):
            raise ValueError(f"{prefix}.processing_bounds_l93_m must contain the core")
        if not math.isclose(
            core[2] - core[0], output_size, abs_tol=0.001
        ) or not math.isclose(core[3] - core[1], output_size, abs_tol=0.001):
            raise ValueError(f"{prefix}.bounds_l93_m does not match output_tile_size_m")
        tile_origin = _origin(tile.get("origin_l93_m"), f"{prefix}.origin_l93_m")
        if any(
            abs(left - right) > 0.001
            for left, right in zip(tile_origin, root_origin, strict=True)
        ):
            raise ValueError(f"{prefix}.origin_l93_m differs from the global origin")
        area = _finite_number(
            tile.get("aoi_intersection_area_m2"),
            f"{prefix}.aoi_intersection_area_m2",
        )
        if area <= 0.0 or area > output_size * output_size + 0.001:
            raise ValueError(f"{prefix}.aoi_intersection_area_m2 is outside tile area")
        source_ids = tile.get("source_tile_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError(f"{prefix}.source_tile_ids must be a non-empty list")

        status = tile.get("status")
        if not isinstance(status, Mapping) or status.get("state") not in {
            "pending",
            "incomplete",
            "planned",
            "running",
            READY_STATE,
            "failed",
        }:
            raise ValueError(f"{prefix}.status.state is invalid")
        assets = tile.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError(f"{prefix}.assets must be an object")
        library = assets.get("blender_library")
        if library is not None:
            if not isinstance(library, Mapping):
                raise ValueError(f"{prefix}.assets.blender_library must be an object")
            _asset_path_if_present(
                library, "path", f"{prefix}.assets.blender_library.path"
            )
            collection_name = library.get("collection_name")
            if collection_name is not None and (
                not isinstance(collection_name, str) or not collection_name.strip()
            ):
                raise ValueError(
                    f"{prefix}.assets.blender_library.collection_name is invalid"
                )
        mid = assets.get("mid_package")
        if mid is not None:
            if not isinstance(mid, Mapping):
                raise ValueError(f"{prefix}.assets.mid_package must be an object")
            _asset_path_if_present(mid, "path", f"{prefix}.assets.mid_package.path")
            digest = mid.get("sha256")
            if digest is not None and (
                not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest)
            ):
                raise ValueError(f"{prefix}.assets.mid_package.sha256 is invalid")
        orthophoto = assets.get("orthophoto")
        if orthophoto is not None:
            # Compatibility with the early draft of the production contract.
            if not isinstance(orthophoto, Mapping):
                raise ValueError(f"{prefix}.assets.orthophoto must be an object")
            for key in ("source_manifest_path", "image_path", "geotiff_path"):
                _asset_path_if_present(
                    orthophoto, key, f"{prefix}.assets.orthophoto.{key}"
                )
        for key in (
            "orthophoto_source",
            "orthophoto_image",
            "orthophoto_geotiff",
            "near_orthophoto_source",
            "near_orthophoto_image",
            "near_orthophoto_geotiff",
        ):
            asset = assets.get(key)
            if asset is not None:
                if not isinstance(asset, Mapping):
                    raise ValueError(f"{prefix}.assets.{key} must be an object")
                _asset_path_if_present(asset, "path", f"{prefix}.assets.{key}.path")
        near_request = tile.get("near_orthophoto_request")
        if near_request is not None:
            if not isinstance(near_request, Mapping):
                raise ValueError(f"{prefix}.near_orthophoto_request must be an object")
            resolution = _finite_number(
                near_request.get("resolution_m"),
                f"{prefix}.near_orthophoto_request.resolution_m",
            )
            if not math.isclose(
                resolution, NATIVE_NEAR_ORTHOPHOTO_RESOLUTION_M, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{prefix}.near_orthophoto_request must use 0.20 m resolution"
                )
        if status.get("state") == READY_STATE:
            library_path = library.get("path") if isinstance(library, Mapping) else None
            mid_path = mid.get("path") if isinstance(mid, Mapping) else None
            ortho_source = assets.get("orthophoto_source")
            ortho_path = (
                ortho_source.get("path")
                if isinstance(ortho_source, Mapping)
                else (
                    orthophoto.get("source_manifest_path")
                    if isinstance(orthophoto, Mapping)
                    else None
                )
            )
            if not library_path and not (mid_path and ortho_path):
                raise ValueError(
                    f"{prefix} is ready but has neither a Blender library nor "
                    "a mid package plus orthophoto source manifest"
                )
        visibility = tile.get("visibility")
        if not isinstance(visibility, Mapping) or not isinstance(
            visibility.get("default_visible"), bool
        ):
            raise ValueError(f"{prefix}.visibility.default_visible must be boolean")


def load_global_05m_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json_object(manifest_path)
    validate_global_05m_manifest(manifest)
    return manifest


def resolve_manifest_asset(manifest_path: str | Path, relative_path: str) -> Path:
    portable = _portable_path(relative_path, "asset path")
    return (Path(manifest_path).expanduser().resolve().parent / portable).resolve()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_tile_asset(
    manifest_path: str | Path,
    tile: Mapping[str, Any],
    *,
    require_exists: bool = True,
) -> TileAssetSelection:
    """Prefer native near imagery, then a library, then the 50 cm fallback."""

    if tile.get("status", {}).get("state") != READY_STATE:
        raise ValueError(f"Tile {tile.get('id')!r} is not ready")
    assets = tile["assets"]
    mid = assets.get("mid_package") or {}
    mid_relative = mid.get("path")
    near_source = assets.get("near_orthophoto_source") or {}
    near_relative = near_source.get("path")
    if mid_relative and near_relative:
        mid_path = resolve_manifest_asset(manifest_path, mid_relative)
        near_path = resolve_manifest_asset(manifest_path, near_relative)
        # The 20 cm profile is optional. Its absent path must not make an
        # otherwise ready legacy tile unusable, while a present corrupt file
        # must fail loudly rather than silently degrading image quality.
        if near_path.is_file() and (mid_path.is_file() or not require_exists):
            expected = mid.get("sha256")
            near_expected = near_source.get("sha256")
            if require_exists and expected and file_sha256(mid_path) != expected:
                raise ValueError(f"Tile {tile['id']!r} mid package SHA-256 mismatch")
            if near_expected and file_sha256(near_path) != near_expected:
                raise ValueError(
                    f"Tile {tile['id']!r} near orthophoto source SHA-256 mismatch"
                )
            return TileAssetSelection(
                kind="source_packages",
                primary_path=mid_path,
                orthophoto_source_path=near_path,
                expected_sha256=expected,
                orthophoto_expected_sha256=near_expected,
                orthophoto_resolution_m=NATIVE_NEAR_ORTHOPHOTO_RESOLUTION_M,
                orthophoto_lod="near_native_0m20",
            )

    library = assets.get("blender_library") or {}
    library_relative = library.get("path")
    if library_relative:
        library_path = resolve_manifest_asset(manifest_path, library_relative)
        if library_path.is_file() or not require_exists:
            expected = library.get("sha256")
            if require_exists and expected and file_sha256(library_path) != expected:
                raise ValueError(
                    f"Tile {tile['id']!r} Blender library SHA-256 mismatch"
                )
            return TileAssetSelection(
                kind="blender_library",
                primary_path=library_path,
                expected_sha256=expected,
                library_collection_name=library.get("collection_name"),
            )

    orthophoto = assets.get("orthophoto") or {}
    orthophoto_source = assets.get("orthophoto_source") or {}
    orthophoto_relative = orthophoto_source.get("path") or orthophoto.get(
        "source_manifest_path"
    )
    if mid_relative and orthophoto_relative:
        mid_path = resolve_manifest_asset(manifest_path, mid_relative)
        orthophoto_path = resolve_manifest_asset(manifest_path, orthophoto_relative)
        if not require_exists or (mid_path.is_file() and orthophoto_path.is_file()):
            expected = mid.get("sha256")
            orthophoto_expected = orthophoto_source.get("sha256")
            if require_exists and expected and file_sha256(mid_path) != expected:
                raise ValueError(f"Tile {tile['id']!r} mid package SHA-256 mismatch")
            if (
                require_exists
                and orthophoto_expected
                and file_sha256(orthophoto_path) != orthophoto_expected
            ):
                raise ValueError(
                    f"Tile {tile['id']!r} orthophoto source SHA-256 mismatch"
                )
            return TileAssetSelection(
                kind="source_packages",
                primary_path=mid_path,
                orthophoto_source_path=orthophoto_path,
                expected_sha256=expected,
                orthophoto_expected_sha256=orthophoto_expected,
                orthophoto_resolution_m=float(
                    tile.get("orthophoto_request", {}).get("resolution_m", 0.5)
                ),
                orthophoto_lod="mid_0m50_fallback",
            )
    if require_exists:
        raise FileNotFoundError(
            f"Tile {tile.get('id')!r} has no complete local ready asset set"
        )
    raise ValueError(f"Tile {tile.get('id')!r} has no usable asset contract")


def ready_tiles(
    manifest: Mapping[str, Any], selected_ids: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Return ready tiles in stable id order and validate explicit selections."""

    tiles = {tile["id"]: tile for tile in manifest["tiles"]}
    if selected_ids is None:
        selected = [
            tile for tile in tiles.values() if tile["status"]["state"] == READY_STATE
        ]
    else:
        identifiers = list(dict.fromkeys(selected_ids))
        missing = [identifier for identifier in identifiers if identifier not in tiles]
        if missing:
            raise ValueError(f"Unknown tile ids: {', '.join(sorted(missing))}")
        not_ready = [
            identifier
            for identifier in identifiers
            if tiles[identifier]["status"]["state"] != READY_STATE
        ]
        if not_ready:
            raise ValueError(f"Tiles are not ready: {', '.join(sorted(not_ready))}")
        selected = [tiles[identifier] for identifier in identifiers]
    return sorted(selected, key=lambda tile: tile["id"])


def tile_distance_to_point_m(
    bounds_l93_m: Sequence[float], point_l93_m: Sequence[float]
) -> float:
    west, south, east, north = _bounds(list(bounds_l93_m), "bounds_l93_m")
    if len(point_l93_m) != 2:
        raise ValueError("point_l93_m must contain x and y")
    x = _finite_number(point_l93_m[0], "point_l93_m[0]")
    y = _finite_number(point_l93_m[1], "point_l93_m[1]")
    delta_x = max(west - x, 0.0, x - east)
    delta_y = max(south - y, 0.0, y - north)
    return math.hypot(delta_x, delta_y)


def tile_is_visible(
    tile: Mapping[str, Any],
    *,
    focus_l93_m: Sequence[float] | None = None,
    radius_m: float | None = None,
    explicitly_selected: bool = False,
) -> bool:
    if explicitly_selected:
        return True
    if focus_l93_m is None and radius_m is None:
        return bool(tile["visibility"]["default_visible"])
    if focus_l93_m is None or radius_m is None:
        raise ValueError("focus_l93_m and radius_m must be provided together")
    radius = _finite_number(radius_m, "radius_m")
    if radius < 0.0:
        raise ValueError("radius_m must be non-negative")
    return tile_distance_to_point_m(tile["bounds_l93_m"], focus_l93_m) <= radius


__all__ = [
    "GLOBAL_05M_MANIFEST_SCHEMA",
    "READY_STATE",
    "TileAssetSelection",
    "file_sha256",
    "load_global_05m_manifest",
    "ready_tiles",
    "resolve_manifest_asset",
    "select_tile_asset",
    "tile_distance_to_point_m",
    "tile_is_visible",
    "validate_global_05m_manifest",
]
