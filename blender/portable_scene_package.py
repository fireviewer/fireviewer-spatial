"""Seal and validate the portable map/perimeter packages consumed by FireViewer.

The producers remain filesystem-only and database-free.  This module adds the
small immutable metadata layer shared by the production API, the browser upload
and the backend import.  It never changes the authored OpenUSD scene.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from fixed_terrain_grid import lod_absolute_heights_mm, read_fixed_terrain
except ImportError:  # pragma: no cover - package import
    from blender.fixed_terrain_grid import (
        lod_absolute_heights_mm,
        read_fixed_terrain,
    )


INVENTORY_SCHEMA = "fireviewer.portable-package-inventory.v1"
MAP_MANIFEST_SCHEMA = "fireviewer.simple-measured-map-package.v2"
MAP_CONTRACT_SCHEMA = "fireviewer.simple-measured-map-upload-contract.v2"
LEGACY_MAP_MANIFEST_SCHEMA = "fireviewer.simple-measured-map-package.v1"
LEGACY_MAP_CONTRACT_SCHEMA = "fireviewer.simple-measured-map-upload-contract.v1"
PERIMETER_MANIFEST_SCHEMA = "fireviewer.observed-perimeter-package.v1"
PERIMETER_CONTRACT_SCHEMA = "fireviewer.observed-perimeter-upload-contract.v1"
MAP_CONTRACT_PATH = "contracts/map-contract.json"
PERIMETER_CONTRACT_PATH = "contracts/perimeter-contract.json"
MANIFEST_NAME = "manifest.json"
INVENTORY_NAME = "dependency-inventory.json"
MAP_ENTRY_STAGE = "zone.usda"
MAP_STANDALONE_SCENE = "zone.blend"
MAP_RECEIPT_NAME = "zone.done.json"
PERIMETER_STAGE_NAME = "geographic-perimeters.usda"
PERIMETER_TIMELINE_NAME = "fire-progression-timeline.json"
PERIMETER_SOURCE_MANIFEST_NAME = "perimeter-layer.manifest.json"
PERIMETER_VIEWER_MANIFEST_NAME = "perimeter-viewer.manifest.json"
PERIMETER_PREVIEW_ROOT = "preview"
MAP_GALLERY_RECEIPT = "qa/zone-gallery-receipt.v1.json"
_METADATA_PATHS = {
    MANIFEST_NAME,
    INVENTORY_NAME,
    MAP_CONTRACT_PATH,
    PERIMETER_CONTRACT_PATH,
}
_MAP_EXCLUDED_FILES = {
    "dataset-entry.json",
    "dataset-publication.json",
    "fireviewer-zone.zip",
    "job-status.json",
}
_MAP_EXCLUDED_ROOTS = {"sources", "download"}
_SHA256_LENGTH = 64


class PortableScenePackageError(RuntimeError):
    """The portable package is incomplete, unsafe or no longer immutable."""


@dataclass(frozen=True, slots=True)
class MapPackageReference:
    package_id: str
    revision: int
    zone_id: str
    map_build_id: str
    contract_sha256: str
    manifest_sha256: str
    bounds_l93_m: tuple[float, float, float, float]
    origin_l93_m: tuple[float, float, float]

    def payload(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "revision": self.revision,
            "zone_id": self.zone_id,
            "map_build_id": self.map_build_id,
            "contract_sha256": self.contract_sha256,
            "horizontal_crs": "EPSG:2154",
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(value: str) -> str:
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or any(character in value for character in ("?", "#", ":"))
    ):
        raise PortableScenePackageError(f"unsafe package path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PortableScenePackageError(f"unsafe package path: {value!r}")
    return path.as_posix()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortableScenePackageError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise PortableScenePackageError(f"{label} must be a JSON object")
    return value


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _map_payload_paths(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        if (
            relative in _METADATA_PATHS
            or path.name in _MAP_EXCLUDED_FILES
            or parts[0] in _MAP_EXCLUDED_ROOTS
            or path.name.endswith(".part")
        ):
            continue
        result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def _perimeter_payload_paths(root: Path) -> list[Path]:
    result = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in _METADATA_PATHS
        and not path.name.endswith(".part")
    ]
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def _inventory(root: Path, files: Iterable[Path], role: str) -> dict[str, Any]:
    records = [
        {
            "path": _safe_path(path.relative_to(root).as_posix()),
            "byte_count": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    return {
        "schema": INVENTORY_SCHEMA,
        "status": "sealed",
        "package_role": role,
        "file_count": len(records),
        "files": records,
    }


def _inventory_by_path(
    value: Mapping[str, Any], role: str
) -> dict[str, dict[str, Any]]:
    if (
        value.get("schema") != INVENTORY_SCHEMA
        or value.get("status") != "sealed"
        or value.get("package_role") != role
    ):
        raise PortableScenePackageError("portable dependency inventory is invalid")
    files = value.get("files")
    if (
        not isinstance(files, list)
        or value.get("file_count") != len(files)
        or not files
    ):
        raise PortableScenePackageError("portable dependency inventory is incomplete")
    result: dict[str, dict[str, Any]] = {}
    for raw in files:
        if not isinstance(raw, dict):
            raise PortableScenePackageError("portable inventory entry is invalid")
        path = _safe_path(raw.get("path") if isinstance(raw.get("path"), str) else "")
        byte_count = raw.get("byte_count")
        sha256 = raw.get("sha256")
        if (
            path in result
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count <= 0
            or not isinstance(sha256, str)
            or len(sha256) != _SHA256_LENGTH
        ):
            raise PortableScenePackageError("portable inventory entry is invalid")
        result[path] = raw
    return result


def _map_height_range(
    root: Path, inventory: Mapping[str, Mapping[str, Any]]
) -> tuple[float, float]:
    terrain_paths = sorted(
        path
        for path in inventory
        if path.startswith("packages/") and path.endswith("/terrain.fvtg")
    )
    if not terrain_paths:
        raise PortableScenePackageError("map package has no fixed terrain payload")
    minimum_mm: int | None = None
    maximum_mm: int | None = None
    for relative in terrain_paths:
        tile = read_fixed_terrain(root / relative)
        heights = lod_absolute_heights_mm(tile, 0)
        current_minimum = int(heights.min())
        current_maximum = int(heights.max())
        minimum_mm = (
            current_minimum if minimum_mm is None else min(minimum_mm, current_minimum)
        )
        maximum_mm = (
            current_maximum if maximum_mm is None else max(maximum_mm, current_maximum)
        )
    assert minimum_mm is not None and maximum_mm is not None
    if maximum_mm < minimum_mm:
        raise PortableScenePackageError("map elevation range is invalid")
    return minimum_mm / 1000.0, maximum_mm / 1000.0


def _finite_bounds(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise PortableScenePackageError(f"{label} must contain {size} coordinates")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise PortableScenePackageError(f"{label} contains non-finite coordinates")
    return result


def _map_control_gallery(
    package_root: Path, by_path: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    receipt = _read_json(package_root / MAP_GALLERY_RECEIPT, "map control gallery")
    receipt_record = by_path.get(MAP_GALLERY_RECEIPT)
    captures = receipt.get("captures")
    if (
        receipt_record is None
        or receipt.get("schema") != "fireviewer.simple-zone-gallery-receipt.v1"
        or receipt.get("status") != "rendered_pending_human_review"
        or receipt.get("human_review_required") is not True
        or receipt.get("accepted_human") is not False
        or receipt.get("capture_count") != 20
        or not isinstance(captures, list)
        or len(captures) != 20
    ):
        raise PortableScenePackageError("map control gallery is incomplete")
    declared_content = receipt.get("receipt_content_sha256")
    receipt_content = dict(receipt)
    receipt_content.pop("receipt_content_sha256", None)
    if declared_content != _sha256_bytes(_canonical_bytes(receipt_content)):
        raise PortableScenePackageError("map control gallery receipt changed")
    if receipt.get("capture_set_sha256") != _sha256_bytes(_canonical_bytes(captures)):
        raise PortableScenePackageError("map control gallery set changed")
    records: list[dict[str, Any]] = []
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise PortableScenePackageError("map control capture is invalid")
        artifact = capture.get("artifact")
        capture_id = capture.get("capture_id")
        category = capture.get("category")
        if (
            not isinstance(artifact, dict)
            or not isinstance(capture_id, str)
            or not capture_id.startswith(f"{index:02d}-")
            or not isinstance(category, str)
        ):
            raise PortableScenePackageError("map control capture identity is invalid")
        path = artifact.get("path")
        inventory_record = by_path.get(path) if isinstance(path, str) else None
        if (
            inventory_record is None
            or inventory_record.get("sha256") != artifact.get("sha256")
            or inventory_record.get("byte_count") != artifact.get("byte_count")
        ):
            raise PortableScenePackageError("map control capture is missing or changed")
        records.append(
            {
                "index": index,
                "capture_id": capture_id,
                "category": category,
                "group": "general" if index < 4 else "detail",
                "caption": f"{capture_id} — {category}",
                "path": path,
                "sha256": inventory_record["sha256"],
                "byte_count": inventory_record["byte_count"],
            }
        )
    return {
        "receipt_path": MAP_GALLERY_RECEIPT,
        "receipt_sha256": receipt_record["sha256"],
        "capture_count": 20,
        "general_count": 4,
        "detail_count": 16,
        "captures": records,
    }


def seal_map_upload_package(root: Path | str) -> dict[str, Any]:
    """Write the canonical browser/backend metadata into one produced map root."""

    package_root = Path(root).resolve(strict=True)
    plan = _read_json(package_root / "zone-plan.json", "zone plan")
    receipt = _read_json(package_root / MAP_RECEIPT_NAME, "zone receipt")
    if (
        plan.get("schema") != "fireviewer.simple-measured-zone-plan.v1"
        or plan.get("status") != "planned"
        or receipt.get("schema") != "fireviewer.simple-measured-zone-production.v1"
        or receipt.get("status") != "technical_scene_produced"
        or receipt.get("accepted_human") is not False
        or receipt.get("zone_id") != plan.get("zone_id")
    ):
        raise PortableScenePackageError(
            "map plan and receipt are not a technical map product"
        )
    zone_id = plan.get("zone_id")
    build_id = receipt.get("build_id")
    if (
        not isinstance(zone_id, str)
        or not zone_id.startswith("GPS-")
        or not isinstance(build_id, str)
        or len(build_id) != _SHA256_LENGTH
    ):
        raise PortableScenePackageError("map identity is not site-compatible")
    entry_stage = package_root / MAP_ENTRY_STAGE
    if not entry_stage.is_file():
        raise PortableScenePackageError("zone.usda is missing")
    inventory = _inventory(package_root, _map_payload_paths(package_root), "map")
    by_path = _inventory_by_path(inventory, "map")
    for required in (
        MAP_ENTRY_STAGE,
        MAP_STANDALONE_SCENE,
        MAP_RECEIPT_NAME,
        "zone-plan.json",
    ):
        if required not in by_path:
            raise PortableScenePackageError(f"map package is missing {required}")
    inventory_bytes = _json_bytes(inventory)
    inventory_reference = {
        "path": INVENTORY_NAME,
        "byte_count": len(inventory_bytes),
        "sha256": _sha256_bytes(inventory_bytes),
        "file_count": inventory["file_count"],
    }
    production_bounds = _finite_bounds(
        plan.get("production_bounds_l93_m"), 4, "production bounds"
    )
    if not (
        plan.get("crs") == "EPSG:2154"
        and production_bounds[0] < production_bounds[2]
        and production_bounds[1] < production_bounds[3]
    ):
        raise PortableScenePackageError("map production bounds are invalid")
    minimum_height, maximum_height = _map_height_range(package_root, by_path)
    spatial_reference = {
        "horizontal_crs": "EPSG:2154",
        "vertical_datum": "NGF-IGN69",
        "up_axis": "Z",
        "meters_per_unit": 1,
        "bounds_l93_m": list(production_bounds),
        "local_origin_l93_m": [
            production_bounds[0],
            production_bounds[1],
            minimum_height,
        ],
        "height_minimum_ngf_ign69_m": minimum_height,
        "height_maximum_ngf_ign69_m": maximum_height,
    }
    package_id = f"map-{zone_id.lower()}-{build_id[:12]}"
    manifest = {
        "schema": MAP_MANIFEST_SCHEMA,
        "status": "active",
        "package_id": package_id,
        "revision": 1,
        "zone_id": zone_id,
        "map_build_id": build_id,
        "entry_stage": MAP_ENTRY_STAGE,
        "entry_stage_sha256": by_path[MAP_ENTRY_STAGE]["sha256"],
        "standalone_scene": MAP_STANDALONE_SCENE,
        "standalone_scene_sha256": by_path[MAP_STANDALONE_SCENE]["sha256"],
        "zone_receipt": {
            "path": MAP_RECEIPT_NAME,
            "sha256": by_path[MAP_RECEIPT_NAME]["sha256"],
        },
        "dependency_inventory": inventory_reference,
        "spatial_reference": spatial_reference,
        "publication": {
            "state": "technical_unpublished",
            "accepted_human": False,
            "automatic_publication": False,
        },
        "capabilities": {
            "openusd_scene": True,
            "embedded_assets": True,
            "control_gallery": False,
            "perimeter_layers_separate": True,
        },
    }
    manifest_bytes = _json_bytes(manifest)
    contract = {
        "schema": MAP_CONTRACT_SCHEMA,
        "contract_status": "active",
        "package": {
            "package_id": package_id,
            "revision": 1,
            "map_build_id": build_id,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "entry_stage": MAP_ENTRY_STAGE,
            "entry_stage_sha256": by_path[MAP_ENTRY_STAGE]["sha256"],
            "standalone_scene": MAP_STANDALONE_SCENE,
            "standalone_scene_sha256": by_path[MAP_STANDALONE_SCENE]["sha256"],
            "zone_receipt": MAP_RECEIPT_NAME,
            "zone_receipt_sha256": by_path[MAP_RECEIPT_NAME]["sha256"],
        },
        "spatial_reference": spatial_reference,
        "release": {
            "upload_allowed": True,
            "automatic_publication": False,
            "initial_site_state": "downloadable_after_server_validation",
            "accepted_human": False,
        },
        "reuse": {
            "published_map_allowed": True,
            "unpublished_local_map_allowed": True,
            "terrain_rebuild_by_simulation": "forbidden",
            "perimeter_rebuild_by_simulation": "forbidden",
        },
    }
    _write_atomic(package_root / INVENTORY_NAME, inventory_bytes)
    _write_atomic(package_root / MANIFEST_NAME, manifest_bytes)
    _write_atomic(package_root / MAP_CONTRACT_PATH, _json_bytes(contract))
    validate_map_upload_package(package_root)
    return manifest


def validate_map_upload_package(root: Path | str) -> MapPackageReference:
    package_root = Path(root).resolve(strict=True)
    manifest_path = package_root / MANIFEST_NAME
    inventory_path = package_root / INVENTORY_NAME
    contract_path = package_root / MAP_CONTRACT_PATH
    manifest = _read_json(manifest_path, "map upload manifest")
    inventory = _read_json(inventory_path, "map dependency inventory")
    contract = _read_json(contract_path, "map upload contract")
    if (
        manifest.get("schema") != MAP_MANIFEST_SCHEMA
        or manifest.get("status") != "active"
    ):
        raise PortableScenePackageError("map upload manifest is incompatible")
    if (
        contract.get("schema") != MAP_CONTRACT_SCHEMA
        or contract.get("contract_status") != "active"
    ):
        raise PortableScenePackageError("map upload contract is incompatible")
    by_path = _inventory_by_path(inventory, "map")
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in _map_payload_paths(package_root)
    }
    if set(by_path) != actual_paths:
        raise PortableScenePackageError(
            "map dependency inventory does not match the package"
        )
    for relative, record in by_path.items():
        path = package_root / relative
        if (
            path.stat().st_size != record["byte_count"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise PortableScenePackageError(f"map dependency changed: {relative}")
    inventory_reference = manifest.get("dependency_inventory")
    if not isinstance(inventory_reference, dict) or inventory_reference != {
        "path": INVENTORY_NAME,
        "byte_count": inventory_path.stat().st_size,
        "sha256": _sha256_file(inventory_path),
        "file_count": len(by_path),
    }:
        raise PortableScenePackageError("map inventory reference is invalid")
    package = contract.get("package")
    spatial = contract.get("spatial_reference")
    publication = manifest.get("publication")
    capabilities = manifest.get("capabilities")
    if (
        not isinstance(package, dict)
        or not isinstance(spatial, dict)
        or package.get("manifest_sha256") != _sha256_file(manifest_path)
        or package.get("package_id") != manifest.get("package_id")
        or package.get("revision") != manifest.get("revision")
        or package.get("map_build_id") != manifest.get("map_build_id")
        or package.get("entry_stage_sha256")
        != by_path.get(MAP_ENTRY_STAGE, {}).get("sha256")
        or package.get("standalone_scene") != MAP_STANDALONE_SCENE
        or manifest.get("standalone_scene") != MAP_STANDALONE_SCENE
        or package.get("standalone_scene_sha256")
        != by_path.get(MAP_STANDALONE_SCENE, {}).get("sha256")
        or manifest.get("standalone_scene_sha256")
        != by_path.get(MAP_STANDALONE_SCENE, {}).get("sha256")
        or package.get("zone_receipt_sha256")
        != by_path.get(MAP_RECEIPT_NAME, {}).get("sha256")
        or spatial != manifest.get("spatial_reference")
        or "control_gallery" in manifest
        or "control_gallery" in contract
        or not isinstance(capabilities, dict)
        or capabilities.get("control_gallery") is not False
        or not isinstance(publication, dict)
        or publication.get("automatic_publication") is not False
    ):
        raise PortableScenePackageError("map upload metadata is not causally bound")
    bounds = _finite_bounds(spatial.get("bounds_l93_m"), 4, "map bounds")
    origin = _finite_bounds(spatial.get("local_origin_l93_m"), 3, "map origin")
    return MapPackageReference(
        package_id=str(manifest["package_id"]),
        revision=int(manifest["revision"]),
        zone_id=str(manifest["zone_id"]),
        map_build_id=str(manifest["map_build_id"]),
        contract_sha256=_sha256_file(contract_path),
        manifest_sha256=_sha256_file(manifest_path),
        bounds_l93_m=(bounds[0], bounds[1], bounds[2], bounds[3]),
        origin_l93_m=(origin[0], origin[1], origin[2]),
    )


def _zip_common_root(names: list[str]) -> str:
    roots = {
        PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts
    }
    if len(roots) == 1 and all("/" in name for name in names):
        return f"{next(iter(roots))}/"
    return ""


def read_map_reference_from_archive(archive_path: Path | str) -> MapPackageReference:
    """Validate map metadata and every inventoried byte without extracting the ZIP."""

    path = Path(archive_path).resolve(strict=True)
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise PortableScenePackageError(f"invalid map archive: {error}") from error
    with archive:
        names: list[str] = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized = _safe_path(info.filename)
            if normalized != info.filename or normalized in names:
                raise PortableScenePackageError(
                    "map archive contains unsafe or duplicate paths"
                )
            names.append(normalized)
        prefix = _zip_common_root(names)
        relative_names = {name.removeprefix(prefix): name for name in names}
        required = {MANIFEST_NAME, INVENTORY_NAME, MAP_CONTRACT_PATH}
        if not required.issubset(relative_names):
            raise PortableScenePackageError("map archive lacks site upload metadata")

        def json_member(relative: str) -> tuple[bytes, dict[str, Any]]:
            raw = archive.read(relative_names[relative])
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PortableScenePackageError(
                    f"invalid {relative} in map archive"
                ) from error
            if not isinstance(value, dict):
                raise PortableScenePackageError(f"invalid {relative} in map archive")
            return raw, value

        manifest_raw, manifest = json_member(MANIFEST_NAME)
        inventory_raw, inventory = json_member(INVENTORY_NAME)
        contract_raw, contract = json_member(MAP_CONTRACT_PATH)
        by_path = _inventory_by_path(inventory, "map")
        if set(relative_names) != required | set(by_path):
            raise PortableScenePackageError(
                "map archive and dependency inventory differ"
            )
        for relative, record in by_path.items():
            with archive.open(relative_names[relative]) as stream:
                digest = hashlib.sha256()
                byte_count = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    byte_count += len(block)
            if (
                byte_count != record["byte_count"]
                or digest.hexdigest() != record["sha256"]
            ):
                raise PortableScenePackageError(
                    f"map archive dependency changed: {relative}"
                )
        inventory_reference = manifest.get("dependency_inventory")
        package = contract.get("package")
        spatial = contract.get("spatial_reference")
        control_gallery = contract.get("control_gallery")
        current_contract = (
            manifest.get("schema") == MAP_MANIFEST_SCHEMA
            and contract.get("schema") == MAP_CONTRACT_SCHEMA
        )
        legacy_contract = (
            manifest.get("schema") == LEGACY_MAP_MANIFEST_SCHEMA
            and contract.get("schema") == LEGACY_MAP_CONTRACT_SCHEMA
        )
        if (
            not (current_contract or legacy_contract)
            or not isinstance(inventory_reference, dict)
            or inventory_reference.get("sha256") != _sha256_bytes(inventory_raw)
            or inventory_reference.get("file_count") != len(by_path)
            or not isinstance(package, dict)
            or package.get("manifest_sha256") != _sha256_bytes(manifest_raw)
            or package.get("entry_stage_sha256")
            != by_path.get(MAP_ENTRY_STAGE, {}).get("sha256")
            or package.get("zone_receipt_sha256")
            != by_path.get(MAP_RECEIPT_NAME, {}).get("sha256")
            or not isinstance(spatial, dict)
        ):
            raise PortableScenePackageError("map archive metadata is not bound")
        if current_contract:
            capabilities = manifest.get("capabilities")
            if (
                package.get("standalone_scene") != MAP_STANDALONE_SCENE
                or manifest.get("standalone_scene") != MAP_STANDALONE_SCENE
                or package.get("standalone_scene_sha256")
                != by_path.get(MAP_STANDALONE_SCENE, {}).get("sha256")
                or manifest.get("standalone_scene_sha256")
                != by_path.get(MAP_STANDALONE_SCENE, {}).get("sha256")
                or "control_gallery" in manifest
                or "control_gallery" in contract
                or not isinstance(capabilities, dict)
                or capabilities.get("control_gallery") is not False
            ):
                raise PortableScenePackageError("map standalone scene is not bound")
        elif (
            not isinstance(control_gallery, dict)
            or manifest.get("control_gallery") != control_gallery
        ):
            raise PortableScenePackageError("legacy map gallery is not bound")
        bounds = _finite_bounds(spatial.get("bounds_l93_m"), 4, "map bounds")
        origin = _finite_bounds(spatial.get("local_origin_l93_m"), 3, "map origin")
        return MapPackageReference(
            package_id=str(manifest["package_id"]),
            revision=int(manifest["revision"]),
            zone_id=str(manifest["zone_id"]),
            map_build_id=str(manifest["map_build_id"]),
            contract_sha256=_sha256_bytes(contract_raw),
            manifest_sha256=_sha256_bytes(manifest_raw),
            bounds_l93_m=(bounds[0], bounds[1], bounds[2], bounds[3]),
            origin_l93_m=(origin[0], origin[1], origin[2]),
        )


def seal_perimeter_upload_package(
    root: Path | str,
    map_reference: MapPackageReference,
) -> dict[str, Any]:
    """Bind one observed timeline package to the exact map selected for upload."""

    package_root = Path(root).resolve(strict=True)
    source_manifest = _read_json(
        package_root / PERIMETER_SOURCE_MANIFEST_NAME, "perimeter source manifest"
    )
    timeline = _read_json(package_root / PERIMETER_TIMELINE_NAME, "fire timeline")
    if (
        source_manifest.get("schema")
        != "fireviewer.geographic-perimeter-layer-package.v1"
        or timeline.get("schema") != "fireviewer.fire-progression-timeline.v1"
        or timeline.get("build_id") != source_manifest.get("build_id")
        or timeline.get("between_observations") != "undefined"
        or timeline.get("prediction") != "none"
    ):
        raise PortableScenePackageError(
            "perimeter timeline is not an observed fixed progression"
        )
    frames = timeline.get("frames")
    if not isinstance(frames, list) or not frames:
        raise PortableScenePackageError("perimeter timeline has no observation")
    viewer_path = package_root / PERIMETER_PREVIEW_ROOT / PERIMETER_VIEWER_MANIFEST_NAME
    viewer = _read_json(viewer_path, "perimeter visual timeline")
    viewer_frames = viewer.get("frames")
    if (
        viewer.get("schema") != "fireviewer.geographic-perimeter-timeline-viewer.v1"
        or viewer.get("status") != "derived_visual_timeline"
        or viewer.get("authoritative") is not False
        or viewer.get("map_zone_id") != map_reference.zone_id
        or viewer.get("layer_build_id") != source_manifest.get("build_id")
        or viewer.get("between_observations") != "undefined"
        or not isinstance(viewer_frames, list)
        or len(viewer_frames) != len(frames)
        or viewer.get("frame_count") != len(frames)
    ):
        raise PortableScenePackageError("perimeter visual timeline is incompatible")
    inventory = _inventory(
        package_root, _perimeter_payload_paths(package_root), "perimeter"
    )
    by_path = _inventory_by_path(inventory, "perimeter")
    for required in (
        PERIMETER_STAGE_NAME,
        PERIMETER_TIMELINE_NAME,
        PERIMETER_SOURCE_MANIFEST_NAME,
        "perimeters.normalized.json",
        f"{PERIMETER_PREVIEW_ROOT}/{PERIMETER_VIEWER_MANIFEST_NAME}",
    ):
        if required not in by_path:
            raise PortableScenePackageError(f"perimeter package is missing {required}")
    visual_frames: list[dict[str, Any]] = []
    for index, record in enumerate(viewer_frames):
        if not isinstance(record, dict) or record.get("index") != index:
            raise PortableScenePackageError("perimeter visual frame is invalid")
        name = record.get("path")
        relative = (
            f"{PERIMETER_PREVIEW_ROOT}/{_safe_path(name)}"
            if isinstance(name, str)
            else ""
        )
        inventory_record = by_path.get(relative)
        if (
            not relative
            or PurePosixPath(str(name)).name != name
            or inventory_record is None
            or inventory_record["sha256"] != record.get("sha256")
            or inventory_record["byte_count"] != record.get("byte_count")
        ):
            raise PortableScenePackageError(
                "perimeter visual frame is missing or changed"
            )
        visual_frames.append(
            {
                "index": index,
                "observed_at": record.get("observed_at"),
                "caption": record.get("caption"),
                "path": relative,
                "sha256": inventory_record["sha256"],
                "byte_count": inventory_record["byte_count"],
            }
        )
    inventory_bytes = _json_bytes(inventory)
    base_map = map_reference.payload()
    layer_build_id = str(source_manifest["build_id"])
    package_id = (
        (
            f"perimeter-{str(source_manifest['dataset_id'])[:32]}-"
            f"{layer_build_id[:12]}-{map_reference.map_build_id[:12]}"
        )
        .lower()
        .replace("_", "-")
    )
    manifest = {
        "schema": PERIMETER_MANIFEST_SCHEMA,
        "status": "active",
        "package_id": package_id,
        "revision": 1,
        "layer_build_id": layer_build_id,
        "base_map": base_map,
        "entry_layer": PERIMETER_STAGE_NAME,
        "entry_layer_sha256": by_path[PERIMETER_STAGE_NAME]["sha256"],
        "timeline": {
            "path": PERIMETER_TIMELINE_NAME,
            "sha256": by_path[PERIMETER_TIMELINE_NAME]["sha256"],
            "state_count": len(frames),
        },
        "source_layer_manifest": {
            "path": PERIMETER_SOURCE_MANIFEST_NAME,
            "sha256": by_path[PERIMETER_SOURCE_MANIFEST_NAME]["sha256"],
        },
        "dependency_inventory": {
            "path": INVENTORY_NAME,
            "byte_count": len(inventory_bytes),
            "sha256": _sha256_bytes(inventory_bytes),
            "file_count": len(by_path),
        },
        "timeline_semantics": {
            "fixed_observed_states": True,
            "between_observations": "undefined",
            "prediction": "none",
            "simulation_execution": "consumer_responsibility",
        },
        "visual_timeline": {
            "authoritative": False,
            "manifest_path": f"{PERIMETER_PREVIEW_ROOT}/{PERIMETER_VIEWER_MANIFEST_NAME}",
            "manifest_sha256": by_path[
                f"{PERIMETER_PREVIEW_ROOT}/{PERIMETER_VIEWER_MANIFEST_NAME}"
            ]["sha256"],
            "frame_count": len(visual_frames),
            "frames": visual_frames,
        },
    }
    manifest_bytes = _json_bytes(manifest)
    state_records = [
        {
            "index": frame.get("index"),
            "frame_id": frame.get("frame_id"),
            "observed_at": frame.get("observed_at"),
            "time_range": frame.get("time_range"),
            "affected": frame.get("affected"),
            "active": frame.get("active"),
        }
        for frame in frames
    ]
    contract = {
        "schema": PERIMETER_CONTRACT_SCHEMA,
        "contract_status": "active",
        "layer_package": {
            "package_id": package_id,
            "revision": 1,
            "layer_build_id": layer_build_id,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "entry_layer": PERIMETER_STAGE_NAME,
            "entry_layer_sha256": by_path[PERIMETER_STAGE_NAME]["sha256"],
        },
        "base_map": base_map,
        "progression": {
            "layer_crs": "EPSG:2154",
            "state_count": len(frames),
            "timeline_path": PERIMETER_TIMELINE_NAME,
            "timeline_sha256": by_path[PERIMETER_TIMELINE_NAME]["sha256"],
            "state_records": state_records,
            "between_observations": "undefined",
            "prediction": "none",
            "visual_timeline": manifest["visual_timeline"],
        },
        "release": {
            "layer_attachment_allowed": True,
            "automatic_publication": False,
            "map_replacement": "forbidden",
        },
    }
    _write_atomic(package_root / INVENTORY_NAME, inventory_bytes)
    _write_atomic(package_root / MANIFEST_NAME, manifest_bytes)
    _write_atomic(package_root / PERIMETER_CONTRACT_PATH, _json_bytes(contract))
    validate_perimeter_upload_package(package_root)
    return manifest


def validate_perimeter_upload_package(root: Path | str) -> dict[str, Any]:
    package_root = Path(root).resolve(strict=True)
    manifest_path = package_root / MANIFEST_NAME
    inventory_path = package_root / INVENTORY_NAME
    contract_path = package_root / PERIMETER_CONTRACT_PATH
    manifest = _read_json(manifest_path, "perimeter upload manifest")
    inventory = _read_json(inventory_path, "perimeter dependency inventory")
    contract = _read_json(contract_path, "perimeter upload contract")
    if (
        manifest.get("schema") != PERIMETER_MANIFEST_SCHEMA
        or manifest.get("status") != "active"
    ):
        raise PortableScenePackageError("perimeter upload manifest is incompatible")
    if (
        contract.get("schema") != PERIMETER_CONTRACT_SCHEMA
        or contract.get("contract_status") != "active"
    ):
        raise PortableScenePackageError("perimeter upload contract is incompatible")
    by_path = _inventory_by_path(inventory, "perimeter")
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in _perimeter_payload_paths(package_root)
    }
    if set(by_path) != actual_paths:
        raise PortableScenePackageError(
            "perimeter inventory does not match the package"
        )
    for relative, record in by_path.items():
        path = package_root / relative
        if (
            path.stat().st_size != record["byte_count"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise PortableScenePackageError(f"perimeter dependency changed: {relative}")
    package = contract.get("layer_package")
    base_map = contract.get("base_map")
    progression = contract.get("progression")
    visual = manifest.get("visual_timeline")
    if (
        not isinstance(package, dict)
        or not isinstance(base_map, dict)
        or not isinstance(progression, dict)
        or package.get("manifest_sha256") != _sha256_file(manifest_path)
        or package.get("package_id") != manifest.get("package_id")
        or package.get("entry_layer_sha256")
        != by_path.get(PERIMETER_STAGE_NAME, {}).get("sha256")
        or base_map != manifest.get("base_map")
        or progression.get("timeline_sha256")
        != by_path.get(PERIMETER_TIMELINE_NAME, {}).get("sha256")
        or progression.get("state_count")
        != manifest.get("timeline", {}).get("state_count")
        or progression.get("between_observations") != "undefined"
        or progression.get("prediction") != "none"
        or not isinstance(visual, dict)
        or progression.get("visual_timeline") != visual
        or visual.get("authoritative") is not False
        or visual.get("frame_count") != progression.get("state_count")
    ):
        raise PortableScenePackageError(
            "perimeter upload metadata is not causally bound"
        )
    visual_frames = visual.get("frames")
    if not isinstance(visual_frames, list) or len(visual_frames) != visual.get(
        "frame_count"
    ):
        raise PortableScenePackageError("perimeter visual timeline is incomplete")
    for index, record in enumerate(visual_frames):
        if not isinstance(record, dict) or record.get("index") != index:
            raise PortableScenePackageError("perimeter visual timeline is invalid")
        path = record.get("path")
        inventory_record = by_path.get(path) if isinstance(path, str) else None
        if (
            inventory_record is None
            or inventory_record["sha256"] != record.get("sha256")
            or inventory_record["byte_count"] != record.get("byte_count")
        ):
            raise PortableScenePackageError("perimeter visual timeline changed")
    return manifest


def materialize_perimeter_upload_package(
    raw_package_root: Path | str,
    map_reference: MapPackageReference,
    work_root: Path | str,
    *,
    viewer_root: Path | str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Create one immutable, map-bound perimeter upload without mutating raw layers."""

    source = Path(raw_package_root).resolve(strict=True)
    viewer_source = Path(viewer_root).resolve(strict=True)
    root = Path(work_root).resolve(strict=True)
    if os.name == "nt" and root.drive.upper() != "D:":
        raise PortableScenePackageError("perimeter upload packages must remain on D:")
    source_manifest = _read_json(
        source / PERIMETER_SOURCE_MANIFEST_NAME, "perimeter source manifest"
    )
    dataset_id = str(source_manifest.get("dataset_id", "")).strip()
    layer_build_id = str(source_manifest.get("build_id", "")).strip()
    if not dataset_id or len(layer_build_id) != _SHA256_LENGTH:
        raise PortableScenePackageError("raw perimeter identity is invalid")
    destination = (
        root
        / "perimeter-uploads"
        / dataset_id
        / layer_build_id
        / map_reference.map_build_id
    )
    if destination.exists():
        manifest = validate_perimeter_upload_package(destination)
        if manifest.get("base_map") != map_reference.payload():
            raise PortableScenePackageError(
                "existing perimeter upload targets another map"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}-staging-", dir=destination.parent
            )
        )
        try:
            for name in (
                PERIMETER_STAGE_NAME,
                PERIMETER_TIMELINE_NAME,
                PERIMETER_SOURCE_MANIFEST_NAME,
                "perimeters.normalized.json",
            ):
                source_path = source / name
                if not source_path.is_file():
                    raise PortableScenePackageError(
                        f"raw perimeter package is missing {name}"
                    )
                shutil.copyfile(source_path, staging / name)
            preview = staging / PERIMETER_PREVIEW_ROOT
            preview.mkdir()
            viewer_manifest = _read_json(
                viewer_source / PERIMETER_VIEWER_MANIFEST_NAME,
                "perimeter viewer manifest",
            )
            viewer_files = {
                PERIMETER_VIEWER_MANIFEST_NAME,
                *(
                    str(record.get("path"))
                    for record in viewer_manifest.get("frames", [])
                    if isinstance(record, dict)
                ),
            }
            actual_viewer_files = {
                path.name for path in viewer_source.iterdir() if path.is_file()
            }
            if viewer_files != actual_viewer_files or any(
                path.is_dir() for path in viewer_source.iterdir()
            ):
                raise PortableScenePackageError("perimeter viewer inventory is invalid")
            for name in sorted(viewer_files):
                if PurePosixPath(name).name != name:
                    raise PortableScenePackageError("perimeter viewer path is unsafe")
                shutil.copyfile(viewer_source / name, preview / name)
            manifest = seal_perimeter_upload_package(staging, map_reference)
            os.replace(staging, destination)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    archive = destination.parent / f"{manifest['package_id']}.zip"
    write_deterministic_package_archive(destination, archive)
    return destination, archive, manifest


def write_deterministic_package_archive(
    package_root: Path | str,
    destination: Path | str,
    *,
    prefix: str | None = None,
) -> Path:
    """Write one immutable ZIP after metadata validation."""

    root = Path(package_root).resolve(strict=True)
    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".part", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for path in sorted(
                root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
            ):
                if not path.is_file() or path.resolve() == target:
                    continue
                relative = path.relative_to(root).as_posix()
                archive_path = (
                    f"{prefix.rstrip('/')}/{relative}" if prefix else relative
                )
                info = zipfile.ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=1)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "INVENTORY_NAME",
    "INVENTORY_SCHEMA",
    "MANIFEST_NAME",
    "MAP_CONTRACT_PATH",
    "MAP_CONTRACT_SCHEMA",
    "MAP_MANIFEST_SCHEMA",
    "PERIMETER_CONTRACT_PATH",
    "PERIMETER_CONTRACT_SCHEMA",
    "PERIMETER_MANIFEST_SCHEMA",
    "MapPackageReference",
    "PortableScenePackageError",
    "materialize_perimeter_upload_package",
    "read_map_reference_from_archive",
    "seal_map_upload_package",
    "seal_perimeter_upload_package",
    "validate_map_upload_package",
    "validate_perimeter_upload_package",
    "write_deterministic_package_archive",
]
