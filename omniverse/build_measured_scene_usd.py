"""Assemble measured terrain, buildings and vegetation into portable USDA.

The assembler deliberately performs no detection and no procedural placement.
Every instance is a direct projection of one ``valid`` candidate from the
canonical MNS/MNT placement inventory.  Ambiguous and rejected candidates are
kept in the receipt, never silently dropped.  Prototype choice is delegated to
the immutable 53-asset catalogue API.

The module only writes USDA text and JSON, so it does not require Kit or
``pxr``.  The terrain remains a relative reference supplied by the accepted
tile package.  Every prototype actually selected for this scene is copied into
a small self-contained bundle with its catalogue-locked colour texture.  A
local ``UsdPreviewSurface`` binding marked ``strongerThanDescendants`` replaces
the source material bindings that point outside the referenced default prim.
No unused prototype, fallback primitive, PBR atlas or absolute asset path is
published.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_SHARED_PROTOTYPE_LOCKS_GUARD = threading.Lock()
_SHARED_PROTOTYPE_LOCKS: dict[tuple[str, str], tuple[threading.Lock, int]] = {}
_FILE_HASH_CACHE_LOCK = threading.Lock()
_FILE_HASH_CACHE: OrderedDict[str, tuple[tuple[int, int, int, int, int], str]] = (
    OrderedDict()
)
_FILE_HASH_IN_FLIGHT: dict[
    tuple[str, tuple[int, int, int, int, int]], threading.Event
] = {}
_FILE_HASH_CACHE_LIMIT = 4096
PROTOTYPE_BUNDLE_MODE_ENV = "FIREVIEWER_PROTOTYPE_BUNDLE_MODE"
PROTOTYPE_BUNDLE_MODES = {"copy", "linked"}

CONTRACT_SCHEMA = "fireviewer.measured-scene-usd-contract.v1"
ALGORITHM = "fireviewer.measured-scene-usd-builder.v1"
INVENTORY_SCHEMA = "fireviewer.mns-mnt-placement-inventory.v1"
CATALOG_SCHEMA = "fireviewer.asset-library.v1"
REFERENCE_CATALOG_SCHEMA = "fireviewer.reference-usd-asset-library.v1"
SCENE_SCHEMA = "fireviewer.measured-scene-usd.v1"
RECEIPT_SCHEMA = "fireviewer.measured-scene-receipt.v1"
SCENE_FILE_NAME = "scene.usda"
RECEIPT_FILE_NAME = "scene.done.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
USAGE_MODES = ("technical_pilot_non_final", "final_scene")
FAMILIES = (
    ("buildings", "building"),
    ("trees", "tree"),
    ("context_assets", None),
)
OPTIONAL_INVENTORY_FAMILIES = {"context_assets"}
FAMILY_PRIMS = {
    "buildings": "Buildings",
    "trees": "Trees",
    "context_assets": "ContextAssets",
}


class MeasuredSceneError(ValueError):
    """Inputs cannot be reconciled into a measured portable scene."""


def _prototype_worker_limit() -> int:
    default = max(1, min(8, (os.cpu_count() or 2) - 1))
    raw = os.environ.get("FIREVIEWER_PROTOTYPE_WORKERS", str(default)).strip()
    try:
        requested = int(raw)
    except ValueError as error:
        raise MeasuredSceneError(
            "FIREVIEWER_PROTOTYPE_WORKERS must be an integer"
        ) from error
    if requested < 1 or requested > 32:
        raise MeasuredSceneError(
            "FIREVIEWER_PROTOTYPE_WORKERS must be between 1 and 32"
        )
    return requested


_PROTOTYPE_WORKER_LIMIT = _prototype_worker_limit()
_PROTOTYPE_IO_SLOTS = threading.BoundedSemaphore(_PROTOTYPE_WORKER_LIMIT)


@dataclass(frozen=True)
class TerrainReference:
    """Coordinate binding for an injectable terrain root stage."""

    root_usd: Path
    origin_l93_m: tuple[float, float]
    vertical_origin_mm: int = 0


@dataclass(frozen=True)
class MeasuredScenePackage:
    output_root: Path
    scene: Path
    receipt: Path
    status: str
    building_instance_count: int
    tree_instance_count: int
    context_asset_instance_count: int


@dataclass(frozen=True)
class _Instance:
    family: str
    candidate_id: str
    asset_id: str
    asset_category: str
    selection_seed: int
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    measured_height_m: float
    measured_horizontal_m: tuple[float, float]


@dataclass(frozen=True)
class _Prototype:
    family: str
    asset_id: str
    reference: str
    source_path: Path
    source_relative: str
    source_sha256: str
    source_byte_count: int
    texture_path: Path | None
    texture_relative: str | None
    texture_sha256: str | None
    texture_byte_count: int | None
    wrapper_relative: str
    wrapper_bytes: bytes
    material_policy: str
    source_up_axis: str
    native_min_y: float
    native_extents: tuple[float, float, float]
    qualification_blockers: tuple[str, ...]
    availability: str
    fallback_resolution: Mapping[str, Any] | None


SelectionApi = Callable[..., Mapping[str, Any]]
PrototypeProgressCallback = Callable[[int, int, str], None]


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Return deterministic UTF-8 JSON and reject non-finite values."""

    rendered = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )
    if pretty:
        rendered += "\n"
    return rendered.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _remember_file_hash(path: Path, digest: str) -> None:
    resolved = path.resolve()
    cache_key = str(resolved)
    signature = _file_signature(resolved)
    with _FILE_HASH_CACHE_LOCK:
        _FILE_HASH_CACHE[cache_key] = (signature, digest)
        _FILE_HASH_CACHE.move_to_end(cache_key)
        while len(_FILE_HASH_CACHE) > _FILE_HASH_CACHE_LIMIT:
            _FILE_HASH_CACHE.popitem(last=False)


def remember_validated_file_hash(path: Path | str, digest: str) -> None:
    """Seed the process cache after the immutable image startup validation."""

    if not SHA256_RE.fullmatch(digest):
        raise MeasuredSceneError("validated file SHA-256 is invalid")
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise MeasuredSceneError(f"validated file is missing: {resolved}")
    _remember_file_hash(resolved, digest)


def validated_file_sha256(path: Path | str) -> str:
    """Return a cached SHA-256 while retaining filesystem tamper detection.

    Container startup seeds this cache for the immutable embedded asset
    library. Tile validators can therefore verify catalogue digests without
    rereading the same USD and texture for every tile.
    """

    return _cached_sha256_file(Path(path))


def _cached_sha256_file(path: Path) -> str:
    """Hash one stable file once per process, including concurrent callers."""

    resolved = path.resolve()
    cache_key = str(resolved)
    while True:
        before = _file_signature(resolved)
        flight_key = (cache_key, before)
        with _FILE_HASH_CACHE_LOCK:
            cached = _FILE_HASH_CACHE.get(cache_key)
            if cached is not None and cached[0] == before:
                _FILE_HASH_CACHE.move_to_end(cache_key)
                return cached[1]
            pending = _FILE_HASH_IN_FLIGHT.get(flight_key)
            if pending is None:
                pending = threading.Event()
                _FILE_HASH_IN_FLIGHT[flight_key] = pending
                owns_hash = True
            else:
                owns_hash = False
        if not owns_hash:
            pending.wait()
            continue
        try:
            digest = sha256_file(resolved)
            after = _file_signature(resolved)
            if after != before:
                raise MeasuredSceneError(f"file changed while hashing: {resolved}")
            _remember_file_hash(resolved, digest)
            return digest
        finally:
            with _FILE_HASH_CACHE_LOCK:
                _FILE_HASH_IN_FLIGHT.pop(flight_key, None)
                pending.set()


@contextmanager
def _shared_prototype_lock(bundle_root: Path, asset_id: str) -> Iterator[None]:
    """Serialize one immutable prototype without blocking unrelated assets."""

    key = (str(bundle_root.resolve()), asset_id)
    with _SHARED_PROTOTYPE_LOCKS_GUARD:
        entry = _SHARED_PROTOTYPE_LOCKS.get(key)
        if entry is None:
            lock = threading.Lock()
            users = 0
        else:
            lock, users = entry
        _SHARED_PROTOTYPE_LOCKS[key] = (lock, users + 1)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _SHARED_PROTOTYPE_LOCKS_GUARD:
            current_lock, current_users = _SHARED_PROTOTYPE_LOCKS[key]
            if current_users == 1:
                del _SHARED_PROTOTYPE_LOCKS[key]
            else:
                _SHARED_PROTOTYPE_LOCKS[key] = (
                    current_lock,
                    current_users - 1,
                )


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasuredSceneError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise MeasuredSceneError(f"{label} must be finite and positive")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MeasuredSceneError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise MeasuredSceneError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise MeasuredSceneError(f"{label} must be a non-empty string")
    return value.strip()


def _require_d_path(
    path: Path | str,
    label: str,
    *,
    kind: str | None = None,
) -> Path:
    lexical_drive = PureWindowsPath(str(path)).drive.upper()
    if lexical_drive == "C:":
        raise MeasuredSceneError(f"{label} is forbidden on C:")
    resolved = Path(path).resolve()
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise MeasuredSceneError(f"{label} must be stored on D: {resolved}")
    if kind == "file" and not resolved.is_file():
        raise MeasuredSceneError(f"{label} is not a file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise MeasuredSceneError(f"{label} is not a directory: {resolved}")
    return resolved


def _inside(root: Path, path: Path, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise MeasuredSceneError(f"{label} escapes portable root {root}") from error
    return path


def _load_json(value: Path | str | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = _require_d_path(value, label, kind="file")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MeasuredSceneError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise MeasuredSceneError(f"{label} must contain a JSON object")
    return payload


def _load_contract(path: Path | str | None = None) -> tuple[dict[str, Any], Path]:
    contract_path = _require_d_path(
        path or Path(__file__).with_name("measured_scene_usd_contract.v1.json"),
        "measured scene contract",
        kind="file",
    )
    contract = _load_json(contract_path, "measured scene contract")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("algorithm") != ALGORITHM
    ):
        raise MeasuredSceneError("unsupported measured scene contract")
    if contract.get("crs") != "EPSG:2154":
        raise MeasuredSceneError("measured scene contract must use EPSG:2154")
    catalog_schemas = contract.get("selection", {}).get("catalog_schemas")
    if catalog_schemas != [CATALOG_SCHEMA, REFERENCE_CATALOG_SCHEMA]:
        raise MeasuredSceneError("measured scene contract catalog schemas differ")
    if contract.get("placement", {}).get("quota") != "forbidden":
        raise MeasuredSceneError("measured scene contract must forbid quotas")
    if contract.get("placement", {}).get("thinning") != "forbidden":
        raise MeasuredSceneError("measured scene contract must forbid thinning")
    if contract.get("placement", {}).get("fallback_primitive") != "forbidden":
        raise MeasuredSceneError(
            "measured scene contract must forbid fallback primitives"
        )
    if contract.get("selection", {}).get("fixed_asset_override") != (
        "a validated fixed_asset_id bypasses the API and must resolve to the exact "
        "same-category catalog asset; seed remains deterministic"
    ):
        raise MeasuredSceneError("fixed asset selection override contract is invalid")
    return contract, contract_path


def _portable_catalog_path(value: Any, label: str) -> PurePosixPath:
    text = _require_nonempty_string(value, label)
    if "\\" in text or "@" in text or "\n" in text or "\r" in text:
        raise MeasuredSceneError(f"{label} is not a portable USD asset path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ":" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MeasuredSceneError(f"{label} must be a confined relative path")
    return path


def _relative_reference(target: Path, *, output_root: Path, portable_root: Path) -> str:
    target = _inside(portable_root, target.resolve(), "USD reference")
    relative = os.path.relpath(target, output_root).replace("\\", "/")
    if "@" in relative or "\n" in relative or "\r" in relative or ":" in relative:
        raise MeasuredSceneError("USD reference is not portable")
    resolved = (output_root / Path(*PurePosixPath(relative).parts)).resolve()
    if resolved != target:
        raise MeasuredSceneError("USD reference resolution differs")
    return relative


def _empty_inventory_family() -> dict[str, Any]:
    return {
        "source_count": 0,
        "valid_count": 0,
        "ambiguous_count": 0,
        "rejected_count": 0,
        "placement_ready_count": 0,
        "placement_blocked_count": 0,
        "instantiated_asset_count": 0,
        "candidates": [],
    }


def _inventory_family(inventory: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    record = inventory.get(family)
    if record is None and family in OPTIONAL_INVENTORY_FAMILIES:
        return _empty_inventory_family()
    if not isinstance(record, Mapping):
        raise MeasuredSceneError(f"placement inventory lacks {family}")
    return record


def _validate_inventory(inventory: Mapping[str, Any]) -> None:
    if (
        inventory.get("schema") != INVENTORY_SCHEMA
        or inventory.get("crs") != "EPSG:2154"
    ):
        raise MeasuredSceneError("placement inventory schema or CRS is invalid")
    supplied_hash = _require_sha256(
        inventory.get("inventory_sha256"), "inventory_sha256"
    )
    without_hash = dict(inventory)
    without_hash.pop("inventory_sha256", None)
    if sha256_bytes(canonical_json_bytes(without_hash)) != supplied_hash:
        raise MeasuredSceneError("placement inventory hash mismatch")
    grid = inventory.get("grid")
    if not isinstance(grid, Mapping):
        raise MeasuredSceneError("placement inventory grid is missing")
    bounds = grid.get("core_bounds_l93_m")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise MeasuredSceneError("placement inventory core bounds are invalid")
    finite_bounds = [_finite(value, "core bound") for value in bounds]
    if finite_bounds[2] <= finite_bounds[0] or finite_bounds[3] <= finite_bounds[1]:
        raise MeasuredSceneError("placement inventory core bounds are empty")
    for family, _category in FAMILIES:
        record = _inventory_family(inventory, family)
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise MeasuredSceneError(f"{family} candidates must be an array")
        source_count = _integer(record.get("source_count"), f"{family}.source_count")
        counts = {
            status: _integer(record.get(f"{status}_count"), f"{family}.{status}_count")
            for status in ("valid", "ambiguous", "rejected")
        }
        if source_count != len(candidates) or source_count != sum(counts.values()):
            raise MeasuredSceneError(f"{family} source reconciliation is invalid")
        if record.get("placement_ready_count") != counts["valid"]:
            raise MeasuredSceneError(f"{family} ready reconciliation is invalid")
        if (
            record.get("placement_blocked_count")
            != counts["ambiguous"] + counts["rejected"]
        ):
            raise MeasuredSceneError(f"{family} blocked reconciliation is invalid")
        if record.get("instantiated_asset_count") != 0:
            raise MeasuredSceneError(
                f"{family} inventory must not pre-instantiate assets"
            )
        identifiers: list[str] = []
        observed = {status: 0 for status in counts}
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise MeasuredSceneError(f"{family} candidate {index} is invalid")
            candidate_id = _require_nonempty_string(
                candidate.get("candidate_id"), "candidate_id"
            )
            identifiers.append(candidate_id)
            status = candidate.get("status")
            if status not in observed:
                raise MeasuredSceneError(f"{family} candidate status is invalid")
            observed[status] += 1
            reasons = candidate.get("reason_codes")
            if not isinstance(reasons, list) or any(
                not isinstance(reason, str) for reason in reasons
            ):
                raise MeasuredSceneError(f"{family} candidate reason codes are invalid")
            if status == "valid" and reasons:
                raise MeasuredSceneError(f"valid {family} candidate has block reasons")
            if family == "context_assets" and status == "valid":
                category = candidate.get("asset_category")
                context = candidate.get("selection_context")
                if (
                    not isinstance(category, str)
                    or not category
                    or not isinstance(context, str)
                    or not context
                ):
                    raise MeasuredSceneError(
                        "valid context asset lacks category or selection context"
                    )
                fixed_asset_id = candidate.get("fixed_asset_id")
                if fixed_asset_id is not None and (
                    not isinstance(fixed_asset_id, str)
                    or not fixed_asset_id
                    or context != "fixed_user_coordinate"
                    or not isinstance(candidate.get("fixed_placement_id"), str)
                    or not candidate.get("fixed_placement_id")
                ):
                    raise MeasuredSceneError(
                        "fixed context asset lacks its exact asset or placement identity"
                    )
        if len(set(identifiers)) != len(identifiers) or observed != counts:
            raise MeasuredSceneError(f"{family} candidate reconciliation is corrupt")


def _validate_catalog(library: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    schema = library.get("schema")
    if schema not in {CATALOG_SCHEMA, REFERENCE_CATALOG_SCHEMA}:
        raise MeasuredSceneError("asset library schema is invalid")
    _require_sha256(library.get("catalog_revision"), "catalog_revision")
    assets = library.get("assets")
    if (
        not isinstance(assets, list)
        or not assets
        or library.get("asset_count") != len(assets)
    ):
        raise MeasuredSceneError("asset library asset array/count is invalid")
    if schema == CATALOG_SCHEMA and len(assets) != 53:
        raise MeasuredSceneError("legacy asset library must contain 53 entries")
    if schema == REFERENCE_CATALOG_SCHEMA:
        try:
            module = importlib.import_module("build_reference_usd_asset_library")
        except ModuleNotFoundError:
            blender_root = Path(__file__).resolve().parents[1] / "blender"
            if str(blender_root) not in sys.path:
                sys.path.insert(0, str(blender_root))
            module = importlib.import_module("build_reference_usd_asset_library")
        validator = getattr(module, "validate_reference_asset_library", None)
        if not callable(validator):
            raise MeasuredSceneError("reference asset validator is unavailable")
        try:
            validator(library)
        except Exception as error:
            raise MeasuredSceneError(
                f"reference asset library is invalid: {error}"
            ) from error
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise MeasuredSceneError(f"asset {index} is invalid")
        asset_id = _require_nonempty_string(asset.get("asset_id"), f"asset {index} id")
        if asset_id in indexed:
            raise MeasuredSceneError(f"duplicate asset id: {asset_id}")
        indexed[asset_id] = asset
    pools = library.get("selection_pools")
    if not isinstance(pools, Mapping) or not {"building", "tree"}.issubset(pools):
        raise MeasuredSceneError("asset library selection pools are invalid")
    for category in ("building", "tree"):
        pool = pools[category]
        if not isinstance(pool, list) or not pool or len(pool) != len(set(pool)):
            raise MeasuredSceneError(f"asset library {category} pool is invalid")
        for asset_id in pool:
            if asset_id not in indexed or indexed[asset_id].get("category") != category:
                raise MeasuredSceneError(
                    f"asset library {category} pool membership is invalid"
                )
    return indexed


def _default_selection_api(
    library: Mapping[str, Any],
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Select after ``_validate_catalog`` has validated this scene's catalogue."""

    reference_catalog = library.get("schema") == REFERENCE_CATALOG_SCHEMA
    module_name = (
        "build_reference_usd_asset_library"
        if reference_catalog
        else "build_asset_library_53"
    )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        blender_root = Path(__file__).resolve().parents[1] / "blender"
        if str(blender_root) not in sys.path:
            sys.path.insert(0, str(blender_root))
        module = importlib.import_module(module_name)
    selector_name = (
        "_select_asset_for_candidate_from_validated_library"
        if reference_catalog
        else "select_asset_for_candidate"
    )
    selector = getattr(module, selector_name, None)
    if not callable(selector):
        raise MeasuredSceneError(
            "asset library deterministic selection API is unavailable"
        )
    return selector(library, **kwargs)


def _select(
    selection_api: SelectionApi,
    library: Mapping[str, Any],
    *,
    category: str,
    zone_id: str,
    candidate_id: str,
    rule_version: str,
    usage: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, int]:
    arguments = {
        "category": category,
        "zone": zone_id,
        "candidate": candidate_id,
        "rule_version": rule_version,
        "usage": usage,
    }
    if library.get("schema") == REFERENCE_CATALOG_SCHEMA:
        arguments["metadata"] = dict(metadata or {})
    try:
        result = selection_api(library, **arguments)
    except Exception as error:
        if isinstance(error, MeasuredSceneError):
            raise
        raise MeasuredSceneError(
            f"catalog selection failed for {candidate_id}: {error}"
        ) from error
    if not isinstance(result, Mapping):
        raise MeasuredSceneError("catalog selection API returned no record")
    asset_id = _require_nonempty_string(result.get("asset_id"), "selected asset id")
    if result.get("category") != category:
        raise MeasuredSceneError(
            f"catalog selection category differs for {candidate_id}"
        )
    selection_seed = _integer(result.get("selection_seed"), "selection_seed")
    expected_status = usage
    if result.get("usage_status") != expected_status:
        raise MeasuredSceneError(f"catalog selection usage differs for {candidate_id}")
    return asset_id, selection_seed


def _fixed_selection_seed(
    *, zone_id: str, candidate_id: str, asset_id: str, rule_version: str
) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"fireviewer.fixed-asset-selection.v1\x00{zone_id}\x00{candidate_id}"
            f"\x00{asset_id}\x00{rule_version}".encode()
        ).digest()[:8],
        "big",
    )


def _native_bounds(
    asset: Mapping[str, Any],
) -> tuple[tuple[float, float, float], float]:
    bounds = asset.get("source_bounds")
    if not isinstance(bounds, Mapping):
        raise MeasuredSceneError(f"asset {asset.get('asset_id')} has no native bounds")
    minimum = bounds.get("minimum")
    maximum = bounds.get("maximum")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        raise MeasuredSceneError(
            f"asset {asset.get('asset_id')} native bounds are invalid"
        )
    low = tuple(_finite(value, "native bound minimum") for value in minimum)
    high = tuple(_finite(value, "native bound maximum") for value in maximum)
    extents = tuple(high[index] - low[index] for index in range(3))
    if any(extent <= 0.0 for extent in extents):
        raise MeasuredSceneError(
            f"asset {asset.get('asset_id')} native bounds are empty"
        )
    return extents, low[1]


def _qualification_blockers(asset: Mapping[str, Any]) -> tuple[str, ...]:
    qualification = asset.get("qualification")
    if not isinstance(qualification, Mapping):
        return ("dimensions_missing", "pivot_missing")
    dimensions = qualification.get("dimensions")
    ground_anchor = qualification.get("ground_anchor")
    blockers: list[str] = []
    if asset.get("availability") == "placeholder_usd":
        blockers.append("placeholder_usd")
    if (
        not isinstance(dimensions, Mapping)
        or dimensions.get("status") != "accepted"
        or dimensions.get("value_m") is None
    ):
        blockers.append("dimensions_missing_or_not_accepted")
    if (
        not isinstance(ground_anchor, Mapping)
        or ground_anchor.get("status") != "accepted"
        or ground_anchor.get("offset_m") is None
    ):
        blockers.append("pivot_missing_or_not_accepted")
    return tuple(blockers)


def _catalog_artifact_target(
    asset: Mapping[str, Any],
    *,
    role: str,
    asset_roots: Mapping[str, Path],
    portable_root: Path,
) -> tuple[Path, PurePosixPath, str, int]:
    artifact = asset.get(role)
    if not isinstance(artifact, Mapping):
        raise MeasuredSceneError(
            f"asset {asset.get('asset_id')} has no {role} artifact"
        )
    logical_root = _require_nonempty_string(
        artifact.get("root"), f"{role} artifact root"
    )
    if logical_root not in asset_roots:
        raise MeasuredSceneError(
            f"missing physical mapping for asset root {logical_root}"
        )
    relative = _portable_catalog_path(artifact.get("path"), f"{role} artifact path")
    physical_root = _inside(portable_root, asset_roots[logical_root], "asset root")
    target = physical_root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(physical_root)
    except ValueError as error:
        raise MeasuredSceneError(f"{role} artifact escapes its logical root") from error
    if not target.is_file():
        raise MeasuredSceneError(
            f"{role} prototype artifact is missing: {relative.as_posix()}"
        )
    expected_hash = _require_sha256(artifact.get("sha256"), f"{role} artifact SHA-256")
    expected_bytes = _integer(
        artifact.get("byte_count"), f"{role} artifact byte_count", minimum=1
    )
    if target.stat().st_size != expected_bytes:
        raise MeasuredSceneError(
            f"{role} prototype byte count differs: {relative.as_posix()}"
        )
    actual_hash = _cached_sha256_file(target)
    if actual_hash != expected_hash:
        raise MeasuredSceneError(
            f"{role} prototype hash differs: {relative.as_posix()}"
        )
    return target, relative, actual_hash, expected_bytes


def _bundle_asset_id(asset_id: str) -> str:
    if BUNDLE_ID_RE.fullmatch(asset_id) is None or asset_id in {".", ".."}:
        raise MeasuredSceneError(f"asset id is not bundle-safe: {asset_id!r}")
    return asset_id


def _render_prototype_wrapper(
    *,
    asset_id: str,
    source_file_name: str,
    texture_reference: str,
    source_up_axis: str = "Y",
) -> bytes:
    """Author the local colour material that wins over broken source bindings."""

    texture_asset = texture_reference.replace("\\", "/")
    if (
        "@" in texture_asset
        or ":" in texture_asset
        or texture_asset.startswith("/")
        or ".." in PurePosixPath(texture_asset).parts
    ):
        raise MeasuredSceneError("prototype texture reference is not portable")
    source_asset = source_file_name.replace("\\", "/")
    if (
        "@" in source_asset
        or ":" in source_asset
        or "/" in source_asset
        or source_asset in {"", ".", ".."}
    ):
        raise MeasuredSceneError("prototype source reference is not portable")
    if source_up_axis not in {"Y", "Z"}:
        raise MeasuredSceneError("prototype source up axis is unsupported")
    source_reference = (
        f"    prepend references = @{source_asset}@\n" if source_up_axis == "Y" else ""
    )
    source_child = (
        ""
        if source_up_axis == "Y"
        else f"""\n    def Xform "Source" (
        prepend references = @{source_asset}@
    )
    {{
        double xformOp:rotateX = -90
        uniform token[] xformOpOrder = ["xformOp:rotateX"]
    }}\n"""
    )
    text = f"""#usda 1.0
(
    defaultPrim = "Prototype"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Prototype" (
    prepend apiSchemas = ["MaterialBindingAPI"]
{source_reference.rstrip()}
)
{{
    custom string fireviewer:asset_id = {_quoted(asset_id)}
    custom string fireviewer:material_policy = "local_color_stronger_than_descendants"
    rel material:binding = </Prototype/FireViewerLooks/ColorMaterial> (
        bindMaterialAs = "strongerThanDescendants"
    )
{source_child}

    def Scope "FireViewerLooks"
    {{
        def Material "ColorMaterial"
        {{
            token outputs:surface.connect = </Prototype/FireViewerLooks/ColorMaterial/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Prototype/FireViewerLooks/ColorMaterial/ColorTexture.outputs:rgb>
                float inputs:metallic = 0
                float inputs:opacity.connect = </Prototype/FireViewerLooks/ColorMaterial/ColorTexture.outputs:a>
                float inputs:opacityThreshold = 0.05
                float inputs:roughness = 0.75
                token outputs:surface
            }}

            def Shader "ColorTexture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{texture_asset}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = </Prototype/FireViewerLooks/ColorMaterial/Texcoord.outputs:result>
                float outputs:a
                float3 outputs:rgb
            }}

            def Shader "Texcoord"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
}}
"""
    return text.encode("utf-8")


def _render_source_package_wrapper(
    *, asset_id: str, source_file_name: str, source_up_axis: str = "Y"
) -> bytes:
    """Reference a self-contained USDZ whose bindings stay below defaultPrim."""

    source_asset = source_file_name.replace("\\", "/")
    if (
        "@" in source_asset
        or ":" in source_asset
        or "/" in source_asset
        or source_asset in {"", ".", ".."}
        or PurePosixPath(source_asset).suffix.casefold() != ".usdz"
    ):
        raise MeasuredSceneError("prototype source package reference is not portable")
    if source_up_axis not in {"Y", "Z"}:
        raise MeasuredSceneError("prototype source package up axis is unsupported")
    source_reference = (
        f"    prepend references = @{source_asset}@\n" if source_up_axis == "Y" else ""
    )
    source_child = (
        ""
        if source_up_axis == "Y"
        else f"""\n    def Xform "Source" (
        prepend references = @{source_asset}@
    )
    {{
        double xformOp:rotateX = -90
        uniform token[] xformOpOrder = ["xformOp:rotateX"]
    }}\n"""
    )
    text = f"""#usda 1.0
(
    defaultPrim = "Prototype"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Prototype" (
{source_reference.rstrip()}
)
{{
    custom string fireviewer:asset_id = {_quoted(asset_id)}
    custom string fireviewer:material_policy = "source_package_pbr"
{source_child}
}}
"""
    return text.encode("utf-8")


def _render_scoped_source_wrapper(
    *, asset_id: str, source_file_name: str, source_up_axis: str
) -> bytes:
    """Reference a normalized USD whose material bindings live below defaultPrim."""

    source_asset = source_file_name.replace("\\", "/")
    if (
        "@" in source_asset
        or ":" in source_asset
        or "/" in source_asset
        or source_asset in {"", ".", ".."}
        or PurePosixPath(source_asset).suffix.casefold()
        not in {".usd", ".usda", ".usdc"}
    ):
        raise MeasuredSceneError("normalized prototype reference is not portable")
    if source_up_axis not in {"Y", "Z"}:
        raise MeasuredSceneError("normalized prototype up axis is unsupported")
    source_reference = (
        f"    prepend references = @{source_asset}@\n" if source_up_axis == "Y" else ""
    )
    source_child = (
        ""
        if source_up_axis == "Y"
        else f"""\n    def Xform "Source" (
        prepend references = @{source_asset}@
    )
    {{
        double xformOp:rotateX = -90
        uniform token[] xformOpOrder = ["xformOp:rotateX"]
    }}\n"""
    )
    return f"""#usda 1.0
(
    defaultPrim = "Prototype"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Prototype" (
{source_reference.rstrip()}
)
{{
    custom string fireviewer:asset_id = {_quoted(asset_id)}
    custom string fireviewer:material_policy = "scoped_source_pbr"
{source_child}
}}
""".encode()


def _plan_prototype_bundle(
    asset: Mapping[str, Any],
    *,
    family: str,
    asset_roots: Mapping[str, Path],
    portable_root: Path,
    bundle_root: Path,
    output_root: Path,
    native_min_y: float,
    native_extents: tuple[float, float, float],
    qualification_blockers: tuple[str, ...],
) -> _Prototype:
    asset_id = _bundle_asset_id(
        _require_nonempty_string(asset.get("asset_id"), "asset id")
    )
    source, _source_catalog_path, source_hash, source_bytes = _catalog_artifact_target(
        asset,
        role="usd",
        asset_roots=asset_roots,
        portable_root=portable_root,
    )
    texture, texture_catalog_path, texture_hash, texture_bytes = (
        _catalog_artifact_target(
            asset,
            role="texture",
            asset_roots=asset_roots,
            portable_root=portable_root,
        )
    )
    material = asset.get("material")
    material_policy = (
        material.get("policy")
        if isinstance(material, Mapping)
        else "fireviewer_color_override"
    )
    stage = asset.get("usd_stage")
    source_up_axis = stage.get("up_axis") if isinstance(stage, Mapping) else "Y"
    source_up_axis = "Y" if source_up_axis is None else source_up_axis
    if source_up_axis not in {"Y", "Z"}:
        raise MeasuredSceneError(f"prototype source up axis is invalid: {asset_id}")
    if material_policy not in {
        "fireviewer_color_override",
        "source_package_color_override",
        "source_package_pbr",
        "scoped_source_pbr",
    }:
        raise MeasuredSceneError(f"prototype material policy is invalid: {asset_id}")
    if source.suffix.casefold() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise MeasuredSceneError(f"prototype source is not USD: {asset_id}")
    if (
        material_policy.startswith("source_package")
        and source.suffix.casefold() != ".usdz"
    ):
        raise MeasuredSceneError(f"prototype source package is not USDZ: {asset_id}")
    if texture.suffix.casefold() not in {".png", ".jpg", ".jpeg"}:
        raise MeasuredSceneError(f"prototype colour texture is not PNG: {asset_id}")
    asset_relative = PurePosixPath(asset_id)
    source_relative = (asset_relative / f"source{source.suffix.casefold()}").as_posix()
    texture_name = texture_catalog_path.name
    if not texture_name or texture_name in {".", ".."} or "@" in texture_name:
        raise MeasuredSceneError(f"prototype texture name is invalid: {asset_id}")
    texture_relative = (asset_relative / "textures" / texture_name).as_posix()
    wrapper_relative = (asset_relative / "prototype.usda").as_posix()
    if material_policy == "source_package_pbr":
        wrapper_bytes = _render_source_package_wrapper(
            asset_id=asset_id,
            source_file_name=PurePosixPath(source_relative).name,
            source_up_axis=source_up_axis,
        )
        packaged_texture_path: Path | None = None
        packaged_texture_relative: str | None = None
        packaged_texture_hash: str | None = None
        packaged_texture_bytes: int | None = None
    elif material_policy == "scoped_source_pbr":
        wrapper_bytes = _render_scoped_source_wrapper(
            asset_id=asset_id,
            source_file_name=PurePosixPath(source_relative).name,
            source_up_axis=source_up_axis,
        )
        packaged_texture_path = texture
        packaged_texture_relative = texture_relative
        packaged_texture_hash = texture_hash
        packaged_texture_bytes = texture_bytes
    else:
        wrapper_bytes = _render_prototype_wrapper(
            asset_id=asset_id,
            source_file_name=PurePosixPath(source_relative).name,
            texture_reference=(PurePosixPath("textures") / texture_name).as_posix(),
            source_up_axis=source_up_axis,
        )
        packaged_texture_path = texture
        packaged_texture_relative = texture_relative
        packaged_texture_hash = texture_hash
        packaged_texture_bytes = texture_bytes
    availability = asset.get("availability", "real_usd")
    if availability not in {"real_usd", "placeholder_usd"}:
        raise MeasuredSceneError(f"prototype availability is invalid: {asset_id}")
    return _Prototype(
        family=family,
        asset_id=asset_id,
        reference=_relative_reference(
            bundle_root.joinpath(*PurePosixPath(wrapper_relative).parts),
            output_root=output_root,
            portable_root=portable_root,
        ),
        source_path=source,
        source_relative=source_relative,
        source_sha256=source_hash,
        source_byte_count=source_bytes,
        texture_path=packaged_texture_path,
        texture_relative=packaged_texture_relative,
        texture_sha256=packaged_texture_hash,
        texture_byte_count=packaged_texture_bytes,
        wrapper_relative=wrapper_relative,
        wrapper_bytes=wrapper_bytes,
        material_policy=material_policy,
        source_up_axis=source_up_axis,
        native_min_y=native_min_y,
        native_extents=native_extents,
        qualification_blockers=qualification_blockers,
        availability=availability,
        fallback_resolution=(
            dict(asset["fallback_resolution"])
            if isinstance(asset.get("fallback_resolution"), Mapping)
            else None
        ),
    )


def _prototype_material_receipt(material_policy: str) -> dict[str, str]:
    if material_policy == "source_package_pbr":
        return {
            "implementation": "source_package_pbr",
            "binding_strength": "authored_below_default_prim",
            "texture_role": "embedded_usdz",
            "source_color_space": "authored",
        }
    if material_policy == "scoped_source_pbr":
        return {
            "implementation": "scoped_source_pbr",
            "binding_strength": "authored_below_default_prim",
            "texture_role": "source_usd_dependency",
            "source_color_space": "authored",
        }
    return {
        "implementation": "UsdPreviewSurface",
        "binding_strength": "strongerThanDescendants",
        "texture_role": "color",
        "source_color_space": "sRGB",
    }


def _geometry_points(geometry: Any) -> list[tuple[float, float]]:
    if not isinstance(geometry, Mapping):
        raise MeasuredSceneError("building footprint geometry is invalid")
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    rings: Iterable[Any]
    if geometry_type == "Polygon" and isinstance(coordinates, list):
        rings = coordinates[:1]
    elif geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        rings = [
            polygon[0]
            for polygon in coordinates
            if isinstance(polygon, list) and polygon
        ]
    else:
        raise MeasuredSceneError("building footprint must be Polygon or MultiPolygon")
    points: list[tuple[float, float]] = []
    for ring in rings:
        if not isinstance(ring, list):
            raise MeasuredSceneError("building footprint ring is invalid")
        for coordinate in ring:
            if not isinstance(coordinate, list) or len(coordinate) < 2:
                raise MeasuredSceneError("building footprint coordinate is invalid")
            points.append(
                (
                    _finite(coordinate[0], "building footprint x"),
                    _finite(coordinate[1], "building footprint y"),
                )
            )
    unique = sorted(set(points))
    if len(unique) < 3:
        raise MeasuredSceneError("building footprint has fewer than three points")
    return unique


def _cross(
    origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
        b[0] - origin[0]
    )


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _footprint_box(geometry: Any) -> tuple[float, float, float]:
    """Return deterministic major width, minor depth and yaw from inventory geometry."""

    hull = _convex_hull(_geometry_points(geometry))
    if len(hull) < 3:
        raise MeasuredSceneError("building footprint has an empty convex hull")
    candidates: list[tuple[float, float, float, float]] = []
    for index, point in enumerate(hull):
        following = hull[(index + 1) % len(hull)]
        dx = following[0] - point[0]
        dy = following[1] - point[1]
        if math.isclose(dx, 0.0, abs_tol=1e-12) and math.isclose(
            dy, 0.0, abs_tol=1e-12
        ):
            continue
        angle = math.atan2(dy, dx)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        along = [x * cosine + y * sine for x, y in hull]
        across = [-x * sine + y * cosine for x, y in hull]
        width = max(along) - min(along)
        depth = max(across) - min(across)
        if depth > width:
            width, depth = depth, width
            angle += math.pi / 2.0
        angle = (angle + math.pi / 2.0) % math.pi - math.pi / 2.0
        candidates.append((width * depth, abs(angle), angle, width, depth))
    if not candidates:
        raise MeasuredSceneError("building footprint has no usable edge")
    _area, _abs_angle, yaw, width, depth = min(
        candidates,
        key=lambda value: tuple(round(component, 9) for component in value),
    )
    if width <= 0.0 or depth <= 0.0:
        raise MeasuredSceneError("building footprint dimensions are empty")
    return width, depth, yaw


def _building_measurement(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    """Read footprint dimensions already present in the placement inventory.

    SIG footprints remain the canonical path.  A MNS component detector may
    instead publish explicit measured width/depth/orientation fields.  Bounds
    are accepted only when they are themselves stored in the inventory; the
    assembler never derives a footprint from a quota, an asset, or a default.
    """

    geometry = candidate.get("footprint_geojson")
    if geometry is not None:
        return _footprint_box(geometry)
    width = candidate.get("component_footprint_width_m")
    depth = candidate.get("component_footprint_depth_m")
    if width is not None or depth is not None:
        width_m = _finite(width, "component footprint width", positive=True)
        depth_m = _finite(depth, "component footprint depth", positive=True)
        yaw = _finite(
            candidate.get("component_orientation_rad", 0.0),
            "component orientation",
        )
        if depth_m > width_m:
            width_m, depth_m = depth_m, width_m
            yaw += math.pi / 2.0
        yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
        return width_m, depth_m, yaw
    bounds = candidate.get("component_bounds_l93_m")
    if bounds is not None:
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise MeasuredSceneError(
                "component bounds must contain west, south, east, north"
            )
        west, south, east, north = (
            _finite(value, "component footprint bound") for value in bounds
        )
        width_m = east - west
        depth_m = north - south
        if width_m <= 0.0 or depth_m <= 0.0:
            raise MeasuredSceneError("component footprint bounds are empty")
        yaw = _finite(
            candidate.get("component_orientation_rad", 0.0),
            "component orientation",
        )
        if depth_m > width_m:
            width_m, depth_m = depth_m, width_m
            yaw += math.pi / 2.0
        yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
        return width_m, depth_m, yaw
    raise MeasuredSceneError(
        "building candidate has neither inventory footprint nor measured component dimensions"
    )


def _axis_orientation(yaw: float) -> tuple[float, float, float, float]:
    """Compose footprint yaw with Y-up to Z-up (+90 degrees around X)."""

    root_half = math.sqrt(0.5)
    cosine = math.cos(yaw / 2.0)
    sine = math.sin(yaw / 2.0)
    return (
        cosine * root_half,
        cosine * root_half,
        sine * root_half,
        sine * root_half,
    )


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise MeasuredSceneError(f"{label} must contain two coordinates")
    return _finite(value[0], f"{label}.x"), _finite(value[1], f"{label}.y")


def _metadata_terms(value: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        str(item)
        for item in value.values()
        if item is not None and isinstance(item, (str, int, float, bool))
    )
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    terms: list[str] = []
    current: list[str] = []
    for character in ascii_text:
        if character.isalnum():
            current.append(character)
        elif current:
            terms.append("".join(current))
            current.clear()
    if current:
        terms.append("".join(current))
    return {term for term in terms if len(term) >= 3 and not term.isdigit()}


def _semantic_tags(terms: set[str]) -> list[str]:
    joined = "_".join(sorted(terms))
    rules = {
        "agricultural": (
            "agricol",
            "bergerie",
            "chai",
            "etable",
            "ferme",
            "grange",
            "hangar",
            "silo",
            "viticole",
        ),
        "commercial": (
            "commerce",
            "commercial",
            "magasin",
            "superette",
            "supermarche",
        ),
        "industrial": ("activite", "atelier", "depot", "industrie", "usine"),
        "residential": ("habitation", "logement", "residentiel"),
        "public_service": ("administratif", "enseignement", "mairie", "public"),
        "religious": ("chapelle", "eglise", "religieux"),
        "degraded": ("abandon", "degrade", "ruine", "vacant"),
        "conifer": ("conifer", "douglas", "epicea", "meleze", "pin", "sapin"),
        "broadleaf": (
            "bouleau",
            "charme",
            "chataignier",
            "chene",
            "erable",
            "feuill",
            "frene",
            "hetre",
            "merisier",
            "noyer",
            "platane",
            "robinier",
            "tilleul",
        ),
        "oak": ("chene",),
        "road_safety": ("autoroute", "route", "routier", "voie"),
        "rail_signal": ("ferree", "rail", "train"),
        "hydro_bank": ("eau", "hydro", "riviere", "ruisseau"),
    }
    tags = {
        tag
        for tag, fragments in rules.items()
        if any(fragment in joined for fragment in fragments)
    }
    if any(term in terms for term in ("mixte", "melange", "mixed")):
        tags.update(("broadleaf", "conifer"))
    return sorted(tags)


def _candidate_category(
    family: str,
    candidate: Mapping[str, Any],
    library: Mapping[str, Any],
) -> str:
    if family == "buildings":
        return "building"
    if family == "trees":
        # These candidates come from the measured crown detector. A short crown
        # can be a sapling or a locally underestimated canopy; it is not evidence
        # for replacing the tree by grass, scrub, or a natural prop.
        return "tree"
    if family == "context_assets":
        return _require_nonempty_string(
            candidate.get("asset_category"), "context asset category"
        )
    raise RuntimeError(f"unknown placement family: {family}")


def _candidate_selection_metadata(
    family: str,
    candidate: Mapping[str, Any],
    category: str,
) -> dict[str, Any]:
    properties = candidate.get("source_properties", {})
    properties = properties if isinstance(properties, Mapping) else {}
    terms = _metadata_terms(properties)
    semantic_tags = _semantic_tags(terms)
    if family == "buildings":
        context = "building"
    elif family == "trees":
        context = (
            "measured_low_vegetation"
            if category == "vegetation"
            else "measured_woody_canopy"
        )
        classification = candidate.get("context_classification")
        if isinstance(classification, str):
            terms.update(_metadata_terms({"classification": classification}))
        semantic_tags = [
            tag
            for tag in _semantic_tags(terms)
            if tag in {"broadleaf", "conifer", "oak"}
        ]
    else:
        context = _require_nonempty_string(
            candidate.get("selection_context"), "context selection context"
        )
    metadata = {
        "context": context,
        "semantic_tags": semantic_tags,
        "reference_terms": sorted(terms),
    }
    if (
        family == "trees"
        and candidate.get("asset_selection_policy")
        == "current_bdtopo_composition_else_bdforet_v1_then_conifer_or_oak_only"
    ):
        metadata["tree_form_policy"] = "conifer_or_oak_only"
    return metadata


def _instance_from_candidate(
    candidate: Mapping[str, Any],
    *,
    family: str,
    asset_id: str,
    asset_category: str,
    selection_seed: int,
    prototype: _Prototype,
    terrain: TerrainReference,
) -> _Instance:
    ground_value = candidate.get(
        "support_elevation_mm", candidate.get("ground_elevation_mm")
    )
    ground_m = (
        _integer(
            ground_value,
            "candidate support/ground elevation mm",
            minimum=-2_147_483_648,
        )
        / 1000.0
    )
    if family == "buildings":
        height_m = (
            _integer(candidate.get("height_cm"), "candidate height_cm", minimum=1)
            / 100.0
        )
        world_x, world_y = _point(
            candidate.get("anchor_l93_m"), "building anchor_l93_m"
        )
        width, depth, yaw = _building_measurement(candidate)
    elif family == "trees":
        height_m = (
            _integer(candidate.get("height_cm"), "candidate height_cm", minimum=1)
            / 100.0
        )
        world_x, world_y = _point(
            candidate.get("position_l93_m"), "tree position_l93_m"
        )
        radius = _finite(
            candidate.get("equivalent_crown_radius_m"),
            "tree crown radius",
            positive=True,
        )
        width = depth = radius * 2.0
        yaw = ((selection_seed & ((1 << 64) - 1)) / float(1 << 64)) * (2.0 * math.pi)
    elif family == "context_assets":
        world_x, world_y = _point(
            candidate.get("position_l93_m"), "context asset position_l93_m"
        )
        width, height_m, depth = prototype.native_extents
        yaw = _finite(candidate.get("yaw_rad", 0.0), "context asset yaw")
    else:  # pragma: no cover - internal invariant
        raise RuntimeError(f"unknown placement family: {family}")
    extents = prototype.native_extents
    if family == "trees":
        if (
            candidate.get("geometry_scale_policy")
            == "measured_crown_diameter_x_measured_hag_height"
        ):
            scale = (width / extents[0], height_m / extents[1], depth / extents[2])
        else:
            uniform_scale = height_m / extents[1]
            scale = (uniform_scale, uniform_scale, uniform_scale)
    elif family == "buildings":
        scale = (width / extents[0], height_m / extents[1], depth / extents[2])
    else:
        scale = (1.0, 1.0, 1.0)
    if any(not math.isfinite(value) or value <= 0.0 for value in scale):
        raise MeasuredSceneError(
            f"candidate {candidate.get('candidate_id')} scale is invalid"
        )
    return _Instance(
        family=family,
        candidate_id=str(candidate["candidate_id"]),
        asset_id=asset_id,
        asset_category=asset_category,
        selection_seed=selection_seed,
        position=(
            world_x - terrain.origin_l93_m[0],
            world_y - terrain.origin_l93_m[1],
            ground_m,
        ),
        scale=scale,
        orientation=_axis_orientation(yaw),
        measured_height_m=height_m,
        measured_horizontal_m=(width, depth),
    )


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise MeasuredSceneError("USD value must be finite")
    if value == 0.0:
        return "0"
    result = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if result == "-0" else result


def _tuple(values: Iterable[float]) -> str:
    return "(" + ", ".join(_number(value) for value in values) + ")"


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return "Asset_" + cleaned


def _stable_instance_id(family: str, candidate_id: str) -> int:
    raw = hashlib.sha256(f"{ALGORITHM}\x00{family}\x00{candidate_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") & ((1 << 63) - 1)


def _array(
    type_name: str, name: str, values: Sequence[str], indent: str = "        "
) -> str:
    if not values:
        return f"{indent}{type_name} {name} = []\n"
    inner = "\n".join(f"{indent}    {value}," for value in values)
    return f"{indent}{type_name} {name} = [\n{inner}\n{indent}]\n"


def _render_scene(
    *,
    terrain_reference: str,
    terrain: TerrainReference,
    zone_id: str,
    inventory_build_id: str,
    catalog_revision: str,
    contract_sha256: str,
    prototypes: Mapping[tuple[str, str], _Prototype],
    instances: Mapping[str, Sequence[_Instance]],
) -> bytes:
    lines: list[str] = [
        "#usda 1.0\n",
        "(\n",
        '    defaultPrim = "MeasuredScene"\n',
        "    metersPerUnit = 1\n",
        '    upAxis = "Z"\n',
        ")\n\n",
        'def Xform "MeasuredScene"\n',
        "{\n",
        f"    custom string fireviewer:schema = {_quoted(SCENE_SCHEMA)}\n",
        f"    custom string fireviewer:algorithm = {_quoted(ALGORITHM)}\n",
        f"    custom string fireviewer:zone_id = {_quoted(zone_id)}\n",
        f"    custom string fireviewer:inventory_build_id = {_quoted(inventory_build_id)}\n",
        f"    custom string fireviewer:catalog_revision = {_quoted(catalog_revision)}\n",
        f"    custom string fireviewer:contract_sha256 = {_quoted(contract_sha256)}\n",
        '    custom string fireviewer:placement_source = "MNT_MNS_inventory_only"\n',
        "    custom bool fireviewer:quota_applied = false\n",
        "    custom bool fireviewer:thinning_applied = false\n",
        f"    custom double2 fireviewer:origin_l93_m = {_tuple(terrain.origin_l93_m)}\n",
        f"    custom int fireviewer:vertical_origin_mm = {terrain.vertical_origin_mm}\n\n",
        '    def Xform "Terrain" (\n',
        f"        prepend references = @{terrain_reference}@\n",
        "    )\n",
        "    {\n",
        '        custom string fireviewer:category = "terrain"\n',
        "        custom int fireviewer:count = 1\n",
        "    }\n\n",
        '    def Scope "Prototypes"\n',
        "    {\n",
    ]
    for family, scope_name in FAMILY_PRIMS.items():
        lines.extend([f'        def Scope "{scope_name}"\n', "        {\n"])
        for (_prototype_family, asset_id), prototype in sorted(prototypes.items()):
            if _prototype_family != family:
                continue
            lines.extend(
                [
                    f'            def Xform "{_identifier(asset_id)}"\n',
                    "            {\n",
                    '                def Xform "Source" (\n',
                    f"                    prepend references = @{prototype.reference}@\n",
                    "                )\n",
                    "                {\n",
                    f"                    double3 xformOp:translate = {_tuple((0.0, -prototype.native_min_y, 0.0))}\n",
                    '                    uniform token[] xformOpOrder = ["xformOp:translate"]\n',
                    "                }\n",
                    "            }\n",
                ]
            )
        lines.append("        }\n")
    lines.extend(["    }\n\n"])

    for family, prim_name in FAMILY_PRIMS.items():
        family_instances = tuple(instances[family])
        prototype_ids = sorted({instance.asset_id for instance in family_instances})
        prototype_index = {
            asset_id: index for index, asset_id in enumerate(prototype_ids)
        }
        ids = [
            _stable_instance_id(family, instance.candidate_id)
            for instance in family_instances
        ]
        if len(ids) != len(set(ids)):
            raise MeasuredSceneError(f"{family} deterministic USD ids collided")
        lines.extend(
            [
                f'    def PointInstancer "{prim_name}"\n',
                "    {\n",
                f'        custom string fireviewer:category = "{("context_asset" if family == "context_assets" else family[:-1])}"\n',
                f"        custom int fireviewer:count = {len(family_instances)}\n",
                f'        custom string fireviewer:family = "{family}"\n',
                f"        custom int fireviewer:source_instance_count = {len(family_instances)}\n",
                "        custom bool fireviewer:quota_applied = false\n",
                "        custom bool fireviewer:thinning_applied = false\n",
                _array(
                    "rel",
                    "prototypes",
                    [
                        f"</MeasuredScene/Prototypes/{prim_name}/{_identifier(asset_id)}>"
                        for asset_id in prototype_ids
                    ],
                ),
                _array("int64[]", "ids", [str(value) for value in ids]),
                _array(
                    "point3f[]",
                    "positions",
                    [_tuple(instance.position) for instance in family_instances],
                ),
                _array(
                    "float3[]",
                    "scales",
                    [_tuple(instance.scale) for instance in family_instances],
                ),
                _array(
                    "quath[]",
                    "orientations",
                    [_tuple(instance.orientation) for instance in family_instances],
                ),
                _array(
                    "int[]",
                    "protoIndices",
                    [
                        str(prototype_index[instance.asset_id])
                        for instance in family_instances
                    ],
                ),
                "    }\n\n",
            ]
        )
    lines.append("}\n")
    return "".join(lines).encode("utf-8")


def _blocked_records(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": str(candidate["candidate_id"]),
            "reason_codes": sorted(str(reason) for reason in candidate["reason_codes"]),
            "status": str(candidate["status"]),
        }
        for candidate in sorted(
            candidates, key=lambda value: str(value["candidate_id"])
        )
        if candidate["status"] != "valid"
    ]


def _prototype_payloads(prototype: _Prototype) -> dict[str, bytes]:
    source_bytes = prototype.source_path.read_bytes()
    if (
        len(source_bytes) != prototype.source_byte_count
        or sha256_bytes(source_bytes) != prototype.source_sha256
    ):
        raise MeasuredSceneError(
            f"USD prototype changed while packaging: {prototype.asset_id}"
        )
    payloads = {
        prototype.source_relative: source_bytes,
        prototype.wrapper_relative: prototype.wrapper_bytes,
    }
    if prototype.texture_path is not None:
        texture_bytes = prototype.texture_path.read_bytes()
        if (
            prototype.texture_relative is None
            or prototype.texture_byte_count is None
            or prototype.texture_sha256 is None
            or len(texture_bytes) != prototype.texture_byte_count
            or sha256_bytes(texture_bytes) != prototype.texture_sha256
        ):
            raise MeasuredSceneError(
                f"texture changed while packaging: {prototype.asset_id}"
            )
        payloads[prototype.texture_relative] = texture_bytes
    return payloads


def _prototype_bundle_mode() -> str:
    mode = os.environ.get(PROTOTYPE_BUNDLE_MODE_ENV, "copy").strip().lower()
    if mode not in PROTOTYPE_BUNDLE_MODES:
        raise MeasuredSceneError(
            f"{PROTOTYPE_BUNDLE_MODE_ENV} must be one of "
            + ", ".join(sorted(PROTOTYPE_BUNDLE_MODES))
        )
    if mode == "linked" and os.name == "nt":
        raise MeasuredSceneError(
            "linked prototype bundles are only supported in the Linux worker"
        )
    return mode


def _linked_prototype_sources(prototype: _Prototype) -> dict[str, Path]:
    sources = {prototype.source_relative: prototype.source_path}
    if prototype.texture_path is not None:
        if prototype.texture_relative is None:
            raise MeasuredSceneError(
                f"prototype texture receipt is incomplete: {prototype.asset_id}"
            )
        sources[prototype.texture_relative] = prototype.texture_path
    return sources


def _prototype_artifact_hashes(
    prototype: _Prototype,
) -> dict[str, tuple[int, str]]:
    artifacts = {
        prototype.source_relative: (
            prototype.source_byte_count,
            prototype.source_sha256,
        ),
        prototype.wrapper_relative: (
            len(prototype.wrapper_bytes),
            sha256_bytes(prototype.wrapper_bytes),
        ),
    }
    if prototype.texture_relative is not None:
        if prototype.texture_byte_count is None or prototype.texture_sha256 is None:
            raise MeasuredSceneError(
                f"prototype texture receipt is incomplete: {prototype.asset_id}"
            )
        artifacts[prototype.texture_relative] = (
            prototype.texture_byte_count,
            prototype.texture_sha256,
        )
    return artifacts


def _publish_shared_prototype(bundle_root: Path, prototype: _Prototype) -> None:
    with _shared_prototype_lock(bundle_root, prototype.asset_id):
        _publish_shared_prototype_locked(bundle_root, prototype)


def _validate_published_shared_prototype(
    asset_root: Path,
    prototype: _Prototype,
    artifacts: Mapping[str, tuple[int, str]],
) -> None:
    if not asset_root.is_dir():
        raise MeasuredSceneError(
            f"shared prototype target is not a directory: {asset_root}"
        )
    expected_relatives = {
        PurePosixPath(relative).relative_to(prototype.asset_id).as_posix()
        for relative in artifacts
    }
    actual = {
        path.relative_to(asset_root).as_posix()
        for path in asset_root.rglob("*")
        if path.is_file()
    }
    if actual != expected_relatives:
        raise MeasuredSceneError(
            f"shared prototype bundle layout differs: {prototype.asset_id}"
        )
    for relative, (expected_bytes, expected_hash) in artifacts.items():
        within_asset = PurePosixPath(relative).relative_to(prototype.asset_id)
        target = asset_root.joinpath(*within_asset.parts)
        if (
            target.stat().st_size != expected_bytes
            or _cached_sha256_file(target) != expected_hash
        ):
            raise MeasuredSceneError(
                "shared prototype bundle is immutable and differs: "
                f"{prototype.asset_id}"
            )


def _publish_shared_prototype_locked(bundle_root: Path, prototype: _Prototype) -> None:
    asset_root = bundle_root / prototype.asset_id
    artifacts = _prototype_artifact_hashes(prototype)
    if asset_root.exists():
        _validate_published_shared_prototype(asset_root, prototype, artifacts)
        return

    bundle_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{prototype.asset_id}.", suffix=".part", dir=bundle_root
        )
    )
    try:
        mode = _prototype_bundle_mode()
        payloads = (
            _prototype_payloads(prototype)
            if mode == "copy"
            else {prototype.wrapper_relative: prototype.wrapper_bytes}
        )
        linked_sources = {} if mode == "copy" else _linked_prototype_sources(prototype)
        if set(payloads) | set(linked_sources) != set(artifacts):
            raise MeasuredSceneError(
                f"prototype payload layout differs: {prototype.asset_id}"
            )
        for relative, content in payloads.items():
            within_asset = PurePosixPath(relative).relative_to(prototype.asset_id)
            target = staging.joinpath(*within_asset.parts)
            _inside(staging, target.resolve(), "shared prototype output")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        for relative, source in linked_sources.items():
            expected_bytes, expected_hash = artifacts[relative]
            resolved_source = source.resolve(strict=True)
            if (
                resolved_source.stat().st_size != expected_bytes
                or _cached_sha256_file(resolved_source) != expected_hash
            ):
                raise MeasuredSceneError(
                    f"prototype changed while linking: {prototype.asset_id}"
                )
            within_asset = PurePosixPath(relative).relative_to(prototype.asset_id)
            target = staging.joinpath(*within_asset.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(resolved_source)
        try:
            os.replace(staging, asset_root)
        except OSError:
            # A second process may publish the same immutable prototype between
            # the initial existence check and this atomic rename.  Accept only
            # the exact expected winner; never overwrite a divergent bundle.
            if not asset_root.exists():
                raise
            _validate_published_shared_prototype(asset_root, prototype, artifacts)
            shutil.rmtree(staging)
        for relative, (_expected_bytes, expected_hash) in artifacts.items():
            within_asset = PurePosixPath(relative).relative_to(prototype.asset_id)
            _remember_file_hash(asset_root.joinpath(*within_asset.parts), expected_hash)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise


def _publish_shared_prototype_with_slot(
    bundle_root: Path, prototype: _Prototype
) -> None:
    with _PROTOTYPE_IO_SLOTS:
        _publish_shared_prototype(bundle_root, prototype)


def _publish_shared_prototypes(
    bundle_root: Path,
    prototypes: Sequence[_Prototype],
    *,
    progress_callback: PrototypeProgressCallback | None = None,
) -> None:
    """Publish one zone-level prototype batch with bounded process-wide I/O."""

    ordered = sorted(prototypes, key=lambda value: (value.family, value.asset_id))
    total = len(ordered)
    if total == 0:
        return
    worker_count = min(_PROTOTYPE_WORKER_LIMIT, total)
    if worker_count == 1:
        for completed, prototype in enumerate(ordered, start=1):
            _publish_shared_prototype_with_slot(bundle_root, prototype)
            if progress_callback is not None:
                progress_callback(completed, total, prototype.asset_id)
        return

    futures: dict[Future[None], str] = {}
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="fireviewer-prototype"
    ) as executor:
        for prototype in ordered:
            future = executor.submit(
                _publish_shared_prototype_with_slot, bundle_root, prototype
            )
            futures[future] = prototype.asset_id
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, futures[future])


def build_measured_scene_usd(
    terrain: TerrainReference,
    placement_inventory: Path | str | Mapping[str, Any],
    asset_library: Path | str | Mapping[str, Any],
    output_root: Path | str,
    *,
    portable_root: Path | str,
    asset_roots: Mapping[str, Path | str],
    asset_bundle_root: Path | str | None = None,
    usage: str = "technical_pilot_non_final",
    selection_api: SelectionApi | None = None,
    prototype_progress_callback: PrototypeProgressCallback | None = None,
    contract_path: Path | str | None = None,
) -> MeasuredScenePackage:
    """Build one portable measured scene and its fail-closed receipt."""

    if usage not in USAGE_MODES:
        raise MeasuredSceneError(f"usage must be one of {USAGE_MODES}")
    contract, contract_file = _load_contract(contract_path)
    portable = _require_d_path(portable_root, "portable root", kind="directory")
    destination = _require_d_path(output_root, "scene output root")
    _inside(portable, destination, "scene output root")
    if destination.exists() and any(
        destination.iterdir() if destination.is_dir() else [destination]
    ):
        raise MeasuredSceneError(f"scene output overwrite is forbidden: {destination}")
    if asset_bundle_root is None:
        bundle_root = destination / "prototypes"
        bundle_scope = "output_local"
    else:
        bundle_root = _require_d_path(
            asset_bundle_root, "explicit shared asset bundle root"
        )
        _inside(portable, bundle_root, "explicit shared asset bundle root")
        if bundle_root == destination or bundle_root.is_relative_to(destination):
            raise MeasuredSceneError(
                "explicit shared asset bundle root must be outside scene output; "
                "omit it for a local bundle"
            )
        if bundle_root.exists() and not bundle_root.is_dir():
            raise MeasuredSceneError(
                "explicit shared asset bundle root is not a directory"
            )
        bundle_scope = "explicit_shared"
    bundle_root_reference = os.path.relpath(bundle_root, destination).replace("\\", "/")
    if "@" in bundle_root_reference or ":" in bundle_root_reference:
        raise MeasuredSceneError("asset bundle root reference is not portable")
    terrain_path = _require_d_path(terrain.root_usd, "terrain root USD", kind="file")
    _inside(portable, terrain_path, "terrain root USD")
    if terrain_path.suffix.casefold() not in {".usd", ".usda", ".usdc"}:
        raise MeasuredSceneError("terrain root must be a USD stage")
    origin = tuple(_finite(value, "terrain origin") for value in terrain.origin_l93_m)
    if len(origin) != 2:
        raise MeasuredSceneError("terrain origin must contain easting and northing")
    if isinstance(terrain.vertical_origin_mm, bool) or not isinstance(
        terrain.vertical_origin_mm, int
    ):
        raise MeasuredSceneError("terrain vertical origin must be integer millimetres")
    terrain = TerrainReference(
        terrain_path, (origin[0], origin[1]), terrain.vertical_origin_mm
    )

    roots: dict[str, Path] = {}
    for logical_root, raw_path in asset_roots.items():
        name = _require_nonempty_string(logical_root, "asset root name")
        root = _require_d_path(raw_path, f"asset root {name}", kind="directory")
        roots[name] = _inside(portable, root, f"asset root {name}")
    if not roots:
        raise MeasuredSceneError("at least one asset root mapping is required")

    inventory = _load_json(placement_inventory, "placement inventory")
    library = _load_json(asset_library, "asset library")
    _validate_inventory(inventory)
    indexed_assets = _validate_catalog(library)
    zone_id = _require_nonempty_string(inventory.get("zone_id"), "zone_id")
    inventory_build_id = _require_sha256(
        inventory.get("build_id"), "inventory build_id"
    )
    catalog_revision = _require_sha256(
        library.get("catalog_revision"), "catalog_revision"
    )
    rule_versions = contract["selection"]["rule_versions"]
    selector = selection_api or _default_selection_api
    output_reference_root = destination
    terrain_reference = _relative_reference(
        terrain_path,
        output_root=output_reference_root,
        portable_root=portable,
    )

    selected: dict[str, list[tuple[Mapping[str, Any], str, str, int]]] = {
        family: [] for family, _ in FAMILIES
    }
    selected_asset_ids: set[tuple[str, str]] = set()
    for family, _default_category in FAMILIES:
        source_family = _inventory_family(inventory, family)
        candidates = sorted(
            source_family["candidates"],
            key=lambda value: str(value["candidate_id"]),
        )
        for candidate in candidates:
            if candidate["status"] != "valid":
                continue
            candidate_id = str(candidate["candidate_id"])
            category = _candidate_category(family, candidate, library)
            rule_version = rule_versions.get(category)
            if not isinstance(rule_version, str) or not rule_version:
                raise MeasuredSceneError(
                    f"selection contract lacks rule version for {category}"
                )
            fixed_asset_id = candidate.get("fixed_asset_id")
            if fixed_asset_id is None:
                asset_id, seed = _select(
                    selector,
                    library,
                    category=category,
                    zone_id=zone_id,
                    candidate_id=candidate_id,
                    rule_version=rule_version,
                    usage=usage,
                    metadata=_candidate_selection_metadata(family, candidate, category),
                )
            else:
                asset_id = _require_nonempty_string(
                    fixed_asset_id, "fixed selected asset id"
                )
                seed = _fixed_selection_seed(
                    zone_id=zone_id,
                    candidate_id=candidate_id,
                    asset_id=asset_id,
                    rule_version=rule_version,
                )
            if (
                asset_id not in indexed_assets
                or indexed_assets[asset_id].get("category") != category
            ):
                raise MeasuredSceneError(
                    f"catalog selected invalid {category} asset {asset_id}"
                )
            selected[family].append((candidate, category, asset_id, seed))
            selected_asset_ids.add((family, asset_id))

    prototypes: dict[tuple[str, str], _Prototype] = {}
    final_blockers: list[str] = []
    for family, asset_id in sorted(selected_asset_ids):
        asset = indexed_assets[asset_id]
        extents, native_min_y = _native_bounds(asset)
        blockers = _qualification_blockers(asset)
        final_blockers.extend(f"asset:{asset_id}:{blocker}" for blocker in blockers)
        prototypes[(family, asset_id)] = _plan_prototype_bundle(
            asset,
            family=family,
            asset_roots=roots,
            portable_root=portable,
            bundle_root=bundle_root,
            output_root=destination,
            native_min_y=native_min_y,
            native_extents=extents,
            qualification_blockers=blockers,
        )
    final_blockers = sorted(set(final_blockers))
    instances: dict[str, list[_Instance]] = {family: [] for family, _ in FAMILIES}
    for family, rows in selected.items():
        for candidate, category, asset_id, seed in rows:
            instances[family].append(
                _instance_from_candidate(
                    candidate,
                    family=family,
                    asset_id=asset_id,
                    asset_category=category,
                    selection_seed=seed,
                    prototype=prototypes[(family, asset_id)],
                    terrain=terrain,
                )
            )
        instances[family].sort(key=lambda value: value.candidate_id)

    non_uniform_buildings = sorted(
        instance.candidate_id
        for instance in instances["buildings"]
        if not (
            math.isclose(
                instance.scale[0], instance.scale[1], rel_tol=0.0, abs_tol=1e-9
            )
            and math.isclose(
                instance.scale[1], instance.scale[2], rel_tol=0.0, abs_tol=1e-9
            )
        )
    )
    final_blockers.extend(
        f"candidate:{candidate_id}:non_uniform_building_scale"
        for candidate_id in non_uniform_buildings
    )
    final_blockers = sorted(set(final_blockers))
    if usage == "final_scene" and final_blockers:
        raise MeasuredSceneError(
            "final scene is blocked by technical-only normalization: "
            + ", ".join(final_blockers)
        )

    reconciliation: dict[str, Any] = {}
    for family, _category in FAMILIES:
        source = _inventory_family(inventory, family)
        instance_count = len(instances[family])
        blocked = _blocked_records(source["candidates"])
        if instance_count != source["valid_count"]:
            raise MeasuredSceneError(
                f"{family} valid-to-instance reconciliation failed"
            )
        if len(blocked) != source["ambiguous_count"] + source["rejected_count"]:
            raise MeasuredSceneError(f"{family} blocked reconciliation failed")
        if source["source_count"] != instance_count + len(blocked):
            raise MeasuredSceneError(f"{family} source-to-scene reconciliation failed")
        candidate_ids = [instance.candidate_id for instance in instances[family]]
        placeholder_candidate_ids = [
            instance.candidate_id
            for instance in instances[family]
            if prototypes[(family, instance.asset_id)].availability == "placeholder_usd"
        ]
        category_counts: dict[str, int] = {}
        for instance in instances[family]:
            category_counts[instance.asset_category] = (
                category_counts.get(instance.asset_category, 0) + 1
            )
        reconciliation[family] = {
            "source_count": source["source_count"],
            "valid_count": source["valid_count"],
            "instance_count": instance_count,
            "blocked_count": len(blocked),
            "blocked_candidates": blocked,
            "instanced_candidate_ids_sha256": sha256_bytes(
                canonical_json_bytes(candidate_ids)
            ),
            "placeholder_instance_count": len(placeholder_candidate_ids),
            "placeholder_candidate_ids_sha256": sha256_bytes(
                canonical_json_bytes(placeholder_candidate_ids)
            ),
            "asset_category_counts": dict(sorted(category_counts.items())),
            "quota_applied": False,
            "thinning_applied": False,
        }

    contract_hash = sha256_file(contract_file)
    scene_bytes = _render_scene(
        terrain_reference=terrain_reference,
        terrain=terrain,
        zone_id=zone_id,
        inventory_build_id=inventory_build_id,
        catalog_revision=catalog_revision,
        contract_sha256=contract_hash,
        prototypes=prototypes,
        instances=instances,
    )
    scene_hash = sha256_bytes(scene_bytes)
    status = (
        "technical_pilot_non_final"
        if usage == "technical_pilot_non_final"
        else "assembled_final_candidate"
    )
    prototype_receipts = [
        {
            "asset_id": prototype.asset_id,
            "family": prototype.family,
            "availability": prototype.availability,
            "fallback_resolution": prototype.fallback_resolution,
            "native_bounds_extents": list(prototype.native_extents),
            "native_min_y": prototype.native_min_y,
            "source_up_axis": prototype.source_up_axis,
            "source_usd": {
                "path": prototype.source_relative,
                "sha256": prototype.source_sha256,
                "byte_count": prototype.source_byte_count,
            },
            "texture": (
                None
                if prototype.texture_path is None
                else {
                    "path": prototype.texture_relative,
                    "sha256": prototype.texture_sha256,
                    "byte_count": prototype.texture_byte_count,
                }
            ),
            "wrapper": {
                "path": prototype.wrapper_relative,
                "sha256": sha256_bytes(prototype.wrapper_bytes),
                "byte_count": len(prototype.wrapper_bytes),
            },
            "material": _prototype_material_receipt(prototype.material_policy),
            "qualification_blockers": list(prototype.qualification_blockers),
        }
        for _key, prototype in sorted(prototypes.items())
    ]
    placeholder_prototype_count = sum(
        prototype.availability == "placeholder_usd" for prototype in prototypes.values()
    )
    placeholder_instance_count = sum(
        reconciliation[family]["placeholder_instance_count"]
        for family, _category in FAMILIES
    )
    receipt_without_hash: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "algorithm": ALGORITHM,
        "status": status,
        "accepted_final": False,
        "zone_id": zone_id,
        "inventory_build_id": inventory_build_id,
        "inventory_sha256": inventory["inventory_sha256"],
        "catalog_revision": catalog_revision,
        "contract_sha256": contract_hash,
        "terrain": {
            "root_reference": terrain_reference,
            "sha256": sha256_file(terrain_path),
            "origin_l93_m": list(terrain.origin_l93_m),
            "vertical_origin_mm": terrain.vertical_origin_mm,
        },
        "scene": {
            "path": SCENE_FILE_NAME,
            "sha256": scene_hash,
            "byte_count": len(scene_bytes),
        },
        "prototype_count": len(prototypes),
        "placeholder_prototype_count": placeholder_prototype_count,
        "placeholder_instance_count": placeholder_instance_count,
        "prototypes": prototype_receipts,
        "prototype_bundle": {
            "root_reference": bundle_root_reference,
            "scope": bundle_scope,
            "selected_asset_count": len(prototypes),
            "unused_catalog_assets_copied": 0,
            "absolute_asset_paths": False,
            "bundle_sha256": sha256_bytes(
                canonical_json_bytes(
                    [
                        {
                            "asset_id": prototype["asset_id"],
                            "availability": prototype["availability"],
                            "fallback_resolution": prototype["fallback_resolution"],
                            "source_usd": prototype["source_usd"],
                            "texture": prototype["texture"],
                            "wrapper": prototype["wrapper"],
                        }
                        for prototype in prototype_receipts
                    ]
                )
            ),
        },
        "reconciliation": reconciliation,
        "final_blockers": final_blockers,
        "placement_policy": {
            "source": "MNT_MNS_inventory_with_stable_SIG_context_features",
            "quota_applied": False,
            "thinning_applied": False,
            "fallback_primitive_used": False,
            "catalog_placeholder_usd_used": placeholder_instance_count > 0,
            "tree_scale": inventory.get("trees", {}).get(
                "geometry_scale_policy", "uniform_from_mns_mnt_height"
            ),
            "tree_base_elevation": inventory.get("trees", {}).get(
                "base_elevation_policy", "point_mnt_ground_elevation"
            ),
            "tree_yaw": "deterministic_selection_seed",
            "non_uniform_building_scale_candidate_ids": non_uniform_buildings,
            "catalogue_entries_consumed": False,
            "repeated_asset_ids_allowed": True,
        },
    }
    receipt_without_hash["build_id"] = sha256_bytes(
        canonical_json_bytes(receipt_without_hash)
    )
    receipt = dict(receipt_without_hash)
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt_without_hash))
    receipt_bytes = canonical_json_bytes(receipt, pretty=True)

    if destination.exists():
        raise MeasuredSceneError(f"scene output overwrite is forbidden: {destination}")
    if not destination.parent.is_dir():
        raise MeasuredSceneError(
            f"scene output parent must already exist: {destination.parent}"
        )
    staging = destination.with_name(f".{destination.name}.part")
    _inside(portable, staging, "scene staging root")
    if staging.exists():
        raise MeasuredSceneError(f"scene staging root already exists: {staging}")
    scene_path = destination / SCENE_FILE_NAME
    receipt_path = destination / RECEIPT_FILE_NAME
    try:
        if bundle_scope == "explicit_shared":
            bundle_root.mkdir(parents=True, exist_ok=True)
            _publish_shared_prototypes(
                bundle_root,
                list(prototypes.values()),
                progress_callback=prototype_progress_callback,
            )
            prototype_payloads: dict[tuple[str, str], dict[str, bytes]] = {}
        else:
            prototype_payloads = {
                key: _prototype_payloads(prototype)
                for key, prototype in sorted(prototypes.items())
            }
        staging.mkdir(parents=False, exist_ok=False)
        if bundle_scope == "output_local":
            (staging / "prototypes").mkdir(parents=False, exist_ok=False)
        for _key, prototype in sorted(prototypes.items()):
            if bundle_scope != "output_local":
                continue
            for relative, content in prototype_payloads[_key].items():
                target = (staging / "prototypes").joinpath(
                    *PurePosixPath(relative).parts
                )
                _inside(staging, target.resolve(), "prototype bundle output")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        (staging / SCENE_FILE_NAME).write_bytes(scene_bytes)
        (staging / RECEIPT_FILE_NAME).write_bytes(receipt_bytes)
        validate_measured_scene_package(staging)
        os.replace(staging, destination)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return MeasuredScenePackage(
        output_root=destination,
        scene=scene_path,
        receipt=receipt_path,
        status=status,
        building_instance_count=len(instances["buildings"]),
        tree_instance_count=len(instances["trees"]),
        context_asset_instance_count=len(instances["context_assets"]),
    )


def _validate_bundle_artifact(
    root: Path,
    record: Any,
    *,
    label: str,
    expected_prefix: PurePosixPath,
) -> tuple[Path, str]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise MeasuredSceneError(f"{label} bundle record is invalid")
    relative = _portable_catalog_path(record.get("path"), f"{label} bundle path")
    if relative.parts[: len(expected_prefix.parts)] != expected_prefix.parts:
        raise MeasuredSceneError(f"{label} bundle path is outside its prototype")
    lexical_path = root.joinpath(*relative.parts)
    _inside(root, lexical_path.absolute(), f"{label} bundle file")
    if lexical_path.is_symlink():
        if _prototype_bundle_mode() != "linked":
            raise MeasuredSceneError(
                f"{label} external bundle link is forbidden outside worker mode"
            )
        resolved_path = lexical_path.resolve(strict=True)
    else:
        resolved_path = lexical_path.resolve()
        _inside(root, resolved_path, f"{label} bundle file")
    if not resolved_path.is_file():
        raise MeasuredSceneError(f"{label} bundle file is missing")
    expected_bytes = _integer(
        record.get("byte_count"), f"{label} bundle byte_count", minimum=1
    )
    expected_hash = _require_sha256(record.get("sha256"), f"{label} bundle sha256")
    if (
        resolved_path.stat().st_size != expected_bytes
        or _cached_sha256_file(resolved_path) != expected_hash
    ):
        raise MeasuredSceneError(f"{label} bundle bytes differ from receipt")
    # USD references, suffix checks and wrapper reconstruction are properties of
    # the portable bundle path.  A linked worker target may intentionally keep its
    # immutable image name, which must not leak into the authored wrapper.
    return lexical_path, relative.as_posix()


def _validate_prototype_bundle(
    root: Path, receipt: Mapping[str, Any], scene_text: str
) -> None:
    records = receipt.get("prototypes")
    count = _integer(receipt.get("prototype_count"), "prototype_count")
    if not isinstance(records, list) or len(records) != count:
        raise MeasuredSceneError("prototype receipt count differs")
    bundle = receipt.get("prototype_bundle")
    if not isinstance(bundle, Mapping) or set(bundle) != {
        "root_reference",
        "scope",
        "selected_asset_count",
        "unused_catalog_assets_copied",
        "absolute_asset_paths",
        "bundle_sha256",
    }:
        raise MeasuredSceneError("prototype bundle receipt is invalid")
    if (
        bundle.get("scope") not in {"output_local", "explicit_shared"}
        or bundle.get("selected_asset_count") != count
        or bundle.get("unused_catalog_assets_copied") != 0
        or bundle.get("absolute_asset_paths") is not False
    ):
        raise MeasuredSceneError("prototype bundle is not minimal and portable")
    root_reference = _require_nonempty_string(
        bundle.get("root_reference"), "prototype bundle root reference"
    )
    if (
        "\\" in root_reference
        or "@" in root_reference
        or ":" in root_reference
        or PurePosixPath(root_reference).is_absolute()
    ):
        raise MeasuredSceneError("prototype bundle root reference is not portable")
    bundle_root = root.joinpath(*PurePosixPath(root_reference).parts).resolve()
    _require_d_path(bundle_root, "resolved prototype bundle root", kind="directory")
    if bundle.get("scope") == "output_local":
        if bundle_root != (root / "prototypes").resolve():
            raise MeasuredSceneError("local prototype bundle root differs")
    elif bundle_root == root or bundle_root.is_relative_to(root):
        raise MeasuredSceneError("shared prototype bundle is not outside scene output")

    expected_files = {SCENE_FILE_NAME, RECEIPT_FILE_NAME}
    bundle_basis: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        required_keys = {
            "asset_id",
            "family",
            "native_bounds_extents",
            "native_min_y",
            "source_up_axis",
            "source_usd",
            "texture",
            "wrapper",
            "material",
            "qualification_blockers",
            "fallback_resolution",
        }
        if not isinstance(record, Mapping) or set(record) not in {
            frozenset(required_keys),
            frozenset(required_keys | {"availability"}),
        }:
            raise MeasuredSceneError(f"prototype receipt {index} is invalid")
        asset_id = _bundle_asset_id(
            _require_nonempty_string(record.get("asset_id"), "prototype asset_id")
        )
        availability = record.get("availability", "real_usd")
        if availability not in {"real_usd", "placeholder_usd"}:
            raise MeasuredSceneError(f"prototype {asset_id} availability is invalid")
        fallback_resolution = record.get("fallback_resolution")
        if fallback_resolution is not None:
            required_resolution = {
                "used",
                "donor_asset_id",
                "donor_category",
                "donor_source_tier",
                "compatibility_mode",
                "metadata_match_score",
                "resolution_sha256",
            }
            if (
                not isinstance(fallback_resolution, Mapping)
                or set(fallback_resolution) != required_resolution
                or not isinstance(fallback_resolution.get("used"), bool)
                or not isinstance(fallback_resolution.get("donor_asset_id"), str)
                or not isinstance(fallback_resolution.get("donor_category"), str)
                or not isinstance(fallback_resolution.get("donor_source_tier"), str)
                or not isinstance(fallback_resolution.get("compatibility_mode"), str)
                or not isinstance(fallback_resolution.get("metadata_match_score"), int)
            ):
                raise MeasuredSceneError(
                    f"prototype {asset_id} fallback resolution is invalid"
                )
            _require_sha256(
                fallback_resolution.get("resolution_sha256"),
                f"prototype {asset_id} fallback resolution sha256",
            )
        source_up_axis = record.get("source_up_axis")
        if source_up_axis not in {"Y", "Z"}:
            raise MeasuredSceneError(f"prototype {asset_id} source up axis is invalid")
        source_path, source_relative = _validate_bundle_artifact(
            bundle_root,
            record.get("source_usd"),
            label=f"prototype {asset_id} source USD",
            expected_prefix=PurePosixPath(asset_id),
        )
        wrapper_path, wrapper_relative = _validate_bundle_artifact(
            bundle_root,
            record.get("wrapper"),
            label=f"prototype {asset_id} wrapper",
            expected_prefix=PurePosixPath(asset_id),
        )
        if source_path.suffix.casefold() not in {".usd", ".usda", ".usdc", ".usdz"}:
            raise MeasuredSceneError(f"prototype {asset_id} source is not USD")
        material = record.get("material")
        direct_pbr = material == _prototype_material_receipt("source_package_pbr")
        scoped_pbr = material == _prototype_material_receipt("scoped_source_pbr")
        texture_relative: str | None
        if direct_pbr:
            if (
                source_path.suffix.casefold() != ".usdz"
                or record.get("texture") is not None
            ):
                raise MeasuredSceneError(
                    f"prototype {asset_id} source PBR package differs"
                )
            texture_relative = None
            expected_wrapper = _render_source_package_wrapper(
                asset_id=asset_id,
                source_file_name=source_path.name,
                source_up_axis=source_up_axis,
            )
        else:
            if not scoped_pbr and material != _prototype_material_receipt(
                "fireviewer_color_override"
            ):
                raise MeasuredSceneError(
                    f"prototype {asset_id} material override differs"
                )
            texture_path, texture_relative = _validate_bundle_artifact(
                bundle_root,
                record.get("texture"),
                label=f"prototype {asset_id} texture",
                expected_prefix=PurePosixPath(asset_id) / "textures",
            )
            if texture_path.suffix.casefold() not in {".png", ".jpg", ".jpeg"}:
                raise MeasuredSceneError(
                    f"prototype {asset_id} colour texture format differs"
                )
            if scoped_pbr:
                expected_wrapper = _render_scoped_source_wrapper(
                    asset_id=asset_id,
                    source_file_name=source_path.name,
                    source_up_axis=source_up_axis,
                )
            else:
                expected_wrapper = _render_prototype_wrapper(
                    asset_id=asset_id,
                    source_file_name=source_path.name,
                    texture_reference=(
                        PurePosixPath("textures") / texture_path.name
                    ).as_posix(),
                    source_up_axis=source_up_axis,
                )
        if wrapper_path.read_bytes() != expected_wrapper:
            raise MeasuredSceneError(
                f"prototype {asset_id} local material wrapper differs"
            )
        wrapper_reference = os.path.relpath(wrapper_path, root).replace("\\", "/")
        source_reference = os.path.relpath(source_path, root).replace("\\", "/")
        if f"@{wrapper_reference}@" not in scene_text:
            raise MeasuredSceneError(
                f"scene does not reference bundled wrapper for {asset_id}"
            )
        if f"@{source_reference}@" in scene_text:
            raise MeasuredSceneError(f"scene bypasses bundled wrapper for {asset_id}")
        if bundle.get("scope") == "output_local":
            expected_files.update(
                f"prototypes/{relative}"
                for relative in (source_relative, texture_relative, wrapper_relative)
                if relative is not None
            )
        asset_files = {
            path.relative_to(bundle_root / asset_id).as_posix()
            for path in (bundle_root / asset_id).rglob("*")
            if path.is_file()
        }
        expected_asset_files = {
            PurePosixPath(relative).relative_to(asset_id).as_posix()
            for relative in (source_relative, texture_relative, wrapper_relative)
            if relative is not None
        }
        if asset_files != expected_asset_files:
            raise MeasuredSceneError(
                f"prototype {asset_id} bundle contains missing or unused files"
            )
        basis = {
            "asset_id": asset_id,
            "source_usd": record["source_usd"],
            "texture": record["texture"],
            "wrapper": record["wrapper"],
        }
        if "availability" in record:
            basis["availability"] = availability
        basis["fallback_resolution"] = fallback_resolution
        bundle_basis.append(basis)
    if _require_sha256(
        bundle.get("bundle_sha256"), "prototype bundle sha256"
    ) != sha256_bytes(canonical_json_bytes(bundle_basis)):
        raise MeasuredSceneError("prototype bundle hash differs")
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise MeasuredSceneError(
            "measured scene package contains missing or unused prototype files"
        )


def validate_measured_scene_package(output_root: Path | str) -> dict[str, Any]:
    """Reopen hashes and reconciliation without requiring OpenUSD."""

    root = _require_d_path(output_root, "measured scene package", kind="directory")
    scene = root / SCENE_FILE_NAME
    receipt_file = root / RECEIPT_FILE_NAME
    if not scene.is_file() or not receipt_file.is_file():
        raise MeasuredSceneError("measured scene package is incomplete")
    receipt = _load_json(receipt_file, "measured scene receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("algorithm") != ALGORITHM:
        raise MeasuredSceneError("measured scene receipt schema is invalid")
    supplied_receipt_hash = _require_sha256(
        receipt.get("receipt_sha256"), "receipt_sha256"
    )
    without_receipt_hash = dict(receipt)
    without_receipt_hash.pop("receipt_sha256", None)
    if (
        sha256_bytes(canonical_json_bytes(without_receipt_hash))
        != supplied_receipt_hash
    ):
        raise MeasuredSceneError("measured scene receipt hash mismatch")
    if receipt.get("accepted_final") is not False:
        raise MeasuredSceneError("the assembler must never grant final acceptance")
    scene_record = receipt.get("scene")
    if (
        not isinstance(scene_record, Mapping)
        or scene_record.get("path") != SCENE_FILE_NAME
    ):
        raise MeasuredSceneError("measured scene receipt scene record is invalid")
    if scene.stat().st_size != scene_record.get("byte_count") or sha256_file(
        scene
    ) != scene_record.get("sha256"):
        raise MeasuredSceneError("measured scene bytes differ from receipt")
    text = scene.read_text(encoding="utf-8")
    if "def Mesh " in text or "def Cube " in text or "fallback" in text.casefold():
        raise MeasuredSceneError(
            "measured scene contains a forbidden fallback primitive"
        )
    _validate_prototype_bundle(root, receipt, text)
    prototype_records = receipt.get("prototypes", [])
    placeholder_prototypes = sum(
        isinstance(record, Mapping) and record.get("availability") == "placeholder_usd"
        for record in prototype_records
    )
    if receipt.get("placeholder_prototype_count", 0) != placeholder_prototypes:
        raise MeasuredSceneError("placeholder prototype count differs")
    placeholder_ids = {
        record["asset_id"]
        for record in prototype_records
        if isinstance(record, Mapping)
        and record.get("availability") == "placeholder_usd"
    }
    recorded_placeholder_instances = receipt.get("placeholder_instance_count", 0)
    reconciled_placeholder_instances = 0
    reconciliation_payload = receipt.get("reconciliation", {})
    receipt_families = ["buildings", "trees"]
    if (
        isinstance(reconciliation_payload, Mapping)
        and "context_assets" in reconciliation_payload
    ):
        receipt_families.append("context_assets")
    for family in receipt_families:
        reconciliation = reconciliation_payload.get(family)
        if not isinstance(reconciliation, Mapping):
            raise MeasuredSceneError(f"measured scene lacks {family} reconciliation")
        family_count = reconciliation.get("placeholder_instance_count", 0)
        if (
            not isinstance(family_count, int)
            or family_count < 0
            or family_count > reconciliation.get("instance_count", -1)
        ):
            raise MeasuredSceneError(f"{family} placeholder count is invalid")
        placeholder_ids_hash = reconciliation.get("placeholder_candidate_ids_sha256")
        if placeholder_ids_hash is not None:
            _require_sha256(
                placeholder_ids_hash,
                f"{family} placeholder candidate ids sha256",
            )
        elif family_count != 0:
            raise MeasuredSceneError(
                f"{family} placeholder candidate ids hash is missing"
            )
        reconciled_placeholder_instances += family_count
    if (
        not isinstance(recorded_placeholder_instances, int)
        or recorded_placeholder_instances < 0
        or recorded_placeholder_instances != reconciled_placeholder_instances
        or (not placeholder_ids and recorded_placeholder_instances != 0)
    ):
        raise MeasuredSceneError("placeholder instance count is invalid")
    placement_policy = receipt.get("placement_policy")
    if recorded_placeholder_instances > 0:
        if (
            not isinstance(placement_policy, Mapping)
            or placement_policy.get("catalog_placeholder_usd_used") is not True
        ):
            raise MeasuredSceneError("placeholder placement policy differs")
    elif (
        isinstance(placement_policy, Mapping)
        and placement_policy.get("catalog_placeholder_usd_used", False) is not False
    ):
        raise MeasuredSceneError("placeholder placement policy differs")
    for family in receipt_families:
        reconciliation = reconciliation_payload.get(family)
        if not isinstance(reconciliation, Mapping):
            raise MeasuredSceneError(f"measured scene lacks {family} reconciliation")
        if reconciliation.get("source_count") != reconciliation.get(
            "instance_count"
        ) + reconciliation.get("blocked_count"):
            raise MeasuredSceneError(
                f"measured scene {family} source reconciliation differs"
            )
        if reconciliation.get("valid_count") != reconciliation.get("instance_count"):
            raise MeasuredSceneError(
                f"measured scene {family} valid reconciliation differs"
            )
        if (
            reconciliation.get("quota_applied") is not False
            or reconciliation.get("thinning_applied") is not False
        ):
            raise MeasuredSceneError(
                f"measured scene {family} illegally modifies quantity"
            )
    return receipt


def _parse_asset_roots(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise MeasuredSceneError("--asset-root must be unique NAME=PATH values")
        result[name] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-root", type=Path, required=True)
    parser.add_argument(
        "--terrain-origin",
        type=float,
        nargs=2,
        required=True,
        metavar=("EASTING", "NORTHING"),
    )
    parser.add_argument("--terrain-vertical-origin-mm", type=int, default=0)
    parser.add_argument("--placement-inventory", type=Path, required=True)
    parser.add_argument("--asset-library", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        action="append",
        default=[],
        help="Logical root mapping NAME=PATH",
    )
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--asset-bundle-root",
        type=Path,
        help=(
            "Explicit immutable shared prototype bundle on D:. If omitted, "
            "the selected prototypes are bundled under output-root/prototypes."
        ),
    )
    parser.add_argument(
        "--usage", choices=USAGE_MODES, default="technical_pilot_non_final"
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--execute", action="store_true", required=True)
    options = parser.parse_args(argv)
    package = build_measured_scene_usd(
        TerrainReference(
            root_usd=options.terrain_root,
            origin_l93_m=tuple(options.terrain_origin),
            vertical_origin_mm=options.terrain_vertical_origin_mm,
        ),
        options.placement_inventory,
        options.asset_library,
        options.output_root,
        portable_root=options.portable_root,
        asset_roots=_parse_asset_roots(options.asset_root),
        asset_bundle_root=options.asset_bundle_root,
        usage=options.usage,
        contract_path=options.contract,
    )
    receipt = validate_measured_scene_package(package.output_root)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "scene": str(package.scene),
                "buildings": package.building_instance_count,
                "trees": package.tree_instance_count,
                "accepted_final": False,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALGORITHM",
    "CONTRACT_SCHEMA",
    "RECEIPT_SCHEMA",
    "SCENE_SCHEMA",
    "MeasuredSceneError",
    "MeasuredScenePackage",
    "TerrainReference",
    "build_measured_scene_usd",
    "canonical_json_bytes",
    "main",
    "validate_measured_scene_package",
]
