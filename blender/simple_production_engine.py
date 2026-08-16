"""Headless production engine for portable measured FireViewer zones.

The pod owns no database.  A job is a deterministic directory below ``/work``:
one JSON plan, one WFS context snapshot, sequential 500 m tile packages, one
shared prototype bundle, one unified ``zone.usda`` and one downloadable ZIP.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from build_asset_library_53 import validate_asset_library
from build_reference_usd_asset_library import (
    SCHEMA as REFERENCE_ASSET_LIBRARY_SCHEMA,
)
from build_reference_usd_asset_library import (
    validate_reference_asset_library,
)
from fixed_asset_placement import (
    EMPTY_REQUEST as EMPTY_FIXED_ASSET_REQUEST,
)
from fixed_asset_placement import (
    FixedAssetPlacementError,
    asset_choices,
)
from fixed_asset_placement import (
    normalize_request as normalize_fixed_asset_request,
)
from fixed_asset_placement import (
    project_request as project_fixed_asset_request,
)
from fixed_asset_placement import (
    request_sha256 as fixed_asset_request_sha256,
)
from fixed_asset_placement import (
    schema_path as fixed_asset_schema_path,
)
from fixed_asset_placement import (
    template_path as fixed_asset_template_path,
)
from portable_scene_package import seal_map_upload_package, validate_map_upload_package
from prepare_simple_measured_tile_sources import PreparedSources, prepare_sources
from prepare_simple_measured_zone_context import ZoneContext, prepare_zone_context
from produce_simple_measured_tile import (
    SimpleMeasuredTilePackage,
    produce_simple_measured_tile,
    validate_simple_measured_tile_package,
)
from build_measured_scene_usd import remember_validated_file_hash
from pyproj import Transformer
from render_simple_zone_gallery import BLEND_NAME

SCHEMA = "fireviewer.simple-production-engine.v1"
PLAN_SCHEMA = "fireviewer.simple-measured-zone-plan.v1"
ZONE_RECEIPT_SCHEMA = "fireviewer.simple-measured-zone-production.v1"
TILE_SIZE_M = 500
HALO_M = 10
SOURCE_METATILE_TILES = 4
SOURCE_METATILE_SIZE_M = TILE_SIZE_M * SOURCE_METATILE_TILES
ENTRY_STAGE = "zone.usda"
ZIP_NAME = "fireviewer-zone.zip"
STATUS_NAME = "job-status.json"
PLAN_NAME = "zone-plan.json"
ZONE_CONTEXT_NAME = "zone-context.json"
ZONE_RECEIPT_NAME = "zone.done.json"
DATASET_ENTRY_NAME = "dataset-entry.json"
DATASET_PUBLICATION_NAME = "dataset-publication.json"
FIXED_ASSET_REQUEST_NAME = "fixed-asset-placements.v1.json"
CAPTURE_COUNT = 0
PARALLEL_HEARTBEAT_SECONDS = 15.0
DEFAULT_TILE_STALL_TIMEOUT_SECONDS = 480
CHECKPOINT_PUBLISH_TIMEOUT_SECONDS = 180.0
CHECKPOINT_COPY_USES_SUBPROCESS = os.name != "nt"
PROTOTYPE_BUNDLE_NAMESPACE_SCHEMA = "fireviewer.prototype-bundle-namespace.v1"
PROTOTYPE_BUNDLE_COLLECTION_NAME = "prototype-bundles"
TILE_CHECKPOINT_SCHEMA = "fireviewer.simple-measured-tile-checkpoint.v1"
TILE_CHECKPOINT_COLLECTION_NAME = "tile-checkpoints"
TILE_CHECKPOINT_VERSION = "v1"


class SimpleProductionError(RuntimeError):
    """The pod configuration or requested zone cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    work_root: Path
    portable_root: Path
    asset_library: Path
    review_batch: Path
    elevation_revision: str
    orthophoto_revision: str
    context_revision: str
    max_side_m: int = 15_000
    max_tiles: int = 900
    tile_workers: int = 6
    source_workers: int = 12
    tile_stall_timeout_seconds: int = DEFAULT_TILE_STALL_TIMEOUT_SECONDS
    blender: Path = Path("/opt/blender/blender")
    dataset_id: str | None = None
    dataset_publication_required: bool = True
    scratch_root: Path | None = None

    @classmethod
    def from_environment(cls) -> ProductionConfig:
        return cls(
            work_root=Path(os.environ.get("FIREVIEWER_WORK_ROOT", "/work")),
            portable_root=Path(os.environ.get("FIREVIEWER_PORTABLE_ROOT", "/")),
            asset_library=Path(
                os.environ.get(
                    "FIREVIEWER_ASSET_LIBRARY",
                    "/opt/fireviewer/assets/reference-asset-library.v1.json",
                )
            ),
            review_batch=Path(
                os.environ.get(
                    "FIREVIEWER_REVIEW_BATCH",
                    "/opt/fireviewer/assets/simready_final_0001_0294",
                )
            ),
            elevation_revision=os.environ.get(
                "FIREVIEWER_ELEVATION_REVISION", "geopf-lidar-hd-current"
            ),
            orthophoto_revision=os.environ.get(
                "FIREVIEWER_ORTHOPHOTO_REVISION", "geopf-orthophotos-current"
            ),
            context_revision=os.environ.get(
                "FIREVIEWER_CONTEXT_REVISION", "bdtopo-v3-current"
            ),
            max_side_m=int(os.environ.get("FIREVIEWER_MAX_SIDE_M", "15000")),
            max_tiles=int(os.environ.get("FIREVIEWER_MAX_TILES", "900")),
            tile_workers=int(os.environ.get("FIREVIEWER_TILE_WORKERS", "6")),
            source_workers=int(os.environ.get("FIREVIEWER_SOURCE_WORKERS", "12")),
            tile_stall_timeout_seconds=int(
                os.environ.get(
                    "FIREVIEWER_TILE_STALL_TIMEOUT_SECONDS",
                    str(DEFAULT_TILE_STALL_TIMEOUT_SECONDS),
                )
            ),
            blender=Path(os.environ.get("FIREVIEWER_BLENDER", "/opt/blender/blender")),
            dataset_id=(os.environ.get("FIREVIEWER_HF_DATASET_ID", "").strip() or None),
            dataset_publication_required=os.environ.get(
                "FIREVIEWER_HF_PUBLICATION_REQUIRED", "1"
            )
            .strip()
            .casefold()
            not in {"0", "false", "no", "off"},
            scratch_root=(
                Path(value)
                if (value := os.environ.get("FIREVIEWER_SCRATCH_ROOT", "").strip())
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TilePlan:
    tile_id: str
    origin_l93_m: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ZonePlan:
    zone_id: str
    latitude: float
    longitude: float
    side_m: int
    center_l93_m: tuple[float, float]
    requested_bounds_l93_m: tuple[float, float, float, float]
    production_bounds_l93_m: tuple[int, int, int, int]
    context_bounds_l93_m: tuple[int, int, int, int]
    tiles: tuple[TilePlan, ...]


PrepareContext = Callable[..., ZoneContext]
PrepareSources = Callable[..., PreparedSources]
ProduceTile = Callable[..., SimpleMeasuredTilePackage]
ProgressCallback = Callable[[float, str], None]
DatasetPublicationProgress = Callable[[str, str], None]
ArchiveReadyCallback = Callable[[Path, int, str], None]
ZipProgress = Callable[[int, int, int, int, str], None]
HF_PUBLICATION_MAX_ATTEMPTS = 1
HF_PUBLICATION_BACKOFF_SECONDS: tuple[float, ...] = ()
GalleryProgress = Callable[[int, str], None]
GalleryItems = list[tuple[str, str]]
RenderGallery = Callable[[Path, bool, GalleryProgress | None], GalleryItems]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_temporary_file(path: Path, *, label: str = "") -> Path:
    """Allocate one same-filesystem staging file owned by this invocation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{label}.part" if label else ".part"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    return Path(temporary)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = _canonical_bytes(payload) + b"\n"
    temporary = _unique_temporary_file(path, label="json")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimpleProductionError(f"{label} JSON invalide: {error}") from error
    if not isinstance(value, dict):
        raise SimpleProductionError(f"{label} doit être un objet JSON")
    return value


def _tile_checkpoint_paths(job_root: Path, tile_id: str) -> tuple[Path, Path]:
    root = job_root / TILE_CHECKPOINT_COLLECTION_NAME / TILE_CHECKPOINT_VERSION
    return root / f"{tile_id}.zip", root / f"{tile_id}.json"


def _validate_tile_checkpoint_record(
    archive_path: Path,
    receipt_path: Path,
    *,
    tile_id: str,
    rehash_archive: bool,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, "reçu de checkpoint de tuile")
    supplied_hash = receipt.get("checkpoint_sha256")
    unsigned = dict(receipt)
    unsigned.pop("checkpoint_sha256", None)
    archive = receipt.get("archive")
    package_receipt = receipt.get("package_receipt")
    if (
        receipt.get("schema") != TILE_CHECKPOINT_SCHEMA
        or receipt.get("status") != "validated_compressed_checkpoint"
        or receipt.get("tile_id") != tile_id
        or supplied_hash != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
        or not isinstance(archive, Mapping)
        or archive.get("file") != archive_path.name
        or not isinstance(archive.get("byte_count"), int)
        or archive.get("byte_count") <= 0
        or not isinstance(archive.get("sha256"), str)
        or len(archive.get("sha256", "")) != 64
        or not isinstance(package_receipt, Mapping)
        or package_receipt.get("file") != "simple-measured-tile-receipt.v1.json"
        or not isinstance(package_receipt.get("byte_count"), int)
        or package_receipt.get("byte_count") <= 0
        or not isinstance(package_receipt.get("sha256"), str)
        or len(package_receipt.get("sha256", "")) != 64
        or not isinstance(receipt.get("file_count"), int)
        or receipt.get("file_count") <= 0
        or not isinstance(receipt.get("uncompressed_byte_count"), int)
        or receipt.get("uncompressed_byte_count") <= 0
        or not archive_path.is_file()
        or archive_path.stat().st_size != archive.get("byte_count")
    ):
        raise SimpleProductionError("Checkpoint de tuile incohérent")
    if rehash_archive and _sha256_file(archive_path) != archive["sha256"]:
        raise SimpleProductionError("Archive de checkpoint de tuile altérée")
    return receipt


def _copy_checkpoint_file_with_timeout(
    source: Path,
    destination: Path,
    *,
    timeout_seconds: float,
) -> None:
    """Copy one local checkpoint artifact without an unbounded volume write."""

    destination.unlink(missing_ok=True)
    try:
        if CHECKPOINT_COPY_USES_SUBPROCESS:
            subprocess.run(
                ["/bin/cp", "--", str(source), str(destination)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        else:
            shutil.copyfile(source, destination)
    except subprocess.TimeoutExpired as error:
        destination.unlink(missing_ok=True)
        raise SimpleProductionError(
            "Publication du checkpoint expirée sur le volume persistant"
        ) from error
    except (OSError, subprocess.CalledProcessError) as error:
        destination.unlink(missing_ok=True)
        raise SimpleProductionError(
            "Publication du checkpoint impossible sur le volume persistant"
        ) from error
    if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
        destination.unlink(missing_ok=True)
        raise SimpleProductionError("Copie du checkpoint incomplète")


def _write_tile_checkpoint(
    package_root: Path,
    job_root: Path,
    *,
    tile_id: str,
    local_staging_root: Path | None = None,
    publish_lock: threading.Lock | None = None,
    publish_timeout_seconds: float = CHECKPOINT_PUBLISH_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Build locally, then atomically publish one compressed tile checkpoint."""

    package = package_root.resolve(strict=True)
    archive_path, receipt_path = _tile_checkpoint_paths(job_root, tile_id)
    package_receipt_path = package / "simple-measured-tile-receipt.v1.json"
    if not package_receipt_path.is_file():
        raise SimpleProductionError("Reçu de tuile absent avant checkpoint")
    package_receipt_record = {
        "file": package_receipt_path.name,
        "byte_count": package_receipt_path.stat().st_size,
        "sha256": _sha256_file(package_receipt_path),
    }

    def existing_checkpoint() -> dict[str, Any] | None:
        if not (archive_path.exists() or receipt_path.exists()):
            return None
        if not archive_path.is_file() or not receipt_path.is_file():
            raise SimpleProductionError("Checkpoint de tuile partiellement publié")
        existing = _validate_tile_checkpoint_record(
            archive_path,
            receipt_path,
            tile_id=tile_id,
            rehash_archive=True,
        )
        if existing.get("package_receipt") != package_receipt_record:
            raise SimpleProductionError(
                "Un checkpoint différent existe déjà pour cette tuile"
            )
        return existing

    existing = existing_checkpoint()
    if existing is not None:
        return existing

    files: list[Path] = []
    for candidate in sorted(
        package.rglob("*"), key=lambda path: path.relative_to(package).as_posix()
    ):
        if candidate.is_symlink():
            raise SimpleProductionError("Lien symbolique interdit dans un checkpoint")
        if candidate.is_file():
            files.append(candidate)
    if not files:
        raise SimpleProductionError("Package de tuile vide avant checkpoint")
    staging_root = (
        package.parent / ".checkpoint-staging"
        if local_staging_root is None
        else local_staging_root
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    local_archive = staging_root / f".{tile_id}.zip.part"
    local_receipt = staging_root / f".{tile_id}.json.part"
    local_archive.unlink(missing_ok=True)
    local_receipt.unlink(missing_ok=True)
    uncompressed_bytes = 0
    try:
        with zipfile.ZipFile(
            local_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for source in files:
                relative = source.relative_to(package).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with (
                    source.open("rb") as input_stream,
                    archive.open(info, "w", force_zip64=True) as output_stream,
                ):
                    shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
                uncompressed_bytes += source.stat().st_size
        archive_record = {
            "file": archive_path.name,
            "byte_count": local_archive.stat().st_size,
            "sha256": _sha256_file(local_archive),
            "compression": "zip_deflate_level_1",
        }
        receipt: dict[str, Any] = {
            "schema": TILE_CHECKPOINT_SCHEMA,
            "status": "validated_compressed_checkpoint",
            "tile_id": tile_id,
            "archive": archive_record,
            "package_receipt": package_receipt_record,
            "file_count": len(files),
            "uncompressed_byte_count": uncompressed_bytes,
        }
        receipt["checkpoint_sha256"] = hashlib.sha256(
            _canonical_bytes(receipt)
        ).hexdigest()
        local_receipt.write_bytes(_canonical_bytes(receipt) + b"\n")

        def publish() -> dict[str, Any]:
            existing = existing_checkpoint()
            if existing is not None:
                return existing
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_archive = _unique_temporary_file(archive_path, label="checkpoint")
            temporary_receipt = _unique_temporary_file(receipt_path, label="checkpoint")
            try:
                _copy_checkpoint_file_with_timeout(
                    local_archive,
                    temporary_archive,
                    timeout_seconds=publish_timeout_seconds,
                )
                _copy_checkpoint_file_with_timeout(
                    local_receipt,
                    temporary_receipt,
                    timeout_seconds=publish_timeout_seconds,
                )
                os.replace(temporary_archive, archive_path)
                os.replace(temporary_receipt, receipt_path)
                return receipt
            except Exception:
                temporary_archive.unlink(missing_ok=True)
                temporary_receipt.unlink(missing_ok=True)
                raise

        if publish_lock is None:
            return publish()
        with publish_lock:
            return publish()
    finally:
        local_archive.unlink(missing_ok=True)
        local_receipt.unlink(missing_ok=True)


def _restore_tile_checkpoint(
    job_root: Path,
    package_root: Path,
    *,
    tile_id: str,
) -> dict[str, Any]:
    """Rehash and safely materialize one checkpoint onto the local SSD."""

    archive_path, receipt_path = _tile_checkpoint_paths(job_root, tile_id)
    record = _validate_tile_checkpoint_record(
        archive_path,
        receipt_path,
        tile_id=tile_id,
        rehash_archive=True,
    )
    destination = package_root.resolve()
    if destination.exists():
        raise SimpleProductionError("Destination locale du checkpoint déjà présente")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.checkpoint.part")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        seen: set[str] = set()
        total_bytes = 0
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise SimpleProductionError("CRC du checkpoint de tuile invalide")
            members = archive.infolist()
            for member in members:
                relative = PurePosixPath(member.filename)
                if (
                    member.is_dir()
                    or relative.is_absolute()
                    or not relative.parts
                    or ".." in relative.parts
                    or "\\" in member.filename
                    or member.filename in seen
                    or member.external_attr >> 16 not in {0, 0o100644}
                ):
                    raise SimpleProductionError("Membre de checkpoint non sûr")
                seen.add(member.filename)
                total_bytes += member.file_size
                target = staging.joinpath(*relative.parts)
                _inside(staging, target, "extraction du checkpoint")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as input_stream, target.open("wb") as output:
                    shutil.copyfileobj(input_stream, output, 1024 * 1024)
        if (
            len(seen) != record["file_count"]
            or total_bytes != record["uncompressed_byte_count"]
        ):
            raise SimpleProductionError("Inventaire du checkpoint de tuile divergent")
        package_receipt = staging / record["package_receipt"]["file"]
        if (
            not package_receipt.is_file()
            or package_receipt.stat().st_size != record["package_receipt"]["byte_count"]
            or _sha256_file(package_receipt) != record["package_receipt"]["sha256"]
        ):
            raise SimpleProductionError("Sceau du package restauré divergent")
        os.replace(staging, destination)
        return record
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _quarantine_tile_checkpoint(job_root: Path, *, tile_id: str) -> list[str]:
    """Preserve a corrupt/partial checkpoint while freeing its active names."""

    archive_path, receipt_path = _tile_checkpoint_paths(job_root, tile_id)
    candidates = [path for path in (archive_path, receipt_path) if path.exists()]
    if not candidates:
        return []
    quarantine = (
        job_root
        / TILE_CHECKPOINT_COLLECTION_NAME
        / "quarantine"
        / tile_id
        / str(time.time_ns())
    )
    quarantine.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for source in candidates:
        if source.is_symlink() or not source.is_file():
            raise SimpleProductionError("Checkpoint corrompu non régulier refusé")
        target = quarantine / source.name
        os.replace(source, target)
        moved.append(target.relative_to(job_root).as_posix())
    return moved


def _prototype_bundle_builder_sha256() -> str:
    """Hash the code that renders and validates prototype bundle artifacts."""

    builder = (
        Path(__file__).resolve().parent.parent
        / "omniverse"
        / "build_measured_scene_usd.py"
    )
    if not builder.is_file():
        raise SimpleProductionError(
            "Le générateur causal du lot de prototypes est absent"
        )
    return _sha256_file(builder)


def _prototype_bundle_namespace(asset_summary: Mapping[str, Any]) -> str:
    """Bind a shared bundle to its immutable catalog and producing code."""

    catalog_sha256 = asset_summary.get("catalog_sha256")
    if (
        not isinstance(catalog_sha256, str)
        or len(catalog_sha256) != 64
        or any(character not in "0123456789abcdef" for character in catalog_sha256)
    ):
        raise SimpleProductionError(
            "L'identité du catalogue ne permet pas de versionner les prototypes"
        )
    identity = {
        "schema": PROTOTYPE_BUNDLE_NAMESPACE_SCHEMA,
        "catalog_sha256": catalog_sha256,
        "builder_sha256": _prototype_bundle_builder_sha256(),
    }
    return "v1-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _prototype_bundle_root(job_root: Path, asset_summary: Mapping[str, Any]) -> Path:
    """Return the causal, restart-stable bundle root for this production."""

    return (
        job_root
        / "shared"
        / PROTOTYPE_BUNDLE_COLLECTION_NAME
        / _prototype_bundle_namespace(asset_summary)
    )


def _regular_file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def _sync_prototype_bundle(
    source_root: Path,
    destination_root: Path,
    *,
    validated_files: dict[
        str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
    ]
    | None = None,
) -> None:
    """Copy only the small immutable link index, preserving embedded-asset links."""

    source = source_root.resolve()
    destination = destination_root.resolve()
    if not source.exists():
        destination.mkdir(parents=True, exist_ok=True)
        return
    if not source.is_dir() or source.is_symlink():
        raise SimpleProductionError("Lot de prototypes source invalide")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise SimpleProductionError("Lot de prototypes destination symbolique refusé")
    for item in sorted(
        source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()
    ):
        relative = item.relative_to(source)
        if any(
            part.startswith(".") and part.endswith(".part") for part in relative.parts
        ):
            continue
        target = destination / relative
        _inside(destination, target, "synchronisation des prototypes")
        if item.is_dir() and not item.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.is_symlink():
            link_target = os.readlink(item)
            if target.is_symlink():
                if os.readlink(target) != link_target:
                    raise SimpleProductionError("Lien de prototype immuable divergent")
                continue
            if target.exists():
                raise SimpleProductionError("Prototype immuable de type divergent")
            target.symlink_to(link_target)
            continue
        if not item.is_file():
            raise SimpleProductionError("Artefact de prototype non régulier")
        if target.exists():
            cache_key = relative.as_posix()
            source_signature = _regular_file_signature(item)
            target_signature = _regular_file_signature(target)
            if validated_files is not None and validated_files.get(cache_key) == (
                source_signature,
                target_signature,
            ):
                continue
            if (
                not target.is_file()
                or target.is_symlink()
                or target_signature[0] != source_signature[0]
                or _sha256_file(target) != _sha256_file(item)
            ):
                raise SimpleProductionError("Artefact de prototype immuable divergent")
            if validated_files is not None:
                validated_files[cache_key] = (source_signature, target_signature)
            continue
        temporary = _unique_temporary_file(target, label="sync")
        try:
            shutil.copyfile(item, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        if validated_files is not None:
            validated_files[relative.as_posix()] = (
                _regular_file_signature(item),
                _regular_file_signature(target),
            )


def _contract_path() -> Path:
    return Path(__file__).with_name("simple_production_engine_contract.v1.json")


def _load_contract() -> dict[str, Any]:
    payload = _load_json(_contract_path(), "contrat du moteur de production")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "locked"
        or payload.get("production", {}).get("tile_size_m") != TILE_SIZE_M
        or payload.get("production", {}).get("mode") != "bounded_parallel"
        or payload.get("production", {}).get("default_parallel_tiles") != 6
        or payload.get("production", {}).get("max_parallel_tiles") != 8
        or payload.get("production", {}).get("default_parallel_source_acquisitions")
        != 12
        or payload.get("production", {}).get("max_parallel_source_acquisitions") != 16
        or payload.get("production", {}).get("tile_inactivity_timeout_seconds")
        != DEFAULT_TILE_STALL_TIMEOUT_SECONDS
        or payload.get("production", {}).get("source_metatile_tiles") != 4
        or payload.get("production", {}).get("resume")
        != "restore_validated_compressed_tile_checkpoints_and_remove_owned_staging"
        or payload.get("production", {}).get("checkpoint_build")
        != "compress_and_hash_on_worker_local_ssd_then_single_file_copy_to_persistent_volume"
        or payload.get("production", {}).get("checkpoint_publish")
        != "serialized_atomic_copy_with_180_second_timeout"
        or payload.get("production", {}).get("wms_deadlines")
        != "one_bounded_metatile_attempt_then_three_bounded_individual_tile_attempts"
        or payload.get("output", {}).get("entry_stage") != ENTRY_STAGE
        or payload.get("output", {}).get("portable_zip")
        != "built_on_worker_local_disk_then_uploaded_once"
        or payload.get("perimeter_layer", {}).get("fixed_layers") is not True
        or payload.get("perimeter_layer", {}).get("timeline_data")
        != "observed_instants_and_explicit_ranges"
        or payload.get("perimeter_layer", {}).get("prediction") != "forbidden"
        or payload.get("perimeter_layer", {}).get("viewer")
        != "one_derived_glb_per_observed_instant_or_range"
        or payload.get("perimeter_layer", {}).get("viewer_authoritative") is not False
        or payload.get("fixed_asset_placement", {}).get("request_schema")
        != "fireviewer.fixed-asset-placement-request.v1"
        or payload.get("fixed_asset_placement", {}).get("z")
        != "sampled_from_tile_MNT_never_user_authored"
        or payload.get("fixed_asset_placement", {}).get("automatic_catalog_selection")
        != "bypassed_for_fixed_asset_id"
        or payload.get("acceptance", {}).get("automatic_human_acceptance") is not False
    ):
        raise SimpleProductionError("Contrat moteur absent, modifié ou non verrouillé")
    return payload


def _absolute(path: Path | str, label: str, *, exists: bool) -> Path:
    lexical = PureWindowsPath(str(path))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise SimpleProductionError(f"{label} doit rester sur D: sous Windows")
    try:
        resolved = Path(path).resolve(strict=exists)
    except OSError as error:
        raise SimpleProductionError(f"{label} introuvable: {path}") from error
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise SimpleProductionError(f"{label} doit rester sur D: sous Windows")
    if exists and not resolved.exists():
        raise SimpleProductionError(f"{label} introuvable: {resolved}")
    return resolved


def _inside(root: Path, child: Path, label: str) -> Path:
    try:
        child.relative_to(root)
    except ValueError as error:
        raise SimpleProductionError(f"{label} sort du volume portable") from error
    return child


def validate_embedded_assets(config: ProductionConfig) -> dict[str, Any]:
    """Rehash every resolved real USD/texture before serving the UI."""

    portable = _absolute(config.portable_root, "racine portable", exists=True)
    library_path = _inside(
        portable,
        _absolute(config.asset_library, "catalogue des assets", exists=True),
        "catalogue des assets",
    )
    review_batch = _inside(
        portable,
        _absolute(config.review_batch, "lot des assets", exists=True),
        "lot des assets",
    )
    library = _load_json(library_path, "catalogue des assets")
    if library.get("schema") == REFERENCE_ASSET_LIBRARY_SCHEMA:
        summary = validate_reference_asset_library(library)
        reference_count = (
            library.get("reference_manifest", {})
            .get("route_counts", {})
            .get("hunyuan3d")
        )
        if summary.get("asset_count") != reference_count:
            raise SimpleProductionError(
                "Le catalogue USD ne couvre pas toutes les références 3D"
            )
    else:
        summary = validate_asset_library(library)
        if (
            summary.get("asset_count") != 53
            or summary.get("category_counts", {}).get("building") != 24
            or summary.get("category_counts", {}).get("tree") != 18
        ):
            raise SimpleProductionError(
                "Le catalogue embarqué historique n'est pas le lot pilote 53/53"
            )
    checked = 0
    for asset in library["assets"]:
        for role in ("usd", "texture"):
            record = asset.get(role)
            if not isinstance(record, Mapping) or record.get("root") != "review_batch":
                raise SimpleProductionError(
                    f"Asset embarqué invalide: {asset.get('asset_id')}.{role}"
                )
            relative = record.get("path")
            if (
                not isinstance(relative, str)
                or not relative
                or "\\" in relative
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
            ):
                raise SimpleProductionError("Chemin d'asset embarqué invalide")
            target = review_batch.joinpath(*PurePosixPath(relative).parts).resolve()
            _inside(review_batch, target, "asset embarqué")
            expected_sha256 = record.get("sha256")
            actual_sha256 = _sha256_file(target) if target.is_file() else None
            if (
                not target.is_file()
                or target.stat().st_size != record.get("byte_count")
                or actual_sha256 != expected_sha256
            ):
                raise SimpleProductionError(
                    f"Asset embarqué absent ou altéré: {asset['asset_id']}.{role}"
                )
            remember_validated_file_hash(target, actual_sha256)
            checked += 1
    return {
        "asset_count": summary["asset_count"],
        "building_assets": summary["category_counts"].get("building", 0),
        "tree_assets": summary["category_counts"].get("tree", 0),
        "real_usd_assets": summary.get("availability_counts", {}).get(
            "real_usd", summary["asset_count"]
        ),
        "placeholder_usd_assets": summary.get("availability_counts", {}).get(
            "placeholder_usd", 0
        ),
        "fallback_usd_assets": summary.get("fallback_asset_count", 0),
        "unique_real_usd_assets": summary.get(
            "unique_real_usd_count", summary["asset_count"]
        ),
        "checked_artifacts": checked,
        "catalog_revision": library["catalog_revision"],
        "catalog_sha256": _sha256_file(library_path),
    }


def validate_embedded_runtime(config: ProductionConfig) -> dict[str, Any]:
    """Prove that Blender, OpenUSD and all production entrypoints are embedded."""

    blender = _absolute(config.blender, "Blender embarqué", exists=True)
    if not blender.is_file():
        raise SimpleProductionError("Blender embarqué est absent")
    required = (
        Path(__file__),
        Path(__file__).with_name("prepare_simple_measured_zone_context.py"),
        Path(__file__).with_name("prepare_simple_measured_tile_sources.py"),
        Path(__file__).with_name("elevation_nodata.py"),
        Path(__file__).with_name("produce_simple_measured_tile.py"),
        Path(__file__).with_name("run_simple_measured_tile_qa.py"),
        Path(__file__).with_name("render_simple_zone_gallery.py"),
        Path(__file__).with_name("build_reference_usd_asset_library.py"),
        Path(__file__).with_name("geographic_perimeter_layer.py"),
        Path(__file__).with_name("geographic_perimeter_layer_contract.v1.json"),
        Path(__file__).with_name("geographic_perimeter_viewer.py"),
        Path(__file__).with_name("portable_scene_package.py"),
        Path(__file__).with_name("fixed_asset_placement.py"),
        fixed_asset_schema_path(),
        fixed_asset_template_path(),
        Path(__file__).parent.parent / "omniverse" / "fixed_terrain_usd.py",
        Path(__file__).parent.parent / "omniverse" / "build_measured_scene_usd.py",
    )
    missing = [str(path) for path in required if not path.resolve().is_file()]
    if missing:
        raise SimpleProductionError(
            "Scripts de production embarqués absents: " + ", ".join(missing)
        )
    expression = (
        "import bpy; from pxr import Usd,UsdGeom,UsdShade; "
        "print('FIREVIEWER_BLENDER_USD=PASS')"
    )
    try:
        result = subprocess.run(
            (
                str(blender),
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-expr",
                expression,
            ),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SimpleProductionError(
            f"Blender/OpenUSD embarqué ne démarre pas: {error}"
        ) from error
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or "FIREVIEWER_BLENDER_USD=PASS" not in output:
        raise SimpleProductionError(
            "Le smoke embarqué Blender/OpenUSD a échoué: " + output[-1000:]
        )
    version_line = next(
        (line.strip() for line in output.splitlines() if line.startswith("Blender ")),
        "",
    )
    if not version_line.startswith("Blender 4.5"):
        raise SimpleProductionError(
            f"Blender embarqué doit être en version 4.5; reçu {version_line!r}"
        )
    return {
        "blender_path": str(blender),
        "blender_version": version_line.removeprefix("Blender "),
        "blender_sha256": _sha256_file(blender),
        "openusd": "bundled_with_blender_passed",
        "production_script_count": len(required),
    }


def validate_config(config: ProductionConfig) -> dict[str, Any]:
    """Validate the headless engine contract and shared production runtime."""

    _load_contract()
    return validate_production_config(config)


def validate_production_config(config: ProductionConfig) -> dict[str, Any]:
    """Validate only the headless production runtime shared by every frontend."""

    if (
        config.max_side_m < TILE_SIZE_M
        or config.max_tiles < 1
        or not 1 <= config.tile_workers <= 8
        or not config.tile_workers <= config.source_workers <= 16
        or not 120 <= config.tile_stall_timeout_seconds <= 3_600
    ):
        raise SimpleProductionError("Limites du pod invalides")
    portable = _absolute(config.portable_root, "racine portable", exists=True)
    work = _absolute(config.work_root, "volume de travail", exists=False)
    _inside(portable, work, "volume de travail")
    work.mkdir(parents=True, exist_ok=True)
    if config.scratch_root is not None:
        scratch = _absolute(config.scratch_root, "volume local rapide", exists=False)
        _inside(portable, scratch, "volume local rapide")
        scratch.mkdir(parents=True, exist_ok=True)
    for label, revision in (
        ("révision élévation", config.elevation_revision),
        ("révision orthophoto", config.orthophoto_revision),
        ("révision contexte", config.context_revision),
    ):
        if not revision or revision != revision.strip():
            raise SimpleProductionError(f"{label} invalide")
    return {
        **validate_embedded_assets(config),
        **validate_embedded_runtime(config),
    }


def plan_zone(
    latitude: float,
    longitude: float,
    side_km: float,
    *,
    max_side_m: int = 15_000,
    max_tiles: int = 900,
    transformer: Transformer | None = None,
) -> ZonePlan:
    """Transform the GPS centre and cover the requested square on the 500 m grid."""

    values = (float(latitude), float(longitude), float(side_km))
    if any(not math.isfinite(value) for value in values):
        raise SimpleProductionError("Latitude, longitude et taille doivent être finies")
    latitude, longitude, side_km = values
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise SimpleProductionError("Coordonnées GPS invalides")
    side_m_float = side_km * 1_000.0
    if side_m_float < TILE_SIZE_M or side_m_float > max_side_m:
        raise SimpleProductionError(
            f"La taille doit être comprise entre 0,5 et {max_side_m / 1000:g} km"
        )
    if not side_m_float.is_integer():
        raise SimpleProductionError(
            "La taille doit correspondre à un nombre entier de mètres"
        )
    side_m = int(side_m_float)
    convert = transformer or Transformer.from_crs(4326, 2154, always_xy=True)
    center_x, center_y = convert.transform(longitude, latitude)
    if (
        not math.isfinite(center_x)
        or not math.isfinite(center_y)
        or not -100_000 <= center_x <= 1_500_000
        or not 5_900_000 <= center_y <= 7_300_000
    ):
        raise SimpleProductionError(
            "Le centre doit être couvert par la projection Lambert-93"
        )
    half = side_m / 2.0
    requested = (
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
    )

    def grid_floor(value: float) -> int:
        nearest = round(value / TILE_SIZE_M) * TILE_SIZE_M
        if abs(value - nearest) <= 0.001:
            return int(nearest)
        return math.floor(value / TILE_SIZE_M) * TILE_SIZE_M

    def grid_ceil(value: float) -> int:
        nearest = round(value / TILE_SIZE_M) * TILE_SIZE_M
        if abs(value - nearest) <= 0.001:
            return int(nearest)
        return math.ceil(value / TILE_SIZE_M) * TILE_SIZE_M

    west = grid_floor(requested[0])
    south = grid_floor(requested[1])
    east = grid_ceil(requested[2])
    north = grid_ceil(requested[3])
    origins = tuple(
        (x, y)
        for y in range(north - TILE_SIZE_M, south - 1, -TILE_SIZE_M)
        for x in range(west, east, TILE_SIZE_M)
    )
    if not origins or len(origins) > max_tiles:
        raise SimpleProductionError(
            f"La zone demande {len(origins)} tuiles; limite du pod: {max_tiles}"
        )
    identity = {
        "latitude": round(latitude, 10),
        "longitude": round(longitude, 10),
        "side_m": side_m,
        "production_bounds_l93_m": [west, south, east, north],
    }
    zone_id = (
        "GPS-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:16].upper()
    )
    tiles = tuple(TilePlan(f"x{x}_y{y}", (x, y)) for x, y in origins)
    return ZonePlan(
        zone_id=zone_id,
        latitude=latitude,
        longitude=longitude,
        side_m=side_m,
        center_l93_m=(center_x, center_y),
        requested_bounds_l93_m=requested,
        production_bounds_l93_m=(west, south, east, north),
        context_bounds_l93_m=(
            west - HALO_M,
            south - HALO_M,
            east + HALO_M,
            north + HALO_M,
        ),
        tiles=tiles,
    )


def _plan_payload(
    plan: ZonePlan,
    config: ProductionConfig,
    *,
    fixed_asset_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "planned",
        "zone_id": plan.zone_id,
        "input": {
            "latitude": plan.latitude,
            "longitude": plan.longitude,
            "side_m": plan.side_m,
            "coordinate_role": "square_center",
        },
        "crs": "EPSG:2154",
        "center_l93_m": list(plan.center_l93_m),
        "requested_bounds_l93_m": list(plan.requested_bounds_l93_m),
        "production_bounds_l93_m": list(plan.production_bounds_l93_m),
        "context_bounds_l93_m": list(plan.context_bounds_l93_m),
        "tile_size_m": TILE_SIZE_M,
        "tile_count": len(plan.tiles),
        "tiles": [
            {"tile_id": tile.tile_id, "origin_l93_m": list(tile.origin_l93_m)}
            for tile in plan.tiles
        ],
        "source_revisions": {
            "elevation": config.elevation_revision,
            "orthophoto": config.orthophoto_revision,
            "context": config.context_revision,
        },
    }
    if fixed_asset_request is not None:
        payload["fixed_asset_placements"] = {
            "path": FIXED_ASSET_REQUEST_NAME,
            "schema": fixed_asset_request["schema"],
            "count": len(fixed_asset_request["placements"]),
            "sha256": hashlib.sha256(
                _canonical_bytes(fixed_asset_request) + b"\n"
            ).hexdigest(),
            "content_sha256": fixed_asset_request_sha256(fixed_asset_request),
        }
    payload["plan_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _status(
    root: Path,
    *,
    state: str,
    message: str,
    completed: int,
    total: int,
    error: str | None = None,
    phase: str | None = None,
    details: Mapping[str, Any] | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": "fireviewer.simple-production-job-status.v1",
        "state": state,
        "message": message,
        "completed_tiles": completed,
        "total_tiles": total,
    }
    if phase is not None:
        payload["phase"] = phase
    if details is not None:
        payload["details"] = dict(details)
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = round(elapsed_seconds, 3)
    if error is not None:
        payload["error"] = error
    _write_json(root / STATUS_NAME, payload)


def _copy_provenance(sources: PreparedSources, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    names = (
        "elevation-source-05m.json",
        "orthophoto-source.json",
        "building-source.json",
        "bdtopo-buildings.geojson",
        "placement-context.json",
        "simple-measured-tile-sources.v1.json",
    )
    for name in names:
        source = sources.root / name
        if not source.is_file():
            raise SimpleProductionError(f"Provenance de tuile absente: {name}")
        shutil.copyfile(source, target / name)


def _remove_sources(job_root: Path, source_root: Path) -> None:
    expected_root = (job_root / "sources").resolve()
    resolved = source_root.resolve(strict=True)
    try:
        resolved.relative_to(expected_root)
    except ValueError as error:
        raise SimpleProductionError("Nettoyage source hors du job refusé") from error
    if resolved == expected_root:
        raise SimpleProductionError("Nettoyage global des sources refusé")
    shutil.rmtree(resolved)


def _remove_interrupted_staging(job_root: Path) -> list[str]:
    """Remove only engine-owned staging paths left by an interrupted invocation."""

    roots_and_patterns = [
        (job_root / "sources", ".*.simple-sources.part"),
        (job_root / "packages", ".*.simple-measured-tile.part"),
        (
            job_root / TILE_CHECKPOINT_COLLECTION_NAME / TILE_CHECKPOINT_VERSION,
            ".*.part",
        ),
    ]
    # Prototype staging directories are uniquely owned and may belong to a
    # concurrent retry.  Their publisher removes its own staging in ``finally``;
    # sweeping them here could delete an active asset from another process.
    removed: list[str] = []
    resolved_job = job_root.resolve(strict=True)
    for root, pattern in roots_and_patterns:
        if not root.is_dir():
            continue
        resolved_root = root.resolve(strict=True)
        try:
            resolved_root.relative_to(resolved_job)
        except ValueError as error:  # pragma: no cover - invariant guard
            raise SimpleProductionError("Racine de staging hors du job") from error
        for staging in sorted(root.glob(pattern)):
            if staging.is_symlink():
                raise SimpleProductionError(
                    f"Staging interrompu symbolique refusé: {staging.name}"
                )
            resolved = staging.resolve(strict=True)
            if resolved.parent != resolved_root:
                raise SimpleProductionError(
                    f"Staging interrompu hors de sa racine: {staging.name}"
                )
            relative = resolved.relative_to(resolved_job).as_posix()
            if staging.is_dir():
                shutil.rmtree(staging)
            elif staging.is_file():
                staging.unlink()
            else:  # pragma: no cover - unusual filesystem entry
                raise SimpleProductionError(
                    f"Type de staging interrompu refusé: {staging.name}"
                )
            removed.append(relative)
    return removed


def _bounded_parallel_tile_results(
    pending_tiles: list[tuple[int, TilePlan]],
    *,
    worker_count: int,
    process_tile: Callable[[int, TilePlan], str],
    stop_event: threading.Event,
    heartbeat_seconds: float = PARALLEL_HEARTBEAT_SECONDS,
    last_activity: Callable[[int, TilePlan], float] | None = None,
    stall_timeout_seconds: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Iterator[str | None]:
    """Run a rolling bounded queue and fail closed on an inactive tile."""

    if not 2 <= worker_count <= 16 or worker_count > len(pending_tiles):
        raise SimpleProductionError("Configuration de production parallèle invalide")
    pending_iterator = iter(pending_tiles)
    executor = ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="fireviewer-tile"
    )
    futures: dict[Future[str], tuple[int, TilePlan]] = {}

    def submit_next() -> bool:
        try:
            tile_index, tile = next(pending_iterator)
        except StopIteration:
            return False
        futures[executor.submit(process_tile, tile_index, tile)] = (tile_index, tile)
        return True

    for _ in range(worker_count):
        submit_next()
    succeeded = False
    try:
        while futures:
            done, _not_done = wait(
                futures,
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if last_activity is not None and stall_timeout_seconds is not None:
                    now = monotonic()
                    stalled = [
                        (tile_index, tile, now - last_activity(tile_index, tile))
                        for tile_index, tile in futures.values()
                        if now - last_activity(tile_index, tile)
                        >= stall_timeout_seconds
                    ]
                    if stalled:
                        tile_index, tile, inactive_seconds = max(
                            stalled, key=lambda item: item[2]
                        )
                        stop_event.set()
                        for future in futures:
                            future.cancel()
                        raise SimpleProductionError(
                            f"Tuile {tile.tile_id} inactive depuis "
                            f"{int(inactive_seconds)} s; reprise requise"
                        )
                yield None
                continue
            failed = next(
                (
                    future
                    for future in done
                    if not future.cancelled() and future.exception() is not None
                ),
                None,
            )
            if failed is not None:
                stop_event.set()
                for queued in futures:
                    if queued is not failed:
                        queued.cancel()
                failed.result()
            for future in done:
                futures.pop(future)
                yield future.result()
                submit_next()
        succeeded = True
    finally:
        if not succeeded:
            stop_event.set()
            for future in futures:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _interleave_tiles_by_metatile(
    pending_tiles: Sequence[tuple[int, TilePlan]],
) -> list[tuple[int, TilePlan]]:
    """Start distinct 4x4 source blocks first instead of parking workers on one."""

    groups: dict[tuple[int, int], list[tuple[int, TilePlan]]] = {}
    for item in pending_tiles:
        x, y = item[1].origin_l93_m
        key = (
            (x // SOURCE_METATILE_SIZE_M) * SOURCE_METATILE_SIZE_M,
            (y // SOURCE_METATILE_SIZE_M) * SOURCE_METATILE_SIZE_M,
        )
        groups.setdefault(key, []).append(item)
    ordered: list[tuple[int, TilePlan]] = []
    offset = 0
    while True:
        added = False
        for values in groups.values():
            if offset < len(values):
                ordered.append(values[offset])
                added = True
        if not added:
            return ordered
        offset += 1


def _scene_counts(package_root: Path) -> tuple[int, int, int, int]:
    receipt = _load_json(package_root / "scene" / "scene.done.json", "reçu scène")
    reconciliation = receipt.get("reconciliation")
    if not isinstance(reconciliation, Mapping):
        raise SimpleProductionError("Reçu scène sans réconciliation")
    values: list[int] = []
    for family in ("buildings", "trees", "context_assets"):
        record = reconciliation.get(family)
        count = (
            record.get("instance_count")
            if isinstance(record, Mapping)
            else 0
            if family == "context_assets"
            else None
        )
        if not isinstance(count, int) or count < 0:
            raise SimpleProductionError(f"Comptage {family} invalide")
        values.append(count)
    placeholder_count = receipt.get("placeholder_instance_count", 0)
    if not isinstance(placeholder_count, int) or placeholder_count < 0:
        raise SimpleProductionError("Comptage des placeholders invalide")
    return values[0], values[1], values[2], placeholder_count


def _expected_request_from_receipt(
    package_root: Path,
    *,
    plan: ZonePlan,
    tile: TilePlan,
    asset_library: Path,
    asset_roots: Mapping[str, Path],
    portable_root: Path,
    asset_bundle_root: Path,
    asset_library_sha256: str | None = None,
) -> dict[str, Any]:
    """Recover the signed request while binding it to the current zone request."""

    receipt = _load_json(
        package_root / "simple-measured-tile-receipt.v1.json", "reçu de tuile"
    )
    request = receipt.get("request")
    if not isinstance(request, Mapping):
        raise SimpleProductionError("Reçu de tuile sans requête scellée")
    expected_fixed = {
        "schema": "fireviewer.simple-measured-tile-request.v1",
        "algorithm": "fireviewer.simple-measured-tile-algorithm.v1",
        "zone_id": plan.zone_id,
        "tile_id": tile.tile_id,
        "crs": "EPSG:2154",
        "tile_origin_l93_m": list(tile.origin_l93_m),
        "core_bounds_l93_m": [
            tile.origin_l93_m[0],
            tile.origin_l93_m[1],
            tile.origin_l93_m[0] + TILE_SIZE_M,
            tile.origin_l93_m[1] + TILE_SIZE_M,
        ],
        "asset_library_sha256": (
            asset_library_sha256
            if asset_library_sha256 is not None
            else _sha256_file(asset_library)
        ),
        "asset_root_names": sorted(asset_roots),
        "usage": "technical_pilot_non_final",
        "mns_fallback_policy": "ground_only_on_hag_validation_failure",
    }
    changed = [
        key for key, value in expected_fixed.items() if request.get(key) != value
    ]
    bundle = request.get("prototype_bundle")
    portable_path = bundle.get("portable_path") if isinstance(bundle, Mapping) else None
    expected_bundle_suffix = (
        "jobs",
        plan.zone_id,
        "shared",
        PROTOTYPE_BUNDLE_COLLECTION_NAME,
        asset_bundle_root.name,
    )
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("scope") != "explicit_shared"
        or not isinstance(portable_path, str)
        or PurePosixPath(portable_path).is_absolute()
        or ".." in PurePosixPath(portable_path).parts
        or tuple(PurePosixPath(portable_path).parts[-5:]) != expected_bundle_suffix
    ):
        changed.append("prototype_bundle")
    if changed:
        raise SimpleProductionError(
            "La requête scellée de la tuile diffère du job courant: "
            + ", ".join(changed)
        )
    if not isinstance(request.get("sources"), Mapping) or not isinstance(
        request.get("pipeline_files"), Mapping
    ):
        raise SimpleProductionError("Requête scellée de tuile incomplète")
    return dict(request)


def _write_zone_stage(job_root: Path, plan: ZonePlan) -> Path:
    west, south, _east, _north = plan.production_bounds_l93_m
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "FireViewerZone"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "FireViewerZone"',
        "{",
        f'    custom string fireviewer:zone_id = "{plan.zone_id}"',
        f"    custom int fireviewer:tile_count = {len(plan.tiles)}",
    ]
    for tile in plan.tiles:
        x, y = tile.origin_l93_m
        reference = f"packages/{tile.tile_id}/scene/scene.usda"
        lines.extend(
            [
                "",
                f'    def Xform "{tile.tile_id}" (',
                f"        prepend references = @{reference}@",
                "    )",
                "    {",
                f"        double3 xformOp:translate = ({x - west}, {y - south}, 0)",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
            ]
        )
    lines.extend(["}", ""])
    destination = job_root / ENTRY_STAGE
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return destination


def _write_zone_receipt(
    job_root: Path,
    plan: ZonePlan,
    asset_summary: Mapping[str, Any],
    portable_root: Path,
    shared_bundle: Path,
    validated_checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    del portable_root, shared_bundle
    tile_records: list[dict[str, Any]] = []
    buildings = 0
    trees = 0
    context_assets = 0
    placeholders = 0
    degraded_mns_tiles = 0
    for tile in plan.tiles:
        package_root = job_root / "packages" / tile.tile_id
        receipt_path = package_root / "simple-measured-tile-receipt.v1.json"
        receipt = _load_json(receipt_path, "reçu de tuile")
        checkpoint = validated_checkpoints.get(tile.tile_id)
        checkpoint_package = (
            checkpoint.get("package_receipt")
            if isinstance(checkpoint, Mapping)
            else None
        )
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("schema") != TILE_CHECKPOINT_SCHEMA
            or not isinstance(checkpoint_package, Mapping)
            or checkpoint_package.get("byte_count") != receipt_path.stat().st_size
            or checkpoint_package.get("sha256") != _sha256_file(receipt_path)
        ):
            raise SimpleProductionError(
                f"Checkpoint validé absent ou divergent: {tile.tile_id}"
            )
        (
            tile_buildings,
            tile_trees,
            tile_context_assets,
            tile_placeholders,
        ) = _scene_counts(package_root)
        buildings += tile_buildings
        trees += tile_trees
        context_assets += tile_context_assets
        placeholders += tile_placeholders
        placement_source = receipt.get("placement", {}).get("source", {})
        degraded = bool(
            isinstance(placement_source, Mapping)
            and placement_source.get("mode") == "degraded_mns_fallback"
        )
        degraded_mns_tiles += int(degraded)
        tile_records.append(
            {
                "tile_id": tile.tile_id,
                "origin_l93_m": list(tile.origin_l93_m),
                "build_id": receipt["build_id"],
                "receipt_sha256": _sha256_file(receipt_path),
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_archive_sha256": checkpoint["archive"]["sha256"],
                "building_count": tile_buildings,
                "tree_count": tile_trees,
                "context_asset_count": tile_context_assets,
                "placeholder_instance_count": tile_placeholders,
                "degraded_mns_fallback": degraded,
            }
        )
    context_path = job_root / ZONE_CONTEXT_NAME
    plan_path = job_root / PLAN_NAME
    stage_path = job_root / ENTRY_STAGE
    payload: dict[str, Any] = {
        "schema": ZONE_RECEIPT_SCHEMA,
        "status": "technical_scene_produced",
        "accepted_human": False,
        "automatic_acceptance": False,
        "zone_id": plan.zone_id,
        "entry_stage": {
            "path": ENTRY_STAGE,
            "byte_count": stage_path.stat().st_size,
            "sha256": _sha256_file(stage_path),
        },
        "plan": {"path": PLAN_NAME, "sha256": _sha256_file(plan_path)},
        "context": {
            "path": ZONE_CONTEXT_NAME,
            "sha256": _sha256_file(context_path),
        },
        "asset_library": {
            "asset_count": asset_summary["asset_count"],
            "catalog_revision": asset_summary["catalog_revision"],
            "catalog_sha256": asset_summary["catalog_sha256"],
            "embedded_artifacts_checked": asset_summary["checked_artifacts"],
            "real_usd_assets": asset_summary.get(
                "real_usd_assets", asset_summary["asset_count"]
            ),
            "placeholder_usd_assets": asset_summary.get("placeholder_usd_assets", 0),
        },
        "tile_count": len(tile_records),
        "building_count": buildings,
        "tree_count": trees,
        "context_asset_count": context_assets,
        "placeholder_instance_count": placeholders,
        "degraded_mns_tile_count": degraded_mns_tiles,
        "tiles": tile_records,
    }
    fixed_request_path = job_root / FIXED_ASSET_REQUEST_NAME
    if fixed_request_path.is_file():
        fixed_request = _load_json(fixed_request_path, "placements fixes")
        payload["fixed_asset_placements"] = {
            "path": FIXED_ASSET_REQUEST_NAME,
            "count": len(fixed_request.get("placements", [])),
            "sha256": _sha256_file(fixed_request_path),
            "content_sha256": fixed_asset_request_sha256(fixed_request),
        }
    payload["build_id"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    _write_json(job_root / ZONE_RECEIPT_NAME, payload)
    return payload


def _scratch_job_path(config: ProductionConfig, zone_id: str) -> Path | None:
    if config.scratch_root is None:
        return None
    scratch = _absolute(config.scratch_root, "volume local rapide", exists=False)
    destination = scratch / "jobs" / zone_id
    _inside(scratch, destination, "staging local du job")
    return destination


def _prepare_scratch_job(config: ProductionConfig, zone_id: str) -> Path | None:
    canonical = _scratch_job_path(config, zone_id)
    if canonical is None:
        return None
    canonical.parent.mkdir(parents=True, exist_ok=True)
    # Scratch is disposable and must never be shared by overlapping retries of
    # the same deterministic zone.  Checkpoints remain under work_root; each
    # invocation gets an isolated SSD directory and cannot delete another run.
    return Path(tempfile.mkdtemp(prefix=f"{zone_id}-", dir=canonical.parent))


def _prepare_local_assembly(
    job_root: Path,
    scratch_job: Path,
    *,
    active_prototype_bundle: Path | None = None,
) -> Path:
    """Mirror checkpoints and only the active causal bundle to local scratch."""

    job = job_root.resolve(strict=True)
    scratch = scratch_job.resolve(strict=True)
    active_bundle_relative: Path | None = None
    if active_prototype_bundle is not None:
        active_bundle = active_prototype_bundle.resolve()
        _inside(job, active_bundle, "lot de prototypes actif")
        active_bundle_relative = active_bundle.relative_to(job)
        if active_bundle_relative.parts[:2] != (
            "shared",
            PROTOTYPE_BUNDLE_COLLECTION_NAME,
        ):
            raise SimpleProductionError("Namespace du lot de prototypes actif invalide")
    # Tile packages already live on the local SSD.  Network checkpoints are
    # recovery artifacts and must never be copied into the downloadable pack.
    excluded_roots = {
        "sources",
        "download",
        "packages",
        TILE_CHECKPOINT_COLLECTION_NAME,
    }
    excluded_files = {
        STATUS_NAME,
        ZIP_NAME,
        BLEND_NAME,
        DATASET_ENTRY_NAME,
        DATASET_PUBLICATION_NAME,
    }
    for source in sorted(
        job.rglob("*"), key=lambda value: value.relative_to(job).as_posix()
    ):
        relative = source.relative_to(job)
        if any(
            part.startswith(".") and part.endswith(".part") for part in relative.parts
        ):
            continue
        if relative.parts[0] in excluded_roots or source.name in excluded_files:
            continue
        if active_bundle_relative is not None and relative.parts[0] == "shared":
            is_active_or_child = (
                relative == active_bundle_relative
                or relative.is_relative_to(active_bundle_relative)
            )
            is_active_parent = active_bundle_relative.is_relative_to(relative)
            if not is_active_or_child and not is_active_parent:
                continue
        target = scratch / relative
        _inside(scratch, target, "assemblage local")
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve(strict=True) == source.resolve(
                strict=True
            ):
                continue
            if (
                target.is_file()
                and not target.is_symlink()
                and target.stat().st_size == source.stat().st_size
                and _sha256_file(target) == _sha256_file(source)
            ):
                continue
            raise SimpleProductionError(
                f"Assemblage local divergent: {relative.as_posix()}"
            )
        if source.is_symlink():
            target.symlink_to(source.resolve(strict=True))
        else:
            shutil.copyfile(source, target)
    return scratch


def _write_zip(
    job_root: Path,
    zone_id: str,
    *,
    destination: Path | None = None,
    progress_callback: ZipProgress | None = None,
) -> Path:
    destination = job_root / ZIP_NAME if destination is None else destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    excluded_roots = {"sources", "download", TILE_CHECKPOINT_COLLECTION_NAME}
    excluded_files = {STATUS_NAME, ZIP_NAME, temporary.name}
    files = [
        path
        for path in job_root.rglob("*")
        if path.is_file()
        and path.name not in excluded_files
        and path.relative_to(job_root).parts[0] not in excluded_roots
        and not any(
            part.startswith(".") and part.endswith(".part")
            for part in path.relative_to(job_root).parts
        )
    ]
    total_bytes = sum(path.stat().st_size for path in files)
    written_bytes = 0
    prefix = PurePosixPath(f"fireviewer-{zone_id}")
    already_compressed_suffixes = {
        ".blend",
        ".glb",
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".usdc",
        ".usdz",
        ".webp",
        ".zip",
    }
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        sorted_files = sorted(
            files, key=lambda value: value.relative_to(job_root).as_posix()
        )
        for file_index, path in enumerate(sorted_files, start=1):
            relative = path.relative_to(job_root).as_posix()
            info = zipfile.ZipInfo((prefix / relative).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in already_compressed_suffixes
                else zipfile.ZIP_DEFLATED
            )
            info.external_attr = 0o100644 << 16
            with (
                path.open("rb") as input_stream,
                archive.open(info, "w", force_zip64=True) as output_stream,
            ):
                while chunk := input_stream.read(1024 * 1024):
                    output_stream.write(chunk)
                    written_bytes += len(chunk)
                    if progress_callback is not None:
                        progress_callback(
                            written_bytes,
                            total_bytes,
                            file_index,
                            len(sorted_files),
                            relative,
                        )
    os.replace(temporary, destination)
    return destination


def _copy_result_sidecars(source_root: Path, destination_root: Path) -> None:
    if source_root.resolve() == destination_root.resolve():
        return
    for name in (
        ZONE_RECEIPT_NAME,
        DATASET_ENTRY_NAME,
        DATASET_PUBLICATION_NAME,
    ):
        source = source_root / name
        if not source.is_file():
            raise SimpleProductionError(f"Métadonnée finale absente: {name}")
        destination = destination_root / name
        if destination.is_file() and source.samefile(destination):
            continue
        shutil.copyfile(source, destination)


def _gallery_items(job_root: Path) -> GalleryItems:
    if not (job_root / BLEND_NAME).is_file():
        raise SimpleProductionError("La scène Blender autonome zone.blend est absente")
    return []


def _remove_legacy_gallery(job_root: Path) -> None:
    gallery_root = job_root / "qa" / "gallery"
    if gallery_root.exists():
        shutil.rmtree(gallery_root)
    legacy_receipt = job_root / "qa" / "zone-gallery-receipt.v1.json"
    legacy_receipt.unlink(missing_ok=True)


def _render_zone_gallery(
    config: ProductionConfig,
    job_root: Path,
    render: bool,
    progress_callback: GalleryProgress | None,
) -> GalleryItems:
    if not render:
        return _gallery_items(job_root)
    configured_scratch = _scratch_job_path(config, job_root.name)
    external_scratch = (
        configured_scratch
        if configured_scratch is not None
        and configured_scratch.resolve() != job_root.resolve()
        else None
    )
    runtime_root = (external_scratch or job_root) / "qa" / "blender-runtime"
    blend_output = (
        external_scratch / BLEND_NAME if external_scratch is not None else None
    )
    runtime_paths = {
        "TEMP": runtime_root / "temp",
        "TMP": runtime_root / "temp",
        "PYTHONPYCACHEPREFIX": runtime_root / "pycache",
        "BLENDER_USER_CONFIG": runtime_root / "config",
        "BLENDER_USER_SCRIPTS": runtime_root / "scripts",
        "BLENDER_USER_DATAFILES": runtime_root / "datafiles",
        "BLENDER_USER_EXTENSIONS": runtime_root / "extensions",
        "XDG_CACHE_HOME": runtime_root / "xdg-cache",
        "XDG_CONFIG_HOME": runtime_root / "xdg-config",
    }
    for path in set(runtime_paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({name: str(path) for name, path in runtime_paths.items()})
    script = Path(__file__).with_name("render_simple_zone_gallery.py").resolve()
    command = (
        str(config.blender),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        "pack",
        "--job-root",
        str(job_root),
    )
    if blend_output is not None:
        command += ("--blend-output", str(blend_output))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    if process.stdout is None:  # pragma: no cover - Popen invariant
        process.kill()
        raise SimpleProductionError("Blender ne fournit aucun journal de rendu")
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line.rstrip())
        messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    output_tail: list[str] = []
    deadline = time.monotonic() + 15 * 60
    stream_done = False
    while not stream_done:
        if time.monotonic() > deadline:
            process.kill()
            raise SimpleProductionError(
                "La création de la scène Blender dépasse 15 minutes"
            )
        try:
            line = messages.get(timeout=1.0)
        except queue.Empty:
            if process.poll() is not None and not reader.is_alive():
                break
            continue
        if line is None:
            stream_done = True
            continue
        output_tail.append(line)
        output_tail = output_tail[-120:]
    return_code = process.wait(timeout=30)
    reader.join(timeout=5)
    if return_code != 0:
        raise SimpleProductionError(
            "La création de la scène Blender autonome a échoué:\n"
            + "\n".join(output_tail[-30:])
        )
    expected_blend = blend_output if blend_output is not None else job_root / BLEND_NAME
    if not expected_blend.is_file():
        details = "\n".join(output_tail[-30:])
        raise SimpleProductionError(
            "Blender a terminé sans produire la scène autonome zone.blend"
            + (f":\n{details}" if details else "")
        )
    if blend_output is not None:
        published_blend = job_root / BLEND_NAME
        published_blend.unlink(missing_ok=True)
        published_blend.symlink_to(blend_output.resolve(strict=True))
    return _gallery_items(job_root)


def _publish_dataset_entry(
    config: ProductionConfig,
    *,
    job_root: Path,
    plan: ZonePlan,
    receipt: Mapping[str, Any],
    archive: Path,
    publication_progress: DatasetPublicationProgress | None = None,
) -> dict[str, Any] | None:
    """Publish one validated private scene pack; never persist the HF token."""

    if config.dataset_id is None:
        return None
    token = os.environ.get("HF_TOKEN", "").strip()
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise SimpleProductionError("Le ZIP est altéré avant publication dataset")
    build_id = receipt.get("build_id")
    if not isinstance(build_id, str) or len(build_id) != 64:
        raise SimpleProductionError("Build ID de zone invalide avant publication")
    archive_record = {
        "file": ZIP_NAME,
        "byte_count": archive.stat().st_size,
        "sha256": _sha256_file(archive),
    }
    capture_records: list[dict[str, Any]] = []
    entry: dict[str, Any] = {
        "schema": "fireviewer.simple-measured-scene-dataset-entry.v1",
        "status": "technical_scene_produced",
        "accepted_human": False,
        "dataset_id": config.dataset_id,
        "zone_id": plan.zone_id,
        "build_id": build_id,
        "request": {
            "latitude": plan.latitude,
            "longitude": plan.longitude,
            "side_m": plan.side_m,
            "production_bounds_l93_m": list(plan.production_bounds_l93_m),
        },
        "counts": {
            "tiles": receipt["tile_count"],
            "buildings": receipt["building_count"],
            "trees": receipt["tree_count"],
            "context_assets": receipt.get("context_asset_count", 0),
            "placeholder_instances": receipt.get("placeholder_instance_count", 0),
        },
        "archive": archive_record,
        "captures": capture_records,
        "zone_receipt": {
            "file": ZONE_RECEIPT_NAME,
            "sha256": _sha256_file(job_root / ZONE_RECEIPT_NAME),
        },
        "container_image": os.environ.get("FIREVIEWER_IMAGE_REFERENCE", "unrecorded"),
    }
    entry["entry_sha256"] = hashlib.sha256(_canonical_bytes(entry)).hexdigest()
    entry_path = job_root / DATASET_ENTRY_NAME
    publication_path = job_root / DATASET_PUBLICATION_NAME
    if publication_path.is_file():
        publication = _load_json(publication_path, "reçu de publication dataset")
        if publication.get("status") == "published_private":
            _validate_existing_publication(
                config,
                plan=plan,
                receipt=receipt,
                publication=publication,
                archive_sha256=archive_record["sha256"],
            )
            return publication
        if (
            publication.get("status") != "failed_pending_retry"
            or publication.get("dataset_id") != config.dataset_id
            or publication.get("zone_id") != plan.zone_id
            or publication.get("build_id") != build_id
            or publication.get("archive_sha256") != archive_record["sha256"]
            or publication.get("entry_sha256") != entry["entry_sha256"]
            or publication.get("captures") != []
        ):
            raise SimpleProductionError(
                "Le reçu de publication différée est incohérent"
            )

    _write_json(entry_path, entry)

    try:
        if not token:
            raise SimpleProductionError(
                "HF_TOKEN absent: publication dans la dataset privée impossible"
            )
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi(token=token)
        info = api.repo_info(repo_id=config.dataset_id, repo_type="dataset")
        if info.private is not True:
            raise SimpleProductionError(
                "La dataset cible doit être privée avant toute publication"
            )
        remote_root = f"zones/{plan.zone_id}/{build_id}"
        operation_specs = [
            (f"{remote_root}/{ZIP_NAME}", archive),
            (f"{remote_root}/{ZONE_RECEIPT_NAME}", job_root / ZONE_RECEIPT_NAME),
            (f"{remote_root}/{DATASET_ENTRY_NAME}", entry_path),
            *(
                (f"{remote_root}/{record['file']}", job_root / record["file"])
                for record in capture_records
            ),
        ]

        commit = None
        for attempt in range(1, HF_PUBLICATION_MAX_ATTEMPTS + 1):
            if publication_progress is not None:
                publication_progress(
                    "dataset_publication_attempt",
                    "Publication Hugging Face — tentative "
                    f"{attempt}/{HF_PUBLICATION_MAX_ATTEMPTS}",
                )
            operations = [
                CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
                for remote, local in operation_specs
            ]
            try:
                commit = api.create_commit(
                    repo_id=config.dataset_id,
                    repo_type="dataset",
                    commit_message=f"Add measured FireViewer scene {plan.zone_id}",
                    operations=operations,
                )
                break
            except Exception as error:
                if (
                    attempt >= HF_PUBLICATION_MAX_ATTEMPTS
                    or not _is_transient_hf_publication_error(error)
                ):
                    raise
                delay = HF_PUBLICATION_BACKOFF_SECONDS[attempt - 1]
                if publication_progress is not None:
                    publication_progress(
                        "dataset_publication_retry",
                        "Upload Hugging Face interrompu — reprise automatique dans "
                        f"{int(delay)} s",
                    )
                time.sleep(delay)
        if commit is None:
            raise SimpleProductionError("Publication Hugging Face sans reçu de commit")
    except SimpleProductionError as error:
        if config.dataset_publication_required:
            raise
        pending = {
            "schema": "fireviewer.simple-measured-scene-dataset-publication.v1",
            "status": "failed_pending_retry",
            "dataset_id": config.dataset_id,
            "zone_id": plan.zone_id,
            "build_id": build_id,
            "archive_sha256": archive_record["sha256"],
            "entry_sha256": entry["entry_sha256"],
            "error_type": type(error).__name__,
            "error": "Publication Hugging Face à reprendre",
            "captures": capture_records,
        }
        _write_json(publication_path, pending)
        return pending
    except Exception as error:
        wrapped = SimpleProductionError(f"Publication dataset privée échouée: {error}")
        if config.dataset_publication_required:
            raise wrapped from error
        pending = {
            "schema": "fireviewer.simple-measured-scene-dataset-publication.v1",
            "status": "failed_pending_retry",
            "dataset_id": config.dataset_id,
            "zone_id": plan.zone_id,
            "build_id": build_id,
            "archive_sha256": archive_record["sha256"],
            "entry_sha256": entry["entry_sha256"],
            "error_type": type(error).__name__,
            "error": "Publication Hugging Face à reprendre",
            "captures": capture_records,
        }
        _write_json(publication_path, pending)
        return pending
    publication = {
        "schema": "fireviewer.simple-measured-scene-dataset-publication.v1",
        "status": "published_private",
        "dataset_id": config.dataset_id,
        "zone_id": plan.zone_id,
        "build_id": build_id,
        "archive_sha256": archive_record["sha256"],
        "entry_sha256": entry["entry_sha256"],
        "commit_oid": commit.oid,
        "path_in_repo": remote_root,
        "captures": capture_records,
    }
    _write_json(publication_path, publication)
    return publication


def _is_transient_hf_publication_error(error: BaseException) -> bool:
    """Recognize retryable Hub/Xet transport failures without retrying auth errors."""

    retryable_fragments = (
        "timeout",
        "timed out",
        "error decoding response body",
        "connection reset",
        "connection aborted",
        "server disconnected",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "status code 502",
        "status code 503",
        "status code 504",
    )
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if any(fragment in str(current).casefold() for fragment in retryable_fragments):
            return True
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _validate_existing_publication(
    config: ProductionConfig,
    *,
    plan: ZonePlan,
    receipt: Mapping[str, Any],
    publication: Mapping[str, Any],
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    build_id = receipt.get("build_id")
    remote_root = f"zones/{plan.zone_id}/{build_id}"
    expected = {
        "schema": "fireviewer.simple-measured-scene-dataset-publication.v1",
        "status": "published_private",
        "dataset_id": config.dataset_id,
        "zone_id": plan.zone_id,
        "build_id": build_id,
        "path_in_repo": remote_root,
        "captures": [],
    }
    if any(publication.get(key) != value for key, value in expected.items()):
        raise SimpleProductionError(
            "Le reçu de publication dataset existant est incohérent"
        )
    for key in ("archive_sha256", "entry_sha256"):
        value = publication.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise SimpleProductionError(
                "Le reçu de publication dataset existant est incohérent"
            )
    commit_oid = publication.get("commit_oid")
    if not isinstance(commit_oid, str) or not commit_oid:
        raise SimpleProductionError(
            "Le reçu de publication dataset existant est incohérent"
        )
    if (
        archive_sha256 is not None
        and publication.get("archive_sha256") != archive_sha256
    ):
        raise SimpleProductionError(
            "Le reçu de publication dataset existant est incohérent"
        )
    return dict(publication)


class ProductionEngine:
    """Sequential, one-job-at-a-time production engine used by the headless API."""

    def __init__(
        self,
        config: ProductionConfig,
        *,
        prepare_context_fn: PrepareContext = prepare_zone_context,
        prepare_sources_fn: PrepareSources = prepare_sources,
        produce_tile_fn: ProduceTile = produce_simple_measured_tile,
        render_gallery_fn: RenderGallery | None = None,
        validate_assets: bool = True,
    ) -> None:
        self.config = config
        self.prepare_context_fn = prepare_context_fn
        self.prepare_sources_fn = prepare_sources_fn
        self.produce_tile_fn = produce_tile_fn
        self.render_gallery_fn = render_gallery_fn or (
            lambda job_root, render, callback: _render_zone_gallery(
                config, job_root, render, callback
            )
        )
        self._lock = threading.Lock()
        if validate_assets:
            self.asset_summary = validate_production_config(config)
            self.asset_library_payload = _load_json(
                _absolute(config.asset_library, "catalogue des assets", exists=True),
                "catalogue des assets",
            )
            self.fixed_asset_choices = tuple(asset_choices(self.asset_library_payload))
        else:
            work = _absolute(config.work_root, "volume de travail", exists=False)
            work.mkdir(parents=True, exist_ok=True)
            self.asset_summary = {
                "asset_count": 53,
                "building_assets": 24,
                "tree_assets": 18,
                "checked_artifacts": 106,
                "catalog_revision": "0" * 64,
                "catalog_sha256": (
                    _sha256_file(config.asset_library)
                    if config.asset_library.is_file()
                    else "0" * 64
                ),
                "blender_path": str(config.blender),
                "blender_version": "test",
                "blender_sha256": "0" * 64,
                "openusd": "test",
                "production_script_count": 12,
            }
            try:
                payload = _load_json(
                    config.asset_library, "catalogue des assets de test"
                )
                choices = tuple(asset_choices(payload))
            except (OSError, SimpleProductionError, FixedAssetPlacementError):
                payload = {}
                choices = ()
            self.asset_library_payload = payload
            self.fixed_asset_choices = choices

    def run(
        self,
        latitude: float,
        longitude: float,
        side_km: float,
        *,
        progress_callback: ProgressCallback | None = None,
        archive_ready_callback: ArchiveReadyCallback | None = None,
        fixed_asset_placements: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[str, str | None, GalleryItems]]:
        if not self._lock.acquire(blocking=False):
            raise SimpleProductionError("Une production est déjà en cours sur ce pod")
        try:
            yield from self._run(
                latitude,
                longitude,
                side_km,
                progress_callback=progress_callback,
                archive_ready_callback=archive_ready_callback,
                fixed_asset_placements=(
                    dict(EMPTY_FIXED_ASSET_REQUEST)
                    if fixed_asset_placements is None
                    else normalize_fixed_asset_request(
                        fixed_asset_placements, self.asset_library_payload
                    )
                ),
            )
        finally:
            self._lock.release()

    def _run(
        self,
        latitude: float,
        longitude: float,
        side_km: float,
        *,
        progress_callback: ProgressCallback | None,
        archive_ready_callback: ArchiveReadyCallback | None,
        fixed_asset_placements: Mapping[str, Any],
    ) -> Iterator[tuple[str, str | None, GalleryItems]]:
        started = time.perf_counter()
        base_plan = plan_zone(
            latitude,
            longitude,
            side_km,
            max_side_m=self.config.max_side_m,
            max_tiles=self.config.max_tiles,
        )
        fixed_count = len(fixed_asset_placements["placements"])
        projected_fixed_assets = (
            project_fixed_asset_request(
                fixed_asset_placements,
                self.asset_library_payload,
                requested_bounds_l93_m=base_plan.requested_bounds_l93_m,
            )
            if fixed_count
            else ()
        )
        plan = (
            replace(
                base_plan,
                zone_id=(
                    f"{base_plan.zone_id}-fixed-"
                    f"{fixed_asset_request_sha256(fixed_asset_placements)[:12]}"
                ),
            )
            if fixed_count
            else base_plan
        )
        work = _absolute(self.config.work_root, "volume de travail", exists=True)
        job_root = work / "jobs" / plan.zone_id
        job_root.mkdir(parents=True, exist_ok=True)
        if fixed_count:
            fixed_request_path = job_root / FIXED_ASSET_REQUEST_NAME
            if fixed_request_path.exists():
                if _load_json(fixed_request_path, "placements fixes existants") != dict(
                    fixed_asset_placements
                ):
                    raise SimpleProductionError(
                        "Les placements fixes existants diffèrent de la requête"
                    )
            else:
                _write_json(fixed_request_path, fixed_asset_placements)

        report_lock = threading.Lock()
        last_report_fraction = 0.0

        def report(
            fraction: float,
            phase: str,
            message: str,
            *,
            completed_tiles: int = 0,
            details: Mapping[str, Any] | None = None,
        ) -> None:
            nonlocal last_report_fraction
            with report_lock:
                bounded_fraction = max(0.0, min(1.0, fraction))
                bounded_fraction = max(last_report_fraction, bounded_fraction)
                last_report_fraction = bounded_fraction
                _status(
                    job_root,
                    state="producing",
                    message=message,
                    completed=completed_tiles,
                    total=len(plan.tiles),
                    phase=phase,
                    details=details,
                    elapsed_seconds=time.perf_counter() - started,
                )
                if progress_callback is not None:
                    progress_callback(bounded_fraction, message)

        report(
            0.01,
            "embedded_runtime_verified",
            "Moteur embarqué validé — Blender "
            f"{self.asset_summary['blender_version']}, OpenUSD, "
            f"{self.asset_summary['asset_count']} assets USD "
            f"({self.asset_summary.get('fallback_usd_assets', 0)} substitutions "
            "USD déterministes)",
        )
        plan_path = job_root / PLAN_NAME
        plan_payload = _plan_payload(
            plan,
            self.config,
            fixed_asset_request=fixed_asset_placements if fixed_count else None,
        )
        if plan_path.exists():
            if _load_json(plan_path, "plan existant") != plan_payload:
                raise SimpleProductionError(
                    "Un job différent utilise déjà cet identifiant"
                )
        else:
            _write_json(plan_path, plan_payload)

        existing_receipt = job_root / ZONE_RECEIPT_NAME
        existing_publication = job_root / DATASET_PUBLICATION_NAME
        if (
            self.config.dataset_id is not None
            and existing_receipt.is_file()
            and existing_publication.is_file()
            and (job_root / ZIP_NAME).is_file()
            and _load_json(existing_publication, "reçu de publication dataset").get(
                "status"
            )
            == "published_private"
        ):
            receipt = _load_json(existing_receipt, "reçu de zone existant")
            if (
                receipt.get("schema") != ZONE_RECEIPT_SCHEMA
                or receipt.get("zone_id") != plan.zone_id
                or receipt.get("tile_count") != len(plan.tiles)
            ):
                raise SimpleProductionError("La production existante est incohérente")
            publication = _validate_existing_publication(
                self.config,
                plan=plan,
                receipt=receipt,
                publication=_load_json(
                    existing_publication, "reçu de publication dataset"
                ),
            )
            report(
                1.0,
                "dataset_publication_reused",
                "Déjà publié — résultat privé Hugging Face réutilisé",
                completed_tiles=len(plan.tiles),
            )
            yield (
                f"Déjà publié — {receipt['tile_count']} terrains, "
                f"{receipt['building_count']} bâtiments, "
                f"{receipt['tree_count']} arbres, résultat privé revalidé.",
                str(job_root / ZIP_NAME),
                [],
            )
            return

        scratch_job = _prepare_scratch_job(self.config, plan.zone_id)

        removed_staging = _remove_interrupted_staging(job_root)
        if removed_staging:
            report(
                0.02,
                "interrupted_staging_removed",
                f"Reprise — {len(removed_staging)} staging(s) interrompu(s) nettoyé(s)",
                details={"removed_staging": removed_staging},
            )

        existing_archive = job_root / ZIP_NAME
        existing_stage = job_root / ENTRY_STAGE
        existing_gallery = job_root / "qa" / "zone-gallery-receipt.v1.json"
        existing_blend = job_root / BLEND_NAME
        if (
            existing_archive.is_file()
            and existing_receipt.is_file()
            and existing_stage.is_file()
            and existing_blend.is_file()
            and not existing_gallery.exists()
        ):
            receipt = _load_json(existing_receipt, "reçu de zone existant")
            if (
                receipt.get("schema") != ZONE_RECEIPT_SCHEMA
                or receipt.get("zone_id") != plan.zone_id
                or receipt.get("tile_count") != len(plan.tiles)
                or receipt.get("entry_stage", {}).get("sha256")
                != _sha256_file(existing_stage)
            ):
                raise SimpleProductionError("La production existante est incohérente")
            validate_map_upload_package(job_root)
            with zipfile.ZipFile(existing_archive) as archive:
                expected = f"fireviewer-{plan.zone_id}/{ENTRY_STAGE}"
                if expected not in archive.namelist() or archive.testzip() is not None:
                    raise SimpleProductionError("Le ZIP existant est altéré")
            existing_archive_sha256 = _sha256_file(existing_archive)
            if archive_ready_callback is not None:
                archive_ready_callback(
                    existing_archive,
                    existing_archive.stat().st_size,
                    existing_archive_sha256,
                )
            gallery = self.render_gallery_fn(job_root, False, None)
            if self.config.dataset_id is not None:
                report(
                    0.98,
                    "dataset_publication",
                    "Publication du pack validé dans la dataset FireViewer privée",
                    completed_tiles=len(plan.tiles),
                )
                _publish_dataset_entry(
                    self.config,
                    job_root=job_root,
                    plan=plan,
                    receipt=receipt,
                    archive=existing_archive,
                    publication_progress=lambda phase, message: report(
                        0.98,
                        phase,
                        message,
                        completed_tiles=len(plan.tiles),
                    ),
                )
            completed_message = (
                f"Déjà produit — {receipt['tile_count']} terrains, "
                f"{receipt['building_count']} bâtiments, {receipt['tree_count']} arbres, "
                f"{receipt.get('context_asset_count', 0)} équipements contextuels, "
                "scène autonome revalidée."
            )
            yield (
                completed_message,
                str(existing_archive),
                gallery,
            )
            return

        total = len(plan.tiles)
        context_storage_root = scratch_job if scratch_job is not None else job_root
        zone_context_path = context_storage_root / ZONE_CONTEXT_NAME
        report(
            0.03,
            "zone_context_download",
            "Contexte IGN — téléchargement BD TOPO/occupation du sol",
        )
        yield f"Préparation de {total} tuiles de 500 m…", None, []
        zone_context = self.prepare_context_fn(
            output_path=zone_context_path,
            zone_id=plan.zone_id,
            bounds_l93_m=plan.context_bounds_l93_m,
            source_revision=self.config.context_revision,
        )
        report(
            0.08,
            "zone_context_validated",
            "Contexte IGN validé — "
            + ", ".join(
                f"{name}: {count}"
                for name, count in sorted(zone_context.feature_counts.items())
            ),
            details={"feature_counts": dict(zone_context.feature_counts)},
        )

        asset_roots = {"review_batch": self.config.review_batch}
        checkpoint_bundle = _prototype_bundle_root(job_root, self.asset_summary)
        source_storage_root = scratch_job if scratch_job is not None else job_root
        package_storage_root = scratch_job if scratch_job is not None else job_root
        shared_bundle = _prototype_bundle_root(package_storage_root, self.asset_summary)
        if scratch_job is not None:
            _sync_prototype_bundle(checkpoint_bundle, shared_bundle)
        production_slots = threading.BoundedSemaphore(self.config.tile_workers)
        prototype_checkpoint_lock = threading.Lock()
        checkpoint_publish_lock = threading.Lock()
        checkpoint_staging_root = package_storage_root / ".checkpoint-staging"

        def write_checkpoint(package_root: Path, tile_id: str) -> dict[str, Any]:
            return _write_tile_checkpoint(
                package_root,
                job_root,
                tile_id=tile_id,
                local_staging_root=checkpoint_staging_root,
                publish_lock=checkpoint_publish_lock,
            )

        prototype_checkpoint_files: dict[
            str,
            tuple[
                tuple[int, int, int, int],
                tuple[int, int, int, int],
            ],
        ] = {}
        completed = 0
        source_phase = {
            "download_mnt_started": (0.04, "MNT 0,5 m — téléchargement"),
            "download_mnt_completed": (0.14, "MNT 0,5 m reçu"),
            "download_mns_started": (0.16, "MNS 0,5 m — téléchargement"),
            "download_mns_completed": (0.26, "MNS 0,5 m reçu"),
            "download_orthophoto_started": (0.28, "Orthophoto 1 m — téléchargement"),
            "download_orthophoto_completed": (0.38, "Orthophoto 1 m reçue"),
            "sources_decoded": (0.43, "Sources raster décodées et co-enregistrées"),
            "tile_context_prepared": (0.47, "Contexte de placement découpé"),
            "sources_published": (0.50, "Sources de tuile validées"),
            "sources_reused": (0.50, "Sources de tuile revalidées et réutilisées"),
            "metatile_reused": (0.38, "Métatuile 4×4 locale réutilisée"),
            "metatile_fallback": (
                0.04,
                "Métatuile indisponible — repli sûr sur la tuile de 500 m",
            ),
        }
        production_phase = {
            "terrain_compiled": (0.60, "Terrain FVTG compilé"),
            "ground_texture_baked": (0.68, "Texture sol orthophoto cuite"),
            "placement_measured": (0.78, "Placement MNS−MNT mesuré"),
            "terrain_usd_exported": (0.84, "Terrain OpenUSD exporté"),
            "prototype_bundle_started": (0.85, "Lot d'assets USD démarré"),
            "prototype_bundle_progress": (0.90, "Assets USD publiés en parallèle"),
            "scene_usd_built": (0.92, "Scène OpenUSD assemblée"),
            "tile_staging_validated": (0.97, "Package de tuile rehashé"),
            "tile_published": (1.00, "Package de tuile publié"),
            "tile_reused": (1.00, "Package existant revalidé et réutilisé"),
            "tile_checkpoint_started": (
                0.98,
                "Compression du checkpoint sur SSD local",
            ),
            "tile_checkpoint_published": (
                0.99,
                "Checkpoint publié sur le volume persistant",
            ),
        }
        tile_progress = [0.0] * total
        tile_last_activity = [time.monotonic()] * total
        tile_progress_lock = threading.Lock()
        completed_tiles: set[int] = set()
        validated_checkpoints: dict[str, dict[str, Any]] = {}
        stop_tiles = threading.Event()

        def validate_published_tile(tile: TilePlan, package_root: Path) -> None:
            validate_simple_measured_tile_package(
                package_root,
                expected_request=_expected_request_from_receipt(
                    package_root,
                    plan=plan,
                    tile=tile,
                    asset_library=self.config.asset_library,
                    asset_roots=asset_roots,
                    portable_root=self.config.portable_root,
                    asset_bundle_root=shared_bundle,
                    asset_library_sha256=self.asset_summary["catalog_sha256"],
                ),
                asset_library=self.config.asset_library,
                asset_roots=asset_roots,
            )

        def process_tile(tile_index: int, tile: TilePlan) -> str:
            with tile_progress_lock:
                tile_last_activity[tile_index] = time.monotonic()

            def tile_report(
                phase: str,
                details: Mapping[str, Any],
                *,
                table: Mapping[str, tuple[float, str]],
            ) -> None:
                if stop_tiles.is_set():
                    raise SimpleProductionError(
                        "Production parallèle interrompue après l'échec d'une tuile"
                    )
                local, label = table.get(phase, (0.0, phase))
                suffixes: list[str] = []
                if isinstance(details.get("byte_count"), int):
                    suffixes.append(f"{details['byte_count'] / 1_048_576:.1f} Mio")
                for key, label_key in (
                    ("building_count", "bâtiments"),
                    ("tree_count", "arbres"),
                    ("context_asset_count", "équipements"),
                ):
                    if isinstance(details.get(key), int):
                        suffixes.append(f"{details[key]} {label_key}")
                prototype_completed = details.get("prototype_completed")
                prototype_total = details.get("prototype_total")
                if (
                    isinstance(prototype_completed, int)
                    and isinstance(prototype_total, int)
                    and prototype_total > 0
                ):
                    suffixes.append(
                        f"{prototype_completed}/{prototype_total} prototypes"
                    )
                message = f"Tuile {tile_index + 1}/{total} — {label}"
                if suffixes:
                    message += " — " + ", ".join(suffixes)
                with tile_progress_lock:
                    tile_progress[tile_index] = max(tile_progress[tile_index], local)
                    tile_last_activity[tile_index] = time.monotonic()
                    fraction = 0.08 + 0.82 * (sum(tile_progress) / total)
                    completed_count = len(completed_tiles)
                report(
                    fraction,
                    phase,
                    message,
                    completed_tiles=completed_count,
                    details={**details, "tile_index": tile_index + 1},
                )

            if stop_tiles.is_set():
                raise SimpleProductionError(
                    "Production parallèle interrompue avant la tuile"
                )
            package_root = package_storage_root / "packages" / tile.tile_id
            if package_root.exists():
                validate_published_tile(tile, package_root)
                tile_report(
                    "tile_checkpoint_started",
                    {"tile_id": tile.tile_id},
                    table=production_phase,
                )
                checkpoint = write_checkpoint(package_root, tile.tile_id)
                with tile_progress_lock:
                    validated_checkpoints[tile.tile_id] = checkpoint
                tile_report(
                    "tile_checkpoint_published",
                    {"tile_id": tile.tile_id},
                    table=production_phase,
                )
                tile_report(
                    "tile_reused", {"tile_id": tile.tile_id}, table=production_phase
                )
            else:
                source_root = source_storage_root / "sources" / tile.tile_id
                sources = self.prepare_sources_fn(
                    output_root=source_root,
                    zone_id=plan.zone_id,
                    tile_id=tile.tile_id,
                    tile_origin_l93_m=tile.origin_l93_m,
                    elevation_revision=self.config.elevation_revision,
                    orthophoto_revision=self.config.orthophoto_revision,
                    zone_context=zone_context_path,
                    fixed_asset_placements=[
                        placement
                        for placement in projected_fixed_assets
                        if placement["owner_tile_origin_l93_m"]
                        == list(tile.origin_l93_m)
                    ],
                    metatile_cache_root=source_storage_root / "metatiles" / "v1",
                    progress_callback=lambda phase, details: tile_report(
                        phase, details, table=source_phase
                    ),
                )
                with production_slots:
                    package = self.produce_tile_fn(
                        mnt_05m=sources.mnt,
                        mns_05m=sources.mns,
                        orthophoto_1m=sources.orthophoto,
                        elevation_source_receipt=sources.elevation_receipt,
                        orthophoto_source_receipt=sources.orthophoto_receipt,
                        placement_context=sources.placement_context,
                        asset_library=self.config.asset_library,
                        asset_roots=asset_roots,
                        portable_root=self.config.portable_root,
                        asset_bundle_root=shared_bundle,
                        asset_bundle_identity_root=checkpoint_bundle,
                        output_root=package_root,
                        zone_id=plan.zone_id,
                        tile_id=tile.tile_id,
                        tile_origin_l93_m=tile.origin_l93_m,
                        progress_callback=lambda phase, details: tile_report(
                            phase, details, table=production_phase
                        ),
                    )
                    if package.output_root.resolve() != package_root.resolve():
                        raise SimpleProductionError(
                            "Le producteur a publié la tuile hors du SSD local"
                        )
                if scratch_job is not None:
                    with prototype_checkpoint_lock:
                        _sync_prototype_bundle(
                            shared_bundle,
                            checkpoint_bundle,
                            validated_files=prototype_checkpoint_files,
                        )
                tile_report(
                    "tile_checkpoint_started",
                    {"tile_id": tile.tile_id},
                    table=production_phase,
                )
                checkpoint = write_checkpoint(package.output_root, tile.tile_id)
                with tile_progress_lock:
                    validated_checkpoints[tile.tile_id] = checkpoint
                tile_report(
                    "tile_checkpoint_published",
                    {"tile_id": tile.tile_id},
                    table=production_phase,
                )
                _copy_provenance(sources, job_root / "provenance" / tile.tile_id)
                _remove_sources(source_storage_root, sources.root)
            with tile_progress_lock:
                tile_progress[tile_index] = 1.0
                completed_tiles.add(tile_index)
                completed_count = len(completed_tiles)
                fraction = 0.08 + 0.82 * (sum(tile_progress) / total)
            report(
                fraction,
                "raw_sources_removed",
                f"Tuile {tile_index + 1}/{total} — package validé, sources brutes absentes",
                completed_tiles=completed_count,
                details={"tile_id": tile.tile_id, "tile_index": tile_index + 1},
            )
            return tile.tile_id

        pending_tiles: list[tuple[int, TilePlan]] = []
        for tile_index, tile in enumerate(plan.tiles):
            package_root = package_storage_root / "packages" / tile.tile_id
            archive_path, checkpoint_receipt_path = _tile_checkpoint_paths(
                job_root, tile.tile_id
            )
            checkpoint: dict[str, Any] | None = None
            if archive_path.exists() or checkpoint_receipt_path.exists():
                try:
                    checkpoint = _restore_tile_checkpoint(
                        job_root,
                        package_root,
                        tile_id=tile.tile_id,
                    )
                    validate_published_tile(tile, package_root)
                except Exception as error:
                    if package_root.exists() and scratch_job is not None:
                        shutil.rmtree(package_root)
                    quarantined = _quarantine_tile_checkpoint(
                        job_root, tile_id=tile.tile_id
                    )
                    report(
                        0.08,
                        "tile_checkpoint_quarantined",
                        f"Reprise — checkpoint invalide ignoré pour {tile.tile_id}",
                        completed_tiles=len(completed_tiles),
                        details={
                            "tile_id": tile.tile_id,
                            "error": str(error),
                            "quarantined": quarantined,
                        },
                    )
                    checkpoint = None
            if checkpoint is None:
                legacy_package = job_root / "packages" / tile.tile_id
                if legacy_package.is_dir() and legacy_package != package_root:
                    try:
                        validate_published_tile(tile, legacy_package)
                        checkpoint = write_checkpoint(legacy_package, tile.tile_id)
                        _restore_tile_checkpoint(
                            job_root,
                            package_root,
                            tile_id=tile.tile_id,
                        )
                        validate_published_tile(tile, package_root)
                    except Exception as error:
                        if package_root.exists():
                            shutil.rmtree(package_root)
                        report(
                            0.08,
                            "legacy_tile_ignored",
                            f"Reprise — ancienne tuile invalide ignorée: {tile.tile_id}",
                            completed_tiles=len(completed_tiles),
                            details={"tile_id": tile.tile_id, "error": str(error)},
                        )
                        checkpoint = None
                elif package_root.is_dir():
                    validate_published_tile(tile, package_root)
                    checkpoint = write_checkpoint(package_root, tile.tile_id)
            if checkpoint is None:
                pending_tiles.append((tile_index, tile))
                continue
            with tile_progress_lock:
                validated_checkpoints[tile.tile_id] = checkpoint
                tile_progress[tile_index] = 1.0
                completed_tiles.add(tile_index)
                completed_count = len(completed_tiles)
                fraction = 0.08 + 0.82 * (sum(tile_progress) / total)
            report(
                fraction,
                "tile_reused",
                f"Reprise — tuile {tile_index + 1}/{total} publiée et revalidée",
                completed_tiles=completed_count,
                details={"tile_id": tile.tile_id, "tile_index": tile_index + 1},
            )

        pending_tiles = _interleave_tiles_by_metatile(pending_tiles)
        worker_count = min(self.config.source_workers, len(pending_tiles))
        if not pending_tiles:
            report(
                0.90,
                "all_tiles_reused",
                f"Reprise — {total}/{total} tuiles publiées réutilisées",
                completed_tiles=total,
            )
        elif worker_count == 1:
            for tile_index, tile in pending_tiles:
                yield f"Tuile {tile_index + 1}/{total} — {tile.tile_id}", None, []
                process_tile(tile_index, tile)
        else:
            report(
                0.08,
                "parallel_tile_production",
                f"Pipeline parallèle — {worker_count} acquisitions et "
                f"{self.config.tile_workers} compilations au maximum",
            )
            for tile_id in _bounded_parallel_tile_results(
                pending_tiles,
                worker_count=worker_count,
                process_tile=process_tile,
                stop_event=stop_tiles,
                last_activity=lambda tile_index, _tile: tile_last_activity[tile_index],
                stall_timeout_seconds=self.config.tile_stall_timeout_seconds,
            ):
                with tile_progress_lock:
                    done_count = len(completed_tiles)
                    fraction = 0.08 + 0.82 * (sum(tile_progress) / total)
                if tile_id is None:
                    report(
                        fraction,
                        "parallel_tile_heartbeat",
                        f"Production active — {done_count}/{total} tuiles terminées, "
                        f"{worker_count} acquisitions et {self.config.tile_workers} "
                        "compilations au maximum en cours",
                        completed_tiles=done_count,
                        details={
                            "max_active_source_count": worker_count,
                            "max_active_tile_count": self.config.tile_workers,
                        },
                    )
                    continue
                yield f"Tuile {done_count}/{total} terminée — {tile_id}", None, []
        completed = len(completed_tiles)

        _status(
            job_root,
            state="packaging",
            message="Assemblage de la scène unifiée",
            completed=completed,
            total=total,
        )
        yield "Assemblage de la scène unifiée et du ZIP…", None, []
        assembly_root = (
            _prepare_local_assembly(
                job_root,
                scratch_job,
                active_prototype_bundle=checkpoint_bundle,
            )
            if scratch_job is not None
            else job_root
        )
        _write_zone_stage(assembly_root, plan)
        receipt_assets = {
            **self.asset_summary,
            "asset_library": str(self.config.asset_library),
            "review_batch": str(self.config.review_batch),
        }
        receipt = _write_zone_receipt(
            assembly_root,
            plan,
            receipt_assets,
            self.config.portable_root,
            shared_bundle,
            validated_checkpoints,
        )
        _remove_legacy_gallery(job_root)
        if assembly_root != job_root:
            _remove_legacy_gallery(assembly_root)
        report(
            0.94,
            "standalone_scene_pack_started",
            "Blender — création de la scène autonome sans captures",
            completed_tiles=completed,
        )
        gallery = self.render_gallery_fn(assembly_root, True, None)
        seal_map_upload_package(assembly_root)
        report(
            0.99, "zip_write", "Compression du pack autonome", completed_tiles=completed
        )
        last_zip_progress_at = 0.0

        def report_zip_progress(
            written_bytes: int,
            total_bytes: int,
            file_index: int,
            file_count: int,
            relative: str,
        ) -> None:
            nonlocal last_zip_progress_at
            now = time.monotonic()
            if written_bytes < total_bytes and now - last_zip_progress_at < 10.0:
                return
            last_zip_progress_at = now
            ratio = written_bytes / total_bytes if total_bytes else 1.0
            report(
                0.99 + 0.004 * ratio,
                "zip_write",
                "Compression du pack autonome — "
                f"{file_index}/{file_count} fichiers, "
                f"{written_bytes / (1024**3):.2f}/"
                f"{total_bytes / (1024**3):.2f} Gio",
                completed_tiles=completed,
                details={
                    "archive_written_bytes": written_bytes,
                    "archive_total_bytes": total_bytes,
                    "archive_file_index": file_index,
                    "archive_file_count": file_count,
                    "archive_current_file": relative,
                },
            )

        archive = _write_zip(
            assembly_root,
            plan.zone_id,
            progress_callback=report_zip_progress,
        )
        archive_sha256 = _sha256_file(archive)
        if archive_ready_callback is not None:
            report(
                0.994,
                "admin_archive_upload",
                "Mise à disposition du ZIP privé dans l'administration",
                completed_tiles=completed,
            )
            archive_ready_callback(archive, archive.stat().st_size, archive_sha256)
        publication = None
        if self.config.dataset_id is not None:
            report(
                0.995,
                "dataset_publication",
                "Publication du pack validé dans la dataset FireViewer privée",
                completed_tiles=completed,
            )
            publication = _publish_dataset_entry(
                self.config,
                job_root=assembly_root,
                plan=plan,
                receipt=receipt,
                archive=archive,
                publication_progress=lambda phase, message: report(
                    0.995,
                    phase,
                    message,
                    completed_tiles=completed,
                ),
            )
            _copy_result_sidecars(assembly_root, job_root)
        elif assembly_root != job_root:
            shutil.copyfile(
                assembly_root / ZONE_RECEIPT_NAME,
                job_root / ZONE_RECEIPT_NAME,
            )
        _status(
            job_root,
            state="completed",
            message="Scène produite",
            completed=completed,
            total=total,
        )
        dataset_suffix = (
            f" Publié dans {publication['dataset_id']}."
            if publication and publication.get("status") == "published_private"
            else " Publication Hugging Face différée; téléchargement admin disponible."
            if publication and publication.get("status") == "failed_pending_retry"
            else ""
        )
        completed_message = (
            f"Terminé — {completed} terrains, {receipt['building_count']} bâtiments, "
            f"{receipt['tree_count']} arbres, "
            f"{receipt.get('context_asset_count', 0)} équipements contextuels, "
            f"scène autonome sans captures.{dataset_suffix} "
            f"Ouvrir {BLEND_NAME} ou {ENTRY_STAGE} après extraction."
        )
        yield (
            completed_message,
            str(archive),
            gallery,
        )


__all__ = [
    "ENTRY_STAGE",
    "SCHEMA",
    "ProductionConfig",
    "ProductionEngine",
    "SimpleProductionError",
    "TilePlan",
    "ZonePlan",
    "plan_zone",
    "validate_config",
    "validate_embedded_assets",
    "validate_embedded_runtime",
    "validate_production_config",
]
