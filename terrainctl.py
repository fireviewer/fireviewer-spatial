"""Fail-closed, single-zone orchestration for adaptive FireViewer terrain builds.

This module owns the public orchestration contract.  It plans a deterministic
500 m Lambert-93 grid, persists immutable phase receipts and delegates heavy
work to phase backends.  The command-line entry point installs the audited
mono-zone production backend; tests may still inject isolated backends.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import time
from typing import Any, Protocol
from urllib.parse import urlsplit

from blender.validate_adaptive_terrain_zone import (
    ACCEPTANCE_SCHEMA as ZONE_VISUAL_SCHEMA,
    ACCEPTED_STATUS as ZONE_VISUAL_ACCEPTED_STATUS,
    HUMAN_REVIEW_SCHEMA as ZONE_VISUAL_HUMAN_REVIEW_SCHEMA,
    JOB_SCHEMA as ZONE_VISUAL_JOB_SCHEMA,
    TECHNICAL_RECEIPT_SCHEMA as ZONE_VISUAL_TECHNICAL_SCHEMA,
    TECHNICAL_STATUS as ZONE_VISUAL_TECHNICAL_STATUS,
    accept_zone_visual_review,
    validate_technical_receipt,
)


ZONE_SPEC_SCHEMA = "fireviewer.zone-spec.v1"
ZONE_PLAN_SCHEMA = "fireviewer.zone-plan.v1"
SOURCE_LOCK_SCHEMA = "fireviewer.zone-source-lock.v1"
TILE_RECEIPT_SCHEMA = "fireviewer.tile.done.v3"
ZONE_ACCEPTANCE_SCHEMA = "fireviewer.zone.acceptance.v1"
RUNTIME_SHADER_PENDING_STATUS = "pending_dedicated_mdl_validation"
RUN_STATE_SCHEMA = "fireviewer.terrain-run-state.v1"
PHASE_RECEIPT_SCHEMA = "fireviewer.terrain-phase-receipt.v1"
CONTROL_RESULT_SCHEMA = "fireviewer.terrain-control-result.v1"

CRS = "EPSG:2154"
TILE_SIZE_M = 500
HALO_M = 10.0
SOURCE_RESOLUTION_M = 2.0
SAFETY_MARGIN_BYTES = 20 * 1024**3
DEFAULT_PILOT_TILES = 9
PHASES = ("plan", "preflight", "pilot", "produce", "qa", "accept", "cleanup")
PHASE_RECEIPT_NAMES = {
    "plan": "zone-plan.v1.json",
    "preflight": "zone-source-lock.v1.json",
    "pilot": "pilot.receipt.v1.json",
    "produce": "production.receipt.v1.json",
    "qa": "qa.receipt.v1.json",
    "accept": "zone.acceptance.v1.json",
    "cleanup": "cleanup.receipt.v1.json",
}
REQUIRED_DEPENDENCIES = {
    "algorithm",
    "clean_pbr_texture_library",
    "ground_texture_contract",
    "surface_correspondence_contract",
    "surface_correspondence_model",
    "surface_features",
    "terrain_quadtree_contract",
    "toolchain",
}
ATLAS_VISUAL_SCHEMA = "fireviewer.ground-atlas-visual-acceptance.v1"
TILE_BLENDER_VISUAL_SCHEMA = "fireviewer.blender-adaptive-terrain-qa.v2"
ZONE_VISUAL_RECEIPT_NAME = "zone-visual.accepted_blender_visual.v2.json"
ZONE_VISUAL_JOB_RELATIVE_PATH = "proofs/zone-visual/zone-visual-job.v1.json"
ZONE_VISUAL_TECHNICAL_RELATIVE_PATH = (
    "proofs/zone-visual/zone-visual-technical-receipt.v1.json"
)
ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH = (
    "proofs/zone-visual/zone-visual-human-review.template.v1.json"
)
ZONE_VISUAL_REVIEW_RELATIVE_PATH = "proofs/zone-visual/zone-visual-human-review.v1.json"
DEFAULT_GLOBAL_COORDINATOR_ROOT = (
    Path(__file__).resolve().parent.parent
    / "fireviewer-work"
    / "terrain-global-coordinator"
)
REQUIRED_TILE_OUTPUTS = {
    "terrain_lod0",
    "terrain_lod1",
    "terrain_lod2",
    "hag_max_2m",
    "ground_profile_ids",
    "ground_profile_weights",
    "ground_confidence",
    "ground_orientation",
    "tile_package",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TerrainControlError(RuntimeError):
    """Base class for a fail-closed terrain orchestration error."""


class ContractError(TerrainControlError):
    """A public terrain contract is absent, unsupported or incoherent."""


class PhaseGateError(TerrainControlError):
    """The requested phase cannot run because a prerequisite is not accepted."""


class BackendUnavailableError(TerrainControlError):
    """No explicitly injected implementation exists for a heavy phase."""


class StoragePolicyError(TerrainControlError):
    """A path or free-space measurement violates the D:-only policy."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON contract must be an object: {path}")
    return payload


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise ContractError(f"{label} is not a safe stable identifier")
    return value


def _is_d_drive(path: Path) -> bool:
    if os.name != "nt":
        return True
    return path.drive.upper() == "D:"


def _require_d_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not _is_d_drive(resolved):
        raise StoragePolicyError(f"{label} must stay on D:, got {resolved}")
    return resolved


def _reject_c_path_strings(value: Any, label: str = "zone spec") -> None:
    """Reject C:-rooted paths anywhere in a declarative contract.

    Backends must not be able to smuggle a local C: cache through an otherwise
    harmless-looking source request or optional metadata field.
    """

    if isinstance(value, str):
        normalized = value.strip().replace("\\", "/")
        if (
            re.match(r"(?i)^c:", normalized)
            or re.match(r"(?i)^file:(?://(?:localhost)?)?/*c:", normalized)
            or re.match(r"(?i)^//\?/c:", normalized)
        ):
            raise StoragePolicyError(f"{label} contains a forbidden C: path")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_c_path_strings(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_c_path_strings(item, f"{label}[{index}]")


def _resolve_contract_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return _require_d_path(candidate, label)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise StoragePolicyError(f"cannot find an existing parent for {path}")
        current = parent
    return current


@dataclass(frozen=True)
class ZoneDefinition:
    path: Path
    payload: dict[str, Any]
    work_root: Path
    export_root: Path

    @property
    def zone_id(self) -> str:
        return str(self.payload["zone_id"])

    @property
    def revision(self) -> str:
        return str(self.payload["revision"])

    @property
    def run_root(self) -> Path:
        return self.work_root / "terrain-runs" / self.zone_id / self.revision

    @property
    def receipt_root(self) -> Path:
        return self.run_root / "receipts"

    @property
    def state_path(self) -> Path:
        return self.run_root / "run-state.v1.json"

    @property
    def active_zone_path(self) -> Path:
        return self.work_root / "terrain-runs" / "active-zone.v1.json"

    @property
    def process_lock_path(self) -> Path:
        return self.work_root / "terrain-runs" / ".terrainctl.lock"

    def dependency_path(self, name: str) -> Path:
        artifacts = self.payload["dependency_artifacts"]
        return Path(str(artifacts[name]["path"]))


def validate_zone_spec(
    payload: Mapping[str, Any], *, spec_path: Path
) -> ZoneDefinition:
    _reject_c_path_strings(payload)
    allowed_top_level = {
        "schema",
        "zone_id",
        "revision",
        "crs",
        "bounds_l93_m",
        "tile_size_m",
        "halo_m",
        "source_resolution_m",
        "dependencies",
        "dependency_artifacts",
        "source_requests",
        "pilot",
        "workspace",
        "storage",
    }
    unknown_top_level = sorted(set(payload) - allowed_top_level)
    if unknown_top_level:
        raise ContractError(
            "unknown zone-spec field(s): " + ", ".join(unknown_top_level)
        )
    if payload.get("schema") != ZONE_SPEC_SCHEMA:
        raise ContractError(
            f"unsupported zone schema: {payload.get('schema')!r}; "
            f"expected {ZONE_SPEC_SCHEMA!r}"
        )
    zone_id = _require_safe_id(payload.get("zone_id"), "zone_id")
    revision = _require_safe_id(payload.get("revision"), "revision")
    if payload.get("crs") != CRS:
        raise ContractError(f"crs must be exactly {CRS}")

    bounds = payload.get("bounds_l93_m")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ContractError("bounds_l93_m must contain west, south, east and north")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in bounds
    ):
        raise ContractError("bounds_l93_m values must be finite numbers")
    west, south, east, north = (float(value) for value in bounds)
    width = east - west
    height = north - south
    if width <= 0 or not math.isclose(width, height, abs_tol=1e-9):
        raise ContractError("bounds_l93_m must describe a non-empty square")
    if any(
        not math.isclose(value / TILE_SIZE_M, round(value / TILE_SIZE_M), abs_tol=1e-9)
        for value in (west, south, east, north)
    ):
        raise ContractError(
            "bounds_l93_m must align to the global 500 m Lambert-93 grid"
        )

    pilot_contract = payload.get("pilot")
    if pilot_contract is not None:
        if not isinstance(pilot_contract, dict) or set(pilot_contract) != {
            "regression_tile_id"
        }:
            raise ContractError(
                "pilot must contain only an optional regression_tile_id"
            )
        regression_tile_id = _require_safe_id(
            pilot_contract.get("regression_tile_id"),
            "pilot.regression_tile_id",
        )
        expected_tile_ids = {
            _tile_id(tile_west, tile_south)
            for tile_south in range(int(round(south)), int(round(north)), TILE_SIZE_M)
            for tile_west in range(int(round(west)), int(round(east)), TILE_SIZE_M)
        }
        if regression_tile_id not in expected_tile_ids:
            raise ContractError(
                "pilot.regression_tile_id must identify a tile inside the zone"
            )

    if payload.get("tile_size_m") != TILE_SIZE_M:
        raise ContractError(f"tile_size_m must be exactly {TILE_SIZE_M}")
    if not math.isclose(float(payload.get("halo_m", math.nan)), HALO_M, abs_tol=1e-12):
        raise ContractError(f"halo_m must be exactly {HALO_M:g}")
    if not math.isclose(
        float(payload.get("source_resolution_m", math.nan)),
        SOURCE_RESOLUTION_M,
        abs_tol=1e-12,
    ):
        raise ContractError(
            f"source_resolution_m must be exactly {SOURCE_RESOLUTION_M:g}"
        )

    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, dict):
        raise ContractError("dependencies must be an object of locked SHA-256 values")
    missing_dependencies = sorted(REQUIRED_DEPENDENCIES - dependencies.keys())
    unexpected_dependencies = sorted(dependencies.keys() - REQUIRED_DEPENDENCIES)
    if missing_dependencies or unexpected_dependencies:
        details = []
        if missing_dependencies:
            details.append("missing=" + ", ".join(missing_dependencies))
        if unexpected_dependencies:
            details.append("unexpected=" + ", ".join(unexpected_dependencies))
        raise ContractError(
            "locked dependencies must match the surface contract exactly: "
            + "; ".join(details)
        )
    for name, digest in dependencies.items():
        _require_safe_id(name, f"dependency name {name!r}")
        _require_hash(digest, f"dependencies.{name}")

    dependency_artifacts = payload.get("dependency_artifacts")
    if not isinstance(dependency_artifacts, dict):
        raise ContractError(
            "dependency_artifacts must bind every dependency hash to a D:-only file"
        )
    if set(dependency_artifacts) != set(dependencies):
        raise ContractError("dependency_artifacts keys must exactly match dependencies")
    dependency_paths: dict[str, dict[str, str]] = {}
    base = spec_path.resolve().parent
    for name, raw_record in dependency_artifacts.items():
        if not isinstance(raw_record, dict) or set(raw_record) != {"path", "sha256"}:
            raise ContractError(
                f"dependency_artifacts.{name} must contain only path and sha256"
            )
        digest = _require_hash(
            raw_record.get("sha256"), f"dependency_artifacts.{name}.sha256"
        )
        if digest != dependencies[name]:
            raise ContractError(
                f"dependency_artifacts.{name}.sha256 differs from dependencies.{name}"
            )
        resolved_dependency = _resolve_contract_path(
            raw_record.get("path"),
            base=base,
            label=f"dependency_artifacts.{name}.path",
        )
        dependency_paths[name] = {"path": str(resolved_dependency), "sha256": digest}

    requests = payload.get("source_requests")
    if not isinstance(requests, list) or not requests:
        raise ContractError("source_requests must be a non-empty list")
    request_keys: set[tuple[str, str]] = set()
    products: set[str] = set()
    paired_requests: dict[str, dict[str, dict[str, Any]]] = {}
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ContractError(f"source_requests[{index}] must be an object")
        unknown_request_fields = sorted(
            set(request)
            - {
                "id",
                "product",
                "request",
                "expected_sha256",
                "expected_byte_count",
                "source_revision_id",
                "license",
                "pilot_score",
            }
        )
        if unknown_request_fields:
            raise ContractError(
                f"source_requests[{index}] has unknown field(s): "
                + ", ".join(unknown_request_fields)
            )
        source_id = _require_safe_id(request.get("id"), f"source_requests[{index}].id")
        product = request.get("product")
        if product not in {"mnt", "mns", "orthophoto"}:
            raise ContractError(
                f"source_requests[{index}].product must be 'mnt', 'mns' or 'orthophoto'"
            )
        source_request = request.get("request")
        if not isinstance(source_request, dict) or not source_request:
            raise ContractError(f"source_requests[{index}].request must be an object")
        if product == "orthophoto":
            common_orthophoto_fields = {
                "service_kind",
                "service_url",
                "layer",
                "core_bounds_l93_m",
                "resolution_m",
                "halo_m",
                "image_format",
                "maximum_download_bytes",
            }
            service_kind = source_request.get("service_kind")
            expected_orthophoto_fields = (
                common_orthophoto_fields
                if service_kind == "wms"
                else common_orthophoto_fields | {"style", "wmts_matrix"}
            )
            if service_kind not in {"wms", "wmts"} or set(source_request) != (
                expected_orthophoto_fields
            ):
                raise ContractError(
                    f"source_requests[{index}] has an invalid canonical "
                    "orthophoto request"
                )
            if source_request.get("resolution_m") != 1:
                raise ContractError("orthophoto resolution_m must be exactly 1")
            if source_request.get("halo_m") != HALO_M:
                raise ContractError(f"orthophoto halo_m must be exactly {HALO_M:g}")
            if source_request.get("image_format") not in {"image/png", "image/jpeg"}:
                raise ContractError("orthophoto image_format is unsupported")
            maximum_download_bytes = source_request.get("maximum_download_bytes")
            if (
                isinstance(maximum_download_bytes, bool)
                or not isinstance(maximum_download_bytes, int)
                or maximum_download_bytes <= 0
            ):
                raise ContractError(
                    "orthophoto maximum_download_bytes must be positive"
                )
            if service_kind == "wmts":
                style = source_request.get("style")
                matrix = source_request.get("wmts_matrix")
                if not isinstance(style, str) or not style.strip():
                    raise ContractError("orthophoto WMTS style must be non-empty")
                matrix_fields = {
                    "matrix_set",
                    "matrix",
                    "top_left_l93_m",
                    "tile_width_px",
                    "tile_height_px",
                    "matrix_width",
                    "matrix_height",
                    "resolution_m",
                }
                if not isinstance(matrix, dict) or set(matrix) != matrix_fields:
                    raise ContractError("orthophoto WMTS matrix is invalid")
                for field in ("matrix_set", "matrix"):
                    _require_safe_id(matrix.get(field), f"wmts_matrix.{field}")
                top_left = matrix.get("top_left_l93_m")
                if (
                    not isinstance(top_left, list)
                    or len(top_left) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or not float(value).is_integer()
                        for value in top_left
                    )
                ):
                    raise ContractError("orthophoto WMTS top-left is invalid")
                for field in (
                    "tile_width_px",
                    "tile_height_px",
                    "matrix_width",
                    "matrix_height",
                ):
                    value = matrix.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                    ):
                        raise ContractError(f"orthophoto WMTS {field} is invalid")
                if matrix.get("resolution_m") != 1:
                    raise ContractError(
                        "orthophoto WMTS matrix resolution_m must be exactly 1"
                    )
        elif set(source_request) != {
            "service_url",
            "layer",
            "core_bounds_l93_m",
            "resolution_m",
        }:
            raise ContractError(
                f"source_requests[{index}].request must contain only "
                "service_url, layer, core_bounds_l93_m and resolution_m"
            )
        if product in {"mnt", "mns"} and source_request.get("resolution_m") != 2:
            raise ContractError(
                f"source_requests[{index}] elevation resolution_m must be exactly 2"
            )
        service_url = source_request.get("service_url")
        if not isinstance(service_url, str) or not service_url.strip():
            raise ContractError(
                f"source_requests[{index}].request.service_url is required"
            )
        parsed_url = urlsplit(service_url)
        allowed_schemes = {"https"} if product == "orthophoto" else {"https", "file"}
        if parsed_url.scheme not in allowed_schemes:
            raise ContractError(
                f"source_requests[{index}] must use an HTTPS or file source"
            )
        if product != "orthophoto" and "ortho" in service_url.casefold():
            raise ContractError("orthophoto sources are forbidden")
        layer = source_request.get("layer")
        if (
            not isinstance(layer, str)
            or not layer.strip()
            or (product != "orthophoto" and "ortho" in layer.casefold())
        ):
            raise ContractError(f"source_requests[{index}].request.layer is invalid")
        core_bounds = source_request.get("core_bounds_l93_m")
        if (
            not isinstance(core_bounds, list)
            or len(core_bounds) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in core_bounds
            )
        ):
            raise ContractError(
                f"source_requests[{index}].request.core_bounds_l93_m is invalid"
            )
        source_west, source_south, source_east, source_north = (
            float(value) for value in core_bounds
        )
        source_width = source_east - source_west
        if (
            source_width <= 0
            or not math.isclose(source_width, TILE_SIZE_M, abs_tol=1e-9)
            or not math.isclose(source_width, source_north - source_south, abs_tol=1e-9)
            or any(
                not math.isclose(
                    value / TILE_SIZE_M, round(value / TILE_SIZE_M), abs_tol=1e-9
                )
                for value in (
                    source_west,
                    source_south,
                    source_east,
                    source_north,
                )
            )
            or source_west < west
            or source_south < south
            or source_east > east
            or source_north > north
        ):
            raise ContractError(
                f"source_requests[{index}] core must be one exact, 500 m aligned tile"
            )
        has_expected_hash = "expected_sha256" in request
        has_expected_bytes = "expected_byte_count" in request
        if has_expected_hash != has_expected_bytes:
            raise ContractError(
                f"source_requests[{index}] must declare both expected_sha256 "
                "and expected_byte_count, or neither"
            )
        if parsed_url.scheme == "file" and not has_expected_hash:
            raise ContractError(
                f"source_requests[{index}] file sources require expected identity"
            )
        if has_expected_hash:
            _require_hash(
                request.get("expected_sha256"),
                f"source_requests[{index}].expected_sha256",
            )
            expected_byte_count = request.get("expected_byte_count")
            if (
                isinstance(expected_byte_count, bool)
                or not isinstance(expected_byte_count, int)
                or expected_byte_count <= 0
            ):
                raise ContractError(
                    f"source_requests[{index}].expected_byte_count must be positive"
                )
        source_revision_id = request.get("source_revision_id")
        if parsed_url.scheme == "https":
            _require_safe_id(
                source_revision_id,
                f"source_requests[{index}].source_revision_id",
            )
        elif source_revision_id is not None:
            _require_safe_id(
                source_revision_id,
                f"source_requests[{index}].source_revision_id",
            )
        license_name = request.get("license")
        if not isinstance(license_name, str) or not license_name.strip():
            raise ContractError(f"source_requests[{index}].license must be non-empty")
        pilot_score = request.get("pilot_score", 0.0)
        if (
            isinstance(pilot_score, bool)
            or not isinstance(pilot_score, (int, float))
            or not math.isfinite(float(pilot_score))
        ):
            raise ContractError(f"source_requests[{index}].pilot_score must be finite")
        key = (source_id, product)
        if key in request_keys:
            raise ContractError(f"duplicate source request: {source_id}/{product}")
        request_keys.add(key)
        products.add(product)
        paired_requests.setdefault(source_id, {})[str(product)] = request
    if products != {"mnt", "mns", "orthophoto"}:
        raise ContractError(
            "source_requests must include MNT, MNS and temporary orthophoto products"
        )
    for source_id, pair in paired_requests.items():
        if set(pair) != {"mnt", "mns", "orthophoto"}:
            raise ContractError(
                f"source band {source_id!r} must contain one MNT, one MNS and one "
                "temporary orthophoto"
            )
        if (
            len(
                {
                    tuple(pair[product]["request"]["core_bounds_l93_m"])
                    for product in ("mnt", "mns", "orthophoto")
                }
            )
            != 1
        ):
            raise ContractError(
                f"source band {source_id!r} has different MNT/MNS/orthophoto cores"
            )

    source_by_core: dict[tuple[float, float, float, float], str] = {}
    for source_id, pair in paired_requests.items():
        source_core = tuple(
            float(value) for value in pair["mnt"]["request"]["core_bounds_l93_m"]
        )
        if source_core in source_by_core:
            raise ContractError(
                "each planned tile must have exactly one MNT/MNS source pair; "
                f"core {source_core} is duplicated by {source_by_core[source_core]!r} "
                f"and {source_id!r}"
            )
        source_by_core[source_core] = source_id
    expected_cores = {
        (
            float(tile_west),
            float(tile_south),
            float(tile_west + TILE_SIZE_M),
            float(tile_south + TILE_SIZE_M),
        )
        for tile_south in range(int(round(south)), int(round(north)), TILE_SIZE_M)
        for tile_west in range(int(round(west)), int(round(east)), TILE_SIZE_M)
    }
    if set(source_by_core) != expected_cores:
        missing_cores = sorted(expected_cores - set(source_by_core))
        extra_cores = sorted(set(source_by_core) - expected_cores)
        detail = []
        if missing_cores:
            detail.append(f"missing={missing_cores[:3]}")
        if extra_cores:
            detail.append(f"unexpected={extra_cores[:3]}")
        raise ContractError(
            "source pair cores must exactly match the planned tile grid; "
            + ", ".join(detail)
        )

    workspace = payload.get("workspace")
    if not isinstance(workspace, dict):
        raise ContractError("workspace must define D:-only work_root and export_root")
    if set(workspace) != {"work_root", "export_root"}:
        raise ContractError("workspace must contain only work_root and export_root")
    work_root = _resolve_contract_path(
        workspace.get("work_root"), base=base, label="workspace.work_root"
    )
    export_root = _resolve_contract_path(
        workspace.get("export_root"), base=base, label="workspace.export_root"
    )
    if (
        work_root == export_root
        or _is_relative_to(work_root, export_root)
        or _is_relative_to(export_root, work_root)
    ):
        raise ContractError("work_root and export_root must be separate trees")

    storage = payload.get("storage")
    if not isinstance(storage, dict):
        raise ContractError("storage must declare estimated_peak_bytes")
    if set(storage) != {"estimated_peak_bytes"}:
        raise ContractError("storage must contain only estimated_peak_bytes")
    estimated_peak = storage.get("estimated_peak_bytes")
    if (
        isinstance(estimated_peak, bool)
        or not isinstance(estimated_peak, int)
        or estimated_peak <= 0
    ):
        raise ContractError("storage.estimated_peak_bytes must be a positive integer")

    normalized = json.loads(canonical_json_bytes(dict(payload)).decode("utf-8"))
    normalized["zone_id"] = zone_id
    normalized["revision"] = revision
    normalized["bounds_l93_m"] = [west, south, east, north]
    normalized["dependency_artifacts"] = dependency_paths
    normalized["source_requests"] = sorted(
        normalized["source_requests"], key=lambda item: (item["product"], item["id"])
    )
    return ZoneDefinition(
        path=_require_d_path(spec_path, "--zone"),
        payload=normalized,
        work_root=work_root,
        export_root=export_root,
    )


def load_zone_spec(path: Path) -> ZoneDefinition:
    resolved = _require_d_path(path, "--zone")
    return validate_zone_spec(_read_json(resolved), spec_path=resolved)


def _morton_code(x: int, y: int) -> int:
    result = 0
    bit = 0
    while x or y:
        result |= (x & 1) << (2 * bit)
        result |= (y & 1) << (2 * bit + 1)
        x >>= 1
        y >>= 1
        bit += 1
    return result


def _tile_id(west: int, south: int) -> str:
    return f"x{west:06d}_y{south:07d}_s{TILE_SIZE_M}"


def _canonical_spec_for_recipe(zone: ZoneDefinition) -> dict[str, Any]:
    payload = dict(zone.payload)
    payload.pop("workspace", None)
    payload.pop("storage", None)
    payload.pop("dependency_artifacts", None)
    return payload


def build_zone_plan(zone: ZoneDefinition) -> dict[str, Any]:
    west, south, east, north = (
        int(round(value)) for value in zone.payload["bounds_l93_m"]
    )
    columns = (east - west) // TILE_SIZE_M
    rows = (north - south) // TILE_SIZE_M
    indexed: list[tuple[int, int, int, dict[str, Any]]] = []
    by_grid: dict[tuple[int, int], str] = {}
    for y_index in range(rows):
        for x_index in range(columns):
            tile_west = west + x_index * TILE_SIZE_M
            tile_south = south + y_index * TILE_SIZE_M
            identifier = _tile_id(tile_west, tile_south)
            by_grid[(x_index, y_index)] = identifier
            indexed.append(
                (
                    _morton_code(x_index, y_index),
                    y_index,
                    x_index,
                    {
                        "id": identifier,
                        "grid": [x_index, y_index],
                        "bounds_l93_m": [
                            tile_west,
                            tile_south,
                            tile_west + TILE_SIZE_M,
                            tile_south + TILE_SIZE_M,
                        ],
                        "processing_bounds_l93_m": [
                            tile_west - HALO_M,
                            tile_south - HALO_M,
                            tile_west + TILE_SIZE_M + HALO_M,
                            tile_south + TILE_SIZE_M + HALO_M,
                        ],
                    },
                )
            )
    tiles = [record[3] for record in sorted(indexed)]
    seams: list[dict[str, str]] = []
    for y_index in range(rows):
        for x_index in range(columns):
            current = by_grid[(x_index, y_index)]
            if x_index + 1 < columns:
                neighbour = by_grid[(x_index + 1, y_index)]
                seams.append(
                    {"id": f"{current}--{neighbour}", "a": current, "b": neighbour}
                )
            if y_index + 1 < rows:
                neighbour = by_grid[(x_index, y_index + 1)]
                seams.append(
                    {"id": f"{current}--{neighbour}", "a": current, "b": neighbour}
                )
    seams.sort(key=lambda item: item["id"])

    plan_basis = {
        "zone": {
            "zone_id": zone.zone_id,
            "revision": zone.revision,
            "crs": CRS,
            "bounds_l93_m": [west, south, east, north],
        },
        "grid": {
            "tile_size_m": TILE_SIZE_M,
            "halo_m": HALO_M,
            "columns": columns,
            "rows": rows,
            "order": "morton",
        },
        "tiles": tiles,
        "seams": seams,
        "source_requests": zone.payload["source_requests"],
        "dependencies": zone.payload["dependencies"],
    }
    recipe_id = canonical_sha256(
        {
            "schema": "fireviewer.terrain-recipe.v1",
            "zone_spec": _canonical_spec_for_recipe(zone),
            "plan": plan_basis,
        }
    )
    plan = {
        "schema": ZONE_PLAN_SCHEMA,
        "recipe_id": recipe_id,
        **plan_basis,
        "pilot": {
            "default_max_tiles": min(DEFAULT_PILOT_TILES, len(tiles)),
            "selection": "contiguous_relief_and_semantic_maximization",
            "regression_tile_id": (
                zone.payload.get("pilot", {}).get("regression_tile_id")
                if isinstance(zone.payload.get("pilot"), dict)
                else None
            ),
        },
        "storage": {
            "estimated_peak_bytes": zone.payload["storage"]["estimated_peak_bytes"],
            "safety_margin_bytes": SAFETY_MARGIN_BYTES,
            "required_free_bytes": zone.payload["storage"]["estimated_peak_bytes"]
            + SAFETY_MARGIN_BYTES,
        },
        "summary": {
            "tile_count": len(tiles),
            "seam_count": len(seams),
            "source_request_count": len(zone.payload["source_requests"]),
        },
    }
    plan["plan_id"] = canonical_sha256(plan)
    return plan


def validate_zone_plan(plan: Mapping[str, Any], zone: ZoneDefinition) -> dict[str, Any]:
    expected = build_zone_plan(zone)
    if dict(plan) != expected:
        raise ContractError(
            "zone plan differs from the canonical plan for this zone spec"
        )
    return expected


def _source_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("id")), str(record.get("product"))


def build_source_lock(
    plan: Mapping[str, Any], backend_result: Mapping[str, Any]
) -> dict[str, Any]:
    estimated_peak_bytes = _require_nonnegative_int(
        backend_result.get("estimated_peak_bytes"), "preflight.estimated_peak_bytes"
    )
    if estimated_peak_bytes < int(plan["storage"]["estimated_peak_bytes"]):
        raise ContractError(
            "preflight peak estimate is below the zone manifest estimate"
        )
    records = backend_result.get("sources")
    if not isinstance(records, list):
        raise ContractError("preflight backend must return a sources list")
    expected = {
        _source_key(request): request
        for request in plan["source_requests"]  # type: ignore[index]
    }
    actual: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ContractError(f"sources[{index}] must be an object")
        key = _source_key(raw_record)
        if key in actual:
            raise ContractError(f"duplicate locked source: {key[0]}/{key[1]}")
        if key not in expected:
            raise ContractError(f"unexpected locked source: {key[0]}/{key[1]}")
        expected_request = expected[key]
        source_revision_id = raw_record.get("source_revision_id")
        if source_revision_id != expected_request.get("source_revision_id"):
            raise ContractError(f"locked source revision changed for {key[0]}/{key[1]}")
        if source_revision_id is not None:
            _require_safe_id(source_revision_id, f"sources[{index}].source_revision_id")
        identity_status = raw_record.get("identity_status")
        expected_scheme = urlsplit(
            str(expected_request["request"]["service_url"])
        ).scheme
        expected_identity_status = (
            "observed_local_file"
            if expected_scheme == "file"
            else (
                "expected_identity_locked"
                if "expected_sha256" in expected_request
                else "revision_locked"
            )
        )
        if identity_status != expected_identity_status:
            raise ContractError(f"invalid source identity status for {key[0]}/{key[1]}")
        raw_digest = raw_record.get("sha256")
        digest = (
            _require_hash(raw_digest, f"sources[{index}].sha256")
            if raw_digest is not None
            else None
        )
        byte_count = raw_record.get("byte_count")
        if byte_count is not None and (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise ContractError(f"sources[{index}].byte_count must be positive or null")
        if (digest is None) != (byte_count is None):
            raise ContractError(
                f"sources[{index}] hash and byte_count must be both known or both null"
            )
        if expected_identity_status != "revision_locked" and digest is None:
            raise ContractError(
                f"sources[{index}] locked identity is unexpectedly absent"
            )
        request_digest = canonical_sha256(expected_request["request"])
        if raw_record.get("request_sha256") != request_digest:
            raise ContractError(f"locked source request changed for {key[0]}/{key[1]}")
        expected_digest = expected_request.get("expected_sha256")
        if expected_digest is not None and digest != expected_digest:
            raise ContractError(
                f"locked source hash differs from expected hash for {key[0]}/{key[1]}"
            )
        license_name = raw_record.get("license")
        if not isinstance(license_name, str) or not license_name.strip():
            raise ContractError(f"sources[{index}].license must be non-empty")
        actual[key] = {
            "id": key[0],
            "product": key[1],
            "request_sha256": request_digest,
            "source_revision_id": source_revision_id,
            "identity_status": identity_status,
            "sha256": digest,
            "byte_count": byte_count,
            "license": license_name,
        }
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ContractError(
            "missing locked sources: "
            + ", ".join(f"{source_id}/{product}" for source_id, product in missing)
        )
    ordered = [
        actual[key] for key in sorted(actual, key=lambda item: (item[1], item[0]))
    ]
    raw_dependency_proofs = backend_result.get("dependency_proofs")
    if not isinstance(raw_dependency_proofs, dict):
        raise ContractError("preflight backend must return dependency_proofs")
    expected_dependencies = plan.get("dependencies")
    if not isinstance(expected_dependencies, dict):
        raise ContractError("zone plan dependencies are absent")
    if set(raw_dependency_proofs) != set(expected_dependencies):
        raise ContractError("dependency proofs must exactly match the zone plan")
    dependency_proofs: dict[str, dict[str, Any]] = {}
    for name, raw_proof in sorted(raw_dependency_proofs.items()):
        if not isinstance(raw_proof, dict) or set(raw_proof) != {
            "file_name",
            "sha256",
            "schema",
            "status",
        }:
            raise ContractError(
                f"dependency_proofs.{name} must contain file_name, sha256, schema and status"
            )
        file_name = raw_proof.get("file_name")
        if (
            not isinstance(file_name, str)
            or not file_name
            or PureWindowsPath(file_name).name != file_name
        ):
            raise ContractError(f"dependency_proofs.{name}.file_name must be portable")
        proof_hash = _require_hash(
            raw_proof.get("sha256"), f"dependency_proofs.{name}.sha256"
        )
        if proof_hash != expected_dependencies[name]:
            raise ContractError(f"dependency proof hash mismatch for {name}")
        schema = raw_proof.get("schema")
        status = raw_proof.get("status")
        if schema is not None and not isinstance(schema, str):
            raise ContractError(
                f"dependency_proofs.{name}.schema must be a string or null"
            )
        if status is not None and not isinstance(status, str):
            raise ContractError(
                f"dependency_proofs.{name}.status must be a string or null"
            )
        if name == "clean_pbr_texture_library" and (
            schema != "fireviewer.clean-pbr-texture-library.v1"
            or status != "accepted_clean_pbr_library"
        ):
            raise ContractError("clean PBR library dependency is not accepted")
        dependency_proofs[name] = {
            "file_name": file_name,
            "sha256": proof_hash,
            "schema": schema,
            "status": status,
        }

    recipe_build_id = canonical_sha256(
        {
            "schema": "fireviewer.terrain-recipe-build.v1",
            "recipe_id": plan["recipe_id"],
            "sources": ordered,
            "dependency_proofs": dependency_proofs,
        }
    )
    return {
        "schema": SOURCE_LOCK_SCHEMA,
        "recipe_id": plan["recipe_id"],
        "plan_id": plan["plan_id"],
        "recipe_build_id": recipe_build_id,
        "identity_status": (
            "fully_observed"
            if all(source["sha256"] is not None for source in ordered)
            else "provisional_source_revision"
        ),
        "estimated_peak_bytes": estimated_peak_bytes,
        "sources": ordered,
        "dependency_proofs": dependency_proofs,
    }


def validate_source_lock(
    source_lock: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    if source_lock.get("schema") != SOURCE_LOCK_SCHEMA:
        raise ContractError("unsupported source lock schema")
    rebuilt = build_source_lock(
        plan,
        {
            "sources": source_lock.get("sources"),
            "dependency_proofs": source_lock.get("dependency_proofs"),
            "estimated_peak_bytes": source_lock.get("estimated_peak_bytes"),
        },
    )
    if dict(source_lock) != rebuilt:
        raise ContractError(
            "source lock is not canonical or its recipe_build_id has changed"
        )
    return rebuilt


def _merkle_root_sha256(leaves: Sequence[str]) -> str:
    if not leaves:
        raise ContractError("source Merkle tree must contain at least one leaf")
    level = [bytes.fromhex(_require_hash(value, "Merkle leaf")) for value in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"\x01" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def build_final_source_identity(
    plan: Mapping[str, Any],
    *,
    recipe_build_id: str,
    observed_sources: Any,
    tile_receipts: Any,
) -> dict[str, Any]:
    """Validate all observed source bytes and derive the accepted build identity."""

    _require_hash(recipe_build_id, "recipe_build_id")
    expected_sources = {
        _source_key(request): request
        for request in plan["source_requests"]  # type: ignore[index]
    }
    if not isinstance(observed_sources, list):
        raise ContractError("observed_sources must be a list")
    validated_sources: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_record in enumerate(observed_sources):
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "id",
            "product",
            "request_sha256",
            "source_revision_id",
            "sha256",
            "byte_count",
            "license",
        }:
            raise ContractError(f"observed_sources[{index}] is non-canonical")
        key = _source_key(raw_record)
        expected = expected_sources.get(key)
        if expected is None or key in validated_sources:
            raise ContractError(
                f"unexpected or duplicate observed source: {key[0]}/{key[1]}"
            )
        if raw_record.get("request_sha256") != canonical_sha256(expected["request"]):
            raise ContractError(
                f"observed source request changed for {key[0]}/{key[1]}"
            )
        if raw_record.get("source_revision_id") != expected.get("source_revision_id"):
            raise ContractError(
                f"observed source revision changed for {key[0]}/{key[1]}"
            )
        digest = _require_hash(
            raw_record.get("sha256"), f"observed_sources[{index}].sha256"
        )
        byte_count = raw_record.get("byte_count")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise ContractError(
                f"observed_sources[{index}].byte_count must be positive"
            )
        if expected.get("expected_sha256") not in {None, digest}:
            raise ContractError(
                f"observed source hash differs from expected for {key[0]}/{key[1]}"
            )
        if expected.get("expected_byte_count") not in {None, byte_count}:
            raise ContractError(
                f"observed source size differs from expected for {key[0]}/{key[1]}"
            )
        license_name = raw_record.get("license")
        if license_name != expected.get("license"):
            raise ContractError(
                f"observed source license changed for {key[0]}/{key[1]}"
            )
        validated_sources[key] = dict(raw_record)
    if set(validated_sources) != set(expected_sources):
        raise ContractError("observed source set does not cover the canonical plan")
    ordered_sources = [
        validated_sources[key]
        for key in sorted(validated_sources, key=lambda item: (item[1], item[0]))
    ]
    source_leaves = [
        canonical_sha256({"schema": "fireviewer.terrain-source-leaf.v1", **record})
        for record in ordered_sources
    ]
    source_merkle_root_sha256 = _merkle_root_sha256(source_leaves)

    if not isinstance(tile_receipts, list):
        raise ContractError("final tile receipt identities must be a list")
    expected_tiles = {tile["id"] for tile in plan["tiles"]}  # type: ignore[index]
    validated_tiles: dict[str, dict[str, str]] = {}
    for index, raw_record in enumerate(tile_receipts):
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "tile_id",
            "tile_done_sha256",
        }:
            raise ContractError(f"tile_receipts[{index}] is non-canonical")
        tile_id = _require_safe_id(raw_record.get("tile_id"), "tile receipt tile_id")
        if tile_id not in expected_tiles or tile_id in validated_tiles:
            raise ContractError(f"unexpected or duplicate final tile: {tile_id}")
        validated_tiles[tile_id] = {
            "tile_id": tile_id,
            "tile_done_sha256": _require_hash(
                raw_record.get("tile_done_sha256"),
                f"tile receipt {tile_id} sha256",
            ),
        }
    if set(validated_tiles) != expected_tiles:
        raise ContractError("final tile receipt set does not cover the zone")
    ordered_tiles = [validated_tiles[tile_id] for tile_id in sorted(validated_tiles)]
    build_id = canonical_sha256(
        {
            "schema": "fireviewer.terrain-build.v1",
            "recipe_build_id": recipe_build_id,
            "source_merkle_root_sha256": source_merkle_root_sha256,
            "sources": ordered_sources,
            "tiles": ordered_tiles,
        }
    )
    return {
        "recipe_build_id": recipe_build_id,
        "build_id": build_id,
        "source_merkle_root_sha256": source_merkle_root_sha256,
        "observed_sources": ordered_sources,
        "tile_receipts": ordered_tiles,
    }


def _validate_relative_artifact(record: Mapping[str, Any], label: str) -> None:
    if set(record) != {"path", "byte_count", "sha256"}:
        raise ContractError(f"{label} must contain only path, byte_count and sha256")
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise ContractError(f"{label}.path must be a non-empty relative path")
    pure = PureWindowsPath(path)
    if pure.is_absolute() or pure.drive or ".." in pure.parts:
        raise ContractError(
            f"{label}.path must stay relative and cannot traverse parents"
        )
    byte_count = record.get("byte_count")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise ContractError(f"{label}.byte_count must be a non-negative integer")
    _require_hash(record.get("sha256"), f"{label}.sha256")


def _validate_stitch_catalog(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "encoding",
        "edge_order",
        "edge_mask_bits",
        "available_masks",
        "lods",
    }:
        raise ContractError(f"{label} is absent or has unknown fields")
    if (
        value.get("encoding") != "fvtq-base-remove-add.v1"
        or value.get("edge_order") != ["west", "east", "south", "north"]
        or value.get("edge_mask_bits") != {"west": 1, "east": 2, "south": 4, "north": 8}
        or value.get("available_masks") != list(range(16))
    ):
        raise ContractError(f"{label} has an unsupported FVTQ stitch contract")
    lods = value.get("lods")
    if not isinstance(lods, dict) or set(lods) != {"lod0", "lod1", "lod2"}:
        raise ContractError(f"{label}.lods must contain exactly LOD0, LOD1 and LOD2")
    for lod, maximum_error_mm in enumerate((500, 2_000, 8_000)):
        records = lods.get(f"lod{lod}")
        if not isinstance(records, list) or len(records) != 16:
            raise ContractError(f"{label}.lod{lod} must contain masks 0..15")
        for mask, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {
                "mask",
                "triangle_count",
                "triangle_indices_sha256",
                "maximum_error_mm",
                "effective_edge_signatures",
            }:
                raise ContractError(f"{label}.lod{lod}[{mask}] is invalid")
            if record.get("mask") != mask:
                raise ContractError(f"{label}.lod{lod} masks are not ordered 0..15")
            if (
                _require_nonnegative_int(
                    record.get("triangle_count"),
                    f"{label}.lod{lod}[{mask}].triangle_count",
                )
                <= 0
            ):
                raise ContractError(f"{label}.lod{lod}[{mask}] has no triangles")
            _require_hash(
                record.get("triangle_indices_sha256"),
                f"{label}.lod{lod}[{mask}].triangle_indices_sha256",
            )
            if (
                _require_nonnegative_int(
                    record.get("maximum_error_mm"),
                    f"{label}.lod{lod}[{mask}].maximum_error_mm",
                )
                > maximum_error_mm
            ):
                raise ContractError(
                    f"{label}.lod{lod}[{mask}] exceeds {maximum_error_mm} mm"
                )
            signatures = record.get("effective_edge_signatures")
            if not isinstance(signatures, list) or len(signatures) != 4:
                raise ContractError(
                    f"{label}.lod{lod}[{mask}] must sign all four edges"
                )
            for edge, digest in zip(value["edge_order"], signatures, strict=True):
                _require_hash(digest, f"{label}.lod{lod}[{mask}].{edge}")
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def validate_tile_receipt(
    receipt: Mapping[str, Any],
    *,
    recipe_id: str,
    recipe_build_id: str,
    tile_ids: set[str],
) -> dict[str, Any]:
    if receipt.get("schema") != TILE_RECEIPT_SCHEMA:
        raise ContractError("unsupported tile receipt schema")
    if set(receipt) != {
        "schema",
        "tile_id",
        "recipe_id",
        "recipe_build_id",
        "normal_halo_sha256",
        "stitch_variants",
        "inputs",
        "ground_material",
        "surface_mapping",
        "outputs",
    }:
        raise ContractError("tile receipt has unknown or missing fields")
    tile_id = receipt.get("tile_id")
    if tile_id not in tile_ids:
        raise ContractError(f"tile receipt references unknown tile {tile_id!r}")
    if (
        receipt.get("recipe_id") != recipe_id
        or receipt.get("recipe_build_id") != recipe_build_id
    ):
        raise ContractError(f"tile receipt identity mismatch for {tile_id}")
    _require_hash(
        receipt.get("normal_halo_sha256"),
        f"tile receipt {tile_id} normal_halo_sha256",
    )
    _validate_stitch_catalog(
        receipt.get("stitch_variants"), f"tile receipt {tile_id} stitch_variants"
    )
    for collection in ("inputs", "outputs"):
        artifacts = receipt.get(collection)
        if not isinstance(artifacts, dict) or not artifacts:
            raise ContractError(f"tile receipt {tile_id} has no {collection}")
        for name, record in artifacts.items():
            if not isinstance(record, dict):
                raise ContractError(
                    f"tile receipt {tile_id} {collection}.{name} is invalid"
                )
            _validate_relative_artifact(record, f"{collection}.{name}")
    surface_input = receipt["inputs"].get("surface_correspondence")
    if (
        not isinstance(surface_input, dict)
        or surface_input.get("path") != "surface-correspondence.json"
    ):
        raise ContractError(
            f"tile receipt {tile_id} has no sealed surface correspondence input"
        )
    ground_material = receipt.get("ground_material")
    expected_material_fields = {
        "schema",
        "zone_path",
        "contract_sha256",
        "source_library_schema",
        "source_library_manifest_sha256",
        "source_library_identity_sha256",
        "source_library_content_sha256",
        "texture_contract_sha256",
        "runtime_shader",
        "runtime_atlas_sha256",
        "material_layer_sha256",
        "visual_acceptance",
    }
    if (
        not isinstance(ground_material, dict)
        or set(ground_material) != expected_material_fields
        or ground_material.get("schema") != "fireviewer.ground-material-contract.v2"
    ):
        raise ContractError(f"tile receipt {tile_id} has no ground material identity")
    material_path = PureWindowsPath(str(ground_material.get("zone_path", "")))
    if (
        not str(ground_material.get("zone_path", ""))
        or material_path.is_absolute()
        or material_path.drive
        or ".." in material_path.parts
    ):
        raise ContractError(
            f"tile receipt {tile_id} ground material path escapes the zone"
        )
    for name in (
        "contract_sha256",
        "source_library_manifest_sha256",
        "source_library_identity_sha256",
        "source_library_content_sha256",
        "texture_contract_sha256",
        "material_layer_sha256",
    ):
        _require_hash(ground_material.get(name), f"ground_material.{name}")
    atlas_hashes = ground_material.get("runtime_atlas_sha256")
    if not isinstance(atlas_hashes, dict) or set(atlas_hashes) != {
        "basecolor",
        "normal",
        "height",
        "orm",
    }:
        raise ContractError(
            f"tile receipt {tile_id} runtime atlas identity is incomplete"
        )
    for role, digest in atlas_hashes.items():
        _require_hash(digest, f"ground_material.runtime_atlas_sha256.{role}")
    if (
        ground_material.get("source_library_schema")
        != "fireviewer.clean-pbr-texture-library.v1"
        or ground_material.get("visual_acceptance") != "accepted_human_visual"
    ):
        raise ContractError(f"tile receipt {tile_id} ground material status is invalid")
    if ground_material.get("runtime_shader") != {
        "schema": "fireviewer.ground-runtime-shader-binding.v1",
        "status": "pending_dedicated_mdl_validation",
        "implementation": None,
        "source_artifact": None,
        "production_textured_runtime_qualified": False,
        "preview_surface_policy": "diagnostic_untextured_only",
        "required_capabilities": [
            "four_profile_id_indirections",
            "rgba8_weighted_pbr_blend",
            "epsg2154_world_projection",
            "undirected_orientation_0_to_pi",
            "world_xy_and_world_triplanar",
        ],
    }:
        raise ContractError(
            f"tile receipt {tile_id} must expose the pending runtime shader gate"
        )
    surface_mapping = receipt.get("surface_mapping")
    if (
        not isinstance(surface_mapping, dict)
        or set(surface_mapping)
        != {
            "schema",
            "crs",
            "bounds_l93_m",
            "grid_size_px",
            "cell_size_m",
            "row_order",
            "profile_count",
            "profile_ids",
            "profile_weights",
            "confidence",
            "orientation",
            "world_projection",
            "variant_selection",
            "runtime_procedural_material",
            "runtime_orthophoto",
            "surface_overlays",
        }
        or surface_mapping.get("schema") != "fireviewer.ground-surface-mapping.v3"
        or surface_mapping.get("crs") != CRS
        or surface_mapping.get("grid_size_px") != [500, 500]
        or surface_mapping.get("cell_size_m") != 1
        or surface_mapping.get("row_order") != "north_to_south"
        or surface_mapping.get("profile_count") != 72
        or surface_mapping.get("profile_ids")
        != {
            "file": "ground-profile-ids.png",
            "mode": "RGBA8",
            "encoding": "four_zero_based_stable_profile_indices",
        }
        or surface_mapping.get("profile_weights")
        != {
            "file": "ground-profile-weights.png",
            "mode": "RGBA8",
            "encoding": "four_profile_weights_sum_exactly_255_per_pixel",
        }
        or surface_mapping.get("confidence")
        != {
            "file": "ground-confidence.png",
            "mode": "L8",
            "encoding": "best_vs_next_semantic_class_margin_0_to_255",
        }
        or surface_mapping.get("orientation")
        != {
            "file": "ground-orientation.png",
            "mode": "L8",
            "encoding": "undirected_angle_0_to_pi_mapped_to_uint8",
        }
        or surface_mapping.get("world_projection")
        != "EPSG:2154 metric XY with no tile-local phase reset"
        or surface_mapping.get("variant_selection") != "baked_profile_id"
        or surface_mapping.get("runtime_procedural_material") != "forbidden"
        or surface_mapping.get("runtime_orthophoto") != "forbidden"
        or surface_mapping.get("surface_overlays") != "not_packaged"
    ):
        raise ContractError(f"tile receipt {tile_id} surface mapping is invalid")
    surface_bounds = surface_mapping.get("bounds_l93_m")
    if (
        not isinstance(surface_bounds, list)
        or len(surface_bounds) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in surface_bounds
        )
        or float(surface_bounds[2]) - float(surface_bounds[0]) != TILE_SIZE_M
        or float(surface_bounds[3]) - float(surface_bounds[1]) != TILE_SIZE_M
        or any(float(value) % TILE_SIZE_M != 0 for value in surface_bounds)
    ):
        raise ContractError(f"tile receipt {tile_id} surface bounds are invalid")
    missing_outputs = sorted(REQUIRED_TILE_OUTPUTS - receipt["outputs"].keys())
    if missing_outputs:
        raise ContractError(
            f"tile receipt {tile_id} is missing canonical outputs: "
            + ", ".join(missing_outputs)
        )
    return json.loads(canonical_json_bytes(dict(receipt)).decode("utf-8"))


def _verify_tile_receipt_files(
    receipt: Mapping[str, Any], *, zone: ZoneDefinition
) -> None:
    tile_id = str(receipt["tile_id"])
    tile_root = zone.export_root / zone.zone_id / zone.revision / "tiles" / tile_id
    done_path = tile_root / "tile.done.v3.json"
    if not done_path.is_file():
        raise ContractError(f"canonical tile receipt is absent: {done_path}")
    disk_receipt = _read_json(done_path)
    if disk_receipt != dict(receipt):
        raise ContractError(f"backend tile receipt differs from disk for {tile_id}")
    for name, record in receipt["outputs"].items():
        relative = PureWindowsPath(str(record["path"]))
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ContractError(f"tile output {tile_id}/{name} escapes its package")
        path = tile_root.joinpath(*relative.parts)
        if not path.is_file():
            raise ContractError(f"tile output is absent: {tile_id}/{name}")
        if (
            path.stat().st_size != record["byte_count"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ContractError(f"tile output hash mismatch: {tile_id}/{name}")


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a non-negative integer")
    return value


def _require_nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ContractError(f"{label} must be a non-negative finite number")
    return float(value)


def _validate_phase_metrics(
    phase: str,
    result: Mapping[str, Any],
    *,
    tile_count: int,
    seam_count: int,
) -> None:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ContractError(f"{phase} backend must return strict metrics")
    common_integer_metrics = {
        "tile_count",
        "source_pair_count",
        "source_bytes",
        "package_bytes",
        "lod0_triangles",
        "lod1_triangles",
        "lod2_triangles",
        "bitwise_rebuild_count",
        "lod0_stitch_variant_count",
        "lod1_stitch_variant_count",
        "lod2_stitch_variant_count",
        "lod0_maximum_final_error_mm",
        "lod1_maximum_final_error_mm",
        "lod2_maximum_final_error_mm",
        "lod0_maximum_stitch_triangles",
        "lod1_maximum_stitch_triangles",
        "lod2_maximum_stitch_triangles",
    }
    for name in common_integer_metrics:
        _require_nonnegative_int(metrics.get(name), f"{phase}.metrics.{name}")
    _require_nonnegative_number(
        metrics.get("elapsed_seconds"), f"{phase}.metrics.elapsed_seconds"
    )
    _require_nonnegative_int(
        metrics.get("peak_python_bytes"), f"{phase}.metrics.peak_python_bytes"
    )
    if metrics["tile_count"] != tile_count:
        raise ContractError(f"{phase} metric tile_count differs from its receipts")
    if metrics["bitwise_rebuild_count"] != tile_count:
        raise ContractError(f"{phase} did not bitwise-rebuild every produced tile")
    for lod, limit_mm in enumerate((500, 2_000, 8_000)):
        if metrics[f"lod{lod}_stitch_variant_count"] != tile_count * 16:
            raise ContractError(
                f"{phase} does not prove all 16 stitch masks for every LOD{lod} tile"
            )
        if metrics[f"lod{lod}_maximum_final_error_mm"] > limit_mm:
            raise ContractError(
                f"{phase} LOD{lod} final stitched error exceeds {limit_mm} mm"
            )
        if metrics[f"lod{lod}_maximum_stitch_triangles"] <= 0:
            raise ContractError(f"{phase} LOD{lod} stitch triangle count is absent")
    if phase == "pilot":
        for name in (
            "projected_zone_package_bytes",
            "projected_peak_disk_bytes",
            "internal_seam_count",
            "blender_report_count",
            "aov_invalid_pixel_count",
        ):
            _require_nonnegative_int(metrics.get(name), f"pilot.metrics.{name}")
        if metrics["aov_invalid_pixel_count"] != 0:
            raise ContractError("pilot Blender AOV contains a non-LOD0 terrain pixel")
        if metrics["blender_report_count"] < 1:
            raise ContractError("pilot requires at least one Blender visual proof")
    elif phase == "produce":
        for name in (
            "built_tile_count",
            "reused_tile_count",
            "raw_source_count",
            "part_file_count",
        ):
            _require_nonnegative_int(metrics.get(name), f"produce.metrics.{name}")
        if metrics["built_tile_count"] + metrics["reused_tile_count"] != tile_count:
            raise ContractError("produce built/reused counts do not cover the zone")
        if metrics["raw_source_count"] or metrics["part_file_count"]:
            raise ContractError("produce left raw terrain sources or .part files")
    elif phase == "qa":
        for name in (
            "seam_count",
            "maximum_height_gap_mm",
            "normal_mismatch_count",
            "stitch_lod_pair_count",
            "stitch_signature_mismatch_count",
            "nodata_pixel_count",
            "fallback_material_count",
            "composition_failure_count",
            "blender_report_count",
            "aov_invalid_pixel_count",
            "deterministic_tile_count",
        ):
            _require_nonnegative_int(metrics.get(name), f"qa.metrics.{name}")
        if metrics["seam_count"] != seam_count:
            raise ContractError("qa metric seam_count differs from the zone plan")
        if metrics["stitch_lod_pair_count"] != seam_count * 7:
            raise ContractError(
                "qa did not validate all seven admissible LOD pairs on every seam"
            )
        if metrics["maximum_height_gap_mm"] > 1:
            raise ContractError("qa shared-edge altitude gap exceeds 1 mm")
        if any(
            metrics[name]
            for name in (
                "nodata_pixel_count",
                "normal_mismatch_count",
                "stitch_signature_mismatch_count",
                "fallback_material_count",
                "composition_failure_count",
                "aov_invalid_pixel_count",
            )
        ):
            raise ContractError(
                "qa contains a blocking terrain, material or AOV failure"
            )
        if metrics["blender_report_count"] < 1:
            raise ContractError("qa requires at least one Blender visual proof")
        if metrics["deterministic_tile_count"] != tile_count:
            raise ContractError("qa did not validate every deterministic tile package")


def _zone_work_artifact(
    zone: ZoneDefinition,
    *,
    relative_path: Any,
    expected_sha256: Any,
    expected_relative_path: str,
    label: str,
) -> Path:
    if relative_path != expected_relative_path:
        raise ContractError(f"{label} has a non-canonical path")
    expected_hash = _require_hash(expected_sha256, f"{label}_sha256")
    relative = PureWindowsPath(expected_relative_path)
    path = zone.work_root.joinpath(*relative.parts).resolve()
    if (
        not _is_relative_to(path, zone.work_root)
        or not path.is_file()
        or sha256_file(path) != expected_hash
    ):
        raise ContractError(f"{label} is absent or changed")
    return path


def _validate_zone_visual_technical_receipt(
    zone: ZoneDefinition,
    *,
    job_relative_path: Any,
    job_sha256: Any,
    technical_relative_path: Any,
    technical_sha256: Any,
    review_template_relative_path: Any,
    review_template_sha256: Any,
    recipe_id: str,
    recipe_build_id: str,
    build_id: str,
) -> dict[str, Any]:
    job_path = _zone_work_artifact(
        zone,
        relative_path=job_relative_path,
        expected_sha256=job_sha256,
        expected_relative_path=ZONE_VISUAL_JOB_RELATIVE_PATH,
        label="zone visual job",
    )
    technical_path = _zone_work_artifact(
        zone,
        relative_path=technical_relative_path,
        expected_sha256=technical_sha256,
        expected_relative_path=ZONE_VISUAL_TECHNICAL_RELATIVE_PATH,
        label="zone visual technical receipt",
    )
    template_path = _zone_work_artifact(
        zone,
        relative_path=review_template_relative_path,
        expected_sha256=review_template_sha256,
        expected_relative_path=ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH,
        label="zone visual review template",
    )
    job = _read_json(job_path)
    if (
        job.get("schema") != ZONE_VISUAL_JOB_SCHEMA
        or job.get("zone_id") != zone.zone_id
        or job.get("revision") != zone.revision
        or job.get("recipe_id") != recipe_id
        or job.get("recipe_build_id") != recipe_build_id
        or job.get("build_id") != build_id
    ):
        raise ContractError("zone visual job identity is invalid")
    try:
        receipt = validate_technical_receipt(technical_path)
    except Exception as error:
        raise ContractError(
            f"zone visual technical receipt is invalid: {error}"
        ) from error
    if (
        receipt.get("schema") != ZONE_VISUAL_TECHNICAL_SCHEMA
        or receipt.get("status") != ZONE_VISUAL_TECHNICAL_STATUS
        or receipt.get("production_visual_gate_passed") is not False
        or receipt.get("human_visual_acceptance") != "pending_exhaustive_review"
        or receipt.get("zone_id") != zone.zone_id
        or receipt.get("revision") != zone.revision
        or receipt.get("recipe_id") != recipe_id
        or receipt.get("recipe_build_id") != recipe_build_id
        or receipt.get("build_id") != build_id
        or receipt.get("job_sha256") != sha256_file(job_path)
    ):
        raise ContractError("zone visual technical receipt identity is invalid")
    aov = receipt.get("aov")
    if (
        not isinstance(aov, dict)
        or aov.get("terrain_lod") != "fireviewer:terrain_lod"
        or aov.get("terrain_coverage") != "fireviewer:terrain_coverage"
        or aov.get("expected_lod") != 0
        or aov.get("invalid_lod_pixel_count") != 0
        or aov.get("invalid_coverage_pixel_count") != 0
        or not isinstance(aov.get("terrain_pixel_count"), int)
        or aov["terrain_pixel_count"] <= 0
    ):
        raise ContractError("zone technical receipt does not prove all-LOD0 coverage")
    template = _read_json(template_path)
    capture_reviews = template.get("capture_reviews")
    captures = receipt.get("captures")
    if (
        template.get("schema") != ZONE_VISUAL_HUMAN_REVIEW_SCHEMA
        or template.get("decision") != "pending"
        or template.get("reviewer") != {"kind": "human", "id": ""}
        or template.get("technical_receipt_sha256") != sha256_file(technical_path)
        or template.get("capture_set_sha256") != receipt.get("capture_set_sha256")
        or not isinstance(captures, list)
        or not isinstance(capture_reviews, list)
        or len(capture_reviews) != len(captures)
        or {
            item.get("capture_id") for item in capture_reviews if isinstance(item, dict)
        }
        != {item.get("capture_id") for item in captures if isinstance(item, dict)}
        or any(
            not isinstance(item, dict) or item.get("decision") != "pending"
            for item in capture_reviews
        )
    ):
        raise ContractError("zone human visual review template is not canonical")
    return receipt


def _validate_zone_visual_receipt(
    zone: ZoneDefinition,
    *,
    relative_path: Any,
    expected_sha256: Any,
    technical_relative_path: Any,
    technical_sha256: Any,
    recipe_id: str,
    recipe_build_id: str,
    build_id: str,
) -> dict[str, Any]:
    if relative_path != ZONE_VISUAL_RECEIPT_NAME:
        raise ContractError("zone visual receipt has a non-canonical path")
    expected_hash = _require_hash(expected_sha256, "visual_receipt_sha256")
    path = zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ContractError("zone visual receipt is absent or changed")
    technical_path = _zone_work_artifact(
        zone,
        relative_path=technical_relative_path,
        expected_sha256=technical_sha256,
        expected_relative_path=ZONE_VISUAL_TECHNICAL_RELATIVE_PATH,
        label="zone visual technical receipt",
    )
    review_path = zone.work_root.joinpath(
        *PureWindowsPath(ZONE_VISUAL_REVIEW_RELATIVE_PATH).parts
    ).resolve()
    if not _is_relative_to(review_path, zone.work_root) or not review_path.is_file():
        raise ContractError("explicit zone human visual review is absent")
    try:
        receipt = accept_zone_visual_review(technical_path, review_path, path)
    except Exception as error:
        raise ContractError(
            f"zone visual acceptance chain is invalid: {error}"
        ) from error
    if (
        receipt.get("schema") != ZONE_VISUAL_SCHEMA
        or receipt.get("status") != ZONE_VISUAL_ACCEPTED_STATUS
        or receipt.get("zone_visual_gate_passed") is not True
        or receipt.get("automatic_acceptance") is not False
        or receipt.get("review_kind") != "explicit_exhaustive_human"
        or receipt.get("zone_id") != zone.zone_id
        or receipt.get("revision") != zone.revision
        or receipt.get("recipe_id") != recipe_id
        or receipt.get("recipe_build_id") != recipe_build_id
        or receipt.get("build_id") != build_id
    ):
        raise ContractError("zone Blender visual receipt is not accepted")
    aov = receipt.get("aov")
    if (
        not isinstance(aov, dict)
        or aov.get("terrain_lod") != "fireviewer:terrain_lod"
        or aov.get("terrain_coverage") != "fireviewer:terrain_coverage"
        or aov.get("expected_lod") != 0
        or aov.get("invalid_lod_pixel_count") != 0
        or aov.get("invalid_coverage_pixel_count") != 0
        or not isinstance(aov.get("terrain_pixel_count"), int)
        or aov["terrain_pixel_count"] <= 0
    ):
        raise ContractError("zone acceptance does not prove all-LOD0 coverage")
    return receipt


def _validate_tile_visual_reports(
    zone: ZoneDefinition, value: Any
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContractError("Blender visual proof list is absent")
    zone_package_root = zone.export_root / zone.zone_id / zone.revision
    validated: list[dict[str, Any]] = []
    seen_tiles: set[str] = set()
    for index, raw_record in enumerate(value):
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "tile_id",
            "report",
            "render",
            "terrain_pixel_count",
            "invalid_pixel_count",
            "maximum_absolute_error",
        }:
            raise ContractError(f"Blender visual proof {index} is invalid")
        tile_id = _require_safe_id(
            raw_record.get("tile_id"), f"visual[{index}].tile_id"
        )
        if tile_id in seen_tiles:
            raise ContractError(f"duplicate Blender visual proof for {tile_id}")
        seen_tiles.add(tile_id)
        resolved_artifacts: dict[str, Path] = {}
        for kind in ("report", "render"):
            artifact = raw_record.get(kind)
            if not isinstance(artifact, dict):
                raise ContractError(f"visual[{index}].{kind} is invalid")
            _validate_relative_artifact(artifact, f"visual[{index}].{kind}")
            relative = PureWindowsPath(str(artifact["path"]))
            path = zone_package_root.joinpath(*relative.parts)
            if (
                not path.is_file()
                or path.stat().st_size != artifact["byte_count"]
                or sha256_file(path) != artifact["sha256"]
            ):
                raise ContractError(
                    f"Blender {kind} proof is absent or changed: {tile_id}"
                )
            resolved_artifacts[kind] = path
        report = _read_json(resolved_artifacts["report"])
        report_aov = report.get("aov")
        if (
            report.get("schema") != TILE_BLENDER_VISUAL_SCHEMA
            or report.get("status") != "accepted_blender_textured_visual"
            or report.get("geometry_lod_status") != "accepted_blender_geometry_lod"
            or report.get("production_visual_gate_passed") is not True
            or report.get("human_visual_acceptance") != "accepted_human_visual"
            or report.get("source_library_status") != "accepted_clean_pbr_library"
            or report.get("tile_id") != tile_id
            or report.get("selected_lod") != 0
            or report.get("render_resolution") != [512, 512]
            or not isinstance(report_aov, dict)
            or report_aov.get("validated_primary_views") != ["topdown", "oblique"]
        ):
            raise ContractError(f"Blender report does not prove LOD0 for {tile_id}")

        internal_artifacts: dict[str, tuple[dict[str, Any], Path]] = {}
        for name, expected_aov, expected_value, requires_pixels in (
            ("lod", "fireviewer:terrain_lod", 0, True),
            ("coverage", "fireviewer:terrain_coverage", 1, False),
            ("oblique_lod", "fireviewer:terrain_lod", 0, True),
            ("oblique_coverage", "fireviewer:terrain_coverage", 1, False),
        ):
            record = report_aov.get(name)
            if (
                not isinstance(record, dict)
                or record.get("name") != expected_aov
                or record.get("expected_value") != expected_value
                or record.get("invalid_pixel_count") != 0
                or (
                    requires_pixels
                    and (
                        not isinstance(record.get("terrain_pixel_count"), int)
                        or record["terrain_pixel_count"] <= 0
                    )
                )
            ):
                raise ContractError(
                    f"Blender report has an invalid {name} proof for {tile_id}"
                )
            relative = PureWindowsPath(str(record.get("path", "")))
            if relative.is_absolute() or relative.drive or len(relative.parts) != 1:
                raise ContractError(
                    f"Blender report artifact path is invalid for {tile_id}/{name}"
                )
            artifact_path = resolved_artifacts["report"].parent / relative.name
            expected_internal_hash = _require_hash(
                record.get("sha256"), f"Blender {tile_id}/{name} sha256"
            )
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size <= 0
                or sha256_file(artifact_path) != expected_internal_hash
            ):
                raise ContractError(
                    f"Blender report artifact is absent or changed: {tile_id}/{name}"
                )
            internal_artifacts[name] = (record, artifact_path)

        beauty = report.get("beauty")
        if not isinstance(beauty, dict):
            raise ContractError(f"Blender beauty proofs are absent for {tile_id}")
        for view in ("topdown", "oblique"):
            record = beauty.get(view)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("terrain_pixel_count"), int)
                or record["terrain_pixel_count"] <= 0
                or not isinstance(record.get("frame_coverage_ratio"), (int, float))
                or record["frame_coverage_ratio"] <= 0
                or not isinstance(record.get("distinct_rgb8_count"), int)
                or record["distinct_rgb8_count"] < 16
            ):
                raise ContractError(
                    f"Blender {view} beauty proof is invalid for {tile_id}"
                )
            relative = PureWindowsPath(str(record.get("path", "")))
            if relative.is_absolute() or relative.drive or len(relative.parts) != 1:
                raise ContractError(
                    f"Blender beauty path is invalid for {tile_id}/{view}"
                )
            artifact_path = resolved_artifacts["report"].parent / relative.name
            expected_internal_hash = _require_hash(
                record.get("sha256"), f"Blender {tile_id}/{view} sha256"
            )
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size <= 0
                or sha256_file(artifact_path) != expected_internal_hash
            ):
                raise ContractError(
                    f"Blender beauty is absent or changed: {tile_id}/{view}"
                )

        lod_record, lod_path = internal_artifacts["lod"]
        if (
            resolved_artifacts["render"] != lod_path
            or raw_record["render"]["sha256"] != lod_record["sha256"]
            or raw_record["render"]["byte_count"] != lod_path.stat().st_size
        ):
            raise ContractError(
                f"Blender outer render proof differs from top-down LOD AOV: {tile_id}"
            )
        expected_terrain_pixels = sum(
            int(internal_artifacts[name][0]["terrain_pixel_count"])
            for name in ("lod", "oblique_lod")
        )
        expected_invalid_pixels = sum(
            int(internal_artifacts[name][0]["invalid_pixel_count"])
            for name in ("lod", "coverage", "oblique_lod", "oblique_coverage")
        )
        expected_maximum_error = max(
            float(internal_artifacts[name][0].get("maximum_absolute_error", 0.0))
            for name in ("lod", "oblique_lod")
        )
        for name in (
            "terrain_pixel_count",
            "invalid_pixel_count",
            "maximum_absolute_error",
        ):
            expected = {
                "terrain_pixel_count": expected_terrain_pixels,
                "invalid_pixel_count": expected_invalid_pixels,
                "maximum_absolute_error": expected_maximum_error,
            }[name]
            if raw_record.get(name) != expected:
                raise ContractError(
                    f"Blender proof summary differs from report: {tile_id}"
                )
        validated.append(dict(raw_record))
    return validated


@dataclass(frozen=True)
class PhaseContext:
    phase: str
    mode: str
    zone: ZoneDefinition
    plan: dict[str, Any]
    source_lock: dict[str, Any] | None
    state: dict[str, Any]
    run_root: Path
    export_root: Path
    temp_root: Path
    cache_root: Path
    environment: dict[str, str]
    download_workers: int
    package_workers: int
    max_tiles: int | None


class PhaseBackend(Protocol):
    def __call__(self, context: PhaseContext) -> Mapping[str, Any]: ...


DiskUsage = Callable[[Path], Any]


def _initial_state(zone: ZoneDefinition, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": RUN_STATE_SCHEMA,
        "zone_id": zone.zone_id,
        "revision": zone.revision,
        "recipe_id": plan["recipe_id"],
        "plan_id": plan["plan_id"],
        "recipe_build_id": None,
        "build_id": None,
        "generation": 0,
        "status": "active",
        "active_phase": None,
        "phases": {
            phase: {
                "status": "pending",
                "attempts": 0,
                "receipt": None,
                "receipt_sha256": None,
                "parameters": None,
                "started_at_utc": None,
                "completed_at_utc": None,
                "last_error": None,
            }
            for phase in PHASES
        },
        "updated_at_utc": utc_now(),
    }


def _validate_state(
    state: Mapping[str, Any], zone: ZoneDefinition, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if state.get("schema") != RUN_STATE_SCHEMA:
        raise ContractError("unsupported run state schema")
    expected = {
        "zone_id": zone.zone_id,
        "revision": zone.revision,
        "recipe_id": plan["recipe_id"],
        "plan_id": plan["plan_id"],
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise PhaseGateError(f"run state {name} no longer matches the zone plan")
    phases = state.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ContractError("run state phase mapping is incomplete")
    return dict(state)


def _pid_is_alive(pid: int) -> bool | None:
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 87:
            return False
        return None
    return True


@contextmanager
def _exclusive_lock(path: Path, *, timeout_s: float = 10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                content = path.read_text(encoding="ascii").strip()
                pid = int(content[4:]) if content.startswith("pid=") else -1
            except (OSError, ValueError, UnicodeError):
                pid = -1
            if _pid_is_alive(pid) is False:
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() - started >= timeout_s:
                raise PhaseGateError(f"another terrainctl process owns {path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


class TerrainController:
    """Stateful orchestrator whose phase implementations are explicitly injected."""

    def __init__(
        self,
        *,
        backends: Mapping[str, PhaseBackend] | None = None,
        disk_usage: DiskUsage = shutil.disk_usage,
        coordinator_root: Path | None = None,
    ) -> None:
        self.backends = dict(backends or {})
        unknown = sorted(set(self.backends) - set(PHASES[1:]))
        if unknown:
            raise ValueError("unknown phase backend(s): " + ", ".join(unknown))
        self.disk_usage = disk_usage
        self.coordinator_root = _require_d_path(
            coordinator_root or DEFAULT_GLOBAL_COORDINATOR_ROOT,
            "terrain coordinator root",
        )

    @property
    def active_zone_path(self) -> Path:
        return self.coordinator_root / "active-zone.v1.json"

    @property
    def process_lock_path(self) -> Path:
        return self.coordinator_root / ".terrainctl.lock"

    def run(
        self,
        zone_path: Path,
        *,
        phase: str,
        mode: str,
        download_workers: int = 2,
        package_workers: int = 1,
        max_tiles: int | None = None,
    ) -> dict[str, Any]:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        if mode not in {"dry-run", "execute", "resume"}:
            raise ValueError(f"unknown mode: {mode}")
        if not 1 <= download_workers <= 4:
            raise ValueError("download_workers must be between 1 and 4")
        if not 1 <= package_workers <= 4:
            raise ValueError("package_workers must be between 1 and 4")
        if max_tiles is not None:
            if phase != "pilot":
                raise ValueError("--max-tiles is only valid for the pilot phase")
            if isinstance(max_tiles, bool) or max_tiles <= 0:
                raise ValueError("--max-tiles must be strictly positive")

        zone = load_zone_spec(zone_path)
        plan = build_zone_plan(zone)
        if mode == "dry-run":
            return self._dry_run(zone, plan, phase)
        if phase != "plan" and phase not in self.backends:
            raise BackendUnavailableError(
                f"phase {phase!r} has no injected backend; no production action was taken"
            )
        with _exclusive_lock(self.process_lock_path):
            self._assert_or_claim_active_zone(zone, plan, phase=phase, mode=mode)
            if phase == "plan":
                return self._run_plan(zone, plan, mode=mode)
            return self._run_backend_phase(
                zone,
                plan,
                phase=phase,
                mode=mode,
                download_workers=download_workers,
                package_workers=package_workers,
                max_tiles=max_tiles,
            )

    def _dry_run(
        self, zone: ZoneDefinition, plan: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        eligible = True
        reason: str | None = None
        if phase != "plan":
            try:
                state, source_lock = self._load_gated_state(zone, plan, phase)
                del state, source_lock
            except TerrainControlError as exc:
                eligible = False
                reason = str(exc)
        free_bytes = int(self.disk_usage(_nearest_existing_parent(zone.work_root)).free)
        required_free = int(plan["storage"]["required_free_bytes"])
        if phase == "preflight" and free_bytes < required_free:
            eligible = False
            reason = (
                f"insufficient disk space: {free_bytes} free, {required_free} required"
            )
        if self.active_zone_path.is_file():
            active = _read_json(self.active_zone_path)
            if active.get("recipe_id") != plan["recipe_id"]:
                eligible = False
                reason = "another zone or recipe is active"
        return {
            "schema": CONTROL_RESULT_SCHEMA,
            "mode": "dry-run",
            "phase": phase,
            "zone_id": zone.zone_id,
            "revision": zone.revision,
            "recipe_id": plan["recipe_id"],
            "plan_id": plan["plan_id"],
            "eligible": eligible,
            "blocked_reason": reason,
            "writes_performed": False,
            "network_access_performed": False,
            "summary": plan["summary"],
            "disk": {
                "estimated_peak_bytes": plan["storage"]["estimated_peak_bytes"],
                "safety_margin_bytes": SAFETY_MARGIN_BYTES,
                "required_free_bytes": required_free,
                "free_bytes": free_bytes,
            },
        }

    def _assert_or_claim_active_zone(
        self,
        zone: ZoneDefinition,
        plan: Mapping[str, Any],
        *,
        phase: str,
        mode: str,
    ) -> None:
        active_path = self.active_zone_path
        if active_path.exists():
            active = _read_json(active_path)
            expected = {
                "schema": "fireviewer.active-terrain-zone.v1",
                "zone_id": zone.zone_id,
                "revision": zone.revision,
                "recipe_id": plan["recipe_id"],
            }
            if any(active.get(key) != value for key, value in expected.items()):
                raise PhaseGateError(
                    "another zone or recipe is active; it must be accepted and cleaned first"
                )
            return
        cleaned_state = None
        if zone.state_path.is_file():
            candidate_state = _read_json(zone.state_path)
            if candidate_state.get("status") == "cleaned":
                cleaned_state = candidate_state
        if cleaned_state is not None:
            # Completed runs remain independently verifiable without reclaiming
            # the global production marker released by cleanup.
            return
        if mode == "resume":
            raise PhaseGateError("cannot resume because no active zone marker exists")
        _atomic_write_json(
            active_path,
            {
                "schema": "fireviewer.active-terrain-zone.v1",
                "zone_id": zone.zone_id,
                "revision": zone.revision,
                "recipe_id": plan["recipe_id"],
                "claimed_at_utc": utc_now(),
            },
        )

    def _run_plan(
        self, zone: ZoneDefinition, plan: dict[str, Any], *, mode: str
    ) -> dict[str, Any]:
        receipt_path = zone.receipt_root / PHASE_RECEIPT_NAMES["plan"]
        if receipt_path.exists():
            validate_zone_plan(_read_json(receipt_path), zone)
            if mode == "execute":
                raise PhaseGateError(
                    "plan is already complete; use --resume to verify it"
                )
            if zone.state_path.is_file():
                state = _validate_state(_read_json(zone.state_path), zone, plan)
            else:
                state = _initial_state(zone, plan)
                phase_state = state["phases"]["plan"]
                phase_state.update(
                    {
                        "status": "completed",
                        "attempts": 1,
                        "receipt": str(receipt_path.relative_to(zone.run_root)).replace(
                            "\\", "/"
                        ),
                        "receipt_sha256": sha256_file(receipt_path),
                        "started_at_utc": utc_now(),
                        "completed_at_utc": utc_now(),
                    }
                )
                state["generation"] = 1
                state["updated_at_utc"] = utc_now()
                _atomic_write_json(zone.state_path, state)
            return self._control_result(zone, plan, state, "plan", resumed=True)
        zone.receipt_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(receipt_path, plan)
        state = _initial_state(zone, plan)
        phase_state = state["phases"]["plan"]
        phase_state.update(
            {
                "status": "completed",
                "attempts": 1,
                "receipt": str(receipt_path.relative_to(zone.run_root)).replace(
                    "\\", "/"
                ),
                "receipt_sha256": sha256_file(receipt_path),
                "started_at_utc": utc_now(),
                "completed_at_utc": utc_now(),
            }
        )
        state["generation"] = 1
        state["updated_at_utc"] = utc_now()
        _atomic_write_json(zone.state_path, state)
        return self._control_result(zone, plan, state, "plan", resumed=False)

    def _load_gated_state(
        self, zone: ZoneDefinition, plan: dict[str, Any], phase: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        plan_path = zone.receipt_root / PHASE_RECEIPT_NAMES["plan"]
        if not plan_path.is_file():
            raise PhaseGateError("plan phase has not completed")
        validate_zone_plan(_read_json(plan_path), zone)
        if not zone.state_path.is_file():
            raise PhaseGateError("run state is absent")
        state = _validate_state(_read_json(zone.state_path), zone, plan)
        phase_index = PHASES.index(phase)
        for previous in PHASES[:phase_index]:
            if state["phases"][previous]["status"] != "completed":
                raise PhaseGateError(f"phase {previous!r} has not completed")
            receipt = zone.receipt_root / PHASE_RECEIPT_NAMES[previous]
            if not receipt.is_file():
                raise PhaseGateError(f"phase {previous!r} receipt is absent")
            expected_hash = state["phases"][previous]["receipt_sha256"]
            if sha256_file(receipt) != expected_hash:
                raise PhaseGateError(f"phase {previous!r} receipt has changed")
        source_lock: dict[str, Any] | None = None
        if phase_index > PHASES.index("preflight"):
            lock_path = zone.receipt_root / PHASE_RECEIPT_NAMES["preflight"]
            source_lock = validate_source_lock(_read_json(lock_path), plan)
            if state.get("recipe_build_id") != source_lock["recipe_build_id"]:
                raise PhaseGateError(
                    "run state recipe_build_id differs from the source lock"
                )
        return state, source_lock

    def _run_backend_phase(
        self,
        zone: ZoneDefinition,
        plan: dict[str, Any],
        *,
        phase: str,
        mode: str,
        download_workers: int,
        package_workers: int,
        max_tiles: int | None,
    ) -> dict[str, Any]:
        state, source_lock = self._load_gated_state(zone, plan, phase)
        current = state["phases"][phase]
        receipt_path = zone.receipt_root / PHASE_RECEIPT_NAMES[phase]
        pilot_limit = (
            min(max_tiles or DEFAULT_PILOT_TILES, len(plan["tiles"]))
            if phase == "pilot"
            else None
        )
        if current["status"] == "completed":
            if (
                not receipt_path.is_file()
                or sha256_file(receipt_path) != current["receipt_sha256"]
            ):
                raise PhaseGateError(
                    f"completed phase {phase!r} has a missing or changed receipt"
                )
            if mode == "execute" and phase != "cleanup":
                raise PhaseGateError(
                    f"phase {phase!r} is already complete; use --resume to verify it"
                )
            if phase == "cleanup":
                self._finalize_cleanup_state(state, zone)
            return self._control_result(zone, plan, state, phase, resumed=True)
        if mode == "execute" and current["status"] in {"running", "failed"}:
            raise PhaseGateError(f"phase {phase!r} is interrupted; use --resume")
        if mode == "resume" and current["status"] == "pending":
            raise PhaseGateError(
                f"phase {phase!r} has no interrupted attempt to resume"
            )
        if mode == "resume" and phase == "pilot":
            recorded_parameters = current.get("parameters")
            if not isinstance(recorded_parameters, dict):
                raise PhaseGateError("pilot resume parameters are absent")
            recorded_limit = recorded_parameters.get("max_tiles")
            if max_tiles is not None and pilot_limit != recorded_limit:
                raise PhaseGateError(
                    "pilot --max-tiles differs from the interrupted attempt"
                )
            pilot_limit = recorded_limit
        if receipt_path.exists():
            if mode != "resume":
                raise PhaseGateError(
                    f"phase {phase!r} already has an uncommitted receipt"
                )
            existing_receipt = _read_json(receipt_path)
            self._validate_existing_phase_receipt(
                phase,
                existing_receipt,
                zone=zone,
                plan=plan,
                source_lock=source_lock,
                max_tiles=pilot_limit,
            )
            if phase == "preflight":
                state["recipe_build_id"] = existing_receipt["recipe_build_id"]
            self._complete_state_phase(state, zone, phase, receipt_path)
            if phase == "cleanup":
                self._finalize_cleanup_state(state, zone)
            return self._control_result(zone, plan, state, phase, resumed=True)

        if phase == "preflight":
            self._assert_disk_budget(
                zone.work_root, plan["storage"]["estimated_peak_bytes"]
            )
        temp_root = zone.run_root / "temp"
        cache_root = zone.run_root / "cache"
        temp_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        redirected_directories = {
            "PYTHONPYCACHEPREFIX": cache_root / "python",
            "BLENDER_USER_CONFIG": cache_root / "blender" / "config",
            "BLENDER_USER_SCRIPTS": cache_root / "blender" / "scripts",
            "BLENDER_USER_EXTENSIONS": cache_root / "blender" / "extensions",
        }
        for redirected in redirected_directories.values():
            redirected.mkdir(parents=True, exist_ok=True)
        current.update(
            {
                "status": "running",
                "attempts": int(current["attempts"]) + 1,
                "started_at_utc": utc_now(),
                "completed_at_utc": None,
                "last_error": None,
                "parameters": {
                    "download_workers": download_workers,
                    "package_workers": package_workers,
                    "max_tiles": pilot_limit,
                },
            }
        )
        state["active_phase"] = phase
        state["generation"] = int(state["generation"]) + 1
        state["updated_at_utc"] = utc_now()
        _atomic_write_json(zone.state_path, state)
        context = PhaseContext(
            phase=phase,
            mode=mode,
            zone=zone,
            plan=plan,
            source_lock=source_lock,
            state=state,
            run_root=zone.run_root,
            export_root=zone.export_root,
            temp_root=temp_root,
            cache_root=cache_root,
            environment={
                "TEMP": str(temp_root),
                "TMP": str(temp_root),
                **{
                    name: str(path)
                    for name, path in sorted(redirected_directories.items())
                },
            },
            download_workers=download_workers,
            package_workers=package_workers,
            max_tiles=pilot_limit,
        )
        try:
            backend_result = self.backends[phase](context)
            if not isinstance(backend_result, Mapping):
                raise ContractError(f"{phase} backend result must be an object")
            result = dict(backend_result)
            if phase == "preflight":
                observed_peak = _require_nonnegative_int(
                    result.get("estimated_peak_bytes"),
                    "preflight.estimated_peak_bytes",
                )
                self._assert_disk_budget(zone.work_root, observed_peak)
            self._validate_backend_audit(result, zone)
            receipt = self._build_phase_receipt(
                phase,
                result,
                zone=zone,
                plan=plan,
                source_lock=source_lock,
                max_tiles=context.max_tiles,
            )
            if phase == "preflight":
                measured_peak = int(result.get("estimated_peak_bytes", 0))
                if measured_peak <= 0:
                    raise ContractError(
                        "preflight must return positive estimated_peak_bytes"
                    )
                self._assert_disk_budget(
                    zone.work_root,
                    max(measured_peak, int(plan["storage"]["estimated_peak_bytes"])),
                )
            _atomic_write_json(receipt_path, receipt)
            if phase == "preflight":
                state["recipe_build_id"] = receipt["recipe_build_id"]
            elif phase == "qa":
                state["build_id"] = receipt["result"]["build_id"]
            self._complete_state_phase(state, zone, phase, receipt_path)
            if phase == "cleanup":
                self._finalize_cleanup_state(state, zone)
            return self._control_result(
                zone, plan, state, phase, resumed=mode == "resume"
            )
        except Exception as exc:
            current.update(
                {
                    "status": "failed",
                    "completed_at_utc": None,
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            )
            state["active_phase"] = phase
            state["generation"] = int(state["generation"]) + 1
            state["updated_at_utc"] = utc_now()
            _atomic_write_json(zone.state_path, state)
            raise

    def _finalize_cleanup_state(
        self, state: dict[str, Any], zone: ZoneDefinition
    ) -> None:
        state["status"] = "cleaned"
        state["active_phase"] = None
        state["updated_at_utc"] = utc_now()
        _atomic_write_json(zone.state_path, state)
        self.active_zone_path.unlink(missing_ok=True)

    def _assert_disk_budget(self, work_root: Path, estimated_peak_bytes: int) -> None:
        free = int(self.disk_usage(_nearest_existing_parent(work_root)).free)
        required = int(estimated_peak_bytes) + SAFETY_MARGIN_BYTES
        if free < required:
            raise StoragePolicyError(
                f"insufficient disk space: {free} bytes free; {required} required "
                "(estimated peak plus 20 GiB)"
            )

    def _validate_backend_audit(
        self, result: Mapping[str, Any], zone: ZoneDefinition
    ) -> None:
        forbidden = result.get("forbidden_c_artifacts")
        if not isinstance(forbidden, list):
            raise ContractError("backend must return forbidden_c_artifacts audit list")
        if forbidden:
            raise StoragePolicyError(
                "backend reported forbidden C: artifacts: "
                + ", ".join(str(item) for item in forbidden)
            )
        created = result.get("created_paths")
        if not isinstance(created, list):
            raise ContractError("backend must return created_paths audit list")
        allowed_roots = (zone.work_root, zone.export_root)
        for index, value in enumerate(created):
            if not isinstance(value, str) or not value:
                raise ContractError(f"created_paths[{index}] must be a path string")
            path = _require_d_path(Path(value), f"created_paths[{index}]")
            if not any(_is_relative_to(path, root) for root in allowed_roots):
                raise StoragePolicyError(
                    f"backend artifact is outside the declared D: roots: {path}"
                )

    def _build_phase_receipt(
        self,
        phase: str,
        result: dict[str, Any],
        *,
        zone: ZoneDefinition,
        plan: dict[str, Any],
        source_lock: dict[str, Any] | None,
        max_tiles: int | None,
    ) -> dict[str, Any]:
        result = dict(result)
        result.pop("created_paths", None)
        result.pop("forbidden_c_artifacts", None)
        if phase == "preflight":
            return build_source_lock(plan, result)
        if source_lock is None:
            raise PhaseGateError(f"phase {phase!r} requires a source lock")
        recipe_id = str(plan["recipe_id"])
        recipe_build_id = str(source_lock["recipe_build_id"])
        tile_ids = {tile["id"] for tile in plan["tiles"]}
        if phase in {"pilot", "produce"}:
            if result.get("status") != "passed":
                raise ContractError(f"{phase} status must be 'passed'")
            raw_receipts = result.get("tile_receipts")
            if not isinstance(raw_receipts, list):
                raise ContractError(f"{phase} backend must return tile_receipts")
            validated = [
                validate_tile_receipt(
                    receipt,
                    recipe_id=recipe_id,
                    recipe_build_id=recipe_build_id,
                    tile_ids=tile_ids,
                )
                for receipt in raw_receipts
                if isinstance(receipt, Mapping)
            ]
            if len(validated) != len(raw_receipts):
                raise ContractError(f"{phase} tile_receipts contain a non-object")
            selected = [receipt["tile_id"] for receipt in validated]
            if len(selected) != len(set(selected)):
                raise ContractError(f"{phase} contains duplicate tile receipts")
            if phase == "produce":
                if set(selected) != tile_ids:
                    raise ContractError(
                        "produce must return exactly one receipt per planned tile"
                    )
            else:
                base_count = min(max_tiles or DEFAULT_PILOT_TILES, len(tile_ids))
                regression_tile_id = plan.get("pilot", {}).get("regression_tile_id")
                expected_counts = {base_count}
                if regression_tile_id is not None and base_count < len(tile_ids):
                    expected_counts.add(base_count + 1)
                if len(selected) not in expected_counts:
                    raise ContractError(
                        "pilot must return the contiguous base block and, when "
                        "outside it, exactly one regression tile"
                    )
                if (
                    regression_tile_id is not None
                    and regression_tile_id not in selected
                ):
                    raise ContractError(
                        "pilot receipts do not include the declared regression tile"
                    )
                contiguous_selection = list(selected)
                if len(selected) == base_count + 1:
                    contiguous_selection.remove(regression_tile_id)
                self._assert_contiguous_tiles(contiguous_selection, plan)
                selection = result.get("selection")
                if (
                    not isinstance(selection, dict)
                    or selection.get("strategy")
                    != "contiguous_relief_and_semantic_maximization"
                    or selection.get("tile_ids") != selected
                ):
                    raise ContractError(
                        "pilot selection proof is absent or non-canonical"
                    )
            for receipt in validated:
                _verify_tile_receipt_files(receipt, zone=zone)
            _validate_phase_metrics(
                phase,
                result,
                tile_count=len(validated),
                seam_count=len(plan["seams"]),
            )
            if phase == "pilot":
                visual_reports = _validate_tile_visual_reports(
                    zone, result.get("visual_reports")
                )
                if {report["tile_id"] for report in visual_reports} != set(selected):
                    raise ContractError(
                        "pilot Blender proofs must cover every selected tile"
                    )
                metrics = result["metrics"]
                if metrics["blender_report_count"] != len(visual_reports) or metrics[
                    "aov_invalid_pixel_count"
                ] != sum(
                    int(report["invalid_pixel_count"]) for report in visual_reports
                ):
                    raise ContractError(
                        "pilot Blender metrics differ from their proof files"
                    )
            result["tile_receipts"] = sorted(
                validated, key=lambda item: item["tile_id"]
            )
            result["tile_receipt_count"] = len(validated)
        elif phase == "qa":
            if result.get("status") != "passed":
                raise ContractError("qa status must be 'passed'")
            validated_tiles = result.get("validated_tile_ids")
            validated_seams = result.get("validated_seam_ids")
            expected_seams = {seam["id"] for seam in plan["seams"]}
            if (
                not isinstance(validated_tiles, list)
                or set(validated_tiles) != tile_ids
                or len(validated_tiles) != len(tile_ids)
            ):
                raise ContractError("qa must validate every planned tile exactly once")
            if (
                not isinstance(validated_seams, list)
                or set(validated_seams) != expected_seams
                or len(validated_seams) != len(expected_seams)
            ):
                raise ContractError("qa must validate every planned seam exactly once")
            result["validated_tile_ids"] = sorted(validated_tiles)
            result["validated_seam_ids"] = sorted(validated_seams)
            _validate_phase_metrics(
                "qa",
                result,
                tile_count=len(tile_ids),
                seam_count=len(expected_seams),
            )
            final_identity = build_final_source_identity(
                plan,
                recipe_build_id=recipe_build_id,
                observed_sources=result.get("observed_sources"),
                tile_receipts=result.get("tile_receipts"),
            )
            for name, expected_value in final_identity.items():
                if result.get(name) != expected_value:
                    raise ContractError(f"qa final source identity differs for {name}")
            _validate_zone_visual_technical_receipt(
                zone,
                job_relative_path=result.get("visual_job"),
                job_sha256=result.get("visual_job_sha256"),
                technical_relative_path=result.get("visual_technical_receipt"),
                technical_sha256=result.get("visual_technical_receipt_sha256"),
                review_template_relative_path=result.get("visual_review_template"),
                review_template_sha256=result.get("visual_review_template_sha256"),
                recipe_id=recipe_id,
                recipe_build_id=recipe_build_id,
                build_id=final_identity["build_id"],
            )
        elif phase == "accept":
            if result.get("status") != "accepted":
                raise ContractError("accept status must be 'accepted'")
            if (
                result.get("runtime_shader")
                != {"status": RUNTIME_SHADER_PENDING_STATUS}
                or result.get("usd_runtime_gate") is not False
            ):
                raise ContractError(
                    "terrain acceptance must keep the USD runtime shader gate pending"
                )
            expected_tiles = len(tile_ids)
            expected_seams = len(plan["seams"])
            if (
                result.get("tile_count") != expected_tiles
                or result.get("seam_count") != expected_seams
            ):
                raise ContractError("accept counts must match the canonical zone plan")
            qa_path = PHASE_RECEIPT_NAMES["qa"]
            actual_qa_path = zone.receipt_root / qa_path
            qa_receipt = _read_json(actual_qa_path)
            qa_result = qa_receipt.get("result")
            if not isinstance(qa_result, dict):
                raise ContractError("acceptance QA receipt has no result")
            final_identity = build_final_source_identity(
                plan,
                recipe_build_id=recipe_build_id,
                observed_sources=qa_result.get("observed_sources"),
                tile_receipts=qa_result.get("tile_receipts"),
            )
            for name in (
                "recipe_build_id",
                "build_id",
                "source_merkle_root_sha256",
            ):
                if result.get(name) != final_identity[name]:
                    raise ContractError(
                        f"accept backend reported a changed final identity: {name}"
                    )
            actual_qa_sha256 = sha256_file(actual_qa_path)
            declared_qa_sha256 = result.get("qa_receipt_sha256")
            if (
                declared_qa_sha256 is not None
                and declared_qa_sha256 != actual_qa_sha256
            ):
                raise ContractError("accept backend reported a changed QA receipt")
            acceptance = {
                "schema": ZONE_ACCEPTANCE_SCHEMA,
                "recipe_id": recipe_id,
                "recipe_build_id": recipe_build_id,
                "build_id": final_identity["build_id"],
                "source_merkle_root_sha256": final_identity[
                    "source_merkle_root_sha256"
                ],
                "status": "accepted",
                "tile_count": expected_tiles,
                "seam_count": expected_seams,
                "qa_receipt": qa_path,
                "qa_receipt_sha256": actual_qa_sha256,
                "visual_receipt": result.get("visual_receipt"),
                "visual_receipt_sha256": result.get("visual_receipt_sha256"),
                "runtime_shader": {
                    "status": RUNTIME_SHADER_PENDING_STATUS,
                },
                "usd_runtime_gate": False,
            }
            _require_hash(acceptance["qa_receipt_sha256"], "qa_receipt_sha256")
            _require_hash(acceptance["visual_receipt_sha256"], "visual_receipt_sha256")
            _validate_zone_visual_receipt(
                zone,
                relative_path=acceptance["visual_receipt"],
                expected_sha256=acceptance["visual_receipt_sha256"],
                technical_relative_path=qa_result.get("visual_technical_receipt"),
                technical_sha256=qa_result.get("visual_technical_receipt_sha256"),
                recipe_id=recipe_id,
                recipe_build_id=recipe_build_id,
                build_id=final_identity["build_id"],
            )
            return acceptance
        elif phase == "cleanup":
            if result.get("status") != "cleaned":
                raise ContractError("cleanup status must be 'cleaned'")
            for name in (
                "raw_source_count",
                "part_file_count",
                "cache_entry_count",
                "forbidden_c_artifact_count",
            ):
                if result.get(name) != 0:
                    raise ContractError(f"cleanup requires {name}=0")
            cleanup_plan = result.get("cleanup_plan")
            if not isinstance(cleanup_plan, list):
                raise ContractError("cleanup must return its bounded cleanup_plan")
            for index, record in enumerate(cleanup_plan):
                if not isinstance(record, dict) or set(record) != {
                    "path",
                    "kind",
                    "existed",
                    "deleted_bytes",
                }:
                    raise ContractError(f"cleanup_plan[{index}] is invalid")
                relative = PureWindowsPath(str(record["path"]))
                if relative.is_absolute() or relative.drive or ".." in relative.parts:
                    raise ContractError("cleanup target escapes the run root")
                if record.get("kind") not in {
                    "raw_sources",
                    "parts",
                    "cache",
                    "temporary",
                }:
                    raise ContractError("cleanup target kind is not approved")
                if not isinstance(record.get("existed"), bool):
                    raise ContractError("cleanup target existed flag must be boolean")
                _require_nonnegative_int(
                    record.get("deleted_bytes"),
                    f"cleanup_plan[{index}].deleted_bytes",
                )
            preserved = result.get("preserved")
            if not isinstance(preserved, dict) or set(preserved) != {
                "tile_count",
                "index_sha256",
                "visual_receipt_sha256",
            }:
                raise ContractError("cleanup preserved proof is incomplete")
            if preserved["tile_count"] != len(tile_ids):
                raise ContractError("cleanup did not preserve every canonical tile")
            _require_hash(preserved["index_sha256"], "cleanup.preserved.index_sha256")
            _require_hash(
                preserved["visual_receipt_sha256"],
                "cleanup.preserved.visual_receipt_sha256",
            )
        return {
            "schema": PHASE_RECEIPT_SCHEMA,
            "phase": phase,
            "recipe_id": recipe_id,
            "recipe_build_id": recipe_build_id,
            "result": result,
        }

    def _assert_contiguous_tiles(
        self, selected: Sequence[str], plan: Mapping[str, Any]
    ) -> None:
        grid_by_id = {
            tile["id"]: tuple(tile["grid"])
            for tile in plan["tiles"]  # type: ignore[index]
        }
        selected_grids = {grid_by_id[tile_id] for tile_id in selected}
        pending = set(selected_grids)
        queue = deque([pending.pop()])
        visited = set(queue)
        while queue:
            x_index, y_index = queue.popleft()
            for neighbour in (
                (x_index - 1, y_index),
                (x_index + 1, y_index),
                (x_index, y_index - 1),
                (x_index, y_index + 1),
            ):
                if neighbour in pending:
                    pending.remove(neighbour)
                    visited.add(neighbour)
                    queue.append(neighbour)
        if len(visited) != len(selected_grids):
            raise ContractError("pilot tile selection must be contiguous")

    def _validate_existing_phase_receipt(
        self,
        phase: str,
        receipt: Mapping[str, Any],
        *,
        zone: ZoneDefinition,
        plan: dict[str, Any],
        source_lock: dict[str, Any] | None,
        max_tiles: int | None,
    ) -> None:
        if phase == "preflight":
            validate_source_lock(receipt, plan)
            return
        if source_lock is None:
            raise ContractError(f"receipt source lock is absent for phase {phase!r}")
        if phase == "accept":
            if receipt.get("schema") != ZONE_ACCEPTANCE_SCHEMA:
                raise ContractError("unsupported acceptance receipt")
            qa_path = zone.receipt_root / PHASE_RECEIPT_NAMES["qa"]
            qa_receipt = _read_json(qa_path)
            qa_result = qa_receipt.get("result")
            if not isinstance(qa_result, dict):
                raise ContractError("acceptance QA receipt has no result")
            final_identity = build_final_source_identity(
                plan,
                recipe_build_id=source_lock["recipe_build_id"],
                observed_sources=qa_result.get("observed_sources"),
                tile_receipts=qa_result.get("tile_receipts"),
            )
            expected = {
                "schema": ZONE_ACCEPTANCE_SCHEMA,
                "recipe_id": plan["recipe_id"],
                "recipe_build_id": source_lock["recipe_build_id"],
                "build_id": final_identity["build_id"],
                "source_merkle_root_sha256": final_identity[
                    "source_merkle_root_sha256"
                ],
                "status": "accepted",
                "tile_count": len(plan["tiles"]),
                "seam_count": len(plan["seams"]),
                "qa_receipt": PHASE_RECEIPT_NAMES["qa"],
                "qa_receipt_sha256": sha256_file(qa_path),
                "visual_receipt": receipt.get("visual_receipt"),
                "visual_receipt_sha256": receipt.get("visual_receipt_sha256"),
                "runtime_shader": {
                    "status": RUNTIME_SHADER_PENDING_STATUS,
                },
                "usd_runtime_gate": False,
            }
            _require_hash(expected["visual_receipt_sha256"], "visual_receipt_sha256")
            _validate_zone_visual_receipt(
                zone,
                relative_path=expected["visual_receipt"],
                expected_sha256=expected["visual_receipt_sha256"],
                technical_relative_path=qa_result.get("visual_technical_receipt"),
                technical_sha256=qa_result.get("visual_technical_receipt_sha256"),
                recipe_id=str(plan["recipe_id"]),
                recipe_build_id=str(source_lock["recipe_build_id"]),
                build_id=final_identity["build_id"],
            )
            if dict(receipt) != expected:
                raise ContractError("acceptance receipt is not canonical")
            return
        if receipt.get("schema") != PHASE_RECEIPT_SCHEMA:
            raise ContractError(f"unsupported receipt for phase {phase!r}")
        result = receipt.get("result")
        if not isinstance(result, dict):
            raise ContractError(f"phase {phase!r} receipt has no result object")
        rebuilt = self._build_phase_receipt(
            phase,
            {
                **result,
                "created_paths": [],
                "forbidden_c_artifacts": [],
            },
            zone=zone,
            plan=plan,
            source_lock=source_lock,
            max_tiles=max_tiles,
        )
        if dict(receipt) != rebuilt:
            raise ContractError(f"phase {phase!r} receipt is not canonical")

    def _complete_state_phase(
        self,
        state: dict[str, Any],
        zone: ZoneDefinition,
        phase: str,
        receipt_path: Path,
    ) -> None:
        phase_state = state["phases"][phase]
        phase_state.update(
            {
                "status": "completed",
                "receipt": str(receipt_path.relative_to(zone.run_root)).replace(
                    "\\", "/"
                ),
                "receipt_sha256": sha256_file(receipt_path),
                "completed_at_utc": utc_now(),
                "last_error": None,
            }
        )
        state["active_phase"] = None
        state["generation"] = int(state["generation"]) + 1
        state["updated_at_utc"] = utc_now()
        _atomic_write_json(zone.state_path, state)

    def _control_result(
        self,
        zone: ZoneDefinition,
        plan: Mapping[str, Any],
        state: Mapping[str, Any],
        phase: str,
        *,
        resumed: bool,
    ) -> dict[str, Any]:
        phase_state = state["phases"][phase]  # type: ignore[index]
        return {
            "schema": CONTROL_RESULT_SCHEMA,
            "mode": "resume" if resumed else "execute",
            "phase": phase,
            "status": phase_state["status"],
            "zone_id": zone.zone_id,
            "revision": zone.revision,
            "recipe_id": plan["recipe_id"],
            "plan_id": plan["plan_id"],
            "recipe_build_id": state.get("recipe_build_id"),
            "build_id": state.get("build_id"),
            "receipt": phase_state["receipt"],
            "receipt_sha256": phase_state["receipt_sha256"],
            "network_access_performed": False if phase == "plan" else None,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", action="append", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--package-workers", type=int, default=1)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None, *, controller: TerrainController | None = None
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all:
        parser.error("--all is forbidden; terrainctl accepts exactly one zone")
    if len(args.zone) != 1:
        parser.error("exactly one --zone manifest is required")
    if args.max_tiles is not None and args.phase != "pilot":
        parser.error("--max-tiles is only valid with --phase pilot")
    mode = "dry-run" if args.dry_run else "execute" if args.execute else "resume"
    if controller is None:
        from terrain_production_backend import build_default_backends

        active_controller = TerrainController(backends=build_default_backends())
    else:
        active_controller = controller
    result = active_controller.run(
        args.zone[0],
        phase=args.phase,
        mode=mode,
        download_workers=args.download_workers,
        package_workers=args.package_workers,
        max_tiles=args.max_tiles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
