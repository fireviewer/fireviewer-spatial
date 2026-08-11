"""Build deterministic, single-zone inputs for the adaptive terrain pipeline.

This module performs no network access and downloads no terrain data.  It only
turns one compact, square Lambert-93 zone request into the exact
``fireviewer.zone-spec.v1`` contract consumed by :mod:`terrainctl`.

The former ``prepare_incident_terrains.py`` / ``global-05m`` generator is
deprecated for new terrain production.  Only its already-approved incident
identities and square EPSG:2154 bounds are preserved in the locked catalogue;
no previous terrain, ground surface, asset placement or 0.5 m representation is
reused.  A new 1 m orthophoto window may be declared per tile, but only as a
temporary recognition input which is excluded from every runtime package.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit


INPUT_SCHEMA = "fireviewer.adaptive-zone-spec-input.v1"
ZONE_SPEC_SCHEMA = "fireviewer.zone-spec.v1"
CATALOG_SCHEMA = "fireviewer.adaptive-terrain-zone-catalog.v1"
CRS = "EPSG:2154"
TILE_SIZE_M = 500
HALO_M = 10
SOURCE_RESOLUTION_M = 2
ORTHOPHOTO_RESOLUTION_M = 1
ORTHOPHOTO_HALO_M = 10
DEFAULT_LICENSE = "Licence Ouverte Etalab 2.0"
DEPENDENCY_NAMES = (
    "algorithm",
    "clean_pbr_texture_library",
    "ground_texture_contract",
    "surface_correspondence_contract",
    "surface_correspondence_model",
    "surface_features",
    "terrain_quadtree_contract",
    "toolchain",
)
CATALOG_PATH = Path(__file__).with_name("adaptive_terrain_zone_catalog.v1.json")

# This content hash is deliberately duplicated outside the catalogue.  Editing
# a bound, count or production order therefore requires an explicit contract
# revision in code as well as a regenerated catalogue signature.
LOCKED_CATALOG_SHA256 = (
    "b0b07aed7147cc6086b2471645486ff51f6dfb6a5f0ef1955c36de12d3871059"
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ZoneSpecGenerationError(ValueError):
    """The compact input cannot produce an approved terrain zone contract."""


@dataclass(frozen=True)
class SourceEndpoint:
    """One immutable HTTPS elevation service and its selected layer."""

    service_url: str
    layer: str


@dataclass(frozen=True)
class OrthophotoEndpoint:
    """One temporary 1 m WMS/WMTS recognition source."""

    service_url: str
    layer: str
    source_revision_id: str
    service_kind: str = "wms"
    image_format: str = "image/png"
    maximum_download_bytes: int = 32 * 1024**2
    style: str = "normal"
    wmts_matrix: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AdaptiveZoneSpecRequest:
    """Compact, network-free input for exactly one adaptive terrain zone."""

    zone_id: str
    revision: str
    bounds_l93_m: tuple[int | float, int | float, int | float, int | float]
    mnt: SourceEndpoint
    mns: SourceEndpoint
    orthophoto: OrthophotoEndpoint
    source_revision_id: str
    dependency_artifacts: Mapping[str, str | Path]
    work_root: str | Path
    export_root: str | Path
    estimated_peak_bytes: int
    license: str = DEFAULT_LICENSE
    pilot_scores: Mapping[str, int | float] | None = None
    regression_tile_id: str | None = None


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ZoneSpecGenerationError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ZoneSpecGenerationError(f"{label} is not a safe stable identifier")
    return value


def _require_d_path(value: str | Path, label: str, *, base: Path | None = None) -> Path:
    path = Path(value)
    if not path.is_absolute() and base is not None:
        path = base / path
    resolved = path.resolve()
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise ZoneSpecGenerationError(f"{label} must stay on D:, got {resolved}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_bounds(
    raw_bounds: Sequence[int | float],
) -> tuple[int, int, int, int]:
    if len(raw_bounds) != 4:
        raise ZoneSpecGenerationError(
            "bounds_l93_m must contain west, south, east and north"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw_bounds
    ):
        raise ZoneSpecGenerationError("bounds_l93_m must contain finite numbers")
    if any(
        not math.isclose(float(value), round(float(value)), abs_tol=1e-9)
        for value in raw_bounds
    ):
        raise ZoneSpecGenerationError("bounds_l93_m must use whole Lambert-93 metres")
    west, south, east, north = (int(round(float(value))) for value in raw_bounds)
    side = east - west
    if side <= 0 or north - south != side:
        raise ZoneSpecGenerationError("bounds_l93_m must describe one non-empty square")
    if any(value % TILE_SIZE_M for value in (west, south, east, north)):
        raise ZoneSpecGenerationError(
            "bounds_l93_m must align to the global 500 m Lambert-93 grid"
        )
    return west, south, east, north


def _validate_endpoint(
    endpoint: SourceEndpoint, label: str, *, allow_orthophoto: bool = False
) -> SourceEndpoint:
    if not isinstance(endpoint, SourceEndpoint):
        raise ZoneSpecGenerationError(f"{label} must be a SourceEndpoint")
    service_url = endpoint.service_url.strip()
    parsed = urlsplit(service_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ZoneSpecGenerationError(
            f"{label}.service_url must be an HTTPS URL without credentials or fragment"
        )
    layer = endpoint.layer.strip()
    if not layer:
        raise ZoneSpecGenerationError(f"{label}.layer must be non-empty")
    if not allow_orthophoto and (
        "ortho" in service_url.casefold() or "ortho" in layer.casefold()
    ):
        raise ZoneSpecGenerationError("orthophoto sources are forbidden")
    return SourceEndpoint(service_url=service_url, layer=layer)


def _validate_orthophoto_endpoint(endpoint: OrthophotoEndpoint) -> OrthophotoEndpoint:
    if not isinstance(endpoint, OrthophotoEndpoint):
        raise ZoneSpecGenerationError(
            "sources.orthophoto must be an OrthophotoEndpoint"
        )
    base = _validate_endpoint(
        SourceEndpoint(endpoint.service_url, endpoint.layer),
        "sources.orthophoto",
        allow_orthophoto=True,
    )
    revision = _require_safe_id(
        endpoint.source_revision_id, "sources.orthophoto.source_revision_id"
    )
    if endpoint.service_kind not in {"wms", "wmts"}:
        raise ZoneSpecGenerationError(
            "sources.orthophoto.service_kind must be wms or wmts"
        )
    if endpoint.image_format not in {"image/png", "image/jpeg"}:
        raise ZoneSpecGenerationError(
            "sources.orthophoto.image_format must be image/png or image/jpeg"
        )
    if (
        isinstance(endpoint.maximum_download_bytes, bool)
        or not isinstance(endpoint.maximum_download_bytes, int)
        or endpoint.maximum_download_bytes <= 0
    ):
        raise ZoneSpecGenerationError(
            "sources.orthophoto.maximum_download_bytes must be positive"
        )
    if endpoint.service_kind == "wmts":
        if not isinstance(endpoint.style, str) or not endpoint.style.strip():
            raise ZoneSpecGenerationError("sources.orthophoto.style must be non-empty")
        if not isinstance(endpoint.wmts_matrix, Mapping):
            raise ZoneSpecGenerationError(
                "sources.orthophoto.wmts_matrix is required for WMTS"
            )
    elif endpoint.wmts_matrix is not None:
        raise ZoneSpecGenerationError(
            "sources.orthophoto.wmts_matrix is forbidden for WMS"
        )
    return OrthophotoEndpoint(
        service_url=base.service_url,
        layer=base.layer,
        source_revision_id=revision,
        service_kind=endpoint.service_kind,
        image_format=endpoint.image_format,
        maximum_download_bytes=endpoint.maximum_download_bytes,
        style=endpoint.style.strip(),
        wmts_matrix=(
            None
            if endpoint.wmts_matrix is None
            else json.loads(_canonical_json_bytes(dict(endpoint.wmts_matrix)))
        ),
    )


def _tile_id(west: int, south: int) -> str:
    return f"x{west:06d}_y{south:07d}_s{TILE_SIZE_M}"


def tile_cores(
    bounds_l93_m: Sequence[int | float],
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """Return every 500 m core in stable south-then-west order."""

    west, south, east, north = _validate_bounds(bounds_l93_m)
    return tuple(
        (
            _tile_id(tile_west, tile_south),
            (
                tile_west,
                tile_south,
                tile_west + TILE_SIZE_M,
                tile_south + TILE_SIZE_M,
            ),
        )
        for tile_south in range(south, north, TILE_SIZE_M)
        for tile_west in range(west, east, TILE_SIZE_M)
    )


def grid_summary(bounds_l93_m: Sequence[int | float]) -> dict[str, int]:
    west, south, east, _north = _validate_bounds(bounds_l93_m)
    side_m = east - west
    dimension = side_m // TILE_SIZE_M
    return {
        "side_m": side_m,
        "columns": dimension,
        "rows": dimension,
        "tile_count": dimension * dimension,
        "source_pair_count": dimension * dimension,
        "source_request_count": 3 * dimension * dimension,
        "seam_count": 2 * dimension * (dimension - 1),
    }


def load_locked_zone_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Load and verify the immutable six-zone bounds catalogue."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZoneSpecGenerationError(
            f"invalid adaptive zone catalogue: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ZoneSpecGenerationError("adaptive zone catalogue must be an object")
    unsigned = dict(payload)
    declared_hash = unsigned.pop("catalog_sha256", None)
    calculated_hash = canonical_sha256(unsigned)
    if (
        payload.get("schema") != CATALOG_SCHEMA
        or declared_hash != calculated_hash
        or declared_hash != LOCKED_CATALOG_SHA256
    ):
        raise ZoneSpecGenerationError("adaptive zone catalogue signature is invalid")

    zones = payload.get("zones")
    production_order = payload.get("production_order")
    totals = payload.get("totals")
    if (
        not isinstance(zones, list)
        or not isinstance(production_order, list)
        or not isinstance(totals, dict)
        or len(zones) != 6
    ):
        raise ZoneSpecGenerationError("adaptive zone catalogue structure is invalid")
    observed_order: list[str] = []
    tile_total = 0
    seam_total = 0
    bounds_basis: list[dict[str, Any]] = []
    for expected_order, raw_zone in enumerate(zones, start=1):
        if not isinstance(raw_zone, dict):
            raise ZoneSpecGenerationError("adaptive zone catalogue entry is invalid")
        zone_id = _require_safe_id(raw_zone.get("zone_id"), "catalogue zone_id")
        if raw_zone.get("order") != expected_order:
            raise ZoneSpecGenerationError("adaptive zone catalogue order is invalid")
        bounds = _validate_bounds(raw_zone.get("bounds_l93_m", ()))
        summary = grid_summary(bounds)
        for name in ("side_m", "tile_count", "source_pair_count", "seam_count"):
            if raw_zone.get(name) != summary[name]:
                raise ZoneSpecGenerationError(
                    f"adaptive zone catalogue {zone_id}/{name} is invalid"
                )
        observed_order.append(zone_id)
        tile_total += summary["tile_count"]
        seam_total += summary["seam_count"]
        bounds_basis.append({"zone_id": zone_id, "bounds_l93_m": list(bounds)})
    if production_order != observed_order:
        raise ZoneSpecGenerationError("production_order differs from catalogue entries")
    expected_totals = {
        "zone_count": len(zones),
        "tile_count": tile_total,
        "source_pair_count": tile_total,
        "seam_count": seam_total,
    }
    if totals != expected_totals:
        raise ZoneSpecGenerationError("adaptive zone catalogue totals are invalid")
    if payload.get("bounds_basis_sha256") != canonical_sha256(bounds_basis):
        raise ZoneSpecGenerationError(
            "adaptive zone catalogue bounds provenance changed"
        )
    return payload


def locked_catalog_entry(zone_id: str) -> dict[str, Any]:
    stable_id = _require_safe_id(zone_id, "zone_id")
    for entry in load_locked_zone_catalog()["zones"]:
        if entry["zone_id"] == stable_id:
            return json.loads(_canonical_json_bytes(entry).decode("utf-8"))
    raise ZoneSpecGenerationError(f"zone {stable_id!r} is absent from the catalogue")


def _normalize_dependency_artifacts(
    artifacts: Mapping[str, str | Path],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(DEPENDENCY_NAMES):
        raise ZoneSpecGenerationError(
            "dependency_artifacts must contain exactly: " + ", ".join(DEPENDENCY_NAMES)
        )
    dependencies: dict[str, str] = {}
    normalized: dict[str, dict[str, str]] = {}
    for name in DEPENDENCY_NAMES:
        path = _require_d_path(artifacts[name], f"dependency_artifacts.{name}")
        if not path.is_file():
            raise ZoneSpecGenerationError(
                f"dependency_artifacts.{name} is not a real file: {path}"
            )
        digest = sha256_file(path)
        dependencies[name] = digest
        normalized[name] = {"path": str(path), "sha256": digest}
    return dependencies, normalized


def build_zone_spec(request: AdaptiveZoneSpecRequest) -> dict[str, Any]:
    """Build one deterministic ``zone-spec.v1`` without touching the network."""

    if not isinstance(request, AdaptiveZoneSpecRequest):
        raise ZoneSpecGenerationError("request must be an AdaptiveZoneSpecRequest")
    zone_id = _require_safe_id(request.zone_id, "zone_id")
    revision = _require_safe_id(request.revision, "revision")
    source_revision_id = _require_safe_id(
        request.source_revision_id, "source_revision_id"
    )
    bounds = _validate_bounds(request.bounds_l93_m)
    mnt = _validate_endpoint(request.mnt, "sources.mnt")
    mns = _validate_endpoint(request.mns, "sources.mns")
    orthophoto = _validate_orthophoto_endpoint(request.orthophoto)
    if not isinstance(request.license, str) or not request.license.strip():
        raise ZoneSpecGenerationError("license must be non-empty")
    if (
        isinstance(request.estimated_peak_bytes, bool)
        or not isinstance(request.estimated_peak_bytes, int)
        or request.estimated_peak_bytes <= 0
    ):
        raise ZoneSpecGenerationError("estimated_peak_bytes must be positive")

    work_root = _require_d_path(request.work_root, "workspace.work_root")
    export_root = _require_d_path(request.export_root, "workspace.export_root")
    if (
        work_root == export_root
        or _is_relative_to(work_root, export_root)
        or _is_relative_to(export_root, work_root)
    ):
        raise ZoneSpecGenerationError(
            "workspace work_root and export_root must be separate trees"
        )
    dependencies, dependency_artifacts = _normalize_dependency_artifacts(
        request.dependency_artifacts
    )

    cores = tile_cores(bounds)
    valid_tile_ids = {tile_id for tile_id, _core in cores}
    raw_scores = request.pilot_scores or {}
    if not isinstance(raw_scores, Mapping):
        raise ZoneSpecGenerationError("pilot_scores must be a tile-id mapping")
    unknown_scores = sorted(set(raw_scores) - valid_tile_ids)
    if unknown_scores:
        raise ZoneSpecGenerationError(
            "pilot_scores reference tiles outside the zone: "
            + ", ".join(unknown_scores[:3])
        )
    scores: dict[str, float] = {}
    for tile_id, raw_score in raw_scores.items():
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
        ):
            raise ZoneSpecGenerationError(f"pilot score for {tile_id} must be finite")
        scores[tile_id] = float(raw_score)

    if request.regression_tile_id is not None:
        regression_tile_id = _require_safe_id(
            request.regression_tile_id, "regression_tile_id"
        )
        if regression_tile_id not in valid_tile_ids:
            raise ZoneSpecGenerationError(
                "regression_tile_id must identify a tile inside the zone"
            )
    else:
        regression_tile_id = None

    source_requests: list[dict[str, Any]] = []
    endpoints = {"mnt": mnt, "mns": mns}
    for product in ("mnt", "mns"):
        endpoint = endpoints[product]
        for tile_id, core in cores:
            source: dict[str, Any] = {
                "id": tile_id,
                "product": product,
                "request": {
                    "service_url": endpoint.service_url,
                    "layer": endpoint.layer,
                    "core_bounds_l93_m": list(core),
                    "resolution_m": SOURCE_RESOLUTION_M,
                },
                "source_revision_id": source_revision_id,
                "license": request.license.strip(),
            }
            if tile_id in scores:
                source["pilot_score"] = scores[tile_id]
            source_requests.append(source)
    for tile_id, core in cores:
        orthophoto_request: dict[str, Any] = {
            "service_kind": orthophoto.service_kind,
            "service_url": orthophoto.service_url,
            "layer": orthophoto.layer,
            "core_bounds_l93_m": list(core),
            "resolution_m": ORTHOPHOTO_RESOLUTION_M,
            "halo_m": ORTHOPHOTO_HALO_M,
            "image_format": orthophoto.image_format,
            "maximum_download_bytes": orthophoto.maximum_download_bytes,
        }
        if orthophoto.service_kind == "wmts":
            orthophoto_request.update(
                {
                    "style": orthophoto.style,
                    "wmts_matrix": dict(orthophoto.wmts_matrix or {}),
                }
            )
        source: dict[str, Any] = {
            "id": tile_id,
            "product": "orthophoto",
            "request": orthophoto_request,
            "source_revision_id": orthophoto.source_revision_id,
            "license": request.license.strip(),
        }
        if tile_id in scores:
            source["pilot_score"] = scores[tile_id]
        source_requests.append(source)
    source_requests.sort(key=lambda item: (item["product"], item["id"]))

    payload: dict[str, Any] = {
        "schema": ZONE_SPEC_SCHEMA,
        "zone_id": zone_id,
        "revision": revision,
        "crs": CRS,
        "bounds_l93_m": list(bounds),
        "tile_size_m": TILE_SIZE_M,
        "halo_m": HALO_M,
        "source_resolution_m": SOURCE_RESOLUTION_M,
        "dependencies": dependencies,
        "dependency_artifacts": dependency_artifacts,
        "source_requests": source_requests,
        "workspace": {
            "work_root": str(work_root),
            "export_root": str(export_root),
        },
        "storage": {"estimated_peak_bytes": request.estimated_peak_bytes},
    }
    if regression_tile_id is not None:
        payload["pilot"] = {"regression_tile_id": regression_tile_id}
    return json.loads(_canonical_json_bytes(payload).decode("utf-8"))


def _resolve_input_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ZoneSpecGenerationError(f"{label} must be a non-empty path")
    return _require_d_path(value, label, base=base)


def request_from_mapping(
    payload: Mapping[str, Any], *, base: Path
) -> AdaptiveZoneSpecRequest:
    """Parse the strict compact JSON input used by the CLI."""

    if not isinstance(payload, Mapping):
        raise ZoneSpecGenerationError("adaptive zone input must be an object")
    allowed = {
        "schema",
        "zone_id",
        "revision",
        "bounds_l93_m",
        "sources",
        "source_revision_id",
        "license",
        "dependency_artifacts",
        "workspace",
        "storage",
        "pilot_scores",
        "regression_tile_id",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ZoneSpecGenerationError(
            "unknown adaptive zone input field(s): " + ", ".join(unknown)
        )
    if payload.get("schema") != INPUT_SCHEMA:
        raise ZoneSpecGenerationError(f"schema must be exactly {INPUT_SCHEMA}")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "mnt",
        "mns",
        "orthophoto",
    }:
        raise ZoneSpecGenerationError(
            "sources must contain exactly mnt, mns and orthophoto"
        )

    endpoints: dict[str, SourceEndpoint] = {}
    for product in ("mnt", "mns"):
        endpoint = sources[product]
        if not isinstance(endpoint, Mapping) or set(endpoint) != {
            "service_url",
            "layer",
        }:
            raise ZoneSpecGenerationError(
                f"sources.{product} must contain only service_url and layer"
            )
        endpoints[product] = SourceEndpoint(
            service_url=str(endpoint["service_url"]),
            layer=str(endpoint["layer"]),
        )

    raw_orthophoto = sources["orthophoto"]
    if not isinstance(raw_orthophoto, Mapping):
        raise ZoneSpecGenerationError("sources.orthophoto must be an object")
    orthophoto_allowed = {
        "service_url",
        "layer",
        "source_revision_id",
        "service_kind",
        "image_format",
        "maximum_download_bytes",
        "style",
        "wmts_matrix",
    }
    unknown_orthophoto = sorted(set(raw_orthophoto) - orthophoto_allowed)
    if unknown_orthophoto:
        raise ZoneSpecGenerationError(
            "unknown sources.orthophoto field(s): " + ", ".join(unknown_orthophoto)
        )
    orthophoto_endpoint = OrthophotoEndpoint(
        service_url=str(raw_orthophoto.get("service_url", "")),
        layer=str(raw_orthophoto.get("layer", "")),
        source_revision_id=str(raw_orthophoto.get("source_revision_id", "")),
        service_kind=str(raw_orthophoto.get("service_kind", "wms")),
        image_format=str(raw_orthophoto.get("image_format", "image/png")),
        maximum_download_bytes=raw_orthophoto.get(  # type: ignore[arg-type]
            "maximum_download_bytes", 32 * 1024**2
        ),
        style=str(raw_orthophoto.get("style", "normal")),
        wmts_matrix=raw_orthophoto.get("wmts_matrix"),  # type: ignore[arg-type]
    )

    raw_artifacts = payload.get("dependency_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ZoneSpecGenerationError("dependency_artifacts must be an object")
    artifacts = {
        str(name): _resolve_input_path(
            value, base=base, label=f"dependency_artifacts.{name}"
        )
        for name, value in raw_artifacts.items()
    }
    workspace = payload.get("workspace")
    if not isinstance(workspace, Mapping) or set(workspace) != {
        "work_root",
        "export_root",
    }:
        raise ZoneSpecGenerationError(
            "workspace must contain only work_root and export_root"
        )
    storage = payload.get("storage")
    if not isinstance(storage, Mapping) or set(storage) != {"estimated_peak_bytes"}:
        raise ZoneSpecGenerationError("storage must contain only estimated_peak_bytes")
    bounds = payload.get("bounds_l93_m")
    if not isinstance(bounds, list):
        raise ZoneSpecGenerationError("bounds_l93_m must be an array")
    pilot_scores = payload.get("pilot_scores")
    if pilot_scores is not None and not isinstance(pilot_scores, Mapping):
        raise ZoneSpecGenerationError("pilot_scores must be an object")
    regression_tile_id = payload.get("regression_tile_id")
    if regression_tile_id is not None and not isinstance(regression_tile_id, str):
        raise ZoneSpecGenerationError("regression_tile_id must be a string")
    license_name = payload.get("license", DEFAULT_LICENSE)
    if not isinstance(license_name, str):
        raise ZoneSpecGenerationError("license must be a string")
    return AdaptiveZoneSpecRequest(
        zone_id=str(payload.get("zone_id", "")),
        revision=str(payload.get("revision", "")),
        bounds_l93_m=tuple(bounds),  # type: ignore[arg-type]
        mnt=endpoints["mnt"],
        mns=endpoints["mns"],
        orthophoto=orthophoto_endpoint,
        source_revision_id=str(payload.get("source_revision_id", "")),
        dependency_artifacts=artifacts,
        work_root=_resolve_input_path(
            workspace["work_root"], base=base, label="workspace.work_root"
        ),
        export_root=_resolve_input_path(
            workspace["export_root"], base=base, label="workspace.export_root"
        ),
        estimated_peak_bytes=storage["estimated_peak_bytes"],  # type: ignore[arg-type]
        license=license_name,
        pilot_scores=pilot_scores,  # type: ignore[arg-type]
        regression_tile_id=regression_tile_id,
    )


def load_generation_request(path: Path) -> AdaptiveZoneSpecRequest:
    resolved = _require_d_path(path, "--input")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZoneSpecGenerationError(f"invalid adaptive zone input: {exc}") from exc
    if not isinstance(payload, dict):
        raise ZoneSpecGenerationError("adaptive zone input must be an object")
    return request_from_mapping(payload, base=resolved.parent)


def write_zone_spec(
    request: AdaptiveZoneSpecRequest, output_path: Path
) -> dict[str, Any]:
    """Atomically write one terrainctl-compatible zone spec on D:."""

    output = _require_d_path(output_path, "--output")
    payload = build_zone_spec(request)

    # Keep the public generator coupled to the exact active terrainctl contract
    # without importing or invoking its production backend.
    try:
        from terrainctl import validate_zone_spec
    except ModuleNotFoundError:  # Direct execution from the blender directory.
        import sys

        repository_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repository_root))
        from terrainctl import validate_zone_spec

    validate_zone_spec(payload, spec_path=output)
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    request = load_generation_request(arguments.input)
    payload = write_zone_spec(request, arguments.output)
    print(
        json.dumps(
            {
                "schema": "fireviewer.adaptive-zone-spec-generation-result.v1",
                "zone_id": payload["zone_id"],
                "revision": payload["revision"],
                "output": str(arguments.output.resolve()),
                "summary": grid_summary(payload["bounds_l93_m"]),
                "network_access_performed": False,
                "downloads_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
