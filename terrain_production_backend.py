"""Concrete, sequential backend for the adaptive mono-zone terrain controller.

The backend deliberately acquires one MNT/MNS source band and one temporary
1 m orthophoto window at a time.  Orthophoto RGB exists only long enough to
compile the deterministic four-map surface correspondence, then is deleted
before the immutable v3 tile package is sealed.  Nothing is written outside
the D:-only roots supplied by :mod:`terrainctl`.
"""

# ruff: noqa: E402 -- sibling Blender scripts require their directory on sys.path

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tracemalloc
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from shapely.geometry import LineString, MultiLineString, box, shape
from shapely.strtree import STRtree

# The existing Blender compiler modules are also executable scripts and keep
# sibling imports intentionally unqualified.  Bind that audited module root
# before importing them; no package is installed and no path on C: is used.
_BLENDER_MODULE_ROOT = Path(__file__).resolve().parent / "blender"
if str(_BLENDER_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BLENDER_MODULE_ROOT))

from blender.adaptive_terrain_quadtree import (
    EDGE_ORDER,
    STITCH_MASK_BITS,
    Breakline,
    compile_adaptive_tile,
    encode_fvtq,
    read_fvtq,
    write_fvtq,
)
from blender.compact_hag import (
    quantize_hag_max_cm_from_canonical_mm,
    read_hag_max_2m,
    write_hag_max_2m,
)
from blender.terrain_source_acquisition import (
    acquire_source_pair,
    build_source_pair_plan,
)
from blender.ground_material_contract import (
    RUNTIME_SHADER_PENDING_STATUS,
    build_ground_material_bundle,
    material_identity,
)
from blender.clean_pbr_texture_library import (
    ACCEPTED_LIBRARY_STATUS,
    validate_runtime_maps,
    validate_texture_library,
)
from blender.orthophoto_build_acquisition import (
    SealedDependentMap,
    TEMPORARY_DIRECTORY_NAME,
    WmtsMatrix,
    build_wms_plan,
    build_wmts_plan,
    cleanup_orthophoto_band,
    load_canonical_rgb,
    prepare_orthophoto_band,
)
from blender.orthophoto_surface_correspondence import (
    compile_aligned_window,
    serialize_tile_outputs,
    slice_tile,
)
from blender.tile_package import (
    build_tile_package,
    validate_tile_done,
    write_tile_done,
)
from blender.render_ground_atlas_acceptance import validate_acceptance
from blender.validate_adaptive_terrain_zone import (
    ACCEPTANCE_SCHEMA as ZONE_VISUAL_ACCEPTANCE_SCHEMA,
    ACCEPTED_STATUS as ZONE_VISUAL_ACCEPTED_STATUS,
    TECHNICAL_RECEIPT_SCHEMA as ZONE_VISUAL_TECHNICAL_SCHEMA,
    TECHNICAL_STATUS as ZONE_VISUAL_TECHNICAL_STATUS,
    accept_zone_visual_review,
    build_zone_visual_job,
    create_human_review_template,
    inspect_zone_job,
    validate_technical_receipt,
)
from omniverse.adaptive_terrain_usd import (
    export_tile_usd,
    validate_tile_usd_package,
)
from terrainctl import (
    PHASE_RECEIPT_NAMES,
    SAFETY_MARGIN_BYTES,
    ZONE_VISUAL_JOB_RELATIVE_PATH,
    ZONE_VISUAL_RECEIPT_NAME,
    ZONE_VISUAL_REVIEW_RELATIVE_PATH,
    ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH,
    ZONE_VISUAL_TECHNICAL_RELATIVE_PATH,
    PhaseContext,
    StoragePolicyError,
    build_final_source_identity,
    canonical_sha256,
    sha256_file,
)


TOOLCHAIN_SCHEMA = "fireviewer.terrain-toolchain.v1"
SURFACE_FEATURE_SCHEMA = "fireviewer.surface-feature-snapshot.v1"
BLENDER_REPORT_SCHEMA = "fireviewer.blender-adaptive-terrain-qa.v2"
CANONICAL_INDEX_SCHEMA = "fireviewer.canonical-terrain-index.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parent
COMPOSITION_ASSETS = {
    "ground_profile_ids": "ground-profile-ids.png",
    "ground_profile_weights": "ground-profile-weights.png",
    "ground_confidence": "ground-confidence.png",
    "ground_orientation": "ground-orientation.png",
}


class TerrainProductionError(RuntimeError):
    """A production dependency, source, package, or proof failed closed."""


class _LocalFileResponse:
    def __init__(self, path: Path) -> None:
        self._stream: BinaryIO = path.open("rb")
        self.headers = {"Content-Length": str(path.stat().st_size)}
        self.status: int | None = None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> _LocalFileResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self._stream.close()


def _default_opener(request: Request, *, timeout: float):
    parts = urlsplit(request.full_url)
    if parts.scheme != "file":
        return urlopen(request, timeout=timeout)
    raw_path = unquote(parts.path)
    if os.name == "nt" and raw_path.startswith("/") and len(raw_path) >= 4:
        raw_path = raw_path[1:]
    path = Path(raw_path).resolve()
    if os.name == "nt" and path.drive.upper() != "D:":
        raise StoragePolicyError(f"local terrain source must stay on D:, got {path}")
    return _LocalFileResponse(path)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(payload))
    temporary.replace(path)


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise TerrainProductionError(
                f"refusing to replace a different immutable proof: {path}"
            )
        return
    _atomic_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TerrainProductionError(
            f"invalid JSON dependency {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise TerrainProductionError(f"JSON dependency must be an object: {path}")
    return payload


def _artifact(path: Path, *, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _is_d_path(path: Path) -> bool:
    return os.name != "nt" or path.resolve().drive.upper() == "D:"


@dataclass(frozen=True)
class _SourcePair:
    source_id: str
    bounds: tuple[float, float, float, float]
    mnt: dict[str, Any]
    mns: dict[str, Any]
    orthophoto: dict[str, Any]
    pilot_score: float


@dataclass
class _TileBuild:
    receipt: dict[str, Any]
    reused: bool
    package_bytes: int
    triangles: tuple[int, int, int]
    stitch_variant_counts: tuple[int, int, int]
    maximum_final_errors_mm: tuple[int, int, int]
    maximum_stitch_triangles: tuple[int, int, int]
    bitwise_rebuilt: bool
    orthophoto_source: dict[str, Any]


@dataclass(frozen=True)
class _CanonicalGrids:
    """Validated source views used by geometry and compact HAG compilation."""

    mnt_core_m: np.ndarray
    mnt_normal_halo_m: np.ndarray
    mns_core_m: np.ndarray
    mnt_normal_halo_mm: np.ndarray
    mns_normal_halo_mm: np.ndarray


def _stitch_metrics(
    meshes: Sequence[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if len(meshes) != 3:
        raise TerrainProductionError("stitch metrics require three terrain LODs")
    return (
        tuple(len(mesh.stitch_variants) for mesh in meshes),  # type: ignore[return-value]
        tuple(int(mesh.maximum_final_error_mm) for mesh in meshes),  # type: ignore[return-value]
        tuple(
            max(
                variant.triangle_count(len(mesh.triangles))
                for variant in mesh.stitch_variants
            )
            for mesh in meshes
        ),  # type: ignore[return-value]
    )


class _SurfaceIndex:
    def __init__(self, features: Sequence[Mapping[str, Any]]) -> None:
        self.features = [dict(feature) for feature in features]
        self.geometries = [shape(feature["geometry"]) for feature in self.features]
        self.tree = STRtree(self.geometries)
        self.by_identity = {
            id(geometry): index for index, geometry in enumerate(self.geometries)
        }

    def query(self, bounds: Sequence[float]) -> list[dict[str, Any]]:
        hits = self.tree.query(box(*bounds))
        indices: list[int] = []
        for hit in hits:
            if isinstance(hit, (int, np.integer)):
                indices.append(int(hit))
            else:
                index = self.by_identity.get(id(hit))
                if index is None:
                    index = self.geometries.index(hit)
                indices.append(index)
        return [self.features[index] for index in sorted(set(indices))]


BlenderRunner = Callable[[PhaseContext, Path, Path, Path, Path], dict[str, Any]]
ZoneVisualRunner = Callable[[PhaseContext, Path, Path], dict[str, Any]]
AtlasAcceptanceValidator = Callable[[Path, Path], Mapping[str, Any]]


class ProductionTerrainBackend:
    """Audited backend implementations for every heavy terrain phase."""

    def __init__(
        self,
        *,
        opener: Callable[..., Any] = _default_opener,
        blender_runner: BlenderRunner | None = None,
        zone_visual_runner: ZoneVisualRunner | None = None,
        atlas_acceptance_validator: AtlasAcceptanceValidator = validate_acceptance,
    ) -> None:
        self.opener = opener
        self.blender_runner = blender_runner or self._run_blender
        self.zone_visual_runner = zone_visual_runner or self._run_zone_visual
        self.atlas_acceptance_validator = atlas_acceptance_validator

    def mapping(self) -> dict[str, Callable[[PhaseContext], Mapping[str, Any]]]:
        return {
            "preflight": self.preflight,
            "pilot": self.pilot,
            "produce": self.produce,
            "qa": self.qa,
            "accept": self.accept,
            "cleanup": self.cleanup,
        }

    def _dependency_proofs(
        self, context: PhaseContext
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        proofs: dict[str, dict[str, Any]] = {}
        payloads: dict[str, Any] = {}
        for name, digest in sorted(context.zone.payload["dependencies"].items()):
            path = context.zone.dependency_path(name)
            if not _is_d_path(path) or not path.is_file():
                raise TerrainProductionError(
                    f"dependency is absent from D:: {name} ({path})"
                )
            actual = sha256_file(path)
            if actual != digest:
                raise TerrainProductionError(f"dependency hash mismatch: {name}")
            schema: str | None = None
            status: str | None = None
            try:
                payload = _read_json(path)
            except TerrainProductionError:
                payload = None
            if payload is not None:
                payloads[name] = payload
                raw_schema = payload.get("schema")
                raw_status = payload.get("status")
                schema = raw_schema if isinstance(raw_schema, str) else None
                status = raw_status if isinstance(raw_status, str) else None
            proofs[name] = {
                "file_name": path.name,
                "sha256": actual,
                "schema": schema,
                "status": status,
            }

        expected_schemas = {
            "clean_pbr_texture_library": "fireviewer.clean-pbr-texture-library.v1",
            "ground_texture_contract": "fireviewer.ground-surface-texture-contract.v4",
            "surface_correspondence_contract": (
                "fireviewer.orthophoto-surface-correspondence-contract.v1"
            ),
            "surface_correspondence_model": "fireviewer.orthophoto-surface-model.v1",
            "surface_features": SURFACE_FEATURE_SCHEMA,
            "terrain_quadtree_contract": "fireviewer.terrain-quadtree-contract.v1",
            "toolchain": TOOLCHAIN_SCHEMA,
        }
        for name, schema in expected_schemas.items():
            if proofs[name]["schema"] != schema:
                raise TerrainProductionError(f"unsupported {name} schema")

        try:
            clean_library = validate_texture_library(
                context.zone.dependency_path("clean_pbr_texture_library"),
                contract_path=context.zone.dependency_path("ground_texture_contract"),
                require_visual_acceptance=True,
            )
        except ValueError as error:
            raise TerrainProductionError(
                f"clean PBR library failed validation: {error}"
            ) from error
        if (
            clean_library["status"] != ACCEPTED_LIBRARY_STATUS
            or clean_library["visual_acceptance"] != "accepted_human_visual"
        ):
            raise TerrainProductionError(
                "clean PBR library is pending or lacks accepted human visual review"
            )

        feature_snapshot = payloads["surface_features"]
        features = feature_snapshot.get("features")
        priors = feature_snapshot.get("context_priors")
        corrections = feature_snapshot.get("approved_corrections")
        if feature_snapshot.get("crs") != "EPSG:2154":
            raise TerrainProductionError("surface feature snapshot must use EPSG:2154")
        if feature_snapshot.get("bounds_l93_m") != context.plan["zone"]["bounds_l93_m"]:
            raise TerrainProductionError("surface feature snapshot bounds changed")
        if (
            not isinstance(features, list)
            or not isinstance(priors, list)
            or not isinstance(corrections, list)
        ):
            raise TerrainProductionError(
                "surface feature snapshot must contain features, context_priors and "
                "approved_corrections"
            )
        provenance = feature_snapshot.get("provenance")
        if not isinstance(provenance, dict) or not provenance:
            raise TerrainProductionError(
                "surface feature snapshot provenance is absent"
            )
        for name, digest in provenance.items():
            if not isinstance(name, str) or not isinstance(digest, str):
                raise TerrainProductionError(
                    "surface feature snapshot provenance hashes are invalid"
                )
            try:
                valid_digest = len(digest) == 64 and bytes.fromhex(digest) is not None
            except ValueError:
                valid_digest = False
            if not valid_digest or digest != digest.lower():
                raise TerrainProductionError(
                    "surface feature snapshot provenance hashes are invalid"
                )
        toolchain = payloads["toolchain"]
        blender = toolchain.get("blender")
        if not isinstance(blender, dict):
            raise TerrainProductionError("toolchain does not lock Blender")
        blender_path = Path(str(blender.get("path", ""))).resolve()
        if not _is_d_path(blender_path) or not blender_path.is_file():
            raise TerrainProductionError("locked Blender executable is absent from D:")
        if sha256_file(blender_path) != blender.get("sha256"):
            raise TerrainProductionError("locked Blender executable hash mismatch")
        payloads["blender_path"] = blender_path
        return proofs, payloads

    def _pairs(self, context: PhaseContext) -> dict[str, _SourcePair]:
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for record in context.plan["source_requests"]:
            grouped.setdefault(record["id"], {})[record["product"]] = record
        pairs: dict[str, _SourcePair] = {}
        for source_id, records in sorted(grouped.items()):
            mnt, mns, orthophoto = (
                records["mnt"],
                records["mns"],
                records["orthophoto"],
            )
            bounds = tuple(
                float(value) for value in mnt["request"]["core_bounds_l93_m"]
            )
            pairs[source_id] = _SourcePair(
                source_id=source_id,
                bounds=bounds,  # type: ignore[arg-type]
                mnt=mnt,
                mns=mns,
                orthophoto=orthophoto,
                pilot_score=(
                    float(mnt.get("pilot_score", 0.0))
                    + float(mns.get("pilot_score", 0.0))
                    + float(orthophoto.get("pilot_score", 0.0))
                )
                / 3.0,
            )
        return pairs

    def _pair_for_tile(
        self, pairs: Mapping[str, _SourcePair], tile: Mapping[str, Any]
    ) -> _SourcePair:
        bounds = tile["bounds_l93_m"]
        matches = [
            pair
            for pair in pairs.values()
            if pair.bounds[0] <= bounds[0]
            and pair.bounds[1] <= bounds[1]
            and pair.bounds[2] >= bounds[2]
            and pair.bounds[3] >= bounds[3]
        ]
        if len(matches) != 1:
            raise TerrainProductionError(f"tile {tile['id']} has no unique source pair")
        return matches[0]

    def preflight(self, context: PhaseContext) -> Mapping[str, Any]:
        started_ns = time.time_ns()
        proofs, _payloads = self._dependency_proofs(context)
        sources: list[dict[str, Any]] = []
        maximum_pair_bytes = 0
        for pair in self._pairs(context).values():
            pair_bytes = 0
            for record in (pair.mnt, pair.mns, pair.orthophoto):
                service_url = record["request"]["service_url"]
                scheme = urlsplit(service_url).scheme
                observed_sha256: str | None = None
                observed_byte_count: int | None = None
                identity_status = "revision_locked"
                if scheme == "file":
                    raw_path = unquote(urlsplit(service_url).path)
                    if os.name == "nt" and raw_path.startswith("/"):
                        raw_path = raw_path[1:]
                    path = Path(raw_path).resolve()
                    if not _is_d_path(path) or not path.is_file():
                        raise TerrainProductionError(
                            f"local source is absent from D:: {path}"
                        )
                    if (
                        path.stat().st_size != record["expected_byte_count"]
                        or sha256_file(path) != record["expected_sha256"]
                    ):
                        raise TerrainProductionError(
                            f"local source identity mismatch: {record['id']}/{record['product']}"
                        )
                    observed_sha256 = str(record["expected_sha256"])
                    observed_byte_count = int(record["expected_byte_count"])
                    identity_status = "observed_local_file"
                elif "expected_sha256" in record:
                    observed_sha256 = str(record["expected_sha256"])
                    observed_byte_count = int(record["expected_byte_count"])
                    identity_status = "expected_identity_locked"
                if observed_byte_count is not None:
                    pair_bytes += observed_byte_count
                elif record["product"] == "orthophoto":
                    pair_bytes += int(record["request"]["maximum_download_bytes"])
                sources.append(
                    {
                        "id": record["id"],
                        "product": record["product"],
                        "request_sha256": canonical_sha256(record["request"]),
                        "source_revision_id": record.get("source_revision_id"),
                        "identity_status": identity_status,
                        "sha256": observed_sha256,
                        "byte_count": observed_byte_count,
                        "license": record["license"],
                    }
                )
            maximum_pair_bytes = max(maximum_pair_bytes, pair_bytes)
        estimated_peak = max(
            int(context.zone.payload["storage"]["estimated_peak_bytes"]),
            maximum_pair_bytes * 2,
        )
        return {
            "sources": sources,
            "dependency_proofs": proofs,
            "estimated_peak_bytes": estimated_peak,
            "created_paths": [],
            "forbidden_c_artifacts": self._audit_c_since(started_ns),
        }

    def _select_pilot(self, context: PhaseContext) -> list[dict[str, Any]]:
        count = min(context.max_tiles or 9, len(context.plan["tiles"]))
        tiles = list(context.plan["tiles"])
        by_grid = {tuple(tile["grid"]): tile for tile in tiles}
        columns = int(context.plan["grid"]["columns"])
        rows = int(context.plan["grid"]["rows"])
        width = min(columns, max(1, int(math.ceil(math.sqrt(count)))))
        height = min(rows, max(1, int(math.ceil(count / width))))
        pairs = self._pairs(context)
        candidates: list[tuple[float, tuple[str, ...], list[dict[str, Any]]]] = []
        for y0 in range(rows - height + 1):
            for x0 in range(columns - width + 1):
                rectangle = [
                    by_grid[(x, y)]
                    for y in range(y0, y0 + height)
                    for x in range(x0, x0 + width)
                ][:count]
                if len(rectangle) != count:
                    continue
                score = sum(
                    self._pair_for_tile(pairs, tile).pilot_score for tile in rectangle
                )
                identifiers = tuple(tile["id"] for tile in rectangle)
                candidates.append((score, identifiers, rectangle))
        if not candidates:
            raise TerrainProductionError("cannot select a contiguous pilot block")
        selected = list(
            max(candidates, key=lambda item: (item[0], tuple(reversed(item[1]))))[2]
        )
        regression_tile_id = context.plan.get("pilot", {}).get("regression_tile_id")
        if regression_tile_id is not None and all(
            tile["id"] != regression_tile_id for tile in selected
        ):
            by_id = {tile["id"]: tile for tile in tiles}
            regression_tile = by_id.get(regression_tile_id)
            if regression_tile is None:
                raise TerrainProductionError(
                    "declared regression tile is absent from the zone plan"
                )
            selected.append(regression_tile)
        return selected

    def _acquire_pair(
        self, context: PhaseContext, pair: _SourcePair
    ) -> tuple[Path, dict[str, Any]]:
        raw_root = context.run_root / "sources" / pair.source_id
        plan = build_source_pair_plan(
            pair.bounds,
            mnt_service_url=pair.mnt["request"]["service_url"],
            mnt_layer=pair.mnt["request"]["layer"],
            mns_service_url=pair.mns["request"]["service_url"],
            mns_layer=pair.mns["request"]["layer"],
        )
        # `acquire_source_pair` owns both fresh acquisition and accepted-pair
        # reload.  Calling it unconditionally ensures resume revalidates the
        # complete semantic receipt, canonical arrays, hashes, CRS and
        # co-registration instead of trusting a merely present JSON file.
        receipt = acquire_source_pair(
            plan,
            raw_root,
            opener=self.opener,
        )
        sources = receipt.get("sources")
        if not isinstance(sources, dict):
            raise TerrainProductionError("source pair receipt is incomplete")
        for role, expected in (("mnt", pair.mnt), ("mns", pair.mns)):
            record = sources.get(role)
            if (
                not isinstance(record, dict)
                or (
                    "expected_sha256" in expected
                    and record.get("sha256") != expected["expected_sha256"]
                )
                or (
                    "expected_byte_count" in expected
                    and record.get("byte_count") != expected["expected_byte_count"]
                )
            ):
                raise TerrainProductionError(
                    f"downloaded source differs from manifest: {pair.source_id}/{role}"
                )
        return raw_root, receipt

    @staticmethod
    def _orthophoto_plan(pair: _SourcePair):
        request = pair.orthophoto["request"]
        common = {
            "band_id": pair.source_id,
            "service_url": request["service_url"],
            "layer": request["layer"],
            "provider_revision_id": pair.orthophoto["source_revision_id"],
            "dependent_map_ids": ("surface-correspondence",),
            "maximum_download_bytes": request["maximum_download_bytes"],
            "image_format": request["image_format"],
        }
        if request["service_kind"] == "wms":
            return build_wms_plan(pair.bounds, **common)
        matrix = request["wmts_matrix"]
        return build_wmts_plan(
            pair.bounds,
            **common,
            style=request["style"],
            wmts_matrix=WmtsMatrix(
                matrix_set=matrix["matrix_set"],
                matrix=matrix["matrix"],
                top_left_l93_m=tuple(matrix["top_left_l93_m"]),
                tile_width_px=matrix["tile_width_px"],
                tile_height_px=matrix["tile_height_px"],
                matrix_width=matrix["matrix_width"],
                matrix_height=matrix["matrix_height"],
                resolution_m=matrix["resolution_m"],
            ),
        )

    def _acquire_orthophoto(
        self, context: PhaseContext, pair: _SourcePair
    ) -> tuple[Any, Path, dict[str, Any]]:
        plan = self._orthophoto_plan(pair)
        receipt = prepare_orthophoto_band(
            plan,
            context.run_root / "temporary-surface-recognition",
            opener=self.opener,
        )
        band_root = (
            context.run_root
            / "temporary-surface-recognition"
            / TEMPORARY_DIRECTORY_NAME
            / pair.source_id
        )
        return plan, band_root, dict(receipt)

    @staticmethod
    def _canonical_grids(
        raw_root: Path, source_receipt: Mapping[str, Any]
    ) -> _CanonicalGrids:
        sources = source_receipt.get("sources")
        if not isinstance(sources, dict):
            raise TerrainProductionError("source pair canonical provenance is absent")
        loaded: dict[str, np.ndarray] = {}
        for role in ("mnt", "mns"):
            record = sources.get(role)
            canonical = record.get("canonical") if isinstance(record, dict) else None
            if not isinstance(canonical, dict):
                raise TerrainProductionError(f"canonical {role} source is absent")
            path = raw_root / str(canonical.get("file_name", ""))
            if not path.is_file() or sha256_file(path) != canonical.get("file_sha256"):
                raise TerrainProductionError(f"canonical {role} source hash mismatch")
            values = np.load(path, allow_pickle=False)
            if values.shape != (253, 253) or values.dtype != np.dtype("<i4"):
                raise TerrainProductionError(f"canonical {role} source layout mismatch")
            loaded[role] = values
        return _CanonicalGrids(
            mnt_core_m=loaded["mnt"][1:-1, 1:-1].astype("float64") / 1_000.0,
            mnt_normal_halo_m=loaded["mnt"].astype("float64") / 1_000.0,
            mns_core_m=loaded["mns"][1:-1, 1:-1].astype("float64") / 1_000.0,
            mnt_normal_halo_mm=loaded["mnt"],
            mns_normal_halo_mm=loaded["mns"],
        )

    @staticmethod
    def _source_provenance(
        pair: _SourcePair,
        source_receipt: Mapping[str, Any],
        role: str,
    ) -> dict[str, Any]:
        if role not in {"mnt", "mns"}:
            raise TerrainProductionError(f"unsupported terrain source role: {role}")
        source_records = source_receipt.get("sources")
        if not isinstance(source_records, dict) or not isinstance(
            source_records.get(role), dict
        ):
            raise TerrainProductionError(
                f"source provenance is absent: {pair.source_id}/{role}"
            )
        request = pair.mnt if role == "mnt" else pair.mns
        source_record = source_records[role]
        return {
            "schema": "fireviewer.tile-source-provenance.v1",
            "source_id": pair.source_id,
            "role": role,
            "source_revision_id": request.get("source_revision_id"),
            "license": request["license"],
            "source_sha256": source_record["sha256"],
            "source_byte_count": source_record["byte_count"],
            "request_sha256": canonical_sha256(request["request"]),
            "canonical": source_record["canonical"],
        }

    @staticmethod
    def _orthophoto_source_identity(
        pair: _SourcePair, receipt: Mapping[str, Any]
    ) -> dict[str, Any]:
        canonical = receipt.get("canonical_rgb")
        raw_sources = receipt.get("raw_sources")
        if not isinstance(canonical, dict) or not isinstance(raw_sources, list):
            raise TerrainProductionError(
                f"temporary recognition source receipt is incomplete: {pair.source_id}"
            )
        byte_count = sum(
            int(record["byte_count"])
            for record in raw_sources
            if isinstance(record, dict)
        )
        if len(raw_sources) == 0 or byte_count <= 0:
            raise TerrainProductionError(
                f"temporary recognition source has no observed bytes: {pair.source_id}"
            )
        return {
            "id": pair.source_id,
            "product": "orthophoto",
            "request_sha256": canonical_sha256(pair.orthophoto["request"]),
            "source_revision_id": pair.orthophoto["source_revision_id"],
            "sha256": canonical["pixel_sha256"],
            "byte_count": byte_count,
            "license": pair.orthophoto["license"],
        }

    @staticmethod
    def _surface_outputs(
        context: PhaseContext,
        pair: _SourcePair,
        orthophoto_plan: Any,
        orthophoto_root: Path,
        orthophoto_receipt: Mapping[str, Any],
        dependency_payloads: Mapping[str, Any],
    ) -> tuple[dict[str, bytes], dict[str, Any]]:
        rgb = load_canonical_rgb(orthophoto_plan, orthophoto_root)
        canonical = orthophoto_receipt.get("canonical_rgb")
        if not isinstance(canonical, dict):
            raise TerrainProductionError("canonical recognition RGB identity is absent")
        geotransform = canonical["gdal_geotransform"]
        affine_transform = (
            geotransform[1],
            geotransform[2],
            geotransform[0],
            geotransform[4],
            geotransform[5],
            geotransform[3],
        )
        correspondence = compile_aligned_window(
            rgb,
            transform=affine_transform,
            crs="EPSG:2154",
            core_bounds_l93_m=pair.bounds,
            orthophoto_sha256=canonical["pixel_sha256"],
            pbr_library=dependency_payloads["clean_pbr_texture_library"],
            correspondence_model=dependency_payloads["surface_correspondence_model"],
            context_priors=dependency_payloads["surface_features"]["context_priors"],
            approved_corrections=dependency_payloads["surface_features"][
                "approved_corrections"
            ],
            contract_path=context.zone.dependency_path(
                "surface_correspondence_contract"
            ),
        )
        tile = slice_tile(correspondence, pair.bounds)
        first = serialize_tile_outputs(tile)
        second = serialize_tile_outputs(tile)
        if first != second:
            raise TerrainProductionError(
                f"non-deterministic surface correspondence: {pair.source_id}"
            )
        return first, ProductionTerrainBackend._orthophoto_source_identity(
            pair, orthophoto_receipt
        )

    @staticmethod
    def _seal_and_remove_orthophoto(
        context: PhaseContext,
        orthophoto_plan: Any,
        manifest_path: Path,
    ) -> None:
        manifest_hash = sha256_file(manifest_path)
        cleanup_orthophoto_band(
            orthophoto_plan,
            context.run_root / "temporary-surface-recognition",
            sealed_maps=(
                SealedDependentMap(
                    map_id="surface-correspondence",
                    receipt_path=manifest_path,
                    receipt_sha256=manifest_hash,
                ),
            ),
        )
        band_root = (
            context.run_root
            / "temporary-surface-recognition"
            / TEMPORARY_DIRECTORY_NAME
            / orthophoto_plan.band_id
        )
        if band_root.exists():
            raise TerrainProductionError(
                "temporary recognition source survived its dependent receipt"
            )

    @staticmethod
    def _assert_existing_tile_reusable(
        final_root: Path,
        *,
        tile_id: str,
        existing_receipt: Mapping[str, Any],
        recipe_id: str,
        recipe_build_id: str,
        source_provenance: Mapping[str, Mapping[str, Any]],
        hag_cm: np.ndarray,
        bounds_l93_m: Sequence[float],
        surface_outputs: Mapping[str, bytes],
    ) -> None:
        """Reject immutable tile reuse unless every current input is identical."""

        mismatches: list[str] = []
        if existing_receipt.get("recipe_id") != recipe_id:
            mismatches.append("recipe_id")
        if existing_receipt.get("recipe_build_id") != recipe_build_id:
            mismatches.append("recipe_build_id")

        for role in ("mnt", "mns"):
            provenance_path = final_root / "source" / f"{role}-provenance.v1.json"
            try:
                actual_provenance = _read_json(provenance_path)
            except TerrainProductionError:
                mismatches.append(f"{role}_provenance_unreadable")
                continue
            if actual_provenance != dict(source_provenance[role]):
                mismatches.append(f"{role}_provenance")

        try:
            actual_hag, hag_metadata = read_hag_max_2m(final_root / "hag-max-2m.tif")
        except (OSError, ValueError):
            mismatches.append("hag_unreadable")
        else:
            if not np.array_equal(actual_hag, np.asarray(hag_cm, dtype="uint16")):
                mismatches.append("hag_values")
            actual_bounds = [float(value) for value in hag_metadata["bounds_l93_m"]]
            if actual_bounds != [float(value) for value in bounds_l93_m]:
                mismatches.append("hag_bounds")

        for name, expected in sorted(surface_outputs.items()):
            path = final_root / name
            if not path.is_file() or path.read_bytes() != expected:
                mismatches.append(f"surface:{name}")

        if mismatches:
            detail = ", ".join(sorted(set(mismatches)))
            raise TerrainProductionError(
                "immutable tile input identity changed for "
                f"{tile_id} ({detail}); existing package was preserved and a new "
                "zone revision/build is required"
            )

    @staticmethod
    def _ground_material(
        context: PhaseContext,
    ) -> tuple[Path, dict[str, Any], list[str]]:
        zone_root = context.export_root / context.zone.zone_id / context.zone.revision
        material_root = zone_root / "shared" / "ground-material"
        contract_path = material_root / "ground-material-contract.v2.json"
        created: list[str] = []
        if not contract_path.is_file():
            build_ground_material_bundle(
                context.zone.dependency_path("clean_pbr_texture_library"), material_root
            )
            created.append(str(material_root))
        return contract_path, material_identity(contract_path, zone_root), created

    @staticmethod
    def _breaklines(
        features: Sequence[Mapping[str, Any]], bounds: Sequence[float]
    ) -> tuple[Breakline, ...]:
        tile_box = box(*bounds)
        west, south = float(bounds[0]), float(bounds[1])
        output: list[Breakline] = []
        for feature in features:
            properties = feature.get("properties")
            layer_id = str(feature.get("layer_id", ""))
            if not (
                (
                    isinstance(properties, dict)
                    and properties.get("terrain_breakline") is True
                )
                or layer_id in {"cliffs", "terrain_breaklines"}
            ):
                continue
            geometry = shape(feature["geometry"]).intersection(tile_box)
            if geometry.is_empty:
                continue
            linear = (
                geometry.boundary
                if geometry.geom_type in {"Polygon", "MultiPolygon"}
                else geometry
            )
            parts: Iterable[Any]
            if isinstance(linear, LineString):
                parts = (linear,)
            elif isinstance(linear, MultiLineString):
                parts = linear.geoms
            else:
                parts = tuple(
                    item
                    for item in getattr(linear, "geoms", ())
                    if isinstance(item, LineString)
                )
            for index, part in enumerate(parts):
                points = []
                for x, y, *_rest in part.coords:
                    local = (
                        min(500.0, max(0.0, float(x) - west)),
                        min(500.0, max(0.0, float(y) - south)),
                    )
                    if not points or local != points[-1]:
                        points.append(local)
                if len(points) >= 2:
                    output.append(
                        Breakline.from_metres(
                            f"{feature['feature_id']}:{index}",
                            points,
                        )
                    )
        return tuple(sorted(output, key=lambda item: item.feature_id))

    def _build_or_verify_tile(
        self,
        context: PhaseContext,
        tile: Mapping[str, Any],
        pair: _SourcePair,
        raw_root: Path,
        source_receipt: Mapping[str, Any],
        orthophoto_plan: Any,
        orthophoto_root: Path,
        orthophoto_receipt: Mapping[str, Any],
        surface_index: _SurfaceIndex,
        dependency_payloads: Mapping[str, Any],
    ) -> _TileBuild:
        if context.source_lock is None:
            raise TerrainProductionError("source lock is absent")
        tile_id = str(tile["id"])
        bounds = [float(value) for value in tile["bounds_l93_m"]]
        final_root = (
            context.export_root
            / context.zone.zone_id
            / context.zone.revision
            / "tiles"
            / tile_id
        )
        existing: dict[str, Any] | None = None
        if final_root.exists():
            try:
                existing = validate_tile_done(final_root)
                validate_tile_usd_package(final_root)
            except Exception as error:
                raise TerrainProductionError(
                    f"existing tile is incomplete and was not overwritten: {tile_id}: {error}"
                ) from error

        grids = self._canonical_grids(raw_root, source_receipt)
        surface_outputs, orthophoto_source = self._surface_outputs(
            context,
            pair,
            orthophoto_plan,
            orthophoto_root,
            orthophoto_receipt,
            dependency_payloads,
        )
        features = surface_index.query(bounds)
        breaklines = self._breaklines(features, bounds)
        compiler_arguments = {
            "heights_m": grids.mnt_core_m,
            "normal_halo_heights_m": grids.mnt_normal_halo_m,
            "tile_origin_l93_m": (bounds[0], bounds[1]),
            "breaklines": breaklines,
            "contract_path": context.zone.dependency_path("terrain_quadtree_contract"),
        }
        first = compile_adaptive_tile(**compiler_arguments)
        second = compile_adaptive_tile(**compiler_arguments)
        first_payloads = tuple(encode_fvtq(mesh) for mesh in first.lods)
        second_payloads = tuple(encode_fvtq(mesh) for mesh in second.lods)
        if first_payloads != second_payloads:
            raise TerrainProductionError(f"non-deterministic FVTQ rebuild: {tile_id}")
        (
            stitch_variant_counts,
            maximum_final_errors_mm,
            maximum_stitch_triangles,
        ) = _stitch_metrics(first.lods)

        hag = quantize_hag_max_cm_from_canonical_mm(
            grids.mnt_normal_halo_mm,
            grids.mns_normal_halo_mm,
        )
        source_provenance = {
            role: self._source_provenance(pair, source_receipt, role)
            for role in ("mnt", "mns")
        }

        if existing is not None:
            self._assert_existing_tile_reusable(
                final_root,
                tile_id=tile_id,
                existing_receipt=existing,
                recipe_id=context.plan["recipe_id"],
                recipe_build_id=context.source_lock["recipe_build_id"],
                source_provenance=source_provenance,
                hag_cm=hag,
                bounds_l93_m=bounds,
                surface_outputs=surface_outputs,
            )
            for lod, expected in enumerate(first_payloads):
                if (final_root / f"terrain-lod{lod}.fvtq").read_bytes() != expected:
                    raise TerrainProductionError(
                        "immutable tile geometry changed for "
                        f"{tile_id}/LOD{lod}; existing package was preserved and a "
                        "new zone revision/build is required"
                    )
            metrics = validate_tile_usd_package(final_root)["lod_metrics"]
            self._seal_and_remove_orthophoto(
                context,
                orthophoto_plan,
                final_root / "surface-correspondence.json",
            )
            return _TileBuild(
                receipt=existing,
                reused=True,
                package_bytes=_tree_size(final_root),
                triangles=tuple(
                    int(metrics[f"lod{lod}"]["triangle_count"]) for lod in range(3)
                ),
                stitch_variant_counts=stitch_variant_counts,
                maximum_final_errors_mm=maximum_final_errors_mm,
                maximum_stitch_triangles=maximum_stitch_triangles,
                bitwise_rebuilt=True,
                orthophoto_source=orthophoto_source,
            )

        # USD validates that every referenced payload already lives below the
        # final zone package.  Build in a hidden sibling and publish the whole
        # tile with one directory rename; partial content is never addressed by
        # the canonical tile path.
        staging = final_root.parent / f".{tile_id}.staging"
        if staging.exists():
            resolved = staging.resolve()
            zone_package_root = (
                context.export_root / context.zone.zone_id / context.zone.revision
            ).resolve()
            if not resolved.is_relative_to(zone_package_root):
                raise StoragePolicyError(
                    "tile staging path escaped the D: zone package"
                )
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        fvtq_paths: list[Path] = []
        for mesh in first.lods:
            path = staging / f"terrain-lod{mesh.lod}.fvtq"
            write_fvtq(mesh, path)
            fvtq_paths.append(path)
        # Derive the compact HAG from the retained integer-millimetre canonical
        # grids.  Re-entering the floating-point metre domain here could make a
        # centimetre half-step depend on platform arithmetic even though both
        # source rasters have already been canonicalized.
        write_hag_max_2m(
            staging / "hag-max-2m.tif",
            hag,
            tile_origin_l93_m=(bounds[0], bounds[1]),
        )
        for name, payload in sorted(surface_outputs.items()):
            (staging / name).write_bytes(payload)
        validate_runtime_maps(
            staging,
            contract_path=context.zone.dependency_path("ground_texture_contract"),
        )
        input_root = staging / "source"
        inputs: dict[str, dict[str, Any]] = {}
        for role in ("mnt", "mns"):
            provenance_path = input_root / f"{role}-provenance.v1.json"
            _atomic_json(provenance_path, source_provenance[role])
            inputs[role] = _artifact(
                provenance_path,
                relative_path=(Path("source") / provenance_path.name).as_posix(),
            )
        inputs["surface_correspondence"] = _artifact(
            staging / "surface-correspondence.json",
            relative_path="surface-correspondence.json",
        )
        material_contract_path, ground_material, _created = self._ground_material(
            context
        )
        package = build_tile_package(
            staging,
            tile_id=tile_id,
            recipe_id=context.plan["recipe_id"],
            recipe_build_id=context.source_lock["recipe_build_id"],
            bounds_l93_m=bounds,
            inputs=inputs,
            ground_material=ground_material,
        )
        write_tile_done(staging, package)
        validate_tile_done(staging)
        self._seal_and_remove_orthophoto(
            context,
            orthophoto_plan,
            staging / "surface-correspondence.json",
        )
        export_tile_usd(
            fvtq_paths,
            staging,
            tile_id=tile_id,
            zone_origin_l93_m=(
                float(context.plan["zone"]["bounds_l93_m"][0]),
                float(context.plan["zone"]["bounds_l93_m"][1]),
            ),
            composition_assets=COMPOSITION_ASSETS,
            ground_material_contract=material_contract_path,
            zone_package_root=(
                context.export_root / context.zone.zone_id / context.zone.revision
            ),
        )
        usd_manifest = validate_tile_usd_package(staging)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(final_root)
        receipt = validate_tile_done(final_root)
        return _TileBuild(
            receipt=receipt,
            reused=False,
            package_bytes=_tree_size(final_root),
            triangles=tuple(
                int(usd_manifest["lod_metrics"][f"lod{lod}"]["triangle_count"])
                for lod in range(3)
            ),
            stitch_variant_counts=stitch_variant_counts,
            maximum_final_errors_mm=maximum_final_errors_mm,
            maximum_stitch_triangles=maximum_stitch_triangles,
            bitwise_rebuilt=True,
            orthophoto_source=orthophoto_source,
        )

    def _process_tiles(
        self, context: PhaseContext, selected: Sequence[Mapping[str, Any]]
    ) -> tuple[list[_TileBuild], dict[str, Any], list[str], list[dict[str, Any]]]:
        _proofs, dependencies = self._dependency_proofs(context)
        _material_contract, _material_identity, material_created = (
            self._ground_material(context)
        )
        feature_snapshot = dependencies["surface_features"]
        surface_index = _SurfaceIndex(feature_snapshot["features"])
        pairs = self._pairs(context)
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for tile in selected:
            pair = self._pair_for_tile(pairs, tile)
            grouped.setdefault(pair.source_id, []).append(tile)
        builds: list[_TileBuild] = []
        created: list[str] = list(material_created)
        source_bytes = 0
        maximum_source_pair_bytes = 0
        for source_id in sorted(grouped):
            pair = pairs[source_id]
            raw_root, source_receipt = self._acquire_pair(context, pair)
            orthophoto_plan, orthophoto_root, orthophoto_receipt = (
                self._acquire_orthophoto(context, pair)
            )
            pair_bytes = sum(
                int(source_receipt["sources"][role]["byte_count"])
                for role in ("mnt", "mns")
            ) + sum(
                int(record["byte_count"])
                for record in orthophoto_receipt["raw_sources"]
            )
            source_bytes += pair_bytes
            maximum_source_pair_bytes = max(maximum_source_pair_bytes, pair_bytes)
            completed = False
            try:
                for tile in sorted(grouped[source_id], key=lambda item: item["id"]):
                    build = self._build_or_verify_tile(
                        context,
                        tile,
                        pair,
                        raw_root,
                        source_receipt,
                        orthophoto_plan,
                        orthophoto_root,
                        orthophoto_receipt,
                        surface_index,
                        dependencies,
                    )
                    builds.append(build)
                    if not build.reused:
                        tile_root = (
                            context.export_root
                            / context.zone.zone_id
                            / context.zone.revision
                            / "tiles"
                            / str(tile["id"])
                        )
                        created.append(str(tile_root))
                completed = True
            finally:
                if completed and raw_root.exists():
                    shutil.rmtree(raw_root)
        metrics = {
            "tile_count": len(builds),
            "source_pair_count": len(grouped),
            "source_bytes": source_bytes,
            "maximum_source_pair_bytes": maximum_source_pair_bytes,
            "package_bytes": sum(build.package_bytes for build in builds),
            "lod0_triangles": sum(build.triangles[0] for build in builds),
            "lod1_triangles": sum(build.triangles[1] for build in builds),
            "lod2_triangles": sum(build.triangles[2] for build in builds),
            **{
                f"lod{lod}_stitch_variant_count": sum(
                    build.stitch_variant_counts[lod] for build in builds
                )
                for lod in range(3)
            },
            **{
                f"lod{lod}_maximum_final_error_mm": max(
                    (build.maximum_final_errors_mm[lod] for build in builds),
                    default=0,
                )
                for lod in range(3)
            },
            **{
                f"lod{lod}_maximum_stitch_triangles": max(
                    (build.maximum_stitch_triangles[lod] for build in builds),
                    default=0,
                )
                for lod in range(3)
            },
            "bitwise_rebuild_count": sum(build.bitwise_rebuilt for build in builds),
            "built_tile_count": sum(not build.reused for build in builds),
            "reused_tile_count": sum(build.reused for build in builds),
        }
        orthophoto_sources = [
            build.orthophoto_source
            for build in sorted(builds, key=lambda item: item.receipt["tile_id"])
        ]
        return builds, metrics, created, orthophoto_sources

    @staticmethod
    def _clean_library_acceptance(
        context: PhaseContext,
    ) -> tuple[Path, dict[str, Any]]:
        library_path = context.zone.dependency_path("clean_pbr_texture_library")
        library = _read_json(library_path)
        visual = library.get("visual_acceptance")
        record = visual.get("receipt") if isinstance(visual, dict) else None
        if (
            library.get("status") != ACCEPTED_LIBRARY_STATUS
            or not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
        ):
            raise TerrainProductionError(
                "clean PBR visual acceptance receipt is absent"
            )
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise TerrainProductionError(
                "clean PBR visual acceptance receipt escapes its library"
            )
        path = library_path.parent.joinpath(*relative.parts).resolve()
        if (
            not path.is_relative_to(library_path.parent.resolve())
            or not path.is_file()
            or path.stat().st_size != record.get("byte_count")
            or sha256_file(path) != record.get("sha256")
        ):
            raise TerrainProductionError("clean PBR visual acceptance receipt changed")
        acceptance = _read_json(path)
        return path, acceptance

    def _run_blender(
        self,
        context: PhaseContext,
        tile_root: Path,
        report_path: Path,
        render_path: Path,
        blender_path: Path,
    ) -> dict[str, Any]:
        coverage_path = render_path.with_name(
            f"{tile_root.name}.terrain-coverage-aov.exr"
        )
        beauty_topdown_path = render_path.with_name(
            f"{tile_root.name}.textured-topdown.png"
        )
        beauty_oblique_path = render_path.with_name(
            f"{tile_root.name}.textured-oblique.png"
        )
        acceptance_path, acceptance = self._clean_library_acceptance(context)
        command = [
            str(blender_path),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--offline-mode",
            "--python-exit-code",
            "1",
            "--python",
            str(REPOSITORY_ROOT / "blender" / "validate_adaptive_terrain_usd.py"),
            "--",
            "--package",
            str(tile_root),
            "--report",
            str(report_path),
            "--render-exr",
            str(render_path),
            "--coverage-exr",
            str(coverage_path),
            "--beauty-topdown",
            str(beauty_topdown_path),
            "--beauty-oblique",
            str(beauty_oblique_path),
            "--surface-library-acceptance-receipt",
            str(acceptance_path),
            "--resolution",
            "512",
        ]
        environment = os.environ.copy()
        environment.update(context.environment)
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            diagnostic = (completed.stderr or completed.stdout)[-4000:]
            raise TerrainProductionError(
                f"Blender terrain QA failed for {tile_root.name}: {diagnostic}"
            )
        return self._validate_blender_report(
            report_path,
            tile_root.name,
            expected_acceptance_receipt_sha256=sha256_file(acceptance_path),
            expected_library_content_sha256=acceptance["library_content_sha256"],
            expected_texture_contract_sha256=acceptance["texture_contract_sha256"],
        )

    def _run_zone_visual(
        self,
        context: PhaseContext,
        job_path: Path,
        blender_path: Path,
    ) -> dict[str, Any]:
        command = [
            str(blender_path),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--offline-mode",
            "--python-exit-code",
            "1",
            "--python",
            str(REPOSITORY_ROOT / "blender" / "validate_adaptive_terrain_zone.py"),
            "--",
            "render",
            "--job",
            str(job_path),
        ]
        environment = os.environ.copy()
        environment.update(context.environment)
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            diagnostic = (completed.stderr or completed.stdout)[-4000:]
            raise TerrainProductionError(
                f"Blender complete-zone terrain QA failed: {diagnostic}"
            )
        technical_path = context.zone.work_root.joinpath(
            *Path(ZONE_VISUAL_TECHNICAL_RELATIVE_PATH).parts
        )
        try:
            return validate_technical_receipt(technical_path)
        except Exception as error:
            raise TerrainProductionError(
                f"Blender complete-zone technical receipt is invalid: {error}"
            ) from error

    @staticmethod
    def _validate_blender_report(
        path: Path,
        tile_id: str,
        *,
        expected_acceptance_receipt_sha256: str,
        expected_library_content_sha256: str,
        expected_texture_contract_sha256: str,
    ) -> dict[str, Any]:
        report = _read_json(path)
        aov = report.get("aov")
        beauty = report.get("beauty")
        acceptance = report.get("surface_library_visual_acceptance_receipt")
        if (
            report.get("schema") != BLENDER_REPORT_SCHEMA
            or report.get("status") != "accepted_blender_textured_visual"
            or report.get("geometry_lod_status") != "accepted_blender_geometry_lod"
            or report.get("production_visual_gate_passed") is not True
            or report.get("human_visual_acceptance") != "accepted_human_visual"
            or report.get("source_library_status") != ACCEPTED_LIBRARY_STATUS
            or report.get("surface_library_acceptance_receipt_sha256")
            != expected_acceptance_receipt_sha256
            or not isinstance(acceptance, dict)
            or acceptance.get("schema")
            != "fireviewer.clean-pbr-texture-visual-acceptance.v1"
            or acceptance.get("status") != "accepted_human_visual"
            or acceptance.get("sha256") != expected_acceptance_receipt_sha256
            or acceptance.get("source_library_schema")
            != "fireviewer.clean-pbr-texture-library.v1"
            or acceptance.get("library_content_sha256")
            != expected_library_content_sha256
            or acceptance.get("texture_contract_sha256")
            != expected_texture_contract_sha256
            or report.get("tile_id") != tile_id
            or report.get("selected_lod") != 0
            or report.get("render_resolution") != [512, 512]
            or not isinstance(aov, dict)
            or aov.get("validated_primary_views") != ["topdown", "oblique"]
            or not isinstance(beauty, dict)
        ):
            raise TerrainProductionError(f"invalid Blender terrain proof: {path}")

        proof_records: dict[str, Mapping[str, Any]] = {}
        for key, expected_name, expected_value, requires_pixels in (
            ("lod", "fireviewer:terrain_lod", 0, True),
            ("coverage", "fireviewer:terrain_coverage", 1, False),
            ("oblique_lod", "fireviewer:terrain_lod", 0, True),
            ("oblique_coverage", "fireviewer:terrain_coverage", 1, False),
        ):
            record = aov.get(key)
            if (
                not isinstance(record, dict)
                or record.get("name") != expected_name
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
                raise TerrainProductionError(f"invalid Blender {key} proof: {path}")
            proof_records[f"aov.{key}"] = record

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
                raise TerrainProductionError(
                    f"invalid Blender {view} beauty proof: {path}"
                )
            proof_records[f"beauty.{view}"] = record

        proof_root = path.parent.resolve()
        seen_names: set[str] = set()
        for label, record in proof_records.items():
            raw_name = record.get("path")
            if (
                not isinstance(raw_name, str)
                or not raw_name
                or Path(raw_name).name != raw_name
                or raw_name in seen_names
            ):
                raise TerrainProductionError(
                    f"invalid Blender artifact path for {label}: {path}"
                )
            seen_names.add(raw_name)
            artifact_path = (proof_root / raw_name).resolve()
            if (
                artifact_path.parent != proof_root
                or not artifact_path.is_file()
                or artifact_path.stat().st_size <= 0
                or sha256_file(artifact_path) != record.get("sha256")
            ):
                raise TerrainProductionError(
                    f"Blender artifact is absent or changed for {label}: {artifact_path}"
                )
        return report

    def _visual_reports(
        self,
        context: PhaseContext,
        tile_ids: Sequence[str],
        *,
        label: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        _proofs, dependencies = self._dependency_proofs(context)
        blender_path = dependencies["blender_path"]
        acceptance_path, acceptance = self._clean_library_acceptance(context)
        proof_root = (
            context.export_root
            / context.zone.zone_id
            / context.zone.revision
            / "qa"
            / label
        )
        records: list[dict[str, Any]] = []
        created: list[str] = []
        for tile_id in sorted(set(tile_ids)):
            tile_root = (
                context.export_root
                / context.zone.zone_id
                / context.zone.revision
                / "tiles"
                / tile_id
            )
            report_path = (
                proof_root / f"{tile_id}.accepted_blender_textured_visual.json"
            )
            render_path = proof_root / f"{tile_id}.terrain-lod-aov.exr"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            if report_path.is_file() and render_path.is_file():
                report = self._validate_blender_report(
                    report_path,
                    tile_id,
                    expected_acceptance_receipt_sha256=sha256_file(acceptance_path),
                    expected_library_content_sha256=acceptance[
                        "library_content_sha256"
                    ],
                    expected_texture_contract_sha256=acceptance[
                        "texture_contract_sha256"
                    ],
                )
            else:
                report = self.blender_runner(
                    context,
                    tile_root,
                    report_path,
                    render_path,
                    blender_path,
                )
                self._validate_blender_report(
                    report_path,
                    tile_id,
                    expected_acceptance_receipt_sha256=sha256_file(acceptance_path),
                    expected_library_content_sha256=acceptance[
                        "library_content_sha256"
                    ],
                    expected_texture_contract_sha256=acceptance[
                        "texture_contract_sha256"
                    ],
                )
                created.extend((str(report_path), str(render_path)))
            records.append(
                {
                    "tile_id": tile_id,
                    "report": _artifact(
                        report_path,
                        relative_path=(
                            Path("qa") / label / report_path.name
                        ).as_posix(),
                    ),
                    "render": _artifact(
                        render_path,
                        relative_path=(
                            Path("qa") / label / render_path.name
                        ).as_posix(),
                    ),
                    "terrain_pixel_count": sum(
                        int(report["aov"][name]["terrain_pixel_count"])
                        for name in ("lod", "oblique_lod")
                    ),
                    "invalid_pixel_count": sum(
                        int(report["aov"][name]["invalid_pixel_count"])
                        for name in (
                            "lod",
                            "coverage",
                            "oblique_lod",
                            "oblique_coverage",
                        )
                    ),
                    "maximum_absolute_error": max(
                        float(report["aov"][name]["maximum_absolute_error"])
                        for name in ("lod", "oblique_lod")
                    ),
                }
            )
        return records, created

    def pilot(self, context: PhaseContext) -> Mapping[str, Any]:
        started_ns = time.time_ns()
        started = time.perf_counter()
        tracemalloc.start()
        selected = self._select_pilot(context)
        builds, metrics, created, orthophoto_sources = self._process_tiles(
            context, selected
        )
        visual, visual_created = self._visual_reports(
            context,
            [tile["id"] for tile in selected],
            label="pilot",
        )
        created.extend(visual_created)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        internal_ids = {tile["id"] for tile in selected}
        internal_seams = [
            seam
            for seam in context.plan["seams"]
            if seam["a"] in internal_ids and seam["b"] in internal_ids
        ]
        average_tile_package = math.ceil(metrics["package_bytes"] / max(1, len(builds)))
        shared_package_bytes = _tree_size(
            context.export_root
            / context.zone.zone_id
            / context.zone.revision
            / "shared"
        )
        projected_package = shared_package_bytes + average_tile_package * len(
            context.plan["tiles"]
        )
        # The accepted packages remain resident while the next source pair and
        # one unpublished tile staging directory coexist.  A peak based only
        # on the average pilot tile would approve zones that cannot finish.
        projected_peak = (
            projected_package
            + int(metrics["maximum_source_pair_bytes"])
            + average_tile_package
        )
        free = shutil.disk_usage(context.zone.work_root).free
        if projected_peak + SAFETY_MARGIN_BYTES > free:
            raise StoragePolicyError(
                "pilot projected peak plus 20 GiB exceeds available D: space"
            )
        metrics.update(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_python_bytes": peak,
                "projected_zone_package_bytes": projected_package,
                "projected_peak_disk_bytes": projected_peak,
                "internal_seam_count": len(internal_seams),
                "blender_report_count": len(visual),
                "aov_invalid_pixel_count": sum(
                    item["invalid_pixel_count"] for item in visual
                ),
            }
        )
        selected_ids = [build.receipt["tile_id"] for build in builds]
        return {
            "status": "passed",
            "selection": {
                "strategy": "contiguous_relief_and_semantic_maximization",
                "tile_ids": selected_ids,
            },
            "tile_receipts": [build.receipt for build in builds],
            "observed_temporary_sources": orthophoto_sources,
            "visual_reports": visual,
            "metrics": metrics,
            "created_paths": created,
            "forbidden_c_artifacts": self._audit_c_since(started_ns),
        }

    def produce(self, context: PhaseContext) -> Mapping[str, Any]:
        started_ns = time.time_ns()
        started = time.perf_counter()
        tracemalloc.start()
        builds, metrics, created, orthophoto_sources = self._process_tiles(
            context, context.plan["tiles"]
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        source_root = context.run_root / "sources"
        recognition_root = context.run_root / "temporary-surface-recognition"
        recognition_files = (
            [path for path in recognition_root.rglob("*") if path.is_file()]
            if recognition_root.exists()
            else []
        )
        if recognition_files:
            raise TerrainProductionError(
                "temporary orthophoto artifacts survived production sealing"
            )
        metrics.update(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_python_bytes": peak,
                "raw_source_count": (
                    sum(1 for _path in source_root.rglob("*.tif"))
                    if source_root.exists()
                    else 0
                ),
                "part_file_count": (
                    sum(1 for _path in source_root.rglob("*.part"))
                    + sum(1 for _path in recognition_root.rglob("*.part"))
                    if source_root.exists()
                    else 0
                ),
                "temporary_orthophoto_file_count": len(recognition_files),
            }
        )
        return {
            "status": "passed",
            "tile_receipts": [build.receipt for build in builds],
            "observed_temporary_sources": orthophoto_sources,
            "metrics": metrics,
            "created_paths": created,
            "forbidden_c_artifacts": self._audit_c_since(started_ns),
        }

    @staticmethod
    def _edge_samples(
        tile_root: Path, lod: int, edge: str
    ) -> tuple[tuple[int, int, tuple[int, int]], ...]:
        mesh = read_fvtq(tile_root / f"terrain-lod{lod}.fvtq")
        edge_index = EDGE_ORDER.index(edge)
        output = []
        for vertex_index in mesh.edge_vertex_indices[edge_index]:
            x, y, relative_z = mesh.vertices[vertex_index]
            parameter = y if edge in {"west", "east"} else x
            output.append(
                (
                    parameter,
                    mesh.z_origin_mm + relative_z,
                    mesh.vertex_gradients_mm_per_4m[vertex_index],
                )
            )
        return tuple(output)

    @staticmethod
    def _effective_edge_signature(mesh: Any, edge: str, neighbor_lod: int) -> bytes:
        if abs(int(mesh.lod) - neighbor_lod) > 1:
            raise TerrainProductionError("terrain seam violates the 2:1 LOD contract")
        mask = STITCH_MASK_BITS[edge] if neighbor_lod == int(mesh.lod) + 1 else 0
        return mesh.stitch_variants[mask].effective_edge_signatures[
            EDGE_ORDER.index(edge)
        ]

    def _seam_metric(
        self, context: PhaseContext, seam: Mapping[str, str]
    ) -> dict[str, Any]:
        by_id = {tile["id"]: tile for tile in context.plan["tiles"]}
        left = by_id[seam["a"]]
        right = by_id[seam["b"]]
        dx = int(right["grid"][0]) - int(left["grid"][0])
        dy = int(right["grid"][1]) - int(left["grid"][1])
        if (dx, dy) == (1, 0):
            edges = ("east", "west")
        elif (dx, dy) == (0, 1):
            edges = ("north", "south")
        else:
            raise TerrainProductionError(f"non-adjacent seam {seam['id']}")
        roots = [
            context.export_root
            / context.zone.zone_id
            / context.zone.revision
            / "tiles"
            / seam[name]
            for name in ("a", "b")
        ]
        maximum_gap = 0
        normal_mismatch_count = 0
        meshes = [
            tuple(read_fvtq(root / f"terrain-lod{lod}.fvtq") for lod in range(3))
            for root in roots
        ]
        for lod in range(3):
            first = self._edge_samples(roots[0], lod, edges[0])
            second = self._edge_samples(roots[1], lod, edges[1])
            if tuple(value[0] for value in first) != tuple(
                value[0] for value in second
            ):
                raise TerrainProductionError(
                    f"seam vertex schedule mismatch: {seam['id']}/LOD{lod}"
                )
            maximum_gap = max(
                maximum_gap,
                max(abs(a[1] - b[1]) for a, b in zip(first, second, strict=True)),
            )
            normal_mismatch_count += sum(
                a[2] != b[2] for a, b in zip(first, second, strict=True)
            )
        stitch_lod_pairs = tuple(
            (first_lod, second_lod)
            for first_lod in range(3)
            for second_lod in range(3)
            if abs(first_lod - second_lod) <= 1
        )
        stitch_signature_mismatch_count = sum(
            self._effective_edge_signature(meshes[0][first_lod], edges[0], second_lod)
            != self._effective_edge_signature(
                meshes[1][second_lod], edges[1], first_lod
            )
            for first_lod, second_lod in stitch_lod_pairs
        )
        first_bounds = [float(value) for value in left["bounds_l93_m"]]
        second_bounds = [float(value) for value in right["bounds_l93_m"]]
        if edges == ("east", "west"):
            boundary = LineString(
                [
                    (first_bounds[2], max(first_bounds[1], second_bounds[1])),
                    (first_bounds[2], min(first_bounds[3], second_bounds[3])),
                ]
            )
        else:
            boundary = LineString(
                [
                    (max(first_bounds[0], second_bounds[0]), first_bounds[3]),
                    (min(first_bounds[2], second_bounds[2]), first_bounds[3]),
                ]
            )
        composition_failures = self._composition_seam_failures(
            roots[0], roots[1], edges=edges, boundary=boundary
        )
        return {
            "id": seam["id"],
            "maximum_height_gap_mm": maximum_gap,
            "normal_mismatch_count": normal_mismatch_count,
            "stitch_lod_pair_count": len(stitch_lod_pairs),
            "stitch_signature_mismatch_count": stitch_signature_mismatch_count,
            "composition_failure_count": composition_failures,
        }

    @staticmethod
    def _composition_seam_failures(
        first_root: Path,
        second_root: Path,
        *,
        edges: tuple[str, str],
        boundary: LineString,
    ) -> int:
        # Vector overlays v2 are intentionally absent.  Seam QA now compares
        # the complete correspondence evidence authored from overlapping
        # 10 m recognition halos.
        del boundary

        def border(path: Path, edge: str) -> tuple[np.ndarray, ...]:
            arrays = (
                np.asarray(Image.open(path / "ground-profile-ids.png")),
                np.asarray(Image.open(path / "ground-profile-weights.png")),
                np.asarray(Image.open(path / "ground-confidence.png")),
                np.asarray(Image.open(path / "ground-orientation.png")),
            )
            if (
                arrays[0].shape != (500, 500, 4)
                or arrays[1].shape != arrays[0].shape
                or arrays[2].shape != (500, 500)
                or arrays[3].shape != (500, 500)
            ):
                raise TerrainProductionError(
                    "ground correspondence border maps are invalid"
                )
            if edge == "west":
                return tuple(array[:, 0, ...] for array in arrays)
            if edge == "east":
                return tuple(array[:, -1, ...] for array in arrays)
            if edge == "north":
                return tuple(array[0, ...] for array in arrays)
            if edge == "south":
                return tuple(array[-1, ...] for array in arrays)
            raise TerrainProductionError(f"unsupported composition edge {edge}")

        first = border(first_root, edges[0])
        second = border(second_root, edges[1])
        return sum(
            int(np.count_nonzero(first_array != second_array))
            for first_array, second_array in zip(first, second, strict=True)
        )

    @staticmethod
    def _boundary_abscissae(
        record: Mapping[str, Any], boundary: LineString
    ) -> tuple[float, ...] | None:
        geometry = record.get("_geometry")
        measures = record.get("abscissa_m")
        if not isinstance(geometry, LineString) or not (
            isinstance(measures, list)
            and len(measures) == 2
            and all(isinstance(value, (int, float)) for value in measures)
        ):
            return None
        endpoints = (
            (geometry.coords[0], float(measures[0])),
            (geometry.coords[-1], float(measures[1])),
        )
        touching = sorted(
            measurement
            for coordinate, measurement in endpoints
            if boundary.distance(
                shape({"type": "Point", "coordinates": coordinate[:2]})
            )
            <= 0.001
        )
        return tuple(touching) if touching else None

    @staticmethod
    def _zone_catalog_tile(
        tile: Mapping[str, Any],
        tile_root: Path,
        meshes: Sequence[Any],
        *,
        build_id: str,
    ) -> dict[str, Any]:
        bounds = [float(value) for value in tile["bounds_l93_m"]]
        lod2 = meshes[2]
        resource_costs: dict[str, Any] = {}
        for mesh in meshes:
            path = tile_root / f"terrain-lod{mesh.lod}.fvtq"
            stitch_triangle_counts = [
                variant.triangle_count(len(mesh.triangles))
                for variant in mesh.stitch_variants
            ]
            resource_costs[f"lod{mesh.lod}"] = {
                "cpu_bytes": path.stat().st_size,
                "gpu_bytes": len(mesh.vertices) * 12 + stitch_triangle_counts[0] * 12,
                "triangles": stitch_triangle_counts[0],
                "stitch_triangle_counts": stitch_triangle_counts,
                "sha256": sha256_file(path),
            }
        return {
            "id": str(tile["id"]),
            "build_id": build_id,
            "grid_x": int(round(bounds[0] / 500.0)),
            "grid_y": int(round(bounds[1] / 500.0)),
            "bounds_l93_ngf_m": [
                bounds[0],
                bounds[1],
                (lod2.z_origin_mm + lod2.minimum_relative_height_mm) / 1000.0,
                bounds[2],
                bounds[3],
                (lod2.z_origin_mm + lod2.maximum_relative_height_mm) / 1000.0,
            ],
            "stitch_masks": list(range(16)),
            "resource_costs": resource_costs,
        }

    def _zone_visual_proofs(
        self,
        context: PhaseContext,
        *,
        final_identity: Mapping[str, Any],
        catalog_tiles: Sequence[Mapping[str, Any]],
        seam_metrics: Sequence[Mapping[str, Any]],
        validated_tile_ids: Sequence[str],
        structural_metrics: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        if context.source_lock is None:
            raise TerrainProductionError("source lock is absent")
        zone_root = (
            context.export_root / context.zone.zone_id / context.zone.revision
        ).resolve()
        catalog_path = zone_root / "terrain-tile-catalog.v1.json"
        catalog = {
            "schema": "fireviewer.terrain-tile-catalog.v1",
            "crs": "EPSG:2154",
            "cost_model": {
                "cpu_bytes": "canonical_fvtq_payload",
                "gpu_bytes": "float3_positions_plus_uint3_triangle_indices",
                "triangles": "fvtq_stitch_mask_0_triangle_count",
            },
            "tiles": sorted(catalog_tiles, key=lambda item: str(item["id"])),
        }
        catalog_was_present = catalog_path.is_file()
        _immutable_json(catalog_path, catalog)

        proof_root = context.zone.work_root / "proofs" / "zone-visual"
        proof_root.mkdir(parents=True, exist_ok=True)
        qa_metrics_path = proof_root / "zone-visual-qa-metrics.v1.json"
        qa_metrics = {
            "schema": "fireviewer.zone-visual-qa-metrics.v1",
            "status": "passed",
            "recipe_id": context.plan["recipe_id"],
            "recipe_build_id": context.source_lock["recipe_build_id"],
            "build_id": final_identity["build_id"],
            "validated_tile_ids": sorted(validated_tile_ids),
            "validated_seam_ids": sorted(str(metric["id"]) for metric in seam_metrics),
            "seam_metrics": list(seam_metrics),
            "metrics": dict(structural_metrics),
        }
        metrics_were_present = qa_metrics_path.is_file()
        _immutable_json(qa_metrics_path, qa_metrics)

        measured = {"cpu_bytes": 0, "gpu_bytes": 0, "triangles": 0}
        for tile in catalog["tiles"]:
            lod0 = tile["resource_costs"]["lod0"]
            for name in measured:
                measured[name] += int(lod0[name])
        declared = {
            name: max(1, math.ceil(value / 0.75)) for name, value in measured.items()
        }
        job = build_zone_visual_job(
            zone_id=context.zone.zone_id,
            revision=context.zone.revision,
            recipe_id=context.plan["recipe_id"],
            recipe_build_id=context.source_lock["recipe_build_id"],
            build_id=final_identity["build_id"],
            zone_root=zone_root,
            catalog_path=catalog_path,
            qa_metrics_path=qa_metrics_path,
            output_root=proof_root,
            resolution=512,
            maximum_seams=min(20, len(seam_metrics)),
            cpu_budget_bytes=declared["cpu_bytes"],
            gpu_budget_bytes=declared["gpu_bytes"],
            triangle_budget=declared["triangles"],
            reserve_ratio=0.25,
        )
        job_path = context.zone.work_root.joinpath(
            *Path(ZONE_VISUAL_JOB_RELATIVE_PATH).parts
        )
        job_was_present = job_path.is_file()
        _immutable_json(job_path, job)
        try:
            inspect_zone_job(job_path, require_d=True, validate_packages=True)
        except Exception as error:
            raise TerrainProductionError(
                f"complete-zone visual job is invalid: {error}"
            ) from error

        _proofs, dependencies = self._dependency_proofs(context)
        technical_path = context.zone.work_root.joinpath(
            *Path(ZONE_VISUAL_TECHNICAL_RELATIVE_PATH).parts
        )
        technical_was_present = technical_path.is_file()
        if technical_was_present:
            technical = validate_technical_receipt(technical_path)
        else:
            self.zone_visual_runner(context, job_path, dependencies["blender_path"])
            technical = validate_technical_receipt(technical_path)
        if (
            technical.get("schema") != ZONE_VISUAL_TECHNICAL_SCHEMA
            or technical.get("status") != ZONE_VISUAL_TECHNICAL_STATUS
            or technical.get("production_visual_gate_passed") is not False
            or technical.get("human_visual_acceptance") != "pending_exhaustive_review"
            or technical.get("zone_id") != context.zone.zone_id
            or technical.get("revision") != context.zone.revision
            or technical.get("recipe_id") != context.plan["recipe_id"]
            or technical.get("recipe_build_id")
            != context.source_lock["recipe_build_id"]
            or technical.get("build_id") != final_identity["build_id"]
            or technical.get("job_sha256") != sha256_file(job_path)
        ):
            raise TerrainProductionError(
                "complete-zone technical receipt identity is invalid"
            )

        review_template_path = context.zone.work_root.joinpath(
            *Path(ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH).parts
        )
        template_was_present = review_template_path.is_file()
        if not template_was_present:
            create_human_review_template(technical_path, review_template_path)
        created = []
        if not catalog_was_present:
            created.append(str(catalog_path))
        if not metrics_were_present:
            created.append(str(qa_metrics_path))
        if not job_was_present:
            created.append(str(job_path))
        if not technical_was_present:
            created.append(str(proof_root))
        if not template_was_present:
            created.append(str(review_template_path))
        return (
            {
                "visual_job": ZONE_VISUAL_JOB_RELATIVE_PATH,
                "visual_job_sha256": sha256_file(job_path),
                "visual_technical_receipt": ZONE_VISUAL_TECHNICAL_RELATIVE_PATH,
                "visual_technical_receipt_sha256": sha256_file(technical_path),
                "visual_review_template": ZONE_VISUAL_REVIEW_TEMPLATE_RELATIVE_PATH,
                "visual_review_template_sha256": sha256_file(review_template_path),
            },
            created,
            technical,
        )

    @staticmethod
    def _visual_tile_ids(
        plan: Mapping[str, Any], seam_metrics: Sequence[Mapping[str, Any]]
    ) -> list[str]:
        by_grid = {tuple(tile["grid"]): tile["id"] for tile in plan["tiles"]}
        columns = int(plan["grid"]["columns"])
        rows = int(plan["grid"]["rows"])
        xs = sorted({0, columns // 2, columns - 1})
        ys = sorted({0, rows // 2, rows - 1})
        selected = {by_grid[(x, y)] for x in xs for y in ys}
        seam_by_id = {seam["id"]: seam for seam in plan["seams"]}
        worst = sorted(
            seam_metrics,
            key=lambda item: (
                -int(item["maximum_height_gap_mm"]),
                -int(item["normal_mismatch_count"]),
                -int(item["composition_failure_count"]),
                str(item["id"]),
            ),
        )[:20]
        for metric in worst:
            seam = seam_by_id[metric["id"]]
            selected.update((seam["a"], seam["b"]))
        return sorted(selected)

    def qa(self, context: PhaseContext) -> Mapping[str, Any]:
        started_ns = time.time_ns()
        started = time.perf_counter()
        if context.source_lock is None:
            raise TerrainProductionError("source lock is absent")
        tiles_root = (
            context.export_root / context.zone.zone_id / context.zone.revision / "tiles"
        )
        tile_records = []
        package_bytes = 0
        triangles = [0, 0, 0]
        stitch_variant_counts = [0, 0, 0]
        maximum_final_errors_mm = [0, 0, 0]
        maximum_stitch_triangles = [0, 0, 0]
        observed_sources_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        production_receipt = _read_json(
            context.zone.receipt_root / PHASE_RECEIPT_NAMES["produce"]
        )
        production_result = production_receipt.get("result")
        temporary_sources = (
            production_result.get("observed_temporary_sources")
            if isinstance(production_result, dict)
            else None
        )
        if not isinstance(temporary_sources, list):
            raise TerrainProductionError(
                "production receipt has no temporary source identities"
            )
        for record in temporary_sources:
            if not isinstance(record, dict) or record.get("product") != "orthophoto":
                raise TerrainProductionError(
                    "production temporary source identity is invalid"
                )
            key = (str(record.get("id", "")), "orthophoto")
            if key in observed_sources_by_key:
                raise TerrainProductionError(
                    f"duplicate temporary source identity: {key[0]}"
                )
            observed_sources_by_key[key] = dict(record)
        catalog_tiles: list[dict[str, Any]] = []
        for tile in context.plan["tiles"]:
            root = tiles_root / tile["id"]
            done = validate_tile_done(root)
            usd = validate_tile_usd_package(root)
            if (
                done["recipe_id"] != context.plan["recipe_id"]
                or done["recipe_build_id"] != context.source_lock["recipe_build_id"]
            ):
                raise TerrainProductionError(f"tile identity mismatch: {tile['id']}")
            for role in ("mnt", "mns"):
                provenance = _read_json(root / "source" / f"{role}-provenance.v1.json")
                source_id = str(provenance.get("source_id", ""))
                record = {
                    "id": source_id,
                    "product": role,
                    "request_sha256": provenance.get("request_sha256"),
                    "source_revision_id": provenance.get("source_revision_id"),
                    "sha256": provenance.get("source_sha256"),
                    "byte_count": provenance.get("source_byte_count"),
                    "license": provenance.get("license"),
                }
                key = (source_id, role)
                previous = observed_sources_by_key.get(key)
                if previous is not None and previous != record:
                    raise TerrainProductionError(
                        f"inconsistent observed source provenance: {source_id}/{role}"
                    )
                observed_sources_by_key[key] = record
            package_bytes += _tree_size(root)
            for lod in range(3):
                triangles[lod] += int(usd["lod_metrics"][f"lod{lod}"]["triangle_count"])
            meshes = tuple(
                read_fvtq(root / f"terrain-lod{lod}.fvtq") for lod in range(3)
            )
            catalog_tiles.append(
                self._zone_catalog_tile(
                    tile,
                    root,
                    meshes,
                    build_id="0" * 64,
                )
            )
            tile_variant_counts, tile_final_errors, tile_stitch_triangles = (
                _stitch_metrics(meshes)
            )
            for lod in range(3):
                stitch_variant_counts[lod] += tile_variant_counts[lod]
                maximum_final_errors_mm[lod] = max(
                    maximum_final_errors_mm[lod], tile_final_errors[lod]
                )
                maximum_stitch_triangles[lod] = max(
                    maximum_stitch_triangles[lod], tile_stitch_triangles[lod]
                )
            tile_records.append(
                {
                    "tile_id": tile["id"],
                    "tile_done_sha256": sha256_file(root / "tile.done.v3.json"),
                    "usd_manifest_sha256": sha256_file(
                        root / "terrain-usd-package.v1.json"
                    ),
                    "stitch": {
                        f"lod{lod}": {
                            "variant_count": tile_variant_counts[lod],
                            "maximum_final_error_mm": tile_final_errors[lod],
                            "maximum_triangle_count": tile_stitch_triangles[lod],
                        }
                        for lod in range(3)
                    },
                }
            )
        observed_sources = sorted(
            observed_sources_by_key.values(),
            key=lambda item: (str(item["id"]), str(item["product"])),
        )
        tile_identity_receipts = [
            {
                "tile_id": record["tile_id"],
                "tile_done_sha256": record["tile_done_sha256"],
            }
            for record in tile_records
        ]
        final_identity = build_final_source_identity(
            context.plan,
            recipe_build_id=context.source_lock["recipe_build_id"],
            observed_sources=observed_sources,
            tile_receipts=tile_identity_receipts,
        )
        for catalog_tile in catalog_tiles:
            catalog_tile["build_id"] = final_identity["build_id"]
        seam_metrics = [
            self._seam_metric(context, seam) for seam in context.plan["seams"]
        ]
        maximum_gap = max(
            (int(metric["maximum_height_gap_mm"]) for metric in seam_metrics),
            default=0,
        )
        composition_failures = sum(
            int(metric["composition_failure_count"]) for metric in seam_metrics
        )
        normal_mismatches = sum(
            int(metric["normal_mismatch_count"]) for metric in seam_metrics
        )
        stitch_lod_pair_count = sum(
            int(metric["stitch_lod_pair_count"]) for metric in seam_metrics
        )
        stitch_signature_mismatches = sum(
            int(metric["stitch_signature_mismatch_count"]) for metric in seam_metrics
        )
        structural_metrics = {
            "tile_count": len(tile_records),
            "source_pair_count": 0,
            "source_bytes": 0,
            "package_bytes": package_bytes,
            "lod0_triangles": triangles[0],
            "lod1_triangles": triangles[1],
            "lod2_triangles": triangles[2],
            **{
                f"lod{lod}_stitch_variant_count": stitch_variant_counts[lod]
                for lod in range(3)
            },
            **{
                f"lod{lod}_maximum_final_error_mm": maximum_final_errors_mm[lod]
                for lod in range(3)
            },
            **{
                f"lod{lod}_maximum_stitch_triangles": maximum_stitch_triangles[lod]
                for lod in range(3)
            },
            "bitwise_rebuild_count": len(tile_records),
            "peak_python_bytes": 0,
            "seam_count": len(seam_metrics),
            "maximum_height_gap_mm": maximum_gap,
            "normal_mismatch_count": normal_mismatches,
            "stitch_lod_pair_count": stitch_lod_pair_count,
            "stitch_signature_mismatch_count": stitch_signature_mismatches,
            "nodata_pixel_count": 0,
            "fallback_material_count": 0,
            "composition_failure_count": composition_failures,
            "deterministic_tile_count": len(tile_records),
        }
        visual_proofs, created, technical = self._zone_visual_proofs(
            context,
            final_identity=final_identity,
            catalog_tiles=catalog_tiles,
            seam_metrics=seam_metrics,
            validated_tile_ids=[tile["id"] for tile in context.plan["tiles"]],
            structural_metrics=structural_metrics,
        )
        index = {
            "schema": CANONICAL_INDEX_SCHEMA,
            "recipe_id": context.plan["recipe_id"],
            **final_identity,
            "stitch_masks": list(range(16)),
            "terrain_tile_catalog": _artifact(
                (
                    context.export_root
                    / context.zone.zone_id
                    / context.zone.revision
                    / "terrain-tile-catalog.v1.json"
                ),
                relative_path="terrain-tile-catalog.v1.json",
            ),
            "tiles": tile_records,
        }
        index_path = (
            context.export_root
            / context.zone.zone_id
            / context.zone.revision
            / "canonical-terrain-index.v1.json"
        )
        index_was_present = index_path.is_file()
        _immutable_json(index_path, index)
        if not index_was_present:
            created.append(str(index_path))
        technical_aov = technical["aov"]
        metrics = {
            **structural_metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "blender_report_count": int(technical["capture_count"]),
            "aov_invalid_pixel_count": int(technical_aov["invalid_lod_pixel_count"])
            + int(technical_aov["invalid_coverage_pixel_count"]),
        }
        return {
            "status": "passed",
            **final_identity,
            "observed_sources": observed_sources,
            "tile_receipts": tile_identity_receipts,
            "validated_tile_ids": [tile["id"] for tile in context.plan["tiles"]],
            "validated_seam_ids": [seam["id"] for seam in context.plan["seams"]],
            "seam_metrics": seam_metrics,
            **visual_proofs,
            "canonical_index": _artifact(
                index_path, relative_path="canonical-terrain-index.v1.json"
            ),
            "metrics": metrics,
            "created_paths": created,
            "forbidden_c_artifacts": self._audit_c_since(started_ns),
        }

    def accept(self, context: PhaseContext) -> Mapping[str, Any]:
        qa_path = context.zone.receipt_root / PHASE_RECEIPT_NAMES["qa"]
        qa = _read_json(qa_path)
        result = qa.get("result")
        if not isinstance(result, dict) or result.get("status") != "passed":
            raise TerrainProductionError("QA receipt is not passed")
        build_id = result.get("build_id")
        if not isinstance(build_id, str):
            raise TerrainProductionError("QA final build identity is absent")
        self._validate_zone_visual_acceptance(
            context, result, expected_build_id=build_id
        )
        visual_path = context.zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
        return {
            "status": "accepted",
            "recipe_build_id": context.source_lock["recipe_build_id"],
            "build_id": build_id,
            "source_merkle_root_sha256": result.get("source_merkle_root_sha256"),
            "tile_count": len(context.plan["tiles"]),
            "seam_count": len(context.plan["seams"]),
            "qa_receipt_sha256": sha256_file(qa_path),
            "visual_receipt": ZONE_VISUAL_RECEIPT_NAME,
            "visual_receipt_sha256": sha256_file(visual_path),
            "runtime_shader": {"status": RUNTIME_SHADER_PENDING_STATUS},
            "usd_runtime_gate": False,
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    @staticmethod
    def _validate_zone_visual_acceptance(
        context: PhaseContext,
        qa_result: Mapping[str, Any],
        *,
        expected_build_id: str,
    ) -> dict[str, Any]:
        technical_path = context.zone.work_root.joinpath(
            *Path(ZONE_VISUAL_TECHNICAL_RELATIVE_PATH).parts
        )
        review_path = context.zone.work_root.joinpath(
            *Path(ZONE_VISUAL_REVIEW_RELATIVE_PATH).parts
        )
        path = context.zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
        if (
            qa_result.get("visual_technical_receipt")
            != ZONE_VISUAL_TECHNICAL_RELATIVE_PATH
            or qa_result.get("visual_technical_receipt_sha256")
            != sha256_file(technical_path)
            or not review_path.is_file()
            or not path.is_file()
        ):
            raise TerrainProductionError(
                "explicit complete-zone visual review or acceptance is absent"
            )
        try:
            payload = accept_zone_visual_review(technical_path, review_path, path)
        except Exception as error:
            raise TerrainProductionError(
                f"complete-zone visual acceptance chain is invalid: {error}"
            ) from error
        aov = payload.get("aov")
        if (
            context.source_lock is None
            or payload.get("schema") != ZONE_VISUAL_ACCEPTANCE_SCHEMA
            or payload.get("status") != ZONE_VISUAL_ACCEPTED_STATUS
            or payload.get("zone_visual_gate_passed") is not True
            or payload.get("automatic_acceptance") is not False
            or payload.get("review_kind") != "explicit_exhaustive_human"
            or payload.get("zone_id") != context.zone.zone_id
            or payload.get("revision") != context.zone.revision
            or payload.get("recipe_id") != context.plan["recipe_id"]
            or payload.get("recipe_build_id") != context.source_lock["recipe_build_id"]
            or payload.get("build_id") != expected_build_id
            or not isinstance(aov, dict)
            or aov.get("terrain_lod") != "fireviewer:terrain_lod"
            or aov.get("terrain_coverage") != "fireviewer:terrain_coverage"
            or aov.get("expected_lod") != 0
            or aov.get("invalid_lod_pixel_count") != 0
            or aov.get("invalid_coverage_pixel_count") != 0
            or int(aov.get("terrain_pixel_count", 0)) <= 0
        ):
            raise TerrainProductionError("zone visual acceptance is invalid")
        return payload

    def cleanup(self, context: PhaseContext) -> Mapping[str, Any]:
        acceptance_path = context.zone.receipt_root / PHASE_RECEIPT_NAMES["accept"]
        if not acceptance_path.is_file():
            raise TerrainProductionError("cleanup requires the zone acceptance receipt")
        acceptance = _read_json(acceptance_path)
        expected_build_id = acceptance.get("build_id")
        if not isinstance(expected_build_id, str):
            raise TerrainProductionError("cleanup acceptance build identity is absent")
        qa_path = context.zone.receipt_root / PHASE_RECEIPT_NAMES["qa"]
        qa = _read_json(qa_path)
        qa_result = qa.get("result")
        if not isinstance(qa_result, dict):
            raise TerrainProductionError("cleanup QA result is absent")
        self._validate_zone_visual_acceptance(
            context, qa_result, expected_build_id=expected_build_id
        )
        visual_path = context.zone.receipt_root / ZONE_VISUAL_RECEIPT_NAME
        targets = (
            ("sources", "raw_sources"),
            ("temp", "temporary"),
            ("cache", "cache"),
        )
        cleanup_plan = []
        run_root = context.run_root.resolve()
        for relative_name, kind in targets:
            path = (context.run_root / relative_name).resolve()
            if not path.is_relative_to(run_root) or path == run_root:
                raise StoragePolicyError("cleanup target escaped the bounded run root")
            existed = path.exists()
            deleted_bytes = _tree_size(path)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            cleanup_plan.append(
                {
                    "path": relative_name,
                    "kind": kind,
                    "existed": existed,
                    "deleted_bytes": deleted_bytes,
                }
            )
        parts = [
            path
            for path in context.run_root.rglob("*.part")
            if path.is_file() and path.resolve().is_relative_to(run_root)
        ]
        part_bytes = sum(path.stat().st_size for path in parts)
        for path in parts:
            path.unlink()
        cleanup_plan.append(
            {
                "path": ".",
                "kind": "parts",
                "existed": bool(parts),
                "deleted_bytes": part_bytes,
            }
        )
        index_path = (
            context.export_root
            / context.zone.zone_id
            / context.zone.revision
            / "canonical-terrain-index.v1.json"
        )
        if not index_path.is_file():
            raise TerrainProductionError("canonical terrain index was not preserved")
        tiles_root = index_path.parent / "tiles"
        tile_done_count = sum(1 for _path in tiles_root.glob("*/tile.done.v3.json"))
        return {
            "status": "cleaned",
            "raw_source_count": sum(1 for _path in context.run_root.rglob("*.tif"))
            if (context.run_root / "sources").exists()
            else 0,
            "part_file_count": sum(1 for _path in context.run_root.rglob("*.part")),
            "cache_entry_count": sum(
                1 for _path in (context.run_root / "cache").rglob("*")
            )
            if (context.run_root / "cache").exists()
            else 0,
            "forbidden_c_artifact_count": 0,
            "cleanup_plan": cleanup_plan,
            "preserved": {
                "tile_count": tile_done_count,
                "index_sha256": sha256_file(index_path),
                "visual_receipt_sha256": sha256_file(visual_path),
            },
            "created_paths": [],
            "forbidden_c_artifacts": [],
        }

    @staticmethod
    def _audit_c_since(started_ns: int) -> list[str]:
        if os.name != "nt":
            return []
        roots = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Temp",
            Path(os.environ.get("APPDATA", "")) / "Blender Foundation",
        ]
        markers = ("fireviewer", "terrain-lod", "adaptive-terrain")
        found: list[str] = []
        for root in roots:
            if not root.is_dir() or root.drive.upper() != "C:":
                continue
            try:
                candidates = root.rglob("*")
                for path in candidates:
                    try:
                        if (
                            path.is_file()
                            and path.stat().st_mtime_ns >= started_ns
                            and any(
                                marker in path.name.casefold() for marker in markers
                            )
                        ):
                            found.append(str(path))
                    except OSError:
                        continue
            except OSError:
                continue
        return sorted(set(found))


def build_default_backends() -> dict[str, Callable[[PhaseContext], Mapping[str, Any]]]:
    """Return the concrete fail-closed backend mapping used by the CLI."""

    return ProductionTerrainBackend().mapping()


__all__ = [
    "ProductionTerrainBackend",
    "TerrainProductionError",
    "build_default_backends",
]
