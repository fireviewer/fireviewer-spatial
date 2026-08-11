"""Temporary, D:-only orthophoto acquisition for build-time recognition.

The runtime terrain contract deliberately contains no orthophoto.  This module
prepares one canonical RGB source band that a future correspondence compiler
may inspect, then deletes the complete band as soon as every declared dependent
map has an externally sealed receipt.

Only a globally aligned 500 m Lambert-93 core is supported.  Acquisition is
locked to 1 m pixels and a 10 m halo; there is no 0.5 m or variable-resolution
mode.  WMS and WMTS requests are canonical, transfers are resumable through
``.part`` files, and no network access occurs while a plan is built.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import BinaryIO, Literal, Protocol
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


CRS = "EPSG:2154"
RESOLUTION_M = 1.0
HALO_M = 10.0
CORE_SIZE_M = 500.0
REQUEST_SIZE_PX = 520
SCHEMA = "fireviewer.orthophoto-build-band.v1"
RECEIPT_SCHEMA = "fireviewer.orthophoto-build-source.v1"
CLEANUP_SCHEMA = "fireviewer.orthophoto-build-cleanup.v1"
RGB_FILE_NAME = "orthophoto-rgb-1m.npy"
RECEIPT_FILE_NAME = "orthophoto-build-source.done.json"
TEMPORARY_DIRECTORY_NAME = "orthophoto-build-bands"
DISK_SAFETY_MARGIN_BYTES = 20 * 1024**3
RECEIPT_DISK_ALLOWANCE_BYTES = 64 * 1024
USER_AGENT = "FireViewer-orthophoto-build-source/1.0"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OrthophotoAcquisitionError(RuntimeError):
    """The temporary orthophoto source failed a reproducibility gate."""


class Response(Protocol):
    headers: Mapping[str, str]
    status: int | None

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> Response: ...

    def __exit__(self, *args: object) -> None: ...


Opener = Callable[..., Response]
DiskUsage = Callable[[Path], object]
ServiceKind = Literal["wms", "wmts"]


@dataclass(frozen=True)
class OrthophotoGrid:
    core_bounds_l93_m: tuple[int, int, int, int]
    request_bounds_l93_m: tuple[int, int, int, int]
    width: int = REQUEST_SIZE_PX
    height: int = REQUEST_SIZE_PX
    resolution_m: float = RESOLUTION_M
    halo_m: float = HALO_M
    crs: str = CRS


@dataclass(frozen=True)
class WmtsMatrix:
    """Explicit 1 m WMTS matrix needed to derive deterministic tile requests."""

    matrix_set: str
    matrix: str
    top_left_l93_m: tuple[float, float]
    tile_width_px: int
    tile_height_px: int
    matrix_width: int
    matrix_height: int
    resolution_m: float = RESOLUTION_M


@dataclass(frozen=True)
class ImageRequest:
    key: str
    url: str
    pixel_width: int
    pixel_height: int
    tile_column: int | None = None
    tile_row: int | None = None


@dataclass(frozen=True)
class OrthophotoBandPlan:
    band_id: str
    service_kind: ServiceKind
    provider_revision_id: str
    layer: str
    image_format: str
    grid: OrthophotoGrid
    requests: tuple[ImageRequest, ...]
    dependent_map_ids: tuple[str, ...]
    maximum_download_bytes: int
    plan_sha256: str
    wmts_matrix: WmtsMatrix | None = None

    @property
    def projected_peak_disk_bytes(self) -> int:
        canonical_bytes = self.grid.width * self.grid.height * 3
        return (
            self.maximum_download_bytes + canonical_bytes + RECEIPT_DISK_ALLOWANCE_BYTES
        )


@dataclass(frozen=True)
class DownloadReceipt:
    byte_count: int
    sha256: str
    resumed_from_byte: int


@dataclass(frozen=True)
class SealedDependentMap:
    """Opaque map receipt identity; its future schema is intentionally unknown."""

    map_id: str
    receipt_path: Path
    receipt_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OrthophotoAcquisitionError(
            f"orthophoto contract is not canonical JSON: {error}"
        ) from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe stable identifier")
    return value


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OrthophotoAcquisitionError(f"{label} must be a lowercase SHA-256")
    return value


def _require_d_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise OrthophotoAcquisitionError(f"{label} must stay on D:, got {resolved}")
    return resolved


def _require_temporary_root(path: Path) -> Path:
    resolved = _require_d_path(path, "temporary_root")
    repository_root = Path(__file__).resolve().parent.parent
    if resolved == repository_root or repository_root in resolved.parents:
        raise OrthophotoAcquisitionError(
            "temporary orthophoto data must remain outside the FireViewer Git worktree"
        )
    for candidate in (resolved, *resolved.parents):
        marker = candidate / ".git"
        if marker.is_file() or (marker.is_dir() and (marker / "HEAD").is_file()):
            raise OrthophotoAcquisitionError(
                "temporary orthophoto root must remain outside a Git worktree"
            )
    return resolved


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError as error:
        raise OrthophotoAcquisitionError(
            f"cannot inspect temporary orthophoto path: {path}"
        ) from error
    return bool(attributes & int(getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def _reject_reparse_tree(root: Path) -> None:
    if _is_reparse_point(root):
        raise OrthophotoAcquisitionError(
            "temporary orthophoto band must not be a reparse point"
        )
    for path in root.rglob("*"):
        if _is_reparse_point(path):
            raise OrthophotoAcquisitionError(
                f"temporary orthophoto band contains a reparse point: {path.name}"
            )


def _validate_endpoint(service_url: str) -> str:
    parts = urlsplit(service_url)
    if (
        parts.scheme != "https"
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise ValueError(
            "orthophoto provider must be an HTTPS endpoint without credentials "
            "or fragment"
        )
    return service_url


def _canonical_url(service_url: str, parameters: Mapping[str, str]) -> str:
    _validate_endpoint(service_url)
    parts = urlsplit(service_url)
    existing = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
    existing_keys = {name.casefold() for name, _value in existing}
    parameter_keys = {name.casefold() for name in parameters}
    if existing_keys & parameter_keys:
        raise ValueError("provider URL duplicates a canonical request parameter")
    query = urlencode(sorted([*existing, *parameters.items()]))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def build_grid(
    core_bounds_l93_m: Sequence[int | float],
) -> OrthophotoGrid:
    """Build the sole accepted 500 m + 10 m, 1 m Lambert-93 grid."""

    if len(core_bounds_l93_m) != 4:
        raise ValueError("core bounds must contain west, south, east and north")
    raw = tuple(float(value) for value in core_bounds_l93_m)
    if not all(math.isfinite(value) for value in raw):
        raise ValueError("core bounds must contain finite Lambert-93 coordinates")
    if any(not value.is_integer() for value in raw):
        raise ValueError("core bounds must align to the global 1 m grid")
    west, south, east, north = (int(value) for value in raw)
    if east - west != int(CORE_SIZE_M) or north - south != int(CORE_SIZE_M):
        raise ValueError("orthophoto band core must be exactly 500 m by 500 m")
    request_bounds = (
        west - int(HALO_M),
        south - int(HALO_M),
        east + int(HALO_M),
        north + int(HALO_M),
    )
    if request_bounds[2] - request_bounds[0] != REQUEST_SIZE_PX:
        raise AssertionError("orthophoto request width drifted from the 1 m contract")
    return OrthophotoGrid(
        core_bounds_l93_m=(west, south, east, north),
        request_bounds_l93_m=request_bounds,
    )


def _validate_common(
    *,
    band_id: str,
    provider_revision_id: str,
    layer: str,
    image_format: str,
    dependent_map_ids: Sequence[str],
    maximum_download_bytes: int,
) -> tuple[str, str, str, str, tuple[str, ...], int]:
    stable_band = _require_safe_id(band_id, "band_id")
    revision = _require_safe_id(provider_revision_id, "provider_revision_id")
    if not isinstance(layer, str) or not layer.strip():
        raise ValueError("layer must be non-empty")
    if image_format not in {"image/png", "image/jpeg"}:
        raise ValueError("image_format must be image/png or image/jpeg")
    if not isinstance(dependent_map_ids, Sequence) or isinstance(
        dependent_map_ids, (str, bytes)
    ):
        raise ValueError("dependent_map_ids must be a non-empty sequence")
    normalized_maps = tuple(
        sorted(
            _require_safe_id(value, "dependent map id") for value in dependent_map_ids
        )
    )
    if not normalized_maps or len(normalized_maps) != len(set(normalized_maps)):
        raise ValueError("dependent_map_ids must be non-empty and unique")
    if (
        isinstance(maximum_download_bytes, bool)
        or not isinstance(maximum_download_bytes, int)
        or maximum_download_bytes <= 0
    ):
        raise ValueError("maximum_download_bytes must be a positive integer")
    return (
        stable_band,
        revision,
        layer.strip(),
        image_format,
        normalized_maps,
        maximum_download_bytes,
    )


def _plan_basis(
    *,
    band_id: str,
    service_kind: ServiceKind,
    provider_revision_id: str,
    layer: str,
    image_format: str,
    grid: OrthophotoGrid,
    requests: Sequence[ImageRequest],
    dependent_map_ids: Sequence[str],
    maximum_download_bytes: int,
    wmts_matrix: WmtsMatrix | None,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "purpose": "temporary_build_time_recognition_only",
        "band_id": band_id,
        "service_kind": service_kind,
        "provider_revision_id": provider_revision_id,
        "layer": layer,
        "image_format": image_format,
        "grid": {
            "crs": grid.crs,
            "resolution_m": grid.resolution_m,
            "halo_m": grid.halo_m,
            "core_bounds_l93_m": list(grid.core_bounds_l93_m),
            "request_bounds_l93_m": list(grid.request_bounds_l93_m),
            "width": grid.width,
            "height": grid.height,
        },
        "requests": [
            {
                "key": request.key,
                "url": request.url,
                "pixel_width": request.pixel_width,
                "pixel_height": request.pixel_height,
                "tile_column": request.tile_column,
                "tile_row": request.tile_row,
            }
            for request in requests
        ],
        "dependent_map_ids": list(dependent_map_ids),
        "maximum_download_bytes": maximum_download_bytes,
        "wmts_matrix": (
            None
            if wmts_matrix is None
            else {
                "matrix_set": wmts_matrix.matrix_set,
                "matrix": wmts_matrix.matrix,
                "top_left_l93_m": list(wmts_matrix.top_left_l93_m),
                "tile_width_px": wmts_matrix.tile_width_px,
                "tile_height_px": wmts_matrix.tile_height_px,
                "matrix_width": wmts_matrix.matrix_width,
                "matrix_height": wmts_matrix.matrix_height,
                "resolution_m": wmts_matrix.resolution_m,
            }
        ),
        "runtime_exclusions": [
            "orthophoto_payload",
            "orthophoto_texture",
            "procedural_ground_material",
            "uniform_0.5m_source",
        ],
    }


def _make_plan(
    *,
    band_id: str,
    service_kind: ServiceKind,
    provider_revision_id: str,
    layer: str,
    image_format: str,
    grid: OrthophotoGrid,
    requests: tuple[ImageRequest, ...],
    dependent_map_ids: tuple[str, ...],
    maximum_download_bytes: int,
    wmts_matrix: WmtsMatrix | None = None,
) -> OrthophotoBandPlan:
    basis = _plan_basis(
        band_id=band_id,
        service_kind=service_kind,
        provider_revision_id=provider_revision_id,
        layer=layer,
        image_format=image_format,
        grid=grid,
        requests=requests,
        dependent_map_ids=dependent_map_ids,
        maximum_download_bytes=maximum_download_bytes,
        wmts_matrix=wmts_matrix,
    )
    return OrthophotoBandPlan(
        band_id=band_id,
        service_kind=service_kind,
        provider_revision_id=provider_revision_id,
        layer=layer,
        image_format=image_format,
        grid=grid,
        requests=requests,
        dependent_map_ids=dependent_map_ids,
        maximum_download_bytes=maximum_download_bytes,
        plan_sha256=_canonical_sha256(basis),
        wmts_matrix=wmts_matrix,
    )


def build_wms_plan(
    core_bounds_l93_m: Sequence[int | float],
    *,
    band_id: str,
    service_url: str,
    layer: str,
    provider_revision_id: str,
    dependent_map_ids: Sequence[str],
    maximum_download_bytes: int,
    image_format: str = "image/png",
) -> OrthophotoBandPlan:
    """Build one canonical WMS 1.3.0 request without network access."""

    common = _validate_common(
        band_id=band_id,
        provider_revision_id=provider_revision_id,
        layer=layer,
        image_format=image_format,
        dependent_map_ids=dependent_map_ids,
        maximum_download_bytes=maximum_download_bytes,
    )
    stable_band, revision, clean_layer, image_format, maps, maximum = common
    grid = build_grid(core_bounds_l93_m)
    parameters = {
        "BBOX": ",".join(str(value) for value in grid.request_bounds_l93_m),
        "CRS": CRS,
        "FORMAT": image_format,
        "HEIGHT": str(grid.height),
        "LAYERS": clean_layer,
        "REQUEST": "GetMap",
        "SERVICE": "WMS",
        "STYLES": "",
        "TRANSPARENT": "FALSE",
        "VERSION": "1.3.0",
        "WIDTH": str(grid.width),
    }
    request = ImageRequest(
        key="wms-band",
        url=_canonical_url(service_url, parameters),
        pixel_width=grid.width,
        pixel_height=grid.height,
    )
    return _make_plan(
        band_id=stable_band,
        service_kind="wms",
        provider_revision_id=revision,
        layer=clean_layer,
        image_format=image_format,
        grid=grid,
        requests=(request,),
        dependent_map_ids=maps,
        maximum_download_bytes=maximum,
    )


def _validate_wmts_matrix(matrix: WmtsMatrix) -> WmtsMatrix:
    if not isinstance(matrix, WmtsMatrix):
        raise ValueError("wmts_matrix must be a WmtsMatrix")
    _require_safe_id(matrix.matrix_set, "wmts_matrix.matrix_set")
    _require_safe_id(matrix.matrix, "wmts_matrix.matrix")
    if len(matrix.top_left_l93_m) != 2 or not all(
        math.isfinite(float(value)) for value in matrix.top_left_l93_m
    ):
        raise ValueError("WMTS top-left origin must contain two finite coordinates")
    if any(
        not math.isclose(float(value), round(float(value)), abs_tol=1e-9)
        for value in matrix.top_left_l93_m
    ):
        raise ValueError("WMTS top-left origin must align to the global 1 m grid")
    if matrix.resolution_m != RESOLUTION_M:
        raise ValueError("WMTS matrix resolution must be exactly 1 m")
    for value, label in (
        (matrix.tile_width_px, "tile_width_px"),
        (matrix.tile_height_px, "tile_height_px"),
        (matrix.matrix_width, "matrix_width"),
        (matrix.matrix_height, "matrix_height"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"WMTS {label} must be a positive integer")
    return matrix


def build_wmts_plan(
    core_bounds_l93_m: Sequence[int | float],
    *,
    band_id: str,
    service_url: str,
    layer: str,
    provider_revision_id: str,
    dependent_map_ids: Sequence[str],
    maximum_download_bytes: int,
    wmts_matrix: WmtsMatrix,
    style: str = "normal",
    image_format: str = "image/png",
) -> OrthophotoBandPlan:
    """Build the exact WMTS GetTile set covering the halo-expanded band."""

    common = _validate_common(
        band_id=band_id,
        provider_revision_id=provider_revision_id,
        layer=layer,
        image_format=image_format,
        dependent_map_ids=dependent_map_ids,
        maximum_download_bytes=maximum_download_bytes,
    )
    stable_band, revision, clean_layer, image_format, maps, maximum = common
    matrix = _validate_wmts_matrix(wmts_matrix)
    if not isinstance(style, str) or not style.strip():
        raise ValueError("WMTS style must be non-empty")
    grid = build_grid(core_bounds_l93_m)
    origin_x, origin_y = (float(value) for value in matrix.top_left_l93_m)
    tile_span_x = matrix.tile_width_px * RESOLUTION_M
    tile_span_y = matrix.tile_height_px * RESOLUTION_M
    west, south, east, north = grid.request_bounds_l93_m
    first_column = math.floor((west - origin_x) / tile_span_x)
    last_column = math.ceil((east - origin_x) / tile_span_x) - 1
    first_row = math.floor((origin_y - north) / tile_span_y)
    last_row = math.ceil((origin_y - south) / tile_span_y) - 1
    if (
        first_column < 0
        or first_row < 0
        or last_column >= matrix.matrix_width
        or last_row >= matrix.matrix_height
    ):
        raise ValueError("halo-expanded band falls outside the declared WMTS matrix")
    requests: list[ImageRequest] = []
    for row in range(first_row, last_row + 1):
        for column in range(first_column, last_column + 1):
            parameters = {
                "FORMAT": image_format,
                "LAYER": clean_layer,
                "REQUEST": "GetTile",
                "SERVICE": "WMTS",
                "STYLE": style.strip(),
                "TILECOL": str(column),
                "TILEMATRIX": matrix.matrix,
                "TILEMATRIXSET": matrix.matrix_set,
                "TILEROW": str(row),
                "VERSION": "1.0.0",
            }
            requests.append(
                ImageRequest(
                    key=f"r{row}_c{column}",
                    url=_canonical_url(service_url, parameters),
                    pixel_width=matrix.tile_width_px,
                    pixel_height=matrix.tile_height_px,
                    tile_column=column,
                    tile_row=row,
                )
            )
    return _make_plan(
        band_id=stable_band,
        service_kind="wmts",
        provider_revision_id=revision,
        layer=clean_layer,
        image_format=image_format,
        grid=grid,
        requests=tuple(requests),
        dependent_map_ids=maps,
        maximum_download_bytes=maximum,
        wmts_matrix=matrix,
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    target = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == target:
            return str(value)
    return None


def _response_status(response: Response) -> int | None:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()  # type: ignore[attr-defined]
    return None if status is None else int(status)


def _copy_limited(response: Response, destination: BinaryIO, maximum: int) -> int:
    received = 0
    while True:
        chunk = response.read(min(1024 * 1024, maximum - received + 1))
        if not chunk:
            return received
        received += len(chunk)
        if received > maximum:
            raise OrthophotoAcquisitionError(
                "orthophoto response exceeds maximum_download_bytes"
            )
        destination.write(chunk)


def receive_to_part(
    request_url: str,
    destination: Path,
    *,
    maximum_bytes: int,
    opener: Opener = urlopen,
    timeout_s: float = 180.0,
) -> DownloadReceipt:
    """Strict HTTP Range transfer retained in ``.part`` until image validation."""

    _validate_endpoint(request_url)
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    if timeout_s <= 0 or not math.isfinite(timeout_s):
        raise ValueError("timeout_s must be positive and finite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > maximum_bytes:
        raise OrthophotoAcquisitionError("existing .part exceeds its disk contract")
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(request_url, headers=headers)

    try:
        response_context = opener(request, timeout=timeout_s)
    except HTTPError as error:
        if offset and error.code == 416:
            digest = sha256_file(part)
            return DownloadReceipt(offset, digest, offset)
        raise
    with response_context as response:
        status = _response_status(response)
        content_length_value = _header(response.headers, "Content-Length")
        content_length = (
            int(content_length_value) if content_length_value is not None else None
        )
        if content_length is not None and offset + content_length > maximum_bytes:
            raise OrthophotoAcquisitionError(
                "orthophoto Content-Length exceeds maximum_download_bytes"
            )
        if offset:
            if status != 206:
                raise OrthophotoAcquisitionError(
                    "resumed orthophoto transfer expected HTTP 206"
                )
            content_range = _header(response.headers, "Content-Range")
            match = _CONTENT_RANGE.fullmatch(content_range or "")
            if match is None or int(match.group(1)) != offset:
                raise OrthophotoAcquisitionError(
                    "resumed orthophoto transfer has invalid Content-Range"
                )
            total = int(match.group(3))
            if total > maximum_bytes or int(match.group(2)) + 1 != total:
                raise OrthophotoAcquisitionError(
                    "resumed orthophoto transfer has an invalid total size"
                )
            mode = "ab"
        else:
            if status not in {None, 200}:
                raise OrthophotoAcquisitionError(
                    f"fresh orthophoto transfer expected HTTP 200, got {status}"
                )
            mode = "wb"
        with part.open(mode) as output:
            _copy_limited(response, output, maximum_bytes - offset)
            output.flush()
            os.fsync(output.fileno())
    size = part.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise OrthophotoAcquisitionError("orthophoto transfer size is invalid")
    return DownloadReceipt(size, sha256_file(part), offset)


def _decode_rgb(path: Path, request: ImageRequest) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.size != (request.pixel_width, request.pixel_height):
                raise OrthophotoAcquisitionError(
                    f"orthophoto image dimensions differ for {request.key}"
                )
            rgba = image.convert("RGBA")
            array = np.asarray(rgba, dtype=np.uint8)
    except OrthophotoAcquisitionError:
        raise
    except Exception as error:
        raise OrthophotoAcquisitionError(
            f"orthophoto response is not a valid image for {request.key}: {error}"
        ) from error
    if array.shape != (request.pixel_height, request.pixel_width, 4):
        raise OrthophotoAcquisitionError("decoded orthophoto image shape is invalid")
    if np.any(array[:, :, 3] != 255):
        raise OrthophotoAcquisitionError(
            f"orthophoto image contains transparency for {request.key}"
        )
    return np.ascontiguousarray(array[:, :, :3])


def _compose_rgb(
    plan: OrthophotoBandPlan, images: Mapping[str, np.ndarray]
) -> np.ndarray:
    if plan.service_kind == "wms":
        return np.ascontiguousarray(images[plan.requests[0].key], dtype=np.uint8)
    matrix = plan.wmts_matrix
    if matrix is None:
        raise OrthophotoAcquisitionError("WMTS plan has no matrix")
    rows = sorted({int(request.tile_row) for request in plan.requests})
    columns = sorted({int(request.tile_column) for request in plan.requests})
    mosaic = np.empty(
        (
            len(rows) * matrix.tile_height_px,
            len(columns) * matrix.tile_width_px,
            3,
        ),
        dtype=np.uint8,
    )
    row_index = {value: index for index, value in enumerate(rows)}
    column_index = {value: index for index, value in enumerate(columns)}
    for request in plan.requests:
        y = row_index[int(request.tile_row)] * matrix.tile_height_px
        x = column_index[int(request.tile_column)] * matrix.tile_width_px
        mosaic[
            y : y + matrix.tile_height_px,
            x : x + matrix.tile_width_px,
        ] = images[request.key]
    origin_x, origin_y = (float(value) for value in matrix.top_left_l93_m)
    mosaic_west = origin_x + columns[0] * matrix.tile_width_px
    mosaic_north = origin_y - rows[0] * matrix.tile_height_px
    west, _south, _east, north = plan.grid.request_bounds_l93_m
    x_offset = int(round(west - mosaic_west))
    y_offset = int(round(mosaic_north - north))
    cropped = mosaic[
        y_offset : y_offset + plan.grid.height,
        x_offset : x_offset + plan.grid.width,
    ]
    if cropped.shape != (plan.grid.height, plan.grid.width, 3):
        raise OrthophotoAcquisitionError("WMTS mosaic does not cover the exact band")
    return np.ascontiguousarray(cropped)


def _rgb_sha256(rgb: np.ndarray) -> str:
    canonical = np.ascontiguousarray(rgb, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(b"FIREVIEWER-ORTHOPHOTO-RGB-1M-EPSG2154-V1\0")
    digest.update(canonical.shape[0].to_bytes(4, "little"))
    digest.update(canonical.shape[1].to_bytes(4, "little"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_npy(path: Path, rgb: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("xb") as output:
            np.save(
                output, np.ascontiguousarray(rgb, dtype=np.uint8), allow_pickle=False
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
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
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _plan_contract(plan: OrthophotoBandPlan) -> dict[str, object]:
    basis = _plan_basis(
        band_id=plan.band_id,
        service_kind=plan.service_kind,
        provider_revision_id=plan.provider_revision_id,
        layer=plan.layer,
        image_format=plan.image_format,
        grid=plan.grid,
        requests=plan.requests,
        dependent_map_ids=plan.dependent_map_ids,
        maximum_download_bytes=plan.maximum_download_bytes,
        wmts_matrix=plan.wmts_matrix,
    )
    if _canonical_sha256(basis) != plan.plan_sha256:
        raise OrthophotoAcquisitionError("orthophoto plan identity is not canonical")
    return basis


def _validate_existing_receipt(
    plan: OrthophotoBandPlan, band_root: Path
) -> dict[str, object]:
    receipt_path = band_root / RECEIPT_FILE_NAME
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OrthophotoAcquisitionError(
            f"invalid orthophoto source receipt: {error}"
        ) from error
    rgb_path = band_root / RGB_FILE_NAME
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "prepared_temporary_build_source"
        or receipt.get("purpose") != "recognition_input_for_correspondence_compiler"
        or receipt.get("plan_sha256") != plan.plan_sha256
        or receipt.get("band_id") != plan.band_id
        or receipt.get("provider_revision_id") != plan.provider_revision_id
        or receipt.get("service_kind") != plan.service_kind
        or receipt.get("raw_sources_retained") is not False
        or receipt.get("dependent_map_ids") != list(plan.dependent_map_ids)
        or receipt.get("projected_peak_disk_bytes") != plan.projected_peak_disk_bytes
        or not rgb_path.is_file()
    ):
        raise OrthophotoAcquisitionError("orthophoto source receipt is incoherent")
    expected_grid = {
        "crs": CRS,
        "resolution_m": RESOLUTION_M,
        "halo_m": HALO_M,
        "core_bounds_l93_m": list(plan.grid.core_bounds_l93_m),
        "request_bounds_l93_m": list(plan.grid.request_bounds_l93_m),
        "width": plan.grid.width,
        "height": plan.grid.height,
    }
    if receipt.get("grid") != expected_grid:
        raise OrthophotoAcquisitionError("orthophoto receipt grid differs from plan")
    raw_sources = receipt.get("raw_sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != len(plan.requests):
        raise OrthophotoAcquisitionError("orthophoto raw source proofs are incomplete")
    total_raw_bytes = 0
    for index, (raw, request) in enumerate(
        zip(raw_sources, plan.requests, strict=True)
    ):
        if not isinstance(raw, dict):
            raise OrthophotoAcquisitionError(
                f"orthophoto raw source proof {index} is invalid"
            )
        byte_count = raw.get("byte_count")
        resumed = raw.get("resumed_from_byte")
        if (
            raw.get("key") != request.key
            or raw.get("request_sha256")
            != hashlib.sha256(request.url.encode("utf-8")).hexdigest()
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count <= 0
            or not isinstance(resumed, int)
            or isinstance(resumed, bool)
            or not 0 <= resumed <= byte_count
        ):
            raise OrthophotoAcquisitionError(
                f"orthophoto raw source proof differs for {request.key}"
            )
        _require_sha256(str(raw.get("sha256")), f"raw source {request.key}")
        total_raw_bytes += byte_count
    if total_raw_bytes > plan.maximum_download_bytes:
        raise OrthophotoAcquisitionError("orthophoto raw source bytes exceed the plan")
    canonical = receipt.get("canonical_rgb")
    if (
        not isinstance(canonical, dict)
        or canonical.get("file_name") != RGB_FILE_NAME
        or canonical.get("file_sha256") != sha256_file(rgb_path)
        or canonical.get("byte_count") != rgb_path.stat().st_size
        or canonical.get("shape") != [plan.grid.height, plan.grid.width, 3]
        or canonical.get("dtype") != "uint8"
        or canonical.get("row_order") != "north_to_south"
        or canonical.get("crs") != CRS
        or canonical.get("gdal_geotransform")
        != [
            plan.grid.request_bounds_l93_m[0],
            RESOLUTION_M,
            0.0,
            plan.grid.request_bounds_l93_m[3],
            0.0,
            -RESOLUTION_M,
        ]
    ):
        raise OrthophotoAcquisitionError("canonical orthophoto RGB was changed")
    load_canonical_rgb(plan, band_root, _validated_receipt=receipt)
    return receipt


def prepare_orthophoto_band(
    plan: OrthophotoBandPlan,
    temporary_root: Path,
    *,
    opener: Opener = urlopen,
    disk_usage: DiskUsage = shutil.disk_usage,
    timeout_s: float = 180.0,
) -> dict[str, object]:
    """Acquire, hash and canonicalize one temporary source band on D:."""

    _plan_contract(plan)
    root = _require_temporary_root(temporary_root)
    parent = root / TEMPORARY_DIRECTORY_NAME
    band_root = parent / plan.band_id
    staging = parent / f".{plan.band_id}.staging"
    for candidate in (band_root, staging):
        if candidate.exists() and _is_reparse_point(candidate):
            raise OrthophotoAcquisitionError(
                "temporary orthophoto band path must not be a reparse point"
            )
    if band_root.is_dir():
        return _validate_existing_receipt(plan, band_root)
    if band_root.exists():
        raise OrthophotoAcquisitionError("orthophoto band target is not a directory")
    parent.mkdir(parents=True, exist_ok=True)
    usage = disk_usage(parent)
    free = int(getattr(usage, "free"))
    required_free = plan.projected_peak_disk_bytes + DISK_SAFETY_MARGIN_BYTES
    if free < required_free:
        raise OrthophotoAcquisitionError(
            f"insufficient D: space: {free} free, {required_free} required "
            "(orthophoto peak plus 20 GiB)"
        )
    staging.mkdir(parents=True, exist_ok=True)

    images: dict[str, np.ndarray] = {}
    raw_receipts: list[dict[str, object]] = []
    total_bytes = 0
    for request in plan.requests:
        remaining_bytes = plan.maximum_download_bytes - total_bytes
        if remaining_bytes <= 0:
            raise OrthophotoAcquisitionError(
                "orthophoto responses exhausted maximum_download_bytes"
            )
        raw_path = staging / f"{request.key}.image"
        receipt = receive_to_part(
            request.url,
            raw_path,
            maximum_bytes=remaining_bytes,
            opener=opener,
            timeout_s=timeout_s,
        )
        total_bytes += receipt.byte_count
        if total_bytes > plan.maximum_download_bytes:
            raise OrthophotoAcquisitionError(
                "orthophoto responses exceed maximum_download_bytes"
            )
        part = raw_path.with_name(raw_path.name + ".part")
        images[request.key] = _decode_rgb(part, request)
        raw_receipts.append(
            {
                "key": request.key,
                "request_sha256": hashlib.sha256(
                    request.url.encode("utf-8")
                ).hexdigest(),
                "byte_count": receipt.byte_count,
                "sha256": receipt.sha256,
                "resumed_from_byte": receipt.resumed_from_byte,
            }
        )

    rgb = _compose_rgb(plan, images)
    if rgb.shape != (plan.grid.height, plan.grid.width, 3) or rgb.dtype != np.uint8:
        raise OrthophotoAcquisitionError("canonical RGB does not match its grid")
    rgb_path = staging / RGB_FILE_NAME
    _atomic_npy(rgb_path, rgb)
    for part in staging.glob("*.image.part"):
        part.unlink()
    receipt_payload: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": "prepared_temporary_build_source",
        "purpose": "recognition_input_for_correspondence_compiler",
        "band_id": plan.band_id,
        "plan_sha256": plan.plan_sha256,
        "provider_revision_id": plan.provider_revision_id,
        "service_kind": plan.service_kind,
        "grid": {
            "crs": CRS,
            "resolution_m": RESOLUTION_M,
            "halo_m": HALO_M,
            "core_bounds_l93_m": list(plan.grid.core_bounds_l93_m),
            "request_bounds_l93_m": list(plan.grid.request_bounds_l93_m),
            "width": plan.grid.width,
            "height": plan.grid.height,
        },
        "raw_sources": raw_receipts,
        "raw_sources_retained": False,
        "canonical_rgb": {
            "file_name": RGB_FILE_NAME,
            "byte_count": rgb_path.stat().st_size,
            "file_sha256": sha256_file(rgb_path),
            "pixel_sha256": _rgb_sha256(rgb),
            "shape": list(rgb.shape),
            "dtype": "uint8",
            "row_order": "north_to_south",
            "crs": CRS,
            "gdal_geotransform": [
                plan.grid.request_bounds_l93_m[0],
                RESOLUTION_M,
                0.0,
                plan.grid.request_bounds_l93_m[3],
                0.0,
                -RESOLUTION_M,
            ],
        },
        "dependent_map_ids": list(plan.dependent_map_ids),
        "retention": "delete_band_after_all_dependent_map_receipts_are_sealed",
        "projected_peak_disk_bytes": plan.projected_peak_disk_bytes,
        "runtime_exclusions": [
            "orthophoto_payload",
            "orthophoto_texture",
            "procedural_ground_material",
            "uniform_0.5m_source",
        ],
    }
    _atomic_json(staging / RECEIPT_FILE_NAME, receipt_payload)
    os.replace(staging, band_root)
    return _validate_existing_receipt(plan, band_root)


def load_canonical_rgb(
    plan: OrthophotoBandPlan,
    band_root: Path,
    *,
    _validated_receipt: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Load the exact RGB array exposed to the future correspondence compiler."""

    _plan_contract(plan)
    root = _require_d_path(band_root, "band_root")
    receipt = (
        _validated_receipt
        if _validated_receipt is not None
        else _validate_existing_receipt(plan, root)
    )
    rgb_path = root / RGB_FILE_NAME
    try:
        with rgb_path.open("rb") as source:
            rgb = np.load(source, allow_pickle=False)
    except Exception as error:
        raise OrthophotoAcquisitionError(
            f"canonical orthophoto RGB cannot be loaded: {error}"
        ) from error
    canonical = receipt.get("canonical_rgb")
    if not isinstance(canonical, Mapping):
        raise OrthophotoAcquisitionError("canonical RGB receipt is absent")
    if (
        rgb.shape != (plan.grid.height, plan.grid.width, 3)
        or rgb.dtype != np.uint8
        or canonical.get("pixel_sha256") != _rgb_sha256(rgb)
    ):
        raise OrthophotoAcquisitionError("canonical orthophoto RGB identity differs")
    return np.ascontiguousarray(rgb)


def cleanup_orthophoto_band(
    plan: OrthophotoBandPlan,
    temporary_root: Path,
    sealed_maps: Sequence[SealedDependentMap],
) -> dict[str, object]:
    """Delete the exact temporary band after every opaque map receipt is sealed."""

    _plan_contract(plan)
    root = _require_temporary_root(temporary_root)
    band_root = (root / TEMPORARY_DIRECTORY_NAME / plan.band_id).resolve()
    expected_parent = (root / TEMPORARY_DIRECTORY_NAME).resolve()
    if band_root.parent != expected_parent or band_root == expected_parent:
        raise OrthophotoAcquisitionError("orthophoto cleanup target escaped its root")
    _validate_existing_receipt(plan, band_root)
    by_id: dict[str, SealedDependentMap] = {}
    for sealed in sealed_maps:
        if not isinstance(sealed, SealedDependentMap):
            raise OrthophotoAcquisitionError("sealed map proof has an invalid type")
        map_id = _require_safe_id(sealed.map_id, "sealed map id")
        if map_id in by_id:
            raise OrthophotoAcquisitionError("sealed map ids must be unique")
        path = _require_d_path(sealed.receipt_path, f"sealed map {map_id}")
        digest = _require_sha256(sealed.receipt_sha256, f"sealed map {map_id}")
        if path == band_root or band_root in path.parents:
            raise OrthophotoAcquisitionError(
                f"dependent map receipt must survive band cleanup: {map_id}"
            )
        if not path.is_file() or sha256_file(path) != digest:
            raise OrthophotoAcquisitionError(
                f"dependent map receipt is absent or changed: {map_id}"
            )
        by_id[map_id] = sealed
    if set(by_id) != set(plan.dependent_map_ids):
        missing = sorted(set(plan.dependent_map_ids) - set(by_id))
        extra = sorted(set(by_id) - set(plan.dependent_map_ids))
        raise OrthophotoAcquisitionError(
            f"dependent map seal set is incomplete: missing={missing}, extra={extra}"
        )
    _reject_reparse_tree(band_root)
    deleted_bytes = sum(
        path.stat().st_size for path in band_root.rglob("*") if path.is_file()
    )
    shutil.rmtree(band_root)
    return {
        "schema": CLEANUP_SCHEMA,
        "status": "temporary_orthophoto_band_deleted",
        "band_id": plan.band_id,
        "plan_sha256": plan.plan_sha256,
        "deleted_bytes": deleted_bytes,
        "deleted_path": str(band_root),
        "sealed_map_receipts": {
            map_id: {
                "path": str(by_id[map_id].receipt_path.resolve()),
                "sha256": by_id[map_id].receipt_sha256,
            }
            for map_id in sorted(by_id)
        },
        "recoverable": False,
    }


__all__ = [
    "CLEANUP_SCHEMA",
    "CORE_SIZE_M",
    "CRS",
    "DISK_SAFETY_MARGIN_BYTES",
    "HALO_M",
    "OrthophotoAcquisitionError",
    "OrthophotoBandPlan",
    "OrthophotoGrid",
    "RECEIPT_FILE_NAME",
    "RESOLUTION_M",
    "RGB_FILE_NAME",
    "SealedDependentMap",
    "WmtsMatrix",
    "build_grid",
    "build_wms_plan",
    "build_wmts_plan",
    "cleanup_orthophoto_band",
    "load_canonical_rgb",
    "prepare_orthophoto_band",
    "receive_to_part",
    "sha256_file",
]
