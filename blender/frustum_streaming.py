"""Deterministic frustum-driven terrain residency planner.

This module is deliberately independent from :mod:`bpy`.  It owns the
geometry, budget and transition contracts that an OpenUSD or Blender adapter
can execute.  Unlike the deprecated radial planner, it has no tile-count cap
and never publishes a lower terrain LOD inside the active camera frustum.

The public transition handshake is intentionally small: ``stage_camera``
validates a complete double-buffered plan, ``tick`` emits at most one action,
and ``commit`` acknowledges the exact action.  The previously published
camera remains active until every LOD0 payload required by the staged camera
has loaded successfully.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TILE_SIZE_M = 500.0
GUARD_RING_COUNT = 1
LOD1_RING_COUNT = 2
DEFAULT_MEMORY_RESERVE_FRACTION = 0.25
MINIMUM_MEMORY_RESERVE_FRACTION = 0.25
DEFAULT_HYSTERESIS_SECONDS = 3.0
DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT = 3
DEFAULT_LOAD_RETRY_BACKOFF_TICKS = 2
DEFAULT_PREDICTION_HORIZON_SECONDS = 2.0
DEFAULT_PREDICTION_INTERVAL_SECONDS = 0.25

LOD0 = 0
LOD1 = 1
LOD2 = 2

EDGE_ORDER = ("west", "east", "south", "north")
STITCH_MASK_BITS = {"west": 1, "east": 2, "south": 4, "north": 8}
ALL_STITCH_MASKS = tuple(range(16))
_EDGE_OFFSETS = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}

LOAD_PAYLOAD = "load_payload"
EVICT_PAYLOADS = "evict_payloads"
PUBLISH_CAMERA = "publish_camera"
NOOP = "noop"

CONTRACT_SCHEMA = "fireviewer.terrain-streaming-contract.v2"
CAMERA_ENVELOPE_SCHEMA = "fireviewer.camera-envelope.v1"
CAMERA_RESIDENCY_PLAN_SCHEMA = "fireviewer.camera-residency-plan.v1"
STREAMING_STATE_SCHEMA = "fireviewer.terrain-streaming-state.v2"

_EPSILON = 1.0e-9

Vec3 = tuple[float, float, float]


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


def _non_negative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_integer(value: Any, field_name: str) -> int:
    result = _non_negative_integer(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return result


def _sha256_digest(value: Any, field_name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return result


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], field_name: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{field_name} keys differ from contract; missing={missing!r}, extra={extra!r}"
        )


def _vec3(value: Sequence[float], field_name: str) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return tuple(
        _finite_number(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _add(left: Vec3, right: Vec3) -> Vec3:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _subtract(left: Vec3, right: Vec3) -> Vec3:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _scale(vector: Vec3, scalar: float) -> Vec3:
    return vector[0] * scalar, vector[1] * scalar, vector[2] * scalar


def _dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _length(vector: Vec3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vec3, field_name: str) -> Vec3:
    length = _length(vector)
    if length <= _EPSILON:
        raise ValueError(f"{field_name} must be non-zero")
    return _scale(vector, 1.0 / length)


def _angle_degrees(left: Vec3, right: Vec3) -> float:
    normalized_left = _normalize(left, "left vector")
    normalized_right = _normalize(right, "right vector")
    cosine = max(-1.0, min(1.0, _dot(normalized_left, normalized_right)))
    return math.degrees(math.acos(cosine))


def _rotate_axis_angle(vector: Vec3, axis: Vec3, angle_radians: float) -> Vec3:
    """Rotate ``vector`` using deterministic Rodrigues arithmetic."""

    unit_axis = _normalize(axis, "angular_velocity_axis")
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return _add(
        _add(
            _scale(vector, cosine),
            _scale(_cross(unit_axis, vector), sine),
        ),
        _scale(unit_axis, _dot(unit_axis, vector) * (1.0 - cosine)),
    )


@dataclass(frozen=True)
class Aabb3D:
    """Axis-aligned Lambert-93 / NGF-IGN69 tile bounds."""

    minimum: Vec3
    maximum: Vec3

    def __post_init__(self) -> None:
        minimum = _vec3(self.minimum, "minimum")
        maximum = _vec3(self.maximum, "maximum")
        if any(minimum[index] > maximum[index] for index in range(3)):
            raise ValueError("AABB minimum cannot exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def corners(self) -> tuple[Vec3, ...]:
        x0, y0, z0 = self.minimum
        x1, y1, z1 = self.maximum
        return (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        )


@dataclass(frozen=True)
class CameraView:
    """One perspective view participating in a camera envelope."""

    view_id: str
    position_l93_ngf_m: Vec3
    forward: Vec3
    up: Vec3
    vertical_fov_deg: float
    aspect_ratio: float
    near_clip_m: float
    far_clip_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.view_id, str) or not self.view_id:
            raise ValueError("view_id must be a non-empty string")
        position = _vec3(self.position_l93_ngf_m, "position_l93_ngf_m")
        forward = _normalize(_vec3(self.forward, "forward"), "forward")
        up_hint = _normalize(_vec3(self.up, "up"), "up")
        right = _cross(forward, up_hint)
        if _length(right) <= _EPSILON:
            raise ValueError("forward and up must not be collinear")
        right = _normalize(right, "camera right")
        orthogonal_up = _normalize(_cross(right, forward), "camera up")
        vertical_fov = _finite_number(self.vertical_fov_deg, "vertical_fov_deg")
        aspect = _finite_number(self.aspect_ratio, "aspect_ratio")
        near_clip = _finite_number(self.near_clip_m, "near_clip_m")
        far_clip = _finite_number(self.far_clip_m, "far_clip_m")
        if not 0.0 < vertical_fov < 179.0:
            raise ValueError("vertical_fov_deg must be between 0 and 179")
        if aspect <= 0.0:
            raise ValueError("aspect_ratio must be strictly positive")
        if near_clip <= 0.0 or far_clip <= near_clip:
            raise ValueError("far_clip_m must be greater than positive near_clip_m")
        object.__setattr__(self, "position_l93_ngf_m", position)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "up", orthogonal_up)
        object.__setattr__(self, "vertical_fov_deg", vertical_fov)
        object.__setattr__(self, "aspect_ratio", aspect)
        object.__setattr__(self, "near_clip_m", near_clip)
        object.__setattr__(self, "far_clip_m", far_clip)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CameraView":
        return cls(
            view_id=str(payload["view_id"]),
            position_l93_ngf_m=_vec3(
                payload["position_l93_ngf_m"], "position_l93_ngf_m"
            ),
            forward=_vec3(payload["forward"], "forward"),
            up=_vec3(payload["up"], "up"),
            vertical_fov_deg=payload["vertical_fov_deg"],
            aspect_ratio=payload["aspect_ratio"],
            near_clip_m=payload["near_clip_m"],
            far_clip_m=payload["far_clip_m"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "position_l93_ngf_m": list(self.position_l93_ngf_m),
            "forward": list(self.forward),
            "up": list(self.up),
            "vertical_fov_deg": self.vertical_fov_deg,
            "aspect_ratio": self.aspect_ratio,
            "near_clip_m": self.near_clip_m,
            "far_clip_m": self.far_clip_m,
        }


@dataclass(frozen=True)
class CameraEnvelope:
    """Union of all views/zooms that must be prepared for one camera."""

    camera_id: str
    views: tuple[CameraView, ...]
    mode: str = "planned"

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        views = tuple(self.views)
        if not views:
            raise ValueError("A camera envelope requires at least one view")
        identifiers = [view.view_id for view in views]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Camera view identifiers must be unique")
        if self.mode not in {"planned", "predicted", "interactive"}:
            raise ValueError("mode must be planned, predicted, or interactive")
        object.__setattr__(self, "views", views)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CameraEnvelope":
        schema = payload.get("schema")
        if schema != CAMERA_ENVELOPE_SCHEMA:
            raise ValueError(
                f"Camera envelope schema must be {CAMERA_ENVELOPE_SCHEMA!r}"
            )
        return cls(
            camera_id=str(payload["camera_id"]),
            mode=str(payload.get("mode", "planned")),
            views=tuple(CameraView.from_mapping(item) for item in payload["views"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMERA_ENVELOPE_SCHEMA,
            "camera_id": self.camera_id,
            "mode": self.mode,
            "views": [view.to_dict() for view in self.views],
        }

    @property
    def sha256(self) -> str:
        """Digest the normalized envelope that produced the frustum union."""

        return _canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class Frustum:
    """Convex perspective frustum with an exact AABB SAT test."""

    corners: tuple[Vec3, ...]
    face_normals: tuple[Vec3, ...]
    edge_directions: tuple[Vec3, ...]

    @classmethod
    def from_camera_view(cls, view: CameraView) -> "Frustum":
        forward = view.forward
        up = view.up
        right = _normalize(_cross(forward, up), "camera right")
        near_center = _add(view.position_l93_ngf_m, _scale(forward, view.near_clip_m))
        far_center = _add(view.position_l93_ngf_m, _scale(forward, view.far_clip_m))
        tangent = math.tan(math.radians(view.vertical_fov_deg) * 0.5)
        near_half_height = tangent * view.near_clip_m
        near_half_width = near_half_height * view.aspect_ratio
        far_half_height = tangent * view.far_clip_m
        far_half_width = far_half_height * view.aspect_ratio

        def corner(
            center: Vec3, half_width: float, half_height: float, x: int, y: int
        ) -> Vec3:
            return _add(
                _add(center, _scale(right, half_width * x)),
                _scale(up, half_height * y),
            )

        corners = (
            corner(near_center, near_half_width, near_half_height, -1, -1),
            corner(near_center, near_half_width, near_half_height, 1, -1),
            corner(near_center, near_half_width, near_half_height, 1, 1),
            corner(near_center, near_half_width, near_half_height, -1, 1),
            corner(far_center, far_half_width, far_half_height, -1, -1),
            corner(far_center, far_half_width, far_half_height, 1, -1),
            corner(far_center, far_half_width, far_half_height, 1, 1),
            corner(far_center, far_half_width, far_half_height, -1, 1),
        )
        faces = (
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (3, 2, 6, 7),
            (0, 3, 7, 4),
            (1, 5, 6, 2),
        )
        centroid = tuple(
            sum(point[axis] for point in corners) / len(corners) for axis in range(3)
        )
        normals: list[Vec3] = []
        for face in faces:
            first, second, third = (corners[index] for index in face[:3])
            normal = _normalize(
                _cross(_subtract(second, first), _subtract(third, first)),
                "frustum face normal",
            )
            if _dot(normal, _subtract(centroid, first)) < 0.0:
                normal = _scale(normal, -1.0)
            normals.append(normal)
        edge_indices = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        edges = tuple(
            _normalize(_subtract(corners[end], corners[start]), "frustum edge")
            for start, end in edge_indices
        )
        return cls(corners=corners, face_normals=tuple(normals), edge_directions=edges)

    def intersects_aabb(self, bounds: Aabb3D, *, epsilon: float = _EPSILON) -> bool:
        """Return exact convex intersection, including tangent contact.

        Face-plane tests alone are only conservative for two arbitrary convex
        polyhedra.  The complete separating-axis set includes both face normal
        families and every cross product between AABB and frustum edge axes.
        """

        aabb_axes: tuple[Vec3, ...] = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        axes: list[Vec3] = [*aabb_axes, *self.face_normals]
        for aabb_axis in aabb_axes:
            for edge in self.edge_directions:
                axis = _cross(aabb_axis, edge)
                if _length(axis) > epsilon:
                    axes.append(_normalize(axis, "separating axis"))
        aabb_corners = bounds.corners
        for axis in axes:
            frustum_projection = [_dot(point, axis) for point in self.corners]
            aabb_projection = [_dot(point, axis) for point in aabb_corners]
            if max(frustum_projection) < min(aabb_projection) - epsilon:
                return False
            if max(aabb_projection) < min(frustum_projection) - epsilon:
                return False
        return True


@dataclass(frozen=True)
class ResourceCost:
    cpu_bytes: int
    gpu_bytes: int
    triangles: int

    def __post_init__(self) -> None:
        for name in ("cpu_bytes", "gpu_bytes", "triangles"):
            object.__setattr__(
                self, name, _non_negative_integer(getattr(self, name), name)
            )

    def __add__(self, other: "ResourceCost") -> "ResourceCost":
        return ResourceCost(
            cpu_bytes=self.cpu_bytes + other.cpu_bytes,
            gpu_bytes=self.gpu_bytes + other.gpu_bytes,
            triangles=self.triangles + other.triangles,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "cpu_bytes": self.cpu_bytes,
            "gpu_bytes": self.gpu_bytes,
            "triangles": self.triangles,
        }


ZERO_COST = ResourceCost(cpu_bytes=0, gpu_bytes=0, triangles=0)


@dataclass(frozen=True, order=True)
class PayloadRef:
    tile_id: str
    lod: int
    stitch_mask: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tile_id, str) or not self.tile_id:
            raise ValueError("tile_id must be a non-empty string")
        if self.lod not in {LOD0, LOD1, LOD2}:
            raise ValueError("lod must be 0, 1, or 2")
        if (
            isinstance(self.stitch_mask, bool)
            or not isinstance(self.stitch_mask, int)
            or self.stitch_mask not in ALL_STITCH_MASKS
        ):
            raise ValueError("stitch_mask must be a four-bit integer from 0 to 15")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PayloadRef":
        return cls(
            tile_id=str(payload["tile_id"]),
            lod=payload["lod"],
            stitch_mask=payload["stitch_mask"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "lod": self.lod,
            "stitch_mask": self.stitch_mask,
        }


@dataclass(frozen=True)
class TerrainTile:
    tile_id: str
    grid_x: int
    grid_y: int
    bounds: Aabb3D
    costs: tuple[ResourceCost, ResourceCost, ResourceCost]
    build_id: str
    payload_sha256: tuple[str, str, str]
    stitch_masks: tuple[int, ...]
    stitch_triangle_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tile_id, str) or not self.tile_id:
            raise ValueError("tile_id must be a non-empty string")
        if isinstance(self.grid_x, bool) or not isinstance(self.grid_x, int):
            raise ValueError("grid_x must be an integer")
        if isinstance(self.grid_y, bool) or not isinstance(self.grid_y, int):
            raise ValueError("grid_y must be an integer")
        if len(self.costs) != 3:
            raise ValueError("costs must contain LOD0, LOD1, and LOD2")
        build_id = _sha256_digest(self.build_id, "build_id")
        if len(self.payload_sha256) != 3:
            raise ValueError("payload_sha256 must contain LOD0, LOD1, and LOD2")
        payload_sha256 = tuple(
            _sha256_digest(value, f"payload_sha256[{lod}]")
            for lod, value in enumerate(self.payload_sha256)
        )
        object.__setattr__(self, "build_id", build_id)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        stitch_masks = tuple(self.stitch_masks)
        if stitch_masks != ALL_STITCH_MASKS:
            raise ValueError(
                "Every FVTQ payload must expose the 16 stitch masks in order 0..15"
            )
        object.__setattr__(self, "stitch_masks", stitch_masks)
        stitch_triangle_counts = tuple(
            tuple(
                _non_negative_integer(value, f"stitch_triangle_counts[{lod}][{mask}]")
                for mask, value in enumerate(counts)
            )
            for lod, counts in enumerate(self.stitch_triangle_counts)
        )
        if len(stitch_triangle_counts) != 3 or any(
            len(counts) != len(ALL_STITCH_MASKS) for counts in stitch_triangle_counts
        ):
            raise ValueError(
                "stitch_triangle_counts must contain 16 counts for each of 3 LODs"
            )
        if any(
            counts[0] != self.costs[lod].triangles
            for lod, counts in enumerate(stitch_triangle_counts)
        ):
            raise ValueError(
                "Mask 0 triangle counts must match the base resource costs"
            )
        object.__setattr__(self, "stitch_triangle_counts", stitch_triangle_counts)
        width = self.bounds.maximum[0] - self.bounds.minimum[0]
        height = self.bounds.maximum[1] - self.bounds.minimum[1]
        if not math.isclose(width, TILE_SIZE_M, abs_tol=1.0e-6):
            raise ValueError("Terrain tile width must be exactly 500 m")
        if not math.isclose(height, TILE_SIZE_M, abs_tol=1.0e-6):
            raise ValueError("Terrain tile height must be exactly 500 m")
        expected_west = self.grid_x * TILE_SIZE_M
        expected_south = self.grid_y * TILE_SIZE_M
        if not math.isclose(
            self.bounds.minimum[0], expected_west, abs_tol=1.0e-6
        ) or not math.isclose(self.bounds.minimum[1], expected_south, abs_tol=1.0e-6):
            raise ValueError(
                "Terrain tile bounds must match the global Lambert-93 500 m grid"
            )

    def cost(self, lod: int, stitch_mask: int = 0) -> ResourceCost:
        if lod not in {LOD0, LOD1, LOD2}:
            raise ValueError("lod must be 0, 1, or 2")
        if stitch_mask not in self.stitch_masks:
            raise ValueError("stitch_mask is not exposed by this FVTQ payload")
        base = self.costs[lod]
        triangle_count = self.stitch_triangle_counts[lod][stitch_mask]
        gpu_bytes = base.gpu_bytes + 12 * (triangle_count - base.triangles)
        if gpu_bytes < 0:
            raise ValueError("A stitch variant cannot have a negative GPU byte cost")
        return ResourceCost(
            cpu_bytes=base.cpu_bytes,
            gpu_bytes=gpu_bytes,
            triangles=triangle_count,
        )

    def expected_sha256(self, lod: int) -> str:
        if lod not in {LOD0, LOD1, LOD2}:
            raise ValueError("lod must be 0, 1, or 2")
        return self.payload_sha256[lod]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TerrainTile":
        bounds = payload["bounds_l93_ngf_m"]
        if len(bounds) != 6:
            raise ValueError("bounds_l93_ngf_m must contain six values")
        cost_payload = payload["resource_costs"]
        costs: list[ResourceCost] = []
        for lod in (LOD0, LOD1, LOD2):
            item = cost_payload[f"lod{lod}"]
            costs.append(
                ResourceCost(
                    cpu_bytes=item["cpu_bytes"],
                    gpu_bytes=item["gpu_bytes"],
                    triangles=item["triangles"],
                )
            )
        return cls(
            tile_id=str(payload["id"]),
            grid_x=payload["grid_x"],
            grid_y=payload["grid_y"],
            bounds=Aabb3D(
                minimum=_vec3((bounds[0], bounds[1], bounds[2]), "bounds minimum"),
                maximum=_vec3((bounds[3], bounds[4], bounds[5]), "bounds maximum"),
            ),
            costs=tuple(costs),  # type: ignore[arg-type]
            build_id=payload["build_id"],
            payload_sha256=tuple(
                cost_payload[f"lod{lod}"]["sha256"] for lod in (LOD0, LOD1, LOD2)
            ),  # type: ignore[arg-type]
            stitch_masks=tuple(payload["stitch_masks"]),
            stitch_triangle_counts=tuple(
                tuple(cost_payload[f"lod{lod}"]["stitch_triangle_counts"])
                for lod in (LOD0, LOD1, LOD2)
            ),
        )


class TerrainTileCatalog:
    """Validated tile lookup with deterministic grid neighbourhoods."""

    def __init__(self, tiles: Iterable[TerrainTile]) -> None:
        ordered = tuple(sorted(tiles, key=lambda tile: tile.tile_id))
        if not ordered:
            raise ValueError("Terrain tile catalog cannot be empty")
        identifiers = [tile.tile_id for tile in ordered]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Terrain tile identifiers must be unique")
        indices = [(tile.grid_x, tile.grid_y) for tile in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError("Terrain tile grid coordinates must be unique")
        build_ids = {tile.build_id for tile in ordered}
        if len(build_ids) != 1:
            raise ValueError("A terrain tile catalog must contain exactly one build_id")
        self._tiles = ordered
        self._by_id = {tile.tile_id: tile for tile in ordered}
        self._by_grid = {(tile.grid_x, tile.grid_y): tile for tile in ordered}
        self._build_id = next(iter(build_ids))

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "TerrainTileCatalog":
        return cls(TerrainTile.from_mapping(item) for item in manifest["tiles"])

    @property
    def tiles(self) -> tuple[TerrainTile, ...]:
        return self._tiles

    @property
    def tile_ids(self) -> tuple[str, ...]:
        return tuple(tile.tile_id for tile in self._tiles)

    @property
    def build_id(self) -> str:
        return self._build_id

    def tile(self, tile_id: str) -> TerrainTile:
        try:
            return self._by_id[tile_id]
        except KeyError as error:
            raise ValueError(f"Unknown terrain tile {tile_id!r}") from error

    def validate_payload(self, payload: PayloadRef) -> TerrainTile:
        tile = self.tile(payload.tile_id)
        if payload.stitch_mask not in tile.stitch_masks:
            raise ValueError(
                f"Tile {payload.tile_id!r} does not expose stitch mask "
                f"{payload.stitch_mask}"
            )
        return tile

    def neighbour(self, tile_id: str, edge: str) -> TerrainTile | None:
        if edge not in _EDGE_OFFSETS:
            raise ValueError(f"Unknown terrain edge {edge!r}")
        tile = self.tile(tile_id)
        offset_x, offset_y = _EDGE_OFFSETS[edge]
        return self._by_grid.get((tile.grid_x + offset_x, tile.grid_y + offset_y))

    def expand_rings(self, tile_ids: Iterable[str], ring_count: int) -> set[str]:
        ring_count = _non_negative_integer(ring_count, "ring_count")
        indices = {
            (self.tile(tile_id).grid_x, self.tile(tile_id).grid_y)
            for tile_id in tile_ids
        }
        if not indices:
            return set()
        expanded: set[str] = set()
        for grid_x, grid_y in indices:
            for offset_x in range(-ring_count, ring_count + 1):
                for offset_y in range(-ring_count, ring_count + 1):
                    tile = self._by_grid.get((grid_x + offset_x, grid_y + offset_y))
                    if tile is not None:
                        expanded.add(tile.tile_id)
        return expanded


@dataclass(frozen=True)
class StreamingBudget:
    cpu_bytes: int
    gpu_bytes: int
    maximum_triangles: int | None = None
    reserve_fraction: float = DEFAULT_MEMORY_RESERVE_FRACTION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cpu_bytes", _positive_integer(self.cpu_bytes, "cpu_bytes")
        )
        object.__setattr__(
            self, "gpu_bytes", _positive_integer(self.gpu_bytes, "gpu_bytes")
        )
        if self.maximum_triangles is not None:
            object.__setattr__(
                self,
                "maximum_triangles",
                _positive_integer(self.maximum_triangles, "maximum_triangles"),
            )
        reserve = _finite_number(self.reserve_fraction, "reserve_fraction")
        if not MINIMUM_MEMORY_RESERVE_FRACTION <= reserve < 1.0:
            raise ValueError(
                f"reserve_fraction must be in [{MINIMUM_MEMORY_RESERVE_FRACTION}, 1)"
            )
        object.__setattr__(self, "reserve_fraction", reserve)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_bytes": self.cpu_bytes,
            "gpu_bytes": self.gpu_bytes,
            "maximum_triangles": self.maximum_triangles,
            "reserve_fraction": self.reserve_fraction,
        }


@dataclass(frozen=True)
class BudgetReport:
    categories: Mapping[str, ResourceCost]
    unreserved: ResourceCost
    required_with_reserve: ResourceCost
    budget: StreamingBudget

    @property
    def within_budget(self) -> bool:
        triangles_ok = (
            self.budget.maximum_triangles is None
            or self.unreserved.triangles <= self.budget.maximum_triangles
        )
        return (
            self.required_with_reserve.cpu_bytes <= self.budget.cpu_bytes
            and self.required_with_reserve.gpu_bytes <= self.budget.gpu_bytes
            and triangles_ok
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": {
                name: cost.to_dict() for name, cost in sorted(self.categories.items())
            },
            "unreserved": self.unreserved.to_dict(),
            "required_with_reserve": self.required_with_reserve.to_dict(),
            "reserve_fraction": self.budget.reserve_fraction,
            "limits": {
                "cpu_bytes": self.budget.cpu_bytes,
                "gpu_bytes": self.budget.gpu_bytes,
                "maximum_triangles": self.budget.maximum_triangles,
            },
            "within_budget": self.within_budget,
        }


class StreamingBudgetExceeded(RuntimeError):
    """Raised before loading when a complete camera plan cannot fit."""

    def __init__(self, report: BudgetReport, context: str) -> None:
        self.report = report
        self.context = context
        super().__init__(
            f"Streaming budget exceeded for {context}: requires "
            f"{report.required_with_reserve.cpu_bytes} CPU bytes and "
            f"{report.required_with_reserve.gpu_bytes} GPU bytes including reserve"
        )


@dataclass(frozen=True)
class ResidencySets:
    """Frustum-derived roles. LOD0 sets may overlap by intent."""

    visible_lod0: tuple[str, ...]
    guard_lod0: tuple[str, ...]
    staging_lod0: tuple[str, ...]
    resident_lod1: tuple[str, ...]
    resident_lod2: tuple[str, ...]

    @property
    def all_lod0(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self.visible_lod0) | set(self.guard_lod0) | set(self.staging_lod0)
            )
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "visible_lod0": list(self.visible_lod0),
            "guard_lod0": list(self.guard_lod0),
            "staging_lod0": list(self.staging_lod0),
            "resident_lod1": list(self.resident_lod1),
            "resident_lod2": list(self.resident_lod2),
        }


@dataclass(frozen=True)
class ResidencyPlan:
    active_camera_id: str
    staging_camera_id: str | None
    sets: ResidencySets
    payloads: tuple[PayloadRef, ...]
    budget_report: BudgetReport

    @property
    def desired_payloads(self) -> frozenset[PayloadRef]:
        return frozenset(self.payloads)

    def payload_for(self, tile_id: str, lod: int) -> PayloadRef:
        matches = tuple(
            payload
            for payload in self.payloads
            if payload.tile_id == tile_id and payload.lod == lod
        )
        if len(matches) != 1:
            raise ValueError(
                f"Residency plan must contain exactly one payload for {tile_id} LOD{lod}"
            )
        return matches[0]

    def ordered_payloads(self) -> tuple[PayloadRef, ...]:
        """Return deterministic load priority for a settled active camera."""

        role_order = (
            (LOD0, self.sets.visible_lod0),
            (LOD0, self.sets.staging_lod0),
            (LOD0, self.sets.guard_lod0),
            (LOD1, self.sets.resident_lod1),
            (LOD2, self.sets.resident_lod2),
        )
        ordered: list[PayloadRef] = []
        seen: set[PayloadRef] = set()
        for lod, tile_ids in role_order:
            for tile_id in sorted(tile_ids):
                payload = self.payload_for(tile_id, lod)
                if payload not in seen:
                    ordered.append(payload)
                    seen.add(payload)
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_camera_id": self.active_camera_id,
            "staging_camera_id": self.staging_camera_id,
            "sets": self.sets.to_dict(),
            "payloads": [payload.to_dict() for payload in self.payloads],
            "budget": self.budget_report.to_dict(),
        }


def stitch_mask_for_neighbors(lod: int, neighbor_lods: Mapping[str, int | None]) -> int:
    """Return the FVTQ stitch mask using W1/E2/S4/N8.

    A bit is set only on the finer tile when the edge neighbour is exactly one
    LOD coarser. A larger delta is not representable by the 2:1 FVTQ variants
    and therefore fails closed.
    """

    if lod not in {LOD0, LOD1, LOD2}:
        raise ValueError("lod must be 0, 1, or 2")
    if set(neighbor_lods) != set(EDGE_ORDER):
        raise ValueError("neighbor_lods must define west, east, south, and north")
    mask = 0
    for edge in EDGE_ORDER:
        neighbor_lod = neighbor_lods[edge]
        if neighbor_lod is None:
            continue
        if neighbor_lod not in {LOD0, LOD1, LOD2}:
            raise ValueError(f"Invalid {edge} neighbour LOD")
        delta = neighbor_lod - lod
        if abs(delta) > 1:
            raise ValueError(f"Adjacent terrain LOD delta exceeds 1 on the {edge} edge")
        if delta == 1:
            mask |= STITCH_MASK_BITS[edge]
    return mask


def compute_stitch_masks(
    catalog: TerrainTileCatalog, lod_by_tile_id: Mapping[str, int]
) -> dict[str, int]:
    """Compute all deterministic FVTQ masks for one exhaustive assignment."""

    if set(lod_by_tile_id) != set(catalog.tile_ids):
        raise ValueError("LOD assignment must contain every catalog tile exactly once")
    masks: dict[str, int] = {}
    for tile_id in catalog.tile_ids:
        lod = lod_by_tile_id[tile_id]
        neighbor_lods = {
            edge: (
                lod_by_tile_id[neighbor.tile_id]
                if (neighbor := catalog.neighbour(tile_id, edge)) is not None
                else None
            )
            for edge in EDGE_ORDER
        }
        masks[tile_id] = stitch_mask_for_neighbors(lod, neighbor_lods)
    return masks


def _payloads_for_sets(
    catalog: TerrainTileCatalog, sets: ResidencySets
) -> tuple[PayloadRef, ...]:
    lod0 = set(sets.all_lod0)
    lod1 = set(sets.resident_lod1)
    lod2 = set(sets.resident_lod2)
    if lod0 & lod1 or lod0 & lod2 or lod1 & lod2:
        raise ValueError("LOD residency assignments must be disjoint")
    if lod0 | lod1 | lod2 != set(catalog.tile_ids):
        raise ValueError("LOD residency assignments must exhaust the tile catalog")
    assignment = {tile_id: LOD0 for tile_id in lod0}
    assignment.update({tile_id: LOD1 for tile_id in lod1})
    assignment.update({tile_id: LOD2 for tile_id in lod2})
    masks = compute_stitch_masks(catalog, assignment)
    payloads = tuple(
        sorted(
            PayloadRef(tile_id, lod, masks[tile_id])
            for tile_id, lod in assignment.items()
        )
    )
    for payload in payloads:
        catalog.validate_payload(payload)
    return payloads


def select_tiles_for_envelope(
    catalog: TerrainTileCatalog, envelope: CameraEnvelope
) -> tuple[str, ...]:
    """Select every tile whose 3D AABB intersects any envelope frustum."""

    frusta = tuple(Frustum.from_camera_view(view) for view in envelope.views)
    return tuple(
        tile.tile_id
        for tile in catalog.tiles
        if any(frustum.intersects_aabb(tile.bounds) for frustum in frusta)
    )


def _sum_costs(
    catalog: TerrainTileCatalog, payloads: Iterable[PayloadRef]
) -> ResourceCost:
    total = ZERO_COST
    for payload in sorted(set(payloads)):
        tile = catalog.validate_payload(payload)
        total = total + tile.cost(payload.lod, payload.stitch_mask)
    return total


def _budget_report(
    catalog: TerrainTileCatalog,
    budget: StreamingBudget,
    categories: Mapping[str, Iterable[PayloadRef]],
) -> BudgetReport:
    normalized: dict[str, ResourceCost] = {}
    claimed: set[PayloadRef] = set()
    for name, payloads in categories.items():
        unique = set(payloads) - claimed
        normalized[name] = _sum_costs(catalog, unique)
        claimed.update(unique)
    total = _sum_costs(catalog, claimed)
    factor = 1.0 + budget.reserve_fraction
    required = ResourceCost(
        cpu_bytes=math.ceil(total.cpu_bytes * factor),
        gpu_bytes=math.ceil(total.gpu_bytes * factor),
        triangles=total.triangles,
    )
    return BudgetReport(
        categories=normalized,
        unreserved=total,
        required_with_reserve=required,
        budget=budget,
    )


def budget_payloads(
    catalog: TerrainTileCatalog,
    budget: StreamingBudget,
    payloads: Iterable[PayloadRef],
    *,
    context: str,
) -> BudgetReport:
    report = _budget_report(catalog, budget, {context: payloads})
    if not report.within_budget:
        raise StreamingBudgetExceeded(report, context)
    return report


def plan_residency(
    catalog: TerrainTileCatalog,
    active_camera: CameraEnvelope,
    budget: StreamingBudget,
    *,
    staging_camera: CameraEnvelope | None = None,
) -> ResidencyPlan:
    """Build a complete, fail-closed active/staging residency plan."""

    visible = set(select_tiles_for_envelope(catalog, active_camera))
    staging = (
        set(select_tiles_for_envelope(catalog, staging_camera))
        if staging_camera is not None
        else set()
    )
    frustum_lod0 = visible | staging
    guard = catalog.expand_rings(frustum_lod0, GUARD_RING_COUNT) - frustum_lod0
    high_detail = visible | guard | staging
    lod1 = catalog.expand_rings(high_detail, LOD1_RING_COUNT) - high_detail
    lod2 = set(catalog.tile_ids) - high_detail - lod1
    sets = ResidencySets(
        visible_lod0=tuple(sorted(visible)),
        guard_lod0=tuple(sorted(guard)),
        staging_lod0=tuple(sorted(staging)),
        resident_lod1=tuple(sorted(lod1)),
        resident_lod2=tuple(sorted(lod2)),
    )
    payloads = _payloads_for_sets(catalog, sets)
    payload_by_key = {(payload.tile_id, payload.lod): payload for payload in payloads}
    # Precedence removes overlaps from accounting without hiding their semantic
    # role in the published sets.
    categories: dict[str, Iterable[PayloadRef]] = {
        "active_lod0": (
            payload_by_key[(tile_id, LOD0)] for tile_id in sets.visible_lod0
        ),
        "staging_lod0": (
            payload_by_key[(tile_id, LOD0)] for tile_id in sets.staging_lod0
        ),
        "guard_lod0": (payload_by_key[(tile_id, LOD0)] for tile_id in sets.guard_lod0),
        "lod1": (payload_by_key[(tile_id, LOD1)] for tile_id in sets.resident_lod1),
        "lod2": (payload_by_key[(tile_id, LOD2)] for tile_id in sets.resident_lod2),
    }
    report = _budget_report(catalog, budget, categories)
    if not report.within_budget:
        raise StreamingBudgetExceeded(report, active_camera.camera_id)
    return ResidencyPlan(
        active_camera_id=active_camera.camera_id,
        staging_camera_id=(staging_camera.camera_id if staging_camera else None),
        sets=sets,
        payloads=payloads,
        budget_report=report,
    )


@dataclass(frozen=True)
class CameraSequencePlan:
    terrain_build_id: str
    entries: tuple[ResidencyPlan, ...]
    camera_envelopes: tuple[CameraEnvelope, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terrain_build_id",
            _sha256_digest(self.terrain_build_id, "terrain_build_id"),
        )
        envelopes = tuple(self.camera_envelopes)
        identifiers = tuple(camera.camera_id for camera in envelopes)
        if not envelopes or len(identifiers) != len(set(identifiers)):
            raise ValueError("Camera envelope bindings must be non-empty and unique")
        if identifiers != tuple(entry.active_camera_id for entry in self.entries):
            raise ValueError("Camera envelope bindings must follow plan entry order")
        object.__setattr__(self, "camera_envelopes", envelopes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMERA_RESIDENCY_PLAN_SCHEMA,
            "terrain_build_id": self.terrain_build_id,
            "selection": "exact_convex_sat_aabb_frustum",
            "camera_envelope_bindings": [
                {
                    "camera_id": camera.camera_id,
                    "sha256": camera.sha256,
                    "envelope": camera.to_dict(),
                }
                for camera in self.camera_envelopes
            ],
            "entries": [entry.to_dict() for entry in self.entries],
        }


def plan_camera_sequence(
    catalog: TerrainTileCatalog,
    cameras: Sequence[CameraEnvelope],
    budget: StreamingBudget,
) -> CameraSequencePlan:
    """Precompute planned camera residency and next-camera preloading."""

    if not cameras:
        raise ValueError("At least one camera envelope is required")
    identifiers = [camera.camera_id for camera in cameras]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Planned camera identifiers must be unique")
    entries = tuple(
        plan_residency(
            catalog,
            active_camera,
            budget,
            staging_camera=(cameras[index + 1] if index + 1 < len(cameras) else None),
        )
        for index, active_camera in enumerate(cameras)
    )
    return CameraSequencePlan(
        terrain_build_id=catalog.build_id,
        entries=entries,
        camera_envelopes=tuple(cameras),
    )


def validate_camera_residency_plan(
    payload: Mapping[str, Any], catalog: TerrainTileCatalog
) -> CameraSequencePlan:
    """Semantically validate a public plan against its immutable tile catalog.

    JSON Schema validates shape. This validator deliberately recomputes the
    exhaustive LOD assignment, all 16-mask selections and every budget value;
    callers must run it before accepting a plan produced outside this process.
    """

    try:
        _require_exact_keys(
            payload,
            {
                "schema",
                "terrain_build_id",
                "selection",
                "camera_envelope_bindings",
                "entries",
            },
            "camera residency plan",
        )
        if payload["schema"] != CAMERA_RESIDENCY_PLAN_SCHEMA:
            raise ValueError("Unsupported camera residency plan schema")
        if payload["selection"] != "exact_convex_sat_aabb_frustum":
            raise ValueError("Unsupported camera residency selection algorithm")
        if (
            _sha256_digest(payload["terrain_build_id"], "terrain_build_id")
            != catalog.build_id
        ):
            raise ValueError("Camera residency plan build_id differs from the catalog")
        raw_bindings = payload["camera_envelope_bindings"]
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise ValueError("Camera residency plan requires envelope bindings")
        camera_envelopes: list[CameraEnvelope] = []
        for index, binding in enumerate(raw_bindings):
            if not isinstance(binding, Mapping):
                raise ValueError(f"camera_envelope_bindings[{index}] must be an object")
            _require_exact_keys(
                binding,
                {"camera_id", "sha256", "envelope"},
                f"camera_envelope_bindings[{index}]",
            )
            envelope = CameraEnvelope.from_mapping(binding["envelope"])
            if envelope.to_dict() != binding["envelope"]:
                raise ValueError("Camera envelope binding is not canonical")
            if binding["camera_id"] != envelope.camera_id:
                raise ValueError("Camera envelope binding id differs from envelope")
            if (
                _sha256_digest(binding["sha256"], "camera envelope sha256")
                != envelope.sha256
            ):
                raise ValueError(
                    "Camera envelope binding SHA-256 differs from envelope"
                )
            camera_envelopes.append(envelope)
        bound_ids = tuple(camera.camera_id for camera in camera_envelopes)
        if len(bound_ids) != len(set(bound_ids)):
            raise ValueError("Camera envelope binding ids must be unique")
        raw_entries = payload["entries"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("Camera residency plan requires at least one entry")
        if len(raw_entries) != len(camera_envelopes):
            raise ValueError("Every plan entry requires exactly one envelope binding")

        entries: list[ResidencyPlan] = []
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"entries[{index}] must be an object")
            _require_exact_keys(
                raw_entry,
                {
                    "active_camera_id",
                    "staging_camera_id",
                    "sets",
                    "payloads",
                    "budget",
                },
                f"entries[{index}]",
            )
            active_camera_id = str(raw_entry["active_camera_id"])
            if not active_camera_id:
                raise ValueError("active_camera_id must be non-empty")
            staging_raw = raw_entry["staging_camera_id"]
            staging_camera_id = None if staging_raw is None else str(staging_raw)
            if staging_camera_id == "":
                raise ValueError("staging_camera_id must be null or non-empty")
            raw_sets = raw_entry["sets"]
            _require_exact_keys(
                raw_sets,
                {
                    "visible_lod0",
                    "guard_lod0",
                    "staging_lod0",
                    "resident_lod1",
                    "resident_lod2",
                },
                f"entries[{index}].sets",
            )

            def ids(name: str) -> tuple[str, ...]:
                values = tuple(str(value) for value in raw_sets[name])
                if values != tuple(sorted(set(values))):
                    raise ValueError(f"{name} must be unique and canonically sorted")
                unknown = set(values) - set(catalog.tile_ids)
                if unknown:
                    raise ValueError(
                        f"{name} contains unknown tile ids: {sorted(unknown)!r}"
                    )
                return values

            sets = ResidencySets(
                visible_lod0=ids("visible_lod0"),
                guard_lod0=ids("guard_lod0"),
                staging_lod0=ids("staging_lod0"),
                resident_lod1=ids("resident_lod1"),
                resident_lod2=ids("resident_lod2"),
            )
            visible = set(sets.visible_lod0)
            staging = set(sets.staging_lod0)
            guard = set(sets.guard_lod0)
            lod1 = set(sets.resident_lod1)
            lod2 = set(sets.resident_lod2)
            if guard & (visible | staging):
                raise ValueError("guard_lod0 must be disjoint from frustum LOD0 sets")
            if (visible | staging | guard) & (lod1 | lod2) or lod1 & lod2:
                raise ValueError("LOD0, LOD1, and LOD2 assignments must be disjoint")
            if visible | staging | guard | lod1 | lod2 != set(catalog.tile_ids):
                raise ValueError("Residency sets must exhaust the tile catalog")

            expected_payloads = _payloads_for_sets(catalog, sets)
            serialized_payloads = tuple(
                PayloadRef.from_mapping(item) for item in raw_entry["payloads"]
            )
            if serialized_payloads != expected_payloads:
                raise ValueError(
                    "Plan payloads or stitch masks do not match the LOD assignment"
                )
            payload_by_key = {
                (item.tile_id, item.lod): item for item in expected_payloads
            }
            raw_budget = raw_entry["budget"]
            limits = raw_budget["limits"]
            budget = StreamingBudget(
                cpu_bytes=limits["cpu_bytes"],
                gpu_bytes=limits["gpu_bytes"],
                maximum_triangles=limits["maximum_triangles"],
                reserve_fraction=raw_budget["reserve_fraction"],
            )
            categories: dict[str, Iterable[PayloadRef]] = {
                "active_lod0": (
                    payload_by_key[(tile_id, LOD0)] for tile_id in sets.visible_lod0
                ),
                "staging_lod0": (
                    payload_by_key[(tile_id, LOD0)] for tile_id in sets.staging_lod0
                ),
                "guard_lod0": (
                    payload_by_key[(tile_id, LOD0)] for tile_id in sets.guard_lod0
                ),
                "lod1": (
                    payload_by_key[(tile_id, LOD1)] for tile_id in sets.resident_lod1
                ),
                "lod2": (
                    payload_by_key[(tile_id, LOD2)] for tile_id in sets.resident_lod2
                ),
            }
            report = _budget_report(catalog, budget, categories)
            if not report.within_budget:
                raise StreamingBudgetExceeded(report, active_camera_id)
            if raw_budget != report.to_dict():
                raise ValueError(
                    "Serialized residency budget does not match the catalog"
                )
            entries.append(
                ResidencyPlan(
                    active_camera_id=active_camera_id,
                    staging_camera_id=staging_camera_id,
                    sets=sets,
                    payloads=expected_payloads,
                    budget_report=report,
                )
            )
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed camera residency plan") from error

    active_ids = tuple(entry.active_camera_id for entry in entries)
    if active_ids != tuple(camera.camera_id for camera in camera_envelopes):
        raise ValueError("Plan entries do not follow the bound camera envelopes")
    if len(active_ids) != len(set(active_ids)):
        raise ValueError("Active camera ids must be unique")
    for index, entry in enumerate(entries):
        expected_staging = active_ids[index + 1] if index + 1 < len(entries) else None
        if entry.staging_camera_id != expected_staging:
            raise ValueError("Each plan entry must stage exactly the next camera")
        limits = entry.budget_report.budget
        expected_entry = plan_residency(
            catalog,
            camera_envelopes[index],
            limits,
            staging_camera=(
                camera_envelopes[index + 1]
                if index + 1 < len(camera_envelopes)
                else None
            ),
        )
        if entry.to_dict() != expected_entry.to_dict():
            raise ValueError(
                "Plan residency does not match its bound camera-envelope frusta"
            )
    return CameraSequencePlan(
        terrain_build_id=catalog.build_id,
        entries=tuple(entries),
        camera_envelopes=tuple(camera_envelopes),
    )


def predict_interactive_envelope(
    current_view: CameraView,
    *,
    camera_id: str,
    linear_velocity_mps: Sequence[float] = (0.0, 0.0, 0.0),
    angular_velocity_axis: Sequence[float] = (0.0, 0.0, 1.0),
    angular_velocity_deg_s: float = 0.0,
    horizon_seconds: float = DEFAULT_PREDICTION_HORIZON_SECONDS,
    interval_seconds: float = DEFAULT_PREDICTION_INTERVAL_SECONDS,
) -> CameraEnvelope:
    """Sample a two-second linear/angular prediction envelope every 250 ms."""

    velocity = _vec3(linear_velocity_mps, "linear_velocity_mps")
    angular_speed = _finite_number(angular_velocity_deg_s, "angular_velocity_deg_s")
    horizon = _finite_number(horizon_seconds, "horizon_seconds")
    interval = _finite_number(interval_seconds, "interval_seconds")
    if horizon <= 0.0 or interval <= 0.0 or interval > horizon:
        raise ValueError(
            "prediction interval must be positive and no greater than horizon"
        )
    axis = _vec3(angular_velocity_axis, "angular_velocity_axis")
    if abs(angular_speed) > _EPSILON:
        axis = _normalize(axis, "angular_velocity_axis")
    sample_count = int(math.floor(horizon / interval + _EPSILON))
    times = [index * interval for index in range(sample_count + 1)]
    if times[-1] < horizon - _EPSILON:
        times.append(horizon)
    views: list[CameraView] = []
    for index, time_seconds in enumerate(times):
        angle = math.radians(angular_speed * time_seconds)
        forward = (
            _rotate_axis_angle(current_view.forward, axis, angle)
            if abs(angle) > _EPSILON
            else current_view.forward
        )
        up = (
            _rotate_axis_angle(current_view.up, axis, angle)
            if abs(angle) > _EPSILON
            else current_view.up
        )
        views.append(
            CameraView(
                view_id=f"predicted-{index:02d}-{time_seconds:.3f}s",
                position_l93_ngf_m=_add(
                    current_view.position_l93_ngf_m,
                    _scale(velocity, time_seconds),
                ),
                forward=forward,
                up=up,
                vertical_fov_deg=current_view.vertical_fov_deg,
                aspect_ratio=current_view.aspect_ratio,
                near_clip_m=current_view.near_clip_m,
                far_clip_m=current_view.far_clip_m,
            )
        )
    return CameraEnvelope(camera_id=camera_id, views=tuple(views), mode="predicted")


def motion_requires_publication_hold(
    previous: CameraView,
    current: CameraView,
    *,
    elapsed_seconds: float,
    maximum_linear_speed_mps: float,
    maximum_angular_speed_deg_s: float,
) -> bool:
    """Detect a teleport/rapid rotation that must use staged publication."""

    elapsed = _finite_number(elapsed_seconds, "elapsed_seconds")
    linear_limit = _finite_number(maximum_linear_speed_mps, "maximum_linear_speed_mps")
    angular_limit = _finite_number(
        maximum_angular_speed_deg_s, "maximum_angular_speed_deg_s"
    )
    if elapsed <= 0.0 or linear_limit <= 0.0 or angular_limit <= 0.0:
        raise ValueError("motion thresholds and elapsed_seconds must be positive")
    distance = _length(
        _subtract(current.position_l93_ngf_m, previous.position_l93_ngf_m)
    )
    angular_distance = max(
        _angle_degrees(previous.forward, current.forward),
        _angle_degrees(previous.up, current.up),
    )
    return (
        distance / elapsed > linear_limit or angular_distance / elapsed > angular_limit
    )


@dataclass(frozen=True)
class FrustumStreamingAction:
    sequence: int
    generation: int
    kind: str
    payloads: tuple[PayloadRef, ...]
    camera_id: str | None
    visible_lod0_tile_ids: tuple[str, ...]
    reason: str
    issued_at_s: float
    expected_build_id: str | None
    expected_sha256: str | None

    @property
    def requires_commit(self) -> bool:
        return self.kind != NOOP

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrustumStreamingAction":
        _require_exact_keys(
            payload,
            {
                "sequence",
                "generation",
                "kind",
                "payloads",
                "camera_id",
                "visible_lod0_tile_ids",
                "reason",
                "issued_at_s",
                "expected_build_id",
                "expected_sha256",
            },
            "streaming action",
        )
        kind = str(payload["kind"])
        if kind not in {LOAD_PAYLOAD, EVICT_PAYLOADS, PUBLISH_CAMERA, NOOP}:
            raise ValueError("Unsupported streaming action kind")
        sequence = _positive_integer(payload["sequence"], "sequence")
        generation = _non_negative_integer(payload["generation"], "generation")
        action_payloads = tuple(
            PayloadRef.from_mapping(item) for item in payload["payloads"]
        )
        if action_payloads != tuple(sorted(set(action_payloads))):
            raise ValueError("Action payloads must be unique and canonically sorted")
        expected_build_id = payload["expected_build_id"]
        expected_sha256 = payload["expected_sha256"]
        if kind == LOAD_PAYLOAD:
            if len(action_payloads) != 1:
                raise ValueError("A load action must contain exactly one payload")
            expected_build_id = _sha256_digest(expected_build_id, "expected_build_id")
            expected_sha256 = _sha256_digest(expected_sha256, "expected_sha256")
        elif expected_build_id is not None or expected_sha256 is not None:
            raise ValueError("Only load actions may carry expected payload hashes")
        camera_id = None if payload["camera_id"] is None else str(payload["camera_id"])
        visible_ids = tuple(payload["visible_lod0_tile_ids"])
        if visible_ids != tuple(sorted(set(visible_ids))):
            raise ValueError("Action visible tile ids must be unique and sorted")
        if kind in {LOAD_PAYLOAD, EVICT_PAYLOADS}:
            if camera_id is not None or visible_ids:
                raise ValueError("Load/eviction actions cannot publish a camera")
        if kind == EVICT_PAYLOADS and not action_payloads:
            raise ValueError("An eviction action requires at least one payload")
        if kind == PUBLISH_CAMERA:
            if not camera_id or not action_payloads:
                raise ValueError("A publication requires a camera and LOD0 payloads")
            if any(ref.lod != LOD0 for ref in action_payloads):
                raise ValueError("A publication handshake may contain only LOD0")
        if kind == NOOP and (action_payloads or camera_id is not None or visible_ids):
            raise ValueError("A noop action cannot carry payload or camera state")
        return cls(
            sequence=sequence,
            generation=generation,
            kind=kind,
            payloads=action_payloads,
            camera_id=camera_id,
            visible_lod0_tile_ids=visible_ids,
            reason=str(payload["reason"]),
            issued_at_s=_finite_number(payload["issued_at_s"], "issued_at_s"),
            expected_build_id=expected_build_id,
            expected_sha256=expected_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "generation": self.generation,
            "kind": self.kind,
            "payloads": [payload.to_dict() for payload in self.payloads],
            "camera_id": self.camera_id,
            "visible_lod0_tile_ids": list(self.visible_lod0_tile_ids),
            "reason": self.reason,
            "issued_at_s": self.issued_at_s,
            "expected_build_id": self.expected_build_id,
            "expected_sha256": self.expected_sha256,
        }


@dataclass(frozen=True)
class FrustumStreamingState:
    terrain_build_id: str
    generation: int
    last_action_sequence: int
    active_camera: CameraEnvelope | None
    staging_camera: CameraEnvelope | None
    published_visible_lod0_tile_ids: tuple[str, ...]
    published_visible_lod0_payloads: tuple[PayloadRef, ...]
    resident_payloads: tuple[PayloadRef, ...]
    quarantined_payloads: tuple[PayloadRef, ...]
    failure_counts: Mapping[PayloadRef, int]
    retained_lod0_until_s: Mapping[PayloadRef, float]
    retry_backoff_remaining_ticks: int
    pending_action: FrustumStreamingAction | None
    last_error: str | None
    budget: StreamingBudget
    hysteresis_seconds: float
    maximum_load_failure_count: int
    load_retry_backoff_ticks: int
    telemetry_counters: Mapping[str, int]

    @property
    def active_camera_id(self) -> str | None:
        return self.active_camera.camera_id if self.active_camera else None

    @property
    def staging_camera_id(self) -> str | None:
        return self.staging_camera.camera_id if self.staging_camera else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STREAMING_STATE_SCHEMA,
            "terrain_build_id": self.terrain_build_id,
            "generation": self.generation,
            "last_action_sequence": self.last_action_sequence,
            "active_camera_id": self.active_camera_id,
            "staging_camera_id": self.staging_camera_id,
            "active_camera": (
                self.active_camera.to_dict() if self.active_camera else None
            ),
            "staging_camera": (
                self.staging_camera.to_dict() if self.staging_camera else None
            ),
            "published_visible_lod0_tile_ids": list(
                self.published_visible_lod0_tile_ids
            ),
            "published_visible_lod0_payloads": [
                payload.to_dict() for payload in self.published_visible_lod0_payloads
            ],
            "resident_payloads": [
                payload.to_dict() for payload in self.resident_payloads
            ],
            "quarantined_payloads": [
                payload.to_dict() for payload in self.quarantined_payloads
            ],
            "failure_counts": [
                {**payload.to_dict(), "count": count}
                for payload, count in sorted(self.failure_counts.items())
            ],
            "retained_lod0": [
                {
                    **payload.to_dict(),
                    "until_s": until_s,
                }
                for payload, until_s in sorted(self.retained_lod0_until_s.items())
            ],
            "pending_action": (
                self.pending_action.to_dict() if self.pending_action else None
            ),
            "retry_backoff_remaining_ticks": self.retry_backoff_remaining_ticks,
            "last_error": self.last_error,
            "budget": self.budget.to_dict(),
            "policy": {
                "hysteresis_seconds": self.hysteresis_seconds,
                "maximum_load_failure_count": self.maximum_load_failure_count,
                "load_retry_backoff_ticks": self.load_retry_backoff_ticks,
            },
            "telemetry_counters": dict(sorted(self.telemetry_counters.items())),
        }


@dataclass(frozen=True)
class FrustumStreamingTelemetry:
    tick_count: int
    staged_camera_count: int
    publication_count: int
    load_attempt_count: int
    load_success_count: int
    load_failure_count: int
    eviction_action_count: int
    evicted_payload_count: int
    action_failure_count: int
    quarantined_payload_count: int
    active_camera_id: str | None
    staging_camera_id: str | None
    resident_payload_count: int
    last_error: str | None


class FrustumStreamingPlanner:
    """Action/commit state machine for complete-camera LOD0 publication."""

    def __init__(
        self,
        catalog: TerrainTileCatalog,
        budget: StreamingBudget,
        *,
        active_camera: CameraEnvelope | None = None,
        resident_payloads: Iterable[PayloadRef] = (),
        published_visible_lod0_tile_ids: Iterable[str] = (),
        hysteresis_seconds: float = DEFAULT_HYSTERESIS_SECONDS,
        maximum_load_failure_count: int = DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT,
        load_retry_backoff_ticks: int = DEFAULT_LOAD_RETRY_BACKOFF_TICKS,
    ) -> None:
        self._catalog = catalog
        self._budget = budget
        self._hysteresis_seconds = _finite_number(
            hysteresis_seconds, "hysteresis_seconds"
        )
        if self._hysteresis_seconds < 0.0:
            raise ValueError("hysteresis_seconds must be non-negative")
        self._maximum_load_failure_count = _positive_integer(
            maximum_load_failure_count, "maximum_load_failure_count"
        )
        self._load_retry_backoff_ticks = _non_negative_integer(
            load_retry_backoff_ticks, "load_retry_backoff_ticks"
        )
        resident = set(resident_payloads)
        for payload in resident:
            catalog.validate_payload(payload)
        budget_payloads(
            catalog,
            budget,
            resident,
            context="restored resident payloads",
        )
        published = tuple(sorted(set(published_visible_lod0_tile_ids)))
        if active_camera is None and published:
            raise ValueError("Published LOD0 requires an active camera")
        if active_camera is not None:
            expected = select_tiles_for_envelope(catalog, active_camera)
            if published != expected:
                raise ValueError(
                    "Published LOD0 tile ids must exactly match the active camera frustum"
                )
            active_plan = plan_residency(catalog, active_camera, budget)
            required = {active_plan.payload_for(tile_id, LOD0) for tile_id in published}
            if not required <= resident:
                raise ValueError("Published LOD0 payloads must be resident")
            if any(
                payload.tile_id in published and payload.lod in {LOD1, LOD2}
                for payload in resident
            ):
                raise ValueError(
                    "A restored visible tile must not retain LOD1 or LOD2 payloads"
                )

        self._active_camera = active_camera
        self._active_plan_cache = active_plan if active_camera is not None else None
        self._staging_camera: CameraEnvelope | None = None
        self._transition_plan: ResidencyPlan | None = None
        self._staging_post_plan: ResidencyPlan | None = None
        self._resident = resident
        self._published = published
        self._published_payloads = tuple(sorted(required)) if active_camera else ()
        self._retained_until: dict[PayloadRef, float] = {}
        self._generation = 0
        self._sequence = 0
        self._pending: FrustumStreamingAction | None = None
        self._failure_counts: dict[PayloadRef, int] = {}
        self._quarantined: set[PayloadRef] = set()
        self._retry_backoff_remaining_ticks = 0
        self._last_error: str | None = None

        self._tick_count = 0
        self._staged_camera_count = 0
        self._publication_count = 0
        self._load_attempt_count = 0
        self._load_success_count = 0
        self._load_failure_count = 0
        self._eviction_action_count = 0
        self._evicted_payload_count = 0
        self._action_failure_count = 0

    @classmethod
    def from_state(
        cls,
        catalog: TerrainTileCatalog,
        budget: StreamingBudget,
        payload: Mapping[str, Any],
    ) -> "FrustumStreamingPlanner":
        """Restore a checkpoint only after full catalog and budget revalidation."""

        try:
            _require_exact_keys(
                payload,
                {
                    "schema",
                    "terrain_build_id",
                    "generation",
                    "last_action_sequence",
                    "active_camera_id",
                    "staging_camera_id",
                    "active_camera",
                    "staging_camera",
                    "published_visible_lod0_tile_ids",
                    "published_visible_lod0_payloads",
                    "resident_payloads",
                    "quarantined_payloads",
                    "failure_counts",
                    "retained_lod0",
                    "retry_backoff_remaining_ticks",
                    "pending_action",
                    "last_error",
                    "budget",
                    "policy",
                    "telemetry_counters",
                },
                "terrain streaming state",
            )
            if payload["schema"] != STREAMING_STATE_SCHEMA:
                raise ValueError("Unsupported terrain streaming state schema")
            build_id = _sha256_digest(payload["terrain_build_id"], "terrain_build_id")
            if build_id != catalog.build_id:
                raise ValueError("Streaming state build_id differs from the catalog")
            if payload["budget"] != budget.to_dict():
                raise ValueError(
                    "Streaming state budget differs from the runtime budget"
                )
            policy = payload["policy"]
            _require_exact_keys(
                policy,
                {
                    "hysteresis_seconds",
                    "maximum_load_failure_count",
                    "load_retry_backoff_ticks",
                },
                "streaming policy",
            )
            active_camera = (
                None
                if payload["active_camera"] is None
                else CameraEnvelope.from_mapping(payload["active_camera"])
            )
            staging_camera = (
                None
                if payload["staging_camera"] is None
                else CameraEnvelope.from_mapping(payload["staging_camera"])
            )
            if (
                active_camera is not None
                and active_camera.to_dict() != payload["active_camera"]
            ):
                raise ValueError("active_camera is not canonical")
            if (
                staging_camera is not None
                and staging_camera.to_dict() != payload["staging_camera"]
            ):
                raise ValueError("staging_camera is not canonical")
            if payload["active_camera_id"] != (
                active_camera.camera_id if active_camera else None
            ):
                raise ValueError("active_camera_id differs from active_camera")
            if payload["staging_camera_id"] != (
                staging_camera.camera_id if staging_camera else None
            ):
                raise ValueError("staging_camera_id differs from staging_camera")

            def payload_tuple(name: str) -> tuple[PayloadRef, ...]:
                refs = tuple(PayloadRef.from_mapping(item) for item in payload[name])
                if refs != tuple(sorted(set(refs))):
                    raise ValueError(f"{name} must be unique and canonically sorted")
                for ref in refs:
                    catalog.validate_payload(ref)
                return refs

            resident = payload_tuple("resident_payloads")
            quarantined = payload_tuple("quarantined_payloads")
            published_payloads = payload_tuple("published_visible_lod0_payloads")
            published_ids = tuple(payload["published_visible_lod0_tile_ids"])
            if published_ids != tuple(sorted(set(published_ids))):
                raise ValueError("Published tile ids must be unique and sorted")
            if tuple(ref.tile_id for ref in published_payloads) != published_ids or any(
                ref.lod != LOD0 for ref in published_payloads
            ):
                raise ValueError(
                    "Published payload refs must exactly bind the published LOD0 ids"
                )

            failure_counts: dict[PayloadRef, int] = {}
            for item in payload["failure_counts"]:
                ref = PayloadRef.from_mapping(item)
                catalog.validate_payload(ref)
                if ref in failure_counts:
                    raise ValueError("Failure count payloads must be unique")
                failure_counts[ref] = _positive_integer(item["count"], "count")
            retained: dict[PayloadRef, float] = {}
            for item in payload["retained_lod0"]:
                ref = PayloadRef.from_mapping(item)
                catalog.validate_payload(ref)
                if ref.lod != LOD0 or ref in retained:
                    raise ValueError("Retained payloads must be unique LOD0 refs")
                retained[ref] = _finite_number(item["until_s"], "until_s")

            generation = _non_negative_integer(payload["generation"], "generation")
            sequence = _non_negative_integer(
                payload["last_action_sequence"], "last_action_sequence"
            )
            retry_backoff = _non_negative_integer(
                payload["retry_backoff_remaining_ticks"],
                "retry_backoff_remaining_ticks",
            )
            pending = (
                None
                if payload["pending_action"] is None
                else FrustumStreamingAction.from_mapping(payload["pending_action"])
            )
            counters = {
                name: _non_negative_integer(value, f"telemetry_counters.{name}")
                for name, value in payload["telemetry_counters"].items()
            }
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed terrain streaming state") from error

        if set(quarantined) & set(resident):
            raise ValueError("A quarantined payload cannot be resident")
        if not set(retained) <= set(resident):
            raise ValueError("Retained LOD0 payloads must be resident")
        if set(published_payloads) - set(resident):
            raise ValueError("Published LOD0 payloads must be resident")
        if active_camera is None and (published_ids or published_payloads):
            raise ValueError("Published payloads require an active camera")
        if staging_camera is not None and active_camera is not None:
            if staging_camera.camera_id == active_camera.camera_id:
                raise ValueError("Active and staging cameras must have distinct ids")

        planner = cls(
            catalog,
            budget,
            active_camera=active_camera,
            resident_payloads=resident,
            published_visible_lod0_tile_ids=published_ids,
            hysteresis_seconds=policy["hysteresis_seconds"],
            maximum_load_failure_count=policy["maximum_load_failure_count"],
            load_retry_backoff_ticks=policy["load_retry_backoff_ticks"],
        )
        if planner._published_payloads != published_payloads:
            raise ValueError(
                "Published stitch masks differ from the active camera residency plan"
            )

        planner._generation = generation
        planner._sequence = sequence
        planner._failure_counts = failure_counts
        planner._quarantined = set(quarantined)
        planner._retained_until = retained
        planner._retry_backoff_remaining_ticks = retry_backoff
        planner._last_error = payload["last_error"]
        if pending is not None:
            if pending.generation != generation or pending.sequence != sequence:
                raise ValueError(
                    "Pending action generation/sequence differs from state"
                )
            for ref in pending.payloads:
                catalog.validate_payload(ref)
            if pending.kind == LOAD_PAYLOAD:
                ref = pending.payloads[0]
                tile = catalog.validate_payload(ref)
                if (
                    pending.expected_build_id != tile.build_id
                    or pending.expected_sha256 != tile.expected_sha256(ref.lod)
                ):
                    raise ValueError(
                        "Pending load integrity fields differ from catalog"
                    )
                if ref in planner._resident or ref in planner._quarantined:
                    raise ValueError(
                        "Pending load payload cannot be resident/quarantined"
                    )
            elif pending.kind == EVICT_PAYLOADS:
                if not set(pending.payloads) <= planner._resident:
                    raise ValueError("Pending eviction contains a non-resident payload")
            elif pending.kind == NOOP:
                raise ValueError("A noop action cannot be pending")
        planner._pending = pending

        if staging_camera is not None:
            if active_camera is None:
                transition_plan = plan_residency(catalog, staging_camera, budget)
                post_plan = transition_plan
            else:
                transition_plan = plan_residency(
                    catalog, active_camera, budget, staging_camera=staging_camera
                )
                post_plan = plan_residency(catalog, staging_camera, budget)
                budget_payloads(
                    catalog,
                    budget,
                    transition_plan.desired_payloads
                    | set(retained)
                    | set(published_payloads),
                    context=f"restored transition to {staging_camera.camera_id}",
                )
                budget_payloads(
                    catalog,
                    budget,
                    post_plan.desired_payloads
                    | set(retained)
                    | set(published_payloads),
                    context=f"restored post-publish {staging_camera.camera_id}",
                )
            planner._staging_camera = staging_camera
            planner._transition_plan = transition_plan
            planner._staging_post_plan = post_plan
        desired_plan = planner._transition_plan or planner._active_plan()
        if pending is not None and pending.kind == EVICT_PAYLOADS:
            if desired_plan is None:
                expected_obsolete = set(planner._resident)
            else:
                protected = set(published_payloads) | set(retained)
                expected_obsolete = (
                    planner._resident - desired_plan.desired_payloads - protected
                )
            if pending.payloads != tuple(sorted(expected_obsolete)):
                raise ValueError(
                    "Pending eviction is not the deterministic obsolete set"
                )
        if pending is not None and pending.kind == LOAD_PAYLOAD:
            if desired_plan is None:
                raise ValueError("Pending load requires an active or staged plan")
            protected = set(published_payloads) | set(retained)
            obsolete = planner._resident - desired_plan.desired_payloads - protected
            if obsolete:
                raise ValueError("Pending load cannot bypass obsolete payload eviction")
            if staging_camera is not None:
                load_order = tuple(
                    desired_plan.payload_for(tile_id, LOD0)
                    for tile_id in desired_plan.sets.all_lod0
                )
            else:
                load_order = desired_plan.ordered_payloads()
            missing = tuple(
                ref
                for ref in load_order
                if ref not in planner._resident and ref not in planner._quarantined
            )
            if not missing or pending.payloads != missing[:1]:
                raise ValueError("Pending load is not the next deterministic payload")
        if pending is not None and pending.kind == PUBLISH_CAMERA:
            if planner._staging_camera is None or planner._transition_plan is None:
                raise ValueError("Pending publication requires a staged camera")
            expected_ids = select_tiles_for_envelope(catalog, planner._staging_camera)
            required = tuple(
                sorted(
                    planner._transition_plan.payload_for(tile_id, LOD0)
                    for tile_id in planner._transition_plan.sets.all_lod0
                )
            )
            if (
                pending.camera_id != planner._staging_camera.camera_id
                or pending.visible_lod0_tile_ids != expected_ids
                or pending.payloads != required
                or not set(required) <= planner._resident
            ):
                raise ValueError("Pending publication is not fully resident/canonical")
            if any(
                ref.tile_id in expected_ids and ref.lod in {LOD1, LOD2}
                for ref in planner._resident
            ):
                raise ValueError(
                    "Pending publication forbids resident LOD1 or LOD2 on visible "
                    "staging tiles"
                )

        expected_counter_names = {
            "tick_count",
            "staged_camera_count",
            "publication_count",
            "load_attempt_count",
            "load_success_count",
            "load_failure_count",
            "eviction_action_count",
            "evicted_payload_count",
            "action_failure_count",
        }
        if set(counters) != expected_counter_names:
            raise ValueError("Streaming telemetry counter set is incomplete")
        if not set(quarantined) <= set(failure_counts):
            raise ValueError("Every quarantined payload needs a failure count")
        for name, value in counters.items():
            setattr(planner, f"_{name}", value)
        return planner

    @property
    def state(self) -> FrustumStreamingState:
        return FrustumStreamingState(
            terrain_build_id=self._catalog.build_id,
            generation=self._generation,
            last_action_sequence=self._sequence,
            active_camera=self._active_camera,
            staging_camera=self._staging_camera,
            published_visible_lod0_tile_ids=self._published,
            published_visible_lod0_payloads=self._published_payloads,
            resident_payloads=tuple(sorted(self._resident)),
            quarantined_payloads=tuple(sorted(self._quarantined)),
            failure_counts=dict(sorted(self._failure_counts.items())),
            retained_lod0_until_s=dict(sorted(self._retained_until.items())),
            retry_backoff_remaining_ticks=self._retry_backoff_remaining_ticks,
            pending_action=self._pending,
            last_error=self._last_error,
            budget=self._budget,
            hysteresis_seconds=self._hysteresis_seconds,
            maximum_load_failure_count=self._maximum_load_failure_count,
            load_retry_backoff_ticks=self._load_retry_backoff_ticks,
            telemetry_counters={
                "tick_count": self._tick_count,
                "staged_camera_count": self._staged_camera_count,
                "publication_count": self._publication_count,
                "load_attempt_count": self._load_attempt_count,
                "load_success_count": self._load_success_count,
                "load_failure_count": self._load_failure_count,
                "eviction_action_count": self._eviction_action_count,
                "evicted_payload_count": self._evicted_payload_count,
                "action_failure_count": self._action_failure_count,
            },
        )

    @property
    def telemetry(self) -> FrustumStreamingTelemetry:
        return FrustumStreamingTelemetry(
            tick_count=self._tick_count,
            staged_camera_count=self._staged_camera_count,
            publication_count=self._publication_count,
            load_attempt_count=self._load_attempt_count,
            load_success_count=self._load_success_count,
            load_failure_count=self._load_failure_count,
            eviction_action_count=self._eviction_action_count,
            evicted_payload_count=self._evicted_payload_count,
            action_failure_count=self._action_failure_count,
            quarantined_payload_count=len(self._quarantined),
            active_camera_id=(
                self._active_camera.camera_id if self._active_camera else None
            ),
            staging_camera_id=(
                self._staging_camera.camera_id if self._staging_camera else None
            ),
            resident_payload_count=len(self._resident),
            last_error=self._last_error,
        )

    def _action(
        self,
        kind: str,
        *,
        now_s: float,
        payloads: Iterable[PayloadRef] = (),
        camera_id: str | None = None,
        visible_lod0_tile_ids: Iterable[str] = (),
        reason: str,
    ) -> FrustumStreamingAction:
        self._sequence += 1
        ordered_payloads = tuple(sorted(payloads))
        expected_build_id: str | None = None
        expected_sha256: str | None = None
        if kind == LOAD_PAYLOAD:
            if len(ordered_payloads) != 1:
                raise RuntimeError("A load action must contain exactly one payload")
            tile = self._catalog.validate_payload(ordered_payloads[0])
            expected_build_id = tile.build_id
            expected_sha256 = tile.expected_sha256(ordered_payloads[0].lod)
        action = FrustumStreamingAction(
            sequence=self._sequence,
            generation=self._generation,
            kind=kind,
            payloads=ordered_payloads,
            camera_id=camera_id,
            visible_lod0_tile_ids=tuple(sorted(visible_lod0_tile_ids)),
            reason=reason,
            issued_at_s=now_s,
            expected_build_id=expected_build_id,
            expected_sha256=expected_sha256,
        )
        if action.requires_commit:
            self._pending = action
        return action

    def _noop(self, now_s: float, reason: str) -> FrustumStreamingAction:
        return self._action(NOOP, now_s=now_s, reason=reason)

    def stage_camera(self, camera: CameraEnvelope, *, now_s: float) -> ResidencyPlan:
        """Validate and stage a camera without altering active publication."""

        now = _finite_number(now_s, "now_s")
        if self._pending is not None:
            raise RuntimeError("Cannot stage a camera while an action is pending")
        if (
            self._active_camera is not None
            and camera.camera_id == self._active_camera.camera_id
        ):
            if camera == self._active_camera:
                raise ValueError("Camera is already active")
            raise ValueError("A camera revision must use a new camera_id")
        if (
            self._staging_camera is not None
            and camera.camera_id == self._staging_camera.camera_id
        ):
            if camera == self._staging_camera:
                return self._transition_plan  # type: ignore[return-value]
            raise ValueError("A staged camera revision must use a new camera_id")

        retained_at_request = {
            payload for payload, expiry in self._retained_until.items() if expiry > now
        }

        if self._active_camera is None:
            plan = plan_residency(self._catalog, camera, self._budget)
            post_plan = plan
        else:
            plan = plan_residency(
                self._catalog,
                self._active_camera,
                self._budget,
                staging_camera=camera,
            )
            budget_payloads(
                self._catalog,
                self._budget,
                plan.desired_payloads | retained_at_request,
                context=f"double-buffer transition to {camera.camera_id}",
            )
            post_plan = plan_residency(self._catalog, camera, self._budget)
            retained = set(self._published_payloads)
            budget_payloads(
                self._catalog,
                self._budget,
                post_plan.desired_payloads | retained | retained_at_request,
                context=f"post-publish hysteresis for {camera.camera_id}",
            )

        self._staging_camera = camera
        self._transition_plan = plan
        self._staging_post_plan = post_plan
        self._generation += 1
        self._staged_camera_count += 1
        self._failure_counts.clear()
        self._quarantined.clear()
        self._retry_backoff_remaining_ticks = 0
        self._last_error = None
        # Expired entries do not have to wait for a later tick merely because
        # a camera request arrived at their deadline.
        self._retained_until = {
            payload: expiry
            for payload, expiry in self._retained_until.items()
            if payload in retained_at_request
        }
        return plan

    def _active_plan(self) -> ResidencyPlan | None:
        return self._active_plan_cache

    def retry_quarantined(
        self, payloads: Iterable[PayloadRef] | None = None
    ) -> tuple[PayloadRef, ...]:
        """Start a new retry generation for explicitly quarantined payloads."""

        if self._pending is not None:
            raise RuntimeError("Cannot retry quarantine while an action is pending")
        selected = set(self._quarantined) if payloads is None else set(payloads)
        if not selected:
            raise ValueError("No quarantined payload was selected for retry")
        unknown = selected - self._quarantined
        if unknown:
            raise ValueError(
                "Retry selection contains payloads that are not quarantined: "
                f"{sorted(unknown)!r}"
            )
        self._quarantined.difference_update(selected)
        for payload in selected:
            self._failure_counts.pop(payload, None)
        self._retry_backoff_remaining_ticks = 0
        self._last_error = None
        self._generation += 1
        return tuple(sorted(selected))

    def tick(self, *, now_s: float) -> FrustumStreamingAction:
        """Emit one deterministic load, publish, eviction, or noop action."""

        now = _finite_number(now_s, "now_s")
        if self._pending is not None:
            raise RuntimeError("The pending streaming action must be committed")
        self._tick_count += 1

        if self._staging_camera is not None:
            if self._transition_plan is None:
                raise RuntimeError("Staged camera is missing its residency plan")
            desired_transition = self._transition_plan.desired_payloads
            protected = set(self._published_payloads)
            protected.update(
                payload
                for payload, expiry in self._retained_until.items()
                if expiry > now
            )
            obsolete = self._resident - desired_transition - protected
            if obsolete:
                self._eviction_action_count += 1
                return self._action(
                    EVICT_PAYLOADS,
                    now_s=now,
                    payloads=obsolete,
                    reason="remove_payloads_outside_double_buffer_plan",
                )

            staging_ids = select_tiles_for_envelope(self._catalog, self._staging_camera)
            required = tuple(
                self._transition_plan.payload_for(tile_id, LOD0)
                for tile_id in self._transition_plan.sets.all_lod0
            )
            missing = tuple(
                payload for payload in required if payload not in self._resident
            )
            quarantined = tuple(
                payload for payload in missing if payload in self._quarantined
            )
            if quarantined:
                return self._noop(now, "staging_contains_quarantined_lod0")
            if missing:
                if self._retry_backoff_remaining_ticks > 0:
                    self._retry_backoff_remaining_ticks -= 1
                    return self._noop(now, "load_retry_backoff")
                self._load_attempt_count += 1
                return self._action(
                    LOAD_PAYLOAD,
                    now_s=now,
                    payloads=missing[:1],
                    reason="load_staging_lod0_one_per_tick",
                )
            return self._action(
                PUBLISH_CAMERA,
                now_s=now,
                payloads=required,
                camera_id=self._staging_camera.camera_id,
                visible_lod0_tile_ids=staging_ids,
                reason="publish_complete_staging_lod0_atomically",
            )

        active_plan = self._active_plan()
        if active_plan is None:
            return self._noop(now, "no_active_or_staged_camera")
        desired = active_plan.desired_payloads
        expired = {
            payload for payload, expiry in self._retained_until.items() if expiry <= now
        }
        for payload in expired:
            self._retained_until.pop(payload, None)
        protected = set(self._retained_until) | set(self._published_payloads)
        obsolete = self._resident - desired - protected
        if obsolete:
            self._eviction_action_count += 1
            return self._action(
                EVICT_PAYLOADS,
                now_s=now,
                payloads=obsolete,
                reason="evict_obsolete_before_loading_active_residency",
            )
        missing_ordered = tuple(
            payload
            for payload in active_plan.ordered_payloads()
            if payload not in self._resident and payload not in self._quarantined
        )
        if missing_ordered:
            if self._retry_backoff_remaining_ticks > 0:
                self._retry_backoff_remaining_ticks -= 1
                return self._noop(now, "load_retry_backoff")
            self._load_attempt_count += 1
            return self._action(
                LOAD_PAYLOAD,
                now_s=now,
                payloads=missing_ordered[:1],
                reason="fill_active_guard_and_context_residency",
            )

        if self._quarantined:
            return self._noop(now, "active_residency_contains_quarantined_payload")
        if self._retained_until:
            return self._noop(now, "hysteresis_retains_previous_lod0")
        return self._noop(now, "stable_complete_residency")

    def commit(
        self,
        action: FrustumStreamingAction,
        *,
        succeeded: bool,
        error: str | None = None,
        observed_build_id: str | None = None,
        observed_sha256: str | None = None,
    ) -> None:
        """Acknowledge the exact pending executor action."""

        if self._pending is None:
            raise RuntimeError("There is no pending streaming action to commit")
        if action != self._pending:
            raise ValueError("Committed action does not match the pending action")
        integrity_failure = False
        if action.kind == LOAD_PAYLOAD and succeeded:
            try:
                observed_build = _sha256_digest(observed_build_id, "observed_build_id")
                observed_digest = _sha256_digest(observed_sha256, "observed_sha256")
            except ValueError as validation_error:
                succeeded = False
                integrity_failure = True
                error = str(validation_error)
            else:
                if (
                    observed_build != action.expected_build_id
                    or observed_digest != action.expected_sha256
                ):
                    succeeded = False
                    integrity_failure = True
                    error = (
                        "Loaded payload build_id or SHA-256 does not match the catalog"
                    )

        publication: (
            tuple[
                CameraEnvelope,
                ResidencyPlan,
                tuple[str, ...],
                set[PayloadRef],
                set[PayloadRef],
            ]
            | None
        ) = None
        if succeeded and action.kind == PUBLISH_CAMERA:
            staging_camera = self._staging_camera
            transition_plan = self._transition_plan
            post_plan = self._staging_post_plan
            if staging_camera is None:
                raise RuntimeError("No staged camera exists for publication")
            if action.camera_id != staging_camera.camera_id:
                raise RuntimeError("Publication camera does not match staging camera")
            expected = select_tiles_for_envelope(self._catalog, staging_camera)
            if action.visible_lod0_tile_ids != expected:
                raise RuntimeError("Publication does not match staged frustum coverage")
            if transition_plan is None:
                raise RuntimeError(
                    "Staged camera is missing its transition residency plan"
                )
            transition_required = {
                transition_plan.payload_for(tile_id, LOD0)
                for tile_id in transition_plan.sets.all_lod0
            }
            if set(action.payloads) != transition_required:
                raise RuntimeError(
                    "Publication payload handshake does not match staged LOD0"
                )
            if not transition_required <= self._resident:
                raise RuntimeError(
                    "Publication requires complete staged and guard LOD0 residency"
                )
            if any(
                payload.tile_id in expected and payload.lod in {LOD1, LOD2}
                for payload in self._resident
            ):
                raise RuntimeError("Publication forbids LOD1 or LOD2 on visible tiles")
            if post_plan is None:
                raise RuntimeError(
                    "Staged camera is missing its settled residency plan"
                )
            target_required = {
                transition_plan.payload_for(tile_id, LOD0) for tile_id in expected
            }
            publication = (
                staging_camera,
                post_plan,
                expected,
                transition_required,
                target_required,
            )

        # Clearing the handshake is the transaction boundary. All integrity
        # and publication preconditions above are read-only, so an exception
        # leaves the exact pending action available for retry or diagnosis.
        self._pending = None
        if not succeeded:
            self._action_failure_count += 1
            if action.kind == LOAD_PAYLOAD:
                self._load_failure_count += 1
                payload = action.payloads[0]
                failure_count = self._failure_counts.get(payload, 0) + 1
                self._failure_counts[payload] = failure_count
                if (
                    integrity_failure
                    or failure_count >= self._maximum_load_failure_count
                ):
                    self._quarantined.add(payload)
                    self._retry_backoff_remaining_ticks = 0
                else:
                    self._retry_backoff_remaining_ticks = self._load_retry_backoff_ticks
            self._last_error = error or f"{action.kind} failed"
            return

        self._last_error = None
        if action.kind == LOAD_PAYLOAD:
            if len(action.payloads) != 1:
                raise RuntimeError("A load action must contain exactly one payload")
            payload = action.payloads[0]
            self._resident.add(payload)
            self._failure_counts.pop(payload, None)
            self._quarantined.discard(payload)
            self._retry_backoff_remaining_ticks = 0
            self._load_success_count += 1
        elif action.kind == EVICT_PAYLOADS:
            self._resident.difference_update(action.payloads)
            for payload in action.payloads:
                self._retained_until.pop(payload, None)
            self._evicted_payload_count += len(action.payloads)
        elif action.kind == PUBLISH_CAMERA:
            if publication is None:
                raise AssertionError("Publication preconditions were not evaluated")
            staging_camera, post_plan, expected, _, target_required = publication
            old_visible = set(self._published_payloads)
            for payload in old_visible - target_required:
                self._retained_until[payload] = (
                    action.issued_at_s + self._hysteresis_seconds
                )
            self._active_camera = staging_camera
            self._active_plan_cache = post_plan
            self._staging_camera = None
            self._transition_plan = None
            self._staging_post_plan = None
            self._published = expected
            self._published_payloads = tuple(sorted(target_required))
            self._publication_count += 1
        else:
            raise RuntimeError(f"Unsupported streaming action: {action.kind}")


def load_streaming_contract(path: Path | str | None = None) -> dict[str, Any]:
    """Load and minimally validate the normative JSON contract."""

    contract_path = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("frustum_streaming_contract.v2.json")
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"Streaming contract schema must be {CONTRACT_SCHEMA!r}")
    expected = {
        ("tile", "size_m"): TILE_SIZE_M,
        ("residency", "guard_lod0_ring_count"): GUARD_RING_COUNT,
        ("residency", "guard_publication_gate"): ("complete_before_camera_publication"),
        ("residency", "lod1_additional_ring_count"): LOD1_RING_COUNT,
        ("transition", "maximum_load_failure_count"): (
            DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT
        ),
        ("transition", "hysteresis_seconds"): DEFAULT_HYSTERESIS_SECONDS,
        ("transition", "payload_integrity_gate"): (
            "catalog_build_id_and_sha256_must_match_observed_load"
        ),
        ("budget", "memory_reserve_fraction"): (DEFAULT_MEMORY_RESERVE_FRACTION),
        ("camera_residency_plan", "camera_binding"): (
            "canonical_camera_envelope_and_sha256_recomputed_before_acceptance"
        ),
        ("camera_residency_plan", "frustum_binding"): (
            "all_residency_sets_recomputed_from_bound_envelopes"
        ),
        ("resume", "pending_publication_revalidated_before_commit"): True,
        ("resume", "commit_preconditions_are_transactional"): True,
        ("resume", "lower_lod_on_visible_tile_permitted"): False,
    }
    for keys, value in expected.items():
        current: Any = payload
        for key in keys:
            current = current[key]
        if current != value:
            raise ValueError(f"Streaming contract {'.'.join(keys)} must be {value!r}")
    return payload


__all__ = [
    "Aabb3D",
    "BudgetReport",
    "CAMERA_ENVELOPE_SCHEMA",
    "CAMERA_RESIDENCY_PLAN_SCHEMA",
    "CONTRACT_SCHEMA",
    "CameraEnvelope",
    "CameraSequencePlan",
    "CameraView",
    "DEFAULT_HYSTERESIS_SECONDS",
    "DEFAULT_MAXIMUM_LOAD_FAILURE_COUNT",
    "EVICT_PAYLOADS",
    "Frustum",
    "FrustumStreamingAction",
    "FrustumStreamingPlanner",
    "FrustumStreamingState",
    "FrustumStreamingTelemetry",
    "GUARD_RING_COUNT",
    "LOAD_PAYLOAD",
    "LOD0",
    "LOD1",
    "LOD1_RING_COUNT",
    "LOD2",
    "MINIMUM_MEMORY_RESERVE_FRACTION",
    "NOOP",
    "PUBLISH_CAMERA",
    "STREAMING_STATE_SCHEMA",
    "PayloadRef",
    "ResidencyPlan",
    "ResidencySets",
    "ResourceCost",
    "StreamingBudget",
    "StreamingBudgetExceeded",
    "TILE_SIZE_M",
    "TerrainTile",
    "TerrainTileCatalog",
    "budget_payloads",
    "load_streaming_contract",
    "motion_requires_publication_hold",
    "plan_camera_sequence",
    "plan_residency",
    "predict_interactive_envelope",
    "select_tiles_for_envelope",
]
