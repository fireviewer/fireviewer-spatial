"""Deterministic placement guardrails shared by the optimized builder.

The policy never invents a geographic candidate. It only classifies measured
candidates, bounds visually rare building roles, and names local procedural
fallback prototypes when a catalogue asset cannot be used safely.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

SCHEMA = "fireviewer.asset-placement-resilience.v1"
RULE_VERSION = "fireviewer.asset-placement-resilience-rules.v1"
# A real asset is still scaled uniformly.  This only rejects a prototype whose
# proportions would under-fill one measured axis by more than 3.5x relative to
# another, while tolerating intentionally stylized but recognizable assets.
MAX_ASSET_SHAPE_LOG_SPREAD = math.log(3.5)

BUILDING_FORMS = {
    "low_rise_house",
    "mid_rise_residential",
    "multi_storey_residential",
}
SPECIAL_BUILDING_ROLES = (
    "fuel_station",
    "religious",
    "public_service",
    "industrial",
    "agricultural",
    "commercial",
)


def building_form(*, height_m: float, footprint_area_m2: float) -> str:
    """Classify a building strictly from measured height and footprint area."""

    height = float(height_m)
    area = float(footprint_area_m2)
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("building height must be finite and positive")
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("building footprint area must be finite and positive")
    if height >= 13.0 or (height >= 10.0 and area >= 250.0):
        return "multi_storey_residential"
    if height >= 8.0 or area >= 180.0:
        return "mid_rise_residential"
    return "low_rise_house"


def special_building_role(
    *, semantic_tags: Iterable[str], reference_terms: Iterable[str]
) -> str | None:
    """Return a precise rare role; a fuel station requires explicit evidence."""

    tags = set(semantic_tags)
    terms = set(reference_terms)
    if "commercial" in tags and terms.intersection({"station", "carburant", "essence"}):
        return "fuel_station"
    for role in SPECIAL_BUILDING_ROLES[1:]:
        if role in tags:
            return role
    return None


def special_role_limit(role: str, *, total_buildings: int) -> int:
    """Bound visual repetition without changing the measured building count."""

    total = max(0, int(total_buildings))
    if role == "fuel_station":
        return 1
    policies = {
        "religious": (2, 0.005, 8),
        "public_service": (3, 0.01, 12),
        "industrial": (4, 0.02, 32),
        "agricultural": (6, 0.04, 64),
        "commercial": (6, 0.03, 48),
    }
    if role not in policies:
        raise ValueError(f"unknown special building role: {role}")
    minimum, ratio, maximum = policies[role]
    return min(maximum, max(minimum, math.ceil(total * ratio)))


def asset_repeat_limit(*, candidate_count: int, compatible_asset_count: int) -> int:
    """Bound one prototype to at most twice its balanced deterministic share."""

    candidates = max(0, int(candidate_count))
    assets = max(1, int(compatible_asset_count))
    return max(8, math.ceil(candidates / assets) * 2)


def asset_shape_log_spread(
    *, native_dimensions_m: Iterable[float], measured_dimensions_m: Iterable[float]
) -> float:
    native = tuple(float(value) for value in native_dimensions_m)
    measured = tuple(float(value) for value in measured_dimensions_m)
    if len(native) != 3 or len(measured) != 3:
        raise ValueError("asset shape comparison requires three dimensions")
    if any(not math.isfinite(value) or value <= 0.0 for value in (*native, *measured)):
        raise ValueError("asset shape dimensions must be finite and positive")
    native_width, native_depth = sorted((native[0], native[2]), reverse=True)
    measured_width, measured_depth = sorted((measured[0], measured[2]), reverse=True)
    ratios = (
        measured_width / native_width,
        measured[1] / native[1],
        measured_depth / native_depth,
    )
    logs = tuple(math.log(value) for value in ratios)
    return max(logs) - min(logs)


def asset_shape_is_compatible(
    *, native_dimensions_m: Iterable[float], measured_dimensions_m: Iterable[float]
) -> bool:
    return asset_shape_log_spread(
        native_dimensions_m=native_dimensions_m,
        measured_dimensions_m=measured_dimensions_m,
    ) <= MAX_ASSET_SHAPE_LOG_SPREAD


def procedural_asset_id(
    *, family: str, building_form_name: str | None = None, semantic_tags: Iterable[str] = ()
) -> str:
    """Name one small immutable procedural prototype for a safe fallback."""

    if family == "buildings":
        if building_form_name not in BUILDING_FORMS:
            raise ValueError("procedural building fallback requires a building form")
        return f"procedural-building-{building_form_name.replace('_', '-')}"
    if family == "trees":
        tags = set(semantic_tags)
        form = "conifer" if "conifer" in tags and "broadleaf" not in tags else "broadleaf"
        return f"procedural-tree-{form}"
    raise ValueError(f"no procedural fallback for family: {family}")


def procedural_selection_seed(*, zone_id: str, candidate_id: str, asset_id: str) -> int:
    basis = f"{SCHEMA}\x00{RULE_VERSION}\x00{zone_id}\x00{candidate_id}\x00{asset_id}"
    return int.from_bytes(hashlib.sha256(basis.encode("utf-8")).digest()[:8], "big")


__all__ = [
    "BUILDING_FORMS",
    "MAX_ASSET_SHAPE_LOG_SPREAD",
    "RULE_VERSION",
    "SCHEMA",
    "SPECIAL_BUILDING_ROLES",
    "asset_repeat_limit",
    "asset_shape_is_compatible",
    "asset_shape_log_spread",
    "building_form",
    "procedural_asset_id",
    "procedural_selection_seed",
    "special_building_role",
    "special_role_limit",
]
