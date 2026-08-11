"""Deterministic profile selection for lightweight FireViewer ground surfaces."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Any


CATALOG_SCHEMA = "fireviewer.ground-surface-atlas-library.v3"
DEFAULT_LINEAR_SEGMENT_LENGTH_M = 250.0
CLIFF_GEOLOGY_VARIANTS = {
    "limestone_strata": ("calcaire", "limestone"),
    "karst_limestone": ("karst", "karstique"),
    "layered_schist": ("schiste", "schist"),
    "blocky_granite": ("granite", "granit"),
    "warm_sandstone": ("gres", "sandstone"),
    "conglomerate": ("conglomerat", "conglomerate", "poudingue"),
    "eroded_marl": ("marne", "marl"),
    "dark_basalt": ("basalte", "basalt", "volcanique"),
}


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _rank(seed: str, profile_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{profile_id}".encode("utf-8")).digest()


def _profiles(
    catalog: dict[str, Any], *, application_mode: str, family: str | None = None
) -> list[dict[str, Any]]:
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise ValueError("Unsupported ground surface atlas catalog")
    profiles = [
        profile
        for profile in catalog.get("profiles", [])
        if profile.get("application_mode") == application_mode
        and (family is None or profile.get("family") == family)
    ]
    if not profiles:
        suffix = "" if family is None else f" and family {family}"
        raise ValueError(f"No profile for application mode {application_mode}{suffix}")
    return profiles


def select_profile(
    catalog: dict[str, Any],
    *,
    application_mode: str,
    seed: str,
    family: str | None = None,
    excluded_profile_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    excluded = set(excluded_profile_ids)
    candidates = [
        profile
        for profile in _profiles(catalog, application_mode=application_mode, family=family)
        if profile["id"] not in excluded
    ]
    if not candidates:
        raise ValueError("Profile anti-repetition exclusions exhausted the candidate pool")
    return min(candidates, key=lambda profile: (_rank(seed, profile["id"]), profile["id"]))


def compose_linear_schedule(
    catalog: dict[str, Any],
    *,
    fire_id: str,
    feature_id: str,
    length_m: float,
    family: str,
    segment_length_m: float = DEFAULT_LINEAR_SEGMENT_LENGTH_M,
) -> list[dict[str, Any]]:
    if not fire_id or not feature_id:
        raise ValueError("fire_id and feature_id are required")
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("Linear feature length must be finite and positive")
    if not math.isfinite(segment_length_m) or segment_length_m <= 0:
        raise ValueError("Linear segment length must be finite and positive")
    pool = _profiles(catalog, application_mode="linear_overlay", family=family)
    if len(pool) < 3:
        raise ValueError("Linear anti-repetition requires at least three family profiles")
    segments = []
    previous: list[str] = []
    segment_count = math.ceil(length_m / segment_length_m)
    catalog_hash = str(catalog.get("catalog_sha256", "unsigned"))
    for index in range(segment_count):
        start = index * segment_length_m
        end = min(length_m, (index + 1) * segment_length_m)
        profile = select_profile(
            catalog,
            application_mode="linear_overlay",
            family=family,
            seed=f"{fire_id}:{feature_id}:{index}:{catalog_hash}",
            excluded_profile_ids=tuple(previous[-2:]),
        )
        segments.append(
            {
                "index": index,
                "start_m": start,
                "end_m": end,
                "profile_id": profile["id"],
                "uv_origin_m": start,
                "uv_direction": "feature_tangent",
            }
        )
        previous.append(profile["id"])
    return segments


def compose_watercourse_schedule(
    catalog: dict[str, Any],
    *,
    fire_id: str,
    feature_id: str,
    length_m: float,
    segment_length_m: float = DEFAULT_LINEAR_SEGMENT_LENGTH_M,
) -> list[dict[str, Any]]:
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("Watercourse length must be finite and positive")
    profiles = _profiles(catalog, application_mode="watercourse_overlay")
    if len(profiles) < 3:
        raise ValueError("Watercourse anti-repetition requires at least three profiles")
    result = []
    previous: list[str] = []
    catalog_hash = str(catalog.get("catalog_sha256", "unsigned"))
    for index in range(math.ceil(length_m / segment_length_m)):
        start = index * segment_length_m
        end = min(length_m, (index + 1) * segment_length_m)
        profile = select_profile(
            catalog,
            application_mode="watercourse_overlay",
            seed=f"{fire_id}:{feature_id}:water:{index}:{catalog_hash}",
            excluded_profile_ids=tuple(previous[-2:]),
        )
        result.append(
            {
                "index": index,
                "start_m": start,
                "end_m": end,
                "profile_id": profile["id"],
                "uv_origin_m": start,
                "uv_direction": "downstream_tangent",
                "bank_derivation": "left_right_from_width_and_flow_direction",
            }
        )
        previous.append(profile["id"])
    return result


def select_cliff_profile(
    catalog: dict[str, Any],
    *,
    geology: str,
    fire_id: str,
    tile_id: str,
) -> dict[str, Any]:
    normalized = _normalize(geology)
    matching_variants = [
        variant
        for variant, keywords in CLIFF_GEOLOGY_VARIANTS.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    if not matching_variants:
        raise ValueError(f"Unsupported cliff geology: {geology}")
    candidates = [
        profile
        for profile in _profiles(
            catalog,
            application_mode="slope_cliff_overlay",
            family="cliff_surface",
        )
        if profile.get("variant") in matching_variants
    ]
    if not candidates:
        raise ValueError(f"No cliff profile available for geology: {geology}")
    seed = f"{fire_id}:{tile_id}:{normalized}:{catalog.get('catalog_sha256', 'unsigned')}"
    return min(candidates, key=lambda profile: (_rank(seed, profile["id"]), profile["id"]))
