"""Validate and apply the semantic context contract for 72 ground profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any, Mapping


CONTEXT_SCHEMA = "fireviewer.ground-context-contract.v1"
RUNTIME_SCHEMA = "fireviewer.ground-surface-runtime-contract.v3"
ATLAS_SCHEMA = "fireviewer.ground-surface-atlas-library.v3"
REQUIRED_SOURCE_IDS = {
    "land_parcels",
    "agricultural_parcels",
    "roads",
    "railways",
    "hydro_lines",
    "hydro_surfaces",
    "landcover",
    "geology",
}
DERIVED_SOURCE_IDS = {
    "incident_burn_severity",
    "MNT_slope",
    "MNT_aspect",
}
REQUIRED_INCIDENTS = {
    "FR-26-00001",
    "FR-30-00001",
    "FR-34-00001",
    "FR-66-00001",
    "FR-77-00001",
    "FR-83-00001",
}


def _normalize(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )


def load_context_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTEXT_SCHEMA:
        raise ValueError("Unsupported ground context contract")
    if payload.get("crs") != "EPSG:2154":
        raise ValueError("Ground context must use EPSG:2154")
    if payload.get("orthophoto_dependency") != "forbidden":
        raise ValueError("Ground context must forbid orthophotos")
    if payload.get("zero_match_policy") != "fail_closed":
        raise ValueError("Ground context selection must fail closed")
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Ground context sources are absent")
    source_ids = [str(source.get("id", "")) for source in sources]
    if set(source_ids) != REQUIRED_SOURCE_IDS or len(source_ids) != len(set(source_ids)):
        raise ValueError("Ground context must declare eight unique official sources")
    incidents = payload.get("incidents")
    if not isinstance(incidents, dict) or set(incidents) != REQUIRED_INCIDENTS:
        raise ValueError("Ground context must bind all six terrain incidents")
    for fire_id, incident in incidents.items():
        department = str(incident.get("department", ""))
        title = str(incident.get("ocsge_title", ""))
        brgm_url = str(incident.get("brgm_url", ""))
        if len(department) != 3 or not department.isdigit():
            raise ValueError(f"Invalid department binding for {fire_id}")
        if f"D{department}" not in title:
            raise ValueError(f"OCS GE title does not match {fire_id}")
        if not brgm_url.startswith("https://infoterre.brgm.fr/telechargements/"):
            raise ValueError(f"Unofficial BRGM URL for {fire_id}")
    return payload


def load_runtime_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("Unsupported ground surface runtime contract")
    return payload


def validate_profile_bindings(
    context_contract: Mapping[str, Any], runtime_contract: Mapping[str, Any]
) -> dict[str, list[str]]:
    bindings = context_contract.get("profile_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("Ground context profile bindings are absent")
    runtime_families = {
        str(family["id"]): [str(variant) for variant in family.get("variant_ids", [])]
        for family in runtime_contract.get("profile_families", [])
    }
    if set(bindings) != set(runtime_families):
        raise ValueError("Ground context and runtime profile families differ")
    known_sources = REQUIRED_SOURCE_IDS | DERIVED_SOURCE_IDS
    profile_tags: dict[str, list[str]] = {}
    for family_id, variants in runtime_families.items():
        family_binding = bindings[family_id]
        source_layers = family_binding.get("source_layers")
        if (
            not isinstance(source_layers, list)
            or not source_layers
            or not set(source_layers) <= known_sources
        ):
            raise ValueError(f"Invalid source layers for {family_id}")
        variant_tags = family_binding.get("variant_tags")
        if not isinstance(variant_tags, dict) or set(variant_tags) != set(variants):
            raise ValueError(f"Context variant tags are incomplete for {family_id}")
        for variant in variants:
            tags = variant_tags[variant]
            if (
                not isinstance(tags, list)
                or not tags
                or any(not isinstance(tag, str) or ":" not in tag for tag in tags)
            ):
                raise ValueError(f"Invalid semantic tags for {family_id}.{variant}")
            profile_tags[f"{family_id}.{variant}"] = tags
    if len(profile_tags) != 72:
        raise ValueError("Ground context must bind exactly 72 profiles")
    return profile_tags


def classify_feature(layer_id: str, properties: Mapping[str, Any]) -> set[str]:
    """Convert approved source attributes into stable semantic selection tags."""

    tags = {f"source:{layer_id}"}
    values = " ".join(_normalize(value) for value in properties.values())
    if layer_id == "land_parcels":
        return tags | {"parcel:land"}
    if layer_id == "agricultural_parcels":
        tags.add("landcover:agriculture")
        agriculture_keywords = {
            "agriculture:vineyard": ("vigne", "vignoble", "viticulture", "vrc"),
            "agriculture:orchard": ("verger", "arboriculture", "fruit"),
            "agriculture:pasture": ("prairie", "paturage", "fourrage"),
            "agriculture:fallow": ("jachere",),
            "agriculture:cereal": ("cereale", "ble", "orge", "mais"),
        }
        for tag, keywords in agriculture_keywords.items():
            if any(keyword in values for keyword in keywords):
                tags.add(tag)
        return tags
    if layer_id == "landcover":
        code_cs = str(properties.get("code_cs", ""))
        code_us = str(properties.get("code_us", ""))
        cs_tags = {
            "CS1.2.1": "landcover:bare_soil",
            "CS1.2.2": "landcover:water",
            "CS2.1.1.1": "landcover:deciduous_forest",
            "CS2.1.1.2": "landcover:conifer_forest",
            "CS2.1.1.3": "landcover:mixed_forest",
            "CS2.1.2": "landcover:shrub",
            "CS2.2.1": "landcover:herbaceous",
        }
        if code_cs in cs_tags:
            tags.add(cs_tags[code_cs])
        if code_cs.startswith("CS2.1.1"):
            tags.add("landcover:forest")
        us_tags = {
            "US1.1": "landcover:agriculture",
            "US4.1.1": "transport:road",
            "US4.1.2": "transport:rail",
        }
        if code_us in us_tags:
            tags.add(us_tags[code_us])
        return tags
    if layer_id == "geology":
        geology_keywords = {
            "geology:karst": ("karst",),
            "geology:limestone": ("calcaire", "limestone", "dolomie"),
            "geology:red_clay": ("argile rouge", "terre rouge"),
            "geology:marl": ("marne", "marl"),
            "geology:schist": ("schiste", "micaschiste", "schist"),
            "geology:granite": ("granite", "granit", "granodiorite"),
            "geology:sandstone": ("gres", "sandstone"),
            "geology:conglomerate": ("conglomerat", "poudingue"),
            "geology:basalt": ("basalte", "basalt", "volcan"),
            "geology:alluvium": ("alluv", "terrasse fluviatile"),
            "geology:siliceous_sand": ("sable siliceux", "silice"),
            "geology:sand": ("sable",),
            "geology:silt": ("limon", "silt"),
            "geology:loam": ("limon", "terre vegetale", "loam"),
            "geology:rock": ("roche", "calcaire", "schiste", "granite", "gres"),
        }
        for tag, keywords in geology_keywords.items():
            if any(keyword in values for keyword in keywords):
                tags.add(tag)
        return tags
    if layer_id == "roads":
        tags.add("transport:road")
        if any(keyword in values for keyword in ("non reve", "empierre", "chemin")):
            tags.update({"road:unpaved", "transport:path"})
        else:
            tags.add("road:asphalt")
        if "sentier" in values:
            tags.add("transport:path")
        if any(keyword in values for keyword in ("piste", "dfci", "service")):
            tags.update({"transport:service", "transport:fire_service"})
        if "gravier" in values or "empierre" in values:
            tags.add("road:gravel")
        return tags
    if layer_id == "railways":
        tags.add("transport:rail")
        if not any(keyword in values for keyword in ("abandon", "hors service")):
            tags.add("rail:active")
        return tags
    if layer_id in {"hydro_lines", "hydro_surfaces"}:
        if "permanent" in values:
            tags.add("hydro:persistent")
        if any(keyword in values for keyword in ("intermittent", "temporaire")):
            tags.add("hydro:seasonal")
        if "chenal" in values:
            tags.add("hydro:braided")
        return tags
    raise ValueError(f"Unsupported ground context layer: {layer_id}")


def select_profile(
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    *,
    family: str,
    semantic_tags: set[str],
    seed: str,
) -> dict[str, Any]:
    if atlas_catalog.get("schema") != ATLAS_SCHEMA:
        raise ValueError("Unsupported ground surface atlas catalog")
    family_binding = context_contract.get("profile_bindings", {}).get(family)
    if not isinstance(family_binding, dict):
        raise ValueError(f"Unbound ground profile family: {family}")
    variant_tags = family_binding["variant_tags"]
    candidates = []
    for profile in atlas_catalog.get("profiles", []):
        if profile.get("family") != family:
            continue
        tags = set(variant_tags[profile["variant"]])
        score = len(tags & semantic_tags)
        if score:
            digest = hashlib.sha256(
                f"{seed}\0{profile['id']}".encode("utf-8")
            ).digest()
            candidates.append((-score, digest, profile["id"], profile))
    if not candidates:
        raise ValueError(f"No {family} profile matches approved semantic context")
    return min(candidates)[-1]
