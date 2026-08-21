"""Build the complete 3D reference catalogue with one USD per reference.

Reference PNG files define the assets that must eventually exist.  They are
never copied to a runtime scene.  A reference without its own reviewed USD is
resolved deterministically to a compatible real USD already present in the
catalogue.  The requested identity and the real donor identity are both kept
in the receipt; runtime black cubes and other primitive placeholders are
forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA = "fireviewer.reference-usd-asset-library.v1"
STATUS = "catalogued_with_deterministic_real_usd_fallbacks"
ALGORITHM = "fireviewer.reference-usd-asset-library-builder.v1"
REFERENCE_ROUTE = "hunyuan3d"
SOURCE_PRECEDENCE = (
    "premium_usdz",
    "simready_normalized",
    "added_hunyuan",
    "reviewed_hunyuan",
)
_SOURCE_PRIORITY = {
    tier: len(SOURCE_PRECEDENCE) - index for index, tier in enumerate(SOURCE_PRECEDENCE)
}
_PREMIUM_NAME_ALIASES = {
    # The downloaded Tripo package kept its generic export name.  There is one
    # and only one chalet reference in the complete FireViewer inventory.
    "chalet_house_3d_model": "febc96eb56d2_06_chalet_alpin",
}

_EQUIPMENT_CATEGORIES = (
    "drainage_equipment",
    "hydro_equipment",
    "pasture_equipment",
    "public_equipment",
    "rail_equipment",
    "road_equipment",
    "sports_equipment",
    "utility_equipment",
)
_FALLBACK_COMPATIBILITY = {
    "building": ("building",),
    "tree": ("tree", "vegetation"),
    "vegetation": ("vegetation", "tree"),
    "vehicle": ("vehicle",),
    **{category: _EQUIPMENT_CATEGORIES for category in _EQUIPMENT_CATEGORIES},
}
_REPLACEMENT_POLICY = (
    "premium_usdz_then_simready_normalized_then_added_hunyuan_then_"
    "reviewed_hunyuan_then_"
    "deterministic_compatible_real_usd"
)


@dataclass(frozen=True)
class CandidateAsset:
    """A source artifact discovered outside the immutable output catalogue."""

    asset_id: str
    tier: str
    source_path: Path
    texture_path: Path | None
    texture_member: str | None
    source_bounds: Mapping[str, Any] | None
    usd_stage: Mapping[str, Any] | None
    material_policy: str
    source_name: str
    source_sha256: str
    source_byte_count: int
    texture_sha256: str
    texture_byte_count: int
    inspection_sha256: str | None = None
    qualification: Mapping[str, Any] | None = None


class DiscoveredCandidates(dict[str, CandidateAsset]):
    """Candidate mapping carrying identities barred by the SimReady review."""

    def __init__(
        self,
        values: Mapping[str, CandidateAsset] | None = None,
        *,
        rejected_asset_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(values or {})
        self.rejected_asset_ids = frozenset(rejected_asset_ids)


_PLACEMENT_CONTEXTS = {
    "building": ("building",),
    "tree": ("measured_woody_canopy",),
    "vegetation": ("measured_low_vegetation",),
    "road_equipment": ("road",),
    "rail_equipment": ("rail",),
    "hydro_equipment": ("hydro",),
    "drainage_equipment": ("road_hydro_crossing",),
    "public_equipment": ("public_space",),
    "utility_equipment": ("utility_network",),
    "pasture_equipment": ("pasture",),
    "sports_equipment": ("sports_ground",),
    "vehicle": ("measured_or_semantically_confirmed_vehicle",),
}

_SEMANTIC_KEYWORDS = {
    "agricultural": (
        "agricol",
        "bergerie",
        "chai",
        "ecurie",
        "etable",
        "ferme",
        "grange",
        "hangar",
        "poulailler",
        "silo",
        "viticole",
    ),
    "commercial": (
        "bar",
        "boulangerie",
        "boutique",
        "cafe",
        "commerce",
        "epicerie",
        "pharmacie",
        "restaurant",
        "superette",
        "supermarche",
    ),
    "industrial": (
        "agroalimentaire",
        "atelier",
        "conditionnement",
        "depot",
        "entrepot",
        "industrie",
        "laiterie",
        "logistique",
        "scierie",
        "usine",
    ),
    "residential": (
        "chalet",
        "collectif",
        "habitat",
        "hlm",
        "immeuble",
        "logement",
        "maison",
        "pavillon",
        "residence",
    ),
    "public_service": (
        "administratif",
        "caserne",
        "ecole",
        "gendarmerie",
        "mairie",
        "medical",
        "poste",
        "salle_polyvalente",
    ),
    "religious": ("chapelle", "eglise", "paroiss", "presbytere"),
    "degraded": ("abandon", "degrade", "effondr", "ruine", "vacant"),
    "conifer": (
        "cypres",
        "douglas",
        "epicea",
        "if_commun",
        "meleze",
        "pin_",
        "sapin",
    ),
    "mediterranean": ("chene_vert", "cypres", "olivier", "pin_maritime"),
    "riparian": ("aulne", "peuplier", "saule"),
    "burned": ("brule", "calcine", "carbonise", "cendre", "mort_sur_pied"),
    "shrub": ("buisson", "genet", "roncier"),
    "grass": ("fleur", "herbe", "prairie", "roseau"),
    "road_safety": ("balise", "barriere", "bordure", "glissiere", "potelet"),
    "road_drainage": ("caniveau", "buse", "evacuation", "fosse", "regard"),
    "rail_signal": ("barriere", "borne", "catenaire", "coffret", "signal"),
    "rail_station": ("abri_de_quai", "gare", "quai"),
    "hydro_crossing": ("gue", "passerelle", "pont"),
    "hydro_bank": ("berge", "enrochement", "fascine", "ponton"),
}


class ReferenceAssetLibraryError(ValueError):
    """The reference list cannot be reconciled with the reviewed USD set."""


def _canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + ("\n" if pretty else "")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(value: Path | str | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        try:
            payload = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReferenceAssetLibraryError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ReferenceAssetLibraryError(f"{label} must be a JSON object")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReferenceAssetLibraryError(f"{label} must be a lowercase SHA-256")
    return value


def _portable_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ReferenceAssetLibraryError(f"{label} is not a portable path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ":" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReferenceAssetLibraryError(f"{label} is not a confined relative path")
    return path.as_posix()


def _portable_basename(value: str) -> str:
    """Return the final component of either a POSIX or Windows receipt path."""

    normalized = value.replace("\\", "/")
    return PurePosixPath(normalized).name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_reference_name(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"^[0-9a-f]{12}_", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s*-\s*copie$", "", stem, flags=re.IGNORECASE)
    decomposed = unicodedata.normalize("NFKD", stem.casefold())
    ascii_text = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).replace("+", "_")
    ascii_text = re.sub(r"^[0-9]{1,3}[_ -]+", "", ascii_text)
    return "_".join(re_split_non_alnum(ascii_text))


def _inspection_index(
    value: Path | str | Mapping[str, Any] | None,
) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    if value is None:
        return {}, None
    payload = _load_json(value, "candidate inspection")
    if payload.get("schema") != "fireviewer.usd-candidate-inspection.v1":
        raise ReferenceAssetLibraryError("candidate inspection schema is invalid")
    supplied = _require_sha256(
        payload.get("content_sha256"), "candidate inspection content hash"
    )
    basis = dict(payload)
    basis.pop("content_sha256", None)
    if _sha256_bytes(_canonical_bytes(basis)) != supplied:
        raise ReferenceAssetLibraryError("candidate inspection content hash differs")
    records = payload.get("artifacts")
    if not isinstance(records, list):
        raise ReferenceAssetLibraryError("candidate inspection artifacts are invalid")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ReferenceAssetLibraryError("candidate inspection record is invalid")
        name = record.get("source_name")
        if not isinstance(name, str) or Path(name).name != name or name in indexed:
            raise ReferenceAssetLibraryError(
                "candidate inspection source names are invalid or duplicated"
            )
        _require_sha256(record.get("source_sha256"), "inspected source hash")
        indexed[name] = record
    return indexed, supplied


def _validated_bounds(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceAssetLibraryError(f"{label} bounds are missing")
    minimum = value.get("minimum")
    maximum = value.get("maximum")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        raise ReferenceAssetLibraryError(f"{label} bounds are invalid")
    low = [float(item) for item in minimum]
    high = [float(item) for item in maximum]
    if any(not math.isfinite(item) for item in (*low, *high)) or any(
        high[index] <= low[index] for index in range(3)
    ):
        raise ReferenceAssetLibraryError(f"{label} bounds are empty or non-finite")
    diagonal = math.sqrt(sum((high[index] - low[index]) ** 2 for index in range(3)))
    return {
        "status": "reported",
        "coordinate_space": str(value.get("coordinate_space", "usd_authored_world")),
        "minimum": low,
        "maximum": high,
        "diagonal": diagonal,
    }


def _validated_stage(value: Any, label: str, *, required: bool) -> dict[str, Any]:
    if value is None and not required:
        return {
            "status": "pending",
            "up_axis": None,
            "meters_per_unit": None,
            "default_prim": None,
        }
    if not isinstance(value, Mapping) or value.get("status") != "inspected":
        raise ReferenceAssetLibraryError(f"{label} USD stage inspection is missing")
    up_axis = value.get("up_axis")
    meters = value.get("meters_per_unit")
    default_prim = value.get("default_prim")
    if up_axis not in {"Y", "Z"}:
        raise ReferenceAssetLibraryError(f"{label} USD stage upAxis is unsupported")
    if not isinstance(meters, (int, float)) or not math.isfinite(meters) or meters <= 0:
        raise ReferenceAssetLibraryError(f"{label} metersPerUnit is invalid")
    if (
        not isinstance(default_prim, str)
        or not default_prim.startswith("/")
        or default_prim == "/"
    ):
        raise ReferenceAssetLibraryError(f"{label} defaultPrim is invalid")
    return {
        "status": "inspected",
        "up_axis": up_axis,
        "meters_per_unit": float(meters),
        "default_prim": default_prim,
    }


def _inspect_usdz_archive(path: Path) -> tuple[str, bytes]:
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ReferenceAssetLibraryError(f"USDZ CRC differs: {path.name}")
            files = [item for item in archive.infolist() if not item.is_dir()]
            if not files or sum(item.file_size for item in files) > 512 * 1024 * 1024:
                raise ReferenceAssetLibraryError(
                    f"USDZ payload is invalid: {path.name}"
                )
            for item in files:
                member = PurePosixPath(item.filename)
                if (
                    member.is_absolute()
                    or "\\" in item.filename
                    or any(part in {"", ".", ".."} for part in member.parts)
                ):
                    raise ReferenceAssetLibraryError(
                        f"USDZ member is not confined: {path.name}"
                    )
            root_layers = [
                item.filename
                for item in files
                if len(PurePosixPath(item.filename).parts) == 1
                and PurePosixPath(item.filename).suffix.casefold()
                in {".usd", ".usda", ".usdc"}
            ]
            if len(root_layers) != 1:
                raise ReferenceAssetLibraryError(
                    f"USDZ must contain one root USD layer: {path.name}"
                )
            base_colors = [
                item.filename
                for item in files
                if re.search(
                    r"(?:^|/)textures/[^/]+_0\.(?:jpg|jpeg|png)$",
                    item.filename,
                    re.IGNORECASE,
                )
            ]
            if len(base_colors) != 1:
                raise ReferenceAssetLibraryError(
                    f"USDZ must contain one base-color texture: {path.name}"
                )
            return base_colors[0], archive.read(base_colors[0])
    except (OSError, zipfile.BadZipFile) as error:
        raise ReferenceAssetLibraryError(
            f"invalid USDZ {path.name}: {error}"
        ) from error


def _metadata_for_candidate(
    path: Path,
    inspection: Mapping[str, Mapping[str, Any]],
    inspection_sha256: str | None,
    *,
    required: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str | None]:
    record = inspection.get(path.name)
    if record is None:
        if required:
            raise ReferenceAssetLibraryError(
                f"candidate inspection is required for {path.name}"
            )
        return (
            None,
            _validated_stage(None, path.name, required=False),
            "fireviewer_color_override",
            None,
        )
    if record.get("source_sha256") != _sha256_file(path):
        raise ReferenceAssetLibraryError(
            f"candidate inspection source hash differs: {path.name}"
        )
    stage = _validated_stage(record.get("usd_stage"), path.name, required=True)
    bounds = _validated_bounds(record.get("source_bounds"), path.name)
    if stage["up_axis"] == "Z":
        source_low = bounds["minimum"]
        source_high = bounds["maximum"]
        bounds = _validated_bounds(
            {
                "coordinate_space": "usd_canonical_y_up_from_z_up",
                "minimum": [source_low[0], source_low[2], -source_high[1]],
                "maximum": [source_high[0], source_high[2], -source_low[1]],
            },
            path.name,
        )
    mesh_count = record.get("mesh_count")
    material_count = record.get("material_count")
    bound_count = record.get("bound_material_mesh_count")
    scope_safe = record.get("material_scope_safe")
    if (
        not isinstance(mesh_count, int)
        or mesh_count < 1
        or not isinstance(material_count, int)
        or material_count < 0
        or not isinstance(bound_count, int)
        or bound_count < 0
        or bound_count > mesh_count
        or not isinstance(scope_safe, bool)
    ):
        raise ReferenceAssetLibraryError(
            f"candidate material inspection is invalid: {path.name}"
        )
    policy = (
        "source_package_pbr"
        if scope_safe and bound_count == mesh_count and material_count > 0
        else "source_package_color_override"
    )
    return bounds, stage, policy, inspection_sha256


def _category(source_relative: str) -> str:
    path = PurePosixPath(source_relative)
    top = path.parts[0].casefold()
    name = path.stem.casefold()

    if top == "01_arbres":
        return "tree"
    if top == "01_lot_1_extension_initiale":
        number = name.split("_", 1)[0]
        return {
            "01": "tree",
            "02": "tree",
            "03": "building",
            "04": "building",
            "05": "building",
            "06": "building",
            "07": "vehicle",
            "08": "vehicle",
        }.get(number, "infrastructure")
    if top == "01_lot_3d_t1_bordures_routieres_securite":
        return "road_equipment"
    if (
        top == "02_batiments"
        or top.startswith("lot_0")
        or top
        in {
            "lot_10_patrimoine_coeur_ancien_et_ambiance_vieille_ville",
            "lot_11_batiments_degrades_vacants_ou_en_transition",
        }
    ):
        return "building"
    if top == "lot_12_elements_complementaires_indispensables_autour_du_bati":
        return "public_equipment"
    if top == "02_lot_2_services_et_habitat":
        if name.startswith("07_"):
            return "utility_equipment"
        if name.startswith("08_"):
            return "vehicle"
        return "building"
    if top == "02_lot_3d_t2_drainage_soutenement_terrassement":
        return "drainage_equipment"
    if top == "03_infrastructures":
        if "gare" in name or "passage_a_niveau" in name:
            return "rail_equipment"
        if "pont" in name:
            return "hydro_equipment"
        return "public_equipment"
    if top == "03_lot_3_post_incendie_et_agricole":
        if name.startswith("07_"):
            return "vehicle"
        if name.startswith("08_"):
            return "utility_equipment"
        if name.startswith(("01_", "02_", "05_", "06_")):
            return "tree"
        return "vegetation"
    if top == "03_lot_3d_t3_cours_eau_berges":
        return "hydro_equipment"
    if top == "04_lot_3d_t4_equipements_ferroviaires":
        return "rail_equipment"
    if top == "04_lot_4_equipements_publics_et_reseaux":
        if name.startswith(("01_", "02_", "03_")):
            return "building"
        if name.startswith(("04_", "05_")):
            return "tree"
        if name.startswith(("06_", "07_")):
            return "vehicle"
        return "utility_equipment"
    if top == "04_vegetaux_et_elements_naturels":
        return "vegetation"
    if top == "05_lot_3d_t5_parcelles_bocage_paturages":
        return "pasture_equipment"
    if top == "05_lot_5_architecture_et_vie_quotidienne":
        return "vehicle" if name.startswith("08_") else "building"
    if top == "05_vehicules_civils" or top in {
        "06_vehicules_secours",
        "08_lot_8_logistique_et_secours",
    }:
        return "vehicle"
    if top == "06_lot_3d_t6_sports_parkings_exterieurs":
        return "sports_equipment"
    if top == "06_lot_6_vegetation_complementaire":
        return "vegetation" if name.startswith(("07_", "08_")) else "tree"
    if top == "07_lot_7_post_incendie_et_infrastructures":
        if name.startswith(("01_", "02_", "03_", "04_")):
            return "vegetation"
        if name.startswith("05_"):
            return "utility_equipment"
        if name.startswith("06_"):
            return "rail_equipment"
        if name.startswith("07_"):
            return "hydro_equipment"
        return "vehicle"
    raise ReferenceAssetLibraryError(
        f"no deterministic 3D category rule for {source_relative}"
    )


def _search_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return "_".join(part for part in re_split_non_alnum(ascii_text) if part)


def re_split_non_alnum(value: str) -> list[str]:
    current: list[str] = []
    parts: list[str] = []
    for character in value:
        if character.isalnum():
            current.append(character)
        elif current:
            parts.append("".join(current))
            current.clear()
    if current:
        parts.append("".join(current))
    return parts


def _placement_profile(source_relative: str, category: str) -> dict[str, Any]:
    if category not in _PLACEMENT_CONTEXTS:
        raise ReferenceAssetLibraryError(
            f"no placement context for asset category {category}"
        )
    searchable = _search_text(source_relative)
    tags = sorted(
        tag
        for tag, fragments in _SEMANTIC_KEYWORDS.items()
        if any(fragment in searchable for fragment in fragments)
    )
    terms = sorted(
        {
            token
            for token in searchable.split("_")
            if len(token) >= 3 and not token.isdigit()
        }
    )
    return {
        "repeatable": True,
        "contexts": list(_PLACEMENT_CONTEXTS[category]),
        "semantic_tags": tags,
        "reference_terms": terms,
        "selection": "deterministic_best_metadata_match_then_stable_hash",
        "quantity_authority": (
            "one_or_more_independent_measured_or_SIG_candidates;"
            "catalogue_entries_are_never_consumed"
        ),
    }


def _fallback_metadata_score(
    target_placement: Mapping[str, Any], donor_placement: Mapping[str, Any]
) -> int:
    """Rank semantic similarity without allowing metadata to change quantity."""

    target_tags = set(target_placement.get("semantic_tags", []))
    donor_tags = set(donor_placement.get("semantic_tags", []))
    target_terms = set(target_placement.get("reference_terms", []))
    donor_terms = set(donor_placement.get("reference_terms", []))
    target_contexts = set(target_placement.get("contexts", []))
    donor_contexts = set(donor_placement.get("contexts", []))
    return (
        100 * len(target_tags & donor_tags)
        + 3 * len(target_terms & donor_terms)
        + 10 * len(target_contexts & donor_contexts)
    )


def _fallback_mode(target_category: str, donor_category: str) -> str:
    if target_category == donor_category:
        return "exact_category"
    if target_category in _EQUIPMENT_CATEGORIES and donor_category in (
        _EQUIPMENT_CATEGORIES
    ):
        return "compatible_equipment"
    if {target_category, donor_category} <= {"tree", "vegetation"}:
        return "compatible_woody_vegetation"
    raise ReferenceAssetLibraryError(
        f"no real USD fallback compatibility from {target_category} to {donor_category}"
    )


def _fallback_resolution(
    *,
    target_asset_id: str,
    target_category: str,
    donor: Mapping[str, Any],
    mode: str,
    metadata_match_score: int,
) -> dict[str, Any]:
    used = target_asset_id != donor["asset_id"]
    basis = {
        "target_asset_id": target_asset_id,
        "target_category": target_category,
        "donor_asset_id": donor["asset_id"],
        "donor_category": donor["category"],
        "donor_source_tier": donor["source_selection"]["tier"],
        "compatibility_mode": mode,
        "metadata_match_score": metadata_match_score,
        "donor_usd_sha256": donor["usd"]["sha256"],
        "donor_texture_sha256": donor["texture"]["sha256"],
    }
    return {
        "used": used,
        "donor_asset_id": donor["asset_id"],
        "donor_category": donor["category"],
        "donor_source_tier": donor["source_selection"]["tier"],
        "compatibility_mode": mode,
        "metadata_match_score": metadata_match_score,
        "resolution_sha256": _sha256_bytes(_canonical_bytes(basis)),
    }


def _select_fallback_donor(
    *,
    target_asset_id: str,
    target_category: str,
    target_placement: Mapping[str, Any],
    direct_assets: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str, int]:
    exact = [asset for asset in direct_assets if asset["category"] == target_category]
    if exact:
        compatible = exact
    else:
        allowed = _FALLBACK_COMPATIBILITY.get(target_category)
        if allowed is None:
            raise ReferenceAssetLibraryError(
                f"no fallback policy for asset category {target_category}"
            )
        compatible = [asset for asset in direct_assets if asset["category"] in allowed]
    if not compatible:
        raise ReferenceAssetLibraryError(
            f"no compatible real USD donor for {target_asset_id} ({target_category})"
        )
    scored = [
        (_fallback_metadata_score(target_placement, asset["placement"]), asset)
        for asset in compatible
    ]
    maximum = max(score for score, _asset in scored)
    finalists = [asset for score, asset in scored if score == maximum]
    finalists.sort(
        key=lambda asset: hashlib.sha256(
            f"{target_asset_id}\x1f{asset['asset_id']}".encode()
        ).hexdigest()
    )
    donor = finalists[0]
    return donor, _fallback_mode(target_category, str(donor["category"])), maximum


def _fallback_record(
    reference: Mapping[str, Any],
    category: str,
    donor: Mapping[str, Any],
    mode: str,
    metadata_match_score: int,
) -> dict[str, Any]:
    asset_id = str(reference["asset_id"])
    source_path = _portable_path(reference["source_relative"], "reference path")
    record = {
        "asset_id": asset_id,
        "category": category,
        "availability": "real_usd",
        "source_selection": dict(donor["source_selection"]),
        "material": dict(donor["material"]),
        "placement": _placement_profile(source_path, category),
        "reference": {
            "path": source_path,
            "byte_count": reference["source_bytes"],
            "sha256": reference["source_sha256"],
            "runtime_embedded": False,
        },
        "usd": dict(donor["usd"]),
        "texture": dict(donor["texture"]),
        "source_bounds": dict(donor["source_bounds"]),
        "usd_stage": dict(donor["usd_stage"]),
        "qualification": dict(donor["qualification"]),
        "replacement": {"key": asset_id, "policy": _REPLACEMENT_POLICY},
    }
    record["fallback_resolution"] = _fallback_resolution(
        target_asset_id=asset_id,
        target_category=category,
        donor=donor,
        mode=mode,
        metadata_match_score=metadata_match_score,
    )
    return record


def _hunyuan_candidate(path: Path, reference_ids: set[str]) -> CandidateAsset:
    asset_id = path.stem
    if asset_id not in reference_ids:
        raise ReferenceAssetLibraryError(
            f"added Hunyuan USD has no reference: {path.name}"
        )
    batch = path.parent.parent
    texture = path.parent / "textures" / f"{asset_id}.png"
    receipt = batch / "reports" / "usd" / f"{asset_id}-usd.json"
    glb_report = batch / "reports" / "glb-validation.json"
    if not texture.is_file() or not receipt.is_file() or not glb_report.is_file():
        raise ReferenceAssetLibraryError(
            f"added Hunyuan evidence is incomplete: {asset_id}"
        )
    receipt_payload = _load_json(receipt, f"Hunyuan USD receipt {asset_id}")
    source_hash = _sha256_file(path)
    texture_hash = _sha256_file(texture)
    if (
        receipt_payload.get("asset") != asset_id
        or receipt_payload.get("passed") is not True
        or receipt_payload.get("usd_sha256") != source_hash
        or receipt_payload.get("structural_validation", {}).get("texture_sha256")
        != texture_hash
    ):
        raise ReferenceAssetLibraryError(f"added Hunyuan receipt differs: {asset_id}")
    validation = _load_json(glb_report, f"Hunyuan GLB validation {asset_id}")
    matches = [
        item
        for item in validation.get("assets", [])
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).stem == asset_id
    ]
    if len(matches) != 1 or matches[0].get("passed") is not True:
        raise ReferenceAssetLibraryError(
            f"added Hunyuan bounds evidence differs: {asset_id}"
        )
    bounds = matches[0].get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ReferenceAssetLibraryError(
            f"added Hunyuan bounds are missing: {asset_id}"
        )
    source_bounds = _validated_bounds(
        {
            "coordinate_space": "source_glb_unscaled",
            "minimum": bounds[0],
            "maximum": bounds[1],
        },
        asset_id,
    )
    return CandidateAsset(
        asset_id=asset_id,
        tier="added_hunyuan",
        source_path=path.resolve(),
        texture_path=texture.resolve(),
        texture_member=None,
        source_bounds=source_bounds,
        usd_stage=_validated_stage(None, asset_id, required=False),
        material_policy="fireviewer_color_override",
        source_name=path.name,
        source_sha256=source_hash,
        source_byte_count=path.stat().st_size,
        texture_sha256=texture_hash,
        texture_byte_count=texture.stat().st_size,
    )


def _simready_candidate_evidence(
    root: Path,
    reference_ids: set[str],
) -> tuple[dict[str, CandidateAsset], frozenset[str]]:
    """Load the immutable, normalized 001-102 library and its rejection set."""

    names = (
        "active-assets.json",
        "simready-validation.json",
        "omniverse-asset-validator.json",
        "rejected-assets.json",
    )
    paths = {name: root / name for name in names}
    if any(not path.is_file() for path in paths.values()):
        raise ReferenceAssetLibraryError("SimReady evidence is incomplete")
    active = _load_json(paths["active-assets.json"], "SimReady active manifest")
    validation = _load_json(paths["simready-validation.json"], "SimReady validation")
    omniverse = _load_json(
        paths["omniverse-asset-validator.json"], "Omniverse asset validation"
    )
    rejected = _load_json(paths["rejected-assets.json"], "SimReady rejected manifest")
    active_records = active.get("assets")
    validation_records = validation.get("assets")
    rejected_records = rejected.get("assets")
    if (
        active.get("schema_version") != 1
        or active.get("status") != "validated_omniverse_minimal_placeable_visual"
        or active.get("property_assignment_intent") != "skip"
        or active.get("meters_per_unit") != 1.0
        or active.get("scale_policy") != "uniform_root_scale_only"
        or active.get("usd_up_axis") != "Z"
        or active.get("usd_root_rotation_degrees") != 0.0
        or not isinstance(active_records, list)
        or active.get("asset_count") != len(active_records)
        or active.get("rejected_count") != 7
    ):
        raise ReferenceAssetLibraryError("SimReady active manifest differs")
    if (
        validation.get("schema_version") != 1
        or validation.get("passed") is not True
        or validation.get("failed_count") != 0
        or validation.get("library_errors") != []
        or validation.get("asset_count") != len(active_records)
        or validation.get("passed_count") != len(active_records)
        or validation.get("expected_active_count") != len(active_records)
        or not isinstance(validation_records, list)
        or len(validation_records) != len(active_records)
    ):
        raise ReferenceAssetLibraryError("SimReady validation differs")
    features = omniverse.get("features")
    if (
        omniverse.get("status") != "PASS"
        or not isinstance(features, list)
        or not any(
            isinstance(feature, Mapping)
            and feature.get("id") == "com.nvidia.usd.minimal_placeable_visual"
            and feature.get("status") == "PASS"
            for feature in features
        )
    ):
        raise ReferenceAssetLibraryError("Omniverse SimReady validation differs")
    if (
        rejected.get("schema_version") != 1
        or rejected.get("status") != "user_rejected"
        or not isinstance(rejected_records, list)
        or rejected.get("asset_count") != len(rejected_records)
        or len(rejected_records) != active.get("rejected_count")
    ):
        raise ReferenceAssetLibraryError("SimReady rejection manifest differs")

    rejected_ids = frozenset(
        str(record.get("asset_id"))
        for record in rejected_records
        if isinstance(record, Mapping)
    )
    if (
        len(rejected_ids) != len(rejected_records)
        or "" in rejected_ids
        or not rejected_ids <= reference_ids
        or sorted(
            int(record.get("index"))
            for record in rejected_records
            if isinstance(record, Mapping)
        )
        != validation.get("expected_rejected_indices")
    ):
        raise ReferenceAssetLibraryError("SimReady rejected identities differ")

    validation_by_id = {
        str(record.get("asset_id")): record
        for record in validation_records
        if isinstance(record, Mapping)
    }
    if len(validation_by_id) != len(validation_records):
        raise ReferenceAssetLibraryError("SimReady validation ids are duplicated")
    evidence_sha256 = _sha256_bytes(
        _canonical_bytes(
            {name: _sha256_file(path) for name, path in sorted(paths.items())}
        )
    )
    candidates: dict[str, CandidateAsset] = {}
    for record in active_records:
        if not isinstance(record, Mapping):
            raise ReferenceAssetLibraryError("SimReady asset record is invalid")
        asset_id = str(record.get("asset_id"))
        index = record.get("index")
        evidence = validation_by_id.get(asset_id)
        if (
            asset_id not in reference_ids
            or asset_id in rejected_ids
            or not isinstance(index, int)
            or index < 1
            or record.get("passed") is not True
            or evidence is None
            or evidence.get("passed") is not True
            or evidence.get("errors") != []
        ):
            raise ReferenceAssetLibraryError(
                f"SimReady asset identity/status differs: {asset_id}"
            )
        directory = root / "assets" / f"{index:03d}_{asset_id}"
        usd = directory / f"{asset_id}.usd"
        glb = directory / f"{asset_id}.glb"
        texture = directory / "textures" / f"{asset_id}.png"
        if any(not path.is_file() for path in (usd, glb, texture)):
            raise ReferenceAssetLibraryError(
                f"SimReady normalized artifact is missing: {asset_id}"
            )
        for key, expected in (("usd", usd), ("glb", glb), ("texture", texture)):
            declared = record.get(key)
            if (
                not isinstance(declared, str)
                or _portable_basename(declared) != expected.name
            ):
                raise ReferenceAssetLibraryError(
                    f"SimReady {key} declaration differs: {asset_id}"
                )
        usd_hash = _sha256_file(usd)
        glb_hash = _sha256_file(glb)
        texture_hash = _sha256_file(texture)
        material = record.get("usd_material_restore")
        scale = record.get("scale_color")
        validation_evidence = evidence.get("evidence")
        if (
            not isinstance(material, Mapping)
            or material.get("passed") is not True
            or material.get("usd_sha256") != usd_hash
            or material.get("source_glb_sha256") != glb_hash
            or material.get("meters_per_unit") != 1.0
            or material.get("up_axis") != "Z"
            or material.get("root_rotation_x_degrees") != 0.0
            or material.get("structural_validation", {}).get("texture_sha256")
            != texture_hash
            or not isinstance(scale, Mapping)
            or scale.get("corrected_sha256") != glb_hash
            or not isinstance(validation_evidence, Mapping)
            or validation_evidence.get("glb_sha256") != glb_hash
            or validation_evidence.get("usd_sha256") != usd_hash
            or validation_evidence.get("texture_sha256") != texture_hash
            or validation_evidence.get("meters_per_unit") != 1.0
            or validation_evidence.get("up_axis") != "Z"
        ):
            raise ReferenceAssetLibraryError(
                f"SimReady artifact evidence differs: {asset_id}"
            )
        ratios = scale.get("axis_scale_ratios")
        after = scale.get("after_geometry")
        if (
            not isinstance(ratios, list)
            or len(ratios) != 3
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in ratios
            )
            or max(float(value) for value in ratios)
            - min(float(value) for value in ratios)
            > 1e-9
            or not isinstance(after, Mapping)
        ):
            raise ReferenceAssetLibraryError(
                f"SimReady uniform scale evidence differs: {asset_id}"
            )
        bounds = after.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ReferenceAssetLibraryError(
                f"SimReady normalized bounds are missing: {asset_id}"
            )
        # The corrected GLB remains Y-up.  Its coordinates are exactly the
        # canonical prototype space used by the measured-scene wrapper; the
        # delivered USD is the corresponding Z-up representation.
        source_bounds = _validated_bounds(
            {
                "coordinate_space": "usd_canonical_y_up_from_simready_z_up",
                "minimum": bounds[0],
                "maximum": bounds[1],
            },
            asset_id,
        )
        expected_extents = [
            source_bounds["maximum"][0] - source_bounds["minimum"][0],
            source_bounds["maximum"][2] - source_bounds["minimum"][2],
            source_bounds["maximum"][1] - source_bounds["minimum"][1],
        ]
        reported_extents = validation_evidence.get("usd_extents_m")
        if (
            not isinstance(reported_extents, list)
            or len(reported_extents) != 3
            or any(
                not math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=1e-5
                )
                for actual, expected in zip(
                    reported_extents, expected_extents, strict=True
                )
            )
        ):
            raise ReferenceAssetLibraryError(
                f"SimReady normalized dimensions differ: {asset_id}"
            )
        target_m = scale.get("target_m")
        if not isinstance(target_m, (int, float)) or float(target_m) <= 0:
            raise ReferenceAssetLibraryError(
                f"SimReady target dimension is invalid: {asset_id}"
            )
        candidates[asset_id] = CandidateAsset(
            asset_id=asset_id,
            tier="simready_normalized",
            source_path=usd.resolve(),
            texture_path=texture.resolve(),
            texture_member=None,
            source_bounds=source_bounds,
            usd_stage=_validated_stage(
                {
                    "status": "inspected",
                    "up_axis": "Z",
                    "meters_per_unit": 1.0,
                    "default_prim": f"/{asset_id}",
                },
                asset_id,
                required=True,
            ),
            material_policy="scoped_source_pbr",
            source_name=usd.name,
            source_sha256=usd_hash,
            source_byte_count=usd.stat().st_size,
            texture_sha256=texture_hash,
            texture_byte_count=texture.stat().st_size,
            inspection_sha256=evidence_sha256,
            qualification={
                "dimensions": {
                    "status": "accepted",
                    "value_m": [float(value) for value in reported_extents],
                },
                "ground_anchor": {
                    "status": "accepted",
                    "offset_m": -source_bounds["minimum"][1],
                },
                "visual": {
                    "status": "validated_omniverse_minimal_placeable_visual",
                    "accepted": False,
                },
            },
        )
    if (
        len(candidates) != len(active_records)
        or set(candidates) & rejected_ids
        or set(candidates) | rejected_ids
        != {str(record.get("asset_id")) for record in active_records} | rejected_ids
    ):
        raise ReferenceAssetLibraryError("SimReady active identities differ")
    return dict(sorted(candidates.items())), rejected_ids


def _final_simready_candidate_evidence(
    root: Path,
    reference_ids: set[str],
    inspection: Mapping[str, Mapping[str, Any]],
    inspection_sha256: str | None,
) -> tuple[dict[str, CandidateAsset], frozenset[str]]:
    """Load the merged, reviewed 001-294 library without its source archives."""

    names = ("active-assets.json", "merge-validation.json", "rejected-assets.json")
    paths = {name: root / name for name in names}
    if any(not path.is_file() for path in paths.values()):
        raise ReferenceAssetLibraryError("final SimReady evidence is incomplete")
    active = _load_json(paths["active-assets.json"], "final SimReady active manifest")
    merge = _load_json(paths["merge-validation.json"], "final SimReady validation")
    rejected = _load_json(
        paths["rejected-assets.json"], "final SimReady rejected manifest"
    )
    active_records = active.get("assets")
    rejected_records = rejected.get("assets")
    range_count = len(reference_ids)
    if (
        active.get("schema_version") != 1
        or active.get("status") != "final_merged"
        or active.get("operation") != "merge_only_no_asset_modification"
        or active.get("range") != {"start": 1, "end": range_count}
        or not isinstance(active_records, list)
        or active.get("asset_count") != len(active_records)
        or active.get("rejected_count") != range_count - len(active_records)
    ):
        raise ReferenceAssetLibraryError("final SimReady active manifest differs")
    if (
        merge.get("schema_version") != 1
        or merge.get("status") != "passed"
        or merge.get("operation") != "merge_only_no_asset_modification"
        or merge.get("range_count") != range_count
        or merge.get("active_count") != len(active_records)
        or merge.get("rejected_count") != active.get("rejected_count")
        or merge.get("active_directory_count") != len(active_records)
        or merge.get("glb_count") != len(active_records)
        or merge.get("usd_count") != len(active_records)
        or merge.get("texture_count") != len(active_records)
    ):
        raise ReferenceAssetLibraryError("final SimReady merge validation differs")
    if (
        rejected.get("schema_version") != 1
        or rejected.get("status") != "excluded_from_final_library"
        or not isinstance(rejected_records, list)
        or rejected.get("asset_count") != len(rejected_records)
        or len(rejected_records) != active.get("rejected_count")
    ):
        raise ReferenceAssetLibraryError("final SimReady rejection manifest differs")

    def indexed_records(
        records: Sequence[Any], label: str
    ) -> tuple[dict[str, Mapping[str, Any]], list[int]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        indices: list[int] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ReferenceAssetLibraryError(f"{label} record is invalid")
            asset_id = str(record.get("asset_id"))
            index = record.get("index")
            if (
                asset_id not in reference_ids
                or asset_id in indexed
                or not isinstance(index, int)
                or not 1 <= index <= range_count
            ):
                raise ReferenceAssetLibraryError(f"{label} identity differs")
            indexed[asset_id] = record
            indices.append(index)
        if len(indices) != len(set(indices)):
            raise ReferenceAssetLibraryError(f"{label} indices are duplicated")
        return indexed, sorted(indices)

    active_by_id, active_indices = indexed_records(active_records, "final SimReady")
    rejected_by_id, rejected_indices = indexed_records(
        rejected_records, "final SimReady rejected"
    )
    if (
        set(active_by_id) & set(rejected_by_id)
        or set(active_by_id) | set(rejected_by_id) != reference_ids
        or active_indices != merge.get("active_indices")
        or rejected_indices != merge.get("rejected_indices")
        or sorted((*active_indices, *rejected_indices))
        != list(range(1, range_count + 1))
    ):
        raise ReferenceAssetLibraryError("final SimReady coverage differs")
    if not inspection or inspection_sha256 is None:
        raise ReferenceAssetLibraryError(
            "final SimReady OpenUSD inspection is required"
        )
    evidence_sha256 = _sha256_bytes(
        _canonical_bytes(
            {
                "candidate_inspection_sha256": inspection_sha256,
                **{name: _sha256_file(path) for name, path in sorted(paths.items())},
            }
        )
    )

    candidates: dict[str, CandidateAsset] = {}
    for asset_id, record in active_by_id.items():
        index = int(record["index"])
        if record.get("status") != "retained" or record.get("passed") is not True:
            raise ReferenceAssetLibraryError(
                f"final SimReady status differs: {asset_id}"
            )
        directory = root / "assets" / f"{index:03d}_{asset_id}"
        usd = directory / f"{asset_id}.usd"
        glb = directory / f"{asset_id}.glb"
        texture = directory / "textures" / f"{asset_id}.png"
        report = directory / "asset-report.json"
        # GLB files remain in the immutable local provenance library. The
        # runtime image deliberately carries only OpenUSD, its texture and the
        # small receipt needed to validate the selected source.
        if any(not path.is_file() for path in (usd, texture, report)):
            raise ReferenceAssetLibraryError(
                f"final SimReady artifact is missing: {asset_id}"
            )
        if _load_json(report, f"final SimReady report {asset_id}") != dict(record):
            raise ReferenceAssetLibraryError(
                f"final SimReady report differs: {asset_id}"
            )
        for key, expected in (("usd", usd), ("glb", glb), ("texture", texture)):
            declared = record.get(key)
            if (
                not isinstance(declared, str)
                or _portable_basename(declared) != expected.name
            ):
                raise ReferenceAssetLibraryError(
                    f"final SimReady {key} declaration differs: {asset_id}"
                )
        bounds, stage, policy, _inspection_hash = _metadata_for_candidate(
            usd, inspection, inspection_sha256, required=True
        )
        if policy not in {"source_package_pbr", "source_package_color_override"}:
            raise ReferenceAssetLibraryError(
                "final SimReady USD qualification differs: "
                f"{asset_id}; stage={stage}; material_policy={policy}"
            )
        assert bounds is not None
        meters_per_unit = float(stage["meters_per_unit"])
        extents = [
            (bounds["maximum"][axis] - bounds["minimum"][axis]) * meters_per_unit
            for axis in range(3)
        ]
        runtime_material_policy = (
            "scoped_source_pbr"
            if policy == "source_package_pbr"
            else "fireviewer_color_override"
        )
        candidates[asset_id] = CandidateAsset(
            asset_id=asset_id,
            tier="simready_normalized",
            source_path=usd.resolve(),
            texture_path=texture.resolve(),
            texture_member=None,
            source_bounds=bounds,
            usd_stage=stage,
            material_policy=runtime_material_policy,
            source_name=usd.name,
            source_sha256=_sha256_file(usd),
            source_byte_count=usd.stat().st_size,
            texture_sha256=_sha256_file(texture),
            texture_byte_count=texture.stat().st_size,
            inspection_sha256=evidence_sha256,
            qualification={
                "dimensions": {"status": "accepted", "value_m": extents},
                "ground_anchor": {
                    "status": "accepted",
                    "offset_m": -bounds["minimum"][1] * meters_per_unit,
                },
                "visual": {
                    "status": "validated_omniverse_minimal_placeable_visual",
                    "accepted": False,
                },
            },
        )
    return dict(sorted(candidates.items())), frozenset(rejected_by_id)


def discover_candidate_assets(
    reference_manifest: Path | str | Mapping[str, Any],
    *,
    hunyuan_roots: Sequence[Path | str] = (),
    premium_usdz_root: Path | str | None = None,
    simready_root: Path | str | None = None,
    candidate_inspection: Path | str | Mapping[str, Any] | None = None,
) -> DiscoveredCandidates:
    """Match candidate filenames to references and apply strict source priority."""

    manifest = _load_json(reference_manifest, "reference manifest")
    records = manifest.get("assets")
    if manifest.get("schema_version") != 1 or not isinstance(records, list):
        raise ReferenceAssetLibraryError("unsupported reference manifest")
    selected = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("route") == REFERENCE_ROUTE
    ]
    reference_ids = {str(record.get("asset_id")) for record in selected}
    if "" in reference_ids or len(reference_ids) != len(selected):
        raise ReferenceAssetLibraryError("reference ids are invalid or duplicated")
    names: dict[str, set[str]] = {}
    for record in selected:
        normalized = _normalized_reference_name(str(record.get("source_relative", "")))
        if not normalized:
            raise ReferenceAssetLibraryError("normalized reference name is invalid")
        names.setdefault(normalized, set()).add(str(record["asset_id"]))
    inspection, inspection_hash = _inspection_index(candidate_inspection)
    candidates: dict[str, CandidateAsset] = {}
    rejected_asset_ids: frozenset[str] = frozenset()

    def select(candidate: CandidateAsset) -> None:
        if (
            candidate.asset_id in rejected_asset_ids
            and candidate.tier != "premium_usdz"
        ):
            return
        previous = candidates.get(candidate.asset_id)
        if previous is not None and previous.tier == candidate.tier:
            raise ReferenceAssetLibraryError(
                f"duplicate {candidate.tier} source for {candidate.asset_id}"
            )
        if (
            previous is None
            or _SOURCE_PRIORITY[candidate.tier] > _SOURCE_PRIORITY[previous.tier]
        ):
            candidates[candidate.asset_id] = candidate

    if simready_root is not None:
        root = Path(simready_root).resolve()
        if not root.is_dir():
            raise ReferenceAssetLibraryError(f"SimReady root is missing: {root}")
        if (root / "merge-validation.json").is_file():
            normalized, rejected_asset_ids = _final_simready_candidate_evidence(
                root, reference_ids, inspection, inspection_hash
            )
        else:
            normalized, rejected_asset_ids = _simready_candidate_evidence(
                root, reference_ids
            )
        for candidate in normalized.values():
            select(candidate)

    for raw_root in hunyuan_roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise ReferenceAssetLibraryError(
                f"Hunyuan candidate root is missing: {root}"
            )
        for path in sorted(root.rglob("*.usd")):
            select(_hunyuan_candidate(path, reference_ids))

    if premium_usdz_root is not None:
        root = Path(premium_usdz_root).resolve()
        if not root.is_dir():
            raise ReferenceAssetLibraryError(f"premium USDZ root is missing: {root}")
        for path in sorted(root.glob("*.usdz")):
            normalized = _normalized_reference_name(path.name)
            matches = names.get(normalized, set())
            asset_id = _PREMIUM_NAME_ALIASES.get(normalized)
            if asset_id is None and len(matches) == 1:
                asset_id = next(iter(matches))
            if asset_id is None or asset_id not in reference_ids:
                raise ReferenceAssetLibraryError(
                    f"premium USDZ has no unique reference: {path.name}"
                )
            member, texture_bytes = _inspect_usdz_archive(path)
            bounds, stage, policy, metadata_hash = _metadata_for_candidate(
                path,
                inspection,
                inspection_hash,
                required=False,
            )
            select(
                CandidateAsset(
                    asset_id=asset_id,
                    tier="premium_usdz",
                    source_path=path.resolve(),
                    texture_path=None,
                    texture_member=member,
                    source_bounds=bounds,
                    usd_stage=stage,
                    material_policy=policy,
                    source_name=path.name,
                    source_sha256=_sha256_file(path),
                    source_byte_count=path.stat().st_size,
                    texture_sha256=_sha256_bytes(texture_bytes),
                    texture_byte_count=len(texture_bytes),
                    inspection_sha256=metadata_hash,
                )
            )
    return DiscoveredCandidates(
        dict(sorted(candidates.items())),
        rejected_asset_ids=sorted(rejected_asset_ids),
    )


def _real_record(
    reference: Mapping[str, Any], reviewed: Mapping[str, Any], category: str
) -> dict[str, Any]:
    source_path = _portable_path(reference["source_relative"], "reference path")
    if reviewed.get("source", {}).get("path") != source_path:
        raise ReferenceAssetLibraryError(
            f"reviewed asset source differs for {reference['asset_id']}"
        )
    if reviewed.get("category") != category:
        raise ReferenceAssetLibraryError(
            f"reviewed asset category differs for {reference['asset_id']}"
        )
    for role in ("usd", "texture"):
        artifact = reviewed.get(role)
        if not isinstance(artifact, Mapping):
            raise ReferenceAssetLibraryError(
                f"reviewed asset lacks {role}: {reference['asset_id']}"
            )
    return {
        "asset_id": reference["asset_id"],
        "category": category,
        "availability": "real_usd",
        "source_selection": {
            "tier": "reviewed_hunyuan",
            "priority": _SOURCE_PRIORITY["reviewed_hunyuan"],
            "source_name": PurePosixPath(str(reviewed["usd"]["path"])).name,
            "source_sha256": reviewed["usd"]["sha256"],
            "inspection_sha256": None,
        },
        "material": {
            "policy": "fireviewer_color_override",
            "source_package": False,
            "pbr_preserved": False,
        },
        "placement": _placement_profile(source_path, category),
        "reference": {
            "path": source_path,
            "byte_count": reference["source_bytes"],
            "sha256": reference["source_sha256"],
            "runtime_embedded": False,
        },
        "usd": dict(reviewed["usd"]),
        "texture": dict(reviewed["texture"]),
        "source_bounds": dict(reviewed["source_bounds"]),
        "usd_stage": dict(reviewed["usd_stage"]),
        "qualification": dict(reviewed["qualification"]),
        "replacement": {
            "key": reference["asset_id"],
            "policy": _REPLACEMENT_POLICY,
        },
    }


def _candidate_record(
    reference: Mapping[str, Any], candidate: CandidateAsset, category: str
) -> dict[str, Any]:
    source_path = _portable_path(reference["source_relative"], "reference path")
    if candidate.source_bounds is None:
        raise ReferenceAssetLibraryError(
            f"candidate bounds inspection is required for {candidate.source_name}"
        )
    if candidate.tier == "premium_usdz" and (
        candidate.usd_stage is None or candidate.usd_stage.get("status") != "inspected"
    ):
        raise ReferenceAssetLibraryError(
            f"premium USDZ inspection is required for {candidate.source_name}"
        )
    if candidate.tier == "premium_usdz":
        usd_path = f"premium-usdz/{candidate.asset_id}.usdz"
        texture_suffix = PurePosixPath(str(candidate.texture_member)).suffix.casefold()
        texture_path = f"premium-usdz/textures/{candidate.asset_id}{texture_suffix}"
    elif candidate.tier == "simready_normalized":
        usd_path = f"simready-normalized/{candidate.asset_id}.usd"
        texture_path = f"simready-normalized/textures/{candidate.asset_id}.png"
    elif candidate.tier == "added_hunyuan":
        usd_path = f"generated-usd/{candidate.asset_id}.usd"
        texture_path = f"generated-usd/textures/{candidate.asset_id}.png"
    else:  # pragma: no cover - internal invariant
        raise ReferenceAssetLibraryError("candidate source tier is unsupported")
    return {
        "asset_id": candidate.asset_id,
        "category": category,
        "availability": "real_usd",
        "source_selection": {
            "tier": candidate.tier,
            "priority": _SOURCE_PRIORITY[candidate.tier],
            "source_name": candidate.source_name,
            "source_sha256": candidate.source_sha256,
            "inspection_sha256": candidate.inspection_sha256,
        },
        "material": {
            "policy": candidate.material_policy,
            "source_package": candidate.tier == "premium_usdz",
            "pbr_preserved": candidate.material_policy
            in {"source_package_pbr", "scoped_source_pbr"},
        },
        "placement": _placement_profile(source_path, category),
        "reference": {
            "path": source_path,
            "byte_count": reference["source_bytes"],
            "sha256": reference["source_sha256"],
            "runtime_embedded": False,
        },
        "usd": {
            "root": "review_batch",
            "path": usd_path,
            "byte_count": candidate.source_byte_count,
            "sha256": candidate.source_sha256,
        },
        "texture": {
            "root": "review_batch",
            "path": texture_path,
            "byte_count": candidate.texture_byte_count,
            "sha256": candidate.texture_sha256,
        },
        "source_bounds": dict(candidate.source_bounds),
        "usd_stage": dict(candidate.usd_stage or {}),
        "qualification": dict(
            candidate.qualification
            or {
                "dimensions": {"status": "pending", "value_m": None},
                "ground_anchor": {"status": "pending", "offset_m": None},
                "visual": {"status": "pending", "accepted": False},
            }
        ),
        "replacement": {
            "key": candidate.asset_id,
            "policy": _REPLACEMENT_POLICY,
        },
    }


def build_reference_asset_library(
    reference_manifest: Path | str | Mapping[str, Any],
    reviewed_library: Path | str | Mapping[str, Any],
    *,
    candidate_assets: Mapping[str, CandidateAsset] | None = None,
) -> dict[str, Any]:
    """Compile every reference to a direct or compatible deterministic real USD."""

    manifest = _load_json(reference_manifest, "reference manifest")
    reviewed = _load_json(reviewed_library, "reviewed asset library")
    records = manifest.get("assets")
    if manifest.get("schema_version") != 1 or not isinstance(records, list):
        raise ReferenceAssetLibraryError("unsupported reference manifest")
    reviewed_assets = reviewed.get("assets")
    if reviewed.get("schema") != "fireviewer.asset-library.v1" or not isinstance(
        reviewed_assets, list
    ):
        raise ReferenceAssetLibraryError("unsupported reviewed asset library")
    reviewed_by_id = {
        str(asset.get("asset_id")): asset
        for asset in reviewed_assets
        if isinstance(asset, Mapping)
    }
    if len(reviewed_by_id) != len(reviewed_assets):
        raise ReferenceAssetLibraryError("reviewed asset ids are invalid or duplicated")

    selected = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("route") == REFERENCE_ROUTE
    ]
    if len(selected) != manifest.get("route_counts", {}).get(REFERENCE_ROUTE):
        raise ReferenceAssetLibraryError("reference manifest route count differs")
    selected.sort(key=lambda value: str(value.get("asset_id")))
    ids = [str(record.get("asset_id")) for record in selected]
    if any(not asset_id for asset_id in ids) or len(ids) != len(set(ids)):
        raise ReferenceAssetLibraryError("3D reference ids are invalid or duplicated")
    if set(reviewed_by_id) - set(ids):
        raise ReferenceAssetLibraryError("reviewed library contains unknown references")
    candidate_source = candidate_assets or {}
    rejected_asset_ids = frozenset(getattr(candidate_source, "rejected_asset_ids", ()))
    candidates = dict(candidate_source)
    if set(candidates) - set(ids) or any(
        key != candidate.asset_id for key, candidate in candidates.items()
    ):
        raise ReferenceAssetLibraryError("candidate assets contain unknown identities")
    if not rejected_asset_ids <= set(ids):
        raise ReferenceAssetLibraryError("rejected assets contain unknown identities")

    direct_by_id: dict[str, dict[str, Any]] = {}
    for reference in selected:
        category = _category(str(reference.get("source_relative")))
        candidate = candidates.get(str(reference["asset_id"]))
        reviewed_asset = (
            None
            if str(reference["asset_id"]) in rejected_asset_ids
            else reviewed_by_id.get(str(reference["asset_id"]))
        )
        direct = (
            _candidate_record(reference, candidate, category)
            if candidate is not None
            else (
                _real_record(reference, reviewed_asset, category)
                if reviewed_asset is not None
                else None
            )
        )
        if direct is not None:
            direct["fallback_resolution"] = _fallback_resolution(
                target_asset_id=str(reference["asset_id"]),
                target_category=category,
                donor=direct,
                mode="direct",
                metadata_match_score=0,
            )
            direct_by_id[str(reference["asset_id"])] = direct
    if not direct_by_id:
        raise ReferenceAssetLibraryError(
            "at least one real USD is required to resolve missing references"
        )

    direct_assets = list(direct_by_id.values())
    assets: list[dict[str, Any]] = []
    for reference in selected:
        asset_id = str(reference["asset_id"])
        direct = direct_by_id.get(asset_id)
        if direct is not None:
            assets.append(direct)
            continue
        category = _category(str(reference.get("source_relative")))
        placement = _placement_profile(str(reference["source_relative"]), category)
        donor, mode, score = _select_fallback_donor(
            target_asset_id=asset_id,
            target_category=category,
            target_placement=placement,
            direct_assets=direct_assets,
        )
        assets.append(_fallback_record(reference, category, donor, mode, score))
    category_counts = dict(
        sorted(Counter(asset["category"] for asset in assets).items())
    )
    availability_counts = dict(
        sorted(Counter(asset["availability"] for asset in assets).items())
    )
    source_counts = dict(
        sorted(
            Counter(
                asset["source_selection"]["tier"] for asset in direct_assets
            ).items()
        )
    )
    all_pools = {
        category: sorted(
            asset["asset_id"] for asset in assets if asset["category"] == category
        )
        for category in category_counts
    }
    available_pools = dict(all_pools)
    fallback_count = len(assets) - len(direct_assets)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "build_algorithm": ALGORITHM,
        "asset_count": len(assets),
        "category_counts": category_counts,
        "availability_counts": availability_counts,
        "source_precedence": list(SOURCE_PRECEDENCE),
        "source_counts": source_counts,
        # Selection keeps the complete expected identity list.  Missing source
        # identities resolve to real compatible USDs, never to primitives.
        "selection_pools": all_pools,
        "available_asset_pools": available_pools,
        "reference_manifest": {
            "asset_count": manifest.get("asset_count"),
            "route_counts": dict(manifest.get("route_counts", {})),
            "content_sha256": _sha256_bytes(_canonical_bytes(manifest)),
            "runtime_images_embedded": False,
        },
        "fallback_policy": {
            "black_placeholder_forbidden": True,
            "strategy": "exact_category_then_compatible_family_metadata_hash",
            "compatibility_groups": {
                category: list(values)
                for category, values in sorted(_FALLBACK_COMPATIBILITY.items())
            },
            "direct_asset_count": len(direct_assets),
            "fallback_asset_count": fallback_count,
            "unique_real_usd_count": len(direct_assets),
        },
        "assets": assets,
    }
    payload["catalog_revision"] = _sha256_bytes(_canonical_bytes(payload))
    validate_reference_asset_library(payload)
    return payload


def validate_reference_asset_library(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA or payload.get("status") != STATUS:
        raise ReferenceAssetLibraryError("reference asset schema/status is invalid")
    supplied = _require_sha256(payload.get("catalog_revision"), "catalog revision")
    without_revision = dict(payload)
    without_revision.pop("catalog_revision", None)
    if _sha256_bytes(_canonical_bytes(without_revision)) != supplied:
        raise ReferenceAssetLibraryError("catalog revision differs")
    assets = payload.get("assets")
    if not isinstance(assets, list) or payload.get("asset_count") != len(assets):
        raise ReferenceAssetLibraryError("asset count differs")
    ids = [asset.get("asset_id") for asset in assets if isinstance(asset, Mapping)]
    if len(ids) != len(assets) or ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ReferenceAssetLibraryError("asset ids must be unique and sorted")
    if payload.get("source_precedence") != list(SOURCE_PRECEDENCE):
        raise ReferenceAssetLibraryError("asset source precedence differs")
    categories = Counter()
    availability = Counter()
    indexed = {str(asset["asset_id"]): asset for asset in assets}
    resolution_keys = {
        "used",
        "donor_asset_id",
        "donor_category",
        "donor_source_tier",
        "compatibility_mode",
        "metadata_match_score",
        "resolution_sha256",
    }
    for asset in assets:
        category = asset.get("category")
        state = asset.get("availability")
        if not isinstance(category, str) or state != "real_usd":
            raise ReferenceAssetLibraryError("asset category/availability is invalid")
        categories[category] += 1
        availability[state] += 1
        source_selection = asset.get("source_selection")
        if not isinstance(source_selection, Mapping) or set(source_selection) != {
            "tier",
            "priority",
            "source_name",
            "source_sha256",
            "inspection_sha256",
        }:
            raise ReferenceAssetLibraryError("asset source selection is invalid")
        tier = source_selection.get("tier")
        if (
            tier not in _SOURCE_PRIORITY
            or source_selection.get("priority") != _SOURCE_PRIORITY[tier]
        ):
            raise ReferenceAssetLibraryError("asset source priority differs")
        source_name = source_selection.get("source_name")
        source_hash = source_selection.get("source_sha256")
        inspection_hash = source_selection.get("inspection_sha256")
        if not isinstance(source_name, str) or Path(source_name).name != source_name:
            raise ReferenceAssetLibraryError("real source selection is invalid")
        _require_sha256(source_hash, "real source hash")
        if inspection_hash is not None:
            _require_sha256(inspection_hash, "source inspection hash")
        material = asset.get("material")
        if not isinstance(material, Mapping) or set(material) != {
            "policy",
            "source_package",
            "pbr_preserved",
        }:
            raise ReferenceAssetLibraryError("asset material policy is invalid")
        material_policy = material.get("policy")
        if tier == "premium_usdz":
            if (
                material_policy
                not in {"source_package_pbr", "source_package_color_override"}
                or material.get("source_package") is not True
                or material.get("pbr_preserved")
                is not (material_policy == "source_package_pbr")
                or inspection_hash is None
            ):
                raise ReferenceAssetLibraryError("premium material policy is invalid")
        elif tier == "simready_normalized":
            valid_materials = (
                {
                    "policy": "scoped_source_pbr",
                    "source_package": False,
                    "pbr_preserved": True,
                },
                {
                    "policy": "fireviewer_color_override",
                    "source_package": False,
                    "pbr_preserved": False,
                },
            )
            if material not in valid_materials or inspection_hash is None:
                raise ReferenceAssetLibraryError("SimReady material policy is invalid")
        elif material != {
            "policy": "fireviewer_color_override",
            "source_package": False,
            "pbr_preserved": False,
        }:
            raise ReferenceAssetLibraryError("non-premium material policy differs")
        placement = asset.get("placement")
        expected_placement = _placement_profile(
            str(asset.get("reference", {}).get("path", "")), category
        )
        if placement != expected_placement:
            raise ReferenceAssetLibraryError("asset placement profile differs")
        reference = asset.get("reference")
        if (
            not isinstance(reference, Mapping)
            or reference.get("runtime_embedded") is not False
        ):
            raise ReferenceAssetLibraryError("reference image must not be embedded")
        _portable_path(reference.get("path"), "reference path")
        _require_sha256(reference.get("sha256"), "reference sha256")
        if (
            not isinstance(reference.get("byte_count"), int)
            or reference["byte_count"] < 1
        ):
            raise ReferenceAssetLibraryError("reference byte_count is invalid")
        for role in ("usd", "texture"):
            artifact = asset.get(role)
            if (
                not isinstance(artifact, Mapping)
                or artifact.get("root") != "review_batch"
            ):
                raise ReferenceAssetLibraryError(f"asset {role} artifact is invalid")
            _portable_path(artifact.get("path"), f"asset {role} path")
            _require_sha256(artifact.get("sha256"), f"asset {role} sha256")
            if (
                not isinstance(artifact.get("byte_count"), int)
                or artifact["byte_count"] < 1
            ):
                raise ReferenceAssetLibraryError(f"asset {role} byte_count is invalid")
        usd_suffix = PurePosixPath(str(asset["usd"]["path"])).suffix.casefold()
        texture_suffix = PurePosixPath(str(asset["texture"]["path"])).suffix.casefold()
        if tier == "premium_usdz":
            if usd_suffix != ".usdz" or texture_suffix not in {".jpg", ".jpeg", ".png"}:
                raise ReferenceAssetLibraryError("premium artifact formats differ")
            _validated_stage(
                asset.get("usd_stage"), str(asset["asset_id"]), required=True
            )
        elif tier == "simready_normalized":
            if usd_suffix not in {".usd", ".usda", ".usdc"} or texture_suffix != ".png":
                raise ReferenceAssetLibraryError("SimReady artifact formats differ")
            _validated_stage(
                asset.get("usd_stage"), str(asset["asset_id"]), required=True
            )
            qualification = asset.get("qualification")
            if (
                not isinstance(qualification, Mapping)
                or qualification.get("dimensions", {}).get("status") != "accepted"
                or qualification.get("ground_anchor", {}).get("status") != "accepted"
                or qualification.get("visual", {}).get("status")
                != "validated_omniverse_minimal_placeable_visual"
            ):
                raise ReferenceAssetLibraryError("SimReady qualification differs")
        elif usd_suffix not in {".usd", ".usda", ".usdc"} or texture_suffix != ".png":
            raise ReferenceAssetLibraryError("USD or texture artifact format differs")
        replacement = asset.get("replacement")
        if replacement != {
            "key": asset["asset_id"],
            "policy": _REPLACEMENT_POLICY,
        }:
            raise ReferenceAssetLibraryError("asset replacement policy differs")
        resolution = asset.get("fallback_resolution")
        if not isinstance(resolution, Mapping) or set(resolution) != resolution_keys:
            raise ReferenceAssetLibraryError("asset fallback resolution is invalid")
        if not isinstance(resolution.get("used"), bool):
            raise ReferenceAssetLibraryError("asset fallback used flag is invalid")
        if not isinstance(resolution.get("donor_asset_id"), str):
            raise ReferenceAssetLibraryError("asset fallback donor id is invalid")
        if not isinstance(resolution.get("donor_category"), str):
            raise ReferenceAssetLibraryError("asset fallback donor category is invalid")
        if resolution.get("donor_source_tier") not in _SOURCE_PRIORITY:
            raise ReferenceAssetLibraryError("asset fallback donor tier is invalid")
        if not isinstance(resolution.get("compatibility_mode"), str):
            raise ReferenceAssetLibraryError("asset fallback compatibility is invalid")
        if (
            not isinstance(resolution.get("metadata_match_score"), int)
            or resolution["metadata_match_score"] < 0
        ):
            raise ReferenceAssetLibraryError("asset fallback score is invalid")
        _require_sha256(resolution.get("resolution_sha256"), "fallback resolution")

    direct_assets = [
        asset for asset in assets if asset["fallback_resolution"]["used"] is False
    ]
    if not direct_assets:
        raise ReferenceAssetLibraryError("catalogue has no direct real USD")
    for asset in assets:
        resolution = asset["fallback_resolution"]
        if resolution["used"] is False:
            expected_resolution = _fallback_resolution(
                target_asset_id=str(asset["asset_id"]),
                target_category=str(asset["category"]),
                donor=asset,
                mode="direct",
                metadata_match_score=0,
            )
        else:
            donor = indexed.get(str(resolution["donor_asset_id"]))
            if donor is None or donor["fallback_resolution"]["used"] is not False:
                raise ReferenceAssetLibraryError(
                    "asset fallback donor must be a direct real USD"
                )
            selected, mode, score = _select_fallback_donor(
                target_asset_id=str(asset["asset_id"]),
                target_category=str(asset["category"]),
                target_placement=asset["placement"],
                direct_assets=direct_assets,
            )
            if selected["asset_id"] != donor["asset_id"]:
                raise ReferenceAssetLibraryError(
                    "asset fallback donor is not the deterministic optimum"
                )
            expected_resolution = _fallback_resolution(
                target_asset_id=str(asset["asset_id"]),
                target_category=str(asset["category"]),
                donor=donor,
                mode=mode,
                metadata_match_score=score,
            )
            for shared_field in (
                "source_selection",
                "material",
                "usd",
                "texture",
                "source_bounds",
                "usd_stage",
                "qualification",
            ):
                if asset[shared_field] != donor[shared_field]:
                    raise ReferenceAssetLibraryError(
                        f"asset fallback {shared_field} differs from donor"
                    )
        if resolution != expected_resolution:
            raise ReferenceAssetLibraryError("asset fallback resolution differs")

    source_counts = Counter(
        asset["source_selection"]["tier"] for asset in direct_assets
    )
    if payload.get("category_counts") != dict(sorted(categories.items())):
        raise ReferenceAssetLibraryError("category counts differ")
    if payload.get("availability_counts") != dict(sorted(availability.items())):
        raise ReferenceAssetLibraryError("availability counts differ")
    if payload.get("source_counts") != dict(sorted(source_counts.items())):
        raise ReferenceAssetLibraryError("source counts differ")
    fallback_count = len(assets) - len(direct_assets)
    expected_policy = {
        "black_placeholder_forbidden": True,
        "strategy": "exact_category_then_compatible_family_metadata_hash",
        "compatibility_groups": {
            category: list(values)
            for category, values in sorted(_FALLBACK_COMPATIBILITY.items())
        },
        "direct_asset_count": len(direct_assets),
        "fallback_asset_count": fallback_count,
        "unique_real_usd_count": len(direct_assets),
    }
    if payload.get("fallback_policy") != expected_policy:
        raise ReferenceAssetLibraryError("fallback policy differs")
    for name in ("selection_pools", "available_asset_pools"):
        pools = payload.get(name)
        if not isinstance(pools, Mapping) or set(pools) != set(categories):
            raise ReferenceAssetLibraryError(f"{name} categories differ")
        for category, pool in pools.items():
            expected = sorted(
                asset["asset_id"] for asset in assets if asset["category"] == category
            )
            if pool != expected:
                raise ReferenceAssetLibraryError(f"{name}.{category} differs")
    return {
        "asset_count": len(assets),
        "category_counts": dict(sorted(categories.items())),
        "availability_counts": dict(sorted(availability.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "direct_asset_count": len(direct_assets),
        "fallback_asset_count": fallback_count,
        "unique_real_usd_count": len(direct_assets),
        "catalog_revision": supplied,
    }


def _require_output(path: Path | str, label: str) -> Path:
    lexical = PureWindowsPath(str(path))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise ReferenceAssetLibraryError(f"{label} must stay on D: on Windows")
    resolved = Path(path).resolve(strict=False)
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise ReferenceAssetLibraryError(f"{label} must stay on D: on Windows")
    return resolved


def _candidate_payloads(candidate: CandidateAsset) -> tuple[bytes, bytes]:
    source = candidate.source_path.read_bytes()
    if (
        len(source) != candidate.source_byte_count
        or _sha256_bytes(source) != candidate.source_sha256
    ):
        raise ReferenceAssetLibraryError(
            f"candidate source changed: {candidate.source_name}"
        )
    if candidate.texture_member is not None:
        member, texture = _inspect_usdz_archive(candidate.source_path)
        if member != candidate.texture_member:
            raise ReferenceAssetLibraryError(
                f"candidate USDZ texture member changed: {candidate.source_name}"
            )
    elif candidate.texture_path is not None:
        texture = candidate.texture_path.read_bytes()
    else:  # pragma: no cover - dataclass invariant
        raise ReferenceAssetLibraryError("candidate texture source is missing")
    if (
        len(texture) != candidate.texture_byte_count
        or _sha256_bytes(texture) != candidate.texture_sha256
    ):
        raise ReferenceAssetLibraryError(
            f"candidate texture changed: {candidate.source_name}"
        )
    return source, texture


def _publish_immutable_directory(
    root: Path, relative_root: str, expected: Mapping[str, bytes]
) -> Path:
    destination = root / relative_root
    expected_names = sorted(expected)
    if destination.exists():
        actual_names = sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        )
        if actual_names != expected_names or any(
            destination.joinpath(*PurePosixPath(name).parts).read_bytes() != content
            for name, content in expected.items()
        ):
            raise ReferenceAssetLibraryError(
                f"existing {relative_root} bundle is immutable and differs"
            )
        return destination
    staging = root / f".{relative_root}.part"
    if staging.exists():
        raise ReferenceAssetLibraryError(f"{relative_root} staging already exists")
    try:
        staging.mkdir(parents=False)
        for name, content in sorted(expected.items()):
            target = staging.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        os.replace(staging, destination)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return destination


def write_reference_asset_library(
    payload: Mapping[str, Any],
    *,
    review_batch_root: Path | str,
    output_catalog: Path | str,
    candidate_assets: Mapping[str, CandidateAsset] | None = None,
    reviewed_source_root: Path | str | None = None,
) -> dict[str, Any]:
    """Publish unique direct sources and the fallback catalogue atomically."""

    summary = validate_reference_asset_library(payload)
    root = _require_output(review_batch_root, "review batch root")
    catalog = _require_output(output_catalog, "reference asset catalog")
    root.mkdir(parents=True, exist_ok=True)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    if (root / "placeholders").exists():
        raise ReferenceAssetLibraryError(
            "legacy placeholder bundle must not enter a fallback-real catalogue"
        )
    candidates = dict(candidate_assets or {})
    generated: dict[str, bytes] = {}
    premium: dict[str, bytes] = {}
    normalized: dict[str, bytes] = {}
    reviewed_files: dict[str, bytes] = {}
    for asset in payload["assets"]:
        if asset["fallback_resolution"]["used"] is True:
            continue
        tier = asset["source_selection"]["tier"]
        if tier == "reviewed_hunyuan":
            if reviewed_source_root is None and candidate_assets is None:
                # Preserve the catalogue-only/unit-test API. Production callers
                # pass the immutable reviewed source root explicitly.
                continue
            source_root = _require_output(
                reviewed_source_root or root, "reviewed source root"
            )
            for role in ("usd", "texture"):
                relative = PurePosixPath(str(asset[role]["path"]))
                if not relative.parts or relative.parts[0] != "usd":
                    raise ReferenceAssetLibraryError(
                        f"reviewed {role} path is invalid: {asset['asset_id']}"
                    )
                source = source_root.joinpath(*relative.parts)
                if (
                    not source.is_file()
                    or source.stat().st_size != asset[role]["byte_count"]
                    or _sha256_file(source) != asset[role]["sha256"]
                ):
                    raise ReferenceAssetLibraryError(
                        f"reviewed {role} artifact differs: {asset['asset_id']}"
                    )
                reviewed_files[PurePosixPath(*relative.parts[1:]).as_posix()] = (
                    source.read_bytes()
                )
            continue
        if tier not in {
            "added_hunyuan",
            "premium_usdz",
            "simready_normalized",
        }:
            continue
        candidate = candidates.get(str(asset["asset_id"]))
        if candidate is None or candidate.tier != tier:
            raise ReferenceAssetLibraryError(
                f"selected candidate materialization is missing: {asset['asset_id']}"
            )
        source, texture = _candidate_payloads(candidate)
        target = {
            "premium_usdz": premium,
            "simready_normalized": normalized,
            "added_hunyuan": generated,
        }[tier]
        source_relative = PurePosixPath(asset["usd"]["path"])
        texture_relative = PurePosixPath(asset["texture"]["path"])
        expected_root = {
            "premium_usdz": "premium-usdz",
            "simready_normalized": "simready-normalized",
            "added_hunyuan": "generated-usd",
        }[tier]
        target[source_relative.relative_to(expected_root).as_posix()] = source
        target[texture_relative.relative_to(expected_root).as_posix()] = texture
    if generated:
        _publish_immutable_directory(root, "generated-usd", generated)
    if premium:
        _publish_immutable_directory(root, "premium-usdz", premium)
    if normalized:
        _publish_immutable_directory(root, "simready-normalized", normalized)
    if reviewed_files:
        _publish_immutable_directory(root, "usd", reviewed_files)
    catalog_bytes = _canonical_bytes(dict(payload), pretty=True)
    if catalog.exists():
        if catalog.read_bytes() != catalog_bytes:
            raise ReferenceAssetLibraryError(
                "existing catalogue is immutable and differs"
            )
    else:
        temporary = catalog.with_name(f".{catalog.name}.part")
        temporary.write_bytes(catalog_bytes)
        os.replace(temporary, catalog)
    return {
        **summary,
        "placeholder_usd_count": 0,
        "fallback_asset_count": summary["fallback_asset_count"],
        "unique_real_usd_count": summary["unique_real_usd_count"],
        "materialized_added_hunyuan_count": sum(
            asset["fallback_resolution"]["used"] is False
            and asset["source_selection"]["tier"] == "added_hunyuan"
            for asset in payload["assets"]
        ),
        "materialized_premium_usdz_count": sum(
            asset["fallback_resolution"]["used"] is False
            and asset["source_selection"]["tier"] == "premium_usdz"
            for asset in payload["assets"]
        ),
        "materialized_simready_normalized_count": sum(
            asset["fallback_resolution"]["used"] is False
            and asset["source_selection"]["tier"] == "simready_normalized"
            for asset in payload["assets"]
        ),
        "materialized_reviewed_hunyuan_count": sum(
            asset["fallback_resolution"]["used"] is False
            and asset["source_selection"]["tier"] == "reviewed_hunyuan"
            for asset in payload["assets"]
        ),
        "catalog_path": str(catalog),
    }


def _runtime_tree_semantic_tags(asset: Mapping[str, Any]) -> set[str]:
    if asset.get("category") not in {"tree", "vegetation"}:
        return set()
    reference = asset.get("reference")
    path = str(reference.get("path", "")) if isinstance(reference, Mapping) else ""
    searchable = _search_text(path)
    broadleaf_fragments = (
        "bouleau",
        "charme",
        "chataignier",
        "chene",
        "erable",
        "frene",
        "hetre",
        "merisier",
        "noyer",
        "platane",
        "robinier",
        "tilleul",
    )
    tags: set[str] = set()
    if any(fragment in searchable for fragment in broadleaf_fragments):
        tags.add("broadleaf")
    if "chene" in searchable:
        tags.add("oak")
    return tags


def _select_asset_for_candidate_from_validated_library(
    library: Mapping[str, Any],
    *,
    category: str,
    zone: str,
    candidate: str,
    rule_version: str,
    usage: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select from a catalogue already validated by the scene builder."""

    pool = library["selection_pools"].get(category)
    if not isinstance(pool, list) or not pool:
        raise ReferenceAssetLibraryError(
            f"no expected USD asset for category {category}"
        )
    if usage not in {"technical_pilot_non_final", "final_scene"}:
        raise ReferenceAssetLibraryError("usage is invalid")
    candidates = [
        asset
        for asset in library["assets"]
        if asset["asset_id"] in pool and asset["category"] == category
    ]
    candidate_metadata = dict(metadata or {})
    context = candidate_metadata.get("context")
    if context is not None:
        if not isinstance(context, str) or not context:
            raise ReferenceAssetLibraryError("selection metadata context is invalid")
        contextual = [
            asset for asset in candidates if context in asset["placement"]["contexts"]
        ]
        if contextual:
            candidates = contextual
    raw_tags = candidate_metadata.get("semantic_tags", [])
    if not isinstance(raw_tags, list) or any(
        not isinstance(tag, str) or not tag for tag in raw_tags
    ):
        raise ReferenceAssetLibraryError("selection metadata semantic_tags are invalid")
    semantic_tags = set(raw_tags)
    raw_terms = candidate_metadata.get("reference_terms", [])
    if not isinstance(raw_terms, list) or any(
        not isinstance(term, str) or not term for term in raw_terms
    ):
        raise ReferenceAssetLibraryError(
            "selection metadata reference_terms are invalid"
        )
    reference_terms = set(raw_terms)
    tree_form_policy = candidate_metadata.get("tree_form_policy")
    if tree_form_policy not in {None, "conifer_or_oak_only"}:
        raise ReferenceAssetLibraryError("selection metadata tree_form_policy is invalid")
    if category == "tree" and tree_form_policy == "conifer_or_oak_only":
        # The measured MNS/MNT family represents woody crowns. The reviewed
        # catalogue also contains orchards, burned trees and decorative species,
        # but this factual forest profile is deliberately limited to the two
        # forms supported by the IGN composition evidence: conifers and oaks.
        desired_tree_forms: set[str] = set()
        if "conifer" in semantic_tags:
            desired_tree_forms.add("conifer")
        if semantic_tags & {"broadleaf", "oak"}:
            desired_tree_forms.add("oak")
        if not desired_tree_forms:
            desired_tree_forms.update(("conifer", "oak"))
        form_compatible = []
        for asset in candidates:
            placement = asset["placement"]
            asset_tags = set(placement["semantic_tags"]) | _runtime_tree_semantic_tags(
                asset
            )
            if desired_tree_forms & asset_tags:
                form_compatible.append(asset)
        if not form_compatible:
            raise ReferenceAssetLibraryError(
                "no conifer or oak USD asset matches the measured tree evidence"
            )
        candidates = form_compatible
    # IGN forest-composition labels describe a form (conifer, oak, mixed), not
    # a precise species.  Once the factual tree-form filter above has been
    # applied, generic terms such as ``foret``, ``fermee`` or ``mixte`` must not
    # collapse a whole forest onto the few catalogue entries that happen to
    # repeat those words.  Keep the term-level tie-breaker for buildings and
    # the other categories, where it carries useful semantic information.
    score_reference_terms = not (
        category == "tree" and tree_form_policy == "conifer_or_oak_only"
    )
    scored: list[tuple[int, Mapping[str, Any]]] = []
    for asset in candidates:
        placement = asset["placement"]
        asset_tags = set(placement["semantic_tags"]) | _runtime_tree_semantic_tags(
            asset
        )
        score = 8 * len(semantic_tags & asset_tags)
        if score_reference_terms:
            score += len(reference_terms & set(placement["reference_terms"]))
        scored.append((score, asset))
    maximum_score = max((score for score, _asset in scored), default=0)
    if maximum_score > 0:
        candidates = [asset for score, asset in scored if score == maximum_score]
    candidates.sort(key=lambda asset: str(asset["asset_id"]))
    if not candidates:
        raise ReferenceAssetLibraryError(
            f"no metadata-compatible expected USD asset for category {category}"
        )
    basis = (
        f"{zone}\x1f{candidate}\x1f{rule_version}\x1f"
        f"{library['catalog_revision']}\x1f{maximum_score}"
    )
    digest = hashlib.sha256(basis.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    selected = candidates[seed % len(candidates)]
    return {
        "asset_id": selected["asset_id"],
        "category": category,
        "selection_seed": seed,
        "usage_status": usage,
        "metadata_match_score": maximum_score,
        "repeatable": True,
    }


def select_asset_for_candidate(
    library: Mapping[str, Any],
    *,
    category: str,
    zone: str,
    candidate: str,
    rule_version: str,
    usage: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select a repeatable compatible asset from a validated catalogue."""

    validate_reference_asset_library(library)
    return _select_asset_for_candidate_from_validated_library(
        library,
        category=category,
        zone=zone,
        candidate=candidate,
        rule_version=rule_version,
        usage=usage,
        metadata=metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-library", type=Path)
    parser.add_argument("--review-batch-root", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--reviewed-source-root", type=Path)
    parser.add_argument("--hunyuan-assets-root", type=Path, action="append", default=[])
    parser.add_argument("--premium-usdz-root", type=Path)
    parser.add_argument("--simready-root", type=Path)
    parser.add_argument("--candidate-inspection", type=Path)
    parser.add_argument("--execute", action="store_true", required=True)
    options = parser.parse_args(argv)
    candidates = discover_candidate_assets(
        options.reference_manifest,
        hunyuan_roots=options.hunyuan_assets_root,
        premium_usdz_root=options.premium_usdz_root,
        simready_root=options.simready_root,
        candidate_inspection=options.candidate_inspection,
    )
    reviewed_library: Path | Mapping[str, Any] = options.reviewed_library or {
        "schema": "fireviewer.asset-library.v1",
        "assets": [],
    }
    payload = build_reference_asset_library(
        options.reference_manifest,
        reviewed_library,
        candidate_assets=candidates,
    )
    result = write_reference_asset_library(
        payload,
        review_batch_root=options.review_batch_root,
        output_catalog=options.output_catalog,
        candidate_assets=candidates,
        reviewed_source_root=options.reviewed_source_root,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALGORITHM",
    "SCHEMA",
    "CandidateAsset",
    "DiscoveredCandidates",
    "ReferenceAssetLibraryError",
    "build_reference_asset_library",
    "discover_candidate_assets",
    "select_asset_for_candidate",
    "validate_reference_asset_library",
    "write_reference_asset_library",
]
