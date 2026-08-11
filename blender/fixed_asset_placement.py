"""Validate and project exact user-authored asset placements.

The public request is deliberately small and fixed.  Users choose one exact
catalogue ``asset_id`` and one WGS84 coordinate.  Production projects the XY to
Lambert-93, assigns the point to exactly one 500 m tile and samples Z later from
the tile MNT.  No user-authored altitude or scale is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from pyproj import Transformer


REQUEST_SCHEMA = "fireviewer.fixed-asset-placement-request.v1"
PROJECTED_SCHEMA = "fireviewer.projected-fixed-asset-placement.v1"
SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:2154"
TILE_SIZE_M = 500
MAX_PLACEMENTS = 5000
PLACEMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
ASSET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
REQUEST_KEYS = frozenset({"schema", "crs", "placements"})
PLACEMENT_KEYS = frozenset(
    {"placement_id", "asset_id", "latitude", "longitude", "yaw_deg"}
)
PROJECTED_KEYS = frozenset(
    {
        "schema",
        "placement_id",
        "asset_id",
        "asset_category",
        "source_wgs84",
        "position_l93_m",
        "owner_tile_origin_l93_m",
        "yaw_rad",
    }
)


class FixedAssetPlacementError(ValueError):
    """The fixed placement request is malformed or cannot be produced."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def schema_path() -> Path:
    return Path(__file__).with_name("fixed_asset_placement_request.v1.schema.json")


def template_path() -> Path:
    return Path(__file__).with_name("fixed_asset_placement_template.v1.json")


def schema_sha256() -> str:
    return hashlib.sha256(schema_path().read_bytes()).hexdigest()


def request_sha256(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(request)).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixedAssetPlacementError(f"{label} doit être un nombre")
    result = float(value)
    if not math.isfinite(result):
        raise FixedAssetPlacementError(f"{label} doit être fini")
    return result


def _asset_index(asset_library: Mapping[str, Any]) -> dict[str, str]:
    assets = asset_library.get("assets")
    if not isinstance(assets, list) or not assets:
        raise FixedAssetPlacementError("Le catalogue des assets est vide ou invalide")
    indexed: dict[str, str] = {}
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise FixedAssetPlacementError(f"Asset {index} invalide dans le catalogue")
        asset_id = asset.get("asset_id")
        category = asset.get("category")
        if (
            not isinstance(asset_id, str)
            or ASSET_ID_RE.fullmatch(asset_id) is None
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise FixedAssetPlacementError(
                f"Identité d'asset invalide à l'index {index}"
            )
        if asset_id in indexed:
            raise FixedAssetPlacementError(
                f"Asset dupliqué dans le catalogue: {asset_id}"
            )
        indexed[asset_id] = category
    declared = asset_library.get("asset_count")
    if declared is not None and declared != len(indexed):
        raise FixedAssetPlacementError(
            "Le compteur du catalogue des assets est incohérent"
        )
    return indexed


def asset_choices(asset_library: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return stable Gradio labels without changing exact asset IDs."""

    indexed = _asset_index(asset_library)
    by_id = {
        str(asset["asset_id"]): asset
        for asset in asset_library["assets"]
        if isinstance(asset, Mapping)
    }
    choices: list[tuple[str, str]] = []
    for asset_id, category in sorted(
        indexed.items(), key=lambda item: (item[1], item[0])
    ):
        reference = by_id[asset_id].get("reference")
        relative = reference.get("path") if isinstance(reference, Mapping) else None
        label = (
            Path(relative).stem if isinstance(relative, str) and relative else asset_id
        )
        label = label.replace("_", " ").replace("-", " ").strip()
        choices.append((f"{category} — {label} [{asset_id}]", asset_id))
    return choices


def normalize_request(
    payload: Mapping[str, Any], asset_library: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact JSON contract and return a canonically ordered copy."""

    if not isinstance(payload, Mapping):
        raise FixedAssetPlacementError("Le JSON de placements doit contenir un objet")
    unknown = sorted(set(payload) - REQUEST_KEYS)
    missing = sorted(REQUEST_KEYS - set(payload))
    if unknown or missing:
        raise FixedAssetPlacementError(
            f"Champs JSON de racine invalides; absents={missing}, inconnus={unknown}"
        )
    if payload.get("schema") != REQUEST_SCHEMA or payload.get("crs") != SOURCE_CRS:
        raise FixedAssetPlacementError(
            f"Le JSON doit utiliser {REQUEST_SCHEMA} en {SOURCE_CRS}"
        )
    placements = payload.get("placements")
    if not isinstance(placements, list) or len(placements) > MAX_PLACEMENTS:
        raise FixedAssetPlacementError(
            f"placements doit être une liste de 0 à {MAX_PLACEMENTS} éléments"
        )
    indexed_assets = _asset_index(asset_library)
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(placements):
        if not isinstance(raw, Mapping):
            raise FixedAssetPlacementError(f"Placement {index} invalide")
        unknown = sorted(set(raw) - PLACEMENT_KEYS)
        missing = sorted(PLACEMENT_KEYS - set(raw))
        if unknown or missing:
            raise FixedAssetPlacementError(
                f"Placement {index}: champs absents={missing}, inconnus={unknown}"
            )
        placement_id = raw.get("placement_id")
        asset_id = raw.get("asset_id")
        if (
            not isinstance(placement_id, str)
            or PLACEMENT_ID_RE.fullmatch(placement_id) is None
        ):
            raise FixedAssetPlacementError(f"placement_id invalide à l'index {index}")
        if placement_id in seen:
            raise FixedAssetPlacementError(f"placement_id dupliqué: {placement_id}")
        seen.add(placement_id)
        if not isinstance(asset_id, str) or asset_id not in indexed_assets:
            raise FixedAssetPlacementError(
                f"asset_id absent du catalogue: {asset_id!r}"
            )
        latitude = _finite_number(raw.get("latitude"), f"placement {index}.latitude")
        longitude = _finite_number(raw.get("longitude"), f"placement {index}.longitude")
        yaw_deg = _finite_number(raw.get("yaw_deg"), f"placement {index}.yaw_deg")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise FixedAssetPlacementError(f"Coordonnées GPS invalides: {placement_id}")
        if not 0 <= yaw_deg < 360:
            raise FixedAssetPlacementError(
                f"yaw_deg doit être compris entre 0 inclus et 360 exclu: {placement_id}"
            )
        normalized.append(
            {
                "placement_id": placement_id,
                "asset_id": asset_id,
                "latitude": latitude,
                "longitude": longitude,
                "yaw_deg": yaw_deg,
            }
        )
    normalized.sort(key=lambda item: item["placement_id"])
    return {"schema": REQUEST_SCHEMA, "crs": SOURCE_CRS, "placements": normalized}


def load_request(path: Path | str, asset_library: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 1_048_576:
        raise FixedAssetPlacementError(
            "Le fichier JSON de placements est absent ou dépasse 1 Mio"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FixedAssetPlacementError(
            f"JSON de placements illisible: {error}"
        ) from error
    return normalize_request(payload, asset_library)


def add_manual_placement(
    request: Mapping[str, Any],
    asset_library: Mapping[str, Any],
    *,
    latitude: Any,
    longitude: Any,
    asset_id: Any,
    yaw_deg: Any = 0,
) -> dict[str, Any]:
    current = normalize_request(request, asset_library)
    if not isinstance(asset_id, str) or not asset_id:
        raise FixedAssetPlacementError("Choisissez un asset dans le catalogue")
    latitude_value = _finite_number(latitude, "latitude du placement")
    longitude_value = _finite_number(longitude, "longitude du placement")
    yaw_value = _finite_number(yaw_deg, "orientation du placement")
    identity = {
        "asset_id": asset_id,
        "latitude": latitude_value,
        "longitude": longitude_value,
        "yaw_deg": yaw_value,
    }
    placement_id = (
        "fixed-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:16]
    )
    if any(row["placement_id"] == placement_id for row in current["placements"]):
        return current
    candidate = {
        "placement_id": placement_id,
        **identity,
    }
    return normalize_request(
        {**current, "placements": [*current["placements"], candidate]}, asset_library
    )


def project_request(
    request: Mapping[str, Any],
    asset_library: Mapping[str, Any],
    *,
    requested_bounds_l93_m: Sequence[float],
    transformer: Transformer | None = None,
) -> tuple[dict[str, Any], ...]:
    normalized = normalize_request(request, asset_library)
    if len(requested_bounds_l93_m) != 4:
        raise FixedAssetPlacementError("L'emprise de production est invalide")
    bounds = tuple(
        _finite_number(value, "borne Lambert-93") for value in requested_bounds_l93_m
    )
    west, south, east, north = bounds
    if east <= west or north <= south:
        raise FixedAssetPlacementError("L'emprise de production est vide")
    indexed_assets = _asset_index(asset_library)
    convert = transformer or Transformer.from_crs(4326, 2154, always_xy=True)
    result: list[dict[str, Any]] = []
    for row in normalized["placements"]:
        x_m, y_m = convert.transform(row["longitude"], row["latitude"])
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise FixedAssetPlacementError(
                f"Projection Lambert-93 impossible: {row['placement_id']}"
            )
        if not (west <= x_m < east and south <= y_m < north):
            raise FixedAssetPlacementError(
                f"Placement hors du carré demandé: {row['placement_id']}"
            )
        x_mm = round(float(x_m), 3)
        y_mm = round(float(y_m), 3)
        owner = (
            math.floor(x_mm / TILE_SIZE_M) * TILE_SIZE_M,
            math.floor(y_mm / TILE_SIZE_M) * TILE_SIZE_M,
        )
        result.append(
            {
                "schema": PROJECTED_SCHEMA,
                "placement_id": row["placement_id"],
                "asset_id": row["asset_id"],
                "asset_category": indexed_assets[row["asset_id"]],
                "source_wgs84": [row["latitude"], row["longitude"]],
                "position_l93_m": [x_mm, y_mm],
                "owner_tile_origin_l93_m": list(owner),
                "yaw_rad": round(math.radians(row["yaw_deg"]), 9),
            }
        )
    result.sort(key=lambda item: item["placement_id"])
    return tuple(result)


def validate_projected_placements(
    placements: Sequence[Mapping[str, Any]],
    *,
    tile_origin_l93_m: Sequence[float] | None = None,
) -> tuple[dict[str, Any], ...]:
    expected_origin: tuple[int, int] | None = None
    if tile_origin_l93_m is not None:
        if len(tile_origin_l93_m) != 2:
            raise FixedAssetPlacementError("Origine de tuile invalide")
        raw_origin = tuple(
            _finite_number(value, "origine de tuile") for value in tile_origin_l93_m
        )
        if any(
            not value.is_integer() or int(value) % TILE_SIZE_M for value in raw_origin
        ):
            raise FixedAssetPlacementError("Origine de tuile non alignée sur 500 m")
        expected_origin = (int(raw_origin[0]), int(raw_origin[1]))
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(placements):
        if not isinstance(raw, Mapping) or set(raw) != PROJECTED_KEYS:
            raise FixedAssetPlacementError(f"Placement projeté {index} invalide")
        placement_id = raw.get("placement_id")
        asset_id = raw.get("asset_id")
        category = raw.get("asset_category")
        if (
            not isinstance(placement_id, str)
            or PLACEMENT_ID_RE.fullmatch(placement_id) is None
            or placement_id in seen
            or not isinstance(asset_id, str)
            or ASSET_ID_RE.fullmatch(asset_id) is None
            or not isinstance(category, str)
            or not category
        ):
            raise FixedAssetPlacementError(
                f"Identité projetée invalide à l'index {index}"
            )
        seen.add(placement_id)
        source = raw.get("source_wgs84")
        position = raw.get("position_l93_m")
        owner = raw.get("owner_tile_origin_l93_m")
        if not isinstance(source, list) or len(source) != 2:
            raise FixedAssetPlacementError(f"Source WGS84 invalide: {placement_id}")
        if not isinstance(position, list) or len(position) != 2:
            raise FixedAssetPlacementError(
                f"Position Lambert-93 invalide: {placement_id}"
            )
        if not isinstance(owner, list) or len(owner) != 2:
            raise FixedAssetPlacementError(
                f"Propriétaire de tuile invalide: {placement_id}"
            )
        latitude = _finite_number(source[0], "latitude source")
        longitude = _finite_number(source[1], "longitude source")
        x_m = _finite_number(position[0], "position X")
        y_m = _finite_number(position[1], "position Y")
        owner_values = tuple(
            _finite_number(value, "origine propriétaire") for value in owner
        )
        yaw = _finite_number(raw.get("yaw_rad"), "yaw projeté")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise FixedAssetPlacementError(f"Source WGS84 invalide: {placement_id}")
        if any(
            not value.is_integer() or int(value) % TILE_SIZE_M for value in owner_values
        ):
            raise FixedAssetPlacementError(f"Propriétaire non aligné: {placement_id}")
        owner_tuple = (int(owner_values[0]), int(owner_values[1]))
        if not (
            owner_tuple[0] <= x_m < owner_tuple[0] + TILE_SIZE_M
            and owner_tuple[1] <= y_m < owner_tuple[1] + TILE_SIZE_M
        ):
            raise FixedAssetPlacementError(
                f"Propriétaire spatial incohérent: {placement_id}"
            )
        if expected_origin is not None and owner_tuple != expected_origin:
            raise FixedAssetPlacementError(
                f"Placement attribué à une autre tuile: {placement_id}"
            )
        if not 0 <= yaw < 2 * math.pi:
            raise FixedAssetPlacementError(
                f"Orientation projetée invalide: {placement_id}"
            )
        normalized.append(
            {
                "schema": PROJECTED_SCHEMA,
                "placement_id": placement_id,
                "asset_id": asset_id,
                "asset_category": category,
                "source_wgs84": [latitude, longitude],
                "position_l93_m": [x_m, y_m],
                "owner_tile_origin_l93_m": list(owner_tuple),
                "yaw_rad": yaw,
            }
        )
    normalized.sort(key=lambda item: item["placement_id"])
    return tuple(normalized)


def rows_for_ui(request: Mapping[str, Any]) -> list[list[Any]]:
    placements = request.get("placements") if isinstance(request, Mapping) else None
    if not isinstance(placements, list):
        return []
    return [
        [
            row.get("placement_id"),
            row.get("asset_id"),
            row.get("latitude"),
            row.get("longitude"),
            row.get("yaw_deg"),
        ]
        for row in placements
        if isinstance(row, Mapping)
    ]


EMPTY_REQUEST = {"schema": REQUEST_SCHEMA, "crs": SOURCE_CRS, "placements": []}


__all__ = [
    "EMPTY_REQUEST",
    "FixedAssetPlacementError",
    "PROJECTED_SCHEMA",
    "REQUEST_SCHEMA",
    "SOURCE_CRS",
    "add_manual_placement",
    "asset_choices",
    "canonical_json_bytes",
    "load_request",
    "normalize_request",
    "project_request",
    "request_sha256",
    "rows_for_ui",
    "schema_path",
    "schema_sha256",
    "template_path",
    "validate_projected_placements",
]
