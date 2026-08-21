"""Download one deterministic BD TOPO context snapshot for a measured zone.

The snapshot is deliberately plain JSON.  It replaces the prebuilt GeoPackage
dependency for on-demand zones while keeping BD TOPO as semantic confirmation:
MNT/MNS still determine the measured positions and heights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlencode

import requests
from shapely import make_valid
from shapely.geometry import mapping, shape

SCHEMA = "fireviewer.simple-measured-zone-context.v1"
STATUS = "downloaded_verified"
CRS = "EPSG:2154"
WFS_ENDPOINT = "https://data.geopf.fr/wfs/ows"
PAGE_SIZE = 5_000
LAYERS = {
    "buildings": "BDTOPO_V3:batiment",
    "vegetation": "BDTOPO_V3:zone_de_vegetation",
    "forest_composition": "BDFORETV1_BDD_FXX_LAMB93_20140403:resu_bdv1_shape",
    "roads": "BDTOPO_V3:troncon_de_route",
    "rail": "BDTOPO_V3:troncon_de_voie_ferree",
    "hydro_lines": "BDTOPO_V3:troncon_hydrographique",
    "hydro_surfaces": "BDTOPO_V3:surface_hydrographique",
}
SORT_FIELDS = {
    LAYERS["forest_composition"]: "dep,tfifn,typn",
}


class SimpleMeasuredZoneContextError(RuntimeError):
    """The WFS response cannot form a reproducible zone context."""


@dataclass(frozen=True, slots=True)
class ZoneContext:
    path: Path
    content_sha256: str
    feature_counts: Mapping[str, int]
    reused: bool


HttpGet = Callable[[str], bytes]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_output(path: Path | str) -> Path:
    lexical = PureWindowsPath(str(path))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise SimpleMeasuredZoneContextError(
            f"Zone context output must stay on D: on Windows: {path}"
        )
    resolved = Path(path).resolve(strict=False)
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise SimpleMeasuredZoneContextError(
            f"Zone context output must stay on D: on Windows: {resolved}"
        )
    return resolved


def _bounds(values: Sequence[float]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise SimpleMeasuredZoneContextError("Bounds need west, south, east, north")
    floats = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or not value.is_integer() for value in floats):
        raise SimpleMeasuredZoneContextError("Bounds must use finite integer metres")
    west, south, east, north = (int(value) for value in floats)
    if west >= east or south >= north:
        raise SimpleMeasuredZoneContextError("Zone bounds are empty")
    return west, south, east, north


def wfs_url(
    layer: str,
    bounds_l93_m: Sequence[int],
    *,
    start_index: int,
    count: int = PAGE_SIZE,
) -> str:
    """Return the canonical paged GeoPlatform WFS request."""

    west, south, east, north = _bounds(bounds_l93_m)
    if layer not in LAYERS.values():
        raise SimpleMeasuredZoneContextError(f"Unsupported WFS layer: {layer}")
    if start_index < 0 or count < 1 or count > PAGE_SIZE:
        raise SimpleMeasuredZoneContextError("Invalid WFS page")
    parameters = [
        ("SERVICE", "WFS"),
        ("VERSION", "2.0.0"),
        ("REQUEST", "GetFeature"),
        ("TYPENAMES", layer),
        ("SRSNAME", CRS),
        ("BBOX", f"{west},{south},{east},{north},{CRS}"),
        ("OUTPUTFORMAT", "application/json"),
        ("SORTBY", SORT_FIELDS.get(layer, "cleabs")),
        ("STARTINDEX", str(start_index)),
        ("COUNT", str(count)),
    ]
    return f"{WFS_ENDPOINT}?{urlencode(parameters)}"


def _default_http_get(url: str) -> bytes:
    response = requests.get(
        url,
        headers={"User-Agent": "FireViewer/simple-measured-zone-context-v1"},
        timeout=(20, 180),
    )
    response.raise_for_status()
    if not response.content:
        raise SimpleMeasuredZoneContextError(f"Empty WFS response: {url}")
    return bytes(response.content)


def _read_page(payload: bytes, layer: str) -> tuple[list[dict[str, Any]], int | None]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimpleMeasuredZoneContextError(
            f"Invalid GeoJSON response for {layer}"
        ) from error
    features = decoded.get("features") if isinstance(decoded, Mapping) else None
    if decoded.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise SimpleMeasuredZoneContextError(
            f"WFS response for {layer} is not a FeatureCollection"
        )
    matched = decoded.get("numberMatched")
    if matched in (None, "unknown"):
        matched_count = None
    elif isinstance(matched, int) and matched >= 0:
        matched_count = matched
    else:
        raise SimpleMeasuredZoneContextError(
            f"WFS numberMatched is invalid for {layer}"
        )
    return features, matched_count


def _normalize_feature(feature: Any, layer: str) -> dict[str, Any]:
    if not isinstance(feature, Mapping) or not isinstance(
        feature.get("geometry"), Mapping
    ):
        raise SimpleMeasuredZoneContextError(f"Malformed feature in {layer}")
    properties = feature.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    source_id = properties.get("cleabs") or feature.get("id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise SimpleMeasuredZoneContextError(f"Feature without stable ID in {layer}")
    try:
        geometry = shape(feature["geometry"])
    except (TypeError, ValueError) as error:
        raise SimpleMeasuredZoneContextError(
            f"Invalid feature geometry in {layer}: {source_id}"
        ) from error
    if geometry.is_empty:
        raise SimpleMeasuredZoneContextError(
            f"Empty feature geometry in {layer}: {source_id}"
        )
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if geometry.is_empty or not geometry.is_valid:
        raise SimpleMeasuredZoneContextError(
            f"Unrepairable feature geometry in {layer}: {source_id}"
        )
    normalized_properties = {
        str(key): value
        for key, value in sorted(properties.items(), key=lambda item: str(item[0]))
        if value is None or isinstance(value, (str, int, float, bool))
    }
    normalized_geometry = json.loads(
        json.dumps(mapping(geometry), allow_nan=False, separators=(",", ":"))
    )
    return {
        "type": "Feature",
        "id": source_id,
        "properties": normalized_properties,
        "geometry": normalized_geometry,
    }


def _download_layer(
    layer: str,
    bounds_l93_m: Sequence[int],
    getter: HttpGet,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    pages: list[dict[str, Any]] = []
    start = 0
    matched: int | None = None
    while True:
        url = wfs_url(layer, bounds_l93_m, start_index=start)
        payload = getter(url)
        features, page_matched = _read_page(payload, layer)
        if matched is None:
            matched = page_matched
        elif page_matched is not None and page_matched != matched:
            raise SimpleMeasuredZoneContextError(
                f"WFS feature count changed during download: {layer}"
            )
        for feature in features:
            normalized = _normalize_feature(feature, layer)
            source_id = normalized["id"]
            if source_id in records:
                raise SimpleMeasuredZoneContextError(
                    f"Duplicate WFS feature ID in {layer}: {source_id}"
                )
            records[source_id] = normalized
        pages.append(
            {
                "request_url": url,
                "response_byte_count": len(payload),
                "response_sha256": _sha256_bytes(payload),
                "feature_count": len(features),
            }
        )
        start += len(features)
        if not features or len(features) < PAGE_SIZE:
            break
        if matched is not None and start >= matched:
            break
    if matched is not None and len(records) != matched:
        raise SimpleMeasuredZoneContextError(
            f"WFS returned {len(records)} of {matched} features for {layer}"
        )
    return [records[key] for key in sorted(records)], pages


def validate_zone_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one canonical zone context without accessing a database."""

    if payload.get("schema") != SCHEMA or payload.get("status") != STATUS:
        raise SimpleMeasuredZoneContextError("Zone context schema/status is invalid")
    if payload.get("crs") != CRS:
        raise SimpleMeasuredZoneContextError("Zone context CRS is invalid")
    _bounds(payload.get("bounds_l93_m", ()))
    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != set(LAYERS):
        raise SimpleMeasuredZoneContextError("Zone context layers are incomplete")
    counts: dict[str, int] = {}
    for role, typename in LAYERS.items():
        record = layers.get(role)
        if not isinstance(record, Mapping) or record.get("typename") != typename:
            raise SimpleMeasuredZoneContextError(f"Invalid zone layer: {role}")
        features = record.get("features")
        pages = record.get("pages")
        if not isinstance(features, list) or not isinstance(pages, list):
            raise SimpleMeasuredZoneContextError(f"Invalid zone data: {role}")
        normalized = [_normalize_feature(feature, typename) for feature in features]
        if normalized != features:
            raise SimpleMeasuredZoneContextError(
                f"Zone features are not canonical: {role}"
            )
        ids = [feature["id"] for feature in features]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise SimpleMeasuredZoneContextError(
                f"Zone feature IDs are not unique and sorted: {role}"
            )
        if record.get("feature_count") != len(features):
            raise SimpleMeasuredZoneContextError(f"Zone feature count differs: {role}")
        counts[role] = len(features)
    if payload.get("feature_counts") != counts:
        raise SimpleMeasuredZoneContextError("Zone context counts differ")
    declared = payload.get("content_sha256")
    without_hash = dict(payload)
    without_hash.pop("content_sha256", None)
    if declared != _sha256_bytes(_canonical_bytes(without_hash)):
        raise SimpleMeasuredZoneContextError("Zone context content hash is invalid")
    return dict(payload)


def load_zone_context(path: Path | str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimpleMeasuredZoneContextError(
            f"Invalid zone context: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SimpleMeasuredZoneContextError("Zone context must be a JSON object")
    return validate_zone_context(payload)


def prepare_zone_context(
    *,
    output_path: Path | str,
    zone_id: str,
    bounds_l93_m: Sequence[float],
    source_revision: str,
    http_get: HttpGet | None = None,
) -> ZoneContext:
    """Download all national context layers once and atomically publish them."""

    destination = _require_output(output_path)
    bounds = _bounds(bounds_l93_m)
    if not isinstance(zone_id, str) or not zone_id.strip():
        raise SimpleMeasuredZoneContextError("zone_id is required")
    if (
        not isinstance(source_revision, str)
        or not source_revision.strip()
        or source_revision != source_revision.strip()
    ):
        raise SimpleMeasuredZoneContextError("A stable source revision is required")
    if destination.exists():
        existing = load_zone_context(destination)
        if (
            existing.get("zone_id") != zone_id
            or existing.get("bounds_l93_m") != list(bounds)
            or existing.get("source", {}).get("revision") != source_revision
        ):
            raise SimpleMeasuredZoneContextError(
                "Existing zone context belongs to another request"
            )
        return ZoneContext(
            destination,
            existing["content_sha256"],
            existing["feature_counts"],
            True,
        )

    getter = http_get or _default_http_get
    layers: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for role, typename in LAYERS.items():
        features, pages = _download_layer(typename, bounds, getter)
        layers[role] = {
            "typename": typename,
            "feature_count": len(features),
            "features": features,
            "pages": pages,
        }
        counts[role] = len(features)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "zone_id": zone_id,
        "crs": CRS,
        "bounds_l93_m": list(bounds),
        "source": {
            "provider": "IGN Geoplateforme",
            "service": "WFS 2.0.0",
            "endpoint": WFS_ENDPOINT,
            "revision": source_revision,
        },
        "feature_counts": counts,
        "layers": layers,
    }
    payload["content_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    validate_zone_context(payload)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.part")
    if staging.exists():
        if staging.parent != destination.parent:
            raise SimpleMeasuredZoneContextError("Unsafe zone-context staging path")
        if staging.is_dir():
            shutil.rmtree(staging)
        else:
            staging.unlink()
    staging.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(staging, destination)
    loaded = load_zone_context(destination)
    return ZoneContext(
        destination,
        loaded["content_sha256"],
        loaded["feature_counts"],
        False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--bounds", nargs=4, type=float, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    options = parser.parse_args(argv)
    result = prepare_zone_context(
        output_path=options.output,
        zone_id=options.zone_id,
        bounds_l93_m=options.bounds,
        source_revision=options.source_revision,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output": str(result.path),
                "content_sha256": result.content_sha256,
                "feature_counts": result.feature_counts,
                "reused": result.reused,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CRS",
    "LAYERS",
    "PAGE_SIZE",
    "SCHEMA",
    "SimpleMeasuredZoneContextError",
    "ZoneContext",
    "load_zone_context",
    "main",
    "prepare_zone_context",
    "validate_zone_context",
    "wfs_url",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
