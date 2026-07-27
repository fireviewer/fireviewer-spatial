#!/usr/bin/env python3
"""Validate one detail-zone source set and write a portable provenance manifest.

The command does not download source payloads. It verifies previously acquired
official IGN files, re-discovers COPC URLs through the official WFS, reads the
WMS capabilities, scans every LiDAR classification, and records only paths
relative to the manifest directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import rasterio
from pyproj import Transformer


WFS_URL = "https://data.geopf.fr/wfs/ows"
WFS_TYPENAME = "IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle"
WMS_URL = "https://data.geopf.fr/wms-r"
WMS_CAPABILITIES_URL = f"{WMS_URL}?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0"
USER_AGENT = "FireWarning-source-validation/1.0"
EXPECTED_ZONE = "montmaur"
EXPECTED_TILE_COUNT = 4
RASTER_PRODUCTS = {
    "mnt": "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93",
    "mns": "IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93",
}
CLASS_LABELS = {
    0: "created_never_classified",
    1: "unclassified",
    2: "ground",
    3: "low_vegetation",
    4: "medium_vegetation",
    5: "high_vegetation",
    6: "building",
    7: "low_point_noise",
    8: "reserved",
    9: "water",
    10: "rail",
    11: "road_surface",
    12: "reserved",
    13: "wire_guard",
    14: "wire_conductor",
    15: "transmission_tower",
    16: "wire_structure_connector",
    17: "bridge_deck",
    18: "high_noise",
    19: "overhead_structure",
    20: "ignored_ground",
    21: "snow",
    22: "temporal_exclusion",
}


class SourceValidationError(RuntimeError):
    pass


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceValidationError(f"JSON illisible {path}: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fetch_bytes(url: str, *, attempts: int = 4) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=180) as response:
                headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
                return response.read(), headers
        except Exception as exc:  # urllib exposes several transport exception types
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise SourceValidationError(f"Lecture distante impossible {url}: {last_error}")


def remote_content_length(url: str) -> int:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                url,
                method="HEAD",
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                value = response.headers.get("Content-Length")
                if value is None:
                    raise SourceValidationError(f"Content-Length absent pour {url}")
                return int(value)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise SourceValidationError(f"HEAD impossible {url}: {last_error}")


def load_zone(
    contract_path: Path, zone_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(contract_path)
    matches = [zone for zone in contract.get("zones", []) if zone.get("id") == zone_id]
    if len(matches) != 1:
        raise SourceValidationError(
            f"La zone {zone_id!r} doit apparaître exactement une fois dans {contract_path}."
        )
    if zone_id != EXPECTED_ZONE:
        raise SourceValidationError(
            f"Cette acquisition est volontairement limitée à {EXPECTED_ZONE!r}; reçu {zone_id!r}."
        )
    zone = matches[0]
    tiles = zone.get("ign_lidar_hd_kilometre_tiles", [])
    if len(tiles) != EXPECTED_TILE_COUNT or len(set(tiles)) != EXPECTED_TILE_COUNT:
        raise SourceValidationError(
            "Le carré Montmaur doit référencer quatre dalles IGN distinctes."
        )
    bounds = zone.get("bounds_l93_metres", [])
    if (
        len(bounds) != 4
        or not math.isclose(bounds[2] - bounds[0], 1000.0)
        or not math.isclose(bounds[3] - bounds[1], 1000.0)
    ):
        raise SourceValidationError("L'emprise Montmaur doit être un carré de 1 000 m.")
    return contract, zone


def tile_bounds(tile: str) -> list[float]:
    try:
        east_text, north_text = tile.split("_")
        east = int(east_text) * 1000
        north = int(north_text) * 1000
    except (ValueError, AttributeError) as exc:
        raise SourceValidationError(
            f"Identifiant de dalle IGN invalide: {tile}"
        ) from exc
    # IGN names the point/raster tile from its north-west index.
    return [float(east), float(north - 1000), float(east + 1000), float(north)]


def discovery(zone: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    source_bounds = [tile_bounds(tile) for tile in zone["ign_lidar_hd_kilometre_tiles"]]
    west = min(bounds[0] for bounds in source_bounds)
    south = min(bounds[1] for bounds in source_bounds)
    east = max(bounds[2] for bounds in source_bounds)
    north = max(bounds[3] for bounds in source_bounds)
    to_wgs84 = Transformer.from_crs(2154, 4326, always_xy=True)
    corners = [to_wgs84.transform(x, y) for x in (west, east) for y in (south, north)]
    bbox = [
        min(point[0] for point in corners) - 0.001,
        min(point[1] for point in corners) - 0.001,
        max(point[0] for point in corners) + 0.001,
        max(point[1] for point in corners) + 0.001,
    ]
    query = urllib.parse.urlencode(
        {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": WFS_TYPENAME,
            "OUTPUTFORMAT": "application/json",
            "SRSNAME": "EPSG:4326",
            "BBOX": ",".join(str(value) for value in bbox) + ",EPSG:4326",
            "COUNT": "50",
        }
    )
    url = f"{WFS_URL}?{query}"
    raw, _ = fetch_bytes(url)
    try:
        collection = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SourceValidationError(f"Réponse WFS non JSON: {exc}") from exc
    by_name = {
        str(feature.get("properties", {}).get("name")): feature.get("properties", {})
        for feature in collection.get("features", [])
    }
    result: dict[str, dict[str, Any]] = {}
    for tile in zone["ign_lidar_hd_kilometre_tiles"]:
        expected = f"LHD_FXX_{tile}_PTS_O_LAMB93_IGN69"
        properties = by_name.get(expected)
        if properties is None:
            raise SourceValidationError(
                f"Dalle COPC absente du WFS officiel: {expected}"
            )
        result[tile] = properties
    return url, result


def wms_capabilities() -> dict[str, dict[str, Any]]:
    raw, headers = fetch_bytes(WMS_CAPABILITIES_URL)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise SourceValidationError(f"GetCapabilities WMS invalide: {exc}") from exc
    namespace = {"w": "http://www.opengis.net/wms"}
    layers: dict[str, dict[str, Any]] = {}
    wanted = set(RASTER_PRODUCTS.values())
    for layer in root.findall(".//w:Layer", namespace):
        name = layer.findtext("w:Name", namespaces=namespace)
        if name not in wanted:
            continue
        minimum_scale = layer.findtext("w:MinScaleDenominator", namespaces=namespace)
        if minimum_scale is None:
            raise SourceValidationError(f"MinScaleDenominator absent pour {name}")
        scale = float(minimum_scale)
        layers[name] = {
            "name": name,
            "title": layer.findtext("w:Title", namespaces=namespace),
            "abstract": layer.findtext("w:Abstract", namespaces=namespace),
            "min_scale_denominator": scale,
            "wms_standard_pixel_size_metres": 0.00028,
            "advertised_finest_ground_pixel_metres": scale * 0.00028,
        }
    if set(layers) != wanted:
        raise SourceValidationError(
            f"Couches MNT/MNS absentes des capacités WMS: {sorted(wanted.difference(layers))}"
        )
    return {
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "content_type": headers.get("content-type"),
        "layers": layers,
    }


def class_label(class_id: int) -> str:
    if class_id in CLASS_LABELS:
        return CLASS_LABELS[class_id]
    return "user_defined" if class_id >= 64 else "unassigned_or_reserved"


def inspect_copc(
    path: Path,
    root: Path,
    tile: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    expected_bounds = tile_bounds(tile)
    with laspy.open(path) as reader:
        header = reader.header
        crs = header.parse_crs()
        copc_vlrs = [
            vlr for vlr in header.vlrs if vlr.user_id == "copc" and vlr.record_id == 1
        ]
        if len(copc_vlrs) != 1:
            raise SourceValidationError(f"VLR COPC absent ou dupliqué: {path.name}")
        if header.version.minor != 4 or header.point_format.id != 6:
            raise SourceValidationError(
                f"Format LAS inattendu {path.name}: {header.version}, point format {header.point_format.id}."
            )
        if crs is None or crs.to_epsg() != 2154:
            raise SourceValidationError(f"CRS COPC inattendu {path.name}: {crs}")
        if not np.allclose(header.scales, [0.01, 0.01, 0.01], rtol=0.0, atol=1e-12):
            raise SourceValidationError(
                f"Quantification COPC inattendue {path.name}: {header.scales}"
            )
        if not np.allclose(header.offsets, [0.0, 0.0, 0.0], rtol=0.0, atol=1e-12):
            raise SourceValidationError(
                f"Offsets COPC inattendus {path.name}: {header.offsets}"
            )
        if not np.allclose(
            header.mins[:2], expected_bounds[:2], rtol=0.0, atol=0.01
        ) or not np.allclose(header.maxs[:2], expected_bounds[2:], rtol=0.0, atol=0.01):
            raise SourceValidationError(
                f"Emprise COPC inattendue {path.name}: {header.mins[:2]} / {header.maxs[:2]}"
            )

        classes: Counter[int] = Counter()
        scanned_points = 0
        for points in reader.chunk_iterator(2_000_000):
            values = np.asarray(points.classification, dtype=np.uint8)
            ids, counts = np.unique(values, return_counts=True)
            classes.update({int(key): int(count) for key, count in zip(ids, counts)})
            scanned_points += len(points)
        if (
            scanned_points != header.point_count
            or sum(classes.values()) != header.point_count
        ):
            raise SourceValidationError(
                f"Scan de classifications incomplet {path.name}: {scanned_points}/{header.point_count}."
            )

        source_url = str(properties.get("url", ""))
        if not source_url.startswith("https://data.geopf.fr/"):
            raise SourceValidationError(
                f"URL COPC non officielle pour {tile}: {source_url}"
            )
        remote_size = remote_content_length(source_url)
        if remote_size != path.stat().st_size:
            raise SourceValidationError(
                f"Taille locale/distante différente {path.name}: {path.stat().st_size}/{remote_size}."
            )
        metadata_raw = properties.get("metadata") or "{}"
        try:
            source_metadata = (
                json.loads(metadata_raw)
                if isinstance(metadata_raw, str)
                else metadata_raw
            )
        except json.JSONDecodeError:
            source_metadata = {"unparsed": metadata_raw}

        return {
            "tile_id": tile,
            "path": path.relative_to(root).as_posix(),
            "source_url": source_url,
            "source_name": properties.get("name"),
            "source_download_name": properties.get("name_download"),
            "source_timestamp": properties.get("timestamp"),
            "source_metadata": source_metadata,
            "format": "COPC.LAZ",
            "las_version": str(header.version),
            "point_format": header.point_format.id,
            "point_count": int(header.point_count),
            "point_density_per_square_metre": float(header.point_count / 1_000_000.0),
            "crs": "EPSG:2154",
            "vertical_reference": "NGF-IGN69",
            "bounds_l93_metres": [
                float(header.mins[0]),
                float(header.mins[1]),
                float(header.mins[2]),
                float(header.maxs[0]),
                float(header.maxs[1]),
                float(header.maxs[2]),
            ],
            "coordinate_quantization_metres": [float(value) for value in header.scales],
            "copc_vlr_verified": True,
            "classification_scan": "complete",
            "observed_classifications": [
                {"id": class_id, "label": class_label(class_id), "count": count}
                for class_id, count in sorted(classes.items())
            ],
            "byte_count": path.stat().st_size,
            "remote_byte_count": remote_size,
            "sha256": sha256_file(path),
        }


def raster_url(product: str, tile: str) -> str:
    bounds = tile_bounds(tile)
    # Request native sample centres on the 0.50 m IGN grid. The 0.25 m shift
    # avoids an unnecessary half-pixel server resampling.
    request_bounds = [
        bounds[0] - 0.25,
        bounds[1] + 0.25,
        bounds[2] - 0.25,
        bounds[3] + 0.25,
    ]
    filename = f"LHD_FXX_{tile}_{product.upper()}_O_0M50_LAMB93_IGN69.tif"
    query = urllib.parse.urlencode(
        {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "EXCEPTIONS": "text/xml",
            "REQUEST": "GetMap",
            "LAYERS": RASTER_PRODUCTS[product],
            "FORMAT": "image/geotiff",
            "STYLES": "",
            "CRS": "EPSG:2154",
            "BBOX": ",".join(str(value) for value in request_bounds),
            "WIDTH": "2000",
            "HEIGHT": "2000",
            "FILENAME": filename,
        }
    )
    return f"{WMS_URL}?{query}"


def is_wms_lambert93(crs: rasterio.crs.CRS | None) -> tuple[bool, dict[str, Any]]:
    if crs is None:
        return False, {}
    parameters = crs.to_dict()
    expected = {
        "proj": "lcc",
        "lat_0": 46.5,
        "lon_0": 3.0,
        "lat_1": 49.0,
        "lat_2": 44.0,
        "x_0": 700000.0,
        "y_0": 6600000.0,
        "units": "m",
    }
    valid = all(
        parameters.get(key) == value
        if isinstance(value, str)
        else parameters.get(key) is not None
        and abs(float(parameters[key]) - value) <= 1e-9
        for key, value in expected.items()
    )
    return valid, parameters


def inspect_raster(path: Path, root: Path, product: str, tile: str) -> dict[str, Any]:
    expected_bounds = tile_bounds(tile)
    shifted_bounds = [
        expected_bounds[0] - 0.25,
        expected_bounds[1] + 0.25,
        expected_bounds[2] - 0.25,
        expected_bounds[3] + 0.25,
    ]
    try:
        with rasterio.open(path) as dataset:
            crs_valid, crs_parameters = is_wms_lambert93(dataset.crs)
            if not crs_valid:
                raise SourceValidationError(
                    f"Projection WMS-R inattendue {path.name}: {dataset.crs}"
                )
            if (
                dataset.driver != "GTiff"
                or dataset.count != 1
                or dataset.dtypes[0] != "float32"
            ):
                raise SourceValidationError(
                    f"Format raster inattendu: {path.name} / {dataset.profile}"
                )
            if dataset.width != 2000 or dataset.height != 2000:
                raise SourceValidationError(
                    f"Dimensions raster incompatibles avec 0,50 m: {path.name} {dataset.width}x{dataset.height}."
                )
            resolution = [
                abs(float(dataset.transform.a)),
                abs(float(dataset.transform.e)),
            ]
            if not np.allclose(resolution, [0.5, 0.5], rtol=0.0, atol=1e-12):
                raise SourceValidationError(
                    f"Résolution raster non conforme {path.name}: {resolution}"
                )
            if not np.allclose(dataset.bounds, shifted_bounds, rtol=0.0, atol=1e-9):
                raise SourceValidationError(
                    f"Emprise raster inattendue {path.name}: {dataset.bounds}"
                )
            values = dataset.read(1, masked=True)
            if values.count() == 0:
                compressed = values.compressed()
                return {
                    "tile_id": tile,
                    "path": path.relative_to(root).as_posix(),
                    "source_url": raster_url(product, tile),
                    "source_layer": RASTER_PRODUCTS[product],
                    "source_request_crs": "EPSG:2154",
                    "format": "GeoTIFF float32",
                    "width": dataset.width,
                    "height": dataset.height,
                    "observed_pixel_size_metres": resolution,
                    "observed_crs_to_epsg": dataset.crs.to_epsg()
                    if dataset.crs
                    else None,
                    "observed_crs_wkt": dataset.crs.to_wkt() if dataset.crs else None,
                    "observed_projection_parameters": crs_parameters,
                    "crs_interpretation": (
                        "Lambert-93 parameters match the EPSG:2154 WMS request; the returned WKT omits "
                        "the RGF93 datum name and therefore GDAL does not resolve an EPSG code."
                    ),
                    "vertical_reference": "NGF-IGN69",
                    "bounds_l93_metres": [float(value) for value in dataset.bounds],
                    "native_sample_centres_policy": "WMS bbox shifted 0.25 m to preserve the 0.50 m grid",
                    "nodata": dataset.nodata,
                    "valid_sample_count": 0,
                    "nodata_sample_count": int(values.size),
                    "elevation_min_metres": None,
                    "elevation_max_metres": None,
                    "elevation_mean_metres": None,
                    "byte_count": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            compressed = values.compressed()
            return {
                "tile_id": tile,
                "path": path.relative_to(root).as_posix(),
                "source_url": raster_url(product, tile),
                "source_layer": RASTER_PRODUCTS[product],
                "source_request_crs": "EPSG:2154",
                "format": "GeoTIFF float32",
                "width": dataset.width,
                "height": dataset.height,
                "observed_pixel_size_metres": resolution,
                "observed_crs_to_epsg": dataset.crs.to_epsg() if dataset.crs else None,
                "observed_crs_wkt": dataset.crs.to_wkt() if dataset.crs else None,
                "observed_projection_parameters": crs_parameters,
                "crs_interpretation": (
                    "Lambert-93 parameters match the EPSG:2154 WMS request; the returned WKT omits "
                    "the RGF93 datum name and therefore GDAL does not resolve an EPSG code."
                ),
                "vertical_reference": "NGF-IGN69",
                "bounds_l93_metres": [float(value) for value in dataset.bounds],
                "native_sample_centres_policy": "WMS bbox shifted 0.25 m to preserve the 0.50 m grid",
                "nodata": dataset.nodata,
                "valid_sample_count": int(values.count()),
                "nodata_sample_count": int(values.size - values.count()),
                "elevation_min_metres": float(compressed.min()),
                "elevation_max_metres": float(compressed.max()),
                "elevation_mean_metres": float(compressed.mean()),
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    except rasterio.errors.RasterioIOError as exc:
        raise SourceValidationError(f"GeoTIFF illisible {path}: {exc}") from exc


def expected_path(root: Path, category: str, tile: str) -> Path:
    if category == "copc":
        name = f"LHD_FXX_{tile}_PTS_LAMB93_IGN69.copc.laz"
    else:
        name = f"LHD_FXX_{tile}_{category.upper()}_O_0M50_LAMB93_IGN69.tif"
    path = root / category / name
    if not path.is_file():
        raise SourceValidationError(f"Source attendue absente: {path}")
    return path


def build_manifest(contract_path: Path, zone_id: str, root: Path) -> dict[str, Any]:
    contract, zone = load_zone(contract_path, zone_id)
    discovery_url, discovered = discovery(zone)
    capabilities = wms_capabilities()
    tiles = zone["ign_lidar_hd_kilometre_tiles"]

    copc_records = []
    for index, tile in enumerate(tiles):
        if index:
            time.sleep(1.05)  # official endpoint advertises one request/second
        copc_records.append(
            inspect_copc(
                expected_path(root, "copc", tile), root, tile, discovered[tile]
            )
        )
    raster_records: dict[str, list[dict[str, Any]]] = {"mnt": [], "mns": []}
    for product in ("mnt", "mns"):
        for tile in tiles:
            raster_records[product].append(
                inspect_raster(expected_path(root, product, tile), root, product, tile)
            )

    detail_bounds = zone["bounds_l93_metres"]
    raster_union = [
        min(record["bounds_l93_metres"][0] for record in raster_records["mnt"]),
        min(record["bounds_l93_metres"][1] for record in raster_records["mnt"]),
        max(record["bounds_l93_metres"][2] for record in raster_records["mnt"]),
        max(record["bounds_l93_metres"][3] for record in raster_records["mnt"]),
    ]
    covers_detail = (
        raster_union[0] <= detail_bounds[0]
        and raster_union[1] <= detail_bounds[1]
        and raster_union[2] >= detail_bounds[2]
        and raster_union[3] >= detail_bounds[3]
    )
    if not covers_detail:
        raise SourceValidationError(
            "Les quatre rasters ne couvrent pas le carré détail Montmaur."
        )

    all_records = copc_records + raster_records["mnt"] + raster_records["mns"]
    return {
        "schema_version": "1.0",
        "manifest_role": "portable_official_source_inventory",
        "generated_at": utc_now(),
        "parent_package_id": contract.get("parent_package_id"),
        "zone": {
            "id": zone["id"],
            "label": zone["label"],
            "official_reference": zone["official_reference"],
            "bounds_l93_metres": detail_bounds,
            "center_l93_metres": zone["center_l93_metres"],
            "center_wgs84_degrees": zone["center_wgs84_degrees"],
            "relation_to_effis_fire": zone["relation_to_effis_fire"],
        },
        "source_contract": {
            "file_name": contract_path.name,
            "schema_version": contract.get("schema_version"),
            "sha256": sha256_file(contract_path),
            "required_resolution_metres": contract.get("detail_resolution_metres"),
            "tile_ids": tiles,
        },
        "official_services": {
            "provider": "IGN Géoplateforme",
            "license": "Licence Ouverte / Open Licence 2.0",
            "copc_wfs": {
                "endpoint": WFS_URL,
                "typename": WFS_TYPENAME,
                "discovery_url": discovery_url,
            },
            "raster_wms": {
                "endpoint": WMS_URL,
                "capabilities_url": WMS_CAPABILITIES_URL,
                **capabilities,
            },
            "official_documentation": [
                "https://geoservices.ign.fr/sites/default/files/2024-11/Fiche_produit_Nuages-de-points-LiDAR.pdf",
                "https://geoservices.ign.fr/sites/default/files/2025-09/Offre_Produit_LiDAR_2025-08_0.pdf",
            ],
        },
        "sources": {
            "copc": copc_records,
            "mnt_0_5_m": raster_records["mnt"],
            "mns_0_5_m": raster_records["mns"],
        },
        "validation": {
            "status": "verified",
            "portable_paths": "relative_to_manifest_directory",
            "copc_tile_count": len(copc_records),
            "mnt_tile_count": len(raster_records["mnt"]),
            "mns_tile_count": len(raster_records["mns"]),
            "source_file_count": len(all_records),
            "source_byte_count": sum(
                int(record["byte_count"]) for record in all_records
            ),
            "copc_point_count": sum(
                int(record["point_count"]) for record in copc_records
            ),
            "raster_union_bounds_l93_metres": raster_union,
            "raster_union_covers_detail_square": covers_detail,
            "raster_resolution_metres": 0.5,
            "raster_resolution_evidence": (
                "Observed 0.50 m GeoTIFF transforms and WMS MinScaleDenominator corresponding "
                "to 0.50 m under the WMS 0.28 mm standard pixel."
            ),
            "all_source_hashes_computed": True,
            "all_copc_classifications_scanned": True,
        },
        "limitations": [
            "The WMS GeoTIFF WKT omits the RGF93 datum name; files are retained unchanged and the exact observed projection parameters are recorded.",
            "The four 1 km native raster tiles cover more than the offset 1 km detail square and must be clipped during model production.",
            "No source outside the Montmaur detail zone was acquired or validated by this manifest.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--zone", default=EXPECTED_ZONE)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.sources.resolve()
    if not root.is_dir():
        raise SystemExit(f"ERREUR: dossier de sources absent: {root}")
    output = args.output.resolve() if args.output else root / "source-manifest.json"
    if output.parent != root:
        raise SystemExit(
            "ERREUR: le manifeste portable doit être écrit à la racine des sources."
        )
    try:
        manifest = build_manifest(args.contract.resolve(), args.zone, root)
    except SourceValidationError as exc:
        raise SystemExit(f"ERREUR: {exc}") from exc
    write_json(output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["validation"]["status"],
                "manifest": str(output),
                **manifest["validation"],
                "manifest_sha256": sha256_file(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
