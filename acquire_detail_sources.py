#!/usr/bin/env python3
"""Acquire official IGN COPC/MNT/MNS sources for one contracted detail zone.

The zone and its kilometre tile identifiers come from ``detail_zones.v1.json``.
COPC URLs are re-discovered from the same official WFS and typename used by
``build_detail_source_manifest.py``.  MNT/MNS URLs are built by that module's
``raster_url`` function, preserving its native 0.50 m grid convention.

Downloads are restartable: data is written to ``*.part``, HTTP Range is used
when a partial file exists, the advertised byte count and file signature are
checked, then ``os.replace`` atomically publishes the final file.  A hidden
state file binds an output directory to one contract, zone and source plan so
the command can resume its own directory but refuses an unrelated directory.

Use ``--plan-only`` to discover and print the source plan without creating an
output directory or downloading payloads.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

from pyproj import Transformer

from build_detail_source_manifest import (
    RASTER_PRODUCTS,
    WFS_TYPENAME,
    WFS_URL,
    raster_url,
    tile_bounds,
)


STATE_SCHEMA = "fireviewer.detail-source-acquisition-state.v1"
REPORT_SCHEMA = "fireviewer.detail-source-acquisition-report.v1"
STATE_FILE_NAME = ".fireviewer-acquisition.json"
REPORT_FILE_NAME = "acquisition-report.json"
USER_AGENT = "FireWarning-detail-source-acquisition/1.0"
OFFICIAL_DOWNLOAD_HOST = "data.geopf.fr"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


class AcquisitionError(RuntimeError):
    """Raised when acquisition cannot continue without risking corrupt data."""


@dataclass(frozen=True)
class AssetPlan:
    """One official payload and its path relative to the acquisition root."""

    category: str
    tile_id: str
    url: str
    relative_path: str
    expected_magic_hex: str
    source_name: str | None = None
    source_timestamp: str | None = None

    def validate(self) -> None:
        if self.category not in {"copc", "mnt", "mns"}:
            raise AcquisitionError(f"Catégorie de source invalide: {self.category!r}")
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme != "https" or parsed.hostname != OFFICIAL_DOWNLOAD_HOST:
            raise AcquisitionError(f"URL IGN non officielle: {self.url}")
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise AcquisitionError(f"Chemin de source non sûr: {self.relative_path!r}")
        if relative.parts[0] != self.category:
            raise AcquisitionError(
                f"Le chemin {self.relative_path!r} ne correspond pas à {self.category!r}"
            )
        try:
            magic = bytes.fromhex(self.expected_magic_hex)
        except ValueError as exc:
            raise AcquisitionError("Signature attendue invalide") from exc
        if not magic:
            raise AcquisitionError("La signature attendue ne peut pas être vide")

    @property
    def expected_magic(self) -> bytes:
        return bytes.fromhex(self.expected_magic_hex)


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"JSON illisible {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AcquisitionError(f"Le document {path} doit être un objet JSON")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    raw = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_zone_contract(
    contract_path: Path, zone_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load any uniquely identified zone; no Montmaur-specific restriction."""

    contract = _read_json(contract_path)
    if contract.get("schema_version") != "1.0":
        raise AcquisitionError(
            f"Version de contrat non supportée: {contract.get('schema_version')!r}"
        )
    if contract.get("horizontal_crs") != "EPSG:2154":
        raise AcquisitionError("Le contrat de détail doit utiliser EPSG:2154")
    resolution = contract.get("detail_resolution_metres")
    if not isinstance(resolution, (int, float)) or not math.isclose(
        float(resolution), 0.5
    ):
        raise AcquisitionError("La résolution contractuelle doit être de 0,5 m")
    matches = [zone for zone in contract.get("zones", ()) if zone.get("id") == zone_id]
    if len(matches) != 1:
        raise AcquisitionError(
            f"La zone {zone_id!r} doit apparaître exactement une fois dans le contrat"
        )
    zone = matches[0]
    tiles = zone.get("ign_lidar_hd_kilometre_tiles")
    if not isinstance(tiles, list) or not tiles or len(set(tiles)) != len(tiles):
        raise AcquisitionError("La zone doit référencer des dalles IGN distinctes")
    for tile in tiles:
        tile_bounds(tile)
    bounds = zone.get("bounds_l93_metres")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or not all(isinstance(value, (int, float)) for value in bounds)
        or float(bounds[0]) >= float(bounds[2])
        or float(bounds[1]) >= float(bounds[3])
    ):
        raise AcquisitionError("Emprise Lambert-93 de zone invalide")
    return contract, zone


def build_wfs_discovery_url(zone: Mapping[str, Any]) -> str:
    """Build the official WFS query covering every contracted source tile."""

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
            "COUNT": str(max(50, len(source_bounds) * 4)),
        }
    )
    return f"{WFS_URL}?{query}"


OpenUrl = Callable[..., Any]
Sleep = Callable[[float], None]


def _open_with_retries(
    request: urllib.request.Request,
    *,
    opener: OpenUrl,
    timeout_seconds: float,
    attempts: int,
    sleep: Sleep,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return opener(request, timeout=timeout_seconds)
        except urllib.error.HTTPError:
            raise
        except Exception as exc:  # urllib exposes several transport exception types
            last_error = exc
            if attempt + 1 < attempts:
                sleep(1.5 * (attempt + 1))
    raise AcquisitionError(
        f"Ouverture distante impossible {request.full_url}: {last_error}"
    )


def fetch_json(
    url: str,
    *,
    opener: OpenUrl = urllib.request.urlopen,
    timeout_seconds: float = 180.0,
    attempts: int = 4,
    sleep: Sleep = time.sleep,
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        response = _open_with_retries(
            request,
            opener=opener,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
            sleep=sleep,
        )
        with response:
            raw = response.read()
    except Exception as exc:
        if isinstance(exc, AcquisitionError):
            raise
        raise AcquisitionError(f"Lecture JSON impossible {url}: {exc}") from exc
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"Réponse distante non JSON {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise AcquisitionError("La réponse WFS doit être un objet JSON")
    return result


def discover_copc(
    zone: Mapping[str, Any],
    *,
    opener: OpenUrl = urllib.request.urlopen,
    timeout_seconds: float = 180.0,
    attempts: int = 4,
    sleep: Sleep = time.sleep,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Resolve each contracted tile to an official WFS COPC feature."""

    url = build_wfs_discovery_url(zone)
    collection = fetch_json(
        url,
        opener=opener,
        timeout_seconds=timeout_seconds,
        attempts=attempts,
        sleep=sleep,
    )
    by_name: dict[str, dict[str, Any]] = {}
    for feature in collection.get("features", ()):
        properties = feature.get("properties", {})
        name = properties.get("name")
        if name is not None:
            by_name[str(name)] = properties
    result: dict[str, dict[str, Any]] = {}
    for tile in zone["ign_lidar_hd_kilometre_tiles"]:
        expected_name = f"LHD_FXX_{tile}_PTS_O_LAMB93_IGN69"
        properties = by_name.get(expected_name)
        if properties is None:
            raise AcquisitionError(
                f"Dalle COPC absente du WFS officiel: {expected_name}"
            )
        source_url = str(properties.get("url", ""))
        parsed = urllib.parse.urlparse(source_url)
        if parsed.scheme != "https" or parsed.hostname != OFFICIAL_DOWNLOAD_HOST:
            raise AcquisitionError(
                f"URL COPC IGN non officielle pour {tile}: {source_url}"
            )
        result[tile] = properties
    return url, result


def build_asset_plan(
    zone: Mapping[str, Any], discovered_copc: Mapping[str, Mapping[str, Any]]
) -> list[AssetPlan]:
    """Build stable local names compatible with the source-manifest validator."""

    result: list[AssetPlan] = []
    for tile in zone["ign_lidar_hd_kilometre_tiles"]:
        properties = discovered_copc.get(tile)
        if properties is None:
            raise AcquisitionError(f"Découverte COPC absente du plan: {tile}")
        result.append(
            AssetPlan(
                category="copc",
                tile_id=tile,
                url=str(properties.get("url", "")),
                relative_path=(f"copc/LHD_FXX_{tile}_PTS_LAMB93_IGN69.copc.laz"),
                expected_magic_hex=b"LASF".hex(),
                source_name=str(properties.get("name") or "") or None,
                source_timestamp=(
                    str(properties.get("timestamp"))
                    if properties.get("timestamp") is not None
                    else None
                ),
            )
        )
    for category in ("mnt", "mns"):
        for tile in zone["ign_lidar_hd_kilometre_tiles"]:
            result.append(
                AssetPlan(
                    category=category,
                    tile_id=tile,
                    url=raster_url(category, tile),
                    relative_path=(
                        f"{category}/LHD_FXX_{tile}_{category.upper()}_O_0M50_"
                        "LAMB93_IGN69.tif"
                    ),
                    # GeoTIFF can be little-endian (II*) or big-endian (MM*).
                    # The downloader validates either TIFF signature below.
                    expected_magic_hex=b"II*\x00".hex(),
                    source_name=RASTER_PRODUCTS[category],
                )
            )
    for asset in result:
        asset.validate()
    return result


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status)


def _header(response: Any, name: str) -> str | None:
    value = response.headers.get(name)
    return str(value) if value is not None else None


def _parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    if value is None:
        return None
    match = _CONTENT_RANGE.fullmatch(value.strip())
    if match is None:
        raise AcquisitionError(f"Content-Range invalide: {value!r}")
    total = None if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


def _valid_file_signature(path: Path, asset: AssetPlan) -> bool:
    with path.open("rb") as stream:
        signature = stream.read(max(4, len(asset.expected_magic)))
    if asset.category in {"mnt", "mns"}:
        return signature[:4] in {b"II*\x00", b"MM\x00*"}
    return signature.startswith(asset.expected_magic)


def _download_once(
    asset: AssetPlan,
    destination: Path,
    *,
    opener: OpenUrl,
    timeout_seconds: float,
    chunk_size: int,
) -> int:
    partial = destination.with_name(f"{destination.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(asset.url, headers=headers)
    try:
        response = opener(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            content_range = exc.headers.get("Content-Range")
            match = re.fullmatch(r"bytes \*/(\d+)", content_range or "")
            if match is not None and int(match.group(1)) == offset:
                if not _valid_file_signature(partial, asset):
                    raise AcquisitionError(
                        f"Signature de fichier invalide après reprise: {partial}"
                    ) from exc
                os.replace(partial, destination)
                return offset
        raise

    with response:
        status = _response_status(response)
        content_range = _parse_content_range(_header(response, "Content-Range"))
        if offset and status == 206:
            if content_range is None or content_range[0] != offset:
                raise AcquisitionError(
                    f"Le serveur n'a pas repris à l'octet {offset}: {content_range}"
                )
            write_offset = offset
            mode = "ab"
        elif status == 200:
            # A server may ignore Range. Restarting only the private .part file
            # is safe; a previously published final asset is never overwritten.
            write_offset = 0
            mode = "wb"
        elif not offset and status == 206 and content_range and content_range[0] == 0:
            write_offset = 0
            mode = "wb"
        else:
            raise AcquisitionError(f"Statut HTTP inattendu {status} pour {asset.url}")

        content_length_text = _header(response, "Content-Length")
        content_length = (
            int(content_length_text) if content_length_text is not None else None
        )
        expected_total = content_range[2] if content_range is not None else None
        if expected_total is None and content_length is not None:
            expected_total = write_offset + content_length
        if expected_total is None:
            raise AcquisitionError(
                f"Taille distante indéterminable; fichier partiel conservé: {asset.url}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        with partial.open(mode) as stream:
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())

    observed_size = partial.stat().st_size
    if observed_size != expected_total:
        raise AcquisitionError(
            f"Téléchargement incomplet {asset.relative_path}: "
            f"{observed_size}/{expected_total} octets; reprise possible"
        )
    if not _valid_file_signature(partial, asset):
        raise AcquisitionError(
            f"Signature inattendue pour {asset.relative_path}; fichier .part conservé"
        )
    os.replace(partial, destination)
    return observed_size


def download_resumable(
    asset: AssetPlan,
    destination: Path,
    *,
    opener: OpenUrl = urllib.request.urlopen,
    timeout_seconds: float = 300.0,
    attempts: int = 4,
    chunk_size: int = 1024 * 1024,
    sleep: Sleep = time.sleep,
) -> int:
    """Download one asset atomically, retrying from the latest .part offset."""

    asset.validate()
    if destination.exists():
        raise AcquisitionError(
            f"Refus d'écraser une source finale existante: {destination}"
        )
    if attempts <= 0 or chunk_size <= 0:
        raise ValueError("attempts and chunk_size must be strictly positive")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _download_once(
                asset,
                destination,
                opener=opener,
                timeout_seconds=timeout_seconds,
                chunk_size=chunk_size,
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                sleep(1.5 * (attempt + 1))
    if isinstance(last_error, AcquisitionError):
        raise last_error
    raise AcquisitionError(
        f"Téléchargement impossible après {attempts} essais {asset.url}: {last_error}"
    ) from last_error


def _plan_payload(assets: Sequence[AssetPlan]) -> list[dict[str, Any]]:
    return [asdict(asset) for asset in assets]


def _plan_sha256(assets: Sequence[AssetPlan]) -> str:
    canonical = json.dumps(
        _plan_payload(assets), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def initialize_or_resume_root(
    root: Path,
    *,
    contract_path: Path,
    contract: Mapping[str, Any],
    zone: Mapping[str, Any],
    discovery_url: str,
    assets: Sequence[AssetPlan],
) -> dict[str, Any]:
    """Create a bound output root, or validate that it is our resumable root."""

    root = root.resolve()
    state_path = root / STATE_FILE_NAME
    contract_digest = sha256_file(contract_path)
    plan_digest = _plan_sha256(assets)
    identity = {
        "contract_sha256": contract_digest,
        "parent_package_id": contract.get("parent_package_id"),
        "zone_id": zone.get("id"),
        "plan_sha256": plan_digest,
    }
    if root.exists():
        if not root.is_dir() or not state_path.is_file():
            raise AcquisitionError(
                f"Le dossier existant n'est pas une acquisition reprenable: {root}"
            )
        state = _read_json(state_path)
        if state.get("schema") != STATE_SCHEMA:
            raise AcquisitionError(f"État d'acquisition non supporté: {state_path}")
        observed_identity = {key: state.get(key) for key in identity}
        if observed_identity != identity:
            raise AcquisitionError(
                "Le dossier de reprise appartient à un autre contrat, une autre "
                "zone ou un autre plan de sources"
            )
        return state

    root.mkdir(parents=True, exist_ok=False)
    for category in ("copc", "mnt", "mns"):
        (root / category).mkdir()
    state = {
        "schema": STATE_SCHEMA,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        **identity,
        "contract_file_name": contract_path.name,
        "discovery_url": discovery_url,
        "assets": _plan_payload(assets),
        "completed": {},
    }
    _atomic_write_json(state_path, state)
    return state


def _safe_destination(root: Path, asset: AssetPlan) -> Path:
    root = root.resolve()
    destination = (root / Path(asset.relative_path)).resolve()
    if root not in destination.parents:
        raise AcquisitionError(
            f"Destination hors du dossier d'acquisition: {destination}"
        )
    return destination


def _verify_completed(
    destination: Path, asset: AssetPlan, record: Mapping[str, Any]
) -> None:
    if not destination.is_file():
        raise AcquisitionError(f"Source marquée complète mais absente: {destination}")
    observed_size = destination.stat().st_size
    if observed_size != record.get("byte_count"):
        raise AcquisitionError(f"Taille d'une source complète modifiée: {destination}")
    if sha256_file(destination) != record.get("sha256"):
        raise AcquisitionError(f"SHA-256 d'une source complète modifié: {destination}")
    if not _valid_file_signature(destination, asset):
        raise AcquisitionError(
            f"Signature d'une source complète modifiée: {destination}"
        )


def acquire_assets(
    root: Path,
    state: dict[str, Any],
    assets: Sequence[AssetPlan],
    *,
    opener: OpenUrl = urllib.request.urlopen,
    timeout_seconds: float = 300.0,
    attempts: int = 4,
    request_interval_seconds: float = 1.05,
    sleep: Sleep = time.sleep,
) -> dict[str, Any]:
    """Acquire all planned assets, persisting completion after every file."""

    root = root.resolve()
    state_path = root / STATE_FILE_NAME
    completed = state.setdefault("completed", {})
    for asset_index, asset in enumerate(assets):
        destination = _safe_destination(root, asset)
        prior = completed.get(asset.relative_path)
        if prior is not None:
            _verify_completed(destination, asset, prior)
            continue
        if destination.exists():
            raise AcquisitionError(
                f"Source finale sans preuve d'acquisition; refus d'écraser: {destination}"
            )
        if asset_index and request_interval_seconds > 0:
            sleep(request_interval_seconds)
        byte_count = download_resumable(
            asset,
            destination,
            opener=opener,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
            sleep=sleep,
        )
        completed[asset.relative_path] = {
            "byte_count": byte_count,
            "sha256": sha256_file(destination),
            "completed_at": utc_now(),
        }
        state["updated_at"] = utc_now()
        _atomic_write_json(state_path, state)
    return state


def build_report(
    contract_path: Path,
    contract: Mapping[str, Any],
    zone: Mapping[str, Any],
    discovery_url: str,
    assets: Sequence[AssetPlan],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    completed = state.get("completed", {})
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": utc_now(),
        "status": "downloaded_not_yet_content_validated",
        "contract": {
            "file_name": contract_path.name,
            "sha256": sha256_file(contract_path),
            "schema_version": contract.get("schema_version"),
            "parent_package_id": contract.get("parent_package_id"),
        },
        "zone": {
            "id": zone.get("id"),
            "label": zone.get("label"),
            "bounds_l93_metres": zone.get("bounds_l93_metres"),
            "tile_ids": zone.get("ign_lidar_hd_kilometre_tiles"),
        },
        "official_services": {
            "copc_wfs_discovery_url": discovery_url,
            "copc_wfs_typename": WFS_TYPENAME,
            "raster_layers": RASTER_PRODUCTS,
        },
        "assets": [
            {
                **asdict(asset),
                **completed.get(asset.relative_path, {}),
            }
            for asset in assets
        ],
        "validation": {
            "planned_asset_count": len(assets),
            "downloaded_asset_count": len(completed),
            "all_downloads_complete": len(completed) == len(assets),
            "atomic_publish_after_byte_count_and_signature": True,
            "content_validation_required": "build_detail_source_manifest.py",
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--request-interval-seconds", type=float, default=1.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    contract_path = args.contract.resolve()
    contract, zone = load_zone_contract(contract_path, args.zone)
    discovery_url, discovered = discover_copc(
        zone,
        timeout_seconds=args.timeout_seconds,
        attempts=args.attempts,
    )
    assets = build_asset_plan(zone, discovered)
    plan_output = {
        "zone": zone["id"],
        "discovery_url": discovery_url,
        "asset_count": len(assets),
        "assets": _plan_payload(assets),
    }
    if args.plan_only:
        print(json.dumps(plan_output, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    root = args.output.resolve()
    state = initialize_or_resume_root(
        root,
        contract_path=contract_path,
        contract=contract,
        zone=zone,
        discovery_url=discovery_url,
        assets=assets,
    )
    state = acquire_assets(
        root,
        state,
        assets,
        timeout_seconds=args.timeout_seconds,
        attempts=args.attempts,
        request_interval_seconds=args.request_interval_seconds,
    )
    report = build_report(
        contract_path,
        contract,
        zone,
        discovery_url,
        assets,
        state,
    )
    _atomic_write_json(root / REPORT_FILE_NAME, report)
    print(
        json.dumps(report["validation"], ensure_ascii=False, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
