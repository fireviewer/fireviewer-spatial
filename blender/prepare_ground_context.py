"""Acquire and clip official ground context for the six terrain-first cases.

The published artifact is one compact EPSG:2154 GeoPackage per square AOI.
Raw department archives are resumable working data and may be removed only
after the derived package and its manifest have passed validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pyogrio
from pyogrio import raw as ogr_raw
from pyproj import CRS
from shapely import from_wkb, make_valid, to_wkb
from shapely.geometry import box, shape

from ground_context_binding import (
    classify_feature,
    load_context_contract,
    load_runtime_contract,
    validate_profile_bindings,
)
from prepare_incident_terrains import CASES, IncidentTerrainCase


MANIFEST_SCHEMA = "fireviewer.ground-context-package.v1"
USER_AGENT = "FireViewer-ground-context/1.0"
PAGE_SIZE = 10_000
NUMERIC_FIELDS = {
    "contenance",
    "largeur_de_chaussee",
    "precision_planimetrique",
    "nombre_de_voies",
    "ossature",
    "MI_PRINX",
    "CARTE",
    "CODE",
    "CODE_LEG",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_by_id(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(source["id"]): dict(source) for source in contract["sources"]}


def ocsge_url(title: str) -> str:
    encoded = urllib.parse.quote(title, safe="")
    return f"https://data.geopf.fr/telechargement/download/OCSGE/{encoded}/{encoded}.7z"


def build_wfs_url(
    source: Mapping[str, Any], bounds: tuple[float, float, float, float], start: int
) -> str:
    query = urllib.parse.urlencode(
        {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": source["typename"],
            "OUTPUTFORMAT": "application/json",
            "SRSNAME": "EPSG:2154",
            "BBOX": ",".join(str(value) for value in bounds) + ",EPSG:2154",
            "COUNT": str(PAGE_SIZE),
            "STARTINDEX": str(start),
            "SORTBY": source["sort_by"],
        }
    )
    return f"{source['endpoint']}?{query}"


def plan_case(
    case: IncidentTerrainCase, contract: Mapping[str, Any]
) -> dict[str, Any]:
    incident = contract["incidents"][case.fire_id]
    title = incident["ocsge_title"]
    return {
        "fire_id": case.fire_id,
        "bounds_epsg2154_m": list(case.square_bounds_epsg2154_m),
        "output_layers": [source["id"] for source in contract["sources"]],
        "wfs_first_page_urls": {
            source["id"]: build_wfs_url(source, case.square_bounds_epsg2154_m, 0)
            for source in contract["sources"]
            if source["transport"] == "WFS_2.0.0"
        },
        "archives": {
            "ocsge": ocsge_url(title),
            "brgm": incident["brgm_url"],
        },
    }


def _open(request: urllib.request.Request, *, attempts: int = 4) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Remote request failed: {request.full_url}: {last_error}")


def download_resumable(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        with destination.open("rb") as stream:
            signature = stream.read(6)
        if destination.suffix == ".7z" and signature != bytes.fromhex("377abcaf271c"):
            raise ValueError(f"Invalid cached 7z signature: {destination}")
        if destination.suffix == ".zip" and not signature.startswith(b"PK"):
            raise ValueError(f"Invalid cached zip signature: {destination}")
        return {
            "url": url,
            "path": destination.name,
            "byte_count": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "cache": "validated_existing",
        }
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    with _open(urllib.request.Request(url, headers=headers)) as response:
        append = start > 0 and response.status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
    os.replace(partial, destination)
    with destination.open("rb") as stream:
        signature = stream.read(6)
    if destination.suffix == ".7z" and signature != bytes.fromhex("377abcaf271c"):
        raise ValueError(f"Invalid 7z signature: {destination}")
    if destination.suffix == ".zip" and not signature.startswith(b"PK"):
        raise ValueError(f"Invalid zip signature: {destination}")
    return {
        "url": url,
        "path": destination.name,
        "byte_count": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "cache": "downloaded",
    }


def _run_7z(seven_zip: Path, arguments: list[str]) -> None:
    result = subprocess.run(
        [str(seven_zip), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"7-Zip failed ({result.returncode}): {result.stderr or result.stdout}"
        )


def extract_archives(
    *,
    seven_zip: Path,
    ocs_archive: Path,
    brgm_archive: Path,
    extract_root: Path,
    geology_token: str,
) -> tuple[Path, Path]:
    if extract_root.exists():
        shutil.rmtree(extract_root)
    ocs_root = extract_root / "ocsge"
    brgm_root = extract_root / "brgm"
    ocs_root.mkdir(parents=True)
    brgm_root.mkdir(parents=True)
    _run_7z(
        seven_zip,
        ["x", "-y", f"-o{ocs_root}", str(ocs_archive), "-ir!*OCCUPATION_SOL.gpkg"],
    )
    _run_7z(
        seven_zip,
        [
            "x",
            "-y",
            f"-o{brgm_root}",
            str(brgm_archive),
            f"GEO050K_HARM_{geology_token}_S_FGEOL_2154.*",
        ],
    )
    ocs_files = list(ocs_root.rglob("OCCUPATION_SOL.gpkg"))
    geology_files = list(
        brgm_root.rglob(f"GEO050K_HARM_{geology_token}_S_FGEOL_2154.shp")
    )
    if len(ocs_files) != 1 or len(geology_files) != 1:
        raise ValueError("Official archive structure does not match the context contract")
    return ocs_files[0], geology_files[0]


def _clip_geometry(geometry: Any, clip_box: Any) -> Any | None:
    if geometry is None or geometry.is_empty:
        return None
    geometry = make_valid(geometry)
    clipped = make_valid(geometry.intersection(clip_box))
    if clipped.is_empty:
        return None
    return clipped


def fetch_wfs_layer(
    source: Mapping[str, Any], bounds: tuple[float, float, float, float]
) -> tuple[list[bytes], list[dict[str, Any]], dict[str, Any]]:
    geometries: list[bytes] = []
    records: list[dict[str, Any]] = []
    page_hashes: list[str] = []
    requests: list[str] = []
    clip_box = box(*bounds)
    start = 0
    while True:
        url = build_wfs_url(source, bounds, start)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with _open(request) as response:
            raw = response.read()
        page_hashes.append(hashlib.sha256(raw).hexdigest())
        requests.append(url)
        payload = json.loads(raw)
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError(f"WFS response has no feature list: {source['id']}")
        for feature in features:
            geometry_value = feature.get("geometry")
            if geometry_value is None:
                continue
            clipped = _clip_geometry(shape(geometry_value), clip_box)
            if clipped is None:
                continue
            properties = feature.get("properties") or {}
            records.append({field: properties.get(field) for field in source["fields"]})
            geometries.append(to_wkb(clipped))
        returned = int(payload.get("numberReturned", len(features)))
        matched_raw = payload.get("numberMatched")
        matched = int(matched_raw) if str(matched_raw).isdigit() else None
        if not features or returned == 0 or (matched is not None and start + returned >= matched):
            break
        start += returned
    if records and all(record.get(source["sort_by"]) is None for record in records):
        raise ValueError(f"WFS primary identifier is absent: {source['id']}")
    return geometries, records, {
        "transport": "WFS_2.0.0",
        "typename": source["typename"],
        "request_count": len(requests),
        "requests": requests,
        "response_sha256": page_hashes,
    }


def read_archive_layer(
    path: Path,
    *,
    layer: str | None,
    fields: list[str],
    bounds: tuple[float, float, float, float],
) -> tuple[list[bytes], list[dict[str, Any]]]:
    info = pyogrio.read_info(path, layer=layer)
    if CRS.from_user_input(info["crs"]).to_epsg() != 2154:
        raise ValueError(f"Archive layer is not EPSG:2154: {path}")
    meta, _, geometry, field_data = ogr_raw.read(
        path,
        layer=layer,
        columns=fields,
        bbox=bounds,
        force_2d=True,
    )
    returned_fields = [str(field) for field in meta["fields"]]
    missing = set(fields) - set(returned_fields)
    if missing:
        raise ValueError(f"Missing archive fields in {path}: {sorted(missing)}")
    field_index = {field: index for index, field in enumerate(returned_fields)}
    clip_box = box(*bounds)
    geometries: list[bytes] = []
    records: list[dict[str, Any]] = []
    for row, encoded in enumerate(geometry):
        decoded = None if encoded is None else from_wkb(encoded)
        clipped = _clip_geometry(decoded, clip_box)
        if clipped is None:
            continue
        records.append(
            {
                field: field_data[field_index[field]][row]
                for field in fields
            }
        )
        geometries.append(to_wkb(clipped))
    return geometries, records


def _field_arrays(
    records: list[dict[str, Any]], fields: list[str]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for field in fields:
        values = [record.get(field) for record in records]
        mask = np.asarray([value is None for value in values], dtype=bool)
        if field in NUMERIC_FIELDS:
            converted = np.asarray(
                [np.nan if value is None else float(value) for value in values],
                dtype=np.float64,
            )
        else:
            converted = np.asarray(
                [None if value is None else str(value).strip() for value in values],
                dtype=object,
            )
        arrays.append(converted)
        masks.append(mask)
    return arrays, masks


def write_layer(
    package: Path,
    *,
    layer_id: str,
    fields: list[str],
    geometries: list[bytes],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    arrays, masks = _field_arrays(records, fields)
    ogr_raw.write(
        package,
        np.asarray(geometries, dtype=object),
        arrays,
        fields,
        field_mask=masks,
        layer=layer_id,
        driver="GPKG",
        geometry_type="Unknown",
        crs="EPSG:2154",
        layer_metadata={"fireviewer_role": layer_id},
    )
    semantic_counts: Counter[str] = Counter()
    for record in records:
        semantic_counts.update(classify_feature(layer_id, record))
    info = pyogrio.read_info(package, layer=layer_id)
    if int(info["features"]) != len(records):
        raise ValueError(f"Written feature count mismatch for {layer_id}")
    total_bounds = [float(value) for value in info["total_bounds"]]
    serialized_bounds = (
        [round(value, 3) for value in total_bounds]
        if total_bounds and all(np.isfinite(value) for value in total_bounds)
        else None
    )
    return {
        "feature_count": len(records),
        "fields": fields,
        "geometry_type": info["geometry_type"],
        "bounds_epsg2154_m": serialized_bounds,
        "semantic_tag_counts": dict(sorted(semantic_counts.items())),
    }


def validate_package(
    package: Path,
    manifest: Mapping[str, Any],
    expected_layers: set[str],
) -> None:
    if not package.is_file() or sha256_file(package) != manifest["package"]["sha256"]:
        raise ValueError("Ground context package SHA-256 mismatch")
    actual_layers = {str(row[0]) for row in pyogrio.list_layers(package)}
    if actual_layers != expected_layers:
        raise ValueError("Ground context package layer set is incomplete")
    for layer_id in expected_layers:
        info = pyogrio.read_info(package, layer=layer_id)
        if CRS.from_user_input(info["crs"]).to_epsg() != 2154:
            raise ValueError(f"Ground context layer is not EPSG:2154: {layer_id}")
        if int(info["features"]) != manifest["layers"][layer_id]["feature_count"]:
            raise ValueError(f"Ground context feature count mismatch: {layer_id}")


def _safe_cleanup(root: Path, targets: list[Path]) -> int:
    root = root.resolve()
    removed_bytes = 0
    for target in targets:
        resolved = target.resolve()
        if root not in resolved.parents:
            raise ValueError(f"Refusing cleanup outside source cache: {resolved}")
        if resolved.is_dir():
            removed_bytes += sum(
                path.stat().st_size for path in resolved.rglob("*") if path.is_file()
            )
            shutil.rmtree(resolved)
        elif resolved.is_file():
            removed_bytes += resolved.stat().st_size
            resolved.unlink()
    return removed_bytes


def execute_case(
    case: IncidentTerrainCase,
    contract: Mapping[str, Any],
    *,
    output_root: Path,
    source_cache_root: Path,
    seven_zip: Path,
    cleanup_sources: bool,
) -> dict[str, Any]:
    case_root = output_root / case.fire_id.lower()
    package = case_root / "ground-context.gpkg"
    manifest_path = case_root / "ground-context-manifest.json"
    expected_layers = {source["id"] for source in contract["sources"]}
    if package.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_package(package, existing, expected_layers)
        current_contract_hash = canonical_sha256(contract)
        if existing.get("context_contract_sha256") != current_contract_hash:
            existing["context_contract_sha256"] = current_contract_hash
            existing["context_revalidated_at"] = utc_now()
            _atomic_json(manifest_path, existing)
            validate_package(package, existing, expected_layers)
        return existing
    if package.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing context: {case_root}")
    case_root.mkdir(parents=True, exist_ok=True)
    temporary_package = case_root / ".ground-context.tmp.gpkg"
    temporary_package.unlink(missing_ok=True)
    incident = contract["incidents"][case.fire_id]
    department = incident["department"]
    geology_token = incident.get("brgm_file_token", department)
    title = incident["ocsge_title"]
    ocs_url = ocsge_url(title)
    brgm_url = incident["brgm_url"]
    ocs_archive = source_cache_root / "archives" / "ocsge" / f"{title}.7z"
    brgm_archive = source_cache_root / "archives" / "brgm" / Path(
        urllib.parse.urlparse(brgm_url).path
    ).name
    extract_root = source_cache_root / "extracted" / case.fire_id.lower()
    sources = _source_by_id(contract)
    source_receipts = {
        "ocsge": download_resumable(ocs_url, ocs_archive),
        "brgm": download_resumable(brgm_url, brgm_archive),
    }
    ocs_path, geology_path = extract_archives(
        seven_zip=seven_zip,
        ocs_archive=ocs_archive,
        brgm_archive=brgm_archive,
        extract_root=extract_root,
        geology_token=geology_token,
    )
    bounds = case.square_bounds_epsg2154_m
    layer_receipts: dict[str, Any] = {}
    try:
        for layer_id in (
            "land_parcels",
            "agricultural_parcels",
            "roads",
            "railways",
            "hydro_lines",
            "hydro_surfaces",
        ):
            source = sources[layer_id]
            geometries, records, provenance = fetch_wfs_layer(source, bounds)
            receipt = write_layer(
                temporary_package,
                layer_id=layer_id,
                fields=source["fields"],
                geometries=geometries,
                records=records,
            )
            receipt["provenance"] = provenance
            layer_receipts[layer_id] = receipt
        for layer_id, path, layer_name in (
            ("landcover", ocs_path, sources["landcover"]["layer"]),
            ("geology", geology_path, None),
        ):
            source = sources[layer_id]
            geometries, records = read_archive_layer(
                path,
                layer=layer_name,
                fields=source["fields"],
                bounds=bounds,
            )
            receipt = write_layer(
                temporary_package,
                layer_id=layer_id,
                fields=source["fields"],
                geometries=geometries,
                records=records,
            )
            receipt["provenance"] = {
                "transport": "department_archive",
                "archive": "ocsge" if layer_id == "landcover" else "brgm",
                "archive_sha256": source_receipts[
                    "ocsge" if layer_id == "landcover" else "brgm"
                ]["sha256"],
            }
            layer_receipts[layer_id] = receipt
        os.replace(temporary_package, package)
    finally:
        temporary_package.unlink(missing_ok=True)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "derived_context_validated",
        "created_at": utc_now(),
        "fire_id": case.fire_id,
        "crs": "EPSG:2154",
        "square_bounds_epsg2154_m": list(bounds),
        "orthophoto_dependency": "forbidden",
        "context_contract_sha256": canonical_sha256(contract),
        "profile_binding_count": 72,
        "sources": source_receipts,
        "layers": layer_receipts,
        "package": {
            "path": package.name,
            "byte_count": package.stat().st_size,
            "sha256": sha256_file(package),
        },
        "source_cleanup": {"status": "pending"},
    }
    validate_package(package, manifest, expected_layers)
    _atomic_json(manifest_path, manifest)
    if cleanup_sources:
        removed = _safe_cleanup(
            source_cache_root,
            [extract_root, ocs_archive, brgm_archive],
        )
        manifest["source_cleanup"] = {
            "status": "completed_after_package_validation",
            "removed_byte_count": removed,
        }
    _atomic_json(manifest_path, manifest)
    validate_package(package, manifest, expected_layers)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("ground_context_contract.v1.json"),
    )
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=Path(__file__).with_name("ground_surface_runtime_contract.v3.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument(
        "--seven-zip", type=Path, default=Path(r"C:\Program Files\7-Zip\7z.exe")
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", choices=sorted(case.fire_id for case in CASES))
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cleanup-sources", action="store_true")
    return parser


def write_batch_catalog(
    output_root: Path,
    contract: Mapping[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(results) != 6 or {result["fire_id"] for result in results} != {
        case.fire_id for case in CASES
    }:
        raise ValueError("Ground context batch catalog requires all six cases")
    layer_totals: Counter[str] = Counter()
    cases = []
    for result in sorted(results, key=lambda item: item["fire_id"]):
        case_root = output_root / result["fire_id"].lower()
        manifest_path = case_root / "ground-context-manifest.json"
        for layer_id, receipt in result["layers"].items():
            layer_totals[layer_id] += int(receipt["feature_count"])
        cases.append(
            {
                "fire_id": result["fire_id"],
                "package_path": (case_root / result["package"]["path"])
                .relative_to(output_root)
                .as_posix(),
                "package_sha256": result["package"]["sha256"],
                "package_byte_count": result["package"]["byte_count"],
                "manifest_path": manifest_path.relative_to(output_root).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "feature_count": sum(
                    int(receipt["feature_count"])
                    for receipt in result["layers"].values()
                ),
            }
        )
    catalog = {
        "schema": "fireviewer.ground-context-catalog.v1",
        "status": "validated_six_case_context",
        "crs": "EPSG:2154",
        "orthophoto_dependency": "forbidden",
        "case_count": 6,
        "context_contract_sha256": canonical_sha256(contract),
        "profile_binding_count": 72,
        "layer_feature_totals": dict(sorted(layer_totals.items())),
        "total_feature_count": sum(layer_totals.values()),
        "total_package_byte_count": sum(
            int(case["package_byte_count"]) for case in cases
        ),
        "cases": cases,
    }
    catalog["catalog_sha256"] = canonical_sha256(catalog)
    _atomic_json(output_root / "ground-context-catalog.json", catalog)
    return catalog


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_context_contract(args.contract.resolve())
    runtime = load_runtime_contract(args.runtime_contract.resolve())
    validate_profile_bindings(contract, runtime)
    cases = list(CASES) if args.all else [next(case for case in CASES if case.fire_id == args.case)]
    if not args.execute:
        print(
            json.dumps(
                [plan_case(case, contract) for case in cases],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.output_root is None or args.source_cache_root is None:
        raise ValueError("--execute requires --output-root and --source-cache-root")
    if not args.seven_zip.is_file():
        raise FileNotFoundError(f"7-Zip not found: {args.seven_zip}")
    results = [
        execute_case(
            case,
            contract,
            output_root=args.output_root.resolve(),
            source_cache_root=args.source_cache_root.resolve(),
            seven_zip=args.seven_zip.resolve(),
            cleanup_sources=args.cleanup_sources,
        )
        for case in cases
    ]
    if args.all:
        write_batch_catalog(args.output_root.resolve(), contract, results)
    print(
        json.dumps(
            [
                {
                    "fire_id": result["fire_id"],
                    "package_sha256": result["package"]["sha256"],
                    "layers": {
                        key: value["feature_count"]
                        for key, value in result["layers"].items()
                    },
                    "source_cleanup": result["source_cleanup"],
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
