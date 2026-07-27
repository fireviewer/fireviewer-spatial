#!/usr/bin/env python3
"""Verify hashes, geometry relationships and the terrain COG of a zone package."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from prepare_zone import PreparationError, sha256_file, verify_cog


class VerificationError(RuntimeError):
    """Raised when a package is incomplete, altered or spatially inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"JSON illisible {path}: {exc}") from exc


def _safe_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise VerificationError(
            f"Chemin d'artefact hors package: {relative_path}"
        ) from exc
    return candidate


def _verify_record(root: Path, record: dict[str, Any], label: str) -> Path:
    path = _safe_path(root, record.get("path", ""))
    if not path.is_file():
        raise VerificationError(f"Artefact absent ({label}): {path}")
    actual_size = path.stat().st_size
    if actual_size != record.get("byte_count"):
        raise VerificationError(
            f"Taille incorrecte ({label}): attendue {record.get('byte_count')}, obtenue {actual_size}."
        )
    actual_hash = sha256_file(path)
    if actual_hash != record.get("sha256"):
        raise VerificationError(
            f"SHA-256 incorrect ({label}): attendu {record.get('sha256')}, obtenu {actual_hash}."
        )
    return path


def _load_single_geometry(path: Path):
    payload = _read_json(path)
    features = payload.get("features", [])
    if payload.get("type") != "FeatureCollection" or len(features) != 1:
        raise VerificationError(
            f"{path.name} doit contenir exactement une entité GeoJSON."
        )
    geometry = shape(features[0].get("geometry"))
    if geometry.is_empty or not geometry.is_valid:
        raise VerificationError(f"Géométrie vide ou invalide: {path.name}")
    return geometry, features[0].get("properties", {})


def _coordinates_are_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_coordinates_are_finite(item) for item in value)
    return False


def _verify_vector_collection(
    path: Path, expected_count: int, aoi_l93
) -> dict[str, Any]:
    payload = _read_json(path)
    features = payload.get("features")
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise VerificationError(f"Collection GeoJSON invalide: {path.name}")
    if len(features) != expected_count:
        raise VerificationError(
            f"Nombre d'entités incorrect ({path.name}): attendu {expected_count}, obtenu {len(features)}."
        )
    null_heights = 0
    for index, feature in enumerate(features):
        raw_geometry = feature.get("geometry") if isinstance(feature, dict) else None
        coordinates = (
            raw_geometry.get("coordinates") if isinstance(raw_geometry, dict) else None
        )
        if not _coordinates_are_finite(coordinates):
            raise VerificationError(
                f"Coordonnée non finie ({path.name}, entité {index})."
            )
        geometry = shape(raw_geometry)
        if (
            geometry.is_empty
            or not geometry.is_valid
            or geometry.geom_type not in {"Polygon", "MultiPolygon"}
        ):
            raise VerificationError(
                f"Géométrie vectorielle invalide ({path.name}, entité {index})."
            )
        # The vector preparer receives the RFC 7946 WGS84 AOI. Its round-trip
        # back to Lambert-93 differs from the canonical L93 file by a few
        # nanometres (measured maximum about 5.3e-9 m on the real package).
        if not aoi_l93.buffer(1e-6).covers(geometry):
            raise VerificationError(
                f"Géométrie hors emprise ({path.name}, entité {index})."
            )
        properties = feature.get("properties") or {}
        for key in ("base_elevation_m", "block_height_m"):
            value = properties.get(key)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(float(value))
            ):
                raise VerificationError(
                    f"Valeur {key} non finie ({path.name}, entité {index})."
                )
        height = properties.get("block_height_m")
        if height is None:
            null_heights += 1
        elif float(height) < 0:
            raise VerificationError(f"Hauteur négative ({path.name}, entité {index}).")
    return {"feature_count": len(features), "null_height_count": null_heights}


def _verify_vector_extension(
    root: Path,
    catalog: dict[str, Any],
    paths: dict[str, Path],
    aoi_l93,
) -> dict[str, Any] | None:
    layer_ids = {"buildings_l93", "vegetation_blocks_l93", "vector_model_manifest"}
    present = layer_ids.intersection(paths)
    if not present:
        return None
    if present != layer_ids:
        raise VerificationError(f"Extension vectorielle partielle: {sorted(present)}")

    vector_manifest = _read_json(paths["vector_model_manifest"])
    origin = vector_manifest.get("origin_l93")
    if not isinstance(origin, dict) or not all(
        isinstance(origin.get(axis), (int, float))
        and not isinstance(origin.get(axis), bool)
        and math.isfinite(float(origin[axis]))
        for axis in ("x", "y", "z")
    ):
        raise VerificationError(
            "Origine Lambert-93 invalide dans vector-manifest.json."
        )

    results: dict[str, Any] = {}
    links = {
        "buildings": "buildings_l93",
        "vegetation": "vegetation_blocks_l93",
    }
    for output_id, layer_id in links.items():
        output = vector_manifest.get("outputs", {}).get(output_id, {})
        catalog_record = catalog["layers"][layer_id]
        path = (
            paths["vector_model_manifest"].parent / output.get("path", "")
        ).resolve()
        if path != paths[layer_id].resolve():
            raise VerificationError(
                f"Chemin incohérent entre le catalogue et vector-manifest: {output_id}"
            )
        if output.get("sha256") != catalog_record.get("sha256"):
            raise VerificationError(
                f"SHA-256 incohérent entre les contrats: {output_id}"
            )
        if output.get("bytes") != catalog_record.get("byte_count"):
            raise VerificationError(
                f"Taille incohérente entre les contrats: {output_id}"
            )
        expected_count = output.get("feature_count")
        if expected_count != catalog_record.get("feature_count") or not isinstance(
            expected_count, int
        ):
            raise VerificationError(
                f"Nombre d'entités incohérent entre les contrats: {output_id}"
            )
        results[output_id] = _verify_vector_collection(path, expected_count, aoi_l93)

    deferred = catalog.get("deferred_layers", {})
    if deferred.get("buildings", {}).get("status") != "produced":
        raise VerificationError(
            "Le catalogue ne marque pas les bâtiments comme produits."
        )
    if deferred.get("vegetation_blocks", {}).get("status") != "produced":
        raise VerificationError(
            "Le catalogue ne marque pas les blocs de végétation comme produits."
        )
    return {"origin_l93": origin, **results}


def verify_package(package_dir: Path) -> dict[str, Any]:
    root = package_dir.resolve()
    manifest_path = root / "package-manifest.json"
    if not manifest_path.is_file():
        raise VerificationError(f"Manifest absent: {manifest_path}")
    manifest = _read_json(manifest_path)
    catalog_path = _verify_record(root, manifest.get("catalog", {}), "catalog")
    catalog = _read_json(catalog_path)
    if catalog.get("package_id") != manifest.get("package_id"):
        raise VerificationError(
            "Les identifiants de package du manifeste et du catalogue diffèrent."
        )

    layers = catalog.get("layers")
    if not isinstance(layers, dict):
        raise VerificationError(
            "Le catalogue ne contient pas de dictionnaire de couches."
        )
    paths: dict[str, Path] = {}
    for layer_id, record in layers.items():
        if not isinstance(record, dict):
            raise VerificationError(f"Contrat de couche invalide: {layer_id}")
        paths[layer_id] = _verify_record(root, record, layer_id)

    required = {
        "terrain_mnt",
        "fire_perimeter_wgs84",
        "area_of_interest_wgs84",
        "fire_perimeter_l93",
        "area_of_interest_l93",
    }
    missing = required.difference(paths)
    if missing:
        raise VerificationError(f"Couches obligatoires absentes: {sorted(missing)}")

    try:
        cog = verify_cog(paths["terrain_mnt"])
    except PreparationError as exc:
        raise VerificationError(str(exc)) from exc

    perimeter_l93, perimeter_properties = _load_single_geometry(
        paths["fire_perimeter_l93"]
    )
    aoi_l93, aoi_properties = _load_single_geometry(paths["area_of_interest_l93"])
    perimeter_wgs84, _ = _load_single_geometry(paths["fire_perimeter_wgs84"])
    aoi_wgs84, _ = _load_single_geometry(paths["area_of_interest_wgs84"])

    if not aoi_l93.covers(perimeter_l93):
        raise VerificationError(
            "L'emprise Lambert-93 ne couvre pas le périmètre de feu."
        )
    if not aoi_wgs84.covers(perimeter_wgs84):
        raise VerificationError("L'emprise WGS84 ne couvre pas le périmètre de feu.")
    if str(perimeter_properties.get("source_feature_id")) != str(
        manifest.get("processing", {}).get("selected_effis_feature_id")
    ):
        raise VerificationError(
            "L'identifiant EFFIS des vecteurs ne correspond pas au manifeste."
        )
    if float(aoi_properties.get("buffer_metres", -1)) != float(
        manifest.get("processing", {}).get("buffer_metres", -2)
    ):
        raise VerificationError(
            "La distance de tampon diffère entre le vecteur et le manifeste."
        )

    expected_area = float(
        catalog.get("extent", {}).get("area_of_interest_square_metres", -1)
    )
    if expected_area <= 0 or abs(aoi_l93.area - expected_area) > max(
        0.01, expected_area * 1e-12
    ):
        raise VerificationError(
            "L'aire Lambert-93 ne correspond pas à celle du catalogue."
        )

    vector_extension = _verify_vector_extension(root, catalog, paths, aoi_l93)

    sources = manifest.get("source_inputs", [])
    source_status = {
        record.get("source_id"): record.get("processing_status") for record in sources
    }
    if (
        source_status.get("ign_lidar_hd_mns_5m")
        != "validated_source_only_not_published"
    ):
        raise VerificationError("Le statut différé du MNS est absent ou incorrect.")

    result = {
        "status": "verified",
        "package_id": manifest.get("package_id"),
        "artifact_count": len(paths) + 1,
        "selected_effis_feature_id": perimeter_properties.get("source_feature_id"),
        "buffer_metres": aoi_properties.get("buffer_metres"),
        "fire_perimeter_area_square_metres": perimeter_l93.area,
        "area_of_interest_square_metres": aoi_l93.area,
        "cog": {
            "path": paths["terrain_mnt"].relative_to(root).as_posix(),
            "dimensions": [cog["width"], cog["height"]],
            "pixel_size_metres": cog["pixel_size_metres"],
            "overviews": cog["overviews"],
            "layout": cog["layout"],
        },
    }
    if vector_extension is not None:
        result["vector_extension"] = vector_extension
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify_package(args.package)
    except VerificationError as exc:
        raise SystemExit(f"ERREUR: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
