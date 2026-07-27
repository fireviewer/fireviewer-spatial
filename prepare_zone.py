#!/usr/bin/env python3
"""Build an isolated, source-traceable 2.5D global zone package.

This command deliberately performs no network access.  It consumes local source
snapshots, selects one EFFIS feature, constructs the metric buffer in Lambert-93
and publishes only the MNT as the global elevation layer.  The MNS is validated
and recorded for later detailed modelling, but is not rendered as terrain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.mask import mask
from rasterio.shutil import copy as raster_copy
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry


WGS84 = CRS.from_epsg(4326)
LAMBERT93 = CRS.from_epsg(2154)
BUFFER_QUAD_SEGS = 16
COG_OVERVIEW_COUNT = 4


class PreparationError(RuntimeError):
    """Raised when a source cannot safely produce the requested package."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def artifact_record(root: Path, path: Path, **metadata: Any) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def source_record(
    path: Path, *, source_id: str, dataset: str, **metadata: Any
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "dataset": dataset,
        "file_name": path.name,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _swap_xy(x: Any, y: Any, z: Any | None = None) -> tuple[Any, ...]:
    if z is None:
        return y, x
    return y, x, z


def _identity_xy(x: Any, y: Any, z: Any | None = None) -> tuple[Any, ...]:
    if z is None:
        return x, y
    return x, y, z


def _assert_wgs84_domain(geometry: BaseGeometry) -> None:
    west, south, east, north = geometry.bounds
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise PreparationError(
            "La géométrie EFFIS normalisée ne se trouve pas dans le domaine WGS84: "
            f"{geometry.bounds}"
        )


def load_effis_feature(
    path: Path,
    fire_id: str,
    *,
    axis_order: str = "lat-lon",
) -> tuple[dict[str, Any], BaseGeometry]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(
            f"Impossible de lire le GeoJSON EFFIS {path}: {exc}"
        ) from exc

    features = payload.get("features")
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise PreparationError(
            "La source EFFIS doit être une FeatureCollection GeoJSON."
        )

    selected = [
        feature
        for feature in features
        if str(feature.get("properties", {}).get("id")) == str(fire_id)
    ]
    if len(selected) != 1:
        raise PreparationError(
            f"L'identifiant EFFIS {fire_id!r} doit sélectionner exactement une entité; "
            f"résultat: {len(selected)}."
        )

    feature = selected[0]
    try:
        raw_geometry = shape(feature["geometry"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparationError(
            f"Géométrie EFFIS invalide pour l'entité {fire_id}: {exc}"
        ) from exc

    coordinate_transform: Callable[..., tuple[Any, ...]]
    if axis_order == "lat-lon":
        coordinate_transform = _swap_xy
    elif axis_order == "lon-lat":
        coordinate_transform = _identity_xy
    else:
        raise PreparationError(f"Ordre d'axes EFFIS non pris en charge: {axis_order!r}")

    geometry_wgs84 = transform_geometry(coordinate_transform, raw_geometry)
    if geometry_wgs84.is_empty or not geometry_wgs84.is_valid:
        raise PreparationError(
            f"La géométrie EFFIS {fire_id} est vide ou topologiquement invalide après normalisation."
        )
    if geometry_wgs84.geom_type not in {"Polygon", "MultiPolygon"}:
        raise PreparationError(
            f"La géométrie EFFIS {fire_id} doit être surfacique; type reçu: "
            f"{geometry_wgs84.geom_type}."
        )
    _assert_wgs84_domain(geometry_wgs84)
    return dict(feature.get("properties", {})), geometry_wgs84


def project_geometry(geometry: BaseGeometry, source: CRS, target: CRS) -> BaseGeometry:
    transformer = Transformer.from_crs(source, target, always_xy=True)
    projected = transform_geometry(transformer.transform, geometry)
    if projected.is_empty or not projected.is_valid:
        raise PreparationError(
            f"La reprojection {source.to_string()} vers {target.to_string()} a produit une géométrie invalide."
        )
    return projected


def build_geometries(
    effis_path: Path,
    fire_id: str,
    buffer_metres: float,
    *,
    axis_order: str,
) -> tuple[dict[str, Any], BaseGeometry, BaseGeometry, BaseGeometry, BaseGeometry]:
    if not math.isfinite(buffer_metres) or buffer_metres <= 0:
        raise PreparationError(
            "La distance de tampon doit être un nombre strictement positif."
        )

    properties, perimeter_wgs84 = load_effis_feature(
        effis_path,
        fire_id,
        axis_order=axis_order,
    )
    perimeter_l93 = project_geometry(perimeter_wgs84, WGS84, LAMBERT93)
    aoi_l93 = perimeter_l93.buffer(buffer_metres, quad_segs=BUFFER_QUAD_SEGS)
    if aoi_l93.is_empty or not aoi_l93.is_valid or not aoi_l93.covers(perimeter_l93):
        raise PreparationError(
            "Le tampon Lambert-93 calculé est invalide ou ne couvre pas le périmètre."
        )
    aoi_wgs84 = project_geometry(aoi_l93, LAMBERT93, WGS84)
    return properties, perimeter_wgs84, perimeter_l93, aoi_wgs84, aoi_l93


def feature_collection(
    *,
    name: str,
    geometry: BaseGeometry,
    properties: dict[str, Any],
    crs: CRS,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "FeatureCollection",
        "name": name,
        "metadata": {"crs": crs.to_string(), **metadata},
        "features": [
            {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(geometry),
            }
        ],
    }
    if crs != WGS84:
        # Projected GeoJSON is intentionally supplied for desktop authoring tools.
        # The RFC 7946 WGS84 file remains the interoperable web representation.
        payload["crs"] = {"type": "name", "properties": {"name": crs.to_string()}}
    return payload


def _grid_metadata(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    return {
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "epsg": dataset.crs.to_epsg() if dataset.crs else None,
        "width": dataset.width,
        "height": dataset.height,
        "bounds": [float(value) for value in dataset.bounds],
        "pixel_size_metres": [
            abs(float(dataset.transform.a)),
            abs(float(dataset.transform.e)),
        ],
        "dtype": dataset.dtypes[0],
        "nodata": dataset.nodata,
    }


def is_lambert93_source_crs(crs: rasterio.crs.CRS | None) -> bool:
    """Recognize EPSG:2154 even in the incomplete WMS-R source WKT.

    The IGN WMS-R response names EPSG:2154 but omits the RGF93 datum name, so
    GDAL cannot resolve ``to_epsg()``.  The projection parameters still match
    Lambert-93 exactly; the published COG is subsequently rewritten with the
    authoritative EPSG definition.
    """

    if crs is None:
        return False
    if crs.to_epsg() == 2154:
        return True
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
    for key, value in expected.items():
        actual = parameters.get(key)
        if isinstance(value, str):
            if actual != value:
                return False
        elif actual is None or abs(float(actual) - value) > 1e-9:
            return False
    return True


def inspect_source_rasters(
    mnt_path: Path, mns_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with rasterio.open(mnt_path) as mnt, rasterio.open(mns_path) as mns:
            for label, dataset in (("MNT", mnt), ("MNS", mns)):
                if dataset.count != 1:
                    raise PreparationError(
                        f"Le {label} doit posséder exactement une bande."
                    )
                if not is_lambert93_source_crs(dataset.crs):
                    raise PreparationError(
                        f"Le {label} doit être en EPSG:2154; CRS reçu: {dataset.crs}."
                    )
                if dataset.width <= 0 or dataset.height <= 0:
                    raise PreparationError(f"La grille {label} est vide.")

            if mnt.width != mns.width or mnt.height != mns.height:
                raise PreparationError(
                    "Les grilles MNT et MNS n'ont pas les mêmes dimensions."
                )
            if mnt.crs != mns.crs or not mnt.transform.almost_equals(mns.transform):
                raise PreparationError(
                    "Les grilles MNT et MNS ne sont pas alignées dans le même référentiel."
                )
            if any(abs(a - b) > 1e-6 for a, b in zip(mnt.bounds, mns.bounds)):
                raise PreparationError("Les emprises raster MNT et MNS diffèrent.")

            return _grid_metadata(mnt), _grid_metadata(mns)
    except rasterio.errors.RasterioIOError as exc:
        raise PreparationError(
            f"Impossible d'ouvrir les sources raster: {exc}"
        ) from exc


def _valid_values(array: np.ndarray, nodata: float | int | None) -> np.ndarray:
    values = np.asarray(array)
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    return values[valid]


def create_mnt_cog(
    mnt_path: Path, output_path: Path, aoi_l93: BaseGeometry
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.staging.tif")
    if temporary_path.exists():
        raise PreparationError(f"Le fichier temporaire existe déjà: {temporary_path}")

    try:
        with rasterio.open(mnt_path) as source:
            if not is_lambert93_source_crs(source.crs):
                raise PreparationError("Le MNT à convertir doit être en EPSG:2154.")
            source_bounds = shape(
                {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [source.bounds.left, source.bounds.bottom],
                            [source.bounds.right, source.bounds.bottom],
                            [source.bounds.right, source.bounds.top],
                            [source.bounds.left, source.bounds.top],
                            [source.bounds.left, source.bounds.bottom],
                        ]
                    ],
                }
            )
            if not source_bounds.covers(aoi_l93):
                raise PreparationError(
                    "Le MNT ne couvre pas entièrement l'emprise EFFIS tamponnée de 1,5 km."
                )

            nodata = source.nodata
            if nodata is None:
                nodata = (
                    -9999.0
                    if np.issubdtype(np.dtype(source.dtypes[0]), np.floating)
                    else 0
                )
            clipped, clipped_transform = mask(
                source,
                [mapping(aoi_l93)],
                crop=True,
                all_touched=False,
                filled=True,
                nodata=nodata,
            )
            valid_values = _valid_values(clipped[0], nodata)
            if valid_values.size == 0:
                raise PreparationError(
                    "Le MNT recadré ne contient aucun échantillon valide."
                )

            profile = source.profile.copy()
            profile.update(
                driver="GTiff",
                # The WMS-R snapshot may carry an incomplete/unnamed WKT even
                # though it resolves to 2154. Publish the authoritative CRS.
                crs=LAMBERT93,
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=clipped_transform,
                count=1,
                nodata=nodata,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="DEFLATE",
                predictor=3
                if np.issubdtype(np.dtype(source.dtypes[0]), np.floating)
                else 2,
            )
            with rasterio.open(temporary_path, "w", **profile) as temporary:
                temporary.write(clipped)
                temporary.update_tags(
                    source_role="MNT",
                    processing="EFFIS perimeter buffered 1500m in EPSG:2154; AOI mask",
                )

        raster_copy(
            temporary_path,
            output_path,
            driver="COG",
            compress="DEFLATE",
            blocksize=512,
            overview_resampling="AVERAGE",
            overview_count=COG_OVERVIEW_COUNT,
            bigtiff="IF_SAFER",
        )
    except rasterio.errors.RasterioError as exc:
        raise PreparationError(f"Échec de production du COG MNT: {exc}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    metadata = verify_cog(output_path)
    metadata.update(
        {
            "valid_pixel_count": int(valid_values.size),
            "masked_pixel_count": int(clipped[0].size - valid_values.size),
            "elevation_min_metres": float(valid_values.min()),
            "elevation_max_metres": float(valid_values.max()),
            "elevation_mean_metres": float(valid_values.mean()),
        }
    )
    return metadata


def verify_cog(path: Path) -> dict[str, Any]:
    try:
        with rasterio.open(path) as dataset:
            if dataset.driver != "GTiff":
                raise PreparationError(
                    f"Le relief publié n'est pas un GeoTIFF: {dataset.driver}."
                )
            if dataset.crs is None or dataset.crs.to_epsg() != 2154:
                raise PreparationError(
                    f"Le COG MNT n'est pas en EPSG:2154: {dataset.crs}."
                )
            is_tiled = bool(dataset.profile.get("tiled"))
            if not is_tiled:
                raise PreparationError("Le GeoTIFF MNT n'est pas tuilé.")
            overviews = dataset.overviews(1)
            if not overviews:
                raise PreparationError(
                    "Le GeoTIFF MNT ne contient aucune vue d'ensemble."
                )
            image_structure = dataset.tags(ns="IMAGE_STRUCTURE")
            if image_structure.get("LAYOUT") != "COG":
                raise PreparationError(
                    "Le GeoTIFF n'annonce pas la structure Cloud Optimized GeoTIFF (LAYOUT=COG)."
                )
            values = dataset.read(1, masked=True)
            if values.count() == 0:
                raise PreparationError("Le COG MNT ne contient aucune altitude valide.")
            return {
                **_grid_metadata(dataset),
                "format": "Cloud Optimized GeoTIFF",
                "is_tiled": is_tiled,
                "block_shape": list(dataset.block_shapes[0]),
                "overviews": overviews,
                "compression": dataset.compression.name
                if dataset.compression
                else None,
                "layout": image_structure.get("LAYOUT"),
            }
    except rasterio.errors.RasterioIOError as exc:
        raise PreparationError(f"Impossible de vérifier le COG {path}: {exc}") from exc


def build_package(
    *,
    mnt_path: Path,
    mns_path: Path,
    effis_path: Path,
    output_dir: Path,
    package_id: str,
    fire_id: str,
    buffer_metres: float,
    axis_order: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_paths = (mnt_path, mns_path, effis_path)
    for source_path in source_paths:
        if not source_path.is_file():
            raise PreparationError(f"Source locale absente: {source_path}")
    if output_dir.exists():
        raise PreparationError(
            f"Le dossier de sortie existe déjà: {output_dir}. Utiliser une nouvelle révision de package."
        )

    generated_at = generated_at or utc_now()
    (
        incident_properties,
        perimeter_wgs84,
        perimeter_l93,
        aoi_wgs84,
        aoi_l93,
    ) = build_geometries(
        effis_path,
        fire_id,
        buffer_metres,
        axis_order=axis_order,
    )
    mnt_source_grid, mns_source_grid = inspect_source_rasters(mnt_path, mns_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    vector_dir = output_dir / "vectors"
    terrain_dir = output_dir / "terrain"

    common_vector_metadata = {
        "generated_at": generated_at,
        "source_feature_id": str(fire_id),
        "source_dataset": "EFFIS MODIS burned-area polygons",
    }
    incident_output_properties = {
        **incident_properties,
        "source_feature_id": str(fire_id),
        "geometry_role": "satellite_detected_burned_area",
        "axis_order_normalized_from": axis_order,
    }
    aoi_properties = {
        "source_feature_id": str(fire_id),
        "geometry_role": "model_area_of_interest",
        "buffer_metres": buffer_metres,
        "buffer_crs": "EPSG:2154",
        "buffer_quad_segs": BUFFER_QUAD_SEGS,
        "area_square_metres": float(aoi_l93.area),
    }

    vector_specs = (
        (
            "fire_perimeter_wgs84",
            vector_dir / "fire-perimeter.geojson",
            perimeter_wgs84,
            incident_output_properties,
            WGS84,
        ),
        (
            "area_of_interest_wgs84",
            vector_dir / "area-of-interest.geojson",
            aoi_wgs84,
            aoi_properties,
            WGS84,
        ),
        (
            "fire_perimeter_l93",
            vector_dir / "fire-perimeter.l93.geojson",
            perimeter_l93,
            incident_output_properties,
            LAMBERT93,
        ),
        (
            "area_of_interest_l93",
            vector_dir / "area-of-interest.l93.geojson",
            aoi_l93,
            aoi_properties,
            LAMBERT93,
        ),
    )

    vector_artifacts: dict[str, dict[str, Any]] = {}
    for artifact_id, path, geometry, properties, crs in vector_specs:
        write_json(
            path,
            feature_collection(
                name=artifact_id.replace("_", "-"),
                geometry=geometry,
                properties=properties,
                crs=crs,
                metadata=common_vector_metadata,
            ),
        )
        vector_artifacts[artifact_id] = artifact_record(
            output_dir,
            path,
            format="GeoJSON",
            crs=crs.to_string(),
            geometry_type=geometry.geom_type,
        )

    cog_path = terrain_dir / "mnt-global.cog.tif"
    cog_metadata = create_mnt_cog(mnt_path, cog_path, aoi_l93)
    terrain_artifact = artifact_record(output_dir, cog_path, **cog_metadata)

    catalog = {
        "schema_version": "1.0",
        "package_id": package_id,
        "generated_at": generated_at,
        "spatial_profile": {
            "horizontal_crs": "EPSG:2154",
            "web_vector_crs": "EPSG:4326",
            "vertical_reference": "NGF-IGN69 (declared by IGN LiDAR-HD source)",
            "horizontal_units": "metres",
            "vertical_units": "metres",
        },
        "incident": {
            "source_feature_id": str(fire_id),
            "fire_date": incident_properties.get("FIREDATE"),
            "final_date": incident_properties.get("FINALDATE"),
            "source_last_update": incident_properties.get("LASTUPDATE"),
            "commune": incident_properties.get("COMMUNE"),
            "reported_area_hectares": incident_properties.get("AREA_HA"),
            "source_class": incident_properties.get("CLASS"),
        },
        "extent": {
            "definition": "EFFIS feature buffered in EPSG:2154",
            "buffer_metres": buffer_metres,
            "buffer_quad_segs": BUFFER_QUAD_SEGS,
            "fire_perimeter_area_square_metres": float(perimeter_l93.area),
            "area_of_interest_square_metres": float(aoi_l93.area),
            "bounds_l93_metres": [float(value) for value in aoi_l93.bounds],
            "bounds_wgs84_degrees": [float(value) for value in aoi_wgs84.bounds],
            "raster_bounds_l93_metres": terrain_artifact["bounds"],
            "raster_envelope_note": (
                "Pixel-aligned 5 m envelope; pixels outside the exact vector AOI are nodata."
            ),
        },
        "layers": {
            "terrain_mnt": {
                **terrain_artifact,
                "role": "global_2_5d_elevation",
                "mask": "area_of_interest_l93",
            },
            **vector_artifacts,
        },
        "deferred_layers": {
            "mns": {
                "status": "validated_source_only_not_published",
                "intended_use": "local detailed modelling and canopy/building-height derivation",
                "source_grid": mns_source_grid,
            },
            "buildings": {"status": "not_processed"},
            "vegetation_blocks": {"status": "not_processed"},
            "hedges": {"status": "not_processed"},
        },
        "limitations": [
            "The EFFIS polygon is a satellite-detected burned-area envelope, not a tactical fire-front survey.",
            "The global terrain uses the MNT only; canopy and roofs from the MNS are intentionally absent.",
            "No BD TOPO building, vegetation or hedge data is included in this revision.",
        ],
    }
    catalog_path = output_dir / "catalog.json"
    write_json(catalog_path, catalog)

    source_inputs = [
        source_record(
            mnt_path,
            source_id="ign_lidar_hd_mnt_5m",
            dataset="IGN LiDAR-HD MNT",
            service="https://data.geopf.fr/wms-r",
            layer="IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93",
            license="Licence Ouverte 2.0",
            processing_status="published_as_masked_cog",
            grid=mnt_source_grid,
        ),
        source_record(
            mns_path,
            source_id="ign_lidar_hd_mns_5m",
            dataset="IGN LiDAR-HD MNS",
            service="https://data.geopf.fr/wms-r",
            layer="IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93",
            license="Licence Ouverte 2.0",
            processing_status="validated_source_only_not_published",
            grid=mns_source_grid,
        ),
        source_record(
            effis_path,
            source_id="effis_modis_burned_area_snapshot",
            dataset="EFFIS MODIS burned-area polygons",
            service="https://maps.effis.emergency.copernicus.eu/effis",
            layer="ms:modis.ba.poly",
            license="CC BY 4.0",
            processing_status="feature_selected_and_axis_normalized",
            selected_feature_id=str(fire_id),
            input_axis_order=axis_order,
        ),
    ]
    manifest = {
        "schema_version": "1.0",
        "package_id": package_id,
        "generated_at": generated_at,
        "catalog": artifact_record(output_dir, catalog_path, format="JSON"),
        "processing": {
            "network_access": "none",
            "selected_effis_feature_id": str(fire_id),
            "effis_axis_order": axis_order,
            "buffer_crs": "EPSG:2154",
            "buffer_metres": buffer_metres,
            "buffer_quad_segs": BUFFER_QUAD_SEGS,
            "mnt_publication": "AOI-masked Cloud Optimized GeoTIFF",
            "mns_publication": "deferred",
        },
        "source_inputs": source_inputs,
        "provenance_note": (
            "All outputs are derived locally from the three hashed snapshots above; "
            "this command performs no download."
        ),
    }
    manifest_path = output_dir / "package-manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnt", type=Path, required=True, help="Local IGN MNT GeoTIFF")
    parser.add_argument("--mns", type=Path, required=True, help="Local IGN MNS GeoTIFF")
    parser.add_argument(
        "--effis", type=Path, required=True, help="Local EFFIS GeoJSON snapshot"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New isolated package directory"
    )
    parser.add_argument(
        "--package-id",
        default="fireviewer-synthetic-zone-fire-2026-global-v1",
        help="Stable package identifier",
    )
    parser.add_argument("--fire-id", default="SYNTH-001", help="EFFIS feature id")
    parser.add_argument("--buffer-metres", type=float, default=1500.0)
    parser.add_argument(
        "--effis-axis-order",
        choices=("lat-lon", "lon-lat"),
        default="lat-lon",
        help="Coordinate order found in the local EFFIS snapshot",
    )
    parser.add_argument(
        "--generated-at",
        help="Optional fixed ISO-8601 timestamp for reproducible tests/releases",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_package(
            mnt_path=args.mnt.resolve(),
            mns_path=args.mns.resolve(),
            effis_path=args.effis.resolve(),
            output_dir=args.output.resolve(),
            package_id=args.package_id,
            fire_id=str(args.fire_id),
            buffer_metres=args.buffer_metres,
            axis_order=args.effis_axis_order,
            generated_at=args.generated_at,
        )
    except PreparationError as exc:
        raise SystemExit(f"ERREUR: {exc}") from exc
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
