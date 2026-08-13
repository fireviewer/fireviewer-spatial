"""RunPod Serverless entrypoint for FireViewer map production.

The handler returns only small immutable metadata. The standalone ZIP is
published by the production engine to the configured private Hugging Face
dataset; no large payload is returned through the RunPod queue API.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from fixed_asset_placement import (
    EMPTY_REQUEST,
    FixedAssetPlacementError,
    normalize_request as normalize_fixed_assets,
    project_request as project_fixed_assets,
    request_sha256 as fixed_assets_request_sha256,
)
from simple_production_api import _plan_payload
from simple_production_engine import ProductionConfig, ProductionEngine, plan_zone

SCHEMA = "fireviewer.runpod-map-worker.v1"
RESULT_SCHEMA = "fireviewer.runpod-map-result.v1"

_ENGINE: ProductionEngine | None = None
_ENGINE_LOCK = Lock()


class RunPodMapContractError(ValueError):
    """The RunPod job input or production result violates the worker contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def normalize_map_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RunPodMapContractError("La requête de carte doit être un objet JSON")
    allowed = {"latitude", "longitude", "side_km", "fixed_asset_placements"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise RunPodMapContractError(
            "Champs de requête inattendus: " + ", ".join(unexpected)
        )
    try:
        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])
        side_km = float(payload["side_km"])
    except (KeyError, TypeError, ValueError) as error:
        raise RunPodMapContractError(
            "latitude, longitude et side_km sont obligatoires"
        ) from error
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise RunPodMapContractError("Coordonnées WGS84 hors limites")
    if not (0.5 <= side_km <= 15):
        raise RunPodMapContractError("side_km doit rester entre 0,5 et 15 km")
    fixed = payload.get("fixed_asset_placements")
    if fixed is not None and not isinstance(fixed, Mapping):
        raise RunPodMapContractError("fixed_asset_placements doit être un objet JSON")
    return {
        "latitude": latitude,
        "longitude": longitude,
        "side_km": side_km,
        "fixed_asset_placements": dict(fixed) if fixed is not None else None,
    }


def request_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(normalize_map_request(payload))).hexdigest()


def _engine() -> ProductionEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = ProductionEngine(ProductionConfig.from_environment())
    return _ENGINE


def _config(engine: ProductionEngine) -> dict[str, Any]:
    config = engine.config
    return {
        "schema": "fireviewer.simple-production-api.v1",
        "limits": {
            "minimum_side_km": 0.5,
            "maximum_side_km": config.max_side_m / 1000,
            "maximum_tiles": config.max_tiles,
            "tile_size_m": 500,
            "parallel_tile_workers": config.tile_workers,
        },
        "assets": {
            "count": engine.asset_summary["asset_count"],
            "catalog_revision": engine.asset_summary["catalog_revision"],
            "choices": [
                {"label": label, "asset_id": asset_id}
                for label, asset_id in engine.fixed_asset_choices
            ],
        },
        "fixed_asset_template": {
            "schema": "fireviewer.fixed-asset-placement-request.v1",
            "crs": "EPSG:4326",
            "placements": [],
        },
        "capabilities": {
            "map_production": True,
            "perimeter_production": False,
            "fixed_asset_placement": True,
            "human_auto_acceptance": False,
            "provider_runpod_serverless": True,
            "degraded_mns_fallback": True,
        },
    }


def _plan(payload: Any, config: ProductionConfig) -> dict[str, Any]:
    request = normalize_map_request(payload)
    return _plan_payload(
        plan_zone(
            request["latitude"],
            request["longitude"],
            request["side_km"],
            max_side_m=config.max_side_m,
            max_tiles=config.max_tiles,
        )
    )


def _validate_fixed_assets(payload: Any, engine: ProductionEngine) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) - {
        "request",
        "latitude",
        "longitude",
        "side_km",
    }:
        raise RunPodMapContractError("Requête de placement invalide")
    try:
        fixed = normalize_fixed_assets(
            payload.get("request"), engine.asset_library_payload
        )
        projected: list[dict[str, Any]] = []
        coordinates = (
            payload.get("latitude"),
            payload.get("longitude"),
            payload.get("side_km"),
        )
        if all(value is not None for value in coordinates):
            plan = plan_zone(
                float(coordinates[0]),
                float(coordinates[1]),
                float(coordinates[2]),
                max_side_m=engine.config.max_side_m,
                max_tiles=engine.config.max_tiles,
            )
            projected = list(
                project_fixed_assets(
                    fixed,
                    engine.asset_library_payload,
                    requested_bounds_l93_m=plan.requested_bounds_l93_m,
                )
            )
    except (FixedAssetPlacementError, TypeError, ValueError) as error:
        raise RunPodMapContractError(str(error)) from error
    return {
        "request": fixed,
        "request_sha256": fixed_assets_request_sha256(fixed),
        "placement_count": len(fixed["placements"]),
        "projected": projected,
    }


def _progress(job: Mapping[str, Any], fraction: float, message: str) -> None:
    try:
        import runpod

        runpod.serverless.progress_update(
            job,
            json.dumps(
                {
                    "schema": "fireviewer.runpod-map-progress.v1",
                    "progress": max(0.0, min(1.0, float(fraction))),
                    "message": str(message),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    except ImportError:  # local unit tests deliberately do not require runpod
        return


def _produce(
    job: Mapping[str, Any], payload: Any, engine: ProductionEngine
) -> dict[str, Any]:
    request = normalize_map_request(payload)
    fixed = normalize_fixed_assets(
        request["fixed_asset_placements"] or dict(EMPTY_REQUEST),
        engine.asset_library_payload,
    )
    archive_path: Path | None = None
    gallery: list[tuple[str, str]] = []
    final_message = "Production terminée"
    for message, archive, items in engine.run(
        request["latitude"],
        request["longitude"],
        request["side_km"],
        progress_callback=lambda fraction, text: _progress(job, fraction, text),
        fixed_asset_placements=fixed,
    ):
        final_message = message
        if archive:
            archive_path = Path(archive)
        if items:
            gallery = items
    if archive_path is None or not archive_path.is_file() or gallery:
        raise RunPodMapContractError("Le moteur n'a pas publié le pack complet")
    job_root = archive_path.parent
    publication = json.loads(
        (job_root / "dataset-publication.json").read_text(encoding="utf-8")
    )
    receipt = json.loads((job_root / "zone.done.json").read_text(encoding="utf-8"))
    captures = publication.get("captures")
    if captures != []:
        raise RunPodMapContractError(
            "La publication privée contient une galerie obsolète"
        )
    remote_root = publication["path_in_repo"]
    return {
        "schema": RESULT_SCHEMA,
        "status": "technical_scene_produced",
        "request_sha256": request_sha256(request),
        "zone_id": receipt["zone_id"],
        "build_id": receipt["build_id"],
        "message": final_message,
        "tile_count": receipt["tile_count"],
        "degraded_mns_tile_count": receipt.get("degraded_mns_tile_count", 0),
        "dataset": {
            "repo_id": publication["dataset_id"],
            "revision": publication["commit_oid"],
            "root": remote_root,
        },
        "archive": {
            "path": f"{remote_root}/fireviewer-zone.zip",
            "sha256": publication["archive_sha256"],
        },
        "captures": [
            {
                "index": record["index"],
                "caption": record["caption"],
                "path": f"{remote_root}/{record['file']}",
                "sha256": record["sha256"],
            }
            for record in captures
        ],
    }


def handler(job: Mapping[str, Any]) -> dict[str, Any]:
    payload = job.get("input")
    if not isinstance(payload, Mapping):
        raise RunPodMapContractError("RunPod input doit être un objet JSON")
    operation = payload.get("operation")
    engine = _engine()
    if operation == "config":
        return _config(engine)
    if operation == "plan":
        return _plan(payload.get("request"), engine.config)
    if operation == "validate_fixed_assets":
        return _validate_fixed_assets(payload.get("request"), engine)
    if operation == "produce_map":
        return _produce(job, payload.get("request"), engine)
    raise RunPodMapContractError("Opération RunPod inconnue")


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()


__all__ = [
    "RESULT_SCHEMA",
    "RunPodMapContractError",
    "handler",
    "normalize_map_request",
    "request_sha256",
]
