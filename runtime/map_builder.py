#!/usr/bin/env python3
"""Cloud-neutral entrypoint for the frozen FireViewer Map Builder.

The adapter translates ``fireviewer.map-job.v1`` into the restored builder's
existing request and environment contract.  It contains no cloud SDK and does
not alter any spatial, placement, asset-selection or export rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from resource_monitor import ResourceMonitor

JOB_SCHEMA = "fireviewer.map-job.v1"
METRICS_SCHEMA = "fireviewer.map-build-metrics.v1"
HASHES_SCHEMA = "fireviewer.map-build-hashes.v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
SCRATCH_DIRECTORIES = (
    "inputs",
    "downloads",
    "extracted",
    "geographic",
    "rasters",
    "vectors",
    "textures",
    "tiles",
    "blender",
    "export",
    "validation",
)


class MapBuilderContractError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MapBuilderContractError(f"invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise MapBuilderContractError("map job must be a JSON object")
    return payload


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise MapBuilderContractError(f"{key} must be an object")
    return value


def normalize_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != JOB_SCHEMA:
        raise MapBuilderContractError(f"schema must be {JOB_SCHEMA}")
    build_id = payload.get("build_id")
    zone_id = payload.get("zone_id")
    if not isinstance(build_id, str) or not SAFE_ID.fullmatch(build_id):
        raise MapBuilderContractError("build_id is invalid")
    if not isinstance(zone_id, str) or not SAFE_ID.fullmatch(zone_id):
        raise MapBuilderContractError("zone_id is invalid")
    center = _required_mapping(payload, "center")
    try:
        latitude = float(center["lat"])
        longitude = float(center["lon"])
        side_m = int(payload["side_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise MapBuilderContractError("center and side_m are invalid") from error
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise MapBuilderContractError("center coordinates are outside valid bounds")
    if side_m <= 0 or side_m % 500 != 0:
        raise MapBuilderContractError("side_m must be a positive multiple of 500")
    builder = _required_mapping(payload, "builder")
    git_commit = builder.get("git_commit")
    image_digest = builder.get("image_digest")
    if not isinstance(git_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise MapBuilderContractError("builder.git_commit must be a full Git SHA")
    if not isinstance(image_digest, str) or not SHA256.fullmatch(image_digest):
        raise MapBuilderContractError("builder.image_digest must be a SHA-256 digest")
    profile = payload.get("profile", "factual-v2")
    if profile not in {"legacy-v1", "factual-v2"}:
        raise MapBuilderContractError("profile is unsupported")
    fixed = payload.get("fixed_asset_placements")
    if fixed is not None and not isinstance(fixed, Mapping):
        raise MapBuilderContractError("fixed_asset_placements must be null or an object")
    return {
        **dict(payload),
        "build_id": build_id,
        "zone_id": zone_id,
        "center": {"lat": latitude, "lon": longitude},
        "side_m": side_m,
        "profile": profile,
        "builder": dict(builder),
        "fixed_asset_placements": dict(fixed) if fixed is not None else None,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_layout(scratch_root: Path, build_id: str) -> Path:
    run_root = (scratch_root / build_id).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise MapBuilderContractError(f"scratch build directory is not empty: {run_root}")
    for name in SCRATCH_DIRECTORIES:
        (run_root / name).mkdir(parents=True, exist_ok=True)
    return run_root


def _find_job_root(run_root: Path, expected_zone: str) -> tuple[Path, dict[str, Any]]:
    candidates = list((run_root / "blender").glob("jobs/*/validation-result.json"))
    candidates.extend((run_root / "export").glob("jobs/*/validation-result.json"))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for result_path in candidates:
        result = _load_json(result_path)
        if result.get("status") == "completed" and result.get("zone_id") == expected_zone:
            matches.append((result_path.parent, result))
    unique = {root.resolve(): result for root, result in matches}
    if len(unique) != 1:
        raise MapBuilderContractError(
            f"expected one completed result for {expected_zone}, found {len(unique)}"
        )
    root, result = next(iter(unique.items()))
    return root, result


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise MapBuilderContractError(f"required artifact is missing: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise MapBuilderContractError(f"required artifact directory is missing: {source.name}")
    shutil.copytree(source, destination)


def publish_output(
    job_root: Path,
    output_root: Path,
    job: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish an atomic local folder; zone.done.json is copied last."""

    if output_root.exists() and any(output_root.iterdir()):
        raise MapBuilderContractError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    _copy_file(job_root / "viewer.glb", output_root / "runtime" / "viewer.glb")
    _copy_file(
        job_root / "viewer-scene.v1.json",
        output_root / "runtime" / "viewer-scene.v1.json",
    )
    _copy_directory(job_root / "viewer-tiled", output_root / "runtime" / "viewer-tiled")
    _copy_file(job_root / "zone.usda", output_root / "scientific" / "zone.usda")
    _copy_file(job_root / "zone.blend", output_root / "scientific" / "zone.blend")
    _copy_directory(job_root / "packages", output_root / "packages")
    _copy_directory(job_root / "provenance", output_root / "provenance")
    if (job_root / "contracts").is_dir():
        _copy_directory(job_root / "contracts", output_root / "manifests" / "contracts")

    manifest_names = (
        "manifest.json",
        "dependency-inventory.json",
        "zone-plan.json",
        "zone-context.json",
        "zone-stage-layout.v1.json",
        "validation-result.json",
    )
    for name in manifest_names:
        _copy_file(job_root / name, output_root / "manifests" / name)
    _write_json(output_root / "manifests" / "request.json", job)

    phase_times = {
        "download": None,
        "extract": None,
        "raster_processing": None,
        "vector_processing": None,
        "terrain": None,
        "placement": None,
        "blender": validation_result.get("timings_seconds", {}).get("sealed_map"),
        "export": validation_result.get("timings_seconds", {}).get("viewer_export"),
        "validation": None,
        "upload": 0.0,
    }
    metrics = {
        "schema": METRICS_SCHEMA,
        "build_id": job["build_id"],
        "zone_id": job["zone_id"],
        "builder": job["builder"],
        "builder_contract": JOB_SCHEMA,
        "runtime_seconds": validation_result.get("timings_seconds", {}).get("total"),
        "phase_times": phase_times,
        "phase_time_basis": (
            "The frozen builder exposes sealed-map and viewer-export durations. "
            "Unavailable overlapping sub-phases remain null rather than fabricated."
        ),
        "resources": dict(resources),
        "counts": validation_result.get("counts"),
        "tile_count": validation_result.get("tile_count"),
    }
    _write_json(output_root / "metrics" / "build-metrics.json", metrics)

    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.json", "zone.done.json"}:
            artifacts.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "byte_count": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    zone_receipt = job_root / "zone.done.json"
    if not zone_receipt.is_file():
        raise MapBuilderContractError("zone.done.json is missing from the validated build")
    artifacts.append(
        {
            "path": "zone.done.json",
            "byte_count": zone_receipt.stat().st_size,
            "sha256": _sha256(zone_receipt),
            "publication_order": "last",
        }
    )
    hashes = {
        "schema": HASHES_SCHEMA,
        "algorithm": "sha256",
        "build_id": job["build_id"],
        "artifacts": artifacts,
    }
    _write_json(output_root / "manifests" / "hashes.json", hashes)

    # This is the validity marker.  No operation may follow that can leave an
    # apparently complete folder with missing artifacts.
    _copy_file(zone_receipt, output_root / "zone.done.json")
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    request_path = Path(args.request).resolve(strict=True)
    job = normalize_job(_load_json(request_path))
    run_root = _prepare_layout(Path(args.scratch_root).resolve(), job["build_id"])
    shutil.copy2(request_path, run_root / "inputs" / "request.json")
    builder_request = {
        "latitude": job["center"]["lat"],
        "longitude": job["center"]["lon"],
        "side_km": job["side_m"] / 1000.0,
        "fixed_asset_placements": job.get("fixed_asset_placements"),
    }
    builder_request_path = run_root / "inputs" / "builder-request.json"
    _write_json(builder_request_path, builder_request)

    environment = os.environ.copy()
    environment.update(
        {
            "FIREVIEWER_WORK_ROOT": str(run_root / "export"),
            "FIREVIEWER_SCRATCH_ROOT": str(run_root / "blender"),
            "FIREVIEWER_VALIDATION_PROVIDER": "container",
            "FIREVIEWER_VALIDATION_PROFILE": str(job["profile"]),
            "FIREVIEWER_VALIDATION_REQUEST_JSON": "",
            "FIREVIEWER_VALIDATION_REQUEST_FILE": str(builder_request_path),
            "FIREVIEWER_VALIDATION_REQUIRE_NINE_TILES": "0",
            "FIREVIEWER_VALIDATION_CREATE_EVIDENCE": "0",
            "FIREVIEWER_VALIDATION_REQUIRE_EVIDENCE_UPLOAD": "0",
            "FIREVIEWER_VALIDATION_PUBLISH_VIEWER": "0",
            "FIREVIEWER_VALIDATION_REQUIRE_VIEWER_PUBLICATION": "0",
            "FIREVIEWER_IMAGE_REFERENCE": str(job["builder"]["image_digest"]),
            "TMPDIR": str(run_root / "blender" / "tmp"),
            "TMP": str(run_root / "blender" / "tmp"),
            "TEMP": str(run_root / "blender" / "tmp"),
            "PYTHONPYCACHEPREFIX": str(run_root / "blender" / "python-cache"),
            "XDG_CACHE_HOME": str(run_root / "downloads" / "cache"),
            "HF_XET_CACHE": str(run_root / "downloads" / "hf-xet"),
            "CPL_TMPDIR": str(run_root / "rasters"),
        }
    )
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    monitor = ResourceMonitor(run_root)
    monitor.start()
    log_path = run_root / "validation" / "builder.log"
    return_code = -1
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(args.builder_entrypoint).resolve(strict=True))],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    resources = monitor.stop()
    if return_code != 0:
        raise MapBuilderContractError(f"frozen builder exited with code {return_code}")

    job_root, validation_result = _find_job_root(run_root, str(job["zone_id"]))
    metrics = publish_output(
        job_root,
        Path(args.output).resolve(),
        job,
        validation_result,
        resources,
    )
    result = {
        "status": "completed",
        "schema": JOB_SCHEMA,
        "build_id": job["build_id"],
        "zone_id": job["zone_id"],
        "output": str(Path(args.output).resolve()),
        "runtime_seconds": metrics["runtime_seconds"],
    }
    print("FIREVIEWER_MAP_BUILD_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--scratch-root", default="/scratch")
    parser.add_argument("--output", default="/output")
    parser.add_argument(
        "--builder-entrypoint",
        default="/opt/fireviewer/fireviewer-spatial/blender/map_validation_job.py",
    )
    args = parser.parse_args()
    try:
        run(args)
    except (MapBuilderContractError, OSError, subprocess.SubprocessError) as error:
        print(f"FIREVIEWER_MAP_BUILD_ERROR {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
