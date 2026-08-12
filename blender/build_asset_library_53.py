"""Build the immutable FireViewer catalogue for the reviewed 53-USD batch.

The source GLB validation report currently labels every entry ``asset_id=glb``.
That field is deliberately rejected as identity evidence.  Identities come from
artifact stems and must resolve to the hash-locked reference manifest.

This module only catalogues assets.  It does not select, place, scale, rotate,
or instantiate anything in a terrain or scene.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

CATALOG_SCHEMA = "fireviewer.asset-library.v1"
CATALOG_STATUS = "catalogued_pending_simready_qualification"
CONTRACT_SCHEMA = "fireviewer.asset-library-contract.v1"
BUILD_ALGORITHM = "fireviewer.asset-library-53-builder.v1"
IDENTITY_STRATEGY = "artifact_stem+reference_manifest"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ASSET_ID_PATTERN = re.compile(r"^[0-9a-f]{12}_[a-z0-9_]+$")
CATEGORIES = (
    "building",
    "pasture_equipment",
    "road_equipment",
    "tree",
    "vehicle",
)
QUALIFICATION_FIELDS = (
    "collision",
    "dimensions",
    "forward_axis",
    "ground_anchor",
    "lod",
    "scale",
    "visual",
)


class AssetLibraryBuildError(ValueError):
    """The reviewed batch cannot produce an unambiguous immutable catalogue."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return stable UTF-8 JSON bytes for hashes and deterministic comparisons."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_seed(
    zone: str,
    candidate: str,
    catalog_revision: str,
    rule_version: str,
) -> int:
    """Return a deterministic 64-bit seed without performing asset placement."""

    values = {
        "zone": zone,
        "candidate": candidate,
        "catalog_revision": catalog_revision,
        "rule_version": rule_version,
    }
    for label, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise AssetLibraryBuildError(f"{label} must be a non-empty string")
        if "\x00" in value:
            raise AssetLibraryBuildError(f"{label} must not contain NUL")
    if SHA256_PATTERN.fullmatch(catalog_revision) is None:
        raise AssetLibraryBuildError("catalog_revision must be a lowercase SHA-256")
    payload = f"{zone}\x00{candidate}\x00{catalog_revision}\x00{rule_version}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def select_asset_for_candidate(
    library: Mapping[str, Any],
    *,
    category: str,
    zone: str,
    candidate: str,
    rule_version: str,
    usage: str,
    contract_path: Path | str | None = None,
) -> dict[str, Any]:
    """Select one stable prototype id without calculating or applying placement.

    ``final_scene`` is fail-closed until the selected pool is fully qualified and
    visually accepted.  The explicit ``technical_pilot_non_final`` mode exposes
    current prototypes to Blender qualification without presenting them as final.
    """

    validate_asset_library(library, contract_path=contract_path)
    pools = library["selection_pools"]
    if category not in pools:
        raise AssetLibraryBuildError(
            f"No deterministic selection pool for category: {category}"
        )
    pool = pools[category]
    by_id = {asset["asset_id"]: asset for asset in library["assets"]}
    if usage == "final_scene":
        blocked = [
            asset_id
            for asset_id in pool
            if by_id[asset_id]["qualification"]["visual"]["accepted"] is not True
            or by_id[asset_id]["eligibility"]["final_scene"] != "allowed"
        ]
        if blocked:
            raise AssetLibraryBuildError(
                "Final scene selection is blocked by pending visual/normalization "
                "qualification"
            )
        usage_status = "final_scene"
    elif usage == "technical_pilot_non_final":
        if any(
            by_id[asset_id]["eligibility"]["technical_pilot"] != "allowed_non_final"
            for asset_id in pool
        ):
            raise AssetLibraryBuildError("Technical pilot selection is not allowed")
        usage_status = "technical_pilot_non_final"
    else:
        raise AssetLibraryBuildError(
            "usage must be final_scene or technical_pilot_non_final"
        )
    seed = selection_seed(
        zone,
        candidate,
        library["catalog_revision"],
        rule_version,
    )
    asset_id = pool[seed % len(pool)]
    return {
        "asset_id": asset_id,
        "category": category,
        "selection_seed": seed,
        "usage_status": usage_status,
        "visual_accepted": by_id[asset_id]["qualification"]["visual"]["accepted"],
    }


def _require_d_path(
    path: Path | str,
    label: str,
    *,
    kind: str | None = None,
) -> Path:
    candidate = Path(path).resolve()
    if os.name == "nt":
        if candidate.drive.casefold() != "d:":
            raise AssetLibraryBuildError(f"{label} must be stored on D: {candidate}")
    elif not candidate.is_absolute():
        raise AssetLibraryBuildError(
            f"{label} must be an absolute container path: {candidate}"
        )
    if kind == "file" and not candidate.is_file():
        raise AssetLibraryBuildError(f"{label} is not a file: {candidate}")
    if kind == "directory" and not candidate.is_dir():
        raise AssetLibraryBuildError(f"{label} is not a directory: {candidate}")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetLibraryBuildError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise AssetLibraryBuildError(f"{label} must be a JSON object")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise AssetLibraryBuildError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AssetLibraryBuildError(f"{label} must be a positive integer")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise AssetLibraryBuildError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AssetLibraryBuildError(f"{label} must be finite") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise AssetLibraryBuildError(f"{label} must be finite and positive")
    return result


def _portable_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AssetLibraryBuildError(f"{label} must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0].endswith(":"):
        raise AssetLibraryBuildError(f"{label} must be a portable relative path")
    return path.as_posix()


def _contained_file(root: Path, path: Path | str, label: str) -> Path:
    candidate = _require_d_path(path, label, kind="file")
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise AssetLibraryBuildError(f"{label} escapes {root}") from error
    return candidate


def _artifact(
    *,
    logical_root: str,
    physical_root: Path,
    path: Path,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    file_path = _contained_file(physical_root, path, label)
    relative = file_path.relative_to(physical_root).as_posix()
    _portable_path(relative, f"{label}.path")
    byte_count = file_path.stat().st_size
    digest = sha256_file(file_path)
    if expected_bytes is not None and byte_count != expected_bytes:
        raise AssetLibraryBuildError(f"{label} byte count differs from its manifest")
    if expected_sha256 is not None and digest != expected_sha256:
        raise AssetLibraryBuildError(f"{label} SHA-256 differs from its manifest")
    return {
        "root": logical_root,
        "path": relative,
        "byte_count": byte_count,
        "sha256": digest,
    }


def _load_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path, "asset library contract")
    required = {
        "schema",
        "catalog_schema",
        "build_algorithm",
        "expected_asset_count",
        "expected_category_counts",
        "expected_selection_pool_counts",
        "identity",
        "category_rules",
        "category_overrides",
        "usd_stage",
        "normalization_policy",
        "qualification_defaults",
        "eligibility_defaults",
    }
    if set(contract) != required:
        raise AssetLibraryBuildError("Asset library contract keys differ")
    if contract["schema"] != CONTRACT_SCHEMA:
        raise AssetLibraryBuildError("Unsupported asset library contract")
    if contract["catalog_schema"] != CATALOG_SCHEMA:
        raise AssetLibraryBuildError("Contract targets another catalogue schema")
    if contract["build_algorithm"] != BUILD_ALGORITHM:
        raise AssetLibraryBuildError("Contract targets another build algorithm")
    _positive_integer(contract["expected_asset_count"], "expected_asset_count")
    counts = contract["expected_category_counts"]
    if not isinstance(counts, dict) or set(counts) != set(CATEGORIES):
        raise AssetLibraryBuildError("Contract category counts differ")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise AssetLibraryBuildError("Contract category counts must be integers")
    if sum(counts.values()) != contract["expected_asset_count"]:
        raise AssetLibraryBuildError("Contract category counts do not total 53")
    pool_counts = contract["expected_selection_pool_counts"]
    if pool_counts != {"building": 24, "tree": 18}:
        raise AssetLibraryBuildError("Contract selection pool counts differ")
    identity = contract["identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "strategy",
        "invalid_reported_asset_ids",
    }:
        raise AssetLibraryBuildError("Contract identity policy differs")
    if identity["strategy"] != IDENTITY_STRATEGY:
        raise AssetLibraryBuildError("Contract identity strategy differs")
    invalid_ids = identity["invalid_reported_asset_ids"]
    if not isinstance(invalid_ids, list) or "glb" not in invalid_ids:
        raise AssetLibraryBuildError("Contract must reject the erroneous 'glb' id")
    defaults = contract["qualification_defaults"]
    if not isinstance(defaults, dict) or set(defaults) != set(QUALIFICATION_FIELDS):
        raise AssetLibraryBuildError("Contract qualification defaults differ")
    if any(
        not isinstance(value, dict) or value.get("status") != "pending"
        for value in defaults.values()
    ):
        raise AssetLibraryBuildError("Unqualified asset fields must remain pending")
    if defaults["visual"] != {"accepted": False, "status": "pending"}:
        raise AssetLibraryBuildError("Visual acceptance must remain fail-closed")
    if defaults["dimensions"].get("value_m") is not None:
        raise AssetLibraryBuildError("Physical dimensions must not be invented")
    if defaults["ground_anchor"].get("offset_m") is not None:
        raise AssetLibraryBuildError("Ground anchor must not be invented")
    if defaults["forward_axis"].get("value") is not None:
        raise AssetLibraryBuildError("Forward axis must not be invented")
    if defaults["scale"].get("uniform_scale") is not None:
        raise AssetLibraryBuildError("Physical scale must not be invented")
    if defaults["lod"].get("levels") != []:
        raise AssetLibraryBuildError("LOD levels must not be invented")
    if contract["eligibility_defaults"] != {
        "final_scene": "blocked",
        "technical_pilot": "allowed_non_final",
    }:
        raise AssetLibraryBuildError("Asset eligibility defaults differ")
    if contract["normalization_policy"] != {
        "bounds_are_physical_dimensions": False,
        "final_scene_requires_accepted_normalization": True,
        "source_stage_units_are_evidence_only": True,
        "technical_pilot_status": "technical_pilot_non_final",
    }:
        raise AssetLibraryBuildError("Asset normalization policy differs")
    return contract


def _classify(source_relative: str, contract: Mapping[str, Any]) -> str:
    overrides = contract["category_overrides"]
    if source_relative in overrides:
        category = overrides[source_relative]
    else:
        matches = [
            rule["category"]
            for rule in contract["category_rules"]
            if source_relative.startswith(rule["source_prefix"])
        ]
        if len(matches) != 1:
            raise AssetLibraryBuildError(
                f"Source category is ambiguous or absent: {source_relative}"
            )
        category = matches[0]
    if category not in CATEGORIES:
        raise AssetLibraryBuildError(f"Unsupported asset category: {category}")
    return category


def _index_reference_assets(
    manifest: Mapping[str, Any],
    reference_root: Path,
) -> dict[str, Mapping[str, Any]]:
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise AssetLibraryBuildError("Reference manifest assets must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(assets):
        if not isinstance(record, dict):
            raise AssetLibraryBuildError(f"Reference asset {index} must be an object")
        asset_id = record.get("asset_id")
        if (
            not isinstance(asset_id, str)
            or ASSET_ID_PATTERN.fullmatch(asset_id) is None
        ):
            raise AssetLibraryBuildError(f"Invalid reference asset id at index {index}")
        if asset_id in result:
            raise AssetLibraryBuildError(f"Duplicate reference asset id: {asset_id}")
        relative = _portable_path(record.get("source_relative"), "source_relative")
        source = reference_root.joinpath(*PurePosixPath(relative).parts).resolve()
        try:
            source.relative_to(reference_root)
        except ValueError as error:
            raise AssetLibraryBuildError(
                f"Reference source escapes its root: {asset_id}"
            ) from error
        declared_source = _require_d_path(record.get("source"), "declared source")
        if declared_source != source:
            raise AssetLibraryBuildError(
                f"Reference source path differs for {asset_id}"
            )
        _positive_integer(record.get("source_bytes"), "source_bytes")
        _sha256(record.get("source_sha256"), "source_sha256")
        result[asset_id] = record
    return result


def _index_conversion_results(
    manifest: Mapping[str, Any],
    batch_root: Path,
) -> dict[str, Mapping[str, Any]]:
    if manifest.get("passed") is not True or manifest.get("failed_count") != 0:
        raise AssetLibraryBuildError("USD conversion manifest is not fully passed")
    results = manifest.get("results")
    if not isinstance(results, list):
        raise AssetLibraryBuildError("USD conversion results must be an array")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(results):
        if not isinstance(record, dict) or record.get("passed") is not True:
            raise AssetLibraryBuildError(f"USD conversion result {index} did not pass")
        asset_id = record.get("asset")
        if (
            not isinstance(asset_id, str)
            or ASSET_ID_PATTERN.fullmatch(asset_id) is None
        ):
            raise AssetLibraryBuildError(
                f"Invalid conversion asset id at index {index}"
            )
        if asset_id in indexed:
            raise AssetLibraryBuildError(f"Duplicate conversion asset id: {asset_id}")
        source_glb = _contained_file(
            batch_root,
            record.get("source_glb"),
            f"source GLB {asset_id}",
        )
        usd = _contained_file(batch_root, record.get("usd"), f"USD {asset_id}")
        if source_glb.parent != batch_root / "glb" or source_glb.stem != asset_id:
            raise AssetLibraryBuildError(f"GLB stem differs from asset id: {asset_id}")
        if usd.parent != batch_root / "usd" or usd.stem != asset_id:
            raise AssetLibraryBuildError(f"USD stem differs from asset id: {asset_id}")
        _sha256(record.get("source_glb_sha256"), "source_glb_sha256")
        _sha256(record.get("usd_sha256"), "usd_sha256")
        structural = record.get("structural_validation")
        if not isinstance(structural, dict):
            raise AssetLibraryBuildError(f"Missing structural validation: {asset_id}")
        texture = _contained_file(
            batch_root,
            structural.get("texture"),
            f"texture {asset_id}",
        )
        if texture.parent != batch_root / "usd" / "textures":
            raise AssetLibraryBuildError(f"Texture path differs for {asset_id}")
        if texture.stem != asset_id:
            raise AssetLibraryBuildError(
                f"Texture stem differs from asset id: {asset_id}"
            )
        _sha256(structural.get("texture_sha256"), "texture_sha256")
        indexed[asset_id] = record
    declared_count = manifest.get("asset_count")
    passed_count = manifest.get("passed_count")
    if declared_count != len(indexed) or passed_count != len(indexed):
        raise AssetLibraryBuildError("USD conversion manifest counts differ")
    return indexed


def _bounds(record: Mapping[str, Any], asset_id: str) -> dict[str, Any]:
    values = record.get("bounds")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(not isinstance(point, list) or len(point) != 3 for point in values)
    ):
        raise AssetLibraryBuildError(f"Invalid GLB bounds for {asset_id}")
    minimum = [_finite(value, "bounds minimum") for value in values[0]]
    maximum = [_finite(value, "bounds maximum") for value in values[1]]
    if any(high <= low for low, high in zip(minimum, maximum, strict=True)):
        raise AssetLibraryBuildError(f"Empty GLB bounds for {asset_id}")
    diagonal = _finite(record.get("bounds_diagonal"), "bounds diagonal", positive=True)
    return {
        "status": "reported",
        "coordinate_space": "source_glb_unscaled",
        "minimum": minimum,
        "maximum": maximum,
        "diagonal": diagonal,
    }


def _index_glb_validation(
    manifest: Mapping[str, Any],
    batch_root: Path,
    invalid_reported_ids: set[str],
) -> dict[str, tuple[Mapping[str, Any], str | None]]:
    if manifest.get("passed") is not True or manifest.get("failed_count") != 0:
        raise AssetLibraryBuildError("GLB validation manifest is not fully passed")
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise AssetLibraryBuildError("GLB validation assets must be an array")
    indexed: dict[str, tuple[Mapping[str, Any], str | None]] = {}
    for index, record in enumerate(assets):
        if not isinstance(record, dict) or record.get("passed") is not True:
            raise AssetLibraryBuildError(f"GLB validation record {index} did not pass")
        glb = _contained_file(
            batch_root,
            record.get("path"),
            f"validated GLB {index}",
        )
        if glb.parent != batch_root / "glb":
            raise AssetLibraryBuildError(f"Validated GLB escapes the GLB folder: {glb}")
        asset_id = glb.stem
        if ASSET_ID_PATTERN.fullmatch(asset_id) is None or asset_id in indexed:
            raise AssetLibraryBuildError(f"Invalid or duplicate GLB stem: {asset_id}")
        reported = record.get("asset_id")
        if reported in invalid_reported_ids:
            rejected_reported_id: str | None = str(reported)
        elif reported == asset_id:
            rejected_reported_id = None
        else:
            raise AssetLibraryBuildError(
                f"GLB report id is neither valid nor explicitly rejected: {reported!r}"
            )
        _positive_integer(record.get("bytes"), "validated GLB bytes")
        _sha256(record.get("sha256"), "validated GLB SHA-256")
        _bounds(record, asset_id)
        indexed[asset_id] = (record, rejected_reported_id)
    declared = manifest.get("asset_count")
    expected = manifest.get("expected_count")
    passed = manifest.get("passed_count")
    if declared != len(indexed) or expected != len(indexed) or passed != len(indexed):
        raise AssetLibraryBuildError("GLB validation manifest counts differ")
    return indexed


_USD_INSPECTOR = r"""
import json
import sys
from pxr import Usd, UsdGeom

result = {}
for raw in json.load(sys.stdin):
    stage = Usd.Stage.Open(raw)
    if stage is None:
        raise RuntimeError(f"Cannot open USD stage: {raw}")
    default_prim = stage.GetDefaultPrim()
    result[raw] = {
        "status": "inspected",
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
    }
json.dump(result, sys.stdout, allow_nan=False, sort_keys=True)
"""


def _discover_usd_python(batch_root: Path) -> Path | None:
    candidates = (
        batch_root
        / ".physical-ai"
        / "venvs"
        / "simready-validate"
        / "Scripts"
        / "python.exe",
        batch_root / ".physical-ai" / "venvs" / "simready-validate" / "bin" / "python",
    )
    return next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )


def inspect_usd_stages(
    usd_paths: Sequence[Path],
    *,
    batch_root: Path,
    usd_python: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Inspect current USD stage metadata using an existing D:-hosted PxR runtime."""

    interpreter = (
        _require_d_path(usd_python, "USD Python interpreter", kind="file")
        if usd_python is not None
        else _discover_usd_python(batch_root)
    )
    if interpreter is None:
        return {}
    paths = [
        str(_contained_file(batch_root, path, "USD inspection input"))
        for path in usd_paths
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TEMP": str(batch_root / ".physical-ai"),
            "TMP": str(batch_root / ".physical-ai"),
        }
    )
    completed = subprocess.run(
        [str(interpreter), "-B", "-c", _USD_INSPECTOR],
        input=json.dumps(paths, ensure_ascii=False),
        text=True,
        capture_output=True,
        cwd=batch_root,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise AssetLibraryBuildError(
            "USD metadata inspection failed: " + completed.stderr.strip()[-2000:]
        )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AssetLibraryBuildError("USD inspector returned invalid JSON") from error
    if not isinstance(raw, dict) or set(raw) != set(paths):
        raise AssetLibraryBuildError("USD inspector returned an incomplete result")
    return {Path(path).stem: value for path, value in raw.items()}


def _usd_stage_record(value: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    if value is None:
        return {
            "status": "pending",
            "up_axis": None,
            "meters_per_unit": None,
            "default_prim": None,
        }
    if not isinstance(value, dict):
        raise AssetLibraryBuildError("USD metadata must be an object")
    up_axis = value.get("up_axis")
    if up_axis != contract["usd_stage"]["required_up_axis"]:
        raise AssetLibraryBuildError(f"USD stage is not Y-up: {up_axis!r}")
    meters_per_unit = _finite(
        value.get("meters_per_unit"),
        "meters_per_unit",
        positive=True,
    )
    default_prim = value.get("default_prim")
    if contract["usd_stage"]["require_default_prim_when_inspected"] and (
        not isinstance(default_prim, str)
        or not default_prim.startswith("/")
        or default_prim == "/"
    ):
        raise AssetLibraryBuildError("Inspected USD stage has no defaultPrim")
    return {
        "status": "inspected",
        "up_axis": up_axis,
        "meters_per_unit": meters_per_unit,
        "default_prim": default_prim,
    }


def _validate_artifact_record(record: Any, label: str) -> None:
    if not isinstance(record, dict) or set(record) != {
        "root",
        "path",
        "byte_count",
        "sha256",
    }:
        raise AssetLibraryBuildError(f"{label} artifact fields differ")
    if record["root"] not in {
        "generated_assets",
        "reference_assets",
        "review_batch",
    }:
        raise AssetLibraryBuildError(f"{label} artifact root differs")
    _portable_path(record["path"], f"{label}.path")
    _positive_integer(record["byte_count"], f"{label}.byte_count")
    _sha256(record["sha256"], f"{label}.sha256")


def _catalog_revision_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "catalog_revision"}


def validate_asset_library(
    payload: Mapping[str, Any],
    *,
    contract_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate identities, category quantities and immutable catalogue revision."""

    contract_file = _require_d_path(
        contract_path or Path(__file__).with_name("asset_library_contract.v1.json"),
        "asset library contract",
        kind="file",
    )
    contract = _load_contract(contract_file)
    expected_keys = {
        "schema",
        "status",
        "build_algorithm",
        "contract_sha256",
        "catalog_revision",
        "asset_count",
        "category_counts",
        "normalization_policy",
        "selection_pools",
        "input_evidence",
        "assets",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise AssetLibraryBuildError("Asset library fields differ")
    if payload["schema"] != CATALOG_SCHEMA or payload["status"] != CATALOG_STATUS:
        raise AssetLibraryBuildError("Asset library schema or status differs")
    if payload["build_algorithm"] != BUILD_ALGORITHM:
        raise AssetLibraryBuildError("Asset library build algorithm differs")
    if payload["contract_sha256"] != sha256_file(contract_file):
        raise AssetLibraryBuildError("Asset library contract hash differs")
    if payload["normalization_policy"] != contract["normalization_policy"]:
        raise AssetLibraryBuildError("Asset library normalization policy differs")
    if payload["catalog_revision"] != canonical_sha256(
        _catalog_revision_payload(payload)
    ):
        raise AssetLibraryBuildError("Asset library revision hash differs")
    evidence = payload["input_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {
        "reference_manifest",
        "usd_conversion_manifest",
        "glb_validation_manifest",
    }:
        raise AssetLibraryBuildError("Asset library input evidence differs")
    for label, artifact in evidence.items():
        _validate_artifact_record(artifact, label)
    assets = payload["assets"]
    if not isinstance(assets, list) or len(assets) != contract["expected_asset_count"]:
        raise AssetLibraryBuildError("Asset library must contain exactly 53 assets")
    ids: list[str] = []
    categories: Counter[str] = Counter()
    asset_keys = {
        "asset_id",
        "category",
        "identity",
        "source",
        "usd",
        "texture",
        "receipt",
        "source_bounds",
        "usd_stage",
        "qualification",
        "eligibility",
    }
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict) or set(asset) != asset_keys:
            raise AssetLibraryBuildError(f"Asset {index} fields differ")
        asset_id = asset["asset_id"]
        if (
            not isinstance(asset_id, str)
            or ASSET_ID_PATTERN.fullmatch(asset_id) is None
        ):
            raise AssetLibraryBuildError(f"Asset {index} id differs")
        if asset_id == "glb":
            raise AssetLibraryBuildError(
                "The erroneous GLB report id cannot be an asset id"
            )
        ids.append(asset_id)
        category = asset["category"]
        if category not in CATEGORIES:
            raise AssetLibraryBuildError(f"Asset {asset_id} category differs")
        categories[category] += 1
        identity = asset["identity"]
        if not isinstance(identity, dict) or set(identity) != {
            "strategy",
            "rejected_reported_asset_id",
        }:
            raise AssetLibraryBuildError(f"Asset {asset_id} identity fields differ")
        if identity["strategy"] != IDENTITY_STRATEGY:
            raise AssetLibraryBuildError(f"Asset {asset_id} identity strategy differs")
        rejected_id = identity["rejected_reported_asset_id"]
        if (
            rejected_id is not None
            and rejected_id not in contract["identity"]["invalid_reported_asset_ids"]
        ):
            raise AssetLibraryBuildError(f"Asset {asset_id} rejected id differs")
        for label in ("source", "usd", "texture", "receipt"):
            _validate_artifact_record(asset[label], f"{asset_id}.{label}")
        if asset["source"]["root"] != "reference_assets":
            raise AssetLibraryBuildError(f"Asset {asset_id} source root differs")
        if any(
            asset[label]["root"] != "review_batch"
            for label in ("usd", "texture", "receipt")
        ):
            raise AssetLibraryBuildError(f"Asset {asset_id} batch root differs")
        bounds = asset["source_bounds"]
        if not isinstance(bounds, dict) or set(bounds) != {
            "status",
            "coordinate_space",
            "minimum",
            "maximum",
            "diagonal",
        }:
            raise AssetLibraryBuildError(f"Asset {asset_id} bounds fields differ")
        if (
            bounds["status"] != "reported"
            or bounds["coordinate_space"] != "source_glb_unscaled"
        ):
            raise AssetLibraryBuildError(f"Asset {asset_id} bounds status differs")
        _bounds(
            {
                "bounds": [bounds["minimum"], bounds["maximum"]],
                "bounds_diagonal": bounds["diagonal"],
            },
            asset_id,
        )
        _usd_stage_record(
            asset["usd_stage"] if asset["usd_stage"]["status"] == "inspected" else None,
            contract,
        )
        if asset["qualification"] != contract["qualification_defaults"]:
            raise AssetLibraryBuildError(f"Asset {asset_id} qualification was invented")
        if asset["eligibility"] != contract["eligibility_defaults"]:
            raise AssetLibraryBuildError(f"Asset {asset_id} eligibility differs")
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise AssetLibraryBuildError("Asset ids must be unique and sorted")
    expected_counts = contract["expected_category_counts"]
    actual_counts = {category: categories[category] for category in CATEGORIES}
    if (
        payload["category_counts"] != expected_counts
        or actual_counts != expected_counts
    ):
        raise AssetLibraryBuildError("Asset category quantities differ")
    if payload["asset_count"] != len(assets):
        raise AssetLibraryBuildError("Asset count differs")
    pools = payload["selection_pools"]
    if not isinstance(pools, dict) or set(pools) != {"building", "tree"}:
        raise AssetLibraryBuildError("Asset selection pools differ")
    by_category = {
        category: [
            asset["asset_id"] for asset in assets if asset["category"] == category
        ]
        for category in pools
    }
    if pools != by_category or any(
        len(pools[category]) != contract["expected_selection_pool_counts"][category]
        for category in pools
    ):
        raise AssetLibraryBuildError("Asset selection pool membership differs")
    return {
        "schema": CATALOG_SCHEMA,
        "status": CATALOG_STATUS,
        "asset_count": len(assets),
        "category_counts": actual_counts,
        "inspected_usd_count": sum(
            asset["usd_stage"]["status"] == "inspected" for asset in assets
        ),
    }


def build_asset_library(
    reference_manifest_path: Path | str,
    batch_root: Path | str,
    *,
    contract_path: Path | str | None = None,
    usd_python: Path | str | None = None,
    usd_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the catalogue in memory from the reviewed 53-asset evidence."""

    reference_manifest_file = _require_d_path(
        reference_manifest_path,
        "reference manifest",
        kind="file",
    )
    batch = _require_d_path(batch_root, "review batch", kind="directory")
    contract_file = _require_d_path(
        contract_path or Path(__file__).with_name("asset_library_contract.v1.json"),
        "asset library contract",
        kind="file",
    )
    contract = _load_contract(contract_file)
    reference_manifest = _read_json(reference_manifest_file, "reference manifest")
    reference_root = _require_d_path(
        reference_manifest.get("reference_root"),
        "reference root",
        kind="directory",
    )
    reference_assets = _index_reference_assets(reference_manifest, reference_root)

    conversion_file = batch / "reports" / "usd" / "usd-conversion-manifest.json"
    glb_validation_file = batch / "reports" / "final-glb-validation-local.json"
    conversion_manifest = _read_json(conversion_file, "USD conversion manifest")
    glb_validation = _read_json(glb_validation_file, "GLB validation manifest")
    conversion = _index_conversion_results(conversion_manifest, batch)
    glb = _index_glb_validation(
        glb_validation,
        batch,
        set(contract["identity"]["invalid_reported_asset_ids"]),
    )
    selected_ids = set(conversion)
    if selected_ids != set(glb):
        raise AssetLibraryBuildError("USD and GLB reviewed asset identities differ")
    if not selected_ids.issubset(reference_assets):
        missing = sorted(selected_ids - set(reference_assets))
        raise AssetLibraryBuildError(
            f"Reviewed assets are absent from reference: {missing}"
        )
    if len(selected_ids) != contract["expected_asset_count"]:
        raise AssetLibraryBuildError("Reviewed batch must contain exactly 53 assets")

    usd_paths = [Path(conversion[asset_id]["usd"]) for asset_id in sorted(selected_ids)]
    if usd_metadata is None:
        metadata = inspect_usd_stages(
            usd_paths,
            batch_root=batch,
            usd_python=usd_python,
        )
    else:
        metadata = {asset_id: dict(value) for asset_id, value in usd_metadata.items()}
        unknown = set(metadata) - selected_ids
        if unknown:
            raise AssetLibraryBuildError(
                f"USD metadata contains unknown assets: {sorted(unknown)}"
            )

    assets: list[dict[str, Any]] = []
    for asset_id in sorted(selected_ids):
        reference = reference_assets[asset_id]
        conversion_record = conversion[asset_id]
        glb_record, rejected_id = glb[asset_id]
        source_relative = _portable_path(
            reference["source_relative"], "source_relative"
        )
        source_path = reference_root.joinpath(*PurePosixPath(source_relative).parts)
        source_artifact = _artifact(
            logical_root="reference_assets",
            physical_root=reference_root,
            path=source_path,
            label=f"source {asset_id}",
            expected_sha256=_sha256(reference["source_sha256"], "source_sha256"),
            expected_bytes=_positive_integer(reference["source_bytes"], "source_bytes"),
        )
        source_glb = Path(conversion_record["source_glb"])
        if (
            sha256_file(source_glb)
            != _sha256(conversion_record["source_glb_sha256"], "source_glb_sha256")
            or sha256_file(source_glb) != _sha256(glb_record["sha256"], "GLB SHA-256")
            or source_glb.stat().st_size != glb_record["bytes"]
        ):
            raise AssetLibraryBuildError(f"GLB evidence differs for {asset_id}")
        usd_artifact = _artifact(
            logical_root="review_batch",
            physical_root=batch,
            path=Path(conversion_record["usd"]),
            label=f"USD {asset_id}",
            expected_sha256=_sha256(conversion_record["usd_sha256"], "usd_sha256"),
        )
        structural = conversion_record["structural_validation"]
        texture_artifact = _artifact(
            logical_root="review_batch",
            physical_root=batch,
            path=Path(structural["texture"]),
            label=f"texture {asset_id}",
            expected_sha256=_sha256(structural["texture_sha256"], "texture_sha256"),
        )
        receipt_path = batch / "reports" / "usd" / f"{asset_id}-usd.json"
        receipt_payload = _read_json(receipt_path, f"USD receipt {asset_id}")
        if receipt_payload != conversion_record:
            raise AssetLibraryBuildError(
                f"USD receipt differs from manifest for {asset_id}"
            )
        receipt_artifact = _artifact(
            logical_root="review_batch",
            physical_root=batch,
            path=receipt_path,
            label=f"USD receipt {asset_id}",
        )
        assets.append(
            {
                "asset_id": asset_id,
                "category": _classify(source_relative, contract),
                "identity": {
                    "strategy": IDENTITY_STRATEGY,
                    "rejected_reported_asset_id": rejected_id,
                },
                "source": source_artifact,
                "usd": usd_artifact,
                "texture": texture_artifact,
                "receipt": receipt_artifact,
                "source_bounds": _bounds(glb_record, asset_id),
                "usd_stage": _usd_stage_record(metadata.get(asset_id), contract),
                "qualification": copy.deepcopy(contract["qualification_defaults"]),
                "eligibility": dict(contract["eligibility_defaults"]),
            }
        )

    counts = Counter(asset["category"] for asset in assets)
    category_counts = {category: counts[category] for category in CATEGORIES}
    if category_counts != contract["expected_category_counts"]:
        raise AssetLibraryBuildError(
            f"Expected category quantities differ: {category_counts}"
        )
    payload: dict[str, Any] = {
        "schema": CATALOG_SCHEMA,
        "status": CATALOG_STATUS,
        "build_algorithm": BUILD_ALGORITHM,
        "contract_sha256": sha256_file(contract_file),
        "asset_count": len(assets),
        "category_counts": category_counts,
        "normalization_policy": dict(contract["normalization_policy"]),
        "selection_pools": {
            category: [
                asset["asset_id"] for asset in assets if asset["category"] == category
            ]
            for category in ("building", "tree")
        },
        "input_evidence": {
            "reference_manifest": _artifact(
                logical_root="generated_assets",
                physical_root=reference_manifest_file.parent,
                path=reference_manifest_file,
                label="reference manifest",
            ),
            "usd_conversion_manifest": _artifact(
                logical_root="review_batch",
                physical_root=batch,
                path=conversion_file,
                label="USD conversion manifest",
            ),
            "glb_validation_manifest": _artifact(
                logical_root="review_batch",
                physical_root=batch,
                path=glb_validation_file,
                label="GLB validation manifest",
            ),
        },
        "assets": assets,
    }
    payload["catalog_revision"] = canonical_sha256(payload)
    validate_asset_library(payload, contract_path=contract_file)
    return payload


def write_asset_library(
    payload: Mapping[str, Any],
    output_path: Path | str,
    *,
    contract_path: Path | str | None = None,
) -> Path:
    """Atomically publish a new catalogue on D:, refusing every overwrite."""

    validate_asset_library(payload, contract_path=contract_path)
    output = _require_d_path(output_path, "asset library output")
    if output.exists():
        raise AssetLibraryBuildError(f"Refusing to overwrite asset library: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        raise AssetLibraryBuildError(f"Refusing existing staging file: {temporary}")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise
    return output


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable catalogue for the reviewed 53-USD batch"
    )
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("asset_library_contract.v1.json"),
    )
    parser.add_argument("--usd-python", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)
    payload = build_asset_library(
        options.reference_manifest,
        options.batch_root,
        contract_path=options.contract,
        usd_python=options.usd_python,
    )
    output = write_asset_library(
        payload,
        options.output,
        contract_path=options.contract,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "catalog_revision": payload["catalog_revision"],
                "asset_count": payload["asset_count"],
                "category_counts": payload["category_counts"],
                "inspected_usd_count": sum(
                    asset["usd_stage"]["status"] == "inspected"
                    for asset in payload["assets"]
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSET_ID_PATTERN",
    "BUILD_ALGORITHM",
    "CATALOG_SCHEMA",
    "CATALOG_STATUS",
    "AssetLibraryBuildError",
    "build_asset_library",
    "canonical_sha256",
    "inspect_usd_stages",
    "main",
    "select_asset_for_candidate",
    "selection_seed",
    "sha256_file",
    "validate_asset_library",
    "write_asset_library",
]
