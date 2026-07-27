"""Build the exact folder inventory accepted by the FireViewer admin uploader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,95}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".fwtile", ".fwterrain"}
SOURCE_PREFIXES = ("far/", "detail/", "imagery/")
UNITY_VALIDATION_SCHEMA = "fireviewer.unity-manual-validation.v1"
UNITY_VERSION = "6000.3.18f1"
UNITY_APPROVAL_STATEMENT = "ACCEPTÉ POUR PUBLICATION"
UNITY_CHECKLIST_IDS = (
    "catalog_loaded",
    "terrain_grounding",
    "vegetation_exclusions",
    "lod_streaming",
    "detail_buildings",
    "no_blocking_visual_artifacts",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def normalize_unity_validation_receipt(
    *,
    receipt_path: Path,
    preview_path: Path,
    source_catalog_path: Path,
    package_id: str,
    zone_id: str,
    revision: int,
) -> dict[str, Any]:
    if not preview_path.is_file() or preview_path.stat().st_size <= 0:
        raise ValueError("Unity validation preview PNG is absent or empty")
    if preview_path.suffix.casefold() != ".png":
        raise ValueError("Unity validation preview must be a PNG file")
    with preview_path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Unity validation preview does not have a PNG signature")
    receipt = read_json(receipt_path)
    if receipt.get("schema") != UNITY_VALIDATION_SCHEMA:
        raise ValueError("Unity validation receipt schema is unsupported")
    if receipt.get("decision") != "accepted":
        raise ValueError("Unity validation receipt decision is not accepted")
    if receipt.get("approval_statement") != UNITY_APPROVAL_STATEMENT:
        raise ValueError("Unity publication approval statement is absent")
    for field, expected in (
        ("package_id", package_id),
        ("zone_id", zone_id),
        ("revision", revision),
        ("unity_version", UNITY_VERSION),
        ("catalog_sha256", sha256_file(source_catalog_path)),
        ("preview_sha256", sha256_file(preview_path)),
    ):
        if receipt.get(field) != expected:
            raise ValueError(
                f"Unity validation receipt {field} does not match production"
            )
    reviewer = receipt.get("reviewer")
    if not isinstance(reviewer, str) or len(reviewer.strip()) < 2:
        raise ValueError("Unity validation receipt reviewer is absent")
    reviewed_at = receipt.get("reviewed_at_utc")
    if not isinstance(reviewed_at, str) or not reviewed_at.endswith("Z"):
        raise ValueError("Unity validation receipt reviewed_at_utc must be UTC")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Unity validation receipt reviewed_at_utc is invalid") from exc
    if parsed_reviewed_at.utcoffset() is None:
        raise ValueError("Unity validation receipt reviewed_at_utc lacks a timezone")
    checklist = receipt.get("checklist")
    if not isinstance(checklist, dict) or set(checklist) != set(UNITY_CHECKLIST_IDS):
        raise ValueError("Unity validation receipt checklist is incomplete")
    failed = sorted(key for key, value in checklist.items() if value is not True)
    if failed:
        raise ValueError(
            "Unity validation receipt contains failed checks: " + ", ".join(failed)
        )
    return {
        "schema": UNITY_VALIDATION_SCHEMA,
        "decision": "accepted",
        "approval_statement": UNITY_APPROVAL_STATEMENT,
        "package_id": package_id,
        "zone_id": zone_id,
        "revision": revision,
        "reviewer": reviewer.strip(),
        "reviewed_at_utc": reviewed_at,
        "unity_version": UNITY_VERSION,
        "catalog_sha256": receipt["catalog_sha256"],
        "preview_sha256": receipt["preview_sha256"],
        "checklist": {key: True for key in UNITY_CHECKLIST_IDS},
    }


def safe_relative_path(value: str, *, prefixes: tuple[str, ...]) -> PurePosixPath:
    if not value or "\\" in value or not value.startswith(prefixes):
        raise ValueError(f"unsupported relative asset path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative asset path: {value!r}")
    if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported Unity runtime asset type: {value!r}")
    return path


def runtime_asset_records(catalog: dict[str, Any]) -> Iterator[dict[str, Any]]:
    lod_policy = catalog.get("lod_policy")
    if not isinstance(lod_policy, dict):
        raise ValueError("remote catalog lacks lod_policy")
    far = lod_policy.get("far")
    if not isinstance(far, dict):
        raise ValueError("remote catalog lacks lod_policy.far")
    for name in ("imagery", "terrain"):
        record = far.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"remote catalog lacks FAR {name}")
        yield record

    tiles = catalog.get("tiles")
    detail = lod_policy.get("detail", {"enabled": True})
    if not isinstance(detail, dict):
        raise ValueError("remote catalog lod_policy.detail is invalid")
    detail_enabled = detail.get("enabled", True)
    if not isinstance(detail_enabled, bool):
        raise ValueError("remote catalog does not declare detail streaming")
    if not isinstance(tiles, list):
        raise ValueError("remote catalog tiles must be an array")
    if not detail_enabled:
        if tiles or int(catalog.get("exported_detail_tile_count", -1)) != 0:
            raise ValueError("global-only catalog must not contain detail tiles")
        return
    if not tiles:
        raise ValueError("remote catalog contains no detail tile")
    for index, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            raise ValueError(f"detail tile {index} is invalid")
        for name in ("imagery", "payload"):
            record = tile.get(name)
            if not isinstance(record, dict):
                raise ValueError(f"detail tile {index} lacks {name}")
            yield record


def catalog_assets(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []

    def visit(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        path = node.get("path")
        if isinstance(path, str) and path.startswith("assets/"):
            assets.append(node)
        for child in node.values():
            visit(child)

    visit(catalog)
    return assets


def validate_site_package(root: Path) -> dict[str, Any]:
    manifest_path = root / "package-manifest.json"
    catalog_path = root / "catalog.json"
    manifest = read_json(manifest_path)
    catalog = read_json(catalog_path)
    package_id = manifest.get("package_id")
    if not isinstance(package_id, str) or not PACKAGE_ID_RE.fullmatch(package_id):
        raise ValueError("package-manifest.json contains an invalid package_id")

    catalog_reference = manifest.get("catalog")
    if (
        not isinstance(catalog_reference, dict)
        or catalog_reference.get("path") != "catalog.json"
    ):
        raise ValueError("package-manifest.json does not reference catalog.json")
    if catalog_reference.get("byte_count") != catalog_path.stat().st_size:
        raise ValueError("catalog byte_count differs from package-manifest.json")
    catalog_sha256 = sha256_file(catalog_path)
    if catalog_reference.get("sha256") != catalog_sha256:
        raise ValueError("catalog sha256 differs from package-manifest.json")

    assets = catalog_assets(catalog)
    if not assets:
        raise ValueError("catalog.json declares no uploadable asset")
    declared: set[str] = set()
    asset_bytes = 0
    for record in assets:
        relative = safe_relative_path(
            str(record.get("path", "")), prefixes=("assets/",)
        )
        path_text = relative.as_posix()
        if path_text in declared:
            raise ValueError(f"catalog.json declares {path_text} more than once")
        declared.add(path_text)
        expected_size = record.get("byte_count")
        expected_sha256 = record.get("sha256")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError(f"catalog asset {path_text} has an invalid byte_count")
        if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(
            expected_sha256
        ):
            raise ValueError(f"catalog asset {path_text} has an invalid sha256")
        asset_path = root.joinpath(*relative.parts)
        if not asset_path.is_file():
            raise FileNotFoundError(f"catalog asset is absent: {path_text}")
        if asset_path.stat().st_size != expected_size:
            raise ValueError(f"catalog asset size differs: {path_text}")
        if sha256_file(asset_path) != expected_sha256:
            raise ValueError(f"catalog asset sha256 differs: {path_text}")
        asset_bytes += expected_size

    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    expected = {"package-manifest.json", "catalog.json", *declared}
    extras = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extras or missing:
        raise ValueError(
            f"package inventory differs: extras={extras[:1]}, missing={missing[:1]}"
        )
    return {
        "status": "valid",
        "package_id": package_id,
        "root": str(root.resolve()),
        "asset_count": len(assets),
        "file_count": len(actual),
        "asset_bytes": asset_bytes,
        "total_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
        "catalog_sha256": catalog_sha256,
        "manifest_sha256": sha256_file(manifest_path),
    }


def build_site_package(
    *,
    source_root: Path,
    output_root: Path,
    package_id: str,
    zone_id: str,
    revision: int,
    unity_validation_receipt: Path,
    unity_preview_png: Path,
) -> dict[str, Any]:
    if not PACKAGE_ID_RE.fullmatch(package_id):
        raise ValueError("package_id does not match the site upload contract")
    if revision <= 0:
        raise ValueError("revision must be positive")
    source_catalog_path = source_root / "catalog.json"
    source_catalog = read_json(source_catalog_path)
    if source_catalog.get("schema") != "fireviewer.remote-tile-catalog.v1":
        raise ValueError("source is not a FireViewer Unity remote tile catalog")
    manual_validation = normalize_unity_validation_receipt(
        receipt_path=unity_validation_receipt,
        preview_path=unity_preview_png,
        source_catalog_path=source_catalog_path,
        package_id=package_id,
        zone_id=zone_id,
        revision=revision,
    )
    if output_root.exists():
        report = validate_site_package(output_root)
        manifest = read_json(output_root / "package-manifest.json")
        if manifest.get("manual_unity_validation") != manual_validation:
            raise ValueError(
                "existing site package has a different Unity validation receipt"
            )
        return report

    catalog = deepcopy(source_catalog)
    records = list(runtime_asset_records(catalog))
    if len(records) != 2 + (2 * int(catalog.get("exported_detail_tile_count", -1))):
        raise ValueError("remote catalog asset count is inconsistent")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    linked = 0
    try:
        seen: set[str] = set()
        for record in records:
            source_relative = safe_relative_path(
                str(record.get("url", "")), prefixes=SOURCE_PREFIXES
            )
            source_text = source_relative.as_posix()
            if source_text in seen:
                raise ValueError(f"remote catalog URL is duplicated: {source_text}")
            seen.add(source_text)
            expected_size = record.get("byte_count")
            expected_sha256 = record.get("sha256")
            if not isinstance(expected_size, int) or expected_size <= 0:
                raise ValueError(
                    f"remote asset {source_text} has an invalid byte_count"
                )
            if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(
                expected_sha256
            ):
                raise ValueError(f"remote asset {source_text} has an invalid sha256")
            source_path = source_root.joinpath(*source_relative.parts)
            if not source_path.is_file() or source_path.stat().st_size != expected_size:
                raise ValueError(
                    f"remote asset is absent or has a different size: {source_text}"
                )
            target_relative = PurePosixPath("assets") / source_relative
            target_path = temporary.joinpath(*target_relative.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_path, target_path)
            linked += 1
            record["path"] = target_relative.as_posix()
            record["url"] = target_relative.as_posix()

        preview_relative = PurePosixPath("assets/validation/unity-preview.png")
        preview_target = temporary.joinpath(*preview_relative.parts)
        preview_target.parent.mkdir(parents=True, exist_ok=True)
        os.link(unity_preview_png, preview_target)
        linked += 1
        preview_record = {
            "path": preview_relative.as_posix(),
            "url": preview_relative.as_posix(),
            "sha256": manual_validation["preview_sha256"],
            "byte_count": unity_preview_png.stat().st_size,
            "media_type": "image/png",
            "role": "manual-unity-validation-preview",
        }

        catalog["package_id"] = package_id
        catalog["zones"] = [{"zone_id": zone_id, "revision_id": f"R{revision}"}]
        catalog["validation"] = {
            "schema": UNITY_VALIDATION_SCHEMA,
            "unity_preview": preview_record,
        }
        catalog["upload_profile"] = {
            "delivery": "private object storage",
            "inventory": "catalog-exact",
            "runtime": "Unity",
            "schema": "fireviewer.site-upload.remote-tiles.v1",
        }
        catalog_raw = json_bytes(catalog)
        (temporary / "catalog.json").write_bytes(catalog_raw)
        manifest = {
            "schema_version": "1.2",
            "package_id": package_id,
            "catalog": {
                "path": "catalog.json",
                "sha256": hashlib.sha256(catalog_raw).hexdigest(),
                "byte_count": len(catalog_raw),
            },
            "zones": [{"zone_id": zone_id, "revision_id": f"R{revision}"}],
            "runtime": {
                "renderer": "Unity",
                "catalog_schema": "fireviewer.remote-tile-catalog.v1",
                "delivery": "remote FAR plus camera-driven detail tiles",
            },
            "spatial_profile": {
                "grid_crs": catalog.get("crs"),
                "linear_unit": catalog.get("linear_unit"),
                "origin_l93_m": catalog.get("origin_l93_m"),
            },
            "inventory": {
                "asset_count": len(records) + 1,
                "asset_bytes": sum(int(record["byte_count"]) for record in records)
                + int(preview_record["byte_count"]),
            },
            "manual_unity_validation": manual_validation,
            "provenance": {
                "source_catalog_sha256": sha256_file(source_catalog_path),
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
        }
        (temporary / "package-manifest.json").write_bytes(json_bytes(manifest))
        report = validate_site_package(temporary)
        temporary.replace(output_root)
        report["root"] = str(output_root.resolve())
        report["link_mode"] = "hardlink"
        report["hardlinked_asset_count"] = linked
        report_path = output_root.parent / f"{output_root.name}-validation.json"
        report_path.write_bytes(json_bytes(report))
        report["validation_report"] = str(report_path.resolve())
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--revision", type=int, required=True)
    parser.add_argument("--unity-validation-receipt", type=Path, required=True)
    parser.add_argument("--unity-preview-png", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_site_package(
        source_root=args.source_root.resolve(),
        output_root=args.output_root.resolve(),
        package_id=args.package_id,
        zone_id=args.zone_id,
        revision=args.revision,
        unity_validation_receipt=args.unity_validation_receipt.resolve(),
        unity_preview_png=args.unity_preview_png.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
