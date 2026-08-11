"""Assemble audited FireViewer captures into the public Hub folder contract.

This command is local-only.  It never creates or uploads a Hugging Face
repository.  Publication is a later gate after both ``sim`` and ``repro``
trees have passed their capture audits and staging audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


CAPTURE_ID_RE = re.compile(
    r"^day_(?P<day>\d{2})_state_(?P<state>\d{3})_view_(?P<point>\d{2})_"
    r"(?P<camera>CAM_\d{2})_zoom_(?P<zoom>\d{2})$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _zoom_folder(zoom_index: int, zoom_multiplier: float) -> str:
    if zoom_index == 2 and abs(zoom_multiplier - 1.0) <= 1e-9:
        return "original"
    multiplier = f"{zoom_multiplier:.2f}".replace(".", "p")
    return f"zoom{zoom_index:02d}_{multiplier}x"


def capture_destination(
    target: dict[str, Any], *, kind: str
) -> tuple[Path, dict[str, Any]]:
    capture_id = str(target.get("capture_id", ""))
    match = CAPTURE_ID_RE.fullmatch(capture_id)
    if match is None:
        raise ValueError(f"invalid capture_id: {capture_id}")
    simulation_time = target.get("simulation_time")
    zoom = target.get("zoom")
    if not isinstance(simulation_time, dict) or not isinstance(zoom, dict):
        raise ValueError(f"capture {capture_id} has no simulation_time or zoom metadata")
    day_index = int(simulation_time["day_index"])
    state_in_day = int(simulation_time["state_in_day"])
    zoom_index = int(zoom["zoom_index"])
    zoom_multiplier = float(zoom["zoom_multiplier"])
    if day_index != int(match.group("day")) or zoom_index != int(match.group("zoom")):
        raise ValueError(f"capture path metadata mismatch: {capture_id}")
    relative = (
        Path("massif-of-justin")
        / kind
        / "raw_files"
        / f"day{day_index:02d}"
        / f"case{state_in_day:02d}"
        / f"point{int(match.group('point')):02d}"
        / _zoom_folder(zoom_index, zoom_multiplier)
    )
    identity = {
        "capture_id": capture_id,
        "dataset_id": target.get("dataset_id"),
        "day_index": day_index,
        "global_state_index": int(match.group("state")),
        "state_in_day": state_in_day,
        "point_index": int(match.group("point")),
        "camera_id": match.group("camera"),
        "zoom_index": zoom_index,
        "zoom_multiplier": zoom_multiplier,
        "is_original_framing": relative.name == "original",
    }
    return relative, identity


def _link_or_copy(source: Path, destination: Path, *, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"
    shutil.copy2(source, destination)
    return "copy"


def _verified_file_hashes(frame: Path, target: dict[str, Any]) -> dict[str, str]:
    trusted = target.get("modality_sha256")
    trusted_hashes = dict(trusted) if isinstance(trusted, dict) else {}
    result: dict[str, str] = {}
    for source in sorted(path for path in frame.iterdir() if path.is_file()):
        digest = trusted_hashes.get(source.name)
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            digest = sha256_file(source)
        result[source.name] = digest
    return result


def assemble(
    *,
    capture_root: Path,
    audit_report: Path,
    destination_root: Path,
    kind: str,
    link_mode: str = "hardlink",
    append: bool = False,
) -> dict[str, Any]:
    if kind not in {"sim", "repro"}:
        raise ValueError("kind must be sim or repro")
    if link_mode not in {"hardlink", "copy"}:
        raise ValueError("link_mode must be hardlink or copy")
    capture_root = capture_root.resolve()
    destination_root = destination_root.resolve()
    audit = read_json(audit_report.resolve())
    if (
        audit.get("schema") != "fireviewer.capture-metadata-audit.v2"
        or audit.get("status") != "passed"
        or int(audit.get("failed_capture_count", -1)) != 0
        or int(audit.get("abstention_warning_count", -1)) != 0
        or int(audit.get("captures", -1)) != int(audit.get("expected_captures", -2))
    ):
        raise RuntimeError("capture audit is not publication-staging clean")
    kind_root = destination_root / "massif-of-justin" / kind
    existing_manifest: dict[str, Any] | None = None
    manifest_path = kind_root / "dataset-manifest.json"
    if kind_root.exists() and any(kind_root.iterdir()):
        if not append:
            raise FileExistsError(f"refusing to merge into non-empty staging tree: {kind_root}")
        if not manifest_path.is_file():
            raise RuntimeError(f"append target has no dataset manifest: {kind_root}")
        existing_manifest = read_json(manifest_path)
        if (
            existing_manifest.get("schema") != "fireviewer.hf-dataset-staging.v1"
            or existing_manifest.get("kind") != kind
        ):
            raise RuntimeError(f"append target has an incompatible dataset manifest: {kind_root}")
    targets = sorted(capture_root.rglob("training-targets.json"))
    if len(targets) != int(audit["captures"]):
        raise RuntimeError(
            f"capture count changed after audit: {len(targets)} != {audit['captures']}"
        )

    capture_ids: set[str] = set()
    destination_paths: set[Path] = set()
    zooms_by_point: dict[tuple[int, int, int], set[int]] = {}
    camera_by_point: dict[tuple[int, int, int], str] = {}
    sample_counts: Counter[str] = Counter()
    transfer_counts: Counter[str] = Counter()
    dataset_ids: set[str] = set()
    source_package_ids: set[str] = set()
    point_receipts: dict[tuple[int, int, int], list[dict[str, Any]]] = {}

    for target_path in targets:
        frame = target_path.parent
        target = read_json(target_path)
        relative, identity = capture_destination(target, kind=kind)
        if identity["capture_id"] in capture_ids or relative in destination_paths:
            raise RuntimeError(f"duplicate capture destination: {identity['capture_id']}")
        capture_ids.add(identity["capture_id"])
        destination_paths.add(relative)
        point_key = (
            int(identity["day_index"]),
            int(identity["state_in_day"]),
            int(identity["point_index"]),
        )
        zooms_by_point.setdefault(point_key, set()).add(int(identity["zoom_index"]))
        previous_camera = camera_by_point.setdefault(point_key, str(identity["camera_id"]))
        if previous_camera != str(identity["camera_id"]):
            raise RuntimeError(f"point camera changed within zoom set: {point_key}")
        sample_counts[str(target.get("sample_kind"))] += 1
        dataset_ids.add(str(target.get("dataset_id")))
        source_package_ids.add(str(target.get("source_package_id")))

        destination = destination_root / relative
        if destination.exists():
            raise FileExistsError(f"capture destination already exists: {destination}")
        file_hashes = _verified_file_hashes(frame, target)
        transferred: dict[str, str] = {}
        for source in sorted(path for path in frame.iterdir() if path.is_file()):
            mode_used = _link_or_copy(
                source, destination / source.name, link_mode=link_mode
            )
            transfer_counts[mode_used] += 1
            transferred[source.name] = mode_used
        receipt = {
            "schema": "fireviewer.hf-capture-package.v1",
            **identity,
            "source_frame_relative": frame.relative_to(capture_root).as_posix(),
            "source_file_sha256": file_hashes,
            "transfer_mode_by_file": transferred,
        }
        write_json(destination / "capture-package.json", receipt)
        point_receipts.setdefault(point_key, []).append(
            {
                "capture_id": identity["capture_id"],
                "folder": relative.name,
                "zoom_index": identity["zoom_index"],
                "zoom_multiplier": identity["zoom_multiplier"],
                "is_original_framing": identity["is_original_framing"],
            }
        )

    incomplete = {
        point: sorted(zooms)
        for point, zooms in zooms_by_point.items()
        if zooms != {1, 2, 3, 4, 5}
    }
    if incomplete:
        raise RuntimeError(f"incomplete point zoom sets: {incomplete}")
    for point_key, captures in point_receipts.items():
        day_index, state_in_day, point_index = point_key
        point_root = (
            kind_root
            / "raw_files"
            / f"day{day_index:02d}"
            / f"case{state_in_day:02d}"
            / f"point{point_index:02d}"
        )
        write_json(
            point_root / "point-manifest.json",
            {
                "schema": "fireviewer.hf-point-package.v1",
                "kind": kind,
                "day_index": day_index,
                "state_in_day": state_in_day,
                "point_index": point_index,
                "camera_id": camera_by_point[point_key],
                "captures": sorted(captures, key=lambda item: int(item["zoom_index"])),
            },
        )

    current_source = {
        "batch_id": capture_root.name,
        "audit_file": audit_report.name,
        "audit_sha256": sha256_file(audit_report.resolve()),
        "captures": len(capture_ids),
        "points": len(zooms_by_point),
    }
    manifest = {
        "schema": "fireviewer.hf-dataset-staging.v1",
        "status": "assembled_from_passed_capture_audit",
        "publication_authorized": False,
        "kind": kind,
        "repository_title": "dataset from simulations",
        "repository_slug": "dataset-from-simulations",
        "folder_contract": f"massif-of-justin/{kind}/raw_files/day**/case**/point**/",
        "dataset_ids": sorted(dataset_ids),
        "source_package_ids": sorted(source_package_ids),
        "captures": len(capture_ids),
        "points": len(zooms_by_point),
        "sample_counts": dict(sample_counts),
        "transfer_counts": dict(transfer_counts),
        "source_capture_root": capture_root.name,
        "source_audit_report": audit_report.name,
        "source_audit_sha256": current_source["audit_sha256"],
        "source_batches": [current_source],
    }
    if existing_manifest is not None:
        previous_sources = existing_manifest.get("source_batches")
        if not isinstance(previous_sources, list):
            previous_sources = [
                {
                    "batch_id": existing_manifest.get("source_capture_root"),
                    "audit_file": existing_manifest.get("source_audit_report"),
                    "audit_sha256": existing_manifest.get("source_audit_sha256"),
                    "captures": int(existing_manifest.get("captures", 0)),
                    "points": int(existing_manifest.get("points", 0)),
                }
            ]
        merged_samples = Counter(existing_manifest.get("sample_counts", {}))
        merged_samples.update(sample_counts)
        merged_transfers = Counter(existing_manifest.get("transfer_counts", {}))
        merged_transfers.update(transfer_counts)
        manifest.update(
            {
                "dataset_ids": sorted(
                    set(existing_manifest.get("dataset_ids", [])) | dataset_ids
                ),
                "source_package_ids": sorted(
                    set(existing_manifest.get("source_package_ids", []))
                    | source_package_ids
                ),
                "captures": int(existing_manifest.get("captures", 0))
                + len(capture_ids),
                "points": int(existing_manifest.get("points", 0))
                + len(zooms_by_point),
                "sample_counts": dict(merged_samples),
                "transfer_counts": dict(merged_transfers),
                "source_capture_root": None,
                "source_audit_report": None,
                "source_audit_sha256": None,
                "source_batches": [*previous_sources, current_source],
            }
        )
    write_json(manifest_path, manifest)
    top_root = destination_root / "massif-of-justin"
    readme = top_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "---\npretty_name: dataset from simulations\n---\n\n"
            "# dataset from simulations\n\n"
            "Audited FireViewer Omniverse simulation and retrospective reproduction captures.\n\n"
            "The `sim` and `repro` trees keep five aligned framings per point. `original` is the "
            "1.00x framing; every capture folder contains RGB, depth, normals, segmentation, "
            "masks, point cloud, optional aerial thermal data, and metadata receipts.\n",
            encoding="utf-8",
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("sim", "repro"), required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    result = assemble(
        capture_root=args.capture_root,
        audit_report=args.audit_report,
        destination_root=args.destination_root,
        kind=args.kind,
        link_mode=args.link_mode,
        append=args.append,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
