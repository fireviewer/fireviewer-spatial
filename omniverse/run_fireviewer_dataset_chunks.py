"""Run audited FireViewer production chunks sequentially and resumably.

The orchestrator never merges into an incomplete chunk and never publishes data.
Each selected state is rendered in its own Isaac Sim process, audited, and only
then admitted to the local campaign receipt before the next state starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from fireviewer_capture_storage import storage_profile_contract


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def storage_capacity_receipt(
    *,
    output_root: Path,
    pilot_report: Path,
    expected_chunk_captures: int,
    minimum_free_gib: float,
    safety_factor: float,
) -> dict[str, Any]:
    if expected_chunk_captures < 1:
        raise ValueError("expected chunk captures must be positive")
    if minimum_free_gib < 0.0:
        raise ValueError("minimum free storage must be non-negative")
    if safety_factor < 1.0:
        raise ValueError("storage safety factor must be at least 1.0")
    report = read_json(pilot_report)
    pilot_captures = int(report.get("captures", 0))
    pilot_root_value = report.get("capture_root")
    if pilot_captures < 1 or not isinstance(pilot_root_value, str):
        raise ValueError("pilot audit has no usable storage measurement")
    pilot_root = Path(pilot_root_value).resolve()
    if not pilot_root.is_dir():
        raise FileNotFoundError(pilot_root)
    pilot_bytes = directory_size_bytes(pilot_root)
    if pilot_bytes < 1:
        raise ValueError("accepted pilot root is empty")
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = int(shutil.disk_usage(output_root).free)
    estimated_chunk_bytes = int(
        math.ceil(
            pilot_bytes
            / float(pilot_captures)
            * int(expected_chunk_captures)
            * float(safety_factor)
        )
    )
    minimum_free_bytes = int(math.ceil(float(minimum_free_gib) * 1024**3))
    required_before_chunk_bytes = minimum_free_bytes + estimated_chunk_bytes
    return {
        "schema": "fireviewer.dataset-storage-capacity.v1",
        "pilot_capture_root": str(pilot_root),
        "pilot_capture_count": pilot_captures,
        "pilot_size_bytes": pilot_bytes,
        "expected_chunk_captures": int(expected_chunk_captures),
        "safety_factor": float(safety_factor),
        "estimated_chunk_bytes": estimated_chunk_bytes,
        "minimum_free_bytes": minimum_free_bytes,
        "required_before_chunk_bytes": required_before_chunk_bytes,
        "free_before_chunk_bytes": free_bytes,
        "admissible": free_bytes >= required_before_chunk_bytes,
    }


def parse_state_spec(value: str, *, maximum: int = 180) -> list[int]:
    indices: list[int] = []
    try:
        for token in (item.strip() for item in value.split(",")):
            if not token:
                raise ValueError("empty state token")
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError("descending state range")
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(token))
    except ValueError as exc:
        raise ValueError(
            "--state-indices must use comma-separated indices or ascending ranges"
        ) from exc
    if not indices:
        raise ValueError("--state-indices cannot be empty")
    if len(indices) != len(set(indices)):
        raise ValueError("--state-indices cannot contain duplicates")
    if any(index < 1 or index > maximum for index in indices):
        raise ValueError(f"--state-indices must stay between 1 and {maximum}")
    return indices


def chunk_id(dataset_id: str, state_index: int) -> str:
    return f"{dataset_id}-state{state_index:03d}"


def build_runner_command(
    *,
    launcher: Path,
    runner: Path,
    stage: Path,
    chunk_root: Path,
    dataset_id: str,
    state_index: int,
    pilot_acceptance_report: Path,
    kit_cache_root: Path,
    resolution: str,
    rt_subframes: int,
    render_product_batch_size: int,
    seconds_per_day: float,
    flow_warmup_updates: int,
    headless: bool,
    dry_run: bool = False,
) -> list[str]:
    command = [
        str(launcher),
        str(runner),
        "--stage",
        str(stage),
        "--output-root",
        str(chunk_root),
        "--dataset-id",
        dataset_id,
        "--production-state-indices",
        str(state_index),
        "--production-chunk-id",
        chunk_id(dataset_id, state_index),
        "--pilot-acceptance-report",
        str(pilot_acceptance_report),
        "--resolution",
        resolution,
        "--rt-subframes",
        str(rt_subframes),
        "--render-product-batch-size",
        str(render_product_batch_size),
        "--seconds-per-day",
        str(seconds_per_day),
        "--flow-warmup-updates",
        str(flow_warmup_updates),
        "--kit-cache-root",
        str(kit_cache_root),
    ]
    if headless:
        command.append("--headless")
    if dry_run:
        command.append("--dry-run")
    return command


def passed_chunk_receipt(
    chunk_root: Path,
    *,
    dataset_id: str,
    state_index: int,
    source_stage_sha256: str,
) -> dict[str, Any] | None:
    audit_path = chunk_root / "audit-report.json"
    contract_path = chunk_root / "run-contract.json"
    if not audit_path.is_file() or not contract_path.is_file():
        return None
    audit = read_json(audit_path)
    contract = read_json(contract_path)
    expected_chunk_id = chunk_id(dataset_id, state_index)
    expected_storage_profile = storage_profile_contract()
    valid = (
        audit.get("schema") == "fireviewer.capture-metadata-audit.v2"
        and audit.get("status") == "passed"
        and int(audit.get("failed_capture_count", -1)) == 0
        and int(audit.get("abstention_warning_count", -1)) == 0
        and int(audit.get("captures", -1)) == int(audit.get("expected_captures", -2))
        and contract.get("schema") == "fireviewer.kit-dataset-production-run.v1"
        and contract.get("run_kind") == "production_chunk"
        and contract.get("dataset_admissible") is True
        and contract.get("full_dataset_capture_authorized") is True
        and contract.get("dataset_id") == dataset_id
        and contract.get("production_chunk_id") == expected_chunk_id
        and contract.get("selected_state_indices") == [state_index]
        and contract.get("source_stage_sha256") == source_stage_sha256
        and contract.get("capture_storage_profile") == expected_storage_profile
        and int(contract.get("expected_capture_cases", -1)) == int(audit.get("captures", -2))
    )
    if not valid:
        return None
    return {
        "state_index": state_index,
        "state_id": f"state_{state_index:03d}",
        "production_chunk_id": expected_chunk_id,
        "captures": int(audit["captures"]),
        "viewpoint_plans": int(audit["viewpoint_plans"]),
        "positive_cases": int(audit["sample_counts"]["positive_fire"]),
        "negative_cases": int(audit["sample_counts"]["negative_context"]),
        "audit_report": str(audit_path),
        "audit_report_sha256": sha256_file(audit_path),
        "run_contract": str(contract_path),
        "run_contract_sha256": sha256_file(contract_path),
        "capture_storage_profile_sha256": expected_storage_profile[
            "profile_sha256"
        ],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--isaac-python", type=Path, required=True)
    result.add_argument("--runner", type=Path, required=True)
    result.add_argument("--auditor", type=Path, required=True)
    result.add_argument("--stage", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--dataset-id", required=True)
    result.add_argument("--state-indices", required=True)
    result.add_argument("--pilot-acceptance-report", type=Path, required=True)
    result.add_argument("--kit-cache-root", type=Path, required=True)
    result.add_argument("--resolution", default="1280x720")
    result.add_argument("--rt-subframes", type=int, default=8)
    result.add_argument("--render-product-batch-size", type=int, default=1)
    result.add_argument("--seconds-per-day", type=float, default=60.0)
    result.add_argument("--flow-warmup-updates", type=int, default=180)
    result.add_argument(
        "--minimum-free-gib",
        type=float,
        default=50.0,
        help="Disk reserve that must remain available before another state starts",
    )
    result.add_argument(
        "--storage-safety-factor",
        type=float,
        default=1.25,
        help="Multiplier applied to accepted-pilot bytes when reserving the next state",
    )
    result.add_argument("--headless", action="store_true")
    result.add_argument(
        "--resume-passed",
        action="store_true",
        help="Skip only chunks with a complete, hashable passed audit receipt",
    )
    result.add_argument("--workdir", type=Path)
    return result


def run(args: argparse.Namespace) -> int:
    launcher = args.isaac_python.resolve()
    runner = args.runner.resolve()
    auditor = args.auditor.resolve()
    stage = args.stage.resolve()
    output_root = args.output_root.resolve()
    pilot_report = args.pilot_acceptance_report.resolve()
    kit_cache_root = args.kit_cache_root.resolve()
    workdir = args.workdir.resolve() if args.workdir else runner.parents[2]
    for required in (launcher, runner, auditor, stage, pilot_report):
        if not required.is_file():
            raise FileNotFoundError(required)
    state_indices = parse_state_spec(args.state_indices)
    source_stage_sha256 = sha256_file(stage)
    capture_storage_profile = storage_profile_contract()
    campaign_progress_path = output_root / "campaign-progress.json"
    receipts: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for sequence, state_index in enumerate(state_indices, start=1):
        chunk_root = output_root / "chunks" / f"state_{state_index:03d}"
        receipt = passed_chunk_receipt(
            chunk_root,
            dataset_id=args.dataset_id,
            state_index=state_index,
            source_stage_sha256=source_stage_sha256,
        )
        if receipt is not None and args.resume_passed:
            receipts.append(receipt)
            write_json_atomic(
                campaign_progress_path,
                {
                    "schema": "fireviewer.dataset-chunk-campaign.v1",
                    "status": "running",
                    "dataset_id": args.dataset_id,
                    "source_stage": str(stage),
                    "source_stage_sha256": source_stage_sha256,
                    "capture_storage_profile": capture_storage_profile,
                    "selected_state_indices": state_indices,
                    "completed_state_indices": [item["state_index"] for item in receipts],
                    "current_state_index": state_index,
                    "current_state_status": "skipped_previously_passed",
                    "receipts": receipts,
                    "publication_authorized": False,
                    "training_admission_authorized": False,
                },
            )
            continue
        if chunk_root.exists() and any(chunk_root.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-admitted chunk output: {chunk_root}"
            )

        capacity = storage_capacity_receipt(
            output_root=output_root,
            pilot_report=pilot_report,
            expected_chunk_captures=100,
            minimum_free_gib=float(args.minimum_free_gib),
            safety_factor=float(args.storage_safety_factor),
        )
        if not capacity["admissible"]:
            write_json_atomic(
                campaign_progress_path,
                {
                    "schema": "fireviewer.dataset-chunk-campaign.v1",
                    "status": "blocked_insufficient_storage",
                    "dataset_id": args.dataset_id,
                    "source_stage": str(stage),
                    "source_stage_sha256": source_stage_sha256,
                    "capture_storage_profile": capture_storage_profile,
                    "selected_state_indices": state_indices,
                    "completed_state_indices": [
                        item["state_index"] for item in receipts
                    ],
                    "current_state_index": state_index,
                    "current_state_status": "not_started_capacity_gate_failed",
                    "storage_capacity": capacity,
                    "receipts": receipts,
                    "publication_authorized": False,
                    "training_admission_authorized": False,
                },
            )
            raise RuntimeError(
                "storage capacity gate refused state "
                f"{state_index}: {capacity['free_before_chunk_bytes']} bytes free, "
                f"{capacity['required_before_chunk_bytes']} required"
            )

        dry_command = build_runner_command(
            launcher=Path(sys.executable),
            runner=runner,
            stage=stage,
            chunk_root=chunk_root,
            dataset_id=args.dataset_id,
            state_index=state_index,
            pilot_acceptance_report=pilot_report,
            kit_cache_root=kit_cache_root,
            resolution=args.resolution,
            rt_subframes=args.rt_subframes,
            render_product_batch_size=args.render_product_batch_size,
            seconds_per_day=args.seconds_per_day,
            flow_warmup_updates=args.flow_warmup_updates,
            headless=args.headless,
            dry_run=True,
        )
        dry_result = subprocess.run(
            dry_command,
            cwd=workdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if dry_result.returncode != 0:
            raise RuntimeError(
                f"production dry-run failed for state {state_index}: {dry_result.stderr}"
            )
        dry_payload = json.loads(dry_result.stdout)
        if (
            dry_payload.get("status") != "dry_run_passed"
            or dry_payload.get("run_kind") != "production_chunk"
            or dry_payload.get("selected_state_indices") != [state_index]
            or int(dry_payload.get("expected_capture_cases", -1)) != 100
            or dry_payload.get("capture_storage_profile")
            != capture_storage_profile
        ):
            raise RuntimeError(f"invalid production dry-run receipt for state {state_index}")

        stdout_path = output_root / "chunks" / f"state_{state_index:03d}.stdout.log"
        stderr_path = output_root / "chunks" / f"state_{state_index:03d}.stderr.log"
        command = build_runner_command(
            launcher=launcher,
            runner=runner,
            stage=stage,
            chunk_root=chunk_root,
            dataset_id=args.dataset_id,
            state_index=state_index,
            pilot_acceptance_report=pilot_report,
            kit_cache_root=kit_cache_root,
            resolution=args.resolution,
            rt_subframes=args.rt_subframes,
            render_product_batch_size=args.render_product_batch_size,
            seconds_per_day=args.seconds_per_day,
            flow_warmup_updates=args.flow_warmup_updates,
            headless=args.headless,
        )
        write_json_atomic(
            campaign_progress_path,
            {
                "schema": "fireviewer.dataset-chunk-campaign.v1",
                "status": "running",
                "dataset_id": args.dataset_id,
                "source_stage": str(stage),
                "source_stage_sha256": source_stage_sha256,
                "capture_storage_profile": capture_storage_profile,
                "selected_state_indices": state_indices,
                "completed_state_indices": [item["state_index"] for item in receipts],
                "current_state_index": state_index,
                "current_state_sequence": sequence,
                "current_state_status": "rendering",
                "storage_capacity": capacity,
                "receipts": receipts,
                "publication_authorized": False,
                "training_admission_authorized": False,
            },
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=workdir,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Isaac production failed for state {state_index}; see {stderr_path}"
            )

        audit_path = chunk_root / "audit-report.json"
        audit_result = subprocess.run(
            [
                sys.executable,
                str(auditor),
                str(chunk_root),
                "--run-contract",
                str(chunk_root / "run-contract.json"),
                "--report",
                str(audit_path),
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if audit_result.returncode != 0:
            raise RuntimeError(
                f"capture audit failed for state {state_index}: {audit_result.stdout}"
            )
        receipt = passed_chunk_receipt(
            chunk_root,
            dataset_id=args.dataset_id,
            state_index=state_index,
            source_stage_sha256=source_stage_sha256,
        )
        if receipt is None:
            raise RuntimeError(f"state {state_index} did not produce an admissible receipt")
        receipts.append(receipt)

    completed_payload = {
        "schema": "fireviewer.dataset-chunk-campaign.v1",
        "status": "complete",
        "dataset_id": args.dataset_id,
        "source_stage": str(stage),
        "source_stage_sha256": source_stage_sha256,
        "capture_storage_profile": capture_storage_profile,
        "selected_state_indices": state_indices,
        "completed_state_indices": [item["state_index"] for item in receipts],
        "captures": sum(int(item["captures"]) for item in receipts),
        "viewpoint_plans": sum(int(item["viewpoint_plans"]) for item in receipts),
        "receipts": receipts,
        "publication_authorized": False,
        "training_admission_authorized": False,
    }
    write_json_atomic(campaign_progress_path, completed_payload)
    print(json.dumps(completed_payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args)
    except BaseException as exc:
        print(
            f"FireViewer chunk campaign failed [{type(exc).__name__}]: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
