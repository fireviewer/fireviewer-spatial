from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Mapping, Sequence
import zlib

import pytest

import run_simple_measured_tile_qa as tile_qa
import validate_measured_scene as measured_qa
import build_measured_scene_usd as measured_scene


BUILDING_COUNT = 2
TREE_COUNT = 3
EXPECTED_COUNTS = {
    "terrain_objects": 1,
    "building_instances": BUILDING_COUNT,
    "tree_instances": TREE_COUNT,
    "context_asset_instances": 0,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _compact_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _png(path: Path, width: int, height: int, pixel_at) -> None:
    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
        )

    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)
        for column in range(width):
            scanlines.extend(pixel_at(column, row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(scanlines), 9))
        + chunk(b"IEND", b"")
    )


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _reconciliation(count: int) -> dict[str, object]:
    return {
        "source_count": count,
        "instance_count": count,
        "blocked_count": 0,
        "valid_count": count,
        "quota_applied": False,
        "thinning_applied": False,
    }


def _write_scene_package(package_root: Path) -> None:
    scene_root = package_root / "scene"
    scene_root.mkdir(parents=True)
    (scene_root / "prototypes").mkdir()
    scene_path = scene_root / measured_scene.SCENE_FILE_NAME
    scene_path.write_text(
        '#usda 1.0\n(def Xform "Scene" { '
        'custom string fireviewer:category = "terrain"; '
        "custom int fireviewer:count = 1 })\n",
        encoding="utf-8",
    )
    receipt_without_hash: dict[str, object] = {
        "schema": measured_scene.RECEIPT_SCHEMA,
        "algorithm": measured_scene.ALGORITHM,
        "status": "technical_pilot_non_final",
        "accepted_final": False,
        "scene": {
            "path": measured_scene.SCENE_FILE_NAME,
            "sha256": _sha256_file(scene_path),
            "byte_count": scene_path.stat().st_size,
        },
        "prototype_count": 0,
        "prototypes": [],
        "prototype_bundle": {
            "root_reference": "prototypes",
            "scope": "output_local",
            "selected_asset_count": 0,
            "unused_catalog_assets_copied": 0,
            "absolute_asset_paths": False,
            "bundle_sha256": measured_scene.sha256_bytes(
                measured_scene.canonical_json_bytes([])
            ),
        },
        "reconciliation": {
            "buildings": _reconciliation(BUILDING_COUNT),
            "trees": _reconciliation(TREE_COUNT),
        },
    }
    receipt_without_hash["build_id"] = measured_scene.sha256_bytes(
        measured_scene.canonical_json_bytes(receipt_without_hash)
    )
    receipt = dict(receipt_without_hash)
    receipt["receipt_sha256"] = measured_scene.sha256_bytes(
        measured_scene.canonical_json_bytes(receipt_without_hash)
    )
    (scene_root / measured_scene.RECEIPT_FILE_NAME).write_bytes(
        measured_scene.canonical_json_bytes(receipt, pretty=True)
    )


def _write_simple_receipt(package_root: Path) -> None:
    output_paths = (
        package_root / "scene" / measured_scene.SCENE_FILE_NAME,
        package_root / "scene" / measured_scene.RECEIPT_FILE_NAME,
        package_root / "ground" / "ground-color.png",
    )
    outputs = {
        path.relative_to(package_root).as_posix(): _artifact(path, package_root)
        for path in output_paths
    }
    request = {
        "zone_id": "synthetic",
        "tile_id": package_root.name,
        "inputs": "local_only",
    }
    receipt: dict[str, object] = {
        "schema": tile_qa.SIMPLE_RECEIPT_SCHEMA,
        "status": "technical_pilot_non_final",
        "accepted_final": False,
        "request": request,
        "build_id": _sha256_bytes(_compact_bytes(request)),
        "placement": {
            "building_valid_count": BUILDING_COUNT,
            "tree_valid_count": TREE_COUNT,
        },
        "outputs": outputs,
    }
    receipt["receipt_sha256"] = _sha256_bytes(_compact_bytes(receipt))
    (package_root / tile_qa.SIMPLE_RECEIPT_NAME).write_bytes(
        measured_qa.canonical_json_bytes(receipt)
    )


def build_synthetic_package(tmp_path: Path) -> tuple[Path, Path]:
    package_root = tmp_path / "x820000_y6312500"
    package_root.mkdir()
    _write_scene_package(package_root)
    ground_path = package_root / "ground" / "ground-color.png"
    colors = ((35, 83, 28), (158, 111, 42), (78, 91, 46), (122, 68, 35))
    _png(
        ground_path,
        250,
        250,
        lambda x, y: colors[(x // 20 + y // 20) % len(colors)],
    )
    _write_simple_receipt(package_root)
    blender = tmp_path / "blender-4.5" / "blender.exe"
    blender.parent.mkdir()
    blender.write_bytes(b"synthetic Blender executable")
    return package_root, blender


@pytest.fixture
def synthetic_package(tmp_path: Path) -> tuple[Path, Path]:
    return build_synthetic_package(tmp_path)


def _render_proofs(
    _scene: Path,
    output: Path,
    captures: Sequence[Mapping[str, object]],
) -> Mapping[str, Path]:
    assert [capture["name"] for capture in captures] == ["topdown", "oblique"]
    result: dict[str, Path] = {}
    for index, name in enumerate(("topdown", "oblique")):
        target = output / f"{name}.png"
        _png(
            target,
            512,
            512,
            lambda x, y, index=index: (
                40 + (x + index * 11) % 120,
                55 + (y + index * 17) % 100,
                35 + (x + y) % 80,
            ),
        )
        result[name] = target
    return result


class FakeBlender:
    def __init__(self, version: str = "4.5.3") -> None:
        self.version = version
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __call__(
        self, command: Sequence[str], environment: Mapping[str, str]
    ) -> tile_qa.ProcessResult:
        argv = tuple(command)
        env = dict(environment)
        self.calls.append((argv, env))
        for name in tile_qa.ENVIRONMENT_PATHS:
            assert name in env
            path = Path(env[name])
            if os.name == "nt":
                assert path.drive.upper() == "D:"
            else:
                assert path.is_absolute()
            assert path.is_dir()
        if argv[1:] == ("--version",):
            return tile_qa.ProcessResult(0, f"Blender {self.version}\n", "")
        assert argv[1:6] == (
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
        )
        assert argv[6] == "--python"
        assert Path(argv[7]).name == "validate_measured_scene.py"
        assert argv[8:11] == ("--", "--job", argv[-1])
        job_path = Path(argv[-1])
        inspected = measured_qa.inspect_job(job_path, require_d=True)
        measured_qa.run_validation(
            job_path,
            require_d=False,
            scene_counter=lambda _path: inspected.expected_counts,
            render_runner=_render_proofs,
        )
        return tile_qa.ProcessResult(0, "synthetic QA complete\n", "")


def test_prepare_derives_counts_and_is_bit_reproducible(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    first = tile_qa.prepare_job(package_root, blender)
    first_bytes = first.read_bytes()
    second = tile_qa.prepare_job(package_root, blender)
    assert second == first
    assert second.read_bytes() == first_bytes
    inspected = measured_qa.inspect_job(first, require_d=True)
    assert inspected.root == package_root.resolve()
    assert inspected.expected_counts == EXPECTED_COUNTS
    assert inspected.output_directory == (package_root / "qa" / "renders").resolve()


def test_execute_uses_locked_blender_command_and_reuses_verified_link(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    fake = FakeBlender()
    link_path = tile_qa.execute_qa(package_root, blender, command_runner=fake)
    first_link_bytes = link_path.read_bytes()
    link = tile_qa.verify_link(package_root, blender)
    assert link["status"] == measured_qa.TECHNICAL_STATUS
    assert link["human_review_required"] is True
    assert link["accepted_human"] is False
    assert link["automatic_acceptance"] is False
    assert link["expected_counts"] == EXPECTED_COUNTS
    assert [call[0][1:] for call in fake.calls] == [
        ("--version",),
        (
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
            str(
                Path(tile_qa.__file__).with_name("validate_measured_scene.py").resolve()
            ),
            "--",
            "--job",
            str(package_root / tile_qa.JOB_NAME),
        ),
    ]
    assert not (package_root / tile_qa.RUNTIME_DIRECTORY).exists()

    assert tile_qa.execute_qa(package_root, blender, command_runner=fake) == link_path
    assert link_path.read_bytes() == first_link_bytes
    assert [call[0][1:] for call in fake.calls].count(("--version",)) == 2
    assert sum("--background" in call[0] for call in fake.calls) == 1


def test_package_tamper_is_rejected_before_blender(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    tile_qa.prepare_job(package_root, blender)
    scene = package_root / "scene" / measured_scene.SCENE_FILE_NAME
    scene.write_bytes(scene.read_bytes() + b"# tampered\n")
    fake = FakeBlender()
    with pytest.raises(
        tile_qa.SimpleMeasuredTileQaRunnerError,
        match="package output scene/scene.usda byte count changed",
    ):
        tile_qa.execute_qa(package_root, blender, command_runner=fake)
    assert fake.calls == []


def test_qa_capture_tamper_invalidates_link(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    tile_qa.execute_qa(package_root, blender, command_runner=FakeBlender())
    capture = package_root / "qa" / "renders" / "topdown.png"
    capture.write_bytes(capture.read_bytes() + b"tampered")
    with pytest.raises(
        tile_qa.SimpleMeasuredTileQaRunnerError,
        match="Technical QA receipt is invalid",
    ):
        tile_qa.verify_link(package_root, blender)


def test_wrong_blender_version_fails_closed_without_qa_receipt(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    with pytest.raises(
        tile_qa.SimpleMeasuredTileQaRunnerError,
        match="Blender must be version 4.5",
    ):
        tile_qa.execute_qa(
            package_root,
            blender,
            command_runner=FakeBlender(version="4.4.9"),
        )
    assert not (package_root / tile_qa.TECHNICAL_RECEIPT_NAME).exists()
    assert not (package_root / tile_qa.LINK_RECEIPT_NAME).exists()
    assert not (package_root / tile_qa.RUNTIME_DIRECTORY).exists()


def test_locked_job_corruption_is_not_overwritten(
    synthetic_package: tuple[Path, Path],
) -> None:
    package_root, blender = synthetic_package
    job_path = tile_qa.prepare_job(package_root, blender)
    job_path.write_bytes(job_path.read_bytes() + b" ")
    with pytest.raises(
        tile_qa.SimpleMeasuredTileQaRunnerError,
        match="Refusing to replace different locked QA artifact",
    ):
        tile_qa.prepare_job(package_root, blender)


def test_contract_locks_tile_local_layout_and_seven_d_variables() -> None:
    contract = json.loads(
        Path(tile_qa.__file__)
        .with_name("simple_measured_tile_qa_runner_contract.v1.json")
        .read_text(encoding="utf-8")
    )
    assert contract["layout"] == {
        "job": tile_qa.JOB_NAME,
        "captures": "qa/renders",
        "technical_receipt": tile_qa.TECHNICAL_RECEIPT_NAME,
        "link_receipt": tile_qa.LINK_RECEIPT_NAME,
        "runtime": "qa/runtime",
    }
    assert contract["blender"]["temporary_environment_variables"] == list(
        tile_qa.ENVIRONMENT_PATHS
    )
    assert len(tile_qa.ENVIRONMENT_PATHS) == 7
    assert contract["scope"]["automatic_human_acceptance"] is False


@pytest.mark.skipif(os.name != "nt", reason="Windows D-only runner contract")
def test_c_drive_paths_are_rejected(
    synthetic_package: tuple[Path, Path],
) -> None:
    _package_root, blender = synthetic_package
    with pytest.raises(
        tile_qa.SimpleMeasuredTileQaRunnerError,
        match="must remain on D:",
    ):
        tile_qa.prepare_job(Path("C:/forbidden/package"), blender)
