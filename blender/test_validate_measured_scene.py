from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from typing import Mapping, Sequence
import zlib

import pytest

import validate_measured_scene as measured_qa


COUNTS = {
    "terrain_objects": 4,
    "building_instances": 125,
    "tree_instances": 830,
    "context_asset_instances": 0,
}


def test_contract_lists_every_required_d_only_runtime_variable() -> None:
    contract = json.loads(
        (
            Path(__file__).with_name("validate_measured_scene_contract.v1.json")
        ).read_text(encoding="utf-8")
    )
    assert set(contract["storage"]["temporary_environment_variables"]) == {
        "TEMP",
        "TMP",
        "PYTHONPYCACHEPREFIX",
        "BLENDER_USER_CONFIG",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_USER_DATAFILES",
        "BLENDER_USER_EXTENSIONS",
    }


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


def _good_ground(path: Path) -> None:
    colors = ((35, 83, 28), (158, 111, 42), (78, 91, 46), (122, 68, 35))
    _png(path, 32, 32, lambda x, y: colors[(x // 4 + y // 4) % len(colors)])


def _write_job(root: Path, *, white_ground: bool = False) -> Path:
    scene = root / "scene.usda"
    scene.write_text(
        '#usda 1.0\n(def Xform "Scene" { custom string fireviewer:category = "terrain"; custom int fireviewer:count = 1 })\n',
        encoding="utf-8",
    )
    texture = root / "ground-color.png"
    if white_ground:
        _png(texture, 16, 16, lambda _x, _y: (248, 248, 246))
    else:
        _good_ground(texture)
    job = measured_qa.build_job(
        job_root=root,
        scene_path=scene,
        ground_texture_path=texture,
        output_directory=root / "proof",
        expected_counts=COUNTS,
    )
    job_path = root / "measured-scene-render-job.v1.json"
    job_path.write_bytes(measured_qa.canonical_json_bytes(job))
    return job_path


def _renderer(
    _scene: Path,
    output: Path,
    captures: Sequence[Mapping[str, object]],
) -> Mapping[str, Path]:
    assert [capture["name"] for capture in captures] == ["topdown", "oblique"]
    result = {}
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


def test_success_is_technical_and_all_evidence_is_rehashed(tmp_path: Path) -> None:
    job_path = _write_job(tmp_path)
    receipt_path = measured_qa.run_validation(
        job_path,
        require_d=False,
        scene_counter=lambda _path: COUNTS,
        render_runner=_renderer,
    )
    assert receipt_path == tmp_path / measured_qa.RECEIPT_FILE_NAME
    receipt = measured_qa.verify_receipt(receipt_path, require_d=False)
    assert receipt["status"] == "rendered_pending_human_review"
    assert receipt["human_review_required"] is True
    assert receipt["counts"] == COUNTS
    assert set(receipt["captures"]) == {"topdown", "oblique"}
    assert receipt["ground_metrics"]["mean_luminance"] < 0.82
    assert receipt["ground_metrics"]["mean_saturation"] > 0.08


def test_scene_tampering_is_rejected_before_render(tmp_path: Path) -> None:
    job_path = _write_job(tmp_path)
    (tmp_path / "scene.usda").write_text("#usda 1.0\n# tampered\n", encoding="utf-8")
    called = False

    def renderer(*_args):
        nonlocal called
        called = True
        return {}

    with pytest.raises(
        measured_qa.MeasuredSceneQaError,
        match="scene (byte count changed|was modified)",
    ):
        measured_qa.run_validation(
            job_path,
            require_d=False,
            scene_counter=lambda _path: COUNTS,
            render_runner=renderer,
        )
    assert called is False


def test_job_content_tampering_is_rejected(tmp_path: Path) -> None:
    job_path = _write_job(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["expected_counts"]["tree_instances"] += 1
    job_path.write_bytes(measured_qa.canonical_json_bytes(job))
    with pytest.raises(measured_qa.MeasuredSceneQaError, match="Job content hash"):
        measured_qa.inspect_job(job_path, require_d=False)


def test_white_pale_flat_ground_is_fail_closed(tmp_path: Path) -> None:
    job_path = _write_job(tmp_path, white_ground=True)
    with pytest.raises(measured_qa.MeasuredSceneQaError, match="pale/white ground"):
        measured_qa.run_validation(
            job_path,
            require_d=False,
            scene_counter=lambda _path: COUNTS,
            render_runner=_renderer,
        )
    assert not (tmp_path / "proof" / "topdown.png").exists()


def test_exact_scene_inventory_is_required(tmp_path: Path) -> None:
    job_path = _write_job(tmp_path)
    wrong = dict(COUNTS)
    wrong["building_instances"] -= 1
    with pytest.raises(measured_qa.MeasuredSceneQaError, match="inventory mismatch"):
        measured_qa.run_validation(
            job_path,
            require_d=False,
            scene_counter=lambda _path: wrong,
            render_runner=_renderer,
        )


def test_capture_tampering_invalidates_receipt(tmp_path: Path) -> None:
    receipt_path = measured_qa.run_validation(
        _write_job(tmp_path),
        require_d=False,
        scene_counter=lambda _path: COUNTS,
        render_runner=_renderer,
    )
    capture = tmp_path / "proof" / "topdown.png"
    capture.write_bytes(capture.read_bytes() + b"tampered")
    with pytest.raises(measured_qa.MeasuredSceneQaError, match="byte count changed"):
        measured_qa.verify_receipt(receipt_path, require_d=False)


def test_receipt_cannot_claim_automatic_acceptance(tmp_path: Path) -> None:
    receipt_path = measured_qa.run_validation(
        _write_job(tmp_path),
        require_d=False,
        scene_counter=lambda _path: COUNTS,
        render_runner=_renderer,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "accepted"
    receipt["human_review_required"] = False
    receipt["receipt_content_sha256"] = measured_qa.canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_content_sha256"
        }
    )
    receipt_path.write_bytes(measured_qa.canonical_json_bytes(receipt))
    with pytest.raises(measured_qa.MeasuredSceneQaError, match="automatic visual"):
        measured_qa.verify_receipt(receipt_path, require_d=False)


def test_png_reader_needs_no_pillow_and_preserves_rgb_values(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    _png(path, 2, 1, lambda x, _y: ((12, 34, 56), (78, 90, 123))[x])
    width, height, pixels = measured_qa.load_rgb8_png(path)
    assert (width, height) == (2, 1)
    assert pixels == [(12, 34, 56), (78, 90, 123)]


def test_wrong_capture_size_is_rejected(tmp_path: Path) -> None:
    def wrong_renderer(
        _scene: Path,
        output: Path,
        _captures: Sequence[Mapping[str, object]],
    ) -> Mapping[str, Path]:
        result = {}
        for name in ("topdown", "oblique"):
            target = output / f"{name}.png"
            _png(target, 256, 256, lambda _x, _y: (40, 90, 35))
            result[name] = target
        return result

    with pytest.raises(measured_qa.MeasuredSceneQaError, match="512x512"):
        measured_qa.run_validation(
            _write_job(tmp_path),
            require_d=False,
            scene_counter=lambda _path: COUNTS,
            render_runner=wrong_renderer,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows D-only environment contract")
def test_runtime_environment_requires_every_blender_path_on_d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = (
        "TEMP",
        "TMP",
        "PYTHONPYCACHEPREFIX",
        "BLENDER_USER_CONFIG",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_USER_DATAFILES",
        "BLENDER_USER_EXTENSIONS",
    )
    for name in names:
        monkeypatch.setenv(name, str(tmp_path))
    measured_qa._validate_runtime_environment(require_d=True)

    monkeypatch.delenv("BLENDER_USER_EXTENSIONS")
    with pytest.raises(
        measured_qa.MeasuredSceneQaError,
        match="BLENDER_USER_EXTENSIONS must be defined on D",
    ):
        measured_qa._validate_runtime_environment(require_d=True)
