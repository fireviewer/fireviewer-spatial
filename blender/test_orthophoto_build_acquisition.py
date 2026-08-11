from __future__ import annotations

from collections import namedtuple
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image
import pytest


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from orthophoto_build_acquisition import (  # noqa: E402
    DISK_SAFETY_MARGIN_BYTES,
    HALO_M,
    OrthophotoAcquisitionError,
    RECEIPT_FILE_NAME,
    RESOLUTION_M,
    RGB_FILE_NAME,
    SealedDependentMap,
    WmtsMatrix,
    build_grid,
    build_wms_plan,
    build_wmts_plan,
    cleanup_orthophoto_band,
    load_canonical_rgb,
    prepare_orthophoto_band,
    receive_to_part,
    sha256_file,
)


DiskUsage = namedtuple("DiskUsage", "total used free")


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int | None = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._stream = io.BytesIO(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeTransport:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requests: list[tuple[str, str | None]] = []

    def __call__(self, request: object, *, timeout: float) -> FakeResponse:
        assert timeout > 0
        url = request.full_url  # type: ignore[attr-defined]
        range_header = request.get_header("Range")  # type: ignore[attr-defined]
        self.requests.append((url, range_header))
        query = parse_qs(urlsplit(url).query, keep_blank_values=True)
        key = (
            "wms-band"
            if query["SERVICE"] == ["WMS"]
            else f"r{query['TILEROW'][0]}_c{query['TILECOL'][0]}"
        )
        payload = self.payloads[key]
        if range_header is None:
            return FakeResponse(payload)
        offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
        remainder = payload[offset:]
        return FakeResponse(
            remainder,
            status=206,
            headers={
                "Content-Length": str(len(remainder)),
                "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
            },
        )


def _png(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        output, format="PNG"
    )
    return output.getvalue()


def _wms_plan(**overrides: object):
    arguments: dict[str, object] = {
        "core_bounds_l93_m": (700_000, 6_600_000, 700_500, 6_600_500),
        "band_id": "x700000_y6600000_s500",
        "service_url": "https://imagery.test/wms",
        "layer": "ORTHO.TEST",
        "provider_revision_id": "provider-2026-08-09",
        "dependent_map_ids": ("ground-profiles", "surface-overlays"),
        "maximum_download_bytes": 8 * 1024 * 1024,
    }
    arguments.update(overrides)
    return build_wms_plan(**arguments)  # type: ignore[arg-type]


def _free_disk(_path: Path) -> DiskUsage:
    return DiskUsage(100 * 1024**3, 0, 100 * 1024**3)


def test_wms_plan_is_canonical_lambert93_one_metre_with_bounded_halo() -> None:
    plan = _wms_plan()

    assert plan.grid.crs == "EPSG:2154"
    assert plan.grid.resolution_m == RESOLUTION_M == 1.0
    assert plan.grid.halo_m == HALO_M == 10.0
    assert plan.grid.request_bounds_l93_m == (
        699_990,
        6_599_990,
        700_510,
        6_600_510,
    )
    assert (plan.grid.width, plan.grid.height) == (520, 520)
    assert len(plan.requests) == 1
    query = parse_qs(urlsplit(plan.requests[0].url).query, keep_blank_values=True)
    assert query == {
        "BBOX": ["699990,6599990,700510,6600510"],
        "CRS": ["EPSG:2154"],
        "FORMAT": ["image/png"],
        "HEIGHT": ["520"],
        "LAYERS": ["ORTHO.TEST"],
        "REQUEST": ["GetMap"],
        "SERVICE": ["WMS"],
        "STYLES": [""],
        "TRANSPARENT": ["FALSE"],
        "VERSION": ["1.3.0"],
        "WIDTH": ["520"],
    }
    assert "0.5" not in plan.requests[0].url
    assert plan.projected_peak_disk_bytes > plan.maximum_download_bytes


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ((0, 0, 500, 499), "exactly 500"),
        ((0.5, 0, 500.5, 500), "global 1 m"),
        ((0, 0, 1_000, 1_000), "exactly 500"),
    ],
)
def test_grid_rejects_unaligned_or_unbounded_bands(
    bounds: tuple[float, float, float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_grid(bounds)


def test_provider_revision_is_required_and_changes_the_plan_hash() -> None:
    first = _wms_plan(provider_revision_id="provider-r1")
    second = _wms_plan(provider_revision_id="provider-r2")
    assert first.plan_sha256 != second.plan_sha256
    with pytest.raises(ValueError, match="provider_revision_id"):
        _wms_plan(provider_revision_id="")


def test_wmts_plan_derives_canonical_sorted_tiles_for_the_exact_band() -> None:
    matrix = WmtsMatrix(
        matrix_set="LAMB93",
        matrix="1m",
        top_left_l93_m=(0.0, 1024.0),
        tile_width_px=256,
        tile_height_px=256,
        matrix_width=8,
        matrix_height=8,
    )
    plan = build_wmts_plan(
        (256, 256, 756, 756),
        band_id="wmts-band",
        service_url="https://imagery.test/wmts",
        layer="ORTHO.TEST",
        provider_revision_id="provider-r1",
        dependent_map_ids=("classification",),
        maximum_download_bytes=16 * 1024 * 1024,
        wmts_matrix=matrix,
    )

    assert [request.key for request in plan.requests] == [
        f"r{row}_c{column}" for row in range(1, 4) for column in range(0, 3)
    ]
    for request in plan.requests:
        query = parse_qs(urlsplit(request.url).query)
        assert query["SERVICE"] == ["WMTS"]
        assert query["REQUEST"] == ["GetTile"]
        assert query["TILEMATRIXSET"] == ["LAMB93"]
        assert query["TILEMATRIX"] == ["1m"]


def test_wmts_matrix_cannot_enable_half_metre_downloads() -> None:
    matrix = WmtsMatrix(
        matrix_set="LAMB93",
        matrix="05m",
        top_left_l93_m=(0.0, 1024.0),
        tile_width_px=256,
        tile_height_px=256,
        matrix_width=8,
        matrix_height=8,
        resolution_m=0.5,
    )
    with pytest.raises(ValueError, match="exactly 1 m"):
        build_wmts_plan(
            (256, 256, 756, 756),
            band_id="wmts-band",
            service_url="https://imagery.test/wmts",
            layer="ORTHO.TEST",
            provider_revision_id="provider-r1",
            dependent_map_ids=("classification",),
            maximum_download_bytes=16 * 1024 * 1024,
            wmts_matrix=matrix,
        )


def test_wmts_matrix_origin_must_align_to_the_one_metre_grid() -> None:
    matrix = WmtsMatrix(
        matrix_set="LAMB93",
        matrix="1m",
        top_left_l93_m=(0.5, 1024.0),
        tile_width_px=256,
        tile_height_px=256,
        matrix_width=8,
        matrix_height=8,
    )
    with pytest.raises(ValueError, match="global 1 m grid"):
        build_wmts_plan(
            (256, 256, 756, 756),
            band_id="wmts-band",
            service_url="https://imagery.test/wmts",
            layer="ORTHO.TEST",
            provider_revision_id="provider-r1",
            dependent_map_ids=("classification",),
            maximum_download_bytes=16 * 1024 * 1024,
            wmts_matrix=matrix,
        )


def test_wms_acquisition_is_hashed_resumable_atomic_and_runtime_excluded(
    tmp_path: Path,
) -> None:
    plan = _wms_plan()
    columns = np.arange(520, dtype=np.uint16)[None, :]
    rows = np.arange(520, dtype=np.uint16)[:, None]
    rgb = np.empty((520, 520, 3), dtype=np.uint8)
    rgb[:, :, 0] = columns % 256
    rgb[:, :, 1] = rows % 256
    rgb[:, :, 2] = (columns + rows) % 256
    payload = _png(rgb)
    transport = FakeTransport({"wms-band": payload})

    receipt = prepare_orthophoto_band(
        plan, tmp_path, opener=transport, disk_usage=_free_disk
    )
    band_root = tmp_path / "orthophoto-build-bands" / plan.band_id

    assert receipt["schema"] == "fireviewer.orthophoto-build-source.v1"
    assert receipt["status"] == "prepared_temporary_build_source"
    assert receipt["provider_revision_id"] == plan.provider_revision_id
    assert receipt["raw_sources_retained"] is False
    assert receipt["canonical_rgb"]["gdal_geotransform"] == [  # type: ignore[index]
        699_990,
        1.0,
        0.0,
        6_600_510,
        0.0,
        -1.0,
    ]
    assert receipt["runtime_exclusions"] == [
        "orthophoto_payload",
        "orthophoto_texture",
        "procedural_ground_material",
        "uniform_0.5m_source",
    ]
    assert (band_root / RECEIPT_FILE_NAME).is_file()
    assert (band_root / RGB_FILE_NAME).is_file()
    assert not list(band_root.rglob("*.part"))
    assert not list(band_root.rglob("*.image"))
    assert np.array_equal(load_canonical_rgb(plan, band_root), rgb)
    raw = receipt["raw_sources"][0]  # type: ignore[index]
    assert raw["sha256"] == hashlib.sha256(payload).hexdigest()
    assert raw["byte_count"] == len(payload)
    request_count = len(transport.requests)
    assert (
        prepare_orthophoto_band(plan, tmp_path, opener=transport, disk_usage=_free_disk)
        == receipt
    )
    assert len(transport.requests) == request_count


def test_wmts_acquisition_mosaics_and_crops_exact_one_metre_rgb(
    tmp_path: Path,
) -> None:
    matrix = WmtsMatrix(
        matrix_set="LAMB93",
        matrix="1m",
        top_left_l93_m=(0.0, 1024.0),
        tile_width_px=256,
        tile_height_px=256,
        matrix_width=8,
        matrix_height=8,
    )
    plan = build_wmts_plan(
        (256, 256, 756, 756),
        band_id="wmts-band",
        service_url="https://imagery.test/wmts",
        layer="ORTHO.TEST",
        provider_revision_id="provider-r1",
        dependent_map_ids=("classification",),
        maximum_download_bytes=16 * 1024 * 1024,
        wmts_matrix=matrix,
    )
    payloads = {}
    for request in plan.requests:
        tile = np.empty((256, 256, 3), dtype=np.uint8)
        tile[:, :, 0] = int(request.tile_column)
        tile[:, :, 1] = int(request.tile_row)
        tile[:, :, 2] = int(request.tile_column) + int(request.tile_row)
        payloads[request.key] = _png(tile)

    prepare_orthophoto_band(
        plan,
        tmp_path,
        opener=FakeTransport(payloads),
        disk_usage=_free_disk,
    )
    rgb = load_canonical_rgb(plan, tmp_path / "orthophoto-build-bands" / plan.band_id)

    assert rgb.shape == (520, 520, 3)
    assert tuple(rgb[0, 0]) == (0, 1, 1)
    assert tuple(rgb[0, 10]) == (1, 1, 2)
    assert tuple(rgb[-1, -1]) == (2, 3, 5)


def test_interrupted_transfer_resumes_from_existing_part(tmp_path: Path) -> None:
    plan = _wms_plan()
    payload = _png(np.full((520, 520, 3), 91, dtype=np.uint8))
    transport = FakeTransport({"wms-band": payload})
    staging = tmp_path / "orthophoto-build-bands" / f".{plan.band_id}.staging"
    staging.mkdir(parents=True)
    part = staging / "wms-band.image.part"
    split = len(payload) // 3
    part.write_bytes(payload[:split])

    receipt = prepare_orthophoto_band(
        plan, tmp_path, opener=transport, disk_usage=_free_disk
    )

    assert transport.requests == [(plan.requests[0].url, f"bytes={split}-")]
    assert receipt["raw_sources"][0]["resumed_from_byte"] == split  # type: ignore[index]


def test_receive_rejects_server_ignoring_range_and_preserves_part(
    tmp_path: Path,
) -> None:
    plan = _wms_plan()
    payload = b"complete-payload"
    destination = tmp_path / "source.image"
    part = destination.with_name(destination.name + ".part")
    part.write_bytes(b"partial")

    def ignoring_range(request: object, *, timeout: float) -> FakeResponse:
        assert request.get_header("Range") == "bytes=7-"  # type: ignore[attr-defined]
        return FakeResponse(payload, status=200)

    with pytest.raises(OrthophotoAcquisitionError, match="expected HTTP 206"):
        receive_to_part(
            plan.requests[0].url,
            destination,
            maximum_bytes=1024,
            opener=ignoring_range,
        )
    assert part.read_bytes() == b"partial"


def test_disk_peak_is_checked_before_any_request(tmp_path: Path) -> None:
    plan = _wms_plan()
    called = False

    def forbidden_opener(_request: object, *, timeout: float) -> FakeResponse:
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    required = plan.projected_peak_disk_bytes + DISK_SAFETY_MARGIN_BYTES
    with pytest.raises(OrthophotoAcquisitionError, match="insufficient D: space"):
        prepare_orthophoto_band(
            plan,
            tmp_path,
            opener=forbidden_opener,
            disk_usage=lambda _path: DiskUsage(required, 1, required - 1),
        )
    assert called is False


def test_transparent_provider_image_fails_closed_and_keeps_resumable_part(
    tmp_path: Path,
) -> None:
    plan = _wms_plan()
    rgba = np.full((520, 520, 4), 255, dtype=np.uint8)
    rgba[0, 0, 3] = 0
    output = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG")
    transport = FakeTransport({"wms-band": output.getvalue()})

    with pytest.raises(OrthophotoAcquisitionError, match="transparency"):
        prepare_orthophoto_band(plan, tmp_path, opener=transport, disk_usage=_free_disk)

    staging = tmp_path / "orthophoto-build-bands" / f".{plan.band_id}.staging"
    assert (staging / "wms-band.image.part").is_file()
    assert not (tmp_path / "orthophoto-build-bands" / plan.band_id).exists()


def test_cleanup_requires_every_opaque_sealed_map_then_removes_only_band(
    tmp_path: Path,
) -> None:
    plan = _wms_plan()
    payload = _png(np.full((520, 520, 3), 127, dtype=np.uint8))
    prepare_orthophoto_band(
        plan,
        tmp_path,
        opener=FakeTransport({"wms-band": payload}),
        disk_usage=_free_disk,
    )
    receipts_root = tmp_path / "sealed-map-receipts"
    receipts_root.mkdir()
    proofs = []
    for map_id in plan.dependent_map_ids:
        path = receipts_root / f"{map_id}.done.json"
        path.write_text(f'{{"map_id":"{map_id}","sealed":true}}\n', "utf-8")
        proofs.append(
            SealedDependentMap(
                map_id=map_id,
                receipt_path=path,
                receipt_sha256=sha256_file(path),
            )
        )
    band_root = tmp_path / "orthophoto-build-bands" / plan.band_id

    with pytest.raises(OrthophotoAcquisitionError, match="incomplete"):
        cleanup_orthophoto_band(plan, tmp_path, proofs[:1])
    assert band_root.is_dir()

    cleanup = cleanup_orthophoto_band(plan, tmp_path, proofs)

    assert cleanup["status"] == "temporary_orthophoto_band_deleted"
    assert cleanup["deleted_bytes"] > 0
    assert cleanup["recoverable"] is False
    assert not band_root.exists()
    assert all(proof.receipt_path.is_file() for proof in proofs)


def test_tampered_map_or_rgb_never_authorizes_cleanup(tmp_path: Path) -> None:
    plan = _wms_plan(dependent_map_ids=("classification",))
    payload = _png(np.full((520, 520, 3), 45, dtype=np.uint8))
    prepare_orthophoto_band(
        plan,
        tmp_path,
        opener=FakeTransport({"wms-band": payload}),
        disk_usage=_free_disk,
    )
    band_root = tmp_path / "orthophoto-build-bands" / plan.band_id
    map_receipt = tmp_path / "classification.done"
    map_receipt.write_bytes(b"sealed")
    sealed = SealedDependentMap("classification", map_receipt, sha256_file(map_receipt))
    map_receipt.write_bytes(b"changed")
    with pytest.raises(OrthophotoAcquisitionError, match="absent or changed"):
        cleanup_orthophoto_band(plan, tmp_path, (sealed,))
    map_receipt.write_bytes(b"sealed")
    (band_root / RGB_FILE_NAME).write_bytes(b"changed")
    with pytest.raises(OrthophotoAcquisitionError, match="canonical orthophoto RGB"):
        cleanup_orthophoto_band(plan, tmp_path, (sealed,))
    assert band_root.is_dir()


def test_changed_raw_provenance_invalidates_the_temporary_source(
    tmp_path: Path,
) -> None:
    plan = _wms_plan()
    payload = _png(np.full((520, 520, 3), 33, dtype=np.uint8))
    prepare_orthophoto_band(
        plan,
        tmp_path,
        opener=FakeTransport({"wms-band": payload}),
        disk_usage=_free_disk,
    )
    band_root = tmp_path / "orthophoto-build-bands" / plan.band_id
    receipt_path = band_root / RECEIPT_FILE_NAME
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt["raw_sources"][0]["request_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), "utf-8")

    with pytest.raises(OrthophotoAcquisitionError, match="proof differs"):
        load_canonical_rgb(plan, band_root)


def test_dependent_receipts_inside_the_band_cannot_be_deleted_as_proof(
    tmp_path: Path,
) -> None:
    plan = _wms_plan(dependent_map_ids=("classification",))
    payload = _png(np.full((520, 520, 3), 22, dtype=np.uint8))
    prepare_orthophoto_band(
        plan,
        tmp_path,
        opener=FakeTransport({"wms-band": payload}),
        disk_usage=_free_disk,
    )
    band_root = tmp_path / "orthophoto-build-bands" / plan.band_id
    proof = band_root / "classification.done"
    proof.write_bytes(b"sealed")

    with pytest.raises(OrthophotoAcquisitionError, match="must survive"):
        cleanup_orthophoto_band(
            plan,
            tmp_path,
            (SealedDependentMap("classification", proof, sha256_file(proof)),),
        )
    assert band_root.is_dir()


def test_temporary_output_is_rejected_inside_a_git_worktree(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", "utf-8")
    plan = _wms_plan()
    with pytest.raises(OrthophotoAcquisitionError, match="outside a Git"):
        prepare_orthophoto_band(
            plan,
            tmp_path / "temporary",
            opener=FakeTransport({}),
            disk_usage=_free_disk,
        )
