from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import rasterio


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from terrain_source_acquisition import (  # noqa: E402
    CANONICAL_SCHEMA,
    CORE_VERTEX_COUNT,
    CRS,
    HALO_M,
    NORMAL_HALO_VERTEX_COUNT,
    RESAMPLING,
    RESOLUTION_M,
    acquire_source_pair,
    build_grid,
    build_source_pair_plan,
    canonicalize_source_raster,
    cleanup_partial_files,
    receive_to_part,
    validate_coregistration,
)
from adaptive_terrain_quadtree import compile_adaptive_tile  # noqa: E402


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int | None,
        headers: dict[str, str],
    ) -> None:
        self._stream = io.BytesIO(payload)
        self.status = status
        self.headers = headers

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeTransport:
    def __init__(self, payload_by_layer: dict[str, bytes]) -> None:
        self.payload_by_layer = payload_by_layer
        self.requests: list[tuple[str, str | None]] = []
        self.ignore_range = False

    def __call__(self, request: object, *, timeout: float) -> FakeResponse:
        assert timeout > 0
        url = request.full_url  # type: ignore[attr-defined]
        range_header = request.get_header("Range")  # type: ignore[attr-defined]
        self.requests.append((url, range_header))
        layer = parse_qs(urlsplit(url).query)["LAYERS"][0]
        payload = self.payload_by_layer[layer]
        if range_header is not None and not self.ignore_range:
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            response = payload[offset:]
            return FakeResponse(
                response,
                status=206,
                headers={
                    "Content-Length": str(len(response)),
                    "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
                },
            )
        return FakeResponse(
            payload,
            status=200,
            headers={"Content-Length": str(len(payload))},
        )


def _geotiff_bytes(
    path: Path,
    *,
    width: int,
    height: int,
    transform: rasterio.Affine,
    base: float,
    nodata_pixel: bool = False,
    values: np.ndarray | None = None,
) -> bytes:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        raster_values = (
            np.arange(width * height, dtype="float32").reshape(height, width)
            if values is None
            else np.asarray(values, dtype="float32").copy()
        )
        if raster_values.shape != (height, width):
            raise ValueError("synthetic raster values have the wrong shape")
        if nodata_pixel:
            raster_values[0, 0] = -9999.0 - base
        dataset.write(raster_values + base, 1)
    payload = path.read_bytes()
    path.unlink()
    return payload


def _plan_and_transport(tmp_path: Path) -> tuple[object, FakeTransport]:
    plan = build_source_pair_plan(
        (700000.0, 6600000.0, 700500.0, 6600500.0),
        mnt_service_url="https://terrain.test/wms",
        mnt_layer="MNT.TEST",
        mns_service_url="https://terrain.test/wms",
        mns_layer="MNS.TEST",
    )
    mnt = _geotiff_bytes(
        tmp_path / "synthetic-mnt.tif",
        width=plan.grid.width,
        height=plan.grid.height,
        transform=plan.grid.transform,
        base=100.0,
    )
    mns = _geotiff_bytes(
        tmp_path / "synthetic-mns.tif",
        width=plan.grid.width,
        height=plan.grid.height,
        transform=plan.grid.transform,
        base=110.0,
    )
    return plan, FakeTransport({"MNT.TEST": mnt, "MNS.TEST": mns})


def test_plan_locks_square_lambert93_two_metre_grid_and_ten_metre_halo() -> None:
    plan = build_source_pair_plan(
        (700000.0, 6600000.0, 700500.0, 6600500.0),
        mnt_service_url="https://terrain.test/wms",
        mnt_layer="MNT.TEST",
        mns_service_url="https://terrain.test/wms",
        mns_layer="MNS.TEST",
    )

    assert plan.grid.crs == CRS
    assert plan.grid.resolution_m == RESOLUTION_M
    assert plan.grid.halo_m == HALO_M
    assert plan.grid.request_bounds_l93_m == (
        699990.0,
        6599990.0,
        700510.0,
        6600510.0,
    )
    assert (plan.grid.width, plan.grid.height) == (260, 260)
    for request in (plan.mnt, plan.mns):
        query = parse_qs(urlsplit(request.url).query, keep_blank_values=True)
        assert query["CRS"] == [CRS]
        assert query["BBOX"] == ["699990.000,6599990.000,700510.000,6600510.000"]
        assert query["WIDTH"] == ["260"]
        assert query["HEIGHT"] == ["260"]
        assert query["RESAMPLING"] == [RESAMPLING]


@pytest.mark.parametrize(
    "bounds, message",
    [
        ((0.0, 0.0, 500.0, 498.0), "square"),
        ((1.0, 0.0, 501.0, 500.0), "align"),
        ((0.0007, 0.0, 500.0007, 500.0), "align"),
    ],
)
def test_plan_rejects_non_square_or_unaligned_bounds(
    bounds: tuple[float, float, float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_grid(bounds)


def test_plan_rejects_orthophoto_and_non_https_sources() -> None:
    common = {
        "core_bounds_l93_m": (0.0, 0.0, 500.0, 500.0),
        "mnt_layer": "MNT.TEST",
        "mns_service_url": "https://terrain.test/wms",
        "mns_layer": "MNS.TEST",
    }
    with pytest.raises(ValueError, match="HTTPS or file"):
        build_source_pair_plan(mnt_service_url="http://terrain.test/wms", **common)
    with pytest.raises(ValueError, match="orthophoto"):
        build_source_pair_plan(
            mnt_service_url="https://terrain.test/wms",
            mnt_layer="ORTHOPHOTOS",
            mns_service_url="https://terrain.test/wms",
            mns_layer="MNS.TEST",
            core_bounds_l93_m=(0.0, 0.0, 500.0, 500.0),
        )


def test_wms_cells_canonicalize_to_shared_south_north_vertices(
    tmp_path: Path,
) -> None:
    plans = [
        build_source_pair_plan(
            (west, 6_600_000.0, west + 500.0, 6_600_500.0),
            mnt_service_url="https://terrain.test/wms",
            mnt_layer="MNT.TEST",
            mns_service_url="https://terrain.test/wms",
            mns_layer="MNS.TEST",
        )
        for west in (700_000.0, 700_500.0)
    ]
    canonical = []
    for index, plan in enumerate(plans):
        columns = np.arange(plan.grid.width, dtype="float64") + 0.5
        rows = np.arange(plan.grid.height, dtype="float64") + 0.5
        eastings = plan.grid.transform.c + columns * plan.grid.transform.a
        northings = plan.grid.transform.f + rows * plan.grid.transform.e
        values = 90.0 + 0.012 * eastings[None, :] + 0.006 * northings[:, None]
        raster_path = tmp_path / f"source-{index}.tif"
        raster_path.write_bytes(
            _geotiff_bytes(
                tmp_path / f"author-{index}.tif",
                width=plan.grid.width,
                height=plan.grid.height,
                transform=plan.grid.transform,
                base=0.0,
                values=values,
            )
        )
        canonical.append(canonicalize_source_raster(raster_path, plan.mnt))

    west, east = canonical
    assert west.core_mm.shape == (251, 251)
    assert west.normal_halo_mm.shape == (253, 253)
    assert west.core_mm[0, 0] < west.core_mm[-1, 0]
    assert np.array_equal(west.core_mm[:, -1], east.core_mm[:, 0])
    west_dx = west.normal_halo_mm[1:-1, 252] - west.normal_halo_mm[1:-1, 250]
    east_dx = east.normal_halo_mm[1:-1, 2] - east.normal_halo_mm[1:-1, 0]
    assert np.array_equal(west_dx, east_dx)
    compiled = compile_adaptive_tile(
        west.core_mm.astype("float64") / 1_000.0,
        normal_halo_heights_m=west.normal_halo_mm.astype("float64") / 1_000.0,
        tile_origin_l93_m=(700_000.0, 6_600_000.0),
    )
    assert compiled.normal_halo_sha256.hex() == west.normal_halo_sha256


def test_acquire_pair_is_synthetic_coregistered_hashed_and_atomic(
    tmp_path: Path,
) -> None:
    plan, transport = _plan_and_transport(tmp_path)
    output = tmp_path / "source-band"

    receipt = acquire_source_pair(plan, output, opener=transport)

    assert receipt["status"] == "downloaded_validated_coregistered_canonicalized"
    assert receipt["excluded"] == ["orthophoto", "uniform_0.5m_source"]
    for role in ("mnt", "mns"):
        destination = output / f"{role}-2m.tif"
        assert destination.is_file()
        assert not destination.with_name(destination.name + ".part").exists()
        source = receipt["sources"][role]
        assert source["byte_count"] == destination.stat().st_size
        assert source["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
        assert source["validation"]["crs"] == CRS
        canonical = source["canonical"]
        assert canonical["schema"] == CANONICAL_SCHEMA
        canonical_path = output / canonical["file_name"]
        assert canonical_path.is_file()
        with canonical_path.open("rb") as source_file:
            halo = np.load(source_file, allow_pickle=False)
        assert halo.shape == (NORMAL_HALO_VERTEX_COUNT, NORMAL_HALO_VERTEX_COUNT)
        assert halo[1:-1, 1:-1].shape == (CORE_VERTEX_COUNT, CORE_VERTEX_COUNT)
    validate_coregistration(output / "mnt-2m.tif", output / "mns-2m.tif")
    persisted = json.loads((output / "source-pair.done.json").read_text("utf-8"))
    assert persisted == receipt
    assert not (output / "source-pair.done.json.tmp").exists()
    request_count = len(transport.requests)
    assert acquire_source_pair(plan, output, opener=transport) == receipt
    assert len(transport.requests) == request_count
    (output / "mnt-2m.tif").unlink()
    (output / "mns-2m.tif").unlink()
    assert acquire_source_pair(plan, output, opener=transport) == receipt
    assert len(transport.requests) == request_count


@pytest.mark.parametrize(
    ("role", "field", "forged_name", "message"),
    (
        ("mnt", "raw", "../outside-mnt.tif", "raw source file name"),
        (
            "mns",
            "canonical",
            "../outside-mns.npy",
            "canonical file name",
        ),
    ),
)
def test_accepted_receipt_rejects_noncanonical_or_traversing_file_names(
    tmp_path: Path,
    role: str,
    field: str,
    forged_name: str,
    message: str,
) -> None:
    plan, transport = _plan_and_transport(tmp_path)
    output = tmp_path / "source-band"
    acquire_source_pair(plan, output, opener=transport)
    receipt_path = output / "source-pair.done.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source = receipt["sources"][role]
    if field == "raw":
        source["file_name"] = forged_name
    else:
        source["canonical"]["file_name"] = forged_name
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        acquire_source_pair(plan, output, opener=transport)


def test_strict_range_resume_appends_exact_remaining_bytes(tmp_path: Path) -> None:
    plan, transport = _plan_and_transport(tmp_path)
    payload = transport.payload_by_layer["MNT.TEST"]
    destination = tmp_path / "mnt-2m.tif"
    part = destination.with_name(destination.name + ".part")
    split = len(payload) // 3
    part.write_bytes(payload[:split])

    receipt = receive_to_part(plan.mnt.url, destination, opener=transport)

    assert transport.requests[-1][1] == f"bytes={split}-"
    assert receipt.resumed_from_byte == split
    assert receipt.byte_count == len(payload)
    assert part.read_bytes() == payload


def test_pair_resume_reuses_complete_part_without_eof_range(tmp_path: Path) -> None:
    plan, transport = _plan_and_transport(tmp_path)
    output = tmp_path / "source-band"
    staging = output.with_name(f".{output.name}.staging")
    staging.mkdir()
    (staging / "mnt-2m.tif.part").write_bytes(transport.payload_by_layer["MNT.TEST"])

    receipt = acquire_source_pair(plan, output, opener=transport)

    requested_layers = [
        parse_qs(urlsplit(url).query)["LAYERS"][0] for url, _range in transport.requests
    ]
    assert requested_layers == ["MNS.TEST"]
    assert (
        receipt["sources"]["mnt"]["resumed_from_byte"]
        == (output / "mnt-2m.tif").stat().st_size
    )
    assert output.is_dir()
    assert not staging.exists()


def test_invalid_eof_part_recovers_from_http_416_with_fresh_download(
    tmp_path: Path,
) -> None:
    plan, transport = _plan_and_transport(tmp_path)
    output = tmp_path / "source-band"
    staging = output.with_name(f".{output.name}.staging")
    staging.mkdir()
    expected_mnt = transport.payload_by_layer["MNT.TEST"]
    (staging / "mnt-2m.tif.part").write_bytes(b"x" * len(expected_mnt))

    def eof_aware_transport(request: object, *, timeout: float) -> FakeResponse:
        range_header = request.get_header("Range")  # type: ignore[attr-defined]
        url = request.full_url  # type: ignore[attr-defined]
        if range_header == f"bytes={len(expected_mnt)}-":
            transport.requests.append((url, range_header))
            raise HTTPError(url, 416, "Range Not Satisfiable", {}, None)
        return transport(request, timeout=timeout)

    receipt = acquire_source_pair(plan, output, opener=eof_aware_transport)

    mnt_requests = [
        range_header
        for url, range_header in transport.requests
        if parse_qs(urlsplit(url).query)["LAYERS"][0] == "MNT.TEST"
    ]
    assert mnt_requests == [f"bytes={len(expected_mnt)}-", None]
    assert (output / "mnt-2m.tif").read_bytes() == expected_mnt
    assert receipt["sources"]["mnt"]["resumed_from_byte"] == 0


def test_fresh_file_url_is_supported_through_injected_opener(tmp_path: Path) -> None:
    payload = b"synthetic-local-source"
    destination = tmp_path / "local-source.tif"
    seen: list[tuple[str, str | None]] = []

    def open_local(request: object, *, timeout: float) -> FakeResponse:
        assert timeout > 0
        seen.append(
            (
                request.full_url,  # type: ignore[attr-defined]
                request.get_header("Range"),  # type: ignore[attr-defined]
            )
        )
        return FakeResponse(
            payload,
            status=None,
            headers={"Content-Length": str(len(payload))},
        )

    receipt = receive_to_part(
        "file:///D:/synthetic/mnt-2m.tif",
        destination,
        opener=open_local,
    )

    assert seen == [("file:///D:/synthetic/mnt-2m.tif", None)]
    assert receipt.byte_count == len(payload)
    assert destination.with_name(destination.name + ".part").read_bytes() == payload


def test_range_resume_fails_closed_when_server_ignores_range(tmp_path: Path) -> None:
    plan, transport = _plan_and_transport(tmp_path)
    transport.ignore_range = True
    destination = tmp_path / "mnt-2m.tif"
    part = destination.with_name(destination.name + ".part")
    original = b"incomplete"
    part.write_bytes(original)

    with pytest.raises(RuntimeError, match="expected HTTP 206"):
        receive_to_part(plan.mnt.url, destination, opener=transport)

    assert part.read_bytes() == original
    assert not destination.exists()


def test_raster_validation_rejects_grid_drift(tmp_path: Path) -> None:
    plan, transport = _plan_and_transport(tmp_path)
    bad_transform = plan.grid.transform * rasterio.Affine.translation(1.0, 0.0)
    bad_mns = _geotiff_bytes(
        tmp_path / "bad-mns.tif",
        width=plan.grid.width,
        height=plan.grid.height,
        transform=bad_transform,
        base=110.0,
    )
    transport.payload_by_layer["MNS.TEST"] = bad_mns
    output = tmp_path / "source-band"

    with pytest.raises(RuntimeError, match="transform"):
        acquire_source_pair(plan, output, opener=transport)

    assert not (output / "mnt-2m.tif").exists()
    assert not (output / "mns-2m.tif").exists()
    staging = output.with_name(f".{output.name}.staging")
    assert (staging / "mnt-2m.tif.part").exists()
    assert (staging / "mns-2m.tif.part").exists()


def test_raster_validation_rejects_any_nodata_pixel(tmp_path: Path) -> None:
    plan, transport = _plan_and_transport(tmp_path)
    transport.payload_by_layer["MNS.TEST"] = _geotiff_bytes(
        tmp_path / "nodata-mns.tif",
        width=plan.grid.width,
        height=plan.grid.height,
        transform=plan.grid.transform,
        base=0.0,
        nodata_pixel=True,
    )

    with pytest.raises(RuntimeError, match="contains 1 nodata pixels"):
        acquire_source_pair(plan, tmp_path / "source-band", opener=transport)


def test_partial_cleanup_is_explicit_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "source-band"
    root.mkdir()
    mnt = root / "mnt-2m.tif"
    mns = root / "mns-2m.tif"
    mnt.with_name(mnt.name + ".part").write_bytes(b"mnt")
    mns.with_name(mns.name + ".part").write_bytes(b"mns")
    unrelated = root / "unrelated.part"
    unrelated.write_bytes(b"keep")

    removed = cleanup_partial_files(root, [mnt])

    assert removed == [mnt.with_name(mnt.name + ".part").resolve()]
    assert unrelated.is_file()
    assert mns.with_name(mns.name + ".part").is_file()
    with pytest.raises(ValueError, match="escapes"):
        cleanup_partial_files(root, [tmp_path.parent / "outside.tif"])
