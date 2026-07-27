from __future__ import annotations

from email.message import Message
import io
import json
from pathlib import Path
from typing import Any
import urllib.parse

import pytest

from acquire_detail_sources import (
    AcquisitionError,
    AssetPlan,
    STATE_FILE_NAME,
    build_asset_plan,
    build_wfs_discovery_url,
    discover_copc,
    download_resumable,
    initialize_or_resume_root,
    load_zone_contract,
)


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._stream = io.BytesIO(payload)
        self.status = status
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "parent_package_id": "test-parent",
        "horizontal_crs": "EPSG:2154",
        "detail_resolution_metres": 0.5,
        "zones": [
            {
                "id": "barsac",
                "label": "Barsac",
                "bounds_l93_metres": [880_880.5, 6_405_709.5, 881_880.5, 6_406_709.5],
                "ign_lidar_hd_kilometre_tiles": ["0880_6406", "0881_6406"],
            }
        ],
    }


def _write_contract(path: Path) -> None:
    path.write_text(json.dumps(_contract()), encoding="utf-8")


def _properties(tile: str) -> dict[str, str]:
    return {
        "name": f"LHD_FXX_{tile}_PTS_O_LAMB93_IGN69",
        "url": f"https://data.geopf.fr/telechargement/download/lidar/{tile}.copc.laz",
        "timestamp": "2025-01-02T03:04:05Z",
    }


def _copc_asset() -> AssetPlan:
    return AssetPlan(
        category="copc",
        tile_id="0880_6406",
        url="https://data.geopf.fr/telechargement/download/test.copc.laz",
        relative_path="copc/LHD_FXX_0880_6406_PTS_LAMB93_IGN69.copc.laz",
        expected_magic_hex=b"LASF".hex(),
    )


def test_loads_non_montmaur_zone_and_validates_contract(tmp_path: Path) -> None:
    contract_path = tmp_path / "detail_zones.v1.json"
    _write_contract(contract_path)

    contract, zone = load_zone_contract(contract_path, "barsac")

    assert contract["parent_package_id"] == "test-parent"
    assert zone["id"] == "barsac"
    with pytest.raises(AcquisitionError, match="exactement une fois"):
        load_zone_contract(contract_path, "ausson")


def test_wfs_discovery_uses_official_contract_and_resolves_tiles() -> None:
    zone = _contract()["zones"][0]
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": _properties(tile)}
            for tile in zone["ign_lidar_hd_kilometre_tiles"]
        ],
    }
    requests = []

    def opener(request: Any, *, timeout: float) -> FakeResponse:
        requests.append((request, timeout))
        raw = json.dumps(collection).encode("utf-8")
        return FakeResponse(raw, headers={"Content-Length": str(len(raw))})

    discovery_url, discovered = discover_copc(
        zone, opener=opener, attempts=1, sleep=lambda _: None
    )
    query = urllib.parse.parse_qs(urllib.parse.urlparse(discovery_url).query)

    assert query["TYPENAMES"] == ["IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle"]
    assert query["SRSNAME"] == ["EPSG:4326"]
    assert query["BBOX"][0].endswith(",EPSG:4326")
    assert set(discovered) == {"0880_6406", "0881_6406"}
    assert requests[0][0].get_header("User-agent").startswith("FireWarning")


def test_plan_reuses_native_half_metre_wms_urls_and_validator_names() -> None:
    zone = _contract()["zones"][0]
    discovered = {
        tile: _properties(tile) for tile in zone["ign_lidar_hd_kilometre_tiles"]
    }

    plan = build_asset_plan(zone, discovered)

    assert len(plan) == 6
    assert [asset.category for asset in plan] == [
        "copc",
        "copc",
        "mnt",
        "mnt",
        "mns",
        "mns",
    ]
    mnt = next(asset for asset in plan if asset.category == "mnt")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(mnt.url).query)
    assert query["WIDTH"] == ["2000"]
    assert query["HEIGHT"] == ["2000"]
    assert query["CRS"] == ["EPSG:2154"]
    assert query["BBOX"] == ["879999.75,6405000.25,880999.75,6406000.25"]
    assert mnt.relative_path.endswith("_MNT_O_0M50_LAMB93_IGN69.tif")


def test_resumes_partial_http_range_then_atomically_publishes(tmp_path: Path) -> None:
    asset = _copc_asset()
    destination = tmp_path / "copc" / "tile.copc.laz"
    destination.parent.mkdir()
    partial = destination.with_name(f"{destination.name}.part")
    payload = b"LASF" + b"abcdefghij"
    partial.write_bytes(payload[:7])
    observed_ranges = []

    def opener(request: Any, *, timeout: float) -> FakeResponse:
        del timeout
        observed_ranges.append(request.get_header("Range"))
        offset = int(request.get_header("Range").split("=")[1].rstrip("-"))
        remainder = payload[offset:]
        return FakeResponse(
            remainder,
            status=206,
            headers={
                "Content-Length": str(len(remainder)),
                "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
            },
        )

    size = download_resumable(asset, destination, opener=opener, attempts=1)

    assert size == len(payload)
    assert observed_ranges == ["bytes=7-"]
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_server_ignoring_range_restarts_only_private_partial(tmp_path: Path) -> None:
    asset = _copc_asset()
    destination = tmp_path / "copc" / "tile.copc.laz"
    destination.parent.mkdir()
    partial = destination.with_name(f"{destination.name}.part")
    partial.write_bytes(b"LASFstale")
    payload = b"LASFcomplete"

    def opener(request: Any, *, timeout: float) -> FakeResponse:
        del timeout
        assert request.get_header("Range") == "bytes=9-"
        return FakeResponse(
            payload,
            status=200,
            headers={"Content-Length": str(len(payload))},
        )

    download_resumable(asset, destination, opener=opener, attempts=1)

    assert destination.read_bytes() == payload


def test_incomplete_response_keeps_part_and_never_publishes(tmp_path: Path) -> None:
    asset = _copc_asset()
    destination = tmp_path / "copc" / "tile.copc.laz"

    def opener(request: Any, *, timeout: float) -> FakeResponse:
        del request, timeout
        return FakeResponse(
            b"LASFshort",
            status=200,
            headers={"Content-Length": "100"},
        )

    with pytest.raises(AcquisitionError, match="incomplet"):
        download_resumable(asset, destination, opener=opener, attempts=1)

    assert not destination.exists()
    assert (
        destination.with_name(f"{destination.name}.part").read_bytes() == b"LASFshort"
    )


def test_new_output_root_is_bound_and_unrelated_directory_is_refused(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "detail_zones.v1.json"
    _write_contract(contract_path)
    contract, zone = load_zone_contract(contract_path, "barsac")
    discovered = {
        tile: _properties(tile) for tile in zone["ign_lidar_hd_kilometre_tiles"]
    }
    assets = build_asset_plan(zone, discovered)
    root = tmp_path / "new-sources"

    first = initialize_or_resume_root(
        root,
        contract_path=contract_path,
        contract=contract,
        zone=zone,
        discovery_url=build_wfs_discovery_url(zone),
        assets=assets,
    )
    second = initialize_or_resume_root(
        root,
        contract_path=contract_path,
        contract=contract,
        zone=zone,
        discovery_url=build_wfs_discovery_url(zone),
        assets=assets,
    )

    assert first == second
    assert (root / STATE_FILE_NAME).is_file()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "mine.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="pas une acquisition reprenable"):
        initialize_or_resume_root(
            unrelated,
            contract_path=contract_path,
            contract=contract,
            zone=zone,
            discovery_url=build_wfs_discovery_url(zone),
            assets=assets,
        )
    assert (unrelated / "mine.txt").read_text(encoding="utf-8") == "preserve"
