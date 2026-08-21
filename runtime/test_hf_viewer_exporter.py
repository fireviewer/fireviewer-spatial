from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import hf_viewer_exporter


def test_publish_viewer_writes_verified_publication_receipt(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path / "build"
    (root / "viewer-tiled").mkdir(parents=True)
    (root / "zone.done.json").write_text(
        json.dumps(
            {
                "schema": "fireviewer.simple-measured-zone-production.v1",
                "status": "technical_scene_produced",
                "zone_id": "GPS-TEST",
                "build_id": "b" * 64,
                "tile_count": 9,
                "degraded_mns_tile_count": 0,
                "placeholder_instance_count": 0,
            }
        ),
        encoding="utf-8",
    )

    viewer = {
        "catalog_path": "viewer-tiled/catalog.json",
        "receipt_path": "viewer-tiled/viewer-tiled-scene.v1.json",
        "catalog_sha256": "c" * 64,
        "catalog_byte_count": 123,
        "payload_file_count": 32,
        "payload_byte_count": 456,
        "bootstrap_asset": {
            "path": "viewer-tiled/far.glb",
            "sha256": "d" * 64,
            "byte_count": 78,
            "media_type": "model/gltf-binary",
        },
        "representation": "complete_tiled_non_simplified_map",
        "completeness": {
            "policy": "fail_closed_exact_visual_scene",
            "mesh_coverage": "complete",
            "family_instance_counts": {
                "buildings": 4,
                "trees": 70_197,
                "context_assets": 0,
            },
        },
    }
    observed: dict[str, object] = {}

    def validate(
        path: Path, *, require_sealed_source_assets: bool
    ) -> tuple[dict, dict]:
        observed["validated_path"] = path
        observed["source_assets"] = require_sealed_source_assets
        return {"status": "complete"}, viewer

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            observed["token_length"] = len(token)

        def repo_info(self, **kwargs: object) -> object:
            observed["repo_info"] = kwargs
            return SimpleNamespace(private=False)

        def upload_folder(self, **kwargs: object) -> object:
            observed["upload"] = kwargs
            return SimpleNamespace(oid="e" * 40)

        def file_exists(self, **kwargs: object) -> bool:
            observed.setdefault("verified", []).append(kwargs)
            return True

    monkeypatch.setattr(hf_viewer_exporter, "validate_tiled_viewer_package", validate)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    destination = tmp_path / "publication.json"
    receipt = hf_viewer_exporter.publish_viewer(
        root,
        repo_id="fireviewer/simple-measured-scenes-v1",
        remote_root=f"maps/GPS-TEST/{'b' * 64}/runtime",
        job_id="map-test-job",
        exporter_image_digest="sha256:" + "f" * 64,
        output_receipt=destination,
        token="test-token-not-a-secret-123456",
    )

    assert receipt["schema"] == "fireviewer.hf-viewer-publication.v1"
    assert receipt["dataset"]["revision"] == "e" * 40
    assert receipt["viewer"]["catalog_path"].endswith(
        "/runtime/viewer-tiled/catalog.json"
    )
    assert receipt["viewer"]["bootstrap_asset"]["path"].endswith(
        "/runtime/viewer-tiled/far.glb"
    )
    assert observed["source_assets"] is False
    assert observed["upload"]["path_in_repo"].endswith("/runtime/viewer-tiled")
    assert len(observed["verified"]) == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == receipt
