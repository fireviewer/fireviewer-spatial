from __future__ import annotations

from pathlib import Path
import sys

import pytest


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from build_archive_manifest import build_manifest, verify_manifest  # noqa: E402


def test_archive_manifest_locks_every_file(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    (root / "sources").mkdir(parents=True)
    (root / "sources/a.bin").write_bytes(b"a" * 128)
    release_bytes = b'{"ready":true}\n'
    (root / "release.json").write_bytes(release_bytes)
    manifest_path = root / "archive-manifest.json"

    manifest = build_manifest(
        root,
        manifest_path,
        archive_id="zone-r1",
        description="test",
    )
    report = verify_manifest(root, manifest_path)

    assert manifest["inventory"] == {
        "file_count": 2,
        "byte_count": 128 + len(release_bytes),
        "sha256_algorithm": "SHA-256",
    }
    assert report["status"] == "valid"

    (root / "sources/a.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size differs"):
        verify_manifest(root, manifest_path)
