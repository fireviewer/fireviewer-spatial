from __future__ import annotations

from types import SimpleNamespace

import pytest

from export_tiled_viewer_prototypes import (
    TiledPrototypeExportError,
    _resolve_prototype_root,
)


class FakeObject:
    def __init__(
        self,
        name: str,
        *,
        parent: FakeObject | None = None,
        asset_id: str | None = None,
    ) -> None:
        self.name = name
        self.parent = parent
        self._properties = (
            {"fireviewer:asset_id": asset_id} if asset_id is not None else {}
        )

    def get(self, key: str) -> object:
        return self._properties.get(key)


def _bpy(*objects: FakeObject) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(objects=list(objects)))


def test_resolve_prototype_root_uses_family_and_asset_metadata() -> None:
    trees = FakeObject("Trees")
    context_assets = FakeObject("ContextAssets")
    tree = FakeObject(
        "Asset_shared", parent=trees, asset_id="shared-asset"
    )
    context = FakeObject(
        "Asset_shared.001", parent=context_assets, asset_id="shared-asset"
    )
    assert (
        _resolve_prototype_root(
            _bpy(tree, context),
            family="context_assets",
            asset_id="shared-asset",
            identifier="Asset_shared",
        )
        is context
    )


def test_resolve_prototype_root_accepts_blender_truncated_name() -> None:
    scope = FakeObject("ContextAssets")
    asset_id = "5341fe716915_05_maison_individuelle_avec_petit_porche_moderne"
    truncated = FakeObject(
        "Asset_5341fe716915_05_maison_individuelle_avec_petit_porche_mod",
        parent=scope,
        asset_id=asset_id,
    )
    assert (
        _resolve_prototype_root(
            _bpy(truncated),
            family="context_assets",
            asset_id=asset_id,
            identifier=f"Asset_{asset_id}",
        )
        is truncated
    )


def test_resolve_prototype_root_accepts_flattened_blend_metadata() -> None:
    root = FakeObject("Asset_long_prototype_name")
    metadata = FakeObject("Source.004", parent=root, asset_id="long-prototype-name")
    assert (
        _resolve_prototype_root(
            _bpy(root, metadata),
            family="buildings",
            asset_id="long-prototype-name",
            identifier="Asset_long_prototype_name",
        )
        is root
    )


def test_resolve_prototype_root_rejects_ambiguous_family_asset() -> None:
    scope = FakeObject("Buildings")
    first = FakeObject("Asset_duplicate", parent=scope, asset_id="duplicate")
    second = FakeObject("Asset_duplicate.001", parent=scope, asset_id="duplicate")
    with pytest.raises(TiledPrototypeExportError, match="absent ou ambigu"):
        _resolve_prototype_root(
            _bpy(first, second),
            family="buildings",
            asset_id="duplicate",
            identifier="Asset_duplicate",
        )
