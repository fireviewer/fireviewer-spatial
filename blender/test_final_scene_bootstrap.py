from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from final_scene_bootstrap import (
    install_persistent_addon,
    package_persistent_addon,
    restore_material_viewport,
)


def _area(editor_type: str) -> SimpleNamespace:
    region = SimpleNamespace(
        view_location=None,
        view_distance=None,
        view_perspective="ORTHO",
    )
    shading = SimpleNamespace(
        type="SOLID",
        use_scene_world=False,
        use_scene_lights=False,
    )
    space = SimpleNamespace(
        shading=shading,
        clip_start=None,
        clip_end=None,
        region_3d=region,
    )
    return SimpleNamespace(type=editor_type, spaces=SimpleNamespace(active=space))


def test_restore_material_viewport_converts_active_python_console() -> None:
    console = _area("CONSOLE")
    bpy = SimpleNamespace(
        context=SimpleNamespace(
            area=console,
            screen=SimpleNamespace(areas=[console]),
        )
    )

    count = restore_material_viewport(
        bpy,
        local_focus=(10.0, 20.0, 360.0),
        view_distance_m=600.0,
    )

    assert count == 1
    assert console.type == "VIEW_3D"
    assert console.spaces.active.shading.type == "MATERIAL"
    assert console.spaces.active.shading.use_scene_world is True
    assert console.spaces.active.shading.use_scene_lights is True
    assert console.spaces.active.region_3d.view_location == (10.0, 20.0, 360.0)
    assert console.spaces.active.region_3d.view_distance == 600.0
    assert console.spaces.active.region_3d.view_perspective == "PERSP"


def test_install_persistent_addon_copies_enables_and_saves(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "fireviewer_spatial_lod_addon"
    source.mkdir()
    (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    user_scripts = tmp_path / "user-scripts" / "addons"
    addons: dict[str, object] = {}
    calls: list[tuple[str, object]] = []

    def enable(name: str, *, default_set: bool, persistent: bool) -> object:
        calls.append(("enable", (name, default_set, persistent)))
        module = object()
        addons[name] = module
        return module

    fake_addon_utils = SimpleNamespace(
        modules=lambda *, refresh: calls.append(("refresh", refresh)),
        enable=enable,
    )
    monkeypatch.setattr(
        "final_scene_bootstrap.importlib.import_module",
        lambda name: fake_addon_utils if name == "addon_utils" else None,
    )
    bpy = SimpleNamespace(
        utils=SimpleNamespace(
            user_resource=lambda _kind, *, path, create: str(user_scripts)
        ),
        context=SimpleNamespace(preferences=SimpleNamespace(addons=addons)),
        ops=SimpleNamespace(wm=SimpleNamespace(save_userpref=lambda: {"FINISHED"})),
    )

    destination = install_persistent_addon(
        bpy,
        source_directory=source,
        module_name=source.name,
    )

    assert destination == user_scripts / source.name
    assert (destination / "__init__.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert calls == [
        ("refresh", True),
        ("enable", (source.name, True, True)),
    ]


def test_package_persistent_addon_contains_importable_package(tmp_path: Path) -> None:
    source = tmp_path / "fireviewer_spatial_lod_addon"
    source.mkdir()
    (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    archive = package_persistent_addon(
        source_directory=source,
        destination_zip=tmp_path / "dist" / "fireviewer-spatial-lod.zip",
    )

    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["fireviewer_spatial_lod_addon/__init__.py"]
