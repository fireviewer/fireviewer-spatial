from __future__ import annotations

import unittest
import gzip
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from build_control_scene import (
    COLOR_MANAGEMENT_DISPLAY,
    COLOR_MANAGEMENT_EXPOSURE,
    COLOR_MANAGEMENT_GAMMA,
    COLOR_MANAGEMENT_LOOK,
    COLOR_MANAGEMENT_VIEW_TRANSFORM,
    LIGHTING_BACKGROUND_COLOR,
    LIGHTING_BACKGROUND_STRENGTH,
    LIGHTING_RIG_SCHEMA,
    LIGHTING_SUN_ANGLE_DEGREES,
    LIGHTING_SUN_ENERGY,
    LIGHTING_SUN_OBJECT_NAME,
    LIGHTING_SUN_ROTATION_DEGREES,
    _configure_color_management,
    _configure_lighting_rig,
    _package_identity,
    _rollback_new_tile_datablocks,
    _resolve_tessellated_triangle,
    _snapshot_tile_datablocks,
    _srgb_to_linear_channel,
    _srgb_to_linear_color,
    _terrain_orthophoto_material_config,
    load_mid_vegetation_package,
    load_orthophoto_source,
)
from tree_instances import build_tree_instance_set


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float


class _Settings:
    pass


class _Scene(dict[str, object]):
    def __init__(self) -> None:
        super().__init__()
        self.display_settings = _Settings()
        self.view_settings = _Settings()


class _Socket:
    def __init__(self) -> None:
        self.default_value: object | None = None


class _Node:
    def __init__(self, node_type: str) -> None:
        self.bl_idname = node_type
        self.name = node_type
        self.label = ""
        if node_type == "ShaderNodeBackground":
            self.inputs = {"Color": _Socket(), "Strength": _Socket()}
            self.outputs = {"Background": _Socket()}
        elif node_type == "ShaderNodeOutputWorld":
            self.inputs = {"Surface": _Socket()}
            self.outputs = {}
        else:
            raise AssertionError(f"Unexpected test node type: {node_type}")


class _Links(list[tuple[_Socket, _Socket]]):
    def new(self, output: _Socket, input_socket: _Socket) -> None:
        self.append((output, input_socket))


class _Nodes(list[_Node]):
    def __init__(self, links: _Links) -> None:
        super().__init__()
        self._links = links

    def clear(self) -> None:
        super().clear()
        self._links.clear()

    def new(self, node_type: str) -> _Node:
        node = _Node(node_type)
        self.append(node)
        return node


class _NodeTree:
    def __init__(self) -> None:
        self.links = _Links()
        self.nodes = _Nodes(self.links)


class _World(dict[str, object]):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.use_nodes = False
        self.node_tree = _NodeTree()


class _NamedData(dict[str, object]):
    def new(self, name: str, *args: object, **kwargs: object) -> object:
        if kwargs.get("type") == "SUN":
            item: object = _Light(name, "SUN")
        elif args:
            item = _Object(name, args[0])
        else:
            item = _World(name)
        self[name] = item
        return item


class _Light(dict[str, object]):
    def __init__(self, name: str, light_type: str) -> None:
        super().__init__()
        self.name = name
        self.type = light_type
        self.energy = 0.0
        self.angle = 0.0


class _Object(dict[str, object]):
    def __init__(self, name: str, data: object) -> None:
        super().__init__()
        self.name = name
        self.data = data
        self.rotation_mode = ""
        self.rotation_euler: tuple[float, float, float] | None = None

    @property
    def type(self) -> str:
        return "LIGHT" if isinstance(self.data, _Light) else "MESH"


class _LinkedObjects(dict[str, _Object]):
    def __init__(self) -> None:
        super().__init__()
        self.link_calls = 0

    def link(self, item: _Object) -> None:
        self[item.name] = item
        self.link_calls += 1


class _Collection:
    def __init__(self) -> None:
        self.objects = _LinkedObjects()


class _LightingScene(dict[str, object]):
    def __init__(self) -> None:
        super().__init__()
        self.world: _World | None = None
        self.collection = _Collection()


class _BpyData:
    def __init__(self) -> None:
        self.worlds = _NamedData()
        self.lights = _NamedData()
        self.objects = _NamedData()


class _Bpy:
    def __init__(self) -> None:
        self.data = _BpyData()


class _RemovableData(list[object]):
    def remove(self, item: object, do_unlink: bool = False) -> None:
        del do_unlink
        super().remove(item)


class _StreamingId(dict[str, object]):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class _StreamingBpy:
    def __init__(self) -> None:
        data = type("StreamingData", (), {})()
        for group in (
            "objects",
            "meshes",
            "curves",
            "materials",
            "images",
            "node_groups",
            "collections",
        ):
            setattr(data, group, _RemovableData([_StreamingId(f"baseline_{group}")]))
        self.data = data


class TessellationCompatibilityTests(unittest.TestCase):
    def test_failed_tile_attempts_rollback_to_exact_datablock_baseline(self) -> None:
        bpy = _StreamingBpy()
        baseline_counts = {
            group: len(getattr(bpy.data, group))
            for group in (
                "objects",
                "meshes",
                "curves",
                "materials",
                "images",
                "node_groups",
                "collections",
            )
        }

        for attempt in range(3):
            snapshot = _snapshot_tile_datablocks(bpy)
            for group in baseline_counts:
                getattr(bpy.data, group).append(
                    _StreamingId(f"failed_{attempt}_{group}")
                )
            removed = _rollback_new_tile_datablocks(bpy, "tile_test", snapshot)
            self.assertEqual(len(removed), len(baseline_counts))
            self.assertEqual(
                {group: len(getattr(bpy.data, group)) for group in baseline_counts},
                baseline_counts,
            )

    def test_common_scene_color_management_is_agx_neutral_and_traced(self) -> None:
        scene = _Scene()

        settings = _configure_color_management(scene)

        self.assertEqual(
            scene.display_settings.display_device, COLOR_MANAGEMENT_DISPLAY
        )
        self.assertEqual(
            scene.view_settings.view_transform, COLOR_MANAGEMENT_VIEW_TRANSFORM
        )
        self.assertEqual(scene.view_settings.look, COLOR_MANAGEMENT_LOOK)
        self.assertEqual(scene.view_settings.exposure, COLOR_MANAGEMENT_EXPOSURE)
        self.assertEqual(scene.view_settings.gamma, COLOR_MANAGEMENT_GAMMA)
        self.assertEqual(
            json.loads(str(scene["fireviewer_color_management_json"])), settings
        )
        self.assertEqual(settings["view_transform"], "AgX")
        self.assertEqual(settings["look"], "None")
        self.assertEqual(settings["exposure_stops"], 0.0)

    def test_lighting_rig_is_exact_traced_and_idempotent(self) -> None:
        bpy = _Bpy()
        scene = _LightingScene()

        first = _configure_lighting_rig(bpy, scene)
        second = _configure_lighting_rig(bpy, scene)

        self.assertEqual(first, second)
        self.assertEqual(first["schema"], LIGHTING_RIG_SCHEMA)
        self.assertEqual(
            first["world"]["color_linear_rgba"],
            [0.18, 0.22, 0.28, 1.0],
        )
        self.assertEqual(first["world"]["strength"], 0.38)
        self.assertEqual(first["sun"]["energy"], 1.6)
        self.assertEqual(first["sun"]["angle_degrees"], 18.0)
        self.assertEqual(
            first["sun"]["rotation_euler_degrees"],
            [38.0, -24.0, -32.0],
        )

        world = scene.world
        assert world is not None
        self.assertTrue(world.use_nodes)
        self.assertEqual(world.color, LIGHTING_BACKGROUND_COLOR[:3])
        self.assertEqual(len(bpy.data.worlds), 1)
        self.assertEqual(len(world.node_tree.nodes), 2)
        self.assertEqual(len(world.node_tree.links), 1)
        background = next(
            node
            for node in world.node_tree.nodes
            if node.bl_idname == "ShaderNodeBackground"
        )
        self.assertEqual(
            background.inputs["Color"].default_value,
            LIGHTING_BACKGROUND_COLOR,
        )
        self.assertEqual(
            background.inputs["Strength"].default_value,
            LIGHTING_BACKGROUND_STRENGTH,
        )

        self.assertEqual(len(bpy.data.objects), 1)
        self.assertEqual(len(bpy.data.lights), 1)
        self.assertEqual(scene.collection.objects.link_calls, 1)
        sun = bpy.data.objects[LIGHTING_SUN_OBJECT_NAME]
        self.assertEqual(sun.data.type, "SUN")
        self.assertEqual(sun.data.energy, LIGHTING_SUN_ENERGY)
        self.assertAlmostEqual(
            sun.data.angle,
            math.radians(LIGHTING_SUN_ANGLE_DEGREES),
        )
        self.assertEqual(
            sun.rotation_euler,
            tuple(math.radians(value) for value in LIGHTING_SUN_ROTATION_DEGREES),
        )

        encoded = json.dumps(second, sort_keys=True, separators=(",", ":"))
        self.assertEqual(scene["fireviewer_lighting_rig_json"], encoded)
        self.assertEqual(world["fireviewer_lighting_rig_json"], encoded)
        self.assertEqual(sun["fireviewer_lighting_rig_json"], encoded)

    def test_global_terrain_uses_neutral_shader_grade_and_exact_mix(self) -> None:
        config = _terrain_orthophoto_material_config(
            "MAT_TerrainOrthophotoIGN2m",
            boundary_tolerance_m=15.0,
            pack_image_in_blend=True,
        )

        self.assertEqual(config.material_name, "MAT_TerrainOrthophotoIGN2m")
        self.assertEqual(config.shader_mode, "blender_balanced")
        self.assertEqual(config.texture_value, 1.0)
        self.assertEqual(config.texture_saturation, 1.0)
        self.assertEqual(config.principled_mix_fraction, 0.45)
        self.assertEqual(config.emission_mix_fraction, 0.55)
        self.assertEqual(config.emission_strength, 1.0)
        self.assertEqual(config.boundary_tolerance_m, 15.0)
        self.assertTrue(config.pack_image_in_blend)

    def test_srgb_palette_is_converted_to_scene_linear_without_changing_alpha(
        self,
    ) -> None:
        self.assertAlmostEqual(_srgb_to_linear_channel(0.5), 0.21404114048223255)
        self.assertEqual(
            _srgb_to_linear_color((0.0, 0.04045, 1.0, 0.75)),
            (0.0, 0.04045 / 12.92, 1.0, 0.75),
        )

    def test_srgb_palette_rejects_out_of_range_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "sRGB channel"):
            _srgb_to_linear_channel(1.01)
        with self.assertRaisesRegex(ValueError, "Alpha channel"):
            _srgb_to_linear_color((0.1, 0.2, 0.3, -0.01))

    def test_package_identity_is_portable_file_name_plus_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "control.json.gz"
            package.write_bytes(b"portable-package")
            file_name, digest = _package_identity(package.resolve())
        self.assertEqual(file_name, "control.json.gz")
        self.assertEqual(digest, hashlib.sha256(b"portable-package").hexdigest())
        self.assertNotIn(str(Path(directory)), file_name)

    def test_blender_52_integer_offsets_resolve_to_mesh_indices(self) -> None:
        self.assertEqual(
            _resolve_tessellated_triangle((1, 2, 0), [10, 12, 14], {}),
            [12, 14, 10],
        )

    def test_legacy_vector_output_resolves_by_coordinates(self) -> None:
        triangle = (Point(1.0, 0.0, 5.0), Point(0.0, 1.0, 5.0), Point(0.0, 0.0, 5.0))
        lookup = {
            (0.0, 0.0, 5.0): 20,
            (1.0, 0.0, 5.0): 22,
            (0.0, 1.0, 5.0): 24,
        }
        self.assertEqual(
            _resolve_tessellated_triangle(triangle, [], lookup),
            [22, 24, 20],
        )


class ExternalMidDistanceInputTests(unittest.TestCase):
    def test_mid_package_validates_instance_buffers_and_geospatial_contract(
        self,
    ) -> None:
        origin = [885_173.0, 6_404_926.0, 320.0]
        tree_instances = build_tree_instance_set(
            [
                {
                    "x_m": 887_000.0,
                    "y_m": 6_401_000.0,
                    "ground_elevation_m": 500.0,
                    "height_m": 12.0,
                    "crown_diameter_m": 6.0,
                }
            ],
            origin,
        )
        payload = {
            "schema": "fireviewer.vegetation-mid-distance-0m50.v1",
            "metadata": {
                "crs": "EPSG:2154",
                "origin_l93_m": origin,
                "bounds_l93_m": [886_500.0, 6_400_500.0, 887_500.0, 6_401_500.0],
            },
            "tree_instances": tree_instances,
            "terrain": {"vertices": [], "faces": []},
            "statistics": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "mid.json.gz"
            package.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))
            loaded = load_mid_vegetation_package(package)
        self.assertEqual(loaded["tree_instances"]["statistics"]["instance_count"], 1)

    def test_orthophoto_source_resolves_and_checks_blender_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "ortho.jpg"
            image.write_bytes(b"jpeg-test-payload")
            source = root / "ortho.source.json"
            source.write_text(
                json.dumps(
                    {
                        "schema": "fireviewer.ign-orthophoto-source.v1",
                        "request": {
                            "crs": "EPSG:2154",
                            "bounds_l93_m": [1.0, 2.0, 3.0, 4.0],
                        },
                        "outputs": [
                            {
                                "role": "blender_rgb_jpeg",
                                "file_name": image.name,
                                "sha256": hashlib.sha256(
                                    image.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded_image, bounds, _ = load_orthophoto_source(source)
        self.assertEqual(loaded_image, image)
        self.assertEqual(bounds, [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
