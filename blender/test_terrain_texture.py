from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from terrain_texture import (
    Lambert93Bounds,
    OrthophotoMaterialConfig,
    SOLID_VIEW_FALLBACK_RGBA,
    assign_lambert93_uv_layer,
    connect_balanced_orthophoto_shader,
    connect_orthophoto_color_to_principled,
    lambert93_uv_from_local_vertex,
    loop_uvs_from_indexed_faces,
    validate_gltf_core_image_path,
)


class Lambert93UvTests(unittest.TestCase):
    def test_local_vertices_round_trip_to_lambert93_image_corners(self) -> None:
        origin = (885_000.0, 6_400_000.0, 317.0)
        bounds = Lambert93Bounds(884_000.0, 6_399_000.0, 886_000.0, 6_401_000.0)

        self.assertEqual(
            lambert93_uv_from_local_vertex((-1_000.0, -1_000.0, -99.0), origin, bounds),
            (0.0, 0.0),
        )
        self.assertEqual(
            lambert93_uv_from_local_vertex((1_000.0, 1_000.0, 9_999.0), origin, bounds),
            (1.0, 1.0),
        )
        self.assertEqual(
            lambert93_uv_from_local_vertex((0.0, 0.0, 0.0), origin, bounds),
            (0.5, 0.5),
        )

    def test_north_up_raster_is_not_vertically_flipped(self) -> None:
        bounds = Lambert93Bounds(100.0, 200.0, 200.0, 400.0)

        south_cell_centre = lambert93_uv_from_local_vertex(
            (5.0, 5.0, 0.0), (100.0, 200.0, 0.0), bounds
        )
        north_cell_centre = lambert93_uv_from_local_vertex(
            (5.0, 195.0, 0.0), (100.0, 200.0, 0.0), bounds
        )

        self.assertEqual(south_cell_centre, (0.05, 0.025))
        self.assertEqual(north_cell_centre, (0.05, 0.975))

    def test_only_tolerated_millimetre_rounding_is_clamped(self) -> None:
        bounds = Lambert93Bounds(0.0, 0.0, 10.0, 10.0)

        self.assertEqual(
            lambert93_uv_from_local_vertex(
                (-0.001, 10.001, 0.0),
                (0.0, 0.0, 0.0),
                bounds,
                boundary_tolerance_m=0.01,
            ),
            (0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "outside orthophoto bounds"):
            lambert93_uv_from_local_vertex(
                (-0.02, 5.0, 0.0),
                (0.0, 0.0, 0.0),
                bounds,
                boundary_tolerance_m=0.01,
            )

    def test_indexed_faces_expand_to_blender_loop_order(self) -> None:
        vertices = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
        ]
        faces = [(0, 1, 2), (0, 2, 3)]

        self.assertEqual(
            loop_uvs_from_indexed_faces(
                vertices, faces, (0.0, 0.0, 0.0), (0.0, 0.0, 10.0, 10.0)
            ),
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )

    def test_invalid_bounds_origin_and_face_index_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "east bound"):
            Lambert93Bounds(10.0, 0.0, 10.0, 20.0)
        with self.assertRaisesRegex(ValueError, "exactly x, y and z"):
            lambert93_uv_from_local_vertex(
                (0.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 10.0, 10.0)
            )
        with self.assertRaisesRegex(ValueError, "references vertex 3"):
            loop_uvs_from_indexed_faces(
                [(0.0, 0.0, 0.0)],
                [(0, 0, 3)],
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 10.0, 10.0),
            )

    def test_core_gltf_image_validation_accepts_only_png_and_jpeg(self) -> None:
        self.assertEqual(validate_gltf_core_image_path("tile.PNG").suffix, ".PNG")
        self.assertEqual(validate_gltf_core_image_path("tile.jpeg").suffix, ".jpeg")
        with self.assertRaisesRegex(ValueError, "not a core glTF image"):
            validate_gltf_core_image_path("orthophoto.cog.tif")

    def test_material_contract_requires_linear_filtered_clamped_sampling(self) -> None:
        config = OrthophotoMaterialConfig()
        config.validate()
        self.assertEqual(config.shader_mode, "blender_balanced")
        self.assertEqual(config.texture_value, 1.0)
        self.assertEqual(config.texture_saturation, 1.0)
        self.assertEqual(config.principled_mix_fraction, 0.45)
        self.assertEqual(config.emission_mix_fraction, 0.55)
        self.assertEqual(config.emission_strength, 1.0)
        self.assertEqual(config.gltf_emission_strength, 0.2)
        self.assertEqual(config.solid_view_fallback_rgba, (0.12, 0.18, 0.10, 1.0))
        self.assertEqual(config.solid_view_fallback_rgba, SOLID_VIEW_FALLBACK_RGBA)
        with self.assertRaisesRegex(ValueError, "interpolation"):
            OrthophotoMaterialConfig(interpolation="Closest").validate()
        with self.assertRaisesRegex(ValueError, "extension"):
            OrthophotoMaterialConfig(extension="REPEAT").validate()
        with self.assertRaisesRegex(ValueError, "emission_strength"):
            OrthophotoMaterialConfig(emission_strength=4.01).validate()
        with self.assertRaisesRegex(ValueError, "must sum to 1"):
            OrthophotoMaterialConfig(
                principled_mix_fraction=0.5,
                emission_mix_fraction=0.55,
            ).validate()
        with self.assertRaisesRegex(ValueError, "solid_view_fallback_rgba"):
            OrthophotoMaterialConfig(
                solid_view_fallback_rgba=(1.0, 1.0, 1.0, 1.1)
            ).validate()


class _Socket:
    def __init__(self) -> None:
        self.default_value: float | None = None


class _Sockets:
    def __init__(
        self,
        named: dict[str, _Socket] | None = None,
        ordered: list[_Socket] | None = None,
    ) -> None:
        self._named = named or {}
        self._ordered = ordered or []

    def get(self, name: str) -> _Socket | None:
        return self._named.get(name)

    def __getitem__(self, key: str | int) -> _Socket:
        if isinstance(key, int):
            return self._ordered[key]
        return self._named[key]


class _Node:
    def __init__(
        self,
        *,
        inputs: dict[str, _Socket] | _Sockets | None = None,
        outputs: dict[str, _Socket] | _Sockets | None = None,
    ) -> None:
        self.inputs = inputs if isinstance(inputs, _Sockets) else _Sockets(inputs)
        self.outputs = outputs if isinstance(outputs, _Sockets) else _Sockets(outputs)


class _Links:
    def __init__(self) -> None:
        self.connections: list[tuple[_Socket, _Socket]] = []

    def new(self, source: _Socket, destination: _Socket) -> None:
        self.connections.append((source, destination))


class OrthophotoEmissionGraphTests(unittest.TestCase):
    def test_same_texture_drives_albedo_and_restrained_emission(self) -> None:
        color = _Socket()
        base_color = _Socket()
        emission_color = _Socket()
        emission_strength = _Socket()
        image = _Node(outputs={"Color": color})
        principled = _Node(
            inputs={
                "Base Color": base_color,
                "Emission Color": emission_color,
                "Emission Strength": emission_strength,
            }
        )
        links = _Links()

        contract = connect_orthophoto_color_to_principled(
            image,
            principled,
            links,
            emission_strength=0.2,
        )

        self.assertEqual(
            links.connections,
            [(color, base_color), (color, emission_color)],
        )
        self.assertEqual(emission_strength.default_value, 0.2)
        self.assertEqual(contract["emission_color_source"], "same_orthophoto_srgb")
        self.assertEqual(contract["emission_strength"], 0.2)

    def test_missing_blender_emission_socket_fails_explicitly(self) -> None:
        image = _Node(outputs={"Color": _Socket()})
        principled = _Node(inputs={"Base Color": _Socket()})
        with self.assertRaisesRegex(RuntimeError, "emission sockets"):
            connect_orthophoto_color_to_principled(
                image,
                principled,
                _Links(),
                emission_strength=0.2,
            )

    def test_balanced_graph_uses_neutral_texture_and_exact_45_55_mix(self) -> None:
        image_color = _Socket()
        adjusted_input = _Socket()
        adjusted_color = _Socket()
        hue = _Socket()
        saturation = _Socket()
        value = _Socket()
        adjustment_factor = _Socket()
        base_color = _Socket()
        principled_shader = _Socket()
        emission_color = _Socket()
        emission_strength = _Socket()
        emission_shader = _Socket()
        mix_factor = _Socket()
        mix_principled = _Socket()
        mix_emission = _Socket()
        mixed_shader = _Socket()
        surface = _Socket()
        image = _Node(outputs={"Color": image_color})
        adjustment = _Node(
            inputs={
                "Hue": hue,
                "Saturation": saturation,
                "Value": value,
                "Fac": adjustment_factor,
                "Color": adjusted_input,
            },
            outputs={"Color": adjusted_color},
        )
        principled = _Node(
            inputs={"Base Color": base_color},
            outputs={"BSDF": principled_shader},
        )
        emission = _Node(
            inputs={"Color": emission_color, "Strength": emission_strength},
            outputs={"Emission": emission_shader},
        )
        mix = _Node(
            inputs=_Sockets(
                ordered=[mix_factor, mix_principled, mix_emission],
            ),
            outputs={"Shader": mixed_shader},
        )
        output = _Node(inputs={"Surface": surface})
        links = _Links()
        config = OrthophotoMaterialConfig()

        contract = connect_balanced_orthophoto_shader(
            image,
            adjustment,
            principled,
            emission,
            mix,
            output,
            links,
            config=config,
        )

        self.assertEqual(value.default_value, 1.0)
        self.assertEqual(saturation.default_value, 1.0)
        self.assertEqual(hue.default_value, 0.5)
        self.assertEqual(adjustment_factor.default_value, 1.0)
        self.assertEqual(mix_factor.default_value, 0.55)
        self.assertEqual(emission_strength.default_value, 1.0)
        self.assertEqual(
            links.connections,
            [
                (image_color, adjusted_input),
                (adjusted_color, base_color),
                (adjusted_color, emission_color),
                (principled_shader, mix_principled),
                (emission_shader, mix_emission),
                (mixed_shader, surface),
            ],
        )
        self.assertEqual(contract["principled_mix_fraction"], 0.45)
        self.assertEqual(contract["emission_mix_fraction"], 0.55)
        self.assertEqual(
            contract["gltf_export"],
            "bake_balanced_graph_or_use_gltf_principled_mode",
        )

    def test_balanced_graph_rejects_gltf_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires blender_balanced"):
            connect_balanced_orthophoto_shader(
                _Node(),
                _Node(),
                _Node(),
                _Node(),
                _Node(),
                _Node(),
                _Links(),
                config=OrthophotoMaterialConfig(shader_mode="gltf_principled"),
            )


@dataclass
class _Coordinate:
    x: float
    y: float
    z: float

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def __len__(self) -> int:
        return 3


@dataclass
class _Vertex:
    co: _Coordinate


@dataclass
class _Loop:
    index: int
    vertex_index: int


@dataclass
class _UvValue:
    uv: tuple[float, float] = (math.nan, math.nan)


class _UvLayer:
    def __init__(self, name: str, loop_count: int) -> None:
        self.name = name
        self.data = [_UvValue() for _ in range(loop_count)]
        self.active_render = False


class _UvLayers:
    def __init__(self, loop_count: int) -> None:
        self._loop_count = loop_count
        self._layers: dict[str, _UvLayer] = {}
        self.active: _UvLayer | None = None

    def get(self, name: str) -> _UvLayer | None:
        return self._layers.get(name)

    def new(self, *, name: str) -> _UvLayer:
        layer = _UvLayer(name, self._loop_count)
        self._layers[name] = layer
        return layer


class _Mesh:
    def __init__(self) -> None:
        self.vertices = [
            _Vertex(_Coordinate(0.0, 0.0, 5.0)),
            _Vertex(_Coordinate(100.0, 0.0, 7.0)),
            _Vertex(_Coordinate(100.0, 100.0, 9.0)),
        ]
        self.loops = [_Loop(0, 0), _Loop(1, 1), _Loop(2, 2)]
        self.uv_layers = _UvLayers(len(self.loops))


class BlenderFreeUvLayerTests(unittest.TestCase):
    def test_mesh_loop_assignment_needs_no_bpy_and_reports_extent(self) -> None:
        mesh = _Mesh()

        statistics = assign_lambert93_uv_layer(
            mesh,
            (879_000.0, 6_398_000.0, 300.0),
            (879_000.0, 6_398_000.0, 879_100.0, 6_398_100.0),
        )

        layer = mesh.uv_layers.get("UVMap")
        assert layer is not None
        self.assertEqual(
            [entry.uv for entry in layer.data], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        )
        self.assertIs(mesh.uv_layers.active, layer)
        self.assertTrue(layer.active_render)
        self.assertEqual(statistics.loop_count, 3)
        self.assertEqual(
            (
                statistics.minimum_u,
                statistics.minimum_v,
                statistics.maximum_u,
                statistics.maximum_v,
            ),
            (0.0, 0.0, 1.0, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
