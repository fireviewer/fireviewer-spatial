from __future__ import annotations

import copy
import json
import math
import unittest

from tree_instances import (
    BUFFER_SCHEMA,
    TREE_INSTANCE_SCHEMA,
    TreeInstanceConfig,
    build_tree_instance_set,
    build_tree_prototype_library,
    decode_instance_attributes,
    decode_numeric_buffer,
)


def _record(
    index: int,
    *,
    tree_type: str | None = None,
    height: float = 12.0,
    diameter: float = 7.0,
) -> dict[str, object]:
    result: dict[str, object] = {
        "source_id": f"tree-{index}",
        "x_m": 880_000.0 + index * 0.25,
        "y_m": 6_400_000.0 + index * 0.125,
        "ground_elevation_m": 430.0 + (index % 7),
        "height_m": height,
        "crown_diameter_m": diameter,
    }
    if tree_type is not None:
        result["vegetation_type"] = tree_type
    return result


class PrototypeLibraryTests(unittest.TestCase):
    def test_library_contains_recognizable_multi_volume_families(self) -> None:
        config = TreeInstanceConfig(
            broadleaf_variant_count=3,
            conifer_variant_count=2,
        )
        prototypes = build_tree_prototype_library(config)

        self.assertEqual(len(prototypes), 5)
        self.assertEqual(
            [prototype["visual_form"] for prototype in prototypes],
            ["broadleaf", "broadleaf", "broadleaf", "conifer", "conifer"],
        )
        for prototype in prototypes:
            mesh = prototype["mesh"]
            self.assertGreaterEqual(prototype["crown_volume_count"], 4)
            self.assertEqual(len(mesh["faces"]), len(mesh["material_indices"]))
            self.assertEqual(len(mesh["faces"]), len(mesh["smooth_faces"]))
            self.assertIn(0, mesh["material_indices"])  # brown trunk slot
            self.assertTrue(any(index > 0 for index in mesh["material_indices"]))
            self.assertAlmostEqual(
                min(vertex[2] for vertex in mesh["vertices"]), 0.0, places=6
            )
            self.assertAlmostEqual(
                max(vertex[2] for vertex in mesh["vertices"]), 1.0, places=6
            )
            self.assertAlmostEqual(
                2.0
                * max(math.hypot(vertex[0], vertex[1]) for vertex in mesh["vertices"]),
                1.0,
                places=5,
            )

    def test_close_up_profile_increases_prototype_detail(self) -> None:
        mid = build_tree_prototype_library(
            TreeInstanceConfig(
                profile="global_mid",
                broadleaf_variant_count=1,
                conifer_variant_count=1,
            )
        )
        detail = build_tree_prototype_library(
            TreeInstanceConfig(
                profile="close_up",
                broadleaf_variant_count=1,
                conifer_variant_count=1,
            )
        )

        self.assertGreater(
            sum(item["mesh"]["estimated_triangle_count"] for item in detail),
            sum(item["mesh"]["estimated_triangle_count"] for item in mid),
        )


class InstanceSetTests(unittest.TestCase):
    def test_every_record_becomes_one_grounded_measured_instance(self) -> None:
        records = [
            _record(0, tree_type="feuillu", height=11.0, diameter=6.0),
            _record(1, tree_type="conifere", height=19.0, diameter=8.0),
            _record(2, height=7.5, diameter=4.0),
        ]
        origin = (879_000.0, 6_399_000.0, 400.0)
        config = TreeInstanceConfig(
            seed=42,
            broadleaf_variant_count=2,
            conifer_variant_count=2,
        )

        first = build_tree_instance_set(records, origin, config)
        second = build_tree_instance_set(copy.deepcopy(records), origin, config)
        decoded = decode_instance_attributes(first)

        self.assertEqual(first, second)
        self.assertEqual(first["schema"], TREE_INSTANCE_SCHEMA)
        self.assertEqual(first["statistics"]["input_record_count"], 3)
        self.assertEqual(first["statistics"]["instance_count"], 3)
        self.assertEqual(first["statistics"]["dropped_record_count"], 0)
        self.assertEqual(
            first["statistics"]["thinning"], "none_one_input_record_per_instance"
        )
        self.assertEqual(list(decoded["position_xyz_m"][:3]), [1000.0, 1000.0, 30.0])
        self.assertAlmostEqual(decoded["scale_xyz"][2], 11.0)
        self.assertAlmostEqual(decoded["scale_xyz"][5], 19.0)
        self.assertAlmostEqual(decoded["scale_xyz"][8], 7.5)
        for index, diameter in enumerate((6.0, 8.0, 4.0)):
            scale_x = decoded["scale_xyz"][index * 3]
            scale_y = decoded["scale_xyz"][index * 3 + 1]
            self.assertAlmostEqual(scale_x * scale_y, diameter * diameter, places=4)
        self.assertEqual(first["statistics"]["explicit_broadleaf_count"], 1)
        self.assertEqual(first["statistics"]["explicit_conifer_count"], 1)
        self.assertEqual(
            first["statistics"]["unknown_visual_broadleaf_count"]
            + first["statistics"]["unknown_visual_conifer_count"],
            1,
        )

    def test_buffers_are_compact_json_serializable_and_integrity_checked(self) -> None:
        result = build_tree_instance_set([_record(3)], (880_000.0, 6_400_000.0, 430.0))
        json.dumps(result, allow_nan=False)
        position_buffer = result["attributes"]["position_xyz_m"]
        self.assertEqual(position_buffer["schema"], BUFFER_SCHEMA)
        self.assertEqual(position_buffer["components"], 3)
        self.assertEqual(len(decode_numeric_buffer(position_buffer)), 3)

        corrupted = copy.deepcopy(position_buffer)
        corrupted["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            decode_numeric_buffer(corrupted)

    def test_global_candidate_count_is_supported_without_thinning(self) -> None:
        candidate_count = 226_893

        def records():
            for index in range(candidate_count):
                yield _record(index, height=5.0 + index % 22, diameter=2.0 + index % 8)

        result = build_tree_instance_set(
            records(),
            (879_000.0, 6_399_000.0, 400.0),
            TreeInstanceConfig(
                broadleaf_variant_count=3,
                conifer_variant_count=3,
            ),
        )
        decoded = decode_instance_attributes(result)

        self.assertEqual(result["statistics"]["input_record_count"], candidate_count)
        self.assertEqual(result["statistics"]["instance_count"], candidate_count)
        self.assertEqual(result["statistics"]["dropped_record_count"], 0)
        self.assertEqual(len(decoded["prototype_index"]), candidate_count)
        self.assertEqual(len(decoded["position_xyz_m"]), candidate_count * 3)
        self.assertFalse(result["statistics"]["realized_instance_geometry"])
        self.assertLess(
            len(result["attributes"]["position_xyz_m"]["data_base64"]),
            candidate_count * 17,
        )

    def test_invalid_measurement_is_not_silently_dropped(self) -> None:
        invalid = _record(1)
        invalid["height_m"] = 0.0
        with self.assertRaisesRegex(ValueError, "height_m.*strictly positive"):
            build_tree_instance_set([invalid], (0.0, 0.0, 0.0))

    def test_instance_references_are_range_checked(self) -> None:
        result = build_tree_instance_set([_record(1)], (0.0, 0.0, 0.0))
        result["prototypes"] = []
        with self.assertRaisesRegex(ValueError, "unknown prototype"):
            decode_instance_attributes(result)


class ConfigurationTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile"):
            TreeInstanceConfig(profile="hero")
        with self.assertRaisesRegex(ValueError, "prototype count"):
            TreeInstanceConfig(broadleaf_variant_count=200, conifer_variant_count=100)
        with self.assertRaisesRegex(ValueError, "unknown_visual_conifer_fraction"):
            TreeInstanceConfig(unknown_visual_conifer_fraction=1.1)


if __name__ == "__main__":
    unittest.main()
