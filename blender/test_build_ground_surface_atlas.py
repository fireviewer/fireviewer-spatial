from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import build_ground_surface_atlas as atlas


def test_periodic_blend_produces_exact_matching_edges() -> None:
    rng = np.random.default_rng(42)
    values = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    output = atlas._periodic_uint8(values, seam_width=12)
    assert np.array_equal(output[:, 0], output[:, -1])
    assert np.array_equal(output[0], output[-1])
    assert atlas.seam_error(output) == 0


def test_builds_four_runtime_atlases_and_parameterized_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(atlas, "EXPECTED_MICRO_SOURCE_COUNT", 1)
    monkeypatch.setattr(atlas, "EXPECTED_PROFILE_COUNT", 1)
    monkeypatch.setattr(atlas, "EXPECTED_APPLICATION_MODES", {"ground_blend"})
    monkeypatch.setattr(atlas, "REQUIRED_INCIDENTS", {"FR-TEST"})
    source_root = tmp_path / "source-basecolor"
    source_root.mkdir()
    rows, columns = np.indices((96, 96))
    values = np.stack(
        (
            (rows * 3 + columns) % 256,
            (rows + columns * 2) % 256,
            (rows * 2 + columns * 3) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(values).save(source_root / "test-ground.png")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema": atlas.CONTRACT_SCHEMA,
                "status": "specified_not_accepted",
                "orthophoto_dependency": "forbidden",
                "scale_contract": {
                    "micro_detail_m": [1.0, 4.0],
                    "meso_variation_m": [16.0, 64.0],
                    "macro_variation_m": [128.0, 512.0],
                    "source_image_role": "offline_micro_detail_only",
                    "direct_source_image_import": "forbidden",
                    "macro_variation_source": "deterministic_procedural_noise",
                },
                "runtime_atlas": {
                    "size_px": 256,
                    "grid_size": 2,
                    "cell_size_px": 128,
                    "gutter_px": 8,
                    "runtime_textures": ["basecolor", "normal", "height", "orm"],
                    "runtime_texture_count": 4,
                },
                "micro_source_groups": [
                    {
                        "id": "test",
                        "physical_scale_m": 2.0,
                        "base_roughness": 0.8,
                        "normal_strength": 4.0,
                        "sources": ["test-ground.png"],
                    }
                ],
                "profile_families": [
                    {
                        "id": "test_family",
                        "application_mode": "ground_blend",
                        "shader": "layered_pbr",
                        "micro_source_groups": ["test"],
                        "variant_ids": ["first"],
                    }
                ],
                "composition": {"ground_blend": {}},
                "required_context_layers": ["test_context"],
                "determinism": {"silent_fallback": "forbidden"},
            }
        ),
        encoding="utf-8",
    )

    catalog = atlas.build_library(contract_path, source_root, tmp_path)
    atlas.validate_catalog(catalog, tmp_path)

    assert catalog["micro_source_count"] == 1
    assert catalog["profile_count"] == 1
    assert catalog["runtime_texture_count"] == 4
    assert set(catalog["runtime_atlas"]["assets"]) == {
        "basecolor",
        "normal",
        "height",
        "orm",
    }
    assert catalog["micro_source_runtime_import"] == "forbidden"
    assert catalog["profiles"][0]["parameters"]["macro_scale_m"] >= 128.0
    assert catalog["micro_sources"][0]["qa"]["maximum_edge_error"] == 0
