from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import ground_material_contract as material


def _sealed_v2_contract(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    png = (
        material.PNG_SIGNATURE
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 4, 4, 8, 6, 0, 0, 0)
        + b"\0\0\0\0"
    )
    runtime_assets: dict[str, dict[str, object]] = {}
    for role in material.RUNTIME_TEXTURE_ROLES:
        path = root / "runtime-atlas" / f"{role}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        runtime_assets[role] = {
            **material._artifact(path, root),
            **material._png_header(path),
        }

    library_content_sha256 = "c" * 64
    layer = root / material.MATERIAL_LAYER_FILE_NAME
    layer.write_bytes(
        material._author_material_layer(
            {"runtime_atlas": {"assets": runtime_assets}},
            source_library_content_sha256=library_content_sha256,
        )
    )
    profile_textures = {
        role: {
            "byte_count": int(runtime_assets[role]["byte_count"]),
            "sha256": runtime_assets[role]["sha256"],
        }
        for role in material.RUNTIME_TEXTURE_ROLES
    }
    profiles = [
        {
            "index": index,
            "id": f"sealed.profile.{index:02d}",
            "surface_basis": "atlas_pbr",
            "atlas_slot": index,
            "atlas_uv": {"offset": [0.0, 0.0], "scale": [1.0, 1.0]},
            "physical_scale_m": 4.0,
            "projection": "world_xy",
            "textures": profile_textures,
            "variant_selection": "baked_profile_id",
            "runtime_modulation": "none",
        }
        for index in range(material.EXPECTED_PROFILE_COUNT)
    ]
    contract = {
        "schema": material.CONTRACT_SCHEMA,
        "material_model": material.MATERIAL_LAYER_SCHEMA,
        "crs": material.CRS,
        "orthophoto_dependency": "forbidden",
        "source_library": {
            "schema": material.CLEAN_LIBRARY_SCHEMA,
            "manifest_sha256": "a" * 64,
            "identity_sha256": "b" * 64,
            "content_sha256": library_content_sha256,
            "texture_contract_sha256": "d" * 64,
            "status": "generated_pending_visual_review",
        },
        "visual_acceptance": "pending_human_review",
        "material_layer": material._artifact(layer, root),
        "runtime_atlas": {
            "width_px": 4,
            "height_px": 4,
            "assets": runtime_assets,
        },
        "profile_count": material.EXPECTED_PROFILE_COUNT,
        "profile_table": profiles,
        "composition": {
            "grid_size_px": [500, 500],
            "cell_size_m": 1,
            "runtime_procedural_material": "forbidden",
            "runtime_orthophoto": "forbidden",
            "variant_selection": "baked into the 72 profile IDs before packaging",
            "surface_overlays": "not_packaged",
        },
        "runtime_shader": {
            "schema": material.RUNTIME_SHADER_SCHEMA,
            "status": material.RUNTIME_SHADER_PENDING_STATUS,
            "implementation": None,
            "source_artifact": None,
            "production_textured_runtime_qualified": False,
            "preview_surface_policy": "diagnostic_untextured_only",
            "required_capabilities": [
                "four_profile_id_indirections",
                "rgba8_weighted_pbr_blend",
                "epsg2154_world_projection",
                "undirected_orientation_0_to_pi",
                "world_xy_and_world_triplanar",
            ],
        },
    }
    contract_path = root / material.CONTRACT_FILE_NAME
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract_path


def test_sealed_contract_validation_imports_without_pillow(tmp_path: Path) -> None:
    contract_path = _sealed_v2_contract(tmp_path)
    expected_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    blender_module_root = Path(material.__file__).resolve().parent
    script = r"""
import importlib.abc
from pathlib import Path
import sys

class NoPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PIL" or fullname.startswith("PIL."):
            raise ModuleNotFoundError("Pillow is unavailable in Blender", name=fullname)
        return None

sys.meta_path.insert(0, NoPillow())
sys.path.insert(0, sys.argv[1])
import ground_material_contract

assert "clean_pbr_texture_library" not in sys.modules
result = ground_material_contract.validate_ground_material_contract(Path(sys.argv[2]))
assert result["schema"] == ground_material_contract.CONTRACT_SCHEMA
print(ground_material_contract.sha256_file(Path(sys.argv[2])))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            script,
            str(blender_module_root),
            str(contract_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected_sha256
