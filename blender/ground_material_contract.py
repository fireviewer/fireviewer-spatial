"""Shared, hash-locked ground material bundle for adaptive terrain zones.

The four runtime atlases are zone assets, never tile assets.  A tile only
records the immutable material contract identity and references the shared USD
material layer.  The Blender QA script is the reference implementation of the
sampling contract.  Ground-material v2 deliberately remains fail-closed for
production USD rendering until a hash-locked dedicated MDL implementation has
been executed and accepted in Omniverse; its PreviewSurface is diagnostic only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
from typing import Any, Mapping

ATLAS_SCHEMA = "fireviewer.ground-surface-atlas-library.v3"
CLEAN_LIBRARY_SCHEMA = "fireviewer.clean-pbr-texture-library.v1"
LEGACY_CONTRACT_SCHEMA = "fireviewer.ground-material-contract.v1"
CONTRACT_SCHEMA = "fireviewer.ground-material-contract.v2"
LEGACY_MATERIAL_LAYER_SCHEMA = "FireViewerGroundSurface_v1"
MATERIAL_LAYER_SCHEMA = "FireViewerGroundSurface_v2"
RUNTIME_SHADER_SCHEMA = "fireviewer.ground-runtime-shader-binding.v1"
RUNTIME_SHADER_PENDING_STATUS = "pending_dedicated_mdl_validation"
LEGACY_CONTRACT_FILE_NAME = "ground-material-contract.v1.json"
CONTRACT_FILE_NAME = "ground-material-contract.v2.json"
MATERIAL_LAYER_FILE_NAME = "ground-material.usda"
ATLAS_CATALOG_FILE_NAME = "ground-surface-atlas-catalog.json"
RUNTIME_TEXTURE_ROLES = ("basecolor", "normal", "height", "orm")
EXPECTED_PROFILE_COUNT = 72
CRS = "EPSG:2154"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ACCEPTED_ATLAS_CATALOG_STATUSES = frozenset(
    {"accepted_blender_visual", "accepted_human_visual"}
)


class GroundMaterialContractError(ValueError):
    """A shared ground material bundle is incomplete or not reproducible."""


def _validate_clean_texture_library(
    library_path: Path,
    *,
    require_visual_acceptance: bool,
) -> dict[str, Any]:
    """Load the Pillow-backed library builder only while authoring a bundle."""

    try:
        from clean_pbr_texture_library import validate_texture_library
    except ModuleNotFoundError as error:
        if error.name != "clean_pbr_texture_library":
            raise
        try:
            from blender.clean_pbr_texture_library import validate_texture_library
        except ModuleNotFoundError as package_error:  # pragma: no cover - import mode
            if package_error.name != "blender":
                raise
            raise error from package_error
    return validate_texture_library(
        library_path,
        require_visual_acceptance=require_visual_acceptance,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _portable_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise GroundMaterialContractError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise GroundMaterialContractError(f"{label} must use portable forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0].endswith(":"):
        raise GroundMaterialContractError(f"{label} escapes its bundle")
    return path


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise GroundMaterialContractError(
            f"Shared artifact escapes bundle: {path}"
        ) from error
    return {
        "path": relative,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _png_header(path: Path) -> dict[str, int]:
    header = path.read_bytes()[:33]
    if len(header) < 33 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise GroundMaterialContractError(f"Runtime atlas is not a PNG: {path}")
    width, height, bit_depth, colour_type = struct.unpack(">IIBB", header[16:26])
    if width <= 0 or height <= 0:
        raise GroundMaterialContractError(
            f"Runtime atlas has invalid dimensions: {path}"
        )
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "colour_type": colour_type,
    }


def _load_catalog(catalog_path: Path) -> dict[str, Any]:
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GroundMaterialContractError(f"Invalid atlas catalog: {error}") from error
    if not isinstance(catalog, dict) or catalog.get("schema") != ATLAS_SCHEMA:
        raise GroundMaterialContractError("Unsupported ground atlas catalog")
    unsigned = dict(catalog)
    declared_catalog_hash = unsigned.pop("catalog_sha256", None)
    if declared_catalog_hash != canonical_sha256(unsigned):
        raise GroundMaterialContractError("Ground atlas catalog hash mismatch")
    profiles = catalog.get("profiles")
    if (
        not isinstance(profiles, list)
        or len(profiles) != EXPECTED_PROFILE_COUNT
        or catalog.get("profile_count") != EXPECTED_PROFILE_COUNT
    ):
        raise GroundMaterialContractError(
            "Ground atlas must contain exactly 72 profiles"
        )
    identifiers = [
        profile.get("id") for profile in profiles if isinstance(profile, dict)
    ]
    if len(identifiers) != EXPECTED_PROFILE_COUNT or any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise GroundMaterialContractError(
            "Ground atlas profile identifiers are invalid"
        )
    if len(set(identifiers)) != len(identifiers):
        raise GroundMaterialContractError(
            "Ground atlas profile identifiers are not unique"
        )
    runtime = catalog.get("runtime_atlas")
    assets = runtime.get("assets") if isinstance(runtime, dict) else None
    if (
        not isinstance(assets, dict)
        or set(assets) != set(RUNTIME_TEXTURE_ROLES)
        or catalog.get("runtime_texture_count") != 4
    ):
        raise GroundMaterialContractError(
            "Ground atlas must declare four runtime textures"
        )
    if catalog.get("orthophoto_dependency") != "forbidden":
        raise GroundMaterialContractError("Ground atlas must forbid orthophotos")
    for role, record in assets.items():
        if (
            not isinstance(record, Mapping)
            or "ortho" in str(record.get("path", "")).casefold()
        ):
            raise GroundMaterialContractError(
                f"Ground atlas runtime asset is invalid: {role}"
            )
    return catalog


def _profile_table(
    catalog: Mapping[str, Any], *, require_clean_pbr: bool
) -> list[dict[str, Any]]:
    sources = {
        source.get("id"): source
        for source in catalog.get("micro_sources", [])
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    }
    table: list[dict[str, Any]] = []
    for index, profile in enumerate(
        sorted(catalog["profiles"], key=lambda value: str(value["id"]))
    ):
        source_id = profile.get("micro_source_id")
        source = sources.get(source_id)
        surface_basis = str(profile.get("surface_basis", ""))
        if source is None and surface_basis not in {"procedural_only"}:
            raise GroundMaterialContractError(
                f"Atlas-backed profile has no micro source: {profile['id']}"
            )
        if require_clean_pbr:
            if surface_basis != "atlas_pbr" or source is None:
                raise GroundMaterialContractError(
                    f"Ground profile must use a clean PBR atlas source: {profile['id']}"
                )
            atlas_uv = source.get("atlas_uv")
            if not isinstance(atlas_uv, Mapping):
                raise GroundMaterialContractError(
                    f"Ground profile has no atlas UV rectangle: {profile['id']}"
                )
            try:
                offset = [float(value) for value in atlas_uv["offset"]]
                scale = [float(value) for value in atlas_uv["scale"]]
                physical_scale_m = float(source["physical_scale_m"])
            except (KeyError, TypeError, ValueError) as error:
                raise GroundMaterialContractError(
                    f"Ground profile has invalid metric atlas sampling: {profile['id']}"
                ) from error
            if (
                len(offset) != 2
                or len(scale) != 2
                or any(value < 0.0 or value > 1.0 for value in offset)
                or any(value <= 0.0 or value > 1.0 for value in scale)
                or any(offset[index] + scale[index] > 1.0 for index in range(2))
                or physical_scale_m <= 0.0
            ):
                raise GroundMaterialContractError(
                    f"Ground profile metric atlas sampling is out of range: {profile['id']}"
                )
            parameters = profile.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise GroundMaterialContractError(
                    f"Ground profile parameters are invalid: {profile['id']}"
                )
            prohibited = {
                str(key)
                for key in parameters
                if "noise" in str(key).lower() or "tint" in str(key).lower()
            }
            if prohibited:
                raise GroundMaterialContractError(
                    f"Ground profile declares procedural material controls: {profile['id']}"
                )
        table.append(
            {
                "index": index,
                "id": profile["id"],
                "surface_basis": surface_basis,
                "atlas_slot": source.get("slot") if source is not None else None,
                "atlas_uv": source.get("atlas_uv") if source is not None else None,
                "physical_scale_m": (
                    float(source.get("physical_scale_m", 4.0))
                    if source is not None
                    else float(profile.get("parameters", {}).get("meso_scale_m", 32.0))
                ),
                **(
                    {
                        "variant_selection": "baked_profile_id",
                        "runtime_modulation": "none",
                    }
                    if require_clean_pbr
                    else {"parameters": dict(profile.get("parameters", {}))}
                ),
            }
        )
    return table


def _clean_profile_table(library: Mapping[str, Any]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    profiles = library.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != EXPECTED_PROFILE_COUNT:
        raise GroundMaterialContractError("Clean PBR library must contain 72 profiles")
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            raise GroundMaterialContractError("Clean PBR profile is invalid")
        textures = profile.get("textures")
        if not isinstance(textures, Mapping) or set(textures) != set(
            RUNTIME_TEXTURE_ROLES
        ):
            raise GroundMaterialContractError(
                f"Clean PBR profile texture set is incomplete: {index}"
            )
        table.append(
            {
                "index": index,
                "id": profile["id"],
                "surface_basis": "atlas_pbr",
                "atlas_slot": profile["atlas_slot"],
                "atlas_uv": profile["atlas_uv"],
                "physical_scale_m": float(profile["physical_scale_m"]),
                "projection": profile["projection"],
                "textures": {
                    role: {
                        "byte_count": int(textures[role]["byte_count"]),
                        "sha256": textures[role]["sha256"],
                    }
                    for role in RUNTIME_TEXTURE_ROLES
                },
                "variant_selection": "baked_profile_id",
                "runtime_modulation": "none",
            }
        )
    return table


def _author_material_layer(
    catalog: Mapping[str, Any],
    *,
    source_library_content_sha256: str,
    legacy_v1: bool = False,
) -> bytes:
    material_schema = (
        LEGACY_MATERIAL_LAYER_SCHEMA if legacy_v1 else MATERIAL_LAYER_SCHEMA
    )
    contract_file_name = LEGACY_CONTRACT_FILE_NAME if legacy_v1 else CONTRACT_FILE_NAME
    composition_grid_size = 100 if legacy_v1 else 500
    composition_cell_size_m = 5 if legacy_v1 else 1
    assets = catalog["runtime_atlas"]["assets"]
    asset_lines = "\n".join(
        f"        custom asset fireviewer:atlas_{role} = @runtime-atlas/{role}.png@"
        for role in RUNTIME_TEXTURE_ROLES
    )
    mapping_declarations = (
        "        asset inputs:surfaceOverlays"
        if legacy_v1
        else """        asset inputs:groundConfidence
        asset inputs:groundOrientation"""
    )
    mapping_connections = (
        "            asset inputs:surfaceOverlays.connect = </FireViewerMaterials/GroundMaterial.inputs:surfaceOverlays>"
        if legacy_v1
        else """            asset inputs:groundConfidence.connect = </FireViewerMaterials/GroundMaterial.inputs:groundConfidence>
            asset inputs:groundOrientation.connect = </FireViewerMaterials/GroundMaterial.inputs:groundOrientation>"""
    )
    runtime_shader_metadata = (
        ""
        if legacy_v1
        else f'''        custom string fireviewer:runtime_shader_schema = "{RUNTIME_SHADER_SCHEMA}"
        custom string fireviewer:runtime_shader_status = "{RUNTIME_SHADER_PENDING_STATUS}"
        custom bool fireviewer:production_textured_runtime_qualified = false
        custom string fireviewer:preview_surface_policy = "diagnostic_untextured_only"'''
    )
    custom_surface_output = (
        "        token outputs:fireviewer:surface.connect = </FireViewerMaterials/GroundMaterial/GroundSurface.outputs:surface>"
        if legacy_v1
        else ""
    )
    ground_surface_shader = (
        f'''
        def Shader "GroundSurface"
        {{
            uniform token info:id = "{material_schema}"
            asset inputs:atlasBasecolor.connect = </FireViewerMaterials/GroundMaterial.inputs:atlasBasecolor>
            asset inputs:atlasNormal.connect = </FireViewerMaterials/GroundMaterial.inputs:atlasNormal>
            asset inputs:atlasHeight.connect = </FireViewerMaterials/GroundMaterial.inputs:atlasHeight>
            asset inputs:atlasOrm.connect = </FireViewerMaterials/GroundMaterial.inputs:atlasOrm>
            asset inputs:profileTable.connect = </FireViewerMaterials/GroundMaterial.inputs:profileTable>
            asset inputs:groundProfileIds.connect = </FireViewerMaterials/GroundMaterial.inputs:groundProfileIds>
            asset inputs:groundProfileWeights.connect = </FireViewerMaterials/GroundMaterial.inputs:groundProfileWeights>
{mapping_connections}
            double2 inputs:tileOriginL93M.connect = </FireViewerMaterials/GroundMaterial.inputs:tileOriginL93M>
            int inputs:compositionGridSize.connect = </FireViewerMaterials/GroundMaterial.inputs:compositionGridSize>
            double inputs:compositionCellSizeM.connect = </FireViewerMaterials/GroundMaterial.inputs:compositionCellSizeM>
            token outputs:surface
        }}
'''
        if legacy_v1
        else ""
    )
    preview_color = "(0.18, 0.32, 0.12)" if legacy_v1 else "(1, 0, 1)"
    text = f'''#usda 1.0
(
    defaultPrim = "FireViewerMaterials"
    metersPerUnit = 1
    upAxis = "Z"
)

def Scope "FireViewerMaterials"
{{
    def Material "GroundMaterial"
    {{
        custom string fireviewer:shader_contract = "{material_schema}"
        custom string fireviewer:crs = "{CRS}"
        custom string fireviewer:source_library_content_sha256 = "{source_library_content_sha256}"
{runtime_shader_metadata}
{asset_lines}
        asset inputs:atlasBasecolor = @runtime-atlas/basecolor.png@
        asset inputs:atlasNormal = @runtime-atlas/normal.png@
        asset inputs:atlasHeight = @runtime-atlas/height.png@
        asset inputs:atlasOrm = @runtime-atlas/orm.png@
        asset inputs:profileTable = @{contract_file_name}@
        asset inputs:groundProfileIds
        asset inputs:groundProfileWeights
{mapping_declarations}
        double2 inputs:tileOriginL93M = (0, 0)
        int inputs:compositionGridSize = {composition_grid_size}
        double inputs:compositionCellSizeM = {composition_cell_size_m}
        token outputs:surface.connect = </FireViewerMaterials/GroundMaterial/PreviewSurface.outputs:surface>
{custom_surface_output}
{ground_surface_shader}

        def Shader "PreviewSurface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = {preview_color}
            float inputs:metallic = 0
            float inputs:roughness = 0.9
            token outputs:surface
        }}
    }}
}}
'''
    # Referencing the fields ensures malformed asset records are rejected before
    # a shared layer is published, even though paths are normalized on copy.
    for role in RUNTIME_TEXTURE_ROLES:
        if not isinstance(assets.get(role), dict):
            raise GroundMaterialContractError(f"Missing atlas asset: {role}")
    return text.encode("utf-8")


def build_ground_material_bundle(
    atlas_catalog_path: Path,
    output_root: Path,
    *,
    legacy_v1: bool = False,
) -> Path:
    """Copy one atlas revision once and author its immutable material contract."""

    source_catalog_path = Path(atlas_catalog_path).resolve()
    source_root = source_catalog_path.parent
    target_root = Path(output_root).resolve()
    try:
        source_payload = json.loads(source_catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GroundMaterialContractError(
            f"Invalid source PBR library: {error}"
        ) from error
    if not isinstance(source_payload, dict):
        raise GroundMaterialContractError("Source PBR library must be a JSON object")

    clean_library = source_payload.get("schema") == CLEAN_LIBRARY_SCHEMA
    if legacy_v1:
        if clean_library:
            raise GroundMaterialContractError(
                "A clean PBR library cannot be downgraded to material v1"
            )
        catalog = _load_catalog(source_catalog_path)
        profile_table = _profile_table(catalog, require_clean_pbr=False)
        runtime_records = catalog["runtime_atlas"]["assets"]
        source_library = {
            "schema": ATLAS_SCHEMA,
            "manifest_sha256": sha256_file(source_catalog_path),
            "identity_sha256": canonical_sha256(catalog),
            "content_sha256": catalog["catalog_sha256"],
            "texture_contract_sha256": None,
            "status": catalog.get("status", "unspecified"),
        }
        visual_acceptance = "pending_human_review"
    elif clean_library:
        try:
            validation = _validate_clean_texture_library(
                source_catalog_path, require_visual_acceptance=False
            )
        except ValueError as error:
            raise GroundMaterialContractError(str(error)) from error
        catalog = source_payload
        profile_table = _clean_profile_table(catalog)
        runtime_records = catalog["runtime_atlases"]
        source_library = {
            "schema": CLEAN_LIBRARY_SCHEMA,
            "manifest_sha256": sha256_file(source_catalog_path),
            "identity_sha256": canonical_sha256(catalog),
            "content_sha256": validation["library_content_sha256"],
            "texture_contract_sha256": validation["texture_contract_sha256"],
            "status": validation["status"],
        }
        visual_acceptance = (
            "accepted_human_visual"
            if validation["visual_acceptance"] == "accepted_human_visual"
            else "pending_human_review"
        )
    else:
        raise GroundMaterialContractError(
            "Material v2 requires fireviewer.clean-pbr-texture-library.v1; "
            "legacy atlas libraries are read-only"
        )
    target_root.mkdir(parents=True, exist_ok=True)

    copied_assets: dict[str, dict[str, Any]] = {}
    for role in RUNTIME_TEXTURE_ROLES:
        source_record = runtime_records[role]
        relative = _portable_relative(
            source_record.get("path"), f"runtime_atlas.{role}"
        )
        source_path = source_root.joinpath(*relative.parts)
        if not source_path.is_file():
            raise GroundMaterialContractError(
                f"Runtime atlas is missing: {source_path}"
            )
        if source_record.get("sha256") != sha256_file(source_path):
            raise GroundMaterialContractError(f"Runtime atlas hash mismatch: {role}")
        png = _png_header(source_path)
        if not clean_library and (
            source_record.get("width") != png["width"]
            or source_record.get("height") != png["height"]
        ):
            raise GroundMaterialContractError(
                f"Runtime atlas dimensions mismatch: {role}"
            )
        target_path = target_root / "runtime-atlas" / f"{role}.png"
        if source_path != target_path:
            temporary = target_path.with_name(f".{target_path.name}.tmp")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, temporary)
            temporary.replace(target_path)
        copied_assets[role] = {**_artifact(target_path, target_root), **png}

    catalog_target: Path | None = None
    if legacy_v1:
        copied_catalog = dict(catalog)
        copied_runtime = dict(copied_catalog["runtime_atlas"])
        copied_runtime["assets"] = {
            role: {
                **dict(catalog["runtime_atlas"]["assets"][role]),
                "path": f"runtime-atlas/{role}.png",
                "byte_count": copied_assets[role]["byte_count"],
                "sha256": copied_assets[role]["sha256"],
            }
            for role in RUNTIME_TEXTURE_ROLES
        }
        copied_catalog["runtime_atlas"] = copied_runtime
        unsigned_catalog = dict(copied_catalog)
        unsigned_catalog.pop("catalog_sha256", None)
        copied_catalog["catalog_sha256"] = canonical_sha256(unsigned_catalog)
        catalog_target = target_root / ATLAS_CATALOG_FILE_NAME
        _atomic_write(catalog_target, canonical_json(copied_catalog))
        profile_table = _profile_table(copied_catalog, require_clean_pbr=False)

    material_layer_path = target_root / MATERIAL_LAYER_FILE_NAME
    _atomic_write(
        material_layer_path,
        _author_material_layer(
            {"runtime_atlas": {"assets": copied_assets}},
            source_library_content_sha256=source_library["content_sha256"],
            legacy_v1=legacy_v1,
        ),
    )
    contract_schema = LEGACY_CONTRACT_SCHEMA if legacy_v1 else CONTRACT_SCHEMA
    material_schema = (
        LEGACY_MATERIAL_LAYER_SCHEMA if legacy_v1 else MATERIAL_LAYER_SCHEMA
    )
    contract: dict[str, Any] = {
        "schema": contract_schema,
        "material_model": material_schema,
        "crs": CRS,
        "orthophoto_dependency": "forbidden",
        "source_library": source_library,
        "visual_acceptance": visual_acceptance,
        "material_layer": _artifact(material_layer_path, target_root),
        "runtime_atlas": (
            {
                "size_px": int(catalog["runtime_atlas"]["size_px"]),
                "grid_size": int(catalog["runtime_atlas"]["grid_size"]),
                "cell_size_px": int(catalog["runtime_atlas"]["cell_size_px"]),
                "gutter_px": int(catalog["runtime_atlas"]["gutter_px"]),
                "assets": copied_assets,
            }
            if legacy_v1
            else {
                "width_px": copied_assets["basecolor"]["width"],
                "height_px": copied_assets["basecolor"]["height"],
                "assets": copied_assets,
            }
        ),
        "profile_count": EXPECTED_PROFILE_COUNT,
        "profile_table": profile_table,
        "composition": (
            {
                "ids": "ground-profile-ids.png RGBA8 zero-based profile indices",
                "weights": "ground-profile-weights.png RGBA8 sum exactly 255",
                "grid_size_px": [100, 100],
                "cell_size_m": 5,
                "row_order": "north_to_south",
                "world_projection": "EPSG:2154 metric XY; phase never resets per tile",
            }
            if legacy_v1
            else {
                "ids": "ground-profile-ids.png RGBA8 zero-based profile indices",
                "weights": "ground-profile-weights.png RGBA8 sum exactly 255",
                "confidence": "ground-confidence.png L8 matcher confidence 0..255",
                "orientation": "ground-orientation.png L8 0..255 maps to 0..pi radians (undirected axis)",
                "grid_size_px": [500, 500],
                "cell_size_m": 1,
                "row_order": "north_to_south",
                "world_projection": "EPSG:2154 metric XY; phase never resets per tile",
                "variant_selection": "baked into the 72 profile IDs before packaging",
                "runtime_procedural_material": "forbidden",
                "runtime_orthophoto": "forbidden",
                "surface_overlays": "not_packaged",
            }
        ),
        "qa_shader": {
            "implementation": f"{material_schema} Blender 4.5 reference",
            "profile_sampling": "four zero-based IDs weighted by RGBA8 values summing to 255",
            "world_coordinates": "EPSG:2154 metric XY with no tile-local phase reset",
            "orientation": (
                "none"
                if legacy_v1
                else "L8 angle rotates metric atlas projection before sampling"
            ),
            "confidence": (
                "not_available"
                if legacy_v1
                else "QA evidence only; never invokes a procedural fallback"
            ),
            "basecolor": "decode sRGB, linearly blend four atlas samples, then multiply AO",
            "normal": "linearly blend and renormalize OpenGL tangent normals",
            "height": "linearly blend the 16-bit height proxy and feed bump height",
            "orm": "R=AO, G=roughness, B=metallic",
            "macro_tint": "forbidden" if not legacy_v1 else "legacy_unspecified",
            "procedural_noise": "forbidden" if not legacy_v1 else "legacy_unspecified",
        },
    }
    if not legacy_v1:
        contract["runtime_shader"] = {
            "schema": RUNTIME_SHADER_SCHEMA,
            "status": RUNTIME_SHADER_PENDING_STATUS,
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
        }
    if legacy_v1:
        if catalog_target is None:  # pragma: no cover - defensive
            raise AssertionError("Legacy atlas target is absent")
        contract.update(
            {
                "source_atlas_catalog_sha256": source_library["manifest_sha256"],
                "atlas_catalog_status": source_library["status"],
                "atlas_catalog": _artifact(catalog_target, target_root),
            }
        )
    contract_path = target_root / (
        LEGACY_CONTRACT_FILE_NAME if legacy_v1 else CONTRACT_FILE_NAME
    )
    _atomic_write(contract_path, canonical_json(contract))
    validate_ground_material_contract(contract_path)
    return contract_path


def _validate_artifact(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, dict) or set(record) < {"path", "byte_count", "sha256"}:
        raise GroundMaterialContractError(f"Invalid artifact record: {label}")
    relative = _portable_relative(record.get("path"), label)
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise GroundMaterialContractError(
            f"Shared material artifact is missing: {label}"
        )
    if record.get("byte_count") != path.stat().st_size or record.get(
        "sha256"
    ) != sha256_file(path):
        raise GroundMaterialContractError(
            f"Shared material artifact hash mismatch: {label}"
        )
    return path


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_v2_profile_table(
    value: Any, *, clean_library: bool
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_PROFILE_COUNT:
        raise GroundMaterialContractError("Ground material profile table is incomplete")
    identifiers: set[str] = set()
    for index, profile in enumerate(value):
        if not isinstance(profile, dict) or profile.get("index") != index:
            raise GroundMaterialContractError("Ground profile indices are not stable")
        identifier = profile.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise GroundMaterialContractError("Ground profile identifiers are invalid")
        identifiers.add(identifier)
        if (
            profile.get("surface_basis") != "atlas_pbr"
            or profile.get("atlas_slot") is None
            or profile.get("variant_selection") != "baked_profile_id"
            or profile.get("runtime_modulation") != "none"
        ):
            raise GroundMaterialContractError(
                f"Ground profile is not clean atlas PBR: {identifier}"
            )
        atlas_uv = profile.get("atlas_uv")
        try:
            offset = [float(item) for item in atlas_uv["offset"]]
            scale = [float(item) for item in atlas_uv["scale"]]
            physical_scale = float(profile["physical_scale_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise GroundMaterialContractError(
                f"Ground profile sampling is invalid: {identifier}"
            ) from error
        if (
            len(offset) != 2
            or len(scale) != 2
            or any(item < 0.0 or item > 1.0 for item in offset)
            or any(item <= 0.0 or item > 1.0 for item in scale)
            or any(offset[axis] + scale[axis] > 1.0 for axis in range(2))
            or physical_scale <= 0.0
        ):
            raise GroundMaterialContractError(
                f"Ground profile sampling is out of range: {identifier}"
            )
        if clean_library:
            if profile.get("projection") not in {"world_xy", "world_triplanar"}:
                raise GroundMaterialContractError(
                    f"Ground profile projection is invalid: {identifier}"
                )
            textures = profile.get("textures")
            if not isinstance(textures, Mapping) or set(textures) != set(
                RUNTIME_TEXTURE_ROLES
            ):
                raise GroundMaterialContractError(
                    f"Ground profile texture identity is incomplete: {identifier}"
                )
            for role, artifact in textures.items():
                if (
                    not isinstance(artifact, Mapping)
                    or not _valid_sha256(artifact.get("sha256"))
                    or not isinstance(artifact.get("byte_count"), int)
                    or artifact["byte_count"] <= 0
                ):
                    raise GroundMaterialContractError(
                        f"Ground profile texture identity is invalid: {identifier}/{role}"
                    )
    return value


def validate_ground_material_contract(contract_path: Path) -> dict[str, Any]:
    """Validate every shared byte and return the parsed material contract."""

    path = Path(contract_path).resolve()
    root = path.parent
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GroundMaterialContractError(
            f"Invalid ground material contract: {error}"
        ) from error
    if not isinstance(contract, dict) or contract.get("schema") not in {
        LEGACY_CONTRACT_SCHEMA,
        CONTRACT_SCHEMA,
    }:
        raise GroundMaterialContractError("Unsupported ground material contract")
    legacy_v1 = contract["schema"] == LEGACY_CONTRACT_SCHEMA
    expected_material_model = (
        LEGACY_MATERIAL_LAYER_SCHEMA if legacy_v1 else MATERIAL_LAYER_SCHEMA
    )
    if contract.get("material_model") != expected_material_model:
        raise GroundMaterialContractError("Ground material model differs from schema")
    if (
        contract.get("crs") != CRS
        or contract.get("orthophoto_dependency") != "forbidden"
    ):
        raise GroundMaterialContractError(
            "Ground material coordinate/dependency contract is invalid"
        )
    source_library = contract.get("source_library")
    if legacy_v1 and source_library is None:
        # Contracts published before v2 carried these values as separate
        # atlas fields.  Reconstruct the read-only identity in memory.
        source_library = {
            "schema": ATLAS_SCHEMA,
            "manifest_sha256": contract.get("source_atlas_catalog_sha256"),
            "identity_sha256": None,
            "content_sha256": None,
            "texture_contract_sha256": None,
            "status": contract.get("atlas_catalog_status"),
        }
    if (
        not isinstance(source_library, Mapping)
        or set(source_library)
        != {
            "schema",
            "manifest_sha256",
            "identity_sha256",
            "content_sha256",
            "texture_contract_sha256",
            "status",
        }
        or not _valid_sha256(source_library.get("manifest_sha256"))
        or (not legacy_v1 and not _valid_sha256(source_library.get("identity_sha256")))
        or (not legacy_v1 and not _valid_sha256(source_library.get("content_sha256")))
        or (
            source_library.get("texture_contract_sha256") is not None
            and not _valid_sha256(source_library.get("texture_contract_sha256"))
        )
    ):
        raise GroundMaterialContractError("Ground material source library is invalid")
    if not legacy_v1 and (
        source_library.get("schema") != CLEAN_LIBRARY_SCHEMA
        or not _valid_sha256(source_library.get("texture_contract_sha256"))
        or source_library.get("status")
        not in {"generated_pending_visual_review", "accepted_clean_pbr_library"}
    ):
        raise GroundMaterialContractError(
            "Ground material v2 requires the hash-locked clean PBR library"
        )
    if (
        contract.get("profile_count") != EXPECTED_PROFILE_COUNT
        or len(contract.get("profile_table", [])) != EXPECTED_PROFILE_COUNT
    ):
        raise GroundMaterialContractError("Ground material profile table is incomplete")
    material_path = _validate_artifact(
        root, contract.get("material_layer"), "material_layer"
    )
    visual_acceptance = contract.get("visual_acceptance")
    if visual_acceptance not in {"pending_human_review", "accepted_human_visual"}:
        raise GroundMaterialContractError("Ground material visual status is invalid")
    if (
        not legacy_v1
        and visual_acceptance == "accepted_human_visual"
        and (
            source_library.get("schema") != CLEAN_LIBRARY_SCHEMA
            or source_library.get("status") != "accepted_clean_pbr_library"
        )
    ):
        raise GroundMaterialContractError(
            "A pending PBR library cannot be promoted to visual acceptance"
        )
    if not legacy_v1:
        runtime_shader = contract.get("runtime_shader")
        if runtime_shader != {
            "schema": RUNTIME_SHADER_SCHEMA,
            "status": RUNTIME_SHADER_PENDING_STATUS,
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
        }:
            raise GroundMaterialContractError(
                "Ground material v2 must remain fail-closed until its dedicated "
                "runtime shader is validated"
            )
    if legacy_v1:
        catalog_path = _validate_artifact(
            root, contract.get("atlas_catalog"), "atlas_catalog"
        )
        catalog = _load_catalog(catalog_path)
        if source_library["content_sha256"] is None:
            source_library = {
                **source_library,
                "identity_sha256": canonical_sha256(catalog),
                "content_sha256": catalog["catalog_sha256"],
            }
        if contract.get("source_atlas_catalog_sha256") != source_library[
            "manifest_sha256"
        ] or contract.get("atlas_catalog_status") != catalog.get(
            "status", "unspecified"
        ):
            raise GroundMaterialContractError("Ground material catalog status changed")
        if (
            visual_acceptance == "accepted_human_visual"
            and catalog.get("status") not in ACCEPTED_ATLAS_CATALOG_STATUSES
        ):
            raise GroundMaterialContractError(
                "A pending atlas catalog cannot be promoted to visual acceptance"
            )
        expected_profile_table = _profile_table(catalog, require_clean_pbr=False)
        if contract.get("profile_table") != expected_profile_table:
            raise GroundMaterialContractError(
                "Ground material profile table differs from the atlas catalog"
            )
    else:
        _validate_v2_profile_table(
            contract.get("profile_table"),
            clean_library=True,
        )
    composition = contract.get("composition")
    if not isinstance(composition, Mapping):
        raise GroundMaterialContractError("Ground material composition is missing")
    if legacy_v1:
        if (
            composition.get("grid_size_px") != [100, 100]
            or composition.get("cell_size_m") != 5
        ):
            raise GroundMaterialContractError(
                "Legacy ground material composition changed"
            )
    elif any(
        (
            composition.get("grid_size_px") != [500, 500],
            composition.get("cell_size_m") != 1,
            composition.get("runtime_procedural_material") != "forbidden",
            composition.get("runtime_orthophoto") != "forbidden",
            composition.get("variant_selection")
            != "baked into the 72 profile IDs before packaging",
            composition.get("surface_overlays") != "not_packaged",
        )
    ):
        raise GroundMaterialContractError("Ground material v2 composition is invalid")
    runtime = contract.get("runtime_atlas")
    if not isinstance(runtime, dict) or set(runtime.get("assets", {})) != set(
        RUNTIME_TEXTURE_ROLES
    ):
        raise GroundMaterialContractError(
            "Ground material runtime atlas set is incomplete"
        )
    if legacy_v1:
        catalog_runtime = catalog["runtime_atlas"]
        for key in ("size_px", "grid_size", "cell_size_px", "gutter_px"):
            if runtime.get(key) != catalog_runtime.get(key):
                raise GroundMaterialContractError(
                    f"Ground material runtime atlas geometry changed: {key}"
                )
    for role in RUNTIME_TEXTURE_ROLES:
        atlas_path = _validate_artifact(
            root, runtime["assets"][role], f"runtime_atlas.{role}"
        )
        png = _png_header(atlas_path)
        record = runtime["assets"][role]
        if any(
            record.get(key) != png[key]
            for key in ("width", "height", "bit_depth", "colour_type")
        ):
            raise GroundMaterialContractError(
                f"Ground material PNG contract mismatch: {role}"
            )
        expected_width = (
            catalog_runtime["assets"][role]["width"]
            if legacy_v1
            else runtime.get("width_px")
        )
        expected_height = (
            catalog_runtime["assets"][role]["height"]
            if legacy_v1
            else runtime.get("height_px")
        )
        if (
            record.get("width") != expected_width
            or record.get("height") != expected_height
        ):
            raise GroundMaterialContractError(
                f"Ground material atlas dimensions differ: {role}"
            )
    layer_text = material_path.read_text(encoding="utf-8")
    if (
        (LEGACY_MATERIAL_LAYER_SCHEMA if legacy_v1 else MATERIAL_LAYER_SCHEMA)
        not in layer_text
        or 'def Material "GroundMaterial"' not in layer_text
        or source_library["content_sha256"] not in layer_text
        or any(
            f"runtime-atlas/{role}.png" not in layer_text
            for role in RUNTIME_TEXTURE_ROLES
        )
        or (legacy_v1 and 'def Shader "GroundSurface"' not in layer_text)
        or (
            not legacy_v1
            and any(
                token not in layer_text
                for token in (
                    "groundConfidence",
                    "groundOrientation",
                    "compositionGridSize = 500",
                    "compositionCellSizeM = 1",
                    f'fireviewer:runtime_shader_status = "{RUNTIME_SHADER_PENDING_STATUS}"',
                    "fireviewer:production_textured_runtime_qualified = false",
                    'fireviewer:preview_surface_policy = "diagnostic_untextured_only"',
                    "color3f inputs:diffuseColor = (1, 0, 1)",
                )
            )
        )
        or (
            not legacy_v1
            and any(
                executable_claim in layer_text
                for executable_claim in (
                    'def Shader "GroundSurface"',
                    "outputs:fireviewer:surface",
                    'info:implementationSource = "sourceAsset"',
                    "outputs:mdl:surface",
                )
            )
        )
        or (
            not legacy_v1
            and any(
                prohibited in layer_text
                for prohibited in ("surfaceOverlays", "orthophoto", "procedural")
            )
        )
    ):
        raise GroundMaterialContractError(
            "Ground material USD layer has no reproducible binding"
        )
    return contract


def material_identity(contract_path: Path, zone_root: Path) -> dict[str, Any]:
    """Return the small immutable identity embedded in every tile package."""

    contract_file = Path(contract_path).resolve()
    zone = Path(zone_root).resolve()
    contract = validate_ground_material_contract(contract_file)
    try:
        zone_path = contract_file.relative_to(zone).as_posix()
    except ValueError as error:
        raise GroundMaterialContractError(
            "Ground material contract is outside its zone package"
        ) from error
    identity = {
        "schema": contract["schema"],
        "zone_path": zone_path,
        "contract_sha256": sha256_file(contract_file),
        "runtime_atlas_sha256": {
            role: contract["runtime_atlas"]["assets"][role]["sha256"]
            for role in RUNTIME_TEXTURE_ROLES
        },
        "material_layer_sha256": contract["material_layer"]["sha256"],
        "visual_acceptance": contract["visual_acceptance"],
    }
    if contract["schema"] == LEGACY_CONTRACT_SCHEMA:
        identity.update(
            {
                "source_atlas_catalog_sha256": contract["source_atlas_catalog_sha256"],
                "atlas_catalog_sha256": contract["atlas_catalog"]["sha256"],
            }
        )
    else:
        identity.update(
            {
                "source_library_schema": contract["source_library"]["schema"],
                "source_library_manifest_sha256": contract["source_library"][
                    "manifest_sha256"
                ],
                "source_library_identity_sha256": contract["source_library"][
                    "identity_sha256"
                ],
                "source_library_content_sha256": contract["source_library"][
                    "content_sha256"
                ],
                "texture_contract_sha256": contract["source_library"][
                    "texture_contract_sha256"
                ],
                "runtime_shader": contract["runtime_shader"],
            }
        )
    return identity


def resolve_zone_asset(zone_root: Path, relative_path: str, label: str) -> Path:
    """Resolve one portable zone path and reject traversal or reparse escapes."""

    relative = _portable_relative(relative_path, label)
    zone = Path(zone_root).resolve()
    resolved = zone.joinpath(*relative.parts).resolve()
    if resolved != zone and zone not in resolved.parents:
        raise GroundMaterialContractError(f"{label} escapes the zone package")
    return resolved


__all__ = [
    "ACCEPTED_ATLAS_CATALOG_STATUSES",
    "ATLAS_CATALOG_FILE_NAME",
    "ATLAS_SCHEMA",
    "CONTRACT_FILE_NAME",
    "CONTRACT_SCHEMA",
    "EXPECTED_PROFILE_COUNT",
    "GroundMaterialContractError",
    "MATERIAL_LAYER_FILE_NAME",
    "MATERIAL_LAYER_SCHEMA",
    "LEGACY_CONTRACT_FILE_NAME",
    "LEGACY_CONTRACT_SCHEMA",
    "LEGACY_MATERIAL_LAYER_SCHEMA",
    "RUNTIME_TEXTURE_ROLES",
    "RUNTIME_SHADER_PENDING_STATUS",
    "RUNTIME_SHADER_SCHEMA",
    "build_ground_material_bundle",
    "canonical_json",
    "canonical_sha256",
    "material_identity",
    "resolve_zone_asset",
    "sha256_file",
    "validate_ground_material_contract",
]
