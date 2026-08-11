"""Pack offline ImageGen micro-details into four runtime surface atlases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


SCHEMA = "fireviewer.ground-surface-atlas-library.v2"
CONTRACT_SCHEMA = "fireviewer.ground-surface-runtime-contract.v2"
EXPECTED_MICRO_SOURCE_COUNT = 21
EXPECTED_PROFILE_COUNT = 72
EXPECTED_RUNTIME_TEXTURE_COUNT = 4
EXPECTED_APPLICATION_MODES = {
    "ground_blend",
    "directional_area",
    "linear_overlay",
    "watercourse_overlay",
    "slope_cliff_overlay",
}
REQUIRED_INCIDENTS = {
    "FR-30-00001",
    "FR-34-00001",
    "FR-83-00001",
    "FR-26-00001",
    "FR-66-00001",
    "FR-77-00001",
}


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
    return hashlib.sha256(encoded).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unsupported ground surface runtime contract")
    if payload.get("orthophoto_dependency") != "forbidden":
        raise ValueError("Ground surface runtime must forbid orthophotos")
    scale = payload.get("scale_contract", {})
    if scale.get("source_image_role") != "offline_micro_detail_only":
        raise ValueError("ImageGen sources must be offline micro-details only")
    if scale.get("direct_source_image_import") != "forbidden":
        raise ValueError("Direct source image imports must be forbidden")
    atlas = payload.get("runtime_atlas", {})
    if atlas.get("runtime_texture_count") != EXPECTED_RUNTIME_TEXTURE_COUNT:
        raise ValueError("Runtime atlas must expose exactly four textures")
    if atlas.get("runtime_textures") != ["basecolor", "normal", "height", "orm"]:
        raise ValueError("Runtime atlas texture roles are invalid")
    atlas_size = int(atlas.get("size_px", 0))
    grid_size = int(atlas.get("grid_size", 0))
    cell_size = int(atlas.get("cell_size_px", 0))
    gutter = int(atlas.get("gutter_px", 0))
    if atlas_size != grid_size * cell_size or cell_size <= 2 * gutter:
        raise ValueError("Runtime atlas dimensions are inconsistent")
    groups = payload.get("micro_source_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Micro source groups are absent")
    group_ids = [str(group.get("id", "")) for group in groups]
    if len(set(group_ids)) != len(group_ids) or any(not item for item in group_ids):
        raise ValueError("Micro source group identifiers must be unique")
    sources = [str(source) for group in groups for source in group.get("sources", [])]
    if len(sources) != EXPECTED_MICRO_SOURCE_COUNT or len(set(sources)) != len(sources):
        raise ValueError(
            f"Ground atlas requires {EXPECTED_MICRO_SOURCE_COUNT} unique micro sources"
        )
    for group in groups:
        if not 1.0 <= float(group.get("physical_scale_m", 0.0)) <= 8.0:
            raise ValueError(f"Invalid micro scale for {group['id']}")
        if not 0.05 <= float(group.get("base_roughness", 0.0)) <= 1.0:
            raise ValueError(f"Invalid roughness for {group['id']}")
        if not 1.0 <= float(group.get("normal_strength", 0.0)) <= 12.0:
            raise ValueError(f"Invalid normal strength for {group['id']}")
    families = payload.get("profile_families")
    if not isinstance(families, list) or not families:
        raise ValueError("Ground surface profile families are absent")
    family_ids = [str(family.get("id", "")) for family in families]
    if len(set(family_ids)) != len(family_ids) or any(not item for item in family_ids):
        raise ValueError("Ground surface family identifiers must be unique")
    variants = [
        f"{family['id']}.{variant}"
        for family in families
        for variant in family.get("variant_ids", [])
    ]
    if len(variants) != EXPECTED_PROFILE_COUNT or len(set(variants)) != len(variants):
        raise ValueError(f"Ground surface runtime requires {EXPECTED_PROFILE_COUNT} profiles")
    application_modes = {str(family.get("application_mode")) for family in families}
    if application_modes != EXPECTED_APPLICATION_MODES:
        raise ValueError("Ground surface application modes are incomplete")
    known_groups = set(group_ids)
    for family in families:
        referenced = set(family.get("micro_source_groups", []))
        if not referenced or not referenced <= known_groups:
            raise ValueError(f"Invalid micro source pool for {family['id']}")
    if payload.get("determinism", {}).get("silent_fallback") != "forbidden":
        raise ValueError("Ground surface composition must fail closed")
    return payload


def periodic_blend(values: np.ndarray, seam_width: int) -> np.ndarray:
    if values.ndim not in {2, 3}:
        raise ValueError("Periodic blending expects a 2D or 3D array")
    height, width = values.shape[:2]
    if seam_width < 2 or seam_width * 2 >= min(height, width):
        raise ValueError("Invalid periodic seam width")
    result = values.astype(np.float32, copy=True)
    for index in range(seam_width):
        weight = 0.5 * (1.0 + math.cos(math.pi * index / (seam_width - 1)))
        left = result[:, index].copy()
        right = result[:, -1 - index].copy()
        average = (left + right) * 0.5
        result[:, index] = left * (1.0 - weight) + average * weight
        result[:, -1 - index] = right * (1.0 - weight) + average * weight
    for index in range(seam_width):
        weight = 0.5 * (1.0 + math.cos(math.pi * index / (seam_width - 1)))
        top = result[index].copy()
        bottom = result[-1 - index].copy()
        average = (top + bottom) * 0.5
        result[index] = top * (1.0 - weight) + average * weight
        result[-1 - index] = bottom * (1.0 - weight) + average * weight
    return result


def _periodic_uint8(values: np.ndarray, seam_width: int) -> np.ndarray:
    output = np.rint(np.clip(periodic_blend(values, seam_width), 0, 255)).astype(
        np.uint8
    )
    output[:, -1] = output[:, 0]
    output[-1] = output[0]
    return output


def _periodic_uint16(values: np.ndarray, seam_width: int) -> np.ndarray:
    output = np.rint(np.clip(periodic_blend(values, seam_width), 0, 65535)).astype(
        np.uint16
    )
    output[:, -1] = output[:, 0]
    output[-1] = output[0]
    return output


def seam_error(values: np.ndarray) -> int:
    signed = values.astype(np.int64)
    return int(
        max(
            np.abs(signed[:, 0] - signed[:, -1]).max(),
            np.abs(signed[0] - signed[-1]).max(),
        )
    )


def prepare_basecolor(source: Path, content_size: int, seam_width: int) -> np.ndarray:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (content_size, content_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        values = np.asarray(image, dtype=np.float32)
    output = _periodic_uint8(values, seam_width)
    if float(output.std()) < 4.0:
        raise ValueError(f"Generated micro-detail is too uniform: {source}")
    return output


def derive_pbr_maps(
    basecolor: np.ndarray,
    *,
    base_roughness: float,
    normal_strength: float,
    seam_width: int,
) -> dict[str, np.ndarray]:
    rgb = basecolor.astype(np.float32) / 255.0
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    blur_radius = max(3.0, basecolor.shape[0] / 28.0)
    blurred = np.asarray(
        Image.fromarray(np.rint(luminance * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        ),
        dtype=np.float32,
    ) / 255.0
    detail = luminance - blurred
    low, high = np.percentile(detail, [1.0, 99.0])
    if math.isclose(float(low), float(high), abs_tol=1e-6):
        raise ValueError("Cannot derive height from a flat micro-detail")
    height = np.clip((detail - low) / (high - low), 0, 1)
    height16 = _periodic_uint16(height * 65535, seam_width)
    height = height16.astype(np.float32) / 65535.0
    gradient_x = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    gradient_y = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    normals = np.stack(
        (-gradient_x * normal_strength, -gradient_y * normal_strength, np.ones_like(height)),
        axis=-1,
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normal = _periodic_uint8((normals * 0.5 + 0.5) * 255, seam_width)
    micro = np.abs(detail)
    micro /= max(float(np.percentile(micro, 99.0)), 1e-6)
    roughness = np.clip(base_roughness + (micro - 0.35) * 0.08, 0.05, 1.0)
    roughness8 = _periodic_uint8(roughness * 255, seam_width)
    orm = np.stack(
        (np.full_like(roughness8, 255), roughness8, np.zeros_like(roughness8)),
        axis=-1,
    )
    return {"height": height16, "normal": normal, "orm": orm}


def _wrap_gutter(values: np.ndarray, gutter: int) -> np.ndarray:
    padding = ((gutter, gutter), (gutter, gutter))
    if values.ndim == 3:
        padding += ((0, 0),)
    return np.pad(values, padding, mode="wrap")


def _save_png(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    Image.fromarray(values).save(temporary, format="PNG", optimize=True, compress_level=9)
    temporary.replace(path)


def _asset_record(path: Path, root: Path, role: str) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "width": width,
        "height": height,
    }


def _profile_parameters(profile_id: str, meso: list[float], macro: list[float]) -> dict[str, Any]:
    digest = hashlib.sha256(profile_id.encode("utf-8")).digest()

    def interpolate(bounds: list[float], value: int) -> float:
        return round(bounds[0] + (bounds[1] - bounds[0]) * value / 255.0, 4)

    return {
        "meso_scale_m": interpolate(meso, digest[1]),
        "macro_scale_m": interpolate(macro, digest[2]),
        "basecolor_gain": round(0.88 + digest[3] / 255.0 * 0.24, 4),
        "hue_rotation_degrees": round(-8.0 + digest[4] / 255.0 * 16.0, 4),
        "roughness_delta": round(-0.08 + digest[5] / 255.0 * 0.16, 4),
        "normal_multiplier": round(0.75 + digest[6] / 255.0 * 0.5, 4),
        "noise_seed": int.from_bytes(digest[7:11], "big"),
    }


def build_library(contract_path: Path, source_root: Path, output_root: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    runtime = contract["runtime_atlas"]
    atlas_size = int(runtime["size_px"])
    grid_size = int(runtime["grid_size"])
    cell_size = int(runtime["cell_size_px"])
    gutter = int(runtime["gutter_px"])
    content_size = cell_size - 2 * gutter
    seam_width = max(4, min(48, content_size // 8))
    groups = contract["micro_source_groups"]
    source_settings = {
        source: group
        for group in groups
        for source in group["sources"]
    }
    expected_sources = set(source_settings)
    actual_sources = {path.name for path in source_root.glob("*.png")}
    if actual_sources != expected_sources:
        missing = sorted(expected_sources - actual_sources)
        unexpected = sorted(actual_sources - expected_sources)
        raise ValueError(
            f"Micro source set mismatch; missing={missing}, unexpected={unexpected}"
        )
    if len(expected_sources) > grid_size * grid_size:
        raise ValueError("Runtime atlas has too few cells")
    atlases = {
        "basecolor": np.full((atlas_size, atlas_size, 3), 128, dtype=np.uint8),
        "normal": np.dstack(
            (
                np.full((atlas_size, atlas_size), 128, dtype=np.uint8),
                np.full((atlas_size, atlas_size), 128, dtype=np.uint8),
                np.full((atlas_size, atlas_size), 255, dtype=np.uint8),
            )
        ),
        "height": np.zeros((atlas_size, atlas_size), dtype=np.uint16),
        "orm": np.dstack(
            (
                np.full((atlas_size, atlas_size), 255, dtype=np.uint8),
                np.full((atlas_size, atlas_size), 230, dtype=np.uint8),
                np.zeros((atlas_size, atlas_size), dtype=np.uint8),
            )
        ),
    }
    source_records: list[dict[str, Any]] = []
    source_by_group: dict[str, list[dict[str, Any]]] = {group["id"]: [] for group in groups}
    for index, source_name in enumerate(sorted(expected_sources)):
        group = source_settings[source_name]
        source = source_root / source_name
        basecolor = prepare_basecolor(source, content_size, seam_width)
        maps = {
            "basecolor": basecolor,
            **derive_pbr_maps(
                basecolor,
                base_roughness=float(group["base_roughness"]),
                normal_strength=float(group["normal_strength"]),
                seam_width=seam_width,
            ),
        }
        row, column = divmod(index, grid_size)
        y0, x0 = row * cell_size, column * cell_size
        maximum_edge_error = 0
        for role, values in maps.items():
            maximum_edge_error = max(maximum_edge_error, seam_error(values))
            atlases[role][y0 : y0 + cell_size, x0 : x0 + cell_size] = _wrap_gutter(
                values, gutter
            )
        record = {
            "id": Path(source_name).stem,
            "group": group["id"],
            "source": {
                "path": Path(
                    os.path.relpath(source.resolve(), output_root.resolve())
                ).as_posix(),
                "sha256": sha256_file(source),
                "byte_count": source.stat().st_size,
                "generator": "openai_builtin_imagegen",
                "runtime_import": "forbidden",
            },
            "physical_scale_m": group["physical_scale_m"],
            "slot": index,
            "atlas_uv": {
                "offset": [
                    round((x0 + gutter) / atlas_size, 10),
                    round((y0 + gutter) / atlas_size, 10),
                ],
                "scale": [
                    round(content_size / atlas_size, 10),
                    round(content_size / atlas_size, 10),
                ],
            },
            "qa": {"automated": "passed", "maximum_edge_error": maximum_edge_error},
        }
        source_records.append(record)
        source_by_group[group["id"]].append(record)
    atlas_root = output_root / "runtime-atlas"
    role_names = {
        "basecolor": "srgb_base_color_atlas",
        "normal": "linear_opengl_tangent_normal_atlas",
        "height": "linear_height_proxy_16bit_atlas",
        "orm": "linear_orm_ao_roughness_metallic_atlas",
    }
    atlas_assets = {}
    for role, values in atlases.items():
        path = atlas_root / f"{role}.png"
        _save_png(values, path)
        atlas_assets[role] = _asset_record(path, output_root, role_names[role])
    scale = contract["scale_contract"]
    profiles = []
    for family in contract["profile_families"]:
        pool = [
            source
            for group_id in family["micro_source_groups"]
            for source in source_by_group[group_id]
        ]
        for variant in family["variant_ids"]:
            profile_id = f"{family['id']}.{variant}"
            digest = hashlib.sha256(profile_id.encode("utf-8")).digest()
            source = pool[int.from_bytes(digest[:2], "big") % len(pool)]
            profiles.append(
                {
                    "id": profile_id,
                    "family": family["id"],
                    "variant": variant,
                    "application_mode": family["application_mode"],
                    "shader": family["shader"],
                    "micro_source_id": source["id"],
                    "micro_source_slot": source["slot"],
                    "parameters": _profile_parameters(
                        profile_id,
                        scale["meso_variation_m"],
                        scale["macro_variation_m"],
                    ),
                    "compatible_incidents": sorted(REQUIRED_INCIDENTS),
                }
            )
    contact_sheet = _write_contact_sheet(source_records, source_root, output_root)
    catalog = {
        "schema": SCHEMA,
        "status": "generated_pending_omniverse_visual_acceptance",
        "orthophoto_dependency": "forbidden",
        "contract": {
            "path": Path(
                os.path.relpath(contract_path.resolve(), output_root.resolve())
            ).as_posix(),
            "sha256": sha256_file(contract_path),
            "status": contract["status"],
        },
        "micro_source_count": len(source_records),
        "micro_source_runtime_import": "forbidden",
        "runtime_texture_count": len(atlas_assets),
        "runtime_atlas": {
            "size_px": atlas_size,
            "grid_size": grid_size,
            "cell_size_px": cell_size,
            "gutter_px": gutter,
            "assets": atlas_assets,
        },
        "profile_count": len(profiles),
        "application_modes": sorted(EXPECTED_APPLICATION_MODES),
        "scale_contract": scale,
        "composition": contract["composition"],
        "required_context_layers": contract["required_context_layers"],
        "micro_sources": source_records,
        "profiles": profiles,
        "contact_sheet": contact_sheet,
        "qa": {
            "automated": "passed",
            "visual": "pending_omniverse_multi_scale_review",
            "maximum_cell_edge_error": max(
                source["qa"]["maximum_edge_error"] for source in source_records
            ),
        },
    }
    catalog["catalog_sha256"] = canonical_sha256(catalog)
    catalog_path = output_root / "ground-surface-atlas-catalog.json"
    temporary = catalog_path.with_name(f".{catalog_path.name}.tmp")
    temporary.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(catalog_path)
    return catalog


def _write_contact_sheet(
    records: list[dict[str, Any]], source_root: Path, output_root: Path
) -> dict[str, Any]:
    columns, cell, label = 4, 256, 38
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        column, row = index % columns, index // columns
        x, y = column * cell, row * (cell + label)
        with Image.open(source_root / f"{record['id']}.png") as image:
            thumbnail = ImageOps.fit(image.convert("RGB"), (cell, cell))
        sheet.paste(thumbnail, (x, y))
        draw.text((x + 5, y + cell + 4), record["id"], fill=(235, 235, 235))
        draw.text((x + 5, y + cell + 20), record["group"], fill=(150, 190, 220))
    path = output_root / "micro-source-contact-sheet.png"
    _save_png(np.asarray(sheet), path)
    return _asset_record(path, output_root, "offline_review_contact_sheet")


def validate_catalog(catalog: dict[str, Any], root: Path) -> None:
    if catalog.get("schema") != SCHEMA:
        raise ValueError("Unsupported ground surface atlas catalog")
    if catalog.get("micro_source_count") != EXPECTED_MICRO_SOURCE_COUNT:
        raise ValueError("Ground surface micro source set is incomplete")
    if catalog.get("profile_count") != EXPECTED_PROFILE_COUNT:
        raise ValueError("Ground surface profile set is incomplete")
    if catalog.get("runtime_texture_count") != EXPECTED_RUNTIME_TEXTURE_COUNT:
        raise ValueError("Ground surface runtime must import exactly four atlas textures")
    if catalog.get("micro_source_runtime_import") != "forbidden":
        raise ValueError("Direct micro source imports must remain forbidden")
    if catalog.get("orthophoto_dependency") != "forbidden":
        raise ValueError("Ground surface atlas must forbid orthophotos")
    unsigned = dict(catalog)
    saved_hash = unsigned.pop("catalog_sha256", None)
    if saved_hash != canonical_sha256(unsigned):
        raise ValueError("Ground surface atlas catalog hash mismatch")
    for asset in catalog["runtime_atlas"]["assets"].values():
        path = root / asset["path"]
        if not path.is_file() or sha256_file(path) != asset["sha256"]:
            raise ValueError(f"Runtime atlas asset mismatch: {path}")
    if any(source["qa"]["maximum_edge_error"] != 0 for source in catalog["micro_sources"]):
        raise ValueError("A micro source cell is not periodic")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    catalog = build_library(
        args.contract.resolve(), args.source_root.resolve(), output_root
    )
    validate_catalog(catalog, output_root)
    print(
        json.dumps(
            {
                "catalog_sha256": catalog["catalog_sha256"],
                "micro_source_count": catalog["micro_source_count"],
                "profile_count": catalog["profile_count"],
                "runtime_texture_count": catalog["runtime_texture_count"],
                "status": catalog["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
