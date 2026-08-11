"""Inspect USD/USDZ candidate stages with Blender's bundled OpenUSD runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import sys
from typing import Any, Sequence


SCHEMA = "fireviewer.usd-candidate-inspection.v1"


class CandidateInspectionError(ValueError):
    """A candidate cannot be qualified for deterministic catalogue use."""


def _canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    rendered = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )
    if pretty:
        rendered += "\n"
    return rendered.encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_output(path: Path | str) -> Path:
    lexical = PureWindowsPath(str(path))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise CandidateInspectionError("inspection output must stay on D: on Windows")
    resolved = Path(path).resolve(strict=False)
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise CandidateInspectionError("inspection output must stay on D: on Windows")
    return resolved


def inspect_candidates(paths: Sequence[Path | str]) -> dict[str, Any]:
    """Inspect current stage metadata, bounds and material binding scope."""

    try:
        from pxr import Usd, UsdGeom, UsdShade
    except ImportError as error:  # pragma: no cover - exercised in Blender image
        raise CandidateInspectionError(
            "Blender OpenUSD Python bindings are unavailable"
        ) from error

    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in sorted(
        (Path(value).resolve() for value in paths), key=lambda item: item.name
    ):
        if raw.suffix.casefold() not in {".usd", ".usda", ".usdc", ".usdz"}:
            raise CandidateInspectionError(f"candidate is not USD: {raw.name}")
        if not raw.is_file() or raw.name in names:
            raise CandidateInspectionError("candidate names are missing or duplicated")
        names.add(raw.name)
        stage = Usd.Stage.Open(str(raw), load=Usd.Stage.LoadAll)
        if stage is None:
            raise CandidateInspectionError(f"cannot open USD stage: {raw.name}")
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            raise CandidateInspectionError(f"USD stage has no defaultPrim: {raw.name}")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        aligned = cache.ComputeWorldBound(default_prim).ComputeAlignedBox()
        minimum = [float(value) for value in aligned.GetMin()]
        maximum = [float(value) for value in aligned.GetMax()]
        if any(not math.isfinite(value) for value in (*minimum, *maximum)) or any(
            maximum[index] <= minimum[index] for index in range(3)
        ):
            raise CandidateInspectionError(f"USD stage bounds are invalid: {raw.name}")
        meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise CandidateInspectionError(f"USD stage contains no mesh: {raw.name}")
        default_path = default_prim.GetPath()
        material_paths: set[str] = set()
        bound_count = 0
        scope_safe = True
        for mesh in meshes:
            material, _relationship = UsdShade.MaterialBindingAPI(
                mesh
            ).ComputeBoundMaterial()
            if not material or not material.GetPrim().IsValid():
                scope_safe = False
                continue
            bound_count += 1
            material_path = material.GetPrim().GetPath()
            material_paths.add(str(material_path))
            if not material_path.HasPrefix(default_path):
                scope_safe = False
        records.append(
            {
                "source_name": raw.name,
                "source_sha256": _sha256_file(raw),
                "source_bounds": {
                    "status": "reported",
                    "coordinate_space": "usd_authored_world",
                    "minimum": minimum,
                    "maximum": maximum,
                },
                "usd_stage": {
                    "status": "inspected",
                    "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                    "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
                    "default_prim": str(default_path),
                },
                "mesh_count": len(meshes),
                "material_count": len(material_paths),
                "bound_material_mesh_count": bound_count,
                "material_scope_safe": scope_safe,
            }
        )
    payload: dict[str, Any] = {"schema": SCHEMA, "artifacts": records}
    payload["content_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def write_inspection(payload: dict[str, Any], output: Path | str) -> Path:
    target = _require_output(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_bytes(payload, pretty=True)
    if target.exists():
        if target.read_bytes() != content:
            raise CandidateInspectionError(
                "existing inspection is immutable and differs"
            )
        return target
    temporary = target.with_name(f".{target.name}.part")
    temporary.write_bytes(content)
    os.replace(temporary, target)
    return target


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = _arguments(argv)
    paths: list[Path] = []
    for root in options.input_root:
        if not root.is_dir():
            raise CandidateInspectionError(f"candidate root is missing: {root}")
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in {".usd", ".usda", ".usdc", ".usdz"}
        )
    output = write_inspection(inspect_candidates(paths), options.output)
    print(
        json.dumps({"output": str(output), "asset_count": len(paths)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    raise SystemExit(main(arguments))


__all__ = [
    "CandidateInspectionError",
    "SCHEMA",
    "inspect_candidates",
    "main",
    "write_inspection",
]
