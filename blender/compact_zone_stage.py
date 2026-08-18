"""Author one compact OpenUSD zone over validated measured tile scenes.

The tile packages remain complete production artifacts.  The downloadable
entry stage is a deployment assembly: every selected prototype is defined once
at zone scope and each 4 x 4 tile group is folded into one payload layer.  The
tile PointInstancers are preserved; only their prototype relationships and
terrain references are retargeted to the compact assembly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "fireviewer.compact-zone-stage-layout.v1"
STATUS = "sealed"
ENTRY_STAGE = "zone.usda"
LAYOUT_NAME = "zone-stage-layout.v1.json"
PAYLOAD_DIRECTORY = "payloads"
MEASURED_SCENE_SCHEMA = "fireviewer.measured-scene-receipt.v1"
TILE_SIZE_M = 500
METATILE_TILES = 4
METATILE_SIZE_M = TILE_SIZE_M * METATILE_TILES
FAMILY_SCOPES = {
    "buildings": "Buildings",
    "trees": "Trees",
    "context_assets": "ContextAssets",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PROTOTYPE_TARGET_RE = re.compile(
    r"</MeasuredScene/Prototypes/"
    r"(Buildings|Trees|ContextAssets)/(Asset_[A-Za-z0-9_]+)>"
)
POINT_INSTANCER_RE = re.compile(
    r'(?m)^[ \t]*def PointInstancer "(Buildings|Trees|ContextAssets)"[ \t]*$'
)
PROTOTYPE_RELATION_RE = re.compile(
    r"(?ms)^[ \t]*rel prototypes[ \t]*=[ \t]*\[(.*?)\][ \t]*(?:\r?\n|$)"
)


class CompactZoneStageError(RuntimeError):
    """Validated tile scenes cannot be assembled without duplication."""


@dataclass(frozen=True, slots=True)
class CompactTile:
    tile_id: str
    origin_l93_m: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Prototype:
    family: str
    scope: str
    asset_id: str
    identifier: str
    wrapper_reference: str
    wrapper_sha256: str
    wrapper_byte_count: int
    native_min_y: float
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class _PrototypeBinding:
    prim_name: str
    targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TileScene:
    tile: CompactTile
    scene_sha256: str
    scene_text: str
    terrain_reference: str
    prototypes: tuple[_Prototype, ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompactZoneStageError(f"{label} invalide: {error}") from error
    if not isinstance(value, dict):
        raise CompactZoneStageError(f"{label} doit être un objet JSON")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _inside(root: Path, child: Path, label: str) -> Path:
    try:
        child.relative_to(root)
    except ValueError as error:
        raise CompactZoneStageError(f"{label} sort du package de zone") from error
    return child


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "@" in value:
        raise CompactZoneStageError(f"{label} invalide")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", "."} for part in relative.parts):
        raise CompactZoneStageError(f"{label} invalide")
    return relative


def _identifier(value: str) -> str:
    return "Asset_" + re.sub(r"[^A-Za-z0-9_]", "_", value)


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise CompactZoneStageError("Valeur OpenUSD non finie")
    if value == 0:
        return "0"
    rendered = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _tuple(values: Sequence[float]) -> str:
    return "(" + ", ".join(_number(float(value)) for value in values) + ")"


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _matching_brace(text: str, opening: int, label: str) -> int:
    if opening >= len(text) or text[opening] != "{":
        raise CompactZoneStageError(f"Bloc OpenUSD {label} sans accolade")
    depth = 0
    quoted = False
    escaped = False
    asset_path = False
    for index in range(opening, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if asset_path:
            if character == "@":
                asset_path = False
            continue
        if character == '"':
            quoted = True
        elif character == "@":
            asset_path = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                break
    raise CompactZoneStageError(f"Bloc OpenUSD {label} non fermé")


def _prototype_from_receipt(
    root: Path,
    bundle_root: Path,
    raw: Any,
) -> _Prototype:
    if not isinstance(raw, Mapping):
        raise CompactZoneStageError("Prototype de tuile invalide")
    family = raw.get("family")
    if family not in FAMILY_SCOPES:
        raise CompactZoneStageError("Famille de prototype de tuile invalide")
    asset_id = raw.get("asset_id")
    if not isinstance(asset_id, str) or not ASSET_ID_RE.fullmatch(asset_id):
        raise CompactZoneStageError("Identifiant de prototype de tuile invalide")
    wrapper = raw.get("wrapper")
    if not isinstance(wrapper, Mapping):
        raise CompactZoneStageError(f"Wrapper de prototype absent: {asset_id}")
    wrapper_relative = _safe_relative(
        wrapper.get("path"), f"wrapper du prototype {asset_id}"
    )
    if wrapper_relative.parts[0] != asset_id:
        raise CompactZoneStageError(f"Wrapper hors prototype: {asset_id}")
    wrapper_path = bundle_root.joinpath(*wrapper_relative.parts).absolute()
    _inside(root, wrapper_path, f"wrapper du prototype {asset_id}")
    if not wrapper_path.is_file():
        raise CompactZoneStageError(f"Wrapper de prototype absent: {asset_id}")
    wrapper_sha256 = wrapper.get("sha256")
    wrapper_byte_count = wrapper.get("byte_count")
    if (
        not isinstance(wrapper_sha256, str)
        or not SHA256_RE.fullmatch(wrapper_sha256)
        or isinstance(wrapper_byte_count, bool)
        or not isinstance(wrapper_byte_count, int)
        or wrapper_byte_count <= 0
        or wrapper_path.stat().st_size != wrapper_byte_count
    ):
        raise CompactZoneStageError(f"Reçu de wrapper divergent: {asset_id}")
    native_min_y = raw.get("native_min_y")
    if isinstance(native_min_y, bool) or not isinstance(native_min_y, (int, float)):
        raise CompactZoneStageError(f"Ancrage natif invalide: {asset_id}")
    native_min_y = float(native_min_y)
    if not math.isfinite(native_min_y):
        raise CompactZoneStageError(f"Ancrage natif invalide: {asset_id}")
    wrapper_reference = os.path.relpath(wrapper_path, root).replace("\\", "/")
    if ":" in wrapper_reference or "@" in wrapper_reference:
        raise CompactZoneStageError(f"Référence de wrapper non portable: {asset_id}")
    identity_basis = {
        "asset_id": asset_id,
        "family": family,
        "native_min_y": native_min_y,
        "source_up_axis": raw.get("source_up_axis"),
        "source_usd": raw.get("source_usd"),
        "texture": raw.get("texture"),
        "wrapper": dict(wrapper),
        "material": raw.get("material"),
        "availability": raw.get("availability", "real_usd"),
        "fallback_resolution": raw.get("fallback_resolution"),
    }
    return _Prototype(
        family=str(family),
        scope=FAMILY_SCOPES[str(family)],
        asset_id=asset_id,
        identifier=_identifier(asset_id),
        wrapper_reference=wrapper_reference,
        wrapper_sha256=wrapper_sha256,
        wrapper_byte_count=wrapper_byte_count,
        native_min_y=native_min_y,
        identity_sha256=_sha256_bytes(_canonical_bytes(identity_basis)),
    )


def _load_tile_scene(root: Path, zone_id: str, tile: CompactTile) -> _TileScene:
    scene_root = root / "packages" / tile.tile_id / "scene"
    scene_path = scene_root / "scene.usda"
    receipt_path = scene_root / "scene.done.json"
    terrain_path = root / "packages" / tile.tile_id / "terrain-tile.usda"
    if (
        not scene_path.is_file()
        or not receipt_path.is_file()
        or not terrain_path.is_file()
    ):
        raise CompactZoneStageError(f"Package de scène incomplet: {tile.tile_id}")
    receipt = _load_json(receipt_path, f"reçu de scène {tile.tile_id}")
    if receipt.get("schema") != MEASURED_SCENE_SCHEMA:
        raise CompactZoneStageError(f"Schéma de scène invalide: {tile.tile_id}")
    if receipt.get("zone_id") != zone_id:
        raise CompactZoneStageError(f"Zone de scène divergente: {tile.tile_id}")
    scene_record = receipt.get("scene")
    if (
        not isinstance(scene_record, Mapping)
        or scene_record.get("path") != "scene.usda"
    ):
        raise CompactZoneStageError(f"Reçu de scène incomplet: {tile.tile_id}")
    scene_bytes = scene_path.read_bytes()
    scene_sha256 = _sha256_bytes(scene_bytes)
    if (
        scene_record.get("byte_count") != len(scene_bytes)
        or scene_record.get("sha256") != scene_sha256
    ):
        raise CompactZoneStageError(f"Scène de tuile altérée: {tile.tile_id}")
    try:
        scene_text = scene_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CompactZoneStageError(
            f"Scène de tuile non USDA: {tile.tile_id}"
        ) from error
    terrain = receipt.get("terrain")
    terrain_reference = (
        terrain.get("root_reference") if isinstance(terrain, Mapping) else None
    )
    _safe_relative(terrain_reference, f"terrain de la tuile {tile.tile_id}")
    bundle = receipt.get("prototype_bundle")
    if not isinstance(bundle, Mapping) or bundle.get("scope") != "explicit_shared":
        raise CompactZoneStageError(
            f"La tuile {tile.tile_id} ne référence pas le lot partagé"
        )
    bundle_reference = _safe_relative(
        bundle.get("root_reference"), f"lot partagé de la tuile {tile.tile_id}"
    )
    bundle_root = scene_root.joinpath(*bundle_reference.parts).resolve()
    _inside(root, bundle_root, f"lot partagé de la tuile {tile.tile_id}")
    if not bundle_root.is_dir():
        raise CompactZoneStageError(f"Lot partagé absent: {tile.tile_id}")
    raw_prototypes = receipt.get("prototypes")
    if not isinstance(raw_prototypes, list) or receipt.get("prototype_count") != len(
        raw_prototypes
    ):
        raise CompactZoneStageError(f"Prototypes de scène invalides: {tile.tile_id}")
    prototypes = tuple(
        _prototype_from_receipt(root, bundle_root, raw) for raw in raw_prototypes
    )
    return _TileScene(
        tile=tile,
        scene_sha256=scene_sha256,
        scene_text=scene_text,
        terrain_reference=str(terrain_reference),
        prototypes=prototypes,
    )


def _compact_tile_body(
    scene: _TileScene,
    *,
    root: Path,
    payload_directory: Path,
) -> tuple[str, tuple[_PrototypeBinding, ...]]:
    marker = 'def Xform "MeasuredScene"'
    marker_index = scene.scene_text.find(marker)
    if marker_index < 0:
        raise CompactZoneStageError(
            f"Racine MeasuredScene absente: {scene.tile.tile_id}"
        )
    opening = scene.scene_text.find("{", marker_index + len(marker))
    closing = _matching_brace(scene.scene_text, opening, "MeasuredScene")
    body = scene.scene_text[opening + 1 : closing]
    prototype_match = re.search(r'(?m)^[ \t]*def Scope "Prototypes"[ \t]*$', body)
    if prototype_match is None:
        raise CompactZoneStageError(f"Scope de prototypes absent: {scene.tile.tile_id}")
    prototype_opening = body.find("{", prototype_match.end())
    prototype_closing = _matching_brace(body, prototype_opening, "Prototypes")
    # Preserve the indentation of the following top-level prim.  Consuming all
    # whitespace here would eat its leading spaces and defeat the later dedent.
    compact = body[: prototype_match.start()] + body[prototype_closing + 1 :]

    terrain_path = root / "packages" / scene.tile.tile_id / "terrain-tile.usda"
    terrain_reference = os.path.relpath(terrain_path, payload_directory).replace(
        "\\", "/"
    )
    old_terrain = f"@{scene.terrain_reference}@"
    if compact.count(old_terrain) != 1:
        raise CompactZoneStageError(f"Référence terrain ambiguë: {scene.tile.tile_id}")
    compact = compact.replace(old_terrain, f"@{terrain_reference}@")

    expected_targets = {
        (prototype.scope, prototype.identifier) for prototype in scene.prototypes
    }
    observed_targets: set[tuple[str, str]] = set()
    bindings: list[_PrototypeBinding] = []
    instancers = list(POINT_INSTANCER_RE.finditer(compact))
    if {match.group(1) for match in instancers} != set(FAMILY_SCOPES.values()):
        raise CompactZoneStageError(
            f"PointInstancers de tuile incomplets: {scene.tile.tile_id}"
        )
    for match in reversed(instancers):
        prim_name = match.group(1)
        opening = compact.find("{", match.end())
        closing = _matching_brace(compact, opening, f"PointInstancer {prim_name}")
        block = compact[opening + 1 : closing]
        relation = PROTOTYPE_RELATION_RE.search(block)
        if relation is None:
            raise CompactZoneStageError(
                f"Relation de prototypes absente: {scene.tile.tile_id}/{prim_name}"
            )
        targets: list[str] = []
        for target_match in PROTOTYPE_TARGET_RE.finditer(relation.group(0)):
            target = (target_match.group(1), target_match.group(2))
            if target[0] != prim_name or target not in expected_targets:
                raise CompactZoneStageError(
                    f"Prototype non reçu dans {scene.tile.tile_id}: {target[1]}"
                )
            observed_targets.add(target)
            targets.append(f"</FireViewerZone/Prototypes/{target[0]}/{target[1]}>")
        if len(targets) != len(set(targets)):
            raise CompactZoneStageError(
                f"Relation de prototypes dupliquée: {scene.tile.tile_id}/{prim_name}"
            )
        bindings.append(_PrototypeBinding(prim_name, tuple(targets)))
        stripped_block = block[: relation.start()] + block[relation.end() :]
        compact = compact[: opening + 1] + stripped_block + compact[closing:]
    if observed_targets != expected_targets:
        raise CompactZoneStageError(
            f"Relations de prototypes incomplètes: {scene.tile.tile_id}"
        )
    if (
        "MeasuredScene/Prototypes" in compact
        or 'def Scope "Prototypes"' in compact
        or "rel prototypes" in compact
    ):
        raise CompactZoneStageError(f"Prototype local résiduel: {scene.tile.tile_id}")
    return textwrap.dedent(compact).strip(), tuple(
        sorted(bindings, key=lambda value: value.prim_name)
    )


def _render_payload(
    *,
    root: Path,
    payload_directory: Path,
    metatile_origin: tuple[int, int],
    scenes: Sequence[_TileScene],
) -> tuple[bytes, list[dict[str, Any]]]:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Metatile"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Metatile"',
        "{",
        f"    custom int fireviewer:tile_count = {len(scenes)}",
        f"    custom int2 fireviewer:origin_l93_m = {_tuple(metatile_origin)}",
    ]
    binding_records: list[dict[str, Any]] = []
    for scene in sorted(scenes, key=lambda item: item.tile.tile_id):
        tile_x, tile_y = scene.tile.origin_l93_m
        local = (tile_x - metatile_origin[0], tile_y - metatile_origin[1], 0)
        body, bindings = _compact_tile_body(
            scene,
            root=root,
            payload_directory=payload_directory,
        )
        for binding in bindings:
            if binding.targets:
                binding_records.append(
                    {
                        "tile_id": scene.tile.tile_id,
                        "prim_name": binding.prim_name,
                        "targets": list(binding.targets),
                    }
                )
        lines.extend(
            [
                "",
                f'    def Xform "{scene.tile.tile_id}"',
                "    {",
                f"        custom string fireviewer:source_scene_sha256 = "
                f"{_quoted(scene.scene_sha256)}",
                f"        double3 xformOp:translate = {_tuple(local)}",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                textwrap.indent(body, "        "),
                "    }",
            ]
        )
    lines.extend(["}", ""])
    return "\n".join(lines).encode("utf-8"), binding_records


def _render_root(
    *,
    zone_id: str,
    tile_count: int,
    production_origin: tuple[int, int],
    prototypes: Sequence[_Prototype],
    payloads: Sequence[Mapping[str, Any]],
) -> bytes:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "FireViewerZone"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "FireViewerZone"',
        "{",
        f"    custom string fireviewer:zone_id = {_quoted(zone_id)}",
        f"    custom int fireviewer:tile_count = {tile_count}",
        f"    custom int fireviewer:prototype_count = {len(prototypes)}",
        f"    custom int fireviewer:payload_count = {len(payloads)}",
        '    custom string fireviewer:assembly = "compact_metatile_payloads_v1"',
        "",
        '    def Scope "Prototypes"',
        "    {",
    ]
    by_scope: dict[str, list[_Prototype]] = defaultdict(list)
    for prototype in prototypes:
        by_scope[prototype.scope].append(prototype)
    for scope in FAMILY_SCOPES.values():
        lines.extend([f'        def Scope "{scope}"', "        {"])
        for prototype in sorted(
            by_scope.get(scope, []), key=lambda item: item.asset_id
        ):
            lines.extend(
                [
                    f'            def Xform "{prototype.identifier}"',
                    "            {",
                    '                def Xform "Source" (',
                    "                    prepend references = "
                    f"@{prototype.wrapper_reference}@",
                    "                )",
                    "                {",
                    "                    double3 xformOp:translate = "
                    f"{_tuple((0, -prototype.native_min_y, 0))}",
                    '                    uniform token[] xformOpOrder = ["xformOp:translate"]',
                    "                }",
                    "            }",
                ]
            )
        lines.append("        }")
    lines.extend(["    }", "", '    def Scope "Metatiles"', "    {"])
    for payload in payloads:
        origin = payload["origin_l93_m"]
        translation = (
            origin[0] - production_origin[0],
            origin[1] - production_origin[1],
            0,
        )
        lines.extend(
            [
                f'        def Xform "{payload["prim_name"]}" (',
                f"            prepend payload = @{payload['path']}@",
                "        )",
                "        {",
                f"            double3 xformOp:translate = {_tuple(translation)}",
                '            uniform token[] xformOpOrder = ["xformOp:translate"]',
            ]
        )
        bindings_by_tile: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for binding in payload["prototype_bindings"]:
            bindings_by_tile[str(binding["tile_id"])].append(binding)
        for tile_id in sorted(bindings_by_tile):
            lines.extend([f'            over "{tile_id}"', "            {"])
            observed_prim_names: set[str] = set()
            for binding in sorted(
                bindings_by_tile[tile_id], key=lambda item: str(item["prim_name"])
            ):
                prim_name = str(binding["prim_name"])
                if prim_name in observed_prim_names:
                    raise CompactZoneStageError(
                        f"Relation de prototype dupliquée: {tile_id}/{prim_name}"
                    )
                observed_prim_names.add(prim_name)
                lines.extend(
                    [
                        f'                over "{prim_name}"',
                        "                {",
                        "                    rel prototypes = [",
                        *[
                            f"                        {target},"
                            for target in binding["targets"]
                        ],
                        "                    ]",
                        "                }",
                    ]
                )
            lines.append("            }")
        lines.append("        }")
    lines.extend(["    }", "}", ""])
    return "\n".join(lines).encode("utf-8")


def _replace_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        existing = {
            path.relative_to(destination).as_posix(): _sha256_file(path)
            for path in destination.rglob("*")
            if path.is_file()
        }
        incoming = {
            path.relative_to(staging).as_posix(): _sha256_file(path)
            for path in staging.rglob("*")
            if path.is_file()
        }
        if existing == incoming:
            shutil.rmtree(staging)
            return
        shutil.rmtree(destination)
    os.replace(staging, destination)


def build_compact_zone_stage(
    job_root: Path | str,
    *,
    zone_id: str,
    production_bounds_l93_m: Sequence[int],
    tiles: Sequence[CompactTile],
) -> Path:
    """Build and validate the compact entry stage without rehashing assets."""

    root = Path(job_root).resolve(strict=True)
    if len(production_bounds_l93_m) != 4 or not tiles:
        raise CompactZoneStageError("Plan de zone compact invalide")
    west, south, east, north = (int(value) for value in production_bounds_l93_m)
    if west >= east or south >= north:
        raise CompactZoneStageError("Emprise de zone compacte invalide")
    tile_ids = [tile.tile_id for tile in tiles]
    if len(tile_ids) != len(set(tile_ids)):
        raise CompactZoneStageError("Identifiants de tuiles dupliqués")
    for tile in tiles:
        x, y = tile.origin_l93_m
        if (
            x % TILE_SIZE_M
            or y % TILE_SIZE_M
            or x < west
            or y < south
            or x + TILE_SIZE_M > east
            or y + TILE_SIZE_M > north
        ):
            raise CompactZoneStageError(f"Tuile hors emprise: {tile.tile_id}")

    scenes = [_load_tile_scene(root, zone_id, tile) for tile in tiles]
    prototypes_by_key: dict[tuple[str, str], _Prototype] = {}
    for scene in scenes:
        for prototype in scene.prototypes:
            key = (prototype.family, prototype.asset_id)
            existing = prototypes_by_key.get(key)
            if (
                existing is not None
                and existing.identity_sha256 != prototype.identity_sha256
            ):
                raise CompactZoneStageError(
                    f"Prototype divergent entre tuiles: {prototype.asset_id}"
                )
            prototypes_by_key[key] = prototype
    prototypes = tuple(prototypes_by_key[key] for key in sorted(prototypes_by_key))
    for prototype in prototypes:
        wrapper = root.joinpath(*PurePosixPath(prototype.wrapper_reference).parts)
        if _sha256_file(wrapper) != prototype.wrapper_sha256:
            raise CompactZoneStageError(
                f"Wrapper de prototype altéré: {prototype.asset_id}"
            )

    groups: dict[tuple[int, int], list[_TileScene]] = defaultdict(list)
    for scene in scenes:
        x, y = scene.tile.origin_l93_m
        origin = (
            west + ((x - west) // METATILE_SIZE_M) * METATILE_SIZE_M,
            south + ((y - south) // METATILE_SIZE_M) * METATILE_SIZE_M,
        )
        groups[origin].append(scene)

    payload_directory = root / PAYLOAD_DIRECTORY
    staging = root / f".{PAYLOAD_DIRECTORY}.part"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=False, exist_ok=False)
    payload_records: list[dict[str, Any]] = []
    try:
        for origin in sorted(groups, key=lambda value: (-value[1], value[0])):
            payload_id = f"x{origin[0]}_y{origin[1]}"
            relative = f"{PAYLOAD_DIRECTORY}/{payload_id}.usda"
            payload_bytes, prototype_bindings = _render_payload(
                root=root,
                payload_directory=payload_directory,
                metatile_origin=origin,
                scenes=groups[origin],
            )
            target = staging / f"{payload_id}.usda"
            target.write_bytes(payload_bytes)
            payload_records.append(
                {
                    "path": relative,
                    "prim_name": f"Metatile_{payload_id}",
                    "origin_l93_m": list(origin),
                    "tile_count": len(groups[origin]),
                    "tile_ids": sorted(scene.tile.tile_id for scene in groups[origin]),
                    "prototype_binding_count": len(prototype_bindings),
                    "prototype_bindings": prototype_bindings,
                    "byte_count": len(payload_bytes),
                    "sha256": _sha256_bytes(payload_bytes),
                }
            )
        _replace_directory(staging, payload_directory)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise

    root_bytes = _render_root(
        zone_id=zone_id,
        tile_count=len(tiles),
        production_origin=(west, south),
        prototypes=prototypes,
        payloads=payload_records,
    )
    stage_path = root / ENTRY_STAGE
    temporary_stage = root / f".{ENTRY_STAGE}.part"
    temporary_stage.write_bytes(root_bytes)
    os.replace(temporary_stage, stage_path)
    layout_without_hash: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "zone_id": zone_id,
        "tile_size_m": TILE_SIZE_M,
        "metatile_tiles": METATILE_TILES,
        "tile_count": len(tiles),
        "payload_count": len(payload_records),
        "prototype_count": len(prototypes),
        "prototype_policy": "one_zone_definition_per_family_asset",
        "instance_policy": "tile_point_instancers_preserved",
        "asset_hash_policy": "validated_tile_receipts_no_zone_rehash",
        "root_stage": {
            "path": ENTRY_STAGE,
            "byte_count": len(root_bytes),
            "sha256": _sha256_bytes(root_bytes),
        },
        "payloads": payload_records,
        "prototypes": [
            {
                "family": prototype.family,
                "asset_id": prototype.asset_id,
                "identifier": prototype.identifier,
                "wrapper_reference": prototype.wrapper_reference,
                "wrapper_sha256": prototype.wrapper_sha256,
                "wrapper_byte_count": prototype.wrapper_byte_count,
                "identity_sha256": prototype.identity_sha256,
            }
            for prototype in prototypes
        ],
    }
    layout = dict(layout_without_hash)
    layout["layout_sha256"] = _sha256_bytes(_canonical_bytes(layout_without_hash))
    _write_json(root / LAYOUT_NAME, layout)
    validate_compact_zone_stage(root, expected_tile_ids=tile_ids)
    return stage_path


def validate_compact_zone_stage(
    job_root: Path | str,
    *,
    expected_tile_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Revalidate the compact composition without opening heavy prototypes."""

    root = Path(job_root).resolve(strict=True)
    layout = _load_json(root / LAYOUT_NAME, "reçu de scène compacte")
    if layout.get("schema") != SCHEMA or layout.get("status") != STATUS:
        raise CompactZoneStageError("Reçu de scène compacte incompatible")
    supplied_hash = layout.get("layout_sha256")
    without_hash = dict(layout)
    without_hash.pop("layout_sha256", None)
    if (
        not isinstance(supplied_hash, str)
        or not SHA256_RE.fullmatch(supplied_hash)
        or supplied_hash != _sha256_bytes(_canonical_bytes(without_hash))
    ):
        raise CompactZoneStageError("Hash du reçu de scène compacte invalide")
    stage_record = layout.get("root_stage")
    if not isinstance(stage_record, Mapping) or stage_record.get("path") != ENTRY_STAGE:
        raise CompactZoneStageError("Racine de scène compacte invalide")
    stage = root / ENTRY_STAGE
    if (
        not stage.is_file()
        or stage.stat().st_size != stage_record.get("byte_count")
        or _sha256_file(stage) != stage_record.get("sha256")
    ):
        raise CompactZoneStageError("Racine de scène compacte altérée")
    root_text = stage.read_text(encoding="utf-8")
    if "packages/" in root_text or "/scene/scene.usda" in root_text:
        raise CompactZoneStageError("La racine compacte référence encore les tuiles")

    payloads = layout.get("payloads")
    prototypes = layout.get("prototypes")
    if (
        not isinstance(payloads, list)
        or layout.get("payload_count") != len(payloads)
        or not isinstance(prototypes, list)
        or layout.get("prototype_count") != len(prototypes)
    ):
        raise CompactZoneStageError("Comptes de scène compacte invalides")
    if root_text.count("prepend payload") != len(payloads):
        raise CompactZoneStageError("Arcs de payload de la racine divergents")
    if root_text.count("prepend references") != len(prototypes):
        raise CompactZoneStageError("Définitions globales de prototypes divergentes")
    expected_binding_count = sum(
        int(record.get("prototype_binding_count", -1))
        for record in payloads
        if isinstance(record, Mapping)
    )
    if root_text.count("rel prototypes") != expected_binding_count:
        raise CompactZoneStageError("Relations globales de prototypes divergentes")

    observed_tiles: list[str] = []
    for record in payloads:
        if not isinstance(record, Mapping):
            raise CompactZoneStageError("Payload compact invalide")
        relative = _safe_relative(record.get("path"), "chemin de payload compact")
        if relative.parts[0] != PAYLOAD_DIRECTORY:
            raise CompactZoneStageError("Payload hors répertoire compact")
        payload = root.joinpath(*relative.parts)
        if (
            not payload.is_file()
            or payload.stat().st_size != record.get("byte_count")
            or _sha256_file(payload) != record.get("sha256")
        ):
            raise CompactZoneStageError(f"Payload compact altéré: {relative}")
        text = payload.read_text(encoding="utf-8")
        if (
            'def Scope "Prototypes"' in text
            or "MeasuredScene/Prototypes" in text
            or "rel prototypes" in text
        ):
            raise CompactZoneStageError(f"Payload avec prototype local: {relative}")
        tile_ids = record.get("tile_ids")
        if (
            not isinstance(tile_ids, list)
            or record.get("tile_count") != len(tile_ids)
            or len(tile_ids) > METATILE_TILES * METATILE_TILES
        ):
            raise CompactZoneStageError(f"Tuiles de payload invalides: {relative}")
        for tile_id in tile_ids:
            if not isinstance(tile_id, str) or f'def Xform "{tile_id}"' not in text:
                raise CompactZoneStageError(f"Tuile absente du payload: {tile_id}")
            observed_tiles.append(tile_id)
        bindings = record.get("prototype_bindings")
        if not isinstance(bindings, list) or record.get(
            "prototype_binding_count"
        ) != len(bindings):
            raise CompactZoneStageError(
                f"Relations de prototypes de payload invalides: {relative}"
            )
        binding_tile_ids: set[str] = set()
        binding_keys: set[tuple[str, str]] = set()
        for binding in bindings:
            if (
                not isinstance(binding, Mapping)
                or binding.get("tile_id") not in tile_ids
                or binding.get("prim_name") not in FAMILY_SCOPES.values()
                or not isinstance(binding.get("targets"), list)
                or not binding["targets"]
            ):
                raise CompactZoneStageError(
                    f"Relation globale de prototype invalide: {relative}"
                )
            tile_id = str(binding["tile_id"])
            prim_name = str(binding["prim_name"])
            binding_key = (tile_id, prim_name)
            if binding_key in binding_keys:
                raise CompactZoneStageError(
                    f"Relation globale de prototype dupliquée: {tile_id}/{prim_name}"
                )
            binding_keys.add(binding_key)
            binding_tile_ids.add(tile_id)
            for target in binding["targets"]:
                if not isinstance(target, str) or target not in root_text:
                    raise CompactZoneStageError(
                        f"Cible globale de prototype absente: {relative}"
                    )
        for tile_id in tile_ids:
            expected_over_count = 1 if tile_id in binding_tile_ids else 0
            if root_text.count(f'            over "{tile_id}"') != expected_over_count:
                raise CompactZoneStageError(
                    f"Spécification OpenUSD de tuile dupliquée: {tile_id}"
                )
    if len(observed_tiles) != layout.get("tile_count") or len(observed_tiles) != len(
        set(observed_tiles)
    ):
        raise CompactZoneStageError("Réconciliation des tuiles compactes invalide")
    if expected_tile_ids is not None and set(observed_tiles) != set(expected_tile_ids):
        raise CompactZoneStageError("La scène compacte ne couvre pas le plan complet")
    return layout


__all__ = [
    "CompactTile",
    "CompactZoneStageError",
    "ENTRY_STAGE",
    "LAYOUT_NAME",
    "METATILE_TILES",
    "PAYLOAD_DIRECTORY",
    "SCHEMA",
    "build_compact_zone_stage",
    "validate_compact_zone_stage",
]
