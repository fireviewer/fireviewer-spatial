"""Repair semantic properties misplaced inside arrays in generated fire-state USDA files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MISPLACED = re.compile(
    r"(?P<array>^[ \t]+(?:int\[\] faceVertexCounts|point3f\[\] points) = \[\n)"
    r"(?P<label>^[ \t]+token\[\] semantics:labels:class = \[[^\n]+\]\n)"
    r"(?P<custom>^[ \t]+custom string fireviewer:semantic_class = [^\n]+\n)",
    re.MULTILINE,
)


def repair_text(text: str) -> tuple[str, int]:
    return MISPLACED.subn(r"\g<label>\g<custom>\g<array>", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states_directory", type=Path)
    parser.add_argument("--write", action="store_true", help="Atomically update the state files in place")
    args = parser.parse_args()

    directory = args.states_directory.resolve()
    files = sorted(directory.glob("state_*.usda"))
    if len(files) != 180:
        raise SystemExit(f"expected 180 state files, found {len(files)} in {directory}")

    total_replacements = 0
    invalid: list[dict[str, object]] = []
    for path in files:
        original = path.read_text(encoding="utf-8")
        repaired, replacements = repair_text(original)
        total_replacements += replacements
        if replacements not in (0, 4):
            invalid.append({"path": str(path), "replacements": replacements})
            continue
        if args.write and replacements:
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text(repaired, encoding="utf-8", newline="\n")
            temporary.replace(path)

    if invalid:
        raise SystemExit(json.dumps({"status": "invalid_repair_count", "files": invalid}, indent=2))
    expected = 0 if total_replacements == 0 else len(files) * 4
    if total_replacements != expected:
        raise SystemExit(f"expected either 0 or {len(files) * 4} misplaced blocks, found {total_replacements}")
    if total_replacements and not args.write:
        raise SystemExit(f"found {total_replacements} misplaced blocks; rerun with --write")

    print(
        json.dumps(
            {
                "status": "state_usd_semantics_valid" if total_replacements == 0 else "state_usd_semantics_repaired",
                "state_files": len(files),
                "replacements": total_replacements,
                "directory": str(directory),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
