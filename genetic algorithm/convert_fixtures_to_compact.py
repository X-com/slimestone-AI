"""Converts a folder of JSON candidate fixture files into compact-format (.dat) files, one
per input file, same stem - used to prep fixtures for cpp's file mode, which only reads the
compact format now (see genetic_ml/compact_format.py).

Usage: py convert_fixtures_to_compact.py <json_folder> <compact_folder>
"""
from __future__ import annotations

import sys
from pathlib import Path

from genetic_ml.compact_format import json_file_to_compact


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: py convert_fixtures_to_compact.py <json_folder> <compact_folder>")
        raise SystemExit(2)

    json_folder = Path(sys.argv[1])
    compact_folder = Path(sys.argv[2])
    compact_folder.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(json_folder.glob("*.json"))
    if not json_paths:
        print(f"no .json files found in {json_folder}")
        return

    total = 0
    for json_path in json_paths:
        compact_path = compact_folder / (json_path.stem + ".dat")
        try:
            count = json_file_to_compact(json_path, compact_path)
        except Exception as exc:
            print(f"  failed to convert {json_path.name}: {exc}")
            continue
        print(f"wrote {compact_path} ({count} candidate(s))")
        total += count

    print(f"done: {len(json_paths)} file(s) read, {total} candidate(s) converted to {compact_folder}")


if __name__ == "__main__":
    main()
