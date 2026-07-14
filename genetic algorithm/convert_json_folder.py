"""Batch-convert every JSON candidate file in data/convert-json into .schematic files
under data/converted-schematics, using genetic_ml.schematic_export.

Run directly (e.g. via code-runner or `py convert_json_folder.py`) - this file lives at
the project root next to main_ga.py so genetic_ml is importable without any path tricks.
"""
from __future__ import annotations

from pathlib import Path

from genetic_ml.schematic_export import convert_json_to_schematics

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "convert-json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "converted-schematics"


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(INPUT_DIR.glob("*.json"))
    if not json_paths:
        print(f"no JSON files found in {INPUT_DIR}")
        return

    total_written = 0
    for json_path in json_paths:
        try:
            written = convert_json_to_schematics(json_path, OUTPUT_DIR)
        except Exception as exc:
            print(f"  failed to convert {json_path.name}: {exc}")
            continue
        for path in written:
            print(f"wrote {path}")
        total_written += len(written)

    print(f"done: {len(json_paths)} file(s) read, {total_written} schematic(s) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
