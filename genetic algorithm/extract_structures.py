"""Extracts a sample of unverified flying machines from compact-working/flyers.data and
converts them into Minecraft structure block .nbt files for manual investigation, written into
data/structure-blocks/.

Set SAMPLE_STRIDE to extract every Nth record by position in flyers.data (0, N, 2N, ...), or
set SAMPLE_INDICES to a specific list of indices to hand-pick instead (SAMPLE_INDICES takes
priority over SAMPLE_STRIDE when both are set).
"""
from __future__ import annotations

from pathlib import Path

from genetic_ml.compact_format import read_compact_file
from genetic_ml.structure_export import export_candidate_to_structure

PROJECT_ROOT = Path(__file__).resolve().parent
FLYERS_PATH = PROJECT_ROOT / "data" / "compact-working" / "flyers.data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "structure-blocks"

SAMPLE_STRIDE: int | None = 1000
SAMPLE_INDICES: list[int] | None = None


def main() -> None:
    candidates = read_compact_file(FLYERS_PATH)
    if not candidates:
        print(f"no records found in {FLYERS_PATH}")
        return

    if SAMPLE_INDICES is not None:
        indices = [i for i in SAMPLE_INDICES if 0 <= i < len(candidates)]
    else:
        stride = SAMPLE_STRIDE or 1
        indices = list(range(0, len(candidates), stride))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in indices:
        candidate = candidates[index]
        try:
            path = export_candidate_to_structure(
                candidate, OUTPUT_DIR, filename=f"flyer-{index}.nbt"
            )
        except Exception as exc:
            print(f"  failed to convert index {index} (id={candidate.get('id')}): {exc}")
            continue
        print(f"wrote {path}")
        written += 1

    print(f"done: {written}/{len(indices)} structure(s) written to {OUTPUT_DIR} (from {len(candidates)} total record(s))")


if __name__ == "__main__":
    main()
