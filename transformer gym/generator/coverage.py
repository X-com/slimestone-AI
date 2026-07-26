"""The doc's single most useful generator diagnostic (§8): a block-type x event-kind matrix over
a shard directory, run BEFORE scaling up generation. An all-zero row/cell (e.g. "slime x
BlockLeftBehind") means the model will never see that combination no matter how much data gets
generated - a coverage gap, not a volume problem.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import generator` resolves
import generator  # noqa: F401,E402
from transformer_gym.simlog_reader import (
    BLOCK_NAMES,
    KIND_NAMES,
    iter_block_events,
    read_block_index,
    read_footer,
    read_initial_state,
)


def build_matrix(shard_dir: Path) -> Counter:
    """Counter keyed by (block_type_id, event_kind) -> occurrence count, across every .simlog
    in shard_dir."""
    matrix: Counter = Counter()
    for path in sorted(shard_dir.glob("*.simlog")):
        data = path.read_bytes()
        footer = read_footer(data)
        initial = {s.stableKey: s.blockTypeId for s in read_initial_state(data, footer)}
        for entry in read_block_index(data, footer):
            block_type = initial.get(entry.originalKey, entry.originalState & 0xFF)
            for ev in iter_block_events(data, entry):
                matrix[(block_type, ev.kind)] += 1
    return matrix


def print_matrix(matrix: Counter) -> None:
    block_types = sorted({bt for bt, _ in matrix})
    kinds = sorted(KIND_NAMES)
    header = "block_type".ljust(18) + "".join(KIND_NAMES[k][:10].ljust(11) for k in kinds)
    print(header)
    empty_rows = []
    for bt in block_types:
        name = BLOCK_NAMES.get(bt, f"id{bt}")
        row = "".join(str(matrix.get((bt, k), 0)).ljust(11) for k in kinds)
        print(name.ljust(18) + row)
        if all(matrix.get((bt, k), 0) == 0 for k in kinds):
            empty_rows.append(name)
    if empty_rows:
        print(f"\nall-zero rows (no events ever recorded): {empty_rows}")


if __name__ == "__main__":
    import sys
    build_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("generated")
    print_matrix(build_matrix(build_dir))
