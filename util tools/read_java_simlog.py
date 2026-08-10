"""Decodes the Java-side mcp1122main.SimEventLog output (see
ignore/mcp1122/src/mcp1122main/SimEventLog.java) - the boolean-gated ("-Dmcp1122main.simlog=true")
binary event logger added to the real Minecraft source so its output can eventually be diffed
against the C++ simulator's own .simlog, tick by tick, to verify the port matches real mechanics.

NOT the SDL10/SDLA format: this is a flat, already-chronological array of the SAME 96-byte SimEvent
record (see verify_simulation_data.py's SimEvent/_EVENT, reused here unchanged) behind an 8-byte
header instead of SDLA's 188-byte trailing footer - see SimEventLog.java's class doc for why
there's no block-index/footer-sections equivalent to build on the Java side.

    header : 4-byte magic "JSDL" + uint32 LE event count
    events : eventCount x SimEvent (96 bytes), already in true chronological (emission) order -
             no block-major regrouping like the C++ file, so no k-way merge is needed here the
             way util tools/simlog_ticks.py needs one for the C++ side.

Usage:
    py "util tools/read_java_simlog.py" path/to/mcp1122.simlog
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_simulation_data import (  # noqa: E402
    _EVENT,
    KIND_NAMES,
    SimEvent,
    _fmt_event,
)

_HEADER = struct.Struct("<4sI")
assert _HEADER.size == 8, _HEADER.size


def read_java_simlog(path: Path) -> list[SimEvent]:
    data = path.read_bytes()
    magic, count = _HEADER.unpack_from(data, 0)
    if magic != b"JSDL":
        raise ValueError(f"bad magic {magic!r} in {path} (expected JSDL)")

    expected = _HEADER.size + count * _EVENT.size
    if len(data) != expected:
        raise ValueError(f"{path}: size {len(data)} != header + {count} events ({expected})")

    return [SimEvent(_EVENT.unpack_from(data, _HEADER.size + i * _EVENT.size)) for i in range(count)]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    path = Path(argv[0])
    events = read_java_simlog(path)
    print(f"{path}: {len(events)} events")
    counts: dict[int, int] = {}
    for ev in events:
        counts[ev.kind] = counts.get(ev.kind, 0) + 1
        if len(argv) > 1 and argv[1] == "--dump":
            print(_fmt_event(ev))

    print()
    for kind in sorted(counts):
        print(f"  {kind:>2} {KIND_NAMES.get(kind, '?'):<26} {counts[kind]:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
