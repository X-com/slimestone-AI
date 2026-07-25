"""Piston-event causality graph visualizer.

Renders a simulation_data (SDL3) log as a directed graph of piston/observer/redstone events - the
mechanism's causal flow - instead of a 3D block-by-block view. Built on verify_simulation_data.py's
decoder (no reimplementation of the binary format).

Why a graph: SDL3 already carries the causal structure in its fields - actorKey (what caused an
event), targetKey (what an event activated), pushGroupId (events from one piston firing), and
tick/subtick (ordering). A directed graph is the literal picture of those fields. A correct log draws
a connected chain (trigger -> observer fires -> piston extends -> slime pushed -> observer detects
change -> next piston fires); a broken attribution shows up as a piston-fire node with no incoming
cause edge - exactly what "verify the log works" needs.

Nodes = piston-centred events (PistonQueued/MoveExecuted/*Blocked) plus the observer/redstone/
BlockPushed events that cause or result from them. Edges = actorKey/targetKey read straight off the
records, plus a pushGroupId cluster grouping one firing's PistonMoveExecuted with its BlockPushed
events. Layout ranks by tick (top to bottom) so causality reads as a DAG.

Usage:
    py graph_visualizer.py                                     # every *.json in flying machines/json
    py graph_visualizer.py --dir <folder>                      # every *.json in <folder> instead
    py graph_visualizer.py <fixture_name> [fixture_name ...]    # just these, from the default folder
    py graph_visualizer.py --dir <folder> <fixture_name> [...]  # just these, from <folder>
    py graph_visualizer.py --self-check                        # invariant check on a known fixture

Fixtures to graph go in "flying machines/graph-json/" (drop the .json files you want logged there -
independent of the main "flying machines/json/" corpus). Each run writes one .dot + .svg per fixture
into a "graph/" subfolder of whichever input folder was used, and prints a PASS/uncaused-move report
per fixture to stdout.

Requires the Graphviz `dot` CLI on PATH (already installed in this environment).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import verify_simulation_data as vsd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_DIR = REPO / "flying machines" / "graph-json"

# Kinds that get their own node. Everything else (RedstoneBlockAppeared/Removed, reserved kinds) is
# skipped as a node but its causal effect (e.g. RedstoneActivatedPiston) still gets drawn.
_NAME_TO_KIND = {v: k for k, v in vsd.KIND_NAMES.items()}
NODE_KINDS = {_NAME_TO_KIND[n] for n in (
    "PistonQueued", "PistonMoveExecuted", "PistonExtendBlocked", "PistonRetractBlocked",
    "ObserverFired", "ObserverActivated", "BlockPushed",
    "RedstoneActivatedPiston", "RedstoneDeactivatedPiston", "PistonNeighborNotified",
)}

KIND_COLORS = {
    "PistonQueued": "#cfe8ff",
    "PistonMoveExecuted": "#8ecae6",
    "PistonExtendBlocked": "#ff6b6b",
    "PistonRetractBlocked": "#ff6b6b",
    "BlockPushed": "#a8dadc",
    "PistonNeighborNotified": "#f3f3f3",
    "ObserverFired": "#ffd166",
    "ObserverActivated": "#ffe8a3",
    "RedstoneActivatedPiston": "#c8e6a0",
    "RedstoneDeactivatedPiston": "#e0e0e0",
}


def _node_id(index: int) -> str:
    return f"n{index}"


def build_dot(data: bytes, name: str) -> str:
    footer = vsd.read_footer(data)
    index = vsd.read_block_index(data, footer)

    # Flatten every block's run back into one list, each tagged with its subject blockKey (already
    # on the event) - order doesn't matter here since edges are keyed by actor/target, not position.
    events = []
    for entry in index:
        events.extend(vsd.iter_block_events(data, entry))

    kept = [ev for ev in events if ev.kind in NODE_KINDS]
    kept.sort(key=lambda ev: ev.globalSeq)

    # actorKey/targetKey/blockKey -> list of node indices that are "this stable block's" most
    # recent piston-move / observer-fire / redstone event, so causal edges (actor->this event) can
    # resolve to the right upstream node without a second pass over positions.
    last_piston_move: dict[int, int] = {}   # pistonSubject -> node idx of its last MoveExecuted
    last_observer_fire: dict[int, int] = {}  # observerSubject -> node idx of its last ObserverFired
    last_redstone_event: dict[int, int] = {}  # redstoneSubject -> node idx of its last (de)activate

    lines = [
        "digraph simlog {",
        '  rankdir=TB;',
        '  node [shape=box, style=filled, fontsize=10, fontname="Consolas"];',
        f'  labelloc=t; label="{name}";',
    ]

    by_tick: dict[int, list[int]] = {}
    for i, ev in enumerate(kept):
        kind_name = vsd.KIND_NAMES[ev.kind]
        color = KIND_COLORS.get(kind_name, "#dddddd")
        pos = vsd.unpack_pos(ev.blockKey)
        label_lines = [f"{kind_name}", f"{pos} t={ev.activationTick}.{ev.activationSubtick}"]
        if ev.kind in (1, 9, 10):  # PistonMoveExecuted / *Blocked
            status = "moved" if ev.flags & vsd.SEF_SUCCESS else "BLOCKED"
            label_lines.append(status)
            if ev.failureReason:
                label_lines.append(vsd.FAILURE_REASON_NAMES.get(ev.failureReason, str(ev.failureReason)))
        if ev.kind == 3:  # ObserverFired
            label_lines.append(f"cause={vsd.CAUSE_NAMES.get((ev.flags >> 2) & 3, '?')}")
        if ev.kind == 15:  # PistonNeighborNotified: generic catch-all cause
            src_name = vsd.BLOCK_NAMES.get(ev.neighborSourceBlockId, f"id{ev.neighborSourceBlockId}")
            label_lines.append(f"from {vsd.unpack_pos(ev.actorKey)} (was {src_name})")
        label = "\\n".join(label_lines)
        lines.append(f'  {_node_id(i)} [label="{label}", fillcolor="{color}"];')
        by_tick.setdefault(ev.activationTick, []).append(i)

        if ev.kind == 1:  # PistonMoveExecuted
            last_piston_move[ev.blockKey] = i
        elif ev.kind == 3:  # ObserverFired
            last_observer_fire[ev.blockKey] = i
        elif ev.kind in (7, 8):  # Redstone(De)ActivatedPiston
            last_redstone_event[ev.blockKey] = i

    # Rank by tick so time flows top-to-bottom.
    for tick, idxs in sorted(by_tick.items()):
        same_rank = ", ".join(_node_id(i) for i in idxs)
        lines.append(f'  {{ rank=same; {same_rank} }}')

    # Edges, read straight off actorKey/targetKey - no inference. A cause-edge is drawn from the
    # most recent upstream event on the actor's subject block (the thing that fired) to this event.
    edges: set[tuple[int, int]] = set()
    for i, ev in enumerate(kept):
        if ev.kind == 2:  # BlockPushed: actorKey is the piston that pushed it
            src = last_piston_move.get(ev.actorKey)
            if src is not None:
                edges.add((src, i))
        elif ev.kind == 4:  # ObserverActivated: same observer's ObserverFired is the cause
            src = last_observer_fire.get(ev.blockKey)
            if src is not None:
                edges.add((src, i))
            # and it activates a piston/observer at targetKey - draw forward to that target's
            # next piston-queue/move if one exists later (best-effort, may not resolve).
        elif ev.kind in (7, 8):  # Redstone(De)ActivatedPiston: targetKey is the piston it drives
            dst = last_piston_move.get(ev.targetKey)
            if dst is not None and dst > i:
                edges.add((i, dst))
        elif ev.kind == 0:  # PistonQueued: the neighbor-notify that led checkForMove to queue it
            for j in range(i - 1, -1, -1):
                if kept[j].kind == 15 and kept[j].blockKey == ev.blockKey:
                    edges.add((j, i))
                    break
        elif ev.kind == 1:  # PistonMoveExecuted: what update reached this piston to re-check power?
            for j in range(i - 1, -1, -1):
                if kept[j].kind == 4 and kept[j].targetKey == ev.blockKey:
                    edges.add((j, i))  # an observer's pulse reached it
                    break
                if kept[j].kind in (7, 8) and kept[j].targetKey == ev.blockKey:
                    edges.add((j, i))  # a directly-adjacent redstone block (de)activated it
                    break
                if kept[j].kind == 2 and kept[j].blockKey == ev.blockKey:
                    # It was relocated by another piston - the relocation itself is the update that
                    # triggers the re-check (which may then read QC power with no separate event, by
                    # design: QC powers but never itself fires an update - see sim_event_log.h).
                    edges.add((j, i))
                    break
                if kept[j].kind == 15 and kept[j].blockKey == ev.blockKey:
                    # Generic catch-all: some neighboring block changed and notified this piston
                    # directly (no more specific kind applies - e.g. its own head appearing next to
                    # its own base). This is what used to show up as an "uncaused" piston move.
                    edges.add((j, i))
                    break

    for a, b in sorted(edges):
        lines.append(f"  {_node_id(a)} -> {_node_id(b)};")

    # Cluster every push group's PistonMoveExecuted with its BlockPushed events - one firing, one box.
    by_group: dict[int, list[int]] = {}
    for i, ev in enumerate(kept):
        if ev.pushGroupId:
            by_group.setdefault(ev.pushGroupId, []).append(i)
    for gid, idxs in sorted(by_group.items()):
        if len(idxs) < 2:
            continue
        members = ", ".join(_node_id(i) for i in idxs)
        lines.append(f"  subgraph cluster_grp{gid} {{ style=dashed; label=\"group {gid}\"; {members}; }}")

    lines.append("}")
    return "\n".join(lines)


def uncaused_piston_moves(data: bytes) -> list[str]:
    """The 'no uncaused piston' invariant: every non-blocked PistonMoveExecuted should have >=1
    incoming cause edge - an ObserverActivated or Redstone*Piston targeting it (a directly-adjacent
    trigger), a prior BlockPushed on itself (it was relocated - the relocation is the update that
    triggers the re-check, which may then read quasi-connectivity power with no separate event, by
    design: QC powers but never itself fires an update), or a PistonNeighborNotified on itself (the
    generic catch-all: some other neighboring block change reached it with no more specific kind
    applying, e.g. its own head appearing next to its own base). Anything else is a genuine gap."""
    footer = vsd.read_footer(data)
    index = vsd.read_block_index(data, footer)
    events = []
    for entry in index:
        events.extend(vsd.iter_block_events(data, entry))
    events.sort(key=lambda ev: ev.globalSeq)

    causes_by_target: dict[int, list[int]] = {}
    for ev in events:
        if ev.kind == 4:  # ObserverActivated
            causes_by_target.setdefault(ev.targetKey, []).append(ev.globalSeq)
        elif ev.kind in (7, 8):  # Redstone(De)ActivatedPiston
            causes_by_target.setdefault(ev.targetKey, []).append(ev.globalSeq)
        elif ev.kind == 2:  # BlockPushed - a self-relocation is also a valid upstream cause
            causes_by_target.setdefault(ev.blockKey, []).append(ev.globalSeq)
        elif ev.kind == 15:  # PistonNeighborNotified - the generic catch-all cause
            causes_by_target.setdefault(ev.blockKey, []).append(ev.globalSeq)

    problems = []
    for ev in events:
        if ev.kind != 1 or not (ev.flags & vsd.SEF_SUCCESS):
            continue
        upstream = causes_by_target.get(ev.blockKey, [])
        if not any(seq < ev.globalSeq for seq in upstream):
            problems.append(f"piston{vsd.unpack_pos(ev.blockKey)} t={ev.activationTick} has no prior cause edge")
    return problems


def render(name: str, workdir: Path, out_dir: Path) -> tuple[Path, bytes]:
    log_path = vsd.run_fixture(name, workdir)
    data = log_path.read_bytes()
    dot_src = build_dot(data, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / f"{name}.dot"
    svg_path = out_dir / f"{name}.svg"
    dot_path.write_text(dot_src, encoding="utf-8")
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    return svg_path, data


def _self_check() -> None:
    """Runs a known-good fixture and asserts the 'no uncaused piston' invariant holds."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        log_path = vsd.run_fixture("simple_observer_engine", workdir)
        data = log_path.read_bytes()
        problems = uncaused_piston_moves(data)
        assert not problems, f"uncaused piston move(s) found: {problems}"
        dot_src = build_dot(data, "simple_observer_engine")
        assert "PistonMoveExecuted" in dot_src
        assert "digraph simlog" in dot_src
    print("self-check PASS")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--self-check":
        _self_check()
        return 0

    fixture_dir = DEFAULT_FIXTURE_DIR
    if argv and argv[0] in ("--dir", "-d"):
        if len(argv) < 2:
            print("usage: py graph_visualizer.py --dir <folder> [fixture_name ...]")
            return 1
        fixture_dir = Path(argv[1])
        argv = argv[2:]
    fixture_dir.mkdir(parents=True, exist_ok=True)  # so dropping files in is all that's needed
    vsd.FIXTURE_DIR = fixture_dir  # run_fixture() reads this module-level constant
    out_dir = fixture_dir / "graph"

    names = argv if argv else [p.stem for p in sorted(fixture_dir.glob("*.json"))]
    if not names:
        print(f"no .json fixtures found in {fixture_dir}\n(drop the .json files you want logged there)")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name in names:
            svg_path, log_bytes = render(name, workdir, out_dir)
            problems = uncaused_piston_moves(log_bytes)
            status = "OK" if not problems else f"{len(problems)} uncaused piston move(s)"
            print(f"{name}: wrote {svg_path} ({status})")
            for p in problems:
                print(f"  ! {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
