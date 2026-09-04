"""Machine library and attempt log - md/ALPHAZERO.md Part 6, DECISIONS points 26 and 29.

Two destinations, and the difference between them is the whole point:

    attempt log     EVERY simulated candidate, working or not, cargo or not
    machine library ONLY non-cargo working modifications

**Everything paid for is recorded.** A search spends 200 simulator calls and ends on one move;
the other 199 outcomes are true labels and cost nothing extra to keep. Training on cargo is
fine - a block riding along is physics the model should know.

**Cargo must never become a parent.** If it enters the library it becomes a base machine and its
children inherit it:

    generation 1     1 useful block,  2 cargo
    generation 10   10 useful blocks, 20 cargo

Three times the cell items to encode, three times the simulation cost, and the model's input
becomes mostly noise. Cargo accumulates monotonically because nothing removes it.

**Library size is not progress** - `canonical_hash` dedupes on structure, so a machine plus one
carried block hashes differently and trivial growth registers as discovery. That is exactly how
the GA convinced itself it was working. Size rising means the loop is running, nothing more.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from rlgym.game import Candidate, canonical_hash


@dataclass
class Entry:
    """One machine in the library."""

    digest: str
    name: str
    candidate: Candidate
    parent: str | None
    generation: int
    period: int
    shift: tuple[int, int, int]
    blocks: int
    round_found: int = 0
    descendants: int = 0


@dataclass
class Attempt:
    """One simulated candidate. Append-only, never revised."""

    digest: str
    parent: str
    round_index: int
    source: str  # top | sampled | uninformed - which share of the budget paid for it
    reward: float
    working: bool
    cargo: bool
    period: int
    shift: tuple[int, int, int]
    blocks: int


class Store:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "library.json"
        self.log_path = self.directory / "attempts.jsonl"
        self.entries: dict[str, Entry] = {}
        # Every digest ever simulated, so a candidate is never paid for twice across rounds -
        # the same transposition idea as inside one search, at the lifetime timescale.
        self.seen: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.entries = {
                digest: Entry(
                    **{**item, "shift": tuple(item["shift"])}
                )
                for digest, item in payload.items()
            }
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.seen[row["digest"]] = row["reward"]

    def save(self) -> None:
        self.index_path.write_text(
            json.dumps({d: asdict(e) for d, e in self.entries.items()}, indent=1),
            encoding="utf-8",
        )

    # --- writing ----------------------------------------------------------------------

    def record(self, attempt: Attempt) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(attempt), default=list) + "\n")
        self.seen[attempt.digest] = attempt.reward

    def admit(
        self,
        candidate: Candidate,
        parent: str | None,
        round_index: int,
        period: int,
        shift: tuple[int, int, int],
        cargo: bool,
        allow_cargo: bool = False,
    ) -> Entry | None:
        """Add a working modification to the library, unless it is cargo.

        Returns None when refused, which is the common case and not an error.
        """
        if cargo and not allow_cargo:
            return None
        digest = canonical_hash(candidate)
        if digest in self.entries:
            return None
        parent_entry = self.entries.get(parent) if parent else None
        entry = Entry(
            digest=digest,
            name=f"gen{(parent_entry.generation + 1) if parent_entry else 0}-{digest[:8]}",
            candidate=candidate,
            parent=parent,
            generation=(parent_entry.generation + 1) if parent_entry else 0,
            period=period,
            shift=tuple(shift),
            blocks=len(candidate["blocks"]),
            round_found=round_index,
        )
        self.entries[digest] = entry
        if parent_entry is not None:
            parent_entry.descendants += 1
        return entry

    def seed(self, name: str, candidate: Candidate, period: int, shift) -> Entry:
        """Put a fixture in as a generation-0 root."""
        digest = canonical_hash(candidate)
        if digest not in self.entries:
            self.entries[digest] = Entry(
                digest=digest,
                name=name,
                candidate=candidate,
                parent=None,
                generation=0,
                period=period,
                shift=tuple(shift),
                blocks=len(candidate["blocks"]),
            )
        return self.entries[digest]

    # --- reading ----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries.values())

    def attempts(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def variety(self) -> dict[str, Any]:
        """The numbers that show whether the library is growing in variety or only in count.

        `DEFERRED.md`'s binning problem, measured rather than argued: descendants concentrated
        on one root and a block histogram narrowing over time are what the self-selected
        training distribution looks like from the outside.
        """
        by_generation: dict[int, int] = {}
        by_root: dict[str, int] = {}
        for entry in self.entries.values():
            by_generation[entry.generation] = by_generation.get(entry.generation, 0) + 1
            root = entry
            while root.parent and root.parent in self.entries:
                root = self.entries[root.parent]
            by_root[root.name] = by_root.get(root.name, 0) + 1
        return {
            "size": len(self.entries),
            "by_generation": dict(sorted(by_generation.items())),
            "roots": len([e for e in self.entries.values() if e.parent is None]),
            "max_descendants_of_one_root": max(by_root.values(), default=0),
        }
