from __future__ import annotations

from genetic_ml.population import Lineage, Population, canonical_hash


def make_candidate(block_count: int, offset: int = 0):
    return {
        "id": offset,
        "trigger": {"x": offset, "y": 0, "z": 0},
        "blocks": [{"x": offset + i, "y": 0, "z": 0, "state": 152} for i in range(block_count)],
    }


def test_canonical_hash_ignores_absolute_position_and_id():
    a = make_candidate(3, offset=0)
    b = make_candidate(3, offset=10)
    b["id"] = 999

    assert canonical_hash(a) == canonical_hash(b)


def test_canonical_hash_differs_for_different_shapes():
    a = make_candidate(3)
    b = make_candidate(4)

    assert canonical_hash(a) != canonical_hash(b)


def test_population_seed_respects_capacity_and_dedupes():
    population = Population(capacity=2)
    duplicate = make_candidate(3, offset=0)
    duplicate_again = make_candidate(3, offset=0)
    duplicate_again["id"] = 42
    distinct = make_candidate(5, offset=0)

    population.seed([duplicate, duplicate_again, distinct])

    assert len(population) == 2


def test_population_admit_evicts_largest_lineage_when_full():
    population = Population(capacity=1)
    big = Lineage(candidate=make_candidate(10), origin="seed", generation_found=0)
    population.lineages.append(big)
    population._known_hashes.add(big.hash)

    small = Lineage(candidate=make_candidate(2), origin="mutation", generation_found=1)
    admitted = population.admit(small)

    assert admitted is True
    assert population.lineages[0] is small


def test_population_admit_rejects_worse_lineage_when_full():
    population = Population(capacity=1)
    small = Lineage(candidate=make_candidate(2), origin="seed", generation_found=0)
    population.lineages.append(small)
    population._known_hashes.add(small.hash)

    big = Lineage(candidate=make_candidate(10), origin="mutation", generation_found=1)
    admitted = population.admit(big)

    assert admitted is False
    assert population.lineages[0] is small
