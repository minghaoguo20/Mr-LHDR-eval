"""Dependency-DAG helpers for the benchmark checklist metric.

A graph is represented as ``parents[j]``: the direct prerequisites of checklist
step ``j``. Parents must refer only to earlier steps, which guarantees a DAG and
matches the checklist's authored order.
"""
from __future__ import annotations

Parents = list[list[int]]


def normalize(parents: Parents, n: int) -> Parents:
    """Repair raw parent lists into sorted, backward-only DAG edges."""
    out: Parents = []
    for j in range(n):
        raw = parents[j] if j < len(parents) else []
        out.append(sorted({
            p for p in raw
            if isinstance(p, int) and not isinstance(p, bool) and 0 <= p < j
        }))
    return out


def ancestor_sets(parents: Parents) -> list[set[int]]:
    """Return the transitive prerequisite set of every checklist step."""
    ancestors: list[set[int]] = []
    for j, direct in enumerate(parents):
        current: set[int] = set()
        for parent in direct:
            current.add(parent)
            current |= ancestors[parent]
        ancestors.append(current)
    return ancestors


def grounded(ancestors: list[set[int]], flags: list[bool]) -> list[bool]:
    """Mark steps satisfied together with every transitive prerequisite."""
    return [
        flag and all(flags[parent] for parent in ancestors[j])
        for j, flag in enumerate(flags)
    ]


def dacs(ancestors: list[set[int]], flags: list[bool]) -> float:
    """Fraction of checklist steps grounded in the dependency DAG."""
    return sum(grounded(ancestors, flags)) / len(flags) if flags else 0.0
