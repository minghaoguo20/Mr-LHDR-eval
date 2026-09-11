"""The benchmark metrics.

FIXED DEFINITION. This module is the single authority for what a benchmark score
is; the numbers in the paper come from exactly this arithmetic. Changing
anything here changes what the benchmark measures, so treat edits the way you
would treat editing the dataset -- tests/test_metrics.py pins the behaviour
against verdicts taken from the frozen run.

Four metrics, each computed per item and then averaged over items:

  OA       Overall Accuracy -- did the final answer match the reference?
  SA       Strict Accuracy -- OA *and* every checklist step satisfied.
  CS       Checklist Score -- fraction of checklist steps satisfied, any order.
  DACS     Dependency-Aware Checklist Score -- fraction of steps that are
           satisfied together with every transitive prerequisite in the
           item's checklist dependency DAG.

Five details that are easy to get wrong and do move the numbers:

  1. A failed run scores zero on all four, not "no data". Failing to answer is
     a result.
  2. A missing verdict also scores zero, for the same reason.
  3. If the judge returns fewer flags than the checklist has steps, the missing
     ones are False. A judge that stops early has not credited those steps.
  4. Extra flags beyond the checklist length are dropped.
  5. Every non-empty checklist must include one checklist_parents entry per
     step. Missing dependency graphs are rejected rather than scored silently.
"""
from __future__ import annotations

from .dag import ancestor_sets, dacs, normalize
from .schema import Item, Response, Score, Verdict

METRICS = ("OA", "SA", "CS", "DACS")


def _normalise(completion: list[bool] | None, checklist_len: int) -> list[bool]:
    """Truncate/pad the judge's flags to exactly checklist_len (details 3 & 4)."""
    flags = list(completion or [])[:checklist_len]
    flags.extend([False] * (checklist_len - len(flags)))
    return flags


def score_item(item: Item, verdict: Verdict | None,
               response: Response | None = None) -> Score:
    """Score one item. A missing/failed response or verdict yields all zeros."""
    failed = response is not None and not response.ok
    if verdict is None or failed:
        return Score(item.id, 0.0, 0.0, 0.0, 0.0)

    denom = len(item.checklist)
    flags = _normalise(verdict.checklist_completion, denom)
    completed = sum(1 for v in flags if v)
    ancestors = ancestor_sets(normalize(item.checklist_parents, denom))

    oa = 1.0 if int(verdict.is_correct or 0) == 1 else 0.0
    cs = completed / denom if denom else 0.0
    dependency_score = dacs(ancestors, flags)
    sa = 1.0 if (oa and denom > 0 and completed == denom) else 0.0
    return Score(item.id, oa, sa, cs, dependency_score)


def score_all(items: list[Item], verdicts: dict[str, Verdict],
              responses: dict[str, Response] | None = None) -> list[Score]:
    """Score every item. Items absent from `verdicts` score zero (detail 2)."""
    responses = responses or {}
    return [score_item(it, verdicts.get(it.id), responses.get(it.id))
            for it in items]


def aggregate(scores: list[Score]) -> dict[str, float]:
    """Mean of each metric over items, as a percentage."""
    n = len(scores)
    if not n:
        return {m: 0.0 for m in METRICS}
    return {
        "OA": 100.0 * sum(s.oa for s in scores) / n,
        "SA": 100.0 * sum(s.sa for s in scores) / n,
        "CS": 100.0 * sum(s.cs for s in scores) / n,
        "DACS": 100.0 * sum(s.dacs for s in scores) / n,
    }
