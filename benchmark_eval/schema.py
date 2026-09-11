"""Data types and the on-disk format.

An evaluation is four objects flowing in one direction:

    Item  --(model)-->  Response  --(judge)-->  Verdict  --(metrics)-->  Score

Every stage is written to jsonl so it can be inspected, resumed, or replaced.
Re-judging a stored Response with a different judge is the whole point of
keeping the stages separate -- it costs one judge call instead of one model run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .crypto import decrypt_row


@dataclass(frozen=True)
class Item:
    """One benchmark question.

    `images` are paths relative to the dataset directory (as exported for
    HuggingFace, e.g. "images/abc.png"), or absolute URLs / data: URIs, which
    are passed through untouched.
    """
    id: str
    question: str
    answer: list[str] = field(default_factory=list)     # reference answer lines
    checklist: list[str] = field(default_factory=list)
    checklist_parents: list[list[int]] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    category: str = ""
    type: str = ""

    def __post_init__(self) -> None:
        if len(self.checklist_parents) != len(self.checklist):
            if self.checklist and not self.checklist_parents:
                raise ValueError(
                    f"Item {self.id!r} has a non-empty checklist but no "
                    "checklist_parents; DACS requires an explicit "
                    "dependency graph"
                )
            raise ValueError(
                f"Item {self.id!r} has {len(self.checklist)} checklist steps "
                f"but {len(self.checklist_parents)} parent lists"
            )

    @property
    def reference(self) -> str:
        """The reference answer as the judge receives it."""
        return "\n".join(self.answer)

    @classmethod
    def from_dict(cls, d: dict) -> Item:
        # Unknown keys (sources, subtasks, *_zh) are ignored rather than
        # rejected: the dataset carries annotation fields the evaluator does
        # not need, and a stricter reader would break on every export change.
        # A `canary` field means question/answer/checklist are ciphertext
        # (see crypto.py); decrypt_row is a no-op otherwise.
        d = decrypt_row(d)
        raw_parents = d.get("checklist_parents") or []
        return cls(
            id=d["id"],
            question=d["question"],
            answer=list(d.get("answer") or []),
            checklist=list(d.get("checklist") or []),
            checklist_parents=[
                list(parents) if isinstance(parents, list) else []
                for parents in raw_parents
            ],
            images=list(d.get("images") or []),
            category=d.get("category", ""),
            type=d.get("type", ""),
        )


@dataclass(frozen=True)
class Response:
    """What a system produced for one item. `error` set means the run failed."""
    item_id: str
    model: str
    text: str | None = None
    error: str | None = None
    with_images: bool = True

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


@dataclass(frozen=True)
class Verdict:
    """One judge's reading of one Response.

    `is_correct` is the final-answer call; `checklist_completion` is one bool
    per checklist step, in order. Either may be None if the judge failed or
    returned something unparseable -- which is recorded, not silently scored
    as zero, so it can be retried.
    """
    item_id: str
    judge_id: str
    is_correct: int | None = None
    checklist_completion: list[bool] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and (
            self.is_correct is not None or self.checklist_completion is not None)


@dataclass(frozen=True)
class Score:
    """Benchmark metrics for one item, each in [0, 1]. See metrics.py for meaning."""
    item_id: str
    oa: float
    sa: float
    cs: float
    dacs: float


# --------------------------------------------------------------------------
# jsonl I/O
# --------------------------------------------------------------------------

def load_items(path: str | Path) -> list[Item]:
    """Read a dataset jsonl (one item per line), e.g. metadata.jsonl."""
    with Path(path).open(encoding="utf-8") as f:
        return [Item.from_dict(json.loads(line)) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
