"""benchmark-eval — the evaluator for our benchmark.

Four stages, each replaceable:

    load_items  ->  run_items  ->  judge_responses  ->  score_all / aggregate

Two of them are fixed definitions and must not be edited casually: `judge.py`
(how an answer is scored) and `metrics.py` (what OA/SA/CS/DACS mean). The
rest is glue you are welcome to replace.
"""
from .client import (OPENAI_WEB_SEARCH, OPENROUTER, OPENROUTER_WEB_SEARCH,
                     Endpoint, env_endpoint)
from .dag import dacs
from .judge import JudgeSpec, env_judge
from .judge import run as judge_one
from .metrics import METRICS, aggregate, score_all, score_item
from .runner import judge_responses, run_items
from .schema import Item, Response, Score, Verdict, load_items

__version__ = "0.3.0"

__all__ = [
    "Item", "Response", "Verdict", "Score", "load_items",
    "Endpoint", "env_endpoint", "OPENROUTER",
    "OPENROUTER_WEB_SEARCH", "OPENAI_WEB_SEARCH",
    "JudgeSpec", "env_judge", "judge_one",
    "run_items", "judge_responses",
    "score_item", "score_all", "aggregate", "dacs", "METRICS",
    "__version__",
]
