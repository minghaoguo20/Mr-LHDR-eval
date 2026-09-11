#!/usr/bin/env python3
"""Aggregate OA/SA/CS/DACS from verdicts. Runs straight from a clone -- no
`pip install` of this package needed, just the two deps in requirements.txt
(score.py does no network calls, but `benchmark_eval.cli` imports openai/httpx
at module level regardless).

    python scripts/score.py --items data/metadata.jsonl --verdicts runs/gpt55.verdicts.jsonl \
                    --responses runs/gpt55.jsonl

All flags are the same as `benchmark-eval score`; see --help or README.md.
"""
import sys
from pathlib import Path

# Make the local `benchmark_eval/` package importable without installing it.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import httpx  # noqa: F401
    import openai  # noqa: F401
except ImportError:
    sys.exit(f"Missing dependencies. Install them with:\n"
             f"  pip install -r {_ROOT / 'requirements.txt'}")

from benchmark_eval.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["score", *sys.argv[1:]]))
