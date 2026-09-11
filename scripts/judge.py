#!/usr/bin/env python3
"""Judge stored responses with an LLM judge. Runs straight from a clone --
no `pip install` of this package needed, just the two deps in requirements.txt.

    python scripts/judge.py --items data/metadata.jsonl --responses runs/gpt55.jsonl \
                    --out runs/gpt55.verdicts.jsonl

All flags are the same as `benchmark-eval judge`; see --help or README.md.
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
    raise SystemExit(main(["judge", *sys.argv[1:]]))
