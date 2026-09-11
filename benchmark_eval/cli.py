"""Command line: benchmark-eval run | judge | score.

The three stages are separate commands on purpose. Model runs are the
expensive part, so they are done once and written to disk; re-judging or
re-scoring them costs nothing and needs no new model calls. This is exactly
how the paper's 12-judge agreement study was produced from a single set of
frozen responses.

    benchmark-eval run   --items data/metadata.jsonl --model openai/gpt-5.5 \
                    --openrouter --out runs/gpt55.jsonl
    benchmark-eval judge --items data/metadata.jsonl --responses runs/gpt55.jsonl \
                    --out runs/gpt55.verdicts.jsonl
    benchmark-eval score --items data/metadata.jsonl --verdicts runs/gpt55.verdicts.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict
from pathlib import Path

from .client import (OPENAI_WEB_SEARCH, OPENROUTER, OPENROUTER_WEB_SEARCH,
                     Endpoint, env_endpoint)
from .crypto import decrypt_row
from .judge import JudgeSpec, env_judge
from .metrics import METRICS, aggregate, score_all
from .runner import judge_responses, run_items
from .schema import (Item, Response, Verdict, load_items, read_jsonl,
                     write_jsonl)


def _progress(label: str):
    def fn(done: int, total: int, value) -> None:
        bad = getattr(value, "error", None)
        end = "\n" if (done == total or bad) else "\r"
        note = f"  last error: {bad[:80]}" if bad else ""
        print(f"  {label} {done}/{total}{note}", end=end, flush=True)
    return fn


def _endpoint(args) -> Endpoint:
    if args.openrouter:
        return Endpoint(OPENROUTER.base_url, OPENROUTER.api_key_env, proxy=args.proxy)
    if args.base_url:
        return Endpoint(args.base_url.rstrip("/"), args.api_key_env, proxy=args.proxy)
    ep = env_endpoint()     # EVAL_BASE_URL / EVAL_API_KEY / EVAL_PROXY
    return Endpoint(ep.base_url, ep.api_key_env, proxy=args.proxy or ep.proxy)


def _download_hf_dataset(repo_id: str) -> Path:
    """Fetch a released dataset repo (metadata.jsonl + images/) via the HF hub cache.

    Kept optional: --items already covers "I have the files locally" (a plain
    git clone or `hf download` works with zero code here), so
    this is only exercised -- and huggingface_hub only imported -- when a
    caller opts into --hf-repo.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise EnvironmentError(
            "huggingface_hub is required for --hf-repo. Install with: "
            "pip install 'benchmark-eval[hf]'"
        )
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


def _judge_spec(args) -> JudgeSpec:
    if args.judge_base_url:
        return JudgeSpec(model=args.judge_model, base_url=args.judge_base_url.rstrip("/"),
                         api_key_env=args.judge_api_key_env, api=args.judge_api,
                         label=args.judge_label or "", proxy=args.judge_proxy)
    spec = env_judge()      # EVAL_JUDGE_*
    if args.judge_proxy:
        spec = JudgeSpec(spec.model, spec.base_url, spec.api_key_env, spec.api,
                         spec.label, args.judge_proxy)
    return spec


# --------------------------------------------------------------------------

def cmd_decrypt(args) -> int:
    """Write a plaintext copy of a dataset jsonl. `run`/`judge`/`score`
    already decrypt items transparently -- this is only for producing a
    human-readable file to inspect or hand to other tools."""
    rows = [decrypt_row(r) for r in read_jsonl(args.items)]
    write_jsonl(args.out, rows)
    print(f"decrypted {len(rows)} rows -> {args.out}")
    return 0


def cmd_run(args) -> int:
    if bool(args.items) == bool(args.hf_repo):
        print("error: pass exactly one of --items or --hf-repo", file=sys.stderr)
        return 2

    image_root = args.image_root
    if args.hf_repo:
        local_dir = _download_hf_dataset(args.hf_repo)
        items_path = local_dir / "metadata.jsonl"
        image_root = image_root or local_dir
        # judge/score reuse this same metadata.jsonl -- print it so it's easy to reuse.
        print(f"downloaded {args.hf_repo} -> {items_path}")
    else:
        items_path = Path(args.items)

    items = load_items(items_path)
    if args.limit:
        items = items[:args.limit]

    out = Path(args.out)
    done: dict[str, dict] = {}
    if args.resume and out.exists():
        # Keep only successful rows: a failed row is worth retrying, and
        # resuming over it would freeze a transient network error into the run.
        done = {r["item_id"]: r for r in read_jsonl(out) if not r.get("error")}
        items = [it for it in items if it.id not in done]
        print(f"resuming: {len(done)} already answered, {len(items)} to go")

    tools = None
    if args.web_search:
        tools = OPENROUTER_WEB_SEARCH if args.openrouter else OPENAI_WEB_SEARCH

    if not items:
        print("nothing to do")
        return 0

    print(f"model={args.model}  items={len(items)}  "
          f"images={'off' if args.no_images else 'on'}  "
          f"web_search={'on' if args.web_search else 'off'}")
    responses = asyncio.run(run_items(
        items, args.model, _endpoint(args),
        image_root=image_root or items_path.parent,
        with_images=not args.no_images, tools=tools,
        concurrency=args.concurrency, timeout=args.timeout,
        on_progress=_progress("answered"),
    ))

    rows = list(done.values()) + [asdict(r) for r in responses]
    write_jsonl(out, rows)
    ok = sum(1 for r in responses if r.ok)
    print(f"wrote {len(rows)} rows to {out}  (this pass: ok={ok} "
          f"err={len(responses) - ok})")
    return 0


def cmd_judge(args) -> int:
    items = load_items(args.items)
    responses = [Response(**r) for r in read_jsonl(args.responses)]
    spec = _judge_spec(args)

    out = Path(args.out)
    done: dict[str, dict] = {}
    if args.resume and out.exists():
        done = {r["item_id"]: r for r in read_jsonl(out)
                if r.get("judge_id") == spec.judge_id and not r.get("error")}
        responses = [r for r in responses if r.item_id not in done]
        print(f"resuming: {len(done)} already judged, {len(responses)} to go")

    if not responses:
        print("nothing to do")
        return 0

    print(f"judge={spec.judge_id}  responses={len(responses)}")
    verdicts = asyncio.run(judge_responses(
        items, responses, spec, concurrency=args.concurrency,
        timeout=args.timeout, on_progress=_progress("judged"),
    ))

    rows = list(done.values()) + [asdict(v) for v in verdicts]
    write_jsonl(out, rows)
    ok = sum(1 for v in verdicts if v.ok)
    print(f"wrote {len(rows)} rows to {out}  (this pass: ok={ok} "
          f"err={len(verdicts) - ok})")
    return 0


def cmd_score(args) -> int:
    items = load_items(args.items)
    verdicts = {}
    for r in read_jsonl(args.verdicts):
        v = Verdict(**r)
        # Later rows win, but never let a failed re-judge overwrite a good one.
        if v.ok or v.item_id not in verdicts:
            verdicts[v.item_id] = v

    responses = {}
    if args.responses:
        responses = {r["item_id"]: Response(**r) for r in read_jsonl(args.responses)}

    scores = score_all(items, verdicts, responses)
    agg = aggregate(scores)

    print(f"items={len(items)}  judged={sum(1 for v in verdicts.values() if v.ok)}")
    print("  ".join(f"{m}={agg[m]:.1f}" for m in METRICS))

    if args.per_item:
        write_jsonl(args.per_item, [asdict(s) for s in scores])
        print(f"per-item scores -> {args.per_item}")
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmark-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def endpoint_args(p, prefix="", default_key="EVAL_API_KEY"):
        p.add_argument(f"--{prefix}base-url", default=None,
                       help="OpenAI-compatible endpoint (default: from env)")
        p.add_argument(f"--{prefix}api-key-env", default=default_key,
                       help=f"env var holding the API key (default: {default_key})")
        p.add_argument(f"--{prefix}proxy", default=None,
                       help="proxy for this endpoint, e.g. socks5://127.0.0.1:1055; "
                            "setting it also ignores HTTP_PROXY/ALL_PROXY")

    r = sub.add_parser("run", help="answer items with a model")
    r.add_argument("--items", default=None, help="dataset jsonl (mutually exclusive with --hf-repo)")
    r.add_argument("--hf-repo", default=None,
                   help="HuggingFace dataset repo id, e.g. Henryeahhh/Mr-LHDR -- "
                        "downloaded via the hub cache and used in place of --items "
                        "(mutually exclusive with --items; needs 'pip install benchmark-eval[hf]')")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--openrouter", action="store_true",
                   help="use OpenRouter (OPENROUTER_API_KEY)")
    endpoint_args(r)
    r.add_argument("--image-root", default=None,
                   help="directory the item image paths are relative to "
                        "(default: the dataset file's directory, or the "
                        "downloaded --hf-repo directory)")
    r.add_argument("--no-images", action="store_true",
                   help="withhold images -- the text-only ablation")
    r.add_argument("--web-search", action="store_true",
                   help="attach the server-side web search tool")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--resume", action="store_true",
                   help="skip items already answered in --out")
    r.set_defaults(fn=cmd_run)

    j = sub.add_parser("judge", help="judge stored responses")
    j.add_argument("--items", required=True)
    j.add_argument("--responses", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--judge-model", default=None)
    j.add_argument("--judge-base-url", default=None)
    j.add_argument("--judge-api-key-env", default="EVAL_JUDGE_API_KEY")
    j.add_argument("--judge-api", default="openai", choices=["openai", "anthropic"],
                   help="wire format. Note: base-url ends in /v1 for openai, "
                        "but must NOT for anthropic")
    j.add_argument("--judge-proxy", default=None,
                   help="proxy for the judge endpoint; also ignores HTTP_PROXY/ALL_PROXY")
    j.add_argument("--judge-label", default=None,
                   help="id to record instead of the model id")
    j.add_argument("--concurrency", type=int, default=8)
    j.add_argument("--timeout", type=float, default=600.0)
    j.add_argument("--resume", action="store_true")
    j.set_defaults(fn=cmd_judge)

    s = sub.add_parser("score", help="compute benchmark metrics from verdicts")
    s.add_argument("--items", required=True)
    s.add_argument("--verdicts", required=True)
    s.add_argument("--responses", default=None,
                   help="optional: lets failed runs score zero explicitly")
    s.add_argument("--per-item", default=None, help="write per-item scores here")
    s.set_defaults(fn=cmd_score)

    d = sub.add_parser("decrypt", help="write a plaintext copy of a dataset jsonl")
    d.add_argument("--items", required=True,
                   help="dataset jsonl to decrypt, e.g. data/metadata.jsonl")
    d.add_argument("--out", required=True, help="path to write the plaintext jsonl")
    d.set_defaults(fn=cmd_decrypt)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except EnvironmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
