"""Orchestration: answer items, then judge the answers.

THE ANSWERING PROTOCOL, which matters as much as the judge prompt:

  * the prompt is the item's question verbatim -- no system prompt, no
    instructions about format, length, or tool use. Systems are compared on
    what they do with the question as a user would ask it.
  * images are attached as-is when the item has them.
  * nothing is retried at this level. A model that fails to answer scores
    zero, because failing to answer is a result (see metrics.py).

Everything here is a pure function over the data types: no database, no files.
Callers decide what to persist, which is what lets the same code back a CLI, a
notebook, and a service.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from . import client as _client
from . import judge as _judge
from .client import Endpoint
from .judge import JudgeSpec
from .schema import Item, Response, Verdict

ProgressFn = Callable[[int, int, Any], None]


async def _gather(tasks: list, on_progress: ProgressFn | None) -> list:
    """Run tasks concurrently, reporting completions as they land.

    Results come back in task order, not completion order, so a run is
    reproducible regardless of which call happened to be slow.
    """
    if on_progress is None:
        return await asyncio.gather(*tasks)
    total, done, results = len(tasks), 0, {}
    for coro in asyncio.as_completed([_indexed(i, t) for i, t in enumerate(tasks)]):
        i, value = await coro
        results[i] = value
        done += 1
        on_progress(done, total, value)
    return [results[i] for i in range(total)]


async def _indexed(i: int, task):
    return i, await task


async def run_items(
    items: list[Item],
    model: str,
    endpoint: Endpoint,
    *,
    image_root: str | Path | None = None,
    with_images: bool = True,
    tools: list[dict[str, Any]] | None = None,
    concurrency: int = 8,
    timeout: float = 600.0,
    proxy: str | None = None,
    on_progress: ProgressFn | None = None,
) -> list[Response]:
    """Answer every item once.

    `with_images=False` withholds the images while keeping everything else
    fixed -- the text-only ablation that tests whether the image is needed.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(item: Item) -> Response:
        async with sem:
            try:
                text = await _client.call(
                    model, item.question, endpoint,
                    images=item.images if with_images else None,
                    image_root=image_root, tools=tools,
                    timeout=timeout, proxy=proxy,
                )
                return Response(item.id, model, text=text, with_images=with_images)
            except Exception as exc:  # noqa: BLE001 -- a failed run is a result
                return Response(item.id, model, error=f"{type(exc).__name__}: {exc}",
                                with_images=with_images)

    return await _gather([one(it) for it in items], on_progress)


async def judge_responses(
    items: list[Item],
    responses: list[Response],
    spec: JudgeSpec,
    *,
    concurrency: int = 8,
    timeout: float = 600.0,
    on_progress: ProgressFn | None = None,
) -> list[Verdict]:
    """Judge every answered response. Failed runs are not sent to the judge."""
    by_id = {it.id: it for it in items}
    pending = [r for r in responses if r.ok and r.item_id in by_id]

    sem = asyncio.Semaphore(concurrency)

    async def one(resp: Response) -> Verdict:
        async with sem:
            return await _judge.run(by_id[resp.item_id], resp.text or "",
                                    spec, timeout=timeout)

    return await _gather([one(r) for r in pending], on_progress)
