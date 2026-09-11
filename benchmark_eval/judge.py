"""LLM judge: scores one answer against the reference answer and checklist.

FIXED DEFINITION, like metrics.py. The prompt and its parsing are identical
for every judge, which is what makes judges comparable and swappable: only
*which model* judges is configurable, never *how* it is asked. The paper's
judge-agreement study (12 judges, 6 vendor families) is only meaningful
because this file did not change between them.

Two wire formats are supported so any mainstream endpoint can judge:

  api="openai"     Chat Completions -- OpenAI, vLLM, Ollama, DashScope, ...
  api="anthropic"  /v1/messages -- Claude and Claude-compatible gateways

Both send exactly the same prompt.

    from benchmark_eval.judge import JudgeSpec, run
    spec = JudgeSpec(model="gpt-5.5", base_url="https://api.openai.com/v1",
                     api_key_env="OPENAI_API_KEY")
    verdict = await run(item, response_text, spec=spec)
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from .schema import Item, Verdict

# The verdict itself is ~50 tokens, but judges that reason internally spend the
# budget on thinking first: at 1024 several models returned an EMPTY text block
# with stop_reason='end_turn' (truncation is not reported as such), which parses
# into a null verdict. This is an upper bound only -- it does not change what a
# judge outputs.
_MAX_TOKENS = 8192

# Gateways return sporadic 5xx under no particular load; retry those. The OpenAI
# SDK already retries internally, so this only wraps the anthropic path.
_RETRY_ATTEMPTS = 4
_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# Long answers are truncated from the left: the tail carries the conclusion,
# which is what the checklist asks about.
_MAX_ANSWER_CHARS = 25000


@dataclass(frozen=True)
class JudgeSpec:
    model: str            # model id sent in the request
    base_url: str         # api="openai": must end in /v1. api="anthropic": must NOT
    api_key_env: str      # env var holding the API key
    api: str = "openai"   # "openai" | "anthropic"
    label: str = ""       # id recorded in the verdict; defaults to model
    proxy: str | None = None   # set it to also ignore HTTP_PROXY/ALL_PROXY

    @property
    def judge_id(self) -> str:
        return self.label or self.model


def env_judge(prefix: str = "EVAL_JUDGE") -> JudgeSpec:
    """Judge configured entirely from the environment.

        EVAL_JUDGE_BASE_URL   e.g. http://localhost:8000/v1
        EVAL_JUDGE_MODEL      served model id
        EVAL_JUDGE_API_KEY    key (any non-empty string for a local server)
        EVAL_JUDGE_API        "openai" (default) or "anthropic"
        EVAL_JUDGE_LABEL      optional id to record instead of the model id
        EVAL_JUDGE_PROXY      optional; also disables the ambient proxy vars

    This is how a self-hosted endpoint (vLLM, Ollama, SGLang) is used as the
    judge without touching code.

    Note the asymmetry between the two wire formats, which is a common source
    of 404s: the OpenAI path needs a base_url ending in /v1 (appended here if
    you leave it off), while the Anthropic path must NOT have it -- "/v1/messages"
    is appended at call time.
    """
    base = (os.environ.get(f"{prefix}_BASE_URL") or "").rstrip("/")
    if not base:
        raise EnvironmentError(f"{prefix}_BASE_URL is not set")
    api = os.environ.get(f"{prefix}_API", "openai")
    if api == "openai" and not base.endswith("/v1"):
        base += "/v1"
    model = os.environ.get(f"{prefix}_MODEL")
    if not model:
        raise EnvironmentError(f"{prefix}_MODEL is not set")
    return JudgeSpec(
        model=model,
        base_url=base,
        api_key_env=f"{prefix}_API_KEY",
        api=api,
        label=os.environ.get(f"{prefix}_LABEL") or "",
        proxy=os.environ.get(f"{prefix}_PROXY") or None,
    )


def _api_key(spec: JudgeSpec) -> str:
    key = os.environ.get(spec.api_key_env)
    if not key:
        raise EnvironmentError(f"Environment variable {spec.api_key_env} is not set")
    return key


def build_prompt(question: str, generated_answer: str, reference_answer: str,
                 checklist: list[str]) -> str:
    """The scoring prompt. Do not edit -- see the module docstring."""
    generated_answer = generated_answer[-_MAX_ANSWER_CHARS:]
    n = len(checklist)
    checklist_lines = "\n".join(f"{i+1}. {item}" for i, item in enumerate(checklist))
    example_score = f"{max(0, n - 1)}/{n}"

    return f"""You are an AI evaluator. Your task is to evaluate the quality of an answer.
I will provide you with the user's question, the reference answer (ground truth), a checklist, and the answer to be evaluated.

--- USER QUESTION ---
{question}

--- REFERENCE ANSWER (Ground Truth) ---
{reference_answer}
This reference answer is considered the correct and ideal response content-wise.

--- REFERENCE CHECKLIST ---
{checklist_lines}

--- MODEL'S GENERATED ANSWER TO EVALUATE ---
{generated_answer}

--- EVALUATION INSTRUCTIONS ---
Please provide your evaluation strictly in the following format on separate lines:
1. Checklist Score: Determine how many of the {n} items in the REFERENCE CHECKLIST have been correctly and completely addressed. The model's generated answer must fully comply with an item for it to be considered complete. State as 'CHECKLIST_SCORE: [correct_items]/{n}' (e.g., CHECKLIST_SCORE: {example_score}).
2. Checklist Result Vector: Provide a 0-1 vector indicating whether each checklist item passed, in order. '1' means fully satisfied, '0' means not fully satisfied. Output as 'CHECKLIST_RESULT: [1,0,1]'.
3. Overall Correctness: Judge whether the generated answer is consistent with the reference answer in core content. Content consistency is key; minor wording differences are acceptable. State as 'OVERALL_CORRECTNESS: [YES/NO]'.

Provide only these three formatted lines as your response.
"""


def parse_response(text: str) -> tuple[int | None, list[bool] | None]:
    """Extract (is_correct, checklist_completion) from the judge's reply."""
    if not text:
        return None, None

    is_correct = None
    m = re.search(r"OVERALL_CORRECTNESS:\s*(YES|NO)", text, re.IGNORECASE)
    if m:
        is_correct = 1 if m.group(1).upper() == "YES" else 0

    completion = None
    vec = re.search(r"CHECKLIST_RESULT:\s*\[([01,\s]+)\]", text, re.IGNORECASE)
    if vec:
        completion = [bool(int(x)) for x in re.findall(r"[01]", vec.group(1))]

    return is_correct, completion


async def _call_openai(spec: JudgeSpec, prompt: str, timeout: float) -> str:
    async with httpx.AsyncClient(proxy=spec.proxy, trust_env=spec.proxy is None,
                                 timeout=timeout) as http:
        client = AsyncOpenAI(api_key=_api_key(spec), base_url=spec.base_url,
                             http_client=http)
        resp = await client.chat.completions.create(
            model=spec.model,
            messages=[{"role": "user", "content": prompt}],
        )
    return resp.choices[0].message.content or ""


async def _call_anthropic(spec: JudgeSpec, prompt: str, timeout: float) -> str:
    """Anthropic /v1/messages over plain httpx.

    Not the anthropic SDK: gateways in this format commonly authenticate with
    `Authorization: Bearer`, not the SDK's `x-api-key` header.
    """
    payload = {
        "model": spec.model,
        "max_tokens": _MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {_api_key(spec)}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(proxy=spec.proxy, trust_env=spec.proxy is None,
                                 timeout=timeout) as http:
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                resp = await http.post(spec.base_url.rstrip("/") + "/v1/messages",
                                       headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRY_STATUS or attempt == _RETRY_ATTEMPTS:
                    raise
            except httpx.TransportError:  # connect/read timeouts, dropped sockets
                if attempt == _RETRY_ATTEMPTS:
                    raise
            else:
                blocks = resp.json().get("content", [])
                return "".join(b.get("text", "") for b in blocks
                               if b.get("type") == "text")
            await asyncio.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError("unreachable")  # the loop either returns or raises


async def run(item: Item, answer: str, spec: JudgeSpec,
              timeout: float = 600.0) -> Verdict:
    """Judge one answer. Never raises: failures come back as Verdict.error."""
    prompt = build_prompt(item.question, answer, item.reference, item.checklist)
    call = _call_anthropic if spec.api == "anthropic" else _call_openai
    try:
        text = await call(spec, prompt, timeout)
    except Exception as exc:  # noqa: BLE001 -- recorded, retried by the caller
        return Verdict(item.id, spec.judge_id,
                       error=f"{type(exc).__name__}: {exc}")

    is_correct, completion = parse_response(text)
    if is_correct is None and completion is None:
        # The call succeeded but nothing parseable came back (e.g. a judge that
        # spent its whole budget thinking). Recorded as an error so it can be
        # retried, rather than landing as a silent zero.
        return Verdict(item.id, spec.judge_id, error="unparseable_judge_output")
    return Verdict(item.id, spec.judge_id, is_correct=is_correct,
                   checklist_completion=completion)
