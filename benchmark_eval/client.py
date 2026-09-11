"""Calling a model through any OpenAI-compatible Chat Completions endpoint.

One code path covers everything the paper evaluated except two self-hosted
frameworks: OpenRouter, OpenAI, DashScope, and any local server that speaks
Chat Completions (vLLM, SGLang, Ollama, llama.cpp).

Deliberately thin. It does not know which models exist or which support
search -- those are the caller's business, passed in as arguments. That is
what keeps this file from rotting as models come and go.

Proxies: setting `proxy` on the Endpoint also switches OFF httpx's use of
HTTP_PROXY / HTTPS_PROXY / ALL_PROXY. That is deliberate and matters for
private endpoints: if the machine exports a general-purpose proxy, an endpoint
reachable only through a *different* hop (a VPN's own SOCKS5, say) would
otherwise have its traffic quietly handed to the wrong proxy, which typically
answers 502. Leave `proxy` unset to keep the standard environment behaviour.
"""
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

# Server-side web search, as used for the paper's tool-augmented group. Pass it
# as `tools=` when reproducing those rows; see README for which models had it.
OPENROUTER_WEB_SEARCH: list[dict[str, Any]] = [
    {"type": "openrouter:web_search", "parameters": {"engine": "native"}}
]
OPENAI_WEB_SEARCH: list[dict[str, Any]] = [{"type": "web_search"}]


@dataclass(frozen=True)
class Endpoint:
    """Where to send requests, which env var holds the key, how to reach it."""
    base_url: str
    api_key_env: str = "EVAL_API_KEY"
    proxy: str | None = None       # e.g. socks5://127.0.0.1:1055; see module docstring

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise EnvironmentError(f"Environment variable {self.api_key_env} is not set")
        return key


OPENROUTER = Endpoint("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")


def env_endpoint(prefix: str = "EVAL") -> Endpoint:
    """Endpoint from the environment -- the way to point at a self-hosted server.

        EVAL_BASE_URL   e.g. http://localhost:8000/v1  (/v1 appended if absent)
        EVAL_API_KEY    key; any non-empty string for a server that ignores it
        EVAL_PROXY      optional; also disables the ambient proxy env vars

    Prefer a hostname over an IP when the endpoint is behind a VPN: addresses
    are often reassigned when a node re-registers, hostnames are not. With a
    SOCKS5 proxy the hostname is resolved at the far end, so it works even when
    the local resolver knows nothing about it.
    """
    base = (os.environ.get(f"{prefix}_BASE_URL") or "").rstrip("/")
    if not base:
        raise EnvironmentError(f"{prefix}_BASE_URL is not set")
    if not base.endswith("/v1"):
        base += "/v1"
    return Endpoint(base, f"{prefix}_API_KEY",
                    proxy=os.environ.get(f"{prefix}_PROXY") or None)


def image_url(ref: str, image_root: str | Path | None = None) -> str:
    """Resolve one image reference to something the API accepts.

    Absolute URLs and data: URIs pass through. Anything else is a path
    relative to `image_root` (the dataset directory) and is inlined as base64,
    which avoids requiring the images to be publicly hosted.
    """
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(image_root or ".") / ref
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _content(prompt: str, images: list[str] | None,
             image_root: str | Path | None) -> Any:
    if not images:
        return prompt
    return [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": image_url(ref, image_root)}}
        for ref in images
    ]


async def call(
    model: str,
    prompt: str,
    endpoint: Endpoint,
    *,
    images: list[str] | None = None,
    image_root: str | Path | None = None,
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 600.0,
    proxy: str | None = None,
) -> str:
    """Send one prompt (optionally with images) and return the text reply.

    `proxy` overrides the endpoint's own setting; either one disables the
    ambient HTTP_PROXY/ALL_PROXY variables.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user",
                      "content": _content(prompt, images, image_root)}],
    }
    if tools:
        kwargs["tools"] = tools

    proxy = proxy or endpoint.proxy
    async with httpx.AsyncClient(proxy=proxy, trust_env=proxy is None,
                                 timeout=timeout) as http:
        client = AsyncOpenAI(api_key=endpoint.api_key,
                             base_url=endpoint.base_url, http_client=http)
        resp = await client.chat.completions.create(**kwargs)

    return resp.choices[0].message.content or ""
