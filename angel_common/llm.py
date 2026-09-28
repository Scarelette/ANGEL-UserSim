"""LLM API clients shared by data generation, training rewards and experiments.

All endpoints, deployments and keys come from environment variables (see
``.env.example``). The paper's experiments used Azure OpenAI deployments and
Anthropic models served through Azure AI Foundry; plain OpenAI / Anthropic
endpoints work too by setting the variables accordingly.

Environment variables
---------------------
AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
    Azure OpenAI resource used for GPT-5 / GPT-4.1 / GPT-4o calls.
ANGEL_GPT5_DEPLOYMENT
    Deployment name of the GPT-5 model (default ``gpt-5``).
ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
    Anthropic key. If ``ANTHROPIC_BASE_URL`` points at an Azure AI Foundry
    ``.../anthropic/`` endpoint, the Foundry client is used.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional

from angel_common.env import get_env, require_env

DEFAULT_API_VERSION = "2024-12-01-preview"


# --------------------------------------------------------------------------- #
# Azure OpenAI
# --------------------------------------------------------------------------- #
def azure_openai_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    async_client: bool = False,
):
    """Build an (Async)AzureOpenAI client from arguments or environment."""
    from openai import AsyncAzureOpenAI, AzureOpenAI

    cls = AsyncAzureOpenAI if async_client else AzureOpenAI
    return cls(
        azure_endpoint=endpoint or require_env("AZURE_OPENAI_ENDPOINT", purpose="Azure OpenAI"),
        api_key=api_key or require_env("AZURE_OPENAI_API_KEY", purpose="Azure OpenAI"),
        api_version=api_version or get_env("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
    )


def gpt5_deployment() -> str:
    return get_env("ANGEL_GPT5_DEPLOYMENT", "gpt-5")


_INVALID_PATTERNS = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "no answer",
    "i cannot answer",
    "i can’t answer",
    "sorry",
    "unable to",
    "as an ai",
    "mitigat",
)


def is_invalid_answer(text: Optional[str]) -> bool:
    if text is None:
        return True
    lower = text.lower().strip()
    return any(p in lower for p in _INVALID_PATTERNS)


_CLIENT_CACHE: Dict[str, Any] = {}


def get_output(
    cur_prompt: str,
    context: Optional[List[Dict[str, str]]] = None,
    max_completion_tokens: int = 512,
    tag: int = 1,
    max_retries: int = 8,
    sleep_seconds: float = 2,
    on_error: Optional[Callable[[Exception], None]] = None,
    reasoning_effort: Optional[str] = None,
    deployment: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Single-turn GPT-5 call with retries (the original ``getOutput``).

    ``tag=1`` additionally requires the answer to contain an XML-style tag and
    rejects refusal-like answers. Returns ``""`` after ``max_retries`` failures.
    """
    client = _CLIENT_CACHE.get("azure")
    if client is None:
        client = _CLIENT_CACHE["azure"] = azure_openai_client()

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": cur_prompt})

    # GPT-5 counts reasoning tokens against max_completion_tokens; too small a
    # budget returns empty content, so keep a 6000-token floor.
    create_kwargs: Dict[str, Any] = dict(
        messages=messages,
        max_completion_tokens=max(int(max_completion_tokens), 6000),
        model=deployment or gpt5_deployment(),
        temperature=1,
    )
    if reasoning_effort is not None:
        create_kwargs["reasoning_effort"] = reasoning_effort

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**create_kwargs)
        except Exception as e:  # network / rate-limit / content-filter
            if on_error is not None:
                try:
                    on_error(e)
                except Exception:
                    pass
            print(f"[Attempt {attempt + 1}] API Error: {e}")
            time.sleep(sleep_seconds)
            continue

        if not response or not response.choices:
            time.sleep(sleep_seconds)
            continue
        content = response.choices[0].message.content
        if not content:
            print(f"[Attempt {attempt + 1}] empty content "
                  f"(finish_reason={response.choices[0].finish_reason}); retrying")
            time.sleep(sleep_seconds)
            continue
        if tag == 1 and (is_invalid_answer(content) or "<" not in content or ">" not in content):
            print(f"[Attempt {attempt + 1}] invalid/untagged answer; retrying")
            time.sleep(sleep_seconds)
            continue
        return content
    return ""


# Backwards-compatible alias for code ported from the research repos.
getOutput = get_output


# --------------------------------------------------------------------------- #
# Anthropic (direct or Azure AI Foundry)
# --------------------------------------------------------------------------- #
def anthropic_client(api_key: Optional[str] = None, base_url: Optional[str] = None, async_client: bool = False):
    """Anthropic client; uses AnthropicFoundry when ANTHROPIC_BASE_URL is an Azure endpoint."""
    import anthropic

    api_key = api_key or require_env("ANTHROPIC_API_KEY", purpose="Anthropic models")
    base_url = base_url or get_env("ANTHROPIC_BASE_URL")
    if base_url and "azure" in base_url:
        cls = getattr(anthropic, "AsyncAnthropicFoundry" if async_client else "AnthropicFoundry")
        return cls(api_key=api_key, base_url=base_url)
    cls = anthropic.AsyncAnthropic if async_client else anthropic.Anthropic
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Small text helpers used across modules
# --------------------------------------------------------------------------- #
def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Return the content of the first <tag>...</tag> block, or None."""
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text or "", re.S)
    return match.group(1).strip() if match else None


def list2text(items: List[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1) if item)
