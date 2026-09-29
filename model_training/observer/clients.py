"""Per-role Azure OpenAI clients.

The GRPO edge judge and GPT-5 may live on different Azure resources. Each role
can point to its own resource via ``ANGEL_<ROLE>_ENDPOINT`` / ``ANGEL_<ROLE>_API_KEY`` /
``ANGEL_<ROLE>_API_VERSION``; unset values fall back to the shared
``AZURE_OPENAI_*`` variables.
"""

from __future__ import annotations

from angel_common.env import get_env, require_env
from angel_common.llm import DEFAULT_API_VERSION, azure_openai_client


def azure_client_for_role(role: str, async_client: bool = False):
    role = role.upper()
    endpoint = get_env(f"ANGEL_{role}_ENDPOINT") or require_env(
        "AZURE_OPENAI_ENDPOINT", purpose=f"Azure OpenAI ({role.lower()})"
    )
    api_key = get_env(f"ANGEL_{role}_API_KEY") or require_env(
        "AZURE_OPENAI_API_KEY", purpose=f"Azure OpenAI ({role.lower()})"
    )
    api_version = get_env(f"ANGEL_{role}_API_VERSION") or get_env("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)
    return azure_openai_client(endpoint=endpoint, api_key=api_key, api_version=api_version, async_client=async_client)


def edge_judge_deployment() -> str:
    """GRPO S2 edge-plausibility judge (gpt-5-mini in the paper)."""
    return get_env("ANGEL_EDGE_JUDGE_DEPLOYMENT", "gpt-5-mini")

