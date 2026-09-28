"""Azure OpenAI clients and deployment names for the profile-expansion experiment.

In the paper setup the agenda therapist and the prompted baselines
(Patient-Psi, Roleplay-doh) ran on one Azure OpenAI resource, while the GPT-5
metric judges ran on another. Each role can therefore point at its own
resource; unset role variables fall back to the shared AZURE_OPENAI_* ones.

    role        endpoint / key / version override         deployment (default)
    therapist   ANGEL_THERAPIST_AZURE_{ENDPOINT,API_KEY,API_VERSION}
                                                          ANGEL_THERAPIST_DEPLOYMENT    (gpt-4-04-14)
    baseline    ANGEL_BASELINE_AZURE_{ENDPOINT,API_KEY,API_VERSION}
                                                          ANGEL_PATIENT_PSI_DEPLOYMENT  (gpt-4)
                                                          ANGEL_ROLEPLAY_DOH_DEPLOYMENT (gpt-4o-2)

The default deployment names are the paper's Azure deployment names, which are
aliases chosen on that resource: ``gpt-4-04-14`` served gpt-4.1 (2025-04-14) and
``gpt-4o-2`` served gpt-4o. Set the variables to your own deployment names.
"""

from __future__ import annotations

from angel_common.env import get_env, require_env
from angel_common.llm import DEFAULT_API_VERSION

_DEFAULT_DEPLOYMENTS = {
    "therapist": "gpt-4-04-14",
    "patient_psi": "gpt-4",
    "roleplay_doh": "gpt-4o-2",
}


def deployment(role: str) -> str:
    """Deployment name for ``therapist`` / ``patient_psi`` / ``roleplay_doh``."""
    return get_env(f"ANGEL_{role.upper()}_DEPLOYMENT", _DEFAULT_DEPLOYMENTS[role])


def role_azure_client(role: str, async_client: bool = False):
    """(Async)AzureOpenAI client for ``therapist`` or ``baseline``."""
    from openai import AsyncAzureOpenAI, AzureOpenAI

    prefix = f"ANGEL_{role.upper()}_AZURE"
    cls = AsyncAzureOpenAI if async_client else AzureOpenAI
    return cls(
        azure_endpoint=require_env(f"{prefix}_ENDPOINT", "AZURE_OPENAI_ENDPOINT", purpose=f"the {role} model"),
        api_key=require_env(f"{prefix}_API_KEY", "AZURE_OPENAI_API_KEY", purpose=f"the {role} model"),
        api_version=get_env(f"{prefix}_API_VERSION", None, "AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION,
    )
