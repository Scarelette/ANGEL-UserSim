"""Azure OpenAI therapist that converses with the simulated patient during rollouts.

Originally ``actor/therapist_azure.py``. The paper's rollouts used custom Azure
deployments on a separate Azure resource: ``ai_therapist`` for the SFT
rollouts and ``ai_therapist_2`` for the DPO rollouts. Set
``ANGEL_THERAPIST_DEPLOYMENT`` (or ``--therapist-deployment``) to your own
deployment. ``ANGEL_THERAPIST_AZURE_ENDPOINT`` / ``ANGEL_THERAPIST_AZURE_API_KEY``
override the shared ``AZURE_OPENAI_*`` variables for this role only.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from angel_common.env import get_env, require_env
from angel_common.llm import azure_openai_client
from model_training.actor.prompts import THERAPIST_SYSTEM_PROMPT


class AITherapist:
    def __init__(self, deployment: Optional[str] = None):
        self.deployment = deployment or require_env(
            "ANGEL_THERAPIST_DEPLOYMENT", purpose="the rollout therapist"
        )
        self.client = azure_openai_client(
            endpoint=get_env("ANGEL_THERAPIST_AZURE_ENDPOINT"),
            api_key=get_env("ANGEL_THERAPIST_AZURE_API_KEY"),
        )

    def _build_messages(self, conversation: List[Dict]) -> List[Dict]:
        """
        Accepts either [{"role": "patient"|"therapist", ...}] or OpenAI-style
        [{"role": "system"|"user"|"assistant", ...}]. External system prompts
        (the patient's) are dropped. OpenAI-style roles are from the patient's
        side (therapist = "user", patient = "assistant"), so they are flipped
        for the therapist. If there is no user message yet, a starter
        message is injected so the therapist can open the session.
        """
        messages: List[Dict] = [{"role": "system", "content": THERAPIST_SYSTEM_PROMPT}]
        has_user = False

        for turn in conversation or []:
            role = (turn.get("role") or "").lower()
            content = (turn.get("content") or "").strip()
            if not content:
                continue

            if role in {"system", "user", "assistant"}:
                if role == "system":
                    continue
                role = "assistant" if role == "user" else "user"
                if role == "user":
                    has_user = True
                messages.append({"role": role, "content": content})
                continue

            if role == "therapist":
                messages.append({"role": "assistant", "content": content})
            else:  # "patient" or unknown
                has_user = True
                messages.append({"role": "user", "content": content})

        if not has_user:
            messages.append({"role": "user", "content": "Start the session with a warm opening and one exploratory question."})
        return messages

    def generate(self, conversation: List[Dict], retries: int = 3) -> str:
        messages = self._build_messages(conversation)
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    max_completion_tokens=13107,
                    temperature=1.0,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Azure Error - Attempt {attempt + 1}] {e}")
                time.sleep(2)
        raise RuntimeError("Azure API failed after retries.")
