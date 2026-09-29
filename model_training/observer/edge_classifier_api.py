"""Yes/No edge-plausibility classifier served as an Azure OpenAI fine-tuned deployment.

In the paper this is a GPT-4 model fine-tuned (via Azure OpenAI fine-tuning) on
the human edge annotations produced by ``edge_classifier_data.py
from-annotations``. It is the edge "reasonability" judge in the automatic
profile evaluation (``eval/score_network_edges.py``).

Configure with ANGEL_EDGE_CLASSIFIER_DEPLOYMENT (required) and optionally
ANGEL_EDGE_CLASSIFIER_ENDPOINT / _API_KEY / _API_VERSION (fallback AZURE_OPENAI_*).
"""

from __future__ import annotations

import time
from typing import Optional

from model_training.observer.clients import azure_client_for_role, edge_classifier_deployment

SYSTEM_PROMPT = (
    "You are a professional mental health expert. "
    "For each request, answer with exactly one word: Yes or No."
)

_CLIENT = None


def build_edge_prompt(complaints: str, edge_from: str, edge_to: str) -> str:
    """Classifier user prompt (verbatim from the original pipeline)."""
    return (
        f"This is the Presenting Complaints of the mental health patient: {complaints}\n"
        f"Based on the patient's symptoms, I construct a symptom relationship for the patient: \n"
        f"{edge_from} -> {edge_to}\n\n"
        "Do you think this link make sense according to the presenting complaints? "
        "Just answer with Yes or No"
    )


def classifier(cur_prompt: str, max_retries: int = 8, sleep_seconds: float = 2, max_tokens: int = 64) -> str:
    """Return the raw classifier answer (expected "Yes"/"No"); raises after ``max_retries``."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = azure_client_for_role("edge_classifier")
    deployment = edge_classifier_deployment()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": cur_prompt + "\n\nAnswer exactly one word: Yes or No."},
    ]
    errors = []
    for attempt in range(max_retries):
        try:
            request = {"messages": messages, "model": deployment, "max_tokens": max_tokens}
            try:
                response = _CLIENT.chat.completions.create(**request)
            except Exception:
                # Newer Azure deployments expect max_completion_tokens.
                request.pop("max_tokens")
                request["max_completion_tokens"] = max_tokens
                response = _CLIENT.chat.completions.create(**request)
        except Exception as e:
            errors.append(f"attempt {attempt + 1} API error: {e}")
            time.sleep(sleep_seconds)
            continue

        if not response or not response.choices:
            errors.append(f"attempt {attempt + 1} empty response")
            time.sleep(sleep_seconds)
            continue
        content: Optional[str] = response.choices[0].message.content
        if content and content.strip():
            return content
        errors.append(f"attempt {attempt + 1} empty content; finish_reason={response.choices[0].finish_reason}")
        time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Classifier returned no content after {max_retries} attempts. Last errors: {'; '.join(errors[-3:])}"
    )


def extract_label(text: Optional[str], label_set=("yes", "no")) -> str:
    """First label found in ``text`` (lower-case), else the first line, else "UNKNOWN"."""
    if text is None:
        return "UNKNOWN"
    normalized = text.strip().lower()
    if not normalized:
        return "UNKNOWN"
    for label in label_set:
        if label.lower() in normalized:
            return label.lower()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    return lines[0] if lines else "UNKNOWN"
