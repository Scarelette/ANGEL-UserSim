"""Observer prompts, output tags, GRPO chat template and completion regex.

The system prompts are verbatim from the training runs that produced
Qwen3-Observer-800 — the checkpoint was tuned against these exact strings, so
do not reword them.

Stage 1 (S1): presenting complaints -> symptoms + external factors (nodes).
Stage 2 (S2): complaints + node list -> directed symptom network (links).
"""

from __future__ import annotations

import re
from typing import Iterable

REASONING_START = "<think>"
REASONING_END = "</think>"
SOLUTION_START = "<GRAPH>"
SOLUTION_END = "</GRAPH>"


def build_system_prompt_s1() -> str:
    return f"""You are a clinical information extraction model.

Your task is to extract:
1) symptoms explicitly supported by the patient’s presenting complaints, and
2) external factors that influence the patient’s mental health.

Rules:
- Use only information explicitly supported by the text.
- Do not infer or invent symptoms or external factors.
- Use concise, clinically meaningful phrases.
- ONLY list symptoms and external factors.
- DO NOT create links or any other fields.

Put your reasoning between {REASONING_START} and {REASONING_END}.
Output ONLY valid JSON between {SOLUTION_START} and {SOLUTION_END}.

Output format:
{SOLUTION_START}
{{
  "symptoms": ["symptom 1", "symptom 2"],
  "external_factors": ["external factor 1", "external factor 2"]
}}
{SOLUTION_END}
"""


def build_system_prompt_s1_datagen() -> str:
    """S1 prompt variant sent to GPT-5 when generating S1 SFT targets.

    Identical to :func:`build_system_prompt_s1` except for the extra
    "Return JSON only ..." line (as in the original data-generation script).
    """
    return f"""You are a clinical information extraction model.

Your task is to extract:
1) symptoms explicitly supported by the patient’s presenting complaints, and
2) external factors that influence the patient’s mental health.

Rules:
- Use only information explicitly supported by the text.
- Do not infer or invent symptoms or external factors.
- Use concise, clinically meaningful phrases.
- ONLY list symptoms and external factors.
- DO NOT create links or any other fields.

Put your reasoning between {REASONING_START} and {REASONING_END}.
Output ONLY valid JSON between {SOLUTION_START} and {SOLUTION_END}.

Output format:
Return JSON only with exactly the following structure:

{SOLUTION_START}
{{
  "symptoms": ["symptom 1", "symptom 2"],
  "external_factors": ["external factor 1", "external factor 2"]
}}
{SOLUTION_END}
"""


def build_system_prompt_s2() -> str:
    return f"""You are a clinical information extraction model.

Given the Symptoms and External Factor List and the patient's Presenting Complaints, construct a **directed symptom network**.

Rules:
- Use **only** symptoms or external factors from the provided list.
- Create a link A → B **only if** the text supports that A causes, triggers, worsens, or maintains B.
- The link direction must reflect cause → effect.
- Do **not** infer relationships from general medical knowledge or symptom co-occurrence.
- Do **not** create links involving symptoms not explicitly mentioned.
- The reasoning **must be** no longer than 500 words.

Put all reasoning between {REASONING_START} and {REASONING_END}.
Output **only valid JSON** between {SOLUTION_START} and {SOLUTION_END}.

Output format:
{SOLUTION_START}
{{
  "links": [
    {{ "from": "symptom 1", "to": "symptom 2" }}
  ]
}}
{SOLUTION_END}
"""


def build_input_s1(complaints: str) -> str:
    return f"Patient Presenting Complaints: {complaints}"


def build_input_s2(complaints: str, symptoms: Iterable[str]) -> str:
    # `symptoms` is interpolated as a Python list repr, exactly as in training.
    return f"""Patient Presenting Complaints: {complaints}

Symptoms and External Factor List: {symptoms}
"""


# --------------------------------------------------------------------------- #
# SFT text format ("### System / ### User / ### Assistant")
# --------------------------------------------------------------------------- #
def format_sft_text(system: str, user: str, assistant: str, eos_token: str) -> str:
    return f"""### System:
{system}

### User:
{user}

### Assistant:
{assistant}{eos_token}
"""


# --------------------------------------------------------------------------- #
# GRPO chat template
# --------------------------------------------------------------------------- #
# The template used for every released GRPO run. It references `system_prompt`
# and `reasoning_start`, which Hugging Face never passes to the renderer, so
# both render as empty strings: the generation prompt is
# "<system>{eos}<user>" with surrounding whitespace and NO "<think>" prefix.
# Kept verbatim for reproducibility (see README, Known issues).
ORIGINAL_GRPO_CHAT_TEMPLATE = """
{% if messages[0]['role'] == 'system' %}
{{ messages[0]['content'] }}{{ eos_token }}
{% set loop_messages = messages[1:] %}
{% else %}
{{ system_prompt }}{{ eos_token }}
{% set loop_messages = messages %}
{% endif %}
{% for message in loop_messages %}
    {% if message['role'] == 'user' %}
        {{ message['content'] }}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'] }}{{ eos_token }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ reasoning_start }}
{% endif %}
"""


def _fixed_grpo_chat_template(system_prompt: str) -> str:
    """Same structure, with the variables actually substituted and no stray whitespace."""
    sp = system_prompt.replace("\\", "\\\\").replace("'", "\\'")
    return (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] }}{{ eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '" + sp + "' }}{{ eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '" + REASONING_START + "' }}{% endif %}"
    )


def setup_chat_template(tokenizer, system_prompt: str, mode: str = "original") -> None:
    """Install the GRPO chat template on ``tokenizer``.

    mode="original": the template used for Qwen3-Observer-800 (see above).
    mode="fixed":    variables substituted, ``<think>`` appended to the generation prompt.
    mode="native":   keep the tokenizer's own (Qwen3) chat template.
    """
    if mode == "original":
        tokenizer.chat_template = ORIGINAL_GRPO_CHAT_TEMPLATE
    elif mode == "fixed":
        tokenizer.chat_template = _fixed_grpo_chat_template(system_prompt)
    elif mode == "native":
        return
    else:
        raise ValueError(f"Unknown chat-template mode {mode!r} (original|fixed|native)")


def build_match_regex(tokenizer, require_reasoning_end: bool = True) -> re.Pattern:
    """Regex capturing the JSON between <GRAPH> ... </GRAPH> at the end of a completion."""
    solution_end_regex = (
        re.escape(SOLUTION_END)
        + r"[\s]*"
        + "(?:" + re.escape(tokenizer.eos_token) + ")?"
    )
    prefix = rf"{re.escape(REASONING_END)}.*?" if require_reasoning_end else ""
    return re.compile(
        rf"{prefix}"
        rf"{re.escape(SOLUTION_START)}\s*"
        rf"(?P<json>\{{.*?\}})"
        rf"\s*{solution_end_regex}"
        rf"\s*$",
        flags=re.MULTILINE | re.DOTALL,
    )
