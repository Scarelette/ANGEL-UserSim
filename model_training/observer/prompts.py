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
