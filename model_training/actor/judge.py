"""Claude NLI-style judge that scores candidate patient turns for DPO pair selection.

Originally ``actor/claude_judge.py`` + ``actor/dpo_utils.py``. The paper used
``claude-opus-4-6`` served through Azure AI Foundry (``AnthropicFoundry``);
set ``ANTHROPIC_BASE_URL`` to your Foundry ``.../anthropic`` endpoint to do the
same, or leave it unset to call the Anthropic API directly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from angel_common.env import get_env
from angel_common.llm import anthropic_client

DEFAULT_JUDGE_DEPLOYMENT = "claude-opus-4-6"

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic_client()
    return _CLIENT


def _clip(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars]


def _extract_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        t = t[i : j + 1]
    return json.loads(t)


def claude_judge_nli(
    *,
    conversation_context_text: str,
    memory_text: str,
    last_message: str,
    state_text: str,
    word_text: str,
    deployment_name: Optional[str] = None,
    max_tokens: int = 700,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Score one candidate; returns the schema consumed by ``total_score``."""
    ctx = _clip(conversation_context_text, 2200)
    mem = _clip(memory_text, 600)
    last = _clip(last_message, 600)
    st = _clip(state_text, 120)
    wd = _clip(word_text, 1200)

    candidate = f"<state>{st}</state>\n<word>{wd}</word>"

    system = "Return valid JSON only."

    user = f"""
You will evaluate a candidate response given a short context snippet.

Return JSON only:
{{
  "consistency": "entailment" | "neutral" | "contradiction",
  "coherence_score": 0-5,
  "specificity_score": 0-5,
  "final_score": 0-100
}}

Definitions:
- entailment: consistent and supported by the context
- neutral: plausible but weakly supported
- contradiction: conflicts with the context

Context:
{ctx}

Memory:
{mem}

Latest message:
{last}

Candidate:
{candidate}
""".strip()

    resp = _client().messages.create(
        model=deployment_name or get_env("ANGEL_JUDGE_DEPLOYMENT", DEFAULT_JUDGE_DEPLOYMENT),
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    out_text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            out_text += block.text

    result = _extract_json(out_text)

    cons = result.get("consistency", "neutral")
    coh = float(result.get("coherence_score", 0))
    spec = float(result.get("specificity_score", 0))
    final = float(result.get("final_score", 0))

    verdict = "bad" if cons == "contradiction" else "good"

    # Only specificity/structure/consistency come from the judge; the other
    # rubric entries are constants (as in the original code).
    return {
        "verdict": verdict,
        "scores": {
            "specificity": spec,
            "structure": coh,
            "state_alignment": 5 if cons == "entailment" else 3,
            "history_consistency": 5 if cons == "entailment" else 3,
            "progress": 3,
            "naturalness": 4,
            "safety": 5,
        },
        "issues": [],
        "rewrite_instructions": "",
        "final_score": final,
        "consistency": cons,
    }


def judge_state_word(
    conversation_context_text: str,
    memory_text: str,
    therapist_msg: str,
    state_text: str,
    word_text: str,
) -> Dict[str, Any]:
    return claude_judge_nli(
        conversation_context_text=conversation_context_text,
        memory_text=memory_text,
        last_message=therapist_msg,
        state_text=state_text,
        word_text=word_text,
        max_tokens=700,
        temperature=0.0,
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {
    "structure": 1.5,
    "specificity": 1.2,
    "state_alignment": 1.2,
    "history_consistency": 1.0,
    "progress": 0.8,
    "naturalness": 0.6,
    "safety": 2.0,
}


def total_score(scores: Dict[str, float], weights: Dict[str, float] = DEFAULT_WEIGHTS) -> float:
    return float(sum(weights[k] * float(scores.get(k, 0.0)) for k in weights))


# --------------------------------------------------------------------------- #
# Near-duplicate filtering (64-bit simhash)
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def simhash64(text: str) -> int:
    toks = _tokenize(text)
    if not toks:
        return 0
    v = [0] * 64
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class Deduper:
    max_hamming: int = 4
    hashes: List[int] = None

    def __post_init__(self):
        if self.hashes is None:
            self.hashes = []

    def is_duplicate(self, text: str) -> bool:
        h = simhash64(text)
        return any(hamming(h, old) <= self.max_hamming for old in self.hashes)

    def add(self, text: str) -> None:
        self.hashes.append(simhash64(text))


def truncate_context(messages: List[Dict[str, str]], max_chars: int = 4000, max_msgs: int = 10) -> str:
    """Last ``max_msgs`` messages as ``ROLE: content`` lines, clipped to ``max_chars``."""
    s = ""
    for m in messages[-max_msgs:]:
        s += f"{(m.get('role') or '').upper()}: {m.get('content') or ''}\n"
    return _clip(s, max_chars)


# --------------------------------------------------------------------------- #
# Lightweight per-conversation memory passed to the judge
# --------------------------------------------------------------------------- #
@dataclass
class Memory:
    last_state: str = ""
    recent_events: List[str] = None

    def __post_init__(self):
        if self.recent_events is None:
            self.recent_events = []

    def update(self, state: str, word: str) -> None:
        self.last_state = (state or "")[:80]
        kept = []
        for s in re.split(r"[.!?\n]+", word or ""):
            s = s.strip()
            if s and any(k in s.lower() for k in
                         ["assignment", "deadline", "exam", "class", "study", "desk", "sleep", "night"]):
                kept.append(s[:200])
        if kept:
            self.recent_events.extend(kept[-2:])
            self.recent_events = self.recent_events[-6:]

    def summary(self) -> str:
        return f"last_state={self.last_state}; recent_events={'; '.join(self.recent_events)}"
