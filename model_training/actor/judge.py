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


def _ask_judge(system: str, user: str, deployment_name: Optional[str], max_tokens: int,
               temperature: float) -> Dict[str, Any]:
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

    return _extract_json(out_text)


def _number(value: Any) -> float:
    """A number from the judge's JSON ("85", "85/100", "4.5 / 5" -> leading number); else 0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(value or ""))
        return float(m.group(1)) if m else 0.0


def _score(result: Dict[str, Any], key: str) -> float:
    """A 0-5 rubric score from the judge's JSON; missing or malformed -> 0."""
    return min(5.0, max(0.0, _number(result.get(key, 0))))


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
    """Score one candidate on every entry of ``DEFAULT_WEIGHTS``; returns the schema consumed by ``total_score``."""
    ctx = _clip(conversation_context_text, 2200)
    mem = _clip(memory_text, 600)
    last = _clip(last_message, 600)
    st = _clip(state_text, 1200)
    wd = _clip(word_text, 1200)

    candidate = f"<state>{st}</state>\n<word>{wd}</word>"

    system = "Return valid JSON only."

    user = f"""
You will evaluate a candidate patient reply in a simulated therapy session.
The candidate has a <state> (the patient's inner state) and a <word> (what the patient says).

Return JSON only:
{{
  "consistency": "entailment" | "neutral" | "contradiction",
  "structure": 0-5,
  "specificity": 0-5,
  "state_alignment": 0-5,
  "history_consistency": 0-5,
  "progress": 0-5,
  "naturalness": 0-5,
  "safety": 0-5,
  "final_score": 0-100
}}

Definitions:
- consistency: entailment = consistent and supported by the context; neutral = plausible but weakly supported; contradiction = conflicts with the context
- structure: the reply is coherent and responds to the latest message
- specificity: concrete, personal detail rather than generic statements
- state_alignment: the <word> expresses the <state>
- history_consistency: agrees with the context and memory (facts, earlier states)
- progress: moves the conversation forward instead of repeating earlier turns
- naturalness: sounds like a real patient speaking, not a therapist or an assistant
- safety: free of harmful instructions and of role breaks (no therapist/assistant text)

Context:
{ctx}

Memory:
{mem}

Latest message:
{last}

Candidate:
{candidate}
""".strip()

    result = _ask_judge(system, user, deployment_name, max_tokens, temperature)
    cons = result.get("consistency", "neutral")
    return {
        "verdict": "bad" if cons == "contradiction" else "good",
        "scores": {k: _score(result, k) for k in DEFAULT_WEIGHTS},
        "issues": [],
        "rewrite_instructions": "",
        "final_score": _number(result.get("final_score", 0)),
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
