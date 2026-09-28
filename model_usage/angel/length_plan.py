"""Per-turn reply-length planning.

Ported from the paper evaluation actor (experiments/profile_expansion). Cheap regex heuristics on the
therapist's latest turn decide whether the patient answers tersely, briefly, or
at moderate length; the patient's current emotional state and a little
randomness shift that. The budget is always clamped to the caller's ceiling, so
this can only ever *shorten* a reply — latency never gets worse than the flat cap.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Invitational / elaborative cues plus wh-words (\bwhat\b also matches "what's").
_OPEN_RE = re.compile(
    r"\b(tell me|say more|more about|describe|walk me through|share|what|how|why)\b",
    re.IGNORECASE,
)
# Yes/no or factual questions. Checked FIRST so "how many/old" stays terse.
_CLOSED_RE = re.compile(
    r"^\s*(do|did|does|have|has|had|are|is|was|were|will|would|can|could|should|may)\b"
    r"|\bhow (old|many|much|often|long)\b|\bwhen (did|was|do|does)\b|\bwhere\b|\bwho\b"
    r"|\bwhat (time|day|date|year)\b|\byes or no\b",
    re.IGNORECASE,
)

_MODES = (
    (26, 1, "For THIS reply, answer very briefly — just a few words or one short sentence."),
    (55, 2, "For THIS reply, keep it short — about one or two sentences."),
    (None, 4, "For THIS reply, you may share a little more if it truly matters to you, but stay concise."),
)

_SHUTDOWN_CUES = ("guarded", "withdraw", "hesitat", "avoid")


@dataclass
class LengthPlan:
    budget_tokens: int
    max_sentences: int
    cue: str
    min_new_tokens: int


def plan_reply_length(
    therapist_message: Optional[str],
    dynamic_state: Optional[Dict[str, Any]],
    ceiling: int,
    rng: Optional[random.Random] = None,
) -> LengthPlan:
    """Choose a length mode from question type + emotional state + variance."""
    rng = rng or random
    message = (therapist_message or "").strip()
    word_count = len(message.split())

    closed = bool(_CLOSED_RE.search(message))
    is_open = (not closed) and bool(_OPEN_RE.search(message))

    if is_open:
        index = 2                                  # moderate
    elif closed or word_count <= 6:
        index = 0                                  # terse
    else:
        index = 1                                  # brief

    state = dynamic_state or {}
    behaviors = " ".join(state.get("current_behaviors", [])).lower()
    emotions = " ".join(state.get("current_emotions", [])).lower()
    if any(cue in behaviors for cue in _SHUTDOWN_CUES):
        index -= 1                                 # shut-down state -> shorter
    if "slightly more open" in emotions:
        index += 1                                 # opening up -> a little more

    roll = rng.random()                            # natural variance
    if roll < 0.25:
        index -= 1
    elif roll > 0.9:
        index += 1

    # Open questions get at least ~1 short sentence: never drop to the terse mode,
    # and hold a token floor so the reply isn't a fragment like "Just...".
    if is_open:
        index = max(index, 1)
    index = max(0, min(len(_MODES) - 1, index))

    mode_budget, max_sentences, cue = _MODES[index]
    budget = ceiling if mode_budget is None else min(ceiling, mode_budget)
    return LengthPlan(
        budget_tokens=budget,
        max_sentences=max_sentences,
        cue=cue,
        min_new_tokens=12 if is_open else 0,
    )
