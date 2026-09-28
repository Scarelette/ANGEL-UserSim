"""Shared helpers for profile-expansion metrics."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "when",
    "with",
    "you",
}


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]+", (text or "").lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def _iter_transcript_turns(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    transcript = record.get("transcript", [])
    if not isinstance(transcript, list):
        return []

    turns: List[Dict[str, Any]] = []
    for item in transcript:
        if not isinstance(item, dict):
            continue

        # New format: topic-aligned blocks
        nested_turns = item.get("turns")
        if isinstance(nested_turns, list):
            for turn in nested_turns:
                if isinstance(turn, dict):
                    turns.append(turn)
            continue

        # Legacy flat format
        if "role" in item and "content" in item:
            turns.append(item)
    return turns


def patient_text(record: Dict[str, Any]) -> str:
    return " ".join(
        turn.get("content", "")
        for turn in _iter_transcript_turns(record)
        if turn.get("role") == "patient"
    )


def patient_turns(record: Dict[str, Any]) -> List[str]:
    return [
        turn.get("content", "")
        for turn in _iter_transcript_turns(record)
        if turn.get("role") == "patient" and turn.get("content")
    ]


def cosine_from_counters(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0

    shared = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
