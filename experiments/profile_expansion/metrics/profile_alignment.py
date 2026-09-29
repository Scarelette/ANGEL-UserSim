"""Profile Alignment metric via GPT-5 prompting."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from angel_common.llm import getOutput
from experiments.profile_expansion.metrics.common import mean


ASPECTS: Dict[str, str] = {
    "background_alignment": "Does the patient preserve given demographic/background facts?",
    "presenting_problem_alignment": "Does the conversation focus on the main problem described in the profile?",
    "emotion_alignment": "Does the patient express emotions consistent with the profile?",
    "personality_interpersonal_alignment": "Does the patient's communication style match the profile?",
    "symptom_alignment": "Are the symptoms compatible with the profile?",
}

ASPECT_KEYS = list(ASPECTS.keys())


def _normalize_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    if 1.0 <= score <= 5.0:
        return score
    return None


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON: {text[:240]}")


def _canonical_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _extract_aspect_block(parsed: Dict[str, Any], aspect_key: str) -> Any:
    if aspect_key in parsed:
        return parsed[aspect_key]

    target = _canonical_key(aspect_key)
    for key, value in parsed.items():
        if _canonical_key(str(key)) == target:
            return value
    return None


def _transcript_text_for_prompt(record: Dict[str, Any]) -> str:
    transcript = record.get("transcript", [])
    if not isinstance(transcript, list):
        return ""

    lines: List[str] = []
    for item in transcript:
        if not isinstance(item, dict):
            continue

        nested_turns = item.get("turns")
        if isinstance(nested_turns, list):
            topic_key = item.get("topic_key", "")
            topic_name = item.get("topic_name", topic_key)
            lines.append(f"[TOPIC] {topic_name} ({topic_key})")
            for turn in nested_turns:
                if not isinstance(turn, dict):
                    continue
                role = (turn.get("role") or "").upper()
                content = (turn.get("content") or "").strip()
                if role and content:
                    lines.append(f"{role}: {content}")
            continue

        role = (item.get("role") or "").upper()
        content = (item.get("content") or "").strip()
        if role and content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _build_profile_alignment_prompt(record: Dict[str, Any]) -> str:
    profile = record.get("short_patient_profile") or ""
    transcript_text = _transcript_text_for_prompt(record)

    aspect_lines = "\n".join(f"- {key}: {description}" for key, description in ASPECTS.items())
    return f"""
You are evaluating profile alignment in a simulated psychotherapy patient conversation.

Task:
Score each aspect from 1 to 5.
- 1 = severe mismatch / clear contradiction
- 2 = mostly misaligned
- 3 = mixed / partial alignment
- 4 = mostly aligned
- 5 = strongly aligned

Aspects:
{aspect_lines}

Short profile:
{profile}

Conversation transcript:
{transcript_text}

Return valid JSON only with this exact structure:
{{
  "background_alignment": {{"score": 1-5, "reason": "short reason"}},
  "presenting_problem_alignment": {{"score": 1-5, "reason": "short reason"}},
  "emotion_alignment": {{"score": 1-5, "reason": "short reason"}},
  "personality_interpersonal_alignment": {{"score": 1-5, "reason": "short reason"}},
  "symptom_alignment": {{"score": 1-5, "reason": "short reason"}}
}}
""".strip()


def score_profile_alignment(record: Dict[str, Any], on_error=None) -> Dict[str, Any]:
    prompt = _build_profile_alignment_prompt(record)

    aspect_scores: Dict[str, Optional[float]] = {}
    aspect_reasons: Dict[str, str] = {}
    raw_output = ""
    parse_error = ""

    try:
        # gpt-5 reasoning tokens count against the budget; without minimal effort
        # the budget can be fully consumed by reasoning, returning empty content.
        raw_output = getOutput(
            prompt,
            context=None,
            max_completion_tokens=6000,
            tag=0,
            max_retries=4,
            on_error=on_error,
            reasoning_effort="minimal",
        )
        parsed = _extract_json(raw_output)

        for key in ASPECT_KEYS:
            block = _extract_aspect_block(parsed, key)
            reason = ""
            score_value: Any = None
            if isinstance(block, dict):
                score_value = block.get("score")
                reason = str(block.get("reason", "")).strip()
            else:
                score_value = block
            aspect_scores[key] = _normalize_score(score_value)
            aspect_reasons[key] = reason

    except Exception as exc:
        parse_error = repr(exc)
        for key in ASPECT_KEYS:
            aspect_scores[key] = None
            aspect_reasons[key] = ""

    valid_scores = [score for score in aspect_scores.values() if score is not None]
    avg_score_1_to_5 = mean(valid_scores)
    score = avg_score_1_to_5 / 5.0 if valid_scores else 0.0

    return {
        "metric": "profile_alignment",
        "score": score,
        "avg_score_1_to_5": avg_score_1_to_5,
        "num_aspects_scored": len(valid_scores),
        "aspect_scores": aspect_scores,
        "aspect_reasons": aspect_reasons,
        "parse_error": parse_error,
        "raw_judge_output": raw_output,
    }
