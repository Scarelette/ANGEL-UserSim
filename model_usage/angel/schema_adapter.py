"""Adapt the Observer's long-profile JSON onto the rich patient schema.

Stage 1 and stage 2 do not speak the same schema. The Observer emits nested
clinical sections (`symptom_details`, `emotional_profile`, `cognitive_patterns`,
...); the Actor's `convert_rich_profile_to_internal` expects the flat rich schema
(`triggers`, `emotions`, `behaviors`, `hidden_state`, ... as lists, with
`disclosure_rules` / `simulation_rules` as objects). This module is the bridge.

Ported from experiments/profile_expansion (Angel initializer).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

RICH_REQUIRED_FIELDS = (
    "identity",
    "background",
    "presenting_problems",
    "triggers",
    "emotions",
    "behaviors",
    "family_context",
    "hidden_state",
    "speaking_style",
    "disclosure_rules",
    "simulation_rules",
)


def coerce_list(value: Any) -> List[str]:
    """Flatten any nesting of str/list/scalar into a list of clean strings."""
    if value is None:
        return []
    if isinstance(value, str):
        text = " ".join(value.split()).strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(coerce_list(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(coerce_list(item))
        return out
    text = " ".join(str(value).split()).strip()
    return [text] if text else []


def merge_lists(*values: Any) -> List[str]:
    """Concatenate coerced lists, dropping case-insensitive duplicates."""
    out: List[str] = []
    seen = set()
    for value in values:
        for item in coerce_list(value):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def age_level_from_identity(identity: Dict[str, Any]) -> str:
    age_value = identity.get("age")
    if isinstance(age_value, bool):
        return "adult"
    if isinstance(age_value, (int, float)):
        return "adolescent" if age_value < 18 else "adult"
    if isinstance(age_value, str):
        match = re.search(r"\d+", age_value)
        if match:
            return "adolescent" if int(match.group(0)) < 18 else "adult"
    return "adult"


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def adapt_observer_profile(
    observer_profile: Dict[str, Any],
    *,
    source_title: str = "",
    short_profile_text: str = "",
) -> Dict[str, Any]:
    """Map one Observer long-profile JSON onto the rich patient schema."""
    if not isinstance(observer_profile, dict):
        raise TypeError(f"observer_profile must be a dict, got {type(observer_profile).__name__}")

    identity = _ensure_dict(observer_profile.get("identity"))
    symptom_details = _ensure_dict(observer_profile.get("symptom_details"))
    emotional = _ensure_dict(observer_profile.get("emotional_profile"))
    cognitive = _ensure_dict(observer_profile.get("cognitive_patterns"))
    behavior = _ensure_dict(observer_profile.get("behavior_patterns"))
    social = _ensure_dict(observer_profile.get("social_and_family_context"))
    hidden = _ensure_dict(observer_profile.get("hidden_state"))
    speaking = _ensure_dict(observer_profile.get("speaking_style"))
    disclosure = _ensure_dict(observer_profile.get("disclosure_rules"))
    summary = observer_profile.get("brief_summary")

    background = merge_lists(observer_profile.get("background"), summary, short_profile_text)
    presenting = merge_lists(observer_profile.get("presenting_problems"))
    triggers = merge_lists(observer_profile.get("triggers"), symptom_details.get("triggers"))
    emotions = merge_lists(
        observer_profile.get("emotions"),
        emotional.get("dominant_emotions"),
        emotional.get("secondary_emotions"),
        emotional.get("emotional_expression_style"),
    )
    behaviors = merge_lists(
        observer_profile.get("behaviors"),
        behavior.get("coping_strategies"),
        behavior.get("avoidance_behaviors"),
        behavior.get("safety_behaviors"),
        behavior.get("interpersonal_patterns"),
        behavior.get("maladaptive_coping"),
        symptom_details.get("functional_impairment"),
    )
    family_context = merge_lists(
        observer_profile.get("family_context"),
        social.get("family_relationships"),
        social.get("romantic_or_peer_relationships"),
        social.get("school_or_work_context"),
        social.get("social_support"),
        social.get("cultural_or_contextual_factors"),
    )
    # `hidden_state` feeds both core_beliefs and the sensitive-topic list, so the
    # cognitive material belongs here rather than being dropped.
    hidden_state = merge_lists(
        observer_profile.get("hidden_state") if not isinstance(observer_profile.get("hidden_state"), dict) else None,
        hidden.get("information_patient_initially_withholds"),
        hidden.get("information_revealed_after_trust"),
        hidden.get("topics_patient_avoids"),
        hidden.get("contradictions_or_ambivalence"),
        cognitive.get("core_beliefs"),
        cognitive.get("automatic_thoughts"),
        cognitive.get("worries_or_ruminations"),
        cognitive.get("interpretation_biases"),
    )

    speaking_style = {
        "age_level": age_level_from_identity(identity),
        "tone": speaking.get("tone") or "natural and conversational",
        "vocabulary": speaking.get("word_choice") or "everyday, non-technical",
        "sentence_style": speaking.get("verbosity") or "short to medium responses",
        "disclosure_style": speaking.get("interaction_style") or "gradual disclosure",
    }

    disclosure_rules = {
        "reveal_gradually": True,
        "do_not_dump_case_summary": True,
        "do_not_use_clinical_jargon": True,
        "topics_likely_early": merge_lists(disclosure.get("early_session")),
        "topics_likely_late": merge_lists(disclosure.get("middle_session"), disclosure.get("late_session")),
        "topics_avoid_unless_asked": merge_lists(hidden.get("topics_patient_avoids")),
    }

    # The Observer emits simulation_rules as a list of prose sentences, but the
    # rich schema reads it as a flag object. Keep the documented flags on and
    # preserve the prose under a non-colliding key so nothing is silently lost.
    simulation_rules: Dict[str, Any] = {
        "stay_in_character": True,
        "speak_as_patient_only": True,
        "no_narration": True,
        "no_researcher_voice": True,
        "keep_responses_conversational": True,
        "do_not_invent_major_facts": True,
        "if_unsure_say_limited_knowledge": True,
    }
    observer_rules = merge_lists(observer_profile.get("simulation_rules"))
    if observer_rules:
        simulation_rules["observer_notes"] = observer_rules

    return {
        "identity": {
            "name": identity.get("name") or "Unknown",
            "age": identity.get("age", "unknown"),
            "gender": identity.get("gender") or "unspecified",
            "role": identity.get("role") or "patient",
            "source_title": identity.get("source_title") or source_title or "",
            "diagnosis_hint": coerce_list(identity.get("diagnosis_hint")),
        },
        "background": background,
        "presenting_problems": presenting,
        "triggers": triggers,
        "emotions": emotions,
        "behaviors": behaviors,
        "family_context": family_context,
        "hidden_state": hidden_state,
        "speaking_style": speaking_style,
        "disclosure_rules": disclosure_rules,
        "simulation_rules": simulation_rules,
        "_meta": {
            "source_title": source_title or identity.get("source_title") or "",
            "stage1_schema_adapted": True,
        },
    }


def build_minimal_rich_profile(short_profile_text: str, *, source_title: str = "") -> Dict[str, Any]:
    """Fallback rich profile built from short text alone (no Observer available)."""
    text = " ".join((short_profile_text or "").split()).strip()
    if not text:
        raise ValueError("short_profile_text is empty")
    return {
        "identity": {
            "name": "Unknown",
            "age": "unknown",
            "gender": "unspecified",
            "role": "patient",
            "source_title": source_title,
            "diagnosis_hint": [],
        },
        "background": [text],
        "presenting_problems": [text],
        "triggers": [],
        "emotions": [],
        "behaviors": [],
        "family_context": [],
        "hidden_state": [],
        "speaking_style": {
            "age_level": "adult",
            "tone": "natural and conversational",
            "vocabulary": "everyday",
            "sentence_style": "short to medium responses",
            "disclosure_style": "gradual disclosure",
        },
        "disclosure_rules": {
            "reveal_gradually": True,
            "do_not_dump_case_summary": True,
            "do_not_use_clinical_jargon": True,
            "topics_likely_early": [],
            "topics_likely_late": [],
            "topics_avoid_unless_asked": [],
        },
        "simulation_rules": {
            "stay_in_character": True,
            "speak_as_patient_only": True,
            "no_narration": True,
            "no_researcher_voice": True,
            "keep_responses_conversational": True,
            "do_not_invent_major_facts": True,
            "if_unsure_say_limited_knowledge": True,
        },
        "_meta": {"source_title": source_title, "minimal_fallback": True},
    }
