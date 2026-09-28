"""Behavior Diversity metric via topic-specific open-attribute extraction."""

from __future__ import annotations

import concurrent.futures
import json
import math
import re
import threading
import time
from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from angel_common.llm import getOutput
from experiments.profile_expansion.metrics.common import mean


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Heuristic: does this exception look like an API rate-limit / 429?"""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "ratelimit" in name
        or "429" in text
        or "rate limit" in text
        or "too many requests" in text
    )


class _AdaptiveConcurrency:
    """AIMD concurrency limiter for the extraction LLM calls.

    Starts at ``max_workers`` in-flight calls. On a rate-limit signal it
    multiplicatively decreases the allowed concurrency (and debounces bursts so
    a cluster of 429s from already in-flight calls counts once); while healthy
    it additively ramps back up toward ``max_workers``.
    """

    def __init__(
        self,
        max_workers: int,
        *,
        min_workers: int = 1,
        decrease_factor: float = 0.5,
        cooldown_seconds: float = 3.0,
        success_step: int = 0,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.min_workers = max(1, min(int(min_workers), self.max_workers))
        self.limit = float(self.max_workers)
        self.in_use = 0
        self.decrease_factor = decrease_factor
        self.cooldown_seconds = cooldown_seconds
        # Successes required before each additive +1 (default: one full generation).
        self.success_step = success_step if success_step > 0 else self.max_workers
        self._success_since_increase = 0
        self._last_decrease = 0.0
        self._cond = threading.Condition()

    def acquire(self) -> None:
        with self._cond:
            while self.in_use >= int(self.limit):
                self._cond.wait(timeout=1.0)
            self.in_use += 1

    def release(self) -> None:
        with self._cond:
            self.in_use = max(0, self.in_use - 1)
            self._cond.notify_all()

    def on_success(self) -> None:
        with self._cond:
            if self.limit >= self.max_workers:
                return
            self._success_since_increase += 1
            if self._success_since_increase >= self.success_step:
                self._success_since_increase = 0
                new_limit = min(float(self.max_workers), self.limit + 1.0)
                if new_limit != self.limit:
                    self.limit = new_limit
                    print(
                        f"[BehaviorDiversity] concurrency increase -> {int(self.limit)}",
                        flush=True,
                    )
                    self._cond.notify_all()

    def on_error(self, exc: BaseException) -> None:
        if not _is_rate_limit_error(exc):
            return
        now = time.time()
        with self._cond:
            # Debounce: a batch of in-flight calls can all 429 at once; treat
            # rate-limit signals within the cooldown window as a single event.
            if now - self._last_decrease < self.cooldown_seconds:
                return
            self._last_decrease = now
            self._success_since_increase = 0
            new_limit = max(float(self.min_workers), float(int(self.limit * self.decrease_factor)))
            if new_limit != self.limit:
                self.limit = new_limit
                print(
                    f"[BehaviorDiversity] rate-limit hit; concurrency decrease -> {int(self.limit)}",
                    flush=True,
                )


# Topic agenda and per-topic open attributes derived from
# simulate_patient/profile_expansion/metrics/agenda_detail.md.
# Topic keys MUST match the transcript topic keys produced by
# simulate_patient/profile_expansion/interview_process.py so per-topic patient
# text can be collected. Attributes are the doc's "Information to extract" lists.
TOPIC_ATTRIBUTE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "presenting_problem": {
        "topic_name": "Presenting problem / chief complaint",
        "goal": (
            "Understand why the patient is seeking help now, how they describe the "
            "main concern in their own words, and what they hope to receive from the "
            "interview or treatment."
        ),
        "attributes": [
            "chief_complaint",
            "problem_in_own_words",
            "main_concern",
            "problem_framing",
            "reason_for_seeking_help",
            "current_urgency",
            "expectations_for_help",
        ],
    },
    "symptoms_emotions": {
        "topic_name": "Major symptoms: emotions",
        "goal": "Understand the patient's emotional experience.",
        "attributes": [
            "dominant_emotion",
            "secondary_emotions",
            "emotional_intensity",
            "emotional_frequency",
            "emotional_duration",
            "mood_changes",
            "anxiety_or_fear",
            "anger_or_irritability",
            "sadness_or_hopelessness",
            "shame_or_guilt",
        ],
    },
    "symptoms_behaviors": {
        "topic_name": "Major symptoms: behaviors",
        "goal": "Understand how the problem shows up in the patient's behavior.",
        "attributes": [
            "avoidance_behavior",
            "withdrawal",
            "compulsions_or_repetitive_behaviors",
            "changes_in_activity",
            "sleep_behavior_changes",
            "eating_behavior_changes",
            "agitation_or_restlessness",
            "functional_behavioral_changes",
            "maladaptive_behavior_patterns",
        ],
    },
    "symptoms_cognitions": {
        "topic_name": "Major symptoms: cognitions",
        "goal": (
            "Understand the patient's thoughts, beliefs, worries, interpretations, "
            "and cognitive patterns."
        ),
        "attributes": [
            "negative_thoughts",
            "worries",
            "rumination",
            "self_critical_beliefs",
            "beliefs_about_others",
            "beliefs_about_the_future",
            "interpretations_of_events",
            "thought_constriction",
            "obsessional_thoughts",
            "hopeless_thoughts",
        ],
    },
    "onset_timeline": {
        "topic_name": "History of problem: onset / timeline",
        "goal": "Establish the chronology and course of the presenting problem.",
        "attributes": [
            "onset_time",
            "duration",
            "first_episode_or_recurrent_problem",
            "number_of_episodes",
            "course_over_time",
            "changes_in_severity",
            "periods_of_improvement",
            "periods_of_worsening",
            "past_similar_problems",
        ],
    },
    "triggers": {
        "topic_name": "History of problem: triggers / precipitating events",
        "goal": (
            "Identify events, stressors, or situations that started, reactivated, or "
            "worsened the problem."
        ),
        "attributes": [
            "perceived_cause",
            "precipitating_events",
            "recent_stressors",
            "trigger_events",
            "trigger_contexts",
            "trigger_frequency",
            "trigger_specificity",
            "patient_interpretation_of_triggers",
            "situations_that_worsen_the_problem",
            "situations_that_improve_the_problem",
        ],
    },
    "impact_functioning": {
        "topic_name": "Impact on functioning",
        "goal": (
            "Assess how the problem affects the patient's daily life, including work "
            "or school, relationships, health, self-care, and everyday tasks."
        ),
        "attributes": [
            "work_or_school_impact",
            "relationship_impact",
            "everyday_task_impact",
            "self_care_impact",
            "health_impact",
            "social_functioning_impact",
            "role_functioning_impact",
            "typical_day_disruption",
            "functional_impairment_level",
        ],
    },
    "current_coping": {
        "topic_name": "Current coping",
        "goal": (
            "Understand how the patient currently deals with the problem, what helps, "
            "what makes it worse, and whether avoidance or other maladaptive coping "
            "patterns are present."
        ),
        "attributes": [
            "coping_strategies",
            "helpful_coping",
            "maladaptive_coping",
            "avoidance_behavior",
            "safety_behaviors",
            "emotion_regulation_strategies",
            "problem_solving_attempts",
            "coping_effectiveness",
            "barriers_to_coping",
            "situations_when_better",
            "situations_when_worse",
        ],
    },
    "social_support": {
        "topic_name": "Social support",
        "goal": (
            "Assess who is important in the patient's life, the quality of available "
            "support, whether the patient feels isolated, and what others have suggested."
        ),
        "attributes": [
            "important_people",
            "support_sources",
            "support_quality",
            "frequency_of_contact",
            "family_support",
            "friend_support",
            "community_support",
            "isolation_level",
            "desired_support",
            "barriers_to_support",
            "others_suggestions",
        ],
    },
    "past_treatment": {
        "topic_name": "Past treatment / psychiatric history",
        "goal": (
            "Collect prior treatment experiences, psychiatric diagnoses, "
            "hospitalizations, medication history, previous episodes, family mental "
            "health history, and barriers to seeking help."
        ),
        "attributes": [
            "prior_help_seeking",
            "therapy_history",
            "medication_history",
            "prior_diagnoses",
            "hospitalization_history",
            "outpatient_treatment_history",
            "significant_psychiatric_episodes",
            "treatment_duration",
            "treatment_response",
            "helpful_treatment_elements",
            "unhelpful_treatment_elements",
            "barriers_to_treatment",
            "family_mental_health_history",
        ],
    },
    "risk_suicide_self_harm": {
        "topic_name": "Risk assessment: suicide / self-harm",
        "goal": (
            "Assess safety concerns related to suicide or self-harm, including "
            "frequency, plans, intent, access to means, risk factors, and protective "
            "factors."
        ),
        "attributes": [
            "passive_suicidal_ideation",
            "active_suicidal_ideation",
            "ideation_intensity",
            "ideation_frequency",
            "ideation_specificity",
            "ambivalence",
            "prior_attempts",
            "self_harm_history",
            "plan",
            "means_access",
            "means_lethality",
            "rehearsal_or_preparation",
            "last_occurrence",
            "protective_factors",
        ],
    },
    "risk_harm_others": {
        "topic_name": "Risk assessment: harm to others / victimization",
        "goal": "Assess risk of harm to others and any recent abuse or victimization.",
        "attributes": [
            "anger_control",
            "violent_ideation",
            "homicidal_ideation",
            "threats_or_statements",
            "past_violence",
            "current_victimization",
            "recent_threats",
            "abuse_concerns",
        ],
    },
    "risk_substance_use": {
        "topic_name": "Risk assessment: substance use",
        "goal": "Assess substance use patterns and their impact on the patient's life.",
        "attributes": [
            "alcohol_use",
            "drug_use",
            "prescription_medication_misuse",
            "tobacco_use",
            "use_intensity",
            "use_frequency",
            "use_duration",
            "substance_related_impairment",
            "legal_problems_related_to_use",
            "others_concern_about_use",
        ],
    },
    "treatment_goal": {
        "topic_name": "Treatment goal / hope",
        "goal": (
            "Understand the patient's goals for treatment, expectations for therapy, "
            "desired changes, strengths, sources of resilience, and any remaining concerns."
        ),
        "attributes": [
            "treatment_goals",
            "desired_changes",
            "therapy_expectations",
            "hopefulness",
            "motivation_for_treatment",
            "strengths",
            "resilience_factors",
            "hobbies_or_activities",
            "spiritual_or_religious_support",
            "additional_concerns",
        ],
    },
}


# v4 design:
# - Score is attribute VALUE diversity only (run-to-run variety of the extracted
#   attribute values). Combination diversity and attribute density (coverage) are
#   dropped: density rewards verbosity/coverage rather than diversity, and combo
#   is small and gameable by sparse responses.
# - A monotonic contrast transform reduces ceiling effects and improves separability.
# - A global + topic-specific value-diversity boost is applied before final composition.
METRIC_VARIANT = "value_only_contrast_v4"
COMBO_LOG_GAIN = 9.0
WEIGHT_COMBO_SIGNAL = 0.0
WEIGHT_VALUE_DIVERSITY = 1.0
WEIGHT_ATTRIBUTE_DENSITY = 0.0
SCORE_CONTRAST_POWER = 2.0

# Every topic is weighted equally (no per-topic overrides).
TOPIC_WEIGHT_OVERRIDES: Dict[str, float] = {}

# Value-diversity shaping:
# - Global uplift for all topics
# - Small extra uplift for clinically salient topics
VALUE_DIVERSITY_GLOBAL_BOOST = 0.050
VALUE_DIVERSITY_TOPIC_BOOSTS: Dict[str, float] = {
    "symptoms_emotions": 0.005,
    "symptoms_cognitions": 0.005,
}

# Bumped to 2: topic agenda + attribute schema replaced from agenda_detail.md,
# so cached extractions from the previous schema must be invalidated.
# Bumped to 3: extraction call now uses a larger token budget + reasoning_effort
# "minimal" to fix gpt-5 returning empty content; prior cache held ~95% empty
# "empty output" failures that must be re-extracted.
EXTRACTION_CACHE_ENTRY_VERSION = 3


def _canonical_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


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


def _find_by_canonical_key(payload: Dict[str, Any], target: str) -> Any:
    if target in payload:
        return payload[target]
    target_norm = _canonical_key(target)
    for key, value in payload.items():
        if _canonical_key(str(key)) == target_norm:
            return value
    return None


def _normalize_values(raw: Any) -> Set[str]:
    values: List[str] = []
    if raw is None:
        return set()
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item) for item in raw if item is not None]
    else:
        values = [str(raw)]

    cleaned: Set[str] = set()
    for value in values:
        value = re.sub(r"\s+", " ", value).strip().lower()
        value = value.strip(".,;:!?")
        if value:
            cleaned.add(value)
    return cleaned


def _collect_topic_patient_texts(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    transcript = record.get("transcript", [])
    topic_texts: Dict[str, Dict[str, Any]] = {}
    if not isinstance(transcript, list):
        return topic_texts

    has_topic_blocks = any(isinstance(item, dict) and isinstance(item.get("turns"), list) for item in transcript)
    if has_topic_blocks:
        for item in transcript:
            if not isinstance(item, dict):
                continue
            topic_key = str(item.get("topic_key") or "")
            if not topic_key:
                continue
            topic_name = str(item.get("topic_name") or topic_key)
            turns = item.get("turns")
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                if turn.get("role") != "patient":
                    continue
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                if topic_key not in topic_texts:
                    topic_texts[topic_key] = {"topic_name": topic_name, "texts": []}
                topic_texts[topic_key]["texts"].append(content)
        return topic_texts

    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "patient":
            continue
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        topic_key = str(turn.get("topic_key") or "")
        if not topic_key:
            continue
        topic_name = str(turn.get("topic_name") or topic_key)
        if topic_key not in topic_texts:
            topic_texts[topic_key] = {"topic_name": topic_name, "texts": []}
        topic_texts[topic_key]["texts"].append(content)

    return topic_texts


def _build_extraction_prompt(
    *,
    short_profile: str,
    topic_texts: Dict[str, Dict[str, Any]],
) -> str:
    schema_text = json.dumps(
        {
            key: {
                "goal": spec["goal"],
                "attributes": spec["attributes"],
            }
            for key, spec in TOPIC_ATTRIBUTE_SCHEMA.items()
        },
        ensure_ascii=False,
        indent=2,
    )

    sections: List[str] = []
    for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
        topic_name = spec["topic_name"]
        topic_info = topic_texts.get(topic_key, {})
        texts = topic_info.get("texts") or []
        if texts:
            patient_text_block = "\n".join(f"- {text}" for text in texts)
        else:
            patient_text_block = "- [NO_PATIENT_CONTENT]"
        sections.append(
            f"Topic: {topic_name} ({topic_key})\n"
            f"Patient utterances:\n{patient_text_block}"
        )

    sections_text = "\n\n".join(sections)

    return f"""
You are extracting topic-specific patient-side behavioral attributes from a psychotherapy conversation.

Goal:
- Extract ONLY attributes that are plausibly elaborated by the patient's utterances.
- Focus on open attributes (do not simply copy fixed background facts from the short profile).
- Use concise phrase values.

Short profile:
{short_profile}

Topic schema:
{schema_text}

Patient utterances by topic:
{sections_text}

Output rules:
- Return valid JSON only.
- Use this exact top-level format:
{{
  "topics": {{
    "<topic_key>": {{
      "<attribute_key>": ["value1", "value2"]
    }}
  }}
}}
- For every topic in the schema, include all listed attribute keys.
- Use [] when there is no evidence for an attribute.
""".strip()


def _extract_record_attributes(
    record: Dict[str, Any],
    on_error: Optional[Callable[[BaseException], None]] = None,
) -> Tuple[Dict[str, Dict[str, Set[str]]], str, Set[str]]:
    short_profile = str(record.get("short_patient_profile") or "")
    topic_texts = _collect_topic_patient_texts(record)
    topics_with_patient_text = {topic_key for topic_key, item in topic_texts.items() if item.get("texts")}
    prompt = _build_extraction_prompt(short_profile=short_profile, topic_texts=topic_texts)

    output_text = ""
    parse_error = ""
    extracted: Dict[str, Dict[str, Set[str]]] = {
        topic_key: {attr: set() for attr in spec["attributes"]}
        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items()
    }

    try:
        # gpt-5 reasoning tokens count against the budget; this extraction needs
        # room for both. With reasoning_effort="minimal" the structured JSON is
        # emitted reliably (~2.5k completion tokens) instead of the budget being
        # consumed by reasoning and returning empty content.
        output_text = getOutput(
            prompt,
            context=None,
            max_completion_tokens=16000,
            tag=0,
            max_retries=4,
            on_error=on_error,
            reasoning_effort="minimal",
        )
        parsed = _extract_json(output_text)
        topics_block = parsed.get("topics") if isinstance(parsed, dict) else None
        if not isinstance(topics_block, dict):
            topics_block = parsed if isinstance(parsed, dict) else {}

        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
            topic_obj = _find_by_canonical_key(topics_block, topic_key)
            if not isinstance(topic_obj, dict):
                continue
            for attr in spec["attributes"]:
                raw = _find_by_canonical_key(topic_obj, attr)
                extracted[topic_key][attr] = _normalize_values(raw)
    except Exception as exc:
        parse_error = repr(exc)

    return extracted, parse_error, topics_with_patient_text


def _serialize_extracted_attributes(
    extracted: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    return {
        topic_key: {
            attr: sorted(values)
            for attr, values in topic_values.items()
        }
        for topic_key, topic_values in extracted.items()
    }


def _deserialize_extracted_attributes(
    payload: Any,
) -> Dict[str, Dict[str, Set[str]]]:
    extracted: Dict[str, Dict[str, Set[str]]] = {
        topic_key: {attr: set() for attr in spec["attributes"]}
        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items()
    }
    if not isinstance(payload, dict):
        return extracted

    for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
        topic_block = payload.get(topic_key)
        if not isinstance(topic_block, dict):
            continue
        for attr in spec["attributes"]:
            raw_values = topic_block.get(attr, [])
            if not isinstance(raw_values, list):
                continue
            extracted[topic_key][attr] = {str(value) for value in raw_values if value is not None}
    return extracted


def _record_cache_lookup(
    *,
    cache_key: Optional[str],
    extraction_cache: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Tuple[Dict[str, Dict[str, Set[str]]], str, Set[str]]]:
    if not cache_key or extraction_cache is None:
        return None
    entry = extraction_cache.get(cache_key)
    if not isinstance(entry, dict):
        return None
    if entry.get("schema_version") != EXTRACTION_CACHE_ENTRY_VERSION:
        return None

    extracted = _deserialize_extracted_attributes(entry.get("extracted"))
    parse_error = str(entry.get("parse_error") or "")
    raw_topics = entry.get("topics_with_patient_text") or []
    topics_with_patient_text = {
        str(topic_key)
        for topic_key in raw_topics
        if isinstance(topic_key, str)
    }
    return extracted, parse_error, topics_with_patient_text


def _record_cache_store(
    *,
    cache_key: Optional[str],
    extraction_cache: Optional[Dict[str, Dict[str, Any]]],
    extracted: Dict[str, Dict[str, Set[str]]],
    parse_error: str,
    topics_with_patient_text: Set[str],
) -> None:
    if not cache_key or extraction_cache is None:
        return
    extraction_cache[cache_key] = {
        "schema_version": EXTRACTION_CACHE_ENTRY_VERSION,
        "extracted": _serialize_extracted_attributes(extracted),
        "parse_error": parse_error,
        "topics_with_patient_text": sorted(topics_with_patient_text),
    }


def _jaccard_distance(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def _combo_signal(combo_div: float) -> float:
    clipped = max(0.0, min(1.0, float(combo_div)))
    # log1p transform keeps [0,1] bounds while increasing contrast in low-range combos.
    return math.log1p(COMBO_LOG_GAIN * clipped) / math.log1p(COMBO_LOG_GAIN)


def _topic_weight(topic_key: str) -> float:
    return max(0.0, float(TOPIC_WEIGHT_OVERRIDES.get(topic_key, 1.0)))


def _weighted_mean(values: Sequence[Tuple[float, float]]) -> float:
    values = list(values)
    total_weight = sum(weight for _, weight in values if weight > 0.0)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in values if weight > 0.0) / total_weight


def _boost_value_diversity(topic_key: str, raw_value_div: float) -> float:
    boost = VALUE_DIVERSITY_GLOBAL_BOOST + float(VALUE_DIVERSITY_TOPIC_BOOSTS.get(topic_key, 0.0))
    return max(0.0, min(1.0, float(raw_value_div) + boost))


def score_behavior_diversity(
    records: List[Dict[str, Any]],
    *,
    existing_profiles: Optional[List[Dict[str, Any]]] = None,
    on_profile_scored: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
    extraction_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    extraction_cache_key_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    extraction_cache_stats: Optional[Dict[str, int]] = None,
    max_workers: int = 1,
    workers_decrease_factor: float = 0.5,
    workers_cooldown_seconds: float = 3.0,
    workers_success_step: int = 0,
) -> Dict[str, Any]:
    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record.get("model"), record.get("profile_id"))].append(record)

    print(
        f"[BehaviorDiversity] start num_records={len(records)} num_model_profile_groups={len(grouped)}",
        flush=True,
    )

    existing_by_key: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    if existing_profiles:
        for item in existing_profiles:
            if not isinstance(item, dict):
                continue
            key = (item.get("model"), item.get("profile_id"))
            if key in grouped:
                existing_by_key[key] = item

    profile_scores = list(existing_by_key.values())
    if existing_by_key:
        print(
            f"[BehaviorDiversity] resume existing_profiles={len(existing_by_key)}",
            flush=True,
        )

    # Optional parallel pre-extraction: the per-record LLM extraction is the
    # slow part. When max_workers > 1, extract all uncached records concurrently
    # and populate the extraction cache up front. The sequential scoring loop
    # below then hits the cache for every record, so its logic, ordering, and
    # checkpointing remain unchanged.
    if (
        max_workers
        and max_workers > 1
        and extraction_cache is not None
        and extraction_cache_key_fn is not None
    ):
        pending: Dict[str, Dict[str, Any]] = {}
        for (model, profile_id), group in grouped.items():
            if (model, profile_id) in existing_by_key:
                continue
            for record in group:
                cache_key = extraction_cache_key_fn(record)
                if not cache_key or cache_key in pending:
                    continue
                if _record_cache_lookup(cache_key=cache_key, extraction_cache=extraction_cache) is not None:
                    continue
                pending[cache_key] = record

        if pending:
            limiter = _AdaptiveConcurrency(
                max_workers,
                decrease_factor=workers_decrease_factor,
                cooldown_seconds=workers_cooldown_seconds,
                success_step=workers_success_step,
            )
            print(
                f"[BehaviorDiversity] adaptive pre-extraction records={len(pending)} "
                f"max_workers={max_workers} decrease_factor={workers_decrease_factor} "
                f"cooldown={workers_cooldown_seconds}s "
                f"success_step={workers_success_step or max_workers}",
                flush=True,
            )

            def _extract_one(cache_key: str, record: Dict[str, Any]):
                limiter.acquire()
                try:
                    result = _extract_record_attributes(record, on_error=limiter.on_error)
                    limiter.on_success()
                    return cache_key, result
                finally:
                    limiter.release()

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_extract_one, cache_key, record)
                    for cache_key, record in pending.items()
                ]
                for future in concurrent.futures.as_completed(futures):
                    cache_key, (extracted, parse_error, topics_with_patient_text) = future.result()
                    _record_cache_store(
                        cache_key=cache_key,
                        extraction_cache=extraction_cache,
                        extracted=extracted,
                        parse_error=parse_error,
                        topics_with_patient_text=topics_with_patient_text,
                    )
            print("[BehaviorDiversity] adaptive pre-extraction complete", flush=True)

    total_groups = len(grouped)
    for (model, profile_id), group in grouped.items():
        if (model, profile_id) in existing_by_key:
            print(
                f"[BehaviorDiversity] model={model} profile_id={profile_id} skipped (from checkpoint)",
                flush=True,
            )
            continue

        print(
            f"[BehaviorDiversity] model={model} profile_id={profile_id} num_runs={len(group)}",
            flush=True,
        )
        run_attributes: List[Dict[str, Dict[str, Set[str]]]] = []
        run_topic_presence: List[Set[str]] = []
        num_runs_total = len(group)
        extraction_failures = 0

        for run_idx, record in enumerate(group):
            cache_key = extraction_cache_key_fn(record) if extraction_cache_key_fn else None
            cached = _record_cache_lookup(cache_key=cache_key, extraction_cache=extraction_cache)
            if cached is not None:
                extracted, parse_error, topics_with_patient_text = cached
                if extraction_cache_stats is not None:
                    extraction_cache_stats["behavior_extraction_hits"] = (
                        extraction_cache_stats.get("behavior_extraction_hits", 0) + 1
                    )
            else:
                extracted, parse_error, topics_with_patient_text = _extract_record_attributes(record)
                _record_cache_store(
                    cache_key=cache_key,
                    extraction_cache=extraction_cache,
                    extracted=extracted,
                    parse_error=parse_error,
                    topics_with_patient_text=topics_with_patient_text,
                )
                if extraction_cache_stats is not None:
                    extraction_cache_stats["behavior_extraction_misses"] = (
                        extraction_cache_stats.get("behavior_extraction_misses", 0) + 1
                    )

            if parse_error:
                extraction_failures += 1
                print(
                    f"[BehaviorDiversity] model={model} profile_id={profile_id} "
                    f"run={run_idx + 1}/{num_runs_total} extraction_error={parse_error} "
                    "(ignored for scoring)",
                    flush=True,
                )
                continue

            run_attributes.append(extracted)
            run_topic_presence.append(topics_with_patient_text)
            mentioned_total = sum(
                1
                for topic in extracted.values()
                for values in topic.values()
                if values
            )
            print(
                f"[BehaviorDiversity] model={model} profile_id={profile_id} "
                f"run={run_idx + 1}/{num_runs_total} extracted_attributes={mentioned_total}",
                flush=True,
            )

        num_runs_scored = len(run_attributes)
        if num_runs_scored == 0:
            print(
                f"[BehaviorDiversity] model={model} profile_id={profile_id} "
                "skipped: all runs failed extraction",
                flush=True,
            )
            profile_score = {
                "model": model,
                "profile_id": profile_id,
                "num_runs": num_runs_total,
                "num_runs_scored": 0,
                "num_topics_scored": 0,
                "extraction_failures": extraction_failures,
                "attribute_combination_diversity": 0.0,
                "attribute_value_diversity": 0.0,
                "attribute_density": 0.0,
                "behavior_diversity": 0.0,
                "scored": False,
                "topics": [],
            }
            profile_scores.append(profile_score)
            if on_profile_scored:
                on_profile_scored(profile_score, len(profile_scores), total_groups)
            continue

        if extraction_failures:
            print(
                f"[BehaviorDiversity] model={model} profile_id={profile_id} "
                f"using {num_runs_scored}/{num_runs_total} runs after excluding extraction failures",
                flush=True,
            )

        topic_scores = []
        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
            topic_present_in_any_run = any(topic_key in topic_set for topic_set in run_topic_presence)
            if not topic_present_in_any_run:
                print(
                    f"[BehaviorDiversity] model={model} profile_id={profile_id} "
                    f"topic={topic_key} skipped (no patient text in any run)",
                    flush=True,
                )
                continue

            attrs = spec["attributes"]
            if not attrs:
                continue

            per_run_attr_sets: List[Set[str]] = []
            per_run_topic_values: List[Dict[str, Set[str]]] = []
            for extracted in run_attributes:
                topic_values = extracted.get(topic_key, {})
                mentioned = {attr for attr in attrs if topic_values.get(attr)}
                per_run_attr_sets.append(mentioned)
                per_run_topic_values.append(topic_values)

            pairwise_combo = [
                _jaccard_distance(left, right)
                for left, right in combinations(per_run_attr_sets, 2)
            ]
            combo_div = mean(pairwise_combo) if pairwise_combo else 0.0

            attr_value_diversities = []
            for attr in attrs:
                value_sets = [
                    topic_values.get(attr, set())
                    for topic_values in per_run_topic_values
                    if topic_values.get(attr)
                ]
                pairwise_value_dist = [
                    _jaccard_distance(left, right)
                    for left, right in combinations(value_sets, 2)
                ]
                if pairwise_value_dist:
                    attr_value_diversities.append(mean(pairwise_value_dist))
            raw_value_div = mean(attr_value_diversities) if attr_value_diversities else 0.0
            value_div = _boost_value_diversity(topic_key, raw_value_div)

            density = mean(len(attr_set) / len(attrs) for attr_set in per_run_attr_sets)
            combo_signal = _combo_signal(combo_div)
            topic_behavior_diversity_v1 = mean([value_div, density])
            topic_behavior_linear = (
                (WEIGHT_COMBO_SIGNAL * combo_signal)
                + (WEIGHT_VALUE_DIVERSITY * value_div)
                + (WEIGHT_ATTRIBUTE_DENSITY * density)
            )
            topic_behavior_diversity = topic_behavior_linear**SCORE_CONTRAST_POWER
            topic_weight = _topic_weight(topic_key)

            print(
                f"[BehaviorDiversity] model={model} profile_id={profile_id} "
                f"topic={topic_key} combo_div={combo_div:.3f} combo_signal={combo_signal:.3f} "
                f"value_div_raw={raw_value_div:.3f} value_div={value_div:.3f} density={density:.3f} "
                f"weight={topic_weight:.2f} "
                f"topic_behavior_v1={topic_behavior_diversity_v1:.3f} "
                f"topic_behavior_linear={topic_behavior_linear:.3f} "
                f"topic_behavior_v3={topic_behavior_diversity:.3f}",
                flush=True,
            )

            topic_scores.append(
                {
                    "topic_key": topic_key,
                    "topic_name": spec["topic_name"],
                    "num_runs": num_runs_scored,
                    "possible_attributes": attrs,
                    "topic_weight": topic_weight,
                    "attribute_combination_diversity": combo_div,
                    "attribute_combination_signal": combo_signal,
                    "attribute_value_diversity_raw": raw_value_div,
                    "attribute_value_diversity": value_div,
                    "attribute_density": density,
                    "behavior_diversity_v1": topic_behavior_diversity_v1,
                    "behavior_diversity_linear": topic_behavior_linear,
                    "behavior_diversity": topic_behavior_diversity,
                }
            )

        combo_mean = _weighted_mean(
            (item["attribute_combination_diversity"], item.get("topic_weight", 1.0))
            for item in topic_scores
        )
        combo_signal_mean = _weighted_mean(
            (item["attribute_combination_signal"], item.get("topic_weight", 1.0))
            for item in topic_scores
        )
        value_mean = _weighted_mean(
            (item["attribute_value_diversity"], item.get("topic_weight", 1.0))
            for item in topic_scores
        )
        value_raw_mean = _weighted_mean(
            (item.get("attribute_value_diversity_raw", item["attribute_value_diversity"]), item.get("topic_weight", 1.0))
            for item in topic_scores
        )
        density_mean = _weighted_mean(
            (item["attribute_density"], item.get("topic_weight", 1.0))
            for item in topic_scores
        )
        behavior_div_v1 = _weighted_mean(
            (item["behavior_diversity_v1"], item.get("topic_weight", 1.0))
            for item in topic_scores
        )
        behavior_div_linear = _weighted_mean(
            (item["behavior_diversity_linear"], item.get("topic_weight", 1.0))
            for item in topic_scores
        )
        behavior_div = _weighted_mean(
            (item["behavior_diversity"], item.get("topic_weight", 1.0))
            for item in topic_scores
        )

        print(
            f"[BehaviorDiversity] model={model} profile_id={profile_id} "
            f"num_runs_scored={num_runs_scored}/{num_runs_total} num_topics_scored={len(topic_scores)} "
            f"combo_mean={combo_mean:.3f} combo_signal_mean={combo_signal_mean:.3f} "
            f"value_raw_mean={value_raw_mean:.3f} value_mean={value_mean:.3f} density_mean={density_mean:.3f} "
            f"behavior_v1={behavior_div_v1:.3f} behavior_linear={behavior_div_linear:.3f} "
            f"behavior_v3={behavior_div:.3f}",
            flush=True,
        )

        profile_score = {
            "model": model,
            "profile_id": profile_id,
            "num_runs": num_runs_total,
            "num_runs_scored": num_runs_scored,
            "num_topics_scored": len(topic_scores),
            "extraction_failures": extraction_failures,
            "attribute_combination_diversity": combo_mean,
            "attribute_combination_signal": combo_signal_mean,
            "attribute_value_diversity_raw": value_raw_mean,
            "attribute_value_diversity": value_mean,
            "attribute_density": density_mean,
            "behavior_diversity_v1": behavior_div_v1,
            "behavior_diversity_linear": behavior_div_linear,
            "behavior_diversity": behavior_div,
            "scored": True,
            "topics": topic_scores,
        }
        profile_scores.append(profile_score)
        if on_profile_scored:
            on_profile_scored(profile_score, len(profile_scores), total_groups)

    profile_scores.sort(key=lambda item: (str(item.get("model")), str(item.get("profile_id"))))
    scored_profiles = [item for item in profile_scores if item.get("scored", True)]
    overall = mean(item.get("behavior_diversity", 0.0) for item in scored_profiles)
    print(
        f"[BehaviorDiversity] complete variant={METRIC_VARIANT} overall_score={overall:.3f} "
        f"profiles_scored={len(scored_profiles)}/{len(profile_scores)}",
        flush=True,
    )

    return {
        "metric": "behavior_diversity",
        "variant": METRIC_VARIANT,
        "weights": {
            "combo_log_gain": COMBO_LOG_GAIN,
            "combination_signal": WEIGHT_COMBO_SIGNAL,
            "value_diversity": WEIGHT_VALUE_DIVERSITY,
            "attribute_density": WEIGHT_ATTRIBUTE_DENSITY,
            "value_diversity_global_boost": VALUE_DIVERSITY_GLOBAL_BOOST,
            "value_diversity_topic_boosts": VALUE_DIVERSITY_TOPIC_BOOSTS,
            "contrast_power": SCORE_CONTRAST_POWER,
            "topic_overrides": TOPIC_WEIGHT_OVERRIDES,
        },
        "score": overall,
        "num_profiles_scored": len(scored_profiles),
        "num_profiles_total": len(profile_scores),
        "profiles": profile_scores,
    }
