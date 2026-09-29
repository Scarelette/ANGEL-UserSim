"""Two-stage Angel initializer for profile-expansion experiments.

Stage 1:
- Expand short profile -> structured long profile JSON using the same prompt
  and generation utilities (stage1_short2long.py, ported from the Observer
  training repo).

Stage 2:
- Initialize the Angel patient simulator with the long profile dict and run
  dialogue generation through patients/angel.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


from angel_common.paths import resolve_model
from experiments.profile_expansion import stage1_short2long

# Resolved through angel_common.paths: --flag -> ANGEL_OBSERVER_MODEL / ANGEL_ACTOR_MODEL
# env var -> models/<name> -> bare name. None means "use the resolver default".
DEFAULT_ANGEL_STAGE1_MODEL_NAME: Optional[str] = None
DEFAULT_ANGEL_STAGE2_MODEL_NAME: Optional[str] = None


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = " ".join(value.split()).strip()
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_coerce_list(item))
        return out
    text = " ".join(str(value).split()).strip()
    return [text] if text else []


def _merge_lists(*values: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        for item in _coerce_list(value):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def _resolve_stage1_model_path(stage1_model_name: Optional[str]) -> str:
    return resolve_model("observer", stage1_model_name)


def _resolve_stage2_model_path(stage2_model_name: Optional[str]) -> str:
    return resolve_model("actor", stage2_model_name)


def _load_stage1_module():
    return stage1_short2long


def _extract_short_profile_text(profile_item: Dict[str, Any]) -> str:
    short_profile = profile_item.get("short_patient_profile")
    if isinstance(short_profile, str) and short_profile.strip():
        return short_profile.strip()

    patient_payload = profile_item.get("patient_processed_result")
    if isinstance(patient_payload, dict):
        complaints = patient_payload.get("complaints")
        if isinstance(complaints, str) and complaints.strip():
            return complaints.strip()

    complaints = profile_item.get("Complaints")
    if isinstance(complaints, str) and complaints.strip():
        return complaints.strip()

    raise ValueError("Missing short profile text. Need short_patient_profile or complaints.")


def _age_level_from_identity(identity: Dict[str, Any]) -> str:
    age_value = identity.get("age")
    if isinstance(age_value, int):
        return "adolescent" if age_value < 18 else "adult"
    if isinstance(age_value, str):
        match = re.search(r"\d+", age_value)
        if match:
            age_num = int(match.group(0))
            return "adolescent" if age_num < 18 else "adult"
    return "adult"


def _is_eval_angel_schema(profile_dict: Dict[str, Any]) -> bool:
    required = {
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
    }
    return isinstance(profile_dict, dict) and required.issubset(set(profile_dict.keys()))


def _adapt_stage1_profile_to_angel(
    stage1_profile: Dict[str, Any],
    *,
    source_title: str,
    short_profile_text: str,
) -> Dict[str, Any]:
    identity = stage1_profile.get("identity") or {}
    symptom_details = stage1_profile.get("symptom_details") or {}
    emotional = stage1_profile.get("emotional_profile") or {}
    cognitive = stage1_profile.get("cognitive_patterns") or {}
    behavior = stage1_profile.get("behavior_patterns") or {}
    social = stage1_profile.get("social_and_family_context") or {}
    hidden = stage1_profile.get("hidden_state") or {}
    speaking = stage1_profile.get("speaking_style") or {}
    disclosure = stage1_profile.get("disclosure_rules") or {}
    summary = stage1_profile.get("brief_summary")

    background = _merge_lists(stage1_profile.get("background"), summary, short_profile_text)
    presenting = _merge_lists(stage1_profile.get("presenting_problems"))
    triggers = _merge_lists(stage1_profile.get("triggers"), symptom_details.get("triggers"))
    emotions = _merge_lists(
        stage1_profile.get("emotions"),
        emotional.get("dominant_emotions"),
        emotional.get("secondary_emotions"),
        emotional.get("emotional_expression_style"),
    )
    behaviors = _merge_lists(
        stage1_profile.get("behaviors"),
        behavior.get("coping_strategies"),
        behavior.get("avoidance_behaviors"),
        behavior.get("safety_behaviors"),
        behavior.get("interpersonal_patterns"),
        behavior.get("maladaptive_coping"),
        symptom_details.get("functional_impairment"),
    )
    family_context = _merge_lists(
        stage1_profile.get("family_context"),
        social.get("family_relationships"),
        social.get("romantic_or_peer_relationships"),
        social.get("school_or_work_context"),
        social.get("social_support"),
        social.get("cultural_or_contextual_factors"),
    )
    hidden_state = _merge_lists(
        stage1_profile.get("hidden_state"),
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
        "age_level": _age_level_from_identity(identity),
        "tone": speaking.get("tone") or "natural and conversational",
        "vocabulary": speaking.get("word_choice") or "everyday, non-technical",
        "sentence_style": speaking.get("verbosity") or "short to medium responses",
        "disclosure_style": speaking.get("interaction_style") or "gradual disclosure",
    }

    topics_likely_early = _merge_lists(disclosure.get("early_session"))
    topics_likely_late = _merge_lists(disclosure.get("middle_session"), disclosure.get("late_session"))
    topics_avoid = _merge_lists(hidden.get("topics_patient_avoids"))

    disclosure_rules = {
        "reveal_gradually": True,
        "do_not_dump_case_summary": True,
        "do_not_use_clinical_jargon": True,
        "topics_likely_early": topics_likely_early,
        "topics_likely_late": topics_likely_late,
        "topics_avoid_unless_asked": topics_avoid,
    }

    simulation_rules = {
        "stay_in_character": True,
        "speak_as_patient_only": True,
        "no_narration": True,
        "no_researcher_voice": True,
        "keep_responses_conversational": True,
        "do_not_invent_major_facts": True,
        "if_unsure_say_limited_knowledge": True,
    }

    return {
        "identity": {
            "name": identity.get("name"),
            "age": identity.get("age"),
            "gender": identity.get("gender") or "unspecified",
            "role": identity.get("role") or "patient",
            "source_title": identity.get("source_title") or source_title or "",
            "diagnosis_hint": _coerce_list(identity.get("diagnosis_hint")),
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


def _build_minimal_angel_profile(profile_item: Dict[str, Any], short_profile_text: str) -> Dict[str, Any]:
    source_title = str(profile_item.get("source_title") or "").strip()
    name = profile_item.get("name")
    age = profile_item.get("age")
    gender = profile_item.get("gender") or "unspecified"
    diagnosis_hint = _coerce_list(profile_item.get("diagnosis_hint"))

    return {
        "identity": {
            "name": name,
            "age": age,
            "gender": gender,
            "role": "patient",
            "source_title": source_title,
            "diagnosis_hint": diagnosis_hint,
        },
        "background": [short_profile_text],
        "presenting_problems": [short_profile_text],
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
        "_meta": {
            "source_title": source_title,
            "stage1_schema_adapted": False,
            "fallback_profile": True,
        },
    }


class TwoStageAngelPatient:
    """Angel wrapper that supports short-profile stage-1 expansion."""

    def __init__(
        self,
        *,
        stage1_model_name: Optional[str] = DEFAULT_ANGEL_STAGE1_MODEL_NAME,
        stage2_model_name: Optional[str] = DEFAULT_ANGEL_STAGE2_MODEL_NAME,
        device_map: str = "auto",
        stage1_torch_dtype: str = "bfloat16",
        stage1_max_new_tokens: int = 3072,
        stage1_temperature: float = 0.1,
        stage1_top_p: float = 0.9,
        stage1_max_attempts: int = 2,
        verbose: bool = False,
    ):
        self.stage1_model_name = _resolve_stage1_model_path(stage1_model_name)
        self.stage2_model_name = _resolve_stage2_model_path(stage2_model_name)
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_temperature = stage1_temperature
        self.stage1_top_p = stage1_top_p
        self.stage1_max_attempts = max(1, int(stage1_max_attempts))
        self.verbose = verbose
        self._log(
            "[AngelInit] "
            f"stage1_model={self.stage1_model_name} stage2_model={self.stage2_model_name}"
        )

        self._stage1_utils = _load_stage1_module()
        self._stage1_generator = self._stage1_utils.LocalProfileGenerator(
            model_path=self.stage1_model_name,
            torch_dtype=stage1_torch_dtype,
        )

        from experiments.profile_expansion.patients.angel import Angel

        self._angel_patient = Angel(
            profile="",
            model_name=self.stage2_model_name,
            device_map=device_map,
        )
        self.system_prompt = self._angel_patient.system_prompt
        self.active_profile = self._angel_patient.active_profile
        self.state_manager = self._angel_patient.state_manager

        self._cached_profile_key: Optional[str] = None
        self._cached_profile_dict: Optional[Dict[str, Any]] = None
        self.last_stage1: Optional[Dict[str, Any]] = None
        self.last_long_profile: Optional[Dict[str, Any]] = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _compute_cache_key(self, profile_item: Dict[str, Any], short_profile_text: str) -> str:
        key_payload = {
            "id": profile_item.get("id"),
            "source_title": profile_item.get("source_title"),
            "name": profile_item.get("name"),
            "short_patient_profile": short_profile_text,
        }
        return json.dumps(key_payload, ensure_ascii=False, sort_keys=True)

    def _generate_from_short_profile(self, profile_item: Dict[str, Any]) -> Dict[str, Any]:
        short_profile_text = _extract_short_profile_text(profile_item)
        cache_key = self._compute_cache_key(profile_item, short_profile_text)

        if self._cached_profile_key == cache_key and isinstance(self._cached_profile_dict, dict):
            self._log("[AngelInit][Stage1] cache hit: reusing generated long profile.")
            return self._cached_profile_dict

        self._log(
            "[AngelInit][Stage1] generating long profile from short profile "
            f"(chars={len(short_profile_text)})"
        )

        stage1: Optional[Dict[str, Any]] = None
        profile_json: Optional[Dict[str, Any]] = None

        for attempt in range(1, self.stage1_max_attempts + 1):
            stage1 = self._stage1_utils.generate_stage1_profile(
                generator=self._stage1_generator,
                short_profile=short_profile_text,
                max_new_tokens=self.stage1_max_new_tokens,
                temperature=self.stage1_temperature,
                top_p=self.stage1_top_p,
            )
            self.last_stage1 = stage1
            profile_json = self._parse_stage1_profile(stage1)

            if isinstance(profile_json, dict):
                if attempt > 1:
                    self._log(f"[AngelInit][Stage1] parse succeeded on retry attempt={attempt}")
                break

            parse_error = stage1.get("profile_json_parse_error")
            preview = (stage1.get("cleaned_model_output") or "").replace("\n", " ").strip()
            if len(preview) > 220:
                preview = preview[:220] + "..."
            self._log(
                "[AngelInit][Stage1] parse failed. "
                f"attempt={attempt}/{self.stage1_max_attempts} "
                f"parse_error={parse_error} preview={preview!r}"
            )

        if not isinstance(profile_json, dict):
            repaired = self._repair_stage1_profile(
                short_profile_text=short_profile_text,
                stage1=stage1 or {},
            )
            if isinstance(repaired, dict):
                profile_json = repaired
                self._log("[AngelInit][Stage1] parse recovered via repair pass.")

        if not isinstance(profile_json, dict):
            self._log(
                "[AngelInit][Stage1] JSON parse failed after retries/repair; using minimal fallback profile.",
            )
            adapted_profile = _build_minimal_angel_profile(profile_item, short_profile_text)
        elif _is_eval_angel_schema(profile_json):
            self._log("[AngelInit][Stage1] parsed profile already in Angel eval schema.")
            adapted_profile = profile_json
        else:
            self._log("[AngelInit][Stage1] adapting stage-1 schema to Angel eval schema.")
            adapted_profile = _adapt_stage1_profile_to_angel(
                profile_json,
                source_title=str(profile_item.get("source_title") or "").strip(),
                short_profile_text=short_profile_text,
            )

        self._cached_profile_key = cache_key
        self._cached_profile_dict = adapted_profile
        return adapted_profile

    def _parse_stage1_profile(self, stage1: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        profile_json = stage1.get("profile_json")
        if isinstance(profile_json, dict):
            return profile_json

        cleaned = stage1.get("cleaned_model_output")
        if isinstance(cleaned, str):
            cleaned_text = cleaned.strip()
            if cleaned_text:
                try:
                    parsed = json.loads(cleaned_text)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        return parsed[0]
                except Exception:
                    pass

                try:
                    json_str = self._stage1_utils.extract_first_json_object(cleaned_text)
                    if json_str:
                        parsed_obj = json.loads(json_str)
                        if isinstance(parsed_obj, dict):
                            return parsed_obj
                except Exception:
                    pass
        return None

    def _repair_stage1_profile(
        self,
        *,
        short_profile_text: str,
        stage1: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        cleaned = (stage1.get("cleaned_model_output") or "").strip()
        raw = (stage1.get("raw_model_output") or "").strip()
        draft = cleaned or raw
        if not draft:
            return None

        repair_prompt = (
            "The following model draft failed JSON parsing. "
            "Rewrite it into one valid JSON object that follows the required schema. "
            "Return JSON only.\n\n"
            f"Short patient description:\n{short_profile_text}\n\n"
            f"Draft output to repair:\n{draft}"
        )
        messages = [
            {"role": "system", "content": self._stage1_utils.SHORT_TO_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": repair_prompt},
        ]
        try:
            repaired_text = self._stage1_generator.generate(
                messages,
                max_new_tokens=self.stage1_max_new_tokens,
                temperature=0.0,
                top_p=0.9,
            )
            repaired_clean = self._stage1_utils.clean_model_output(repaired_text)
            try:
                parsed = json.loads(repaired_clean)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed[0]
            except Exception:
                pass

            try:
                return self._stage1_utils.parse_profile_json(repaired_clean)
            except Exception:
                return None
        except Exception as exc:
            self._log(f"[AngelInit][Stage1] repair pass failed: {exc}")
            return None

    def initialize_from_profile_item(self, profile_item: Dict[str, Any]) -> Dict[str, Any]:
        payload = profile_item.get("angel_processed_result")
        if isinstance(payload, dict) and _is_eval_angel_schema(payload):
            self._log("[AngelInit] using existing angel_processed_result from input row.")
            selected_profile = payload
        elif isinstance(payload, dict):
            self._log("[AngelInit] adapting provided angel_processed_result schema.")
            short_profile_text = _extract_short_profile_text(profile_item)
            selected_profile = _adapt_stage1_profile_to_angel(
                payload,
                source_title=str(profile_item.get("source_title") or "").strip(),
                short_profile_text=short_profile_text,
            )
        else:
            selected_profile = self._generate_from_short_profile(profile_item)

        self.last_long_profile = selected_profile
        self._angel_patient.set_profile_from_dict(selected_profile)
        self.system_prompt = self._angel_patient.system_prompt
        self.active_profile = self._angel_patient.active_profile
        self.state_manager = self._angel_patient.state_manager

        source_title = selected_profile.get("identity", {}).get("source_title")
        self._log(
            "[AngelInit][Stage2] profile initialized for Angel generation. "
            f"source_title={source_title!r}"
        )
        return selected_profile

    def get_stage1_long_profile(self) -> Optional[Dict[str, Any]]:
        return self.last_long_profile

    async def generate(self, conversation: List[Dict[str, Any]], **kwargs) -> str:
        return await self._angel_patient.generate(conversation, **kwargs)
