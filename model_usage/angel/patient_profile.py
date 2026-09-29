import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_JSONL_PATH = Path(__file__).resolve().parents[1] / "examples" / "profiles.jsonl"


def _ensure_list(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return value
    return [value]


def _ensure_dict(value) -> Dict[str, Any]:
    """Coerce a field documented as an object into a dict.

    Rich profiles arrive from the public API, where a caller may send an object
    field as a bare string. Rule-style fields are read as `.get(key, default)`,
    so a non-mapping has no meaningful interpretation -> fall back to {} and let
    the documented defaults apply, rather than raising AttributeError mid-request.
    """
    if isinstance(value, dict):
        return value
    return {}


def format_bullet_list(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


STYLE_KEYS = ["age_level", "tone", "vocabulary", "sentence_style", "disclosure_style"]


def _coerce_style_items(value: Any) -> List[str]:
    """Turn any shape of `speaking_style` into the internal list of style phrases.

    speaking_style is never a reason to reject a profile: it is optional, and
    callers send it as the documented object, as one freeform sentence, as a list
    of phrases, or not at all. Anything unusable yields [] so the caller gets the
    "natural and conversational" default instead of an error.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        items: List[str] = []
        for key in STYLE_KEYS:
            items.extend(_coerce_style_items(value.get(key)))
        if not items:
            # Only when no documented key produced anything: undocumented keys
            # still carry style information, so use them rather than drop them.
            for key, entry in value.items():
                if key not in STYLE_KEYS:
                    items.extend(_coerce_style_items(entry))
        return items
    if isinstance(value, (list, tuple, set)):
        items = []
        for entry in value:
            items.extend(_coerce_style_items(entry))
        return items
    if not value:                      # "", 0, False -> nothing to say
        return []
    text = str(value).strip()
    return [text] if text else []


def is_rich_profile_schema(profile: Dict[str, Any]) -> bool:
    # `identity` alone discriminates the rich schema. speaking_style is optional,
    # so requiring it here would reject valid profiles that simply omit it.
    return isinstance(profile, dict) and "identity" in profile


def validate_rich_profile(profile: Dict[str, Any]) -> None:
    required_fields = [
        "identity",
        "background",
        "presenting_problems",
        "triggers",
        "emotions",
        "behaviors",
        "family_context",
        "hidden_state",
        "disclosure_rules",
        "simulation_rules",
    ]
    # speaking_style is deliberately NOT required and never type-checked: any
    # shape is accepted and normalized by _coerce_style_items, and an absent one
    # falls back to the default style. It must never fail a request.
    if not isinstance(profile, dict):
        raise ValueError(f"Rich profile must be an object, got {type(profile).__name__}")

    missing = [field for field in required_fields if field not in profile]
    if missing:
        raise ValueError(f"Rich profile is missing required fields: {missing}")

    for field in ["disclosure_rules", "simulation_rules"]:
        if not isinstance(profile[field], dict):
            raise ValueError(f"{field} must be an object, got {type(profile[field]).__name__}")

    identity = profile["identity"]
    # Must precede the membership loop: `"name" not in "some string"` is a
    # substring test, so a string identity would pass and crash downstream.
    if not isinstance(identity, dict):
        raise ValueError(f"identity must be an object, got {type(identity).__name__}")
    for field in ["name", "age", "gender", "role"]:
        if field not in identity:
            raise ValueError(f"identity.{field} is required")


def make_profile_id(raw_profile: Dict[str, Any], line_num: int | None = None) -> str:
    identity = _ensure_dict(raw_profile.get("identity"))
    if identity.get("id"):
        return str(identity["id"])

    name = str(identity.get("name", "unknown")).strip().lower().replace(" ", "_")
    source_title = str(identity.get("source_title", _ensure_dict(raw_profile.get("_meta")).get("source_title", "unknown_source")))
    source_title = source_title.strip().lower().replace(" ", "_")
    if line_num is None:
        return f"{name}__{source_title}"
    return f"{name}__{source_title}"


def get_profile_name(raw_profile: Dict[str, Any]) -> str:
    return _ensure_dict(raw_profile.get("identity")).get("name", "Unknown")


def get_profile_source(raw_profile: Dict[str, Any]) -> str:
    return (_ensure_dict(raw_profile.get("identity")).get("source_title")
            or _ensure_dict(raw_profile.get("_meta")).get("source_title", ""))


def get_profile_label(raw_profile: Dict[str, Any], line_num: int | None = None) -> str:
    name = get_profile_name(raw_profile)
    age = _ensure_dict(raw_profile.get("identity")).get("age", "")
    source = get_profile_source(raw_profile)
    profile_id = make_profile_id(raw_profile, line_num)

    suffix = f" ({source})" if source else ""
    age_text = f", {age}" if age else ""
    # return f"{name}{age_text}{suffix} [{profile_id}]"
    return f"{name}{age_text}{suffix}"


def convert_rich_profile_to_internal(raw_profile: Dict[str, Any], line_num: int | None = None) -> Dict[str, Any]:
    validate_rich_profile(raw_profile)

    identity = _ensure_dict(raw_profile.get("identity"))
    disclosure_rules = _ensure_dict(raw_profile.get("disclosure_rules"))
    simulation_rules = _ensure_dict(raw_profile.get("simulation_rules"))

    background_list = _ensure_list(raw_profile.get("background"))
    background_text = " ".join(str(x).strip() for x in background_list if str(x).strip())

    style_items = _coerce_style_items(raw_profile.get("speaking_style"))

    behavior_rules = []
    if simulation_rules.get("stay_in_character", True):
        behavior_rules.append("stay in character as the patient")
    if simulation_rules.get("speak_as_patient_only", True):
        behavior_rules.append("speak as the patient only")
    if simulation_rules.get("no_narration", True):
        behavior_rules.append("do not narrate actions or hidden reasoning")
    if simulation_rules.get("no_researcher_voice", True):
        behavior_rules.append("do not sound like a researcher or case report")
    if simulation_rules.get("keep_responses_conversational", True):
        behavior_rules.append("keep responses conversational")
    if simulation_rules.get("do_not_invent_major_facts", True):
        behavior_rules.append("do not invent major life facts not supported by the profile")
    if simulation_rules.get("if_unsure_say_limited_knowledge", True):
        behavior_rules.append("if unsure, respond naturally with limited knowledge")
    if disclosure_rules.get("reveal_gradually", True):
        behavior_rules.append("reveal personal history gradually instead of all at once")
    if disclosure_rules.get("do_not_dump_case_summary", True):
        behavior_rules.append("do not dump the whole case summary in one response")
    if disclosure_rules.get("do_not_use_clinical_jargon", True):
        behavior_rules.append("do not use clinical jargon unless the user introduces it")

    sensitive_topics = []
    sensitive_topics.extend(_ensure_list(disclosure_rules.get("topics_likely_late")))
    sensitive_topics.extend(_ensure_list(disclosure_rules.get("topics_avoid_unless_asked")))
    if not sensitive_topics:
        sensitive_topics = _ensure_list(raw_profile.get("hidden_state"))

    profile_id = make_profile_id(raw_profile, line_num)

    internal = {
        "profile_id": profile_id,
        "name": identity.get("name", "Unknown"),
        "age": identity.get("age", "unknown"),
        "gender": identity.get("gender", "unspecified"),
        "role": identity.get("role", "patient"),
        "background": background_text,
        "presenting_problems": _ensure_list(raw_profile.get("presenting_problems")),
        "speaking_style": style_items if style_items else ["natural and conversational"],
        "behavior_rules": behavior_rules,
        "core_beliefs": _ensure_list(raw_profile.get("hidden_state")),
        "current_emotions": _ensure_list(raw_profile.get("emotions")),
        "current_behaviors": _ensure_list(raw_profile.get("behaviors")),
        "sensitive_topics": sensitive_topics,
        "triggers": _ensure_list(raw_profile.get("triggers")),
        "family_context": _ensure_list(raw_profile.get("family_context")),
        "diagnosis_hint": _ensure_list(identity.get("diagnosis_hint")),
        "topics_likely_early": _ensure_list(disclosure_rules.get("topics_likely_early")),
        "topics_likely_late": _ensure_list(disclosure_rules.get("topics_likely_late")),
        "topics_avoid_unless_asked": _ensure_list(disclosure_rules.get("topics_avoid_unless_asked")),
        "source_title": identity.get("source_title") or _ensure_dict(raw_profile.get("_meta")).get("source_title", ""),
        "_raw_profile": raw_profile,
    }
    return internal


def load_profiles_from_jsonl(jsonl_path: Path = DEFAULT_JSONL_PATH) -> List[Tuple[int, Dict[str, Any]]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    profiles = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not is_rich_profile_schema(raw):
                raise ValueError(f"Invalid JSONL record at line {line_num}: must follow rich patient schema")
            profiles.append((line_num, raw))

    if not profiles:
        raise ValueError(f"No valid profiles found in {jsonl_path}")
    return profiles


def list_profile_options(jsonl_path: Path = DEFAULT_JSONL_PATH) -> List[Dict[str, str]]:
    options = []
    for line_num, raw in load_profiles_from_jsonl(jsonl_path):
        options.append({
            "id": make_profile_id(raw, line_num),
            "label": get_profile_label(raw, line_num),
        })
    return options


def load_profile_by_id(profile_id: str, jsonl_path: Path = DEFAULT_JSONL_PATH) -> Dict[str, Any]:
    for line_num, raw in load_profiles_from_jsonl(jsonl_path):
        if make_profile_id(raw, line_num) == profile_id:
            return convert_rich_profile_to_internal(raw, line_num)
    raise ValueError(f"Profile id '{profile_id}' not found")
