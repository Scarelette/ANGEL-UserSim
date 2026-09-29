import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_JSONL_PATH = Path(__file__).resolve().parents[1] / "examples" / "profiles.jsonl"


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
    # speaking_style is optional and never type-checked, so it cannot fail a request.
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
    """Validate a rich profile and return its public fields plus the profile itself.

    The Actor renders ``_raw_profile`` into its prompt; the other fields name the
    patient in listings and API responses.
    """
    validate_rich_profile(raw_profile)
    identity = _ensure_dict(raw_profile.get("identity"))
    return {
        "profile_id": make_profile_id(raw_profile, line_num),
        "name": identity.get("name", "Unknown"),
        "age": identity.get("age", "unknown"),
        "gender": identity.get("gender", "unspecified"),
        "role": identity.get("role", "patient"),
        "source_title": get_profile_source(raw_profile),
        "_raw_profile": raw_profile,
    }


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
