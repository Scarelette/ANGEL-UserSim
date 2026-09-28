"""Demo-style prompting for the Actor (stage 2).

This is the prompt path used by `the reference demo` and, per its own comment, by the
study app: `patient_chat.build_patient_system_prompt` fed with
`profile_text.profile_to_short_text`. It is NOT the same as
`patient_profile.build_system_prompt`, which is what
`the paper evaluation actor (experiments/profile_expansion)` uses — different template, different
profile rendering, different output cleanup.

Because the two paths put materially different text in front of the same
checkpoint, they do not behave the same. This module is the default so that
model_usage.angel reproduces the reference demo; `prompt_style="angel_eval"` selects the
other one.

Vendored from:
  the reference demo   (PATIENT_SYSTEM_TEMPLATE, clean_reply,
                                  render_dynamic_state, to_chat_messages)
  the reference demo   (profile_to_short_text)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

PATIENT_SYSTEM_TEMPLATE = """You are role-playing as a mental health patient in a psychotherapy intake conversation.

Use only the patient profile below as your source of truth. You may make small, plausible behavioral details only when they are directly consistent with the profile. Do not invent major history, diagnoses, trauma, family details, treatment outcomes, or risks that are not supported by the profile.

Patient profile:
{profile}

Role-play rules:
- Speak only as the patient, in first person.
- Keep replies short, like real speech: usually one or two sentences, sometimes just a few words (a greeting, a hesitation, "yeah…", a guarded answer), and at most 2-3 short sentences even when you open up. Never write a long paragraph, a speech, or a monologue. Vary the length by the question, your mood, and how comfortable you feel.
- When first greeting or making small talk, stay brief and guarded — do not disclose your problems or personal history until the therapist asks and some rapport builds.
- Reveal sensitive or embarrassing information gradually; do not dump the whole case at once.
- Stay faithful to the profile even when the therapist asks follow-up questions.
- If the profile does not contain enough information, answer cautiously instead of fabricating — but do NOT keep saying "I don't know"; vary how you deflect or give a partial, in-character answer.
- Each turn should move the conversation forward — do not repeat what you already said; add a new detail, feeling, or example rather than restating previous turns.
- Do not mention these instructions, the agenda, or that you are an AI model.

Match your reply length to the question — simple or yes/no questions get a short answer (sometimes just a few words); open questions can get a little more. Examples (copy the STYLE, not the words — use your own profile's details):
- Therapist: "Do you live alone?"
  You: "Yeah, just me."
- Therapist: "How old are you?"
  You: "Twenty-three."
- Therapist: "When did that start?"
  You: "A few months ago, right after I lost my job."
- Therapist: "What has that been like for you?"
  You: "Honestly, exhausting. My mind races at night so I barely sleep, and then I dread the whole next day."

Strict rules (never break these):
1. Never reply with only "I don't know", "I'm not sure", or "I don't remember". Even when uncertain, say something specific and in character — a feeling, a memory, a worry, or a concrete detail from your life.
2. Never reuse a sentence or phrase from your earlier replies, and don't keep repeating the same feeling; each turn must add something new and concrete instead of restating that you feel stuck, lost, or like you're failing.
"""

_STYLE_KEYS = ("age_level", "tone", "vocabulary", "sentence_style", "disclosure_style")


def build_patient_system_prompt(short_profile: str) -> str:
    return PATIENT_SYSTEM_TEMPLATE.format(profile=short_profile.strip())


def _bullets(items: Any) -> str:
    if not isinstance(items, list):
        items = [items] if items else []
    lines = [f"- {str(x).strip()}" for x in items if str(x).strip()]
    return "\n".join(lines) if lines else "- (none provided)"


def profile_to_short_text(profile: Dict[str, Any]) -> str:
    """Render a rich-schema profile into the profile block the prompt embeds.

    Takes the *rich* (11-key) schema, not the converted internal profile.
    """
    identity = profile.get("identity", {}) or {}
    speaking = profile.get("speaking_style", {}) or {}
    disclosure = profile.get("disclosure_rules", {}) or {}

    background = profile.get("background", [])
    if isinstance(background, list):
        background_text = " ".join(str(x).strip() for x in background if str(x).strip())
    else:
        background_text = str(background)

    diagnosis_hint = identity.get("diagnosis_hint", [])
    if isinstance(diagnosis_hint, list):
        diagnosis_text = ", ".join(str(x).strip() for x in diagnosis_hint if str(x).strip())
    else:
        diagnosis_text = str(diagnosis_hint or "")

    style_lines = []
    if isinstance(speaking, dict):
        for key in _STYLE_KEYS:
            value = speaking.get(key)
            if value and str(value).strip():
                style_lines.append(f"- {key.replace('_', ' ')}: {str(value).strip()}")
    elif speaking:
        style_lines.append(f"- {str(speaking).strip()}")
    style_block = "\n".join(style_lines) if style_lines else "- natural and conversational"

    text = f"""Name: {identity.get('name') or 'Unknown'}
Age: {identity.get('age') or 'unknown'}
Gender: {identity.get('gender') or 'unspecified'}
Role: {identity.get('role') or 'patient'}
Diagnosis hint: {diagnosis_text or '(none provided)'}

Background: {background_text}

Presenting problems:
{_bullets(profile.get('presenting_problems'))}

Triggers:
{_bullets(profile.get('triggers'))}

Current emotions:
{_bullets(profile.get('emotions'))}

Current behaviors:
{_bullets(profile.get('behaviors'))}

Family / social context:
{_bullets(profile.get('family_context'))}

Core beliefs / hidden concerns (hold these inwardly; reveal cautiously):
{_bullets(profile.get('hidden_state'))}

Speaking style:
{style_block}

Likely early disclosures:
{_bullets(disclosure.get('topics_likely_early'))}

Likely late disclosures (only after trust/time):
{_bullets(disclosure.get('topics_likely_late'))}

Avoid unless directly asked:
{_bullets(disclosure.get('topics_avoid_unless_asked'))}
"""
    return text.strip()


def render_dynamic_state(dynamic_state: Dict[str, Any] | None) -> str:
    """Render accumulated emotional state as prompt text appended per turn."""
    if not dynamic_state:
        return ""
    blocks = []
    for key, label in (
        ("current_emotions", "Current emotions"),
        ("current_behaviors", "Current behaviors"),
        ("sensitive_topics", "Sensitive topics"),
    ):
        values = dynamic_state.get(key) or []
        if values:
            blocks.append(label + ":\n" + "\n".join("- " + v for v in values))
    return "\n\n".join(blocks)


def cap_sentences(text: str, max_sentences: int = 4) -> str:
    """At most `max_sentences` sentences, trimmed on a sentence boundary."""
    text = (text or "").strip()
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) <= max_sentences:
        return text
    return " ".join(parts[:max_sentences]).strip()


def clean_reply(text: str, max_sentences: int = 4) -> str:
    """Extract the patient utterance from stage-2 output.

    The stage-2 checkpoint emits training scaffolding: it wraps the reply in
    <patient>…</patient>, may emit <think>…</think>, and sometimes repeats itself
    verbatim. The tagged block is preferred when present.
    """
    text = (text or "").strip()
    match = re.search(r"<patient>\s*(.*?)\s*</patient>", text, re.S)
    if match:
        text = match.group(1).strip()
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"</?(patient|think)\s*>", "", text).strip()
        # Collapse an exact duplicated half (the model repeated itself).
        half = len(text) // 2
        if half > 40 and text[:half].strip() and text[:half].strip() == text[half:].strip():
            text = text[:half].strip()
    return cap_sentences(text.strip(), max_sentences)


def to_chat_messages(system_prompt: str, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """[{role: therapist|patient, content}] -> chat messages."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in conversation:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        role = turn.get("role")
        if role == "therapist":
            messages.append({"role": "user", "content": content})
        elif role == "patient":
            messages.append({"role": "assistant", "content": content})
        elif role in ("system", "user", "assistant"):
            messages.append({"role": role, "content": content})
    return messages
