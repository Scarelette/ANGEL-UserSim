"""Cleanup of raw Actor output.

Ported from the paper evaluation actor (experiments/profile_expansion). These are pure functions with
no model dependency, which is what makes them testable without a GPU.

The chain is: pull out any <state> block, drop <think> artifacts, strip role
prefixes, normalize whitespace, drop duplicate paragraphs, trim a truncated
tail, then cap sentence count.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

FALLBACK_REPLY = (
    "It's hard to put into words, but it just feels like a lot pressing down on me right now."
)

# "I don't know / I do not know / I don't remember / I'm not sure / not really
# sure / no idea", tolerant of straight or curly apostrophes.
_REFUSAL_RE = re.compile(
    r"i\s+(?:do\s*n[’']?t|do\s+not)\s+(?:know|remember)"
    r"|i\s*[’']?m\s+not\s+sure"
    r"|not\s+really\s+sure"
    r"|no\s+idea",
    flags=re.IGNORECASE,
)

_QUESTION_STARTERS = (
    "is it easier",
    "any changes from yesterday",
    "what did you do differently",
    "how does it feel now compared to before",
    "what worries you most",
    "what triggers your anxiety",
    "what makes you want to avoid things",
    "what helps you calm down",
    "what happens if you try",
    "what would you like to change",
    "what would make you feel better",
    "what are you hoping for",
    "what support do you need",
    "what is your ideal outcome",
)


def extract_state(text: str) -> Tuple[Optional[str], str]:
    match = re.search(r"<state>\s*(.*?)\s*</state>", text, flags=re.IGNORECASE | re.DOTALL)
    state = match.group(1).strip() if match else None
    cleaned = re.sub(r"<state>\s*.*?\s*</state>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return state, cleaned.strip()


def remove_think_artifacts(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # An unclosed <think> leaves a dangling "</think>" split point; keep the
    # distinct segments so a truncated reasoning block doesn't eat the reply.
    parts = [p.strip() for p in re.split(r"</think>", text, flags=re.IGNORECASE) if p.strip()]
    if not parts:
        return ""

    deduped: List[str] = []
    seen = set()
    for part in parts:
        key = re.sub(r"\s+", " ", part.strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(part)

    if len(deduped) == 1:
        return deduped[0]
    return "\n\n".join(deduped)


def remove_role_leakage(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r"^(system|user|assistant|therapist|patient)\s*\n+", "", text, flags=re.IGNORECASE
    ).strip()
    text = re.sub(
        r"^(system|user|assistant|therapist|patient)\s*:\s*", "", text, flags=re.IGNORECASE
    ).strip()
    return text


def normalize_format(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def remove_duplicate_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return text.strip()

    kept: List[str] = []
    seen = set()
    for paragraph in paragraphs:
        key = re.sub(r"\s+", " ", paragraph.lower())
        if key not in seen:
            seen.add(key)
            kept.append(paragraph)
    return "\n\n".join(kept).strip()


def trim_truncated_ending(text: str) -> str:
    """Cut back to the last sentence-ending punctuation when the tail is a
    mid-sentence fragment (the token budget ran out)."""
    text = text.strip()
    if not text:
        return text
    if re.search(r'[.!?]["\')\]]?$', text):
        return text
    matches = list(re.finditer(r'[.!?]["\')\]]?', text))
    if matches:
        return text[: matches[-1].end()].strip()
    return text


def looks_like_user_prompt_echo(text: str) -> bool:
    """Detect the failure where the model parrots the therapist's question bank
    back instead of answering as the patient."""
    lowered = text.lower().strip()
    if sum(1 for q in _QUESTION_STARTERS if q in lowered) >= 3:
        return True
    has_first_person = any(x in lowered for x in (" i ", " i'm ", " i’m ", " me ", " my "))
    return lowered.count("?") >= 5 and not has_first_person


def postprocess_model_output(raw_text: str) -> Dict[str, Optional[str]]:
    state, text = extract_state(raw_text or "")
    text = remove_think_artifacts(text)
    text = remove_role_leakage(text)
    text = normalize_format(text)
    text = remove_duplicate_paragraphs(text)
    text = trim_truncated_ending(text)
    return {"state": state, "clean_text": text}


def is_refusal(text: str) -> bool:
    """True when the reply is empty or an 'I don't know'-style non-answer."""
    if not text or not text.strip():
        return True
    return bool(_REFUSAL_RE.search(text))


def _norm_words(text: str) -> List[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def too_similar(text: str, recent: List[str], threshold: float = 0.6) -> bool:
    """True when `text` near-duplicates a recent patient reply (word-set
    Jaccard) — catches the despair-repetition loop."""
    words = set(_norm_words(text))
    if len(words) < 4:
        return False
    for previous in recent or []:
        previous_words = set(_norm_words(previous))
        if not previous_words:
            continue
        union = len(words | previous_words)
        if union and len(words & previous_words) / union >= threshold:
            return True
    return False


def cap_sentences(text: str, max_sentences: int = 4) -> str:
    """Backstop against rambling: keep at most `max_sentences` sentences."""
    text = text.strip()
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= max_sentences:
        return text
    return " ".join(sentences[:max_sentences]).strip()


def clean_reply(raw_text: str, max_sentences: int = 4) -> str:
    """Full cleanup chain: raw model text -> reply shown to the user."""
    clean_text = postprocess_model_output(raw_text)["clean_text"] or ""
    if looks_like_user_prompt_echo(clean_text):
        return FALLBACK_REPLY
    return cap_sentences(clean_text, max_sentences=max_sentences)
