"""Normalize raw patient generations to exactly one <state>/<word> pair (used during DPO rollouts)."""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

# --- regex ---
STATE_RE = re.compile(r"<state>(.*?)</state>", re.S | re.I)
WORD_RE  = re.compile(r"<word>(.*?)</word>",  re.S | re.I)
# Chinese / Japanese / Korean characters and CJK punctuation; replies must be English.
CJK_RE   = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")

# --- strip instruction-echo / wrapper artifacts (reduces jailbreak/content_filter triggers) ---
ECHO_PATTERNS = [
    r"END OF INSTRUCTIONS.*",
    r"REMINDER:.*",
    r"EXPRESSION EXAMPLE.*",
    r"for reference only.*",
    r"Answer this in the required format.*",
    r"Answer with the required format.*",
    r"No markdown\..*",
    r"No extra text\..*",
    r"Every response Must be wrapped.*",
    r"Remain fully consistent.*",
    r"therapist's question:.*",
]
ECHO_RE = re.compile("|".join(ECHO_PATTERNS), re.I | re.S)

# --- placeholder/template detector (your exact failure case) ---
PLACEHOLDER_SUBSTRS = [
    "your new state during the conversation",
    "are therapist's words trigger",
    "your first-person narrative responding to the therapist",
    "your first-person narrative responding to the",
]

def sanitize_echo(text: str) -> str:
    if not text:
        return ""
    text = text.replace("</response>", "").replace("<response>", "")
    text = re.sub(ECHO_RE, "", text).strip()
    return text.strip()

def _is_placeholder(state: str, word: str) -> bool:
    s = (state or "").strip().lower()
    w = (word or "").strip().lower()
    if not s or not w:
        return True
    return any(p in s for p in PLACEHOLDER_SUBSTRS) or any(p in w for p in PLACEHOLDER_SUBSTRS)

def _extract_pairs(text: str) -> List[Tuple[str, str]]:
    states = [s.strip() for s in STATE_RE.findall(text)]
    words  = [w.strip() for w in WORD_RE.findall(text)]
    n = min(len(states), len(words))
    return [(states[i], words[i]) for i in range(n)]

# Picks the FIRST non-placeholder pair; falls back to the LAST pair.
def coerce_last_pair(text: str) -> str:
    """
    Force EXACTLY ONE <state> and ONE <word>.
    Selection strategy:
      1) sanitize echoes
      2) choose FIRST pair that is NOT a placeholder/template
      3) fallback: last pair (if any)
    Returns "" if no usable tags exist.
    """
    text = sanitize_echo(text)
    pairs = _extract_pairs(text)
    if not pairs:
        return ""

    # first non-placeholder
    for s, w in pairs:
        if not _is_placeholder(s, w):
            return f"<state>{s}</state>\n<word>{w}</word>"

    # fallback: last pair
    s_last, w_last = pairs[-1]
    return f"<state>{s_last}</state>\n<word>{w_last}</word>"

def parse_single_pair(text: str) -> Tuple[bool, Optional[str], Optional[str], List[str]]:
    """
    Strict parse: must contain exactly one <state> and one <word>, both non-empty,
    must NOT be the placeholder/template pair, and must not contain CJK text.
    """
    issues: List[str] = []
    if not text:
        return False, None, None, ["empty text"]

    states = STATE_RE.findall(text)
    words  = WORD_RE.findall(text)

    if len(states) != 1:
        issues.append(f"expected 1 <state>, got {len(states)}")
    if len(words) != 1:
        issues.append(f"expected 1 <word>, got {len(words)}")
    if issues:
        return False, None, None, issues

    s = states[0].strip()
    w = words[0].strip()

    if not s:
        issues.append("empty state")
    if not w:
        issues.append("empty word")
    if _is_placeholder(s, w):
        issues.append("placeholder template pair")
    if CJK_RE.search(s) or CJK_RE.search(w):
        issues.append("non-English (CJK) text")

    if issues:
        return False, None, None, issues
    return True, s, w, []

def build_format_fix_prompt(draft: str) -> str:
    """
    Safety-friendly format-fix prompt (avoid jailbreak-y phrasing).
    Also explicitly forbids placeholder/template text and asks for English.
    """
    draft = sanitize_echo(draft)
    return (
        "Rewrite the following draft into EXACTLY:\n"
        "<state>...</state>\n"
        "<word>...</word>\n\n"
        "Rules:\n"
        "- Output only these two blocks.\n"
        "- Keep the meaning as close as possible.\n"
        "- First-person.\n"
        "- Write in English only, even if the draft contains other languages.\n"
        "- Do not include any extra headers, examples, reminders, or meta text.\n"
        "- Do NOT output placeholder/template text (e.g., 'Your new state during the conversation').\n"
        "- If multiple pairs appear, keep the first non-placeholder pair.\n\n"
        "Draft:\n"
        f"{draft}\n"
    )

def postprocess_patient_output(
    raw_text: str,
    patient_generate_fn: Callable[..., str],
    patient_view_messages: List[Dict[str, str]],
    gpt_format_fix_fn: Optional[Callable[[str], str]] = None,
    *,
    retry_patient_once: bool = True,
) -> Tuple[bool, str, Optional[str], Optional[str], List[str]]:
    """
    Output-only strategy:
      1) local sanitize + coerce_last_pair (now: first non-placeholder pair)
      2) strict parse
      3) if fail: resample patient once (lower temperature)
      4) if still fail: optional GPT format-fix then local coerce again
    """
    # A) local coerce
    coerced = coerce_last_pair(raw_text)
    candidate = coerced if coerced else sanitize_echo(raw_text)

    ok, s, w, issues = parse_single_pair(candidate)
    if ok:
        return True, candidate, s, w, []

    # B) resample patient once (cheap)
    if retry_patient_once:
        try:
            raw_retry = patient_generate_fn(patient_view_messages, temperature=0.2, max_new_tokens=512)
        except Exception:
            raw_retry = ""

        coerced2 = coerce_last_pair(raw_retry)
        if coerced2:
            ok2, s2, w2, issues2 = parse_single_pair(coerced2)
            if ok2:
                return True, coerced2, s2, w2, []

    # C) GPT format-fix if provided
    if gpt_format_fix_fn is not None:
        try:
            fixed = gpt_format_fix_fn(build_format_fix_prompt(raw_text))
        except Exception:
            fixed = ""

        fixed2 = coerce_last_pair(fixed) or coerce_last_pair(raw_text)
        ok3, s3, w3, issues3 = parse_single_pair(fixed2)
        if ok3:
            return True, fixed2, s3, w3, []

        return False, fixed2 or candidate, None, None, issues3

    return False, candidate, None, None, issues