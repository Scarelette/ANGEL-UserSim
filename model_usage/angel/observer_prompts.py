"""Stage-1 (Observer) prompts and JSON extraction.

Vendored from the Observer training code (model_training/observer) so this package
does not importlib-load a script out of a sibling repository. The prompt text is
reproduced verbatim — changing it changes what the Observer checkpoint was tuned
to produce.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

SHORT_TO_JSON_SYSTEM_PROMPT = """You are an expert clinical case-profile writer for mental-health patient simulation research.

Your task is to expand a short patient description into a rich, realistic, internally consistent simulated patient profile. The profile will be used to train and evaluate AI systems in clinical-style conversations.

Important requirements:
1. Preserve all explicit facts from the short description.
2. Do not contradict the short description.
3. You may infer reasonable missing details, but they must be clinically plausible and clearly consistent with the original description.
4. Do not overfit to a generic template. Create diverse patients with varied backgrounds, personalities, symptoms, coping styles, relationship patterns, emotional expression, and communication styles.
5. Avoid stereotypes. Demographic details should influence the profile only when clinically or contextually relevant.
6. The profile should be complete enough for multi-turn roleplay, including what the patient readily discloses, what they hide, how they respond emotionally, and how symptoms appear in conversation.
7. Do not write treatment recommendations. Focus on the patient profile, presentation, behavior, and simulation rules.
8. Do not diagnose beyond the information provided. If a diagnosis is included in the short description, treat it as a diagnosis hint, not as a final clinical conclusion.
9. The output must be valid JSON only. Do not include markdown, comments, or extra explanation.
"""

_DIVERSITY_BLOCK = """
Diversity constraints:
- Vary the patient's openness level: some patients are guarded, some are talkative, some are vague, some intellectualize, and some minimize symptoms.
- Vary symptom presentation: symptoms should not always be severe, dramatic, or neatly organized.
- Vary emotional expression: include patients who are flat, irritable, ashamed, humorous, anxious, confused, defensive, or overly agreeable when appropriate.
- Vary social context: do not always give the patient supportive family or clear insight.
- Vary roleplay difficulty: some patients should resist direct questions, change topics, contradict themselves, or only reveal key information gradually.
- Avoid using the same phrases across profiles unless clinically appropriate.
"""

_OUTPUT_SCHEMA = """{
  "identity": {
    "name": "",
    "age": null,
    "gender": "",
    "role": "patient",
    "source_title": "",
    "diagnosis_hint": []
  },
  "background": [
    ""
  ],
  "presenting_problems": [
    ""
  ],
  "symptom_details": {
    "onset_and_course": "",
    "frequency_and_intensity": "",
    "triggers": [],
    "maintaining_factors": [],
    "functional_impairment": []
  },
  "emotional_profile": {
    "dominant_emotions": [],
    "emotional_expression_style": "",
    "shame_or_fear_points": [],
    "typical_reaction_when_distressed": ""
  },
  "cognitive_patterns": {
    "core_beliefs": [],
    "automatic_thoughts": [],
    "worries_or_ruminations": [],
    "interpretation_biases": []
  },
  "behavior_patterns": {
    "avoidance_behaviors": [],
    "safety_behaviors": [],
    "coping_strategies": [],
    "interpersonal_patterns": []
  },
  "social_and_family_context": {
    "family_relationships": "",
    "romantic_or_peer_relationships": "",
    "school_or_work_context": "",
    "social_support": "",
    "cultural_or_contextual_factors": ""
  },
  "risk_and_protective_factors": {
    "risk_factors": [],
    "protective_factors": [],
    "safety_notes": ""
  },
  "hidden_state": {
    "information_patient_initially_withholds": [],
    "information_revealed_after_trust": [],
    "topics_patient_avoids": [],
    "contradictions_or_ambivalence": []
  },
  "speaking_style": {
    "verbosity": "",
    "tone": "",
    "word_choice": "",
    "interaction_style": "",
    "example_phrases": []
  },
  "disclosure_rules": {
    "early_session": "",
    "middle_session": "",
    "late_session": "",
    "when_challenged": "",
    "when_supported": ""
  },
  "simulation_rules": [
    "",
    "",
    ""
  ],
  "brief_summary": ""
}"""


def build_long_profile_user_prompt(short_patient_profile: str) -> str:
    return f"""
Please expand the following short patient description into a complete long simulated patient profile.

Short patient description:
{short_patient_profile}

Generate a detailed patient profile in valid JSON format using the schema below.

Requirements:
- The profile should be clinically realistic and suitable for a mental-health patient simulation.
- Preserve the core facts from the short description.
- Add plausible missing details to make the patient usable in a long multi-turn dialogue.
- The patient should have a distinct personality, speech style, emotional pattern, and disclosure behavior.
- Include enough diversity across generated profiles. Avoid making every patient similarly introspective, cooperative, or articulate.
- The profile should support at least 15-20 turns of conversation.
- Include hidden information that the patient may reveal only after trust is built.
- Include simulation rules for how the patient should respond during roleplay.
- Keep the profile specific, concrete, and grounded in the given short description.
- Do not include therapist responses.
- Return valid JSON only.

{_DIVERSITY_BLOCK}

Output JSON schema:

{_OUTPUT_SCHEMA}
""".strip()


def build_long_profile_messages(short_patient_profile: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SHORT_TO_JSON_SYSTEM_PROMPT},
        {"role": "user", "content": build_long_profile_user_prompt(short_patient_profile)},
    ]


# --- output cleanup -------------------------------------------------------


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def strip_think_block(text: str) -> str:
    text = text or ""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def clean_model_output(text: str) -> str:
    return strip_code_fences(strip_think_block(text))


def extract_first_json_object(text: str) -> Optional[str]:
    """Return the first brace-balanced JSON object in `text`, or None.

    Brace counting is string-aware: a `{` or `}` inside a JSON string literal
    must not change the nesting depth, or profiles whose prose contains a brace
    truncate mid-object.
    """
    text = clean_model_output(text)
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_profile_json(text: str) -> Dict[str, Any]:
    json_str = extract_first_json_object(text)
    if not json_str:
        raise ValueError("Could not find a JSON object in model output.")
    return json.loads(json_str)
