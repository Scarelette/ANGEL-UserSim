"""Stage 1 of Angel (the Observer): short patient profile -> structured long profile JSON.

Ported from GRPO-Qwen3/GRPO/short2long_profile_generation.py, keeping only what
the profile-expansion experiment uses (local Observer generation + parsing).
The prompt strings are verbatim: they are what the Observer checkpoint was
trained against, so do not reword them.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


STAGE2_PROFILE_REWRITE_SYSTEM_PROMPT = """You are an expert mental-health case writer.

You will receive:
1. A stage-1 draft profile, which may be a JSONL-style record, valid structured JSON, incomplete JSON, malformed JSON, mixed JSON/text, or plain text notes.

Your task:
1. Convert the draft profile itself into one coherent natural-language long patient profile in prose.
2. Treat the draft profile as the primary source material.
3. Preserve as much useful information from the draft as possible, including identity, background, symptoms, emotional style, cognitive patterns, behavior patterns, interpersonal context, hidden material, speaking style, and disclosure behavior when available.
4. Describe how the patient presents in conversation, what they reveal early, and what they tend to hide.
5. Do not mention JSON, JSONL, missing fields, parser errors, truncation, or model limitations.
6. Do not add therapist advice, diagnosis confirmation, or treatment planning.
7. Return prose only.
"""


def build_long_profile_user_prompt(short_patient_profile: str) -> str:
    diversity_block = """
Diversity constraints:
- Vary the patient's openness level: some patients are guarded, some are talkative, some are vague, some intellectualize, and some minimize symptoms.
- Vary symptom presentation: symptoms should not always be severe, dramatic, or neatly organized.
- Vary emotional expression: include patients who are flat, irritable, ashamed, humorous, anxious, confused, defensive, or overly agreeable when appropriate.
- Vary social context: do not always give the patient supportive family or clear insight.
- Vary roleplay difficulty: some patients should resist direct questions, change topics, contradict themselves, or only reveal key information gradually.
- Avoid using the same phrases across profiles unless clinically appropriate.
"""

    prompt = f"""
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

{diversity_block}

Output JSON schema:

{{
  "identity": {{
    "name": "",
    "age": null,
    "gender": "",
    "role": "patient",
    "source_title": "",
    "diagnosis_hint": []
  }},
  "background": [
    ""
  ],
  "presenting_problems": [
    ""
  ],
  "symptom_details": {{
    "onset_and_course": "",
    "frequency_and_intensity": "",
    "triggers": [],
    "maintaining_factors": [],
    "functional_impairment": []
  }},
  "emotional_profile": {{
    "dominant_emotions": [],
    "emotional_expression_style": "",
    "shame_or_fear_points": [],
    "typical_reaction_when_distressed": ""
  }},
  "cognitive_patterns": {{
    "core_beliefs": [],
    "automatic_thoughts": [],
    "worries_or_ruminations": [],
    "interpretation_biases": []
  }},
  "behavior_patterns": {{
    "avoidance_behaviors": [],
    "safety_behaviors": [],
    "coping_strategies": [],
    "interpersonal_patterns": []
  }},
  "social_and_family_context": {{
    "family_relationships": "",
    "romantic_or_peer_relationships": "",
    "school_or_work_context": "",
    "social_support": "",
    "cultural_or_contextual_factors": ""
  }},
  "risk_and_protective_factors": {{
    "risk_factors": [],
    "protective_factors": [],
    "safety_notes": ""
  }},
  "hidden_state": {{
    "information_patient_initially_withholds": [],
    "information_revealed_after_trust": [],
    "topics_patient_avoids": [],
    "contradictions_or_ambivalence": []
  }},
  "speaking_style": {{
    "verbosity": "",
    "tone": "",
    "word_choice": "",
    "interaction_style": "",
    "example_phrases": []
  }},
  "disclosure_rules": {{
    "early_session": "",
    "middle_session": "",
    "late_session": "",
    "when_challenged": "",
    "when_supported": ""
  }},
  "simulation_rules": [
    "",
    "",
    ""
  ],
  "brief_summary": ""
}}
""".strip()

    return prompt


def build_long_profile_messages(short_patient_profile: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SHORT_TO_JSON_SYSTEM_PROMPT},
        {"role": "user", "content": build_long_profile_user_prompt(short_patient_profile)},
    ]


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
    """Return the first brace-balanced JSON object in ``text``, or None.

    Brace counting skips JSON string literals, so a ``{`` or ``}`` inside the
    profile prose does not end the object early.
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




class LocalProfileGenerator:
    def __init__(self, model_path: str, torch_dtype: str = "bfloat16"):
        dtype = getattr(torch, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                # temperature=temperature,
                top_p=top_p,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()



def generate_stage1_profile(
    generator: Any,
    short_profile: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    raw_output = generator.generate(
        build_long_profile_messages(short_profile),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    cleaned_output = clean_model_output(raw_output)
    parse_error = None
    profile_json = None

    try:
        profile_json = parse_profile_json(cleaned_output)
    except Exception as exc:
        parse_error = repr(exc)

    return {
        "raw_model_output": raw_output,
        "cleaned_model_output": cleaned_output,
        "profile_json": profile_json,
        "profile_json_parse_error": parse_error,
    }


