"""Short patient profile -> long profile (stage 1) -> natural-language description (stage 2).

Stage 1 generator is selected by ``--stage1-model``:
  * a local path / Hub id (default: the Observer, ``resolve_model("observer")``)
  * ``azure`` or a ``gpt*`` Azure deployment name
  * a ``claude*`` model name (Anthropic API or Azure AI Foundry)
  * ``gemini`` / ``gemini-*`` (Vertex AI; needs GOOGLE_CLOUD_PROJECT)
Stage 2 always uses an Azure OpenAI deployment (``--stage2-deployment``,
default ANGEL_GPT5_DEPLOYMENT or ``gpt-5``).

The prompts are verbatim from the paper runs.
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import jsonlines
import torch
from openai import AzureOpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from angel_common.env import get_env, require_env
from angel_common.llm import anthropic_client, azure_openai_client
from angel_common.paths import resolve_model

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


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


def build_stage2_profile_description_prompt(
    target_words: int = 700,
    patient_profile_json: Optional[Dict[str, Any]] = None,
    raw_profile_text: Optional[str] = None,
) -> str:
    sections = [
        "Convert the following stage-1 patient profile draft into a complete long-form textual patient profile description.",
        "",
        f"Target length: around {target_words} words.",
        "",
        "Requirements:",
        "- Write fully in prose, not bullet points.",
        "- Treat the stage-1 draft as the primary source material to be converted.",
        "- The stage-1 draft may be a JSONL-style record, valid JSON, incomplete JSON, malformed JSON, mixed JSON/text, or free text.",
        "- Keep the result clinically realistic and internally consistent.",
        "- Include how the patient behaves in conversation, what they reveal early, what they hide, and what topics are difficult for them.",
        "- Preserve useful details about identity, background, symptoms, emotional style, cognitive patterns, behavior patterns, interpersonal context, hidden state, speaking style, and disclosure rules when available.",
        "- Do not mention missing JSON fields, JSONL formatting problems, parsing problems, truncation, or model errors.",
        "- Do not add therapist advice, diagnosis confirmation, treatment plan, or evaluation comments.",
        "- Return only the long profile description.",
    ]

    if patient_profile_json is not None:
        profile_text = json.dumps(patient_profile_json, ensure_ascii=False, indent=2)
        sections.extend(
            [
                "",
                "Structured stage-1 draft to convert:",
                profile_text,
            ]
        )

    if raw_profile_text:
        sections.extend(
            [
                "",
                "Raw stage-1 draft to convert:",
                raw_profile_text,
            ]
        )

    return "\n".join(sections).strip()


def build_long_profile_description_prompt(
    patient_profile_json: Dict[str, Any],
    target_words: int = 700,
) -> str:
    return build_stage2_profile_description_prompt(
        target_words=target_words,
        patient_profile_json=patient_profile_json,
    )


def build_raw_profile_description_prompt(
    raw_profile_text: str,
    target_words: int = 700,
) -> str:
    return build_stage2_profile_description_prompt(
        target_words=target_words,
        raw_profile_text=raw_profile_text,
    )


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
    text = clean_model_output(text)
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_profile_json(text: str) -> Dict[str, Any]:
    json_str = extract_first_json_object(text)
    if not json_str:
        raise ValueError("Could not find a JSON object in model output.")
    return json.loads(json_str)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with jsonlines.open(path, mode="r") as reader:
        for item in reader:
            yield item


def build_azure_client() -> AzureOpenAI:
    return azure_openai_client()


def get_azure_deployment_name(explicit_name: Optional[str] = None) -> str:
    deployment_name = (explicit_name or get_env("ANGEL_GPT5_DEPLOYMENT", None, "AZURE_OPENAI_DEPLOYMENT") or "").strip()
    if not deployment_name:
        raise ValueError("Missing Azure deployment. Set ANGEL_GPT5_DEPLOYMENT or pass an Azure deployment name.")
    return deployment_name


def is_azure_stage1_model(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    if normalized in {"azure", "azure-openai"}:
        return True
    if normalized.startswith("gpt"):
        return True
    azure_deployment = (get_env("ANGEL_GPT5_DEPLOYMENT", None, "AZURE_OPENAI_DEPLOYMENT") or "").strip().lower()
    return bool(azure_deployment) and normalized == azure_deployment


def is_gemini_stage1_model(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    return normalized == "gemini" or normalized.startswith("gemini-")


def extract_openai_message_text(response: Any) -> str:
    if response is None or not getattr(response, "choices", None):
        return ""
    message = getattr(response.choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    text_parts.append(part["text"])
            else:
                part_type = getattr(part, "type", None)
                part_text = getattr(part, "text", None)
                if part_type == "text" and part_text:
                    text_parts.append(part_text)
        return "\n".join(text_parts).strip()
    return str(content).strip()


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


class AzureProfileGenerator:
    def __init__(self, client: AzureOpenAI, deployment_name: str):
        self.client = client
        self.deployment_name = deployment_name

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        last_error = None
        max_completion_tokens = max(max_new_tokens, 8096)
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=1,
                )
                text = extract_openai_message_text(response)
                if text:
                    return text
                finish_reason = None
                if response is not None and getattr(response, "choices", None):
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                last_error = ValueError(f"Azure returned an empty stage-1 response. finish_reason={finish_reason!r}")
            except Exception as exc:
                last_error = exc
            time.sleep(1 + attempt)
        raise ValueError(f"Azure stage-1 generation failed for deployment '{self.deployment_name}': {last_error}")


class AnthropicProfileGenerator:
    def __init__(self, api_key: Optional[str], endpoint: Optional[str], deployment_name: str):
        self.client = anthropic_client(api_key=api_key, base_url=endpoint)
        self.deployment_name = deployment_name

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 8096,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        # Convert messages to Anthropic format
        system_message = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})
                # user_messages.append({"role": msg["role"], "content": 'Hi!'})

        response = self.client.messages.create(
            model=self.deployment_name,
            max_tokens=max_new_tokens,
            # temperature=temperature,
            # top_p=top_p,
            system=system_message,
            messages=user_messages,
        )
        return response.content[0].text.strip()


class GeminiProfileGenerator:
    def __init__(
        self,
        model_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        if genai is None or genai_types is None:
            raise ImportError("Google Gen AI library not installed. Install with: pip install google-genai")
        self.model_name = "gemini-2.5-flash" if model_name.strip().lower() == "gemini" else model_name
        self.client = genai.Client(
            vertexai=True,
            project=project or require_env("GOOGLE_CLOUD_PROJECT", purpose="Gemini on Vertex AI"),
            location=location or get_env("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        system_message = ""
        user_parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_message = content
            elif content:
                user_parts.append(content)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents="\n\n".join(user_parts),
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_new_tokens,
                system_instruction=system_message or None,
            ),
        )
        return (response.text or "").strip()


def get_short_profile_text(item: Dict[str, Any], input_field: str) -> str:
    value = item.get(input_field)
    if value is None:
        raise ValueError(f"Missing input field '{input_field}'.")
    if not isinstance(value, str):
        raise ValueError(f"Input field '{input_field}' must be a string.")
    value = value.strip()
    if not value:
        raise ValueError(f"Input field '{input_field}' is empty.")
    return value


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


def generate_long_profile_description(
    client: AzureOpenAI,
    deployment_name: str,
    patient_profile_json: Optional[Dict[str, Any]] = None,
    raw_profile_text: Optional[str] = None,
    target_words: int = 700,
    max_tokens: int = 1800,
    max_retries: int = 3,
) -> str:
    user_prompt = build_stage2_profile_description_prompt(
        patient_profile_json=patient_profile_json,
        raw_profile_text=raw_profile_text,
        target_words=target_words,
    )
    last_text = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": STAGE2_PROFILE_REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
                temperature=1,
            )
            last_text = extract_openai_message_text(response)
            if last_text:
                return last_text
        except Exception as exc:
            last_text = repr(exc)
        time.sleep(1 + attempt)
    raise ValueError("Azure returned an empty natural-language profile from the stage-1 draft.")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split()).strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _flatten_texts(value: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                cleaned = _normalize_text(item)
                if cleaned:
                    texts.append(cleaned)
    elif isinstance(value, str):
        cleaned = _normalize_text(value)
        if cleaned:
            texts.append(cleaned)
    return texts


def _join_items(items: List[str], conj: str = "and") -> str:
    filtered = [item for item in (_normalize_text(x) for x in items) if item]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} {conj} {filtered[1]}"
    return f"{', '.join(filtered[:-1])}, {conj} {filtered[-1]}"


def render_structured_profile_as_prose(
    short_profile: str,
    profile_json: Dict[str, Any],
) -> str:
    identity = profile_json.get("identity") or {}
    background = _flatten_texts(profile_json.get("background"))
    presenting = _flatten_texts(profile_json.get("presenting_problems"))
    symptom_details = profile_json.get("symptom_details") or {}
    emotional = profile_json.get("emotional_profile") or {}
    cognitive = profile_json.get("cognitive_patterns") or {}
    behavior = profile_json.get("behavior_patterns") or {}
    social = profile_json.get("social_and_family_context") or {}
    hidden = profile_json.get("hidden_state") or {}
    speaking = profile_json.get("speaking_style") or {}
    disclosure = profile_json.get("disclosure_rules") or {}
    summary = _normalize_text(profile_json.get("brief_summary"))

    name = _normalize_text(identity.get("name")) or "The patient"
    age = identity.get("age")
    gender = _normalize_text(identity.get("gender"))
    diagnosis_hints = _flatten_texts(identity.get("diagnosis_hint"))

    intro_parts = []
    if name != "The patient":
        intro_parts.append(name)
    else:
        intro_parts.append("The patient")
    if age not in (None, ""):
        intro_parts.append(f"is {age} years old")
    if gender:
        intro_parts.append(gender)

    intro = " ".join(intro_parts).strip()
    if not intro.endswith("."):
        intro += "."

    first_para_parts = [intro]
    if summary:
        first_para_parts.append(summary)
    else:
        first_para_parts.append(short_profile)
    if background:
        first_para_parts.append("Their background includes " + _join_items(background, conj="and") + ".")
    if diagnosis_hints:
        first_para_parts.append(
            "Key clinical concerns suggested by the case include "
            + _join_items(diagnosis_hints, conj="and")
            + "."
        )

    second_parts = []
    if presenting:
        second_parts.append("Current difficulties include " + _join_items(presenting, conj="and") + ".")
    onset = _normalize_text(symptom_details.get("onset_and_course"))
    freq = _normalize_text(symptom_details.get("frequency_and_intensity"))
    triggers = _flatten_texts(symptom_details.get("triggers"))
    maintaining = _flatten_texts(symptom_details.get("maintaining_factors"))
    impairment = _flatten_texts(symptom_details.get("functional_impairment"))
    if onset:
        second_parts.append(onset)
    if freq:
        second_parts.append(freq)
    if triggers:
        second_parts.append("Symptoms are often triggered by " + _join_items(triggers, conj="and") + ".")
    if maintaining:
        second_parts.append(
            "The pattern is maintained by " + _join_items(maintaining, conj="and") + "."
        )
    if impairment:
        second_parts.append(
            "This affects daily functioning through " + _join_items(impairment, conj="and") + "."
        )

    third_parts = []
    emotions = _flatten_texts(emotional.get("dominant_emotions"))
    expression = _normalize_text(emotional.get("emotional_expression_style"))
    shame = _flatten_texts(emotional.get("shame_or_fear_points"))
    distress = _normalize_text(emotional.get("typical_reaction_when_distressed"))
    beliefs = _flatten_texts(cognitive.get("core_beliefs"))
    worries = _flatten_texts(cognitive.get("worries_or_ruminations"))
    avoidance = _flatten_texts(behavior.get("avoidance_behaviors"))
    coping = _flatten_texts(behavior.get("coping_strategies"))
    interpersonal = _flatten_texts(behavior.get("interpersonal_patterns"))
    if emotions:
        third_parts.append("Emotionally, the patient is often marked by " + _join_items(emotions, conj="and") + ".")
    if expression:
        third_parts.append(expression + ".")
    if shame:
        third_parts.append("Sensitive areas include " + _join_items(shame, conj="and") + ".")
    if distress:
        third_parts.append("When distressed, the patient typically " + distress.rstrip(".") + ".")
    if beliefs:
        third_parts.append("Underlying beliefs include " + _join_items(beliefs, conj="and") + ".")
    if worries:
        third_parts.append("They also tend to ruminate about " + _join_items(worries, conj="and") + ".")
    if avoidance:
        third_parts.append("Common avoidance patterns include " + _join_items(avoidance, conj="and") + ".")
    if coping:
        third_parts.append("They usually cope by " + _join_items(coping, conj="and") + ".")
    if interpersonal:
        third_parts.append("Interpersonally, they tend to " + _join_items(interpersonal, conj="and") + ".")

    fourth_parts = []
    family = _normalize_text(social.get("family_relationships"))
    peers = _normalize_text(social.get("romantic_or_peer_relationships"))
    work = _normalize_text(social.get("school_or_work_context"))
    support = _normalize_text(social.get("social_support"))
    contextual = _normalize_text(social.get("cultural_or_contextual_factors"))
    withheld = _flatten_texts(hidden.get("information_patient_initially_withholds"))
    revealed = _flatten_texts(hidden.get("information_revealed_after_trust"))
    avoids = _flatten_texts(hidden.get("topics_patient_avoids"))
    contradictions = _flatten_texts(hidden.get("contradictions_or_ambivalence"))
    verbosity = _normalize_text(speaking.get("verbosity"))
    tone = _normalize_text(speaking.get("tone"))
    word_choice = _normalize_text(speaking.get("word_choice"))
    interaction = _normalize_text(speaking.get("interaction_style"))
    early = _normalize_text(disclosure.get("early_session"))
    middle = _normalize_text(disclosure.get("middle_session"))
    late = _normalize_text(disclosure.get("late_session"))
    challenged = _normalize_text(disclosure.get("when_challenged"))
    supported = _normalize_text(disclosure.get("when_supported"))
    if family:
        fourth_parts.append(family + ".")
    if peers:
        fourth_parts.append(peers + ".")
    if work:
        fourth_parts.append(work + ".")
    if support:
        fourth_parts.append("Their available support is " + support.rstrip(".") + ".")
    if contextual:
        fourth_parts.append(contextual + ".")
    if verbosity or tone or word_choice or interaction:
        style_bits = [bit for bit in [verbosity, tone, word_choice, interaction] if bit]
        fourth_parts.append("In conversation, the patient tends to be " + _join_items(style_bits, conj="and") + ".")
    if early:
        fourth_parts.append("Early in conversation, " + early.rstrip(".") + ".")
    if middle:
        fourth_parts.append("With some rapport, " + middle.rstrip(".") + ".")
    if late:
        fourth_parts.append("After trust develops, " + late.rstrip(".") + ".")
    if challenged:
        fourth_parts.append("If pushed too directly, " + challenged.rstrip(".") + ".")
    if supported:
        fourth_parts.append("When they feel supported, " + supported.rstrip(".") + ".")
    if withheld:
        fourth_parts.append("At first, the patient is likely to withhold " + _join_items(withheld, conj="and") + ".")
    if revealed:
        fourth_parts.append("More vulnerable material that may emerge later includes " + _join_items(revealed, conj="and") + ".")
    if avoids:
        fourth_parts.append("Topics they tend to avoid include " + _join_items(avoids, conj="and") + ".")
    if contradictions:
        fourth_parts.append("The presentation may include ambivalence or contradictions around " + _join_items(contradictions, conj="and") + ".")

    paragraphs = [
        " ".join(first_para_parts).strip(),
        " ".join(second_parts).strip(),
        " ".join(third_parts).strip(),
        " ".join(fourth_parts).strip(),
    ]
    return "\n\n".join([p for p in paragraphs if p])


def fallback_long_description(
    short_profile: str,
    profile_json: Optional[Dict[str, Any]],
    cleaned_model_output: str,
) -> str:
    if isinstance(profile_json, dict):
        return render_structured_profile_as_prose(short_profile, profile_json)
    extracted_json = extract_first_json_object(cleaned_model_output)
    if extracted_json:
        try:
            return render_structured_profile_as_prose(short_profile, json.loads(extracted_json))
        except Exception:
            pass
    if cleaned_model_output:
        cleaned_text = re.sub(r"\s+", " ", cleaned_model_output).strip()
        cleaned_text = re.sub(r"^[A-Z][A-Za-z\s]+:\s*", "", cleaned_text)
        if "{" not in cleaned_text and len(cleaned_text.split()) >= 40:
            return cleaned_text
    return (
        f"{short_profile} In conversation, the patient can describe the main symptoms, but they are likely "
        "to be somewhat selective, guarded, or uneven in how much detail they share at first. As rapport "
        "builds, more context about stressors, emotional reactions, and coping patterns may emerge, along "
        "with details the patient initially minimizes or avoids."
    )


def convert_short_profiles_to_text(
    input_path: str,
    output_path: str,
    stage2_deployment: str,
    input_field: str = "complaints",
    limit: Optional[int] = None,
    stage1_model: Optional[str] = None,
    model_api: Optional[str] = None,
    model_endpoint: Optional[str] = None,
) -> None:
    # Static generation params
    max_new_tokens = 3072
    temperature = 0.1
    top_p = 0.9
    target_words = 700

    client = build_azure_client()
    stage1_model = stage1_model or resolve_model("observer")

    # Stage 1 can use Azure GPT, Claude, Gemini, or a local model path.
    if is_azure_stage1_model(stage1_model):
        stage1_deployment = (
            get_azure_deployment_name()
            if (stage1_model or "").strip().lower() in {"azure", "azure-openai"}
            else get_azure_deployment_name(stage1_model)
        )
        generator = AzureProfileGenerator(client=client, deployment_name=stage1_deployment)
    elif stage1_model.startswith("claude"):
        generator = AnthropicProfileGenerator(api_key=model_api, endpoint=model_endpoint, deployment_name=stage1_model)
    elif is_gemini_stage1_model(stage1_model):
        generator = GeminiProfileGenerator(model_name=stage1_model)
    else:
        # Otherwise treat the value as a local model path.
        generator = LocalProfileGenerator(model_path=stage1_model)

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed = 0
    seen = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for idx, item in enumerate(tqdm(iter_jsonl(input_path), desc="Short -> Text")):
            if limit is not None and seen >= limit:
                break
            seen += 1

            try:
                short_profile = get_short_profile_text(item, input_field=input_field)
                stage1 = generate_stage1_profile(
                    generator=generator,
                    short_profile=short_profile,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # print('s1: ', stage1)
                profile_json = stage1["profile_json"]
                stage2_error = None

                if isinstance(profile_json, dict):
                    try:
                        long_description = generate_long_profile_description(
                            client=client,
                            deployment_name=stage2_deployment,
                            patient_profile_json=profile_json,
                            raw_profile_text=stage1["cleaned_model_output"],
                            target_words=target_words,
                        )
                    except Exception as exc:
                        stage2_error = repr(exc)
                        print("stage2_error:", stage2_error)
                        long_description = fallback_long_description(
                            short_profile=short_profile,
                            profile_json=profile_json,
                            cleaned_model_output=stage1["cleaned_model_output"],
                        )
                else:
                    try:
                        long_description = generate_long_profile_description(
                            client=client,
                            deployment_name=stage2_deployment,
                            raw_profile_text=stage1["cleaned_model_output"],
                            target_words=target_words,
                        )
                    except Exception as exc:
                        stage2_error = repr(exc)
                        print("stage2_error:", stage2_error)
                        long_description = fallback_long_description(
                            short_profile=short_profile,
                            profile_json=profile_json,
                            cleaned_model_output=stage1["cleaned_model_output"],
                        )

                # print('s2: ', long_description)
                writer.write(
                    {
                        "id": item.get("id", idx),
                        "source_title": item.get("source_title") or item.get("title"),
                        "short_profile": short_profile,
                        "long_profile_description": long_description,
                        "stage2_error": stage2_error,
                        "original_item": item,
                    }
                )
                processed += 1
            except Exception as exc:
                writer.write(
                    {
                        "id": item.get("id", idx),
                        "source_title": item.get("source_title") or item.get("title"),
                        "short_profile": item.get(input_field),
                        "error": repr(exc),
                        "original_item": item,
                    }
                )

    print(f"Processed {seen} input rows and saved {processed} natural-language profiles to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate long patient profiles. Stage 1 can use Azure GPT, Claude, Gemini, or a local model. Stage 2 always uses Azure GPT.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output-text", required=True, help="Output JSONL path for natural-language profiles.")
    parser.add_argument("--input-field", default="short_patient_profile", help="Field containing the short patient profile.")
    parser.add_argument(
        "--stage1-model",
        "--model-name",
        dest="stage1_model",
        default=None,
        help="Stage 1 model (default: the Observer, ANGEL_OBSERVER_MODEL): use 'azure' or an Azure deployment name, a Claude model name, 'gemini'/'gemini-2.5-flash', or a local model path.",
    )
    parser.add_argument(
        "--stage2-deployment",
        "--deployment-name",
        dest="stage2_deployment",
        default=None,
        help="Stage 2 Azure deployment name. Defaults to ANGEL_GPT5_DEPLOYMENT (gpt-5).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    convert_short_profiles_to_text(
        input_path=args.input,
        output_path=args.output_text,
        stage2_deployment=args.stage2_deployment or get_env("ANGEL_GPT5_DEPLOYMENT", "gpt-5", "AZURE_OPENAI_DEPLOYMENT"),
        input_field=args.input_field,
        limit=args.limit,
        stage1_model=args.stage1_model,
    )


if __name__ == "__main__":
    main()
