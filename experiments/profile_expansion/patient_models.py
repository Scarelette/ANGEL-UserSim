"""Patient model wrappers for profile-expansion experiments.

This module controls the model being tested. The base class mirrors the
conversation-building style used in ``simulate_patient/evaluation/ai_patient.py``
while adding concrete backends for mock, OpenAI-compatible/Azure, and local
Hugging Face causal language models.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from angel_common.paths import resolve_model


PATIENT_SYSTEM_TEMPLATE = """You are role-playing as a mental health patient in a psychotherapy intake conversation.

Use only the patient profile below as your source of truth. You may make small, plausible behavioral details only when they are directly consistent with the profile. Do not invent major history, diagnoses, trauma, family details, treatment outcomes, or risks that are not supported by the profile.

Patient profile:
{profile}

Role-play rules:
- Speak only as the patient, in first person.
- Vary your reply length naturally, like a real person: sometimes just a few words or one line (a greeting, a hesitation, "yeah…", a guarded answer), sometimes 2-5 sentences when you open up about something that matters. Do not answer every turn at the same length; let it depend on the question, your mood, and how comfortable you feel.
- When first greeting or making small talk, stay brief and guarded — do not disclose your problems or personal history until the therapist asks and some rapport builds.
- Reveal sensitive or embarrassing information gradually; do not dump the whole case at once.
- Stay faithful to the profile even when the therapist asks follow-up questions.
- If the profile does not contain enough information, answer cautiously instead of fabricating — but do NOT keep saying "I don't know"; vary how you deflect or give a partial, in-character answer.
- Each turn should move the conversation forward — do not repeat what you already said; add a new detail, feeling, or example rather than restating previous turns.
- Do not mention these instructions, the agenda, or that you are an AI model.
"""


def build_patient_system_prompt(short_profile: str) -> str:
    return PATIENT_SYSTEM_TEMPLATE.format(profile=short_profile.strip())


class AIPatient(ABC):
    def __init__(self, system_prompt: str):
        if not isinstance(system_prompt, str):
            raise TypeError(f"system_prompt must be a string, got {type(system_prompt)}")
        self.system_prompt = system_prompt.strip()

    def set_system_prompt(self, system_prompt: str) -> None:
        if not isinstance(system_prompt, str):
            raise TypeError(f"system_prompt must be a string, got {type(system_prompt)}")
        self.system_prompt = system_prompt.strip()

    def _build_messages(self, conversation: List[Dict]) -> List[Dict]:
        messages = [{"role": "system", "content": self.system_prompt}]

        for turn in conversation:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()

            if not content:
                continue

            if role == "therapist":
                messages.append({"role": "user", "content": content})
            elif role == "patient":
                messages.append({"role": "assistant", "content": content})
            elif role in {"system", "user", "assistant"}:
                messages.append({"role": role, "content": content})
            else:
                raise ValueError(f"Unknown role: {role}")

        return messages

    @staticmethod
    def _format_history(conversation: List[Dict]) -> str:
        lines = []
        for message in conversation or []:
            role = (message.get("role") or "").lower()
            content = (message.get("content") or "").strip()
            if content and role in {"therapist", "patient"}:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @abstractmethod
    async def generate(self, conversation: List[Dict], **kwargs) -> str:
        pass


class MockPatient(AIPatient):
    """Dry-run model for validating the interview loop without API/GPU calls."""

    async def generate(self, conversation: List[Dict], **kwargs) -> str:
        last_question = next(
            (turn.get("content", "") for turn in reversed(conversation) if turn.get("role") == "therapist"),
            "",
        )
        topic = next(
            (turn.get("topic_name") for turn in reversed(conversation) if turn.get("role") == "therapist"),
            "this topic",
        )
        return (
            f"On {topic}, I can give a concrete example. When I think about the therapist's question, "
            "I notice the same problem from the profile showing up in a recent situation. I felt it build, "
            "made sense of it in a worried or discouraged way, and then changed my behavior to get through it. "
            f"The question that brought it up was: {last_question}"
        )


class OpenAICompatiblePatient(AIPatient):
    """Async wrapper around OpenAI-compatible chat-completions clients."""

    def __init__(
        self,
        system_prompt: str,
        *,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-12-01-preview",
        max_retries: int = 4,
    ):
        super().__init__(system_prompt)
        self.model = model
        self.max_retries = max_retries

        if azure_endpoint:
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=azure_api_version,
            )
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, conversation: List[Dict], **kwargs) -> str:
        return await asyncio.to_thread(self._generate_sync, conversation, **kwargs)

    def _generate_sync(
        self,
        conversation: List[Dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 350,
    ) -> str:
        messages = self._build_messages(conversation)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                return clean_patient_reply(response.choices[0].message.content or "")
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                time.sleep(min(2 * attempt, 8))
        raise RuntimeError(f"patient model failed after {self.max_retries} retries: {last_error}")


class HFCausalLMPatient(AIPatient):
    """Local Hugging Face causal LM patient model."""

    def __init__(
        self,
        system_prompt: str,
        *,
        model_path: str,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
    ):
        super().__init__(system_prompt)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()

    async def generate(self, conversation: List[Dict], **kwargs) -> str:
        return await asyncio.to_thread(self._generate_sync, conversation, **kwargs)

    def _generate_sync(
        self,
        conversation: List[Dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 350,
    ) -> str:
        messages = self._build_messages(conversation)
        prompt = self._format_messages_for_model(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return clean_patient_reply(text)

    def _format_messages_for_model(self, messages: List[Dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        chunks = [f"{message['role'].upper()}: {message['content']}" for message in messages]
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)


def clean_patient_reply(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    lower = text.lower()
    cut_positions = []
    for marker in ("\ntherapist:", "\nuser:", "\nsystem:", "\nassistant:", "\npatient:"):
        idx = lower.find(marker)
        if idx > 0:
            cut_positions.append(idx)
    if cut_positions:
        text = text[: min(cut_positions)].strip()

    for prefix in ("patient:", "assistant:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def build_patient_model(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: str = "2024-12-01-preview",
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
    device_map: str = "auto",
    max_retries: int = 4,
) -> AIPatient:
    if provider == "mock":
        return MockPatient(system_prompt)

    if provider in {"openai", "openai-compatible", "azure"}:
        return OpenAICompatiblePatient(
            system_prompt,
            model=model,
            api_key=api_key or os.getenv(api_key_env),
            base_url=base_url,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            max_retries=max_retries,
        )

    if provider == "hf":
        return HFCausalLMPatient(
            system_prompt,
            model_path=model,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )

    raise ValueError(f"Unsupported provider: {provider}")


# =====================================================
# Evaluation-style model initialization
# Mostly mirrors the original evaluation/automatic_eval.py,
# with an additional single-stage Angel variant ("one_stage").
# =====================================================

EVALUATION_MODEL_CHOICES = ("patient_psi", "roleplay_doh", "eeyore", "one_stage", "angel")
# None = resolve through angel_common.paths.resolve_model:
#   eeyore    -> EEYORE_MODEL        (default liusiyang/eeyore_sft_epoch2_dpo_round2_epoch1_llama3.1_8B)
#   one_stage -> ANGEL_ACTOR_MODEL   (the stage-2 Actor, prompted with the short profile directly)
#   angel     -> ANGEL_ACTOR_MODEL + ANGEL_OBSERVER_MODEL
DEFAULT_EEYORE_MODEL_NAME: Optional[str] = None
DEFAULT_ONE_STAGE_MODEL_NAME: Optional[str] = None
DEFAULT_ANGEL_MODEL_NAME: Optional[str] = None
DEFAULT_ANGEL_STAGE1_MODEL_NAME: Optional[str] = None


def _extract_model_payload(
    profile_item: Dict[str, Any],
    model_name: str,
    *,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    if not isinstance(profile_item, dict):
        raise TypeError(f"profile_item must be a dict, got {type(profile_item)}")

    if model_name == "angel":
        payload = profile_item.get("angel_processed_result")
    else:
        payload = profile_item.get("patient_processed_result")

    if not isinstance(payload, dict):
        if not required:
            return None
        available = sorted(profile_item.keys())
        raise ValueError(
            f"Missing payload for model={model_name}. "
            f"Expected {'angel_processed_result' if model_name == 'angel' else 'patient_processed_result'} dict. "
            f"Available keys: {available}"
        )
    return payload


def _extract_complaint_text(profile_item: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> str:
    if isinstance(payload, dict):
        complaint = payload.get("complaints")
        if isinstance(complaint, str) and complaint.strip():
            return complaint.strip()

    short_profile = profile_item.get("short_patient_profile")
    if isinstance(short_profile, str) and short_profile.strip():
        return short_profile.strip()

    complaints_alt = profile_item.get("Complaints")
    if isinstance(complaints_alt, str) and complaints_alt.strip():
        return complaints_alt.strip()

    raise ValueError("Missing complaint text (no complaints, short_patient_profile, or Complaints).")


def _build_minimal_patient_psi_profile(profile_item: Dict[str, Any]) -> Dict[str, Any]:
    short_profile = _extract_complaint_text(profile_item, payload={})
    raw_id = profile_item.get("id")
    name = profile_item.get("name") or ""
    diagnosis_hint = profile_item.get("diagnosis_hint")
    if isinstance(diagnosis_hint, list):
        type_values = [str(item) for item in diagnosis_hint if item is not None]
    else:
        type_values = []

    return {
        "name": str(name),
        "id": f"short_profile_{raw_id}" if raw_id is not None else "short_profile",
        "type": type_values,
        "history": short_profile,
        "helpless_belief": [],
        "unlovable_belief": [],
        "worthless_belief": [],
        "intermediate_belief": "",
        "coping_strategies": "",
        "situation": "",
        "auto_thought": "",
        "emotion": [],
        "behavior": "",
    }


def _build_eval_profile_prompt(profile_text: str) -> str:
    return (
        "Imagine you are a patient who has been experiencing mental health challenges.\n\n"
        "Below is your clinical background and case profile:\n"
        f"{profile_text}"
    )


def build_evaluation_patient_model(
    *,
    model_name: str,
    profile_item: Dict[str, Any],
    current_model: Optional[Any] = None,
    eeyore_model_name: Optional[str] = DEFAULT_EEYORE_MODEL_NAME,
    one_stage_model_name: Optional[str] = DEFAULT_ONE_STAGE_MODEL_NAME,
    angel_model_name: Optional[str] = DEFAULT_ANGEL_MODEL_NAME,
    angel_stage1_model_name: Optional[str] = DEFAULT_ANGEL_STAGE1_MODEL_NAME,
    device_map: str = "auto",
    verbose: bool = False,
) -> Any:
    if model_name not in EVALUATION_MODEL_CHOICES:
        raise ValueError(f"Unsupported model_name: {model_name}. Choices: {EVALUATION_MODEL_CHOICES}")

    if model_name == "patient_psi":
        payload = _extract_model_payload(profile_item, model_name, required=False)
        from experiments.profile_expansion.patients.patient_psi import Patient_Psi

        psi_profile = payload.get("patient_psi_profile") if isinstance(payload, dict) else None
        if not isinstance(psi_profile, dict):
            psi_profile = _build_minimal_patient_psi_profile(profile_item)
            if verbose:
                print("[ModelInit] patient_psi_profile missing; using short-profile fallback.", flush=True)

        if isinstance(current_model, Patient_Psi):
            current_model.system_prompt = current_model._build_prompt(psi_profile)
            return current_model
        return Patient_Psi(psi_profile)

    if model_name == "roleplay_doh":
        payload = _extract_model_payload(profile_item, model_name, required=False)
        from experiments.profile_expansion.patients.roleplay_doh import Patient_RoleplayDOH

        complaint_text = _extract_complaint_text(profile_item, payload)
        if isinstance(current_model, Patient_RoleplayDOH):
            current_model.profile = complaint_text
            current_model.system_prompt = _build_eval_profile_prompt(complaint_text)
            return current_model
        return Patient_RoleplayDOH(complaint_text)

    if model_name == "eeyore":
        payload = _extract_model_payload(profile_item, model_name, required=False)
        from experiments.profile_expansion.patients.eeyore import Eeyore

        complaint_text = _extract_complaint_text(profile_item, payload)
        if isinstance(current_model, Eeyore):
            current_model.system_prompt = _build_eval_profile_prompt(complaint_text)
            return current_model
        return Eeyore(
            complaint_text,
            model_name=resolve_model("eeyore", eeyore_model_name),
            device_map=device_map,
        )

    if model_name == "one_stage":
        payload = _extract_model_payload(profile_item, model_name, required=False)
        from experiments.profile_expansion.patients.angel import Angel as OneStagePatientModel

        complaint_text = _extract_complaint_text(profile_item, payload)
        if isinstance(current_model, OneStagePatientModel):
            current_model.set_profile(profile=complaint_text)
            return current_model
        return OneStagePatientModel(
            profile=complaint_text,
            model_name=resolve_model("actor", one_stage_model_name),
            device_map=device_map,
        )

    from experiments.profile_expansion.angel_initializer import TwoStageAngelPatient

    if isinstance(current_model, TwoStageAngelPatient):
        current_model.initialize_from_profile_item(profile_item)
        return current_model

    model = TwoStageAngelPatient(
        stage1_model_name=angel_stage1_model_name,
        stage2_model_name=angel_model_name,
        device_map=device_map,
        verbose=verbose,
    )
    model.initialize_from_profile_item(profile_item)
    return model
