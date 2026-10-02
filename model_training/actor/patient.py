"""Local HF patient models used in the two rollout stages.

* ``PromptedPatient`` — SFT rollouts. A strong instruction model
  (``Qwen/Qwen3-30B-A3B-Instruct-2507`` in the paper) prompted with the
  symptom-network system prompt.
* ``AdapterPatient`` — DPO rollouts. Qwen3-8B (4-bit) + the SFT LoRA adapter,
  sampling several candidates per turn.

Both render the conversation with the chat template (``chat_format``), the
same format SFT, DPO and inference use.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model_training.actor.chat_format import render, to_chat_messages

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _nf4_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


class PromptedPatient:
    """Patient for SFT data generation (roles ``patient`` / ``therapist``)."""

    def __init__(self, model_name: str, system_prompt: str = ""):
        print("Loading patient model:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.patient_prompt = system_prompt
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
        )
        self.model.eval()

    def set_system_prompt(self, system_prompt: str) -> None:
        self.patient_prompt = system_prompt

    def _build_prompt(self, conversation) -> str:
        return render(self.tokenizer, to_chat_messages(self.patient_prompt, conversation), add_generation_prompt=True)

    def generate(self, conversation, max_new_tokens: int = 128, temperature: float = 0.8,
                 top_p: float = 0.9) -> str:
        prompt = self._build_prompt(conversation)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._clean_reply(self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip())

    @staticmethod
    def _clean_reply(text: str) -> str:
        # Cut role leakage. NOTE: lowercases the reply when leakage is found;
        # ~30% of the released SFT turns are lowercase because of this.
        for token in ["therapist:", "assistant:"]:
            if token in text.lower():
                text = text.lower().split(token)[0]
        return text.strip()


class AdapterPatient:
    """Qwen3-8B + SFT adapter; returns ``n`` sampled candidates per call."""

    def __init__(self, base_model_name: str, sft_adapter: Optional[str] = None,
                 system_prompt: str = "", load_in_4bit: bool = True):
        from peft import PeftModel

        print("Loading patient base model:", base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.patient_prompt = system_prompt

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=DTYPE,
            trust_remote_code=True,
            quantization_config=_nf4_config() if load_in_4bit else None,
        )
        if sft_adapter:
            print("Loading SFT adapter:", sft_adapter)
            self.model = PeftModel.from_pretrained(base_model, sft_adapter)
        else:
            self.model = base_model
        self.model.eval()

    def set_system_prompt(self, system_prompt: str) -> None:
        self.patient_prompt = system_prompt

    def _build_prompt(self, conversation) -> str:
        """Exactly the DPO training prompt: system prompt + full conversation."""
        return render(self.tokenizer, to_chat_messages(self.patient_prompt, conversation), add_generation_prompt=True)

    def generate(self, conversation, max_new_tokens: int = 256, temperature: float = 0.8,
                 top_p: float = 0.9, n: int = 1) -> Union[str, List[str]]:
        prompt = self._build_prompt(conversation)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=n,
            )
        replies = [
            self._clean_reply(self.tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True).strip())
            for i in range(outputs.shape[0])
        ]
        return replies if n > 1 else replies[0]

    @staticmethod
    def _clean_reply(text: str) -> str:
        if not text:
            return text
        lower = text.lower()
        cut = None
        for m in ["\ntherapist:", "\nassistant:", "\nuser:", "\npatient:"]:
            idx = lower.find(m)
            if idx != -1:
                cut = idx if cut is None else min(cut, idx)
        if cut is not None:
            text = text[:cut]
        return text.strip()
