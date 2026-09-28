import asyncio
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from experiments.profile_expansion.patients.ai_patient import AIPatient


class Eeyore(AIPatient):
    def __init__(
        self,
        profile: str,
        model_name: str = "liusiyang/eeyore_sft_epoch2_dpo_round2_epoch1_llama3.1_8B",
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        if not isinstance(profile, str):
            raise TypeError(f"profile must be a string, got {type(profile)}")

        # self.system_prompt = profile.strip()
        system_prompt = f'''Imagine you are a patient who has been experiencing mental health challenges.

Below is your clinical background and case profile:
{profile}'''
        super().__init__(system_prompt=system_prompt)
        self.model_name = model_name

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Some models do not define pad token; align it with eos token for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

    # -------------------------------------------------
    # Build conversation messages
    # -------------------------------------------------
    # def _build_messages(self, conversation: List[Dict]) -> List[Dict]:
    #     messages = [{"role": "system", "content": self.system_prompt}]

    #     for turn in conversation:
    #         role = turn.get("role")
    #         content = turn.get("content", "")

    #         if not content:
    #             continue

    #         if role == "therapist":
    #             messages.append({"role": "user", "content": content})
    #         elif role == "patient":
    #             messages.append({"role": "assistant", "content": content})
    #         else:
    #             raise ValueError(f"Unknown role: {role}")

    #     return messages

    # -------------------------------------------------
    # Internal sync generation
    # -------------------------------------------------
    def _generate_sync(
        self,
        conversation: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        messages = self._build_messages(conversation)

        # input_ids = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=True,
        #     add_generation_prompt=True,
        #     return_tensors="pt",
        # ).to(self.model.device)

        # attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    # -------------------------------------------------
    # Async wrapper
    # -------------------------------------------------
    async def generate(
        self,
        conversation: List[Dict],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync,
            conversation,
            max_new_tokens,
            temperature,
            top_p,
            do_sample,
        )