import asyncio
import random
import re
from typing import List, Dict, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from angel_common.paths import resolve_model
from experiments.profile_expansion.patients.ai_patient import AIPatient
from experiments.profile_expansion.patients.sys_prompt import generate_system_prompt_profile
from experiments.profile_expansion.patients.patient_profile import (
    convert_rich_profile_to_internal,
    build_system_prompt,
)
from experiments.profile_expansion.patients.state_manager import PatientStateManager


class Angel(AIPatient):
    def __init__(
        self,
        profile: str = None,
        profile_dict: Dict[str, Any] = None,
        model_name: Optional[str] = None,  # None -> ANGEL_ACTOR_MODEL / models/qwen3-8b-dpo-merged
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.active_profile = None
        self.state_manager = None

        # Initial system prompt: a rich profile dict (two-stage Angel), else the
        # short profile text (one-stage ablation), else an empty profile.

        if profile_dict is not None:
            if not isinstance(profile_dict, dict):
                raise TypeError(
                    f"profile_dict must be a dict, got {type(profile_dict)}"
                )

            self.active_profile = convert_rich_profile_to_internal(
                raw_profile=profile_dict,
                line_num=None,
            )

            self.state_manager = PatientStateManager(
                profile=self.active_profile,
                max_turns=12,
            )

            system_prompt = build_system_prompt(
                profile=self.active_profile,
                dynamic_state=self.state_manager.get_dynamic_state(),
            )

        elif profile:
            system_prompt = generate_system_prompt_profile(profile)

        else:
            system_prompt = generate_system_prompt_profile("")

        super().__init__(system_prompt=system_prompt)

        self.model_name = model_name = resolve_model("actor", model_name)

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def set_profile_from_dict(
        self,
        profile_dict: Dict[str, Any],
        line_num: Optional[int] = None,
    ):
        """
        Update Angel with a rich profile dict without reloading the model.

        Expected input format example:

        {
            "identity": {
                "name": "Mr. Adler",
                "age": null,
                "gender": "male",
                "role": "patient",
                "source_title": "...",
                "diagnosis_hint": [...]
            },
            "background": [...],
            "presenting_problems": [...],
            "triggers": [...],
            "emotions": [...],
            "behaviors": [...],
            "family_context": [...],
            "hidden_state": [...],
            "speaking_style": {...},
            "disclosure_rules": {...},
            "simulation_rules": {...},
            "_meta": {...}
        }
        """

        if not isinstance(profile_dict, dict):
            raise TypeError(
                f"profile_dict must be a dict, got {type(profile_dict)}"
            )

        self.active_profile = convert_rich_profile_to_internal(
            raw_profile=profile_dict,
            line_num=line_num,
        )

        self.state_manager = PatientStateManager(
            profile=self.active_profile,
            max_turns=12,
        )

        self.system_prompt = build_system_prompt(
            profile=self.active_profile,
            dynamic_state=self.state_manager.get_dynamic_state(),
        )

        print(
            f"[Angel] Loaded dict profile: "
            f"name={self.active_profile.get('name')}, "
            f"profile_id={self.active_profile.get('profile_id')}"
        )

    def update_dynamic_state_from_user(self, user_message: str):
        if self.state_manager is None or self.active_profile is None:
            return

        self.state_manager.update_from_user_message(user_message)

        self.system_prompt = build_system_prompt(
            profile=self.active_profile,
            dynamic_state=self.state_manager.get_dynamic_state(),
        )

    # =====================================================
    # Generation
    # =====================================================

    def _get_latest_user_message(self, conversation: List[Dict]) -> Optional[str]:
        for msg in reversed(conversation or []):
            role = msg.get("role")
            content = msg.get("content", "")

            if role in {"therapist", "user"} and isinstance(content, str) and content.strip():
                return content.strip()

        return None

    # ---- dynamic reply length ------------------------------------------------
    # Cheap heuristics on the therapist's latest turn: open/invitational vs
    # closed/factual questions.
    # Invitational / elaborative cues + wh-words (\bwhat\b also matches "what's").
    _OPEN_RE = re.compile(
        r"\b(tell me|say more|more about|describe|walk me through|share|what|how|why)\b",
        re.IGNORECASE,
    )
    # Yes/no or factual questions (checked FIRST, so "how many/old" stays closed).
    _CLOSED_RE = re.compile(
        r"^\s*(do|did|does|have|has|had|are|is|was|were|will|would|can|could|should|may)\b"
        r"|\bhow (old|many|much|often|long)\b|\bwhen (did|was|do|does)\b|\bwhere\b|\bwho\b"
        r"|\bwhat (time|day|date|year)\b|\byes or no\b",
        re.IGNORECASE,
    )

    def _length_plan(self, therapist_msg, dynamic_state, ceiling):
        """Pick a per-turn length mode from question type + emotional state + a little
        randomness. Returns (budget_tokens, max_sentences, cue). budget <= ceiling, so
        this only ever shortens replies (latency never worse than the flat cap)."""
        msg = (therapist_msg or "").strip()
        n_words = len(msg.split())
        closed = bool(self._CLOSED_RE.search(msg))          # yes/no or factual (checked first)
        is_open = (not closed) and bool(self._OPEN_RE.search(msg))
        if is_open:
            idx = 2                       # moderate
        elif closed or n_words <= 6:
            idx = 0                       # terse
        else:
            idx = 1                       # brief
        beh = " ".join((dynamic_state or {}).get("current_behaviors", [])).lower()
        emo = " ".join((dynamic_state or {}).get("current_emotions", [])).lower()
        if any(k in beh for k in ("guarded", "withdraw", "hesitat", "avoid")):
            idx -= 1                      # shut-down state -> shorter
        if "slightly more open" in emo:
            idx += 1                      # opening up -> can say a little more
        r = random.random()               # natural variance; sometimes shorter than expected
        if r < 0.25:
            idx -= 1
        elif r > 0.9:
            idx += 1
        # Open/invitational questions get at least ~1 short sentence: never drop to the
        # terse "few words" mode, and enforce a token floor so we don't get a one-word
        # fragment like "Just...".
        if is_open:
            idx = max(idx, 1)
        idx = max(0, min(2, idx))
        budget, max_sents, cue = [
            (min(ceiling, 26), 1, "For THIS reply, answer very briefly — just a few words or one short sentence."),
            (min(ceiling, 55), 2, "For THIS reply, keep it short — about one or two sentences."),
            (ceiling, 4, "For THIS reply, you may share a little more if it truly matters to you, but stay concise."),
        ][idx]
        min_tokens = 12 if is_open else 0   # ~1 short sentence floor for open questions
        return budget, max_sents, cue, min_tokens

    def _generate_sync(
        self,
        conversation: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.15,
        length_cue: Optional[str] = None,
        max_sentences: int = 4,
        min_new_tokens: int = 0,
    ) -> str:
        messages = self._build_messages(conversation)
        if length_cue and messages and messages[0].get("role") == "system":
            messages = [dict(m) for m in messages]
            messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + length_cue

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )

        model_inputs = {
            k: v.to(self.model.device)
            for k, v in model_inputs.items()
        }

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        raw_response = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        return self._postprocess(raw_response, max_sentences=max_sentences)

    async def generate(
        self,
        conversation: List[Dict],
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        user_message = self._get_latest_user_message(conversation)

        if user_message:
            self.update_dynamic_state_from_user(user_message)

        # Dynamic per-turn length: vary the token budget + sentence cap with the
        # question type and the patient's current emotional state (plus a little
        # randomness). Budget is capped at max_new_tokens, so replies only shorten.
        dyn = self.state_manager.get_dynamic_state() if self.state_manager else None
        budget, max_sents, cue, min_toks = self._length_plan(user_message, dyn, max_new_tokens)

        reply = await asyncio.to_thread(
            self._generate_sync,
            conversation,
            budget,
            temperature,
            top_p,
            do_sample,
            1.15,
            cue,
            max_sents,
            min_toks,
        )

        # Break the "I don't know" / despair-repetition loop: if the reply is a
        # refusal OR too similar to recent patient turns, regenerate with rising
        # temperature and repetition penalty so we escape the mode instead of
        # resampling back into it (keeping the same length budget/cue).
        recent = [t.get("content", "") for t in conversation if t.get("role") == "patient"][-3:]
        attempts = 0
        while (self._is_refusal(reply) or self._too_similar(reply, recent)) and attempts < 3:
            attempts += 1
            esc_temp = min(1.05, temperature + 0.08 * attempts)
            esc_rep = min(1.25, 1.15 + 0.04 * attempts)
            reply = await asyncio.to_thread(
                self._generate_sync,
                conversation,
                budget,
                esc_temp,
                top_p,
                do_sample,
                esc_rep,
                cue,
                max_sents,
                min_toks,
            )

        return reply

    # =====================================================
    # Post-processing
    # =====================================================

    @staticmethod
    def extract_state(text: str) -> tuple[Optional[str], str]:
        match = re.search(
            r"<state>\s*(.*?)\s*</state>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        state = match.group(1).strip() if match else None

        cleaned = re.sub(
            r"<state>\s*.*?\s*</state>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        return state, cleaned.strip()

    @staticmethod
    def remove_think_artifacts(text: str) -> str:
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]

        if not parts:
            return ""

        deduped = []
        seen = set()

        for p in parts:
            key = re.sub(r"\s+", " ", p.strip().lower())
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        if len(deduped) == 1:
            return deduped[0]

        return "\n\n".join(deduped)

    @staticmethod
    def remove_role_leakage(text: str) -> str:
        text = text.strip()

        text = re.sub(
            r"^(system|user|assistant|therapist|patient)\s*\n+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        text = re.sub(
            r"^(system|user|assistant|therapist|patient)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        return text

    @staticmethod
    def normalize_format(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text.strip()

    @staticmethod
    def remove_duplicate_paragraphs(text: str) -> str:
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", text)
            if p.strip()
        ]

        if not paragraphs:
            return text.strip()

        kept = []
        seen = set()

        for p in paragraphs:
            key = re.sub(r"\s+", " ", p.lower())
            if key not in seen:
                seen.add(key)
                kept.append(p)

        return "\n\n".join(kept).strip()

    @staticmethod
    def trim_truncated_ending(text: str) -> str:
        text = text.strip()

        if not text:
            return text

        if re.search(r'[.!?]["\')\]]?$', text):
            return text

        matches = list(re.finditer(r'[.!?]["\')\]]?', text))

        if matches:
            return text[:matches[-1].end()].strip()

        return text

    @staticmethod
    def looks_like_user_prompt_echo(text: str) -> bool:
        lowered = text.lower().strip()

        question_starters = [
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
        ]

        count = sum(1 for q in question_starters if q in lowered)

        if count >= 3:
            return True

        num_questions = lowered.count("?")
        has_first_person = any(
            x in lowered
            for x in [" i ", " i'm ", " i’m ", " me ", " my "]
        )

        if num_questions >= 5 and not has_first_person:
            return True

        return False

    @classmethod
    def postprocess_model_output(
        cls,
        raw_text: str,
    ) -> Dict[str, Optional[str]]:
        state, text = cls.extract_state(raw_text)

        text = cls.remove_think_artifacts(text)
        text = cls.remove_role_leakage(text)
        text = cls.normalize_format(text)
        text = cls.remove_duplicate_paragraphs(text)
        text = cls.trim_truncated_ending(text)

        return {
            "state": state,
            "clean_text": text,
        }

    # Matches "I don't know / I do not know / I don't remember / I'm not sure /
    # not really sure / no idea", tolerant of straight OR curly apostrophes.
    _REFUSAL_RE = re.compile(
        r"i\s+(?:do\s*n[’']?t|do\s+not)\s+(?:know|remember)"
        r"|i\s*[’']?m\s+not\s+sure"
        r"|not\s+really\s+sure"
        r"|no\s+idea",
        flags=re.IGNORECASE,
    )

    @classmethod
    def _is_refusal(cls, text: str) -> bool:
        """True if the reply is empty or contains an 'I don't know'-style phrase."""
        if not text or not text.strip():
            return True
        return bool(cls._REFUSAL_RE.search(text))

    @staticmethod
    def _norm_words(s: str) -> list:
        return re.findall(r"[a-z']+", (s or "").lower())

    @classmethod
    def _too_similar(cls, text: str, recent: list, thresh: float = 0.6) -> bool:
        """True if `text` near-duplicates any of the recent patient replies
        (word-set Jaccard) — catches the despair-repetition loop."""
        w = set(cls._norm_words(text))
        if len(w) < 4:
            return False
        for prev in recent or []:
            pw = set(cls._norm_words(prev))
            if not pw:
                continue
            union = len(w | pw)
            if union and len(w & pw) / union >= thresh:
                return True
        return False

    @staticmethod
    def _cap_sentences(text: str, max_sentences: int = 4) -> str:
        """Backstop against rambling: keep at most `max_sentences` sentences."""
        text = text.strip()
        if not text:
            return text
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= max_sentences:
            return text
        return " ".join(sentences[:max_sentences]).strip()

    def _postprocess(self, text: str, max_sentences: int = 4) -> str:
        result = self.postprocess_model_output(text)
        clean_text = result["clean_text"] or ""

        if self.looks_like_user_prompt_echo(clean_text):
            return "It's hard to put into words, but it just feels like a lot pressing down on me right now."

        return self._cap_sentences(clean_text, max_sentences=max_sentences)