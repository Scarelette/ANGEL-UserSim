"""Stage 2 — the Actor model (the simulated patient).

Two prompt styles drive the same checkpoint:

`prompt_style="patient_demo"` (**default**), used by the interactive demo and the
user study:
  - system prompt: `demo_prompt.PATIENT_SYSTEM_TEMPLATE`, filled with the profile
    rendered by `profile_to_short_text`
  - dynamic state starts **empty** (`PatientStateManager({})`) and accumulates
    only from therapist keywords; the profile's own emotions/behaviors reach the
    model through the rendered profile block instead
  - the **full** conversation is sent every turn (no history window)
  - per turn the system prompt gets `\\n\\n<state block>` then `\\n\\n<length cue>`
  - output cleaned by `demo_prompt.clean_reply` (prefers `<patient>…</patient>`)

`prompt_style="angel_eval"`, the prompt of the paper's profile-expansion experiment:
  - system prompt: `patient_profile.build_system_prompt`, rebuilt each turn
  - dynamic state **seeded from the profile**
  - history windowed to `max_turns`
  - output cleaned by `postprocess.clean_reply` (handles `<state>`, role leakage,
    duplicate paragraphs, prompt-echo fallback)

Same weights, noticeably different behaviour.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from . import demo_prompt, postprocess
from .backends import Backend, build_backend
from .config import ActorConfig
from .length_plan import plan_reply_length
from .patient_profile import build_system_prompt, convert_rich_profile_to_internal
from .state_manager import PatientStateManager

PROMPT_STYLES = ("patient_demo", "angel_eval")


class Actor:
    """A single simulated patient bound to one profile."""

    def __init__(
        self,
        profile: Dict[str, Any],
        config: Optional[ActorConfig] = None,
        *,
        backend: Optional[Backend] = None,
        backend_kind: str = "auto",
        device_map: str = "auto",
        dtype: str = "bfloat16",
        rng: Optional[random.Random] = None,
        is_internal_profile: bool = False,
        prompt_style: Optional[str] = None,
    ) -> None:
        self.config = config or ActorConfig()
        self.prompt_style = prompt_style or self.config.prompt_style
        if self.prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}, got {self.prompt_style!r}")

        self.backend = backend or build_backend(
            backend_kind, self.config.model_path, device_map=device_map, dtype=dtype
        )
        self.rng = rng or random.Random()
        self.set_profile(profile, is_internal_profile=is_internal_profile)

    # -- profile / state ---------------------------------------------------

    def set_profile(self, profile: Dict[str, Any], *, is_internal_profile: bool = False) -> None:
        """Bind a new profile and reset the conversation.

        Both the rich profile and the converted internal profile are retained:
        `patient_demo` renders the rich form into the prompt, `angel_eval` builds
        its prompt from the internal form.
        """
        if is_internal_profile:
            self.profile = profile
            # `_raw_profile` is stashed by convert_rich_profile_to_internal.
            self.rich_profile = profile.get("_raw_profile") or {}
        else:
            self.profile = convert_rich_profile_to_internal(profile, None)
            self.rich_profile = copy.deepcopy(profile)

        if self.prompt_style == "patient_demo":
            # Built once, from the profile only.
            self.base_system_prompt = demo_prompt.build_patient_system_prompt(
                demo_prompt.profile_to_short_text(self.rich_profile)
            )
            # Empty profile: state accumulates from therapist keywords alone.
            self.state_manager = PatientStateManager(profile={}, max_turns=self.config.max_turns)
        else:
            self.state_manager = PatientStateManager(
                profile=self.profile, max_turns=self.config.max_turns
            )
            self.base_system_prompt = build_system_prompt(
                profile=self.profile, dynamic_state=self.state_manager.get_dynamic_state()
            )

        self.conversation: List[Dict[str, str]] = []

    def reset(self) -> None:
        """Clear the transcript, keep the profile."""
        self.set_profile(self.profile, is_internal_profile=True)

    @property
    def num_turns(self) -> int:
        return sum(1 for turn in self.conversation if turn["role"] == "patient")

    def transcript(self) -> List[Dict[str, str]]:
        """Flat role/content transcript, using user/assistant role names."""
        return [
            {"role": "user" if turn["role"] == "therapist" else "assistant", "content": turn["content"]}
            for turn in self.conversation
        ]

    # -- prompt assembly ---------------------------------------------------

    def _system_prompt_for_turn(self, cue: Optional[str]) -> str:
        dynamic_state = self.state_manager.get_dynamic_state()

        if self.prompt_style == "patient_demo":
            prompt = self.base_system_prompt.rstrip()
            state_block = demo_prompt.render_dynamic_state(dynamic_state)
            if state_block:
                prompt += "\n\n" + state_block
            if cue:
                prompt += "\n\n" + cue
            return prompt

        # angel_eval rebuilds the whole prompt from the evolving dynamic state.
        prompt = build_system_prompt(profile=self.profile, dynamic_state=dynamic_state)
        if cue:
            prompt = prompt.rstrip() + "\n\n" + cue
        return prompt

    def _build_messages(self, conversation: List[Dict[str, str]], cue: Optional[str]) -> List[Dict[str, str]]:
        if self.prompt_style == "patient_demo":
            # The full conversation is sent every turn.
            return demo_prompt.to_chat_messages(self._system_prompt_for_turn(cue), conversation)

        # angel_eval windows the history to the last `max_turns` exchanges.
        window = self.config.max_turns * 2
        windowed = conversation[-window:] if window > 0 else conversation
        return demo_prompt.to_chat_messages(self._system_prompt_for_turn(cue), windowed)

    def _clean(self, raw: str, max_sentences: int) -> str:
        if self.prompt_style == "patient_demo":
            return demo_prompt.clean_reply(raw, max_sentences)
        return postprocess.clean_reply(raw, max_sentences=max_sentences)

    # -- generation --------------------------------------------------------

    def _generate_once(
        self,
        conversation: List[Dict[str, str]],
        *,
        budget: int,
        min_new_tokens: int,
        max_sentences: int,
        temperature: float,
        repetition_penalty: float,
        cue: Optional[str],
    ) -> str:
        raw = self.backend.generate(
            self._build_messages(conversation, cue),
            max_new_tokens=budget,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            enable_thinking=False,
        )
        return self._clean(raw, max_sentences)

    def reply(self, user_message: str, *, record: bool = True) -> str:
        """Generate the patient's reply to one therapist message."""
        message = (user_message or "").strip()
        if not message:
            raise ValueError("user_message is empty")

        # State is updated before the prompt is built, so the shift this message
        # causes is visible in this turn's reply (matches both upstreams).
        self.state_manager.update_from_user_message(message)

        conversation = self.conversation + [{"role": "therapist", "content": message}]

        plan = plan_reply_length(
            message, self.state_manager.get_dynamic_state(), self.config.max_new_tokens, rng=self.rng
        )

        reply = self._generate_once(
            conversation,
            budget=plan.budget_tokens,
            min_new_tokens=plan.min_new_tokens,
            max_sentences=plan.max_sentences,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
            cue=plan.cue,
        )

        # Break the "I don't know" / despair-repetition loop: escalate temperature
        # and repetition penalty rather than resampling back into the same mode.
        recent = [t["content"] for t in self.conversation if t["role"] == "patient"][-3:]
        attempt = 0
        while (
            postprocess.is_refusal(reply) or postprocess.too_similar(reply, recent)
        ) and attempt < self.config.max_retries:
            attempt += 1
            reply = self._generate_once(
                conversation,
                budget=plan.budget_tokens,
                min_new_tokens=plan.min_new_tokens,
                max_sentences=plan.max_sentences,
                temperature=min(1.05, self.config.temperature + 0.08 * attempt),
                repetition_penalty=min(1.25, self.config.repetition_penalty + 0.04 * attempt),
                cue=plan.cue,
            )

        reply = (reply or "").strip()
        if record:
            self.conversation.append({"role": "therapist", "content": message})
            self.conversation.append({"role": "patient", "content": reply})
        return reply

    async def areply(self, user_message: str, *, record: bool = True) -> str:
        """Async wrapper for callers driving several patients concurrently."""
        import asyncio

        return await asyncio.to_thread(self.reply, user_message, record=record)

    def unload(self) -> None:
        self.backend.unload()


def profile_summary(internal_profile: Dict[str, Any]) -> Dict[str, Any]:
    """The public view of a profile (id, name, age, ...)."""
    return {
        "profile_id": internal_profile.get("profile_id"),
        "name": internal_profile.get("name"),
        "age": internal_profile.get("age"),
        "gender": internal_profile.get("gender"),
        "role": internal_profile.get("role"),
        "source_title": internal_profile.get("source_title"),
    }
