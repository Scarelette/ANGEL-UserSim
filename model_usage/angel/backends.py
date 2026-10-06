"""Text-generation backends.

Three implementations behind one small interface:

- `VLLMBackend`  — vLLM, the default engine (the one the released demo uses).
                   Needs a GPU and ~16 GB.
- `HFBackend`    — transformers. Same weights, same prompts, but a different
                   sampling implementation, so output is not bit-comparable
                   with vLLM even at identical settings.
- `StubBackend`  — deterministic canned output. No torch, no weights, no GPU.

`build_backend("auto", ...)` prefers vLLM and falls back to transformers.

The stub is what makes this package testable: the whole pipeline (profile
adaptation, session state, reply cleanup, CLI) runs against `StubBackend` on a
login node, and swapping in a real backend changes nothing but the text source.
"""

from __future__ import annotations

import sys

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Anything that can turn chat messages into text."""

    name: str

    def generate(self, messages: List[Dict[str, str]], **params: Any) -> str:
        ...

    def unload(self) -> None:
        ...


class HFBackend:
    """transformers-backed causal LM.

    Loading is lazy: constructing the object is cheap, and the checkpoint is read
    on the first `generate` call (or an explicit `load()`). That keeps
    `--help`, config validation, and profile listing fast.
    """

    name = "hf"

    def __init__(
        self,
        model_path: str,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # The cuDNN SDPA backend fails to build an execution plan on some GPU
        # nodes ("No valid execution plans built"), which makes every generate()
        # raise. Disabling it falls back to FlashAttention / mem-efficient / math
        # kernels, which work across our node types.
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_cudnn_sdp(False)
            except Exception:
                pass

        torch_dtype = getattr(torch, self.dtype) if torch.cuda.is_available() else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

    def _apply_chat_template(self, messages: List[Dict[str, str]], enable_thinking: Optional[bool]):
        """Tokenize chat messages, tolerating tokenizers without `enable_thinking`.

        `enable_thinking` is a Qwen3 chat-template argument. Older or non-Qwen
        templates reject the kwarg, so fall back rather than hard-fail.
        """
        kwargs: Dict[str, Any] = dict(
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        if enable_thinking is not None:
            try:
                return self._tokenizer.apply_chat_template(
                    messages, enable_thinking=enable_thinking, **kwargs
                )
            except TypeError:
                pass
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 180,
        min_new_tokens: int = 0,
        temperature: float = 0.75,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        enable_thinking: Optional[bool] = None,
        **_ignored: Any,
    ) -> str:
        import torch

        self.load()

        model_inputs = self._apply_chat_template(messages, enable_thinking)
        model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}
        input_ids = model_inputs["input_ids"]

        generate_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=model_inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        # temperature is only meaningful when sampling; passing it with
        # do_sample=False makes transformers warn on every call.
        if do_sample:
            generate_kwargs["temperature"] = temperature
        if no_repeat_ngram_size:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.no_grad():
            output_ids = self._model.generate(**generate_kwargs)

        new_tokens = output_ids[0][input_ids.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        """Drop the weights and free VRAM.

        Needed because Observer + Actor are ~16 GB each: on a 40 GB card the
        Observer must go before the Actor arrives.
        """
        if self._model is None:
            return
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Stub backend
# --------------------------------------------------------------------------

_STUB_REPLIES = (
    "Mostly tired, I think. It's been a long few weeks.",
    "I keep going over the same evening in my head, the one after my shift ended.",
    "Honestly? I stopped answering my sister's calls about a month ago.",
    "There's a knot in my chest most mornings before I even get out of bed.",
    "I used to run in the evenings. I haven't done that since spring.",
    "My manager asked if I was okay and I just said I was fine.",
)


def _stub_observer_profile(short_text: str) -> Dict[str, Any]:
    """A schema-complete Observer response, so adapter/session tests exercise
    the real mapping code rather than a hand-written rich profile."""
    text = short_text or ""
    # A rendered profile block starts "Name: <name>"; free text just has the name in
    # prose. Read the labelled form first so the label itself is not mistaken for
    # the name.
    name_match = re.search(r"^Name:\s*(\S+)", text, re.MULTILINE) or re.search(
        r"\b([A-Z][a-z]{2,})\b", text
    )
    # Prefer an explicit "34 years old" / "34yo"; fall back to any standalone
    # two-digit number, which is how ages usually appear in short referral notes.
    age_match = re.search(r"\b(\d{1,2})\s*[- ]?\s*(?:year|yr|yo)\b", text, re.IGNORECASE) or re.search(
        r"\b(\d{2})\b", text
    )
    return {
        "identity": {
            "name": name_match.group(1) if name_match else "Alex",
            "age": int(age_match.group(1)) if age_match else 31,
            "gender": "unspecified",
            "role": "patient",
            "source_title": "stub backend",
            "diagnosis_hint": ["stub hint"],
        },
        "background": ["Lives alone in a rented flat.", "Works shift hours at a warehouse."],
        "presenting_problems": ["Low mood most days", "Trouble falling asleep"],
        "symptom_details": {
            "onset_and_course": "Gradual over four months.",
            "frequency_and_intensity": "Most days, moderate.",
            "triggers": ["Sunday evenings", "Unanswered messages"],
            "maintaining_factors": ["Isolation"],
            "functional_impairment": ["Missing shifts"],
        },
        "emotional_profile": {
            "dominant_emotions": ["flat", "irritable"],
            "emotional_expression_style": "understated",
            "shame_or_fear_points": ["Being seen as unreliable"],
            "typical_reaction_when_distressed": "Goes quiet.",
        },
        "cognitive_patterns": {
            "core_beliefs": ["I let people down"],
            "automatic_thoughts": ["They will stop asking eventually"],
            "worries_or_ruminations": ["Replaying conversations"],
            "interpretation_biases": ["Reads neutral messages as annoyance"],
        },
        "behavior_patterns": {
            "avoidance_behaviors": ["Screening calls"],
            "safety_behaviors": ["Arriving early to avoid small talk"],
            "coping_strategies": ["Long walks at night"],
            "interpersonal_patterns": ["Withdraws before being rejected"],
        },
        "social_and_family_context": {
            "family_relationships": "One sister, contact has lapsed.",
            "romantic_or_peer_relationships": "No current relationship.",
            "school_or_work_context": "Warehouse shift work.",
            "social_support": "Thin.",
            "cultural_or_contextual_factors": "First in family to move cities.",
        },
        "risk_and_protective_factors": {
            "risk_factors": ["Isolation"],
            "protective_factors": ["Stable housing"],
            "safety_notes": "No stated intent.",
        },
        "hidden_state": {
            "information_patient_initially_withholds": ["The lapsed contact with the sister"],
            "information_revealed_after_trust": ["A missed funeral"],
            "topics_patient_avoids": ["Family obligations"],
            "contradictions_or_ambivalence": ["Says they want space, then mentions loneliness"],
        },
        "speaking_style": {
            "verbosity": "short answers",
            "tone": "flat, a little wry",
            "word_choice": "plain",
            "interaction_style": "answers what is asked, rarely volunteers",
            "example_phrases": ["It's fine.", "I guess."],
        },
        "disclosure_rules": {
            "early_session": "Sleep and work stress",
            "middle_session": "Withdrawal from friends",
            "late_session": "The missed funeral",
            "when_challenged": "Deflects with humour",
            "when_supported": "Offers one more detail",
        },
        "simulation_rules": [
            "Stay understated.",
            "Do not volunteer the funeral until trust is built.",
        ],
        # A real Observer writes a genuine one-or-two-sentence summary. Echoing the
        # whole input here would mask bugs in how callers feed stage 1 (a rendered
        # profile block is thousands of characters), so keep it short like the real
        # thing: first sentence of the input, capped.
        "brief_summary": (
            (re.split(r"(?<=[.!?])\s+", " ".join((short_text or "").split()))[0][:200]
            or "Adult presenting with low mood and sleep disruption.")
        ),
    }


class StubBackend:
    """Deterministic fake generator for tests and no-GPU smoke runs.

    Detects which stage is calling from the system prompt: the Observer's system
    prompt is the profile-writer instruction, so it gets JSON; anything else is
    the Actor and gets a short patient line. Replies cycle through distinct
    sentences so the near-duplicate retry loop terminates.
    """

    name = "stub"

    def __init__(self, replies: Optional[List[str]] = None) -> None:
        self.replies = list(replies) if replies else list(_STUB_REPLIES)
        self.calls: List[List[Dict[str, str]]] = []

    def generate(self, messages: List[Dict[str, str]], **params: Any) -> str:
        self.calls.append([dict(m) for m in messages])

        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        if "clinical case-profile writer" in system:
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            short_text = ""
            match = re.search(r"Short patient description:\n(.*?)\n\nGenerate", user, re.DOTALL)
            if match:
                short_text = match.group(1).strip()
            return json.dumps(_stub_observer_profile(short_text), indent=2)

        # Actor: advance through the reply list so each turn differs.
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        index = (turn + len(self.calls) - 1) % len(self.replies)
        return self.replies[index]

    def unload(self) -> None:
        return None


class VLLMBackend:
    """vLLM-backed causal LM (the default engine).

    bfloat16, a fixed `gpu_memory_utilization` fraction, `enforce_eager=True`
    (no CUDA-graph compilation, so it also runs where `nvcc` is unavailable) and
    the FlashInfer sampler disabled. Engine calls are serialized behind a lock.

    Sampling goes through `SamplingParams`, which is *not* bit-comparable with
    transformers' `generate` even at identical settings.
    """

    name = "vllm"

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 8192,
    ) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self._llm is not None:
            return
        # Must be set before vllm is imported.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM

        self._llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enforce_eager=True,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 90,
        min_new_tokens: int = 0,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        enable_thinking: Optional[bool] = None,
        seed: Optional[int] = None,
        **_ignored: Any,
    ) -> str:
        self.load()
        from vllm import SamplingParams

        sp_kwargs: Dict[str, Any] = dict(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens
        )
        # Optional seed for reproducible sampling.
        if seed is not None:
            sp_kwargs["seed"] = seed
        if repetition_penalty is not None:
            sp_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size:
            sp_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if min_new_tokens:
            sp_kwargs["min_tokens"] = min_new_tokens
        # Drop optional args this vLLM rejects one at a time, so an unsupported
        # one doesn't take the others with it. vLLM has no no_repeat_ngram_size
        # at all (transformers-only); min_tokens is supported by any recent vLLM.
        for optional in ("no_repeat_ngram_size", "min_tokens", None):
            try:
                sampling = SamplingParams(**sp_kwargs)
                break
            except TypeError:
                if optional is None:
                    raise
                sp_kwargs.pop(optional, None)

        # No per-call progress bars: they clutter the interactive chat.
        chat_kwargs: Dict[str, Any] = {"use_tqdm": False}
        if enable_thinking is not None:  # Qwen3: toggle the <think> block
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        with self._lock:
            outputs = self._llm.chat([messages], sampling, **chat_kwargs)
        return outputs[0].outputs[0].text

    def unload(self) -> None:
        if self._llm is None:
            return
        self._llm = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


_VLLM_ERROR: Optional[str] = None
_VLLM_CHECKED = False


def vllm_available() -> bool:
    """True if vLLM's engine can actually be imported (not just installed).

    A broken install (e.g. a C++ runtime mismatch) fails only when the engine
    is imported, so `auto` imports it here and falls back to transformers.
    """
    global _VLLM_CHECKED, _VLLM_ERROR
    if not _VLLM_CHECKED:
        _VLLM_CHECKED = True
        import importlib.util

        if importlib.util.find_spec("vllm") is None:
            _VLLM_ERROR = "not installed"
        else:
            try:
                from vllm import LLM, SamplingParams  # noqa: F401
            except Exception as exc:  # ImportError, OSError, ...
                _VLLM_ERROR = f"{type(exc).__name__}: {exc}"
    return _VLLM_ERROR is None


class NoGPUError(RuntimeError):
    """No CUDA GPU is visible (e.g. on a login node)."""


def _gpu_available() -> bool:
    # A missing torch must surface as ModuleNotFoundError (with the install
    # hint), not be misreported as "no GPU".
    import torch

    # device_count() asks NVML and leaves CUDA uninitialized; is_available()
    # would initialize it here and break vLLM's forked engine process
    # ("Cannot re-initialize CUDA in forked subprocess").
    return torch.cuda.device_count() > 0


_NO_GPU_HINT = (
    "no GPU found; you are probably on a login node. Get a GPU node first "
    "(e.g. `srun --gres=gpu:1 --mem=64G --time=01:00:00 --pty bash`), "
    "or use `--backend stub` for canned replies (`--backend hf` forces a very slow CPU run)."
)


def build_backend(
    kind: str,
    model_path: str,
    *,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.4,
    max_model_len: int = 8192,
) -> Backend:
    """Construct a backend.

    `auto` prefers vLLM and falls back to transformers.
    """
    kind = (kind or "auto").lower()

    if kind == "stub":
        return StubBackend()

    # vLLM fails on CPU-only hosts with an opaque "Device string must not be
    # empty"; catch that here. An explicit `hf` is still allowed on CPU.
    if kind in ("auto", "vllm") and not _gpu_available():
        raise NoGPUError(_NO_GPU_HINT)

    if kind == "auto":
        if vllm_available():
            kind = "vllm"
        else:
            if _VLLM_ERROR != "not installed":
                print(f"[warn] vLLM could not be loaded ({_VLLM_ERROR}); using transformers instead.",
                      file=sys.stderr)
            kind = "hf"

    if kind == "vllm":
        return VLLMBackend(
            model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    if kind == "hf":
        return HFBackend(model_path, device_map=device_map, dtype=dtype)

    raise ValueError(f"Unknown backend '{kind}' (use 'auto', 'vllm', 'hf' or 'stub')")
