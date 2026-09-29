"""Configuration for the two-stage Angel model runner.

Generation settings are fixed to the released demo's values. What you can set:

- model weights: ``--observer-model`` / ``--actor-model``, else
  ``ANGEL_OBSERVER_MODEL`` / ``ANGEL_ACTOR_MODEL``, else
  ``<repo>/models/Qwen3-Observer-800`` / ``<repo>/models/qwen3-8b-dpo-merged``,
  else a Hugging Face Hub id (``angel_common.paths.resolve_model``);
- the inference engine: ``--backend``, else ``ANGEL_BACKEND`` (default ``auto``);
- vLLM's GPU memory share: ``ANGEL_VLLM_GPU_MEM`` (default ``0.4``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from angel_common.paths import resolve_model

PACKAGE_DIR = Path(__file__).resolve().parent
USAGE_DIR = PACKAGE_DIR.parent

# Example profiles shipped with the repo (synthetic; replace with your own).
DEFAULT_JSONL_PATH = USAGE_DIR / "examples" / "profiles.jsonl"


def _is_local(path: str) -> bool:
    return Path(path).exists()


@dataclass
class ObserverConfig:
    """Stage 1: short profile text -> structured long-profile JSON."""

    model_path: str = field(default_factory=lambda: resolve_model("observer"))
    max_new_tokens: int = 3072
    temperature: float = 0.7
    top_p: float = 0.9
    max_attempts: int = 2
    # Qwen3 thinking stays on: the Observer was trained with the <think> block.
    # The JSON extractor strips it afterwards.
    enable_thinking: bool = True


@dataclass
class ActorConfig:
    """Stage 2: long profile + dialogue history -> patient reply."""

    model_path: str = field(default_factory=lambda: resolve_model("actor"))
    max_new_tokens: int = 90          # ceiling for the per-turn length plan
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    max_turns: int = 12               # history kept by the dynamic-state tracker
    max_retries: int = 3              # regenerations of a refusal / repeated reply
    max_sentences: int = 4


@dataclass
class RunnerConfig:
    """Top-level runtime settings."""

    observer: ObserverConfig = field(default_factory=ObserverConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    jsonl_path: Path = DEFAULT_JSONL_PATH
    device_map: str = "auto"
    dtype: str = "bfloat16"
    # Observer + Actor are ~16 GB each. Freeing stage 1 after expansion keeps the
    # pipeline inside a 40 GB card; keep_both is for larger cards doing batch runs.
    keep_both_resident: bool = False
    seed: Optional[int] = None
    # "auto" prefers vLLM and falls back to transformers; "vllm" / "hf" force
    # one engine; "stub" is a no-GPU fake for tests.
    backend: str = field(default_factory=lambda: os.environ.get("ANGEL_BACKEND") or "auto")
    # vLLM engine settings (0.4 leaves room for both models on one large GPU).
    vllm_gpu_memory_utilization: float = field(
        default_factory=lambda: float(os.environ.get("ANGEL_VLLM_GPU_MEM") or 0.4)
    )
    vllm_max_model_len: int = 8192

    def resolved_backend(self) -> str:
        """The concrete engine that will be used (resolves 'auto')."""
        if self.backend != "auto":
            return self.backend
        from .backends import vllm_available

        return "vllm" if vllm_available() else "hf"

    def missing_paths(self, *, need_observer: bool, need_actor: bool) -> List[str]:
        """Report unusable paths so callers fail with a readable message rather
        than a transformers stack trace 30 seconds later."""
        problems: List[str] = []
        if not self.jsonl_path.exists():
            problems.append(f"profiles JSONL not found: {self.jsonl_path}")
        if self.backend == "stub":
            return problems  # the stub backend needs no weights at all
        if need_observer and not _looks_loadable(self.observer.model_path):
            problems.append(
                f"observer model not found: {self.observer.model_path}\n"
                "    set ANGEL_OBSERVER_MODEL or pass --observer-model"
            )
        if need_actor and not _looks_loadable(self.actor.model_path):
            problems.append(
                f"actor model not found: {self.actor.model_path}\n"
                "    set ANGEL_ACTOR_MODEL or pass --actor-model"
            )
        return problems

    def preflight(self, *, need_observer: bool = False, need_actor: bool = True) -> None:
        problems = self.missing_paths(need_observer=need_observer, need_actor=need_actor)
        if problems:
            raise FileNotFoundError(
                "model_usage.angel configuration problems:\n  - " + "\n  - ".join(problems)
            )

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "resolved_backend": self.resolved_backend(),
            "observer_model": self.observer.model_path,
            "actor_model": self.actor.model_path,
            "observer_local": _is_local(self.observer.model_path),
            "actor_local": _is_local(self.actor.model_path),
            "jsonl_path": str(self.jsonl_path),
            "device_map": self.device_map,
            "dtype": self.dtype,
            "keep_both_resident": self.keep_both_resident,
            "actor_generation": {
                "max_new_tokens": self.actor.max_new_tokens,
                "temperature": self.actor.temperature,
                "top_p": self.actor.top_p,
                "repetition_penalty": self.actor.repetition_penalty,
                "no_repeat_ngram_size": self.actor.no_repeat_ngram_size,
                "max_retries": self.actor.max_retries,
            },
            "observer_generation": {
                "max_new_tokens": self.observer.max_new_tokens,
                "temperature": self.observer.temperature,
                "top_p": self.observer.top_p,
                "max_attempts": self.observer.max_attempts,
                "enable_thinking": self.observer.enable_thinking,
            },
        }


def _looks_loadable(model: str) -> bool:
    """A local directory, or something shaped like a Hub id (``org/name``)."""
    if Path(model).exists():
        return True
    return model.count("/") == 1 and not model.startswith((".", "/"))
