"""Stage 1 — the Observer model.

Takes a short free-text patient description and expands it into a structured
long profile, then adapts that onto the rich schema the Actor consumes.

Every conversation runs through this stage by default, so the Actor always
role-plays the Observer's long profile. It is skipped only with `expand=False`,
for a profile that is already an Observer expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .backends import Backend, build_backend
from .config import ObserverConfig
from .demo_prompt import profile_to_short_text
from .observer_prompts import build_long_profile_messages, parse_profile_json
from .schema_adapter import adapt_observer_profile


@dataclass
class ExpansionResult:
    """Both stages of the stage-1 output, so callers can log the raw form."""

    rich_profile: Dict[str, Any]
    observer_profile: Dict[str, Any]
    raw_text: str
    attempts: int
    errors: List[str] = field(default_factory=list)


class Observer:
    def __init__(
        self,
        config: Optional[ObserverConfig] = None,
        *,
        backend: Optional[Backend] = None,
        backend_kind: str = "auto",
        device_map: str = "auto",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 8192,
    ) -> None:
        self.config = config or ObserverConfig()
        self.backend = backend or build_backend(
            backend_kind,
            self.config.model_path,
            device_map=device_map,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def expand(
        self,
        short_profile: Union[str, Dict[str, Any]],
        *,
        source_title: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ExpansionResult:
        """Expand a short profile into a rich profile.

        `short_profile` is either free text or a rich-schema profile dict. A dict
        is rendered to text with `profile_to_short_text` first.

        Stage 1 must emit parseable JSON. A first sample that truncates or wraps
        the object in prose is common, so retry up to `config.max_attempts` and
        surface every failure rather than silently falling back.
        """
        if isinstance(short_profile, dict):
            # An empty dict renders to a template of "Unknown"/"(none provided)"
            # placeholders, which would expand into a generic invented patient.
            if not short_profile:
                raise ValueError("short_profile is empty")
            text = profile_to_short_text(short_profile).strip()
            if not source_title:
                identity = short_profile.get("identity") or {}
                source_title = str(identity.get("source_title") or "")
            # The adapter appends `short_profile_text` to `background` as one
            # bullet. That is right for a one-paragraph referral note, but a
            # rendered profile block is thousands of characters and would be
            # flattened (newlines collapsed) into a single unreadable entry that
            # then reaches the Actor's prompt. The block's facts were the
            # Observer's input, so its own background/summary already carry them.
            background_seed = ""
        else:
            text = (short_profile or "").strip()
            background_seed = text
        if not text:
            raise ValueError("short_profile is empty")

        messages = build_long_profile_messages(text)
        errors: List[str] = []
        raw = ""

        for attempt in range(1, max(1, self.config.max_attempts) + 1):
            raw = self.backend.generate(
                messages,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                top_p=top_p if top_p is not None else self.config.top_p,
                do_sample=True,
                # Thinking stays on by default: the Observer was GRPO-trained with
                # the <think> block. parse_profile_json strips it afterwards.
                enable_thinking=self.config.enable_thinking,
            )
            try:
                observer_profile = parse_profile_json(raw)
            except Exception as exc:
                errors.append(f"attempt {attempt}: {exc}")
                continue

            rich_profile = adapt_observer_profile(
                observer_profile, source_title=source_title, short_profile_text=background_seed
            )
            return ExpansionResult(
                rich_profile=rich_profile,
                observer_profile=observer_profile,
                raw_text=raw,
                attempts=attempt,
                errors=errors,
            )

        raise ValueError(
            "Observer did not return parseable JSON after "
            f"{self.config.max_attempts} attempt(s):\n  - " + "\n  - ".join(errors)
            + f"\n\nLast raw output (truncated):\n{raw[:600]}"
        )

    def unload(self) -> None:
        self.backend.unload()
