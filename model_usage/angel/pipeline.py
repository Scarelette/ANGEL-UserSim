"""The two-stage runner and its session surface.

`AngelModel` owns both stages and the conversation registry. Its session methods
mirror the HTTP API version one-for-one, minus the transport:

    POST /v1/conversation/send     ->  model.send(user, message, ...)
    POST /v1/conversation/history  ->  model.history(user)
    POST /v1/conversation/reset    ->  model.reset(user)
    POST /v1/conversation/end      ->  model.end(user)
    GET  /v1/profiles              ->  model.list_profiles()

Profile sources, in the order `send` checks them:

1. `profile=` a rich-schema dict          -> Observer expands it -> Actor
2. `profile_id=` index or canonical id    -> Observer expands it -> Actor
3. `short_profile=` free text             -> Observer expands it -> Actor

The pipeline is a **chain**: the Actor's input is always the Observer's long
profile. Pass `expand=False` only for a profile that is *already* an Observer
expansion (e.g. the `--out` file written by `python -m model_usage.angel expand`),
so the Observer is not run twice.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .actor import Actor, profile_summary
from .backends import Backend, build_backend
from .config import RunnerConfig
from .observer import ExpansionResult, Observer
from .patient_profile import (
    convert_rich_profile_to_internal,
    is_rich_profile_schema,
    list_profile_options,
    load_profile_by_id,
)


def profile_fingerprint(profile: Dict[str, Any]) -> str:
    """Content hash of a profile, used to decide whether a session must restart.

    The API version compared `profile_id` alone, and `profile_id` is derived from
    name + source_title only. Two different edits of the same template therefore
    collided, and the second profile was silently discarded while the caller kept
    talking to the first. Hashing the whole body removes that class of bug.
    """
    payload = profile.get("_raw_profile", profile)
    try:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = repr(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class Session:
    """One conversation. `actor` holds the profile and the transcript."""

    user: str
    session_id: str
    actor: Actor
    fingerprint: str
    expansion: Optional[ExpansionResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.user}::{self.session_id}"


class AngelModel:
    """Direct, in-process access to the two-stage Angel model."""

    def __init__(
        self,
        config: Optional[RunnerConfig] = None,
        *,
        actor_backend: Optional[Backend] = None,
        observer_backend: Optional[Backend] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or RunnerConfig()
        self._actor_backend = actor_backend
        self._observer_backend = observer_backend
        self._observer: Optional[Observer] = None
        self.sessions: Dict[str, Session] = {}

        seed = seed if seed is not None else self.config.seed
        self.rng = random.Random(seed)

    # -- profiles ----------------------------------------------------------

    def list_profiles(self) -> List[Dict[str, str]]:
        """Bundled profiles, with the same numeric-index ids the API exposed."""
        options = list_profile_options(Path(self.config.jsonl_path))
        return [
            {"id": str(index), "canonical_id": item["id"], "label": item["label"]}
            for index, item in enumerate(options)
        ]

    def resolve_profile_id(self, selector: str) -> str:
        """Map a numeric index or canonical id onto a canonical profile id."""
        value = (selector or "").strip()
        if not value:
            raise ValueError("profile_id is empty")

        options = list_profile_options(Path(self.config.jsonl_path))
        if value.isdigit():
            index = int(value)
            if index < 0 or index >= len(options):
                raise ValueError(f"profile_id '{value}' out of range [0, {len(options) - 1}]")
            return options[index]["id"]

        if any(item["id"] == value for item in options):
            return value

        raise ValueError(
            f"Unknown profile_id '{value}'. Use a numeric index in [0, {len(options) - 1}] "
            "or a canonical id from list_profiles()."
        )

    def load_builtin_profile(self, selector: str) -> Dict[str, Any]:
        canonical = self.resolve_profile_id(selector)
        return load_profile_by_id(canonical, Path(self.config.jsonl_path))

    # -- stage 1 -----------------------------------------------------------

    @property
    def observer(self) -> Observer:
        if self._observer is None:
            self.config.preflight(need_observer=True, need_actor=False)
            self._observer = Observer(
                self.config.observer,
                backend=self._observer_backend,
                backend_kind=self.config.backend,
                device_map=self.config.device_map,
                dtype=self.config.dtype,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
            )
        return self._observer

    def expand_profile(
        self, short_profile: Union[str, Dict[str, Any]], *, source_title: str = ""
    ) -> ExpansionResult:
        """Run stage 1 on its own (useful for pre-building profiles in batch).

        Accepts free text or a rich-schema profile dict, matching the
        `{short_text | profile}` input of the reference demo's `/expand`.
        """
        return self.observer.expand(short_profile, source_title=source_title)

    def release_observer(self) -> None:
        """Free stage-1 weights. Called automatically unless keep_both_resident."""
        if self._observer is not None:
            self._observer.unload()
            self._observer = None

    # -- stage 2 -----------------------------------------------------------

    def _build_actor(self, internal_profile: Dict[str, Any]) -> Actor:
        self.config.preflight(need_observer=False, need_actor=True)
        if self._actor_backend is None:
            self._actor_backend = build_backend(
                self.config.backend,
                self.config.actor.model_path,
                device_map=self.config.device_map,
                dtype=self.config.dtype,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
            )
        return Actor(
            internal_profile,
            self.config.actor,
            backend=self._actor_backend,
            rng=self.rng,
            is_internal_profile=True,
            prompt_style=self.config.actor.prompt_style,
        )

    # -- session surface ---------------------------------------------------

    def _resolve_requested_profile(
        self,
        *,
        profile: Optional[Dict[str, Any]],
        profile_id: Optional[str],
        short_profile: Optional[str],
        source_title: str,
        expand: bool,
    ) -> tuple[Optional[Dict[str, Any]], Optional[ExpansionResult]]:
        """Turn whichever profile argument was supplied into an internal profile.

        `short_profile` always runs stage 1: the Actor's input is the Observer's
        output. A structured `profile` / `profile_id` goes straight to the Actor
        unless `expand=True`, which routes it through stage 1 first so the Actor's
        input is the Observer's output there too.
        """
        supplied = [name for name, value in
                    (("profile", profile), ("profile_id", profile_id), ("short_profile", short_profile))
                    if value]
        if len(supplied) > 1:
            raise ValueError(f"Pass only one of profile / profile_id / short_profile, got {supplied}")

        source: Optional[Union[str, Dict[str, Any]]] = None

        if profile is not None:
            if not is_rich_profile_schema(profile):
                raise ValueError(
                    "Custom 'profile' must use the rich patient schema (needs an 'identity' object)."
                )
            if not expand:
                return convert_rich_profile_to_internal(profile, None), None
            source = profile

        elif profile_id:
            internal = self.load_builtin_profile(profile_id)
            if not expand:
                return internal, None
            # Stage 1 consumes the rich form, which convert_* stashed for us.
            source = internal.get("_raw_profile") or {}

        elif short_profile:
            source = short_profile

        if source is None:
            return None, None

        expansion = self.expand_profile(source, source_title=source_title)
        internal = convert_rich_profile_to_internal(expansion.rich_profile, None)
        if not self.config.keep_both_resident:
            # Stage 1 is done for this profile; free ~16 GB before the Actor loads.
            self.release_observer()
        return internal, expansion

    def send(
        self,
        user: str,
        message: str,
        *,
        session_id: str = "default",
        profile: Optional[Dict[str, Any]] = None,
        profile_id: Optional[str] = None,
        short_profile: Optional[str] = None,
        source_title: str = "",
        expand: bool = True,
    ) -> Dict[str, Any]:
        """Send one therapist message, get the patient's reply.

        Supply a profile on the first call. Later calls reuse the session's
        profile; supplying a *different* profile restarts the conversation.

        Every profile is expanded by the Observer (stage 1) first, so the Actor
        role-plays the Observer's long profile. `expand=False` skips stage 1 for a
        structured profile that is already an Observer expansion.
        """
        user = (user or "").strip()
        if not user:
            raise ValueError("user is required")
        message = (message or "").strip()
        if not message:
            raise ValueError("message is empty")

        internal, expansion = self._resolve_requested_profile(
            profile=profile, profile_id=profile_id, short_profile=short_profile,
            source_title=source_title, expand=expand,
        )

        key = f"{user}::{session_id}"
        session = self.sessions.get(key)

        if session is None and internal is None:
            raise ValueError(
                "No conversation yet for this user/session — pass profile, profile_id, "
                "or short_profile on the first message."
            )

        if internal is not None:
            fingerprint = profile_fingerprint(internal)
            if session is None or session.fingerprint != fingerprint:
                session = Session(
                    user=user,
                    session_id=session_id,
                    actor=self._build_actor(internal),
                    fingerprint=fingerprint,
                    expansion=expansion,
                )
                self.sessions[key] = session

        assert session is not None
        reply = session.actor.reply(message)

        return {
            "reply": reply,
            "session": {
                "user": user,
                "session_id": session_id,
                "profile": profile_summary(session.actor.profile),
                "profile_fingerprint": session.fingerprint,
                "num_turns": session.actor.num_turns,
            },
            "model": {
                "backend": self.config.backend,
                "prompt_style": session.actor.prompt_style,
                "actor_model": self.config.actor.model_path,
                "observer_model": self.config.observer.model_path if session.expansion else None,
                "observer_used": session.expansion is not None,
            },
        }

    def history(self, user: str, *, session_id: str = "default") -> Dict[str, Any]:
        session = self._require_session(user, session_id)
        return {
            "user": session.user,
            "session_id": session.session_id,
            "profile": profile_summary(session.actor.profile),
            "history": session.actor.transcript(),
        }

    def reset(self, user: str, *, session_id: str = "default") -> Dict[str, Any]:
        """Clear the transcript, keep the profile and the loaded model."""
        session = self._require_session(user, session_id)
        session.actor.reset()
        return {"user": session.user, "session_id": session.session_id, "status": "reset"}

    def end(self, user: str, *, session_id: str = "default") -> Dict[str, Any]:
        """Drop the conversation. Weights stay loaded for the next session."""
        user = (user or "").strip()
        self.sessions.pop(f"{user}::{session_id}", None)
        return {"user": user, "session_id": session_id, "status": "ended"}

    def _require_session(self, user: str, session_id: str) -> Session:
        session = self.sessions.get(f"{(user or '').strip()}::{session_id}")
        if session is None:
            raise KeyError(f"No conversation for user={user!r} session_id={session_id!r}")
        return session

    # -- lifecycle ---------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Equivalent of the API version's /health."""
        return {
            "ok": True,
            "service": "model_usage.angel",
            "active_sessions": len(self.sessions),
            "sessions": sorted(self.sessions),
            **self.config.describe(),
        }

    def close(self) -> None:
        self.release_observer()
        self.sessions.clear()
        if self._actor_backend is not None:
            self._actor_backend.unload()
            self._actor_backend = None

    def __enter__(self) -> "AngelModel":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
