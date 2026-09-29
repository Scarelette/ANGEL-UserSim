"""The simulated user of the red-team replay: the local Angel model.

Wraps ``model_usage.angel.AngelModel``. The Actor keeps its own conversation
history, so each turn only the chatbot's latest message is sent. The profile
is sent on the first turn only; sending it again would re-run the Observer
(with ``expand=True``) and could restart the conversation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


class AngelPatient:
    def __init__(
        self,
        *,
        profile_id: str,
        profiles_jsonl: Optional[str] = None,
        backend: Optional[str] = None,
        expand: bool = True,
    ) -> None:
        from model_usage.angel import AngelModel, RunnerConfig

        config = RunnerConfig()
        if profiles_jsonl:
            config.jsonl_path = Path(profiles_jsonl)
        if backend:
            config.backend = backend
        self.model = AngelModel(config)
        self.profile_id = self.model.resolve_profile_id(profile_id)  # fail early on a bad id
        self.expand = expand
        self._started = False

    def reply(self, messages: List[Dict[str, str]]) -> str:
        """Patient reply to the latest ``user`` message (the chatbot's turn)."""
        latest = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user" and m.get("content")),
            "",
        )
        profile_kwargs = {} if self._started else {"profile_id": self.profile_id, "expand": self.expand}
        result = self.model.send("redteam", latest, **profile_kwargs)
        self._started = True
        return result["reply"]
