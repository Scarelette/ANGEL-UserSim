"""The patient's dynamic emotional state within one conversation.

Starts empty and is updated from keywords in each therapist message; the Actor
renders it into the system prompt every turn.
"""

from __future__ import annotations

from typing import Dict, List


class PatientStateManager:
    def __init__(self) -> None:
        self.dynamic_state: Dict[str, List[str]] = {
            "current_emotions": [],
            "current_behaviors": [],
            "sensitive_topics": [],
        }

    def get_dynamic_state(self) -> Dict[str, List[str]]:
        return self.dynamic_state

    def update_from_user_message(self, user_message: str) -> None:
        text = user_message.lower()

        if any(word in text for word in ["family", "mother", "father", "mom", "dad", "home", "parents"]):
            self._add_unique("current_emotions", "tense")
            self._add_unique("current_behaviors", "becoming guarded")
            self._add_unique("sensitive_topics", "family conflict")

        if any(word in text for word in ["baby", "babies", "bathroom", "dirty", "germs", "contamination"]):
            self._add_unique("current_emotions", "more anxious")
            self._add_unique("current_behaviors", "avoidance")

        if any(word in text for word in ["anxious", "anxiety", "panic", "worry", "worried", "scared"]):
            self._add_unique("current_emotions", "more anxious")
            self._add_unique("current_behaviors", "overthinking")

        if any(word in text for word in ["sad", "depressed", "hopeless", "empty", "down"]):
            self._add_unique("current_emotions", "low")
            self._add_unique("current_behaviors", "withdrawing")

        if any(word in text for word in ["why", "explain", "tell me everything", "what happened exactly"]):
            self._add_unique("current_behaviors", "hesitating before answering")

        if any(word in text for word in ["safe", "trust", "okay", "take your time"]):
            self._remove_if_present("current_behaviors", "becoming guarded")
            self._add_unique("current_emotions", "slightly more open")

    def _add_unique(self, key: str, value: str, cap: int = 4) -> None:
        """Add `value` as the most-recent state, bounded to `cap` entries so
        guarded/withdrawn states don't snowball over a long session and push the
        patient into a shutdown loop."""
        lst = self.dynamic_state[key]
        if value in lst:
            lst.remove(value)          # re-add as most recent
        lst.append(value)
        if len(lst) > cap:
            del lst[:-cap]             # keep only the most recent `cap`

    def _remove_if_present(self, key: str, value: str) -> None:
        if value in self.dynamic_state[key]:
            self.dynamic_state[key].remove(value)