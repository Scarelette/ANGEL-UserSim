from abc import ABC, abstractmethod
from typing import List, Dict


class AIPatient(ABC):
    def __init__(self, system_prompt: str):
        if not isinstance(system_prompt, str):
            raise TypeError(f"system_prompt must be a string, got {type(system_prompt)}")
        self.system_prompt = system_prompt.strip()

    # -------------------------------------------------
    # Build conversation messages
    # -------------------------------------------------
    def _build_messages(self, conversation: List[Dict]) -> List[Dict]:
        messages = [{"role": "system", "content": self.system_prompt}]

        for turn in conversation:
            role = turn.get("role")
            content = turn.get("content", "")

            if not content:
                continue

            if role == "therapist":
                messages.append({"role": "user", "content": content})
            elif role == "patient":
                messages.append({"role": "assistant", "content": content})
            elif role in {"system", "user", "assistant"}:
                messages.append({"role": role, "content": content})
            else:
                raise ValueError(f"Unknown role: {role}")

        return messages

    @staticmethod
    def _format_history(conversation: List[Dict]) -> str:
        lines = []
        for m in (conversation or []):
            role = (m.get("role") or "").lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in {"therapist", "patient"}:
                continue
            lines.append(f"{role}:{content}")
        return "\n".join(lines)

    @abstractmethod
    async def generate(self, conversation: List[Dict], **kwargs) -> str:
        pass