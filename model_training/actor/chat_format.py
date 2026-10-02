"""The single prompt format the Actor sees: Qwen3's chat template, thinking off.

Used by the rollouts (patient models), SFT, DPO and, with the same settings,
``model_usage`` at inference. Conversations use the roles ``system`` /
``user`` (therapist) / ``assistant`` (patient).

With thinking off, Qwen3's generation prompt is
``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``. ``tokenize_for_sft``
writes every patient turn after exactly that prefix, so each SFT target is
trained in the position the model generates it at inference and in DPO.
"""

from __future__ import annotations

from typing import Dict, List

_ROLE = {"therapist": "user", "user": "user", "patient": "assistant", "assistant": "assistant", "system": "system"}


def to_chat_messages(system_prompt: str, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """System prompt + turns with therapist/patient mapped to user/assistant; empty turns dropped."""
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    for turn in conversation or []:
        role = _ROLE.get((turn.get("role") or "").lower())
        content = (turn.get("content") or "").strip()
        if role and role != "system" and content:
            messages.append({"role": role, "content": content})
    return messages


def render(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render ``messages`` as text with the tokenizer's chat template, thinking disabled."""
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:  # tokenizers whose template has no thinking switch
        return tokenizer.apply_chat_template(messages, **kwargs)


def generation_prefix(tokenizer) -> str:
    """The text the chat template appends for a new assistant turn (thinking off)."""
    msgs = [{"role": "user", "content": "x"}]
    return render(tokenizer, msgs, add_generation_prompt=True)[len(render(tokenizer, msgs)):]


def tokenize_for_sft(tokenizer, messages: List[Dict[str, str]]) -> Dict[str, List[int]]:
    """``input_ids`` and ``assistant_masks`` (1 = trained) for one whole conversation.

    System and therapist turns are ChatML blocks, as the chat template writes
    them. Each patient turn is ``generation_prefix + reply + <|im_end|>``; the
    loss covers the reply and ``<|im_end|>`` (so the model learns to stop).
    """
    prefix = generation_prefix(tokenizer)
    input_ids: List[int] = []
    mask: List[int] = []

    def add(text: str, trained: bool) -> None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        mask.extend([int(trained)] * len(ids))

    for msg in messages:
        if msg["role"] == "assistant":
            add(prefix, False)
            add(msg["content"] + "<|im_end|>", True)
            add("\n", False)
        else:
            add(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n", False)
    return {"input_ids": input_ids, "assistant_masks": mask}
