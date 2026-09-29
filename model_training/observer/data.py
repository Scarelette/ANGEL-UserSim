"""GRPO / SFT dataset loading for the Observer."""

from __future__ import annotations

import ast
from typing import Any, List, Optional

from model_training.observer.prompts import (
    build_input_s1,
    build_input_s2,
    build_system_prompt_s1,
    build_system_prompt_s2,
)


def _as_list(value: Any) -> List[str]:
    """`gpt5_nodes` is a list; tolerate a stringified list too."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [value]
    return []


def load_grpo_dataset(data_path: str, stage: str, limit: Optional[int] = None):
    """Build the GRPO prompt dataset.

    Expected JSONL fields per row:
      - ``Complaints``: str   — presenting complaints
      - ``gpt5_nodes``: list  — reference symptoms + external factors (GPT-5)

    S1 rows get ``prompt`` + ``ref_nodes``; S2 rows get ``prompt`` + ``complaints``
    (the node list is placed in the prompt).
    """
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path, split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    if stage == "s1":
        system_prompt = build_system_prompt_s1()

        def to_row(x):
            return {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_input_s1(x["Complaints"])},
                ],
                "ref_nodes": _as_list(x["gpt5_nodes"]),
            }
    elif stage == "s2":
        system_prompt = build_system_prompt_s2()

        def to_row(x):
            return {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_input_s2(x["Complaints"], _as_list(x["gpt5_nodes"]))},
                ],
                "complaints": x["Complaints"],
                "nodes": _as_list(x["gpt5_nodes"]),  # the edge reward checks endpoints against this list
            }
    else:
        raise ValueError(f"stage must be 's1' or 's2', got {stage!r}")

    return dataset.map(to_row)


def load_sft_text_dataset(data_path: str, tokenizer, limit: Optional[int] = None):
    """Chat-format JSONL ({"messages": [system, user, assistant]}) -> ``text`` column
    rendered with the tokenizer's (Qwen3) chat template, the same format GRPO and
    inference use."""
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path, split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    def fmt(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    return dataset.map(fmt, remove_columns=dataset.column_names)
