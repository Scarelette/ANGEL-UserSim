"""Step 3 — convert SFT rollouts into chat-format SFT examples.

The graph-specific patient prompt is replaced by the generic
``prompts.SFT_SYSTEM_PROMPT``; every patient turn is normalized to
``<state>...</state>\\n<word>...</word>`` and becomes an ``assistant`` turn,
therapist turns become ``user`` turns.

    python -m model_training.actor.build_sft_data \
        --inputs data/actor/sft_rollouts/NM_mask_p{1,2,3,4,5}.jsonl \
        --output data/actor/sft_training.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from angel_common.paths import DATA_DIR, resolve_path
from model_training.actor.prompts import SFT_SYSTEM_PROMPT


def extract_from_patient(content: str):
    state_match = re.search(r"<state>(.*?)</state>", content, re.DOTALL)
    word_match = re.search(r"<word>(.*?)</word>", content, re.DOTALL)
    return (state_match.group(1).strip() if state_match else None,
            word_match.group(1).strip() if word_match else None)


def normalize_patient_turn(content: str) -> str:
    state, word = extract_from_patient(content)
    before_state = content.split("<state>")[0].strip()
    if state and word:
        return "<state>" + state + "</state>\n<word>" + word + "</word>"
    if state:
        return "<state>" + state + "</state>\n<word>" + before_state + "</word>"
    if word:
        return "<state></state>\n<word>" + word + "</word>"
    return "<state></state>\n<word>" + before_state + "</word>"


def convert(sample: dict) -> dict:
    final_conv = [{"role": "system", "content": SFT_SYSTEM_PROMPT}]
    for msg in sample.get("messages", []):
        content = msg.get("content", "")
        if msg.get("role") == "patient":
            final_conv.append({"role": "assistant", "content": normalize_patient_turn(content)})
        else:
            final_conv.append({"role": "user", "content": content})
    return {"messages": final_conv}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="rollout JSONL files from rollout_sft")
    ap.add_argument("--output", default=str(DATA_DIR / "actor" / "sft_training.jsonl"))
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "a") as writer:
        for path in args.inputs:
            with open(resolve_path(path)) as reader:
                for line in reader:
                    if line.strip():
                        writer.write(json.dumps(convert(json.loads(line)), ensure_ascii=False) + "\n")
                        n += 1
    print(f"wrote {n} examples to {out}")


if __name__ == "__main__":
    main()
