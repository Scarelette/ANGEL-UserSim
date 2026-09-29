"""Readable demo: real conversations from the real weights.

Not a test — this exists to show what the model actually says. Three parts:

  1. example patients (expanded by the Observer), multi-turn
  2. stage 1 on a free-text referral note, printing what the Observer invented
  3. a conversation with that expanded patient (the chain: Actor input = Observer output)

    python model_usage/scripts/gpu_demo.py      # on a GPU node
    sbatch model_usage/scripts/gpu_demo.slurm   # Slurm (edit the header first)
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))

BACKEND = os.environ.get("ANGEL_DEMO_BACKEND", "vllm")

INTAKE = [
    "Hi, thanks for coming in today. How have you been feeling lately?",
    "How long has it been going on?",
    "What does a bad day look like for you?",
    "Do you live alone?",
    "Is there anyone you talk to about this?",
    "What would you like to be different a few months from now?",
]

NOTE = """Mara is 34 and came in after her GP suggested it. She says she has been
"running on empty" since a restructure at work in the spring, sleeps badly, and has
stopped seeing the friends she used to climb with. She is polite and a little
dismissive about how much it is affecting her, and changes the subject when her
older brother comes up."""


def wrap(text: str, indent: str = "              ") -> str:
    return textwrap.fill(text, width=96, initial_indent="", subsequent_indent=indent)


def banner(title: str) -> None:
    print(f"\n\n{'─' * 98}\n  {title}\n{'─' * 98}", flush=True)


def converse(model, user: str, label: str, **profile_kwargs) -> None:
    first = True
    for message in INTAKE:
        t0 = time.time()
        result = model.send(user, message, **(profile_kwargs if first else {}))
        took = time.time() - t0
        if first:
            name = result["session"]["profile"].get("name") or "patient"
            print(f"  patient: {name}   ({label})\n")
            first = False
        print(f"  therapist  {wrap(message)}")
        print(f"  {name:<10} {wrap(result['reply'])}")
        print(f"             [{took:.1f}s]\n", flush=True)


def main() -> int:
    from model_usage.angel import AngelModel
    from model_usage.angel.config import RunnerConfig

    config = RunnerConfig()
    config.backend = BACKEND
    print(f"engine: {config.resolved_backend()}")
    print(f"actor : {config.actor.model_path}")
    print(f"observer: {config.observer.model_path}")

    model = AngelModel(config)

    # ---- 1. example patients -------------------------------------------------
    banner("1.  Example patient — Sam, 29 (work anxiety)")
    converse(model, "demo1", "example profile, Observer -> Actor", profile_id="0")

    banner("2.  Example patient — Daniel, 52 (low mood, drinking)")
    converse(model, "demo2", "example profile, Observer -> Actor", profile_id="1")

    # ---- 3. stage 1 on a referral note --------------------------------------
    banner("3.  Stage 1 — Observer expands a free-text referral note")
    print("  input note:")
    print(textwrap.indent(textwrap.fill(" ".join(NOTE.split()), width=90), "    "))
    t0 = time.time()
    expansion = model.expand_profile(NOTE, source_title="GP referral")
    print(f"\n  expanded in {time.time() - t0:.1f}s ({expansion.attempts} attempt)\n")

    rich = expansion.rich_profile
    ident = rich["identity"]
    print(f"  identity   : {ident.get('name')}, {ident.get('age')}, {ident.get('gender')}, "
          f"{ident.get('role')}")
    print(f"  diagnosis  : {', '.join(ident.get('diagnosis_hint') or []) or '(none)'}")
    for field in ("presenting_problems", "triggers", "emotions", "behaviors", "hidden_state"):
        print(f"\n  {field}:")
        for item in rich[field][:6]:
            print(f"    - {wrap(item, '      ')}")
    print("\n  speaking_style:")
    for key, value in rich["speaking_style"].items():
        print(f"    {key:<17} {value}")

    # ---- 4. chat with the expanded patient ---------------------------------
    banner("4.  Conversation with the expanded patient (Actor input = Observer output)")
    converse(model, "demo3", "stage 1 -> stage 2 chain", short_profile=NOTE)

    model.close()
    print("\n\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
