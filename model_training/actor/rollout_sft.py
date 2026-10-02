"""Step 2 — SFT rollouts: a prompted patient talks to the therapist.

The patient (``Qwen/Qwen3-30B-A3B-Instruct-2507`` in the paper) is prompted
with ``prompts.generate_system_prompt(new_graph, mask)`` and speaks first; the
Azure therapist replies. One conversation per masked network.

    python -m model_training.actor.rollout_sft \
        --input data/actor/masked/NM_mask_p1.jsonl \
        --output data/actor/sft_rollouts/NM_mask_p1.jsonl

Output rows: {"messages": [{"role": "patient"|"therapist", "content": ...}, ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from angel_common.paths import resolve_path
from model_training.actor.patient import PromptedPatient
from model_training.actor.prompts import generate_system_prompt
from model_training.actor.therapist import AITherapist

DEFAULT_PATIENT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def _should_stop(text: str) -> bool:
    stop_phrases = ["let's summarize", "we can end here", "we will continue next time", "session today"]
    return any(p in (text or "").lower() for p in stop_phrases)


def is_masked(item: dict) -> bool:
    """True if mask_generator tagged every edge; rows it left unmasked hold raw {from, to} edges."""
    edges = (item.get("new_graph") or []) + (item.get("mask") or [])
    return bool(edges) and all(isinstance(e, dict) and "value" in e and "tag" in e for e in edges)


def arena(patient: PromptedPatient, therapist: AITherapist, patient_list, mask_list,
          max_turns: int = 15, verbose: bool = True, early_stop: bool = False,
          paper_prompt: bool = False):
    patient.set_system_prompt(generate_system_prompt(patient_list, mask_list, paper_categories=paper_prompt))
    conversation = []
    for turn in range(max_turns):
        if verbose:
            print(f"\n--- Turn {turn + 1} ---")
        try:
            patient_reply = patient.generate(conversation)
        except Exception as e:
            print("[Patient Error]", e)
            break
        conversation.append({"role": "patient", "content": patient_reply})
        if verbose:
            print("Patient:", patient_reply)

        try:
            therapist_reply = therapist.generate(conversation)
        except Exception as e:
            print("[Therapist Error]", e)
            break
        conversation.append({"role": "therapist", "content": therapist_reply})
        if verbose:
            print("Therapist:", therapist_reply)

        if early_stop and _should_stop(therapist_reply):
            break
    return conversation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="masked networks from mask_generator")
    ap.add_argument("--output", required=True, help="conversations JSONL (appended)")
    ap.add_argument("--patient-model", default=DEFAULT_PATIENT_MODEL)
    ap.add_argument("--therapist-deployment", default=None, help="default: $ANGEL_THERAPIST_DEPLOYMENT")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--start", type=int, default=0, help="first row index (for sharding)")
    ap.add_argument("--end", type=int, default=None, help="stop before this row index")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--on-unmasked", choices=["skip", "stop"], default="skip",
                    help="row left unmasked by mask_generator: 'skip' continues past it; 'stop' ends "
                         "the file there, as the paper's run did (423 conversations)")
    ap.add_argument("--paper-prompt", action="store_true",
                    help="leave \"Physiological Sensation\" out of the patient prompt's states, as the paper's run did")
    args = ap.parse_args()

    patient = PromptedPatient(args.patient_model)
    therapist = AITherapist(deployment=args.therapist_deployment)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(resolve_path(args.input)) as reader, open(out, "a", buffering=1) as f_out:
        for idx, line in enumerate(reader):
            if idx < args.start or (args.end is not None and idx >= args.end) or not line.strip():
                continue
            item = json.loads(line)
            if not is_masked(item):
                # The paper's run ended at the first such row in each file.
                if args.on_unmasked == "stop":
                    print(f"[stop at row {idx}] unmasked network")
                    break
                print(f"[skip row {idx}] unmasked network")
                skipped += 1
                continue
            conversation = arena(patient, therapist, item["new_graph"], item["mask"],
                                 max_turns=args.max_turns, verbose=not args.quiet,
                                 paper_prompt=args.paper_prompt)
            f_out.write(json.dumps({"messages": conversation}, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} conversations, skipped {skipped} unmasked networks")


if __name__ == "__main__":
    main()
