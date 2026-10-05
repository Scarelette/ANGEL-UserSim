"""Step 5 — DPO rollouts: best-vs-worst candidate pairs judged by Claude.

Therapist speaks first. Each patient turn, the SFT Actor (Qwen3-8B 4-bit +
SFT adapter) samples ``num_candidates`` (5) replies over 3 random sampling
presets; replies are normalized to one <state>/<word> pair (resample / GPT-5
format fix if needed), near-duplicates are dropped, each survivor is scored by
the Claude judge, and (best, worst) becomes a DPO pair. The best reply
continues the conversation.

    python -m model_training.actor.rollout_dpo \
        --input data/actor/masked/NM_mask_p1.jsonl \
        --out-conv data/actor/dpo/out_conversations_p1.jsonl \
        --out-dpo data/actor/dpo/out_dpo_p1.jsonl

Use --start/--end to shard a file across jobs (the paper sharded by hand).
Output DPO rows: {"context_messages": [...], "chosen": str, "rejected": str, "metadata": {...}}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from angel_common.llm import get_output
from angel_common.paths import MODELS_DIR, resolve_model, resolve_path
from model_training.actor.judge import Deduper, Memory, judge_state_word, total_score, truncate_context
from model_training.actor.patient import AdapterPatient
from model_training.actor.postprocess import postprocess_patient_output
from model_training.actor.prompts import generate_system_prompt
from model_training.actor.therapist import AITherapist

PRESETS = [
    {"temperature": 0.2, "top_p": 0.85},
    {"temperature": 0.4, "top_p": 0.90},
    {"temperature": 0.7, "top_p": 0.92},
    {"temperature": 0.9, "top_p": 0.95},
    {"temperature": 1.1, "top_p": 0.98},
    {"temperature": 0.6, "top_p": 0.85},
]


def _should_stop(text: str) -> bool:
    stop_phrases = ["let's summarize", "we can end here", "we will continue next time", "session today"]
    return any(p in (text or "").lower() for p in stop_phrases)


def _gpt_format_fix(prompt: str) -> str:
    # tag=0: no "<>" gating. The original passed max_completion_tokens=512 but its
    # client hardcoded 16384.
    return get_output(prompt, context=None, tag=0, max_completion_tokens=16384, max_retries=2)


def arena_with_dpo(
    patient: AdapterPatient,
    therapist: AITherapist,
    patient_list: list,
    mask_list: list,
    max_turns: int = 8,
    verbose: bool = True,
    early_stop: bool = False,
    deduper: Optional[Deduper] = None,
    num_candidates: int = 5,
    min_gap: float = 0.0,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    try:
        patient_system_prompt = generate_system_prompt(patient_list or [], mask_list or [])
    except Exception as e:
        print("[generate_system_prompt failed]", type(e).__name__, e)
        return [], []

    patient.set_system_prompt(patient_system_prompt)
    conversation: List[Dict[str, str]] = []  # OpenAI roles: therapist=user, patient=assistant
    dpo_samples: List[Dict[str, Any]] = []
    deduper = deduper or Deduper(max_hamming=4)
    memory = Memory()

    def _patient_generate(messages, temperature=0.8, max_new_tokens=256, top_p=0.9):
        out = patient.generate(messages, temperature=temperature, top_p=top_p,
                               max_new_tokens=max_new_tokens, n=1)
        return out[0] if isinstance(out, list) else out

    for turn in range(max_turns):
        if verbose:
            print(f"\n--- Turn {turn + 1} ---")

        try:
            therapist_reply = therapist.generate(conversation)
        except Exception as e:
            print("[Therapist Error]", type(e).__name__, e)
            break
        conversation.append({"role": "user", "content": therapist_reply})
        if verbose:
            print("Therapist:", therapist_reply)
        if early_stop and _should_stop(therapist_reply):
            break

        patient_view = [{"role": "system", "content": patient_system_prompt}] + conversation
        dpo_context = patient_view[:]
        ctx_text = truncate_context(dpo_context)
        mem_text = memory.summary()

        # 1) candidates: 3 random presets, ceil(num_candidates / 3) samples each
        presets = PRESETS[:]
        random.shuffle(presets)
        presets = presets[:max(2, min(num_candidates, len(presets)))]
        use_cfgs = presets[:min(3, len(presets))]
        per_cfg_n = (num_candidates + len(use_cfgs) - 1) // len(use_cfgs)

        candidates: List[Dict[str, Any]] = []
        for cfg in use_cfgs:
            try:
                raw_list = patient.generate(patient_view, max_new_tokens=256, temperature=cfg["temperature"],
                                            top_p=cfg["top_p"], n=per_cfg_n)
                if isinstance(raw_list, str):
                    raw_list = [raw_list]
            except Exception as e:
                if verbose:
                    print("[Patient Candidate Error]", type(e).__name__, e)
                continue

            for raw in raw_list:
                ok, fixed_text, s, w, issues = postprocess_patient_output(
                    raw_text=raw,
                    patient_generate_fn=lambda msgs, temperature=cfg["temperature"], max_new_tokens=512, _p=cfg["top_p"]:
                        _patient_generate(msgs, temperature=temperature, max_new_tokens=max_new_tokens, top_p=_p),
                    patient_view_messages=patient_view,
                    gpt_format_fix_fn=_gpt_format_fix,
                    retry_patient_once=True,
                )
                if not ok or s is None or w is None:
                    if verbose:
                        print("[Candidate Unfixable]", issues)
                    continue
                if verbose:
                    print("[Patient]:", fixed_text)
                candidates.append({"text": fixed_text, "state": s, "word": w, "cfg": cfg})

        if len(candidates) > num_candidates:
            random.shuffle(candidates)
            candidates = candidates[:num_candidates]

        local_dedup = Deduper(max_hamming=4)
        unique = []
        for c in candidates:
            if not local_dedup.is_duplicate(c["text"]):
                local_dedup.add(c["text"])
                unique.append(c)
        candidates = unique

        if len(candidates) < 2:
            if verbose:
                print("[Too few valid candidates] skipping DPO for this turn")
            if candidates:
                conversation.append({"role": "assistant", "content": candidates[0]["text"]})
                memory.update(candidates[0]["state"], candidates[0]["word"])
            continue

        # 2) judge
        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for c in candidates:
            try:
                j = judge_state_word(conversation_context_text=ctx_text, memory_text=mem_text,
                                     therapist_msg=therapist_reply, state_text=c["state"], word_text=c["word"])
                scored.append((total_score(j.get("scores", {})), c, j))
            except Exception as e:
                if verbose:
                    print("[Judge Error for candidate]", type(e).__name__, e)

        scored.sort(key=lambda x: x[0])
        if len(scored) < 2:
            if verbose:
                print("[Too few judged candidates] skipping DPO for this turn")
            if scored:
                best = scored[-1][1]
                conversation.append({"role": "assistant", "content": best["text"]})
                memory.update(best["state"], best["word"])
            continue

        worst_score, worst_c, worst_j = scored[0]
        best_score, best_c, best_j = scored[-1]
        chosen_msg, rejected_msg = best_c["text"], worst_c["text"]
        score_gap = best_score - worst_score

        # 3) optional minimum separation (off in the paper: min_gap=0)
        if min_gap > 0 and score_gap < min_gap:
            conversation.append({"role": "assistant", "content": chosen_msg})
            memory.update(best_c["state"], best_c["word"])
            continue

        # 4) dedup + write
        key = ctx_text + "\n" + therapist_reply + "\n" + chosen_msg + "\n" + rejected_msg
        if not deduper.is_duplicate(key):
            deduper.add(key)
            dpo_samples.append({
                "context_messages": dpo_context,
                "chosen": chosen_msg,
                "rejected": rejected_msg,
                "metadata": {
                    "type": "best_vs_worst_from_candidates",
                    "num_candidates_raw": len(presets),
                    "num_candidates_valid": len(candidates),
                    "num_candidates_judged": len(scored),
                    "best_score": best_score,
                    "worst_score": worst_score,
                    "score_gap": score_gap,
                    "best_cfg": best_c["cfg"],
                    "worst_cfg": worst_c["cfg"],
                    "best_judge": best_j,
                    "worst_judge": worst_j,
                },
            })

        # 5) continue with the chosen reply
        conversation.append({"role": "assistant", "content": chosen_msg})
        memory.update(best_c["state"], best_c["word"])

    return conversation, dpo_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="masked networks from mask_generator")
    ap.add_argument("--out-conv", required=True, help="conversations JSONL (appended)")
    ap.add_argument("--out-dpo", required=True, help="DPO pairs JSONL (appended)")
    ap.add_argument("--base-model", default=None, help="default: resolve_model('base') -> Qwen/Qwen3-8B")
    ap.add_argument("--sft-adapter", default=str(MODELS_DIR / "Qwen-3-8B-Patient-SFT-Actor-5"))
    ap.add_argument("--therapist-deployment", default=None, help="default: $ANGEL_THERAPIST_DEPLOYMENT")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=0.0)
    ap.add_argument("--dedup-hamming", type=int, default=4)
    ap.add_argument("--start", type=int, default=0, help="first row index (for sharding)")
    ap.add_argument("--end", type=int, default=None, help="stop before this row index")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    patient = AdapterPatient(resolve_model("base", args.base_model), sft_adapter=str(resolve_path(args.sft_adapter)))
    therapist = AITherapist(deployment=args.therapist_deployment)
    deduper = Deduper(max_hamming=args.dedup_hamming)

    for p in (args.out_conv, args.out_dpo):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conv_count = dpo_count = 0
    with open(resolve_path(args.input)) as reader, open(args.out_conv, "a", buffering=1) as w_conv, \
            open(args.out_dpo, "a", buffering=1) as w_dpo:
        for idx, line in enumerate(reader):
            if idx < args.start or (args.end is not None and idx >= args.end) or not line.strip():
                continue
            item = json.loads(line)
            conv, dpo = arena_with_dpo(
                patient, therapist, item["new_graph"], item["mask"],
                max_turns=args.max_turns, verbose=args.verbose, deduper=deduper,
                num_candidates=args.num_candidates, min_gap=args.min_gap,
            )
            if not conv:
                continue
            w_conv.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
            conv_count += 1
            for sample in dpo:
                w_dpo.write(json.dumps(sample, ensure_ascii=False) + "\n")
                dpo_count += 1
    print(f"wrote {conv_count} conversations, {dpo_count} DPO pairs")


if __name__ == "__main__":
    main()
