"""Step 5 (graph option) — graph-grounded DPO pairs, as described in the paper.

For each masked network (the conditioning network Ñ = visible + masked edges):

1. The therapist writes one opening message, shared by all conversations.
2. The SFT Actor (Qwen3-8B 4-bit + SFT adapter) holds ``--num-conversations``
   (4) whole conversations with the therapist, ``--max-turns`` (8) patient
   turns each, sampling every reply at ``--temperature`` / ``--top-p``. Each
   reply is normalised to one <state>/<word> pair as in ``rollout_dpo``.
3. The frozen Observer rebuilds a network from each conversation's patient
   utterances (the <word> parts only; the <state> parts are the Actor's
   hidden reasoning and would leak the graph).
4. Each conversation's graph reconstruction score is the similarity of the
   rebuilt network to Ñ (``graph_similarity``).
5. The best and worst conversation become one DPO pair (``--pairs all``: every
   pair), if their scores differ by at least ``--min-gap``.

Train on the pairs with ``train_graph_dpo``, which puts the DPO loss on the
patient turns only.

    python -m model_training.actor.rollout_graph_dpo \
        --input data/actor/masked/NM_mask_p1.jsonl \
        --out-conv data/actor/graph_dpo/conv_p1.jsonl \
        --out-dpo data/actor/graph_dpo/dpo_p1.jsonl

GPU: the Actor (4-bit, ~6 GB) and the Observer (bf16, ~17 GB) are loaded together.
Use --start/--end to shard a file across jobs.

Output DPO rows: {"prompt_messages": [system, therapist opening],
"chosen": [patient, therapist, ..., patient], "rejected": [...], "metadata": {...}}
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from angel_common.llm import get_output
from angel_common.paths import MODELS_DIR, resolve_model, resolve_path
from model_training.actor.graph_similarity import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NODE_WEIGHT,
    DEFAULT_THRESHOLD,
    graph_similarity,
    sentence_embedder,
)
from model_training.actor.patient import AdapterPatient
from model_training.actor.postprocess import postprocess_patient_output
from model_training.actor.prompts import generate_system_prompt
from model_training.actor.rollout_sft import is_masked
from model_training.actor.therapist import AITherapist
from model_training.observer.predict_network import load_observer, predict_network


def _gpt_format_fix(prompt: str) -> str:
    return get_output(prompt, context=None, tag=0, max_completion_tokens=16384, max_retries=2)


def run_conversation(
    patient: AdapterPatient,
    therapist: AITherapist,
    opening: str,
    max_turns: int,
    temperature: float,
    top_p: float,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """One conversation after the shared opening; None if a patient reply cannot be fixed."""
    conversation: List[Dict[str, str]] = [{"role": "user", "content": opening}]  # therapist=user, patient=assistant
    words: List[str] = []

    def _patient_generate(messages, temperature=temperature, max_new_tokens=512, top_p=top_p):
        return patient.generate(messages, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)

    for turn in range(max_turns):
        raw = patient.generate(conversation, max_new_tokens=256, temperature=temperature, top_p=top_p)
        ok, text, state, word, issues = postprocess_patient_output(
            raw_text=raw,
            patient_generate_fn=_patient_generate,
            patient_view_messages=conversation,
            gpt_format_fix_fn=_gpt_format_fix,
            retry_patient_once=True,
        )
        if not ok:
            print(f"[turn {turn + 1}] unfixable patient reply, dropping conversation:", issues)
            return None
        conversation.append({"role": "assistant", "content": text})
        words.append(word)
        if verbose:
            print("Patient:", text)
        if turn == max_turns - 1:
            break
        try:
            reply = therapist.generate(conversation)
        except Exception as e:
            print("[Therapist Error]", type(e).__name__, e)
            return None
        conversation.append({"role": "user", "content": reply})
        if verbose:
            print("Therapist:", reply)
    return {"messages": conversation, "patient_words": words}


def rollout_network(
    item: Dict[str, Any],
    patient: AdapterPatient,
    therapist: AITherapist,
    observer,
    embed,
    *,
    num_conversations: int = 4,
    max_turns: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.9,
    pairs: str = "best-worst",
    min_gap: float = 0.05,
    threshold: float = DEFAULT_THRESHOLD,
    node_weight: float = DEFAULT_NODE_WEIGHT,
    verbose: bool = False,
):
    """Scored conversations and DPO pairs for one masked network."""
    system_prompt = generate_system_prompt(item["new_graph"], item["mask"])
    patient.set_system_prompt(system_prompt)
    opening = therapist.generate([])
    if verbose:
        print("Therapist:", opening)

    obs_tokenizer, obs_model = observer
    scored: List[Dict[str, Any]] = []
    for k in range(num_conversations):
        if verbose:
            print(f"\n--- conversation {k + 1}/{num_conversations} ---")
        conv = run_conversation(patient, therapist, opening, max_turns, temperature, top_p, verbose)
        if conv is None:
            continue
        try:
            rebuilt = predict_network(obs_tokenizer, obs_model, "\n".join(conv["patient_words"]))
        except Exception as e:
            print("[Observer failed, dropping conversation]", type(e).__name__, e)
            continue
        sim = graph_similarity(rebuilt["symptoms"], rebuilt["graph"], item.get("symptoms"),
                               item["new_graph"], item["mask"],
                               embed=embed, threshold=threshold, node_weight=node_weight)
        print(f"[conversation {k + 1}] graph reconstruction score {sim['score']:.3f}")
        scored.append({**conv, "score": sim["score"], "similarity": sim,
                       "reconstructed": {"symptoms": rebuilt["symptoms"], "graph": rebuilt["graph"]}})

    scored.sort(key=lambda c: c["score"], reverse=True)
    if pairs == "best-worst":
        candidates = [(scored[0], scored[-1])] if len(scored) >= 2 else []
    else:
        candidates = list(itertools.combinations(scored, 2))  # sorted, so (higher, lower)

    prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": opening}]
    dpo_rows = []
    for chosen, rejected in candidates:
        gap = chosen["score"] - rejected["score"]
        if gap < min_gap or gap <= 0:
            continue
        dpo_rows.append({
            "prompt_messages": prompt_messages,
            "chosen": chosen["messages"][1:],
            "rejected": rejected["messages"][1:],
            "metadata": {
                "type": "graph_reconstruction",
                "chosen_score": chosen["score"],
                "rejected_score": rejected["score"],
                "score_gap": gap,
                "chosen_similarity": chosen["similarity"],
                "rejected_similarity": rejected["similarity"],
                "num_conversations_scored": len(scored),
            },
        })
    return prompt_messages, scored, dpo_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="masked networks from mask_generator")
    ap.add_argument("--out-conv", required=True, help="scored conversations JSONL (appended)")
    ap.add_argument("--out-dpo", required=True, help="DPO pairs JSONL (appended)")
    ap.add_argument("--base-model", default=None, help="default: resolve_model('base') -> Qwen/Qwen3-8B")
    ap.add_argument("--sft-adapter", default=str(MODELS_DIR / "Qwen-3-8B-Patient-SFT-Actor-5"))
    ap.add_argument("--observer-model", default=None,
                    help="default: resolve_model('observer') ($ANGEL_OBSERVER_MODEL, models/Angel-Observer, or ChengLi0228/Angel-Observer)")
    ap.add_argument("--therapist-deployment", default=None, help="default: $ANGEL_THERAPIST_DEPLOYMENT")
    ap.add_argument("--num-conversations", type=int, default=4, help="conversations per network")
    ap.add_argument("--max-turns", type=int, default=8, help="patient turns per conversation")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--pairs", choices=["best-worst", "all"], default="best-worst")
    ap.add_argument("--min-gap", type=float, default=0.05, help="minimum score gap for a pair")
    ap.add_argument("--node-weight", type=float, default=DEFAULT_NODE_WEIGHT,
                    help="score = w * node F1 + (1 - w) * edge F1")
    ap.add_argument("--match-threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="cosine similarity needed to align two node names")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--start", type=int, default=0, help="first row index (for sharding)")
    ap.add_argument("--end", type=int, default=None, help="stop before this row index")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    patient = AdapterPatient(resolve_model("base", args.base_model), sft_adapter=str(resolve_path(args.sft_adapter)))
    observer = load_observer(resolve_model("observer", args.observer_model))
    therapist = AITherapist(deployment=args.therapist_deployment)
    embed = sentence_embedder(args.embedding_model)

    for p in (args.out_conv, args.out_dpo):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conv_count = dpo_count = skipped = 0
    with open(resolve_path(args.input)) as reader, open(args.out_conv, "a", buffering=1) as w_conv, \
            open(args.out_dpo, "a", buffering=1) as w_dpo:
        for idx, line in enumerate(reader):
            if idx < args.start or (args.end is not None and idx >= args.end) or not line.strip():
                continue
            item = json.loads(line)
            if not is_masked(item):
                print(f"[skip row {idx}] unmasked network")
                skipped += 1
                continue
            print(f"\n=== row {idx} ===")
            try:
                prompt_messages, scored, dpo_rows = rollout_network(
                    item, patient, therapist, observer, embed,
                    num_conversations=args.num_conversations, max_turns=args.max_turns,
                    temperature=args.temperature, top_p=args.top_p, pairs=args.pairs,
                    min_gap=args.min_gap, threshold=args.match_threshold,
                    node_weight=args.node_weight, verbose=args.verbose,
                )
            except Exception as e:
                print(f"[row {idx} failed]", type(e).__name__, e)
                continue
            for c in scored:
                w_conv.write(json.dumps({"row": idx, "messages": c["messages"], "score": c["score"],
                                         "similarity": c["similarity"], "reconstructed": c["reconstructed"]},
                                        ensure_ascii=False) + "\n")
                conv_count += 1
            for row in dpo_rows:
                w_dpo.write(json.dumps({**row, "metadata": {**row["metadata"], "row": idx}},
                                       ensure_ascii=False) + "\n")
                dpo_count += 1
    print(f"wrote {conv_count} scored conversations, {dpo_count} DPO pairs; skipped {skipped} unmasked networks")


if __name__ == "__main__":
    main()
