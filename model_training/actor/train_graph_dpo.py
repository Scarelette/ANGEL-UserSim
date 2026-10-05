"""Step 7 (graph option) — DPO on whole conversations from ``rollout_graph_dpo``.

Each pair shares a prompt (patient system prompt + therapist opening) and
differs in everything after it, therapist turns included. The therapist turns
are not the Actor's output, so a sequence's log-probability is summed over the
patient turns only (the same tokens SFT trains on, ``chat_format.tokenize_for_sft``).
The loss is standard sigmoid DPO on those sums:

    loss = -log σ(β [(log π(chosen) - log π_ref(chosen)) - (log π(rejected) - log π_ref(rejected))])

As in ``train_dpo``, the policy is the SFT Actor plus a fresh LoRA and the
reference is the same model with that LoRA disabled, so β is the KL penalty
that keeps the Actor close to the SFT model. Hyperparameters default to
``train_dpo``'s: 4-bit nf4 base, LoRA r=16 / alpha=32 / dropout 0.05, lr 2e-6
(linear decay), β 0.1, 16 pairs per step, 2 epochs, seed 42. Conversations
longer than ``--max-length`` tokens are cut at the end (a warning counts them).

    python -m model_training.actor.train_graph_dpo \
        --sft-model models/qwen3-8b-sft-merged \
        --train-file data/actor/graph_dpo/dpo_training.jsonl \
        --output-dir models/qwen3-8b-graph-dpo-lora

Single GPU. Merge the adapter into the SFT Actor with ``merge`` afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from angel_common.env import get_env
from angel_common.paths import DATA_DIR, MODELS_DIR, resolve_path
from model_training.actor.chat_format import tokenize_for_sft
from model_training.actor.train_dpo import _bnb_config, _dpo_lora

SEED = 42


def encode_pair(tokenizer, row: Dict, max_length: int) -> Dict[str, List[int]]:
    """Token ids and patient-turn masks for the chosen and the rejected conversation."""
    out = {}
    for side in ("chosen", "rejected"):
        enc = tokenize_for_sft(tokenizer, row["prompt_messages"] + row[side])
        out[f"{side}_ids"] = enc["input_ids"][:max_length]
        out[f"{side}_mask"] = enc["assistant_masks"][:max_length]
        out[f"{side}_truncated"] = len(enc["input_ids"]) > max_length
    return out


def sequence_logp(model, ids: List[int], mask: List[int]) -> torch.Tensor:
    """Sum of log p(token) over the tokens where ``mask`` is 1 (the patient turns)."""
    input_ids = torch.tensor([ids], device=model.device)
    # position t predicts token t + 1; keep only positions whose next token is trained
    pos = torch.tensor([t for t in range(len(ids) - 1) if mask[t + 1]], device=model.device)
    logits = model(input_ids=input_ids).logits[0, pos].float()
    target = input_ids[0, pos + 1]
    return torch.gather(F.log_softmax(logits, dim=-1), 1, target.unsqueeze(-1)).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-model", default=str(MODELS_DIR / "qwen3-8b-sft-merged"),
                    help="SFT Actor: Qwen3-8B with the SFT LoRA merged in (output of merge.py)")
    ap.add_argument("--train-file", default=str(DATA_DIR / "actor" / "graph_dpo" / "dpo_training.jsonl"))
    ap.add_argument("--output-dir", default=str(MODELS_DIR / "qwen3-8b-graph-dpo-lora"), help="DPO LoRA adapter")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--pairs-per-step", type=int, default=16, help="pairs accumulated per optimizer step")
    ap.add_argument("--max-length", type=int, default=8192, help="tokens per conversation")
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--report-to", default="none", help="'wandb' to log to Weights & Biases")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    sft_model = str(resolve_path(args.sft_model))
    tokenizer = AutoTokenizer.from_pretrained(sft_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    with open(resolve_path(args.train_file)) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    data = [encode_pair(tokenizer, r, args.max_length) for r in rows]
    n_cut = sum(d["chosen_truncated"] or d["rejected_truncated"] for d in data)
    if n_cut:
        print(f"WARNING: {n_cut}/{len(data)} pairs have a conversation over --max-length {args.max_length}; "
              "their ends are cut")
    data = [d for d in data if sum(d["chosen_mask"]) and sum(d["rejected_mask"])]
    print(f"{len(data)} pairs")
    if not data:
        raise SystemExit("no usable pairs")

    model = AutoModelForCausalLM.from_pretrained(
        sft_model, quantization_config=_bnb_config(), torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, _dpo_lora())
    model.print_trainable_parameters()

    # Reference log-probs: the SFT Actor (LoRA disabled) never changes, so compute them once.
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for i, d in enumerate(data):
            d["ref_chosen"] = sequence_logp(model, d["chosen_ids"], d["chosen_mask"]).item()
            d["ref_rejected"] = sequence_logp(model, d["rejected_ids"], d["rejected_mask"]).item()
            if (i + 1) % 50 == 0:
                print(f"reference log-probs: {i + 1}/{len(data)}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(data) / args.pairs_per_step)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps_per_epoch * args.epochs)

    if args.report_to == "wandb":
        import wandb
        wandb.init(project=get_env("WANDB_PROJECT", "qwen-dpo-training"),
                   name=get_env("WANDB_RUN_NAME", "qwen3-8b-actor-graph-dpo"),
                   config={**vars(args), "sft_model": sft_model, "pairs": len(data)})

    model.train()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        for start in range(0, len(order), args.pairs_per_step):
            batch = order[start:start + args.pairs_per_step]
            stats = {"loss": 0.0, "accuracy": 0.0, "margin": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0}
            for i in batch:
                d = data[i]
                chosen_reward = args.beta * (sequence_logp(model, d["chosen_ids"], d["chosen_mask"]) - d["ref_chosen"])
                rejected_reward = args.beta * (
                    sequence_logp(model, d["rejected_ids"], d["rejected_mask"]) - d["ref_rejected"])
                margin = chosen_reward - rejected_reward
                loss = -F.logsigmoid(margin)
                (loss / len(batch)).backward()
                stats["loss"] += loss.item() / len(batch)
                stats["accuracy"] += float(margin.item() > 0) / len(batch)
                stats["margin"] += margin.item() / len(batch)
                stats["chosen_reward"] += chosen_reward.item() / len(batch)
                stats["rejected_reward"] += rejected_reward.item() / len(batch)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.logging_steps == 0:
                stats.update(epoch=epoch + (start + len(batch)) / len(order), lr=scheduler.get_last_lr()[0])
                print(f"step {step}: " + ", ".join(f"{k} {v:.4g}" for k, v in stats.items()))
                if args.report_to == "wandb":
                    wandb.log(stats, step=step)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved graph-DPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
