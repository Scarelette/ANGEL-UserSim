"""Step 6 — DPO on the judged pairs -> ``dpo_trl_final`` (LoRA adapter).

Faithful port of ``actor/dpo/train_dpo_2.py`` @ 652db58, the exact commit
recorded by the W&B run that produced ``models/dpo_trl_final`` (864 steps,
2 epochs over 6,901 pairs, single H200, trl 0.27.1 / transformers 5.0.0 /
peft 0.18.1). Hyperparameters: 4-bit nf4 base, new LoRA r=16 / alpha=32 /
dropout=0.05, lr 2e-6 (linear), beta 0.1 (sigmoid loss), batch 2 x
grad-accum 8, max_length 4096, max_prompt_length 2048, bf16, seed 42.

    python -m model_training.actor.train_dpo \
        --train-file data/actor/dpo/dpo_training.jsonl \
        --output-dir models/dpo_trl_final

IMPORTANT (see README "Known issues"): the original code wraps the SFT
``PeftModel`` in a second ``get_peft_model`` with the same adapter name
("default"), which re-initializes the LoRA layers in place — the SFT weights
are overwritten and the policy starts from plain Qwen3-8B, while the
reference model is base + SFT. This is the default (``--init released``) so
the released adapter can be reproduced. ``--init stack-on-sft`` trains a
separate "dpo" adapter on top of the frozen SFT adapter instead (untested).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import DPOConfig, DPOTrainer

from angel_common.env import get_env
from angel_common.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, resolve_model, resolve_path

SEED = 42
DEVICE = "cuda"


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _dpo_lora() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


class TrainingMonitorCallback(TrainerCallback):
    """Extra W&B logging: grad norm and GPU memory per step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        import wandb
        if logs and wandb.run is not None:
            wandb.log(logs)

    def on_step_end(self, args, state, control, **kwargs):
        import wandb
        if wandb.run is None:
            return
        model = kwargs["model"]
        total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        wandb.log({"grad_norm": total_norm})
        if torch.cuda.is_available():
            wandb.log({"gpu_mem_gb": torch.cuda.memory_allocated() / 1e9})


class KLMonitorCallback(TrainerCallback):
    """Every 20 steps log a crude policy-vs-reference log-prob gap on one example."""

    def __init__(self, tokenizer, ref_model, dataset):
        self.tokenizer, self.ref_model, self.dataset = tokenizer, ref_model, dataset

    def on_step_end(self, args, state, control, **kwargs):
        import wandb
        if state.global_step % 20 != 0 or wandb.run is None:
            return
        model = kwargs["model"]
        sample = self.dataset[0]
        inputs = self.tokenizer(sample["prompt"] + sample["chosen"], return_tensors="pt",
                                truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            policy_lp = torch.nn.functional.log_softmax(model(**inputs).logits, dim=-1)
            ref_lp = torch.nn.functional.log_softmax(self.ref_model(**inputs).logits, dim=-1)
        wandb.log({"approx_kl": (policy_lp - ref_lp).mean().item()})


def build_policy(base_model: str, sft_adapter: str, init: str):
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=_bnb_config(), torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=None,
    ).to(DEVICE)

    if init == "released":
        policy = PeftModel.from_pretrained(base, sft_adapter)
        for p in policy.parameters():
            p.requires_grad = False
        # Re-wrapping re-initializes the "default" LoRA layers (SFT weights lost).
        policy = get_peft_model(policy, _dpo_lora())
    else:  # stack-on-sft
        policy = PeftModel.from_pretrained(base, sft_adapter, adapter_name="sft")
        policy.add_adapter("dpo", _dpo_lora())
        policy.base_model.set_adapter(["sft", "dpo"])
        for name, p in policy.named_parameters():
            p.requires_grad = ".dpo." in name
    policy.print_trainable_parameters()
    policy.config.use_cache = False
    return policy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=None, help="default: resolve_model('base') -> Qwen/Qwen3-8B")
    ap.add_argument("--sft-adapter", default=str(MODELS_DIR / "Qwen-3-8B-Patient-SFT-Actor-5"))
    ap.add_argument("--train-file", default=str(DATA_DIR / "actor" / "dpo" / "dpo_training.jsonl"))
    ap.add_argument("--output-dir", default=str(MODELS_DIR / "dpo_trl_final"), help="final adapter")
    ap.add_argument("--checkpoint-dir", default=str(OUTPUTS_DIR / "actor_dpo"), help="Trainer output_dir")
    ap.add_argument("--init", choices=["released", "stack-on-sft"], default="released")
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument("--no-monitor", action="store_true", help="skip the extra grad-norm / KL W&B callbacks")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    base_model = resolve_model("base", args.base_model)
    sft_adapter = str(resolve_path(args.sft_adapter))

    if args.report_to == "wandb":
        import wandb
        wandb.init(
            project=get_env("WANDB_PROJECT", "qwen-dpo-training"),
            name=get_env("WANDB_RUN_NAME", "qwen3-8b-actor-dpo"),
            config={"model": "Qwen3-8B", "lr": 2e-6, "beta": 0.1, "batch_size": 2, "grad_accum": 8,
                    "epochs": 2, "max_length": 4096, "init": args.init},
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    policy_model = build_policy(base_model, sft_adapter, args.init)

    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=_bnb_config(), torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=None,
    ).to(DEVICE)
    ref_model = PeftModel.from_pretrained(ref_base, sft_adapter)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.config.use_cache = False

    def preprocess(example):
        prompt = tokenizer.apply_chat_template(example["context_messages"], tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": example["chosen"], "rejected": example["rejected"]}

    dataset = load_dataset("json", data_files=str(resolve_path(args.train_file)))["train"]
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=1)
    print("Sample:")
    print(dataset[0]["prompt"][:300])

    dpo_config = DPOConfig(
        output_dir=args.checkpoint_dir,
        learning_rate=2e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        bf16=True,
        beta=0.1,
        max_length=4096,
        max_prompt_length=2048,
        logging_steps=10,
        logging_first_step=True,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=args.report_to,
        seed=SEED,
    )

    trainer = DPOTrainer(model=policy_model, ref_model=ref_model, args=dpo_config, train_dataset=dataset)
    if args.report_to == "wandb" and not args.no_monitor:
        trainer.add_callback(TrainingMonitorCallback())
        trainer.add_callback(KLMonitorCallback(tokenizer, ref_model, dataset))

    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("DPO training finished.")


if __name__ == "__main__":
    main()
