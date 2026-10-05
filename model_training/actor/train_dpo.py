"""DPO on the judged preference pairs, starting from the SFT Actor.

Policy and reference are both the SFT Actor (Qwen3-8B with the SFT LoRA merged
in, see ``merge.py``). A fresh LoRA is trained on top of it with DPO; the
reference is the same model with that LoRA disabled, so the DPO KL anchor is
exactly the SFT model. Merge the result into the SFT Actor to get the final
SFT + DPO Actor.

Hyperparameters (as in the paper): 4-bit nf4 base, LoRA r=16 / alpha=32 /
dropout=0.05 on all attention and MLP projections, lr 2e-6 (linear), beta 0.1
(sigmoid loss), batch 2 x grad-accum 8, 2 epochs, max_length 4096,
max_prompt_length 2048, bf16, seed 42.

    python -m model_training.actor.train_dpo \
        --sft-model models/qwen3-8b-sft-merged \
        --train-file data/actor/dpo/dpo_training.jsonl \
        --output-dir models/qwen3-8b-dpo-lora
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import DPOConfig, DPOTrainer

from angel_common.env import get_env
from angel_common.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, resolve_path
from model_training.actor.chat_format import render

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
    """Every 20 steps log a crude policy-vs-reference (SFT) log-prob gap on one example."""

    def __init__(self, tokenizer, dataset):
        self.tokenizer, self.dataset = tokenizer, dataset

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
            with model.disable_adapter():  # reference = SFT Actor without the DPO LoRA
                ref_lp = torch.nn.functional.log_softmax(model(**inputs).logits, dim=-1)
        wandb.log({"approx_kl": (policy_lp - ref_lp).mean().item()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-model", default=str(MODELS_DIR / "qwen3-8b-sft-merged"),
                    help="SFT Actor: Qwen3-8B with the SFT LoRA merged in (output of merge.py)")
    ap.add_argument("--train-file", default=str(DATA_DIR / "actor" / "dpo" / "dpo_training.jsonl"))
    ap.add_argument("--output-dir", default=str(MODELS_DIR / "qwen3-8b-dpo-lora"), help="final DPO LoRA adapter")
    ap.add_argument("--checkpoint-dir", default=str(OUTPUTS_DIR / "actor_dpo"), help="Trainer output_dir")
    ap.add_argument("--report-to", default="none", help="'wandb' to log to Weights & Biases")
    ap.add_argument("--no-monitor", action="store_true", help="skip the extra grad-norm / KL W&B callbacks")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    sft_model = str(resolve_path(args.sft_model))

    if args.report_to == "wandb":
        import wandb
        wandb.init(
            project=get_env("WANDB_PROJECT", "qwen-dpo-training"),
            name=get_env("WANDB_RUN_NAME", "qwen3-8b-actor-dpo"),
            config={"model": "Qwen3-8B", "lr": 2e-6, "beta": 0.1, "batch_size": 2, "grad_accum": 8,
                    "epochs": 2, "max_length": 4096, "sft_model": sft_model},
        )

    tokenizer = AutoTokenizer.from_pretrained(sft_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy = SFT Actor + trainable DPO LoRA (added by DPOTrainer via peft_config).
    # Reference = the same SFT Actor with the LoRA disabled (ref_model=None).
    model = AutoModelForCausalLM.from_pretrained(
        sft_model, quantization_config=_bnb_config(), torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=None,
    ).to(DEVICE)
    model.config.use_cache = False

    def preprocess(example):
        # same prompt the Actor sees in the rollouts and at inference (thinking off)
        prompt = render(tokenizer, example["context_messages"], add_generation_prompt=True)
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

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_dpo_lora(),
    )
    trainer.model.print_trainable_parameters()
    if args.report_to == "wandb" and not args.no_monitor:
        trainer.add_callback(TrainingMonitorCallback())
        trainer.add_callback(KLMonitorCallback(tokenizer, dataset))

    # Single-GPU script: with several GPUs visible (and no torchrun) the Trainer
    # would wrap the 4-bit model in DataParallel and fail with "found at least
    # two devices, cuda:0 and cuda:1". The Trainer does the same for
    # model-parallel models.
    trainer.args._n_gpu = 1
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("DPO training finished.")


if __name__ == "__main__":
    main()
