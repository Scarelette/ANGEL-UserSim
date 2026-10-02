"""Step 4 — QLoRA SFT of Qwen3-8B on the rollout data -> ``Qwen-3-8B-Patient-SFT-Actor-5``.

Hyperparameters as in the paper: 4-bit nf4 base, LoRA r=64 / alpha=16 /
dropout=0.05 on all attention+MLP projections, lr 1e-4, 5 epochs, batch 4 x
grad-accum 4, paged_adamw_32bit, warmup 3%, bf16.

Each example is a whole conversation in the Qwen3 chat format the Actor uses
everywhere (``chat_format``). The loss covers the patient turns only.

    python -m model_training.actor.train_sft \
        --train-file data/actor/sft_training.jsonl \
        --output-dir models/Qwen-3-8B-Patient-SFT-Actor-5
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from angel_common.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, resolve_model, resolve_path
from model_training.actor.chat_format import tokenize_for_sft


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=None, help="default: resolve_model('base') -> Qwen/Qwen3-8B")
    ap.add_argument("--train-file", default=str(DATA_DIR / "actor" / "sft_training.jsonl"))
    ap.add_argument("--output-dir", default=str(MODELS_DIR / "Qwen-3-8B-Patient-SFT-Actor-5"),
                    help="where the LoRA adapter is saved")
    ap.add_argument("--checkpoint-dir", default=str(OUTPUTS_DIR / "actor_sft"),
                    help="Trainer output_dir (intermediate checkpoints)")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=8192,
                    help="tokens per conversation; longer ones are truncated (a warning counts them)")
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    model_dir = resolve_model("base", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_arguments = SFTConfig(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        logging_steps=10,
        warmup_ratio=0.03,
        logging_strategy="steps",
        learning_rate=args.lr,
        bf16=True,
        fp16=False,
        group_by_length=True,
        report_to=args.report_to,
        max_length=args.max_length,
    )

    dataset = load_dataset("json", data_files=str(resolve_path(args.train_file)), split="train")
    dataset = dataset.map(lambda ex: tokenize_for_sft(tokenizer, ex["messages"]),
                          remove_columns=dataset.column_names)
    n_long = sum(len(ids) > args.max_length for ids in dataset["input_ids"])
    if n_long:
        print(f"WARNING: {n_long}/{len(dataset)} conversations exceed --max-length {args.max_length} "
              "and will be truncated")
    print(f"{len(dataset)} conversations")

    # Pre-tokenized input_ids + assistant_masks: SFTTrainer only truncates and
    # sets labels outside the patient turns to -100.
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

    if trainer.is_world_process_zero():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(args.output_dir)  # adapter only, not merged
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved SFT adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
