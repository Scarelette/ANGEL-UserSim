"""Supervised fine-tuning of the Observer (S1 or S2) with QLoRA.

    S1: Qwen/Qwen3-8B                   + data/observer/sft_training.jsonl
        (4 GPUs x bs 4 x grad-acc 4, 3 epochs = 264 steps in the paper)
    S2: Qwen-3-8B-GRPO-600-S1-merged    + data/observer/sft_training_s2.jsonl
        (3 epochs = 102 steps)

LoRA r=64, alpha=16, dropout 0.05 on all projection layers; lr 1e-4,
paged_adamw_32bit, warmup_ratio 0.03. Examples are rendered with the Qwen3
chat template, the same format GRPO and inference use (loss on the full
sequence).
Training runs on a 4-bit copy of the base (QLoRA), so only the LoRA adapter is
saved; ``merge`` then applies it to the full-precision base:

    torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s1 --output models/Qwen-3-8B-Patient-SFT-lora
    python -m model_training.observer.merge --base Qwen/Qwen3-8B \
        --adapter models/Qwen-3-8B-Patient-SFT-lora --output models/Qwen-3-8B-Patient-SFT
"""

from __future__ import annotations

import argparse
import gc
import os

from angel_common.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, resolve_model

STAGE_DEFAULTS = {
    "s1": dict(data=DATA_DIR / "observer" / "sft_training.jsonl", base=None),
    "s2": dict(data=DATA_DIR / "observer" / "sft_training_s2.jsonl", base=MODELS_DIR / "Qwen-3-8B-GRPO-600-S1-merged"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["s1", "s2"], required=True)
    p.add_argument("--base-model", default=None,
                   help="S1 default: Qwen/Qwen3-8B (ANGEL_BASE_MODEL). S2 default: models/Qwen-3-8B-GRPO-600-S1-merged.")
    p.add_argument("--data", default=None, help="Chat-format JSONL ({'messages': [...]}).")
    p.add_argument("--output", required=True, help="Directory for the SFT LoRA adapter.")
    p.add_argument("--checkpoint-dir", default=str(OUTPUTS_DIR / "observer" / "sft"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    d = STAGE_DEFAULTS[args.stage]
    args.base_model = args.base_model or (str(d["base"]) if d["base"] else resolve_model("base"))
    args.data = args.data or str(d["data"])
    return args


def main() -> None:
    args = parse_args()

    import torch
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
    )
    from trl import SFTConfig, SFTTrainer

    from model_training.observer.data import load_sft_text_dataset

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": local_rank},
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

    # SFTConfig, not transformers.TrainingArguments: TRL 0.27.1 converts the
    # latter to an SFTConfig and crashes with transformers 5 (KeyError
    # 'push_to_hub_token'). Same fields, so the resulting config is identical.
    training_arguments = SFTConfig(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        logging_steps=10,
        warmup_ratio=0.03,
        logging_strategy="steps",
        learning_rate=args.lr,
        bf16=True,
        fp16=False,
        group_by_length=True,
        report_to="none",
    )

    dataset = load_sft_text_dataset(args.data, tokenizer, limit=args.limit)
    print(dataset[0]["text"][:500])
    print(f"{len(dataset)} training examples")

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

    if trainer.is_world_process_zero():
        # Adapter only: merging into the 4-bit training copy would give a 4-bit model.
        trainer.model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"Saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
