"""Step 4 — QLoRA SFT of Qwen3-8B on the rollout data -> ``Qwen-3-8B-Patient-SFT-Actor-5``.

Hyperparameters are those of the released adapter (``GRPO-Qwen3/Finetune/
finetune_hf.py`` @ 4ab9b7e; r/alpha/dropout/targets match the adapter_config
of the released adapter): 4-bit nf4 base, LoRA r=64 / alpha=16 / dropout=0.05
on all attention+MLP projections, lr 1e-4, 5 epochs, batch 4 x grad-accum 4,
paged_adamw_32bit, warmup 3%, bf16.

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
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

from angel_common.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, resolve_model, resolve_path


def format_messages(example, eos_token: str):
    system = user = assistant = ""
    for msg in example["messages"]:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            user = msg["content"]
        elif msg["role"] == "assistant":
            assistant = msg["content"]
    # NOTE: keeps only the LAST user/assistant turn of each conversation.
    text = f"""### System:
{system}

### User:
{user}

### Assistant:
{assistant}{eos_token}
"""
    return {"text": text}


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

    training_arguments = TrainingArguments(
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
    )

    dataset = load_dataset("json", data_files=str(resolve_path(args.train_file)), split="train")
    # The original keeps the "messages" column next to the new "text" column.
    # With TRL >= 0.20 a "messages" column makes SFTTrainer treat the data as
    # conversational (chat template over all turns); see README "Known issues".
    dataset = dataset.map(
        lambda ex: format_messages(ex, tokenizer.eos_token),
        remove_columns=[c for c in dataset.column_names if c != "messages"],
    )
    print(dataset[0])
    print(len(dataset))

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
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(args.output_dir)  # adapter only, not merged
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved SFT adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
