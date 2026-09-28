"""Distil the Yes/No edge-plausibility classifier into Qwen3-0.6B (QLoRA SFT).

Used as the reward model for the ``--reward local`` S2 GRPO variant.
Reproduces the original run: 90/5/5 train/val/test split (seed 42), LoRA r=8
alpha=16, lr 2e-4, 3 epochs, bs 4 x grad-acc 4; merged and saved as
``Qwen3-0.6B-Classifier``.

Training text format (as in the original run — note there is no
"### Assistant:" marker, so the loss covers the whole sequence):

    ### System:
    {system}
    {user}
    {answer}<eos>

    python -m model_training.observer.train_edge_classifier --output models/Qwen3-0.6B-Classifier
"""

from __future__ import annotations

import argparse
import gc

from angel_common.paths import DATA_DIR, OUTPUTS_DIR


def _split_messages(example):
    parts = {"system": "", "user": "", "assistant": ""}
    for msg in example["messages"]:
        if msg["role"] in parts:
            parts[msg["role"]] = msg["content"]
    return parts


def evaluate(model, tokenizer, dataset, yes_token: int, no_token: int) -> dict:
    """Accuracy / macro-F1 from the next-token Yes-vs-No logit after the prompt."""
    import torch
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from tqdm import tqdm

    model.eval()
    y_true, y_pred = [], []
    for ex in tqdm(dataset, desc="Evaluating"):
        prompt, answer = ex["prompt"], ex["answer"].strip().lower()
        if answer not in ("yes", "no"):
            continue
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        y_pred.append("yes" if logits[yes_token] > logits[no_token] else "no")
        y_true.append(answer)
    print(classification_report(y_true, y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data", default=str(DATA_DIR / "observer" / "classifier_sft.jsonl"))
    p.add_argument("--output", required=True, help="Directory for the merged classifier.")
    p.add_argument("--checkpoint-dir", default=str(OUTPUTS_DIR / "observer" / "edge_classifier"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1, "Yes/No must be single tokens"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    ds = load_dataset("json", data_files=args.data)["train"].shuffle(seed=42)
    train_test = ds.train_test_split(test_size=0.1)
    val_test = train_test["test"].train_test_split(test_size=0.5)

    def to_text(example):
        m = _split_messages(example)
        return {"text": f"### System:\n{m['system']}\n{m['user']}\n{m['assistant']}{tokenizer.eos_token}\n"}

    def to_eval(example):
        m = _split_messages(example)
        return {"prompt": f"### System:\n{m['system']}\n{m['user']}\n", "answer": m["assistant"]}

    train_dataset = train_test["train"].map(to_text, remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.checkpoint_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=50,
            warmup_ratio=0.03,
            logging_strategy="steps",
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            bf16=True,
            fp16=False,
            group_by_length=True,
            report_to="none",
        ),
        train_dataset=train_dataset,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

    if not args.skip_eval:
        for name, split in (("validation", val_test["train"]), ("test", val_test["test"])):
            metrics = evaluate(trainer.model, tokenizer, split.map(to_eval), yes_ids[0], no_ids[0])
            print(f"{name} metrics:", metrics)

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output)
    print("Saved classifier to", args.output)


if __name__ == "__main__":
    main()
