"""GRPO training for the Observer (stage S1 nodes or S2 edges).

QLoRA (4-bit nf4) on a merged SFT model with TRL's GRPOTrainer. The released
model was trained in 100-step increments, each run resuming the previous LoRA
adapter (``--init-adapter``). Launch with torchrun / accelerate for multi-GPU:

    torchrun --nproc_per_node 4 -m model_training.observer.train_grpo --stage s1 ...

Defaults per stage reproduce the paper runs (4 GPUs):
    S1: lr 5e-6, 4 prompts x 8 generations / device, LoRA r=16 alpha=32
    S2: lr 1e-5, 4 prompts x 4 generations / device, LoRA r=32 alpha=64
        (local-classifier runs used 6 x 6 on 8 GPUs)
"""

from __future__ import annotations

import argparse
import os

from angel_common.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, resolve_model

STAGE_DEFAULTS = {
    "s1": dict(model=str(MODELS_DIR / "Qwen-3-8B-Patient-SFT"), lr=5e-6, num_generations=8, lora_r=16, lora_alpha=32),
    "s2": dict(model=str(MODELS_DIR / "Qwen-3-8B-Patient-SFT-S2"), lr=1e-5, num_generations=4, lora_r=32, lora_alpha=64),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["s1", "s2"], required=True)
    p.add_argument("--model", default=None, help="Merged SFT model to train (default: models/Qwen-3-8B-Patient-SFT[-S2]).")
    p.add_argument("--data", default=str(DATA_DIR / "observer" / "grpo_training.jsonl"),
                   help="JSONL with Complaints + gpt5_nodes (see data/examples/observer/grpo_training.jsonl).")
    p.add_argument("--init-adapter", default=None,
                   help="LoRA adapter to continue training (previous 100-step run). Omit to start a fresh LoRA.")
    p.add_argument("--adapter-out", required=True, help="Where to save the trained LoRA adapter.")
    p.add_argument("--output-dir", default=None, help="Trainer checkpoints (default: outputs/observer/<run-name>).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-project", default=None, help="Enable W&B logging to this project.")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=4, help="per_device_train_batch_size")
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="Use only the first N rows (debugging).")
    p.add_argument("--chat-template", choices=["original", "fixed", "native"], default="original",
                   help="'original' reproduces the released runs (see README, Known issues).")
    # S2 reward
    p.add_argument("--reward", choices=["azure", "local", "format_only"], default="azure",
                   help="S2 edge reward. Ignored for S1.")
    p.add_argument("--judge-max-concurrent", type=int, default=3)
    p.add_argument("--edge-classifier-model", default=None,
                   help="local reward: Yes/No classifier (default: resolve_model('edge_classifier')).")
    p.add_argument("--edge-dump", default=None, help="format_only reward: append proposed edges to this JSONL.")
    args = p.parse_args()

    d = STAGE_DEFAULTS[args.stage]
    args.model = args.model or d["model"]
    args.lr = args.lr if args.lr is not None else d["lr"]
    args.num_generations = args.num_generations or d["num_generations"]
    args.lora_r = args.lora_r or d["lora_r"]
    args.lora_alpha = args.lora_alpha or d["lora_alpha"]
    args.run_name = args.run_name or f"grpo-observer-{args.stage}"
    args.output_dir = args.output_dir or str(OUTPUTS_DIR / "observer" / args.run_name)
    return args


def build_reward(args, match_regex):
    from model_training.observer import rewards

    if args.stage == "s1":
        return rewards.symptom_graph_reward_s1(match_regex)

    if args.reward == "azure":
        from model_training.observer.clients import azure_client_for_role, edge_judge_deployment

        engine = rewards.AzureAsyncRewardEngine(
            client=azure_client_for_role("edge_judge", async_client=True),
            deployment_name=edge_judge_deployment(),
            max_concurrent=args.judge_max_concurrent,
            max_retries=5,
        )
        return rewards.symptom_graph_reward_s2_azure(match_regex, engine)

    if args.reward == "local":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = resolve_model("edge_classifier", args.edge_classifier_model)
        rtok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        rtok.pad_token = rtok.eos_token
        rmodel = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
            trust_remote_code=True,
        )
        rmodel.eval()
        for param in rmodel.parameters():
            param.requires_grad_(False)
        return rewards.symptom_graph_reward_s2_local(match_regex, rmodel, rtok)

    return rewards.symptom_graph_reward_s2_format_only(match_regex, edge_dump_path=args.edge_dump)


def main() -> None:
    args = parse_args()

    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.wandb_project and global_rank == 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
    elif global_rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    from model_training.observer.data import load_grpo_dataset
    from model_training.observer.prompts import (
        build_match_regex,
        build_system_prompt_s1,
        build_system_prompt_s2,
        setup_chat_template,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": local_rank},
    )
    model.config.use_cache = False

    peft_config = None
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    system_prompt = build_system_prompt_s1() if args.stage == "s1" else build_system_prompt_s2()
    setup_chat_template(tokenizer, system_prompt, mode=args.chat_template)
    dataset = load_grpo_dataset(args.data, stage=args.stage, limit=args.limit)

    training_args = GRPOConfig(
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        output_dir=args.output_dir,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name,
        remove_unused_columns=False,        # reward fns need ref_nodes / complaints
        ddp_find_unused_parameters=False,
        dataloader_drop_last=(args.stage == "s2"),
        disable_tqdm=(global_rank != 0),
    )

    # The completion regex requires the "</think>" the model writes itself.
    match_regex = build_match_regex(tokenizer)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[build_reward(args, match_regex)],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    if global_rank == 0:
        trainer.model.print_trainable_parameters()

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(args.adapter_out)
        print(f"Saved adapter to {args.adapter_out}")


if __name__ == "__main__":
    main()
