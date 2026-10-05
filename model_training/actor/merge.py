"""Merge a LoRA adapter into its base model and save a standalone checkpoint.

Used twice in the Actor pipeline (see README):

    # after SFT:  Qwen3-8B + SFT LoRA  ->  SFT Actor (the DPO starting point and reference)
    python -m model_training.actor.merge \
        --adapter models/Qwen-3-8B-Patient-SFT-Actor-5 \
        --output models/qwen3-8b-sft-merged

    # after DPO:  SFT Actor + DPO LoRA  ->  final Actor (SFT + DPO)
    python -m model_training.actor.merge \
        --base-model models/qwen3-8b-sft-merged \
        --adapter models/qwen3-8b-dpo-lora \
        --output models/Angel-Actor

The script verifies that the adapter actually changed the weights, so a merge
that silently drops an adapter fails loudly instead of producing a copy of the
base model.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from angel_common.paths import resolve_model, resolve_path


def _probe_weight(model) -> torch.Tensor:
    """A LoRA-targeted weight (first layer q_proj) used to check the merge took effect."""
    return model.get_submodule("model.layers.0.self_attn.q_proj").weight.detach().float().clone()


def merge(base: str, adapter: Path, out: Path, dtype: torch.dtype) -> None:
    if out.exists():
        shutil.rmtree(out)

    print("Loading base model:", base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map="cpu", trust_remote_code=True)
    before = _probe_weight(model)

    print("Loading adapter:", adapter)
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()

    delta = (_probe_weight(merged) - before).norm().item()
    print(f"|ΔW| on layers.0.q_proj after merge: {delta:.6f}")
    if delta == 0.0:
        raise RuntimeError(f"Merging {adapter} did not change the weights; refusing to save {out}.")

    print("Saving merged model to", out)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)

    AutoModelForCausalLM.from_pretrained(out, dtype=dtype, device_map="cpu", trust_remote_code=True)
    print("Reload merged model OK.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=None,
                    help="model the adapter was trained on (default: resolve_model('base') -> Qwen/Qwen3-8B)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory")
    ap.add_argument("--output", required=True, help="output directory for the merged checkpoint")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = ap.parse_args()

    merge(
        resolve_model("base", args.base_model),
        resolve_path(args.adapter),
        Path(args.output),
        getattr(torch, args.dtype),
    )


if __name__ == "__main__":
    main()
