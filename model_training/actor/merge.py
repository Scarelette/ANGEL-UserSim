"""Step 7 — merge adapters into a standalone Actor checkpoint.

Modes
-----
released (default)
    Qwen3-8B + SFT adapter, saved in float16. This reproduces the released
    ``qwen3-8b-dpo-merged`` bit-for-bit: its weights equal
    fp16(W_base + SFT LoRA delta) with zero DPO component. The original merge
    script loaded ``dpo_trl_final`` via ``load_adapter(..., "dpo")`` but
    ``merge_and_unload()`` only merges the *active* adapter ("default" = SFT),
    so DPO never entered the released weights.
policy
    Qwen3-8B + ``dpo_trl_final`` — the model the DPO run actually optimized
    (its SFT layers were re-initialized, see train_dpo.py). The adapter's
    doubly nested key names are remapped before loading.
stacked
    Qwen3-8B + SFT + DPO, for an adapter trained with ``train_dpo --init
    stack-on-sft`` (``<dpo-adapter>/sft`` and ``<dpo-adapter>/dpo``).

    python -m model_training.actor.merge --mode released \
        --sft-adapter models/Qwen-3-8B-Patient-SFT-Actor-5 \
        --output models/qwen3-8b-dpo-merged
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from angel_common.paths import MODELS_DIR, resolve_model, resolve_path

NESTED_PREFIX = "base_model.model.base_model.model."
PLAIN_PREFIX = "base_model.model."


def _unnest_adapter(adapter_dir: Path, tmp: Path) -> Path:
    """Copy an adapter whose keys carry a doubly nested PeftModel prefix, flattened."""
    from safetensors.torch import load_file, save_file

    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    fixed = {(PLAIN_PREFIX + k[len(NESTED_PREFIX):] if k.startswith(NESTED_PREFIX) else k): v
             for k, v in state.items()}
    save_file(fixed, str(tmp / "adapter_model.safetensors"))
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = cfg.get("base_model_name_or_path") or "Qwen/Qwen3-8B"
    (tmp / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return tmp


def merge(mode: str, base: str, sft: Path, dpo: Path, out: Path, dtype: torch.dtype) -> None:
    if out.exists():
        shutil.rmtree(out)

    print("Loading base model...", base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map="cpu", trust_remote_code=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        if mode == "released":
            model = PeftModel.from_pretrained(model, str(sft))
            merged = model.merge_and_unload()
        elif mode == "policy":
            model = PeftModel.from_pretrained(model, str(_unnest_adapter(dpo, Path(tmpdir))))
            merged = model.merge_and_unload()
        elif mode == "stacked":
            model = PeftModel.from_pretrained(model, str(dpo / "sft"), adapter_name="sft")
            model.load_adapter(str(dpo / "dpo"), adapter_name="dpo")
            merged = model.merge_and_unload(adapter_names=["sft", "dpo"])
        else:
            raise ValueError(mode)

        print("Saving merged model to", out)
        merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)

    AutoModelForCausalLM.from_pretrained(out, dtype=dtype, device_map="cpu", trust_remote_code=True)
    print("Reload merged model OK.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["released", "policy", "stacked"], default="released")
    ap.add_argument("--base-model", default=None, help="default: resolve_model('base') -> Qwen/Qwen3-8B")
    ap.add_argument("--sft-adapter", default=str(MODELS_DIR / "Qwen-3-8B-Patient-SFT-Actor-5"))
    ap.add_argument("--dpo-adapter", default=str(MODELS_DIR / "dpo_trl_final"))
    ap.add_argument("--output", default=str(MODELS_DIR / "qwen3-8b-dpo-merged"))
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16",
                    help="the released checkpoint is float16")
    args = ap.parse_args()

    merge(
        args.mode,
        resolve_model("base", args.base_model),
        resolve_path(args.sft_adapter),
        resolve_path(args.dpo_adapter),
        Path(args.output),
        getattr(torch, args.dtype),
    )


if __name__ == "__main__":
    main()
