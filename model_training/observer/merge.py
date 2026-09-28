"""Merge a LoRA adapter into a base model and save a standalone checkpoint.

Paper lineage (see README):

  S1:  python -m model_training.observer.merge --base Qwen/Qwen3-8B \
           --adapter models/Qwen-3-8B-GRPO-600 --output models/Qwen-3-8B-GRPO-600-S1-merged
  S2:  python -m model_training.observer.merge --base models/Qwen-3-8B-Patient-SFT-S2 \
           --adapter models/Qwen-3-8B-GRPO-s2-set2-800-gpt --output models/Qwen3-Observer-800 \
           --device cpu --rebuild-clean

Note the S1 merge onto ``Qwen/Qwen3-8B`` although the S1 GRPO adapter was
trained on top of ``Qwen-3-8B-Patient-SFT`` — this is what the original merge
script did (README, Known issues). Pass ``--base models/Qwen-3-8B-Patient-SFT``
to merge onto the model the adapter was trained against.
"""

from __future__ import annotations

import argparse
import copy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="Base model path or Hub id.")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory.")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto", help="device_map for loading ('auto' or 'cpu').")
    p.add_argument("--rebuild-clean", action="store_true",
                   help="Re-instantiate a fresh model from config and load the merged state dict "
                        "(as done for Qwen3-Observer-800) so no PEFT/quantization wrappers remain.")
    p.add_argument("--tokenizer", default=None, help="Tokenizer source (default: --base).")
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading base:", args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map=args.device)
    print("Loading adapter:", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.config.model_type = "qwen3"
    model.config.architectures = ["Qwen3ForCausalLM"]

    if args.rebuild_clean:
        clean = AutoModelForCausalLM.from_config(copy.deepcopy(model.config), torch_dtype=torch.bfloat16)
        clean.load_state_dict(model.state_dict(), strict=True)
        model = clean

    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.tokenizer or args.base, trust_remote_code=True).save_pretrained(args.output)
    print("Saved merged model to", args.output)


if __name__ == "__main__":
    main()
