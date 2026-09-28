"""Repository-relative paths and model resolution.

Nothing in this repository hardcodes an absolute path. Every model or data
location is resolved through this module, in this order:

    explicit argument  ->  environment variable  ->  <repo>/models/<default dir>
                       ->  default (a Hugging Face Hub id, or the bare name)

Relative paths given by argument or env var are resolved against the current
working directory first and the repository root second, so both
``--model models/foo`` and ``ANGEL_ACTOR_MODEL=/abs/path`` work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from angel_common.env import load_env

load_env()  # so ANGEL_*_DIR / ANGEL_*_MODEL set in .env apply below

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("ANGEL_DATA_DIR") or REPO_ROOT / "data")
MODELS_DIR = Path(os.environ.get("ANGEL_MODELS_DIR") or REPO_ROOT / "models")
OUTPUTS_DIR = Path(os.environ.get("ANGEL_OUTPUT_DIR") or REPO_ROOT / "outputs")

# key -> (env var, default local dir name under models/, fallback id)
# The fallback is used when nothing is found locally; for released checkpoints,
# set it to the Hugging Face Hub id once the weights are uploaded.
MODEL_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    # Stage 1: GRPO-trained Qwen3-8B, short profile -> long profile / symptom network.
    "observer": ("ANGEL_OBSERVER_MODEL", "Qwen3-Observer-800", "Qwen3-Observer-800"),
    # Stage 2: SFT + DPO Qwen3-8B patient role-play model.
    "actor": ("ANGEL_ACTOR_MODEL", "qwen3-8b-dpo-merged", "qwen3-8b-dpo-merged"),
    # Qwen3-0.6B Yes/No edge-plausibility classifier (optional local GRPO reward).
    "edge_classifier": ("ANGEL_EDGE_CLASSIFIER_MODEL", "Qwen3-0.6B-Classifier", "Qwen3-0.6B-Classifier"),
    # Public base / baseline models.
    "base": ("ANGEL_BASE_MODEL", "Qwen3-8B", "Qwen/Qwen3-8B"),
    "eeyore": (
        "EEYORE_MODEL",
        "eeyore",
        "liusiyang/eeyore_sft_epoch2_dpo_round2_epoch1_llama3.1_8B",
    ),
}


def resolve_path(path_like: str | os.PathLike) -> Path:
    """Resolve a user-supplied path against cwd, then the repo root."""
    p = Path(path_like).expanduser()
    if p.is_absolute() or p.exists():
        return p
    repo_relative = REPO_ROOT / p
    return repo_relative if repo_relative.exists() else p


def resolve_model(key: str, override: Optional[str] = None) -> str:
    """Return a local path or Hub id for a registered model key."""
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key {key!r}. Known: {sorted(MODEL_REGISTRY)}")
    env_var, local_name, fallback = MODEL_REGISTRY[key]
    for candidate in (override, os.environ.get(env_var)):
        if candidate:
            p = resolve_path(candidate)
            return str(p) if p.exists() else candidate
    local = MODELS_DIR / local_name
    if local.exists():
        return str(local)
    return fallback


def describe_models() -> str:
    """Human-readable table of where each model currently resolves."""
    rows = []
    for key, (env_var, _, _) in MODEL_REGISTRY.items():
        target = resolve_model(key)
        where = "local" if Path(target).exists() else "hub/unresolved"
        rows.append(f"{key:16s} {env_var:30s} {where:15s} {target}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe_models())
