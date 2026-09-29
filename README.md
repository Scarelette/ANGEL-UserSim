# Angel: simulated mental-health patients

Angel simulates a psychotherapy patient in two steps:

1. The **Observer** (`Qwen3-Observer-800`, Qwen3-8B trained with SFT + GRPO)
   reads a short patient description and writes a detailed **long profile**,
   organised around the patient's symptom network.
2. The **Actor** (`qwen3-8b-dpo-merged`, Qwen3-8B trained with SFT + DPO)
   **role-plays the patient in that long profile** across a multi-turn therapy
   conversation, revealing sensitive material gradually.

```
short description ──► Observer ──► long profile ──► Actor ──► patient replies
```

## Quick start: talk to a patient

You need a GPU (40 GB or more) and the two models (see [Models](#models)).

```bash
git clone <this repo> && cd angel-patient-sim
pip install -r model_usage/requirements-usage.txt

export ANGEL_OBSERVER_MODEL=/path/to/Qwen3-Observer-800     # or put both models in models/
export ANGEL_ACTOR_MODEL=/path/to/qwen3-8b-dpo-merged

python -m model_usage.angel chat --short-profile-file model_usage/examples/example_short_profile.txt
```

You type the therapist's messages and the patient answers. See
[`model_usage/README.md`](model_usage/README.md) for writing your own patient,
the Python API and troubleshooting. No GPU yet? Add `--backend stub` to try the
commands with canned replies.

## What's in the repository

| Folder | What it's for |
|---|---|
| [`model_usage/`](model_usage/) | **Use Angel**: short description → Observer → long profile → Actor, as a chat or from Python. No API keys needed. |
| [`model_training/observer/`](model_training/observer/) | Train the Observer: data building, SFT, and GRPO for symptom nodes (stage 1) and edges (stage 2, rewarded by a gpt-5-mini judge). |
| [`model_training/actor/`](model_training/actor/) | Train the Actor: masked symptom networks, conversations with an LLM therapist, SFT, then DPO on top of the SFT model. |
| [`experiments/profile_expansion/`](experiments/profile_expansion/) | The main experiment: Angel vs. Patient-Psi, Roleplay-doh, Eeyore and a one-stage ablation in a fixed intake interview. Measures simulation diversity, behavior diversity and profile alignment. |
| [`experiments/safety_exp/`](experiments/safety_exp/) | The safety experiment: how commercial chatbots respond to a user (played by Angel) whose beliefs escalate toward delusion, scored with a clinical codebook. |
| `angel_common/` | Shared code: model lookup, settings from `.env`, API clients. |
| `data/examples/` | Small **synthetic** examples of every input file. No real patient data. |
| `scripts/check_secrets.py` | Checks the repository for keys and machine-specific paths before you publish. |

Each folder has its own `README.md` with the exact commands and the settings it
needs, and its own `requirements-*.txt`. Run every command from the repository
root.

## Models

| Model | Folder name | Size | Base | Hugging Face |
|---|---|---|---|---|
| Observer | `Qwen3-Observer-800` | 16 GB (bf16) | Qwen/Qwen3-8B | *TBA* |
| Actor | `qwen3-8b-dpo-merged` | 16 GB (bf16) | Qwen/Qwen3-8B | *TBA* |

Angel looks for each model in this order:
1. the command-line flag (`--observer-model`, `--actor-model`);
2. the environment variable (`ANGEL_OBSERVER_MODEL`, `ANGEL_ACTOR_MODEL`);
3. the `models/<folder name>` folder in the repository;
4. the Hugging Face id.

`python -m angel_common.paths` shows where each model is found. After uploading
the weights to Hugging Face, put their ids in `MODEL_REGISTRY` in
`angel_common/paths.py`.

## Settings and API keys

Nothing secret is stored in the code. All settings live in **one file**,
`.env`, at the repository root:

```bash
cp .env.example .env     # fill in what you need; every script reads it
```

`.env` is gitignored. A variable exported in your shell overrides it. What each
part needs:
- **`model_usage`:** nothing (only the model paths, if they are not in `models/`).
- **Training and the main experiment:** an Azure OpenAI resource (GPT-5,
  gpt-5-mini, GPT-4.1, GPT-4, GPT-4o deployments) and, for Actor training, an Anthropic
  key.
- **Safety experiment:** Anthropic, plus Azure OpenAI, Vertex AI or OpenRouter
  for the chatbots you test.

Before every push:

```bash
python scripts/check_secrets.py          # or install it as .git/hooks/pre-commit
```

## Data

The paper's training and evaluation data come from published clinical case
reports, or were generated from them, so **they are not included**.
`data/examples/` has synthetic rows in exactly the format each script expects.

## Citation

*TBA*

## License

*TBA*
