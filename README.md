# Angel: two-stage simulated mental-health patients

Angel simulates a psychotherapy patient in two stages:

1. **Observer** (`Qwen3-Observer-800`, Qwen3-8B trained with SFT + GRPO) reads
   a short patient description. It builds a symptom network and expands the
   description into a structured **long profile**.
2. **Actor** (`qwen3-8b-dpo-merged`, Qwen3-8B trained with SFT + DPO)
   **role-plays the patient described by the Observer's long profile** across a
   multi-turn therapy conversation. It discloses sensitive material gradually.

```
short profile ──► Observer ──► long profile ──► Actor ──► patient dialogue
```

## Repository layout

| Module | What it contains |
|---|---|
| [`model_usage/`](model_usage/) | Run Angel: short profile → Observer → long profile → Actor. CLI and Python API, no API keys needed. **Start here.** |
| [`model_training/observer/`](model_training/observer/) | Observer training: data building, SFT, GRPO for stage 1 (symptom nodes) and stage 2 (edges), reward models, the edge classifier, merging, and automatic profile evaluation. |
| [`model_training/actor/`](model_training/actor/) | Actor training: masking symptom networks, SFT and DPO rollouts against an LLM therapist, SFT, DPO, merging. |
| [`experiments/profile_expansion/`](experiments/profile_expansion/) | Main paper experiment. Angel vs. Patient-Psi, Roleplay-doh, Eeyore and a one-stage ablation under a fixed intake agenda. Metrics: simulation diversity, behavior diversity, profile alignment; the fixed-attribute experiment; the main results table. |
| [`experiments/safety_exp/`](experiments/safety_exp/) | Safety experiment. Red-team contexts, queries to commercial LLMs, codebook-based LLM judging, main results (mean risk / safety per model). |
| `angel_common/` | Shared code: repo-relative paths, model resolution, environment-based credentials, LLM clients. |
| `data/examples/` | Small **synthetic** examples of every input format. No real patient data. |
| `scripts/check_secrets.py` | Pre-publish scan for keys, tokens and machine-specific paths. |

Each module has its own `README.md` (steps, provenance, known issues),
`ENV_VARS.md` and `requirements-*.txt`.

## Quick start

```bash
git clone <this repo> && cd angel-patient-sim
pip install -r model_usage/requirements-usage.txt

# get the weights (see "Models" below) into models/, or export their paths
export ANGEL_OBSERVER_MODEL=/path/to/Qwen3-Observer-800
export ANGEL_ACTOR_MODEL=/path/to/qwen3-8b-dpo-merged

python -m model_usage.angel chat --short-profile-file model_usage/examples/example_short_profile.txt
# no GPU? try the plumbing with fake text:
python -m model_usage.angel demo --backend stub
```

Run every command from the repository root (`python -m <module>`).

## Models

| Role | Directory name | Size | Base | Hugging Face |
|---|---|---|---|---|
| Observer (stage 1) | `Qwen3-Observer-800` | 16 GB (bf16) | Qwen/Qwen3-8B | *TBA* |
| Actor (stage 2) | `qwen3-8b-dpo-merged` | 16 GB (bf16) | Qwen/Qwen3-8B | *TBA* |
| Edge classifier (optional GRPO reward) | `Qwen3-0.6B-Classifier` | 0.8 GB | Qwen/Qwen3-0.6B | *TBA* |

No code contains an absolute path. A model is found by trying, in order:

1. the `--…-model` command-line flag;
2. the environment variable (`ANGEL_OBSERVER_MODEL`, `ANGEL_ACTOR_MODEL`,
   `ANGEL_EDGE_CLASSIFIER_MODEL`, `ANGEL_BASE_MODEL`, `EEYORE_MODEL`);
3. `models/<directory name>` inside the repo;
4. a Hugging Face Hub id.

`python -m angel_common.paths` shows where each model currently resolves. Once
the weights are on the Hub, set the fallback ids in `angel_common/paths.py`
(`MODEL_REGISTRY`).

## Credentials

No credential is stored in code. Copy `.env.example` to `.env`, which is
gitignored and loaded automatically when `python-dotenv` is installed, or
export the variables in your shell. Which variables a module needs:

- **`model_usage`:** none.
- **Training, data generation and experiments:** an Azure OpenAI resource
  (`AZURE_OPENAI_*` plus deployment names) and/or an Anthropic key
  (`ANTHROPIC_API_KEY`, and optionally `ANTHROPIC_BASE_URL` for Azure AI
  Foundry).
- **`safety_exp`:** additionally OpenRouter and Vertex AI.

Before every push, run:

```bash
python scripts/check_secrets.py          # or install it as .git/hooks/pre-commit
```

## Data

The original training and evaluation data come from published clinical case
reports and generated conversations grounded in them, so **they are not
included**. `data/examples/` has synthetic rows with the exact schemas each
script expects. Each module README says which real files the paper used.

## Citation

*TBA*

## License

*TBA*
