# Environment variables — profile-expansion experiment

All credentials are read from the environment (or `<repo>/.env`, loaded
automatically when `python-dotenv` is installed). Nothing is hardcoded.

## Credentials and endpoints

| Variable | Required for | Notes |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | GPT-5 judges (profile alignment, behavior extraction, fixed-attribute extraction); fallback for every role below | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | same | |
| `AZURE_OPENAI_API_VERSION` | optional | default `2024-12-01-preview` |
| `ANGEL_THERAPIST_AZURE_ENDPOINT` / `_API_KEY` / `_API_VERSION` | optional | separate resource for the agenda therapist; falls back to `AZURE_OPENAI_*` |
| `ANGEL_BASELINE_AZURE_ENDPOINT` / `_API_KEY` / `_API_VERSION` | optional | separate resource for the Patient-Psi and Roleplay-doh baselines; falls back to `AZURE_OPENAI_*` |
| `ANTHROPIC_API_KEY` | `fixed_attributes/generate_masked_profile_variants` | |
| `ANTHROPIC_BASE_URL` | optional | set to an Azure AI Foundry `.../anthropic/` endpoint to use Foundry; unset = api.anthropic.com |
| `HF_TOKEN` | optional | only if a Hugging Face model you load is gated (read by `transformers` directly) |

In the paper setup the therapist and baselines ran on one Azure resource and the
GPT-5 judges on another; with a single resource, set only `AZURE_OPENAI_*`.

## Deployment / model names

| Variable | Default (paper) | Role |
|---|---|---|
| `ANGEL_GPT5_DEPLOYMENT` | `gpt-5` | GPT-5 judge deployment (`angel_common.llm.get_output`) |
| `ANGEL_THERAPIST_DEPLOYMENT` | `gpt-4-04-14` | agenda therapist; the paper's deployment alias for **gpt-4.1 (2025-04-14)** |
| `ANGEL_PATIENT_PSI_DEPLOYMENT` | `gpt-4` | Patient-Psi baseline |
| `ANGEL_ROLEPLAY_DOH_DEPLOYMENT` | `gpt-4o-2` | Roleplay-doh baseline; the paper's deployment alias for **gpt-4o** |

Deployment names are chosen per Azure resource — set these to your own names.

## Local model weights

Resolved by `angel_common.paths.resolve_model`: `--flag` → env var →
`<ANGEL_MODELS_DIR>/<default dir>` → default id.

| Variable | Default dir under `models/` | Used by |
|---|---|---|
| `ANGEL_OBSERVER_MODEL` | `Qwen3-Observer-800` | `angel` (stage 1) |
| `ANGEL_ACTOR_MODEL` | `qwen3-8b-dpo-merged` | `angel` (stage 2) and `one_stage` |
| `EEYORE_MODEL` | — (Hub id `liusiyang/eeyore_sft_epoch2_dpo_round2_epoch1_llama3.1_8B`) | `eeyore` baseline |

## Paths

| Variable | Default | Meaning |
|---|---|---|
| `ANGEL_DATA_DIR` | `<repo>/data` | inputs are read from `$ANGEL_DATA_DIR/profile_expansion/` |
| `ANGEL_OUTPUT_DIR` | `<repo>/outputs` | everything is written under `$ANGEL_OUTPUT_DIR/profile_expansion/` |
| `ANGEL_MODELS_DIR` | `<repo>/models` | local checkpoints |
