# Environment variables — `model_training/observer`

Set them in your shell or in `<repo>/.env` (loaded by `angel_common.env`). No
credential is stored in code.

| Variable | Used by | Purpose | Default |
|---|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | build_data, GRPO azure reward, edge classifier, eval | Azure OpenAI resource URL (shared fallback for every role below) | — (required when an Azure call is made) |
| `AZURE_OPENAI_API_KEY` | same | Azure OpenAI key | — (required) |
| `AZURE_OPENAI_API_VERSION` | same | API version | `2024-12-01-preview` |
| `ANGEL_GPT5_DEPLOYMENT` | build_data, eval/short2long_profile_generation, auto pipeline | GPT-5 deployment (data generation; stage-2 long-profile prose) | `gpt-5` (falls back to `AZURE_OPENAI_DEPLOYMENT` if set) |
| `ANGEL_EDGE_JUDGE_DEPLOYMENT` | train_grpo `--reward azure` | S2 GRPO edge-plausibility judge | `gpt-5-mini` |
| `ANGEL_EDGE_JUDGE_ENDPOINT` / `_API_KEY` / `_API_VERSION` | same | Separate Azure resource for the judge | fall back to `AZURE_OPENAI_*` |
| `ANGEL_EDGE_CLASSIFIER_DEPLOYMENT` | edge_classifier_data `label`, eval/score_network_edges | Your Azure **fine-tuned** Yes/No edge classifier (private fine-tune) | — (required; no public default) |
| `ANGEL_EDGE_CLASSIFIER_ENDPOINT` / `_API_KEY` / `_API_VERSION` | same | Separate Azure resource for the classifier | fall back to `AZURE_OPENAI_*` |
| `ANTHROPIC_API_KEY` | eval/short2long (`--stage1-model claude-*`) | Anthropic key | — (required for Claude) |
| `ANTHROPIC_BASE_URL` | same | Optional; an Azure AI Foundry `.../anthropic/` URL selects the Foundry client | unset (direct Anthropic API) |
| `GOOGLE_CLOUD_PROJECT` | eval/short2long (`--stage1-model gemini*`) | Vertex AI project | — (required for Gemini) |
| `GOOGLE_CLOUD_LOCATION` | same | Vertex AI region | `us-central1` |
| `GOOGLE_APPLICATION_CREDENTIALS` | same (read by `google-genai`) | Path to a service-account JSON **outside the repo** | — |
| `ANGEL_OBSERVER_MODEL` | predict_network, eval | Observer checkpoint path / Hub id | `models/Qwen3-Observer-800` |
| `ANGEL_BASE_MODEL` | sft `--stage s1` | Base model | `Qwen/Qwen3-8B` |
| `ANGEL_EDGE_CLASSIFIER_MODEL` | train_grpo `--reward local` | Local Qwen3-0.6B classifier | `models/Qwen3-0.6B-Classifier` |
| `ANGEL_DATA_DIR` / `ANGEL_MODELS_DIR` / `ANGEL_OUTPUT_DIR` | all (via `angel_common.paths`) | Relocate data / models / outputs | `<repo>/data`, `<repo>/models`, `<repo>/outputs` |
| `WANDB_API_KEY` (+ other `WANDB_*`) | train_grpo `--wandb-project` | Weights & Biases logging | logging off unless `--wandb-project` |
| `HF_TOKEN` | any Hub download | Only needed for gated/private Hub models | unset |
| `LOCAL_RANK`, `RANK` | sft, train_grpo | Set by `torchrun` / `accelerate` | `0` |
