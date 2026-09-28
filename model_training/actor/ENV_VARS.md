# Environment variables — `model_training/actor`

Set these in your shell or in `<repo>/.env` (loaded automatically if
`python-dotenv` is installed). No credential is ever read from source.

| Variable | Used by | Purpose | Default |
|---|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | mask_generator, rollout_dpo (format fix), therapist | Azure OpenAI resource URL | — (required) |
| `AZURE_OPENAI_API_KEY` | same | Azure OpenAI key | — (required) |
| `AZURE_OPENAI_API_VERSION` | same | API version | `2024-12-01-preview` |
| `ANGEL_GPT5_DEPLOYMENT` | mask_generator, rollout_dpo | GPT-5 deployment (pattern classification, node typing, `<state>/<word>` format fix) | `gpt-5` |
| `ANGEL_THERAPIST_DEPLOYMENT` | rollout_sft, rollout_dpo | Therapist chat deployment. The paper used custom Azure deployments `ai_therapist` (SFT rollouts) and `ai_therapist_2` (DPO rollouts); `--therapist-deployment` overrides | — (required) |
| `ANGEL_THERAPIST_AZURE_ENDPOINT` | therapist | Optional: therapist on a different Azure resource than GPT-5 (the paper did this) | falls back to `AZURE_OPENAI_ENDPOINT` |
| `ANGEL_THERAPIST_AZURE_API_KEY` | therapist | Optional key for that resource | falls back to `AZURE_OPENAI_API_KEY` |
| `ANTHROPIC_API_KEY` | rollout_dpo (judge) | Anthropic / Azure AI Foundry key | — (required for DPO rollouts) |
| `ANTHROPIC_BASE_URL` | rollout_dpo (judge) | Set to an Azure AI Foundry `https://<resource>.services.ai.azure.com/anthropic` endpoint to use `AnthropicFoundry` (as in the paper); unset = Anthropic API | unset |
| `ANGEL_JUDGE_DEPLOYMENT` | rollout_dpo (judge) | Claude model / Foundry deployment name | `claude-opus-4-6` |
| `ANGEL_BASE_MODEL` | rollout_dpo, train_sft, train_dpo, merge | Base model path or Hub id (via `angel_common.paths.resolve_model("base")`) | `models/Qwen3-8B` if present, else `Qwen/Qwen3-8B` |
| `ANGEL_DATA_DIR` / `ANGEL_MODELS_DIR` / `ANGEL_OUTPUT_DIR` | all | Relocate `data/`, `models/`, `outputs/` | `<repo>/data`, `<repo>/models`, `<repo>/outputs` |
| `HF_TOKEN` | all HF loads | Only needed for gated/private Hub repos | unset |
| `WANDB_PROJECT` | train_dpo | W&B project | `qwen-dpo-training` |
| `WANDB_RUN_NAME` | train_dpo | W&B run name | `qwen3-8b-actor-dpo` |
| `WANDB_API_KEY` | train_dpo | W&B login (or run `wandb login`; use `--report-to none` to disable) | unset |
