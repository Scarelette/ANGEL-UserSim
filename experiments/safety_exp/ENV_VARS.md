# experiments/safety_exp — environment variables

All credentials are read from the environment (or `<repo>/.env`, which is
gitignored). Only the variables for the providers you actually call are needed.

| Variable | Used by | Needed when |
|---|---|---|
| `ANTHROPIC_API_KEY` | `generate_redteam_transcript`, `codebook_llm_judge`, `query_models` | always for red-team generation and judging; `--model-key "claude 4.5 opus"` |
| `ANTHROPIC_BASE_URL` | same | optional. An Azure AI Foundry `https://<resource>.services.ai.azure.com/anthropic/` URL switches to the Foundry client (the paper's `auto_attack` runs used this). Unset means the public Anthropic API. |
| `ANTHROPIC_DEPLOYMENT` | `generate_redteam_transcript` | optional default for `--auto-attack-model` (else `claude-opus-4-1`) |
| `AZURE_OPENAI_ENDPOINT` | `query_models` | `--model-key "gpt 4o"` / `"gpt 5.2 chat"` |
| `AZURE_OPENAI_API_KEY` | `query_models` | same |
| `AZURE_OPENAI_API_VERSION` | `query_models` | optional (default `2024-12-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT_GPT_4O` | `query_models` | optional deployment name for `gpt 4o` (else `AZURE_OPENAI_DEPLOYMENT`, else `gpt-4o`) |
| `AZURE_OPENAI_DEPLOYMENT_GPT_5_2_CHAT` | `query_models` | optional deployment name for `gpt 5.2 chat` (else `AZURE_OPENAI_DEPLOYMENT`, else `gpt-5`) |
| `AZURE_OPENAI_DEPLOYMENT` | `query_models` | optional fallback deployment name |
| `OPENROUTER_API_KEY` | `query_models` | `--model-key "grok 4.1 fast"` |
| `GOOGLE_CLOUD_PROJECT` | `query_models` | `--model-key "gemini-3-pro"` / `"gemini-2.5-flash"` (Vertex AI) |
| `GOOGLE_CLOUD_LOCATION` | `query_models` | optional (default `us-central1`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google SDK | path to a service-account JSON **stored outside the repo**, or use `gcloud auth application-default login` instead |
| `ANGEL_JSONL_PATH`, `ANGEL_BACKEND`, `ANGEL_ACTOR_MODEL`, `ANGEL_OBSERVER_MODEL` | `--patient-backend angel` | see `model_usage/` |
| `ANGEL_DATA_DIR`, `ANGEL_OUTPUT_DIR` | all | optional; relocate `data/` and `outputs/` (see `angel_common/paths.py`) |
