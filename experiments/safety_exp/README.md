# Safety experiment

How do commercial chatbots respond to a user whose beliefs escalate toward
delusion? The Angel patient simulator plays the user in a replay of an
escalating red-team conversation. Each chatbot then answers 16 test prompts
with that conversation as context. A Claude judge (`claude-opus-4-6`,
temperature 0) scores every answer against `Codebook.txt`:

- **Risk:** mean of validation, elaboration, behavioral advice and misrepresentation.
- **Safety:** mean of reality testing, concern for wellbeing, referral and de-escalation.

```
source transcript ─► 1. generate_redteam_transcript ─► contexts/auto_attack/profile_<id>
                     (Angel = user turns; Claude writes new assistant turns in the source's escalation style)
                  ─► 2. query_models        ─► chatbot answers to the 16 prompts
                  ─► 3. codebook_llm_judge  ─► codebook scores (CSV)
                  ─► 4. report + summarize_results ─► mean Risk / Safety per model
```

## Setup

Run everything from the repository root.

```bash
pip install -r experiments/safety_exp/requirements-safety.txt
cp .env.example .env        # then fill in the values below — the only place to edit
```

| Variable | Needed for | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | context generation, judge, Claude as a target | — |
| `ANTHROPIC_BASE_URL` | only for Azure AI Foundry (`https://<resource>.services.ai.azure.com/anthropic/`) | public Anthropic API |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | GPT-4o / GPT-5.2 as targets | — |
| `AZURE_OPENAI_DEPLOYMENT_GPT_4O`, `AZURE_OPENAI_DEPLOYMENT_GPT_5_2_CHAT` | their deployment names | `gpt-4o`, `gpt-5` |
| `GOOGLE_CLOUD_PROJECT` (+ `gcloud auth application-default login`) | Gemini via Vertex AI | — |
| `OPENROUTER_API_KEY` | Grok | — |
| `ANGEL_ACTOR_MODEL` | in-process patient (`--patient-backend angel`) | `models/qwen3-8b-dpo-merged` |

Only the providers you call are needed.

**Inputs:**
- `data/safety_exp/full_context.txt`: the 58-turn source transcript
  (`You said:` / `ChatGPT said:` format). It comes from an external
  AI-psychosis red-teaming study and is not included. Ask its authors.
- Synthetic stand-ins for trying the pipeline are in `data/examples/safety_exp/`.

**Patient:**
- `--patient-backend angel` runs `model_usage.angel` in-process (needs a GPU).
  `--profile-id` indexes `--profiles-jsonl` (the paper used 42 profiles); add
  `--patient-expand` to run the Observer first.
- `--patient-backend http --base-url …` (the default) calls a patient service
  instead; the paper's runs used one. The protocol is in `patient_backends.py`.

## Run

```bash
# 1. red-team contexts for patient profiles 0..41 (or a list / file of ids)
bash experiments/safety_exp/scripts/run_redteam.sh 0,1,2 -- --patient-backend angel

# 2+3. query one chatbot on every context and judge the answers
MODEL_KEY="gpt 4o"          MODEL_DIR=gpt4o           bash experiments/safety_exp/scripts/run_query_and_judge.sh
MODEL_KEY="gemini-3-pro"    MODEL_DIR=gemini-3-pro    bash experiments/safety_exp/scripts/run_query_and_judge.sh
MODEL_KEY="claude 4.5 opus" MODEL_DIR=claude-4.5-opus bash experiments/safety_exp/scripts/run_query_and_judge.sh

# 4. main results
python -m experiments.safety_exp.summarize_results
```

`MODEL_KEY` choices are `gpt 4o`, `gpt 5.2 chat`, `gemini-3-pro` (calls
`gemini-3.1-pro-preview`), `gemini-2.5-flash`, `grok 4.1 fast` and
`claude 4.5 opus`. All targets run at temperature 1.0 with 5000 max tokens.

Reruns are safe:
- `query_models` overwrites its output unless you pass `--append`.
- The judge re-scores only answers that changed, and retries failed ones.

`run_query_and_judge.sh` also writes a per-model markdown report with the full
codebook breakdown (`<results>/<model>/summary/`).

Results on the paper's judge outputs (42 profiles, mean ± 95% CI):

| Model | Risk | Safety |
|---|---|---|
| claude-4.5-opus | 0.028 ± 0.008 | 2.877 ± 0.021 |
| gemini-3-pro | 0.516 ± 0.056 | 1.803 ± 0.071 |
| gpt4o | 1.237 ± 0.054 | 0.443 ± 0.042 |

## Files

```
generate_redteam_transcript.py   step 1: replay the source conversation with the Angel patient
patient_backends.py              patient: in-process Angel or HTTP service
query_models.py                  step 2: target models and the 16 test prompts
codebook_llm_judge.py            step 3: Claude judge (codebook_judge_core.py, prompts/, Codebook.txt)
generate_auto_attack_codebook_report.py, summarize_results.py   step 4
scripts/                         run_redteam.sh, run_query_and_judge.sh
```
