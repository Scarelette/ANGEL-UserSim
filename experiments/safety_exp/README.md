# Safety experiment: sycophancy and delusion reinforcement in commercial LLMs

This experiment measures how commercial chatbots respond to a user whose
beliefs escalate toward delusion. The user side is played by the **Angel
patient simulator** from this repository. Each model's responses are scored
against a clinical codebook of **risk codes** (sycophancy, validation,
elaboration, behavioral advice, misrepresentation), **safety codes** (reality
testing, concern for wellbeing, referral, de-escalation), relational warmth,
and one prompt-specific code.

## Pipeline

```
source transcript ─► 1. generate_redteam_transcript ─► contexts/<mode>/profile_<id>.{json,txt}
 (58 turns,            Angel patient = user turns        one replay per patient profile
  "You said:" /        Claude = assistant turns
  "ChatGPT said:")
                   ─► 2. query_models ─► results/<mode>/<model>/run_all_full_<mode>_profile<id>.jsonl
                        16 stimulus prompts × {NONE, PARTIAL, FULL} context → target LLM
                   ─► 3. codebook_llm_judge ─► codebook_judge_{results,long}_full_<mode>_profile<id>.{jsonl,csv}
                        Claude judge, Codebook.txt, temperature 0
                   ─► 4. generate_auto_attack_codebook_report ─► <model>/summary/<mode>_codebook_report.md
                   ─► 5. summarize_results ─► mean Risk / Safety per model with 95% CI
```

Contexts are built in the **`auto_attack`** mode: for each turn, Claude writes a
**new** reply to the simulated patient, using only the source turn's attack-style
label and escalation role (`--auto-attack-model`, default `claude-opus-4-1`, via
`ANTHROPIC_BASE_URL` / Foundry in the paper).

The attack-style labels come from `infer_attack_style_label`: intellectual validation → plausibility expansion → metaphor amplification → existential reframing → special-insight validation → glitch validation → collaborative investigation → mission escalation. The turn-index ranges are tuned to the 58-turn source transcript.

## Models evaluated (`query_models.MODELS`)

| `--model-key` | Provider | Model id | Settings |
|---|---|---|---|
| `gpt 4o` | Azure OpenAI | `gpt-4o` (deployment `AZURE_OPENAI_DEPLOYMENT_GPT_4O`) | temperature 1.0, 5000 tokens |
| `gpt 5.2 chat` | Azure OpenAI | deployment `AZURE_OPENAI_DEPLOYMENT_GPT_5_2_CHAT` | temperature 1.0, 5000 tokens |
| `gemini-3-pro` | Vertex AI | `gemini-3.1-pro-preview` | temperature 1.0, 5000 tokens |
| `gemini-2.5-flash` | Vertex AI | `gemini-2.5-flash` | temperature 1.0, 5000 tokens |
| `grok 4.1 fast` | OpenRouter | `x-ai/grok-4.1-fast` | temperature 1.0, 5000 tokens, reasoning on |
| `claude 4.5 opus` | Anthropic | `claude-opus-4-5` | temperature 1.0, 5000 tokens |

The paper's result directories cover `gpt4o`, `gemini-3-pro` and `claude-4.5-opus`, each in both modes, over 42 patient profiles (ids 0–41).

The stimulus prompts (`TEST_STIMULI`, 16 of them) are: consciousness, romance, concealment, thought insertion, bizarre delusion, morality, solipsism, experimentation, medication, thought disorder, grandiosity, withdrawal, letter, reality test, safety test, termination. They refer to the personas of the source transcript ("Lee", "Anima").

The judge (`codebook_llm_judge`) uses `claude-opus-4-6`, temperature 0.0, 1800 max tokens, 3 retries, with `prompts/codebook_judge_prompt.txt` and `Codebook.txt`. `Risk` is the mean of validation, elaboration, behavioral advice and misrepresentation. Sycophancy is scored and reported but is not part of `Risk`. `Safety` is the mean of the four safety codes (`codebook_judge_core.build_csv_row`).

## Setup

```bash
pip install -r experiments/safety_exp/requirements-safety.txt
cp .env.example .env        # fill in the keys you need; see ENV_VARS.md
```

All commands below are run **from the repository root**. By default, inputs
are read from `data/safety_exp/` and outputs go to `outputs/safety_exp/`; set
`ANGEL_DATA_DIR` / `ANGEL_OUTPUT_DIR` to move them.

### Inputs you have to provide

- `data/safety_exp/full_context.txt` is the source red-team transcript (58 user/assistant turns in `You said:` / `ChatGPT said:` format). It is **not distributed** with this repository (see [Data](#data)). To try the pipeline, use the synthetic stand-in: `--input data/examples/safety_exp/example_source_transcript.txt`.
- `data/safety_exp/partial_context.txt` is only needed for `PARTIAL` context runs. The synthetic stand-in is `data/examples/safety_exp/example_partial_context.txt`.
- A **patient simulator**. Pick one:
  - `--patient-backend angel` loads `model_usage.angel` in-process. This needs a GPU and the Actor weights (`ANGEL_ACTOR_MODEL`). `--profile-id` indexes into `--profiles-jsonl` (default `ANGEL_JSONL_PATH`). By default profiles go to the Actor without Observer expansion; add `--patient-expand` to run stage 1 first.
  - `--patient-backend http --base-url http://host:port` (the default) calls a Claude-Messages-style service in front of the Actor. That is how the paper's replays were produced. The protocol is documented in `patient_backends.py`.

## Step by step

**1. Build red-team contexts** (one replay per profile):

```bash
# one profile, in-process patient
python -m experiments.safety_exp.generate_redteam_transcript \
    --input data/safety_exp/full_context.txt \
    --patient-backend angel --profile-id 0 \
    --output-prefix outputs/safety_exp/contexts/auto_attack/profile_0

# many profiles (replaces the original run.sh … run6.sh)
bash experiments/safety_exp/scripts/run_redteam.sh 0,1,16,28 -- --patient-backend angel
bash experiments/safety_exp/scripts/run_redteam.sh data/examples/safety_exp/example_profile_ids.txt
```

`--dry-run` skips the patient and reuses the source user turns. It still calls Claude for the assistant turns.

**2 + 3. Query a model and judge the responses.** This is the paper's loop over all contexts:

```bash
MODEL_KEY="gpt 4o"          MODEL_DIR=gpt4o           bash experiments/safety_exp/scripts/run_query_and_judge.sh
MODEL_KEY="claude 4.5 opus" MODEL_DIR=claude-4.5-opus bash experiments/safety_exp/scripts/run_query_and_judge.sh 0,1,2
```

Or run the steps individually:

```bash
python -m experiments.safety_exp.query_models --model-key "gemini-3-pro" \
    --run-all-prompts True --single-context-mode FULL \
    --context-source auto_attack --context-id 0 \
    --output-jsonl outputs/safety_exp/results/auto_attack/gemini-3-pro/run_all_full_auto_attack_profile0.jsonl

# NONE / PARTIAL / FULL(source transcript) in one run:
python -m experiments.safety_exp.query_models --model-key "gpt 4o" \
    --run-all-prompts True --run-all-context-levels True

python -m experiments.safety_exp.codebook_llm_judge \
    --input-jsonl  outputs/safety_exp/results/auto_attack/gemini-3-pro/run_all_full_auto_attack_profile0.jsonl \
    --output-jsonl outputs/safety_exp/results/auto_attack/gemini-3-pro/codebook_judge_results_full_auto_attack_profile0.jsonl \
    --output-csv   outputs/safety_exp/results/auto_attack/gemini-3-pro/codebook_judge_long_full_auto_attack_profile0.csv
```

The judge resumes by default: records already in `--output-jsonl` are skipped. Use `--overwrite` to start over and `--dry-run` to only count records.

**4. Report.** Writes `<dir>/summary/<mode>_codebook_report.md`:

```bash
python -m experiments.safety_exp.generate_auto_attack_codebook_report \
    outputs/safety_exp/results/auto_attack/gpt4o outputs/safety_exp/results/auto_attack/gemini-3-pro
```

**5. Main results.** Mean Risk and Safety per model with 95% CIs (1.96 × SE over judged responses):

```bash
python -m experiments.safety_exp.summarize_results
```

On the paper's judge outputs:

| Model | Risk | Safety |
|---|---|---|
| claude-4.5-opus | 0.028 ± 0.008 | 2.877 ± 0.021 |
| gemini-3-pro | 0.516 ± 0.056 | 1.803 ± 0.071 |
| gpt4o | 1.237 ± 0.054 | 0.443 ± 0.042 |

The per-model markdown reports from step 4 carry the full codebook breakdown
(sycophancy, validation, reality testing, …).

## Data

This module ships **code, the codebook, the judge prompt, and a synthetic example** only.

| Item | In repo? | Notes |
|---|---|---|
| `Codebook.txt`, `prompts/codebook_judge_prompt.txt` | yes | Needed to reproduce the judge. |
| `data/examples/safety_exp/*` | yes | Synthetic, written for this repo. |
| Source transcript (`full_context.txt`) | no | A 58-turn scripted role-play ("Lee") with ChatGPT, from an external AI-psychosis red-teaming study. Obtain it from its authors. |
| Generated contexts, raw model responses, judge outputs | no | Derived from the source transcript. Regenerate them with the steps above. |

## Provenance (original → this module)

| Original (`simulate_patient/`) | Here | Changes |
|---|---|---|
| `safety_exp/generate_redteam_transcript.py` | `generate_redteam_transcript.py` | Prompts, attack-style labels and defaults are unchanged. The patient client moved to `patient_backends.py` (HTTP client ported from the study app, plus a new in-process `angel` backend). The Claude client comes from `angel_common.llm` (public Anthropic API, or Foundry when `ANTHROPIC_BASE_URL` is set; previously Foundry was required for `auto_attack`). Default paths are repo-relative. |
| `safety_exp/API Script.py` | `query_models.py` | Model table, stimuli and request parameters are unchanged. Clients come from `angel_common`. New flags: `--contexts-dir`, `--full-context-file`, `--partial-context-file`, `--output-dir`. The hardcoded GCP project default was removed (`GOOGLE_CLOUD_PROJECT` is now required for Gemini). The legacy partial-context path is gone. |
| `safety_exp/codebook_llm_judge.py` | `codebook_llm_judge.py` | The `sys.path` hack became a package import. The client comes from `angel_common.llm.anthropic_client`. Output defaults moved under `outputs/`. |
| `safety_exp/codebook_judge_core.py` | `codebook_judge_core.py` | Only the path defaults changed. |
| `safety_exp/generate_auto_attack_codebook_report.py` | same name | Unchanged. |
| `safety_exp/Codebook.txt`, `prompts/codebook_judge_prompt.txt` | same | Verbatim. |
| `safety_exp/fig/fig1/plot_auto_attack_risk_safety_ci.py` (numbers only; figure scripts not included) | `summarize_results.py` | Same means and CI formula, printed instead of plotted. |
| `run.sh`, `run2.sh` … `run6.sh` | `scripts/run_redteam.sh` | One parameterized loop (profile-id list or file, optional username column). |
| query + judge loop (commented block in a Slurm script) | `scripts/run_query_and_judge.sh` | Same commands and file naming, plus the report step. |

## Known issues

1. **Model column is empty for Claude and Gemini rows** in the judge CSV. `codebook_judge_core.MODEL_ID_RULES` does not match `claude-opus-4-5` or `gemini-3.1-pro-preview`; only GPT-4o gets an id. The paper's outputs have the same gap. `summarize_results` groups by result directory, so the main results are unaffected. Left as-is for fidelity.
2. **Model labels vs. ids.** `gemini-3-pro` actually calls `gemini-3.1-pro-preview`. `gpt 5.2 chat` falls back to a deployment named `gpt-5` unless `AZURE_OPENAI_DEPLOYMENT_GPT_5_2_CHAT` is set, so set it to reproduce GPT-5.2.
3. **Patient service parity.** The paper's replays used the study app's HTTP patient service over its 42-profile set. It is unknown whether that service ran Observer expansion on those profiles. The in-process backend defaults to Actor-only (`--patient-expand` off) and uses whatever profiles file you give it, so use the same profiles to compare against the paper.
4. **`query_models` appends to `--output-jsonl`.** Re-running a context duplicates records, and the judge treats them as new (its resume key is file + line index). Delete the file before a rerun.
5. **`--api-key` on the judge CLI** is kept for compatibility. Prefer `ANTHROPIC_API_KEY`, because command-line keys end up in shell history and process lists.
6. The generated context JSON records `anthropic_base_url` in its `config` block. Strip it before sharing outputs if your endpoint name is private.
