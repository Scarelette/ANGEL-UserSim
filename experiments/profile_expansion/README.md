# Profile-expansion experiment (main paper experiment)

Each simulated patient receives the **same short profile** and is interviewed by
the same LLM therapist following a fixed 14-topic intake agenda. The experiment
measures how well each patient simulator *expands* the short profile into a
realistic, varied conversation that stays faithful to it:

| Paper name | Metric key | What it measures |
|---|---|---|
| Simulation Diversity | `simulation_diversity` | per-topic answer diversity across repeated runs of the same profile: mean nearest-neighbour cosine distance between runs' answers (MiniLM embeddings), scaled by the share of informative answers |
| Behavior Diversity | `behavior_diversity` | diversity of behavioral attributes extracted per topic by a GPT-5 judge (`value_only_contrast_v4`) |
| Profile Alignment | `profile_alignment` | GPT-5 judge, 5 aspects scored 1–5, rescaled to 0–1 |

Patient models compared (`--model`):

| `--model` | System | Backend |
|---|---|---|
| `angel` | **Angel** — Observer (stage 1) expands short → long profile, Actor (stage 2) role-plays it | local: `ANGEL_OBSERVER_MODEL` + `ANGEL_ACTOR_MODEL` |
| `one_stage` | ablation: the Actor prompted with the short profile directly | local: `ANGEL_ACTOR_MODEL` |
| `eeyore` | Eeyore | local: `EEYORE_MODEL` (HF Hub) |
| `patient_psi` | Patient-Psi | Azure OpenAI (`ANGEL_PATIENT_PSI_DEPLOYMENT`) |
| `roleplay_doh` | Roleplay-doh | Azure OpenAI (`ANGEL_ROLEPLAY_DOH_DEPLOYMENT`) |

The therapist is an Azure OpenAI model (`ANGEL_THERAPIST_DEPLOYMENT`; gpt-4.1 in
the paper). For every agenda topic it asks an opening question, then decides
after each patient answer whether to `move_next` or `follow_up`, up to
`--max-therapist-turns-per-topic` (default 8) turns per topic. The agenda
(`interview_process.py`): presenting problem; symptoms — emotions, behaviors,
cognitions; onset/timeline; triggers; impact on functioning; current coping;
social support; past treatment; risk — suicide/self-harm, harm to others,
substance use; treatment goal.

## Setup

All commands run **from the repository root** with `python -m`.

```bash
pip install -r experiments/profile_expansion/requirements-profile-expansion.txt
cp .env.example .env      # fill in keys; see ENV_VARS.md for every variable
```

Model weights (see the repository README for where to get them):

```bash
export ANGEL_OBSERVER_MODEL=/path/to/Qwen3-Observer-800     # or place under models/
export ANGEL_ACTOR_MODEL=/path/to/qwen3-8b-dpo-merged
```

Angel and Eeyore each need one GPU (~16 GB bf16 per 8B model; Angel loads both
stages, ~32 GB). The API-backed models need no GPU.

**Inputs.** `data/profile_expansion/selected_50_short_patient_profiles_v2.jsonl`
(50 profiles). These were derived from published case reports and are not
included; `data/examples/profile_expansion/short_profiles.example.jsonl` has two
synthetic rows with the same schema. Outputs go to
`outputs/profile_expansion/` (override with `ANGEL_OUTPUT_DIR`).

Quick check with the synthetic rows:

```bash
python -m experiments.profile_expansion.run_agenda_experiment \
  --model patient_psi \
  --input data/examples/profile_expansion/short_profiles.example.jsonl \
  --output outputs/profile_expansion/results/smoke_patient_psi.jsonl --limit 1
```

## Reproducing the paper

### 1. Generate transcripts

25 runs per profile per model; `--resume` skips finished (profile, run) pairs.

```bash
for M in angel eeyore patient_psi roleplay_doh; do
  python -m experiments.profile_expansion.run_agenda_experiment \
    --model $M \
    --output outputs/profile_expansion/results/${M}_agenda_runs25.jsonl \
    --num-runs 25 --resume
done
# ablation
python -m experiments.profile_expansion.run_agenda_experiment \
  --model one_stage --output outputs/profile_expansion/results/one_stage_agenda.jsonl --num-runs 4 --resume
```

On Slurm: `MODEL=angel sbatch experiments/profile_expansion/scripts/run_agenda.sbatch`
(edit the `#SBATCH` placeholders first). API-backed models accept `--workers N`
(AIMD back-off on rate limits).

### 2. Metrics

```bash
M=angel
python -m experiments.profile_expansion.evaluate_metrics \
  --input  outputs/profile_expansion/results/${M}_agenda_runs25.jsonl \
  --output outputs/profile_expansion/results/${M}_agenda_runs25.metrics.json \
  --run-range 3 25 --workers 8 --resume
```

`--run-range 3 25` is a prefix sweep: it writes
`<model>_agenda_runs25.metrics.run{k}.json`, each pooling the first *k* runs per
profile. All three metrics are computed by default (`--metrics` selects a
subset). Judge outputs are cached
per record in `<input>.record_cache.json`, so re-runs are cheap.

The main results use one combined file per model at *k*=4:

```bash
python -m experiments.profile_expansion.combine_metrics \
  --inputs outputs/profile_expansion/results/${M}_agenda_runs25.metrics.run4.json \
  --output outputs/profile_expansion/results/clean/${M}_agenda_runs5.metrics.combined.run4.json
```

(The `runs5` in the file name is kept from the paper's runs; there the 4-run
metrics came from the first five runs, computed in a separate alignment pass and
diversity pass — `combine_metrics` accepts several `--inputs` for that case.)

Metric files from the paper's runs name Simulation Diversity
`min_distance_diversity`; `combine_metrics` and `report_main_results` read that
name too.

`scripts/evaluate.sh` runs steps 2–3 for all four models.

### 3. Main results table

```bash
python -m experiments.profile_expansion.report_main_results \
  --json outputs/profile_expansion/main_results.json
```

Prints Simulation Diversity, Behavior Diversity and Profile Alignment per model
(mean over profiles, 95% bootstrap CI with 10k resamples, seed 42). On the
paper's metric files it reproduces the reported numbers exactly:

| Model | Simulation Diversity | Behavior Diversity | Profile Alignment |
|---|---|---|---|
| Eeyore | 0.1259 [0.1029, 0.1497] | 0.9417 [0.9267, 0.9560] | 0.9184 [0.9075, 0.9284] |
| **Angel** | **0.3924** [0.3821, 0.4034] | **0.9729** [0.9619, 0.9816] | 0.9411 [0.9280, 0.9527] |
| Patient-psi | 0.3230 [0.3159, 0.3301] | 0.9243 [0.9129, 0.9359] | 0.9716 [0.9502, 0.9880] |
| Roleplay-doh | 0.3386 [0.3315, 0.3457] | 0.9370 [0.9248, 0.9490] | 0.9938 [0.9871, 0.9982] |

## Layout

```
run_agenda_experiment.py     CLI: transcripts for one model over a profile file
interview_process.py         agenda, therapist (Azure), topic-transition logic
patient_models.py            patient-model factory (the five --model choices)
angel_initializer.py         Angel two-stage wrapper (Observer -> schema adapter -> Actor)
stage1_short2long.py         Observer prompts + local generation (verbatim prompts)
patients/                    Angel Actor, Eeyore, Patient-Psi, Roleplay-doh, profile/state helpers
azure_clients.py             per-role Azure clients and deployment names
evaluate_metrics*.py         metric runner (checkpoints, record cache, run sweeps)
metrics/                     profile_alignment, behavior_diversity, simulation_diversity
combine_metrics.py           merge metric sections into one file per model
report_main_results.py       main results table (means + 95% bootstrap CIs)
layout.py                    default input/output locations
scripts/                     Slurm / shell drivers (no credentials)
```
