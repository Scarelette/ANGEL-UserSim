# Profile expansion (main experiment)

Every patient simulator gets the same short profile and is interviewed by the
same LLM therapist over a fixed 14-topic intake agenda. We measure how well it
expands the profile into a varied, faithful conversation.

| Metric | What it measures |
|---|---|
| **Simulation Diversity** (`simulation_diversity`) | how different a patient's answers to the same topic are across repeated runs (nearest-neighbour cosine distance of MiniLM embeddings, scaled by the share of informative answers) |
| **Behavior Diversity** (`behavior_diversity`) | how varied the behaviours are that a GPT-5 judge extracts from each topic |
| **Profile Alignment** (`profile_alignment`) | GPT-5 judge: faithfulness to the profile on 5 aspects, 1–5 rescaled to 0–1 |

| `--model` | Patient simulator | Runs on |
|---|---|---|
| `angel` | Angel: Observer expands the profile, Actor role-plays it | GPU |
| `one_stage` | ablation: Actor on the short profile | GPU |
| `eeyore` | Eeyore | GPU |
| `patient_psi` | Patient-Psi | Azure OpenAI |
| `roleplay_doh` | Roleplay-doh | Azure OpenAI |

## Setup

Run everything from the repository root.

```bash
pip install -r experiments/profile_expansion/requirements-profile-expansion.txt
cp .env.example .env        # then fill in the values below — the only place to edit
```

Settings this experiment reads from `.env`:

| Variable | Needed for | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | therapist, GPT-5 judges, API baselines | — |
| `ANGEL_THERAPIST_DEPLOYMENT` | therapist | `gpt-4.1` |
| `ANGEL_GPT5_DEPLOYMENT` | profile-alignment and behavior judges | `gpt-5` |
| `ANGEL_PATIENT_PSI_DEPLOYMENT`, `ANGEL_ROLEPLAY_DOH_DEPLOYMENT` | API baselines | `gpt-4`, `gpt-4o` |
| `ANGEL_OBSERVER_MODEL`, `ANGEL_ACTOR_MODEL` | `angel`, `one_stage` | `models/Angel-Observer`, `models/Angel-Actor` |
| `EEYORE_MODEL` | `eeyore` | `liusiyang/eeyore_sft_epoch2_dpo_round2_epoch1_llama3.1_8B` |

Optional:
- `AZURE_OPENAI_API_VERSION` (default `2024-12-01-preview`).
- If the therapist or the baselines live on a different Azure resource, set
  `ANGEL_THERAPIST_AZURE_ENDPOINT` / `_API_KEY`, or
  `ANGEL_BASELINE_AZURE_ENDPOINT` / `_API_KEY`.
- `ANGEL_DATA_DIR` / `ANGEL_OUTPUT_DIR` move the input and output folders
  (default `data/` and `outputs/`).

**Input:** `data/profile_expansion/selected_50_short_patient_profiles_v2.jsonl`.
The paper's 50 profiles come from published case reports and are not included;
`data/examples/profile_expansion/short_profiles.example.jsonl` has two synthetic
rows in the same format.

Quick check on one synthetic profile:

```bash
python -m experiments.profile_expansion.run_agenda_experiment --model patient_psi \
  --input data/examples/profile_expansion/short_profiles.example.jsonl \
  --output outputs/profile_expansion/results/smoke.jsonl --limit 1
```

## Reproduce the paper

**1. Transcripts** — 25 runs per profile per model (`--resume` skips finished runs):

```bash
for M in angel eeyore patient_psi roleplay_doh; do
  python -m experiments.profile_expansion.run_agenda_experiment --model $M --num-runs 25 --resume \
    --output outputs/profile_expansion/results/${M}_agenda_runs25.jsonl
done
```

On Slurm: `MODEL=angel sbatch experiments/profile_expansion/scripts/run_agenda.sbatch`
(fill in the `#SBATCH` account/partition first).

**2. Metrics and main results** — for all four models:

```bash
bash experiments/profile_expansion/scripts/evaluate.sh
```

This computes the three metrics for the first *k* = 3…25 runs of each profile
(`evaluate_metrics --run-range 3 25`), gathers each model's *k* = 4 results into
`outputs/profile_expansion/results/clean/` (`combine_metrics`), and prints the
main results table (`report_main_results`: mean over profiles, 95% bootstrap CI).
On the paper's metric files it gives:

| Model | Simulation Diversity | Behavior Diversity | Profile Alignment |
|---|---|---|---|
| Eeyore | 0.1259 [0.1029, 0.1497] | 0.9417 [0.9267, 0.9560] | 0.9184 [0.9075, 0.9284] |
| **Angel** | **0.3924** [0.3821, 0.4034] | **0.9729** [0.9619, 0.9816] | 0.9411 [0.9280, 0.9527] |
| Patient-psi | 0.3230 [0.3159, 0.3301] | 0.9243 [0.9129, 0.9359] | 0.9716 [0.9502, 0.9880] |
| Roleplay-doh | 0.3386 [0.3315, 0.3457] | 0.9370 [0.9248, 0.9490] | 0.9938 [0.9871, 0.9982] |

Judge outputs are cached next to the input (`<input>.record_cache.json`), so
reruns are cheap. The paper's metric files call Simulation Diversity
`min_distance_diversity`; `combine_metrics` and `report_main_results` accept it.

## Files

```
run_agenda_experiment.py   generate transcripts for one --model
interview_process.py       agenda and therapist
patient_models.py          the five patient simulators (code in patients/)
angel_initializer.py       Angel: Observer -> long profile -> Actor
stage1_short2long.py       Observer prompts and generation
azure_clients.py           Azure clients and deployment names per role
evaluate_metrics*.py       metric runner (checkpoints, caching, run sweeps)
metrics/                   profile_alignment, behavior_diversity, simulation_diversity
combine_metrics.py         merge metric files into one per model
report_main_results.py     main results table
scripts/                   run_agenda.sbatch, evaluate.sh
```
