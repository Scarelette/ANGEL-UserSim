# Profile-expansion experiment (main paper experiment)

Each simulated patient receives the **same short profile** and is interviewed by
the same LLM therapist following a fixed 14-topic intake agenda. The experiment
measures how well each patient simulator *expands* the short profile into a
realistic, varied conversation that stays faithful to it:

| Paper name | Metric key | What it measures |
|---|---|---|
| Simulation Diversity | `min_distance_diversity` | per-topic answer diversity across repeated runs of the same profile (MiniLM embeddings, min-distance, informative-coverage adjusted) |
| Behavior Diversity | `behavior_diversity` | diversity of behavioral attributes extracted per topic by a GPT-5 judge (`value_only_contrast_v4`) |
| Profile Alignment | `profile_alignment` | GPT-5 judge, 5 aspects scored 1–5, rescaled to 0–1 |
| (appendix) | `semantic_diversity`, `group_diversity` | mean pairwise / kNN variants of the embedding diversity |

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
(main experiment, 50 profiles) and `..._v3.jsonl` (fixed-attribute experiment,
45 profiles). These were derived from published case reports and are not
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

### 1. Generate transcripts (all figures)

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
  --metrics profile_alignment,semantic_diversity,behavior_diversity,group_diversity,min_distance_diversity \
  --run-range 3 25 --workers 8 --resume
```

`--run-range 3 25` is a prefix sweep: it writes
`<model>_agenda_runs25.metrics.run{k}.json`, each pooling the first *k* runs per
profile. Note that the default `--metrics` omits `profile_alignment` and
`min_distance_diversity`, so pass the list explicitly. Judge outputs are cached
per record in `<input>.record_cache.json`, so re-runs are cheap.

The overall and stage-level figures read one combined file per model at *k*=4:

```bash
python -m experiments.profile_expansion.combine_metrics \
  --inputs outputs/profile_expansion/results/${M}_agenda_runs25.metrics.run4.json \
  --output outputs/profile_expansion/results/clean/${M}_agenda_runs5.metrics.combined.run4.json
```

(The `runs5` in the file name is kept because the figure scripts glob it; in
the paper the 4-run metrics came from the first five runs, computed in a
separate alignment pass and diversity pass — `combine_metrics` accepts several
`--inputs` for that case.)

`scripts/evaluate_and_plot.sh` runs steps 2–3 for all four models.

### 3. Paper figures

```bash
python -m experiments.profile_expansion.figures.plot_paper_figures --print-stats
```

| Output (`outputs/profile_expansion/figures/`) | Paper figure | Script |
|---|---|---|
| `overall_model_performance.pdf` | Overall model performance | `figures.plot_paper_figures` |
| `stage_aspect_performance.pdf` | Stage- and aspect-level performance | `figures.plot_paper_figures` |
| `diversity_by_runs.pdf` | Diversity vs. number of runs | `figures.plot_paper_figures` |
| `diversity_by_fixed_attributes.pdf` | Diversity vs. fixed attributes | `figures.plot_paper_figures` (needs step 4) |
| `metric_analysis.pdf`, `run_count_sensitivity.pdf` | Metric construct validity | `figures.plot_metric_analysis` (needs steps 4–5) |
| `stage_aspect_summary.pdf` / `stage_aspect_performance_point.pdf` | Stage/aspect variants | `figures.plot_stage_aspect_paper` / `figures.plot_stage_aspect_points` |
| `profile_exp_summary.pdf` | Summary panel | `figures.plot_profile_exp_summary` |
| `stage_profile_discrimination.pdf`, `conversation_profile_structure.pdf` | Appendix | `figures.plot_metric_appendix` (needs step 5) |

`results/figure_stats.txt` holds the paper's numbers (means and 95% bootstrap
CIs) and `results/figure_snippets.tex` the LaTeX figure blocks. Running
`plot_paper_figures --print-stats` on the paper's metric files reproduces
`figure_stats.txt` exactly.

### 4. Fixed-attribute experiment

Tests whether diversity falls as more profile attributes are held fixed.

```bash
# (a) variants: for k = 1..N fixed attributes, Claude rewrites each profile keeping only k
python -m experiments.profile_expansion.fixed_attributes.generate_masked_profile_variants \
  --samples-per-count 3 --profile-workers 2 --workers 6 --resume
# (b) first run: one transcript per (profile, k) for up to 10 profiles with >= 22 attributes
for M in angel eeyore patient_psi roleplay_doh; do
  python -m experiments.profile_expansion.run_fixattr_variant_experiment \
    --model $M --output outputs/profile_expansion/fixed_attributes/fixattr_variant_${M}.jsonl --resume
done
# (c) the union of variants used in (b) -> input for the extra runs
python -m experiments.profile_expansion.fixed_attributes.export_variant_inputs
# (d) 3 more runs per variant, merged into fixattr_variant_<model>_runs4.jsonl
MODELS="angel eeyore patient_psi roleplay_doh" sbatch experiments/profile_expansion/scripts/fixattr_runs.sbatch
# (e) diversity metrics per variant (k = 4 runs)
bash experiments/profile_expansion/fixed_attributes/eval_fixattr_variant_diversity.sh 4
# (f) per-metric plots (Figure 4 itself comes from plot_paper_figures)
python -m experiments.profile_expansion.fixed_attributes.plot_fixed_attr_diversity_four_metrics
```

`fixed_attributes.extract_fixed_attributes` writes the per-profile attribute
inventory (`..._v3.fixed_attributes.jsonl`) on its own; the variant generator
performs the same extraction internally.

### 5. Metric validity and correlation (appendix)

Run on the Patient-Psi 25-run transcripts by default (`--input` to change):

```bash
python -m experiments.profile_expansion.metric_validity.plot_validity_compact_combined \
  --dump-scores outputs/profile_expansion/metric_validity/final/score_validity_samples.json \
  --dump-stages outputs/profile_expansion/metric_validity/final/stage_validity_means.json
python -m experiments.profile_expansion.metric_validity.validate_scores_combined
python -m experiments.profile_expansion.metric_validity.plot_heatmap_four_metrics
python -m experiments.profile_expansion.metric_validity.plot_clusters_four_metrics
python -m experiments.profile_expansion.metric_correlation.analyze_metric_correlation
```

Each script's `--help` lists its options; `validate_diversity_metrics`,
`validate_metric_scores`, `plot_conversation_clusters` and
`plot_per_topic_bars_four_metrics` are the single-metric versions.

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
metrics/                     profile_alignment, semantic/min/kNN diversity, behavior_diversity
combine_metrics.py           merge metric sections into the files the figures read
run_fixattr_variant_experiment.py, fixed_attributes/   fixed-attribute experiment
metric_validity/, metric_correlation/                  appendix analyses
figures/                     paper figures
layout.py                    default input/output locations
scripts/                     Slurm / shell drivers (no credentials)
results/                     paper numbers (aggregates only) and LaTeX figure snippets
```

## Provenance

Ported from the research code (`simulate_patient/`, `GRPO-Qwen3/`). Logic,
prompts, sampling parameters and metric definitions are unchanged; changes are
limited to imports, paths, credentials, and the items under *Known issues*.

| Original | Here |
|---|---|
| `profile_expansion/{run_agenda_experiment,interview_process,patient_models,angel_initializer,run_fixattr_variant_experiment,evaluate_metrics*}.py` | same names |
| `profile_expansion/metrics/{common,profile_alignment,semantic_diversity,semantic_diversity_knn,semantic_diversity_min,behavior_diversity}.py` | `metrics/` |
| `evaluation/{angel,ai_patient,patient_profile,state_manager,patient_psi,roleplay_doh,eeyore}.py`, `actor/sys_prompt.py` | `patients/` |
| `GRPO-Qwen3/GRPO/short2long_profile_generation.py` (local-generation parts) | `stage1_short2long.py` |
| `profile_expansion/{extract_fixed_attributes,generate_masked_profile_variants}.py`, `unit_test/fix_attr/{merge_existing_plus_runs.py,eval_fixattr_variant_diversity.sh,plot_fixed_attr_diversity_four_metrics.py}` | `fixed_attributes/` |
| `unit_test/metric_validity/*.py`, `unit_test/metric_correlation/*.py` | `metric_validity/`, `metric_correlation/` |
| `results/fig/{acl_fig_style,plot_paper_figures,plot_metric_analysis,plot_metric_appendix,plot_stage_aspect_paper,plot_stage_aspect_points}.py`, `results/fig/summary/{paper_style,plot_profile_exp_summary}.py` | `figures/` |
| `output_generator.getOutput` | `angel_common.llm.get_output` |
| `run.slurm`, `run_fixattr_*.slurm`, `run_patient_psi_array.slurm` | `scripts/` (credentials removed) |
| — (done by hand) | `combine_metrics.py`, `fixed_attributes/export_variant_inputs.py`, `layout.py`, `azure_clients.py` |

Not ported: `results_old/`, `metrics/semantic_diversity_v0.py`,
`recompute_metrics_firstk_changed_only.py`,
`compute_adjusted_topic_scores_rglobal.py`,
`evaluate_semantic_diversity_informative.py`, `plot_run_metrics_dot_ci.py`,
superseded figure scripts (`plot_topic_model_*`, `plot_points_acl.py`,
`plot_aspect_alignment_acl.py`, `run_number/`, `summary/plot_run*`,
`unit_test_attribute/`), and `unit_test/complex_stage/` (its inputs came from the
superseded `results_old/` pipeline).

## Known issues

- **Patient-Psi input.** `patient_models.build_evaluation_patient_model` looks for
  `patient_processed_result.patient_psi_profile`, but the input rows carry
  `patient_psi_profile` at the top level, so Patient-Psi is built from the short
  profile (`history` = short profile, other CCD fields empty). The paper's numbers
  reflect this behavior; it also means all models receive the same information.
- **Observer temperature is ignored.** `LocalProfileGenerator.generate` samples with
  `top_p=0.9` at the transformers default temperature 1.0 (the checkpoint's
  `generation_config.json` sets none); its `temperature` argument (0.1 by
  default in `TwoStageAngelPatient`) is unused — commented out upstream. Kept as run.
- **Profile Alignment is stochastic.** The GPT-5 judge runs at temperature 1. The
  paper's 4-run alignment scores come from a separate judging pass over the first
  five runs; re-judging the same transcripts gives values ~0.02 different (Angel:
  0.941 in the paper vs. 0.918 in a later pass).
- **Default `--metrics`** of `evaluate_metrics` is
  `semantic_diversity,behavior_diversity,group_diversity`; pass the full list
  (as above) to get the paper metrics.
- **Deployment names** `gpt-4-04-14` and `gpt-4o-2` are aliases on the paper's
  Azure resource, not public model ids; set `ANGEL_*_DEPLOYMENT`.
- **Stage-1 JSON extraction** (`extract_first_json_object`) counts braces without
  tracking string literals; a `{` inside a string can truncate the object. Failed
  parses fall through to a retry and a repair pass, then a minimal profile. Kept
  as run.
- `patients/angel.py` prints each system prompt (`sys_P:`) to stdout.
- Removed from the port: an optional plotting mode in
  `plot_fixed_attr_diversity_four_metrics.py` / `plot_metric_analysis.py` that
  multiplied fixed-attribute scores by a synthetic downward drift. The paper's
  Figure 4 plots measured values and does not use it.
- The Angel stage-1 → stage-2 schema adapter (`angel_initializer._adapt_stage1_profile_to_angel`)
  discards the Observer's free-text `simulation_rules`; kept as run.
