# Observer training (stage 1 of Angel)

The **Observer** reads a patient's presenting complaints (or a patient profile)
and builds a **directed symptom network** in two steps:

| Step | Input | Output |
|---|---|---|
| **S1: nodes** | presenting complaints | `<think>…</think><GRAPH>{"symptoms": [...], "external_factors": [...]}</GRAPH>` |
| **S2: edges** | complaints + S1 node list | `<think>…</think><GRAPH>{"links": [{"from": ..., "to": ...}]}</GRAPH>` |

One checkpoint handles both steps: `Qwen3-Observer-800`, a Qwen3-8B trained with
QLoRA SFT followed by GRPO (TRL). The same model also expands short profiles into
long ones for the Actor (see `model_usage/` and `eval/short2long_profile_generation.py`).

Run everything from the **repository root** as modules, e.g.
`python -m model_training.observer.train_grpo --help`.
Install with `pip install -r model_training/observer/requirements-observer.txt`.
Environment variables are listed in [ENV_VARS.md](ENV_VARS.md).

## Pipeline and model lineage

```
Qwen/Qwen3-8B ──SFT S1 (sft_training.jsonl)──► Qwen-3-8B-Patient-SFT            (merged, bnb-4bit)
      │                                             │
      │                                   GRPO S1, 6 × 100 steps (node-recall reward)
      │                                             ▼
      │                                    LoRA  Qwen-3-8B-GRPO-600  (r=16)
      └────────────── merge (onto Qwen/Qwen3-8B) ───┘
                              ▼
               Qwen-3-8B-GRPO-600-S1-merged    (bf16)
                              │ SFT S2 (sft_training_s2.jsonl)
                              ▼
               Qwen-3-8B-Patient-SFT-S2       (merged, bnb-4bit)
                              │ GRPO S2, 8 × 100 steps (Azure gpt-5-mini edge judge)
                              ▼
               LoRA  Qwen-3-8B-GRPO-s2-set2-800-gpt  (r=32)
                              │ merge (--rebuild-clean)
                              ▼
               Qwen3-Observer-800             (bf16, released)
```

Each GRPO run is 100 steps and resumes the previous run's LoRA adapter, so the
numbers in the names (`-600`, `-800`) are cumulative steps. The trainer states
confirm 4 GPUs for every stage:
- S1 SFT: 264 steps.
- S2 SFT: 102 steps.
- GRPO: 4 GPUs × 4 prompts per device.

## Step by step

Outputs go under `data/observer/` and `models/` by default. The example files in
`data/examples/observer/` show every input format.

### 0. Build the data (GPT-5)

The input is `data/observer/case_report_final_all.jsonl`, with fields `title` and
`Complaints`. The paper used 510 case reports; see [Data](#data).

```bash
M="python -m model_training.observer.build_data"
$M add-nodes                                   # -> case_report_final_all_nodes.jsonl   (gpt5_nodes)
$M s1-responses --field Complaints \
   --input data/observer/case_report_final_all_nodes.jsonl \
   --output data/observer/case_report_final_all_nodes_sft.jsonl   # run twice (paper: 2 x 510 = 1020 rows)
$M augment                                     # -> ..._aug_p.jsonl   (new_complaints)
$M s1-responses --field new_complaints         # -> ..._aug_p_sft.jsonl (4579 rows)
$M extract-nodes                               # -> case_report_aug_p_grpo.jsonl
$M s2-responses                                # -> sft_training_s2_data.jsonl (paper: 2138 rows)
$M make-sft --stage s1                         # -> sft_training.jsonl     (5599 rows)
$M make-sft --stage s2                         # -> sft_training_s2.jsonl  (2138 rows)
$M make-grpo                                   # -> grpo_training.jsonl    (5089 rows)
```

Checked against the original files: `make-sft` (S1 and S2) and `make-grpo` rebuild
`sft_training.jsonl`, `sft_training_s2.jsonl` and `grpo_training.jsonl` from the
original intermediate files. The S2 output is byte-identical; S1 and GRPO contain
the same rows in a different order.

### 1. SFT for S1

```bash
torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s1 \
    --output models/Qwen-3-8B-Patient-SFT
```

### 2. GRPO for S1 (6 × 100 steps)

```bash
G="torchrun --nproc_per_node 4 -m model_training.observer.train_grpo --stage s1"
$G --adapter-out models/Qwen-3-8B-GRPO-100 --run-name grpo-s1-100
for s in 200 300 400 500 600; do
  $G --init-adapter models/Qwen-3-8B-GRPO-$((s-100)) --adapter-out models/Qwen-3-8B-GRPO-$s --run-name grpo-s1-$s
done
```

Settings:
- Reward: `0.4·format + 0.6·node_recall`. Node recall uses all-MiniLM-L6-v2 cosine similarity ≥ 0.75 against `gpt5_nodes`, mapped to [-1, 1].
- Training: lr 5e-6 with cosine schedule and 20 warmup steps; 8 generations per prompt; LoRA r=16, α=32.
- Lengths: max prompt 2048 tokens, max completion 4096 tokens.

### 3. Merge S1

```bash
python -m model_training.observer.merge --base Qwen/Qwen3-8B \
    --adapter models/Qwen-3-8B-GRPO-600 --output models/Qwen-3-8B-GRPO-600-S1-merged
```

### 4. SFT for S2 (on the merged S1 model)

```bash
torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s2 \
    --base-model models/Qwen-3-8B-GRPO-600-S1-merged --output models/Qwen-3-8B-Patient-SFT-S2
```

### 5. GRPO for S2 (8 × 100 steps, Azure judge)

```bash
G="torchrun --nproc_per_node 4 -m model_training.observer.train_grpo --stage s2 --reward azure"
$G --adapter-out models/Qwen-3-8B-GRPO-s2-100 --run-name grpo-s2-100
for s in 200 300 400 500 600 700 800; do
  $G --init-adapter models/Qwen-3-8B-GRPO-s2-$((s-100)) --adapter-out models/Qwen-3-8B-GRPO-s2-$s --run-name grpo-s2-$s
done
```

Settings:
- Reward: `0.1·format + 0.6·precision + 0.3·coverage − size_penalty` (weights: `--w-format`, `--w-precision`, `--w-coverage`).
  - **precision**: mean plausibility of the proposed edges in [-1, 1]. The judge (`ANGEL_EDGE_JUDGE_DEPLOYMENT`, gpt-5-mini, up to 3 concurrent requests) scores each edge 0–1. Edges that break the task rules score -1 without a judge call: an endpoint missing from the provided node list, a self-loop, or a duplicate.
  - **coverage**: share of the listed nodes joined by at least one supported edge (judge score ≥ 0.5), mapped to [-1, 1]. Without it, a graph with one safe edge scores as well as the full network.
  - **size_penalty**: 0.05 per edge beyond `--max-edges` (10), capped at 0.3.
  - Judge scores are cached per (complaints, edge), so the same edge gets the same score in every generation of a prompt and GRPO's group comparison reflects the graphs, not judge noise. A failed judge call counts the edge as uncertain (0.5).
  - GRPO normalises the final reward across each prompt's generations, so edge scores are not normalised again inside a rollout.

  On a fixed test case the reward orders graphs as intended:

  | proposed graph | reward |
  |---|---|
  | all 8 correct edges | 0.89 |
  | 3 correct edges | 0.59 |
  | 1 correct edge | 0.44 |
  | 8 correct + 4 implausible | 0.47 |
  | 5 correct + 3 with invented/self-loop nodes | 0.39 |
  | 16 edges, 8 correct | 0.10 |
- Training: lr 1e-5; 4 generations per prompt; LoRA r=32, α=64.

The surviving adapters (`…-set2-{400,600,700,800}-gpt`) and W&B runs show 100-step
increments ending at 800. The earliest increments cannot be fully recovered from
the logs, so the loop above is the intended schedule, not a verified replay.

Alternative S2 rewards:
- `--reward local`: the Qwen3-0.6B classifier with `0.3·format + 0.5·edge_score`. This was the "set2-2" runs, trained on 8 GPUs with `--batch-size 6 --num-generations 6`. It was not used for the released model.
- `--reward format_only --edge-dump data/observer/edge_dump_s2.jsonl`: harvests candidate edges to use as edge-classifier training data.

### 6. Merge S2, giving Qwen3-Observer-800

```bash
python -m model_training.observer.merge --base models/Qwen-3-8B-Patient-SFT-S2 \
    --adapter models/Qwen-3-8B-GRPO-s2-800 --output models/Qwen3-Observer-800 \
    --device cpu --rebuild-clean
```

### Inference

```bash
python -m model_training.observer.predict_network --input data/examples/observer/case_reports.jsonl \
    --input-field Complaints --output outputs/observer/networks.jsonl
```

S1 runs, then S2 using the tokenizer's (Qwen3) chat template, with temperature
0.1 and top_p 0.9. S2 is retried up to 10 times until a valid `<GRAPH>` block
parses.

## Edge classifiers (optional)

| Classifier | How it is built | Used for |
|---|---|---|
| Azure fine-tuned GPT-4 ("edge classifier") | `edge_classifier_data.py from-annotations --annotation-dir <csvs>`, then Azure OpenAI fine-tuning (outside this repo) | labelling harvested edges; the edge "reasonability" score in `eval/` |
| Qwen3-0.6B (`Qwen3-0.6B-Classifier`) | `edge_classifier_data.py label`, then `to-sft`, then `train_edge_classifier.py` | the `--reward local` GRPO variant |

```bash
python -m model_training.observer.edge_classifier_data label      # edge_dump_s2 -> predicted_edges (Azure classifier)
python -m model_training.observer.edge_classifier_data to-sft     # -> classifier_sft.jsonl (paper: 8316 rows)
python -m model_training.observer.train_edge_classifier --output models/Qwen3-0.6B-Classifier
```

## Automatic profile evaluation (`eval/`)

This compares long-profile generators on two measures:
- **Reasonability**: the mean Yes-rate of the edge classifier over the edges of the network the Observer builds from each generated long profile.
- **Diversity**: self-BLEU and embedding distance as the number of generations per short profile, *k*, grows.

```bash
python -m model_training.observer.eval.auto_profile_eval_pipeline \
    --input data/examples/observer/short_profiles.jsonl --input-field short_patient_profile \
    --stage1-models models/Qwen3-Observer-800 --generations-per-model 12 \
    --generation-gpus 0,1,2,3 --network-gpus 0,1,2,3 --run-name ours
# baselines: --stage1-models gpt-5 | claude-opus-4-5 | gemini-2.5-flash | Qwen/Qwen3-8B
python -m model_training.observer.eval.plot_reasonability_diversity_summary
python -m model_training.observer.eval.plot_diversity_by_k
```

The pipeline writes `outputs/observer/auto_eval_runs/<run>/report.{md,json}`.
Paper runs on 50 short profiles:

| Generator | Avg reasonability | Avg edges per profile |
|---|---|---|
| Ours (Observer-800) | 0.892 | 11.0 |
| Claude | 0.894 | 5.7 |
| Gemini | 0.907 | 5.9 |
| GPT-5 | 0.879 | 11.8 |
| Qwen3-8B | 0.879 | 11.7 |

## Known issues

These issues are in the code that produced the released checkpoint. The defaults
reproduce them so that the released model can be re-trained exactly; where a fix
exists, it is behind a flag.

1. **The GRPO chat template is broken.**
   - Cause: `ORIGINAL_GRPO_CHAT_TEMPLATE` references `system_prompt` and `reasoning_start`, which Hugging Face never passes when rendering.
   - Effect: the generation prompt is `"\nSYS<|im_end|>\n        USER\n\n"`, with stray whitespace and no `<think>`.
   - Fix: `--chat-template fixed` renders `SYS<|im_end|>USER<think>`; `native` keeps the Qwen3 template.
   - Related: the prompt format differs across stages. SFT uses `### System/### User/### Assistant`, GRPO uses the template above, and inference uses the Qwen3 native template.
2. **The S1 SFT data is mis-paired.**
   - Of the 5599 rows, 4579 put the *original* complaints in the user turn and, as the assistant turn, GPT-5's answer for the *augmented* complaints (`new_complaints`).
   - Fix: `build_data make-sft --stage s1 --pair-augmented` pairs each answer with the text it was generated from.
3. **The S1 merge used a different base than training.** The S1 GRPO adapter was trained on `Qwen-3-8B-Patient-SFT` but merged onto `Qwen/Qwen3-8B`, which drops the S1 SFT weights. Pass `--base models/Qwen-3-8B-Patient-SFT` to merge onto the model the adapter was trained against.
4. **SFT merges into a 4-bit base.**
   - `sft.py` merges the LoRA into the bitsandbytes-nf4 model, as the original did, so `Qwen-3-8B-Patient-SFT(-S2)` are 4-bit checkpoints.
   - GRPO then trains on them, and Observer-800 is a bf16 export of that model.
   - Use `--save-adapter-only` and `merge.py` with a bf16 base to avoid this.
5. **The local classifier reward** prompts the classifier in a different format from the one it was trained on.
6. **Edge-classifier SFT problems.**
   - The training text has no `### Assistant:` marker, so the loss covers the whole sequence; kept as is.
   - The original evaluation generated from the full text including the label. Here evaluation scores the Yes/No logit on the prompt only, which changes the reported metrics but not the model.
7. **GRPO data repeats cases.** `grpo_training.jsonl` = the 510 case reports + 4579 augmented rows, but GRPO reads only `Complaints` and `gpt5_nodes`, and those are the *original* case's values in the augmented rows. So each case appears about 10 times.

## Provenance

| Original | New |
|---|---|
| `GRPO-Qwen3/GRPO/data.py` | `prompts.py`, `data.py` |
| `GRPO-Qwen3/GRPO/reward.py`, `reward_s2.py`, `reward_s2_local.py`, `classifier/azure_reward_engine.py`; the edge-dump reward from git `d61ccd5` | `rewards.py` |
| `GRPO-Qwen3/GRPO/train_grpo.py`, `train_grpo_local.py` (S1 settings from git `c7b6533`) | `train_grpo.py` |
| `GRPO-Qwen3/Finetune/finetune_hf.py` at git `c7b6533` (S1) and `0352b6d` (S2), before it was repurposed for the Actor | `sft.py` |
| `GRPO-Qwen3/GRPO/merge.py`, `merge_stage2.py` | `merge.py` |
| `GRPO-Qwen3/GRPO/eval.py`, `long_profile_to_network.py` | `predict_network.py` |
| `simulate_patient/data_process/add_node_gpt5.py`, `data_generation_sft.py` (S1 `conversation_gen` from git `d19559f`) | `build_data.py` |
| `GRPO-Qwen3/classifier/classifier_generator.py` | `edge_classifier_api.py` |
| `GRPO-Qwen3/classifier/fine_tune.py`, `classifier/test.py` | `edge_classifier_data.py` |
| `GRPO-Qwen3/Finetune/finetune_classifier.py` | `train_edge_classifier.py` |
| `GRPO-Qwen3/GRPO/{auto_profile_eval_pipeline, short2long_profile_generation, merge_long_profiles, score_network_edges, diversity_metrics, plot_diversity_by_k, plot_reasonability_diversity_summary}.py` | `eval/` (same names) |

Not ported:
- **Math-tutorial leftovers:** the Qwen3-0.6B math GRPO tutorial (`train.py`, `model_init.py`, `dataset_math_reward.py`, root `merge.py`).
- **Superseded Unsloth S1 attempt:** `train_network.py`, `train_s1_debug.py`, `dataset_network_reward.py`, `simulate_patient/observer/reward.py`.
- **Toy and scratch scripts:** `rewards_test.py`, `data_test.py`, `vis.py`, `gemini_test.py`.
- **Broken script:** `data_process_s2.py`.
- **Pre-Observer network generation:** `simulate_patient/ToM_construct/`, which generated GPT-5 networks and distractor nodes/edges.

## Data

This module ships **synthetic** examples only (`data/examples/observer/`). The
real inputs are derived from published psychotherapy case reports, which are
copyrighted:

| File (original location) | Rows | Content |
|---|---|---|
| `case_report_final_all.jsonl` / `_nodes.jsonl` | 510 | title, abstract, presenting-complaints excerpt, GPT-5 nodes |
| `sft_training.jsonl` | 5599 | S1 SFT (case-report text + GPT-5 answers) |
| `sft_training_s2.jsonl` | 2138 | S2 SFT (GPT-5-augmented complaints + GPT-5 networks) |
| `grpo_training.jsonl` | 5089 | GRPO prompts |
| `classifier_sft.jsonl` / `predicted_edges.jsonl` | 8316 | edge Yes/No labels |
| `data/case report/classifier/annotation/*.csv` | — | human edge annotations |
