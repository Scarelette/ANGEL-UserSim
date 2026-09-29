# Observer training (stage 1 of Angel)

The **Observer** reads a patient's presenting complaints and builds a
**directed symptom network** in two steps:

| Step | Input | Output |
|---|---|---|
| **S1: nodes** | presenting complaints | `<think>…</think><GRAPH>{"symptoms": [...], "external_factors": [...]}</GRAPH>` |
| **S2: edges** | complaints + the S1 nodes | `<think>…</think><GRAPH>{"links": [{"from": ..., "to": ...}]}</GRAPH>` |

One checkpoint does both: `Qwen3-Observer-800`, Qwen3-8B trained with QLoRA SFT
then GRPO. The same model expands short patient profiles into long ones for the
Actor (`model_usage/`).

```
Qwen/Qwen3-8B ─SFT S1─► Qwen-3-8B-Patient-SFT ─GRPO S1, 6×100 steps─► LoRA Qwen-3-8B-GRPO-600
                                                                        │ merge onto Qwen/Qwen3-8B
                                                                        ▼
                     Qwen-3-8B-Patient-SFT-S2 ◄─SFT S2── Qwen-3-8B-GRPO-600-S1-merged
                              │ GRPO S2, 8×100 steps (LLM edge judge)
                              ▼ merge
                     Qwen3-Observer-800
```

Each GRPO run is 100 steps and resumes the previous run's LoRA, so `-600` and
`-800` are cumulative steps. The paper used 4 GPUs throughout.

## Setup

Run everything from the repository root.

```bash
pip install -r model_training/observer/requirements-observer.txt
cp .env.example .env        # then fill in the values below — the only place to edit
```

| Variable | Needed for | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | GPT-5 data building, the S2 edge judge | — |
| `ANGEL_GPT5_DEPLOYMENT` | GPT-5 (data building, long-profile prose in `eval/`) | `gpt-5` |
| `ANGEL_EDGE_JUDGE_DEPLOYMENT` | S2 GRPO edge judge | `gpt-5-mini` |
| `ANGEL_BASE_MODEL` | base model | `Qwen/Qwen3-8B` |
| `ANGEL_OBSERVER_MODEL` | Observer for `predict_network` and `eval/` | `models/Qwen3-Observer-800` |

Optional:
- `ANGEL_EDGE_CLASSIFIER_DEPLOYMENT`: only for the "reasonability" score in `eval/`; the name of your fine-tuned edge-classifier deployment (see [Edge classifier](#edge-classifier-for-eval)).
- `ANGEL_EDGE_JUDGE_*` / `ANGEL_EDGE_CLASSIFIER_*` `_ENDPOINT` and `_API_KEY`, if those models live on another Azure resource.
- `ANTHROPIC_API_KEY` and `GOOGLE_CLOUD_PROJECT`, for the Claude and Gemini baselines in `eval/`.
- `--wandb-project` plus `WANDB_API_KEY`, for W&B logging.
- `ANGEL_DATA_DIR` / `ANGEL_MODELS_DIR` / `ANGEL_OUTPUT_DIR` to move the folders.

Outputs go under `data/observer/` and `models/`. `data/examples/observer/` has
synthetic examples of every input file.

## Steps

```bash
# 0. data (GPT-5), from data/observer/case_report_final_all.jsonl (fields: title, Complaints)
M="python -m model_training.observer.build_data"
$M add-nodes                      # reference nodes (gpt5_nodes)
$M s1-responses --field Complaints --input data/observer/case_report_final_all_nodes.jsonl \
   --output data/observer/case_report_final_all_nodes_sft.jsonl       # paper: run twice
$M augment                        # paraphrased complaints (new_complaints)
$M s1-responses --field new_complaints
$M extract-nodes
$M s2-responses
$M make-sft --stage s1            # sft_training.jsonl     (paper: 5599 rows)
$M make-sft --stage s2            # sft_training_s2.jsonl  (paper: 2138 rows)
$M make-grpo                      # grpo_training.jsonl    (paper: 5089 rows)

# 1. SFT S1
torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s1 --output models/Qwen-3-8B-Patient-SFT

# 2. GRPO S1, 6 × 100 steps
G="torchrun --nproc_per_node 4 -m model_training.observer.train_grpo --stage s1"
$G --adapter-out models/Qwen-3-8B-GRPO-100
for s in 200 300 400 500 600; do
  $G --init-adapter models/Qwen-3-8B-GRPO-$((s-100)) --adapter-out models/Qwen-3-8B-GRPO-$s
done

# 3. merge S1
python -m model_training.observer.merge --base Qwen/Qwen3-8B \
    --adapter models/Qwen-3-8B-GRPO-600 --output models/Qwen-3-8B-GRPO-600-S1-merged

# 4. SFT S2
torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s2 \
    --base-model models/Qwen-3-8B-GRPO-600-S1-merged --output models/Qwen-3-8B-Patient-SFT-S2

# 5. GRPO S2, 8 × 100 steps
G="torchrun --nproc_per_node 4 -m model_training.observer.train_grpo --stage s2"
$G --adapter-out models/Qwen-3-8B-GRPO-s2-100
for s in 200 300 400 500 600 700 800; do
  $G --init-adapter models/Qwen-3-8B-GRPO-s2-$((s-100)) --adapter-out models/Qwen-3-8B-GRPO-s2-$s
done

# 6. merge S2 -> the Observer
python -m model_training.observer.merge --base models/Qwen-3-8B-Patient-SFT-S2 \
    --adapter models/Qwen-3-8B-GRPO-s2-800 --output models/Qwen3-Observer-800 --device cpu --rebuild-clean
```

**Settings (paper values)**

| | Stage 1 | Stage 2 |
|---|---|---|
| SFT | LoRA r=64 / α=16, lr 1e-4, 3 epochs | same |
| GRPO lr | 5e-6 (cosine, 20 warmup steps) | 1e-5 |
| generations per prompt | 8 | 4 |
| GRPO LoRA | r=16 / α=32 | r=32 / α=64 |
| max prompt / completion length | 2048 / 4096 tokens | 2048 / 4096 tokens |

**Stage 1 reward:** `0.4·format + 0.6·node recall`. Node recall is the share of
the reference nodes (`gpt5_nodes`) matched at MiniLM cosine similarity ≥ 0.75,
mapped to [-1, 1].

**Stage 2 reward:** `0.1·format + 0.6·precision + 0.3·coverage − size penalty`.
The weights can be set with `--w-format`, `--w-precision` and `--w-coverage`.
- **Precision** is the mean plausibility of the proposed edges, in [-1, 1].
  - An LLM judge (gpt-5-mini) scores each edge 0–1.
  - Edges with a node not in the given list, self-loops and duplicates score −1 without a judge call.
- **Coverage** is the share of the given nodes joined by a supported edge (judge score ≥ 0.5). Without it, one safe edge would score as well as the full network.
- **Size penalty** is 0.05 per edge beyond `--max-edges` (10), capped at 0.3.
- **Consistency:** judge scores are cached per (complaints, edge), so every generation of a prompt is compared on the same scores.

**Build a network with a trained Observer:**

```bash
python -m model_training.observer.predict_network --input data/examples/observer/case_reports.jsonl \
    --input-field Complaints --output outputs/observer/networks.jsonl
```

## Edge classifier (for `eval/`)

The evaluation's "reasonability" score comes from a Yes/No edge classifier: a
GPT-4 deployment fine-tuned on human edge annotations with Azure OpenAI.
`edge_classifier_data` builds the fine-tuning data from the annotation CSVs:

```bash
python -m model_training.observer.edge_classifier_data from-annotations --annotation-dir <csvs>
```

Run the fine-tuning in Azure, then set `ANGEL_EDGE_CLASSIFIER_DEPLOYMENT` to the
deployment's name.

## Automatic profile evaluation (`eval/`)

This compares long-profile generators on two measures:
- **Reasonability:** how plausible the classifier finds the edges of the
  network the Observer builds from each generated profile.
- **Diversity:** self-BLEU and embedding distance across repeated generations.

```bash
python -m model_training.observer.eval.auto_profile_eval_pipeline \
    --input data/examples/observer/short_profiles.jsonl --input-field short_patient_profile \
    --stage1-models models/Qwen3-Observer-800 --generations-per-model 12 --run-name ours
# baselines: --stage1-models gpt-5 | claude-opus-4-5 | gemini-2.5-flash | Qwen/Qwen3-8B
# several GPUs: add --generation-gpus 0,1,2,3 --network-gpus 0,1,2,3
```

The report is written to `outputs/observer/auto_eval_runs/<run>/report.md`.
The paper's results on 50 profiles:

| Generator | Reasonability | Edges per profile |
|---|---|---|
| Ours (Observer-800) | 0.892 | 11.0 |
| Claude | 0.894 | 5.7 |
| Gemini | 0.907 | 5.9 |
| GPT-5 | 0.879 | 11.8 |
| Qwen3-8B | 0.879 | 11.7 |

## Known issues

These are kept as in the paper's runs. Where a flag fixes an issue, it is off
by default.

1. **Broken GRPO chat template.** It references variables Hugging Face never
   fills in, so the prompt has stray whitespace and no `<think>`. Stages also
   format prompts differently: SFT uses `### System/User/Assistant`, GRPO this
   template, inference the Qwen3 template. The flag `--chat-template fixed` or
   `native` fixes the template.
2. **Mis-paired stage-1 SFT data.** 4579 of 5599 rows pair the original
   complaints with GPT-5's answer for the paraphrased ones. Fix:
   `make-sft --stage s1 --pair-augmented`.
3. **Stage-1 merge base.** The S1 GRPO adapter was trained on
   `Qwen-3-8B-Patient-SFT` but merged onto `Qwen/Qwen3-8B`. Fix: pass
   `--base models/Qwen-3-8B-Patient-SFT` to merge onto the model it was trained
   on.
4. **4-bit SFT merges.** `sft` merges into the 4-bit base. Fix:
   `--save-adapter-only`, then `merge` onto a bf16 base.
5. **Repeated GRPO cases.** GRPO data repeats each of the 510 cases about 10
   times: the paraphrased rows carry the original complaints.

## Data

Only synthetic examples are included. The paper's data come from published
case reports, which are copyrighted:
- 510 case reports with presenting complaints;
- 5599 stage-1 and 2138 stage-2 SFT rows;
- 5089 GRPO prompts;
- human edge annotations (for the edge classifier).
