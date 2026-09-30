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
Qwen/Qwen3-8B
  │ SFT S1 + merge
  ▼
Qwen-3-8B-Patient-SFT
  │ GRPO S1 (6×100 steps) + merge
  ▼
Qwen-3-8B-GRPO-600-S1-merged
  │ SFT S2 + merge
  ▼
Qwen-3-8B-Patient-SFT-S2
  │ GRPO S2 (8×100 steps, gpt-5-mini edge judge) + merge
  ▼
Qwen3-Observer-800
```

Each step trains a LoRA adapter (on a 4-bit copy of the model) and then
`merge` applies it to the model it was trained on, giving a full-precision
(bf16) model for the next step.

Each GRPO run is 100 steps and resumes the previous run's LoRA, so `-600` and
`-800` are cumulative steps. The paper used 4 GPUs throughout.

## Setup

Run everything from the repository root.

```bash
pip install -r model_training/observer/requirements-observer.txt
cp .env.example .env        # then fill in the values below — the only place to edit
```

Training uses three models:

| Model | Used for | Setting (default) |
|---|---|---|
| **GPT-5** (Azure OpenAI) | building the training data | `ANGEL_GPT5_DEPLOYMENT` (`gpt-5`) |
| **gpt-5-mini** (Azure OpenAI) | scoring edges during stage-2 GRPO | `ANGEL_EDGE_JUDGE_DEPLOYMENT` (`gpt-5-mini`) |
| **Qwen3-8B** | the model being trained | `ANGEL_BASE_MODEL` (`Qwen/Qwen3-8B`, downloaded automatically) |

In `.env`, set your Azure OpenAI resource:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your key>
```

If your Azure deployments are named `gpt-5` and `gpt-5-mini`, that is all.
Otherwise, also set the two deployment names.

Add `--wandb-project <name>` to log training to Weights & Biases.

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

# 1. SFT S1, then merge onto Qwen3-8B
torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s1 --output models/Qwen-3-8B-Patient-SFT-lora
python -m model_training.observer.merge --base Qwen/Qwen3-8B \
    --adapter models/Qwen-3-8B-Patient-SFT-lora --output models/Qwen-3-8B-Patient-SFT

# 2. GRPO S1, 6 × 100 steps
G="torchrun --nproc_per_node 4 -m model_training.observer.train_grpo --stage s1"
$G --adapter-out models/Qwen-3-8B-GRPO-100
for s in 200 300 400 500 600; do
  $G --init-adapter models/Qwen-3-8B-GRPO-$((s-100)) --adapter-out models/Qwen-3-8B-GRPO-$s
done

# 3. merge the S1 GRPO adapter onto the SFT model it was trained on
python -m model_training.observer.merge --base models/Qwen-3-8B-Patient-SFT \
    --adapter models/Qwen-3-8B-GRPO-600 --output models/Qwen-3-8B-GRPO-600-S1-merged

# 4. SFT S2, then merge
torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s2 \
    --base-model models/Qwen-3-8B-GRPO-600-S1-merged --output models/Qwen-3-8B-Patient-SFT-S2-lora
python -m model_training.observer.merge --base models/Qwen-3-8B-GRPO-600-S1-merged \
    --adapter models/Qwen-3-8B-Patient-SFT-S2-lora --output models/Qwen-3-8B-Patient-SFT-S2

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
    --output outputs/observer/networks.jsonl
```

## Prompt format

Every stage uses the same prompt: the task's system prompt and the complaints,
rendered with Qwen3's chat template. That covers SFT, GRPO, `predict_network`
and `model_usage`. The Observer answers with `<think>…</think>` reasoning, then
the `<GRAPH>` JSON. Each stage-1 SFT answer is paired with the complaints GPT-5
answered: the original text, or its paraphrase for augmented rows.


## Data

Only synthetic examples are included. The paper's data come from published
case reports, which are copyrighted:
- 510 case reports with presenting complaints;
- 5599 stage-1 and 2138 stage-2 SFT rows;
- 5089 GRPO prompts.
