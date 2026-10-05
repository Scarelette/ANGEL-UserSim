# Observer training (stage 1 of Angel)

The **Observer** reads a patient's presenting complaints and builds a
**directed symptom network** in two steps:

| Step | Input | Output |
|---|---|---|
| **S1: nodes** | presenting complaints | `<think>…</think><GRAPH>{"symptoms": [...], "external_factors": [...]}</GRAPH>` |
| **S2: edges** | complaints + the S1 nodes | `<think>…</think><GRAPH>{"links": [{"from": ..., "to": ...}]}</GRAPH>` |

One checkpoint does both steps: `Angel-Observer`, which is Qwen3-8B trained
with QLoRA SFT and then GRPO. The same model also expands short patient
profiles into long ones for the Actor (`model_usage/`).

Just want to use the trained Observer? It's on Hugging Face as
[`ChengLi0228/Angel-Observer`](https://huggingface.co/ChengLi0228/Angel-Observer),
so you don't need this folder; see
[`model_usage/`](../../model_usage/), or
[build networks](#build-networks-with-a-trained-observer) directly.

## Pipeline at a glance

```
case reports (title + Complaints)
  │ 0. build_data (GPT-5)       training data for S1, S2 and GRPO
  ▼
Qwen/Qwen3-8B
  │ 1. SFT S1 + merge
  ▼
Qwen-3-8B-Patient-SFT
  │ 2–3. GRPO S1 (6 × 100 steps) + merge
  ▼
Qwen-3-8B-GRPO-600-S1-merged
  │ 4. SFT S2 + merge
  ▼
Qwen-3-8B-Patient-SFT-S2
  │ 5–6. GRPO S2 (8 × 100 steps, gpt-5-mini edge judge) + merge
  ▼
Angel-Observer
```

Each training step trains a LoRA adapter on a 4-bit copy of the model. `merge`
then applies the adapter to the model it was trained on and saves a
full-precision (bf16) model for the next step.

Each GRPO run is 100 steps and continues from the previous run's LoRA, so the
`-600` and `-800` in the names are cumulative steps.

## What you need

- **GPUs:** the paper used 4 GPUs (`torchrun --nproc_per_node 4`) for every
  SFT and GRPO step. Fewer GPUs work with a smaller `--nproc_per_node`, but
  batch sizes are per GPU, so the effective batch, and with it the results,
  will differ.
- **Azure OpenAI:** GPT-5 builds the training data (step 0), and gpt-5-mini
  scores edges during S2 GRPO (step 5).
- **Input data:** a JSONL file of case reports, one per line, with a `title`
  and a `Complaints` (presenting-complaints text) field, saved as
  `data/observer/case_report_final_all.jsonl`. The paper trained on 510 of
  the 516 published case reports in [PSYCHE](../../README.md#-psyche-dataset-coming-soon);
  the case report texts themselves can't be redistributed. A synthetic example is in
  `data/examples/observer/case_reports.jsonl`.

## Setup

Run every command from the **repository root**.

```bash
pip install -r model_training/observer/requirements-observer.txt
cp .env.example .env        # fill in the values below; .env is the only place to edit
```

In `.env`, set your Azure OpenAI resource:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your key>
```

If your deployments are named `gpt-5` and `gpt-5-mini`, that's all.
Otherwise, set the names too:

| Variable | Used for | Default |
|---|---|---|
| `ANGEL_GPT5_DEPLOYMENT` | building the training data | `gpt-5` |
| `ANGEL_EDGE_JUDGE_DEPLOYMENT` | scoring edges in S2 GRPO | `gpt-5-mini` |
| `ANGEL_EDGE_JUDGE_ENDPOINT`, `ANGEL_EDGE_JUDGE_API_KEY` | the edge judge on a different Azure resource | the `AZURE_OPENAI_*` values |
| `ANGEL_BASE_MODEL` | the model being trained | `Qwen/Qwen3-8B` (downloaded automatically) |
| `ANGEL_DATA_DIR`, `ANGEL_MODELS_DIR`, `ANGEL_OUTPUT_DIR` | move `data/`, `models/`, `outputs/` | inside the repo |

To log GRPO runs to Weights & Biases, add `--wandb-project <name>` to
`train_grpo` (and set `WANDB_API_KEY`). SFT runs don't log to W&B.

Every script accepts `--help`, which also checks that the install works:

```bash
python -m model_training.observer.build_data --help
```

## Steps

All paths below are defaults; each can be changed with its flag. Data goes
under `data/observer/` and models under `models/`.

### 0. Build the training data (GPT-5)

```bash
M="python -m model_training.observer.build_data"

$M add-nodes              # GPT-5 lists each case's nodes (gpt5_nodes)

# S1 answers on the original complaints. The paper ran this command TWICE
# (it appends), giving two answers per case: 1020 rows.
$M s1-responses --field Complaints --input data/observer/case_report_final_all_nodes.jsonl \
   --output data/observer/case_report_final_all_nodes_sft.jsonl

$M augment                # GPT-5 writes complaints for a similar patient (new_complaints)
$M s1-responses           # S1 answers on those paraphrased complaints
$M extract-nodes          # nodes from those answers, as S2 inputs
$M s2-responses           # S2 answers (edges)

$M make-sft --stage s1    # -> sft_training.jsonl     (paper: 5599 rows)
$M make-sft --stage s2    # -> sft_training_s2.jsonl  (paper: 2138 rows)
$M make-grpo              # -> grpo_training.jsonl    (paper: 5089 rows)
```

The GPT-5 steps append to their output files. If one stops partway, rerun it
with `--start <row>` to resume. To start over, delete the output file first.

### 1. SFT S1, then merge onto Qwen3-8B

```bash
torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s1 \
    --output models/Qwen-3-8B-Patient-SFT-lora
python -m model_training.observer.merge --base Qwen/Qwen3-8B \
    --adapter models/Qwen-3-8B-Patient-SFT-lora --output models/Qwen-3-8B-Patient-SFT
```

### 2. GRPO S1, 6 × 100 steps

```bash
G="torchrun --nproc_per_node 4 -m model_training.observer.train_grpo --stage s1"
$G --adapter-out models/Qwen-3-8B-GRPO-100
for s in 200 300 400 500 600; do
  $G --init-adapter models/Qwen-3-8B-GRPO-$((s-100)) --adapter-out models/Qwen-3-8B-GRPO-$s
done
```

### 3. Merge the S1 GRPO adapter onto the SFT model it was trained on

```bash
python -m model_training.observer.merge --base models/Qwen-3-8B-Patient-SFT \
    --adapter models/Qwen-3-8B-GRPO-600 --output models/Qwen-3-8B-GRPO-600-S1-merged
```

### 4. SFT S2, then merge

```bash
torchrun --nproc_per_node 4 -m model_training.observer.sft --stage s2 \
    --base-model models/Qwen-3-8B-GRPO-600-S1-merged --output models/Qwen-3-8B-Patient-SFT-S2-lora
python -m model_training.observer.merge --base models/Qwen-3-8B-GRPO-600-S1-merged \
    --adapter models/Qwen-3-8B-Patient-SFT-S2-lora --output models/Qwen-3-8B-Patient-SFT-S2
```

### 5. GRPO S2, 8 × 100 steps

```bash
G="torchrun --nproc_per_node 4 -m model_training.observer.train_grpo --stage s2"
$G --adapter-out models/Qwen-3-8B-GRPO-s2-100
for s in 200 300 400 500 600 700 800; do
  $G --init-adapter models/Qwen-3-8B-GRPO-s2-$((s-100)) --adapter-out models/Qwen-3-8B-GRPO-s2-$s
done
```

### 6. Merge S2 into the final Observer

```bash
python -m model_training.observer.merge --base models/Qwen-3-8B-Patient-SFT-S2 \
    --adapter models/Qwen-3-8B-GRPO-s2-800 --output models/Angel-Observer --device cpu --rebuild-clean
```

## Build networks with a trained Observer

```bash
python -m model_training.observer.predict_network \
    --input data/examples/observer/case_reports.jsonl --output outputs/observer/networks.jsonl
```

The complaints are read from `--input-field` (default `Complaints`). Cases
that fail go to `--error-output` (default `outputs/observer/network_model_err.jsonl`).
The model comes from `--model-path` or `ANGEL_OBSERVER_MODEL`. The default is
`models/Angel-Observer` if it exists, else the released
[`ChengLi0228/Angel-Observer`](https://huggingface.co/ChengLi0228/Angel-Observer). To build the Actor's input, write to
`data/actor/network_models.jsonl`; see [`model_training/README.md`](../README.md).

## Settings and rewards (paper values)

| | Stage 1 | Stage 2 |
|---|---|---|
| SFT | LoRA r=64 / α=16, lr 1e-4, 3 epochs | same |
| GRPO lr | 5e-6 (cosine, 20 warmup steps) | 1e-5 |
| generations per prompt | 8 | 4 |
| GRPO LoRA | r=16 / α=32 | r=32 / α=64 |
| max prompt / completion length | 2048 / 4096 tokens | 2048 / 4096 tokens |

**Stage 1 reward:** `0.4·format + 0.6·node recall`. Node recall is the share of
the reference nodes (`gpt5_nodes`) that the answer matches at MiniLM cosine
similarity ≥ 0.75, mapped to [-1, 1].

**Stage 2 reward:** `0.1·format + 0.6·precision + 0.3·coverage − size penalty`.
Set the weights with `--w-format`, `--w-precision` and `--w-coverage`.
- **Precision** is the mean plausibility of the proposed edges, in [-1, 1].
  - An LLM judge (gpt-5-mini) scores each edge from 0 to 1.
  - Edges with a node that isn't in the given list, self-loops and duplicates
    score −1 without a judge call.
- **Coverage** is the share of the given nodes joined by a supported edge
  (judge score ≥ 0.5). Without it, one safe edge would score as well as the
  full network.
- **Size penalty** is 0.05 per edge beyond `--max-edges` (10), capped at 0.3.
- **Consistency:** judge scores are cached per (complaints, edge), so every
  generation for a prompt is compared on the same scores.

## Troubleshooting

- **`merge` refuses to save ("did not change the weights"):** the adapter
  learned nothing. On a very small dataset (for example `data/examples/`),
  SFT may take a single optimizer step, which falls in the 3% warmup at
  learning rate 0. Pass a larger `--epochs` (e.g. 10) to `sft`.
- **401 "invalid subscription key" although `.env` is right:** variables
  exported in your shell override `.env`. Check for an old
  `AZURE_OPENAI_API_KEY` in `~/.bashrc`, and `unset` it.
- **`predict_network` says "saved 0 network-model samples":** the model gave
  no valid network for those cases, and they went to `--error-output`. The
  command itself worked. This is expected from a barely trained model, such as
  one trained on `data/examples/`; with `Angel-Observer` it should be rare.
- **Duplicate rows after rerunning step 0:** the GPT-5 steps append. Delete
  the output file first, or resume with `--start`.

## Prompt format

Every stage uses the same prompt: the task's system prompt plus the complaints,
in Qwen3's chat template. That covers SFT, GRPO, `predict_network` and
`model_usage`. The Observer answers with `<think>…</think>` reasoning, then
the `<GRAPH>` JSON. Each stage-1 SFT answer is paired with the text GPT-5
actually answered: the original complaints, or the paraphrase for augmented rows.

## Data

Only synthetic examples are included (`data/examples/observer/`, one file per
pipeline stage). The paper's data come from published case reports, which are
copyrighted:
- 510 case reports with presenting complaints (of PSYCHE's 516);
- 5599 stage-1 and 2138 stage-2 SFT rows;
- 5089 GRPO prompts.
