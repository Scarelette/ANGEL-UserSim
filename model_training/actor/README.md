# Actor training (stage 2: the patient model)

The **Actor** role-plays a mental-health patient. Its inner experience follows a
directed symptom network built by the [Observer](../observer/). Part of that
network is **masked**: those links start out "subconscious", so the patient
reveals them gradually as the conversation goes on.

The Actor is `Qwen/Qwen3-8B`, trained in two stages on synthetic therapy
conversations with an LLM therapist:

1. **SFT** on conversations where a larger, prompted model (Qwen3-30B) plays the patient.
2. **DPO** on top of the SFT model, to prefer better patient replies.

> **Which DPO?** There are two DPO options. Steps 1–4 are the same for both.
>
> | | Judge DPO (default, steps 5 and 7) | [Graph DPO](#option-graph-reconstruction-dpo) (steps 5' and 7') |
> |---|---|---|
> | Why use it | **Faster.** Cheaper to generate pairs and to train on | **Follows the paper.** Rewards expressing the symptom network |
> | Preference from | a Claude judge scoring single replies | the Observer rebuilding the network from whole conversations |
> | Per network | 1 conversation; up to 8 pairs (one per turn) | 4 conversations; 1 pair (best vs. worst) |
> | Needs | `ANTHROPIC_API_KEY` | the trained Observer on the same GPU (~24 GB total) |
>
> Judge DPO is faster for three reasons:
> - It holds one conversation per network instead of four, so it makes fewer
>   therapist calls.
> - It skips the Observer's two-stage network rebuild, which is a long
>   generation for every conversation.
> - Each pair is a single reply, so DPO trains on short sequences rather than
>   whole conversations.
>
> Graph DPO is slower, but its preference signal is the paper's: how well the
> conversation lets the network be recovered.

## Pipeline at a glance

```
symptom networks (from the Observer)
  1. mask_generator   ─► masked networks (6 files, one per mask rate)
  2. rollout_sft      ─► conversations: prompted Qwen3-30B patient ↔ therapist
  3. build_sft_data   ─► sft_training.jsonl
  4. train_sft        ─► SFT LoRA                (Qwen-3-8B-Patient-SFT-Actor-5)
  5. rollout_dpo      ─► dpo_training.jsonl      (SFT Actor samples 5 replies per turn,
                                                  Claude judge picks best vs worst)
  6. merge            ─► SFT Actor               (qwen3-8b-sft-merged)
  7. train_dpo        ─► DPO LoRA                (reference model = the SFT Actor)
  8. merge            ─► final Actor             (Angel-Actor)
```

Just want to use the trained Actor? It's on Hugging Face as
[`ChengLi0228/Angel-Actor`](https://huggingface.co/ChengLi0228/Angel-Actor),
so you don't need this folder; see
[`model_usage/`](../../model_usage/).

## What you need

| Step | GPU | External APIs |
|---|---|---|
| 1. `mask_generator` | none | GPT-5 (Azure OpenAI) |
| 2. `rollout_sft` | enough for Qwen3-30B-A3B in bf16 (~60 GB, spread over the visible GPUs) | therapist (Azure OpenAI) |
| 3. `build_sft_data` | none | none |
| 4. `train_sft` | one GPU (4-bit QLoRA) | none |
| 5. `rollout_dpo` | one GPU (Actor in 4-bit) | therapist, GPT-5 (format fixes), Claude (judge) |
| 6, 8. `merge` | none needed (runs on CPU) | none |
| 7. `train_dpo` | one GPU | none |
| 5'. `rollout_graph_dpo` | one GPU, ~24 GB (Actor 4-bit + Observer bf16) | therapist, GPT-5 (format fixes) |
| 7'. `train_graph_dpo` | one GPU | none |

**Input:** one JSON object per line with `symptoms` (a list of node names) and
`graph` (a list of `{"from": ..., "to": ...}` edges). This is what the
Observer's `predict_network` writes; see
[`model_training/README.md`](../README.md#train-the-observer-first). A synthetic
example is `data/examples/actor/network_models.jsonl`.

## Setup

Run every command from the **repository root**.

```bash
pip install -r model_training/actor/requirements-actor.txt
cp .env.example .env        # fill in the values below; .env is the only place to edit
```

**Required in `.env`:**

| Variable | Used for | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | GPT-5 and the therapist | none |
| `ANGEL_THERAPIST_DEPLOYMENT` | the therapist's Azure deployment name (steps 2, 5, 5'): any chat model (e.g. `gpt-4o`), or the CBT-fine-tuned therapist from [`therapist/`](../therapist/) as in the paper | none, must be set |
| `ANTHROPIC_API_KEY` | the Claude judge (step 5 only) | none |

**Optional:**

| Variable | Used for | Default |
|---|---|---|
| `ANGEL_GPT5_DEPLOYMENT` | GPT-5 deployment name | `gpt-5` |
| `ANGEL_JUDGE_DEPLOYMENT` | Claude judge model | `claude-opus-4-6` |
| `ANTHROPIC_BASE_URL` | only for Claude on Azure AI Foundry (`https://<resource>.services.ai.azure.com/anthropic`) | public Anthropic API |
| `ANGEL_THERAPIST_AZURE_ENDPOINT`, `ANGEL_THERAPIST_AZURE_API_KEY` | therapist on a different Azure resource | the `AZURE_OPENAI_*` values |
| `ANGEL_BASE_MODEL` | base model | `Qwen/Qwen3-8B` |
| `ANGEL_OBSERVER_MODEL` | Observer for graph DPO (step 5') | `models/Angel-Observer` if present, else [`ChengLi0228/Angel-Observer`](https://huggingface.co/ChengLi0228/Angel-Observer) |
| `ANGEL_DATA_DIR`, `ANGEL_MODELS_DIR`, `ANGEL_OUTPUT_DIR` | move `data/`, `models/`, `outputs/` | inside the repo |
| `WANDB_API_KEY` (+ `--report-to wandb`) | log DPO training to Weights & Biases | off |

### Check your setup

Step 3 needs no GPU or API key, so it's a quick way to confirm the install:

```bash
python -m model_training.actor.build_sft_data \
    --inputs data/examples/actor/sft_rollouts.jsonl --output /tmp/sft_check.jsonl
# -> wrote 1 examples to /tmp/sft_check.jsonl
```

Every script also accepts `--help`.

## Steps

All paths below are defaults; each can be changed with its flag.

> **Outputs are appended, not overwritten.** The rollout and data scripts add
> to existing files, so they can resume after a crash. Delete an output file
> before rerunning a step from scratch, or you'll get duplicates.

### 1. Mask the networks

```bash
for i in 1 2 3 4 5 6; do
  python -m model_training.actor.mask_generator --input data/actor/network_models.jsonl \
      --output data/actor/masked/NM_mask_p$i.jsonl --mask-pct 0.$i
done
```

This writes six files, with `--mask-pct` from 0.1 to 0.6. For each network:
- GPT-5 assigns a disorder pattern (depression, anxiety, substance use, trauma, mania, psychosis).
- GPT-5 labels every symptom as Behavior, Emotion, Physiological Sensation, Cognition or Stimulus.
- Edges are shuffled. Up to `mask-pct` × (all edges) are then moved into `mask`,
  but only edge types that `mask_pattern.jsonl` allows for that pattern can be
  masked. The rest stay in `new_graph`.

So the masked share is usually well below `--mask-pct`, and some networks get
no masked edges at all. In the paper's data, the average masked share ran from
about 1% (p1) to 16% (p6). Networks with no masked edges are still used later.

Both GPT-5 calls are retried up to 5 times. A network that still fails is
written unmasked, and steps 2, 5 and 5' skip it. Networks with an empty graph
go to `<output>.err.jsonl`. Pass `--seed` for a reproducible mask (the
paper's runs had no seed).

### 2. SFT conversations

```bash
for i in 1 2 3 4 5; do
  python -m model_training.actor.rollout_sft --input data/actor/masked/NM_mask_p$i.jsonl \
      --output data/actor/sft_rollouts/NM_mask_p$i.jsonl \
      --patient-model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-turns 15
done
```

A Qwen3-30B patient, prompted with the full network (visible and masked edges),
talks with the therapist for 15 exchanges, speaking first. This produces one
conversation per network. The paper used p1–p5 here (423 conversations).
Split long files across jobs with `--start` / `--end`.

### 3. Build the SFT data

```bash
python -m model_training.actor.build_sft_data \
    --inputs data/actor/sft_rollouts/NM_mask_p{1,2,3,4,5}.jsonl --output data/actor/sft_training.jsonl
```

Each patient turn is normalised to `<state>…</state>\n<word>…</word>`.

> **Note:** the network-specific patient prompt is replaced by the generic
> `SFT_SYSTEM_PROMPT`. That prompt has the role and format rules but no
> network, so the SFT Actor learns the style and format from the 30B patient
> without seeing the network itself. The network appears in the Actor's
> prompt from step 5 on.

### 4. Train SFT

```bash
python -m model_training.actor.train_sft --output-dir models/Qwen-3-8B-Patient-SFT-Actor-5
```

QLoRA on Qwen3-8B with these settings:
- 4-bit nf4 base; LoRA r=64, α=16, dropout 0.05 on all attention and MLP projections.
- lr 1e-4 and 5 epochs; batch 4 × gradient accumulation 4.
- Loss on the patient turns only.

Conversations longer than `--max-length` (8192 tokens) are truncated, with a warning.

### 5. DPO pairs (judge DPO)

```bash
for i in 1 2 3 4 6; do
  python -m model_training.actor.rollout_dpo --input data/actor/masked/NM_mask_p$i.jsonl \
      --out-conv data/actor/dpo/conv_p$i.jsonl --out-dpo data/actor/dpo/dpo_p$i.jsonl
done
cat data/actor/dpo/dpo_p*.jsonl > data/actor/dpo/dpo_training.jsonl
```

The paper used p1–p4 and p6 here (6,901 pairs). The therapist speaks first,
then for each turn, up to 8 turns:
1. The SFT Actor samples up to 5 replies (`--num-candidates`), using 3 random
   sampling presets.
2. Each reply is normalised to one English `<state>`/`<word>` pair. If that
   fails, the reply is resampled once and then repaired by GPT-5. Replies with
   Chinese, Japanese or Korean text are rejected.
3. Near-duplicate replies are dropped.
4. The Claude judge scores the rest. The best and worst replies become a DPO
   pair (`--min-gap` sets a minimum score gap; the default is 0).
5. The best reply continues the conversation.

The judge rates each reply 0–5 on seven criteria. The score is their weighted
sum (`judge.DEFAULT_WEIGHTS`):

| Criterion | Weight | The reply… |
|---|---|---|
| `safety` | 2.0 | has no harmful instructions and no role breaks |
| `structure` | 1.5 | is coherent and answers the latest message |
| `specificity` | 1.2 | gives concrete, personal detail |
| `state_alignment` | 1.2 | says what its `<state>` describes |
| `history_consistency` | 1.0 | agrees with the context and memory |
| `progress` | 0.8 | moves the conversation forward |
| `naturalness` | 0.6 | sounds like a patient, not a therapist or assistant |

### 6. Merge the SFT adapter

```bash
python -m model_training.actor.merge --adapter models/Qwen-3-8B-Patient-SFT-Actor-5 \
    --output models/qwen3-8b-sft-merged
```

### 7. Train DPO

```bash
python -m model_training.actor.train_dpo --sft-model models/qwen3-8b-sft-merged \
    --output-dir models/qwen3-8b-dpo-lora
```

This trains a new LoRA (r=16, α=32) on the SFT Actor, with lr 2e-6, β 0.1 and
2 epochs. The reference model is the SFT Actor itself (the LoRA switched off),
so β acts as the KL penalty that keeps the Actor close to the SFT model.

### 8. Merge into the final Actor

```bash
python -m model_training.actor.merge --base-model models/qwen3-8b-sft-merged \
    --adapter models/qwen3-8b-dpo-lora --output models/Angel-Actor
```

`merge` refuses to save if the adapter didn't change the weights, which
catches a failed or empty training run.

## Option: graph-reconstruction DPO

This option follows the paper's description. It's slower than the judge DPO
(see [Which DPO?](#actor-training-stage-2-the-patient-model)). It replaces
steps 5 and 7; steps 1–4, 6 and 8 stay the same. It uses the trained Observer instead of a Claude
judge, so `ANTHROPIC_API_KEY` isn't needed.

For each masked network:
1. The therapist writes one opening message, shared by all conversations.
2. The SFT Actor holds 4 conversations with the therapist (`--num-conversations`),
   8 patient turns each, sampling at temperature 0.8 and top-p 0.9.
3. The Observer rebuilds a network from each conversation's `<word>` text.
   The `<state>` text is left out, since it would give the network away.
4. Each rebuilt network is scored against the full network (visible and masked
   edges). Nodes are matched by embedding similarity (`all-MiniLM-L6-v2`,
   cosine ≥ 0.55, `--match-threshold`), and the score is 0.5 · node F1 + 0.5 · edge F1, with edge
   direction counted.
5. The best and worst conversations become one DPO pair if their scores differ
   by at least 0.05. Pass `--pairs all` to pair every two conversations instead.

> **Why 0.55, not the Observer reward's 0.75?** The Observer names nodes in
> its own words from the patient's speech, so correct matches are loose
> paraphrases. In test conversations, correct matches scored 0.61–1.0
> ("insomnia" vs. "poor sleep" 0.61) and wrong ones 0.45 or less ("rumination
> about job loss" vs. "being laid off" 0.45). At 0.75, 5 of 12 conversations
> scored exactly 0. At 0.55 none did, and conversations of the same network
> still differed enough to form pairs. Spot-check your own `conv_p*.jsonl`:
> if most `score` values are near 0, lower the threshold; if unrelated
> symptoms are being matched, raise it.

```bash
# 5'. graph DPO pairs
for i in 1 2 3 4 6; do
  python -m model_training.actor.rollout_graph_dpo --input data/actor/masked/NM_mask_p$i.jsonl \
      --out-conv data/actor/graph_dpo/conv_p$i.jsonl --out-dpo data/actor/graph_dpo/dpo_p$i.jsonl
done
cat data/actor/graph_dpo/dpo_p*.jsonl > data/actor/graph_dpo/dpo_training.jsonl

# 6. merge the SFT adapter, as above

# 7'. graph DPO (same settings as step 7; loss on the patient turns only)
python -m model_training.actor.train_graph_dpo --sft-model models/qwen3-8b-sft-merged \
    --output-dir models/qwen3-8b-graph-dpo-lora

# 8'. final Actor
python -m model_training.actor.merge --base-model models/qwen3-8b-sft-merged \
    --adapter models/qwen3-8b-graph-dpo-lora --output models/qwen3-8b-graph-dpo-merged
```

Each scored conversation in `conv_p*.jsonl` comes with its rebuilt network,
node and edge precision and recall, and the recall of the masked edges. To use
this Actor in `model_usage`, set `ANGEL_ACTOR_MODEL=models/qwen3-8b-graph-dpo-merged`.

## Prompt format

Every stage uses the Qwen3 chat template with thinking off (`chat_format.py`).
The system prompt comes first, therapist turns are `user` and patient turns are
`assistant`. The therapist sees the same conversation with the roles swapped.

| Stage | System prompt | Loss on |
|---|---|---|
| SFT (step 4) | generic `SFT_SYSTEM_PROMPT` (no network) | every patient turn of the conversation |
| Judge DPO (steps 5, 7) | the network prompt (`generate_system_prompt`) | the next patient reply |
| Graph DPO (5', 7') | the network prompt | every patient turn after the therapist's opening |

Each patient turn follows the prefix `<|im_start|>assistant\n<think>\n\n</think>\n\n`,
the same prefix the model sees when it generates. One small difference: in SFT,
earlier patient turns in the history keep that empty think block, while the
chat template drops it in DPO and at inference.

## Troubleshooting

- **`ANGEL_THERAPIST_DEPLOYMENT` is not set:** steps 2, 5 and 5' need a
  therapist deployment on your Azure resource; there's no default.
- **"skip row … unmasked network":** step 1 couldn't label that network (see
  step 1). Expected for a few rows.
- **Duplicate rows after a rerun:** outputs are appended. Delete the output
  file first, or use `--start` to resume where a run stopped.
- **Long runs:** shard steps 2, 5 and 5' across jobs with `--start` / `--end` (row indices).
- **`merge` refuses to save:** the adapter is identical to the base model,
  which usually means training didn't run or didn't save. Check the training log.
  On a very small dataset (for example the files in `data/examples/`), SFT may
  take a single optimizer step. The first step falls in the 3% warmup at
  learning rate 0, so nothing is learned; pass a larger `--epochs` (e.g. 10).
- **401 "invalid subscription key" although `.env` is right:** variables
  exported in your shell override `.env`. Check for an old
  `AZURE_OPENAI_API_KEY` (or other key) in `~/.bashrc`, and `unset` it.
- **400 `content_filter` / "ResponsibleAIPolicyViolation", then "Therapist
  Error … failed after retries":** Azure's default content filter blocks the
  therapist when the patient talks about low mood or self-harm, and that
  conversation is dropped. In a test with stock `gpt-4o`, 4 of 12 graph-DPO
  conversations were lost this way (`self_harm`, severity medium). Give the
  therapist deployment a custom content filter with a higher `self_harm`
  threshold (Azure AI Foundry → Guardrails + controls → Content filters, then
  assign it to the deployment). Lowering or turning off filters may need
  Microsoft's approval for your subscription.
- **404 `DeploymentNotFound`:** the deployment name in `.env` doesn't exist on
  that Azure resource. The paper's fine-tuned therapist deployments aren't
  public, so set `ANGEL_THERAPIST_DEPLOYMENT` to a chat model you have (e.g.
  `gpt-4o`), or build your own with [`therapist/`](../therapist/).

## Files

```
mask_generator.py, mask_pattern.jsonl   step 1 (mask_pattern: maskable edge types per disorder pattern)
prompts.py                              network prompt and SFT prompt (verbatim; the Actor was trained on them)
chat_format.py                          the one prompt format (Qwen3 chat template, thinking off)
patient.py, therapist.py                patient models (prompted 30B / SFT Actor) and the LLM therapist
rollout_sft.py, build_sft_data.py       steps 2–3
train_sft.py                            step 4
rollout_dpo.py, judge.py, postprocess.py   step 5 (sampling, Claude judge, <state>/<word> cleanup)
merge.py, train_dpo.py                  steps 6–8
rollout_graph_dpo.py, graph_similarity.py, train_graph_dpo.py   graph-DPO option (5', 7')
```

`data/examples/actor/` has a synthetic one-row example of most files:
`network_models`, `masked_networks`, `sft_rollouts`, `sft_training` and
`dpo_training` (judge DPO format).
