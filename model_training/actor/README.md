# Actor training (stage 2: the patient model)

The Actor role-plays a mental-health patient whose inner experience follows a
directed symptom network. Part of the network is **masked** (initially
"subconscious"), so the patient reveals it gradually. Training data are
synthetic therapy conversations with an LLM therapist. The Actor is trained
with SFT, then DPO on top of the SFT model:

```
symptom networks (from the Observer)
  1. mask_generator   ─► masked networks
  2. rollout_sft      ─► conversations: prompted Qwen3-30B patient ↔ therapist
  3. build_sft_data   ─► sft_training.jsonl
  4. train_sft        ─► SFT LoRA (Qwen-3-8B-Patient-SFT-Actor-5)
  5. rollout_dpo      ─► dpo_training.jsonl: the SFT Actor samples 5 replies per turn,
                         a Claude judge picks best vs worst
  6. merge            ─► qwen3-8b-sft-merged   (SFT Actor)
  7. train_dpo        ─► DPO LoRA, trained on the SFT Actor (also the reference)
  8. merge            ─► qwen3-8b-dpo-merged   (final Actor = Qwen3-8B + SFT + DPO)
```

## Setup

Run everything from the repository root.

```bash
pip install -r model_training/actor/requirements-actor.txt
cp .env.example .env        # then fill in the values below — the only place to edit
```

| Variable | Needed for | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | GPT-5 (steps 1, 5) and the therapist (steps 2, 5) | — |
| `ANGEL_GPT5_DEPLOYMENT` | GPT-5: node typing, disorder pattern, reply format fix | `gpt-5` |
| `ANGEL_THERAPIST_DEPLOYMENT` | therapist chat model (steps 2, 5) | — (required) |
| `ANTHROPIC_API_KEY` | Claude judge (step 5) | — |
| `ANTHROPIC_BASE_URL` | only for Azure AI Foundry (`https://<resource>.services.ai.azure.com/anthropic`) | public Anthropic API |
| `ANGEL_JUDGE_DEPLOYMENT` | Claude judge model | `claude-opus-4-6` |
| `ANGEL_BASE_MODEL` | base model | `Qwen/Qwen3-8B` |

Optional:
- `ANGEL_THERAPIST_AZURE_ENDPOINT` / `_API_KEY` if the therapist is on a different Azure resource.
- `ANGEL_DATA_DIR` / `ANGEL_MODELS_DIR` to move `data/` and `models/`.
- W&B logging for DPO with `--report-to wandb` (plus `WANDB_API_KEY`).

**Input:** one JSON object per case with `symptoms` (node names) and `graph`
(`{"from", "to"}` edges): the Observer's network output (see
[`model_training/observer`](../observer/)). Synthetic one-row
examples of every file in the pipeline are in `data/examples/actor/`.

## Steps

Every path below is a default and can be changed with its flag.

```bash
# 1. mask 10–60 % of the maskable edges (one file per mask rate)
for i in 1 2 3 4 5 6; do
  python -m model_training.actor.mask_generator --input data/actor/network_models.jsonl \
      --output data/actor/masked/NM_mask_p$i.jsonl --mask-pct 0.$i
done

# 2. SFT conversations: prompted 30B patient ↔ therapist, 15 exchanges each
#    (networks mask_generator left unmasked are skipped; add --on-unmasked stop to
#    end each file at the first one, as the paper's run did)
for i in 1 2 3 4 5; do
  python -m model_training.actor.rollout_sft --input data/actor/masked/NM_mask_p$i.jsonl \
      --output data/actor/sft_rollouts/NM_mask_p$i.jsonl \
      --patient-model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-turns 15
done

# 3. SFT examples (with --on-unmasked stop in step 2: the paper's 423-conversation file)
python -m model_training.actor.build_sft_data \
    --inputs data/actor/sft_rollouts/NM_mask_p{1,2,3,4,5}.jsonl --output data/actor/sft_training.jsonl

# 4. SFT: QLoRA r=64 / α=16, lr 1e-4, 5 epochs, loss on patient turns
python -m model_training.actor.train_sft --output-dir models/Qwen-3-8B-Patient-SFT-Actor-5

# 5. DPO pairs from the SFT Actor (paper: p1–p4, p6 and an earlier p1 test run = 6,901 pairs;
#    shard long files with --start/--end)
for i in 1 2 3 4 6; do
  python -m model_training.actor.rollout_dpo --input data/actor/masked/NM_mask_p$i.jsonl \
      --out-conv data/actor/dpo/conv_p$i.jsonl --out-dpo data/actor/dpo/dpo_p$i.jsonl
done
cat data/actor/dpo/dpo_p*.jsonl > data/actor/dpo/dpo_training.jsonl

# 6. SFT Actor
python -m model_training.actor.merge --adapter models/Qwen-3-8B-Patient-SFT-Actor-5 \
    --output models/qwen3-8b-sft-merged

# 7. DPO: LoRA r=16 / α=32, lr 2e-6, β 0.1, 2 epochs; reference = the SFT Actor
python -m model_training.actor.train_dpo --sft-model models/qwen3-8b-sft-merged \
    --output-dir models/qwen3-8b-dpo-lora

# 8. final Actor
python -m model_training.actor.merge --base-model models/qwen3-8b-sft-merged \
    --adapter models/qwen3-8b-dpo-lora --output models/qwen3-8b-dpo-merged
```

`merge` refuses to save if an adapter did not change the weights.

In step 5, each therapist turn works like this:
1. The SFT Actor samples up to 5 replies, using 3 random sampling presets.
2. Each reply is normalised to one `<state>`/`<word>` pair; if that fails, it is resampled, then repaired by GPT-5.
3. Near-duplicate replies are dropped.
4. The Claude judge scores the rest; the best and worst become a DPO pair.
5. The best reply continues the conversation, for up to 8 turns.

## Prompt format

Every stage uses the same format: the Qwen3 chat template with thinking off
(`chat_format.py`). The system prompt comes first, then therapist turns as
`user` and patient turns as `assistant`. This holds for the rollout patients,
SFT, DPO and `model_usage`.
- **SFT** trains on whole conversations, with loss on the patient turns only.
  Each patient turn follows the same `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  prefix the model sees when it generates (`--max-length`, default 8192 tokens).
  One difference: in the conversation history, earlier patient turns keep that
  empty think block in SFT, while the chat template drops it at inference and
  in DPO.
- **DPO** prompts are the conversation so far plus that prefix, as in the rollouts.

## Differences from the paper's run

- **Prompt format.** The paper's Actor was trained before the format above was
  unified: its rollouts used a plain-text prompt and its SFT used TRL's default
  chat formatting with loss on all tokens.
- **SFT rollouts.** The paper's run stopped each file at the first network that
  `mask_generator` left unmasked, giving about 85 networks per file (423 total).
  `rollout_sft` now skips those networks; `--on-unmasked stop` reproduces the
  paper's run.
- **Masking.** The paper's run took GPT-5's first reply, so a network stayed
  unmasked when GPT-5 returned too few type labels or an unknown pattern.
  `mask_generator` now retries until both are valid; `--paper-compat`
  reproduces the paper's run.
- **Patient prompt.** The paper's patient prompt left "Physiological Sensation"
  out of its state categories, so those states were never listed.
  `rollout_sft` and `rollout_dpo` now include them; `--paper-prompt` reproduces
  the paper's run.

## Known issues

These are kept as in the paper's runs.

1. **Lowercasing.** About 30 % of SFT patient turns are lowercase: the reply
   cleaner lowercases a reply when it removes role leakage.
2. **Therapist roles.** In DPO rollouts the therapist sees its own turns as
   `user` turns; `--therapist-fix-roles` flips them.
3. **Chinese text.** About 1.6 % of DPO pairs contain Chinese text from the
   GPT-5 format fix.
4. **Judge rubric.** Only `structure`, `specificity` and the entailment fields
   of the judge's rubric vary; the others are constants.

## Files

```
mask_generator.py, mask_pattern.jsonl   step 1 (which edge types may be masked per disorder pattern)
prompts.py                              patient prompt from a symptom network (verbatim)
chat_format.py                          the one prompt format (Qwen3 chat template, thinking off)
patient.py, therapist.py                patient models (prompted 30B / SFT Actor) and LLM therapist
rollout_sft.py, build_sft_data.py       steps 2–3
train_sft.py                            step 4
rollout_dpo.py, judge.py, postprocess.py   step 5 (candidate sampling, Claude judge, <state>/<word> cleanup)
merge.py, train_dpo.py                  steps 6–8
```
