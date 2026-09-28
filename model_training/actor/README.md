# Actor training (stage 2: the patient model)

The Actor role-plays a mental-health patient whose inner experience follows a
directed symptom network. Part of the network is **masked** ("initially
subconscious") so the patient discloses it gradually. Training data are
synthetic therapy conversations generated against an LLM therapist.

```
Observer symptom networks (symptoms + directed edges)
      │  1. mask_generator      GPT-5 labels node types, masks 10–60 % of maskable edges
      ▼
masked networks  ──────────────┬──────────────────────────────────────────┐
      │  2. rollout_sft         │ Qwen3-30B-A3B-Instruct patient ↔ therapist│
      ▼                         │                                          │
SFT rollouts                    │                                          │
      │  3. build_sft_data      │                                          │
      ▼                         │                                          │
sft_training.jsonl (423 conv.)  │                                          │
      │  4. train_sft           │                                          │
      ▼                         ▼                                          │
Qwen-3-8B-Patient-SFT-Actor-5 ──►  5. rollout_dpo  (SFT Actor samples 5 candidates / turn,
      │                              Claude judge picks best vs worst)      │
      │                         dpo_training.jsonl (6,901 pairs)           │
      │                              6. train_dpo  ─► dpo_trl_final (LoRA)  │
      │  7. merge --mode released                                           │
      ▼                                                                     │
qwen3-8b-dpo-merged   (= Qwen3-8B + SFT adapter; see "Known issues")  ◄─────┘
```

All scripts run from the repository root. Credentials come from environment
variables only — see [ENV_VARS.md](ENV_VARS.md). Install with
`pip install -r model_training/actor/requirements-actor.txt`.

## Steps

Paths below are the defaults; every one is an argument. `data/` and `models/`
are relative to the repo root (override with `ANGEL_DATA_DIR`/`ANGEL_MODELS_DIR`).
Synthetic one-row examples of every file format live in
[`data/examples/actor/`](../../data/examples/actor/).

### 1. Mask the symptom networks
Input: one JSON object per case with `symptoms` (list of node strings) and
`grah` (list of `{"from", "to"}` edges) — the Observer's network output
(see `model_training/observer`). Extra keys (`title`, `abstract`,
`complaints`) are passed through.

```bash
for i in 1 2 3 4 5 6; do
  python -m model_training.actor.mask_generator \
      --input data/actor/network_models.jsonl \
      --output data/actor/masked/NM_mask_p$i.jsonl \
      --mask-pct 0.$i
done
```
Output adds `mask` and `new_graph`: lists of
`{"value": {"from", "to"}, "tag": {"parent": <type>, "child": <type>}}`.
Which (parent, child) type pairs may be masked per disorder pattern is defined
in [`mask_pattern.jsonl`](mask_pattern.jsonl).

### 2. SFT rollouts (prompted 30B patient ↔ therapist)
```bash
export ANGEL_THERAPIST_DEPLOYMENT=<your chat deployment>   # paper: "ai_therapist"
for i in 1 2 3 4 5; do
  python -m model_training.actor.rollout_sft \
      --input data/actor/masked/NM_mask_p$i.jsonl \
      --output data/actor/sft_rollouts/NM_mask_p$i.jsonl \
      --patient-model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-turns 15
done
```
The patient speaks first; 15 patient/therapist exchanges per conversation.

### 3. Build SFT examples
```bash
python -m model_training.actor.build_sft_data \
    --inputs data/actor/sft_rollouts/NM_mask_p{1,2,3,4,5}.jsonl \
    --output data/actor/sft_training.jsonl
```
Verified to reproduce the released `sft_training.jsonl` exactly (423/423 rows,
same order) from the original rollout files.

### 4. SFT (QLoRA)
```bash
python -m model_training.actor.train_sft \
    --train-file data/actor/sft_training.jsonl \
    --output-dir models/Qwen-3-8B-Patient-SFT-Actor-5
```
Qwen3-8B in 4-bit nf4, LoRA r=64 / α=16 / dropout 0.05 on q,k,v,o,gate,up,down;
lr 1e-4, 5 epochs, batch 4 × grad-accum 4, paged AdamW 32-bit, warmup 3 %, bf16.

### 5. DPO rollouts (SFT Actor candidates judged by Claude)
```bash
export ANGEL_THERAPIST_DEPLOYMENT=<your chat deployment>   # paper: "ai_therapist_2"
for i in 1 2 3; do
  python -m model_training.actor.rollout_dpo \
      --input data/actor/masked/NM_mask_p$i.jsonl \
      --out-conv data/actor/dpo/out_conversations_p$i.jsonl \
      --out-dpo  data/actor/dpo/out_dpo_p$i.jsonl \
      --sft-adapter models/Qwen-3-8B-Patient-SFT-Actor-5
done
# the paper also used small runs on p4, p6 and a first p1 test run (out_dpo_1)
cat data/actor/dpo/out_dpo_1.jsonl data/actor/dpo/out_dpo_p{1,2,3,4,6}.jsonl \
    > data/actor/dpo/dpo_training.jsonl
```
Per therapist turn: 3 random sampling presets × 2 samples → up to 5
candidates, each normalized to one `<state>/<word>` pair (resample at T=0.2,
then GPT-5 format fix), simhash de-duplicated, scored by
`claude-opus-4-6` (weighted rubric in `judge.py`), best vs worst written as a
pair; the best reply continues the conversation. Max 8 turns. Shard long
files with `--start/--end`. The released `dpo_training.jsonl` is exactly the
concatenation above in that order (verified).

### 6. DPO
```bash
python -m model_training.actor.train_dpo \
    --train-file data/actor/dpo/dpo_training.jsonl \
    --sft-adapter models/Qwen-3-8B-Patient-SFT-Actor-5 \
    --output-dir models/dpo_trl_final
```
4-bit policy and reference; new LoRA r=16 / α=32 / dropout 0.05; lr 2e-6
(linear), β=0.1 sigmoid loss, batch 2 × grad-accum 8, 2 epochs (864 steps),
max length 4096 / prompt 2048, bf16, seed 42. Reference = base + SFT adapter.
Prompts are the `context_messages` rendered with the Qwen3 chat template.

### 7. Merge → `qwen3-8b-dpo-merged`
```bash
python -m model_training.actor.merge --mode released \
    --sft-adapter models/Qwen-3-8B-Patient-SFT-Actor-5 \
    --output models/qwen3-8b-dpo-merged
```
`--mode released` reproduces the published checkpoint (float16). `--mode
policy` merges `dpo_trl_final` instead, `--mode stacked` merges SFT + DPO for an
adapter trained with `train_dpo --init stack-on-sft`.

## Lineage of the released `qwen3-8b-dpo-merged`

| Artifact | Produced by | Evidence |
|---|---|---|
| masked networks `NM_800_mask_p1…p6` | `mask_generator` @ d7cdae1, `--mask-pct 0.1…0.6` | max mask ratio per file = 0.1 … 0.6 |
| SFT rollouts `sft_qwen30b/NM_800_mask_p1…p5` | arena (uncommitted, 2026-02-22) with Qwen3-30B-A3B-Instruct-2507, 15 turns | logs `actor_sft_p*`; roles/turn counts; lowercase replies from the 3865c34 `_clean_reply` |
| `sft_training.jsonl` (423) | `sft_data_process.py` @ c11ad52 | reproduced exactly by `build_sft_data` |
| `Qwen-3-8B-Patient-SFT-Actor-5` | `GRPO-Qwen3/Finetune/finetune_hf.py` @ 4ab9b7e | adapter_config (r64/α16), output name, data path |
| `dpo_training.jsonl` (6,901) | `arena_new.py` + `build_dpo_from_graphs.py` @ 09aa197/652db58 | exact concatenation of `out_dpo_*` |
| `dpo_trl_final` | `actor/dpo/train_dpo_2.py` @ 652db58 | W&B run `51bolf11` (git commit recorded, 864/864 steps); `training_args.bin` |
| **`qwen3-8b-dpo-merged`** | `actor/dpo/merge_model.py` @ 4b7ebbb | weights = fp16(Qwen3-8B + SFT delta) exactly; projection on DPO delta = 0.000 |

## Provenance (original → this module)

| Original (research repos) | Here |
|---|---|
| `actor/sys_prompt.py::generate_system_prompt`, `actor/sft_data_process.py` prompt, `actor/therapist_azure.py` prompt | `prompts.py` (verbatim) |
| `actor/mask_generator.py` | `mask_generator.py` |
| `data/network_model/mask_pattern.jsonl` (`actor/write_jsonl.py`) | `mask_pattern.jsonl` |
| `actor/patient_qwen.py` @ 3865c34 (SFT) / @ 652db58 (DPO) | `patient.py` (`PromptedPatient` / `AdapterPatient`) |
| `actor/therapist_azure.py` | `therapist.py` |
| `actor/arena.py` (SFT rollouts) | `rollout_sft.py` |
| `actor/sft_data_process.py` | `build_sft_data.py` |
| `GRPO-Qwen3/Finetune/finetune_hf.py` @ 4ab9b7e | `train_sft.py` |
| `actor/arena_new.py` + `actor/scripts/build_dpo_from_graphs.py` | `rollout_dpo.py` |
| `actor/claude_judge.py` + `actor/dpo_utils.py` | `judge.py` |
| `actor/patient_output_postprocess.py` | `postprocess.py` (verbatim) |
| `actor/dpo/train_dpo_2.py` @ 652db58 | `train_dpo.py` |
| `actor/dpo/merge_model.py` @ 4b7ebbb | `merge.py` |

Not ported: `arena.py` @ HEAD (a 2-turn variant written after the SFT run;
`arena_new.py` is the DPO version), `sys_prompt.generate_system_prompt_GPT`
and `patient_gpt5.py` (unused GPT-5 patient), `train_dpo_mask_kl.py`
(superseded draft), `claude_test.py`, `write_jsonl.py`,
`simulate_patient/finetune.py` (unrelated early script). **Not part of the
release:** the "stage-2" DPO scripts `train_dpo_3.py` / `train_dpo_4.py`
(trained on top of `qwen3-8b-dpo-merged`, producing
`qwen3-8b-dpo-stage2-lora-*`) and the later `Qwen-3-8B-Patient-SFT-final`
(eeyore_depression_sft on top of the merged model) — ask the authors if you
need them.

## Known issues

These are kept as in the original runs so the released artifacts can be
reproduced; flags are given where a fix is available.

1. **The released Actor contains no DPO.** `merge_model.py` loaded the DPO
   adapter with `load_adapter(..., adapter_name="dpo")` but
   `merge_and_unload()` merges only the active adapter (the SFT "default").
   `qwen3-8b-dpo-merged` is exactly Qwen3-8B + SFT LoRA in fp16. Use
   `merge --mode policy|stacked` for a DPO model.
2. **The DPO run did not start from the SFT model.** Wrapping the SFT
   `PeftModel` in a second `get_peft_model` with the same adapter name
   re-initializes the LoRA layers in place (reproduced with peft 0.18.1): the
   policy started from plain Qwen3-8B while the reference was base + SFT
   (step-1 `rewards/chosen` = −4.93 instead of ≈0). `train_dpo --init
   stack-on-sft` trains a separate "dpo" adapter over the frozen SFT one
   (checked on a tiny model only, not re-run at scale).
3. **Prompt format mismatch.** Rollouts render the dialogue as plain text
   (`therapist: … / patient: …`), SFT uses a `### System/### User/###
   Assistant` string built from the *last* user/assistant turn only, while
   the dataset keeps its `messages` column — with trl 0.27.1 (checked in
   `sft_trainer.py`) `SFTTrainer` then treats rows as conversational and
   applies the chat template to the full conversation, ignoring `text`.
   Because a plain `TrainingArguments` is passed, SFT also inherits
   `SFTConfig.max_length=1024`, so long conversations are truncated, and the
   LM collator puts loss on all tokens (not assistant-only). DPO uses the chat
   template. Pin the versions in `requirements-actor.txt` to reproduce.
4. **SFT rollouts stopped early.** The original rollout crashed at the first
   network that `mask_generator` had left unmasked (untagged edges), so only
   the first ~85 networks of each file were used (423 total). `rollout_sft`
   now skips such rows; use `--end` to cap rows if you want the same subset.
5. `mask_generator`: networks where GPT-5 returns fewer type labels than
   symptoms, an unknown pattern, or no `<type>` tag are kept unmasked; the
   category list omits "Physiological Sensation" when building the patient
   prompt (`generate_system_prompt` only lists Cognition/Emotion/Behavior/
   Stimulus); sampling is unseeded (`--seed` added).
6. `PromptedPatient._clean_reply` lowercases a reply whenever it cuts role
   leakage (~30 % of SFT patient turns are lowercase).
7. DPO rollouts pass OpenAI-style roles straight through to the therapist,
   so the therapist model sees its own turns as `user` turns
   (`--therapist-fix-roles` flips them).
8. ~1.6 % of released DPO pairs (113 chosen / 114 rejected) contain Chinese
   text, introduced by the GPT-5 format-fix fallback. Filter if undesired.
9. The judge only varies `structure`, `specificity` and the
   entailment-derived fields; the other rubric entries are constants.
