# model_usage — run the two-stage Angel patient simulator

Angel simulates a mental-health patient in two stages. The Actor always
role-plays the Observer's output:

```
short patient description ──► Observer (stage 1) ──► long profile ──► Actor (stage 2) ──► patient replies
      (free text or a                Qwen3-Observer-800       (structured JSON)      qwen3-8b-dpo-merged
       structured profile)
```

| Stage | Model | Job |
|---|---|---|
| 1. **Observer** | `Qwen3-Observer-800` | Expands a short description into a structured long profile (symptoms, emotions, cognitive and behavior patterns, risk and protective factors, …). |
| 2. **Actor** | `qwen3-8b-dpo-merged` | Role-plays the patient described by that long profile over a multi-turn conversation. |

Each checkpoint is about 16 GB in bf16. By default the Observer is freed after
expansion, before the Actor loads, so one 40 GB GPU is enough. Use `--keep-both`
on an 80 GB GPU when expanding many profiles in a row.

## Setup

```bash
pip install -r model_usage/requirements-usage.txt     # vLLM (default) or torch+transformers
```

Put the weights under `models/` (see the top-level README), or point to them:

```bash
export ANGEL_OBSERVER_MODEL=/path/to/Qwen3-Observer-800
export ANGEL_ACTOR_MODEL=/path/to/qwen3-8b-dpo-merged
python -m angel_common.paths          # shows where each model resolves
```

Models are resolved in this order: `--observer-model` / `--actor-model` flag,
then the environment variable, then `models/<name>`, then a Hugging Face Hub id.
This module needs no API keys.

## Command line

Run every command from the repository root. Add `--backend stub` to try the
plumbing without a GPU; it uses the same code path but returns fake text.

```bash
# free-text note -> Observer -> Actor, interactive
python -m model_usage.angel chat --short-profile-file model_usage/examples/example_short_profile.txt

# a structured profile (also expanded by the Observer first)
python -m model_usage.angel chat --profile-file model_usage/examples/example_custom_profile.json

# one message, or a scripted two-turn demo
python -m model_usage.angel say  --profile-id 0 "Hi, how have you been feeling?"
python -m model_usage.angel demo --profile-id 1 --turns 3

# stage 1 only: save the long profile, then chat with it later without re-running the Observer
python -m model_usage.angel expand --short-profile-file model_usage/examples/example_short_profile.txt \
    --out outputs/mara_long.json --raw-out outputs/mara_observer_raw.json
python -m model_usage.angel chat --profile-file outputs/mara_long.json --no-expand

python -m model_usage.angel --list      # example profiles in model_usage/examples/profiles.jsonl
python -m model_usage.angel --status    # resolved configuration
```

In-chat commands: `/reset` `/history` `/profiles` `/status` `/quit`.

`--no-expand` is only for a file that is *already* an Observer expansion (the
`--out` of `expand`). Everything else goes through the Observer.

## Python API

```python
from model_usage.angel import AngelModel

with AngelModel() as model:
    r = model.send("therapist-1", "Hi, what brings you in today?",
                   short_profile=open("model_usage/examples/example_short_profile.txt").read())
    print(r["reply"])                                   # Actor reply
    print(model.send("therapist-1", "How long has that been going on?")["reply"])
    print(model.history("therapist-1")["history"])
    model.reset("therapist-1")                          # clear transcript, keep patient
```

To keep the long profile itself:

```python
with AngelModel() as model:
    expansion = model.expand_profile("Mara, 34, running on empty since spring.")
    expansion.observer_profile      # the Observer's raw JSON
    expansion.rich_profile          # the same content in the Actor's schema
```

## Inference engines

| `--backend` | Engine | Use when |
|---|---|---|
| `auto` *(default)* | vLLM, falling back to transformers | normal use |
| `vllm` | vLLM | matching the released demo's outputs |
| `hf` | transformers | vLLM is unavailable, or you need `device_map` control |
| `stub` | none | tests and plumbing checks without a GPU |

`ANGEL_VLLM_GPU_MEM` (default `0.4`) sets vLLM's GPU memory share. Raise it on
smaller cards, or use `--backend hf`.

## Prompt styles

The same Actor checkpoint is driven by one of two prompts (`--prompt-style`):

| | `patient_demo` *(default)* | `angel_eval` |
|---|---|---|
| used by | the interactive demo and user study | the paper's profile-expansion experiment |
| system prompt | short roleplay template with the profile rendered as text | built from profile fields, rebuilt each turn |
| dynamic emotional state | starts empty, updated from therapist keywords | seeded from the profile |
| history | full conversation | last 12 turns |

To reproduce the paper's numbers, use `experiments/profile_expansion`. It carries
its own copy of the evaluation actor.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ANGEL_OBSERVER_MODEL` | `models/Qwen3-Observer-800` | stage-1 weights (path or Hub id) |
| `ANGEL_ACTOR_MODEL` | `models/qwen3-8b-dpo-merged` | stage-2 weights (path or Hub id) |
| `ANGEL_BACKEND` | `auto` | `auto` / `vllm` / `hf` / `stub` |
| `ANGEL_DEMO_BACKEND` | `vllm` | engine used by `scripts/gpu_demo.py` |
| `ANGEL_PROMPT_STYLE` | `patient_demo` | `patient_demo` / `angel_eval` |
| `ANGEL_JSONL_PATH` | `model_usage/examples/profiles.jsonl` | profiles for `--profile-id` |
| `ANGEL_KEEP_BOTH` | `0` | keep the Observer loaded after expansion |
| `ANGEL_DEVICE_MAP`, `ANGEL_DTYPE` | `auto`, `bfloat16` | transformers loading |
| `ANGEL_VLLM_GPU_MEM`, `ANGEL_VLLM_MAX_LEN` | `0.4`, `8192` | vLLM engine |
| `ANGEL_OBSERVER_MAX_NEW_TOKENS`, `ANGEL_OBSERVER_TEMPERATURE`, `ANGEL_OBSERVER_TOP_P`, `ANGEL_OBSERVER_MAX_ATTEMPTS`, `ANGEL_OBSERVER_THINKING` | `3072`, `0.7`, `0.9`, `2`, `1` | Observer decoding |
| `ANGEL_ACTOR_MAX_NEW_TOKENS`, `ANGEL_ACTOR_TEMPERATURE`, `ANGEL_ACTOR_TOP_P`, `ANGEL_ACTOR_REPETITION_PENALTY`, `ANGEL_ACTOR_NO_REPEAT_NGRAM`, `ANGEL_ACTOR_MAX_TURNS`, `ANGEL_ACTOR_MAX_RETRIES`, `ANGEL_ACTOR_MAX_SENTENCES` | `90`, `0.8`, `0.9`, `1.15`, `3`, `12`, `3`, `4` | Actor decoding |

The Observer keeps Qwen3's `<think>` block on (`ANGEL_OBSERVER_THINKING=1`),
because it was GRPO-trained with it. The JSON is extracted afterwards.

## Files

| File | Purpose |
|---|---|
| `angel/pipeline.py` | `AngelModel`: the Observer → Actor chain and the session API (`send`, `history`, `reset`, `end`) |
| `angel/observer.py`, `angel/observer_prompts.py` | stage 1: prompts (verbatim from training) and JSON parsing |
| `angel/schema_adapter.py` | maps the Observer's long-profile JSON onto the Actor's profile schema |
| `angel/actor.py`, `angel/demo_prompt.py`, `angel/patient_profile.py`, `angel/state_manager.py`, `angel/length_plan.py`, `angel/postprocess.py` | stage 2: prompting, dynamic state, reply-length planning, output cleanup |
| `angel/backends.py` | vLLM / transformers / stub engines |
| `angel/cli.py` | `python -m model_usage.angel` |
| `examples/` | synthetic example profiles (not real patients) |
| `scripts/gpu_demo.py`, `scripts/gpu_demo.slurm` | readable end-to-end demo on real weights |
| `tests/` | offline unit tests: `python -m unittest discover -s model_usage/tests -t .` |
