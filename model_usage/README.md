# model_usage — run the Angel patient simulator

Angel simulates a mental-health patient in two stages. The Actor always
role-plays the Observer's output:

```
short patient description ──► Observer (stage 1) ──► long profile ──► Actor (stage 2) ──► patient replies
                              Qwen3-Observer-800      (structured JSON)   qwen3-8b-dpo-merged
```

| Stage | Model | Job |
|---|---|---|
| 1. **Observer** | `Qwen3-Observer-800` | Expands a short description into a structured long profile (symptoms, emotions, cognitive and behavior patterns, risk and protective factors, …). |
| 2. **Actor** | `qwen3-8b-dpo-merged` | Role-plays the patient described by that long profile over a multi-turn conversation. |

Each checkpoint is about 16 GB in bf16. The Observer is freed before the Actor
loads, so one 40 GB GPU is enough.

## Setup

```bash
pip install -r model_usage/requirements-usage.txt
```

Put the weights under `models/` (see the top-level README), or point to them,
either in `<repo>/.env`:

```
ANGEL_OBSERVER_MODEL=/path/to/Qwen3-Observer-800
ANGEL_ACTOR_MODEL=/path/to/qwen3-8b-dpo-merged
```

or in your shell (note the `export`; without it Python does not see them):

```bash
export ANGEL_OBSERVER_MODEL=/path/to/Qwen3-Observer-800
export ANGEL_ACTOR_MODEL=/path/to/qwen3-8b-dpo-merged
```

Run on a machine with a GPU.

No API keys are needed.

## Talk to a patient

Write a short free-text description of the patient, then chat. The Observer
expands the description, and the Actor answers as that patient:

```bash
python -m model_usage.angel chat --short-profile-file model_usage/examples/example_short_profile.txt
```

Run it from the repository root. You type the therapist's messages. In-chat
commands: `/history`, `/reset` (start over with the same patient), `/quit`.

Add `--backend stub` to try it without a GPU; it runs the same code with
canned replies.

Other commands:
- `say` sends a single message.
- `expand` runs only the Observer and saves the long profile.
- `--list` shows the example profiles.
- `--help` lists all commands.

## Python

```python
from model_usage.angel import AngelModel

with AngelModel() as model:
    note = open("model_usage/examples/example_short_profile.txt").read()
    print(model.send("me", "Hi, what brings you in today?", short_profile=note)["reply"])
    print(model.send("me", "How long has that been going on?")["reply"])
    print(model.history("me")["history"])
```

`model.expand_profile(note)` returns the Observer's long profile on its own.

## Configuration

The Observer and Actor always use the released demo's prompt and generation
settings. The settings below live in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `ANGEL_OBSERVER_MODEL`, `ANGEL_ACTOR_MODEL` | `models/Qwen3-Observer-800`, `models/qwen3-8b-dpo-merged` | model weights (path or Hugging Face id) |
| `ANGEL_BACKEND` | `auto` | `auto` (vLLM, else transformers), `vllm`, `hf`, or `stub` |
| `ANGEL_VLLM_GPU_MEM` | `0.4` | share of GPU memory vLLM may use; raise it on smaller GPUs |

## Files

| File | Purpose |
|---|---|
| `angel/pipeline.py` | `AngelModel`: the Observer → Actor chain and the conversation API |
| `angel/observer.py`, `angel/observer_prompts.py`, `angel/schema_adapter.py` | stage 1: prompts, JSON parsing, mapping the long profile onto the Actor's schema |
| `angel/actor.py`, `angel/demo_prompt.py`, `angel/state_manager.py`, `angel/length_plan.py` | stage 2: prompt, emotional state, reply length and cleanup |
| `angel/backends.py`, `angel/config.py`, `angel/cli.py` | engines, settings, command line |
| `examples/` | synthetic example profiles (not real patients) |
| `scripts/gpu_demo.py` | scripted end-to-end demo on the real weights |
| `tests/` | offline tests: `python -m unittest discover -s model_usage/tests -t .` |
