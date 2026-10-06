# Talk to a simulated patient

Describe a patient in a few sentences, then chat with them as their therapist.

Angel works in two steps:

1. The **Observer** reads your short description and writes a detailed patient
   profile: symptoms, feelings, thought patterns, behaviours, background and
   risks.
2. The **Actor** role-plays that patient. It answers in short, natural replies
   and opens up gradually.

```
your description ──► Observer ──► detailed profile ──► Actor ──► patient replies
```

## Quick start

**1. Install** (from the repository root):

```bash
pip install -r model_usage/requirements-usage.txt
```

**2. Get the two models** (about 16 GB each). You can skip this step: if
they aren't found locally, Angel downloads them from Hugging Face on first use:

| Model | Hugging Face | Local folder |
|---|---|---|
| Observer | [`ChengLi0228/Angel-Observer`](https://huggingface.co/ChengLi0228/Angel-Observer) | `models/Angel-Observer/` |
| Actor | [`ChengLi0228/Angel-Actor`](https://huggingface.co/ChengLi0228/Angel-Actor) | `models/Angel-Actor/` |

To keep a local copy in the repository's `models/` folder:

```bash
hf download ChengLi0228/Angel-Observer --local-dir models/Angel-Observer
hf download ChengLi0228/Angel-Actor    --local-dir models/Angel-Actor
```

If they are somewhere else, tell Angel where. Either add these two lines to
`.env` in the repository root:

```
ANGEL_OBSERVER_MODEL=/path/to/Angel-Observer
ANGEL_ACTOR_MODEL=/path/to/Angel-Actor
```

or export them in your shell. `export` is needed; without it Python does not
see the variables:

```bash
export ANGEL_OBSERVER_MODEL=/path/to/Angel-Observer
export ANGEL_ACTOR_MODEL=/path/to/Angel-Actor
```

**3. Use a machine with a GPU** (40 GB or more). On a Slurm cluster, get a GPU
node first, for example:

```bash
srun --gres=gpu:1 --mem=64G --time=01:00:00 --pty bash
```

**4. Start chatting:**

```bash
python -m model_usage.angel chat --short-profile-file model_usage/examples/example_short_profile.txt
```

The first reply takes a few minutes while the models load; after that, replies
are quick. You type the therapist's lines, for example (replies vary from run
to run):

```
[you] Hi, what brings you in today?
[patient] Mostly tired, I think. It's been a long few weeks.
[you] What has been making it hard?
[patient] Work, mostly. There was a restructure in the spring and I just... haven't caught up.
```

Commands during the chat:

| Command | What it does |
|---|---|
| `/history` | show the conversation so far |
| `/reset` | start over with the same patient |
| `/quit` | end the chat |

No GPU yet? Add `--backend stub` to try the commands with canned replies.

## Your own patient

Write a few sentences in a text file, in plain language: who the patient is,
what brought them in, how it affects their life, and how they come across. The
example (`model_usage/examples/example_short_profile.txt`) is:

> Mara is 34 and came in after her GP suggested it. She says she has been
> "running on empty" since a restructure at work in the spring, sleeps badly, and
> has stopped seeing the friends she used to climb with. She is polite and a
> little dismissive about how much it is affecting her, and changes the subject
> when her older brother comes up.

Then:

```bash
python -m model_usage.angel chat --short-profile-file my_patient.txt
```

For a one-line description, use `--short-profile "..."` instead of a file.

## Other things you can do

**Send one message and exit:**

```bash
python -m model_usage.angel say --short-profile-file my_patient.txt "Hi, how have you been?"
```

**Save the detailed profile and reuse it later.** The Observer's profile
differs a little each run, so saving it keeps the same patient across sessions:

```bash
python -m model_usage.angel expand --short-profile-file my_patient.txt --out my_patient_profile.json
python -m model_usage.angel chat --profile-file my_patient_profile.json --no-expand
```

**Use it from Python:**

```python
from model_usage.angel import AngelModel

with AngelModel() as model:
    note = open("model_usage/examples/example_short_profile.txt").read()
    print(model.send("me", "Hi, what brings you in today?", short_profile=note)["reply"])
    print(model.send("me", "How long has that been going on?")["reply"])
```

`python -m model_usage.angel --help` lists every option.

## Troubleshooting

**`observer model not found` or `actor model not found`.** Angel can't find
the models. Check the paths from step 2; if you set them in the shell, make sure
you used `export`.

**`` version `CXXABI_1.3.15' not found ``.** The system's C++ library is
shadowing your conda environment's. Run this, then try again:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

If you leave it, Angel still works: it falls back to a slower engine and prints
a warning.

**Out of GPU memory.** Let vLLM use more of the GPU with
`export ANGEL_VLLM_GPU_MEM=0.8`, or switch engines with `--backend hf`.

**`no GPU found`** (older versions: `Device string must not be empty`). You are
on a login node or a CPU-only machine; see step 3. `--backend hf` forces a CPU
run, which works but is very slow.

## Settings

The patient's prompt and generation settings are fixed. You can only choose
the models and the engine, either in `.env` or with `export`:

| Setting | Default | |
|---|---|---|
| `ANGEL_OBSERVER_MODEL` | `models/Angel-Observer` if present, else `ChengLi0228/Angel-Observer` | Observer model folder (or Hugging Face id) |
| `ANGEL_ACTOR_MODEL` | `models/Angel-Actor` if present, else `ChengLi0228/Angel-Actor` | Actor model folder (or Hugging Face id) |
| `ANGEL_BACKEND` | `auto` | `auto` uses vLLM if it works, else transformers; also `vllm`, `hf`, `stub` |
| `ANGEL_VLLM_GPU_MEM` | `0.4` | share of GPU memory vLLM may use |

## What's in this folder

| Path | Contents |
|---|---|
| `angel/` | the code: `pipeline.py` (Observer → Actor), `observer*.py`, `actor.py`, `cli.py`, … |
| `examples/` | example patients (made up, not real people) |
| `scripts/gpu_demo.py` | a scripted demo conversation on the real models |
| `tests/` | tests that run without a GPU: `python -m unittest discover -s model_usage/tests -t .` |
