# ANGEL-UserSim: simulated mental-health patients

[![Live demo](https://img.shields.io/badge/%F0%9F%8E%AE_Live_demo-try_it_now-ff5c8a?style=for-the-badge)](https://eval.angel-simulation.org/demo)
[![Angel-Observer](https://img.shields.io/badge/%F0%9F%A4%97_Angel--Observer-Hugging_Face-ffcc4d?style=for-the-badge)](https://huggingface.co/ChengLi0228/Angel-Observer)
[![Angel-Actor](https://img.shields.io/badge/%F0%9F%A4%97_Angel--Actor-Hugging_Face-ffcc4d?style=for-the-badge)](https://huggingface.co/ChengLi0228/Angel-Actor)
[![PSYCHE dataset](https://img.shields.io/badge/%F0%9F%93%9A_PSYCHE_dataset-coming_soon-8a7dff?style=for-the-badge)](#-psyche-dataset-coming-soon)

> ### 🛋️ You're the therapist. Your patient is waiting.
>
> **[▶ Open the live demo: eval.angel-simulation.org/demo](https://eval.angel-simulation.org/demo)**. No install, no GPU, no sign-up.
>
> 1. **🧑 Meet a patient.** Pick one of 330 curated profiles, or write your own (an AI assistant helps you fill it in).
> 2. **🕸️ See inside their head.** Angel turns the profile into a **symptom network**: which thoughts, feelings and events set off which.
> 3. **💬 Start the session.** Type your first question. The patient answers in character, holds things back, and opens up as you earn their trust. Turn on 🎙️ voice to hear them.
>
> **🎯 Try this:** every patient keeps some core beliefs and concerns to themselves at first. Can you ask the questions that bring them out?
>
> <sub>Research preview: every patient is simulated and nothing here is therapy. Please don't enter real personal or health information.</sub>

Angel simulates a psychotherapy patient in two steps:

1. The **Observer** (`Angel-Observer`, Qwen3-8B trained with SFT + GRPO)
   reads a short patient description and writes a detailed **long profile**,
   organised around the patient's symptom network.
2. The **Actor** (`Angel-Actor`, Qwen3-8B trained with SFT + DPO)
   **role-plays the patient in that long profile** across a multi-turn therapy
   conversation, revealing sensitive material gradually.

```
short description ──► Observer ──► long profile ──► Actor ──► patient replies
```

## Quick start: talk to a patient

Just want to try it? Use the **[live demo](https://eval.angel-simulation.org/demo)**.
To run Angel yourself, you need a GPU (40 GB or more). The two models download from Hugging Face on
first use (about 16 GB each; see [Models](#models)).

```bash
git clone https://github.com/Scarelette/ANGEL-UserSim.git && cd ANGEL-UserSim
pip install -r model_usage/requirements-usage.txt

python -m model_usage.angel chat --short-profile-file model_usage/examples/example_short_profile.txt
```

You type the therapist's messages and the patient answers. See
[`model_usage/README.md`](model_usage/README.md) for writing your own patient,
the Python API and troubleshooting. No GPU yet? Add `--backend stub` to try the
commands with canned replies.

## What's in the repository

| Folder | What it's for |
|---|---|
| [`model_usage/`](model_usage/) | **Use Angel**: short description → Observer → long profile → Actor, as a chat or from Python. No API keys needed. |
| [`model_training/`](model_training/) | **Train both models**: which to train first (the Observer) and how its output feeds the Actor. |
| [`model_training/observer/`](model_training/observer/) | Train the Observer: data building, SFT, and GRPO for symptom nodes (stage 1) and edges (stage 2, rewarded by a gpt-5-mini judge). |
| [`model_training/actor/`](model_training/actor/) | Train the Actor: masked symptom networks, conversations with an LLM therapist, SFT, then DPO on top of the SFT model. |
| [`experiments/profile_expansion/`](experiments/profile_expansion/) | The main experiment: Angel vs. Patient-Psi, Roleplay-doh, Eeyore and a one-stage ablation in a fixed intake interview. Measures simulation diversity, behavior diversity and profile alignment. |
| [`experiments/safety_exp/`](experiments/safety_exp/) | The safety experiment: how commercial chatbots respond to a user (played by Angel) whose beliefs escalate toward delusion, scored with a clinical codebook. |
| `angel_common/` | Shared code: model lookup, settings from `.env`, API clients. |
| `data/examples/` | Small **synthetic** examples of every input file. No real patient data. |
| `scripts/check_secrets.py` | Checks the repository for keys and machine-specific paths before you publish. |

Each folder has its own `README.md` with the exact commands and the settings it
needs, and its own `requirements-*.txt`. Run every command from the repository
root.

## Models

| Model | Folder name | Size | Base | Hugging Face |
|---|---|---|---|---|
| Observer | `Angel-Observer` | 16 GB (bf16) | Qwen/Qwen3-8B | [`ChengLi0228/Angel-Observer`](https://huggingface.co/ChengLi0228/Angel-Observer) |
| Actor | `Angel-Actor` | 16 GB (bf16) | Qwen/Qwen3-8B | [`ChengLi0228/Angel-Actor`](https://huggingface.co/ChengLi0228/Angel-Actor) |

Angel looks for each model in this order:
1. the command-line flag (`--observer-model`, `--actor-model`);
2. the environment variable (`ANGEL_OBSERVER_MODEL`, `ANGEL_ACTOR_MODEL`);
3. the `models/<folder name>` folder in the repository;
4. the Hugging Face id, downloaded on first use into the Hugging Face cache
   (`~/.cache/huggingface`, or `$HF_HOME`).

Earlier versions called the folders `Qwen3-Observer-800` and
`qwen3-8b-dpo-merged`; rename an old copy, or point the variables at it.

`python -m angel_common.paths` shows where each model is found. To keep a
local copy instead, download it into `models/`:

```bash
hf download ChengLi0228/Angel-Observer --local-dir models/Angel-Observer
hf download ChengLi0228/Angel-Actor    --local-dir models/Angel-Actor
```

## Settings and API keys

Nothing secret is stored in the code. All settings live in **one file**,
`.env`, at the repository root:

```bash
cp .env.example .env     # fill in what you need; every script reads it
```

`.env` is gitignored. A variable exported in your shell overrides it. What each
part needs:
- **`model_usage`:** nothing (only the model paths, if they are not in `models/`).
- **Training and the main experiment:** an Azure OpenAI resource (GPT-5,
  gpt-5-mini, GPT-4.1, GPT-4, GPT-4o deployments) and, for Actor training, an Anthropic
  key.
- **Safety experiment:** Anthropic, plus Azure OpenAI, Vertex AI or OpenRouter
  for the chatbots you test.

Before every push:

```bash
python scripts/check_secrets.py          # or install it as .git/hooks/pre-commit
```

## 📚 PSYCHE dataset (coming soon)

**PSYCHE** (**P**sychological **S**imulation Dataset for **Y**ielding
**C**ognitive and **H**uman b**E**haviors) is the first graph-grounded dataset
for psychological user simulation. **We will release it soon.**

Each sample pairs a patient profile with a **Network Model**: the patient's
psychological mechanisms as a directed graph of cause → effect relationships.

| | Count | How it's built |
|---|---:|---|
| 🗂️ Psychotherapy case reports | **516** | the source cases (510 of them train the Observer) |
| 👥 Patient profiles | **3,096** | GPT-5 augments each case into 6 profiles |
| 🎭 Training samples | **18,576** | 6 masking levels per profile–network pair |

**Why mask?** You rarely see all of someone's psychology at once. In each
sample, a subset of the network's relationships is masked: those mechanisms
start out hidden and can surface as the conversation goes on. The masking
levels match the Actor's training masks ([`model_training/actor/`](model_training/actor/#1-mask-the-networks)).

Until the release, `data/examples/` has small **synthetic** rows in exactly the
format each script expects. No real patient data is in this repository.

## Citation

*TBA*

## License

*TBA*
