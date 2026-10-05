# Training Angel's models

Angel has two models, both trained from `Qwen/Qwen3-8B`. A third, optional one,
the therapist, is only used to make the Actor's training conversations:

| Model | What it does | Trained with | Final checkpoint |
|---|---|---|---|
| **Observer** ([`observer/`](observer/)) | reads a patient's presenting complaints and builds a directed symptom network | SFT + GRPO | `models/Angel-Observer` |
| **Actor** ([`actor/`](actor/)) | role-plays the patient in a therapy conversation, following the symptom network | SFT + DPO | `models/Angel-Actor` |
| *Therapist* ([`therapist/`](therapist/)), optional | the LLM therapist the Actor talks to while its training data is made | GPT-4o fine-tuned on Azure OpenAI | your Azure deployment |

You only need this folder to train the models yourself. The trained models
are on Hugging Face: [`ChengLi0228/Angel-Observer`](https://huggingface.co/ChengLi0228/Angel-Observer)
and [`ChengLi0228/Angel-Actor`](https://huggingface.co/ChengLi0228/Angel-Actor).
To use them, see [`model_usage/`](../model_usage/).

| I want to… | Go to |
|---|---|
| train the Observer | [`observer/README.md`](observer/README.md) |
| train the Actor (I already have symptom networks) | [`actor/README.md`](actor/README.md) |
| train the Actor, using the released Observer | step 2 below, then [`actor/README.md`](actor/README.md) |
| train both from case reports | the three steps below |
| fine-tune the CBT therapist the paper used (optional) | [`therapist/README.md`](therapist/README.md) |
| build symptom networks with a trained Observer | step 2 below |

## Train the Observer first

The Actor learns from symptom networks, and those networks are written by the
Observer. So the order is:

```
case reports (presenting complaints)
  │
  │ 1. observer/  train the Observer (SFT + GRPO)
  ▼
Angel-Observer
  │
  │ 2. observer/predict_network  build one symptom network per case
  ▼
data/actor/network_models.jsonl
  │
  │ 3. actor/  train the Actor (masking, rollouts, SFT, DPO)
  ▼
Angel-Actor
```

1. **Observer.** Follow [`observer/README.md`](observer/README.md). It ends with
   `models/Angel-Observer`.
2. **Networks for the Actor.** Run the trained Observer on your case reports
   and write the output where the Actor looks for it. Without a local
   `models/Angel-Observer`, this uses the released
   [`ChengLi0228/Angel-Observer`](https://huggingface.co/ChengLi0228/Angel-Observer),
   so you can skip step 1:

   ```bash
   python -m model_training.observer.predict_network \
       --input data/observer/case_report_final_all.jsonl \
       --output data/actor/network_models.jsonl
   ```

   Each row has `symptoms` (node names) and `graph` (`{"from", "to"}` edges),
   which is the Actor's input. `--input-field` picks the text field
   (default `Complaints`). Cases that fail are written to
   `outputs/observer/network_model_err.jsonl`.
3. **Actor.** Follow [`actor/README.md`](actor/README.md), starting from
   step 1 (`mask_generator`). It ends with `models/Angel-Actor`.
   For DPO you can choose between two options:
   - **Judge DPO (default, faster):** a Claude judge picks the best and worst
     reply at each turn.
   - **[Graph DPO](actor/README.md#option-graph-reconstruction-dpo) (follows the
     paper, slower):** it prefers the conversation from which the Observer
     best rebuilds the symptom network.

Already have symptom networks in that format? Then you can skip steps 1 and 2
and train only the Actor.

## Setup

Run every command from the repository root. Each model has its own
requirements file:

```bash
pip install -r model_training/observer/requirements-observer.txt
pip install -r model_training/actor/requirements-actor.txt
cp .env.example .env        # keys and deployment names; the only place to edit
```

What each part needs in `.env`:

| | Observer | Actor |
|---|---|---|
| Azure OpenAI (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`) | GPT-5 builds the training data; gpt-5-mini scores edges in GRPO | GPT-5 masks networks and fixes reply format; a chat model plays the therapist (`ANGEL_THERAPIST_DEPLOYMENT`, required) |
| Anthropic (`ANTHROPIC_API_KEY`) | — | Claude judges replies to build DPO pairs (default DPO only) |
| GPUs | 4 in the paper (`torchrun --nproc_per_node 4`) | one GPU per step, except the SFT rollouts, which load `Qwen3-30B-A3B-Instruct-2507` (~60 GB) across the visible GPUs |

Each README lists every variable it reads. Models are saved under `models/`
and data under `data/` (move them with `ANGEL_MODELS_DIR` / `ANGEL_DATA_DIR`).

## Using your trained models

`model_usage` finds the models in `models/` by their default folder names
(falling back to the released Hugging Face models), or you can point to yours:

```bash
export ANGEL_OBSERVER_MODEL=models/Angel-Observer
export ANGEL_ACTOR_MODEL=models/Angel-Actor
python -m model_usage.angel chat --short-profile-file model_usage/examples/example_short_profile.txt
```

## Example data

`data/examples/observer/` and `data/examples/actor/` have small synthetic
examples of every file in both pipelines. No real patient data is included;
the paper's training data come from copyrighted case reports.
