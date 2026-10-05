# Therapist fine-tuning (for the Actor's training conversations)

The Actor learns from synthetic therapy conversations with an LLM **therapist**
(Actor steps 2, 5 and 5'). In the paper, that therapist was **GPT-4o
fine-tuned on CBT treatment-manual material**, so it asks questions the way a
CBT therapist would. This folder rebuilds that therapist:

```
treatment manual (PDF)
  1. build_data chunk      ─► book_chunks.jsonl          (~350-word chunks by chapter/section)
  2. build_data polish     ─► book_chunks_polished.jsonl (GPT-5 makes each chunk coherent)
  3. build_data label      ─► book_chunks_labeled.jsonl  (GPT-5: procedure / rationale /
                                                          adjustment / case / theory)
  4. build_data qa         ─► qa_samples.jsonl           (GPT-5 writes 2–3 Q&A per chunk)
  5. build_data make-chat  ─► therapist_training.jsonl   (chat fine-tuning rows)
  6. fine_tune             ─► Azure OpenAI fine-tuning job ─► deploy ─► ANGEL_THERAPIST_DEPLOYMENT
```

> **Do you need this?** No, it's optional. Any chat model works as the
> therapist: set `ANGEL_THERAPIST_DEPLOYMENT` to, say, a `gpt-4o` deployment
> and the Actor pipeline runs. The fine-tuned therapist only makes the
> conversations closer to the paper's.

## The paper's data

| | |
|---|---|
| Source | Barlow (ed.), *Clinical Handbook of Psychological Disorders: A Step-by-Step Treatment Manual*, 4th edition, including chapters on panic disorder, social anxiety, the unified protocol, bipolar disorder, psychosis and eating disorders |
| Chunks | 677 |
| Fine-tuning rows | 1,566 |
| Base model | GPT-4o, fine-tuned on Azure OpenAI with default hyperparameters |

The book is copyrighted, so neither it nor the data built from it is in this
repository. The Q&A are GPT-5 paraphrases (the prompt says "Do NOT quote the
original text"), but they're still derived from the book. Use a manual you
have the rights to. Everything you put under `data/` is gitignored except
`data/examples/`.

## What you need

- **Azure OpenAI** with GPT-5 for steps 2–4 (`AZURE_OPENAI_ENDPOINT`,
  `AZURE_OPENAI_API_KEY`, `ANGEL_GPT5_DEPLOYMENT`), and a region where you can
  fine-tune GPT-4o for step 6. Fine-tuning and hosting the result cost money.
- **No GPU.**
- **A treatment manual as a PDF** with selectable text (not a scan).

```bash
pip install -r model_training/therapist/requirements-therapist.txt
```

## Steps

Run from the repository root. All outputs go under `data/therapist/` by default.

```bash
B="python -m model_training.therapist.build_data"
$B chunk --pdf data/therapist/manual.pdf      # 1. PDF -> chunks
$B polish                                      # 2. GPT-5
$B label                                       # 3. GPT-5
$B qa                                          # 4. GPT-5
$B make-chat                                   # 5. -> therapist_training.jsonl

python -m model_training.therapist.fine_tune --dry-run        # 6a. check the file (free)
python -m model_training.therapist.fine_tune --wait           # 6b. upload + fine-tune
```

- **Steps 2–4** call GPT-5 once per chunk. They append to their output and can
  resume with `--start <row>`; delete the output file to start over.
  Chunks GPT-5 can't polish, label or turn into valid JSON are skipped, with a
  message.
- **Step 1** detects chapters ("Chapter 3") and sections (Assessment,
  Treatment, Case Illustration, Discussion, Summary, "Stage N") from heading
  lines, and starts a new chunk at each one. Chunks with no detected chapter
  are kept; check `book_chunks.jsonl` and delete front matter (title pages,
  references) before step 2 to save GPT-5 calls.
- **Step 6** needs at least 10 rows. The base model defaults to
  `gpt-4o-2024-08-06` (`--model`); use one your region can fine-tune. When the
  job succeeds:
  1. Deploy the fine-tuned model in Azure AI Foundry (Fine-tuning → your job → Deploy).
     Give the deployment a custom content filter with a higher `self_harm`
     threshold; the default one blocks many patient replies (see the Actor
     README's Troubleshooting).
  2. Set `ANGEL_THERAPIST_DEPLOYMENT=<your deployment name>` in `.env`. Set
     `ANGEL_THERAPIST_AZURE_ENDPOINT` and `ANGEL_THERAPIST_AZURE_API_KEY` too if
     it's on a different resource than GPT-5.

## Try it on the example

`data/examples/therapist/` has a two-page synthetic manual and the output of
every step for it (2 chunks → 5 rows). To rerun it (about 10 GPT-5 calls):

```bash
B="python -m model_training.therapist.build_data"
$B chunk --pdf data/examples/therapist/synthetic_manual.pdf --max-words 80 \
    --output /tmp/therapist/book_chunks.jsonl
$B polish --input /tmp/therapist/book_chunks.jsonl --output /tmp/therapist/book_chunks_polished.jsonl
$B label --input /tmp/therapist/book_chunks_polished.jsonl --output /tmp/therapist/book_chunks_labeled.jsonl
$B qa --input /tmp/therapist/book_chunks_labeled.jsonl --output /tmp/therapist/qa_samples.jsonl
$B make-chat --input /tmp/therapist/qa_samples.jsonl --output /tmp/therapist/therapist_training.jsonl
```

`fine_tune --dry-run` on these 5 rows correctly refuses (Azure needs 10).

## Prompts

`prompts.py` has every prompt, copied verbatim from the scripts that built
the paper's data:

| Prompt | Used in |
|---|---|
| `POLISH_PROMPT` | step 2 |
| `ROLE_PROMPT` | step 3 |
| `QA_SYSTEM_PROMPT` and `QA_PROMPTS` | step 4 (one template per role) |
| `FT_SYSTEM_PROMPT` | step 5, the system message of every fine-tuning row |

During the Actor's rollouts the therapist is called with a different system
prompt, `model_training.actor.prompts.THERAPIST_SYSTEM_PROMPT` (CBT-style
Socratic questions, no diagnosis, under 200 words).

## Differences from the original scripts

- **Chunking keeps text after a heading.** The original chunker dropped a
  whole text block when it began with a heading. When PyMuPDF returns a page
  without blank lines, that lost the whole page.
- **Roles must match.** Role labels are matched to the five known roles;
  replies that don't match are skipped. The original kept GPT-5's raw reply.
- **No JSON repair step.** Q&A rows are written with `json.dumps`, so the old
  `process.py` step that repaired broken JSON lines isn't needed.
