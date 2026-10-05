"""Build the therapist's fine-tuning data from a treatment manual (PDF) with GPT-5.

The paper's therapist was fine-tuned on Q&A that GPT-5 wrote from a CBT
treatment manual (Barlow (ed.), *Clinical Handbook of Psychological
Disorders*). That book is copyrighted, so neither it nor the data built from
it is included; bring your own manual. Synthetic examples of every file are in
``data/examples/therapist/``.

Steps, in order (each is a subcommand; outputs default under data/therapist/):

  chunk       PDF -> ~350-word chunks by chapter/section      -> book_chunks.jsonl
  polish      GPT-5 makes each chunk coherent                 -> book_chunks_polished.jsonl
  label       GPT-5 labels each chunk's role                  -> book_chunks_labeled.jsonl
              (procedure / rationale / adjustment / case / theory)
  qa          GPT-5 writes 2-3 Q&A pairs per chunk, by role   -> qa_samples.jsonl
  make-chat   Q&A -> chat rows {"messages": [system, user, assistant]}
                                                              -> therapist_training.jsonl

``therapist_training.jsonl`` is in the Azure OpenAI chat fine-tuning format;
``fine_tune`` uploads it and starts the job. The GPT-5 steps append and can
be resumed with ``--start``.

    python -m model_training.therapist.build_data chunk --pdf data/therapist/manual.pdf
    python -m model_training.therapist.build_data polish
    python -m model_training.therapist.build_data label
    python -m model_training.therapist.build_data qa
    python -m model_training.therapist.build_data make-chat
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List

from angel_common.llm import get_output
from angel_common.paths import DATA_DIR, resolve_path
from model_training.therapist.prompts import (
    FT_SYSTEM_PROMPT,
    POLISH_PROMPT,
    QA_PROMPTS,
    QA_SYSTEM_PROMPT,
    ROLE_PROMPT,
    ROLES,
)

OUT = DATA_DIR / "therapist"
_MAX_TOKENS = 16384  # completion budget the original GPT-5 calls used

CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)", re.I)
STAGE_RE = re.compile(r"^(Stage\s+\d+.*)", re.I)
SECTION_RE = re.compile(r"^(Assessment|Treatment|Case Illustration|Discussion|Summary)", re.I)


def read_jsonl(path) -> List[Dict]:
    with open(resolve_path(path), encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows: Iterable[Dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# chunk: PDF -> chunks (from pdf_parser/run_pipeline.py)
# --------------------------------------------------------------------------- #
def chunk_pdf(pdf_path, max_words: int = 350) -> List[Dict]:
    import fitz  # PyMuPDF

    paras, chapter, section = [], None, None
    for i, page in enumerate(fitz.open(str(pdf_path))):
        for para in (p.strip() for p in page.get_text().split("\n\n")):
            # A heading ("Chapter 3", "Treatment", ...) is consumed line by line,
            # so text that follows it in the same block is kept. (The original
            # script dropped the whole block, which loses a page when PyMuPDF
            # returns it without blank lines.)
            while para:
                head, _, rest = para.partition("\n")
                head = head.strip()
                if len(head.split()) > 8:
                    break
                if m := CHAPTER_RE.match(head):
                    chapter, section = int(m.group(1)), None
                elif m := STAGE_RE.match(head) or SECTION_RE.match(head):
                    section = m.group(1)
                else:
                    break
                para = rest.strip()
            if para:
                paras.append({"page": i + 1, "chapter": chapter, "section": section, "text": para})

    chunks, buf, n_words, meta = [], [], 0, None
    for p in paras:
        if meta and buf and (p["chapter"], p["section"]) != (meta["chapter"], meta["section"]):
            chunks.append({**meta, "content": " ".join(buf)})
            buf, n_words, meta = [], 0, None
        if meta is None:
            meta = {"chapter": p["chapter"], "section": p["section"], "page_start": p["page"]}
        buf.append(p["text"])
        n_words += len(p["text"].split())
        if n_words >= max_words:
            chunks.append({**meta, "content": " ".join(buf)})
            buf, n_words, meta = [], 0, None
    if buf:
        chunks.append({**meta, "content": " ".join(buf)})
    return [{"id": str(uuid.uuid4()), **c} for c in chunks]


def cmd_chunk(args) -> None:
    chunks = chunk_pdf(resolve_path(args.pdf), args.max_words)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to {out}")


# --------------------------------------------------------------------------- #
# polish / label (from pdf_parser/chunks_refine.py)
# --------------------------------------------------------------------------- #
def cmd_polish(args) -> None:
    def rows():
        for i, row in enumerate(read_jsonl(args.input)):
            if i < args.start:
                continue
            text = get_output(POLISH_PROMPT.format(content=row["content"]), tag=0,
                              max_completion_tokens=_MAX_TOKENS).strip()
            if not text:
                print(f"[{i}] skipped: empty GPT-5 answer")
                continue
            print(f"[{i}] polished")
            yield {**row, "polished_content": text}

    print(f"Wrote {append_jsonl(args.output, rows())} rows to {args.output}")


def match_role(reply: str):
    text = (reply or "").strip().lower()
    found = [r for r in ROLES if re.search(rf"\b{r}\b", text)]
    return found[0] if len(found) == 1 else None


def cmd_label(args) -> None:
    def rows():
        for i, row in enumerate(read_jsonl(args.input)):
            if i < args.start:
                continue
            reply = get_output(ROLE_PROMPT.format(content=row["polished_content"]), tag=0,
                               max_completion_tokens=_MAX_TOKENS)
            role = match_role(reply)
            if role is None:
                print(f"[{i}] skipped: unclear role {reply!r}")
                continue
            print(f"[{i}] {role}")
            yield {**row, "role": role}

    print(f"Wrote {append_jsonl(args.output, rows())} rows to {args.output}")


# --------------------------------------------------------------------------- #
# qa / make-chat (from sft_generation/generate_sft_samples.py)
# --------------------------------------------------------------------------- #
def parse_examples(reply: str) -> List[Dict[str, str]]:
    """The JSON array GPT-5 returns, tolerating code fences and smart quotes."""
    text = (reply or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        items = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        try:
            items = json.loads(text[i:j + 1].replace("“", '"').replace("”", '"'))
        except json.JSONDecodeError:
            return []
    return [x for x in items if isinstance(x, dict) and x.get("instruction") and x.get("response")]


def cmd_qa(args) -> None:
    def rows():
        for i, row in enumerate(read_jsonl(args.input)):
            if i < args.start:
                continue
            role = row.get("role")
            if role not in QA_PROMPTS:
                continue
            reply = get_output(QA_PROMPTS[role].format(content=row["polished_content"]), tag=0,
                               max_completion_tokens=_MAX_TOKENS, system_prompt=QA_SYSTEM_PROMPT)
            examples = parse_examples(reply)
            if not examples:
                print(f"[{i}] skipped: no valid JSON examples")
                continue
            print(f"[{i}] {role}: {len(examples)} examples")
            for ex in examples:
                yield {"input": ex["instruction"], "output": ex["response"],
                       "meta": {"chapter": row.get("chapter"), "section": row.get("section"),
                                "role": role, "source_id": row.get("id")}}

    print(f"Wrote {append_jsonl(args.output, rows())} rows to {args.output}")


def cmd_make_chat(args) -> None:
    rows = ({"messages": [{"role": "system", "content": FT_SYSTEM_PROMPT},
                          {"role": "user", "content": r["input"]},
                          {"role": "assistant", "content": r["output"]}]}
            for r in read_jsonl(args.input))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {sum(1 for _ in open(out))} rows to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("chunk", help="PDF -> chunks")
    s.add_argument("--pdf", required=True, help="the treatment manual")
    s.add_argument("--output", default=str(OUT / "book_chunks.jsonl"))
    s.add_argument("--max-words", type=int, default=350)
    s.set_defaults(fn=cmd_chunk)

    for name, fn, inp, out in [
        ("polish", cmd_polish, "book_chunks.jsonl", "book_chunks_polished.jsonl"),
        ("label", cmd_label, "book_chunks_polished.jsonl", "book_chunks_labeled.jsonl"),
        ("qa", cmd_qa, "book_chunks_labeled.jsonl", "qa_samples.jsonl"),
    ]:
        s = sub.add_parser(name)
        s.add_argument("--input", default=str(OUT / inp))
        s.add_argument("--output", default=str(OUT / out), help="appended")
        s.add_argument("--start", type=int, default=0, help="skip rows before this index (resume)")
        s.set_defaults(fn=fn)

    s = sub.add_parser("make-chat", help="Q&A -> chat fine-tuning rows (overwrites)")
    s.add_argument("--input", default=str(OUT / "qa_samples.jsonl"))
    s.add_argument("--output", default=str(OUT / "therapist_training.jsonl"))
    s.set_defaults(fn=cmd_make_chat)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
