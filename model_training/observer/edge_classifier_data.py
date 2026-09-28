"""Build training data for the Yes/No edge-plausibility classifiers.

Pipeline used in the paper:

1. ``from-annotations``: human-annotated edge CSVs -> chat JSONL. This data was
   used to fine-tune a GPT-4 deployment on Azure OpenAI (done in the Azure
   portal / fine-tuning API; not scripted here).
2. Harvest candidate edges by running S2 GRPO with ``--reward format_only
   --edge-dump edge_dump_s2.jsonl`` (see train_grpo.py).
3. ``label``: label each harvested edge with the Azure fine-tuned classifier
   (edge_classifier_api.py) -> predicted_edges.jsonl.
4. ``to-sft``: keep only the chat messages -> classifier_sft.jsonl, the
   training set for train_edge_classifier.py (Qwen3-0.6B).

Annotation CSV columns: ``complaints`` (only on the first row of each case;
later rows inherit it), ``parent node``, ``child node``, ``Annotation`` (Yes/No).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from angel_common.paths import DATA_DIR

EXPERT_SYSTEM = "You are a professional mental health expert."


def _annotation_prompt(complaints: str, parent: str, child: str) -> str:
    # Verbatim (including the "-->" arrow) from the original annotation export.
    return (
        f"This is the Presenting Complaints of the mental health patient: \n{complaints}\n"
        f"Based on the patient's symptoms, I construct a symptom relationship for the patient: \n"
        f"{parent} --> {child}\n\n"
        "Do you think this link make sense according to the presenting complaints? Just answer with Yes or No"
    )


def _messages(user: str, answer: str):
    return {"messages": [
        {"role": "system", "content": EXPERT_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]}


def from_annotations(annotation_dir: str, output: str) -> None:
    n = 0
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as writer:
        for name in sorted(os.listdir(annotation_dir)):
            if not name.endswith(".csv"):
                continue
            with open(os.path.join(annotation_dir, name), "r", encoding="utf-8") as f:
                complaints = ""
                for row in csv.DictReader(f):
                    if len(row["complaints"]) > 2:
                        complaints = row["complaints"]
                    item = _messages(
                        _annotation_prompt(complaints, row["parent node"], row["child node"]),
                        row["Annotation"].strip(),
                    )
                    writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                    n += 1
    print(f"Wrote {n} annotated edges to {output}")


def label(edge_dump: str, output: str, start: int, end: int | None) -> None:
    """Label harvested edges with the Azure fine-tuned classifier; resumable (appends)."""
    from tqdm import tqdm

    from model_training.observer.edge_classifier_api import build_edge_prompt, classifier, extract_label

    with open(edge_dump, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    done = 0
    if os.path.exists(output):
        with open(output, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
    print(f"Skipping the first {done} rows already in {output}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a", encoding="utf-8") as writer:
        for i, sample in enumerate(tqdm(rows)):
            if i < start + done or (end is not None and i > end):
                continue
            prompt = build_edge_prompt(sample["complaint"], sample["from"], sample["to"])
            try:
                output_text = classifier(prompt)
            except Exception as e:
                print(f"[Error @ {i}] {e}")
                output_text = ""
            pred = extract_label(output_text).strip()
            record = {
                "complaint": sample["complaint"],
                "edge": {"from": sample["from"], "to": sample["to"]},
                "label": pred,
                **_messages(prompt, pred),
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 5 == 0:
                writer.flush()


def to_sft(predicted: str, output: str) -> None:
    n = 0
    with open(predicted, "r", encoding="utf-8") as fin, open(output, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                fout.write(json.dumps({"messages": json.loads(line)["messages"]}, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} rows to {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("from-annotations")
    a.add_argument("--annotation-dir", required=True)
    a.add_argument("--output", default=str(DATA_DIR / "observer" / "classifier_annotations.jsonl"))

    b = sub.add_parser("label")
    b.add_argument("--edge-dump", default=str(DATA_DIR / "observer" / "edge_dump_s2.jsonl"))
    b.add_argument("--output", default=str(DATA_DIR / "observer" / "predicted_edges.jsonl"))
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--end", type=int, default=None, help="Last row index to label (inclusive).")

    c = sub.add_parser("to-sft")
    c.add_argument("--predicted", default=str(DATA_DIR / "observer" / "predicted_edges.jsonl"))
    c.add_argument("--output", default=str(DATA_DIR / "observer" / "classifier_sft.jsonl"))

    args = p.parse_args()
    if args.cmd == "from-annotations":
        from_annotations(args.annotation_dir, args.output)
    elif args.cmd == "label":
        label(args.edge_dump, args.output, args.start, args.end)
    else:
        to_sft(args.predicted, args.output)


if __name__ == "__main__":
    main()
