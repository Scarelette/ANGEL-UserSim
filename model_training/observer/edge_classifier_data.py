"""Fine-tuning data for the Yes/No edge-plausibility classifier.

``from-annotations`` turns the human-annotated edge CSVs into chat JSONL. The
paper fine-tuned a GPT-4 deployment on it with Azure OpenAI fine-tuning (done in
the Azure portal / fine-tuning API; not scripted here). The resulting classifier
(edge_classifier_api.py) gives the edge "reasonability" score in ``eval/``.

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("from-annotations")
    a.add_argument("--annotation-dir", required=True)
    a.add_argument("--output", default=str(DATA_DIR / "observer" / "classifier_annotations.jsonl"))

    args = p.parse_args()
    from_annotations(args.annotation_dir, args.output)


if __name__ == "__main__":
    main()
