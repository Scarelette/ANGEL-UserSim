"""Build Observer SFT / GRPO data from case reports with GPT-5 (Azure OpenAI).

Input: a JSONL of case reports with at least ``title`` and ``Complaints``
(presenting-complaints text). The paper used 510 published psychotherapy case
reports; those texts are not redistributed — see data/examples/observer/.

Steps, in order (each is a subcommand; outputs default under data/observer/):

  add-nodes     Complaints -> gpt5_nodes (GPT-5 symptom + external-factor lists)
                                                           -> case_report_final_all_nodes.jsonl
  s1-responses  GPT-5 answers the S1 prompt on a text field -> gpt5_response
                  * on Complaints, run twice (1020 rows)    -> case_report_final_all_nodes_sft.jsonl
                  * on new_complaints after `augment`       -> case_report_final_all_nodes_aug_p_sft.jsonl
  augment       GPT-5 writes a similar patient's complaints -> new_complaints
                                                           -> case_report_final_all_nodes_aug_p.jsonl
  extract-nodes aug rows -> {title, Complaints=new_complaints, gpt5_nodes from the S1 response}
                                                           -> case_report_aug_p_grpo.jsonl
  s2-responses  GPT-5 answers the S2 prompt (Complaints + gpt5_nodes) -> gpt5_response
                                                           -> sft_training_s2_data.jsonl
  make-sft      rows -> {"messages": [system, user, assistant]}
                  --stage s1: sft_training.jsonl   --stage s2: sft_training_s2.jsonl
  make-grpo     concatenate node files                     -> grpo_training.jsonl

All GPT-5 calls go through angel_common.llm.get_output (env: AZURE_OPENAI_*,
ANGEL_GPT5_DEPLOYMENT). Every generation step appends and can be resumed with
``--start``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from angel_common.paths import DATA_DIR
from model_training.observer.prompts import (
    build_input_s1,
    build_input_s2,
    build_system_prompt_s1_datagen,
    build_system_prompt_s2,
)

OUT = DATA_DIR / "observer"


# --------------------------------------------------------------------------- #
# IO + parsing helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def convert2list(text: str) -> List[str]:
    """Numbered items inside the first <tag>...</tag> block."""
    if text is None or not text.strip():
        return []
    tag_match = re.search(r"<\s*([a-zA-Z0-9_]+)\s*>", text)
    if not tag_match:
        raise ValueError("No opening tag found.")
    tag = tag_match.group(1)
    content = re.search(fr"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>", text, re.S)
    if not content:
        raise ValueError(f"No <{tag}> block found.")
    items = re.findall(r"\d+\.\s*(.*?)\s*(?=\n\d+\.|$)", content.group(1).strip(), re.S)
    return [re.sub(r"^\d+\.\s*", "", s).strip() for s in items]


def convert2list_v2(text: str) -> List[str]:
    """Items written as repeated <symptom>...</symptom> tags."""
    if text is None or not text.strip():
        return []
    return [m.strip() for m in re.findall(r"<\s*symptom\s*>(.*?)</\s*symptom\s*>", text, re.S)]


def _tagged_list(output: str) -> List[str]:
    try:
        items = convert2list(output)
    except ValueError:
        items = []
    return items or convert2list_v2(output)


GRAPH_RE = re.compile(r"<GRAPH>\s*(\{.*?\})\s*</GRAPH>", re.DOTALL)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def add_nodes(args) -> None:
    from angel_common.llm import get_output

    for i, obj in enumerate(read_jsonl(args.input)):
        if i < args.start:
            continue
        complaints = obj["Complaints"]
        external_prompt = (
            "Please extract the external factors that influence the patient’s mental health "
            "based on the below Presenting Complaints.\n"
            + complaints
            + "\nThe external factors are enclosed within <external>...</external>"
        )
        symptom_prompt = (
            "Please extract the symptoms of the patient based on the below Presenting Complaints.\n"
            + complaints
            + "\nThe symptoms are enclosed within <symptom>...</symptom>"
        )
        external_list = _tagged_list(get_output(external_prompt, max_completion_tokens=16384))
        symptom_list = _tagged_list(get_output(symptom_prompt, max_completion_tokens=16384))
        obj["gpt5_nodes"] = symptom_list + external_list
        append_jsonl(args.output, obj)
        print(i, obj.get("title"), len(obj["gpt5_nodes"]), "nodes")


def s1_responses(args) -> None:
    from angel_common.llm import get_output

    for i, obj in enumerate(read_jsonl(args.input)):
        if i < args.start:
            continue
        try:
            prompt = build_system_prompt_s1_datagen() + "\n\nPresenting Complaints:\n" + obj[args.field]
            obj["gpt5_response"] = get_output(prompt, max_completion_tokens=16384).strip()
            append_jsonl(args.output, obj)
        except Exception as e:
            print(f"[{i}] skipped: {e}")


def augment(args) -> None:
    from angel_common.llm import get_output

    for i, obj in enumerate(read_jsonl(args.input)):
        if i < args.start:
            continue
        prompt = (
            "Please generate a similar patient's Presenting Complaints according to the one below: "
            "\n\nPresenting Complaints:\n" + obj["Complaints"]
        )
        try:
            output = get_output(prompt, tag=3, max_completion_tokens=16384).strip()
            obj["new_complaints"] = output.replace("Presenting Complaints:", "").strip()
            append_jsonl(args.output, obj)
        except Exception as e:
            print(f"[{i}] skipped: {e}")


def extract_nodes(args) -> None:
    def rows():
        for item in read_jsonl(args.input):
            match = GRAPH_RE.search(item.get("gpt5_response", ""))
            if not match:
                continue
            try:
                graph = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            yield {
                "title": item.get("title"),
                "Complaints": item.get("new_complaints"),
                "gpt5_nodes": graph.get("symptoms", []) + graph.get("external_factors", []),
            }

    print(f"Wrote {write_jsonl(args.output, rows())} rows to {args.output}")


def s2_responses(args) -> None:
    from angel_common.llm import get_output

    for i, obj in enumerate(read_jsonl(args.input)):
        if i < args.start or (args.end is not None and i > args.end):
            continue
        try:
            prompt = build_system_prompt_s2() + "\n\n" + build_input_s2(obj["Complaints"], obj["gpt5_nodes"])
            obj["gpt5_response"] = get_output(prompt, max_completion_tokens=16384).strip()
            append_jsonl(args.output, obj)
        except Exception as e:
            print(f"[{i}] skipped: {e}")


def make_sft(args) -> None:
    def rows():
        for path in args.inputs:
            for row in read_jsonl(path):
                if args.stage == "s1":
                    complaints = row["Complaints"]
                    if args.pair_augmented and row.get("new_complaints"):
                        complaints = row["new_complaints"]
                    system, user = build_system_prompt_s1_datagen(), build_input_s1(complaints)
                else:
                    system, user = build_system_prompt_s2(), build_input_s2(row["Complaints"], row["gpt5_nodes"])
                yield {"messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["gpt5_response"]},
                ]}

    print(f"Wrote {write_jsonl(args.output, rows())} rows to {args.output}")


def make_grpo(args) -> None:
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    print(f"Wrote {write_jsonl(args.output, rows)} rows to {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def step(name, fn, inp, out, **extra):
        s = sub.add_parser(name)
        s.add_argument("--input", default=str(OUT / inp))
        s.add_argument("--output", default=str(OUT / out))
        s.add_argument("--start", type=int, default=0, help="Skip rows before this index (resume).")
        for flag, kw in extra.items():
            s.add_argument(flag.replace("_", "-"), **kw)
        s.set_defaults(fn=fn)

    step("add-nodes", add_nodes, "case_report_final_all.jsonl", "case_report_final_all_nodes.jsonl")
    step("s1-responses", s1_responses, "case_report_final_all_nodes_aug_p.jsonl",
         "case_report_final_all_nodes_aug_p_sft.jsonl",
         __field=dict(default="new_complaints", help="Text field to answer (Complaints or new_complaints)."))
    step("augment", augment, "case_report_final_all_nodes.jsonl", "case_report_final_all_nodes_aug_p.jsonl")
    step("extract-nodes", extract_nodes, "case_report_final_all_nodes_aug_p_sft.jsonl", "case_report_aug_p_grpo.jsonl")
    step("s2-responses", s2_responses, "case_report_aug_p_grpo.jsonl", "sft_training_s2_data.jsonl",
         __end=dict(type=int, default=None, help="Last row index (inclusive)."))

    s = sub.add_parser("make-sft")
    s.add_argument("--stage", choices=["s1", "s2"], required=True)
    s.add_argument("--inputs", nargs="+", default=None,
                   help="S1 default: nodes_sft + aug_p_sft files; S2 default: sft_training_s2_data.jsonl.")
    s.add_argument("--output", default=None)
    s.add_argument("--pair-augmented", action="store_true",
                   help="S1: use new_complaints as the user turn for augmented rows (the GPT-5 answer was "
                        "generated from it). Default reproduces the released data, which paired the ORIGINAL "
                        "complaints with it (README, Known issues).")
    s.set_defaults(fn=make_sft)

    g = sub.add_parser("make-grpo")
    g.add_argument("--inputs", nargs="+", default=[
        str(OUT / "case_report_final_all_nodes.jsonl"),
        str(OUT / "case_report_final_all_nodes_aug_p_sft.jsonl"),
    ])
    g.add_argument("--output", default=str(OUT / "grpo_training.jsonl"))
    g.set_defaults(fn=make_grpo)

    args = p.parse_args()
    if args.cmd == "make-sft":
        if args.inputs is None:
            args.inputs = (
                [str(OUT / "case_report_final_all_nodes_sft.jsonl"), str(OUT / "case_report_final_all_nodes_aug_p_sft.jsonl")]
                if args.stage == "s1" else [str(OUT / "sft_training_s2_data.jsonl")]
            )
        args.output = args.output or str(OUT / ("sft_training.jsonl" if args.stage == "s1" else "sft_training_s2.jsonl"))
    args.fn(args)


if __name__ == "__main__":
    main()
