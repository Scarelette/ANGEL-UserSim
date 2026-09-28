"""Run the Observer end to end: profile text -> S1 nodes -> S2 directed symptom network.

Works on any text field: presenting complaints (``--input-field complaints``)
or Observer/LLM-generated long profiles (``--input-field long_profile_description``).
Generation: sampling, temperature 0.1, top_p 0.9; S2 is retried until a
<GRAPH> block parses.

    python -m model_training.observer.predict_network --input data.jsonl --output nets.jsonl \
        --error-output nets_err.jsonl --input-field complaints
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from angel_common.paths import OUTPUTS_DIR, resolve_model
from model_training.observer.prompts import build_input_s1, build_input_s2, build_system_prompt_s1, build_system_prompt_s2


GRAPH_START = "<GRAPH>"
GRAPH_END = "</GRAPH>"


def extract_symptoms_and_merge(text: str) -> List[str]:
    match = re.search(r"<GRAPH>\s*(\{.*?\})\s*</GRAPH>", text, re.S)
    if not match:
        raise ValueError("No <GRAPH> block found in stage-1 response.")

    graph_json = match.group(1)
    data = json.loads(graph_json)

    combined: List[str] = []
    if "symptoms" in data:
        combined.extend(data["symptoms"])
    if "external_factors" in data:
        combined.extend(data["external_factors"])
    return combined


def extract_graph(text: str) -> List[Dict[str, str]]:
    match = re.search(r"<GRAPH>\s*(\{.*?\})\s*</GRAPH>", text, re.S)
    if not match:
        raise ValueError("No <GRAPH> block found in stage-2 response.")

    graph_json = match.group(1)
    data = json.loads(graph_json)
    return data.get("links", [])


def generate_ans(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    gen_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def get_profile_text(item: Dict[str, Any], input_field: str) -> str:
    value = item.get(input_field)
    if value is None:
        raise ValueError(f"Missing input field '{input_field}'.")
    if not isinstance(value, str):
        raise ValueError(f"Input field '{input_field}' must be a string.")
    value = value.strip()
    if not value:
        raise ValueError(f"Input field '{input_field}' is empty.")
    return value


def build_stage1_messages(profile_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt_s1()},
        {"role": "user", "content": build_input_s1(profile_text)},
    ]


def build_stage2_messages(profile_text: str, symptoms: List[str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt_s2()},
        {"role": "user", "content": build_input_s2(profile_text, symptoms)},
    ]


def process_profiles(
    input_path: str,
    output_path: str,
    error_path: str,
    model_path: str,
    input_field: str,
    limit: Optional[int],
    stage1_max_new_tokens: int,
    stage2_max_new_tokens: int,
    stage2_retries: int,
) -> None:
    for path in (output_path, error_path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processed = 0
    seen = 0
    with jsonlines.open(output_path, mode="w") as writer:
        with jsonlines.open(error_path, mode="w") as writer_err:
            with jsonlines.open(input_path, mode="r") as reader:
                for idx, item in enumerate(reader):
                    if limit is not None and seen >= limit:
                        break
                    seen += 1

                    try:
                        profile_text = get_profile_text(item, input_field)
                        print(item.get("source_title") or item.get("id", idx))

                        stage1_messages = build_stage1_messages(profile_text)
                        stage1_response = generate_ans(
                            tokenizer=tokenizer,
                            model=model,
                            messages=stage1_messages,
                            max_new_tokens=stage1_max_new_tokens,
                        )
                        symptoms = extract_symptoms_and_merge(stage1_response)
                        print("S:\n", symptoms)

                        stage2_messages = build_stage2_messages(profile_text, symptoms)
                        stage2_response = ""
                        links: List[Dict[str, str]] = []
                        stage2_error = None

                        for attempt in range(stage2_retries):
                            stage2_response = generate_ans(
                                tokenizer=tokenizer,
                                model=model,
                                messages=stage2_messages,
                                max_new_tokens=stage2_max_new_tokens,
                            )
                            try:
                                links = extract_graph(stage2_response)
                                break
                            except Exception as exc:
                                stage2_error = repr(exc)
                                if attempt == stage2_retries - 1:
                                    raise

                        new_obj = {
                            "id": item.get("id", idx),
                            "source_title": item.get("source_title") or item.get("title"),
                            input_field: profile_text,
                            "symptoms": symptoms,
                            "graph": links,
                            "stage1_response": stage1_response,
                            "stage2_response": stage2_response,
                            "stage2_error": stage2_error,
                            "original_item": item,
                        }
                        writer.write(new_obj)
                        print("G:\n", links)

                        processed += 1
                        if processed % 5 == 0:
                            writer._fp.flush()
                            os.fsync(writer._fp.fileno())
                    except Exception as exc:
                        writer_err.write(
                            {
                                "id": item.get("id", idx),
                                "source_title": item.get("source_title") or item.get("title"),
                                "error": repr(exc),
                                "original_item": item,
                            }
                        )
                        writer_err._fp.flush()
                        os.fsync(writer_err._fp.fileno())

    print(f"Processed {seen} rows and saved {processed} network-model samples to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build symptom networks (S1 nodes -> S2 links) with the Observer model.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path containing long patient profiles.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "observer" / "network_model.jsonl"),
        help="Output JSONL path for generated network-model data.",
    )
    parser.add_argument(
        "--error-output",
        default=str(OUTPUTS_DIR / "observer" / "network_model_err.jsonl"),
        help="Output JSONL path for failed rows.",
    )
    parser.add_argument(
        "--input-field",
        default="long_profile_description",
        help="Field containing the long patient profile text.",
    )
    parser.add_argument("--model-path", default=None, help="Default: resolve_model('observer') (ANGEL_OBSERVER_MODEL).")
    parser.add_argument("--stage1-max-new-tokens", type=int, default=2048)
    parser.add_argument("--stage2-max-new-tokens", type=int, default=2048)
    parser.add_argument("--stage2-retries", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_profiles(
        input_path=args.input,
        output_path=args.output,
        error_path=args.error_output,
        model_path=resolve_model("observer", args.model_path),
        input_field=args.input_field,
        limit=args.limit,
        stage1_max_new_tokens=args.stage1_max_new_tokens,
        stage2_max_new_tokens=args.stage2_max_new_tokens,
        stage2_retries=args.stage2_retries,
    )


if __name__ == "__main__":
    main()
