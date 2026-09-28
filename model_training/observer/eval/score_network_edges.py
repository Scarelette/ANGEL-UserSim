import argparse
import json
import os
from typing import Any, Dict, List, Optional

from model_training.observer.edge_classifier_api import build_edge_prompt as _build_edge_prompt
from model_training.observer.edge_classifier_api import classifier


def get_text_field(item: Dict[str, Any], input_field: str) -> str:
    value = item.get(input_field)
    if value is None:
        raise ValueError(f"Missing input field '{input_field}'.")
    if not isinstance(value, str):
        raise ValueError(f"Input field '{input_field}' must be a string.")
    value = value.strip()
    if not value:
        raise ValueError(f"Input field '{input_field}' is empty.")
    return value


def get_edges(item: Dict[str, Any], graph_field: str) -> List[Dict[str, str]]:
    value = item.get(graph_field)
    if value is None:
        raise ValueError(f"Missing graph field '{graph_field}'.")
    if not isinstance(value, list):
        raise ValueError(f"Graph field '{graph_field}' must be a list.")

    edges: List[Dict[str, str]] = []
    for idx, edge in enumerate(value):
        if not isinstance(edge, dict):
            raise ValueError(f"Edge at index {idx} is not an object.")
        edge_from = edge.get("from")
        edge_to = edge.get("to")
        if not isinstance(edge_from, str) or not edge_from.strip():
            raise ValueError(f"Edge at index {idx} has invalid 'from'.")
        if not isinstance(edge_to, str) or not edge_to.strip():
            raise ValueError(f"Edge at index {idx} has invalid 'to'.")
        edges.append({"from": edge_from.strip(), "to": edge_to.strip()})
    return edges


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON in {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl_row(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def edge_to_text(edge: Dict[str, str]) -> str:
    return f"{edge['from']} -> {edge['to']}"


def extract_label(text: Optional[str], label_set: List[str]) -> str:
    if text is None:
        return "UNKNOWN"

    normalized = text.strip().lower()
    if not normalized:
        return "UNKNOWN"

    for label in label_set:
        if label.lower() in normalized:
            return label.upper()

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return "UNKNOWN"

    return lines[0].upper()


def build_edge_prompt(complaints: str, edge: Dict[str, str]) -> str:
    return _build_edge_prompt(complaints, edge["from"], edge["to"])


def score_label_to_float(label: str) -> float:
    if label == "YES":
        return 1.0
    if label == "NO":
        return 0.0
    raise ValueError(f"Unexpected classifier label: {label}")


def score_one_sample(
    item: Dict[str, Any],
    input_field: str,
    graph_field: str,
    idx: int,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    complaints = get_text_field(item, input_field)
    edges = get_edges(item, graph_field)

    if not edges:
        return {
            "id": item.get("id", idx),
            "source_title": item.get("source_title") or item.get("title"),
            input_field: complaints,
            graph_field: [],
            "edge_scores": [],
            "average_edge_score": None,
            "score_error": None,
            "original_item": item,
        }

    edge_scores = []
    for edge_id, edge in enumerate(edges):
        prompt = build_edge_prompt(complaints, edge)
        raw_output = classifier(
            prompt,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        label = extract_label(raw_output, ["yes", "no"])
        if label == "UNKNOWN":
            raise ValueError(
                f"Unexpected classifier label: {label}; "
                f"raw_output={raw_output!r}; edge={edge_to_text(edge)!r}"
            )
        score = score_label_to_float(label)
        edge_scores.append(
            {
                "edge_id": edge_id,
                "from": edge["from"],
                "to": edge["to"],
                "score": score,
                "label": label,
                "raw_output": raw_output,
            }
        )

    average_edge_score = sum(x["score"] for x in edge_scores) / len(edge_scores)

    return {
        "id": item.get("id", idx),
        "source_title": item.get("source_title") or item.get("title"),
        input_field: complaints,
        graph_field: edges,
        "edge_scores": edge_scores,
        "average_edge_score": average_edge_score,
        "score_error": None,
        "original_item": item,
    }


def process_samples(
    input_path: str,
    output_path: str,
    error_output_path: str,
    input_field: str,
    graph_field: str,
    limit: Optional[int],
    max_retries: int,
    sleep_seconds: float,
) -> None:
    items = read_jsonl(input_path)

    if limit is not None:
        items = items[:limit]

    for path in (output_path, error_output_path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    processed = 0
    with open(output_path, mode="w", encoding="utf-8") as writer:
        with open(error_output_path, mode="w", encoding="utf-8") as writer_err:
            for idx, item in enumerate(items):
                try:
                    result = score_one_sample(
                        item=item,
                        input_field=input_field,
                        graph_field=graph_field,
                        idx=idx,
                        max_retries=max_retries,
                        sleep_seconds=sleep_seconds,
                    )
                    write_jsonl_row(writer, result)
                    processed += 1
                    if processed % 5 == 0:
                        writer.flush()
                        os.fsync(writer.fileno())
                except Exception as exc:
                    write_jsonl_row(
                        writer_err,
                        {
                            "id": item.get("id", idx),
                            "source_title": item.get("source_title") or item.get("title"),
                            "score_error": repr(exc),
                            "original_item": item,
                        }
                    )
                    writer_err.flush()
                    os.fsync(writer_err.fileno())

    print(f"Processed {len(items)} rows and saved {processed} scored samples to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score each network edge with the Azure fine-tuned edge classifier (ANGEL_EDGE_CLASSIFIER_DEPLOYMENT) and compute per-sample average scores.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path containing graph edges to score.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for scored graph samples.",
    )
    parser.add_argument(
        "--error-output",
        required=True,
        help="Output JSONL path for scoring failures.",
    )
    parser.add_argument(
        "--input-field",
        default="long_profile_description",
        help="Field containing the patient profile text used as classifier context.",
    )
    parser.add_argument(
        "--graph-field",
        default="graph",
        help="Field containing the graph edge list.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_samples(
        input_path=args.input,
        output_path=args.output,
        error_output_path=args.error_output,
        input_field=args.input_field,
        graph_field=args.graph_field,
        limit=args.limit,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
