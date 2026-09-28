import argparse
import json
import math
import os
import re
from collections import Counter
from itertools import combinations
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


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


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def select_profiles(profiles: List[str], num_profiles: Optional[int]) -> List[str]:
    """
    Select a subset of generated profiles to participate in metrics calculation.
    If num_profiles is None or greater than or equal to the total number of profiles,
    return all profiles. Otherwise, return the first num_profiles profiles.
    If num_profiles is specified, it must be at least 2.
    """
    if num_profiles is None or num_profiles >= len(profiles):
        return profiles
    if num_profiles < 2:
        raise ValueError("num_profiles must be at least 2 if specified.")
    return profiles[:num_profiles]


def build_global_idf(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    doc_freq: Counter = Counter()
    num_docs = 0
    for row in rows:
        for text in row.get("generated_profiles", []):
            if not isinstance(text, str) or not text.strip():
                continue
            doc_freq.update(set(tokenize(text)))
            num_docs += 1

    if num_docs == 0:
        return {}

    return {
        token: math.log((1.0 + num_docs) / (1.0 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def tfidf_vector(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    tokens = tokenize(text)
    if not tokens:
        return {}

    term_freq = Counter(tokens)
    total = len(tokens)
    vector = {
        token: (count / total) * idf.get(token, 1.0)
        for token, count in term_freq.items()
    }

    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {token: value / norm for token, value in vector.items()}


def sparse_cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(value * vec_b.get(token, 0.0) for token, value in vec_a.items())


def count_ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def closest_reference_length(candidate_len: int, reference_lens: Sequence[int]) -> int:
    return min(reference_lens, key=lambda ref_len: (abs(ref_len - candidate_len), ref_len))


def bleu_score(candidate_tokens: Sequence[str], reference_tokens_list: Sequence[Sequence[str]], max_order: int) -> float:
    if not candidate_tokens:
        return 0.0
    if not reference_tokens_list:
        return 0.0

    precisions: List[float] = []
    for order in range(1, max_order + 1):
        cand_counts = count_ngrams(candidate_tokens, order)
        total = sum(cand_counts.values())
        if total == 0:
            precisions.append(0.0)
            continue

        max_ref_counts: Counter = Counter()
        for ref_tokens in reference_tokens_list:
            ref_counts = count_ngrams(ref_tokens, order)
            for ngram, count in ref_counts.items():
                if count > max_ref_counts[ngram]:
                    max_ref_counts[ngram] = count

        overlap = sum(min(count, max_ref_counts[ngram]) for ngram, count in cand_counts.items())
        precisions.append((overlap + 1.0) / (total + 1.0))

    if any(p == 0.0 for p in precisions):
        return 0.0

    log_precision = sum(math.log(p) for p in precisions) / max_order
    cand_len = len(candidate_tokens)
    ref_len = closest_reference_length(cand_len, [len(ref) for ref in reference_tokens_list])
    brevity_penalty = 1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / cand_len))
    return brevity_penalty * math.exp(log_precision)


def self_bleu(texts: Sequence[str], max_order: int) -> float:
    if len(texts) < 2:
        return 0.0

    tokenized = [tokenize(text) for text in texts]
    scores: List[float] = []
    for idx, candidate_tokens in enumerate(tokenized):
        references = tokenized[:idx] + tokenized[idx + 1 :]
        scores.append(bleu_score(candidate_tokens, references, max_order=max_order))
    return float(mean(scores))


class EmbeddingEncoder:
    def __init__(self, backend: str, model_name: str, idf: Optional[Dict[str, float]] = None) -> None:
        self.backend = backend
        self.model_name = model_name
        self.idf = idf or {}
        self._model = None

        if backend == "sentence-transformers":
            if SentenceTransformer is None:
                raise ImportError("sentence_transformers is not installed.")
            self._model = SentenceTransformer(model_name, device="cpu")

    def pairwise_cosine_distance_stats(self, texts: Sequence[str]) -> Dict[str, float]:
        if len(texts) < 2:
            return {
                "mean_pairwise_cosine_distance": 0.0,
                "min_pairwise_cosine_distance": 0.0,
                "max_pairwise_cosine_distance": 0.0,
                "median_pairwise_cosine_distance": 0.0,
            }

        if self.backend == "sentence-transformers":
            embeddings = self._model.encode(list(texts), normalize_embeddings=True)
            distances = [
                float(1.0 - float(np.dot(embeddings[i], embeddings[j])))
                for i, j in combinations(range(len(texts)), 2)
            ]
        else:
            vectors = [tfidf_vector(text, self.idf) for text in texts]
            distances = [
                float(1.0 - sparse_cosine_similarity(vectors[i], vectors[j]))
                for i, j in combinations(range(len(vectors)), 2)
            ]

        return {
            "mean_pairwise_cosine_distance": float(mean(distances)),
            "min_pairwise_cosine_distance": float(min(distances)),
            "max_pairwise_cosine_distance": float(max(distances)),
            "median_pairwise_cosine_distance": float(median(distances)),
        }


def build_encoder(backend: str, model_name: str, rows: Sequence[Dict[str, Any]]) -> EmbeddingEncoder:
    if backend == "auto":
        if SentenceTransformer is not None:
            return EmbeddingEncoder("sentence-transformers", model_name=model_name)
        return EmbeddingEncoder("tfidf", model_name="tfidf", idf=build_global_idf(rows))

    if backend == "sentence-transformers":
        return EmbeddingEncoder("sentence-transformers", model_name=model_name)

    if backend == "tfidf":
        return EmbeddingEncoder("tfidf", model_name="tfidf", idf=build_global_idf(rows))

    raise ValueError(f"Unsupported backend: {backend}")


def pairwise_cosine_distance_stats(texts: Sequence[str], encoder: EmbeddingEncoder) -> Dict[str, float]:
    if len(texts) < 2:
        return {
            "mean_pairwise_cosine_distance": 0.0,
            "min_pairwise_cosine_distance": 0.0,
            "max_pairwise_cosine_distance": 0.0,
            "median_pairwise_cosine_distance": 0.0,
        }
    return encoder.pairwise_cosine_distance_stats(texts)


def analyze_row(row: Dict[str, Any], bleu_order: int, encoder: EmbeddingEncoder, num_profiles: Optional[int]) -> Dict[str, Any]:
    short_profile_id = row.get("short_profile_id")
    short_profile = row.get("short_profile")
    generated_profiles = row.get("generated_profiles")

    if not isinstance(short_profile_id, str) or not short_profile_id:
        raise ValueError("Each row must contain a non-empty 'short_profile_id'.")
    if not isinstance(short_profile, str) or not short_profile.strip():
        raise ValueError(f"Row {short_profile_id} is missing a non-empty 'short_profile'.")
    if not isinstance(generated_profiles, list) or not generated_profiles:
        raise ValueError(f"Row {short_profile_id} is missing a non-empty 'generated_profiles' list.")
    if not all(isinstance(text, str) and text.strip() for text in generated_profiles):
        raise ValueError(f"Row {short_profile_id} contains an empty generated profile.")

    cleaned_profiles = [text.strip() for text in generated_profiles]
    cleaned_profiles = select_profiles(cleaned_profiles, num_profiles)
    cosine_stats = pairwise_cosine_distance_stats(cleaned_profiles, encoder=encoder)

    return {
        "short_profile_id": short_profile_id,
        "original_id": row.get("original_id"),
        "source_title": row.get("source_title"),
        "num_generated_profiles": len(cleaned_profiles),
        "cosine_backend": encoder.backend,
        "cosine_model": encoder.model_name,
        "self_bleu": self_bleu(cleaned_profiles, max_order=bleu_order),
        **cosine_stats,
    }


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {}

    keys = [
        "self_bleu",
        "mean_pairwise_cosine_distance",
        "min_pairwise_cosine_distance",
        "max_pairwise_cosine_distance",
        "median_pairwise_cosine_distance",
    ]
    summary: Dict[str, float] = {"num_short_profiles": len(results)}
    for key in keys:
        values = [float(row[key]) for row in results]
        summary[f"avg_{key}"] = float(mean(values))
        summary[f"median_{key}"] = float(median(values))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute diversity metrics for generated long profiles grouped by short profile."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Merged JSONL file produced by merge_long_profiles.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Per-short-profile diversity report in JSONL format.",
    )
    parser.add_argument(
        "--summary-output",
        required=True,
        help="Overall summary metrics in JSON format.",
    )
    parser.add_argument(
        "--bleu-order",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Maximum n-gram order for self-BLEU.",
    )
    parser.add_argument(
        "--embedding-backend",
        default="sentence-transformers",
        choices=["auto", "sentence-transformers", "tfidf"],
        help="Backend for cosine distance. 'auto' prefers sentence-transformers and falls back to tfidf.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name when using that backend.",
    )
    parser.add_argument(
        "--num-profiles",
        type=int,
        default=None,
        help="Number of generated profiles per short profile to include in metrics calculation. "
             "If None, use all profiles. Must be at least 2 if specified.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = read_jsonl(args.input)
    encoder = build_encoder(args.embedding_backend, args.embedding_model, rows)
    results = [analyze_row(row, bleu_order=args.bleu_order, encoder=encoder, num_profiles=args.num_profiles) for row in rows]
    summary = summarize(results)
    summary["cosine_backend"] = encoder.backend
    summary["cosine_model"] = encoder.model_name

    write_jsonl(args.output, results)
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Analyzed {len(results)} short profiles from {args.input}")
    print(f"Per-case report: {args.output}")
    print(f"Summary report: {args.summary_output}")
    if summary:
        print(
            "Average metrics: "
            f"self_bleu={summary['avg_self_bleu']:.4f}, "
            f"mean_pairwise_cosine_distance={summary['avg_mean_pairwise_cosine_distance']:.4f}"
        )
        print(f"Cosine backend: {encoder.backend} ({encoder.model_name})")


if __name__ == "__main__":
    main()
