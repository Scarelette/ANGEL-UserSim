"""Semantic Diversity metric with informative-response coverage adjustment.

This metric is profile-level semantic diversity across repeated runs.
It first filters per-topic patient text by an informativeness rule, then
computes embedding-space pairwise diversity among informative runs.
The final score is coverage-adjusted:

semantic_diversity = raw_semantic_diversity * informative_run_ratio
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments.profile_expansion.metrics.common import mean


METRIC_VARIANT = "informative_coverage_adjusted_v1"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_EMBED_MODEL_CACHE: Dict[str, Any] = {}


def _get_embed_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    model = _EMBED_MODEL_CACHE.get(model_name)
    if model is not None:
        return model

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _EMBED_MODEL_CACHE[model_name] = model
    return model


def _clean_text_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _is_informative_text(
    text: str,
    *,
    min_words: int,
    min_unique_words: int,
) -> bool:
    words = _clean_text_words(text)
    if len(words) < min_words:
        return False
    return len(set(words)) >= min_unique_words


def _trim_words(text: str, max_words: Optional[int]) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    if max_words is None or max_words <= 0:
        return cleaned
    return " ".join(cleaned.split()[:max_words])


def _extract_topic_patient_turns(record: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    transcript = record.get("transcript", [])
    if not isinstance(transcript, list):
        return {}

    topic_map: Dict[str, Dict[str, Any]] = {}

    # New format: [{"topic_key","topic_name","turns":[...]}]
    has_topic_blocks = any(isinstance(item, dict) and isinstance(item.get("turns"), list) for item in transcript)
    if has_topic_blocks:
        for item in transcript:
            if not isinstance(item, dict):
                continue
            topic_key = str(item.get("topic_key") or "")
            turns = item.get("turns")
            if not topic_key or not isinstance(turns, list):
                continue
            topic_name = str(item.get("topic_name") or topic_key)
            texts = [
                (turn.get("content") or "").strip()
                for turn in turns
                if isinstance(turn, dict) and turn.get("role") == "patient" and (turn.get("content") or "").strip()
            ]
            if texts:
                topic_map[topic_key] = {
                    "topic_key": topic_key,
                    "topic_name": topic_name,
                    "texts": texts,
                }
        return topic_map

    # Legacy flat format.
    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "patient":
            continue
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        topic_key = str(turn.get("topic_key") or "__unknown_topic__")
        topic_name = str(turn.get("topic_name") or topic_key)
        if topic_key not in topic_map:
            topic_map[topic_key] = {
                "topic_key": topic_key,
                "topic_name": topic_name,
                "texts": [],
            }
        topic_map[topic_key]["texts"].append(content)

    return topic_map


def _record_topic_texts(
    record: Mapping[str, Any],
    *,
    turns_per_topic: int,
    max_words_per_topic: Optional[int],
    min_words_informative: int,
    min_unique_words_informative: int,
) -> Dict[str, Dict[str, Any]]:
    topic_turns = _extract_topic_patient_turns(record)
    out: Dict[str, Dict[str, Any]] = {}

    k = max(1, int(turns_per_topic))
    for topic_key, topic_item in topic_turns.items():
        texts = topic_item.get("texts") or []
        selected = texts[:k]
        joined = _trim_words("\n".join(selected), max_words_per_topic)
        if not joined:
            continue
        out[topic_key] = {
            "topic_name": str(topic_item.get("topic_name") or topic_key),
            "text": joined,
            "informative": _is_informative_text(
                joined,
                min_words=min_words_informative,
                min_unique_words=min_unique_words_informative,
            ),
        }
    return out


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    sim = float(np.dot(left, right))
    return max(0.0, min(2.0, 1.0 - sim))


def _build_embeddings(
    texts: Sequence[str],
    *,
    embedding_model: str,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    embedding_cache_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    if not texts:
        return {}

    # De-duplicate while preserving order.
    unique_texts = list(dict.fromkeys(str(text) for text in texts))
    vectors_by_text: Dict[str, np.ndarray] = {}
    missing_texts: List[str] = []

    for text in unique_texts:
        cache_key = hashlib.md5(f"{embedding_model}\n{text}".encode("utf-8")).hexdigest()
        cached = embedding_cache.get(cache_key) if embedding_cache is not None else None
        if isinstance(cached, np.ndarray):
            vectors_by_text[text] = cached
            if embedding_cache_stats is not None:
                embedding_cache_stats["semantic_embedding_hits"] = (
                    embedding_cache_stats.get("semantic_embedding_hits", 0) + 1
                )
        else:
            missing_texts.append(text)
            if embedding_cache_stats is not None:
                embedding_cache_stats["semantic_embedding_misses"] = (
                    embedding_cache_stats.get("semantic_embedding_misses", 0) + 1
                )

    if missing_texts:
        model = _get_embed_model(embedding_model)
        encoded = np.asarray(
            model.encode(
                missing_texts,
                normalize_embeddings=True,
                batch_size=256,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        for idx, text in enumerate(missing_texts):
            vector = encoded[idx]
            vectors_by_text[text] = vector
            if embedding_cache is not None:
                cache_key = hashlib.md5(f"{embedding_model}\n{text}".encode("utf-8")).hexdigest()
                embedding_cache[cache_key] = vector

    return vectors_by_text


def score_semantic_diversity(
    records: List[Dict[str, Any]],
    *,
    turns_per_topic: int = 1,
    max_words_per_topic: Optional[int] = 60,
    min_words_informative: int = 20,
    min_unique_words_informative: int = 8,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    existing_profiles: Optional[List[Dict[str, Any]]] = None,
    on_profile_scored: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    embedding_cache_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record.get("model"), record.get("profile_id"))].append(record)

    print(
        f"[SemanticDiversity] start num_records={len(records)} "
        f"num_model_profile_groups={len(grouped)} variant={METRIC_VARIANT}",
        flush=True,
    )

    existing_by_key: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    if existing_profiles:
        for item in existing_profiles:
            if not isinstance(item, dict):
                continue
            if item.get("metric_variant") != METRIC_VARIANT:
                continue
            key = (item.get("model"), item.get("profile_id"))
            if key in grouped:
                existing_by_key[key] = item

    profile_scores = list(existing_by_key.values())
    if existing_by_key:
        print(
            f"[SemanticDiversity] resume existing_profiles={len(existing_by_key)}",
            flush=True,
        )

    total_groups = len(grouped)

    for (model, profile_id), group in grouped.items():
        if (model, profile_id) in existing_by_key:
            print(
                f"[SemanticDiversity] model={model} profile_id={profile_id} skipped (from checkpoint)",
                flush=True,
            )
            continue

        per_run_topics = [
            _record_topic_texts(
                record,
                turns_per_topic=turns_per_topic,
                max_words_per_topic=max_words_per_topic,
                min_words_informative=min_words_informative,
                min_unique_words_informative=min_unique_words_informative,
            )
            for record in group
        ]

        informative_texts: Dict[str, None] = {}
        for topic_map in per_run_topics:
            for item in topic_map.values():
                if item.get("informative"):
                    informative_texts[str(item["text"])] = None
        embeddings = _build_embeddings(
            list(informative_texts.keys()),
            embedding_model=embedding_model,
            embedding_cache=embedding_cache,
            embedding_cache_stats=embedding_cache_stats,
        )

        topic_keys = sorted({topic_key for run in per_run_topics for topic_key in run.keys()})
        topic_scores: List[Dict[str, Any]] = []
        total_topic_runs = 0
        informative_topic_runs = 0

        for topic_key in topic_keys:
            topic_name = next(
                (run[topic_key]["topic_name"] for run in per_run_topics if topic_key in run),
                topic_key,
            )
            informative_vecs: List[np.ndarray] = []
            num_runs_with_topic = 0

            for run in per_run_topics:
                item = run.get(topic_key)
                if not item:
                    continue
                num_runs_with_topic += 1
                total_topic_runs += 1
                if item.get("informative"):
                    informative_topic_runs += 1
                    informative_vecs.append(embeddings[str(item["text"])])

            if len(informative_vecs) < 2:
                topic_scores.append(
                    {
                        "topic_key": topic_key,
                        "topic_name": topic_name,
                        "num_runs_with_topic": num_runs_with_topic,
                        "num_informative_runs": len(informative_vecs),
                        "avg_pairwise_cosine_distance": None,
                    }
                )
                continue

            pairwise = [
                _cosine_distance(informative_vecs[i], informative_vecs[j])
                for i, j in combinations(range(len(informative_vecs)), 2)
            ]
            topic_scores.append(
                {
                    "topic_key": topic_key,
                    "topic_name": topic_name,
                    "num_runs_with_topic": num_runs_with_topic,
                    "num_informative_runs": len(informative_vecs),
                    "avg_pairwise_cosine_distance": mean(pairwise),
                }
            )

        scored_topic_values = [
            item["avg_pairwise_cosine_distance"]
            for item in topic_scores
            if item["avg_pairwise_cosine_distance"] is not None
        ]
        raw_diversity = max(0.0, mean(scored_topic_values) if scored_topic_values else 0.0)
        informative_run_ratio = informative_topic_runs / total_topic_runs if total_topic_runs else 0.0
        adjusted_diversity = raw_diversity * informative_run_ratio

        print(
            f"[SemanticDiversity] model={model} profile_id={profile_id} num_runs={len(group)} "
            f"num_topics_total={len(topic_keys)} num_topics_scored={len(scored_topic_values)} "
            f"raw={raw_diversity:.3f} coverage={informative_run_ratio:.3f} adjusted={adjusted_diversity:.3f}",
            flush=True,
        )

        profile_score = {
            "model": model,
            "profile_id": profile_id,
            "num_runs": len(group),
            "num_topics_total": len(topic_keys),
            "num_topics_scored": len(scored_topic_values),
            "total_topic_runs": total_topic_runs,
            "informative_topic_runs": informative_topic_runs,
            "informative_run_ratio": informative_run_ratio,
            "topics": topic_scores,
            "metric_variant": METRIC_VARIANT,
            "semantic_diversity_raw": raw_diversity,
            "semantic_diversity_adjusted": adjusted_diversity,
            # Keep backward-compatible key name for downstream tooling.
            "semantic_diversity": adjusted_diversity,
        }

        profile_scores.append(profile_score)
        if on_profile_scored:
            on_profile_scored(profile_score, len(profile_scores), total_groups)

    profile_scores.sort(key=lambda item: (str(item.get("model")), str(item.get("profile_id"))))

    overall_adjusted = mean(item.get("semantic_diversity", 0.0) for item in profile_scores)
    overall_raw = mean(item.get("semantic_diversity_raw", 0.0) for item in profile_scores)
    print(
        f"[SemanticDiversity] complete overall_adjusted={overall_adjusted:.3f} "
        f"overall_raw={overall_raw:.3f}",
        flush=True,
    )
    return {
        "metric": "semantic_diversity",
        "variant": METRIC_VARIANT,
        "config": {
            "turns_per_topic": max(1, int(turns_per_topic)),
            "max_words_per_topic": max_words_per_topic,
            "min_words_informative": max(1, int(min_words_informative)),
            "min_unique_words_informative": max(1, int(min_unique_words_informative)),
            "embedding_model": embedding_model,
        },
        "score": overall_adjusted,
        "score_raw": overall_raw,
        "profiles": profile_scores,
    }
