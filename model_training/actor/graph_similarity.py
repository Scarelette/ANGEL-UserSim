"""Graph reconstruction score: how close a reconstructed network is to the conditioning one.

Used by ``rollout_graph_dpo``. The Observer names nodes in its own words, so
nodes are first aligned one-to-one by sentence-embedding cosine similarity
(greedy, highest pair first; a pair counts only at ``threshold`` or above).
Edges are then compared through that alignment, keeping their direction.

The threshold is lower than the Observer's S1 node reward (0.75). There the
Observer and GPT-5 name nodes from the same complaints text; here the Observer
only hears the patient talk, so its names are looser paraphrases. On test
conversations, correct matches scored 0.61-1.0 ("insomnia" / "poor sleep"
0.61) and wrong ones 0.45 or less, so 0.55 sits in the gap.

    score = node_weight * node_F1 + (1 - node_weight) * edge_F1      (in [0, 1])

The result also has the recall of the visible and of the masked conditioning
edges, to see which part of the network a conversation expressed.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.55
DEFAULT_NODE_WEIGHT = 0.5

Edge = Tuple[str, str]
EmbedFn = Callable[[List[str]], np.ndarray]

_MODELS: Dict[str, object] = {}


def sentence_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu") -> EmbedFn:
    """Normalised sentence embeddings from ``sentence-transformers`` (model cached per name)."""
    if model_name not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS[model_name] = SentenceTransformer(model_name, device=device)
    model = _MODELS[model_name]
    return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True))


def _norm(name: str) -> str:
    return " ".join(str(name).split()).strip()


def edge_pairs(edges: Iterable[dict]) -> List[Edge]:
    """``(from, to)`` pairs from ``{"from", "to"}`` edges or mask rows ``{"value": {"from", "to"}}``."""
    out: List[Edge] = []
    for e in edges or []:
        if not isinstance(e, dict):
            continue
        e = e.get("value", e)
        if isinstance(e, dict) and e.get("from") and e.get("to"):
            out.append((_norm(e["from"]), _norm(e["to"])))
    return out


def nodes_of(symptoms: Optional[Iterable[str]], edges: Iterable[Edge]) -> List[str]:
    """Listed symptoms plus every edge endpoint, de-duplicated in order."""
    seen: Dict[str, None] = {}
    for n in list(symptoms or []) + [n for e in edges for n in e]:
        n = _norm(n)
        if n:
            seen.setdefault(n, None)
    return list(seen)


def align_nodes(pred: Sequence[str], ref: Sequence[str], embed: EmbedFn,
                threshold: float = DEFAULT_THRESHOLD) -> Dict[str, str]:
    """One-to-one map predicted node -> reference node, greedy by cosine, pairs >= ``threshold`` only."""
    if not pred or not ref:
        return {}
    sims = embed(list(pred)) @ embed(list(ref)).T
    order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]
    mapping: Dict[str, str] = {}
    used_ref: Set[int] = set()
    for i, j in order:
        if sims[i, j] < threshold:
            break
        if pred[i] in mapping or j in used_ref:
            continue
        mapping[pred[i]] = ref[j]
        used_ref.add(j)
    return mapping


def _f1(tp: int, n_pred: int, n_ref: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def graph_similarity(
    pred_symptoms: Optional[Iterable[str]],
    pred_edges: Iterable[dict],
    ref_symptoms: Optional[Iterable[str]],
    visible_edges: Iterable[dict],
    masked_edges: Iterable[dict] = (),
    *,
    embed: Optional[EmbedFn] = None,
    threshold: float = DEFAULT_THRESHOLD,
    node_weight: float = DEFAULT_NODE_WEIGHT,
) -> Dict[str, float]:
    """Score a reconstructed network against the conditioning network (visible + masked edges)."""
    embed = embed or sentence_embedder()
    pred_e = set(edge_pairs(pred_edges))
    vis_e, mask_e = set(edge_pairs(visible_edges)), set(edge_pairs(masked_edges))
    ref_e = vis_e | mask_e
    pred_n = nodes_of(pred_symptoms, pred_e)
    ref_n = nodes_of(ref_symptoms, ref_e)

    mapping = align_nodes(pred_n, ref_n, embed, threshold)
    node_p, node_r, node_f1 = _f1(len(mapping), len(pred_n), len(ref_n))

    mapped_e = {(mapping[a], mapping[b]) for a, b in pred_e if a in mapping and b in mapping}
    edge_p, edge_r, edge_f1 = _f1(len(mapped_e & ref_e), len(pred_e), len(ref_e))

    return {
        "score": node_weight * node_f1 + (1.0 - node_weight) * edge_f1,
        "node_precision": node_p, "node_recall": node_r, "node_f1": node_f1,
        "edge_precision": edge_p, "edge_recall": edge_r, "edge_f1": edge_f1,
        "visible_edge_recall": len(mapped_e & vis_e) / len(vis_e) if vis_e else 0.0,
        "masked_edge_recall": len(mapped_e & mask_e) / len(mask_e) if mask_e else 0.0,
        "n_pred_nodes": len(pred_n), "n_pred_edges": len(pred_e),
        "n_ref_nodes": len(ref_n), "n_ref_edges": len(ref_e),
    }
