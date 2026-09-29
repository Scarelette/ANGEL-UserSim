"""GRPO reward functions for the Observer (TRL ``reward_funcs`` signature).

S1 (nodes):  0.4 * format + 0.6 * node semantic recall vs. GPT-5 reference nodes.
S2 (edges):  LLM-judge edge reward (gpt-5-mini; see ``EdgeRewardConfig``):
                        w_format * format + w_precision * precision + w_coverage * coverage
                        − soft size penalty
                      precision = mean judge plausibility of the proposed edges in [-1, 1]
                      (edges that break the rules score -1 without a judge call);
                      coverage  = share of listed nodes joined by a supported edge, in [-1, 1]

Group-relative normalisation of the final reward is left to GRPO itself, so
edge scores are *not* normalised inside a rollout: the edge term is the raw
mean plausibility, which is what makes plausible graphs score higher.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------- #
# Format rewards
# --------------------------------------------------------------------------- #
def format_reward_s1(json_str: str) -> float:
    try:
        obj = json.loads(json_str)
    except Exception:
        return -1.0
    if set(obj.keys()) != {"symptoms", "external_factors"}:
        return -0.8
    if (
        not isinstance(obj["symptoms"], list)
        or not isinstance(obj["external_factors"], list)
        or not all(isinstance(x, str) for x in obj["symptoms"])
        or not all(isinstance(x, str) for x in obj["external_factors"])
    ):
        return -0.6
    return 0.5


def format_reward_s2(json_str: str) -> float:
    try:
        obj = json.loads(json_str)
    except Exception:
        return -1.0
    if set(obj.keys()) != {"links"}:
        return -0.8
    links = obj["links"]
    if not isinstance(links, list):
        return -0.6
    for e in links:
        if not isinstance(e, dict) or "from" not in e or "to" not in e:
            return -0.6
    return 0.5


def _match_json(match_regex, text: str) -> Optional[str]:
    m = match_regex.search(text)
    if m is None:
        return None
    return m.group("json") if "json" in m.groupdict() else m.group(1)


# --------------------------------------------------------------------------- #
# S1: node semantic recall
# --------------------------------------------------------------------------- #
_ST_MODEL = None


def _sentence_model(name: str = "all-MiniLM-L6-v2"):
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _ST_MODEL = SentenceTransformer(name, device="cpu")
    return _ST_MODEL


def node_semantic_recall(ref_nodes: List[str], pred_nodes: List[str], threshold: float = 0.75) -> float:
    """Fraction of reference nodes matched (cosine >= threshold) by a predicted node, mapped to [-1, 1]."""
    if not ref_nodes or not pred_nodes:
        return -1.0
    import torch

    model = _sentence_model()
    with torch.no_grad():
        ref_emb = model.encode(ref_nodes, normalize_embeddings=True)
        pred_emb = model.encode(pred_nodes, normalize_embeddings=True)
    max_sims = (ref_emb @ pred_emb.T).max(axis=1)
    recall = (max_sims >= threshold).sum() / len(ref_nodes)
    return 2.0 * recall - 1.0


def symptom_graph_reward_s1(match_regex) -> Callable:
    def reward_fn(prompts, completions, ref_nodes, **kwargs):
        rewards = []
        for completion, refs in zip(completions, ref_nodes):
            json_str = _match_json(match_regex, completion[0]["content"])
            if json_str is None:
                rewards.append(-1.0)
                continue
            try:
                obj = json.loads(json_str)
            except Exception:
                rewards.append(-1.0)
                continue
            pred_nodes = obj.get("symptoms", []) + obj.get("external_factors", [])
            if not pred_nodes:
                rewards.append(-1.0)
                continue
            r_format = format_reward_s1(json_str)
            r_nodes = node_semantic_recall(refs, pred_nodes)
            rewards.append(0.4 * r_format + 0.6 * r_nodes)
        return np.array(rewards, dtype=np.float32).tolist()

    return reward_fn


# --------------------------------------------------------------------------- #
# S2 (azure): LLM-judge edge plausibility
# --------------------------------------------------------------------------- #
class AzureAsyncRewardEngine:
    """Scores a batch of edges 0..1 with an Azure OpenAI chat deployment (gpt-5-mini in the paper)."""

    def __init__(
        self,
        client,
        deployment_name: str,
        max_concurrent: int = 3,
        max_retries: int = 5,
        temperature: float = 1.0,
    ):
        self.client = client
        self.deployment_name = deployment_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.temperature = temperature

    def build_prompt(self, complaints: str, edges: List[str]) -> str:
        prompt = f"""This is the Presenting Complaints of the mental health patient:

{complaints}

Below are several proposed symptom relationships.

For each relationship:
- Evaluate whether the link is clinically plausible based on the presenting complaints.
- Score each link from 0 to 1.
    1 = clearly supported by the complaints
    0 = clearly not supported
    0.5 = uncertain / weakly implied

Return your answer in STRICT JSON format as a list of objects:
[
  {{"edge_id": 0, "score": 0.85}},
  {{"edge_id": 1, "score": 0.20}}
]

Symptom Relationships:
"""
        for i, edge in enumerate(edges):
            prompt += f"{i}. {edge}\n"
        return prompt

    @staticmethod
    def safe_json_parse(text: str):
        try:
            return json.loads(text)
        except Exception:
            text = text.strip()
            if "[" in text:
                text = text[text.find("["):]
            if "]" in text:
                text = text[: text.rfind("]") + 1]
            return json.loads(text)

    async def call_api(self, prompt: str) -> str:
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {"role": "system", "content": "You are a professional mental health expert."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print("Azure error:", type(e), repr(e))
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError("Max retries exceeded")

    async def score_batch(self, complaints: str, edges: List[str]) -> np.ndarray:
        raw_output = await self.call_api(self.build_prompt(complaints, edges))
        results = self.safe_json_parse(raw_output)
        reward_dict = {int(item["edge_id"]): float(item["score"]) for item in results}
        # 0..1 judge score -> [-1, 1]; an edge the judge skipped counts as uncertain (0.5).
        return np.array([2 * min(max(reward_dict.get(i, 0.5), 0.0), 1.0) - 1 for i in range(len(edges))])


@dataclass
class EdgeRewardConfig:
    """Weights of the S2 edge reward. Defaults sum to 1 so the reward stays in about [-1, 1]."""

    w_format: float = 0.1       # format reward is 0.5 when valid -> contributes 0.05
    w_precision: float = 0.6    # mean plausibility of proposed edges
    w_coverage: float = 0.3     # listed nodes connected by a supported edge
    support_threshold: float = 0.5   # judge score >= this counts as a supported edge
    max_edges: int = 10              # edges beyond this are penalised ...
    penalty_per_extra_edge: float = 0.05
    max_size_penalty: float = 0.3    # ... up to this much


def _norm_node(name: str) -> str:
    return " ".join(str(name).lower().split())


def _edge_key(complaints: str, src: str, dst: str) -> str:
    return hashlib.sha256(f"{complaints}\x1f{_norm_node(src)}\x1f{_norm_node(dst)}".encode("utf-8")).hexdigest()


class EdgeJudge:
    """Scores edges 0..1 with the Azure judge, caching per (complaints, edge).

    The cache makes an edge proposed by several generations of the same prompt
    receive the same score, so GRPO's within-group comparison reflects the
    graphs rather than judge sampling noise, and avoids repeated API calls.
    """

    def __init__(self, engine: "AzureAsyncRewardEngine", max_cache: int = 200_000):
        self.engine = engine
        self.cache: Dict[str, float] = {}
        self.max_cache = max_cache

    def score(self, complaints: str, edges: List[Tuple[str, str]]) -> List[Optional[float]]:
        """0..1 score per edge, or None if the judge call failed."""
        keys = [_edge_key(complaints, a, b) for a, b in edges]
        todo = sorted({k: e for k, e in zip(keys, edges) if k not in self.cache}.items())
        if todo:
            try:
                raw = asyncio.run(self.engine.score_batch(complaints, [f"{a} -> {b}" for _, (a, b) in todo]))
                if len(self.cache) > self.max_cache:
                    self.cache.clear()
                for (k, _), r in zip(todo, raw):
                    self.cache[k] = (float(r) + 1.0) / 2.0  # [-1, 1] -> 0..1
            except Exception as e:
                print("Edge judge error:", e)
        return [self.cache.get(k) for k in keys]


def score_edge_graph(
    complaints: str,
    edges: List[Dict[str, Any]],
    nodes: Optional[List[str]],
    judge: EdgeJudge,
    cfg: EdgeRewardConfig,
) -> Dict[str, float]:
    """Precision / coverage / size penalty for one proposed network.

    Edges that break the task rules get plausibility -1 without a judge call:
    an endpoint not in the provided node list, a self-loop, or a duplicate.
    Returns None-free floats; if the judge fails, judged edges count as
    uncertain (0.5) so an API outage does not look like a bad graph.
    """
    node_set = {_norm_node(n) for n in nodes} if nodes else None
    seen = set()
    per_edge: List[Optional[float]] = []  # None = to be judged
    to_judge: List[Tuple[str, str]] = []
    for e in edges:
        a, b = str(e.get("from", "")), str(e.get("to", ""))
        na, nb = _norm_node(a), _norm_node(b)
        invalid = (
            not na or not nb or na == nb or (na, nb) in seen
            or (node_set is not None and (na not in node_set or nb not in node_set))
        )
        seen.add((na, nb))
        if invalid:
            per_edge.append(-1.0)
        else:
            per_edge.append(None)
            to_judge.append((a, b))

    judged = iter(judge.score(complaints, to_judge)) if to_judge else iter(())
    plaus: List[float] = []
    supported_nodes = set()
    edge_iter = iter(edges)
    for val in per_edge:
        e = next(edge_iter)
        if val is not None:
            plaus.append(val)
            continue
        s01 = next(judged)
        s01 = 0.5 if s01 is None else s01
        plaus.append(2.0 * s01 - 1.0)
        if s01 >= cfg.support_threshold:
            supported_nodes.update({_norm_node(e["from"]), _norm_node(e["to"])})

    precision = float(np.mean(plaus)) if plaus else -1.0
    if node_set:
        coverage = 2.0 * len(supported_nodes & node_set) / len(node_set) - 1.0
    else:
        coverage = 0.0
    extra = max(0, len(edges) - cfg.max_edges)
    size_penalty = min(cfg.max_size_penalty, cfg.penalty_per_extra_edge * extra)
    return {"precision": precision, "coverage": coverage, "size_penalty": size_penalty}


def symptom_graph_reward_s2_azure(
    match_regex,
    reward_engine: "AzureAsyncRewardEngine",
    config: Optional[EdgeRewardConfig] = None,
) -> Callable:
    cfg = config or EdgeRewardConfig()
    judge = EdgeJudge(reward_engine)

    def reward_fn(prompts, completions, complaints, nodes=None, **kwargs):
        node_lists = nodes if nodes is not None else [None] * len(completions)
        rewards = []
        for completion, complaint, node_list in zip(completions, complaints, node_lists):
            json_str = _match_json(match_regex, completion[0]["content"])
            if json_str is None:
                rewards.append(-1.0)
                continue
            r_format = format_reward_s2(json_str)
            if r_format < 0:
                rewards.append(r_format)
                continue
            edges = json.loads(json_str)["links"]
            if not edges:
                rewards.append(-1.0)
                continue
            parts = score_edge_graph(complaint, edges, node_list, judge, cfg)
            rewards.append(
                cfg.w_format * r_format
                + cfg.w_precision * parts["precision"]
                + cfg.w_coverage * parts["coverage"]
                - parts["size_penalty"]
            )
        return np.array(rewards, dtype=np.float32).tolist()

    return reward_fn
