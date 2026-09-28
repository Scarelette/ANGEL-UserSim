"""GRPO reward functions for the Observer (TRL ``reward_funcs`` signature).

S1 (nodes):  0.4 * format + 0.6 * node semantic recall vs. GPT-5 reference nodes.
S2 (edges):  one of three edge rewards
  - ``azure``       : 0.2 * format + 0.8 * LLM-judge edge plausibility − 0.2 if > 10 edges
                      (the reward used for the released Qwen3-Observer-800)
  - ``local``       : 0.3 * format + 0.5 * local Qwen3-0.6B Yes/No classifier score
  - ``format_only`` : 0.3 * format (clipped to [-0.5, 0.5]); optionally dumps every
                      proposed edge to JSONL — used to harvest edges for the
                      edge-classifier training data.

See README "Known issues" for the two behaviours kept for reproducibility:
the Azure judge's per-rollout z-normalisation (``edge_score_norm="group"``) and
the local reward's rank-0 broadcast (``per_rank=False``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional

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
        edge_score_norm: str = "group",
    ):
        if edge_score_norm not in ("group", "none"):
            raise ValueError("edge_score_norm must be 'group' or 'none'")
        self.client = client
        self.deployment_name = deployment_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.temperature = temperature
        self.edge_score_norm = edge_score_norm

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
        reward_dict = {item["edge_id"]: item["score"] for item in results}
        scores = np.array([2 * reward_dict[i] - 1 for i in range(len(edges))])  # -> [-1, 1]
        if self.edge_score_norm == "group":
            # Original behaviour: z-normalise within the rollout. The caller then
            # takes the mean, which is ~0 by construction (see README, Known issues).
            scores = (scores - scores.mean()) / (scores.std() + 1e-6)
        return scores


def llm_edge_plausibility(complaints: str, edges: List[Dict[str, str]], reward_engine: AzureAsyncRewardEngine) -> float:
    if not edges:
        return -1.0
    try:
        edge_strs = [f"{e['from']} -> {e['to']}" for e in edges]
        scores = asyncio.run(reward_engine.score_batch(complaints, edge_strs))
        return float(scores.mean())
    except Exception as e:
        print("Reward error:", e)
        return 0.0


def symptom_graph_reward_s2_azure(match_regex, reward_engine: AzureAsyncRewardEngine) -> Callable:
    def reward_fn(prompts, completions, complaints, **kwargs):
        rewards = []
        for completion, complaint in zip(completions, complaints):
            json_str = _match_json(match_regex, completion[0]["content"])
            if json_str is None:
                rewards.append(-1.0)
                continue
            try:
                edges = json.loads(json_str).get("links", [])
            except Exception:
                rewards.append(-1.0)
                continue
            r_format = format_reward_s2(json_str)
            if r_format < 0:
                rewards.append(r_format)
                continue
            r_edges = llm_edge_plausibility(complaint, edges, reward_engine)
            size_penalty = -0.2 if len(edges) > 10 else 0.0
            rewards.append(0.2 * r_format + 0.8 * r_edges + size_penalty)
        return np.array(rewards, dtype=np.float32).tolist()

    return reward_fn


# --------------------------------------------------------------------------- #
# S2 (local): Qwen3-0.6B Yes/No edge classifier
# --------------------------------------------------------------------------- #
def _edge_cache_key(complaints: str, edges: List[Dict[str, str]]) -> str:
    payload = {
        "complaints": complaints,
        "edges": [
            {"from": e["from"], "to": e["to"]}
            for e in edges
            if isinstance(e, dict) and "from" in e and "to" in e
        ],
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def symptom_graph_reward_s2_local(match_regex, model, tokenizer, per_rank: bool = False) -> Callable:
    """Local-classifier edge reward.

    per_rank=False reproduces the original run: only rank 0 scores (its own)
    completions and broadcasts the rewards to every rank. per_rank=True has each
    rank score its own completions (correct under DDP).
    """
    import torch
    import torch.distributed as dist

    def is_rank_0():
        return (not dist.is_initialized()) or dist.get_rank() == 0

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1, "Yes/No must be single tokens"
    yes_token_id, no_token_id = yes_ids[0], no_ids[0]
    edge_cache: Dict[str, float] = {}

    def edge_score(complaint: str, edges: List[Dict[str, str]]) -> float:
        if not edges:
            return -1.0
        key = _edge_cache_key(complaint, edges)
        if key in edge_cache:
            return edge_cache[key]
        device = next(model.parameters()).device
        total, valid = 0.0, 0
        for e in edges:
            if "from" not in e or "to" not in e:
                continue
            prompt = (
                f"This is the Presenting Complaints of the mental health patient:\n"
                f"{complaint}\n\n"
                f"Based on the patient's symptoms, I construct a symptom relationship:\n"
                f"- {e['from']} -> {e['to']}\n\n"
                f"Does this link make sense according to the presenting complaints?\n"
                f"Answer with exactly one word: Yes or No.\nAnswer:"
            )
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    last_logits = model(**inputs).logits[:, -1, :]
                diff = last_logits[0, yes_token_id] - last_logits[0, no_token_id]
                total += torch.tanh(diff / 5.0).item()  # -> [-1, 1]
                valid += 1
            except Exception:
                continue
        score = total / valid if valid else 0.0
        edge_cache[key] = score
        return score

    def compute(completions, complaints) -> List[float]:
        rewards = []
        for completion, complaint in zip(completions, complaints):
            json_str = _match_json(match_regex, completion[0]["content"])
            if json_str is None:
                rewards.append(-1.0)
                continue
            try:
                edges = json.loads(json_str).get("links", [])
            except Exception:
                rewards.append(-1.0)
                continue
            r_format = format_reward_s2(json_str)
            if r_format < 0:
                rewards.append(r_format)
                continue
            rewards.append(0.3 * r_format + 0.5 * edge_score(complaint, edges))
        return rewards

    def reward_fn(prompts, completions, complaints, **kwargs):
        if per_rank:
            return compute(completions, complaints)
        device = next(model.parameters()).device
        rewards = compute(completions, complaints) if is_rank_0() else [0.0] * len(completions)
        tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        if dist.is_initialized():
            dist.broadcast(tensor, src=0)
        return tensor.tolist()

    return reward_fn


# --------------------------------------------------------------------------- #
# S2 (format_only): edge harvesting for classifier data
# --------------------------------------------------------------------------- #
def symptom_graph_reward_s2_format_only(match_regex, edge_dump_path: Optional[str] = None) -> Callable:
    """Format-only S2 reward; rank 0 appends every proposed edge to ``edge_dump_path``.

    Dump rows: {complaint, from, to, model_output, step, reward_format} — the
    input format of ``edge_classifier_data.py label``.
    """

    def is_rank_0():
        import torch.distributed as dist

        return (not dist.is_initialized()) or dist.get_rank() == 0

    def reward_fn(prompts, completions, complaints, **kwargs):
        rewards = []
        step = kwargs.get("step", -1)
        for completion, complaint in zip(completions, complaints):
            text = completion[0]["content"]
            json_str = _match_json(match_regex, text)
            if json_str is None:
                rewards.append(-1.0)
                continue
            try:
                edges = json.loads(json_str).get("links", [])
            except Exception:
                rewards.append(-1.0)
                continue
            r_format = format_reward_s2(json_str)
            if r_format < 0:
                rewards.append(r_format)
                continue
            if edge_dump_path and is_rank_0():
                with open(edge_dump_path, "a", encoding="utf-8") as f:
                    for e in edges:
                        if "from" in e and "to" in e:
                            f.write(json.dumps({
                                "complaint": complaint, "from": e["from"], "to": e["to"],
                                "model_output": text, "step": step, "reward_format": r_format,
                            }, ensure_ascii=False) + "\n")
            rewards.append(max(min(0.3 * r_format, 0.5), -0.5))
        return rewards

    return reward_fn
