"""Step 1 — mask part of each symptom network ("hidden" edges the patient
does not initially recognize).

For every network (``symptoms`` + directed ``graph`` edges) GPT-5:
  1. classifies the case into a pattern from ``mask_pattern.jsonl``
     (depression / anxiety / substance use / trauma / mania / Psychosis), and
  2. labels every symptom as Behavior / Emotion / Physiological Sensation /
     Cognition / Stimulus.
Up to ``--mask-pct`` of the edges whose (parent type, child type) is maskable
for that pattern are moved to ``mask``; the rest go to ``new_graph``.

The paper ran this six times with --mask-pct 0.1 ... 0.6 (files p1 ... p6).

Both GPT-5 calls are retried until the pattern is one of the known names and
there is exactly one label per symptom.

    python -m model_training.actor.mask_generator \
        --input data/actor/network_models.jsonl \
        --output data/actor/masked/NM_mask_p1.jsonl --mask-pct 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from angel_common.llm import extract_tag_content, get_output
from angel_common.paths import DATA_DIR, resolve_path

DEFAULT_PATTERNS = Path(__file__).with_name("mask_pattern.jsonl")

# GPT-5 calls at data-generation time used a 16384-token completion budget.
_MAX_TOKENS = 16384


def read_mask_pattern(path: Path = DEFAULT_PATTERNS) -> dict:
    mask_pattern = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                mask_pattern[obj["pattern"]] = obj["possible_mask"]
    return mask_pattern


def match_pattern(reply: str, patterns) -> str | None:
    """Map GPT's reply to a pattern name, ignoring case, quotes and markdown."""
    by_lower = {p.lower(): p for p in patterns}
    text = re.sub(r"[*`'\".]", "", reply or "").strip().lower()
    if text in by_lower:
        return by_lower[text]
    found = [p for low, p in by_lower.items() if re.search(rf"\b{re.escape(low)}\b", text)]
    return found[0] if len(found) == 1 else None


def get_pattern(symptoms, mask_type, max_retry: int = 5) -> str:
    mask_type = list(mask_type)
    prompt = f"""Below are the patient's symptoms:

{symptoms}

Please classify the patient into one of the following categories:
{mask_type}

Only output the category name. Do not provide any explanation."""

    for attempt in range(max_retry):
        # tag=3: no tag / refusal validation (original call: getOutput(prompt, 3)).
        reply = get_output(prompt, tag=3, max_completion_tokens=_MAX_TOKENS).strip()
        pattern = match_pattern(reply, mask_type)
        if pattern is not None:
            return pattern
        print(f"[pattern attempt {attempt}] unknown pattern: {reply!r}")
    raise ValueError(f"no known pattern after {max_retry} attempts (last reply: {reply!r})")


def node_labeler(symptoms, max_retry: int = 5) -> dict:
    for attempt in range(max_retry):
        print("Attempt:", attempt)
        prompt = f"""
    Please classify each symptom into 5 types: Behavior, Emotion, Physiological Sensation, Cognition and Stimulus (i.e., an event). \nThis is the Symptom List: {symptoms}

    Behaviors are actions that people take. Behaviors are discrete; they start and stop and do not last forever. Behaviors can be voluntary or involuntary. Behaviors are either discussed as the presence of certain actions (e.g., the patient ran away, he started talking to themselves, she took a minute to think to herself) or the absence of behavior (e.g., he avoided large crowds, he was isolate to his room nearly all day). Examples of voluntary behaviors include: walking, talking, breathing heavily, avoiding eye contact, using complex word choices to describe your thoughts). Examples of involuntary behaviors include: yelling in response to something frightening, blinking, pulling your hand back after touching something hot, flinching when an objects approaches you quickly. Involuntary movements such as sweating and heartbeat are better classified as physiological sensations. 

    Emotions for the sake of labeling these nodes are subjective feelings people experience. Emotions are typically either described by an emotion word or a particular valence (positive vs negative, good vs bad) or arousal (high vs low arousal). Examples of difference emotion words are: anger, frustrated, annoyed, aggravated, agitated, provoked, hostile, irritable, exasperated, agitated, envy, jealous, resentful, disgust, contempt, revolted, sadness,agony, hurt, sadness, sorrow, depressed, disappointed, dismayed, displeased, regretful, guilty, lonely, despair, grief, surprise, stunned, shocked, dismayed, disillusioned, perplexed, amazed, astonished, awe-struck, overcome, speechless, astounded, moved, stimulated, touched, delighted, pleased, amused, joy, optimistic, hopeful, inspired, cheerful, proud, triumphant, happy, blissful, joyful, content, delighted, pleased, amused, enthusiastic, excited, love, affectionate, fondness, romantic, longing, sentimental, attracted, desire, passion, infatuation, tenderness, caring, compassionate, peaceful, satisfied, relieved, fear, scared, frightened, helpless, anxious, worried, nervous, uneasy, fearful, terrified, panic, terror, horror, mortified, horrified

    Cognitions are thoughts that people have. They are appraisals, interpretations, opinions, predictions, and beliefs. They describe how a patient things about themselves, other people, the world, the future. Cognitions are often expressed as "I feel" without an emotion word. For example: "I feel like he does not like me", "I feel broken", "I feel like, if I do that, something really bad will happen to me" are all examples of cognitions. On the contrary, "I feel sad", "I feel hurt", and "I am annoyed" are examples of emotions, not cognitions. Examples of cognitions: believing that the voices have power over you, believing that you will be able to accomplish a goal, perceiving someone as dangerous.

    A stimulus is an external event that is perceived by a person using their sense of sight, touch, smell, taste, or hearing.  Stimuli can be defined singularly (e.g., he smelled a fire) or can be more complex as an event (e.g., she witnessed a car accident where someone died). Examples are: seeing someone with a gun, touching a cold ice cube, smelling somebody's body odor, tasting something sweet, hearing a loud noise. For the purpose of this classification, only stimuli external to the person can be classified as a stimulus.

    A physiological sensation is an internal interoceptive experience produced by activity in the body's physiological systems (e.g., cardiovascular system, digestive system, nervous system). Examples of physiological sensations include feeling a fast heartbeat, sweating, dry throat, tightness in chest, lightheadedness, shortness of breath, nausea, physical discomfort.

    You should give each symptom a type and output the type list. The explanation is enclosed within <exp>...</exp>. The type list is enclosed within <type>[Behavior, Emotion, Physiological Sensation, Cognition,...]</type>. 
    """
        output = get_output(prompt, max_completion_tokens=_MAX_TOKENS)
        type_text = extract_tag_content(output, "type")
        type_list = re.findall(r"Behavior|Emotion|Cognition|Stimulus|Physiological Sensation", type_text or "")
        print("Type:", type_list)
        if len(type_list) == len(symptoms):
            return dict(zip(symptoms, type_list))
        print(f"[label attempt {attempt}] {len(type_list)} labels for {len(symptoms)} symptoms")
    raise ValueError(f"no label list matching {len(symptoms)} symptoms after {max_retry} attempts")


def graph_label_match(graph, node_dict) -> list:
    """Attach (parent type, child type) tags; drop edges whose endpoints are unlabeled."""
    graph_labeled = []
    for item in graph:
        try:
            graph_labeled.append({
                "value": item,
                "tag": {"parent": node_dict[item["from"]], "child": node_dict[item["to"]]},
            })
        except (KeyError, TypeError):
            continue
    return graph_labeled


def get_mask(symptoms, graph, percentage: float, mask_pattern: dict):
    pattern = get_pattern(symptoms, mask_pattern.keys())
    possible_masks = mask_pattern[pattern]
    node_dict = node_labeler(symptoms)
    graph_labeled = graph_label_match(graph, node_dict)

    random.shuffle(graph_labeled)
    mask_len = int(len(graph_labeled) * percentage)
    new_graph, mask = [], []
    for edge in graph_labeled:
        if edge["tag"] in possible_masks and len(mask) < mask_len:
            mask.append(edge)
        else:
            new_graph.append(edge)
    return mask, new_graph


def process_graphs(input_file: Path, output_file: Path, err_file: Path, mask_percentage: float,
                   patterns_file: Path = DEFAULT_PATTERNS) -> None:
    mask_pattern = read_mask_pattern(patterns_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(input_file) as reader, open(output_file, "a") as f, open(err_file, "a") as f_err:
        for line in reader:
            if not line.strip():
                continue
            obj = json.loads(line)
            # "graph" is what the Observer's predict_network writes; the paper's
            # network files spell it "grah".
            symptoms, graph = obj["symptoms"], obj.get("graph", obj.get("grah"))
            if len(graph) == 0:
                f_err.write(json.dumps(obj) + "\n")
                continue
            try:
                mask, new_graph = get_mask(symptoms, graph, mask_percentage, mask_pattern)
            except Exception as e:  # unknown pattern, label/symptom count mismatch, API failure
                # Keep the network unmasked. new_graph then holds raw {"from","to"}
                # edges without "value"/"tag"; the rollout scripts skip such rows.
                print("[mask failed, keeping graph unmasked]", type(e).__name__, e)
                mask, new_graph = [], graph
            obj["mask"] = mask
            obj["new_graph"] = new_graph
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DATA_DIR / "actor" / "network_models.jsonl"),
                    help="JSONL with 'symptoms' (list) and 'graph' (list of {from,to}) per case: the output of "
                         "model_training.observer.predict_network")
    ap.add_argument("--output", required=True, help="output JSONL (appended)")
    ap.add_argument("--err-output", default=None, help="rows with empty graphs (default: <output>.err.jsonl)")
    ap.add_argument("--mask-pct", type=float, default=0.4, help="max fraction of edges to mask")
    ap.add_argument("--patterns", default=str(DEFAULT_PATTERNS))
    ap.add_argument("--seed", type=int, default=None, help="random seed (original runs were unseeded)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    out = Path(args.output)
    err = Path(args.err_output) if args.err_output else out.with_suffix(".err.jsonl")
    process_graphs(resolve_path(args.input), out, err, args.mask_pct, resolve_path(args.patterns))


if __name__ == "__main__":
    main()
