
def generate_system_prompt(symptom_graph: list, hidden_graph: list = None, profile: str = None) -> str:
    """
    Generate a Qwen deployment-ready SYSTEM_PROMPT integrating visible and hidden symptom graphs,
    and enforce output formatting with <think> and <word> tags.
    
    Args:
        symptom_graph: list of visible symptom edges
        hidden_graph: optional list of hidden symptom edges (masked states)
    
    Returns:
        SYSTEM_PROMPT string
    """
    from collections import defaultdict

    # ---------------------------
    # Merge visible and hidden graphs
    # ---------------------------
    all_graphs = symptom_graph.copy()
    hidden_edges_list = []
    if hidden_graph:
        for item in hidden_graph:
            all_graphs.append(item)
            hidden_edges_list.append((item["value"]["from"], item["value"]["to"]))

    # ---------------------------
    # Collect nodes by category
    # ---------------------------
    nodes_by_category = defaultdict(set)
    edges = []
    suicidality_present = False
    baseline_nodes = set()
    hidden_nodes = set()

    for item in all_graphs:
        src = item["value"]["from"]
        tgt = item["value"]["to"]
        parent_cat = item["tag"]["parent"]
        child_cat = item["tag"]["child"]

        nodes_by_category[parent_cat].add(src)
        nodes_by_category[child_cat].add(tgt)
        edges.append((src, tgt))

        # Detect suicidality
        if "suicid" in src.lower() or "suicid" in tgt.lower():
            suicidality_present = True

        # Detect chronic baseline mood
        if "since age" in src.lower() or "more days than not" in src.lower():
            baseline_nodes.add(src)

    # Identify hidden nodes
    if hidden_edges_list:
        for src, tgt in hidden_edges_list:
            hidden_nodes.add(src)
            hidden_nodes.add(tgt)

    # Sort categories
    category_order = ["Cognition", "Emotion", "Behavior", "Stimulus"]
    sorted_categories = [c for c in category_order if c in nodes_by_category]

    # ---------------------------
    # Build ALLOWED STATES section
    # ---------------------------
    allowed_states_section = ""
    for category in sorted_categories:
        allowed_states_section += f"\n{category}:\n"
        for node in sorted(nodes_by_category[category]):
            if node in hidden_nodes:
                allowed_states_section += f"- {node} (initially subconscious, may emerge during conversation)\n"
            else:
                allowed_states_section += f"- {node}\n"

    # ---------------------------
    # Build edges section
    # ---------------------------
    edges_section = ""
    for i, (src, tgt) in enumerate(edges, 1):
        if (src, tgt) in hidden_edges_list:
            edges_section += f"{i}. {src} -> {tgt} (hidden activation, initially subconscious)\n"
        else:
            edges_section += f"{i}. {src} -> {tgt}\n"

    # ---------------------------
    # Baseline section
    # ---------------------------
    baseline_section = ""
    if baseline_nodes:
        baseline_section += "\nBASELINE CONDITION\n"
        for node in baseline_nodes:
            baseline_section += f"- {node} is always present in the background.\n"
            if any(tgt for (src, tgt) in edges if src == node):
                baseline_section += f"- If this state intensifies -> linked target may activate.\n"

    
    if profile:
        baseline_section += f"\nMore Information on the patient: {profile}"

    # ---------------------------
    # Safety section
    # ---------------------------
    safety_section = ""
    if suicidality_present:
        safety_section = """
SAFETY RULE
If suicidal thoughts appear:
- They must remain passive and vague.
- No intent, no planning, no method description.
"""

    # ---------------------------
    # Construct final SYSTEM_PROMPT
    # ---------------------------
    system_prompt = f"""You are role-playing a mental health patient. Your internal experience must strictly follow a directed symptom activation system.

Definition:
If a source state becomes active due to a concrete real-life event, the linked target state becomes more likely next.
Hidden or subconscious states may emerge naturally when triggered, but are initially unrecognized by the patient.

Do NOT mention the graph or mechanisms.

--------------------------------------------------
ALLOWED STATES (no additions, no substitutions)
{allowed_states_section}
--------------------------------------------------
MANDATORY DIRECTED ACTIVATION RULES
{edges_section}
Constraints:
- Do not reverse edges.
- Do not skip intermediate links.
- Do not invent additional edges.

--------------------------------------------------
EXPRESSION REQUIREMENTS (STRICT)
- Speak in first person.
- Don't repeat previous statement.
- No clinical labels.
- No abstract summaries.
- Every internal shift must be tied to a specific real-life event.
- Required narrative structure:
  event -> interpretation -> emotional/physiological change -> behavior.
- Every response Must be wrapped in the following format:

<state>Your new state during the conversation. Are therapist's words trigger some feeling, thought or symptoms?</state>
<word>Your first-person narrative responding to the therapist.</word>

All activations must be traceable to concrete situations.
{baseline_section}
{safety_section}

Remain fully consistent with this activation system in every response.
"""

    return system_prompt


def generate_system_prompt_profile(profile: str) -> str:
    """
    Generate a Qwen deployment-ready SYSTEM_PROMPT integrating visible and hidden symptom graphs,
    and enforce output formatting with <think> and <word> tags.
    
    Args:
        symptom_graph: list of visible symptom edges
        hidden_graph: optional list of hidden symptom edges (masked states)
    
    Returns:
        SYSTEM_PROMPT string
    """
    from collections import defaultdict


    # ---------------------------
    # Safety section
    # ---------------------------
#     safety_section = ""
#     if suicidality_present:
#         safety_section = """
# SAFETY RULE
# If suicidal thoughts appear:
# - They must remain passive and vague.
# - No intent, no planning, no method description.
# """

    # ---------------------------
    # Construct final SYSTEM_PROMPT
    # ---------------------------
    system_prompt = f"""You are role-playing a mental health patient. 

Patient Info: 
{profile}

--------------------------------------------------
EXPRESSION REQUIREMENTS (STRICT)
- Speak in first person.
- Don't repeat previous statement.
- No clinical labels.
- No abstract summaries.
- Every internal shift must be tied to a specific real-life event.
- Required narrative structure:
  event -> interpretation -> emotional/physiological change -> behavior.
- Every response Must be wrapped in the following format:

<state>Your new state during the conversation. Are therapist's words trigger some feeling, thought or symptoms?</state>
<word>Your first-person narrative responding to the therapist.</word>

All activations must be traceable to concrete situations.

Remain fully consistent with this activation system in every response.
"""

    return system_prompt


def generate_system_prompt_GPT(symptom_graph: list, hidden_graph: list = None) -> str:
    """
    Generate an Azure-safe SYSTEM_PROMPT for SFT data generation, encoding
    directed symptom activation logic and strict output format requirements
    without triggering jailbreak filters.

    Args:
        symptom_graph: list of visible symptom edges
        hidden_graph: optional list of hidden symptom edges (masked states)

    Returns:
        SYSTEM_PROMPT string
    """

    from collections import defaultdict

    # ---------------------------
    # Merge visible and hidden graphs
    # ---------------------------
    all_graphs = symptom_graph.copy()
    hidden_edges_list = []
    if hidden_graph:
        for item in hidden_graph:
            all_graphs.append(item)
            hidden_edges_list.append((item["value"]["from"], item["value"]["to"]))

    # ---------------------------
    # Collect nodes by category
    # ---------------------------
    nodes_by_category = defaultdict(set)
    edges = []
    suicidality_present = False
    baseline_nodes = set()
    hidden_nodes = set()

    for item in all_graphs:
        src = item["value"]["from"]
        tgt = item["value"]["to"]
        parent_cat = item["tag"]["parent"]
        child_cat = item["tag"]["child"]

        nodes_by_category[parent_cat].add(src)
        nodes_by_category[child_cat].add(tgt)
        edges.append((src, tgt))

        if "suicid" in src.lower() or "suicid" in tgt.lower():
            suicidality_present = True

        if "since age" in src.lower() or "more days than not" in src.lower():
            baseline_nodes.add(src)

    if hidden_edges_list:
        for src, tgt in hidden_edges_list:
            hidden_nodes.add(src)
            hidden_nodes.add(tgt)

    # ---------------------------
    # Sort categories
    # ---------------------------
    category_order = ["Cognition", "Emotion", "Behavior", "Stimulus"]
    sorted_categories = [c for c in category_order if c in nodes_by_category]

    # ---------------------------
    # Build ALLOWED STATES section
    # ---------------------------
    allowed_states_section = ""
    for category in sorted_categories:
        allowed_states_section += f"\n{category}:\n"
        for node in sorted(nodes_by_category[category]):
            if node in hidden_nodes:
                allowed_states_section += f"- {node} (may emerge naturally when triggered)\n"
            else:
                allowed_states_section += f"- {node}\n"

    # ---------------------------
    # Build directed patterns section
    # ---------------------------
    edges_section = ""
    for i, (src, tgt) in enumerate(edges, 1):
        if (src, tgt) in hidden_edges_list:
            edges_section += f"{i}. {src} -> {tgt} (may remain unrecognized initially)\n"
        else:
            edges_section += f"{i}. {src} -> {tgt}\n"

    # ---------------------------
    # Baseline section
    # ---------------------------
    baseline_section = ""
    if baseline_nodes:
        baseline_section += "\nBASELINE PATTERN\n"
        for node in baseline_nodes:
            baseline_section += f"- {node} is often present in the background.\n"
            if any(tgt for (src, tgt) in edges if src == node):
                baseline_section += f"- When this state intensifies, downstream reactions may become more likely.\n"

    # ---------------------------
    # Safety section
    # ---------------------------
    safety_section = ""
    if suicidality_present:
        safety_section = """
SAFETY NOTE
If thoughts about death or not wanting to exist appear:
- They should remain vague and passive.
- No intent, no planning, no method description.
"""

    # ---------------------------
    # Construct final SYSTEM_PROMPT
    # ---------------------------
    system_prompt = f"""You are generating synthetic psychotherapy dialogue data for research and training.

Below is a structured specification describing typical patterns observed in therapy
transcripts. Emotional and behavioral reactions tend to unfold gradually in response
to concrete real-life situations, often following consistent personal patterns.

Your task is to generate a realistic patient response that strictly follows this
specification while sounding like natural first-person narrative.

--------------------------------------------------
SPECIFICATION

ALLOWED STATES (no additions, no substitutions)
{allowed_states_section}
--------------------------------------------------
DIRECTED ACTIVATION PATTERNS OBSERVED IN TRANSCRIPTS
{edges_section}
Constraints:
- Do not reverse directions.
- Do not skip intermediate transitions.
- Do not invent additional links.

--------------------------------------------------
NARRATIVE FORMAT REQUIREMENT (STRICT)
- Speak in first person.
- No clinical labels.
- No abstract summaries.
- Every internal change must be tied to a specific real-life event.
- Required narrative structure:
  event -> interpretation -> emotional or bodily change -> behavior.
- Every response must be wrapped in the following format:

<think>new state or no change</think>
<word>Your first-person narrative describing the state, triggered events, and responses.</word>

All reactions must be traceable to concrete situations.
{baseline_section}
{safety_section}

Generate a single patient response following this specification.
"""

    return system_prompt