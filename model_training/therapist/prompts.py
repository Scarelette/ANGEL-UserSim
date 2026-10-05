"""Prompts for the therapist training data, verbatim from the original scripts.

``POLISH_PROMPT`` and ``ROLE_PROMPT`` come from ``pdf_parser/chunks_refine.py``,
``QA_SYSTEM_PROMPT`` / ``QA_PROMPTS`` from ``sft_generation/prompts.py`` and
``FT_SYSTEM_PROMPT`` from ``sft_generation/generate_sft_samples.py``.
"""

POLISH_PROMPT = (
    "Please help me check if the content semantics are complete "
    "and coherent. If not please polish the content to make it "
    "complete and coherent. Note that you should also remove the useless\\n.\n\n"
    "Content:\n{content}"
)

ROLE_PROMPT = (
    "Please classify the following section into different roles: procedure, rationale, adjustment, case and theory "
    "Note that ONLY answer with the role of the content\\n.\n\n"
    "Content:\n{content}"
)

ROLES = ["procedure", "rationale", "adjustment", "case", "theory"]

QA_SYSTEM_PROMPT = """You are a senior clinical supervisor.
Your task is to convert treatment manual content into training examples.
Do NOT quote the original text.
Do NOT add techniques not present in the source.
Preserve logical order and conditions.
Use professional, neutral language.
"""

QA_PROMPTS = {
    "procedure": """Based on the following manual content, generate 3 decision-making training examples.

Each example must include:
- instruction: a concrete clinical decision question (e.g. "If ..., what should be done?")
- response: a stepwise, conditional answer consistent with the manual

Output a JSON array with keys: instruction, response.

CONTENT:
<<<{content}>>>
""",

    "rationale": """Based on the following manual content, generate 3 explanation-style training examples.

Each example must include:
- instruction: a question like "How would you explain ... to a patient?"
- response: a professional, non-blaming explanation consistent with the manual

Output a JSON array with keys: instruction, response.

CONTENT:
<<<{content}>>>
""",

    "adjustment": """Based on the following manual content, generate 2 boundary-condition training examples.

Each example must include:
- instruction: a question about when a common approach may not be appropriate
- response: an answer explaining limitations or necessary adjustments

Output a JSON array with keys: instruction, response.

CONTENT:
<<<{content}>>>
""",

    "case": """Based on the following case material, generate 2 case-reasoning training examples.

Each example must include:
- instruction: a question asking how to reason or decide in this case
- response: a reasoning-based answer grounded in the manual logic

Output a JSON array with keys: instruction, response.

CONTENT:
<<<{content}>>>
""",

    "theory": """Based on the following theoretical content, generate 2 conceptual training examples.

Each example must include:
- instruction: a question asking to explain or apply the concept
- response: a clear explanation consistent with the theory

Output a JSON array with keys: instruction, response.

CONTENT:
<<<{content}>>>
""",
}

# System prompt of every fine-tuning row. The rollouts later talk to the
# fine-tuned model with ``model_training.actor.prompts.THERAPIST_SYSTEM_PROMPT``.
FT_SYSTEM_PROMPT = (
    "You are a licensed mental health therapist (e.g., clinical psychologist or licensed counselor) "
    "with extensive training in evidence-based practice.\n"
    "Your role is to conduct therapy sessions in a professional, ethical, and empathetic manner."
)
