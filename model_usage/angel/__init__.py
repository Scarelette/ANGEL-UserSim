"""model_usage.angel — run the two-stage Angel simulated-patient model directly.

Stage 1 (Observer) expands a short patient description into a structured long
profile. Stage 2 (Actor) role-plays the patient over a multi-turn conversation.
No HTTP server, no Slurm, no GPU pool: import it and talk to the model.

    from model_usage.angel import AngelModel

    with AngelModel() as model:
        print(model.send("me", "Hi, how are you feeling?", profile_id="0")["reply"])
        print(model.send("me", "Do you want to talk about it?")["reply"])
"""

from .actor import PROMPT_STYLES, Actor, profile_summary
from .backends import Backend, HFBackend, StubBackend, build_backend
from .config import ActorConfig, ObserverConfig, RunnerConfig
from .observer import ExpansionResult, Observer
from . import demo_prompt, postprocess
from .pipeline import AngelModel, Session, profile_fingerprint
from .schema_adapter import adapt_observer_profile, build_minimal_rich_profile

__version__ = "0.1.0"

__all__ = [
    "Actor",
    "PROMPT_STYLES",
    "ActorConfig",
    "AngelModel",
    "Backend",
    "ExpansionResult",
    "HFBackend",
    "Observer",
    "ObserverConfig",
    "RunnerConfig",
    "Session",
    "StubBackend",
    "adapt_observer_profile",
    "build_backend",
    "build_minimal_rich_profile",
    "demo_prompt",
    "postprocess",
    "profile_fingerprint",
    "profile_summary",
]
