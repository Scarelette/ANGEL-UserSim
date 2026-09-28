"""Credential handling.

Credentials are read from environment variables only — never from source. A
``.env`` file at the repository root is loaded automatically if
``python-dotenv`` is installed (``.env`` is gitignored; copy ``.env.example``).
"""

from __future__ import annotations

import os
from typing import Optional

from angel_common.paths import REPO_ROOT

_LOADED = False


def load_env() -> None:
    """Load <repo>/.env once, without overriding variables already set."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)


def get_env(name: str, default: Optional[str] = None, *alternates: str) -> Optional[str]:
    """Read ``name`` (or the first set alternate), falling back to ``default``."""
    load_env()
    for key in (name, *alternates):
        value = os.environ.get(key)
        if value:
            return value
    return default


def require_env(name: str, *alternates: str, purpose: str = "") -> str:
    """Read a required variable or fail with a message naming what to set."""
    value = get_env(name, None, *alternates)
    if value:
        return value
    names = " or ".join((name, *alternates))
    why = f" (needed for {purpose})" if purpose else ""
    raise RuntimeError(
        f"Missing environment variable {names}{why}. "
        "Set it in your shell or in .env (see .env.example)."
    )
