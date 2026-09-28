"""Credential handling: one ``.env`` file at the repository root.

Every key, endpoint and deployment name is read from environment variables,
never from source. Put them in ``<repo>/.env`` (copy ``.env.example``; the file
is gitignored). It is loaded automatically the first time any ``angel_common``
module is imported, so every script in the repository sees the same values.
Variables already set in the shell take precedence over ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

_LOADED = False


def _parse_line(line: str) -> Optional[tuple]:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line[len("export "):]
    key, _, value = line.partition("=")
    key, value = key.strip(), value.strip()
    if value[:1] in ("'", '"') and value[-1:] == value[:1]:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return (key, value) if key else None


def load_env(path: Path = ENV_FILE) -> None:
    """Load ``path`` once into ``os.environ`` without overriding set variables."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed and parsed[1] and parsed[0] not in os.environ:
            os.environ[parsed[0]] = parsed[1]


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
        f"Set it in {ENV_FILE} (copy .env.example) or in your shell."
    )


load_env()
