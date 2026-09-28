"""Shared utilities: repo-relative paths, model resolution, env-based credentials, LLM clients.

Importing any ``angel_common`` module loads ``<repo>/.env`` (see ``env.py``).
"""

from angel_common import env as _env  # noqa: F401  (loads .env on import)
