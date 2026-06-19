"""Minimal .env loader (no external dependency).

`.env` takes precedence over pre-existing process environment variables so the
key used is the one the user explicitly put in `.env` (auditable), not whatever
happened to be exported in the shell.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path = _DEFAULT_ENV_PATH) -> dict[str, str]:
    """Load KEY=VALUE lines from `.env` into os.environ (overriding existing)."""
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value
        loaded[key] = value
    return loaded


def require_openai_key() -> str:
    """Return OPENAI_API_KEY, loading `.env` first. Raise with guidance if absent."""
    loaded = load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in .env (see .env.example): "
            "OPENAI_API_KEY=sk-..."
        )
    source = ".env" if "OPENAI_API_KEY" in loaded else "process environment"
    logger.info("OPENAI_API_KEY loaded from %s (len %d)", source, len(key))
    return key
