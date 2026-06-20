"""Lightweight settings read from ``AGENTRECALL_*`` environment variables.

Stdlib only — no pydantic, no extra dependency. Used by the CLI and MCP server so the
same database can be configured once via the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .embeddings import DEFAULT_MODEL


@dataclass
class Settings:
    db_path: str = "agentrecall.db"
    namespace: str = "default"
    embeddings: str = "auto"  # "auto" | "true" | "false"
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            db_path=env.get("AGENTRECALL_DB", cls.db_path),
            namespace=env.get("AGENTRECALL_NAMESPACE", cls.namespace),
            embeddings=env.get("AGENTRECALL_EMBEDDINGS", cls.embeddings).lower(),
            model=env.get("AGENTRECALL_MODEL", cls.model),
        )

    def embeddings_value(self) -> bool | str:
        """Map the string setting to the ``Memory(embeddings=...)`` argument."""
        value = self.embeddings.lower()
        if value == "true":
            return True
        if value == "false":
            return False
        return "auto"


__all__ = ["Settings"]
