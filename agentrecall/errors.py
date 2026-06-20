"""Exception hierarchy for agentrecall.

All errors raised by the library subclass :class:`AgentRecallError`, so callers can
``except AgentRecallError`` to catch anything the library throws.
"""

from __future__ import annotations


class AgentRecallError(Exception):
    """Base class for all agentrecall errors."""


class MemoryNotFound(AgentRecallError):
    """Raised by ``get()`` / ``update()`` / ``delete()`` when the id does not exist."""

    def __init__(self, memory_id: int) -> None:
        self.memory_id = memory_id
        super().__init__(f"No memory with id {memory_id!r}")


class EmbeddingsUnavailable(AgentRecallError):
    """Raised when semantic mode is required (``embeddings=True``) but the optional
    ``[semantic]`` extra (model2vec / sqlite-vec) is not installed or cannot load."""


class StoreError(AgentRecallError):
    """Wraps an unexpected SQLite-level failure."""


__all__ = [
    "AgentRecallError",
    "MemoryNotFound",
    "EmbeddingsUnavailable",
    "StoreError",
]
