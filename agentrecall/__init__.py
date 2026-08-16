"""agentrecall — agent memory in a single SQLite file.

No vector DB, no server, no cloud. Keyword recall works out of the box on stdlib alone;
install ``agentrecall-db[semantic]`` for torch-free hybrid semantic search.
"""

from __future__ import annotations

from .duration import parse_duration
from .embeddings import Embedder, Model2VecEmbedder, get_default_embedder
from .errors import (
    AgentRecallError,
    EmbeddingsUnavailable,
    MemoryNotFound,
    StoreError,
)
from .memory import Memory
from .models import MemoryHit, MemoryRecord

__version__ = "0.2.0"

__all__ = [
    "Memory",
    "MemoryRecord",
    "MemoryHit",
    "Embedder",
    "Model2VecEmbedder",
    "get_default_embedder",
    "parse_duration",
    "AgentRecallError",
    "MemoryNotFound",
    "EmbeddingsUnavailable",
    "StoreError",
    "__version__",
]
