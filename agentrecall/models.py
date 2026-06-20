"""Dataclasses surfaced to callers: :class:`MemoryRecord` and :class:`MemoryHit`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class MemoryRecord:
    """A single stored memory, as returned by the public API."""

    id: int
    content: str
    namespace: str
    tags: list[str]
    metadata: dict
    importance: float
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None
    access_count: int

    def to_dict(self) -> dict:
        """JSON-safe dict (datetimes become ISO-8601 strings)."""
        return {
            "id": self.id,
            "content": self.content,
            "namespace": self.namespace,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
            "access_count": self.access_count,
        }


@dataclass(slots=True)
class MemoryHit:
    """A search result: a :class:`MemoryRecord` plus a relevance ``score``
    (higher = more relevant)."""

    record: MemoryRecord
    score: float

    @property
    def id(self) -> int:
        return self.record.id

    @property
    def content(self) -> str:
        return self.record.content

    @property
    def metadata(self) -> dict:
        return self.record.metadata

    @property
    def tags(self) -> list[str]:
        return self.record.tags

    def to_dict(self) -> dict:
        return {"score": self.score, **self.record.to_dict()}


__all__ = ["MemoryRecord", "MemoryHit"]
