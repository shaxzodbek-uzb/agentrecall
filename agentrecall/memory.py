"""The public facade: :class:`Memory`.

Composes the SQLite store, an optional embedder, and the ranking helpers into the small,
sync API that callers use::

    from agentrecall import Memory

    with Memory("agent.db") as mem:
        mem.add("The user prefers dark mode", tags=["preference"])
        for hit in mem.search("ui settings the user likes", k=3):
            print(hit.score, hit.content)
"""

from __future__ import annotations

import importlib.util
import os
from datetime import datetime, timezone

from .embeddings import Embedder, get_default_embedder
from .errors import EmbeddingsUnavailable, MemoryNotFound
from .models import MemoryHit, MemoryRecord
from .search import fuse_and_rank, sanitize_fts_query
from .store import SQLiteStore


def _sqlite_vec_available() -> bool:
    return importlib.util.find_spec("sqlite_vec") is not None


class Memory:
    """Persistent agent memory backed by a single SQLite file."""

    def __init__(
        self,
        path: str | os.PathLike = "agentrecall.db",
        *,
        namespace: str = "default",
        embeddings: bool | str = "auto",
        embedder: Embedder | None = None,
        recency_weight: float = 0.0,
        importance_weight: float = 0.0,
    ) -> None:
        self.namespace = namespace
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self._embedder, self._semantic = self._resolve_embeddings(embeddings, embedder)
        self._store = SQLiteStore(path, embedding_dim=None)

    @staticmethod
    def _resolve_embeddings(
        embeddings: bool | str, embedder: Embedder | None
    ) -> tuple[Embedder | None, bool]:
        if embeddings is False:
            return None, False

        if embeddings is True:
            chosen = embedder or get_default_embedder()
            if chosen is None:
                raise EmbeddingsUnavailable(
                    "embeddings=True but model2vec is not installed. "
                    "Install it with:  pip install 'agentrecall[semantic]'"
                )
            if not _sqlite_vec_available():
                raise EmbeddingsUnavailable(
                    "embeddings=True but sqlite-vec is not installed. "
                    "Install it with:  pip install 'agentrecall[semantic]'"
                )
            return chosen, True

        # "auto" (the default): enable semantic only if everything is present.
        if embeddings != "auto":
            raise ValueError(f"embeddings must be True, False, or 'auto', got {embeddings!r}")
        chosen = embedder or get_default_embedder()
        if chosen is not None and _sqlite_vec_available():
            return chosen, True
        return None, False

    @property
    def semantic_enabled(self) -> bool:
        return self._semantic

    def _embed(self, text: str) -> list[float]:
        assert self._embedder is not None
        return self._embedder.embed([text])[0]

    # -- writes ---------------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        importance: float = 1.0,
        namespace: str | None = None,
    ) -> MemoryRecord:
        embedding = self._embed(content) if self._semantic else None
        return self._store.add(
            content=content,
            namespace=namespace if namespace is not None else self.namespace,
            tags=list(tags) if tags else [],
            metadata=dict(metadata) if metadata else {},
            importance=importance,
            embedding=embedding,
        )

    def add_many(self, items: list[str | dict]) -> list[MemoryRecord]:
        norm: list[dict] = [{"content": it} if isinstance(it, str) else dict(it) for it in items]
        for i, d in enumerate(norm):
            if "content" not in d or not isinstance(d["content"], str):
                raise ValueError(
                    f"add_many item {i} must be a str or a dict with a string 'content' "
                    f"key, got {items[i]!r}"
                )
        if self._semantic and norm:
            assert self._embedder is not None
            embeddings = self._embedder.embed([d["content"] for d in norm])
        else:
            embeddings = [None] * len(norm)
        return [
            self._store.add(
                content=d["content"],
                namespace=d.get("namespace") or self.namespace,
                tags=list(d.get("tags") or []),
                metadata=dict(d.get("metadata") or {}),
                importance=float(d.get("importance", 1.0)),
                embedding=emb,
            )
            for d, emb in zip(norm, embeddings, strict=True)
        ]

    # -- reads / search -------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        namespace: str | None = None,
        tags: list[str] | None = None,
        recency_weight: float | None = None,
        importance_weight: float | None = None,
    ) -> list[MemoryHit]:
        if k <= 0:
            return []
        ns = namespace if namespace is not None else self.namespace
        pool = max(k * 4, 20)

        fts_query = sanitize_fts_query(query)
        fts_hits = (
            self._store.fts_search(fts_query, namespace=ns, tags=tags, limit=pool)
            if fts_query
            else []
        )
        vec_hits: list[tuple[int, float]] = []
        if self._semantic:
            vec_hits = self._store.vec_search(
                self._embed(query), namespace=ns, tags=tags, limit=pool
            )
        if not fts_hits and not vec_hits:
            return []

        ids = {mid for mid, _ in fts_hits} | {mid for mid, _ in vec_hits}
        records: dict[int, MemoryRecord] = {}
        for mid in ids:
            record = self._store.get(mid)
            if record is not None:
                records[mid] = record

        hits = fuse_and_rank(
            fts_hits,
            vec_hits,
            records,
            recency_weight=recency_weight if recency_weight is not None else self.recency_weight,
            importance_weight=importance_weight
            if importance_weight is not None
            else self.importance_weight,
            now=datetime.now(timezone.utc),
            limit=k,
        )
        self._store.touch([h.id for h in hits])
        return hits

    def get(self, memory_id: int) -> MemoryRecord:
        record = self._store.get(memory_id)
        if record is None:
            raise MemoryNotFound(memory_id)
        return record

    def update(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        importance: float | None = None,
    ) -> MemoryRecord:
        embedding = self._embed(content) if (self._semantic and content is not None) else None
        return self._store.update(
            memory_id,
            content=content,
            tags=tags,
            metadata=metadata,
            importance=importance,
            embedding=embedding,
        )

    def delete(self, memory_id: int) -> bool:
        return self._store.delete(memory_id)

    def all(
        self,
        *,
        namespace: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        return self._store.all(
            namespace=namespace if namespace is not None else self.namespace,
            tags=tags,
            limit=limit,
            offset=offset,
        )

    def count(self, *, namespace: str | None = None) -> int:
        return self._store.count(namespace=namespace if namespace is not None else self.namespace)

    def forget(
        self,
        *,
        before: datetime | None = None,
        namespace: str | None = None,
        keep_last: int | None = None,
    ) -> int:
        """Bulk-delete memories; returns how many were removed.

        ``before`` (naive datetimes are treated as UTC) and ``keep_last`` (newest N per
        namespace) combine as a **union**. With neither argument this is a no-op.
        """
        return self._store.purge(
            before=before,
            namespace=namespace if namespace is not None else self.namespace,
            keep_last=keep_last,
        )

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["Memory"]
