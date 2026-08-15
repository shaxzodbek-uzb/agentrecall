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
from datetime import datetime, timedelta, timezone

from .duration import parse_duration
from .embeddings import Embedder, get_default_embedder
from .errors import EmbeddingsUnavailable, MemoryNotFound
from .models import MemoryHit, MemoryRecord
from .search import fuse_and_rank, sanitize_fts_query
from .store import UNSET, SQLiteStore


def _sqlite_vec_available() -> bool:
    return importlib.util.find_spec("sqlite_vec") is not None


def _resolve_expiry(
    ttl: str | int | float | timedelta | None,
    expires_at: datetime | None,
) -> datetime | None:
    """Turn the ``ttl`` / ``expires_at`` pair into a single absolute deadline.

    The two spellings are mutually exclusive: accepting both would leave the caller
    guessing which one won.
    """
    if ttl is not None and expires_at is not None:
        raise ValueError("pass either ttl or expires_at, not both")
    if ttl is not None:
        return datetime.now(timezone.utc) + parse_duration(ttl)
    return expires_at


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
                    "Install it with:  pip install 'agentrecall-db[semantic]'"
                )
            if not _sqlite_vec_available():
                raise EmbeddingsUnavailable(
                    "embeddings=True but sqlite-vec is not installed. "
                    "Install it with:  pip install 'agentrecall-db[semantic]'"
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
        ttl: str | int | float | timedelta | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryRecord:
        """Store a memory.

        Pass ``ttl`` (``"30d"``, ``"12h"``, ``3600`` seconds, or a ``timedelta``) or an
        absolute ``expires_at`` to make the memory **temporary**: once the deadline
        passes it stops showing up in :meth:`search`, :meth:`all`, :meth:`get` and
        :meth:`count`. Without either, the memory is permanent — the default.
        """
        embedding = self._embed(content) if self._semantic else None
        return self._store.add(
            content=content,
            namespace=namespace if namespace is not None else self.namespace,
            tags=list(tags) if tags else [],
            metadata=dict(metadata) if metadata else {},
            importance=importance,
            embedding=embedding,
            expires_at=_resolve_expiry(ttl, expires_at),
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
                expires_at=_resolve_expiry(d.get("ttl"), d.get("expires_at")),
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
        include_expired: bool = False,
    ) -> list[MemoryHit]:
        if k <= 0:
            return []
        ns = namespace if namespace is not None else self.namespace
        pool = max(k * 4, 20)

        fts_query = sanitize_fts_query(query)
        fts_hits = (
            self._store.fts_search(
                fts_query, namespace=ns, tags=tags, limit=pool, include_expired=include_expired
            )
            if fts_query
            else []
        )
        vec_hits: list[tuple[int, float]] = []
        if self._semantic:
            vec_hits = self._store.vec_search(
                self._embed(query),
                namespace=ns,
                tags=tags,
                limit=pool,
                include_expired=include_expired,
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

    def get(self, memory_id: int, *, include_expired: bool = False) -> MemoryRecord:
        record = self._store.get(memory_id, include_expired=include_expired)
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
        ttl: str | int | float | timedelta | None = None,
        expires_at: datetime | None | object = UNSET,
    ) -> MemoryRecord:
        """Update a memory in place; omitted fields keep their current value.

        ``ttl`` restarts the countdown from now. Pass ``expires_at=None`` explicitly to
        drop an existing deadline and make the memory permanent again. Works on an
        already-expired memory, which is how you revive one.
        """
        if ttl is not None:
            if expires_at is not UNSET:
                raise ValueError("pass either ttl or expires_at, not both")
            expires_at = _resolve_expiry(ttl, None)
        embedding = self._embed(content) if (self._semantic and content is not None) else None
        return self._store.update(
            memory_id,
            content=content,
            tags=tags,
            metadata=metadata,
            importance=importance,
            embedding=embedding,
            expires_at=expires_at,
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
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        return self._store.all(
            namespace=namespace if namespace is not None else self.namespace,
            tags=tags,
            limit=limit,
            offset=offset,
            include_expired=include_expired,
        )

    def count(self, *, namespace: str | None = None, include_expired: bool = False) -> int:
        return self._store.count(
            namespace=namespace if namespace is not None else self.namespace,
            include_expired=include_expired,
        )

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

    def purge_expired(self, *, namespace: str | None = None) -> int:
        """Delete memories whose TTL has elapsed; returns how many were removed.

        Purely a housekeeping call — expired memories are already invisible to reads, so
        running it (or never running it) can't change what an agent recalls.
        """
        return self._store.purge_expired(
            namespace=namespace if namespace is not None else self.namespace
        )

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["Memory"]
