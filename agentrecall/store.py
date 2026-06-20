"""The SQLite storage layer.

One ordinary SQLite file holds everything: the ``memories`` table, an FTS5 keyword index
kept in sync by triggers, and — only when semantic mode is active — a ``sqlite-vec``
vector index. No external services, no second process.

This module is pure storage: it knows nothing about embedders or ranking policy. The
:class:`~agentrecall.memory.Memory` facade composes it with those.
"""

from __future__ import annotations

import json
import os
import sqlite3
import warnings
from datetime import datetime, timezone

from .errors import EmbeddingsUnavailable, MemoryNotFound, StoreError
from .models import MemoryRecord

SCHEMA_VERSION = 1


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class SQLiteStore:
    """Low-level CRUD + search primitives over a single SQLite database file."""

    def __init__(self, path: str | os.PathLike, *, embedding_dim: int | None = None) -> None:
        self.path = str(path)
        self._want_vectors = embedding_dim is not None
        self._declared_dim = embedding_dim
        try:
            self.conn = sqlite3.connect(self.path)
        except sqlite3.Error as exc:  # pragma: no cover - filesystem-level failure
            raise StoreError(f"could not open database {self.path!r}: {exc}") from exc
        self.conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self.fts_enabled = self._detect_fts5()
        self._vec_loaded = False
        self._ensure_schema()
        # If this DB already has a vector index (a reopened semantic store), load the
        # sqlite-vec extension up front so the very first search() can't crash with
        # "no such module: vec0" before any write has triggered a lazy load.
        if self.has_vectors():
            self._load_vec_extension()

    # -- setup ----------------------------------------------------------------

    def _apply_pragmas(self) -> None:
        for pragma in ("PRAGMA journal_mode=WAL", "PRAGMA synchronous=NORMAL"):
            try:
                self.conn.execute(pragma)
            except sqlite3.Error:  # pragma: no cover - e.g. :memory: rejects WAL
                pass

    def _detect_fts5(self) -> bool:
        try:
            self.conn.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(x)")
            self.conn.execute("DROP TABLE temp.__fts5_probe")
            return True
        except sqlite3.Error:  # pragma: no cover - only on FTS5-less SQLite builds
            warnings.warn(
                "SQLite was built without FTS5; agentrecall falls back to slower "
                "LIKE-based keyword search.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False

    def _ensure_schema(self) -> None:
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace        TEXT    NOT NULL DEFAULT 'default',
                content          TEXT    NOT NULL,
                tags             TEXT    NOT NULL DEFAULT '[]',
                metadata         TEXT    NOT NULL DEFAULT '{}',
                importance       REAL    NOT NULL DEFAULT 1.0,
                created_at       TEXT    NOT NULL,
                updated_at       TEXT    NOT NULL,
                last_accessed_at TEXT,
                access_count     INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_memories_created   ON memories(created_at);
            """
        )
        if self.fts_enabled:
            c.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, tags,
                    content='memories', content_rowid='id',
                    tokenize='porter unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.id, new.content, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.id, old.content, old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.id, old.content, old.tags);
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.id, new.content, new.tags);
                END;
                """
            )
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    # -- vector index (lazy, only when embeddings are written) ----------------

    def _ensure_vec(self, dim: int) -> None:
        if self.has_vectors():
            return
        if not self._vec_loaded:
            try:
                import sqlite_vec

                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.enable_load_extension(False)
            except Exception as exc:  # ImportError, or extension load rejected
                raise EmbeddingsUnavailable(
                    "Semantic search needs the optional 'semantic' extra (sqlite-vec). "
                    "Install it with:  pip install 'agentrecall-db[semantic]'"
                ) from exc
            self._vec_loaded = True
        # No commit here on purpose: the CREATE participates in the surrounding
        # add()/update() transaction, so a failure writing the vector rolls back the
        # content row too (atomic first-embedding write).
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0("
            f"memory_id INTEGER PRIMARY KEY, embedding float[{dim}])"
        )

    def _load_vec_extension(self) -> bool:
        if self._vec_loaded:
            return True
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self._vec_loaded = True
            return True
        except Exception:
            return False

    def has_vectors(self) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_vec'"
        ).fetchone()
        return row is not None

    def _write_embedding(self, memory_id: int, embedding: list[float]) -> None:
        self._ensure_vec(len(embedding))
        payload = json.dumps([float(x) for x in embedding])
        self.conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
        self.conn.execute(
            "INSERT INTO memories_vec(memory_id, embedding) VALUES (?, ?)",
            (memory_id, payload),
        )

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            namespace=row["namespace"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            importance=row["importance"],
            created_at=_parse_dt(row["created_at"]),  # type: ignore[arg-type]
            updated_at=_parse_dt(row["updated_at"]),  # type: ignore[arg-type]
            last_accessed_at=_parse_dt(row["last_accessed_at"]),
            access_count=row["access_count"],
        )

    @staticmethod
    def _tag_clause(tags: list[str] | None, alias: str = "memories") -> tuple[str, list]:
        """Return an ``AND ...`` fragment matching rows that contain *all* given tags."""
        if not tags:
            return "", []
        parts = [f"EXISTS (SELECT 1 FROM json_each({alias}.tags) WHERE value = ?)" for _ in tags]
        return " AND " + " AND ".join(parts), list(tags)

    # -- CRUD -----------------------------------------------------------------

    def add(
        self,
        *,
        content: str,
        namespace: str,
        tags: list[str],
        metadata: dict,
        importance: float,
        embedding: list[float] | None = None,
    ) -> MemoryRecord:
        now = _utcnow().isoformat()
        cur = self.conn.execute(
            "INSERT INTO memories(namespace, content, tags, metadata, importance, "
            "created_at, updated_at, access_count) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (namespace, content, json.dumps(tags), json.dumps(metadata), importance, now, now),
        )
        memory_id = int(cur.lastrowid)
        if embedding is not None:
            self._write_embedding(memory_id, embedding)
        self.conn.commit()
        record = self.get(memory_id)
        assert record is not None
        return record

    def get(self, memory_id: int) -> MemoryRecord | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def update(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        importance: float | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryRecord:
        if self.get(memory_id) is None:
            raise MemoryNotFound(memory_id)
        sets: list[str] = []
        params: list = []
        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags))
        if metadata is not None:
            sets.append("metadata = ?")
            params.append(json.dumps(metadata))
        if importance is not None:
            sets.append("importance = ?")
            params.append(importance)
        sets.append("updated_at = ?")
        params.append(_utcnow().isoformat())
        params.append(memory_id)
        self.conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params)
        if embedding is not None:
            self._write_embedding(memory_id, embedding)
        self.conn.commit()
        record = self.get(memory_id)
        assert record is not None
        return record

    def delete(self, memory_id: int) -> bool:
        if self.has_vectors():
            self.conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def touch(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        now = _utcnow().isoformat()
        self.conn.executemany(
            "UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 "
            "WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )
        self.conn.commit()

    def all(
        self,
        *,
        namespace: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list = []
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        tag_sql, tag_params = self._tag_clause(tags)
        sql += tag_sql
        params.extend(tag_params)
        sql += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self, *, namespace: str | None = None) -> int:
        if namespace is None:
            row = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE namespace = ?", (namespace,)
            ).fetchone()
        return int(row[0])

    # -- search primitives ----------------------------------------------------

    def fts_search(
        self,
        query: str,
        *,
        namespace: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[tuple[int, float]]:
        if not query:
            return []
        if not self.fts_enabled:
            return self._like_search(query, namespace=namespace, tags=tags, limit=limit)
        sql = (
            "SELECT memories_fts.rowid AS id, bm25(memories_fts) AS rank "
            "FROM memories_fts JOIN memories ON memories.id = memories_fts.rowid "
            "WHERE memories_fts MATCH ?"
        )
        params: list = [query]
        if namespace is not None:
            sql += " AND memories.namespace = ?"
            params.append(namespace)
        tag_sql, tag_params = self._tag_clause(tags)
        sql += tag_sql
        params.extend(tag_params)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Defensive: a malformed MATCH slipped through — degrade to LIKE.
            return self._like_search(query, namespace=namespace, tags=tags, limit=limit)
        # bm25 returns lower = better; expose higher = better.
        return [(r["id"], -float(r["rank"])) for r in rows]

    def _like_search(
        self,
        query: str,
        *,
        namespace: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[tuple[int, float]]:
        import re

        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []
        sql = "SELECT id, content, tags FROM memories WHERE 1=1"
        params: list = []
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        tag_sql, tag_params = self._tag_clause(tags)
        sql += tag_sql
        params.extend(tag_params)
        scored: list[tuple[int, float]] = []
        for row in self.conn.execute(sql, params):
            # Mirror the FTS index, which covers both content and tags.
            text = (row["content"] or "").lower() + " " + (row["tags"] or "").lower()
            hits = sum(1 for tok in tokens if tok in text)
            if hits:
                scored.append((row["id"], float(hits)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def vec_search(
        self,
        embedding: list[float],
        *,
        namespace: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[tuple[int, float]]:
        # Load the extension whenever it isn't loaded yet — even if the vec table
        # already exists (reopened DB) — otherwise the MATCH below hits an unloaded
        # vec0 module and raises "no such module: vec0".
        if not self._load_vec_extension() or not self.has_vectors():
            return []
        pool = max(limit * 4, 50)
        payload = json.dumps([float(x) for x in embedding])
        sql = (
            "WITH knn AS ("
            "  SELECT memory_id, distance FROM memories_vec "
            "  WHERE embedding MATCH ? AND k = ? ORDER BY distance"
            ") "
            "SELECT knn.memory_id AS id, knn.distance AS distance "
            "FROM knn JOIN memories ON memories.id = knn.memory_id WHERE 1=1"
        )
        params: list = [payload, pool]
        if namespace is not None:
            sql += " AND memories.namespace = ?"
            params.append(namespace)
        tag_sql, tag_params = self._tag_clause(tags)
        sql += tag_sql
        params.extend(tag_params)
        sql += " ORDER BY knn.distance LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        # cosine distance in [0, 2]; expose cosine similarity (higher = better).
        return [(r["id"], 1.0 - float(r["distance"])) for r in rows]

    # -- maintenance ----------------------------------------------------------

    def purge(
        self,
        *,
        before: datetime | None = None,
        namespace: str | None = None,
        keep_last: int | None = None,
    ) -> int:
        """Delete memories and return how many were removed.

        Passing both ``before`` and ``keep_last`` removes the **union**: rows older than
        ``before`` *or* beyond the newest ``keep_last`` per namespace. With neither
        criterion this is a no-op (it refuses to wipe everything by accident).
        """
        if before is None and keep_last is None:
            return 0  # refuse to wipe everything by accident
        ids: set[int] = set()
        if before is not None:
            # Stored timestamps are UTC ISO strings compared lexically; normalize the
            # caller's cutoff to UTC so a non-UTC (or naive) `before` can't delete rows
            # that are actually newer than the intended instant.
            if before.tzinfo is None:
                before = before.replace(tzinfo=timezone.utc)
            before = before.astimezone(timezone.utc)
            sql = "SELECT id FROM memories WHERE created_at < ?"
            params: list = [before.isoformat()]
            if namespace is not None:
                sql += " AND namespace = ?"
                params.append(namespace)
            ids.update(r[0] for r in self.conn.execute(sql, params))
        if keep_last is not None:
            sql = (
                "SELECT id FROM (SELECT id, ROW_NUMBER() OVER "
                "(PARTITION BY namespace ORDER BY created_at DESC, id DESC) AS rn "
                "FROM memories"
            )
            params = []
            if namespace is not None:
                sql += " WHERE namespace = ?"
                params.append(namespace)
            sql += ") WHERE rn > ?"
            params.append(keep_last)
            ids.update(r[0] for r in self.conn.execute(sql, params))
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        id_list = list(ids)
        if self.has_vectors():
            self.conn.execute(
                f"DELETE FROM memories_vec WHERE memory_id IN ({placeholders})", id_list
            )
        cur = self.conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", id_list)
        self.conn.commit()
        return cur.rowcount

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass


__all__ = ["SQLiteStore", "SCHEMA_VERSION"]
