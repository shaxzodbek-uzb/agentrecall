# agentrecall — Canonical Build Spec (single source of truth)

> Every builder reads THIS file and implements exactly the signatures, names, and
> behaviors below. Do not invent extra public API. Match names character-for-character.
> When in doubt, prefer fewer moving parts and **stdlib over dependencies**.

## What it is (positioning)

**Agent memory in a single SQLite file.** No vector database, no server, no cloud, no
API key. `pip install agentrecall`, point it at a path, and you have persistent memory
for an LLM agent that you can `cp`, `git diff`, and read from any language.

Most memory layers (mem0, Letta/MemGPT, Zep, Cognee) are **infrastructure**: they want a
vector store, a Postgres, a running server, or a hosted cloud + API key, and many call an
LLM to "extract" facts. `agentrecall` inverts that: it is a **library**, the store is one
ordinary SQLite file, retrieval is deterministic, and **nothing leaves the machine**.

Three guarantees that no mainstream memory layer combines:

1. **Zero infrastructure.** The core has **no third-party dependencies** — it uses only
   Python's stdlib `sqlite3`. Keyword recall (SQLite FTS5, BM25) works out of the box on a
   fresh `pip install` with nothing else installed.
2. **Semantic search with no torch, no GPU, no download server.** Install the optional
   `[semantic]` extra and you get hybrid (keyword + vector) recall powered by
   **model2vec** static embeddings (~10 MB, CPU-only, no PyTorch) stored in **sqlite-vec**
   — still one file, still offline.
3. **Verbatim & deterministic.** `agentrecall` never calls an LLM to mutate your memories.
   What you `add()` is what is stored. No silent fact-extraction, no cloud round-trip, no
   non-determinism. (Competitors call an LLM on every write.)

Ships as a **Python library**, an **MCP server** (`agentrecall serve`, an
embeddings-capable replacement for the official JSONL/keyword-only memory server), and a
**CLI**.

Tagline: *"Agent memory in a single SQLite file — no vector DB, no server, no cloud."*

## Package layout

```
agentrecall/
  __init__.py          # public exports (see "Public API" below) + __version__
  errors.py            # exception types
  models.py            # MemoryRecord, MemoryHit (dataclasses)
  embeddings.py        # Embedder protocol, Model2VecEmbedder, get_default_embedder()
  store.py             # SQLiteStore: schema, migrations, CRUD, FTS5 + sqlite-vec primitives
  search.py            # sanitize_fts_query, reciprocal_rank_fusion, fuse_and_rank
  memory.py            # Memory — the public facade (ties store + embeddings + search)
  config.py            # Settings (env AGENTRECALL_*) — stdlib only, no pydantic
  mcp_server.py        # build_mcp_server(memory) -> MCP server (optional extra 'mcp')
  cli.py               # `agentrecall` entry point (argparse, stdlib)
tests/                 # pytest; the FTS-only path must need NO third-party deps
examples/              # runnable snippets
```

Distribution name `agentrecall`; import package `agentrecall`. Python >=3.10.
License MIT (holder: "Shaxzodbek Sobirov / Blaze"). **Core deps: none.** Everything
beyond stdlib is an optional extra.

## Storage schema (documented & language-agnostic)

A `agentrecall` database is a plain SQLite file. Any language can read it. Schema
(`schema_version = 1`, tracked in `PRAGMA user_version`):

```sql
CREATE TABLE memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace       TEXT    NOT NULL DEFAULT 'default',
    content         TEXT    NOT NULL,
    tags            TEXT    NOT NULL DEFAULT '[]',   -- JSON array of strings
    metadata        TEXT    NOT NULL DEFAULT '{}',   -- JSON object
    importance      REAL    NOT NULL DEFAULT 1.0,
    created_at      TEXT    NOT NULL,                -- ISO-8601 UTC, e.g. 2026-06-20T09:00:00+00:00
    updated_at      TEXT    NOT NULL,
    last_accessed_at TEXT,                           -- nullable
    access_count    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_memories_namespace ON memories(namespace);
CREATE INDEX idx_memories_created   ON memories(created_at);

-- Keyword index (always present). external-content FTS mirrors `memories`.
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, tags,
    content='memories', content_rowid='id',
    tokenize='porter unicode61'
);
-- triggers memories_ai / memories_ad / memories_au keep the FTS index in sync.

-- Vector index (ONLY created when semantic mode is active and sqlite-vec loads):
CREATE VIRTUAL TABLE memories_vec USING vec0(memory_id INTEGER PRIMARY KEY, embedding float[DIM]);
```

Rules:
- All timestamps are timezone-aware **UTC** ISO-8601 strings on disk; surfaced as aware
  `datetime` objects in `MemoryRecord`.
- `tags` and `metadata` are JSON text columns. `tags` is always a `list[str]`; `metadata`
  always a `dict`.
- If FTS5 is unavailable in the runtime's SQLite build (rare), `SQLiteStore` falls back to
  a `LIKE`-based scan and sets `store.fts_enabled = False` (a warning is emitted once).
- The vector table is created lazily the first time an embedding is written, using the
  active embedder's dimension. Opening the same DB later in FTS-only mode must NOT error.

## Core types & exact signatures

### errors.py
```python
class AgentRecallError(Exception):
    """Base class for all agentrecall errors."""
class MemoryNotFound(AgentRecallError):
    """Raised by get()/update()/delete() when the id does not exist."""
    def __init__(self, memory_id: int): ...
class EmbeddingsUnavailable(AgentRecallError):
    """Raised when semantic mode is required (embeddings=True) but model2vec / sqlite-vec
    are not installed or cannot load."""
class StoreError(AgentRecallError):
    """Wraps an unexpected SQLite-level failure."""
```

### models.py
```python
@dataclass(slots=True)
class MemoryRecord:
    id: int
    content: str
    namespace: str
    tags: list[str]
    metadata: dict
    importance: float
    created_at: datetime          # aware UTC
    updated_at: datetime          # aware UTC
    last_accessed_at: datetime | None
    access_count: int

    def to_dict(self) -> dict: ...        # JSON-safe (datetimes -> isoformat strings)

@dataclass(slots=True)
class MemoryHit:
    record: MemoryRecord
    score: float                  # higher = more relevant
    # convenience pass-throughs:
    @property
    def id(self) -> int: ...
    @property
    def content(self) -> str: ...
    @property
    def metadata(self) -> dict: ...
    @property
    def tags(self) -> list[str]: ...
    def to_dict(self) -> dict: ...        # {"score": ..., **record.to_dict()}
```

### embeddings.py
```python
@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...   # one vector per input

DEFAULT_MODEL = "minishlab/potion-base-8M"

class Model2VecEmbedder:
    """Static-embedding embedder (no torch, no GPU). Lazy-loads the model on first embed.
    Raises EmbeddingsUnavailable if model2vec is not installed."""
    def __init__(self, model: str = DEFAULT_MODEL): ...
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

def get_default_embedder() -> Embedder | None:
    """Return a Model2VecEmbedder if model2vec is importable, else None. Never raises."""
```

### store.py
```python
class SQLiteStore:
    fts_enabled: bool
    def __init__(self, path: str | os.PathLike, *, embedding_dim: int | None = None): ...
        # embedding_dim != None => prepare for vectors (table created on first write).
    def add(self, *, content: str, namespace: str, tags: list[str], metadata: dict,
            importance: float, embedding: list[float] | None = None) -> MemoryRecord: ...
    def get(self, memory_id: int) -> MemoryRecord | None: ...
    def update(self, memory_id: int, *, content=None, tags=None, metadata=None,
               importance=None, embedding=None) -> MemoryRecord: ...   # MemoryNotFound if absent
    def delete(self, memory_id: int) -> bool: ...
    def touch(self, memory_ids: list[int]) -> None: ...   # bump last_accessed_at + access_count
    def all(self, *, namespace: str | None = None, tags: list[str] | None = None,
            limit: int | None = None, offset: int = 0) -> list[MemoryRecord]: ...
    def count(self, *, namespace: str | None = None) -> int: ...
    def fts_search(self, query: str, *, namespace: str | None, tags: list[str] | None,
                   limit: int) -> list[tuple[int, float]]: ...   # (id, bm25_relevance) best-first
    def vec_search(self, embedding: list[float], *, namespace: str | None,
                   tags: list[str] | None, limit: int) -> list[tuple[int, float]]: ...  # (id, similarity)
    def has_vectors(self) -> bool: ...
    def purge(self, *, before: datetime | None = None, namespace: str | None = None,
              keep_last: int | None = None) -> int: ...   # returns rows deleted
    def close(self) -> None: ...
```
- `fts_search` returns a **relevance** score where higher is better (negate raw bm25).
- `vec_search` returns cosine **similarity** in [0, 1]-ish where higher is better (convert
  from sqlite-vec cosine distance via `1 - distance`).
- Tag filtering matches memories that contain **all** requested tags
  (`json_each` over the `tags` column).

### search.py
```python
def sanitize_fts_query(query: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH string: extract word tokens,
    double-quote each, join with OR. Returns '' if no usable tokens."""

def reciprocal_rank_fusion(rankings: list[list[int]], *, k: int = 60) -> dict[int, float]:
    """RRF: for each ranked id-list, add 1/(k + rank0) (rank0 is 0-based). Returns id->score."""

def fuse_and_rank(
    fts_hits: list[tuple[int, float]],
    vec_hits: list[tuple[int, float]],
    records: dict[int, MemoryRecord],
    *, recency_weight: float, importance_weight: float, now: datetime, limit: int,
) -> list[MemoryHit]:
    """Combine keyword + vector candidates with RRF, then apply optional multiplicative
    recency and importance boosts, sort desc by final score, return top `limit` hits.
    recency boost = exp(-age_days / 30) scaled by recency_weight; importance boost scaled
    by importance_weight. Weights of 0 mean pure relevance (RRF only)."""
```

### memory.py — the public facade
```python
class Memory:
    def __init__(
        self,
        path: str | os.PathLike = "agentrecall.db",
        *,
        namespace: str = "default",
        embeddings: bool | str = "auto",      # "auto" | True | False
        embedder: Embedder | None = None,
        recency_weight: float = 0.0,
        importance_weight: float = 0.0,
    ): ...
    # "auto": use `embedder` if given, else a default model2vec embedder if importable,
    #         else fall back to keyword-only (no error).
    # True:   require semantic mode; raise EmbeddingsUnavailable if unavailable.
    # False:  keyword-only even if model2vec is installed.

    @property
    def semantic_enabled(self) -> bool: ...

    def add(self, content: str, *, tags: list[str] | None = None,
            metadata: dict | None = None, importance: float = 1.0,
            namespace: str | None = None) -> MemoryRecord: ...
    def add_many(self, items: list[str | dict]) -> list[MemoryRecord]: ...
        # each item is a str (content) or a dict of add() kwargs incl. "content".
    def search(self, query: str, *, k: int = 5, namespace: str | None = None,
               tags: list[str] | None = None, recency_weight: float | None = None,
               importance_weight: float | None = None) -> list[MemoryHit]: ...
    def get(self, memory_id: int) -> MemoryRecord: ...            # MemoryNotFound if absent
    def update(self, memory_id: int, *, content=None, tags=None,
               metadata=None, importance=None) -> MemoryRecord: ...
    def delete(self, memory_id: int) -> bool: ...
    def all(self, *, namespace: str | None = None, tags: list[str] | None = None,
            limit: int | None = None, offset: int = 0) -> list[MemoryRecord]: ...
    def count(self, *, namespace: str | None = None) -> int: ...
    def forget(self, *, before: datetime | None = None, namespace: str | None = None,
               keep_last: int | None = None) -> int: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Memory": ...
    def __exit__(self, *exc) -> None: ...
```
Behavior notes:
- `namespace=None` in a method means "use the instance default namespace". `search`/`all`/
  `count`/`forget` with an explicit `namespace` override it for that call.
- `search` re-embeds the query (if semantic), runs `fts_search` and (if semantic)
  `vec_search` each over `max(k*4, 20)` candidates, fuses with `fuse_and_rank`, calls
  `store.touch()` on the returned ids, and returns `k` hits. If the sanitized query is
  empty AND not semantic, returns `[]`.
- `recency_weight`/`importance_weight` default to the instance values; per-call args
  override.
- `add` with semantic mode computes the embedding via the embedder before storing.

### config.py
```python
@dataclass
class Settings:
    db_path: str = "agentrecall.db"
    namespace: str = "default"
    embeddings: str = "auto"          # "auto" | "true" | "false"
    model: str = DEFAULT_MODEL
    @classmethod
    def from_env(cls) -> "Settings": ...   # reads AGENTRECALL_DB / _NAMESPACE / _EMBEDDINGS / _MODEL
```
Stdlib only (`os.environ`). No pydantic.

### mcp_server.py  (optional extra: `mcp`)
```python
def build_mcp_server(memory: Memory, *, name: str = "agentrecall"):
    """Return a configured mcp.server.fastmcp.FastMCP exposing tools:
       remember(content, tags=None, metadata=None, importance=1.0) -> dict
       recall(query, k=5, tags=None) -> list[dict]
       forget(memory_id) -> bool
       list_memories(limit=20) -> list[dict]
       memory_stats() -> dict
    Importing this module without the `mcp` extra raises EmbeddingsUnavailable? No —
    raises a clear ImportError with install hint inside build_mcp_server()."""
def serve(memory: Memory, *, transport: str = "stdio") -> None: ...
```

### cli.py
```python
def main(argv: list[str] | None = None) -> int: ...
```
Subcommands (argparse), all honoring `--db` and `AGENTRECALL_*` env:
- `add <content> [--tags a,b] [--importance N] [--namespace NS]`
- `search <query> [-k N] [--namespace NS] [--tags a,b] [--json]`
- `list [--namespace NS] [--limit N] [--json]`
- `get <id>` / `delete <id>`
- `forget [--before ISO] [--keep-last N] [--namespace NS]`
- `stats`
- `export [--format json|md] [--namespace NS]`
- `serve [--transport stdio]`  (MCP server)
Human-readable tables by default; `--json` for machine output.

## __init__.py public exports
```python
from agentrecall.memory import Memory
from agentrecall.models import MemoryRecord, MemoryHit
from agentrecall.embeddings import Embedder, Model2VecEmbedder, get_default_embedder
from agentrecall.errors import (
    AgentRecallError, MemoryNotFound, EmbeddingsUnavailable, StoreError,
)
__version__ = "0.1.0"
__all__ = [ ...all of the above... ]
```

## Dependencies
- **Core:** none (stdlib `sqlite3`, `json`, `argparse`, `dataclasses`, `datetime`).
- `[semantic]` = `model2vec>=0.3`, `sqlite-vec>=0.1.6`, `numpy`.
- `[mcp]`     = `mcp>=1.2`.
- `[dev]`     = `pytest>=8`, `ruff>=0.6`.
- `[all]`     = semantic + mcp.

## Testing rules
- Tests for the **FTS-only** path must pass with **zero third-party deps** installed.
- Semantic tests are guarded by `pytest.importorskip("model2vec")` /
  `importorskip("sqlite_vec")` and must be skipped (not failed) when deps are absent.
- No test may hit the network at import/collection time. (model2vec model download is
  allowed only inside an explicitly-marked semantic test.)
- Cover: add/get/update/delete, namespaces, tag filtering (all-of), FTS recall &
  ranking, RRF fusion math, query sanitization (injection-safe), forget/purge variants,
  round-trip persistence (close/reopen), `to_dict()` JSON-safety, FTS-fallback path.

## Non-goals (v0.1) — keep the surface small
- No automatic LLM fact-extraction / summarization (by design — see positioning).
- No ANN index / billion-scale vectors (sqlite-vec linear scan; agent-scale only).
- No async API (sync is enough; `sqlite3` is fast and local). Document threadsafety:
  one `Memory` per thread, or open with `check_same_thread=False` at your own risk.
- No knowledge-graph / entity-relation modeling (that's Cognee/Zep's lane).
