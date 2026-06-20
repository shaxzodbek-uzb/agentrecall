# agentrecall

**Agent memory in a single SQLite file.** No vector database, no server, no cloud, no API key.

```bash
pip install agentrecall
```

```python
from agentrecall import Memory

with Memory("agent.db") as mem:
    mem.add("The user prefers dark mode", tags=["preference"])
    mem.add("User's name is Aziz; lives in Tashkent", metadata={"kind": "fact"})

    for hit in mem.search("what UI settings does the user like?", k=3):
        print(hit.score, hit.content)
```

That's the whole setup. `agent.db` is an ordinary SQLite file you can `cp`, `git diff`,
back up, inspect with any SQLite tool, and read from any language. Nothing else is running.

---

## Why another memory library?

Most "memory layers" for agents are **infrastructure**. To get started you stand up a
vector database, run a server, sign up for a cloud, or hand over an API key — and many of
them call an LLM on every write to "extract" facts, which is slow, costs tokens, and is
non-deterministic.

`agentrecall` is the opposite. It is a **library**, the store is **one file**, recall is
**deterministic**, and **nothing leaves the machine**.

| | infra needed | semantic search | offline | stores | LLM call per write |
|---|---|---|---|---|---|
| **agentrecall** | **none** (1 file) | ✅ torch-free, opt-in | ✅ | SQLite | ❌ verbatim |
| mem0 | vector DB / cloud | ✅ | ⚠️ | vector + KV + graph | ✅ |
| Letta / MemGPT | server + Postgres | ✅ | ⚠️ | Postgres + pgvector | ✅ |
| Zep | server + datastore | ✅ | ⚠️ | knowledge graph | ✅ |
| official MCP memory server | none | ❌ keyword only | ✅ | JSONL flat file | ❌ |

Three things `agentrecall` does that nothing else combines:

1. **Zero infrastructure.** The core has **no third-party dependencies** — keyword recall
   runs on Python's stdlib `sqlite3` (FTS5 + BM25). A fresh `pip install agentrecall` with
   nothing else works.
2. **Semantic search with no torch, no GPU, no download server.** Add the `[semantic]`
   extra and you get hybrid keyword + vector recall powered by
   [model2vec](https://github.com/MinishLab/model2vec) static embeddings (~10 MB, CPU-only)
   stored in [sqlite-vec](https://github.com/asg017/sqlite-vec). Still one file, still offline.
3. **Verbatim & deterministic.** `agentrecall` never calls an LLM to mutate your memories.
   What you `add()` is what is stored — no silent fact-extraction, no cloud round-trip, no
   surprise token bills.

---

## Install

```bash
pip install agentrecall                 # core: keyword recall, stdlib only
pip install "agentrecall[semantic]"     # + torch-free semantic search (model2vec + sqlite-vec)
pip install "agentrecall[mcp]"          # + MCP server
pip install "agentrecall[all]"          # everything
```

## Semantic search (optional, torch-free)

```python
from agentrecall import Memory

# embeddings="auto" (the default) turns semantic on automatically *iff* the
# [semantic] extra is installed, and silently stays keyword-only otherwise.
mem = Memory("agent.db", embeddings="auto")
print(mem.semantic_enabled)   # True once you've installed agentrecall[semantic]

mem.add("I love hiking in the mountains on weekends")
hits = mem.search("outdoor hobbies")     # matches even with zero shared keywords
```

Search is **hybrid**: keyword (FTS5/BM25) and vector (cosine) candidates are blended with
[Reciprocal Rank Fusion](https://learn.microsoft.com/azure/search/hybrid-search-ranking),
so you get the precision of keywords and the recall of embeddings. Bring your own embedder
(OpenAI, a local model, anything) by passing `embedder=` — any object with `.dim` and
`.embed(texts) -> list[list[float]]`.

Optional ranking boosts:

```python
mem = Memory("agent.db", recency_weight=0.5, importance_weight=0.3)
mem.add("Critical: API key rotates on the 1st", importance=3.0)
mem.search("api key", recency_weight=1.0)   # per-call override
```

## Namespaces

Isolate memories per user, per agent, or per session with a namespace:

```python
alice = Memory("app.db", namespace="user:alice")
bob   = Memory("app.db", namespace="user:bob")     # same file, isolated memories
alice.add("prefers metric units")
bob.search("units")          # never sees Alice's memories
```

## As an MCP server

Give Claude (or any MCP client) persistent, searchable memory — an embeddings-capable
alternative to the official keyword-only JSONL memory server:

```bash
pip install "agentrecall[mcp]"
agentrecall serve --db ~/.agent-memory.db
```

```jsonc
// Claude Desktop / Claude Code MCP config
{
  "mcpServers": {
    "memory": {
      "command": "agentrecall",
      "args": ["serve", "--db", "/Users/me/.agent-memory.db"]
    }
  }
}
```

Tools exposed: `remember`, `recall`, `forget`, `list_memories`, `memory_stats`.

## CLI

```bash
agentrecall add "Deadline is July 7" --tags project --importance 2
agentrecall search "when is the deadline" -k 3
agentrecall list --limit 10
agentrecall stats
agentrecall forget --keep-last 1000        # prune to the newest 1000 per namespace
agentrecall export --format md > memories.md
```

Every command honours `--db`, `--namespace`, and the `AGENTRECALL_DB` /
`AGENTRECALL_NAMESPACE` / `AGENTRECALL_EMBEDDINGS` environment variables.

## API at a glance

```python
mem.add(content, *, tags=None, metadata=None, importance=1.0, namespace=None) -> MemoryRecord
mem.add_many([str | dict, ...])                                              -> list[MemoryRecord]
mem.search(query, *, k=5, namespace=None, tags=None,
           recency_weight=None, importance_weight=None)                      -> list[MemoryHit]
mem.get(id) / mem.update(id, ...) / mem.delete(id)
mem.all(*, namespace=None, tags=None, limit=None, offset=0)                  -> list[MemoryRecord]
mem.count(*, namespace=None) -> int
mem.forget(*, before=None, namespace=None, keep_last=None) -> int           # deleted count
```

`tags` filtering matches memories containing **all** of the requested tags.

## The database is just SQLite

No magic. Open it with anything:

```sql
sqlite3 agent.db "SELECT id, content, importance, created_at FROM memories ORDER BY created_at DESC LIMIT 5;"
```

Schema: a `memories` table (with JSON `tags`/`metadata` columns), an FTS5 index kept in
sync by triggers, and — only in semantic mode — a `sqlite-vec` vector table. See
[`SPEC.md`](SPEC.md) for the full contract.

## Scope & limits (honest defaults)

- **Agent-scale, not web-scale.** sqlite-vec uses a linear scan (no ANN index yet) — great
  for thousands of memories per namespace, not millions of RAG chunks.
- **Sync, single-process.** Use one `Memory` per thread. `sqlite3` is fast and local; there
  is no async API by design.
- **No knowledge graph / entity-relation modeling.** That's [Cognee](https://github.com/topoteretes/cognee)
  and [Zep](https://www.getzep.com/)'s lane. `agentrecall` stays small on purpose.
- **No automatic summarization.** Memories are stored verbatim. If you want LLM-distilled
  memories, distill before you `add()` — your call, your model, your tokens.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

The keyword (FTS-only) test suite runs with **zero third-party dependencies**. Semantic
tests are skipped automatically when the `[semantic]` extra isn't installed.

## License

MIT © 2026 Shaxzodbek Sobirov / Blaze
