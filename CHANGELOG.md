# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-06-20

Initial release.

### Added
- `Memory` — agent memory backed by a single SQLite file, with `add`, `add_many`,
  `search`, `get`, `update`, `delete`, `all`, `count`, and `forget`.
- Keyword recall via SQLite FTS5 (BM25) with **zero third-party dependencies**, plus a
  `LIKE`-based fallback for SQLite builds without FTS5.
- Optional torch-free semantic search (`agentrecall[semantic]`): model2vec static
  embeddings stored in sqlite-vec, fused with keyword results via Reciprocal Rank Fusion.
- Namespaces for per-user / per-agent isolation; tag and metadata storage; all-of tag
  filtering; optional recency and importance ranking boosts.
- Injection-safe FTS query sanitization.
- MCP server (`agentrecall[mcp]`, `agentrecall serve`) exposing `remember` / `recall` /
  `forget` / `list_memories` / `memory_stats` — an embeddings-capable alternative to the
  keyword-only JSONL memory server.
- `agentrecall` CLI: `add`, `search`, `list`, `get`, `delete`, `forget`, `stats`,
  `export`, `serve`.

[Unreleased]: https://github.com/shaxzodbek-uzb/agentrecall/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shaxzodbek-uzb/agentrecall/releases/tag/v0.1.0
