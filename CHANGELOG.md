# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-08-16

### Added
- **Expiring memories.** `add`, `add_many` and `update` accept `ttl=` (`"30d"`,
  `"12h"`, `"1h30m"`, or a number of seconds) or an absolute `expires_at=`. An
  expired memory stops being recalled — it is filtered out of `search`, `get`,
  `all` and `count` — without being deleted, so nothing is lost to a clock skew or
  a TTL you later regret.
- `include_expired=True` on every read to see past the deadline, and
  `Memory.purge_expired()` to delete for real. The two are deliberately separate:
  expiry hides, purging destroys.
- CLI: `add --ttl`, `--include-expired` on `search`/`list`/`get`,
  and `forget --expired`.
- MCP: `remember` takes `ttl`, and a new `forget_expired` tool.
- `agentrecall import` — load `export --format json` output, or JSONL, from a file
  or stdin. Round-trips with `export`, so moving a memory store between machines no
  longer needs a hand-written script.

### Changed
- **Database schema is now version 2** (adds `expires_at`). Existing files are
  migrated in place on open; no action is needed and 0.1.0 databases keep working.

### Fixed
- **Keyword search silently missed rows in databases created before the FTS index
  existed.** Those rows were never indexed, so `search` returned nothing for them
  and reported no error. The index is now rebuilt whenever it has to be created
  against a table that already has rows. Row counts cannot detect this drift: an
  external-content FTS5 table answers `count(*)` from the content table, so it
  always agrees with itself.

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

[Unreleased]: https://github.com/shaxzodbek-uzb/agentrecall/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shaxzodbek-uzb/agentrecall/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shaxzodbek-uzb/agentrecall/releases/tag/v0.1.0
