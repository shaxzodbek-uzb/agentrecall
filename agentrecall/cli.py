"""``agentrecall`` command-line interface (argparse, stdlib only).

All subcommands honour ``--db`` and the ``AGENTRECALL_*`` environment variables. Use
``--json`` on read commands for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from . import __version__
from .config import Settings
from .memory import Memory


def _split_tags(value: str | None) -> list[str] | None:
    if not value:
        return None
    tags = [t.strip() for t in value.split(",") if t.strip()]
    return tags or None


def _build_memory(args: argparse.Namespace) -> Memory:
    settings = Settings.from_env()
    db_path = getattr(args, "db", None) or settings.db_path
    namespace = getattr(args, "namespace", None) or settings.namespace
    embeddings = settings.embeddings_value()
    if getattr(args, "embeddings", None):
        embeddings = {"auto": "auto", "true": True, "false": False}[args.embeddings]
    return Memory(db_path, namespace=namespace, embeddings=embeddings)


def _truncate(text: str, width: int = 80) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _cmd_add(args: argparse.Namespace) -> int:
    with _build_memory(args) as mem:
        try:
            record = mem.add(
                args.content,
                tags=_split_tags(args.tags),
                importance=args.importance,
                ttl=args.ttl,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    expiry = f"  (expires {record.expires_at:%Y-%m-%d %H:%M}Z)" if record.expires_at else ""
    print(f"#{record.id}  {_truncate(record.content)}{expiry}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    with _build_memory(args) as mem:
        hits = mem.search(args.query, k=args.k, tags=_split_tags(args.tags))
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("(no matches)")
        return 0
    for hit in hits:
        print(f"{hit.score:6.3f}  #{hit.id:<4} {_truncate(hit.content)}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with _build_memory(args) as mem:
        records = mem.all(
            limit=args.limit,
            tags=_split_tags(args.tags),
            include_expired=args.include_expired,
        )
    if args.json:
        print(json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2))
        return 0
    if not records:
        print("(empty)")
        return 0
    for record in records:
        tags = f"  [{', '.join(record.tags)}]" if record.tags else ""
        mark = "  (expired)" if record.is_expired() else ""
        print(f"#{record.id:<4} {_truncate(record.content)}{tags}{mark}")
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    from .errors import MemoryNotFound

    with _build_memory(args) as mem:
        try:
            record = mem.get(args.id)
        except MemoryNotFound:
            print(f"no memory with id {args.id}", file=sys.stderr)
            return 1
    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    with _build_memory(args) as mem:
        ok = mem.delete(args.id)
    print("deleted" if ok else "not found")
    return 0 if ok else 1


def _cmd_forget(args: argparse.Namespace) -> int:
    before = datetime.fromisoformat(args.before) if args.before else None
    with _build_memory(args) as mem:
        removed = mem.purge_expired() if args.expired else 0
        removed += mem.forget(before=before, keep_last=args.keep_last)
    print(f"forgot {removed} memorie(s)")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    with _build_memory(args) as mem:
        live = mem.count()
        data = {
            "namespace": mem.namespace,
            "count": live,
            "expired": mem.count(include_expired=True) - live,
            "semantic": mem.semantic_enabled,
        }
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    with _build_memory(args) as mem:
        records = mem.all()
    if args.format == "json":
        print(json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2))
    else:  # markdown
        for record in records:
            tags = f" `{' '.join(record.tags)}`" if record.tags else ""
            print(f"- **#{record.id}**{tags} — {record.content}")
    return 0


def _load_records(raw: str) -> list[dict]:
    """Parse either a JSON array (``export --format json``) or one JSON object per line."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("expected a JSON array of memory objects")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _cmd_import(args: argparse.Namespace) -> int:
    raw = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    try:
        data = _load_records(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"could not parse input: {exc}", file=sys.stderr)
        return 2

    items: list[dict] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
            print(f"entry {i} has no string 'content' field", file=sys.stderr)
            return 2
        item: dict = {
            "content": entry["content"],
            "tags": entry.get("tags") or [],
            "metadata": entry.get("metadata") or {},
            "importance": entry.get("importance", 1.0),
        }
        if args.namespace is None and entry.get("namespace"):
            item["namespace"] = entry["namespace"]
        if entry.get("expires_at"):
            item["expires_at"] = datetime.fromisoformat(entry["expires_at"])
        items.append(item)

    with _build_memory(args) as mem:
        records = mem.add_many(items)
        expired = sum(1 for r in records if r.is_expired())
    # Ids are assigned by the destination database, so they are not preserved.
    note = f" ({expired} already past their expiry)" if expired else ""
    print(f"imported {len(records)} memorie(s){note}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .mcp_server import serve

    mem = _build_memory(args)
    try:
        serve(mem, transport=args.transport)
    finally:
        mem.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentrecall",
        description="Agent memory in a single SQLite file.",
    )
    parser.add_argument("--version", action="version", version=f"agentrecall {__version__}")
    parser.add_argument("--db", help="database file (env AGENTRECALL_DB)")
    parser.add_argument("--namespace", help="namespace (env AGENTRECALL_NAMESPACE)")
    parser.add_argument(
        "--embeddings", choices=["auto", "true", "false"], help="semantic search mode"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="store a memory")
    p_add.add_argument("content")
    p_add.add_argument("--tags", help="comma-separated tags")
    p_add.add_argument("--importance", type=float, default=1.0)
    p_add.add_argument(
        "--ttl",
        help="make the memory temporary, e.g. 30d, 12h, 1h30m, or a number of seconds",
    )
    p_add.set_defaults(func=_cmd_add)

    p_search = sub.add_parser("search", help="search memories")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--tags", help="comma-separated tags filter")
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=_cmd_search)

    p_list = sub.add_parser("list", help="list memories")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--tags", help="comma-separated tags filter")
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument(
        "--include-expired",
        action="store_true",
        dest="include_expired",
        help="also show memories whose TTL has elapsed",
    )
    p_list.set_defaults(func=_cmd_list)

    p_get = sub.add_parser("get", help="show one memory")
    p_get.add_argument("id", type=int)
    p_get.set_defaults(func=_cmd_get)

    p_delete = sub.add_parser("delete", help="delete one memory")
    p_delete.add_argument("id", type=int)
    p_delete.set_defaults(func=_cmd_delete)

    p_forget = sub.add_parser("forget", help="bulk-delete by age or count")
    p_forget.add_argument("--before", help="ISO-8601 datetime; delete older")
    p_forget.add_argument("--keep-last", type=int, dest="keep_last")
    p_forget.add_argument(
        "--expired",
        action="store_true",
        help="permanently delete memories whose TTL has elapsed",
    )
    p_forget.set_defaults(func=_cmd_forget)

    p_stats = sub.add_parser("stats", help="show counts and config")
    p_stats.set_defaults(func=_cmd_stats)

    p_export = sub.add_parser("export", help="dump memories")
    p_export.add_argument("--format", choices=["json", "md"], default="json")
    p_export.set_defaults(func=_cmd_export)

    p_import = sub.add_parser(
        "import",
        help="load memories from 'export --format json' output (or JSONL); '-' reads stdin",
    )
    p_import.add_argument("path", help="file to read, or '-' for stdin")
    p_import.set_defaults(func=_cmd_import)

    p_serve = sub.add_parser("serve", help="run the MCP server")
    p_serve.add_argument("--transport", default="stdio")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
