"""Expose a :class:`~agentrecall.memory.Memory` as an MCP server.

This is a drop-in, embeddings-capable alternative to the official memory MCP server
(which persists to a keyword-only JSONL flat file). Here memories live in a portable
SQLite file and recall can be hybrid keyword + semantic.

Requires the optional ``mcp`` extra:  ``pip install 'agentrecall-db[mcp]'``.
"""

from __future__ import annotations

from .memory import Memory


def build_mcp_server(memory: Memory, *, name: str = "agentrecall"):
    """Return a configured FastMCP server exposing remember/recall/forget tools."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The agentrecall MCP server needs the optional 'mcp' extra. "
            "Install it with:  pip install 'agentrecall-db[mcp]'"
        ) from exc

    server = FastMCP(name)

    @server.tool()
    def remember(
        content: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        importance: float = 1.0,
        ttl: str | None = None,
    ) -> dict:
        """Store a new memory verbatim and return the stored record.

        Set ``ttl`` ("30d", "12h", "1h30m", or seconds) for facts that go stale — a
        temporary preference, a value that's only true for this session. The memory
        stops being recalled once the deadline passes.
        """
        record = memory.add(content, tags=tags, metadata=metadata, importance=importance, ttl=ttl)
        return record.to_dict()

    @server.tool()
    def recall(query: str, k: int = 5, tags: list[str] | None = None) -> list[dict]:
        """Search stored memories; returns the most relevant first."""
        return [hit.to_dict() for hit in memory.search(query, k=k, tags=tags)]

    @server.tool()
    def forget(memory_id: int) -> bool:
        """Delete a memory by id. Returns True if a row was removed."""
        return memory.delete(memory_id)

    @server.tool()
    def forget_expired() -> int:
        """Permanently delete memories whose TTL has elapsed. Returns how many went."""
        return memory.purge_expired()

    @server.tool()
    def list_memories(limit: int = 20) -> list[dict]:
        """List the most recent memories."""
        return [record.to_dict() for record in memory.all(limit=limit)]

    @server.tool()
    def memory_stats() -> dict:
        """Return counts and the active configuration."""
        return {
            "count": memory.count(),
            "namespace": memory.namespace,
            "semantic": memory.semantic_enabled,
        }

    return server


def serve(memory: Memory, *, transport: str = "stdio") -> None:
    """Build and run the MCP server (blocking)."""
    build_mcp_server(memory).run(transport=transport)


__all__ = ["build_mcp_server", "serve"]
