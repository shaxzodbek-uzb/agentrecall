"""MCP server build test (skipped without the `mcp` extra)."""

import json

import pytest

pytest.importorskip("mcp")

from agentrecall import Memory
from agentrecall.mcp_server import build_mcp_server


def test_build_mcp_server(tmp_path):
    mem = Memory(str(tmp_path / "m.db"), embeddings=False)
    try:
        server = build_mcp_server(mem)
        assert server is not None
        # FastMCP servers carry the configured name.
        assert getattr(server, "name", "agentrecall") == "agentrecall"
    finally:
        mem.close()


def test_tools_registered(tmp_path):
    import asyncio

    mem = Memory(str(tmp_path / "m.db"), embeddings=False)
    try:
        server = build_mcp_server(mem)
        tools = asyncio.run(server.list_tools())
        names = {t.name for t in tools}
        assert {"remember", "recall", "forget", "list_memories", "memory_stats"} <= names
    finally:
        mem.close()


def test_tool_execution_end_to_end(tmp_path):
    import asyncio

    mem = Memory(str(tmp_path / "m.db"), embeddings=False)
    try:
        server = build_mcp_server(mem)

        async def run():
            await server.call_tool("remember", {"content": "buy oat milk", "tags": ["shop"]})
            return await server.call_tool("recall", {"query": "oat milk"})

        result = asyncio.run(run())
        # call_tool's return shape varies across mcp versions (TextContent blocks and/or a
        # structured payload); flatten to a string and assert the content round-tripped.
        assert "oat milk" in json.dumps(result, default=str)
        assert mem.count() == 1
    finally:
        mem.close()
