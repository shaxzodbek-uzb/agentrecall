"""Quickstart: store a few memories and search them.

Run with:  python examples/quickstart.py
Works on a bare `pip install agentrecall` (keyword recall). Install
`agentrecall[semantic]` to make `search()` match on meaning too.
"""

from agentrecall import Memory

with Memory("quickstart.db") as mem:
    mem.add("The user prefers dark mode in the UI", tags=["preference", "ui"])
    mem.add("User's name is Aziz and he lives in Tashkent", metadata={"kind": "fact"})
    mem.add("Project deadline is July 7th", tags=["project"], importance=2.0)

    print(f"semantic search: {mem.semantic_enabled}")
    print(f"stored {mem.count()} memories\n")

    for query in ["what does the user like?", "when is the deadline?"]:
        print(f"? {query}")
        for hit in mem.search(query, k=2):
            print(f"  {hit.score:6.3f}  {hit.content}")
        print()
