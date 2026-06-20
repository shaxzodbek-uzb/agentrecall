"""End-to-end tests for the Memory facade on the keyword (FTS-only) path.

These must pass with zero third-party dependencies installed.
"""

import json

import pytest

from agentrecall import Memory, MemoryNotFound


def test_add_and_get_roundtrip(mem):
    rec = mem.add("hello world", tags=["greeting"], metadata={"lang": "en"}, importance=2.0)
    assert rec.id > 0
    fetched = mem.get(rec.id)
    assert fetched.content == "hello world"
    assert fetched.tags == ["greeting"]
    assert fetched.metadata == {"lang": "en"}
    assert fetched.importance == 2.0
    assert fetched.namespace == "default"
    assert fetched.access_count == 0
    assert fetched.created_at.tzinfo is not None


def test_get_missing_raises(mem):
    with pytest.raises(MemoryNotFound):
        mem.get(999)


def test_count(mem):
    assert mem.count() == 0
    mem.add("one")
    mem.add("two")
    assert mem.count() == 2


def test_search_finds_relevant_excludes_irrelevant(mem):
    mem.add("the elephant walked across the savanna")
    mem.add("quantum entanglement in superconductors")
    hits = mem.search("elephant", k=5)
    assert len(hits) == 1
    assert "elephant" in hits[0].content


def test_search_empty_query_returns_empty(mem):
    mem.add("something")
    assert mem.search("", k=5) == []
    assert mem.search("!!!", k=5) == []


def test_search_touches_access_count(mem):
    rec = mem.add("distinctive zebra content")
    mem.search("zebra")
    after = mem.get(rec.id)
    assert after.access_count == 1
    assert after.last_accessed_at is not None


def test_search_injection_safe(mem):
    mem.add("dark mode preference")
    # None of these should raise an FTS syntax error.
    for q in ["dark OR (mode", 'mode AND "', "NEAR(dark", "*", "a:b:c", '"""']:
        mem.search(q, k=3)


def test_update_changes_content_and_search(mem):
    rec = mem.add("original aardvark text")
    assert len(mem.search("aardvark")) == 1
    mem.update(rec.id, content="rewritten platypus text")
    assert mem.search("aardvark") == []
    assert len(mem.search("platypus")) == 1


def test_update_missing_raises(mem):
    with pytest.raises(MemoryNotFound):
        mem.update(12345, content="x")


def test_delete(mem):
    rec = mem.add("temporary kangaroo note")
    assert mem.delete(rec.id) is True
    assert mem.count() == 0
    assert mem.delete(rec.id) is False
    assert mem.search("kangaroo") == []


def test_add_many(mem):
    recs = mem.add_many(
        [
            "plain string memory",
            {"content": "dict memory", "tags": ["t"], "importance": 3.0},
        ]
    )
    assert len(recs) == 2
    assert recs[1].tags == ["t"]
    assert recs[1].importance == 3.0
    assert mem.count() == 2


def test_namespace_isolation(db_path):
    alice = Memory(db_path, namespace="user:alice", embeddings=False)
    bob = Memory(db_path, namespace="user:bob", embeddings=False)
    try:
        alice.add("alice likes coffee beverages")
        bob.add("bob likes coffee beverages")
        assert alice.count() == 1
        assert bob.count() == 1
        a_hits = alice.search("coffee")
        assert len(a_hits) == 1
        assert all(h.record.namespace == "user:alice" for h in a_hits)
    finally:
        alice.close()
        bob.close()


def test_namespace_override_per_call(mem):
    mem.add("in default namespace")
    mem.add("in other namespace", namespace="other")
    assert mem.count() == 1
    assert mem.count(namespace="other") == 1
    assert len(mem.all(namespace="other")) == 1


def test_tag_filter_matches_all_of(mem):
    mem.add("memory ax", tags=["a", "x"])
    mem.add("memory ay", tags=["a", "y"])
    mem.add("memory none", tags=[])
    assert {r.id for r in mem.all(tags=["a"])} == {1, 2}
    assert {r.id for r in mem.all(tags=["a", "x"])} == {1}
    assert mem.all(tags=["a", "z"]) == []


def test_tag_filter_in_search(mem):
    mem.add("shared keyword apple", tags=["fruit"])
    mem.add("shared keyword apple", tags=["company"])
    hits = mem.search("apple", tags=["fruit"])
    assert len(hits) == 1
    assert hits[0].tags == ["fruit"]


def test_all_ordering_and_limit(mem):
    for i in range(5):
        mem.add(f"memory number {i}")
    recs = mem.all(limit=2)
    assert len(recs) == 2
    # newest first
    assert recs[0].id > recs[1].id


def test_forget_keep_last(mem):
    for i in range(10):
        mem.add(f"memory {i}")
    removed = mem.forget(keep_last=3)
    assert removed == 7
    assert mem.count() == 3


def test_forget_no_criteria_is_noop(mem):
    mem.add("keep me")
    assert mem.forget() == 0
    assert mem.count() == 1


def test_forget_before(mem):
    from datetime import datetime, timezone

    mem.add("old then new")
    future = datetime.now(timezone.utc).replace(year=2099)
    removed = mem.forget(before=future)
    assert removed == 1
    assert mem.count() == 0


def test_persistence_across_reopen(db_path):
    with Memory(db_path, embeddings=False) as m:
        rec = m.add("persisted memory", tags=["keep"], importance=4.0)
        rid = rec.id
    with Memory(db_path, embeddings=False) as m:
        again = m.get(rid)
        assert again.content == "persisted memory"
        assert again.tags == ["keep"]
        assert again.importance == 4.0
        assert len(m.search("persisted")) == 1


def test_to_dict_is_json_safe(mem):
    rec = mem.add("json safety", tags=["x"], metadata={"n": 1})
    payload = json.dumps(rec.to_dict())  # must not raise
    assert "json safety" in payload
    hit = mem.search("json")[0]
    json.dumps(hit.to_dict())
    assert hit.to_dict()["score"] == hit.score


def test_invalid_embeddings_value_raises(db_path):
    with pytest.raises(ValueError):
        Memory(db_path, embeddings="sometimes")


def test_search_nonpositive_k_returns_empty(mem):
    mem.add("alpha distinctive content")
    assert mem.search("alpha", k=0) == []
    assert mem.search("alpha", k=-3) == []


def test_add_many_missing_content_raises(mem):
    with pytest.raises(ValueError):
        mem.add_many([{"tags": ["x"]}])  # dict without 'content'
    assert mem.count() == 0  # nothing partially written


def test_forget_before_and_keep_last_is_union(mem):
    from datetime import datetime, timezone

    for i in range(5):
        mem.add(f"row {i}")
    future = datetime.now(timezone.utc).replace(year=2099)
    # before=future matches all 5; keep_last=2 matches 3; union => all 5 deleted
    # (an intersection would have deleted only 3).
    assert mem.forget(before=future, keep_last=2) == 5
    assert mem.count() == 0


def test_forget_before_non_utc_does_not_overdelete(mem):
    # Regression: a cutoff instant in the past, expressed in a non-UTC timezone whose
    # ISO string sorts lexically *after* the stored UTC string, must not delete a row
    # that is actually newer than the cutoff instant.
    from datetime import datetime, timedelta, timezone

    mem.add("newer than the cutoff instant")
    cutoff_past = (datetime.now(timezone.utc) - timedelta(hours=1)).astimezone(
        timezone(timedelta(hours=5))
    )
    assert mem.forget(before=cutoff_past) == 0
    assert mem.count() == 1
