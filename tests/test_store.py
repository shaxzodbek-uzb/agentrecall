"""Store-level tests: FTS fallback, vector-table state, purge variants."""

from datetime import datetime, timezone

from agentrecall.store import SQLiteStore


def _store(tmp_path):
    return SQLiteStore(str(tmp_path / "s.db"))


def test_fts_enabled_on_modern_sqlite(tmp_path):
    store = _store(tmp_path)
    assert store.fts_enabled is True
    store.close()


def test_like_fallback_path(tmp_path):
    store = _store(tmp_path)
    store.add(content="orange fox jumps", namespace="default", tags=[], metadata={}, importance=1.0)
    store.add(content="blue whale swims", namespace="default", tags=[], metadata={}, importance=1.0)
    # Force the no-FTS5 fallback and confirm keyword recall still works.
    store.fts_enabled = False
    hits = store.fts_search("fox", namespace="default", tags=None, limit=5)
    assert len(hits) == 1
    contents = {store.get(mid).content for mid, _ in hits}
    assert "orange fox jumps" in contents
    store.close()


def test_has_vectors_false_without_semantic(tmp_path):
    store = _store(tmp_path)
    assert store.has_vectors() is False
    # vec_search must degrade gracefully to [] (no vector table, extension may be absent).
    assert store.vec_search([0.1, 0.2], namespace=None, tags=None, limit=5) == []
    store.close()


def test_purge_namespace_scoped(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.add(content=f"a{i}", namespace="A", tags=[], metadata={}, importance=1.0)
    for i in range(5):
        store.add(content=f"b{i}", namespace="B", tags=[], metadata={}, importance=1.0)
    removed = store.purge(namespace="A", keep_last=2)
    assert removed == 3
    assert store.count(namespace="A") == 2
    assert store.count(namespace="B") == 5
    store.close()


def test_purge_keep_last_is_per_namespace(tmp_path):
    store = _store(tmp_path)
    for i in range(4):
        store.add(content=f"a{i}", namespace="A", tags=[], metadata={}, importance=1.0)
    for i in range(4):
        store.add(content=f"b{i}", namespace="B", tags=[], metadata={}, importance=1.0)
    removed = store.purge(keep_last=1)  # keep newest 1 *per namespace*
    assert removed == 6
    assert store.count(namespace="A") == 1
    assert store.count(namespace="B") == 1
    store.close()


def test_purge_before(tmp_path):
    store = _store(tmp_path)
    store.add(content="old", namespace="default", tags=[], metadata={}, importance=1.0)
    future = datetime.now(timezone.utc).replace(year=2099)
    assert store.purge(before=future) == 1
    assert store.count() == 0
    store.close()


def test_purge_no_criteria_noop(tmp_path):
    store = _store(tmp_path)
    store.add(content="x", namespace="default", tags=[], metadata={}, importance=1.0)
    assert store.purge() == 0
    assert store.count() == 1
    store.close()


def test_touch_increments(tmp_path):
    store = _store(tmp_path)
    rec = store.add(content="x", namespace="default", tags=[], metadata={}, importance=1.0)
    store.touch([rec.id])
    store.touch([rec.id])
    assert store.get(rec.id).access_count == 2
    store.close()
