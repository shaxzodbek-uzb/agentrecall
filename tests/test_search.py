"""Pure-logic tests for the ranking helpers — no database, no deps."""

from datetime import datetime, timedelta, timezone

from agentrecall.models import MemoryRecord
from agentrecall.search import (
    fuse_and_rank,
    reciprocal_rank_fusion,
    sanitize_fts_query,
)


def test_sanitize_basic():
    assert sanitize_fts_query("hello world") == '"hello" OR "world"'


def test_sanitize_empty():
    assert sanitize_fts_query("") == ""
    assert sanitize_fts_query("   ") == ""
    assert sanitize_fts_query("!!! ??? ---") == ""


def test_sanitize_quotes_operators_as_literals():
    # FTS5 operators AND/OR/NOT must not survive as operators.
    out = sanitize_fts_query('dark AND mode OR "')
    assert out == '"dark" OR "AND" OR "mode" OR "OR"'
    assert '"' in out and "(" not in out


def test_sanitize_no_raw_double_quote_breaks():
    # A lone double quote / parens / star must not appear unquoted.
    out = sanitize_fts_query('a"b (c) d*')
    for tok in ("a", "b", "c", "d"):
        assert f'"{tok}"' in out


def test_rrf_orders_by_blended_rank():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 1]])
    # id 1: appears at rank 0 and rank 1; id 3: rank 2 and rank 0; id 2: rank 1 only.
    assert scores[1] > scores[3] > scores[2]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def _rec(memory_id, *, importance=1.0, age_days=0.0):
    now = datetime.now(timezone.utc)
    created = now - timedelta(days=age_days)
    return MemoryRecord(
        id=memory_id,
        content=f"memory {memory_id}",
        namespace="default",
        tags=[],
        metadata={},
        importance=importance,
        created_at=created,
        updated_at=created,
        last_accessed_at=None,
        access_count=0,
    )


def test_fuse_pure_relevance_when_weights_zero():
    records = {1: _rec(1), 2: _rec(2)}
    hits = fuse_and_rank(
        [(1, 5.0), (2, 1.0)],
        [],
        records,
        recency_weight=0.0,
        importance_weight=0.0,
        now=datetime.now(timezone.utc),
        limit=10,
    )
    assert [h.id for h in hits] == [1, 2]


def test_fuse_importance_boost_changes_order():
    # Same RRF rank order, but id 2 is far more important.
    records = {1: _rec(1, importance=1.0), 2: _rec(2, importance=50.0)}
    hits = fuse_and_rank(
        [(1, 5.0), (2, 1.0)],
        [],
        records,
        recency_weight=0.0,
        importance_weight=1.0,
        now=datetime.now(timezone.utc),
        limit=10,
    )
    assert hits[0].id == 2


def test_fuse_recency_boost_prefers_recent():
    records = {1: _rec(1, age_days=365.0), 2: _rec(2, age_days=0.0)}
    hits = fuse_and_rank(
        [(1, 5.0), (2, 5.0)],  # equal base rank
        [],
        records,
        recency_weight=5.0,
        importance_weight=0.0,
        now=datetime.now(timezone.utc),
        limit=10,
    )
    assert hits[0].id == 2


def test_fuse_drops_missing_records_and_respects_limit():
    records = {1: _rec(1)}  # id 2 has no record
    hits = fuse_and_rank(
        [(1, 1.0), (2, 1.0)],
        [],
        records,
        recency_weight=0.0,
        importance_weight=0.0,
        now=datetime.now(timezone.utc),
        limit=1,
    )
    assert [h.id for h in hits] == [1]
