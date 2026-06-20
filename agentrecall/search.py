"""Ranking: query sanitization, reciprocal-rank fusion, and boost application.

Kept dependency-free and pure so it is trivial to unit-test without a database.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

from .models import MemoryHit, MemoryRecord

_WORD = re.compile(r"\w+", re.UNICODE)


def sanitize_fts_query(query: str) -> str:
    """Turn arbitrary user text into a safe FTS5 ``MATCH`` string.

    FTS5 treats characters like ``"``, ``*``, ``(``, ``:`` and the words ``AND``/``OR``/
    ``NOT``/``NEAR`` as query operators, so passing raw user input to ``MATCH`` can raise
    ``fts5: syntax error`` or match unexpectedly. We extract plain word tokens, double-
    quote each (so operator-words become literals), and join them with ``OR`` for recall.

    Returns ``""`` when the input has no usable tokens (caller should skip FTS).
    """
    tokens = _WORD.findall(query or "")
    if not tokens:
        return ""
    # \w never contains a double quote, so quoting is injection-safe.
    return " OR ".join(f'"{tok}"' for tok in tokens)


def reciprocal_rank_fusion(rankings: list[list[int]], *, k: int = 60) -> dict[int, float]:
    """Reciprocal-rank fusion of several best-first id rankings.

    Each list contributes ``1 / (k + rank)`` (rank is 0-based) to its ids. This blends
    keyword (BM25) and vector (cosine) candidate lists without needing their raw, non-
    comparable scores.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, memory_id in enumerate(ranking):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_and_rank(
    fts_hits: list[tuple[int, float]],
    vec_hits: list[tuple[int, float]],
    records: dict[int, MemoryRecord],
    *,
    recency_weight: float,
    importance_weight: float,
    now: datetime,
    limit: int,
) -> list[MemoryHit]:
    """Fuse keyword + vector candidates and apply optional recency/importance boosts.

    With both weights at ``0`` the result is pure relevance (RRF order). Otherwise the
    fused base score is scaled by ``1 + recency_weight*recency + importance_weight*importance``
    where ``recency = exp(-age_days / 30)`` decays over ~a month.
    """
    fused = reciprocal_rank_fusion([[mid for mid, _ in fts_hits], [mid for mid, _ in vec_hits]])

    hits: list[MemoryHit] = []
    for memory_id, base in fused.items():
        record = records.get(memory_id)
        if record is None:
            continue
        score = base
        if recency_weight or importance_weight:
            age_days = max(0.0, (now - record.created_at).total_seconds() / 86400.0)
            recency = math.exp(-age_days / 30.0)
            score = base * (1.0 + recency_weight * recency + importance_weight * record.importance)
        hits.append(MemoryHit(record=record, score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


__all__ = ["sanitize_fts_query", "reciprocal_rank_fusion", "fuse_and_rank"]
