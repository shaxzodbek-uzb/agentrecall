"""Semantic-path tests.

The fake-embedder tests exercise the full sqlite-vec integration deterministically and
need no network. The model2vec tests are skipped if the model can't be downloaded.
"""

import math

import pytest

from agentrecall import Memory

sqlite_vec = pytest.importorskip("sqlite_vec")


class FakeEmbedder:
    """Deterministic, offline embedder for testing the vector path."""

    dim = 6

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for i, ch in enumerate(text.lower()):
                vec[i % self.dim] += ord(ch) % 13
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


@pytest.fixture
def fake_mem(tmp_path):
    m = Memory(str(tmp_path / "vec.db"), embeddings=True, embedder=FakeEmbedder())
    try:
        yield m
    finally:
        m.close()


def test_fake_embedder_enables_semantic(fake_mem):
    assert fake_mem.semantic_enabled is True


def test_vector_table_created_on_first_add(fake_mem):
    assert fake_mem._store.has_vectors() is False
    fake_mem.add("first memory")
    assert fake_mem._store.has_vectors() is True


def test_vector_search_returns_nearest(fake_mem):
    fake_mem.add("alpha alpha alpha")
    fake_mem.add("zzzz different content entirely")
    # Query identical to the first memory -> it should come back as a hit.
    hits = fake_mem.search("alpha alpha alpha", k=2)
    assert hits
    assert hits[0].content == "alpha alpha alpha"


def test_semantic_update_reembeds(fake_mem):
    rec = fake_mem.add("original")
    fake_mem.update(rec.id, content="completely rewritten")
    # vector row still present and queryable
    hits = fake_mem.search("completely rewritten", k=1)
    assert hits and hits[0].id == rec.id


def test_semantic_delete_removes_vector(fake_mem):
    rec = fake_mem.add("to be removed")
    assert fake_mem.delete(rec.id) is True
    assert fake_mem.search("to be removed", k=5) == []


def test_embeddings_true_without_model2vec(tmp_path, monkeypatch):
    # If model2vec is absent and no embedder is given, embeddings=True must raise.
    import agentrecall.memory as memory_mod

    monkeypatch.setattr(memory_mod, "get_default_embedder", lambda: None)
    from agentrecall import EmbeddingsUnavailable

    with pytest.raises(EmbeddingsUnavailable):
        Memory(str(tmp_path / "x.db"), embeddings=True)


# -- real model2vec (skipped offline) ---------------------------------------


def test_model2vec_real_recall(tmp_path):
    pytest.importorskip("model2vec")
    mem = Memory(str(tmp_path / "real.db"), embeddings=True)
    try:
        try:
            mem.add("I love hiking in the mountains on weekends", metadata={"k": "hike"})
        except Exception as exc:  # model download failed (offline CI)
            pytest.skip(f"model2vec model unavailable: {exc}")
        mem.add("My favorite programming language is Python", metadata={"k": "py"})
        hits = mem.search("outdoor activities in nature", k=2)
        scores = {h.metadata.get("k"): h.score for h in hits}
        assert "hike" in scores
        if "py" in scores:
            assert scores["hike"] >= scores["py"]
    finally:
        mem.close()
