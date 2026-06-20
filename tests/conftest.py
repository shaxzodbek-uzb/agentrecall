import pytest

from agentrecall import Memory


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def mem(db_path):
    m = Memory(db_path, embeddings=False)
    try:
        yield m
    finally:
        m.close()
