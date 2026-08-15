"""Expiring memories: duration parsing, TTL visibility rules, purge, and migration."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agentrecall import Memory, MemoryNotFound, parse_duration
from agentrecall.cli import main
from agentrecall.store import SQLiteStore


def _past(seconds: int = 5) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


def _future(seconds: int = 3600) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


# -- parse_duration ----------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("30d", timedelta(days=30)),
        ("12h", timedelta(hours=12)),
        ("1h30m", timedelta(hours=1, minutes=30)),
        ("2w", timedelta(weeks=2)),
        ("45s", timedelta(seconds=45)),
        ("90", timedelta(seconds=90)),
        ("1.5h", timedelta(hours=1.5)),
        (3600, timedelta(hours=1)),
        (timedelta(days=2), timedelta(days=2)),
    ],
)
def test_parse_duration_accepts(value, expected):
    assert parse_duration(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "banana", "30x", "-5", 0, -1, True, None, []])
def test_parse_duration_rejects(value):
    with pytest.raises(ValueError):
        parse_duration(value)


# -- add / visibility --------------------------------------------------------


def test_add_without_ttl_never_expires(mem):
    record = mem.add("permanent fact")
    assert record.expires_at is None
    assert record.is_expired() is False


def test_add_with_ttl_sets_a_future_deadline(mem):
    record = mem.add("temporary fact", ttl="1h")
    assert record.expires_at is not None
    assert record.is_expired() is False
    assert timedelta(minutes=59) < record.expires_at - datetime.now(timezone.utc)


def test_ttl_and_expires_at_are_mutually_exclusive(mem):
    with pytest.raises(ValueError, match="not both"):
        mem.add("x", ttl="1h", expires_at=_future())


def test_expired_memory_drops_out_of_search(mem):
    mem.add("the user prefers dark mode", expires_at=_past())
    mem.add("the user prefers tabs", expires_at=_future())
    contents = [hit.content for hit in mem.search("user prefers", k=10)]
    assert contents == ["the user prefers tabs"]


def test_expired_memory_drops_out_of_list_and_count(mem):
    mem.add("gone", expires_at=_past())
    mem.add("here")
    assert [r.content for r in mem.all()] == ["here"]
    assert mem.count() == 1
    assert mem.count(include_expired=True) == 2


def test_include_expired_brings_them_back(mem):
    mem.add("gone", expires_at=_past())
    assert len(mem.all(include_expired=True)) == 1
    assert len(mem.search("gone", k=5, include_expired=True)) == 1


def test_get_hides_expired_by_default(mem):
    record = mem.add("gone", expires_at=_past())
    with pytest.raises(MemoryNotFound):
        mem.get(record.id)
    assert mem.get(record.id, include_expired=True).content == "gone"


def test_expiry_boundary_is_exclusive_of_the_deadline(mem):
    """A memory expiring exactly now is already gone, not still live."""
    record = mem.add("edge", expires_at=datetime.now(timezone.utc))
    assert record.is_expired() is True


def test_naive_and_offset_datetimes_normalize_to_utc(mem):
    """A deadline given in another zone must not shift when compared in SQL."""
    plus_five = timezone(timedelta(hours=5))
    # 1 hour ahead of now, expressed in UTC+5 — still in the future.
    record = mem.add("tashkent", expires_at=datetime.now(plus_five) + timedelta(hours=1))
    assert record.is_expired() is False
    assert mem.count() == 1


# -- update / renew ----------------------------------------------------------


def test_update_ttl_revives_an_expired_memory(mem):
    record = mem.add("stale", expires_at=_past())
    assert mem.count() == 0
    mem.update(record.id, ttl="1h")
    assert mem.count() == 1
    assert mem.get(record.id).is_expired() is False


def test_update_expires_at_none_makes_it_permanent(mem):
    record = mem.add("temporary", ttl="1h")
    updated = mem.update(record.id, expires_at=None)
    assert updated.expires_at is None


def test_update_leaves_expiry_alone_when_not_mentioned(mem):
    record = mem.add("temporary", ttl="1h")
    updated = mem.update(record.id, content="edited")
    assert updated.expires_at == record.expires_at


def test_update_rejects_ttl_and_expires_at_together(mem):
    record = mem.add("x")
    with pytest.raises(ValueError, match="not both"):
        mem.update(record.id, ttl="1h", expires_at=_future())


# -- purge -------------------------------------------------------------------


def test_purge_expired_only_removes_expired(mem):
    mem.add("gone", expires_at=_past())
    mem.add("also gone", expires_at=_past())
    mem.add("still here", ttl="1h")
    mem.add("permanent")
    assert mem.purge_expired() == 2
    assert mem.count(include_expired=True) == 2
    assert mem.purge_expired() == 0


def test_purge_expired_leaves_a_never_expiring_store_untouched(mem):
    mem.add("a")
    mem.add("b")
    assert mem.purge_expired() == 0
    assert mem.count() == 2


def test_add_many_accepts_ttl_per_item(mem):
    records = mem.add_many(
        [
            {"content": "short", "ttl": "1h"},
            {"content": "expired", "expires_at": _past()},
            {"content": "forever"},
        ]
    )
    assert records[0].expires_at is not None
    assert records[1].is_expired() is True
    assert records[2].expires_at is None
    assert mem.count() == 2


# -- schema migration --------------------------------------------------------


def test_opens_a_v1_database_and_adds_the_column(tmp_path):
    """A database written before TTL existed must keep working, with rows never expiring."""
    path = str(tmp_path / "v1.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace        TEXT    NOT NULL DEFAULT 'default',
            content          TEXT    NOT NULL,
            tags             TEXT    NOT NULL DEFAULT '[]',
            metadata         TEXT    NOT NULL DEFAULT '{}',
            importance       REAL    NOT NULL DEFAULT 1.0,
            created_at       TEXT    NOT NULL,
            updated_at       TEXT    NOT NULL,
            last_accessed_at TEXT,
            access_count     INTEGER NOT NULL DEFAULT 0
        );
        PRAGMA user_version = 1;
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memories(content, created_at, updated_at) VALUES ('legacy fact', ?, ?)",
        (now, now),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(path)
    try:
        columns = {r["name"] for r in store.conn.execute("PRAGMA table_info(memories)")}
        assert "expires_at" in columns
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 2
        records = store.all()
        assert [r.content for r in records] == ["legacy fact"]
        assert records[0].expires_at is None
    finally:
        store.close()

    with Memory(path, embeddings=False) as m:
        assert m.count() == 1
        assert [h.content for h in m.search("legacy")] == ["legacy fact"]


# -- serialization -----------------------------------------------------------


def test_to_dict_round_trips_expiry(mem):
    record = mem.add("temporary", ttl="1h")
    data = record.to_dict()
    assert datetime.fromisoformat(data["expires_at"]) == record.expires_at
    assert mem.add("permanent").to_dict()["expires_at"] is None


# -- CLI ---------------------------------------------------------------------


def test_cli_add_ttl_and_forget_expired(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "--embeddings", "false", "add", "temp", "--ttl", "1h"]) == 0
    assert "expires" in capsys.readouterr().out

    main(["--db", db, "--embeddings", "false", "list"])
    assert "temp" in capsys.readouterr().out

    # Nothing is expired yet, so --expired is a no-op.
    main(["--db", db, "--embeddings", "false", "forget", "--expired"])
    assert "forgot 0" in capsys.readouterr().out


def test_cli_rejects_a_bad_ttl(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "--embeddings", "false", "add", "x", "--ttl", "banana"]) == 2
    assert "invalid duration" in capsys.readouterr().err


def test_cli_export_import_round_trip(tmp_path, capsys):
    src, dst = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    main(["--db", src, "--embeddings", "false", "add", "hello", "--tags", "greeting"])
    main(["--db", src, "--embeddings", "false", "add", "temp", "--ttl", "1h"])
    capsys.readouterr()  # drop the add output so only the export lands in the dump
    main(["--db", src, "--embeddings", "false", "export"])
    dump = tmp_path / "dump.json"
    dump.write_text(capsys.readouterr().out, encoding="utf-8")

    assert main(["--db", dst, "--embeddings", "false", "import", str(dump)]) == 0
    assert "imported 2" in capsys.readouterr().out

    with Memory(dst, embeddings=False) as m:
        by_content = {r.content: r for r in m.all()}
        assert by_content["hello"].tags == ["greeting"]
        assert by_content["hello"].expires_at is None
        assert by_content["temp"].expires_at is not None


def test_cli_import_reads_jsonl_and_stdin(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "c.db")
    lines = "\n".join(json.dumps({"content": c}) for c in ["one", "two"])
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(lines))
    assert main(["--db", db, "--embeddings", "false", "import", "-"]) == 0
    assert "imported 2" in capsys.readouterr().out


def test_cli_import_rejects_malformed_entries(tmp_path, capsys):
    db = str(tmp_path / "d.db")
    bad = tmp_path / "bad.json"
    bad.write_text('[{"tags": []}]', encoding="utf-8")
    assert main(["--db", db, "--embeddings", "false", "import", str(bad)]) == 2
    assert "content" in capsys.readouterr().err


def test_cli_stats_reports_expired_count(tmp_path, capsys):
    db = str(tmp_path / "e.db")
    with Memory(db, embeddings=False) as m:
        m.add("gone", expires_at=_past())
        m.add("here")
    main(["--db", db, "--embeddings", "false", "stats"])
    data = json.loads(capsys.readouterr().out)
    assert data["count"] == 1
    assert data["expired"] == 1
