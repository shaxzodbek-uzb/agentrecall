"""CLI tests via main(); forced to keyword mode so no model downloads."""

import json

from agentrecall.cli import main


def run(db, *args):
    return main(["--db", db, "--embeddings", "false", *args])


def test_add_and_list(db_path, capsys):
    assert run(db_path, "add", "buy milk", "--tags", "todo,home") == 0
    capsys.readouterr()
    assert run(db_path, "list", "--json") == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert len(data) == 1
    assert data[0]["content"] == "buy milk"
    assert data[0]["tags"] == ["todo", "home"]


def test_search_json(db_path, capsys):
    run(db_path, "add", "the capital of France is Paris")
    capsys.readouterr()
    assert run(db_path, "search", "France", "-k", "3", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data and "France" in data[0]["content"]
    assert "score" in data[0]


def test_stats(db_path, capsys):
    run(db_path, "add", "one")
    run(db_path, "add", "two")
    capsys.readouterr()
    assert run(db_path, "stats") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["count"] == 2
    assert data["semantic"] is False


def test_get_and_delete(db_path, capsys):
    run(db_path, "add", "deletable")
    capsys.readouterr()
    assert run(db_path, "get", "1") == 0
    assert json.loads(capsys.readouterr().out)["id"] == 1
    assert run(db_path, "delete", "1") == 0
    assert "deleted" in capsys.readouterr().out
    # deleting again -> exit code 1
    assert run(db_path, "delete", "1") == 1


def test_get_missing_exit_code(db_path, capsys):
    assert run(db_path, "get", "404") == 1


def test_forget_keep_last(db_path, capsys):
    for i in range(6):
        run(db_path, "add", f"item {i}")
    capsys.readouterr()
    assert run(db_path, "forget", "--keep-last", "2") == 0
    assert "forgot 4" in capsys.readouterr().out


def test_export_markdown(db_path, capsys):
    run(db_path, "add", "exported note", "--tags", "x")
    capsys.readouterr()
    assert run(db_path, "export", "--format", "md") == 0
    out = capsys.readouterr().out
    assert "exported note" in out
    assert out.lstrip().startswith("- **#1**")
