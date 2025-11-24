from __future__ import annotations

from pathlib import Path

from src.hamlet.tools.kb_repo_management.kb_repo_maintainance_tools import (
    DeleteFromKnowledgeBase,
    ListKnowledgeBaseDirectory,
    MoveOrRenameInKnowledgeBase,
    SeeKnowledgeBaseFile,
)


def test_list_kb_directory_outputs_entries(repo_indexer_stub) -> None:
    folder = repo_indexer_stub.root / "section"
    folder.mkdir()
    (folder / "a.txt").write_text("A", encoding="utf-8")
    (folder / "b.txt").write_text("B", encoding="utf-8")

    tool = ListKnowledgeBaseDirectory(repo_indexer_stub)
    result = tool(directory="section")

    assert set(result.splitlines()) == {"a.txt", "b.txt"}


def test_see_kb_file_numbers_lines(repo_indexer_stub) -> None:
    target = repo_indexer_stub.root / "info.md"
    target.write_text("first\nsecond\n", encoding="utf-8")

    tool = SeeKnowledgeBaseFile(repo_indexer_stub)
    output = tool(file_path="info.md")

    assert output.startswith("1:")
    assert "2: second" in output


def test_move_or_rename_in_kb_moves_file(repo_indexer_stub) -> None:
    src = repo_indexer_stub.root / "old.txt"
    src.write_text("data", encoding="utf-8")

    tool = MoveOrRenameInKnowledgeBase(repo_indexer_stub)
    response = tool(source_path="old.txt", destination_path="archive/new.txt", overwrite=False)

    assert "archive" in response
    assert not src.exists()
    assert (repo_indexer_stub.root / "archive" / "new.txt").exists()


def test_delete_from_kb_removes_path(repo_indexer_stub) -> None:
    entry = repo_indexer_stub.root / "remove.md"
    entry.write_text("bye", encoding="utf-8")

    tool = DeleteFromKnowledgeBase(repo_indexer_stub)
    msg = tool(target_path="remove.md")

    assert "Deleted" in msg
    assert not entry.exists()
