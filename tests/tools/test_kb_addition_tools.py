from __future__ import annotations

from pathlib import Path

from hamlet.tools.kb_repo_management.kb_repo_addition_tools import (
    AppendToKnowledgeBaseFile,
    CopyToKnowledgeBase,
    WriteToKnowledgeBase,
)


def test_write_to_kb_creates_and_indexes_file(repo_indexer_stub) -> None:
    tool = WriteToKnowledgeBase(repo_indexer_stub)

    message = tool(content="hello", destination_path="notes.md", overwrite=False)

    target = repo_indexer_stub.root / "notes.md"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello"
    assert repo_indexer_stub.updated_paths == [target]
    assert "notes.md" in message


def test_write_to_kb_generates_unique_suffix(repo_indexer_stub) -> None:
    tool = WriteToKnowledgeBase(repo_indexer_stub)
    tool(content="first", destination_path="report.md", overwrite=False)
    message = tool(content="second", destination_path="report.md", overwrite=False)

    generated = sorted(repo_indexer_stub.root.glob("report*.md"))
    assert len(generated) == 2
    assert any("report_" in path.stem for path in generated)
    assert "indexed" in message.lower()


def test_copy_to_kb_from_working_dir(repo_indexer_stub, working_dir: Path) -> None:
    source = working_dir / "draft.txt"
    source.write_text("draft", encoding="utf-8")

    tool = CopyToKnowledgeBase(repo_indexer_stub, str(working_dir))
    response = tool(source_path="draft.txt", destination_path="folder/draft.txt", overwrite=False)

    destination = repo_indexer_stub.root / "folder" / "draft.txt"
    assert destination.exists()
    assert destination.read_text(encoding="utf-8") == "draft"
    assert destination in repo_indexer_stub.updated_paths
    assert "Indexed 1 file" in response


def test_append_to_kb_inserts_before_match(repo_indexer_stub) -> None:
    target = repo_indexer_stub.root / "doc.md"
    target.write_text("line 1\nline 2\n", encoding="utf-8")

    tool = AppendToKnowledgeBaseFile(repo_indexer_stub)
    message = tool(
        target_file="doc.md",
        new_content="inserted",
        insert_mode="before",
        match_string="line 2",
    )

    assert "inserted" in target.read_text(encoding="utf-8")
    assert message.startswith("Inserted content before")
    assert target in repo_indexer_stub.updated_paths
