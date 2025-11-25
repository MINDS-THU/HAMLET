from __future__ import annotations

from pathlib import Path

from hamlet.tools.kb_repo_management.kb_repo_retrieval_tools import (
    CopyFromKnowledgeBase,
    KeywordSearchKnowledgeBase,
    SemanticSearchKnowledgeBase,
)


def test_semantic_search_returns_indexer_text(repo_indexer_stub) -> None:
    tool = SemanticSearchKnowledgeBase(repo_indexer_stub)

    output = tool(query="graph theory")

    assert output == "unique:graph theory:3"
    assert repo_indexer_stub.semantic_queries[-1] == ("graph theory", 3)


def test_keyword_search_in_file(repo_indexer_stub) -> None:
    target = repo_indexer_stub.root / "doc.txt"
    target.write_text("alpha\nkeyword beta\n", encoding="utf-8")

    tool = KeywordSearchKnowledgeBase(repo_indexer_stub)
    result = tool(path="doc.txt", keyword="keyword", context_lines=0)

    assert "Matches in [doc.txt]" in result
    assert "keyword beta" in result


def test_copy_from_kb_to_working_dir(repo_indexer_stub, working_dir: Path) -> None:
    src = repo_indexer_stub.root / "paper.md"
    src.write_text("content", encoding="utf-8")

    tool = CopyFromKnowledgeBase(repo_indexer_stub, str(working_dir))
    msg = tool(source_path="paper.md", destination_path="copies/paper.md", overwrite=False)

    copied = working_dir / "copies" / "paper.md"
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == "content"
    assert "copied" in msg.lower()
