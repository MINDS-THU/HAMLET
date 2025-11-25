from __future__ import annotations

from pathlib import Path
import types

import pytest

from hamlet.tools.file_editing import file_editing_tools
from hamlet.tools.file_editing.file_editing_tools import (
    CreateFileWithContent,
    DeleteFileOrFolder,
    ListDir,
    LoadObjectFromPythonFile,
    ModifyFile,
    ReadBinaryAsMarkdown,
    SearchKeyword,
    SeeTextFile,
)


def test_list_dir_returns_contents(working_dir: Path) -> None:
    (working_dir / "a.txt").write_text("hello", encoding="utf-8")
    (working_dir / "b.txt").write_text("world", encoding="utf-8")

    tool = ListDir(str(working_dir))
    result = tool(directory=".")

    assert set(result.splitlines()) == {"a.txt", "b.txt"}


def test_see_text_file_numbers_lines(working_dir: Path) -> None:
    target = working_dir / "notes.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    tool = SeeTextFile(str(working_dir))
    output = tool(filename="notes.txt")

    assert "1:alpha" in output
    assert "2:beta" in output


def test_read_binary_as_markdown_uses_markitdown(monkeypatch: pytest.MonkeyPatch, working_dir: Path) -> None:
    target = working_dir / "file.bin"
    target.write_bytes(b"stub")

    class DummyMark:
        def convert(self, path: str):
            assert Path(path) == target
            return types.SimpleNamespace(text_content="converted text")

    monkeypatch.setattr(file_editing_tools, "MarkItDown", lambda: DummyMark())

    tool = ReadBinaryAsMarkdown(str(working_dir))
    assert tool(filename="file.bin") == "converted text"


def test_read_binary_as_markdown_handles_missing(working_dir: Path) -> None:
    tool = ReadBinaryAsMarkdown(str(working_dir))
    assert "does not exist" in tool(filename="missing.bin")


def test_modify_file_replaces_requested_lines(working_dir: Path) -> None:
    target = working_dir / "script.py"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    tool = ModifyFile(str(working_dir))
    message = tool(filename="script.py", start_line=2, end_line=2, new_content="TWO")

    assert message == "Content modified."
    assert target.read_text(encoding="utf-8") == "one\nTWO\nthree\n"


def test_create_file_with_content_builds_parents(working_dir: Path) -> None:
    tool = CreateFileWithContent(str(working_dir))
    rel_path = Path("nested") / "example.txt"

    response = tool(filename=str(rel_path), content="payload")

    created = working_dir / rel_path
    assert response == "File created successfully."
    assert created.read_text(encoding="utf-8") == "payload"


def test_search_keyword_recurses_directory(working_dir: Path) -> None:
    src = working_dir / "pkg"
    src.mkdir()
    (src / "file_a.py").write_text("target line\n", encoding="utf-8")
    (src / "file_b.py").write_text("nothing here", encoding="utf-8")

    tool = SearchKeyword(str(working_dir))
    result = tool(path="pkg", keyword="target", context_lines=0)

    assert "Matches in [pkg" in result  # path separators vary by platform
    assert "target line" in result
    assert "file_b" not in result


def test_delete_file_or_folder_supports_bulk_delete(working_dir: Path) -> None:
    (working_dir / "temp.txt").write_text("bye", encoding="utf-8")
    (working_dir / "dir").mkdir()
    (working_dir / "dir" / "inner.txt").write_text("bye", encoding="utf-8")

    tool = DeleteFileOrFolder(str(working_dir))
    message = tool(filename="")

    assert message.startswith("All files")
    assert not any(working_dir.iterdir())


def test_load_object_from_python_file(working_dir: Path) -> None:
    module_path = working_dir / "helpers.py"
    module_path.write_text(
        """
class Echo:
    def apply(self):
        return "ok"
""".strip(),
        encoding="utf-8",
    )

    tool = LoadObjectFromPythonFile(str(working_dir))
    Echo = tool(filename="helpers.py", object_name="Echo")

    assert Echo().apply() == "ok"
