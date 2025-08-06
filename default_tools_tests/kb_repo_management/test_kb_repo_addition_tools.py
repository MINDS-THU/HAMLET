import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from default_tools.kb_repo_management.kb_repo_addition_tools import (
    WriteToKnowledgeBase,
    CopyToKnowledgeBase,
    AppendToKnowledgeBaseFile,
)

TEST_DIR = Path(__file__).parent / "temp_data" / "addition"

class TestKBAdditionTools(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        (TEST_DIR / "working").mkdir(parents=True)
        (TEST_DIR / "kb").mkdir(parents=True)

        self.indexer = MagicMock()
        self.indexer.root = TEST_DIR / "kb"
        self.indexer.update_file = MagicMock()

        # Working file
        (TEST_DIR / "working" / "sample.txt").write_text("Sample content\n")

        # Write tool
        self.write_tool = WriteToKnowledgeBase(self.indexer)
        self.copy_tool = CopyToKnowledgeBase(self.indexer, working_dir=str(TEST_DIR / "working"))
        self.append_tool = AppendToKnowledgeBaseFile(self.indexer)

    def test_write_file(self):
        result = self.write_tool.forward("Initial content\n", "docs/write.txt", overwrite=False)
        self.assertIn("Wrote content", result)

    def test_write_file_overwrite_true(self):
        dst = TEST_DIR / "kb" / "docs/overwrite.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("Old content")
        result = self.write_tool.forward("New content\n", "docs/overwrite.txt", overwrite=True)
        self.assertIn("Wrote content", result)
        self.assertIn("New content", dst.read_text())

    def test_append_file_end_and_before(self):
        self.write_tool.forward("Initial line\n", "notes/append.md", overwrite=True)
        self.append_tool.forward("notes/append.md", "Appended end\n", "end")
        self.append_tool.forward("notes/append.md", "Insert before\n", "before", "Initial")
        content = (TEST_DIR / "kb/notes/append.md").read_text()
        self.assertIn("Insert before", content)
        self.assertIn("Appended end", content)

    def test_append_after_and_fallback(self):
        self.write_tool.forward("==Header==\n", "notes/append2.md", overwrite=True)
        self.append_tool.forward("notes/append2.md", "After header\n", "after", "==Header==")
        self.append_tool.forward("notes/append2.md", "Fallback insert\n", "before", "Not in file")
        content = (TEST_DIR / "kb/notes/append2.md").read_text()
        self.assertIn("After header", content)
        self.assertIn("Fallback insert", content)

    def test_copy_file(self):
        result = self.copy_tool.forward("sample.txt", "docs/sample_copy.txt", overwrite=True)
        self.assertTrue((TEST_DIR / "kb/docs/sample_copy.txt").exists())

    def test_copy_directory(self):
        (TEST_DIR / "working" / "subdir").mkdir()
        (TEST_DIR / "working" / "subdir" / "f.txt").write_text("hello")
        result = self.copy_tool.forward("subdir", "docs/copied_dir", overwrite=False)
        self.assertTrue((TEST_DIR / "kb/docs/copied_dir/f.txt").exists())

def tearDownModule():
    # Clean up the temp_data directory specific to this test module
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    unittest.main()
