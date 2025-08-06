import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from default_tools.kb_repo_management.kb_repo_maintanence_tools import (
    ListKnowledgeBaseDirectory,
    SeeKnowledgeBaseFile,
    MoveOrRenameInKnowledgeBase,
    DeleteFromKnowledgeBase,
)

TEST_DIR = Path(__file__).parent / "temp_data" / "maintenance"

class TestKBMaintenanceTools(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        (TEST_DIR / "docs").mkdir(parents=True)
        (TEST_DIR / "tmp").mkdir(parents=True)
        (TEST_DIR / "docs" / "readme.md").write_text("Hello\nWorld")
        (TEST_DIR / "tmp" / "to_delete.txt").write_text("Delete me")

        self.indexer = MagicMock()
        self.indexer.root = TEST_DIR

    def test_list_directory(self):
        tool = ListKnowledgeBaseDirectory(self.indexer)
        result = tool.forward("docs")
        self.assertIn("readme.md", result)

    def test_see_file(self):
        tool = SeeKnowledgeBaseFile(self.indexer)
        result = tool.forward("docs/readme.md")
        self.assertIn("1: Hello", result)

    def test_see_file_nonexistent(self):
        tool = SeeKnowledgeBaseFile(self.indexer)
        result = tool.forward("nonexistent.md")
        self.assertIn("does not exist", result)

    def test_move_or_rename(self):
        tool = MoveOrRenameInKnowledgeBase(self.indexer)
        tool.forward("docs/readme.md", "docs/renamed.md", overwrite=True)
        self.assertTrue((TEST_DIR / "docs/renamed.md").exists())

    def test_delete(self):
        tool = DeleteFromKnowledgeBase(self.indexer)
        result = tool.forward("tmp/to_delete.txt")
        self.assertFalse((TEST_DIR / "tmp/to_delete.txt").exists())

    def test_delete_nonexistent(self):
        tool = DeleteFromKnowledgeBase(self.indexer)
        result = tool.forward("no/such/file.txt")
        self.assertIn("does not exist", result)

def tearDownModule():
    # Clean up the temp_data directory specific to this test module
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

if __name__ == "__main__":
    unittest.main()
