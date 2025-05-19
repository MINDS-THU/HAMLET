import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from general_tools.kb_repo_management.kb_repo_retrieval_tools import (
    SemanticSearchKnowledgeBase,
    KeywordSearchKnowledgeBase,
    CopyFromKnowledgeBase,
)

TEST_DIR = Path(__file__).parent / "temp_data" / "retrieve"

class TestKBRetrievalTools(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TEST_DIR, ignore_errors=True)

        # Create KB and working dir
        self.kb_root = TEST_DIR / "kb"
        self.working_dir = TEST_DIR / "working"
        self.kb_root.mkdir(parents=True)
        self.working_dir.mkdir(parents=True)

        # Create sample content
        (self.kb_root / "docs").mkdir()
        (self.kb_root / "docs" / "f1.md").write_text("simplex method\npivot\n")
        (self.kb_root / "docs" / "f2.md").write_text("dijkstra path\n")
        (self.kb_root / "docs" / "binary.pdf").write_bytes(b"\xFF\xFE\x00\x00\x00garbage")

        # Mock indexer
        self.indexer = MagicMock()
        self.indexer.root = self.kb_root
        self.indexer.get_query_results.return_value = "Mock semantic result"

    def test_semantic_search(self):
        tool = SemanticSearchKnowledgeBase(self.indexer)
        result = tool.forward("shortest path")
        self.assertIn("Mock semantic result", result)

    def test_keyword_search_file(self):
        tool = KeywordSearchKnowledgeBase(self.indexer)
        result = tool.forward("docs/f1.md", "pivot", 0)
        self.assertIn("pivot", result)

    def test_keyword_search_folder(self):
        tool = KeywordSearchKnowledgeBase(self.indexer)
        result = tool.forward("docs", "dijkstra", 0)
        self.assertIn("f2.md", result)

    def test_keyword_search_no_match(self):
        tool = KeywordSearchKnowledgeBase(self.indexer)
        result = tool.forward("docs/f1.md", "nonexistent", 0)
        self.assertIn("No matches", result)

    def test_keyword_search_binary(self):
        tool = KeywordSearchKnowledgeBase(self.indexer)
        result = tool.forward("docs/binary.pdf", "pivot", 0)
        self.assertIn("Cannot read binary", result)

    def test_copy_file_overwrite_true(self):
        dst_path = self.working_dir / "copied.md"
        dst_path.write_text("old")  # existing file
        tool = CopyFromKnowledgeBase(self.indexer, working_dir=str(self.working_dir))
        result = tool.forward("docs/f1.md", "copied.md", overwrite=True)
        self.assertIn("overwriting", result)
        self.assertIn("simplex", dst_path.read_text())

    def test_copy_file_overwrite_false(self):
        dst_path = self.working_dir / "copied.md"
        dst_path.write_text("old")
        tool = CopyFromKnowledgeBase(self.indexer, working_dir=str(self.working_dir))
        result = tool.forward("docs/f1.md", "copied.md", overwrite=False)
        self.assertIn("no overwrite", result)
        self.assertTrue("copied_1.md" in result or "copied_2.md" in result)

    def test_copy_folder(self):
        # Copy entire 'docs' directory to working dir
        tool = CopyFromKnowledgeBase(self.indexer, working_dir=str(self.working_dir))
        result = tool.forward("docs", "docs_copy", overwrite=False)
        self.assertTrue((self.working_dir / "docs_copy" / "f2.md").exists())
        self.assertIn("Directory", result)

def tearDownModule():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

if __name__ == "__main__":
    unittest.main()
