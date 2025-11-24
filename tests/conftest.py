import importlib
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_markitdown_stub() -> None:
    if "markitdown" in sys.modules:
        return

    stub = types.ModuleType("markitdown")

    class _Result:
        def __init__(self, text: str):
            self.text_content = text

    class MarkItDown:  # pragma: no cover - stub only enables imports
        def convert(self, path: str):
            return _Result(text=f"stub:{path}")

    setattr(stub, "MarkItDown", MarkItDown)
    sys.modules["markitdown"] = stub


def _install_repo_indexer_stub() -> None:
    module_name = "src.hamlet.tools.kb_repo_management.repo_indexer"
    if module_name in sys.modules:
        return

    pkg = importlib.import_module("src.hamlet.tools.kb_repo_management")
    stub = types.ModuleType(module_name)

    class RepoIndexer:  # pragma: no cover - stub only enables imports
        def __init__(self, root: str | Path = "knowledge_base", **_: object):
            self.root = Path(root)
            self.updated_files: list[Path] = []
            self.unique_queries: list[tuple[str, int]] = []

        def update_file(self, path: str | Path) -> None:
            self.updated_files.append(Path(path))

        def get_query_results(self, query: str, k: int = 3) -> str:
            self.unique_queries.append((query, k))
            return f"Results for {query} (top {k})"

        def get_unique_query_results(self, query: str, k: int = 3, max_attempts: int = 10) -> str:  # noqa: ARG002
            self.unique_queries.append((query, k))
            return f"Unique results for {query} (top {k})"

    setattr(stub, "RepoIndexer", RepoIndexer)
    sys.modules[module_name] = stub
    setattr(pkg, "repo_indexer", stub)


def _install_stubs() -> None:
    _install_markitdown_stub()
    _install_repo_indexer_stub()

_install_stubs()


class RepoIndexerStub:
    """Lightweight test double that mimics the subset of RepoIndexer the tools rely on."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.updated_paths: list[Path] = []
        self.semantic_queries: list[tuple[str, int]] = []

    def update_file(self, path: str | Path) -> None:
        self.updated_paths.append(Path(path))

    def get_query_results(self, query: str, k: int = 3) -> str:
        self.semantic_queries.append((query, k))
        return f"results:{query}:{k}"

    def get_unique_query_results(self, query: str, k: int = 3, max_attempts: int = 10) -> str:  # noqa: ARG002
        self.semantic_queries.append((query, k))
        return f"unique:{query}:{k}"


@pytest.fixture
def kb_root(tmp_path: Path) -> Path:
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    return kb_dir


@pytest.fixture
def repo_indexer_stub(kb_root: Path) -> RepoIndexerStub:
    return RepoIndexerStub(kb_root)


@pytest.fixture
def working_dir(tmp_path: Path) -> Path:
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    return work_dir
