"""
Knowledge Base Repo Maintenance Tools
This module contains tools for managing files and folders in a knowledge base stored as a structured repository.
Supports listing, viewing, moving, renaming, and deleting files within the knowledge base.
"""

from smolagents.tools import Tool
import os
import shutil
from pathlib import Path


class ListKnowledgeBaseDirectory(Tool):
    name = "list_knowledge_base_directory"
    description = (
        "List all files and folders inside a directory in the knowledge base. "
        "Use this to explore the structure of the knowledge base."
    )
    inputs = {
        "directory": {"type": "string", "description": "Relative path of the directory in the knowledge base."}
    }
    output_type = "string"

    def __init__(self, repo_indexer):
        super().__init__()
        self.root = Path(repo_indexer.root)

    def forward(self, directory: str) -> str:
        chosen_dir = self.root / directory
        if not chosen_dir.exists():
            return f"The directory '{directory}' does not exist in the knowledge base."
        if not chosen_dir.is_dir():
            return f"The path '{directory}' is not a directory."
        files = os.listdir(chosen_dir)
        if not files:
            return f"The directory '{directory}' is empty."
        return "\n".join(files)


class SeeKnowledgeBaseFile(Tool):
    name = "see_knowledge_base_file"
    description = (
        "View the content of a plain text file in the knowledge base (e.g., .txt, .md, .py). "
        "Avoid using this tool on binary files like .pdf, .docx, or images."
    )
    inputs = {
        "file_path": {"type": "string", "description": "Relative path of the file in the knowledge base."}
    }
    output_type = "string"

    def __init__(self, repo_indexer):
        super().__init__()
        self.root = Path(repo_indexer.root)

    def forward(self, file_path: str) -> str:
        filepath = self.root / file_path
        if not filepath.exists():
            return f"The file '{file_path}' does not exist in the knowledge base."
        if not filepath.is_file():
            return f"The path '{file_path}' is not a file."
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            return f"Cannot read '{file_path}' — it may be a binary or non-text file."
        formatted_lines = [f"{i+1}: {line}" for i, line in enumerate(lines)]
        return "".join(formatted_lines)


class MoveOrRenameInKnowledgeBase(Tool):
    name = "move_or_rename_in_knowledge_base"
    description = (
        "Move or rename a file or folder within the knowledge base. "
        "Use this to reorganize files or to change file/folder names. "
        "If overwrite=True, replaces the destination. If overwrite=False, adds a numeric suffix to avoid conflict."
    )
    inputs = {
        "source_path": {"type": "string", "description": "Current path of the file or folder in the knowledge base."},
        "destination_path": {"type": "string", "description": "New desired path or name in the knowledge base."},
        "overwrite": {"type": "boolean", "description": "Whether to overwrite the destination if it exists."}
    }
    output_type = "string"

    def __init__(self, repo_indexer):
        super().__init__()
        self.root = Path(repo_indexer.root)

    def _get_unique_path(self, base_path: Path) -> Path:
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(f"{base_path.stem}_{counter}{base_path.suffix}")
            counter += 1
        return new_path

    def forward(self, source_path: str, destination_path: str, overwrite: bool) -> str:
        src = self.root / source_path
        dst = self.root / destination_path

        if not src.exists():
            return f"Source '{source_path}' does not exist in the knowledge base."

        if dst.exists() and not overwrite:
            dst = self._get_unique_path(dst)

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved or renamed '{source_path}' to '{dst.relative_to(self.root)}' in the knowledge base."


class DeleteFromKnowledgeBase(Tool):
    name = "delete_from_knowledge_base"
    description = (
        "Delete a file or folder from the knowledge base. "
        "Use this to remove outdated or invalid content."
    )
    inputs = {
        "target_path": {"type": "string", "description": "Path to the file or folder to delete."}
    }
    output_type = "string"

    def __init__(self, repo_indexer):
        super().__init__()
        self.root = Path(repo_indexer.root)

    def forward(self, target_path: str) -> str:
        target = self.root / target_path

        if not target.exists():
            return f"The path '{target_path}' does not exist in the knowledge base."

        if target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
        return f"Deleted '{target_path}' from the knowledge base."
