# 📚 Knowledge Base Repository Management Tools

This module provides a set of tools to **manage a structured knowledge base** (code, text, etc.) using folder-based repositories. It supports:

* 🔍 Retrieving knowledge using semantic and keyword search
* 📁 Adding new content (write, copy, append)
* 🧹 Maintaining the structure (list, move, delete)

All tools are built to be used by agents (e.g., LLMs) through the `smolagents.Tool` interface, with schema-aware I/O and semantic index updates via `RepoIndexer`.

---

## Installation

To use these tools, first install the required packages:

```bash
pip install faiss-cpu numpy openai watchdog
```

## 📁 Structure

```
kb_repo_management/
├── kb_repo_addition_tools.py       # Add new files/content
├── kb_repo_maintanence_tools.py    # Manage/restructure files/folders
├── kb_repo_retrieval_tools.py      # Search and retrieve knowledge
├── repo_indexer.py                 # Index and semantic search backend
```

---

## 🔍 Retrieval Tools

### 1. `semantic_search_knowledge_base`

**Purpose:** Retrieve relevant content based on a semantic query.

```json
{
  "query": "shortest path algorithm"
}
```

➡️ Returns relevant file paths and content snippets using the semantic index.

---

### 2. `keyword_search_knowledge_base`

**Purpose:** Search for exact keywords in a file or folder.

```json
{
  "path": "textual_knowledge/algorithms",
  "keyword": "simplex",
  "context_lines": 2
}
```

➡️ Returns matching lines with file name, line number, and surrounding context.

---

## ➕ Addition Tools

### 3. `write_to_knowledge_base`

**Purpose:** Create a new file with the given content.

```json
{
  "content": "# Notes on Duality\n...",
  "destination_path": "textual_knowledge/notes/duality.md",
  "overwrite": false
}
```

➡️ Creates a new file or appends a number to avoid overwriting. Updates semantic index.

---

### 4. `copy_to_knowledge_base`

**Purpose:** Copy file/folder from working directory to knowledge base.

```json
{
  "source_path": "drafts/notes.txt",
  "destination_path": "textual_knowledge/notes/final_notes.txt",
  "overwrite": true
}
```

➡️ Merges folders or replaces files if `overwrite=true`. Indexes the copied content.

---

### 5. `append_to_knowledge_base_file`

**Purpose:** Insert new content at a specific location in a text file.

```json
{
  "target_file": "textual_knowledge/algorithms/simplex.md",
  "new_content": "This is a useful implementation tip.",
  "insert_mode": "after",
  "match_string": "# Simplex Method"
}
```

➡️ Appends or inserts based on mode and target line. Reindexes the file.

---

## 🔧 Maintenance Tools

### 6. `list_knowledge_base_directory`

**Purpose:** Explore folder structure.

```json
{
  "directory": "code_implementation/linear"
}
```

➡️ Lists files/subfolders.

---

### 7. `see_knowledge_base_file`

**Purpose:** View plain text file content.

```json
{
  "filename": "code_implementation/linear/simplex.py"
}
```

➡️ Returns file lines with line numbers.

---

### 8. `move_or_rename_in_knowledge_base`

**Purpose:** Move or rename file/folder.

```json
{
  "source_path": "textual_knowledge/notes/draft.md",
  "destination_path": "textual_knowledge/notes/final.md",
  "overwrite": true
}
```

➡️ Moves file or folder. Replaces if `overwrite=true`.

---

### 9. `delete_from_knowledge_base`

**Purpose:** Delete a file or folder.

```json
{
  "target_path": "textual_knowledge/obsolete_notes.md"
}
```

➡️ Deletes the file/folder from the knowledge base.

---

## 🧠 How It Works

* All tools assume a `RepoIndexer` is initialized with the root of your knowledge base.
* Any modified file is passed to `repo_indexer.update_file()` to update the semantic search index.
* Tools are structured for use with LLM agents, but can also be used programmatically.

---

## 🧪 Testing

Run test cases for each tool category:

```bash
python general_tools_tests/run_tests.py --tool kb_repo_management
```

> All test artifacts are created in `general_tools_tests/kb_repo_management/temp_data`.