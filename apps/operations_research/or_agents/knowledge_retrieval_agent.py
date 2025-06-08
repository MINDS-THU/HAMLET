from smolagents import LiteLLMModel, ToolCallingAgent, CodeAgent, GradioUI
from smolagents.monitoring import LogLevel
from general_tools.kb_repo_management.repo_indexer import RepoIndexer
from general_tools.kb_repo_management.kb_repo_retrieval_tools import (
    SemanticSearchKnowledgeBase,
    KeywordSearchKnowledgeBase,
    CopyFromKnowledgeBase,
)
from general_tools.kb_repo_management.kb_repo_maintanence_tools import (
    ListKnowledgeBaseDirectory,
    SeeKnowledgeBaseFile,
)

import yaml
import importlib
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

from dotenv import load_dotenv
load_dotenv(override=True)

description = """
**Purpose**:  
Acts as a comprehensive knowledge retrieval agent for operations research and optimization. It can search, retrieve, and copy information from a structured local knowledge base (including code, documentation, and best practices).

**When to use**:
- To look up the meaning of a term, concept, or acronym in operations research.
- To find code examples, solution procedures, or best practices for optimization problems.
- To explore, view, or copy files and folders from the knowledge base to your working directory.
- To perform semantic or keyword-based searches across all knowledge base files.

**Capabilities**:
- **ListKnowledgeBaseDirectory**: List all files and folders inside a directory in the knowledge base.
- **SeeKnowledgeBaseFile**: View the content of a plain text file in the knowledge base.
- **CopyFromKnowledgeBase**: Copy files or folders from the knowledge base to the working directory, with conflict resolution.
- **KeywordSearchKnowledgeBase**: Search for keywords in files or folders, with context lines and file/line references.
- **SemanticSearchKnowledgeBase**: Perform semantic search to find the most relevant files, code snippets, or documentation for a query.

**Examples**:
- List all available algorithms in the knowledge base.
- View the content of a specific Python or Markdown file.
- Copy a code example from the knowledge base to your working directory.
- Search for all occurrences of "branch and bound" in the knowledge base, with context.
- Find the most relevant code or documentation for "network flow optimization".
"""

def create_knowledge_retrieval_agent(idx, model_id="gpt-4.1", working_directory="working_directory", max_steps=10, verbosity_level=LogLevel.INFO):
    
    # Create the knowledge base retrieval agent
    kb_retrieval_tools = [
        ListKnowledgeBaseDirectory(idx),
        SeeKnowledgeBaseFile(idx),
        CopyFromKnowledgeBase(idx, working_directory),
        KeywordSearchKnowledgeBase(idx),
        SemanticSearchKnowledgeBase(idx),
    ]


    # Load the prompt template
    knowledge_retrieval_prompt_template = yaml.safe_load(
                importlib.resources.files("apps.operations_research.or_agents.prompts").joinpath("knowledge_retrieval.yaml").read_text(encoding="utf-8")
            )

    knowledge_retrieval_agent = ToolCallingAgent(
        tools=kb_retrieval_tools,
        model=LiteLLMModel(model_id=model_id),
        prompt_templates=knowledge_retrieval_prompt_template,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
        name="knowledge_retrieval_agent",
        description=description,
    )
    # knowledge_retrieval_agent.prompt_templates["managed_agent"]["task"] += """
    # You are a knowledge retrieval specialist for operations research and optimization. You have access to the following tools:

    # - **ListKnowledgeBaseDirectory**: List all files and folders in a specified directory of the knowledge base.
    # - **SeeKnowledgeBaseFile**: View the content of a plain text file in the knowledge base.
    # - **CopyFromKnowledgeBase**: Copy files or folders from the knowledge base to the working directory, handling name conflicts as needed.
    # - **KeywordSearchKnowledgeBase**: Search for a keyword in a file or recursively in all files within a folder, returning matches with file names, line numbers, and context.
    # - **SemanticSearchKnowledgeBase**: Perform a semantic search to find the most relevant files, code snippets, or documentation for a query.

    # **How to work:**
    # 1. Carefully interpret the manager's query as a real-world question, not just keywords.
    # 2. Choose the most appropriate tool(s) for the task:
    # - Use **semantic search** for broad or conceptual queries.
    # - Use **keyword search** for specific terms or code patterns.
    # - Use **listing** and **viewing** tools to explore or display file contents.
    # - Use **copy** to transfer relevant files to the working directory.
    # 3. If the query is ambiguous or lacks detail, use `final_answer("Your clarification question")` to request more information from the manager.
    # 4. When searching, iterate as needed: refine search terms, explore related files, or combine results from multiple tools.
    # 5. Always provide clear, concise, and well-organized results. Include file names, paths, and context where relevant.
    # 6. Prioritize high-quality, relevant sources—prefer official documentation, well-commented code, and authoritative references.
    # 7. If copying files, ensure you do not overwrite existing files unless explicitly instructed; otherwise, add a numeric suffix to avoid conflicts.

    # Your goal is to help the manager efficiently find, understand, and utilize knowledge from the knowledge base.
    # """

    return knowledge_retrieval_agent

if __name__ == "__main__":
    # Step 1: Create a simple knowledge base with several code examples
    from pathlib import Path
    import shutil

    knowledge_base_dir = Path("demo_knowledge_base")
    working_dir = Path("demo_working_directory")
    # Clean up any previous runs
    # if knowledge_base_dir.exists():
    #     shutil.rmtree(knowledge_base_dir)
    if working_dir.exists():
        shutil.rmtree(working_dir)
    knowledge_base_dir.mkdir(parents=True)
    working_dir.mkdir(parents=True)

    # Add several Python files to the knowledge base
    code_dir = knowledge_base_dir / "examples"
    code_dir.mkdir()

    # The correct file for the greeting message
    (code_dir / "hello.py").write_text(
        'def say_hello():\n    print("Hello, world!")\n'
    )

    # Distractor: prints a number
    (code_dir / "print_number.py").write_text(
        'def print_number(n):\n    print(f"Number: {n}")\n'
    )

    # Distractor: unrelated utility
    (code_dir / "math_utils.py").write_text(
        'def add(a, b):\n    return a + b\n'
    )

    # Distractor: documentation file
    (code_dir / "README.md").write_text(
        "# Examples\n\nThis folder contains example Python scripts for various purposes."
    )

    # Instantiate indexer (auto sync + live updates) ---------------------------
    idx = RepoIndexer(
        str(knowledge_base_dir),
        watch=False,
        index_dir=None,
        embed_model="text-embedding-3-small",
    )
    print("[demo] Initial index built.\n")

    # Step 2: Create the agent
    agent = create_knowledge_retrieval_agent(
        idx,
        model_id="gpt-4.1",
        working_directory=str(working_dir),
    )

    # Step 3: Ask the agent to use semantic search to find the code
    print("\n--- Asking agent to find code that prints a greeting (semantic search) ---\n")
    agent.run("Find a code example in the knowledge base that prints a greeting message.")

    # Step 4: Ask the agent to copy the file to the working directory
    print("\n--- Asking agent to copy the relevant file to the working directory ---\n")
    agent.run("Copy the file that contains the greeting function to my working directory as hello.py.")

    # Step 5: Show the result in the working directory
    copied_file = working_dir / "hello.py"
    if copied_file.exists():
        print("\n--- Content of copied file ---\n")
        print(copied_file.read_text())
    else:
        print("\nFile was not copied.")

    # Clean up at the very end
    if knowledge_base_dir.exists():
        shutil.rmtree(knowledge_base_dir)
    if working_dir.exists():
        shutil.rmtree(working_dir)