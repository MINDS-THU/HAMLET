from smolagents import LiteLLMModel, ToolCallingAgent
from smolagents.monitoring import LogLevel
from general_tools.kb_repo_management.repo_indexer import RepoIndexer
from general_tools.kb_repo_management.kb_repo_addition_tools import (
    WriteToKnowledgeBase,
    CopyToKnowledgeBase,
    AppendToKnowledgeBaseFile,
)
from general_tools.kb_repo_management.kb_repo_maintanence_tools import (
    ListKnowledgeBaseDirectory,
    SeeKnowledgeBaseFile,
    MoveOrRenameInKnowledgeBase,
    DeleteFromKnowledgeBase,
)
from general_tools.kb_repo_management.kb_repo_retrieval_tools import (
    SemanticSearchKnowledgeBase,
    KeywordSearchKnowledgeBase,
)
import yaml
import importlib
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv(override=True)

description = """
**Purpose**:  
The knowledge_curation_agent is responsible for storing, organizing, and maintaining new knowledge in the operations research knowledge base. Use this agent to add, update, move, rename, or delete files and folders within the knowledge base.

**Important:**  
Do **not** use this agent to search for or retrieve knowledge. For retrieval, use the knowledge_retrieval_agent, which can search, view, and copy knowledge base content but cannot modify it.

**How to use**:
- Use this agent when you want to store new knowledge, update existing content, or reorganize the knowledge base.
- When saving files or folders from the working directory, provide the source path and a description of the content or its intended purpose. The agent will decide the best location and naming in the knowledge base.
- The agent cannot see or manage the working directory directly; it only operates on the knowledge base and only with the information you provide.

**Capabilities**:
- Create, overwrite, or append to files in the knowledge base.
- Move, rename, or delete files and folders in the knowledge base.
- Perform semantic or keyword search within the knowledge base to avoid duplication or for organization.
- List and view files/folders in the knowledge base.
"""

def create_knowledge_curation_agent(
    idx,
    model_id="gpt-4.1",
    working_directory="working_directory",
    max_steps=10,
    verbosity_level=LogLevel.INFO,
):

    kb_curation_tools = [
        WriteToKnowledgeBase(idx),
        CopyToKnowledgeBase(idx, working_directory),
        AppendToKnowledgeBaseFile(idx),
        ListKnowledgeBaseDirectory(idx),
        SeeKnowledgeBaseFile(idx),
        MoveOrRenameInKnowledgeBase(idx),
        DeleteFromKnowledgeBase(idx),
        SemanticSearchKnowledgeBase(idx),
        KeywordSearchKnowledgeBase(idx),
    ]

    # Load the prompt template
    knowledge_curation_prompt_template = yaml.safe_load(
                importlib.resources.files("apps.operations_research.or_agents.prompts").joinpath("knowledge_curation.yaml").read_text(encoding="utf-8")
            )


    agent = ToolCallingAgent(
        tools=kb_curation_tools,
        model=LiteLLMModel(model_id=model_id),
        prompt_templates=knowledge_curation_prompt_template,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
        name="knowledge_curation_agent",
        description=description,
    )

    # agent.prompt_templates["managed_agent"]["task"] += """
    # You are a knowledge curation and maintenance specialist for operations research and optimization.
    # You receive explicit instructions from a manager agent, including source paths or new knowledge pieces.
    # You can only see and manage files and folders within the knowledge base.

    # Your responsibilities:
    # - Store new knowledge (text, code, or documentation) from the working directory in the most appropriate location and with a clear, descriptive name.
    # - Append to or update existing files as directed.
    # - Move, rename, or delete files and folders to maintain a logical, organized structure.
    # - Use semantic and keyword search to avoid duplicates and merge related content when adding or organizing knowledge.
    # - List or display files/folders as requested.

    # Best practices:
    # - Maintain a clear folder structure, such as:
    #     - `algorithms/`
    #     - `code_examples/`
    #     - `textual_knowledge/`
    #     - `glossary/`
    # - Group related files together (e.g., all network flow algorithms in `algorithms/network_flow/`).
    # - Use descriptive, consistent names for files and folders.
    # - Before adding new content, check for existing similar files and merge or update as appropriate.
    # - Keep the knowledge base concise, well-organized, and easy to navigate.
    # - Make all organizational decisions (destination, naming, structure) based on the content and its purpose, not on manager-supplied paths.

    # **Do not** perform knowledge retrieval or answer content queries. If asked to retrieve or search for knowledge, respond that this is the responsibility of the knowledge_retrieval_agent.

    # **Important formatting rule:**  
    # - All generated code, texts, and documentation must use only standard ASCII characters.  
    # - Do not use special Unicode symbols (such as arrows like `⇒`, smart quotes, or other non-ASCII characters).  
    # - Use plain ASCII equivalents (e.g., use `<=`, `>=`, `->`, `=>` instead of Unicode arrows).
    # """

    return agent

if __name__ == "__main__":
    from pathlib import Path
    import shutil

    knowledge_base_dir = Path("demo_knowledge_base")
    working_dir = Path("demo_working_directory")
    # Clean up any previous runs
    if knowledge_base_dir.exists():
        shutil.rmtree(knowledge_base_dir)
    if working_dir.exists():
        shutil.rmtree(working_dir)
    knowledge_base_dir.mkdir(parents=True)
    working_dir.mkdir(parents=True)

    # Prepare a file in the working directory to test copy functionality
    (working_dir / "kruskal.py").write_text(
        'def kruskal(graph):\n    """Minimum spanning tree algorithm"""\n    pass\n'
    )

    # Instantiate indexer (auto sync + live updates) ---------------------------
    idx = RepoIndexer(
        str(knowledge_base_dir),
        watch=False,
        index_dir=None,
        embed_model="text-embedding-3-small",
    )
    print("[demo] Initial index built.\n")


    # Step 1: Create the agent
    agent = create_knowledge_curation_agent(
        idx,
        model_id="gpt-4.1",
        working_directory=str(working_dir),
    )

    # Step 2: Add a new algorithm description (text)
    print("\n--- Adding a new algorithm description (text) ---\n")
    agent.run(
        "Create a new file 'algorithms/greedy.md' in the knowledge base with a description of the greedy algorithm."
    )

    # Step 3: Add a new code file
    print("\n--- Adding a new code file ---\n")
    agent.run(
        "Create a new file 'algorithms/greedy.py' in the knowledge base with a Python implementation of a greedy algorithm."
    )

    # Step 4: Append to the markdown file
    print("\n--- Appending to the markdown file ---\n")
    agent.run(
        "Append the following to 'algorithms/greedy.md':\n\nGreedy algorithms make the locally optimal choice at each step."
    )

    # Step 5: Move the markdown file to a new folder
    print("\n--- Moving the markdown file to a new folder ---\n")
    agent.run(
        "Move 'algorithms/greedy.md' to 'textual_knowledge/algorithms/greedy.md'."
    )

    # Step 6: List files in the new folder
    print("\n--- Listing the 'textual_knowledge/algorithms' directory ---\n")
    agent.run(
        "List all files in the 'textual_knowledge/algorithms' directory."
    )

    # Step 7: View the content of the moved markdown file
    print("\n--- Viewing the content of the moved markdown file ---\n")
    agent.run(
        "Show the content of 'textual_knowledge/algorithms/greedy.md'."
    )

    # Step 8: Copy a code file from the working directory into the knowledge base
    print("\n--- Copying a code file from working directory into the knowledge base ---\n")
    agent.run(
        "Copy 'kruskal.py' from the working directory into 'algorithms/kruskal.py' in the knowledge base."
    )

    # Step 9: Semantic search for 'minimum spanning tree'
    print("\n--- Semantic search for 'minimum spanning tree' ---\n")
    agent.run(
        "Find files in the knowledge base related to 'minimum spanning tree'."
    )

    # Step 10: Delete the greedy algorithm code file
    print("\n--- Deleting the greedy algorithm code file ---\n")
    agent.run(
        "Delete the file 'algorithms/greedy.py' from the knowledge base."
    )

    # Step 11: List files in 'algorithms' to confirm deletion
    print("\n--- Listing files in 'algorithms' to confirm deletion ---\n")
    agent.run(
        "List all files in the 'algorithms' directory."
    )

    # Clean up at the very end
    if knowledge_base_dir.exists():
        shutil.rmtree(knowledge_base_dir)
    if working_dir.exists():
        shutil.rmtree(working_dir)