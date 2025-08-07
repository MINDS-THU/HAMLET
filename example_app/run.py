import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from smolagents import LiteLLMModel, CodeAgent, ToolCallingAgent
from src.agent_conversation_ui import GradioUI
from src.monitoring import AgentLogger, LogLevel
from default_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    CreateFileWithContent,
)
from default_tools.kb_repo_management.repo_indexer import RepoIndexer
from default_tools.kb_repo_management.kb_repo_retrieval_tools import (
    SemanticSearchKnowledgeBase,
    KeywordSearchKnowledgeBase,
    CopyFromKnowledgeBase,
)
from default_tools.kb_repo_management.kb_repo_maintanence_tools import (
    ListKnowledgeBaseDirectory,
    SeeKnowledgeBaseFile,
    MoveOrRenameInKnowledgeBase,
    DeleteFromKnowledgeBase
)
from default_tools.kb_repo_management.kb_repo_addition_tools import (
    WriteToKnowledgeBase,
    CopyToKnowledgeBase,
    AppendToKnowledgeBaseFile,
)
from default_tools.visual_qa.visual_qa import visualizer
from default_tools.open_deep_search.ods_tool import OpenDeepSearchTool
from default_tools.md2pdf.md2pdf_tool import CompileMarkdownToPDF
import tempfile
import time

# Load environment variables
load_dotenv(override=True)

def create_example_agent(model_id="gpt-4.1", working_directory="working_dir", persistent_storage="persistent_storage", index_dir="index_dir", signature=None):

    ### Set up the model ###
    # here we use LiteLLMModel.
    # Alternatively, you can use InferenceClientModel, VLLMModel or TransformersModel depending on your chosen LLM model backend
    model = LiteLLMModel(model_id=model_id)

    ### Set up the tools ###
    # tools for working with the local working directory
    working_directory_file_editing_tools = [
        ListDir(working_directory),
        SeeFile(working_directory),
        ModifyFile(working_directory),
        CreateFileWithContent(working_directory),
    ]

    # tools for doing websearch
    search_tool = OpenDeepSearchTool(model_name=model_id, reranker="jina")
    if not search_tool.is_initialized:
        search_tool.setup()
    search_tools = [search_tool]

    # tools for working with the knowledge base (a persistent storage used for RAG)
    # Here we only use it to retrieve logs from previous interactions, which acts as a long-term memory for the agent
    # You can also use it to store and retrieve files, code snippets, papers, etc.
    if persistent_storage is not None and index_dir is not None:
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required but not set")
            
        # Instantiate indexer (auto sync + live updates)
        idx = RepoIndexer(
            persistent_storage,
            watch=True,
            index_dir=Path(index_dir),
            embed_model="text-embedding-3-small",
            openai_api_key=openai_api_key,
        )
        knowledge_base_retrieval_tools = [
            ListKnowledgeBaseDirectory(idx),
            SeeKnowledgeBaseFile(idx),
            SemanticSearchKnowledgeBase(idx),
        ]
        knowledge_base_update_tools = [
            WriteToKnowledgeBase(idx),
            CopyToKnowledgeBase(idx, working_directory),
        ]
    else:
        knowledge_base_retrieval_tools = []
        knowledge_base_update_tools = []

    # tool for reading image
    visual_qa_tools = [visualizer]

    ### Set up the agent ###
    app_name="example_app"
    # Here we configure the logger to save the agent's log to a txt file in the persistent storage
    mananger_logger = AgentLogger(
            level=LogLevel.DEBUG,
            save_to_file=os.path.join(persistent_storage, f"manager_agent_log_{signature}.txt"),
            name=app_name
        )
    # manager agent is responsible for directly talking with user and call sub-agents to complete user tasks
    manager_agent = CodeAgent(
        tools=working_directory_file_editing_tools+search_tools+knowledge_base_retrieval_tools+knowledge_base_update_tools+visual_qa_tools,
        model=model,
        managed_agents=[],
        planning_interval=None,
        max_steps=50,
        logger=mananger_logger,
        name=app_name,
        description="This is an example LLM agent application showcasing the functionality of HAMLET."
    )
    mananger_logger.visualize_agent_tree(manager_agent)
    return manager_agent

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the Example Agent")
    argparser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4.1",
        help="The ID of the model to use for the agent.",
    )
    argparser.add_argument(
        "--mode",
        type=str,
        default="gradio",
        choices=["gradio", "cli"],
        help="The mode to run the agent in. 'gradio' for web interface, 'cli' for command line interface.",
    )
    argparser.add_argument(
        "--working_directory",
        type=str,
        default=None,
        help="The directory where the agent will store its working files.",
    )
    argparser.add_argument(
        "--persistent_storage",
        type=str,
        default=None,
        help="A structured directory that contains the persistent files, e.g. code snippets, papers, and other resources.",
    )
    argparser.add_argument(
        "--index_dir",
        type=str,
        default=None,
        help="The directory where the vector store index will be stored.",
    )
    args = argparser.parse_args()

    # Ensure the base temp_files directory exists
    base_temp_dir = "example_app/working_dirs"
    Path(base_temp_dir).mkdir(parents=True, exist_ok=True)

    # Set the save directory to a default if not provided
    if args.working_directory is None:
        args.working_directory = tempfile.mkdtemp(dir=base_temp_dir, prefix="working_directory_")
    if args.persistent_storage is None:
        args.persistent_storage = "example_app/persistent_storage"
    if args.index_dir is None:
        args.index_dir = "example_app/index_dir"
    # Ensure the persistent storage and index directories exist
    Path(args.persistent_storage).mkdir(parents=True, exist_ok=True)
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    
    # create a date time signature
    date_time_signature = time.strftime("%Y%m%d_%H%M%S")
    
    # Create the agent
    manager_agent = create_example_agent(
        model_id = args.model_id,
        working_directory = args.working_directory,
        persistent_storage = args.persistent_storage,
        index_dir = args.index_dir,
        signature = date_time_signature
    )

    if args.mode == "cli":
        # Run the agent in CLI mode
        while True:
            try:
                manager_agent.run("Based on the conversation so far, talk with the user to understand the user's task and complete the task.", reset=False)
                print("Agent finished running. Waiting for next command...")
                print("Press Ctrl+C to exit.")
            except KeyboardInterrupt:
                print("Exiting...")
                break
    else:
        # Run the agent in Gradio mode
        print("Launching Gradio UI...")
        GradioUI(agent=manager_agent, file_upload_folder=args.working_directory).launch(share=False)