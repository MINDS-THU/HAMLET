from smolagents import LiteLLMModel, GradioUI, ToolCallingAgent, CodeAgent
from smolagents.monitoring import LogLevel
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import argparse
import tempfile
import os
import yaml
import importlib
# import optimizer agents
from .or_agents.mathematical_optimizer_agent import create_mathematical_optimizer_agent
from .or_agents.combinatorial_optimizer_agent import create_combinatorial_optimizer_agent
from .or_agents.metaheuristic_optimizer_agent import create_metaheuristic_optimizer_agent
from .or_agents.general_optimizer_agent import create_general_optimizer_agent

# import knowledge management agents
from .or_agents.knowledge_retrieval_agent import create_knowledge_retrieval_agent
from .or_agents.knowledge_curation_agent import create_knowledge_curation_agent
from .or_agents.web_browsing_agent import create_web_browsing_agent
# import tools available to the manager agent
from general_tools.talk_to_user.talk_to_user_tool import TalkToUser
from general_tools.kb_repo_management.repo_indexer import RepoIndexer
from general_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    DeleteFileOrFolder,
    CreateFileWithContent,
)

def create_manager_agent(model_id="gpt-4.1", knowledge_base_directory="apps/operations_research/or_knowledge_base", index_dir="apps/operations_research/or_vector_store", working_directory=None):
    # Define the working directory
    if working_directory is None:
        # Use a temporary directory if not specified
        working_directory = tempfile.mkdtemp()
    else:
        # Create the working directory if it doesn't exist
        Path(working_directory).mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {working_directory}")

    # Create the web-browsing agent
    downloads_folder = Path(working_directory) / "downloads"
    downloads_folder = str(downloads_folder)  # Ensure downloads_folder is a string
    web_browsing_agent = create_web_browsing_agent(model_id=model_id, downloads_folder=downloads_folder)

    # Define the knowledge base directory
    # create the knowledge base directory if it doesn't exist
    Path(knowledge_base_directory).mkdir(parents=True, exist_ok=True)
    print(f"Knowledge base directory: {knowledge_base_directory}")

    # Instantiate indexer (auto sync + live updates) ---------------------------
    idx = RepoIndexer(
        knowledge_base_directory,
        watch=False,
        index_dir=Path(index_dir),
        embed_model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    print("[demo] Initial index built.\n")

    # Create the knowledge retrieval agent
    knowledge_retrieval_agent = create_knowledge_retrieval_agent(
        idx,
        working_directory=working_directory,
        model_id=model_id,
        verbosity_level=LogLevel.DEBUG
    )
    # Create the knowledge curation agent
    knowledge_curation_agent = create_knowledge_curation_agent(
        idx,
        working_directory=working_directory,
        model_id=model_id,
        verbosity_level=LogLevel.DEBUG
    )
    # Create the mathematical optimizer agent
    mathematical_optimizer_agent = create_mathematical_optimizer_agent(
        model_id=model_id,
        managed_agents=[web_browsing_agent, knowledge_curation_agent],
        working_directory=working_directory,
        verbosity_level=LogLevel.DEBUG
    )
    # Create the combinatorial optimizer agent
    combinatorial_optimizer_agent = create_combinatorial_optimizer_agent(
        model_id=model_id,
        managed_agents=[web_browsing_agent, knowledge_curation_agent],
        working_directory=working_directory,
        verbosity_level=LogLevel.DEBUG
    )
    # Create the metaheuristic optimizer agent
    metaheuristic_optimizer_agent = create_metaheuristic_optimizer_agent(
        model_id=model_id,
        managed_agents=[web_browsing_agent, knowledge_curation_agent],
        working_directory=working_directory,
        verbosity_level=LogLevel.DEBUG
    )
    # Create the general optimizer agent
    general_optimizer_agent = create_general_optimizer_agent(
        model_id=model_id,
        managed_agents=[web_browsing_agent, knowledge_curation_agent],
        working_directory=working_directory,
        verbosity_level=LogLevel.DEBUG
    )
    # Load the prompt template
    manager_prompt_template = yaml.safe_load(
                importlib.resources.files("apps.operations_research.or_agents.prompts").joinpath("manager.yaml").read_text(encoding="utf-8")
            )

    # Create the manager agent
    manager_agent = CodeAgent(
        tools=[
            TalkToUser(),
            ListDir(working_directory),
            SeeFile(working_directory),
            ModifyFile(working_directory),
            DeleteFileOrFolder(working_directory),
            CreateFileWithContent(working_directory),
            ],
        managed_agents=[
            web_browsing_agent,
            knowledge_retrieval_agent,
            # knowledge_curation_agent,
            general_optimizer_agent,
            mathematical_optimizer_agent,
            combinatorial_optimizer_agent,
            metaheuristic_optimizer_agent,
            ],
        prompt_templates=manager_prompt_template,
        additional_authorized_imports=['numpy', 'numpy.*', 'random', 'random.*', 'math', 'math.*'],
        model=LiteLLMModel(model_id=model_id),
        name="or_agent",
        description="An agent that can solve operations research problems.",
        verbosity_level=LogLevel.DEBUG,
    )
    return manager_agent

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the Operations Research Agent")
    argparser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4.1",
        help="The ID of the model to use for the agent.",
    )
    argparser.add_argument(
        "--working_directory",
        type=str,
        default=None,
        help="The directory where the agent will store its working files.",
    )
    argparser.add_argument(
        "--knowledge_base_directory",
        type=str,
        default=None,
        help="A structured directory that contains the knowledge base files, e.g. code snippets, papers, and other resources.",
    )
    argparser.add_argument(
        "--index_dir",
        type=str,
        default=None,
        help="The directory where the vector store index will be stored.",
    )
    argparser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["gradio", "cli"],
        help="The mode to run the agent in. 'gradio' for web interface, 'cli' for command line interface.",
    )
    args = argparser.parse_args()
    # Ensure the base temp_files directory exists
    base_temp_dir = "apps/operations_research/temp_files"
    Path(base_temp_dir).mkdir(parents=True, exist_ok=True)

    if args.working_directory is None:
        args.working_directory = tempfile.mkdtemp(dir="apps/operations_research/temp_files", prefix="working_directory_")
    if args.knowledge_base_directory is None:
        args.knowledge_base_directory = tempfile.mkdtemp(dir="apps/operations_research/temp_files", prefix="knowledge_base_")
    if args.index_dir is None:
        args.index_dir = tempfile.mkdtemp(dir="apps/operations_research/temp_files", prefix="index_dir_")
    
    # Create the agent
    manager_agent = create_manager_agent(
        model_id=args.model_id, 
        knowledge_base_directory=args.knowledge_base_directory,
        index_dir=args.index_dir,
        working_directory=args.working_directory
        )
    
    if args.mode == "cli":
        # Run the agent in CLI mode
        while True:
            try:
                manager_agent.run("Based on the conversation so far, talk with the user.", reset=False)
                print("Agent finished running. Waiting for next command...")
                print("Press Ctrl+C to exit.")
            except KeyboardInterrupt:
                print("Exiting...")
                break
    else:
        # Run the agent in Gradio mode
        print("Launching Gradio UI...")
        GradioUI(agent=manager_agent, file_upload_folder=args.working_directory).launch()