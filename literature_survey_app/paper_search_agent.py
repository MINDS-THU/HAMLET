from smolagents import ToolCallingAgent, CodeAgent, Tool
from smolagents.monitoring import LogLevel
import yaml
import importlib.resources
import os
from typing import Optional, List

# Tools: web search, get paper, and file editing
from default_tools.open_deep_search.ods_tool import OpenDeepSearchTool
from default_tools.get_paper_from_url.get_paper_from_url_tool import GetPaperFromURL
from default_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeTextFile,
    DeleteFileOrFolder,
    SearchKeyword,
)
from src.models import LiteLLMModel
from dotenv import load_dotenv
load_dotenv(override=True)

def create_paper_search_agent(
    working_directory: str,
    model: LiteLLMModel,
    max_steps: int = 5,
    verbosity_level: LogLevel = LogLevel.INFO,
) -> CodeAgent:
    """
    Create an agent that can:
    - Ask clarifying questions to understand the user's paper search intent.
    - Run parallel web searches for multiple queries.
    - Propose candidate paper URLs and, upon confirmation, fetch and save them as Markdown.
    - Remove irrelevant Markdown files using deletion tools.
    - Iterate until satisfied.
    """
    # create working_directory if not exists
    os.makedirs(working_directory, exist_ok=True)

    # Instantiate tools
    web_search = OpenDeepSearchTool(
        max_queries=6, 
        timeout=60,
        quick_mode=True,
        max_results=5,
        model_name="gpt-5", 
        reranker="jina")
    # Ensure underlying search agent is ready (setup defines self.search_tool)
    if hasattr(web_search, "setup"):
        web_search.setup()

    get_papers = GetPaperFromURL(working_directory)
    list_dir = ListDir(working_directory)
    see_text = SeeTextFile(working_directory)
    delete_path = DeleteFileOrFolder(working_directory)
    search_keyword = SearchKeyword(working_directory)

    tools = [
        web_search,
        get_papers,
        list_dir,
        see_text,
        delete_path,
        search_keyword,
    ]

    # Load the prompt template for paper search agent
    prompt_templates = yaml.safe_load(
        importlib.resources.files("literature_survey_app.prompts").joinpath("paper_search_agent.yaml").read_text(encoding="utf-8")
    )

    agent = CodeAgent(
        tools=tools,
        model=model,
        prompt_templates=prompt_templates,
        max_steps=max_steps,
        planning_interval=max_steps+1,
        verbosity_level=verbosity_level,
        name="paper_search_agent",
        description=(
            "Interactive paper search agent: builds multiple queries, returns URLs, fetches papers as Markdown, "
            "summarizes content first, then asks the user for pruning and additional directions."
        ),
    )
    return agent

if __name__ == "__main__":
    # Minimal interactive run: ask for topic, then let agent proceed.
    # create working directory, if exists, clear it
    wd = os.path.join(os.getcwd(), "paper_search_working_dir")
    if os.path.exists(wd):
        import shutil
        shutil.rmtree(wd)
    os.makedirs(wd, exist_ok=True)
    model = LiteLLMModel(model_id="gpt-5", api_base=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    agent = create_paper_search_agent(
        working_directory=wd,
        model=model,
        max_steps=5,
        verbosity_level=LogLevel.DEBUG,
        )
    agent.run("I am interested in papers that uses LLM to automate or assist the construction of simulation, especially discrete-event simulation.")