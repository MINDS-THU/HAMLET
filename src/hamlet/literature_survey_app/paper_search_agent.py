import json
from pydantic import BaseModel, Field
import yaml
import importlib.resources
import os
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv(override=True)

from src.hamlet.core.agents import CodeAgent
from src.hamlet.core.monitoring import LogLevel
from src.hamlet.core.utils import get_fields_info
from src.hamlet.core.models import LiteLLMModel

# Tools: web search, get paper, and file editing
from src.hamlet.tools.open_deep_search.ods_tool import OpenDeepSearchTool
from src.hamlet.tools.get_paper_from_url.get_paper_from_url_tool import GetPaperFromURL
from src.hamlet.tools.file_editing.file_editing_tools import (
    ListDir,
    SeeTextFile,
    DeleteFileOrFolder,
    SearchKeyword,
)


def create_paper_search_agent(
    working_directory: str,
    model: LiteLLMModel,
    max_steps: int = 5,
    verbosity_level: LogLevel = LogLevel.INFO,
    output_schema: Optional[BaseModel] = None,
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
        # max_queries=6, 
        # timeout=60,
        # quick_mode=True,
        # max_results=5,
        model_name="gpt-5-mini", 
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
        importlib.resources.files("src.hamlet.literature_survey_app.prompts").joinpath("paper_search_agent.yaml").read_text(encoding="utf-8")
    )

    agent = CodeAgent(
        tools=tools,
        model=model,
        prompt_templates=prompt_templates,
        output_schema=output_schema,
        max_steps=max_steps,
        # planning_interval=max_steps+1,
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
    model = LiteLLMModel(model_id="gpt-5-mini", api_base=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))

    class SingleOutput(BaseModel):
        title: str = Field(..., description="The title of the paper")
        year: Optional[int] = Field(None, description="The publication year of the paper")
        authors: Optional[str] = Field(None, description="The authors of the paper")
        short_summary: str = Field(..., description="A summary of the paper")
        url: str = Field(..., description="The URL of the paper")
        save_path: str = Field(..., description="The file path where the paper is saved")

    class Output(BaseModel):
        summary: str = Field(..., description="A summary of the search results and papers found")
        papers: List[SingleOutput] = Field(..., description="A list of papers found and summarized by the agent")
        
    # print("Fields info for SingleOutput:\n", '\n'.join(get_fields_info(Output)))

    agent = create_paper_search_agent(
        working_directory=wd,
        model=model,
        max_steps=3,
        verbosity_level=LogLevel.DEBUG,
        output_schema=Output,
        )
    output = agent.run("I am interested in recent papers on large language models for robotics.")
    print(type(output))
    # full_results = agent.run("I am interested in recent papers on large language models for robotics.", return_full_result=True)

    # with open("src/hamlet/literature_survey_app/paper_search_agent_result_code.json", "w") as f:
    #     json.dump(full_results.steps, f, indent=2)

    """
    ssh -vvv -N -R 7890:localhost:7890 -p 2228 lijinbo@166.111.59.11
    export http_proxy=http://127.0.0.1:7890
    export https_proxy=http://127.0.0.1:7890

    uv sync
    uv sync --extra tools --extra train
    uv run python -m src.hamlet.literature_survey_app.paper_search_agent
    """