import os
from dotenv import load_dotenv
from literature_survey_app.text_inspector_tool import TextInspectorTool
from default_tools.text_web_browser.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from default_tools.visual_qa.visual_qa import Visualizer
from smolagents import (
    GoogleSearchTool,
    ToolCallingAgent,
    monitoring,
)
from src.models import LiteLLMModel
load_dotenv()

def create_web_browsing_agent(model, downloads_folder="downloads_folder", max_steps=10):
    """
    Create an agent that will browse the internet to support solver agents.
    Args:
        model_id (str): The ID of the model to use.
        downloads_folder (str): The folder where downloaded files will be stored.
    Returns:
        CodeAgent: The configured web browsing agent.
    """
    # Ensure downloads_folder is a string and not empty
    downloads_folder = str(downloads_folder)
    if not downloads_folder.strip():
        raise ValueError("downloads_folder must be a valid non-empty path string.")

    # Create the downloads folder if it doesn't exist
    os.makedirs(downloads_folder, exist_ok=True)

    # Define the browser configuration
    browser_config = {
        "viewport_size": 1024 * 5,
        "downloads_folder": downloads_folder,
        "request_kwargs": {
            "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"},
            "timeout": 300,
        },
        "serpapi_key": os.getenv("SERPAPI_API_KEY"),
    }

    # Define the tools
    text_limit = 100000
    browser = SimpleTextBrowser(**browser_config)
    web_tools = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
        Visualizer(working_dir=downloads_folder),
    ]
    # Create the web browsing agent
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=web_tools,
        max_steps=max_steps,
        verbosity_level=monitoring.LogLevel.DEBUG,
        planning_interval=4,
        name="search_agent",
        description="""An assistant agent specialized in searching the internet and reading content from provided URLs.
        Use this agent when you:
        - Want to inspect the content in the link included in the provided papers.
        - Need to look up specific information or resources.

        Always provide a clear, complete question or task in natural language.
        """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You are responsible for executing complex, multi-step web browsing tasks to support other agents working on literature survey writing.
    Your goals include:
    - Extracting relevant content from .txt files or inspecting .pdf/.doc/.md files using 'inspect_file_as_text'.
    - Summarizing information across multiple web sources when appropriate.

    When performing a search:
    1. Interpret the query as a real-world question, not just keywords.
    2. If needed, iterate: refine search terms, explore related pages, or analyze multiple sources.
    3. If you need clarification or the question is underspecified, use `final_answer("Your clarification question")` to request more detail.
    4. Always aim to provide comprehensive, accurate information in your final response.
    """
    return text_webbrowser_agent

if __name__ == "__main__":
    agent = create_web_browsing_agent(model_id="gpt-4.1")
    agent.run("Read the content of https://arxiv.org/abs/2405.01695")
    # GradioUI(manager_agent).launch()