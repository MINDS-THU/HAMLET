import os
from dotenv import load_dotenv
from general_tools.text_inspector.text_inspector_tool import TextInspectorTool
from general_tools.text_web_browser.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from general_tools.visual_qa.visual_qa import visualizer
from smolagents import (
    GoogleSearchTool,
    GradioUI,
    ToolCallingAgent,
    monitoring,
    LiteLLMModel,
)
load_dotenv()

def create_web_browsing_agent(model_id="gpt-4.1", downloads_folder="downloads_folder", max_steps=10):
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

    # Define the model parameters
    model_params = {
        "model_id": model_id,
    }
    model = LiteLLMModel(**model_params)

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
        visualizer,
    ]
    # Create the web browsing agent
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=web_tools,
        max_steps=max_steps,
        verbosity_level=monitoring.LogLevel.DEBUG,
        planning_interval=4,
        name="search_agent",
        description="""An assistant agent specialized in searching the internet to retrieve up-to-date technical information, documentation, or troubleshooting help.
        Use this agent when you:
        - Encounter API errors or require updated usage examples.
        - Need explanations or comparisons of algorithms or packages.
        - Seek online resources (e.g., GitHub issues, StackOverflow, documentation, academic sources).

        Always provide a clear, complete question or task in natural language.
        You may also specify constraints such as a date range, specific domains (e.g., stackoverflow.com), or document types (e.g., .pdf).
        """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You are responsible for executing complex, multi-step web browsing tasks to support other agents working on technical problem solving.
    Your goals include:
    - Finding correct and recent code examples or documentation.
    - Troubleshooting errors by consulting forums, issue trackers, or official docs.
    - Comparing approaches, libraries, or APIs when asked.
    - Extracting relevant content from .txt files or inspecting .pdf/.doc/.md files using 'inspect_file_as_text'.
    - Summarizing information across multiple web sources when appropriate.

    When performing a search:
    1. Interpret the query as a real-world question, not just keywords.
    2. If needed, iterate: refine search terms, explore related pages, or analyze multiple sources.
    3. If you need clarification or the question is underspecified, use `final_answer("Your clarification question")` to request more detail.

    Always prioritize high-quality, relevant sources such as official documentation, Stack Overflow, GitHub, academic articles, or recent blogs.
    """
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    return text_webbrowser_agent

if __name__ == "__main__":
    agent = create_web_browsing_agent(model_id="gpt-4.1")
    agent.run("Find out what methods does the ConcreteModel class of Pyomo have.")
    # GradioUI(manager_agent).launch()