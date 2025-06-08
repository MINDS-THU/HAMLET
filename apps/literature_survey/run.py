import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from smolagents import LiteLLMModel, CodeAgent, ToolCallingAgent, GradioUI
from smolagents.monitoring import LogLevel
from general_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    CreateFileWithContent,
)
from general_tools.open_deep_search.ods_tool import OpenDeepSearchTool
from .md2pdf_tool import CompileMarkdownToPDF
import tempfile

# Load environment variables
load_dotenv(override=True)

def create_literature_survey_agent(model_id="gpt-4.1", save_dir="apps/literature_survey/report"):
    os.makedirs(save_dir, exist_ok=True)

    # Set up OpenDeepSearch tool
    search_tool = OpenDeepSearchTool(model_name=model_id, reranker="jina")
    if not search_tool.is_initialized:
        search_tool.setup()

    # Set up the model
    model = LiteLLMModel(model_id=model_id)

    # Create the writing agent
    writing_agent = ToolCallingAgent(
        tools=[
            search_tool,
            ListDir(save_dir),
            SeeFile(save_dir),
            ModifyFile(save_dir),
            CreateFileWithContent(save_dir),
        ],
        model=model,
        verbosity_level=LogLevel.DEBUG,
        planning_interval=3,
        max_steps=50,
        name="writing_agent",
        description="""This agent is responsible for writing technically detailed sections of a literature survey in Markdown format.

The manager must:
- Tell this agent what the overall research topic is.
- Specify the current subtopic or section it should write.

Given that information, this agent will:
- Search and synthesize relevant technical material.
- Write a well-structured Markdown section.
- Include formal definitions, algorithms, examples, and citations to academic or technical sources.
- Format the output using proper Markdown conventions (headers, bullet points, inline code, hyperlinks).

The agent saves each section as a separate `.md` file in the specified output directory.
"""
    )
    writing_agent.prompt_templates["managed_agent"]["task"] += """You are writing **one Markdown section** of a literature survey based on the topic and task assigned by the manager agent.
The content should be self-contained and technically detailed. Include formal definitions, key methods, or equations where appropriate.

Formatting rules (minimal but strict)
-------------------------------------
1. **Headings**
• Use `#` for the section title, `##`, `###`, … for sub-levels.  
• **Do NOT hard-code numbers** (Pandoc will number automatically).

2. **References inside the text**  
• Cite like this: `[1]`, `[2]`, … (numeric, square-bracket).

3. **References list at the end of *each* section**
    •  Blank line **before** the first item, **between** items, and **after** the last one (Pandoc renders a clean list).

References

[1] Author et al. Title. Venue/URL, Year.

[2] ...

4. General Markdown hygiene  
• Bullet lists: start at column 0 with `- ` or `* `, keep one blank line
before/after the list.  
• Use inline math `$…$` or fenced blocks for LaTeX.  
• Prefer hyperlinks `[text](url)` over bare URLs.

Deliverables
------------
• Save the section to a `.md` file in the working directory, e.g. '01_introduction.md'.  
• Return a short summary plus the filename.
"""

    # Create the review agent
    review_agent = ToolCallingAgent(
        tools=[ListDir(save_dir), SeeFile(save_dir)],
        model=model,
        verbosity_level=LogLevel.DEBUG,
        planning_interval=3,
        max_steps=50,
        name="review_agent",
    description="""This agent is responsible for reviewing the **complete draft** of a literature survey written in Markdown format.

The manager should:
- Tell the agent the topic of the survey for context.
- Specify the things to check for in the review (e.g., technical accuracy, completeness, clarity).

The review agent will:
- Evaluate the full document for technical accuracy, completeness, and clarity.
- Ensure consistent formatting, terminology, and citation style across sections.
- Identify any gaps, inconsistencies, or formatting issues.
- Provide feedback and suggestions for improvement, but will not edit the document unless explicitly instructed.
"""
    )

    review_agent.prompt_templates["managed_agent"]["task"] += """You are reviewing **all Markdown sections** of the survey.

Check and report on
-------------------
• Technical accuracy and completeness.  
• Logical flow across sections.  
• **Consistent formatting**:
- Headings **without hard-coded numbers**.  
- Numeric in-text citations `[n]`.  
- Reference lists formatted exactly as described to the writing agent
(blank line before, between, after).  
- Uniform bullet-list style.

Output
------
• Overall assessment.  
• Precise list of issues per file (line numbers if useful).  
• Concrete, actionable fixes.
"""

    # Create the literature survey agent
    literature_survey_agent = CodeAgent(
        tools=[
            ListDir(save_dir),
            SeeFile(save_dir),
            ModifyFile(save_dir),
            CompileMarkdownToPDF(save_dir),
        ],
        model=model,
        managed_agents=[writing_agent, review_agent],
        verbosity_level=LogLevel.DEBUG,
        planning_interval=3,
        max_steps=50,
        name="literature_survey_agent",
        description="This agent is responsible for managing the writing and review agents."
    )

    return literature_survey_agent

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the Literature Survey Agent")
    argparser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4.1",
        help="The ID of the model to use for the agent.",
    )
    argparser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The directory to save the generated report.",
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
    base_temp_dir = "apps/literature_survey/reports"
    Path(base_temp_dir).mkdir(parents=True, exist_ok=True)

    # Set the save directory to a default if not provided
    if args.save_dir is None:
        args.save_dir = tempfile.mkdtemp(dir=base_temp_dir, prefix="report_")

    # Create the agent
    literature_survey_agent = create_literature_survey_agent(
        model_id=args.model_id,
        save_dir=args.save_dir,
    )

    if args.mode == "cli":
        # Run the agent in CLI mode
        while True:
            try:
                literature_survey_agent.run("Based on the conversation so far, talk with the user.", reset=False)
                print("Agent finished running. Waiting for next command...")
                print("Press Ctrl+C to exit.")
            except KeyboardInterrupt:
                print("Exiting...")
                break
    else:
        # Run the agent in Gradio mode
        print("Launching Gradio UI...")
        GradioUI(agent=literature_survey_agent, file_upload_folder=args.save_dir).launch()