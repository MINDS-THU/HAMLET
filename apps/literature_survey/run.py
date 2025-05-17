import sys
import os
import argparse
# Add src/ to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dotenv import load_dotenv
from smolagents import LiteLLMModel, CodeAgent, ToolCallingAgent, GradioUI
from smolagents.monitoring import LogLevel
from general_tools.open_deep_search.ods_tool import OpenDeepSearchTool
from general_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    CreateFileWithContent,
)
from .md2pdf_tool import CompileMarkdownToPDF

# Load environment variables
load_dotenv(override=True)

# set up argparse
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model-id",
    type=str,
    default="gpt-4.1",
    help="The model ID to use for the agent.",
)
argparser.add_argument(
    "--save-dir",
    type=str,
    default="apps/literature_survey/report",
    help="The directory to save the generated report.",
)
args = argparser.parse_args()

# Set up the save directory
# if the directory already exists, add a number to the end
if os.path.exists(args.save_dir):
    i = 1
    while os.path.exists(f"{args.save_dir}_{i}"):
        i += 1
    args.save_dir = f"{args.save_dir}_{i}"

os.makedirs(args.save_dir, exist_ok=True)

# set up ods search tool
search_tool = OpenDeepSearchTool(
    model_name="gpt-4.1",
    reranker="jina"
)
if not search_tool.is_initialized:
    search_tool.setup()
# set up model and agent
model = LiteLLMModel(
    model_id=args.model_id,
)

writing_agent = ToolCallingAgent(
    tools=[search_tool, ListDir(args.save_dir), SeeFile(args.save_dir), ModifyFile(args.save_dir), CreateFileWithContent(args.save_dir)],
    model=model,
    verbosity_level=LogLevel.DEBUG,
    planning_interval=3,
    max_steps=50,
    name="writing_agent", 
    description="""
This agent is responsible for writing technically detailed sections of a literature survey in Markdown format.

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
# writing_agent.prompt_templates["managed_agent"]["task"] += """
# You are writing a section of a literature survey based on the topic and task assigned by the manager agent.

# Use Markdown formatting in your output.

# Your section should:
# - Section and subsection number and title (e.g., 1. Introduction).
# - Be self-contained and technically detailed.
# - Include formal definitions, key methods, or equations where appropriate.
# - Use proper structure with `##` section headers, bullet points, and hyperlinks.
# - Cite academic or technical sources using inline links or footnotes.

# Save your section as a `.md` file in the working directory. After writing, provide a summary of the section and the file name where it is saved.
# """
# --------------------------------------------------------------------------
# 1. WRITING‑AGENT  ▸ tighten section / reference formatting
# --------------------------------------------------------------------------
writing_agent.prompt_templates["managed_agent"]["task"] += """
You are writing **one Markdown section** of a literature survey based on the topic and task assigned by the manager agent.
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

review_agent = ToolCallingAgent(
    tools=[ListDir(args.save_dir), SeeFile(args.save_dir)],
    model=model,
    verbosity_level=LogLevel.DEBUG,
    planning_interval=3,
    max_steps=50,
    name="review_agent",
    description="""
This agent is responsible for reviewing the **complete draft** of a literature survey written in Markdown format.

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
# review_agent.prompt_templates["managed_agent"]["task"] += """
# You are reviewing a **complete draft** of a technical literature survey written in Markdown format, with each section being in a separate file.

# The manager will provide:
# - The overall topic of the report.
# - The things to check for in the review (e.g., technical accuracy, completeness, clarity).

# Your review should address:
# - Technical accuracy and completeness.
# - Logical flow and coherence across sections.
# - Consistency in formatting, headings, terminology, and citation style.
# - Clarity and readability of explanations.

# Please return:
# - A summary of your overall assessment.
# - A list of any identified issues or areas needing revision (refer to specific sections if possible).
# - Concrete suggestions for improvement.

# Fully review the document and provide feedback.
# """

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


literature_survey_agent = CodeAgent(
    tools=[ListDir(args.save_dir), SeeFile(args.save_dir), ModifyFile(args.save_dir), CompileMarkdownToPDF(args.save_dir)],
    model=model,
    managed_agents=[writing_agent, review_agent],
    verbosity_level=LogLevel.DEBUG,
    planning_interval=3,
    max_steps=50,
    name="literature_survey_agent",
    description="This agent is responsible for managing the writing and review agents."
)

literature_survey_agent.system_prompt += """\nGenerate a technical literature survey on the given topic.

**Phase 1**: Decompose the report into structured subtopics. Suggested sections:
1. Introduction and Background
2. Mathematical Formulation
3. Classical Algorithms and Techniques
4. Modern Approaches (e.g., Deep Learning + Bilevel)
5. Applications in Practice (e.g., ML, Energy, Economics)
6. Open Challenges and Future Directions
7. References

**Phase 2**: For each subtopic:
- Assign it to the writing agent as a separate task.
- Ask for a detailed, self-contained Markdown section with sources and technical depth.

**Phase 3**:
- Ask the review_agent to review all sections.
- Ensure structure, formatting, and citations are consistent.

**Phase 4**: Compile the reviewed sections into a complete Markdown report.
- Ensure the final report is well-formatted, relevant to the topic and contains sufficient details.
- Include a reference section with all sources cited in the report.
- Save the final report as a single pdf file.

Formatting policy (applies to every section)
-------------------------------------------
• Headings use `#`, `##`, … **without manual numbers**; Pandoc will add them.  
• In-text citations are numeric `[n]`; each section ends with a `## References`
list, one numbered entry per source, blank lines before / between / after
list items.  
• Bullet lists start with `- ` or `* ` at column 0 and are surrounded by one
blank line.  
Enforce these rules when delegating tasks or reviewing output.
"""

GradioUI(literature_survey_agent).launch()