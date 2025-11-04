import sys
import os
import argparse
import re
import json

from dotenv import load_dotenv
from smolagents import CodeAgent, ToolCallingAgent
from src.models import LiteLLMModel

from smolagents.monitoring import LogLevel
sys.path.insert(0, os.getcwd())
from default_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeTextFile,
    ModifyFile,
    CreateFileWithContent,
)
from smolagents.tools import Tool

# Import agent prompts from the new prompts package
from literature_survey_app.prompts import (
    WRITING_AGENT_DESCRIPTION,
    WRITING_AGENT_TASK_PROMPT,
    SURVEY_WRITING_AGENT_SYSTEM_PROMPT
)
from literature_survey_app.web_browsing_agent import create_web_browsing_agent

class ValidateLatex(Tool):
    name = "validate_latex"
    description = (
        "Validate LaTeX expressions for syntax errors and common issues. "
        "Checks for balanced delimiters, proper command syntax, and common LaTeX patterns. "
        "Returns detailed error messages with positions for any issues found."
    )
    inputs = {
        "latex_str": {
            "type": "string", 
            "description": "The LaTeX expression to validate."
        }
    }
    output_type = "string"

    def __init__(self, working_dir: str):
        super().__init__()
        self.working_dir = working_dir
        self.delimiters = {
            '{': '}',
            '(': ')',
            '[': ']',
            '$': '$',
            '\\(': '\\)',
            '\\[': '\\]'
        }
        self.common_errors = [
            (r'\\[a-zA-Z]+{', r'Missing closing brace after LaTeX command'),
            (r'\\[a-zA-Z]+\s+{', r'Space between LaTeX command and opening brace'),
            (r'\\[a-zA-Z]+$', r'LaTeX command at end of string'),
            (r'\\[a-zA-Z]+\s+\\[a-zA-Z]+', r'Space between LaTeX commands'),
        ]

    def forward(self, latex_str: str) -> str:
        # Check for balanced delimiters
        stack = []
        for i, char in enumerate(latex_str):
            if char in self.delimiters:
                stack.append((char, i))
            elif char in self.delimiters.values():
                if not stack:
                    return f"Error: Unmatched closing delimiter '{char}' at position {i}"
                last_open, pos = stack.pop()
                if self.delimiters[last_open] != char:
                    return f"Error: Mismatched delimiters: '{last_open}' at {pos} and '{char}' at {i}"
        
        if stack:
            return f"Error: Unmatched opening delimiter '{stack[-1][0]}' at position {stack[-1][1]}"
        
        # Check for common LaTeX errors
        for pattern, error_msg in self.common_errors:
            if re.search(pattern, latex_str):
                return f"Error: {error_msg}"
        
        return "LaTeX expression is valid"


# Load environment variables
load_dotenv(override=True)

def create_survey_writing_agent(model: LiteLLMModel, output_dir: str = "literature_survey_output") -> tuple[ToolCallingAgent, ToolCallingAgent]:
    """Run the literature survey agent with the specified topic, model, and output directory."""
    # Create web browsing agent
    web_browsing_agent = create_web_browsing_agent(model=model, downloads_folder=output_dir)
    # Create the writing agent
    writing_agent = ToolCallingAgent(
        tools=[ListDir(output_dir), SeeTextFile(output_dir), ModifyFile(output_dir), CreateFileWithContent(output_dir)],
        model=model,
        verbosity_level=LogLevel.DEBUG,
        planning_interval=15,
        max_steps=15,
        name="writing_agent", 
        description=WRITING_AGENT_DESCRIPTION
    )

    # Add specific prompt template for the writing agent
    writing_agent.prompt_templates["managed_agent"]["task"] += WRITING_AGENT_TASK_PROMPT

    # Update literature_survey_agent to include final_revision_agent
    survey_writing_agent = ToolCallingAgent(
        tools=[ListDir(output_dir), SeeTextFile(output_dir), ModifyFile(output_dir), CreateFileWithContent(output_dir)],
        model=model,
        managed_agents=[web_browsing_agent, writing_agent],
        verbosity_level=LogLevel.DEBUG,
        planning_interval=6,
        max_steps=20,
        name="literature_survey_agent",
        description="This agent is responsible for managing the web_browsing and writing agents."
    )

    survey_writing_agent.system_prompt += SURVEY_WRITING_AGENT_SYSTEM_PROMPT

    return writing_agent, survey_writing_agent


#GradioUI(literature_survey_agent).launch()
#This UI sometimes has difficulties showing up. Replace the GradioUI launch with direct agent execution
if __name__ == "__main__":
    # set up argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="The model ID to use for the agent.",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="apps/literature_survey/report",
        help="The directory to save the generated report.",
    )
    argparser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="The topic for the literature survey.",
    )
    args = argparser.parse_args()

    # Set up the save directory
    # if the directory already exists, add a number to the end
    if os.path.exists(f"{output_dir}/{args.topic}"):
        i = 1
        while os.path.exists(f"{output_dir}/{args.topic}_{i}"):
            i += 1
        output_dir = f"{output_dir}/{args.topic}_{i}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the agent directly
    literature_survey_agent = create_literature_survey_agent()

    literature_survey_agent.run(f"Write a literature survey on current research on {topic}, both theoretical and empirical. Your audience will be researchers and practitioners in the field of machine learning. Be sure to include technical details and references to the literature.")
    
    
    

   
   
   

  