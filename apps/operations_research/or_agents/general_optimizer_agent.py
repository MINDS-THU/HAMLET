from smolagents import LiteLLMModel, CodeAgent
from smolagents.monitoring import LogLevel
from general_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    CreateFileWithContent,
    LoadObjectFromPythonFile,
)
import yaml
import importlib
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

def create_general_optimizer_agent(model_id="gpt-4.1", managed_agents=[], working_directory="working_directory", max_steps=20, verbosity_level=LogLevel.INFO):
    """
    Create an agent that will solve general-purpose operations research problems using Python scripting, simulation, or custom algorithms.
    Args:
        model_id (str): The ID of the model to use.
        working_directory (str): The directory where the optimization files will be stored.
    Returns:
        CodeAgent: The configured general optimizer agent.
    """

    tools = [
        ListDir(working_directory),
        SeeFile(working_directory),
        ModifyFile(working_directory),
        CreateFileWithContent(working_directory),
        LoadObjectFromPythonFile(working_directory),
    ]

    model = LiteLLMModel(model_id=model_id)

    # Load the prompt template
    general_optimizer_prompt_template = yaml.safe_load(
        importlib.resources.files("apps.operations_research.or_agents.prompts").joinpath("general_optimizer.yaml").read_text(encoding="utf-8")
    )

    description = """
    General Purpose Optimizer Agent
    Best for: Custom algorithmic, or scripting tasks that do not fit the scope of mathematical, combinatorial, or metaheuristic optimizer agents.

    Problem Types:
    - Monte Carlo simulation
    - Custom Python algorithms
    - Stochastic process simulation
    - General scripting for operations research

    When to Use:
    - The problem cannot be formulated as a mathematical program, combinatorial problem, or metaheuristic search.
    - Flexible, code-based solutions are required.
    """

    general_optimizer_agent = CodeAgent(
        tools=tools,
        managed_agents=managed_agents,
        prompt_templates=general_optimizer_prompt_template,
        verbosity_level=verbosity_level,
        additional_authorized_imports=['numpy', 'numpy.*', 'random', 'random.*', 'math', 'math.*'],
        max_steps=max_steps,
        model=model,
        name="general_optimizer_agent",
        description=description,
    )

    return general_optimizer_agent