from smolagents import LiteLLMModel, CodeAgent
from smolagents.monitoring import LogLevel
from general_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    CreateFileWithContent,
    LoadObjectFromPythonFile,
)
from pathlib import Path
import yaml
import importlib
from dotenv import load_dotenv
load_dotenv(override=True)

from .web_browsing_agent import create_web_browsing_agent


def create_metaheuristic_optimizer_agent(model_id="gpt-4.1", managed_agents=[], working_directory="working_directory", max_steps=20, verbosity_level=LogLevel.INFO):
    """
    Create an agent that will solve heuristic and simulation-based optimization problems using pymoo.
    Args:
        model_id (str): The ID of the model to use.
        working_directory (str): The directory where the optimization files will be stored.
    Returns:
        CodeAgent: The configured metaheuristic optimizer agent.
    """

    # Load the prompt template
    metaheuristic_optimizer_prompt_template = yaml.safe_load(
                importlib.resources.files("apps.operations_research.or_agents.prompts").joinpath("metaheuristic_optimizer.yaml").read_text(encoding="utf-8")
            )

    tools = [
        ListDir(working_directory),
        SeeFile(working_directory),
        ModifyFile(working_directory),
        CreateFileWithContent(working_directory),
        LoadObjectFromPythonFile(working_directory),
    ]

    model = LiteLLMModel(model_id=model_id)
    description = """
    Metaheuristic Optimizer (pymoo)
    Best for: Complex, black-box, or multi-objective problems where traditional methods are ineffective.

    Problem Types:
    - Multi-objective optimization (e.g., Pareto front analysis)
    - Black-box optimization (no explicit mathematical model)
    - Highly nonlinear or non-convex problems
    - Problems with noisy or simulation-based evaluations

    Optimization Techniques:
    - Genetic Algorithms (GA)
    - Differential Evolution (DE)
    - Particle Swarm Optimization (PSO)
    - NSGA-II, NSGA-III, MOEA/D, and other evolutionary algorithms

    Typical Applications:
    - Engineering design optimization
    - Hyperparameter tuning in machine learning
    - Simulation-based optimization
    - Problems with expensive or noisy objective evaluations

    When to Use:
    - The problem lacks a clear mathematical formulation.
    - Multiple conflicting objectives need to be optimized simultaneously.
    - Evaluations are expensive, noisy, or derived from simulations.
    """

    metaheuristic_optimizer_agent = CodeAgent(
        tools=tools,
        managed_agents=managed_agents,
        prompt_templates=metaheuristic_optimizer_prompt_template,
        additional_authorized_imports=['pymoo', 'pymoo.*', 'numpy', 'numpy.*', 'random', 'random.*', 'math', 'math.*'],
        verbosity_level=verbosity_level,
        max_steps=max_steps,
        model=model,
        name="metaheuristic_optimizer_agent",
        description=description,
    )

    return metaheuristic_optimizer_agent