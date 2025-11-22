"""Demonstrate concurrent code blocks + early stop strategies."""

import json
import os

from dotenv import load_dotenv

from src.hamlet.core.agents import CodeAgent
from src.hamlet.core.models import LiteLLMModel
from src.hamlet.core.monitoring import LogLevel


def build_agent() -> CodeAgent:
    """Encourage the LLM to emit multiple code blocks per step."""

    model_id = os.getenv("HAMLET_MODEL_ID", "gpt-5-mini")
    model = LiteLLMModel(model_id=model_id)
    return CodeAgent(
        model=model,
        tools=[],
        name="ParallelMathAgent",
        description="Shows how multiple code blocks can run concurrently with early stopping.",
        verbosity_level=LogLevel.DEBUG,
    )

def main() -> None:
    load_dotenv()
    agent = build_agent()

    task = ("Use at least two independent strategies for summing the squares from 1 to 75, and use `Early Stop Strategy: code` to check the 0-500000 bound.")
    # task = (
    #     "First brainstorm at least two independent strategies for summing the squares from 1 to 75, then in a new step "
    #     "execute both strategies concurrently in one action (Code#1, Code#2, etc.) with `Early Stop Strategy: code` checking the 0-500000 bound. "
    #     "Afterward, take an additional step to compare the logged outputs from both methods before emitting the final answer."
    # )
    agent.run(task)


if __name__ == "__main__":
    main()
