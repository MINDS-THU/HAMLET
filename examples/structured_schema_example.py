"""Show how to enforce structured inputs/outputs with CodeAgent schemas."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from hamlet.core import CodeAgent, LiteLLMModel
from hamlet.core.monitoring import LogLevel


class PersonaRequest(BaseModel):
    persona: str = Field(..., description="Short description of the audience you are writing for.")
    goal: str = Field(..., description="Goal that the plan should optimize for.")
    tone: str = Field(..., description="Tone or style to follow in the answer.")


class PersonaResponse(BaseModel):
    summary: str = Field(..., description="One paragraph recap tailored to the persona.")
    talking_points: list[str] = Field(..., description="Concise bullet points with key facts.", min_length=3)
    next_steps: list[str] = Field(..., description="Actionable next steps to pursue.", min_length=1)


def build_agent() -> CodeAgent:
    """Create an agent that validates both the structured request and response bodies."""

    model_id = os.getenv("HAMLET_MODEL_ID", "gpt-5-mini")
    model = LiteLLMModel(model_id=model_id)

    return CodeAgent(
        model=model,
        tools=[],
        name="StructuredPersonaAgent",
        description="Accepts typed input data and emits a validated PersonaResponse.",
        input_schema=PersonaRequest,
        output_schema=PersonaResponse,
        verbosity_level=LogLevel.DEBUG,
    )


def main() -> None:
    load_dotenv()

    agent = build_agent()
    prompt = (
        "Use the structured variables to prepare a briefing for the given persona. "
        "Produce your final_answer as a dict that matches PersonaResponse."
    )
    structured_input = PersonaRequest(
        persona="Supply-chain analyst at an EV company",
        goal="Highlight the top three inventory risks for Q1",
        tone="Confident yet concise",
    )

    run_result = agent.run(prompt, structured_input=structured_input)

    print("Validated response:\n")
    print(run_result)


if __name__ == "__main__":
    main()
