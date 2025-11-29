"""Example of using CodeAgentEnv with output_schema for RLVR training.

This example demonstrates how to:
1. Define an output schema for structured outputs
2. Create a CodeAgentEnv with the output_schema
3. Use the environment for training with reward functions that validate output format

RLVR (Reinforcement Learning from Verifier Feedback) uses verifier functions
to provide rewards based on code execution and output validation.
"""

from __future__ import annotations

import os

from datasets import Dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from hamlet.core.models import OpenAIServerModel
from hamlet.train.codeagent_environment import CodeAgentEnv

import asyncio

class MathResult(BaseModel):
    """Structured output schema for math problem solutions."""
    
    answer: float = Field(..., description="The numerical answer to the math problem")
    method: str = Field(..., description="Brief description of the method used")
    steps: list[str] = Field(..., description="Step-by-step explanation", min_length=1)


def create_dataset() -> Dataset:
    """Create a simple dataset for training."""
    return Dataset.from_dict({
        "question": [
            "Calculate 15 * 23",
            "What is 144 / 12?",
            "Compute 7^3",
        ],
        "answer": [
            "345",
            "12",
            "343",
        ],
    })


def build_environment() -> CodeAgentEnv:
    """Create a CodeAgentEnv configured for RLVR training with output_schema."""
    
    # Define agent configuration
    agent_kwargs = {
        "output_schema": MathResult,  # Enforce structured output format
        "tools": [],
        "max_steps": 3,
    }
    
    # Define model configuration
    model_kwargs = {
        "timeout": 30,
    }
    
    # Create dataset
    dataset = create_dataset()
    
    # Create environment
    env = CodeAgentEnv(
        agent_kwargs=agent_kwargs,
        model_kwargs=model_kwargs,
        model_class=OpenAIServerModel,
        dataset=dataset,
        message_type="chat",
        max_turns=3,
    )
    
    return env


def demonstrate_reward_functions(env: CodeAgentEnv) -> None:
    """Demonstrate the reward functions available in CodeAgentRubric."""
    
    print("Reward functions in CodeAgentRubric:")
    print("=" * 60)
    
    # Get the initial rubric (agent is None at this point)
    initial_rubric = env.rubric
    
    print(f"\nInitial rubric (before agent initialization):")
    print(f"  Number of reward functions: {len(initial_rubric.reward_funcs)}")
    print("  Reward functions:")
    for i, (func, weight) in enumerate(zip(initial_rubric.reward_funcs, initial_rubric.reward_weights), 1):
        print(f"    {i}. {func.__name__} (weight: {weight})")
        if func.__name__ == "successful_parsing_reward_func":
            print("       - Rewards successful code block parsing")
        elif func.__name__ == "successful_code_execution_reward_func":
            print("       - Rewards successful code execution")
    
    print("\nNote: After the first rollout, when agent is initialized with output_schema,")
    print("      the 'successful_output_schema_reward_func' will be automatically added")
    print("      to validate final answers against the output_schema.")
    
    print("\n" + "=" * 60)


async def run_single_rollout(env: CodeAgentEnv) -> None:
    """Run a single rollout and show the rewards."""
    
    from openai import AsyncOpenAI
    
    print("\nRunning a single rollout:")
    print("=" * 60)
    
    # Create OpenAI client (you can use any compatible API)
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )
    
    # Prepare a prompt
    prompt = [{"role": "user", "content": "Calculate 15 * 23 and return the result in the required format."}]
    
    # Run rollout
    completion, state = await env.rollout(
        client=client,
        model=os.getenv("HAMLET_MODEL_ID", "gpt-4o-mini"),
        prompt=prompt,
        answer="345",
        task="math",
    )
    
    # Score the rollout
    score_result = await env.rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer="345",
        state=state,
        task="math",
    )
    
    print(f"\nReward metrics:")
    for metric_name, value in score_result.metrics.items():
        print(f"  - {metric_name}: {value:.2f}")
    
    print(f"\nTotal reward: {score_result.reward:.2f}")
    print("=" * 60)


def main() -> None:
    """Main function demonstrating RLVR setup."""
    
    load_dotenv()
    
    print("RLVR Training Example with Output Schema")
    print("=" * 60)
    print("\nThis example shows how to use CodeAgentEnv for RLVR training")
    print("with structured output validation.\n")
    
    # Build environment
    print("Building CodeAgentEnv...")
    env = build_environment()
    print("✓ Environment created\n")
    
    # Show reward functions
    demonstrate_reward_functions(env)
    
    # Set OPENAI_API_KEY and OPENAI_BASE_URL environment variables
    asyncio.run(run_single_rollout(env))
    
    print("\n" + "=" * 60)
    print("Example setup complete!")
    print("\nTo use this environment for training:")
    print("1. Use it with GRPOTrainer for reinforcement learning")
    print("2. The reward functions will automatically validate:")
    print("   - Code parsing success")
    print("   - Code execution success")
    print("   - Output format compliance (when output_schema is defined)")
    print("=" * 60)


if __name__ == "__main__":
    main()

