"""Example of GRPO training with CodeAgentEnv and output_schema.

This example demonstrates how to:
1. Start vLLM server for model inference
2. Set up CodeAgentEnv with output_schema
3. Train the model using GRPOTrainer with parameter updates

Prerequisites:
- vLLM installed: `pip install vllm` or `uv sync --extra train`
- Start vLLM server before running training script
- Model path accessible

To start vLLM server:
    ./start_vllm_server.sh [MODEL_PATH] [PORT]
    
    Or manually:
    uv run python -m hamlet.train.inference.vllm_server --model /path/to/model --port 8000
"""

from __future__ import annotations

import os

from datasets import Dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from hamlet.core.models import OpenAIServerModel
from hamlet.train.codeagent_environment import CodeAgentEnv
from hamlet.train.grpo_config import GRPOConfig
from hamlet.train.grpo_trainer import GRPOTrainer
from hamlet.train.utils.model_utils import get_model_and_tokenizer


class MathResult(BaseModel):
    """Structured output schema for math problem solutions."""
    
    answer: float = Field(..., description="The numerical answer to the math problem")
    method: str = Field(..., description="Brief description of the method used")
    steps: list[str] = Field(..., description="Step-by-step explanation", min_length=1)


def create_dataset() -> Dataset:
    """Create a training dataset."""
    return Dataset.from_dict({
        "question": [
            "Calculate 15 * 23",
            "What is 144 / 12?",
            "Compute 7^3",
            "Find 25 + 17",
            "What is 100 - 45?",
            "Calculate 8 * 9",
            "What is 81 / 9?",
            "Compute 2^10",
        ],
        "answer": [
            "345",
            "12",
            "343",
            "42",
            "55",
            "72",
            "9",
            "1024",
        ],
    })


def get_local_model_path() -> str:
    """Get the local model path.
    
    Update this path to point to your model directory.
    Or set MODEL_PATH environment variable.
    """
    # Use environment variable if set
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        return model_path
    
    # Default model path - update this to your model location
    return "/path/to/your/model"


def build_environment() -> CodeAgentEnv:
    """Create a CodeAgentEnv configured for GRPO training with output_schema."""
    
    # Define agent configuration
    agent_kwargs = {
        "output_schema": MathResult,  # Enforce structured output format
        "tools": [],
        "max_steps": 3,
    }
    
    # Define model configuration (for vLLM server connection)
    # Note: base_url and api_key will be automatically injected from the
    # AsyncOpenAI client during rollout. model_id will be set from the
    # model parameter passed to rollout (from model.config._name_or_path).
    model_kwargs = {
        "timeout": 30,
    }
    
    # Create dataset
    dataset = create_dataset()
    
    # Create environment
    # Note: CodeAgentEnv will use OpenAIServerModel to connect to vLLM server
    env = CodeAgentEnv(
        agent_kwargs=agent_kwargs,
        model_kwargs=model_kwargs,
        model_class=OpenAIServerModel,
        dataset=dataset,
        message_type="chat",
        max_turns=3,
    )
    
    return env


def create_training_config(model_path: str) -> GRPOConfig:
    """Create GRPO training configuration."""
    
    config = GRPOConfig(
        # Output and logging
        output_dir="./outputs/grpo_training",
        run_name="grpo_output_schema_example",
        logging_steps=1,
        save_steps=10,
        save_strategy="steps",
        log_completions=True,
        
        # Training parameters
        learning_rate=1e-6,
        max_steps=20,  # Small number for example
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=4,  # Must divide batch_size * gradient_accumulation_steps
        
        # Model parameters
        max_seq_len=2048,
        max_prompt_length=512,
        
        # Generation parameters
        temperature=0.7,
        top_p=0.9,
        
        # GRPO specific
        num_iterations=1,
        epsilon=0.2,
        beta=0.001,
        
        # vLLM server configuration
        vllm_server_host="localhost",
        vllm_server_port=8000,
        vllm_server_timeout=300.0,
        
        # Other
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )
    
    return config


def main() -> None:
    """Main function demonstrating GRPO training setup."""
    
    load_dotenv()
    
    print("GRPO Training Example with Output Schema")
    print("=" * 70)
    print("\nThis example shows how to train a model using GRPO with")
    print("structured output validation.\n")
    
    # Step 1: Get local model path
    print("[Step 1] Locating local model...")
    try:
        model_path = get_local_model_path()
        print(f"✓ Found model at: {model_path}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease:")
        print("1. Set MODEL_PATH environment variable")
        print("2. Or update the path in get_local_model_path() function")
        return
    
    # Step 2: Build environment
    print("\n[Step 2] Building CodeAgentEnv...")
    env = build_environment()
    print("✓ Environment created")
    
    # Step 3: Load model and tokenizer for training
    print("\n[Step 3] Loading model and tokenizer for training...")
    print(f"  Loading from: {model_path}")
    try:
        import torch
        model, tokenizer = get_model_and_tokenizer(
            model_path,
            use_liger=False,  # Set to True if you have liger_kernel installed
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
        )
        print("✓ Model and tokenizer loaded")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease ensure:")
        print("1. The model path is correct")
        print("2. transformers is installed")
        print("3. You have sufficient GPU memory")
        return
    
    # Step 4: Create training config
    print("\n[Step 4] Creating training configuration...")
    config = create_training_config(model_path)
    print("✓ Training config created")
    print(f"  - Output directory: {config.output_dir}")
    print(f"  - Max steps: {config.max_steps}")
    print(f"  - Learning rate: {config.learning_rate}")
    
    # Step 5: Create GRPOTrainer
    print("\n[Step 5] Creating GRPOTrainer...")
    print("  Note: This requires vLLM server to be running.")
    print("  Start it with:")
    print(f"    ./start_vllm_server.sh {model_path} 8000")
    print(f"    Or: uv run python -m hamlet.train.inference.vllm_server --model {model_path} --port 8000")
    
    try:
        trainer = GRPOTrainer(
            model=model,
            env=env,
            args=config,
            processing_class=tokenizer,
        )
        print("✓ GRPOTrainer created")
    except Exception as e:
        print(f"✗ Error creating trainer: {e}")
        print("\nPlease ensure:")
        print("1. vLLM server is running on localhost:8000")
        print("2. The server was started with the same model path")
        print("3. All required dependencies are installed")
        return
    
    # Step 6: Show reward functions
    print("\n[Step 6] Reward functions in CodeAgentRubric:")
    print("-" * 70)
    # Note: Agent will be initialized on first rollout
    initial_rubric = env.rubric
    print(f"Initial reward functions: {len(initial_rubric.reward_funcs)}")
    for func in initial_rubric.reward_funcs:
        print(f"  - {func.__name__}")
    print("\nAfter first rollout (when agent is initialized),")
    print("'successful_output_schema_reward_func' will be added automatically.")
    
    # Step 7: Start training
    print("\n" + "=" * 70)
    print("[Step 7] Starting training...")
    print("=" * 70)
    print("\nTraining will:")
    print("1. Generate rollouts using vLLM server")
    print("2. Score them using CodeAgentRubric (including output_schema validation)")
    print("3. Update model parameters based on rewards")
    print("4. Sync updated weights back to vLLM server")
    print("\nPress Ctrl+C to stop training early.\n")
    
    try:
        trainer.train()
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Model saved to: {config.output_dir}")
        print("=" * 70)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        print("\nCommon issues:")
        print("1. vLLM server not running or not accessible")
        print("2. Model path incorrect")
        print("3. Insufficient GPU memory")
        print("4. Port 8000 already in use")


if __name__ == "__main__":
    main()

