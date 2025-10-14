"""Minimal GRPO training example for the CodeAgent environment.

Notes
- This example assumes you have a vLLM server running (OpenAI-compatible) and that
  you want to train a local HF model while generating rollouts against that server.
- Adjust model paths, batch sizes, and vLLM ports to match your setup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.hamlet.tools.file_editing.file_editing_tools import (
    CreateFileWithContent,
    ListDir,
    ModifyFile,
    SeeTextFile,
)
from src.hamlet.train.codeagent_environment import CodeAgentEnv


from datasets import Dataset, load_dataset


def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return text
    return text.split("####")[1].strip()
def preprocess_gsm8k(x: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": x["question"],
        "answer": extract_hash_answer(x["answer"]),
    }

def load_environment() -> CodeAgentEnv:
    dataset = load_dataset("openai/gsm8k", "main")["train"]

    dataset = dataset.map(
        preprocess_gsm8k, num_proc=10, remove_columns=dataset.column_names
    )  # type: ignore
    if "temp_answer" in dataset.column_names:
        dataset = dataset.rename_column("temp_answer", "answer")
    dataset = dataset.select(range(5))

    workspace = Path(__file__).resolve().parent / "simple_agent_workspace" / "rollout_tool_check"
    workspace.mkdir(parents=True, exist_ok=True)
    env = CodeAgentEnv(
        dataset=dataset,
        message_type="chat",
        max_turns=10,
        agent_kwargs={
            "max_steps": 6,
            "use_structured_outputs_internally": False,
            "tools": [
                ListDir(working_dir=str(workspace)),
                SeeTextFile(working_dir=str(workspace)),
                ModifyFile(working_dir=str(workspace)),
                CreateFileWithContent(working_dir=str(workspace)),
            ],
        },
    )
    return env

from openai import AsyncOpenAI

def main() -> None:
    # Choose a local or HF model to train
    # model_name = os.environ.get("HAMLET_TRAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    # model, tokenizer = hamlet.train.get_model_and_tokenizer(model_name)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    client = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    env = load_environment()
    results = env.evaluate(client=client, model="gpt-5")
    print(results)
    # args = hamlet.train.grpo_defaults(run_name="codeagent")
    # # ~16k context example
    # args.max_prompt_length = 12288
    # args.max_tokens = 1024
    # args.max_seq_len = 13312

    # # Training knobs — adjust for your hardware
    # args.per_device_train_batch_size = 1
    # args.num_generations = 2
    # args.gradient_accumulation_steps = 2
    # args.eval_strategy = "steps"
    # args.eval_steps = 10
    # args.save_strategy = "steps"
    # args.save_steps = 100
    # args.max_steps = 20

    # # vLLM settings for rollout generation
    # args.vllm_server_host = os.environ.get("VLLM_HOST", "0.0.0.0")
    # args.vllm_server_port = int(os.environ.get("VLLM_PORT", "8000"))
    # args.vllm_server_timeout = 300.0

    # model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    # args.dataloader_drop_last = True

    # trainer = hamlet.train.GRPOTrainer(
    #     model=model,
    #     processing_class=tokenizer,
    #     env=env,
    #     peft_config=hamlet.train.lora_defaults(),
    #     args=args,
    # )
    # trainer.train()


if __name__ == "__main__":
    main()