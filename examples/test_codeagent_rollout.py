"""Utility script to exercise the CodeAgent training environment.

Run this file to spin up a small `CodeAgentEnv`, submit a single rollout
using an asynchronous OpenAI-compatible client (for example a vLLM server),
and print both the generated conversation and rubric scores.

Environment variables:
    HAMLET_EXAMPLE_BASE_URL  Base URL for the OpenAI-compatible endpoint.
                             Defaults to ``http://localhost:8000/v1``.
    HAMLET_EXAMPLE_API_KEY   API key/token for the endpoint. Defaults to ``EMPTY``.
    HAMLET_EXAMPLE_MODEL     Model identifier to request on the endpoint.
                             Defaults to ``gpt-4o-mini``.

The script uses a minimal synthetic dataset so it can run without any
external files. Feel free to edit the prompt or provide your own dataset
for more realistic smoke tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset
from openai import AsyncOpenAI

from src.hamlet.tools.file_editing.file_editing_tools import (
    CreateFileWithContent,
    ListDir,
    ModifyFile,
    SeeTextFile,
)
from src.hamlet.train.codeagent_environment import CodeAgentEnv
from src.hamlet.train.utils.types import ChatCompletionMessageParam, Messages, State
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

WORKSPACE_ROOT = PROJECT_ROOT / "examples" / "simple_agent_workspace" / "rollout_tool_check"
EXPECTED_TOOL_NAMES = {
    "list_dir",
    "see_text_file",
    "modify_file",
    "create_file_with_content",
}


def _prepare_workspace() -> Path:
    if WORKSPACE_ROOT.exists():
        shutil.rmtree(WORKSPACE_ROOT)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT


def _validate_tool_usage(state: State) -> None:
    full_steps = state.get("full_steps")
    if not isinstance(full_steps, list):
        raise RuntimeError("Rollout state missing 'full_steps'; cannot validate tool usage.")

    observed_tool_names: list[str] = []
    for step in full_steps:
        tool_calls = step.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                observed_tool_names.append(name)

    print("\nTool call summary:")
    if not observed_tool_names:
        raise RuntimeError(
            "The rollout completed without invoking any tools. Confirm the model supports tool calling and retry."
        )

    for idx, tool_name in enumerate(observed_tool_names, start=1):
        print(f"  - [{idx}] {tool_name}")

    unexpected_tools = sorted({name for name in observed_tool_names if name not in EXPECTED_TOOL_NAMES})
    if unexpected_tools:
        print("Encountered additional tool names:", ", ".join(unexpected_tools))
    else:
        print("All tool invocations used the expected file editing tools.")

    created_file = WORKSPACE_ROOT / "tool_success.txt"
    if not created_file.exists():
        raise RuntimeError(
            "Expected `tool_success.txt` to be created inside the rollout workspace, but the file was not found."
        )

    content = created_file.read_text(encoding="utf-8").strip()
    print("tool_success.txt contents:")
    print(content)
    if "Paris" not in content:
        raise RuntimeError(
            "tool_success.txt does not contain the expected answer. Tool execution may have failed."
        )

    print("Tool validation succeeded: the agent invoked tools and wrote the expected output.")


def _build_demo_dataset() -> Dataset:
    """Create a single-example dataset compatible with ``CodeAgentEnv``."""

    prompt_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a meticulous Python coding assistant."},
        {
            "role": "user",
            "content": (
                "Use the available filesystem tools to inspect the working directory, "
                "create a file named `tool_success.txt` containing the capital of France, "
                "and then show the contents of that file. Make sure you call at least one tool and return the final confirmation once the check passes."
            ),
        },
    ]

    return Dataset.from_list(
        [
            {
                "id": 0,
                "prompt": prompt_messages,
                "answer": "",
                "task": "codegen",
                "info": {},
            }
        ]
    )


def load_environment() -> CodeAgentEnv:
    """Instantiate a ``CodeAgentEnv`` wired with a tiny demo dataset."""

    dataset = _build_demo_dataset()
    workspace = _prepare_workspace()
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


def _print_messages(messages: Iterable[ChatCompletionMessageParam]) -> None:
    for idx, message in enumerate(messages, start=1):
        role = message.get("role", "assistant")
        content = message.get("content", "")
        print(f"[{idx}] {role}: {content}")


async def _run_single_rollout(env: CodeAgentEnv, client: AsyncOpenAI, model_name: str) -> None:
    if env.dataset is None:
        raise ValueError("Environment dataset is not initialised.")

    sample = env.dataset[0]
    prompt: Messages = sample["prompt"]
    answer = sample.get("answer", "")
    task = sample.get("task", "demo-task")
    info = sample.get("info", {})

    # print("Prompt:")
    # _print_messages(prompt)  # type: ignore[arg-type]

    completion: Messages
    state: State
    # Sampling args to test propagation into the underlying model
    sampling_args: dict[str, Any] = {
        "temperature": 0.2,
        "top_p": None,
        "n": None,
        # Use a generous token budget to avoid truncating the agent's code generation
        "max_completion_tokens": None,
        "extra_body": {"metadata": {"example": "sampling_args_test"}},
    }
    completion, state = await env.rollout(
        client=client,
        model=model_name,
        prompt=prompt,
        answer=answer,
        task=task,
        info=info,
        sampling_args=sampling_args,
    )

    print("\n=========== Rollout completion ---")
    # if isinstance(completion, list):
    #     _print_messages(completion)
    # else:
    #     print(completion)
    print(completion)
    print("\n=========== State summary:")
    # print(json.dumps(state, indent=2, default=str))
    print(state)

    # Verify that sampling_args were applied to the underlying model via kwargs
    _verify_sampling_applied(env, sampling_args)

    print("\n--- Scoring rollout ---")
    scores = await env.rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer=answer,
        state=state,
        task=task,
        info=info,
    )
    print("Reward:", scores.reward)
    print("Metrics:", json.dumps(scores.metrics, indent=2))

    # Try to validate tool usage, but don't fail the run if the model didn't cooperate.
    try:
        _validate_tool_usage(state)
    except RuntimeError as e:
        print("\n[warning] Tool validation skipped:", str(e))


async def _async_main() -> None:
    env = load_environment()

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model_name = "gpt-5"

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        await _run_single_rollout(env, client, model_name)
    finally:
        await client.close()


def _verify_sampling_applied(env: CodeAgentEnv, sampling_args: dict[str, Any]) -> None:
    """Check that the model received the translated sampling args.

    We access the model implementation (OpenAI-compatible) and inspect its kwargs,
    which are forwarded on each generation call.
    """
    model = getattr(env, "_model", None)
    if model is None:
        raise RuntimeError("Model is not initialized on the environment; cannot verify sampling args.")

    model_kwargs = getattr(model, "kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise RuntimeError("Model.kwargs is not a dict; cannot verify sampling args.")

    # Translate expectations (max_completion_tokens -> max_tokens) as CodeAgentEnv.rollout does
    expected = dict(sampling_args)
    if "max_completion_tokens" in expected and "max_tokens" not in expected:
        expected["max_tokens"] = expected.pop("max_completion_tokens")

    print("\nApplied model kwargs subset:")
    observed_subset = {
        k: model_kwargs.get(k)
        for k in ["temperature", "top_p", "n", "max_tokens", "extra_body"]
        if k in model_kwargs
    }
    print(json.dumps(observed_subset, indent=2, default=str))

    # Validate direct matches for simple scalars
    for k in ["temperature", "top_p", "n", "max_tokens"]:
        if k in expected:
            if model_kwargs.get(k) != expected[k]:
                raise RuntimeError(
                    f"Sampling arg '{k}' not applied as expected: observed={model_kwargs.get(k)} expected={expected[k]}"
                )

    # Validate extra_body is present and contains our metadata
    if "extra_body" in expected:
        observed_extra = model_kwargs.get("extra_body", {}) or {}
        exp_extra = expected["extra_body"] or {}
        # Shallow subset check
        for key, val in exp_extra.items():
            if key not in observed_extra:
                raise RuntimeError(f"extra_body missing key '{key}' in model kwargs")
            if observed_extra[key] != val:
                raise RuntimeError(
                    f"extra_body value mismatch for key '{key}': observed={observed_extra[key]} expected={val}"
                )

    print("Sampling args successfully applied to model kwargs.")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
