"""Minimal Gradio UI demo for a CodeAgent.

Run `uv pip install -e .` once so `import hamlet` works everywhere, then:

    uv run python examples/gradio_gui_example.py
"""

import os
from dotenv import load_dotenv

from hamlet.core import CodeAgent, LiteLLMModel
from hamlet.serve import GradioUI
from hamlet.tools.file_editing.file_editing_tools import (
    CreateFileWithContent,
    ListDir,
    ModifyFile,
    SeeTextFile,
)


def build_agent(working_dir: str) -> CodeAgent:
    """Create a CodeAgent configured with file-editing tools and verbose logging."""

    model_id = os.getenv("HAMLET_MODEL_ID", "gpt-5-mini")
    model = LiteLLMModel(model_id=model_id)

    file_tools = [
        ListDir(working_dir=working_dir),
        SeeTextFile(working_dir=working_dir),
        ModifyFile(working_dir=working_dir),
        CreateFileWithContent(working_dir=working_dir),
    ]

    return CodeAgent(
        model=model,
        tools=file_tools,
        name="GUIExampleAgent",
        description=(
            "A starter agent that can explore a sandbox directory and perform simple file tasks."
        ),
        verbosity_level=2,
    )


def main() -> None:
    load_dotenv()

    working_dir = os.getenv("HAMLET_WORKSPACE_DIR", "./examples/simple_agent_workspace")
    readme_path = os.getenv(
        "HAMLET_AGENT_README",
        "https://github.com/MINDS-THU/HAMLET/blob/main/README.md",
    )
    os.makedirs(working_dir, exist_ok=True)

    agent = build_agent(working_dir)
    GradioUI(agent, file_upload_folder=working_dir, readme_md_path=readme_path).launch(share=False)


if __name__ == "__main__":
    main()
