HAMLET is a lightweight, all-in-one framework for building and iterating on LLM agents. It lets you define agents, launch a Gradio GUI, instrument runs with Langfuse, and fine-tune models using the built-in GRPO trainer. The framework is modified primarily on two open-source projects: [smolagents](https://github.com/huggingface/smolagents) for the core agent structure (`src/hamlet/core`) and [verifiers](https://github.com/PrimeIntellect-ai/verifiers) for the training stack (`src/hamlet/train`).

## Installation

Follow the steps below to set up HAMLET in a fresh environment. The project uses [uv](https://github.com/astral-sh/uv) to manage Python dependencies because it keeps lock files fast and reproducible. You can still use `pip`, but uv is the recommended path.

### 1. Prerequisites
- Python 3.10 or newer (3.11+ works as well)
- uv (install with the command below or follow the instructions in the uv repository)

```powershell
pip install uv
```

### 2. Clone the repository
```powershell
git clone https://github.com/MINDS-THU/HAMLET.git
cd HAMLET
```

### 3. Install dependencies
Install the base runtime:
```powershell
uv sync
```

Optional extras:
- `uv sync --extra tools` for the toolchain utilities (file editing, retrieval, visual QA, etc.).
- `uv sync --extra train` for the training stack (GRPO trainer, vLLM client, etc.).

### 4. Use the environment
- Run commands inside the uv-managed env with `uv run`, e.g. `uv run pytest` or `uv run python examples\gradio_gui_example.py`.
- Alternatively, activate the virtual environment directly: `.\.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Unix shells.

## Getting Started

Jump into the `examples/` directory to see how to use HAMLET in action:
- `examples/gradio_gui_example.py`: launches a Gradio UI for interactive tool usage.
- `examples/parallel_code_blocks_example.py`: shows multi-tool calls dispatched in parallel.
- `examples/structured_schema_example.py`: demonstrates structured input/output schemas.
