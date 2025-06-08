# Operations Research Agent

This agent automates the process of solving operations research (OR) and optimization problems using optimization solvers, a local knowledge base, and web search. It supports algebraic, discrete, and heuristic optimization, and can curate, retrieve, and apply knowledge from structured repositories. The agent is modular, extensible, and designed for both research and practical applications.

---

## Agent and Tools Overview

- **Manager Agent**: Orchestrates the workflow, assigns tasks to specialized agents, and manages the overall optimization process.
- **Mathematical Optimizer Agent**: Solves algebraic optimization problems (e.g., LP, MIP, NLP, MINLP) using Pyomo and open-source solvers.
- **Combinatorial Optimizer Agent**: Handles discrete and combinatorial optimization tasks (e.g., scheduling, routing, assignment) using Google OR-Tools.
- **Metaheuristic Optimizer Agent**: Applies heuristic and metaheuristic algorithms (e.g., genetic algorithms, evolutionary methods) for complex or black-box problems using pymoo.
- **General Optimizer Agent**: Handles any other situations that do not fit the above categories, including simulation-based, custom algorithmic, or general scripting problems.
- **Knowledge Curation Agent**: Adds, updates, and organizes knowledge in the local repository.
- **Knowledge Retrieval Agent**: Searches and retrieves information from the local knowledge base and supplements with web search if needed.
- **Web Browsing Agent**: Performs web searches to gather additional data or context.
- **[File Editing Tools](../../general_tools/file_editing/file_editing_tools.py)**: Manage reading, writing, and modifying files in the working directory.
- **[Knowledge Base Tools](../../general_tools/kb_repo_management/)**: Index, search, and maintain the local knowledge repository.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/cyrilli/COOPA.git
cd COOPA
```

### 2. Install Dependencies

- **Python packages** (conda installation recommended):

From the root of `COOPA`, execute:

```bash
conda create -n coopa_env python=3.10
conda activate coopa_env
pip install -r requirements.txt
```

Before running the agent, you must ensure that the solver packages used by the three optimizer agents are installed:

- **Mathematical Optimizer Agent**:  
  Uses [Pyomo](http://www.pyomo.org/) with the following open-source solvers:
  - **GLPK** (for Linear and Mixed-Integer Programming)
  - **IPOPT** (for Nonlinear Programming)

- **Combinatorial Optimizer Agent**:  
  Uses [Google OR-Tools](https://developers.google.com/optimization) for constraint programming, routing, assignment, and network flow problems.

- **Metaheuristic Optimizer Agent**:  
  Uses [pymoo](https://pymoo.org/) for genetic algorithms, evolutionary algorithms, and multi-objective optimization.

**Install the solvers as follows:**

```bash
# Install GLPK (for Pyomo LP/MIP)
conda install -c conda-forge glpk

# Install IPOPT (for Pyomo NLP/MINLP)
conda install -c conda-forge ipopt
```
Google OR-Tools and Pyomoo are already included in requirements.txt.

> **Note:**  
> If you want to use other solvers (e.g., commercial solvers like Gurobi, CPLEX, or other open-source solvers), you can do so by adjusting the prompts in the optimizer agent files under `apps/operations_research/or_agents/`.
>
> Ensure that the solver executables are available in your system PATH.

### 3. Register for Required API Keys

Some agents use external services for web search and language models. Register and obtain API keys for:

- **Serper API** (Google Search): [https://serper.dev/](https://serper.dev/)
- **OpenAI API** (or compatible LLM provider): [https://platform.openai.com/](https://platform.openai.com/)
- **Jina API** (for reranking): [https://jina.ai/](https://jina.ai/)
- **HF_TOKEN** (for Hugging Face model access or Gradio deployment): [https://huggingface.co/](https://huggingface.co/)

### 4. Configure Environment Variables

Create a `.env` file in the root of `COOPA` and add your API keys:

```
SERPER_API_KEY=your_serper_api_key
OPENAI_API_KEY=your_openai_api_key
JINA_API_KEY=your_jina_api_key
HF_TOKEN=your_huggingface_token  # Optional: for Hugging Face model access or Gradio deployment
# Optionally, add other keys as needed
```

---

## Running the Operations Research Agent

From the root of `COOPA`, you can run the agent by:

```bash
python -m apps.operations_research.run --model_id <your-model-id> --working_directory <path-to-working-directory> --knowledge_base_directory <path-to-knowledge-base> --index_dir <path-to-index-directory> --mode <cli-or-gradio>
```

- `--model_id` (optional): The LLM model to use (default: `gpt-4.1`)
- `--working_directory` (optional): Path to the directory where the agent will store its working files. If not specified, a temporary directory will be created under `apps/operations_research/temp_files/`
- `--knowledge_base_directory` (optional): Path to the directory where the agent will store its working files. If not specified, a temporary directory will be created under `apps/operations_research/temp_files/`
- `--index_dir` (optional): Path to the directory where the vector store index will be stored. If not specified, a temporary directory will be created under `apps/operations_research/temp_files/`
- `--mode` (optional): The mode to run the agent in. Choose between:
  - `gradio`: Launches a web-based interface for interactive control.
  - `cli`: Runs the agent in the command-line interface for direct interaction.

**What is the knowledge base directory?**

The knowledge base directory is a folder containing files relevant to operations research, organized for clarity and ease of retrieval. This is where you can store files that you think that may be helpful for LLM agent to solve the problem.

You can include:
- Python scripts or Jupyter notebooks with optimization models or algorithms
- PDF or Markdown documents with theory, solved examples, or best practices
- Data files, references, or notes

If no knowledge base directory is provided, the LLM is instructed to dynamically create and maintain a knowledge base as it solves new problems, organizing relevant information, code snippets, and solutions for future reuse.

**Note:**  
Currently, binary files like PDF can be stored in the knowledge base, but **only text files** (such as `.py`, `.md`, `.txt`, `.ipynb`) are chunked and indexed for semantic search. PDF and other binary files are not chunked or searchable.

**Example structure:**
```
or_knowledge_base/
├── algorithms/
│   ├── network_flow/
│   │   ├── edmonds_karp.py
│   │   └── min_cost_flow.md
│   ├── linear_programming/
│   │   └── simplex_method.md
│   └── integer_programming/
│       └── branch_and_bound.py
├── code_examples/
│   ├── scheduling/
│   │   └── job_shop_example.ipynb
│   └── routing/
│       └── tsp_example.py
├── textual_knowledge/
│   ├── duality_theory.md
│   └── decomposition_methods.md
├── glossary/
│   └── or_terms.md
└── references/
    └── classic_papers.pdf
```

**How does the agent use it?**

- When you launch the agent, it will **read and chunk** all files in the specified knowledge base directory.
- It then builds a **vector store** (semantic search index) from these chunks, enabling the agent to retrieve relevant knowledge and code snippets during problem solving.
- This allows the agent to ground its reasoning and solutions in your curated knowledge, and to continually expand and organize the knowledge base as you use it.

After execution, the agent will first process and index the knowledge base directory for semantic search, before interacting with the user.


If `gradio` mode is chosen, a Gradio web UI will launch for interactive control. Example output:

```bash
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxx.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
```

---

## Notes

- Ensure your API keys are valid and have sufficient quota.
- You can customize agent behavior and prompts in `apps/operations_research/run.py` and the `or_agents` submodules.
- The knowledge base can be extended with new algorithms, code, and documentation using the curation agent.
- For best results, keep your environment and dependencies up to date.

---

## Example

Execute the following command from the root folder of COOPA:

```bash
python -m apps.operations_research.run --model_id gpt-4.1 --mode cli
```
Then the manager agent will engage in conversation with you in the commandline window. For example:
```base
[LLM]: Hello! What can I help you with today?
[User]: Solve this: A laundromat can buy two types of washing machines, a top-loading model and a front-loading model. The top-loading model can wash 50 items per day while the front-loading model can wash 75 items per day. The top-loading model consumes 85 kWh per day while the front-loading model consumes 100 kWh per day. The laundromat must be able to wash at least 5000 items per day and has available 7000 kWh per day. Since the top-loading machine are harder to use, at most 40% of the machines can be top-loading. Further, at least 10 machines should be front-loading. How many of each machine should the laundromat buy to minimize the total number of washing machines?

...

Out - Final answer: To minimize the total number of washing machines, the laundromat should buy 0 top-loading machines
and 67 front-loading machines.

Agent finished running. Waiting for next command...

Press Ctrl+C to exit.

[LLM]: I have solved the problem. The laundromat should buy 0 top-loading machines and 67 front-loading machines. Let me know what you want me to do next.
[User]: You have correctly solved the previous problem! Can you save the problem description and the solution code to the knowledge base in case we may encounter similar problem in the future?

...

Out - Final answer: The washing machine allocation optimization problem and its Pyomo code have been stored in the
knowledge base at: code_examples/optimization/washing_machine_allocation_pyomo.md for future reuse.

Agent finished running. Waiting for next command...

Press Ctrl+C to exit.
```

---