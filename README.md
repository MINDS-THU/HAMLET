# HAMLET (Hierarchical Agents for Multi-level Learning, Execution & Tasking)

HAMLET is a framework built on top of smolagents for creating and training hierarchical language model agents that decompose and solve complex tasks through multi-level planning.

---

## Default Tools
The following tools are provided to enable LLM agent the ability to read and edit local files, maintain external memory, browse the Internet, etc.
- **[File Editing Tools](../../general_tools/file_editing/file_editing_tools.py)**: Manage reading, writing, and modifying files in the working directory.
- **[Knowledge Base Tools](../../general_tools/kb_repo_management/)**: Index, search, and maintain the local knowledge repository.

Tools from MCP servers are also supported via smolagents' [MCPClient](https://huggingface.co/docs/smolagents/en/tutorials/tools#use-tools-from-an-mcp-server) class.
---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MINDS-THU/HAMLET.git
cd HAMLET
```

### 2. Install Dependencies

- **Python packages** (conda installation recommended):

From the root of `HAMLET`, execute:

```bash
conda create -n hamlet_env python=3.10
conda activate hamlet_env
pip install -r requirements.txt
conda install -c conda-forge pandoc
```

### 3. Register for Required API Keys

Some agents use external services for web search and language models. Register and obtain API keys for:

- **Serper API** (Google Search): [https://serper.dev/](https://serper.dev/)
- **OpenAI API** (or compatible LLM provider): [https://platform.openai.com/](https://platform.openai.com/)
- **Jina API** (for reranking): [https://jina.ai/](https://jina.ai/)
- **HF_TOKEN** (for Hugging Face model access or Gradio deployment): [https://huggingface.co/](https://huggingface.co/)

### 4. Configure Environment Variables

Create a `.env` file in the root of `HAMLET` and add your API keys:

```
SERPER_API_KEY=your_serper_api_key
OPENAI_API_KEY=your_openai_api_key
JINA_API_KEY=your_jina_api_key
HF_TOKEN=your_huggingface_token  # Optional: for Hugging Face model access or Gradio deployment
# Optionally, add other keys as needed
```

---