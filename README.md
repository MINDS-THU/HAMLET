## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/cyrilli/COOPA.git
cd COOPA
```

### 2. Install Dependencies
From root of `COOPA`, execute
```bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables

Create a `.env` file in the root of `COOPA_v3` and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key
# Optionally, add other keys as needed
```
### 4. Run an App
```
python -m apps.literature_survey.run
```

---
Features to add:
- add compatibility with `.pdf`, `.html` files for `kb_repo_management/repo_indexer.py`
- structured output specification and validation for `CodeAgent` and `ToolCallingAgent`, e.g. using [instructor](https://github.com/567-labs/instructor)
- better experience retrieval and reuse, e.g. using [muscle-mem](https://github.com/567-labs/instructor)
