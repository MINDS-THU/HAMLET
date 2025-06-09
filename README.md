## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/cyrilli/COOPA.git
cd COOPA
```

### 2. Install Dependencies
Please refer to the installation instructions in readme file of the app you want to use.

### 3. Configure Environment Variables

Create a `.env` file in the root of `COOPA` and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key
# Optionally, add other keys as needed for the app you want to use
```
### 4. Run an App
```
python -m apps.literature_survey.run
```

---