from typing import Optional

from pydantic import BaseModel, Field
from src.hamlet.core.models import LiteLLMModel
from src.hamlet.core.agents import CodeAgent
from src.hamlet.core.monitoring import LogLevel
from src.hamlet.tools.file_editing.file_editing_tools import ListDir, SeeTextFile, ModifyFile, CreateFileWithContent
import json
import os

from dotenv import load_dotenv
load_dotenv()

model = LiteLLMModel(model_id="gpt-5-mini")

# working_dir = "./examples/simple_agent_workspace"
working_dir = "./examples"

os.makedirs(working_dir, exist_ok=True)
# agent = CodeAgent(model=model, tools=[
#     ListDir(working_dir=working_dir),
#     SeeTextFile(working_dir=working_dir),
#     ModifyFile(working_dir=working_dir),
#     CreateFileWithContent(working_dir=working_dir)
# ])

class Answer(BaseModel):
    description: str = Field(..., description="The description of the method")
    result: int = Field(..., description="The final result of the computation")

agent = CodeAgent(
    model=model, 
    tools=[
        ListDir(working_dir=working_dir),
        SeeTextFile(working_dir=working_dir),
        ModifyFile(working_dir=working_dir),
        CreateFileWithContent(working_dir=working_dir)
        ],
        output_schema=Answer,
    verbosity_level=LogLevel.DEBUG)

# full_res = agent.run("What is the capital of France? Write the answer to a file named capital.txt Then read the file to check if everything looks good. Then modify the file to add some descriptions of the capital city.", return_full_result=True)
# full_res = agent.run("What is the result of the following operation: 1 + 2 + 3 + ... + 99 + 100? Use more than one method to compute concurrently. Use the Check Method 'prompt'. And use the 'final_answer' tool in a new step if your answer pass the check.", return_full_result=True)
# full_res = agent.run("What is the result of the following operation: 1 + 2 + 3 + ... + 99 + 100? Use the 'final_answer' tool in the second step.", return_full_result=True)
output = agent.run("What is the result of the following operation: 1 + 2 + 3 + ... + 99 + 100?")
print("Final output:", output)
print("Type of final output:", type(output))

# write to json file

# with open("examples/test_parallel.json", "w") as f:
#     json.dump(full_res.steps, f, indent=2)

"""
ssh -vvv -N -R 7890:localhost:7890 -p 2228 lijinbo@166.111.59.11
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

uv sync
uv sync --extra tools
uv run python -m examples.simple_agent
"""