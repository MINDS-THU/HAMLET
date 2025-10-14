from src.hamlet.core.models import LiteLLMModel
from src.hamlet.core.agents import CodeAgent
from src.hamlet.tools.file_editing.file_editing_tools import ListDir, SeeTextFile, ModifyFile, CreateFileWithContent
import json
import os

from dotenv import load_dotenv
load_dotenv()

model = LiteLLMModel(model_id="gpt-4.1", temperature=0)
working_dir = "./examples/simple_agent_workspace"
os.makedirs(working_dir, exist_ok=True)
agent = CodeAgent(model=model, tools=[
    ListDir(working_dir=working_dir),
    SeeTextFile(working_dir=working_dir),
    ModifyFile(working_dir=working_dir),
    CreateFileWithContent(working_dir=working_dir)
])
full_res = agent.run("What is the capital of France? Write the answer to a file named capital.txt Then read the file to check if everything looks good. Then modify the file to add some descriptions of the capital city.", return_full_result=True)

# print("Full result:")
# print(full_res)

# write to json file

with open("examples/simple_agent_result.json", "w") as f:
    json.dump(full_res.steps, f, indent=2)