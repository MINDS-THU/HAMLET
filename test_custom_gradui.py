from smolagents import CodeAgent, LiteLLMModel
from custom_gradio_ui import GradioUI
from general_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    DeleteFileOrFolder,
    CreateFileWithContent,
)  
from dotenv import load_dotenv
load_dotenv()

# create temp_working_dir/ folder
import os
if not os.path.exists("./temp_working_dir"):
    os.makedirs("./temp_working_dir")

agent = CodeAgent(
    model=LiteLLMModel(model_id="gpt-4.1"),
    tools=[
        ListDir("./temp_working_dir"),
        SeeFile("./temp_working_dir"),
        ModifyFile("./temp_working_dir"),
        DeleteFileOrFolder("./temp_working_dir"),
        CreateFileWithContent("./temp_working_dir"),
    ],
    verbosity_level=1,
    name="example_agent",
    description="This is an example agent.",
    step_callbacks=[],
)

GradioUI(agent, file_upload_folder="./temp_working_dir").launch()