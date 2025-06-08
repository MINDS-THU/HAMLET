from smolagents import CodeAgent, LiteLLMModel
from src.custom_gradio_ui import GradioUI
from general_tools.file_editing.file_editing_tools import (
    ListDir,
    SeeFile,
    ModifyFile,
    DeleteFileOrFolder,
    CreateFileWithContent,
)  
from dotenv import load_dotenv
load_dotenv()



agent = CodeAgent(
    model=LiteLLMModel(model_id="gpt-4.1"),
    tools=[
        ListDir("./data"),
        SeeFile("./data"),
        ModifyFile("./data"),
        DeleteFileOrFolder("./data"),
        CreateFileWithContent("./data"),
    ],
    verbosity_level=1,
    # planning_interval=3,
    name="example_agent",
    description="This is an example agent.",
    step_callbacks=[],
)

GradioUI(agent, file_upload_folder="./data").launch()