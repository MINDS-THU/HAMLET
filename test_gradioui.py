from src import (
    load_tool,
    CodeAgent,
    LiteLLMModel,
    GradioUI
)

openai_api_key = 
# Import tool from Hub
image_generation_tool = load_tool(repo_id="m-ric/text-to-image", trust_remote_code=True)

model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'

# Initialize the agent with the image generation tool
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()
