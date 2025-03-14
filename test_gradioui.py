from src import (
    load_tool,
    CodeAgent,
    LiteLLMModel,
    GradioUI
)

openai_api_key = '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'
# Import tool from Hub
image_generation_tool = load_tool(repo_id="m-ric/text-to-image", trust_remote_code=True)

model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'

# Initialize the agent with the image generation tool
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()