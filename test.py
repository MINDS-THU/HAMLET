# !pip install smolagents[litellm]
from src import CodeAgent, ToolCallingAgent, LiteLLMModel, LogLevel

openai_api_key = 
model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True, verbosity_level=LogLevel.DEBUG)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
