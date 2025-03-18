# !pip install smolagents[litellm]
from src import CodeAgent, ToolCallingAgent, LiteLLMModel, LogLevel

openai_api_key = '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'
model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True, verbosity_level=LogLevel.DEBUG)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)