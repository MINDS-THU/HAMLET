# !pip install smolagents[litellm]
from src import CodeAgent, ToolCallingAgent, LiteLLMModel, LogLevel

openai_api_key = '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'
model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True, verbosity_level=LogLevel.DEBUG)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)