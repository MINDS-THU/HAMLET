# !pip install smolagents[litellm]
from src.agents import CodeAgent
from src.models import LiteLLMModel
from src.monitoring import LogLevel
from src.default_tools import DuckDuckGoSearchTool, VisitWebpageTool

openai_api_key = 

# Then we run the agentic part!
model = LiteLLMModel(model_id="gpt-4o-mini", api_key=openai_api_key) # Could use 'gpt-4o'

search_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
    verbosity_level=LogLevel.DEBUG
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
    verbosity_level=LogLevel.DEBUG
)
manager_agent.run("If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?")
