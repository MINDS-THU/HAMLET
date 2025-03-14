# from openinference.instrumentation.smolagents import SmolagentsInstrumentor
# from phoenix.otel import register


# register()
# SmolagentsInstrumentor().instrument(skip_dep_check=True)

from src import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
    VisitWebpageTool,
)


# Then we run the agentic part!
model = LiteLLMModel(model_id="gpt-4o-mini", api_key='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') # Could use 'gpt-4o'

search_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)
manager_agent.run("If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?")
