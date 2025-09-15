from typing import Optional, Literal, List
import asyncio
from smolagents import Tool
from .ods_agent import OpenDeepSearchAgent

class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your queries (think a Google search) then returns the final answer that is processed by an llm."""
    inputs = {
        "queries": {
            "type": "any",
            "description": "The list of search queries to perform",
        },
    }
    output_type = "string"

    def __init__(
        self,
        max_queries: int = 1,
        model_name: Optional[str] = None,
        reranker: str = "infinity",
        search_provider: Literal["serper", "searxng"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None
    ):
        super().__init__()
        self.max_queries = max_queries
        self.search_model_name = model_name  # LiteLLM model name
        self.reranker = reranker
        self.search_provider = search_provider
        self.serper_api_key = serper_api_key
        self.searxng_instance_url = searxng_instance_url
        self.searxng_api_key = searxng_api_key

    def forward(self, queries: List[str]):
        output = ""
        if not queries:
            return "No queries provided."
        if len(queries) > self.max_queries:
            output += f"{len(queries)} queries are provided, which exceeds the maximum allowed of {self.max_queries}. The rest will be ignored for now.\n"
            queries = queries[:self.max_queries]
            
        async def run_all():
            tasks = [self.search_tool.ask(q) for q in queries]
            return await asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        results = loop.run_until_complete(run_all())
        # Format output, with indexing for clarity
        for i, (query, result) in enumerate(zip(queries, results), 1):
            output += f"Query {i}: {query}\nResult {i}: {result}\n\n"
        return output.strip()

    def setup(self):
        self.search_tool = OpenDeepSearchAgent(
            self.search_model_name,
            reranker=self.reranker,
            search_provider=self.search_provider,
            serper_api_key=self.serper_api_key,
            searxng_instance_url=self.searxng_instance_url,
            searxng_api_key=self.searxng_api_key
        )