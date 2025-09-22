from typing import Optional, Literal, List
import asyncio
from smolagents import Tool
from .ods_agent import OpenDeepSearchAgent

class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your queries (think a Google search) then returns the final answer that is processed by an llm.
    Set quick_mode=True to skip scraping/reranking/LLM and return only top links and snippets from the search engine (fast discovery mode)."""
    inputs = {
        "queries": {
            "type": "any",
            "description": (
                "A list of search query strings to run (parallelized internally). "
                "Example: ['LLM automate discrete-event simulation construction 2021..2024 site:arxiv.org OR site:openreview.net OR site:proceedings.mlr.press OR filetype:pdf. Please include direct links to the papers and PDFs.', "
                "'Large language models for simulation automation survey 2021..2024. Please include direct links to the papers and PDFs.'] "
                "Note: Passing a single multi-line string with multiple numbered queries will be treated as ONE query. "
                "To search multiple queries in parallel, pass a Python list of separate strings."
            ),
        },
    }
    output_type = "string"

    def __init__(
        self,
        max_queries: int = 1,
        model_name: str = "gpt-5",
        quick_mode: bool = True,
        max_results: int = 5,
        reranker: str = "infinity",
        timeout: int = 120, # timeout in seconds for each call
        search_provider: Literal["serper", "searxng"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None
    ):
        # "quick_mode": {
        #     "type": "boolean",
        #     "description": "When true, skip scraping/reranking/LLM and return a formatted list of top links and snippets (faster).",
        #     "nullable": True
        # },
        # "max_results": {
        #     "type": "integer",
        #     "description": "Number of top results to return per query in quick_mode (default 5).",
        #     "nullable": True
        # },
        super().__init__()
        self.quick_mode = quick_mode
        self.max_results = max_results
        self.max_queries = max_queries
        self.search_model_name = model_name  # LiteLLM model name
        self.reranker = reranker
        self.timeout = timeout
        self.search_provider: Literal["serper", "searxng"] = search_provider
        self.serper_api_key = serper_api_key
        self.searxng_instance_url = searxng_instance_url
        self.searxng_api_key = searxng_api_key

    def forward(self, queries: List[str]):  # type: ignore[override]
        print("===============================================")
        print("===============================================")
        print("===============================================")
        output = ""
        if not queries:
            return "No queries provided."
        if type(queries) is str:
            queries = [queries]
        if len(queries) > self.max_queries:
            output += f"{len(queries)} queries are provided, which exceeds the maximum allowed of {self.max_queries}. The rest will be ignored for now.\n"
            queries = queries[:self.max_queries]
            
        async def run_all():
            # Quick mode: SERP only (links and snippets), no scraping/reranking/LLM
            if self.quick_mode:
                async def run_serp(q: str):
                    # Offload blocking SERP call to a thread and apply timeout per query
                    try:
                        res = await asyncio.wait_for(
                            asyncio.to_thread(self.search_tool.serp_search.get_sources, q),
                            timeout=self.timeout,
                        )
                        return res
                    except Exception as e:
                        return e

                tasks = [run_serp(q) for q in queries]
                return await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Full mode: build context and LLM answer per query
                tasks = [
                    asyncio.wait_for(self.search_tool.ask(q, max_sources=3), timeout=self.timeout)
                    for q in queries
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)
        # --- Async execution strategy ---
        # Previous implementation used get_event_loop + run_until_complete, which fails
        # in worker threads (no loop set) producing: RuntimeError: There is no current event loop.
        # For a Flask / threaded production context the safest, least-surprising approach
        # is to spin up a fresh event loop per call via asyncio.run(). This avoids nested
        # loop patching (nest_asyncio) and prevents cross-thread loop reuse issues.
        # If we are somehow already inside a running loop (very rare for this sync Tool.forward),
        # we fallback to creating a new loop manually because asyncio.run() cannot be called
        # from an existing running loop.
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            # Create an isolated loop to avoid nest_asyncio complexity and potential deadlocks.
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                results = new_loop.run_until_complete(run_all())
            finally:
                new_loop.close()
                asyncio.set_event_loop(running_loop)
        else:
            # No running loop in this thread: simplest & safe.
            results = asyncio.run(run_all())
        # Format output, with indexing for clarity
        for i, (query, result) in enumerate(zip(queries, results), 1):
            # Handle timeouts/exceptions uniformly
            if isinstance(result, asyncio.TimeoutError):
                output += (
                    f"Query {i}: {query}\n"
                    f"Result {i}: Timeout after {self.timeout}s while searching. Try simplifying the query or reducing sources.\n\n"
                )
                continue
            if isinstance(result, Exception):
                output += (
                    f"Query {i}: {query}\n"
                    f"Result {i}: Error: {type(result).__name__}: {result}\n\n"
                )
                continue

            if self.quick_mode:
                # result is a SearchResult from SERP
                try:
                    failed = bool(getattr(result, "failed", False))
                    if failed:
                        err = getattr(result, "error", "Unknown error")
                        output += f"Query {i}: {query}\nResult {i}: SERP error: {err}\n\n"
                        continue
                    data = getattr(result, "data", {}) or {}
                    organic = data.get("organic", []) or []
                    formatted = [
                        f"  {j}. {item.get('title','').strip()}\n     url: {item.get('link','').strip()}\n     snippet: {item.get('snippet','').strip()}"
                        for j, item in enumerate(organic[: max(1, self.max_results) ], 1)
                    ]
                    body = "\n".join(formatted) if formatted else "  (no results)"
                    output += f"Query {i}: {query}\nTop results:\n{body}\n\n"
                except Exception as e:
                    output += f"Query {i}: {query}\nResult {i}: Formatting error: {e}\n\n"
            else:
                # Full mode result is an LLM string
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