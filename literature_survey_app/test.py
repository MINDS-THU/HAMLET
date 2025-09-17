import sys
import os

# Add src/ to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dotenv import load_dotenv
sys.path.insert(0, os.getcwd())
from default_tools.open_deep_search.ods_tool import OpenDeepSearchTool

# Load environment variables
load_dotenv(override=True)



# set up ods search tool
search_tool = OpenDeepSearchTool(
    max_queries=3,
    model_name="gpt-5",
    reranker="jina"
)
if not search_tool.is_initialized:
    search_tool.setup()

res= search_tool.forward(
    [
        "Help me find a survey paper that summarizes recent research on reinforcement learning for finetuning LLMs. I need link to the paper.",
        "Are there any papers on the topic of bayesian interpretation of in-context learning of LLMs? I need link to the paper.",
        "Are there any papers on the topic of retrieval-augmented generation (RAG) for LLMs? I need link to the paper."
    ],
    quick_mode=True,
    max_results=5
)
print(res)

# urls = [
#     "https://arxiv.org/html/2412.10400v1",
#     "https://openreview.net/forum?id=inpkC8UrDu",
#     "https://raw.githubusercontent.com/mlresearch/v258/main/assets/zhang25d/zhang25d.pdf",
#     "https://arxiv.org/html/2306.04891v2",
# ]

# # Initialize RAG tool
# # rag = RAG(pdf_directory=None, urls=urls, model_name="gpt-4.1")
# # rag_tool.forward("bayesian interpretation of in-context learning", urls=urls)

# new_docs = extract_docs_from_urls(urls)
# print(new_docs)
# for doc in new_docs:
#     title = doc.metadata.get('title')
#     # Fallback logic for missing title in doc metadata
#     if not title or not title.strip():
#         # Try to use first non-empty line of content
#         for line in doc.page_content.splitlines():
#             if line.strip():
#                 title = line.strip()
#                 break
#         else:
#             title = 'Untitled Document'
#     filename = re.sub(r'\W+', '_', title).lower() + '.txt'
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write(doc.page_content)