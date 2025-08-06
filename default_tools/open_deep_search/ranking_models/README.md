## Semantic Search and Reranking

The OpenDeepSearch library provides a flexible framework for semantic search and document reranking. At its core is the `BaseSemanticSearcher` class, which can be extended to implement various reranking strategies.

### Creating Your Own Reranker

To implement your own reranker, simply inherit from `BaseSemanticSearcher` and implement the `_get_embeddings()` method:

```python
from opendeepsearch.ranking_models.base_reranker import BaseSemanticSearcher
import torch
class MyCustomReranker(BaseSemanticSearcher):
    def init(self):
        # Initialize your embedding model here
        super().__init__()
        self.model = YourEmbeddingModel()
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        # Implement your embedding logic here
        pass
    embeddings = self.model.encode(texts)
    return torch.tensor(embeddings)
```

The base class automatically handles:
- Similarity score calculation
- Score normalization (softmax, scaling, or none)
- Document reranking
- Top-k selection

### Using Infinity Rerankers

For high-performance reranking, we support [Infinity](https://github.com/michaelfeil/infinity) rerankers which offer state-of-the-art performance. To use an Infinity reranker, first start the Infinity server:

```bash
# requires ~16-32GB VRAM NVIDIA Compute Capability >= 8.0
docker run \
-v $PWD/data:/app/.cache --gpus "0" -p "7997":"7997" \
michaelf34/infinity:0.0.68-trt-onnx \
v2 --model-id Alibaba-NLP/gte-Qwen2-7B-instruct --revision "refs/pr/38" \
--dtype bfloat16 --batch-size 8 --device cuda --engine torch --port 7997 \
--no-bettertransformer
```

This will start an Infinity server using the Qwen2-7B-instruct model. The server will be available at `localhost:7997`.

Key parameters:
- `--model-id`: The Hugging Face model ID to use (see [supported models](https://github.com/michaelfeil/infinity#supported-tasks-and-models-by-infinity))
- `--dtype`: Data type for inference (bfloat16 recommended for modern GPUs)
- `--batch-size`: Batch size for inference
- `--port`: Port to expose the server on

For specialized deployments, Infinity provides several Docker images:
- `latest-cpu` - For CPU-only inference
- `latest-rocm` - For AMD ROCm GPUs
- `latest-trt-onnx` - For NVIDIA GPUs with TensorRT/ONNX optimizations

See the [Infinity documentation](https://michaelfeil.github.io/infinity/) for more details on deployment options and configuration.

Note: Ensure you have sufficient VRAM (16-32GB) and a compatible NVIDIA GPU (Compute Capability ≥ 8.0) before running the Infinity server.

### Using Jina AI (or API based) Rerankers

Jina AI provides powerful embedding models through their API service. The `JinaReranker` class offers a simple way to leverage these models:

```python
from opendeepsearch.ranking_models.jina_reranker import JinaReranker

# Initialize with your API key
reranker = JinaReranker(api_key="your_api_key")  # or set JINA_API_KEY env variable

# Example usage
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a type of machine learning",
    "Natural language processing uses machine learning"
]

# Get top 2 most relevant documents
results = reranker.search(query, documents, k=2)
```

The JinaReranker uses Jina's v3 embeddings by default, which provides:
- 1024-dimensional embeddings
- Optimized for text matching tasks
- State-of-the-art performance for semantic search

To use JinaReranker:
1. Sign up for a Jina AI API key at https://jina.ai
2. Either pass the API key directly or set it as an environment variable `JINA_API_KEY`
3. Optionally specify a different model using the `model` parameter

Note: Unlike Infinity rerankers which run locally, Jina rerankers require an internet connection and API credits.