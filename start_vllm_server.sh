#!/bin/bash
# Script to start vLLM server for GRPO training
#
# Usage:
#   ./start_vllm_server.sh [MODEL_PATH] [PORT]
#
# Example:
#   ./start_vllm_server.sh /path/to/model 8000
#
# Environment variables:
#   MODEL_PATH: Path to the model (default: from environment or /mnt/d/Qwen)
#   PORT: Server port (default: 8000)
#   TENSOR_PARALLEL_SIZE: Number of GPUs for tensor parallelism (default: 1)
#   GPU_MEMORY_UTILIZATION: GPU memory utilization (default: 0.9)

set -e

# Default values
MODEL_PATH="${1:-${MODEL_PATH:-/mnt/d/Qwen}}"
PORT="${2:-${PORT:-8000}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "=========================================="

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable or provide it as first argument"
    exit 1
fi

# Check if config.json exists in model path
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Warning: config.json not found in $MODEL_PATH"
    echo "Attempting to find model directory..."
    # Try to find config.json in subdirectories
    FOUND_CONFIG=$(find "$MODEL_PATH" -maxdepth 3 -name "config.json" -type f | head -1)
    if [ -n "$FOUND_CONFIG" ]; then
        MODEL_PATH=$(dirname "$FOUND_CONFIG")
        echo "Found model at: $MODEL_PATH"
    else
        echo "Error: Could not find config.json in $MODEL_PATH or subdirectories"
        exit 1
    fi
fi

# Start vLLM server using hamlet's vllm_server module
echo ""
echo "Starting server..."
echo "Press Ctrl+C to stop the server"
echo ""

uv run python -m hamlet.train.inference.vllm_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host "0.0.0.0" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code

