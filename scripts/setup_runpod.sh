#!/bin/bash
set -euo pipefail

# Environment overrides:
VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm}"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$VLLM_ROOT/.venv}"

cd "$VLLM_ROOT"

# Create and activate the virtual environment.
$PYTHON -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade packaging tools.
pip install --upgrade pip setuptools wheel

# Install torch aligned with CUDA 12.8 and the editable repo.
pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Optional HTTP/runtime helpers for benchmarks and instrumentation.
pip install fastapi==0.135.1 httpx uvicorn[standard] redis

# Capture exact dependency list.
pip freeze | tee "$VENV_DIR/requirements.txt"

cat <<'EOF'
Run it on your Runpod instance with bash (bash scripts/setup_runpod.sh), then source .venv/bin/activate before running benchmarks.
To run inference after sourcing "$VENV_DIR/bin/activate":

export MODEL_NAME=llama-3-70b
python -m vllm entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --port 8100 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size 1

curl http://localhost:8100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "'"$MODEL_NAME"'", "prompt": "Translate to French: The benchmark is running.", "max_tokens": 32, "temperature": 0.2}'
EOF
