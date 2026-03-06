#!/usr/bin/env bash

# Standalone disaggregated benchmark with custom prompts.
# Starts prefill+decode vLLM servers and a proxy, verifies decode path, then benchmarks.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8000}"

PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"

KV_PORT_PREFILL="${KV_PORT_PREFILL:-14579}"
KV_PORT_DECODE="${KV_PORT_DECODE:-14580}"

DATASET_PATH="${DATASET_PATH:-benchmarks/disagg_benchmarks/data/disagg_kv_prompts_20.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
SKIP_CHAT_TEMPLATE="${SKIP_CHAT_TEMPLATE:-1}"

MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
BENCH_NO_STREAM="${BENCH_NO_STREAM:-0}"

RESULTS_DIR="${RESULTS_DIR:-benchmarks/disagg_benchmarks/results_disagg_custom}"
RESULT_FILENAME="${RESULT_FILENAME:-disagg_custom_none.json}"

PREFILL_LOG="${PREFILL_LOG:-/tmp/vllm_prefill_disagg.log}"
DECODE_LOG="${DECODE_LOG:-/tmp/vllm_decode_disagg.log}"
PROXY_LOG="${PROXY_LOG:-/tmp/vllm_proxy_disagg.log}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-420}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-60}"

PREFILL_PID=""
DECODE_PID=""
PROXY_PID=""

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

wait_for_http() {
  local url="$1"
  local timeout_s="${2:-300}"
  local child_pid="${3:-}"
  local label="${4:-service}"
  local log_file="${5:-}"
  local start
  local elapsed=0
  local last_report=0
  start="$(date +%s)"
  while true; do
    if curl -sS "${url}" >/dev/null 2>&1; then
      if (( elapsed > 0 )); then
        echo "Ready: ${label} (${url}) after ${elapsed}s"
      fi
      return 0
    fi
    if [[ -n "${child_pid}" ]] && ! kill -0 "${child_pid}" >/dev/null 2>&1; then
      echo "Process exited while waiting for ${label} (${url}), pid=${child_pid}" >&2
      return 1
    fi
    elapsed=$(( "$(date +%s)" - start ))
    if (( elapsed - last_report >= 10 )); then
      echo "Waiting for ${label} (${url})... ${elapsed}s elapsed"
      if [[ -n "${log_file}" ]] && [[ -f "${log_file}" ]]; then
        tail -n 3 "${log_file}" | sed 's/^/  /'
      fi
      last_report="${elapsed}"
    fi
    if (( elapsed > timeout_s )); then
      echo "Timed out waiting for ${label} (${url}) after ${timeout_s}s" >&2
      return 1
    fi
    sleep 1
  done
}

show_logs() {
  echo ""
  echo "===== ${PREFILL_LOG} (tail) ====="
  tail -n 80 "${PREFILL_LOG}" || true
  echo ""
  echo "===== ${DECODE_LOG} (tail) ====="
  tail -n 80 "${DECODE_LOG}" || true
  echo ""
  echo "===== ${PROXY_LOG} (tail) ====="
  tail -n 80 "${PROXY_LOG}" || true
}

cleanup() {
  for pid in "${PROXY_PID}" "${PREFILL_PID}" "${DECODE_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  PREFILL_PID=""
  DECODE_PID=""
  PROXY_PID=""
}

trap cleanup EXIT

require_cmd vllm
require_cmd curl
require_cmd python3
require_cmd nvidia-smi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

if ! GPU_LIST="$(nvidia-smi -L 2>/tmp/disagg_custom_nvidia.err)"; then
  echo "Failed to query GPUs with nvidia-smi" >&2
  tail -n 30 /tmp/disagg_custom_nvidia.err || true
  exit 1
fi
GPU_COUNT="$(printf "%s\n" "${GPU_LIST}" | wc -l | xargs)"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "This script expects at least 2 visible GPUs (prefill+decode). Found: ${GPU_COUNT}" >&2
  printf "%s\n" "${GPU_LIST}" >&2
  exit 1
fi

run_smoke_completion() {
  local label="$1"
  local url="$2"
  local timeout_s="$3"
  local payload="$4"
  local resp

  if ! resp="$(curl -sS --max-time "${timeout_s}" -X POST "${url}" \
    -H 'Content-Type: application/json' \
    -d "${payload}")"; then
    echo "${label} smoke test failed: timeout/error after ${timeout_s}s at ${url}" >&2
    return 1
  fi

  python3 - "${label}" "${resp}" <<'PY'
import json
import sys

label = sys.argv[1]
payload = json.loads(sys.argv[2])
if "choices" not in payload:
    raise SystemExit(f"{label} smoke test failed, unexpected response: {payload}")
print(f"{label} smoke test passed.")
PY
}

mkdir -p "${RESULTS_DIR}"
: >"${PREFILL_LOG}"
: >"${DECODE_LOG}"
: >"${PROXY_LOG}"

HOST_IP="$(hostname -I | awk '{print $1}')"
export VLLM_HOST_IP="${HOST_IP}"

PREFILL_KV_CFG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2,\"kv_buffer_size\":5e9,\"kv_ip\":\"${HOST_IP}\",\"kv_port\":${KV_PORT_PREFILL}}"
DECODE_KV_CFG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_rank\":1,\"kv_parallel_size\":2,\"kv_buffer_size\":5e9,\"kv_ip\":\"${HOST_IP}\",\"kv_port\":${KV_PORT_DECODE}}"

echo "Starting disaggregated stack (no transfer compression)..."
echo "  MODEL=${MODEL}"
echo "  PREFILL_GPU=${PREFILL_GPU}"
echo "  DECODE_GPU=${DECODE_GPU}"
echo "  PREFILL_PORT=${PREFILL_PORT}"
echo "  DECODE_PORT=${DECODE_PORT}"
echo "  PROXY_PORT=${PROXY_PORT}"
echo "  DATASET_PATH=${DATASET_PATH}"
echo "  NUM_PROMPTS=${NUM_PROMPTS}"
echo "  OUTPUT_LEN=${OUTPUT_LEN}"
echo "  RESULT=${RESULTS_DIR}/${RESULT_FILENAME}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${PREFILL_GPU}" vllm serve "${MODEL}" \
  --port "${PREFILL_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-transfer-config "${PREFILL_KV_CFG}" \
  >"${PREFILL_LOG}" 2>&1 &
PREFILL_PID="$!"
echo "  prefill pid=${PREFILL_PID} log=${PREFILL_LOG}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${DECODE_GPU}" vllm serve "${MODEL}" \
  --port "${DECODE_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-transfer-config "${DECODE_KV_CFG}" \
  >"${DECODE_LOG}" 2>&1 &
DECODE_PID="$!"
echo "  decode pid=${DECODE_PID} log=${DECODE_LOG}"

if ! wait_for_http "http://localhost:${PREFILL_PORT}/v1/models" "${STARTUP_TIMEOUT}" "${PREFILL_PID}" "prefill server" "${PREFILL_LOG}"; then
  show_logs
  exit 1
fi
if ! wait_for_http "http://localhost:${DECODE_PORT}/v1/models" "${STARTUP_TIMEOUT}" "${DECODE_PID}" "decode server" "${DECODE_LOG}"; then
  show_logs
  exit 1
fi

echo "Running direct backend sanity checks..."
if ! run_smoke_completion \
  "Prefill direct" \
  "http://localhost:${PREFILL_PORT}/v1/completions" \
  "${SMOKE_TIMEOUT}" \
  "{\"model\":\"${MODEL}\",\"prompt\":\"Prefill direct smoke test.\",\"max_tokens\":1,\"stream\":false}"; then
  show_logs
  exit 1
fi

if ! run_smoke_completion \
  "Decode direct" \
  "http://localhost:${DECODE_PORT}/v1/completions" \
  "${SMOKE_TIMEOUT}" \
  "{\"model\":\"${MODEL}\",\"prompt\":\"Decode direct smoke test.\",\"max_tokens\":1,\"stream\":false}"; then
  show_logs
  exit 1
fi

PYTHONUNBUFFERED=1 python3 benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
  --port "${PROXY_PORT}" \
  --prefill-url "http://localhost:${PREFILL_PORT}" \
  --decode-url "http://localhost:${DECODE_PORT}" \
  --kv-host "${HOST_IP}" \
  --prefill-kv-port "${KV_PORT_PREFILL}" \
  --decode-kv-port "${KV_PORT_DECODE}" \
  >"${PROXY_LOG}" 2>&1 &
PROXY_PID="$!"
echo "  proxy pid=${PROXY_PID} log=${PROXY_LOG}"

if ! wait_for_http "http://localhost:${PROXY_PORT}/v1/completions" 120 "${PROXY_PID}" "disagg proxy" "${PROXY_LOG}"; then
  show_logs
  exit 1
fi

echo "Running decode-path smoke test via proxy..."
if ! run_smoke_completion \
  "Proxy disagg decode-path" \
  "http://localhost:${PROXY_PORT}/v1/completions" \
  "${SMOKE_TIMEOUT}" \
  "{\"model\":\"${MODEL}\",\"prompt\":\"Decode path smoke test.\",\"max_tokens\":1,\"stream\":false}"; then
  echo "Proxy decode-path smoke test failed." >&2
  show_logs
  exit 1
fi

dataset_args=(
  --dataset-name custom
  --dataset-path "${DATASET_PATH}"
  --custom-output-len "${OUTPUT_LEN}"
)
if [[ "${SKIP_CHAT_TEMPLATE}" == "1" ]]; then
  dataset_args+=(--skip-chat-template)
fi

stream_args=()
if [[ "${BENCH_NO_STREAM}" == "1" ]]; then
  stream_args+=(--no-stream)
fi

echo "Running disaggregated benchmark..."
vllm bench serve \
  --backend vllm \
  --model "${MODEL}" \
  --endpoint /v1/completions \
  --port "${PROXY_PORT}" \
  "${dataset_args[@]}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --request-rate "${REQUEST_RATE}" \
  "${stream_args[@]}" \
  --save-result \
  --result-dir "${RESULTS_DIR}" \
  --result-filename "${RESULT_FILENAME}"

python3 - "${RESULTS_DIR}/${RESULT_FILENAME}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
if not os.path.exists(path):
    print(f"Result file missing: {path}", file=sys.stderr)
    sys.exit(1)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("")
print("Disaggregated benchmark summary (compression=none)")
print(f"  request_throughput: {data.get('request_throughput', 0.0):.4f} req/s")
print(f"  output_throughput : {data.get('output_throughput', 0.0):.4f} tok/s")
print(f"  mean_ttft_ms      : {data.get('mean_ttft_ms', 0.0):.2f}")
print(f"  mean_itl_ms       : {data.get('mean_itl_ms', 0.0):.2f}")
print(f"  mean_e2el_ms      : {data.get('mean_e2el_ms', 0.0):.2f}")
print(f"  result_json       : {path}")
PY
