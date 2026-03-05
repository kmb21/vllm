#!/usr/bin/env bash

# Strict A/B/C benchmark for KV transfer compression in disaggregated P/D serving.
#
# Runs identical workload for each compression mode and reports:
# - request throughput
# - output throughput
# - TTFT / ITL / E2EL
#
# Default modes: none, fp8, int8
#
# Example:
#   bash benchmarks/disagg_benchmarks/ab_kv_transfer_compression.sh
#
# Tunable env vars:
#   MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#   NUM_PROMPTS=200
#   INPUT_LEN=4096
#   OUTPUT_LEN=16
#   MAX_CONCURRENCY=16
#   REQUEST_RATE=inf
#   MODES="none fp8 int8"
#   RESULTS_DIR="benchmarks/disagg_benchmarks/results_kv_transfer_ab"
#   KV_PORT_PREFILL=14579
#   KV_PORT_DECODE=14580

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MODES="${MODES:-none fp8 int8}"
RESULTS_DIR="${RESULTS_DIR:-benchmarks/disagg_benchmarks/results_kv_transfer_ab}"
KV_PORT_PREFILL="${KV_PORT_PREFILL:-14579}"
KV_PORT_DECODE="${KV_PORT_DECODE:-14580}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"

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
  local start
  start="$(date +%s)"
  while true; do
    if curl -sS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start > timeout_s )); then
      echo "Timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

show_startup_logs() {
  local mode="$1"
  echo ""
  echo "===== /tmp/vllm_prefill_${mode}.log (tail) ====="
  tail -n 80 "/tmp/vllm_prefill_${mode}.log" || true
  echo ""
  echo "===== /tmp/vllm_decode_${mode}.log (tail) ====="
  tail -n 80 "/tmp/vllm_decode_${mode}.log" || true
  echo ""
  echo "===== /tmp/vllm_proxy_${mode}.log (tail) ====="
  tail -n 80 "/tmp/vllm_proxy_${mode}.log" || true
}

cleanup_children() {
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

trap cleanup_children EXIT

make_kv_config() {
  local role="$1"
  local rank="$2"
  local mode="$3"

  if [[ "${mode}" == "none" ]]; then
    cat <<EOF
{"kv_connector":"P2pNcclConnector","kv_role":"${role}","kv_rank":${rank},"kv_parallel_size":2,"kv_buffer_size":5e9}
EOF
  else
    cat <<EOF
{"kv_connector":"P2pNcclConnector","kv_role":"${role}","kv_rank":${rank},"kv_parallel_size":2,"kv_buffer_size":5e9,"kv_transfer_compression":"${mode}","kv_transfer_scale_sharing":"per_tensor"}
EOF
  fi
}

start_disagg_stack() {
  local mode="$1"
  local host_ip
  host_ip="$(hostname -I | awk '{print $1}')"
  export VLLM_HOST_IP="${host_ip}"

  local prefill_cfg decode_cfg
  prefill_cfg="$(make_kv_config kv_producer 0 "${mode}")"
  decode_cfg="$(make_kv_config kv_consumer 1 "${mode}")"

  echo "Starting prefill server (${mode})..."
  CUDA_VISIBLE_DEVICES="${PREFILL_GPU}" vllm serve "${MODEL}" \
    --port "${PREFILL_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.70 \
    --kv-transfer-config "${prefill_cfg}" \
    >/tmp/vllm_prefill_"${mode}".log 2>&1 &
  PREFILL_PID="$!"

  echo "Starting decode server (${mode})..."
  CUDA_VISIBLE_DEVICES="${DECODE_GPU}" vllm serve "${MODEL}" \
    --port "${DECODE_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.70 \
    --kv-transfer-config "${decode_cfg}" \
    >/tmp/vllm_decode_"${mode}".log 2>&1 &
  DECODE_PID="$!"

  if ! wait_for_http "http://localhost:${PREFILL_PORT}/v1/models" 300; then
    show_startup_logs "${mode}"
    return 1
  fi
  if ! wait_for_http "http://localhost:${DECODE_PORT}/v1/models" 300; then
    show_startup_logs "${mode}"
    return 1
  fi

  echo "Starting disagg proxy..."
  python3 benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
    --port "${PROXY_PORT}" \
    --prefill-url "http://localhost:${PREFILL_PORT}" \
    --decode-url "http://localhost:${DECODE_PORT}" \
    --kv-host "${host_ip}" \
    --prefill-kv-port "${KV_PORT_PREFILL}" \
    --decode-kv-port "${KV_PORT_DECODE}" \
    >/tmp/vllm_proxy_"${mode}".log 2>&1 &
  PROXY_PID="$!"

  if ! wait_for_http "http://localhost:${PROXY_PORT}/v1/completions" 120; then
    show_startup_logs "${mode}"
    return 1
  fi
}

run_bench() {
  local mode="$1"
  local out_file="${RESULTS_DIR}/${mode}.json"
  echo "Running benchmark for mode=${mode}"

  vllm bench serve \
    --backend vllm \
    --model "${MODEL}" \
    --endpoint /v1/completions \
    --port "${PROXY_PORT}" \
    --dataset-name random \
    --num-prompts "${NUM_PROMPTS}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${REQUEST_RATE}" \
    --save-result \
    --result-dir "${RESULTS_DIR}" \
    --result-filename "$(basename "${out_file}")"
}

print_summary() {
  python3 - "$RESULTS_DIR" $MODES <<'PY'
import json
import os
import sys

results_dir = sys.argv[1]
modes = sys.argv[2:]

rows = []
for mode in modes:
    path = os.path.join(results_dir, f"{mode}.json")
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows.append(
        {
            "mode": mode,
            "req/s": data.get("request_throughput", 0.0),
            "tok/s": data.get("output_throughput", 0.0),
            "ttft_ms": data.get("mean_ttft_ms", 0.0),
            "itl_ms": data.get("mean_itl_ms", 0.0),
            "e2el_ms": data.get("mean_e2el_ms", 0.0),
        }
    )

if not rows:
    print("No result files found.")
    sys.exit(1)

base = next((r for r in rows if r["mode"] == "none"), rows[0])

print("")
print("KV Transfer Compression A/B Summary")
print("mode    req/s    tok/s    ttft_ms   itl_ms    e2el_ms   d_req%   d_ttft%")
for r in rows:
    d_req = ((r["req/s"] - base["req/s"]) / base["req/s"] * 100.0) if base["req/s"] else 0.0
    d_ttft = ((r["ttft_ms"] - base["ttft_ms"]) / base["ttft_ms"] * 100.0) if base["ttft_ms"] else 0.0
    print(
        f'{r["mode"]:<7s}'
        f'{r["req/s"]:>7.2f}  '
        f'{r["tok/s"]:>7.2f}  '
        f'{r["ttft_ms"]:>8.2f}  '
        f'{r["itl_ms"]:>7.2f}  '
        f'{r["e2el_ms"]:>8.2f}  '
        f'{d_req:>7.2f}  '
        f'{d_ttft:>8.2f}'
    )
print("")
print(f"Raw JSON results: {results_dir}")
PY
}

require_cmd vllm
require_cmd python3
require_cmd curl
require_cmd nvidia-smi

if ! GPU_LIST="$(nvidia-smi -L 2>/tmp/kv_ab_nvidia_smi.err)"; then
  echo "Failed to query GPUs with nvidia-smi."
  tail -n 20 /tmp/kv_ab_nvidia_smi.err || true
  exit 1
fi
GPU_COUNT="$(printf "%s\n" "${GPU_LIST}" | wc -l | xargs)"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "This script needs 2 visible GPUs. Found: ${GPU_COUNT}."
  echo "Use a 2-GPU environment, or run with PREFILL_GPU and DECODE_GPU pointing to two valid devices."
  echo "Visible GPUs:"
  printf "%s\n" "${GPU_LIST}"
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "Benchmark setup:"
echo "  MODEL=${MODEL}"
echo "  NUM_PROMPTS=${NUM_PROMPTS}"
echo "  INPUT_LEN=${INPUT_LEN}"
echo "  OUTPUT_LEN=${OUTPUT_LEN}"
echo "  MAX_CONCURRENCY=${MAX_CONCURRENCY}"
echo "  REQUEST_RATE=${REQUEST_RATE}"
echo "  MODES=${MODES}"
echo "  RESULTS_DIR=${RESULTS_DIR}"
echo ""

for mode in ${MODES}; do
  cleanup_children
  start_disagg_stack "${mode}"
  run_bench "${mode}"
done

cleanup_children
print_summary
