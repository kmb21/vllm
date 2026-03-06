#!/usr/bin/env python3

"""Disaggregated prefill/decode benchmark runner for custom prompts.

Flow:
1) Start prefill and decode vLLM servers with P2P KV transfer (no compression).
2) Run direct smoke tests against each backend.
3) Start proxy and run disagg decode-path smoke test through proxy.
4) Run `vllm bench serve` against proxy with custom dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _prepend_pythonpath(env: dict[str, str], path: str) -> None:
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{path}:{existing}" if existing else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disaggregated custom benchmark (prefill+decode+proxy)."
    )
    parser.add_argument("--model", default=_env_str("MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--max-model-len", type=int, default=_env_int("MAX_MODEL_LEN", 2048))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_env_float("GPU_MEMORY_UTILIZATION", 0.70),
    )
    parser.add_argument("--prefill-port", type=int, default=_env_int("PREFILL_PORT", 8100))
    parser.add_argument("--decode-port", type=int, default=_env_int("DECODE_PORT", 8200))
    parser.add_argument("--proxy-port", type=int, default=_env_int("PROXY_PORT", 8000))
    parser.add_argument("--prefill-gpu", type=int, default=_env_int("PREFILL_GPU", 0))
    parser.add_argument("--decode-gpu", type=int, default=_env_int("DECODE_GPU", 1))
    parser.add_argument("--kv-port-prefill", type=int, default=_env_int("KV_PORT_PREFILL", 14579))
    parser.add_argument("--kv-port-decode", type=int, default=_env_int("KV_PORT_DECODE", 14580))
    parser.add_argument(
        "--dataset-path",
        default=_env_str(
            "DATASET_PATH",
            "benchmarks/disagg_benchmarks/data/disagg_kv_prompts_200.jsonl",
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=_env_int("NUM_PROMPTS", 200))
    parser.add_argument("--output-len", type=int, default=_env_int("OUTPUT_LEN", 8))
    parser.add_argument("--skip-chat-template", action="store_true", default=_env_int("SKIP_CHAT_TEMPLATE", 1) == 1)
    parser.add_argument("--max-concurrency", type=int, default=_env_int("MAX_CONCURRENCY", 8))
    parser.add_argument("--request-rate", default=_env_str("REQUEST_RATE", "inf"))
    parser.add_argument("--bench-no-stream", action="store_true", default=_env_int("BENCH_NO_STREAM", 0) == 1)
    parser.add_argument(
        "--results-dir",
        default=_env_str("RESULTS_DIR", "benchmarks/disagg_benchmarks/results_disagg_custom"),
    )
    parser.add_argument(
        "--result-filename",
        default=_env_str("RESULT_FILENAME", ""),
    )
    parser.add_argument(
        "--kv-transfer-compression",
        choices=["none", "int8", "fp8"],
        default=_env_str("KV_TRANSFER_COMPRESSION", "none"),
        help="KV transfer compression mode for producer/consumer.",
    )
    parser.add_argument(
        "--kv-transfer-scale-sharing",
        default=_env_str("KV_TRANSFER_SCALE_SHARING", "per_tensor"),
        help="Scale-sharing mode when compression is enabled.",
    )
    parser.add_argument(
        "--fp8-transit-storage-path",
        default=_env_str("FP8_TRANSIT_STORAGE_PATH", "/tmp/kv_transit"),
        help="Shared storage path used by FP8TransitConnector.",
    )
    parser.add_argument("--startup-timeout", type=int, default=_env_int("STARTUP_TIMEOUT", 420))
    parser.add_argument("--smoke-timeout", type=int, default=_env_int("SMOKE_TIMEOUT", 60))
    parser.add_argument(
        "--direct-handoff-smoke",
        action="store_true",
        default=_env_int("DIRECT_HANDOFF_SMOKE", 0) == 1,
        help="Run direct prefill->decode handoff smoke before starting proxy.",
    )
    parser.add_argument("--prefill-log", default=_env_str("PREFILL_LOG", "/tmp/vllm_prefill_disagg.log"))
    parser.add_argument("--decode-log", default=_env_str("DECODE_LOG", "/tmp/vllm_decode_disagg.log"))
    parser.add_argument("--proxy-log", default=_env_str("PROXY_LOG", "/tmp/vllm_proxy_disagg.log"))
    parser.add_argument(
        "--save-responses",
        action="store_true",
        default=_env_int("SAVE_RESPONSES", 0) == 1,
        help="Save per-prompt model outputs to JSONL for accuracy comparison.",
    )
    parser.add_argument(
        "--responses-filename",
        default=_env_str("RESPONSES_FILENAME", ""),
        help="Optional output filename for saved responses JSONL.",
    )
    return parser.parse_args()


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.procs: list[subprocess.Popen] = []

    def cleanup(self) -> None:
        for p in reversed(self.procs):
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 8
        for p in reversed(self.procs):
            if p.poll() is None:
                timeout = max(0.0, deadline - time.time())
                try:
                    p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.procs.clear()

    def _start_proc(self, cmd: list[str], env: dict[str, str], log_path: str) -> subprocess.Popen:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("", encoding="utf-8")
        log_f = open(log_path, "a", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f, text=True)
        self.procs.append(p)
        return p

    def _wait_http(self, url: str, timeout_s: int, proc: subprocess.Popen | None, label: str, log_path: str) -> None:
        start = time.time()
        last_report = 0
        while True:
            try:
                with urllib.request.urlopen(url, timeout=2):
                    elapsed = int(time.time() - start)
                    if elapsed > 0:
                        print(f"Ready: {label} ({url}) after {elapsed}s")
                    return
            except urllib.error.HTTPError as e:
                # Treat non-5xx HTTP responses as service readiness. Example:
                # GET /v1/completions on proxy returns 405 when alive.
                if 400 <= e.code < 500:
                    elapsed = int(time.time() - start)
                    if elapsed > 0:
                        print(
                            f"Ready: {label} ({url}) after {elapsed}s "
                            f"(HTTP {e.code})"
                        )
                    return
            except Exception:
                pass

            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"{label} exited early with code {proc.returncode}")

            elapsed = int(time.time() - start)
            if elapsed - last_report >= 10:
                print(f"Waiting for {label} ({url})... {elapsed}s elapsed")
                try:
                    tail = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
                    for line in tail:
                        print(f"  {line}")
                except Exception:
                    pass
                last_report = elapsed
            if elapsed > timeout_s:
                raise TimeoutError(f"Timed out waiting for {label} ({url}) after {timeout_s}s")
            time.sleep(1)

    def _post_json(
        self,
        url: str,
        payload: dict,
        timeout: int,
        headers: dict[str, str] | None = None,
    ) -> dict:
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} at {url}: {body}") from e

    def _smoke(
        self,
        label: str,
        url: str,
        timeout_s: int,
        model: str,
        prompt: str,
        max_tokens: int = 1,
    ) -> None:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False,
        }
        out = self._post_json(url, payload, timeout_s)
        if "choices" not in out:
            raise RuntimeError(f"{label} smoke test failed, unexpected response: {out}")
        print(f"{label} smoke test passed.")

    def _load_prompts(self, dataset_path: Path, num_prompts: int) -> list[str]:
        prompts: list[str] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt")
                if isinstance(prompt, str):
                    prompts.append(prompt)
                if len(prompts) >= num_prompts:
                    break
        if len(prompts) < num_prompts:
            raise RuntimeError(
                f"Dataset has only {len(prompts)} prompts; requested {num_prompts}."
            )
        return prompts

    def _extract_text(self, out: dict[str, Any]) -> str:
        choices = out.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        txt = first.get("text")
        if isinstance(txt, str):
            return txt
        msg = first.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
        return ""

    def _save_responses_jsonl(
        self,
        proxy_url: str,
        dataset: Path,
        response_path: Path,
    ) -> None:
        args = self.args
        prompts = self._load_prompts(dataset, args.num_prompts)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        with response_path.open("w", encoding="utf-8") as fout:
            for idx, prompt in enumerate(prompts):
                payload = {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.output_len,
                    "stream": False,
                    "temperature": 0.0,
                }
                out = self._post_json(proxy_url, payload, args.smoke_timeout)
                rec = {
                    "index": idx,
                    "mode": args.kv_transfer_compression,
                    "model": args.model,
                    "prompt": prompt,
                    "output_text": self._extract_text(out),
                    "raw_response": out,
                }
                fout.write(json.dumps(rec, ensure_ascii=True) + "\n")
        print(f"Saved responses for {len(prompts)} prompts: {response_path}")

    def _disagg_handoff_smoke(self, host_ip: str) -> None:
        args = self.args
        request_id = (
            f"___prefill_addr_{host_ip}:{args.kv_port_prefill}"
            f"___decode_addr_{host_ip}:{args.kv_port_decode}_{uuid.uuid4().hex}"
        )
        headers = {
            "X-Request-Id": request_id,
            "X-KV-Target": f"localhost:{args.decode_port}",
        }

        # Stage 1: prefill with one token so KV is produced/saved.
        prefill_payload = {
            "model": args.model,
            "prompt": "Direct disagg handoff smoke test.",
            "max_tokens": 1,
            "stream": False,
        }
        prefill_out = self._post_json(
            f"http://localhost:{args.prefill_port}/v1/completions",
            prefill_payload,
            args.smoke_timeout,
            headers=headers,
        )
        if "choices" not in prefill_out:
            raise RuntimeError(
                f"Direct disagg handoff prefill failed, unexpected response: {prefill_out}"
            )

        # Stage 2: decode with same request_id so consumer loads transferred KV.
        decode_payload = {
            "model": args.model,
            "prompt": "Direct disagg handoff smoke test.",
            "max_tokens": 2,
            "stream": False,
        }
        decode_out = self._post_json(
            f"http://localhost:{args.decode_port}/v1/completions",
            decode_payload,
            args.smoke_timeout,
            headers=headers,
        )
        if "choices" not in decode_out:
            raise RuntimeError(
                f"Direct disagg handoff decode failed, unexpected response: {decode_out}"
            )
        print("Direct disagg handoff smoke test passed (prefill->decode with shared request_id).")

    def _show_logs(self) -> None:
        for lp in [self.args.prefill_log, self.args.decode_log, self.args.proxy_log]:
            print(f"\n===== {lp} (tail) =====")
            if not Path(lp).exists():
                print("(missing)")
                continue
            lines = Path(lp).read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            for line in lines:
                print(line)

    def _make_kv_cfg(self, role: str, rank: int, host_ip: str) -> dict:
        args = self.args
        if args.kv_transfer_compression == "int8":
            raise RuntimeError(
                "KV_TRANSFER_COMPRESSION=int8 is not supported by KVTransferConfig "
                "in this vLLM checkout. Use 'none' (P2pNcclConnector) or 'fp8' "
                "(FP8TransitConnector)."
            )

        connector_name = "P2pNcclConnector"
        cfg: dict = {
            "kv_connector": connector_name,
            "kv_role": role,
            "kv_rank": rank,
            "kv_parallel_size": 2,
            "kv_buffer_size": 5e9,
            "kv_ip": host_ip,
            "kv_port": args.kv_port_prefill if role == "kv_producer" else args.kv_port_decode,
        }

        if args.kv_transfer_compression == "fp8":
            connector_name = "FP8TransitConnector"
            cfg["kv_connector"] = connector_name
            cfg["kv_connector_extra_config"] = {
                "shared_storage_path": args.fp8_transit_storage_path,
                "compression": "fp8",
            }
        return cfg

    def run(self) -> int:
        args = self.args
        repo_root = str(Path(__file__).resolve().parents[2])
        dataset = Path(args.dataset_path)
        if not dataset.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        try:
            gpu_list = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to query GPUs via nvidia-smi:\n{e.output}") from e
        gpu_count = len([ln for ln in gpu_list.splitlines() if ln.strip()])
        if gpu_count < 2:
            raise RuntimeError(f"Need at least 2 visible GPUs. Found {gpu_count}.\n{gpu_list}")

        host_ip = subprocess.check_output(
            ["bash", "-lc", "hostname -I | awk '{print $1}'"], text=True
        ).strip()
        os.environ["VLLM_HOST_IP"] = host_ip

        if not args.result_filename:
            args.result_filename = f"disagg_custom_{args.kv_transfer_compression}.json"

        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / args.result_filename

        prefill_cfg = self._make_kv_cfg("kv_producer", 0, host_ip)
        decode_cfg = self._make_kv_cfg("kv_consumer", 1, host_ip)

        print(f"Starting disaggregated stack (compression={args.kv_transfer_compression})...")
        print(f"  MODEL={args.model}")
        print(f"  PREFILL_GPU={args.prefill_gpu}")
        print(f"  DECODE_GPU={args.decode_gpu}")
        print(f"  RESULT={result_path}")

        env_prefill = os.environ.copy()
        env_prefill["PYTHONUNBUFFERED"] = "1"
        env_prefill["CUDA_VISIBLE_DEVICES"] = str(args.prefill_gpu)
        # Disagg KV handoff requires stable request IDs across prefill/decode.
        env_prefill["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
        _prepend_pythonpath(env_prefill, repo_root)

        env_decode = os.environ.copy()
        env_decode["PYTHONUNBUFFERED"] = "1"
        env_decode["CUDA_VISIBLE_DEVICES"] = str(args.decode_gpu)
        # Disagg KV handoff requires stable request IDs across prefill/decode.
        env_decode["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
        _prepend_pythonpath(env_decode, repo_root)

        prefill_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--port",
            str(args.prefill_port),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--kv-transfer-config",
            json.dumps(prefill_cfg, separators=(",", ":")),
        ]
        decode_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--port",
            str(args.decode_port),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--kv-transfer-config",
            json.dumps(decode_cfg, separators=(",", ":")),
        ]

        prefill_p = self._start_proc(prefill_cmd, env_prefill, args.prefill_log)
        print(f"  prefill pid={prefill_p.pid} log={args.prefill_log}")
        decode_p = self._start_proc(decode_cmd, env_decode, args.decode_log)
        print(f"  decode pid={decode_p.pid} log={args.decode_log}")

        self._wait_http(
            f"http://localhost:{args.prefill_port}/v1/models",
            args.startup_timeout,
            prefill_p,
            "prefill server",
            args.prefill_log,
        )
        self._wait_http(
            f"http://localhost:{args.decode_port}/v1/models",
            args.startup_timeout,
            decode_p,
            "decode server",
            args.decode_log,
        )

        print("Running direct backend sanity checks...")
        if args.direct_handoff_smoke:
            self._disagg_handoff_smoke(host_ip)
        else:
            print("Skipping direct handoff smoke (set --direct-handoff-smoke to enable).")

        env_proxy = os.environ.copy()
        env_proxy["PYTHONUNBUFFERED"] = "1"
        _prepend_pythonpath(env_proxy, repo_root)
        proxy_cmd = [
            "python3",
            "benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py",
            "--port",
            str(args.proxy_port),
            "--prefill-url",
            f"http://localhost:{args.prefill_port}",
            "--decode-url",
            f"http://localhost:{args.decode_port}",
            "--kv-host",
            host_ip,
            "--prefill-kv-port",
            str(args.kv_port_prefill),
            "--decode-kv-port",
            str(args.kv_port_decode),
        ]
        proxy_p = self._start_proc(proxy_cmd, env_proxy, args.proxy_log)
        print(f"  proxy pid={proxy_p.pid} log={args.proxy_log}")

        self._wait_http(
            f"http://localhost:{args.proxy_port}/v1/completions",
            120,
            proxy_p,
            "disagg proxy",
            args.proxy_log,
        )

        print("Running decode-path smoke test via proxy...")
        self._smoke(
            "Proxy disagg decode-path",
            f"http://localhost:{args.proxy_port}/v1/completions",
            args.smoke_timeout,
            args.model,
            "Decode path smoke test.",
            max_tokens=max(4, args.output_len),
        )

        bench_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--model",
            args.model,
            "--endpoint",
            "/v1/completions",
            "--port",
            str(args.proxy_port),
            "--dataset-name",
            "custom",
            "--dataset-path",
            str(dataset),
            "--custom-output-len",
            str(args.output_len),
            "--num-prompts",
            str(args.num_prompts),
            "--max-concurrency",
            str(args.max_concurrency),
            "--request-rate",
            args.request_rate,
            "--save-result",
            "--result-dir",
            str(results_dir),
            "--result-filename",
            args.result_filename,
        ]
        if args.skip_chat_template:
            bench_cmd.append("--skip-chat-template")
        if args.bench_no_stream:
            bench_cmd.append("--no-stream")

        print("Running disaggregated benchmark...")
        bench_env = os.environ.copy()
        _prepend_pythonpath(bench_env, repo_root)
        subprocess.run(bench_cmd, check=True, env=bench_env)

        if not result_path.exists():
            raise FileNotFoundError(f"Missing benchmark result: {result_path}")
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if args.save_responses:
            if not args.responses_filename:
                args.responses_filename = f"disagg_custom_{args.kv_transfer_compression}_responses.jsonl"
            responses_path = results_dir / args.responses_filename
            print("Saving per-prompt responses for accuracy comparison...")
            self._save_responses_jsonl(
                f"http://localhost:{args.proxy_port}/v1/completions",
                dataset,
                responses_path,
            )

        print(
            f"\nDisaggregated benchmark summary "
            f"(compression={args.kv_transfer_compression})"
        )
        print(f"  request_throughput: {data.get('request_throughput', 0.0):.4f} req/s")
        print(f"  output_throughput : {data.get('output_throughput', 0.0):.4f} tok/s")
        print(f"  mean_ttft_ms      : {data.get('mean_ttft_ms', 0.0):.2f}")
        print(f"  mean_itl_ms       : {data.get('mean_itl_ms', 0.0):.2f}")
        print(f"  mean_e2el_ms      : {data.get('mean_e2el_ms', 0.0):.2f}")
        print(f"  result_json       : {result_path}")
        return 0


def main() -> int:
    args = parse_args()
    runner = Runner(args)

    def _signal_handler(_sig, _frame):
        runner.cleanup()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        return runner.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        runner._show_logs()
        return 1
    finally:
        runner.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())



# # no compression baseline
# VLLM_KV_DEBUG=1 DISAGG_PROXY_DEBUG=1 VLLM_LOGGING_LEVEL=DEBUG \
# KV_TRANSFER_COMPRESSION=none \
# python3 benchmarks/disagg_benchmarks/disagg_custom_prefill_decode_benchmark.py


# # int8
# VLLM_KV_DEBUG=1 DISAGG_PROXY_DEBUG=1 VLLM_LOGGING_LEVEL=DEBUG \
# KV_TRANSFER_COMPRESSION=int8 \
# python3 benchmarks/disagg_benchmarks/disagg_custom_prefill_decode_benchmark.py

# # fp8
# VLLM_KV_DEBUG=1 DISAGG_PROXY_DEBUG=1 VLLM_LOGGING_LEVEL=DEBUG \
# KV_TRANSFER_COMPRESSION=fp8 FP8_TRANSIT_STORAGE_PATH=/tmp/kv_transit \
# python3 benchmarks/disagg_benchmarks/disagg_custom_prefill_decode_benchmark.py

