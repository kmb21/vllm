# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FP8TransitConnector — Phase 1 of cross-region KV compression research.

Transit-only FP8 quantization for P/D disaggregated prefill:
  - Local KV cache stays in BF16/FP16 (no attention kernel changes)
  - Only the bytes written to the shared transport are FP8
  - ~2x bandwidth reduction: BF16 (2 bytes) → FP8 (1 byte) + tiny scale overhead
  - Supports "none" mode (raw copy) for baseline comparison

Usage:
  Prefill (producer):
    --kv-transfer-config '{
      "kv_connector": "FP8TransitConnector",
      "kv_role": "kv_producer",
      "kv_connector_extra_config": {
        "shared_storage_path": "/tmp/kv_transit",
        "compression": "fp8"
      }
    }'

  Decode (consumer):
    --kv-transfer-config '{
      "kv_connector": "FP8TransitConnector",
      "kv_role": "kv_consumer",
      "kv_connector_extra_config": {
        "shared_storage_path": "/tmp/kv_transit",
        "compression": "fp8"
      }
    }'

Metrics logged each interval:
  Compression ratio, Total MB uncompressed, Total MB compressed,
  Avg/P90 compress ms, Avg decompress ms
"""

import os
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch

from vllm._custom_ops import scaled_fp8_quant
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Optional backend-specific metadata types for KV extraction/injection.
try:
    from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
except ImportError:
    MLACommonMetadata = None  # type: ignore[misc,assignment]

try:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
except ImportError:
    TritonAttentionMetadata = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# FP8 dtype helper
# ---------------------------------------------------------------------------

def _fp8_dtype() -> torch.dtype:
    """Return the correct FP8 dtype for the current platform."""
    from vllm.platforms import current_platform
    if current_platform.is_rocm():
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class FP8TransitStats(KVConnectorStats):
    """
    Metrics for one logging interval. Tracks compression ratio,
    volume, and latency overhead of quantize/dequantize operations.
    """

    def __post_init__(self) -> None:
        if not self.data:
            self.reset()

    def reset(self) -> None:
        self.data: dict[str, list[float]] = {
            # Bytes before quantization (what would have gone on the wire uncompressed)
            "bytes_uncompressed": [],
            # Bytes after quantization (fp8 payload + scale tensor)
            "bytes_compressed": [],
            # Time to quantize one layer's KV for one request (producer side)
            "compression_latency_s": [],
            # Time to dequantize one layer's KV for one request (consumer side)
            "decompression_latency_s": [],
        }

    def record_compress(
        self, bytes_unc: int, bytes_cmp: int, latency_s: float
    ) -> None:
        self.data["bytes_uncompressed"].append(float(bytes_unc))
        self.data["bytes_compressed"].append(float(bytes_cmp))
        self.data["compression_latency_s"].append(latency_s)

    def record_decompress(self, latency_s: float) -> None:
        self.data["decompression_latency_s"].append(latency_s)

    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self.data.values())

    def aggregate(self, other: "KVConnectorStats") -> "FP8TransitStats":
        if not other.is_empty():
            for k in self.data:
                if k in other.data:
                    self.data[k].extend(other.data[k])
        return self

    def reduce(self) -> dict[str, int | float]:
        total_unc = sum(self.data["bytes_uncompressed"])
        total_cmp = sum(self.data["bytes_compressed"])
        ratio = round(total_unc / total_cmp, 3) if total_cmp > 0 else 0.0

        result: dict[str, int | float] = {
            "Compression ratio":       ratio,
            "Total MB uncompressed":   round(total_unc / 2**20, 3),
            "Total MB compressed":     round(total_cmp / 2**20, 3),
        }

        comp = self.data["compression_latency_s"]
        if comp:
            result["Avg compress ms"] = round(mean(comp) * 1e3, 3)
            result["P90 compress ms"] = round(
                sorted(comp)[max(0, int(len(comp) * 0.9) - 1)] * 1e3, 3
            )

        decomp = self.data["decompression_latency_s"]
        if decomp:
            result["Avg decompress ms"] = round(mean(decomp) * 1e3, 3)

        return result


# ---------------------------------------------------------------------------
# Per-request metadata passed from scheduler to worker
# ---------------------------------------------------------------------------

@dataclass
class ReqMeta:
    token_ids: torch.Tensor    # [num_tokens_aligned] — used as hash key for file path
    slot_mapping: torch.Tensor  # [num_tokens] — paged KV slot indices
    is_store: bool              # True on producer, False on consumer
    mm_hashes: list[str]        # multimodal feature hashes for cache key

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> "ReqMeta":
        num_tokens = _align_to_block(len(token_ids), block_size)
        token_ids_t = torch.tensor(token_ids)[:num_tokens]

        block_ids_t = torch.tensor(block_ids)
        n_blocks = block_ids_t.shape[0]
        offsets = torch.arange(block_size)
        # slot = block_id * block_size + offset
        slot_mapping = (
            offsets.reshape(1, block_size)
            + block_ids_t.reshape(n_blocks, 1) * block_size
        ).flatten()[:num_tokens]

        return ReqMeta(
            token_ids=token_ids_t,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )


@dataclass
class FP8TransitMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, mm_hashes)
        )


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------

class FP8TransitConnector(KVConnectorBase_V1):
    """
    Disaggregated P/D connector with FP8 transit compression.

    Transport: shared filesystem (disk/NFS/tmpfs). Each layer's KV for each
    request is written as a safetensors file by the producer and read by the
    consumer.

    Compression: fp8_e4m3fn dynamic per-tensor quantization via
    vllm._custom_ops.scaled_fp8_quant. The fp8 tensor is stored as int8
    (same width, any safetensors version) with a separate float32 scale.

    Set compression="none" in kv_connector_extra_config to disable compression
    and use raw BF16 — useful as a bandwidth baseline.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)

        self._block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config
        self._storage_path: str = extra.get("shared_storage_path", "/tmp/kv_transit")
        self._compression: str = extra.get("compression", "fp8")

        # Scheduler-side: tracks requests that still need their KV loaded.
        self._requests_need_load: dict[str, "Request"] = {}

        # Worker-side: accumulates metrics between logging intervals.
        self._stats = FP8TransitStats()

        logger.info(
            "FP8TransitConnector init: storage=%s, compression=%s, role=%s",
            self._storage_path,
            self._compression,
            role,
        )

    # -----------------------------------------------------------------------
    # Compression helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _quantize_fp8(
        kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize KV tensor to FP8 (dynamic per-tensor scale).

        Args:
            kv: any shape, any float dtype, on GPU.

        Returns:
            kv_int8: same shape as kv, dtype=int8 (bit-identical to fp8).
            scale: scalar float32 tensor.
        """
        orig_shape = kv.shape
        # scaled_fp8_quant requires 2D input: (M, N)
        kv_2d = kv.reshape(-1, kv.shape[-1]).to(torch.float32)
        kv_fp8, scale = scaled_fp8_quant(kv_2d)
        # Store fp8 as int8: same 1-byte width, compatible with all safetensors versions.
        kv_int8 = kv_fp8.view(torch.int8).reshape(orig_shape)
        return kv_int8, scale

    @staticmethod
    def _dequantize_fp8(
        kv_int8: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Dequantize int8-stored FP8 back to target_dtype.

        Args:
            kv_int8: fp8 data stored as int8, same shape as original kv.
            scale: scalar float32 scale from quantization.
            target_dtype: dtype of the local KV cache (e.g. bfloat16).

        Returns:
            Reconstructed KV in target_dtype, same shape as kv_int8.
        """
        fp8_dtype = _fp8_dtype()
        # Reinterpret int8 bits as fp8 — zero-copy, same memory.
        kv_fp8 = kv_int8.view(fp8_dtype)
        # Flatten to 2D, dequant, reshape back.
        kv_f32 = kv_fp8.reshape(-1, kv_fp8.shape[-1]).to(torch.float32) * scale
        return kv_f32.to(target_dtype).reshape(kv_int8.shape)

    # -----------------------------------------------------------------------
    # KV extraction / injection (copied from ExampleConnector, handles MLA/Triton/default)
    # -----------------------------------------------------------------------

    def _extract_kv(
        self,
        layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        """
        Extract token KVs from a paged KV buffer using slot_mapping.

        Returns tensor shaped [2, T, H*D] for standard attention,
        [T, xxx] for MLA.
        """
        if MLACommonMetadata is not None and isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = layer.shape[0], layer.shape[1]
            return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
        elif TritonAttentionMetadata is not None and isinstance(
            attn_metadata, TritonAttentionMetadata
        ):
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            return layer[block_idxs, :, offsets]
        else:
            # Standard paged attention: [2, num_pages, page_size, head_dim]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

    def _inject_kv(
        self,
        kv_cache_layer: torch.Tensor,
        kv: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: Any,
    ) -> None:
        """
        Inject reconstructed KVs into the paged KV cache buffer.
        """
        if MLACommonMetadata is not None and isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = kv_cache_layer.shape[0], kv_cache_layer.shape[1]
            kv_cache_layer.reshape(num_pages * page_size, -1)[slot_mapping, ...] = kv
        elif TritonAttentionMetadata is not None and isinstance(
            attn_metadata, TritonAttentionMetadata
        ):
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            kv_cache_layer[block_idxs, :, offsets] = kv
        else:
            num_pages, page_size = kv_cache_layer.shape[1], kv_cache_layer.shape[2]
            kv_cache_layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...] = kv

    # -----------------------------------------------------------------------
    # File path helpers
    # -----------------------------------------------------------------------

    def _req_folder(
        self,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        create: bool = False,
    ) -> str:
        data = token_ids.numpy().tobytes()
        if mm_hashes:
            data += "-".join(mm_hashes).encode()
        h = safe_hash(data, usedforsecurity=False).hexdigest()
        folder = os.path.join(self._storage_path, h)
        if create:
            os.makedirs(folder, exist_ok=True)
        return folder

    def _kv_path(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        create_folder: bool = False,
    ) -> str:
        folder = self._req_folder(token_ids, mm_hashes, create=create_folder)
        return os.path.join(folder, f"{layer_name}.safetensors")

    # -----------------------------------------------------------------------
    # Worker-side: save
    # -----------------------------------------------------------------------

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Load all KVs for all consumer requests before the forward pass.
        Called once per step on the decode (consumer) instance.
        """
        meta = self._get_connector_metadata()
        assert isinstance(meta, FP8TransitMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        for req in meta.requests:
            if req.is_store:
                continue

            slot_mapping_gpu = req.slot_mapping.to("cuda")

            for layer_name, layer in forward_context.no_compile_layers.items():
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue

                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                target_dtype = kv_cache_layer.dtype

                path = self._kv_path(layer_name, req.token_ids, req.mm_hashes)

                t0 = time.perf_counter()
                saved = safetensors.torch.load_file(path)
                kv_stored = saved["kv"].to(kv_cache_layer.device)

                if self._compression == "fp8":
                    scale = saved["scale"].to(kv_cache_layer.device)
                    kv = self._dequantize_fp8(kv_stored, scale, target_dtype)
                else:
                    kv = kv_stored.to(target_dtype)

                self._stats.record_decompress(time.perf_counter() - t0)

                layer_attn_meta = (
                    attn_metadata[layer_name]
                    if isinstance(attn_metadata, dict)
                    else attn_metadata
                )
                self._inject_kv(
                    kv_cache_layer, kv, slot_mapping_gpu, layer_attn_meta
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Load is synchronous in start_load_kv — nothing to wait for."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """
        Extract, compress, and save one layer's KV for all producer requests.
        Called per attention layer on the prefill (producer) instance.
        """
        meta = self._get_connector_metadata()
        assert isinstance(meta, FP8TransitMetadata)

        for req in meta.requests:
            if not req.is_store:
                continue

            slot_mapping_gpu = req.slot_mapping.to(kv_layer.device)
            layer_attn_meta = (
                attn_metadata[layer_name]
                if isinstance(attn_metadata, dict)
                else attn_metadata
            )

            t0 = time.perf_counter()
            kv_raw = self._extract_kv(kv_layer, slot_mapping_gpu, layer_attn_meta)

            if self._compression == "fp8":
                kv_stored, scale = self._quantize_fp8(kv_raw)
                compress_time = time.perf_counter() - t0
                bytes_unc = kv_raw.nbytes
                bytes_cmp = kv_stored.nbytes + scale.nbytes
                self._stats.record_compress(bytes_unc, bytes_cmp, compress_time)
                tensors = {
                    "kv": kv_stored.cpu(),
                    "scale": scale.cpu(),
                }
            else:
                compress_time = time.perf_counter() - t0
                bytes_raw = kv_raw.nbytes
                self._stats.record_compress(bytes_raw, bytes_raw, compress_time)
                tensors = {"kv": kv_raw.cpu()}

            path = self._kv_path(
                layer_name, req.token_ids, req.mm_hashes, create_folder=True
            )
            safetensors.torch.save_file(tensors, path)

    def wait_for_save(self) -> None:
        """Save is synchronous in save_kv_layer — nothing to wait for."""
        return

    # -----------------------------------------------------------------------
    # Worker-side: stats
    # -----------------------------------------------------------------------

    def get_kv_connector_stats(self) -> FP8TransitStats | None:
        if self._stats.is_empty():
            return None
        stats = self._stats
        self._stats = FP8TransitStats()
        return stats

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> FP8TransitStats | None:
        return FP8TransitStats(data=data) if data is not None else FP8TransitStats()

    # -----------------------------------------------------------------------
    # Scheduler-side
    # -----------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Return how many tokens can be loaded from the external KV store.
        On the consumer side, checks whether the producer has already written
        a folder for this request's prefix. On the producer side, returns 0.
        """
        if not self._kv_transfer_config.is_kv_consumer:
            return 0, False

        if not self._found_match(request):
            return 0, False

        token_ids = list(request.prompt_token_ids or [])
        num_tokens = _align_to_block(len(token_ids) - 1, self._block_size)
        return max(0, num_tokens - num_computed_tokens), False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> FP8TransitMetadata:
        meta = FP8TransitMetadata()
        total_loads = 0

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            block_ids = new_req.block_ids[0]

            if new_req.req_id in self._requests_need_load:
                # Consumer: load KV that producer has written
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
                total_loads += 1
            elif self._kv_transfer_config.is_kv_producer:
                # Producer: save KV for this new request's full prompt prefix
                if not self._found_match_for_prompt(token_ids, mm_hashes):
                    meta.add_request(
                        token_ids=token_ids,
                        block_ids=block_ids,
                        block_size=self._block_size,
                        is_store=True,
                        mm_hashes=mm_hashes,
                    )

        # Handle resumed (preempted) cached requests that need loading
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id not in cached.resumed_req_ids:
                continue
            if req_id not in self._requests_need_load:
                continue

            request = self._requests_need_load[req_id]
            num_computed = cached.num_computed_tokens[i]
            num_new = scheduler_output.num_scheduled_tokens[req_id]
            total_tokens = num_computed + num_new
            token_ids = request.all_token_ids[:total_tokens]
            block_ids = cached.new_block_ids[i][0]

            meta.add_request(
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=self._block_size,
                is_store=False,
                mm_hashes=[f.identifier for f in request.mm_features],
            )
            total_loads += 1

        assert total_loads == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    # -----------------------------------------------------------------------
    # Match helpers
    # -----------------------------------------------------------------------

    def _found_match(self, request: "Request") -> bool:
        return self._found_match_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
        )

    def _found_match_for_prompt(
        self, token_ids: list[int], mm_hashes: list[str]
    ) -> bool:
        num_tokens = _align_to_block(len(token_ids) - 1, self._block_size)
        t = torch.tensor(token_ids)[:num_tokens]
        folder = self._req_folder(t, mm_hashes, create=False)
        return os.path.exists(folder)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _align_to_block(num_tokens: int, block_size: int) -> int:
    """Round num_tokens DOWN to the nearest block_size boundary."""
    return (num_tokens - 1) // block_size * block_size