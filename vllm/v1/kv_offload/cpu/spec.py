# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.evoke_rope_delta import EvokeRopeDeltaRotator
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class CPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # calculate kv_bytes_per_offloaded_block
        assert kv_cache_config is not None
        if kv_cache_config.num_blocks > 0:
            total_gpu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * vllm_config.parallel_config.world_size
        else:
            kv_bytes_per_block = 0

        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        world_size = vllm_config.parallel_config.world_size
        self.cpu_page_size_per_worker: int = (
            kv_bytes_per_offloaded_block // world_size if world_size > 0 else 0
        )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

        # EVOKE smart-recovery RoPE rotator config. Opt-in: requires the
        # user to set `evoke_recovery_enabled=true` plus the model's RoPE
        # parameters in kv_connector_extra_config. When enabled,
        # create_handlers constructs an EvokeRopeDeltaRotator from the
        # canonical KV cache tensors so recovered blocks get re-anchored
        # via inverse+forward RoPE on the transfer stream after
        # swap_blocks_batch and before the completion event.
        self.evoke_recovery_enabled: bool = bool(
            self.extra_config.get("evoke_recovery_enabled", False)
        )
        self.evoke_rope_head_dim: int = int(
            self.extra_config.get("evoke_rope_head_dim", 0)
        )
        self.evoke_rope_base: float = float(
            self.extra_config.get("evoke_rope_base", 1000000.0)
        )
        self.evoke_rope_num_layers: int = int(
            self.extra_config.get("evoke_rope_num_layers", 0)
        )
        self.evoke_rope_num_kv_heads: int = int(
            self.extra_config.get("evoke_rope_num_kv_heads", 0)
        )
        self.evoke_rope_max_position: int = int(
            self.extra_config.get("evoke_rope_max_position", 131072)
        )
        self.evoke_rope_is_neox: bool = bool(
            self.extra_config.get("evoke_rope_is_neox", True)
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuGpuOffloadingHandlers:
        rotator = self._maybe_build_rope_rotator(kv_caches)
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            evoke_rope_rotator=rotator,
        )

    def _maybe_build_rope_rotator(
        self, kv_caches: CanonicalKVCaches
    ) -> EvokeRopeDeltaRotator | None:
        """Construct an EvokeRopeDeltaRotator from the canonical KV cache
        tensors and the user-supplied RoPE parameters, when EVOKE smart-
        recovery is enabled in the extra config. Returns None otherwise.

        Per-layer K view extraction assumes the canonical layout
        `(num_blocks, 2, block_size, num_kv_heads, head_size)` per layer
        (FlashAttention's default after canonicalization). If the layout is
        different (e.g. cross-layer HND), this returns None with a logged
        warning so the user can adjust the connector layout config.
        """
        if not self.evoke_recovery_enabled:
            return None
        if (
            self.evoke_rope_head_dim <= 0
            or self.evoke_rope_num_layers <= 0
            or self.evoke_rope_num_kv_heads <= 0
        ):
            logger.warning(
                "EVOKE recovery enabled but RoPE config incomplete "
                "(head_dim=%d, num_layers=%d, num_kv_heads=%d); rotator "
                "will not be constructed and recovered blocks will not be "
                "re-anchored. Set evoke_rope_head_dim, evoke_rope_num_layers, "
                "evoke_rope_num_kv_heads in kv_connector_extra_config.",
                self.evoke_rope_head_dim,
                self.evoke_rope_num_layers,
                self.evoke_rope_num_kv_heads,
            )
            return None
        try:
            import torch

            k_views_per_layer = self._extract_per_layer_k_views(kv_caches)
            if k_views_per_layer is None:
                return None

            inv_freq = 1.0 / (
                self.evoke_rope_base
                ** (
                    torch.arange(
                        0,
                        self.evoke_rope_head_dim,
                        2,
                        dtype=torch.float32,
                    )
                    / self.evoke_rope_head_dim
                )
            )
            t = torch.arange(self.evoke_rope_max_position, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(
                device=k_views_per_layer[0].device,
                dtype=k_views_per_layer[0].dtype,
            )
            rotator = EvokeRopeDeltaRotator(
                k_views_per_layer=k_views_per_layer,
                cos_sin_cache=cos_sin_cache,
                head_size=self.evoke_rope_head_dim,
                is_neox=self.evoke_rope_is_neox,
            )
            logger.info(
                "EVOKE smart-recovery rotator constructed for %d layers, "
                "head_dim=%d, base=%g, max_position=%d, is_neox=%s",
                self.evoke_rope_num_layers,
                self.evoke_rope_head_dim,
                self.evoke_rope_base,
                self.evoke_rope_max_position,
                self.evoke_rope_is_neox,
            )
            return rotator
        except Exception as e:
            logger.warning(
                "EVOKE recovery enabled but rotator construction failed: "
                "%s. Recovered blocks will not be re-anchored.",
                e,
            )
            return None

    def _extract_per_layer_k_views(self, kv_caches: CanonicalKVCaches) -> list | None:
        """Return per-layer K tensor views of shape
        (num_blocks, block_size, num_kv_heads, head_size).

        Handles two known canonical layouts:

        1. Per-layer: len(kv_caches.tensors) == num_layers; each tensor has
           shape (num_blocks, 2, block_size, num_kv_heads, head_size). K is
           tensor[:, 0]. This is the FlashAttention default without
           prefer_cross_layer_blocks.

        2. Cross-layer HND: len(kv_caches.tensors) == 1; the tensor has
           shape (num_blocks, num_kv_heads, num_layers, 2, block_size,
           head_size). K per layer is tensor[:, :, layer, 0, :, :], which
           needs permute(0, 2, 1, 3) to land on (num_blocks, block_size,
           num_kv_heads, head_size). This is what OffloadingConnector
           forces via get_required_kvcache_layout="HND".

        Returns None when neither layout matches (with a logged warning so
        the user can adjust the kv_connector layout or RoPE config).
        """
        n_tensors = len(kv_caches.tensors)
        if n_tensors == self.evoke_rope_num_layers:
            views = []
            for kv_cache_tensor in kv_caches.tensors:
                tensor = kv_cache_tensor.tensor
                if tensor.dim() != 5 or tensor.shape[1] != 2:
                    logger.warning(
                        "EVOKE recovery: per-layer canonical KV tensor has "
                        "unexpected shape %s; expected "
                        "(num_blocks, 2, block_size, num_kv_heads, head_size). "
                        "Skipping rotator construction.",
                        tuple(tensor.shape),
                    )
                    return None
                views.append(tensor[:, 0])
            return views
        if n_tensors == 1:
            tensor = kv_caches.tensors[0].tensor
            # vLLM hands us the flat int8 byte view (num_blocks, page_size_bytes).
            # Reshape into the logical HND layout: (num_blocks, num_kv_heads,
            # num_layers, 2, block_size, head_size) viewed as the model's K dtype
            # (bfloat16 for Qwen2.5).
            import torch as _torch

            num_kv_heads = self.evoke_rope_num_kv_heads
            num_layers = self.evoke_rope_num_layers
            head_size = self.evoke_rope_head_dim
            kv_dtype = _torch.bfloat16  # Qwen2.5 + FLASH_ATTN default
            element_bytes = _torch.empty(0, dtype=kv_dtype).element_size()
            elements_per_block = num_kv_heads * num_layers * 2 * head_size
            if tensor.dim() == 6:
                hnd = tensor
            elif tensor.dim() == 2:
                bytes_per_block = int(tensor.shape[1])
                if bytes_per_block % (elements_per_block * element_bytes) != 0:
                    logger.warning(
                        "EVOKE recovery: cannot derive block_size from flat "
                        "page_size_bytes=%d (does not divide evenly by "
                        "num_kv_heads*num_layers*2*head_size*dtype_bytes=%d). "
                        "Skipping rotator construction.",
                        bytes_per_block,
                        elements_per_block * element_bytes,
                    )
                    return None
                block_size = bytes_per_block // (elements_per_block * element_bytes)
                hnd = tensor.view(kv_dtype).view(
                    tensor.shape[0],
                    num_kv_heads,
                    num_layers,
                    2,
                    block_size,
                    head_size,
                )
            else:
                logger.warning(
                    "EVOKE recovery: cross-layer KV tensor has unexpected "
                    "shape %s; expected (num_blocks, num_kv_heads, "
                    "num_layers=%d, 2, block_size, head_size) or a "
                    "(num_blocks, page_size_bytes) int8 view. Skipping "
                    "rotator construction.",
                    tuple(tensor.shape),
                    num_layers,
                )
                return None
            if hnd.shape[2] != num_layers:
                logger.warning(
                    "EVOKE recovery: HND tensor num_layers dim is %d, "
                    "config evoke_rope_num_layers=%d. Skipping rotator "
                    "construction.",
                    hnd.shape[2],
                    num_layers,
                )
                return None
            views = []
            for layer_idx in range(num_layers):
                k_layer = hnd[:, :, layer_idx, 0, :, :]
                k_layer = k_layer.permute(0, 2, 1, 3)
                views.append(k_layer)
            return views
        logger.warning(
            "EVOKE recovery: kv_caches has %d tensors, expected either "
            "num_layers=%d (per-layer) or 1 (cross-layer HND). Skipping "
            "rotator construction.",
            n_tensors,
            self.evoke_rope_num_layers,
        )
        return None

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike GPUs"
                )
            self._handlers = self.create_handlers(kv_caches)

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
