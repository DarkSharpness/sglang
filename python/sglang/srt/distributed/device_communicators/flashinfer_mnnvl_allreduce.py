"""
Wrap flashinfer's MNNVL AllReduce as a custom-all-reduce backend.

This is a thin adapter around ``flashinfer.comm.create_allreduce_fusion_workspace``
+ ``flashinfer.comm.allreduce_fusion`` (pattern=kAllReduce) so that the same
call sites that drive ``CustomAllReduceV2`` can transparently use flashinfer's
multicast-based MNNVL kernel instead of our pull/push v2 implementation. The
interface (``disabled`` / ``capture`` / ``should_custom_ar`` / ``custom_all_reduce``
/ ``close``) matches what ``parallel_state.GroupCoordinator`` expects from a
custom-AR object.

Choose this backend when:
- the cluster has an MNNVL/IMEX fabric (GB200/GB300 NVL72), AND
- you want flashinfer's NVSwitch-multicast publish path (it scales beyond our
  ``kMaxNumGPU = 8`` pull/push kernels because it doesn't need to write to
  every peer individually).

Enable via the env var ``SGLANG_OPT_USE_FLASHINFER_MNNVL_AR=1``; the dispatcher
in ``custom_all_reduce.py`` then picks this class instead of ``CustomAllReduceV2``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_weak_contiguous,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int env %s=%r, using default %d", name, raw, default)
        return default


class FlashInferMnnvlAllReduce:
    """Custom-AR backend backed by flashinfer's MNNVL AllReduce kernel.

    The flashinfer kernel is shaped for ``[token_num, hidden_dim]`` inputs.
    A workspace is sized once (in bytes) at construction; at call time we
    factor ``input.numel()`` as ``(numel/hidden_dim) x hidden_dim`` and reshape
    the input view accordingly. ``should_custom_ar`` rejects shapes that don't
    factor cleanly so the caller falls back to NCCL.

    Tunables (env / ctor args):
        * ``SGLANG_FLASHINFER_MNNVL_AR_HIDDEN_DIM`` -- hidden dim used for the
          ``view(-1, H)`` reshape. Default 8192. Set this to your model's
          actual hidden dim for best alignment / least fallback.
        * ``SGLANG_FLASHINFER_MNNVL_AR_MAX_TOKENS`` -- max tokens the workspace
          must accommodate. Default 8192.
        * ``SGLANG_FLASHINFER_MNNVL_AR_DTYPE`` -- workspace dtype hint
          (``bfloat16`` / ``float16`` / ``float32``). Workspace is sized in
          bytes; runtime dtype just needs to fit.
    """

    # Default workspace shape -- chosen so the buffer is ~256 MiB, large enough
    # to cover any single TP-AR tensor we'd realistically all-reduce in
    # decode/prefill (max hidden=8192 x 8192 tokens x bf16 = 128 MiB; x3 lamport
    # buffers gives ~384 MiB which the workspace allocator pads up to fabric
    # granularity).
    DEFAULT_HIDDEN_DIM = 8192
    DEFAULT_MAX_TOKENS = 8192

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        hidden_dim: Optional[int] = None,
        max_token_num: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.workspace = None
        self._allreduce_fusion = None
        self._pattern = None

        # Resolve config (ctor arg -> env var -> default).
        hidden_dim = hidden_dim or _env_int(
            "SGLANG_FLASHINFER_MNNVL_AR_HIDDEN_DIM", self.DEFAULT_HIDDEN_DIM
        )
        max_token_num = max_token_num or _env_int(
            "SGLANG_FLASHINFER_MNNVL_AR_MAX_TOKENS", self.DEFAULT_MAX_TOKENS
        )
        dtype = dtype or _resolve_dtype_from_env(
            os.environ.get("SGLANG_FLASHINFER_MNNVL_AR_DTYPE", "bfloat16")
        )

        self.hidden_dim = hidden_dim
        self.max_token_num = max_token_num
        self.dtype = dtype
        self.elem_size = torch.tensor([], dtype=dtype).element_size()
        # Total per-call byte budget. flashinfer's workspace can serve any
        # (tokens, hidden) decomposition whose tokens*hidden*elem_size fits
        # this; see is_buffer_size_sufficient in flashinfer/comm/trtllm_mnnvl_ar.
        self.max_bytes = max_token_num * hidden_dim * self.elem_size

        try:
            from flashinfer.comm import (
                AllReduceFusionPattern,
                allreduce_fusion,
                create_allreduce_fusion_workspace,
            )
            from flashinfer.comm.mnnvl import TorchDistBackend
        except ImportError as e:
            logger.warning("flashinfer MNNVL AR unavailable, disabling: %s", e)
            return

        try:
            self.workspace = create_allreduce_fusion_workspace(
                backend="mnnvl",
                world_size=self.world_size,
                rank=self.rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                comm_backend=TorchDistBackend(group=group),
            )
        except Exception as e:
            logger.warning("Failed to create flashinfer MNNVL workspace: %s", e)
            return

        self._allreduce_fusion = allreduce_fusion
        self._pattern = AllReduceFusionPattern.kAllReduce
        self.disabled = False
        log_info_on_rank0(
            logger,
            f"FlashInfer MNNVL AR initialized "
            f"(world={self.world_size}, hidden={hidden_dim}, "
            f"max_tokens={max_token_num}, dtype={dtype})",
        )

    @contextmanager
    def capture(self):
        # flashinfer's allreduce_fusion is graph-captureable as-is -- the
        # Lamport-buffer index advances per-call, kernels read peer pointers
        # that are stable for the workspace lifetime, and there is no
        # IPC-handle registration to do up-front.
        yield

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not is_weak_contiguous(inp):
            return False
        numel = inp.numel()
        elem_size = inp.element_size()
        if numel * elem_size % 16 != 0:
            return False  # 16B alignment required by the packed-vec loads
        if numel * elem_size > self.max_bytes:
            return False  # workspace too small for this input
        # Need (numel = tokens * hidden_dim) with tokens <= max_token_num.
        if numel % self.hidden_dim != 0:
            return False
        if numel // self.hidden_dim > self.max_token_num:
            return False
        return True

    def custom_all_reduce(self, input_: torch.Tensor) -> Optional[torch.Tensor]:
        # Defensive: caller normally checks should_custom_ar first.
        original_shape = input_.shape
        view = input_.contiguous().view(-1, self.hidden_dim)
        out = self._allreduce_fusion(
            input=view,
            workspace=self.workspace,
            pattern=self._pattern,
            launch_with_pdl=True,
        )
        return out.view(original_shape)

    def close(self):
        # Idempotent -- `__del__` may also call this after the caller has torn
        # down the process group. The flashinfer workspace destructor handles
        # VMM/cuMemMap cleanup once we drop our last reference.
        if self.disabled:
            return
        self.disabled = True
        self.workspace = None
        self._allreduce_fusion = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _resolve_dtype_from_env(name: str) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dt = table.get(name.strip().lower())
    if dt is None:
        logger.warning("Unknown dtype %r; defaulting to bfloat16", name)
        return torch.bfloat16
    return dt
