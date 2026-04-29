"""
Symmetric peer memory backed by CUDA Multi-Node NVLink (MNNVL) fabric handles.

The custom-all-reduce v2 control plane normally exchanges peer storage via
``cudaIpc*`` handles, which only work for processes on the same node. To run
the same algorithm across an MNNVL fabric (e.g. GB200 NVL72), we replace that
with the CUDA Virtual Memory Management (VMM) API: each rank allocates a chunk
of fabric memory, exports a portable shareable handle, all-gathers handles
across the TP group, and maps every rank's chunk into its own virtual address
space at a deterministic offset.

The result is a symmetric ``[world_size]`` array of device pointers. Reads and
writes against those pointers are routed over NVLink by the GPU, so the
existing pull/push kernels work unchanged --- only the peer-memory acquisition
path differs from the cudaIpc version.

The implementation is intentionally minimal and modeled on flashinfer's
``SymmDeviceMemory`` (multicast disabled) but uses ``torch.distributed`` for
collective handle exchange so it integrates with sglang's existing TP group.
"""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

try:  # cuda-python >= 12.9
    from cuda.bindings import driver as _cuda
except ImportError:  # pragma: no cover - older cuda-python fallback
    try:
        from cuda import cuda as _cuda  # type: ignore
    except ImportError as e:  # pragma: no cover
        _cuda = None  # type: ignore
        _IMPORT_ERR = e
    else:
        _IMPORT_ERR = None
else:
    _IMPORT_ERR = None


def _check(err, *result):
    """Translate ``CUresult`` into a Python exception."""
    if err != _cuda.CUresult.CUDA_SUCCESS:
        try:
            _, name = _cuda.cuGetErrorName(err)
            _, msg = _cuda.cuGetErrorString(err)
            name = name.decode() if isinstance(name, bytes) else str(name)
            msg = msg.decode() if isinstance(msg, bytes) else str(msg)
        except Exception:
            name, msg = str(err), ""
        raise RuntimeError(f"CUDA driver error: {name} ({msg})")
    return result[0] if len(result) == 1 else result


def _round_up(value: int, gran: int) -> int:
    return (value + gran - 1) // gran * gran


def is_mnnvl_fabric_supported(device_idx: Optional[int] = None) -> bool:
    """Return True if the device exposes a usable fabric (MNNVL) handle type.

    This only checks the driver capability; it doesn't probe whether the
    cluster is healthy. We let the actual ``cuMemCreate`` call surface fabric
    misconfiguration at construction time.
    """
    if _cuda is None:
        return False
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    try:
        supported = _check(
            *_cuda.cuDeviceGetAttribute(
                _cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                device_idx,
            )
        )
        return bool(supported)
    except Exception as e:
        logger.debug("MNNVL fabric support probe failed: %s", e)
        return False


class MnnvlSymmMemory:
    """Symmetric fabric-backed device memory, exchanged across a TP group.

    On construction, every rank in ``group`` allocates ``size_bytes`` of
    pinned device memory via the VMM API, exports it as a fabric handle,
    all-gathers handles, and maps every rank's allocation at a deterministic
    virtual offset. After construction, ``peer_ptrs[r]`` is a device pointer
    that reads/writes rank ``r``'s allocation, including ``peer_ptrs[my_rank]``
    which aliases ``local_ptr``.

    Resources are released by ``close()`` (also called from ``__del__``).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        size_bytes: int,
    ) -> None:
        if _cuda is None:
            raise ImportError(
                "cuda-python is required for MNNVL symmetric memory; "
                f"original import error: {_IMPORT_ERR!r}"
            )
        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be positive, got {size_bytes}")
        if device.type != "cuda":
            raise ValueError(f"Expected CUDA device, got {device}")
        if not is_mnnvl_fabric_supported(device.index):
            raise RuntimeError(
                "MNNVL fabric handles are not supported on this device; "
                "an MNNVL-capable platform (e.g. GB200) is required."
            )

        self._group = group
        self._device_idx = device.index
        self._world_size = dist.get_world_size(group=group)
        self._rank = dist.get_rank(group=group)
        # Track allocation/mapping state so we can clean up on errors.
        self._mem_handles: List[int] = []
        self._base_ptr: int = 0
        self._aligned_size: int = 0
        self._stride: int = 0
        self.local_ptr: int = 0
        self.peer_ptrs: List[int] = []

        try:
            self._setup(size_bytes)
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------ #
    # Allocation
    # ------------------------------------------------------------------ #
    def _alloc_prop(self) -> "_cuda.CUmemAllocationProp":
        prop = _cuda.CUmemAllocationProp()
        prop.type = _cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = (
            _cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
        prop.location = _cuda.CUmemLocation()
        prop.location.type = _cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._device_idx
        return prop

    def _setup(self, size_bytes: int) -> None:
        prop = self._alloc_prop()

        granularity = _check(
            *_cuda.cuMemGetAllocationGranularity(
                prop,
                _cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        aligned_size = _round_up(size_bytes, granularity)
        self._aligned_size = aligned_size
        self._stride = aligned_size

        # 1. allocate local fabric memory
        local_handle = _check(*_cuda.cuMemCreate(aligned_size, prop, 0))

        # 2. export local handle to a portable fabric handle (64 bytes).
        local_fabric_handle = _check(
            *_cuda.cuMemExportToShareableHandle(
                local_handle,
                _cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                0,
            )
        )

        # 3. all-gather fabric handles. The handle is a struct whose .data
        # field is a 64-byte string; pickling it as bytes is sufficient.
        local_bytes = bytes(local_fabric_handle.data)
        gathered: List[Optional[bytes]] = [None] * self._world_size
        dist.all_gather_object(gathered, local_bytes, group=self._group)
        for i, b in enumerate(gathered):
            if not isinstance(b, (bytes, bytearray)) or len(b) != len(local_bytes):
                raise RuntimeError(
                    f"Bad fabric handle from rank {i}: type={type(b)} "
                    f"len={len(b) if hasattr(b, '__len__') else '?'}"
                )

        # 4. reserve a single contiguous VA range for all peer slots.
        total_size = aligned_size * self._world_size
        base_ptr = _check(*_cuda.cuMemAddressReserve(total_size, granularity, 0, 0))
        self._base_ptr = int(base_ptr)

        # 5. map local memory and import + map peer memories.
        # cuda-python's binding for fabric handles accepts the raw 64-byte
        # ``data`` blob directly as the osHandle argument (same pattern used
        # by flashinfer's MnnvlMemory).
        mem_handles: List[int] = [0] * self._world_size
        for i in range(self._world_size):
            slot_ptr = self._base_ptr + i * aligned_size
            if i == self._rank:
                handle = local_handle
            else:
                handle = _check(
                    *_cuda.cuMemImportFromShareableHandle(
                        gathered[i],
                        _cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                    )
                )
            _check(*_cuda.cuMemMap(slot_ptr, aligned_size, 0, handle, 0))
            mem_handles[i] = int(handle)

        self._mem_handles = mem_handles

        # 6. enable RW access from this device for the entire VA range.
        access = _cuda.CUmemAccessDesc()
        access.location = _cuda.CUmemLocation()
        access.location.type = _cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self._device_idx
        access.flags = _cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(*_cuda.cuMemSetAccess(self._base_ptr, total_size, [access], 1))

        # 7. expose peer pointers.
        self.peer_ptrs = [
            self._base_ptr + i * aligned_size for i in range(self._world_size)
        ]
        self.local_ptr = self.peer_ptrs[self._rank]

        # 8. zero-init local slot so signal counters start clean even if the
        # caller forgets. (The C++ side also memsets the signal regions.)
        _check(*_cuda.cuMemsetD8(self.local_ptr, 0, aligned_size))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def aligned_size(self) -> int:
        return self._aligned_size

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if sys.is_finalizing() or _cuda is None:
            return
        # Best-effort teardown; swallow exceptions individually so that one
        # failure (e.g. context already destroyed) doesn't leak the rest.
        for i, handle in enumerate(self._mem_handles):
            if not handle:
                continue
            slot_ptr = self._base_ptr + i * self._aligned_size
            try:
                _check(*_cuda.cuMemUnmap(slot_ptr, self._aligned_size))
            except Exception as e:
                logger.debug("cuMemUnmap rank=%d failed: %s", i, e)
            try:
                _check(*_cuda.cuMemRelease(handle))
            except Exception as e:
                logger.debug("cuMemRelease rank=%d failed: %s", i, e)
        self._mem_handles = []

        if self._base_ptr:
            total = self._aligned_size * self._world_size
            try:
                _check(*_cuda.cuMemAddressFree(self._base_ptr, total))
            except Exception as e:
                logger.debug("cuMemAddressFree failed: %s", e)
            self._base_ptr = 0

        self.local_ptr = 0
        self.peer_ptrs = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
