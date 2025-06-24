import math
import threading
import time
from typing import List, Literal, overload
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.kvcacheio import transfer_kv_all_layer
from sgl_kernel import cuda_utils # type: ignore

def init_context(a: int):
    cnt = 132 # H100
    print(f"Available SM count: {cnt}")
    b = cnt - a
    device_id = torch.cuda.current_device()
    stream_a, stream_b = cuda_utils.create_green_context(device_id, a, 1, 1)
    device = torch.device(f"cuda:{device_id}")
    stream_a = torch.cuda.ExternalStream(stream_ptr=stream_a, device=device)
    stream_b = torch.cuda.ExternalStream(stream_ptr=stream_b, device=device)
    print(f"Created Greenctx stream with SM_a={a} and SM_b={b}")
    return stream_a, stream_b, a, b

DEVICE = "cuda:0"

def _make_paged_cache(num_pages: int, page_size: int, nhead_k: int, head_dim: int, layers: int = 32):
    return torch.empty(
        (layers, num_pages, page_size, nhead_k, head_dim),
        dtype=torch.float16,
        device=DEVICE,
        requires_grad=False,
    )

def _make_cu_seqlens(seq_lens: list[int]):
    return torch.nn.functional.pad(
        torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32), dim=0), (1, 0),
    ).to(DEVICE, dtype=torch.int32)

def _make_host(tensor: torch.Tensor):
    return torch.empty_like(tensor, device="cpu", pin_memory=True)

class Context:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        nhead_k: int,
        head_dim: int,
        max_batch_size: int = 64,
        max_seq_len: int = 8192,
    ):
        self.k_cache = _make_paged_cache(num_pages, page_size, nhead_k, head_dim)
        self.v_cache = _make_paged_cache(num_pages, page_size, nhead_k, head_dim)
        self.k_host_cache = _make_host(self.k_cache)
        self.v_host_cache = _make_host(self.v_cache)
        self.page_table = torch.randint(
            num_pages // 2 + 1, num_pages,
            (max_batch_size, max_seq_len),
            dtype=torch.int32,
            device=DEVICE,
            requires_grad=False
        )
        self.indices = torch.arange(
            0, (num_pages // 2),
            dtype=torch.int64,
            device=DEVICE,
            requires_grad=False
        )
        self.page_size = page_size
        self.head_dim = head_dim
        self.nhead_k = nhead_k

class Batch:
    @overload
    def __init__(self, is_decode: Literal[True], *, seq_lens_q: List[int] | None = None, seq_lens_k: List[int]):
        ...

    @overload
    def __init__(self, is_decode: Literal[False], *, seq_lens_q: List[int], seq_lens_k: List[int] | None = None):
        ...

    def __init__(
        self,
        is_decode: bool,
        seq_lens_q: List[int] | None = None,
        seq_lens_k: List[int] | None = None,
    ):
        if is_decode:
            assert seq_lens_k is not None, "For decode phase, seq_lens_k must be provided."
            self.seq_lens_q = [1] * len(seq_lens_k)
            self.seq_lens_k = seq_lens_k
        else:
            assert seq_lens_q is not None, "For prefill phase, seq_lens_q must be provided."
            self.seq_lens_q = seq_lens_q
            self.seq_lens_k = seq_lens_k or seq_lens_q
        assert len(self.seq_lens_q) == len(self.seq_lens_k), "seq_lens_q and seq_lens_k must have the same length."
        self.batch_size = len(self.seq_lens_q)
        self.cu_seqlens_q = _make_cu_seqlens(self.seq_lens_q)
        self.cu_seqlens_k = _make_cu_seqlens(self.seq_lens_k)
        self.max_seqlen_q = max(self.seq_lens_q)
        self.max_seqlen_k = max(self.seq_lens_k)
        self.q = torch.zeros(
            (sum(self.seq_lens_q), 32, 128),
            dtype=torch.float16,
            device=DEVICE,
            requires_grad=False
        )
        self.cache_seqlens = torch.tensor(
            self.seq_lens_k, dtype=torch.int32, device=DEVICE, requires_grad=False
        )

def run_batch(ctx: Context, batch: Batch, sm_margin: int = 0, layers: int = 32):
    assert batch.batch_size <= ctx.page_table.shape[0]
    assert batch.max_seqlen_k <= ctx.page_table.shape[1]
    page_table = ctx.page_table[:batch.batch_size, :batch.max_seqlen_k]
    for i in range(layers):
        flash_attn_with_kvcache(
            q=batch.q.contiguous().half(),
            k_cache=ctx.k_cache[i],
            v_cache=ctx.v_cache[i],
            page_table=page_table,
            cache_seqlens=batch.cache_seqlens,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k_new=batch.cu_seqlens_k,
            max_seqlen_q=batch.max_seqlen_q,
            softmax_scale=1.0 / math.sqrt(128),
            causal=True,
            sm_margin=sm_margin,
        )

def run_copy(ctx: Context, batch: Batch):
    stream = torch.cuda.current_stream(DEVICE)
    assert batch.batch_size <= ctx.page_table.shape[0]
    assert batch.max_seqlen_k <= ctx.page_table.shape[1]
    indices = ctx.indices.contiguous().view(-1)

    transfer_kv_all_layer(
        src_k=ctx.k_cache,
        dst_k=ctx.k_host_cache,
        src_v=ctx.v_cache,
        dst_v=ctx.v_host_cache,
        src_indices=indices,
        dst_indices=indices.clone(),
        io_backend="kernel",
        page_size=1,
        src_layer_offset=math.prod(ctx.k_cache.shape[1:]),
        dst_layer_offset=math.prod(ctx.k_host_cache.shape[1:]),
        item_size=(ctx.head_dim * ctx.nhead_k),
        num_layers=ctx.k_cache.shape[0],
    )


def copy_thread(ctx: Context, batch: Batch, stream):
    print("Copying KV cache to host...")
    time.sleep(0.1)
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    while True:
        with torch.cuda.stream(stream):
            tic.record(stream)
            run_copy(ctx, batch)
            toc.record(stream)
            stream.synchronize()
            elapsed = tic.elapsed_time(toc)
            print(f"Copy time: {elapsed:.2f} ms")


@lambda f: f()
@torch.no_grad()
def main():
    stream_a, stream_b, sm_a, sm_b = init_context(8)
    print(f"SM_a: {sm_a}, SM_b: {sm_b}")
    ctx = Context(num_pages=8192, page_size=8, nhead_k=8, head_dim=128)
    batch_decode = Batch(is_decode=True, seq_lens_k=[2048] * 8)

    threading.Thread(
        target=copy_thread, args=(ctx, batch_decode, stream_a), daemon=True
    ).start()

    print("Running batch decode...")
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    while True:
        with torch.cuda.stream(stream_b):
            tic.record(stream_b)
            run_batch(ctx, batch_decode, sm_margin=0)
            toc.record(stream_b)
            stream_b.synchronize()
            elapsed = tic.elapsed_time(toc)
            print(f"Batch decode time: {elapsed:.2f} ms")
