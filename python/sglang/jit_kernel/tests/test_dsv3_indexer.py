"""
Tests for the DeepSeek V3.2 sparse indexer JIT kernel.

The indexer performs, for each batch element:
  1. Dequantize FP8 paged KV cache -> float32
  2. Compute attention logits: q @ K^T  (per-head)
  3. Apply ReLU and weighted sum across heads
  4. Select top-K indices (via torch.topk)

Reference implementation is pure PyTorch. The CUDA kernel fuses steps 1-3
into a single pass over the paged KV cache.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=300, suite="nightly-kernel-1-gpu", nightly=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64
# KV cache layout: [num_pages, 64, 1, 132] uint8
# Each token row: 128 bytes FP8 data + 4 bytes float32 scale = 132 bytes
BYTES_PER_TOKEN = HEAD_DIM + 4  # 132

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def make_kv_cache(num_pages: int) -> torch.Tensor:
    """Build a random FP8 KV cache [num_pages, 64, 1, 132] as uint8.

    Layout per page (64 tokens, 1 KV head):
      - bytes [0 : 64*128)  -> FP8 E4M3 key values  [64, 128]
      - bytes [64*128 : 64*132) -> float32 per-token scales [64]
    """
    k_index_cache_fp8 = torch.empty(
        num_pages, PAGE_SIZE, 1, BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda"
    )
    kv_flat = k_index_cache_fp8.view(num_pages, -1)

    # Fill FP8 data region
    kv_flat[:, : PAGE_SIZE * HEAD_DIM].view(torch.float8_e4m3fn).copy_(
        torch.randn(
            num_pages, PAGE_SIZE * HEAD_DIM, dtype=torch.float32, device="cuda"
        ).to(torch.float8_e4m3fn)
    )
    # Fill scale region (positive values)
    kv_flat[:, PAGE_SIZE * HEAD_DIM :].view(torch.float32).copy_(
        torch.randn(num_pages, PAGE_SIZE, dtype=torch.float32, device="cuda").abs()
    )
    return kv_flat.view(num_pages, PAGE_SIZE, 1, BYTES_PER_TOKEN)


def make_test_data(batch_size: int, seq_len_range: tuple[int, int]):
    """Build random test inputs for the DSV3 indexer.

    Returns:
        (q, k_cache, weights, seq_lens, block_table)
    """
    lo, hi = seq_len_range
    seq_lens = torch.randint(
        lo, hi + 1, (batch_size,), dtype=torch.int32, device="cuda"
    )

    num_pages_per_seq = ((seq_lens + PAGE_SIZE - 1) // PAGE_SIZE).sum().item()
    max_num_pages = int((int(seq_lens.max().item()) + PAGE_SIZE - 1) // PAGE_SIZE)
    # Ensure max_num_pages is divisible by 2 (kernel requirement)
    max_num_pages = (max_num_pages + 1) // 2 * 2

    q = torch.randn(
        batch_size, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device="cuda"
    ).to(torch.float8_e4m3fn)
    k_cache = make_kv_cache(int(num_pages_per_seq))
    weights = torch.randn(batch_size, NUM_HEADS, dtype=torch.float32, device="cuda")
    block_table = torch.zeros(
        batch_size, max_num_pages, dtype=torch.int32, device="cuda"
    )

    page_offset = 0
    for b in range(batch_size):
        n = int((int(seq_lens[b].item()) + PAGE_SIZE - 1) // PAGE_SIZE)
        block_table[b, :n] = torch.arange(
            page_offset, page_offset + n, dtype=torch.int32, device="cuda"
        )
        page_offset += n

    return q, k_cache, weights, seq_lens, block_table


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch)
# ---------------------------------------------------------------------------


def dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 KV cache.

    Input:  [num_pages, 64, 1, 132] uint8
    Output: [num_pages, 64, 128] float32 (dequantized keys)

    Each token's 132 bytes = 128 FP8 bytes + 4 scale bytes (float32).
    Dequantized value = fp8_value.to(float32) * scale
    """
    k = k_index_cache_fp8.view(torch.uint8)
    num_pages = k.shape[0]
    kv_flat = k.view(num_pages, PAGE_SIZE * BYTES_PER_TOKEN)

    # Extract FP8 data: first PAGE_SIZE * HEAD_DIM bytes
    fp8_bytes = kv_flat[:, : PAGE_SIZE * HEAD_DIM].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, PAGE_SIZE, HEAD_DIM).view(
        torch.float8_e4m3fn
    )
    fp8_float = fp8_tensor.to(torch.float32)

    # Extract scales: remaining bytes as float32
    scale_bytes = kv_flat[:, PAGE_SIZE * HEAD_DIM :].contiguous()
    scale = scale_bytes.view(num_pages, PAGE_SIZE, 4).view(
        torch.float32
    )  # [num_pages, 64, 1]

    return fp8_float * scale


@torch.no_grad()
def dsv3_indexer_ref(q_fp8, k_cache_fp8, weights, seq_lens, block_table):
    """Pure-PyTorch reference: compute weighted ReLU attention logits.

    Args:
        q_fp8:       [batch, 64, 128] fp8
        k_cache_fp8: [num_pages, 64, 1, 132] uint8
        weights:     [batch, 64] float32
        seq_lens:    [batch] int32
        block_table: [batch, max_num_pages] int32

    Returns:
        logits: list of [seq_len] float32 tensors per batch
    """
    batch_size, num_heads, head_dim = q_fp8.shape
    q = q_fp8.to(torch.float32)
    K_all = dequant_fp8_kv_cache(k_cache_fp8)  # [num_pages, 64, 128]

    logits = []
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            logits.append(torch.zeros(0, device=q.device))
            continue

        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]  # [num_pages_for_seq, 64, 128]
        K = K_paged.reshape(-1, head_dim)[:seq_len]  # [seq_len, 128]

        q_b = q[b]  # [64, 128]
        scores = q_b @ K.T  # [64, seq_len]
        scores_relu = torch.relu(scores)

        w = weights[b]  # [64]
        final_scores = (scores_relu * w[:, None]).sum(dim=0)  # [seq_len]

        logits.append(final_scores)

    return logits


@torch.no_grad()
def dsv3_topk_indexer_ref(q_fp8, k_cache_fp8, weights, seq_lens, block_table, topk):
    """Pure-PyTorch reference: logits + top-K selection.

    Returns:
        topk_indices: [batch, topk] int32
        logits: list of [seq_len] float32 tensors per batch
    """
    batch_size = q_fp8.shape[0]
    device = q_fp8.device

    logits = dsv3_indexer_ref(q_fp8, k_cache_fp8, weights, seq_lens, block_table)
    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue
        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(logits[b], actual_topk)
        topk_indices[b, :actual_topk] = topk_idx.to(torch.int32)

    return topk_indices, logits


# ---------------------------------------------------------------------------
# Tests -- reference implementation sanity checks
# ---------------------------------------------------------------------------


class TestDequantFP8:
    """Verify FP8 dequantization is self-consistent."""

    def test_dequant_shape(self):
        k_cache = make_kv_cache(4)
        result = dequant_fp8_kv_cache(k_cache)
        assert result.shape == (4, PAGE_SIZE, HEAD_DIM)
        assert result.dtype == torch.float32

    def test_dequant_nonzero(self):
        k_cache = make_kv_cache(2)
        result = dequant_fp8_kv_cache(k_cache)
        assert result.abs().sum().item() > 0, "Dequantized cache is all zeros"


class TestReferenceIndexer:
    """Verify the reference implementation produces valid outputs."""

    @pytest.mark.parametrize("batch_size", [1, 4, 64, 256])
    @pytest.mark.parametrize("seq_len", [4096, 8192])
    @pytest.mark.parametrize("topk", [512, 1024, 2048])
    def test_output_shapes(self, batch_size, seq_len, topk):
        data = make_test_data(batch_size, (seq_len, seq_len))
        indices, logits = dsv3_topk_indexer_ref(*data, topk=topk)

        assert indices.shape == (batch_size, topk)
        assert indices.dtype == torch.int32
        assert len(logits) == batch_size

        seq_lens = data[3]
        for b in range(batch_size):
            sl = int(seq_lens[b].item())
            assert logits[b].shape == (sl,)

    @pytest.mark.parametrize("topk", [512, 1024, 2048])
    def test_indices_in_range(self, topk):
        batch_size = 2
        seq_len = 8192
        data = make_test_data(batch_size, (seq_len, seq_len))
        indices, logits = dsv3_topk_indexer_ref(*data, topk=topk)

        seq_lens = data[3]
        for b in range(batch_size):
            sl = int(seq_lens[b].item())
            valid = indices[b][indices[b] >= 0]
            assert (valid < sl).all(), f"Batch {b}: indices out of range"

    @pytest.mark.parametrize("topk", [512, 1024, 2048])
    def test_topk_values_sorted_desc(self, topk):
        """Top-K indices should correspond to the largest logit values."""
        data = make_test_data(1, (8192, 8192))
        indices, logits = dsv3_topk_indexer_ref(*data, topk=topk)

        logit_vals = logits[0]
        topk_vals = logit_vals[indices[0][indices[0] >= 0].long()]
        all_vals_sorted = torch.sort(logit_vals, descending=True)[0]
        kth_val = all_vals_sorted[topk - 1]
        assert topk_vals.min() >= kth_val - 1e-5

    @pytest.mark.parametrize("seq_len", [100, 256, 511])
    @pytest.mark.parametrize("topk", [512, 1024, 2048])
    def test_short_sequences(self, seq_len, topk):
        """When seq_len < topk, all indices should be selected."""
        data = make_test_data(1, (seq_len, seq_len))
        indices, logits = dsv3_topk_indexer_ref(*data, topk=topk)

        sl = int(data[3][0].item())
        valid = indices[0][indices[0] >= 0]
        expected_count = min(topk, sl)
        assert len(valid) == expected_count

    def test_logits_nonnegative_with_positive_weights(self):
        """With positive weights and ReLU, all logits should be >= 0."""
        data = make_test_data(1, (4096, 4096))
        q, k_cache, weights, seq_lens, block_table = data
        weights = weights.abs()
        logits = dsv3_indexer_ref(q, k_cache, weights, seq_lens, block_table)
        assert (logits[0] >= -1e-6).all()


# ---------------------------------------------------------------------------
# Tests -- kernel vs reference (logits only, topk via torch.topk)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 4, 32])
@pytest.mark.parametrize("L", [4096, 8192, 4096 * 8])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_dsv3_indexer_kernel(B, L, topk):
    """Compare the CUDA kernel logits against the PyTorch reference,
    then apply torch.topk on both and compare."""
    try:
        from sglang.jit_kernel.dsv3_indexer import dsv3_indexer
    except ImportError:
        pytest.skip("dsv3_indexer kernel not yet implemented")
    try:
        # Trigger JIT compilation; may fail on non-SM100a machines
        from sglang.jit_kernel.dsv3_indexer import can_use_dsv3_indexer

        if not can_use_dsv3_indexer():
            pytest.skip("dsv3_indexer requires SM100a (Blackwell)")
    except Exception:
        pytest.skip("dsv3_indexer compilation failed (requires SM100a)")

    lo = max(topk + 1, int(L * 0.7))
    hi = min(1 << 19, int(L * 1.3))
    data = make_test_data(B, (lo, hi))

    q, kv_cache, weights, seq_lens, block_table = data
    logits_ref = dsv3_indexer_ref(*data)
    logits_kernel = dsv3_indexer(
        q, kv_cache, weights, seq_lens, block_table, max_model_len=hi
    )

    for i in range(B):
        prefix = f"B={B}, L={L}, topk={topk}, batch {i}"
        sl = int(seq_lens[i].item())

        logit_ref = logits_ref[i][:sl]
        logit_k = logits_kernel[i, :sl]

        # Logit difference tolerance (FP8 dequant + GEMM accumulation)
        diff = float((logit_k - logit_ref).abs().max())
        assert diff <= 1.0, f"{prefix}: kernel logit max-diff {diff}"

        # Top-K comparison using torch.topk on both
        actual_topk = min(topk, sl)
        _, inds_ref = torch.topk(logit_ref, actual_topk)
        _, inds_k = torch.topk(logit_k, actual_topk)

        ref_topk_vals = torch.sort(logit_ref[inds_ref])[0]
        kernel_topk_vals = torch.sort(logit_k[inds_k])[0]
        topk_diff = float((ref_topk_vals - kernel_topk_vals).abs().max())
        assert topk_diff <= 0.5, f"{prefix}: topk max-diff {topk_diff}"


# ---------------------------------------------------------------------------
# Tests -- indexer metadata (SM mapping)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 4, 32, 128])
def test_indexer_metadata(B):
    """Test SM mapping metadata computation."""
    try:
        from sglang.jit_kernel.dsv3_indexer import (
            can_use_dsv3_indexer,
            get_indexer_metadata,
        )
    except ImportError:
        pytest.skip("get_indexer_metadata kernel not yet implemented")
    try:
        if not can_use_dsv3_indexer():
            pytest.skip("dsv3_indexer requires SM100a (Blackwell)")
    except Exception:
        pytest.skip("dsv3_indexer compilation failed (requires SM100a)")

    seq_lens = torch.randint(1024, 65536, (B,), dtype=torch.int32, device="cuda")
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    sm_map = get_indexer_metadata(seq_lens, num_sms)

    # Shape check: output is exactly num_sms rows
    assert sm_map.shape == (num_sms, 4)
    assert sm_map.dtype == torch.int32

    # Each SM should map to a valid batch index
    batch_ids = sm_map[:, 0]
    assert (batch_ids >= 0).all()
    assert (batch_ids < B).all()

    # Page offsets and num_pages should be non-negative
    assert (sm_map[:, 2] >= 0).all()
    assert (sm_map[:, 3] >= 0).all()


# ---------------------------------------------------------------------------
# Tests -- cluster-based radix top-K kernel
# ---------------------------------------------------------------------------


def _skip_unless_topk():
    try:
        from sglang.jit_kernel.dsv3_indexer import can_use_dsv3_indexer

        if not can_use_dsv3_indexer():
            pytest.skip("dsv3 kernels require SM100a (Blackwell)")
    except Exception:
        pytest.skip("dsv3 kernel compilation failed (requires SM100a)")


def _check_topk(logits, seq_lens, K, indices):
    """Verify kernel top-K indices match torch.topk reference values."""
    B = logits.shape[0]
    for b in range(B):
        sl = int(seq_lens[b].item())
        actual_k = min(K, sl)

        ref_vals, _ = torch.topk(logits[b, :sl], actual_k)
        ref_sorted = ref_vals.sort(descending=True).values

        valid_idx = indices[b, :actual_k]
        assert (valid_idx >= 0).all(), f"batch {b}: -1 in valid range"
        assert (valid_idx < sl).all(), f"batch {b}: index >= seq_len"

        got_sorted = logits[b, valid_idx.long()].sort(descending=True).values
        diff = float((ref_sorted - got_sorted).abs().max())
        assert diff <= 1e-4, f"batch {b}: value diff {diff:.6f}"

        if actual_k < K:
            assert (indices[b, actual_k:] == -1).all(), f"batch {b}: bad padding"


@pytest.mark.parametrize("B", [1, 4, 32, 128])
@pytest.mark.parametrize("L", [4096, 8192, 32768, 131072])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_dsv3_topk_kernel(B, L, topk):
    """Compare cluster radix top-K against torch.topk."""
    _skip_unless_topk()
    from sglang.jit_kernel.dsv3_indexer import dsv3_topk

    logits = torch.randn(B, L, device="cuda", dtype=torch.float32)
    seq_lens = torch.full((B,), L, device="cuda", dtype=torch.int32)

    indices = dsv3_topk(logits, seq_lens, topk)
    torch.cuda.synchronize()

    _check_topk(logits, seq_lens, topk, indices)


@pytest.mark.parametrize("nc", [1, 2, 4, 8])
def test_dsv3_topk_cluster_sizes(nc):
    """Verify correctness at each cluster size."""
    _skip_unless_topk()
    from sglang.jit_kernel.dsv3_indexer import dsv3_topk

    B, L, K = 4, 16384, 2048
    logits = torch.randn(B, L, device="cuda", dtype=torch.float32)
    seq_lens = torch.full((B,), L, device="cuda", dtype=torch.int32)

    indices = dsv3_topk(logits, seq_lens, K, num_clusters=nc)
    torch.cuda.synchronize()

    _check_topk(logits, seq_lens, K, indices)


@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_dsv3_topk_varlen(topk):
    """Variable-length sequences within a batch."""
    _skip_unless_topk()
    from sglang.jit_kernel.dsv3_indexer import dsv3_topk

    B = 16
    lo, hi = topk + 1, 32768
    seq_lens = torch.randint(lo, hi, (B,), dtype=torch.int32, device="cuda")
    max_len = int(seq_lens.max().item())

    logits = torch.randn(B, max_len, device="cuda", dtype=torch.float32)
    indices = dsv3_topk(logits, seq_lens, topk)
    torch.cuda.synchronize()

    _check_topk(logits, seq_lens, topk, indices)


@pytest.mark.parametrize("sl", [64, 256, 511])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_dsv3_topk_short_seq(sl, topk):
    """When seq_len < K, all valid indices should be selected and rest padded."""
    _skip_unless_topk()
    from sglang.jit_kernel.dsv3_indexer import dsv3_topk

    logits = torch.randn(2, sl, device="cuda", dtype=torch.float32)
    seq_lens = torch.tensor([sl, sl], device="cuda", dtype=torch.int32)

    indices = dsv3_topk(logits, seq_lens, topk, num_clusters=1)
    torch.cuda.synchronize()

    _check_topk(logits, seq_lens, topk, indices)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
