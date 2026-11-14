"""Tests for the LeapQuant GDN reference hook (sglang/srt/layers/attention/linear/leapquant_gdn.py).

RANDOM DATA, not real activations. Run: python test_leapquant_gdn.py [quick]
1. fp64 hook vs leapquant_ref (KDA reference with the GDN scalar gate broadcast over K): verify (T = 4) / decode
   (T = 1) chains with random acceptance, outputs, records and the materialized state.
2. hook vs the production Triton GDN recurrence (same materialized start state): layout / gate / GQA conventions.
3. 1-layer chained drift (fp32 hook) vs the exact fp64 recurrence, plus per-step BF16 state for scale.
4. prefill round trip: unpack -> fp32 -> pack keeps the state (only quantization error) and tracking bytes.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leapquant_ref as ref  # noqa: E402

import sglang  # noqa: E402
from sglang.srt.layers.attention.linear import leapquant_gdn as lq  # noqa: E402

print("sglang from", sglang.__file__)
DEV = os.environ.get("LQ_TEST_DEVICE", "cuda")
K = V = 128
F64 = torch.float64


def make_layer(HK, HV, gen):
    # Qwen3-Next style init: A ~ U(1, 16), dt log-uniform in [1e-3, 1e-1], dt_bias = softplus^-1(dt)
    A_log = torch.log(torch.empty(HV).uniform_(1, 16, generator=gen)).to(DEV)
    dt = torch.exp(
        torch.empty(HV).uniform_(math.log(1e-3), math.log(1e-1), generator=gen)
    )
    dt_bias = (dt + torch.log(-torch.expm1(-dt))).to(DEV)
    return A_log, dt_bias


def make_inputs(N, T, HK, HV, gen):
    rn = lambda *s: torch.randn(*s, generator=gen).to(DEV)
    q = rn(N, T, HK, K).bfloat16()
    k = rn(N, T, HK, K).bfloat16()
    v = rn(N, T, HV, V).bfloat16()
    a = rn(N, T, HV).bfloat16()
    b = rn(N, T, HV).bfloat16()
    return q, k, v, a, b


def ref_inputs(q, k, v, a, b, A_log, dt_bias, HV):
    g, beta = lq.gdn_gates(a, b, A_log, dt_bias)  # [N,T,HV] fp32
    qe = lq._expand_qk(q.double(), HV)
    ke = lq._expand_qk(k.double(), HV)
    return qe, ke, v.double(), g.unsqueeze(-1).expand(*g.shape, K).contiguous(), beta


def exact_window(S, q, k, v, a, b, A_log, dt_bias, HV, n=None):
    qe, ke, vv, g, beta = ref_inputs(q, k, v, a, b, A_log, dt_bias, HV)
    return ref.exact_verify(S, qe, ke, vv, g, beta, n_acc=n)


def rel(a, b, dims=(-1,)):
    return (a - b).norm(dim=dims) / b.norm(dim=dims).clamp_min(1e-300)


def start_state(N, HV, HK, gen, A_log, dt_bias, tokens=64):
    S = torch.zeros(N, HV, V, K, dtype=F64, device=DEV)
    for _ in range(tokens // 8):
        q, k, v, a, b = make_inputs(N, 8, HK, HV, gen)
        S = exact_window(S, q, k, v, a, b, A_log, dt_bias, HV)[1]
    return S


def new_pool(N, HV, S0):
    pool = torch.zeros(N + 2, HV, V, K, dtype=torch.float32, device=DEV)
    idx = torch.arange(1, N + 1, device=DEV, dtype=torch.int32)
    pool[idx.long()] = S0.float()
    lq.pack_slots_inplace(pool, idx)
    return pool, idx


def test_vs_ref(steps=40):
    """fp64 compute: hook == GDN oracle (leapquant_ref.gdn_*) to rounding, for T = 4 (verify + commit) and T = 1
    (decode, bf16 beta), both refit modes, phase offset on/off."""
    lq.set_compute_dtype(F64)
    N, HK, HV = 3, 2, 4
    out = {}
    saved = (lq.REFIT, lq.POWER_ITERS, lq.PHASE_OFFSET)
    for refit in ("warm2", "cold1"):
        for phase in (False, True):
            lq.REFIT, lq.POWER_ITERS, lq.PHASE_OFFSET = (
                refit,
                {"warm2": 2, "cold1": 1}[refit],
                phase,
            )
            for T in (4, 1):
                cfg = ref.GDNConfig(
                    P=lq.P,
                    C=lq.P + 8,
                    r=lq.R,
                    refit=refit,
                    phase_offset=phase,
                    floor=lq.FLOOR,
                )
                layer_idx = 5
                st, A_log, dt_bias, S0, gen = ref.gdn_make_test_state(
                    N, HK, HV, cfg, layer_idx=layer_idx, seed=11, device=DEV
                )
                pool = torch.zeros(N + 2, HV, V, K, dtype=torch.float32, device=DEV)
                idx = torch.arange(1, N + 1, device=DEV, dtype=torch.int32)
                pool[idx.long()] = S0.float()
                lq.pack_slots_inplace(pool, idx, layer_idx=layer_idx)
                worst_o = worst_s = 0.0
                zmis = 0
                for _ in range(steps):
                    q, k, v, a, b = ref.gdn_random_inputs(N, T, HK, HV, gen, DEV)
                    if T == 1:
                        o_ref, _ = ref.gdn_decode(st, q, k, v, a, b, A_log, dt_bias)
                        o = lq.verify_window(
                            pool,
                            idx,
                            q,
                            k,
                            v.double(),
                            a,
                            b,
                            A_log,
                            dt_bias,
                            K**-0.5,
                            commit_n=torch.ones(N, dtype=torch.int64, device=DEV),
                            beta_bf16=True,
                        )
                    else:
                        n = torch.randint(1, T + 1, (N,), generator=gen).to(DEV)
                        o_ref, _ = ref.gdn_verify(st, q, k, v, a, b, A_log, dt_bias)
                        ref.gdn_commit(st, n)
                        o = lq.verify_window(
                            pool, idx, q, k, v.double(), a, b, A_log, dt_bias, K**-0.5
                        )
                        lq.commit_after_verify(pool.unsqueeze(0), idx, n - 1)
                    assert torch.isfinite(o).all() and torch.isfinite(o_ref).all()
                    worst_o = max(worst_o, float(rel(o, o_ref).max()))
                    f = lq._fields(lq.region(pool)[idx.long()], V, K)
                    worst_s = max(
                        worst_s,
                        float(
                            rel(
                                lq._materialize(f, lq._l_of(f)),
                                ref.gdn_materialize(st),
                                (-2, -1),
                            ).max()
                        ),
                    )
                    zmis += int((f.Z != st.Z).sum())
                    assert torch.equal(lq._l_of(f), st.l.long()), (
                        "window length differs"
                    )
                    assert torch.equal(f.l[:, 0, 1].long(), st.lim.long()), (
                        "window limit differs"
                    )
                key = f"refit={refit} phase={phase} T={T}"
                out[key] = (worst_o, worst_s, zmis, st.refits)
                print(
                    f"[vs GDN oracle fp64] {key}: worst out {worst_o:.2e}, worst state {worst_s:.2e}, "
                    f"Z mismatches {zmis}, refits {st.refits}"
                )
                assert worst_o < 1e-9 and worst_s < 1e-9, out
    lq.REFIT, lq.POWER_ITERS, lq.PHASE_OFFSET = saved
    lq.set_compute_dtype(torch.float32)
    return out


def test_vs_production_kernel():
    """Same start state (materialized LeapQuant state), one T = 4 window: hook vs the production Triton recurrence."""
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    lq.set_compute_dtype(torch.float32)
    gen = torch.Generator().manual_seed(2)
    N, HK, HV, T = 4, 16, 32, 4
    A_log, dt_bias = make_layer(HK, HV, gen)
    S0 = start_state(N, HV, HK, gen, A_log, dt_bias)
    pool, idx = new_pool(N, HV, S0)
    f = lq._fields(lq.region(pool)[idx.long()], V, K)
    S_mat = lq._materialize(f, lq._l_of(f)).float()
    q, k, v, a, b = make_inputs(N, T, HK, HV, gen)
    o_hook = lq.target_verify(
        SimpleLayer(HK, HV, A_log, dt_bias),
        pool,
        idx,
        q.reshape(1, N * T, HK, K),
        k.reshape(1, N * T, HK, K),
        v.reshape(1, N * T, HV, V),
        a.reshape(N * T, HV),
        b.reshape(N * T, HV),
        T,
    )
    prod_pool = torch.zeros(N + 1, HV, V, K, device=DEV)
    prod_pool[1:] = S_mat
    qsl = torch.arange(0, N * T + 1, T, device=DEV, dtype=torch.int32)
    o_prod = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a.reshape(N * T, HV),
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q.reshape(1, N * T, HK, K),
        k=k.reshape(1, N * T, HK, K),
        v=v.reshape(1, N * T, HV, V),
        b=b.reshape(N * T, HV),
        initial_state_source=prod_pool,
        initial_state_indices=torch.arange(1, N + 1, device=DEV, dtype=torch.int32),
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=qsl,
    )
    assert o_hook.shape == o_prod.shape, (o_hook.shape, o_prod.shape)
    e = rel(o_hook.float(), o_prod.float())
    print(
        f"[vs production Triton, same start state, T=4] out rel err mean {e.mean():.2e} max {e.max():.2e} "
        f"(bf16 outputs + fp16 records)"
    )
    assert e.max() < 2e-2, e.max()
    # decode T = 1 path shape
    mixed = torch.cat(
        [q[:, 0].reshape(N, -1), k[:, 0].reshape(N, -1), v[:, 0].reshape(N, -1)], -1
    )
    od = lq.decode(
        SimpleLayer(HK, HV, A_log, dt_bias), pool, idx, mixed, a[:, 0], b[:, 0]
    )
    assert od.shape == (N, 1, HV, V), od.shape


class SimpleLayer:
    def __init__(self, HK, HV, A_log, dt_bias):
        self.num_q_heads = self.num_k_heads = HK
        self.num_v_heads = HV
        self.head_q_dim = self.head_k_dim = K
        self.head_v_dim = V
        self.q_dim = self.k_dim = HK * K
        self.v_dim = HV * V
        self.A_log, self.dt_bias = A_log, dt_bias


def test_chain_drift(steps=2000, T=4):
    """fp32 hook, 1 layer, random acceptance, vs exact fp64; per-step BF16 state shown for scale."""
    lq.set_compute_dtype(torch.float32)
    gen = torch.Generator().manual_seed(3)
    N, HK, HV = 8, 16, 32
    A_log, dt_bias = make_layer(HK, HV, gen)
    S0 = start_state(N, HV, HK, gen, A_log, dt_bias, tokens=256)
    pool, idx = new_pool(N, HV, S0)
    S = S0.clone()
    Sb = S0.bfloat16().double()
    hist, histb = [], []
    for step in range(steps):
        q, k, v, a, b = make_inputs(N, T, HK, HV, gen)
        n = torch.randint(1, T + 1, (N,), generator=gen).to(DEV)
        o_ref, S = exact_window(S, q, k, v, a, b, A_log, dt_bias, HV, n)
        ob, Sb_full = exact_window(Sb, q, k, v, a, b, A_log, dt_bias, HV, n)
        Sb = (
            Sb_full.bfloat16().double()
        )  # bf16 per commit (coarser than per token; scale only)
        o = lq.verify_window(pool, idx, q, k, v, a, b, A_log, dt_bias, K**-0.5)
        lq.commit_after_verify(pool.unsqueeze(0), idx, n - 1)
        hist.append(float(rel(o.double(), o_ref).mean()))
        histb.append(float(rel(ob, o_ref).mean()))
        assert math.isfinite(hist[-1])
    q4 = steps // 4
    m = lambda h: sum(h[-q4:]) / q4
    f = lq._fields(lq.region(pool)[idx.long()], V, K)
    es = float(rel(lq._materialize(f, lq._l_of(f)), S, (-2, -1)).mean())
    print(
        f"[chain fp32, random data] {steps} windows T={T}: out rel err first-quarter {sum(hist[:q4]) / q4:.2e}, "
        f"last-quarter {m(hist):.2e}; state rel err end {es:.2e}; bf16-per-commit last-quarter {m(histb):.2e}"
    )
    assert m(hist) < 0.05 and es < 0.05


def test_prefill_round_trip():
    lq.set_compute_dtype(torch.float32)
    gen = torch.Generator().manual_seed(4)
    N, HK, HV = 3, 2, 4
    A_log, dt_bias = make_layer(HK, HV, gen)
    S0 = start_state(N, HV, HK, gen, A_log, dt_bias)
    pool, idx = new_pool(N, HV, S0)
    # a few windows so l > 0, then unpack: must equal the materialized state
    for _ in range(3):
        q, k, v, a, b = make_inputs(N, 4, HK, HV, gen)
        lq.verify_window(pool, idx, q, k, v, a, b, A_log, dt_bias, K**-0.5)
        lq.commit_after_verify(pool.unsqueeze(0), idx, torch.full((N,), 1, device=DEV))
    f = lq._fields(lq.region(pool)[idx.long()], V, K)
    S_lq = lq._materialize(f, lq._l_of(f)).float()
    # slot copy is a byte copy: copy slot 1 -> spare slot N+1 and compare decodes
    pool[N + 1] = pool[1]
    f2 = lq._fields(lq.region(pool)[torch.tensor([N + 1], device=DEV)], V, K)
    assert torch.equal(lq._materialize(f2, lq._l_of(f2)).float(), S_lq[:1])
    lq.unpack_slots_inplace(pool, idx)
    assert torch.equal(pool[idx.long()], S_lq)
    lq.pack_slots_inplace(pool, idx)
    f = lq._fields(lq.region(pool)[idx.long()], V, K)
    e = rel(lq._materialize(f, lq._l_of(f)).float(), S_lq, (-2, -1))
    print(
        f"[prefill round trip] unpack exact, pack rel err max {e.max():.2e}, l after pack {lq._l_of(f).tolist()}"
    )
    assert e.max() < 2e-2 and int(lq._l_of(f).abs().sum()) == 0
    # zeroed slot decodes to the zero state
    z = torch.zeros(1, HV, V, K, device=DEV)
    fz = lq._fields(lq.region(z)[torch.zeros(1, dtype=torch.long, device=DEV)], V, K)
    assert lq._materialize(fz, lq._l_of(fz)).abs().max() == 0


if __name__ == "__main__":
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    test_vs_ref(steps=20 if quick else 60)
    if DEV == "cuda":
        test_vs_production_kernel()
    test_prefill_round_trip()
    test_chain_drift(steps=200 if quick else 2000)
    print("ALL PASSED")
