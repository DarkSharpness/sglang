"""LeapQuant 8-bit GDN state, PyTorch reference path (accuracy only, opt-in).

Enabled by SGLANG_LEAPQUANT_REF=1. Off = production code path untouched.

Semantics follow test/manual/leapquant/leapquant_ref.py (v2 rounding, `u_round="store"`), specialised to GDN:
scalar per-head log-decay g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), k heads shared by
HV / HK value heads (repeat-interleave), q scaled by K^-0.5 after l2norm (eps 1e-6 inside the sqrt).

Storage: the whole format lives inside the existing fp32 temporal pool slot. Each head's 64 KiB fp32 [V, K] region
holds (byte offsets for K = V = 128, r = 4, CAP = 32):
  Z int8 [V,K] | s fp32 [V] | c fp32 [K] | K~ fp16 [r,K] | U~ fp16 [r,V] | G fp32 [CAP] | k^ fp16 [CAP,K] |
  u fp16 [CAP,V] | l int32 [4]
so every slot copy (radix COW / tracking / HiCache) is a byte copy of the format, and a zeroed slot decodes to the
zero state. Prefill temporarily turns the batch's slots back into fp32 (materialize), runs the production extend
kernel, and packs them again (plus any tracking slot the extend wrote in fp32).

All ops are batched over (requests, heads) and free of host syncs and data-dependent shapes, so they can be
captured in CUDA graphs (refit is computed for every row and selected with a mask).
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
_LOGGED = set()


def _log_once(path: str, **info) -> None:
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    key = (path, capturing)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            f"LEAPQUANT_PATH {path} graph_capture={capturing} "
            + " ".join(f"{k}={v}" for k, v in info.items())
        )


ENABLED = os.environ.get("SGLANG_LEAPQUANT_REF", "0") == "1"
P = int(os.environ.get("SGLANG_LEAPQUANT_P", "16"))
R = int(os.environ.get("SGLANG_LEAPQUANT_R", "4"))
# refit mode: "warm2" (2 iterations warm-started from K~^T) or "cold1" (1 iteration from the fixed Q0, official)
REFIT = os.environ.get("SGLANG_LEAPQUANT_REFIT", "warm2")
assert REFIT in ("warm2", "cold1"), REFIT
POWER_ITERS = {"warm2": 2, "cold1": 1}[REFIT]
# per-layer phase offset: GDN layer i (pool index) refits its first window after (i mod P) tokens
PHASE_OFFSET = os.environ.get("SGLANG_LEAPQUANT_PHASE", "0") == "1"
FLOOR = float(os.environ.get("SGLANG_LEAPQUANT_FLOOR", "1e-6"))
CDT = (
    torch.float64
    if os.environ.get("SGLANG_LEAPQUANT_FP64", "0") == "1"
    else torch.float32
)
# Debug / falsification knobs (never on for scores).
DEBUG_ZERO_BASE = os.environ.get("SGLANG_LEAPQUANT_DEBUG_ZERO_BASE", "0") == "1"
DUMP_DIR = os.environ.get("SGLANG_LEAPQUANT_DUMP_DIR")
DUMP_LAYERS = tuple(
    int(x) for x in os.environ.get("SGLANG_LEAPQUANT_DUMP_LAYERS", "0,30").split(",")
)
DUMP_MAX = int(os.environ.get("SGLANG_LEAPQUANT_DUMP_MAX", "60"))
_DUMP_COUNT = {}
_COMMIT_COUNT = [0]
CAP = 72  # physical record slots per head (logical C = P + T <= CAP); 72 allows P up to 64 with T = 4
SEED = 0
L2_EPS = 1e-6


def set_compute_dtype(dtype: torch.dtype) -> None:
    """Tests switch between fp64 (exact match with the reference) and fp32 (production-like)."""
    global CDT
    CDT = dtype


# --------------------------------------------------------------------------------------------- byte layout


def _layout(V: int, K: int):
    spec = [
        ("Z", torch.int8, (V, K)),
        ("s", torch.float32, (V,)),
        ("c", torch.float32, (K,)),
        ("Kt", torch.float16, (R, K)),
        ("Ut", torch.float16, (R, V)),
        ("G", torch.float32, (CAP,)),
        ("kh", torch.float16, (CAP, K)),
        ("u", torch.float16, (CAP, V)),
        ("l", torch.int32, (4,)),
    ]
    out, off = [], 0
    for name, dt, shape in spec:
        n = 1
        for d in shape:
            n *= d
        nbytes = n * torch.tensor([], dtype=dt).element_size()
        out.append((name, dt, shape, off, nbytes))
        off += (nbytes + 15) // 16 * 16
    assert off <= V * K * 4, "LeapQuant format does not fit the fp32 slot"
    return out, off


def region(pool_layer: torch.Tensor) -> torch.Tensor:
    """[S, HV, V, K] fp32 pool of one layer -> uint8 [S, HV, LQB] view of the LeapQuant bytes."""
    S_, HV, V, K = pool_layer.shape
    assert pool_layer.dtype == torch.float32, "LeapQuant needs the fp32 state pool"
    assert pool_layer.stride()[1:] == (V * K, K, 1), (
        "LeapQuant needs a contiguous [HV, V, K] slot"
    )
    _, lqb = _layout(V, K)
    return pool_layer.view(torch.uint8).view(S_, HV, V * K * 4)[..., :lqb]


def _fields(rows: torch.Tensor, V: int, K: int) -> SimpleNamespace:
    """Typed views into contiguous rows [N, HV, LQB] uint8."""
    N, HV, _ = rows.shape
    lay, _ = _layout(V, K)
    f = {}
    for name, dt, shape, off, nbytes in lay:
        f[name] = rows[..., off : off + nbytes].view(dt).view(N, HV, *shape)
    return SimpleNamespace(**f)


def _safe_idx(idx: torch.Tensor) -> torch.Tensor:
    # padded rows (-1) read / write the reserved dummy slot 0
    idx = idx.to(torch.int64)
    return torch.where(idx < 0, torch.zeros_like(idx), idx)


_Q0_CACHE = {}


def seeded_q0(K: int, r: int, device, dtype) -> torch.Tensor:
    """Same tensor as leapquant_ref.seeded_q0 (CPU generator, fp64), cast."""
    key = (K, r, str(device), dtype)
    if key not in _Q0_CACHE:
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        q0 = torch.randn(K, r, generator=gen, dtype=torch.float64).to(
            torch.float32
        )  # fp32 values (oracle gdn_q0)
        _Q0_CACHE[key] = q0.to(device=device, dtype=dtype)
    return _Q0_CACHE[key]


# --------------------------------------------------------------------------------------------- math


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + L2_EPS)


def _base_dense(f) -> torch.Tensor:
    B = f.s.to(CDT).unsqueeze(-1) * f.Z.to(CDT) * f.c.to(CDT).unsqueeze(-2)
    if R > 0:
        B = B + f.Ut.to(CDT).transpose(-1, -2) @ f.Kt.to(CDT)
    return B


def _l_of(f) -> torch.Tensor:
    return f.l[:, 0, 0].to(torch.int64)  # [N]; all heads carry the same l


def _G_at(f, l: torch.Tensor) -> torch.Tensor:
    """G_l per (row, head) [N, HV] fp32 (0 when l == 0)."""
    idx = (l - 1).clamp(min=0).view(-1, 1, 1).expand(-1, f.G.shape[1], 1)
    Gl = torch.gather(f.G, 2, idx).squeeze(2)
    return torch.where((l > 0).view(-1, 1), Gl, torch.zeros_like(Gl))


def _materialize(f, l: torch.Tensor) -> torch.Tensor:
    """S_l = B exp(G_l) + sum_{j<l} u_j (k^_j exp(G_l - G_j))^T, [N, HV, V, K] in CDT."""
    Gl = _G_at(f, l).to(CDT)  # [N, HV]
    mask = (torch.arange(CAP, device=l.device).view(1, CAP) < l.view(-1, 1)).view(
        -1, 1, CAP
    )
    diff = torch.where(
        mask,
        Gl.unsqueeze(-1) - f.G.to(CDT),
        torch.full_like(f.G, -float("inf"), dtype=CDT),
    )
    w = torch.exp(diff)  # [N, HV, CAP]
    S = _base_dense(f) * torch.exp(Gl)[..., None, None]
    S = S + (f.u.to(CDT) * w.unsqueeze(-1)).transpose(-1, -2) @ f.kh.to(CDT)
    return S


def _orth(Y: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the columns of Y [..., K, r] with a positive R diagonal (= the reference's sign-fixed
    QR); modified Gram-Schmidt applied twice for fp32 robustness. A zero column stays zero."""
    for _ in range(2):
        cols = []
        for j in range(Y.shape[-1]):
            v = Y[..., j]
            for q in cols:
                v = v - (q * v).sum(-1, keepdim=True) * q
            n = v.norm(dim=-1, keepdim=True)
            v = torch.where(
                n > 0, v / n.clamp_min(torch.finfo(Y.dtype).tiny), torch.zeros_like(v)
            )
            cols.append(v)
        Y = torch.stack(cols, -1)
    return Y


def _refit(S: torch.Tensor, Q: Optional[torch.Tensor]):
    """DESIGN section 5 steps 2-5 for S [N, HV, V, K] (CDT); Q [N, HV, K, r] start block (None -> seeded Q_0)."""
    N, HV, V, K = S.shape
    out = {}
    if R > 0:
        if Q is None:
            Q = seeded_q0(K, R, S.device, CDT).expand(N, HV, K, R)
        Q = Q.to(CDT)
        St = S.transpose(-1, -2)
        for _ in range(POWER_ITERS):
            Q = _orth(St @ (S @ Q))
        Kt = Q.transpose(-1, -2).to(torch.float16)
        Ktc = Kt.to(CDT)
        Ut = (S @ Ktc.transpose(-1, -2)).transpose(-1, -2).to(torch.float16)
        Rm = S - Ut.to(CDT).transpose(-1, -2) @ Ktc
        out["Kt"], out["Ut"] = Kt, Ut
    else:
        Rm = S
    c = torch.sqrt(Rm.abs().mean(dim=-2)).clamp_min(FLOOR).to(torch.float32)
    Rp = Rm / c.to(CDT).unsqueeze(-2)
    s = (Rp.abs().amax(dim=-1) / 127.0).to(torch.float32)
    s = torch.where(s == 0, torch.ones_like(s), s)
    Z = torch.round(Rp / s.to(CDT).unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    out["Z"], out["s"], out["c"] = Z, s, c
    return out


def _assign_base(f, new: dict, mask: Optional[torch.Tensor]) -> None:
    for name, val in new.items():
        dst = getattr(f, name)
        if mask is None:
            dst.copy_(val)
        else:
            m = mask.view(-1, *([1] * (dst.dim() - 1)))
            dst.copy_(torch.where(m, val, dst))


def gdn_gates(a, b, A_log, dt_bias, beta_bf16=False):
    """g = -exp(A_log) softplus(a + dt_bias) (fp32), beta = sigmoid(b) (bf16-rounded on the decode path, as the
    production packed decode); a, b [..., HV]."""
    x = a.to(CDT) + dt_bias.to(CDT)
    g = (-torch.exp(A_log.to(CDT)) * F.softplus(x, beta=1.0, threshold=20.0)).float()
    beta = (
        torch.sigmoid(b.to(CDT))
        .to(torch.bfloat16 if beta_bf16 else torch.float32)
        .float()
    )
    return g, beta


def _verify_rows(f, q, k, v, g, beta, scale):
    """One verify window on gathered rows (DESIGN v2, parallel form). q, k [N,T,HV,K] raw (post-conv, GQA expanded),
    v [N,T,HV,V], g, beta [N,T,HV] fp32. Writes records l..l+T-1 into f; returns o [N,T,HV,V] (CDT)."""
    N, T, HV, K = k.shape
    dev = k.device
    l = _l_of(f)
    qd = (_l2norm(q.to(CDT)) * scale).transpose(1, 2)  # [N,HV,T,K]
    kd = (
        _l2norm(k.to(CDT)).to(torch.float16).to(CDT).transpose(1, 2)
    )  # rounded k^, used everywhere
    vv = v.to(CDT).transpose(1, 2)  # [N,HV,T,V]
    gt = g.float().transpose(1, 2)  # [N,HV,T] fp32
    bt = beta.to(CDT).transpose(1, 2)
    cur = _G_at(f, l)  # fp32
    Gs = []
    for t in range(T):
        cur = cur + gt[..., t]  # fp32 add, as the reference
        Gs.append(cur)
    Gd = torch.stack(Gs, -1)  # [N,HV,T] fp32
    Gdc = Gd.to(CDT)
    eG = torch.exp(Gdc).unsqueeze(-1)
    X = torch.cat([kd, qd], dim=2)  # [N,HV,2T,K]
    Xe = X * torch.cat([eG, eG], dim=2)
    base = f.s.to(CDT).unsqueeze(-1) * (
        f.Z.to(CDT) @ (f.c.to(CDT).unsqueeze(-2) * Xe).transpose(-1, -2)
    )
    if R > 0:
        base = base + f.Ut.to(CDT).transpose(-1, -2) @ (
            f.Kt.to(CDT) @ Xe.transpose(-1, -2)
        )
    base = base.transpose(-1, -2)  # [N,HV,2T,V]
    if DEBUG_ZERO_BASE:
        base = base * 0.0
    # committed records j < l
    maskc = (torch.arange(CAP, device=dev).view(1, CAP) < l.view(-1, 1)).view(
        N, 1, 1, CAP
    )
    dc = Gdc.unsqueeze(-1) - f.G.to(CDT).unsqueeze(-2)  # [N,HV,T,CAP]
    dec = torch.exp(torch.where(maskc, dc, torch.full_like(dc, -float("inf"))))
    coef = (X @ f.kh.to(CDT).transpose(-1, -2)) * torch.cat(
        [dec, dec], dim=2
    )  # [N,HV,2T,CAP]
    WO = base + coef @ f.u.to(CDT)
    W0, O0 = WO[:, :, :T], WO[:, :, T:]
    # intra-draft T x T
    ar = torch.arange(T, device=dev)
    lower = ar.view(T, 1) > ar.view(1, T)
    lower_eq = ar.view(T, 1) >= ar.view(1, T)
    dd = Gdc.unsqueeze(-1) - Gdc.unsqueeze(-2)  # [N,HV,m,j]
    ninf = torch.full_like(dd, -float("inf"))
    A = (kd @ kd.transpose(-1, -2)) * torch.exp(torch.where(lower, dd, ninf))
    Qk = (qd @ kd.transpose(-1, -2)) * torch.exp(torch.where(lower_eq, dd, ninf))
    Nl = bt.unsqueeze(-1) * A  # strictly lower -> nilpotent
    eye = torch.eye(T, dtype=CDT, device=dev)
    Minv, Pw = eye, eye
    for _ in range(T - 1):
        Pw = -(Nl @ Pw)
        Minv = Minv + Pw
    U = Minv @ (bt.unsqueeze(-1) * (vv - W0))  # [N,HV,T,V], unrounded
    O = O0 + Qk @ U
    # records at l .. l+T-1
    pos = (l.view(N, 1) + ar.view(1, T)).clamp(max=CAP - 1)  # [N,T]
    pg = pos.view(N, 1, T).expand(N, HV, T)
    f.G.scatter_(2, pg, Gd)
    f.kh.scatter_(2, pg.unsqueeze(-1).expand(N, HV, T, K), kd.to(torch.float16))
    f.u.scatter_(
        2, pg.unsqueeze(-1).expand(N, HV, T, vv.shape[-1]), U.to(torch.float16)
    )
    return O.transpose(1, 2)  # [N,T,HV,V]


def _commit_rows(f, n: torch.Tensor) -> None:
    """l += n; rows with l >= P are refit (warm start from K~^T) and l <- 0. Masked, sync-free."""
    l_new = _l_of(f) + n.to(torch.int64)
    lim = f.l[:, 0, 1].to(torch.int64)
    lim = torch.where(
        lim <= 0, torch.full_like(lim, P), lim
    )  # zeroed slot / no phase -> P
    need = l_new >= lim
    S = _materialize(f, l_new)
    warm = REFIT == "warm2" and R > 0
    new = _refit(S, f.Kt.to(CDT).transpose(-1, -2) if warm else None)
    _assign_base(f, new, need)
    _commit_rows.last_need = need
    f.l[:, :, 0] = (
        torch.where(need, torch.zeros_like(l_new), l_new).to(torch.int32).view(-1, 1)
    )
    f.l[:, :, 1] = (
        torch.where(need, torch.full_like(lim, P), lim).to(torch.int32).view(-1, 1)
    )


def _pack_rows(S: torch.Tensor, rows: torch.Tensor, layer_idx: int = 0) -> None:
    """fp32 state -> format with an empty window (seeded Q_0), written into rows."""
    V, K = S.shape[-2:]
    f = _fields(rows, V, K)
    _assign_base(f, _refit(S.to(CDT), None), None)
    f.G.zero_()
    f.kh.zero_()
    f.u.zero_()
    f.l.zero_()
    f.l[:, :, 1] = (layer_idx % P) if (PHASE_OFFSET and layer_idx % P != 0) else P


def _expand_qk(x: torch.Tensor, HV: int) -> torch.Tensor:
    HK = x.shape[-2]
    return x if HK == HV else x.repeat_interleave(HV // HK, dim=-2)


# --------------------------------------------------------------------------------------------- hooks


def verify_window(
    pool_layer,
    cache_indices,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    commit_n=None,
    beta_bf16=False,
):
    """q, k [N,T,HK,K] (post-conv), v [N,T,HV,V], a, b [N,T,HV]. Reads the slots, writes the draft records (and
    commits commit_n tokens when given, i.e. plain decode). Returns o [N,T,HV,V] in v's dtype."""
    N, T, HV, V = v.shape
    K = k.shape[-1]
    _log_once(
        "verify_window",
        T=T,
        N=N,
        HV=HV,
        commit=commit_n is not None,
        P=P,
        R=R,
        refit=REFIT,
        phase=PHASE_OFFSET,
        beta_bf16=beta_bf16,
        cdt=CDT,
    )
    reg = region(pool_layer)
    idx = _safe_idx(cache_indices[:N])
    rows = reg[idx]  # gathered copy [N, HV, LQB]
    f = _fields(rows, V, K)
    g, beta = gdn_gates(a, b, A_log, dt_bias, beta_bf16=beta_bf16)
    o = _verify_rows(f, _expand_qk(q, HV), _expand_qk(k, HV), v, g, beta, scale)
    if commit_n is not None:
        _commit_rows(f, commit_n)
    reg[idx] = rows
    return o.to(v.dtype)


def decode(layer, pool_layer, cache_indices, mixed_qkv, a, b):
    """Plain decode T = 1 on the packed post-conv mixed_qkv [B, q+k+v]; returns [B, 1, HV, V]."""
    Bsz = mixed_qkv.shape[0]
    q, k, v = torch.split(mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
    q = q.reshape(Bsz, 1, layer.num_q_heads, layer.head_q_dim)
    k = k.reshape(Bsz, 1, layer.num_k_heads, layer.head_k_dim)
    v = v.reshape(Bsz, 1, layer.num_v_heads, layer.head_v_dim)
    n = torch.ones(Bsz, dtype=torch.int64, device=mixed_qkv.device)
    return verify_window(
        pool_layer,
        cache_indices,
        q,
        k,
        v,
        a.view(Bsz, 1, -1),
        b.view(Bsz, 1, -1),
        layer.A_log,
        layer.dt_bias,
        layer.head_k_dim**-0.5,
        commit_n=n,
        beta_bf16=True,
    )


def target_verify(
    layer, pool_layer, cache_indices, query, key, value, a, b, draft_token_num
):
    """query/key [1, B*T, HK, K], value [1, B*T, HV, V], a/b [B*T, HV]; returns o [1, B*T, HV, V]."""
    T = draft_token_num
    Bsz = a.reshape(-1, layer.num_v_heads).shape[0] // T
    q = query.reshape(Bsz, T, layer.num_k_heads, layer.head_k_dim)
    k = key.reshape(Bsz, T, layer.num_k_heads, layer.head_k_dim)
    v = value.reshape(Bsz, T, layer.num_v_heads, layer.head_v_dim)
    o = verify_window(
        pool_layer,
        cache_indices,
        q,
        k,
        v,
        a.reshape(Bsz, T, -1),
        b.reshape(Bsz, T, -1),
        layer.A_log,
        layer.dt_bias,
        layer.head_k_dim**-0.5,
    )
    return o.reshape(1, Bsz * T, layer.num_v_heads, layer.head_v_dim)


def commit_after_verify(
    temporal_all,
    state_indices,
    last_correct_step_indices,
    track_indices=None,
    steps_to_track=None,
):
    """temporal_all [L, S, HV, V, K]. Accepted n = last_correct + 1: l += n, refit at l >= P. Tracking (prefix
    cache): the track slot gets the slot's bytes with l = l_old + steps_to_track + 1 (taken before the refit)."""
    Bsz = state_indices.shape[0]
    if Bsz == 0:
        return
    _log_once(
        "commit_after_verify",
        track=track_indices is not None and steps_to_track is not None,
    )
    idx = _safe_idx(state_indices)
    n = (last_correct_step_indices[:Bsz].to(torch.int64) + 1).clamp(min=0)
    V, K = temporal_all.shape[-2:]
    has_track = track_indices is not None and steps_to_track is not None
    if has_track:
        steps = steps_to_track[:Bsz].to(torch.int64)
        tvalid = (steps >= 0) & (track_indices[:Bsz] >= 0) & (state_indices >= 0)
        tdst = torch.where(
            tvalid, track_indices[:Bsz].to(torch.int64), torch.zeros_like(steps)
        )
    for layer_i in range(temporal_all.shape[0]):
        reg = region(temporal_all[layer_i])
        rows = reg[idx]
        f = _fields(rows, V, K)
        if has_track:
            trows = rows.clone()
            tf = _fields(trows, V, K)
            lt = torch.where(tvalid, _l_of(f) + steps + 1, _l_of(f))
            tf.l[:, :, 0] = lt.to(torch.int32).view(-1, 1)
            reg[tdst] = trows
        _commit_rows(f, n)
        reg[idx] = rows
        if DUMP_DIR is not None and _COMMIT_COUNT[0] < DUMP_MAX:
            need = _commit_rows.last_need
            logger.info(
                f"LEAPQUANT_REFITS commit={_COMMIT_COUNT[0]} layer_pool_idx={layer_i} "
                f"refits={int(need[state_indices >= 0].sum())} n={n.tolist()} l_after={_l_of(f).tolist()}"
            )
    if DUMP_DIR is not None:
        _COMMIT_COUNT[0] += 1


def unpack_slots_inplace(pool_layer, cache_indices) -> None:
    """Prefill entry: LeapQuant bytes -> fp32 state, in place (the extend kernel reads/writes fp32)."""
    if cache_indices.numel() == 0:
        return
    S_, HV, V, K = pool_layer.shape
    _log_once("unpack_prefill")
    idx = _safe_idx(cache_indices)
    rows = region(pool_layer)[idx]
    f = _fields(rows, V, K)
    S = _materialize(f, _l_of(f))
    pool_layer[idx] = S.to(torch.float32)


def pack_slots_inplace(
    pool_layer, slot_indices, tag: str = "pack_prefill", layer_idx: int = 0
) -> None:
    """Prefill exit: fp32 state -> LeapQuant bytes (empty window), in place."""
    if slot_indices is None or slot_indices.numel() == 0:
        return
    _log_once(tag)
    S_, HV, V, K = pool_layer.shape
    idx = _safe_idx(slot_indices.flatten())
    S = pool_layer[idx]
    rows = torch.zeros(
        (idx.shape[0], HV, region(pool_layer).shape[-1]),
        dtype=torch.uint8,
        device=S.device,
    )
    _pack_rows(S, rows, layer_idx)
    region(pool_layer)[idx] = rows


def debug_dump(tag: str, layer_id: int, out: torch.Tensor) -> None:
    """SGLANG_LEAPQUANT_DUMP_DIR: save the first DUMP_MAX outputs of the listed layers (eager runs only)."""
    if DUMP_DIR is None or layer_id not in DUMP_LAYERS:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    c = _DUMP_COUNT.get((tag, layer_id), 0)
    if c >= DUMP_MAX:
        return
    _DUMP_COUNT[(tag, layer_id)] = c + 1
    os.makedirs(DUMP_DIR, exist_ok=True)
    torch.save(
        out.detach().float().cpu(),
        os.path.join(DUMP_DIR, f"{tag}_L{layer_id}_{c:03d}.pt"),
    )
