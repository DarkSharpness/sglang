"""LeapQuant KDA target-verify reference (DESIGN.md sections 2-5), PyTorch, no sglang dependency.

Everything is computed in fp64 between the specified rounding points:
  - G (cumulative log-decay since window start): fp32, G_m = fp32(G_{m-1} + fp32(g_m))
  - record k^: fp16, record u: fp16 (rounded at once; the rounded value is used everywhere)
  - compensators K~, U~: fp16 (U~ fitted against the rounded K~)
  - smoothing c, row scale s: fp32 (Z is quantised against the rounded c and s)
  - Z: int8, round-half-even, clamp +-127
With LQConfig(exact=True) all storage is fp64 and Z holds the unrounded R'/s (lossless format), which is
what the self-test compares against the exact recurrence.

Layouts (N requests, H heads, T verify tokens, K = V = 128, C record slots, r compensators):
  inputs  q, k, g: [N, T, H, K]; v: [N, T, H, V]; beta: [N, T, H]   (token-major, = production [N*T, H, D])
  output  o: [N, T, H, V]
  state   S: [N, H, V, K]  (value rows, key columns, K contiguous; production layout)
  LQState fields: Z [N,H,V,K], s [N,H,V], c [N,H,K], Kt [N,H,r,K], Ut [N,H,r,V],
                  G [N,H,C,K], kh [N,H,C,K], u [N,H,C,V], l [N] int32.
  Record slot j (0-based) holds window position j+1.  Verify writes slots l..l+T-1 and never touches l or
  the base; commit advances l and refits when l >= P.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

F64 = torch.float64
L2_EPS = 1e-6


# --------------------------------------------------------------------------------------------- helpers


def l2norm(x: torch.Tensor, eps: float = L2_EPS) -> torch.Tensor:
    """Production form (kda_decode_mtp.py:424/481): x * rsqrt(sum(x^2) + eps)."""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def prep_qk(q: torch.Tensor, k: torch.Tensor):
    """Raw post-conv q, k -> (q^, k^) in fp64: l2norm, q scaled by K^-0.5."""
    q = q.to(F64)
    k = k.to(F64)
    return l2norm(q) * q.shape[-1] ** -0.5, l2norm(k)


def apply_onorm(
    o: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Gated output RMSNorm as fused in production (kda_decode_mtp.py:763-783): o * rsqrt(mean(o^2)+eps) * w * sigmoid(gate)."""
    o = o.to(F64)
    rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
    return o * rms * weight.to(F64) * torch.sigmoid(gate.to(F64))


def _rnd(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round to dtype, return fp64 (no-op for fp64)."""
    return x if dtype == F64 else x.to(dtype).to(F64)


def _tok_major_to_head_major(x: torch.Tensor) -> torch.Tensor:
    # [N, T, H, D] -> [N, H, T, D]; [N, T, H] -> [N, H, T]
    return x.transpose(1, 2)


def seeded_q0(K: int, r: int, seed: int = 0) -> torch.Tensor:
    """Fixed Gaussian start block Q_0 [K, r] fp64 (CPU generator, shared by all heads). The kernel test should
    pass exactly this tensor (cast to its working dtype) to its pack/refit when there is no prior K~."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(K, r, generator=gen, dtype=F64)


def orth(Y: torch.Tensor) -> torch.Tensor:
    """Canonical orthonormal basis of the columns of Y [..., K, r]: QR with a non-negative R diagonal.
    Any QR (Householder, MGS, Cholesky-QR) of a full-rank Y gives the same Q once signs are fixed this way."""
    Q, R = torch.linalg.qr(Y)
    d = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    return Q * d.unsqueeze(-2)


# --------------------------------------------------------------------------------------------- exact recurrence


def exact_verify(S, q, k, v, g, beta, n_acc=None):
    """Exact fp64 KDA recurrence over one verify window (DESIGN.md section 2).
    S [N,H,V,K]; inputs token-major. Returns (o [N,T,H,V] fp64, S_commit [N,H,V,K]) where S_commit is the state
    after n_acc[n] tokens (n_acc None -> after all T)."""
    S = S.to(F64)
    qh, kh = prep_qk(q, k)
    qh, kh = _tok_major_to_head_major(qh), _tok_major_to_head_major(kh)
    v = _tok_major_to_head_major(v.to(F64))
    g = _tok_major_to_head_major(g.to(F64))
    beta = _tok_major_to_head_major(beta.to(F64))
    N, H, T, _ = qh.shape
    if n_acc is None:
        n_acc = torch.full((N,), T, device=S.device, dtype=torch.int64)
    n_acc = n_acc.to(S.device)
    outs = []
    S_commit = S.clone()
    for t in range(T):
        S = S * torch.exp(g[:, :, t]).unsqueeze(-2)
        u = beta[:, :, t, None] * (
            v[:, :, t] - torch.einsum("nhvk,nhk->nhv", S, kh[:, :, t])
        )
        S = S + u.unsqueeze(-1) * kh[:, :, t].unsqueeze(-2)
        outs.append(torch.einsum("nhvk,nhk->nhv", S, qh[:, :, t]))
        sel = (n_acc == t + 1).view(N, 1, 1, 1)
        S_commit = torch.where(sel, S, S_commit)
    o = torch.stack(outs, dim=1)  # [N, T, H, V]
    return o, S_commit


def exact_tokens(S, q, k, v, g, beta):
    """Plain sequential recurrence over all tokens (e.g. a synthetic 'prefill'); returns the final state."""
    return exact_verify(S, q, k, v, g, beta)[1]


# --------------------------------------------------------------------------------------------- LeapQuant state


@dataclass(frozen=True)
class LQConfig:
    P: int = 16  # refit when l >= P
    C: int | None = None  # record capacity, default P + 8
    r: int = 4  # compensator rank
    smoothing: bool = True
    floor: float = 1e-6  # smoothing floor: c = max(sqrt(mean|R|), floor)
    power_iters: int = 4
    seed: int = 0  # seeded Q_0
    exact: bool = False  # lossless storage (fp64 everywhere, Z unrounded)
    T: int = 8  # max verify window
    base_x_fp16: bool = False  # verify: round c*exp(G)*x to fp16 before the Z product (single f16 MMA operand)
    # "store" (v2): U of one verify in parallel (WY/UT) form, unrounded inside the verify, fp16 only when stored.
    # "step" (v1): sequential, u_m rounded to fp16 at once and the rounded value fed to later drafts and o.
    u_round: str = "store"

    @property
    def cap(self) -> int:
        return self.C if self.C is not None else self.P + self.T

    # storage dtypes
    @property
    def dt_z(self):
        return F64 if self.exact else torch.int8

    @property
    def dt_scale(self):
        return F64 if self.exact else torch.float32

    @property
    def dt_comp(self):
        return F64 if self.exact else torch.float16

    @property
    def dt_G(self):
        return F64 if self.exact else torch.float32

    @property
    def dt_rec(self):
        return F64 if self.exact else torch.float16


@dataclass
class LQState:
    cfg: LQConfig
    Z: torch.Tensor  # [N,H,V,K] int8 (fp64 if exact)
    s: torch.Tensor  # [N,H,V] fp32
    c: torch.Tensor  # [N,H,K] fp32
    Kt: torch.Tensor  # [N,H,r,K] fp16
    Ut: torch.Tensor  # [N,H,r,V] fp16
    G: torch.Tensor  # [N,H,C,K] fp32
    kh: torch.Tensor  # [N,H,C,K] fp16
    u: torch.Tensor  # [N,H,C,V] fp16
    l: torch.Tensor  # [N] int32
    stats: dict = field(default_factory=lambda: {"refits": 0})

    def clone(self) -> LQState:
        return LQState(
            self.cfg,
            *(
                getattr(self, f).clone()
                for f in ("Z", "s", "c", "Kt", "Ut", "G", "kh", "u", "l")
            ),
            stats=dict(self.stats),
        )

    def nbytes_per_head(self) -> int:
        n_heads = self.Z.shape[0] * self.Z.shape[1]
        tot = sum(
            getattr(self, f).numel() * getattr(self, f).element_size()
            for f in ("Z", "s", "c", "Kt", "Ut", "G", "kh", "u")
        )
        return tot // n_heads


def _base_dense(Z, s, c, Kt, Ut) -> torch.Tensor:
    """B = diag(s) Z diag(c) + U~^T K~ in fp64, [..., V, K]."""
    B = s.to(F64).unsqueeze(-1) * Z.to(F64) * c.to(F64).unsqueeze(-2)
    if Kt.shape[-2] > 0:
        B = B + torch.einsum("...rv,...rk->...vk", Ut.to(F64), Kt.to(F64))
    return B


def _materialize(Z, s, c, Kt, Ut, G, kh, u, l) -> torch.Tensor:
    """S_l = B * exp(G_l) (per K column) + sum_{j<=l} u_j (k^_j * exp(G_l - G_j))^T, fp64 [n,H,V,K]."""
    n, H, C, K = G.shape
    B = _base_dense(Z, s, c, Kt, Ut)
    l = l.to(torch.int64)
    has = (l > 0).view(n, 1, 1)
    Gl = G.to(F64)[
        torch.arange(n, device=G.device), :, (l - 1).clamp(min=0), :
    ]  # [n,H,K]
    Gl = torch.where(has, Gl, torch.zeros_like(Gl))
    S = B * torch.exp(Gl).unsqueeze(-2)
    mask = (torch.arange(C, device=G.device).view(1, C) < l.view(n, 1)).view(n, 1, C, 1)
    diff = torch.where(
        mask, Gl.unsqueeze(2) - G.to(F64), torch.full_like(G, -math.inf, dtype=F64)
    )
    kd = kh.to(F64) * torch.exp(diff)  # [n,H,C,K]
    S = S + torch.einsum("nhcv,nhck->nhvk", u.to(F64), kd)
    return S


def materialize(state: LQState) -> torch.Tensor:
    """Unpack: format -> fp64 state [N,H,V,K] (S_l of the committed window)."""
    return _materialize(
        state.Z,
        state.s,
        state.c,
        state.Kt,
        state.Ut,
        state.G,
        state.kh,
        state.u,
        state.l,
    )


unpack = materialize


def refit_fields(S: torch.Tensor, cfg: LQConfig, Q_start: torch.Tensor | None):
    """DESIGN.md section 5 steps 2-5 on S [n,H,V,K] (fp64). Q_start [n,H,K,r] or None (seeded Q_0).
    Returns dict(Z, s, c, Kt, Ut) in storage dtypes."""
    S = S.to(F64)
    n, H, V, K = S.shape
    r = cfg.r
    if r > 0:
        if Q_start is None:
            Q = seeded_q0(K, r, cfg.seed).to(S.device).expand(n, H, K, r).clone()
        else:
            Q = Q_start.to(F64)
        for _ in range(cfg.power_iters):
            Q = orth(
                torch.einsum(
                    "nhvk,nhvj->nhkj", S, torch.einsum("nhvk,nhkr->nhvr", S, Q)
                )
            )
        Kt = _rnd(Q.transpose(-1, -2), cfg.dt_comp)  # [n,H,r,K], rounded
        Ut = _rnd(
            torch.einsum("nhvk,nhrk->nhrv", S, Kt), cfg.dt_comp
        )  # (S K~^T)^T with the rounded K~
        R = S - torch.einsum("nhrv,nhrk->nhvk", Ut, Kt)
    else:
        Kt = S.new_zeros(n, H, 0, K)
        Ut = S.new_zeros(n, H, 0, V)
        R = S
    if cfg.smoothing:
        c = _rnd(
            torch.sqrt(R.abs().mean(dim=-2)).clamp_min(cfg.floor), cfg.dt_scale
        )  # [n,H,K]
    else:
        c = torch.ones(n, H, K, dtype=F64, device=S.device)
    Rp = R / c.unsqueeze(-2)
    amax = Rp.abs().amax(dim=-1)  # [n,H,V]
    s = _rnd(amax / 127.0, cfg.dt_scale)
    s = torch.where(s == 0, torch.ones_like(s), s)
    Zf = Rp / s.unsqueeze(-1)
    if cfg.exact:
        Z = Zf
    else:
        Z = (
            torch.round(Zf).clamp(-127, 127).to(torch.int8)
        )  # torch.round = round half to even
    return dict(
        Z=Z,
        s=s.to(cfg.dt_scale),
        c=c.to(cfg.dt_scale),
        Kt=Kt.to(cfg.dt_comp),
        Ut=Ut.to(cfg.dt_comp),
    )


def pack(S: torch.Tensor, cfg: LQConfig = LQConfig()) -> LQState:
    """fp32/fp64 state [N,H,V,K] -> LeapQuant format with an empty window (refit with l = 0, seeded Q_0)."""
    N, H, V, K = S.shape
    dev = S.device
    f = refit_fields(S, cfg, None)
    C = cfg.cap
    st = LQState(
        cfg=cfg,
        G=torch.zeros(N, H, C, K, dtype=cfg.dt_G, device=dev),
        kh=torch.zeros(N, H, C, K, dtype=cfg.dt_rec, device=dev),
        u=torch.zeros(N, H, C, V, dtype=cfg.dt_rec, device=dev),
        l=torch.zeros(N, dtype=torch.int32, device=dev),
        **f,
    )
    return st


def verify(
    state: LQState,
    q,
    k,
    v,
    g,
    beta,
    onorm_gate=None,
    onorm_weight=None,
    onorm_eps=None,
    write=True,
):
    """One verify window (DESIGN.md section 4). Inputs token-major [N,T,H,*] (q, k raw post-conv; g = log-decay;
    beta = sigmoid(b)). Writes the T draft records into slots l..l+T-1 (if write) and returns
    (o [N,T,H,V] fp64, records dict(G [N,H,T,K], kh [N,H,T,K], u [N,H,T,V]) in storage dtypes).
    o is before the output norm unless all onorm_* arguments are given (gate [N,T,H,V], weight [V], eps)."""
    cfg = state.cfg
    qh, kh = prep_qk(q, k)
    qh, kh = _tok_major_to_head_major(qh), _tok_major_to_head_major(kh)
    v = _tok_major_to_head_major(v.to(F64))
    g32 = _tok_major_to_head_major(_rnd(g.to(F64), cfg.dt_G))  # g enters as fp32
    beta = _tok_major_to_head_major(beta.to(F64))
    N, H, T, K = qh.shape
    C = state.G.shape[2]
    dev = qh.device
    l = state.l.to(torch.int64)
    assert int(l.max()) + T <= C, (
        "record capacity exceeded (commit must refit before l + T > C)"
    )
    ar = torch.arange(N, device=dev)
    slots = torch.arange(C, device=dev).view(1, C)

    Zc = state.Z.to(F64)
    s64 = state.s.to(F64)
    c64 = state.c.to(F64)
    Kt64, Ut64 = state.Kt.to(F64), state.Ut.to(F64)
    xdt = torch.float16 if cfg.base_x_fp16 else F64

    def base_apply(x):  # B x = s * (Z (c * x)) + U~^T (K~ x), x [N,H,K]
        y = s64 * torch.einsum("nhvk,nhk->nhv", Zc, _rnd(c64 * x, xdt))
        if Kt64.shape[-2] > 0:
            y = y + torch.einsum(
                "nhrv,nhr->nhv", Ut64, torch.einsum("nhrk,nhk->nhr", Kt64, x)
            )
        return y

    recG = state.G.to(F64).clone()
    recK = state.kh.to(F64).clone()
    recU = state.u.to(F64).clone()

    Gprev = recG[ar, :, (l - 1).clamp(min=0), :]
    Gprev = torch.where((l > 0).view(N, 1, 1), Gprev, torch.zeros_like(Gprev))

    if cfg.u_round == "store":
        # G for all drafts (sequential fp32 cumulative sum), rounded k^
        Gs = []
        for t in range(T):
            Gprev = _rnd(Gprev + g32[:, :, t], cfg.dt_G)
            Gs.append(Gprev)
        Gd = torch.stack(Gs, 2)  # [N,H,T,K]
        kd = _rnd(kh, cfg.dt_rec)  # [N,H,T,K]
        eG = torch.exp(Gd)
        # committed-record reads (slots < l) for all T k's and q's
        mc = (slots < l.view(N, 1)).view(N, 1, 1, C, 1)
        diff = Gd.unsqueeze(3) - recG.unsqueeze(2)  # [N,H,T,C,K]
        kdec = recK.unsqueeze(2) * torch.exp(
            torch.where(mc, diff, torch.full_like(diff, -math.inf))
        )
        W0 = torch.stack(
            [base_apply(eG[:, :, t] * kd[:, :, t]) for t in range(T)], 2
        ) + torch.einsum("nhtck,nhtk,nhcv->nhtv", kdec, kd, recU)
        O0 = torch.stack(
            [base_apply(eG[:, :, t] * qh[:, :, t]) for t in range(T)], 2
        ) + torch.einsum("nhtck,nhtk,nhcv->nhtv", kdec, qh, recU)
        # intra-draft T x T: A[m,j] = <k^_j exp(G_m - G_j), k^_m> (j < m), Qk[m,j] = <..., q^_m> (j <= m)
        dd = Gd.unsqueeze(3) - Gd.unsqueeze(2)  # [N,H,m,j,K]
        tri = torch.arange(T, device=dev)
        lower = (tri.view(T, 1) > tri.view(1, T)).view(1, 1, T, T, 1)
        lower_eq = (tri.view(T, 1) >= tri.view(1, T)).view(1, 1, T, T, 1)
        ninf = torch.full_like(dd, -math.inf)
        A = torch.einsum(
            "nhmjk,nhmk->nhmj",
            kd.unsqueeze(2) * torch.exp(torch.where(lower, dd, ninf)),
            kd,
        )
        Qk = torch.einsum(
            "nhmjk,nhmk->nhmj",
            kd.unsqueeze(2) * torch.exp(torch.where(lower_eq, dd, ninf)),
            qh,
        )
        bt = beta.unsqueeze(-1)  # [N,H,T,1]
        M = torch.eye(T, dtype=F64, device=dev) + bt * A  # unit lower triangular
        U = torch.linalg.solve_triangular(
            M, bt * (v - W0), upper=False, unitriangular=True
        )  # [N,H,T,V]
        o = (O0 + Qk @ U).transpose(1, 2)  # [N,T,H,V], uses the unrounded U
        rec = dict(G=Gd.to(cfg.dt_G), kh=kd.to(cfg.dt_rec), u=U.to(cfg.dt_rec))
        if write:
            for t in range(T):
                state.G[ar, :, l + t, :] = rec["G"][:, :, t]
                state.kh[ar, :, l + t, :] = rec["kh"][:, :, t]
                state.u[ar, :, l + t, :] = rec["u"][:, :, t]
        if onorm_gate is not None:
            o = apply_onorm(o, onorm_gate, onorm_weight, onorm_eps)
        return o, rec
    assert cfg.u_round == "step", cfg.u_round

    out_G, out_k, out_u, outs = [], [], [], []
    for t in range(T):
        Gm = _rnd(Gprev + g32[:, :, t], cfg.dt_G)  # [N,H,K]
        km = _rnd(
            kh[:, :, t], cfg.dt_rec
        )  # k^ rounded to fp16 first; only the rounded value is used
        # (base read, intra-draft dots, o, records). q^ and G are not rounded to fp16.
        qm = qh[:, :, t]
        eG = torch.exp(Gm)
        diff = Gm.unsqueeze(2) - recG  # [N,H,C,K], <= 0 on valid slots
        # committed records + earlier drafts: slots < l + t
        mw = (slots < (l + t).view(N, 1)).view(N, 1, C, 1)
        dec = torch.exp(torch.where(mw, diff, torch.full_like(diff, -math.inf)))
        ck = torch.einsum("nhck,nhk->nhc", recK * dec, km)
        w = base_apply(eG * km) + torch.einsum("nhc,nhcv->nhv", ck, recU)
        um = _rnd(beta[:, :, t, None] * (v[:, :, t] - w), cfg.dt_rec)
        # write the record at slot l + t (working copy; storage below)
        recG[ar, :, l + t, :] = Gm
        recK[ar, :, l + t, :] = km
        recU[ar, :, l + t, :] = um
        mo = (slots < (l + t + 1).view(N, 1)).view(N, 1, C, 1)
        diff = Gm.unsqueeze(2) - recG
        dec = torch.exp(torch.where(mo, diff, torch.full_like(diff, -math.inf)))
        cq = torch.einsum("nhck,nhk->nhc", recK * dec, qm)
        o = base_apply(eG * qm) + torch.einsum("nhc,nhcv->nhv", cq, recU)
        outs.append(o)
        out_G.append(Gm)
        out_k.append(km)
        out_u.append(um)
        Gprev = Gm

    o = torch.stack(outs, dim=1)  # [N,T,H,V]
    rec = dict(
        G=torch.stack(out_G, 2).to(cfg.dt_G),
        kh=torch.stack(out_k, 2).to(cfg.dt_rec),
        u=torch.stack(out_u, 2).to(cfg.dt_rec),
    )
    if write:
        for t in range(T):
            state.G[ar, :, l + t, :] = rec["G"][:, :, t]
            state.kh[ar, :, l + t, :] = rec["kh"][:, :, t]
            state.u[ar, :, l + t, :] = rec["u"][:, :, t]
    if onorm_gate is not None:
        o = apply_onorm(o, onorm_gate, onorm_weight, onorm_eps)
    return o, rec


def commit(
    state: LQState, n_accepted: torch.Tensor, P: int | None = None
) -> torch.Tensor:
    """l += n (n in 1..T per request); refit every request with l >= P (all heads of that request).
    Returns the bool mask of refitted requests."""
    cfg = state.cfg
    P = cfg.P if P is None else P
    state.l = (
        state.l.to(torch.int64) + n_accepted.to(state.l.device).to(torch.int64)
    ).to(torch.int32)
    need = state.l >= P
    idx = need.nonzero().flatten()
    if idx.numel() > 0:
        sub = lambda f: getattr(state, f)[idx]
        S = _materialize(
            *(sub(f) for f in ("Z", "s", "c", "Kt", "Ut", "G", "kh", "u", "l"))
        )
        Q_start = sub("Kt").to(F64).transpose(-1, -2) if cfg.r > 0 else None
        f = refit_fields(S, cfg, Q_start)
        for name, val in f.items():
            getattr(state, name)[idx] = val
        state.l[idx] = 0
        state.stats["refits"] += int(idx.numel())
    return need


# ============================================================================================= GDN kernel oracle
#
# Qwen3.5 / Qwen3-Next gated delta net: scalar log-decay per (token, value head), k/q heads shared by HV / HK value
# heads (repeat-interleave: value head h reads key head h // (HV // HK)). Everything below is fp64 between the rounding
# points listed in API.md "GDN kernel oracle". Layouts (token-major inputs, as production):
#   q, k [N, T, HK, K] post-conv (+SiLU) activations; v [N, T, HV, V]; a, b [N, T, HV] raw (bf16 in production);
#   A_log, dt_bias [HV] fp32; optional norm gate z [N, T, HV, V], norm weight [V], eps.
#   State S [N, HV, V, K].  GDNState fields:
#     Z int8 [N,HV,V,K]  s fp32 [N,HV,V]  c fp32 [N,HV,K]  Kt fp16 [N,HV,r,K]  Ut fp16 [N,HV,r,V]
#     G fp32 [N,HV,C]  kh fp16 [N,HK,C,K] (per KEY head)  u fp16 [N,HV,C,V]  l int32 [N]  lim int32 [N]
#   lim = the window limit of the current window (refit when l >= lim): P, or the shorter first window of the
#   per-layer phase offset (layer i: (i mod P) tokens, P when i mod P == 0).


@dataclass(frozen=True)
class GDNConfig:
    # defaults on HOLD at P = 16, C = 20 (main 2026-11-14); format v4 candidate = GDNConfig(P=64, C=68) (refit warm2)
    P: int = 16
    C: int = 20
    r: int = 4
    refit: str = "warm2"  # "warm2": 2 iterations warm-started from the previous K~^T; "cold1": 1 iteration from Q0
    floor: float = 1e-6
    phase_offset: bool = False
    # phase_carry: at a refit the next window's limit is P - overshoot (l - lim), so every layer keeps its phase on a
    # fixed P-token grid; without it verify overshoot (n up to 4) re-synchronises the layers within a few windows.
    phase_carry: bool = False
    base_x: str = "exact"  # verify/decode base operand: "exact" (fp64), "fp16" (single), "hilo" (fp16 hi + fp16 lo)
    exact: bool = False  # lossless storage (self-test only)
    seed: int = 0

    @property
    def iters(self) -> int:
        return {"warm2": 2, "cold1": 1, "sticky": 2}[
            self.refit
        ]  # sticky: iterations used by pack only

    @property
    def warm(self) -> bool:
        return self.refit in ("warm2", "sticky")

    def dt(self, name):
        if self.exact:
            return F64
        return {
            "Z": torch.int8,
            "s": torch.float32,
            "c": torch.float32,
            "Kt": torch.float16,
            "Ut": torch.float16,
            "G": torch.float32,
            "kh": torch.float16,
            "u": torch.float16,
        }[name]


@dataclass
class GDNState:
    cfg: GDNConfig
    Z: torch.Tensor
    s: torch.Tensor
    c: torch.Tensor
    Kt: torch.Tensor
    Ut: torch.Tensor
    G: torch.Tensor
    kh: torch.Tensor
    u: torch.Tensor
    l: torch.Tensor
    lim: torch.Tensor
    refits: int = 0

    FIELDS = ("Z", "s", "c", "Kt", "Ut", "G", "kh", "u", "l", "lim")

    def clone(self) -> GDNState:
        return GDNState(
            self.cfg,
            *(getattr(self, f).clone() for f in self.FIELDS),
            refits=self.refits,
        )


def gdn_q0(K: int, r: int, seed: int = 0) -> torch.Tensor:
    """The fixed start block Q_0 [K, r] as fp32 values (the kernels store exactly this fp32 tensor)."""
    return seeded_q0(K, r, seed).to(torch.float32)


def gdn_gates(a, b, A_log, dt_bias, beta_dtype=torch.float32):
    """g = fp32(-exp(A_log) * softplus(a + dt_bias)) (softplus threshold 20), beta = sigmoid(b) rounded to beta_dtype
    (decode: bf16 as production packed decode; verify: fp32). Returned as fp64 values. [N, T, HV]."""
    x = a.to(F64) + dt_bias.to(F64)
    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    g = (-torch.exp(A_log.to(F64)) * sp).to(torch.float32).to(F64)
    beta = torch.sigmoid(b.to(F64)).to(beta_dtype).to(F64)
    return g, beta


def gdn_expand(x: torch.Tensor, HV: int, dim: int = -2) -> torch.Tensor:
    HK = x.shape[dim]
    return x if HK == HV else x.repeat_interleave(HV // HK, dim=dim)


def gdn_exact(S, q, k, v, a, b, A_log, dt_bias, n_acc=None, beta_dtype=torch.float32):
    """Exact fp64 GDN recurrence over one window (same gates/beta rounding as the oracle path being compared)."""
    HV = v.shape[2]
    g, beta = gdn_gates(a, b, A_log, dt_bias, beta_dtype)
    qe, ke = gdn_expand(q.to(F64), HV), gdn_expand(k.to(F64), HV)
    K = q.shape[-1]
    return exact_verify(
        S, qe, ke, v.to(F64), g.unsqueeze(-1).expand(*g.shape, K), beta, n_acc=n_acc
    )


def _gdn_lim0(cfg: GDNConfig, layer_idx: int) -> int:
    if cfg.phase_offset and layer_idx % cfg.P != 0:
        return layer_idx % cfg.P
    return cfg.P


def _gdn_refit(
    S: torch.Tensor,
    cfg: GDNConfig,
    Kt_prev: torch.Tensor | None,
    prev: dict | None = None,
):
    """Refit core on S [n, HV, V, K] fp64 -> dict(Z, s, c, Kt, Ut) in storage dtypes.
    refit="sticky" (EXPERIMENT, drift study only): with a previous state, keep K~ (no power iteration; U~ re-projected),
    keep c, and keep each row's s while the new absmax scale stays within [0.5, 1] of it, so that requantising an
    almost unchanged slow-decay state reproduces the same int8 entries instead of re-drawing the rounding noise."""
    n, HV, V, K = S.shape
    if cfg.refit == "sticky" and prev is not None and Kt_prev is not None:
        Kt = Kt_prev.to(F64) if cfg.r > 0 else S.new_zeros(n, HV, 0, K)
        Ut = _rnd((S @ Kt.transpose(-1, -2)).transpose(-1, -2), cfg.dt("Ut"))
        R = S - Ut.transpose(-1, -2) @ Kt
        c = prev["c"].to(F64)
        Rp = R / c.unsqueeze(-2)
        s_new = _rnd(Rp.abs().amax(dim=-1) / 127.0, cfg.dt("s"))
        s_old = prev["s"].to(F64)
        s = torch.where((s_new <= s_old) & (s_new >= 0.5 * s_old), s_old, s_new)
        s = torch.where(s == 0, torch.ones_like(s), s)
        Z = torch.round(Rp / s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return dict(
            Z=Z,
            s=s.to(cfg.dt("s")),
            c=c.to(cfg.dt("c")),
            Kt=Kt.to(cfg.dt("Kt")),
            Ut=Ut.to(cfg.dt("Ut")),
        )
    if cfg.r > 0:
        if cfg.warm and Kt_prev is not None:
            Q = Kt_prev.to(F64).transpose(-1, -2)
        else:
            Q = gdn_q0(K, cfg.r, cfg.seed).to(S.device, F64).expand(n, HV, K, cfg.r)
        for _ in range(cfg.iters):
            Q = orth(S.transpose(-1, -2) @ (S @ Q))
        Kt = _rnd(Q.transpose(-1, -2), cfg.dt("Kt"))
        Ut = _rnd((S @ Kt.transpose(-1, -2)).transpose(-1, -2), cfg.dt("Ut"))
        R = S - Ut.transpose(-1, -2) @ Kt
    else:
        Kt, Ut, R = S.new_zeros(n, HV, 0, K), S.new_zeros(n, HV, 0, V), S
    c = _rnd(torch.sqrt(R.abs().mean(dim=-2)).clamp_min(cfg.floor), cfg.dt("c"))
    Rp = R / c.unsqueeze(-2)
    s = _rnd(Rp.abs().amax(dim=-1) / 127.0, cfg.dt("s"))
    s = torch.where(s == 0, torch.ones_like(s), s)
    Zf = Rp / s.unsqueeze(-1)
    Z = Zf if cfg.exact else torch.round(Zf).clamp(-127, 127).to(torch.int8)
    return dict(
        Z=Z,
        s=s.to(cfg.dt("s")),
        c=c.to(cfg.dt("c")),
        Kt=Kt.to(cfg.dt("Kt")),
        Ut=Ut.to(cfg.dt("Ut")),
    )


def gdn_pack(
    S: torch.Tensor,
    cfg: GDNConfig = GDNConfig(),
    HK: int | None = None,
    layer_idx: int = 0,
) -> GDNState:
    """fp32/fp64 state [N, HV, V, K] -> format with an empty window (refit from Q0, cfg.iters iterations)."""
    N, HV, V, K = S.shape
    HK = HV if HK is None else HK
    dev = S.device
    f = _gdn_refit(S.to(F64), cfg, None)
    return GDNState(
        cfg=cfg,
        G=torch.zeros(N, HV, cfg.C, dtype=cfg.dt("G"), device=dev),
        kh=torch.zeros(N, HK, cfg.C, K, dtype=cfg.dt("kh"), device=dev),
        u=torch.zeros(N, HV, cfg.C, V, dtype=cfg.dt("u"), device=dev),
        l=torch.zeros(N, dtype=torch.int32, device=dev),
        lim=torch.full((N,), _gdn_lim0(cfg, layer_idx), dtype=torch.int32, device=dev),
        **f,
    )


def _gdn_base(st: GDNState) -> torch.Tensor:
    return _base_dense(st.Z, st.s, st.c, st.Kt, st.Ut)


def gdn_materialize(st: GDNState, l: torch.Tensor | None = None) -> torch.Tensor:
    """S_l = e^{G_l} B + sum_{j<l} u_j (k^_j e^{G_l - G_j})^T  ->  fp64 [N, HV, V, K] (unpack)."""
    l = (st.l if l is None else l).to(torch.int64)
    N, HV, C = st.G.shape
    G = st.G.to(F64)
    Gl = torch.gather(
        G, 2, (l - 1).clamp(min=0).view(N, 1, 1).expand(N, HV, 1)
    ).squeeze(2)
    Gl = torch.where((l > 0).view(N, 1), Gl, torch.zeros_like(Gl))
    mask = torch.arange(C, device=G.device).view(1, 1, C) < l.view(N, 1, 1)
    w = torch.exp(
        torch.where(mask, Gl.unsqueeze(-1) - G, torch.full_like(G, -math.inf))
    )  # [N,HV,C]
    kh = gdn_expand(st.kh.to(F64), HV, dim=1)  # [N,HV,C,K]
    S = _gdn_base(st) * torch.exp(Gl)[..., None, None]
    return S + (st.u.to(F64) * w.unsqueeze(-1)).transpose(-1, -2) @ kh


def _gdn_base_apply(st: GDNState, X: torch.Tensor) -> torch.Tensor:
    """B X for X [N, HV, M, K] (rows = vectors) -> [N, HV, M, V], with the configured base operand rounding:
    s * (Z (c * x)) + U~^T (K~ x); the rounding (fp16 / hi+lo) applies to the operand c*x of the Z product only."""
    cx = st.c.to(F64).unsqueeze(-2) * X
    if st.cfg.base_x == "fp16":
        cx = _rnd(cx, torch.float16)
    elif st.cfg.base_x == "hilo":
        hi = _rnd(cx, torch.float16)
        cx = hi + _rnd(cx - hi, torch.float16)
    y = (cx @ st.Z.to(F64).transpose(-1, -2)) * st.s.to(F64).unsqueeze(-2)
    if st.Kt.shape[-2] > 0:
        y = y + (X @ st.Kt.to(F64).transpose(-1, -2)) @ st.Ut.to(F64)
    return y


def gdn_window(
    st: GDNState,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    beta_dtype=torch.float32,
    write=True,
    gb=None,
):
    """One window of T tokens (verify T = 4, decode T = 1), parallel (WY) form, U unrounded inside the window.
    Writes records at l .. l+T-1 (G, u per value head; k^ per key head) when write; never touches base / l.
    Returns (o [N,T,HV,V] fp64 pre-norm, records dict(G [N,HV,T], kh [N,HK,T,K], u [N,HV,T,V]))."""
    cfg = st.cfg
    N, T, HK, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    dev = k.device
    l = st.l.to(torch.int64)
    assert int(l.max()) + T <= cfg.C, "record capacity exceeded"
    if gb is None:
        g, beta = gdn_gates(a, b, A_log, dt_bias, beta_dtype)  # [N,T,HV]
    else:  # precomputed gates (e.g. real dumps): g fp32 values, beta already rounded per path
        g, beta = gb[0].to(torch.float32).to(F64), gb[1].to(F64)
    qh = l2norm(q.to(F64)) * K**-0.5  # [N,T,HK,K]
    kh_key = _rnd(
        l2norm(k.to(F64)), cfg.dt("kh")
    )  # rounded k^ per key head, used everywhere
    qd = gdn_expand(qh, HV).transpose(1, 2)  # [N,HV,T,K]
    kd = gdn_expand(kh_key, HV).transpose(1, 2)
    vv = v.to(F64).transpose(1, 2)
    gt, bt = g.transpose(1, 2), beta.transpose(1, 2)  # [N,HV,T]
    C = cfg.C
    G = st.G.to(F64)
    Gprev = torch.gather(
        G, 2, (l - 1).clamp(min=0).view(N, 1, 1).expand(N, HV, 1)
    ).squeeze(2)
    Gprev = torch.where((l > 0).view(N, 1), Gprev, torch.zeros_like(Gprev))
    Gs = []
    for t in range(T):  # fp32 cumulative sum in token order
        Gprev = _rnd(Gprev + gt[..., t], cfg.dt("G"))
        Gs.append(Gprev)
    Gd = torch.stack(Gs, -1)  # [N,HV,T]
    eG = torch.exp(Gd).unsqueeze(-1)
    X = torch.cat([kd, qd], dim=2)  # [N,HV,2T,K]
    base = _gdn_base_apply(st, X * torch.cat([eG, eG], dim=2))
    maskc = torch.arange(C, device=dev).view(1, 1, 1, C) < l.view(N, 1, 1, 1)
    dc = Gd.unsqueeze(-1) - G.unsqueeze(-2)  # [N,HV,T,C]
    dec = torch.exp(torch.where(maskc, dc, torch.full_like(dc, -math.inf)))
    khr = gdn_expand(st.kh.to(F64), HV, dim=1)  # [N,HV,C,K]
    WO = base + ((X @ khr.transpose(-1, -2)) * torch.cat([dec, dec], dim=2)) @ st.u.to(
        F64
    )
    W0, O0 = WO[:, :, :T], WO[:, :, T:]
    ar = torch.arange(T, device=dev)
    dd = Gd.unsqueeze(-1) - Gd.unsqueeze(-2)
    ninf = torch.full_like(dd, -math.inf)
    A = (kd @ kd.transpose(-1, -2)) * torch.exp(
        torch.where(ar.view(T, 1) > ar.view(1, T), dd, ninf)
    )
    Qk = (qd @ kd.transpose(-1, -2)) * torch.exp(
        torch.where(ar.view(T, 1) >= ar.view(1, T), dd, ninf)
    )
    M = torch.eye(T, dtype=F64, device=dev) + bt.unsqueeze(-1) * A
    U = torch.linalg.solve_triangular(
        M, bt.unsqueeze(-1) * (vv - W0), upper=False, unitriangular=True
    )
    o = (O0 + Qk @ U).transpose(1, 2)  # [N,T,HV,V]
    rec = dict(
        G=Gd.to(cfg.dt("G")),
        kh=kh_key.transpose(1, 2).to(cfg.dt("kh")),
        u=U.to(cfg.dt("u")),
    )
    if write:
        ar_n = torch.arange(N, device=dev)
        for t in range(T):
            st.G[ar_n, :, l + t] = rec["G"][:, :, t]
            st.kh[ar_n, :, l + t, :] = rec["kh"][:, :, t]
            st.u[ar_n, :, l + t, :] = rec["u"][:, :, t]
    return o, rec


def gdn_commit(st: GDNState, n: torch.Tensor) -> torch.Tensor:
    """l += n; every slot with l >= lim is refit (all its heads), then lim <- P (or P - (l - lim) with phase_carry)
    and l <- 0. Returns the refit mask."""
    cfg = st.cfg
    st.l = (st.l.to(torch.int64) + n.to(st.l.device).to(torch.int64)).to(torch.int32)
    need = st.l >= st.lim
    idx = need.nonzero().flatten()
    if idx.numel() > 0:
        sub = st.clone()
        for f in GDNState.FIELDS:
            setattr(sub, f, getattr(st, f)[idx])
        S = gdn_materialize(sub)
        new = _gdn_refit(S, cfg, sub.Kt, prev=dict(c=sub.c, s=sub.s))
        for name, val in new.items():
            getattr(st, name)[idx] = val
        if cfg.phase_carry:
            st.lim[idx] = (cfg.P - (st.l[idx] - st.lim[idx])).to(torch.int32)
        else:
            st.lim[idx] = cfg.P
        st.l[idx] = 0
        st.refits += int(idx.numel())
    return need


def gdn_decode(
    st: GDNState, q, k, v, a, b, A_log, dt_bias, z=None, norm_w=None, eps=None, gb=None
):
    """Plain decode (T = 1): window with bf16-rounded beta (production packed decode) + commit n = 1.
    q, k [N, 1, HK, K], v [N, 1, HV, V], a, b [N, 1, HV]. Returns (o, refit mask); o normed if z/norm_w/eps given."""
    o, _ = gdn_window(
        st, q, k, v, a, b, A_log, dt_bias, beta_dtype=torch.bfloat16, gb=gb
    )
    need = gdn_commit(
        st, torch.ones(st.l.shape[0], dtype=torch.int64, device=st.l.device)
    )
    if z is not None:
        o = gdn_onorm(o, z, norm_w, eps)
    return o, need


def gdn_verify(
    st: GDNState, q, k, v, a, b, A_log, dt_bias, z=None, norm_w=None, eps=None, gb=None
):
    """Target verify (T = 4, chain): fp32 beta, writes the 4 draft records, read-only on base / l."""
    o, rec = gdn_window(
        st, q, k, v, a, b, A_log, dt_bias, beta_dtype=torch.float32, gb=gb
    )
    if z is not None:
        o = gdn_onorm(o, z, norm_w, eps)
    return o, rec


def gdn_onorm(o, z, w, eps):
    """Qwen3.5 output norm (RMSNormGated, norm_before_gate, swish): o * rsqrt(mean(o^2) + eps) * w * silu(z)."""
    o = o.to(F64)
    zz = z.to(F64)
    return (
        o
        * torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
        * w.to(F64)
        * (zz * torch.sigmoid(zz))
    )


def gdn_random_layer(HV: int, gen: torch.Generator, device="cpu"):
    """Qwen3-Next-style gate parameters: A ~ U(1, 16), dt log-uniform [1e-3, 1e-1], dt_bias = softplus^-1(dt)."""
    A_log = torch.log(torch.empty(HV).uniform_(1, 16, generator=gen)).to(torch.float32)
    dt = torch.exp(
        torch.empty(HV).uniform_(math.log(1e-3), math.log(1e-1), generator=gen)
    )
    dt_bias = (dt + torch.log(-torch.expm1(-dt))).to(torch.float32)
    return A_log.to(device), dt_bias.to(device)


def gdn_random_inputs(N, T, HK, HV, gen, device="cpu", K=128, V=128):
    """RANDOM DATA (not real activations): bf16 q, k, v ~ N(0,1) (post-conv stand-in), a, b ~ N(0,1) bf16."""
    rn = lambda *s: torch.randn(*s, generator=gen).to(torch.bfloat16).to(device)
    return rn(N, T, HK, K), rn(N, T, HK, K), rn(N, T, HV, V), rn(N, T, HV), rn(N, T, HV)


def gdn_make_test_state(
    N,
    HK,
    HV,
    cfg: GDNConfig = GDNConfig(),
    layer_idx=0,
    prefill=64,
    fill=0,
    seed=0,
    device="cpu",
    K=128,
    V=128,
):
    """Test state: exact fp64 recurrence over `prefill` random tokens -> fp32 -> pack, then `fill` decode steps through
    the oracle (so l > 0 and the records are populated). Returns (state, A_log, dt_bias, S_exact_fp64, generator)."""
    gen = torch.Generator().manual_seed(seed)
    A_log, dt_bias = gdn_random_layer(HV, gen, device)
    S = torch.zeros(N, HV, V, K, dtype=F64, device=device)
    for _ in range(prefill // 8):
        q, k, v, a, b = gdn_random_inputs(N, 8, HK, HV, gen, device, K, V)
        S = gdn_exact(S, q, k, v, a, b, A_log, dt_bias)[1]
    S = S.to(torch.float32).to(F64)  # the pool holds fp32
    st = gdn_pack(S, cfg, HK=HK, layer_idx=layer_idx)
    for _ in range(fill):
        q, k, v, a, b = gdn_random_inputs(N, 1, HK, HV, gen, device, K, V)
        S = gdn_exact(S, q, k, v, a, b, A_log, dt_bias, beta_dtype=torch.bfloat16)[1]
        gdn_decode(st, q, k, v, a, b, A_log, dt_bias)
    return st, A_log, dt_bias, S, gen


# --------------------------------------------------------------------------------------------- GDN packed slot layout
# Option A: the format aliases the fp32 temporal slot [HV, V, K] (64 KiB per value head). Byte offsets per value head
# (agent R's gdn_lq_format.cuh, C = 20). k^ of key head i lives in value head 2i's region (the odd region's bytes are
# unused); l, lim are replicated in EVERY value head's region and all copies hold equal values (2026-11-14 update).
def gdn_slot_layout(C: int = 20, r: int = 4, K: int = 128, V: int = 128):
    """(name, dtype, shape per value head, byte offset) in agent R's field order; every field 16-B aligned."""
    spec = [
        ("Z", torch.int8, (V, K)),
        ("s", torch.float32, (V,)),
        ("c", torch.float32, (K,)),
        ("Kt", torch.float16, (r, K)),
        ("Ut", torch.float16, (r, V)),
        ("u", torch.float16, (C, V)),
        ("kh", torch.float16, (C, K)),
        ("G", torch.float32, (C,)),
        ("llim", torch.int32, (2,)),
    ]
    out, off = [], 0
    for name, dt, shape in spec:
        out.append((name, dt, shape, off))
        nb = math.prod(shape) * torch.tensor([], dtype=dt).element_size()
        off += (nb + 15) // 16 * 16
    assert off <= K * V * 4, (
        f"format (C={C}, r={r}) does not fit the fp32 slot: {off} B"
    )
    return tuple(out)


GDN_SLOT_LAYOUT = (
    gdn_slot_layout()
)  # current default C = 20; v4 candidate: gdn_slot_layout(68)
GDN_SLOT_BYTES = 128 * 128 * 4


def _slot_views(buf_u8: torch.Tensor, C: int = 20, r: int = 4):
    """buf_u8 [N, HV, 65536] uint8 -> dict of typed views [N, HV, *shape]."""
    N, HV, _ = buf_u8.shape
    out = {}
    for name, dt, shape, off in gdn_slot_layout(C, r):
        nb = math.prod(shape) * torch.tensor([], dtype=dt).element_size()
        out[name] = buf_u8[..., off : off + nb].view(dt).view(N, HV, *shape)
    return out


def gdn_to_slots(st: GDNState) -> torch.Tensor:
    """GDNState -> the packed slot buffer as the fp32 pool sees it: [N, HV, V, K] fp32 (bytes per GDN_SLOT_LAYOUT;
    unused bytes zero). Requires K = V = 128, HV = 2 HK and a non-exact config; offsets from gdn_slot_layout(C, r)."""
    cfg = st.cfg
    assert not cfg.exact
    N, HV, V, K = st.Z.shape
    HK = st.kh.shape[1]
    assert HV == 2 * HK, (
        "layout assumes the 2:1 GQA of Qwen3.5 (k^ in the even value head)"
    )
    buf = torch.zeros(N, HV, GDN_SLOT_BYTES, dtype=torch.uint8, device=st.Z.device)
    v = _slot_views(buf, cfg.C, cfg.r)
    for name in ("Z", "s", "c", "Kt", "Ut", "u", "G"):
        v[name].copy_(getattr(st, name))
    v["kh"][:, 0::2].copy_(st.kh)
    llim = torch.stack([st.l, st.lim], -1).to(torch.int32)  # [N, 2]
    v["llim"].copy_(llim.unsqueeze(1).expand(N, HV, 2))
    return buf.view(torch.float32).view(N, HV, V, K)


def gdn_from_slots(
    slots: torch.Tensor, cfg: GDNConfig = GDNConfig(), check_replicas: bool = True
) -> GDNState:
    """Inverse of gdn_to_slots: [N, HV, V, K] fp32 pool rows -> GDNState (asserts the l/lim replicas agree)."""
    N, HV, V, K = slots.shape
    buf = slots.contiguous().view(torch.uint8).view(N, HV, GDN_SLOT_BYTES)
    v = _slot_views(buf, cfg.C, cfg.r)
    llim = v["llim"].to(torch.int64)  # [N, HV, 2]
    if check_replicas:
        assert torch.equal(llim, llim[:, :1].expand_as(llim)), (
            "l / lim replicas differ across value heads"
        )
    return GDNState(
        cfg=cfg,
        Z=v["Z"].clone(),
        s=v["s"].clone(),
        c=v["c"].clone(),
        Kt=v["Kt"].clone(),
        Ut=v["Ut"].clone(),
        G=v["G"].clone(),
        kh=v["kh"][:, 0::2].clone(),
        u=v["u"].clone(),
        l=llim[:, 0, 0].to(torch.int32),
        lim=llim[:, 0, 1].to(torch.int32),
    )


def gdn_refit_diagnostics(
    S_l: torch.Tensor,
    st_k: GDNState,
    st_r: GDNState,
    Kt_prev: torch.Tensor | None = None,
    quant_slack: float = 1.05,
    quant_abs: float = 1e-4,
    base_rel: float = 0.2,
    base_abs: float = 2e-3,
    piv_min: float = 1e-4,
    rs_min: float = 1e-2,
):
    """Per-head refit acceptance (API.md "GDN kernel oracle", refit part; one implementation for K and R).
    S_l [N, HV, V, K]: the state the refit was applied to (fp64, e.g. gdn_materialize(st, l_new) before the refit).
    st_k / st_r: kernel / oracle state after the refit (their l is ignored: the base is read at l = 0).
    Kt_prev: the pre-refit K~ (warm2 start block); None -> Q0 (cold1, or pack).
    Returns a dict of per-head [N, HV] tensors:
      quant_k, quant_r = |B - S_l| / |S_l| (kernel / oracle), base_diff = |B_k - B_r| / |B_r|,
      piv = d_j / G_jj (j = 0..3) of the fp64 Cholesky-QR of W = S^T S Q (Q = the last power iteration's input),
      rs = |R| / |S_l| with R = S_l - U~_r^T K~_r, dz_max, dz1_frac (|dZ| = 1 fraction), z_eligible
      (piv[3] >= piv_min and rs >= rs_min), ok_quant, ok_base, ok (both), plus 'summary' (python dict)."""
    cfg = st_r.cfg
    S = S_l.to(F64)
    N, HV, V, K = S.shape
    zero = lambda st: torch.zeros_like(st.l)
    B_k = gdn_materialize(st_k, zero(st_k))
    B_r = gdn_materialize(st_r, zero(st_r))
    frob = lambda x: x.flatten(-2).norm(dim=-1)
    nS = frob(S).clamp_min(1e-300)
    quant_k = frob(B_k - S) / nS
    quant_r = frob(B_r - S) / nS
    base_diff = frob(B_k - B_r) / frob(B_r).clamp_min(1e-300)
    # start block of the last power iteration
    if cfg.warm and Kt_prev is not None:
        Q = Kt_prev.to(F64).transpose(-1, -2)
    else:
        Q = gdn_q0(K, cfg.r, cfg.seed).to(S.device, F64).expand(N, HV, K, cfg.r)
    for _ in range(cfg.iters - 1):
        Q = orth(S.transpose(-1, -2) @ (S @ Q))
    W = S.transpose(-1, -2) @ (S @ Q)  # [N,HV,K,r]
    Gjj = (W * W).sum(-2).clamp_min(1e-300)
    # Cholesky-QR pivots d_j = L_jj^2 of W^T W equal R_jj^2 of a QR of W (QR used: robust for rank-deficient W)
    Rq = torch.linalg.qr(W).R
    piv = Rq.diagonal(dim1=-2, dim2=-1) ** 2 / Gjj  # [N,HV,r]
    Rres = S - st_r.Ut.to(F64).transpose(-1, -2) @ st_r.Kt.to(F64)
    rs = frob(Rres) / nS
    dZ = (st_k.Z.to(torch.int32) - st_r.Z.to(torch.int32)).abs()
    dz_max = dZ.flatten(-2).amax(-1)
    dz1_frac = (dZ == 1).to(F64).flatten(-2).mean(-1)
    z_eligible = (piv[..., -1] >= piv_min) & (rs >= rs_min)
    ok_quant = quant_k <= quant_slack * quant_r + quant_abs
    ok_base = (base_diff <= base_rel * quant_r) | (base_diff <= base_abs)
    ok = ok_quant & ok_base
    el = z_eligible
    summary = dict(
        heads=N * HV,
        ok=int(ok.sum()),
        ok_quant=int(ok_quant.sum()),
        ok_base=int(ok_base.sum()),
        quant_ratio_max=float((quant_k / quant_r.clamp_min(1e-300)).max()),
        base_diff_max=float(base_diff.max()),
        z_eligible=int(el.sum()),
        z_eligible_dz1_frac_mean=float(dz1_frac[el].mean())
        if el.any()
        else float("nan"),
        z_eligible_dz_max=int(dz_max[el].max()) if el.any() else 0,
    )
    return dict(
        quant_k=quant_k,
        quant_r=quant_r,
        base_diff=base_diff,
        piv=piv,
        rs=rs,
        dz_max=dz_max,
        dz1_frac=dz1_frac,
        z_eligible=z_eligible,
        ok_quant=ok_quant,
        ok_base=ok_base,
        ok=ok,
        summary=summary,
    )


def gdn_self_test(device="cpu", N=3, HK=2, HV=4, steps=40, verbose=True):
    """Lossless format (exact=True) must equal the exact GDN recurrence to ~1e-13 for decode chains and verify+commit
    chains, every refit mode / phase option; quantised: verify output vs later re-read at the fp16-u level."""
    res = {}
    for refit in ("warm2", "cold1"):
        for phase in (False, True):
            for T in (1, 4):
                cfg = GDNConfig(
                    refit=refit, phase_offset=phase, phase_carry=phase, exact=True
                )
                st, A_log, dt_bias, S, gen = gdn_make_test_state(
                    N, HK, HV, cfg, layer_idx=5, seed=7, device=device
                )
                worst = 0.0
                for _ in range(steps):
                    q, k, v, a, b = gdn_random_inputs(N, T, HK, HV, gen, device)
                    if T == 1:
                        o_ref, S = gdn_exact(
                            S, q, k, v, a, b, A_log, dt_bias, beta_dtype=torch.bfloat16
                        )
                        o, _ = gdn_decode(st, q, k, v, a, b, A_log, dt_bias)
                    else:
                        n = torch.randint(1, T + 1, (N,), generator=gen).to(device)
                        o_ref, S = gdn_exact(S, q, k, v, a, b, A_log, dt_bias, n_acc=n)
                        o, _ = gdn_verify(st, q, k, v, a, b, A_log, dt_bias)
                        gdn_commit(st, n)
                    e_o = float(_relerr(o, o_ref, (-1,)).max())
                    e_s = float(_relerr(gdn_materialize(st), S, (-2, -1)).max())
                    worst = max(worst, e_o, e_s)
                key = f"exact refit={refit} phase={phase} T={T}"
                res[key] = (worst, st.refits)
                assert worst < 1e-11, (key, worst)
    # quantised: re-read of committed records equals the verify output up to the fp16 rounding of u
    st, A_log, dt_bias, S, gen = gdn_make_test_state(
        N, HK, HV, GDNConfig(), seed=8, device=device
    )
    rr = []
    for _ in range(steps):
        q, k, v, a, b = gdn_random_inputs(N, 4, HK, HV, gen, device)
        n = torch.randint(1, 5, (N,), generator=gen).to(device)
        o, _ = gdn_verify(st, q, k, v, a, b, A_log, dt_bias)
        need = gdn_commit(st, n)
        qh = gdn_expand(l2norm(q.to(F64)) * 128**-0.5, HV)
        ar = torch.arange(N, device=device)
        o_re = torch.einsum("nhvk,nhk->nhv", gdn_materialize(st), qh[ar, n - 1])
        if (~need).any():
            rr.append(_relerr(o_re[~need], o[ar, n - 1][~need], (-1,)).flatten())
    rr = torch.cat(rr)
    res["quant verify vs re-read (mean, max)"] = (float(rr.mean()), float(rr.max()))
    # packed slot layout round trip (HV = 2 HK)
    st2, *_ = gdn_make_test_state(2, 2, 4, GDNConfig(), fill=5, seed=9, device=device)
    rt = gdn_from_slots(gdn_to_slots(st2), st2.cfg)
    for f_ in GDNState.FIELDS:
        assert torch.equal(getattr(rt, f_), getattr(st2, f_)), f_
    assert torch.equal(gdn_materialize(rt), gdn_materialize(st2))
    res["slot layout round trip"] = ("ok",)
    # refit diagnostics: oracle vs itself passes; a perturbed "kernel" Z is caught
    st3, A3, d3_, _, g3 = gdn_make_test_state(
        2, 2, 4, GDNConfig(refit="cold1"), fill=12, seed=10, device=device
    )
    gdn_verify(
        st3, *gdn_random_inputs(2, 4, 2, 4, g3, device), A3, d3_
    )  # records l .. l+3
    n3 = torch.full((2,), 4, dtype=torch.int64, device=device)
    S_l = gdn_materialize(st3, st3.l.to(torch.int64) + n3)
    r3 = st3.clone()
    gdn_commit(r3, n3)
    d = gdn_refit_diagnostics(S_l, r3, r3)
    assert bool(d["ok"].all()), d["summary"]
    k3 = r3.clone()
    k3.Z = (
        (k3.Z.to(torch.int32) + torch.randint(-3, 4, k3.Z.shape, device=device))
        .clamp(-127, 127)
        .to(torch.int8)
    )
    d2 = gdn_refit_diagnostics(S_l, k3, r3)
    zrel = (
        d2["rs"] >= 1e-2
    )  # heads whose int8 residual matters; near-exact low-rank heads sit under the 1e-4 floor
    assert not bool(d2["ok"][zrel].any()), d2["summary"]
    res[
        "refit diagnostics (self ok / perturbed passing / heads / perturbed passing with rs>=1e-2)"
    ] = (
        d["summary"]["ok"],
        d2["summary"]["ok"],
        d2["summary"]["heads"],
        int(d2["ok"][zrel].sum()),
    )
    assert float(rr.max()) < 5e-3
    if verbose:
        for kk, vv in res.items():
            print(f"[gdn] {kk}: {vv}")
        print("GDN SELF-TEST PASSED")
    return res


# --------------------------------------------------------------------------------------------- self-test


def _rand_inputs(N, T, H, K, gen, device, g_lo=-5.0):
    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=F64).to(device)

    q, k, v = rn(N, T, H, K), rn(N, T, H, K), rn(N, T, H, K)
    x = rn(N, T, H, K) * 3.0 + rn(1, 1, H, K) * 4.0
    g = g_lo * torch.sigmoid(x)
    beta = torch.sigmoid(rn(N, T, H))
    return [t.to(torch.float32) for t in (q, k, v, g, beta)]


def _relerr(a, b, dims):
    return (a - b).norm(dim=dims) / b.norm(dim=dims).clamp_min(1e-300)


def self_test(device="cpu", N=3, H=4, K=128, steps=60, verbose=True):
    """Lossless format (exact=True, fp64 records, unrounded Z) must reproduce the exact recurrence to ~1e-12 over a
    chain of verify+commit windows with random acceptance (checks window algebra, decay bookkeeping, the
    triangular dependency, refit/pack bookkeeping). Then: quantised format sanity and verify == later re-read."""
    gen = torch.Generator(device="cpu").manual_seed(1234)
    T = 8
    # nontrivial starting state: exact recurrence over 64 tokens from zero
    S0 = torch.zeros(N, H, K, K, dtype=F64, device=device)
    for _ in range(8):
        S0 = exact_tokens(S0, *_rand_inputs(N, T, H, K, gen, device))
    res = {}
    for u_round in ("store", "step"):
        for P in (16, 32):
            for r in (0, 4):
                cfg = LQConfig(P=P, r=r, exact=True, u_round=u_round)
                st = pack(S0, cfg)
                S = S0.clone()
                worst_o = worst_s = 0.0
                for _ in range(steps):
                    inp = _rand_inputs(N, T, H, K, gen, device)
                    n = torch.randint(1, T + 1, (N,), generator=gen).to(device)
                    o_ref, S = exact_verify(S, *inp, n_acc=n)
                    o, _ = verify(st, *inp)
                    commit(st, n)
                    worst_o = max(worst_o, float(_relerr(o, o_ref, (-1,)).max()))
                    worst_s = max(
                        worst_s, float(_relerr(materialize(st), S, (-2, -1)).max())
                    )
                res[f"exact u_round={u_round} P={P} r={r}"] = (
                    worst_o,
                    worst_s,
                    st.stats["refits"],
                )
                assert worst_o < 1e-11 and worst_s < 1e-11, (
                    u_round,
                    P,
                    r,
                    worst_o,
                    worst_s,
                )
    # quantised format: verify output vs a later re-read of the committed records.
    # "step": identical (same rounded u). "store": differs only by the fp16 rounding of u.
    for u_round, tol in (("step", 1e-12), ("store", 5e-3)):
        cfg = LQConfig(u_round=u_round)
        st = pack(S0, cfg)
        S = S0.clone()
        rr = []
        worst_q_o = 0.0
        for _ in range(steps):
            inp = _rand_inputs(N, T, H, K, gen, device)
            n = torch.randint(1, T + 1, (N,), generator=gen).to(device)
            o_ref, S = exact_verify(S, *inp, n_acc=n)
            o, _ = verify(st, *inp)
            worst_q_o = max(worst_q_o, float(_relerr(o, o_ref, (-1,)).mean()))
            refit = commit(st, n)
            # re-read: for requests without a refit, S_l q^_n vs the verify output of token n
            Sl = materialize(st)
            qh, _ = prep_qk(inp[0], inp[1])
            ar = torch.arange(N, device=device)
            o_re = torch.einsum("nhvk,nhk->nhv", Sl, qh[ar, n - 1])
            keep = ~refit
            if keep.any():
                rr.append(_relerr(o_re[keep], o[ar, n - 1][keep], (-1,)).flatten())
        rr = torch.cat(rr)
        res[f"quant u_round={u_round}: verify vs re-read rel err (mean, max)"] = (
            float(rr.mean()),
            float(rr.max()),
        )
        res[f"quant u_round={u_round}: mean out rel err vs exact (sanity)"] = (
            worst_q_o,
        )
        assert float(rr.max()) < tol, (u_round, float(rr.max()))
    # determinism
    st1, st2 = pack(S0, cfg), pack(S0, cfg)
    for f_ in ("Z", "s", "c", "Kt", "Ut"):
        assert torch.equal(getattr(st1, f_), getattr(st2, f_))
    if verbose:
        for kk, vv in res.items():
            print(f"{kk}: {vv}")
        print(
            f"bytes/head (P=16, C=24, r=4): {pack(S0[:1, :1], LQConfig()).nbytes_per_head()}"
        )
        print("SELF-TEST PASSED")
    return res


if __name__ == "__main__":
    import sys

    torch.set_num_threads(
        min(8, torch.get_num_threads())
    )  # tiny ops: many threads only add overhead
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if len(sys.argv) <= 2 or sys.argv[2] != "gdn":
        self_test(device=dev)
    gdn_self_test(device=dev)
