"""Spiking network on the connectome topology, trained online with e-prop.

Neuron model (discrete time, normalised voltage: rest = 0, threshold = 1):

    g_t = alpha * g_{t-1} + sum_i w_ji z_i(t - D) + I_in_j(t)        synaptic current
    u_t = beta  * u_{t-1} + (1 - beta) * g_t - z_{t-1}                 membrane (soft reset)
    z_t = H(u_t - 1) * [not refractory]
    y_t = kappa * y_{t-1} + (1 - kappa) * (W_out z_t[DN] + b)          leaky readout

Time constants follow Shiu et al. (tau_mem 20 ms, tau_syn 5 ms, delay 1.8 ms,
refractory 2.2 ms). Weights are w = sign * theta with theta >= 0, so the sign
of every connection (from the connectome) never changes during training.

e-prop (Bellec et al. 2020) with gradients stopped through the reset and the
refractory mask. Presynaptic traces factorise per neuron:

    zs_i = alpha * zs_i + z_i(t - D)          d g_j / d w_ji
    zm_i = beta  * zm_i + (1 - beta) * zs_i   d u_j / d w_ji
    e_ji = psi_j * zm_i                        eligibility trace
    ebar_ji = kappa * ebar_ji + e_ji           readout-filtered eligibility
    dE/dw_ji  ~=  sum_t L_j(t) * ebar_ji(t)

Learning signal L (cfg.learning_signal):
    'eprop'   : L_j = (1-kappa) sum_k W_out_kj delta_k  only for descending j
                (pure e-prop; only synapses onto descending neurons learn)
    'random'  : descending neurons as in 'eprop'; all other neurons receive
                (1-kappa) sum_k B_kj delta_k with a fixed random matrix B
    'spatial' : L = L0 + gain * W^T (psi * L), iterated `spatial_hops` times,
                i.e. the learning signal is propagated backwards through the
                connectome within the current time step (e-prop + spatial
                backprop, cf. OSTL, Bohnstingl et al. 2022)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .graph import HostGraph


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SNNConfig:
    n: int                      # number of neurons (static)
    n_inputs: int               # input channels C (static)
    n_actions: int              # readout dimension A (static)
    dt: float = 1.0             # ms
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    tau_out: float = 5.0
    t_delay: float = 1.8
    t_refrac: float = 2.2
    gamma: float = 0.3          # pseudo-derivative dampening
    learning_signal: str = 'spatial'   # 'eprop' | 'random' | 'spatial'
    spatial_hops: int = 3
    spatial_gain: float = 2.0   # best alignment with BPTT on connectome subgraphs
    filter_eligibility: bool = True    # keep per-synapse readout-filtered trace
    w_scale: float = 0.275 / 7.0       # Shiu: 0.275 mV per synapse / 7 mV to threshold
    in_scale: float = 0.1

    @property
    def alpha(self):
        return float(np.exp(-self.dt / self.tau_syn))

    @property
    def beta(self):
        return float(np.exp(-self.dt / self.tau_mem))

    @property
    def kappa(self):
        return float(np.exp(-self.dt / self.tau_out))

    @property
    def delay_steps(self):
        return max(1, int(round(self.t_delay / self.dt)))

    @property
    def refrac_steps(self):
        return max(0, int(round(self.t_refrac / self.dt)))


class DeviceGraph(NamedTuple):
    pre: jax.Array        # (E,) sorted by post
    post: jax.Array       # (E,)
    sign: jax.Array       # (E,)
    perm_pre: jax.Array   # (E,) permutation sorting edges by pre
    pre_sorted: jax.Array  # (E,) pre[perm_pre]
    in_idx: jax.Array     # (S,)
    out_idx: jax.Array    # (D,)
    b_rand: jax.Array     # (A, n) fixed random feedback (used by 'random')


def to_device(g: HostGraph, n_actions: int, seed=0) -> DeviceGraph:
    perm = np.argsort(g.pre, kind='stable').astype(np.int32)
    rng = np.random.default_rng(seed + 12345)
    b = rng.normal(0, 1 / np.sqrt(len(g.out_idx)), (n_actions, g.n)).astype(np.float32)
    return DeviceGraph(
        pre=jnp.asarray(g.pre), post=jnp.asarray(g.post), sign=jnp.asarray(g.sign),
        perm_pre=jnp.asarray(perm), pre_sorted=jnp.asarray(g.pre[perm]),
        in_idx=jnp.asarray(g.in_idx), out_idx=jnp.asarray(g.out_idx),
        b_rand=jnp.asarray(b))


def init_params(g: HostGraph, cfg: SNNConfig, seed=0):
    rng = np.random.default_rng(seed)
    S, D = len(g.in_idx), len(g.out_idx)
    return {
        'theta': jnp.asarray(g.count * cfg.w_scale, jnp.float32),
        'w_in': jnp.asarray(rng.normal(0, cfg.in_scale, (S, cfg.n_inputs)), jnp.float32),
        'w_out': jnp.asarray(rng.normal(0, 1 / np.sqrt(D), (cfg.n_actions, D)), jnp.float32),
        'b_out': jnp.zeros((cfg.n_actions,), jnp.float32),
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def init_state(params, cfg: SNNConfig, batch: int):
    n, C, A = cfg.n, cfg.n_inputs, cfg.n_actions
    E = params['theta'].shape[0]
    S, D = params['w_in'].shape[0], params['w_out'].shape[1]
    f = jnp.float32
    st = {
        'u': jnp.zeros((n, batch), f), 'g': jnp.zeros((n, batch), f),
        'z': jnp.zeros((n, batch), f),
        'zbuf': jnp.zeros((cfg.delay_steps, n, batch), f),
        'ptr': jnp.zeros((), jnp.int32),
        'ref': jnp.zeros((n, batch), jnp.int32),
        'y': jnp.zeros((A, batch), f),
        # traces
        'zs': jnp.zeros((n, batch), f), 'zm': jnp.zeros((n, batch), f),
        'xs': jnp.zeros((C, batch), f), 'xm': jnp.zeros((C, batch), f),
        'zk': jnp.zeros((D, batch), f), 'tb': jnp.zeros((batch,), f),
        'e_in': jnp.zeros((S, C, batch), f),
        'e_th': jnp.zeros((E, batch) if cfg.filter_eligibility else (1, batch), f),
        # accumulated gradients (summed over batch and time)
        'G': {k: jnp.zeros_like(v) for k, v in params.items()},
        'loss': jnp.zeros((), f),
        'nspk': jnp.zeros((), f),
    }
    return st


def reset_slots(st, mask):
    """Zero all per-sample state for batch slots where mask (B,) is True."""
    keep = (~mask).astype(jnp.float32)
    out = dict(st)
    for k, v in st.items():
        if k in ('G', 'loss', 'nspk', 'ptr'):
            continue
        out[k] = (v * keep).astype(v.dtype) if v.dtype != jnp.int32 else v * (~mask)
    return out


# ---------------------------------------------------------------------------
# Spike function with surrogate derivative (used for BPTT reference)
# ---------------------------------------------------------------------------

def make_spike_fn(gamma):
    @jax.custom_jvp
    def spike(v):
        return (v > 0).astype(jnp.float32)

    @spike.defjvp
    def _jvp(primals, tangents):
        (v,), (dv,) = primals, tangents
        return spike(v), gamma * jnp.maximum(0.0, 1.0 - jnp.abs(v)) * dv

    return spike


def _segsum(data, ids, n):
    return jax.ops.segment_sum(data, ids, num_segments=n, indices_are_sorted=True)


# ---------------------------------------------------------------------------
# Forward dynamics (shared by e-prop and BPTT)
# ---------------------------------------------------------------------------

def _dynamics(params, dg: DeviceGraph, cfg: SNNConfig, st, x, spike_fn):
    n = cfg.n
    w = dg.sign * params['theta']
    z_del = st['zbuf'][st['ptr']]
    syn = _segsum(w[:, None] * z_del[dg.pre], dg.post, n)
    inp = jnp.zeros_like(syn).at[dg.in_idx].set(params['w_in'] @ x)
    g = cfg.alpha * st['g'] + syn + inp
    u = cfg.beta * st['u'] + (1 - cfg.beta) * g - jax.lax.stop_gradient(st['z'])
    not_ref = jax.lax.stop_gradient((st['ref'] == 0).astype(jnp.float32))
    z = spike_fn(u - 1.0) * not_ref
    zs = jax.lax.stop_gradient(z)
    ref = jnp.where(zs > 0, cfg.refrac_steps, jnp.maximum(st['ref'] - 1, 0))
    y = cfg.kappa * st['y'] + (1 - cfg.kappa) * (
        params['w_out'] @ z[dg.out_idx] + params['b_out'][:, None])
    zbuf = st['zbuf'].at[st['ptr']].set(z)
    ptr = (st['ptr'] + 1) % cfg.delay_steps
    return dict(u=u, g=g, z=z, zbuf=zbuf, ptr=ptr, ref=ref, y=y), z_del, not_ref


# ---------------------------------------------------------------------------
# e-prop step
# ---------------------------------------------------------------------------

def _learning_signal(params, dg, cfg, delta, psi):
    """delta: (A, B) output error; psi: (n, B). Returns L (n, B)."""
    k1 = 1 - cfg.kappa
    L0 = jnp.zeros_like(psi).at[dg.out_idx].set(k1 * params['w_out'].T @ delta)
    mode = cfg.learning_signal
    if mode == 'eprop':
        return L0
    if mode == 'random':
        Lr = k1 * dg.b_rand.T @ delta
        is_out = jnp.zeros((cfg.n, 1)).at[dg.out_idx].set(1.0)
        return L0 + (1 - is_out) * Lr
    if mode == 'spatial':
        w = dg.sign * params['theta']
        L = L0
        for _ in range(cfg.spatial_hops):
            v = w[:, None] * (psi * L)[dg.post]            # per edge, (E, B)
            back = _segsum(v[dg.perm_pre], dg.pre_sorted, cfg.n)
            L = L0 + cfg.spatial_gain * back
        return L
    raise ValueError(mode)


def eprop_step(params, dg: DeviceGraph, cfg: SNNConfig, st, x, ystar, lw):
    """One time step. x: (C,B) input, ystar: (A,B) target, lw: (B,) loss weight."""
    spike_fn = lambda v: (v > 0).astype(jnp.float32)
    new, z_del, not_ref = _dynamics(params, dg, cfg, st, x, spike_fn)
    u, z, y = new['u'], new['z'], new['y']
    a, b, k = cfg.alpha, cfg.beta, cfg.kappa

    psi = cfg.gamma * jnp.maximum(0.0, 1.0 - jnp.abs(u - 1.0)) * not_ref

    # presynaptic traces
    zs = a * st['zs'] + z_del
    zm = b * st['zm'] + (1 - b) * zs
    xs = a * st['xs'] + x
    xm = b * st['xm'] + (1 - b) * xs
    zk = k * st['zk'] + (1 - k) * z[dg.out_idx]
    tb = k * st['tb'] + (1 - k)

    # eligibility traces
    psi_in = psi[dg.in_idx]                                    # (S,B)
    e_in = k * st['e_in'] + psi_in[:, None, :] * xm[None, :, :]
    e_now = psi[dg.post] * zm[dg.pre]                          # (E,B)
    if cfg.filter_eligibility:
        e_th = k * st['e_th'] + e_now
        e_th_store = e_th
    else:  # approximation: no readout filtering of the eligibility trace
        e_th = e_now
        e_th_store = st['e_th']

    # output error and learning signal
    delta = (y - ystar) * lw[None, :]
    L = _learning_signal(params, dg, cfg, delta, psi)

    G = st['G']
    G = {
        'theta': G['theta'] + dg.sign * jnp.sum(L[dg.post] * e_th, axis=1),
        'w_in': G['w_in'] + jnp.einsum('sb,scb->sc', L[dg.in_idx], e_in),
        'w_out': G['w_out'] + delta @ zk.T,
        'b_out': G['b_out'] + delta @ tb,
    }
    out = dict(st)
    out.update(new)
    out.update(zs=zs, zm=zm, xs=xs, xm=xm, zk=zk, tb=tb, e_in=e_in, e_th=e_th_store, G=G,
               loss=st['loss'] + 0.5 * jnp.sum(delta ** 2),
               nspk=st['nspk'] + jnp.sum(z))
    return out


@partial(jax.jit, static_argnames=('cfg',), donate_argnames=('st',))
def eprop_chunk(params, dg, cfg: SNNConfig, st, x, ystar, lw, n_steps_arr):
    """Run len(n_steps_arr) e-prop steps with constant input x and target ystar."""
    def body(s, _):
        return eprop_step(params, dg, cfg, s, x, ystar, lw), None
    st, _ = jax.lax.scan(body, st, n_steps_arr)
    return st


def eprop_run_sequence(params, dg, cfg, st, xs, ys, lws):
    """Run a full sequence (T, C, B) / (T, A, B) / (T, B) with e-prop (jit)."""
    @jax.jit
    def run(params, st, xs, ys, lws):
        def body(s, inp):
            x, y, l = inp
            return eprop_step(params, dg, cfg, s, x, y, l), None
        st, _ = jax.lax.scan(body, st, (xs, ys, lws))
        return st
    return run(params, st, xs, ys, lws)


# ---------------------------------------------------------------------------
# BPTT reference (exact gradient of the same surrogate model)
# ---------------------------------------------------------------------------

def bptt_loss(params, dg, cfg, st0, xs, ys, lws):
    spike_fn = make_spike_fn(cfg.gamma)

    def body(s, inp):
        x, ystar, lw = inp
        new, _, _ = _dynamics(params, dg, cfg, s, x, spike_fn)
        delta = (new['y'] - ystar) * lw[None, :]
        return new, 0.5 * jnp.sum(delta ** 2)

    s0 = {k: st0[k] for k in ('u', 'g', 'z', 'zbuf', 'ptr', 'ref', 'y')}
    _, losses = jax.lax.scan(body, s0, (xs, ys, lws))
    return jnp.sum(losses)


# ---------------------------------------------------------------------------
# Optimiser helpers
# ---------------------------------------------------------------------------

def take_grads(st):
    """Return accumulated grads and a state with zeroed accumulators."""
    G = st['G']
    st = dict(st)
    st['G'] = {k: jnp.zeros_like(v) for k, v in G.items()}
    stats = {'loss': st['loss'], 'nspk': st['nspk']}
    st['loss'] = jnp.zeros_like(st['loss'])
    st['nspk'] = jnp.zeros_like(st['nspk'])
    return G, st, stats


def project(params):
    """Keep connection magnitudes non-negative (sign fixed by connectome)."""
    p = dict(params)
    p['theta'] = jnp.maximum(p['theta'], 0.0)
    return p


def with_mode(cfg: SNNConfig, **kw) -> SNNConfig:
    return replace(cfg, **kw)
