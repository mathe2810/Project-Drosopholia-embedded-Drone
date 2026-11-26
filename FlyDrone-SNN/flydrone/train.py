"""Train the connectome-topology SNN with e-prop to fly a drone (imitation of PID).

Example (Colab, A100/L4):
    cd /content/drive/MyDrive/FlyDrone-SNN
    python -m flydrone.train --data fly-brain/data \
        --ann data_ext/flywire_neuron_annotations.tsv \
        --ckpt flydrone/ckpt/connectome_spatial \
        --variant connectome --mode spatial --act one_d_rpm --n_envs 16 --episodes 300

Quick smoke test on a small subgraph (CPU OK):
    python -m flydrone.train --subgraph --episodes 2 --ctrl_steps 20 --n_envs 2
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

try:
    import cupy as cp
except ImportError:
    cp = None

from flydrone.graph import load_connectome, make_variant, extract_subgraph
from flydrone import snn
from flydrone.envs import DroneBatch, FakeDroneBatch, encode_obs, N_INPUTS


_SCATTER_BATCH_CUDA = r"""
extern "C" __global__ void scatter_spikes_batch(
    const float* __restrict__ z_del,
    const int* __restrict__ pre_ptr,
    const int* __restrict__ post_csc,
    const float* __restrict__ w_csc,
    float* __restrict__ syn,
    int n,
    int B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid >> 5;
    if (i >= n) return;
    int lane = threadIdx.x & 31;
    const float* zi = z_del + (long long)i * B;
    bool any = false;
    for (int b = 0; b < B; b++) {
        if (zi[b] > 0.0f) {
            any = true;
            break;
        }
    }
    if (!any) return;
    int start = pre_ptr[i];
    int end = pre_ptr[i + 1];
    for (int e = start + lane; e < end; e += 32) {
        float w = w_csc[e];
        int p = post_csc[e];
        float* syn_p = syn + (long long)p * B;
        for (int b = 0; b < B; b++) {
            if (zi[b] > 0.0f) {
                atomicAdd(&syn_p[b], w * zi[b]);
            }
        }
    }
}
"""

_LIF_EPROP_CUDA = r"""
extern "C" __global__ void lif_eprop_step(
    float* __restrict__ u,
    float* __restrict__ g,
    float* __restrict__ z,
    float* __restrict__ zbuf_slot,
    int* __restrict__ ref,
    float* __restrict__ syn,
    const float* __restrict__ inp,
    float* __restrict__ zs,
    float* __restrict__ zm,
    float* __restrict__ psi_out,
    float alpha,
    float beta,
    float gamma,
    int refrac_steps,
    int nB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nB) return;
    float z_prev = z[idx];
    float z_del = zbuf_slot[idx];
    int r_prev = ref[idx];
    float not_ref = (r_prev == 0) ? 1.0f : 0.0f;

    float gi = alpha * g[idx] + syn[idx] + inp[idx];
    float ui = beta * u[idx] + (1.0f - beta) * gi - z_prev;
    float zi = (ui > 1.0f && r_prev == 0) ? 1.0f : 0.0f;
    ref[idx] = (zi > 0.0f) ? refrac_steps : (r_prev > 0 ? r_prev - 1 : 0);

    float psi_i = gamma * fmaxf(0.0f, 1.0f - fabsf(ui - 1.0f)) * not_ref;
    float zsi = alpha * zs[idx] + z_del;
    float zmi = beta * zm[idx] + (1.0f - beta) * zsi;

    g[idx] = gi;
    u[idx] = ui;
    z[idx] = zi;
    zbuf_slot[idx] = zi;
    syn[idx] = 0.0f;
    zs[idx] = zsi;
    zm[idx] = zmi;
    psi_out[idx] = psi_i;
}
"""

_SPATIAL_BACKPROP_CUDA = r"""
extern "C" __global__ void spatial_backprop(
    const int* __restrict__ post_ptr,
    const int* __restrict__ pre_csr,
    const float* __restrict__ w_csr,
    const float* __restrict__ m_back,
    float* __restrict__ L_out,
    float gain,
    int n,
    int B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int j = tid >> 5;
    if (j >= n) return;
    int lane = threadIdx.x & 31;
    const float* mj = m_back + (long long)j * B;
    bool any = false;
    for (int b = 0; b < B; b++) {
        if (mj[b] != 0.0f) {
            any = true;
            break;
        }
    }
    if (!any) return;
    int start = post_ptr[j];
    int end = post_ptr[j + 1];
    for (int e = start + lane; e < end; e += 32) {
        float gw = gain * w_csr[e];
        float* out_p = L_out + (long long)pre_csr[e] * B;
        for (int b = 0; b < B; b++) {
            float mval = mj[b];
            if (mval != 0.0f) {
                atomicAdd(&out_p[b], gw * mval);
            }
        }
    }
}
"""

_ACCUM_THETA_GRAD_CUDA = r"""
extern "C" __global__ void accum_theta_grad(
    const int* __restrict__ pre_csr,
    const int* __restrict__ post_csr,
    const float* __restrict__ sign_csr,
    const signed char* __restrict__ m_nonzero,
    const float* __restrict__ M_T,
    const float* __restrict__ Z_T,
    float* __restrict__ g_theta,
    int E,
    int TB) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int post = post_csr[e];
    if (!m_nonzero[post]) return;
    const float* m_row = M_T + (long long)post * TB;
    const float* z_row = Z_T + (long long)pre_csr[e] * TB;
    float acc = 0.0f;
    if ((TB & 3) == 0) {
        const float4* m4 = reinterpret_cast<const float4*>(m_row);
        const float4* z4 = reinterpret_cast<const float4*>(z_row);
        int TB4 = TB >> 2;
        for (int k = 0; k < TB4; k++) {
            float4 mv = m4[k];
            float4 zv = z4[k];
            acc += mv.x * zv.x + mv.y * zv.y + mv.z * zv.z + mv.w * zv.w;
        }
    } else {
        for (int k = 0; k < TB; k++) {
            acc += m_row[k] * z_row[k];
        }
    }
    g_theta[e] += sign_csr[e] * acc;
}
"""

_ACCUM_THETA_GRAD_FILTERED_CUDA = r"""
extern "C" __global__ void accum_theta_grad_filtered(
    const int* __restrict__ pre_csr,
    const int* __restrict__ post_csr,
    const float* __restrict__ sign_csr,
    const signed char* __restrict__ p_nonzero,
    const float* __restrict__ L_carry,
    const float* __restrict__ M_T,
    const float* __restrict__ P_T,
    const float* __restrict__ Z_T,
    float* __restrict__ e_th,
    float* __restrict__ g_theta,
    float kappa_T,
    int E,
    int n_sub,
    int B) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int post = post_csr[e];
    int pre = pre_csr[e];
    int TB = n_sub * B;
    float* eth_row = e_th + (long long)e * B;
    const float* lc_row = L_carry + (long long)post * B;

    float grad_acc = 0.0f;
    for (int b = 0; b < B; b++) {
        grad_acc += lc_row[b] * eth_row[b];
    }

    if (!p_nonzero[post]) {
        for (int b = 0; b < B; b++) {
            eth_row[b] *= kappa_T;
        }
        g_theta[e] += sign_csr[e] * grad_acc;
        return;
    }

    const float* m_row = M_T + (long long)post * TB;
    const float* p_row = P_T + (long long)post * TB;
    const float* z_row = Z_T + (long long)pre * TB;

    if (B <= 16) {
        float eb[16];
        for (int b = 0; b < B; b++) {
            eb[b] = kappa_T * eth_row[b];
        }
        for (int s = 0; s < n_sub; s++) {
            int off = s * B;
            for (int b = 0; b < B; b++) {
                float zv = z_row[off + b];
                grad_acc += m_row[off + b] * zv;
                eb[b] += p_row[off + b] * zv;
            }
        }
        for (int b = 0; b < B; b++) {
            eth_row[b] = eb[b];
        }
    } else {
        for (int b = 0; b < B; b++) {
            float eb = kappa_T * eth_row[b];
            for (int s = 0; s < n_sub; s++) {
                int idx = s * B + b;
                float zv = z_row[idx];
                grad_acc += m_row[idx] * zv;
                eb += p_row[idx] * zv;
            }
            eth_row[b] = eb;
        }
    }
    g_theta[e] += sign_csr[e] * grad_acc;
}
"""


class GPUEpropRunner:
    """CuPy CUDA e-prop trainer for the connectome SNN."""

    def __init__(self, g, cfg: snn.SNNConfig, batch: int, seed: int = 0):
        self.cfg = cfg
        self.n = int(cfg.n)
        self.B = int(batch)
        self.C = int(cfg.n_inputs)
        self.A = int(cfg.n_actions)
        self.S = len(g.in_idx)
        self.D = len(g.out_idx)
        self.E = int(g.n_edges)
        self.delay_steps = int(cfg.delay_steps)
        self.refrac_steps = int(cfg.refrac_steps)
        self.alpha = np.float32(cfg.alpha)
        self.beta = np.float32(cfg.beta)
        self.kappa = np.float32(cfg.kappa)
        self.gamma = np.float32(cfg.gamma)
        self.spatial_gain = np.float32(cfg.spatial_gain)

        perm = np.argsort(g.pre, kind='stable').astype(np.int32)
        pre_ptr = np.searchsorted(g.pre[perm], np.arange(self.n + 1)).astype(np.int32)
        self.pre_ptr = cp.asarray(pre_ptr)
        self.post_csc = cp.asarray(g.post[perm].astype(np.int32))
        self.edge_csc = cp.asarray(perm)

        post_ptr = np.searchsorted(g.post, np.arange(self.n + 1)).astype(np.int32)
        self.post_ptr = cp.asarray(post_ptr)
        self.pre_csr = cp.asarray(g.pre.astype(np.int32))
        self.post_csr = cp.asarray(g.post.astype(np.int32))
        self.sign_csr = cp.asarray(g.sign.astype(np.float32))

        self.in_idx = cp.asarray(g.in_idx.astype(np.int32))
        self.out_idx = cp.asarray(g.out_idx.astype(np.int32))
        if cfg.learning_signal == 'random':
            rng = np.random.default_rng(seed + 12345)
            b = rng.normal(0, 1 / np.sqrt(self.D), (self.A, self.n)).astype(np.float32)
            self.b_rand = cp.asarray(b)
            is_out = np.zeros((self.n, 1), dtype=np.float32)
            is_out[g.out_idx] = 1.0
            self.non_out_mask = cp.asarray(1.0 - is_out)

        self.u = cp.zeros((self.n, self.B), cp.float32)
        self.g = cp.zeros((self.n, self.B), cp.float32)
        self.z = cp.zeros((self.n, self.B), cp.float32)
        self.zs = cp.zeros((self.n, self.B), cp.float32)
        self.zm = cp.zeros((self.n, self.B), cp.float32)
        self.zbuf = cp.zeros((self.delay_steps, self.n, self.B), cp.float32)
        self.ptr = 0
        self.ref = cp.zeros((self.n, self.B), cp.int32)
        self.y = cp.zeros((self.A, self.B), cp.float32)
        self.xs = cp.zeros((self.C, self.B), cp.float32)
        self.xm = cp.zeros((self.C, self.B), cp.float32)
        self.zk = cp.zeros((self.D, self.B), cp.float32)
        self.tb = cp.zeros((self.B,), cp.float32)
        self.e_in = cp.zeros((self.S, self.C, self.B), cp.float32)
        self.e_th = cp.zeros((self.E if cfg.filter_eligibility else 1, self.B), cp.float32)

        self.syn = cp.zeros((self.n, self.B), cp.float32)
        self.inp = cp.zeros((self.n, self.B), cp.float32)
        self.psi = cp.zeros((self.n, self.B), cp.float32)
        self.L0 = cp.zeros((self.n, self.B), cp.float32)
        self.L = cp.zeros((self.n, self.B), cp.float32)
        self.L_new = cp.zeros((self.n, self.B), cp.float32)
        self.m_back = cp.zeros((self.n, self.B), cp.float32)

        self.G = {
            'theta': cp.zeros((self.E,), cp.float32),
            'w_in': cp.zeros((self.S, self.C), cp.float32),
            'w_out': cp.zeros((self.A, self.D), cp.float32),
            'b_out': cp.zeros((self.A,), cp.float32),
        }
        self.loss = cp.zeros((), cp.float32)
        self.nspk = cp.zeros((), cp.float32)
        self._hist_steps = 0

        self.scatter_kernel = cp.RawKernel(_SCATTER_BATCH_CUDA, 'scatter_spikes_batch')
        self.lif_kernel = cp.RawKernel(_LIF_EPROP_CUDA, 'lif_eprop_step')
        self.spatial_kernel = cp.RawKernel(_SPATIAL_BACKPROP_CUDA, 'spatial_backprop')
        self.accum_kernel = cp.RawKernel(_ACCUM_THETA_GRAD_CUDA, 'accum_theta_grad')
        self.accum_filtered_kernel = cp.RawKernel(_ACCUM_THETA_GRAD_FILTERED_CUDA, 'accum_theta_grad_filtered')
        self.block_size = 256
        self.grid_warp = ((self.n * 32) + self.block_size - 1) // self.block_size
        self.grid_nB = (self.n * self.B + self.block_size - 1) // self.block_size
        self.grid_E = (self.E + self.block_size - 1) // self.block_size

    def _ensure_hist(self, n_sub: int):
        if self._hist_steps != n_sub:
            self.L_hist = cp.empty((n_sub, self.n, self.B), cp.float32)
            self.psi_hist = cp.empty((n_sub, self.n, self.B), cp.float32)
            self.zm_hist = cp.empty((n_sub, self.n, self.B), cp.float32)
            k_f = float(self.kappa)
            self._k_pow = cp.asarray(
                [k_f ** (n_sub - 1 - s) for s in range(n_sub)], dtype=cp.float32
            )[:, None, None]
            self._kappa_T = np.float32(k_f ** n_sub)
            self._hist_steps = n_sub

    def sync_params(self, params):
        self.theta = cp.asarray(np.asarray(params['theta'], dtype=np.float32))
        self.w_in = cp.asarray(np.asarray(params['w_in'], dtype=np.float32))
        self.w_out = cp.asarray(np.asarray(params['w_out'], dtype=np.float32))
        self.b_out = cp.asarray(np.asarray(params['b_out'], dtype=np.float32))
        self.w_csr = self.sign_csr * self.theta
        self.w_csc = self.w_csr[self.edge_csc]

    def reset_slots(self, mask):
        mask_np = np.asarray(mask, dtype=bool)
        if not mask_np.any():
            return
        per_sample = (self.u, self.g, self.z, self.zs, self.zm, self.zbuf,
                      self.ref, self.y, self.xs, self.xm, self.zk, self.tb,
                      self.e_in, self.e_th)
        if mask_np.all():
            for arr in per_sample:
                arr.fill(0)
            return
        keep_f = cp.asarray((~mask_np).astype(np.float32))
        keep_i = cp.asarray((~mask_np).astype(np.int32))
        for arr in per_sample:
            arr *= keep_i if arr.dtype == cp.int32 else keep_f

    def take_grads(self):
        G_host = {k: jnp.asarray(cp.asnumpy(v)) for k, v in self.G.items()}
        stats = {'loss': float(self.loss), 'nspk': float(self.nspk)}
        for v in self.G.values():
            v.fill(0)
        self.loss.fill(0)
        self.nspk.fill(0)
        return G_host, stats

    def run_chunk(self, x_np: np.ndarray, ystar_T_np: np.ndarray, lw_np: np.ndarray,
                  n_sub: int, learn: bool = True) -> np.ndarray:
        if learn:
            self._ensure_hist(n_sub)
        x = cp.asarray(x_np, dtype=cp.float32)
        ystar = cp.asarray(ystar_T_np, dtype=cp.float32)
        lw = cp.asarray(lw_np, dtype=cp.float32)
        self.inp[self.in_idx] = self.w_in @ x

        a, b, k = self.alpha, self.beta, self.kappa
        one_minus_b = np.float32(1.0) - b
        one_minus_k = np.float32(1.0) - k
        n_i32 = np.int32(self.n)
        B_i32 = np.int32(self.B)
        nB_i32 = np.int32(self.n * self.B)
        ref_i32 = np.int32(self.refrac_steps)
        mode = self.cfg.learning_signal

        for t in range(n_sub):
            z_del = self.zbuf[self.ptr]
            self.scatter_kernel(
                (self.grid_warp,), (self.block_size,),
                (z_del, self.pre_ptr, self.post_csc, self.w_csc, self.syn, n_i32, B_i32),
            )
            self.lif_kernel(
                (self.grid_nB,), (self.block_size,),
                (self.u, self.g, self.z, z_del, self.ref, self.syn, self.inp,
                 self.zs, self.zm, self.psi, a, b, self.gamma, ref_i32, nB_i32),
            )
            self.ptr = (self.ptr + 1) % self.delay_steps
            z_out = self.z[self.out_idx]
            self.y = k * self.y + one_minus_k * (self.w_out @ z_out + self.b_out[:, None])
            delta = (self.y - ystar) * lw[None, :]
            self.loss += 0.5 * cp.sum(delta ** 2)
            self.nspk += cp.sum(self.z)

            if learn:
                self.xs = a * self.xs + x
                self.xm = b * self.xm + one_minus_b * self.xs
                self.zk = k * self.zk + one_minus_k * z_out
                self.tb = k * self.tb + one_minus_k
                psi_in = self.psi[self.in_idx]
                self.e_in *= k
                self.e_in += psi_in[:, None, :] * self.xm[None, :, :]

                self.L0.fill(0)
                self.L0[self.out_idx] = one_minus_k * (self.w_out.T @ delta)
                if mode == 'eprop':
                    L_cur = self.L0
                elif mode == 'random':
                    L_cur = self.L0 + self.non_out_mask * (one_minus_k * (self.b_rand.T @ delta))
                elif mode == 'spatial':
                    cp.copyto(self.L, self.L0)
                    for _ in range(self.cfg.spatial_hops):
                        cp.multiply(self.psi, self.L, out=self.m_back)
                        cp.copyto(self.L_new, self.L0)
                        self.spatial_kernel(
                            (self.grid_warp,), (self.block_size,),
                            (self.post_ptr, self.pre_csr, self.w_csr, self.m_back,
                             self.L_new, self.spatial_gain, n_i32, B_i32),
                        )
                        self.L, self.L_new = self.L_new, self.L
                    L_cur = self.L
                else:
                    raise ValueError(mode)

                self.G['w_in'] += cp.einsum('sb,scb->sc', L_cur[self.in_idx], self.e_in)
                self.G['w_out'] += delta @ self.zk.T
                self.G['b_out'] += delta @ self.tb

                cp.copyto(self.L_hist[t], L_cur)
                cp.copyto(self.psi_hist[t], self.psi)
                cp.copyto(self.zm_hist[t], self.zm)

        if learn:
            TB = n_sub * self.B
            Z_T = cp.ascontiguousarray(self.zm_hist.transpose(1, 0, 2)).reshape(self.n, TB)
            if self.cfg.filter_eligibility:
                for s in range(n_sub - 2, -1, -1):
                    self.L_hist[s] += k * self.L_hist[s + 1]
                L_carry = cp.ascontiguousarray(k * self.L_hist[0])
                P_T = cp.ascontiguousarray(
                    (self.psi_hist * self._k_pow).transpose(1, 0, 2)
                ).reshape(self.n, TB)
                p_nonzero = cp.any(P_T != 0, axis=1).astype(cp.int8)
                self.psi_hist *= self.L_hist
                M_T = cp.ascontiguousarray(self.psi_hist.transpose(1, 0, 2)).reshape(self.n, TB)
                self.accum_filtered_kernel(
                    (self.grid_E,), (self.block_size,),
                    (self.pre_csr, self.post_csr, self.sign_csr, p_nonzero,
                     L_carry, M_T, P_T, Z_T, self.e_th, self.G['theta'],
                     self._kappa_T, np.int32(self.E), np.int32(n_sub), B_i32),
                )
            else:
                self.psi_hist *= self.L_hist
                M_T = cp.ascontiguousarray(self.psi_hist.transpose(1, 0, 2)).reshape(self.n, TB)
                m_nonzero = cp.any(M_T != 0, axis=1).astype(cp.int8)
                self.accum_kernel(
                    (self.grid_E,), (self.block_size,),
                    (self.pre_csr, self.post_csr, self.sign_csr, m_nonzero, M_T, Z_T,
                     self.G['theta'], np.int32(self.E), np.int32(TB)),
                )

        return cp.asnumpy(self.y).T


def _gpu_name() -> str | None:
    if cp is None:
        return None
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            name = cp.cuda.runtime.getDeviceProperties(0)['name']
            return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return None
    return None


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='fly-brain/data')
    ap.add_argument('--ann', default='data_ext/flywire_neuron_annotations.tsv')
    ap.add_argument('--ckpt', default='flydrone/ckpt/run')
    ap.add_argument('--variant', default='connectome', choices=['connectome', 'shuffled', 'random'])
    ap.add_argument('--mode', default='spatial', choices=['eprop', 'random', 'spatial'])
    ap.add_argument('--spatial_hops', type=int, default=3)
    ap.add_argument('--spatial_gain', type=float, default=2.0)
    ap.add_argument('--no_filter_elig', action='store_true',
                    help='drop per-synapse readout-filtered trace (halves memory, approximation)')
    ap.add_argument('--act', default='one_d_rpm', choices=['one_d_rpm', 'rpm'])
    ap.add_argument('--n_envs', type=int, default=16)
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--ctrl_steps', type=int, default=None, help='control steps per episode')
    ap.add_argument('--ctrl_freq', type=int, default=50)
    ap.add_argument('--dt', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_theta', type=float, default=1e-4)
    ap.add_argument('--update_every', type=int, default=10, help='control steps between updates')
    ap.add_argument('--dagger_episodes', type=int, default=100,
                    help='teacher-execution probability decays 1 -> 0 over this many episodes')
    ap.add_argument('--eval_every', type=int, default=10)
    ap.add_argument('--w_scale_x', type=float, default=1.0)
    ap.add_argument('--in_scale', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--subgraph', action='store_true', help='debug on a small subgraph')
    ap.add_argument('--fake_env', action='store_true', help='smoke test without pybullet')
    ap.add_argument('--cpu', action='store_true', help='force JAX CPU training instead of CuPy GPU')
    return ap.parse_args()


def build(args):
    d = Path(args.data)
    g = load_connectome(d / '2025_Completeness_783.csv', d / '2025_Connectivity_783.parquet', args.ann)
    g = make_variant(g, args.variant, args.seed)
    if args.subgraph:
        g = extract_subgraph(g, n_out=50, n_hidden=2000, n_in=500, seed=args.seed)
    print(g.summary(), flush=True)
    n_actions = 4 if args.act == 'rpm' else 1
    cfg = snn.SNNConfig(
        n=g.n, n_inputs=N_INPUTS, n_actions=n_actions, dt=args.dt,
        learning_signal=args.mode, spatial_hops=args.spatial_hops, spatial_gain=args.spatial_gain,
        filter_eligibility=not args.no_filter_elig, w_scale=0.275 / 7 * args.w_scale_x,
        in_scale=args.in_scale)
    return g, cfg


def make_optimizer(args):
    return optax.multi_transform(
        {'theta': optax.adam(args.lr_theta), 'other': optax.adam(args.lr)},
        {'theta': 'theta', 'w_in': 'other', 'w_out': 'other', 'b_out': 'other'})


def make_opt_step(opt):
    @jax.jit
    def _step(G, opt_state, params, norm):
        G = jax.tree_util.tree_map(lambda v: v * norm, G)
        upd, opt_state = opt.update(G, opt_state, params)
        params = snn.project(optax.apply_updates(params, upd))
        return params, opt_state
    return _step


@jax.jit
def _reset_slots(st, mask):
    return snn.reset_slots(st, mask)


def run_episode(args, cfg, params, dg, st, env, opt, opt_state, beta, learn, n_sub, rng,
                runner=None, opt_step=None):
    steps_arr = jnp.zeros((n_sub,), jnp.int32)
    obs = env.reset()
    if runner is not None:
        runner.reset_slots(np.ones((env.n,), bool))
    else:
        st = _reset_slots(st, jnp.ones((env.n,), bool))
    T = args.ctrl_steps or env.max_steps
    lw_np = np.ones((env.n,), np.float32)
    lw = jnp.asarray(lw_np)
    log = {'loss': 0.0, 'nspk': 0.0, 'dist': [], 'updates': 0}
    for k in range(T):
        target = env.teacher()                                   # (B, A)
        x_np = encode_obs(obs)
        if runner is not None:
            student = runner.run_chunk(x_np, target.T, lw_np, n_sub, learn=learn)
        else:
            st = snn.eprop_chunk(params, dg, cfg, st, jnp.asarray(x_np), jnp.asarray(target.T), lw, steps_arr)
            student = np.asarray(st['y']).T                      # (B, A)
        use_teacher = rng.random(env.n) < beta
        action = np.where(use_teacher[:, None], target, student)
        obs, _, done, dist = env.step(action)
        log['dist'].append(dist.mean())
        if done.any():
            if runner is not None:
                runner.reset_slots(done)
            else:
                st = _reset_slots(st, jnp.asarray(done))
        if (k + 1) % args.update_every == 0 or k == T - 1:
            if runner is not None:
                G, stats = runner.take_grads()
            else:
                G, st, stats = snn.take_grads(st)
            log['loss'] += float(stats['loss']); log['nspk'] += float(stats['nspk'])
            if learn:
                norm = np.float32(1.0 / (env.n * n_sub * args.update_every))
                if opt_step is not None:
                    params, opt_state = opt_step(G, opt_state, params, norm)
                else:
                    G = jax.tree_util.tree_map(lambda v: v * norm, G)
                    upd, opt_state = opt.update(G, opt_state, params)
                    params = snn.project(optax.apply_updates(params, upd))
                if runner is not None:
                    runner.sync_params(params)
                log['updates'] += 1
    total_steps = T * n_sub * env.n
    log['loss'] /= total_steps
    log['rate_hz'] = log['nspk'] / (cfg.n * total_steps * cfg.dt * 1e-3)
    log['dist'] = float(np.mean(log['dist']))
    lengths = env.pop_finished()
    log['crashes'] = len([l for l in lengths if l < env.max_steps - 1])
    return params, st, opt_state, log


def save_ckpt(path, params, opt_state, episode, history):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'ckpt.pkl', 'wb') as f:
        pickle.dump({'params': jax.device_get(params), 'opt_state': jax.device_get(opt_state),
                     'episode': episode}, f)
    (path / 'history.json').write_text(json.dumps(history, indent=1))


def main():
    args = parse()
    g, cfg = build(args)
    params = snn.init_params(g, cfg, args.seed)
    opt = make_optimizer(args)
    opt_state = opt.init(params)
    opt_step = make_opt_step(opt)
    start, history = 0, []
    ckpt = Path(args.ckpt)
    if (ckpt / 'ckpt.pkl').exists():
        with open(ckpt / 'ckpt.pkl', 'rb') as f:
            c = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, c['params'])
        opt_state = jax.tree_util.tree_map(jnp.asarray, c['opt_state'])
        start = c['episode'] + 1
        history = json.loads((ckpt / 'history.json').read_text())
        print(f'Resumed from episode {start}', flush=True)
    (ckpt).mkdir(parents=True, exist_ok=True)
    (ckpt / 'args.json').write_text(json.dumps(vars(args), indent=1))

    env_cls = FakeDroneBatch if args.fake_env else DroneBatch
    env = env_cls(n_envs=args.n_envs, act=args.act, ctrl_freq=args.ctrl_freq, seed=args.seed)
    n_sub = max(1, int(round(1000.0 / args.ctrl_freq / args.dt)))
    rng = np.random.default_rng(args.seed)

    gpu_name = None if args.cpu else _gpu_name()
    if gpu_name is not None:
        runner = GPUEpropRunner(g, cfg, env.n, seed=args.seed)
        runner.sync_params(params)
        dg, st = None, None
        print(f'{n_sub} SNN steps per control step; device: GPU ({gpu_name})', flush=True)
    else:
        runner = None
        dg = snn.to_device(g, cfg.n_actions, args.seed)
        st = snn.init_state(params, cfg, env.n)
        print(f'{n_sub} SNN steps per control step; devices: {jax.devices()}', flush=True)

    for ep in range(start, args.episodes):
        beta = max(0.0, 1.0 - ep / max(1, args.dagger_episodes))
        t0 = time.time()
        params, st, opt_state, log = run_episode(
            args, cfg, params, dg, st, env, opt, opt_state, beta, True, n_sub, rng,
            runner=runner, opt_step=opt_step)
        log.update(episode=ep, beta=beta, sec=time.time() - t0, phase='train')
        history.append(log)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in log.items()}),
              flush=True)
        if (ep + 1) % args.eval_every == 0 or ep == args.episodes - 1:
            _, st, _, ev = run_episode(args, cfg, params, dg, st, env, opt, opt_state,
                                       0.0, False, n_sub, rng, runner=runner, opt_step=opt_step)
            ev.update(episode=ep, phase='eval')
            history.append(ev)
            print('EVAL', json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in ev.items()}), flush=True)
            save_ckpt(ckpt, params, opt_state, ep, history)
    env.close()


if __name__ == '__main__':
    main()
