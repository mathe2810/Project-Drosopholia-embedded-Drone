"""Fly a trained FlyDrone-SNN checkpoint in the gym-pybullet-drones GUI.

Runs the SNN forward dynamics on CUDA GPU via CuPy when available (or JAX on CPU
with --cpu).

    python -m flydrone.scripts.fly_gui --ckpt flydrone/ckpt/connectome_spatial_1d
    python -m flydrone.scripts.fly_gui --teacher      # PID teacher, for comparison
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

try:
    import cupy as cp
except ImportError:
    cp = None

from flydrone import snn
from flydrone.envs import DroneBatch, encode_obs
from flydrone.train import build


_SCATTER_CUDA = r"""
extern "C" __global__ void scatter_spikes(
    const float* __restrict__ z_del,
    const int* __restrict__ pre_ptr,
    const int* __restrict__ post_csc,
    const float* __restrict__ w_csc,
    float* __restrict__ syn,
    int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid >> 5;
    if (i >= n) return;
    if (z_del[i] <= 0.0f) return;
    int lane = threadIdx.x & 31;
    int start = pre_ptr[i];
    int end = pre_ptr[i + 1];
    for (int e = start + lane; e < end; e += 32) {
        atomicAdd(&syn[post_csc[e]], w_csc[e]);
    }
}
"""

_LIF_CUDA = r"""
extern "C" __global__ void lif_step(
    float* __restrict__ u,
    float* __restrict__ g,
    float* __restrict__ z,
    float* __restrict__ zbuf_slot,
    int* __restrict__ ref,
    float* __restrict__ syn,
    const float* __restrict__ inp,
    float alpha,
    float beta,
    int refrac_steps,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = alpha * g[i] + syn[i] + inp[i];
    float ui = beta * u[i] + (1.0f - beta) * gi - z[i];
    int ri = ref[i];
    float zi = (ui > 1.0f && ri == 0) ? 1.0f : 0.0f;
    ref[i] = (zi > 0.0f) ? refrac_steps : (ri > 0 ? ri - 1 : 0);
    g[i] = gi;
    u[i] = ui;
    z[i] = zi;
    zbuf_slot[i] = zi;
    syn[i] = 0.0f;
}
"""


class GPUForwardRunner:
    """CuPy CUDA forward simulator for the connectome SNN (batch=1)."""

    def __init__(self, g, params, cfg):
        self.cfg = cfg
        self.n = int(cfg.n)
        self.delay_steps = int(cfg.delay_steps)
        self.refrac_steps = int(cfg.refrac_steps)
        self.alpha = np.float32(cfg.alpha)
        self.beta = np.float32(cfg.beta)
        self.kappa = np.float32(cfg.kappa)

        theta = np.asarray(params['theta'], dtype=np.float32)
        w = (g.sign * theta).astype(np.float32)
        perm = np.argsort(g.pre, kind='stable')
        pre_sorted = g.pre[perm]
        pre_ptr = np.searchsorted(pre_sorted, np.arange(self.n + 1)).astype(np.int32)
        post_csc = g.post[perm].astype(np.int32)
        w_csc = w[perm].astype(np.float32)

        self.pre_ptr = cp.asarray(pre_ptr)
        self.post_csc = cp.asarray(post_csc)
        self.w_csc = cp.asarray(w_csc)
        self.in_idx = cp.asarray(g.in_idx.astype(np.int32))
        self.out_idx = cp.asarray(g.out_idx.astype(np.int32))
        self.w_in = cp.asarray(np.asarray(params['w_in'], dtype=np.float32))
        self.w_out = cp.asarray(np.asarray(params['w_out'], dtype=np.float32))
        self.b_out = cp.asarray(np.asarray(params['b_out'], dtype=np.float32)).reshape(-1, 1)

        self.u = cp.zeros((self.n,), cp.float32)
        self.g = cp.zeros((self.n,), cp.float32)
        self.z = cp.zeros((self.n,), cp.float32)
        self.zbuf = cp.zeros((self.delay_steps, self.n), cp.float32)
        self.syn = cp.zeros((self.n,), cp.float32)
        self.inp = cp.zeros((self.n,), cp.float32)
        self.ref = cp.zeros((self.n,), cp.int32)
        self.y = cp.zeros((cfg.n_actions, 1), cp.float32)
        self.ptr = 0

        self.scatter_kernel = cp.RawKernel(_SCATTER_CUDA, 'scatter_spikes')
        self.lif_kernel = cp.RawKernel(_LIF_CUDA, 'lif_step')
        self.block_size = 256
        self.grid_scatter = ((self.n * 32) + self.block_size - 1) // self.block_size
        self.grid_lif = (self.n + self.block_size - 1) // self.block_size

    def reset(self):
        self.u.fill(0)
        self.g.fill(0)
        self.z.fill(0)
        self.zbuf.fill(0)
        self.syn.fill(0)
        self.inp.fill(0)
        self.ref.fill(0)
        self.y.fill(0)
        self.ptr = 0

    def step_chunk(self, x_np: np.ndarray, n_sub: int):
        x = cp.asarray(x_np, dtype=cp.float32)
        self.inp[self.in_idx] = (self.w_in @ x)[:, 0]
        nspk = cp.zeros((), cp.float32)
        k = self.kappa
        one_minus_k = np.float32(1.0) - k
        n_i32 = np.int32(self.n)
        ref_i32 = np.int32(self.refrac_steps)
        for _ in range(n_sub):
            z_del = self.zbuf[self.ptr]
            self.scatter_kernel(
                (self.grid_scatter,), (self.block_size,),
                (z_del, self.pre_ptr, self.post_csc, self.w_csc, self.syn, n_i32),
            )
            self.lif_kernel(
                (self.grid_lif,), (self.block_size,),
                (self.u, self.g, self.z, z_del, self.ref, self.syn, self.inp,
                 self.alpha, self.beta, ref_i32, n_i32),
            )
            out_drive = (self.w_out @ self.z[self.out_idx])[:, None] + self.b_out
            self.y = k * self.y + one_minus_k * out_drive
            self.ptr = (self.ptr + 1) % self.delay_steps
            nspk += cp.sum(self.z)
        return np.clip(cp.asnumpy(self.y).T, -1, 1), float(nspk)


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


@partial(jax.jit, static_argnames=('cfg',))
def forward_chunk(params, dg, cfg, st, x, n_steps_arr):
    spike_fn = lambda v: (v > 0).astype(jnp.float32)

    def body(s, _):
        new, _, _ = snn._dynamics(params, dg, cfg, s, x, spike_fn)
        return new, jnp.sum(new['z'])
    st, nspk = jax.lax.scan(body, st, n_steps_arr)
    return st, nspk.sum()


def init_fwd_state(cfg, B):
    f = jnp.float32
    n, A = cfg.n, cfg.n_actions
    return dict(u=jnp.zeros((n, B), f), g=jnp.zeros((n, B), f), z=jnp.zeros((n, B), f),
                zbuf=jnp.zeros((cfg.delay_steps, n, B), f), ptr=jnp.zeros((), jnp.int32),
                ref=jnp.zeros((n, B), jnp.int32), y=jnp.zeros((A, B), f))


def main():
    import pybullet as p

    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='flydrone/ckpt/connectome_spatial_1d')
    ap.add_argument('--episodes', type=int, default=None, help='max episodes (default: run until closed)')
    ap.add_argument('--episode_sec', type=float, default=1e6)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--teacher', action='store_true', help='fly the PID teacher instead')
    ap.add_argument('--no_realtime', action='store_true')
    ap.add_argument('--cpu', action='store_true', help='run SNN on CPU with JAX instead of GPU')
    a = ap.parse_args()

    ckpt = Path(a.ckpt)
    targs = argparse.Namespace(**json.loads((ckpt / 'args.json').read_text()))
    env = DroneBatch(n_envs=1, act=targs.act, ctrl_freq=targs.ctrl_freq,
                     episode_sec=a.episode_sec, seed=a.seed, gui=True)
    client_id = env.envs[0].CLIENT

    runner = None
    if not a.teacher:
        print('Loading connectome...', flush=True)
        g, cfg = build(targs)
        with open(ckpt / 'ckpt.pkl', 'rb') as f:
            raw_params = pickle.load(f)['params']
        n_sub = max(1, int(round(1000.0 / targs.ctrl_freq / targs.dt)))
        gpu_name = None if a.cpu else _gpu_name()
        if gpu_name is not None:
            print(f'GPU: {gpu_name}', flush=True)
            runner = GPUForwardRunner(g, raw_params, cfg)
        else:
            print(f'CPU: {jax.devices()[0]}', flush=True)
            dg = snn.to_device(g, cfg.n_actions, targs.seed)
            params = jax.tree_util.tree_map(jnp.asarray, raw_params)
            steps_arr = jnp.zeros((n_sub,), jnp.int32)

    ep = 0
    try:
        while p.isConnected(client_id):
            obs = env.reset()
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0, cameraYaw=-30, cameraPitch=-30,
                cameraTargetPosition=env._state(0)[:3].tolist(),
                physicsClientId=client_id,
            )
            time.sleep(0.05)
            if not a.teacher:
                if runner is not None:
                    runner.reset()
                else:
                    st = init_fwd_state(cfg, 1)
            dists, crashed, k = [], False, 0
            while p.isConnected(client_id):
                t0 = time.time()
                if a.teacher:
                    action = env.teacher()
                elif runner is not None:
                    action, nspk = runner.step_chunk(encode_obs(obs), n_sub)
                else:
                    st, nspk = forward_chunk(params, dg, cfg, st, jnp.asarray(encode_obs(obs)), steps_arr)
                    action = np.clip(np.asarray(st['y']).T, -1, 1)
                obs, _, done, dist = env.step(action)
                cam = p.getDebugVisualizerCamera(physicsClientId=client_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=cam[10], cameraYaw=cam[8], cameraPitch=cam[9],
                    cameraTargetPosition=env._state(0)[:3].tolist(),
                    physicsClientId=client_id,
                )
                dists.append(float(dist[0]))
                if k % 25 == 0:
                    z = obs[0, 2] + env.target[2]
                    extra = '' if a.teacher else f' rate={float(nspk) / (cfg.n * n_sub * 1e-3):.1f}Hz'
                    print(f'ep {ep} t={k * env.ctrl_dt:4.1f}s z={z:.2f} dist={dist[0]:.3f} '
                          f'act={action[0].round(3)}{extra}', flush=True)
                if done[0]:
                    crashed = k < env.max_steps - 2
                    break
                k += 1
                if not a.no_realtime:
                    time.sleep(max(0.0, env.ctrl_dt - (time.time() - t0)))
            if dists:
                print(f'== episode {ep}: mean dist {np.mean(dists):.3f}, '
                      f'{"CRASH/out of bounds" if crashed else "completed"}', flush=True)
            ep += 1
            if a.episodes is not None and ep >= a.episodes:
                break
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        try:
            if p.isConnected(client_id):
                env.close()
        except p.error:
            pass


if __name__ == '__main__':
    main()
