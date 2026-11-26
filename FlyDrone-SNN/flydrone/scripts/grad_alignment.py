"""Gradient alignment of e-prop variants vs BPTT on a connectome subgraph.

Usage:
    python -m flydrone.scripts.grad_alignment --data fly-brain/data --ann data_ext/flywire_neuron_annotations.tsv
"""
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from flydrone.graph import load_connectome, extract_subgraph, make_variant
from flydrone import snn


def cos(a, b):
    a, b = np.ravel(np.asarray(a)), np.ravel(np.asarray(b))
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def make_data(cfg, T, B, seed):
    rng = np.random.default_rng(seed)
    xs = np.cumsum(rng.normal(0, 0.3, (T, cfg.n_inputs, B)), 0)
    xs = np.abs(np.tanh(xs)) + 0.3
    ys = np.sin(np.linspace(0, 6, T))[:, None, None] * rng.normal(0, 1, (1, cfg.n_actions, B))
    return (jnp.asarray(xs, jnp.float32), jnp.asarray(ys, jnp.float32),
            jnp.ones((T, B), jnp.float32))


def evaluate(sub, cfg, params, xs, ys, lws):
    dg = snn.to_device(sub, cfg.n_actions)
    B = xs.shape[2]
    st0 = snn.init_state(params, cfg, B)
    gb = jax.grad(snn.bptt_loss)(params, dg, cfg, st0, xs, ys, lws)
    st = snn.eprop_run_sequence(params, dg, cfg, snn.init_state(params, cfg, B), xs, ys, lws)
    ge = st['G']
    is_out = np.zeros(sub.n, bool); is_out[sub.out_idx] = True
    m = is_out[np.asarray(sub.post)]
    gth, bth = np.asarray(ge['theta']), np.asarray(gb['theta'])
    return {
        'theta_all': cos(gth, bth), 'theta_onto_out': cos(gth[m], bth[m]),
        'theta_hidden': cos(gth[~m], bth[~m]), 'w_in': cos(ge['w_in'], gb['w_in']),
        'w_out': cos(ge['w_out'], gb['w_out']),
        'rate_hz': float(st['nspk']) / (sub.n * B * xs.shape[0] * cfg.dt * 1e-3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='fly-brain/data')
    ap.add_argument('--ann', default='data_ext/flywire_neuron_annotations.tsv')
    ap.add_argument('--variant', default='connectome')
    ap.add_argument('--T', type=int, default=300)
    ap.add_argument('--B', type=int, default=4)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--w_scales', type=float, nargs='+', default=[1.0, 2.0, 4.0])
    ap.add_argument('--out', default='flydrone/results/grad_alignment.csv')
    a = ap.parse_args()

    d = Path(a.data)
    full = load_connectome(d / '2025_Completeness_783.csv', d / '2025_Connectivity_783.parquet', a.ann)
    full = make_variant(full, a.variant)
    rows = []
    configs = [('eprop', 0, 1.0), ('random', 0, 1.0)] + [
        ('spatial', h, gain) for h in (1, 2, 3) for gain in (0.5, 1.0, 2.0, 5.0)]
    for seed in range(a.seeds):
        sub = extract_subgraph(full, seed=seed)
        print(sub.summary(), flush=True)
        for ws in a.w_scales:
            for mode, hops, gain in configs:
                cfg = snn.SNNConfig(n=sub.n, n_inputs=6, n_actions=4, learning_signal=mode,
                                    spatial_hops=max(hops, 1), spatial_gain=gain,
                                    w_scale=0.275 / 7 * ws, in_scale=0.6)
                params = snn.init_params(sub, cfg, seed)
                xs, ys, lws = make_data(cfg, a.T, a.B, seed)
                r = evaluate(sub, cfg, params, xs, ys, lws)
                r.update(seed=seed, w_scale_x=ws, mode=mode, hops=hops, gain=gain)
                rows.append(r)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    summ = df.groupby(['w_scale_x', 'mode', 'hops', 'gain'])[
        ['theta_all', 'theta_onto_out', 'theta_hidden', 'w_in', 'rate_hz']].mean().round(3)
    print(summ.to_string())


if __name__ == '__main__':
    main()
