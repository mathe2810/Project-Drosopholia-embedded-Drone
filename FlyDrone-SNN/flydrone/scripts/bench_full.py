"""Speed / memory benchmark of one e-prop control chunk on the full connectome.

    python -m flydrone.scripts.bench_full --batches 4 8 16 --mode spatial
"""
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from flydrone.graph import load_connectome
from flydrone import snn
from flydrone.envs import N_INPUTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='fly-brain/data')
    ap.add_argument('--ann', default='data_ext/flywire_neuron_annotations.tsv')
    ap.add_argument('--batches', type=int, nargs='+', default=[4, 8, 16])
    ap.add_argument('--mode', default='spatial')
    ap.add_argument('--n_sub', type=int, default=20)
    ap.add_argument('--no_filter_elig', action='store_true')
    a = ap.parse_args()
    d = Path(a.data)
    g = load_connectome(d / '2025_Completeness_783.csv', d / '2025_Connectivity_783.parquet', a.ann)
    print(g.summary(), jax.devices(), flush=True)
    for B in a.batches:
        cfg = snn.SNNConfig(n=g.n, n_inputs=N_INPUTS, n_actions=4, learning_signal=a.mode,
                            filter_eligibility=not a.no_filter_elig, in_scale=0.6)
        params = snn.init_params(g, cfg)
        dg = snn.to_device(g, 4)
        st = snn.init_state(params, cfg, B)
        x = jnp.ones((N_INPUTS, B)) * 0.5
        y = jnp.zeros((4, B)); lw = jnp.ones((B,)); steps = jnp.zeros((a.n_sub,), jnp.int32)
        try:
            st = snn.eprop_chunk(params, dg, cfg, st, x, y, lw, steps)
            jax.block_until_ready(st['y'])
            t0 = time.time(); reps = 3
            for _ in range(reps):
                st = snn.eprop_chunk(params, dg, cfg, st, x, y, lw, steps)
            jax.block_until_ready(st['y'])
            dt = (time.time() - t0) / reps
            mem = ''
            try:
                s = jax.devices()[0].memory_stats()
                mem = f"peak {s.get('peak_bytes_in_use', 0) / 2**30:.1f} GB"
            except Exception:
                pass
            ep = dt * 400  # 8 s episode at 50 Hz
            print(f'B={B:3d}: {dt * 1000:8.1f} ms / control step ({a.n_sub} SNN steps) | '
                  f'~{ep / 60:.1f} min per {B} episodes | {mem}', flush=True)
        except Exception as e:  # OOM etc.
            print(f'B={B}: FAILED {type(e).__name__}: {str(e)[:200]}', flush=True)
        del st


if __name__ == '__main__':
    main()
