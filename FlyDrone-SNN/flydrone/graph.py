"""Connectome topology loading, control graphs and subgraph extraction.

Only the *topology* of the FlyWire v783 connectome is used: who connects to
whom, the sign of each connection (excitatory / inhibitory, a property of the
presynaptic neuron) and the synapse count (used only for weight
initialisation). All weights are trained afterwards.

Input layer  = neurons with FlyWire super_class == 'sensory'
Output layer = neurons with FlyWire super_class == 'descending'
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ANNOTATION_URL = (
    'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/'
    'main/supplemental_files/Supplemental_file1_neuron_annotations.tsv'
)


@dataclass
class HostGraph:
    """Graph on the host (numpy). Edges are sorted by postsynaptic index."""
    n: int
    pre: np.ndarray        # int32 (E,)
    post: np.ndarray       # int32 (E,)
    sign: np.ndarray       # float32 (E,)  +1 / -1
    count: np.ndarray      # float32 (E,)  synapse count (init only)
    in_idx: np.ndarray     # int32 (S,)  sensory neurons
    out_idx: np.ndarray    # int32 (D,)  descending neurons
    flywire_ids: np.ndarray  # int64 (n,)
    name: str = 'connectome'

    @property
    def n_edges(self):
        return int(len(self.pre))

    def summary(self):
        return (f'{self.name}: {self.n} neurons, {self.n_edges} edges, '
                f'{len(self.in_idx)} input (sensory), '
                f'{len(self.out_idx)} output (descending), '
                f'{(self.sign > 0).mean():.1%} excitatory edges')


def _sort_by_post(g: HostGraph) -> HostGraph:
    order = np.lexsort((g.pre, g.post))
    g.pre, g.post = g.pre[order], g.post[order]
    g.sign, g.count = g.sign[order], g.count[order]
    return g


def ensure_annotations(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading FlyWire annotations -> {path}')
        urllib.request.urlretrieve(ANNOTATION_URL, path)
    return path


def load_connectome(comp_path, conn_path, ann_path) -> HostGraph:
    comp = pd.read_csv(comp_path, index_col=0)
    n = len(comp)
    ann = pd.read_csv(ensure_annotations(ann_path), sep='\t', low_memory=False,
                      usecols=['root_id', 'super_class'])
    sc = (ann.drop_duplicates('root_id').set_index('root_id')['super_class']
          .reindex(comp.index).fillna('unannotated').to_numpy())

    con = pd.read_parquet(conn_path, columns=[
        'Presynaptic_Index', 'Postsynaptic_Index', 'Connectivity', 'Excitatory'])
    g = HostGraph(
        n=n,
        pre=con['Presynaptic_Index'].to_numpy(np.int32),
        post=con['Postsynaptic_Index'].to_numpy(np.int32),
        sign=np.where(con['Excitatory'].to_numpy() > 0, 1.0, -1.0).astype(np.float32),
        count=con['Connectivity'].to_numpy(np.float32),
        in_idx=np.where(sc == 'sensory')[0].astype(np.int32),
        out_idx=np.where(sc == 'descending')[0].astype(np.int32),
        flywire_ids=comp.index.to_numpy(np.int64),
    )
    return _sort_by_post(g)


# ---------------------------------------------------------------------------
# Control graphs
# ---------------------------------------------------------------------------

def degree_preserving_shuffle(g: HostGraph, seed=0) -> HostGraph:
    """Configuration-model shuffle.

    Postsynaptic endpoints are randomly re-matched to presynaptic endpoints, so
    every neuron keeps exactly its in-degree and out-degree. Sign and synapse
    count stay attached to the presynaptic side (sign is a property of the
    presynaptic neuron). Self-loops / multi-edges may appear; they are kept so
    that the edge count is identical.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(g.n_edges)
    out = HostGraph(n=g.n, pre=g.pre.copy(), post=g.post[perm].copy(),
                    sign=g.sign.copy(), count=g.count.copy(),
                    in_idx=g.in_idx.copy(), out_idx=g.out_idx.copy(),
                    flywire_ids=g.flywire_ids, name='shuffled')
    return _sort_by_post(out)


def random_graph(g: HostGraph, seed=0) -> HostGraph:
    """Erdos-Renyi graph with the same N and E; signs follow the real
    presynaptic neuron's sign; counts are resampled from the real distribution."""
    rng = np.random.default_rng(seed)
    neuron_sign = np.ones(g.n, np.float32)
    neuron_sign[g.pre] = g.sign
    pre = rng.integers(0, g.n, g.n_edges).astype(np.int32)
    post = rng.integers(0, g.n, g.n_edges).astype(np.int32)
    out = HostGraph(n=g.n, pre=pre, post=post, sign=neuron_sign[pre],
                    count=rng.choice(g.count, g.n_edges).astype(np.float32),
                    in_idx=g.in_idx.copy(), out_idx=g.out_idx.copy(),
                    flywire_ids=g.flywire_ids, name='random')
    return _sort_by_post(out)


def make_variant(g: HostGraph, variant: str, seed=0) -> HostGraph:
    if variant == 'connectome':
        return g
    if variant == 'shuffled':
        return degree_preserving_shuffle(g, seed)
    if variant == 'random':
        return random_graph(g, seed)
    raise ValueError(f'unknown graph variant {variant!r}')


# ---------------------------------------------------------------------------
# Small subgraph (for e-prop vs BPTT gradient tests)
# ---------------------------------------------------------------------------

def extract_subgraph(g: HostGraph, n_out=10, n_hidden=150, n_in=60, seed=0) -> HostGraph:
    """Induced subgraph keeping the sensory -> ... -> descending structure.

    Picks n_out descending neurons, their strongest presynaptic non-sensory
    partners (hidden), and sensory neurons presynaptic to that set.
    """
    rng = np.random.default_rng(seed)
    is_in = np.zeros(g.n, bool); is_in[g.in_idx] = True
    is_out = np.zeros(g.n, bool); is_out[g.out_idx] = True

    # descending neurons that receive at least some sensory input
    cand = g.out_idx[np.isin(g.out_idx, g.post[is_in[g.pre]])]
    outs = rng.choice(cand, size=min(n_out, len(cand)), replace=False)

    m = np.isin(g.post, outs) & ~is_in[g.pre] & ~is_out[g.pre]
    w = pd.Series(g.count[m]).groupby(g.pre[m]).sum().sort_values(ascending=False)
    hidden = w.index.to_numpy()[:n_hidden]

    tgt = np.concatenate([outs, hidden])
    m = np.isin(g.post, tgt) & is_in[g.pre]
    w = pd.Series(g.count[m]).groupby(g.pre[m]).sum().sort_values(ascending=False)
    ins = w.index.to_numpy()[:n_in]

    keep = np.concatenate([ins, hidden, outs]).astype(np.int64)
    remap = -np.ones(g.n, np.int64); remap[keep] = np.arange(len(keep))
    e = (remap[g.pre] >= 0) & (remap[g.post] >= 0)
    sub = HostGraph(
        n=len(keep),
        pre=remap[g.pre[e]].astype(np.int32), post=remap[g.post[e]].astype(np.int32),
        sign=g.sign[e].copy(), count=g.count[e].copy(),
        in_idx=np.arange(len(ins), dtype=np.int32),
        out_idx=np.arange(len(ins) + len(hidden), len(keep), dtype=np.int32),
        flywire_ids=g.flywire_ids[keep], name=f'{g.name}_sub')
    return _sort_by_post(sub)


def random_small_graph(n=40, p=0.15, n_in=10, n_out=8, seed=0) -> HostGraph:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n)) < p
    pre, post = np.nonzero(a)
    neuron_sign = np.where(rng.random(n) < 0.7, 1.0, -1.0).astype(np.float32)
    g = HostGraph(n=n, pre=pre.astype(np.int32), post=post.astype(np.int32),
                  sign=neuron_sign[pre], count=rng.integers(1, 10, len(pre)).astype(np.float32),
                  in_idx=np.arange(n_in, dtype=np.int32),
                  out_idx=np.arange(n - n_out, n, dtype=np.int32),
                  flywire_ids=np.arange(n, dtype=np.int64), name='random_small')
    return _sort_by_post(g)
