"""Connectome I/O analysis: sensory (input) and descending (output) neurons.

Reports counts, degree statistics and graph distances sensory -> descending
on the FlyWire v783 graph used by fly-brain.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
COMP = ROOT / 'fly-brain/data/2025_Completeness_783.csv'
CONN = ROOT / 'fly-brain/data/2025_Connectivity_783.parquet'
ANN = ROOT / 'data_ext/flywire_neuron_annotations.tsv'


def multi_source_hops(adj_csr, sources, max_hops=20):
    """BFS hop distance from any source (directed, along adj rows = pre)."""
    n = adj_csr.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    frontier = np.zeros(n, dtype=bool)
    frontier[sources] = True
    dist[sources] = 0
    for h in range(1, max_hops + 1):
        nxt = (adj_csr.T @ frontier.astype(np.float32)) > 0  # successors
        nxt &= dist < 0
        if not nxt.any():
            break
        dist[nxt] = h
        frontier = nxt
    return dist


def main():
    comp = pd.read_csv(COMP, index_col=0)
    n = len(comp)
    ann = pd.read_csv(ANN, sep='\t', low_memory=False,
                      usecols=['root_id', 'super_class', 'cell_class', 'side'])
    ann = ann.set_index('root_id').reindex(comp.index)
    sc = ann['super_class'].fillna('unannotated').to_numpy()

    con = pd.read_parquet(CONN, columns=['Presynaptic_Index', 'Postsynaptic_Index',
                                         'Connectivity', 'Excitatory'])
    pre = con['Presynaptic_Index'].to_numpy()
    post = con['Postsynaptic_Index'].to_numpy()
    E = len(con)
    adj = sp.csr_matrix((np.ones(E, np.float32), (pre, post)), shape=(n, n))
    indeg = np.asarray(adj.sum(0)).ravel()
    outdeg = np.asarray(adj.sum(1)).ravel()

    sens = np.where(sc == 'sensory')[0]
    dn = np.where(sc == 'descending')[0]

    out = {
        'n_neurons': int(n), 'n_edges': int(E),
        'excitatory_edge_frac': float(con['Excitatory'].gt(0).mean()),
        'synapse_count_median': float(con['Connectivity'].median()),
        'super_class_counts': pd.Series(sc).value_counts().to_dict(),
        'sensory': {
            'n': int(len(sens)),
            'cell_class': ann['cell_class'].iloc[sens].value_counts().to_dict(),
            'with_outgoing_edges': int((outdeg[sens] > 0).sum()),
            'median_indeg': float(np.median(indeg[sens])),
        },
        'descending': {
            'n': int(len(dn)),
            'with_incoming_edges': int((indeg[dn] > 0).sum()),
            'median_indeg': float(np.median(indeg[dn])),
            'median_outdeg_in_brain': float(np.median(outdeg[dn])),
        },
        'isolated_neurons': int(((indeg + outdeg) == 0).sum()),
    }

    # forward hops from sensory; backward hops to DN
    d_from_s = multi_source_hops(adj, sens)
    d_to_dn = multi_source_hops(adj.T.tocsr(), dn)
    dn_hops = d_from_s[dn]
    out['hops_sensory_to_dn'] = {
        'reachable_dn': int((dn_hops >= 0).sum()),
        'hist': pd.Series(dn_hops[dn_hops >= 0]).value_counts().sort_index().to_dict(),
    }
    on_path = (d_from_s >= 0) & (d_to_dn >= 0)
    out['neurons_on_sensory_to_dn_paths'] = int(on_path.sum())
    out['hops_to_dn_hist_all'] = pd.Series(d_to_dn[d_to_dn >= 0]).value_counts().sort_index().to_dict()

    print(json.dumps(out, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else str(x)))


if __name__ == '__main__':
    main()
