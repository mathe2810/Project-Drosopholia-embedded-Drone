---
license: agpl-3.0
tags:
- SNN
- Drosophila
- Robotics
- Connectomics
- PyBullet
---

# FlyDrone-SNN

This project uses **only the topology** of the FlyWire v783 connectome: who connects to whom and the sign of each connection (excitatory/inhibitory). A spiking neural network (LIF) is built on this topology and **all weights are trained with e-prop** to fly a drone in gym-pybullet-drones.

| | |
|---|---|
| Network | 138,639 neurons, 15,091,983 synapses. Topology and signs are fixed; weight magnitudes are trained |
| Input | `super_class == sensory` neurons (16,352) |
| Output | `super_class == descending` neurons (1,299) → readout layer → rotor commands |
| Learning | e-prop + 3-hop spatial learning signal over the connectome (`spatial`, default) |
| Task | PID teacher imitation (DAgger), first `one_d_rpm` (altitude only), then `rpm` (4 motors) |
| Control experiments | `--variant shuffled` (connections shuffled while preserving in/out degrees) and `--variant random` (Erdős–Rényi) |

## Files

- [flydrone/graph.py](flydrone/graph.py): connectome loading, control graphs, test subgraphs
- [flydrone/snn.py](flydrone/snn.py): LIF dynamics, e-prop (`eprop` / `random` / `spatial`), BPTT reference
- [flydrone/envs.py](flydrone/envs.py): batched gym-pybullet-drones environment and PID teacher (`FakeDroneBatch`: smoke test environment without pybullet)
- [flydrone/train.py](flydrone/train.py): training loop, checkpoints, evaluation
- [flydrone/tests/test_eprop.py](flydrone/tests/test_eprop.py): e-prop and BPTT correctness tests
- [flydrone/scripts/grad_alignment.py](flydrone/scripts/grad_alignment.py): alignment of learning signal methods with BPTT
- [flydrone/scripts/bench_full.py](flydrone/scripts/bench_full.py): full-network speed/memory benchmark
- [flydrone/scripts/analyze_io.py](flydrone/scripts/analyze_io.py): input/output neuron and graph distance analysis
- [notebooks/flydrone_colab.ipynb](notebooks/flydrone_colab.ipynb): end-to-end execution in Colab

## Results (local, CPU)

**Tests:** all passed.
- e-prop forward loss matches BPTT.
- Without recurrence, input and readout gradients match BPTT exactly (cosine > 0.9999).
- Eligibility traces of recurrent synapses match BPTT exactly in the feedforward case (cosine 1.0000).

**Gradient alignment with BPTT on a connectome subgraph** (2 subgraphs, ~220 neurons; cosine similarity):

| Method | Synapses onto descending neurons | Other synapses | All |
|---|---|---|---|
| `eprop` (pure) | 0.98–1.00 | 0 (not learning) | 0.47–0.74 |
| `random` | 0.98–1.00 | ~0.1 | ~0.08 |
| `spatial`, 3 hops, gain 2 | 0.99–1.00 | 0.73–0.83 | 0.76–0.93 |

Input matrix (`w_in`) alignment is weaker: 0.3–0.56.

Known limitation: in strongly recurrent regimes (above ~4x the Shiu weight scale, on small random graphs), surrogate-gradient BPTT itself becomes unreliable and alignment degrades.

## Local smoke test

```bash
python -m pytest flydrone/tests -q -s
python -m flydrone.train --subgraph --fake_env --episodes 4 --n_envs 4
```

## Metrics to track

- `rate_hz`: mean firing rate. Target is ~5–30 Hz; tuned via `--in_scale`.
- `loss`: imitation loss.
- `EVAL dist`: mean distance to target when flying only the student network.
- `crashes`: number of episodes that crashed or went out of bounds.