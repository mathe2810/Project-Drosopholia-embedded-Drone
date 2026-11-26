"""e-prop correctness tests against BPTT (exact gradient of the same model).

Run:  python -m pytest flydrone/tests -q     (or: python flydrone/tests/test_eprop.py)
"""
import numpy as np
import jax
import jax.numpy as jnp

from flydrone.graph import random_small_graph
from flydrone import snn

jax.config.update('jax_enable_x64', False)


def _setup(g, mode='eprop', T=200, B=3, seed=0, recurrent=True, all_out=False):
    if all_out:
        g.in_idx = np.arange(g.n, dtype=np.int32)
        g.out_idx = np.arange(g.n, dtype=np.int32)
    cfg = snn.SNNConfig(n=g.n, n_inputs=5, n_actions=2, learning_signal=mode,
                        in_scale=0.6, w_scale=0.15)
    params = snn.init_params(g, cfg, seed)
    if not recurrent:
        params['theta'] = jnp.zeros_like(params['theta'])
    dg = snn.to_device(g, cfg.n_actions, seed)
    rng = np.random.default_rng(seed)
    # smooth random inputs and targets
    xs = np.cumsum(rng.normal(0, 0.3, (T, cfg.n_inputs, B)), 0)
    xs = np.abs(np.tanh(xs)) + 0.3
    ys = np.sin(np.linspace(0, 6, T))[:, None, None] * rng.normal(0, 1, (1, cfg.n_actions, B))
    lws = np.ones((T, B))
    return cfg, params, dg, jnp.asarray(xs, jnp.float32), jnp.asarray(ys, jnp.float32), jnp.asarray(lws, jnp.float32)


def _grads(cfg, params, dg, xs, ys, lws):
    B = xs.shape[2]
    st = snn.init_state(params, cfg, B)
    st = snn.eprop_run_sequence(params, dg, cfg, st, xs, ys, lws)
    st0 = snn.init_state(params, cfg, B)
    loss_bptt, g_bptt = jax.value_and_grad(snn.bptt_loss)(params, dg, cfg, st0, xs, ys, lws)
    return st, loss_bptt, g_bptt


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_forward_loss_matches():
    g = random_small_graph(seed=1)
    cfg, params, dg, xs, ys, lws = _setup(g)
    st, loss_bptt, _ = _grads(cfg, params, dg, xs, ys, lws)
    assert float(st['nspk']) > 0, 'network silent; test not informative'
    np.testing.assert_allclose(float(st['loss']), float(loss_bptt), rtol=1e-4)


def test_eprop_exact_without_recurrence():
    """No recurrent weights, every neuron is input and output -> e-prop is exact."""
    g = random_small_graph(n=30, seed=2)
    cfg, params, dg, xs, ys, lws = _setup(g, recurrent=False, all_out=True)
    st, _, gb = _grads(cfg, params, dg, xs, ys, lws)
    assert float(st['nspk']) > 0
    for k in ('w_in', 'w_out', 'b_out'):
        ge = np.asarray(st['G'][k]); gt = np.asarray(gb[k])
        assert cos(ge, gt) > 0.9999, (k, cos(ge, gt))
        np.testing.assert_allclose(ge, gt, rtol=2e-3, atol=1e-5 * np.abs(gt).max())


def test_readout_grad_exact_with_recurrence():
    g = random_small_graph(seed=3)
    cfg, params, dg, xs, ys, lws = _setup(g)
    st, _, gb = _grads(cfg, params, dg, xs, ys, lws)
    for k in ('w_out', 'b_out'):
        np.testing.assert_allclose(np.asarray(st['G'][k]), np.asarray(gb[k]), rtol=2e-3,
                                   atol=1e-5 * float(np.abs(gb[k]).max()))


def test_recurrent_alignment_positive():
    """Moderate recurrence (comparable to the Shiu weight scale). In strongly
    recurrent / chaotic regimes surrogate BPTT itself becomes unreliable."""
    g = random_small_graph(seed=4)
    is_out = np.zeros(g.n, bool); is_out[g.out_idx] = True
    m = is_out[np.asarray(g.post)]
    for mode in ('eprop', 'spatial'):
        cfg, params, dg, xs, ys, lws = _setup(g, mode=mode)
        params = dict(params); params['theta'] = params['theta'] * 0.25
        st, _, gb = _grads(cfg, params, dg, xs, ys, lws)
        ge, gt = np.asarray(st['G']['theta']), np.asarray(gb['theta'])
        c_out, c_hid = cos(ge[m], gt[m]), cos(ge[~m], gt[~m])
        print(mode, 'onto-out cos', c_out, 'hidden cos', c_hid)
        assert c_out > 0.9, (mode, c_out)
        if mode == 'spatial':
            assert c_hid > 0.5, (mode, c_hid)


if __name__ == '__main__':
    test_forward_loss_matches(); print('forward ok')
    test_eprop_exact_without_recurrence(); print('exact ok')
    test_readout_grad_exact_with_recurrence(); print('readout ok')
    test_recurrent_alignment_positive(); print('alignment ok')
