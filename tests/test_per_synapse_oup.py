"""Unit tests for PerSynapseOUP + PerSynapseNoisyNetwork (P1).

Covers:
- initial state shape and dtype
- drift = -(x - mean)/tau
- diffusion zero on inhibitory synapses
- stationary marginal variance ~= noise_std^2 * tau / 2 within tolerance
- per-synapse noise routes into args via PerSynapseNoisyNetwork
- multiplicative-on-weight injection in compute_voltage_update preserves the
  existing per-neuron path when per-synapse noise is absent
"""

from __future__ import annotations

import diffrax as dfx
import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from adaptive_SNN.models.networks.base import LIFState  # noqa: E402
from adaptive_SNN.models.networks.default_LIF import LIFNetwork  # noqa: E402
from adaptive_SNN.models.networks.per_synapse_noisy_network import (  # noqa: E402
    PerSynapseNoisyNetwork,
    PerSynapseNoisyNetworkState,
)
from adaptive_SNN.models.noise.per_synapse_oup import PerSynapseOUP  # noqa: E402


def _make_excitatory_mask(N_neurons: int, N_inputs: int, n_E_in: int) -> jnp.ndarray:
    """Half recurrent excitatory + n_E_in excitatory inputs."""
    n_E_rec = N_neurons // 2
    return jnp.concatenate(
        [
            jnp.ones((n_E_rec,), dtype=bool),
            jnp.zeros((N_neurons - n_E_rec,), dtype=bool),
            jnp.ones((n_E_in,), dtype=bool),
            jnp.zeros((N_inputs - n_E_in,), dtype=bool),
        ]
    )


def test_initial_state_shape():
    N, N_in, n_E_in = 4, 3, 2
    mask = _make_excitatory_mask(N, N_in, n_E_in)
    oup = PerSynapseOUP(
        N_neurons=N, N_inputs=N_in, excitatory_mask=mask, tau=6e-3, noise_std=1e-9
    )
    s0 = oup.initial
    assert s0.shape == (N, N + N_in)
    assert jnp.all(s0 == 0.0)


def test_drift_value():
    N, N_in, n_E_in = 3, 2, 1
    mask = _make_excitatory_mask(N, N_in, n_E_in)
    oup = PerSynapseOUP(
        N_neurons=N, N_inputs=N_in, excitatory_mask=mask, tau=6e-3, noise_std=1e-9
    )
    x = jr.normal(jr.PRNGKey(0), (N, N + N_in))
    d = oup.drift(0.0, x, {})
    expected = -x / 6e-3
    assert jnp.allclose(d, expected, atol=1e-12)


def test_diffusion_zero_on_inhibitory_synapses():
    N, N_in, n_E_in = 4, 3, 2
    mask = _make_excitatory_mask(N, N_in, n_E_in)
    oup = PerSynapseOUP(
        N_neurons=N, N_inputs=N_in, excitatory_mask=mask, tau=6e-3, noise_std=2e-9
    )
    op = oup.diffusion(0.0, oup.initial, {})
    # mv with unit Wiener step returns scale element-wise
    unit_dw = jnp.ones((N, N + N_in))
    out = op.mv(unit_dw)
    # Inhibitory columns (where mask is False) must be zero
    inh_cols = ~mask
    assert jnp.all(out[:, inh_cols] == 0.0)
    # Excitatory columns must be sqrt(2/tau) * noise_std
    expected_exc = jnp.sqrt(2.0 / 6e-3) * 2e-9
    assert jnp.allclose(out[:, mask], expected_exc, atol=1e-15)


def test_stationary_marginal_variance():
    """Simulate the pure OU SDE for ~30 tau and check stationary variance."""
    N, N_in, n_E_in = 2, 1, 1
    mask = _make_excitatory_mask(N, N_in, n_E_in)
    tau, sigma = 6e-3, 1.5e-9
    oup = PerSynapseOUP(
        N_neurons=N, N_inputs=N_in, excitatory_mask=mask, tau=tau, noise_std=sigma
    )
    # Use VirtualBrownianTree for deterministic check; simulate t in [0, 30 tau]
    t0, t1 = 0.0, 30 * tau
    key = jr.PRNGKey(7)
    process_noise = dfx.VirtualBrownianTree(
        t0, t1, shape=oup.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea, tol=1e-5
    )
    terms = dfx.MultiTerm(
        dfx.ODETerm(oup.drift), dfx.ControlTerm(oup.diffusion, process_noise)
    )
    # Save dense over the second half (after burn-in of >10 tau)
    ts_save = jnp.linspace(15 * tau, t1, 600)
    sol = dfx.diffeqsolve(
        terms=terms,
        solver=dfx.EulerHeun(),
        t0=t0,
        t1=t1,
        dt0=tau / 50,
        y0=oup.initial,
        args={},
        adjoint=dfx.ForwardMode(),
        saveat=dfx.SaveAt(ts=ts_save),
        max_steps=20000,
    )
    ys = sol.ys  # (T, N, N+N_in)
    exc_cols = mask
    samples = ys[:, :, exc_cols]
    var_empirical = jnp.var(samples)
    # SDE: dx = -(x/tau) dt + sigma * sqrt(2/tau) dW
    # Stationary variance: (sigma^2 * 2/tau) / (2/tau) = sigma^2.
    var_theory = sigma**2
    # Allow 30% tolerance (finite sample + finite burn-in over modest # of taus).
    assert jnp.abs(var_empirical - var_theory) / var_theory < 0.30, (
        f"empirical var={var_empirical:.3e}, theory={var_theory:.3e}"
    )
    # Inhibitory must remain ~0
    inh_samples = ys[:, :, ~exc_cols]
    assert jnp.all(jnp.abs(inh_samples) < 1e-15)


def test_per_synapse_noisy_network_routes_into_args():
    """PerSynapseNoisyNetwork.drift must populate args['per_synapse_excitatory_noise']."""
    N, N_in = 3, 2
    dt = 0.1e-3
    net = LIFNetwork(N_neurons=N, N_inputs=N_in, dt=dt, key=jr.PRNGKey(0))
    mask = net.excitatory_mask
    oup = PerSynapseOUP(
        N_neurons=N, N_inputs=N_in, excitatory_mask=mask, tau=6e-3, noise_std=1e-9
    )
    netw = PerSynapseNoisyNetwork(net, oup)

    captured = {}
    base_drift = net.drift

    def wrapped_drift(t, x, args):
        captured["seen"] = args.get("per_synapse_excitatory_noise")
        captured["std"] = args.get("per_synapse_noise_std")
        return base_drift(t, x, args)

    netw_patched = netw.__class__.__new__(netw.__class__)
    object.__setattr__(netw_patched, "base_network", net)
    object.__setattr__(netw_patched, "noise_model", oup)
    object.__setattr__(netw_patched, "min_noise_std", netw.min_noise_std)
    # Just sanity: monkey-patch the base drift via attribute setter is awkward
    # in eqx; instead drive `netw.drift` and verify args inside `base_network.drift`
    args = {
        "get_input_spikes": lambda t, s, a: jnp.zeros((N, N_in)),
        "get_learning_rate": lambda t, s, a: jnp.array([0.0]),
        "RPE": jnp.array([0.0]),
        "use_noise": jnp.array(True),
        "per_synapse_noise_std_target": 1e-9,
    }
    # Inject a known per-synapse noise state and check that compute_voltage_update sees it.
    fake_noise = jnp.ones((N, N + N_in)) * 5e-9 * mask.astype(jnp.float64)[None, :]
    state = PerSynapseNoisyNetworkState(net.initial, fake_noise)

    # Compute drift; we then poke compute_voltage_update again via base_network
    # to verify the args contract.
    args_copy = dict(args)
    netw.drift(0.0, state, args_copy)
    assert "per_synapse_excitatory_noise" in args_copy
    assert args_copy["per_synapse_excitatory_noise"].shape == (N, N + N_in)
    assert jnp.allclose(args_copy["per_synapse_excitatory_noise"], fake_noise)
    assert "per_synapse_noise_std" in args_copy


def test_compute_voltage_update_no_per_synapse_noise_matches_baseline():
    """When per_synapse_excitatory_noise is absent, behaviour is unchanged."""
    N, N_in = 5, 3
    net = LIFNetwork(N_neurons=N, N_inputs=N_in, dt=0.1e-3, key=jr.PRNGKey(1))

    # State with non-trivial W, G
    state = net.initial
    rng = jr.PRNGKey(2)
    W = jnp.where(jnp.isnan(state.W), jnp.nan, 5e-9 * jr.uniform(rng, state.W.shape))
    G_vals = jr.uniform(jr.split(rng)[1], state.G.shape) * 0.5
    state = LIFState(
        V=state.V,
        S=state.S,
        W=W,
        G=G_vals,
        firing_rate=state.firing_rate,
        mean_E_conductance=state.mean_E_conductance,
        var_E_conductance=state.var_E_conductance,
        time_since_last_spike=state.time_since_last_spike,
        spike_buffer=state.spike_buffer,
        buffer_index=state.buffer_index,
        features=state.features,
    )

    base_args = {
        "excitatory_noise": jnp.zeros((N,)),
        "RPE": jnp.array([0.0]),
        "get_input_spikes": lambda t, s, a: jnp.zeros((N, N_in)),
        "get_learning_rate": lambda t, s, a: jnp.array([0.0]),
        "get_desired_balance": lambda t, s, a: 0.0,
        "noise_scale_hyperparam": 0.0,
        "use_noise": jnp.array([False] * N),
    }
    dV_baseline = net.compute_voltage_update(0.0, state, base_args)

    # Same args plus explicit zero per-synapse noise
    args2 = dict(base_args)
    args2["per_synapse_excitatory_noise"] = jnp.zeros_like(W)
    dV_with_zero_ps = net.compute_voltage_update(0.0, state, args2)
    assert jnp.allclose(dV_baseline, dV_with_zero_ps, atol=1e-15)


def test_multiplicative_on_weight_injection_effect():
    """Nonzero per-synapse noise on an active synapse changes dV in the right direction."""
    N, N_in = 2, 1
    net = LIFNetwork(
        N_neurons=N, N_inputs=N_in, dt=0.1e-3, key=jr.PRNGKey(3), fully_connected_input=True
    )
    state = net.initial
    # Force excitatory mask: neuron 0 excitatory, neuron 1 inhibitory, input excitatory
    mask = jnp.array([True, False, True])
    object.__setattr__(net, "excitatory_mask", mask)
    object.__setattr__(
        net, "synaptic_time_constants", jnp.where(mask, net.tau_E, net.tau_I)
    )

    # Build a state where:
    # - W has value 1e-9 for one specific excitatory synapse onto neuron 0 from input 0
    # - G has positive value on that synapse (synapse is active)
    W = jnp.full(state.W.shape, jnp.nan)
    W = W.at[0, 2].set(1e-9)  # excitatory input → neuron 0
    G = jnp.zeros(state.G.shape)
    G = G.at[0, 2].set(1.0)  # active
    state = LIFState(
        V=jnp.full((N,), net.resting_potential),
        S=state.S,
        W=W,
        G=G,
        firing_rate=state.firing_rate,
        mean_E_conductance=state.mean_E_conductance,
        var_E_conductance=state.var_E_conductance,
        time_since_last_spike=state.time_since_last_spike,
        spike_buffer=state.spike_buffer,
        buffer_index=state.buffer_index,
        features=state.features,
    )
    args_no_noise = {
        "excitatory_noise": jnp.zeros((N,)),
        "use_noise": jnp.array([False] * N),
    }
    dV0 = net.compute_voltage_update(0.0, state, args_no_noise)

    # Add positive per-synapse noise on the active excitatory synapse
    ps_noise = jnp.zeros_like(W)
    ps_noise = ps_noise.at[0, 2].set(2e-9)
    args_with_noise = dict(args_no_noise)
    args_with_noise["per_synapse_excitatory_noise"] = ps_noise
    dV1 = net.compute_voltage_update(0.0, state, args_with_noise)

    # Excitatory injection at rest (V < E_E) drives V upward → dV[0] > dV0[0]
    assert dV1[0] > dV0[0]
    # Neuron 1 has no excitatory synapse active here so dV should be unchanged
    assert jnp.isclose(dV1[1], dV0[1], atol=1e-12)
