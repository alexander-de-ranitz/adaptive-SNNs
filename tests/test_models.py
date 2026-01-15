import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from helpers import (
    allclose_pytree,
    make_baseline_state,
    make_default_args,
    make_LIF_model,
    make_Noisy_LIF_model,
    make_noisy_state,
    make_OUP_model,
)

from adaptive_SNN.models import NoisyNetwork, NoisyNetworkState


def test_initial_state():
    N_neurons, N_inputs = 10, 3
    model = make_LIF_model(N_neurons=N_neurons, N_inputs=N_inputs)
    state = model.initial

    assert state.V.shape == (N_neurons,)
    assert state.S.shape == (N_neurons,)
    assert state.W.shape == (N_neurons, N_neurons + N_inputs)
    assert state.G.shape == (N_neurons, N_neurons + N_inputs)


def test_drift_shapes():
    N_neurons, N_inputs = 10, 3
    model = make_LIF_model(N_neurons=N_neurons, N_inputs=N_inputs)
    state = model.initial
    args = make_default_args(N_neurons, N_inputs)

    drift = model.drift(0.0, state, args)

    assert drift.V.shape == (N_neurons,)
    assert drift.S.shape == (N_neurons,)
    assert drift.W.shape == (N_neurons, N_neurons + N_inputs)
    assert drift.G.shape == (N_neurons, N_neurons + N_inputs)


def test_drift_voltage_output():
    N, N_inputs = 10, 3
    model = make_LIF_model(N_neurons=N, N_inputs=N_inputs)
    state = make_baseline_state(model)
    args = make_default_args(N, N_inputs)

    derivs = model.drift(0.0, state, args)
    dv = derivs.V
    expected_dv = (
        1
        / model.membrane_capacitance
        * -model.leak_conductance
        * (state.V - model.resting_potential)
    )

    assert jnp.allclose(dv, expected_dv)


def test_drift_conductance_output():
    N, N_inputs = 10, 3
    model = make_LIF_model(N_neurons=N, N_inputs=N_inputs)
    state = make_baseline_state(model, G=jnp.ones((N, N + N_inputs)))
    args = make_default_args(N, N_inputs)

    derivs = model.drift(0.0, state, args)
    dg = derivs.G
    expected_dg = -1 / model.synaptic_time_constants * state.G

    assert jnp.allclose(dg, expected_dg)


def test_zero_diffusion():
    N, N_inputs = 10, 3
    key = jr.PRNGKey(0)
    model = make_LIF_model(N_neurons=N, N_inputs=N_inputs, key=key)
    state = make_baseline_state(model)
    args = make_default_args(N, N_inputs)

    terms = model.terms(key)
    solver = dfx.EulerHeun()
    sol = dfx.diffeqsolve(
        terms=terms,
        solver=solver,
        t0=0.0,
        t1=0.1,
        dt0=0.01,
        y0=state,
        args=args,
        adjoint=dfx.ForwardMode(),
    )
    final_state = sol.ys

    assert allclose_pytree(final_state, state)


def test_recurrent_current():
    """Test that the recurrent current is computed correctly for both excitatory and inhibitory neurons."""
    for neuron_type in ["excitatory", "inhibitory"]:
        N = 10
        model = make_LIF_model(N_neurons=N, N_inputs=0)
        args = make_default_args(N, 0)

        # Override neuron types for test
        excitatory_mask = (
            jnp.ones((N,)) if neuron_type == "excitatory" else jnp.zeros((N,))
        )
        excitatory_mask = jnp.array(excitatory_mask, dtype=bool)
        object.__setattr__(model, "excitatory_mask", excitatory_mask)
        object.__setattr__(
            model,
            "synaptic_time_constants",
            jnp.where(model.excitatory_mask, model.tau_E, model.tau_I),
        )

        # Set specific weights and conductances
        weights = jnp.zeros((N, N))
        weights = weights.at[0, 1].set(1).at[0, 2].set(2).at[1, 0].set(1)

        conductances = jnp.zeros((N, N))
        conductances = conductances.at[0, 1].set(1).at[0, 2].set(4).at[1, 0].set(0.5)

        state = make_baseline_state(model, W=weights, G=conductances)
        derivs = model.drift(0.0, state, args)
        dv, dS = derivs.V, derivs.S

        assert jnp.all(dS == 0)

        # Calculate expected quantal size
        reversal_potential = (
            model.reversal_potential_E
            if neuron_type == "excitatory"
            else model.reversal_potential_I
        )
        quantal_size = (
            1
            * 1
            * (reversal_potential - model.resting_potential)
            / model.membrane_capacitance
        )

        assert jnp.allclose(dv[0], 9 * quantal_size)  # 1*1 + 2*4 = 9
        assert jnp.allclose(dv[1], 0.5 * quantal_size)  # 1*0.5 = 0.5
        assert jnp.all(dv[2:] == 0)


def test_input_current():
    N_neurons, N_inputs = 4, 3
    model = make_LIF_model(N_neurons=N_neurons, N_inputs=N_inputs)
    args = make_default_args(N_neurons, N_inputs)

    assert jnp.all(model.excitatory_mask.at[-N_inputs:].get() == 1)

    # Set specific weights and conductances for input connections
    weights = jnp.zeros((N_neurons, N_neurons + N_inputs))
    weights = (
        weights.at[0, N_neurons + 0]
        .set(1)
        .at[0, N_neurons + 1]
        .set(2)
        .at[1, N_neurons + 2]
        .set(3)
    )

    conductances = jnp.zeros((N_neurons, N_neurons + N_inputs))
    conductances = (
        conductances.at[0, N_neurons + 0]
        .set(1)
        .at[0, N_neurons + 1]
        .set(4)
        .at[1, N_neurons + 2]
        .set(0.5)
    )

    state = make_baseline_state(model, W=weights, G=conductances)
    derivs = model.drift(0.0, state, args)
    dv, dS = derivs.V, derivs.S

    assert jnp.all(dS == 0)

    # Calculate expected quantal size
    quantal_size = (
        model.reversal_potential_E - model.resting_potential
    ) / model.membrane_capacitance

    expected_dv_0 = (1 * 1 + 2 * 4) * quantal_size
    expected_dv_1 = (3 * 0.5) * quantal_size

    assert jnp.isclose(dv[0], expected_dv_0)
    assert jnp.isclose(dv[1], expected_dv_1)
    assert jnp.all(dv[2:] == 0)


def test_OUP_shapes():
    dim = 3
    model = make_OUP_model(dim=dim, tau=0.1, noise_std=0.3)
    initial_state = model.initial

    assert initial_state.shape == (dim,)

    drift = model.drift(0.0, initial_state, None)
    diffusion = model.diffusion(0.0, initial_state, None)

    assert drift.shape == (dim,)
    assert diffusion.shape == (dim, dim)


def test_OUP_drift():
    dim = 3
    model = make_OUP_model(dim=dim, tau=0.1, noise_std=0.3)
    initial_state = jnp.array([1.0, -1.0, 0.5])

    drift = model.drift(0.0, initial_state, None)
    expected_drift = -1.0 / model.tau * (initial_state - model.mean)

    assert jnp.allclose(drift, expected_drift)


def test_OUP_diffusion():
    dim = 3
    model = make_OUP_model(dim=dim, tau=0.1, noise_std=0.3)
    initial_state = jnp.array([1.0, -1.0, 0.5])

    diffusion = model.diffusion(0.0, initial_state, None)
    expected_diffusion = jnp.eye(dim) * model.noise_std * jnp.sqrt(2.0 / model.tau)

    assert jnp.allclose(diffusion, expected_diffusion)


def test_OUP_convergence():
    """Test that the OUP converges to mean over time."""
    key = jr.PRNGKey(0)
    mean = jnp.array([0.5, -0.5, 1.0])
    dim = 3

    noise_model = make_OUP_model(dim=dim, tau=1, noise_std=0, mean=mean)
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = jnp.array([1.0, 2.0, -3.0])

    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=0,
        t1=10,
        dt0=0.1,
        y0=init_state,
        adjoint=dfx.ForwardMode(),
        max_steps=1000,
    )

    assert jnp.all(jnp.abs(sol.ys[-1, :] - mean) < 0.01)


def test_OUP_zero_mean():
    key = jr.PRNGKey(0)
    dim = 3

    noise_model = make_OUP_model(dim=dim, tau=1, noise_std=0.1)
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = jnp.array([0.0, 0.0, 0.0])

    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=0,
        t1=1000,
        dt0=0.1,
        y0=init_state,
        adjoint=dfx.ForwardMode(),
        max_steps=None,
    )

    mean = jnp.mean(sol.ys, axis=0)
    assert jnp.all(jnp.abs(mean) < 1)


def test_weight_plasticity():
    """Test that dW is computed correctly when RPE and noises are provided."""
    N = 4
    key = jr.PRNGKey(123)
    model = make_LIF_model(N_neurons=N, N_inputs=0, key=key)

    # Override excitatory mask for test
    excitatory_mask = jnp.array([True, False, True, False], dtype=bool)
    object.__setattr__(model, "excitatory_mask", excitatory_mask)
    object.__setattr__(
        model,
        "synaptic_time_constants",
        jnp.where(model.excitatory_mask, model.tau_E, model.tau_I),
    )

    # Set up state with specific conductances
    G = jnp.zeros((N, N))
    G = G.at[0, 1].set(1.0).at[0, 2].set(0.5).at[1, 0].set(2) * model.synaptic_increment
    state = make_baseline_state(model, G=G)

    # Define deterministic noise and RPE
    E_noise = (jnp.arange(N, dtype=jnp.float32) + 1.0) * model.synaptic_increment
    noise_std = jnp.arange(3, N + 3, dtype=state.V.dtype) * 0.1

    RPE_value = 2.0

    args = make_default_args(
        N,
        0,
        excitatory_noise=E_noise,
        RPE=RPE_value,
        get_learning_rate=lambda t, x, a: 0.1,
        noise_std=noise_std,
    )

    derivs = model.drift(0.0, state, args)
    dW = derivs.W

    # Build expected dW
    noise_per_synapse = jnp.outer(E_noise / noise_std, excitatory_mask)
    expected_dW = 0.1 * RPE_value * noise_per_synapse * G / model.synaptic_increment

    assert jnp.allclose(dW, expected_dW)
    assert dW[0, 1] == 0.0  # inhibitory presynaptic neuron

    # Manual sanity check to ensure elements are in the right place
    assert (
        dW[0, 2]
        == 0.1
        * RPE_value
        * E_noise[0]
        / noise_std[0]
        * G[0, 2]
        / model.synaptic_increment
    )
    assert (
        dW[1, 0]
        == 0.1
        * RPE_value
        * E_noise[1]
        / noise_std[1]
        * G[1, 0]
        / model.synaptic_increment
    )


def test_excitatory_noise_only_affects_voltage_correctly():
    N = 7
    model = make_LIF_model(N_neurons=N, N_inputs=0)
    state = make_baseline_state(model)

    noise_E = jnp.arange(N, dtype=state.V.dtype)

    args = make_default_args(
        N,
        0,
        excitatory_noise=noise_E,
    )

    derivs = model.drift(0.0, state, args)
    dv, dG = derivs.V, derivs.G

    expected = (
        noise_E * (model.reversal_potential_E - state.V)
    ) / model.membrane_capacitance

    assert jnp.allclose(dv, expected)
    assert jnp.all(dG == 0)


def test_noise_is_unique():
    N = 5
    key = jr.PRNGKey(3)
    model = make_Noisy_LIF_model(
        N_neurons=N, N_inputs=0, noise_std=0.5, tau=1.0, key=key
    )

    initial_state: NoisyNetworkState = model.initial

    # Make sure the initial network state has non-zero conductances variance so that noise > 0
    initial_state = eqx.tree_at(
        lambda s: s.network_state.var_E_conductance,
        initial_state,
        jnp.ones((N,)),
    )

    args = make_default_args(N, 0, noise_scale_hyperparam=1.0)
    solver = dfx.EulerHeun()
    terms = model.terms(jr.PRNGKey(0))

    y1, _, _, _, _ = solver.step(terms, 0.0, 0.01, initial_state, args, None, False)
    noise_state = y1.noise_state

    assert not jnp.all(noise_state == 0)
    assert jnp.unique(noise_state).size > 1


def test_NoisyNeuronModel_forwards_noise_into_network_drift():
    N = 5
    key = jr.PRNGKey(4)
    model = make_Noisy_LIF_model(
        N_neurons=N, N_inputs=0, noise_std=0.5, tau=1.0, key=key
    )

    network = model.base_network
    network_state = make_baseline_state(network)
    args = make_default_args(N, 0)

    noise_state = jnp.arange(N, dtype=network_state.V.dtype)
    x = make_noisy_state(network_state, noise_state)

    noisy_network_drift = model.drift(0.0, x, args)
    network_drift = noisy_network_drift.network_state
    dV, dS, dW, dG = network_drift.V, network_drift.S, network_drift.W, network_drift.G

    V_now = network_state.V
    expected_dv = (
        noise_state * (network.reversal_potential_E - V_now)
    ) / network.membrane_capacitance

    assert jnp.allclose(dV, expected_dv)
    assert jnp.all(dG == 0)
    assert jnp.all(dS == 0)
    assert jnp.all(dW == 0)

    # OU drift is -1/tau * (state - mean)
    assert jnp.allclose(
        noisy_network_drift.noise_state,
        -1.0 / model.noise_model.tau * (noise_state - model.noise_model.mean),
    )


def test_NoisyNeuronModel_diffusion():
    N = 5
    key = jr.PRNGKey(5)
    model = make_Noisy_LIF_model(
        N_neurons=N, N_inputs=0, noise_std=0.0, tau=1.0, key=key
    )

    network = model.base_network
    network_state = make_baseline_state(network)
    initial_state = make_noisy_state(network_state)
    args = make_default_args(N, 0, noise_scale_hyperparam=0.0)

    noise = dfx.UnsafeBrownianPath(
        shape=model.noise_shape, key=jr.PRNGKey(12)
    ).evaluate(0, 0.01)
    diffusion = model.diffusion(0.0, initial_state, args).mv(noise)

    expected_diffusion = jax.tree.map(lambda x: jnp.zeros_like(x), initial_state)

    assert allclose_pytree(diffusion, expected_diffusion)


def test_spike_generation():
    N = 5
    model = make_LIF_model(N_neurons=N, N_inputs=0, key=jr.PRNGKey(6))

    # Set synaptic delays to zero for test
    object.__setattr__(
        model, "synaptic_delay_matrix", jnp.zeros_like(model.synaptic_delay_matrix)
    )

    V = jnp.array([-50.0, -55.0, -49.0, -60.0, -48.0]) * 1e-3
    state = make_baseline_state(model, V=V)
    args = make_default_args(N, 0)

    new_state = model.spike_and_reset(0.0, state, args)

    expected_spikes = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0])
    expected_V_new = (
        jnp.array([-50.0, -55.0, model.V_reset * 1e3, -60.0, model.V_reset * 1e3])
        * 1e-3
    )

    assert jnp.allclose(new_state.S, expected_spikes)
    assert jnp.allclose(new_state.V, expected_V_new)

    mask = jnp.array(expected_spikes, dtype=bool)
    assert jnp.allclose(
        new_state.G[[0, 1, 3, 4, 0, 1, 2, 3], [2, 2, 2, 2, 4, 4, 4, 4]],
        model.synaptic_increment,
        atol=1e-10,
    )  # Check synaptic increments where expected
    assert jnp.allclose(new_state.G[:, jnp.invert(mask)], 0.0, atol=1e-10)
    assert jnp.all(new_state.W == state.W)

    expected_time_since_last_spike = jnp.array([jnp.inf, jnp.inf, 0.0, jnp.inf, 0.0])
    assert jnp.allclose(new_state.time_since_last_spike, expected_time_since_last_spike)

    # Check that voltage remains fixed after spike
    drift = model.drift(0.0, new_state, args)
    assert jnp.all(drift.V[mask] == 0.0)
    assert jnp.all(drift.V[jnp.invert(mask)] != 0.0)


def test_spike_generation_with_input():
    N_neurons, N_inputs = 4, 3
    model = make_LIF_model(N_neurons=N_neurons, N_inputs=N_inputs, key=jr.PRNGKey(7))

    # Set synaptic delays to zero for test
    object.__setattr__(
        model, "synaptic_delay_matrix", jnp.zeros_like(model.synaptic_delay_matrix)
    )

    V = jnp.array([-70.0, -70.0, -45.0, -60.0]) * 1e-3
    state = make_baseline_state(model, V=V)

    def input_spikes_fn(t, x, args):
        spike_vec = jnp.array([1.0, 0.0, 0.0])
        return jnp.broadcast_to(spike_vec, (N_neurons, N_inputs))

    args = make_default_args(N_neurons, N_inputs, get_input_spikes=input_spikes_fn)

    new_state = model.spike_and_reset(0.0, state, args=args)
    V_new, spikes, W_new, G_new = new_state.V, new_state.S, new_state.W, new_state.G

    expected_spikes = jnp.array([0, 0, 1, 0], dtype=bool)
    expected_V_new = new_state.V.at[2].set(model.V_reset)

    assert jnp.allclose(spikes, expected_spikes)
    assert jnp.allclose(V_new, expected_V_new)

    recurrent_G = G_new[:, : model.N_neurons]
    input_G = G_new[:, model.N_neurons :]

    assert jnp.allclose(recurrent_G[:, 2], model.synaptic_increment)
    assert jnp.allclose(recurrent_G[:, jnp.array([0, 1, 3])], 0.0, atol=1e-10)
    assert jnp.allclose(input_G[:, 0], model.synaptic_increment)
    assert jnp.allclose(input_G[:, 1:], 0.0, atol=1e-10)
    assert jnp.all(W_new == state.W)

    # Check conductance decay
    derivs = model.drift(0.0, new_state, args)
    _, dS, _, dG = derivs.V, derivs.S, derivs.W, derivs.G
    assert jnp.all(dS == 0)
    assert jnp.all(dG[:, model.N_neurons + 0] < 0)
    assert jnp.all(dG[:, model.N_neurons + 1 :] == 0)
    assert jnp.all(dG[:, 2] < 0)
    assert jnp.all(dG[:, jnp.array([0, 1, 3])] == 0)


def test_force_balance_no_change():
    N_neurons, N_inputs = 10, 3
    input_types = jnp.array([1, 0, 1])

    model = make_LIF_model(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        input_neuron_types=input_types,
        key=jr.PRNGKey(7),
    )

    args = make_default_args(
        N_neurons,
        N_inputs,
        get_desired_balance=lambda t, x, args: jnp.array([0.0]),
    )

    state = model.initial
    balance = model.compute_balance(0, state, args)
    assert balance.shape == (N_neurons,)

    state_after = model.force_balanced_weights(0, model.initial, args=args)
    balance_after = model.compute_balance(0, state_after, args=args)
    assert jnp.allclose(balance_after, balance)
    assert jnp.allclose(state_after.W, state.W)


def test_force_balance_mini():
    N_neurons, N_inputs = 1, 3
    input_types = jnp.array([1, 0, 1])

    model = make_LIF_model(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        input_neuron_types=input_types,
        key=jr.PRNGKey(7),
    )

    target_balance = 2.0
    args = make_default_args(
        N_neurons,
        N_inputs,
        get_desired_balance=lambda t, x, args: jnp.array([target_balance]),
    )

    base_W = jnp.full((N_neurons, N_neurons + N_inputs), -jnp.inf)
    base_W = base_W.at[0, 1].set(1.0).at[0, 2].set(0.5).at[0, 3].set(1.0)
    state = make_baseline_state(model, W=base_W)

    balance = model.compute_balance(0, state, args)
    assert balance.shape == (N_neurons,)

    state = model.force_balanced_weights(0, state, args=args)
    balance_after = model.compute_balance(0, state, args=args)
    assert jnp.allclose(balance_after, target_balance)

    # Compute approx. charge induced at rest for each synapse type (see Kumar, 2008)
    # the ratio of these is what we define as the balance
    charge_I = (
        state.W[0][2]
        * jnp.abs(model.reversal_potential_I - model.resting_potential)
        * model.tau_I
    )
    charge_E = (
        (state.W[0][1] + state.W[0][3])
        / 2.0
        * (model.reversal_potential_E - model.resting_potential)
        * model.tau_E
    )

    assert jnp.allclose(charge_I / charge_E, target_balance)
    assert state.W[0][0] == -jnp.inf  # No self connection
    assert state.W[0][1] == 1.0 and state.W[0][1] == 1.0  # Exc weights unchanged


def test_force_balance_random():
    N_neurons, N_inputs = 50, 10
    key = jr.PRNGKey(7)
    input_types = jr.bernoulli(key, p=0.5, shape=(N_inputs,)).astype(jnp.int32)

    key, subkey = jr.split(key)
    model = make_LIF_model(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        input_neuron_types=input_types,
        key=subkey,
    )

    args = make_default_args(
        N_neurons, N_inputs, get_desired_balance=lambda t, x, args: jnp.array([2.0])
    )

    state = model.initial
    balance = model.compute_balance(0, state, args)
    assert balance.shape == (N_neurons,)

    state = model.force_balanced_weights(0, model.initial, args=args)
    balance_after = model.compute_balance(0, state, args=args)
    assert jnp.allclose(
        balance_after, args["get_desired_balance"](0, state, args), atol=1e-5
    )


def test_synaptic_delays():
    N_neurons, N_inputs = 4, 2
    dt = 1e-4
    model = make_LIF_model(
        N_neurons=N_neurons, N_inputs=N_inputs, key=jr.PRNGKey(0), dt=dt
    )

    voltages = jnp.array([-70.0, -49.0, -45.0, -60.0]) * 1e-3
    expected_spikes = jnp.array([0, 1, 1, 0], dtype=bool)
    init_state = make_baseline_state(
        model,
        V=voltages,
        W=jnp.fill_diagonal(
            jnp.ones((N_neurons, N_neurons + N_inputs)), -jnp.inf, inplace=False
        ),
    )  # Set some voltages above threshold to trigger spikes, set all w=1
    args = make_default_args(N_neurons, N_inputs)

    state = model.spike_and_reset(
        0.0, init_state, args
    )  # Generate spikes to fill buffer
    assert jnp.all(state.S == expected_spikes)
    assert jnp.all(
        state.spike_buffer[0] == expected_spikes
    )  # Spikes recorded in buffer
    assert state.buffer_index == 1  # Buffer index advanced
    assert jnp.all(state.G == 0.0)  # No conductance change yet due to delays

    max_delay = jnp.max(model.synaptic_delay_matrix)
    t = 0.0
    state = init_state
    num_events = 0
    while t < max_delay + dt:
        state = model.spike_and_reset(t, state, args)
        i, j = jnp.nonzero(state.G == model.synaptic_increment)
        num_events += jnp.size(i)

        # Check that spikes only appear after correct delay
        assert jnp.allclose(jnp.round(model.synaptic_delay_matrix[i, j], decimals=4), t)

        t += dt
        state = eqx.tree_at(
            lambda s: s.G, state, jnp.zeros_like(state.G)
        )  # Reset conductances for next step

    assert num_events == 6  # 2 spikes * 3 post-synaptic targets each


def test_noise_scaling():
    N_neurons = 1
    model = make_Noisy_LIF_model(
        N_neurons=N_neurons, N_inputs=1, noise_std=1.0, tau=1.0
    )

    state = model.initial
    noise_scale_hyperparam = 3.14
    args = make_default_args(
        N_neurons, 0, noise_scale_hyperparam=noise_scale_hyperparam
    )

    syn_var = jnp.array([20e-9])

    # Set variance to known value for test
    state = eqx.tree_at(lambda s: s.network_state.var_E_conductance, state, syn_var)

    # Compute desired noise std and compare to expected
    desired_noise_std = model.compute_desired_noise_std(0.0, state, args)
    expected_noise_std = noise_scale_hyperparam * jnp.sqrt(syn_var)

    assert expected_noise_std > NoisyNetwork.min_noise_std, (
        "Test setup invalid: synaptic std too low"
    )
    assert jnp.isclose(desired_noise_std, expected_noise_std, atol=1e-10)


def test_noise_scaling_min_clip():
    N_neurons = 1
    model = make_Noisy_LIF_model(
        N_neurons=N_neurons, N_inputs=1, noise_std=1.0, tau=1.0
    )

    state = model.initial
    noise_scale_hyperparam = 0.1  # Should not be zero, as the minimum noise is only used for noise scales > 0.0
    args = make_default_args(
        N_neurons, 0, noise_scale_hyperparam=noise_scale_hyperparam
    )

    # Compute desired noise std. Since synaptic variance is zero at init, should be clipped to min_noise_std
    desired_noise_std = model.compute_desired_noise_std(0.0, state, args)
    assert jnp.isclose(desired_noise_std, NoisyNetwork.min_noise_std)
