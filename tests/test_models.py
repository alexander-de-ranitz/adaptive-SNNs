import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from helpers import (
    make_baseline_state,
    make_default_args,
    make_LIF_model,
    make_Noisy_LIF_model,
    make_noisy_state,
    make_OUP_model,
)

from adaptive_SNN.models import (
    LIFState,
)


def assert_state_shapes(state: LIFState, N_neurons: int, N_inputs: int):
    """Assert that all state components have correct shapes."""
    assert state.V.shape == (N_neurons,)
    assert state.S.shape == (N_neurons + N_inputs,)
    assert state.W.shape == (N_neurons, N_neurons + N_inputs)
    assert state.G.shape == (N_neurons, N_neurons + N_inputs)
    assert state.time_since_last_spike.shape == (N_neurons,)


def assert_drift_shapes(drift: LIFState, N_neurons: int, N_inputs: int):
    """Assert that drift terms have correct shapes."""
    assert drift.V.shape == (N_neurons,)
    assert drift.S.shape == (N_neurons + N_inputs,)
    assert drift.W.shape == (N_neurons, N_neurons + N_inputs)
    assert drift.G.shape == (N_neurons, N_neurons + N_inputs)


def assert_diffusion_shapes(diffusion, N_neurons: int, N_inputs: int):
    """Assert that diffusion terms have correct shapes."""
    diff = diffusion.pytree
    assert diff.V.matrix.shape == (N_neurons,)
    assert diff.S.matrix.shape == (N_neurons + N_inputs,)
    assert diff.W.matrix.shape == (N_neurons, N_neurons + N_inputs)
    assert diff.G.matrix.shape == (N_neurons, N_neurons + N_inputs)


# ============================================================================
def test_initial_state():
    N, N_inputs = 10, 3
    model = make_LIF_model(N_neurons=N, N_inputs=N_inputs)
    init_state = model.initial

    assert_state_shapes(init_state, N, N_inputs)
    assert model.excitatory_mask.shape == (N + N_inputs,)
    assert model.synaptic_time_constants.shape == (N + N_inputs,)


def test_drift_diffusion_shapes():
    N, N_inputs = 10, 3
    model = make_LIF_model(N_neurons=N, N_inputs=N_inputs)
    state = model.initial
    args = make_default_args(N, N_inputs)

    drift = model.drift(0.0, state, args)
    diffusion = model.diffusion(0.0, state, args)

    assert_drift_shapes(drift, N, N_inputs)
    assert_diffusion_shapes(diffusion, N, N_inputs)


def test_noise_shape():
    N, N_inputs = 10, 3
    model = make_LIF_model(N_neurons=N, N_inputs=N_inputs)
    noise_shape = model.noise_shape

    assert noise_shape.V.shape == (N,)
    assert noise_shape.S.shape == (N + N_inputs,)
    assert noise_shape.W.shape == (N, N + N_inputs)
    assert noise_shape.G.shape == (N, N + N_inputs)


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

    assert jnp.allclose(final_state.V, state.V)
    assert jnp.allclose(final_state.S, state.S)
    assert jnp.allclose(final_state.W, state.W)
    assert jnp.allclose(final_state.G, state.G)


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
    model = make_OUP_model(dim=dim, tau=0.1, noise_scale=0.3)
    initial_state = model.initial

    assert initial_state.shape == (dim,)

    drift = model.drift(0.0, initial_state, None)
    diffusion = model.diffusion(0.0, initial_state, None)

    assert drift.shape == (dim,)
    assert diffusion.shape == (dim, dim)


def test_OUP_drift():
    dim = 3
    model = make_OUP_model(dim=dim, tau=0.1, noise_scale=0.3)
    initial_state = jnp.array([1.0, -1.0, 0.5])

    drift = model.drift(0.0, initial_state, None)
    expected_drift = -1.0 / model.tau * (initial_state - model.mean)

    assert jnp.allclose(drift, expected_drift)


def test_OUP_diffusion():
    dim = 3
    model = make_OUP_model(dim=dim, tau=0.1, noise_scale=0.3)
    initial_state = jnp.array([1.0, -1.0, 0.5])

    diffusion = model.diffusion(0.0, initial_state, None)
    expected_diffusion = jnp.eye(dim) * jnp.sqrt(model.noise_scale)

    assert jnp.allclose(diffusion, expected_diffusion)


def test_OUP_convergence():
    """Test that the OUP converges to mean over time."""
    key = jr.PRNGKey(0)
    mean = jnp.array([0.5, -0.5, 1.0])
    dim = 3

    noise_model = make_OUP_model(dim=dim, tau=1, noise_scale=0, mean=mean)
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

    noise_model = make_OUP_model(dim=dim, tau=1, noise_scale=0.1)
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

    # Set up state with specific conductances
    G = jnp.zeros((N, N))
    G = G.at[0, 1].set(1.0).at[0, 2].set(0.5).at[1, 0].set(2) * model.synaptic_increment
    state = make_baseline_state(model, G=G)

    # Define deterministic noise and RPE
    E_noise = (jnp.arange(N, dtype=jnp.float32) + 1.0) * model.synaptic_increment
    RPE_value = 2.0

    args = make_default_args(
        N,
        0,
        excitatory_noise=E_noise,
        RPE=RPE_value,
        get_learning_rate=lambda t, x, a: 0.1,
    )

    derivs = model.drift(0.0, state, args)
    dW = derivs.W

    # Build expected dW
    E_component = jnp.outer(E_noise, excitatory_mask)
    expected_dW = (
        0.1
        * RPE_value
        * E_component
        / model.synaptic_increment
        * G
        / model.synaptic_increment
    )

    assert jnp.allclose(dW, expected_dW)
    assert dW[0, 1] == 0.0  # inhibitory presynaptic neuron
    assert (
        dW[0, 2]
        == 0.1
        * RPE_value
        * E_noise[0]
        / model.synaptic_increment
        * G[0, 2]
        / model.synaptic_increment
    )
    assert (
        dW[1, 0]
        == 0.1
        * RPE_value
        * E_noise[1]
        / model.synaptic_increment
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
        N_neurons=N, N_inputs=0, noise_scale=0.5, tau=1.0, key=key
    )

    initial_state = model.initial
    args = make_default_args(N, 0)
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
        N_neurons=N, N_inputs=0, noise_scale=0.5, tau=1.0, key=key
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
        N_neurons=N, N_inputs=0, noise_scale=0.5, tau=1.0, key=key
    )

    network = model.base_network
    network_state = make_baseline_state(network)
    noise_state = jnp.arange(N, dtype=network_state.V.dtype)
    initial_state = make_noisy_state(network_state, noise_state)
    args = make_default_args(N, 0)

    noisy_network_diff = model.diffusion(0.0, initial_state, args).pytree
    network_diff = noisy_network_diff.network_state.pytree
    dV, dS, dW, dG = (
        network_diff.V.matrix,
        network_diff.S.matrix,
        network_diff.W.matrix,
        network_diff.G.matrix,
    )

    # Network diffusion is zero
    assert jnp.allclose(dV, 0.0)
    assert jnp.allclose(dS, 0.0)
    assert jnp.allclose(dG, 0.0)
    assert jnp.allclose(dW, 0.0)

    # OU diffusion is identity * sqrt(noise_scale)
    assert jnp.allclose(
        noisy_network_diff.noise_state,
        jnp.eye(N) * jnp.sqrt(model.noise_model.noise_scale),
    )


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
        return jnp.array([1.0, 0.0, 0.0])

    args = make_default_args(N_neurons, N_inputs, get_input_spikes=input_spikes_fn)

    new_state = model.spike_and_reset(0.0, state, args=args)
    V_new, spikes, W_new, G_new = new_state.V, new_state.S, new_state.W, new_state.G

    expected_spikes = jnp.array([0, 0, 1, 0, 1, 0, 0], dtype=bool)
    expected_V_new = new_state.V.at[2].set(model.V_reset)

    assert jnp.allclose(spikes, expected_spikes)
    assert jnp.allclose(V_new, expected_V_new)
    assert jnp.allclose(
        G_new[[0, 1, 3, 4, 0, 1, 2, 3], [2, 2, 2, 2, 4, 4, 4, 4]],
        model.synaptic_increment,
        atol=1e-10,
    )  # Check synaptic increments where expected
    assert jnp.allclose(G_new[:, jnp.invert(expected_spikes)], 0.0, atol=1e-10)
    assert jnp.all(W_new == state.W)

    # Check conductance decay
    derivs = model.drift(0.0, new_state, args)
    _, dS, _, dG = derivs.V, derivs.S, derivs.W, derivs.G
    assert jnp.all(dS == 0)
    assert jnp.all(dG[:, model.N_neurons + 0] < 0)
    assert jnp.all(dG[:, model.N_neurons + 1 :] == 0)
    assert jnp.all(dG[jnp.arange(N_neurons) != 2, 2] < 0)
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
        desired_balance=lambda t, x, args: jnp.array([0.0]),  # 0.0 = no balancing
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

    args = make_default_args(
        N_neurons, N_inputs, get_desired_balance=lambda t, x, args: jnp.array([1])
    )

    state = model.initial
    balance = model.compute_balance(0, state, args)
    assert balance.shape == (N_neurons,)

    state = model.force_balanced_weights(0, model.initial, args=args)
    balance_after = model.compute_balance(0, state, args=args)
    assert jnp.allclose(balance_after, args["get_desired_balance"](0, state, args))
    assert state.W[0][2] == 2.0  # Inh weight is rescaled
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
    expected_spikes = jnp.array(
        [0, 1, 1, 0, 0, 0], dtype=bool
    )  # Last two are input neurons
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
