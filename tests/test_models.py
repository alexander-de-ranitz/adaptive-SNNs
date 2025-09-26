import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr

from adaptive_SNN.models.models import OUP, LIFNetwork, NoisyLIFModel


def _baseline_state(model: LIFNetwork):
    N_neurons = model.N_neurons
    N_inputs = model.N_inputs
    V = jnp.ones((N_neurons,)) * model.resting_potential  # set V = V_rest so leak = 0
    W = jnp.zeros((N_neurons, N_neurons + N_inputs))  # zero weights
    G = jnp.zeros((N_neurons, N_neurons + N_inputs))  # zero conductance
    return (V, W, G)


def test_initial_state():
    N = 10
    key = jr.PRNGKey(0)
    model = LIFNetwork(N_neurons=N, N_inputs=3, key=key)
    V_init, W_init, G_init = model.initial

    assert W_init.shape == (N, N + model.N_inputs)
    assert model.excitatory_mask.shape == (N + model.N_inputs,)
    assert model.synaptic_time_constants.shape == (N + model.N_inputs,)
    assert V_init.shape == (N,)
    assert G_init.shape == (N, N + model.N_inputs)


def test_drift_diffusion_shapes():
    N = 10
    key = jr.PRNGKey(0)
    model = LIFNetwork(N_neurons=N, N_inputs=3, key=key)
    state = model.initial

    drift = model.drift(0.0, state, None)
    diffusion = model.diffusion(0.0, state, None)

    dV, dW, dG = drift
    diffV, diffW, diffG = diffusion
    assert dV.shape == (N,)
    assert dW.shape == (N, N + model.N_inputs)
    assert dG.shape == (N, N + model.N_inputs)

    assert diffV.shape == (N,)
    assert diffW.shape == (N, N + model.N_inputs)
    assert diffG.shape == (N, N + model.N_inputs)


def test_noise_shape():
    N = 10
    key = jr.PRNGKey(0)
    model = LIFNetwork(N_neurons=N, N_inputs=3, key=key)
    noise_shape = model.noise_shape

    assert noise_shape[0].shape == (N,)
    assert noise_shape[1].shape == (N, N + model.N_inputs)
    assert noise_shape[2].shape == (N, N + model.N_inputs)


def test_drift_voltage_output():
    N = 10
    key = jr.PRNGKey(0)
    model = LIFNetwork(N_neurons=N, N_inputs=3, key=key)

    # Manually set initial state- V at zero, so leak current is predictable
    initial_state = _baseline_state(model)
    dv, _, _ = model.drift(0.0, initial_state, None)
    expected_dv = (
        1
        / model.membrane_conductance
        * -model.leak_conductance
        * (initial_state[0] - model.resting_potential)
    )

    assert jnp.allclose(dv, expected_dv)


def test_drift_conductance_output():
    N = 10
    key = jr.PRNGKey(0)
    model = LIFNetwork(N_neurons=N, N_inputs=3, key=key)

    # Manually set initial state- G at ones, so decay is predictable
    initial_state = (jnp.zeros((N,)), model.weights, jnp.ones((N, N + model.N_inputs)))
    _, _, dg = model.drift(0.0, initial_state, None)
    expected_dg = -1 / model.synaptic_time_constants * initial_state[2]

    assert jnp.allclose(dg, expected_dg)


def test_recurrent_current():
    """Test that the recurrent current is computed correctly for both excitatory and inhibitory neurons."""
    for type in ["excitatory", "inhibitory"]:
        N = 10
        key = jr.PRNGKey(0)
        model = LIFNetwork(N_neurons=N, N_inputs=0, key=key)

        # Set specific weights and conductances for testing
        excitatory_mask = (
            jnp.ones((N + model.N_inputs,))
            if type == "excitatory"
            else jnp.zeros((N + model.N_inputs,))
        )
        excitatory_mask = jnp.array(excitatory_mask, dtype=bool)
        object.__setattr__(
            model, "excitatory_mask", excitatory_mask
        )  # override neuron types for test

        # Set a few weights to non-zero values (only within recurrent part; first N columns)
        weights = jnp.zeros((N, N + model.N_inputs))
        weights = weights.at[0, 1].set(1)  # set w_01 = 1
        weights = weights.at[0, 2].set(2)  # set w_02 = 2
        weights = weights.at[1, 0].set(1)  # set w_10 = 1

        # Set conductance to non-zero where weights are non-zero (only within recurrent part)
        conductances = jnp.zeros((N, N + model.N_inputs))
        conductances = conductances.at[0, 1].set(1)  # set g_01 = 1
        conductances = conductances.at[0, 2].set(4)  # set g_02 = 4
        conductances = conductances.at[1, 0].set(0.5)  # set g_10 = 0.5

        # Initial state with all neurons at resting potential
        initial_state = (
            jnp.ones((N,)) * model.resting_potential,
            weights,
            conductances,
        )

        # Get output
        dv, _, _ = model.drift(0.0, initial_state, None)

        # Calculate expected quantal size
        # Quantal size is the size of the voltage jump for a single synaptic event with conductance 1 and weight 1
        reversal_potential = (
            model.reversal_potential_E
            if type == "excitatory"
            else model.reversal_potential_I
        )
        quantal_size = (
            1
            * 1
            * (reversal_potential - model.resting_potential)
            / model.membrane_conductance
        )

        assert jnp.allclose(dv[0], 9 * quantal_size)  # 1*1 + 2*4 = 9
        assert jnp.allclose(dv[1], 0.5 * quantal_size)  # 1*0.5 = 0.5
        assert jnp.all(dv[2:] == 0)


def test_input_current():
    N_neurons = 4
    N_inputs = 3
    key = jr.PRNGKey(0)
    model = LIFNetwork(N_neurons=N_neurons, N_inputs=N_inputs, key=key)

    assert jnp.all(
        model.excitatory_mask.at[-N_inputs:].get() == 1
    )  # input neurons are excitatory

    # Set specific weights for testing
    weights = jnp.zeros((N_neurons, N_neurons + N_inputs))
    weights = weights.at[0, N_neurons + 0].set(1)  # set w_04 = 1
    weights = weights.at[0, N_neurons + 1].set(2)  # set w_05 = 2
    weights = weights.at[1, N_neurons + 2].set(3)  # set w_16 = 3

    # Set conductance to non-zero where weights are non-zero (only within input part)
    conductances = jnp.zeros((N_neurons, N_neurons + N_inputs))
    conductances = conductances.at[0, N_neurons + 0].set(1)  # set g_04 = 1
    conductances = conductances.at[0, N_neurons + 1].set(4)  # set g_05 = 4
    conductances = conductances.at[1, N_neurons + 2].set(0.5)  # set g_16 = 0.5

    # Initial state with all neurons at resting potential
    initial_state = (
        jnp.ones((N_neurons,)) * model.resting_potential,
        weights,
        conductances,
    )

    # Get output
    dv, _, _ = model.drift(0.0, initial_state, None)

    # Calculate expected quantal size
    quantal_size = (
        model.reversal_potential_E - model.resting_potential
    ) / model.membrane_conductance

    expected_dv_0 = (1 * 1 + 2 * 4) * quantal_size
    expected_dv_1 = (3 * 0.5) * quantal_size

    assert jnp.isclose(
        dv[0], expected_dv_0
    )  # neuron 0 gets input from input neurons 0 and 1
    assert jnp.isclose(dv[1], expected_dv_1)  # neuron 1 gets input from input neuron 2
    assert jnp.all(dv[2:] == 0)  # other neurons get no input


def test_OUP_shapes():
    dim = 3
    model = OUP(theta=0.1, noise_scale=0.3, dim=dim)
    initial_state = model.initial
    assert initial_state.shape == (dim,)
    drift = model.drift(0.0, initial_state, None)
    diffusion = model.diffusion(0.0, initial_state, None)
    assert drift.shape == (dim,)
    assert diffusion.shape == (dim, dim)


def test_OUP_drift():
    dim = 3
    model = OUP(theta=0.1, noise_scale=0.3, dim=dim)
    initial_state = jnp.array([1.0, -1.0, 0.5])
    drift = model.drift(0.0, initial_state, None)
    expected_drift = -model.theta * initial_state
    assert jnp.allclose(drift, expected_drift)


def test_OUP_diffusion():
    dim = 3
    model = OUP(theta=0.1, noise_scale=0.3, dim=dim)
    initial_state = jnp.array([1.0, -1.0, 0.5])
    diffusion = model.diffusion(0.0, initial_state, None)
    expected_diffusion = jnp.eye(dim) * model.noise_scale
    assert jnp.allclose(diffusion, expected_diffusion)


def test_OUP_convergence():
    """Test that the OUP converges to mean over time."""
    key = jr.PRNGKey(0)
    t0 = 0
    t1 = 10
    dt0 = 0.1
    mean = jnp.array([0.5, -0.5, 1.0])
    dim = 3
    noise_model = OUP(
        theta=1, noise_scale=0, mean=mean, dim=dim
    )  # No noise, should decay to mean
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = jnp.array([1.0, 2.0, -3.0])  # Set initial state away from zero

    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=init_state,
        adjoint=dfx.ForwardMode(),
        max_steps=1000,
    )
    x = sol.ys
    assert jnp.all(jnp.abs(x[-1, :] - mean) < 0.01)  # Should be close to zero


def test_OUP_zero_mean():
    """Test that the OUP with noise has roughly zero mean over long time."""
    key = jr.PRNGKey(0)
    t0 = 0
    t1 = 1000
    dt0 = 0.1
    mean = 0
    dim = 3
    noise_model = OUP(theta=1, noise_scale=0.1, dim=dim)  # With noise
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = jnp.array([0.0, 0.0, 0.0])  # Start at zero

    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=init_state,
        adjoint=dfx.ForwardMode(),
        max_steps=None,
    )
    x = sol.ys
    mean = jnp.mean(x, axis=0)
    assert jnp.all(jnp.abs(mean) < 1)  # Mean should be close to zero over long time


def test_excitatory_noise_only_affects_voltage_correctly():
    N = 7
    key = jr.PRNGKey(0)
    model = LIFNetwork(N_neurons=N, key=key)

    state = _baseline_state(model)

    # Excitatory noise vector (distinct values to avoid accidental symmetry)
    noise_E = jnp.arange(N, dtype=state[0].dtype)
    noise_I = jnp.zeros((N,), dtype=state[0].dtype)

    args = {
        "excitatory_noise": lambda t, x, a: noise_E,
        "inhibitory_noise": lambda t, x, a: noise_I,
    }

    dv, _, dG = model.drift(0.0, state, args)

    expected = (
        noise_E * (model.reversal_potential_E - state[0])
    ) / model.membrane_conductance

    assert jnp.allclose(dv, expected)
    assert jnp.all(dG == 0)


def test_inhibitory_noise_only_affects_voltage_correctly():
    N = 5
    key = jr.PRNGKey(1)
    model = LIFNetwork(N_neurons=N, key=key)

    state = _baseline_state(model)

    noise_E = jnp.zeros((N,), dtype=state[0].dtype)
    noise_I = jnp.linspace(0.0, 1.0, N, dtype=state[0].dtype)

    args = {
        "excitatory_noise": lambda t, x, a: noise_E,
        "inhibitory_noise": lambda t, x, a: noise_I,
    }

    dv, _, dG = model.drift(0.0, state, args)

    expected = (
        noise_I * (model.reversal_potential_I - state[0])
    ) / model.membrane_conductance

    assert jnp.allclose(dv, expected)
    assert jnp.all(dG == 0)


def test_both_noises_add_linearly():
    N = 6
    key = jr.PRNGKey(2)
    model = LIFNetwork(N_neurons=N, key=key)
    state = _baseline_state(model)

    # Some distinct values to avoid accidental symmetry
    noise_E = jnp.linspace(0.1, 2.5, N, dtype=state[0].dtype)
    noise_I = jnp.linspace(0.0, 1.0, N, dtype=state[0].dtype)

    args = {
        "excitatory_noise": lambda t, x, a: noise_E,
        "inhibitory_noise": lambda t, x, a: noise_I,
    }

    dv, _, _ = model.drift(0.0, state, args)

    expected = (
        noise_I * (model.reversal_potential_I - state[0])
        + noise_E * (model.reversal_potential_E - state[0])
    ) / model.membrane_conductance

    assert jnp.allclose(dv, expected)


def test_noise_is_unique():
    N = 5
    key = jr.PRNGKey(3)
    network = LIFNetwork(N_neurons=N, key=key)
    noise_E = OUP(theta=1.0, noise_scale=0.5, dim=N)
    noise_I = OUP(theta=1.0, noise_scale=0.5, dim=N)

    model = NoisyLIFModel(
        N_neurons=N, neuron_model=network, noise_I_model=noise_I, noise_E_model=noise_E
    )
    initial_state = model.initial
    solver = dfx.EulerHeun()
    terms = model.terms(jr.PRNGKey(0))
    y1, _, _, _, _ = solver.step(terms, 0.0, 0.01, initial_state, None, None, False)
    (V, W, G), noise_E_state, noise_I_state = y1
    assert not jnp.all(noise_E_state == noise_I_state)
    assert not jnp.all(noise_E_state == 0)
    assert not jnp.all(noise_I_state == 0)
    assert jnp.unique(noise_E_state).size > 1
    assert jnp.unique(noise_I_state).size > 1


def test_NoisyNeuronModel_forwards_noise_into_network_drift():
    N = 5
    key = jr.PRNGKey(4)
    network = LIFNetwork(N_neurons=N, key=key)

    # Create OU processes for E/I noise with simple dynamics
    noise_E = OUP(theta=1.0, noise_scale=0.5, dim=N)
    noise_I = OUP(theta=0.5, noise_scale=0.7, dim=N)

    model = NoisyLIFModel(
        N_neurons=N, neuron_model=network, noise_I_model=noise_I, noise_E_model=noise_E
    )

    V, W, G = _baseline_state(network)
    noise_E_state = jnp.arange(N, dtype=V.dtype)
    noise_I_state = jnp.arange(N, dtype=V.dtype)[::-1]

    # x packs (network_state, noise_E_state, noise_I_state)
    x = ((V, W, G), noise_E_state, noise_I_state)

    (dV, dW, dG), d_E, d_I = model.drift(0.0, x, args=None)

    expected_dv = (
        noise_I_state * (network.reversal_potential_I - V)
        + noise_E_state * (network.reversal_potential_E - V)
    ) / network.membrane_conductance

    # Network receives noise states through args and uses them
    assert jnp.allclose(dV, expected_dv)
    assert jnp.all(dG == 0)

    # OU drifts are -theta * state
    assert jnp.allclose(d_E, -noise_E.theta * noise_E_state)
    assert jnp.allclose(d_I, -noise_I.theta * noise_I_state)


def test_NoisyNeuronModel_diffusion():
    N = 5
    key = jr.PRNGKey(5)
    network = LIFNetwork(N_neurons=N, key=key)
    noise_E = OUP(theta=1.0, noise_scale=0.5, dim=N)
    noise_I = OUP(theta=1.0, noise_scale=0.5, dim=N)

    model = NoisyLIFModel(
        N_neurons=N, neuron_model=network, noise_I_model=noise_I, noise_E_model=noise_E
    )
    V, W, G = _baseline_state(network)
    noise_E_state = jnp.arange(N, dtype=V.dtype)
    noise_I_state = jnp.arange(N, dtype=V.dtype)[::-1]
    x = ((V, W, G), noise_E_state, noise_I_state)
    (diff_V, diff_W, diff_G), diff_E, diff_I = model.diffusion(0.0, x, args=None)

    # Network diffusion is zero
    assert jnp.allclose(diff_V, 0.0)
    assert jnp.allclose(diff_G, 0.0)
    assert jnp.allclose(diff_W, 0.0)
    # OU diffusions are identity * noise_scale
    assert jnp.allclose(diff_E, jnp.eye(N) * noise_E.noise_scale)
    assert jnp.allclose(diff_I, jnp.eye(N) * noise_I.noise_scale)


def test_spike_generation():
    N = 5
    key = jr.PRNGKey(6)
    model = LIFNetwork(N_neurons=N, key=key)

    V = (
        jnp.array([-50.0, -55.0, -49.0, -60.0, -48.0]) * 1e-3
    )  # Some above and some below threshold (-50mV)
    G = jnp.zeros((N, N + model.N_inputs))
    state = (V, model.weights, G)

    (V_new, W_new, G_new), spikes = model.compute_spikes_and_update(
        0.0, state, args=None
    )

    expected_spikes = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0])
    expected_V_new = (
        jnp.array([-50.0, -55.0, model.V_reset * 1e3, -60.0, model.V_reset * 1e3])
        * 1e-3
    )

    assert jnp.allclose(spikes, expected_spikes)
    assert jnp.allclose(V_new, expected_V_new)

    mask = jnp.array(expected_spikes, dtype=bool)
    assert jnp.allclose(G_new[:, mask], model.synaptic_increment)
    assert jnp.allclose(G_new[:, jnp.invert(mask)], 0.0)
    # Weights unchanged
    assert jnp.all(W_new == state[1])


def test_spike_generation_with_input():
    N_neurons = 4
    N_inputs = 3
    key = jr.PRNGKey(7)
    model = LIFNetwork(N_neurons=N_neurons, N_inputs=N_inputs, key=key)

    V = jnp.array([-70.0, -70.0, -45.0, -60.0]) * 1e-3  # Neuron 2 will spike
    G = jnp.zeros((N_neurons, N_neurons + N_inputs))
    state = (V, model.weights, G)

    def input_spikes_fn(t, x, args):
        return jnp.array([1.0, 0.0, 0.0])  # Input neurons 0 spikes

    args = {"input_spikes": input_spikes_fn}

    (V_new, W_new, G_new), spikes = model.compute_spikes_and_update(
        0.0, state, args=args
    )
    state = (V_new, W_new, G_new)  # Update state to new state after spikes

    expected_spikes = jnp.array(
        [0, 0, 1, 0, 1, 0, 0], dtype=bool
    )  # Neuron 2 and input neuron 0 spike
    expected_V_new = state[0].at[2].set(model.V_reset)  # Neuron 2 resets

    assert jnp.allclose(spikes, expected_spikes)
    assert jnp.allclose(V_new, expected_V_new)

    assert jnp.allclose(G_new[:, expected_spikes], model.synaptic_increment)
    assert jnp.allclose(G_new[:, jnp.invert(expected_spikes)], 0.0)

    # Check that conductance decays correctly after spike
    dV, dW, dG = model.drift(0.0, state, args)
    assert jnp.all(
        dG[:, model.N_neurons + 0] < 0
    )  # Conductance from input neuron 0 should decay
    assert jnp.all(
        dG[:, model.N_neurons + 1 :] == 0
    )  # Conductance from other input neurons should not change
    assert jnp.all(dG[:, 2] < 0)  # Conductance from spiking neuron 2 should decay
    assert jnp.all(
        dG[:, jnp.array([0, 1, 3])] == 0
    )  # Other neurons are at zero conductance, should not change
