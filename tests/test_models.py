from adaptive_SNN.models.models import NeuronModel, OUP
import jax.random as jr
import jax.numpy as jnp
import diffrax as dfx
from adaptive_SNN.models.models import NeuronModel, NoisyNeuronModel, OUP


def _baseline_state(model: NeuronModel):
    N = model.N_neurons
    V = jnp.ones((N,)) * model.resting_potential  # set V = V_rest so leak = 0
    G = jnp.zeros((N, N))
    return (V, G)


def test_parameter_shapes():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)

    assert model.input_weights.shape == (N, 3)
    assert model.recurrent_weights.shape == (N, N)
    assert model.excitatory_mask.shape == (N,)
    assert model.synaptic_time_constants.shape == (N,)

def test_initial_state():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)
    initial_state = model.initial

    assert initial_state[0].shape == (N,)
    assert initial_state[1].shape == (N, N)

def test_drift_diffusion_shapes():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)
    initial_state = model.initial

    drift = model.drift(0.0, initial_state, None)
    diffusion = model.diffusion(0.0, initial_state, None)

    assert drift[0].shape == (N,)
    assert drift[1].shape == (N, N)

    assert diffusion[0].shape == (N,)
    assert diffusion[1].shape == (N, N)

def test_noise_shape():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)
    noise_shape = model.noise_shape

    assert noise_shape[0].shape == (N,)
    assert noise_shape[1].shape == (N, N)

def test_drift_voltage_output():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)

    # Manually set initial state- V at zero, so leak current is predictable
    initial_state = jnp.zeros((N,)), jnp.zeros((N, N))
    drift = model.drift(0.0, initial_state, None)
    dv = drift[0]
    expected_dv = 1/model.membrane_conductance * -model.leak_conductance * (initial_state[0] - model.resting_potential)

    assert jnp.allclose(dv, expected_dv)

def test_drift_conductance_output():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)

    # Manually set initial state- G at ones, so decay is predictable
    initial_state = jnp.zeros((N,)), jnp.ones((N, N))
    drift = model.drift(0.0, initial_state, None)
    dg = drift[1]
    expected_dg = -1/model.synaptic_time_constants * initial_state[1]

    assert jnp.allclose(dg, expected_dg)


def test_recurrent_current():
    """Test that the recurrent current is computed correctly for both excitatory and inhibitory neurons."""
    for type in ['excitatory', 'inhibitory']:
        N = 10
        key = jr.PRNGKey(0)
        model = NeuronModel(N_neurons=N, num_inputs=3, key=key)

        # Set specific weights and conductances for testing
        excitatory_mask = jnp.ones((N,)) if type == 'excitatory' else  jnp.zeros((N,))
        object.__setattr__(model, 'excitatory_mask', excitatory_mask) # override neuron types for test

        # Set a few weights to non-zero values
        recurrent_weights = jnp.zeros((N, N))
        recurrent_weights = recurrent_weights.at[0, 1].set(1)  # set w_01 = 1
        recurrent_weights = recurrent_weights.at[0, 2].set(2)  # set w_02 = 2
        recurrent_weights = recurrent_weights.at[1, 0].set(1)  # set w_N = 1
        object.__setattr__(model, 'recurrent_weights', recurrent_weights) # override weights for test
        
        # Set conductance to 1 where weights are non-zero
        conductances = jnp.zeros((N, N))
        conductances = conductances.at[0, 1].set(1)  # set g_01 = 1
        conductances = conductances.at[0, 2].set(4)  # set g_02 = 4
        conductances = conductances.at[1, 0].set(0.5)  # set g_10 = 0.5

        # Initial state with all neurons at resting potential
        initial_state = jnp.ones((N,)) * model.resting_potential, conductances

        # Get output
        drift = model.drift(0.0, initial_state, None)
        dv = drift[0]
        
        # Calculate expected quantal size
        # Quantal size is the size of the voltage jump for a single synaptic event with conductance 1 and weight 1
        reversal_potential = model.reversal_potential_E if type == 'excitatory' else model.reversal_potential_I
        quantal_size = 1 * 1 * (reversal_potential - model.resting_potential) / model.membrane_conductance

        assert dv[0] == 9 * quantal_size # 1*1 + 2*4 = 9
        assert dv[1] == 0.5 * quantal_size # 1*0.5 = 0.5
        assert jnp.all(dv[2:] == 0)

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
    """Test that the OUP converges to zero mean over time."""
    key = jr.PRNGKey(0)
    t0=0
    t1=10
    dt0=0.1

    dim = 3
    noise_model = OUP(theta=1, noise_scale=0, dim = dim) # No noise, should decay to zero
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = jnp.array([1.0, 2.0, -3.0]) # Set initial state away from zero

    sol = dfx.diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt0, y0=init_state, adjoint=dfx.ForwardMode(), max_steps=1000)
    t = sol.ts
    x = sol.ys
    assert jnp.all(jnp.abs(x[-1, :]) < 0.01)  # Should be close to zero

def test_OUP_zero_mean():
    """Test that the OUP with noise has roughly zero mean over long time."""
    key = jr.PRNGKey(0)
    t0=0
    t1=1000
    dt0=0.1

    dim = 3
    noise_model = OUP(theta=1, noise_scale=0.1, dim = dim) # With noise
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = jnp.array([0.0, 0.0, 0.0]) # Start at zero

    sol = dfx.diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt0, y0=init_state, adjoint=dfx.ForwardMode(), max_steps=None)
    t = sol.ts
    x = sol.ys
    mean = jnp.mean(x, axis=0)
    assert jnp.all(jnp.abs(mean) < 1)  # Mean should be close to zero over long time


def test_excitatory_noise_only_affects_voltage_correctly():
    N = 7
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, key=key)

    # Remove recurrent effects; isolate noise contribution
    object.__setattr__(model, "recurrent_weights", jnp.zeros((N, N)))

    state = _baseline_state(model)

    # Excitatory noise vector (distinct values to avoid accidental symmetry)
    noise_E = jnp.arange(N, dtype=state[0].dtype)
    noise_I = jnp.zeros((N,), dtype=state[0].dtype)

    args = {
        "excitatory_noise": lambda t, x, a: noise_E,
        "inhibitory_noise": lambda t, x, a: noise_I,
    }

    dv, dG = model.drift(0.0, state, args)

    expected = (noise_E * (model.reversal_potential_E - state[0])) / model.membrane_conductance

    assert jnp.allclose(dv, expected)
    assert jnp.all(dG == 0)


def test_inhibitory_noise_only_affects_voltage_correctly():
    N = 5
    key = jr.PRNGKey(1)
    model = NeuronModel(N_neurons=N, key=key)

    object.__setattr__(model, "recurrent_weights", jnp.zeros((N, N)))
    state = _baseline_state(model)

    noise_E = jnp.zeros((N,), dtype=state[0].dtype)
    noise_I = jnp.linspace(0.0, 1.0, N, dtype=state[0].dtype)

    args = {
        "excitatory_noise": lambda t, x, a: noise_E,
        "inhibitory_noise": lambda t, x, a: noise_I,
    }

    dv, dG = model.drift(0.0, state, args)

    expected = (noise_I * (model.reversal_potential_I - state[0])) / model.membrane_conductance

    assert jnp.allclose(dv, expected)
    assert jnp.all(dG == 0)


def test_both_noises_add_linearly():
    N = 6
    key = jr.PRNGKey(2)
    model = NeuronModel(N_neurons=N, key=key)
    object.__setattr__(model, "recurrent_weights", jnp.zeros((N, N)))
    state = _baseline_state(model)

    # Some distinct values to avoid accidental symmetry
    noise_E = jnp.linspace(0.1, 2.5, N, dtype=state[0].dtype)
    noise_I = jnp.linspace(0.0, 1.0, N, dtype=state[0].dtype)

    args = {
        "excitatory_noise": lambda t, x, a: noise_E,
        "inhibitory_noise": lambda t, x, a: noise_I,
    }

    dv, _ = model.drift(0.0, state, args)

    expected = (
        noise_I * (model.reversal_potential_I - state[0])
        + noise_E * (model.reversal_potential_E - state[0])
    ) / model.membrane_conductance

    assert jnp.allclose(dv, expected)


def test_NoisyNeuronModel_forwards_noise_into_network_drift():
    N = 5
    key = jr.PRNGKey(4)
    network = NeuronModel(N_neurons=N, key=key)
    # Remove recurrent effects
    object.__setattr__(network, "recurrent_weights", jnp.zeros((N, N)))

    # Create OU processes for E/I noise with simple dynamics
    noise_E = OUP(theta=1.0, noise_scale=0.5, dim=N)
    noise_I = OUP(theta=0.5, noise_scale=0.7, dim=N)

    model = NoisyNeuronModel(N_neurons=N, neuron_model=network, noise_I_model=noise_I, noise_E_model=noise_E)

    V, G = _baseline_state(network)
    noise_E_state = jnp.arange(N, dtype=V.dtype)
    noise_I_state = jnp.arange(N, dtype=V.dtype)[::-1]

    # x packs (network_state, noise_E_state, noise_I_state)
    x = ((V, G), noise_E_state, noise_I_state)

    (dV, dG), d_E, d_I = model.drift(0.0, x, args=None)


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
    network = NeuronModel(N_neurons=N, key=key)
    noise_E = OUP(theta=1.0, noise_scale=0.5, dim=N)
    noise_I = OUP(theta=1.0, noise_scale=0.5, dim=N)

    model = NoisyNeuronModel(N_neurons=N, neuron_model=network, noise_I_model=noise_I, noise_E_model=noise_E)
    V, G = _baseline_state(network)
    noise_E_state = jnp.arange(N, dtype=V.dtype)
    noise_I_state = jnp.arange(N, dtype=V.dtype)[::-1]
    x = ((V, G), noise_E_state, noise_I_state)
    (diff_V, diff_G), diff_E, diff_I = model.diffusion(0.0, x, args=None)

    # Network diffusion is zero
    assert jnp.allclose(diff_V, 0.0)
    assert jnp.allclose(diff_G, 0.0)
    # OU diffusions are identity * noise_scale
    assert jnp.allclose(diff_E, jnp.eye(N) * noise_E.noise_scale)
    assert jnp.allclose(diff_I, jnp.eye(N) * noise_I.noise_scale)
