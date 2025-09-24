from adaptive_SNN.models.models import NeuronModel, OUP
import jax.random as jr
import jax.numpy as jnp
import diffrax as dfx

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
    assert initial_state[2].shape == (N,)

def test_drift_diffusion_shapes():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)
    initial_state = model.initial

    drift = model.drift(0.0, initial_state, None)
    diffusion = model.diffusion(0.0, initial_state, None)

    assert drift[0].shape == (N,)
    assert drift[1].shape == (N, N)
    assert drift[2].shape == (N,)

    assert diffusion[0].shape == (N,)
    assert diffusion[1].shape == (N, N)
    assert diffusion[2].shape == (N,)

def test_noise_shape():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)
    noise_shape = model.noise_shape

    assert noise_shape[0].shape == (N,)
    assert noise_shape[1].shape == (N, N)
    assert noise_shape[2].shape == (N,)

def test_drift_voltage_output():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)
    # Manually set initial state for predictable output
    initial_state = jnp.zeros((N,)), jnp.zeros((N, N)), jnp.zeros((N,))
    drift = model.drift(0.0, initial_state, None)
    dv = drift[0]
    expected_dv = 1/model.membrane_conductance * -model.leak_conductance * (initial_state[0] - model.resting_potential)

    assert jnp.allclose(dv, expected_dv)

def test_drift_conductance_output():
    N = 10
    key = jr.PRNGKey(0)
    model = NeuronModel(N_neurons=N, num_inputs=3, key=key)
    # Manually set initial state for predictable output
    initial_state = jnp.zeros((N,)), jnp.ones((N, N)), jnp.zeros((N,))
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
        initial_state = jnp.ones((N,)) * model.resting_potential, conductances, jnp.zeros((N,))

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