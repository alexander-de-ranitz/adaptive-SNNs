import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import jax

from adaptive_SNN.utils.solver import run_SNN_simulation
from adaptive_SNN.models.models import NeuronModel, NoisyNeuronModel, OUP

class DeterministicOUP(OUP):
    """This class is identical to OUP, except it uses a VirtualBrownianTree for the noise terms.
      This makes the noise deterministic given the same key, which is useful for testing."""
    t0 : float
    t1 : float

    def __init__(self, theta: float = 1.0, noise_scale: float = 1, dim: int = 1, t0: float = 0.0, t1: float = 1.0):
        super().__init__(theta=theta, noise_scale=noise_scale, dim=dim)
        self.t0 = t0
        self.t1 = t1

    def terms(self, key):
        process_noise = dfx.VirtualBrownianTree(self.t0, self.t1, shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea, tol=1e-3)
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise))

class DeterministicNoisyNeuronModel(NoisyNeuronModel):
    """This class is identical to NoisyNeuronModel, except it uses a VirtualBrownianTree for the noise terms.
      This makes the noise deterministic given the same key, which is useful for testing."""
    t0: float
    t1: float

    def __init__(self, N_neurons: int, neuron_model, noise_I_model, noise_E_model, t0: float = 0.0, t1: float = 1.0):
        super().__init__(N_neurons, neuron_model, noise_I_model, noise_E_model)
        self.t0 = t0
        self.t1 = t1

    def terms(self, key):
        process_noise = dfx.VirtualBrownianTree(self.t0, self.t1, shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea, tol=1e-3)
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise))
    
def _make_quiet_model(N_neurons: int, N_inputs: int, key: jr.PRNGKey) -> NoisyNeuronModel:
	"""Helper to build a NoisyNeuronModel with no recurrent coupling and no OU diffusion.

	This keeps the dynamics simple/predictable for testing the solver wrapper.
	"""
	network = NeuronModel(N_neurons=N_neurons, N_inputs=N_inputs, key=key)
	# Remove recurrent effects so G-noise does not affect V
	object.__setattr__(network, "weights", jnp.zeros((N_neurons, N_neurons + network.N_inputs)))

	# OU processes with zero diffusion so their states remain constant (deterministic)
	noise_E = OUP(theta=1.0, noise_scale=0.0, dim=N_neurons)
	noise_I = OUP(theta=1.0, noise_scale=0.0, dim=N_neurons)

	return NoisyNeuronModel(N_neurons=N_neurons, neuron_model=network, noise_I_model=noise_I, noise_E_model=noise_E)


def test_solver_timesteps():
	N_neurons = 4
	N_inputs = 2
	key = jr.PRNGKey(0)
	model = _make_quiet_model(N_neurons, N_inputs, key)

	t0, t1, dt0 = 0.0, 1.0, 0.1

	# Prepare initial state from model
	y0 = model.initial
	solver = dfx.Euler()

	# Our method
	save_every = 1
	sol_1, _ = run_SNN_simulation(model, solver, t0, t1, dt0, y0, save_every_n_steps=save_every, args=None)
	sol_1_ts = sol_1.ts

	# Direct diffrax call for comparison
	terms = model.terms(jr.PRNGKey(0))
	saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
	sol_2 = dfx.diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=saveat, adjoint=dfx.ForwardMode())
	
	# Remove any -inf timepoints from sol_2 (pre-allocated but not used)
	sol_2_ts = sol_2.ts
	mask = ~jnp.isinf(sol_2_ts)
	sol_2_ts = sol_2_ts[mask]
	
	assert sol_1_ts.shape == sol_2_ts.shape
	assert jnp.allclose(sol_1_ts, sol_2_ts)

def test_solver_output_noiseless():
	N_neurons = 4
	N_inputs = 2
	key = jr.PRNGKey(0)
	model = _make_quiet_model(N_neurons, N_inputs, key)
      
	t0, t1, dt0 = 0.0, 1.0, 0.1

	# Prepare initial state from model
	y0 = model.initial
	solver = dfx.EulerHeun()

	# Our method
	save_every = 1
	sol_1, spikes = run_SNN_simulation(model, solver, t0, t1, dt0, y0, save_every_n_steps=save_every, args=None)

	# Direct diffrax call for comparison
	terms = model.terms(jr.PRNGKey(0))
	saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
	sol_2 = dfx.diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=saveat, adjoint=dfx.ForwardMode())
	
	(V, G), noise_E, noise_I = sol_1.ys
	(V2, G2), noise_E2, noise_I2 = sol_2.ys

	# Remove any -inf timepoints from sol_2 (pre-allocated but not used)
	V2 = V2[~jnp.isinf(sol_2.ts)]
	G2 = G2[~jnp.isinf(sol_2.ts)]
	noise_E2 = noise_E2[~jnp.isinf(sol_2.ts)]
	noise_I2 = noise_I2[~jnp.isinf(sol_2.ts)]

	# Check shapes
	assert V.shape == V2.shape == (len(sol_1.ts), N_neurons)
	assert G.shape == G2.shape == (len(sol_1.ts), N_neurons, N_neurons + N_inputs)
	assert noise_E.shape == noise_E2.shape == (len(sol_1.ts), N_neurons)
	assert noise_I.shape == noise_I2.shape == (len(sol_1.ts), N_neurons)
	
	assert spikes.shape == (len(sol_1.ts), N_neurons)
	assert jnp.allclose(spikes, 0.0)

	# Check values are close
	assert jnp.allclose(V, V2)
	assert jnp.allclose(G, G2)
	assert jnp.allclose(noise_E, noise_E2)
	assert jnp.allclose(noise_I, noise_I2)


def test_solver_output_with_noise():
	"""Tests that our custom solver function produces the same output as a direct diffrax call.
	Note that this only works when our model does not spike, this is not implemented in the diffrax solver."""

	N_neurons = 3
	N_inputs = 2
	key = jr.PRNGKey(0)
	t0, t1, dt0 = 0.0, 1.0, 0.1

	network = NeuronModel(N_neurons=N_neurons, N_inputs=N_inputs, key=key)

	noise_E = DeterministicOUP(theta=1.0, noise_scale=1.0, dim=N_neurons, t0=t0, t1=t1+1e2)
	noise_I = DeterministicOUP(theta=1.0, noise_scale=1.0, dim=N_neurons, t0=t0, t1=t1+1e2)

	model = DeterministicNoisyNeuronModel(N_neurons=N_neurons, neuron_model=network, noise_I_model=noise_I, noise_E_model=noise_E, t0=t0, t1=t1+1e2)


	# Prepare initial state from model
	y0 = model.initial
	solver = dfx.Euler()

	# Our method
	save_every = 1
	sol_1, spikes = run_SNN_simulation(model, solver, t0, t1, dt0, y0, save_every_n_steps=save_every, args=None)

	# Direct diffrax call for comparison
	terms = model.terms(jr.PRNGKey(0))
	saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
	sol_2 = dfx.diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=saveat, adjoint=dfx.ForwardMode())
	
	(V, G), noise_E, noise_I = sol_1.ys
	(V2, G2), noise_E2, noise_I2 = sol_2.ys

	# Remove any -inf timepoints from sol_2 (pre-allocated but not used)
	V2 = V2[~jnp.isinf(sol_2.ts)]
	G2 = G2[~jnp.isinf(sol_2.ts)]
	noise_E2 = noise_E2[~jnp.isinf(sol_2.ts)]
	noise_I2 = noise_I2[~jnp.isinf(sol_2.ts)]

	# Check shapes
	assert V.shape == V2.shape == (len(sol_1.ts), N_neurons)
	assert G.shape == G2.shape == (len(sol_1.ts), N_neurons, N_neurons + N_inputs)
	assert noise_E.shape == noise_E2.shape == (len(sol_1.ts), N_neurons)
	assert noise_I.shape == noise_I2.shape == (len(sol_1.ts), N_neurons)
	assert spikes.shape == (len(sol_1.ts), N_neurons)
      
	# Check that we have some noise
	assert not jnp.allclose(noise_E, 0.0)
	assert not jnp.allclose(noise_I, 0.0)

	# Check values are close
	assert jnp.allclose(spikes, 0.0) # No spikes should occur- dfx.diffeqsolve does not deal with spikes
	assert jnp.allclose(V, V2)
	assert jnp.allclose(G, G2)
	assert jnp.allclose(noise_E, noise_E2)
	assert jnp.allclose(noise_I, noise_I2)
