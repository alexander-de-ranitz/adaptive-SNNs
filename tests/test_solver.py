import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import Solution
from jaxtyping import PyTree

from adaptive_SNN.models.models import OUP, LIFNetwork, NoisyNetwork, NoisyNetworkState
from adaptive_SNN.utils.solver import simulate_noisy_SNN


def _default_args(N_neurons, N_inputs):
    return {
        "excitatory_noise": lambda t, x, a: jnp.zeros((N_neurons,)),
        "inhibitory_noise": lambda t, x, a: jnp.zeros((N_neurons,)),
        "RPE": lambda t, x, a: jnp.array([0.0]),
        "input_spikes": lambda t, x, a: jnp.zeros((N_inputs,)),
        "learning_rate": lambda t, x, a: jnp.array([0.0]),
        "desired_balance": lambda t, x, a: 0.0,  # = no balancing
    }


def get_non_inf_ts_ys(sol: Solution) -> tuple[PyTree, PyTree]:
    """Get all non-inf values of ts and ys from a diffrax Solution."""
    ts = sol.ts
    ys = sol.ys
    mask = ~jnp.isinf(ts)
    ts_clean = ts[mask]
    ys_clean = jax.tree.map(lambda y: y[mask], ys)
    return ts_clean, ys_clean


def allclose_pytree(x: PyTree, y: PyTree, atol=1e-6):
    """Check if two PyTrees are allclose."""
    return jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=atol), x, y))


class DeterministicOUP(OUP):
    """This class is identical to OUP, except it uses a VirtualBrownianTree for the noise terms.
    This makes the noise deterministic given the same key, which is useful for testing."""

    t0: float
    t1: float

    def __init__(
        self,
        theta: float = 1.0,
        noise_scale: float = 1,
        dim: int = 1,
        t0: float = 0.0,
        t1: float = 1.0,
    ):
        super().__init__(theta=theta, noise_scale=noise_scale, dim=dim)
        self.t0 = t0
        self.t1 = t1

    def terms(self, key):
        process_noise = dfx.VirtualBrownianTree(
            self.t0,
            self.t1,
            shape=self.noise_shape,
            key=key,
            levy_area=dfx.SpaceTimeLevyArea,
            tol=1e-3,
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )


class DeterministicNoisyNeuronModel(NoisyNetwork):
    """This class is identical to NoisyNeuronModel, except it uses a VirtualBrownianTree for the noise terms.
    This makes the noise deterministic given the same key, which is useful for testing."""

    t0: float
    t1: float

    def __init__(
        self,
        neuron_model,
        noise_I_model,
        noise_E_model,
        t0: float = 0.0,
        t1: float = 1.0,
    ):
        super().__init__(neuron_model, noise_I_model, noise_E_model)
        self.t0 = t0
        self.t1 = t1

    def terms(self, key):
        process_noise = dfx.VirtualBrownianTree(
            self.t0,
            self.t1,
            shape=self.noise_shape,
            key=key,
            levy_area=dfx.SpaceTimeLevyArea,
            tol=1e-3,
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )


def _make_noiseless_network(
    N_neurons: int, N_inputs: int, key: jr.PRNGKey
) -> NoisyNetwork:
    """Helper to build a NoisyNeuronModel with no recurrent coupling and no OU diffusion.

    This keeps the dynamics simple/predictable for testing the solver wrapper.
    """
    network = LIFNetwork(N_neurons=N_neurons, N_inputs=N_inputs, key=key)

    # OU processes with zero diffusion so their states remain constant (deterministic)
    noise_E = OUP(theta=1.0, noise_scale=0.0, dim=N_neurons)
    noise_I = OUP(theta=1.0, noise_scale=0.0, dim=N_neurons)

    return NoisyNetwork(
        neuron_model=network,
        noise_I_model=noise_I,
        noise_E_model=noise_E,
    )


def test_solver_timesteps():
    N_neurons = 4
    N_inputs = 0
    key = jr.PRNGKey(0)
    model = _make_noiseless_network(N_neurons, N_inputs, key)

    t0, t1, dt0 = 0.0, 1.0, 0.1

    # Prepare initial state from model
    y0 = model.initial
    solver = dfx.Euler()
    args = _default_args(N_neurons, N_inputs)

    # Our method
    save_every = 1
    sol_1 = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, y0, save_every_n_steps=save_every, args=args
    )
    sol_1_ts = sol_1.ts

    # Direct diffrax call for comparison
    terms = model.terms(jr.PRNGKey(0))
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
    sol_2 = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
    )

    # Remove any -inf timepoints from sol_2 (pre-allocated but not used)
    sol_2_ts, _ = get_non_inf_ts_ys(sol_2)

    assert sol_1_ts.shape == sol_2_ts.shape
    assert jnp.allclose(sol_1_ts, sol_2_ts)


def test_solver_output_noiseless():
    N_neurons = 4
    N_inputs = 0
    key = jr.PRNGKey(0)
    model = _make_noiseless_network(N_neurons, N_inputs, key)

    t0, t1, dt0 = 0.0, 1.0, 0.1

    # Prepare initial state from model
    y0 = model.initial
    solver = dfx.Euler()
    args = _default_args(N_neurons, N_inputs)

    # Our method
    save_every = 1
    sol_1 = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, y0, save_every_n_steps=save_every, args=args
    )

    # Direct diffrax call for comparison
    terms = model.terms(jr.PRNGKey(0))
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
    sol_2 = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        args=args,
        adjoint=dfx.ForwardMode(),
    )
    sol_2_ts, sol_2_ys = get_non_inf_ts_ys(sol_2)
    sol_1_state: NoisyNetworkState = sol_1.ys

    print(sol_1_state.network_state.V)
    assert jnp.allclose(sol_1.ts, sol_2_ts)
    assert allclose_pytree(sol_1.ys, sol_2_ys)


def test_solver_output_with_noise():
    """Tests that our custom solver function produces the same output as a direct diffrax call.
    Note that this only works when our model does not spike, this is not implemented in the diffrax solver."""

    N_neurons = 3
    N_inputs = 0
    key = jr.PRNGKey(0)
    t0, t1, dt0 = 0.0, 1.0, 0.01

    network = LIFNetwork(N_neurons=N_neurons, N_inputs=N_inputs, key=key)

    noise_E = DeterministicOUP(
        theta=1.0, noise_scale=1e-9, dim=N_neurons, t0=t0, t1=t1 + dt0
    )
    noise_I = DeterministicOUP(
        theta=1.0, noise_scale=1e-9, dim=N_neurons, t0=t0, t1=t1 + dt0
    )

    model = DeterministicNoisyNeuronModel(
        neuron_model=network,
        noise_I_model=noise_I,
        noise_E_model=noise_E,
        t0=t0,
        t1=t1 + dt0,
    )

    # Prepare initial state from model
    y0 = model.initial
    solver = dfx.Euler()
    args = _default_args(N_neurons, N_inputs)

    # Our method
    save_every = 1
    sol_1 = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, y0, save_every_n_steps=save_every, args=args
    )

    # Direct diffrax call for comparison
    terms = model.terms(jr.PRNGKey(0))
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
    sol_2 = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        args=args,
        adjoint=dfx.ForwardMode(),
    )

    sol_2_ts, sol_2_ys = get_non_inf_ts_ys(sol_2)

    assert jnp.allclose(sol_1.ts, sol_2_ts)
    assert allclose_pytree(sol_1.ys, sol_2_ys)
