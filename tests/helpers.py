import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import Solution
from jaxtyping import PyTree

from adaptive_SNN.models import (
    OUP,
    LIFNetwork,
    LIFState,
    NoisyNetwork,
    NoisyNetworkState,
)

# ============================================================================
# Model Creation Helpers
# ============================================================================


def make_LIF_model(
    N_neurons=10,
    N_inputs=3,
    dt=0.1e-3,
    input_neuron_types=None,
    fully_connected_input=True,
    input_weight=1.0,
    key=jr.PRNGKey(0),
):
    """Create a LIFNetwork with configurable parameters."""
    return LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt,
        input_neuron_types=input_neuron_types,
        fully_connected_input=fully_connected_input,
        input_weight=input_weight,
        key=key,
    )


def make_OUP_model(dim=3, tau=1.0, noise_scale=0.3, mean=0.0):
    """Create an Ornstein-Uhlenbeck Process model with configurable parameters."""
    return OUP(tau=tau, noise_scale=noise_scale, mean=mean, dim=dim)


def make_Noisy_LIF_model(
    N_neurons=10, N_inputs=3, noise_scale=0.0, tau=1.0, dt=0.1e-3, key=jr.PRNGKey(0)
):
    """Create a NoisyNetwork with LIF neurons and OU noise processes."""
    network = make_LIF_model(N_neurons, N_inputs, dt=dt, key=key)
    noise_model = OUP(tau=tau, noise_scale=noise_scale, dim=N_neurons)
    return NoisyNetwork(neuron_model=network, noise_model=noise_model)


# ============================================================================
# State Creation Helpers
# ============================================================================


def make_baseline_state(model: LIFNetwork, **overrides) -> LIFState:
    """Create a baseline LIFState with sensible defaults.

    Default state has:
    - V at resting potential
    - All spikes, weights, conductances at zero
    - time_since_last_spike at infinity
    - Empty spike buffer

    Args:
        model: LIFNetwork model to create state for
        **overrides: Dict of field names and values to override defaults
                     e.g., V=custom_voltages, W=custom_weights
    """
    N_neurons = model.N_neurons
    N_inputs = model.N_inputs

    state = LIFState(
        V=jnp.ones((N_neurons,)) * model.resting_potential,
        S=jnp.zeros((N_neurons + N_inputs,)),
        W=jnp.zeros((N_neurons, N_neurons + N_inputs)),
        G=jnp.zeros((N_neurons, N_neurons + N_inputs)),
        time_since_last_spike=jnp.ones((N_neurons,)) * jnp.inf,
        spike_buffer=jnp.zeros((model.buffer_size, N_neurons + N_inputs)),
        buffer_index=jnp.array(0, dtype=jnp.int32),
    )

    # Apply overrides using eqx.tree_at
    for field_name, value in overrides.items():
        state = eqx.tree_at(lambda s: getattr(s, field_name), state, value)

    return state


def make_noisy_state(network_state: LIFState, noise_state=None) -> NoisyNetworkState:
    """Create a NoisyNetworkState from a LIFState and optional noise state."""
    N_neurons = network_state.V.shape[0]
    if noise_state is None:
        noise_state = jnp.zeros((N_neurons,))
    return NoisyNetworkState(network_state, noise_state)


# ============================================================================
# Args Creation Helpers
# ============================================================================


def make_default_args(N_neurons, N_inputs, **overrides):
    """Create default args dict with sensible defaults.

    Args:
        N_neurons: Number of neurons
        N_inputs: Number of input neurons
        **overrides: Dict of arg names to override, e.g., RPE=1.5
    """
    args = {
        "excitatory_noise": jnp.zeros((N_neurons,)),
        "RPE": jnp.array([0.0]),
        "get_input_spikes": lambda t, x, a: jnp.zeros((N_inputs,)),
        "get_learning_rate": lambda t, x, a: jnp.array([0.0]),
        "get_desired_balance": lambda t, x, a: 0.0,
    }
    args.update(overrides)
    return args


# ============================================================================
# Other Helpers
# ============================================================================


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
        tau: float = 1.0,
        noise_scale: float = 1,
        dim: int = 1,
        t0: float = 0.0,
        t1: float = 1.0,
    ):
        super().__init__(tau=tau, noise_scale=noise_scale, dim=dim)
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
        noise_model,
        t0: float = 0.0,
        t1: float = 1.0,
    ):
        super().__init__(neuron_model, noise_model)
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
