import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_snn.models.environments import SingleSynapseLearningEnv
from adaptive_snn.models.networks import AbstractNeuronModel, GatedLIFNetwork
from adaptive_snn.utils.config import SimulationConfig

# Poisson input rates (Hz): background E input, background I input, and the input to the synapse to be learned
RATES = jnp.array([4990.0, 1250.0, 10.0])
I_TO_E_RATIO = 8.0  # Background I weight relative to the background E weight
DT = 1e-4


def create_single_synapse_config(
    key: Array,
    *,
    network_cls: type[AbstractNeuronModel] = GatedLIFNetwork,
    N_neurons: int | None = None,
    twin: bool = False,
    background_weight: float = 1.0,
    learned_weight: float = 0.0,
    noise_floor: float = 1e-9,
    balance: float = jnp.nan,
    balance_rate: float | None = None,
    t1: float = 100.0,
) -> SimulationConfig:
    """Unconnected neurons that all receive the same three Poisson inputs: background E and I inputs, and a low-rate
    input to the synapse to be learned. Weights are frozen (lr = 0) and there is no reward prediction (RPE = reward).

    Every spike of the network output increments the reward, which decays exponentially. The input spike trains are
    drawn from args["input_key"], such that batched runs can each receive their own inputs.

    Args:
        key: PRNG key, split into the simulation key and the input key.
        network_cls: Network class to simulate.
        N_neurons: Number of neurons (default 1). Not to be set when twin=True.
        twin: If True, simulate two neurons with identical inputs, weights and initial state, of which only neuron 0
            is perturbed. The network output (and thus the reward) is the spike train of neuron 0 minus that of its
            unperturbed twin, such that the RPE equals the causal effect of the perturbations on the reward.
            Balancing should then remain disabled, as it would make the twin drift away from neuron 0.
        background_weight: Weight of the background E synapse. The background I weight is I_TO_E_RATIO times larger.
        learned_weight: Weight of the synapse to be learned.
        noise_floor: Std of the perturbations (the noise is not scaled with the input variance).
        balance: Target E/I balance; NaN disables balancing.
        balance_rate: Rate at which the I weights are adjusted to reach the target balance (None: network default).
        t1: End time of the simulation.
    """
    if twin and N_neurons not in (None, 2):
        raise ValueError(
            f"twin=True simulates exactly 2 neurons, got N_neurons={N_neurons}"
        )
    N_neurons = 2 if twin else (1 if N_neurons is None else N_neurons)
    N_inputs = len(RATES)
    key, input_key = jr.split(key)

    def input_spike_fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint(t / DT), dtype=jnp.int64)
        spikes = jr.poisson(
            jr.fold_in(args["input_key"], step_idx), RATES * DT, shape=(1, N_inputs)
        )
        return jnp.tile(spikes, (N_neurons, 1))

    if twin:
        network_output_fn = lambda t, agent_state, args, env_state: jnp.atleast_1d(
            agent_state.network_state.S[0] - agent_state.network_state.S[1]
        )
        use_noise = jnp.array([True, False])
    else:
        network_output_fn = lambda t, agent_state, args, env_state: jnp.atleast_1d(
            agent_state.network_state.S[0]
        )
        use_noise = jnp.array([True] * N_neurons)

    return SimulationConfig(
        network_cls=network_cls,
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        input_types=jnp.array([1, 0, 1]),
        initial_weight_matrix=jnp.tile(
            jnp.array(
                [jnp.nan] * N_neurons
                + [background_weight, I_TO_E_RATIO * background_weight, learned_weight]
            ),
            (N_neurons, 1),
        ),
        mean_synaptic_delay=0.0,
        t0=0.0,
        t1=t1,
        dt=DT,
        actor_warmup_time=0.0,
        lr=0.0,
        balance=balance,
        balance_rate=balance_rate,
        noise_floor=noise_floor,
        noise_scaling_factor=0.0,
        use_noise=use_noise,
        key=key,
        save_at=SaveAt(t1=True),
        network_output_fn=network_output_fn,
        input_spike_fn=input_spike_fn,
        additional_args={"input_key": input_key},
        environment_model=SingleSynapseLearningEnv,
        environment_kwargs={"tau_reward": 0.01, "reward_dim": 1},
        reward_fn=lambda t, x, args: x.environment_state.reward,
        RPE_fn=lambda t, x, args, reward: reward,
    )
