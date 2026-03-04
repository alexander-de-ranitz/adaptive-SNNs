import inspect
import os
from dataclasses import dataclass, fields

import diffrax as dfx
import equinox as eqx
import jax.random as jr
import numpy as np
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models import (
    OUP,
    Agent,
    AgentEnvSystem,
    LIFNetwork,
    NoisyNetwork,
    NoisyNetworkState,
)
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.environments.base import EnvironmentABC
from adaptive_SNN.models.networks.base import NeuronModelABC
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.solver import simulate_noisy_SNN


@dataclass
class RateLearningSimulationConfig:
    model: NeuronModelABC

    # Time parameters
    t0: float = 0.0
    t1: float = 100.0
    dt: float = 1e-4
    warmup_time: float = 10.0  # Time before learning starts, in seconds
    save_at: SaveAt = dfx.SaveAt()

    # Model hyperparameters
    N_neurons = 1
    N_inputs = 2
    noise_level: float = 0.1
    lr: float = 1e-3
    initial_weight: float = 5.0
    balance: float = 1.75
    min_noise_std: float = 5e-9

    # Reward model
    reward_rate: float = 0.1
    reward_fn: str = "MSE"

    # Environment parameters
    environment_model: EnvironmentABC = SpikeRateEnvironment
    target_state: float = 10.0
    environment_kwargs: dict = eqx.field(default_factory=lambda: {})

    # Input parameters
    exc_rate: float = 5000.0
    inh_rate: float = 1250
    input_types = jnp.array([1, 0])  # 1 for excitatory, 0 for inhibitory
    weight_std: float = 0.0
    fully_connected_input: bool = True

    # Other
    key_seed: int = 0
    save_file: str = "results/rate_learning"
    save_results: bool = True
    network_output_fn: callable = (
        None  # Optional function to extract network output for reward calculation
    )

    def print_to_file(self):
        log_file = self.save_file + "_info.txt"
        with open(log_file, "w") as f:
            for field in fields(self):
                if field.name == "save_at":
                    continue  # skip printing save_at details for brevity
                if field.name == "network_output_fn":
                    f.write("Network output function:\n")
                    if self.network_output_fn is not None:
                        f.write(inspect.getsource(self.network_output_fn) + "\n")
                    else:
                        f.write("None\n")
                    continue
                f.write(f"{field.name}: {getattr(self, field.name)}\n")

            # print model details separately
            f.write(f"Model details:\n{self.model}\n")
            for field in fields(self.model):
                try:
                    f.write(f"  {field.name}: {getattr(self.model, field.name)}\n")
                except AttributeError:
                    f.write(f"  {field.name}: <unavailable>\n")


def run_rate_learning_simulation(config: RateLearningSimulationConfig):
    if os.path.exists(config.save_file + ".npz") and config.save_results:
        print(f"File {config.save_file}.npz already exists. Not running simulation.")
        return

    if config.save_results:
        config.print_to_file()

    key = jr.fold_in(jr.PRNGKey(0), config.key_seed)
    key, network_key, spike_key, simulation_key = jr.split(key, 4)

    # Set up models
    neuron_model = config.model(
        N_neurons=config.N_neurons,
        N_inputs=config.N_inputs,
        dt=config.dt,
        fully_connected_input=config.fully_connected_input,
        input_weight=config.initial_weight,
        input_types=config.input_types,
        weight_std=config.weight_std,
        key=network_key,
    )

    noise_model = OUP(tau=neuron_model.tau_E, dim=config.N_neurons)

    network = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
        min_noise_std=config.min_noise_std,
    )

    agent = Agent(
        neuron_model=network,
        reward_model=RewardModel(reward_rate=config.reward_rate),
    )

    model = AgentEnvSystem(
        agent=agent, environment=config.environment_model(**config.environment_kwargs)
    )
    solver = dfx.EulerHeun()
    init_state = model.initial

    rates = jnp.array([config.exc_rate, config.inh_rate])  # firing rate in Hz
    if config.reward_fn == "MSE":
        reward_fn = lambda t, x, args: jnp.squeeze(
            -jnp.square(x.environment_state - config.target_state)
        )
    else:
        raise ValueError(f"Unsupported reward function: {config.reward_fn}")

    args = {
        "get_learning_rate": lambda t, x, args: jnp.where(
            t < config.warmup_time, 0.0, config.lr
        ),
        "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
            agent_state.noisy_network.network_state.S[0]
        ),
        "reward_fn": reward_fn,
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.fold_in(spike_key, jnp.rint(t / config.dt)),
            rates * config.dt,
            shape=(config.N_neurons, config.N_inputs),
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([config.balance]),
        "noise_scale_hyperparam": config.noise_level,
    }

    sol = simulate_noisy_SNN(
        model,
        solver,
        config.t0,
        config.t1,
        config.dt,
        init_state,
        save_at=config.save_at,
        args=args,
        key=simulation_key,
    )

    os.makedirs(os.path.dirname(config.save_file), exist_ok=True)

    if config.save_results:
        np.savez(
            config.save_file,
            sol=sol,
        )

    return sol, model


def run_noisy_network_simulation(
    t0: float,
    t1: float,
    dt: float,
    save_at: dfx.SaveAt,
    output_file: str,
    balance: float,
    noise_level: float,
    key_seed: int,
    initial_weight: float,
):
    key = jr.fold_in(jr.PRNGKey(0), key_seed)

    N_neurons = 1
    N_inputs = 2

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    network_key = jr.PRNGKey(0)  # Doesn't actually matter since weights fixed

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt,
        fully_connected_input=True,
        input_weight=initial_weight,
        input_types=jnp.array([1, 0]),
        weight_std=0.0,
        key=network_key,
    )

    noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)

    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
        min_noise_std=5e-9,
    )

    solver = dfx.EulerHeun()
    init_state = model.initial

    key, spike_key, simulation_key = jr.split(key, 3)

    args = {
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.fold_in(spike_key, jnp.round(t / dt).astype(int)),
            rates * dt,
            shape=(N_neurons, N_inputs),
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([balance]),
        "noise_scale_hyperparam": noise_level,
    }

    sol = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt,
        init_state,
        save_at=save_at,
        args=args,
        key=simulation_key,
    )

    state: NoisyNetworkState = sol.ys

    np.savez(output_file, times=sol.ts, sol=state)
