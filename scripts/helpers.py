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
    NoisyNetwork,
)
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.environments.base import EnvironmentABC
from adaptive_SNN.models.networks.base import NeuronModelABC
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.solver import simulate_noisy_SNN


@dataclass
class SimulationConfig:
    model: NeuronModelABC

    # Time parameters
    t0: float = 0.0
    t1: float = 100.0
    dt: float = 1e-4
    warmup_time: float = 10.0  # Time before learning starts, in seconds
    save_at: SaveAt = dfx.SaveAt()

    # Model hyperparameters
    N_neurons: int = 1
    N_inputs: int = 2
    connection_prob: float = 0.1
    noise_level: float = 0.1
    lr: float = 1e-3
    initial_weight: float = 5.0
    balance: float = 1.75
    min_noise_std: float = 5e-9
    fraction_excitatory_recurrent: float = 0.8
    fraction_excitatory_input: float = 0.8
    initial_weight_matrix: jnp.ndarray | None = (
        None  # Optional initial weight matrix of shape (N_neurons, N_neurons + N_inputs)
    )

    # Reward model
    reward_rate: float = 0.1
    reward_fn: callable = None

    # Environment parameters
    environment_model: EnvironmentABC = SpikeRateEnvironment
    target_state: float = 10.0
    environment_kwargs: dict = eqx.field(default_factory=lambda: {})

    # Input parameters
    input_spike_fn: callable = None
    input_types: jnp.ndarray | None = None  # 1 for excitatory, 0 for inhibitory

    weight_std: float = 0.0
    fully_connected_input: bool = True

    # Other
    key_seed: int = 0
    save_file: str = "results/rate_learning"
    save_results: bool = True
    network_output_fn: callable = (
        None  # Function to extract network output for reward calculation
    )

    def print_to_file(self):
        log_file = self.save_file + "_info.txt"
        with open(log_file, "w") as f:
            for field in fields(self):
                if field.name == "save_at":
                    continue  # skip printing save_at details for brevity
                if field.name in ["network_output_fn", "input_spike_fn", "reward_fn"]:
                    f.write(f"{field.name}:\n")
                    if getattr(self, field.name) is not None:
                        f.write(inspect.getsource(getattr(self, field.name)) + "\n")
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


def run_simulation(config: SimulationConfig):
    if os.path.exists(config.save_file + ".npz") and config.save_results:
        print(f"File {config.save_file}.npz already exists. Not running simulation.")
        sol = np.load(config.save_file + ".npz", allow_pickle=True)["sol"].item()
        return sol, None

    os.makedirs(os.path.dirname(config.save_file), exist_ok=True)
    if config.save_results:
        config.print_to_file()

    key = jr.fold_in(jr.PRNGKey(0), config.key_seed)
    key, network_key, spike_key, simulation_key = jr.split(key, 4)

    # Set up models
    neuron_model = config.model(
        N_neurons=config.N_neurons,
        N_inputs=config.N_inputs,
        connection_prob=config.connection_prob,
        dt=config.dt,
        initial_weight_matrix=config.initial_weight_matrix,
        fully_connected_input=config.fully_connected_input,
        input_weight=config.initial_weight,
        input_types=config.input_types,
        fraction_excitatory_input=config.fraction_excitatory_input,
        fraction_excitatory_recurrent=config.fraction_excitatory_recurrent,
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

    args = {
        "get_learning_rate": lambda t, x, args: jnp.where(
            t < config.warmup_time, 0.0, config.lr
        ),
        "network_output_fn": config.network_output_fn,
        "reward_fn": config.reward_fn,
        "get_input_spikes": config.input_spike_fn,
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

    if config.save_results:
        np.savez(
            config.save_file,
            sol=sol,
        )

    return sol, model
