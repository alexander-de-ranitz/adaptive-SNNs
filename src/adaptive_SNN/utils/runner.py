from pathlib import Path

import diffrax as dfx
import jax.random as jr
import numpy as np
from jax import numpy as jnp

from adaptive_SNN.models import AgentEnvSystem, NeuralNoiseOUP
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.config import SimulationConfig


def _load_existing_solution(save_file: str) -> tuple[dfx.Solution, AgentEnvSystem]:
    data = np.load(save_file, allow_pickle=True)
    return data["sol"].item(), data["model"].item()


def run_simulation(
    config: SimulationConfig,
    save_results: bool = True,
    overwrite: bool = False,
    load_if_exists: bool = True,
):
    """Run a simulation and optionally reuse or overwrite saved results.

    Behavior when a save file exists:
    - overwrite=False, load_if_exists=True: load and return saved solution.
    - overwrite=False, load_if_exists=False: raise FileExistsError.
    - overwrite=True: run simulation and replace stored result.
    """
    save_file = config.normalized_save_file()
    save_path = Path(save_file)

    if save_results and save_path.exists() and not overwrite:
        if load_if_exists:
            print(f"Loading existing result from {save_file}")
            return _load_existing_solution(save_file)
        raise FileExistsError(
            f"Result file already exists at {save_file}. "
            "Use overwrite=True to rerun or load_if_exists=True to load it."
        )

    if save_results:
        config.ensure_output_directory()
        config.print_to_file()

    if isinstance(config.key, int):
        key = jr.PRNGKey(config.key)
    else:
        key = config.key

    key, network_key, simulation_key = jr.split(key, 3)

    neuron_model = config.base_network_cls(
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
        mean_synaptic_delay=config.mean_synaptic_delay,
        key=network_key,
        **config.base_network_kwargs,
    )

    noise_model = NeuralNoiseOUP(tau=neuron_model.tau_E, dim=config.N_neurons)

    network = config.noisy_network_cls(
        neuron_model=neuron_model,
        noise_model=noise_model,
        min_noise_std=config.min_noise_std,
    )

    reward_noise = config.reward_noise_model(**config.reward_noise_kwargs)
    RPE_model = config.RPE_model(**config.RPE_model_kwargs)
    agent = config.agent_cls(
        neuron_model=network,
        reward_model=config.reward_model(**config.reward_kwargs),
        reward_noise=reward_noise,
        RPE_model=RPE_model,
    )

    model = config.agent_env_system_cls(
        agent=agent,
        environment=config.environment_model(**config.environment_kwargs),
    )

    solver = dfx.EulerHeun()
    init_state = model.initial

    args = {
        "get_learning_rate": lambda t, x, args: jnp.where(
            t < config.warmup_time,
            0.0,
            config.lr,
        ),
        "network_output_fn": config.network_output_fn,
        "reward_fn": config.reward_fn,
        "get_input_spikes": config.input_spike_fn,
        "get_desired_balance": lambda t, x, args: jnp.array([config.balance]),
        "noise_scale_hyperparam": config.noise_level,
        **config.args,
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

    if save_results:
        np.savez(save_file, sol=sol, model=model)

    return sol, model
