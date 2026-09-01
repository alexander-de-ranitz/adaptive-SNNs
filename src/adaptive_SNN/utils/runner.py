import os

import jax
from jaxtyping import PyTree

jax.config.update("jax_enable_x64", True)
from pathlib import Path

import diffrax as dfx
import jax.random as jr
import numpy as np
from jax import numpy as jnp

from adaptive_SNN.models import AgentEnvSystem
from adaptive_SNN.solver import solve_ODE, solve_ODE_batched
from adaptive_SNN.utils.config import SimulationConfig


def _serialize_pytree(tree, downcast_to_float32: bool = True):
    leaves, treedef = jax.tree.flatten(tree)
    # Downcast float64 arrays to float32 to save space, if specified
    leaves = [
        (
            lambda arr: arr.astype(np.float32)
            if arr.dtype.kind == "f" and arr.itemsize > 4 and downcast_to_float32
            else arr
        )(np.asarray(jax.device_get(leaf)))
        for leaf in leaves
    ]

    # Since our pytrees can contain arrays of different shapes and sizes, we can't stack them into a single array.
    # instead, we store it as an object array
    # np tries to be smart and convert it to a regular array but this fails, so we manually create an object array and fill it
    leaves_array = np.empty(len(leaves), dtype=object)
    leaves_array[:] = leaves
    treedef_array = np.empty(1, dtype=object)
    treedef_array[0] = treedef
    return leaves_array, treedef_array


def _deserialize_pytree(leaves_array: np.ndarray, treedef_array: np.ndarray):
    leaves = list(leaves_array)
    treedef = treedef_array.item()
    return jax.tree.unflatten(treedef, leaves)


def _load_existing_solution(save_file: str) -> tuple[dfx.Solution, AgentEnvSystem]:
    data = np.load(save_file, allow_pickle=True)

    # For backward compatibility
    if "sol" in data:
        sol = data["sol"].item()
        model = data["model"].item() if "model" in data else None
        return sol, model

    # Load data and reconstruct the solution object
    ys = _deserialize_pytree(data["ys"], data["ys_tree_def"])
    ts = _deserialize_pytree(data["ts"], data["ts_tree_def"])
    sol = dfx.Solution(
        ys=ys,
        ts=ts,
        t0=ts[0],
        t1=ts[-1],
        interpolation=None,
        stats=None,
        result=dfx.RESULTS.successful,
        solver_state=None,
        controller_state=None,
        made_jump=False,
        event_mask=None,
    )

    # Load the model if it was saved, otherwise return None
    if "model" in data:
        model = data["model"].item()
    else:
        model = None

    return sol, model


def setup_simulation(
    config: SimulationConfig,
) -> tuple[AgentEnvSystem, PyTree, jr.PRNGKey]:
    """Set up the model, initial state, and key for a simulation based on the config.

    returns:
        model: The AgentEnvSystem model to be simulated.
        args: A dictionary of arguments to be passed to the model during simulation.
        simulation_key: A JAX PRNGKey for random number generation during the simulation."""
    if isinstance(config.key, int):
        key = jr.PRNGKey(config.key)
    else:
        key = config.key

    key, network_key, simulation_key = jr.split(key, 3)

    neuron_model = config.network_cls(
        N_neurons=config.N_neurons,
        N_inputs=config.N_inputs,
        connection_prob_E=config.connection_prob_E,
        connection_prob_I=config.connection_prob_I,
        dt=config.dt,
        initial_weight_matrix=config.initial_weight_matrix,
        initial_input_weight=config.initial_input_weight,
        initial_rec_weight=config.initial_rec_weight,
        fully_connected_input=config.fully_connected_input,
        input_types=config.input_types,
        fraction_excitatory_input=config.fraction_excitatory_input,
        fraction_excitatory_recurrent=config.fraction_excitatory_recurrent,
        rec_weight_std=config.rec_weight_std,
        mean_synaptic_delay=config.mean_synaptic_delay,
        min_noise_std=config.min_noise_std,
        input_weight_std=config.input_weight_std,
        key=network_key,
        **config.base_network_kwargs,
    )

    agent = config.agent_cls(
        neuron_model=neuron_model,
        reward_prediction_model=config.reward_prediction_model(
            **config.reward_predictor_kwargs
        ),
    )

    model = config.agent_env_system_cls(
        agent=agent,
        environment=config.environment_model(**config.environment_kwargs),
        agent_output_shape=config.network_output_shape,
    )

    args = {
        "get_learning_rate": lambda t, x, args: jnp.where(
            t < args["actor_warmup_time"], 0.0, args["lr"]
        ),
        "get_critic_lr": lambda t, x, args: jnp.where(
            t < args["critic_warmup_time"], 0.0, args["critic_lr"]
        ),
        "network_output_fn": config.network_output_fn,
        "reward_fn": config.reward_fn,
        "input_spike_fn": config.input_spike_fn,
        "get_desired_balance": lambda t, x, args: jnp.array([args["balance"]]),
        "lr": jnp.asarray(config.lr),
        "critic_lr": jnp.asarray(config.critic_lr),
        "actor_warmup_time": jnp.asarray(config.actor_warmup_time),
        "critic_warmup_time": jnp.asarray(config.critic_warmup_time),
        "balance": jnp.asarray(config.balance),
        "noise_scale_hyperparam": jnp.asarray(config.noise_level),
        **config.args,
    }

    return model, args, simulation_key


def run_simulation(
    config: SimulationConfig,
    save_results: bool = True,
    overwrite: bool = False,
    load_if_exists: bool = True,
    save_model: bool = False,
    return_final_state: bool = False,
    downcast_to_float32: bool = True,
    y0: PyTree | None = None,
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

    model, args, simulation_key = setup_simulation(config)
    if y0 is None:
        y0 = model.initial
    else:
        # Due to stale metadata in the saved state,
        # we need to reconstruct the pytree structure
        # of y0 to match the model's initial state.
        # Otherwise, we get :
        # "TypeError: cond branch outputs must have the same pytree structure, but they differ"
        y0 = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.initial), jax.tree_util.tree_leaves(y0)
        )

    solver = dfx.EulerHeun()

    sol = solve_ODE(
        model,
        solver,
        config.t0,
        config.t1,
        config.dt,
        y0,
        save_at=config.save_at,
        args=args,
        return_final_state=return_final_state,
        key=simulation_key,
    )

    if save_results:
        ys_values, ys_tree_def = _serialize_pytree(
            sol.ys, downcast_to_float32=downcast_to_float32
        )
        ts_values, ts_tree_def = _serialize_pytree(
            sol.ts, downcast_to_float32=downcast_to_float32
        )
        if save_model:
            np.savez(
                save_file,
                ys=ys_values,
                ys_tree_def=ys_tree_def,
                ts=ts_values,
                ts_tree_def=ts_tree_def,
                model=model,
            )
        else:
            np.savez(
                save_file,
                ys=ys_values,
                ys_tree_def=ys_tree_def,
                ts=ts_values,
                ts_tree_def=ts_tree_def,
            )

    return sol, model


def run_batched_simulation(
    configs: list[SimulationConfig],
    save_results: bool = True,
    overwrite: bool = False,
    load_if_exists: bool = True,
    save_model: bool = False,
    return_final_state: bool = False,
    downcast_to_float32: bool = True,
    y0s: list[PyTree] | None = None,
):
    """Run multiple simulations in parallel based on a list of configs.

    All configs use the same args, taken from the first config. #TODO: Can we relax this?
    """
    assert all(
        config.t0 == configs[0].t0
        and config.t1 == configs[0].t1
        and config.dt == configs[0].dt
        for config in configs
    ), "All configs must have the same time parameters for batched simulation."

    models, args_list, keys = zip(*(setup_simulation(config) for config in configs))
    if y0s is None:
        y0s = [model.initial for model in models]
    else:
        # Due to stale metadata in the saved state,
        # we need to reconstruct the pytree structure
        # of each y0 to match its model's initial state.
        # Otherwise, we get :
        # "TypeError: cond branch outputs must have the same pytree structure, but they differ"
        y0s = [
            jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(model.initial),
                jax.tree_util.tree_leaves(y0),
            )
            for model, y0 in zip(models, y0s)
        ]

    solver = dfx.EulerHeun()
    sols = solve_ODE_batched(
        models=models,
        solver=solver,
        t0=configs[0].t0,
        t1=configs[0].t1,
        dt0=configs[0].dt,
        y0s=y0s,
        save_at=configs[0].save_at,
        args=args_list,
        return_final_state=return_final_state,
        keys=keys,
    )

    if save_results:
        for i, config in enumerate(configs):
            os.makedirs(Path(config.save_file).parent, exist_ok=True)
            config.print_to_file()
            ys_i = jax.tree.map(lambda arr: arr[i], sols.ys)
            ts_i = sols.ts[i]
            ys_values, ys_tree_def = _serialize_pytree(
                ys_i, downcast_to_float32=downcast_to_float32
            )
            ts_values, ts_tree_def = _serialize_pytree(
                ts_i, downcast_to_float32=downcast_to_float32
            )
            if save_model:
                np.savez(
                    config.save_file,
                    ys=ys_values,
                    ys_tree_def=ys_tree_def,
                    ts=ts_values,
                    ts_tree_def=ts_tree_def,
                    model=models[i],
                )
            else:
                np.savez(
                    config.save_file,
                    ys=ys_values,
                    ys_tree_def=ys_tree_def,
                    ts=ts_values,
                    ts_tree_def=ts_tree_def,
                )
    return sols, models
