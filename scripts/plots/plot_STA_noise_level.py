import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt

from adaptive_SNN.simulation_configs.single_synapse_config import (
    create_single_synapse_learning_config,
)
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state
from adaptive_SNN.visualization import plot_noise_STA


def plot_noise_level_STA():
    save_at = SaveAt(
        steps=True,
        fn=lambda t, x, args: save_part_of_state(
            x,
            S=True,
            V=True,
            perturbations=True,
            var_E_conductance=True,
            mean_E_conductance=True,
        ),
    )

    sols = []
    key = jr.PRNGKey(15105)
    min_noises = [1e-10, 1e-9, 10e-9]
    for i, noise_level in enumerate(min_noises):
        key = jr.fold_in(key, i)
        config = create_single_synapse_learning_config(
            key=key, initial_synapse_weight=0
        )
        config.min_noise_std = noise_level
        config.save_at = save_at
        config.t1 = 100

        sol, model = run_simulation(config, save_results=False)
        sols.append(sol)

    plot_noise_STA(
        sols,
        model.agent.network,
        noise_levels=min_noises,
        neurons_to_plot=jnp.array([0]),
    )


if __name__ == "__main__":
    plot_noise_level_STA()
