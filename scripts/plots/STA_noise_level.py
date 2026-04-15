import sys
from pathlib import Path

# Allow running this file directly while importing from the repository modules.
_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS_FILE.parents[1]
_REPO_ROOT = _THIS_FILE.parents[2]
for _path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt

from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state
from adaptive_SNN.visualization import plot_noise_STA
from scripts.simulation_configs.single_neuron_simulation import (
    create_single_neuron_config_extra_synapse,
)


def plot_noise_level_STA():
    config = create_single_neuron_config_extra_synapse(N_neurons=1)
    config.t1 = 20

    config.save_at = SaveAt(
        steps=True,
        fn=lambda t, x, args: save_part_of_state(
            x,
            S=True,
            V=True,
            noise_state=True,
            var_E_conductance=True,
            mean_E_conductance=True,
        ),
    )
    config.args["use_noise"] = jnp.array([True])

    sols = []
    key = jr.PRNGKey(15105)
    noise_levels = [0.01, 0.1, 0.5]
    for i, noise_level in enumerate(noise_levels):
        config.noise_level = noise_level
        key = jr.fold_in(key, i)
        config.key = key

        sol, model = run_simulation(config, save_results=False)
        sols.append(sol)

    plot_noise_STA(sols, model.agent.noisy_network, noise_levels=noise_levels)


if __name__ == "__main__":
    plot_noise_level_STA()
