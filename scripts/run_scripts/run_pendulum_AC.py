import time

import diffrax as dfx
from jax import numpy as jnp

from adaptive_snn.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_snn.utils.runner import run_simulation
from adaptive_snn.utils.save_helper import save_part_of_state


def main():
    start = time.time()
    config = create_pendulum_AC_config(N_neurons=1000)
    config.lr = 0.0
    config.t1 = 5.0
    config.save_at = dfx.SaveAt(
        ts=jnp.linspace(config.t0, config.t1, 1000),
        fn=lambda t, state, args: save_part_of_state(
            state,
            # environment_state=True,
            # agent_output=True,
            # reward_signal=True,
            # value=True,
            # RPE=True,
            # features=True,
            # weights=True,
            filtered_spike_trains=True,
        ),
    )
    config.save_file = "results/pendulum_results.npz"
    sol, model = run_simulation(config, save_results=False, return_final_state=True)
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
