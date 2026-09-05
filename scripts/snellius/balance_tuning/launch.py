import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.snellius.templates.launch import build_jobs, parse_output_dir, submit

MODULE_PATH = "scripts.snellius.balance_tuning.run"


def param_grid():
    seed = 9876
    for i in range(1):
        seed += 543
        for w in jnp.linspace(0.0, 2, 21):
            for b in jnp.linspace(0.0, 0.05, 21):
                name = f"w_{w:.4f}_b_{b:.4f}_seed_{seed}_iter_{i}"
                yield (
                    name,
                    {
                        "--initial_weight": w,
                        "--balance": b,
                        "--key_seed": seed,
                    },
                )


def main():
    log_dir, results_dir = parse_output_dir()
    jobs = build_jobs(MODULE_PATH, param_grid(), log_dir, results_dir)
    submit(jobs)


if __name__ == "__main__":
    main()
