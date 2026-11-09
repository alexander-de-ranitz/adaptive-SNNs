import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.snellius._templates.launch import build_jobs, parse_output_dir, submit

# Same experiment as single_synapse_learning, over a wider range of perturbation sizes and synaptic weights
MODULE_PATH = "scripts.snellius.single_synapse_learning.run"
N_SEEDS = 4  # Independent networks (of n_trials parallel trials each) per parameter combination


def param_grid():
    seed = 1234 * 5432
    for i in range(N_SEEDS):
        seed += 123 * 321
        for noise_level in [1e-9, 5e-9, 25e-9]:
            for w in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0]:
                name = (
                    f"noise_{(noise_level * 1e9):.3f}_"
                    + f"w_{w:.4f}_"
                    + f"seed_{seed}_iter_{i}"
                )
                yield (
                    name,
                    {
                        "--key_seed": seed,
                        "--noise_level": noise_level,
                        "--initial_weight": w,
                        "--n_trials": 1000,
                    },
                )


def main():
    log_dir, results_dir = parse_output_dir()
    jobs = build_jobs(MODULE_PATH, param_grid(), log_dir, results_dir)
    submit(jobs)


if __name__ == "__main__":
    main()
