import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.snellius._templates.launch import build_jobs, parse_output_dir, submit

MODULE_PATH = "scripts.snellius.single_synapse_learning.run"


def param_grid():
    # Every job simulates n_trials trials in parallel, which serve as each other's external noise.
    # Independent jobs (seeds) act as independent networks.
    seed = 1234
    for i in range(24):
        seed += 4321
        for noise_level in [1e-9]:
            for w in [1.0]:
                name = (
                    "batched_"
                    + f"noise_{(noise_level * 1e9):.3f}_"
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
