import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.snellius._templates.launch import build_jobs, parse_output_dir, submit

MODULE_PATH = "scripts.snellius.single_synapse_learning.extended_firing_statistics.run"


def param_grid():
    seed = 1234 * 5432
    for i in range(1):
        seed += 123 * 321
        for noise_level in [1e-9, 5e-9, 25e-9]:
            for balance in [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05]:
                for w in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0]:
                    name = (
                        f"noise_{(noise_level * 1e9):.3f}_"
                        + f"w_{w:.4f}_"
                        + f"b_{balance:.5f}_"
                        + f"seed_{seed}_iter_{i}"
                    )
                    yield (
                        name,
                        {
                            "--key_seed": seed,
                            "--noise_level": noise_level,
                            "--balance": balance,
                            "--initial_weight": w,
                        },
                    )


def main():
    log_dir, results_dir = parse_output_dir()
    jobs = build_jobs(MODULE_PATH, param_grid(), log_dir, results_dir)
    submit(jobs)


if __name__ == "__main__":
    main()
