import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.snellius.templates.launch import build_jobs, parse_output_dir, submit

MODULE_PATH = "scripts.snellius.single_synapse_learning.run"

DELTA_VS = [
    0.0,
    0.5**5,
    0.5**7,
    0.5**9,
    0.5**11,
    0.5**13,
    0.5**15,
    0.5**17,
    0.5**19,
    0.5**21,
]


def param_grid():
    seed = 1234
    for i in range(10):
        seed += 129
        for delta_v in DELTA_VS:
            for noise_level in [1e-9]:
                for balance in [0.001]:
                    for w in [1.0]:
                        dv = (
                            f"dv_{delta_v}_" if type(delta_v) is float else "no_gating_"
                        )
                        name = (
                            dv
                            + f"noise_{(noise_level * 1e9):.3f}_"
                            + f"w_{w:.4f}_"
                            + f"b_{balance:.5f}_"
                            + f"seed_{seed}_iter_{i}"
                        )
                        yield (
                            name,
                            {
                                "--delta_V": delta_v,
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
