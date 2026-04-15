import argparse
import shlex
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager

try:
    from scripts.snellius.MWE.define_experiments import generate_experiment_configs_new
except ModuleNotFoundError:
    # Support running launch.py directly via file path on cluster nodes.
    from define_experiments import (
        generate_experiment_configs_new,
    )


def create_jobs():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    module_path = "scripts.snellius.MWE.MWE"

    parser = argparse.ArgumentParser(description="Launch MWEjobs on Snellius")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/default_folder/",
        help="Directory to store output files (overrides default)",
    )
    cmd_args = parser.parse_args()

    log_dir = (
        Path(cmd_args.output_dir) / "logs" if cmd_args.output_dir else base_dir / "logs"
    )
    results_dir = (
        Path(cmd_args.output_dir) / "results"
        if cmd_args.output_dir
        else base_dir / "results"
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    jobs = Jobs()

    configs = generate_experiment_configs_new()

    base_key = 100
    num_iterations = 12
    for i in range(num_iterations):
        key_i = i * 100  # Keep one shared seed per iteration, unique across iterations
        seed = base_key + key_i
        for cfg_idx, cfg in enumerate(configs):
            name = (
                f"results_cfg{cfg_idx:02d}_"
                f"lr{cfg.learning_rate:.6f}_"
                f"rnr{cfg.reward_noise_rate:.6f}_"
                f"rns{cfg.reward_noise_std:.6f}_"
                f"mns{cfg.min_noise_std:.6e}_"
                f"ir{cfg.input_rate}_"
                f"rdt{cfg.RPE_decay_tau:.6f}_"
                f"dv{cfg.delta_V:.6f}_"
                f"seed{seed}"
            )

            cmd_args = [
                "--noise_level",
                "0.0",
                "--learning_rate",
                str(cfg.learning_rate),
                "--min_noise_std",
                str(cfg.min_noise_std),
                "--reward_noise_rate",
                str(cfg.reward_noise_rate),
                "--reward_noise_std",
                str(cfg.reward_noise_std),
                "--input_rate",
                str(cfg.input_rate),
                "--RPE_decay_tau",
                str(cfg.RPE_decay_tau),
                "--delta_V",
                str(cfg.delta_V),
                "--key_seed",
                str(seed),
                "--output_file",
                str(results_dir / name),
            ]

            # Create a bash script to set PYTHONPATH and run the processing script
            bash_script = "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f'export PYTHONPATH={shlex.quote(str(base_dir))}:"${{PYTHONPATH:-}}"',
                    f"{shlex.quote(sys.executable)} -m {shlex.quote(module_path)} {' '.join(shlex.quote(arg) for arg in cmd_args)}",
                ]
            )

            jobs.add(
                name=name,
                script=bash_script,
                stdout=str(log_dir / f"job.out.{name}"),
                stderr=str(log_dir / f"job.err.{name}"),
                wd=str(base_dir),
                numCores=1,
                iteration=1,
            )

    return jobs


def main():
    jobs = create_jobs()
    manager = LocalManager()

    print(f"Submitting {len(jobs.jobs())} jobs to the local manager...")
    manager.submit(jobs)
    manager.wait4all()
    manager.finish()


if __name__ == "__main__":
    main()
