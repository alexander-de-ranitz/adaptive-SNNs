import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def create_jobs():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    module_path = "scripts.snellius.network_tuning.run_network"

    parser = argparse.ArgumentParser(description="Launch jobs on Snellius")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/default_folder/",
        help="Directory to store output files",
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

    seed = 0
    num_iterations = 1
    balances = [0.5, 0.75, 1.0]
    # N_external_inputs = np.linspace(25,162.5,12, dtype=np.int32)
    N_external_inputs = [150]
    rec_weights = np.arange(0, 12, 2)
    for i in range(num_iterations):
        for b in balances:
            for w in rec_weights:
                for N_in in N_external_inputs:
                    id = f"N_{N_in}_b_{b:6f}_rw_{w:6f}"
                    name = id + f"seed_{seed}"

                    cmd_args = [
                        "--balance",
                        str(b),
                        "--N_external_inputs",
                        str(N_in),
                        "--key_seed",
                        str(seed),
                        "--rec_weight",
                        str(w),
                        "--output_file",
                        str(results_dir / name),
                    ]

                    # Create a bash script to set PYTHONPATH and run the processing script
                    bash_script = "\n".join(
                        [
                            "#!/usr/bin/env bash",
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
        seed += 12345
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
