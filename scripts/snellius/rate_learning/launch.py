import shlex
import sys
from pathlib import Path

import numpy as np
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def create_jobs():
    jobs = Jobs()
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_path = base_dir / "scripts" / "snellius" / "rate_learning_job.py"

    initial_weights = np.linspace(0.0, 10.0, 5)
    noise_levels = np.linspace(0.0, 1.0, 10)
    learning_rates = []
    iterations = 1
    base_key = 0
    key_i = 0
    for iw in initial_weights:
        for nl in noise_levels:
            for lr in learning_rates:
                key_i += 1

                args = [
                    "--initial_weight",
                    str(iw),
                    "--min_noise_std",
                    str(nl),
                    "--learning_rate",
                    str(lr),
                    "--iterations",
                    str(iterations),
                    "--key_seed",
                    str(base_key + key_i),
                    "--output_file",
                    str(log_dir / f"result_iw{iw}_nl{nl}_lr{lr}"),
                ]

                job_name = f"rate_learning_iw{iw}_nl{nl}_lr{lr}"

                # Create a bash script to set PYTHONPATH and run the processing script
                bash_script = "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        f'export PYTHONPATH={shlex.quote(str(base_dir))}:"${{PYTHONPATH:-}}"',
                        f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))} {' '.join(shlex.quote(arg) for arg in args)}",
                    ]
                )

                jobs.add(
                    name=job_name,
                    script=bash_script,
                    stdout=str(log_dir / f"job.out.{job_name}"),
                    stderr=str(log_dir / f"job.err.{job_name}"),
                    wd=str(base_dir),
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
