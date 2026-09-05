import argparse
import shlex
from pathlib import Path

from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def create_jobs():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    module_path = "scripts.snellius.pendulum_AC_learned_I.run"

    parser = argparse.ArgumentParser(description="Launch jobs on Snellius")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/default_folder/",
        help="Directory to store output files",
    )
    parsed_args = parser.parse_args()

    log_dir = (
        Path(parsed_args.output_dir) / "logs"
        if parsed_args.output_dir
        else base_dir / "logs"
    )
    results_dir = (
        Path(parsed_args.output_dir) / "results"
        if parsed_args.output_dir
        else base_dir / "results"
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    jobs = Jobs()

    seed = 0

    for model in ["gated"]:
        for N_parallel in [4]:
            name = f"{model}_pendulum_AC_seed_{seed}_Nparallel_{N_parallel}"
            cmd_args = [
                "--model",
                model,
                "--N_parallel",
                str(N_parallel),
                "--key_seed",
                str(seed),
                "--output_file",
                str(results_dir / name),
            ]

            # Create a bash script to set PYTHONPATH and run the processing script
            bash_script = "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f'export PYTHONPATH={shlex.quote(str(base_dir))}:"${{PYTHONPATH:-}}"',
                    f"python -u -m {shlex.quote(module_path)} {' '.join(shlex.quote(arg) for arg in cmd_args)}",
                ]
            )

            jobs.add_std(
                name=name,
                execution={
                    "script": bash_script,
                    "stdout": str(log_dir / f"job.out.{name}"),
                    "stderr": str(log_dir / f"job.err.{name}"),
                    "wd": str(base_dir),
                },
                resources={
                    "numNodes": 1,
                    "numCores": {"exact": 18},
                    "nodeCrs": {"gpu": 1},
                },
            )
    return jobs


def main():
    jobs = create_jobs()

    manager = LocalManager(
        server_args=[
            "--net",
            "--disable-nl",
            "--resources",
            "slurm",
            "--envschema",
            "slurm",
        ],
    )

    print(f"Submitting {len(jobs.jobs())} jobs to the local manager...", flush=True)
    try:
        manager.submit(jobs)
        manager.wait4all()
        manager.finish()
    except Exception as e:
        print(f"ERROR during job submission or execution: {repr(e)}", flush=True)
        return


if __name__ == "__main__":
    main()
