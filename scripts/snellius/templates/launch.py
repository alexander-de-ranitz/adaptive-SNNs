"""Shared qcg-pilotjob launch helper for Snellius runs"""

import argparse
import shlex
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager

REPO_ROOT = Path(__file__).resolve().parents[3]

# Server args needed for GPU sweeps driven through SLURM/qcg-pilotjob.
_GPU_SERVER_ARGS = [
    "--net",
    "--disable-nl",
    "--resources",
    "slurm",
    "--envschema",
    "slurm",
]


def parse_output_dir() -> tuple[Path, Path]:
    """Parse ``--output_dir`` and return the created ``(log_dir, results_dir)``."""
    parser = argparse.ArgumentParser(description="Launch jobs on Snellius")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/default_folder/",
        help="Directory to store output files",
    )
    args = parser.parse_args()

    base = Path(args.output_dir) if args.output_dir else REPO_ROOT
    log_dir = base / "logs"
    results_dir = base / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, results_dir


def _bash_script(module_path: str, cmd_args: list[str]) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'export PYTHONPATH={shlex.quote(str(REPO_ROOT))}:"${{PYTHONPATH:-}}"',
            f"{shlex.quote(sys.executable)} -u -m {shlex.quote(module_path)} "
            + " ".join(shlex.quote(arg) for arg in cmd_args),
        ]
    )


def build_jobs(
    module_path: str,
    param_grid: Iterable[tuple[str, Mapping[str, object]]],
    log_dir: Path,
    results_dir: Path,
    gpu: bool = False,
    num_cores: int = 1,
) -> Jobs:
    """Build one job per (name, {cli_flag: value}) entry in param_grid.

    --output_file <results_dir>/<name>is appended to every job's args.
    Set gpu=True to add a GPU to the allocation.
    """
    jobs = Jobs()
    for name, params in param_grid:
        cmd_args: list[str] = []
        for flag, value in params.items():
            cmd_args += [flag, str(value)]
        cmd_args += ["--output_file", str(results_dir / name)]

        script = _bash_script(module_path, cmd_args)
        stdout = str(log_dir / f"job.out.{name}")
        stderr = str(log_dir / f"job.err.{name}")

        if gpu:
            jobs.add_std(
                name=name,
                execution={
                    "script": script,
                    "stdout": stdout,
                    "stderr": stderr,
                    "wd": str(REPO_ROOT),
                },
                resources={
                    "numNodes": 1,
                    "numCores": {"exact": num_cores},
                    "nodeCrs": {"gpu": 1},
                },
            )
        else:
            jobs.add(
                name=name,
                script=script,
                stdout=stdout,
                stderr=stderr,
                wd=str(REPO_ROOT),
                numCores=num_cores,
                iteration=1,
            )
    return jobs


def submit(jobs: Jobs, *, gpu: bool = False) -> None:
    """Submit ``jobs`` to a local qcg-pilotjob manager and wait for completion."""
    manager = LocalManager(server_args=_GPU_SERVER_ARGS) if gpu else LocalManager()

    print(f"Submitting {len(jobs.jobs())} jobs to the local manager...", flush=True)
    try:
        manager.submit(jobs)
        manager.wait4all()
        manager.finish()
    except Exception as e:
        print(f"ERROR during job submission or execution: {repr(e)}", flush=True)
