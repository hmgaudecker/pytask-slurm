"""Configure pytask-slurm."""

from __future__ import annotations

from typing import Any

from pytask import hookimpl


@hookimpl
def pytask_parse_config(config: dict[str, Any]) -> None:
    """Parse the configuration."""
    config["markers"]["slurm"] = (
        "Override SLURM resources for a task. Options: partition, time, mem,"
        " cpus_per_task, account, qos, gpus, python_unbuffered. Pass"
        ' extra="..." for any other sbatch option (e.g.'
        ' extra="--constraint=a100 --mail-type=END").'
    )
    config.setdefault("slurm", False)

    # SLURM resource options — passed through to sbatch.  When None, the
    # corresponding flag is omitted and SLURM uses the cluster's own default.
    config.setdefault("slurm_partition", None)
    config.setdefault("slurm_time", None)
    config.setdefault("slurm_mem", None)
    config.setdefault("slurm_cpus_per_task", None)
    config.setdefault("slurm_account", None)
    config.setdefault("slurm_qos", None)
    config.setdefault("slurm_gpus", None)
    config.setdefault("slurm_extra", None)
    # When True, the batch script exports `PYTHONUNBUFFERED=1` before the
    # runner. Compute-node stdout is a file (not a TTY), so Python defaults
    # to block-buffered I/O and per-period log lines only land at process
    # exit. Opt in per-task when live progress visibility matters.
    config.setdefault("slurm_python_unbuffered", False)
    # Per-task environment variables, exported in the batch script before the
    # runner starts. Use for backend tuning that JAX/XLA/pylcm read at process
    # start — typical entries: `JAX_COMPILATION_CACHE_DIR`,
    # `XLA_PYTHON_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_ALLOCATOR`,
    # `XLA_FLAGS`. Values pass through `shlex.quote`, but shell references
    # like `$SLURM_JOB_ID` survive unexpanded so the worker shell evaluates
    # them on the compute node (handy for node-local cache paths).
    config.setdefault("slurm_env", {})

    # pytask-slurm executor parameters — control the plugin's own scheduling
    # loop, not passed to sbatch.
    config.setdefault("slurm_max_jobs", 50)
    config.setdefault("slurm_poll_interval", 10)
    config.setdefault("slurm_unknown_timeout", 300)


@hookimpl(trylast=True)
def pytask_post_parse(config: dict[str, Any]) -> None:
    """Register the SLURM executor if --slurm is active."""
    if not config["slurm"]:
        return

    if config["pdb"] or config["trace"] or config["dry_run"]:
        return

    from pytask_slurm import execute, logging  # noqa: PLC0415

    config["pm"].register(execute)
    config["pm"].register(logging)
