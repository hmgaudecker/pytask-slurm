"""Submit tasks to SLURM via sbatch."""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
from pytask import console, get_marks
from pytask_parallel.utils import (
    create_kwargs_for_task,
    get_module,
    should_pickle_module_by_value,
    strip_annotation_locals,
)

if TYPE_CHECKING:
    from pytask import PTask

_SLURM_MARK_KEYS = frozenset(
    {"partition", "time", "mem", "cpus_per_task", "account", "qos", "gpus"}
)
_SLURM_INT_KEYS = frozenset({"cpus_per_task", "gpus"})


@dataclass(frozen=True)
class SlurmJob:
    """Tracks a submitted SLURM job."""

    job_id: str
    task_name: str
    payload_path: Path
    result_path: Path
    log_path: Path
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class TaskPayload:
    """Serialized payload sent to the SLURM worker."""

    task: Any
    kwargs: dict[str, Any]
    console_options: Any
    session_filterwarnings: tuple[str, ...]
    show_locals: bool
    task_filterwarnings: list[Any]
    result_path: str
    # TODO(v1.0): Remove this field.  It existed before the JSON sidecar
    # switch; the runner now reads sys_path from the sidecar instead.
    sys_path: list[str] | None = None


def _validate_slurm_option(
    key: str,
    value: Any,  # noqa: ANN401
    task_name: str,
    source: str = "SLURM option",
) -> None:
    """Validate a single SLURM option value (type and range)."""
    if key in _SLURM_INT_KEYS:
        if isinstance(value, bool) or not isinstance(value, int):
            msg = (
                f"{source} {key!r} for task {task_name!r} "
                f"must be an int, got {type(value).__name__}."
            )
            raise ValueError(msg)
        if value <= 0:
            msg = (
                f"{source} {key!r} for task {task_name!r} "
                f"must be a positive integer, got {value}."
            )
            raise ValueError(msg)
    else:
        if not isinstance(value, str):
            msg = (
                f"{source} {key!r} for task {task_name!r} "
                f"must be a str, got {type(value).__name__}."
            )
            raise ValueError(msg)
        if not value:
            msg = (
                f"{source} {key!r} for task {task_name!r} must not be an empty string."
            )
            raise ValueError(msg)


def _validate_mark(mark: Any, task_name: str) -> dict[str, Any]:  # noqa: ANN401
    """Validate and return kwargs from a single ``@pytask.mark.slurm`` decorator."""
    if mark.args:
        msg = (
            f"@pytask.mark.slurm for task {task_name!r} received positional "
            f"arguments {mark.args!r}. Use keyword arguments instead, e.g. "
            f"@pytask.mark.slurm(partition='gpu')."
        )
        raise ValueError(msg)

    unknown = set(mark.kwargs) - _SLURM_MARK_KEYS - {"extra"}
    if unknown:
        msg = (
            f"@pytask.mark.slurm for task {task_name!r} received unknown "
            f"kwargs {sorted(unknown)}. Well-known options: "
            f"{sorted(_SLURM_MARK_KEYS)}. Use extra=\"...\" for arbitrary "
            f"sbatch flags."
        )
        raise ValueError(msg)

    for key, value in mark.kwargs.items():
        if key == "extra":
            continue
        if value is None:
            msg = (
                f"@pytask.mark.slurm kwarg {key!r} for task {task_name!r} "
                f"must not be None."
            )
            raise ValueError(msg)
        _validate_slurm_option(
            key, value, task_name, source="@pytask.mark.slurm kwarg"
        )

    extra = mark.kwargs.get("extra")
    if extra is not None:
        if not isinstance(extra, str):
            msg = (
                f"@pytask.mark.slurm 'extra' for task {task_name!r} "
                f"must be a str, got {type(extra).__name__}."
            )
            raise TypeError(msg)

    return dict(mark.kwargs)


def _get_slurm_options(task: PTask, session_config: dict[str, Any]) -> dict[str, Any]:
    """Build SLURM resource options by merging global defaults with per-task marks."""
    options: dict[str, Any] = {
        "partition": session_config["slurm_partition"],
        "time": session_config["slurm_time"],
        "mem": session_config["slurm_mem"],
        "cpus_per_task": session_config["slurm_cpus_per_task"],
        "account": session_config["slurm_account"],
        "qos": session_config["slurm_qos"],
        "gpus": session_config["slurm_gpus"],
        "extra": session_config["slurm_extra"],
    }

    marks = get_marks(task, "slurm")

    if len(marks) > 1:
        msg = (
            f"Task {task.name!r} has {len(marks)} @pytask.mark.slurm decorators, "
            f"but only one is allowed. Merge them into a single decorator."
        )
        raise ValueError(msg)

    # Validate all non-None config values upfront (catches invalid config even
    # when a mark overrides the key, so misconfigurations don't go unnoticed).
    for key, value in options.items():
        if key == "extra":
            if value is not None and not isinstance(value, str):
                msg = f"slurm_extra must be a string, got {type(value).__name__}"
                raise TypeError(msg)
        elif value is not None:
            _validate_slurm_option(key, value, task.name, source="Global config")

    if marks:
        options.update(_validate_mark(marks[0], task.name))

    return options


def _write_batch_script(
    python: str,
    payload_path: Path,
    result_path: Path,
    script_path: Path,
) -> None:
    """Write a SLURM batch script that runs the pytask runner.

    Using a script file instead of ``--wrap`` avoids quoting issues and lets us
    redirect stderr to stdout within the script so that all output (including
    Python tracebacks) ends up in the single ``--output`` log file.

    The script prints diagnostic lines before and after the Python command so
    the log file is never empty — even if Python fails to start.
    """
    q_python = shlex.quote(python)
    q_payload = shlex.quote(str(payload_path))
    q_result = shlex.quote(str(result_path))
    script_path.write_text(
        f"#!/bin/bash\n"
        f"# pytask-slurm batch script (auto-generated)\n"
        f"exec 2>&1\n"
        f'echo "pytask-slurm: starting (pid=$$, host=$(hostname))"\n'
        f'echo "pytask-slurm: python={q_python}"\n'
        f"{q_python} -m pytask_slurm.runner {q_payload} {q_result}\n"
        f"_exit_code=$?\n"
        f'echo "pytask-slurm: runner exited with code $_exit_code"\n'
        f"exit $_exit_code\n"
    )
    script_path.chmod(0o755)


def _build_sbatch_cmd(
    opts: dict[str, Any],
    task_hash: str,
    *,
    log_path: Path,
    script_path: Path,
) -> list[str]:
    """Build the sbatch command list from task options."""
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=pytask-{task_hash}",
        f"--output={log_path}",
    ]

    if opts["time"] is not None:
        cmd.append(f"--time={opts['time']}")
    if opts["mem"] is not None:
        cmd.append(f"--mem={opts['mem']}")
    if opts["cpus_per_task"] is not None:
        cmd.append(f"--cpus-per-task={opts['cpus_per_task']}")
    if opts["partition"]:
        cmd.append(f"--partition={opts['partition']}")
    if opts["account"]:
        cmd.append(f"--account={opts['account']}")
    if opts["qos"]:
        cmd.append(f"--qos={opts['qos']}")
    if opts["gpus"] is not None:
        cmd.append(f"--gpus={opts['gpus']}")

    # Append extra sbatch flags (merged: config default, overridden by mark).
    extra = opts.get("extra")
    if extra:
        cmd.extend(shlex.split(extra))

    cmd.append(str(script_path))
    return cmd


def submit_task(
    task: PTask,
    session_config: dict[str, Any],
    work_dir: Path,
) -> SlurmJob:
    """Serialize a task and submit it to SLURM via sbatch."""
    task_hash = hashlib.sha256(task.name.encode()).hexdigest()[:16]

    payload_path = work_dir / f"{task_hash}_payload.pkl"
    result_path = work_dir / f"{task_hash}_result.pkl"
    log_path = work_dir / f"{task_hash}.log"

    # Remove stale result file so we don't read old results.
    result_path.unlink(missing_ok=True)

    # Prepare the task for pickling (same as pytask-parallel process worker).
    strip_annotation_locals(task)
    task_module = get_module(task.function, getattr(task, "path", None))
    if should_pickle_module_by_value(task_module):
        cloudpickle.register_pickle_by_value(task_module)

    kwargs = create_kwargs_for_task(task, remote=False)

    payload = TaskPayload(
        task=task,
        kwargs=kwargs,
        console_options=console.options,
        session_filterwarnings=session_config["filterwarnings"],
        show_locals=session_config["show_locals"],
        task_filterwarnings=get_marks(task, "filterwarnings"),
        result_path=str(result_path),
    )

    # Write sys.path as a JSON sidecar so the runner can restore it before
    # unpickling. pytask dynamically loads task modules without adding their
    # directories to sys.path, so we must include them explicitly.
    import json  # noqa: PLC0415

    runner_path = sys.path.copy()
    task_path = getattr(task, "path", None)
    if task_path is not None:
        task_dir = str(Path(task_path).parent)
        if task_dir not in runner_path:
            runner_path.insert(0, task_dir)
    root_dir = str(session_config.get("root", ""))
    if root_dir and root_dir not in runner_path:
        runner_path.insert(0, root_dir)

    sys_path_file = work_dir / f"{task_hash}_syspath.json"
    sys_path_file.write_text(json.dumps(runner_path))

    with payload_path.open("wb") as f:
        cloudpickle.dump(payload, f)

    # Write batch script and build sbatch command.
    script_path = work_dir / f"{task_hash}_job.sh"
    _write_batch_script(sys.executable, payload_path, result_path, script_path)

    opts = _get_slurm_options(task, session_config)
    cmd = _build_sbatch_cmd(
        opts,
        task_hash,
        log_path=log_path,
        script_path=script_path,
    )

    result = subprocess.run(  # noqa: S603
        cmd, capture_output=True, check=False, text=True, timeout=30
    )

    if result.returncode != 0:
        msg = f"sbatch failed for task {task.name!r}: {result.stderr.strip()}"
        raise RuntimeError(msg)

    job_id = result.stdout.strip().split(";")[0]

    return SlurmJob(
        job_id=job_id,
        task_name=task.name,
        payload_path=payload_path,
        result_path=result_path,
        log_path=log_path,
    )
