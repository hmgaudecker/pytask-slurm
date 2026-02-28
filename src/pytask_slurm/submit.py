"""Submit tasks to SLURM via sbatch."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
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

_SLURM_MARK_KEYS = frozenset({"partition", "time", "mem", "cpus_per_task", "account"})
_SLURM_INT_KEYS = frozenset({"cpus_per_task"})


@dataclass(frozen=True)
class SlurmJob:
    """Tracks a submitted SLURM job."""

    job_id: str
    task_name: str
    payload_path: Path
    result_path: Path
    log_path: Path


@dataclass(frozen=True)
class TaskPayload:
    """Serialized payload sent to the SLURM worker."""

    task: Any
    kwargs: dict[str, Any]
    console_options: Any
    session_filterwarnings: tuple[str, ...]
    show_locals: bool
    task_filterwarnings: tuple[Any, ...]
    result_path: str


def _validate_slurm_option(
    key: str, value: Any, task_name: str, source: str = "SLURM option"
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
                f"{source} {key!r} for task {task_name!r} "
                f"must not be an empty string."
            )
            raise ValueError(msg)


def _get_slurm_options(task: PTask, session_config: dict[str, Any]) -> dict[str, Any]:
    """Build SLURM resource options by merging global defaults with per-task marks."""
    options = {
        "partition": session_config["slurm_partition"],
        "time": session_config["slurm_time"],
        "mem": session_config["slurm_mem"],
        "cpus_per_task": session_config["slurm_cpus_per_task"],
        "account": session_config["slurm_account"],
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
        if value is not None:
            _validate_slurm_option(key, value, task.name, source="Global config")

    if marks:
        mark = marks[0]

        if mark.args:
            msg = (
                f"@pytask.mark.slurm for task {task.name!r} received positional "
                f"arguments {mark.args!r}. Use keyword arguments instead, e.g. "
                f"@pytask.mark.slurm(partition='gpu')."
            )
            raise ValueError(msg)

        unknown = set(mark.kwargs) - _SLURM_MARK_KEYS
        if unknown:
            msg = (
                f"Unknown @pytask.mark.slurm kwargs for task {task.name!r}: "
                f"{sorted(unknown)}. Allowed: {sorted(_SLURM_MARK_KEYS)}."
            )
            raise ValueError(msg)

        for key, value in mark.kwargs.items():
            if value is None:
                msg = (
                    f"@pytask.mark.slurm kwarg {key!r} for task {task.name!r} "
                    f"must not be None."
                )
                raise ValueError(msg)
            _validate_slurm_option(
                key, value, task.name, source="@pytask.mark.slurm kwarg"
            )

        options.update(mark.kwargs)

    # time, mem, and cpus_per_task are always passed to sbatch unconditionally,
    # so they must not be None after merging.  partition and account are
    # optional (only appended when truthy), so None is valid for them.
    for key in ("time", "mem", "cpus_per_task"):
        if options[key] is None:
            msg = (
                f"SLURM option {key!r} for task {task.name!r} "
                f"must not be None."
            )
            raise ValueError(msg)

    return options


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

    # Build sbatch command using merged per-task + global options.
    opts = _get_slurm_options(task, session_config)

    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=pytask-{task_hash}",
        f"--time={opts['time']}",
        f"--mem={opts['mem']}",
        f"--cpus-per-task={opts['cpus_per_task']}",
        f"--output={log_path}",
    ]

    if opts["partition"]:
        cmd.append(f"--partition={opts['partition']}")
    if opts["account"]:
        cmd.append(f"--account={opts['account']}")

    runner_cmd = f"{sys.executable} -m pytask_slurm.runner {payload_path} {result_path}"
    cmd.append(f"--wrap={runner_cmd}")

    result = subprocess.run(
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
