"""Submit tasks to SLURM via sbatch."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import cloudpickle
from pytask import console
from pytask import get_marks

from pytask_parallel.utils import create_kwargs_for_task
from pytask_parallel.utils import get_module
from pytask_parallel.utils import should_pickle_module_by_value
from pytask_parallel.utils import strip_annotation_locals

if TYPE_CHECKING:
    from pytask import PTask


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
    sys_path: list[str]


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
        sys_path=sys.path.copy(),
    )

    # Build the sys.path that the runner needs. pytask dynamically loads task modules
    # without adding their directories to sys.path, so we must include them explicitly.
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

    with open(payload_path, "wb") as f:
        cloudpickle.dump(payload, f)

    # Build sbatch command.
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=pytask-{task_hash}",
        f"--time={session_config['slurm_time']}",
        f"--mem={session_config['slurm_mem']}",
        f"--cpus-per-task={session_config['slurm_cpus_per_task']}",
        f"--output={log_path}",
    ]

    if session_config["slurm_partition"]:
        cmd.append(f"--partition={session_config['slurm_partition']}")
    if session_config["slurm_account"]:
        cmd.append(f"--account={session_config['slurm_account']}")

    runner_cmd = f"{sys.executable} -m pytask_slurm.runner {payload_path} {result_path}"
    cmd.append(f"--wrap={runner_cmd}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

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
