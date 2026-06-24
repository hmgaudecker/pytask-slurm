"""Execute tasks via SLURM."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cloudpickle
from _pytask.node_protocols import PPathNode
from pytask import ExecutionReport, PNode, PythonNode, Session, hookimpl
from pytask.tree_util import tree_leaves, tree_map, tree_structure
from pytask_parallel.typing import CarryOverPath

from pytask_slurm.cancel import cancel_jobs
from pytask_slurm.monitor import SlurmJobResult, SlurmJobStatus, poll_job_statuses
from pytask_slurm.submit import (
    SlurmJob,
    clear_job_record,
    reattach_task,
    submit_task,
)

if TYPE_CHECKING:
    from pytask import PTask
    from pytask_parallel.wrappers import WrapperResult

logger = logging.getLogger(__name__)

_PENDING_STATUSES = frozenset({SlurmJobStatus.PENDING, SlurmJobStatus.RUNNING})

_LOG_TAIL_MAX_CHARS = 4000


def _read_log_tail(log_path: Path, max_chars: int = _LOG_TAIL_MAX_CHARS) -> str:
    """Read the tail of a SLURM log file, returning at most *max_chars*."""
    try:
        if log_path.exists():
            return log_path.read_text(errors="replace")[-max_chars:]
    except OSError:
        pass
    return ""


def _refresh_nfs_cache(directory: Path) -> None:
    """Force NFS attribute cache refresh by listing the directory.

    On shared filesystems, recently-written files may not be visible to the
    login node immediately. Calling ``os.listdir()`` triggers a metadata
    lookup that invalidates the NFS attribute cache for the directory.
    """
    try:
        os.listdir(directory)
    except OSError:
        pass


@hookimpl(tryfirst=True)
def pytask_execute_build(session: Session) -> bool | None:
    """Execute tasks by submitting them as SLURM jobs.

    `tryfirst=True` ensures this hook runs before any other
    `pytask_execute_build` impl (notably `pytask-parallel`'s, which is
    also registered when both plugins are active). Returning a non-None
    value short-circuits pluggy's hook chain, so pytask-parallel never
    gets to dispatch the same ready tasks to its local process pool.

    Three-phase loop (same structure as pytask-parallel):
    1. Submit ready tasks via sbatch (up to max_jobs).
    2. Poll sacct for completed/failed jobs and collect reports.
    3. Process reports, log results, update scheduler.

    """
    __tracebackhide__ = True
    reports = session.execution_reports
    running_jobs: dict[str, SlurmJob] = {}

    work_dir = session.config["root"] / ".pytask" / "slurm"
    work_dir.mkdir(parents=True, exist_ok=True)

    poll_interval: float = session.config["slurm_poll_interval"]
    max_jobs: int = session.config["slurm_max_jobs"]
    unknown_timeout: int = session.config["slurm_unknown_timeout"]

    try:
        while session.scheduler.is_active():
            try:
                newly_collected_reports = _submit_ready_tasks(
                    session, running_jobs, max_jobs, work_dir
                )
                newly_collected_reports.extend(
                    _collect_completed_jobs(session, running_jobs, unknown_timeout)
                )
                _process_reports(session, newly_collected_reports, reports)

                if session.should_stop:
                    break

                if running_jobs:
                    time.sleep(poll_interval)
                else:
                    time.sleep(0.1)

            except KeyboardInterrupt:
                break

    finally:
        _cancel_remaining_jobs(session, running_jobs)

    return True


def _cancel_remaining_jobs(session: Session, running_jobs: dict[str, SlurmJob]) -> None:
    """Cancel jobs still running at controller exit, unless opted out.

    With `slurm_cancel_on_exit = false` the jobs are left running so the work
    survives a controller death; a restarted controller re-attaches to them.
    """
    if not session.config["slurm_cancel_on_exit"]:
        return
    remaining_ids = [j.job_id for j in running_jobs.values()]
    if remaining_ids:
        cancel_jobs(remaining_ids)


def _submit_ready_tasks(
    session: Session,
    running_jobs: dict[str, SlurmJob],
    max_jobs: int,
    work_dir: Path,
) -> list[ExecutionReport]:
    """Submit ready tasks via sbatch, returning reports for failed submissions."""
    newly_collected: list[ExecutionReport] = []
    n_new_tasks = max_jobs - len(running_jobs)
    if n_new_tasks < 1:
        return newly_collected

    ready_tasks = list(session.scheduler.get_ready(n_new_tasks))

    for task_name in ready_tasks:
        task = cast("PTask", session.dag.nodes[task_name])
        session.hook.pytask_execute_task_log_start(session=session, task=task)
        try:
            session.hook.pytask_execute_task_setup(session=session, task=task)
            reattached = reattach_task(task, work_dir)
            if reattached is not None:
                logger.info(
                    "Re-attached to running SLURM job %s for task %r instead of "
                    "resubmitting (controller restart).",
                    reattached.job_id,
                    task_name,
                )
                running_jobs[task_name] = reattached
            else:
                running_jobs[task_name] = submit_task(task, session.config, work_dir)
        except Exception:  # noqa: BLE001
            report = ExecutionReport.from_task_and_exception(task, sys.exc_info())
            newly_collected.append(report)
            session.scheduler.done(task_name)

    return newly_collected


def _is_actionable_status(
    status: SlurmJobStatus, slurm_job: SlurmJob, unknown_timeout: int
) -> bool:
    """Return True if *status* means the job should be collected now."""
    if status in _PENDING_STATUSES:
        return False
    if status == SlurmJobStatus.UNKNOWN:
        if time.monotonic() - slurm_job.submitted_at < unknown_timeout:
            return False
        logger.warning(
            "SLURM job %s for task %r has been in UNKNOWN state for over "
            "%d seconds; treating as failed.",
            slurm_job.job_id,
            slurm_job.task_name,
            unknown_timeout,
        )
    return True


_RESULT_FILE_MIN_AGE_SECONDS = 5


def _result_file_is_stable(path: Path) -> bool:
    """Return True if *path* was last modified at least a few seconds ago.

    Guards against reading a partially-written result pickle: the runner
    creates the file before writing is complete, so we wait until the mtime
    is old enough to assume the write has finished.
    """
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age >= _RESULT_FILE_MIN_AGE_SECONDS


def _check_result_file_fallback(
    session: Session,
    running_jobs: dict[str, SlurmJob],
    reported_job_ids: set[str],
    already_done: set[str],
) -> tuple[list[ExecutionReport], list[str]]:
    """Detect completed jobs whose result file exists but sacct/squeue missed them.

    Returns a tuple of (reports, fallback_task_names).
    """
    reports: list[ExecutionReport] = []
    fallback_task_names: list[str] = []
    for task_name, slurm_job in running_jobs.items():
        if slurm_job.job_id in reported_job_ids or task_name in already_done:
            continue
        if slurm_job.result_path.exists() and _result_file_is_stable(
            slurm_job.result_path
        ):
            logger.info(
                "SLURM job %s for task %r not reported by sacct/squeue but "
                "result file exists; treating as completed.",
                slurm_job.job_id,
                task_name,
            )
            task = session.dag.nodes[task_name]
            reports.append(_process_completed_job(session, task, slurm_job))
            fallback_task_names.append(task_name)
    return reports, fallback_task_names


def _collect_completed_jobs(
    session: Session,
    running_jobs: dict[str, SlurmJob],
    unknown_timeout: int,
) -> list[ExecutionReport]:
    """Poll sacct and return reports for completed/failed jobs."""
    if not running_jobs:
        return []

    newly_collected: list[ExecutionReport] = []
    job_id_to_task_name = {j.job_id: name for name, j in running_jobs.items()}
    results = poll_job_statuses(list(job_id_to_task_name))
    completed_task_names: list[str] = []

    # Refresh NFS cache once before reading any result/log files.
    work_dirs_refreshed: set[Path] = set()
    for task_name in running_jobs:
        slurm_job = running_jobs[task_name]
        parent = slurm_job.result_path.parent
        if parent not in work_dirs_refreshed:
            _refresh_nfs_cache(parent)
            work_dirs_refreshed.add(parent)

    for job_id, job_result in results.items():
        task_name = job_id_to_task_name.get(job_id)
        if task_name is None:
            continue
        slurm_job = running_jobs[task_name]
        if not _is_actionable_status(job_result.status, slurm_job, unknown_timeout):
            continue

        task = cast("PTask", session.dag.nodes[task_name])
        report = _build_terminal_report(session, task, slurm_job, job_result)
        newly_collected.append(report)
        completed_task_names.append(task_name)

    # Result-file fallback: if a job disappeared from both sacct and squeue
    # (e.g. sacct is unavailable and the job finished so squeue no longer
    # lists it), check whether the result pickle exists.
    fallback_reports, fallback_names = _check_result_file_fallback(
        session, running_jobs, set(results), set(completed_task_names)
    )
    newly_collected.extend(fallback_reports)
    completed_task_names.extend(fallback_names)

    for task_name in completed_task_names:
        finished = running_jobs.pop(task_name)
        clear_job_record(finished)
        session.scheduler.done(task_name)

    return newly_collected


def _process_reports(
    session: Session,
    newly_collected: list[ExecutionReport],
    reports: list[ExecutionReport],
) -> None:
    """Log and store execution reports."""
    for report in newly_collected:
        session.hook.pytask_execute_task_process_report(session=session, report=report)
        session.hook.pytask_execute_task_log_end(session=session, report=report)
        reports.append(report)


def _build_terminal_report(
    session: Session,
    task: PTask,
    slurm_job: SlurmJob,
    job_result: SlurmJobResult,
) -> ExecutionReport:
    """Build the execution report for a job that has reached a terminal state.

    A task that raised records its structured traceback in the result pickle and
    exits non-zero, so a failed job is reported from that pickle when present; only
    a job killed before writing one falls back to the SLURM-state / log-tail report.
    """
    if job_result.status == SlurmJobStatus.COMPLETED:
        exit_code = job_result.exit_code
        if exit_code is None or exit_code == 0:
            return _process_completed_job(session, task, slurm_job)
        return _read_result_report(session, task, slurm_job) or _process_nonzero_exit(
            task, slurm_job, exit_code
        )
    return _read_result_report(session, task, slurm_job) or _process_failed_job(
        task, slurm_job, job_result.status
    )


def _read_result_report(
    session: Session,
    task: PTask,
    slurm_job: SlurmJob,
) -> ExecutionReport | None:
    """Return the report from this run's result pickle, or None to fall back.

    A non-zero exit or a non-COMPLETED status means the job failed. The common case
    is a task that raised: the runner records the traceback in the result pickle and
    exits non-zero, so reading that pickle surfaces the real exception (the same
    detail as a clean run). `submit_task` removes any stale pickle before submitting,
    so a present pickle always belongs to this run. Returns None when no pickle was
    written — the job was killed mid-task by OOM or timeout — so the caller falls back
    to the SLURM-state / log-tail report.
    """
    if slurm_job.result_path.exists() and _result_file_is_stable(slurm_job.result_path):
        return _process_completed_job(session, task, slurm_job)
    return None


def _process_completed_job(
    session: Session,
    task: PTask,
    slurm_job: SlurmJob,
) -> ExecutionReport:
    """Read the result pickle and build an execution report."""
    try:
        with slurm_job.result_path.open("rb") as f:
            # Security: pickle is self-generated by the SLURM worker, not user-supplied.
            wrapper_result: WrapperResult = cloudpickle.load(f)
    except Exception as deserialize_err:  # noqa: BLE001
        # Include the SLURM log in the error so the user can see what went
        # wrong (e.g. the runner crashed before writing the result pickle).
        log_content = _read_log_tail(slurm_job.log_path)

        msg = (
            f"SLURM job {slurm_job.job_id} for task {slurm_job.task_name!r} "
            f"reported COMPLETED but the result file is missing or unreadable: "
            f"{slurm_job.result_path}"
        )
        if log_content:
            msg += f"\n\nSLURM log (last {_LOG_TAIL_MAX_CHARS} chars):\n{log_content}"
        else:
            msg += "\n\nSLURM log is empty — the worker process may have crashed at startup."

        exc = RuntimeError(msg)
        exc.__cause__ = deserialize_err
        return ExecutionReport.from_task_and_exception(
            task, (type(exc), exc, exc.__traceback__)
        )

    session.warnings.extend(wrapper_result.warning_reports)

    if wrapper_result.stdout:
        task.report_sections.append(("call", "stdout", wrapper_result.stdout))
    if wrapper_result.stderr:
        task.report_sections.append(("call", "stderr", wrapper_result.stderr))

    if wrapper_result.exc_info is not None:
        return ExecutionReport.from_task_and_exception(
            task,
            wrapper_result.exc_info,
        )

    _update_carry_over_products(task, wrapper_result.carry_over_products)

    # Refresh NFS cache for directories containing products.  The SLURM
    # worker writes product files on a compute node; the NFS attribute
    # cache on the login node may not yet reflect these writes.
    product_dirs_refreshed: set[Path] = set()
    for node in tree_leaves(task.produces):
        if isinstance(node, PPathNode):
            parent = node.path.parent
            if parent not in product_dirs_refreshed:
                _refresh_nfs_cache(parent)
                product_dirs_refreshed.add(parent)

    try:
        session.hook.pytask_execute_task_teardown(session=session, task=task)
    except Exception:  # noqa: BLE001
        return ExecutionReport.from_task_and_exception(task, sys.exc_info())

    return ExecutionReport.from_task(task)


def _process_nonzero_exit(
    task: PTask,
    slurm_job: SlurmJob,
    exit_code: int,
) -> ExecutionReport:
    """Build a report for a job that SLURM reports as COMPLETED but exited non-zero."""
    log_content = _read_log_tail(slurm_job.log_path)

    msg = (
        f"SLURM job {slurm_job.job_id} for task {slurm_job.task_name!r} "
        f"reported COMPLETED but exited with code {exit_code}"
    )
    if log_content:
        msg += f"\n\nSLURM log (last {_LOG_TAIL_MAX_CHARS} chars):\n{log_content}"
    else:
        msg += "\n\nSLURM log is empty or not yet available."

    exc = RuntimeError(msg)
    return ExecutionReport.from_task_and_exception(
        task, (type(exc), exc, exc.__traceback__)
    )


def _process_failed_job(
    task: PTask,
    slurm_job: SlurmJob,
    status: SlurmJobStatus,
) -> ExecutionReport:
    """Build a failure report from a SLURM job that did not complete."""
    log_content = _read_log_tail(slurm_job.log_path)

    msg = f"SLURM job {slurm_job.job_id} {status.value}"
    if log_content:
        msg += f"\n\nSLURM log (last {_LOG_TAIL_MAX_CHARS} chars):\n{log_content}"

    exc = RuntimeError(msg)
    return ExecutionReport.from_task_and_exception(
        task, (type(exc), exc, exc.__traceback__)
    )


def _update_carry_over_products(
    task: PTask,
    carry_over_products: Any,  # noqa: ANN401
) -> None:
    """Update products carried over from the SLURM worker.

    Copied from pytask-parallel's ``_update_carry_over_products``.

    """

    def _update_carry_over_node(
        x: PNode, y: CarryOverPath | PythonNode | None
    ) -> PNode:
        if y is None:
            return x
        if isinstance(x, PPathNode) and isinstance(y, CarryOverPath):
            x.path.write_bytes(y.content)
            return x
        if isinstance(y, PythonNode):
            x.save(y.load())
            return x
        raise NotImplementedError

    structure_carry_over_products = tree_structure(carry_over_products)
    structure_produces = tree_structure(task.produces)
    if structure_produces.is_prefix(structure_carry_over_products, strict=False):
        task.produces = tree_map(
            _update_carry_over_node,
            task.produces,
            carry_over_products,
        )
