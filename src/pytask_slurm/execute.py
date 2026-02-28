"""Execute tasks via SLURM."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
from _pytask.node_protocols import PPathNode
from pytask import ExecutionReport, PNode, PythonNode, Session, hookimpl
from pytask.tree_util import tree_map, tree_structure
from pytask_parallel.typing import CarryOverPath

from pytask_slurm.cancel import cancel_jobs
from pytask_slurm.monitor import SlurmJobStatus, poll_job_statuses
from pytask_slurm.submit import SlurmJob, submit_task

if TYPE_CHECKING:
    from pytask import PTask
    from pytask_parallel.wrappers import WrapperResult

logger = logging.getLogger(__name__)

_PENDING_STATUSES = frozenset({SlurmJobStatus.PENDING, SlurmJobStatus.RUNNING})

# How long a job may stay in UNKNOWN status before being treated as failed.
_UNKNOWN_TIMEOUT_SECONDS = 600


@hookimpl
def pytask_execute_build(session: Session) -> bool | None:
    """Execute tasks by submitting them as SLURM jobs.

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

    try:
        while session.scheduler.is_active():
            try:
                newly_collected_reports = _submit_ready_tasks(
                    session, running_jobs, max_jobs, work_dir
                )
                newly_collected_reports.extend(
                    _collect_completed_jobs(session, running_jobs)
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
        remaining_ids = [j.job_id for j in running_jobs.values()]
        if remaining_ids:
            cancel_jobs(remaining_ids)

    return True


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
        task = session.dag.nodes[task_name]["task"]
        session.hook.pytask_execute_task_log_start(session=session, task=task)
        try:
            session.hook.pytask_execute_task_setup(session=session, task=task)
            slurm_job = submit_task(task, session.config, work_dir)
            running_jobs[task_name] = slurm_job
        except Exception:  # noqa: BLE001
            report = ExecutionReport.from_task_and_exception(task, sys.exc_info())
            newly_collected.append(report)
            session.scheduler.done(task_name)

    return newly_collected


def _is_actionable_status(status: SlurmJobStatus, slurm_job: SlurmJob) -> bool:
    """Return True if *status* means the job should be collected now."""
    if status in _PENDING_STATUSES:
        return False
    if status == SlurmJobStatus.UNKNOWN:
        if time.monotonic() - slurm_job.submitted_at < _UNKNOWN_TIMEOUT_SECONDS:
            return False
        logger.warning(
            "SLURM job %s for task %r has been in UNKNOWN state for over "
            "%d seconds; treating as failed.",
            slurm_job.job_id,
            slurm_job.task_name,
            _UNKNOWN_TIMEOUT_SECONDS,
        )
    return True


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
        if slurm_job.result_path.exists():
            logger.info(
                "SLURM job %s for task %r not reported by sacct/squeue but "
                "result file exists; treating as completed.",
                slurm_job.job_id,
                task_name,
            )
            task = session.dag.nodes[task_name]["task"]
            reports.append(_process_completed_job(session, task, slurm_job))
            fallback_task_names.append(task_name)
    return reports, fallback_task_names


def _collect_completed_jobs(
    session: Session,
    running_jobs: dict[str, SlurmJob],
) -> list[ExecutionReport]:
    """Poll sacct and return reports for completed/failed jobs."""
    if not running_jobs:
        return []

    newly_collected: list[ExecutionReport] = []
    job_id_to_name = {j.job_id: name for name, j in running_jobs.items()}
    statuses = poll_job_statuses(list(job_id_to_name))
    completed_task_names: list[str] = []

    for job_id, status in statuses.items():
        task_name = job_id_to_name.get(job_id)
        if task_name is None:
            continue
        slurm_job = running_jobs[task_name]
        if not _is_actionable_status(status, slurm_job):
            continue

        task = session.dag.nodes[task_name]["task"]
        if status == SlurmJobStatus.COMPLETED:
            report = _process_completed_job(session, task, slurm_job)
        else:
            report = _process_failed_job(task, slurm_job, status)

        newly_collected.append(report)
        completed_task_names.append(task_name)

    # Result-file fallback: if a job disappeared from both sacct and squeue
    # (e.g. sacct is unavailable and the job finished so squeue no longer
    # lists it), check whether the result pickle exists.
    fallback_reports, fallback_names = _check_result_file_fallback(
        session, running_jobs, set(statuses), set(completed_task_names)
    )
    newly_collected.extend(fallback_reports)
    completed_task_names.extend(fallback_names)

    for task_name in completed_task_names:
        running_jobs.pop(task_name)
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
    except Exception:  # noqa: BLE001
        return ExecutionReport.from_task_and_exception(task, sys.exc_info())

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

    try:
        session.hook.pytask_execute_task_teardown(session=session, task=task)
    except Exception:  # noqa: BLE001
        return ExecutionReport.from_task_and_exception(task, sys.exc_info())

    return ExecutionReport.from_task(task)


def _process_failed_job(
    task: PTask,
    slurm_job: SlurmJob,
    status: SlurmJobStatus,
) -> ExecutionReport:
    """Build a failure report from a SLURM job that did not complete."""
    log_content = ""
    try:
        if slurm_job.log_path.exists():
            log_content = slurm_job.log_path.read_text(errors="replace")[-4000:]
    except OSError:
        pass

    msg = f"SLURM job {slurm_job.job_id} {status.value}"
    if log_content:
        msg += f"\n\nSLURM log (last 4000 chars):\n{log_content}"

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
