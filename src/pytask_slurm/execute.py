"""Execute tasks via SLURM."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING
from typing import Any

import cloudpickle
from _pytask.node_protocols import PPathNode
from pytask import ExecutionReport
from pytask import PNode
from pytask import PythonNode
from pytask import Session
from pytask import hookimpl
from pytask.tree_util import tree_map
from pytask.tree_util import tree_structure

from pytask_parallel.typing import CarryOverPath
from pytask_slurm.cancel import cancel_jobs
from pytask_slurm.monitor import SlurmJobStatus
from pytask_slurm.monitor import poll_job_statuses
from pytask_slurm.submit import SlurmJob
from pytask_slurm.submit import submit_task

if TYPE_CHECKING:
    from pytask import PTask
    from pytask_parallel.wrappers import WrapperResult


@hookimpl
def pytask_execute_build(session: Session) -> bool | None:  # noqa: C901, PLR0912, PLR0915
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

    poll_interval = session.config["slurm_poll_interval"]
    max_jobs = session.config["slurm_max_jobs"]

    try:
        while session.scheduler.is_active():
            try:
                newly_collected_reports = []

                # Phase 1: Submit ready tasks.
                n_new_tasks = max_jobs - len(running_jobs)
                ready_tasks = (
                    list(session.scheduler.get_ready(n_new_tasks))
                    if n_new_tasks >= 1
                    else []
                )

                for task_name in ready_tasks:
                    task = session.dag.nodes[task_name]["task"]
                    session.hook.pytask_execute_task_log_start(
                        session=session, task=task
                    )
                    try:
                        session.hook.pytask_execute_task_setup(
                            session=session, task=task
                        )
                        slurm_job = submit_task(task, session.config, work_dir)
                        running_jobs[task_name] = slurm_job
                    except Exception:  # noqa: BLE001
                        report = ExecutionReport.from_task_and_exception(
                            task, sys.exc_info()
                        )
                        newly_collected_reports.append(report)
                        session.scheduler.done(task_name)

                # Phase 2: Poll for completed jobs.
                if running_jobs:
                    job_id_to_name = {
                        j.job_id: name for name, j in running_jobs.items()
                    }
                    statuses = poll_job_statuses(list(job_id_to_name))

                    for job_id, status in statuses.items():
                        task_name = job_id_to_name.get(job_id)
                        if task_name is None:
                            continue

                        if status in (
                            SlurmJobStatus.PENDING,
                            SlurmJobStatus.RUNNING,
                            SlurmJobStatus.UNKNOWN,
                        ):
                            continue

                        slurm_job = running_jobs[task_name]
                        task = session.dag.nodes[task_name]["task"]

                        if status == SlurmJobStatus.COMPLETED:
                            report = _process_completed_job(
                                session, task, slurm_job
                            )
                        else:
                            report = _process_failed_job(
                                session, task, slurm_job, status
                            )

                        newly_collected_reports.append(report)
                        running_jobs.pop(task_name)
                        session.scheduler.done(task_name)

                # Phase 3: Process reports.
                for report in newly_collected_reports:
                    session.hook.pytask_execute_task_process_report(
                        session=session, report=report
                    )
                    session.hook.pytask_execute_task_log_end(
                        session=session, report=report
                    )
                    reports.append(report)

                if session.should_stop:
                    break

                if running_jobs:
                    time.sleep(poll_interval)
                elif not ready_tasks:
                    time.sleep(0.1)

            except KeyboardInterrupt:
                break

    finally:
        # Always cancel remaining SLURM jobs -- they persist beyond process lifetime.
        remaining_ids = [j.job_id for j in running_jobs.values()]
        if remaining_ids:
            cancel_jobs(remaining_ids)

    return True


def _process_completed_job(
    session: Session,
    task: PTask,
    slurm_job: SlurmJob,
) -> ExecutionReport:
    """Read the result pickle and build an execution report."""
    try:
        with open(slurm_job.result_path, "rb") as f:
            wrapper_result: WrapperResult = cloudpickle.load(f)  # noqa: S301
    except Exception:  # noqa: BLE001
        return ExecutionReport.from_task_and_exception(task, sys.exc_info())

    session.warnings.extend(wrapper_result.warning_reports)

    if wrapper_result.stdout:
        task.report_sections.append(("call", "stdout", wrapper_result.stdout))
    if wrapper_result.stderr:
        task.report_sections.append(("call", "stderr", wrapper_result.stderr))

    if wrapper_result.exc_info is not None:
        return ExecutionReport.from_task_and_exception(
            task, wrapper_result.exc_info  # type: ignore[arg-type]
        )

    _update_carry_over_products(task, wrapper_result.carry_over_products)

    try:
        session.hook.pytask_execute_task_teardown(session=session, task=task)
    except Exception:  # noqa: BLE001
        return ExecutionReport.from_task_and_exception(task, sys.exc_info())

    return ExecutionReport.from_task(task)


def _process_failed_job(
    session: Session,
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
    carry_over_products: Any,
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
