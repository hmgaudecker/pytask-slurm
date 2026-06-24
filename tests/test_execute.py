"""Unit tests for pytask_slurm.execute helpers."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cloudpickle
import pytest
from pytask import console

from pytask_slurm.execute import (
    _LOG_TAIL_MAX_CHARS,
    _cancel_remaining_jobs,
    _check_result_file_fallback,
    _is_actionable_status,
    _process_nonzero_exit,
    _read_result_report,
    _result_file_is_stable,
)
from pytask_slurm.monitor import SlurmJobStatus
from pytask_slurm.runner import run_task
from pytask_slurm.submit import SlurmJob


def _make_slurm_job(
    job_id: str = "12345",
    task_name: str = "task_example",
    submitted_at: float = 0.0,
    result_path: Path | None = None,
    log_path: Path | None = None,
) -> SlurmJob:
    return SlurmJob(
        job_id=job_id,
        task_name=task_name,
        payload_path=Path("/fake/payload.pkl"),
        result_path=result_path or Path("/fake/result.pkl"),
        log_path=log_path or Path("/fake/job.log"),
        submitted_at=submitted_at,
    )


class TestCancelRemainingJobs:
    """Cancel-on-exit is opt-out so long runs survive a controller death."""

    def test_cancels_running_jobs_by_default(self) -> None:
        session = MagicMock()
        session.config = {"slurm_cancel_on_exit": True}
        running = {"t": _make_slurm_job(job_id="999")}
        with patch("pytask_slurm.execute.cancel_jobs") as cancel:
            _cancel_remaining_jobs(session, running)
        cancel.assert_called_once_with(["999"])

    def test_leaves_jobs_running_when_opted_out(self) -> None:
        session = MagicMock()
        session.config = {"slurm_cancel_on_exit": False}
        running = {"t": _make_slurm_job(job_id="999")}
        with patch("pytask_slurm.execute.cancel_jobs") as cancel:
            _cancel_remaining_jobs(session, running)
        cancel.assert_not_called()


class TestIsActionableStatus:
    def test_pending_not_actionable(self) -> None:
        job = _make_slurm_job()
        assert _is_actionable_status(SlurmJobStatus.PENDING, job, 600) is False

    def test_running_not_actionable(self) -> None:
        job = _make_slurm_job()
        assert _is_actionable_status(SlurmJobStatus.RUNNING, job, 600) is False

    def test_completed_is_actionable(self) -> None:
        job = _make_slurm_job()
        assert _is_actionable_status(SlurmJobStatus.COMPLETED, job, 600) is True

    def test_failed_is_actionable(self) -> None:
        job = _make_slurm_job()
        assert _is_actionable_status(SlurmJobStatus.FAILED, job, 600) is True

    @patch("pytask_slurm.execute.time")
    def test_unknown_within_timeout_not_actionable(self, mock_time: MagicMock) -> None:
        mock_time.monotonic.return_value = 100.0
        job = _make_slurm_job(submitted_at=0.0)
        assert _is_actionable_status(SlurmJobStatus.UNKNOWN, job, 600) is False

    @patch("pytask_slurm.execute.time")
    def test_unknown_after_timeout_is_actionable(
        self, mock_time: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_time.monotonic.return_value = 700.0
        job = _make_slurm_job(submitted_at=0.0)
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.execute"):
            result = _is_actionable_status(SlurmJobStatus.UNKNOWN, job, 600)
        assert result is True
        assert "UNKNOWN state" in caplog.text

    @patch("pytask_slurm.execute.time")
    def test_unknown_with_custom_timeout(self, mock_time: MagicMock) -> None:
        mock_time.monotonic.return_value = 60.0
        job = _make_slurm_job(submitted_at=0.0)
        # Actionable with timeout=50 (elapsed 60 > 50) …
        assert _is_actionable_status(SlurmJobStatus.UNKNOWN, job, 50) is True
        # … but not with timeout=600 (elapsed 60 < 600).
        assert _is_actionable_status(SlurmJobStatus.UNKNOWN, job, 600) is False


class TestCheckResultFileFallback:
    def _make_session(self) -> MagicMock:
        session = MagicMock()
        session.dag.nodes.__getitem__.return_value = MagicMock()
        return session

    @patch("pytask_slurm.execute._process_completed_job")
    def test_picks_up_job_with_result_file(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        result_file = tmp_path / "result.pkl"
        result_file.touch()
        job = _make_slurm_job(job_id="111", task_name="task_a", result_path=result_file)
        running_jobs = {"task_a": job}

        mock_process.return_value = MagicMock()
        session = self._make_session()

        with patch("pytask_slurm.execute._result_file_is_stable", return_value=True):
            reports, names = _check_result_file_fallback(
                session,
                running_jobs,
                reported_job_ids=set(),
                already_done=set(),
            )
        assert len(reports) == 1
        assert names == ["task_a"]

    @patch("pytask_slurm.execute._process_completed_job")
    def test_skips_job_already_in_reported_ids(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        result_file = tmp_path / "result.pkl"
        result_file.touch()
        job = _make_slurm_job(job_id="111", task_name="task_a", result_path=result_file)
        running_jobs = {"task_a": job}

        session = self._make_session()
        reports, names = _check_result_file_fallback(
            session,
            running_jobs,
            reported_job_ids={"111"},
            already_done=set(),
        )
        assert len(reports) == 0
        assert names == []
        mock_process.assert_not_called()

    @patch("pytask_slurm.execute._process_completed_job")
    def test_skips_job_already_done(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        result_file = tmp_path / "result.pkl"
        result_file.touch()
        job = _make_slurm_job(job_id="111", task_name="task_a", result_path=result_file)
        running_jobs = {"task_a": job}

        session = self._make_session()
        reports, names = _check_result_file_fallback(
            session,
            running_jobs,
            reported_job_ids=set(),
            already_done={"task_a"},
        )
        assert len(reports) == 0
        assert names == []
        mock_process.assert_not_called()

    @patch("pytask_slurm.execute._process_completed_job")
    def test_skips_job_with_unstable_result_file(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        result_file = tmp_path / "result.pkl"
        result_file.touch()
        job = _make_slurm_job(job_id="111", task_name="task_a", result_path=result_file)
        running_jobs = {"task_a": job}

        session = self._make_session()
        with patch("pytask_slurm.execute._result_file_is_stable", return_value=False):
            reports, names = _check_result_file_fallback(
                session,
                running_jobs,
                reported_job_ids=set(),
                already_done=set(),
            )
        assert len(reports) == 0
        assert names == []
        mock_process.assert_not_called()

    @patch("pytask_slurm.execute._process_completed_job")
    def test_skips_job_without_result_file(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        result_file = tmp_path / "nonexistent.pkl"
        job = _make_slurm_job(job_id="111", task_name="task_a", result_path=result_file)
        running_jobs = {"task_a": job}

        session = self._make_session()
        reports, names = _check_result_file_fallback(
            session,
            running_jobs,
            reported_job_ids=set(),
            already_done=set(),
        )
        assert len(reports) == 0
        assert names == []
        mock_process.assert_not_called()


class TestResultFileIsStable:
    def test_old_file_is_stable(self, tmp_path: Path) -> None:
        p = tmp_path / "result.pkl"
        p.touch()
        os.utime(p, (time.time() - 10, time.time() - 10))
        assert _result_file_is_stable(p) is True

    def test_fresh_file_is_not_stable(self, tmp_path: Path) -> None:
        p = tmp_path / "result.pkl"
        p.touch()
        assert _result_file_is_stable(p) is False

    def test_missing_file_is_not_stable(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.pkl"
        assert _result_file_is_stable(p) is False


class TestProcessNonzeroExit:
    """Tests for _process_nonzero_exit (COMPLETED + nonzero exit code)."""

    def test_report_contains_exit_code(self, tmp_path: Path) -> None:
        log_path = tmp_path / "job.log"
        log_path.write_text("some output\nTraceback: error here")
        job = _make_slurm_job(log_path=log_path)
        task = MagicMock()
        task.name = "task_example"

        report = _process_nonzero_exit(task, job, exit_code=42)

        assert report.exc_info is not None
        exc = report.exc_info[1]
        assert "exited with code 42" in str(exc)
        assert "COMPLETED" in str(exc)

    def test_log_content_included(self, tmp_path: Path) -> None:
        log_path = tmp_path / "job.log"
        log_path.write_text("worker crashed here")
        job = _make_slurm_job(log_path=log_path)
        task = MagicMock()
        task.name = "task_example"

        report = _process_nonzero_exit(task, job, exit_code=1)

        exc = report.exc_info[1]
        assert "worker crashed here" in str(exc)
        assert f"last {_LOG_TAIL_MAX_CHARS} chars" in str(exc)

    def test_empty_log_message(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nonexistent.log"
        job = _make_slurm_job(log_path=log_path)
        task = MagicMock()
        task.name = "task_example"

        report = _process_nonzero_exit(task, job, exit_code=137)

        exc = report.exc_info[1]
        assert "exited with code 137" in str(exc)
        assert "empty or not yet available" in str(exc)


class TestReadResultReport:
    @patch("pytask_slurm.execute._process_completed_job")
    def test_reads_result_pickle_when_present_and_stable(
        self, mock_completed: MagicMock, tmp_path: Path
    ) -> None:
        """A failed job whose result pickle exists is read for the real traceback."""
        result_file = tmp_path / "result.pkl"
        result_file.write_bytes(b"data")
        old = time.time() - 100
        os.utime(result_file, (old, old))
        job = _make_slurm_job(result_path=result_file)

        result = _read_result_report(MagicMock(), MagicMock(), job)

        mock_completed.assert_called_once()
        assert result is mock_completed.return_value

    def test_returns_none_when_result_pickle_missing(self, tmp_path: Path) -> None:
        """A job killed before writing a result pickle returns None to fall back."""
        job = _make_slurm_job(result_path=tmp_path / "missing.pkl")

        assert _read_result_report(MagicMock(), MagicMock(), job) is None


def _failing_execute(**_kwargs: object) -> None:
    raise ValueError("kaboom")


class TestRunnerExitCode:
    def _run_failing_task(self, tmp_path: Path) -> Path:
        payload_path = tmp_path / "p_payload.pkl"
        result_path = tmp_path / "p_result.pkl"
        task = SimpleNamespace(name="task_x", execute=_failing_execute)
        payload = SimpleNamespace(
            task=task,
            kwargs={},
            console_options=console.options,
            session_filterwarnings=[],
            show_locals=False,
            task_filterwarnings=[],
        )
        with payload_path.open("wb") as f:
            cloudpickle.dump(payload, f)
        with pytest.raises(SystemExit) as excinfo:
            run_task(str(payload_path), str(result_path))
        assert excinfo.value.code == 1
        return result_path

    def test_exits_nonzero_when_task_raises(self, tmp_path: Path) -> None:
        """A raising task makes the runner exit 1 so SLURM reports the job FAILED."""
        self._run_failing_task(tmp_path)

    def test_writes_result_pickle_with_traceback_before_exit(
        self, tmp_path: Path
    ) -> None:
        """The result pickle with the traceback is written despite the non-zero exit."""
        result_path = self._run_failing_task(tmp_path)
        with result_path.open("rb") as f:
            wrapper_result = cloudpickle.load(f)
        assert wrapper_result.exc_info is not None
