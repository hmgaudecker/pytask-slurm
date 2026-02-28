"""Unit tests for pytask_slurm.execute helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pytask_slurm.execute import (
    _check_result_file_fallback,
    _is_actionable_status,
)
from pytask_slurm.monitor import SlurmJobStatus
from pytask_slurm.submit import SlurmJob, _warn_on_conflicting_extra


def _make_slurm_job(
    job_id: str = "12345",
    task_name: str = "task_example",
    submitted_at: float = 0.0,
    result_path: Path | None = None,
) -> SlurmJob:
    return SlurmJob(
        job_id=job_id,
        task_name=task_name,
        payload_path=Path("/fake/payload.pkl"),
        result_path=result_path or Path("/fake/result.pkl"),
        log_path=Path("/fake/job.log"),
        submitted_at=submitted_at,
    )


class TestIsActionableStatus:
    def test_pending_not_actionable(self) -> None:
        job = _make_slurm_job()
        assert _is_actionable_status(SlurmJobStatus.PENDING, job) is False

    def test_running_not_actionable(self) -> None:
        job = _make_slurm_job()
        assert _is_actionable_status(SlurmJobStatus.RUNNING, job) is False

    def test_completed_is_actionable(self) -> None:
        job = _make_slurm_job()
        assert _is_actionable_status(SlurmJobStatus.COMPLETED, job) is True

    def test_failed_is_actionable(self) -> None:
        job = _make_slurm_job()
        assert _is_actionable_status(SlurmJobStatus.FAILED, job) is True

    @patch("pytask_slurm.execute.time")
    def test_unknown_within_timeout_not_actionable(self, mock_time: MagicMock) -> None:
        mock_time.monotonic.return_value = 100.0
        job = _make_slurm_job(submitted_at=0.0)
        assert _is_actionable_status(SlurmJobStatus.UNKNOWN, job) is False

    @patch("pytask_slurm.execute.time")
    def test_unknown_after_timeout_is_actionable(
        self, mock_time: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_time.monotonic.return_value = 700.0
        job = _make_slurm_job(submitted_at=0.0)
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.execute"):
            result = _is_actionable_status(SlurmJobStatus.UNKNOWN, job)
        assert result is True
        assert "UNKNOWN state" in caplog.text


class TestCheckResultFileFallback:
    def _make_session(self) -> MagicMock:
        session = MagicMock()
        session.dag.nodes.__getitem__.return_value = {
            "task": MagicMock(),
        }
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


class TestWarnOnConflictingExtra:
    def test_warns_on_generated_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.submit"):
            _warn_on_conflicting_extra(["--wrap=something"])
        assert "--wrap" in caplog.text
        assert "conflicts" in caplog.text

    def test_no_warning_for_unrelated_flag(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.submit"):
            _warn_on_conflicting_extra(["--gres=gpu:1", "--nodelist=node01"])
        assert caplog.text == ""

    def test_warns_on_flag_with_equals(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.submit"):
            _warn_on_conflicting_extra(["--mem=16G"])
        assert "--mem" in caplog.text
