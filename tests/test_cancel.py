"""Unit tests for pytask_slurm.cancel."""

from __future__ import annotations

from unittest.mock import patch

from pytask_slurm.cancel import cancel_jobs


class TestCancelJobs:
    def test_empty_list_does_nothing(self) -> None:
        with patch("pytask_slurm.cancel.subprocess.run") as mock_run:
            cancel_jobs([])
        mock_run.assert_not_called()

    def test_calls_scancel_with_job_ids(self) -> None:
        with patch("pytask_slurm.cancel.subprocess.run") as mock_run:
            cancel_jobs(["100", "200", "300"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["scancel", "100", "200", "300"]

    def test_never_raises_on_error(self) -> None:
        with patch(
            "pytask_slurm.cancel.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            cancel_jobs(["100"])  # Should not raise.

    def test_never_raises_on_timeout(self) -> None:
        import subprocess

        with patch(
            "pytask_slurm.cancel.subprocess.run",
            side_effect=subprocess.TimeoutExpired("scancel", 30),
        ):
            cancel_jobs(["100"])  # Should not raise.
