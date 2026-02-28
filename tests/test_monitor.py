"""Unit tests for pytask_slurm.monitor."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from pytask_slurm.monitor import (
    SlurmJobStatus,
    _parse_state,
    _sacct_cmd,
    _squeue_cmd,
    poll_job_statuses,
)


class TestParseState:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("COMPLETED", SlurmJobStatus.COMPLETED),
            ("RUNNING", SlurmJobStatus.RUNNING),
            ("PENDING", SlurmJobStatus.PENDING),
            ("FAILED", SlurmJobStatus.FAILED),
            ("TIMEOUT", SlurmJobStatus.TIMEOUT),
            ("OUT_OF_MEMORY", SlurmJobStatus.OUT_OF_MEMORY),
            ("NODE_FAIL", SlurmJobStatus.NODE_FAIL),
            ("CANCELLED", SlurmJobStatus.CANCELLED),
            ("COMPLETING", SlurmJobStatus.RUNNING),
            ("CONFIGURING", SlurmJobStatus.PENDING),
            ("REQUEUED", SlurmJobStatus.PENDING),
            ("SUSPENDED", SlurmJobStatus.PENDING),
            ("PREEMPTED", SlurmJobStatus.FAILED),
        ],
    )
    def test_known_states(self, raw: str, expected: SlurmJobStatus) -> None:
        assert _parse_state(raw) == expected

    def test_cancelled_with_suffix(self) -> None:
        assert _parse_state("CANCELLED by 12345") == SlurmJobStatus.CANCELLED

    def test_unknown_state(self) -> None:
        assert _parse_state("SOMETHING_NEW") == SlurmJobStatus.UNKNOWN

    def test_empty_string(self) -> None:
        assert _parse_state("") == SlurmJobStatus.UNKNOWN


class TestPollJobStatuses:
    def test_empty_list(self) -> None:
        assert poll_job_statuses([]) == {}

    def test_parses_sacct_output(self) -> None:
        fake_stdout = "1001|COMPLETED\n1002|RUNNING\n1003|FAILED\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_stdout
            mock_run.return_value.returncode = 0
            result = poll_job_statuses(["1001", "1002", "1003"])

        assert result == {
            "1001": SlurmJobStatus.COMPLETED,
            "1002": SlurmJobStatus.RUNNING,
            "1003": SlurmJobStatus.FAILED,
        }

    def test_handles_timeout(self) -> None:
        with patch(
            "pytask_slurm.monitor.subprocess.run",
            side_effect=subprocess.TimeoutExpired("sacct", 30),
        ):
            assert poll_job_statuses(["1001"]) == {}

    def test_handles_missing_sacct(self) -> None:
        with patch(
            "pytask_slurm.monitor.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert poll_job_statuses(["1001"]) == {}

    def test_skips_lines_without_delimiter(self) -> None:
        fake_stdout = "1001|COMPLETED\ngarbage_no_pipe\n1002|RUNNING\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_stdout
            mock_run.return_value.returncode = 0
            result = poll_job_statuses(["1001", "1002"])

            mock_run.assert_called_once()
            assert "--jobs=1001,1002" in mock_run.call_args[0][0]

        assert result == {
            "1001": SlurmJobStatus.COMPLETED,
            "1002": SlurmJobStatus.RUNNING,
        }

    def test_handles_extra_trailing_delimiters(self) -> None:
        fake_stdout = "1001|COMPLETED|extra|fields\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_stdout
            mock_run.return_value.returncode = 0
            result = poll_job_statuses(["1001"])

            mock_run.assert_called_once()
            assert "--jobs=1001" in mock_run.call_args[0][0]

        assert result == {"1001": SlurmJobStatus.COMPLETED}

    def test_constructs_correct_command(self) -> None:
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 0
            poll_job_statuses(["100", "200"])

            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "sacct"
            assert "-X" in cmd
            assert "--parsable2" in cmd
            assert "--noheader" in cmd
            assert "--jobs=100,200" in cmd


class TestCommandBuilders:
    def test_sacct_cmd(self) -> None:
        cmd = _sacct_cmd(["1", "2"])
        assert cmd[0] == "sacct"
        assert "--jobs=1,2" in cmd

    def test_squeue_cmd(self) -> None:
        cmd = _squeue_cmd(["1", "2"])
        assert cmd[0] == "squeue"
        assert "--jobs=1,2" in cmd
        assert "--noheader" in cmd
        assert "--format=%i|%T" in cmd


class TestSqueueFallback:
    def test_falls_back_to_squeue_on_sacct_failure(self) -> None:
        squeue_stdout = "1001|RUNNING\n1002|PENDING\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.side_effect = [
                FileNotFoundError,
                type("Result", (), {"stdout": squeue_stdout, "returncode": 0})(),
            ]
            result = poll_job_statuses(["1001", "1002"])

        assert result == {
            "1001": SlurmJobStatus.RUNNING,
            "1002": SlurmJobStatus.PENDING,
        }
        assert mock_run.call_count == 2  # noqa: PLR2004
        # First call was sacct, second was squeue.
        assert mock_run.call_args_list[0][0][0][0] == "sacct"
        assert mock_run.call_args_list[1][0][0][0] == "squeue"

    def test_falls_back_to_squeue_on_sacct_timeout(self) -> None:
        squeue_stdout = "500|COMPLETED\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.TimeoutExpired("sacct", 30),
                type("Result", (), {"stdout": squeue_stdout, "returncode": 0})(),
            ]
            result = poll_job_statuses(["500"])

        assert result == {"500": SlurmJobStatus.COMPLETED}

    def test_returns_empty_when_both_fail(self) -> None:
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.side_effect = [FileNotFoundError, FileNotFoundError]
            result = poll_job_statuses(["1001"])

        assert result == {}
        assert mock_run.call_count == 2  # noqa: PLR2004

    def test_sacct_success_skips_squeue(self) -> None:
        sacct_stdout = "1001|COMPLETED\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = sacct_stdout
            mock_run.return_value.returncode = 0
            result = poll_job_statuses(["1001"])

        assert result == {"1001": SlurmJobStatus.COMPLETED}
        mock_run.assert_called_once()
