"""Unit tests for pytask_slurm.monitor."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from pytask_slurm.monitor import (
    SlurmJobResult,
    SlurmJobStatus,
    _parse_exit_code,
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
        fake_stdout = "1001|COMPLETED|0:0\n1002|RUNNING|0:0\n1003|FAILED|1:0\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_stdout
            mock_run.return_value.returncode = 0
            result = poll_job_statuses(["1001", "1002", "1003"])

        assert result == {
            "1001": SlurmJobResult(SlurmJobStatus.COMPLETED, exit_code=0),
            "1002": SlurmJobResult(SlurmJobStatus.RUNNING, exit_code=0),
            "1003": SlurmJobResult(SlurmJobStatus.FAILED, exit_code=1),
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
        fake_stdout = "1001|COMPLETED|0:0\ngarbage_no_pipe\n1002|RUNNING|0:0\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_stdout
            mock_run.return_value.returncode = 0
            result = poll_job_statuses(["1001", "1002"])

            mock_run.assert_called_once()
            assert "--jobs=1001,1002" in mock_run.call_args[0][0]

        assert result == {
            "1001": SlurmJobResult(SlurmJobStatus.COMPLETED, exit_code=0),
            "1002": SlurmJobResult(SlurmJobStatus.RUNNING, exit_code=0),
        }

    def test_handles_extra_trailing_delimiters(self) -> None:
        fake_stdout = "1001|COMPLETED|0:0|extra|fields\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = fake_stdout
            mock_run.return_value.returncode = 0
            result = poll_job_statuses(["1001"])

            mock_run.assert_called_once()
            assert "--jobs=1001" in mock_run.call_args[0][0]

        assert result == {"1001": SlurmJobResult(SlurmJobStatus.COMPLETED, exit_code=0)}

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
            assert "--format=JobIDRaw,State,ExitCode" in cmd
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


class TestParseExitCode:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0:0", 0),
            ("1:0", 1),
            ("2:0", 2),
            ("127:0", 127),
            ("0:9", 0),
            ("", None),
            ("bogus", None),
        ],
    )
    def test_parse_exit_code(self, raw: str, expected: int | None) -> None:
        assert _parse_exit_code(raw) == expected


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
            "1001": SlurmJobResult(SlurmJobStatus.RUNNING, exit_code=None),
            "1002": SlurmJobResult(SlurmJobStatus.PENDING, exit_code=None),
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

        assert result == {
            "500": SlurmJobResult(SlurmJobStatus.COMPLETED, exit_code=None)
        }

    def test_returns_empty_when_both_fail(self) -> None:
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.side_effect = [FileNotFoundError, FileNotFoundError]
            result = poll_job_statuses(["1001"])

        assert result == {}
        assert mock_run.call_count == 2  # noqa: PLR2004

    def test_sacct_success_skips_squeue(self) -> None:
        sacct_stdout = "1001|COMPLETED|0:0\n"
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.return_value.stdout = sacct_stdout
            mock_run.return_value.returncode = 0
            result = poll_job_statuses(["1001"])

        assert result == {"1001": SlurmJobResult(SlurmJobStatus.COMPLETED, exit_code=0)}
        mock_run.assert_called_once()

    def test_returns_empty_when_sacct_and_squeue_nonzero_exit(self) -> None:
        sacct_result = type(
            "Result", (), {"stdout": "", "returncode": 1, "stderr": "error"}
        )()
        squeue_result = type(
            "Result", (), {"stdout": "garbage", "returncode": 1, "stderr": "error"}
        )()
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.side_effect = [sacct_result, squeue_result]
            result = poll_job_statuses(["1001"])

        assert result == {}
        assert mock_run.call_count == 2  # noqa: PLR2004

    def test_falls_back_to_squeue_on_sacct_nonzero_exit(self) -> None:
        squeue_stdout = "1001|RUNNING\n"
        sacct_result = type(
            "Result", (), {"stdout": "", "returncode": 1, "stderr": "error"}
        )()
        squeue_result = type("Result", (), {"stdout": squeue_stdout, "returncode": 0})()
        with patch("pytask_slurm.monitor.subprocess.run") as mock_run:
            mock_run.side_effect = [sacct_result, squeue_result]
            result = poll_job_statuses(["1001"])

        assert result == {
            "1001": SlurmJobResult(SlurmJobStatus.RUNNING, exit_code=None)
        }
        assert mock_run.call_count == 2  # noqa: PLR2004
        assert mock_run.call_args_list[0][0][0][0] == "sacct"
        assert mock_run.call_args_list[1][0][0][0] == "squeue"
