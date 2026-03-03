"""Unit tests for pytask_slurm.logging."""

from __future__ import annotations

from io import StringIO
from typing import Any
from unittest.mock import patch

from pytask_slurm.logging import pytask_log_session_header


def _make_session(config: dict[str, Any]) -> Any:  # noqa: ANN401
    """Create a minimal session stub with the given config."""
    return type("Session", (), {"config": config})()


# Must contain every key that pytask_log_session_header reads from session.config.
_BASE_CONFIG: dict[str, Any] = {
    "slurm_partition": "gpu",
    "slurm_time": "02:00:00",
    "slurm_mem": "8G",
    "slurm_cpus_per_task": 4,
    "slurm_gpus": None,
    "slurm_account": "research",
    "slurm_qos": "high",
    "slurm_extra": None,
}


class TestLogSessionHeader:
    def test_all_fields_present(self) -> None:
        session = _make_session(_BASE_CONFIG)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "SLURM:" in output
        assert "partition=gpu" in output
        assert "time=02:00:00" in output
        assert "mem=8G" in output
        assert "cpus_per_task=4" in output
        assert "account=research" in output
        assert "qos=high" in output

    def test_none_partition_omitted(self) -> None:
        config = {**_BASE_CONFIG, "slurm_partition": None}
        session = _make_session(config)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "partition=" not in output

    def test_none_account_omitted(self) -> None:
        config = {**_BASE_CONFIG, "slurm_account": None}
        session = _make_session(config)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "account=" not in output

    def test_none_qos_omitted(self) -> None:
        config = {**_BASE_CONFIG, "slurm_qos": None}
        session = _make_session(config)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "qos=" not in output

    def test_both_optional_fields_none(self) -> None:
        config = {**_BASE_CONFIG, "slurm_account": None, "slurm_qos": None}
        session = _make_session(config)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "account=" not in output
        assert "qos=" not in output
        assert "SLURM:" in output

    def test_extra_shown_when_set(self) -> None:
        config = {**_BASE_CONFIG, "slurm_extra": "--gres=gpu:1 --constraint=a100"}
        session = _make_session(config)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "extra='--gres=gpu:1 --constraint=a100'" in output

    def test_extra_omitted_when_none(self) -> None:
        session = _make_session(_BASE_CONFIG)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "extra=" not in output

    def test_gpus_shown_when_set(self) -> None:
        config = {**_BASE_CONFIG, "slurm_gpus": 2}
        session = _make_session(config)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "gpus=2" in output

    def test_gpus_omitted_when_none(self) -> None:
        session = _make_session(_BASE_CONFIG)
        buf = StringIO()
        with patch("pytask_slurm.logging.console") as mock_console:
            mock_console.print = lambda text: buf.write(text)
            pytask_log_session_header(session)
        output = buf.getvalue()
        assert "gpus=" not in output
