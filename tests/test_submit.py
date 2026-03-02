"""Unit tests for pytask_slurm.submit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pytask import Mark

from pytask_slurm.submit import (
    _build_sbatch_cmd,
    _get_slurm_options,
    _warn_on_conflicting_extra,
)

_DEFAULT_CONFIG: dict[str, Any] = {
    "slurm_partition": "default",
    "slurm_time": "01:00:00",
    "slurm_mem": "4G",
    "slurm_cpus_per_task": 1,
    "slurm_account": "myaccount",
    "slurm_qos": None,
    "slurm_gpus": None,
}


class _FakeTask:
    """Minimal stub — only ``name`` is needed by ``_get_slurm_options``."""

    def __init__(self, name: str = "task_example") -> None:
        self.name = name


def _call(marks: list[Mark]) -> dict[str, Any]:
    # Patches get_marks at the module where it is imported; if the import path
    # in pytask_slurm.submit changes, this patch must be updated accordingly.
    with patch("pytask_slurm.submit.get_marks", return_value=marks):
        return _get_slurm_options(_FakeTask(), _DEFAULT_CONFIG)  # type: ignore[arg-type]


def _mark(**kwargs: Any) -> Mark:  # noqa: ANN401
    return Mark(name="slurm", args=(), kwargs=kwargs)


class TestGetSlurmOptionsValidation:
    def test_cpus_per_task_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be an int, got bool"):
            _call([_mark(cpus_per_task=True)])

    def test_cpus_per_task_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer, got 0"):
            _call([_mark(cpus_per_task=0)])

    def test_cpus_per_task_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer, got -1"):
            _call([_mark(cpus_per_task=-1)])

    def test_cpus_per_task_float_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be an int, got float"):
            _call([_mark(cpus_per_task=2.5)])

    def test_cpus_per_task_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be an int, got str"):
            _call([_mark(cpus_per_task="4")])

    def test_string_key_with_non_string_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a str, got int"):
            _call([_mark(mem=16)])

    def test_account_non_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a str, got int"):
            _call([_mark(account=123)])

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be an empty string"):
            _call([_mark(partition="")])

    def test_none_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be None"):
            _call([_mark(time=None)])

    def test_multiple_marks_rejected(self) -> None:
        marks = [_mark(mem="8G"), _mark(time="02:00:00")]
        with pytest.raises(ValueError, match=r"2 @pytask\.mark\.slurm decorators"):
            _call(marks)

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(ValueError, match="positional arguments"):
            _call([Mark(name="slurm", args=("gpu",), kwargs={})])

    def test_unknown_kwarg_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown @pytask\.mark\.slurm kwargs"):
            _call([_mark(typo="x")])

    def test_valid_override_merges(self) -> None:
        result = _call([_mark(mem="16G", cpus_per_task=4)])
        assert result["mem"] == "16G"
        assert result["cpus_per_task"] == 4  # noqa: PLR2004
        assert result["time"] == "01:00:00"
        assert result["partition"] == "default"
        assert result["account"] == "myaccount"


class TestGetSlurmOptionsConfigValidation:
    """Validation of global config values (not just mark kwargs)."""

    def test_config_cpus_per_task_bool_rejected(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_cpus_per_task": True}
        with (
            patch("pytask_slurm.submit.get_marks", return_value=[]),
            pytest.raises(ValueError, match=r"Global config.*must be an int, got bool"),
        ):
            _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]

    def test_config_mem_int_rejected(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_mem": 0}
        with (
            patch("pytask_slurm.submit.get_marks", return_value=[]),
            pytest.raises(ValueError, match=r"Global config.*must be a str, got int"),
        ):
            _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]

    def test_config_time_empty_string_rejected(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_time": ""}
        with (
            patch("pytask_slurm.submit.get_marks", return_value=[]),
            pytest.raises(
                ValueError,
                match=r"Global config.*must not be an empty string",
            ),
        ):
            _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]

    def test_config_none_partition_allowed(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_partition": None}
        with patch("pytask_slurm.submit.get_marks", return_value=[]):
            result = _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]
        assert result["partition"] is None

    def test_config_none_time_rejected(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_time": None}
        with (
            patch("pytask_slurm.submit.get_marks", return_value=[]),
            pytest.raises(ValueError, match="must not be None"),
        ):
            _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]

    def test_config_none_mem_rejected(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_mem": None}
        with (
            patch("pytask_slurm.submit.get_marks", return_value=[]),
            pytest.raises(ValueError, match="must not be None"),
        ):
            _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]

    def test_config_none_cpus_per_task_rejected(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_cpus_per_task": None}
        with (
            patch("pytask_slurm.submit.get_marks", return_value=[]),
            pytest.raises(ValueError, match="must not be None"),
        ):
            _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]

    def test_invalid_config_caught_even_when_mark_overrides(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_cpus_per_task": True}
        with (
            patch(
                "pytask_slurm.submit.get_marks",
                return_value=[_mark(cpus_per_task=4)],
            ),
            pytest.raises(
                ValueError,
                match=r"Global config.*must be an int, got bool",
            ),
        ):
            _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]


class TestQosOption:
    def test_qos_from_config(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_qos": "high"}
        with patch("pytask_slurm.submit.get_marks", return_value=[]):
            result = _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]
        assert result["qos"] == "high"

    def test_qos_none_allowed(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_qos": None}
        with patch("pytask_slurm.submit.get_marks", return_value=[]):
            result = _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]
        assert result["qos"] is None

    def test_qos_from_mark(self) -> None:
        result = _call([_mark(qos="low")])
        assert result["qos"] == "low"

    def test_qos_mark_overrides_config(self) -> None:
        config = {**_DEFAULT_CONFIG, "slurm_qos": "high"}
        with patch("pytask_slurm.submit.get_marks", return_value=[_mark(qos="low")]):
            result = _get_slurm_options(_FakeTask(), config)  # type: ignore[arg-type]
        assert result["qos"] == "low"

    def test_qos_non_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a str, got int"):
            _call([_mark(qos=123)])

    def test_qos_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be an empty string"):
            _call([_mark(qos="")])


class TestGpusOption:
    def test_gpus_accepted(self) -> None:
        result = _call([_mark(gpus=1)])
        assert result["gpus"] == 1

    def test_gpus_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer, got 0"):
            _call([_mark(gpus=0)])

    def test_gpus_none_omits_flag(self, tmp_path: Path) -> None:
        opts = {**_FULL_OPTS, "gpus": None}
        cmd = _sbatch_cmd(tmp_path, opts=opts)
        assert not any(f.startswith("--gpus=") for f in cmd)

    def test_gpus_present_in_sbatch(self, tmp_path: Path) -> None:
        opts = {**_FULL_OPTS, "gpus": 2}
        cmd = _sbatch_cmd(tmp_path, opts=opts)
        assert "--gpus=2" in cmd


_FULL_OPTS: dict[str, Any] = {
    "partition": "gpu",
    "time": "02:00:00",
    "mem": "8G",
    "cpus_per_task": 4,
    "account": "research",
    "qos": "high",
    "gpus": None,
}


def _sbatch_cmd(
    tmp_path: Path,
    opts: dict[str, Any] | None = None,
    extra: str | None = None,
) -> list[str]:
    config = {**_DEFAULT_CONFIG, "slurm_extra": extra}
    return _build_sbatch_cmd(
        opts or _FULL_OPTS,
        config,
        "abc123",
        (tmp_path / "log", tmp_path / "p", tmp_path / "r"),
    )


class TestBuildSbatchCmd:
    """Tests for _build_sbatch_cmd."""

    def test_all_options_present(self, tmp_path: Path) -> None:
        cmd = _sbatch_cmd(tmp_path)
        assert cmd[0] == "sbatch"
        assert "--parsable" in cmd
        assert "--job-name=pytask-abc123" in cmd
        assert "--time=02:00:00" in cmd
        assert "--mem=8G" in cmd
        assert "--cpus-per-task=4" in cmd
        assert "--partition=gpu" in cmd
        assert "--account=research" in cmd
        assert "--qos=high" in cmd
        assert any(flag.startswith("--wrap=") for flag in cmd)

    def test_none_optional_fields_omitted(self, tmp_path: Path) -> None:
        opts = {**_FULL_OPTS, "partition": None, "account": None, "qos": None}
        cmd = _sbatch_cmd(tmp_path, opts=opts)
        assert not any(f.startswith("--partition=") for f in cmd)
        assert not any(f.startswith("--account=") for f in cmd)
        assert not any(f.startswith("--qos=") for f in cmd)

    def test_slurm_extra_appended(self, tmp_path: Path) -> None:
        cmd = _sbatch_cmd(tmp_path, extra="--gres=gpu:1 --nodelist=node01")
        assert "--gres=gpu:1" in cmd
        assert "--nodelist=node01" in cmd

    def test_slurm_extra_non_string_rejected(self, tmp_path: Path) -> None:
        opts: dict[str, Any] = {
            "partition": "default",
            "time": "01:00:00",
            "mem": "4G",
            "cpus_per_task": 1,
            "account": "myaccount",
            "qos": None,
            "gpus": None,
        }
        config = {**_DEFAULT_CONFIG, "slurm_extra": 42}
        paths = (tmp_path / "log", tmp_path / "payload", tmp_path / "result")
        with pytest.raises(TypeError, match="slurm_extra must be a string, got int"):
            _build_sbatch_cmd(opts, config, "abc123", paths)


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

    def test_warns_on_partition_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.submit"):
            _warn_on_conflicting_extra(["--partition=gpu"])
        assert "--partition" in caplog.text

    def test_warns_on_short_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.submit"):
            _warn_on_conflicting_extra(["-p", "gpu"])
        assert "-p" in caplog.text
        assert "conflicts" in caplog.text

    def test_warns_on_short_time_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.submit"):
            _warn_on_conflicting_extra(["-t", "02:00:00"])
        assert "-t" in caplog.text

    def test_warns_on_combined_short_flag(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="pytask_slurm.submit"):
            _warn_on_conflicting_extra(["-pgpu"])
        assert "-pgpu" in caplog.text
        assert "conflicts" in caplog.text
