"""Unit tests for pytask_slurm.submit._get_slurm_options."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pytask import Mark

from pytask_slurm.submit import _get_slurm_options

_DEFAULT_CONFIG = {
    "slurm_partition": "default",
    "slurm_time": "01:00:00",
    "slurm_mem": "4G",
    "slurm_cpus_per_task": 1,
    "slurm_account": "myaccount",
}


class _FakeTask:
    def __init__(self, name: str = "task_example") -> None:
        self.name = name


def _call(marks: list[Mark]) -> dict:
    # Patches get_marks at the module where it is imported; if the import path
    # in pytask_slurm.submit changes, this patch must be updated accordingly.
    with patch("pytask_slurm.submit.get_marks", return_value=marks):
        return _get_slurm_options(_FakeTask(), _DEFAULT_CONFIG)


def _mark(**kwargs) -> Mark:
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
        with pytest.raises(ValueError, match="2 @pytask.mark.slurm decorators"):
            _call(marks)

    def test_valid_override_merges(self) -> None:
        result = _call([_mark(mem="16G", cpus_per_task=4)])
        assert result["mem"] == "16G"
        assert result["cpus_per_task"] == 4
        assert result["time"] == "01:00:00"
        assert result["partition"] == "default"
        assert result["account"] == "myaccount"
