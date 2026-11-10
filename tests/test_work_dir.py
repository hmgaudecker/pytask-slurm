"""The SLURM work directory comes from `slurm_work_dir`, else `<root>/.pytask/slurm`."""

from __future__ import annotations

from pathlib import Path

import pytest

from pytask_slurm.execute import get_work_dir


def test_get_work_dir_defaults_to_pytask_dir_in_root(tmp_path: Path) -> None:
    config = {"root": tmp_path, "slurm_work_dir": None}
    assert get_work_dir(config) == tmp_path / ".pytask" / "slurm"


def test_get_work_dir_expands_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config = {"root": tmp_path, "slurm_work_dir": "$DATA_ROOT/project/slurm"}
    assert get_work_dir(config) == tmp_path / "data" / "project" / "slurm"


def test_get_work_dir_resolves_relative_path_against_root(tmp_path: Path) -> None:
    config = {"root": tmp_path, "slurm_work_dir": "jobs"}
    assert get_work_dir(config) == tmp_path / "jobs"


def test_get_work_dir_fails_on_unset_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UNSET_DATA_ROOT", raising=False)
    config = {"root": tmp_path, "slurm_work_dir": "$UNSET_DATA_ROOT/slurm"}
    with pytest.raises(ValueError, match="UNSET_DATA_ROOT"):
        get_work_dir(config)
