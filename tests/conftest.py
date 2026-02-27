"""Test configuration for pytask-slurm."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


class _SysPathsSnapshot:
    def __init__(self) -> None:
        self.__saved = sys.path.copy(), sys.meta_path.copy()

    def restore(self) -> None:
        sys.path[:], sys.meta_path[:] = self.__saved


class _SysModulesSnapshot:
    def __init__(self) -> None:
        self.__saved = sys.modules.copy()

    def restore(self) -> None:
        sys.modules.clear()
        sys.modules.update(self.__saved)


@contextmanager
def restore_sys_path_and_module_after_test_execution() -> Generator[None]:
    sys_path_snapshot = _SysPathsSnapshot()
    sys_modules_snapshot = _SysModulesSnapshot()
    yield
    sys_modules_snapshot.restore()
    sys_path_snapshot.restore()


@pytest.fixture(autouse=True)
def _restore_sys_path_and_module_after_test_execution() -> Generator[None]:
    """Restore sys.path and sys.modules after every test.

    Without this, task modules named ``task_example`` from one test leak into the
    next via ``sys.modules``, causing stale-module errors.
    """
    with restore_sys_path_and_module_after_test_execution():
        yield
