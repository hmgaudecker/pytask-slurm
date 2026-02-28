"""Integration test for pytask-slurm using mock SLURM commands."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest
from pytask import ExitCode, build

MOCK_SLURM_DIR = Path(__file__).parent / "mock_slurm"


@pytest.fixture
def mock_slurm_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put mock SLURM scripts on PATH and set up state directory."""
    state_dir = tmp_path / "slurm_state"
    state_dir.mkdir()
    monkeypatch.setenv("MOCK_SLURM_STATE", str(state_dir))

    # Prepend mock_slurm to PATH so sbatch/sacct/scancel resolve to our mocks.
    monkeypatch.setenv("PATH", f"{MOCK_SLURM_DIR}:{os.environ['PATH']}")

    return state_dir


@pytest.mark.usefixtures("mock_slurm_env")
def test_simple_task(tmp_path: Path) -> None:
    """A single task that writes a file should complete via mock SLURM."""
    source = textwrap.dedent("""\
        from pathlib import Path
        from typing import Annotated

        from pytask import Product


        def task_hello(
            output: Annotated[Path, Product] = Path("hello.txt"),
        ) -> None:
            output.write_text("hello from slurm")
    """)
    tmp_path.joinpath("task_example.py").write_text(source)

    session = build(
        paths=tmp_path,
        slurm=True,
        slurm_poll_interval=0.5,
    )

    assert session.exit_code == ExitCode.OK
    assert tmp_path.joinpath("hello.txt").exists()
    assert tmp_path.joinpath("hello.txt").read_text() == "hello from slurm"


@pytest.mark.usefixtures("mock_slurm_env")
def test_two_independent_tasks(tmp_path: Path) -> None:
    """Two independent tasks should both complete."""
    source = textwrap.dedent("""\
        from pathlib import Path
        from typing import Annotated

        from pytask import Product


        def task_one(
            output: Annotated[Path, Product] = Path("out_1.txt"),
        ) -> None:
            output.write_text("1")


        def task_two(
            output: Annotated[Path, Product] = Path("out_2.txt"),
        ) -> None:
            output.write_text("2")
    """)
    tmp_path.joinpath("task_example.py").write_text(source)

    session = build(
        paths=tmp_path,
        slurm=True,
        slurm_poll_interval=0.5,
    )

    assert session.exit_code == ExitCode.OK
    assert tmp_path.joinpath("out_1.txt").read_text() == "1"
    assert tmp_path.joinpath("out_2.txt").read_text() == "2"


@pytest.mark.usefixtures("mock_slurm_env")
def test_failing_task(tmp_path: Path) -> None:
    """A task that raises should be reported as failed."""
    source = textwrap.dedent("""\
        def task_fail() -> None:
            msg = "intentional failure"
            raise RuntimeError(msg)
    """)
    tmp_path.joinpath("task_example.py").write_text(source)

    session = build(
        paths=tmp_path,
        slurm=True,
        slurm_poll_interval=0.5,
    )

    assert session.exit_code == ExitCode.FAILED


@pytest.mark.usefixtures("mock_slurm_env")
def test_task_with_dependency(tmp_path: Path) -> None:
    """A task that depends on another task's output."""
    source = textwrap.dedent("""\
        from pathlib import Path
        from typing import Annotated

        from pytask import Product


        def task_produce(
            output: Annotated[Path, Product] = Path("intermediate.txt"),
        ) -> None:
            output.write_text("data")


        def task_consume(
            dep: Path = Path("intermediate.txt"),
            output: Annotated[Path, Product] = Path("final.txt"),
        ) -> None:
            output.write_text(dep.read_text() + " processed")
    """)
    tmp_path.joinpath("task_example.py").write_text(source)

    session = build(
        paths=tmp_path,
        slurm=True,
        slurm_poll_interval=0.5,
    )

    assert session.exit_code == ExitCode.OK
    assert tmp_path.joinpath("final.txt").read_text() == "data processed"


def test_per_task_mark_override(tmp_path: Path, mock_slurm_env: Path) -> None:
    """@pytask.mark.slurm(...) overrides should flow through to sbatch args."""
    source = textwrap.dedent("""\
        from pathlib import Path
        from typing import Annotated

        import pytask
        from pytask import Product


        @pytask.mark.slurm(mem="16G", time="02:00:00")
        def task_custom(
            output: Annotated[Path, Product] = Path("out.txt"),
        ) -> None:
            output.write_text("done")
    """)
    tmp_path.joinpath("task_example.py").write_text(source)

    session = build(
        paths=tmp_path,
        slurm=True,
        slurm_poll_interval=0.5,
    )

    assert session.exit_code == ExitCode.OK
    assert tmp_path.joinpath("out.txt").read_text() == "done"

    # Verify the sbatch args recorded by mock sbatch contain the overrides.
    jobs = json.loads((mock_slurm_env / "jobs.json").read_text())
    assert len(jobs) == 1
    (job,) = jobs.values()
    sbatch_args = job["sbatch_args"]
    sbatch_line = " ".join(sbatch_args)
    assert "--mem=16G" in sbatch_line
    assert "--time=02:00:00" in sbatch_line
    # Non-overridden defaults should still appear with their default values.
    assert "--cpus-per-task=1" in sbatch_line


def test_bare_slurm_mark_uses_defaults(tmp_path: Path, mock_slurm_env: Path) -> None:
    """@pytask.mark.slurm() with no kwargs should use global defaults."""
    source = textwrap.dedent("""\
        from pathlib import Path
        from typing import Annotated

        import pytask
        from pytask import Product


        @pytask.mark.slurm()
        def task_bare(
            output: Annotated[Path, Product] = Path("out.txt"),
        ) -> None:
            output.write_text("done")
    """)
    tmp_path.joinpath("task_example.py").write_text(source)

    session = build(
        paths=tmp_path,
        slurm=True,
        slurm_poll_interval=0.5,
    )

    assert session.exit_code == ExitCode.OK
    assert tmp_path.joinpath("out.txt").read_text() == "done"

    # Verify sbatch args contain the global defaults (not silently dropped).
    jobs = json.loads((mock_slurm_env / "jobs.json").read_text())
    assert len(jobs) == 1
    (job,) = jobs.values()
    sbatch_args = job["sbatch_args"]
    sbatch_line = " ".join(sbatch_args)
    assert "--time=01:00:00" in sbatch_line
    assert "--mem=4G" in sbatch_line
    assert "--cpus-per-task=1" in sbatch_line


@pytest.mark.usefixtures("mock_slurm_env")
def test_positional_mark_args_raises(tmp_path: Path) -> None:
    """Positional args in @pytask.mark.slurm should produce a clear error."""
    source = textwrap.dedent("""\
        from pathlib import Path
        from typing import Annotated

        import pytask
        from pytask import Product


        @pytask.mark.slurm("gpu")
        def task_pos(
            output: Annotated[Path, Product] = Path("out.txt"),
        ) -> None:
            output.write_text("done")
    """)
    tmp_path.joinpath("task_example.py").write_text(source)

    session = build(
        paths=tmp_path,
        slurm=True,
        slurm_poll_interval=0.5,
    )

    assert session.exit_code == ExitCode.FAILED

    failed = [r for r in session.execution_reports if r.exc_info and r.exc_info[1]]
    assert failed, "Expected at least one execution report with exception info"
    exc = failed[0].exc_info[1]  # type: ignore[index]
    assert isinstance(exc, ValueError)
    assert "positional arguments" in str(exc)
    assert "gpu" in str(exc)


@pytest.mark.usefixtures("mock_slurm_env")
def test_unknown_mark_kwarg_raises(tmp_path: Path) -> None:
    """Typos in @pytask.mark.slurm kwargs should produce a clear error."""
    source = textwrap.dedent("""\
        from pathlib import Path
        from typing import Annotated

        import pytask
        from pytask import Product


        @pytask.mark.slurm(partitoin="gpu")
        def task_typo(
            output: Annotated[Path, Product] = Path("out.txt"),
        ) -> None:
            output.write_text("done")
    """)
    tmp_path.joinpath("task_example.py").write_text(source)

    session = build(
        paths=tmp_path,
        slurm=True,
        slurm_poll_interval=0.5,
    )

    assert session.exit_code == ExitCode.FAILED

    # Verify the error is specifically a ValueError about the unknown kwarg.
    # Note: relies on pytask's execution_reports / exc_info internals.
    failed = [r for r in session.execution_reports if r.exc_info and r.exc_info[1]]
    assert failed, "Expected at least one execution report with exception info"
    exc = failed[0].exc_info[1]  # type: ignore[index]
    assert isinstance(exc, ValueError)
    assert "Unknown @pytask.mark.slurm kwargs" in str(exc)
    assert "partitoin" in str(exc)
