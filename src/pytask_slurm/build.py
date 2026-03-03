"""Extend the build command with the --slurm flag."""

from __future__ import annotations

import click
from pytask import hookimpl


@hookimpl
def pytask_extend_command_line_interface(cli: click.Group) -> None:
    """Extend the command line interface."""
    additional_parameters = [
        click.Option(
            ["--slurm"],
            is_flag=True,
            default=False,
            help="Submit tasks as SLURM jobs via sbatch.",
        ),
    ]
    cli.commands["build"].params.extend(additional_parameters)
