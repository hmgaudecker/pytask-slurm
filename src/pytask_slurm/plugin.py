"""Entry-point for the plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pytask import hookimpl

from pytask_slurm import build, config

if TYPE_CHECKING:
    from pluggy import PluginManager


@hookimpl
def pytask_add_hooks(pm: PluginManager) -> None:
    """Register plugins."""
    pm.register(build)
    pm.register(config)
