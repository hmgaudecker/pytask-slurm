"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full list see the
documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html

"""

from __future__ import annotations

from importlib.metadata import version

# -- Project information ---------------------------------------------------------------

project = "pytask_slurm"
author = "Hans-Martin von Gaudecker"
copyright = f"2025, {author}"  # noqa: A001

# The version, including alpha/beta/rc tags, but not commit hash and datestamps
release = version("pytask_slurm")
# The short X.Y version.
version = ".".join(release.split(".")[:2])  # ty: ignore[invalid-assignment]

# -- General configuration -------------------------------------------------------------

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"
pygments_dark_style = "monokai"

# -- Extensions configuration ----------------------------------------------------------

add_module_names = True

copybutton_prompt_text = r"\$ |>>> |In \[\d\]: "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "pytask": ("https://pytask-dev.readthedocs.io/en/stable/", None),
    "pytask_parallel": (
        "https://pytask-parallel.readthedocs.io/en/stable/",
        None,
    ),
    "python": ("https://docs.python.org/3.12", None),
}

myst_enable_extensions = ["deflist"]
myst_footnote_transition = False

# -- Options for HTML output -----------------------------------------------------------

html_theme = "furo"

html_favicon = "_static/images/pytask.ico"

html_static_path = ["_static"]

html_domain_indices = True
html_use_index = True
html_split_index = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_logo": "images/pytask_w_text_light.svg",
    "dark_logo": "images/pytask_w_text_dark.svg",
}


def setup(app):  # noqa: ANN001, ANN201
    """Configure sphinx."""
    app.add_object_type(
        "confval",
        "confval",
        objname="configuration value",
        indextemplate="pair: %s; configuration value",
    )
