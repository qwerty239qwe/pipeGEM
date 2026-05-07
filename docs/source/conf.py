# Configuration file for the Sphinx documentation builder.
#
# Full list of options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = 'pipeGEM'
copyright = '2025, Yu-Te Lin'
author = 'Yu-Te Lin'
release = '0.1.1'

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_design',
]

autosummary_generate = True

# Napoleon (NumPy docstring) settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = False

# Intersphinx: link to external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'cobra': ('https://cobrapy.readthedocs.io/en/latest', None),
}

templates_path = ['_templates']

# Exclude build artefacts and stray development files
exclude_patterns = [
    '_build',
    '**.ipynb_checkpoints',
    'tmp.ipynb',
    'task_result.json',
]

# ---------------------------------------------------------------------------
# Notebook execution (myst-nb 1.x)
# ---------------------------------------------------------------------------
# "auto": skip execution only if every code cell already has stored outputs;
# execute the notebook if any cell is missing outputs.
# Build with: uv run sphinx-build source build html   (from the docs/ dir)
# This ensures the correct virtual-environment kernel is used.
nb_execution_mode = 'auto'
nb_execution_timeout = 300      # seconds per notebook
nb_execution_allow_errors = False
nb_execution_raise_on_error = True

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    'navbar_start': ['navbar-logo'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'navbar_persistent': ['search-button'],
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/qwerty239qwe/pipeGEM',
            'icon': 'fa-brands fa-github',
        },
    ],
}
