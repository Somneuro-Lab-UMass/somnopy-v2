# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'somnopy'
copyright = '2025, Roger Balcells Sanchez, Thea Ng, Atif Abedeen, Lindsey Mooney'
author = 'Roger Balcells Sanchez, Thea Ng, Atif Abedeen, Lindsey Mooney'
release = '0.0.8'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # Enable autodoc to extract documentation from docstrings
    'sphinx.ext.napoleon',  # Optional: support for Google and NumPy style docstrings
    'sphinx.ext.autosummary',  # Generate summary tables for modules/classes
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.intersphinx',  # Link to external documentation (e.g., Python docs)
]
templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = []
# '_static']
