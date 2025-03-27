# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Your Project Name'
author = 'Your Name'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',  # Automatically document modules from docstrings
    'sphinx.ext.napoleon',  # (Optional) Supports Google and NumPy style docstrings
]

html_theme = "sphinx_rtd_theme"
