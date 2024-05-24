# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys

#Creating a path to code folder in order to update documentation
code_path = pathlib.Path(__file__).resolve().parents[2] / 'brainage'
sys.path.insert(0, code_path.as_posix())


project = 'BrainAge'
copyright = '2024, Valerio Caporioni, Jacopo Resasco'
author = 'Valerio Caporioni, Jacopo Resasco'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
